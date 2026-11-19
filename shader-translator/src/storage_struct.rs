// SPDX-License-Identifier: GPL-2.0-or-later

use spirv::{Op, StorageClass};
use std::collections::{HashMap, HashSet};

const HEADER: usize = 5;
const STORAGE: u32 = StorageClass::StorageBuffer as u32;

/// A SPIR-V struct loaded from storage can contain members that are also used
/// by atomic instructions. WGSL cannot load such a struct as a value. Load its
/// individual members so Naga can translate the atomic reads as atomicLoad.
pub fn lower(input: &[u8]) -> Result<Option<Vec<u8>>, &'static str> {
    if input.len() < HEADER * 4 || !input.len().is_multiple_of(4) {
        return Ok(None);
    }
    let word = |i: usize| u32::from_le_bytes(input[i * 4..i * 4 + 4].try_into().unwrap());
    if word(0) != spirv::MAGIC_NUMBER {
        return Ok(None);
    }
    let mut cursor = HEADER;
    while cursor < input.len() / 4 {
        let header = word(cursor);
        let count = (header >> 16) as usize;
        if count == 0 || count > input.len() / 4 - cursor {
            return Err("invalid SPIR-V instruction size");
        }
        if (Op::AtomicLoad as u32..=Op::AtomicXor as u32).contains(&(header & 0xffff)) {
            break;
        }
        cursor += count;
    }
    if cursor == input.len() / 4 {
        return Ok(None);
    }

    let words: Vec<_> = (0..input.len() / 4).map(word).collect();
    let mut module = SplitLoads {
        next_id: words[3],
        ..Default::default()
    };
    let mut instructions = Vec::new();
    let mut storage_pointers = HashSet::new();
    let mut storage_pointer_types = HashSet::new();
    let mut offset = HEADER;
    while offset < words.len() {
        let header = words[offset];
        let count = (header >> 16) as usize;
        if count == 0 || count > words.len() - offset {
            return Err("invalid SPIR-V instruction size");
        }
        let args = &words[offset + 1..offset + count];
        let op = Op::from_u32(header & 0xffff);
        match (op, args) {
            (Some(Op::TypeStruct), [id, members @ ..]) => {
                module.members.insert(*id, members);
            }
            (Some(Op::TypeInt), [id, 32, 0]) => module.uint = Some(*id),
            (Some(Op::TypePointer), [id, STORAGE, target]) => {
                module.pointers.insert(*target, *id);
                storage_pointer_types.insert(*id);
            }
            (Some(Op::Variable | Op::AccessChain | Op::InBoundsAccessChain), [ty, id, ..])
                if storage_pointer_types.contains(ty) =>
            {
                storage_pointers.insert(*id);
            }
            _ => {}
        }
        instructions.push((header, op, args));
        offset += count;
    }
    if !instructions.iter().any(|&(_, op, args)| {
        matches!((op, args), (Some(Op::Load), [ty, _, pointer])
            if module.members.contains_key(ty) && storage_pointers.contains(pointer))
    }) {
        return Ok(None);
    }

    let mut output = words[..HEADER].to_vec();
    let mut body = Vec::new();
    let mut in_functions = false;
    for (header, op, args) in instructions {
        in_functions |= op == Some(Op::Function);
        if let (Some(Op::Load), [ty, result, pointer]) = (op, args) {
            if module.members.contains_key(ty) && storage_pointers.contains(pointer) {
                module.load(&mut body, *ty, *result, *pointer);
                continue;
            }
        }
        let section = if in_functions { &mut body } else { &mut output };
        section.push(header);
        section.extend_from_slice(args);
    }
    output[3] = module.next_id;
    output.extend(module.declarations);
    output.extend(body);
    Ok(Some(
        output.into_iter().flat_map(u32::to_le_bytes).collect(),
    ))
}

#[derive(Default)]
struct SplitLoads<'a> {
    next_id: u32,
    uint: Option<u32>,
    members: HashMap<u32, &'a [u32]>,
    pointers: HashMap<u32, u32>,
    indices: HashMap<u32, u32>,
    declarations: Vec<u32>,
}

fn take_id(next: &mut u32) -> u32 {
    let id = *next;
    *next += 1;
    id
}

fn emit(output: &mut Vec<u32>, op: Op, operands: &[u32]) {
    output.push(((operands.len() + 1) as u32) << 16 | op as u32);
    output.extend_from_slice(operands);
}

impl SplitLoads<'_> {
    fn load(&mut self, output: &mut Vec<u32>, ty: u32, result: u32, pointer: u32) {
        let Some(&members) = self.members.get(&ty) else {
            emit(output, Op::Load, &[ty, result, pointer]);
            return;
        };
        let uint = *self.uint.get_or_insert_with(|| {
            let id = take_id(&mut self.next_id);
            emit(&mut self.declarations, Op::TypeInt, &[id, 32, 0]);
            id
        });
        let mut components = vec![ty, result];
        for (index, &member) in members.iter().enumerate() {
            let index = *self.indices.entry(index as u32).or_insert_with(|| {
                let id = take_id(&mut self.next_id);
                emit(
                    &mut self.declarations,
                    Op::Constant,
                    &[uint, id, index as u32],
                );
                id
            });
            let pointer_type = *self.pointers.entry(member).or_insert_with(|| {
                let id = take_id(&mut self.next_id);
                emit(
                    &mut self.declarations,
                    Op::TypePointer,
                    &[id, STORAGE, member],
                );
                id
            });
            let member_pointer = take_id(&mut self.next_id);
            let value = take_id(&mut self.next_id);
            emit(
                output,
                Op::AccessChain,
                &[pointer_type, member_pointer, pointer, index],
            );
            self.load(output, member, value, member_pointer);
            components.push(value);
        }
        emit(output, Op::CompositeConstruct, &components);
    }
}
