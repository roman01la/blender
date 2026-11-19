// SPDX-License-Identifier: GPL-2.0-or-later

use spirv::{Decoration, Op, StorageClass};
use std::collections::HashMap;

const HEADER_WORDS: usize = 5;

fn emit(output: &mut Vec<u32>, op: Op, operands: &[u32]) {
    output.push(((operands.len() + 1) as u32) << 16 | op as u32);
    output.extend_from_slice(operands);
}

fn take_id(next: &mut u32) -> u32 {
    let id = *next;
    *next += 1;
    id
}

/// Lower glslang's named GLSL samplers to separate WebGPU resources. A sampler
/// at set 2 binding N becomes a texture at N and a sampler at N + 1. Split
/// function parameters in the same way so GLSL helpers can forward both handles.
pub fn lower(input: &[u8]) -> Result<Option<Vec<u8>>, &'static str> {
    if input.len() < HEADER_WORDS * 4 || !input.len().is_multiple_of(4) {
        return Ok(None);
    }
    let words: Vec<_> = input
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    if words[0] != spirv::MAGIC_NUMBER {
        return Ok(None);
    }

    let mut instructions = Vec::new();
    let mut sampled_types = HashMap::new();
    let mut pointer_types = HashMap::new();
    let mut pointers = HashMap::new();
    let mut next_id = words[3];
    let mut offset = HEADER_WORDS;
    while offset < words.len() {
        let header = words[offset];
        let length = (header >> 16) as usize;
        if length == 0 || length > words.len() - offset {
            return Err("invalid SPIR-V instruction size");
        }
        let op = Op::from_u32(header & 0xffff);
        let operands = &words[offset + 1..offset + length];
        match (op, operands) {
            (Some(Op::TypeSampledImage), [id, image]) => {
                sampled_types.insert(*id, *image);
            }
            (Some(Op::TypePointer), [id, storage, ty])
                if *storage == StorageClass::UniformConstant as u32
                    && sampled_types.contains_key(ty) =>
            {
                // Keep the original pointer ID for the texture pointer.
                pointer_types.insert(*id, take_id(&mut next_id));
            }
            (Some(Op::Variable | Op::FunctionParameter), [ty, id, ..])
                if pointer_types.contains_key(ty) =>
            {
                pointers.insert(*id, take_id(&mut next_id));
            }
            _ => {}
        }
        instructions.push((header, op, operands));
        offset += length;
    }
    if pointers.is_empty() {
        return Ok(None);
    }

    let mut output = words[..HEADER_WORDS].to_vec();
    let mut sampler_type = None;
    let mut expanded = Vec::new();
    for (header, op, operands) in instructions {
        match (op, operands) {
            (Some(Op::Decorate), [id, decoration, value]) if pointers.contains_key(id) => {
                let sampler_value = match Decoration::from_u32(*decoration) {
                    Some(Decoration::DescriptorSet) if *value == 2 => *value,
                    Some(Decoration::Binding) if *value < 128 && *value % 2 == 0 => value + 1,
                    _ => return Err("unsupported combined sampler binding"),
                };
                emit(
                    &mut output,
                    Op::Decorate,
                    &[pointers[id], *decoration, sampler_value],
                );
            }
            (Some(Op::TypePointer), [id, storage, ty]) if pointer_types.contains_key(id) => {
                let sampler = *sampler_type.get_or_insert_with(|| {
                    let id = take_id(&mut next_id);
                    emit(&mut output, Op::TypeSampler, &[id]);
                    id
                });
                emit(
                    &mut output,
                    Op::TypePointer,
                    &[*id, *storage, sampled_types[ty]],
                );
                emit(
                    &mut output,
                    Op::TypePointer,
                    &[pointer_types[id], *storage, sampler],
                );
                continue;
            }
            (Some(Op::Variable), [ty, id, storage]) if pointers.contains_key(id) => {
                emit(
                    &mut output,
                    Op::Variable,
                    &[pointer_types[ty], pointers[id], *storage],
                );
            }
            (Some(Op::FunctionParameter), [ty, id]) if pointers.contains_key(id) => {
                emit(&mut output, Op::FunctionParameter, &[*ty, *id]);
                emit(
                    &mut output,
                    Op::FunctionParameter,
                    &[pointer_types[ty], pointers[id]],
                );
                continue;
            }
            (Some(op @ (Op::TypeFunction | Op::FunctionCall)), [first, second, rest @ ..]) => {
                let (tail, replacements) = if op == Op::TypeFunction {
                    (rest, &pointer_types)
                } else {
                    (&rest[1..], &pointers)
                };
                let prefix = operands.len() - tail.len();
                expanded.clear();
                expanded.extend_from_slice(&operands[..prefix]);
                for id in tail {
                    expanded.push(*id);
                    expanded.extend(replacements.get(id).copied());
                }
                debug_assert_eq!(&expanded[..2], &[*first, *second]);
                emit(&mut output, op, &expanded);
                continue;
            }
            (Some(Op::Load), [ty, id, pointer]) if pointers.contains_key(pointer) => {
                let image = take_id(&mut next_id);
                let sampler = take_id(&mut next_id);
                emit(&mut output, Op::Load, &[sampled_types[ty], image, *pointer]);
                emit(
                    &mut output,
                    Op::Load,
                    &[sampler_type.unwrap(), sampler, pointers[pointer]],
                );
                emit(&mut output, Op::SampledImage, &[*ty, *id, image, sampler]);
                continue;
            }
            _ => {}
        }
        output.push(header);
        output.extend_from_slice(operands);
    }
    output[3] = next_id;
    Ok(Some(
        output.into_iter().flat_map(u32::to_le_bytes).collect(),
    ))
}
