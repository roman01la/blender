// SPDX-License-Identifier: GPL-2.0-or-later

use spirv::{GlslStd450Op as GLOp, Op};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy)]
struct IntType {
    signed: bool,
    lanes: u32,
}

/// GLSL's bitCount and find* builtins can return signed integers from unsigned
/// inputs. WGSL returns the input type, which is what Naga infers after import.
/// Keep the input signedness for the builtin, then reinterpret its result.
pub fn lower(input: &[u8]) -> Result<Option<Vec<u8>>, &'static str> {
    const HEADER: usize = 5;
    let word_count = input.len() / 4;
    if word_count < HEADER || !input.len().is_multiple_of(4) {
        return Ok(None);
    }
    let word = |i: usize| u32::from_le_bytes(input[i * 4..i * 4 + 4].try_into().unwrap());
    if word(0) != spirv::MAGIC_NUMBER {
        return Ok(None);
    }

    let mut int_types = HashMap::<u32, IntType>::new();
    let mut value_types = HashMap::new();
    let mut glsl_sets = HashSet::new();
    let mut rewrites = Vec::new();
    let mut cursor = HEADER;
    while cursor < word_count {
        let header = word(cursor);
        let length = (header >> 16) as usize;
        if length == 0 || length > word_count - cursor {
            return Err("invalid SPIR-V instruction size");
        }
        let op = Op::from_u32(header & 0xffff);
        if op == Some(Op::ExtInstImport)
            && length >= 5
            && input[(cursor + 2) * 4..(cursor + length) * 4].starts_with(b"GLSL.std.450\0")
        {
            glsl_sets.insert(word(cursor + 1));
        }
        if op == Some(Op::TypeInt) && length == 4 && word(cursor + 2) == 32 {
            int_types.insert(
                word(cursor + 1),
                IntType {
                    signed: word(cursor + 3) != 0,
                    lanes: 1,
                },
            );
        }
        if op == Some(Op::TypeVector) && length == 4 {
            if let Some(&scalar) = int_types.get(&word(cursor + 2)) {
                int_types.insert(
                    word(cursor + 1),
                    IntType {
                        lanes: word(cursor + 3),
                        ..scalar
                    },
                );
            }
        }
        if length >= 3 && int_types.contains_key(&word(cursor + 1)) {
            let result_type = word(cursor + 1);
            let result_id = word(cursor + 2);
            let operand = match (op, length) {
                (Some(Op::BitCount), 4) => Some(word(cursor + 3)),
                (Some(Op::ExtInst), 6)
                    if glsl_sets.contains(&word(cursor + 3))
                        && matches!(
                            GLOp::from_u32(word(cursor + 4)),
                            Some(GLOp::FindILsb | GLOp::FindSMsb | GLOp::FindUMsb)
                        ) =>
                {
                    Some(word(cursor + 5))
                }
                _ => None,
            };
            if let Some(&operand_type) = operand.and_then(|id| value_types.get(&id)) {
                let a = int_types[&result_type];
                let b = int_types[&operand_type];
                if a.lanes == b.lanes && a.signed != b.signed {
                    rewrites.push((cursor, length, operand_type, result_type, result_id));
                }
            }
            value_types.insert(result_id, result_type);
        }
        cursor += length;
    }
    if rewrites.is_empty() {
        return Ok(None);
    }

    let mut output = Vec::with_capacity(input.len() + rewrites.len() * 16);
    let mut next_id = word(3);
    let mut copied = 0;
    for (cursor, length, operand_type, result_type, result_id) in rewrites {
        output.extend_from_slice(&input[copied * 4..(cursor + 1) * 4]);
        output.extend_from_slice(&operand_type.to_le_bytes());
        output.extend_from_slice(&next_id.to_le_bytes());
        output.extend_from_slice(&input[(cursor + 3) * 4..(cursor + length) * 4]);
        for operand in [
            4 << 16 | Op::Bitcast as u32,
            result_type,
            result_id,
            next_id,
        ] {
            output.extend_from_slice(&operand.to_le_bytes());
        }
        next_id = next_id.checked_add(1).ok_or("SPIR-V ID bound overflow")?;
        copied = cursor + length;
    }
    output.extend_from_slice(&input[copied * 4..]);
    output[3 * 4..4 * 4].copy_from_slice(&next_id.to_le_bytes());
    Ok(Some(output))
}
