// SPDX-License-Identifier: GPL-2.0-or-later

use spirv::Op;
use std::collections::HashMap;

/// Naga's SPIR-V importer uses float division for SMod and uses the vector
/// size as the scalar width. Preserve the integer result for both scalars and
/// vectors by correcting a signed remainder when the divisor has the other sign.
pub fn lower(input: &[u8]) -> Result<Option<Vec<u8>>, &'static str> {
    const HEADER: usize = 5;
    if input.len() < HEADER * 4 || !input.len().is_multiple_of(4) {
        return Ok(None);
    }
    let word = |i: usize| u32::from_le_bytes(input[i * 4..i * 4 + 4].try_into().unwrap());
    if word(0) != spirv::MAGIC_NUMBER {
        return Ok(None);
    }
    let mut offset = HEADER;
    let mut found = false;
    while offset < input.len() / 4 {
        let header = word(offset);
        let count = (header >> 16) as usize;
        if count == 0 || count > input.len() / 4 - offset {
            return Err("invalid SPIR-V instruction size");
        }
        if header & 0xffff == Op::SMod as u32 {
            found = true;
            break;
        }
        offset += count;
    }
    if !found {
        return Ok(None);
    }

    let words: Vec<_> = (0..input.len() / 4).map(word).collect();
    let mut types = HashMap::new();
    let mut vectors = HashMap::new();
    let mut bool_type = None;
    let mut needed = Vec::new();
    let mut offset = HEADER;
    let mut first_function = None;
    while offset < words.len() {
        let header = words[offset];
        let count = (header >> 16) as usize;
        if count == 0 || count > words.len() - offset {
            return Err("invalid SPIR-V instruction size");
        }
        let op = Op::from_u32(header & 0xffff);
        let args = &words[offset + 1..offset + count];
        match (op, args) {
            (Some(Op::TypeBool), [id]) => bool_type = Some(*id),
            (Some(Op::TypeVector), [id, scalar, size]) => {
                types.insert(*id, *size);
                vectors.insert((*scalar, *size), *id);
            }
            (Some(Op::Function), _) => {
                first_function.get_or_insert(offset);
            }
            (Some(Op::SMod), [ty, _, _, _]) if !needed.contains(ty) => needed.push(*ty),
            _ => {}
        }
        offset += count;
    }
    let first_function = first_function.ok_or("signed modulo outside a SPIR-V function")?;
    let mut output = words[..first_function].to_vec();
    let mut next_id = words[3];
    let mut id = || {
        let previous = next_id;
        next_id += 1;
        previous
    };
    let bool_type = bool_type.unwrap_or_else(|| {
        let ty = id();
        emit(&mut output, Op::TypeBool, &[ty]);
        ty
    });
    let mut operands = HashMap::new();
    for ty in needed {
        let condition = if let Some(&size) = types.get(&ty) {
            *vectors.entry((bool_type, size)).or_insert_with(|| {
                let vector = id();
                emit(&mut output, Op::TypeVector, &[vector, bool_type, size]);
                vector
            })
        } else {
            bool_type
        };
        let zero = id();
        emit(&mut output, Op::ConstantNull, &[ty, zero]);
        operands.insert(ty, (condition, zero));
    }

    let mut offset = first_function;
    while offset < words.len() {
        let header = words[offset];
        let count = (header >> 16) as usize;
        let instruction = &words[offset..offset + count];
        if header & 0xffff == Op::SMod as u32 {
            let [_, ty, result, lhs, rhs] = *instruction else {
                return Err("invalid SPIR-V signed modulo");
            };
            let (condition, zero) = operands[&ty];
            let remainder = id();
            let nonzero = id();
            let sign_bits = id();
            let signs_differ = id();
            let signed_divisor = id();
            let correction = id();
            emit(&mut output, Op::SRem, &[ty, remainder, lhs, rhs]);
            emit(
                &mut output,
                Op::INotEqual,
                &[condition, nonzero, remainder, zero],
            );
            emit(
                &mut output,
                Op::BitwiseXor,
                &[ty, sign_bits, remainder, rhs],
            );
            emit(
                &mut output,
                Op::SLessThan,
                &[condition, signs_differ, sign_bits, zero],
            );
            emit(
                &mut output,
                Op::Select,
                &[ty, signed_divisor, signs_differ, rhs, zero],
            );
            emit(
                &mut output,
                Op::Select,
                &[ty, correction, nonzero, signed_divisor, zero],
            );
            emit(&mut output, Op::IAdd, &[ty, result, remainder, correction]);
        } else {
            output.extend_from_slice(instruction);
        }
        offset += count;
    }
    output[3] = next_id;
    Ok(Some(
        output.into_iter().flat_map(u32::to_le_bytes).collect(),
    ))
}

fn emit(output: &mut Vec<u32>, op: Op, operands: &[u32]) {
    output.push(((operands.len() + 1) as u32) << 16 | op as u32);
    output.extend_from_slice(operands);
}
