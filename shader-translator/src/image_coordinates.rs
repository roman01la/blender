// SPDX-License-Identifier: GPL-2.0-or-later

use spirv::{Dim, Op};
use std::collections::HashSet;

/// Naga's SPIR-V array-image importer splits integer coordinates into a coordinate vector
/// and a layer scalar. Its type table must contain a 32-bit integer coordinate vector even
/// when the original module only declares a wider vector with the layer included.
/// Reassemble array size queries from dimensions and layer count, which are distinct in WGSL.
pub fn lower(input: &[u8]) -> Result<Option<Vec<u8>>, &'static str> {
    const HEADER: usize = 5;
    let words = input.len() / 4;
    if words < HEADER || !input.len().is_multiple_of(4) {
        return Ok(None);
    }
    let word = |i: usize| u32::from_le_bytes(input[i * 4..i * 4 + 4].try_into().unwrap());
    if word(0) != spirv::MAGIC_NUMBER {
        return Ok(None);
    }

    let mut array_images = HashSet::new();
    let mut integer_types = [None; 2];
    let mut vectors = Vec::new();
    let mut cursor = HEADER;
    while cursor < words {
        let header = word(cursor);
        let length = (header >> 16) as usize;
        if length == 0 || length > words - cursor {
            return Err("invalid SPIR-V instruction size");
        }
        match (Op::from_u32(header & 0xffff), length) {
            (Some(Op::TypeImage), 9..=10) => {
                if matches!(
                    Dim::from_u32(word(cursor + 3)),
                    Some(Dim::Dim2D | Dim::DimCube)
                ) && word(cursor + 5) != 0
                {
                    array_images.insert(word(cursor + 1));
                }
            }
            (Some(Op::TypeSampledImage), 3) if array_images.contains(&word(cursor + 2)) => {
                array_images.insert(word(cursor + 1));
            }
            (Some(Op::TypeInt), 4) if word(cursor + 2) == 32 => {
                let signed = word(cursor + 3) as usize;
                if signed < integer_types.len() {
                    integer_types[signed] = Some(word(cursor + 1));
                }
            }
            (Some(Op::TypeVector), 4) => {
                vectors.push((word(cursor + 1), word(cursor + 2), word(cursor + 3)));
            }
            (Some(Op::Function), _) => break,
            _ => {}
        }
        cursor += length;
    }
    if array_images.is_empty() || cursor == words {
        return Ok(None);
    }

    let mut output = (0..cursor).map(word).collect::<Vec<_>>();
    let mut next_id = word(3);
    for scalar in integer_types.into_iter().flatten() {
        if !vectors
            .iter()
            .any(|&(_, component, count)| component == scalar && count == 2)
        {
            let id = allocate(&mut next_id)?;
            emit(&mut output, Op::TypeVector, &[id, scalar, 2]);
            vectors.push((id, scalar, 2));
        }
    }

    let mut array_values = HashSet::new();
    while cursor < words {
        let header = word(cursor);
        let length = (header >> 16) as usize;
        if length == 0 || length > words - cursor {
            return Err("invalid SPIR-V instruction size");
        }
        let opcode = Op::from_u32(header & 0xffff);
        if matches!(opcode, Some(Op::ImageQuerySize | Op::ImageQuerySizeLod))
            && length >= 4
            && array_values.contains(&word(cursor + 3))
        {
            let result_type = word(cursor + 1);
            let result = word(cursor + 2);
            let image = word(cursor + 3);
            let scalar = vectors
                .iter()
                .find_map(|&(id, scalar, count)| {
                    (id == result_type && count == 3).then_some(scalar)
                })
                .ok_or("unsupported SPIR-V array texture size result")?;
            let pair = vectors
                .iter()
                .find_map(|&(id, component, count)| {
                    (component == scalar && count == 2).then_some(id)
                })
                .ok_or("missing SPIR-V array texture coordinate type")?;
            let dimensions = allocate(&mut next_id)?;
            let x = allocate(&mut next_id)?;
            let y = allocate(&mut next_id)?;
            let layers = allocate(&mut next_id)?;
            let mut operands = vec![pair, dimensions, image];
            operands.extend((cursor + 4..cursor + length).map(word));
            emit(&mut output, opcode.unwrap(), &operands);
            emit(
                &mut output,
                Op::CompositeExtract,
                &[scalar, x, dimensions, 0],
            );
            emit(
                &mut output,
                Op::CompositeExtract,
                &[scalar, y, dimensions, 1],
            );
            // The Naga SPIR-V front end exposes scalar image queries through this instruction.
            // Replace the placeholder with NumLayers before validating the imported module.
            emit(&mut output, Op::ImageQuerySamples, &[scalar, layers, image]);
            emit(
                &mut output,
                Op::CompositeConstruct,
                &[result_type, result, x, y, layers],
            );
        } else {
            if matches!(
                opcode,
                Some(
                    Op::Load
                        | Op::Image
                        | Op::SampledImage
                        | Op::FunctionParameter
                        | Op::FunctionCall
                        | Op::CopyObject
                        | Op::Phi
                        | Op::Select
                )
            ) && length >= 3
                && array_images.contains(&word(cursor + 1))
            {
                array_values.insert(word(cursor + 2));
            }
            output.extend((cursor..cursor + length).map(word));
        }
        cursor += length;
    }
    if next_id == word(3) {
        return Ok(None);
    }
    output[3] = next_id;
    Ok(Some(
        output.into_iter().flat_map(u32::to_le_bytes).collect(),
    ))
}

fn allocate(next: &mut u32) -> Result<u32, &'static str> {
    let id = *next;
    *next = id.checked_add(1).ok_or("SPIR-V ID bound overflow")?;
    Ok(id)
}

fn emit(output: &mut Vec<u32>, op: Op, operands: &[u32]) {
    output.push(((operands.len() as u32 + 1) << 16) | op as u32);
    output.extend_from_slice(operands);
}
