// SPDX-License-Identifier: GPL-2.0-or-later

use naga::{Expression, Handle, ImageClass, ImageQuery, Module, Type, TypeInner};
use spirv::{Dim, ImageOperands, Op};
use std::collections::HashMap;

fn emit(words: &mut Vec<u32>, op: Op, operands: &[u32]) {
    words.push((((operands.len() + 1) as u32) << 16) | op as u32);
    words.extend_from_slice(operands);
}

struct Identifiers(u32);

impl Identifiers {
    fn next(&mut self) -> u32 {
        let id = self.0;
        self.0 += 1;
        id
    }
}

/// Expand the SPIR-V coordinate before Naga separates spatial and layer coordinates.
pub fn lower(input: &[u8]) -> Result<Option<Vec<u8>>, &'static str> {
    const HEADER: usize = 5;
    if input.len() < HEADER * 4 || !input.len().is_multiple_of(4) {
        return Ok(None);
    }
    // Avoid allocating for the common case, where the module has no 1D image type.
    let word = |i: usize| u32::from_le_bytes(input[i * 4..i * 4 + 4].try_into().unwrap());
    if word(0) != spirv::MAGIC_NUMBER {
        return Ok(None);
    }
    let mut index = HEADER;
    let mut found = false;
    while index < input.len() / 4 {
        let header = word(index);
        let length = (header >> 16) as usize;
        if length == 0 || length > input.len() / 4 - index {
            return Err("invalid SPIR-V instruction size");
        }
        if header & 0xffff == Op::TypeImage as u32
            && length >= 9
            && word(index + 3) == Dim::Dim1D as u32
        {
            found = true;
            break;
        }
        index += length;
    }
    if !found {
        return Ok(None);
    }

    let words: Vec<_> = input
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    let mut instructions = Vec::new();
    let mut i = HEADER;
    while i < words.len() {
        let length = (words[i] >> 16) as usize;
        if length == 0 || length > words.len() - i {
            return Err("invalid SPIR-V instruction size");
        }
        instructions.push((words[i] & 0xffff, &words[i + 1..i + length]));
        i += length;
    }

    let mut images = HashMap::new();
    for &(op, args) in &instructions {
        if let [id, _, dim, depth, arrayed, 0, 1, ..] = args {
            if op == Op::TypeImage as u32 && *dim == Dim::Dim1D as u32 && *depth == 0 {
                images.insert(*id, *arrayed != 0);
            }
        }
    }
    if images.is_empty() {
        return Ok(None);
    }

    let mut image_types = images.clone();
    let mut types = HashMap::new();
    for &(op, args) in &instructions {
        if let [id, image] = args {
            if op == Op::TypeSampledImage as u32 {
                if let Some(&arrayed) = images.get(image) {
                    image_types.insert(*id, arrayed);
                }
            }
        }
        if let [ty, id, ..] = args {
            if matches!(
                Op::from_u32(op),
                Some(
                    Op::Load
                        | Op::FunctionParameter
                        | Op::SampledImage
                        | Op::Image
                        | Op::CopyObject
                )
            ) {
                if let Some(&arrayed) = image_types.get(ty) {
                    types.insert(*id, arrayed);
                }
            }
        }
    }

    let mut ids = Identifiers(words[3]);
    let coordinate_types = |space| {
        instructions.iter().find_map(|&(op, args)| match args {
            [id, 32, 1] if space == Op::TypeInt && op == space as u32 => Some(*id),
            [id, 32] if space == Op::TypeFloat && op == space as u32 => Some(*id),
            _ => None,
        })
    };
    let mut declarations = Vec::new();
    let mut prepare_type = |op, operands: &[u32]| {
        if let Some(id) = coordinate_types(op) {
            id
        } else {
            let id = ids.next();
            emit(&mut declarations, op, &[&[id], operands].concat());
            id
        }
    };
    let sint = prepare_type(Op::TypeInt, &[32, 1]);
    let float = prepare_type(Op::TypeFloat, &[32]);
    let lookup_vector = |scalar, size| {
        instructions.iter().find_map(|&(op, args)| match args {
            [id, component, count]
                if op == Op::TypeVector as u32 && *component == scalar && *count == size =>
            {
                Some(*id)
            }
            _ => None,
        })
    };
    let mut prepare_vector = |scalar, size| {
        lookup_vector(scalar, size).unwrap_or_else(|| {
            let id = ids.next();
            emit(&mut declarations, Op::TypeVector, &[id, scalar, size]);
            id
        })
    };
    let ivec2 = prepare_vector(sint, 2);
    let ivec3 = prepare_vector(sint, 3);
    let vec2 = prepare_vector(float, 2);
    let vec3 = prepare_vector(float, 3);
    let zero = ids.next();
    let half = ids.next();
    emit(&mut declarations, Op::Constant, &[sint, zero, 0]);
    emit(
        &mut declarations,
        Op::Constant,
        &[float, half, 0.5f32.to_bits()],
    );

    let mut output = words[..HEADER].to_vec();
    for (op, args) in instructions {
        if op == Op::Function as u32 && !declarations.is_empty() {
            output.append(&mut declarations);
        }
        if op == Op::TypeImage as u32 && images.contains_key(&args[0]) {
            let mut new_args = args.to_vec();
            new_args[2] = Dim::Dim2D as u32;
            emit(&mut output, Op::TypeImage, &new_args);
            continue;
        }

        if let Some(&arrayed) = args.get(2).and_then(|image| types.get(image)) {
            let operation = Op::from_u32(op);
            if matches!(
                operation,
                Some(Op::ImageFetch | Op::ImageSampleImplicitLod | Op::ImageSampleExplicitLod)
            ) {
                let &[ty, id, image, coordinate, ref operands @ ..] = args else {
                    return Err("invalid SPIR-V texture operation");
                };
                let sample = operation != Some(Op::ImageFetch);
                let (scalar, vector, height) = if sample {
                    (float, if arrayed { vec3 } else { vec2 }, half)
                } else {
                    (sint, if arrayed { ivec3 } else { ivec2 }, zero)
                };
                if operands.first().is_some_and(|mask| {
                    *mask & !(ImageOperands::LOD | ImageOperands::BIAS).bits() != 0
                }) {
                    return Err("unsupported 1D texture sample operands");
                }
                let xy_layer = ids.next();
                let coordinates = if arrayed {
                    let x = ids.next();
                    let layer = ids.next();
                    emit(
                        &mut output,
                        Op::CompositeExtract,
                        &[scalar, x, coordinate, 0],
                    );
                    emit(
                        &mut output,
                        Op::CompositeExtract,
                        &[scalar, layer, coordinate, 1],
                    );
                    vec![vector, xy_layer, x, height, layer]
                } else {
                    vec![vector, xy_layer, coordinate, height]
                };
                emit(&mut output, Op::CompositeConstruct, &coordinates);
                emit(
                    &mut output,
                    operation.unwrap(),
                    &[&[ty, id, image, xy_layer], operands].concat(),
                );
                continue;
            }
            if matches!(operation, Some(Op::ImageQuerySize | Op::ImageQuerySizeLod)) {
                let &[ty, id, image, ref level @ ..] = args else {
                    return Err("invalid SPIR-V texture size query");
                };
                if ty != if arrayed { ivec2 } else { sint } {
                    return Err("unsupported 1D texture query result type");
                }
                let dimensions = ids.next();
                emit(
                    &mut output,
                    operation.unwrap(),
                    &[
                        &[if arrayed { ivec3 } else { ivec2 }, dimensions, image],
                        level,
                    ]
                    .concat(),
                );
                if !arrayed {
                    emit(
                        &mut output,
                        Op::CompositeExtract,
                        &[sint, id, dimensions, 0],
                    );
                    continue;
                }
                let width = ids.next();
                let layers = ids.next();
                emit(
                    &mut output,
                    Op::CompositeExtract,
                    &[sint, width, dimensions, 0],
                );
                // Naga's SPIR-V size query omits array layers. Import this placeholder as an
                // expression, then repair it to NumLayers before validating the Naga module.
                emit(&mut output, Op::ImageQuerySamples, &[sint, layers, image]);
                emit(
                    &mut output,
                    Op::CompositeConstruct,
                    &[ty, id, width, layers],
                );
                continue;
            }
        }

        output.push((((args.len() + 1) as u32) << 16) | op);
        output.extend_from_slice(args);
    }
    output[3] = ids.0;
    Ok(Some(
        output.into_iter().flat_map(u32::to_le_bytes).collect(),
    ))
}

pub fn repair_layer_queries(module: &mut Module) {
    let repair = |function: &mut naga::Function| {
        let image_type = |image: Handle<Expression>| -> Option<Handle<Type>> {
            match function.expressions[image] {
                Expression::GlobalVariable(global) => Some(module.global_variables[global].ty),
                Expression::FunctionArgument(index) => Some(function.arguments[index as usize].ty),
                Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => {
                    if let Expression::GlobalVariable(global) = function.expressions[base] {
                        if let TypeInner::BindingArray { base, .. } =
                            module.types[module.global_variables[global].ty].inner
                        {
                            return Some(base);
                        }
                    }
                    None
                }
                _ => None,
            }
        };
        let placeholders: Vec<_> = function
            .expressions
            .iter()
            .filter_map(|(handle, expression)| {
                let Expression::ImageQuery {
                    image,
                    query: ImageQuery::NumSamples,
                } = *expression
                else {
                    return None;
                };
                // NumSamples is valid for multisampled images only. The lowering pass uses it
                // to carry layer-count queries on single-sampled arrays through the SPIR-V frontend.
                matches!(
                    module.types[image_type(image)?].inner,
                    TypeInner::Image {
                        arrayed: true,
                        class: ImageClass::Sampled { multi: false, .. }
                            | ImageClass::Depth { multi: false }
                            | ImageClass::Storage { .. },
                        ..
                    }
                )
                .then_some((handle, image))
            })
            .collect();
        for (handle, image) in placeholders {
            *function.expressions.get_mut(handle) = Expression::ImageQuery {
                image,
                query: ImageQuery::NumLayers,
            };
        }
    };
    for (_, function) in module.functions.iter_mut() {
        repair(function);
    }
    for entry in &mut module.entry_points {
        repair(&mut entry.function);
    }
}
