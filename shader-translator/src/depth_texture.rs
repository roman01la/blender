// SPDX-License-Identifier: GPL-2.0-or-later

use naga::{Expression, Function, Module, SwizzleComponent, VectorSize};
use spirv::{Decoration, Op};
use std::collections::{HashMap, HashSet};

/// GLSL represents non-comparison depth reads as float color samplers. Give the
/// marked bindings distinct SPIR-V types so Naga imports them as depth textures.
/// Color bindings can share the original SPIR-V image and sampled-image types.
pub fn lower(input: &[u8], depth_slots: u64) -> Result<Option<Vec<u8>>, &'static str> {
    const HEADER: usize = 5;
    if depth_slots == 0 || input.len() < HEADER * 4 || !input.len().is_multiple_of(4) {
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
    let mut offset = HEADER;
    let mut texture_set = HashSet::new();
    let mut depth_variables = HashSet::new();
    while offset < words.len() {
        let header = words[offset];
        let length = (header >> 16) as usize;
        if length == 0 || length > words.len() - offset {
            return Err("invalid SPIR-V instruction size");
        }
        let op = Op::from_u32(header & 0xffff);
        let operands = &words[offset + 1..offset + length];
        if op == Some(Op::Decorate) {
            if let [id, decoration, value] = *operands {
                if decoration == Decoration::DescriptorSet as u32 && value == 2 {
                    texture_set.insert(id);
                }
                if decoration == Decoration::Binding as u32
                    && value < 128
                    && value % 2 == 0
                    && depth_slots & (1 << (value / 2)) != 0
                {
                    depth_variables.insert(id);
                }
            }
        }
        instructions.push((header, op, operands));
        offset += length;
    }
    depth_variables.retain(|id| texture_set.contains(id));
    if depth_variables.is_empty() {
        return Ok(None);
    }

    // The sampler split has already exposed texture arguments independently.
    // Carry a depth binding's type through the functions that sample it.
    let mut functions = Vec::<(u32, u32, Vec<(u32, u32)>)>::new();
    let mut function_indices = HashMap::new();
    for &(_, op, operands) in &instructions {
        match (op, operands) {
            (Some(Op::Function), [_, id, _, signature]) => {
                function_indices.insert(*id, functions.len());
                functions.push((*id, *signature, Vec::new()));
            }
            (Some(Op::FunctionParameter), [ty, id]) => {
                functions.last_mut().unwrap().2.push((*ty, *id));
            }
            _ => {}
        }
    }
    let mut depth_values = depth_variables;
    loop {
        let count = depth_values.len();
        for &(_, op, operands) in &instructions {
            match (op, operands) {
                (
                    Some(Op::Load | Op::SampledImage | Op::Image | Op::CopyObject),
                    [_, id, source, ..],
                ) if depth_values.contains(source) => {
                    depth_values.insert(*id);
                }
                (Some(Op::FunctionCall), [_, _, function, arguments @ ..]) => {
                    for (argument, (_, parameter)) in arguments
                        .iter()
                        .zip(&functions[function_indices[function]].2)
                    {
                        if depth_values.contains(argument) {
                            depth_values.insert(*parameter);
                        }
                    }
                }
                _ => {}
            }
        }
        if depth_values.len() == count {
            break;
        }
    }

    let mut signatures: HashMap<_, Vec<_>> = HashMap::new();
    for (function, signature, parameters) in &functions {
        if parameters
            .iter()
            .any(|(_, parameter)| depth_values.contains(parameter))
        {
            signatures
                .entry(*signature)
                .or_default()
                .push((*function, parameters));
        }
    }

    let mut pointers = HashSet::new();
    for &(_, op, operands) in &instructions {
        if let [pointer, id, ..] = *operands {
            if matches!(op, Some(Op::Variable | Op::FunctionParameter))
                && depth_values.contains(&id)
            {
                pointers.insert(pointer);
            }
        }
    }
    let mut images = HashSet::new();
    for &(_, op, operands) in &instructions {
        if let [id, _, image] = *operands {
            if op == Some(Op::TypePointer) && pointers.contains(&id) {
                images.insert(image);
            }
        }
    }

    let mut output = words[..HEADER].to_vec();
    let mut next_id = words[3];
    let mut replacements = HashMap::new();
    let mut function_signatures = HashMap::new();
    let mut remapped_signatures = HashMap::new();
    for (header, op, operands) in instructions {
        output.push(header);
        let start = output.len();
        output.extend_from_slice(operands);
        match (op, operands) {
            (Some(Op::TypeImage), [id, _, _, _, _, _, _, _]) if images.contains(id) => {
                let mut depth_type = operands.to_vec();
                depth_type[0] = next_id;
                depth_type[3] = 1;
                output.push(header);
                output.extend_from_slice(&depth_type);
                replacements.insert(*id, next_id);
                next_id += 1;
            }
            (Some(Op::TypePointer), [id, storage, image]) if pointers.contains(id) => {
                let image = *replacements.get(image).ok_or("missing depth image type")?;
                output.extend_from_slice(&[header, next_id, *storage, image]);
                replacements.insert(*id, next_id);
                next_id += 1;
            }
            (Some(Op::TypeSampledImage), [id, image]) if images.contains(image) => {
                let image = *replacements
                    .get(image)
                    .ok_or("missing sampled depth image type")?;
                output.extend_from_slice(&[header, next_id, image]);
                replacements.insert(*id, next_id);
                next_id += 1;
            }
            (Some(Op::TypeFunction), [id, result_type, ..]) => {
                if let Some(retyped) = signatures.get(id) {
                    for (function, parameters) in retyped {
                        let arguments: Vec<_> = parameters
                            .iter()
                            .map(|(ty, param)| {
                                if depth_values.contains(param) {
                                    replacements[ty]
                                } else {
                                    *ty
                                }
                            })
                            .collect();
                        let signature = *remapped_signatures
                            .entry((*result_type, arguments.clone()))
                            .or_insert_with(|| {
                                let signature = next_id;
                                next_id += 1;
                                output.extend_from_slice(&[header, signature, *result_type]);
                                output.extend(arguments);
                                signature
                            });
                        function_signatures.insert(*function, signature);
                    }
                }
            }
            (Some(Op::Function), [_, function, _, _]) => {
                if let Some(signature) = function_signatures.get(function) {
                    output[start + 3] = *signature;
                }
            }
            (Some(Op::FunctionCall), [_, _, function, arguments @ ..]) => {
                if arguments
                    .iter()
                    .zip(&functions[function_indices[function]].2)
                    .any(|(argument, (_, parameter))| {
                        depth_values.contains(argument) != depth_values.contains(parameter)
                    })
                {
                    return Err("a shader parameter cannot represent both depth and color images");
                }
            }
            (Some(Op::Variable | Op::FunctionParameter), [pointer, id, ..])
                if depth_values.contains(id) =>
            {
                output[start] = *replacements
                    .get(pointer)
                    .ok_or("missing depth image pointer")?;
            }
            (Some(Op::Load | Op::SampledImage | Op::Image | Op::CopyObject), [ty, id, ..])
                if depth_values.contains(id) =>
            {
                output[start] = *replacements
                    .get(ty)
                    .ok_or("missing depth image value type")?;
            }
            _ => {}
        }
    }
    output[3] = next_id;
    Ok(Some(
        output.into_iter().flat_map(u32::to_le_bytes).collect(),
    ))
}

/// Ordinary depth sampling returns a scalar. A depth gather already returns
/// four texels, but the SPIR-V importer splats the result of both operations.
/// An identity swizzle preserves the expression handle and its emission scope.
pub fn repair_gathers(module: &mut Module) {
    fn repair(function: &mut Function) {
        let gathers: Vec<_> = function
            .expressions
            .iter()
            .filter_map(|(handle, expression)| {
                let Expression::Splat {
                    size: VectorSize::Quad,
                    value,
                } = *expression
                else {
                    return None;
                };
                matches!(
                    function.expressions[value],
                    Expression::ImageSample {
                        gather: Some(_),
                        ..
                    }
                )
                .then_some((handle, value))
            })
            .collect();
        for (handle, vector) in gathers {
            use SwizzleComponent::{W, X, Y, Z};
            function.expressions[handle] = Expression::Swizzle {
                size: VectorSize::Quad,
                vector,
                pattern: [X, Y, Z, W],
            };
        }
    }
    for (_, function) in module.functions.iter_mut() {
        repair(function);
    }
    for entry_point in &mut module.entry_points {
        repair(&mut entry_point.function);
    }
}
