// SPDX-License-Identifier: GPL-2.0-or-later

use naga::{Expression, Function, GlobalVariable, Module, SampleLevel};

/// Blender's OpenColorIO LUTs have a single mip. Select it explicitly so
/// tetrahedral interpolation can sample inside pixel-dependent branches.
pub fn lower(module: &mut Module) {
    let globals = &module.global_variables;
    for (_, function) in module.functions.iter_mut() {
        lower_function(function, globals);
    }
    for entry in &mut module.entry_points {
        lower_function(&mut entry.function, globals);
    }
}

fn lower_function(function: &mut Function, globals: &naga::Arena<GlobalVariable>) {
    let samples: Vec<_> = function
        .expressions
        .iter()
        .filter_map(|(handle, expression)| {
            let Expression::ImageSample {
                image,
                gather: None,
                level: SampleLevel::Auto,
                depth_ref: None,
                ..
            } = expression
            else {
                return None;
            };
            let Expression::GlobalVariable(global) = function.expressions[*image] else {
                return None;
            };
            globals[global]
                .name
                .as_deref()
                .is_some_and(is_ocio_lut)
                .then_some(handle)
        })
        .collect();
    for handle in samples {
        if let Expression::ImageSample { level, .. } = &mut function.expressions[handle] {
            *level = SampleLevel::Zero;
        }
    }
}

fn is_ocio_lut(name: &str) -> bool {
    let Some(name) = name
        .strip_prefix("to_display_")
        .or_else(|| name.strip_prefix("to_scene_"))
    else {
        return false;
    };
    name.strip_prefix("lut1d_")
        .or_else(|| name.strip_prefix("lut3d_"))
        .and_then(|name| name.strip_suffix("Sampler"))
        .is_some_and(|index| !index.is_empty() && index.bytes().all(|b| b.is_ascii_digit()))
}
