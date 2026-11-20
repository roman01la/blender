// SPDX-License-Identifier: GPL-2.0-or-later

use naga::{back::wgsl, front::spv, valid};
use std::sync::Mutex;

mod border_sampler;
mod depth_texture;
mod float_literals;
mod image_coordinates;
mod integer_builtins;
mod ocio_lut;
mod packed_normal;
mod sampled_image;
mod signed_mod;
mod specialization;
mod storage_struct;
mod switch_default;
mod switch_fallthrough;
mod texture_1d;

// The host calls the translator synchronously and copies the result before calling it again.
// Four groups of 128 bindings, plus two words of border-sampler override slots.
const BINDING_MASK_WORDS: usize = 18;
type BindingMasks = [u32; BINDING_MASK_WORDS];
static RESULT: Mutex<(Vec<u8>, BindingMasks)> = Mutex::new((Vec::new(), [0; BINDING_MASK_WORDS]));

#[no_mangle]
pub extern "C" fn naga_alloc(size: usize) -> *mut u8 {
    Box::into_raw(vec![0u8; size].into_boxed_slice()).cast::<u8>()
}

/// Translates one SPIR-V module and consumes the allocation from naga_alloc.
/// Returns zero on success and one if the result contains a diagnostic.
///
/// # Safety
/// The pointer and size must identify an outstanding allocation returned by
/// `naga_alloc`, with every byte initialized.
#[no_mangle]
pub unsafe extern "C" fn naga_translate(
    ptr: *mut u8,
    size: usize,
    depth_lo: u32,
    depth_hi: u32,
    packed_normals: u32,
) -> i32 {
    let input = unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, size)) };
    let (status, result, bindings) = match translate(
        &input,
        u64::from(depth_lo) | u64::from(depth_hi) << 32,
        packed_normals,
    ) {
        Ok((wgsl, bindings)) => (0, wgsl, bindings),
        Err(message) => (1, message, [0; BINDING_MASK_WORDS]),
    };
    *RESULT.lock().unwrap() = (result.into_bytes(), bindings);
    status
}

#[no_mangle]
pub extern "C" fn naga_result_ptr() -> *const u8 {
    RESULT.lock().unwrap().0.as_ptr()
}

#[no_mangle]
pub extern "C" fn naga_result_size() -> usize {
    RESULT.lock().unwrap().0.len()
}

#[no_mangle]
pub extern "C" fn naga_binding_masks_ptr() -> *const u32 {
    RESULT.lock().unwrap().1.as_ptr()
}

fn translate(
    input: &[u8],
    depth_slots: u64,
    packed_normals: u32,
) -> Result<(String, BindingMasks), String> {
    let options = spv::Options {
        // The Blender WebGPU code generator already uses WebGPU clip coordinates.
        adjust_coordinate_space: false,
        ..Default::default()
    };
    let sampled = sampled_image::lower(input)?;
    let input = sampled.as_deref().unwrap_or(input);
    let lowered = texture_1d::lower(input)?;
    let input = lowered.as_deref().unwrap_or(input);
    let depth = depth_texture::lower(input, depth_slots)?;
    let input = depth.as_deref().unwrap_or(input);
    let signed_modulo = signed_mod::lower(input)?;
    let input = signed_modulo.as_deref().unwrap_or(input);
    let integers = integer_builtins::lower(input)?;
    let input = integers.as_deref().unwrap_or(input);
    let storage = storage_struct::lower(input)?;
    let input = storage.as_deref().unwrap_or(input);
    let switches = switch_default::lower(input)?;
    let input = switches.as_deref().unwrap_or(input);
    let coordinates = image_coordinates::lower(input)?;
    let input = coordinates.as_deref().unwrap_or(input);
    let mut module = spv::parse_u8_slice(input, &options)
        .map_err(|error| format!("SPIR-V import failed: {error:?}"))?;
    if depth.is_some() {
        depth_texture::repair_gathers(&mut module);
    }
    if lowered.is_some() || coordinates.is_some() {
        texture_1d::repair_layer_queries(&mut module);
    }
    if packed_normals != 0 {
        packed_normal::lower(&mut module, packed_normals)?;
    }
    ocio_lut::lower(&mut module);
    lower_volume_sampling(&mut module);
    declare_builtin_result_types(&mut module);
    // WGSL's pack/unpack2x16float operate on u32 and f32 without the f16 extension.
    // Clip-region shaders write gl_ClipDistance, which the WGSL writer emits with
    // `enable clip_distances;` for devices created with the matching feature.
    let info = valid::Validator::new(
        valid::ValidationFlags::all(),
        valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32
            | valid::Capabilities::DUAL_SOURCE_BLENDING
            | valid::Capabilities::CLIP_DISTANCES,
    )
    .validate(&module)
    .map_err(|error| format!("WebGPU validation failed: {error:?}"))?;
    // This writer-only expansion repeats expressions in disjoint WGSL scopes;
    // the expression types and usage metadata come from the original IR.
    let expanded_switches = switch_fallthrough::lower(&mut module);
    let wgsl = wgsl::write_string(&module, &info, wgsl::WriterFlags::empty())
        .map_err(|error| format!("WGSL generation failed: {error:?}"))?;
    let wgsl = if expanded_switches {
        switch_fallthrough::repair_declarations(wgsl)
    } else {
        wgsl
    };
    let wgsl = specialization::lower(wgsl)?;
    let wgsl = float_literals::lower(wgsl);
    let (wgsl, border_slots) = border_sampler::lower(wgsl);
    let mut bindings = [0; BINDING_MASK_WORDS];
    bindings[16] = border_slots as u32;
    bindings[17] = (border_slots >> 32) as u32;
    for (index, _) in module.entry_points.iter().enumerate() {
        let entry = info.get_entry_point(index);
        for (handle, global) in module.global_variables.iter() {
            if let Some(binding) = global.binding {
                if !entry[handle].is_empty() {
                    if binding.group >= 4 || binding.binding >= 128 {
                        return Err(
                            "shader resource exceeds Blender's WebGPU binding range".to_owned()
                        );
                    }
                    let address = binding.group * 128 + binding.binding;
                    bindings[(address / 32) as usize] |= 1 << (address % 32);
                }
            }
        }
    }
    Ok((wgsl, bindings))
}

fn lower_volume_sampling(module: &mut naga::Module) {
    use naga::{Expression, SampleLevel};

    let evaluates_froxels = module.functions.iter().any(|(_, function)| {
        function
            .name
            .as_deref()
            .is_some_and(|name| name.starts_with("eevee_surf_volume("))
    });
    if !evaluates_froxels {
        return;
    }
    // The volume fragment shader evaluates a node tree at each occupied froxel.
    // Neighboring fragments may evaluate different slices, so they cannot
    // supply screen-space derivatives for the automatic mip-level calculation.
    for function in module
        .functions
        .iter_mut()
        .map(|(_, function)| function)
        .chain(
            module
                .entry_points
                .iter_mut()
                .map(|entry| &mut entry.function),
        )
    {
        for (_, expression) in function.expressions.iter_mut() {
            if let Expression::ImageSample { level, .. } = expression {
                if *level == SampleLevel::Auto {
                    *level = SampleLevel::Zero;
                }
            }
        }
    }
}

fn declare_builtin_result_types(module: &mut naga::Module) {
    use naga::{Expression, MathFunction, PredeclaredType, Scalar, VectorSize};
    let mut modf = false;
    let mut frexp = false;
    for function in module
        .functions
        .iter()
        .map(|(_, function)| function)
        .chain(module.entry_points.iter().map(|entry| &entry.function))
    {
        for (_, expression) in function.expressions.iter() {
            if let Expression::Math { fun, .. } = expression {
                modf |= *fun == MathFunction::Modf;
                frexp |= *fun == MathFunction::Frexp;
            }
        }
    }
    // SPIR-V imports the struct-returning instructions without seeding the
    // builtin result type table used by Naga's WGSL type resolution.
    for size in [
        None,
        Some(VectorSize::Bi),
        Some(VectorSize::Tri),
        Some(VectorSize::Quad),
    ] {
        if modf {
            module.generate_predeclared_type(PredeclaredType::ModfResult {
                size,
                scalar: Scalar::F32,
            });
        }
        if frexp {
            module.generate_predeclared_type(PredeclaredType::FrexpResult {
                size,
                scalar: Scalar::F32,
            });
        }
    }
}
