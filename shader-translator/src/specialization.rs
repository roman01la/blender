// SPDX-License-Identifier: GPL-2.0-or-later

use std::collections::HashMap;

/// Resolve Blender's scoped GLSL aliases to WGSL overrides. The GLSL
/// initialization assignments avoid SPIR-V specialization operations and name
/// substitution, but a private variable would lose uniformity in WGSL.
pub fn lower(source: String) -> Result<String, &'static str> {
    let Some(start) = source.find("\nfn blender_init_specializations") else {
        return Ok(source);
    };
    let helper = &source[start + 1..];
    let signature_end = helper
        .find("() {\n")
        .ok_or("invalid specialization initializer signature")?;
    let function = helper[3..signature_end].trim();
    let body = &helper[signature_end + "() {\n".len()..];
    let (body, rest) = body
        .split_once("\n}\n")
        .ok_or("missing specialization initializer body")?;
    let mut expressions = HashMap::new();
    let mut overrides = HashMap::new();
    for line in body.lines().map(str::trim) {
        if line.is_empty() || line == "return;" {
            continue;
        }
        let statement = line
            .strip_suffix(';')
            .ok_or("invalid specialization initializer statement")?;
        let (name, value) = statement
            .split_once(" = ")
            .ok_or("missing specialization initializer assignment")?;
        if let Some(name) = name.strip_prefix("let ") {
            expressions.insert(name, value);
        } else {
            overrides.insert(name, expressions.get(value).copied().unwrap_or(value));
        }
    }

    let invoke = format!("{function}();");
    let mut lowered = String::with_capacity(source.len());
    for line in source[..start + 1].lines().chain(rest.lines()) {
        if line.trim() == invoke {
            continue;
        }
        if let Some(declaration) = line.strip_prefix("var<private> ") {
            if let Some((name, ty)) = declaration.split_once(": ") {
                if let Some(initializer) = overrides.remove(name) {
                    let ty = ty
                        .strip_suffix(';')
                        .ok_or("invalid specialization alias declaration")?;
                    lowered.push_str(&format!("override {name}: {ty} = {initializer};\n"));
                    continue;
                }
            }
        }
        lowered.push_str(line);
        lowered.push('\n');
    }
    if !overrides.is_empty() {
        return Err("missing specialization alias declaration");
    }
    Ok(lowered)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restores_uniform_overrides_and_keeps_float_bits() {
        let source = "
@id(0) override bits: u32 = 0u;
@id(1) override switch_value: bool = false;
var<private> radius: f32;
var<private> enabled: bool;

fn blender_init_specializations_u0028_() {
    let _e1 = bitcast<f32>(bits);
    radius = _e1;
    enabled = switch_value;
    return;
}

@compute @workgroup_size(1)
fn main() {
    blender_init_specializations_u0028_();
    if enabled { workgroupBarrier(); }
}
";
        let output = lower(source.to_owned()).unwrap();
        assert!(output.contains("override radius: f32 = bitcast<f32>(bits);"));
        assert!(output.contains("override enabled: bool = switch_value;"));
        assert!(!output.contains("blender_init_specializations"));
        assert!(output.contains("if enabled { workgroupBarrier(); }"));
    }
}
