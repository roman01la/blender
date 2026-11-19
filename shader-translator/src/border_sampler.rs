// SPDX-License-Identifier: GPL-2.0-or-later

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write;
use std::ops::Range;

const OVERRIDE_BASE: u32 = 1024;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Texture {
    dimensions: usize,
    array: bool,
    scalar: &'static str,
    depth: bool,
}

impl Texture {
    fn from_type(tokens: &[Token]) -> Option<Self> {
        let (dimensions, array, depth) = match tokens.first()?.text {
            "texture_2d" => (2, false, false),
            "texture_2d_array" => (2, true, false),
            "texture_3d" => (3, false, false),
            "texture_depth_2d" => (2, false, true),
            "texture_depth_2d_array" => (2, true, true),
            _ => return None,
        };
        let scalar = if depth {
            "f32"
        } else {
            match tokens.get(2)?.text {
                "f32" => "f32",
                "u32" => "u32",
                "i32" => "i32",
                _ => return None,
            }
        };
        Some(Self {
            dimensions,
            array,
            scalar,
            depth,
        })
    }

    fn wgsl(self) -> String {
        let array = if self.array { "_array" } else { "" };
        if self.depth {
            format!("texture_depth_2d{array}")
        } else {
            format!("texture_{}d{array}<{}>", self.dimensions, self.scalar)
        }
    }
}

#[derive(Clone, Copy)]
enum State {
    Slot(u32),
    Argument(usize),
}

impl State {
    fn expression(self, slots: &mut u64, arguments: &mut Vec<usize>) -> String {
        match self {
            State::Slot(slot) => {
                *slots |= 1 << slot;
                format!("blender_webgpu_border_slot_{slot}")
            }
            State::Argument(index) => {
                if !arguments.contains(&index) {
                    arguments.push(index);
                }
                format!("blender_webgpu_border_arg_{index}")
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Sample {
    Auto,
    Bias,
    Level,
    Grad,
    Gather,
}

impl Sample {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "textureSample" => Some(Self::Auto),
            "textureSampleBias" => Some(Self::Bias),
            "textureSampleLevel" => Some(Self::Level),
            "textureSampleGrad" => Some(Self::Grad),
            "textureGather" => Some(Self::Gather),
            _ => None,
        }
    }

    fn builtin(self) -> &'static str {
        match self {
            Self::Auto => "textureSample",
            Self::Bias => "textureSampleBias",
            Self::Level => "textureSampleLevel",
            Self::Grad => "textureSampleGrad",
            Self::Gather => "textureGather",
        }
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Helper<'a> {
    texture: Texture,
    sample: Sample,
    offset: Option<&'a str>,
    component: Option<&'a str>,
}

#[derive(Clone, Copy)]
struct Token<'a> {
    text: &'a str,
    start: usize,
    end: usize,
}

// The input is Naga's output, without comments or user-defined grammar extensions.
fn tokens(source: &str) -> Vec<Token<'_>> {
    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let mut end = 0;
    while end < bytes.len() {
        if bytes[end].is_ascii_whitespace() {
            end += 1;
            continue;
        }
        let start = end;
        end += source[start..].chars().next().unwrap().len_utf8();
        if bytes[start].is_ascii_alphanumeric() || bytes[start] == b'_' {
            while end < bytes.len() && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_') {
                end += 1;
            }
        }
        tokens.push(Token {
            text: &source[start..end],
            start,
            end,
        });
    }
    tokens
}

fn template(name: &str) -> bool {
    name == "array"
        || name == "ptr"
        || name == "bitcast"
        || name == "atomic"
        || matches!(
            name,
            "texture_1d"
                | "texture_2d"
                | "texture_2d_array"
                | "texture_3d"
                | "texture_cube"
                | "texture_cube_array"
                | "texture_multisampled_2d"
                | "vec2"
                | "vec3"
                | "vec4"
                | "mat2x2"
                | "mat2x3"
                | "mat2x4"
                | "mat3x2"
                | "mat3x3"
                | "mat3x4"
                | "mat4x2"
                | "mat4x3"
                | "mat4x4"
        )
}

fn arguments<'a>(tokens: &[Token<'a>], open: usize) -> (Vec<Range<usize>>, usize) {
    debug_assert_eq!(tokens[open].text, "(");
    let mut stack = Vec::new();
    let mut result = Vec::new();
    let mut start = open + 1;
    for i in start..tokens.len() {
        let token = tokens[i].text;
        if stack.last() == Some(&token) {
            stack.pop();
            continue;
        }
        match token {
            "(" => stack.push(")"),
            "[" => stack.push("]"),
            "<" if template(tokens[i - 1].text) => stack.push(">"),
            "," | ")" if stack.is_empty() => {
                if start < i {
                    result.push(start..i);
                }
                if token == ")" {
                    return (result, i);
                }
                start = i + 1;
            }
            _ => {}
        }
    }
    unreachable!("Naga generated an unbalanced WGSL argument list")
}

fn slice<'a>(source: &'a str, tokens: &[Token], range: Range<usize>) -> &'a str {
    &source[tokens[range.start].start..tokens[range.end - 1].end]
}

fn global_textures<'a>(tokens: &[Token<'a>]) -> HashMap<&'a str, (Texture, State)> {
    let mut globals = HashMap::new();
    for (i, window) in tokens.windows(14).enumerate() {
        if window[0].text != "@"
            || window[1].text != "group"
            || window[3].text != "2"
            || window[5].text != "@"
            || window[6].text != "binding"
            || window[10].text != "var"
        {
            continue;
        }
        let Some(slot) = window[8]
            .text
            .parse::<u32>()
            .ok()
            .filter(|id| id % 2 == 0 && *id < 128)
        else {
            continue;
        };
        if let Some(texture) = Texture::from_type(&tokens[i + 13..]) {
            globals.insert(window[11].text, (texture, State::Slot(slot / 2)));
        }
    }
    globals
}

fn sampling_call<'a>(
    source: &'a str,
    tokens: &[Token<'a>],
    name: &str,
    args: &[Range<usize>],
    texture_of: impl Fn(&str) -> Option<(Texture, State)>,
) -> Option<(Helper<'a>, State, Range<usize>)> {
    let sample = Sample::from_name(name)?;
    let (texture, state, component) =
        if let Some((texture, state)) = texture_of(slice(source, tokens, args.first()?.clone())) {
            (texture, state, None)
        } else if sample == Sample::Gather {
            let (texture, state) = texture_of(slice(source, tokens, args.get(1)?.clone()))?;
            (texture, state, Some(slice(source, tokens, args[0].clone())))
        } else {
            return None;
        };
    if sample != Sample::Gather && texture.scalar != "f32" {
        return None;
    }
    let fixed_count = 3
        + usize::from(texture.array)
        + usize::from(component.is_some())
        + match sample {
            Sample::Bias | Sample::Level => 1,
            Sample::Grad => 2,
            _ => 0,
        };
    let offset = args
        .get(fixed_count)
        .map(|range| slice(source, tokens, range.clone()));
    Some((
        Helper {
            texture,
            sample,
            offset,
            component,
        },
        state,
        args[usize::from(component.is_some())].start..args[fixed_count - 1].end,
    ))
}

fn shared_helpers(out: &mut String) {
    out.push_str(
        r#"
fn blender_webgpu_border_weight(x: f32, size: f32, mode: u32, linear: bool) -> f32 {
    if (mode != 3u) { return 1.0; }
    if (!linear) { return select(0.0, 1.0, x >= 0.0 && x < size); }
    return clamp(min(x + 0.5, size - x + 0.5), 0.0, 1.0);
}

fn blender_webgpu_border_gather_texel(p: vec2<f32>, size: vec2<f32>, state: u32) -> bool {
    return ((state & 3u) != 3u || (p.x >= 0.0 && p.x < size.x)) &&
           (((state >> 2u) & 3u) != 3u || (p.y >= 0.0 && p.y < size.y));
}
"#,
    );
}

fn write_helper(out: &mut String, id: usize, helper: &Helper) {
    let t = helper.texture;
    let texture = t.wgsl();
    let n = t.dimensions;
    let array_arg = if t.array { ", layer: i32" } else { "" };
    let layer = if t.array { ", layer" } else { "" };
    let coord = format!("vec{n}<f32>");
    let offset = helper
        .offset
        .map_or_else(String::new, |offset| format!(", {offset}"));
    let texel_offset = helper
        .offset
        .map_or_else(String::new, |offset| format!(" + {coord}({offset})"));
    let sample = helper.sample;
    let name = sample.builtin();
    let result = if t.depth && sample != Sample::Gather {
        "f32".to_owned()
    } else {
        format!("vec4<{}>", t.scalar)
    };
    let zero = format!("{result}(0)");
    if sample == Sample::Gather {
        let component = helper
            .component
            .map_or_else(String::new, |c| format!("{c}, "));
        writeln!(out, r#"
fn blender_webgpu_border_sample_{id}(image: {texture}, s: sampler, uv: {coord}{array_arg}, state: u32) -> {result} {{
    let sampled = {name}({component}image, s, uv{layer}{offset});
    if (state == 0u) {{ return sampled; }}
    let size = vec2<f32>(textureDimensions(image));
    let p = floor(uv * size{texel_offset} - vec2<f32>(0.5));
    let inside = vec4<bool>(
        blender_webgpu_border_gather_texel(p + vec2<f32>(0.0, 1.0), size, state),
        blender_webgpu_border_gather_texel(p + vec2<f32>(1.0, 1.0), size, state),
        blender_webgpu_border_gather_texel(p + vec2<f32>(1.0, 0.0), size, state),
        blender_webgpu_border_gather_texel(p, size, state));
    return select({zero}, sampled, inside);
}}
"#).unwrap();
        return;
    }
    let mip = if t.depth { "level" } else { "f32(level)" };
    let axis_z = if n == 3 {
        " * blender_webgpu_border_weight(p.z, size.z, (state >> 2u) & 3u, linear)"
    } else {
        ""
    };
    writeln!(out, r#"
fn blender_webgpu_border_level_{id}(image: {texture}, s: sampler, uv: {coord}{array_arg}, level: i32, state: u32) -> {result} {{
    let size = {coord}(textureDimensions(image, level));
    let p = uv * size{texel_offset};
    let linear = (state & 16u) != 0u;
    let weight = blender_webgpu_border_weight(p.x, size.x, state & 3u, linear)
               * blender_webgpu_border_weight(p.y, size.y, (state >> 2u) & 3u, linear){axis_z};
    if (weight == 0.0) {{ return {zero}; }}
    return textureSampleLevel(image, s, uv{layer}, {mip}{offset}) * weight;
}}
"#).unwrap();

    let (extra_args, extra, lod) = match sample {
        Sample::Auto | Sample::Bias | Sample::Grad => {
            let mut lod = format!("    let dims = {coord}(textureDimensions(image));\n");
            let (args, extra, x, y) = match sample {
                Sample::Auto => (String::new(), "", "dpdx(uv)", "dpdy(uv)"),
                Sample::Bias => (", bias: f32".to_owned(), ", bias", "dpdx(uv)", "dpdy(uv)"),
                _ => (
                    format!(", dx: {coord}, dy: {coord}"),
                    ", dx, dy",
                    "dx",
                    "dy",
                ),
            };
            let bias = if sample == Sample::Bias {
                " + clamp(bias, -16.0, 15.99)"
            } else {
                ""
            };
            writeln!(lod,
                     "    let x = {x} * dims;\n    let y = {y} * dims;\n    let lod = 0.5 * log2(max(max(dot(x, x), dot(y, y)), 1.0e-30)){bias};").unwrap();
            (args, extra, lod)
        }
        Sample::Level if t.depth => (
            ", lod_value: i32".to_owned(),
            ", lod_value",
            "    let lod = f32(lod_value);\n".to_owned(),
        ),
        Sample::Level => (", lod: f32".to_owned(), ", lod", String::new()),
        Sample::Gather => unreachable!(),
    };
    writeln!(out, r#"
fn blender_webgpu_border_sample_{id}(image: {texture}, s: sampler, uv: {coord}{array_arg}{extra_args}, state: u32) -> {result} {{
    if (state == 0u) {{ return {name}(image, s, uv{layer}{extra}{offset}); }}
    if ((state & 32u) == 0u) {{ return blender_webgpu_border_level_{id}(image, s, uv{layer}, 0, state); }}
{lod}    let limited = clamp(lod, 0.0, f32(textureNumLevels(image) - 1u));
    if ((state & 16u) == 0u) {{
        let nearest = i32(floor(limited + 0.5));
        return blender_webgpu_border_level_{id}(image, s, uv{layer}, nearest, state);
    }}
    let lo = i32(floor(limited));
    let first = blender_webgpu_border_level_{id}(image, s, uv{layer}, lo, state);
    if (limited == f32(lo)) {{ return first; }}
    let second = blender_webgpu_border_level_{id}(image, s, uv{layer}, lo + 1, state);
    return mix(first, second, fract(limited));
}}
"#).unwrap();
}

const INTEGER_HELPER: &str = r#"
fn blender_webgpu_border_integer_address(p: f32, size: f32, mode: u32) -> i32 {
    if (mode == 1u) {
        return i32(clamp(p - size * floor(p / size), 0.0, size - 1.0));
    }
    if (mode == 2u) {
        let period = 2.0 * size;
        let repeated = p - period * floor(p / period);
        return i32(clamp(min(repeated, period - repeated - 1.0), 0.0, size - 1.0));
    }
    return i32(clamp(p, 0.0, size - 1.0));
}
"#;

fn integer_sample(
    params: &[Range<usize>],
    tokens: &[Token],
    texture: Texture,
    state: &str,
) -> String {
    // GLSL scalar/vector parameters are lowered to pointers to function-local values.
    let param = |i: usize| {
        let tokens = &tokens[params[i].clone()];
        if tokens.get(2).is_some_and(|token| token.text == "ptr") {
            format!("(*{})", tokens[0].text)
        } else {
            tokens[0].text.to_owned()
        }
    };
    let image = param(0);
    let uv = param(1);
    let lod = param(2);
    let n = texture.dimensions;
    let scalar = texture.scalar;
    let uv_plane = if texture.array { ".xy" } else { "" };
    let layer = if texture.array {
        format!(", i32(clamp(floor({uv}.z + 0.5), 0.0, f32(textureNumLayers({image}) - 1u)))")
    } else {
        String::new()
    };
    let mut out = format!(
        r#"
    if ({state} != 0u) {{
        let blender_level = clamp({lod}, 0.0, f32(textureNumLevels({image}) - 1u));
        let blender_mip = select(0, i32(floor(blender_level + 0.5)), ({state} & 32u) != 0u);
        let blender_size = vec{n}<f32>(textureDimensions({image}, blender_mip));
        let blender_texel = floor({uv}{uv_plane} * blender_size);
        let blender_x = {state} & 3u;
        let blender_yz = ({state} >> 2u) & 3u;
"#
    );
    let mut address = Vec::with_capacity(n);
    for (axis, mode) in [('x', "blender_x"), ('y', "blender_yz"), ('z', "blender_yz")]
        .into_iter()
        .take(n)
    {
        writeln!(out, "        if ({mode} == 3u && !(blender_texel.{axis} >= 0.0 && blender_texel.{axis} < blender_size.{axis})) {{ return vec4<{scalar}>(0); }}").unwrap();
        address.push(format!("blender_webgpu_border_integer_address(blender_texel.{axis}, blender_size.{axis}, {mode})"));
    }
    let address = address.join(", ");
    writeln!(
        out,
        "        return textureLoad({image}, vec{n}<i32>({address}){layer}, blender_mip);\n    }}"
    )
    .unwrap();
    out
}

/// Add per-texture address overrides to Naga's sampling expressions. Texture
/// handles can cross GLSL helper calls, so forward their state alongside them.
pub fn lower(source: String) -> (String, u64) {
    if !source.contains("textureSample")
        && !source.contains("textureGather")
        && !source.contains("blender_webgpu_integer_")
    {
        return (source, 0);
    }
    let tokens = tokens(&source);
    let globals = global_textures(&tokens);
    if globals.is_empty() {
        return (source, 0);
    }
    let mut edits = Vec::<(Range<usize>, String)>::new();
    let mut function_args = HashMap::<&str, Vec<usize>>::new();
    let mut helpers = BTreeMap::<Helper, usize>::new();
    let mut integer_helpers = false;
    let mut slots = 0;
    let mut i = 0;
    while i + 2 < tokens.len() {
        if tokens[i].text != "fn" {
            i += 1;
            continue;
        }
        let name = tokens[i + 1].text;
        let (params, signature_end) = arguments(&tokens, i + 2);
        let mut local_textures = HashMap::new();
        for (index, range) in params.iter().enumerate() {
            let param = &tokens[range.clone()];
            if param.get(1).is_some_and(|token| token.text == ":") {
                if let Some(texture) = Texture::from_type(&param[2..]) {
                    local_textures.insert(param[0].text, (texture, State::Argument(index)));
                }
            }
        }
        let texture_of = |name: &str| {
            local_textures
                .get(name)
                .or_else(|| globals.get(name))
                .copied()
        };
        let mut need_args = Vec::new();
        i = signature_end + 1;
        while i < tokens.len() && tokens[i].text != "{" {
            i += 1;
        }
        if name.starts_with("blender_webgpu_integer_") && params.len() == 3 {
            if let Some((texture, state)) = texture_of(tokens[params[0].start].text) {
                integer_helpers = true;
                let state = state.expression(&mut slots, &mut need_args);
                let at = tokens[i].end;
                edits.push((at..at, integer_sample(&params, &tokens, texture, &state)));
            }
        }
        i += 1;
        let mut braces = 1;
        while i + 1 < tokens.len() && braces != 0 {
            match tokens[i].text {
                "{" => braces += 1,
                "}" => braces -= 1,
                _ if tokens[i + 1].text == "(" => {
                    let call = tokens[i].text;
                    if Sample::from_name(call).is_some() || function_args.contains_key(call) {
                        let (args, close) = arguments(&tokens, i + 1);
                        if let Some(required) = function_args.get(call) {
                            let mut insert = String::new();
                            for arg in required {
                                let texture = slice(&source, &tokens, args[*arg].clone());
                                let state = texture_of(texture).map_or_else(
                                    || "0u".to_owned(),
                                    |(_, state)| state.expression(&mut slots, &mut need_args),
                                );
                                write!(insert, ", {state}").unwrap();
                            }
                            let at = tokens[close].start;
                            edits.push((at..at, insert));
                        } else if let Some((helper, state, call_args)) =
                            sampling_call(&source, &tokens, call, &args, texture_of)
                        {
                            let state = state.expression(&mut slots, &mut need_args);
                            let id = helpers.len();
                            let id = helpers.entry(helper).or_insert(id);
                            // Only remove the compile-time component and offset
                            // arguments. Edits inside the other operands survive.
                            let start = tokens[call_args.start].start;
                            let end = tokens[call_args.end - 1].end;
                            edits.push((
                                tokens[i].start..start,
                                format!("blender_webgpu_border_sample_{id}("),
                            ));
                            edits.push((end..tokens[close].start, format!(", {state}")));
                        }
                    }
                }
                _ => {}
            }
            i += 1;
        }
        if !need_args.is_empty() {
            need_args.sort_unstable();
            let mut args = String::new();
            for arg in &need_args {
                write!(args, ", blender_webgpu_border_arg_{arg}: u32").unwrap();
            }
            let at = tokens[signature_end].start;
            edits.push((at..at, args));
            function_args.insert(name, need_args);
        }
    }
    if slots == 0 {
        return (source, 0);
    }
    edits.sort_by_key(|(range, _)| range.start);
    let mut out = String::with_capacity(source.len() + helpers.len() * 1500);
    let mut offset = 0;
    for (range, change) in edits {
        out.push_str(&source[offset..range.start]);
        out.push_str(&change);
        offset = range.end;
    }
    out.push_str(&source[offset..]);
    for slot in 0..64 {
        if slots & (1 << slot) != 0 {
            writeln!(
                out,
                "@id({}) override blender_webgpu_border_slot_{slot}: u32 = 0u;",
                OVERRIDE_BASE + slot
            )
            .unwrap();
        }
    }
    if !helpers.is_empty() {
        shared_helpers(&mut out);
    }
    if integer_helpers {
        out.push_str(INTEGER_HELPER);
    }
    for (helper, id) in helpers {
        write_helper(&mut out, id, &helper);
    }
    (out, slots)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(source: &str) {
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
    }

    #[test]
    fn forwards_border_flags_for_implicit_and_explicit_sampling() {
        let source = r#"
@group(2) @binding(124) var color: texture_2d<f32>;
@group(2) @binding(125) var color_sampler: sampler;
@group(2) @binding(0) var layers: texture_2d_array<f32>;
@group(2) @binding(1) var layer_sampler: sampler;
@group(2) @binding(6) var volume: texture_3d<f32>;
@group(2) @binding(7) var volume_sampler: sampler;

fn sample_tree(s: sampler, t: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
    return textureSample(t, s, uv)
         + textureSampleBias(t, s, uv, -0.4)
         + textureSampleGrad(t, s, uv, dpdx(uv), dpdy(uv))
         + textureGather(1, t, s, uv, vec2<i32>(1, -1));
}

fn forward(t: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
    return sample_tree(color_sampler, t, uv);
}

@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let a = forward(color, uv);
    let b = textureSampleLevel(layers, layer_sampler, uv, 2, 0.25);
    let c = textureSampleLevel(volume, volume_sampler, vec3<f32>(uv, 0.1), 1.5);
    return a + b + c;
}
"#;
        let (output, mask) = lower(source.to_owned());
        assert_eq!(mask, (1 << 62) | (1 << 3) | 1);
        assert!(output.contains("forward(color, uv, blender_webgpu_border_slot_62)"));
        assert!(output.contains("sample_tree(color_sampler, t, uv, blender_webgpu_border_arg_0)"));
        validate(&output);
    }

    #[test]
    fn preserves_raw_loads_and_comparison_sampling() {
        let source = r#"
@group(2) @binding(0) var depth: texture_depth_2d;
@group(2) @binding(1) var samp: sampler_comparison;
@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(textureSampleCompare(depth,samp,uv,0.3) + textureLoad(depth,vec2<i32>(0),0));
}
"#;
        let (result, mask) = lower(source.to_owned());
        assert_eq!(mask, 0);
        assert_eq!(result, source);
        validate(&result);
    }

    #[test]
    fn handles_integer_sampler_markers_forwarding_pointer_parameters() {
        let source = r#"
@group(2) @binding(42) var color: texture_2d_array<u32>;
fn blender_webgpu_integer_2d_array_u0028(t: texture_2d_array<u32>,
                                         uv: ptr<function, vec3<f32>>,
                                         lod: ptr<function, f32>) -> vec4<u32> {
    return textureLoad(t, vec2<i32>(0), i32((*uv).z), 0);
}
@compute @workgroup_size(1) fn main() {
    var uv = vec3<f32>(-0.25, 0, 1);
    var lod = 0.0;
    let sampled = blender_webgpu_integer_2d_array_u0028(color, &uv, &lod);
}
"#;
        let (result, mask) = lower(source.to_owned());
        assert_eq!(mask, 1 << 21);
        assert!(result.contains("blender_webgpu_integer_2d_array_u0028(color, &uv, &lod, blender_webgpu_border_slot_21)"));
        assert!(result.contains("textureLoad(t, vec2<i32>(0), i32((*uv).z), 0)"));
        validate(&result);
    }
}
