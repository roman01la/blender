// SPDX-License-Identifier: GPL-2.0-or-later

const F32_MAX_LITERAL: &str = "340282350000000000000000000000000000000f";
const BITCAST: &str = "bitcast<f32>(";
const RUNTIME_BITCAST: &str = "blender_wgsl_runtime_f32(";

pub(crate) fn lower(source: String) -> String {
    // A shortest-roundtrip Rust decimal can still be mathematically above
    // f32::MAX. WGSL then rejects it before rounding.
    let source = if source.contains(F32_MAX_LITERAL) {
        source.replace(F32_MAX_LITERAL, "bitcast<f32>(2139095039u)")
    } else {
        source
    };

    let mut remaining = source.as_str();
    let mut lowered = None;
    while let Some(start) = remaining.find(BITCAST) {
        let argument = start + BITCAST.len();
        let digits = remaining[argument..]
            .bytes()
            .take_while(u8::is_ascii_digit)
            .count();
        let end = argument + digits;
        if remaining[end..].starts_with("u)")
            && remaining[argument..end]
                .parse::<u32>()
                .is_ok_and(|bits| bits & 0x7f80_0000 == 0x7f80_0000)
        {
            // NaNs and infinities must be reconstructed at runtime, since
            // WGSL constant evaluation rejects non-finite bitcast results.
            let output = lowered.get_or_insert_with(|| {
                let prefix = source.len() - remaining.len();
                let mut output = String::with_capacity(source.len() + 112);
                output.push_str(&source[..prefix]);
                output
            });
            output.push_str(&remaining[..start]);
            output.push_str(RUNTIME_BITCAST);
        } else if let Some(output) = &mut lowered {
            output.push_str(&remaining[..argument]);
        }
        remaining = &remaining[argument..];
    }

    match lowered {
        Some(mut output) => {
            output.push_str(remaining);
            output.push_str(
                "\nfn blender_wgsl_runtime_f32(bits: u32) -> f32 { return bitcast<f32>(bits); }\n",
            );
            output
        }
        None => source,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_finite_and_non_finite_f32_bit_patterns() {
        let output = lower(format!(
            "a = -{F32_MAX_LITERAL}; b = bitcast<f32>(2143289344u); c = bitcast<f32>(1u); d = bitcast<f32>(4286578688u);"
        ));
        assert!(output.contains("a = -bitcast<f32>(2139095039u);"));
        assert!(output.contains("b = blender_wgsl_runtime_f32(2143289344u);"));
        assert!(output.contains("c = bitcast<f32>(1u);"));
        assert!(output.contains("d = blender_wgsl_runtime_f32(4286578688u);"));
    }
}
