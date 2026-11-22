# SPDX-License-Identifier: Apache-2.0

"""Expose Cycles' surface denoising accumulation to the compact shader adapter."""

from pathlib import Path
from textwrap import dedent, indent


def prepare_denoising_header(work: Path, cycles: Path) -> Path:
    source = (cycles / "kernel/film/denoising_passes.h").read_text()
    source = source.split("void film_write_denoising_features_surface(", 1)[1]

    def section(start: str, end: str) -> str:
        return source[source.index(start):source.index(end, source.index(start))]

    fields = section("  float3 normal = ", "  for (int i")
    fields = fields.replace("  float specular_roughness = 0.0f;\n", "")
    accumulate = section("    const float closure_weight", "\n  }\n\n  /* Fraction")
    # The compact renderer uses a world-space normal for each opaque closure.
    accumulate = (
        accumulate.split("    /* All closures", 1)[0] +
        "    normal += N * closure_weight;\n" +
        accumulate.split("              closure_weight;\n", 1)[1]
    )
    roughness_begin = accumulate.index("    /* bsdf_get_specular_roughness_squared")
    roughness_end = accumulate.index("    if (use_albedo_roughness_weighting)")
    accumulate = (accumulate[:roughness_begin] +
                  "    const float diffuse_weight = smoothstep(0.0f, 0.15f, roughness);\n\n" +
                  accumulate[roughness_end:])
    # Keep both accumulators as SSA values. Folding the stock branch into a
    # pointer select would require VariablePointers, which WebGPU lacks.
    accumulate = (
        accumulate[:accumulate.index("    if (use_albedo_roughness_weighting)")] +
        "    const bool diffuse = CLOSURE_IS_BSDF_DIFFUSE(type) || CLOSURE_IS_BSSRDF(type);\n"
        "    const bool specular = CLOSURE_IS_BSDF_GLOSSY(type) || CLOSURE_IS_GLASS(type);\n"
        "    diffuse_albedo += albedo * (use_albedo_roughness_weighting ? diffuse_weight : "
        "float(diffuse));\n"
        "    specular_albedo += albedo * (use_albedo_roughness_weighting ? 1.0f - diffuse_weight : "
        "float(specular));\n"
        + accumulate[accumulate.index("    sum_weight +="):]
    )
    finish = section("  float feature_weight = ", "  const bool is_first_bounce")
    finish = finish.replace("    specular_roughness /= sum_weight;\n", "")
    finish = finish.replace("kernel_data.film.denoising_pass_options_flag", "options")
    # The caller consumes the RGB albedo, normal and deferred throughput. The
    # other denoising data passes are outside its film layout.
    header = work / "denoising_features.h"
    text = """/* SPDX-License-Identifier: Apache-2.0 */
/* Derived during the shader build from Cycles' kernel/film/denoising_passes.h. */
#pragma once

struct DenoisingFeatureAccumulator {
"""
    text += fields
    text += """  void add(const Spectrum albedo,
           const float3 N,
           const float roughness,
           const ClosureType type,
           const bool use_albedo_roughness_weighting)
  {
"""
    text += indent(dedent(accumulate), "    ") + "\n  }\n\n"
    text += """  void finish(const uint options, Spectrum &throughput, Spectrum &albedo, float3 &guide_normal)
  {
    const Spectrum denoising_feature_throughput = throughput;
"""
    text += indent(dedent(finish), "    ")
    text += """    albedo = ensure_finite(diffuse_albedo * feature_weight * denoising_feature_throughput);
    const float total_weight = sum_weight + average(transparent_albedo);
    const float opaque_fraction = total_weight > 0.0f ? sum_weight / total_weight : 0.0f;
    guide_normal = normal * opaque_fraction * feature_weight * average(denoising_feature_throughput);
    const Spectrum deferred_albedo = transparent_albedo + (follow_reflections ?
                                        specular_albedo * deferred_feature_weight : zero_spectrum());
    throughput = reduce_max(fabs(deferred_albedo)) > 1e-4f ?
                     throughput * deferred_albedo : zero_spectrum();
  }
};
"""
    header.write_text(text)
    return work


def prepare_volume_denoising_header(work: Path, cycles: Path) -> None:
    source = (cycles / "kernel/film/denoising_passes.h").read_text()
    start = source.index("void film_write_denoising_features_volume(")
    first = source.index("  if (scatter &&", start)
    policy = source[first:source.index("\n}\n", first)]
    # Keep the stock albedo multiplication, camera-facing normal and feature
    # termination. Redirect film writes to the path retired by the GPU film.
    replacements = {
        "kernel_data.film.pass_denoising_normal != PASS_UNUSED": "write_normal",
        "kernel_data.film.pass_denoising_albedo != PASS_UNUSED": "write_albedo",
        "INTEGRATOR_STATE_WRITE(state, path, flag) &= ~PATH_RAY_DENOISING_FEATURES;":
            "feature_throughput = 0.0f;",
        "make_float3(0.0f, 0.0f, -1.0f)": "float3(0.0f, 0.0f, -1.0f)",
        "film_write_pass_float3(buffer + kernel_data.film.pass_denoising_normal, denoising_normal);":
            "normal_sum.w += denoising_normal.z;",
        "const Spectrum denoising_albedo = ensure_finite(denoising_feature_throughput * albedo);":
            "const float3 denoising_albedo = guide_finite(denoising_feature_throughput * albedo);",
        "film_write_pass_spectrum(buffer + kernel_data.film.pass_denoising_albedo, denoising_albedo);":
            "albedo_sum.xyz += denoising_albedo;",
    }
    for original, replacement in replacements.items():
        if policy.count(original) != 1:
            raise ValueError(f"Cannot adapt stock volume denoising feature: {original}")
        policy = policy.replace(original, replacement)
    (work / "volume-denoising.hlsl").write_text("""/* Generated from Cycles' denoising_passes.h. */
float3 guide_finite(float3 rgb)
{
  const uint3 exponent = __builtin_bit_cast(uint3, rgb) & 0x7f800000u;
  return float3(exponent.x != 0x7f800000u ? rgb.x : 0.0f,
                exponent.y != 0x7f800000u ? rgb.y : 0.0f,
                exponent.z != 0x7f800000u ? rgb.z : 0.0f);
}

void accumulate_volume_features(float3 albedo, bool scatter, bool write_albedo,
                                bool write_normal, inout float4 feature_throughput,
                                inout float4 albedo_sum, inout float4 normal_sum)
{
  const float3 denoising_feature_throughput = feature_throughput.xyz;
""" + policy + "\n}\n")
