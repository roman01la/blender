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
    fields = fields.replace("  Spectrum transparent_albedo = zero_spectrum();\n", "")
    fields = fields.replace("  float specular_roughness = 0.0f;\n", "")
    accumulate = section("    const float closure_weight", "\n  }\n\n  /* Fraction")
    # The compact renderer has neither hair nor transparent closures.
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
    # other denoising data passes and transparent closures are outside its film layout.
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
    const float opaque_fraction = sum_weight > 0.0f ? 1.0f : 0.0f;
    guide_normal = normal * opaque_fraction * feature_weight * average(denoising_feature_throughput);
    const Spectrum deferred_albedo = follow_reflections ?
                                        specular_albedo * deferred_feature_weight : zero_spectrum();
    throughput = reduce_max(fabs(deferred_albedo)) > 1e-4f ?
                     throughput * deferred_albedo : zero_spectrum();
  }
};
"""
    header.write_text(text)
    return work
