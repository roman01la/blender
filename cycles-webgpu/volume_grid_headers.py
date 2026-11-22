# SPDX-License-Identifier: Apache-2.0

"""Expose stock NanoVDB filtering and null-collision transport for packed grids."""

import re


def prepare_volume_grid_headers(work, cycles):
    def section(path, begin, end):
        source = (cycles / "kernel" / path).read_text()
        first = source.index(begin)
        return source[first:source.index(end, first)]

    filters = section("util/image_3d.h", "/* Cubic interpolation weights. */",
                      "template<typename OutT, typename T>")
    value = section("geom/volume.h", "template<typename T> ccl_device_inline T volume_attribute_value",
                    "ccl_device_template_spec float4 volume_attribute_value")
    (work / "volume_grid_filters.h").write_text(
        "CCL_NAMESPACE_BEGIN\n" + filters + value + "CCL_NAMESPACE_END\n")

    shade = "integrator/shade_volume.h"
    transmittance = section(shade, "template<const bool shadow, typename IntegratorGenericState>\n"
                            "ccl_device Spectrum volume_transmittance(",
                            "/* Compute the volumetric transmittance")
    # Pass in the same two uniform draws. Shadow paths don't need the renderer's
    # sampling table just to select the estimator order and the step offset.
    transmittance = transmittance.replace(
        "template<const bool shadow, typename IntegratorGenericState>\n", "")
    transmittance = transmittance.replace("volume_transmittance(KernelGlobals kg,\n"
                                         "                                         IntegratorGenericState state,",
                                         "volume_grid_transmittance(VolumeGlobals *kg,\n"
                                         "                                              const uint active,")
    transmittance = transmittance.replace("const ccl_private RNGState *rng_state,\n",
                                         "const float rand,\n"
                                         "                                         const float shade_offset,\n")
    transmittance = transmittance.replace("  const float rand = path_rng_1D(\n"
                                         "      kg, 0, rng_state->sample, "
                                         "rng_state->rng_offset + PRNG_VOLUME_EXPANSION_ORDER);\n", "")
    transmittance = transmittance.replace("  const float shade_offset = "
                                         "path_state_rng_1D(kg, rng_state, PRNG_VOLUME_SHADE_OFFSET);\n", "")
    transmittance = transmittance.replace("volume_shader_eval_extinction<shadow>(",
                                         "volume_grid_eval_extinction(").replace(
        "kg, state, sd, path_visibility, path_flag", "kg, active, sd, path_visibility, path_flag")
    # Use named parity bins so LLVM keeps these sums in registers instead of
    # lowering a dynamic private array to SPIR-V byte-addressed memset.
    transmittance = transmittance.replace("  Spectrum tau_j[2];\n"
                                         "  tau_j[0] = zero_spectrum();\n"
                                         "  tau_j[1] = zero_spectrum();",
                                         "  Spectrum tau_even = zero_spectrum();\n"
                                         "  Spectrum tau_odd = zero_spectrum();")
    transmittance = transmittance.replace("      tau_j[step % 2] += tau * 2.0f;",
                                         "      if ((step & 1) == 0) {\n"
                                         "        tau_even += tau * 2.0f;\n"
                                         "      }\n"
                                         "      else {\n"
                                         "        tau_odd += tau * 2.0f;\n"
                                         "      }")
    transmittance = transmittance.replace("exp(-tau_j[0]) + exp(-tau_j[1])",
                                         "exp(-tau_even) + exp(-tau_odd)")
    (work / "volume_grid_transmittance.h").write_text(transmittance)

    # The stock octree driver can advance across many different maxima. The
    # WebGPU stack driver supplies one conservative maximum per bounded span.
    advances = section(shade, "/* Accumulate transmittance for equiangular",
                       "/** Advance to the next candidate indirect scatter position,").replace(
        "OctreeTracing", "WebGPUVolumeSegment")
    advances = advances.replace("KernelGlobals kg,\n", "VolumeGlobals *kg,\n    const uint active,\n")
    advances = advances.replace("volume_transmittance<false>(\n"
                                "      kg, state, ray, sd, sigma.range(), t, rng_state, "
                                "path_visibility, path_flag);",
                                "volume_grid_transmittance(\n"
                                "      kg, active, ray, sd, sigma.range(), t,\n"
                                "      path_rng_1D(kg, 0, rng_state->sample, "
                                "rng_state->rng_offset + PRNG_VOLUME_EXPANSION_ORDER),\n"
                                "      path_state_rng_1D(kg, rng_state, PRNG_VOLUME_SHADE_OFFSET),\n"
                                "      path_visibility, path_flag);")

    reservoir = section(shade, "/* Candidate scatter position for VSPG. */",
                        "/* Estimate volume majorant optical depth")
    reservoir = reservoir.replace("  VolumeSampleCandidate candidate;",
                                  "  VolumeSampleCandidate candidate{};")
    finalize = section(shade, "ccl_device_inline void volume_distance_sampling_finalize(",
                       "  const uint lcg_state = reservoir.candidate.lcg_state;")
    finalize = re.sub(r"    (?:KernelGlobals kg|const ccl_private Ray \*ccl_restrict ray|"
                      r"ccl_private ShaderData \*ccl_restrict sd),\n", "", finalize)
    finalize += "}\n\n"
    steps = section(shade, "ccl_device bool volume_integrate_should_stop(",
                    "/* Multiple Importance Sampling between equiangular")
    steps = steps.replace("KernelGlobals kg,\n", "VolumeGlobals *kg,\n    const uint active,\n")
    steps = steps.replace("volume_shader_sample(kg, state, sd, &coeff)",
                          "volume_grid_shader_sample(kg, active, state, sd, &coeff)")
    steps = steps.replace("      guiding_record_volume_emission(kg, state, coeff.emission);\n", "")
    steps = steps.replace("    if (!result.indirect_scatter) {\n"
                          "      /* Record emission until scatter position. */\n"
                          "    }\n", "")
    # The adapter restores the stock LCG seed and copies phases when it shades
    # each selected direct or indirect point.
    steps = re.sub(r" +volume_shader_copy_phases\(&result\.(?:direct|indirect)_phases, sd\);\n",
                   "", steps)

    (work / "volume_grid_steps.h").write_text(advances + reservoir + finalize + steps)
