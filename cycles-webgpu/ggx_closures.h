/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

using GgxFloat4 = float __attribute__((ext_vector_type(4)));

/* Cycles computes a minimum microfacet alpha from the smallest previous
 * sampling PDF. It blurs the closure after setting up the energy weights. */
inline float cycles_glossy_blur(const float filter_glossy, const float min_ray_pdf)
{
  if (filter_glossy == 0x1.fffffep127f) {
    return 0.0f;
  }
  const float blur_pdf = filter_glossy * min_ray_pdf;
  return blur_pdf < 1.0f ? __builtin_sqrtf(1.0f - blur_pdf) * 0.5f : 0.0f;
}

/* The view direction points away from the surface as in ShaderData::wi.
 * normal.w marks a back-facing hit. The result carries the BSDF PDF in w. */
GgxFloat4 cycles_ggx_eval(GgxFloat4 material,
                         GgxFloat4 tint_alpha,
                         GgxFloat4 weights,
                         GgxFloat4 glass,
                         GgxFloat4 normal,
                         GgxFloat4 specular_normal,
                         GgxFloat4 wi,
                         GgxFloat4 wo,
                         float blur_alpha,
                         float transparent_sample_weight);

/* The sample weight carries throughput in RGB and the BSDF PDF in w.
 * The outgoing vector carries the MIS-weighted roughness squared in w. */
unsigned int cycles_ggx_sample(GgxFloat4 material,
                               GgxFloat4 tint_alpha,
                               GgxFloat4 weights,
                               GgxFloat4 glass,
                               GgxFloat4 normal,
                               GgxFloat4 geometric_normal,
                               GgxFloat4 specular_normal,
                               GgxFloat4 wi,
                               GgxFloat4 random,
                               GgxFloat4 transparent,
                               float blur_alpha,
                               bool ray_differentials,
                               bool only_transparent,
                               GgxFloat4 &weight,
                               GgxFloat4 &wo);

GgxFloat4 cycles_standalone_eval(GgxFloat4 material,
                                GgxFloat4 parameters,
                                GgxFloat4 secondary,
                                GgxFloat4 extra,
                                GgxFloat4 hair,
                                GgxFloat4 normal,
                                GgxFloat4 geometric_normal,
                                GgxFloat4 wi,
                                GgxFloat4 wo,
                                float blur_alpha,
                                unsigned int &lcg_state);

unsigned int cycles_standalone_sample(GgxFloat4 material,
                                      GgxFloat4 parameters,
                                      GgxFloat4 secondary,
                                      GgxFloat4 extra,
                                      GgxFloat4 hair,
                                      GgxFloat4 normal,
                                      GgxFloat4 geometric_normal,
                                      GgxFloat4 wi,
                                      GgxFloat4 random,
                                      float blur_alpha,
                                      GgxFloat4 &weight,
                                      GgxFloat4 &wo);
