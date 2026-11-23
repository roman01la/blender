/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "../blender-platform/cycles-webgpu/material_layout.h"

using GgxFloat4 = float __attribute__((ext_vector_type(4)));

#ifdef CYCLES_WEBGPU_ORDERED_CLOSURES
__attribute__((pure)) GgxFloat4 cycles_material_lobe(unsigned int vector_index);
#endif

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

inline GgxFloat4 cycles_mixed_glossy_bump(const GgxFloat4 bump,
                                         const GgxFloat4 auxiliary,
                                         const unsigned int mixed_flags)
{
  /* Both bump directions use the shader's correction flag in bump.w. */
  if (mixed_flags & CYCLES_WEBGPU_MIXED_BUMP_INDEPENDENT) {
    return {auxiliary.x, auxiliary.y, auxiliary.z, bump.w};
  }
  return (mixed_flags & CYCLES_WEBGPU_MIXED_BUMP_DIFFUSE_ONLY) ? GgxFloat4{} : bump;
}

GgxFloat4 cycles_bump_specular_normal(GgxFloat4 bump,
                                      GgxFloat4 geometric,
                                      GgxFloat4 incident,
                                      GgxFloat4 original_specular);
float cycles_bump_shadowing(GgxFloat4 shading_normal,
                            GgxFloat4 bump,
                            GgxFloat4 direction,
                            bool is_eval);

/* The view direction points away from the surface as in ShaderData::wi.
 * normal.w marks a back-facing hit. The result carries the BSDF PDF in w. */
GgxFloat4 cycles_ggx_eval(GgxFloat4 material,
                         GgxFloat4 tint_alpha,
                         GgxFloat4 weights,
                         GgxFloat4 glass,
                         GgxFloat4 auxiliary,
                         GgxFloat4 normal,
                         GgxFloat4 specular_normal,
                         GgxFloat4 shader_specular_normal,
                         GgxFloat4 bump,
                         GgxFloat4 wi,
                         GgxFloat4 wo,
                         float blur_alpha,
                         bool light_passes,
                         GgxFloat4 &diffuse_ratio,
                         GgxFloat4 &glossy_ratio);

/* The sample weight carries throughput in RGB and the BSDF PDF in w.
 * The outgoing vector carries the MIS-weighted roughness squared in w. */
unsigned int cycles_ggx_sample(GgxFloat4 material,
                               GgxFloat4 tint_alpha,
                               GgxFloat4 weights,
                               GgxFloat4 glass,
                               GgxFloat4 normal,
                               GgxFloat4 geometric_normal,
                               GgxFloat4 specular_normal,
                               GgxFloat4 shader_specular_normal,
                               GgxFloat4 bump,
                               GgxFloat4 wi,
                               GgxFloat4 random,
                               GgxFloat4 auxiliary,
                               float blur_alpha,
                               bool ray_differentials,
                               bool only_transparent,
                               GgxFloat4 &weight,
                               GgxFloat4 &wo,
                               bool light_passes,
                               GgxFloat4 &diffuse_ratio,
                               GgxFloat4 &glossy_ratio);

GgxFloat4 cycles_standalone_eval(GgxFloat4 material,
                                GgxFloat4 parameters,
                                GgxFloat4 secondary,
                                GgxFloat4 extra,
                                GgxFloat4 hair,
                                GgxFloat4 ordered_layers,
                                GgxFloat4 normal,
                                GgxFloat4 geometric_normal,
                                GgxFloat4 wi,
                                GgxFloat4 wo,
                                float blur_alpha,
                                unsigned int &lcg_state,
                                GgxFloat4 &diffuse_ratio);

unsigned int cycles_standalone_sample(GgxFloat4 material,
                                      GgxFloat4 parameters,
                                      GgxFloat4 secondary,
                                      GgxFloat4 extra,
                                      GgxFloat4 hair,
                                      GgxFloat4 ordered_layers,
                                      GgxFloat4 normal,
                                      GgxFloat4 geometric_normal,
                                      GgxFloat4 wi,
                                      GgxFloat4 random,
                                      float blur_alpha,
                                      GgxFloat4 &weight,
                                      GgxFloat4 &wo,
                                      GgxFloat4 &diffuse_ratio);
