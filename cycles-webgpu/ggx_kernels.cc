/* SPDX-License-Identifier: Apache-2.0 */

#include "ggx_closures.h"

#include <type_traits>
#include "../blender-platform/cycles-webgpu/material_layout.h"

#ifdef __aarch64__
#  include <arm_neon.h>
#endif

#define CCL_NAMESPACE_BEGIN namespace ccl {
#define CCL_NAMESPACE_END }
#include "kernel/constants.h"
#include "kernel/types.h"

#define __KERNEL_GPU__
#include "kernel/globals.h"
#undef __KERNEL_GPU__

__attribute__((pure)) float cycles_ggx_lookup(unsigned int index);

namespace ccl {
struct GgxGlobals {
  KernelData data;
};
using KernelGlobals = const GgxGlobals *;
}  // namespace ccl

#define kernel_data (kg->data)
#define kernel_data_fetch(name, index) cycles_ggx_lookup(index)
#undef ccl_device
#define ccl_device static inline __attribute__((always_inline))
#include "kernel/closure/bsdf_diffuse.h"
#include "kernel/closure/bsdf_transparent.h"
#include "kernel/closure/bsdf_microfacet.h"
#include "kernel/closure/bsdf_ashikhmin_shirley.h"
#include "kernel/closure/bsdf_ashikhmin_velvet.h"
#include "kernel/closure/bsdf_sheen.h"
#include "kernel/closure/bsdf_toon.h"
#include "cosine_hemisphere.h"
#include "bump_shadowing.h"
#if !defined(CYCLES_WEBGPU_EXCLUDE_STANDALONE) && !defined(CYCLES_WEBGPU_EXCLUDE_HAIR)
#  include "kernel/closure/bsdf_hair.h"
#  include "kernel/closure/bsdf_principled_hair_huang.h"
#endif

namespace ccl {
namespace {

#ifdef CYCLES_WEBGPU_DENOISE_SURFACE
#  include "denoising_features.h"
#endif

constexpr int schlick_table_offset = GGX_GEN_SCHLICK_IOR_S_RES_ROUGH *
                                     GGX_GEN_SCHLICK_IOR_S_RES_MU *
                                     GGX_GEN_SCHLICK_IOR_S_RES_IOR;
constexpr int energy_table_offset = schlick_table_offset +
                                    GGX_GEN_SCHLICK_S_RES_ROUGH * GGX_GEN_SCHLICK_S_RES_MU *
                                        GGX_GEN_SCHLICK_S_RES_IOR;
constexpr int average_energy_table_offset = energy_table_offset + GGX_E_RES_ROUGH * GGX_E_RES_MU;
constexpr int glass_energy_table_size = GGX_GLASS_E_RES_ROUGH * GGX_GLASS_E_RES_MU *
                                        GGX_GLASS_E_RES_IOR;
constexpr int glass_average_table_size = GGX_GLASS_E_RES_ROUGH * GGX_GLASS_E_RES_IOR;
constexpr int glass_energy_table_offset = average_energy_table_offset + GGX_E_RES_ROUGH;
constexpr int glass_average_energy_table_offset = glass_energy_table_offset +
                                                  glass_energy_table_size;
constexpr int inverse_glass_energy_table_offset = glass_average_energy_table_offset +
                                                  glass_average_table_size;
constexpr int inverse_glass_average_energy_table_offset = inverse_glass_energy_table_offset +
                                                          glass_energy_table_size;
static_assert(schlick_table_offset == CYCLES_WEBGPU_GGX_TABLE_WORDS);
static_assert(energy_table_offset == 2 * CYCLES_WEBGPU_GGX_TABLE_WORDS);
static_assert(GGX_E_RES_ROUGH * GGX_E_RES_MU == CYCLES_WEBGPU_GGX_ENERGY_WORDS);
static_assert(GGX_E_RES_ROUGH == CYCLES_WEBGPU_GGX_AVERAGE_ENERGY_WORDS);
static_assert(glass_energy_table_offset == CYCLES_WEBGPU_GGX_LOOKUP_WORDS);
static_assert(glass_energy_table_size == CYCLES_WEBGPU_GGX_GLASS_ENERGY_WORDS);
static_assert(glass_average_table_size == CYCLES_WEBGPU_GGX_GLASS_AVERAGE_ENERGY_WORDS);

constexpr GgxGlobals energy_globals = [] {
  GgxGlobals globals{};
  globals.data.tables.ggx_E = energy_table_offset;
  globals.data.tables.ggx_Eavg = average_energy_table_offset;
  globals.data.tables.ggx_glass_E = glass_energy_table_offset;
  globals.data.tables.ggx_glass_Eavg = glass_average_energy_table_offset;
  globals.data.tables.ggx_glass_inv_E = inverse_glass_energy_table_offset;
  globals.data.tables.ggx_glass_inv_Eavg = inverse_glass_average_energy_table_offset;
  globals.data.tables.ggx_gen_schlick_ior_s = 0;
  globals.data.tables.ggx_gen_schlick_s = schlick_table_offset;
  return globals;
}();

enum Lobe { CoatReflection, Metal, Glass, Specular, Subsurface, Diffuse, Translucent };

#ifdef CYCLES_WEBGPU_REFLECTION_ONLY
constexpr bool glass_enabled = false;
#else
constexpr bool glass_enabled = true;
#endif

struct ReflectiveClosures {
  Spectrum metal_weight;
  Spectrum specular_weight;
  Spectrum diffuse_weight;
  Spectrum subsurface_weight;
  Spectrum metal_f0;
  Spectrum metal_tint_b;
  Spectrum specular_f0;
  float metal_sample_weight;
  float specular_sample_weight;
  float diffuse_sample_weight;
  float subsurface_sample_weight;
  float alpha_x;
  float alpha_y;
  float3 tangent;
  float eta;
  float metal_energy_scale;
  float specular_energy_scale;
  bool has_metal;
  bool has_specular;
  bool has_diffuse;
};

ccl_device float3 rgb(const GgxFloat4 v)
{
  return make_float3(v.x, v.y, v.z);
}

ccl_device GgxFloat4 vector(const float3 v)
{
  return {v.x, v.y, v.z, 0.0f};
}

#if !defined(CYCLES_WEBGPU_EXCLUDE_STANDALONE) && !defined(CYCLES_WEBGPU_EXCLUDE_HAIR)
#  include "hair_closures.h"
#endif

/* A negative transmission model packs the Principled anisotropic aspect.
 * Both axes are clamped before the roughness blur, as in the stock setup. */
ccl_device float principled_alpha_x(const float alpha, const float model)
{
  return model < 0.0f ? saturatef(alpha / -model) : alpha;
}

ccl_device float principled_alpha_y(const float alpha, const float model)
{
  return model < 0.0f ? saturatef(alpha * -model) : alpha;
}

ccl_device bool setup_white_coat(MicrofacetBsdf &bsdf,
                                  const float mixed_weight,
                                  const float ior,
                                  const float alpha,
                                  const float3 N,
                                  const float3 wi,
                                  const float blur_alpha,
                                  float &base_mix)
{
  Spectrum weight = make_spectrum(mixed_weight);
  const float sample_weight = closure_sample_weight(0, weight);
  if (!(sample_weight > 0.0f)) {
    return false;
  }

  /* White Coat has no refraction tint. Use the stock reflection, dielectric
   * energy correction and lower-layer attenuation before applying blur. */
  bsdf.weight = weight;
  bsdf.sample_weight = sample_weight;
  bsdf.N = N;
  bsdf.T = zero_float3();
  bsdf.alpha_x = bsdf.alpha_y = alpha;
  bsdf.ior = ior;
  bsdf_microfacet_ggx_setup(&bsdf);
  bsdf_microfacet_setup_fresnel_dielectric(&energy_globals, &bsdf, wi);
  const Spectrum albedo = bsdf_microfacet_estimate_albedo(&energy_globals, wi, &bsdf, true, false);
  base_mix = closure_layering_weight(albedo * weight, make_spectrum(base_mix)).x;
  bsdf_microfacet_blur(reinterpret_cast<ShaderClosure *>(&bsdf), blur_alpha);
  return true;
}

ccl_device bool setup_compact_white_coat(MicrofacetBsdf &bsdf,
                                          GgxFloat4 &tint_alpha,
                                          GgxFloat4 &weights,
                                          const GgxFloat4 glass,
                                          const float3 N,
                                          const float3 shader_N,
                                          const float3 wi,
                                          const float blur_alpha)
{
  if (!(glass.x < 0.0f) || CYCLES_WEBGPU_GGX_HAS_TRANSLUCENT(glass)) {
    return false;
  }
  const float signed_ior = tint_alpha.x;
  const float alpha = tint_alpha.y;
  tint_alpha = {1.0f, 1.0f, 1.0f, tint_alpha.w};
  float mix = weights.w;
  const float3 coat_N = signed_ior < 0.0f ? shader_N : N;
  const bool coat = setup_white_coat(
      bsdf, -glass.x, fabsf(signed_ior), alpha, coat_N, wi, blur_alpha, mix);
  weights.w = mix;
  return coat;
}

ccl_device bool setup_glass(MicrofacetBsdf &bsdf,
                            FresnelGeneralizedSchlick &fresnel,
                            const GgxFloat4 material,
                            const GgxFloat4 tint_alpha,
                            const GgxFloat4 weights,
                            const GgxFloat4 glass,
                            const GgxFloat4 auxiliary,
                            const GgxFloat4 normal,
                            const float3 wi)
{
  if (!(glass.x > CLOSURE_WEIGHT_CUTOFF)) {
    return false;
  }

  const bool refraction = glass.z == CYCLES_WEBGPU_REFRACTION_GGX ||
                          glass.z == CYCLES_WEBGPU_REFRACTION_BECKMANN;
  const Spectrum color = rgb(material);
  bsdf.weight = make_spectrum(glass.x * weights.w);
  if (refraction) {
    bsdf.weight *= color;
  }
  bsdf.sample_weight = closure_sample_weight(0, bsdf.weight);
  if (!(bsdf.sample_weight > 0.0f)) {
    return false;
  }

  bsdf.N = rgb(normal);
  bsdf.alpha_x = principled_alpha_x(tint_alpha.w, glass.z);
  bsdf.alpha_y = principled_alpha_y(tint_alpha.w, glass.z);
  bsdf.ior = normal.w != 0.0f ? 1.0f / glass.y : glass.y;
  bsdf.T = glass.z < 0.0f ? rgb(auxiliary) : zero_float3();
  if (refraction) {
    if (glass.z == CYCLES_WEBGPU_REFRACTION_BECKMANN) {
      bsdf_microfacet_beckmann_refraction_setup(&bsdf);
    }
    else {
      bsdf_microfacet_ggx_refraction_setup(&bsdf);
    }
    return true;
  }
  if (glass.z > 0.0f) {
    fresnel = {.thin_film = {},
               .tint = {color, color},
               .f0 = make_spectrum(F0_from_ior(glass.y)),
               .f90 = one_spectrum(),
               .exponent = -glass.y};
  }
  else {
    fresnel = generalized_schlick_setup(
        glass.y, true, true, rgb(tint_alpha), sqrt(min(color, one_spectrum())), {});
  }
  if (glass.z == CYCLES_WEBGPU_GLASS_BECKMANN) {
    bsdf_microfacet_beckmann_glass_setup(&bsdf);
  }
  else {
    bsdf_microfacet_ggx_glass_setup(&bsdf);
  }
  bsdf_microfacet_setup_fresnel_generalized_schlick(
      &energy_globals, &bsdf, wi, &fresnel, material.w == CYCLES_WEBGPU_MATERIAL_MULTISCATTER_GGX);
  return true;
}

ccl_device bool is_beckmann_transmission(const ClosureType type)
{
  return type == CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID ||
         type == CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID;
}

ccl_device Spectrum accumulate_microfacet(const MicrofacetBsdf &bsdf,
                                 const float3 wi,
                                 const float3 wo,
                                 Spectrum &eval,
                                 float &pdf)
{
  float lobe_pdf = 0.0f;
  const auto *closure = reinterpret_cast<const ShaderClosure *>(&bsdf);
  const Spectrum lobe_eval =
      is_beckmann_transmission(bsdf.type) ?
          bsdf_microfacet_beckmann_eval(&energy_globals, closure, wi, wo, &lobe_pdf) :
          bsdf_microfacet_ggx_eval(&energy_globals, closure, wi, wo, &lobe_pdf);
  if (lobe_pdf != 0.0f) {
    const Spectrum weighted = lobe_eval * bsdf.weight;
    eval += weighted;
    pdf += lobe_pdf * bsdf.sample_weight;
    return weighted;
  }
  return zero_spectrum();
}

/* Match the non-diffuse evaluation check in bump_shadowing_term.
 * The corrected specular normal may differ from ShaderData::N. */
ccl_device bool valid_specular_eval(const float3 N, const float3 specular_N, const float3 wo)
{
  return dot(N, wo) * dot(N, specular_N) * dot(specular_N, wo) >= 0.0f;
}

/* Keep the adjusted lobe weight separate from energy_scale for closure layering. */
ccl_device float preserve_reflective_energy(const float3 N,
                                             const float3 wi,
                                             const float alpha_x,
                                             const float alpha_y,
                                             const Spectrum Fss,
                                             Spectrum &weight,
                                             float &sample_weight)
{
  MicrofacetBsdf bsdf{};
  bsdf.type = CLOSURE_BSDF_MICROFACET_GGX_ID;
  bsdf.N = N;
  bsdf.alpha_x = alpha_x;
  bsdf.alpha_y = alpha_y;
  bsdf.weight = weight;
  bsdf.sample_weight = sample_weight;
  microfacet_ggx_preserve_energy(&energy_globals, &bsdf, wi, Fss);
  weight = bsdf.weight;
  sample_weight = bsdf.sample_weight;
  return bsdf.energy_scale;
}

ccl_device void setup_closures(ReflectiveClosures &closures,
                               const GgxFloat4 material,
                               const GgxFloat4 tint_alpha,
                               const GgxFloat4 weights,
                               const GgxFloat4 glass,
                               const GgxFloat4 auxiliary,
                               const float3 N,
                               const float3 wi)
{
  const Spectrum base_color = rgb(material);
  const Spectrum tint = rgb(tint_alpha);
  const float metallic = weights.x;
  const float eta = weights.y;
  const float alpha_x = principled_alpha_x(tint_alpha.w, glass.z);
  const float alpha_y = principled_alpha_y(tint_alpha.w, glass.z);
  const float rough = alpha_to_roughness_index(alpha_x, alpha_y);
  const float y = mu_to_y_index(dot(N, wi));
  const Spectrum white = one_spectrum();
  const bool multiscatter = material.w == CYCLES_WEBGPU_MATERIAL_MULTISCATTER_GGX;
  Spectrum weight = make_spectrum(weights.w);
  closures.alpha_x = alpha_x;
  closures.alpha_y = alpha_y;
  closures.tangent = glass.z < 0.0f ? rgb(auxiliary) : zero_float3();
  closures.eta = eta;
  closures.metal_energy_scale = 1.0f;
  closures.specular_energy_scale = 1.0f;
  closures.metal_sample_weight = 0.0f;
  closures.specular_sample_weight = 0.0f;
  closures.has_metal = false;
  closures.has_specular = false;

  /* Use the same F82 and generalized Schlick estimates as
   * bsdf_microfacet_estimate_albedo(). They determine the lobe probabilities
   * and the diffuse contribution below dielectric reflection. Keeping the
   * coefficients as values avoids closure-storage pointer casts in shaders. */
  if (metallic > CLOSURE_WEIGHT_CUTOFF) {
    closures.metal_weight = metallic * weight;
    const float sample_weight = closure_sample_weight(0, closures.metal_weight);
    closures.has_metal = sample_weight > 0.0f;
    if (closures.has_metal) {
      closures.metal_f0 = min(base_color, white);
      const Spectrum f82_tint = min(tint, white);
      closures.metal_tint_b = isequal(f82_tint, white) ?
                                 zero_spectrum() :
                                 fresnel_f82tint_B(closures.metal_f0, f82_tint);
      const float s = lookup_table_read_3D(nullptr,
                                          rough,
                                          y,
                                          0.5f,
                                          schlick_table_offset,
                                          GGX_GEN_SCHLICK_S_RES_ROUGH,
                                          GGX_GEN_SCHLICK_S_RES_MU,
                                          GGX_GEN_SCHLICK_S_RES_IOR);
      closures.metal_sample_weight = sample_weight * average(mix(closures.metal_f0, white, s));
      if (multiscatter) {
        const Spectrum Fss = fresnel_f82_Fss(closures.metal_f0, closures.metal_tint_b);
        closures.metal_energy_scale = preserve_reflective_energy(
            N, wi, alpha_x, alpha_y, Fss, closures.metal_weight, closures.metal_sample_weight);
      }
    }
    weight *= 1.0f - metallic;
  }

  if (eta != 1.0f) {
    closures.specular_weight = weight;
    const float sample_weight = closure_sample_weight(0, closures.specular_weight);
    closures.has_specular = sample_weight > 0.0f;
    if (closures.has_specular) {
      closures.specular_f0 = saturate(weights.z * tint);
      const float s = lookup_table_read_3D(nullptr,
                                          rough,
                                          y,
                                          ior_to_z_index(eta),
                                          0,
                                          GGX_GEN_SCHLICK_IOR_S_RES_ROUGH,
                                          GGX_GEN_SCHLICK_IOR_S_RES_MU,
                                          GGX_GEN_SCHLICK_IOR_S_RES_IOR);
      const Spectrum albedo = mix(closures.specular_f0, white, s);
      closures.specular_sample_weight = sample_weight * average(albedo);
      if (multiscatter) {
        const float F0 = F0_from_ior(eta);
        const float average_fresnel = fresnel_dielectric_Fss(eta);
        const float f = saturatef(inverse_lerp(F0, 1.0f, average_fresnel));
        const Spectrum Fss = mix(closures.specular_f0, white, f);
        closures.specular_energy_scale = preserve_reflective_energy(
            N, wi, alpha_x, alpha_y, Fss, closures.specular_weight, closures.specular_sample_weight);
      }
      weight = closure_layering_weight(closures.specular_weight * reduce_max(albedo), weight);
    }
  }

  /* bssrdf_alloc and bssrdf_setup with all RGB radii active. */
  const float subsurface = glass.w;
  closures.subsurface_weight = saturate(base_color) * subsurface * weight;
  const float bssrdf_weight = fabsf(average(closures.subsurface_weight));
  closures.subsurface_sample_weight = bssrdf_weight >= CLOSURE_WEIGHT_CUTOFF ?
                                         bssrdf_weight * SPECTRUM_CHANNELS : 0.0f;
  closures.diffuse_weight = base_color * (1.0f - subsurface) * weight;
  closures.diffuse_sample_weight = closure_sample_weight(0, closures.diffuse_weight);
  closures.has_diffuse = closures.diffuse_sample_weight > 0.0f;
}

template<Lobe lobe>
ccl_device Spectrum reflection_fresnel(const ReflectiveClosures &closures, const float cos_HI)
{
  if constexpr (lobe == Metal) {
    return fresnel_f82(cos_HI, closures.metal_f0, closures.metal_tint_b);
  }
  else {
    /* Principled's generalized Schlick uses negative exponent -eta to remap
     * the real dielectric Fresnel curve between its RGB F0 and white. */
    const float f = fresnel_dielectric(cos_HI, closures.eta, nullptr);
    const float f0 = F0_from_ior(closures.eta);
    const float s = saturatef(inverse_lerp(f0, 1.0f, f));
    return mix(closures.specular_f0, one_spectrum(), s);
  }
}

ccl_device void reflective_ggx_basis(const ReflectiveClosures &closures,
                                     const float3 N,
                                     float3 &X,
                                     float3 &Y)
{
  if (closures.alpha_x == closures.alpha_y) {
    make_orthonormals(N, &X, &Y);
  }
  else {
    make_orthonormals_tangent(N, closures.tangent, &X, &Y);
  }
}

ccl_device void blur_reflective_ggx(ReflectiveClosures &closures, const float alpha)
{
  closures.alpha_x = fmaxf(closures.alpha_x, alpha);
  closures.alpha_y = fmaxf(closures.alpha_y, alpha);
}

ccl_device void eval_ggx(const ReflectiveClosures &closures,
                        const float3 N,
                        const float3 wi,
                        const float3 wo,
                        const float3 H,
                        const float cos_NH,
                        const Spectrum fresnel,
                        Spectrum &eval,
                        float &pdf)
{
  if (is_zero(fresnel)) {
    return;
  }
  /* Pure reflection specialization of bsdf_microfacet_eval<GGX>(). */
  const float cos_NI = dot(N, wi);
  const float cos_NO = dot(N, wo);
  const float alpha_x = closures.alpha_x;
  const float alpha_y = closures.alpha_y;
  float D, lambdaI, lambdaO;
  if (alpha_x == alpha_y) {
    const float alpha2 = alpha_x * alpha_y;
    D = bsdf_D<MicrofacetType::GGX>(alpha2, cos_NH);
    lambdaI = bsdf_lambda<MicrofacetType::GGX>(alpha2, cos_NI);
    lambdaO = bsdf_lambda<MicrofacetType::GGX>(alpha2, cos_NO);
  }
  else {
    float3 X, Y;
    reflective_ggx_basis(closures, N, X, Y);
    const float3 local_H = make_float3(dot(X, H), dot(Y, H), cos_NH);
    const float3 local_I = make_float3(dot(X, wi), dot(Y, wi), cos_NI);
    const float3 local_O = make_float3(dot(X, wo), dot(Y, wo), cos_NO);
    D = bsdf_aniso_D<MicrofacetType::GGX>(alpha_x, alpha_y, local_H);
    lambdaI = bsdf_aniso_lambda<MicrofacetType::GGX>(alpha_x, alpha_y, local_I);
    lambdaO = bsdf_aniso_lambda<MicrofacetType::GGX>(alpha_x, alpha_y, local_O);
  }
  const float common = D / cos_NI * 0.25f;
  pdf = common / (1.0f + lambdaI);
  eval = fresnel * common / (1.0f + lambdaO + lambdaI);
}

template<Lobe lobe>
ccl_device void accumulate_lobe(const ReflectiveClosures &closures,
                                 const float3 N,
                                 const float3 wi,
                                 const float3 wo,
                                 Spectrum &sum_eval,
                                 float &sum_pdf)
{
  const float cos_NI = dot(N, wi);
  const float cos_NO = dot(N, wo);
  if constexpr (lobe == Diffuse) {
    /* Flat Lambert as in bsdf_diffuse_eval(). */
    const float pdf = fmaxf(cos_NO, 0.0f) * M_1_PI_F;
    sum_eval += make_spectrum(pdf) * closures.diffuse_weight;
    sum_pdf += pdf * closures.diffuse_sample_weight;
  }
  else if (cos_NI > 0.0f && cos_NO >= 0.0f &&
           !alpha_is_almost_specular(closures.alpha_x, closures.alpha_y))
  {
    const float3 h = wi + wo;
    const float3 H = h * safe_divide(1.0f, len(h));
    Spectrum eval = zero_spectrum();
    float pdf = 0.0f;
    eval_ggx(closures, N, wi, wo, H, dot(N, H),
             reflection_fresnel<lobe>(closures, dot(H, wi)),
             eval,
             pdf);
    if (pdf != 0.0f) {
      if constexpr (lobe == Metal) {
        sum_eval += eval * closures.metal_energy_scale * closures.metal_weight;
        sum_pdf += pdf * closures.metal_sample_weight;
      }
      else {
        sum_eval += eval * closures.specular_energy_scale * closures.specular_weight;
        sum_pdf += pdf * closures.specular_sample_weight;
      }
    }
  }
}

}  // namespace
}  // namespace ccl

[[gnu::always_inline]] GgxFloat4 cycles_bump_specular_normal(
    GgxFloat4 bump, GgxFloat4 geometric, GgxFloat4 incident, GgxFloat4 original_specular)
{
  using namespace ccl;
  if (bump.w == 0.0f) {
    return original_specular;
  }
  ShaderData sd{};
  sd.Ng = rgb(geometric);
  sd.wi = rgb(incident);
  sd.shader_flag = (uint(bump.w) & 1u) == 0u ? SD_USE_BUMP_MAP_CORRECTION : 0;
  return vector(maybe_ensure_valid_specular_reflection(&sd, rgb(bump)));
}

[[gnu::always_inline]] float cycles_bump_shadowing(
    GgxFloat4 shading_normal, GgxFloat4 bump, GgxFloat4 direction, bool is_eval)
{
  using namespace ccl;
  if (bump.w == 0.0f) {
    return 1.0f;
  }
  ShaderData sd{};
  sd.N = rgb(shading_normal);
  sd.shader_flag = (uint(bump.w) & 1u) == 0u ? SD_USE_BUMP_MAP_CORRECTION : 0;
  ShaderClosure closure{};
  closure.type = CLOSURE_BSDF_DIFFUSE_ID;
  closure.N = rgb(bump);
  return bump_shadowing_term(&sd, &closure, rgb(direction), is_eval);
}

static ccl::Spectrum accumulate_diffuse(const ccl::ReflectiveClosures &closures,
                               const GgxFloat4 normal,
                               const GgxFloat4 bump,
                               const ccl::float3 wi,
                               const ccl::float3 wo,
                               ccl::Spectrum &eval,
                               float &pdf)
{
  using namespace ccl;
  const float correction = cycles_bump_shadowing(normal, bump, vector(wo), true);
  if (correction > 0.0f) {
    Spectrum diffuse = zero_spectrum();
    accumulate_lobe<Diffuse>(closures, bump.w == 0.0f ? rgb(normal) : rgb(bump),
                             wi, wo, diffuse, pdf);
    const Spectrum weighted = diffuse * correction;
    eval += weighted;
    return weighted;
  }
  return zero_spectrum();
}

/* A separate Translucent lobe uses the stock corrected specular normal. Like
 * other diffuse lobes it receives bump correction in bsdf_eval(). */
static ccl::Spectrum accumulate_principled_translucent(
    const GgxFloat4 translucent,
    const float sample_weight,
    const GgxFloat4 normal,
    GgxFloat4 specular_normal,
    const GgxFloat4 wi,
    const GgxFloat4 wo,
    ccl::Spectrum &eval,
    float &weighted_pdf)
{
  using namespace ccl;
  if (!(sample_weight > 0.0f)) {
    return zero_spectrum();
  }
  specular_normal.w = (unsigned(-translucent.w) & CYCLES_WEBGPU_GGX_TRANSLUCENT_CORRECTION) ?
                          2.0f : 1.0f;
  const float correction = cycles_bump_shadowing(normal, specular_normal, wo, true);
  if (!(correction > 0.0f)) {
    return zero_spectrum();
  }
  const DiffuseBsdf bsdf{.N = rgb(specular_normal)};
  float pdf = 0.0f;
  const Spectrum contribution = bsdf_translucent_eval(
      reinterpret_cast<const ShaderClosure *>(&bsdf), rgb(wi), rgb(wo), &pdf) *
      rgb(translucent) * correction;
  eval += contribution;
  weighted_pdf += pdf * sample_weight;
  return contribution;
}

static bool glass_is_transmission(const ccl::MicrofacetBsdf &glass, const ccl::float3 wo)
{
  using namespace ccl;
  return CLOSURE_IS_BSDF_TRANSMISSION(glass.type) ||
         (CLOSURE_IS_GLASS(glass.type) && dot(glass.N, wo) <= 0.0f);
}

static void cycles_bsdf_film_weights(const ccl::Spectrum diffuse,
                                     const ccl::Spectrum transmission,
                                     const ccl::Spectrum sum,
                                     GgxFloat4 &diffuse_ratio,
                                     GgxFloat4 &glossy_ratio)
{
  using namespace ccl;
  /* Stock bsdf_eval_pass_*_weight(). These fused surface records carry the
   * Glass transmission term explicitly until the remaining glossy sum is known. */
  diffuse_ratio = vector(safe_divide(diffuse, sum));
  glossy_ratio = vector(safe_divide(max(sum - diffuse - transmission, zero_spectrum()), sum));
}

[[gnu::always_inline]] GgxFloat4 cycles_ggx_eval(GgxFloat4 material,
                                                GgxFloat4 tint_alpha,
                                                GgxFloat4 weights,
                                                GgxFloat4 glass,
                                                GgxFloat4 auxiliary,
                                                GgxFloat4 normal,
                                                GgxFloat4 specular_normal,
                                                GgxFloat4 shader_specular_normal,
                                                GgxFloat4 bump,
                                                GgxFloat4 view,
                                                GgxFloat4 light,
                                                float blur_alpha,
                                                bool light_passes,
                                                GgxFloat4 &diffuse_ratio,
                                                GgxFloat4 &glossy_ratio)
{
  using namespace ccl;
  diffuse_ratio = glossy_ratio = {};
  const float3 N = rgb(normal);
  const float3 specular_N = rgb(specular_normal);
  const float3 wi = rgb(view);
  const float3 wo = rgb(light);
  const bool has_translucent = CYCLES_WEBGPU_GGX_HAS_TRANSLUCENT(glass);
  const auto translucent = has_translucent ? glass : GgxFloat4{};
  const bool translucent_first = has_translucent &&
      (unsigned(-translucent.w) & CYCLES_WEBGPU_GGX_TRANSLUCENT_FIRST) != 0;
  Spectrum translucent_color = rgb(translucent);
  const float translucent_weight = closure_sample_weight(0, translucent_color);
  if (has_translucent) {
    glass = {};
  }
  MicrofacetBsdf coat_bsdf{};
  const bool has_coat = setup_compact_white_coat(coat_bsdf, tint_alpha, weights, glass, specular_N,
                                                rgb(shader_specular_normal), wi, blur_alpha);
  FresnelGeneralizedSchlick glass_fresnel{};
  /* Keep the private Fresnel pointer valid across the setup branch so it
   * compiles without a pointer phi in SPIR-V. */
  MicrofacetBsdf glass_bsdf{.type = CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID,
                           .fresnel_type = MicrofacetFresnel::GENERALIZED_SCHLICK,
                           .fresnel = &glass_fresnel};
  const bool has_glass = glass_enabled && setup_glass(
      glass_bsdf, glass_fresnel, material, tint_alpha, weights, glass, auxiliary,
      GgxFloat4{specular_N.x, specular_N.y, specular_N.z, normal.w}, wi);
  if (has_glass) {
    bsdf_microfacet_blur(reinterpret_cast<ShaderClosure *>(&glass_bsdf), blur_alpha);
  }
  if (glass_enabled && glass.x > CLOSURE_WEIGHT_CUTOFF) {
    weights.w *= 1.0f - glass.x;
  }
  ReflectiveClosures closures;
  setup_closures(closures, material, tint_alpha, weights, glass, auxiliary, specular_N, wi);
  blur_reflective_ggx(closures, blur_alpha);
  Spectrum eval = zero_spectrum();
  Spectrum transmission = zero_spectrum();
  Spectrum diffuse = zero_spectrum();
  float weighted_pdf = 0.0f;
  if (translucent_first) {
    diffuse += accumulate_principled_translucent(
        translucent, translucent_weight, normal, specular_normal, view, light, eval, weighted_pdf);
  }
  const bool specular_eval = valid_specular_eval(N, specular_N, wo);
  if (has_coat && valid_specular_eval(N, coat_bsdf.N, wo)) {
    accumulate_microfacet(coat_bsdf, wi, wo, eval, weighted_pdf);
  }
  if (closures.has_metal && specular_eval) {
    accumulate_lobe<Metal>(closures, specular_N, wi, wo, eval, weighted_pdf);
  }
  if (has_glass && specular_eval) {
    const Spectrum glass_eval = accumulate_microfacet(glass_bsdf, wi, wo, eval, weighted_pdf);
    if (light_passes && glass_is_transmission(glass_bsdf, wo)) {
      transmission = glass_eval;
    }
  }
  if (closures.has_specular && specular_eval) {
    accumulate_lobe<Specular>(closures, specular_N, wi, wo, eval, weighted_pdf);
  }
  if (closures.has_diffuse) {
    diffuse += accumulate_diffuse(closures, normal, bump, wi, wo, eval, weighted_pdf);
  }
  if (!translucent_first && has_translucent) {
    diffuse += accumulate_principled_translucent(
        translucent, translucent_weight, normal, specular_normal, view, light, eval, weighted_pdf);
  }
  if (light_passes) {
    cycles_bsdf_film_weights(diffuse, transmission, eval, diffuse_ratio, glossy_ratio);
  }
  const float transparent_sample_weight = glass.z >= 0.0f && glass.w == 0.0f ?
                                              fabsf(auxiliary.w) : 0.0f;
  const float total = coat_bsdf.sample_weight + closures.metal_sample_weight +
                      (has_glass ? glass_bsdf.sample_weight : 0.0f) +
                      closures.specular_sample_weight + closures.subsurface_sample_weight +
                      closures.diffuse_sample_weight + transparent_sample_weight +
                      translucent_weight;
  return {eval.x, eval.y, eval.z, total > 0.0f ? weighted_pdf / total : 0.0f};
}

[[gnu::always_inline]] unsigned int cycles_ggx_sample(GgxFloat4 material,
                                                       GgxFloat4 tint_alpha,
                                                       GgxFloat4 weights,
                                                       GgxFloat4 glass,
                                                       GgxFloat4 normal,
                                                       GgxFloat4 geometric_normal,
                                                       GgxFloat4 specular_normal,
                                                       GgxFloat4 shader_specular_normal,
                                                       GgxFloat4 bump,
                                                       GgxFloat4 view,
                                                       GgxFloat4 random_seed,
                                                       GgxFloat4 auxiliary,
                                                       float blur_alpha,
                                                       bool ray_differentials,
                                                       bool only_transparent,
                                                       GgxFloat4 &weight,
                                                       GgxFloat4 &out,
                                                       bool light_passes,
                                                       GgxFloat4 &diffuse_ratio,
                                                       GgxFloat4 &glossy_ratio)
{
  using namespace ccl;
  diffuse_ratio = glossy_ratio = {};
  weight = {0.0f, 0.0f, 0.0f, 0.0f};
  out = {0.0f, 0.0f, 0.0f, 0.0f};
  const float3 N = rgb(normal);
  const float3 specular_N = rgb(specular_normal);
  const float3 wi = rgb(view);
  const float3 Ng = rgb(geometric_normal);
  const bool has_translucent = CYCLES_WEBGPU_GGX_HAS_TRANSLUCENT(glass);
  const auto translucent_lobe = has_translucent ? glass : GgxFloat4{};
  const bool translucent_first = has_translucent &&
      (unsigned(-translucent_lobe.w) & CYCLES_WEBGPU_GGX_TRANSLUCENT_FIRST) != 0;
  Spectrum translucent_color = rgb(translucent_lobe);
  const float translucent = closure_sample_weight(0, translucent_color);
  if (has_translucent) {
    glass = {};
  }
  const GgxFloat4 transparent = glass.z >= 0.0f && glass.w == 0.0f ? auxiliary : GgxFloat4{};
  const float transparent_weight = fabsf(transparent.w);
  if (only_transparent) {
    if (transparent_weight > 0.0f) {
      weight = {transparent.x, transparent.y, transparent.z, 1e6f};
      return LABEL_TRANSMIT | LABEL_TRANSPARENT;
    }
    return LABEL_NONE;
  }
  const float cos_NI = dot(specular_N, wi);
  MicrofacetBsdf coat_bsdf{};
  const bool has_coat = setup_compact_white_coat(coat_bsdf, tint_alpha, weights, glass, specular_N,
                                                rgb(shader_specular_normal), wi, blur_alpha);
  FresnelGeneralizedSchlick glass_fresnel{};
  MicrofacetBsdf glass_bsdf{.type = CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID,
                           .fresnel_type = MicrofacetFresnel::GENERALIZED_SCHLICK,
                           .fresnel = &glass_fresnel};
  const bool has_glass = glass_enabled && setup_glass(
      glass_bsdf, glass_fresnel, material, tint_alpha, weights, glass, auxiliary,
      GgxFloat4{specular_N.x, specular_N.y, specular_N.z, normal.w}, wi);
  if (has_glass) {
    bsdf_microfacet_blur(reinterpret_cast<ShaderClosure *>(&glass_bsdf), blur_alpha);
  }
  if (glass_enabled && glass.x > CLOSURE_WEIGHT_CUTOFF) {
    weights.w *= 1.0f - glass.x;
  }
  ReflectiveClosures closures;
  setup_closures(closures, material, tint_alpha, weights, glass, auxiliary, specular_N, wi);
  blur_reflective_ggx(closures, blur_alpha);

  const float coat_weight = coat_bsdf.sample_weight;
  const float metal = closures.metal_sample_weight;
  const float glass_weight = has_glass ? glass_bsdf.sample_weight : 0.0f;
  const float specular = closures.specular_sample_weight;
  const float diffuse = closures.diffuse_sample_weight;
  const float subsurface = closures.subsurface_sample_weight;
  const bool transparent_first = transparent.w < 0.0f;
  const float transparent_prefix = transparent_first ? transparent_weight : 0.0f;
  const float translucent_prefix = translucent_first ? translucent : 0.0f;
  const float prefix = translucent_prefix + transparent_prefix;
  const float coat_end = prefix + coat_weight;
  const float metal_end = coat_end + metal;
  const float glass_end = metal_end + glass_weight;
  const float specular_end = glass_end + specular;
  const float subsurface_end = specular_end + subsurface;
  const float opaque_end = subsurface_end + diffuse;
  const float principled_end = opaque_end + (transparent_first ? 0.0f : transparent_weight);
  const float total = principled_end + (translucent_first ? 0.0f : translucent);
  if (!(total > 0.0f)) {
    return LABEL_NONE;
  }

  const float r = random_seed.z * total;
  if (transparent_weight > 0.0f &&
      (transparent_first ? ((r >= translucent_prefix && r < prefix) ||
                           (r >= total && !translucent_first)) :
                          (r >= opaque_end && r < principled_end)))
  {
    Spectrum eval;
    float3 direction;
    float pdf;
    const int label = bsdf_transparent_sample(nullptr, Ng, wi, &eval, &direction, &pdf);
    eval *= rgb(transparent);
    float weighted_pdf = pdf * transparent_weight;
    Spectrum translucent_eval = zero_spectrum();
    if (translucent_first) {
      translucent_eval = accumulate_principled_translucent(translucent_lobe, translucent, normal,
          specular_normal, view, vector(direction), eval, weighted_pdf);
    }
    const bool specular_eval = valid_specular_eval(N, specular_N, direction);
    if (specular_eval && closures.has_metal) {
      accumulate_lobe<Metal>(closures, specular_N, wi, direction, eval, weighted_pdf);
    }
    if (specular_eval && has_glass) {
      accumulate_microfacet(glass_bsdf, wi, direction, eval, weighted_pdf);
    }
    if (specular_eval && closures.has_specular) {
      accumulate_lobe<Specular>(closures, specular_N, wi, direction, eval, weighted_pdf);
    }
    if (closures.has_diffuse) {
      accumulate_diffuse(closures, normal, bump, wi, direction, eval, weighted_pdf);
    }
    if (!translucent_first && has_translucent) {
      translucent_eval = accumulate_principled_translucent(translucent_lobe, translucent, normal,
          specular_normal, view, vector(direction), eval, weighted_pdf);
    }
    const float mixed_pdf = weighted_pdf / total;
    const Spectrum multiplier = eval / mixed_pdf;
    weight = {multiplier.x, multiplier.y, multiplier.z, mixed_pdf};
    if (light_passes && has_translucent) {
      diffuse_ratio = vector(safe_divide(translucent_eval, eval));
    }
    return label;
  }

  /* Pick and remap from surface_shader_bsdf_bssrdf_pick(). */
  Lobe lobe = closures.has_metal ? Metal : has_glass ? Glass :
              closures.has_specular ? Specular : Diffuse;
  float3 random = rgb(random_seed);
  const bool multiple = has_coat || translucent > 0.0f || transparent_weight > 0.0f ||
                        subsurface > 0.0f || (has_glass &&
                         (closures.has_metal || closures.has_specular || closures.has_diffuse)) ||
                        (closures.has_metal && closures.has_specular) ||
                        (closures.has_diffuse && (closures.has_metal || closures.has_specular));
  if (multiple) {
    if (translucent > 0.0f &&
        (translucent_first ? (r < translucent_prefix || r >= total) :
                             (r >= principled_end && r < total)))
    {
      lobe = Translucent;
    }
    else if (r < coat_end) {
      lobe = CoatReflection;
      random.z = (r - prefix) / coat_weight;
    }
    else if (r < metal_end) {
      lobe = Metal;
      random.z = (r - coat_end) / metal;
    }
    else if (glass_enabled && r < glass_end) {
      lobe = Glass;
      random.z = (r - metal_end) / glass_weight;
    }
    else if (r < specular_end) {
      lobe = Specular;
      random.z = (r - glass_end) / specular;
    }
    else if (subsurface > 0.0f && (r < subsurface_end ||
                                      diffuse == 0.0f)) {
      lobe = Subsurface;
    }
    else if (r < opaque_end) {
      lobe = Diffuse;
    }
  }

  Spectrum eval = zero_spectrum();
  float3 wo;
  float pdf = 0.0f;
  float selected_weight;
  int label = LABEL_REFLECT | LABEL_GLOSSY;
  if (lobe == Subsurface) {
    /* surface_shader_bssrdf_sample_weight compensates for choosing this closure. */
    eval = closures.subsurface_weight * (total / subsurface);
    wo = zero_float3();
    pdf = 1.0f;
    selected_weight = subsurface;
    label = LABEL_SUBSURFACE_SCATTER;
  }
  else if (lobe == Translucent) {
    const DiffuseBsdf bsdf{.N = specular_N};
    label = bsdf_translucent_sample(reinterpret_cast<const ShaderClosure *>(&bsdf),
                                    Ng, wi, make_float2(random), &eval, &wo, &pdf);
    eval *= translucent_color;
    selected_weight = translucent;
  }
  else if (lobe == Diffuse) {
    wo = zero_float3();
    webgpu_sample_cos_hemisphere(bump.w == 0.0f ? N : rgb(bump), make_float2(random), &wo, &pdf);
    if (!(dot(Ng, wo) > 0.0f)) {
      return LABEL_NONE;
    }
    eval = make_spectrum(pdf) * closures.diffuse_weight *
           cycles_bump_shadowing(normal, bump, vector(wo), false);
    selected_weight = diffuse;
    label = LABEL_REFLECT | LABEL_DIFFUSE;
  }
  else if (glass_enabled && lobe == Glass) {
    float2 sampled_roughness;
    float eta;
    const auto *closure = reinterpret_cast<const ShaderClosure *>(&glass_bsdf);
    if (is_beckmann_transmission(glass_bsdf.type)) {
      label = bsdf_microfacet_beckmann_sample(
          &energy_globals, closure, Ng, wi, random, &eval, &wo, &pdf, &sampled_roughness, &eta);
    }
    else {
      label = bsdf_microfacet_ggx_sample(
          &energy_globals, closure, Ng, wi, random, &eval, &wo, &pdf, &sampled_roughness, &eta);
    }
    eval *= glass_bsdf.weight;
    selected_weight = glass_weight;
  }
  else if (lobe == CoatReflection) {
    float2 sampled_roughness;
    float eta;
    const auto *closure = reinterpret_cast<const ShaderClosure *>(&coat_bsdf);
    label = bsdf_microfacet_ggx_sample(
        &energy_globals, closure, Ng, wi, random, &eval, &wo, &pdf, &sampled_roughness, &eta);
    eval *= coat_bsdf.weight;
    selected_weight = coat_weight;
  }
  else {
    if (!(cos_NI > 0.0f) || random.z >= 1.0f) {
      return LABEL_NONE;
    }
    /* Specular and anisotropic reflection from bsdf_microfacet_sample<GGX>(). */
    const bool singular = alpha_is_almost_specular(closures.alpha_x, closures.alpha_y);
    float3 H = specular_N;
    float cos_NH = 1.0f;
    if (!singular) {
      float3 X, Y;
      reflective_ggx_basis(closures, specular_N, X, Y);
      const float3 local_I = make_float3(dot(X, wi), dot(Y, wi), cos_NI);
      const float3 local_H = microfacet_ggx_sample_vndf(
          local_I, closures.alpha_x, closures.alpha_y, make_float2(random));
      H = to_global(local_H, X, Y, specular_N);
      cos_NH = local_H.z;
    }
    const float cos_HI = dot(H, wi);
    wo = 2.0f * cos_HI * H - wi;
    if (dot(specular_N, wo) < 0.0f || dot(Ng, wo) < 0.0f) {
      return LABEL_NONE;
    }
    const bool metallic = lobe == Metal;
    Spectrum fresnel;
    if (metallic) {
      fresnel = reflection_fresnel<Metal>(closures, cos_HI);
    }
    else {
      fresnel = reflection_fresnel<Specular>(closures, cos_HI);
    }
    if (is_zero(fresnel)) {
      return LABEL_NONE;
    }
    if (singular) {
      pdf = 1e6f;
      eval = fresnel * pdf;
      label = LABEL_REFLECT | LABEL_SINGULAR;
    }
    else {
      eval_ggx(closures, specular_N, wi, wo, H, cos_NH, fresnel, eval, pdf);
    }
    if (metallic) {
      eval *= closures.metal_energy_scale;
      eval *= closures.metal_weight;
      selected_weight = metal;
    }
    else {
      eval *= closures.specular_energy_scale;
      eval *= closures.specular_weight;
      selected_weight = specular;
    }
  }
  if (pdf == 0.0f) {
    return LABEL_NONE;
  }

  const float alpha_squared = closures.alpha_x * closures.alpha_y;
  const float glass_roughness_squared = glass_bsdf.alpha_x * glass_bsdf.alpha_y;
  const float coat_roughness_squared = coat_bsdf.alpha_x * coat_bsdf.alpha_y;
  float roughness_squared = (lobe == Diffuse || lobe == Translucent) ? 1.0f :
                            lobe == Glass ? glass_roughness_squared :
                            lobe == CoatReflection ? coat_roughness_squared : alpha_squared;
  Spectrum diffuse_eval = light_passes && (lobe == Diffuse || lobe == Translucent) ?
                              eval : zero_spectrum();
  const bool glass_transmission = light_passes && has_glass && glass_is_transmission(glass_bsdf, wo);
  Spectrum transmission_eval = glass_transmission && lobe == Glass ? eval : zero_spectrum();
  /* The balance heuristic also evaluates the other lobes at the sampled
   * direction. The selected lobe is accumulated first, as in Cycles. */
  if (multiple && lobe != Subsurface) {
    float weighted_pdf = pdf * selected_weight;
    float roughness_pdf = weighted_pdf * roughness_squared;
    float sum_weight = selected_weight + subsurface + transparent_prefix;
    Spectrum translucent_eval = zero_spectrum();
    if (lobe != Translucent && translucent_first) {
      float other_pdf = 0.0f;
      translucent_eval = accumulate_principled_translucent(translucent_lobe, translucent, normal,
          specular_normal, view, vector(wo), eval, other_pdf);
      weighted_pdf += other_pdf;
      if (ray_differentials) {
        roughness_pdf += other_pdf;
      }
      sum_weight += translucent;
    }
    const bool specular_eval = valid_specular_eval(N, specular_N, wo);
    if (has_coat && lobe != CoatReflection) {
      if (valid_specular_eval(N, coat_bsdf.N, wo)) {
        float other_pdf = 0.0f;
        accumulate_microfacet(coat_bsdf, wi, wo, eval, other_pdf);
        weighted_pdf += other_pdf;
        if (ray_differentials) {
          roughness_pdf += other_pdf * coat_roughness_squared;
        }
      }
      sum_weight += coat_weight;
    }
    if (closures.has_metal && lobe != Metal) {
      if (specular_eval) {
        float other_pdf = 0.0f;
        accumulate_lobe<Metal>(closures, specular_N, wi, wo, eval, other_pdf);
        weighted_pdf += other_pdf;
        if (ray_differentials) {
          roughness_pdf += other_pdf * alpha_squared;
        }
      }
      sum_weight += metal;
    }
    if (has_glass && lobe != Glass) {
      if (specular_eval) {
        float other_pdf = 0.0f;
        const Spectrum glass_eval = accumulate_microfacet(glass_bsdf, wi, wo, eval, other_pdf);
        if (glass_transmission) {
          transmission_eval = glass_eval;
        }
        weighted_pdf += other_pdf;
        if (ray_differentials) {
          roughness_pdf += other_pdf * glass_roughness_squared;
        }
      }
      sum_weight += glass_weight;
    }
    if (closures.has_specular && lobe != Specular) {
      if (specular_eval) {
        float other_pdf = 0.0f;
        accumulate_lobe<Specular>(closures, specular_N, wi, wo, eval, other_pdf);
        weighted_pdf += other_pdf;
        if (ray_differentials) {
          roughness_pdf += other_pdf * alpha_squared;
        }
      }
      sum_weight += specular;
    }
    if (closures.has_diffuse && lobe != Diffuse) {
      float other_pdf = 0.0f;
      diffuse_eval = accumulate_diffuse(closures, normal, bump, wi, wo, eval, other_pdf);
      weighted_pdf += other_pdf;
      if (ray_differentials) {
        roughness_pdf += other_pdf;
      }
      sum_weight += diffuse;
    }
    if (lobe != Translucent && !translucent_first && has_translucent) {
      float other_pdf = 0.0f;
      translucent_eval = accumulate_principled_translucent(translucent_lobe, translucent, normal,
          specular_normal, view, vector(wo), eval, other_pdf);
      weighted_pdf += other_pdf;
      if (ray_differentials) {
        roughness_pdf += other_pdf;
      }
      sum_weight += translucent;
    }
    diffuse_eval += translucent_eval;
    if (ray_differentials) {
      /* Match _surface_shader_bsdf_eval_mis() used by stock shade_surface. */
      roughness_squared = weighted_pdf > 0.0f ? roughness_pdf / weighted_pdf : 0.0f;
    }
    pdf = weighted_pdf / (sum_weight + (transparent_first ? 0.0f : transparent_weight));
  }
  if (!(pdf > 0.0f) || is_zero(eval)) {
    return LABEL_NONE;
  }

  const Spectrum sample_weight = eval / pdf;
  if (light_passes) {
    cycles_bsdf_film_weights(diffuse_eval, transmission_eval, eval, diffuse_ratio, glossy_ratio);
  }
  weight = {sample_weight.x, sample_weight.y, sample_weight.z, pdf};
  out = vector(wo);
  out.w = ray_differentials ? roughness_squared : 0.0f;
  return label;
}

namespace {
/* Keep the stock Fresnel storage live for the BSDF callback. Specializing each
 * branch also keeps physical and F82 conductor parameters separately typed. */
template<typename Evaluate>
[[gnu::always_inline]] auto with_standalone_glossy(
    GgxFloat4 material,
    GgxFloat4 geometry,
    GgxFloat4 primary,
    GgxFloat4 secondary,
    const ccl::float3 N,
    const ccl::float3 wi,
    const float blur_alpha,
    const Evaluate evaluate)
{
  using namespace ccl;
  const uint model = uint(secondary.w);
  const ClosureType distribution = ClosureType(model & 0xffu);
  const ClosureType fresnel_type = ClosureType(model >> 8u);
  const bool multiscatter = distribution == CLOSURE_BSDF_MICROFACET_MULTI_GGX_ID;
  MicrofacetBsdf bsdf{};
  bsdf.N = N;
  bsdf.T = rgb(geometry);
  bsdf.alpha_x = geometry.w;
  bsdf.alpha_y = primary.w;
  bsdf.ior = 1.0f;
  bsdf.weight = rgb(material);
  bsdf.sample_weight = closure_sample_weight(0, bsdf.weight);
  if (distribution == CLOSURE_BSDF_ASHIKHMIN_SHIRLEY_ID) {
    bsdf_ashikhmin_shirley_setup(&bsdf);
  }
  else if (distribution == CLOSURE_BSDF_MICROFACET_BECKMANN_ID) {
    bsdf_microfacet_beckmann_setup(&bsdf);
  }
  else {
    bsdf_microfacet_ggx_setup(&bsdf);
  }
  if (fresnel_type == CLOSURE_BSDF_PHYSICAL_CONDUCTOR) {
    FresnelConductor fresnel = {
        .thin_film = {.thickness = 0.0f, .ior = 1.0f},
        .ior = {max(rgb(primary), zero_spectrum()), max(rgb(secondary), zero_spectrum())}};
    bsdf_microfacet_setup_fresnel_conductor(&energy_globals, &bsdf, wi, &fresnel, multiscatter);
    bsdf_microfacet_blur(reinterpret_cast<ShaderClosure *>(&bsdf), blur_alpha);
    return evaluate(bsdf);
  }
  if (fresnel_type == CLOSURE_BSDF_F82_CONDUCTOR) {
    FresnelF82Tint fresnel = {
        .thin_film = {.thickness = 0.0f, .ior = 1.0f}, .f0 = saturate(rgb(primary))};
    bsdf_microfacet_setup_fresnel_f82_tint(
        &energy_globals, &bsdf, wi, &fresnel, saturate(rgb(secondary)), multiscatter);
    bsdf_microfacet_blur(reinterpret_cast<ShaderClosure *>(&bsdf), blur_alpha);
    return evaluate(bsdf);
  }
  if (secondary.x != 0.0f) {
    bsdf.T = rotate_around_axis(bsdf.T, N, secondary.x * M_2PI_F);
  }
  if (multiscatter) {
    bsdf_microfacet_setup_fresnel_constant(&energy_globals, &bsdf, wi, rgb(primary));
  }
  auto *closure = reinterpret_cast<ShaderClosure *>(&bsdf);
  if (distribution == CLOSURE_BSDF_ASHIKHMIN_SHIRLEY_ID) {
    bsdf_ashikhmin_shirley_blur(closure, blur_alpha);
  }
  else {
    bsdf_microfacet_blur(closure, blur_alpha);
  }
  return evaluate(bsdf);
}

}  // namespace

#ifndef CYCLES_WEBGPU_EXCLUDE_STANDALONE
/* Evaluate the other closure for Cycles' one-sample MIS, including the
 * corrected Translucent normal used by svm_node_closure_bsdf(). */
template<bool translucent>
static GgxFloat4 evaluate_diffuse_translucent_lobe(const GgxFloat4 color,
                                                 const float sample_weight,
                                                 const GgxFloat4 normal,
                                                 const GgxFloat4 translucent_normal,
                                                 const GgxFloat4 incoming,
                                                 const GgxFloat4 outgoing,
                                                 const bool bump_correction)
{
  using namespace ccl;
  if (!(sample_weight > 0.0f)) {
    return {};
  }
  const DiffuseBsdf bsdf{.N = rgb(translucent ? translucent_normal : normal)};
  const auto *closure = reinterpret_cast<const ShaderClosure *>(&bsdf);
  float pdf = 0.0f;
  Spectrum eval;
  if constexpr (translucent) {
    /* Both stock lobe types are diffuse for bump_shadowing_term(). */
    const float correction = cycles_bump_shadowing(
        normal,
        GgxFloat4{translucent_normal.x, translucent_normal.y, translucent_normal.z,
                  bump_correction ? 2.0f : 1.0f},
        outgoing, true);
    if (!(correction > 0.0f)) {
      return {};
    }
    eval = bsdf_translucent_eval(closure, rgb(incoming), rgb(outgoing), &pdf) * correction;
  }
  else {
    eval = bsdf_diffuse_eval(closure, rgb(incoming), rgb(outgoing), &pdf);
  }
  eval *= rgb(color);
  return {eval.x, eval.y, eval.z, pdf};
}

static void setup_mixed_glossy(ccl::MicrofacetBsdf &bsdf,
                               const GgxFloat4 glossy,
                               const GgxFloat4 specular_normal,
                               const unsigned int distribution,
                               const float blur_alpha)
{
  using namespace ccl;
  bsdf.N = rgb(specular_normal);
  bsdf.weight = rgb(glossy);
  bsdf.T = zero_float3();
  bsdf.ior = 1.0f;
  bsdf.alpha_x = glossy.w;
  bsdf.alpha_y = glossy.w;
  if (distribution == CLOSURE_BSDF_MICROFACET_BECKMANN_ID) {
    bsdf_microfacet_beckmann_setup(&bsdf);
  }
  else {
    bsdf_microfacet_ggx_setup(&bsdf);
  }
  bsdf_microfacet_blur(reinterpret_cast<ShaderClosure *>(&bsdf), blur_alpha);
}

static int sample_mixed_glossy(const ccl::MicrofacetBsdf &bsdf,
                               const ccl::float3 Ng,
                               const ccl::float3 wi,
                               const ccl::float3 random,
                               ccl::Spectrum &eval,
                               ccl::float3 &wo,
                               float &pdf)
{
  using namespace ccl;
  float2 roughness;
  float eta;
  const auto *closure = reinterpret_cast<const ShaderClosure *>(&bsdf);
  return bsdf.type == CLOSURE_BSDF_MICROFACET_BECKMANN_ID ?
             bsdf_microfacet_beckmann_sample(&energy_globals, closure, Ng, wi, random,
                                            &eval, &wo, &pdf, &roughness, &eta) :
             bsdf_microfacet_ggx_sample(&energy_globals, closure, Ng, wi, random,
                                       &eval, &wo, &pdf, &roughness, &eta);
}

static GgxFloat4 evaluate_mixed_glossy(const ccl::MicrofacetBsdf &bsdf,
                                        const ccl::float3 N,
                                        const ccl::float3 wi,
                                        const ccl::float3 wo)
{
  using namespace ccl;
  if (!valid_specular_eval(N, bsdf.N, wo)) {
    return {};
  }
  float pdf = 0.0f;
  const auto *closure = reinterpret_cast<const ShaderClosure *>(&bsdf);
  const Spectrum eval = bsdf.type == CLOSURE_BSDF_MICROFACET_BECKMANN_ID ?
      bsdf_microfacet_beckmann_eval(&energy_globals, closure, wi, wo, &pdf) :
      bsdf_microfacet_ggx_eval(&energy_globals, closure, wi, wo, &pdf);
  const Spectrum weighted = eval * bsdf.weight;
  return {weighted.x, weighted.y, weighted.z, pdf};
}

static GgxFloat4 evaluate_mixed_glossy_diffuse(const GgxFloat4 material,
                                                 const float sample_weight,
                                                 const GgxFloat4 normal,
                                                 const GgxFloat4 bump,
                                                 const GgxFloat4 incoming,
                                                 const GgxFloat4 outgoing)
{
  const float correction = cycles_bump_shadowing(normal, bump, outgoing, true);
  if (!(correction > 0.0f)) {
    return {};
  }
  const GgxFloat4 diffuse = evaluate_diffuse_translucent_lobe<false>(
      material, sample_weight, bump.w != 0.0f ? bump : normal, {}, incoming, outgoing, false);
  return {diffuse.x * correction, diffuse.y * correction, diffuse.z * correction, diffuse.w};
}

#ifdef CYCLES_WEBGPU_ORDERED_CLOSURES
namespace ccl {
namespace {

struct OrderedClosure {
  GgxFloat4 color_alpha;
  GgxFloat4 bump;
  unsigned int type;

  bool diffuse() const
  {
    return CLOSURE_IS_BSDF_DIFFUSE(type);
  }
};

ccl_device float ordered_closure_sample_weight(Spectrum color, const unsigned int type)
{
  /* Transparent setup preserves the sign of extinction. */
  return type == CLOSURE_BSDF_TRANSPARENT_ID ? fabsf(average(color)) :
                                              closure_sample_weight(0, color);
}

ccl_device unsigned int ordered_closure_address(const GgxFloat4 material,
                                                const unsigned int index)
{
  return cycles_ordered_tail_index(material.x) + CYCLES_WEBGPU_ORDERED_LOBE_VECTORS * index;
}

ccl_device OrderedClosure read_ordered_closure(const GgxFloat4 material,
                                               const unsigned int index)
{
  const unsigned int address = ordered_closure_address(material, index);
  const GgxFloat4 color = cycles_material_lobe(address);
  const GgxFloat4 normal_alpha = cycles_material_lobe(address + 1);
  const unsigned int type = unsigned(color.w);
  const float bump_mode = (type & CYCLES_WEBGPU_ORDERED_LOBE_BUMP) ?
                             CYCLES_WEBGPU_NORMAL_CLOSURE + material.z : 0.0f;
  return {{color.x, color.y, color.z, normal_alpha.w},
          {normal_alpha.x, normal_alpha.y, normal_alpha.z, bump_mode},
          type & ~CYCLES_WEBGPU_ORDERED_LOBE_BUMP};
}

ccl_device void setup_ordered_glossy(MicrofacetBsdf &bsdf,
                                    const OrderedClosure lobe,
                                    const GgxFloat4 geometric,
                                    const GgxFloat4 wi,
                                    const GgxFloat4 specular_normal,
                                    const float blur_alpha)
{
  const auto N = cycles_bump_specular_normal(lobe.bump, geometric, wi, specular_normal);
  setup_mixed_glossy(bsdf, lobe.color_alpha, N, lobe.type, blur_alpha);
}

struct OrderedClosureEval {
  Spectrum eval = zero_spectrum();
  Spectrum diffuse = zero_spectrum();
  float pdf = 0.0f;
  float pdf_roughness_squared = 0.0f;
  float sample_weight = 0.0f;

  void add(const GgxFloat4 value,
           const float weight,
           const float roughness_squared,
           const bool is_diffuse)
  {
    eval += rgb(value);
    if (is_diffuse) {
      diffuse += rgb(value);
    }
    const float weighted_pdf = value.w * weight;
    pdf += weighted_pdf;
    pdf_roughness_squared += weighted_pdf * roughness_squared;
    sample_weight += weight;
  }
};

ccl_device void accumulate_ordered_closure(OrderedClosureEval &result,
                                            const OrderedClosure lobe,
                                            const GgxFloat4 normal,
                                            const GgxFloat4 geometric,
                                            const GgxFloat4 specular_normal,
                                            const GgxFloat4 wi,
                                            const GgxFloat4 wo,
                                            const float blur_alpha,
                                            const bool bump_correction)
{
  const float weight = ordered_closure_sample_weight(rgb(lobe.color_alpha), lobe.type);
  if (lobe.type == CLOSURE_BSDF_TRANSPARENT_ID) {
    /* The stock evaluator counts the weight but a singular lobe contributes no density. */
    result.sample_weight += weight;
  }
  else if (lobe.type == CLOSURE_BSDF_TRANSLUCENT_ID) {
    const GgxFloat4 translucent_normal = cycles_bump_specular_normal(
        lobe.bump, geometric, wi, specular_normal);
    result.add(evaluate_diffuse_translucent_lobe<true>(
                   lobe.color_alpha, weight, normal, translucent_normal, wi, wo, bump_correction),
               weight, 1.0f, true);
  }
  else if (lobe.diffuse()) {
    result.add(evaluate_mixed_glossy_diffuse(lobe.color_alpha, weight, normal, lobe.bump, wi, wo),
               weight, 1.0f, true);
  }
  else {
    MicrofacetBsdf bsdf{};
    setup_ordered_glossy(bsdf, lobe, geometric, wi, specular_normal, blur_alpha);
    result.add(evaluate_mixed_glossy(bsdf, rgb(normal), rgb(wi), rgb(wo)),
               weight, bsdf.alpha_x * bsdf.alpha_y, false);
  }
}

ccl_device GgxFloat4 evaluate_ordered_closures(const GgxFloat4 material,
                                              const GgxFloat4 normal,
                                              const GgxFloat4 geometric,
                                              const GgxFloat4 specular_normal,
                                              const GgxFloat4 wi,
                                              const GgxFloat4 wo,
                                              const float blur_alpha,
                                              GgxFloat4 &diffuse_ratio)
{
  OrderedClosureEval result;
  const unsigned int count = min(unsigned(material.y), CYCLES_WEBGPU_MAX_ORDERED_LOBES);
  for (unsigned int index = 0; index < count; index++) {
    accumulate_ordered_closure(result, read_ordered_closure(material, index),
                               normal, geometric, specular_normal, wi, wo, blur_alpha,
                               material.z != 0.0f);
  }
  diffuse_ratio = vector(safe_divide(result.diffuse, result.eval));
  return {result.eval.x, result.eval.y, result.eval.z, safe_divide(result.pdf, result.sample_weight)};
}

ccl_device unsigned int sample_ordered_closures(const GgxFloat4 material,
                                                const GgxFloat4 normal,
                                                const GgxFloat4 geometric,
                                                const GgxFloat4 specular_normal,
                                                const GgxFloat4 wi,
                                                const GgxFloat4 random,
                                                const float blur_alpha,
                                                GgxFloat4 &weight,
                                                GgxFloat4 &out,
                                                GgxFloat4 &diffuse_ratio)
{
  const unsigned int count = min(unsigned(material.y), CYCLES_WEBGPU_MAX_ORDERED_LOBES);
  float total = 0.0f;
  float sample_weights[CYCLES_WEBGPU_MAX_ORDERED_LOBES];
  for (unsigned int index = 0; index < count; index++) {
    const auto color_type = cycles_material_lobe(ordered_closure_address(material, index));
    const auto type = unsigned(color_type.w) & ~CYCLES_WEBGPU_ORDERED_LOBE_BUMP;
    sample_weights[index] = ordered_closure_sample_weight(rgb(color_type), type);
    total += sample_weights[index];
  }
  if (!(total > 0.0f)) {
    return LABEL_NONE;
  }

  /* Match surface_shader_bsdf_bssrdf_pick, including the first-closure
   * fallback if the random value rounds to the end of the interval. */
  unsigned int sampled = 0;
  float3 rand = rgb(random);
  if (count > 1) {
    const float selector = rand.z * total;
    float partial = 0.0f;
    for (unsigned int index = 0; index < count; index++) {
      const float next = partial + sample_weights[index];
      if (selector < next) {
        sampled = index;
        rand.z = (selector - partial) / sample_weights[index];
        break;
      }
      partial = next;
    }
  }

  const OrderedClosure selected = read_ordered_closure(material, sampled);
  float3 wo = zero_float3();
  Spectrum eval = zero_spectrum();
  float pdf = 0.0f;
  float roughness_squared = 1.0f;
  int label;
  if (selected.type == CLOSURE_BSDF_TRANSPARENT_ID) {
    label = bsdf_transparent_sample(nullptr, rgb(geometric), rgb(wi), &eval, &wo, &pdf);
    roughness_squared = 0.0f;
  }
  else if (selected.type == CLOSURE_BSDF_TRANSLUCENT_ID) {
    const GgxFloat4 corrected = cycles_bump_specular_normal(
        selected.bump, geometric, wi, specular_normal);
    const DiffuseBsdf translucent{.N = rgb(corrected)};
    const auto *closure = reinterpret_cast<const ShaderClosure *>(&translucent);
    /* bsdf_sample() only applies bump shadowing to reflected samples. */
    label = bsdf_translucent_sample(closure, rgb(geometric), rgb(wi), make_float2(rand.x, rand.y),
                                    &eval, &wo, &pdf);
  }
  else if (selected.diffuse()) {
    const DiffuseBsdf diffuse{.N = rgb(selected.bump.w != 0.0f ? selected.bump : normal)};
    const auto *closure = reinterpret_cast<const ShaderClosure *>(&diffuse);
    label = bsdf_diffuse_sample(closure, rgb(geometric), rgb(wi), make_float2(rand.x, rand.y),
                                &eval, &wo, &pdf);
    eval *= cycles_bump_shadowing(normal, selected.bump, vector(wo), false);
  }
  else {
    MicrofacetBsdf bsdf{};
    setup_ordered_glossy(bsdf, selected, geometric, wi, specular_normal, blur_alpha);
    label = sample_mixed_glossy(bsdf, rgb(geometric), rgb(wi), rand, eval, wo, pdf);
    roughness_squared = bsdf.alpha_x * bsdf.alpha_y;
  }
  if (!(pdf > 0.0f)) {
    return LABEL_NONE;
  }

  /* The sampled BSDF contributes its sample evaluation, while other BSDFs
   * contribute their evaluations at the generated direction. Match the
   * stock MIS accumulator's order and relative sample weights. */
  OrderedClosureEval result;
  const Spectrum weighted = eval * rgb(selected.color_alpha);
  result.add({weighted.x, weighted.y, weighted.z, pdf}, sample_weights[sampled], roughness_squared,
             selected.diffuse());
  const GgxFloat4 out_direction = vector(wo);
  for (unsigned int index = 0; index < count; index++) {
    if (index != sampled) {
      accumulate_ordered_closure(result, read_ordered_closure(material, index),
                                 normal, geometric, specular_normal, wi, out_direction, blur_alpha,
                                 material.z != 0.0f);
    }
  }
  const float mixture_pdf = safe_divide(result.pdf, result.sample_weight);
  if (!(mixture_pdf > 0.0f) || is_zero(result.eval)) {
    return LABEL_NONE;
  }
  diffuse_ratio = vector(safe_divide(result.diffuse, result.eval));
  const Spectrum throughput = result.eval / mixture_pdf;
  weight = {throughput.x, throughput.y, throughput.z, mixture_pdf};
  out = {wo.x, wo.y, wo.z, safe_divide(result.pdf_roughness_squared, result.pdf)};
  return label;
}

}  // namespace
}  // namespace ccl
#endif
#endif

#include "ordered_principled.h"

[[gnu::always_inline]] GgxFloat4 cycles_standalone_eval(GgxFloat4 material,
                                                       GgxFloat4 parameters,
                                                       GgxFloat4 secondary,
                                                       GgxFloat4 extra,
                                                       GgxFloat4 hair,
                                                       GgxFloat4 ordered_layers,
                                                       GgxFloat4 normal,
                                                       GgxFloat4 geometric_normal,
                                                       GgxFloat4 incoming,
                                                       GgxFloat4 outgoing,
                                                       float blur_alpha,
                                                       unsigned int &lcg_state,
                                                       GgxFloat4 &diffuse_ratio)
{
  using namespace ccl;
  const bool glossy = CYCLES_WEBGPU_MATERIAL_IS_GLOSSY(material.w) ||
                      material.w == CYCLES_WEBGPU_MATERIAL_GLOSSY_TOON ||
                      material.w == CYCLES_WEBGPU_MATERIAL_VELVET;
  diffuse_ratio = vector(make_float3(glossy ? 0.0f : 1.0f));
  const float3 N = rgb(normal);
  const float3 wi = rgb(incoming);
  const float3 wo = rgb(outgoing);
#ifndef CYCLES_WEBGPU_EXCLUDE_STANDALONE
#ifdef CYCLES_WEBGPU_ORDERED_CLOSURES
  if (material.w == CYCLES_WEBGPU_MATERIAL_ORDERED_PRINCIPLED) {
    return eval_ordered_principled(material, parameters, secondary, ordered_layers,
                                   normal, geometric_normal, extra,
                                   incoming, outgoing, blur_alpha, diffuse_ratio);
  }
  if (CYCLES_WEBGPU_MATERIAL_HAS_ORDERED_CLOSURES(material.w)) {
    return evaluate_ordered_closures(material, normal, geometric_normal, extra,
                                     incoming, outgoing, blur_alpha, diffuse_ratio);
  }
#endif
  if (material.w == CYCLES_WEBGPU_MATERIAL_DIFFUSE_GLOSSY) {
    const float total = secondary.x + secondary.y;
    if (!(total > 0.0f)) {
      return {};
    }
    const GgxFloat4 diffuse = evaluate_mixed_glossy_diffuse(
        material, secondary.x, normal, hair, incoming, outgoing);
    GgxFloat4 glossy{};
    if (secondary.y > 0.0f) {
      MicrofacetBsdf bsdf{};
      setup_mixed_glossy(bsdf, parameters, extra, uint(secondary.w), blur_alpha);
      glossy = evaluate_mixed_glossy(bsdf, N, wi, wo);
    }
    const Spectrum eval = rgb(diffuse) + rgb(glossy);
    diffuse_ratio = vector(safe_divide(rgb(diffuse), eval));
    const float pdf = (diffuse.w * secondary.x + glossy.w * secondary.y) / total;
    return {eval.x, eval.y, eval.z, pdf};
  }
  if (material.w == CYCLES_WEBGPU_MATERIAL_DIFFUSE_TRANSLUCENT) {
    const float total = secondary.x + secondary.y;
    if (!(total > 0.0f)) {
      return {};
    }
    const GgxFloat4 diffuse = evaluate_diffuse_translucent_lobe<false>(
        material, secondary.x, normal, extra, incoming, outgoing, secondary.w != 0.0f);
    const GgxFloat4 translucent = evaluate_diffuse_translucent_lobe<true>(
        parameters, secondary.y, normal, extra, incoming, outgoing, secondary.w != 0.0f);
    const Spectrum eval = rgb(diffuse) + rgb(translucent);
    const float pdf = (diffuse.w * secondary.x + translucent.w * secondary.y) / total;
    return {eval.x, eval.y, eval.z, pdf};
  }
#endif
#if !defined(CYCLES_WEBGPU_EXCLUDE_STANDALONE) && !defined(CYCLES_WEBGPU_EXCLUDE_HAIR)
  if (CYCLES_WEBGPU_MATERIAL_IS_HAIR(material.w)) {
    return with_hair_closure(
        material, parameters, secondary, extra, hair, N, rgb(geometric_normal), wi, blur_alpha,
        [&](ShaderData &sd, const auto &bsdf, const int flags) __attribute__((always_inline)) {
          if (!flags || dot(N, wo) * dot(N, bsdf.N) * dot(bsdf.N, wo) < 0.0f) {
            return GgxFloat4{};
          }
          using Bsdf = std::remove_cvref_t<decltype(bsdf)>;
          const auto *sc = reinterpret_cast<const ShaderClosure *>(&bsdf);
          float pdf = 0.0f;
          Spectrum eval;
          if constexpr (std::is_same_v<Bsdf, ChiangHairBSDF>) {
            GgxGlobals globals{};
            globals.data.film.rgb_to_y = make_float4(hair.x, hair.y, hair.z, 0.0f);
            eval = bsdf_hair_chiang_eval(&globals, &sd, sc, wo, &pdf);
          }
          else if constexpr (std::is_same_v<Bsdf, HuangHairBSDF>) {
            eval = bsdf_hair_huang_eval(&energy_globals, &sd, sc, wo, &pdf);
            lcg_state = sd.lcg_state;
          }
          else {
            eval = bsdf.type == CLOSURE_BSDF_HAIR_REFLECTION_ID ?
                       bsdf_hair_reflection_eval(sc, wi, wo, &pdf) :
                       bsdf_hair_transmission_eval(sc, wi, wo, &pdf);
          }
          eval *= bsdf.weight;
          return GgxFloat4{eval.x, eval.y, eval.z, pdf};
        });
  }
#endif
  if (CYCLES_WEBGPU_MATERIAL_IS_GLOSSY(material.w)) {
    return with_standalone_glossy(
        material, parameters, secondary, extra, N, wi, blur_alpha, [&](const MicrofacetBsdf &bsdf) {
          float pdf = 0.0f;
          Spectrum eval;
          const auto *closure = reinterpret_cast<const ShaderClosure *>(&bsdf);
          if (bsdf.type == CLOSURE_BSDF_ASHIKHMIN_SHIRLEY_ID) {
            eval = bsdf_ashikhmin_shirley_eval(closure, wi, wo, &pdf);
          }
          else if (bsdf.type == CLOSURE_BSDF_MICROFACET_BECKMANN_ID) {
            eval = bsdf_microfacet_beckmann_eval(&energy_globals, closure, wi, wo, &pdf);
          }
          else {
            eval = bsdf_microfacet_ggx_eval(&energy_globals, closure, wi, wo, &pdf);
          }
          eval *= bsdf.weight;
          if (material.w == CYCLES_WEBGPU_MATERIAL_GLOSSY_TRANSPARENT) {
            /* Transparent contributes zero directional density to the NEE direction. */
            pdf *= safe_divide(bsdf.sample_weight, bsdf.sample_weight + fabsf(hair.w));
          }
          return GgxFloat4{eval.x, eval.y, eval.z, pdf};
        });
  }
  float pdf;
  Spectrum eval;
  if (material.w == CYCLES_WEBGPU_MATERIAL_TRANSLUCENT) {
    const DiffuseBsdf bsdf{.N = N};
    eval = bsdf_translucent_eval(reinterpret_cast<const ShaderClosure *>(&bsdf), wi, wo, &pdf);
  }
  else if (material.w == CYCLES_WEBGPU_MATERIAL_VELVET) {
    VelvetBsdf bsdf{.N = N, .sigma = parameters.x};
    bsdf_ashikhmin_velvet_setup(&bsdf);
    eval = bsdf_ashikhmin_velvet_eval(
        reinterpret_cast<const ShaderClosure *>(&bsdf), wi, wo, &pdf);
  }
  else if (material.w == CYCLES_WEBGPU_MATERIAL_SHEEN) {
    SheenBsdf bsdf{.N = N, .transformA = parameters.x, .transformB = parameters.y};
    make_orthonormals_safe_tangent(N, wi, &bsdf.T, &bsdf.B);
    eval = bsdf_sheen_eval(reinterpret_cast<const ShaderClosure *>(&bsdf), wi, wo, &pdf);
  }
  else {
    ToonBsdf bsdf{.N = N, .size = parameters.x, .smooth = parameters.y};
    const auto *closure = reinterpret_cast<const ShaderClosure *>(&bsdf);
    if (material.w == CYCLES_WEBGPU_MATERIAL_DIFFUSE_TOON) {
      bsdf_diffuse_toon_setup(&bsdf);
      eval = bsdf_diffuse_toon_eval(closure, wi, wo, &pdf);
    }
    else {
      bsdf_glossy_toon_setup(&bsdf);
      eval = bsdf_glossy_toon_eval(closure, wi, wo, &pdf);
    }
  }
  eval *= rgb(material);
  return {eval.x, eval.y, eval.z, pdf};
}

[[gnu::always_inline]] unsigned int cycles_standalone_sample(GgxFloat4 material,
                                                             GgxFloat4 parameters,
                                                             GgxFloat4 secondary,
                                                             GgxFloat4 extra,
                                                             GgxFloat4 hair,
                                                             GgxFloat4 ordered_layers,
                                                             GgxFloat4 normal,
                                                             GgxFloat4 geometric_normal,
                                                             GgxFloat4 incoming,
                                                             GgxFloat4 random,
                                                             float blur_alpha,
                                                             GgxFloat4 &weight,
                                                             GgxFloat4 &out,
                                                             GgxFloat4 &diffuse_ratio)
{
  using namespace ccl;
  const bool glossy = CYCLES_WEBGPU_MATERIAL_IS_GLOSSY(material.w) ||
                      material.w == CYCLES_WEBGPU_MATERIAL_GLOSSY_TOON ||
                      material.w == CYCLES_WEBGPU_MATERIAL_VELVET;
  diffuse_ratio = vector(make_float3(glossy ? 0.0f : 1.0f));
  const float3 N = rgb(normal);
  const float3 Ng = rgb(geometric_normal);
  const float3 wi = rgb(incoming);
  const float2 rand = make_float2(random.x, random.y);
#ifndef CYCLES_WEBGPU_EXCLUDE_STANDALONE
#ifdef CYCLES_WEBGPU_ORDERED_CLOSURES
  if (material.w == CYCLES_WEBGPU_MATERIAL_ORDERED_PRINCIPLED) {
    return sample_ordered_principled(material, parameters, secondary, ordered_layers,
                                     normal, geometric_normal, extra,
                                     incoming, random, blur_alpha, weight, out, diffuse_ratio);
  }
  if (CYCLES_WEBGPU_MATERIAL_HAS_ORDERED_CLOSURES(material.w)) {
    return sample_ordered_closures(material, normal, geometric_normal, extra,
                                   incoming, random, blur_alpha, weight, out, diffuse_ratio);
  }
#endif
  if (material.w == CYCLES_WEBGPU_MATERIAL_DIFFUSE_GLOSSY) {
    const float total = secondary.x + secondary.y;
    if (!(total > 0.0f)) {
      return LABEL_NONE;
    }
    const bool glossy_first = (uint(secondary.z) & CYCLES_WEBGPU_MIXED_GLOSSY_FIRST) != 0u;
    const float first = glossy_first ? secondary.y : secondary.x;
    const float selector = random.z * total;
    const bool pick_first = selector < first || selector >= total;
    const bool pick_glossy = !(secondary.x > 0.0f) ||
                             (secondary.y > 0.0f && pick_first == glossy_first);
    float3 wo = zero_float3();
    float pdf = 0.0f;
    Spectrum eval = zero_spectrum();
    int label;
    MicrofacetBsdf bsdf{};
    setup_mixed_glossy(bsdf, parameters, extra, uint(secondary.w), blur_alpha);
    if (pick_glossy) {
      float z = random.z;
      if (secondary.x > 0.0f && selector < total) {
        z = (selector - (glossy_first ? 0.0f : first)) / secondary.y;
      }
      const float3 rand_lobe = make_float3(random.x, random.y, z);
      label = sample_mixed_glossy(bsdf, Ng, wi, rand_lobe, eval, wo, pdf);
    }
    else {
      const DiffuseBsdf diffuse{.N = hair.w != 0.0f ? rgb(hair) : N};
      const auto *closure = reinterpret_cast<const ShaderClosure *>(&diffuse);
      label = bsdf_diffuse_sample(closure, Ng, wi, rand, &eval, &wo, &pdf);
      eval *= cycles_bump_shadowing(normal, hair, vector(wo), false);
    }
    if (!(pdf > 0.0f)) {
      return LABEL_NONE;
    }
    const GgxFloat4 diffuse = pick_glossy ? evaluate_mixed_glossy_diffuse(
        material, secondary.x, normal, hair, incoming, vector(wo)) :
        GgxFloat4{eval.x * material.x, eval.y * material.y, eval.z * material.z, pdf};
    GgxFloat4 glossy{};
    if (pick_glossy) {
      glossy = {eval.x * parameters.x, eval.y * parameters.y, eval.z * parameters.z, pdf};
    }
    else if (secondary.y > 0.0f) {
      glossy = evaluate_mixed_glossy(bsdf, N, wi, wo);
    }
    const float diffuse_pdf = diffuse.w * secondary.x;
    const float glossy_pdf = glossy.w * secondary.y;
    const float weighted_pdf = diffuse_pdf + glossy_pdf;
    const float mixture_pdf = weighted_pdf / total;
    const Spectrum scattering = rgb(diffuse) + rgb(glossy);
    if (!(mixture_pdf > 0.0f) || is_zero(scattering)) {
      return LABEL_NONE;
    }
    diffuse_ratio = vector(safe_divide(rgb(diffuse), scattering));
    const Spectrum throughput = scattering / mixture_pdf;
    weight = {throughput.x, throughput.y, throughput.z, mixture_pdf};
    out = {wo.x, wo.y, wo.z, (diffuse_pdf + glossy_pdf * bsdf.alpha_x * bsdf.alpha_y) /
                              weighted_pdf};
    return label;
  }
  if (material.w == CYCLES_WEBGPU_MATERIAL_DIFFUSE_TRANSLUCENT) {
    const float total = secondary.x + secondary.y;
    if (!(total > 0.0f)) {
      return LABEL_NONE;
    }
    /* surface_shader_bsdf_bssrdf_pick() chooses by the stock closure sample
     * weight in SVM emission order. These cosine lobes only use rand.xy. */
    const bool translucent_first = secondary.z != 0.0f;
    const float first_weight = translucent_first ? secondary.y : secondary.x;
    const float selector = random.z * total;
    const bool pick_first = selector < first_weight || selector >= total;
    const bool transmit = !(secondary.x > 0.0f) ||
                          (secondary.y > 0.0f && pick_first == translucent_first);

    const DiffuseBsdf bsdf{.N = transmit ? rgb(extra) : N};
    const auto *closure = reinterpret_cast<const ShaderClosure *>(&bsdf);
    float3 wo = zero_float3();
    float pdf = 0.0f;
    Spectrum eval = zero_spectrum();
    const int label = transmit ?
                          bsdf_translucent_sample(closure, Ng, wi, rand, &eval, &wo, &pdf) :
                          bsdf_diffuse_sample(closure, Ng, wi, rand, &eval, &wo, &pdf);
    if (!(pdf > 0.0f) || is_zero(eval)) {
      return LABEL_NONE;
    }

    /* Use the selected closure's sample evaluation verbatim. Other lobes
     * use bsdf_eval(), matching surface_shader_bsdf_sample_closure(). */
    const GgxFloat4 outgoing = vector(wo);
    const bool bump_correction = secondary.w != 0.0f;
    const GgxFloat4 other = transmit ?
        evaluate_diffuse_translucent_lobe<false>(
            material, secondary.x, normal, extra, incoming, outgoing, bump_correction) :
        evaluate_diffuse_translucent_lobe<true>(
            parameters, secondary.y, normal, extra, incoming, outgoing, bump_correction);
    eval *= rgb(transmit ? parameters : material);
    eval += rgb(other);
    const float selected_weight = transmit ? secondary.y : secondary.x;
    const float other_weight = transmit ? secondary.x : secondary.y;
    const float mixture_pdf = (pdf * selected_weight + other.w * other_weight) / total;
    const Spectrum throughput = eval / mixture_pdf;
    weight = {throughput.x, throughput.y, throughput.z, mixture_pdf};
    out = outgoing;
    return label;
  }
#endif
#if !defined(CYCLES_WEBGPU_EXCLUDE_STANDALONE) && !defined(CYCLES_WEBGPU_EXCLUDE_HAIR)
  if (CYCLES_WEBGPU_MATERIAL_IS_HAIR(material.w)) {
    return with_hair_closure(
        material, parameters, secondary, extra, hair, N, Ng, wi, blur_alpha,
        [&](ShaderData &sd, const auto &bsdf, const int flags) __attribute__((always_inline)) {
          if (!flags) {
            return int(LABEL_NONE);
          }
          using Bsdf = std::remove_cvref_t<decltype(bsdf)>;
          const auto *sc = reinterpret_cast<const ShaderClosure *>(&bsdf);
          float3 wo = zero_float3();
          float2 sampled_roughness;
          float pdf = 0.0f;
          Spectrum eval = zero_spectrum();
          int label;
          if constexpr (std::is_same_v<Bsdf, ChiangHairBSDF>) {
            GgxGlobals globals{};
            globals.data.film.rgb_to_y = make_float4(hair.x, hair.y, hair.z, 0.0f);
            label = bsdf_hair_chiang_sample(&globals, sc, &sd, rgb(random), &eval, &wo,
                                          &pdf, &sampled_roughness);
          }
          else if constexpr (std::is_same_v<Bsdf, HuangHairBSDF>) {
            label = bsdf_hair_huang_sample(&energy_globals, sc, &sd, rgb(random), &eval, &wo,
                                          &pdf, &sampled_roughness);
          }
          else {
            label = bsdf.type == CLOSURE_BSDF_HAIR_REFLECTION_ID ?
                        bsdf_hair_reflection_sample(sc, Ng, wi, rand, &eval, &wo, &pdf,
                                                    &sampled_roughness) :
                        bsdf_hair_transmission_sample(sc, Ng, wi, rand, &eval, &wo, &pdf,
                                                      &sampled_roughness);
          }
          eval *= bsdf.weight;
          if (!(pdf > 0.0f) || is_zero(eval)) {
            return int(LABEL_NONE);
          }
          const Spectrum sample_weight = eval / pdf;
          weight = {sample_weight.x, sample_weight.y, sample_weight.z, pdf};
          out = {wo.x, wo.y, wo.z, 1.0f};
          return label;
        });
  }
#endif
  if (CYCLES_WEBGPU_MATERIAL_IS_GLOSSY(material.w)) {
    struct Sample {
      GgxFloat4 weight;
      GgxFloat4 outgoing;
      int label;
    };
    const Sample result = with_standalone_glossy(
        material, parameters, secondary, extra, N, wi, blur_alpha,
        [&](const MicrofacetBsdf &bsdf) __attribute__((always_inline)) {
          float3 wo;
          float2 sampled_roughness;
          float pdf = 0.0f;
          float eta;
          Spectrum eval;
          int label;
          const auto *closure = reinterpret_cast<const ShaderClosure *>(&bsdf);
          if (bsdf.type == CLOSURE_BSDF_ASHIKHMIN_SHIRLEY_ID) {
            label = bsdf_ashikhmin_shirley_sample(
                closure, Ng, wi, rand, &eval, &wo, &pdf, &sampled_roughness);
          }
          else if (bsdf.type == CLOSURE_BSDF_MICROFACET_BECKMANN_ID) {
            label = bsdf_microfacet_beckmann_sample(
                &energy_globals, closure, Ng, wi, rgb(random), &eval, &wo, &pdf, &sampled_roughness, &eta);
          }
          else {
            label = bsdf_microfacet_ggx_sample(
                &energy_globals, closure, Ng, wi, rgb(random), &eval, &wo, &pdf, &sampled_roughness, &eta);
          }
          eval *= bsdf.weight;
          if (!(pdf > 0.0f) || is_zero(eval)) {
            return Sample{};
          }
          const Spectrum sample_weight = eval / pdf;
          return Sample{{sample_weight.x, sample_weight.y, sample_weight.z, pdf},
                        {wo.x, wo.y, wo.z, bsdf.alpha_x * bsdf.alpha_y},
                        label};
        });
    weight = result.weight;
    out = result.outgoing;
    return result.label;
  }
  float3 wo;
  float pdf;
  Spectrum eval;
  int label;
  if (material.w == CYCLES_WEBGPU_MATERIAL_TRANSLUCENT) {
    const DiffuseBsdf bsdf{.N = N};
    label = bsdf_translucent_sample(
        reinterpret_cast<const ShaderClosure *>(&bsdf), Ng, wi, rand, &eval, &wo, &pdf);
  }
  else if (material.w == CYCLES_WEBGPU_MATERIAL_VELVET) {
    VelvetBsdf bsdf{.N = N, .sigma = parameters.x};
    bsdf_ashikhmin_velvet_setup(&bsdf);
    label = bsdf_ashikhmin_velvet_sample(
        reinterpret_cast<const ShaderClosure *>(&bsdf), Ng, wi, rand, &eval, &wo, &pdf);
  }
  else if (material.w == CYCLES_WEBGPU_MATERIAL_SHEEN) {
    SheenBsdf bsdf{.N = N, .transformA = parameters.x, .transformB = parameters.y};
    make_orthonormals_safe_tangent(N, wi, &bsdf.T, &bsdf.B);
    label = bsdf_sheen_sample(
        reinterpret_cast<const ShaderClosure *>(&bsdf), Ng, wi, rand, &eval, &wo, &pdf);
  }
  else {
    ToonBsdf bsdf{.N = N, .size = parameters.x, .smooth = parameters.y};
    const auto *closure = reinterpret_cast<const ShaderClosure *>(&bsdf);
    if (material.w == CYCLES_WEBGPU_MATERIAL_DIFFUSE_TOON) {
      bsdf_diffuse_toon_setup(&bsdf);
      label = bsdf_diffuse_toon_sample(closure, Ng, wi, rand, &eval, &wo, &pdf);
    }
    else {
      bsdf_glossy_toon_setup(&bsdf);
      label = bsdf_glossy_toon_sample(closure, Ng, wi, rand, &eval, &wo, &pdf);
    }
  }
  if (!(pdf > 0.0f) || is_zero(eval)) {
    return LABEL_NONE;
  }
  const Spectrum sample_weight = rgb(material) * eval / pdf;
  weight = {sample_weight.x, sample_weight.y, sample_weight.z, pdf};
  out = vector(wo);
  return label;
}

#ifdef CYCLES_WEBGPU_DENOISE_SURFACE
[[gnu::always_inline]] void cycles_surface_denoise_guides(
    GgxFloat4 material,
    GgxFloat4 tint_alpha,
    GgxFloat4 weights,
    GgxFloat4 glass,
    GgxFloat4 hair,
    GgxFloat4 normal,
    GgxFloat4 specular_normal,
    GgxFloat4 geometric_normal,
    GgxFloat4 bump,
    GgxFloat4 view,
    float blur_alpha,
    unsigned int options,
    GgxFloat4 feature_throughput,
    GgxFloat4 &next_feature_throughput,
    GgxFloat4 &albedo,
    GgxFloat4 &guide_normal)
{
  using namespace ccl;
  DenoisingFeatureAccumulator features;
  const uint mixed_flags = material.w == CYCLES_WEBGPU_MATERIAL_DIFFUSE_GLOSSY ?
                               uint(weights.z) : 0u;
  const bool bump_diffuse = bump.w != 0.0f && !(mixed_flags & CYCLES_WEBGPU_MIXED_BUMP_GLOSSY_ONLY);
  const float3 N = bump_diffuse ? rgb(bump) : rgb(normal);
  const float3 shader_specular_N = rgb(specular_normal);
  const GgxFloat4 glossy_bump = cycles_mixed_glossy_bump(bump, hair, mixed_flags);
  specular_normal = cycles_bump_specular_normal(
      glossy_bump, geometric_normal, view, specular_normal);
  const bool roughness_weighting = (options & DENOISING_PASS_USE_ALBEDO_ROUGHNESS_WEIGHTING) != 0;

  if (material.w == CYCLES_WEBGPU_MATERIAL_TRANSPARENT) {
    features.transparent_albedo = rgb(material);
  }
  else if (material.w == CYCLES_WEBGPU_MATERIAL_RAY_PORTAL) {
    features.add(rgb(material), N, 0.0f, CLOSURE_BSDF_RAY_PORTAL_ID, roughness_weighting);
  }
  else if (material.w == 1.0f || material.w == CYCLES_WEBGPU_MATERIAL_MIXED_TRANSPARENT) {
    if (material.w == CYCLES_WEBGPU_MATERIAL_MIXED_TRANSPARENT) {
      features.transparent_albedo = rgb(tint_alpha);
    }
    Spectrum diffuse = rgb(material);
    if (closure_sample_weight(0, diffuse) > 0.0f) {
      features.add(diffuse, N, 1.0f, CLOSURE_BSDF_DIFFUSE_ID, roughness_weighting);
    }
  }
#ifndef CYCLES_WEBGPU_EXCLUDE_STANDALONE
#ifdef CYCLES_WEBGPU_ORDERED_CLOSURES
  else if (material.w == CYCLES_WEBGPU_MATERIAL_ORDERED_PRINCIPLED) {
    for (uint index = 0; index < 2u; ++index) {
      const auto branch = read_ordered_principled_branch(
          material, index, index == 0u ? tint_alpha : weights, glass, geometric_normal,
          specular_normal, view, blur_alpha);
      if (branch.sheen.w > 0.0f) {
        features.add(make_spectrum(branch.sheen.w), ordered_principled_sheen_normal(branch, normal),
                     1.0f, CLOSURE_BSDF_SHEEN_ID, roughness_weighting);
      }
      for_ordered_principled_glossy(
          branch, [&](const MicrofacetBsdf &bsdf) __attribute__((always_inline)) {
            const Spectrum albedo = bsdf.weight * bsdf_microfacet_estimate_albedo(
                &energy_globals, rgb(view), &bsdf, true, true);
            features.add(albedo, bsdf.N, sqrtf(bsdf.alpha_x * bsdf.alpha_y), bsdf.type,
                         roughness_weighting);
          });
      if (branch.closures.has_diffuse) {
        const float3 diffuse_N = rgb(branch.bump.w != 0.0f ? branch.bump : normal);
        features.add(branch.closures.diffuse_weight, diffuse_N, 1.0f, CLOSURE_BSDF_DIFFUSE_ID,
                     roughness_weighting);
      }
    }
  }
  else if (CYCLES_WEBGPU_MATERIAL_HAS_ORDERED_CLOSURES(material.w)) {
    features.transparent_albedo = rgb(hair);
    const unsigned int count = min(unsigned(material.y), CYCLES_WEBGPU_MAX_ORDERED_LOBES);
    for (unsigned int index = 0; index < count; index++) {
      const OrderedClosure lobe = read_ordered_closure(material, index);
      if (lobe.type == CLOSURE_BSDF_TRANSPARENT_ID) {
        continue;
      }
      if (lobe.diffuse()) {
        const GgxFloat4 lobe_N = lobe.type == CLOSURE_BSDF_TRANSLUCENT_ID ?
                                    cycles_bump_specular_normal(
                                        lobe.bump, geometric_normal, view, specular_normal) :
                                lobe.bump.w != 0.0f ? lobe.bump : normal;
        features.add(rgb(lobe.color_alpha), rgb(lobe_N), 1.0f, ClosureType(lobe.type),
                     roughness_weighting);
      }
      else {
        MicrofacetBsdf bsdf{};
        setup_ordered_glossy(bsdf, lobe, geometric_normal, view, specular_normal, blur_alpha);
        const Spectrum albedo = bsdf.weight * bsdf_microfacet_estimate_albedo(
            &energy_globals, rgb(view), &bsdf, true, true);
        features.add(albedo, bsdf.N, sqrtf(bsdf.alpha_x * bsdf.alpha_y), bsdf.type,
                     roughness_weighting);
      }
    }
  }
#endif
  else if (material.w == CYCLES_WEBGPU_MATERIAL_DIFFUSE_GLOSSY) {
    if (weights.x > 0.0f) {
      features.add(rgb(material), N, 1.0f, CLOSURE_BSDF_DIFFUSE_ID, roughness_weighting);
    }
    if (weights.y > 0.0f) {
      MicrofacetBsdf bsdf{};
      setup_mixed_glossy(bsdf, tint_alpha, specular_normal, uint(weights.w), blur_alpha);
      const Spectrum albedo = bsdf.weight * bsdf_microfacet_estimate_albedo(
          &energy_globals, rgb(view), &bsdf, true, true);
      features.add(albedo, rgb(specular_normal), sqrtf(bsdf.alpha_x * bsdf.alpha_y),
                   bsdf.type, roughness_weighting);
    }
  }
  else if (material.w == CYCLES_WEBGPU_MATERIAL_DIFFUSE_TRANSLUCENT) {
    if (weights.x > 0.0f) {
      features.add(rgb(material), N, 1.0f, CLOSURE_BSDF_DIFFUSE_ID, roughness_weighting);
    }
    if (weights.y > 0.0f) {
      features.add(rgb(tint_alpha), rgb(specular_normal), 1.0f, CLOSURE_BSDF_TRANSLUCENT_ID,
                   roughness_weighting);
    }
  }
  else if (CYCLES_WEBGPU_MATERIAL_IS_GLOSSY(material.w)) {
    if (material.w == CYCLES_WEBGPU_MATERIAL_GLOSSY_TRANSPARENT) {
      features.transparent_albedo = rgb(hair);
    }
    const float3 wi = rgb(view);
    const GgxFloat4 albedo_roughness = with_standalone_glossy(
        material, tint_alpha, weights, glass, rgb(specular_normal), wi, blur_alpha,
        [&](const MicrofacetBsdf &bsdf) __attribute__((always_inline)) {
          const Spectrum estimate = bsdf_microfacet_estimate_albedo(
              &energy_globals, wi, &bsdf, true, true);
          const Spectrum albedo = bsdf.weight * estimate;
          return GgxFloat4{albedo.x, albedo.y, albedo.z, sqrtf(bsdf.alpha_x * bsdf.alpha_y)};
        });
    features.add(rgb(albedo_roughness), rgb(specular_normal), albedo_roughness.w,
                 CLOSURE_BSDF_MICROFACET_GGX_ID, roughness_weighting);
  }
  else if (CYCLES_WEBGPU_MATERIAL_IS_HAIR(material.w)) {
    const float3 closure_normal = CYCLES_WEBGPU_MATERIAL_USES_SPECULAR_NORMAL(material.w) ?
                                     rgb(specular_normal) : N;
    with_hair_closure(material, tint_alpha, weights, glass, hair, closure_normal,
        rgb(geometric_normal), rgb(view), blur_alpha,
        [&](ShaderData &sd, const auto &bsdf, const int flags) __attribute__((always_inline)) {
          if (!flags) {
            features.transparent_albedo = bsdf.weight;
            return 0;
          }
          using Bsdf = std::remove_cvref_t<decltype(bsdf)>;
          const auto *sc = reinterpret_cast<const ShaderClosure *>(&bsdf);
          Spectrum albedo = bsdf.weight;
          if constexpr (std::is_same_v<Bsdf, ChiangHairBSDF>) {
            albedo *= bsdf_hair_chiang_albedo(&sd, sc);
          }
          else if constexpr (std::is_same_v<Bsdf, HuangHairBSDF>) {
            albedo *= bsdf_hair_huang_albedo(&sd, sc);
          }
          features.add(albedo, bsdf.N, 1.0f, bsdf.type, roughness_weighting);
          return 0;
        });
  }
  else if (CYCLES_WEBGPU_MATERIAL_IS_STANDALONE(material.w)) {
    const bool translucent = material.w == CYCLES_WEBGPU_MATERIAL_TRANSLUCENT;
    const ClosureType type = translucent ? CLOSURE_BSDF_TRANSLUCENT_ID :
                             material.w == CYCLES_WEBGPU_MATERIAL_GLOSSY_TOON ?
                                 CLOSURE_BSDF_GLOSSY_TOON_ID : CLOSURE_BSDF_DIFFUSE_ID;
    Spectrum albedo = rgb(material);
    if (closure_sample_weight(0, albedo) > 0.0f) {
      features.add(albedo, translucent ? rgb(specular_normal) : N, 1.0f, type, roughness_weighting);
    }
  }
#endif
  else if (material.w == CYCLES_WEBGPU_MATERIAL_GGX ||
           material.w == CYCLES_WEBGPU_MATERIAL_MULTISCATTER_GGX)
  {
    if (CYCLES_WEBGPU_GGX_CAN_BE_TRANSPARENT(glass) && hair.w != 0.0f) {
      features.transparent_albedo = rgb(hair);
    }
    const float3 specular_N = rgb(specular_normal);
    const bool has_translucent = CYCLES_WEBGPU_GGX_HAS_TRANSLUCENT(glass);
    if (has_translucent) {
      features.add(rgb(glass), specular_N, 1.0f, CLOSURE_BSDF_TRANSLUCENT_ID, roughness_weighting);
      glass = {};
    }
    const float3 wi = rgb(view);
    MicrofacetBsdf coat_bsdf{};
    if (setup_compact_white_coat(coat_bsdf, tint_alpha, weights, glass, specular_N,
                                shader_specular_N, wi, blur_alpha))
    {
      const Spectrum albedo = coat_bsdf.weight * bsdf_microfacet_estimate_albedo(
          &energy_globals, wi, &coat_bsdf, true, true);
      features.add(albedo, coat_bsdf.N, coat_bsdf.alpha_x, coat_bsdf.type, roughness_weighting);
    }
    FresnelGeneralizedSchlick glass_fresnel{};
    MicrofacetBsdf glass_bsdf{.type = CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID,
                             .fresnel_type = MicrofacetFresnel::GENERALIZED_SCHLICK,
                             .fresnel = &glass_fresnel};
    if (setup_glass(glass_bsdf,
                    glass_fresnel,
                    material,
                    tint_alpha,
                    weights,
                    glass,
                    hair,
                    GgxFloat4{specular_N.x, specular_N.y, specular_N.z, normal.w},
                    wi))
    {
      bsdf_microfacet_blur(reinterpret_cast<ShaderClosure *>(&glass_bsdf), blur_alpha);
      const Spectrum estimate = bsdf_microfacet_estimate_albedo(
          &energy_globals, wi, &glass_bsdf, true, true);
      features.add(glass_bsdf.weight * estimate,
                   specular_N,
                   sqrtf(glass_bsdf.alpha_x * glass_bsdf.alpha_y),
                   glass_bsdf.type,
                   roughness_weighting);
    }
    if (glass.x > CLOSURE_WEIGHT_CUTOFF) {
      weights.w *= 1.0f - glass.x;
    }

    ReflectiveClosures closures;
    setup_closures(closures, material, tint_alpha, weights, glass, hair, specular_N, wi);
    blur_reflective_ggx(closures, blur_alpha);
    const float alpha = sqrtf(closures.alpha_x * closures.alpha_y);
    if (closures.has_metal) {
      FresnelF82Tint fresnel = {.thin_film = {}, .f0 = closures.metal_f0, .b = closures.metal_tint_b};
      const MicrofacetBsdf bsdf{.type = CLOSURE_BSDF_MICROFACET_GGX_ID,
                                .N = specular_N,
                                .alpha_x = closures.alpha_x,
                                .alpha_y = closures.alpha_y,
                                .fresnel_type = MicrofacetFresnel::F82_TINT,
                                .fresnel = &fresnel};
      const Spectrum estimate = bsdf_microfacet_estimate_albedo(&energy_globals, wi, &bsdf, true, true);
      features.add(closures.metal_weight * estimate, specular_N, alpha,
                   CLOSURE_BSDF_MICROFACET_GGX_ID, roughness_weighting);
    }
    if (closures.has_specular) {
      FresnelGeneralizedSchlick fresnel = {.thin_film = {},
                                           .tint = {one_spectrum(), zero_spectrum()},
                                           .f0 = closures.specular_f0,
                                           .f90 = one_spectrum(),
                                           .exponent = -closures.eta};
      const MicrofacetBsdf bsdf{.type = CLOSURE_BSDF_MICROFACET_GGX_ID,
                                .N = specular_N,
                                .alpha_x = closures.alpha_x,
                                .alpha_y = closures.alpha_y,
                                .ior = closures.eta,
                                .fresnel_type = MicrofacetFresnel::GENERALIZED_SCHLICK,
                                .fresnel = &fresnel};
      const Spectrum estimate = bsdf_microfacet_estimate_albedo(&energy_globals, wi, &bsdf, true, true);
      features.add(closures.specular_weight * estimate, specular_N, alpha,
                   CLOSURE_BSDF_MICROFACET_GGX_ID, roughness_weighting);
    }
    if (closures.subsurface_sample_weight > 0.0f) {
      features.add(closures.subsurface_weight, specular_N, 1.0f,
                   CLOSURE_BSSRDF_RANDOM_WALK_ID, roughness_weighting);
    }
    if (closures.has_diffuse) {
      features.add(closures.diffuse_weight, N, 1.0f, CLOSURE_BSDF_DIFFUSE_ID, roughness_weighting);
    }
  }

  Spectrum throughput = rgb(feature_throughput);
  Spectrum albedo_delta;
  float3 normal_delta;
  features.finish(options, throughput, albedo_delta, normal_delta);
  next_feature_throughput = vector(throughput);
  albedo = vector(albedo_delta);
  guide_normal = vector(normal_delta);
}
#endif

#ifdef CYCLES_WEBGPU_FILM_SURFACE
[[gnu::always_inline]] void cycles_surface_film_passes(
    GgxFloat4 material,
    GgxFloat4 tint_alpha,
    GgxFloat4 weights,
    GgxFloat4 glass,
    GgxFloat4 auxiliary,
    GgxFloat4 bump,
    GgxFloat4 normal,
    GgxFloat4 specular_normal,
    GgxFloat4 geometric_normal,
    GgxFloat4 view,
    const unsigned int color_mask,
    GgxFloat4 &out_normal,
    GgxFloat4 &out_diffuse,
    GgxFloat4 &out_glossy,
    GgxFloat4 &out_transmission,
    GgxFloat4 &out_ao_normal)
{
  using namespace ccl;
  Spectrum diffuse = zero_spectrum();
  Spectrum glossy = zero_spectrum();
  Spectrum transmission = zero_spectrum();
  float3 normal_sum = zero_float3();
  float3 ao_normal_sum = zero_float3();
  Spectrum surface_weight = rgb(material);
  const float3 mesh_N = rgb(normal);
  const uint mixed_flags = material.w == CYCLES_WEBGPU_MATERIAL_DIFFUSE_GLOSSY ?
                               uint(weights.z) : 0u;
  const bool bump_diffuse = bump.w != 0.0f && !(mixed_flags & CYCLES_WEBGPU_MIXED_BUMP_GLOSSY_ONLY);
  const float3 N = bump_diffuse ? rgb(bump) : mesh_N;
  const GgxFloat4 glossy_bump = cycles_mixed_glossy_bump(bump, auxiliary, mixed_flags);
  const float3 Ns = rgb(
      cycles_bump_specular_normal(glossy_bump, geometric_normal, view, specular_normal));
  const float3 wi = rgb(view);
  /* The data pass uses raw closure weights, not the denoiser's albedo weights. */
  const auto add_normal = [&](const float3 closure_N, const Spectrum weight, const bool ao = false) {
    const float3 weighted = closure_N * fabsf(average(weight));
    normal_sum += weighted;
    if (ao) {
      ao_normal_sum += weighted;
    }
  };
  const auto add_microfacet = [&](const MicrofacetBsdf &bsdf) {
    add_normal(bsdf.N, bsdf.weight);
    const bool is_glass = CLOSURE_IS_GLASS(bsdf.type);
    if (!CLOSURE_IS_BSDF_MICROFACET(bsdf.type)) {
      glossy += bsdf.weight;
      return;
    }
    if ((color_mask & 2u) && (is_glass || CLOSURE_IS_BSDF_GLOSSY(bsdf.type))) {
      glossy += bsdf.weight * bsdf_microfacet_estimate_albedo(
                                  &energy_globals, wi, &bsdf, true, false);
    }
    if ((color_mask & 4u) && (is_glass || CLOSURE_IS_BSDF_TRANSMISSION(bsdf.type))) {
      transmission += bsdf.weight * bsdf_microfacet_estimate_albedo(
                                       &energy_globals, wi, &bsdf, false, true);
    }
  };

  if (material.w == CYCLES_WEBGPU_MATERIAL_GGX ||
      material.w == CYCLES_WEBGPU_MATERIAL_MULTISCATTER_GGX)
  {
    if (CYCLES_WEBGPU_GGX_HAS_TRANSLUCENT(glass)) {
      const Spectrum translucent = rgb(glass);
      diffuse += translucent;
      add_normal(Ns, translucent, true);
      add_normal(mesh_N, rgb(auxiliary));
      glass = {};
    }
    MicrofacetBsdf coat_bsdf{};
    if (setup_compact_white_coat(
            coat_bsdf, tint_alpha, weights, glass, Ns, rgb(specular_normal), wi, 0.0f))
    {
      add_microfacet(coat_bsdf);
    }
    FresnelGeneralizedSchlick glass_fresnel{};
    MicrofacetBsdf glass_bsdf{.type = CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID,
                             .fresnel_type = MicrofacetFresnel::GENERALIZED_SCHLICK,
                             .fresnel = &glass_fresnel};
    if (setup_glass(glass_bsdf, glass_fresnel, material, tint_alpha, weights, glass, auxiliary,
                    GgxFloat4{Ns.x, Ns.y, Ns.z, normal.w}, wi))
    {
      add_microfacet(glass_bsdf);
    }
    if (glass.x > CLOSURE_WEIGHT_CUTOFF) {
      weights.w *= 1.0f - glass.x;
    }

    ReflectiveClosures closures;
    setup_closures(closures, material, tint_alpha, weights, glass, auxiliary, Ns, wi);
    if (closures.has_metal) {
      FresnelF82Tint fresnel = {.thin_film = {}, .f0 = closures.metal_f0, .b = closures.metal_tint_b};
      const MicrofacetBsdf bsdf{.weight = closures.metal_weight,
                                .type = CLOSURE_BSDF_MICROFACET_GGX_ID,
                                .N = Ns,
                                .alpha_x = closures.alpha_x,
                                .alpha_y = closures.alpha_y,
                                .fresnel_type = MicrofacetFresnel::F82_TINT,
                                .fresnel = &fresnel};
      add_microfacet(bsdf);
    }
    if (closures.has_specular) {
      FresnelGeneralizedSchlick fresnel = {.thin_film = {},
                                           .tint = {one_spectrum(), zero_spectrum()},
                                           .f0 = closures.specular_f0,
                                           .f90 = one_spectrum(),
                                           .exponent = -closures.eta};
      const MicrofacetBsdf bsdf{.weight = closures.specular_weight,
                                .type = CLOSURE_BSDF_MICROFACET_GGX_ID,
                                .N = Ns,
                                .alpha_x = closures.alpha_x,
                                .alpha_y = closures.alpha_y,
                                .ior = closures.eta,
                                .fresnel_type = MicrofacetFresnel::GENERALIZED_SCHLICK,
                                .fresnel = &fresnel};
      add_microfacet(bsdf);
    }
    if (closures.subsurface_sample_weight > 0.0f) {
      add_normal(Ns, closures.subsurface_weight);
      diffuse += closures.subsurface_weight;
    }
    if (closures.has_diffuse) {
      add_normal(N, closures.diffuse_weight, true);
      diffuse += closures.diffuse_weight;
    }
  }
  else if (CYCLES_WEBGPU_MATERIAL_IS_GLOSSY(material.w)) {
    if (material.w == CYCLES_WEBGPU_MATERIAL_GLOSSY_TRANSPARENT) {
      /* bsdf_transparent_setup() uses the shader normal for its closure. */
      add_normal(mesh_N, rgb(auxiliary));
    }
    with_standalone_glossy(
        material, tint_alpha, weights, glass, Ns, wi, 0.0f,
        [&](const MicrofacetBsdf &bsdf) __attribute__((always_inline)) {
          add_microfacet(bsdf);
          return 0;
        });
  }
#ifdef CYCLES_WEBGPU_ORDERED_CLOSURES
  else if (material.w == CYCLES_WEBGPU_MATERIAL_ORDERED_PRINCIPLED) {
    for (uint index = 0; index < 2u; ++index) {
      const auto branch = read_ordered_principled_branch(
          material, index, index == 0u ? tint_alpha : weights, glass, geometric_normal,
          specular_normal, view, 0.0f);
      if (branch.sheen.w > 0.0f) {
        const Spectrum weight = make_spectrum(branch.sheen.w);
        diffuse += weight;
        add_normal(ordered_principled_sheen_normal(branch, normal), weight, true);
      }
      for_ordered_principled_glossy(branch, add_microfacet);
      if (branch.closures.has_diffuse) {
        const Spectrum weight = branch.closures.diffuse_weight;
        diffuse += weight;
        add_normal(rgb(branch.bump.w != 0.0f ? branch.bump : normal), weight, true);
      }
    }
  }
  else if (CYCLES_WEBGPU_MATERIAL_HAS_ORDERED_CLOSURES(material.w)) {
    const unsigned int count = min(unsigned(material.y), CYCLES_WEBGPU_MAX_ORDERED_LOBES);
    for (unsigned int index = 0; index < count; index++) {
      const OrderedClosure lobe = read_ordered_closure(material, index);
      const Spectrum weight = rgb(lobe.color_alpha);
      if (lobe.type == CLOSURE_BSDF_TRANSPARENT_ID) {
        add_normal(mesh_N, weight);
      }
      else if (lobe.diffuse()) {
        const GgxFloat4 lobe_N = lobe.type == CLOSURE_BSDF_TRANSLUCENT_ID ?
                                    cycles_bump_specular_normal(
                                        lobe.bump, geometric_normal, view, specular_normal) :
                                lobe.bump.w != 0.0f ? lobe.bump : normal;
        diffuse += weight;
        add_normal(rgb(lobe_N), weight, true);
      }
      else {
        MicrofacetBsdf bsdf{};
        setup_ordered_glossy(bsdf, lobe, geometric_normal, view, specular_normal, 0.0f);
        add_microfacet(bsdf);
      }
    }
  }
#endif
  else if (material.w == CYCLES_WEBGPU_MATERIAL_DIFFUSE_GLOSSY) {
    if (weights.x > 0.0f) {
      diffuse = surface_weight;
      add_normal(N, diffuse, true);
    }
    if (weights.y > 0.0f) {
      glossy = rgb(tint_alpha);
      add_normal(Ns, glossy);
    }
  }
  else if (material.w == CYCLES_WEBGPU_MATERIAL_DIFFUSE_TRANSLUCENT) {
    if (weights.x > 0.0f) {
      diffuse = surface_weight;
      add_normal(N, diffuse, true);
    }
    if (weights.y > 0.0f) {
      diffuse += rgb(tint_alpha);
      add_normal(Ns, rgb(tint_alpha), true);
    }
  }
  else if (closure_sample_weight(0, surface_weight) > 0.0f) {
    if (material.w == CYCLES_WEBGPU_MATERIAL_TRANSLUCENT) {
      add_normal(Ns, surface_weight, true);
      diffuse = surface_weight;
    }
    else if (material.w == CYCLES_WEBGPU_MATERIAL_GLOSSY_TOON ||
             material.w == CYCLES_WEBGPU_MATERIAL_VELVET)
    {
      add_normal(N, surface_weight);
      glossy = surface_weight;
    }
    else if (material.w == 1.0f || material.w == CYCLES_WEBGPU_MATERIAL_MIXED_TRANSPARENT ||
             material.w == CYCLES_WEBGPU_MATERIAL_DIFFUSE_TOON ||
             material.w == CYCLES_WEBGPU_MATERIAL_SHEEN)
    {
      add_normal(N, surface_weight, true);
      diffuse = surface_weight;
    }
  }
  float3 fallback_normal = mesh_N;
  if ((material.w == 1.0f || material.w == CYCLES_WEBGPU_MATERIAL_GLOSSY ||
       material.w == CYCLES_WEBGPU_MATERIAL_DIFFUSE_GLOSSY) &&
      bump.w >= CYCLES_WEBGPU_NORMAL_SHADER)
  {
    /* The stock Normal and AO pass helpers fall back to ShaderData::N. */
    fallback_normal = rgb(bump.w >= CYCLES_WEBGPU_NORMAL_BOTH ? auxiliary : bump);
  }
  out_normal = vector(is_zero(normal_sum) ? fallback_normal : safe_normalize(normal_sum));
  out_diffuse = vector(diffuse);
  out_glossy = vector(glossy);
  out_transmission = vector(transmission);
  out_ao_normal = vector(is_zero(ao_normal_sum) ? fallback_normal : safe_normalize(ao_normal_sum));
  out_ao_normal.w = 1.0f;
}
#endif
