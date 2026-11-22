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

enum Lobe { Metal, Glass, Specular, Subsurface, Diffuse };

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
  float alpha;
  float eta;
  float energy_scale;
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

ccl_device bool setup_glass(MicrofacetBsdf &bsdf,
                            FresnelGeneralizedSchlick &fresnel,
                            const GgxFloat4 material,
                            const GgxFloat4 tint_alpha,
                            const GgxFloat4 weights,
                            const GgxFloat4 glass,
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
  bsdf.alpha_x = bsdf.alpha_y = tint_alpha.w;
  bsdf.ior = normal.w != 0.0f ? 1.0f / glass.y : glass.y;
  bsdf.T = zero_float3();
  if (refraction) {
    if (glass.z == CYCLES_WEBGPU_REFRACTION_BECKMANN) {
      bsdf_microfacet_beckmann_refraction_setup(&bsdf);
    }
    else {
      bsdf_microfacet_ggx_refraction_setup(&bsdf);
    }
    return true;
  }
  if (glass.z != 0.0f) {
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

ccl_device void accumulate_glass(const MicrofacetBsdf &bsdf,
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
    eval += lobe_eval * bsdf.weight;
    pdf += lobe_pdf * bsdf.sample_weight;
  }
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
                                             const float alpha,
                                             const Spectrum Fss,
                                             Spectrum &weight,
                                             float &sample_weight)
{
  MicrofacetBsdf bsdf{};
  bsdf.type = CLOSURE_BSDF_MICROFACET_GGX_ID;
  bsdf.N = N;
  bsdf.alpha_x = bsdf.alpha_y = alpha;
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
                               const float subsurface,
                               const float3 N,
                               const float3 wi)
{
  const Spectrum base_color = rgb(material);
  const Spectrum tint = rgb(tint_alpha);
  const float metallic = weights.x;
  const float eta = weights.y;
  const float alpha = tint_alpha.w;
  const float rough = alpha_to_roughness_index(alpha, alpha);
  const float y = mu_to_y_index(dot(N, wi));
  const Spectrum white = one_spectrum();
  const bool multiscatter = material.w == CYCLES_WEBGPU_MATERIAL_MULTISCATTER_GGX;
  Spectrum weight = make_spectrum(weights.w);
  closures.alpha = alpha;
  closures.eta = eta;
  closures.energy_scale = 1.0f;
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
        closures.energy_scale = preserve_reflective_energy(
            N, wi, alpha, Fss, closures.metal_weight, closures.metal_sample_weight);
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
        closures.energy_scale = preserve_reflective_energy(
            N, wi, alpha, Fss, closures.specular_weight, closures.specular_sample_weight);
      }
      weight = closure_layering_weight(closures.specular_weight * reduce_max(albedo), weight);
    }
  }

  /* bssrdf_alloc and bssrdf_setup with all RGB radii active. */
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

ccl_device void eval_ggx(const float alpha2,
                         const float cos_NI,
                         const float cos_NO,
                         const float cos_NH,
                         const Spectrum fresnel,
                         Spectrum &eval,
                         float &pdf)
{
  if (is_zero(fresnel)) {
    return;
  }
  /* Isotropic, reflecting case of bsdf_microfacet_eval<GGX>(). Reflecting
   * lobes have no transmission probability. */
  const float D = bsdf_D<MicrofacetType::GGX>(alpha2, cos_NH);
  const float lambdaI = bsdf_lambda<MicrofacetType::GGX>(alpha2, cos_NI);
  const float lambdaO = bsdf_lambda<MicrofacetType::GGX>(alpha2, cos_NO);
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
  else if (cos_NI > 0.0f && cos_NO >= 0.0f) {
    const float3 h = wi + wo;
    const float3 H = h * safe_divide(1.0f, len(h));
    Spectrum eval = zero_spectrum();
    float pdf = 0.0f;
    eval_ggx(sqr(closures.alpha),
             cos_NI,
             cos_NO,
             dot(N, H),
             reflection_fresnel<lobe>(closures, dot(H, wi)),
             eval,
             pdf);
    if (pdf != 0.0f) {
      eval *= closures.energy_scale;
      if constexpr (lobe == Metal) {
        sum_eval += eval * closures.metal_weight;
        sum_pdf += pdf * closures.metal_sample_weight;
      }
      else {
        sum_eval += eval * closures.specular_weight;
        sum_pdf += pdf * closures.specular_sample_weight;
      }
    }
  }
}

}  // namespace
}  // namespace ccl

[[gnu::always_inline]] GgxFloat4 cycles_ggx_eval(GgxFloat4 material,
                                                GgxFloat4 tint_alpha,
                                                GgxFloat4 weights,
                                                GgxFloat4 glass,
                                                GgxFloat4 normal,
                                                GgxFloat4 specular_normal,
                                                GgxFloat4 view,
                                                GgxFloat4 light,
                                                float blur_alpha,
                                                float transparent_sample_weight)
{
  using namespace ccl;
  const float3 N = rgb(normal);
  const float3 specular_N = rgb(specular_normal);
  const float3 wi = rgb(view);
  const float3 wo = rgb(light);
  FresnelGeneralizedSchlick glass_fresnel{};
  /* Keep the private Fresnel pointer valid across the setup branch so it
   * compiles without a pointer phi in SPIR-V. */
  MicrofacetBsdf glass_bsdf{.type = CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID,
                           .fresnel_type = MicrofacetFresnel::GENERALIZED_SCHLICK,
                           .fresnel = &glass_fresnel};
  const bool has_glass = glass_enabled && setup_glass(
      glass_bsdf, glass_fresnel, material, tint_alpha, weights, glass,
      GgxFloat4{specular_N.x, specular_N.y, specular_N.z, normal.w}, wi);
  if (has_glass) {
    bsdf_microfacet_blur(reinterpret_cast<ShaderClosure *>(&glass_bsdf), blur_alpha);
  }
  if (glass_enabled && glass.x > CLOSURE_WEIGHT_CUTOFF) {
    weights.w *= 1.0f - glass.x;
  }
  ReflectiveClosures closures;
  setup_closures(closures, material, tint_alpha, weights, glass.w, specular_N, wi);
  closures.alpha = fmaxf(closures.alpha, blur_alpha);
  Spectrum eval = zero_spectrum();
  float weighted_pdf = 0.0f;
  const bool specular_eval = valid_specular_eval(N, specular_N, wo);
  if (closures.has_metal && specular_eval) {
    accumulate_lobe<Metal>(closures, specular_N, wi, wo, eval, weighted_pdf);
  }
  if (has_glass && specular_eval) {
    accumulate_glass(glass_bsdf, wi, wo, eval, weighted_pdf);
  }
  if (closures.has_specular && specular_eval) {
    accumulate_lobe<Specular>(closures, specular_N, wi, wo, eval, weighted_pdf);
  }
  if (closures.has_diffuse) {
    accumulate_lobe<Diffuse>(closures, N, wi, wo, eval, weighted_pdf);
  }
  const float total = closures.metal_sample_weight +
                      (has_glass ? glass_bsdf.sample_weight : 0.0f) +
                      closures.specular_sample_weight + closures.subsurface_sample_weight +
                      closures.diffuse_sample_weight + transparent_sample_weight;
  return {eval.x, eval.y, eval.z, total > 0.0f ? weighted_pdf / total : 0.0f};
}

[[gnu::always_inline]] unsigned int cycles_ggx_sample(GgxFloat4 material,
                                                       GgxFloat4 tint_alpha,
                                                       GgxFloat4 weights,
                                                       GgxFloat4 glass,
                                                       GgxFloat4 normal,
                                                       GgxFloat4 geometric_normal,
                                                       GgxFloat4 specular_normal,
                                                       GgxFloat4 view,
                                                       GgxFloat4 random_seed,
                                                       GgxFloat4 transparent,
                                                       float blur_alpha,
                                                       bool ray_differentials,
                                                       bool only_transparent,
                                                       GgxFloat4 &weight,
                                                       GgxFloat4 &out)
{
  using namespace ccl;
  weight = {0.0f, 0.0f, 0.0f, 0.0f};
  out = {0.0f, 0.0f, 0.0f, 0.0f};
  const float3 N = rgb(normal);
  const float3 specular_N = rgb(specular_normal);
  const float3 wi = rgb(view);
  const float3 Ng = rgb(geometric_normal);
  const float transparent_weight = fabsf(transparent.w);
  if (only_transparent) {
    if (transparent_weight > 0.0f) {
      weight = {transparent.x, transparent.y, transparent.z, 1e6f};
      return LABEL_TRANSMIT | LABEL_TRANSPARENT;
    }
    return LABEL_NONE;
  }
  const float cos_NI = dot(specular_N, wi);
  FresnelGeneralizedSchlick glass_fresnel{};
  MicrofacetBsdf glass_bsdf{.type = CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID,
                           .fresnel_type = MicrofacetFresnel::GENERALIZED_SCHLICK,
                           .fresnel = &glass_fresnel};
  const bool has_glass = glass_enabled && setup_glass(
      glass_bsdf, glass_fresnel, material, tint_alpha, weights, glass,
      GgxFloat4{specular_N.x, specular_N.y, specular_N.z, normal.w}, wi);
  if (has_glass) {
    bsdf_microfacet_blur(reinterpret_cast<ShaderClosure *>(&glass_bsdf), blur_alpha);
  }
  if (glass_enabled && glass.x > CLOSURE_WEIGHT_CUTOFF) {
    weights.w *= 1.0f - glass.x;
  }
  ReflectiveClosures closures;
  setup_closures(closures, material, tint_alpha, weights, glass.w, specular_N, wi);
  closures.alpha = fmaxf(closures.alpha, blur_alpha);

  const float metal = closures.metal_sample_weight;
  const float glass_weight = has_glass ? glass_bsdf.sample_weight : 0.0f;
  const float specular = closures.specular_sample_weight;
  const float diffuse = closures.diffuse_sample_weight;
  const float subsurface = closures.subsurface_sample_weight;
  const bool transparent_first = transparent.w < 0.0f;
  const float prefix = transparent_first ? transparent_weight : 0.0f;
  const float metal_end = prefix + metal;
  const float glass_end = metal_end + glass_weight;
  const float specular_end = glass_end + specular;
  const float subsurface_end = specular_end + subsurface;
  const float opaque_end = subsurface_end + diffuse;
  const float total = opaque_end + (transparent_first ? 0.0f : transparent_weight);
  if (!(total > 0.0f)) {
    return LABEL_NONE;
  }

  const float r = random_seed.z * total;
  if (transparent_weight > 0.0f &&
      (transparent_first ? (r < prefix || r >= total) : (r >= opaque_end && r < total)))
  {
    Spectrum eval;
    float3 direction;
    float pdf;
    const int label = bsdf_transparent_sample(nullptr, Ng, wi, &eval, &direction, &pdf);
    eval *= rgb(transparent);
    float weighted_pdf = pdf * transparent_weight;
    const bool specular_eval = valid_specular_eval(N, specular_N, direction);
    if (specular_eval && closures.has_metal) {
      accumulate_lobe<Metal>(closures, specular_N, wi, direction, eval, weighted_pdf);
    }
    if (specular_eval && closures.has_specular) {
      accumulate_lobe<Specular>(closures, specular_N, wi, direction, eval, weighted_pdf);
    }
    if (closures.has_diffuse) {
      accumulate_lobe<Diffuse>(closures, N, wi, direction, eval, weighted_pdf);
    }
    const float mixed_pdf = weighted_pdf / total;
    const Spectrum multiplier = eval / mixed_pdf;
    weight = {multiplier.x, multiplier.y, multiplier.z, mixed_pdf};
    return label;
  }

  /* Pick and remap from surface_shader_bsdf_bssrdf_pick(). */
  Lobe lobe = closures.has_metal ? Metal : has_glass ? Glass :
              closures.has_specular ? Specular : Diffuse;
  float3 random = rgb(random_seed);
  const bool multiple = transparent_weight > 0.0f || subsurface > 0.0f || (has_glass &&
                         (closures.has_metal || closures.has_specular || closures.has_diffuse)) ||
                        (closures.has_metal && closures.has_specular) ||
                        (closures.has_diffuse && (closures.has_metal || closures.has_specular));
  if (multiple) {
    if (r < metal_end) {
      lobe = Metal;
      random.z = (r - prefix) / metal;
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
  else if (lobe == Diffuse) {
    wo = zero_float3();
    webgpu_sample_cos_hemisphere(N, make_float2(random), &wo, &pdf);
    if (!(dot(Ng, wo) > 0.0f)) {
      return LABEL_NONE;
    }
    eval = make_spectrum(pdf) * closures.diffuse_weight;
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
  else {
    if (!(cos_NI > 0.0f) || random.z >= 1.0f) {
      return LABEL_NONE;
    }
    /* Isotropic non-singular reflection from bsdf_microfacet_sample<GGX>(). */
    float3 X, Y;
    make_orthonormals(specular_N, &X, &Y);
    const float3 local_I = make_float3(dot(X, wi), dot(Y, wi), cos_NI);
    const float3 local_H = microfacet_ggx_sample_vndf(
        local_I, closures.alpha, closures.alpha, make_float2(random));
    const float3 H = to_global(local_H, X, Y, specular_N);
    const float cos_HI = dot(H, wi);
    wo = 2.0f * cos_HI * H - wi;
    const float cos_NO = dot(specular_N, wo);
    if (cos_NO < 0.0f || dot(Ng, wo) < 0.0f) {
      return LABEL_NONE;
    }
    if (lobe == Metal) {
      const Spectrum fresnel = reflection_fresnel<Metal>(closures, cos_HI);
      eval_ggx(sqr(closures.alpha), cos_NI, cos_NO, local_H.z, fresnel, eval, pdf);
      eval *= closures.energy_scale;
      eval *= closures.metal_weight;
      selected_weight = metal;
    }
    else {
      const Spectrum fresnel = reflection_fresnel<Specular>(closures, cos_HI);
      eval_ggx(sqr(closures.alpha), cos_NI, cos_NO, local_H.z, fresnel, eval, pdf);
      eval *= closures.energy_scale;
      eval *= closures.specular_weight;
      selected_weight = specular;
    }
  }
  if (pdf == 0.0f) {
    return LABEL_NONE;
  }

  const float alpha_squared = sqr(closures.alpha);
  const float glass_roughness_squared = glass_bsdf.alpha_x * glass_bsdf.alpha_y;
  float roughness_squared = lobe == Diffuse ? 1.0f :
                                lobe == Glass ? glass_roughness_squared : alpha_squared;
  /* The balance heuristic also evaluates the other lobes at the sampled
   * direction. The selected lobe is accumulated first, as in Cycles. */
  if (multiple && lobe != Subsurface) {
    float weighted_pdf = pdf * selected_weight;
    float roughness_pdf = weighted_pdf * roughness_squared;
    float sum_weight = selected_weight + subsurface + prefix;
    const bool specular_eval = valid_specular_eval(N, specular_N, wo);
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
        accumulate_glass(glass_bsdf, wi, wo, eval, other_pdf);
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
      accumulate_lobe<Diffuse>(closures, N, wi, wo, eval, other_pdf);
      weighted_pdf += other_pdf;
      if (ray_differentials) {
        roughness_pdf += other_pdf;
      }
      sum_weight += diffuse;
    }
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

[[gnu::always_inline]] GgxFloat4 cycles_standalone_eval(GgxFloat4 material,
                                                       GgxFloat4 parameters,
                                                       GgxFloat4 secondary,
                                                       GgxFloat4 extra,
                                                       GgxFloat4 hair,
                                                       GgxFloat4 normal,
                                                       GgxFloat4 geometric_normal,
                                                       GgxFloat4 incoming,
                                                       GgxFloat4 outgoing,
                                                       float blur_alpha,
                                                       unsigned int &lcg_state)
{
  using namespace ccl;
  const float3 N = rgb(normal);
  const float3 wi = rgb(incoming);
  const float3 wo = rgb(outgoing);
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
  if (material.w == CYCLES_WEBGPU_MATERIAL_GLOSSY) {
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
                                                             GgxFloat4 normal,
                                                             GgxFloat4 geometric_normal,
                                                             GgxFloat4 incoming,
                                                             GgxFloat4 random,
                                                             float blur_alpha,
                                                             GgxFloat4 &weight,
                                                             GgxFloat4 &out)
{
  using namespace ccl;
  const float3 N = rgb(normal);
  const float3 Ng = rgb(geometric_normal);
  const float3 wi = rgb(incoming);
  const float2 rand = make_float2(random.x, random.y);
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
  if (material.w == CYCLES_WEBGPU_MATERIAL_GLOSSY) {
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
  const float3 N = rgb(normal);
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
  else if (material.w == CYCLES_WEBGPU_MATERIAL_GLOSSY) {
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
    if (glass.w == 0.0f && hair.w != 0.0f) {
      features.transparent_albedo = rgb(hair);
    }
    const float3 specular_N = rgb(specular_normal);
    const float3 wi = rgb(view);
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
                    GgxFloat4{specular_N.x, specular_N.y, specular_N.z, normal.w},
                    wi))
    {
      bsdf_microfacet_blur(reinterpret_cast<ShaderClosure *>(&glass_bsdf), blur_alpha);
      const Spectrum estimate = bsdf_microfacet_estimate_albedo(
          &energy_globals, wi, &glass_bsdf, true, true);
      features.add(glass_bsdf.weight * estimate,
                   specular_N,
                   glass_bsdf.alpha_x,
                   glass_bsdf.type,
                   roughness_weighting);
    }
    if (glass.x > CLOSURE_WEIGHT_CUTOFF) {
      weights.w *= 1.0f - glass.x;
    }

    ReflectiveClosures closures;
    setup_closures(closures, material, tint_alpha, weights, glass.w, specular_N, wi);
    const float alpha = fmaxf(closures.alpha, blur_alpha);
    if (closures.has_metal) {
      FresnelF82Tint fresnel = {.thin_film = {}, .f0 = closures.metal_f0, .b = closures.metal_tint_b};
      const MicrofacetBsdf bsdf{.type = CLOSURE_BSDF_MICROFACET_GGX_ID,
                                .N = specular_N,
                                .alpha_x = alpha,
                                .alpha_y = alpha,
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
                                .alpha_x = alpha,
                                .alpha_y = alpha,
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
