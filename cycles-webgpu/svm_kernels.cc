/* SPDX-License-Identifier: Apache-2.0 */

#include "../blender-platform/cycles-webgpu/material_layout.h"

#ifdef __aarch64__
#  include <arm_neon.h>
#endif

#define CCL_NAMESPACE_BEGIN namespace ccl {
#define CCL_NAMESPACE_END }
#include "kernel/types.h"

#define __KERNEL_GPU__
#include "kernel/globals.h"
#include "util/types_image.h"
#undef __KERNEL_GPU__

__attribute__((pure)) unsigned int cycles_svm_word(unsigned int index);

namespace ccl {
/* Stock lookup tables follow the packed SVM records. */
struct SvmPackedFloatWords {
  unsigned int offset;
  float operator[](const int index) const
  {
    return __uint_as_float(cycles_svm_word(offset + unsigned(index)));
  }
};
struct SvmGlobals {
  KernelData data;
  const uint *svm_nodes;
  SvmPackedFloatWords lookup_table;
  SvmPackedFloatWords ies;
};
using KernelGlobals = const SvmGlobals *;
}  // namespace ccl

#define kernel_data (kg->data)
#define kernel_data_fetch(name, index) (kg->name[index])
#define CYCLES_WEBGPU_RUNTIME_SVM
#ifdef CYCLES_WEBGPU_SVM_SPECIALIZED_HEADER
#  include CYCLES_WEBGPU_SVM_SPECIALIZED_HEADER
#else
#  include "../blender-platform/cycles-webgpu/svm_material_eval.h"
#endif
#ifndef CYCLES_WEBGPU_SHADOW_SVM
#  include "kernel/closure/bsdf_microfacet.h"
#endif
#ifdef CYCLES_WEBGPU_SAMPLED_EMISSION
#  include "kernel/closure/emissive.h"
#else
#  include "kernel/closure/bsdf_sheen.h"
#  include "kernel/closure/bsdf_principled_hair_huang.h"
#endif
#include "kernel/util/ies.h"

using Float4 = float __attribute__((ext_vector_type(4)));
using UInt4 = unsigned int __attribute__((ext_vector_type(4)));

__attribute__((pure)) ccl::WebGPUSvmContextWords cycles_svm_context(unsigned int lane,
                                                                  unsigned int field);

#ifdef CYCLES_WEBGPU_SCENE_QUERIES
float cycles_query_ao(UInt4 position, UInt4 hit, UInt4 rng, Float4 input, unsigned samples,
                      unsigned flags);
Float4 cycles_query_bevel(UInt4 position, UInt4 hit, UInt4 rng, Float4 input, Float4 Ng,
                          unsigned samples);
void cycles_query_raycast(UInt4 position, UInt4 hit, Float4 ray_origin, Float4 ray_direction,
                          unsigned only_local, Float4 &data, Float4 &normal);
#endif
#ifndef CYCLES_WEBGPU_SAMPLED_EMISSION
/* Reuse the setup's projected ellipse test for the transparent path and
 * for shadow extinction. The temporary closure's weight is unnecessary. */
static bool cycles_huang_is_transparent(const ccl::WebGPUSvmMaterial &material,
                                        const UInt4 normal,
                                        const UInt4 direction)
{
  using namespace ccl;
  ShaderData sd{};
  sd.type = PRIMITIVE_TRIANGLE;
  sd.num_closure = 1;
  sd.dPdu = material.surface_tangent;
  sd.N = webgpu_svm_context_xyz(normal);
  sd.wi = -webgpu_svm_context_xyz(direction);
  HuangHairExtra extra{};
  extra.pixel_coverage = 1.0f;
  HuangHairBSDF bsdf{};
  bsdf.aspect_ratio = material.surface_hair_aspect;
  bsdf.extra = &extra;
  return bsdf_hair_huang_setup(&sd, &bsdf, 0) == 0;
}
#endif

#if !defined(CYCLES_WEBGPU_SAMPLED_EMISSION) && !defined(CYCLES_WEBGPU_SHADOW_SVM)
void cycles_svm_write_aov(unsigned int lane, unsigned int offset, unsigned int color, Float4 value);
#endif
#ifdef CYCLES_WEBGPU_ORDERED_CLOSURES
void cycles_svm_write_ordered_lobe(unsigned int lane,
                                  unsigned int index,
                                  Float4 color_and_type,
                                  Float4 normal_and_alpha);
#endif

namespace {
unsigned int packed_lookup_offset(const unsigned int texture, const unsigned int kind)
{
  const unsigned int lookup = cycles_svm_word(texture + ccl::webgpu_svm_ies_offset);
  if (cycles_svm_word(1) & CYCLES_WEBGPU_SVM_LOOKUP_TABLES) {
    return cycles_svm_word(texture + lookup + kind);
  }
  return kind == ccl::webgpu_svm_lookup_ies ? lookup : 0;
}

#ifndef CYCLES_WEBGPU_SHADOW_SVM
template<typename FetchSurface>
ccl::Spectrum pack_surface_emission(const ccl::WebGPUSvmMaterial &material,
                                     const FetchSurface surface,
                                     const bool reflective_caustics)
{
  using namespace ccl;
  if (is_zero(material.coat_emission)) {
    return material.emission;
  }

  const Spectrum base_weight = make_spectrum(material.mix_weight);
  Spectrum coat_weight = make_spectrum(material.coat.weight * material.mix_weight);
  if (!reflective_caustics || !(material.coat.weight > CLOSURE_WEIGHT_CUTOFF) ||
      !(closure_sample_weight(0, coat_weight) > 0.0f))
  {
    return material.emission + material.coat_emission * base_weight;
  }

  /* The producer shades the emissive layer before parts of the lower
   * reflection and diffuse lobes alter the material weight. */
  ShaderData sd{};
  sd.type = PRIMITIVE_TRIANGLE;
  sd.shader_flag = surface(CYCLES_WEBGPU_SVM_PATH).w;
  sd.N = webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_NORMAL));
  sd.Ng = webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_GEOMETRIC));
  sd.wi = -webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_DIRECTION));
  MicrofacetBsdf bsdf{};
  bsdf.N = maybe_ensure_valid_specular_reflection(&sd, safe_normalize_fallback(sd.N, sd.N));
  bsdf.ior = fabsf(material.coat.ior);
  bsdf.alpha_x = bsdf.alpha_y = material.coat.alpha;
  bsdf_microfacet_ggx_setup(&bsdf);
  bsdf.fresnel_type = MicrofacetFresnel::DIELECTRIC;

  const unsigned int texture = webgpu_svm_header_words + cycles_svm_word(0);
  const unsigned int lookup = cycles_svm_word(texture + webgpu_svm_ies_offset);
  const unsigned int sheen = cycles_svm_word(texture + lookup + webgpu_svm_lookup_sheen);
  /* The stock dielectric albedo table follows the optional Sheen table. */
  const unsigned int coat = lookup + webgpu_svm_lookup_words +
                            (sheen != 0u ? CYCLES_WEBGPU_SHEEN_LTC_WORDS : 0u);
  SvmGlobals globals;
  globals.data.tables.ggx_gen_schlick_ior_s = 0;
  globals.lookup_table = {texture + coat};
  const Spectrum albedo = bsdf_microfacet_estimate_albedo(&globals, sd.wi, &bsdf, true, false);
  return material.emission +
         material.coat_emission * closure_layering_weight(albedo * coat_weight, base_weight);
}
#endif

#ifndef CYCLES_WEBGPU_SAMPLED_EMISSION
struct PackedSheen {
  ccl::Spectrum weight;
  ccl::float3 ltc;
};

PackedSheen pack_sheen(const unsigned int texture,
                       const ccl::float3 wi,
                       const ccl::float3 normal,
                       const ccl::Spectrum weight,
                       const float roughness)
{
  using namespace ccl;
  SvmGlobals globals;
  globals.data.tables.sheen_ltc = 0;
  globals.lookup_table = {texture + packed_lookup_offset(texture, webgpu_svm_lookup_sheen)};
  ShaderData sd{};
  sd.num_closure_left = 1;
  sd.wi = wi;
  bsdf_sheen_setup(&globals, &sd, weight, normal, roughness);
  if (sd.num_closure == 0 || sd.closure[0].type != CLOSURE_BSDF_SHEEN_ID) {
    return {zero_spectrum(), zero_float3()};
  }
  const auto *sheen = reinterpret_cast<const SheenBsdf *>(&sd.closure[0]);
  return {sheen->weight, make_float3(sheen->transformA, sheen->transformB, sheen->roughness)};
}
#endif

/* The host already resolved attributes and particles while packing. */
struct PackedWordsHost : ccl::WebGPUSvmPackedHost {
  unsigned int texture;
  unsigned int primitive;
  float u;
  float v;
  unsigned int lane;
  bool film_aov;
#ifdef CYCLES_WEBGPU_SCENE_QUERIES
  static constexpr bool uses_ray_queries = true;
  float ambient_occlusion(const ccl::float3 normal, const float distance, const unsigned samples,
                          const unsigned flags) const
  {
    return cycles_query_ao(cycles_svm_context(lane, CYCLES_WEBGPU_SVM_POSITION),
                           cycles_svm_context(lane, CYCLES_WEBGPU_QUERY_HIT),
                           cycles_svm_context(lane, CYCLES_WEBGPU_QUERY_RNG),
                           Float4{normal.x, normal.y, normal.z, distance}, samples, flags);
  }
  ccl::WebGPUSvmRaycastHit raycast(const ccl::float3 position, const ccl::float3 direction,
                                  const float distance, const bool only_local) const
  {
    Float4 data, normal;
    cycles_query_raycast(cycles_svm_context(lane, CYCLES_WEBGPU_SVM_POSITION),
                         cycles_svm_context(lane, CYCLES_WEBGPU_QUERY_HIT),
                         Float4{position.x, position.y, position.z, distance},
                         Float4{direction.x, direction.y, direction.z, 0.0f},
                         unsigned(only_local), data, normal);
    return {data.x, data.y, data.z, ccl::make_float3(normal.x, normal.y, normal.z)};
  }
  ccl::float3 bevel(const ccl::float3 normal, const float radius, const unsigned samples) const
  {
    const auto Ng = cycles_svm_context(lane, CYCLES_WEBGPU_SVM_GEOMETRIC);
    const Float4 geometric{ccl::__uint_as_float(Ng.x), ccl::__uint_as_float(Ng.y),
                            ccl::__uint_as_float(Ng.z), 0.0f};
    const Float4 value = cycles_query_bevel(cycles_svm_context(lane, CYCLES_WEBGPU_SVM_POSITION),
                                            cycles_svm_context(lane, CYCLES_WEBGPU_QUERY_HIT),
                                            cycles_svm_context(lane, CYCLES_WEBGPU_QUERY_RNG),
                                            Float4{normal.x, normal.y, normal.z, radius},
                                            geometric, samples);
    return ccl::make_float3(value.x, value.y, value.z);
  }
#endif
  bool evaluate_aov() const
  {
    return film_aov;
  }
#if !defined(CYCLES_WEBGPU_SAMPLED_EMISSION) && !defined(CYCLES_WEBGPU_SHADOW_SVM)
  void write_aov(const int offset, const bool color, const ccl::float4 value) const
  {
    cycles_svm_write_aov(lane, unsigned(offset), unsigned(color),
                         Float4{value.x, value.y, value.z, value.w});
  }
#endif
#ifdef CYCLES_WEBGPU_ORDERED_CLOSURES
#  ifndef CYCLES_WEBGPU_SAMPLED_EMISSION
  ccl::float4 setup_paired_sheen(const ccl::float3 normal,
                                const float weight,
                                const float roughness) const
  {
    const auto wi = -ccl::webgpu_svm_context_xyz(cycles_svm_context(lane, CYCLES_WEBGPU_SVM_DIRECTION));
    const PackedSheen sheen = pack_sheen(texture, wi, normal, ccl::make_spectrum(weight), roughness);
    return ccl::make_float4(sheen.ltc, sheen.weight.x);
  }
#  endif

  bool write_ordered_lobe(const unsigned int index,
                          const unsigned int type,
                          const ccl::Spectrum weight,
                          const ccl::float3 normal,
                          const float alpha) const
  {
    /* Apply the same stock caustic gate as the standalone closure setup. */
    const unsigned int closure_type = type & ~CYCLES_WEBGPU_ORDERED_LOBE_BUMP;
    if (closure_type == ccl::CLOSURE_BSDF_MICROFACET_GGX_ID ||
        closure_type == ccl::CLOSURE_BSDF_MICROFACET_BECKMANN_ID)
    {
      const auto path = cycles_svm_context(lane, CYCLES_WEBGPU_SVM_PATH);
      if ((path.w & CYCLES_WEBGPU_SHADER_NO_REFLECTIVE_CAUSTICS) != 0u &&
          (path.x & ccl::PATH_RAY_VISIBILITY_DIFFUSE) != 0u)
      {
        return false;
      }
    }
    cycles_svm_write_ordered_lobe(
        lane, index, Float4{weight.x, weight.y, weight.z, float(type)},
        Float4{normal.x, normal.y, normal.z, alpha});
    return true;
  }
#endif
  bool attribute_sample(const unsigned int slot,
                        ccl::WebGPUSvmAttributeSample &sample,
                        const float du = 0.0f,
                        const float dv = 0.0f,
                        const float filter_width = 0.0f) const
  {
    sample = ccl::webgpu_svm_triangle_attribute(
        cycles_svm_word, texture, slot, primitive, u, v, du, dv, filter_width);
    return true;
  }
  bool triangle_corners(const unsigned int slot, ccl::WebGPUSvmTriangleCorners &corners) const
  {
    corners = ccl::webgpu_svm_triangle_corners(cycles_svm_word, texture, slot, primitive);
    return true;
  }
  bool ies(const unsigned int slot, const float h_angle, const float v_angle, float &value) const
  {
    const unsigned int table = packed_lookup_offset(texture, ccl::webgpu_svm_lookup_ies);
    if (table == 0) {
      return false;
    }
    /* Only the IES accessor is read, so leave the kernel data uninitialized. */
    ccl::SvmGlobals globals;
    globals.ies = {texture + table};
    value = ccl::kernel_ies_interp(&globals, int(slot), h_angle, v_angle);
    return true;
  }
  ccl::float3 cie_color_match_row(const int row) const
  {
    return ccl::webgpu_svm_read_rgb(cycles_svm_word,
                                    texture + ccl::webgpu_svm_cie_table_offset + 3u * unsigned(row));
  }
};

template<typename FetchSurface>
ccl::WebGPUSvmBumpInputs load_bump_differentials(const FetchSurface surface,
                                               const unsigned int texture,
                                               const unsigned int primitive)
{
  using namespace ccl;
  WebGPUSvmBumpInputs bump{};
  if ((cycles_svm_word(1) & CYCLES_WEBGPU_SVM_BUMP) == 0) {
    return bump;
  }
  const auto ray = surface(CYCLES_WEBGPU_SVM_DIFFERENTIAL);
  const float3 Ng = webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_GEOMETRIC));
  const float dP = differential_transfer_compact(
      __uint_as_float(ray.x), zero_float3(), __uint_as_float(ray.y), __uint_as_float(ray.z));
  if (dP == 0.0f || is_zero(Ng)) {
    return bump;
  }
  bump.position = differential_from_compact(Ng, dP);
  const float3 dpdu = webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_TRIANGLE_DPDU));
  const float3 dpdv = webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_TRIANGLE_DPDV));
  differential_dudv(&bump.u, &bump.v, dpdu, dpdv, bump.position, Ng);
  if (primitive < cycles_svm_word(texture)) {
    const WebGPUSvmUv a = webgpu_svm_triangle_uv(cycles_svm_word, texture, primitive, 0.0f, 0.0f);
    const WebGPUSvmUv u = webgpu_svm_triangle_uv(cycles_svm_word, texture, primitive, 1.0f, 0.0f) - a;
    const WebGPUSvmUv v = webgpu_svm_triangle_uv(cycles_svm_word, texture, primitive, 0.0f, 1.0f) - a;
    const WebGPUSvmUv dx = u * bump.u.dx + v * bump.v.dx;
    const WebGPUSvmUv dy = u * bump.u.dy + v * bump.v.dy;
    bump.uv = {make_float3(dx.x, dx.y, 0.0f), make_float3(dy.x, dy.y, 0.0f)};
  }
  if (primitive < cycles_svm_word(texture + 2)) {
    const float3 a = webgpu_svm_triangle_generated(cycles_svm_word, texture, primitive, 0.0f, 0.0f);
    const float3 u = webgpu_svm_triangle_generated(cycles_svm_word, texture, primitive, 1.0f, 0.0f) - a;
    const float3 v = webgpu_svm_triangle_generated(cycles_svm_word, texture, primitive, 0.0f, 1.0f) - a;
    bump.generated = {u * bump.u.dx + v * bump.v.dx, u * bump.u.dy + v * bump.v.dy};
  }
  return bump;
}

template<typename FetchSurface>
bool read_material(const unsigned int shader,
                   const unsigned int primitive,
                   const float u,
                   const float v,
                   const unsigned int lane,
                   const bool film_aov,
                   ccl::WebGPUSvmMaterial &material,
                   const FetchSurface surface,
                   const bool ordered_closures = false)
{
  const auto read_word = [](const unsigned int index) {
    return cycles_svm_word(ccl::webgpu_svm_header_words + index);
  };
#ifdef CYCLES_WEBGPU_VALIDATED_SVM_WORDS
  const auto fetch = ccl::WebGPUSvmValidatedWords{read_word};
#else
  const auto fetch = read_word;
#endif
  const unsigned int word_count = cycles_svm_word(0);
  const unsigned int texture = ccl::webgpu_svm_header_words + word_count;
  const auto uv = ccl::webgpu_svm_triangle_uv(cycles_svm_word, texture, primitive, u, v);
  const auto generated = ccl::webgpu_svm_triangle_generated(cycles_svm_word, texture, primitive, u, v);
  const auto color_space = ccl::webgpu_svm_read_color_space(cycles_svm_word, texture);
  const auto image = [texture](int id, ccl::float2 co, bool alpha, ccl::float4 &color) {
    return ccl::webgpu_svm_image(cycles_svm_word, texture, id, co, alpha, color);
  };
  bool uses_uv = false;
  bool uses_generated = false;
#ifdef CYCLES_WEBGPU_SAMPLED_EMISSION
  /* The scene encoder excludes sampled emitters that need ray differentials. */
  const ccl::WebGPUSvmBumpInputs bump{};
#else
  const auto bump = load_bump_differentials(surface, texture, primitive);
#endif
  return ccl::webgpu_svm_eval_material(
      fetch, image, uv, generated, color_space, uses_uv, uses_generated, word_count,
      cycles_svm_word(1) & CYCLES_WEBGPU_SVM_SHADER_COUNT, shader, material, surface,
      PackedWordsHost{{}, texture, primitive, u, v, lane, film_aov}, bump, ordered_closures);
}
}  // namespace

#ifdef CYCLES_WEBGPU_SAMPLED_EMISSION
Float4 cycles_svm_sampled_emission(UInt4 sample,
                                  UInt4 direction,
                                  UInt4 position,
                                  UInt4 normal,
                                  UInt4 geometric,
                                  unsigned int lane)
{
  using namespace ccl;
  const auto view = [](const UInt4 words) {
    return make_float3(__uint_as_float(words.x), __uint_as_float(words.y), __uint_as_float(words.z));
  };
  const float emission_pdf = emissive_pdf(view(geometric), -view(direction));
  if (emission_pdf == 0.0f) {
    return {};
  }
  const auto surface = [&](const unsigned int field) -> WebGPUSvmContextWords {
    switch (field) {
      case CYCLES_WEBGPU_SVM_POSITION:
        return position;
      case CYCLES_WEBGPU_SVM_NORMAL:
        return normal;
      case CYCLES_WEBGPU_SVM_GEOMETRIC:
        return geometric;
      case CYCLES_WEBGPU_SVM_DIRECTION:
        return direction;
      case CYCLES_WEBGPU_SVM_INTERSECTION:
        return {direction.w, sample.z, sample.w, position.w};
      case CYCLES_WEBGPU_SVM_PATH:
        /* integrate_light_nee() evaluates emission with no ray visibility. */
        return {PATH_RAY_VISIBILITY_NONE, PATH_RAY_EMISSION, cycles_svm_context(lane, field).z,
                cycles_svm_context(lane, field).w};
      default:
        return cycles_svm_context(lane, field);
    }
  };
  WebGPUSvmMaterial material;
  if (!read_material(normal.w,
                     sample.x - 1,
                     __uint_as_float(sample.z),
                     __uint_as_float(sample.w),
                     lane,
                     false,
                     material,
                     surface))
  {
    return {};
  }
  const Spectrum emission = pack_surface_emission(material, surface, true) * emission_pdf;
  return {emission.x, emission.y, emission.z, 0};
}
#elif defined(CYCLES_WEBGPU_SHADOW_SVM)
__attribute__((pure)) ccl::WebGPUSvmContextWords cycles_svm_shadow_context(unsigned int field,
                                                                         unsigned int object,
                                                                         unsigned int shader);

Float4 cycles_svm_shadow_transparency(UInt4 hit,
                                      UInt4 direction,
                                      UInt4 position,
                                      UInt4 normal,
                                      UInt4 geometric,
                                      UInt4 dpdu,
                                      unsigned int object)
{
  using namespace ccl;
  const auto path = cycles_svm_shadow_context(CYCLES_WEBGPU_SVM_PATH, object, normal.w);
  const auto surface = [&](const unsigned int field) -> WebGPUSvmContextWords {
    switch (field) {
      case CYCLES_WEBGPU_SVM_POSITION:
        return position;
      case CYCLES_WEBGPU_SVM_NORMAL:
        return normal;
      case CYCLES_WEBGPU_SVM_GEOMETRIC:
        return geometric;
      case CYCLES_WEBGPU_SVM_DIRECTION:
        return direction;
      case CYCLES_WEBGPU_SVM_INTERSECTION:
        return {direction.w, hit.z, hit.w, position.w};
      case CYCLES_WEBGPU_SVM_PATH:
        return {PATH_RAY_VISIBILITY_SHADOW, PATH_RAY_FLAG_NONE, 0, path.w};
      case CYCLES_WEBGPU_SVM_TRIANGLE_DPDU:
        return dpdu;
      default:
        return cycles_svm_shadow_context(field, object, normal.w);
    }
  };
  WebGPUSvmMaterial material;
  if (!read_material(normal.w,
                     hit.x,
                     __uint_as_float(hit.z),
                     __uint_as_float(hit.w),
                     0,
                     false,
                     material,
                     surface,
                     (path.w & CYCLES_WEBGPU_SHADER_ORDERED_CLOSURES) != 0u))
  {
    return {};
  }
  if (material.mixed_transparency) {
    return {material.transparency.x, material.transparency.y, material.transparency.z, 0.0f};
  }
  const bool transparent = material.surface_closure == CYCLES_WEBGPU_MATERIAL_TRANSPARENT ||
                           (material.surface_closure == CYCLES_WEBGPU_MATERIAL_HAIR_HUANG &&
                            cycles_huang_is_transparent(material, normal, direction));
  return transparent ? Float4{material.color.x, material.color.y, material.color.z, 0.0f} :
                       Float4{};
}
#else
void cycles_svm_material(unsigned int shader,
                         unsigned int primitive,
                         float u,
                         float v,
                         unsigned int lane,
                         unsigned int eval_options,
                         Float4 &base,
                         Float4 &specular,
                         Float4 &layers,
                         Float4 &transmission,
                         Float4 &subsurface,
                         Float4 &emission,
                         Float4 &closure_normal)
{
  specular = {};
  layers = {};
  transmission = {};
  subsurface = {};
  emission = {};
  closure_normal = {};
  ccl::WebGPUSvmMaterial material;
  const auto surface = [lane](const unsigned int field) { return cycles_svm_context(lane, field); };
  const bool subsurface_exit = (eval_options & CYCLES_WEBGPU_SVM_EVAL_SUBSURFACE_EXIT) != 0;
#ifdef CYCLES_WEBGPU_ORDERED_CLOSURES
  /* Use one evaluator call for both modes. Short-circuiting the context read
   * can make LLVM duplicate the whole interpreter during shader inlining. */
  const unsigned int ordered_mask = subsurface_exit ? 0u : CYCLES_WEBGPU_SHADER_ORDERED_CLOSURES;
  const bool ordered_closures = (surface(CYCLES_WEBGPU_SVM_PATH).w & ordered_mask) != 0u;
#else
  constexpr bool ordered_closures = false;
#endif
  if (!read_material(shader, primitive, u, v, lane,
                     (eval_options & CYCLES_WEBGPU_SVM_EVAL_AOV) != 0, material, surface,
                     ordered_closures))
  {
    base = subsurface_exit ? Float4{1, 1, 1, 1} : Float4{0, 0, 0, 2};
    return;
  }

  if (ordered_closures) {
    base = {0.0f, float(material.ordered_lobe_count), 0.0f, material.surface_closure};
    if (material.surface_closure == CYCLES_WEBGPU_MATERIAL_ORDERED_PRINCIPLED) {
      const auto first = material.ordered_principled_tangent_0;
      const auto second = material.ordered_principled_tangent_1;
      const auto coat = material.ordered_principled_coat;
      specular = {first.x, first.y, first.z, first.w};
      layers = {second.x, second.y, second.z, second.w};
      transmission = {coat.x, coat.y, coat.z, coat.w};
    }
    subsurface = {material.transparency.x, material.transparency.y, material.transparency.z,
                  float(material.ordered_lobe_features)};
    return;
  }

  if (subsurface_exit) {
    using namespace ccl;
    /* Stock integrate_surface() uses the exit shader only for its BSSRDF
     * normal, then replaces all closures and emission with white diffuse. */
    base = {1, 1, 1, 1};
    const bool burley_diffuse = material.subsurface > 0.0f &&
                                material.subsurface_method == CLOSURE_BSSRDF_BURLEY_ID &&
                                (surface(CYCLES_WEBGPU_SVM_PATH).y & PATH_RAY_DIFFUSE_ANCESTOR);
    float bssrdf_weight = 0.0f;
    if (material.subsurface > 0.0f) {
      const float metal_attenuation = material.metallic > CLOSURE_WEIGHT_CUTOFF ?
                                          1.0f - material.metallic : 1.0f;
      bssrdf_weight = average(saturate(material.color)) * material.subsurface *
                      material.mix_weight * metal_attenuation;
    }
    if (!burley_diffuse && bssrdf_weight >= CLOSURE_WEIGHT_CUTOFF &&
        (material.closure_normal_overridden || material.displacement_bump))
    {
      ShaderData sd{};
      sd.Ng = webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_GEOMETRIC));
      sd.wi = -webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_DIRECTION));
      sd.shader_flag = (eval_options & CYCLES_WEBGPU_SVM_EVAL_BUMP_CORRECTION) != 0 ?
                           SD_USE_BUMP_MAP_CORRECTION : 0;
      const float3 N = material.closure_normal_overridden ? material.closure_normal :
                                                           material.displacement_normal;
      const float3 corrected = normalize(maybe_ensure_valid_specular_reflection(&sd, N));
      closure_normal = {corrected.x, corrected.y, corrected.z, CYCLES_WEBGPU_NORMAL_CLOSURE};
      if (material.displacement_bump) {
        const auto original = material.displacement_normal;
        closure_normal.w = CYCLES_WEBGPU_NORMAL_BOTH;
        subsurface = {original.x, original.y, original.z, 0.0f};
      }
    }
    else if (material.displacement_bump) {
      const auto N = material.displacement_normal;
      closure_normal = {N.x, N.y, N.z, CYCLES_WEBGPU_NORMAL_SHADER};
    }
    return;
  }

  if (material.transparent_emission != 0u && material.transparent_sample_weight == 0.0f) {
    /* The host keeps the Transparent candidate for shadow metadata. A hit
     * with zero stock closure weight contributes emission as an opaque surface. */
    material.surface_closure = 0.0f;
  }
  if (material.surface_closure == CYCLES_WEBGPU_MATERIAL_HAIR_HUANG &&
      cycles_huang_is_transparent(material, surface(CYCLES_WEBGPU_SVM_NORMAL),
                                  surface(CYCLES_WEBGPU_SVM_DIRECTION)))
  {
    material.surface_closure = CYCLES_WEBGPU_MATERIAL_TRANSPARENT;
  }
  if (material.closure_normal_overridden) {
    closure_normal = {material.closure_normal.x, material.closure_normal.y,
                      material.closure_normal.z, material.displacement_bump ?
                          CYCLES_WEBGPU_NORMAL_BOTH : CYCLES_WEBGPU_NORMAL_CLOSURE};
  }
  else if (material.displacement_bump) {
    const auto N = material.displacement_normal;
    closure_normal = {N.x, N.y, N.z, CYCLES_WEBGPU_NORMAL_SHADER};
  }
  const auto path = surface(CYCLES_WEBGPU_SVM_PATH);
  const bool reflective_caustics =
      (path.w & CYCLES_WEBGPU_SHADER_NO_REFLECTIVE_CAUSTICS) == 0u ||
      (path.x & ccl::PATH_RAY_VISIBILITY_DIFFUSE) == 0u;
  const bool glossy_transparent =
      material.surface_closure == CYCLES_WEBGPU_MATERIAL_GLOSSY_TRANSPARENT;
  const bool suppress_glossy = !reflective_caustics &&
      (CYCLES_WEBGPU_MATERIAL_IS_GLOSSY(material.surface_closure) ||
       material.surface_closure == CYCLES_WEBGPU_MATERIAL_GLOSSY_TOON);
  base = suppress_glossy ? Float4{0, 0, 0, 1} :
                          Float4{material.color.x, material.color.y, material.color.z, 1.0f};
  /* The indirect clamp rides along for the path update's film_clamp_light(). */
  const auto emission_rgb = pack_surface_emission(material, surface, reflective_caustics);
  emission = {emission_rgb.x,
              emission_rgb.y,
              emission_rgb.z,
              ccl::__uint_as_float(cycles_svm_word(ccl::webgpu_svm_header_words + cycles_svm_word(0) +
                                                  ccl::webgpu_svm_indirect_clamp_offset))};
  if (material.holdout) {
    base.w = CYCLES_WEBGPU_MATERIAL_HOLDOUT;
  }
  else if (material.surface_closure != 0.0f && (!suppress_glossy || glossy_transparent)) {
    base.w = material.surface_closure;
    if (material.surface_closure == CYCLES_WEBGPU_MATERIAL_MIXED_TRANSPARENT) {
      specular = {material.transparency.x, material.transparency.y, material.transparency.z,
                  material.transparent_sample_weight};
      layers = {material.diffuse_sample_weight, float(material.transparent_first), 0.0f, 0.0f};
    }
    else if (material.surface_closure == CYCLES_WEBGPU_MATERIAL_DIFFUSE_TRANSLUCENT) {
      specular = {material.translucent_color.x, material.translucent_color.y,
                  material.translucent_color.z, 0.0f};
      layers = {material.diffuse_sample_weight, material.translucent_sample_weight,
                float(material.translucent_first), 0.0f};
    }
    else if (material.surface_closure == CYCLES_WEBGPU_MATERIAL_DIFFUSE_GLOSSY) {
      const auto glossy = reflective_caustics ? material.glossy_color : ccl::zero_spectrum();
      specular = {glossy.x, glossy.y, glossy.z, material.surface_parameters.x};
      layers = {material.diffuse_sample_weight,
                reflective_caustics ? material.glossy_sample_weight : 0.0f,
                float(material.mixed_glossy_flags), float(material.surface_model)};
      if (material.mixed_glossy_flags & CYCLES_WEBGPU_MIXED_BUMP_INDEPENDENT) {
        const auto N = material.mixed_glossy_normal;
        subsurface = {N.x, N.y, N.z, 0.0f};
      }
    }
    else if (material.surface_closure == CYCLES_WEBGPU_MATERIAL_RAY_PORTAL) {
      specular = {material.surface_fresnel_a.x,
                  material.surface_fresnel_a.y,
                  material.surface_fresnel_a.z, 0.0f};
      layers = {material.surface_fresnel_b.x,
                material.surface_fresnel_b.y,
                material.surface_fresnel_b.z, 0.0f};
    }
    else if (CYCLES_WEBGPU_MATERIAL_IS_HAIR(material.surface_closure)) {
      specular = {material.surface_tangent.x, material.surface_tangent.y,
                  material.surface_tangent.z, material.surface_parameters.x};
      layers = {material.surface_fresnel_a.x, material.surface_fresnel_a.y,
                material.surface_fresnel_a.z, material.surface_parameters.y};
      transmission = {material.surface_fresnel_b.x, material.surface_fresnel_b.y,
                      material.surface_fresnel_b.z, material.surface_hair_ior};
      if (material.surface_closure == CYCLES_WEBGPU_MATERIAL_HAIR_CHIANG) {
        const unsigned int color = ccl::webgpu_svm_header_words + cycles_svm_word(0) +
                                   ccl::webgpu_svm_color_space_offset;
        subsurface = {ccl::__uint_as_float(cycles_svm_word(color)),
                      ccl::__uint_as_float(cycles_svm_word(color + 1)),
                      ccl::__uint_as_float(cycles_svm_word(color + 2)), 0.0f};
      }
      else {
        subsurface.x = material.surface_hair_aspect;
      }
    }
    else if (CYCLES_WEBGPU_MATERIAL_IS_GLOSSY(material.surface_closure)) {
      specular = {material.surface_tangent.x,
                  material.surface_tangent.y,
                  material.surface_tangent.z,
                  material.surface_parameters.x};
      layers = {material.surface_fresnel_a.x,
                material.surface_fresnel_a.y,
                material.surface_fresnel_a.z,
                material.surface_parameters.y};
      transmission = {material.surface_fresnel_b.x,
                      material.surface_fresnel_b.y,
                      material.surface_fresnel_b.z,
                      float(material.surface_model)};
      if (glossy_transparent) {
        subsurface = {material.transparency.x, material.transparency.y, material.transparency.z,
                      material.transparent_first ? -material.transparent_sample_weight :
                                                   material.transparent_sample_weight};
      }
    }
    else if (material.surface_closure == CYCLES_WEBGPU_MATERIAL_SHEEN) {
      const unsigned int texture = ccl::webgpu_svm_header_words + cycles_svm_word(0);
      const auto wi = -ccl::webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_DIRECTION));
      const auto N = ccl::webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_NORMAL));
      const PackedSheen sheen = pack_sheen(texture, wi, N, material.color, material.surface_parameters.x);
      if (ccl::is_zero(sheen.weight)) {
        base = {0, 0, 0, 1};
      }
      else {
        base = {sheen.weight.x, sheen.weight.y, sheen.weight.z, CYCLES_WEBGPU_MATERIAL_SHEEN};
        specular = {sheen.ltc.x, sheen.ltc.y, 0, 0};
      }
    }
    else {
      specular = {material.surface_parameters.x, material.surface_parameters.y, 0, 0};
    }
  }
  else if (material.ggx) {
    base.w = material.multiscatter ? CYCLES_WEBGPU_MATERIAL_MULTISCATTER_GGX :
                                     CYCLES_WEBGPU_MATERIAL_GGX;
    specular = {material.specular_tint.x,
                material.specular_tint.y,
                material.specular_tint.z,
                material.alpha};
    layers = {material.metallic, material.eta, material.f0, material.mix_weight};
    float transmission_weight = material.transmission;
    if (!reflective_caustics) {
      /* Stock Principled still attenuates the lower layers by metallic, but
       * skips both reflective closures and dielectric specular attenuation. */
      const float metallic = material.metallic > CLOSURE_WEIGHT_CUTOFF ?
                                 material.metallic : 0.0f;
      layers = {0.0f, 1.0f, 0.0f, material.mix_weight * (1.0f - metallic)};
      /* Transmission eligibility requires both caustics settings to agree.
       * Stock Principled keeps the transmission attenuation on diffuse rays
       * with both caustics disabled, even when it skips creating Glass. */
      if (transmission_weight > CLOSURE_WEIGHT_CUTOFF) {
        layers.w *= 1.0f - transmission_weight;
        transmission_weight = 0.0f;
      }
    }
    /* Cycles converts Burley to a diffuse BSDF after the first diffuse bounce. */
    const bool burley_after_diffuse =
        material.subsurface > 0.0f &&
        material.subsurface_method == ccl::CLOSURE_BSSRDF_BURLEY_ID &&
        (path.y & ccl::PATH_RAY_DIFFUSE_ANCESTOR);
    const float subsurface_weight = burley_after_diffuse ? 0.0f : material.subsurface;
    transmission = {transmission_weight,
                    material.glass_ior,
                    subsurface_weight > 0.0f ? float(material.subsurface_method) :
                                              material.standalone_transmission,
                    subsurface_weight};
    if (reflective_caustics && material.coat.weight > CLOSURE_WEIGHT_CUTOFF &&
        material.mix_weight > 0.0f)
    {
      /* Preserve the BSSRDF's IOR, method and weight. The base specular tint
       * is white for this layout, freeing two words for the Coat controls. */
      transmission.x = -material.coat.weight * material.mix_weight;
      specular.x = material.coat.ior;
      specular.y = material.coat.alpha;
    }
    if (material.principled_translucent != 0u && material.translucent_sample_weight > 0.0f) {
      const auto color = material.translucent_color;
      const unsigned int translucent_flags = CYCLES_WEBGPU_GGX_TRANSLUCENT |
          (material.translucent_first ? CYCLES_WEBGPU_GGX_TRANSLUCENT_FIRST : 0u) |
          ((eval_options & CYCLES_WEBGPU_SVM_EVAL_BUMP_CORRECTION) ?
               CYCLES_WEBGPU_GGX_TRANSLUCENT_CORRECTION : 0u);
      transmission = {color.x, color.y, color.z, -float(translucent_flags)};
    }
    if (subsurface_weight > 0.0f) {
      subsurface = {material.subsurface_radius.x,
                    material.subsurface_radius.y,
                    material.subsurface_radius.z,
                    material.subsurface_anisotropy};
    }
    else if (material.anisotropy_aspect > 0.0f) {
      transmission.z = -material.anisotropy_aspect;
      subsurface = {material.surface_tangent.x,
                    material.surface_tangent.y,
                    material.surface_tangent.z,
                    0.0f};
    }
    else if (material.mixed_transparency) {
      subsurface = {material.transparency.x,
                    material.transparency.y,
                    material.transparency.z,
                    material.transparent_first ? -material.transparent_sample_weight :
                                                  material.transparent_sample_weight};
    }
  }
  if (closure_normal.w == CYCLES_WEBGPU_NORMAL_BOTH) {
    const auto N = material.displacement_normal;
    subsurface = {N.x, N.y, N.z, 0.0f};
  }
}
#endif
