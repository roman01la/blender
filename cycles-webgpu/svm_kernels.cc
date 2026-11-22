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
#include "../blender-platform/cycles-webgpu/svm_material_eval.h"
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

namespace {
unsigned int packed_lookup_offset(const unsigned int texture, const unsigned int kind)
{
  const unsigned int lookup = cycles_svm_word(texture + ccl::webgpu_svm_ies_offset);
  if (cycles_svm_word(1) & CYCLES_WEBGPU_SVM_SHEEN) {
    return cycles_svm_word(texture + lookup + kind);
  }
  return kind == ccl::webgpu_svm_lookup_ies ? lookup : 0;
}

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
  bool attribute_sample(const unsigned int slot, ccl::WebGPUSvmAttributeSample &sample) const
  {
    sample = ccl::webgpu_svm_triangle_attribute(cycles_svm_word, texture, slot, primitive, u, v);
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
bool read_material(const unsigned int shader,
                   const unsigned int primitive,
                   const float u,
                   const float v,
                   const unsigned int lane,
                   const bool film_aov,
                   ccl::WebGPUSvmMaterial &material,
                   const FetchSurface surface)
{
  const auto fetch = [](const unsigned int index) {
    return cycles_svm_word(ccl::webgpu_svm_header_words + index);
  };
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
  return ccl::webgpu_svm_eval_material(
      fetch, image, uv, generated, color_space, uses_uv, uses_generated, word_count,
      cycles_svm_word(1) & CYCLES_WEBGPU_SVM_SHADER_COUNT, shader, material, surface,
      PackedWordsHost{{}, texture, primitive, u, v, lane, film_aov});
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
        return {PATH_RAY_VISIBILITY_NONE, PATH_RAY_EMISSION, cycles_svm_context(lane, field).z, 0};
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
  const Spectrum emission = material.emission * emission_pdf;
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
        return {PATH_RAY_VISIBILITY_SHADOW, PATH_RAY_FLAG_NONE, 0, 0};
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
                     surface))
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
                         unsigned int film_aov,
                         Float4 &base,
                         Float4 &specular,
                         Float4 &layers,
                         Float4 &transmission,
                         Float4 &subsurface,
                         Float4 &emission)
{
  specular = {};
  layers = {};
  transmission = {};
  subsurface = {};
  emission = {};
  ccl::WebGPUSvmMaterial material;
  const auto surface = [lane](const unsigned int field) { return cycles_svm_context(lane, field); };
  if (!read_material(shader, primitive, u, v, lane, film_aov != 0, material, surface)) {
    base = {0, 0, 0, 2};
    return;
  }

  if (material.surface_closure == CYCLES_WEBGPU_MATERIAL_HAIR_HUANG &&
      cycles_huang_is_transparent(material, surface(CYCLES_WEBGPU_SVM_NORMAL),
                                  surface(CYCLES_WEBGPU_SVM_DIRECTION)))
  {
    material.surface_closure = CYCLES_WEBGPU_MATERIAL_TRANSPARENT;
  }
  base = {material.color.x, material.color.y, material.color.z, 1.0f};
  /* The indirect clamp rides along for the path update's film_clamp_light(). */
  emission = {material.emission.x,
              material.emission.y,
              material.emission.z,
              ccl::__uint_as_float(cycles_svm_word(ccl::webgpu_svm_header_words + cycles_svm_word(0) +
                                                  ccl::webgpu_svm_indirect_clamp_offset))};
  if (material.holdout) {
    base.w = CYCLES_WEBGPU_MATERIAL_HOLDOUT;
  }
  else if (material.surface_closure != 0.0f) {
    base.w = material.surface_closure;
    if (material.surface_closure == CYCLES_WEBGPU_MATERIAL_MIXED_TRANSPARENT) {
      specular = {material.transparency.x, material.transparency.y, material.transparency.z,
                  material.transparent_sample_weight};
      layers = {material.diffuse_sample_weight, float(material.transparent_first), 0.0f, 0.0f};
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
    else if (material.surface_closure == CYCLES_WEBGPU_MATERIAL_GLOSSY) {
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
    }
    else if (material.surface_closure == CYCLES_WEBGPU_MATERIAL_SHEEN) {
      ccl::SvmGlobals globals;
      globals.data.tables.sheen_ltc = 0;
      const unsigned int texture = ccl::webgpu_svm_header_words + cycles_svm_word(0);
      globals.lookup_table = {texture + packed_lookup_offset(texture, ccl::webgpu_svm_lookup_sheen)};
      ccl::ShaderData sd{};
      sd.num_closure_left = 1;
      sd.wi = -ccl::webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_DIRECTION));
      const auto N = ccl::webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_NORMAL));
      ccl::bsdf_sheen_setup(&globals, &sd, material.color, N, material.surface_parameters.x);
      if (sd.num_closure == 0 || sd.closure[0].type != ccl::CLOSURE_BSDF_SHEEN_ID) {
        base = {0, 0, 0, 1};
      }
      else {
        const auto *sheen = reinterpret_cast<const ccl::SheenBsdf *>(&sd.closure[0]);
        base = {sheen->weight.x, sheen->weight.y, sheen->weight.z, CYCLES_WEBGPU_MATERIAL_SHEEN};
        specular = {sheen->transformA, sheen->transformB, 0, 0};
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
    /* Cycles converts Burley to a diffuse BSDF after the first diffuse bounce. */
    const bool burley_after_diffuse =
        material.subsurface > 0.0f &&
        material.subsurface_method == ccl::CLOSURE_BSSRDF_BURLEY_ID &&
        (surface(CYCLES_WEBGPU_SVM_PATH).y & ccl::PATH_RAY_DIFFUSE_ANCESTOR);
    const float subsurface_weight = burley_after_diffuse ? 0.0f : material.subsurface;
    transmission = {material.transmission,
                    material.glass_ior,
                    subsurface_weight > 0.0f ? float(material.subsurface_method) :
                                              material.standalone_transmission,
                    subsurface_weight};
    if (subsurface_weight > 0.0f) {
      subsurface = {material.subsurface_radius.x,
                    material.subsurface_radius.y,
                    material.subsurface_radius.z,
                    material.subsurface_anisotropy};
    }
    else if (material.mixed_transparency) {
      subsurface = {material.transparency.x,
                    material.transparency.y,
                    material.transparency.z,
                    material.transparent_first ? -material.transparent_sample_weight :
                                                  material.transparent_sample_weight};
    }
  }
}
#endif
