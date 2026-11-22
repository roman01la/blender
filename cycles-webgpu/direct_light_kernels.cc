/* SPDX-License-Identifier: Apache-2.0 */

#include <bit>

#ifdef __aarch64__
#  include <arm_neon.h>
#endif

#define CCL_NAMESPACE_BEGIN namespace ccl {
#define CCL_NAMESPACE_END }
#include "kernel/closure/bsdf_diffuse.h"
#include "kernel/sample/lcg.h"

#define __KERNEL_GPU__
#include "kernel/globals.h"
#include "util/types_image.h"
#undef __KERNEL_GPU__
/* The packed light data contains static triangle vertices and transforms. */
#undef __OBJECT_MOTION__

__attribute__((pure)) unsigned int cycles_light_word(unsigned int index);
#ifdef CYCLES_WEBGPU_SHADE_BACKGROUND
unsigned int cycles_film_transparent();
#endif

namespace ccl {

struct LightGlobals {
  KernelData data;
  uint sampling_offset;
#ifdef CYCLES_WEBGPU_POINT_DISTRIBUTION
  uint distribution_offset;
#endif
#ifdef CYCLES_WEBGPU_LIGHT_TREE
  uint tree_lamp_offset, tree_node_offset, tree_emitter_offset;
  uint tree_object_offset, tree_triangle_indices_offset, tree_triangle_offset;
  uint tree_shader_offset, tree_triangle_to_tree_offset;
  uint world_marginal_offset, world_conditional_offset;
  /* Stock geometry helpers can borrow the decoded object for one fetch. */
  mutable KernelObject object{};
#endif
};

using KernelGlobals = const LightGlobals *;

#define KERNEL_DATA_ARRAY(type, name) \
  struct name##_tag { \
    using Element = type; \
  };
#include "kernel/data_arrays.h"

/* The point-light sampler does not read scene globals. Related functions in the
 * upstream header still need declarations, with no definition if called. */
template<typename Tag> const typename Tag::Element &light_fetch(KernelGlobals, Tag, uint);

#ifdef CYCLES_WEBGPU_LIGHT_TREE
ccl_device_inline const KernelObject &light_fetch(KernelGlobals kg, objects_tag, uint index);
ccl_device_inline uint light_fetch(KernelGlobals kg, object_flag_tag, uint object);
#endif

#ifdef CYCLES_WEBGPU_WAVEFRONT_LIGHT
inline float light_fetch(KernelGlobals kg, sample_pattern_lut_tag, uint index)
{
  return std::bit_cast<float>(cycles_light_word(kg->sampling_offset + index));
}
#endif

}  // namespace ccl

#define kernel_data (kg->data)
#define kernel_data_fetch(name, index) light_fetch(kg, name##_tag{}, index)
#include "kernel/light/point.h"

#ifdef CYCLES_WEBGPU_WAVEFRONT_LIGHT
#  include "../blender-platform/cycles-webgpu/material_layout.h"
#  include "../blender-platform/cycles-webgpu/lighting_layout.h"
#  ifdef CYCLES_WEBGPU_POINT_DISTRIBUTION
namespace ccl {

inline KernelLightDistribution light_fetch(KernelGlobals kg, light_distribution_tag, uint index)
{
  const uint word = kg->distribution_offset + CYCLES_WEBGPU_POINT_DISTRIBUTION_WORDS * index;
  return {std::bit_cast<float>(cycles_light_word(word)),
          std::bit_cast<int>(cycles_light_word(word + 1)),
          0,
          0};
}

}  // namespace ccl
#    include "kernel/light/distribution.h"
#  endif
#  include "../blender-platform/cycles-webgpu/svm_material_eval.h"
#  undef ccl_device
#  define ccl_device static inline __attribute__((always_inline))
#  include "kernel/camera/projection.h"
/* Metal can round a nonzero sine to zero in the spherical rectangle sampler.
 * At this magnitude sin(angle) correctly rounds to angle at float precision. */
static inline float cycles_rectangle_sinf(const float angle)
{
  return fabsf(angle) < 0x1p-12f ? angle : sinf(angle);
}
#  define sinf cycles_rectangle_sinf
#  include "kernel/light/area.h"
#  undef sinf
#  ifdef CYCLES_WEBGPU_LIGHT_TREE
#    include "kernel/bvh/util.h"
namespace ccl {

inline float2 light_fetch(KernelGlobals kg, light_background_marginal_cdf_tag, uint index)
{
  const uint word = kg->world_marginal_offset + 2 * index;
  return make_float2(std::bit_cast<float>(cycles_light_word(word)),
                     std::bit_cast<float>(cycles_light_word(word + 1)));
}

inline float2 light_fetch(KernelGlobals kg, light_background_conditional_cdf_tag, uint index)
{
  const uint word = kg->world_conditional_offset + 2 * index;
  return make_float2(std::bit_cast<float>(cycles_light_word(word)),
                     std::bit_cast<float>(cycles_light_word(word + 1)));
}

}  // namespace ccl
#    include "light_tree_adapter.h"
#  endif
#  include "kernel/sample/mis.h"
#  include "kernel/sample/pattern.h"
#  include "ggx_closures.h"
#endif

using UInt4 = unsigned int __attribute__((ext_vector_type(4)));
using Float4 = float __attribute__((ext_vector_type(4)));

#ifdef CYCLES_WEBGPU_WAVEFRONT_LIGHT
float cycles_ggx_lookup(const unsigned int index)
{
  return std::bit_cast<float>(cycles_light_word(CYCLES_WEBGPU_LIGHT_HEADER_WORDS + index));
}
#endif

void cycles_bvh2_trace(
    UInt4 origin, UInt4 direction, UInt4 options, UInt4 &hit, UInt4 &coordinates
#ifdef CYCLES_WEBGPU_WAVEFRONT_LIGHT
    , UInt4 geometric
#endif
);

namespace {

#ifdef CYCLES_WEBGPU_WAVEFRONT_LIGHT
bool is_ggx(const Float4 material)
{
  return material.w == CYCLES_WEBGPU_MATERIAL_GGX ||
         material.w == CYCLES_WEBGPU_MATERIAL_MULTISCATTER_GGX;
}
#endif

enum SampleStatus : unsigned int {
  Background,
  Unsupported,
  Backlit,
  Shadowed,
  Lit,
};

ccl::float3 unpack_rgb(const UInt4 words)
{
  return ccl::make_float3(
      std::bit_cast<float>(words.x), std::bit_cast<float>(words.y), std::bit_cast<float>(words.z));
}

ccl::float3 read_rgb(const unsigned int offset)
{
  return unpack_rgb(UInt4{
      cycles_light_word(offset), cycles_light_word(offset + 1), cycles_light_word(offset + 2), 0});
}

UInt4 pack_ray(const ccl::float3 v, const float t)
{
  return {std::bit_cast<unsigned int>(v.x),
          std::bit_cast<unsigned int>(v.y),
          std::bit_cast<unsigned int>(v.z),
          std::bit_cast<unsigned int>(t)};
}

#ifdef CYCLES_WEBGPU_WAVEFRONT_LIGHT
float read_float(const unsigned int offset)
{
  return std::bit_cast<float>(cycles_light_word(offset));
}

[[gnu::always_inline]] void setup_light_sampling(ccl::LightGlobals &globals,
                                                const unsigned int words)
{
  globals.sampling_offset = words + CYCLES_WEBGPU_LIGHT_SAMPLING_WORDS;
  auto &integrator = globals.data.integrator;
  integrator.sampling_pattern = cycles_light_word(words);
  integrator.sobol_index_mask = cycles_light_word(words + 2);
  integrator.blue_noise_sequence_length = cycles_light_word(words + 3);
  integrator.tabulated_sobol_sequence_size = cycles_light_word(words + 4);
  integrator.pixel_jitter = ccl::make_float2(read_float(words + 5), read_float(words + 6));
  integrator.scrambling_distance = read_float(words + 7);
}

[[gnu::always_inline]] ccl::float3 light_sample_random(ccl::KernelGlobals globals,
                                                      const unsigned int rng_pixel,
                                                      const unsigned int sample,
                                                      const unsigned int bounce)
{
  const int dimension = (int(bounce) + 1) * ccl::PRNG_BOUNCE_NUM + ccl::PRNG_LIGHT;
  return ccl::path_rng_3D(globals, rng_pixel, sample, dimension);
}

#  ifdef CYCLES_WEBGPU_POINT_DISTRIBUTION
bool sample_point_distribution(const unsigned int descriptor,
                               const UInt4 sampling,
                               const UInt4 path,
                               const ccl::float3 P,
                               const ccl::float3 N,
                               const int runtime_flags,
                               ccl::LightSample &light,
                               unsigned int &strength_offset,
                               const unsigned int subsurface_steps)
{
  ccl::LightGlobals globals;
  setup_light_sampling(globals, cycles_light_word(descriptor + CYCLES_WEBGPU_POINT_SAMPLING));
  globals.distribution_offset = cycles_light_word(descriptor + CYCLES_WEBGPU_POINT_CDF);
  globals.data.integrator.num_distribution = int(
      cycles_light_word(descriptor + CYCLES_WEBGPU_POINT_COUNT));
  globals.data.integrator.distribution_pdf_lights = read_float(descriptor +
                                                               CYCLES_WEBGPU_POINT_PDF);

  /* Cycles reserves the first two light random dimensions for position sampling. */
  const auto random = light_sample_random(&globals, sampling.x, sampling.y, path.x + subsurface_steps);
  ccl::light_distribution_sample(&globals, random.z, &light);
  const int lamp_id = ~ccl::light_fetch(
      &globals, ccl::light_distribution_tag{}, unsigned(light.emitter_id)).prim;
  const unsigned int lamp_offset = cycles_light_word(descriptor + CYCLES_WEBGPU_POINT_EMITTERS) +
                                   CYCLES_WEBGPU_POINT_EMITTER_WORDS * unsigned(lamp_id);
  if (float(path.x) > read_float(lamp_offset + 3)) {
    return false;
  }

  ccl::KernelLight lamp{};
  lamp.co = read_rgb(lamp_offset);
  lamp.spot.eval_fac = 0.25f * M_1_PI_F;
  if (!ccl::point_light_sample(&lamp, ccl::make_float2(random), P, N, runtime_flags, &light)) {
    return false;
  }
  light.pdf *= light.pdf_selection;
  strength_offset = lamp_offset + 4;
  return true;
}
#  endif

void unpack_area_light(const unsigned int descriptor, ccl::KernelLight &lamp)
{
  lamp.co = read_rgb(0);
  lamp.area.axis_u = read_rgb(descriptor + CYCLES_WEBGPU_AREA_AXIS_U);
  lamp.area.len_u = read_float(descriptor + CYCLES_WEBGPU_AREA_AXIS_U + 3);
  lamp.area.axis_v = read_rgb(descriptor + CYCLES_WEBGPU_AREA_AXIS_V);
  lamp.area.len_v = read_float(descriptor + CYCLES_WEBGPU_AREA_AXIS_V + 3);
  lamp.area.dir = read_rgb(descriptor + CYCLES_WEBGPU_AREA_DIRECTION);
  lamp.area.invarea = read_float(descriptor + CYCLES_WEBGPU_AREA_DIRECTION + 3);
  /* The scene gate accepts 180-degree spread. Its attenuation is one; avoid
   * depending on the GPU preserving the stock subnormal normalization factor. */
  lamp.area.tan_half_spread = FLT_MAX;
  lamp.area.normalize_spread = 0.0f;
}

#  ifdef CYCLES_WEBGPU_LIGHT_TREE
[[gnu::always_inline]]
#  endif
ccl::Spectrum environment_radiance(const unsigned int width,
                                   const unsigned int height,
                                   const unsigned int interpolation,
                                   const unsigned int texel_offset,
                                   const ccl::float3 direction)
{
  /* This is the unconnected Vector input of the stock Environment Texture:
   * shader_setup_from_background stores the ray direction in ShaderData::P. */
  const auto normalized = ccl::safe_normalize(direction);
  if (ccl::is_zero(normalized)) {
    return read_rgb(texel_offset);
  }
  float u, v;
  if (normalized.x == 0.0f && normalized.y == 0.0f) {
    /* WGSL atan2 at the poles is undefined. Cycles wraps the image, so
     * longitude zero and one select the same texels. */
    const bool negative_x = (std::bit_cast<unsigned int>(normalized.x) >> 31) != 0;
    u = negative_x ? 0.0f : 0.5f;
    v = normalized.z > 0.0f ? 1.0f : 0.0f;
  }
  else {
    const auto uv = ccl::direction_to_equirectangular(normalized);
    u = uv.x;
    v = uv.y;
  }
  const auto texel = [&](const unsigned int index) {
    return read_rgb(texel_offset + 3 * index);
  };
  return ccl::webgpu_svm_sample_image(texel, width, height, ccl::EXTENSION_REPEAT,
                                      interpolation, ccl::make_float2(u, v));
}

#  ifdef CYCLES_WEBGPU_LIGHT_TREE
[[gnu::always_inline]] void setup_world_sampling(ccl::LightGlobals &globals, const unsigned int descriptor)
{
  globals.data.background.map_res_x = cycles_light_word(descriptor + CYCLES_WEBGPU_TREE_WORLD_MAP_WIDTH);
  globals.data.background.map_res_y = cycles_light_word(descriptor + CYCLES_WEBGPU_TREE_WORLD_MAP_HEIGHT);
  globals.world_marginal_offset = cycles_light_word(descriptor + CYCLES_WEBGPU_TREE_WORLD_MARGINAL_CDF);
  globals.world_conditional_offset = cycles_light_word(descriptor + CYCLES_WEBGPU_TREE_WORLD_CONDITIONAL_CDF);
}

[[gnu::always_inline]] ccl::Spectrum tree_environment(const unsigned int descriptor, const ccl::float3 direction)
{
  const unsigned int width = cycles_light_word(descriptor + CYCLES_WEBGPU_TREE_WORLD_WIDTH);
  if (width == 0) {
    return read_rgb(8);
  }
  return environment_radiance(width,
                              cycles_light_word(descriptor + CYCLES_WEBGPU_TREE_WORLD_HEIGHT),
                              cycles_light_word(descriptor + CYCLES_WEBGPU_TREE_WORLD_INTERPOLATION),
                              cycles_light_word(descriptor + CYCLES_WEBGPU_TREE_WORLD_PIXELS),
                              direction);
}

#    ifndef CYCLES_WEBGPU_SHADE_BACKGROUND
template<bool in_volume_segment, bool allow_triangles = true>
[[gnu::always_inline]] bool sample_selected_tree_light(ccl::LightGlobals &globals,
                                                       const unsigned int descriptor,
                                                       const ccl::float2 rand,
                                                       const unsigned int bounce,
                                                       const ccl::float3 P,
                                                       const ccl::float3 N,
                                                       const int runtime_flags,
                                                       ccl::LightSample &light,
                                                       ccl::Spectrum &strength,
                                                       bool &sample_emission)
{
  const auto emitter = ccl::light_fetch(&globals, ccl::light_tree_emitters_tag{}, light.emitter_id);
  if (allow_triangles && ccl::is_triangle(&emitter)) {
    /* The tree sampler keeps the selected instance in light.object. The emitter
     * record itself may belong to the instance sharing its mesh subtree. */
    const int object = light.object == OBJECT_NONE ? emitter.object_id : light.object;
    if (!ccl::triangle_light_sample<in_volume_segment>(
            &globals, emitter.triangle.id, object, rand, -1.0f, &light, P))
    {
      return false;
    }
    light.shader |= emitter.visibility_flag;
    const auto shader = ccl::light_tree_shader_word(&globals, light.shader & ccl::SHADER_MASK);
#      ifdef CYCLES_WEBGPU_DEFER_SHADOW
    sample_emission = cycles_light_word(descriptor + CYCLES_WEBGPU_TREE_TEXTURED_EMITTERS) != 0 &&
                      !(cycles_light_word(shader + CYCLES_WEBGPU_TREE_SHADER_FLAGS) &
                        ccl::SD_HAS_CONSTANT_EMISSION);
#      endif
    /* The sampled-light SVM stage applies the actual emission before shadow tracing. */
    strength = sample_emission ? ccl::one_spectrum() :
                                 read_rgb(shader + CYCLES_WEBGPU_TREE_SHADER_EMISSION);
    light.pdf *= light.pdf_selection;
    return in_volume_segment || light.pdf > 0.0f;
  }

  if (!ccl::is_light(&emitter)) {
    return false;
  }
  light.prim = ~emitter.light.id;
  const ccl::KernelLight lamp = ccl::light_fetch(&globals, ccl::lights_tag{}, light.prim);
  if (float(bounce) > lamp.max_bounces) {
    return false;
  }
  light.type = static_cast<ccl::LightType>(lamp.type);
  light.shader = lamp.shader_id;
  if (light.type == ccl::LIGHT_POINT) {
    if (!ccl::point_light_sample(&lamp, rand, P, N, runtime_flags, &light)) {
      return false;
    }
  }
  else if (light.type == ccl::LIGHT_AREA) {
    auto area = lamp;
    /* Only full-spread rectangles are admitted by the host. */
    area.area.invarea = fabsf(area.area.invarea);
    area.area.tan_half_spread = FLT_MAX;
    area.area.normalize_spread = 0.0f;
    if (!ccl::area_light_sample<in_volume_segment>(&area, rand, P, &light)) {
      return false;
    }
  }
  else {
    if constexpr (in_volume_segment) {
      /* Same dummy sample as light_sample<true> for distant emitters. */
      light.P = light.D = ccl::zero_float3();
      light.t = FLT_MAX;
      light.pdf = light.pdf_selection;
      light.eval_fac = 0.0f;
      strength = ccl::zero_spectrum();
      return true;
    }
    /* With only the world map enabled, the stock sampler uses its CDF directly. */
    setup_world_sampling(globals, descriptor);
    light.D = ccl::background_map_sample(&globals, rand, &light.pdf);
    light.t = FLT_MAX;
    light.eval_fac = 1.0f;
    strength = tree_environment(descriptor, light.D);
    light.pdf *= light.pdf_selection;
    return light.pdf > 0.0f;
  }

  const unsigned int lamp_word = globals.tree_lamp_offset +
                                 light.prim * CYCLES_WEBGPU_TREE_LAMP_WORDS;
  strength = read_rgb(lamp_word + CYCLES_WEBGPU_TREE_LAMP_STRENGTH);
  light.pdf *= light.pdf_selection;
  return in_volume_segment || light.pdf > 0.0f;
}

[[gnu::always_inline]] bool sample_mixed_lights(const unsigned int descriptor,
                                                const UInt4 sampling,
                                                const unsigned int bounce,
                                                const unsigned int subsurface_steps,
                                                const ccl::float3 P,
                                                const ccl::float3 N,
                                                const int runtime_flags,
                                                ccl::LightSample &light,
                                                ccl::Spectrum &strength,
                                                bool &sample_emission)
{
  ccl::LightGlobals globals;
  ccl::setup_light_tree(globals, descriptor);
  setup_light_sampling(globals, cycles_light_word(descriptor + CYCLES_WEBGPU_TREE_SAMPLING));
  const auto random = light_sample_random(&globals, sampling.x, sampling.y, bounce + subsurface_steps);
  light.object = OBJECT_NONE;
  return ccl::light_tree_sample<false>(
             &globals, random.z, P, N, 0.0f, OBJECT_NONE, runtime_flags, &light) &&
         sample_selected_tree_light<false>(globals, descriptor, ccl::make_float2(random), bounce,
                                           P, N, runtime_flags, light, strength, sample_emission);
}
#    endif
#  endif
#endif

struct ShadowRay {
  UInt4 origin, direction, options;
#if defined(CYCLES_WEBGPU_LIGHT_TREE) && defined(CYCLES_WEBGPU_DEFER_SHADOW)
  UInt4 emission;
#endif
};

#ifdef CYCLES_WEBGPU_LIGHT_TREE
[[gnu::always_inline]]
#endif
unsigned int direct_surface_light(UInt4 position,
                                  UInt4 normal,
                                  UInt4 hit,
                                  Float4 material,
                                  Float4 &result,
                                  [[maybe_unused]] UInt4 ray = {},
                                  [[maybe_unused]] Float4 specular = {},
                                  [[maybe_unused]] Float4 parameters = {},
                                  [[maybe_unused]] Float4 glass = {},
                                  [[maybe_unused]] UInt4 spec_normal = {},
                                  [[maybe_unused]] UInt4 geometric = {},
                                  [[maybe_unused]] UInt4 shadow_along = {},
                                  [[maybe_unused]] UInt4 shadow_opposite = {},
                                  [[maybe_unused]] UInt4 sampling = {},
                                  [[maybe_unused]] UInt4 path = {},
                                  [[maybe_unused]] unsigned int descriptor = 0,
                                  [[maybe_unused]] unsigned int subsurface_steps = 0,
                                  [[maybe_unused]] ShadowRay *shadow = nullptr,
                                  [[maybe_unused]] Float4 hair = {},
                                  [[maybe_unused]] unsigned int *hair_lcg = nullptr)
{
  result = {0, 0, 0, 1};
  const ccl::float3 P = unpack_rgb(position);
  const ccl::float3 N = unpack_rgb(normal);
  int runtime_flags = ccl::SR_BSDF;
#ifdef CYCLES_WEBGPU_WAVEFRONT_LIGHT
  const bool translucent = material.w == CYCLES_WEBGPU_MATERIAL_TRANSLUCENT;
  if (translucent || CYCLES_WEBGPU_MATERIAL_HAIR_TRANSMITS(material.w)) {
    runtime_flags |= ccl::SR_BSDF_HAS_TRANSMISSION;
  }
#endif
#if defined(CYCLES_WEBGPU_WAVEFRONT_LIGHT) && !defined(CYCLES_WEBGPU_REFLECTION_ONLY)
  if (is_ggx(material) && glass.x > 0.0f) {
    runtime_flags |= ccl::SR_BSDF_HAS_TRANSMISSION;
  }
#endif

  ccl::LightSample light{};
#ifdef CYCLES_WEBGPU_LIGHT_TREE
#  ifndef CYCLES_WEBGPU_SHADE_BACKGROUND
  ccl::Spectrum strength;
  bool sample_emission = false;
  if (!sample_mixed_lights(descriptor,
                           sampling,
                           path.x,
                           subsurface_steps,
                           P,
                           N,
                           runtime_flags,
                           light,
                           strength,
                           sample_emission))
  {
    return Backlit;
  }
#  else
  return Backlit;
#  endif
#else
  unsigned int strength_offset = 4;
#  ifdef CYCLES_WEBGPU_POINT_DISTRIBUTION
  if (!sample_point_distribution(
          descriptor, sampling, path, P, N, runtime_flags, light, strength_offset, subsurface_steps))
  {
    return Backlit;
  }
#else
  ccl::KernelLight lamp{};
#  ifdef CYCLES_WEBGPU_WAVEFRONT_LIGHT
  const bool area_light = descriptor != 0;
  if (area_light) {
    unpack_area_light(descriptor, lamp);
    ccl::LightGlobals globals;
    setup_light_sampling(globals, cycles_light_word(descriptor + CYCLES_WEBGPU_LIGHT_SAMPLING));
    const auto random = light_sample_random(&globals, sampling.x, sampling.y, path.x + subsurface_steps);
    if (!ccl::area_light_sample<false>(&lamp, ccl::make_float2(random), P, &light))
    {
      return Backlit;
    }
  }
  else
#  endif
  {
    /* A single normalized, radius-zero point light, as packed by PointLight::copy_to_kernel. */
    lamp.co = read_rgb(0);
    lamp.spot.eval_fac = 0.25f * M_1_PI_F;
    ccl::point_light_sample(&lamp, ccl::zero_float2(), P, N, runtime_flags, &light);
  }
#  endif
#endif

#if !defined(CYCLES_WEBGPU_LIGHT_TREE) || !defined(CYCLES_WEBGPU_SHADE_BACKGROUND)
  if (light.type == ccl::LIGHT_TRIANGLE &&
      light.prim == std::bit_cast<int>(hit.y) && light.object == std::bit_cast<int>(hit.z))
  {
    const auto Ng = unpack_rgb(geometric);
    const bool transmit = ccl::dot(light.D, N) < 0.0f;
    if (ccl::dot(light.D, transmit ? -Ng : Ng) > 0.0f) {
      return Backlit;
    }
  }
  bool valid_hemisphere;
  float bsdf_pdf;
  ccl::Spectrum bsdf;
#ifdef CYCLES_WEBGPU_WAVEFRONT_LIGHT
  if (CYCLES_WEBGPU_MATERIAL_IS_STANDALONE(material.w)) {
    const auto n = CYCLES_WEBGPU_MATERIAL_USES_SPECULAR_NORMAL(material.w) ?
                       unpack_rgb(spec_normal) : N;
    /* The stock bump shadowing term rejects a direction crossing the corrected
     * closure normal but still on the other side of the smooth surface normal. */
    if (ccl::dot(N, light.D) * ccl::dot(N, n) * ccl::dot(n, light.D) < 0.0f) {
      return Backlit;
    }
    const auto wi = -unpack_rgb(ray);
    const float blur = cycles_glossy_blur(read_float(CYCLES_WEBGPU_LIGHT_FILTER_GLOSSY),
                                         std::bit_cast<float>(path.w));
    const auto Ng = unpack_rgb(geometric);
    unsigned int lcg = hair_lcg ? *hair_lcg : 0u;
    /* Preserve all bits of the stock LCG, including patterns that encode NaN. */
    hair.z = float(lcg & 0xffffu);
    hair.w = float(lcg >> 16u);
    const Float4 eval = cycles_standalone_eval(
        material,
        specular,
        parameters,
        glass,
        hair,
        Float4{n.x, n.y, n.z, 0},
        Float4{Ng.x, Ng.y, Ng.z, 0},
        Float4{wi.x, wi.y, wi.z, 0},
        Float4{light.D.x, light.D.y, light.D.z, 0},
        blur,
        lcg);
    if (hair_lcg) {
      *hair_lcg = lcg;
    }
    bsdf = ccl::make_float3(eval.x, eval.y, eval.z);
    bsdf_pdf = eval.w;
    valid_hemisphere = bsdf_pdf > 0.0f;
  }
  else if (is_ggx(material)) {
    /* Glass lobes can also sample the light through the back of the interface. */
#  ifdef CYCLES_WEBGPU_REFLECTION_ONLY
    valid_hemisphere = ccl::dot(N, light.D) > 0.0f;
#  else
    valid_hemisphere = glass.x > 0.0f || ccl::dot(N, light.D) > 0.0f;
#  endif
    const auto wi = -unpack_rgb(ray);
    const auto spec_N = unpack_rgb(spec_normal);
    const float blur = cycles_glossy_blur(read_float(CYCLES_WEBGPU_LIGHT_FILTER_GLOSSY),
                                           std::bit_cast<float>(path.w));
    const Float4 eval = cycles_ggx_eval(material, specular, parameters, glass,
                                       Float4{N.x, N.y, N.z, position.w == 2 ? 1.0f : 0.0f},
                                       Float4{spec_N.x, spec_N.y, spec_N.z, 0},
                                       Float4{wi.x, wi.y, wi.z, 0},
                                       Float4{light.D.x, light.D.y, light.D.z, 0},
                                       blur,
                                       glass.w == 0.0f ? fabsf(hair.w) : 0.0f);
    bsdf = ccl::make_float3(eval.x, eval.y, eval.z);
    bsdf_pdf = eval.w;
  }
  else
#endif
  {
    ccl::DiffuseBsdf diffuse{};
    diffuse.N = N;
    const auto albedo = ccl::make_float3(material.x, material.y, material.z);
    bsdf = albedo * ccl::bsdf_diffuse_eval(
        reinterpret_cast<const ccl::ShaderClosure *>(&diffuse), N, light.D, &bsdf_pdf);
#ifdef CYCLES_WEBGPU_WAVEFRONT_LIGHT
    if (material.w == CYCLES_WEBGPU_MATERIAL_MIXED_TRANSPARENT) {
      const float weight_sum = parameters.x + specular.w;
      bsdf_pdf *= weight_sum > 0.0f ? parameters.x / weight_sum : 0.0f;
    }
#endif
    valid_hemisphere = bsdf_pdf > 0.0f;
  }
  if (!valid_hemisphere || !(light.pdf > 0.0f)) {
    return Backlit;
  }

  ccl::float3 shadow_origin = P;
  ccl::float3 shadow_direction = light.D;
  float shadow_distance = light.t;
  bool skip_self = true;
#ifdef CYCLES_WEBGPU_WAVEFRONT_LIGHT
  const float cutoff = std::bit_cast<float>(shadow_along.w);
  if (cutoff > 0.0f) {
    /* Match shadow_ray_offset() using the two offsets prepared by the surface shader. */
    const float NL = ccl::dot(N, light.D);
    const bool transmit = NL < 0.0f;
    const auto Ng = unpack_rgb(geometric);
    const float NgL = ccl::dot(transmit ? -Ng : Ng, light.D);
    const float abs_NL = fabsf(NL);
    const float amount = abs_NL < cutoff ?
                             ccl::clamp(2.0f - (NgL + abs_NL) / cutoff, 0.0f, 1.0f) :
                             ccl::clamp(1.0f - NgL / cutoff, 0.0f, 1.0f);
    if (amount > 0.0f) {
      shadow_origin += unpack_rgb(transmit ? shadow_opposite : shadow_along) * amount;
      skip_self = NgL > 0.0f;
    }
  }
  if (light.t != FLT_MAX) {
    /* Match Cycles' shadow_ray_setup() for a finite light sample. */
    shadow_direction = ccl::safe_normalize_len(light.P - shadow_origin, &shadow_distance);
  }
#endif

  ShadowRay shadow_ray = {
      pack_ray(shadow_origin, 0.0f),
      pack_ray(shadow_direction, shadow_distance),
      UInt4{ccl::PATH_RAY_VISIBILITY_SHADOW_OPAQUE,
            skip_self ? hit.z : static_cast<unsigned int>(OBJECT_NONE),
            skip_self ? hit.y : static_cast<unsigned int>(PRIM_NONE), 0}};
  if (light.type == ccl::LIGHT_TRIANGLE) {
    /* The shadow ray starts at tmin 0. These two unused words carry the
     * sampled primitive and object for the stock BVH endpoint exclusion. */
    shadow_ray.origin.w = unsigned(light.prim) + 1;
    shadow_ray.options.w = unsigned(light.object);
    if (!(light.shader & ccl::SHADER_CAST_SHADOW)) {
      shadow_ray.options.x = 0;
    }
#if defined(CYCLES_WEBGPU_LIGHT_TREE) && defined(CYCLES_WEBGPU_DEFER_SHADOW)
    if (sample_emission) {
      ccl::LightGlobals tree;
      ccl::setup_light_tree(tree, descriptor);
      /* Match integrator_shade_light_nee() using the offset shadow ray. */
      const auto uv = ccl::triangle_light_uv(
          &tree, light.object, light.prim, -1.0f, shadow_origin, shadow_direction);
      shadow_ray.emission = {shadow_ray.origin.w,
                             ccl::light_fetch(&tree, ccl::tri_shader_tag{}, light.prim),
                             std::bit_cast<unsigned int>(uv.x),
                             std::bit_cast<unsigned int>(uv.y)};
    }
#endif
  }
#ifdef CYCLES_WEBGPU_DEFER_SHADOW
  *shadow = shadow_ray;
#else
  /* Inline traversal so its HLSL scene handle stays in the entry point. */
  UInt4 shadow_hit, coordinates;
  if (shadow_ray.options.x != 0) {
    [[clang::always_inline]] cycles_bvh2_trace(
      shadow_ray.origin, shadow_ray.direction, shadow_ray.options, shadow_hit, coordinates
#ifdef CYCLES_WEBGPU_WAVEFRONT_LIGHT
      , geometric
#endif
    );
    if (shadow_hit.x != 0) {
      return Shadowed;
    }
  }
#endif

#ifndef CYCLES_WEBGPU_LIGHT_TREE
  const ccl::Spectrum strength = read_rgb(strength_offset);
#endif
  float mis_weight = 1.0f;
#ifdef CYCLES_WEBGPU_LIGHT_TREE
  if (light.type != ccl::LIGHT_POINT || (light.shader & ccl::SHADER_USE_MIS)) {
    mis_weight = ccl::power_heuristic(light.pdf, bsdf_pdf);
  }
#elif defined(CYCLES_WEBGPU_WAVEFRONT_LIGHT) && !defined(CYCLES_WEBGPU_POINT_DISTRIBUTION)
  if (area_light) {
    mis_weight = ccl::power_heuristic(light.pdf, bsdf_pdf);
  }
#endif
  const ccl::Spectrum rgb = bsdf * strength * (light.eval_fac / light.pdf * mis_weight);
  result = {rgb.x, rgb.y, rgb.z, 1};
  return Lit;
#endif
}

Float4 background_radiance()
{
  const auto background = read_rgb(8);
  return {background.x, background.y, background.z, 1};
}

#if defined(CYCLES_WEBGPU_WAVEFRONT_LIGHT) && defined(CYCLES_WEBGPU_LIGHT_TREE)
[[gnu::always_inline]] float forward_tree_selection_pdf(
    ccl::KernelGlobals tree,
    const ccl::float3 P,
    const ccl::float3 N,
    const unsigned int path_flag,
    const unsigned int emitter,
    [[maybe_unused]] const float volume_segment_length)
{
#  ifdef CYCLES_WEBGPU_VOLUME_FORWARD
  if (volume_segment_length > 0.0f) {
    /* At a volume bounce, Cycles stores D * scatter_distance in the normal
     * slot so that the NEE selection PDF uses the original ray segment. */
    return ccl::light_tree_pdf<true>(tree,
                                    P - N,
                                    ccl::safe_normalize(N),
                                    volume_segment_length,
                                    path_flag,
                                    OBJECT_NONE,
                                    emitter,
                                    OBJECT_NONE);
  }
#  endif
  return ccl::webgpu_light_tree_pdf(tree, P, N, path_flag, emitter);
}

[[gnu::always_inline]] Float4 accumulate_forward_lamps(
    ccl::KernelGlobals tree,
    const unsigned int descriptor,
    const UInt4 origin,
    const ccl::float3 direction,
    const float mesh_distance,
    const ccl::PathRayVisibility visibility,
    const UInt4 path,
    const ccl::float3 previous_N,
    const float volume_segment_length,
    const ccl::float3 throughput,
    const unsigned int transparent_bounce,
    const float clamp_limit)
{
  const unsigned int first = cycles_light_word(descriptor + CYCLES_WEBGPU_TREE_FIRST_FORWARD_LAMP);
  const unsigned int end = cycles_light_word(descriptor + CYCLES_WEBGPU_TREE_LAMP_COUNT);
  const unsigned int limit = cycles_light_word(descriptor + CYCLES_WEBGPU_TREE_TRANSPARENT_MAX_BOUNCE);
  const unsigned int remaining = limit > transparent_bounce ? limit - transparent_bounce : 1;
  const bool primary = (visibility & ccl::PATH_RAY_VISIBILITY_CAMERA) != 0;
  ccl::Ray ray{};
  ray.P = unpack_rgb(origin);
  ray.D = direction;
  ray.tmin = std::bit_cast<float>(origin.w);
  ray.tmax = mesh_distance;

  ccl::Spectrum radiance = ccl::zero_spectrum();
  unsigned int previous = end;
  unsigned int hits = 0;
  /* Use Cycles' nearest-lamp order and transparent-bounce accounting. The
   * mesh hit is already known, so there is no need to traverse it again. */
  while (hits < remaining && ray.tmin < ray.tmax) {
    unsigned int nearest = end;
    float distance = ray.tmax;
    for (unsigned int index = first; index < end; ++index) {
      const unsigned int words = ccl::light_tree_lamp_word(tree, index);
      const unsigned int type = cycles_light_word(words + CYCLES_WEBGPU_TREE_LAMP_TYPE);
      if (index == previous || (type != ccl::LIGHT_AREA && type != ccl::LIGHT_POINT)) {
        continue;
      }
      const int shader = std::bit_cast<int>(cycles_light_word(words + CYCLES_WEBGPU_TREE_LAMP_SHADER));
      if (primary ? (shader & ccl::SHADER_EXCLUDE_CAMERA) : !(shader & ccl::SHADER_USE_MIS)) {
        continue;
      }
      const ccl::KernelLight lamp = ccl::light_fetch(tree, ccl::lights_tag{}, index);
      float t;
      const bool hit = type == ccl::LIGHT_POINT ? ccl::point_light_intersect(&lamp, &ray, &t) :
                                                ccl::area_light_intersect(&lamp, &ray, &t);
      if (hit && t < distance) {
        distance = t;
        nearest = index;
      }
    }
    if (nearest == end) {
      break;
    }
    ++hits;
    previous = nearest;
    ray.tmin = ccl::intersection_t_offset(distance);
    const ccl::KernelLight lamp = ccl::light_fetch(tree, ccl::lights_tag{}, nearest);
    if (!ccl::is_light_shader_visible_to_path(lamp.shader_id, visibility, path.y)) {
      continue;
    }
    const auto eval = lamp.type == ccl::LIGHT_POINT ?
                          ccl::point_light_eval_from_intersection(
                              &lamp, ray.P, ray.D, distance, previous_N, path.y) :
                          ccl::area_light_eval_from_intersection(&lamp, ray.P, ray.D, distance);
    const unsigned int words = ccl::light_tree_lamp_word(tree, nearest);
    float mis = 1.0f;
    if (!primary && !(path.y & ccl::PATH_RAY_MIS_SKIP)) {
      const unsigned int emitter = cycles_light_word(words + CYCLES_WEBGPU_TREE_LAMP_EMITTER);
      const float pdf = eval.pdf * forward_tree_selection_pdf(tree,
                                                              ray.P,
                                                              previous_N,
                                                              path.y,
                                                              emitter,
                                                              volume_segment_length);
      mis = ccl::power_heuristic(std::bit_cast<float>(path.z), pdf);
    }
    auto contribution = ccl::ensure_finite(
        throughput * (read_rgb(words + CYCLES_WEBGPU_TREE_LAMP_STRENGTH) * (eval.eval_fac * mis)));
    const float sum = ccl::reduce_add(ccl::fabs(contribution));
    if (sum > clamp_limit) {
      contribution *= clamp_limit / sum;
    }
    radiance += contribution;
  }
  /* A negative count marks already-weighted, individually clamped emission.
   * The legacy single-area path still outputs its positive clamp limit. */
  return {radiance.x, radiance.y, radiance.z, -float(hits)};
}
#endif

}  // namespace

#ifdef CYCLES_WEBGPU_VOLUME_SCATTER
/* Split stock light_sample_from_volume_segment() from light_sample<false>()
 * so both calls retain the same Light Tree selection and random coordinates. */
[[gnu::always_inline]] void cycles_volume_light_sample(const unsigned int emitter,
                                                        const Float4 position,
                                                        const Float4 direction,
                                                        const Float4 random,
                                                        const unsigned int bounce,
                                                        UInt4 &sampled,
                                                        Float4 &point_pdf,
                                                        Float4 &direction_distance,
                                                        Float4 &strength_selection,
                                                        Float4 &interval)
{
  using namespace ccl;
  sampled = {~0u, 0, 0, 0};
  point_pdf = {};
  direction_distance = {};
  strength_selection = {};
  interval = {};

  const uint descriptor = cycles_light_word(CYCLES_WEBGPU_LIGHT_TREE_OFFSET);
  LightGlobals globals;
  setup_light_tree(globals, descriptor);
  const float3 P = make_float3(position.x, position.y, position.z);
  const float3 D = make_float3(direction.x, direction.y, direction.z);
  const float2 rand = make_float2(random.x, random.y);
  constexpr int flags = SR_BSDF_HAS_TRANSMISSION;
  LightSample light{};
  light.object = OBJECT_NONE;
  Spectrum strength;
  bool sampled_emission = false;
  Interval<float> range{0.0f, position.w};
  if (emitter == ~0u) {
    if (!light_tree_sample<true>(
            &globals, random.z, P, D, position.w, OBJECT_NONE, flags, &light) ||
        !sample_selected_tree_light<true, false>(
            globals, descriptor, rand, bounce, P, D, flags, light, strength, sampled_emission))
    {
      return;
    }
    /* Match volume_valid_direct_ray_segment for the admitted emitter types. */
    if (light.type == LIGHT_AREA) {
      const KernelLight lamp = light_fetch(&globals, lights_tag{}, light.prim);
      if (!area_light_valid_ray_segment(&lamp.area, P - lamp.co, D, &range)) {
        return;
      }
    }
    if (range.is_empty()) {
      return;
    }
  }
  else {
    light.emitter_id = int(emitter);
    light.pdf_selection = direction.w;
    if (!sample_selected_tree_light<false, false>(globals,
                                          descriptor,
                                          rand,
                                          bounce,
                                          P,
                                          zero_float3(),
                                          flags,
                                          light,
                                          strength,
                                          sampled_emission))
    {
      return;
    }
  }
  if ((light.shader & SHADER_EXCLUDE_SCATTER) ||
      !(light.type == LIGHT_AREA || light.type == LIGHT_POINT || light.type == LIGHT_BACKGROUND))
  {
    return;
  }
  sampled = {uint(light.emitter_id), uint(light.shader), uint(light.type), uint(light.prim)};
  point_pdf = {light.P.x, light.P.y, light.P.z, light.pdf};
  direction_distance = {light.D.x, light.D.y, light.D.z, light.t};
  strength *= light.eval_fac;
  strength_selection = {strength.x, strength.y, strength.z, light.pdf_selection};
  interval = {range.min, range.max, 0.0f, 0.0f};
}
#endif

#if defined(CYCLES_WEBGPU_LIGHT_TREE) && !defined(CYCLES_WEBGPU_SHADE_BACKGROUND)
float cycles_triangle_hit_mis(UInt4 hit,
                              UInt4 position,
                              UInt4 geometric,
                              UInt4 ray,
                              UInt4 origin,
                              UInt4 path,
                              UInt4 mis_origin_normal)
{
  if ((path.y & ccl::PATH_RAY_MIS_SKIP) || (ray.w & ccl::PATH_RAY_VISIBILITY_CAMERA)) {
    return 1.0f;
  }

  const unsigned int descriptor = cycles_light_word(CYCLES_WEBGPU_LIGHT_TREE_OFFSET);
  if (hit.z >= cycles_light_word(descriptor + CYCLES_WEBGPU_TREE_OBJECT_COUNT) ||
      hit.y >= cycles_light_word(descriptor + CYCLES_WEBGPU_TREE_TRIANGLE_INDEX_COUNT))
  {
    return 1.0f;
  }

  ccl::LightGlobals tree;
  ccl::setup_light_tree(tree, descriptor);
  if (ccl::light_tree_triangle_index(&tree, hit.y) == CYCLES_WEBGPU_TREE_NO_LAMP ||
      ccl::light_fetch(&tree, ccl::light_to_tree_tag{}, hit.z) == CYCLES_WEBGPU_TREE_NO_LAMP)
  {
    return 1.0f;
  }

  const auto shader = ccl::light_fetch(&tree, ccl::tri_shader_tag{}, hit.y) & ccl::SHADER_MASK;
  const int shader_flags = ccl::light_fetch(&tree, ccl::shaders_tag{}, shader).flags;
  if (!(shader_flags & (position.w == 2 ? ccl::SD_MIS_BACK : ccl::SD_MIS_FRONT))) {
    return 1.0f;
  }

  const auto first_primitive = ccl::light_fetch(&tree, ccl::object_prim_offset_tag{}, hit.z);
  const auto first_triangle = ccl::light_fetch(&tree, ccl::object_lookup_offset_tag{}, hit.z);
  const auto lookup = hit.y - first_primitive + first_triangle;
  if (hit.y < first_primitive || lookup < first_triangle ||
      lookup >= cycles_light_word(descriptor + CYCLES_WEBGPU_TREE_TRIANGLE_TO_TREE_COUNT))
  {
    return 1.0f;
  }
  const auto emitter = ccl::light_fetch(&tree, ccl::triangle_to_tree_tag{}, lookup);
  if (emitter == CYCLES_WEBGPU_TREE_NO_LAMP) {
    return 1.0f;
  }

  /* Match light_sample_mis_weight_forward_surface(), using the already
   * available hit, previous surface normal and BSDF PDF. */
  ccl::ShaderData sd{};
  sd.P = unpack_rgb(position);
  sd.Ng = unpack_rgb(geometric);
  sd.wi = -unpack_rgb(ray);
  sd.object = std::bit_cast<int>(hit.z);
  sd.prim = std::bit_cast<int>(hit.y);
  sd.time = -1.0f;
  const float t = std::bit_cast<float>(hit.w);
  const float pdf = ccl::triangle_light_pdf(&tree, &sd, t) *
                    ccl::light_tree_pdf<false>(&tree,
                                               unpack_rgb(origin),
                                               unpack_rgb(mis_origin_normal),
                                               0.0f,
                                               path.y,
                                               sd.object,
                                               emitter,
                                               OBJECT_NONE);
  return ccl::power_heuristic(std::bit_cast<float>(path.z), pdf);
}
#endif

#ifdef CYCLES_WEBGPU_WAVEFRONT_LIGHT
[[gnu::always_inline]] void cycles_wavefront_light(UInt4 position,
                            UInt4 normal,
                            UInt4 spec_normal,
                            UInt4 geometric,
                            UInt4 shadow_along,
                            UInt4 shadow_opposite,
                            UInt4 hit,
                            UInt4 ray,
                            UInt4 origin,
                            UInt4 path,
                            UInt4 sampling,
                            unsigned int subsurface_steps,
                            Float4 material,
                            Float4 specular,
                            Float4 parameters,
                            Float4 glass,
                            Float4 hair,
                            Float4 &result,
                            Float4 &area_result,
                            unsigned int &hair_lcg
#ifdef CYCLES_WEBGPU_LIGHT_TREE
                            , UInt4 mis_origin_normal, Float4 throughput
#endif
#ifdef CYCLES_WEBGPU_DEFER_SHADOW
                            , UInt4 &shadow_origin,
                            UInt4 &shadow_direction,
                            UInt4 &shadow_options
#  ifdef CYCLES_WEBGPU_LIGHT_TREE
                            , UInt4 &sampled_emission
#  endif
#endif
                            )
{
#ifdef CYCLES_WEBGPU_DEFER_SHADOW
  ShadowRay shadow{};
  shadow_origin = {};
  shadow_direction = {};
  shadow_options = {};
#  ifdef CYCLES_WEBGPU_LIGHT_TREE
  sampled_emission = {};
#  endif
#endif
  area_result = {};
  hair_lcg = 0;
  if (material.w == CYCLES_WEBGPU_MATERIAL_HAIR_HUANG) {
    hair_lcg = ccl::lcg_state_init(
        sampling.x, (path.x + subsurface_steps + 1) * ccl::PRNG_BOUNCE_NUM,
        sampling.y, 0xb4bc3953u);
  }
  const bool primary = (ray.w & ccl::PATH_RAY_VISIBILITY_CAMERA) != 0;
  const float indirect_clamp = read_float(3);
  /* Cycles uses bounce - 1 for forward emission and bounce for NEE. */
  const float forward_clamp = path.x <= 1 ? FLT_MAX : indirect_clamp;
  const float nee_clamp = path.x == 0 ? FLT_MAX : indirect_clamp;
#ifdef CYCLES_WEBGPU_POINT_DISTRIBUTION
  const unsigned int descriptor = cycles_light_word(CYCLES_WEBGPU_POINT_LIGHTS_OFFSET);
  if (position.w == 0) {
    result = primary ? background_radiance() : Float4{0, 0, 0, 1};
    result.w = -forward_clamp;
    return;
  }
#else
#  ifdef CYCLES_WEBGPU_LIGHT_TREE
  const unsigned int descriptor = cycles_light_word(CYCLES_WEBGPU_LIGHT_TREE_OFFSET);
  const auto previous_P = unpack_rgb(origin);
  const auto previous_N = unpack_rgb(mis_origin_normal);
  const float volume_segment_length = std::bit_cast<float>(mis_origin_normal.w);
  ccl::LightGlobals tree;
  ccl::setup_light_tree(tree, descriptor);
#  else
  const unsigned int descriptor = cycles_light_word(CYCLES_WEBGPU_LIGHT_EXTENDED);
  const bool has_area = descriptor != 0;
#  endif
  const auto direction = unpack_rgb(ray);
  const auto visibility = static_cast<ccl::PathRayVisibility>(ray.w);
#  ifdef CYCLES_WEBGPU_LIGHT_TREE
  if (!(path.y & ccl::PATH_RAY_SUBSURFACE) &&
      cycles_light_word(descriptor + CYCLES_WEBGPU_TREE_FIRST_FORWARD_LAMP) !=
      CYCLES_WEBGPU_TREE_NO_LAMP)
  {
    area_result = accumulate_forward_lamps(&tree,
                                          descriptor,
                                          origin,
                                          direction,
                                          std::bit_cast<float>(hit.w),
                                          visibility,
                                          path,
                                          previous_N,
                                          volume_segment_length,
                                          ccl::make_float3(throughput.x, throughput.y, throughput.z),
                                          sampling.w,
                                          forward_clamp);
  }
#  else
  if (has_area && !(path.y & ccl::PATH_RAY_SUBSURFACE)) {
    const int lamp_shader = std::bit_cast<int>(
        cycles_light_word(descriptor + CYCLES_WEBGPU_AREA_SHADER));
    if ((primary && !(lamp_shader & ccl::SHADER_EXCLUDE_CAMERA)) ||
        (!primary && (lamp_shader & ccl::SHADER_USE_MIS)))
    {
      ccl::KernelLight lamp{};
      unpack_area_light(descriptor, lamp);
      ccl::Ray forward{};
      forward.P = unpack_rgb(origin);
      forward.D = direction;
      forward.tmin = std::bit_cast<float>(origin.w);
      forward.tmax = std::bit_cast<float>(hit.w);
      float t;
      const bool intersects = ccl::area_light_intersect(&lamp, &forward, &t);
      if (intersects && t < forward.tmax) {
        area_result = {0, 0, 0, forward_clamp};
        if (ccl::is_light_shader_visible_to_path(lamp_shader, visibility, path.y)) {
          const auto eval = ccl::area_light_eval_from_intersection(&lamp, forward.P, forward.D, t);
          const float bsdf_pdf = std::bit_cast<float>(path.z);
          float mis = 1.0f;
          if (!(path.y & ccl::PATH_RAY_MIS_SKIP) && !primary) {
            mis = ccl::power_heuristic(bsdf_pdf, eval.pdf);
          }
          const auto rgb = read_rgb(4) * (eval.eval_fac * mis);
          area_result = {rgb.x, rgb.y, rgb.z, forward_clamp};
        }
        /* Cycles continues the same ray after a forward light hit. The closest
         * mesh intersection behind the emitter is already known. */
      }
    }
  }
#  endif
  if (position.w == 0) {
#  ifdef CYCLES_WEBGPU_SHADE_BACKGROUND
    /* Film Transparent hides camera misses without changing world lighting
     * on surfaces. Match the stock background kernel's early skip. */
    if ((path.y & ccl::PATH_RAY_TRANSPARENT_BACKGROUND) && cycles_film_transparent()) {
      result.w = -forward_clamp;
      return;
    }
#  endif
#  ifdef CYCLES_WEBGPU_LIGHT_TREE
    const int world_shader = std::bit_cast<int>(
        cycles_light_word(descriptor + CYCLES_WEBGPU_TREE_WORLD_SHADER));
    ccl::Spectrum rgb = ccl::zero_spectrum();
    if (ccl::is_light_shader_visible_to_path(world_shader, visibility, path.y)) {
      rgb = tree_environment(descriptor, direction);
      const unsigned int emitter = cycles_light_word(descriptor + CYCLES_WEBGPU_TREE_WORLD_EMITTER);
      if (emitter != CYCLES_WEBGPU_TREE_NO_LAMP && !(path.y & ccl::PATH_RAY_MIS_SKIP) && !primary) {
        setup_world_sampling(tree, descriptor);
        const float pdf = ccl::background_map_pdf(&tree, direction) *
                          forward_tree_selection_pdf(&tree, previous_P, previous_N, path.y,
                                                     emitter, volume_segment_length);
        rgb *= ccl::power_heuristic(std::bit_cast<float>(path.z), pdf);
      }
    }
    result = {rgb.x, rgb.y, rgb.z, -forward_clamp};
#  else
    if (has_area) {
      const int world_shader = std::bit_cast<int>(
          cycles_light_word(descriptor + CYCLES_WEBGPU_WORLD_SHADER));
      const unsigned int width = cycles_light_word(descriptor + CYCLES_WEBGPU_WORLD_WIDTH);
      auto rgb = ccl::zero_spectrum();
      if (ccl::is_light_shader_visible_to_path(world_shader, visibility, path.y)) {
        rgb = width ? environment_radiance(width,
                                           cycles_light_word(descriptor + CYCLES_WEBGPU_WORLD_HEIGHT),
                                           cycles_light_word(descriptor + CYCLES_WEBGPU_WORLD_INTERPOLATION),
                                           descriptor + CYCLES_WEBGPU_EXTENDED_LIGHT_WORDS,
                                           direction) :
                      read_rgb(8);
      }
      result = {rgb.x, rgb.y, rgb.z, -forward_clamp};
    }
    else {
      /* The legacy point-light subset allows world radiance on camera rays only. */
      result = primary ? background_radiance() : Float4{0, 0, 0, 1};
      result.w = -forward_clamp;
    }
#  endif
    return;
  }
#endif
  if ((material.w == 1.0f || material.w == CYCLES_WEBGPU_MATERIAL_MIXED_TRANSPARENT ||
       is_ggx(material) || CYCLES_WEBGPU_MATERIAL_IS_STANDALONE(material.w)) &&
      std::bit_cast<float>(sampling.z) > 0.0f &&
      !(path.y & ccl::PATH_RAY_TERMINATE_AFTER_TRANSPARENT))
  {
    direct_surface_light(position,
                         normal,
                         hit,
                         material,
                         result,
                         ray,
                         specular,
                         parameters,
                         glass,
                         spec_normal,
                         geometric,
                         shadow_along,
                         shadow_opposite,
                         sampling,
                         path,
                         descriptor,
                         subsurface_steps,
#ifdef CYCLES_WEBGPU_DEFER_SHADOW
                         &shadow,
#else
                         nullptr,
#endif
                         hair,
                         &hair_lcg);
#ifdef CYCLES_WEBGPU_DEFER_SHADOW
    shadow_origin = shadow.origin;
    shadow_direction = shadow.direction;
    shadow_options = shadow.options;
#  ifdef CYCLES_WEBGPU_LIGHT_TREE
    sampled_emission = shadow.emission;
#  endif
#endif
  }
  else {
    result = {0, 0, 0, 0};
    return;
  }
  /* The path update applies the limit after multiplying by path throughput. */
  result.w = nee_clamp;
}
#else
unsigned int cycles_direct_light(UInt4 position, UInt4 normal, UInt4 hit, Float4 &result)
{
  const unsigned int material = normal.w & ccl::SHADER_MASK;
  if (position.w == 0) {
    result = background_radiance();
    return Background;
  }
  if (position.w == 3 || material >= cycles_light_word(3)) {
    result = {0, 0, 0, 0};
    return Unsupported;
  }
  const auto albedo = read_rgb(12 + 4 * material);
  return direct_surface_light(
      position, normal, hit, Float4{albedo.x, albedo.y, albedo.z, 1.0f}, result);
}
#endif
