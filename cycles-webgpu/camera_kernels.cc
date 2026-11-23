/* SPDX-License-Identifier: Apache-2.0 */

#include "camera_kernels.hh"
#include "../blender-platform/cycles-webgpu/material_layout.h"
#include "../blender-platform/cycles-webgpu/path_state_layout.h"

#include <bit>

#ifdef __aarch64__
#  include <arm_neon.h>
#endif

#define CCL_NAMESPACE_BEGIN namespace ccl {
#define CCL_NAMESPACE_END }
#include "kernel/types.h"

/* Install the same kernel-global interface as the existing Cycles devices.
 * This entry point reads only the camera data and disables motion. */
#define __KERNEL_GPU__
#include "kernel/globals.h"
#undef __KERNEL_GPU__

namespace ccl {
struct CameraGlobals {
  KernelData data{};
  const DecomposedTransform *camera_motion = nullptr;
  const float *lookup_table = nullptr;
};
using KernelGlobals = const CameraGlobals *;
}

#define kernel_data (kg->data)
#define kernel_data_array(name) (kg->name)
#ifdef CYCLES_PRIMARY_RAY_QUEUE
/* Matches the sampling buffer packed by WebGPUPreview. */
__attribute__((pure)) unsigned int cycles_camera_sample_word(unsigned int index);
static constexpr unsigned int sampling_header_words = 7;
static float camera_lookup_table(unsigned int i)
{
  return std::bit_cast<float>(cycles_camera_sample_word(sampling_header_words + i));
}
static float camera_sample_pattern_lut(unsigned int i)
{
  return camera_lookup_table(FILTER_TABLE_SIZE + i);
}
#  define kernel_data_fetch(name, i) camera_##name(i)
#else
#  define kernel_data_fetch(name, i) (kg->name[i])
#endif
#undef ccl_device
#define ccl_device static inline __attribute__((always_inline))
#include "kernel/camera/camera.h"
#ifdef CYCLES_PRIMARY_RAY_QUEUE
#  include "kernel/sample/pattern.h"
#  ifdef CYCLES_WEBGPU_CAMERA_SAMPLE
#    define __KERNEL_GPU__
#    include "kernel/util/profiler.h"
#    undef __KERNEL_GPU__
#    include "kernel/integrator/init_from_camera.h"
#  endif

#  ifdef CYCLES_WEBGPU_ADAPTIVE_CAMERA_RAYS
unsigned int cycles_adaptive_sample_class(const unsigned int sample)
{
  return ccl::sample_is_class_A(cycles_camera_sample_word(0), int(sample));
}
#  endif

static void load_sampling(ccl::KernelIntegrator &integrator,
                          unsigned int pattern,
                          float scrambling_distance)
{
  integrator.sampling_pattern = pattern;
  integrator.seed = cycles_camera_sample_word(1);
  integrator.sobol_index_mask = cycles_camera_sample_word(2);
  integrator.blue_noise_sequence_length = cycles_camera_sample_word(3);
  integrator.tabulated_sobol_sequence_size = cycles_camera_sample_word(4);
  integrator.scrambling_distance = scrambling_distance;
  integrator.pixel_jitter = ccl::make_float2(
      std::bit_cast<float>(cycles_camera_sample_word(5)),
      std::bit_cast<float>(cycles_camera_sample_word(6)));
}
#endif

#if defined(CYCLES_WEBGPU_PATH_UPDATE) || defined(CYCLES_WEBGPU_SHADE_BACKGROUND) || \
    defined(CYCLES_WEBGPU_DENOISE_SURFACE) || defined(CYCLES_WEBGPU_FILM_SURFACE) || \
    defined(CYCLES_WEBGPU_FILM_AO)
static unsigned int path_settings_offset()
{
  unsigned int offset = sampling_header_words + FILTER_TABLE_SIZE;
  if (cycles_camera_sample_word(0) == ccl::SAMPLING_PATTERN_TABULATED_SOBOL) {
    offset += cycles_camera_sample_word(4) * NUM_TAB_SOBOL_PATTERNS * NUM_TAB_SOBOL_DIMENSIONS;
  }
  return offset;
}

unsigned int cycles_path_transparent_max_bounce()
{
  return cycles_camera_sample_word(path_settings_offset() + 8) &
         CYCLES_WEBGPU_TRANSPARENT_BOUNCE_MASK;
}

unsigned int cycles_film_transparent()
{
  return cycles_camera_sample_word(path_settings_offset() + 9);
}

float cycles_path_direct_clamp()
{
  return std::bit_cast<float>(cycles_camera_sample_word(path_settings_offset() + 10));
}

#endif

#if defined(CYCLES_WEBGPU_PATH_UPDATE) || defined(CYCLES_WEBGPU_DENOISE_SURFACE) || \
    defined(CYCLES_WEBGPU_FILM_SURFACE)
#  include "../blender-platform/cycles-webgpu/material_layout.h"

float cycles_ggx_lookup(const unsigned int index)
{
  return std::bit_cast<float>(
      cycles_camera_sample_word(path_settings_offset() + CYCLES_WEBGPU_PATH_SETTINGS_WORDS + index));
}

#endif

#ifdef CYCLES_WEBGPU_DENOISE_SURFACE
#  include "ggx_closures.h"

float cycles_denoise_blur_alpha(const float min_ray_pdf)
{
  return cycles_glossy_blur(std::bit_cast<float>(cycles_camera_sample_word(path_settings_offset() + 7)),
                            min_ray_pdf);
}
#endif

#ifdef CYCLES_WEBGPU_PATH_UPDATE
#  include "../blender-platform/cycles-webgpu/material_layout.h"
#  include "ggx_closures.h"

#  include "kernel/sample/mapping.h"
#  include "kernel/closure/bsdf_diffuse.h"
#  include "kernel/closure/bsdf_transparent.h"
#  include "cosine_hemisphere.h"
#  include "kernel/integrator/path_state.h"
#  include "transparent_ray_util.h"

static_assert(CYCLES_WEBGPU_PATH_MIS_SKIP == ccl::PATH_RAY_MIS_SKIP);
static_assert(CYCLES_WEBGPU_PATH_MIS_HAD_TRANSMISSION == ccl::PATH_RAY_MIS_HAD_TRANSMISSION);
static_assert(CYCLES_WEBGPU_PATH_TRANSPARENT_BACKGROUND == ccl::PATH_RAY_TRANSPARENT_BACKGROUND);
static_assert(CYCLES_WEBGPU_PATH_TERMINATE_AFTER_TRANSPARENT ==
              ccl::PATH_RAY_TERMINATE_AFTER_TRANSPARENT);
static_assert(CYCLES_WEBGPU_PATH_TERMINATE_ON_NEXT_SURFACE ==
              ccl::PATH_RAY_TERMINATE_ON_NEXT_SURFACE);

UInt4 cycles_path_settings()
{
  const unsigned int offset = path_settings_offset();
  return {cycles_camera_sample_word(offset),
          cycles_camera_sample_word(offset + 1),
          cycles_camera_sample_word(offset + 2),
          cycles_camera_sample_word(offset + 3)};
}

UInt4 cycles_path_lobe_settings()
{
  const unsigned int offset = path_settings_offset();
  return {cycles_camera_sample_word(offset + 4),
          cycles_camera_sample_word(offset + 5),
          cycles_camera_sample_word(offset + 6),
          cycles_camera_sample_word(offset + 7)};
}

[[gnu::always_inline]] UInt4 cycles_transparent_advance(UInt4 rng,
                                                        UInt4 path,
                                                        Float4 throughput,
                                                        unsigned int portal)
{
  using namespace ccl;
  CameraGlobals globals;
  const uint settings = cycles_camera_sample_word(path_settings_offset() + 8);
  auto &integrator = globals.data.integrator;
  load_sampling(integrator,
                cycles_camera_sample_word(0),
                std::bit_cast<float>(cycles_path_settings().z));
  integrator.transparent_max_bounce = settings & CYCLES_WEBGPU_TRANSPARENT_BOUNCE_MASK;
  integrator.transparent_min_bounce = settings >> 16;

  IntegratorStateCPU state{};
  state.path.rng_pixel = rng.x;
  state.path.sample = int(rng.y);
  state.path.rng_offset = rng.z * PRNG_BOUNCE_NUM;
  state.path.flag = rng.w;
  state.path.visibility = path.x;
  state.path.transparent_bounce = path.y & CYCLES_WEBGPU_TRANSPARENT_BOUNCE_MASK;
  state.path.portal_bounce = path.y >> CYCLES_WEBGPU_PORTAL_BOUNCE_SHIFT;
  state.path.throughput = PackedSpectrum(make_float3(throughput.x, throughput.y, throughput.z));

  const int label = LABEL_TRANSMIT | (portal ? LABEL_RAY_PORTAL : LABEL_TRANSPARENT);
  path_state_next(&globals, &state, label, portal ? SR_RAY_PORTAL : SR_TRANSPARENT);
  const float p = path_state_continuation_probability(&globals, &state, state.path.flag);
  float next_rr = 0.0f;
  if (!(state.path.flag & PATH_RAY_TERMINATE_ON_NEXT_SURFACE) &&
      (p == 1.0f ||
       (p > 0.0f &&
        path_rng_1D(&globals, rng.x, state.path.sample, state.path.rng_offset + PRNG_TERMINATE) < p)))
  {
    next_rr = 1.0f / p;
  }
  return {std::bit_cast<uint>(intersection_t_offset(std::bit_cast<float>(path.z))),
          state.path.flag,
          uint(state.path.transparent_bounce) |
              (uint(state.path.portal_bounce) << CYCLES_WEBGPU_PORTAL_BOUNCE_SHIFT),
          std::bit_cast<uint>(next_rr)};
}

static bool sample_mixed_transparent(ccl::float3 &random,
                                     const ccl::ClosureType opaque_type,
                                     const float opaque_weight,
                                     const float transparent_weight,
                                     const bool transparent_first)
{
  using namespace ccl;
  ShaderData sd{};
  sd.num_closure = 2;
  const int first = transparent_first ? 0 : 1;
  sd.closure[first].type = CLOSURE_BSDF_TRANSPARENT_ID;
  sd.closure[first].sample_weight = transparent_weight;
  sd.closure[1 - first].type = opaque_type;
  sd.closure[1 - first].sample_weight = opaque_weight;
  return surface_shader_bsdf_bssrdf_pick(&sd, &random)->type == CLOSURE_BSDF_TRANSPARENT_ID;
}

[[gnu::always_inline]] Float4 cycles_sample_mixed_transparency(UInt4 rng,
                                                             Float4 diffuse,
                                                             Float4 transparent,
                                                             Float4 weights,
                                                             Float4 normal,
                                                             Float4 ray)
{
  using namespace ccl;
  if (!(transparent.w > 0.0f)) {
    return {};
  }
  if (!(weights.x > 0.0f) || (rng.w & PATH_RAY_TERMINATE_AFTER_TRANSPARENT)) {
    return {transparent.x, transparent.y, transparent.z, 1.0f};
  }

  CameraGlobals globals;
  load_sampling(globals.data.integrator,
                cycles_camera_sample_word(0),
                std::bit_cast<float>(cycles_path_settings().z));
  float3 random = path_rng_3D(
      &globals, rng.x, int(rng.y), rng.z * PRNG_BOUNCE_NUM + PRNG_SURFACE_BSDF);
  if (!sample_mixed_transparent(
          random, CLOSURE_BSDF_DIFFUSE_ID, weights.x, transparent.w, weights.y != 0.0f)) {
    return {};
  }

  const float3 wi = -make_float3(ray.x, ray.y, ray.z);
  Spectrum value;
  float3 wo;
  float pdf;
  bsdf_transparent_sample(nullptr, zero_float3(), wi, &value, &wo, &pdf);
  DiffuseBsdf lambert{};
  lambert.N = make_float3(normal.x, normal.y, normal.z);
  float diffuse_pdf;
  const Spectrum diffuse_eval = bsdf_diffuse_eval(
      reinterpret_cast<const ShaderClosure *>(&lambert), wi, wo, &diffuse_pdf);
  const float mixed_pdf = (pdf * transparent.w + diffuse_pdf * weights.x) /
                          (transparent.w + weights.x);
  const Spectrum weight = (value * make_float3(transparent.x, transparent.y, transparent.z) +
                           diffuse_eval * make_float3(diffuse.x, diffuse.y, diffuse.z)) /
                          mixed_pdf;
  return {weight.x, weight.y, weight.z, 1.0f};
}

using UInt2 = unsigned int __attribute__((ext_vector_type(2)));

[[gnu::always_inline]] UInt2 cycles_absorption_bounce(UInt4 rng, UInt4 path, Float4 throughput)
{
  using namespace ccl;
  CameraGlobals globals;
  const UInt4 settings = cycles_path_settings();
  load_sampling(globals.data.integrator, cycles_camera_sample_word(0),
                std::bit_cast<float>(settings.z));
  globals.data.integrator.min_bounce = int(settings.y);

  IntegratorStateCPU state{};
  state.path.bounce = path.x;
  state.path.throughput = PackedSpectrum(make_float3(throughput.x, throughput.y, throughput.z));
  const float probability = path_state_continuation_probability(&globals, &state, rng.w);
  const bool accepted = probability == 1.0f ||
                        (probability > 0.0f && path_rng_1D(&globals, rng.x, int(rng.y),
                          rng.z * PRNG_BOUNCE_NUM + PRNG_TERMINATE) < probability);
  return {std::bit_cast<uint>(intersection_t_offset(std::bit_cast<float>(path.y))),
          std::bit_cast<uint>(accepted ? 1.0f / probability : 0.0f)};
}

[[gnu::always_inline]] Float4 cycles_advance_differential(Float4 ray,
                                                        float filter,
                                                        unsigned int portal)
{
  const float dP = ccl::differential_transfer_compact(ray.x, ccl::zero_float3(), ray.y, ray.z);
  /* Match bsdf_widen_dD(). A portal carries the incoming angular differential. */
  const float scale = filter == FLT_MAX ? 0.0f : ccl::min(1.0f, 1.0f / filter);
  const float dD = portal ? ray.y : ccl::max(ray.y, scale * sqrtf(ray.w));
  return {dP, dD, 0.0f, 0.0f};
}

[[gnu::always_inline]]
unsigned int cycles_surface_bounce(unsigned int rng_pixel,
                                   unsigned int sample_index,
                                   UInt4 position,
                                   UInt4 normal,
                                   UInt4 geometric_normal,
                                   UInt4 specular_normal,
                                   UInt4 ray,
                                   UInt4 settings,
                                   UInt4 lobe_settings,
                                   UInt4 path_counts,
                                   unsigned int bounce,
                                   unsigned int subsurface_steps,
                                   unsigned int ray_differentials,
                                   float min_ray_pdf,
                                   Float4 material,
                                   Float4 specular,
                                   Float4 parameters,
                                   Float4 glass,
                                   Float4 hair,
                                   Float4 bump,
                                   Float4 throughput,
                                   Float4 &next_throughput,
                                   float &next_pdf,
                                   UInt4 &origin,
                                   UInt4 &direction,
                                   UInt4 &next_counts,
                                   float &surface_rr,
                                   float &roughness_squared,
                                   unsigned int light_passes,
                                   Float4 &diffuse_ratio,
                                   Float4 &glossy_ratio)
{
  ccl::CameraGlobals globals;
  auto &integrator = globals.data.integrator;
  load_sampling(integrator, cycles_camera_sample_word(0), std::bit_cast<float>(settings.z));
  integrator.max_bounce = settings.x;
  integrator.min_bounce = settings.y;
  integrator.max_diffuse_bounce = lobe_settings.x;
  integrator.max_glossy_bounce = lobe_settings.y;
  integrator.max_transmission_bounce = lobe_settings.z;

  const int sample = int(sample_index);
  const int rng_offset = (int(bounce + subsurface_steps) + 1) * ccl::PRNG_BOUNCE_NUM;
  auto random = ccl::path_rng_3D(
      &globals, rng_pixel, sample, rng_offset + ccl::PRNG_SURFACE_BSDF);
  surface_rr = 0.0f;
  roughness_squared = 1.0f;
  diffuse_ratio = glossy_ratio = {};
  next_throughput = throughput;
  next_pdf = 0.0f;
  origin = {};
  direction = {};
  next_counts = path_counts;

  const auto N = bump.w >= CYCLES_WEBGPU_NORMAL_BOTH ?
                     ccl::make_float3(hair.x, hair.y, hair.z) :
                 bump.w >= CYCLES_WEBGPU_NORMAL_SHADER ?
                     ccl::make_float3(bump.x, bump.y, bump.z) :
                     ccl::make_float3(std::bit_cast<float>(normal.x),
                                      std::bit_cast<float>(normal.y),
                                      std::bit_cast<float>(normal.z));
  const auto Ng = ccl::make_float3(std::bit_cast<float>(geometric_normal.x),
                                  std::bit_cast<float>(geometric_normal.y),
                                  std::bit_cast<float>(geometric_normal.z));
  const Float4 wi{-std::bit_cast<float>(ray.x),
                  -std::bit_cast<float>(ray.y),
                  -std::bit_cast<float>(ray.z),
                  0};
  const bool mixed_glossy = material.w == CYCLES_WEBGPU_MATERIAL_DIFFUSE_GLOSSY;
  const unsigned int mixed_flags = mixed_glossy ? static_cast<unsigned int>(parameters.z) : 0u;
  const Float4 glossy_bump = cycles_mixed_glossy_bump(bump, hair, mixed_flags);
  const Float4 corrected = cycles_bump_specular_normal(
      glossy_bump, Float4{Ng.x, Ng.y, Ng.z, 0}, wi, std::bit_cast<Float4>(specular_normal));
  const auto Ns = ccl::make_float3(corrected.x, corrected.y, corrected.z);
  ccl::float3 D;
  float pdf;
  int label;
  const bool standalone = CYCLES_WEBGPU_MATERIAL_IS_STANDALONE(material.w);
  const bool ordered = CYCLES_WEBGPU_MATERIAL_HAS_ORDERED_CLOSURES(material.w);
  const bool ggx = material.w == CYCLES_WEBGPU_MATERIAL_GGX ||
                   material.w == CYCLES_WEBGPU_MATERIAL_MULTISCATTER_GGX;
  const bool mixed_diffuse_translucent = material.w == CYCLES_WEBGPU_MATERIAL_DIFFUSE_TRANSLUCENT;
  const bool translucent =
      (ggx && CYCLES_WEBGPU_GGX_HAS_TRANSLUCENT(glass)) ||
      (standalone && (material.w == CYCLES_WEBGPU_MATERIAL_TRANSLUCENT ||
                      (mixed_diffuse_translucent && parameters.y > 0.0f) ||
                      (ordered && (unsigned(hair.w) & CYCLES_WEBGPU_ORDERED_HAS_TRANSLUCENT))));
  if (standalone) {
    if (ordered && (path_counts.w & ccl::PATH_RAY_TERMINATE_AFTER_TRANSPARENT)) {
      if (!(unsigned(hair.w) & CYCLES_WEBGPU_ORDERED_HAS_TRANSPARENT)) {
        return 0;
      }
      next_throughput *= Float4{hair.x, hair.y, hair.z, 0.0f};
      return CYCLES_WEBGPU_MIXED_TRANSPARENT_PENDING;
    }
    float glossy_probability = 1.0f;
    if (material.w == CYCLES_WEBGPU_MATERIAL_GLOSSY_TRANSPARENT) {
      const bool only_transparent =
          (path_counts.w & ccl::PATH_RAY_TERMINATE_AFTER_TRANSPARENT) != 0;
      ccl::Spectrum glossy = ccl::make_float3(material.x, material.y, material.z);
      const float glossy_weight = ccl::closure_sample_weight(0, glossy);
      const float transparent_weight = fabsf(hair.w);
      const Float4 transparent{hair.x, hair.y, hair.z, 0.0f};
      if (only_transparent || !(glossy_weight > 0.0f)) {
        if (!(transparent_weight > 0.0f)) {
          return 0;
        }
        next_throughput *= transparent;
        return CYCLES_WEBGPU_MIXED_TRANSPARENT_PENDING;
      }
      if (transparent_weight > 0.0f) {
        const float total = glossy_weight + transparent_weight;
        if (sample_mixed_transparent(random, ccl::CLOSURE_BSDF_MICROFACET_GGX_ID,
                                     glossy_weight, transparent_weight, hair.w < 0.0f))
        {
          /* The reflected closure has zero density along the unchanged ray. */
          next_throughput *= transparent * (total / transparent_weight);
          return CYCLES_WEBGPU_MIXED_TRANSPARENT_PENDING;
        }
        glossy_probability = glossy_weight / total;
      }
    }
    const auto closure_normal = CYCLES_WEBGPU_MATERIAL_USES_SPECULAR_NORMAL(material.w) ? Ns : N;
    Float4 weight, wo;
    const bool mixed_diffuse = mixed_diffuse_translucent || mixed_glossy || ordered;
    const Float4 extra = mixed_diffuse ? corrected : glass;
    const Float4 diffuse_bump = (mixed_flags & CYCLES_WEBGPU_MIXED_BUMP_GLOSSY_ONLY) ?
                                    Float4{} : bump;
    const Float4 lobe_data = mixed_glossy ? diffuse_bump : hair;
    label = cycles_standalone_sample(material,
                                     specular,
                                     parameters,
                                     extra,
                                     lobe_data,
                                     glass,
                                     Float4{closure_normal.x, closure_normal.y, closure_normal.z, 0},
                                     Float4{Ng.x, Ng.y, Ng.z, 0},
                                     wi,
                                     Float4{random.x, random.y, random.z, 0},
                                     cycles_glossy_blur(std::bit_cast<float>(lobe_settings.w), min_ray_pdf),
                                     weight,
                                     wo,
                                     diffuse_ratio);
    glossy_ratio = Float4{1.0f, 1.0f, 1.0f, 0.0f} - diffuse_ratio;
    if (label == ccl::LABEL_NONE) {
      return 0;
    }
    D = ccl::make_float3(wo.x, wo.y, wo.z);
    if (CYCLES_WEBGPU_MATERIAL_IS_GLOSSY(material.w) || mixed_glossy || ordered)
    {
      roughness_squared = wo.w;
    }
    pdf = weight.w * glossy_probability;
    next_throughput *= weight / glossy_probability;
    if (label & ccl::LABEL_TRANSPARENT) {
      /* The ordered Transparent mixture has only diffuse and singular lobes. */
      glossy_ratio = {};
      return CYCLES_WEBGPU_MIXED_TRANSPARENT_PENDING;
    }
  }
  else if (ggx) {
    Float4 weight, wo;
    label = cycles_ggx_sample(material,
                              specular,
                              parameters,
                              glass,
                              Float4{N.x, N.y, N.z, position.w == 2 ? 1.0f : 0.0f},
                              Float4{Ng.x, Ng.y, Ng.z, 0},
                              Float4{Ns.x, Ns.y, Ns.z, 0},
                              std::bit_cast<Float4>(specular_normal),
                              bump,
                              wi,
                              Float4{random.x, random.y, random.z, 0},
                              hair,
                              cycles_glossy_blur(std::bit_cast<float>(lobe_settings.w), min_ray_pdf),
                              ray_differentials != 0,
                              (path_counts.w & ccl::PATH_RAY_TERMINATE_AFTER_TRANSPARENT) != 0,
                              weight,
                              wo,
                              light_passes != 0u,
                              diffuse_ratio,
                              glossy_ratio);
    if (label == 0) {
      return 0;
    }
    D = ccl::make_float3(wo.x, wo.y, wo.z);
    roughness_squared = wo.w;
    pdf = weight.w;
    next_throughput *= weight;
    if (label & ccl::LABEL_TRANSPARENT) {
      return CYCLES_WEBGPU_MIXED_TRANSPARENT_PENDING;
    }
    if (label == ccl::LABEL_SUBSURFACE_SCATTER) {
      return CYCLES_WEBGPU_PATH_SUBSURFACE_PENDING;
    }
  }
  else {
    const auto diffuse_normal = bump.w == 0.0f ? N : ccl::make_float3(bump.x, bump.y, bump.z);
    ccl::webgpu_sample_cos_hemisphere(
        diffuse_normal, ccl::make_float2(random.x, random.y), &D, &pdf);
    if (!(pdf > 0.0f && ccl::dot(Ng, D) > 0.0f)) {
      return 0;
    }
    const float shadowing = cycles_bump_shadowing(
        Float4{N.x, N.y, N.z, 0}, bump, Float4{D.x, D.y, D.z, 0}, false);
    if (shadowing == 0.0f) {
      return 0;
    }
    next_throughput *= shadowing;
    if (material.w == CYCLES_WEBGPU_MATERIAL_MIXED_TRANSPARENT) {
      /* Transparent has zero directional density in the diffuse direction. */
      const float diffuse_probability = parameters.x / (parameters.x + specular.w);
      pdf *= diffuse_probability;
      next_throughput /= diffuse_probability;
    }
    label = ccl::LABEL_REFLECT | ccl::LABEL_DIFFUSE;
    diffuse_ratio = {1.0f, 1.0f, 1.0f, 0.0f};
  }
  next_pdf = pdf;

  /* Marshal the lane into Cycles' scalar state so the stock path transition,
   * visibility and continuation code also govern the WebGPU wavefront. The
   * terminal ray stays queued to find lights or the background. */
  ccl::IntegratorStateCPU state{};
  state.path.bounce = bounce;
  state.path.diffuse_bounce = path_counts.x;
  state.path.glossy_bounce = path_counts.y;
  state.path.transmission_bounce = path_counts.z;
  state.path.rng_offset = rng_offset;
  state.path.flag = path_counts.w & ~ccl::PATH_RAY_SUBSURFACE;
  state.path.throughput = ccl::PackedSpectrum(
      ccl::make_float3(next_throughput.x, next_throughput.y, next_throughput.z));
  ccl::path_state_next(
      &globals,
      &state,
      label,
      (translucent || CYCLES_WEBGPU_MATERIAL_HAIR_TRANSMITS(material.w) ||
       (!standalone && glass.x > 0.0f)) ? ccl::SR_BSDF_HAS_TRANSMISSION : 0);
  const unsigned int visibility = ccl::path_state_ray_visibility(&state);
  next_counts = {state.path.diffuse_bounce,
                 state.path.glossy_bounce,
                 state.path.transmission_bounce,
                 state.path.flag};

  /* Intersect closest only tests roulette for a surface hit. Preserve the
   * unscaled throughput in case this ray reaches a light or the background. */
  const float probability = ccl::path_state_continuation_probability(&globals, &state, state.path.flag);
  if (probability == 1.0f ||
      (probability > 0.0f &&
       ccl::path_rng_1D(&globals, rng_pixel, sample, state.path.rng_offset + ccl::PRNG_TERMINATE) <
           probability))
  {
    surface_rr = 1.0f / probability;
  }

  D = ccl::normalize(D);
  /* Closest intersection has the triangle data needed to test this origin. */
  origin = {position.x, position.y, position.z, 0};
  direction = {std::bit_cast<unsigned int>(D.x),
               std::bit_cast<unsigned int>(D.y),
               std::bit_cast<unsigned int>(D.z),
               std::bit_cast<unsigned int>(FLT_MAX)};
  return visibility;
}
#endif

static ccl::float4 camera_row(Float4 row)
{
  return ccl::make_float4(row.x, row.y, row.z, row.w);
}

/* Keep the ray outputs as shader-local values when linking the C++ and HLSL stages. */
[[gnu::always_inline]]
void cycles_camera_ray(UInt4 pixel,
                       Float4 projection_x,
                       Float4 projection_y,
                       Float4 projection_z,
                       Float4 projection_w,
                       Float4 world_x,
                       Float4 world_y,
                       Float4 world_z,
                       Float4 clip,
                       Float4 lens,
                       Float4 aperture,
                       Float4 dx,
                       Float4 dy,
                       Float4 &origin,
                       Float4 &direction,
                       unsigned int &rng_pixel,
                       float &dP,
                       float &dD)
{
  ccl::CameraGlobals globals;
  ccl::KernelCamera &camera = globals.data.cam;
  camera.rastertocamera.x = camera_row(projection_x);
  camera.rastertocamera.y = camera_row(projection_y);
  camera.rastertocamera.z = camera_row(projection_z);
  camera.rastertocamera.w = camera_row(projection_w);
  camera.cameratoworld = {camera_row(world_x), camera_row(world_y), camera_row(world_z)};
  camera.nearclip = clip.x;
  camera.cliplength = clip.y;
  camera.differential_scale = clip.w;
  const bool orthographic = lens.x == float(ccl::CAMERA_ORTHOGRAPHIC);
  camera.type = orthographic ? ccl::CAMERA_ORTHOGRAPHIC : ccl::CAMERA_PERSPECTIVE;
  camera.shuttertime = -1.0f;
  camera.aperturesize = lens.y;
  camera.focaldistance = lens.z;
  camera.inv_aperture_ratio = lens.w;
  camera.blades = aperture.x;
  camera.bladesrotation = aperture.y;
  camera.dx = camera_row(dx);
  camera.dy = camera_row(dy);

  ccl::Ray ray{};
  rng_pixel = 0;
#ifdef CYCLES_WEBGPU_CAMERA_SAMPLE
  const auto pattern = cycles_camera_sample_word(0);
  if (pattern != ~0u) {
    load_sampling(globals.data.integrator, pattern, clip.z);
    const int x = int(pixel.x);
    const int y = int(pixel.y);
    const int sample = int(pixel.z);
    rng_pixel = ccl::path_rng_pixel_init(&globals, sample, x, y);
    int cache_miss = 0;
    ccl::integrate_camera_sample(&globals, sample, x, y, rng_pixel, &ray, cache_miss);
  }
  else
#endif
  {
    const auto raster = ccl::make_float2(float(pixel.x) + 0.5f, float(pixel.y) + 0.5f);
    if (orthographic) {
      ccl::camera_sample_orthographic(&globals, raster, ccl::zero_float2(), &ray);
    }
    else {
      ccl::camera_sample_perspective(&globals, raster, ccl::zero_float2(), &ray);
    }
  }
  origin = {ray.P.x, ray.P.y, ray.P.z, ray.tmin};
  direction = {ray.D.x, ray.D.y, ray.D.z, ray.tmax};
  dP = ray.dP;
  dD = ray.dD;
}

#undef kernel_data
#undef kernel_data_array
#undef kernel_data_fetch
#undef CCL_NAMESPACE_BEGIN
#undef CCL_NAMESPACE_END
#undef ccl_device_inline
#define CCL_NAMESPACE_BEGIN \
  namespace ccl { \
  struct WorkPixelContext { \
    WorkPixelData kernel_data;
#define CCL_NAMESPACE_END \
  }; \
  }
#define ccl_device_inline inline __attribute__((always_inline))

namespace ccl {
struct WorkPixelData {
  struct {
    float scrambling_distance;
  } integrator;
};
}

#include "kernel/device/gpu/work_stealing.h"

UInt4 cycles_work_pixel(unsigned int tile_x,
                        unsigned int tile_y,
                        unsigned int tile_width,
                        unsigned int tile_height,
                        unsigned int start_sample,
                        unsigned int num_samples,
                        unsigned int work_index,
                        float scrambling_distance)
{
  ccl::KernelWorkTile tile{};
  tile.x = tile_x;
  tile.y = tile_y;
  tile.w = tile_width;
  tile.h = tile_height;
  tile.start_sample = start_sample;
  tile.num_samples = num_samples;

  ccl::WorkPixelContext context{{{scrambling_distance}}};
  unsigned int x, y, sample;
  context.get_work_pixel(&tile, work_index, &x, &y, &sample);
  return {x, y, sample, 0};
}

#ifdef CYCLES_WEBGPU_FILM_AO
[[gnu::always_inline]] Float4 cycles_film_ao_sample(
    unsigned int rng_pixel, unsigned int sample, unsigned int bounce, Float4 normal)
{
  using namespace ccl;
  CameraGlobals globals;
  load_sampling(globals.data.integrator,
                cycles_camera_sample_word(0),
                std::bit_cast<float>(cycles_camera_sample_word(path_settings_offset() + 2)));
  const float2 random = path_rng_2D(
      &globals, rng_pixel, int(sample), (int(bounce) + 1) * PRNG_BOUNCE_NUM + PRNG_SURFACE_BSDF);
  float3 direction;
  float pdf;
  sample_cos_hemisphere(make_float3(normal.x, normal.y, normal.z), random, &direction, &pdf);
  return {direction.x, direction.y, direction.z, 0.0f};
}
#endif
