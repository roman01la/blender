/* SPDX-License-Identifier: Apache-2.0 */

#include "volume_kernels.hh"

#ifdef __aarch64__
#  include <arm_neon.h>
#endif

#define CCL_NAMESPACE_BEGIN namespace ccl {
#define CCL_NAMESPACE_END }
#include "kernel/types.h"
#include <bit>
#define __KERNEL_GPU__
#include "kernel/globals.h"
#undef __KERNEL_GPU__

#if defined(CYCLES_WEBGPU_VOLUME_SCATTER) || defined(CYCLES_WEBGPU_VOLUME_EXTINCTION_SHADOW)
#  include "../blender-platform/cycles-webgpu/volume_layout.h"
__attribute__((pure)) unsigned int cycles_volume_word(unsigned int index);
#endif

namespace ccl {
struct VolumeGlobals {
  KernelData data;
  float density_scale;
  uint grid_offset;
};
using KernelGlobals = const VolumeGlobals *;

/* This adapter handles homogeneous triangle volumes without primitive attributes. */
ccl_device_inline float object_volume_density(KernelGlobals kg, int /*object*/)
{
  return kg->density_scale;
}

#ifndef CYCLES_WEBGPU_VOLUME_GRID
ccl_device_inline AttributeDescriptor find_attribute(KernelGlobals, const ShaderData *, int)
{
  return {ATTR_ELEMENT_NONE, NODE_ATTR_FLOAT, 0};
}

template<typename T>
ccl_device_inline T primitive_volume_attribute(KernelGlobals,
                                                const ShaderData *,
                                                const AttributeDescriptor &,
                                                bool)
{
  return T{};
}
#endif

ccl_device_inline bool is_attribute_found(const AttributeDescriptor &desc)
{
  return desc.element != ATTR_ELEMENT_NONE;
}
}  // namespace ccl

#ifdef CYCLES_WEBGPU_VOLUME_GRID
#  include "volume_grid_sampling.h"
#endif

#define kernel_data (kg->data)
#define __VOLUME__
#undef ccl_device
#define ccl_device static inline __attribute__((always_inline))
#undef ccl_device_noinline
#define ccl_device_noinline static inline __attribute__((always_inline))
#include "kernel/closure/alloc.h"
#include "kernel/closure/emissive.h"
#include "kernel/closure/volume.h"
#include "kernel/tables.h"
#include "kernel/svm/node_types.h"
/* Bytecode reads are supplied by the dispatch. The unused stock buffer reader
 * stays undefined so accidental use fails at link time. */
namespace ccl {
const uint &volume_svm_word(KernelGlobals, uint);
}
#define kernel_data_fetch(name, index) volume_svm_word(kg, index)
#include "kernel/svm/util.h"
#undef kernel_data_fetch
#include "kernel/util/colorspace.h"
#include "volume_helpers.h"

#ifdef CYCLES_WEBGPU_VOLUME_SCATTER
#  include "../blender-platform/cycles-webgpu/lighting_layout.h"
#  include "../blender-platform/cycles-webgpu/path_state_layout.h"

__attribute__((pure)) unsigned int cycles_camera_sample_word(unsigned int index);
__attribute__((pure)) unsigned int cycles_light_word(unsigned int index);

static float volume_sample_pattern_lut(unsigned int index)
{
  return std::bit_cast<float>(cycles_camera_sample_word(7u + FILTER_TABLE_SIZE + index));
}

#  define kernel_data_fetch(name, index) volume_##name(index)
#  include "kernel/integrator/path_state.h"
#  undef kernel_data_fetch
#  include "kernel/util/differential.h"
#  include "kernel/light/common.h"
#  include "kernel/sample/mis.h"
#  include "util/color.h"
#  include "volume_scatter_helpers.h"
#endif

namespace {

using ccl::uint;

template<typename Node> Node volume_node(const unsigned int *words)
{
  Node node;
  __builtin_memcpy(&node, words, sizeof(Node));
  return node;
}

VolumeFloat4 packed(const ccl::Spectrum color, const float w)
{
  return {color.x, color.y, color.z, w};
}

void volume_evaluate_closure(
    ccl::KernelGlobals globals,
    ccl::ShaderData &sd,
    const VolumeUInt4 options,
    const ccl::Spectrum closure_weight,
    const unsigned int *words)
{
  using namespace ccl;
  const auto visibility = PathRayVisibility(options.y);
  const uint flags = options.z;
  /* The synchronized literal entries never read the SVM stack. */
  float *const stack = nullptr;
  switch (options.x) {
    case NODE_CLOSURE_VOLUME:
      svm_node_closure_volume<SHADER_TYPE_VOLUME>(
          globals, &sd, stack, closure_weight, volume_node<SVMNodeClosureVolume>(words));
      break;
    case NODE_VOLUME_COEFFICIENTS:
      svm_node_volume_coefficients<SHADER_TYPE_VOLUME>(
          globals, &sd, stack, closure_weight, volume_node<SVMNodeVolumeCoefficients>(words),
          visibility, flags);
      break;
    case NODE_PRINCIPLED_VOLUME:
      svm_node_principled_volume<SHADER_TYPE_VOLUME>(
          globals, &sd, stack, closure_weight, volume_node<SVMNodePrincipledVolume>(words),
          visibility, flags);
      break;
    default:
      break;
  }
}

void volume_begin_evaluation(ccl::ShaderData &sd, const ccl::PathRayVisibility visibility,
                             const uint flags)
{
  sd.num_closure = 0;
  sd.num_closure_left = (visibility & ccl::PATH_RAY_VISIBILITY_SHADOW) ||
                                (flags & ccl::PATH_RAY_EXTINCTION) ? 0 : MAX_CLOSURE;
  sd.runtime_flag = ccl::SR_IS_VOLUME_SHADER_EVAL;
}

ccl::VolumeShaderCoefficients volume_finish_evaluation(const ccl::ShaderData &sd,
                                                       ccl::ShaderVolumePhases &phases)
{
  using namespace ccl;
  volume_shader_copy_phases(&phases, &sd);
  const volatile auto &sigma = sd.closure_transparent_extinction;
  const volatile auto &emitted = sd.closure_emission_background;
  VolumeShaderCoefficients coeff{
      (sd.runtime_flag & SR_EXTINCTION) ? make_float3(sigma.x, sigma.y, sigma.z) : zero_spectrum(),
      zero_spectrum(),
      (sd.runtime_flag & SR_EMISSION) ? make_float3(emitted.x, emitted.y, emitted.z) : zero_spectrum()};
#pragma clang loop unroll(disable)
  for (int i = 0; i < phases.num_closure; ++i) {
    coeff.sigma_s += phases.closure[i].weight;
  }
  return coeff;
}

#if defined(CYCLES_WEBGPU_VOLUME_SCATTER) || defined(CYCLES_WEBGPU_VOLUME_EXTINCTION_SHADOW)
float volume_float_word(const uint index)
{
  return std::bit_cast<float>(cycles_volume_word(index));
}

void volume_eval_record(ccl::VolumeGlobals &globals, ccl::ShaderData &sd, const uint record,
                        const ccl::PathRayVisibility visibility, const uint flags)
{
  using namespace ccl;
  globals.density_scale = volume_float_word(record + CYCLES_WEBGPU_ABSORPTION_OBJECT_DENSITY);
  globals.grid_offset = cycles_volume_word(record + CYCLES_WEBGPU_VOLUME_GRID_OFFSET);
  sd.object = int(cycles_volume_word(record + CYCLES_WEBGPU_ABSORPTION_OBJECT));
  uint words[CYCLES_WEBGPU_VOLUME_NODE_WORDS];
#  pragma clang loop unroll(full)
  for (uint i = 0; i < CYCLES_WEBGPU_VOLUME_NODE_WORDS; ++i) {
    words[i] = cycles_volume_word(record + CYCLES_WEBGPU_VOLUME_DATA + i);
  }
  const Spectrum weight = make_float3(volume_float_word(record + CYCLES_WEBGPU_VOLUME_WEIGHT),
                                     volume_float_word(record + CYCLES_WEBGPU_VOLUME_WEIGHT + 1),
                                     volume_float_word(record + CYCLES_WEBGPU_VOLUME_WEIGHT + 2));
  const uint opcode = cycles_volume_word(record + CYCLES_WEBGPU_VOLUME_OPCODE);
  if (opcode & CYCLES_WEBGPU_VOLUME_ADD_ABSORPTION) {
    SVMNodeClosureVolume absorption{};
    absorption.closure_type = CLOSURE_VOLUME_ABSORPTION_ID;
    absorption.mix_weight_offset = SVM_STACK_INVALID;
    absorption.density.bits = cycles_volume_word(record + CYCLES_WEBGPU_VOLUME_ADD_ABSORPTION_DENSITY);
    const Spectrum color = make_float3(volume_float_word(record + CYCLES_WEBGPU_ABSORPTION_COLOR),
                                        volume_float_word(record + CYCLES_WEBGPU_ABSORPTION_COLOR + 1),
                                        volume_float_word(record + CYCLES_WEBGPU_ABSORPTION_COLOR + 2));
    svm_node_closure_volume<SHADER_TYPE_VOLUME>(&globals, &sd, nullptr, color, absorption);
  }
  volume_evaluate_closure(
      &globals, sd, VolumeUInt4{opcode & ~CYCLES_WEBGPU_VOLUME_ADD_ABSORPTION,
                                uint(visibility), flags, 0u},
      weight, words);
}

void volume_eval_stack(ccl::VolumeGlobals &globals, ccl::ShaderData &sd, const uint active,
                       const ccl::PathRayVisibility visibility, const uint flags)
{
  volume_begin_evaluation(sd, visibility, flags);
  const uint base = cycles_volume_word(CYCLES_WEBGPU_SURFACE_ABSORPTION_OFFSET);
  const uint count = cycles_volume_word(CYCLES_WEBGPU_SURFACE_VOLUME_COUNT);
  for (uint i = 0; i < count; ++i) {
    if (active & (1u << i)) {
      volume_eval_record(globals, sd, base + i * CYCLES_WEBGPU_VOLUME_SCATTER_WORDS,
                         visibility, flags);
    }
  }
}

ccl::VolumeShaderCoefficients volume_load_stack_coefficients(
    ccl::VolumeGlobals &globals, ccl::ShaderData &sd, ccl::ShaderVolumePhases &phases,
    const uint active, const ccl::PathRayVisibility visibility, const uint flags)
{
  volume_eval_stack(globals, sd, active, visibility, flags);
  return volume_finish_evaluation(sd, phases);
}
#endif

int volume_sample_phase(const ccl::ShaderData &sd,
                        const ccl::ShaderVolumePhases &phases,
                        ccl::float2 rand,
                        ccl::BsdfEval &eval,
                        ccl::float3 &direction,
                        float &pdf,
                        float &roughness)
{
  using namespace ccl;
  const int selected = volume_shader_phase_pick(&phases, &rand);
  ClosureType type = CLOSURE_NONE_ID;
  float sample_weight = 0.0f, p0 = 0.0f, p1 = 0.0f, p2 = 0.0f;
#pragma clang loop unroll(full)
  for (int i = 0; i < MAX_CLOSURE; ++i) {
    const volatile ShaderVolumeClosure &phase = phases.closure[i];
    /* Read fields at fixed typed offsets before selecting the sampled slot.
     * SPIR-V cannot address a private aggregate through a selected byte offset. */
    const ClosureType slot_type = phase.type;
    const float slot_weight = phase.sample_weight;
    const float slot_p0 = phase.param[0];
    const float slot_p1 = phase.param[1];
    const float slot_p2 = phase.param[2];
    if (selected == i) {
      type = slot_type;
      sample_weight = slot_weight;
      p0 = slot_p0;
      p1 = slot_p1;
      p2 = slot_p2;
    }
  }
  const ShaderVolumeClosure sample{zero_spectrum(), type, sample_weight, {p0, p1, p2}};
  return volume_shader_phase_sample(&sd, &phases, &sample, selected, rand, &eval, &direction,
                                    &pdf, &roughness);
}

}  // namespace

#ifdef CYCLES_WEBGPU_VOLUME_GRID
#  include "volume_grid_transport.h"
#endif

/* One literal, homogeneous volume SVM record. The transport caller must check
 * that the shader needs no SVM stack, object attributes, or other closures. */
void cycles_volume_coefficients(VolumeUInt4 options,
                                VolumeFloat4 weight,
                                VolumeUInt4 data0,
                                VolumeUInt4 data1,
                                VolumeUInt4 data2,
                                VolumeUInt4 data3,
                                VolumeUInt4 data4,
                                VolumeFloat4 incoming,
                                VolumeFloat4 random,
                                VolumeFloat4 &extinction,
                                VolumeFloat4 &scattering,
                                VolumeFloat4 &transmittance,
                                VolumeFloat4 &outgoing,
                                VolumeFloat4 &emission)
{
  using namespace ccl;
  VolumeGlobals globals{};
  globals.data.film.is_rec709 = true;
  globals.density_scale = weight.w;
  ShaderData sd;
  sd.wi = make_float3(incoming.x, incoming.y, incoming.z);
  ShaderVolumePhases phases;
  const uint words[]{data0.x, data0.y, data0.z, data0.w, data1.x, data1.y, data1.z, data1.w,
                     data2.x, data2.y, data2.z, data2.w, data3.x, data3.y, data3.z, data3.w,
                     data4.x, data4.y, data4.z, data4.w};
  sd.object = 0;
  volume_begin_evaluation(sd, PathRayVisibility(options.y), options.z);
  volume_evaluate_closure(&globals, sd, options, make_float3(weight.x, weight.y, weight.z), words);
  VolumeShaderCoefficients coeff = volume_finish_evaluation(sd, phases);

  const float distance = random.z;
  extinction = packed(coeff.sigma_t, float(phases.num_closure));
  scattering = packed(coeff.sigma_s, 0.0f);
  transmittance = packed(volume_color_transmittance(coeff.sigma_t, distance), 0.0f);
  outgoing = {};
  emission = packed(volume_emission_integrate(&coeff, sd.runtime_flag, distance), 0.0f);
  if (phases.num_closure > 0) {
    BsdfEval eval{};
    float3 direction = zero_float3();
    float pdf, roughness;
    const int label = volume_sample_phase(
        sd, phases, make_float2(random.x, random.y), eval, direction, pdf, roughness);
    BsdfEval direct{};
    const float eval_pdf = _volume_shader_phase_eval_mis(
        &sd, &phases, direction, -1, &direct, 0.0f, 0.0f);
    transmittance.w = pdf;
    outgoing = packed(direction, eval_pdf);
    scattering.w = float(label);
    emission.w = roughness;
  }
}

#ifdef CYCLES_WEBGPU_VOLUME_EXTINCTION_SHADOW
VolumeFloat4 cycles_volume_shadow_transmittance(unsigned int active,
                                                VolumeFloat4 origin,
                                                VolumeFloat4 direction,
#  ifdef CYCLES_WEBGPU_VOLUME_GRID
                                                VolumeUInt4 rng)
#  else
                                                VolumeUInt4 /*rng*/)
#  endif
{
  using namespace ccl;
  VolumeGlobals globals{};
  globals.data.film.is_rec709 = true;
  ShaderData sd;
  sd.wi = -make_float3(direction.x, direction.y, direction.z);
  sd.P = make_float3(origin.x, origin.y, origin.z) - sd.wi * origin.w;
#  ifdef CYCLES_WEBGPU_VOLUME_GRID
  float majorant;
  if (volume_grid_stack_majorant(
          globals, sd, active, PATH_RAY_VISIBILITY_SHADOW, PATH_RAY_EXTINCTION, majorant))
  {
    Ray ray{};
    ray.P = make_float3(origin.x, origin.y, origin.z);
    ray.D = -sd.wi;
    ray.tmin = origin.w;
    ray.tmax = direction.w;
    sd.lcg_state = lcg_state_init(rng.x, rng.z, rng.y, 0x15b4f88d);
    uint random = lcg_state_init(rng.x ^ __float_as_uint(origin.w), rng.z, rng.y, 0x6d45a681);
    const float order = lcg_step_float(&random);
    const float shade_offset = lcg_step_float(&random);
    return packed(volume_grid_transmittance(&globals, active, &ray, &sd, majorant,
                                             {ray.tmin, ray.tmax}, order, shade_offset,
                                             PATH_RAY_VISIBILITY_SHADOW, PATH_RAY_EXTINCTION),
                  0.0f);
  }
#  endif
  volume_eval_stack(globals, sd, active, PATH_RAY_VISIBILITY_SHADOW, PATH_RAY_EXTINCTION);
  const volatile auto &sigma = sd.closure_transparent_extinction;
  const Spectrum coeff = (sd.runtime_flag & SR_EXTINCTION) ?
                             make_float3(sigma.x, sigma.y, sigma.z) : zero_spectrum();
  return packed(volume_color_transmittance(coeff, direction.w - origin.w), 0.0f);
}
#endif

#ifdef CYCLES_WEBGPU_VOLUME_SCATTER
void cycles_volume_light_sample(unsigned int emitter,
                                VolumeFloat4 position,
                                VolumeFloat4 direction,
                                VolumeFloat4 random,
                                unsigned int bounce,
                                VolumeUInt4 &sampled,
                                VolumeFloat4 &point_pdf,
                                VolumeFloat4 &direction_distance,
                                VolumeFloat4 &strength_selection,
                                VolumeFloat4 &interval);

namespace {

void volume_load_sampling(ccl::KernelIntegrator &integrator)
{
  unsigned int offset = 7u + FILTER_TABLE_SIZE;
  integrator.sampling_pattern = cycles_camera_sample_word(0);
  integrator.seed = cycles_camera_sample_word(1);
  integrator.sobol_index_mask = cycles_camera_sample_word(2);
  integrator.blue_noise_sequence_length = cycles_camera_sample_word(3);
  integrator.tabulated_sobol_sequence_size = cycles_camera_sample_word(4);
  integrator.pixel_jitter = ccl::make_float2(std::bit_cast<float>(cycles_camera_sample_word(5)),
                                            std::bit_cast<float>(cycles_camera_sample_word(6)));
  if (integrator.sampling_pattern == ccl::SAMPLING_PATTERN_TABULATED_SOBOL) {
    offset += integrator.tabulated_sobol_sequence_size * NUM_TAB_SOBOL_PATTERNS *
              NUM_TAB_SOBOL_DIMENSIONS;
  }
  integrator.max_bounce = cycles_camera_sample_word(offset) &
                          ~CYCLES_WEBGPU_PATH_SETTINGS_DIFFERENTIALS;
  integrator.min_bounce = cycles_camera_sample_word(offset + 1);
  integrator.scrambling_distance = std::bit_cast<float>(cycles_camera_sample_word(offset + 2));
}

}  // namespace

/* phase_path.x distinguishes transmission, scatter, and termination. The
 * other fields are the stock next flags, visibility, and bounce on scatter. */
void cycles_volume_scatter(VolumeFloat4 origin,
                            VolumeFloat4 direction,
                            VolumeFloat4 throughput,
                            VolumeUInt4 path,
                            VolumeUInt4 sampling,
                            VolumeFloat4 &emission,
                            VolumeFloat4 &next_throughput,
                            VolumeUInt4 &phase_path,
                            VolumeFloat4 &phase_origin,
                            VolumeFloat4 &phase_direction,
                            VolumeFloat4 &mis_origin,
                            VolumeFloat4 &direct,
                            VolumeUInt4 &shadow_origin,
                            VolumeUInt4 &shadow_direction,
                            VolumeUInt4 &shadow_options
#  ifdef CYCLES_WEBGPU_VOLUME_GUIDES
                            , VolumeFloat4 &guide
#  endif
                            )
{
  using namespace ccl;
  const uint record = cycles_volume_word(CYCLES_WEBGPU_SURFACE_ABSORPTION_OFFSET);
  VolumeGlobals globals{};
  globals.data.film.is_rec709 = true;
  KernelIntegrator &integrator = globals.data.integrator;
  integrator.sample_clamp_direct = volume_float_word(record + CYCLES_WEBGPU_VOLUME_DIRECT_CLAMP);
  integrator.sample_clamp_indirect = volume_float_word(record + CYCLES_WEBGPU_VOLUME_INDIRECT_CLAMP);
  integrator.differential_widen_scale = fminf(
      1.0f, std::bit_cast<float>(cycles_light_word(CYCLES_WEBGPU_LIGHT_FILTER_GLOSSY)));
  integrator.direct_light_sampling_type = DIRECT_LIGHT_SAMPLING_MIS;
  const uint lighting = cycles_light_word(CYCLES_WEBGPU_LIGHT_TREE_OFFSET);
  if (cycles_light_word(lighting + CYCLES_WEBGPU_TREE_NODE_COUNT) == 0) {
    integrator.light_inv_rr_threshold =
        std::bit_cast<float>(cycles_light_word(CYCLES_WEBGPU_LIGHT_INV_RR_THRESHOLD));
  }

  ShaderData sd;
  sd.wi = -make_float3(direction.x, direction.y, direction.z);
  sd.time = 0.0f;
  sd.dP = 0.0f;
  sd.P = make_float3(origin.x, origin.y, origin.z) - sd.wi * origin.w;
  ShaderVolumePhases phases;
  emission = {};
  next_throughput = throughput;
  phase_path = {};
  phase_origin = {};
  phase_direction = {};
  mis_origin = {};
  direct = {};
  shadow_origin = {};
  shadow_direction = {};
  shadow_options = {};
#  ifdef CYCLES_WEBGPU_VOLUME_GUIDES
  guide = {};
#  endif
  if (path.w == 0 || !(direction.w > origin.w)) {
    return;
  }
#  ifdef CYCLES_WEBGPU_VOLUME_GRID
  sd.lcg_state = lcg_state_init(sampling.x, sampling.z, sampling.y, 0x15b4f88d);
  float majorant;
  const bool has_grid = volume_grid_stack_majorant(
      globals, sd, path.w, PathRayVisibility(path.z), sampling.w, majorant);
  VolumeShaderCoefficients coeff = has_grid ? VolumeShaderCoefficients{} :
      volume_load_stack_coefficients(globals, sd, phases, path.w, PathRayVisibility(path.z),
                                     sampling.w);
  const bool has_scatter = has_grid || !is_zero(coeff.sigma_s);
#  else
  VolumeShaderCoefficients coeff = volume_load_stack_coefficients(
      globals, sd, phases, path.w, PathRayVisibility(path.z), sampling.w);
  const bool has_scatter = !is_zero(coeff.sigma_s);
#  endif
  volume_load_sampling(integrator);
  integrator.max_volume_bounce = cycles_volume_word(record + CYCLES_WEBGPU_VOLUME_MAX_BOUNCES);

  IntegratorStateCPU state{};
  state.path.flag = sampling.w;
#  ifdef CYCLES_WEBGPU_VOLUME_GUIDES
  state.path.flag |= PATH_RAY_DENOISING_FEATURES;
#  endif
  state.path.bounce = path.x;
  state.path.volume_bounce = path.y;
  state.path.visibility = path.z;
  state.path.rng_pixel = sampling.x;
  state.path.sample = int(sampling.y);
  state.path.rng_offset = sampling.z;
  state.path.throughput = make_float3(throughput.x, throughput.y, throughput.z);
  const RNGState rng{sampling.x, sampling.z, int(sampling.y)};
  const bool terminate = throughput.w == 0.0f || (sampling.w & PATH_RAY_TERMINATE_IN_NEXT_VOLUME);
  if (terminate) {
    state.path.flag |= PATH_RAY_TERMINATE_IN_NEXT_VOLUME;
  }

  Ray ray{};
  ray.P = make_float3(origin.x, origin.y, origin.z);
  ray.D = -sd.wi;
  ray.tmin = origin.w;
  ray.tmax = direction.w;
  const float dt = ray.tmax - ray.tmin;
  const float3 start = ray.P + ray.D * ray.tmin;

  VolumeSampleMethod method = VOLUME_SAMPLE_NONE;
  VolumeUInt4 candidate{~0u, 0, 0, 0};
  VolumeFloat4 candidate_point{}, candidate_direction{}, candidate_strength{}, valid_interval{};
  VolumeFloat4 random{};
  EquiangularCoefficients equiangular{zero_float3(), {ray.tmin, ray.tmax}};
  if (!(state.path.flag & PATH_RAY_TERMINATE) && has_scatter) {
    random = packed(path_state_rng_3D(&globals, &rng, PRNG_LIGHT), 0.0f);
    cycles_volume_light_sample(
        ~0u, packed(start, dt), packed(ray.D, 0.0f), random, state.path.bounce, candidate,
        candidate_point, candidate_direction, candidate_strength, valid_interval);
    if (candidate.x != ~0u && !(candidate.y & SHADER_EXCLUDE_SCATTER) &&
        valid_interval.x < valid_interval.y)
    {
      equiangular.P = make_float3(candidate_point.x, candidate_point.y, candidate_point.z);
      equiangular.t_range = {ray.tmin + valid_interval.x, ray.tmin + valid_interval.y};
      uint sampling_methods = VOLUME_SAMPLE_NONE;
      const uint count = cycles_volume_word(CYCLES_WEBGPU_SURFACE_VOLUME_COUNT);
      for (uint i = 0; i < count; ++i) {
        if (path.w & (1u << i)) {
          const uint shader = cycles_volume_word(
              record + i * CYCLES_WEBGPU_VOLUME_SCATTER_WORDS + CYCLES_WEBGPU_VOLUME_SHADER_FLAGS);
          sampling_methods |= (shader & SD_VOLUME_MIS) ? VOLUME_SAMPLE_MIS :
                              (shader & SD_VOLUME_EQUIANGULAR) ? VOLUME_SAMPLE_EQUIANGULAR :
                                                                VOLUME_SAMPLE_DISTANCE;
        }
      }
      method = VOLUME_SAMPLE_DISTANCE;
      if (candidate_direction.w != FLT_MAX) {
        method = VolumeSampleMethod(sampling_methods);
      }
    }
  }

  VolumeIntegrateState vstate;
  VolumeIntegrateResult result{};
  volume_integrate_state_init(&globals, &state, method, &rng, ray.tmin, vstate);
  volume_integrate_result_init(&state, &ray, vstate, equiangular, result);
#  ifdef CYCLES_WEBGPU_VOLUME_GRID
  uint direct_seed = sd.lcg_state, indirect_seed = sd.lcg_state;
  if (has_grid) {
    volume_grid_integrate(globals, path.w, &state, ray, sd, rng, majorant,
                          vstate, result, direct_seed, indirect_seed);
  }
  else
#  endif
  {
    volume_integrate_homogeneous(
        &globals, &state, &ray, coeff, sd.runtime_flag, &rng, vstate, equiangular.t_range, result);
  }
  volume_direct_scatter_mis(&ray, vstate, equiangular, result);
  Spectrum L = vstate.emission;
  film_clamp_light(&globals, &L, int(state.path.bounce) - 1);
  emission = packed(L, 0.0f);
#  ifdef CYCLES_WEBGPU_VOLUME_GUIDES
  /* The stock film records both volume emission and integrated albedo as guide
   * albedo, before stopping feature writes at an indirect scatter. */
  guide = packed(vstate.albedo + L, float(result.indirect_scatter));
  state.path.flag &= ~PATH_RAY_DENOISING_FEATURES;
#  endif

  if (terminate) {
    phase_path.x = 2;
    next_throughput = {};
    return;
  }

#  ifdef CYCLES_WEBGPU_VOLUME_GRID
  if (has_grid && result.direct_scatter) {
    sd.P = ray.P + result.direct_t * ray.D;
    sd.lcg_state = direct_seed;
    volume_load_stack_coefficients(globals, sd, phases, path.w, PathRayVisibility(path.z), sampling.w);
  }
#  endif
  if (result.direct_scatter && phases.num_closure > 0) {
    const float3 P = ray.P + result.direct_t * ray.D;
    VolumeUInt4 sampled;
    VolumeFloat4 point_pdf, direction_distance, strength, interval;
    cycles_volume_light_sample(candidate.x, packed(P, 0.0f),
                               packed(ray.D, candidate_strength.w), random, state.path.bounce,
                               sampled, point_pdf, direction_distance, strength, interval);
    if (sampled.x != ~0u && !(sampled.y & SHADER_EXCLUDE_SCATTER) && point_pdf.w > 0.0f) {
      LightSample ls{};
      ls.P = make_float3(point_pdf.x, point_pdf.y, point_pdf.z);
      ls.D = make_float3(direction_distance.x, direction_distance.y, direction_distance.z);
      ls.t = direction_distance.w;
      ls.shader = int(sampled.y);
      ls.prim = int(sampled.w);
      ls.object = OBJECT_NONE;
      BsdfEval eval{};
      const float phase_pdf = _volume_shader_phase_eval_mis(
          &sd, &phases, ls.D, -1, &eval, 0.0f, 0.0f);
      const float mis = light_sample_mis_weight_nee(
          &globals, point_pdf.w, (sampled.y & SHADER_USE_MIS) ? phase_pdf : 0.0f);
      bsdf_eval_mul(&eval, make_float3(strength.x, strength.y, strength.z) * (mis / point_pdf.w));
      const float terminate = path_state_rng_light_termination(&globals, &rng);
      const Spectrum direct_l = light_sample_terminate(&globals, &eval, terminate) ?
                                    zero_spectrum() :
                                    result.direct_throughput * bsdf_eval_sum(&eval) * throughput.w;
      if (!is_zero(direct_l)) {
        Ray shadow{};
        light_sample_to_volume_shadow_ray(&sd, &ls, P, &shadow);
        shadow_origin = {__float_as_uint(shadow.P.x),
                         __float_as_uint(shadow.P.y),
                         __float_as_uint(shadow.P.z), 0};
        shadow_direction = {__float_as_uint(shadow.D.x), __float_as_uint(shadow.D.y),
                            __float_as_uint(shadow.D.z), __float_as_uint(shadow.tmax)};
        shadow_options = {(ls.shader & SHADER_CAST_SHADOW) ? 1u : 2u,
                          ~0u,
                          ~0u,
                          uint(state.path.bounce != 0)};
        direct = packed(direct_l, 0.0f);
      }
    }
  }

  next_throughput = packed(result.indirect_throughput, 0.0f);
#  ifdef CYCLES_WEBGPU_VOLUME_GRID
  if (has_grid && result.indirect_scatter) {
    sd.P = ray.P + result.indirect_t * ray.D;
    sd.lcg_state = indirect_seed;
    volume_load_stack_coefficients(globals, sd, phases, path.w, PathRayVisibility(path.z), sampling.w);
  }
#  endif
  if (!result.indirect_scatter || phases.num_closure == 0) {
    return;
  }

  BsdfEval phase_eval{};
  float3 wo = zero_float3();
  float phase_pdf, sampled_roughness;
  const int label = volume_sample_phase(sd, phases,
      path_state_rng_2D(&globals, &rng, PRNG_VOLUME_PHASE), phase_eval, wo, phase_pdf,
      sampled_roughness);
  if (!(phase_pdf > 0.0f) || is_zero(phase_eval.sum)) {
    phase_path.x = 2;
    next_throughput = {};
    return;
  }

  const float3 P = ray.P + ray.D * result.indirect_t;
  const Spectrum next = result.indirect_throughput * (throughput.w / phase_pdf) * phase_eval.sum;
  state.path.throughput = next;
  path_state_next(&globals, &state, label, sd.runtime_flag);
  const float p = path_state_continuation_probability(&globals, &state, state.path.flag);
  const bool accepted = p == 1.0f ||
                        (p > 0.0f && path_rng_1D(&globals, sampling.x, int(sampling.y),
                          state.path.rng_offset + PRNG_TERMINATE) < p);
  next_throughput = packed(next, result.indirect_t);
  emission.w = volume_phase_widen_dD(&globals, 0.0f, sampled_roughness);
  phase_path = {1u, state.path.flag, path_state_ray_visibility(&state), state.path.bounce};
  phase_origin = packed(P, accepted ? 1.0f / p : 0.0f);
  phase_direction = packed(normalize(wo), phase_pdf);
  mis_origin = packed(P - start, dt);
}
#endif
