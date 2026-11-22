/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/sample/mapping.h"

namespace {
using namespace ccl;

constexpr int VOLUME_MAX_STEPS = 1024; /* Cycles kernel/integrator/shade_volume.h */

bool volume_grid_stack_majorant(VolumeGlobals &globals,
                                 ShaderData &sd,
                                 const uint active,
                                 const PathRayVisibility visibility,
                                 const uint flags,
                                 float &majorant)
{
  const uint first = cycles_volume_word(CYCLES_WEBGPU_SURFACE_ABSORPTION_OFFSET);
  const uint count = cycles_volume_word(CYCLES_WEBGPU_SURFACE_VOLUME_COUNT);
  majorant = 0.0f;
  bool has_grid = false;
  volume_begin_evaluation(sd, visibility, flags);
  for (uint i = 0; i < count; ++i) {
    if (!(active & (1u << i))) {
      continue;
    }
    const uint record = first + i * CYCLES_WEBGPU_VOLUME_SCATTER_WORDS;
    const uint grid = cycles_volume_word(record + CYCLES_WEBGPU_VOLUME_GRID_OFFSET);
    if (grid != 0) {
      has_grid = true;
      majorant += grid_float_word(grid + CYCLES_WEBGPU_GRID_MAJORANT);
    }
    else {
      volume_eval_record(globals, sd, record, visibility, flags);
    }
  }
  if (has_grid) {
    const auto &sigma = sd.closure_transparent_extinction;
    majorant += (sd.runtime_flag & SR_EXTINCTION) ? reduce_max(sigma) : 0.0f;
    const auto &emission = sd.closure_emission_background;
    majorant += (sd.runtime_flag & SR_EMISSION) ? reduce_max(emission) : 0.0f;
  }
  return has_grid;
}

Spectrum volume_grid_eval_extinction(VolumeGlobals *globals,
                                      const uint active,
                                      ShaderData *sd,
                                      const PathRayVisibility visibility,
                                      const uint flags)
{
  volume_eval_stack(*globals, *sd, active, visibility, flags | PATH_RAY_EXTINCTION);
  const volatile auto &sigma = sd->closure_transparent_extinction;
  return (sd->runtime_flag & SR_EXTINCTION) ? make_float3(sigma.x, sigma.y, sigma.z) :
                                             zero_spectrum();
}

#include "volume_grid_transmittance.h"

#ifdef CYCLES_WEBGPU_VOLUME_SCATTER
struct WebGPUVolumeSegment {
  Extrema<float> sigma;
  Interval<float> t;
};

bool volume_grid_shader_sample(VolumeGlobals *globals,
                                const uint active,
                                IntegratorState state,
                                ShaderData *sd,
                                VolumeShaderCoefficients *coeff)
{
  volume_eval_stack(*globals, *sd, active, state->path.visibility, state->path.flag);
  const volatile auto &sigma = sd->closure_transparent_extinction;
  const volatile auto &emission = sd->closure_emission_background;
  coeff->sigma_t = (sd->runtime_flag & SR_EXTINCTION) ? make_float3(sigma.x, sigma.y, sigma.z) :
                                                       zero_spectrum();
  coeff->emission = (sd->runtime_flag & SR_EMISSION) ?
                         make_float3(emission.x, emission.y, emission.z) :
                         zero_spectrum();
  coeff->sigma_s = zero_spectrum();
  if (sd->runtime_flag & SR_SCATTER) {
    for (int i = 0; i < sd->num_closure; ++i) {
      const volatile auto &weight = sd->closure[i].weight;
      coeff->sigma_s += make_float3(weight.x, weight.y, weight.z);
    }
  }
  return sd->runtime_flag & (SR_EXTINCTION | SR_SCATTER | SR_EMISSION);
}

#include "volume_grid_steps.h"

void volume_grid_integrate(VolumeGlobals &globals,
                            const uint active,
                            const IntegratorState state,
                            const Ray &ray,
                            ShaderData &sd,
                            RNGState rng,
                            const float majorant,
                            VolumeIntegrateState &vstate,
                            VolumeIntegrateResult &result,
                            uint &direct_seed,
                            uint &indirect_seed)
{
  const WebGPUVolumeSegment segment{{0.0f, majorant}, {ray.tmin, ray.tmax}};
  vstate.vspg = false;
  vstate.optical_depth = majorant * segment.t.length();
  VolumeSampleReservoir reservoir = path_state_rng_1D(&globals, &rng, PRNG_VOLUME_RESERVOIR);
  path_state_rng_scramble(&rng, 0xe35fad82);

  volume_equiangular_transmittance(
      &globals, active, state, &ray, segment.sigma, segment.t, &sd, &rng, vstate, result);
  const bool equiangular = vstate.direct_sample_method == VOLUME_SAMPLE_EQUIANGULAR &&
                           vstate.use_mis;
  /* Keep the bounded loop counter at 32 bits, as required by WGSL. */
  volatile uint step = 0;
  while (step <= VOLUME_MAX_STEPS) {
    step = step + 1;
    float residual_optical_depth;
    vstate.rscatter = path_state_rng_1D(&globals, &rng, PRNG_VOLUME_SCATTER_DISTANCE);
    if (volume_indirect_scatter_advance(
            segment, equiangular, residual_optical_depth, vstate, result))
    {
      break;
    }
    rng.rng_offset += PRNG_BOUNCE_NUM;
    volume_integrate_step_scattering(&globals, active, state, &ray, majorant, &sd,
                                      vstate, result, reservoir);
    if (volume_integrate_should_stop(result)) {
      break;
    }
  }

  volume_distance_sampling_finalize(state, vstate, result, reservoir);
  indirect_seed = reservoir.is_empty() ? sd.lcg_state : reservoir.candidate.lcg_state;
  direct_seed = vstate.direct_sample_method == VOLUME_SAMPLE_DISTANCE ? indirect_seed :
                                                                        sd.lcg_state;
  volume_equiangular_direct_scatter(&globals, active, state, &ray, &sd, vstate, result);
}
#endif

}  // namespace
