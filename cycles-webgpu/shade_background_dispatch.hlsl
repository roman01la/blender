/* SPDX-License-Identifier: Apache-2.0 */

#include "../blender-platform/cycles-webgpu/path_state_layout.h"
#include "../blender-platform/cycles-webgpu/lighting_layout.h"

typedef float float3 __attribute__((ext_vector_type(3)));

StructuredBuffer<uint> ray_words : register(t0);
StructuredBuffer<uint> hit_words : register(t1);
StructuredBuffer<uint> light_words : register(t2);
StructuredBuffer<uint> camera_sampling : register(t3);
RWStructuredBuffer<uint> path_words : register(u4);
#ifdef CYCLES_WEBGPU_DENOISE_BACKGROUND
RWStructuredBuffer<float4> guide_state : register(u5);
RWStructuredBuffer<float4> radiance : register(u6);
#else
RWStructuredBuffer<float4> radiance : register(u5);
#endif

#define CYCLES_WEBGPU_EMISSION_DELTA 1
#include "wavefront_queue.hlsl"
#include "wavefront_light.hlsl"
#include "wavefront_radiance.hlsl"
#define CYCLES_WEBGPU_LIGHT_PASS_BUFFER radiance
#include "light_passes.hlsl"

uint cycles_light_word(uint index)
{
  return light_words[index];
}

uint cycles_camera_sample_word(uint index)
{
  return camera_sampling[index];
}

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
  uint lane;
  if (!cycles_queue_slot(id.x, lane)) {
    return;
  }

  const uint path = lane * CYCLES_WEBGPU_PATH_WORDS;
  const uint capacity = ray_words[0];
  const bool light_passes = cycles_light_pass_enabled(capacity);
  const uint queued = path + CYCLES_WEBGPU_PATH_QUEUED_KERNEL;
  if (path_words[queued] == CYCLES_WEBGPU_PATH_TERMINATED) {
    cycles_write_emission_delta(ray_words[0], lane, 0u, 0u, float3(0, 0, 0));
    if (light_passes) {
      cycles_light_pass_deltas(capacity, lane, 0u, float3(0, 0, 0), float3(0, 0, 0), false);
    }
    return;
  }

  const uint ray_base = 1 + 12 * lane;
  const uint4 origin = uint4(ray_words[ray_base], ray_words[ray_base + 1],
                             ray_words[ray_base + 2], ray_words[ray_base + 3]);
  const uint4 ray = uint4(ray_words[ray_base + 4], ray_words[ray_base + 5],
                          ray_words[ray_base + 6], ray_words[ray_base + 8]);
  const uint4 state = uint4(path_words[path + CYCLES_WEBGPU_PATH_BOUNCE],
                            path_words[path + CYCLES_WEBGPU_PATH_FLAGS],
                            path_words[path + CYCLES_WEBGPU_PATH_MIS_RAY_PDF], 0);
  const uint4 closest = uint4(0, 0, 0, hit_words[8 * lane + 4]);
  // A classified miss has no surface contribution. Passing a known absent hit
  // lets the compiler strip BSDF evaluation and shadow traversal.
  const uint4 no_surface = uint4(0, 0, 0, 0);
  const float4 no_material = float4(0, 0, 0, 0);
  const uint weight = path + CYCLES_WEBGPU_PATH_THROUGHPUT;
  const float3 throughput = __builtin_bit_cast(
      float3, uint3(path_words[weight], path_words[weight + 1], path_words[weight + 2]));
  float4 world, area;
  float4 unused_diffuse_ratio, unused_glossy_ratio;
  uint unused_hair_lcg;
#ifdef CYCLES_WEBGPU_LIGHT_TREE
  float4 sun_emission;
  const uint mis_normal_word = path + CYCLES_WEBGPU_PATH_MIS_ORIGIN_N;
  uint volume_segment_length = 0u;
#  ifdef CYCLES_WEBGPU_VOLUME_FORWARD
  if ((ray.w & CYCLES_WEBGPU_VOLUME_SCATTER_VISIBILITY) != 0u) {
    const bool differentials = (cycles_light_word(CYCLES_WEBGPU_LIGHT_EXTENDED) &
                                CYCLES_WEBGPU_LIGHT_VOLUME_DIFFERENTIALS) != 0u;
    volume_segment_length = ray_words[CYCLES_WEBGPU_RAY_VOLUME_MIS_LENGTH(
        ray_words[0], lane, differentials)];
  }
#  endif
  const uint4 mis_origin_normal = uint4(path_words[mis_normal_word],
                                       path_words[mis_normal_word + 1],
                                       path_words[mis_normal_word + 2], volume_segment_length);
#endif
  cycles_wavefront_light(no_surface, no_surface, no_surface, no_surface, no_surface, no_surface,
                         closest, ray, origin, state,
                         uint4(0, 0, 0, path_words[path + CYCLES_WEBGPU_PATH_TRANSPARENT_BOUNCE] &
                                            CYCLES_WEBGPU_TRANSPARENT_BOUNCE_MASK),
                         0u,
                         no_material, no_material, no_material, no_material, no_material,
                         no_material, world, area, unused_hair_lcg, 0u, unused_diffuse_ratio,
                         unused_glossy_ratio
#ifdef CYCLES_WEBGPU_LIGHT_TREE
                         , mis_origin_normal, float4(throughput, 0.0f), sun_emission
#endif
                         );

  const uint sum = path + CYCLES_WEBGPU_PATH_RADIANCE;
  float3 rgb = __builtin_bit_cast(
      float3, uint3(path_words[sum], path_words[sum + 1], path_words[sum + 2]));
  bool trace_past_area = true;
#ifdef CYCLES_WEBGPU_DENOISE_BACKGROUND
  const uint feature = lane * 3;
  const float4 feature_throughput = guide_state[feature];
  float3 albedo = float3(0, 0, 0);
#endif
  float3 emission_delta = float3(0, 0, 0);
  const uint forward_hits = cycles_forward_light_hits(area);
  if (forward_hits != 0) {
    trace_past_area = cycles_trace_past_area(path, forward_hits);
    const float3 contribution = cycles_forward_emission(area, throughput);
    rgb += contribution;
    emission_delta += contribution;
#ifdef CYCLES_WEBGPU_DENOISE_BACKGROUND
    albedo += feature_throughput.xyz * contribution;
#endif
  }
#ifdef CYCLES_WEBGPU_LIGHT_TREE
  if (trace_past_area) {
    rgb += sun_emission.xyz;
    emission_delta += sun_emission.xyz;
#  ifdef CYCLES_WEBGPU_DENOISE_BACKGROUND
    albedo += feature_throughput.xyz * sun_emission.xyz;
#  endif
  }
#endif
  float3 forward_delta = emission_delta;
  if (trace_past_area && !cycles_write_transparent_background(path, state.y, throughput) &&
      world.w < 0.0f) {
    const float3 contribution = clamp_light(throughput * world.xyz, -world.w);
    rgb += contribution;
    forward_delta += contribution;
#ifdef CYCLES_WEBGPU_DENOISE_BACKGROUND
    albedo += feature_throughput.xyz * contribution;
#endif
  }
#ifdef CYCLES_WEBGPU_DENOISE_BACKGROUND
  guide_state[feature + 1] += float4(albedo, 0.0f);
#endif
  const uint3 bits = __builtin_bit_cast(uint3, rgb);
  path_words[sum] = bits.x;
  path_words[sum + 1] = bits.y;
  path_words[sum + 2] = bits.z;
  cycles_write_emission_delta(ray_words[0], lane, state.x, state.y, emission_delta);
  if (light_passes) {
    cycles_light_pass_deltas(capacity, lane, state.x, forward_delta, float3(0, 0, 0), false);
  }
  path_words[queued] = CYCLES_WEBGPU_PATH_TERMINATED;
}
