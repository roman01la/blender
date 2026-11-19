/* SPDX-License-Identifier: Apache-2.0 */

#include "../blender-platform/cycles-webgpu/path_state_layout.h"

typedef float float3 __attribute__((ext_vector_type(3)));

StructuredBuffer<uint> ray_words : register(t0);
StructuredBuffer<uint> hit_words : register(t1);
StructuredBuffer<uint> light_words : register(t2);
StructuredBuffer<uint> camera_sampling : register(t3);
RWStructuredBuffer<uint> path_words : register(u4);
#ifdef CYCLES_WEBGPU_DENOISE_BACKGROUND
RWStructuredBuffer<float4> guide_state : register(u5);
#endif

#include "wavefront_queue.hlsl"
#include "wavefront_light.hlsl"
#include "wavefront_radiance.hlsl"

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
  const uint queued = path + CYCLES_WEBGPU_PATH_QUEUED_KERNEL;
  if (path_words[queued] == CYCLES_WEBGPU_PATH_TERMINATED) {
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
  float4 world, area;
#ifdef CYCLES_WEBGPU_LIGHT_TREE
  const uint mis_normal_word = path + CYCLES_WEBGPU_PATH_MIS_ORIGIN_N;
  const uint4 mis_origin_normal = uint4(path_words[mis_normal_word],
                                       path_words[mis_normal_word + 1],
                                       path_words[mis_normal_word + 2], 0);
#endif
  cycles_wavefront_light(no_surface, no_surface, no_surface, no_surface, no_surface, no_surface,
                         closest, ray, origin, state, no_surface,
                         no_material, no_material, no_material, no_material, world, area
#ifdef CYCLES_WEBGPU_LIGHT_TREE
                         , mis_origin_normal
#endif
                         );

  const uint weight = path + CYCLES_WEBGPU_PATH_THROUGHPUT;
  const float3 throughput = __builtin_bit_cast(
      float3, uint3(path_words[weight], path_words[weight + 1], path_words[weight + 2]));
  const uint sum = path + CYCLES_WEBGPU_PATH_RADIANCE;
  float3 rgb = __builtin_bit_cast(
      float3, uint3(path_words[sum], path_words[sum + 1], path_words[sum + 2]));
  bool trace_past_area = true;
#ifdef CYCLES_WEBGPU_DENOISE_BACKGROUND
  const uint feature = lane * 3;
  const float4 feature_throughput = guide_state[feature];
  float3 albedo = float3(0, 0, 0);
#endif
  if (area.w > 0.0f) {
    trace_past_area = cycles_trace_past_area(path);
    const float3 contribution = clamp_light(throughput * area.xyz, area.w);
    rgb += contribution;
#ifdef CYCLES_WEBGPU_DENOISE_BACKGROUND
    albedo += feature_throughput.xyz * contribution;
#endif
  }
  if (trace_past_area && world.w < 0.0f) {
    const float3 contribution = clamp_light(throughput * world.xyz, -world.w);
    rgb += contribution;
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
  path_words[queued] = CYCLES_WEBGPU_PATH_TERMINATED;
}
