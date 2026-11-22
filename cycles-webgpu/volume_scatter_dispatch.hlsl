/* SPDX-License-Identifier: Apache-2.0 */

#include "../blender-platform/cycles-webgpu/path_state_layout.h"
#include "../blender-platform/cycles-webgpu/volume_layout.h"

RWStructuredBuffer<uint> ray_words : register(u0);
RWStructuredBuffer<uint> hit_words : register(u1);
StructuredBuffer<uint> metadata_words : register(t2);
RWStructuredBuffer<uint> path_words : register(u3);
RWStructuredBuffer<uint> volume_state : register(u4);
StructuredBuffer<uint> sampling_words : register(t5);
StructuredBuffer<uint> light_words : register(t6);
RWStructuredBuffer<uint4> shadow_words : register(u7);

#include "wavefront_queue.hlsl"

uint cycles_volume_word(uint index)
{
  return metadata_words[index];
}

uint cycles_camera_sample_word(uint index)
{
  return sampling_words[index];
}

uint cycles_light_word(uint index)
{
  return light_words[index];
}

extern void cycles_volume_scatter(float4 origin, float4 direction, float4 throughput,
                                   uint4 path, uint4 sampling, out float4 emission,
                                   out float4 next_throughput,
                                   out uint4 phase_path, out float4 phase_origin,
                                   out float4 phase_direction, out float4 mis_origin,
                                   out float4 direct, out uint4 shadow_origin,
                                   out uint4 shadow_direction, out uint4 shadow_options
#ifdef CYCLES_WEBGPU_VOLUME_GUIDES
                                   , out float4 guide
#endif
                                   );

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
  uint lane;
  if (!cycles_queue_slot(id.x, lane)) {
    return;
  }
  const uint capacity = ray_words[0];
  const uint ray = 1 + 12 * lane;
  const uint hit = 8 * lane;
  const uint path = CYCLES_WEBGPU_PATH_WORDS * lane;
  const uint weight = path + CYCLES_WEBGPU_PATH_THROUGHPUT;
  const uint film = path + CYCLES_WEBGPU_PATH_RADIANCE;
  const uint flags = path_words[path + CYCLES_WEBGPU_PATH_FLAGS];
  const uint bounce = path_words[path + CYCLES_WEBGPU_PATH_BOUNCE];
  const uint steps = path_words[path + CYCLES_WEBGPU_PATH_EXTRA_RNG_STEPS];
  const uint volume_bounce = CYCLES_WEBGPU_VOLUME_BOUNCE * capacity + lane;
  const uint stack = volume_state[lane];
  const uint active = stack & CYCLES_WEBGPU_VOLUME_MEDIA_MASK;
  const bool integrate = active != 0u && hit_words[hit] != 0u;
  const float4 origin = __builtin_bit_cast(
      float4, uint4(ray_words[ray], ray_words[ray + 1], ray_words[ray + 2], ray_words[ray + 3]));
  const float4 direction = __builtin_bit_cast(
      float4, uint4(ray_words[ray + 4], ray_words[ray + 5], ray_words[ray + 6], hit_words[hit + 4]));
  const float4 throughput = __builtin_bit_cast(
      float4, uint4(path_words[weight], path_words[weight + 1], path_words[weight + 2],
                    path_words[path + CYCLES_WEBGPU_PATH_SURFACE_RR]));
  const uint4 rng = uint4(ray_words[ray + CYCLES_WEBGPU_RAY_RNG_PIXEL],
                         path_words[path + CYCLES_WEBGPU_PATH_PIXEL + 2],
                         (bounce + steps + 1u) * CYCLES_WEBGPU_PRNG_BOUNCE_NUM, flags);
  const uint4 input_path = uint4(bounce, volume_state[volume_bounce], ray_words[ray + 8],
                                 integrate ? active : 0u);

  float4 emission, next_throughput, phase_origin, phase_direction, mis_origin, direct;
  uint4 phase_path, shadow_origin, shadow_direction, shadow_options;
#ifdef CYCLES_WEBGPU_VOLUME_GUIDES
  float4 guide;
#endif
  cycles_volume_scatter(origin, direction, throughput, input_path, rng,
                        emission, next_throughput, phase_path, phase_origin,
                        phase_direction, mis_origin, direct, shadow_origin, shadow_direction,
                        shadow_options
#ifdef CYCLES_WEBGPU_VOLUME_GUIDES
                        , guide
#endif
                        );

  const uint shadow = 4 * lane;
  shadow_words[shadow + 2] = shadow_options;
#ifdef CYCLES_WEBGPU_VOLUME_GUIDES
  /* Volume NEE has no surface offset record in this slot. The matching queued
   * shadow dispatch consumes its guide payload even without a light sample. */
  shadow_words[shadow + 3] = __builtin_bit_cast(uint4, guide);
#endif
  if (shadow_options.x != 0u) {
    const uint rgb = CYCLES_WEBGPU_VOLUME_DIRECT * capacity + lane;
    const uint4 rgb_bits = __builtin_bit_cast(uint4, direct);
    volume_state[rgb] = rgb_bits.x;
    volume_state[rgb + capacity] = rgb_bits.y;
    volume_state[rgb + 2 * capacity] = rgb_bits.z;
    volume_state[CYCLES_WEBGPU_VOLUME_SHADOW_STACK * capacity + lane] = stack;
    shadow_words[shadow] = shadow_origin;
    shadow_words[shadow + 1] = shadow_direction;
  }

  if (integrate) {
    const uint4 next = __builtin_bit_cast(uint4, next_throughput);
    path_words[weight] = next.x;
    path_words[weight + 1] = next.y;
    path_words[weight + 2] = next.z;
    const float4 current = __builtin_bit_cast(
        float4, uint4(path_words[film], path_words[film + 1], path_words[film + 2], 0));
    const uint4 summed = __builtin_bit_cast(uint4, current + emission);
    path_words[film] = summed.x;
    path_words[film + 1] = summed.y;
    path_words[film + 2] = summed.z;
  }

  if (phase_path.x == 1u) {
    const uint4 phase_P = __builtin_bit_cast(uint4, phase_origin);
    const uint4 phase_D = __builtin_bit_cast(uint4, phase_direction);
    const uint4 phase_MIS = __builtin_bit_cast(uint4, mis_origin);
    const uint normal = path + CYCLES_WEBGPU_PATH_MIS_ORIGIN_N;
    const uint min_pdf = path + CYCLES_WEBGPU_PATH_MIN_RAY_PDF;
    path_words[path + CYCLES_WEBGPU_PATH_BOUNCE] = phase_path.w;
    path_words[path + CYCLES_WEBGPU_PATH_FLAGS] = phase_path.y;
    path_words[path + CYCLES_WEBGPU_PATH_SURFACE_RR] = phase_P.w;
    path_words[path + CYCLES_WEBGPU_PATH_MIS_RAY_PDF] = phase_D.w;
    path_words[min_pdf] = __builtin_elementwise_min(path_words[min_pdf], phase_D.w);
    path_words[normal] = phase_MIS.x;
    path_words[normal + 1] = phase_MIS.y;
    path_words[normal + 2] = phase_MIS.z;
    ray_words[ray] = phase_P.x;
    ray_words[ray + 1] = phase_P.y;
    ray_words[ray + 2] = phase_P.z;
    ray_words[ray + 3] = 0u;
    ray_words[ray + 4] = phase_D.x;
    ray_words[ray + 5] = phase_D.y;
    ray_words[ray + 6] = phase_D.z;
    ray_words[ray + 7] = 0x7f7fffffu;
    ray_words[ray + 8] = phase_path.z;
    ray_words[ray + 9] = 0xffffffffu;
    ray_words[ray + 10] = 0xffffffffu;
    ray_words[CYCLES_WEBGPU_RAY_VOLUME_MIS_LENGTH(capacity, lane)] = phase_MIS.w;
    volume_state[volume_bounce] = input_path.y + 1u;
    hit_words[hit + CYCLES_WEBGPU_HIT_SPECIAL] = CYCLES_WEBGPU_VOLUME_SCATTER_HIT;
  }
  else {
    volume_state[lane] = volume_state[CYCLES_WEBGPU_VOLUME_NEXT_BOUNDARY * capacity + lane];
    if (phase_path.x == 2u) {
      hit_words[hit + CYCLES_WEBGPU_HIT_SPECIAL] = CYCLES_WEBGPU_ABSORPTION_TERMINATE;
    }
  }
}
