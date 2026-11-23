/* SPDX-License-Identifier: Apache-2.0 */

#include "../blender-platform/cycles-webgpu/path_state_layout.h"

StructuredBuffer<uint> scene_words : register(t0);
StructuredBuffer<uint> ray_words : register(t1);
StructuredBuffer<uint4> shadow_words : register(t2);
StructuredBuffer<float4> radiance : register(t3);
RWStructuredBuffer<uint> path_words : register(u4);

typedef float float3 __attribute__((ext_vector_type(3)));

#include "wavefront_radiance.hlsl"

uint cycles_scene_word(uint index)
{
  return scene_words[index];
}

extern void cycles_bvh2_trace(uint4 origin, uint4 direction, uint4 options,
                              out uint4 hit, out uint4 coordinates, uint4 geometric);

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
  const uint capacity = ray_words[0];
  // Keep this offset in sync with the shared compact shadow queue.
  const uint queue = 2u + 17u * capacity;
  if (id.x >= ray_words[queue]) {
    return;
  }
  const uint lane = ray_words[queue + 1u + id.x];
  const uint shadow = 4u * lane;
  uint4 hit, coordinates;
  cycles_bvh2_trace(shadow_words[shadow], shadow_words[shadow + 1u],
                    shadow_words[shadow + 2u], hit, coordinates, shadow_words[shadow + 3u]);
  if (hit.x == 0u) {
    const float4 direct = radiance[lane];
    const float3 nee = clamp_light(radiance[capacity + lane].xyz * direct.xyz, direct.w);
    const uint sum = lane * CYCLES_WEBGPU_PATH_WORDS + CYCLES_WEBGPU_PATH_RADIANCE;
    const uint3 rgb = uint3(path_words[sum], path_words[sum + 1u], path_words[sum + 2u]);
    const uint3 result = __builtin_bit_cast(uint3, __builtin_bit_cast(float3, rgb) + nee);
    path_words[sum] = result.x;
    path_words[sum + 1u] = result.y;
    path_words[sum + 2u] = result.z;
  }
}
