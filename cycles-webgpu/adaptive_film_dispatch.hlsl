/* SPDX-License-Identifier: Apache-2.0 */

#include "../blender-platform/cycles-webgpu/path_state_layout.h"

StructuredBuffer<uint> completed_paths : register(t0);
StructuredBuffer<uint> ray_words : register(t1);
StructuredBuffer<uint> film_parameters : register(t2);
RWStructuredBuffer<float4> deltas : register(u3);

#include "wavefront_queue.hlsl"

extern void cycles_adaptive_film_delta(uint sample, uint pattern, float4 rgba,
                                      out float4 combined, out float4 adaptive);

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
  uint lane;
  if (!cycles_queue_slot(id.x, lane) || film_parameters[9] == 0xffffffffu) {
    return;
  }

  const uint path = lane * CYCLES_WEBGPU_PATH_WORDS;
  if (completed_paths[path + CYCLES_WEBGPU_PATH_QUEUED_KERNEL] != CYCLES_WEBGPU_PATH_TERMINATED) {
    return;
  }

  const uint sample = completed_paths[path + CYCLES_WEBGPU_PATH_PIXEL + 2u];
  const uint sum = path + CYCLES_WEBGPU_PATH_RADIANCE;
  const uint4 bits = uint4(completed_paths[sum], completed_paths[sum + 1u],
                          completed_paths[sum + 2u], completed_paths[sum + 3u]);
  float4 combined, adaptive;
  cycles_adaptive_film_delta(sample, film_parameters[10], __builtin_bit_cast(float4, bits),
                             combined, adaptive);
  deltas[lane] = combined;
  deltas[ray_words[0] + lane] = adaptive;
}
