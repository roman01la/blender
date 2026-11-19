/* SPDX-License-Identifier: Apache-2.0 */

#include "../blender-platform/cycles-webgpu/material_layout.h"

StructuredBuffer<uint> svm_words : register(t0);
StructuredBuffer<uint> surface_words : register(t1);
StructuredBuffer<uint> ray_words : register(t2);
RWStructuredBuffer<float4> materials : register(u3);
StructuredBuffer<uint> hit_words : register(t4);

#ifdef CYCLES_WEBGPU_COMPACT_PATHS
#include "wavefront_queue.hlsl"
#endif

uint cycles_svm_word(uint index)
{
  return svm_words[index];
}

extern void cycles_svm_material(uint shader, uint primitive, float u, float v,
                                out float4 base, out float4 specular, out float4 layers,
                                out float4 glass);

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
#ifdef CYCLES_WEBGPU_COMPACT_PATHS
  uint lane;
  if (!cycles_queue_slot(id.x, lane)) {
    return;
  }
#else
  const uint lane = id.x;
  if (lane >= ray_words[0]) {
    return;
  }
#endif

  const uint base = 8 * lane;
  const uint surface_status = surface_words[base + 3];
  float4 result = float4(0, 0, 0, 0);
  float4 specular = float4(0, 0, 0, 0);
  float4 layers = float4(0, 0, 0, 0);
  float4 glass = float4(0, 0, 0, 0);
  if (surface_status == 1 || surface_status == 2) {
    cycles_svm_material(surface_words[base + 7], hit_words[base + 1],
                        __builtin_bit_cast(float, hit_words[base + 5]),
                        __builtin_bit_cast(float, hit_words[base + 6]), result, specular, layers,
                        glass);
  }
  else if (surface_status != 0) {
    result.w = 2;
  }
  const uint material = CYCLES_WEBGPU_MATERIAL_VECTORS * lane;
  materials[material] = result;
  materials[material + 1] = specular;
  materials[material + 2] = layers;
  materials[material + 3] = glass;
}
