/* SPDX-License-Identifier: Apache-2.0 */

#include "../blender-platform/cycles-webgpu/material_layout.h"
#include "../blender-platform/cycles-webgpu/path_state_layout.h"

StructuredBuffer<uint> svm_words : register(t0);
StructuredBuffer<uint4> shadow_words : register(t1);
StructuredBuffer<uint> ray_words : register(t2);
RWStructuredBuffer<float4> samples : register(u3);
StructuredBuffer<uint> path_words : register(t4);
StructuredBuffer<uint> camera_words : register(t5);
StructuredBuffer<uint> metadata_words : register(t6);
StructuredBuffer<uint> scene_words : register(t7);

#include "svm_context.hlsl"
#include "wavefront_queue.hlsl"

uint cycles_svm_word(uint index)
{
  return svm_words[index];
}

uint cycles_scene_word(uint index)
{
  return scene_words[index];
}

uint cycles_surface_word(uint index)
{
  return metadata_words[index];
}

uint4 cycles_svm_context(uint lane, uint field)
{
  const uint path = CYCLES_WEBGPU_PATH_WORDS * lane;
  if (field == CYCLES_WEBGPU_SVM_BOUNCES) {
    return uint4(path_words[path + CYCLES_WEBGPU_PATH_BOUNCE],
                 path_words[path + CYCLES_WEBGPU_PATH_DIFFUSE_BOUNCE],
                 path_words[path + CYCLES_WEBGPU_PATH_GLOSSY_BOUNCE],
                 path_words[path + CYCLES_WEBGPU_PATH_TRANSMISSION_BOUNCE]);
  }
  if (field == CYCLES_WEBGPU_SVM_PATH) {
    const uint shader_flags = cycles_svm_scene_context(
        field, shadow_words[4 * lane + 2].w, shadow_words[4 * ray_words[0] + lane].y).w;
    return uint4(0, 0, path_words[path + CYCLES_WEBGPU_PATH_TRANSPARENT_BOUNCE], shader_flags);
  }
  return cycles_svm_scene_context(field, shadow_words[4 * lane + 2].w,
                                 shadow_words[4 * ray_words[0] + lane].y);
}

extern void cycles_triangle_sample_surface(uint4 sample, uint object, uint4 direction,
                                           out uint4 position, out uint4 normal,
                                           out uint4 geometric);
extern float4 cycles_svm_sampled_emission(uint4 sample, uint4 direction, uint4 position,
                                          uint4 normal, uint4 geometric, uint lane);

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
  uint lane;
  if (!cycles_queue_slot(id.x, lane)) {
    return;
  }
  const uint4 sample = shadow_words[4 * ray_words[0] + lane];
  if (sample.x == 0) {
    return;
  }
  const uint shadow = 4 * lane;
  const uint4 direction = shadow_words[shadow + 1];
  const uint object = shadow_words[shadow + 2].w;
  uint4 position, normal, geometric;
  cycles_triangle_sample_surface(sample, object, direction, position, normal, geometric);
  const float4 emission = cycles_svm_sampled_emission(sample, direction, position, normal,
                                                      geometric, lane);
  const float4 direct = samples[lane];
  samples[lane] = float4(direct.xyz * emission.xyz, direct.w);
}
