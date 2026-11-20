/* SPDX-License-Identifier: Apache-2.0 */

#include "../blender-platform/cycles-webgpu/material_layout.h"
#include "../blender-platform/cycles-webgpu/path_state_layout.h"

StructuredBuffer<uint> svm_words : register(t0);
StructuredBuffer<uint> surface_words : register(t1);
StructuredBuffer<uint> ray_words : register(t2);
RWStructuredBuffer<float4> materials : register(u3);
StructuredBuffer<uint> hit_words : register(t4);
StructuredBuffer<uint> path_words : register(t5);
StructuredBuffer<uint> camera_words : register(t6);
StructuredBuffer<uint> metadata_words : register(t7);

#ifdef CYCLES_WEBGPU_COMPACT_PATHS
#include "wavefront_queue.hlsl"
#endif

uint cycles_svm_word(uint index)
{
  return svm_words[index];
}

extern void cycles_svm_material(uint shader, uint primitive, float u, float v,
                                uint lane,
                                out float4 base, out float4 specular, out float4 layers,
                                out float4 glass, out float4 subsurface, out float4 emission);

uint4 cycles_svm_context(uint lane, uint field)
{
  const uint surface = 8 * lane;
  const uint ray = 1 + 12 * lane;
  const uint path = CYCLES_WEBGPU_PATH_WORDS * lane;
  if (field <= CYCLES_WEBGPU_SVM_GEOMETRIC) {
    const uint base = field == CYCLES_WEBGPU_SVM_POSITION ? surface :
                      field == CYCLES_WEBGPU_SVM_NORMAL ? surface + 4 :
                                                          8 * ray_words[0] + surface;
    return uint4(surface_words[base], surface_words[base + 1], surface_words[base + 2], 0);
  }
  if (field == CYCLES_WEBGPU_SVM_DIRECTION) {
    return uint4(ray_words[ray + 4], ray_words[ray + 5], ray_words[ray + 6], 0);
  }
  if (field == CYCLES_WEBGPU_SVM_INTERSECTION) {
    return uint4(hit_words[surface + 4], hit_words[surface + 5], hit_words[surface + 6],
                 surface_words[surface + 3]);
  }
  if (field == CYCLES_WEBGPU_SVM_BOUNCES) {
    return uint4(path_words[path + CYCLES_WEBGPU_PATH_BOUNCE],
                 path_words[path + CYCLES_WEBGPU_PATH_DIFFUSE_BOUNCE],
                 path_words[path + CYCLES_WEBGPU_PATH_GLOSSY_BOUNCE],
                 path_words[path + CYCLES_WEBGPU_PATH_TRANSMISSION_BOUNCE]);
  }
  if (field >= CYCLES_WEBGPU_SVM_WORLD_TO_CAMERA && field < CYCLES_WEBGPU_SVM_WORLD_TO_NDC_END) {
    const uint row = CYCLES_WEBGPU_CAMERA_SVM_OFFSET + 4 * (field - CYCLES_WEBGPU_SVM_WORLD_TO_CAMERA);
    return uint4(camera_words[row], camera_words[row + 1], camera_words[row + 2], camera_words[row + 3]);
  }
  if (field >= CYCLES_WEBGPU_SVM_CAMERA_TO_WORLD && field < CYCLES_WEBGPU_SVM_CAMERA_TO_WORLD_END) {
    const uint row = CYCLES_WEBGPU_CAMERA_TO_WORLD_OFFSET + 4 * (field - CYCLES_WEBGPU_SVM_CAMERA_TO_WORLD);
    return uint4(camera_words[row], camera_words[row + 1], camera_words[row + 2], camera_words[row + 3]);
  }
  if (field >= CYCLES_WEBGPU_SVM_OBJECT_TRANSFORM) {
    const uint object = hit_words[surface + 2];
    if (field < CYCLES_WEBGPU_SVM_OBJECT_INVERSE) {
      const uint row = metadata_words[0] + 13 * object + 1 +
                       4 * (field - CYCLES_WEBGPU_SVM_OBJECT_TRANSFORM);
      return uint4(metadata_words[row], metadata_words[row + 1],
                   metadata_words[row + 2], metadata_words[row + 3]);
    }
    const uint base = metadata_words[CYCLES_WEBGPU_SURFACE_OBJECT_SVM_OFFSET] +
                      CYCLES_WEBGPU_SURFACE_OBJECT_WORDS * object;
    if (field <= CYCLES_WEBGPU_SVM_OBJECT_COLOR) {
      const uint row = base + 4 * (field - CYCLES_WEBGPU_SVM_OBJECT_INVERSE);
      return uint4(metadata_words[row], metadata_words[row + 1],
                   metadata_words[row + 2], metadata_words[row + 3]);
    }
    if (field == CYCLES_WEBGPU_SVM_MATERIAL_INDEX) {
      const uint shader = surface_words[surface + 7] & CYCLES_WEBGPU_SVM_SHADER_ID_MASK;
      const uint shader_pass = metadata_words[CYCLES_WEBGPU_SURFACE_SHADER_PASS_OFFSET] + shader;
      const int pass_id = __builtin_bit_cast(int, metadata_words[shader_pass]);
      return uint4(0, 0, 0, __builtin_bit_cast(uint, float(pass_id)));
    }
    return uint4(0, 0, 0, metadata_words[base + (field == CYCLES_WEBGPU_SVM_OBJECT_INDEX ? 16 : 17)]);
  }
  return uint4(ray_words[ray + 8], path_words[path + CYCLES_WEBGPU_PATH_FLAGS],
               path_words[path + CYCLES_WEBGPU_PATH_TRANSPARENT_BOUNCE], 0);
}

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
  float4 subsurface = float4(0, 0, 0, 0);
  float4 emission = float4(0, 0, 0, 0);
  if (hit_words[base + 7] != 0) {
    // Cycles replaces the entry material with white diffuse scattering at the exit.
    result = float4(1, 1, 1, 1);
  }
  else if (surface_status == 1 || surface_status == 2) {
    cycles_svm_material(surface_words[base + 7], hit_words[base + 1],
                        __builtin_bit_cast(float, hit_words[base + 5]),
                        __builtin_bit_cast(float, hit_words[base + 6]), lane,
                        result, specular, layers, glass, subsurface, emission);
  }
  else if (surface_status != 0) {
    result.w = 2;
  }
  const uint material = CYCLES_WEBGPU_MATERIAL_VECTORS * lane;
  materials[material] = result;
  materials[material + 1] = specular;
  materials[material + 2] = layers;
  materials[material + 3] = glass;
  materials[material + 4] = subsurface;
  materials[material + 5] = emission;
}
