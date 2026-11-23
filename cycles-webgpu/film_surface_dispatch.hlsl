/* SPDX-License-Identifier: Apache-2.0 */

#include "../blender-platform/cycles-webgpu/material_layout.h"
#include "../blender-platform/cycles-webgpu/path_state_layout.h"
#include "../blender-platform/cycles-webgpu/volume_layout.h"

typedef float float3 __attribute__((ext_vector_type(3)));

StructuredBuffer<uint> ray_words : register(t0);
StructuredBuffer<uint> surface_words : register(t1);
StructuredBuffer<float4> materials : register(t2);
StructuredBuffer<uint> camera_sampling : register(t3);
StructuredBuffer<float4> radiance : register(t4);
RWStructuredBuffer<uint> path_words : register(u5);
RWStructuredBuffer<float4> pass_values : register(u6);
StructuredBuffer<uint> pass_offsets : register(t7);

#include "ordered_closure_access.hlsl"
#include "wavefront_queue.hlsl"
#include "wavefront_radiance.hlsl"

float3 surface_transparency(float4 material, float4 specular, float4 glass, float4 auxiliary)
{
  if (material.w == CYCLES_WEBGPU_MATERIAL_TRANSPARENT) {
    return material.xyz;
  }
  if (material.w == CYCLES_WEBGPU_MATERIAL_MIXED_TRANSPARENT) {
    return specular.xyz;
  }
  if (material.w == CYCLES_WEBGPU_MATERIAL_GLOSSY_TRANSPARENT ||
      CYCLES_WEBGPU_MATERIAL_HAS_ORDERED_CLOSURES(material.w)) {
    return auxiliary.xyz;
  }
  const bool ggx = material.w == CYCLES_WEBGPU_MATERIAL_GGX ||
                   material.w == CYCLES_WEBGPU_MATERIAL_MULTISCATTER_GGX;
  if (ggx && CYCLES_WEBGPU_GGX_CAN_BE_TRANSPARENT(glass) && auxiliary.w != 0.0f) {
    return auxiliary.xyz;
  }
  return float3(0, 0, 0);
}

uint cycles_camera_sample_word(uint index)
{
  return camera_sampling[index];
}

extern void cycles_surface_film_passes(float4 material, float4 tint_alpha, float4 weights,
                                      float4 glass, float4 auxiliary, float4 bump,
                                      float4 surface_normal, float4 specular_normal,
                                      float4 geometric_normal, float4 wi, uint color_mask,
                                      out float4 normal, out float4 diffuse, out float4 glossy,
                                      out float4 transmission, out float4 ao_normal);

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
  uint lane;
  if (!cycles_queue_slot(id.x, lane)) {
    return;
  }

  const uint output = 6 * lane;
  pass_values[output] = float4(0);
  pass_values[output + 1] = float4(0);
  pass_values[output + 2] = float4(0);
  pass_values[output + 3] = float4(0);
  pass_values[output + 4] = float4(0);
  pass_values[output + 5] = float4(0);
  const uint path = lane * CYCLES_WEBGPU_PATH_WORDS;
  const uint flags = path_words[path + CYCLES_WEBGPU_PATH_FLAGS];
  const float surface_rr = __builtin_bit_cast(float, path_words[path + CYCLES_WEBGPU_PATH_SURFACE_RR]);
  if ((flags & CYCLES_WEBGPU_PATH_TRANSPARENT_BACKGROUND) == 0u ||
      (flags & (CYCLES_WEBGPU_PATH_SUBSURFACE | CYCLES_WEBGPU_PATH_TERMINATE_ON_NEXT_SURFACE |
                CYCLES_WEBGPU_PATH_TERMINATE_AFTER_TRANSPARENT)) != 0u ||
      path_words[path + CYCLES_WEBGPU_PATH_QUEUED_KERNEL] == CYCLES_WEBGPU_PATH_TERMINATED ||
      !(surface_rr > 0.0f))
  {
    return;
  }

  const uint base = 8 * lane;
  const uint status = surface_words[base + 3];
  const uint material_base = CYCLES_WEBGPU_MATERIAL_VECTORS * lane;
  const float4 material = materials[material_base];
  const float4 specular = materials[material_base + 1];
  const float4 glass = materials[material_base + 3];
  const float4 auxiliary = materials[material_base + 4];
  const uint forward_hits = cycles_forward_light_hits(radiance[ray_words[0] + lane]);
  const uint transparent_bounces = path_words[path + CYCLES_WEBGPU_PATH_TRANSPARENT_BOUNCE] &
                                   CYCLES_WEBGPU_TRANSPARENT_BOUNCE_MASK;
  if ((status != 1u && status != 2u) || material.w == CYCLES_WEBGPU_MATERIAL_HOLDOUT ||
      material.w == CYCLES_WEBGPU_MATERIAL_UNSHADED ||
      (forward_hits != 0u && !cycles_trace_past_area_limit(transparent_bounces + forward_hits)))
  {
    return;
  }

  const uint tail = 8 * ray_words[0] + base;
  const uint ray = 1 + 12 * lane;
  float4 normal, diffuse, glossy, transmission, ao_normal;
  const uint color_mask = uint(pass_offsets[1] != 0xffffffffu) |
                          uint(pass_offsets[2] != 0xffffffffu) << 1u |
                          uint(pass_offsets[3] != 0xffffffffu) << 2u;
  float4 surface_normal = __builtin_bit_cast(
      float4, uint4(surface_words[base + 4], surface_words[base + 5], surface_words[base + 6], 0));
  surface_normal.w = float(status == 2u);
  cycles_surface_film_passes(
      material, specular, materials[material_base + 2], glass, auxiliary,
      materials[material_base + CYCLES_WEBGPU_MATERIAL_NORMAL_SLOT],
      surface_normal,
      __builtin_bit_cast(float4, uint4(surface_words[tail + 4], surface_words[tail + 5],
                                      surface_words[tail + 6], 0)),
      __builtin_bit_cast(float4, uint4(surface_words[tail], surface_words[tail + 1],
                                      surface_words[tail + 2], 0)),
      -__builtin_bit_cast(float4, uint4(ray_words[ray + 4], ray_words[ray + 5], ray_words[ray + 6], 0)),
      color_mask, normal, diffuse, glossy, transmission, ao_normal);

  const uint weight = path + CYCLES_WEBGPU_PATH_THROUGHPUT;
  const float4 throughput = __builtin_bit_cast(
      float4, uint4(path_words[weight], path_words[weight + 1], path_words[weight + 2], 0)) *
      surface_rr;
  const float3 alpha = __builtin_elementwise_min(
      float3(1, 1, 1),
      __builtin_elementwise_max(float3(0, 0, 0),
                                  float3(1, 1, 1) -
                                      surface_transparency(material, specular, glass, auxiliary)));
  const float average_alpha = (alpha.x + alpha.y + alpha.z) * (1.0f / 3.0f);
  if ((flags & CYCLES_WEBGPU_PATH_SINGLE_PASS_DONE) == 0u &&
      (pass_offsets[7] == 0u || average_alpha >= __builtin_bit_cast(float, pass_offsets[6])))
  {
    pass_values[output] = normal;
    path_words[path + CYCLES_WEBGPU_PATH_FLAGS] = flags | CYCLES_WEBGPU_PATH_SINGLE_PASS_DONE;
  }
  pass_values[output + 1] = diffuse * throughput;
  pass_values[output + 2] = glossy * throughput;
  pass_values[output + 3] = transmission * throughput;
  const float distance = __builtin_bit_cast(float, pass_offsets[5]);
  ao_normal.w = pass_offsets[7] == 0u ? distance : -distance;
  pass_values[output + 4] = ao_normal;
  pass_values[output + 5] = float4(alpha, float(forward_hits));
}
