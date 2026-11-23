/* SPDX-License-Identifier: Apache-2.0 */

#include "../blender-platform/cycles-webgpu/material_layout.h"
#include "../blender-platform/cycles-webgpu/path_state_layout.h"

typedef float float3 __attribute__((ext_vector_type(3)));

StructuredBuffer<uint> scene_words : register(t0);
StructuredBuffer<uint> surface_words : register(t1);
StructuredBuffer<uint> camera_sampling : register(t2);
StructuredBuffer<uint> points : register(t3);
StructuredBuffer<float4> materials : register(t4);
RWStructuredBuffer<uint> hit_words : register(u5);
RWStructuredBuffer<uint> ray_words : register(u6);
RWStructuredBuffer<uint> path_words : register(u7);

#include "wavefront_queue.hlsl"

uint cycles_scene_word(uint index)
{
  return scene_words[index];
}

uint cycles_surface_word(uint index)
{
  return surface_words[index];
}

uint cycles_camera_sample_word(uint index)
{
  return camera_sampling[index];
}

extern void cycles_subsurface_walk(float4 entry, float4 direction, float4 normal,
                                   float4 albedo, float4 radius, float4 weight,
                                   uint4 rng, uint4 surface, out uint4 hit, out float4 coordinates,
                                   out float4 exit_direction, out float4 throughput,
                                   float4 geometric, uint backfacing, float4 shading_normal,
                                   float4 closure_normal, out float4 exit_origin, out uint exit_flags);

float4 load_point(uint offset, float w)
{
  return __builtin_bit_cast(float4, uint4(points[offset], points[offset + 1], points[offset + 2],
                                        __builtin_bit_cast(uint, w)));
}

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
  uint lane;
  if (!cycles_queue_slot(id.x, lane)) {
    return;
  }

  const uint ray = 1 + 12 * lane;
  const uint surface = 8 * lane;
  const uint normals = 8 * ray_words[0] + surface;
  const uint path = lane * CYCLES_WEBGPU_PATH_WORDS;
  const uint weight = path + CYCLES_WEBGPU_PATH_THROUGHPUT;
  const uint steps = path + CYCLES_WEBGPU_PATH_EXTRA_RNG_STEPS;
  const uint material = CYCLES_WEBGPU_MATERIAL_VECTORS * lane;
  const float4 base = materials[material];
  const float4 specular = materials[material + 1];
  const float4 glass = materials[material + 3];
  const float4 radius = materials[material + 4];
  const float4 closure_normal = materials[material + CYCLES_WEBGPU_MATERIAL_NORMAL_SLOT];
  const float4 shading_normal = closure_normal.w >= CYCLES_WEBGPU_NORMAL_SHADER ?
                                    closure_normal : load_point(surface + 4, 0.0f);
  const float4 input_throughput = __builtin_bit_cast(
      float4, uint4(path_words[weight], path_words[weight + 1], path_words[weight + 2],
                   __builtin_bit_cast(uint, specular.w)));
  const float4 incident = __builtin_bit_cast(
      float4, uint4(ray_words[ray + 4], ray_words[ray + 5], ray_words[ray + 6], 0));

  const uint bounce = path_words[path + CYCLES_WEBGPU_PATH_BOUNCE];
  // Cycles advances once for entry, and again for the diffuse exit.
  const uint rng_offset = (bounce + path_words[steps] + 2) * CYCLES_WEBGPU_PRNG_BOUNCE_NUM;
  const uint4 rng = uint4(ray_words[ray + CYCLES_WEBGPU_RAY_RNG_PIXEL], rng_offset,
                          path_words[path + CYCLES_WEBGPU_PATH_PIXEL + 2], 0);
  const uint4 entry_hit = uint4(hit_words[surface + 1], hit_words[surface + 2], 0, 0);
  uint4 hit;
  float4 coordinates, direction, throughput, origin;
  uint exit_flags;
  cycles_subsurface_walk(load_point(surface, 0.0f), incident,
                         load_point(normals + 4, glass.z),
                         float4(__builtin_elementwise_min(base.xyz, float3(1, 1, 1)), glass.y),
                         radius, input_throughput, rng, entry_hit, hit, coordinates, direction,
                         throughput, load_point(normals, 0.0f), points[surface + 3] == 2u,
                         shading_normal, closure_normal, origin, exit_flags);

  const uint queued = path + CYCLES_WEBGPU_PATH_QUEUED_KERNEL;
  if (hit.x == 0u) {
    path_words[queued] = CYCLES_WEBGPU_PATH_TERMINATED;
    return;
  }

  const uint4 weight_bits = __builtin_bit_cast(uint4, throughput);
  path_words[weight] = weight_bits.x;
  path_words[weight + 1] = weight_bits.y;
  path_words[weight + 2] = weight_bits.z;
  path_words[steps] += 2u;
  path_words[path + CYCLES_WEBGPU_PATH_FLAGS] |= exit_flags;
  path_words[path + CYCLES_WEBGPU_PATH_SURFACE_RR] = __builtin_bit_cast(uint, 1.0f);
  hit_words[surface] = hit.x;
  hit_words[surface + 1] = hit.y;
  hit_words[surface + 2] = hit.z;
  hit_words[surface + 3] = hit.w;
  const uint4 coords = __builtin_bit_cast(uint4, coordinates);
  hit_words[surface + 4] = coords.x;
  hit_words[surface + 5] = coords.y;
  hit_words[surface + 6] = coords.z;
  hit_words[surface + CYCLES_WEBGPU_HIT_SPECIAL] = CYCLES_WEBGPU_HIT_SUBSURFACE_EXIT;
  const uint4 ray_origin = __builtin_bit_cast(uint4, origin);
  const uint4 ray_direction = __builtin_bit_cast(uint4, direction);
  ray_words[ray] = ray_origin.x;
  ray_words[ray + 1] = ray_origin.y;
  ray_words[ray + 2] = ray_origin.z;
  ray_words[ray + 3] = ray_origin.w;
  ray_words[ray + 4] = ray_direction.x;
  ray_words[ray + 5] = ray_direction.y;
  ray_words[ray + 6] = ray_direction.z;
  ray_words[ray + 7] = ray_direction.w;
  ray_words[ray + 8] &= ~CYCLES_WEBGPU_CAMERA_VISIBILITY;
  path_words[queued] = CYCLES_WEBGPU_PATH_SHADE_SURFACE;
}
