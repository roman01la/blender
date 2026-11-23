/* SPDX-License-Identifier: Apache-2.0 */

#include "../blender-platform/cycles-webgpu/path_state_layout.h"

typedef float float3 __attribute__((ext_vector_type(3)));

#ifdef CYCLES_WEBGPU_FILM_AO_PREPARE
#  include "../blender-platform/cycles-webgpu/material_layout.h"
RWStructuredBuffer<uint4> shadow_words : register(u0);
#else
StructuredBuffer<uint> scene_words : register(t0);
#endif
StructuredBuffer<uint> ray_words : register(t1);
StructuredBuffer<uint> hit_words : register(t2);
StructuredBuffer<uint> surface_words : register(t3);
StructuredBuffer<uint> path_words : register(t4);
StructuredBuffer<uint> camera_sampling : register(t5);
RWStructuredBuffer<float4> pass_values : register(u6);
#ifdef CYCLES_WEBGPU_FILM_AO_PREPARE
StructuredBuffer<float4> materials : register(t7);
#else
StructuredBuffer<uint> metadata_words : register(t7);
#endif

#include "wavefront_queue.hlsl"
#include "wavefront_radiance.hlsl"

#ifndef CYCLES_WEBGPU_FILM_AO_PREPARE
uint cycles_scene_word(uint index)
{
  return scene_words[index];
}

uint cycles_surface_word(uint index)
{
  return metadata_words[index];
}

#endif

uint cycles_camera_sample_word(uint index)
{
  return camera_sampling[index];
}

extern float4 cycles_film_ao_sample(uint rng_pixel, uint sample, uint bounce, float4 normal);
#ifdef CYCLES_WEBGPU_FILM_AO_PREPARE
extern void cycles_film_ao_ray(uint4 position, uint4 normal, uint4 geometric,
                               uint4 shadow_along, uint4 shadow_opposite, uint4 hit,
                               float4 direction, out uint4 origin, out uint4 options);
#else
extern float4 cycles_film_ao_visibility(uint4 position, uint4 normal, uint4 geometric,
                                        uint4 shadow_along, uint4 shadow_opposite, uint4 hit,
                                        float4 direction, uint transparent, uint transparent_bounce);

#endif

uint4 surface_record(uint index)
{
  return uint4(surface_words[index], surface_words[index + 1],
               surface_words[index + 2], surface_words[index + 3]);
}

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
  uint lane;
  if (!cycles_queue_slot(id.x, lane)) {
    return;
  }
  const uint output = 6 * lane + 4;
  const float4 ao_normal = pass_values[output];
  const float4 opacity = pass_values[output + 1];
  pass_values[output] = float4(0);
#ifdef CYCLES_WEBGPU_FILM_AO_PREPARE
  const uint shadow_record = 4 * lane;
  shadow_words[shadow_record + 2] = uint4(0);
  if (ao_normal.w == 0.0f || (opacity.x == 0.0f && opacity.y == 0.0f && opacity.z == 0.0f)) {
    return;
  }
#else
  if (ao_normal.w == 0.0f) {
    return;
  }
#endif

  const uint path = lane * CYCLES_WEBGPU_PATH_WORDS;
  const uint ray = 1 + 12 * lane;
  float4 direction = cycles_film_ao_sample(
      ray_words[ray + CYCLES_WEBGPU_RAY_RNG_PIXEL],
      path_words[path + CYCLES_WEBGPU_PATH_PIXEL + 2],
      path_words[path + CYCLES_WEBGPU_PATH_BOUNCE] +
          path_words[path + CYCLES_WEBGPU_PATH_EXTRA_RNG_STEPS],
      ao_normal);
  direction.w = __builtin_elementwise_abs(ao_normal.w);

  const uint base = 8 * lane;
  const uint tail = 8 * ray_words[0] + base;
  const uint shadow = tail + 8 * ray_words[0];
  const uint4 hit = uint4(hit_words[base], hit_words[base + 1],
                          hit_words[base + 2], hit_words[base + 3]);
#ifdef CYCLES_WEBGPU_FILM_AO_PREPARE
  const uint material = CYCLES_WEBGPU_MATERIAL_VECTORS * lane;
  const float4 bump = materials[material + CYCLES_WEBGPU_MATERIAL_NORMAL_SLOT];
  uint4 normal = surface_record(base + 4);
  /* The AO lobe normal sets the sample direction. Like a direct shadow ray,
   * the offset uses ShaderData::N before applying the closure-local bump. */
  if (bump.w >= CYCLES_WEBGPU_NORMAL_SHADER) {
    const float4 shader_normal = bump.w >= CYCLES_WEBGPU_NORMAL_BOTH ?
                                     materials[material + 4] : bump;
    normal = __builtin_bit_cast(uint4, shader_normal);
  }
  const uint4 geometric = surface_record(tail);
  uint4 origin, options;
  cycles_film_ao_ray(surface_record(base), normal, geometric, surface_record(shadow),
                     surface_record(shadow + 4), hit, direction, origin, options);
  shadow_words[shadow_record] = origin;
  shadow_words[shadow_record + 1] = __builtin_bit_cast(uint4, direction);
  options.x |= CYCLES_WEBGPU_SHADOW_AO_PASS;
  shadow_words[shadow_record + 2] = options;
  shadow_words[shadow_record + 3] = geometric;
#else
  const uint transparent_bounce = (path_words[path + CYCLES_WEBGPU_PATH_TRANSPARENT_BOUNCE] &
                                   CYCLES_WEBGPU_TRANSPARENT_BOUNCE_MASK) + uint(opacity.w);
  const float4 visibility = cycles_film_ao_visibility(
      surface_record(base), surface_record(base + 4), surface_record(tail),
      surface_record(shadow), surface_record(shadow + 4), hit, direction,
      uint(ao_normal.w < 0.0f), transparent_bounce);
#endif
  const uint weight = path + CYCLES_WEBGPU_PATH_THROUGHPUT;
  const float3 throughput = __builtin_bit_cast(
      float3, uint3(path_words[weight], path_words[weight + 1], path_words[weight + 2]));
  const float surface_rr = __builtin_bit_cast(float, path_words[path + CYCLES_WEBGPU_PATH_SURFACE_RR]);
#ifdef CYCLES_WEBGPU_FILM_AO_PREPARE
  pass_values[output] = float4(throughput * surface_rr * opacity.xyz, cycles_path_direct_clamp());
#else
  pass_values[output] = float4(
      clamp_light(throughput * surface_rr * opacity.xyz * visibility.xyz, cycles_path_direct_clamp()),
      0.0f);
#endif
}
