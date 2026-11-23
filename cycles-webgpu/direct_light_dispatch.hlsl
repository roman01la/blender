/* SPDX-License-Identifier: Apache-2.0 */

#include "../blender-platform/cycles-webgpu/lighting_layout.h"

typedef float float3 __attribute__((ext_vector_type(3)));

#ifdef CYCLES_WEBGPU_DEFER_SHADOW
RWStructuredBuffer<uint4> shadow_words : register(u0);
#else
StructuredBuffer<uint> scene_words : register(t0);
#endif
StructuredBuffer<uint> ray_words : register(t1);

#include "wavefront_queue.hlsl"

StructuredBuffer<uint> hit_words : register(t2);
StructuredBuffer<uint> surface_words : register(t3);
StructuredBuffer<uint> light_words : register(t4);
RWStructuredBuffer<float4> samples : register(u5);
#ifdef CYCLES_WEBGPU_WAVEFRONT_LIGHT
#  include "../blender-platform/cycles-webgpu/material_layout.h"
#  include "../blender-platform/cycles-webgpu/path_state_layout.h"
#  ifdef CYCLES_WEBGPU_LIGHT_TREE
#    include "../blender-platform/cycles-webgpu/lighting_layout.h"
#  endif
#  if defined(CYCLES_WEBGPU_LIGHT_TREE) || !defined(CYCLES_WEBGPU_EXCLUDE_HAIR)
RWStructuredBuffer<float4> materials : register(u6);
#  else
StructuredBuffer<float4> materials : register(t6);
#  endif
StructuredBuffer<uint> path_words : register(t7);
#endif

#include "ordered_closure_access.hlsl"

#ifndef CYCLES_WEBGPU_DEFER_SHADOW
uint cycles_scene_word(uint index)
{
  return scene_words[index];
}
#endif

uint cycles_light_word(uint index)
{
  return light_words[index];
}

#ifdef CYCLES_WEBGPU_WAVEFRONT_LIGHT
#  include "wavefront_light.hlsl"
#  define CYCLES_WEBGPU_LIGHT_PASS_BUFFER samples
#  include "light_passes.hlsl"
#  ifdef CYCLES_WEBGPU_LIGHT_TREE
extern float cycles_triangle_hit_mis(uint4 hit, uint4 position, uint4 geometric,
                                    uint4 ray, uint4 origin, uint4 path,
                                    uint4 mis_origin_normal);
#  endif
#else
extern uint cycles_direct_light(uint4 position, uint4 normal, uint4 primary_hit,
                                out float4 result);
#endif

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
  uint lane;
  if (!cycles_queue_slot(id.x, lane)) {
    return;
  }
  const uint base = 8 * lane;
  const uint4 position = uint4(surface_words[base], surface_words[base + 1],
                               surface_words[base + 2], surface_words[base + 3]);
  const uint4 normal = uint4(surface_words[base + 4], surface_words[base + 5],
                             surface_words[base + 6], surface_words[base + 7]);
  const uint4 hit = uint4(hit_words[base], hit_words[base + 1],
                          hit_words[base + 2], hit_words[base + 3]);
  float4 result;
#ifdef CYCLES_WEBGPU_WAVEFRONT_LIGHT
  const uint ray_base = 1 + 12 * lane;
  const uint4 origin = uint4(ray_words[ray_base], ray_words[ray_base + 1],
                             ray_words[ray_base + 2], ray_words[ray_base + 3]);
  const uint4 ray = uint4(ray_words[ray_base + 4], ray_words[ray_base + 5],
                          ray_words[ray_base + 6], ray_words[ray_base + 8]);
  const uint material = CYCLES_WEBGPU_MATERIAL_VECTORS * lane;
  const uint path_base = CYCLES_WEBGPU_PATH_WORDS * lane;
  const uint4 path = uint4(path_words[path_base + CYCLES_WEBGPU_PATH_BOUNCE],
                           path_words[path_base + CYCLES_WEBGPU_PATH_FLAGS],
                           path_words[path_base + CYCLES_WEBGPU_PATH_MIS_RAY_PDF],
                           path_words[path_base + CYCLES_WEBGPU_PATH_MIN_RAY_PDF]);
  const uint pixel_base = path_base + CYCLES_WEBGPU_PATH_PIXEL;
  const uint4 sampling = uint4(ray_words[ray_base + CYCLES_WEBGPU_RAY_RNG_PIXEL],
                               path_words[pixel_base + 2],
                               path_words[path_base + CYCLES_WEBGPU_PATH_SURFACE_RR],
                               path_words[path_base + CYCLES_WEBGPU_PATH_TRANSPARENT_BOUNCE] &
                                   CYCLES_WEBGPU_TRANSPARENT_BOUNCE_MASK);
  const uint normal_base = 8 * ray_words[0] + base;
  const uint4 geometric = uint4(surface_words[normal_base], surface_words[normal_base + 1],
                                 surface_words[normal_base + 2], surface_words[normal_base + 3]);
  const uint4 spec_normal = uint4(surface_words[normal_base + 4], surface_words[normal_base + 5],
                                  surface_words[normal_base + 6], 0);
  const uint shadow_base = normal_base + 8 * ray_words[0];
  const uint4 shadow_along = uint4(surface_words[shadow_base], surface_words[shadow_base + 1],
                                   surface_words[shadow_base + 2], surface_words[shadow_base + 3]);
  const uint4 shadow_opposite = uint4(surface_words[shadow_base + 4], surface_words[shadow_base + 5],
                                      surface_words[shadow_base + 6], 0);
  const uint4 closest = uint4(hit.xyz, hit_words[base + 4]);
  float4 area_result;
#ifdef CYCLES_WEBGPU_DEFER_SHADOW
  uint4 shadow_origin, shadow_direction, shadow_options;
#  ifdef CYCLES_WEBGPU_LIGHT_TREE
  uint4 sampled_emission;
#  endif
#endif
#ifdef CYCLES_WEBGPU_LIGHT_TREE
  const uint mis_normal_word = path_base + CYCLES_WEBGPU_PATH_MIS_ORIGIN_N;
  uint volume_segment_length = 0u;
#  ifdef CYCLES_WEBGPU_VOLUME_FORWARD
  if ((ray.w & CYCLES_WEBGPU_VOLUME_SCATTER_VISIBILITY) != 0u) {
    const bool differentials = (cycles_light_word(CYCLES_WEBGPU_LIGHT_EXTENDED) &
                                CYCLES_WEBGPU_LIGHT_VOLUME_DIFFERENTIALS) != 0u;
    volume_segment_length = ray_words[CYCLES_WEBGPU_RAY_VOLUME_MIS_LENGTH(
        ray_words[0], lane, differentials)];
  }
#  endif
  const uint4 mis_origin_normal = uint4(path_words[mis_normal_word],
                                       path_words[mis_normal_word + 1],
                                       path_words[mis_normal_word + 2], volume_segment_length);
  const uint weight = path_base + CYCLES_WEBGPU_PATH_THROUGHPUT;
  const float4 throughput = __builtin_bit_cast(
      float4, uint4(path_words[weight], path_words[weight + 1], path_words[weight + 2], 0));
  const uint emission_slot = material + CYCLES_WEBGPU_MATERIAL_EMISSION_SLOT;
  const float4 emission = materials[emission_slot];
  if (closest.x != 0 && (position.w == 1 || position.w == 2) &&
      (emission.x != 0.0f || emission.y != 0.0f || emission.z != 0.0f)) {
    const float mis = cycles_triangle_hit_mis(closest, position, geometric, ray, origin, path,
                                             mis_origin_normal);
    materials[emission_slot] = float4(emission.xyz * mis, emission.w);
  }
#endif
  uint hair_lcg;
  float4 diffuse_ratio, glossy_ratio;
  const uint capacity = ray_words[0];
  const bool save_ratios = path.x == 0u && cycles_light_pass_enabled(capacity);
  const float4 hair = materials[material + 4];
  cycles_wavefront_light(position, normal, spec_normal, geometric, shadow_along, shadow_opposite,
                         closest, ray, origin, path, sampling,
                         path_words[path_base + CYCLES_WEBGPU_PATH_EXTRA_RNG_STEPS],
                         materials[material], materials[material + 1], materials[material + 2],
                         materials[material + 3], hair,
                         materials[material + CYCLES_WEBGPU_MATERIAL_NORMAL_SLOT],
                         result, area_result, hair_lcg, uint(save_ratios), diffuse_ratio,
                         glossy_ratio
#ifdef CYCLES_WEBGPU_LIGHT_TREE
                         , mis_origin_normal, throughput
#endif
#ifdef CYCLES_WEBGPU_DEFER_SHADOW
                         , shadow_origin, shadow_direction, shadow_options
#  ifdef CYCLES_WEBGPU_LIGHT_TREE
                         , sampled_emission
#  endif
#endif
                         );
  if (save_ratios) {
    const uint record = cycles_light_pass_record(capacity, lane);
    samples[record + 2u] = diffuse_ratio;
    samples[record + 3u] = glossy_ratio;
  }
#ifdef CYCLES_WEBGPU_DEFER_SHADOW
  const uint shadow = 4 * lane;
  shadow_words[shadow] = shadow_origin;
  shadow_words[shadow + 1] = shadow_direction;
  shadow_words[shadow + 2] = shadow_options;
  shadow_words[shadow + 3] = geometric;
#  ifdef CYCLES_WEBGPU_LIGHT_TREE
  const uint descriptor = light_words[CYCLES_WEBGPU_LIGHT_TREE_OFFSET];
  if (light_words[descriptor + CYCLES_WEBGPU_TREE_TEXTURED_EMITTERS] != 0) {
    /* The optional tail stores triangle ID, shader, and barycentrics for sampled SVM.
     * Clear the primitive sentinel even when a path did not sample textured emission. */
    shadow_words[4 * ray_words[0] + lane] = sampled_emission;
  }
#  endif
#endif
#  ifndef CYCLES_WEBGPU_EXCLUDE_HAIR
  if (materials[material].w == CYCLES_WEBGPU_MATERIAL_HAIR_HUANG) {
    materials[material + 4] = float4(hair.xy, float(hair_lcg & 0xffffu),
                                     float(hair_lcg >> 16u));
  }
#  endif
  samples[ray_words[0] + lane] = area_result;
#else
  cycles_direct_light(position, normal, hit, result);
#endif
  samples[lane] = result;
}
