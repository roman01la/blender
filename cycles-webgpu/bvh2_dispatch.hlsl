/* SPDX-License-Identifier: Apache-2.0 */

StructuredBuffer<uint> scene_words : register(t0);
#ifdef CYCLES_WEBGPU_OFFSET_CLOSEST
RWStructuredBuffer<uint> ray_words : register(u1);
StructuredBuffer<uint> previous_surface : register(t3);
#else
StructuredBuffer<uint> ray_words : register(t1);
#endif

#include "wavefront_queue.hlsl"

#ifdef CYCLES_WEBGPU_INTERSECT_SHADOW
StructuredBuffer<uint4> shadow_words : register(t2);
RWStructuredBuffer<float4> radiance : register(u3);
#  ifdef CYCLES_WEBGPU_TRANSPARENT_SHADOW
#    include "../blender-platform/cycles-webgpu/material_layout.h"
#    include "../blender-platform/cycles-webgpu/path_state_layout.h"
typedef float float3 __attribute__((ext_vector_type(3)));
StructuredBuffer<uint> metadata_words : register(t4);
StructuredBuffer<uint> path_words : register(t5);

#    include "wavefront_radiance.hlsl"

#    ifdef CYCLES_WEBGPU_SHADOW_SVM
StructuredBuffer<uint> svm_words : register(t6);
StructuredBuffer<uint> camera_words : register(t7);

#      include "svm_context.hlsl"

uint cycles_svm_word(uint index)
{
  return svm_words[index];
}

uint4 cycles_svm_shadow_context(uint field, uint object, uint shader)
{
  return cycles_svm_scene_context(field, object, shader);
}
#    endif

uint cycles_surface_word(uint index)
{
  return metadata_words[index];
}

extern float4 cycles_bvh2_transparent_shadow(uint4 origin, uint4 direction, uint4 options,
                                             uint4 geometric, uint transparent_bounce);
#  endif
#else
RWStructuredBuffer<uint> hit_words : register(u2);
#endif

uint cycles_scene_word(uint index)
{
  return scene_words[index];
}

#ifdef CYCLES_WEBGPU_OFFSET_CLOSEST
uint cycles_surface_word(uint index)
{
  return previous_surface[index];
}

extern void cycles_bvh2_trace(uint4 origin, uint4 direction, uint4 options,
                              out uint4 hit, out uint4 coordinates, uint previous_geometric,
                              out uint4 adjusted_origin);
#elif defined(CYCLES_WEBGPU_INTERSECT_SHADOW)
extern void cycles_bvh2_trace(uint4 origin, uint4 direction, uint4 options,
                              out uint4 hit, out uint4 coordinates, uint4 geometric);
#else
extern void cycles_bvh2_trace(uint4 origin, uint4 direction, uint4 options,
                              out uint4 hit, out uint4 coordinates);
#endif

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
  uint lane;
  if (!cycles_queue_slot(id.x, lane)) {
    return;
  }
#ifdef CYCLES_WEBGPU_INTERSECT_SHADOW
  const uint shadow = 4 * lane;
  uint4 options = shadow_words[shadow + 2];
  if (options.x == 0u) {
    return;
  }
#  ifdef CYCLES_WEBGPU_TRANSPARENT_SHADOW
#    if defined(CYCLES_WEBGPU_SHADOW_SVM) && !defined(CYCLES_WEBGPU_VOLUME_SHADOW_MATERIALS)
  const bool ao_pass = (options.x & CYCLES_WEBGPU_SHADOW_AO_PASS) != 0u;
  options.x &= ~CYCLES_WEBGPU_SHADOW_AO_PASS;
  /* AO reuses this dispatch with the surface pass buffer at binding 3. */
  const uint output = ao_pass ? 6 * lane + 4 : lane;
  const float4 forward = radiance[ao_pass ? output + 1 : ray_words[0] + lane];
  const uint forward_hits = ao_pass ? uint(forward.w) : cycles_forward_light_hits(forward);
#    else
  const uint forward_hits = cycles_forward_light_hits(radiance[ray_words[0] + lane]);
#    endif
  const uint transparent_bounce = (path_words[CYCLES_WEBGPU_PATH_WORDS * lane +
                                              CYCLES_WEBGPU_PATH_TRANSPARENT_BOUNCE] &
                                   CYCLES_WEBGPU_TRANSPARENT_BOUNCE_MASK) + forward_hits;
  const float4 visibility = cycles_bvh2_transparent_shadow(
      shadow_words[shadow], shadow_words[shadow + 1], options, shadow_words[shadow + 3],
      transparent_bounce);
#    if defined(CYCLES_WEBGPU_SHADOW_SVM) && !defined(CYCLES_WEBGPU_VOLUME_SHADOW_MATERIALS)
  const float4 sample = radiance[output];
  const float3 weighted = sample.xyz * visibility.xyz;
  radiance[output] = ao_pass ? float4(clamp_light(weighted, sample.w), 0.0f) :
                              float4(weighted, sample.w);
#    else
  const float4 sample = radiance[lane];
  radiance[lane] = float4(sample.xyz * visibility.xyz, sample.w);
#    endif
#  else
  uint4 hit, coordinates;
  cycles_bvh2_trace(shadow_words[shadow], shadow_words[shadow + 1], options,
                    hit, coordinates, shadow_words[shadow + 3]);
  if (hit.x != 0u) {
    radiance[lane] = float4(0.0f, 0.0f, 0.0f, radiance[lane].w);
  }
#  endif
#else
  const uint ray = 1 + 12 * lane;
  const uint4 origin = uint4(ray_words[ray], ray_words[ray + 1],
                             ray_words[ray + 2], ray_words[ray + 3]);
  const uint4 direction = uint4(ray_words[ray + 4], ray_words[ray + 5],
                                ray_words[ray + 6], ray_words[ray + 7]);
  uint4 options = uint4(ray_words[ray + 8], ray_words[ray + 9],
                              ray_words[ray + 10], 0u);
  uint4 hit, coordinates;
  const uint offset = 8 * lane;
#ifdef CYCLES_WEBGPU_OFFSET_CLOSEST
  const uint previous_geometric = 8 * ray_words[0] + offset;
  if (options.z != 0xffffffffu) {
    const float u = __builtin_bit_cast(float, hit_words[offset + 5]);
    const float v = __builtin_bit_cast(float, hit_words[offset + 6]);
    const float w = 1.0f - u - v;
    // Keep the stock ray offset when the previous hit is within float roundoff
    // of a shared edge, where either adjacent triangle can win the hit test.
    const float tolerance = 4.0f * 0x1.0p-23f;
    options.w = u <= tolerance || v <= tolerance || w <= tolerance;
  }
  uint4 adjusted_origin;
  cycles_bvh2_trace(origin, direction, options, hit, coordinates,
                    previous_geometric, adjusted_origin);
  if (adjusted_origin.x != origin.x || adjusted_origin.y != origin.y ||
      adjusted_origin.z != origin.z) {
    /* Later shading uses the same origin for analytic light intersections. */
    ray_words[ray] = adjusted_origin.x;
    ray_words[ray + 1] = adjusted_origin.y;
    ray_words[ray + 2] = adjusted_origin.z;
  }
#else
  cycles_bvh2_trace(origin, direction, options, hit, coordinates);
#endif
  hit_words[offset] = hit.x;
  hit_words[offset + 1] = hit.y;
  hit_words[offset + 2] = hit.z;
  hit_words[offset + 3] = hit.w;
  hit_words[offset + 4] = coordinates.x;
  hit_words[offset + 5] = coordinates.y;
  hit_words[offset + 6] = coordinates.z;
  hit_words[offset + 7] = coordinates.w;
#endif
}
