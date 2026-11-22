/* SPDX-License-Identifier: Apache-2.0 */

#include "../blender-platform/cycles-webgpu/material_layout.h"
#include "../blender-platform/cycles-webgpu/volume_layout.h"

StructuredBuffer<uint> scene_words : register(t0);
StructuredBuffer<uint> surface_words : register(t1);
StructuredBuffer<uint> ray_words : register(t2);

#include "wavefront_queue.hlsl"

StructuredBuffer<uint> hit_words : register(t3);
RWStructuredBuffer<uint> shading_points : register(u4);

uint cycles_scene_word(uint index)
{
  return scene_words[index];
}

uint cycles_surface_word(uint index)
{
  return surface_words[index];
}

#ifdef CYCLES_WEBGPU_COMPACT_PATHS
extern void cycles_triangle_derivatives(uint4 hit, uint backfacing, out uint4 dpdu, out uint4 dpdv);
extern void cycles_triangle_surface(uint4 hit, uint4 coordinates, uint4 direction,
                                    out uint4 position, out uint4 normal,
                                    out uint4 geometric, out uint4 specular,
                                    out uint4 shadow_along, out uint4 shadow_opposite);
#else
extern void cycles_triangle_surface(uint4 hit, uint4 coordinates, uint4 direction,
                                    out uint4 position, out uint4 normal);
#endif

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
  uint lane;
  if (!cycles_queue_slot(id.x, lane)) {
    return;
  }
  const uint base = 8 * lane;
  const uint4 hit = uint4(hit_words[base], hit_words[base + 1],
                          hit_words[base + 2], hit_words[base + 3]);
  const uint4 coordinates = uint4(hit_words[base + 4], hit_words[base + 5],
                                  hit_words[base + 6], hit_words[base + 7]);
  const uint direction_base = 5 + 12 * lane;
  const uint4 direction = uint4(ray_words[direction_base], ray_words[direction_base + 1],
                                ray_words[direction_base + 2], ray_words[direction_base + 3]);
  uint4 position, normal;
#ifdef CYCLES_WEBGPU_COMPACT_PATHS
  uint4 geometric, specular, shadow_along, shadow_opposite;
  cycles_triangle_surface(hit, coordinates, direction, position, normal, geometric, specular,
                          shadow_along, shadow_opposite);
  /* Keep the original surface records contiguous for the SVM stage. */
  const uint tail = 8 * ray_words[0] + base;
  shading_points[tail] = geometric.x;
  shading_points[tail + 1] = geometric.y;
  shading_points[tail + 2] = geometric.z;
  shading_points[tail + 3] = geometric.w;
  shading_points[tail + 4] = specular.x;
  shading_points[tail + 5] = specular.y;
  shading_points[tail + 6] = specular.z;
  shading_points[tail + CYCLES_WEBGPU_SURFACE_VOLUME_HIT] = coordinates.w;
  const uint shadow = tail + 8 * ray_words[0];
  shading_points[shadow] = shadow_along.x;
  shading_points[shadow + 1] = shadow_along.y;
  shading_points[shadow + 2] = shadow_along.z;
  shading_points[shadow + 3] = shadow_along.w;
  shading_points[shadow + 4] = shadow_opposite.x;
  shading_points[shadow + 5] = shadow_opposite.y;
  shading_points[shadow + 6] = shadow_opposite.z;
  shading_points[shadow + 7] = shadow_opposite.w;
#  ifdef CYCLES_WEBGPU_QUERY_HITS
  /* Give shader queries access to the original intersection without a ninth
   * shader binding. The query SVM binds the BVH in the freed slot. */
  const uint query_hit = shadow + CYCLES_WEBGPU_QUERY_HIT_WORDS * ray_words[0];
  shading_points[query_hit] = hit.x;
  shading_points[query_hit + 1] = hit.y;
  shading_points[query_hit + 2] = hit.z;
  shading_points[query_hit + 3] = hit.w;
  shading_points[query_hit + 4] = coordinates.x;
  shading_points[query_hit + 5] = coordinates.y;
  shading_points[query_hit + 6] = coordinates.z;
  shading_points[query_hit + 7] = coordinates.w;
  const uint tangent_block = 4 * 8 * ray_words[0];
#  else
  const uint tangent_block = 3 * 8 * ray_words[0];
#  endif
  if ((position.w == 1 || position.w == 2) &&
      CYCLES_WEBGPU_SURFACE_HAS_HAIR(surface_words[0]))
  {
    uint4 dpdu, dpdv;
    cycles_triangle_derivatives(hit, position.w == 2, dpdu, dpdv);
    const uint tangent = tangent_block + CYCLES_WEBGPU_SURFACE_HAIR_WORDS * lane;
    shading_points[tangent] = dpdu.x;
    shading_points[tangent + 1] = dpdu.y;
    shading_points[tangent + 2] = dpdu.z;
    shading_points[tangent + 4] = dpdv.x;
    shading_points[tangent + 5] = dpdv.y;
    shading_points[tangent + 6] = dpdv.z;
  }
#else
  cycles_triangle_surface(hit, coordinates, direction, position, normal);
#endif
  shading_points[base] = position.x;
  shading_points[base + 1] = position.y;
  shading_points[base + 2] = position.z;
  shading_points[base + 3] = position.w;
  shading_points[base + 4] = normal.x;
  shading_points[base + 5] = normal.y;
  shading_points[base + 6] = normal.z;
  shading_points[base + 7] = normal.w;
}
