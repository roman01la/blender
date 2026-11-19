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
  const uint4 options = shadow_words[shadow + 2];
  if (options.x == 0u) {
    return;
  }
  uint4 hit, coordinates;
  cycles_bvh2_trace(shadow_words[shadow], shadow_words[shadow + 1], options,
                    hit, coordinates, shadow_words[shadow + 3]);
  if (hit.x != 0u) {
    radiance[lane] = float4(0.0f, 0.0f, 0.0f, radiance[lane].w);
  }
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
