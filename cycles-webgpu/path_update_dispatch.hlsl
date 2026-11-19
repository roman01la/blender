/* SPDX-License-Identifier: Apache-2.0 */

#include "../blender-platform/cycles-webgpu/path_state_layout.h"
#include "../blender-platform/cycles-webgpu/material_layout.h"

typedef float float3 __attribute__((ext_vector_type(3)));

RWStructuredBuffer<uint> ray_words : register(u0);
StructuredBuffer<uint> hit_words : register(t1);
StructuredBuffer<uint> surface_words : register(t2);
StructuredBuffer<uint4> pixels : register(t3);
StructuredBuffer<uint> camera_sampling : register(t4);
StructuredBuffer<float4> materials : register(t5);
RWStructuredBuffer<float4> radiance : register(u6);
RWStructuredBuffer<uint> path_words : register(u7);

#include "wavefront_queue.hlsl"
#include "wavefront_radiance.hlsl"

uint cycles_camera_sample_word(uint index)
{
  return camera_sampling[index];
}

extern uint4 cycles_path_settings();
extern uint4 cycles_path_lobe_settings();
extern uint cycles_surface_bounce(uint rng_pixel, uint sample_index, uint4 position,
                                  uint4 normal, uint4 geometric_normal,
                                  uint4 specular_normal, uint4 ray, uint4 settings,
                                  uint4 lobe_settings, uint4 path_counts, uint bounce,
                                  float min_ray_pdf, float4 material, float4 specular,
                                  float4 parameters, float4 glass, float4 throughput,
                                  out float4 next_throughput, out float next_pdf, out uint4 origin,
                                  out uint4 direction, out uint4 next_counts, out float surface_rr);

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
  uint lane;
  if (!cycles_queue_slot(id.x, lane)) {
    return;
  }

  const uint path = lane * CYCLES_WEBGPU_PATH_WORDS;
  const uint sum = path + CYCLES_WEBGPU_PATH_RADIANCE;
  const uint queued = path + CYCLES_WEBGPU_PATH_QUEUED_KERNEL;
  const uint bounce_word = path + CYCLES_WEBGPU_PATH_BOUNCE;
  const uint bounce = path_words[bounce_word];
  const uint ray = 1 + 12 * lane;
  const uint surface = 8 * lane;
  const uint weight = path + CYCLES_WEBGPU_PATH_THROUGHPUT;
  const uint3 throughput_bits = uint3(path_words[weight], path_words[weight + 1],
                                      path_words[weight + 2]);
  const float3 throughput = __builtin_bit_cast(float3, throughput_bits);
  const uint3 rgb_bits = uint3(path_words[sum], path_words[sum + 1], path_words[sum + 2]);
  float3 rgb = __builtin_bit_cast(float3, rgb_bits);
  const float opacity = __builtin_bit_cast(float, path_words[sum + 3]);
  const float4 incoming = radiance[lane];
  const float4 area = radiance[ray_words[0] + lane];
  const uint material_base = CYCLES_WEBGPU_MATERIAL_VECTORS * lane;
  const float4 material = materials[material_base];
  const bool ggx = material.w == CYCLES_WEBGPU_MATERIAL_GGX ||
                   material.w == CYCLES_WEBGPU_MATERIAL_MULTISCATTER_GGX;
  const bool active = path_words[queued] != CYCLES_WEBGPU_PATH_TERMINATED;
  const bool area_hit = active && area.w > 0.0f;
  bool trace_past_area = true;
  if (area_hit) {
    trace_past_area = cycles_trace_past_area(path);
  }
  const uint4 counts = uint4(path_words[path + CYCLES_WEBGPU_PATH_DIFFUSE_BOUNCE],
                              path_words[path + CYCLES_WEBGPU_PATH_GLOSSY_BOUNCE],
                              path_words[path + CYCLES_WEBGPU_PATH_TRANSMISSION_BOUNCE],
                              path_words[path + CYCLES_WEBGPU_PATH_FLAGS]);
  const float surface_rr = __builtin_bit_cast(float, path_words[path + CYCLES_WEBGPU_PATH_SURFACE_RR]);
  const bool shade_surface = active && trace_past_area && incoming.w >= 0.0f && surface_rr > 0.0f &&
                             (counts.w & CYCLES_WEBGPU_PATH_TERMINATE_AFTER_TRANSPARENT) == 0;
  const float3 surface_throughput = throughput * surface_rr;
  float3 next_throughput = ggx ? surface_throughput : surface_throughput * material.xyz;
  const uint4 settings = cycles_path_settings();
  const float light_max_bounces = __builtin_bit_cast(float4, settings).w;
  uint visibility = 0;
  uint4 origin = uint4(0, 0, 0, 0);
  uint4 direction = uint4(0, 0, 0, 0);

  /* Analytic lights contribute along the segment before the next surface
   * or world hit. Only the opaque surface applies roulette. */
  if (area_hit) {
    rgb += clamp_light(throughput * area.xyz, area.w);
  }
  if (active && trace_past_area && incoming.w < 0.0f) {
    rgb += clamp_light(throughput * incoming.xyz, -incoming.w);
  }
  if (shade_surface && incoming.w > 0.0f && float(bounce) <= light_max_bounces) {
    rgb += clamp_light(surface_throughput * incoming.xyz, incoming.w);
  }
  if (shade_surface && bounce < settings.x && (material.w == 1.0f || ggx) &&
      (next_throughput.x > 0.0f || next_throughput.y > 0.0f || next_throughput.z > 0.0f)) {
    const uint4 position = uint4(surface_words[surface], surface_words[surface + 1],
                                 surface_words[surface + 2], surface_words[surface + 3]);
    const uint4 normal = uint4(surface_words[surface + 4], surface_words[surface + 5],
                               surface_words[surface + 6], surface_words[surface + 7]);
    const uint normals = 8 * ray_words[0] + surface;
    const uint4 geom_N = uint4(surface_words[normals], surface_words[normals + 1],
                               surface_words[normals + 2], 0);
    const uint4 spec_N = uint4(surface_words[normals + 4], surface_words[normals + 5],
                               surface_words[normals + 6], 0);
    const uint4 incoming_ray = uint4(ray_words[ray + 4], ray_words[ray + 5], ray_words[ray + 6], 0);
    const float min_pdf = __builtin_bit_cast(float, path_words[path + CYCLES_WEBGPU_PATH_MIN_RAY_PDF]);
    float4 bounced_throughput;
    float bounced_pdf;
    uint4 next_counts;
    float next_rr;
    visibility = cycles_surface_bounce(ray_words[ray + CYCLES_WEBGPU_RAY_RNG_PIXEL], pixels[lane].z,
                                       position, normal, geom_N, spec_N, incoming_ray,
                                       settings, cycles_path_lobe_settings(), counts, bounce, min_pdf,
                                       material, materials[material_base + 1],
                                       materials[material_base + 2], materials[material_base + 3],
                                       float4(next_throughput, 0.0f), bounced_throughput,
                                       bounced_pdf, origin, direction, next_counts, next_rr);
    next_throughput = bounced_throughput.xyz;
    if (visibility != 0) {
      path_words[path + CYCLES_WEBGPU_PATH_DIFFUSE_BOUNCE] = next_counts.x;
      path_words[path + CYCLES_WEBGPU_PATH_GLOSSY_BOUNCE] = next_counts.y;
      path_words[path + CYCLES_WEBGPU_PATH_TRANSMISSION_BOUNCE] = next_counts.z;
      path_words[path + CYCLES_WEBGPU_PATH_FLAGS] = next_counts.w;
      path_words[path + CYCLES_WEBGPU_PATH_MIS_RAY_PDF] = __builtin_bit_cast(uint, bounced_pdf);
      const float next_min_pdf = __builtin_elementwise_min(min_pdf, bounced_pdf);
      path_words[path + CYCLES_WEBGPU_PATH_MIN_RAY_PDF] = __builtin_bit_cast(uint, next_min_pdf);
      const uint mis_normal = path + CYCLES_WEBGPU_PATH_MIS_ORIGIN_N;
      path_words[mis_normal] = normal.x;
      path_words[mis_normal + 1] = normal.y;
      path_words[mis_normal + 2] = normal.z;
      path_words[path + CYCLES_WEBGPU_PATH_SURFACE_RR] = __builtin_bit_cast(uint, next_rr);
    }
  }
  const uint3 next_bits = __builtin_bit_cast(uint3, next_throughput);
  const uint3 sum_bits = __builtin_bit_cast(uint3, rgb);
  path_words[weight] = next_bits.x;
  path_words[weight + 1] = next_bits.y;
  path_words[weight + 2] = next_bits.z;
  path_words[sum] = sum_bits.x;
  path_words[sum + 1] = sum_bits.y;
  path_words[sum + 2] = sum_bits.z;
  path_words[queued] = visibility == 0 ? CYCLES_WEBGPU_PATH_TERMINATED :
                                        CYCLES_WEBGPU_PATH_INTERSECT_CLOSEST;
  path_words[bounce_word] = bounce + 1;
  radiance[lane] = float4(rgb, opacity);

  /* An empty ray interval keeps finished slots out of the existing traversal. */
  ray_words[ray] = origin.x;
  ray_words[ray + 1] = origin.y;
  ray_words[ray + 2] = origin.z;
  ray_words[ray + 3] = origin.w;
  ray_words[ray + 4] = direction.x;
  ray_words[ray + 5] = direction.y;
  ray_words[ray + 6] = direction.z;
  ray_words[ray + 7] = visibility == 0 ? 0 : direction.w;
  ray_words[ray + 8] = visibility;
  ray_words[ray + 9] = hit_words[surface + 2];
  ray_words[ray + 10] = hit_words[surface + 1];
}
