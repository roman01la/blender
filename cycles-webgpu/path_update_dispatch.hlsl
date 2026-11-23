/* SPDX-License-Identifier: Apache-2.0 */

#include "../blender-platform/cycles-webgpu/path_state_layout.h"
#include "../blender-platform/cycles-webgpu/material_layout.h"
#include "../blender-platform/cycles-webgpu/volume_layout.h"

typedef uint uint2 __attribute__((ext_vector_type(2)));
typedef float float3 __attribute__((ext_vector_type(3)));

RWStructuredBuffer<uint> ray_words : register(u0);
StructuredBuffer<uint> hit_words : register(t1);
StructuredBuffer<uint> surface_words : register(t2);
StructuredBuffer<uint4> pixels : register(t3);
StructuredBuffer<uint> camera_sampling : register(t4);
StructuredBuffer<float4> materials : register(t5);
RWStructuredBuffer<float4> radiance : register(u6);
RWStructuredBuffer<uint> path_words : register(u7);

#include "ordered_closure_access.hlsl"

#define CYCLES_WEBGPU_EMISSION_DELTA 1
#include "wavefront_queue.hlsl"
#include "wavefront_radiance.hlsl"
#define CYCLES_WEBGPU_LIGHT_PASS_BUFFER radiance
#include "light_passes.hlsl"

uint cycles_camera_sample_word(uint index)
{
  return camera_sampling[index];
}

extern uint4 cycles_path_settings();
extern uint4 cycles_path_lobe_settings();
extern float4 cycles_advance_differential(float4 ray, float glossy_filter, uint portal);
extern uint4 cycles_transparent_advance(uint4 rng, uint4 path, float4 throughput,
                                       uint portal);
extern float4 cycles_sample_mixed_transparency(uint4 rng, float4 diffuse, float4 transparent,
                                               float4 weights, float4 normal, float4 ray);
extern uint2 cycles_absorption_bounce(uint4 rng, uint4 path, float4 throughput);
extern uint cycles_surface_bounce(uint rng_pixel, uint sample_index, uint4 position,
                                  uint4 normal, uint4 geometric_normal,
                                  uint4 specular_normal, uint4 ray, uint4 settings,
                                  uint4 lobe_settings, uint4 path_counts, uint bounce,
                                  uint subsurface_steps, uint ray_differentials,
                                  float min_ray_pdf, float4 material, float4 specular,
                                  float4 parameters, float4 glass, float4 hair,
                                  float4 bump, float4 throughput,
                                  out float4 next_throughput, out float next_pdf, out uint4 origin,
                                  out uint4 direction, out uint4 next_counts, out float surface_rr,
                                  out float roughness_squared, uint light_passes,
                                  out float4 diffuse_ratio, out float4 glossy_ratio);

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
  const uint capacity = ray_words[0];
  const bool light_passes = cycles_light_pass_enabled(capacity);
  const uint subsurface_steps = path_words[path + CYCLES_WEBGPU_PATH_EXTRA_RNG_STEPS];
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
  const bool standalone = CYCLES_WEBGPU_MATERIAL_IS_STANDALONE(material.w);
  const bool portal = material.w == CYCLES_WEBGPU_MATERIAL_RAY_PORTAL;
  const bool mixed = material.w == CYCLES_WEBGPU_MATERIAL_MIXED_TRANSPARENT;
  const bool mixed_ggx = ggx &&
                         CYCLES_WEBGPU_GGX_CAN_BE_TRANSPARENT(materials[material_base + 3]) &&
                         materials[material_base + 4].w != 0.0f;
  const bool mixed_glossy = standalone && material.w == CYCLES_WEBGPU_MATERIAL_GLOSSY_TRANSPARENT &&
                            materials[material_base + 4].w != 0.0f;
  const bool mixed_ordered = CYCLES_WEBGPU_MATERIAL_HAS_ORDERED_CLOSURES(material.w) &&
                             (uint(materials[material_base + 4].w) &
                              CYCLES_WEBGPU_ORDERED_HAS_TRANSPARENT) != 0u;
  const uint boundary = hit_words[surface + CYCLES_WEBGPU_HIT_SPECIAL];
  const bool volume = boundary == CYCLES_WEBGPU_ABSORPTION_BOUNDARY ||
                      boundary == CYCLES_WEBGPU_ABSORPTION_TERMINATE;
  const bool active = path_words[queued] != CYCLES_WEBGPU_PATH_TERMINATED;
  const uint forward_hits = cycles_forward_light_hits(area);
  const bool area_hit = active && forward_hits != 0;
  bool trace_past_area = true;
  if (area_hit) {
    trace_past_area = cycles_trace_past_area(path, forward_hits);
  }
  const uint4 counts = uint4(path_words[path + CYCLES_WEBGPU_PATH_DIFFUSE_BOUNCE],
                              path_words[path + CYCLES_WEBGPU_PATH_GLOSSY_BOUNCE],
                              path_words[path + CYCLES_WEBGPU_PATH_TRANSMISSION_BOUNCE],
                              path_words[path + CYCLES_WEBGPU_PATH_FLAGS]);
  const float surface_rr = __builtin_bit_cast(float, path_words[path + CYCLES_WEBGPU_PATH_SURFACE_RR]);
  const uint4 rng = uint4(ray_words[ray + CYCLES_WEBGPU_RAY_RNG_PIXEL], pixels[lane].z,
                           bounce + subsurface_steps + 1, counts.w);
  float4 mixed_sample = float4(0, 0, 0, 0);
  if (mixed && active && trace_past_area && surface_rr > 0.0f) {
    const uint4 normal = uint4(surface_words[surface + 4], surface_words[surface + 5],
                               surface_words[surface + 6], 0);
    const uint4 direction = uint4(ray_words[ray + 4], ray_words[ray + 5], ray_words[ray + 6], 0);
    mixed_sample = cycles_sample_mixed_transparency(
        rng, material, materials[material_base + 1], materials[material_base + 2],
        __builtin_bit_cast(float4, normal), __builtin_bit_cast(float4, direction));
  }
  bool transparent = material.w == CYCLES_WEBGPU_MATERIAL_TRANSPARENT || portal ||
                           mixed_sample.w != 0.0f;
  const bool shade_surface = !volume && active && trace_past_area && incoming.w >= 0.0f && surface_rr > 0.0f &&
                             (counts.w & CYCLES_WEBGPU_PATH_TERMINATE_AFTER_TRANSPARENT) == 0;
  const float3 surface_throughput = throughput * surface_rr;
  float3 next_throughput = ggx || standalone ? surface_throughput :
                               surface_throughput * (mixed_sample.w != 0.0f ?
                                                         mixed_sample.xyz :
                                                         material.xyz);
  uint4 settings = cycles_path_settings();
  const bool ray_differentials = (settings.x & CYCLES_WEBGPU_PATH_SETTINGS_DIFFERENTIALS) != 0u;
  settings.x &= ~CYCLES_WEBGPU_PATH_SETTINGS_DIFFERENTIALS;
  const float light_max_bounces = __builtin_bit_cast(float4, settings).w;
  uint visibility = 0;
  float roughness_squared = 0.0f;
  bool skip_self = !transparent;
  uint4 origin = uint4(0, 0, 0, 0);
  uint4 direction = uint4(0, 0, 0, 0);

  /* Analytic lights contribute along the segment before the next surface
   * or world hit. Only the opaque surface applies roulette. */
  float3 emission_delta = float3(0, 0, 0);
  if (area_hit) {
    const float3 area_contribution = cycles_forward_emission(area, throughput);
    rgb += area_contribution;
    emission_delta += area_contribution;
  }
  const bool transparent_background = active && trace_past_area && hit_words[surface] == 0u &&
                                      cycles_write_transparent_background(
                                          path, counts.w, throughput);
  /* integrate_surface_holdout(): camera paths keep Cycles' transparent background
   * flag whatever the film setting, so they write the holdout transparency. */
  if (active && trace_past_area && hit_words[surface] != 0u &&
      material.w == CYCLES_WEBGPU_MATERIAL_HOLDOUT &&
      (counts.w & CYCLES_WEBGPU_PATH_TRANSPARENT_BACKGROUND) != 0u) {
    const float3 weighted = throughput * material.xyz;
    const float transparent = (weighted.x + weighted.y + weighted.z) * (1.0f / 3.0f);
    path_words[sum + 3] = __builtin_bit_cast(uint, 1.0f - transparent);
  }
  /* integrate_surface_emission(): the Light Tree dispatch has already applied
   * the forward MIS weight to sampled emitters. It precedes Russian roulette, and
   * film_clamp_light() uses the direct limit through the first bounce. */
  const float4 emission = materials[material_base + CYCLES_WEBGPU_MATERIAL_EMISSION_SLOT];
  if (active && trace_past_area && hit_words[surface] != 0u &&
      (emission.x != 0.0f || emission.y != 0.0f || emission.z != 0.0f)) {
    const float3 contribution = clamp_light(
        throughput * emission.xyz, bounce <= 1u ? cycles_path_direct_clamp() : emission.w);
    rgb += contribution;
    emission_delta += contribution;
  }
  float3 forward_delta = emission_delta;
  if (active && trace_past_area && !transparent_background && incoming.w < 0.0f) {
    const float3 contribution = clamp_light(throughput * incoming.xyz, -incoming.w);
    rgb += contribution;
    forward_delta += contribution;
  }
  const bool has_nee = shade_surface && incoming.w > 0.0f && float(bounce) <= light_max_bounces;
  float3 nee = float3(0, 0, 0);
  if (has_nee) {
    nee = clamp_light(surface_throughput * incoming.xyz, incoming.w);
    rgb += nee;
  }
  if (light_passes) {
    cycles_light_pass_deltas(capacity, lane, bounce, forward_delta, nee, has_nee);
  }
  const bool ordinary_bounce = shade_surface && bounce < settings.x &&
                               (material.w == 1.0f || mixed || ggx || standalone);
  const bool terminal_mixed = (mixed_ggx || mixed_glossy || mixed_ordered) && active && trace_past_area &&
                              surface_rr > 0.0f &&
                              (counts.w & CYCLES_WEBGPU_PATH_TERMINATE_AFTER_TRANSPARENT) != 0 &&
                              (counts.w & CYCLES_WEBGPU_PATH_TERMINATE_ON_NEXT_SURFACE) == 0;
  if (!transparent && !volume && (ordinary_bounce || terminal_mixed) &&
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
    float4 diffuse_ratio, glossy_ratio;
    const bool save_ratios = light_passes && bounce == 0u;
    visibility = cycles_surface_bounce(ray_words[ray + CYCLES_WEBGPU_RAY_RNG_PIXEL], pixels[lane].z,
                                       position, normal, geom_N, spec_N, incoming_ray,
                                       settings, cycles_path_lobe_settings(), counts, bounce,
                                       subsurface_steps, uint(ray_differentials), min_pdf,
                                       material, materials[material_base + 1],
                                       materials[material_base + 2], materials[material_base + 3],
                                       materials[material_base + 4],
                                       materials[material_base + CYCLES_WEBGPU_MATERIAL_NORMAL_SLOT],
                                       float4(next_throughput, 0.0f), bounced_throughput,
                                       bounced_pdf, origin, direction, next_counts, next_rr,
                                       roughness_squared, uint(save_ratios), diffuse_ratio,
                                       glossy_ratio);
    next_throughput = bounced_throughput.xyz;
    if (visibility == CYCLES_WEBGPU_MIXED_TRANSPARENT_PENDING) {
      if (save_ratios && (CYCLES_WEBGPU_MATERIAL_HAS_ORDERED_CLOSURES(material.w) ||
                          (ggx && CYCLES_WEBGPU_GGX_HAS_TRANSLUCENT(materials[material_base + 3])))) {
        const uint record = cycles_light_pass_record(capacity, lane);
        radiance[record] = diffuse_ratio;
        radiance[record + 1u] = glossy_ratio;
      }
      transparent = true;
      skip_self = false;
      visibility = 0;
    }
    else if (visibility != 0 && visibility != CYCLES_WEBGPU_PATH_SUBSURFACE_PENDING) {
      if (save_ratios) {
        const uint record = cycles_light_pass_record(capacity, lane);
        radiance[record] = diffuse_ratio;
        radiance[record + 1u] = glossy_ratio;
      }
      path_words[path + CYCLES_WEBGPU_PATH_DIFFUSE_BOUNCE] = next_counts.x;
      path_words[path + CYCLES_WEBGPU_PATH_GLOSSY_BOUNCE] = next_counts.y;
      path_words[path + CYCLES_WEBGPU_PATH_TRANSMISSION_BOUNCE] = next_counts.z;
      path_words[path + CYCLES_WEBGPU_PATH_FLAGS] = next_counts.w;
      path_words[path + CYCLES_WEBGPU_PATH_MIS_RAY_PDF] = __builtin_bit_cast(uint, bounced_pdf);
      const float next_min_pdf = __builtin_elementwise_min(min_pdf, bounced_pdf);
      path_words[path + CYCLES_WEBGPU_PATH_MIN_RAY_PDF] = __builtin_bit_cast(uint, next_min_pdf);
      const uint mis_normal = path + CYCLES_WEBGPU_PATH_MIS_ORIGIN_N;
      const float4 material_normal = materials[material_base + CYCLES_WEBGPU_MATERIAL_NORMAL_SLOT];
      uint3 mis_N = normal.xyz;
      if (material_normal.w >= CYCLES_WEBGPU_NORMAL_SHADER) {
        const float4 shader_normal = material_normal.w >= CYCLES_WEBGPU_NORMAL_BOTH ?
                                         materials[material_base + 4] : material_normal;
        mis_N = uint3(__builtin_bit_cast(uint, shader_normal.x),
                      __builtin_bit_cast(uint, shader_normal.y),
                      __builtin_bit_cast(uint, shader_normal.z));
      }
      path_words[mis_normal] = mis_N.x;
      path_words[mis_normal + 1] = mis_N.y;
      path_words[mis_normal + 2] = mis_N.z;
      path_words[path + CYCLES_WEBGPU_PATH_SURFACE_RR] = __builtin_bit_cast(uint, next_rr);
    }
  }
  if (volume) {
    if (boundary == CYCLES_WEBGPU_ABSORPTION_BOUNDARY && active && trace_past_area &&
        surface_rr > 0.0f && (counts.w & CYCLES_WEBGPU_PATH_TERMINATE_ON_NEXT_SURFACE) == 0u) {
      next_throughput = surface_throughput;
      visibility = ray_words[ray + 8];
      direction = uint4(ray_words[ray + 4], ray_words[ray + 5], ray_words[ray + 6], ray_words[ray + 7]);
      const uint4 rng = uint4(ray_words[ray + CYCLES_WEBGPU_RAY_RNG_PIXEL], pixels[lane].z,
                               bounce + subsurface_steps + 2, counts.w);
      const uint2 next = cycles_absorption_bounce(
          rng, uint4(bounce, hit_words[surface + 4], 0, 0), float4(next_throughput, 0));
      origin = uint4(ray_words[ray], ray_words[ray + 1], ray_words[ray + 2], next.x);
      path_words[path + CYCLES_WEBGPU_PATH_SURFACE_RR] = next.y;
      path_words[path + CYCLES_WEBGPU_PATH_EXTRA_RNG_STEPS] = subsurface_steps + 1;
      skip_self = false;
    }
  }
  else if (transparent && active && trace_past_area && surface_rr > 0.0f &&
      (counts.w & CYCLES_WEBGPU_PATH_TERMINATE_ON_NEXT_SURFACE) == 0 &&
      (next_throughput.x != 0.0f || next_throughput.y != 0.0f || next_throughput.z != 0.0f)) {
    const uint counter = path + CYCLES_WEBGPU_PATH_TRANSPARENT_BOUNCE;
    visibility = ray_words[ray + 8];
    direction = uint4(ray_words[ray + 4], ray_words[ray + 5], ray_words[ray + 6],
                       ray_words[ray + 7]);
    const uint4 next = cycles_transparent_advance(
        rng, uint4(visibility, path_words[counter], hit_words[surface + 4], 0),
        float4(next_throughput, 0), uint(portal));
    origin = uint4(ray_words[ray], ray_words[ray + 1], ray_words[ray + 2], next.x);
    if (portal) {
      const float3 destination = materials[material_base + 1].xyz;
      const uint3 position = uint3(surface_words[surface], surface_words[surface + 1],
                                  surface_words[surface + 2]);
      const float3 delta = __builtin_bit_cast(float3, position) - destination;
      // integrate_surface_ray_portal resets self-intersection only after teleporting.
      skip_self = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z <= 1.0e-9f;
      origin = uint4(__builtin_bit_cast(uint3, destination), 0);
      direction = uint4(__builtin_bit_cast(uint3, materials[material_base + 2].xyz),
                        __builtin_bit_cast(uint, 3.40282347e+38f));
    }
    path_words[path + CYCLES_WEBGPU_PATH_FLAGS] = next.y;
    path_words[counter] = next.z;
    path_words[path + CYCLES_WEBGPU_PATH_SURFACE_RR] = next.w;
    path_words[path + CYCLES_WEBGPU_PATH_EXTRA_RNG_STEPS] = subsurface_steps + 1;
  }
  const uint3 next_bits = __builtin_bit_cast(uint3, next_throughput);
  const uint3 sum_bits = __builtin_bit_cast(uint3, rgb);
  path_words[weight] = next_bits.x;
  path_words[weight + 1] = next_bits.y;
  path_words[weight + 2] = next_bits.z;
  path_words[sum] = sum_bits.x;
  path_words[sum + 1] = sum_bits.y;
  path_words[sum + 2] = sum_bits.z;
  radiance[lane] = float4(rgb, opacity);
  cycles_write_emission_delta(ray_words[0], lane, bounce, counts.w, emission_delta);
  if (visibility == CYCLES_WEBGPU_PATH_SUBSURFACE_PENDING) {
    if (ray_differentials) {
      /* Stock subsurface entry preserves the surface position differential
       * and clears the angular differential for the internal ray. */
      const uint differential = CYCLES_WEBGPU_RAY_DIFFERENTIAL_BASE(ray_words[0], lane);
      const float dP = __builtin_bit_cast(float, ray_words[differential]);
      const float dD = __builtin_bit_cast(float, ray_words[differential + 1]);
      const float t = __builtin_bit_cast(float, hit_words[surface + 4]);
      ray_words[differential] = __builtin_bit_cast(uint, dP + t * dD);
      ray_words[differential + 1] = 0u;
    }
    path_words[queued] = CYCLES_WEBGPU_PATH_INTERSECT_SUBSURFACE;
    return;
  }

  path_words[queued] = visibility == 0 ? CYCLES_WEBGPU_PATH_TERMINATED :
                                        CYCLES_WEBGPU_PATH_INTERSECT_CLOSEST;
  path_words[bounce_word] = bounce + uint(!transparent && !volume);

  if (ray_differentials && visibility != 0 && !volume && (!transparent || portal)) {
    const uint differential = CYCLES_WEBGPU_RAY_DIFFERENTIAL_BASE(ray_words[0], lane);
    const uint4 ray_diff = uint4(ray_words[differential], ray_words[differential + 1],
                                 hit_words[surface + 4],
                                 __builtin_bit_cast(uint, roughness_squared));
    const float glossy_filter = __builtin_bit_cast(float, cycles_path_lobe_settings().w);
    const uint4 advanced = __builtin_bit_cast(
        uint4, cycles_advance_differential(__builtin_bit_cast(float4, ray_diff),
                                           glossy_filter, uint(portal)));
    ray_words[differential] = advanced.x;
    ray_words[differential + 1] = advanced.y;
  }

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
  // Straight transparency excludes the hit through tmin and preserves the ray origin.
  ray_words[ray + 9] = skip_self ? hit_words[surface + 2] : 0xffffffffu;
  ray_words[ray + 10] = skip_self ? hit_words[surface + 1] : 0xffffffffu;
}
