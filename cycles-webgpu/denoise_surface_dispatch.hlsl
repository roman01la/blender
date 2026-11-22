/* SPDX-License-Identifier: Apache-2.0 */

#include "../blender-platform/cycles-webgpu/path_state_layout.h"
#include "../blender-platform/cycles-webgpu/material_layout.h"
#include "../blender-platform/cycles-webgpu/volume_layout.h"

typedef float float3 __attribute__((ext_vector_type(3)));

StructuredBuffer<uint> ray_words : register(t0);
StructuredBuffer<uint> surface_words : register(t1);
StructuredBuffer<float4> materials : register(t2);
StructuredBuffer<uint> camera_sampling : register(t3);
StructuredBuffer<float4> radiance : register(t4);
StructuredBuffer<uint> path_words : register(t5);
RWStructuredBuffer<float4> guide_state : register(u6);
StructuredBuffer<uint> guide_params : register(t7);

#include "wavefront_queue.hlsl"
#include "wavefront_radiance.hlsl"

uint cycles_camera_sample_word(uint index)
{
  return camera_sampling[index];
}

extern float cycles_denoise_blur_alpha(float min_ray_pdf);
extern void cycles_surface_denoise_guides(float4 material,
                                          float4 tint_alpha,
                                          float4 weights,
                                          float4 glass,
                                          float4 hair,
                                          float4 normal,
                                          float4 specular_normal,
                                          float4 geometric_normal,
                                          float4 view,
                                          float blur_alpha,
                                          uint options,
                                          float4 feature_throughput,
                                          out float4 next_feature_throughput,
                                          out float4 albedo,
                                          out float4 guide_normal);

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
  uint lane;
  if (!cycles_queue_slot(id.x, lane)) {
    return;
  }

  const uint path = lane * CYCLES_WEBGPU_PATH_WORDS;
  if (path_words[path + CYCLES_WEBGPU_PATH_QUEUED_KERNEL] == CYCLES_WEBGPU_PATH_TERMINATED) {
    return;
  }

  const uint feature = lane * 3;
  const float4 feature_throughput = guide_state[feature];
  if (feature_throughput.x == 0.0f && feature_throughput.y == 0.0f &&
      feature_throughput.z == 0.0f)
  {
    return;
  }

  const uint weight = path + CYCLES_WEBGPU_PATH_THROUGHPUT;
  const float3 throughput = __builtin_bit_cast(
      float3, uint3(path_words[weight], path_words[weight + 1], path_words[weight + 2]));
  const float4 incoming = radiance[lane];
  const float4 area = radiance[ray_words[0] + lane];
  float3 emission = float3(0, 0, 0);
  bool trace_past_area = true;
  const uint forward_hits = cycles_forward_light_hits(area);
  if (forward_hits != 0) {
    trace_past_area = cycles_trace_past_area_limit(
        (path_words[path + CYCLES_WEBGPU_PATH_TRANSPARENT_BOUNCE] &
         CYCLES_WEBGPU_TRANSPARENT_BOUNCE_MASK) + forward_hits);
    emission += cycles_forward_emission(area, throughput);
  }
  if (trace_past_area && incoming.w < 0.0f) {
    emission += clamp_light(throughput * incoming.xyz, -incoming.w);
  }

  const uint surface = 8 * lane;
  const uint normals = 8 * ray_words[0] + surface;
  const uint material_base = CYCLES_WEBGPU_MATERIAL_VECTORS * lane;
  const uint surface_status = surface_words[surface + 3];
  if (trace_past_area && (surface_status == 1 || surface_status == 2)) {
    /* Match film_write_emission_or_background_pass(). The Light Tree stage
     * has already weighted this emission by the forward MIS factor. */
    const float4 surface_emission = materials[material_base + CYCLES_WEBGPU_MATERIAL_EMISSION_SLOT];
    const uint bounce = path_words[path + CYCLES_WEBGPU_PATH_BOUNCE];
    emission += clamp_light(throughput * surface_emission.xyz,
                            bounce <= 1u ? 3.40282347e+38f : surface_emission.w);
  }

  float4 albedo = float4(feature_throughput.xyz * emission, 0.0f);
  const float surface_rr = __builtin_bit_cast(
      float, path_words[path + CYCLES_WEBGPU_PATH_SURFACE_RR]);
  const uint flags = path_words[path + CYCLES_WEBGPU_PATH_FLAGS];
  float4 material = materials[material_base];
  const float4 packed_transparency = materials[material_base + 4];
  if ((flags & CYCLES_WEBGPU_PATH_TERMINATE_AFTER_TRANSPARENT) != 0) {
    /* Stock Cycles suppresses non-transparent closures at this path limit. */
    if (material.w == CYCLES_WEBGPU_MATERIAL_MIXED_TRANSPARENT) {
      material = float4(materials[material_base + 1].xyz, CYCLES_WEBGPU_MATERIAL_TRANSPARENT);
    }
    else if ((material.w == CYCLES_WEBGPU_MATERIAL_GGX ||
              material.w == CYCLES_WEBGPU_MATERIAL_MULTISCATTER_GGX) &&
             materials[material_base + 3].w == 0.0f && packed_transparency.w != 0.0f) {
      material = float4(packed_transparency.xyz, CYCLES_WEBGPU_MATERIAL_TRANSPARENT);
    }
  }
  const float closure = material.w;
  const bool transparent = closure == CYCLES_WEBGPU_MATERIAL_TRANSPARENT ||
                           closure == CYCLES_WEBGPU_MATERIAL_RAY_PORTAL;
  const uint volume_hit = surface_words[normals + CYCLES_WEBGPU_SURFACE_VOLUME_HIT];
  const bool volume_boundary = volume_hit == CYCLES_WEBGPU_ABSORPTION_BOUNDARY ||
                               volume_hit == CYCLES_WEBGPU_ABSORPTION_TERMINATE;
  // Blender writes denoising features at the subsurface entry only.
  const bool shade_surface = !volume_boundary && (flags & CYCLES_WEBGPU_PATH_SUBSURFACE) == 0 &&
                             trace_past_area && incoming.w >= 0.0f && surface_rr > 0.0f &&
                             (flags & (CYCLES_WEBGPU_PATH_TERMINATE_ON_NEXT_SURFACE |
                                       (transparent ? 0u :
                                          CYCLES_WEBGPU_PATH_TERMINATE_AFTER_TRANSPARENT))) == 0;
  if (shade_surface) {
    const uint ray = 1 + 12 * lane;
    const float4 normal = __builtin_bit_cast(
        float4, uint4(surface_words[surface + 4], surface_words[surface + 5],
                      surface_words[surface + 6], surface_words[surface + 7]));
    const float4 specular_N = __builtin_bit_cast(
        float4, uint4(surface_words[normals + 4], surface_words[normals + 5],
                      surface_words[normals + 6], 0));
    const float4 geometric_N = __builtin_bit_cast(
        float4, uint4(surface_words[normals], surface_words[normals + 1],
                      surface_words[normals + 2], 0));
    const float4 wi = -__builtin_bit_cast(
        float4, uint4(ray_words[ray + 4], ray_words[ray + 5], ray_words[ray + 6], 0));
    const float min_pdf = __builtin_bit_cast(
        float, path_words[path + CYCLES_WEBGPU_PATH_MIN_RAY_PDF]);
    float4 next_feature_throughput, surface_albedo, surface_normal;
    cycles_surface_denoise_guides(material, materials[material_base + 1],
                                  materials[material_base + 2], materials[material_base + 3],
                                  materials[material_base + 4], normal, specular_N, geometric_N,
                                  wi, cycles_denoise_blur_alpha(min_pdf),
                                  guide_params[2], feature_throughput,
                                  next_feature_throughput, surface_albedo, surface_normal);
    albedo += surface_albedo;
    guide_state[feature + 2] += surface_normal;
    guide_state[feature] = next_feature_throughput;
  }
  guide_state[feature + 1] += albedo;
}
