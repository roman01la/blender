/* SPDX-License-Identifier: Apache-2.0 */

#include "../blender-platform/cycles-webgpu/material_layout.h"
#include "../blender-platform/cycles-webgpu/path_state_layout.h"
#include "../blender-platform/cycles-webgpu/volume_layout.h"

#ifndef CYCLES_WEBGPU_LOCAL_CLOSURES
StructuredBuffer<uint> svm_words : register(t0);
StructuredBuffer<uint> surface_words : register(t1);
StructuredBuffer<uint> ray_words : register(t2);
RWStructuredBuffer<float4> materials : register(u3);
#ifdef CYCLES_WEBGPU_SCENE_QUERIES
StructuredBuffer<uint> scene_words : register(t4);
#else
StructuredBuffer<uint> hit_words : register(t4);
#endif
StructuredBuffer<uint> path_words : register(t5);
StructuredBuffer<uint> camera_words : register(t6);
StructuredBuffer<uint> metadata_words : register(t7);

#endif

#ifdef CYCLES_WEBGPU_HARDWARE_SVM_TEXTURES
/* Retain a scalar-argument function for the WGSL atlas bridge. The HLSL
 * frontend used by this build supports buffer but not sampled texture types. */
__attribute__((noinline))
float4 cycles_svm_hardware_sample(uint image, uint width, uint height, uint extension,
                                  float u, float v)
{
  return float4(float(image + width + height + extension), u, v, 1.0);
}
#endif

#include "svm_context.hlsl"

#ifdef CYCLES_WEBGPU_COMPACT_PATHS
#include "wavefront_queue.hlsl"
#endif

uint cycles_svm_word(uint index)
{
  return svm_words[index];
}

#ifdef CYCLES_WEBGPU_SCENE_QUERIES
uint cycles_scene_word(uint index)
{
  return scene_words[index];
}

uint cycles_surface_word(uint index)
{
  return metadata_words[index];
}

uint cycles_camera_sample_word(uint index)
{
  return camera_words[CYCLES_WEBGPU_QUERY_CAMERA_OFFSET + index];
}
#endif

uint cycles_svm_hit_word(uint index)
{
#ifdef CYCLES_WEBGPU_SCENE_QUERIES
  return surface_words[3 * CYCLES_WEBGPU_QUERY_HIT_WORDS * ray_words[0] + index];
#else
  return hit_words[index];
#endif
}

extern void cycles_svm_material(uint shader, uint primitive, float u, float v,
                                uint lane, uint eval_options,
                                out float4 base, out float4 specular, out float4 layers,
                                out float4 glass, out float4 subsurface, out float4 emission,
                                out float4 closure_normal);

void cycles_svm_write_aov(uint lane, uint offset, uint color, float4 value)
{
#ifndef CYCLES_WEBGPU_LOCAL_CLOSURES
  const uint colors = camera_words[CYCLES_WEBGPU_CAMERA_AOV_COLORS];
  const uint values = camera_words[CYCLES_WEBGPU_CAMERA_AOV_VALUES];
  const uint slot = color != 0 ? offset / 4 : offset;
  if (slot < (color != 0 ? colors : values)) {
    const uint base = CYCLES_WEBGPU_MATERIAL_VECTORS * ray_words[0] + lane * (colors + values);
    materials[base + (color != 0 ? slot : colors + slot)] += value;
  }
#endif
}

#ifdef CYCLES_WEBGPU_ORDERED_CLOSURES
uint cycles_svm_ordered_base(uint lane)
{
  const uint aov_slots = camera_words[CYCLES_WEBGPU_CAMERA_AOV_COLORS] +
                         camera_words[CYCLES_WEBGPU_CAMERA_AOV_VALUES];
  return (CYCLES_WEBGPU_MATERIAL_VECTORS + aov_slots) * ray_words[0] +
         CYCLES_WEBGPU_ORDERED_PATH_VECTORS * lane;
}

void cycles_svm_write_ordered_lobe(uint lane, uint index, float4 color, float4 normal)
{
  const uint address = cycles_svm_ordered_base(lane) + CYCLES_WEBGPU_ORDERED_LOBE_VECTORS * index;
  materials[address] = color;
  materials[address + 1u] = normal;
}
#endif

uint4 cycles_svm_context(uint lane, uint field)
{
  const uint surface = 8 * lane;
  const uint ray = 1 + 12 * lane;
  const uint path = CYCLES_WEBGPU_PATH_WORDS * lane;
  if (field <= CYCLES_WEBGPU_SVM_GEOMETRIC) {
    const uint base = field == CYCLES_WEBGPU_SVM_POSITION ? surface :
                      field == CYCLES_WEBGPU_SVM_NORMAL ? surface + 4 :
                                                          8 * ray_words[0] + surface;
    return uint4(surface_words[base], surface_words[base + 1], surface_words[base + 2],
                 field == CYCLES_WEBGPU_SVM_POSITION ? surface_words[base + 3] : 0);
  }
  if (field == CYCLES_WEBGPU_SVM_DIRECTION) {
    return uint4(ray_words[ray + 4], ray_words[ray + 5], ray_words[ray + 6], 0);
  }
  if (field == CYCLES_WEBGPU_SVM_INTERSECTION) {
    return uint4(cycles_svm_hit_word(surface + 4), cycles_svm_hit_word(surface + 5),
                 cycles_svm_hit_word(surface + 6),
                 surface_words[surface + 3]);
  }
  if (field == CYCLES_WEBGPU_SVM_BOUNCES) {
    return uint4(path_words[path + CYCLES_WEBGPU_PATH_BOUNCE],
                 path_words[path + CYCLES_WEBGPU_PATH_DIFFUSE_BOUNCE],
                 path_words[path + CYCLES_WEBGPU_PATH_GLOSSY_BOUNCE],
                 path_words[path + CYCLES_WEBGPU_PATH_TRANSMISSION_BOUNCE]);
  }
  if (field == CYCLES_WEBGPU_SVM_DIFFERENTIAL) {
    if ((svm_words[1] & CYCLES_WEBGPU_SVM_NEEDS_DIFFERENTIALS) == 0) {
      return uint4(0, 0, 0, 0);
    }
    const uint base = CYCLES_WEBGPU_RAY_DIFFERENTIAL_BASE(ray_words[0], lane);
    return uint4(ray_words[base], ray_words[base + 1], cycles_svm_hit_word(surface + 4), 0);
  }
#ifdef CYCLES_WEBGPU_SCENE_QUERIES
  if (field == CYCLES_WEBGPU_QUERY_RNG) {
    const uint bounce = path_words[path + CYCLES_WEBGPU_PATH_BOUNCE];
    const uint steps = path_words[path + CYCLES_WEBGPU_PATH_EXTRA_RNG_STEPS];
    return uint4(ray_words[ray + CYCLES_WEBGPU_RAY_RNG_PIXEL],
                 (bounce + steps + 1) * CYCLES_WEBGPU_PRNG_BOUNCE_NUM,
                 path_words[path + CYCLES_WEBGPU_PATH_PIXEL + 2],
                 path_words[path + CYCLES_WEBGPU_PATH_MIN_RAY_PDF]);
  }
  if (field == CYCLES_WEBGPU_QUERY_HIT) {
    return uint4(cycles_svm_hit_word(surface), cycles_svm_hit_word(surface + 1),
                 cycles_svm_hit_word(surface + 2), cycles_svm_hit_word(surface + 3));
  }
#endif
#ifdef CYCLES_WEBGPU_COMPACT_PATHS
  if (field == CYCLES_WEBGPU_SVM_TRIANGLE_DPDU || field == CYCLES_WEBGPU_SVM_TRIANGLE_DPDV) {
#  ifdef CYCLES_WEBGPU_SCENE_QUERIES
    const uint tangent_block = 4 * 8 * ray_words[0];
#  else
    const uint tangent_block = 3 * 8 * ray_words[0];
#  endif
    const uint tangent = tangent_block + CYCLES_WEBGPU_SURFACE_HAIR_WORDS * lane +
                         (field == CYCLES_WEBGPU_SVM_TRIANGLE_DPDV ? 4 : 0);
    return uint4(surface_words[tangent], surface_words[tangent + 1], surface_words[tangent + 2], 0);
  }
#endif
  if (field >= CYCLES_WEBGPU_SVM_WORLD_TO_CAMERA) {
    return cycles_svm_scene_context(field, cycles_svm_hit_word(surface + 2),
                                   surface_words[surface + 7]);
  }
  const uint shader = surface_words[surface + 7] & CYCLES_WEBGPU_SVM_SHADER_ID_MASK;
  const uint shader_flags = metadata_words[metadata_words[4] + shader];
  return uint4(ray_words[ray + 8], path_words[path + CYCLES_WEBGPU_PATH_FLAGS],
               path_words[path + CYCLES_WEBGPU_PATH_TRANSPARENT_BOUNCE], shader_flags);
}

#ifdef CYCLES_WEBGPU_LOCAL_CLOSURES
__attribute__((noinline))
void cycles_local_material(uint lane)
{
#else
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
#ifdef CYCLES_WEBGPU_COMPACT_PATHS
  uint lane;
  if (!cycles_queue_slot(id.x, lane)) {
    return;
  }
#else
  const uint lane = id.x;
  if (lane >= ray_words[0]) {
    return;
  }
#endif

#endif
  const uint base = 8 * lane;
  const uint surface_status = surface_words[base + 3];
  bool film_aov = false;
#ifndef CYCLES_WEBGPU_LOCAL_CLOSURES
  if ((svm_words[1] & CYCLES_WEBGPU_SVM_AOV) != 0) {
    const uint path = CYCLES_WEBGPU_PATH_WORDS * lane;
    const uint path_flags = path_words[path + CYCLES_WEBGPU_PATH_FLAGS];
    const uint slots = camera_words[CYCLES_WEBGPU_CAMERA_AOV_COLORS] +
                       camera_words[CYCLES_WEBGPU_CAMERA_AOV_VALUES];
    film_aov = slots != 0 && path_words[path + CYCLES_WEBGPU_PATH_BOUNCE] == 0 &&
               (path_flags & CYCLES_WEBGPU_PATH_TRANSPARENT_BACKGROUND) != 0;
    if (film_aov) {
      const uint aov = CYCLES_WEBGPU_MATERIAL_VECTORS * ray_words[0] + lane * slots;
      for (uint i = 0; i < slots; ++i) {
        materials[aov + i] = float4(0, 0, 0, 0);
      }
    }
  }
#endif
  float4 result = float4(0, 0, 0, 0);
  float4 specular = float4(0, 0, 0, 0);
  float4 layers = float4(0, 0, 0, 0);
  float4 glass = float4(0, 0, 0, 0);
  float4 subsurface = float4(0, 0, 0, 0);
  float4 emission = float4(0, 0, 0, 0);
  float4 closure_normal = float4(0, 0, 0, 0);
  const uint special_hit = cycles_svm_hit_word(base + CYCLES_WEBGPU_HIT_SPECIAL);
  const bool subsurface_exit = special_hit == CYCLES_WEBGPU_HIT_SUBSURFACE_EXIT;
  const uint shader = surface_words[base + 7] & CYCLES_WEBGPU_SVM_SHADER_ID_MASK;
  const uint shader_flags = surface_status == 1 || surface_status == 2 ?
                                metadata_words[metadata_words[4] + shader] : 0u;
#ifdef CYCLES_WEBGPU_LOCAL_CLOSURES
  const uint path = CYCLES_WEBGPU_PATH_WORDS * lane;
  const uint path_flags = path_words[path + CYCLES_WEBGPU_PATH_FLAGS];
  // integrator_intersect_terminate() skips shading after roulette if the hit cannot emit.
  // Keep the lane for forward lamps, light passes and film retirement, but leave its
  // closure records zero. Camera/background paths can still need Holdout or AOVs.
  // Keep SD_HAS_TRANSPARENT_SHADOW, SD_HAS_VOLUME, SD_HAS_ONLY_VOLUME,
  // SD_HAS_CONSTANT_EMISSION and SD_HAS_EMISSION on the normal shading route.
  const uint terminal_shader_flags =
      (1u << 17u) | (1u << 18u) | (1u << 19u) | (1u << 27u) | (1u << 29u);
  const bool reject_surface = special_hit == 0u && path_words[path + CYCLES_WEBGPU_PATH_BOUNCE] > 0u &&
                              (path_flags & (CYCLES_WEBGPU_PATH_TRANSPARENT_BACKGROUND |
                                             CYCLES_WEBGPU_PATH_SUBSURFACE)) == 0u &&
                              (shader_flags & terminal_shader_flags) == 0u &&
                              (svm_words[1] & CYCLES_WEBGPU_SVM_RAY_PORTAL) == 0u &&
                              __builtin_bit_cast(float, path_words[path + CYCLES_WEBGPU_PATH_SURFACE_RR]) ==
                                  0.0f;
#else
  const bool reject_surface = false;
#endif
  const bool bump_correction = (shader_flags & CYCLES_WEBGPU_SHADER_BUMP_CORRECTION) != 0u;
  const bool bump_exit = subsurface_exit &&
                         (shader_flags & CYCLES_WEBGPU_SHADER_BSSRDF_BUMP) != 0u;
  if (subsurface_exit && !bump_exit) {
    // The stock exit shader always replaces the material with white diffuse.
    result = float4(1, 1, 1, 1);
  }
  else if (!reject_surface &&
           (special_hit == 0u || special_hit == CYCLES_WEBGPU_MIXED_VOLUME_BOUNDARY || bump_exit) &&
           (surface_status == 1 || surface_status == 2)) {
    const uint eval_options = (film_aov && !subsurface_exit ? CYCLES_WEBGPU_SVM_EVAL_AOV : 0u) |
                              (bump_exit ? CYCLES_WEBGPU_SVM_EVAL_SUBSURFACE_EXIT : 0u) |
                              (bump_correction ? CYCLES_WEBGPU_SVM_EVAL_BUMP_CORRECTION : 0u);
    cycles_svm_material(surface_words[base + 7], cycles_svm_hit_word(base + 1),
                        __builtin_bit_cast(float, cycles_svm_hit_word(base + 5)),
                        __builtin_bit_cast(float, cycles_svm_hit_word(base + 6)), lane, eval_options,
                        result, specular, layers, glass, subsurface, emission, closure_normal);
#ifdef CYCLES_WEBGPU_ORDERED_CLOSURES
    if (CYCLES_WEBGPU_MATERIAL_HAS_ORDERED_CLOSURES(result.w)) {
      const uint tail = cycles_svm_ordered_base(lane);
      result.x = tail <= CYCLES_WEBGPU_ORDERED_TAIL_FLOAT_LIMIT ?
                     float(tail) :
                     __builtin_bit_cast(float, tail + CYCLES_WEBGPU_ORDERED_TAIL_BITS_BIAS);
      result.z = float(bump_correction);
    }
#endif
    const bool diffuse_translucent = result.w == CYCLES_WEBGPU_MATERIAL_DIFFUSE_TRANSLUCENT;
    if (closure_normal.w != 0.0f || diffuse_translucent) {
      if (closure_normal.w != 0.0f && bump_correction) {
        closure_normal.w += 1.0f;
      }
      if (diffuse_translucent) {
        layers.w = float(bump_correction);
      }
    }
  }
  else if (surface_status != 0) {
    result.w = CYCLES_WEBGPU_MATERIAL_UNSHADED;
  }
#ifdef CYCLES_WEBGPU_LOCAL_CLOSURES
  local_closure_result = result;
  local_closure_specular = specular;
  local_closure_layers = layers;
  local_closure_glass = glass;
  local_closure_subsurface = subsurface;
  local_closure_emission = emission;
  local_closure_closure_normal = closure_normal;
#else
  const uint material = CYCLES_WEBGPU_MATERIAL_VECTORS * lane;
  materials[material] = result;
  materials[material + 1] = specular;
  materials[material + 2] = layers;
  materials[material + 3] = glass;
  materials[material + 4] = subsurface;
  materials[material + CYCLES_WEBGPU_MATERIAL_EMISSION_SLOT] = emission;
  materials[material + CYCLES_WEBGPU_MATERIAL_NORMAL_SLOT] = closure_normal;
#endif
}
