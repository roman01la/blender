/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "../blender-platform/cycles-webgpu/volume_layout.h"
#include "kernel/closure/volume.h"

namespace ccl {
namespace {

uint absorption_offset()
{
  return cycles_volume_word(CYCLES_WEBGPU_SURFACE_ABSORPTION_OFFSET);
}

uint absorption_object()
{
  return cycles_volume_word(absorption_offset() + CYCLES_WEBGPU_ABSORPTION_OBJECT);
}

#if defined(CYCLES_WEBGPU_VOLUME_INTERVAL) || defined(CYCLES_WEBGPU_VOLUME_EXTINCTION_SHADOW)
uint volume_medium(const uint object)
{
  const uint base = absorption_offset();
  const uint count = cycles_volume_word(CYCLES_WEBGPU_SURFACE_VOLUME_COUNT);
  for (uint i = 0; i < count; ++i) {
    if (cycles_volume_word(base + i * CYCLES_WEBGPU_VOLUME_SCATTER_WORDS +
                           CYCLES_WEBGPU_ABSORPTION_OBJECT) == object)
    {
      return 1u << i;
    }
  }
  return 0;
}
#endif

float absorption_value(const uint word)
{
  return __uint_as_float(cycles_volume_word(absorption_offset() + word));
}

Spectrum absorption_transmittance(const float distance)
{
  const auto color = make_float3(absorption_value(CYCLES_WEBGPU_ABSORPTION_COLOR),
                                 absorption_value(CYCLES_WEBGPU_ABSORPTION_COLOR + 1),
                                 absorption_value(CYCLES_WEBGPU_ABSORPTION_COLOR + 2));
  const float strength = fmaxf(absorption_value(CYCLES_WEBGPU_ABSORPTION_DENSITY), 0.0f) *
                         absorption_value(CYCLES_WEBGPU_ABSORPTION_OBJECT_DENSITY);
  /* svm_node_closure_volume for a single literal Volume Absorption closure. */
  return volume_color_transmittance((one_spectrum() - color) * strength, distance);
}

void absorption_globals(TraversalGlobals &globals)
{
  globals.data.bvh.root = std::bit_cast<int>(cycles_scene_word(0));
  for (uint slot = 1; slot < SceneOffsetCount; ++slot) {
    globals.offsets[slot] = cycles_scene_word(slot);
  }
}

bool absorption_backfacing(KernelGlobals kg, const Intersection &hit, const float3 direction)
{
  ShaderData sd;
  sd.object = hit.object;
  sd.prim = hit.prim;
  sd.object_flag = cycles_volume_word(cycles_volume_word(0) + 13 * uint(hit.object));
  auto normal = triangle_normal(kg, &sd);
  if (!(sd.object_flag & SD_OBJECT_TRANSFORM_APPLIED)) {
    object_normal_transform(kg, &sd, &normal);
  }
  return dot(normal, direction) > 0.0f;
}

template<bool>
ccl_device_inline bool bvh_volume_anyhit_triangle_filter(KernelGlobals,
                                                           int object,
                                                           int prim,
                                                           const RaySelfPrimitives &self,
                                                           uint)
{
#ifdef CYCLES_WEBGPU_VOLUME_INTERVAL
  return volume_medium(uint(object)) == 0 || intersection_skip_self(self, object, prim);
#else
  return uint(object) != absorption_object() || intersection_skip_self(self, object, prim);
#endif
}

#define BVH_FUNCTION_NAME bvh_intersect_absorption_boundary
#define BVH_FUNCTION_FEATURES 0
#include "kernel/bvh/volume.h"

uint absorption_camera_stack(KernelGlobals kg, const float3 origin)
{
  Ray ray{};
  ray.P = origin;
  /* As in integrator_volume_stack_init, an exit from an object the probe
   * never entered means the camera starts inside it. */
  ray.D = make_float3(0.0f, 0.0f, 1.0f);
  ray.tmax = FLT_MAX;
  ray.self = {PRIM_NONE, OBJECT_NONE, PRIM_NONE, OBJECT_NONE};
  Intersection hit;
#ifdef CYCLES_WEBGPU_VOLUME_INTERVAL
  const uint count = cycles_volume_word(CYCLES_WEBGPU_SURFACE_VOLUME_COUNT);
  uint seen = 0;
  uint inside = 0;
  for (uint step = 0; step < 2u * (count + 1u) &&
                      bvh_intersect_absorption_boundary(kg, &ray, &hit, PATH_RAY_VISIBILITY_CAMERA);
       ++step)
  {
    const uint medium = volume_medium(uint(hit.object));
    if ((seen & medium) == 0 && absorption_backfacing(kg, hit, ray.D)) {
      inside |= medium;
    }
    seen |= medium;
    ray.tmin = intersection_t_offset(hit.t);
    ray.self.object = hit.object;
    ray.self.prim = hit.prim;
    if (seen == ((1u << count) - 1u)) {
      break;
    }
  }
  return inside;
#else
  return bvh_intersect_absorption_boundary(kg, &ray, &hit, PATH_RAY_VISIBILITY_CAMERA) &&
                 absorption_backfacing(kg, hit, ray.D) ? CYCLES_WEBGPU_ABSORPTION_INSIDE : 0u;
#endif
}

float3 absorption_vector(const UInt4 words)
{
  return make_float3(__uint_as_float(words.x), __uint_as_float(words.y), __uint_as_float(words.z));
}

}  // namespace
}  // namespace ccl

using AbsorptionFloat4 = float __attribute__((ext_vector_type(4)));

#ifdef CYCLES_WEBGPU_VOLUME_EXTINCTION_SHADOW
AbsorptionFloat4 cycles_volume_shadow_transmittance(unsigned int active,
                                                     AbsorptionFloat4 origin,
                                                     AbsorptionFloat4 direction,
                                                     UInt4 rng);
#endif

#ifdef CYCLES_WEBGPU_ABSORPTION_SHADOW
AbsorptionFloat4 cycles_absorption_shadow(UInt4 origin,
                                          UInt4 direction,
                                          UInt4 options,
                                          UInt4 geometric,
                                          unsigned int state
#  ifdef CYCLES_WEBGPU_VOLUME_EXTINCTION_SHADOW
                                          , UInt4 rng
#  endif
                                          )
{
  using namespace ccl;
#  ifdef CYCLES_WEBGPU_VOLUME_SCATTER_SHADOW
  if (options.x == 2u) {
    return {1.0f, 1.0f, 1.0f, 0.0f};
  }
#  endif
  TraversalGlobals globals;
  absorption_globals(globals);
  Ray ray{};
  ray.P = absorption_vector(origin);
  ray.D = absorption_vector(direction);
  ray.tmax = __uint_as_float(direction.w);
#  ifdef CYCLES_WEBGPU_VOLUME_SCATTER_SHADOW
  /* A volume scatter point is neither a surface nor a triangle emitter. */
  ray.self = {PRIM_NONE, OBJECT_NONE, PRIM_NONE, OBJECT_NONE};
#  else
  ray.self.object = std::bit_cast<int>(options.y);
  ray.self.prim = std::bit_cast<int>(options.z);
  ray.self.light_object = origin.w ? std::bit_cast<int>(options.w) : OBJECT_NONE;
  ray.self.light_prim = origin.w ? std::bit_cast<int>(origin.w - 1) : PRIM_NONE;
  if (ray.self.prim != PRIM_NONE && ray.self.object != OBJECT_NONE &&
      needs_surface_ray_offset(&globals, ray, geometric.w))
  {
    ray.P = offset_surface_ray(ray.P, geometric);
  }
#  endif
  if (!intersection_ray_valid(&ray)) {
    return {1.0f, 1.0f, 1.0f, 0.0f};
  }
  Intersection hit;
#  ifdef CYCLES_WEBGPU_VOLUME_EXTINCTION_SHADOW
  uint active = state & CYCLES_WEBGPU_VOLUME_MEDIA_MASK;
  uint bounds = state / CYCLES_WEBGPU_VOLUME_BOUNDARY_INCREMENT;
  Spectrum transmittance = one_spectrum();
#  else
  bool inside = (state & CYCLES_WEBGPU_ABSORPTION_INSIDE) != 0;
  uint bounds = state & ~CYCLES_WEBGPU_ABSORPTION_INSIDE;
  float distance = 0.0f;
#  endif
  float segment_start = ray.tmin;
  /* The transparent visibility bit requests the closest shadow hit rather
   * than ending traversal at any hit. Only our boundary can be transparent. */
  while (bvh_intersect_static_triangles(
      &globals, &ray, &hit, PATH_RAY_VISIBILITY_SHADOW_TRANSPARENT)) {
#  ifdef CYCLES_WEBGPU_VOLUME_EXTINCTION_SHADOW
    const uint medium = volume_medium(uint(hit.object));
    if (medium == 0 || ++bounds > VOLUME_BOUNDS_MAX) {
      return {};
    }
    if (active != 0 && hit.t > segment_start) {
      const AbsorptionFloat4 T = cycles_volume_shadow_transmittance(
          active, AbsorptionFloat4{ray.P.x, ray.P.y, ray.P.z, segment_start},
          AbsorptionFloat4{ray.D.x, ray.D.y, ray.D.z, hit.t}, rng);
      transmittance *= make_float3(T.x, T.y, T.z);
    }
    if (absorption_backfacing(&globals, hit, ray.D)) {
      active &= ~medium;
    }
    else {
      active |= medium;
    }
#  else
    if (uint(hit.object) != absorption_object() || ++bounds > VOLUME_BOUNDS_MAX) {
      return {};
    }
    if (inside) {
      distance += fmaxf(hit.t - segment_start, 0.0f);
    }
    inside = !absorption_backfacing(&globals, hit, ray.D);
#  endif
    segment_start = hit.t;
    ray.tmin = intersection_t_offset(hit.t);
    ray.self.object = hit.object;
    ray.self.prim = hit.prim;
  }
#  ifdef CYCLES_WEBGPU_VOLUME_EXTINCTION_SHADOW
  if (active != 0 && ray.tmax > segment_start) {
    const AbsorptionFloat4 T = cycles_volume_shadow_transmittance(
        active, AbsorptionFloat4{ray.P.x, ray.P.y, ray.P.z, segment_start},
        AbsorptionFloat4{ray.D.x, ray.D.y, ray.D.z, ray.tmax}, rng);
    transmittance *= make_float3(T.x, T.y, T.z);
  }
  const Spectrum T = transmittance;
#  else
  if (inside) {
    distance += fmaxf(ray.tmax - segment_start, 0.0f);
  }
  const Spectrum T = absorption_transmittance(distance);
#  endif
  return {T.x, T.y, T.z, 0.0f};
}
#else
AbsorptionFloat4 cycles_absorption_segment(UInt4 origin,
                                           UInt4 direction,
                                           UInt4 hit,
                                           unsigned int hit_t,
                                           unsigned int initialize,
                                           unsigned int state,
                                           unsigned int &next_state
#  ifdef CYCLES_WEBGPU_VOLUME_INTERVAL
                                           , unsigned int &prior_state
#  endif
                                           )
{
  using namespace ccl;
  TraversalGlobals globals;
  absorption_globals(globals);
  if (initialize != 0) {
    state = absorption_camera_stack(&globals, absorption_vector(origin));
  }
#  ifdef CYCLES_WEBGPU_VOLUME_INTERVAL
  prior_state = state;
  const Spectrum T = one_spectrum();
  const uint medium = hit.x ? volume_medium(hit.z) : 0;
  uint boundary = 0;
  if (medium) {
    state += CYCLES_WEBGPU_VOLUME_BOUNDARY_INCREMENT;
    Intersection isect{};
    isect.prim = std::bit_cast<int>(hit.y);
    isect.object = std::bit_cast<int>(hit.z);
    if (absorption_backfacing(&globals, isect, absorption_vector(direction))) {
      state &= ~medium;
    }
    else {
      state |= medium;
    }
    boundary = state / CYCLES_WEBGPU_VOLUME_BOUNDARY_INCREMENT <= VOLUME_BOUNDS_MAX ?
                   CYCLES_WEBGPU_ABSORPTION_BOUNDARY : CYCLES_WEBGPU_ABSORPTION_TERMINATE;
  }
#  else
  const bool inside = (state & CYCLES_WEBGPU_ABSORPTION_INSIDE) != 0;
  /* A miss discards the bounded medium, matching volume_stack_clean(). */
  const float length = (hit.x != 0 && inside) ?
                           fmaxf(__uint_as_float(hit_t) - __uint_as_float(origin.w), 0.0f) :
                           0.0f;
  const Spectrum T = absorption_transmittance(length);
  uint boundary = 0;
  if (hit.x != 0 && hit.z == absorption_object()) {
    const uint bounds = (state & ~CYCLES_WEBGPU_ABSORPTION_INSIDE) + 1u;
    Intersection isect{};
    isect.prim = std::bit_cast<int>(hit.y);
    isect.object = std::bit_cast<int>(hit.z);
    state = bounds | (absorption_backfacing(&globals, isect, absorption_vector(direction)) ?
                          0u : CYCLES_WEBGPU_ABSORPTION_INSIDE);
    boundary = bounds <= VOLUME_BOUNDS_MAX ? CYCLES_WEBGPU_ABSORPTION_BOUNDARY :
                                            CYCLES_WEBGPU_ABSORPTION_TERMINATE;
  }
#  endif
  next_state = state;
  return {T.x, T.y, T.z, float(boundary)};
}
#endif
