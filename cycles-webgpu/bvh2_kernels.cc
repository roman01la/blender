/* SPDX-License-Identifier: Apache-2.0 */

#include <bit>

#ifdef __aarch64__
#  include <arm_neon.h>
#endif

#define CCL_NAMESPACE_BEGIN namespace ccl {
#define CCL_NAMESPACE_END }
#include "kernel/types.h"

#define __KERNEL_GPU__
#include "kernel/globals.h"
#undef __KERNEL_GPU__

/* The triangle traversal variant does not implement shadow linking. */
#undef __SHADOW_LINKING__

__attribute__((pure)) unsigned int cycles_scene_word(unsigned int index);
#if defined(CYCLES_WEBGPU_SURFACE) || defined(CYCLES_WEBGPU_OFFSET_CLOSEST) || \
    defined(CYCLES_WEBGPU_SUBSURFACE)
__attribute__((pure)) unsigned int cycles_surface_word(unsigned int index);
#endif
#ifdef CYCLES_WEBGPU_SUBSURFACE
__attribute__((pure)) unsigned int cycles_camera_sample_word(unsigned int index);
#endif

namespace ccl {

struct KernelImageTexture;
struct KernelTileDescriptor;
struct KernelImageUDIM;
struct KernelImageInfo;

enum SceneOffset : uint {
  SceneNodes = 1,
  SceneLeaves,
  SceneVisibility,
  ScenePrimitiveIndex,
  ScenePrimitiveObject,
  SceneObjectNode,
  SceneObjects,
  SceneTriangleIndices,
  SceneTriangleVertices,
  SceneOffsetCount,
};

struct TraversalGlobals {
  KernelData data{};
  uint offsets[SceneOffsetCount];
  /* Stock object helpers sometimes take the address of a fetched record.
   * The static triangle variant reads one object per fetch and keeps no references. */
  mutable KernelObject object{};
};

using KernelGlobals = const TraversalGlobals *;

#define KERNEL_DATA_ARRAY(type, name) \
  struct name##_tag { \
    using Element = type; \
  };
#include "kernel/data_arrays.h"

/* The included stock headers declare helpers for other kernel variants.
 * Leaving their adapters undefined makes accidental use a link error. */
template<typename Tag> const typename Tag::Element &scene_fetch(KernelGlobals, Tag, uint index);
template<typename Tag> const typename Tag::Element *scene_array(KernelGlobals, Tag);

ccl_device_inline float scene_float(uint word)
{
  return std::bit_cast<float>(cycles_scene_word(word));
}

ccl_device_inline float4 scene_float4(uint word)
{
  return make_float4(
      scene_float(word), scene_float(word + 1), scene_float(word + 2), scene_float(word + 3));
}

#ifdef CYCLES_WEBGPU_SURFACE
ccl_device_inline float4 surface_float4(uint word)
{
  return make_float4(std::bit_cast<float>(cycles_surface_word(word)),
                     std::bit_cast<float>(cycles_surface_word(word + 1)),
                     std::bit_cast<float>(cycles_surface_word(word + 2)),
                     std::bit_cast<float>(cycles_surface_word(word + 3)));
}

ccl_device_inline uint scene_fetch(KernelGlobals, tri_shader_tag, uint index)
{
#  ifdef CYCLES_WEBGPU_COMPACT_PATHS
  return cycles_surface_word(cycles_surface_word(1) + index);
#  else
  return cycles_surface_word(cycles_surface_word(1) + index) & ~SHADER_SMOOTH_NORMAL;
#  endif
}
#endif

#if defined(CYCLES_WEBGPU_SURFACE) && defined(CYCLES_WEBGPU_COMPACT_PATHS)
ccl_device_inline packed_normal scene_fetch(KernelGlobals, attributes_normal_tag, uint index)
{
  return std::bit_cast<packed_normal>(cycles_surface_word(cycles_surface_word(3) + index));
}
#endif

ccl_device_inline float4 scene_fetch(KernelGlobals kg, bvh_nodes_tag, uint index)
{
  return scene_float4(kg->offsets[SceneNodes] + 4 * index);
}

ccl_device_inline float4 scene_fetch(KernelGlobals kg, bvh_leaf_nodes_tag, uint index)
{
  return scene_float4(kg->offsets[SceneLeaves] + 4 * index);
}

ccl_device_inline uint scene_fetch(KernelGlobals kg, prim_visibility_tag, uint index)
{
  return cycles_scene_word(kg->offsets[SceneVisibility] + index);
}

ccl_device_inline uint scene_fetch(KernelGlobals kg, prim_index_tag, uint index)
{
  return cycles_scene_word(kg->offsets[ScenePrimitiveIndex] + index);
}

ccl_device_inline uint scene_fetch(KernelGlobals kg, prim_object_tag, uint index)
{
  return cycles_scene_word(kg->offsets[ScenePrimitiveObject] + index);
}

ccl_device_inline uint scene_fetch(KernelGlobals kg, object_node_tag, uint index)
{
  return cycles_scene_word(kg->offsets[SceneObjectNode] + index);
}

ccl_device_inline const KernelObject &scene_fetch(KernelGlobals kg, objects_tag, uint index)
{
  const uint word = kg->offsets[SceneObjects] + 13 * index;
  kg->object.position_offset = std::bit_cast<int>(cycles_scene_word(word));
  kg->object.itfm = {scene_float4(word + 1), scene_float4(word + 5), scene_float4(word + 9)};
#ifdef CYCLES_WEBGPU_SURFACE
  const uint surface = cycles_surface_word(0) + 13 * index + 1;
  kg->object.tfm = {surface_float4(surface),
                   surface_float4(surface + 4),
                   surface_float4(surface + 8)};
#  ifdef CYCLES_WEBGPU_COMPACT_PATHS
  kg->object.mesh_volume.normal_offset = std::bit_cast<int>(
      cycles_surface_word(cycles_surface_word(2) + 2 * index));
#  endif
#endif
  return kg->object;
}

ccl_device_inline uint3 scene_fetch(KernelGlobals kg, tri_vindex_tag, uint index)
{
  const uint word = kg->offsets[SceneTriangleIndices] + 3 * index;
  return make_uint3(
      cycles_scene_word(word), cycles_scene_word(word + 1), cycles_scene_word(word + 2));
}

ccl_device_inline float3 scene_fetch(KernelGlobals kg, tri_verts_tag, uint index)
{
  const uint word = kg->offsets[SceneTriangleVertices] + 3 * index;
  return make_float3(scene_float(word), scene_float(word + 1), scene_float(word + 2));
}

#ifdef CYCLES_WEBGPU_SUBSURFACE
ccl_device_inline uint scene_fetch(KernelGlobals, object_flag_tag, uint index)
{
  return cycles_surface_word(cycles_surface_word(0) + 13 * index);
}

ccl_device_inline float scene_fetch(KernelGlobals, sample_pattern_lut_tag, uint index)
{
  return std::bit_cast<float>(cycles_camera_sample_word(7 + FILTER_TABLE_SIZE + index));
}
#endif

}  // namespace ccl

#define kernel_data (kg->data)
#define kernel_data_fetch(name, index) scene_fetch(kg, name##_tag{}, index)
#define kernel_data_array(name) scene_array(kg, name##_tag{})
#include "kernel/bvh/nodes.h"
#include "kernel/bvh/types.h"
#include "kernel/bvh/util.h"
#include "kernel/geom/triangle_intersect.h"
#if defined(CYCLES_WEBGPU_SURFACE) && defined(CYCLES_WEBGPU_COMPACT_PATHS)
#  pragma push_macro("ccl_device")
#  undef ccl_device
#  define ccl_device static inline __attribute__((always_inline))
#  include "kernel/closure/bsdf_util.h"
#  pragma pop_macro("ccl_device")
#endif

namespace ccl {

#if defined(CYCLES_WEBGPU_SURFACE) && defined(CYCLES_WEBGPU_COMPACT_PATHS)
/* The linear envelope in Cycles' shadow_ray_smooth_surface_offset(). */
ccl_device_inline float terminator_envelope_at_vertex(const float3 p,
                                                      const float3 vertex,
                                                      const float3 other_a,
                                                      const float3 other_b,
                                                      const float3 normal)
{
  const float h = max(max(dot(other_a - vertex, normal), dot(other_b - vertex, normal)), 0.0f);
  return max(dot(vertex - p, normal) + h, 0.0f);
}

ccl_device_inline float terminator_envelope(const float3 p,
                                            const float3 verts[3],
                                            const float3 normals[3],
                                            const float side)
{
  const float ha = terminator_envelope_at_vertex(p, verts[0], verts[1], verts[2], side * normals[0]);
  const float hb = terminator_envelope_at_vertex(p, verts[1], verts[0], verts[2], side * normals[1]);
  const float hc = terminator_envelope_at_vertex(p, verts[2], verts[0], verts[1], side * normals[2]);
  return min(min(ha, hb), hc);
}

/* Evaluate the two static-triangle branches of shadow_ray_smooth_surface_offset
 * before the light stage, while packed vertex normals are still available. */
ccl_device_inline void triangle_terminator_offsets(KernelGlobals kg,
                                                   const ShaderData *sd,
                                                   float3 &along,
                                                   float3 &opposite)
{
  float3 verts[3], normals[3];
  triangle_vertices_and_normals(kg, sd, verts, normals);
  const float u = 1.0f - sd->u - sd->v;
  const float v = sd->u;
  const float w = sd->v;
  const float3 point = verts[0] * u + verts[1] * v + verts[2] * w;
  float3 n = normals[0] * u + normals[1] * v + normals[2] * w;
  if (!(sd->object_flag & SD_OBJECT_TRANSFORM_APPLIED)) {
    object_dir_transform(kg, sd, &n);
  }

  const float a = dot(normals[2] - normals[0], verts[0] - verts[2]);
  const float b = dot(normals[2] - normals[1], verts[1] - verts[2]);
  const float c = dot(normals[1] - normals[0], verts[1] - verts[0]);
  const float h = a * u * (u - 1) + (a + b + c) * u * v + b * v * (v - 1);
  const float front = max(terminator_envelope(point, verts, normals, 1.0f), h * 0.5f);
  const float back = min(-terminator_envelope(point, verts, normals, -1.0f), h * 0.5f);
  const float facing = dot(n, sd->Ng);
  along = n * (facing > 0 ? front : back);
  opposite = n * (facing < 0 ? front : back);
}
#endif

#define BVH_FUNCTION_NAME bvh_intersect_static_triangles
#define BVH_FUNCTION_FEATURES 0
#include "kernel/bvh/traversal.h"

#ifdef CYCLES_WEBGPU_SUBSURFACE
#  define BVH_FUNCTION_NAME bvh_intersect_local_static_triangles
#  define BVH_FUNCTION_FEATURES 0
#  include "kernel/bvh/local.h"

template<bool single_hit>
ccl_device_inline bool scene_intersect_local(KernelGlobals kg,
                                              const Ray *ray,
                                              LocalIntersection *isect,
                                              const int object,
                                              uint *lcg_state,
                                              const int max_hits)
{
  if (!intersection_ray_valid(ray)) {
    isect->num_hits = 0;
    return false;
  }
  return bvh_intersect_local_static_triangles(kg, ray, isect, object, lcg_state, max_hits);
}
#endif
}  // namespace ccl

using UInt4 = unsigned int __attribute__((ext_vector_type(4)));

#ifdef CYCLES_WEBGPU_SUBSURFACE
#  undef ccl_device
#  define ccl_device static inline __attribute__((always_inline))
#  include "kernel/util/differential.h"
#  include "kernel/integrator/subsurface_random_walk.h"
#  include "kernel/closure/bssrdf.h"
#  include "kernel/integrator/subsurface_entry.h"

using Float4 = float __attribute__((ext_vector_type(4)));

/* The extra components carry the entry controls: time, BSSRDF type
 * (Skin toggle in the probe), IOR, encoded anisotropy, and microfacet alpha. */
#ifdef CYCLES_WEBGPU_COMPACT_PATHS
#  include "../blender-platform/cycles-webgpu/path_state_layout.h"
#endif

[[gnu::always_inline]] void cycles_subsurface_walk(const Float4 entry,
                                                  const Float4 incident,
                                                  const Float4 normal,
                                                  const Float4 albedo,
                                                  const Float4 radius,
                                                  const Float4 weight,
                                                  const UInt4 rng,
                                                  const UInt4 surface,
                                                  UInt4 &hit,
                                                  Float4 &coordinates,
                                                  Float4 &exit_direction,
                                                  Float4 &throughput
#ifdef CYCLES_WEBGPU_COMPACT_PATHS
                                                  , const Float4 geometric,
                                                  const Float4 shading_normal,
                                                  Float4 &exit_origin
#endif
                                                  )
{
  using namespace ccl;
  TraversalGlobals globals;
  globals.data.bvh.root = std::bit_cast<int>(cycles_scene_word(0));
  for (uint slot = 1; slot < SceneOffsetCount; ++slot) {
    globals.offsets[slot] = cycles_scene_word(slot);
  }
  auto &integrator = globals.data.integrator;
  integrator.sampling_pattern = cycles_camera_sample_word(0);
  integrator.sobol_index_mask = cycles_camera_sample_word(2);
  integrator.blue_noise_sequence_length = cycles_camera_sample_word(3);
  integrator.tabulated_sobol_sequence_size = cycles_camera_sample_word(4);
  integrator.pixel_jitter = make_float2(std::bit_cast<float>(cycles_camera_sample_word(5)),
                                        std::bit_cast<float>(cycles_camera_sample_word(6)));
#ifdef CYCLES_WEBGPU_COMPACT_PATHS
  uint path_settings = 7 + FILTER_TABLE_SIZE;
  if (integrator.sampling_pattern == ccl::SAMPLING_PATTERN_TABULATED_SOBOL) {
    path_settings += integrator.tabulated_sobol_sequence_size *
                     NUM_TAB_SOBOL_PATTERNS * NUM_TAB_SOBOL_DIMENSIONS;
  }
  integrator.scrambling_distance = std::bit_cast<float>(cycles_camera_sample_word(path_settings + 2));
#else
  integrator.scrambling_distance = std::bit_cast<float>(rng.w);
#endif

  IntegratorStateCPU state{};
  state.ray.P = make_float3(entry.x, entry.y, entry.z);
  state.ray.time = entry.w;
  state.isect.prim = int(surface.x);
  state.isect.object = int(surface.y);
#ifdef CYCLES_WEBGPU_COMPACT_PATHS
  state.subsurface.N = make_float3(shading_normal.x, shading_normal.y, shading_normal.z);
#else
  state.subsurface.N = make_float3(normal.x, normal.y, normal.z);
#endif
  state.subsurface.albedo = make_float3(albedo.x, albedo.y, albedo.z);
  state.subsurface.radius = make_float3(radius.x, radius.y, radius.z);
  state.subsurface.anisotropy = radius.w;
  state.path.throughput = make_float3(weight.x, weight.y, weight.z);
  RNGState rng_state{rng.x, rng.y, int(rng.z)};
  ShaderData sd{};
  sd.wi = -make_float3(incident.x, incident.y, incident.z);
  Bssrdf bssrdf{};
#ifdef CYCLES_WEBGPU_COMPACT_PATHS
  bssrdf.type = ClosureType(int(normal.w));
#else
  bssrdf.type = normal.w == 0.0f ? CLOSURE_BSSRDF_RANDOM_WALK_ID :
                                  CLOSURE_BSSRDF_RANDOM_WALK_SKIN_ID;
#endif
  bssrdf.N = make_float3(normal.x, normal.y, normal.z);
  bssrdf.ior = albedo.w;
  bssrdf.alpha = weight.w;
#ifdef CYCLES_WEBGPU_COMPACT_PATHS
  /* SVM has validated all three radii and clamped the uniform inputs. */
  bssrdf.radius = state.subsurface.radius;
  bssrdf.albedo = state.subsurface.albedo;
  bssrdf_setup_radius(&bssrdf, bssrdf.type);
  state.subsurface.radius = bssrdf.radius;
  if (bssrdf.type == CLOSURE_BSSRDF_RANDOM_WALK_SKIN_ID) {
    bssrdf.alpha = 1.0f;
  }
  const float3 Ng = make_float3(geometric.x, geometric.y, geometric.z);
#else
  const float3 Ng = bssrdf.N;
#endif
  float3 outgoing = zero_float3();
  const bool entered = subsurface_entry_bounce(&globals, &bssrdf, &sd, &rng_state, &outgoing);
  state.ray.D = outgoing;
  Ray ray{};
  LocalIntersection local{};
  if (entered && dot(Ng, outgoing) < 0.0f &&
      subsurface_random_walk(&globals, &state, rng_state, ray, local))
  {
    const Intersection &exit = local.hits[0];
    hit = {1, uint(exit.prim), uint(exit.object), uint(exit.type)};
    coordinates = {exit.t, exit.u, exit.v, 0.0f};
#ifdef CYCLES_WEBGPU_COMPACT_PATHS
    /* Match subsurface_scatter(): approach the exit from outside the object. */
    const float3 P = ray.P + ray.D * ray.tmax * 2.0f;
    exit_origin = {P.x, P.y, P.z, 0.0f};
    exit_direction = {-ray.D.x, -ray.D.y, -ray.D.z, ray.tmax};
#else
    exit_direction = {ray.D.x, ray.D.y, ray.D.z, 0.0f};
#endif
    const Spectrum result = state.path.throughput;
    throughput = {result.x, result.y, result.z, 0.0f};
  }
  else {
    hit = {};
    coordinates = {};
    exit_direction = {};
    throughput = {};
#ifdef CYCLES_WEBGPU_COMPACT_PATHS
    exit_origin = {};
#endif
  }
}
#endif

#ifdef CYCLES_WEBGPU_SURFACE
#  ifdef CYCLES_WEBGPU_COMPACT_PATHS
[[gnu::always_inline]] void cycles_triangle_surface(UInt4 hit,
                                                     UInt4 coordinates,
                                                     UInt4 direction,
                                                     UInt4 &position,
                                                     UInt4 &normal,
                                                     UInt4 &geometric,
                                                     UInt4 &specular,
                                                     UInt4 &shadow_along,
                                                     UInt4 &shadow_opposite)
#  else
void cycles_triangle_surface(UInt4 hit, UInt4 coordinates, UInt4 direction,
                             UInt4 &position, UInt4 &normal)
#  endif
{
  position = {0, 0, 0, 0};
  normal = {0, 0, 0, 0xffffffffu};
#  ifdef CYCLES_WEBGPU_COMPACT_PATHS
  geometric = {0, 0, 0, 0};
  specular = {0, 0, 0, 0};
  shadow_along = {0, 0, 0, 0};
  shadow_opposite = {0, 0, 0, 0};
#  endif
  if (hit.x == 0) {
    return;
  }
  if (hit.w != ccl::PRIMITIVE_TRIANGLE
#  ifndef CYCLES_WEBGPU_COMPACT_PATHS
      || (cycles_surface_word(cycles_surface_word(1) + hit.y) & ccl::SHADER_SMOOTH_NORMAL)
#  endif
  )
  {
    position.w = 3;
    return;
  }

  ccl::TraversalGlobals globals;
  for (unsigned int slot = 1; slot < ccl::SceneOffsetCount; ++slot) {
    globals.offsets[slot] = cycles_scene_word(slot);
  }
  ccl::ShaderData sd{};
  sd.prim = std::bit_cast<int>(hit.y);
  sd.object = std::bit_cast<int>(hit.z);
  sd.type = ccl::PRIMITIVE_TRIANGLE;
  sd.object_flag = cycles_surface_word(cycles_surface_word(0) + 13 * hit.z);
  sd.u = std::bit_cast<float>(coordinates.y);
  sd.v = std::bit_cast<float>(coordinates.z);
  ccl::triangle_shader_setup(&globals, &sd);
  if (!(sd.object_flag & ccl::SD_OBJECT_TRANSFORM_APPLIED)) {
    ccl::object_normal_transform(&globals, &sd, &sd.Ng);
#  ifdef CYCLES_WEBGPU_COMPACT_PATHS
    ccl::object_normal_transform(&globals, &sd, &sd.N);
#  endif
  }
  const ccl::float3 ray_direction = ccl::make_float3(std::bit_cast<float>(direction.x),
                                                    std::bit_cast<float>(direction.y),
                                                    std::bit_cast<float>(direction.z));
  const bool backface = ccl::dot(sd.Ng, ray_direction) > 0;
  if (backface) {
    sd.Ng = -sd.Ng;
    sd.N = -sd.N;
  }

#  ifdef CYCLES_WEBGPU_COMPACT_PATHS
  sd.wi = -ray_direction;
  if (sd.shader & ccl::SHADER_SMOOTH_NORMAL) {
    sd.shader_flag = cycles_surface_word(cycles_surface_word(4) + (sd.shader & ccl::SHADER_MASK));
    const uint32_t cutoff = cycles_surface_word(cycles_surface_word(2) + 2 * hit.z + 1);
    if (std::bit_cast<float>(cutoff) > 0.0f) {
      ccl::float3 along, opposite;
      ccl::triangle_terminator_offsets(&globals, &sd, along, opposite);
      shadow_along = {std::bit_cast<uint32_t>(along.x),
                      std::bit_cast<uint32_t>(along.y),
                      std::bit_cast<uint32_t>(along.z),
                      cutoff};
      shadow_opposite = {std::bit_cast<uint32_t>(opposite.x),
                         std::bit_cast<uint32_t>(opposite.y),
                         std::bit_cast<uint32_t>(opposite.z),
                         0};
    }
  }
  const ccl::float3 glossy = ccl::maybe_ensure_valid_specular_reflection(&sd, sd.N);
  geometric = {std::bit_cast<unsigned int>(sd.Ng.x),
               std::bit_cast<unsigned int>(sd.Ng.y),
               std::bit_cast<unsigned int>(sd.Ng.z),
               sd.object_flag};
  specular = {std::bit_cast<unsigned int>(glossy.x),
              std::bit_cast<unsigned int>(glossy.y),
              std::bit_cast<unsigned int>(glossy.z),
              0};
#  else
  sd.N = sd.Ng;
#  endif

  position = {std::bit_cast<unsigned int>(sd.P.x),
              std::bit_cast<unsigned int>(sd.P.y),
              std::bit_cast<unsigned int>(sd.P.z),
              backface ? 2u : 1u};
  normal = {std::bit_cast<unsigned int>(sd.N.x),
            std::bit_cast<unsigned int>(sd.N.y),
            std::bit_cast<unsigned int>(sd.N.z),
            static_cast<unsigned int>(sd.shader)};
}
#endif

#if defined(CYCLES_WEBGPU_OFFSET_CLOSEST) || defined(CYCLES_WEBGPU_INTERSECT_SHADOW) || \
    defined(CYCLES_WEBGPU_WAVEFRONT_LIGHT)
/* Match integrate_surface_ray_offset() for the supported static triangles. */
static bool needs_surface_ray_offset(ccl::KernelGlobals kg,
                                     const ccl::Ray &ray,
                                     const unsigned int object_flags)
{
  ccl::float3 verts[3];
  ccl::triangle_vertices(kg, ray.self.object, ray.self.prim, verts);
  auto P = ray.P;
  auto D = ray.D;
  if (!(object_flags & ccl::SD_OBJECT_TRANSFORM_APPLIED)) {
    const ccl::Transform itfm = ccl::object_fetch_transform(
        kg, ray.self.object, ccl::OBJECT_INVERSE_TRANSFORM);
    P = ccl::transform_point(&itfm, P);
    D = ccl::transform_direction(&itfm, D);
  }
  return !ccl::ray_triangle_intersect_self(P, D, verts);
}

static ccl::float3 offset_surface_ray(const ccl::float3 P, const UInt4 geometric)
{
  const auto Ng = ccl::make_float3(std::bit_cast<float>(geometric.x),
                                  std::bit_cast<float>(geometric.y),
                                  std::bit_cast<float>(geometric.z));
  return ccl::ray_offset(P, Ng);
}
#endif

void cycles_bvh2_trace(UInt4 origin,
                       UInt4 direction,
                       UInt4 options,
                       UInt4 &hit,
                       UInt4 &coordinates
#ifdef CYCLES_WEBGPU_OFFSET_CLOSEST
                       , unsigned int previous_geometric,
                       UInt4 &adjusted_origin
#elif defined(CYCLES_WEBGPU_INTERSECT_SHADOW) || defined(CYCLES_WEBGPU_WAVEFRONT_LIGHT)
                       , UInt4 geometric
#endif
)
{
  ccl::TraversalGlobals globals;
  globals.data.bvh.root = std::bit_cast<int>(cycles_scene_word(0));
  for (unsigned int slot = 1; slot < ccl::SceneOffsetCount; ++slot) {
    globals.offsets[slot] = cycles_scene_word(slot);
  }

  ccl::Ray ray{};
  ray.P = ccl::make_float3(std::bit_cast<float>(origin.x),
                           std::bit_cast<float>(origin.y),
                           std::bit_cast<float>(origin.z));
  ray.D = ccl::make_float3(std::bit_cast<float>(direction.x),
                           std::bit_cast<float>(direction.y),
                           std::bit_cast<float>(direction.z));
  ray.tmin = std::bit_cast<float>(origin.w);
  ray.tmax = std::bit_cast<float>(direction.w);
  ray.self.object = std::bit_cast<int>(options.y);
  ray.self.prim = std::bit_cast<int>(options.z);
  ray.self.light_object = OBJECT_NONE;
  ray.self.light_prim = PRIM_NONE;

#ifdef CYCLES_WEBGPU_OFFSET_CLOSEST
  adjusted_origin = origin;
  if (ray.self.prim != PRIM_NONE && ray.self.object != OBJECT_NONE) {
    const auto object_flags = cycles_surface_word(previous_geometric + 3);
    if (options.w != 0 || needs_surface_ray_offset(&globals, ray, object_flags)) {
      const UInt4 geometric = {cycles_surface_word(previous_geometric),
                               cycles_surface_word(previous_geometric + 1),
                               cycles_surface_word(previous_geometric + 2),
                               object_flags};
      ray.P = offset_surface_ray(ray.P, geometric);
      adjusted_origin = {std::bit_cast<unsigned int>(ray.P.x),
                         std::bit_cast<unsigned int>(ray.P.y),
                         std::bit_cast<unsigned int>(ray.P.z),
                         origin.w};
    }
  }
#elif defined(CYCLES_WEBGPU_INTERSECT_SHADOW) || defined(CYCLES_WEBGPU_WAVEFRONT_LIGHT)
  if (ray.self.prim != PRIM_NONE && ray.self.object != OBJECT_NONE &&
      needs_surface_ray_offset(&globals, ray, geometric.w))
  {
    ray.P = offset_surface_ray(ray.P, geometric);
  }
#endif

  ccl::Intersection intersection{};
  intersection.t = ray.tmax;
  intersection.prim = PRIM_NONE;
  intersection.object = OBJECT_NONE;
  const bool found = ccl::intersection_ray_valid(&ray) &&
                     ccl::bvh_intersect_static_triangles(&globals, &ray, &intersection, options.x);

  hit = {static_cast<unsigned int>(found),
         std::bit_cast<unsigned int>(intersection.prim),
         std::bit_cast<unsigned int>(intersection.object),
         std::bit_cast<unsigned int>(intersection.type)};
  coordinates = {std::bit_cast<unsigned int>(intersection.t),
                 std::bit_cast<unsigned int>(intersection.u),
                 std::bit_cast<unsigned int>(intersection.v),
                 0};
}
