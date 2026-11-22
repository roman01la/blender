/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

/* Included inside the BVH translation unit so the stock kernels reuse the
 * static triangle instantiations and the same packed object/normal data. */
#pragma push_macro("__OBJECT_MOTION__")
#undef __OBJECT_MOTION__
#pragma push_macro("ccl_device")
#undef ccl_device
#define ccl_device static inline __attribute__((always_inline))

#include "kernel/util/differential.h"

namespace ccl {

ccl_device_inline bool scene_intersect(KernelGlobals kg,
                                        const Ray *ray,
                                        const uint visibility,
                                        Intersection *isect)
{
  return intersection_ray_valid(ray) && bvh_intersect_static_triangles(kg, ray, isect, visibility);
}

ccl_device_inline bool scene_intersect_shadow(KernelGlobals kg,
                                               const Ray *ray,
                                               const uint visibility)
{
  Intersection isect;
  return scene_intersect(kg, ray, visibility, &isect);
}

/* Static triangle subset of shader_setup_from_ray(). The triangle interpolator
 * supplies transformed geometry and smooth normals, including back faces. */
ccl_device_inline void shader_setup_from_ray(KernelGlobals kg,
                                              ShaderData *sd,
                                              const Ray *ray,
                                              const Intersection *isect)
{
  sd->u = isect->u;
  sd->v = isect->v;
  sd->ray_length = isect->t;
  sd->type = isect->type;
  sd->object = isect->object;
  sd->object_flag = scene_fetch(kg, object_flag_tag{}, sd->object);
  sd->prim = isect->prim;
  sd->time = ray->time;
  sd->wi = -ray->D;
  triangle_shader_setup(kg, sd);
  if (!(sd->object_flag & SD_OBJECT_TRANSFORM_APPLIED)) {
    object_normal_transform(kg, sd, &sd->N);
    object_normal_transform(kg, sd, &sd->Ng);
  }
  if (dot(sd->Ng, sd->wi) < 0.0f) {
    sd->Ng = -sd->Ng;
    sd->N = -sd->N;
  }
}

}  // namespace ccl

#include "kernel/svm/webgpu_scene_queries.h"

#pragma pop_macro("ccl_device")
#pragma pop_macro("__OBJECT_MOTION__")

using Float4 = float __attribute__((ext_vector_type(4)));

namespace {

void setup_query_globals(ccl::TraversalGlobals &globals)
{
  using namespace ccl;
  globals.data.bvh.bvh_layout = BVH_LAYOUT_BVH2;
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
  uint path_settings = 7 + FILTER_TABLE_SIZE;
  if (integrator.sampling_pattern == SAMPLING_PATTERN_TABULATED_SOBOL) {
    path_settings += integrator.tabulated_sobol_sequence_size *
                     NUM_TAB_SOBOL_PATTERNS * NUM_TAB_SOBOL_DIMENSIONS;
  }
  integrator.scrambling_distance = std::bit_cast<float>(cycles_camera_sample_word(path_settings + 2));
  integrator.ao_bounces_distance = std::bit_cast<float>(cycles_camera_sample_word(path_settings + 3));
}

void setup_query_surface(ccl::KernelGlobals kg, ccl::ShaderData &sd, const UInt4 position,
                          const UInt4 hit)
{
  sd.P = ccl::make_float3(std::bit_cast<float>(position.x),
                          std::bit_cast<float>(position.y),
                          std::bit_cast<float>(position.z));
  sd.prim = std::bit_cast<int>(hit.y);
  sd.object = std::bit_cast<int>(hit.z);
  sd.object_flag = ccl::scene_fetch(kg, ccl::object_flag_tag{}, sd.object);
  sd.type = ccl::PRIMITIVE_TRIANGLE;
  sd.time = -1.0f;
  sd.runtime_flag = position.w == 2 ? ccl::SR_BACKFACING : 0;
}

void setup_query_path(ccl::IntegratorStateCPU &state, const UInt4 rng)
{
  state.path.rng_pixel = rng.x;
  state.path.rng_offset = rng.y;
  state.path.sample = int(rng.z);
  state.path.min_ray_pdf = std::bit_cast<float>(rng.w);
}

}  // namespace

float cycles_query_ao(UInt4 position, UInt4 hit, UInt4 rng, Float4 input,
                      unsigned int samples, unsigned int flags)
{
  ccl::TraversalGlobals globals;
  setup_query_globals(globals);
  ccl::IntegratorStateCPU state{};
  setup_query_path(state, rng);
  ccl::ShaderData sd{};
  setup_query_surface(&globals, sd, position, hit);
  return ccl::svm_ao(&globals, &state, &sd, ccl::make_float3(input.x, input.y, input.z),
                     input.w, int(samples), int(flags));
}

Float4 cycles_query_bevel(UInt4 position, UInt4 hit, UInt4 rng, Float4 input, Float4 Ng,
                          unsigned int samples)
{
  ccl::TraversalGlobals globals;
  setup_query_globals(globals);
  ccl::IntegratorStateCPU state{};
  setup_query_path(state, rng);
  ccl::ShaderData sd{};
  setup_query_surface(&globals, sd, position, hit);
  sd.N = ccl::make_float3(input.x, input.y, input.z);
  sd.Ng = ccl::make_float3(Ng.x, Ng.y, Ng.z);
  const ccl::float3 normal = ccl::svm_bevel(&globals, &state, &sd, input.w, int(samples));
  return {normal.x, normal.y, normal.z, 0.0f};
}

void cycles_query_raycast(UInt4 position,
                          UInt4 hit,
                          Float4 ray_origin,
                          Float4 ray_direction,
                          unsigned int only_local,
                          Float4 &data,
                          Float4 &normal)
{
  ccl::TraversalGlobals globals;
  setup_query_globals(globals);
  ccl::ShaderData sd{}, hit_sd{};
  setup_query_surface(&globals, sd, position, hit);
  data = {ray_origin.w, 0.0f, 0.0f, 0.0f};
  normal = {};
  if (ccl::svm_raycast(&globals, nullptr, &sd,
                       ccl::make_float3(ray_origin.x, ray_origin.y, ray_origin.z),
                       ccl::make_float3(ray_direction.x, ray_direction.y, ray_direction.z),
                       ray_origin.w, only_local != 0, 0.0f, hit_sd))
  {
    data = {hit_sd.ray_length, 1.0f, sd.object == hit_sd.object ? 1.0f : 0.0f, 0.0f};
    normal = {hit_sd.N.x, hit_sd.N.y, hit_sd.N.z, 0.0f};
  }
}
