/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "material_layout.h"
#include "path_state_layout.h"

#include "kernel/svm/util.h"
#include "util/projection.h"
#include "util/transform.h"

CCL_NAMESPACE_BEGIN

using WebGPUSvmContextWords = uint __attribute__((ext_vector_type(4)));

/* The host eligibility check needs slot dependencies, while the values come
 * from the live surface and path slots during shading. */
struct WebGPUSvmNoSurface {
  ccl_device_inline_method WebGPUSvmContextWords operator()(const uint field) const
  {
    if (field >= CYCLES_WEBGPU_SVM_WORLD_TO_CAMERA && field < CYCLES_WEBGPU_SVM_WORLD_TO_NDC_END) {
      const uint first = field < CYCLES_WEBGPU_SVM_WORLD_TO_NDC ?
                             CYCLES_WEBGPU_SVM_WORLD_TO_CAMERA :
                             CYCLES_WEBGPU_SVM_WORLD_TO_NDC;
      WebGPUSvmContextWords row{};
      row[field - first] = __float_as_uint(1.0f);
      return row;
    }
    if (field >= CYCLES_WEBGPU_SVM_CAMERA_TO_WORLD && field < CYCLES_WEBGPU_SVM_CAMERA_TO_WORLD_END) {
      WebGPUSvmContextWords row{};
      row[field - CYCLES_WEBGPU_SVM_CAMERA_TO_WORLD] = __float_as_uint(1.0f);
      return row;
    }
    if (field >= CYCLES_WEBGPU_SVM_OBJECT_TRANSFORM && field < CYCLES_WEBGPU_SVM_OBJECT_COLOR) {
      const uint first = field < CYCLES_WEBGPU_SVM_OBJECT_INVERSE ?
                             CYCLES_WEBGPU_SVM_OBJECT_TRANSFORM :
                             CYCLES_WEBGPU_SVM_OBJECT_INVERSE;
      WebGPUSvmContextWords row{};
      row[field - first] = __float_as_uint(1.0f);
      return row;
    }
    if (field == CYCLES_WEBGPU_SVM_POSITION) {
      return {0, 0, __float_as_uint(1.0f), 0};
    }
    if (field == CYCLES_WEBGPU_SVM_NORMAL) {
      /* A unit normal keeps normal-dependent host evaluations finite. */
      return {0, 0, __float_as_uint(1.0f), 0};
    }
    return {};
  }
};

struct WebGPUSvmPath {
  WebGPUSvmContextWords bounces;
  uint transparent_bounce;
};

ccl_device_inline int integrator_state_bounce(const WebGPUSvmPath &state, uint32_t /*flags*/)
{
  return int(state.bounces.x);
}

ccl_device_inline int integrator_state_diffuse_bounce(const WebGPUSvmPath &state, uint32_t /*flags*/)
{
  return int(state.bounces.y);
}

ccl_device_inline int integrator_state_glossy_bounce(const WebGPUSvmPath &state, uint32_t /*flags*/)
{
  return int(state.bounces.z);
}

ccl_device_inline int integrator_state_transmission_bounce(const WebGPUSvmPath &state,
                                                            uint32_t /*flags*/)
{
  return int(state.bounces.w);
}

ccl_device_inline int integrator_state_transparent_bounce(const WebGPUSvmPath &state,
                                                           uint32_t /*flags*/)
{
  return int(state.transparent_bounce & CYCLES_WEBGPU_TRANSPARENT_BOUNCE_MASK);
}

ccl_device_inline int integrator_state_portal_bounce(KernelGlobals /*kg*/,
                                                     const WebGPUSvmPath &state,
                                                     uint32_t /*flags*/)
{
  return int(state.transparent_bounce >> CYCLES_WEBGPU_PORTAL_BOUNCE_SHIFT);
}

ccl_device_inline float3 webgpu_svm_context_xyz(const WebGPUSvmContextWords bits)
{
  return make_float3(__uint_as_float(bits.x), __uint_as_float(bits.y), __uint_as_float(bits.z));
}

ccl_device_inline float4 webgpu_svm_context_xyzw(const WebGPUSvmContextWords bits)
{
  return make_float4(__uint_as_float(bits.x),
                     __uint_as_float(bits.y),
                     __uint_as_float(bits.z),
                     __uint_as_float(bits.w));
}

template<typename FetchSurface>
ccl_device_inline Transform webgpu_svm_context_transform(FetchSurface surface, const uint first)
{
  return {webgpu_svm_context_xyzw(surface(first)),
          webgpu_svm_context_xyzw(surface(first + 1)),
          webgpu_svm_context_xyzw(surface(first + 2))};
}

template<typename FetchSurface>
ccl_device_inline float3 webgpu_svm_camera_position(FetchSurface surface)
{
  const Transform world_to_camera = webgpu_svm_context_transform(
      surface, CYCLES_WEBGPU_SVM_WORLD_TO_CAMERA);
  return transform_point(&world_to_camera,
                         webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_POSITION)));
}

template<typename FetchSurface>
ccl_device_inline float3 webgpu_svm_window_position(FetchSurface surface)
{
  ProjectionTransform world_to_ndc;
  world_to_ndc.x = webgpu_svm_context_xyzw(surface(CYCLES_WEBGPU_SVM_WORLD_TO_NDC));
  world_to_ndc.y = webgpu_svm_context_xyzw(surface(CYCLES_WEBGPU_SVM_WORLD_TO_NDC + 1));
  world_to_ndc.z = webgpu_svm_context_xyzw(surface(CYCLES_WEBGPU_SVM_WORLD_TO_NDC + 2));
  world_to_ndc.w = webgpu_svm_context_xyzw(surface(CYCLES_WEBGPU_SVM_WORLD_TO_NDC + 3));
  const float3 position = webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_POSITION));
  float3 window = transform_perspective(&world_to_ndc, position);
  window.z = 0.0f;
  return window;
}

template<typename FetchSurface>
ccl_device_inline void webgpu_svm_shader_view(FetchSurface surface, ShaderData &sd)
{
  sd.N = webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_NORMAL));
  sd.wi = -webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_DIRECTION));
  sd.runtime_flag = surface(CYCLES_WEBGPU_SVM_INTERSECTION).w == 2 ? SR_BACKFACING : 0;
}

CCL_NAMESPACE_END

#include "kernel/svm/fresnel.h"
#include "kernel/svm/light_path.h"
