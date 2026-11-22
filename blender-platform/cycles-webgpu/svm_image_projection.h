/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

CCL_NAMESPACE_BEGIN

/* The seven regions of svm_node_tex_image_box(), separated from the kernel
 * texture fetch so the WebGPU image buffer can supply each face. */
ccl_device_inline float3 webgpu_svm_image_box_weights(const float3 object_normal,
                                                      const float blend)
{
  float3 N = fabs(object_normal);
  N /= N.x + N.y + N.z;
  const float limit = 0.5f * (1.0f + blend);

  if (N.x > limit * (N.x + N.y) && N.x > limit * (N.x + N.z)) {
    return make_float3(1.0f, 0.0f, 0.0f);
  }
  if (N.y > limit * (N.x + N.y) && N.y > limit * (N.y + N.z)) {
    return make_float3(0.0f, 1.0f, 0.0f);
  }
  if (N.z > limit * (N.x + N.z) && N.z > limit * (N.y + N.z)) {
    return make_float3(0.0f, 0.0f, 1.0f);
  }
  if (blend > 0.0f) {
    const float midpoint = 0.5f * (1.0f - blend);
    if (N.z < (1.0f - limit) * (N.y + N.x)) {
      const float x = saturatef((N.x / (N.x + N.y) - midpoint) / blend);
      return make_float3(x, 1.0f - x, 0.0f);
    }
    if (N.x < (1.0f - limit) * (N.y + N.z)) {
      const float y = saturatef((N.y / (N.y + N.z) - midpoint) / blend);
      return make_float3(0.0f, y, 1.0f - y);
    }
    if (N.y < (1.0f - limit) * (N.x + N.z)) {
      const float x = saturatef((N.x / (N.x + N.z) - midpoint) / blend);
      return make_float3(x, 0.0f, 1.0f - x);
    }
    return ((2.0f - limit) * N + make_float3(limit - 1.0f)) / (2.0f * limit - 1.0f);
  }
  return make_float3(1.0f, 0.0f, 0.0f);
}

ccl_device_inline float2 webgpu_svm_image_box_uv(const float3 co,
                                                 const float3 object_normal,
                                                 const uint axis)
{
  if (axis == 0) {
    return make_float2(object_normal.x < 0.0f ? 1.0f - co.y : co.y, co.z);
  }
  if (axis == 1) {
    return make_float2(object_normal.y > 0.0f ? 1.0f - co.x : co.x, co.z);
  }
  return make_float2(object_normal.z > 0.0f ? 1.0f - co.y : co.y, co.x);
}

CCL_NAMESPACE_END
