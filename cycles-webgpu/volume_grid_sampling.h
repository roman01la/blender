/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "../blender-platform/cycles-webgpu/volume_grid_layout.h"
#include "kernel/sample/lcg.h"
#include "util/transform.h"
#define __KERNEL_GPU__
#include "util/types_image.h"
#undef __KERNEL_GPU__

/* Generated directly from Cycles kernel/util/image_3d.h. */
#include "volume_grid_filters.h"

__attribute__((pure)) unsigned int cycles_volume_word(unsigned int index);

CCL_NAMESPACE_BEGIN

ccl_device_inline float grid_float_word(const uint index)
{
  return std::bit_cast<float>(cycles_volume_word(index));
}

ccl_device_inline AttributeDescriptor volume_grid_attribute(const uint base, const int id)
{
  if (base != 0) {
    for (uint slot = 0; slot < CYCLES_WEBGPU_GRID_ATTRIBUTE_COUNT; ++slot) {
      const uint descriptor = base + CYCLES_WEBGPU_GRID_ATTRIBUTES +
                              slot * CYCLES_WEBGPU_GRID_ATTRIBUTE_WORDS;
      if (cycles_volume_word(descriptor + CYCLES_WEBGPU_GRID_CHANNELS) != 0 &&
          cycles_volume_word(descriptor + CYCLES_WEBGPU_GRID_ATTRIBUTE_ID) == uint(id))
      {
        return {ATTR_ELEMENT_VOXEL, NODE_ATTR_FLOAT, int(descriptor)};
      }
    }
  }
  return {ATTR_ELEMENT_NONE, NODE_ATTR_FLOAT, 0};
}

ccl_device_inline AttributeDescriptor find_attribute(KernelGlobals kg, const ShaderData *, int id)
{
  return volume_grid_attribute(kg->grid_offset, id);
}

struct WebGPUGridAccessor {
  uint voxels;
  uint channels;
  int3 origin;
  int3 extent;
  float4 background;

  ccl_device_inline_method float4 getValue(const int3 index) const
  {
    const int3 p = index - origin;
    if (p.x < 0 || p.y < 0 || p.z < 0 || p.x >= extent.x || p.y >= extent.y ||
        p.z >= extent.z)
    {
      return background;
    }
    const uint voxel = (uint(p.z) * uint(extent.y) + uint(p.y)) * uint(extent.x) + uint(p.x);
    const uint offset = voxels + channels * voxel;
    const float x = grid_float_word(offset);
    if (channels == 1) {
      return make_float4(x, x, x, 1.0f);
    }
    return make_float4(x,
                       grid_float_word(offset + 1),
                       grid_float_word(offset + 2),
                       channels == 4 ? grid_float_word(offset + 3) : 1.0f);
  }
};

ccl_device_inline float4 volume_grid_read_row(const uint index)
{
  return make_float4(grid_float_word(index),
                     grid_float_word(index + 1),
                     grid_float_word(index + 2),
                     grid_float_word(index + 3));
}

ccl_device_inline int3 volume_grid_read_coord(const uint index)
{
  return make_int3(std::bit_cast<int>(cycles_volume_word(index)),
                   std::bit_cast<int>(cycles_volume_word(index + 1)),
                   std::bit_cast<int>(cycles_volume_word(index + 2)));
}

ccl_device_inline float4 volume_grid_sample(const uint base,
                                             ccl_private ShaderData *sd,
                                             const uint at,
                                             const bool stochastic)
{
  const uint start = cycles_volume_word(at + CYCLES_WEBGPU_GRID_VOXELS);
  if (start == 0) {
    return zero_float4();
  }
  const uint channels = cycles_volume_word(at + CYCLES_WEBGPU_GRID_CHANNELS);
  float4 background = volume_grid_read_row(at + CYCLES_WEBGPU_GRID_BACKGROUND);
  if (channels == 1) {
    background = make_float4(background.x, background.x, background.x, 1.0f);
  }
  else if (channels == 3) {
    background.w = 1.0f;
  }
  WebGPUGridAccessor voxels{base + start,
                           channels,
                           volume_grid_read_coord(at + CYCLES_WEBGPU_GRID_ORIGIN),
                           volume_grid_read_coord(at + CYCLES_WEBGPU_GRID_SIZE),
                           background};
  const uint transform = at + CYCLES_WEBGPU_GRID_TRANSFORM;
  const Transform world_to_index{volume_grid_read_row(transform),
                                 volume_grid_read_row(transform + 4),
                                 volume_grid_read_row(transform + 8)};
  float3 P = transform_point(&world_to_index, sd->P);
  auto interpolation = InterpolationType(cycles_volume_word(at + CYCLES_WEBGPU_GRID_INTERPOLATION));
  if (stochastic) {
    float3 rand = lcg_step_float3(&sd->lcg_state);
    P = interp_stochastic(P, interpolation, rand);
  }
  if (interpolation == INTERPOLATION_CLOSEST) {
    return voxels.getValue(make_int3(floor(P)));
  }
  if (interpolation == INTERPOLATION_LINEAR) {
    return kernel_image_interp_trilinear_nanovdb<float4>(voxels, P);
  }
  return kernel_image_interp_tricubic_nanovdb<float4>(voxels, P);
}

template<typename T>
ccl_device_inline T primitive_volume_attribute(KernelGlobals kg,
                                                ShaderData *sd,
                                                const AttributeDescriptor &desc,
                                                const bool stochastic)
{
  float4 value = volume_grid_sample(kg->grid_offset, sd, uint(desc.offset), stochastic);
  if (value.w > 1e-6f && value.w != 1.0f) {
    value = make_float4(make_float3(value) / value.w, value.w);
  }
  return volume_attribute_value<T>(value);
}

CCL_NAMESPACE_END
