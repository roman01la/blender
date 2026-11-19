/* SPDX-License-Identifier: Apache-2.0 */

StructuredBuffer<uint> work_tile : register(t0);
StructuredBuffer<float4> camera : register(t1);
RWStructuredBuffer<uint4> pixel_samples : register(u2);
#ifdef CYCLES_PRIMARY_RAY_QUEUE
#  include "../blender-platform/cycles-webgpu/path_state_layout.h"
RWStructuredBuffer<uint> ray_words : register(u3);
StructuredBuffer<uint> camera_sampling : register(t4);

uint cycles_camera_sample_word(uint index)
{
  return camera_sampling[index];
}
#else
RWStructuredBuffer<float4> rays : register(u3);
#endif

extern uint4 cycles_work_pixel(uint x,
                               uint y,
                               uint w,
                               uint h,
                               uint start_sample,
                               uint num_samples,
                               uint work_index,
                               float scrambling_distance);

extern void cycles_camera_ray(uint4 pixel,
                              float4 projection_x,
                              float4 projection_y,
                              float4 projection_z,
                              float4 projection_w,
                              float4 world_x,
                              float4 world_y,
                              float4 world_z,
                              float4 clip,
                              out float4 origin,
                              out float4 direction,
                              out uint rng_pixel);

#ifdef CYCLES_WEBGPU_ADAPTIVE_CAMERA_RAYS
extern uint cycles_adaptive_sample_class(uint sample);
#endif

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
  const float4 clip = camera[7];
#if defined(CYCLES_WEBGPU_ADAPTIVE_CAMERA_RAYS)
  const uint path_queue = 1u + 12u * ray_words[0];
  if (id.x >= work_tile[2] || id.x >= ray_words[path_queue]) {
    return;
  }
  const uint index = ray_words[path_queue + 1u + id.x];
  const uint4 work_pixel = pixel_samples[index];
  if (work_pixel.w == 0xffffffffu) {
    return;
  }
  const uint visibility = work_tile[3];
#elif defined(CYCLES_WEBGPU_STOCK_CAMERA)
  const uint max_tile_size = work_tile[1];
  const uint tile = id.x / max_tile_size;
  if (tile >= work_tile[0]) {
    return;
  }
  const uint base_word = 4u + 8u * tile;
  const uint offset = id.x - tile * max_tile_size;
  if (offset >= work_tile[base_word + 7u]) {
    return;
  }
  // The stock tile offset is relative to this admission. Each candidate is
  // mapped to a free physical slot without moving any live path state.
  const uint admission_index = work_tile[base_word + 6u] + offset;
  const uint path_queue = 1u + 12u * ray_words[0];
  if (admission_index >= ray_words[path_queue]) {
    return;
  }
  const uint index = ray_words[path_queue + 1u + admission_index];
  const uint visibility = work_tile[3];
  uint4 work_pixel = cycles_work_pixel(
      work_tile[base_word], work_tile[base_word + 1u], work_tile[base_word + 2u],
      work_tile[base_word + 3u], work_tile[base_word + 4u], work_tile[base_word + 5u],
      offset, clip.z);
  // Track the nominal scheduler sample alongside the effective RNG sample.
  work_pixel.w = work_pixel.z;
#else
  const uint index = id.x;
  if (index >= work_tile[6]) {
    return;
  }
  const uint ray_count = work_tile[6];
  const uint visibility = work_tile[7];
  const uint4 work_pixel = cycles_work_pixel(
      work_tile[0], work_tile[1], work_tile[2], work_tile[3],
      work_tile[4], work_tile[5], index, clip.z);
#endif

#ifdef CYCLES_WEBGPU_ADAPTIVE_CAMERA_SLOTS
  pixel_samples[index] = work_pixel;
#else
  float4 origin, direction;
  uint rng_pixel;
  cycles_camera_ray(
      work_pixel, camera[0], camera[1], camera[2], camera[3],
      camera[4], camera[5], camera[6], clip, origin, direction, rng_pixel);

  pixel_samples[index] = work_pixel;
#ifdef CYCLES_PRIMARY_RAY_QUEUE
#  ifndef CYCLES_WEBGPU_STOCK_CAMERA
  // The standalone BVH dispatch reads this buffer after the camera completes.
  if (index == 0) {
    ray_words[0] = ray_count;
  }
#  endif
  const uint base = 1 + 12 * index;
#  ifdef CYCLES_WEBGPU_ADAPTIVE_CAMERA_RAYS
  // The per-slot class follows the compact queue at the tail of the ray buffer.
  ray_words[2u + 13u * ray_words[0] + index] = cycles_adaptive_sample_class(work_pixel.z);
#  endif
  const uint4 start = __builtin_bit_cast(uint4, origin);
  const uint4 end = __builtin_bit_cast(uint4, direction);
  ray_words[base] = start.x;
  ray_words[base + 1] = start.y;
  ray_words[base + 2] = start.z;
  ray_words[base + 3] = start.w;
  ray_words[base + 4] = end.x;
  ray_words[base + 5] = end.y;
  ray_words[base + 6] = end.z;
  ray_words[base + 7] = end.w;
  ray_words[base + 8] = visibility;
  ray_words[base + 9] = 0xffffffffu;
  ray_words[base + 10] = 0xffffffffu;
#  ifdef CYCLES_WEBGPU_STOCK_CAMERA
  ray_words[base + CYCLES_WEBGPU_RAY_RNG_PIXEL] = rng_pixel;
#  else
  ray_words[base + CYCLES_WEBGPU_RAY_RNG_PIXEL] = 0u;
#  endif
#else
  rays[2 * index] = origin;
  rays[2 * index + 1] = direction;
#endif
#endif
}
