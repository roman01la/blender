/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "util/defines.h"

#include <array>
#include <cstddef>
#include <cstdint>

CCL_NAMESPACE_BEGIN

struct WebGPUOIDNImage {
  float *pixels = nullptr;
  size_t pixel_stride = 0;
  size_t row_stride = 0;
};

struct WebGPUOIDNRequest {
  const char *model = nullptr;
  int width = 0;
  int height = 0;
  bool hdr = false;
  bool srgb = false;
  WebGPUOIDNImage color, albedo, normal, output;
};

class Device;
class device_memory;

struct WebGPUOIDNFilmTile {
  uint32_t id, width, height, pass_stride, source_width, source_x, source_y, target_x, target_y,
      tile_width, tile_height;
};

struct WebGPUOIDNFilmOutput {
  uint32_t id, samples, sample_count, noisy, denoised, albedo, normal;
};

bool cycles_webgpu_oidn_append_tile(Device &device,
                                   const device_memory &buffer,
                                   const WebGPUOIDNFilmTile &tile);
bool cycles_webgpu_oidn_finish_film(const WebGPUOIDNFilmOutput &film,
                                    const std::array<const char *, 3> &models,
                                    float *pixels,
                                    std::array<double, 3> &times);
void cycles_webgpu_oidn_drop_film(uint32_t id);

const char *cycles_webgpu_oidn_model(bool color,
                                     bool albedo,
                                     bool normal,
                                     bool clean_aux,
                                     bool high,
                                     bool fast);

inline constexpr char webgpu_oidn_frame_prefix[] = "cycles-webgpu-resident:";

bool cycles_webgpu_oidn_available();
bool cycles_webgpu_oidn_filter(const WebGPUOIDNRequest &request);

CCL_NAMESPACE_END
