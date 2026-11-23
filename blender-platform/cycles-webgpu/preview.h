/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "device/kernel.h"
#include "device/webgpu/material_layout.h"
#include "device/webgpu/scene.h"
#include "util/unique_ptr.h"

#include <array>
#include <span>
#include <vector>

CCL_NAMESPACE_BEGIN

class BufferParams;
class Device;
class RenderBuffers;
class Scene;
class WebGPURenderPipeline;
struct WebGPUOIDNFilmTile;
struct IntegratorQueueCounter;
struct KernelWorkTile;

class WebGPUPreview {
 public:
  explicit WebGPUPreview(Device &device);
  ~WebGPUPreview();

  void prepare(const Scene &scene, bool scene_changed);
  void reset();
  bool uses_volume() const
  {
    return volume_;
  }
  bool begin_batch(const BufferParams &params,
                   int start_sample,
                   int samples_num,
                   uint32_t &path_capacity,
                   size_t film_words);
  bool enqueue_camera(std::span<const KernelWorkTile> tiles);
  bool enqueue_stage(DeviceKernel stage);
  bool read_queue(IntegratorQueueCounter &counts, bool &samples_overlap);
  bool finish_batch();
  bool adaptive_filter(float threshold, bool reset, int &active_pixels);
  bool publish_preview(const BufferParams &params, float exposure, int samples, uint32_t ticket);
  bool read_film(RenderBuffers &buffers, const BufferParams &params);
  bool append_resident_film(const WebGPUOIDNFilmTile &tile);

 private:
  bool initialize();
  bool report_error();

  Device &device_;
  unique_ptr<WebGPURenderPipeline> pipeline_;
  CyclesWebGPUScene scene_;
  std::array<float, CYCLES_WEBGPU_CAMERA_WORDS> camera_{};
  std::array<float, CYCLES_WEBGPU_CAMERA_LENS_WORDS> camera_lens_{};
  std::array<uint32_t, 16> denoising_config_{};
  std::vector<uint32_t> sampling_;
  bool scene_checked_ = false;
  bool scene_supported_ = false;
  bool volume_ = false;
  bool scene_pending_ = false;
  bool reset_pending_ = true;
  bool failed_ = false;
};

CCL_NAMESPACE_END
