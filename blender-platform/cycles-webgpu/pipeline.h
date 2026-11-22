/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "device/kernel.h"
#include "device/webgpu/material_layout.h"
#include "util/unique_ptr.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

CCL_NAMESPACE_BEGIN

class Device;
class BufferParams;
struct WebGPUOIDNFilmTile;

struct IntegratorQueueCounter;
struct KernelWorkTile;

struct WebGPURenderShaders {
  std::string_view camera, bvh2, surface, film, state_init, wavefront_queue;
  std::string (*load_shader)(const char *name) = nullptr;
};

/** Progressive Cycles rendering for the experimental WebGPU device. Each pipeline owns the
 * registered kernel slots on its device. All calls, including destruction, belong to a render
 * pthread distinct from the WebGPU owner. The device must outlive the pipeline. */
class WebGPURenderPipeline {
 public:
  /* Keep one full HD slice within the film binding budget while camera work is tiled by the
   * scheduler. A 64-thread adaptive dispatch needs at most 32,768 workgroups. */
  static constexpr uint32_t max_crop_pixels = 2 * 1024 * 1024;
  static constexpr uint32_t max_path_capacity = 64 * 1024;

  explicit WebGPURenderPipeline(Device &device);
  ~WebGPURenderPipeline();

  bool initialize(const WebGPURenderShaders &shaders);
  /** Upload packed scene bindings and immutable camera sampling tables. */
  bool upload_scene(std::span<const uint32_t> bvh,
                    std::span<const uint32_t> surface,
                    std::span<const uint32_t> light,
                    std::span<const uint32_t> svm,
                    std::span<const uint32_t> sampling,
                    bool has_subsurface,
                    bool advanced_query_svm);
  /** Reset the GPU film and return the allocated capacity for the stock path scheduler. */
  bool reset(std::span<const float, CYCLES_WEBGPU_CAMERA_WORDS> camera,
             std::span<const uint32_t, 16> denoising_config,
             const BufferParams &params,
             size_t film_words,
             uint32_t &path_capacity);
  bool begin_batch(int start_sample, int samples_num);
  /** Execute stages selected by PathTraceWorkGPU and return its current queue counters. */
  bool enqueue_camera(std::span<const KernelWorkTile> tiles);
  bool enqueue_stage(DeviceKernel stage);
  bool read_queue(IntegratorQueueCounter &counts, bool &samples_overlap);
  bool finish_batch();
  bool adaptive_filter(float threshold, bool reset, int &active_pixels);
  bool publish_preview(const BufferParams &params, float exposure, int samples, uint32_t ticket);
  /** Copy the native film to Cycles at its output boundary. */
  bool read_film(std::span<float> film);
  bool append_resident_film(const WebGPUOIDNFilmTile &tile);
  std::string_view error_message() const;

 private:
  struct Impl;
  unique_ptr<Impl> impl_;
};

CCL_NAMESPACE_END
