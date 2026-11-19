/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "device/webgpu/preview.h"
#include "integrator/path_trace_work_cpu.h"
#include "integrator/path_trace_work_gpu.h"
#include "util/scoped_defer.h"
#include "util/vector.h"

CCL_NAMESPACE_BEGIN

/* The stock GPU scheduler owns tile admission and kernel selection. This work maps
 * the selected kernels onto WebGPU's packed path state and device-resident film. */
class PathTraceWorkWebGPU final : public PathTraceWorkGPU {
 public:
  PathTraceWorkWebGPU(Device *device,
                      Film *film,
                      DeviceScene *device_scene,
                      const bool *cancel_requested_flag);

  void prepare_scene(const Scene &scene, bool scene_changed);
  void alloc_work_memory() override;
  void init_execution() override;
  void deinit_execution() override;
  void render_samples(RenderStatistics &statistics,
                      int start_sample,
                      int samples_num,
                      int sample_offset) override;
  void copy_to_display(PathTraceDisplay *display, PassMode pass_mode, int num_samples) override;
  bool copy_render_buffers_from_device() override;
  bool append_resident_film(uint32_t id, const BufferParams &full_params);
  bool has_resident_film() const;
  void use_exported_film();
  bool copy_render_buffers_to_device() override;
  bool zero_render_buffers() override;
  int adaptive_sampling_converge_filter_count_active(float threshold, bool reset) override;
  void cryptomatte_postproces() override;
  void denoise_volume_guiding_buffers() override;
#if defined(WITH_PATH_GUIDING)
  void guiding_init_kernel_globals(void *guiding_field,
                                   void *sample_data_storage,
                                   bool train) override;
#endif

 protected:
  void enqueue_reset() override;
  void compact_main_paths(int num_active_paths) override;
  void enqueue_work_tile_kernel(DeviceKernel kernel,
                                int num_work_tiles,
                                int max_tile_work_size) override;
  void enqueue_path_iteration(DeviceKernel kernel, int num_paths_limit) override;
  bool update_queue_counter_and_cache() override;
  bool kernel_creates_shadow_paths(DeviceKernel kernel) override;

 private:
  class CPUWork : public PathTraceWorkCPU {
   public:
    using PathTraceWorkCPU::PathTraceWorkCPU;

    template<typename Operation>
    decltype(auto) with_buffers(unique_ptr<RenderBuffers> &buffers, Operation &&operation)
    {
      buffers_.swap(buffers);
      SCOPED_DEFER(buffers_.swap(buffers));
      return operation(*this);
    }
  };

  template<typename Operation> decltype(auto) with_cpu_work(Operation &&operation)
  {
    cpu_work_.set_effective_buffer_params(effective_big_tile_params_,
                                          effective_buffer_params_,
                                          effective_denoised_big_tile_params_,
                                          effective_denoised_buffer_params_);
    return cpu_work_.with_buffers(buffers_, operation);
  }

  bool render_on_gpu(RenderStatistics &statistics, int start_sample, int samples_num);
  bool replay_gpu_samples_on_cpu();

  struct AdaptiveCheckpoint {
    int end_sample;
    float threshold;
    bool reset;
  };

  WebGPUPreview renderer_;
  CPUWork cpu_work_;
  vector<AdaptiveCheckpoint> adaptive_checkpoints_;
  int gpu_completed_samples_ = 0;
  bool kernel_failed_ = false;
  bool samples_overlap_ = false;
  bool gpu_film_dirty_ = false;
  bool cpu_execution_active_ = false;
  bool use_cpu_ = false;
};

CCL_NAMESPACE_END
