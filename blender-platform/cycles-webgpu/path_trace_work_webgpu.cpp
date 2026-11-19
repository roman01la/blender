/* SPDX-License-Identifier: Apache-2.0 */

#include "device/webgpu/path_trace_work_webgpu.h"

#include "device/webgpu/pipeline.h"
#include "device/webgpu/oidn.h"
#include "integrator/path_trace_display.h"
#include "scene/scene.h"
#include "session/display_driver.h"
#include "util/log.h"

#include <emscripten/threading.h>
#include <limits>
#include <span>
#include <utility>

CCL_NAMESPACE_BEGIN

PathTraceWorkWebGPU::PathTraceWorkWebGPU(Device *device,
                                         Film *film,
                                         DeviceScene *device_scene,
                                         const bool *cancel_requested_flag)
    : PathTraceWorkGPU(device, film, device_scene, cancel_requested_flag),
      renderer_(*device),
      cpu_work_(device, film, device_scene, cancel_requested_flag)
{
}

void PathTraceWorkWebGPU::prepare_scene(const Scene &scene, const bool scene_changed)
{
  renderer_.prepare(scene, scene_changed);
}

void PathTraceWorkWebGPU::alloc_work_memory()
{
  if (integrator_queue_counter_.size() == 0) {
    integrator_queue_counter_.alloc(1);
  }
}

void PathTraceWorkWebGPU::init_execution()
{
  cpu_work_.init_execution();
  cpu_execution_active_ = true;
}

void PathTraceWorkWebGPU::deinit_execution()
{
  cpu_work_.deinit_execution();
  cpu_execution_active_ = false;
}

void PathTraceWorkWebGPU::render_samples(RenderStatistics &statistics,
                                         int start_sample,
                                         int samples_num,
                                         const int sample_offset)
{
  if (samples_num == 0) {
    return;
  }
  if (!use_cpu_ && sample_offset == 0 && render_on_gpu(statistics, start_sample, samples_num)) {
    gpu_completed_samples_ = start_sample + samples_num;
    gpu_film_dirty_ = true;
    LOG_INFO << "Cycles WebGPU stock scheduler completed a " << effective_buffer_params_.width
             << " x " << effective_buffer_params_.height << " tile at sample " << start_sample
             << " (" << samples_num << " samples)"
             << (samples_overlap_ ? " with overlapping samples" : "");
    return;
  }
  if (is_cancel_requested()) {
    return;
  }

  if (!use_cpu_) {
    LOG_INFO << "Cycles WebGPU using CPU for a " << effective_buffer_params_.width << " x "
             << effective_buffer_params_.height << " tile at sample " << start_sample;
    if (gpu_completed_samples_ > 0 && !replay_gpu_samples_on_cpu()) {
      return;
    }
    use_cpu_ = true;
  }
  with_cpu_work([&](CPUWork &work) {
    work.render_samples(statistics, start_sample, samples_num, sample_offset);
  });
}

bool PathTraceWorkWebGPU::render_on_gpu(RenderStatistics &statistics,
                                        const int start_sample,
                                        const int samples_num)
{
  const BufferParams &params = effective_buffer_params_;
  if (is_cancel_requested() || params.width <= 0 || params.height <= 0) {
    return false;
  }
  const uint64_t pixels = uint64_t(params.width) * params.height;
  if (pixels > WebGPURenderPipeline::max_crop_pixels) {
    return false;
  }

  uint32_t capacity = uint32_t(max_num_paths_);
  if (!renderer_.begin_batch(
          params, start_sample, samples_num, capacity, buffers_->buffer.data_size))
  {
    return false;
  }
  max_num_paths_ = int(capacity);
  min_num_active_main_paths_ = max_num_paths_ / 4;

  PathTraceWorkGPU::render_samples(statistics, start_sample, samples_num, 0);
  return !kernel_failed_ && !is_cancel_requested() && renderer_.finish_batch();
}

void PathTraceWorkWebGPU::enqueue_reset()
{
  *integrator_queue_counter_.data() = {};
  max_active_main_path_index_ = 0;
  kernel_failed_ = false;
  samples_overlap_ = false;
}

void PathTraceWorkWebGPU::compact_main_paths(const int num_active_paths)
{
  /* Stock tile offsets describe the dense logical path range. The camera
   * dispatch maps newly admitted paths onto free physical WebGPU slots. */
  max_active_main_path_index_ = num_active_paths;
}

void PathTraceWorkWebGPU::enqueue_work_tile_kernel(const DeviceKernel kernel,
                                                   const int num_work_tiles,
                                                   const int /*max_tile_work_size*/)
{
  kernel_failed_ = kernel != DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA ||
                   !renderer_.enqueue_camera(std::span(work_tiles_.data(), num_work_tiles));
}

void PathTraceWorkWebGPU::enqueue_path_iteration(const DeviceKernel kernel,
                                                 const int /*num_paths_limit*/)
{
  kernel_failed_ = !renderer_.enqueue_stage(kernel);
}

bool PathTraceWorkWebGPU::update_queue_counter_and_cache()
{
  bool samples_overlap = false;
  kernel_failed_ = kernel_failed_ ||
                   !renderer_.read_queue(*integrator_queue_counter_.data(), samples_overlap);
  samples_overlap_ |= samples_overlap;
  return !kernel_failed_;
}

bool PathTraceWorkWebGPU::kernel_creates_shadow_paths(const DeviceKernel /*kernel*/)
{
  /* WebGPU traces opaque shadows within the same surface iteration. */
  return false;
}

void PathTraceWorkWebGPU::copy_to_display(PathTraceDisplay *display,
                                          const PassMode pass_mode,
                                          const int num_samples)
{
  if (!use_cpu_ &&
      display->graphics_interop_get_device().type == GraphicsInteropDevice::WEBGPU)
  {
    if (!has_resident_film()) {
      return;
    }

    const PassAccessor::PassAccessInfo pass = get_display_pass_access_info(pass_mode);
    const BufferParams &params = effective_buffer_params_;
    const int2 size = display->get_texture_size();
    if (pass.type != PASS_COMBINED || pass.mode != PassMode::NOISY ||
        params.window_width != size.x || params.window_height != size.y ||
        get_display_destination_template(display, pass_mode).offset != 0)
    {
      return;
    }

    GraphicsInteropBuffer &interop = display->graphics_interop_get_buffer();
    if (interop.has_new_handle()) {
      const int64_t ticket = interop.take_handle();
      if (ticket > 0 && ticket <= std::numeric_limits<uint32_t>::max() &&
          !renderer_.publish_preview(
              params, device_scene_->data.film.exposure, num_samples, uint32_t(ticket)))
      {
        LOG_WARNING << "Cycles WebGPU could not publish the progressive preview";
      }
    }
    return;
  }

  if (copy_render_buffers_from_device()) {
    with_cpu_work([&](CPUWork &work) { work.copy_to_display(display, pass_mode, num_samples); });
  }
}

bool PathTraceWorkWebGPU::has_resident_film() const
{
  return !use_cpu_ && gpu_completed_samples_ > 0;
}

bool PathTraceWorkWebGPU::append_resident_film(const uint32_t id, const BufferParams &full_params)
{
  const BufferParams &source = effective_buffer_params_;
  const int x = source.full_x - full_params.full_x + source.window_x;
  const int y = source.full_y - full_params.full_y + source.window_y;
  if (!has_resident_film() || source.pass_stride != full_params.pass_stride || x < 0 || y < 0 ||
      source.window_x < 0 || source.window_y < 0 ||
      x + source.window_width > full_params.width ||
      y + source.window_height > full_params.height)
  {
    return false;
  }

  const WebGPUOIDNFilmTile tile{id,
                               uint32_t(full_params.width),
                               uint32_t(full_params.height),
                               uint32_t(full_params.pass_stride),
                               uint32_t(source.width),
                               uint32_t(source.window_x),
                               uint32_t(source.window_y),
                               uint32_t(x),
                               uint32_t(y),
                               uint32_t(source.window_width),
                               uint32_t(source.window_height)};
  if (!renderer_.append_resident_film(tile)) {
    return false;
  }
  LOG_INFO << "Cycles WebGPU resident film tile " << tile.tile_width << " x "
           << tile.tile_height << " at " << x << "," << y;
  return true;
}

void PathTraceWorkWebGPU::use_exported_film()
{
  gpu_film_dirty_ = false;
}

bool PathTraceWorkWebGPU::copy_render_buffers_from_device()
{
  if (!gpu_film_dirty_) {
    return buffers_->copy_from_device();
  }
  if (emscripten_is_main_runtime_thread()) {
    /* Wait for GPU readback only from a render pthread. */
    return false;
  }
  if (renderer_.read_film(*buffers_, effective_buffer_params_)) {
    gpu_film_dirty_ = false;
    return true;
  }
  return replay_gpu_samples_on_cpu();
}

bool PathTraceWorkWebGPU::replay_gpu_samples_on_cpu()
{
  if (is_cancel_requested()) {
    return false;
  }

  const int end_sample = std::exchange(gpu_completed_samples_, 0);
  LOG_INFO << "Cycles WebGPU using CPU to replay " << end_sample << " completed samples";
  gpu_film_dirty_ = false;
  use_cpu_ = true;
  buffers_->zero();
  const bool needs_init = !cpu_execution_active_;
  if (needs_init) {
    init_execution();
  }
  SCOPED_DEFER(if (needs_init) { deinit_execution(); });
  RenderStatistics statistics;
  with_cpu_work([&](CPUWork &work) {
    /* Replay earlier convergence checks at the same sample boundaries so
     * the CPU camera skips pixels that have already converged. The scheduler
     * can also request multiple checks at the same sample with lower thresholds. */
    int start_sample = 0;
    for (const AdaptiveCheckpoint &checkpoint : adaptive_checkpoints_) {
      if (checkpoint.end_sample > start_sample) {
        work.render_samples(statistics, start_sample, checkpoint.end_sample - start_sample, 0);
        start_sample = checkpoint.end_sample;
      }
      if (is_cancel_requested()) {
        return;
      }
      work.adaptive_sampling_converge_filter_count_active(checkpoint.threshold, checkpoint.reset);
    }
    if (end_sample > start_sample && !is_cancel_requested()) {
      work.render_samples(statistics, start_sample, end_sample - start_sample, 0);
    }
  });
  adaptive_checkpoints_.clear();
  return !is_cancel_requested();
}

bool PathTraceWorkWebGPU::copy_render_buffers_to_device()
{
  if (gpu_completed_samples_ > 0) {
    /* A stock caller supplied replacement host film. Subsequent CPU work
     * can continue from it without discarding or mixing its accumulated sums. */
    adaptive_checkpoints_.clear();
    gpu_completed_samples_ = 0;
    gpu_film_dirty_ = false;
    use_cpu_ = true;
  }
  buffers_->copy_to_device();
  return true;
}

bool PathTraceWorkWebGPU::zero_render_buffers()
{
  renderer_.reset();
  adaptive_checkpoints_.clear();
  use_cpu_ = false;
  gpu_completed_samples_ = 0;
  gpu_film_dirty_ = false;
  buffers_->zero();
  return true;
}

int PathTraceWorkWebGPU::adaptive_sampling_converge_filter_count_active(const float threshold,
                                                                        const bool reset)
{
  if (!use_cpu_ && gpu_completed_samples_ > 0) {
    int active_pixels;
    if (renderer_.adaptive_filter(threshold, reset, active_pixels)) {
      adaptive_checkpoints_.push_back({gpu_completed_samples_, threshold, reset});
      gpu_film_dirty_ = true;
      LOG_INFO << "Cycles WebGPU adaptive convergence on " << effective_buffer_params_.width
               << " x " << effective_buffer_params_.height << ": " << active_pixels
               << " active pixels at threshold " << threshold << " after sample "
               << gpu_completed_samples_;
      return active_pixels;
    }
    if (!replay_gpu_samples_on_cpu()) {
      return 0;
    }
  }
  return with_cpu_work([&](CPUWork &work) {
    return work.adaptive_sampling_converge_filter_count_active(threshold, reset);
  });
}

void PathTraceWorkWebGPU::cryptomatte_postproces()
{
  with_cpu_work([](CPUWork &work) { work.cryptomatte_postproces(); });
}

void PathTraceWorkWebGPU::denoise_volume_guiding_buffers()
{
  with_cpu_work([](CPUWork &work) { work.denoise_volume_guiding_buffers(); });
}

#if defined(WITH_PATH_GUIDING)
void PathTraceWorkWebGPU::guiding_init_kernel_globals(void *guiding_field,
                                                      void *sample_data_storage,
                                                      const bool train)
{
  cpu_work_.guiding_init_kernel_globals(guiding_field, sample_data_storage, train);
}
#endif

CCL_NAMESPACE_END
