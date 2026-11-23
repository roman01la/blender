/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "device/memory.h"
#include "device/queue.h"
#include "device/webgpu/device.h"
#include "device/webgpu/path_state_layout.h"
#include "kernel/integrator/state.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <span>
#include <string_view>
#include <utility>

CCL_NAMESPACE_BEGIN

/* These kernels build the device path index arrays and counters consumed by
 * PathTraceWorkGPU. A live path keeps its physical slot until retirement. */
class WebGPUWavefrontQueue {
 public:
  explicit WebGPUWavefrontQueue(Device &device)
      : counts_(&device, "WebGPU integrator queue counters")
  {
  }

  bool initialize(const std::string_view source)
  {
    constexpr auto storage = WebGPUKernelBinding::Type::Storage;
    constexpr auto read_only = WebGPUKernelBinding::Type::ReadOnlyStorage;
    constexpr WebGPUKernelBinding path = {0, storage, CYCLES_WEBGPU_PATH_WORDS * sizeof(uint32_t)};
    constexpr WebGPUKernelBinding ray = {1, storage, sizeof(uint32_t)};
    constexpr WebGPUKernelBinding compact[] = {path, ray};
    constexpr WebGPUKernelBinding classify[] = {path, ray, {3, read_only, 8 * sizeof(uint32_t)}};
    constexpr WebGPUKernelBinding counters[] = {
        path, ray, {2, storage, counter_bytes}, {4, read_only, 16}};
    Device &device = *counts_.device;
    const bool registered = device_webgpu_register_kernel(
               device, capacity_kernel, source, 1, compact, "set_capacity") &&
           device_webgpu_register_kernel(
               device, reset_kernel, source, 1, {&ray, 1}, "reset_queue") &&
           device_webgpu_register_kernel(
               device, compact_free_kernel, source, 64, compact, "compact_free") &&
           device_webgpu_register_kernel(
               device, compact_intersect_kernel, source, 64, compact, "compact_intersect") &&
           device_webgpu_register_kernel(
               device, compact_surface_kernel, source, 64, compact, "compact_surface") &&
           device_webgpu_register_kernel(
               device, compact_subsurface_kernel, source, 64, compact, "compact_subsurface") &&
           device_webgpu_register_kernel(
               device, compact_background_kernel, source, 64, compact, "compact_background") &&
           device_webgpu_register_kernel(
               device, classify_kernel, source, 64, classify, "classify_intersection") &&
           device_webgpu_register_kernel(
               device, count_kernel, source, 64, counters, "count_queues");
    if (!registered) {
      return false;
    }

    if (source.find("// Cycles WebGPU queue transitions\n") != source.npos) {
      constexpr WebGPUKernelBinding deltas[] = {
          path, ray, {2, storage, delta_counter_words * sizeof(uint32_t)}, {4, read_only, 16}};
      constexpr WebGPUKernelBinding admissions[] = {
          deltas[0], deltas[1], deltas[2], deltas[3], {5, read_only, 3 * sizeof(uint32_t)}};
      transitions_loaded_ =
          device_webgpu_register_kernel(device, webgpu_count_queue_delta_kernel, source, 64,
                                        deltas, "count_queue_delta") &&
          device_webgpu_register_kernel(device, webgpu_count_admitted_delta_kernel, source, 64,
                                        admissions, "count_admitted_delta");
      if (transitions_loaded_) {
        LOG_INFO << "Cycles WebGPU tracks queued path transitions";
      }
      return transitions_loaded_;
    }
    return true;
  }

  bool alloc()
  {
    counts_.alloc_to_device(transitions_loaded_ ? delta_counter_words : counter_words);
    return counts_.device_pointer != 0;
  }

  void begin_batch(const int samples_num)
  {
    active_by_sample_.fill(0);
    pending_stage_ = DEVICE_KERNEL_NUM;
    /* The sample IDs of a stock scheduler batch are contiguous. A window up
     * to the histogram width maps them uniquely without an extra GPU upload. */
    use_transitions_ = transitions_loaded_ && samples_num > 0 &&
                       size_t(samples_num) <= sample_bins;
  }

  bool prepare_surface_sort(std::string_view source)
  {
    if (source.empty() ||
        !device_webgpu_supports_workgroup(
            *counts_.device, surface_sort_group_size, surface_sort_storage_size))
    {
      return true;
    }
    constexpr auto storage = WebGPUKernelBinding::Type::Storage;
    constexpr auto read_only = WebGPUKernelBinding::Type::ReadOnlyStorage;
    constexpr WebGPUKernelBinding bindings[] = {
        {0, storage, sizeof(uint32_t)},
        {1, read_only, 8 * sizeof(uint32_t)},
        {2, read_only, 2 * sizeof(uint32_t)}};
    surface_sort_loaded_ = device_webgpu_register_kernel(
        *counts_.device, webgpu_sort_surface_paths_kernel, source, surface_sort_group_size,
        bindings, "sort_surface_paths");
    return surface_sort_loaded_;
  }

  bool use_surface_sort(const size_t shader_count) const
  {
    return surface_sort_loaded_ && shader_count > 1 && shader_count <= max_surface_sort_shaders;
  }

  bool set_capacity(DeviceQueue &queue, const device_memory &rays, const device_memory &paths)
  {
    const DeviceKernelArguments args(&paths.device_pointer, &rays.device_pointer);
    return queue.enqueue(capacity_kernel, 1, args);
  }

  bool enqueue_admission(int capacity,
                         const device_memory &rays,
                         const device_memory &paths,
                         std::span<const WebGPUDispatch> kernels)
  {
    DCHECK(!kernels.empty());
    pending_admissions_ = kernels.back().work_size;
    pending_stage_ = DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA;
    return enqueue(compact_free_kernel, capacity, rays, paths, kernels);
  }

  bool enqueue_stage(DeviceKernel stage,
                     int capacity,
                     const device_memory &rays,
                     const device_memory &paths,
                     std::span<const WebGPUDispatch> kernels,
                     std::span<const WebGPUDispatch> after = {})
  {
    pending_stage_ = stage;
    const DeviceKernel compact_kernel = stage == DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST ?
                                            compact_intersect_kernel :
                                        stage == DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE ?
                                            compact_surface_kernel :
                                        stage == DEVICE_KERNEL_INTEGRATOR_INTERSECT_SUBSURFACE ?
                                            compact_subsurface_kernel :
                                            compact_background_kernel;
    return enqueue(compact_kernel, capacity, rays, paths, kernels, after);
  }

  bool enqueue_material_stage(const DeviceKernel stage,
                              const int capacity,
                              const device_memory &rays,
                              const device_memory &paths,
                              const device_memory &surfaces,
                              const device_memory &svm,
                              const bool sort,
                              const WebGPUDispatch &setup,
                              const WebGPUDispatch &evaluate)
  {
    if (!sort || stage != DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE || evaluate.work_size < 2) {
      const WebGPUDispatch commands[] = {setup, evaluate};
      return enqueue_stage(stage, capacity, rays, paths, commands);
    }

    const DeviceKernelArguments sort_args(
        &rays.device_pointer, &surfaces.device_pointer, &svm.device_pointer);
    const WebGPUDispatch commands[] = {
        setup,
        {webgpu_sort_surface_paths_kernel, sort_args, surface_sort_work_size(evaluate.work_size)},
        evaluate};
    return enqueue_stage(stage, capacity, rays, paths, commands);
  }

  static uint32_t surface_sort_work_size(const uint32_t count)
  {
    const uint32_t partitions = (count - 1) / surface_sort_partition_size + 1;
    return partitions * surface_sort_group_size;
  }

  bool read(int capacity,
            const device_memory &rays,
            const device_memory &paths,
            const device_memory &pixels,
            const device_memory &tiles,
            IntegratorQueueCounter &counts,
            bool &samples_overlap)
  {
    const DeviceKernel stage = std::exchange(pending_stage_, DEVICE_KERNEL_NUM);
    if (use_transitions_) {
      const bool admitted = stage == DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA;
      const int work_size = admitted ? int(pending_admissions_) :
                            int(stage) < DEVICE_GPU_KERNEL_INTEGRATOR_NUM ?
                                counts.num_queued[stage] :
                                0;
      if (work_size > 0) {
        DeviceKernelArguments args(&paths.device_pointer,
                                   &rays.device_pointer,
                                   &counts_.device_pointer,
                                   &pixels.device_pointer);
        if (admitted) {
          args.add(&tiles.device_pointer);
        }
        const WebGPUDispatch dispatch{admitted ? webgpu_count_admitted_delta_kernel : webgpu_count_queue_delta_kernel, args};
        std::array<uint32_t, delta_counter_words> data;
        if (!read_counts(work_size, dispatch, data)) {
          return false;
        }
        if (apply_transitions(capacity, stage, uint32_t(work_size), data, counts)) {
          samples_overlap = std::ranges::count_if(
                                active_by_sample_, [](const uint32_t count) { return count != 0; }) >
                            1;
          return true;
        }
      }
      /* A stage must preserve the original compact main path list through
       * retirement. The full scan recovers the exact counters if it did not. */
      LOG_WARNING << "Cycles WebGPU queue delta disagrees with the scheduled path list; "
                     "recounting all live paths";
      transitions_loaded_ = false;
      use_transitions_ = false;
    }

    const DeviceKernelArguments args(&paths.device_pointer,
                                     &rays.device_pointer,
                                     &counts_.device_pointer,
                                     &pixels.device_pointer);
    const WebGPUDispatch dispatch{count_kernel, args};
    std::array<uint32_t, delta_counter_words> data;
    if (!read_counts(capacity, dispatch, data)) {
      return false;
    }
    std::memcpy(&counts, data.data(), sizeof(counts));
    const uint32_t inverted_minimum = data[counter_words - 2];
    const uint32_t maximum = data[counter_words - 1];
    samples_overlap = inverted_minimum != 0 && ~inverted_minimum != maximum;
    return true;
  }

  static constexpr DeviceKernel classify_kernel =
      DEVICE_KERNEL_INTEGRATOR_QUEUED_SHADOW_PATHS_ARRAY;

 private:
  static_assert(DEVICE_GPU_KERNEL_INTEGRATOR_NUM == 18);
  static_assert(sizeof(IntegratorQueueCounter) == 19 * sizeof(uint32_t));
  static constexpr size_t counter_words = sizeof(IntegratorQueueCounter) / sizeof(uint32_t) + 2;
  static constexpr size_t counter_bytes = counter_words * sizeof(uint32_t);
  static constexpr size_t sample_bins = 64;
  static constexpr size_t delta_work_word = counter_words + sample_bins;
  static constexpr size_t delta_error_word = delta_work_word + 1;
  static constexpr size_t delta_counter_words = delta_error_word + 1;
  /* Match wavefront-sort.wgsl. The indices and buckets use 12 KiB of workgroup storage. */
  static constexpr uint32_t surface_sort_partition_size = 2048;
  static constexpr uint32_t surface_sort_group_size = 256;
  static constexpr size_t max_surface_sort_shaders = 1024;
  static constexpr uint32_t surface_sort_storage_size =
      sizeof(uint32_t) * (surface_sort_partition_size + max_surface_sort_shaders);

  bool read_counts(const int work_size,
                   const WebGPUDispatch &dispatch,
                   std::array<uint32_t, delta_counter_words> &data)
  {
    return device_webgpu_read_queue_counters(
        *counts_.device,
        work_size,
        dispatch,
        counts_,
        std::as_writable_bytes(std::span(data).first(counts_.device_size / sizeof(uint32_t))));
  }

  bool apply_transitions(const int capacity,
                         const DeviceKernel stage,
                         const uint32_t work_size,
                         const std::array<uint32_t, delta_counter_words> &data,
                         IntegratorQueueCounter &counts)
  {
    if (data[delta_work_word] != work_size || data[delta_error_word] != 0) {
      return false;
    }

    const bool admitted = stage == DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA;
    IntegratorQueueCounter updated = counts;
    if (!admitted) {
      updated.num_queued[stage] = 0;
    }
    uint32_t queued = 0;
    for (int kernel = 0; kernel < DEVICE_GPU_KERNEL_INTEGRATOR_NUM; kernel++) {
      if (data[kernel] > work_size - queued ||
          data[kernel] > uint32_t(capacity - updated.num_queued[kernel]))
      {
        return false;
      }
      queued += data[kernel];
      updated.num_queued[kernel] += int(data[kernel]);
    }

    const uint32_t expected_sample_delta = admitted ? queued : work_size - queued;
    uint32_t sample_delta = 0;
    auto samples = active_by_sample_;
    for (size_t bin = 0; bin < sample_bins; bin++) {
      const uint32_t changed = data[counter_words + bin];
      if (changed > expected_sample_delta - sample_delta ||
          changed > (admitted ? uint32_t(capacity) - samples[bin] : samples[bin]))
      {
        return false;
      }
      sample_delta += changed;
      if (admitted) {
        samples[bin] += changed;
      }
      else {
        samples[bin] -= changed;
      }
    }
    if (sample_delta != expected_sample_delta) {
      return false;
    }

    active_by_sample_ = samples;
    counts = updated;
    return true;
  }

  bool enqueue(DeviceKernel compact_kernel,
               int capacity,
               const device_memory &rays,
               const device_memory &paths,
               std::span<const WebGPUDispatch> kernels,
               std::span<const WebGPUDispatch> after = {})
  {
    const DeviceKernelArguments reset_args(&rays.device_pointer);
    const DeviceKernelArguments compact_args(&paths.device_pointer, &rays.device_pointer);
    vector<WebGPUDispatch> commands;
    commands.reserve(2 + kernels.size());
    commands.push_back({reset_kernel, reset_args, 1});
    commands.push_back({compact_kernel, compact_args});
    for (const WebGPUDispatch &kernel : kernels) {
      commands.push_back(kernel);
    }
    if (after.empty()) {
      return device_webgpu_enqueue(*counts_.device, capacity, commands);
    }
    const WebGPUSubmission submissions[] = {{capacity, commands}, {capacity, after}};
    return device_webgpu_enqueue_submissions(*counts_.device, submissions);
  }

  static constexpr DeviceKernel capacity_kernel =
      DEVICE_KERNEL_INTEGRATOR_COMPACT_SHADOW_PATHS_ARRAY;
  static constexpr DeviceKernel compact_free_kernel =
      DEVICE_KERNEL_INTEGRATOR_TERMINATED_SHADOW_PATHS_ARRAY;
  static constexpr DeviceKernel reset_kernel = DEVICE_KERNEL_INTEGRATOR_COMPACT_STATES;
  static constexpr DeviceKernel compact_intersect_kernel =
      DEVICE_KERNEL_INTEGRATOR_QUEUED_PATHS_ARRAY;
  static constexpr DeviceKernel compact_surface_kernel =
      DEVICE_KERNEL_INTEGRATOR_SORTED_PATHS_ARRAY;
  static constexpr DeviceKernel compact_subsurface_kernel =
      DEVICE_KERNEL_INTEGRATOR_SORT_WRITE_PASS;
  static constexpr DeviceKernel compact_background_kernel =
      DEVICE_KERNEL_INTEGRATOR_COMPACT_PATHS_ARRAY;
  static constexpr DeviceKernel count_kernel = DEVICE_KERNEL_INTEGRATOR_ACTIVE_PATHS_ARRAY;
  device_only_memory<uint32_t> counts_;
  std::array<uint32_t, sample_bins> active_by_sample_ = {};
  DeviceKernel pending_stage_ = DEVICE_KERNEL_NUM;
  uint32_t pending_admissions_ = 0;
  bool surface_sort_loaded_ = false;
  bool transitions_loaded_ = false;
  bool use_transitions_ = false;
};

CCL_NAMESPACE_END
