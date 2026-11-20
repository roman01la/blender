/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "device/memory.h"
#include "device/queue.h"
#include "device/webgpu/device.h"
#include "device/webgpu/path_state_layout.h"
#include "kernel/integrator/state.h"

#include <array>
#include <cstring>
#include <span>
#include <string_view>

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
    return device_webgpu_register_kernel(
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
  }

  bool alloc()
  {
    counts_.alloc_to_device(counter_words);
    return counts_.device_pointer != 0;
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
    return enqueue(compact_free_kernel, capacity, rays, paths, kernels);
  }

  bool enqueue_stage(DeviceKernel stage,
                     int capacity,
                     const device_memory &rays,
                     const device_memory &paths,
                     std::span<const WebGPUDispatch> kernels)
  {
    const DeviceKernel compact_kernel = stage == DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST ?
                                            compact_intersect_kernel :
                                        stage == DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE ?
                                            compact_surface_kernel :
                                        stage == DEVICE_KERNEL_INTEGRATOR_INTERSECT_SUBSURFACE ?
                                            compact_subsurface_kernel :
                                            compact_background_kernel;
    return enqueue(compact_kernel, capacity, rays, paths, kernels);
  }

  bool read(int capacity,
            const device_memory &rays,
            const device_memory &paths,
            const device_memory &pixels,
            IntegratorQueueCounter &counts,
            bool &samples_overlap)
  {
    const DeviceKernelArguments args(&paths.device_pointer,
                                     &rays.device_pointer,
                                     &counts_.device_pointer,
                                     &pixels.device_pointer);
    const WebGPUDispatch dispatch{count_kernel, args};
    std::array<uint32_t, counter_words> data;
    if (!device_webgpu_read_queue_counters(
            *counts_.device, capacity, dispatch, counts_, std::as_writable_bytes(std::span(data))))
    {
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

  bool enqueue(DeviceKernel compact_kernel,
               int capacity,
               const device_memory &rays,
               const device_memory &paths,
               std::span<const WebGPUDispatch> kernels)
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
    return device_webgpu_enqueue(*counts_.device, capacity, commands);
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
};

CCL_NAMESPACE_END
