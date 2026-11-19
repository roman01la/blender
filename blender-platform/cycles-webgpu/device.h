/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "device/kernel.h"
#include "util/unique_ptr.h"

#include <cstddef>
#include <functional>
#include <span>
#include <string>
#include <string_view>

namespace wgpu {
class Device;
class DeviceInfo;
class Buffer;
}  // namespace wgpu

CCL_NAMESPACE_BEGIN

class Device;
class DeviceInfo;
class Profiler;
class Stats;
class device_memory;
struct DeviceKernelArguments;

struct WebGPUKernelBinding {
  enum class Type {
    Storage,
    ReadOnlyStorage,
    Uniform,
  };

  uint32_t binding;
  Type type;
  size_t minimum_size;
};

/** Normal Cycles device factory. Compute resources are acquired on the render pthread. */
DeviceInfo device_webgpu_info();
unique_ptr<Device> device_webgpu_create(const DeviceInfo &info,
                                       Stats &stats,
                                       Profiler &profiler,
                                       bool headless);

/** Runtime errors affect the GPU queue while CPU scene memory and fallback remain usable. */
std::string device_webgpu_error(const Device &device);

struct WebGPUComputeLimits {
  size_t storage_buffer_bytes = 0;
  uint32_t workgroups_per_dimension = 0;
};

/** Limits of the initialized compute device. */
WebGPUComputeLimits device_webgpu_compute_limits(const Device &device);

/** Initialize the compute device and use its negotiated limits for tile scheduling. */
WebGPUComputeLimits device_webgpu_prepare_compute_limits(Device &device);

/** Write an already allocated GPU buffer from transient host storage. The caller may release
 * the byte span when this call returns. */
bool device_webgpu_upload(Device &device,
                          const device_memory &buffer,
                          std::span<const std::byte> bytes);

/** Read a whole GPU buffer directly into caller-owned storage. */
bool device_webgpu_read(Device &device, const device_memory &buffer, std::span<std::byte> bytes);

/** Borrow the buffer on its owning worker to issue foreign WebGPU commands. */
bool device_webgpu_with_buffer(
    Device &device,
    const device_memory &buffer,
    const std::function<void(const wgpu::Device &, const wgpu::Buffer &)> &operation);

/** Create on the WebGPU-owning worker. This device is available to explicit
 * kernel callers while Cycles' render kernels and scene storage are being ported. */
unique_ptr<Device> device_webgpu_create_experimental(const wgpu::Device &device,
                                                     Stats &stats,
                                                     Profiler &profiler);

/** Compile on the owning worker while the caller's Cycles pthread waits.
 * Group zero binds the POINTER arguments in descriptor order. The registered
 * shader entry point handles any final partial workgroup. */
bool device_webgpu_register_kernel(Device &device,
                                   DeviceKernel kernel,
                                   std::string_view wgsl,
                                   uint32_t workgroup_size,
                                   std::span<const WebGPUKernelBinding> bindings,
                                   std::string_view entry_point = "main");

struct WebGPUDispatch {
  DeviceKernel kernel;
  const DeviceKernelArguments &args;
  uint32_t work_size = 0;
  device_ptr indirect = 0;
};

/** Record dispatches in order on the owning worker. Stages inherit work_size
 * unless they provide a direct size or a buffer containing indirect group counts. */
bool device_webgpu_enqueue(Device &device, int work_size, std::span<const WebGPUDispatch> dispatches);

/** Clear the counter buffer, dispatch its producer and read the full snapshot in queue order. */
bool device_webgpu_read_queue_counters(Device &device,
                                      int work_size,
                                      const WebGPUDispatch &dispatch,
                                      const device_memory &counters,
                                      std::span<std::byte> bytes);

CCL_NAMESPACE_END
