/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "device/kernel.h"
#include "util/unique_ptr.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
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

constexpr DeviceKernel webgpu_prepare_ao_pass_kernel = DeviceKernel(DEVICE_KERNEL_NUM + 1);
constexpr DeviceKernel webgpu_svm_ao_textures_kernel = DeviceKernel(DEVICE_KERNEL_NUM + 2);
constexpr DeviceKernel webgpu_sort_surface_paths_kernel = DeviceKernel(DEVICE_KERNEL_NUM + 3);
constexpr DeviceKernel webgpu_svm_validated_textures_kernel = DeviceKernel(DEVICE_KERNEL_NUM + 4);
constexpr DeviceKernel webgpu_svm_hardware_textures_kernel = DeviceKernel(DEVICE_KERNEL_NUM + 5);
constexpr DeviceKernel webgpu_reset_volume_shadow_queue = DeviceKernel(DEVICE_KERNEL_NUM + 6);
constexpr DeviceKernel webgpu_compact_volume_shadow_queue = DeviceKernel(DEVICE_KERNEL_NUM + 7);
constexpr DeviceKernel webgpu_compact_guided_volume_shadow_queue = DeviceKernel(DEVICE_KERNEL_NUM + 8);
constexpr DeviceKernel webgpu_finish_volume_shadow_grid_kernel = DeviceKernel(DEVICE_KERNEL_NUM + 9);

struct WebGPUKernelBinding {
  enum class Type {
    Storage,
    ReadOnlyStorage,
    Uniform,
    FloatTexture,
    FilteringSampler,
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

  /* Packed shader input offsets are unsigned 32-bit word indices. */
  size_t storage_buffer_words() const
  {
    return std::min(storage_buffer_bytes / sizeof(uint32_t), size_t(UINT32_MAX));
  }
};

/** Limits of the initialized compute device. */
WebGPUComputeLimits device_webgpu_compute_limits(const Device &device);

/** Maximum representable number of items after the render's path kernels are registered. */
uint32_t device_webgpu_path_dispatch_capacity(Device &device);

/** Whether a kernel can use the requested number of lanes and bytes of workgroup storage. */
bool device_webgpu_supports_workgroup(const Device &device, uint32_t size, uint32_t storage_bytes);

/** Maximum negotiated workgroup width along one dimension, or zero if unavailable. */
uint32_t device_webgpu_max_1d_workgroup_size(const Device &device);

/** Return the maximum dimension for a filterable RGBA float atlas, or zero if unavailable. */
uint32_t device_webgpu_filterable_texture_limit(const Device &device);

/** Return a device-local atlas handle. A failed optional allocation leaves software sampling usable. */
device_ptr device_webgpu_create_texture_atlas(Device &device,
                                            uint32_t width,
                                            uint32_t height,
                                            std::span<const float> rgba);
void device_webgpu_free_texture_atlas(Device &device, device_ptr atlas);

/** Initialize the compute device and use its negotiated limits for tile scheduling. */
WebGPUComputeLimits device_webgpu_prepare_compute_limits(Device &device);

/** Write an already allocated GPU buffer from transient host storage. The caller may release
 * the byte span when this call returns. */
bool device_webgpu_upload(Device &device,
                          const device_memory &buffer,
                          std::span<const std::byte> bytes);

/** Upload a range aligned to WebGPU's four-byte write granularity. */
bool device_webgpu_upload(Device &device,
                          const device_memory &buffer,
                          size_t byte_offset,
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

/** An optional pipeline variant can fail compilation without invalidating the device. */
bool device_webgpu_try_register_kernel(Device &device,
                                       DeviceKernel kernel,
                                       std::string_view wgsl,
                                       uint32_t workgroup_size,
                                       std::span<const WebGPUKernelBinding> bindings,
                                       std::string_view entry_point,
                                       std::string &compile_error);

struct WebGPUDispatch {
  DeviceKernel kernel;
  const DeviceKernelArguments &args;
  uint32_t work_size = 0;
  device_ptr indirect = 0;
  uint32_t workgroups_y = 1;
  device_ptr texture_atlas = 0;
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
