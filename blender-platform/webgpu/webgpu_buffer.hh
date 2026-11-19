/* SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include <webgpu/webgpu_cpp.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>

namespace blender::gpu {

class WebGPUBuffer {
 public:
  using ReadCallback = std::function<void(std::span<const std::byte>)>;

  WebGPUBuffer(const wgpu::Device &device, size_t size, wgpu::BufferUsage usage, const char *name);

  const wgpu::Buffer &handle() const
  {
    return buffer_;
  }

  size_t size() const
  {
    return size_;
  }

  size_t allocation_size() const
  {
    return allocation_size_;
  }

  void write(const void *data) const;
  void write_aligned(size_t offset, const void *data, size_t size) const;
  void clear(uint32_t value) const;
  void copy_from(const WebGPUBuffer &src, size_t src_offset, size_t dst_offset, size_t size) const;

  /** Data is valid for the duration of the callback. A null data pointer reports a mapping
   * failure. */
  void read_async(ReadCallback callback) const;
  void read_async(size_t offset, size_t size, ReadCallback callback) const;

 private:
  const size_t size_;
  const size_t allocation_size_;
  wgpu::Device device_;
  wgpu::Buffer buffer_;
};

/** Buffer transfers that also support byte ranges inside four-byte WebGPU copy units.
 * Unaligned transfers require buffers created with Storage usage. */
class WebGPUBufferTransfers {
 public:
  explicit WebGPUBufferTransfers(const wgpu::Device &device) : device_(device) {}

  void write(const WebGPUBuffer &dst, size_t offset, const void *data, size_t size);
  void copy(const WebGPUBuffer &src,
            size_t src_offset,
            const WebGPUBuffer &dst,
            size_t dst_offset,
            size_t size);

 private:
  wgpu::Device device_;
  wgpu::ComputePipeline byte_copy_pipeline_;
};

}  // namespace blender::gpu
