/* SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include "GPU_index_buffer.hh"
#include "GPU_vertex_buffer.hh"

#include "webgpu_buffer.hh"
#include "webgpu_context.hh"

#include <memory>

namespace blender::gpu {

/** WebGPU handles are released on the runtime thread, even for worker-owned geometry. */
struct WebGPUGeometryBufferDeleter {
  void operator()(WebGPUBuffer *buffer) const;
};

using WebGPUGeometryBufferPtr = std::unique_ptr<WebGPUBuffer, WebGPUGeometryBufferDeleter>;

class WebGPUVertexBuffer final : public VertBuf {
 public:
  WebGPUBuffer &buffer_for_use();
  WebGPUTexelBuffer texel_buffer();
  void read_async(WebGPUBuffer::ReadCallback callback);
  void bind_as_ssbo(uint binding) override;
  void bind_as_texture(uint binding) override;
  void wrap_handle(uint64_t handle) override;
  void update_sub(uint start, uint len, const void *data) override;
  void copy_sub(VertBuf &source,
                uint source_first_vertex,
                uint dest_first_vertex,
                uint vertex_count) override;
  void read(void *data) const override;

 private:
  WebGPUGeometryBufferPtr buffer_;
  WebGPUGeometryBufferPtr texel_metadata_;
  std::array<uint32_t, 4> texel_layout_{};

  void acquire_data() override;
  void resize_data() override;
  void release_data() override;
  void upload_data() override;
};

class WebGPUIndexBuffer final : public IndexBuf {
 public:
  WebGPUBuffer &buffer_for_use();
  void read_async(WebGPUBuffer::ReadCallback callback);
  void upload_data() override;
  void bind_as_ssbo(uint binding) override;
  void read(uint32_t *data) const override;
  void update_sub(uint start, uint len, const void *data) override;
  void copy_sub(IndexBuf &source,
                uint source_first_index,
                uint dest_first_index,
                uint index_count) override;

 private:
  WebGPUGeometryBufferPtr buffer_;

  size_t byte_offset() const;
  void strip_restart_indices() override;
};

}  // namespace blender::gpu
