/* SPDX-License-Identifier: GPL-2.0-or-later */

#include "webgpu_geometry.hh"

#include "MEM_guardedalloc.h"

#include "gpu_texture_private.hh"
#include "webgpu_context.hh"

#include <emscripten/proxying.h>
#include <emscripten/threading.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace blender::gpu {

[[noreturn]] static void invalid_geometry_operation(const char *message)
{
  std::fprintf(stderr, "WebGPU geometry: %s.\n", message);
  std::abort();
}

void WebGPUGeometryBufferDeleter::operator()(WebGPUBuffer *buffer) const
{
  if (emscripten_is_main_runtime_thread()) {
    delete buffer;
    return;
  }

  static emscripten::ProxyingQueue queue;
  if (!queue.proxyAsync(emscripten_main_runtime_thread_id(), [buffer]() { delete buffer; })) {
    invalid_geometry_operation("could not schedule buffer release on the runtime thread");
  }
}

void WebGPUVertexBuffer::acquire_data()
{
  release_data();
  if (usage_ != GPU_USAGE_DEVICE_ONLY) {
    data_ = MEM_new_array_uninitialized<uchar>(size_alloc_get(), "WebGPU vertex data");
  }
}

void WebGPUVertexBuffer::resize_data()
{
  if (data_) {
    data_ = static_cast<uchar *>(MEM_realloc_uninitialized(data_, size_alloc_get()));
  }
}

void WebGPUVertexBuffer::release_data()
{
  MEM_SAFE_DELETE(data_);
  buffer_.reset();
  flag &= ~GPU_VERTBUF_DATA_UPLOADED;
}

void WebGPUVertexBuffer::upload_data()
{
  const size_t size = size_alloc_get();
  if (!buffer_ || buffer_->size() != size) {
    WebGPUGeometryBufferPtr new_buffer(
        new WebGPUBuffer(webgpu_active_device(),
                         size,
                         wgpu::BufferUsage::Vertex | wgpu::BufferUsage::Storage,
                         "Blender vertex buffer"));
    if (buffer_ && !data_) {
      webgpu_geometry_transfers().copy(
          *buffer_, 0, *new_buffer, 0, std::min(buffer_->size(), size));
    }
    buffer_ = std::move(new_buffer);
  }
  if ((flag & GPU_VERTBUF_DATA_DIRTY) && data_) {
    webgpu_geometry_transfers().write(*buffer_, 0, data_, size_used_get());
    if (usage_ == GPU_USAGE_STATIC) {
      MEM_SAFE_DELETE(data_);
    }
  }
  flag &= ~GPU_VERTBUF_DATA_DIRTY;
  flag |= GPU_VERTBUF_DATA_UPLOADED;
}

WebGPUBuffer &WebGPUVertexBuffer::buffer_for_use()
{
  upload_data();
  return *buffer_;
}

void WebGPUVertexBuffer::bind_as_ssbo(uint binding)
{
  webgpu_bind_geometry_as_storage(binding, buffer_for_use().handle());
}

WebGPUTexelBuffer WebGPUVertexBuffer::texel_buffer()
{
  if (format.attr_len == 0) {
    invalid_geometry_operation("a texel buffer needs a vertex format");
  }
  const auto type = format.attrs[0].type.format;
  const auto flags = to_format_flag(TextureFormat(type));
  const uint32_t lanes = format_component_len(DataFormat(type));
  const uint32_t bytes = to_bytesize(DataFormat(type));
  /* Other backends view the buffer as a stream of texels in the first attribute's format.
   * Grease Pencil packs several equal-sized attributes per vertex, including integer bits
   * viewed as RGBA32Float. Keep the original allocation and index each attribute as one texel. */
  if (bytes == 0 || lanes == 0 || format.stride != bytes * format.attr_len) {
    invalid_geometry_operation("a texel buffer needs tightly packed, equal-sized attributes");
  }
  for (uint32_t i = 0; i < format.attr_len; i++) {
    if (format.attrs[i].offset != i * bytes || format.attrs[i].type.size() != bytes) {
      invalid_geometry_operation("a texel buffer needs tightly packed, equal-sized attributes");
    }
  }
  const size_t texel_count = size_used_get() / bytes;
  if (texel_count > INT_MAX) {
    invalid_geometry_operation("texel buffer length exceeds the shader index range");
  }
  const uint32_t bits = ELEM(
                            type, VertAttrType::SNORM_10_10_10_2, VertAttrType::UNORM_10_10_10_2) ?
                            10 :
                            (bytes * 8) / lanes;
  const std::array<uint32_t, 4> metadata{uint32_t(texel_count),
                                         lanes,
                                         bits,
                                         uint32_t(bool(flags & GPU_FORMAT_SIGNED)) |
                                             (uint32_t(bool(flags & GPU_FORMAT_FLOAT)) << 1)};
  const auto data = buffer_for_use().handle();
  if (!texel_metadata_ || metadata != texel_layout_) {
    texel_metadata_.reset(new WebGPUBuffer(webgpu_active_device(),
                                           sizeof(metadata),
                                           wgpu::BufferUsage::Uniform,
                                           "Blender texel buffer metadata"));
    texel_metadata_->write(metadata.data());
    texel_layout_ = metadata;
  }
  return {data, texel_metadata_->handle()};
}

void WebGPUVertexBuffer::bind_as_texture(const uint binding)
{
  if (binding < uint(INT_MAX)) {
    webgpu_bind_texel_buffer(int(binding), texel_buffer());
  }
}

void WebGPUVertexBuffer::wrap_handle(uint64_t /*handle*/)
{
  invalid_geometry_operation("external vertex buffer handles are unavailable");
}

void WebGPUVertexBuffer::update_sub(const uint start, const uint len, const void *data)
{
  if (start > size_alloc_get() || len > size_alloc_get() - start) {
    invalid_geometry_operation("vertex buffer update exceeds the allocation");
  }
  if (!len) {
    return;
  }
  WebGPUBuffer &buffer = buffer_for_use();
  webgpu_geometry_transfers().write(buffer, start, data, len);
  if (data_) {
    std::memmove(data_ + start, data, len);
  }
}

void WebGPUVertexBuffer::copy_sub(VertBuf &source,
                                  const uint source_first_vertex,
                                  const uint dest_first_vertex,
                                  const uint vertex_count)
{
  if (source.format.stride != format.stride || source_first_vertex > source.vertex_len ||
      vertex_count > source.vertex_len - source_first_vertex || dest_first_vertex > vertex_len ||
      vertex_count > vertex_len - dest_first_vertex)
  {
    invalid_geometry_operation("vertex copy has mismatched strides or exceeds the used range");
  }
  auto &src = static_cast<WebGPUVertexBuffer &>(source);
  const size_t stride = format.stride;
  WebGPUBuffer &src_buffer = src.buffer_for_use();
  WebGPUBuffer &dst_buffer = buffer_for_use();
  const size_t src_offset = size_t(source_first_vertex) * stride;
  const size_t dst_offset = size_t(dest_first_vertex) * stride;
  const size_t size = size_t(vertex_count) * stride;
  webgpu_geometry_transfers().copy(src_buffer, src_offset, dst_buffer, dst_offset, size);
  if (data_ && src.data_ && size) {
    std::memmove(data_ + dst_offset, src.data_ + src_offset, size);
  }
}

void WebGPUVertexBuffer::read(void *data) const
{
  if (!size_used_get()) {
    return;
  }
  if (!data_ || (flag & GPU_VERTBUF_DATA_UPLOADED)) {
    invalid_geometry_operation("device vertex data requires asynchronous readback");
  }
  std::memcpy(data, data_, size_used_get());
}

void WebGPUVertexBuffer::read_async(WebGPUBuffer::ReadCallback callback)
{
  WebGPUBuffer &buffer = buffer_for_use();
  webgpu_flush();
  buffer.read_async(0, size_used_get(), std::move(callback));
}

size_t WebGPUIndexBuffer::byte_offset() const
{
  return size_t(index_start_) * to_bytesize(index_type_);
}

void WebGPUIndexBuffer::strip_restart_indices()
{
  auto *start = static_cast<uint32_t *>(data_);
  index_len_ = uint32_t(std::remove(start, start + index_len_, RESTART_INDEX) - start);
}

void WebGPUIndexBuffer::upload_data()
{
  if (is_subrange_) {
    src_->upload_data();
    return;
  }
  if (!buffer_) {
    buffer_.reset(new WebGPUBuffer(webgpu_active_device(),
                                   size_get(),
                                   wgpu::BufferUsage::Index | wgpu::BufferUsage::Storage,
                                   "Blender index buffer"));
  }
  if (data_) {
    webgpu_geometry_transfers().write(*buffer_, 0, data_, size_get());
    MEM_SAFE_DELETE_VOID(data_);
  }
}

WebGPUBuffer &WebGPUIndexBuffer::buffer_for_use()
{
  if (is_subrange_) {
    return static_cast<WebGPUIndexBuffer *>(src_)->buffer_for_use();
  }
  upload_data();
  return *buffer_;
}

void WebGPUIndexBuffer::bind_as_ssbo(const uint binding)
{
  webgpu_bind_geometry_as_storage(binding, buffer_for_use().handle());
}

void WebGPUIndexBuffer::read(uint32_t *data) const
{
  if (!size_get()) {
    return;
  }
  const auto *owner = is_subrange_ ? static_cast<const WebGPUIndexBuffer *>(src_) : this;
  if (!owner->data_) {
    invalid_geometry_operation("device index data requires asynchronous readback");
  }
  std::memcpy(data, static_cast<const std::byte *>(owner->data_) + byte_offset(), size_get());
}

void WebGPUIndexBuffer::read_async(WebGPUBuffer::ReadCallback callback)
{
  WebGPUBuffer &buffer = buffer_for_use();
  webgpu_flush();
  buffer.read_async(byte_offset(), size_get(), std::move(callback));
}

void WebGPUIndexBuffer::update_sub(const uint start, const uint len, const void *data)
{
  if (start > size_get() || len > size_get() - start) {
    invalid_geometry_operation("index buffer update exceeds the view");
  }
  if (len) {
    WebGPUBuffer &buffer = buffer_for_use();
    webgpu_geometry_transfers().write(buffer, byte_offset() + start, data, len);
  }
}

void WebGPUIndexBuffer::copy_sub(IndexBuf &source,
                                 const uint source_first_index,
                                 const uint dest_first_index,
                                 const uint index_count)
{
  auto &src = static_cast<WebGPUIndexBuffer &>(source);
  if (src.index_type_ != index_type_ || source_first_index > src.index_len_ ||
      index_count > src.index_len_ - source_first_index || dest_first_index > index_len_ ||
      index_count > index_len_ - dest_first_index)
  {
    invalid_geometry_operation("index copy has mismatched index widths or exceeds the view");
  }
  const size_t stride = to_bytesize(index_type_);
  WebGPUBuffer &src_buffer = src.buffer_for_use();
  WebGPUBuffer &dst_buffer = buffer_for_use();
  webgpu_geometry_transfers().copy(src_buffer,
                                   src.byte_offset() + size_t(source_first_index) * stride,
                                   dst_buffer,
                                   byte_offset() + size_t(dest_first_index) * stride,
                                   size_t(index_count) * stride);
}

}  // namespace blender::gpu
