/* SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include "gpu_texture_private.hh"

#include <webgpu/webgpu_cpp.h>

#include <array>
#include <map>
#include <memory>

namespace blender::gpu {

struct WebGPUTexelBuffer;
class WebGPUVertexBuffer;

struct WebGPUStorageTextureFormat {
  wgpu::TextureFormat format = wgpu::TextureFormat::Undefined;
  const char *glsl_layout = nullptr;
  bool read_write = false;
  bool requires_tier1 = false;
};

WebGPUStorageTextureFormat webgpu_storage_texture_format(TextureFormat format);

class WebGPUTexture final : public Texture {
 public:
  explicit WebGPUTexture(const char *name) : Texture(name) {}

  void generate_mipmap() override;
  void copy_to(Texture *texture, IndexRange mip_levels) override;
  void clear(double4 data) override;
  void swizzle_set(const char swizzle_mask[4]) override;
  void read(int mip, eGPUDataFormat format, void *dst) override;
  void update_sub(int mip,
                  int offset[3],
                  int extent[3],
                  eGPUDataFormat format,
                  const void *data,
                  uint unpack_row_length) override;
  void update_sub(int offset[3],
                  int extent[3],
                  eGPUDataFormat format,
                  GPUPixelBuffer *pixels) override;

  const std::shared_ptr<wgpu::TextureView> &sampling_view();
  const std::shared_ptr<wgpu::TextureView> &storage_view();
  const std::shared_ptr<wgpu::Buffer> &atomic_buffer() const
  {
    return atomic_buffer_;
  }
  WebGPUTexelBuffer texel_buffer();
  wgpu::TextureFormat webgpu_format() const;
  TextureFormat transfer_format() const;
  uint32_t backing_mip(const int mip) const
  {
    return base_mip_ + mip;
  }
  uint32_t backing_layer(const int layer) const
  {
    return base_layer_ + layer;
  }
  const wgpu::Texture &handle() const
  {
    return texture_;
  }
  wgpu::TextureView attachment_view(
      int mip,
      int layer,
      wgpu::TextureUsage usage = wgpu::TextureUsage::RenderAttachment,
      wgpu::TextureFormat format = wgpu::TextureFormat::Undefined) const;
  const std::shared_ptr<wgpu::TextureView> &cached_sampling_view() const
  {
    return sampling_view_;
  }

 private:
  bool init_internal() override;
  bool init_internal(VertBuf *vertices) override;
  bool init_internal(Texture *source, bool use_stencil) override;
  void clear_storage(double4 data);
  bool init_atomic_buffer();
  void clear_atomic_buffer(uint32_t value);
  void update_atomic_buffer(int mip,
                            const int offset[3],
                            const int extent[3],
                            eGPUDataFormat format,
                            const void *data,
                            uint row_length);

  wgpu::Texture texture_;
  uint32_t base_mip_ = 0;
  uint32_t base_layer_ = 0;
  bool use_stencil_ = false;
  std::array<char, 4> sampling_swizzle_ = {'r', 'g', 'b', 'a'};
  wgpu::ComputePipeline clear_pipeline_;
  std::shared_ptr<wgpu::Buffer> atomic_buffer_;
  WebGPUVertexBuffer *texel_source_ = nullptr;
  std::shared_ptr<wgpu::TextureView> sampling_view_;
  std::map<int, std::shared_ptr<wgpu::TextureView>> storage_views_;
};

}  // namespace blender::gpu
