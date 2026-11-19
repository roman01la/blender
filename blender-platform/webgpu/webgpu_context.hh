/* SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include <webgpu/webgpu_cpp.h>

#include "gpu_framebuffer_private.hh"

#include <array>
#include <cstddef>
#include <span>

namespace blender::gpu {

class WebGPUBufferTransfers;
class WebGPUBuffer;
class StorageBuf;

/** WebGPU handles are created and consumed on the runtime thread. */
const wgpu::Device &webgpu_active_device();
void webgpu_flush();
void webgpu_generate_mipmaps(const wgpu::Texture &texture);
void webgpu_upload_depth(const wgpu::Texture &texture,
                         uint32_t mip,
                         wgpu::Origin3D origin,
                         wgpu::Extent3D size,
                         uint32_t row_length,
                         const float *data);
WebGPUBufferTransfers &webgpu_geometry_transfers();
void webgpu_bind_geometry_as_storage(unsigned int binding, const wgpu::Buffer &buffer);
wgpu::Buffer webgpu_bound_storage(int binding);
wgpu::Buffer webgpu_bound_uniform(int binding);
const WebGPUBuffer &webgpu_indirect_buffer(StorageBuf &buffer);

struct WebGPUSampledTexture {
  wgpu::TextureView view;
  wgpu::Sampler sampler;
};

WebGPUSampledTexture webgpu_bound_texture(int binding);
uint64_t webgpu_emulated_sampler_mask();
uint8_t webgpu_emulated_sampler_parameters(int binding);
wgpu::TextureView webgpu_bound_image(int binding);
wgpu::Buffer webgpu_bound_atomic_texture(int binding, bool image);
wgpu::TextureView webgpu_subpass_input(int attachment);
bool webgpu_supports_storage_texture_reads();

struct WebGPUTexelBuffer {
  wgpu::Buffer data;
  wgpu::Buffer metadata;
};

void webgpu_bind_texel_buffer(int slot, const WebGPUTexelBuffer &texels);
WebGPUTexelBuffer webgpu_bound_texel_buffer(int slot);

struct WebGPURenderTargets {
  std::array<wgpu::TextureFormat, GPU_FB_MAX_COLOR_ATTACHMENT> colors{};
  wgpu::TextureFormat depth = wgpu::TextureFormat::Undefined;
  bool stencil = false;
  bool texture_coordinates = false;

  auto operator<=>(const WebGPURenderTargets &) const = default;
};

WebGPURenderTargets webgpu_render_targets();
wgpu::RenderPassEncoder webgpu_render_pass();
void webgpu_render_set_pipeline(const wgpu::RenderPipeline &pipeline);
void webgpu_render_set_bind_group(uint32_t index,
                                  const wgpu::BindGroup &group,
                                  std::span<const uint32_t> offsets = {});
void webgpu_render_set_stencil_reference(uint32_t reference);
const wgpu::Buffer &webgpu_default_vertex_buffer();
const wgpu::Buffer &webgpu_triangle_fan_indices(uint32_t index_count);

struct WebGPUIndexedPoints {
  wgpu::Buffer indices;
  uint32_t first;
  uint32_t count;
  uint32_t base;
  bool short_indices;
};

/** Expand one instance stream, or emit the original vertex IDs when source is null. */
wgpu::Buffer webgpu_gather_indexed_points(const WebGPUIndexedPoints &points,
                                          const wgpu::Buffer &source,
                                          size_t offset,
                                          uint32_t stride);

struct WebGPUPointCommands {
  wgpu::Buffer draw_arguments;
  wgpu::Buffer shader_arguments;
  uint32_t shader_stride;
};

WebGPUPointCommands webgpu_expand_point_commands(const wgpu::Buffer &input,
                                                 uint32_t first_word,
                                                 uint32_t stride_words,
                                                 uint32_t count,
                                                 uint32_t point_capacity,
                                                 bool indexed);

struct WebGPUUniformSlice {
  wgpu::Buffer buffer;
  uint32_t offset;
};

WebGPUUniformSlice webgpu_upload_draw_uniforms(std::span<const std::byte> bytes);
WebGPUUniformSlice webgpu_upload_immediate_vertices(std::span<const std::byte> bytes);

}  // namespace blender::gpu
