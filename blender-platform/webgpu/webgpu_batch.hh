/* SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include "GPU_batch.hh"
#include "GPU_state.hh"
#include "gpu_immediate_private.hh"
#include "webgpu_context.hh"

#include <webgpu/webgpu_cpp.h>

#include <compare>
#include <vector>

namespace blender::gpu {

struct WebGPUVertexAttribute {
  wgpu::VertexFormat format;
  uint32_t offset;
  uint32_t location;
  bool packed_snorm = false;

  auto operator<=>(const WebGPUVertexAttribute &) const = default;
};

struct WebGPUVertexLayout {
  uint32_t stride;
  std::vector<WebGPUVertexAttribute> attributes;

  auto operator<=>(const WebGPUVertexLayout &) const = default;
};

struct WebGPURenderPipelineKey {
  WebGPURenderTargets targets;
  wgpu::PrimitiveTopology topology;
  wgpu::IndexFormat strip_index_format = wgpu::IndexFormat::Undefined;
  wgpu::ColorWriteMask write_mask;
  wgpu::CullMode cull_mode;
  wgpu::FrontFace front_face;
  GPUBlend blend;
  GPUDepthTest depth_test;
  bool depth_write;
  GPUStencilTest stencil_test;
  GPUStencilOp stencil_op;
  uint8_t stencil_compare_mask;
  uint8_t stencil_write_mask;
  bool point_sprites;
  bool indirect_point_sprites = false;
  wgpu::IndexFormat point_index_format = wgpu::IndexFormat::Undefined;
  std::vector<WebGPUVertexLayout> vertices;

  auto operator<=>(const WebGPURenderPipelineKey &) const = default;
};

class WebGPUBatch final : public Batch {
 public:
  void draw(int vertex_first, int vertex_count, int instance_first, int instance_count) override;
  void draw_indirect(StorageBuf *indirect, intptr_t offset) override;
  void multi_draw_indirect(StorageBuf *indirect,
                           int count,
                           intptr_t offset,
                           intptr_t stride) override;
};

class WebGPUImmediate final : public Immediate {
 public:
  uchar *begin() override;
  void end() override;

 private:
  std::vector<uchar> staging_;
  std::vector<uchar> expanded_;
};

[[noreturn]] void webgpu_draw_unsupported(const char *feature);

}  // namespace blender::gpu
