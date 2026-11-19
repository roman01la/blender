/* SPDX-License-Identifier: GPL-2.0-or-later */

#include "webgpu_batch.hh"

#include "GPU_matrix.hh"
#include "gpu_context_private.hh"
#include "gpu_immediate_private.hh"
#include "gpu_shader_interface.hh"
#include "webgpu_context.hh"
#include "webgpu_geometry.hh"
#include "webgpu_shader.hh"

#include <algorithm>
#include <bitset>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace blender::gpu {

[[noreturn]] void webgpu_draw_unsupported(const char *feature)
{
  std::fprintf(stderr, "Blender's WebGPU batch backend does not implement %s yet.\n", feature);
  std::abort();
}

static wgpu::VertexFormat vertex_format(const VertAttrType type)
{
  using W = wgpu::VertexFormat;
  switch (type) {
    case VertAttrType::SFLOAT_32:
      return W::Float32;
    case VertAttrType::SFLOAT_32_32:
      return W::Float32x2;
    case VertAttrType::SFLOAT_32_32_32:
      return W::Float32x3;
    case VertAttrType::SFLOAT_32_32_32_32:
      return W::Float32x4;
    case VertAttrType::SFLOAT_16_16:
      return W::Float16x2;
    case VertAttrType::SFLOAT_16_16_16_16:
      return W::Float16x4;
    case VertAttrType::UINT_32:
      return W::Uint32;
    case VertAttrType::UINT_32_32:
      return W::Uint32x2;
    case VertAttrType::UINT_32_32_32:
      return W::Uint32x3;
    case VertAttrType::UINT_32_32_32_32:
      return W::Uint32x4;
    case VertAttrType::SINT_32:
      return W::Sint32;
    case VertAttrType::SINT_32_32:
      return W::Sint32x2;
    case VertAttrType::SINT_32_32_32:
      return W::Sint32x3;
    case VertAttrType::SINT_32_32_32_32:
      return W::Sint32x4;
    case VertAttrType::UINT_8_8_8_8:
      return W::Uint8x4;
    case VertAttrType::SINT_8_8_8_8:
      return W::Sint8x4;
    case VertAttrType::SNORM_8_8_8_8:
      return W::Snorm8x4;
    case VertAttrType::UNORM_8_8_8_8:
      return W::Unorm8x4;
    case VertAttrType::UINT_16_16:
      return W::Uint16x2;
    case VertAttrType::UINT_16_16_16_16:
      return W::Uint16x4;
    case VertAttrType::SINT_16_16:
      return W::Sint16x2;
    case VertAttrType::SINT_16_16_16_16:
      return W::Sint16x4;
    case VertAttrType::SNORM_16_16:
      return W::Snorm16x2;
    case VertAttrType::SNORM_16_16_16_16:
      return W::Snorm16x4;
    case VertAttrType::UNORM_16_16:
      return W::Unorm16x2;
    case VertAttrType::UNORM_16_16_16_16:
      return W::Unorm16x4;
    case VertAttrType::UNORM_10_10_10_2:
      return W::Unorm10_10_10_2;
    case VertAttrType::SNORM_10_10_10_2:
      return W::Uint32;
    case VertAttrType::UINT_8_DEPRECATED:
      return W::Uint8;
    case VertAttrType::UINT_8_8_DEPRECATED:
      return W::Uint8x2;
    case VertAttrType::SINT_8_DEPRECATED:
      return W::Sint8;
    case VertAttrType::SINT_8_8_DEPRECATED:
      return W::Sint8x2;
    case VertAttrType::UNORM_8_DEPRECATED:
      return W::Unorm8;
    case VertAttrType::UNORM_8_8_DEPRECATED:
      return W::Unorm8x2;
    case VertAttrType::SNORM_8_DEPRECATED:
      return W::Snorm8;
    case VertAttrType::SNORM_8_8_DEPRECATED:
      return W::Snorm8x2;
    case VertAttrType::UINT_16_DEPRECATED:
      return W::Uint16;
    case VertAttrType::SINT_16_DEPRECATED:
      return W::Sint16;
    case VertAttrType::UNORM_16_DEPRECATED:
      return W::Unorm16;
    case VertAttrType::SNORM_16_DEPRECATED:
      return W::Snorm16;
    default:
      std::fprintf(stderr, "Unsupported WebGPU vertex format %u.\n", unsigned(type));
      webgpu_draw_unsupported("conversion of this Blender vertex format");
  }
}

static wgpu::PrimitiveTopology topology(const GPUPrimType type)
{
  using T = wgpu::PrimitiveTopology;
  switch (type) {
    case GPU_PRIM_POINTS:
      return T::PointList;
    case GPU_PRIM_LINES:
      return T::LineList;
    case GPU_PRIM_LINE_STRIP:
      return T::LineStrip;
    case GPU_PRIM_TRIS:
      return T::TriangleList;
    case GPU_PRIM_TRI_STRIP:
      return T::TriangleStrip;
    default:
      webgpu_draw_unsupported("lowering this Blender primitive topology");
  }
}

static wgpu::CullMode cull_mode(const GPUFaceCullTest mode)
{
  switch (mode) {
    case GPU_CULL_NONE:
      return wgpu::CullMode::None;
    case GPU_CULL_FRONT:
      return wgpu::CullMode::Front;
    case GPU_CULL_BACK:
      return wgpu::CullMode::Back;
  }
  webgpu_draw_unsupported("this face cull mode");
}

struct BufferBinding {
  wgpu::Buffer buffer;
  size_t offset;
  uint32_t vertex_count = 0;
};

using AttributeMask = std::bitset<GPU_VERT_ATTR_MAX_LEN>;

static void add_vertex_attributes(const GPUVertFormat &format,
                                  const GPUVertAttr &attr,
                                  uint32_t offset,
                                  const ShaderInterface &interface,
                                  AttributeMask &found,
                                  WebGPUVertexLayout &layout)
{
  for (uint name_index = 0; name_index < attr.name_len; name_index++) {
    const ShaderInput *input = interface.attr_get(
        GPU_vertformat_attr_name_get(&format, &attr, name_index));
    if (input && !found.test(input->location)) {
      found.set(input->location);
      layout.attributes.push_back({vertex_format(attr.type.format),
                                   offset,
                                   uint32_t(input->location),
                                   attr.type.format == VertAttrType::SNORM_10_10_10_2});
    }
  }
}

static bool add_default_vertex_layout(const AttributeMask &found,
                                      const ShaderInterface &interface,
                                      std::vector<WebGPUVertexLayout> &layouts)
{
  if (found.count() == interface.attr_len_) {
    return false;
  }
  WebGPUVertexLayout layout{0, {}};
  for (const int location : IndexRange(GPU_VERT_ATTR_MAX_LEN)) {
    if (found[location] || !(interface.enabled_attr_mask_ & (1u << location))) {
      continue;
    }
    using shader::Type;
    using W = wgpu::VertexFormat;
    W format;
    switch (Type(interface.attr_types_[location])) {
      case Type::float_t:
      case Type::float2_t:
      case Type::float3_t:
      case Type::float4_t:
        format = W::Float32;
        break;
      case Type::int_t:
      case Type::int2_t:
      case Type::int3_t:
      case Type::int4_t:
        format = W::Sint32;
        break;
      case Type::uint_t:
      case Type::uint2_t:
      case Type::uint3_t:
      case Type::uint4_t:
        format = W::Uint32;
        break;
      default:
        webgpu_draw_unsupported("a default value for this vertex input type");
    }
    /* A zero scalar fetch is expanded to (0, 0, 0, 1) where the shader expects more lanes. */
    layout.attributes.push_back({format, 0, uint32_t(location)});
  }
  layouts.push_back(std::move(layout));
  return true;
}

static std::vector<BufferBinding> build_vertex_layouts(const Batch &batch,
                                                       const ShaderInterface &interface,
                                                       std::vector<WebGPUVertexLayout> &layouts)
{
  AttributeMask found;
  std::vector<BufferBinding> bindings;
  for (VertBuf *base : batch.verts) {
    if (!base) {
      continue;
    }
    auto &vertices = static_cast<WebGPUVertexBuffer &>(*base);
    const GPUVertFormat &format = vertices.format;
    WebGPUVertexLayout interleaved{format.stride, {}};
    size_t offset = 0;
    for (uint attr_index = 0; attr_index < format.attr_len; attr_index++) {
      const GPUVertAttr &attr = format.attrs[attr_index];
      if (format.deinterleaved) {
        const uint32_t stride = attr.type.size();
        WebGPUVertexLayout layout{stride, {}};
        add_vertex_attributes(format, attr, 0, interface, found, layout);
        if (!layout.attributes.empty()) {
          layouts.push_back(std::move(layout));
          bindings.push_back({vertices.buffer_for_use().handle(), offset, vertices.vertex_len});
        }
        offset += stride * vertices.vertex_len;
      }
      else {
        add_vertex_attributes(format, attr, attr.offset, interface, found, interleaved);
      }
    }
    if (!interleaved.attributes.empty()) {
      layouts.push_back(std::move(interleaved));
      bindings.push_back({vertices.buffer_for_use().handle(), 0, vertices.vertex_len});
    }
  }
  if (add_default_vertex_layout(found, interface, layouts)) {
    bindings.push_back({webgpu_default_vertex_buffer(), 0});
  }
  return bindings;
}

static WebGPURenderPipelineKey pipeline_key(const Context &context,
                                            const WebGPUShader &shader,
                                            GPUPrimType type,
                                            wgpu::IndexFormat index_format)
{
  const GPUState &state = context.state_manager->state;
  const GPUStateMutable &dynamic = context.state_manager->mutable_state;
  WebGPURenderPipelineKey key{};
  key.targets = webgpu_render_targets();
  key.point_sprites = shader.uses_point_sprites();
  if (key.point_sprites && type != GPU_PRIM_POINTS) {
    webgpu_draw_unsupported("using a point sprite shader with non-point geometry");
  }
  key.topology = key.point_sprites ? wgpu::PrimitiveTopology::TriangleStrip : topology(type);
  if (type == GPU_PRIM_TRI_STRIP || type == GPU_PRIM_LINE_STRIP) {
    key.strip_index_format = index_format;
  }
  key.write_mask = wgpu::ColorWriteMask(state.write_mask & GPU_WRITE_COLOR);
  key.cull_mode = key.point_sprites ? wgpu::CullMode::None :
                                      cull_mode(GPUFaceCullTest(state.culling_test));
  const bool invert = bool(state.invert_facing) != key.targets.texture_coordinates;
  key.front_face = invert ? wgpu::FrontFace::CW : wgpu::FrontFace::CCW;
  key.blend = GPUBlend(state.blend);
  if (key.targets.depth != wgpu::TextureFormat::Undefined) {
    key.depth_test = GPUDepthTest(state.depth_test);
    key.depth_write = key.depth_test != GPU_DEPTH_NONE && (state.write_mask & GPU_WRITE_DEPTH);
  }
  if (key.targets.stencil && state.stencil_test != GPU_STENCIL_NONE) {
    key.stencil_test = GPUStencilTest(state.stencil_test);
    key.stencil_op = GPUStencilOp(state.stencil_op);
    key.stencil_compare_mask = dynamic.stencil_compare_mask;
    key.stencil_write_mask = state.write_mask & GPU_WRITE_STENCIL ? dynamic.stencil_write_mask : 0;
  }
  return key;
}

static void bind_draw(const wgpu::RenderPassEncoder &pass,
                      WebGPUShader &shader,
                      const WebGPURenderPipelineKey &key)
{
  webgpu_render_set_pipeline(shader.render_pipeline(key));
  shader.bind_draw_resources(pass);
  if (key.targets.stencil) {
    webgpu_render_set_stencil_reference(Context::get()->state_manager->mutable_state.stencil_reference);
  }
}

static void draw_point_sprites(const wgpu::RenderPassEncoder &pass,
                               WebGPUShader &shader,
                               const WebGPURenderPipelineKey &key,
                               int first_point,
                               int point_count,
                               int first_instance,
                               int instance_count,
                               bool indexed = false)
{
  for (int i = 0; i < instance_count; i++) {
    shader.set_point_draw_state(first_instance + i, indexed);
    bind_draw(pass, shader, key);
    pass.Draw(4, point_count, 0, first_point);
  }
}

static WebGPUVertexLayout point_id_layout(const WebGPUShader &shader, const bool indexed)
{
  return {indexed ? 4u : 0u, {{wgpu::VertexFormat::Uint32, 0, shader.point_vertex_id_location()}}};
}

static wgpu::IndexFormat batch_index_format(const WebGPUBatch &batch)
{
  return !batch.elem            ? wgpu::IndexFormat::Undefined :
         batch.elem->is_32bit() ? wgpu::IndexFormat::Uint32 :
                                  wgpu::IndexFormat::Uint16;
}

static wgpu::RenderPassEncoder prepare_batch_draw(WebGPUBatch &batch,
                                                  WebGPUShader &shader,
                                                  const wgpu::IndexFormat index_format,
                                                  WebGPURenderPipelineKey &key,
                                                  const int first_vertex = 0,
                                                  const int vertex_count = 0)
{
  auto bindings = build_vertex_layouts(batch, *shader.interface, key.vertices);
  /* Allocation and uploads can submit the current encoder. Complete them before opening a pass. */
  const wgpu::Buffer indices =
      batch.elem ? static_cast<WebGPUIndexBuffer *>(batch.elem)->buffer_for_use().handle() :
                   wgpu::Buffer{};
  if (key.point_sprites) {
    if (indices) {
      const WebGPUIndexedPoints points{indices,
                                       batch.elem->index_start_get() + uint32_t(first_vertex),
                                       uint32_t(vertex_count),
                                       batch.elem->index_base_get(),
                                       index_format == wgpu::IndexFormat::Uint16};
      for (size_t slot = 0; slot < bindings.size(); slot++) {
        const uint32_t stride = key.vertices[slot].stride;
        if (stride != 0) {
          auto &binding = bindings[slot];
          binding = {webgpu_gather_indexed_points(points, binding.buffer, binding.offset, stride),
                     0};
        }
      }
      bindings.push_back({webgpu_gather_indexed_points(points, {}, 0, 0), 0});
    }
    else {
      bindings.push_back({webgpu_default_vertex_buffer(), 0});
    }
    key.vertices.push_back(point_id_layout(shader, bool(indices)));
  }
  const wgpu::RenderPassEncoder pass = webgpu_render_pass();
  if (!pass) {
    return {};
  }
  for (size_t slot = 0; slot < bindings.size(); slot++) {
    pass.SetVertexBuffer(uint32_t(slot), bindings[slot].buffer, bindings[slot].offset);
  }
  if (indices && !key.point_sprites) {
    pass.SetIndexBuffer(indices, index_format);
  }
  return pass;
}

void WebGPUBatch::draw(const int vertex_first,
                       const int vertex_count,
                       const int instance_first,
                       const int instance_count)
{
  const bool expand_fan = prim_type == GPU_PRIM_TRI_FAN && !elem;
  if (expand_fan && vertex_count < 3) {
    return;
  }
  const uint64_t fan_index_count = expand_fan ? uint64_t(vertex_count - 2) * 3 : 0;
  if (fan_index_count > UINT32_MAX) {
    webgpu_draw_unsupported("this triangle fan index count");
  }
  const wgpu::Buffer fan_indices = expand_fan ?
                                       webgpu_triangle_fan_indices(uint32_t(fan_index_count)) :
                                       wgpu::Buffer{};
  Context &context = *Context::get();
  auto &shader = static_cast<WebGPUShader &>(*context.shader);
  const auto index_format = batch_index_format(*this);
  auto key = pipeline_key(context, shader, expand_fan ? GPU_PRIM_TRIS : prim_type, index_format);
  const auto pass = prepare_batch_draw(
      *this, shader, index_format, key, vertex_first, vertex_count);
  if (!pass) {
    return;
  }
  if (key.point_sprites) {
    draw_point_sprites(pass,
                       shader,
                       key,
                       elem ? 0 : vertex_first,
                       vertex_count,
                       instance_first,
                       instance_count,
                       elem != nullptr);
    return;
  }
  bind_draw(pass, shader, key);
  if (expand_fan) {
    pass.SetIndexBuffer(fan_indices, wgpu::IndexFormat::Uint32);
    pass.DrawIndexed(uint32_t(fan_index_count), instance_count, 0, vertex_first, instance_first);
  }
  else if (elem) {
    pass.DrawIndexed(vertex_count,
                     instance_count,
                     elem->index_start_get() + vertex_first,
                     int32_t(elem->index_base_get()),
                     instance_first);
  }
  else {
    pass.Draw(vertex_count, instance_count, vertex_first, instance_first);
  }
}

void WebGPUBatch::draw_indirect(StorageBuf *indirect, const intptr_t offset)
{
  multi_draw_indirect(indirect, 1, offset, 0);
}

void WebGPUBatch::multi_draw_indirect(StorageBuf *indirect,
                                      const int count,
                                      const intptr_t offset,
                                      const intptr_t stride)
{
  if (count == 0) {
    return;
  }
  const uint64_t command_size = (elem ? 5 : 4) * sizeof(uint32_t);
  const uint64_t step = stride == 0 ? command_size : uint64_t(stride);
  const WebGPUBuffer &command_buffer = webgpu_indirect_buffer(*indirect);
  const auto &commands = command_buffer.handle();
  const uint64_t command_buffer_size = command_buffer.allocation_size();
  if (count < 0 || offset < 0 || stride < 0 || step < command_size || (offset | step) % 4 ||
      uint64_t(offset) > command_buffer_size || command_buffer_size - offset < command_size ||
      uint64_t(count - 1) > (command_buffer_size - offset - command_size) / step)
  {
    webgpu_draw_unsupported("this indirect draw buffer range");
  }
  Context &context = *Context::get();
  auto &shader = static_cast<WebGPUShader &>(*context.shader);
  const auto index_format = batch_index_format(*this);
  auto key = pipeline_key(context, shader, prim_type, index_format);
  if (key.point_sprites) {
    key.indirect_point_sprites = true;
    key.point_index_format = index_format;
    key.topology = wgpu::PrimitiveTopology::TriangleList;
    const auto bindings = build_vertex_layouts(*this, *shader.interface, key.vertices);
    const wgpu::Buffer indices =
        elem ? static_cast<WebGPUIndexBuffer *>(elem)->buffer_for_use().handle() : wgpu::Buffer{};
    uint32_t point_capacity = UINT32_MAX / 6;
    if (elem) {
      point_capacity = std::min(point_capacity, elem->index_start_get() + elem->index_len_get());
    }
    std::vector<wgpu::Buffer> streams(bindings.size());
    std::vector<uint32_t> stream_offsets(bindings.size());
    for (size_t slot = 0; slot < bindings.size(); slot++) {
      if (key.vertices[slot].stride == 0) {
        continue;
      }
      if (bindings[slot].offset > UINT32_MAX) {
        webgpu_draw_unsupported("this indirect point vertex stream offset");
      }
      streams[slot] = bindings[slot].buffer;
      stream_offsets[slot] = uint32_t(bindings[slot].offset);
      if (!elem) {
        point_capacity = std::min(point_capacity, bindings[slot].vertex_count);
      }
    }
    const auto expanded = webgpu_expand_point_commands(commands,
                                                       uint32_t(offset / 4),
                                                       uint32_t(step / 4),
                                                       uint32_t(count),
                                                       point_capacity,
                                                       elem != nullptr);
    const auto pass = webgpu_render_pass();
    if (pass) {
      const auto &pipeline = shader.render_pipeline(key);
      webgpu_render_set_pipeline(pipeline);
      if (key.targets.stencil) {
        webgpu_render_set_stencil_reference(context.state_manager->mutable_state.stencil_reference);
      }
      shader.draw_indirect_points(
          pass, pipeline, expanded, indices, streams, stream_offsets, count);
    }
    return;
  }
  const auto pass = prepare_batch_draw(*this, shader, index_format, key);
  if (!pass) {
    return;
  }
  bind_draw(pass, shader, key);
  for (int i = 0; i < count; i++) {
    const uint64_t command_offset = uint64_t(offset) + uint64_t(i) * step;
    if (elem) {
      pass.DrawIndexedIndirect(commands, command_offset);
    }
    else {
      pass.DrawIndirect(commands, command_offset);
    }
  }
}

uchar *WebGPUImmediate::begin()
{
  staging_.resize(size_t(vertex_len + 1) * vertex_format.stride);
  return staging_.data();
}

void WebGPUImmediate::end()
{
  if (vertex_idx == 0) {
    return;
  }
  const size_t stride = vertex_format.stride;
  uint count = vertex_idx;
  GPUPrimType type = prim_type;
  const uchar *vertices = staging_.data();
  if (type == GPU_PRIM_LINE_LOOP && !shader->is_polyline) {
    std::memcpy(staging_.data() + count * stride, vertices, stride);
    count++;
    type = GPU_PRIM_LINE_STRIP;
  }
  if (type == GPU_PRIM_TRI_FAN) {
    if (count < 3) {
      return;
    }
    const uint triangles = count - 2;
    expanded_.resize(size_t(triangles) * 3 * stride);
    for (uint i = 0; i < triangles; i++) {
      uchar *out = expanded_.data() + size_t(i) * 3 * stride;
      std::memcpy(out, vertices, stride);
      std::memcpy(out + stride, vertices + size_t(i + 1) * stride, 2 * stride);
    }
    count = triangles * 3;
    vertices = expanded_.data();
    type = GPU_PRIM_TRIS;
  }
  auto &program = static_cast<WebGPUShader &>(*shader);
  const auto bytes = std::as_bytes(std::span(vertices, count * stride));
  const WebGPUUniformSlice slice = webgpu_upload_immediate_vertices(bytes);
  if (shader->is_polyline) {
    for (const uint slot :
         {GPU_SSBO_POLYLINE_POS_BUF_SLOT, GPU_SSBO_POLYLINE_COL_BUF_SLOT, GPU_SSBO_INDEX_BUF_SLOT})
    {
      webgpu_bind_geometry_as_storage(slot, slice.buffer);
    }
    polyline_draw_workaround(slice.offset);
    return;
  }

  GPU_matrix_bind(shader);
  WebGPURenderPipelineKey key = pipeline_key(
      *Context::get(), program, type, wgpu::IndexFormat::Undefined);
  AttributeMask found;
  WebGPUVertexLayout layout{uint32_t(stride), {}};
  for (const uint i : IndexRange(vertex_format.attr_len)) {
    const GPUVertAttr &attr = vertex_format.attrs[i];
    add_vertex_attributes(vertex_format, attr, attr.offset, *program.interface, found, layout);
  }
  const bool has_immediate_layout = !layout.attributes.empty();
  if (has_immediate_layout) {
    key.vertices.push_back(std::move(layout));
  }
  const bool has_default_layout = add_default_vertex_layout(
      found, *program.interface, key.vertices);
  const auto pass = webgpu_render_pass();
  if (pass) {
    if (has_immediate_layout) {
      pass.SetVertexBuffer(0, slice.buffer, slice.offset);
    }
    if (has_default_layout) {
      pass.SetVertexBuffer(uint32_t(has_immediate_layout), webgpu_default_vertex_buffer());
    }
    if (key.point_sprites) {
      pass.SetVertexBuffer(uint32_t(key.vertices.size()), webgpu_default_vertex_buffer());
      key.vertices.push_back(point_id_layout(program, false));
      draw_point_sprites(pass, program, key, 0, count, 0, 1);
    }
    else {
      bind_draw(pass, program, key);
      pass.Draw(count, 1, 0, 0);
    }
  }
}

}  // namespace blender::gpu
