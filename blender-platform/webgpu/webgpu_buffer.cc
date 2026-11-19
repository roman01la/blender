/* SPDX-License-Identifier: GPL-2.0-or-later */

#include "webgpu_buffer.hh"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <utility>

namespace blender::gpu {

static size_t aligned_buffer_size(const size_t size)
{
  if (size > std::numeric_limits<size_t>::max() - 3) {
    std::fputs("WebGPU buffer size overflow.\n", stderr);
    std::abort();
  }
  return std::max(size_t(4), (size + 3) & ~size_t(3));
}

static void submit(const wgpu::Device &device, const wgpu::CommandEncoder &encoder)
{
  const wgpu::CommandBuffer commands = encoder.Finish();
  device.GetQueue().Submit(1, &commands);
}

WebGPUBuffer::WebGPUBuffer(const wgpu::Device &device,
                           const size_t size,
                           const wgpu::BufferUsage usage,
                           const char *name)
    : size_(size), allocation_size_(aligned_buffer_size(size)), device_(device)
{
  wgpu::BufferDescriptor desc;
  desc.label = name;
  desc.size = allocation_size_;
  desc.usage = usage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
  buffer_ = device_.CreateBuffer(&desc);
}

void WebGPUBuffer::write(const void *data) const
{
  write_aligned(0, data, size_);
}

void WebGPUBuffer::write_aligned(const size_t offset, const void *data, const size_t size) const
{
  if (offset > size_ || size > size_ - offset || offset % 4 ||
      (size % 4 && offset + size != size_))
  {
    std::fputs("Invalid aligned WebGPU buffer write range.\n", stderr);
    std::abort();
  }
  if (!size) {
    return;
  }
  const auto *bytes = static_cast<const std::byte *>(data);
  const size_t prefix = size & ~size_t(3);
  const wgpu::Queue queue = device_.GetQueue();
  if (prefix) {
    queue.WriteBuffer(buffer_, offset, bytes, prefix);
  }
  if (prefix != size) {
    std::array<std::byte, 4> tail{};
    std::memcpy(tail.data(), bytes + prefix, size - prefix);
    queue.WriteBuffer(buffer_, offset + prefix, tail.data(), tail.size());
  }
}

void WebGPUBuffer::clear(const uint32_t value) const
{
  if (value == 0) {
    const wgpu::CommandEncoder encoder = device_.CreateCommandEncoder();
    encoder.ClearBuffer(buffer_, 0, allocation_size_);
    submit(device_, encoder);
    return;
  }

  /* Bound temporary memory independently of the buffer size. */
  std::array<uint32_t, 4096> words;
  words.fill(value);
  const wgpu::Queue queue = device_.GetQueue();
  for (size_t offset = 0; offset < allocation_size_;) {
    const size_t count = std::min(sizeof(words), allocation_size_ - offset);
    queue.WriteBuffer(buffer_, offset, words.data(), count);
    offset += count;
  }
}

void WebGPUBuffer::copy_from(const WebGPUBuffer &src,
                             const size_t src_offset,
                             const size_t dst_offset,
                             const size_t size) const
{
  if (src_offset > src.size_ || size > src.size_ - src_offset || dst_offset > size_ ||
      size > size_ - dst_offset || (src_offset | dst_offset | size) % 4 != 0 ||
      src.handle().Get() == handle().Get())
  {
    std::fputs("Invalid WebGPU buffer copy range.\n", stderr);
    std::abort();
  }
  if (size) {
    const wgpu::CommandEncoder encoder = device_.CreateCommandEncoder();
    encoder.CopyBufferToBuffer(src.buffer_, src_offset, buffer_, dst_offset, size);
    submit(device_, encoder);
  }
}

void WebGPUBuffer::read_async(ReadCallback callback) const
{
  read_async(0, size_, std::move(callback));
}

void WebGPUBuffer::read_async(const size_t offset, const size_t size, ReadCallback callback) const
{
  if (offset > size_ || size > size_ - offset) {
    std::fputs("Invalid WebGPU buffer readback range.\n", stderr);
    std::abort();
  }
  if (!size) {
    static constexpr std::byte empty{};
    callback({&empty, 0});
    return;
  }
  const size_t start = offset & ~size_t(3);
  const size_t sub_offset = offset - start;
  const size_t copy_size = aligned_buffer_size(sub_offset + size);
  wgpu::BufferDescriptor desc;
  desc.size = copy_size;
  desc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
  wgpu::Buffer staging = device_.CreateBuffer(&desc);
  const wgpu::CommandEncoder encoder = device_.CreateCommandEncoder();
  encoder.CopyBufferToBuffer(buffer_, start, staging, 0, copy_size);
  submit(device_, encoder);

  staging.MapAsync(
      wgpu::MapMode::Read,
      0,
      copy_size,
      wgpu::CallbackMode::AllowSpontaneous,
      [staging, sub_offset, size, cb = std::move(callback)](wgpu::MapAsyncStatus status,
                                                            wgpu::StringView) {
        if (status != wgpu::MapAsyncStatus::Success) {
          cb({});
          return;
        }
        const auto *data = static_cast<const std::byte *>(staging.GetConstMappedRange());
        cb(data ? std::span(data + sub_offset, size) : std::span<const std::byte>{});
        staging.Unmap();
      });
}

void WebGPUBufferTransfers::write(const WebGPUBuffer &dst,
                                  const size_t offset,
                                  const void *data,
                                  const size_t size)
{
  if (offset > dst.size() || size > dst.size() - offset) {
    std::fputs("Invalid WebGPU buffer write range.\n", stderr);
    std::abort();
  }
  if (!size) {
    return;
  }
  if (offset % 4 == 0 && (size % 4 == 0 || offset + size == dst.size())) {
    dst.write_aligned(offset, data, size);
    return;
  }
  const WebGPUBuffer upload(device_, size, wgpu::BufferUsage::Storage, "Unaligned upload");
  upload.write(data);
  copy(upload, 0, dst, offset, size);
}

static wgpu::ComputePipeline make_byte_copy_pipeline(const wgpu::Device &device)
{
  wgpu::ShaderSourceWGSL code;
  code.code = R"(
struct CopyRange {
  source_byte: u32,
  destination_byte: u32,
  size: u32,
}
@group(0) @binding(0) var<storage, read> source: array<u32>;
@group(0) @binding(1) var<storage, read_write> destination: array<u32>;
@group(0) @binding(2) var<uniform> range: CopyRange;

@compute @workgroup_size(64)
fn copy_bytes(@builtin(global_invocation_id) id: vec3u) {
  // Each invocation is the only writer for its destination word.
  let word = range.destination_byte / 4u + id.x;
  let end_byte = range.destination_byte + range.size;
  if (word * 4u >= end_byte) {
    return;
  }
  let first_byte = max(word * 4u, range.destination_byte);
  let last_byte = min(word * 4u + 4u, end_byte);
  var result = destination[word];
  for (var byte = first_byte; byte < last_byte; byte++) {
    let src = range.source_byte + (byte - range.destination_byte);
    let value = (source[src / 4u] >> ((src % 4u) * 8u)) & 255u;
    let shift = (byte % 4u) * 8u;
    result = (result & ~(255u << shift)) | (value << shift);
  }
  destination[word] = result;
}
)";
  wgpu::ShaderModuleDescriptor shader;
  shader.nextInChain = &code;
  wgpu::ComputePipelineDescriptor pipeline;
  pipeline.label = "Blender buffer byte copy";
  pipeline.compute.module = device.CreateShaderModule(&shader);
  pipeline.compute.entryPoint = "copy_bytes";
  return device.CreateComputePipeline(&pipeline);
}

void WebGPUBufferTransfers::copy(const WebGPUBuffer &src,
                                 size_t src_offset,
                                 const WebGPUBuffer &dst,
                                 size_t dst_offset,
                                 size_t size)
{
  if (src_offset > src.size() || size > src.size() - src_offset || dst_offset > dst.size() ||
      size > dst.size() - dst_offset)
  {
    std::fputs("Invalid WebGPU buffer copy range.\n", stderr);
    std::abort();
  }
  if (!size) {
    return;
  }
  const bool same_buffer = src.handle().Get() == dst.handle().Get();
  if (!same_buffer && (src_offset | dst_offset | size) % 4 == 0) {
    dst.copy_from(src, src_offset, dst_offset, size);
    return;
  }
  if (same_buffer && src_offset == dst_offset) {
    return;
  }

  const wgpu::CommandEncoder encoder = device_.CreateCommandEncoder();
  wgpu::Buffer source = src.handle();
  std::optional<WebGPUBuffer> scratch;
  if (same_buffer) {
    /* Snapshot the source before writing to an overlapping or aliased range. */
    const size_t start = src_offset & ~size_t(3);
    src_offset -= start;
    const size_t copy_size = aligned_buffer_size(src_offset + size);
    scratch.emplace(device_, copy_size, wgpu::BufferUsage::Storage, "Aliased buffer copy");
    source = scratch->handle();
    encoder.CopyBufferToBuffer(src.handle(), start, source, 0, copy_size);
    if ((src_offset | dst_offset | size) % 4 == 0) {
      encoder.CopyBufferToBuffer(source, src_offset, dst.handle(), dst_offset, size);
      submit(device_, encoder);
      return;
    }
  }
  if (!byte_copy_pipeline_) {
    byte_copy_pipeline_ = make_byte_copy_pipeline(device_);
  }

  /* Bind small, aligned windows so large buffers need not fit in one storage binding.
   * Cut chunks on destination word boundaries. */
  constexpr size_t window_alignment = 256;
  constexpr size_t chunk_limit = 4 * 1024 * 1024;
  const wgpu::BindGroupLayout layout = byte_copy_pipeline_.GetBindGroupLayout(0);
  const wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
  pass.SetPipeline(byte_copy_pipeline_);
  while (size) {
    const size_t src_base = src_offset & ~(window_alignment - 1);
    const size_t dst_base = dst_offset & ~(window_alignment - 1);
    const size_t src_byte = src_offset - src_base;
    const size_t dst_byte = dst_offset - dst_base;
    const size_t chunk = std::min(size, chunk_limit - dst_byte);
    const std::array<uint32_t, 4> range = {
        uint32_t(src_byte), uint32_t(dst_byte), uint32_t(chunk), 0};
    const WebGPUBuffer params(device_, sizeof(range), wgpu::BufferUsage::Uniform, "Copy range");
    params.write(range.data());
    std::array<wgpu::BindGroupEntry, 3> entries{};
    entries[0].buffer = source;
    entries[0].offset = src_base;
    entries[0].size = aligned_buffer_size(src_byte + chunk);
    entries[1].binding = 1;
    entries[1].buffer = dst.handle();
    entries[1].offset = dst_base;
    entries[1].size = aligned_buffer_size(dst_byte + chunk);
    entries[2].binding = 2;
    entries[2].buffer = params.handle();
    entries[2].size = params.size();
    wgpu::BindGroupDescriptor group;
    group.layout = layout;
    group.entryCount = entries.size();
    group.entries = entries.data();
    pass.SetBindGroup(0, device_.CreateBindGroup(&group));
    const size_t words = aligned_buffer_size(dst_byte % 4 + chunk) / 4;
    pass.DispatchWorkgroups(uint32_t((words + 63) / 64));
    src_offset += chunk;
    dst_offset += chunk;
    size -= chunk;
  }
  pass.End();
  submit(device_, encoder);
}

}  // namespace blender::gpu
