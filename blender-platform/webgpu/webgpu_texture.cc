/* SPDX-License-Identifier: GPL-2.0-or-later */

#include "webgpu_texture.hh"

#include "BLI_math_half.hh"

#include "webgpu_batch.hh"
#include "webgpu_context.hh"
#include "webgpu_geometry.hh"

#include <emscripten.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace blender::gpu {

EM_JS_DEPS(webgpu_cycles_preview_copy, "$WebGPU");
EM_JS(bool, webgpu_cycles_preview_ready, (uint32_t ticket), {
  return Module.ghostBrowserWindow?.hasCyclesPreview(ticket >>> 0) ?? false;
});
EM_JS(bool,
      webgpu_cycles_preview_copy,
      (WGPUDevice device, WGPUTexture texture, uint32_t ticket),
      {
        const bitmap = Module.ghostBrowserWindow?.takeCyclesPreview(ticket >>> 0);
        if (!bitmap) {
          return false;
        }
        try {
          const target = WebGPU.getJsObject(texture);
          if (target.width !== bitmap.width || target.height !== bitmap.height) {
            return false;
          }
          WebGPU.getJsObject(device).queue.copyExternalImageToTexture(
            {source: bitmap},
            {texture: target, colorSpace: "srgb", premultipliedAlpha: true},
            [bitmap.width, bitmap.height]);
          return true;
        } catch (error) {
          console.warn("Could not import the Cycles GPU preview", error);
          return false;
        } finally {
          bitmap.close();
        }
      });

bool webgpu_has_cycles_preview(const uint32_t ticket)
{
  return webgpu_cycles_preview_ready(ticket);
}

bool webgpu_import_cycles_preview(Texture &texture, const uint32_t ticket)
{
  const auto &target = static_cast<WebGPUTexture &>(texture);
  return webgpu_cycles_preview_copy(webgpu_active_device().Get(), target.handle().Get(), ticket);
}

struct PixelFormat {
  wgpu::TextureFormat texture;
  eGPUDataFormat upload;
  uint32_t components;
  uint32_t bytes;
};

static PixelFormat pixel_format(const TextureFormat format)
{
  using T = wgpu::TextureFormat;
  switch (format) {
    case TextureFormat::UINT_8:
      return {T::R8Uint, GPU_DATA_UBYTE, 1, 1};
    case TextureFormat::UINT_8_8:
      return {T::RG8Uint, GPU_DATA_UBYTE, 2, 2};
    case TextureFormat::UINT_8_8_8_8:
      return {T::RGBA8Uint, GPU_DATA_UBYTE, 4, 4};
    case TextureFormat::UINT_16:
      return {T::R16Uint, GPU_DATA_UINT, 1, 2};
    case TextureFormat::UINT_16_16:
      return {T::RG16Uint, GPU_DATA_UINT, 2, 4};
    case TextureFormat::UINT_16_16_16_16:
      return {T::RGBA16Uint, GPU_DATA_UINT, 4, 8};
    case TextureFormat::UINT_32:
      return {T::R32Uint, GPU_DATA_UINT, 1, 4};
    case TextureFormat::UINT_32_32:
      return {T::RG32Uint, GPU_DATA_UINT, 2, 8};
    case TextureFormat::UINT_32_32_32_32:
      return {T::RGBA32Uint, GPU_DATA_UINT, 4, 16};
    case TextureFormat::SINT_32:
      return {T::R32Sint, GPU_DATA_INT, 1, 4};
    case TextureFormat::UNORM_8:
      return {T::R8Unorm, GPU_DATA_UBYTE, 1, 1};
    case TextureFormat::UNORM_8_8:
      return {T::RG8Unorm, GPU_DATA_UBYTE, 2, 2};
    case TextureFormat::UNORM_8_8_8_8:
      return {T::RGBA8Unorm, GPU_DATA_UBYTE, 4, 4};
    case TextureFormat::UNORM_16_16:
      /* Keep enough precision for both normalized G-buffer channels in a format that
       * supports read-write storage usage. */
      return {T::RGBA32Float, GPU_DATA_FLOAT, 4, 16};
    case TextureFormat::SRGBA_8_8_8_8:
      return {T::RGBA8UnormSrgb, GPU_DATA_UBYTE, 4, 4};
    case TextureFormat::UNORM_10_10_10_2:
      return {T::RGB10A2Unorm, GPU_DATA_2_10_10_10_REV, 4, 4};
    case TextureFormat::SFLOAT_16:
      return {T::R16Float, GPU_DATA_HALF_FLOAT, 1, 2};
    case TextureFormat::SFLOAT_16_16:
      return {T::RG16Float, GPU_DATA_HALF_FLOAT, 2, 4};
    case TextureFormat::SFLOAT_16_16_16:
    case TextureFormat::SFLOAT_16_16_16_16:
    case TextureFormat::UFLOAT_11_11_10:
      /* WebGPU has no three-channel half-float format. Its packed RGB format also
       * cannot support read-write storage usage. */
      return {T::RGBA16Float, GPU_DATA_HALF_FLOAT, 4, 8};
    case TextureFormat::SFLOAT_32:
      return {T::R32Float, GPU_DATA_FLOAT, 1, 4};
    case TextureFormat::SFLOAT_32_32:
      return {T::RG32Float, GPU_DATA_FLOAT, 2, 8};
    case TextureFormat::SFLOAT_32_32_32_32:
      return {T::RGBA32Float, GPU_DATA_FLOAT, 4, 16};
    case TextureFormat::UNORM_16_DEPTH:
      return {T::Depth16Unorm, GPU_DATA_FLOAT, 1, 2};
    case TextureFormat::SFLOAT_32_DEPTH:
      return {T::Depth32Float, GPU_DATA_FLOAT, 1, 4};
    case TextureFormat::SFLOAT_32_DEPTH_UINT_8:
      return {T::Depth32FloatStencil8, GPU_DATA_FLOAT, 1, 8};
    default:
      std::fprintf(
          stderr, "Unsupported Blender texture format: %s.\n", GPU_texture_format_name(format));
      webgpu_draw_unsupported("this texture pixel format");
  }
}

WebGPUStorageTextureFormat webgpu_storage_texture_format(const TextureFormat format)
{
  using T = wgpu::TextureFormat;
  /* Keep storage declarations consistent with the backing textures in pixel_format(). */
  switch (format) {
    case TextureFormat::UINT_8:
      return {T::R8Uint, "r8ui", true, true};
    case TextureFormat::UINT_16:
      return {T::R16Uint, "r16ui", true, true};
    case TextureFormat::UINT_32:
      return {T::R32Uint, "r32ui", true};
    case TextureFormat::SINT_32:
      return {T::R32Sint, "r32i", true};
    case TextureFormat::SFLOAT_32:
      return {T::R32Float, "r32f", true};
    case TextureFormat::UNORM_8_8_8_8:
      return {T::RGBA8Unorm, "rgba8"};
    case TextureFormat::UNORM_10_10_10_2:
      return {T::RGB10A2Unorm, "rgb10_a2", false, true};
    case TextureFormat::SFLOAT_16:
      return {T::R16Float, "r16f", true, true};
    case TextureFormat::SFLOAT_16_16:
      return {T::RG16Float, "rg16f", false, true};
    case TextureFormat::UINT_8_8_8_8:
      return {T::RGBA8Uint, "rgba8ui"};
    case TextureFormat::UINT_16_16_16_16:
      return {T::RGBA16Uint, "rgba16ui", true};
    case TextureFormat::UFLOAT_11_11_10:
    case TextureFormat::SFLOAT_16_16_16_16:
      return {T::RGBA16Float, "rgba16f", true};
    case TextureFormat::UINT_32_32_32_32:
      return {T::RGBA32Uint, "rgba32ui"};
    case TextureFormat::UNORM_16_16:
    case TextureFormat::SFLOAT_32_32_32_32:
      return {T::RGBA32Float, "rgba32f", true};
    default:
      return {};
  }
}

static wgpu::TextureViewDimension view_dimension(const GPUTextureType type)
{
  using V = wgpu::TextureViewDimension;
  switch (type) {
    case GPU_TEXTURE_1D:
    case GPU_TEXTURE_2D:
      return V::e2D;
    case GPU_TEXTURE_1D_ARRAY:
    case GPU_TEXTURE_2D_ARRAY:
      return V::e2DArray;
    case GPU_TEXTURE_CUBE:
      return V::Cube;
    case GPU_TEXTURE_CUBE_ARRAY:
      return V::CubeArray;
    case GPU_TEXTURE_3D:
      return V::e3D;
    default:
      webgpu_draw_unsupported("this texture dimension");
  }
}

bool WebGPUTexture::init_internal()
{
  if (gpu_image_usage_flags_ & GPU_TEXTURE_USAGE_MEMORY_EXPORT) {
    webgpu_draw_unsupported("externally shared textures");
  }
  if (gpu_image_usage_flags_ & GPU_TEXTURE_USAGE_ATOMIC) {
    return init_atomic_buffer();
  }

  const auto dimension = view_dimension(type_);
  wgpu::TextureDescriptor desc{};
  desc.label = name_.c_str();
  desc.dimension = dimension == wgpu::TextureViewDimension::e3D ? wgpu::TextureDimension::e3D :
                                                                  wgpu::TextureDimension::e2D;
  desc.size = {uint32_t(w_), uint32_t(h_), uint32_t(std::max(d_, 1))};
  if (type_ == GPU_TEXTURE_1D_ARRAY) {
    desc.size = {uint32_t(w_), 1, uint32_t(h_)};
  }
  else if (type_ == GPU_TEXTURE_1D) {
    desc.size = {uint32_t(w_), 1, 1};
  }
  desc.mipLevelCount = mipmaps_;
  desc.format = pixel_format(format_).texture;
  if (desc.format == wgpu::TextureFormat::Depth32FloatStencil8 &&
      !webgpu_active_device().HasFeature(wgpu::FeatureName::Depth32FloatStencil8))
  {
    return false;
  }
  desc.usage = wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::CopySrc;
  const bool render_mipmaps = mipmaps_ > 1 && desc.dimension == wgpu::TextureDimension::e2D &&
                              !(format_flag_ & (GPU_FORMAT_DEPTH | GPU_FORMAT_INTEGER));
  if (gpu_image_usage_flags_ & GPU_TEXTURE_USAGE_SHADER_READ) {
    desc.usage |= wgpu::TextureUsage::TextureBinding;
  }
  if ((gpu_image_usage_flags_ & GPU_TEXTURE_USAGE_ATTACHMENT) ||
      ELEM(format_, TextureFormat::SFLOAT_32_DEPTH, TextureFormat::UNORM_16_DEPTH))
  {
    /* CPU depth uploads use a render pass even on Blender's read-only textures. */
    desc.usage |= wgpu::TextureUsage::RenderAttachment;
  }
  if (render_mipmaps) {
    desc.usage |= wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::RenderAttachment;
  }
  const auto storage_format = webgpu_storage_texture_format(format_);
  if ((gpu_image_usage_flags_ &
       (GPU_TEXTURE_USAGE_SHADER_READ | GPU_TEXTURE_USAGE_SHADER_WRITE)) &&
      ELEM(type_, GPU_TEXTURE_2D, GPU_TEXTURE_2D_ARRAY, GPU_TEXTURE_3D) &&
      storage_format.glsl_layout &&
      (!storage_format.requires_tier1 ||
       webgpu_active_device().HasFeature(wgpu::FeatureName::TextureFormatsTier1)))
  {
    desc.usage |= wgpu::TextureUsage::StorageBinding;
  }
  const wgpu::TextureFormat linear_view_format = wgpu::TextureFormat::RGBA8Unorm;
  if (desc.format == wgpu::TextureFormat::RGBA8UnormSrgb &&
      (desc.usage & wgpu::TextureUsage::RenderAttachment))
  {
    desc.viewFormatCount = 1;
    desc.viewFormats = &linear_view_format;
  }
  texture_ = webgpu_active_device().CreateTexture(&desc);
  return bool(texture_);
}

/* Atomic images use a flat storage buffer with dimensions in its first four words. All shader
 * reads and writes, including sampler texel fetches, use this same allocation. */
static constexpr size_t atomic_data_offset = sizeof(uint32_t) * 4;

bool WebGPUTexture::init_atomic_buffer()
{
  if (!ELEM(format_, TextureFormat::UINT_32, TextureFormat::SINT_32) ||
      !ELEM(type_, GPU_TEXTURE_2D, GPU_TEXTURE_2D_ARRAY, GPU_TEXTURE_3D) || mipmaps_ != 1 ||
      (gpu_image_usage_flags_ & GPU_TEXTURE_USAGE_ATTACHMENT))
  {
    webgpu_draw_unsupported("this atomic texture format, dimension or usage");
  }
  const auto &device = webgpu_active_device();
  const std::array<uint32_t, 4> dimensions{uint32_t(w_), uint32_t(h_), uint32_t(std::max(1, d_)), 0};
  const uint64_t bytes = atomic_data_offset +
                         uint64_t(dimensions[0]) * dimensions[1] * dimensions[2] * sizeof(uint32_t);
  /* The uvec4 header sets the alignment of the whole std430 block. */
  const uint64_t size = (bytes + atomic_data_offset - 1) & ~(atomic_data_offset - 1);
  wgpu::Limits limits{};
  if (device.GetLimits(&limits) != wgpu::Status::Success ||
      size > std::min(limits.maxStorageBufferBindingSize, limits.maxBufferSize))
  {
    return false;
  }
  wgpu::BufferDescriptor descriptor{};
  descriptor.label = name_.c_str();
  descriptor.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst |
                     wgpu::BufferUsage::CopySrc;
  descriptor.size = size;
  auto buffer = device.CreateBuffer(&descriptor);
  if (!buffer) {
    return false;
  }
  device.GetQueue().WriteBuffer(buffer, 0, dimensions.data(), atomic_data_offset);
  atomic_buffer_ = std::make_shared<wgpu::Buffer>(std::move(buffer));
  return true;
}

void WebGPUTexture::clear_atomic_buffer(const uint32_t value)
{
  webgpu_flush();
  const auto &device = webgpu_active_device();
  const auto &buffer = *atomic_buffer_;
  if (value == 0) {
    const auto encoder = device.CreateCommandEncoder();
    encoder.ClearBuffer(buffer, atomic_data_offset, buffer.GetSize() - atomic_data_offset);
    const auto commands = encoder.Finish();
    device.GetQueue().Submit(1, &commands);
    return;
  }

  /* The full-size shadow atlas is cleared by Blender's page shader. This CPU operation also
   * supports small auxiliary textures initialized to non-zero sentinel values. */
  std::array<uint32_t, 4096> words;
  words.fill(value);
  const auto queue = device.GetQueue();
  for (uint64_t offset = atomic_data_offset; offset < buffer.GetSize(); offset += sizeof(words)) {
    const size_t size = std::min<uint64_t>(sizeof(words), buffer.GetSize() - offset);
    queue.WriteBuffer(buffer, offset, words.data(), size);
  }
}

void WebGPUTexture::update_atomic_buffer(const int mip,
                                        const int offset[3],
                                        const int extent[3],
                                        const eGPUDataFormat format,
                                        const void *data,
                                        const uint row_length)
{
  const int depth = type_ == GPU_TEXTURE_2D ? 1 : extent[2];
  const int3 size(w_, h_, std::max(1, d_));
  const int3 upload(extent[0], extent[1], depth);
  const size_t stride = row_length ? row_length : std::max(0, extent[0]);
  if (mip != 0 || format != (format_ == TextureFormat::UINT_32 ? GPU_DATA_UINT : GPU_DATA_INT) ||
      !data || stride < size_t(std::max(0, extent[0])))
  {
    webgpu_draw_unsupported("this atomic texture upload");
  }
  for (int axis = 0; axis < 3; axis++) {
    if (upload[axis] < 0 || offset[axis] < 0 || offset[axis] > size[axis] - upload[axis]) {
      webgpu_draw_unsupported("this atomic texture upload range");
    }
  }
  if (!upload.x || !upload.y || !upload.z) {
    return;
  }
  webgpu_flush();
  const auto queue = webgpu_active_device().GetQueue();
  const auto *source = static_cast<const uint32_t *>(data);
  for (int z = 0; z < depth; z++) {
    for (int y = 0; y < upload.y; y++) {
      const size_t start = offset[0] + size_t(w_) * (offset[1] + y + size_t(h_) * (offset[2] + z));
      queue.WriteBuffer(*atomic_buffer_,
                        atomic_data_offset + start * sizeof(uint32_t),
                        source + stride * (y + size_t(z) * upload.y),
                        size_t(upload.x) * sizeof(uint32_t));
    }
  }
}

bool WebGPUTexture::init_internal(VertBuf *vertices)
{
  texel_source_ = static_cast<WebGPUVertexBuffer *>(vertices);
  return true;
}

WebGPUTexelBuffer WebGPUTexture::texel_buffer()
{
  return texel_source_->texel_buffer();
}

bool WebGPUTexture::init_internal(Texture *source, const bool use_stencil)
{
  const auto &parent = static_cast<WebGPUTexture &>(*source);
  if (!parent.texture_ || parent.texture_.GetFormat() != pixel_format(format_).texture) {
    webgpu_draw_unsupported("creating a view with a different WebGPU backing format");
  }
  texture_ = parent.texture_;
  base_mip_ = parent.base_mip_ + mip_min_;
  base_layer_ = parent.base_layer_ + view_layer_start_;
  use_stencil_ = use_stencil || parent.use_stencil_;
  sampling_swizzle_ = parent.sampling_swizzle_;
  return true;
}

static constexpr std::array<char, 4> identity_swizzle = {'r', 'g', 'b', 'a'};

static wgpu::ComponentSwizzle swizzle_component(const char component)
{
  using C = wgpu::ComponentSwizzle;
  switch (component) {
    case 'r':
      return C::R;
    case 'g':
      return C::G;
    case 'b':
      return C::B;
    case 'a':
      return C::A;
    case '0':
      return C::Zero;
    case '1':
      return C::One;
    default:
      webgpu_draw_unsupported("this texture component swizzle");
  }
}

const std::shared_ptr<wgpu::TextureView> &WebGPUTexture::sampling_view()
{
  if (!sampling_view_) {
    wgpu::TextureViewDescriptor desc{};
    desc.label = name_.c_str();
    desc.dimension = view_dimension(type_);
    desc.usage = wgpu::TextureUsage::TextureBinding;
    desc.baseMipLevel = base_mip_;
    desc.mipLevelCount = mipmaps_;
    if (type_ != GPU_TEXTURE_3D) {
      desc.baseArrayLayer = base_layer_;
      desc.arrayLayerCount = layer_count();
    }
    if (use_stencil_) {
      desc.aspect = wgpu::TextureAspect::StencilOnly;
    }
    else if (format_flag_ & GPU_FORMAT_DEPTH) {
      desc.aspect = wgpu::TextureAspect::DepthOnly;
    }
    wgpu::TextureComponentSwizzleDescriptor swizzle{};
    if (sampling_swizzle_ != identity_swizzle) {
      swizzle.swizzle = {swizzle_component(sampling_swizzle_[0]),
                         swizzle_component(sampling_swizzle_[1]),
                         swizzle_component(sampling_swizzle_[2]),
                         swizzle_component(sampling_swizzle_[3])};
      desc.nextInChain = &swizzle;
    }
    sampling_view_ = std::make_shared<wgpu::TextureView>(texture_.CreateView(&desc));
  }
  return sampling_view_;
}

const std::shared_ptr<wgpu::TextureView> &WebGPUTexture::storage_view()
{
  if (!(texture_.GetUsage() & wgpu::TextureUsage::StorageBinding)) {
    webgpu_draw_unsupported("using this texture as a storage image");
  }
  auto &view = storage_views_[base_mip_];
  if (!view) {
    wgpu::TextureViewDescriptor desc{};
    desc.label = name_.c_str();
    desc.dimension = view_dimension(type_);
    desc.usage = wgpu::TextureUsage::StorageBinding;
    desc.baseMipLevel = base_mip_;
    desc.mipLevelCount = 1;
    if (type_ != GPU_TEXTURE_3D) {
      desc.baseArrayLayer = base_layer_;
      desc.arrayLayerCount = layer_count();
    }
    view = std::make_shared<wgpu::TextureView>(texture_.CreateView(&desc));
  }
  return view;
}

wgpu::TextureFormat WebGPUTexture::webgpu_format() const
{
  return texture_.GetFormat();
}

TextureFormat WebGPUTexture::transfer_format() const
{
  switch (format_) {
    case TextureFormat::SFLOAT_16_16_16:
    case TextureFormat::UFLOAT_11_11_10:
      return TextureFormat::SFLOAT_16_16_16_16;
    case TextureFormat::UNORM_16_16:
      return TextureFormat::SFLOAT_32_32_32_32;
    default:
      return format_;
  }
}

wgpu::TextureView WebGPUTexture::attachment_view(const int mip,
                                                 const int layer,
                                                 const wgpu::TextureUsage usage,
                                                 const wgpu::TextureFormat format) const
{
  if (type_ == GPU_TEXTURE_3D || (layer < 0 && layer_count() > 1)) {
    webgpu_draw_unsupported("layered or 3D framebuffer attachments");
  }
  wgpu::TextureViewDescriptor desc{};
  desc.label = name_.c_str();
  desc.dimension = wgpu::TextureViewDimension::e2D;
  desc.format = format;
  desc.usage = usage;
  desc.baseMipLevel = base_mip_ + mip;
  desc.mipLevelCount = 1;
  desc.baseArrayLayer = base_layer_ + std::max(0, layer);
  desc.arrayLayerCount = 1;
  return texture_.CreateView(&desc);
}

void WebGPUTexture::update_sub(const int mip,
                               int offset[3],
                               int extent[3],
                               const eGPUDataFormat format,
                               const void *data,
                               const uint unpack_row_length)
{
  if (atomic_buffer_) {
    update_atomic_buffer(mip, offset, extent, format, data, unpack_row_length);
    return;
  }
  const PixelFormat pixels = pixel_format(format_);
  /* Blender uses Y for the layers of a 1D array and sets the unused Z dimension to zero. */
  const bool line_array = type_ == GPU_TEXTURE_1D_ARRAY;
  const int3 upload_extent = {
      extent[0],
      type_ == GPU_TEXTURE_1D ? 1 : extent[1],
      ELEM(type_, GPU_TEXTURE_1D, GPU_TEXTURE_1D_ARRAY, GPU_TEXTURE_2D) ? 1 : extent[2]};
  if (mip < 0 || mip >= mipmaps_) {
    webgpu_draw_unsupported("this upload mip level");
  }
  int3 mip_extent = mip_size_get(mip);
  if (line_array) {
    mip_extent.z = 1;
  }
  for (int axis = 0; axis < 3; axis++) {
    if (offset[axis] < 0 || upload_extent[axis] < 0 ||
        offset[axis] > mip_extent[axis] - upload_extent[axis])
    {
      webgpu_draw_unsupported("this texture upload range");
    }
  }
  const uint32_t width = upload_extent.x;
  const uint32_t height = line_array ? 1 : upload_extent.y;
  const uint32_t depth = line_array ? upload_extent.y : upload_extent.z;
  if (width == 0 || height == 0 || depth == 0) {
    return;
  }

  uint32_t row_length = unpack_row_length ? unpack_row_length : width;
  if (!data || row_length < width) {
    webgpu_draw_unsupported("this texture upload source");
  }
  const wgpu::Origin3D origin = line_array ?
                                    wgpu::Origin3D{uint32_t(offset[0]), 0, base_layer_ + offset[1]} :
                                    wgpu::Origin3D{uint32_t(offset[0]),
                                                   uint32_t(offset[1]),
                                                   base_layer_ + offset[2]};
  const wgpu::Extent3D size{width, height, depth};
  const size_t rows = size_t(height) * depth;
  if (format_flag_ & GPU_FORMAT_DEPTH_STENCIL) {
    if ((format_flag_ & GPU_FORMAT_STENCIL) || !ELEM(format, GPU_DATA_FLOAT, GPU_DATA_UINT)) {
      webgpu_draw_unsupported("this CPU upload to a depth/stencil texture");
    }
    if (row_length > std::numeric_limits<uint32_t>::max() / sizeof(float)) {
      webgpu_draw_unsupported("this depth upload row stride");
    }
    if (format == GPU_DATA_FLOAT) {
      webgpu_upload_depth(
          texture_, base_mip_ + mip, origin, size, row_length, static_cast<const float *>(data));
    }
    else {
      const auto *source = static_cast<const uint32_t *>(data);
      constexpr double unit = 1.0 / std::numeric_limits<uint32_t>::max();
      std::vector<float> depth_values(rows * width);
      for (size_t row = 0; row < rows; row++) {
        for (uint32_t x = 0; x < width; x++) {
          depth_values[row * width + x] = float(source[row * row_length + x] * unit);
        }
      }
      webgpu_upload_depth(texture_, base_mip_ + mip, origin, size, width, depth_values.data());
    }
    return;
  }
  std::vector<uint8_t> bytes;
  std::vector<uint16_t> halves;
  std::vector<uint32_t> packed_colors;
  std::vector<float> padded_colors;
  if (ELEM(format_, TextureFormat::SFLOAT_16_16_16, TextureFormat::UFLOAT_11_11_10)) {
    const bool unsigned_rgb = format_ == TextureFormat::UFLOAT_11_11_10;
    if (!ELEM(format, GPU_DATA_FLOAT, GPU_DATA_HALF_FLOAT) &&
        !(unsigned_rgb && format == GPU_DATA_10_11_11_REV))
    {
      webgpu_draw_unsupported("this RGB float texture upload format");
    }
    halves.resize(rows * width * 4);
    for (size_t row = 0; row < rows; row++) {
      for (uint32_t x = 0; x < width; x++) {
        const size_t index = row * row_length + x;
        uint16_t *dst = halves.data() + (row * width + x) * 4;
        if (format == GPU_DATA_FLOAT) {
          const float *src = static_cast<const float *>(data) + index * 3;
          for (int component = 0; component < 3; component++) {
            dst[component] = math::float_to_half(
                unsigned_rgb ? std::max(src[component], 0.0f) : src[component]);
          }
        }
        else if (format == GPU_DATA_HALF_FLOAT) {
          const uint16_t *src = static_cast<const uint16_t *>(data) + index * 3;
          for (int component = 0; component < 3; component++) {
            dst[component] = unsigned_rgb && (src[component] & 0x8000u) ? 0 : src[component];
          }
        }
        else {
          const uint32_t packed = static_cast<const uint32_t *>(data)[index];
          /* Packed channels and half floats share a five-bit exponent. */
          dst[0] = (packed & 0x7ffu) << 4;
          dst[1] = ((packed >> 11) & 0x7ffu) << 4;
          dst[2] = (packed >> 22) << 5;
        }
        dst[3] = 0x3c00u;
      }
    }
    data = halves.data();
    row_length = width;
  }
  else if (format_ == TextureFormat::UNORM_16_16 && format == GPU_DATA_FLOAT) {
    const auto *source = static_cast<const float *>(data);
    const auto unorm16 = [](const float value) {
      const float clamped = std::clamp(std::isnan(value) ? 0.0f : value, 0.0f, 1.0f);
      return std::round(clamped * 65535.0f) / 65535.0f;
    };
    padded_colors.resize(rows * width * 4);
    for (size_t row = 0; row < rows; row++) {
      for (uint32_t x = 0; x < width; x++) {
        const float *src = source + (row * row_length + x) * 2;
        float *dst = padded_colors.data() + (row * width + x) * 4;
        dst[0] = unorm16(src[0]);
        dst[1] = unorm16(src[1]);
        dst[2] = 0.0f;
        dst[3] = 1.0f;
      }
    }
    data = padded_colors.data();
    row_length = width;
  }
  else if (format_ == TextureFormat::UNORM_16_16) {
    webgpu_draw_unsupported("this normalized two-channel texture upload format");
  }
  else if (format_ == TextureFormat::UNORM_10_10_10_2 && format == GPU_DATA_FLOAT) {
    const auto *source = static_cast<const float *>(data);
    const auto unorm = [](const float value, const float max_code) {
      return uint32_t(
          std::lround(std::clamp(std::isnan(value) ? 0.0f : value, 0.0f, 1.0f) * max_code));
    };
    packed_colors.resize(rows * width);
    for (size_t row = 0; row < rows; row++) {
      for (uint32_t x = 0; x < width; x++) {
        const float *rgba = source + (row * row_length + x) * 4;
        packed_colors[row * width + x] = unorm(rgba[0], 1023.0f) |
                                         (unorm(rgba[1], 1023.0f) << 10) |
                                         (unorm(rgba[2], 1023.0f) << 20) |
                                         (unorm(rgba[3], 3.0f) << 30);
      }
    }
    data = packed_colors.data();
    row_length = width;
  }
  else if ((format_flag_ & GPU_FORMAT_INTEGER) && !(format_flag_ & GPU_FORMAT_SIGNED)) {
    const uint32_t component_size = pixels.bytes / pixels.components;
    if (format == GPU_DATA_UINT && component_size < sizeof(uint32_t)) {
      const auto *source = static_cast<const uint32_t *>(data);
      const size_t row_components = size_t(width) * pixels.components;
      const size_t source_stride = size_t(row_length) * pixels.components;
      const auto pack = [&](auto &packed) {
        using Component = typename std::decay_t<decltype(packed)>::value_type;
        constexpr auto limit = uint32_t(std::numeric_limits<Component>::max());
        packed.resize(rows * row_components);
        for (size_t row = 0; row < rows; row++) {
          for (size_t i = 0; i < row_components; i++) {
            packed[row * row_components + i] = Component(
                std::min(source[row * source_stride + i], limit));
          }
        }
        data = packed.data();
      };
      if (component_size == 1) {
        pack(bytes);
      }
      else {
        pack(halves);
      }
      row_length = width;
    }
    else if (format != pixels.upload) {
      webgpu_draw_unsupported("this unsigned integer texture upload format");
    }
  }
  else if (format != pixels.upload) {
    if (format != GPU_DATA_FLOAT ||
        (pixels.upload != GPU_DATA_UBYTE && pixels.upload != GPU_DATA_HALF_FLOAT))
    {
      webgpu_draw_unsupported("this pixel upload conversion");
    }
    const auto *source = static_cast<const float *>(data);
    const size_t row_components = size_t(width) * pixels.components;
    const size_t source_stride = size_t(row_length) * pixels.components;
    if (pixels.upload == GPU_DATA_UBYTE) {
      bytes.resize(rows * row_components);
      for (size_t row = 0; row < rows; row++) {
        for (size_t i = 0; i < row_components; i++) {
          const float value = source[row * source_stride + i];
          bytes[row * row_components + i] = uint8_t(
              std::lround(std::clamp(std::isnan(value) ? 0.0f : value, 0.0f, 1.0f) * 255.0f));
        }
      }
      data = bytes.data();
    }
    else {
      halves.resize(rows * row_components);
      for (size_t row = 0; row < rows; row++) {
        math::float_to_half_array(
            source + row * source_stride, halves.data() + row * row_components, row_components);
      }
      data = halves.data();
    }
    row_length = width;
  }

  wgpu::TexelCopyTextureInfo destination{};
  destination.texture = texture_;
  destination.mipLevel = base_mip_ + mip;
  destination.origin = origin;
  wgpu::TexelCopyBufferLayout layout{};
  layout.bytesPerRow = row_length * pixels.bytes;
  layout.rowsPerImage = height;
  const size_t byte_count = size_t(layout.bytesPerRow) * (rows - 1) + size_t(width) * pixels.bytes;
  /* Queue writes must follow any previously recorded draws using this texture. */
  webgpu_flush();
  webgpu_active_device().GetQueue().WriteTexture(&destination, data, byte_count, &layout, &size);
}

void WebGPUTexture::copy_to(Texture *destination, const IndexRange mip_levels)
{
  auto &target = static_cast<WebGPUTexture &>(*destination);
  webgpu_flush();
  auto encoder = webgpu_active_device().CreateCommandEncoder();
  if (atomic_buffer_ || target.atomic_buffer_) {
    if (!atomic_buffer_ || !target.atomic_buffer_ ||
        atomic_buffer_->GetSize() != target.atomic_buffer_->GetSize() ||
        mip_levels != IndexRange(1))
    {
      webgpu_draw_unsupported("this atomic texture copy");
    }
    encoder.CopyBufferToBuffer(*atomic_buffer_,
                               atomic_data_offset,
                               *target.atomic_buffer_,
                               atomic_data_offset,
                               atomic_buffer_->GetSize() - atomic_data_offset);
    const auto commands = encoder.Finish();
    webgpu_active_device().GetQueue().Submit(1, &commands);
    return;
  }
  wgpu::TexelCopyTextureInfo source_copy{};
  source_copy.texture = texture_;
  source_copy.origin.z = base_layer_;
  wgpu::TexelCopyTextureInfo target_copy{};
  target_copy.texture = target.texture_;
  target_copy.origin.z = target.base_layer_;
  for (const int mip : mip_levels) {
    source_copy.mipLevel = base_mip_ + mip;
    target_copy.mipLevel = target.base_mip_ + mip;
    const int3 size = mip_size_get(mip);
    const bool line_array = type_ == GPU_TEXTURE_1D_ARRAY;
    const wgpu::Extent3D extent{uint32_t(size.x),
                                line_array ? 1u : uint32_t(size.y),
                                line_array ? uint32_t(size.y) : uint32_t(size.z)};
    encoder.CopyTextureToTexture(&source_copy, &target_copy, &extent);
  }
  const wgpu::CommandBuffer commands = encoder.Finish();
  webgpu_active_device().GetQueue().Submit(1, &commands);
}

void WebGPUTexture::generate_mipmap()
{
  if (mipmaps_ <= 1) {
    return;
  }
  if (source_texture_) {
    webgpu_draw_unsupported("mipmap generation for a texture view");
  }
  if (type_ == GPU_TEXTURE_3D || (format_flag_ & (GPU_FORMAT_DEPTH | GPU_FORMAT_INTEGER))) {
    webgpu_draw_unsupported("mipmap generation for 3D, depth or integer textures");
  }
  webgpu_generate_mipmaps(texture_);
}

static const char *storage_format_wgsl(const wgpu::TextureFormat format)
{
  using T = wgpu::TextureFormat;
  switch (format) {
    case T::R8Uint:
      return "r8uint";
    case T::R16Uint:
      return "r16uint";
    case T::R32Uint:
      return "r32uint";
    case T::R32Sint:
      return "r32sint";
    case T::R32Float:
      return "r32float";
    case T::RGBA8Unorm:
      return "rgba8unorm";
    case T::RGB10A2Unorm:
      return "rgb10a2unorm";
    case T::RGBA8Uint:
      return "rgba8uint";
    case T::R16Float:
      return "r16float";
    case T::RG16Float:
      return "rg16float";
    case T::RGBA16Float:
      return "rgba16float";
    case T::RGBA16Uint:
      return "rgba16uint";
    case T::RGBA32Float:
      return "rgba32float";
    case T::RGBA32Uint:
      return "rgba32uint";
    default:
      webgpu_draw_unsupported("clearing this WebGPU storage texture format");
  }
}

void WebGPUTexture::clear_storage(const double4 data)
{
  if (!(texture_.GetUsage() & wgpu::TextureUsage::StorageBinding) ||
      !ELEM(type_, GPU_TEXTURE_2D, GPU_TEXTURE_2D_ARRAY, GPU_TEXTURE_3D))
  {
    webgpu_draw_unsupported("clearing a texture without a WebGPU attachment or storage view");
  }

  webgpu_flush();
  const auto &device = webgpu_active_device();
  const bool integer = (format_flag_ & GPU_FORMAT_INTEGER) != 0;
  const bool signed_integer = integer && (format_flag_ & GPU_FORMAT_SIGNED);
  if (!clear_pipeline_) {
    const char *dimension = type_ == GPU_TEXTURE_3D         ? "3d" :
                            type_ == GPU_TEXTURE_2D_ARRAY ? "2d_array" :
                                                           "2d";
    const char *coordinates = type_ == GPU_TEXTURE_3D         ? "vec3i(id)" :
                              type_ == GPU_TEXTURE_2D_ARRAY ? "vec2i(id.xy), i32(id.z)" :
                                                             "vec2i(id.xy)";
    const char *value = signed_integer ? "bitcast<vec4i>(clear_value)" :
                        integer        ? "clear_value" :
                                         "bitcast<vec4f>(clear_value)";
    const char *directive = webgpu_storage_texture_format(format_).requires_tier1 ?
                                "requires texture_formats_tier1;\n" :
                                "";
    const std::string source =
        std::string(directive) + "@group(0) @binding(0) var dst: texture_storage_" + dimension + "<" +
        storage_format_wgsl(texture_.GetFormat()) + ", write>;\n"
        "@group(0) @binding(1) var<uniform> clear_value: vec4u;\n"
        "@compute @workgroup_size(8, 8, 1)\n"
        "fn clear_texture(@builtin(global_invocation_id) id: vec3u) {\n"
        "  if (any(id.xy >= textureDimensions(dst).xy)) { return; }\n"
        "  textureStore(dst, " + coordinates + ", " + value + ");\n}\n";
    wgpu::ShaderSourceWGSL wgsl{};
    wgsl.code = source.c_str();
    wgpu::ShaderModuleDescriptor module_desc{};
    module_desc.nextInChain = &wgsl;
    module_desc.label = "Blender storage texture clear";
    wgpu::ComputePipelineDescriptor pipeline_desc{};
    pipeline_desc.label = module_desc.label;
    pipeline_desc.compute.module = device.CreateShaderModule(&module_desc);
    pipeline_desc.compute.entryPoint = "clear_texture";
    clear_pipeline_ = device.CreateComputePipeline(&pipeline_desc);
  }

  std::array<uint32_t, 4> values;
  for (int i = 0; i < 4; i++) {
    if (signed_integer) {
      values[i] = uint32_t(int32_t(data[i]));
    }
    else if (integer) {
      values[i] = uint32_t(data[i]);
    }
    else {
      values[i] = std::bit_cast<uint32_t>(float(data[i]));
    }
  }

  wgpu::BufferDescriptor uniform_desc{};
  uniform_desc.label = "Blender storage texture clear value";
  uniform_desc.size = sizeof(values);
  uniform_desc.usage = wgpu::BufferUsage::Uniform;
  uniform_desc.mappedAtCreation = true;
  const auto uniform = device.CreateBuffer(&uniform_desc);
  std::memcpy(uniform.GetMappedRange(), values.data(), sizeof(values));
  uniform.Unmap();

  std::array<wgpu::BindGroupEntry, 2> entries{};
  entries[0].textureView = *storage_view();
  entries[1].binding = 1;
  entries[1].buffer = uniform;
  entries[1].size = sizeof(values);
  wgpu::BindGroupDescriptor bind_group_desc{};
  bind_group_desc.layout = clear_pipeline_.GetBindGroupLayout(0);
  bind_group_desc.entryCount = entries.size();
  bind_group_desc.entries = entries.data();
  const auto bind_group = device.CreateBindGroup(&bind_group_desc);

  const int3 size = mip_size_get(0);
  const auto encoder = device.CreateCommandEncoder();
  const auto pass = encoder.BeginComputePass();
  pass.SetPipeline(clear_pipeline_);
  pass.SetBindGroup(0, bind_group);
  pass.DispatchWorkgroups((size.x + 7) / 8, (size.y + 7) / 8, size.z);
  pass.End();
  const auto commands = encoder.Finish();
  device.GetQueue().Submit(1, &commands);
}

void WebGPUTexture::clear(const double4 data)
{
  if (atomic_buffer_) {
    clear_atomic_buffer(format_ == TextureFormat::SINT_32 ? uint32_t(int32_t(data.x)) :
                                                            uint32_t(data.x));
    return;
  }
  if (!(texture_.GetUsage() & wgpu::TextureUsage::RenderAttachment) || type_ == GPU_TEXTURE_3D) {
    clear_storage(data);
    return;
  }
  webgpu_flush();
  const auto &device = webgpu_active_device();
  auto encoder = device.CreateCommandEncoder();
  for (int layer = 0; layer < layer_count(); layer++) {
    wgpu::RenderPassDescriptor pass{};
    wgpu::RenderPassColorAttachment color{};
    wgpu::RenderPassDepthStencilAttachment depth{};
    if (format_flag_ & GPU_FORMAT_DEPTH) {
      depth.view = attachment_view(0, layer);
      depth.depthLoadOp = wgpu::LoadOp::Clear;
      depth.depthStoreOp = wgpu::StoreOp::Store;
      depth.depthClearValue = float(data.x);
      if (format_flag_ & GPU_FORMAT_STENCIL) {
        depth.stencilLoadOp = wgpu::LoadOp::Load;
        depth.stencilStoreOp = wgpu::StoreOp::Store;
      }
      pass.depthStencilAttachment = &depth;
    }
    else {
      color.view = attachment_view(0, layer);
      color.loadOp = wgpu::LoadOp::Clear;
      color.storeOp = wgpu::StoreOp::Store;
      color.clearValue = {data.x, data.y, data.z, data.w};
      pass.colorAttachmentCount = 1;
      pass.colorAttachments = &color;
    }
    encoder.BeginRenderPass(&pass).End();
  }
  const auto commands = encoder.Finish();
  device.GetQueue().Submit(1, &commands);
}

void WebGPUTexture::swizzle_set(const char swizzle_mask[4])
{
  const std::array<char, 4> swizzle = {
      swizzle_mask[0], swizzle_mask[1], swizzle_mask[2], swizzle_mask[3]};
  if (swizzle == sampling_swizzle_) {
    return;
  }
  if (swizzle != identity_swizzle &&
      !webgpu_active_device().HasFeature(wgpu::FeatureName::TextureComponentSwizzle))
  {
    webgpu_draw_unsupported("texture channel swizzles on this WebGPU device");
  }
  sampling_swizzle_ = swizzle;
  sampling_view_.reset();
}

void WebGPUTexture::update_sub(int[3], int[3], eGPUDataFormat, GPUPixelBuffer *)
{
  webgpu_draw_unsupported("texture updates from pixel buffers");
}

}  // namespace blender::gpu
