/* SPDX-License-Identifier: GPL-2.0-or-later */

#include "webgpu_backend.hh"

#include "BLI_math_half.hh"

#include "GHOST_ContextWebGPU.hh"
#include "GHOST_SystemWeb.hh"
#include "GPU_work_in_flight.hh"
#include "MEM_guardedalloc.h"

#include "gpu_capabilities_private.hh"
#include "gpu_context_private.hh"
#include "gpu_immediate_private.hh"
#include "gpu_platform_private.hh"
#include "gpu_query.hh"
#include "gpu_storage_buffer_private.hh"
#include "gpu_texture_pool_private.hh"
#include "gpu_uniform_buffer_private.hh"
#include "webgpu_batch.hh"
#include "webgpu_buffer.hh"
#include "webgpu_context.hh"
#include "webgpu_geometry.hh"
#include "webgpu_shader.hh"
#include "webgpu_texture.hh"

#include <emscripten/promise.h>
#include <emscripten/threading.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <tuple>
#include <vector>

namespace blender::gpu {

[[noreturn]] static void unsupported(const char *operation)
{
  std::fprintf(stderr, "Blender's WebGPU backend does not implement %s yet.\n", operation);
  std::abort();
}

class WebGPUQueueCompletion {
 public:
  static std::shared_ptr<WebGPUQueueCompletion> request(const wgpu::Queue &queue)
  {
    auto completion = std::make_shared<WebGPUQueueCompletion>();
    queue.OnSubmittedWorkDone(
        wgpu::CallbackMode::AllowSpontaneous,
        [completion](wgpu::QueueWorkDoneStatus status, wgpu::StringView) {
          completion->status_ = status;
          if (completion->waiter_) {
            emscripten_promise_resolve(completion->waiter_, EM_PROMISE_FULFILL, nullptr);
          }
        });
    return completion;
  }

  void wait()
  {
    if (!status_) {
      BLI_assert(waiter_ == nullptr);
      waiter_ = emscripten_promise_create();
      (void)emscripten_promise_await_unchecked(waiter_);
      emscripten_promise_destroy(waiter_);
      waiter_ = nullptr;
    }
    if (status_ != wgpu::QueueWorkDoneStatus::Success) {
      std::fputs("WebGPU queue work failed.\n", stderr);
      std::abort();
    }
  }

 private:
  std::optional<wgpu::QueueWorkDoneStatus> status_;
  em_promise_t waiter_ = nullptr;
};

class WebGPUStateManager final : public StateManager {
 public:
  /* State is baked into WebGPU render pipelines and consumed when beginning a
   * pass. */
  void apply_state() override {}
  void force_state() override {}
  void issue_barrier(GPUBarrier bits) override;

  void texture_bind(Texture *texture, GPUSamplerState state, int slot) override;
  void texture_unbind(Texture *texture) override;
  void texture_unbind_all() override
  {
    textures_.clear();
    texels_.clear();
    emulated_sampler_mask_ = 0;
  }
  WebGPUSampledTexture bound_texture(int slot) const;
  uint64_t emulated_sampler_mask() const
  {
    return emulated_sampler_mask_;
  }
  uint8_t emulated_sampler_parameters(int slot) const;
  void texel_bind(int slot, const WebGPUTexelBuffer &texels, const Texture *owner = nullptr);
  WebGPUTexelBuffer bound_texels(int slot) const;
  void image_bind(Texture *texture, int slot) override;
  void image_unbind(Texture *texture) override;
  void image_unbind_all() override
  {
    images_.clear();
  }
  wgpu::TextureView bound_image(int slot) const;
  wgpu::Buffer bound_atomic_texture(int slot, bool image) const;

 private:
  struct TextureBinding {
    std::weak_ptr<wgpu::TextureView> view;
    wgpu::Sampler sampler;
    std::weak_ptr<wgpu::Buffer> atomic_buffer;
    uint8_t emulated_sampler_parameters = 0;
  };

  struct ImageBinding {
    const Texture *owner;
    std::weak_ptr<wgpu::TextureView> view;
    std::weak_ptr<wgpu::Buffer> atomic_buffer;
  };

  void set_emulated_sampler_bit(int slot, bool enabled);

  std::map<int, TextureBinding> textures_;
  std::map<int, ImageBinding> images_;
  std::map<int, std::pair<WebGPUTexelBuffer, const Texture *>> texels_;
  std::map<uint32_t, wgpu::Sampler> samplers_;
  uint64_t emulated_sampler_mask_ = 0;
};

static wgpu::AddressMode address_mode(const GPUSamplerExtendMode mode)
{
  switch (mode) {
    case GPU_SAMPLER_EXTEND_MODE_EXTEND:
    case GPU_SAMPLER_EXTEND_MODE_CLAMP_TO_BORDER:
      return wgpu::AddressMode::ClampToEdge;
    case GPU_SAMPLER_EXTEND_MODE_REPEAT:
      return wgpu::AddressMode::Repeat;
    case GPU_SAMPLER_EXTEND_MODE_MIRRORED_REPEAT:
      return wgpu::AddressMode::MirrorRepeat;
  }
  unsupported("this sampler address mode");
}

void WebGPUStateManager::set_emulated_sampler_bit(const int slot, const bool enabled)
{
  if (slot >= 0 && slot < 64) {
    const uint64_t bit = uint64_t(1) << slot;
    emulated_sampler_mask_ &= ~bit;
    emulated_sampler_mask_ |= enabled ? bit : 0;
  }
}

uint8_t WebGPUStateManager::emulated_sampler_parameters(const int slot) const
{
  const auto found = textures_.find(slot);
  return found == textures_.end() ? 0 : found->second.emulated_sampler_parameters;
}

void WebGPUStateManager::texture_bind(Texture *texture,
                                      const GPUSamplerState state,
                                      const int slot)
{
  if (slot < 0) {
    return;
  }
  auto &web_texture = static_cast<WebGPUTexture &>(*texture);
  if (texture->type_get() == GPU_TEXTURE_BUFFER) {
    texel_bind(slot, web_texture.texel_buffer(), texture);
    return;
  }
  texels_.erase(slot);
  if (const auto &buffer = web_texture.atomic_buffer()) {
    set_emulated_sampler_bit(slot, false);
    textures_[slot] = {{}, nullptr, buffer};
    return;
  }
  if (state.type != GPU_SAMPLER_STATE_TYPE_PARAMETERS) {
    unsupported("this custom texture sampler");
  }
  const bool linear = state.filtering & GPU_SAMPLER_FILTERING_LINEAR;
  const bool mipmap = state.filtering & GPU_SAMPLER_FILTERING_MIPMAP;
  const bool border = state.extend_x == GPU_SAMPLER_EXTEND_MODE_CLAMP_TO_BORDER ||
                      state.extend_yz == GPU_SAMPLER_EXTEND_MODE_CLAMP_TO_BORDER;
  /* The shader translator uses these bits to apply the sampler's addressing and LOD. */
  const uint8_t parameters = uint8_t(state.extend_x) | (uint8_t(state.extend_yz) << 2) |
                             (uint8_t(linear) << 4) | (uint8_t(mipmap) << 5);
  if ((texture->format_flag_get() & GPU_FORMAT_INTEGER) &&
      !(texture->format_flag_get() & GPU_FORMAT_SIGNED))
  {
    if (linear) {
      unsupported("linear sampling for unsigned integer textures");
    }
    set_emulated_sampler_bit(slot, parameters != 0);
    textures_[slot] = {web_texture.sampling_view(), nullptr, {}, parameters};
    return;
  }
  const uint32_t key = uint32_t(state.filtering) | (uint32_t(state.extend_x) << 8) |
                       (uint32_t(state.extend_yz) << 12);
  wgpu::Sampler &sampler = samplers_[key];
  if (!sampler) {
    wgpu::SamplerDescriptor desc{};
    desc.addressModeU = address_mode(state.extend_x);
    desc.addressModeV = desc.addressModeW = address_mode(state.extend_yz);
    desc.minFilter = desc.magFilter = linear ? wgpu::FilterMode::Linear :
                                               wgpu::FilterMode::Nearest;
    desc.mipmapFilter = linear && mipmap ? wgpu::MipmapFilterMode::Linear :
                                           wgpu::MipmapFilterMode::Nearest;
    if (!mipmap) {
      desc.lodMaxClamp = 0;
    }
    /* Border weights are computed for the isotropic texel footprint. */
    if (linear && mipmap && !border) {
      desc.maxAnisotropy = GPU_anisotropic_samples_get(state.filtering);
    }
    sampler = webgpu_active_device().CreateSampler(&desc);
  }
  set_emulated_sampler_bit(slot, border);
  textures_[slot] = {web_texture.sampling_view(), sampler, {}, border ? parameters : uint8_t(0)};
}

void WebGPUStateManager::texture_unbind(Texture *texture)
{
  if (texture->type_get() == GPU_TEXTURE_BUFFER) {
    std::erase_if(texels_,
                  [texture](const auto &binding) { return binding.second.second == texture; });
    return;
  }
  const auto &web_texture = static_cast<const WebGPUTexture &>(*texture);
  const auto &view = web_texture.cached_sampling_view();
  const auto &buffer = web_texture.atomic_buffer();
  std::erase_if(textures_, [&](const auto &binding) {
    const auto bound_view = binding.second.view.lock();
    const auto bound_buffer = binding.second.atomic_buffer.lock();
    const bool unbind = bound_buffer ? buffer == bound_buffer : !bound_view || view == bound_view;
    if (unbind) {
      set_emulated_sampler_bit(binding.first, false);
    }
    return unbind;
  });
}

void WebGPUStateManager::texel_bind(const int slot,
                                    const WebGPUTexelBuffer &texels,
                                    const Texture *owner)
{
  set_emulated_sampler_bit(slot, false);
  textures_.erase(slot);
  texels_[slot] = {texels, owner};
}

WebGPUTexelBuffer WebGPUStateManager::bound_texels(const int slot) const
{
  const auto found = texels_.find(slot);
  return found == texels_.end() ? WebGPUTexelBuffer{} : found->second.first;
}

WebGPUSampledTexture WebGPUStateManager::bound_texture(const int slot) const
{
  const auto found = textures_.find(slot);
  if (found == textures_.end()) {
    return {};
  }
  if (const auto view = found->second.view.lock()) {
    return {*view, found->second.sampler};
  }
  return {};
}

void WebGPUStateManager::image_bind(Texture *texture, const int slot)
{
  auto &web_texture = static_cast<WebGPUTexture &>(*texture);
  if (const auto &buffer = web_texture.atomic_buffer()) {
    images_[slot] = {texture, {}, buffer};
  }
  else {
    images_[slot] = {texture, web_texture.storage_view(), {}};
  }
}

void WebGPUStateManager::image_unbind(Texture *texture)
{
  std::erase_if(images_, [texture](const auto &binding) {
    return binding.second.owner == texture ||
           (binding.second.view.expired() && binding.second.atomic_buffer.expired());
  });
}

wgpu::TextureView WebGPUStateManager::bound_image(const int slot) const
{
  if (const auto found = images_.find(slot); found != images_.end()) {
    if (const auto view = found->second.view.lock()) {
      return *view;
    }
  }
  return {};
}

wgpu::Buffer WebGPUStateManager::bound_atomic_texture(const int slot, const bool image) const
{
  if (image) {
    if (const auto found = images_.find(slot); found != images_.end()) {
      if (const auto buffer = found->second.atomic_buffer.lock()) {
        return *buffer;
      }
    }
  }
  else if (const auto found = textures_.find(slot); found != textures_.end()) {
    if (const auto buffer = found->second.atomic_buffer.lock()) {
      return *buffer;
    }
  }
  return {};
}

class WebGPUContext;

class WebGPUQueryPool final : public QueryPool {
 public:
  ~WebGPUQueryPool() override;
  void init(GPUQueryType type) override;
  void begin_query() override;
  void end_query() override;
  void get_occlusion_result(MutableSpan<uint32_t> r_values) override;

  bool needs_new_set() const;
  const wgpu::QuerySet &query_set();
  void begin_pass_query(const wgpu::RenderPassEncoder &pass);
  void end_pass_query(const wgpu::RenderPassEncoder &pass);

 private:
  /* Page offsets are multiples of the WebGPU resolve destination alignment. */
  static constexpr uint32_t queries_per_set = 256;
  static constexpr uint64_t result_size = sizeof(uint64_t);
  static constexpr uint64_t result_set_stride = queries_per_set * result_size;

  struct QueryPage {
    wgpu::QuerySet set;
    std::vector<uint32_t> logical_indices;
  };

  WebGPUContext *context_ = nullptr;
  std::vector<QueryPage> pages_;
  std::optional<uint32_t> logical_query_;
  uint32_t query_count_ = 0;
  bool pass_query_open_ = false;
};

struct ClearPipelineKey {
  WebGPURenderTargets targets;
  uint32_t colors;
  bool depth;
  bool stencil;

  auto operator<=>(const ClearPipelineKey &) const = default;
};

static bool unsigned_color_format(const wgpu::TextureFormat format)
{
  using F = wgpu::TextureFormat;
  return ELEM(format,
              F::R8Uint,
              F::RG8Uint,
              F::RGBA8Uint,
              F::R16Uint,
              F::RG16Uint,
              F::RGBA16Uint,
              F::R32Uint,
              F::RG32Uint,
              F::RGBA32Uint);
}

class WebGPUFramebuffer final : public FrameBuffer {
 public:
  WebGPUFramebuffer(const char *name, WebGPUContext *window_context = nullptr)
      : FrameBuffer(name), window_context_(window_context)
  {
  }

  ~WebGPUFramebuffer() override;
  void bind(bool enabled_srgb) override;
  WebGPURenderTargets targets();
  wgpu::RenderPassEncoder begin_pass(const wgpu::CommandEncoder &encoder,
                                     Span<std::optional<double4>> clear_colors = {},
                                     std::optional<float> clear_depth = {},
                                     std::optional<uint32_t> clear_stencil = {},
                                     bool finish_loadstore = false,
                                     const wgpu::QuerySet &occlusion_queries = {});
  bool is_canvas() const
  {
    return window_context_ != nullptr;
  }
  bool needs_update() const
  {
    return dirty_attachments_;
  }
  int4 scissor_rect() const;
  bool check(char error[256]) override;
  void clear(GPUFrameBufferBits buffers, double4 color, float depth, uint stencil) override;
  void clear_multi(Span<double4> colors) override;
  void clear_attachment(GPUAttachmentType type, double4 value) override;

  void attachment_set_loadstore_op(GPUAttachmentType type, GPULoadStore action) override;
  void finish_loadstore(WebGPUContext &context);
  void subpass_transition_impl(GPUAttachmentState depth_state,
                               Span<GPUAttachmentState> color_states) override;
  wgpu::TextureView subpass_input(int attachment) const
  {
    return subpass_inputs_[attachment];
  }
  void read(GPUFrameBufferBits planes,
            eGPUDataFormat format,
            const int rect[4],
            int channels,
            int slot,
            void *data) override;
  void blit_to(GPUFrameBufferBits planes,
               int src_slot,
               FrameBuffer *destination,
               int dst_slot,
               int dst_x,
               int dst_y) override;

 private:
  bool update_attachments();
  void clear_attachments(Span<std::optional<double4>> colors,
                         std::optional<float> depth = {},
                         std::optional<uint32_t> stencil = {});

  WebGPUContext *window_context_;
  WebGPURenderTargets targets_;
  std::array<wgpu::TextureView, GPU_FB_MAX_ATTACHMENT> views_;
  wgpu::Texture raster_target_;
  std::array<std::optional<GPULoadStore>, GPU_FB_MAX_ATTACHMENT> loadstore_actions_{};
  std::array<wgpu::TextureView, GPU_FB_MAX_COLOR_ATTACHMENT> subpass_inputs_{};
  std::optional<uint32_t> subpass_colors_;
  bool subpass_depth_ = true;
  bool pending_attachment_loads_ = false;
  bool srgb_enabled_ = true;
  bool has_srgb_color_ = false;
  bool valid_ = false;
};

using BufferBindings = std::map<int, wgpu::Buffer>;

class WebGPUContext final : public Context {
 public:
  GHOST_ContextWebGPU &ghost;
  BufferBindings uniforms;
  BufferBindings storage;
  WebGPUBufferTransfers transfers;

  WebGPUContext(GHOST_IWindow *window, GHOST_ContextWebGPU &context)
      : ghost(context), transfers(context.device()), queue_(context.device().GetQueue())
  {
    ghost_window_ = window;
    state_manager = new WebGPUStateManager;
    back_left = active_fb = new WebGPUFramebuffer("WebGPU canvas", this);
    imm = new WebGPUImmediate;
  }

  ~WebGPUContext() override
  {
    finish_framebuffer();
    flush();
    free_resources();
  }

  static WebGPUContext &active()
  {
    auto *ctx = static_cast<WebGPUContext *>(Context::get());
    if (!ctx || !emscripten_is_main_runtime_thread()) {
      unsupported("resource access outside the active runtime-thread context");
    }
    return *ctx;
  }

  void activate() override
  {
    is_active_ = true;
    thread_ = pthread_self();
    immActivate();
  }
  void deactivate() override
  {
    finish_framebuffer();
    flush();
    immDeactivate();
    is_active_ = false;
  }

  void begin_frame() override {}
  void end_frame() override
  {
    finish_framebuffer();
    flush();
  }

  wgpu::CommandEncoder &encoder()
  {
    if (!encoder_) {
      encoder_ = ghost.device().CreateCommandEncoder();
    }
    return encoder_;
  }

  const wgpu::RenderPipeline &blit_pipeline(wgpu::TextureFormat format,
                                            wgpu::ColorWriteMask mask,
                                            bool flip_y)
  {
    const auto key = std::tuple(format, mask, flip_y);
    if (const auto found = blit_pipelines_.find(key); found != blit_pipelines_.end()) {
      return found->second;
    }
    const auto &shader = texture_copy_shader();
    wgpu::ConstantEntry orientation{};
    orientation.key = "flip_y";
    orientation.value = flip_y ? 1 : 0;
    wgpu::ColorTargetState target{};
    target.format = format;
    target.writeMask = mask;
    wgpu::FragmentState fragment{};
    fragment.module = shader;
    fragment.entryPoint = "color_frag";
    fragment.constantCount = 1;
    fragment.constants = &orientation;
    fragment.targetCount = 1;
    fragment.targets = &target;
    wgpu::RenderPipelineDescriptor desc{};
    desc.label = "Blender framebuffer blit";
    desc.vertex.module = shader;
    desc.vertex.entryPoint = "vert";
    desc.fragment = &fragment;
    return blit_pipelines_.emplace(key, ghost.device().CreateRenderPipeline(&desc)).first->second;
  }

  const wgpu::RenderPipeline &clear_pipeline(const ClearPipelineKey &key)
  {
    if (const auto found = clear_pipelines_.find(key); found != clear_pipelines_.end()) {
      return found->second;
    }
    const auto &device = ghost.device();
    std::ostringstream wgsl;
    wgsl << "@group(0) @binding(0) var<uniform> values: array<vec4u, "
         << GPU_FB_MAX_COLOR_ATTACHMENT + 1 << ">;\n"
         << "@vertex fn vert(@builtin(vertex_index) vertex: u32) -> @builtin(position) vec4f {\n"
            "  let p = vec2f(f32((vertex << 1u) & 2u), f32(vertex & 2u));\n"
            "  return vec4f(p * 2.0 - 1.0, "
         << (key.depth ?
                 "bitcast<f32>(values[" + std::to_string(GPU_FB_MAX_COLOR_ATTACHMENT) + "].x)" :
                 "0.0")
         << ", 1.0);\n}\n";

    std::array<wgpu::ColorTargetState, GPU_FB_MAX_COLOR_ATTACHMENT> colors{};
    uint32_t color_count = 0;
    if (key.colors != 0) {
      wgsl << "struct ClearColors {\n";
    }
    for (const int slot : IndexRange(colors.size())) {
      const auto format = key.targets.colors[slot];
      if (format == wgpu::TextureFormat::Undefined) {
        continue;
      }
      const bool write = (key.colors & (1u << slot)) != 0;
      colors[slot].format = format;
      colors[slot].writeMask = write ? wgpu::ColorWriteMask::All : wgpu::ColorWriteMask::None;
      color_count = slot + 1;
      if (write) {
        wgsl << "  @location(" << slot << ") color" << slot << ": "
             << (unsigned_color_format(format) ? "vec4u" : "vec4f") << ",\n";
      }
    }
    if (key.colors != 0) {
      wgsl << "}\n@fragment fn frag() -> ClearColors {\n  var out: ClearColors;\n";
      for (const int slot : IndexRange(color_count)) {
        if (key.colors & (1u << slot)) {
          wgsl << "  out.color" << slot << " = ";
          if (unsigned_color_format(colors[slot].format)) {
            wgsl << "values[" << slot << "];\n";
          }
          else {
            wgsl << "bitcast<vec4f>(values[" << slot << "]);\n";
          }
        }
      }
      wgsl << "  return out;\n}\n";
    }
    else {
      wgsl << "@fragment fn frag() {}\n";
    }

    const std::string code = wgsl.str();
    wgpu::ShaderSourceWGSL source{};
    source.code = code.c_str();
    wgpu::ShaderModuleDescriptor module{};
    module.nextInChain = &source;
    const auto shader = device.CreateShaderModule(&module);

    wgpu::DepthStencilState depth{};
    depth.format = key.targets.depth;
    depth.depthCompare = wgpu::CompareFunction::Always;
    depth.depthWriteEnabled = key.depth;
    if (key.targets.stencil) {
      depth.stencilFront.compare = wgpu::CompareFunction::Always;
      depth.stencilFront.passOp = key.stencil ? wgpu::StencilOperation::Replace :
                                                wgpu::StencilOperation::Keep;
      depth.stencilBack = depth.stencilFront;
      depth.stencilWriteMask = key.stencil ? 0xff : 0;
    }
    wgpu::FragmentState fragment{};
    fragment.module = shader;
    fragment.entryPoint = "frag";
    fragment.targetCount = color_count;
    fragment.targets = colors.data();
    wgpu::RenderPipelineDescriptor desc{};
    desc.label = "Blender scissored clear";
    desc.vertex.module = shader;
    desc.vertex.entryPoint = "vert";
    desc.fragment = &fragment;
    desc.depthStencil = depth.format == wgpu::TextureFormat::Undefined ? nullptr : &depth;
    return clear_pipelines_.emplace(key, device.CreateRenderPipeline(&desc)).first->second;
  }

  void generate_mipmaps(const wgpu::Texture &texture)
  {
    end_render_pass();
    const auto &device = ghost.device();
    const auto &pipeline = mipmap_pipeline(texture.GetFormat());
    const auto layout = pipeline.GetBindGroupLayout(0);
    wgpu::TextureViewDescriptor view{};
    view.dimension = wgpu::TextureViewDimension::e2D;
    view.mipLevelCount = 1;
    view.arrayLayerCount = 1;
    const uint32_t layers = texture.GetDepthOrArrayLayers();
    const uint32_t mips = texture.GetMipLevelCount();
    for (uint32_t layer = 0; layer < layers; layer++) {
      view.baseArrayLayer = layer;
      view.usage = wgpu::TextureUsage::TextureBinding;
      view.baseMipLevel = 0;
      auto source = texture.CreateView(&view);
      for (uint32_t mip = 1; mip < mips; mip++) {
        wgpu::BindGroupEntry entry{};
        entry.textureView = source;
        wgpu::BindGroupDescriptor group{};
        group.layout = layout;
        group.entryCount = 1;
        group.entries = &entry;

        view.baseMipLevel = mip;
        view.usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::TextureBinding;
        source = texture.CreateView(&view);
        wgpu::RenderPassColorAttachment color{};
        color.view = source;
        color.loadOp = wgpu::LoadOp::Clear;
        color.storeOp = wgpu::StoreOp::Store;
        wgpu::RenderPassDescriptor desc{};
        desc.label = "Blender mipmap generation";
        desc.colorAttachmentCount = 1;
        desc.colorAttachments = &color;
        auto pass = encoder().BeginRenderPass(&desc);
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, device.CreateBindGroup(&group));
        pass.Draw(3, 1, 0, 0);
        pass.End();
      }
    }
  }

  void upload_depth(const wgpu::Texture &texture,
                    const uint32_t mip,
                    const wgpu::Origin3D origin,
                    const wgpu::Extent3D size,
                    const uint32_t row_length,
                    const float *data)
  {
    const auto &device = ghost.device();
    const bool single_texel_mip = std::max(1u, texture.GetWidth() >> mip) == 1 &&
                                  std::max(1u, texture.GetHeight() >> mip) == 1;

    wgpu::Texture staging;
    wgpu::RenderPipeline pipeline;
    if (!single_texel_mip) {
      pipeline = depth_upload_pipeline(texture.GetFormat());
      wgpu::TextureDescriptor desc{};
      desc.label = "Blender depth upload source";
      desc.size = size;
      desc.format = wgpu::TextureFormat::R32Float;
      desc.usage = wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::TextureBinding;
      staging = device.CreateTexture(&desc);
      wgpu::TexelCopyTextureInfo copy{};
      copy.texture = staging;
      wgpu::TexelCopyBufferLayout layout{};
      layout.bytesPerRow = row_length * sizeof(float);
      layout.rowsPerImage = size.height;
      const size_t rows = size_t(size.height) * size.depthOrArrayLayers;
      const size_t bytes = (size_t(row_length) * (rows - 1) + size.width) * sizeof(float);
      /* This texture is private to the upload. No pending command reads from it
       * yet. */
      device.GetQueue().WriteTexture(&copy, data, bytes, &layout, &size);
    }

    end_render_pass();
    wgpu::TextureViewDescriptor view{};
    view.dimension = wgpu::TextureViewDimension::e2D;
    view.mipLevelCount = 1;
    view.arrayLayerCount = 1;
    for (uint32_t layer = 0; layer < size.depthOrArrayLayers; layer++) {
      view.baseMipLevel = mip;
      view.baseArrayLayer = origin.z + layer;
      wgpu::RenderPassDepthStencilAttachment depth{};
      depth.view = texture.CreateView(&view);
      depth.depthLoadOp = single_texel_mip ? wgpu::LoadOp::Clear : wgpu::LoadOp::Load;
      depth.depthStoreOp = wgpu::StoreOp::Store;
      /* EmdawnWebGPU forwards this field even when the depth attachment is
       * loaded. */
      depth.depthClearValue = 0.0f;
      if (single_texel_mip) {
        const float value = data[size_t(layer) * row_length];
        depth.depthClearValue = std::isnan(value) ? 0.0f : std::clamp(value, 0.0f, 1.0f);
      }

      wgpu::RenderPassDescriptor desc{};
      desc.label = "Blender depth upload";
      desc.depthStencilAttachment = &depth;
      auto pass = encoder().BeginRenderPass(&desc);
      if (!single_texel_mip) {
        view.baseMipLevel = 0;
        view.baseArrayLayer = layer;
        wgpu::BindGroupEntry source{};
        source.textureView = staging.CreateView(&view);
        wgpu::BindGroupDescriptor group{};
        group.layout = pipeline.GetBindGroupLayout(0);
        group.entryCount = 1;
        group.entries = &source;
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, device.CreateBindGroup(&group));
        pass.SetViewport(origin.x, origin.y, size.width, size.height, 0.0f, 1.0f);
        /* The fullscreen triangle extends outside the viewport. Limit writes to
         * the upload. */
        pass.SetScissorRect(origin.x, origin.y, size.width, size.height);
        pass.Draw(3, 1, 0, 0);
      }
      pass.End();
    }
  }

  void end_render_pass()
  {
    if (render_pass_) {
      if (query_pool_) {
        query_pool_->end_pass_query(render_pass_);
      }
      render_pass_.End();
      render_pass_ = nullptr;
      render_viewport_.reset();
      render_scissor_.reset();
      render_pipeline_ = nullptr;
      for (auto &binding : render_bindings_) {
        binding.group = nullptr;
        binding.offsets.clear();
      }
      render_stencil_reference_.reset();
    }
  }

  void finish_framebuffer()
  {
    end_render_pass();
    if (active_fb) {
      static_cast<WebGPUFramebuffer &>(*active_fb).finish_loadstore(*this);
    }
  }

  void clear_framebuffer(WebGPUFramebuffer &framebuffer,
                         const Span<std::optional<double4>> colors,
                         const std::optional<float> depth,
                         const std::optional<uint32_t> stencil)
  {
    end_render_pass();
    auto pass = framebuffer.begin_pass(encoder(), colors, depth, stencil);
    if (active_fb == &framebuffer && !query_pool_) {
      /* The next draw can use the cleared attachments without storing and reloading them. */
      render_pass_ = std::move(pass);
    }
    else {
      pass.End();
    }
  }

  wgpu::Buffer gather_indexed_points(const WebGPUIndexedPoints &points,
                                     const wgpu::Buffer &source,
                                     const size_t offset,
                                     const uint32_t stride)
  {
    end_render_pass();
    const auto &device = ghost.device();
    if (!point_gather_pipeline_) {
      wgpu::ShaderSourceWGSL code{};
      code.code = R"(
struct PointRange {
  first: u32,
  count: u32,
  base: u32,
  short_indices: u32,
  offset_words: u32,
  stride_words: u32,
}
@group(0) @binding(0) var<storage, read> indices: array<u32>;
@group(0) @binding(1) var<storage, read> source: array<u32>;
@group(0) @binding(2) var<storage, read_write> result: array<u32>;
@group(0) @binding(3) var<uniform> range: PointRange;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let point = id.x + id.y * groups.x * 64u;
  if point >= range.count { return; }
  let offset = range.first + point;
  var index: u32;
  if range.short_indices != 0u {
    index = (indices[offset / 2u] >> ((offset % 2u) * 16u)) & 65535u;
  } else {
    index = indices[offset];
  }
  index += range.base;
  if range.stride_words == 0u {
    result[point] = index;
  } else {
    let start = range.offset_words + index * range.stride_words;
    let out_start = point * range.stride_words;
    for (var word = 0u; word < range.stride_words; word++) {
      result[out_start + word] = source[start + word];
    }
  }
}
)";
      wgpu::ShaderModuleDescriptor module{};
      module.nextInChain = &code;
      wgpu::ComputePipelineDescriptor desc{};
      desc.label = "Blender indexed point gathering";
      desc.compute.module = device.CreateShaderModule(&module);
      desc.compute.entryPoint = "main";
      point_gather_pipeline_ = device.CreateComputePipeline(&desc);
    }
    const std::array<uint32_t, 6> range{points.first,
                                        points.count,
                                        points.base,
                                        uint32_t(points.short_indices),
                                        uint32_t(offset / 4),
                                        stride / 4};
    const auto uniforms = upload_draw_uniforms(std::as_bytes(std::span(range)));
    wgpu::BufferDescriptor buffer{};
    buffer.label = "Blender gathered point attributes";
    buffer.size = uint64_t(points.count) * (source ? stride : 4);
    buffer.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::Vertex;
    const auto result = device.CreateBuffer(&buffer);
    std::array<wgpu::BindGroupEntry, 4> entries{};
    entries[0].buffer = points.indices;
    entries[1].buffer = source ? source : points.indices;
    entries[2].buffer = result;
    entries[3].buffer = uniforms.buffer;
    entries[3].offset = uniforms.offset;
    entries[3].size = sizeof(range);
    for (uint32_t i = 0; i < entries.size(); i++) {
      entries[i].binding = i;
    }
    wgpu::BindGroupDescriptor group{};
    group.layout = point_gather_pipeline_.GetBindGroupLayout(0);
    group.entryCount = entries.size();
    group.entries = entries.data();
    auto pass = encoder().BeginComputePass();
    pass.SetPipeline(point_gather_pipeline_);
    pass.SetBindGroup(0, device.CreateBindGroup(&group));
    const uint32_t groups = (points.count + 63) / 64;
    const uint32_t groups_x = std::min(groups, 65535u);
    pass.DispatchWorkgroups(groups_x, (groups + groups_x - 1) / groups_x);
    pass.End();
    return result;
  }

  WebGPUPointCommands expand_point_commands(const wgpu::Buffer &input,
                                            const uint32_t first_word,
                                            const uint32_t stride_words,
                                            const uint32_t count,
                                            const uint32_t point_capacity,
                                            const bool indexed)
  {
    const auto &device = ghost.device();
    if (!point_commands_pipeline_) {
      wgpu::ShaderSourceWGSL code{};
      code.code = R"(
struct InputRange {
  first_word: u32,
  stride_words: u32,
  count: u32,
  point_capacity: u32,
  state_words: u32,
  indexed: u32,
}
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> draws: array<vec4u>;
@group(0) @binding(2) var<storage, read_write> state: array<u32>;
@group(0) @binding(3) var<uniform> range: InputRange;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
  let command = id.x + id.y * groups.x * 64u;
  if command >= range.count { return; }
  let src = range.first_word + command * range.stride_words;
  let dst = command * range.state_words;
  let first_point = input[src + 2u];
  var points = 0u;
  if first_point < range.point_capacity {
    points = min(input[src], range.point_capacity - first_point);
  }
  // Vertex fetch and the first instance are handled by the vertex stage.
  draws[command] = vec4u(points * 6u, input[src + 1u], 0u, 0u);
  state[dst] = first_point;
  if range.indexed != 0u {
    state[dst + 1u] = input[src + 3u];
    state[dst + 2u] = input[src + 4u];
  } else {
    state[dst + 1u] = 0u;
    state[dst + 2u] = input[src + 3u];
  }
}
)";
      wgpu::ShaderModuleDescriptor shader{};
      shader.label = "Blender indirect point command expansion";
      shader.nextInChain = &code;
      wgpu::ComputePipelineDescriptor desc{};
      desc.label = shader.label;
      desc.compute.module = device.CreateShaderModule(&shader);
      desc.compute.entryPoint = "main";
      point_commands_pipeline_ = device.CreateComputePipeline(&desc);
      if (device.GetLimits(&point_command_limits_) != wgpu::Status::Success) {
        unsupported("reading limits for indirect point draws");
      }
    }

    const uint32_t state_stride = std::max(16u,
                                           point_command_limits_.minUniformBufferOffsetAlignment);
    const uint64_t state_size = uint64_t(count) * state_stride;
    if (state_size > point_command_limits_.maxStorageBufferBindingSize || state_size > UINT32_MAX)
    {
      unsupported("this indirect point command count");
    }
    const std::array<uint32_t, 6> range{
        first_word, stride_words, count, point_capacity, state_stride / 4, uint32_t(indexed)};
    const auto uniform = upload_draw_uniforms(std::as_bytes(std::span(range)));

    wgpu::BufferDescriptor buffer{};
    buffer.label = "Blender expanded point draw arguments";
    buffer.size = uint64_t(count) * 4 * sizeof(uint32_t);
    buffer.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::Indirect;
    const auto draw_arguments = device.CreateBuffer(&buffer);
    buffer.label = "Blender expanded point shader arguments";
    buffer.size = state_size;
    buffer.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::Uniform;
    const auto shader_arguments = device.CreateBuffer(&buffer);

    std::array<wgpu::BindGroupEntry, 4> entries{};
    entries[0].buffer = input;
    entries[1].buffer = draw_arguments;
    entries[2].buffer = shader_arguments;
    entries[3].buffer = uniform.buffer;
    entries[3].offset = uniform.offset;
    entries[3].size = sizeof(range);
    for (uint32_t i = 0; i < entries.size(); i++) {
      entries[i].binding = i;
    }
    wgpu::BindGroupDescriptor group{};
    group.layout = point_commands_pipeline_.GetBindGroupLayout(0);
    group.entryCount = entries.size();
    group.entries = entries.data();

    end_render_pass();
    auto pass = encoder().BeginComputePass();
    pass.SetPipeline(point_commands_pipeline_);
    pass.SetBindGroup(0, device.CreateBindGroup(&group));
    const uint32_t groups = (count + 63u) / 64u;
    const uint32_t groups_x = std::min(groups,
                                       point_command_limits_.maxComputeWorkgroupsPerDimension);
    pass.DispatchWorkgroups(groups_x, (groups + groups_x - 1u) / groups_x);
    pass.End();
    return {draw_arguments, shader_arguments, state_stride};
  }

  wgpu::RenderPassEncoder render_pass()
  {
    auto &framebuffer = static_cast<WebGPUFramebuffer &>(*active_fb);
    if (framebuffer.needs_update()) {
      end_render_pass();
    }
    if (!framebuffer.check(nullptr)) {
      return nullptr;
    }
    const int height = framebuffer.size_get().y;
    int viewport[4];
    active_fb->viewport_get(viewport);
    if (viewport[2] <= 0 || viewport[3] <= 0) {
      return nullptr;
    }
    const int64_t viewport_y = framebuffer.is_canvas() ?
                                   int64_t(height) - viewport[1] - viewport[3] :
                                   viewport[1];

    const int4 scissor = framebuffer.scissor_rect();
    if (scissor.z == 0 || scissor.w == 0) {
      return nullptr;
    }

    if (query_pool_ && query_pool_->needs_new_set()) {
      end_render_pass();
    }
    if (!render_pass_) {
      render_pass_ = query_pool_ ?
                         framebuffer.begin_pass(encoder(), {}, {}, {}, false, query_pool_->query_set()) :
                         framebuffer.begin_pass(encoder());
    }
    if (render_pass_) {
      const std::array<float, 4> rect = {
          float(viewport[0]), float(viewport_y), float(viewport[2]), float(viewport[3])};
      if (render_viewport_ != rect) {
        render_pass_.SetViewport(rect[0], rect[1], rect[2], rect[3], 0, 1);
        render_viewport_ = rect;
      }
      if (render_scissor_ != scissor) {
        render_pass_.SetScissorRect(scissor.x, scissor.y, scissor.z, scissor.w);
        render_scissor_ = scissor;
      }
      if (query_pool_) {
        query_pool_->begin_pass_query(render_pass_);
      }
    }
    return render_pass_;
  }

  void set_render_pipeline(const wgpu::RenderPipeline &pipeline)
  {
    if (render_pipeline_.Get() != pipeline.Get()) {
      render_pass_.SetPipeline(pipeline);
      render_pipeline_ = pipeline;
    }
  }

  void set_render_bind_group(const uint32_t index,
                              const wgpu::BindGroup &group,
                              const std::span<const uint32_t> offsets)
  {
    auto &binding = render_bindings_[index];
    if (binding.group.Get() == group.Get() &&
        std::equal(offsets.begin(), offsets.end(), binding.offsets.begin(), binding.offsets.end()))
    {
      return;
    }
    render_pass_.SetBindGroup(index, group, offsets.size(), offsets.data());
    binding.group = group;
    binding.offsets.assign(offsets.begin(), offsets.end());
  }

  void set_render_stencil_reference(const uint32_t reference)
  {
    if (render_stencil_reference_ != reference) {
      render_pass_.SetStencilReference(reference);
      render_stencil_reference_ = reference;
    }
  }

  wgpu::ComputePassEncoder begin_compute_pass()
  {
    end_render_pass();
    auto &compute = static_cast<WebGPUShader &>(*shader);
    const StringRefNull shader_name = compute.name_get();
    int light_write_slot = -1;
    if (shader_name == "eevee_shadow_tilemap_bounds") {
      light_write_slot = 8;
    }
    else if (shader_name == "eevee_shadow_tilemap_amend") {
      light_write_slot = 5;
    }

    wgpu::Buffer original_lights;
    auto light_binding = storage.end();
    if (light_write_slot >= 0) {
      /* LightRenderData describes the culled light buffer at slot 1. The shadow
       * compute shaders read immutable light metadata through this binding and
       * update clip distances or LOD through a separate binding. The metadata
       * reads cannot use a writable binding because they control barriers. */
      light_binding = storage.find(1);
      const auto writes = storage.find(light_write_slot);
      if (light_binding != storage.end() && writes != storage.end() &&
          light_binding->second.Get() == writes->second.Get())
      {
        original_lights = light_binding->second;
        const uint64_t size = original_lights.GetSize();
        if (!shadow_lights_snapshot_ || shadow_lights_snapshot_.GetSize() != size) {
          wgpu::BufferDescriptor desc{};
          desc.label = "EEVEE shadow light metadata";
          desc.size = size;
          desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::Storage;
          shadow_lights_snapshot_ = ghost.device().CreateBuffer(&desc);
        }
        /* Encode the copy between dispatches so GPU updates are visible without
         * a separate submission or a CPU wait. */
        encoder().CopyBufferToBuffer(original_lights, 0, shadow_lights_snapshot_, 0, size);
        light_binding->second = shadow_lights_snapshot_;
      }
    }
    auto pass = encoder().BeginComputePass();
    compute.bind_compute_pipeline(pass);
    if (original_lights) {
      light_binding->second = original_lights;
    }
    return pass;
  }

  const wgpu::Buffer &default_vertex_buffer()
  {
    if (!default_vertex_buffer_) {
      wgpu::BufferDescriptor desc{};
      desc.label = "Blender default vertex attributes";
      desc.size = sizeof(uint32_t);
      desc.usage = wgpu::BufferUsage::Vertex;
      default_vertex_buffer_ = ghost.device().CreateBuffer(&desc);
    }
    return default_vertex_buffer_;
  }

  const wgpu::Buffer &triangle_fan_indices(const uint32_t index_count)
  {
    const uint64_t bytes = uint64_t(index_count) * sizeof(uint32_t);
    if (!triangle_fan_indices_ || triangle_fan_indices_.GetSize() < bytes) {
      wgpu::Limits limits{};
      if (ghost.device().GetLimits(&limits) != wgpu::Status::Success ||
          bytes > limits.maxBufferSize || bytes > SIZE_MAX)
      {
        webgpu_draw_unsupported("this triangle fan index buffer size");
      }

      /* Every fan starts at index zero, so shorter fans can reuse a prefix of this buffer. */
      std::vector<uint32_t> indices(index_count);
      for (uint32_t i = 0; i < index_count / 3; i++) {
        indices[3 * i] = 0;
        indices[3 * i + 1] = i + 1;
        indices[3 * i + 2] = i + 2;
      }
      const WebGPUBuffer buffer(
          ghost.device(), size_t(bytes), wgpu::BufferUsage::Index, "Blender triangle fan indices");
      buffer.write(indices.data());
      triangle_fan_indices_ = buffer.handle();
    }
    return triangle_fan_indices_;
  }

  WebGPUUniformSlice upload_draw_uniforms(std::span<const std::byte> bytes)
  {
    return upload(bytes,
                  uniform_pages_,
                  64 * 1024,
                  256,
                  wgpu::BufferUsage::Uniform,
                  "Blender draw uniforms");
  }

  WebGPUUniformSlice upload_immediate_vertices(std::span<const std::byte> bytes)
  {
    return upload(bytes,
                  vertex_pages_,
                  256 * 1024,
                  4,
                  wgpu::BufferUsage::Vertex | wgpu::BufferUsage::Storage,
                  "Blender immediate vertices");
  }

  void update_buffer(const WebGPUBuffer &buffer, const void *data, const size_t size)
  {
    if (size == 0) {
      return;
    }
    constexpr size_t transfer_page_size = 256 * 1024;
    if (size > transfer_page_size || size > buffer.size() ||
        (size % 4 != 0 && size != buffer.size()))
    {
      /* Large writes go straight to the queue. Partial words use the byte-range transfer path
       * to preserve adjacent bytes. */
      flush();
      transfers.write(buffer, 0, data, size);
      return;
    }
    end_render_pass();
    const auto bytes = std::span(static_cast<const std::byte *>(data), size);
    const WebGPUUniformSlice slice = upload(
        bytes, transfer_pages_, transfer_page_size, 4, wgpu::BufferUsage::CopySrc, "Blender buffer uploads");
    encoder().CopyBufferToBuffer(
        slice.buffer, slice.offset, buffer.handle(), 0, (size + 3) & ~size_t(3));
  }

  void clear_buffer(const WebGPUBuffer &buffer)
  {
    end_render_pass();
    encoder().ClearBuffer(buffer.handle(), 0, buffer.allocation_size());
  }

  void flush() override
  {
    end_render_pass();
    if (!encoder_) {
      return;
    }
    /* Upload pages are written once up front. Copies in the command buffer apply updates
     * in order, so each draw still sees the contents that were current when it was recorded. */
    uniform_pages_.submit_uploads(queue_);
    vertex_pages_.submit_uploads(queue_);
    transfer_pages_.submit_uploads(queue_);
    const wgpu::CommandBuffer commands = encoder_.Finish();
    queue_.Submit(1, &commands);
    encoder_ = nullptr;
  }
  void finish() override
  {
    flush();
    WebGPUQueueCompletion::request(queue_)->wait();
  }
  void memory_statistics_get(int *total, int *free) override
  {
    *total = *free = 0;
  }
  bool debug_capture_begin(const char *) override
  {
    return false;
  }
  void debug_capture_end() override {}
  void *debug_capture_scope_create(const char *) override
  {
    return nullptr;
  }
  bool debug_capture_scope_begin(void *) override
  {
    return false;
  }
  void debug_capture_scope_end(void *) override {}
  void debug_unbind_all_ubo() override
  {
    uniforms.clear();
  }
  void debug_unbind_all_ssbo() override
  {
    storage.clear();
  }

 private:
  friend class WebGPUQueryPool;

  const wgpu::Queue queue_;
  wgpu::CommandEncoder encoder_;
  wgpu::RenderPassEncoder render_pass_;
  std::optional<std::array<float, 4>> render_viewport_;
  std::optional<int4> render_scissor_;
  wgpu::RenderPipeline render_pipeline_;
  struct RenderBinding {
    wgpu::BindGroup group;
    std::vector<uint32_t> offsets;
  };
  std::array<RenderBinding, 4> render_bindings_;
  std::optional<uint32_t> render_stencil_reference_;
  WebGPUQueryPool *query_pool_ = nullptr;
  wgpu::Buffer default_vertex_buffer_;
  wgpu::Buffer triangle_fan_indices_;
  wgpu::Buffer shadow_lights_snapshot_;
  wgpu::ComputePipeline point_gather_pipeline_;
  wgpu::ComputePipeline point_commands_pipeline_;
  wgpu::Limits point_command_limits_;
  wgpu::ShaderModule texture_copy_shader_;
  std::map<std::tuple<wgpu::TextureFormat, wgpu::ColorWriteMask, bool>, wgpu::RenderPipeline>
      blit_pipelines_;
  std::map<ClearPipelineKey, wgpu::RenderPipeline> clear_pipelines_;
  std::map<wgpu::TextureFormat, wgpu::RenderPipeline> depth_upload_pipelines_;
  std::map<wgpu::TextureFormat, wgpu::RenderPipeline> mipmap_pipelines_;
  wgpu::PipelineLayout mipmap_layout_;

  const wgpu::ShaderModule &texture_copy_shader()
  {
    if (!texture_copy_shader_) {
      wgpu::ShaderSourceWGSL source{};
      source.code = R"(
@group(0) @binding(0) var image: texture_2d<f32>;
override flip_y: bool = false;

struct Vertex {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

@vertex fn vert(@builtin(vertex_index) index: u32) -> Vertex {
  let uv = vec2f(f32((index << 1u) & 2u), f32(index & 2u));
  return Vertex(vec4f(uv.x * 2 - 1, 1 - uv.y * 2, 0, 1), uv);
}

fn load_texel(uv: vec2f, flip: bool) -> vec4f {
  let size = vec2i(textureDimensions(image));
  var pixel = clamp(vec2i(uv * vec2f(size)), vec2i(0), size - 1);
  if flip { pixel.y = size.y - pixel.y - 1; }
  return textureLoad(image, pixel, 0);
}

@fragment fn color_frag(input: Vertex) -> @location(0) vec4f {
  return load_texel(input.uv, flip_y);
}

@fragment fn depth_frag(input: Vertex) -> @builtin(frag_depth) f32 {
  return clamp(load_texel(input.uv, false).r, 0.0, 1.0);
}

@fragment fn mipmap_frag(input: Vertex) -> @location(0) vec4f {
  let src_size = vec2i(textureDimensions(image));
  let dst_size = max(src_size / 2, vec2i(1));
  let dst = vec2i(input.position.xy);
  if all(src_size == dst_size * 2) {
    let pixel = dst * 2;
    return 0.25 * textureLoad(image, pixel, 0) +
           0.25 * textureLoad(image, pixel + vec2i(1, 0), 0) +
           0.25 * textureLoad(image, pixel + vec2i(0, 1), 0) +
           0.25 * textureLoad(image, pixel + vec2i(1, 1), 0);
  }

  // Weight each source texel by its overlap with the destination footprint.
  // Integer boundaries include the last row or column of odd-sized mips.
  let lo = dst * src_size;
  let hi = lo + src_size;
  let first = lo / dst_size;
  let end = (hi + dst_size - 1) / dst_size;
  var color = vec4f(0);
  for (var y = first.y; y < end.y; y++) {
    let y_overlap = min(hi.y, (y + 1) * dst_size.y) - max(lo.y, y * dst_size.y);
    for (var x = first.x; x < end.x; x++) {
      let x_overlap = min(hi.x, (x + 1) * dst_size.x) - max(lo.x, x * dst_size.x);
      let weight = f32(x_overlap * y_overlap) / f32(src_size.x * src_size.y);
      color += weight * textureLoad(image, vec2i(x, y), 0);
    }
  }
  return color;
}
)";
      wgpu::ShaderModuleDescriptor desc{};
      desc.label = "Blender texture transfer";
      desc.nextInChain = &source;
      texture_copy_shader_ = ghost.device().CreateShaderModule(&desc);
    }
    return texture_copy_shader_;
  }

  const wgpu::RenderPipeline &mipmap_pipeline(const wgpu::TextureFormat format)
  {
    if (const auto found = mipmap_pipelines_.find(format); found != mipmap_pipelines_.end()) {
      return found->second;
    }
    const auto &device = ghost.device();
    if (!mipmap_layout_) {
      wgpu::BindGroupLayoutEntry entry{};
      entry.visibility = wgpu::ShaderStage::Fragment;
      entry.texture.sampleType = wgpu::TextureSampleType::UnfilterableFloat;
      entry.texture.viewDimension = wgpu::TextureViewDimension::e2D;
      wgpu::BindGroupLayoutDescriptor group{};
      group.entryCount = 1;
      group.entries = &entry;
      const auto binding_layout = device.CreateBindGroupLayout(&group);
      wgpu::PipelineLayoutDescriptor layout{};
      layout.bindGroupLayoutCount = 1;
      layout.bindGroupLayouts = &binding_layout;
      mipmap_layout_ = device.CreatePipelineLayout(&layout);
    }
    const auto &shader = texture_copy_shader();
    wgpu::ColorTargetState target{};
    target.format = format;
    wgpu::FragmentState fragment{};
    fragment.module = shader;
    fragment.entryPoint = "mipmap_frag";
    fragment.targetCount = 1;
    fragment.targets = &target;
    wgpu::RenderPipelineDescriptor desc{};
    desc.label = "Blender mipmap generation";
    desc.layout = mipmap_layout_;
    desc.vertex.module = shader;
    desc.vertex.entryPoint = "vert";
    desc.fragment = &fragment;
    return mipmap_pipelines_.emplace(format, device.CreateRenderPipeline(&desc)).first->second;
  }

  const wgpu::RenderPipeline &depth_upload_pipeline(const wgpu::TextureFormat format)
  {
    if (const auto cached = depth_upload_pipelines_.find(format);
        cached != depth_upload_pipelines_.end())
    {
      return cached->second;
    }
    const auto &device = ghost.device();
    const auto &shader = texture_copy_shader();
    wgpu::DepthStencilState depth{};
    depth.format = format;
    depth.depthWriteEnabled = true;
    depth.depthCompare = wgpu::CompareFunction::Always;
    wgpu::FragmentState fragment{};
    fragment.module = shader;
    fragment.entryPoint = "depth_frag";
    wgpu::RenderPipelineDescriptor desc{};
    desc.label = "Blender depth upload";
    desc.vertex.module = shader;
    desc.vertex.entryPoint = "vert";
    desc.depthStencil = &depth;
    desc.fragment = &fragment;
    return depth_upload_pipelines_.emplace(format, device.CreateRenderPipeline(&desc))
        .first->second;
  }

  struct UploadPages {
    struct Page {
      wgpu::Buffer buffer;
      size_t capacity = 0;
      std::vector<std::byte> bytes;
    };
    std::vector<Page> buffers;
    size_t page = 0;

    void submit_uploads(const wgpu::Queue &queue)
    {
      for (size_t i = 0; i < buffers.size() && i <= page; i++) {
        Page &upload = buffers[i];
        if (!upload.bytes.empty()) {
          queue.WriteBuffer(upload.buffer, 0, upload.bytes.data(), upload.bytes.size());
          upload.bytes.clear();
        }
      }
      page = 0;
    }
  };
  UploadPages uniform_pages_;
  UploadPages vertex_pages_;
  UploadPages transfer_pages_;

  WebGPUUniformSlice upload(std::span<const std::byte> bytes,
                            UploadPages &pages,
                            const size_t default_size,
                            const size_t alignment,
                            const wgpu::BufferUsage usage,
                            const char *label)
  {
    const size_t required = (bytes.size() + alignment - 1) & ~(alignment - 1);
    if (pages.page < pages.buffers.size()) {
      const auto &page = pages.buffers[pages.page];
      if (!page.bytes.empty() && required > page.capacity - page.bytes.size()) {
        pages.page++;
      }
    }
    if (pages.page == pages.buffers.size()) {
      pages.buffers.emplace_back();
    }
    auto &page = pages.buffers[pages.page];
    if (!page.buffer || page.capacity < required) {
      wgpu::BufferDescriptor desc{};
      desc.label = label;
      desc.size = std::max(default_size, required);
      desc.usage = usage | wgpu::BufferUsage::CopyDst;
      page.buffer = ghost.device().CreateBuffer(&desc);
      page.capacity = desc.size;
      page.bytes.reserve(page.capacity);
    }
    const size_t offset = page.bytes.size();
    page.bytes.insert(page.bytes.end(), bytes.begin(), bytes.end());
    page.bytes.resize(offset + required);
    return {page.buffer, uint32_t(offset)};
  }
};

WebGPUQueryPool::~WebGPUQueryPool()
{
  if (context_) {
    context_->end_render_pass();
    context_->query_pool_ = nullptr;
  }
}

void WebGPUQueryPool::init(const GPUQueryType type)
{
  if (type != GPU_QUERY_OCCLUSION) {
    unsupported("this query type");
  }
  auto &context = WebGPUContext::active();
  BLI_assert(!context_ && !context.query_pool_);
  context.end_render_pass();
  context.query_pool_ = this;
  context_ = &context;
}

void WebGPUQueryPool::begin_query()
{
  BLI_assert(context_ && !logical_query_);
  logical_query_ = query_count_++;
}

void WebGPUQueryPool::end_query()
{
  BLI_assert(context_ && logical_query_);
  end_pass_query(context_->render_pass_);
  logical_query_.reset();
}

bool WebGPUQueryPool::needs_new_set() const
{
  return logical_query_ && !pass_query_open_ && !pages_.empty() &&
         pages_.back().logical_indices.size() == queries_per_set;
}

const wgpu::QuerySet &WebGPUQueryPool::query_set()
{
  if (pages_.empty() || needs_new_set()) {
    auto &page = pages_.emplace_back();
    page.logical_indices.reserve(queries_per_set);
    wgpu::QuerySetDescriptor desc{};
    desc.label = "Blender occlusion queries";
    desc.type = wgpu::QueryType::Occlusion;
    desc.count = queries_per_set;
    page.set = context_->ghost.device().CreateQuerySet(&desc);
  }
  return pages_.back().set;
}

void WebGPUQueryPool::begin_pass_query(const wgpu::RenderPassEncoder &pass)
{
  if (!logical_query_ || pass_query_open_) {
    return;
  }
  auto &indices = pages_.back().logical_indices;
  pass.BeginOcclusionQuery(uint32_t(indices.size()));
  indices.push_back(*logical_query_);
  pass_query_open_ = true;
}

void WebGPUQueryPool::end_pass_query(const wgpu::RenderPassEncoder &pass)
{
  if (pass_query_open_) {
    BLI_assert(pass);
    pass.EndOcclusionQuery();
    pass_query_open_ = false;
  }
}

void WebGPUQueryPool::get_occlusion_result(MutableSpan<uint32_t> r_values)
{
  BLI_assert(context_ && !logical_query_ && r_values.size() == query_count_);
  std::fill(r_values.begin(), r_values.end(), 0u);

  auto &context = *context_;
  context.end_render_pass();
  context.query_pool_ = nullptr;
  context_ = nullptr;

  if (pages_.empty() || (pages_.size() == 1 && pages_.front().logical_indices.empty())) {
    return;
  }

  const uint64_t size = (pages_.size() - 1) * result_set_stride +
                        pages_.back().logical_indices.size() * result_size;
  const auto &device = context.ghost.device();
  wgpu::BufferDescriptor result_desc{};
  result_desc.label = "Blender resolved occlusion queries";
  result_desc.size = size;
  result_desc.usage = wgpu::BufferUsage::QueryResolve | wgpu::BufferUsage::CopySrc;
  const auto results = device.CreateBuffer(&result_desc);

  wgpu::BufferDescriptor staging_desc{};
  staging_desc.label = "Blender occlusion readback";
  staging_desc.size = size;
  staging_desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
  const auto staging = device.CreateBuffer(&staging_desc);

  const auto &encoder = context.encoder();
  for (size_t i = 0; i < pages_.size(); ++i) {
    const auto &page = pages_[i];
    if (!page.logical_indices.empty()) {
      encoder.ResolveQuerySet(
          page.set, 0, uint32_t(page.logical_indices.size()), results, i * result_set_stride);
    }
  }
  encoder.CopyBufferToBuffer(results, 0, staging, 0, size);
  context.flush();

  const em_promise_t completion = emscripten_promise_create();
  staging.MapAsync(
      wgpu::MapMode::Read,
      0,
      size,
      wgpu::CallbackMode::AllowSpontaneous,
      [this, staging, size, r_values, completion](wgpu::MapAsyncStatus status, wgpu::StringView) {
        if (status == wgpu::MapAsyncStatus::Success) {
          const auto *raw = static_cast<const std::byte *>(staging.GetConstMappedRange(0, size));
          if (raw) {
            for (size_t i = 0; i < pages_.size(); ++i) {
              const auto &indices = pages_[i].logical_indices;
              for (size_t j = 0; j < indices.size(); ++j) {
                uint64_t value;
                std::memcpy(&value, raw + i * result_set_stride + j * result_size, result_size);
                /* A logical query may have spanned several render passes. */
                r_values[indices[j]] |= uint32_t(value != 0);
              }
            }
          }
          staging.Unmap();
        }
        else {
          std::fputs("WebGPU occlusion query mapping failed.\n", stderr);
        }
        emscripten_promise_resolve(completion, EM_PROMISE_FULFILL, nullptr);
      });
  (void)emscripten_promise_await_unchecked(completion);
  emscripten_promise_destroy(completion);
}

WebGPUFramebuffer::~WebGPUFramebuffer()
{
  if (auto *context = static_cast<WebGPUContext *>(Context::get());
      context && context->active_fb == this)
  {
    context->finish_framebuffer();
    context->active_fb = is_canvas() ? nullptr : context->back_left;
  }
}

bool WebGPUFramebuffer::update_attachments()
{
  if (is_canvas()) {
    const auto &texture = window_context_->ghost.drawable();
    if (!texture) {
      return false;
    }
    const int width = texture.GetWidth();
    const int height = texture.GetHeight();
    if (width != width_ || height != height_) {
      size_set(width, height);
      viewport_reset();
      scissor_reset();
    }
    targets_.colors[0] = texture.GetFormat();
    dirty_attachments_ = false;
    return true;
  }
  if (!dirty_attachments_) {
    return valid_;
  }
  if (auto *context = static_cast<WebGPUContext *>(Context::get());
      context && context->active_fb == this)
  {
    context->finish_framebuffer();
  }
  int2 size(0);
  targets_ = {};
  targets_.texture_coordinates = true;
  views_ = {};
  valid_ = false;
  has_srgb_color_ = false;
  dirty_attachments_ = false;
  for (int i = 0; i < GPU_FB_MAX_ATTACHMENT; i++) {
    const auto &attachment = attachments_[i];
    if (!attachment.tex) {
      continue;
    }
    const auto &texture = static_cast<const WebGPUTexture &>(*attachment.tex);
    if (!(texture.usage_get() & GPU_TEXTURE_USAGE_ATTACHMENT) || attachment.mip < 0 ||
        attachment.mip >= texture.mip_count() || attachment.layer >= texture.layer_count())
    {
      return false;
    }
    const int2 attachment_size(
        texture.mip_width_get(attachment.mip),
        texture.type_get() == GPU_TEXTURE_1D_ARRAY ? 1 : texture.mip_height_get(attachment.mip));
    if (size.x && size != attachment_size) {
      return false;
    }
    size = attachment_size;
    auto format = texture.webgpu_format();
    if (i < GPU_FB_COLOR_ATTACHMENT0) {
      targets_.depth = format;
      targets_.stencil = i == GPU_FB_DEPTH_STENCIL_ATTACHMENT;
    }
    else {
      if (format == wgpu::TextureFormat::RGBA8UnormSrgb) {
        has_srgb_color_ = true;
        if (!srgb_enabled_) {
          format = wgpu::TextureFormat::RGBA8Unorm;
        }
      }
      targets_.colors[i - GPU_FB_COLOR_ATTACHMENT0] = format;
    }
    views_[i] = texture.attachment_view(
        attachment.mip, attachment.layer, wgpu::TextureUsage::RenderAttachment, format);
  }
  if (size.x == 0 && width_ > 0 && height_ > 0) {
    /* Storage-only raster passes still need a raster extent in WebGPU. A single-channel
     * attachment supplies it, and its pixels can be discarded at the end of every pass. */
    size = {width_, height_};
    if (!raster_target_ || raster_target_.GetWidth() != uint32_t(width_) ||
        raster_target_.GetHeight() != uint32_t(height_))
    {
      wgpu::TextureDescriptor descriptor{};
      descriptor.label = "Blender storage-only raster target";
      descriptor.usage = wgpu::TextureUsage::RenderAttachment;
      if (EM_ASM_INT({ return typeof GPUTextureUsage.TRANSIENT_ATTACHMENT === "number"; })) {
        descriptor.usage |= wgpu::TextureUsage::TransientAttachment;
      }
      descriptor.format = wgpu::TextureFormat::R8Unorm;
      descriptor.size = {uint32_t(width_), uint32_t(height_), 1};
      raster_target_ = webgpu_active_device().CreateTexture(&descriptor);
      viewport_reset();
      scissor_reset();
    }
    if (!raster_target_) {
      return false;
    }
    targets_.colors[0] = raster_target_.GetFormat();
    views_[GPU_FB_COLOR_ATTACHMENT0] = raster_target_.CreateView();
  }
  else {
    raster_target_ = nullptr;
  }
  if (size.x != width_ || size.y != height_) {
    size_set(size.x, size.y);
    viewport_reset();
    scissor_reset();
  }
  valid_ = size.x != 0 && size.y != 0;
  return valid_;
}

void WebGPUFramebuffer::bind(const bool enabled_srgb)
{
  auto &context = WebGPUContext::active();
  /* A linear view lets Blender write display-encoded UI colors verbatim. The
   * texture's ordinary sampled view still decodes those bytes for compositing. */
  const bool change_srgb_view = has_srgb_color_ && enabled_srgb != srgb_enabled_;
  if (context.active_fb != this || dirty_attachments_ || change_srgb_view || subpass_colors_ ||
      use_explicit_load_store_ ||
      std::any_of(loadstore_actions_.begin(),
                  loadstore_actions_.end(),
                  [](const auto &action) { return action.has_value(); }))
  {
    context.finish_framebuffer();
  }
  context.active_fb = this;
  srgb_enabled_ = enabled_srgb;
  dirty_attachments_ |= change_srgb_view;
  update_attachments();
  if (subpass_colors_) {
    for (const int slot : IndexRange(GPU_FB_MAX_COLOR_ATTACHMENT)) {
      const auto type = GPU_FB_COLOR_ATTACHMENT0 + slot;
      set_color_attachment_bit(type, attachments_[type].tex != nullptr);
    }
    subpass_colors_.reset();
    subpass_inputs_ = {};
    subpass_depth_ = true;
  }
  const auto format = targets_.colors[0];
  context.shader_builtin_srgb_transform = enabled_srgb &&
                                          (format == wgpu::TextureFormat::RGBA8UnormSrgb ||
                                           format == wgpu::TextureFormat::BGRA8UnormSrgb);
  context.shader_builtin_srgb_is_dirty = true;
}

bool WebGPUFramebuffer::check(char error[256])
{
  const bool valid = update_attachments();
  if (!valid && error) {
    std::snprintf(
        error, 256, "Invalid WebGPU framebuffer attachments or missing canvas drawable.");
  }
  return valid;
}

WebGPURenderTargets WebGPUFramebuffer::targets()
{
  update_attachments();
  auto targets = targets_;
  if (subpass_colors_) {
    for (const int slot : IndexRange(GPU_FB_MAX_COLOR_ATTACHMENT)) {
      if (!(*subpass_colors_ & (1u << slot))) {
        targets.colors[slot] = wgpu::TextureFormat::Undefined;
      }
    }
    if (!subpass_depth_) {
      targets.depth = wgpu::TextureFormat::Undefined;
      targets.stencil = false;
    }
  }
  return targets;
}

void WebGPUFramebuffer::subpass_transition_impl(const GPUAttachmentState depth_state,
                                                Span<GPUAttachmentState> color_states)
{
  auto &context = WebGPUContext::active();
  if (context.active_fb != this || is_canvas() || !update_attachments()) {
    unsupported("subpass transitions without an active texture framebuffer");
  }
  context.end_render_pass();
  if (pending_attachment_loads_) {
    begin_pass(context.encoder()).End();
  }

  /* A sampled attachment and a writable attachment cannot overlap within a WebGPU pass. Split
   * on each transition and bind a single-layer read view of the exact previous attachment. */
  subpass_colors_ = 0;
  subpass_depth_ = depth_state == GPU_ATTACHMENT_WRITE;
  subpass_inputs_ = {};
  for (const int slot : color_states.index_range()) {
    if (color_states[slot] == GPU_ATTACHMENT_WRITE) {
      *subpass_colors_ |= 1u << slot;
    }
    else if (color_states[slot] == GPU_ATTACHMENT_READ) {
      const auto &attachment = attachments_[GPU_FB_COLOR_ATTACHMENT0 + slot];
      const auto &texture = static_cast<const WebGPUTexture &>(*attachment.tex);
      subpass_inputs_[slot] = texture.attachment_view(
          attachment.mip, attachment.layer, wgpu::TextureUsage::TextureBinding);
    }
  }
}

int4 WebGPUFramebuffer::scissor_rect() const
{
  if (!scissor_test_) {
    return {0, 0, width_, height_};
  }
  const int left = std::clamp(scissor_[0], 0, width_);
  const int right = std::clamp<int64_t>(int64_t(scissor_[0]) + scissor_[2], left, width_);
  const int bottom = std::clamp(scissor_[1], 0, height_);
  const int top = std::clamp<int64_t>(int64_t(scissor_[1]) + scissor_[3], bottom, height_);
  return {left, is_canvas() ? height_ - top : bottom, right - left, top - bottom};
}

void WebGPUFramebuffer::attachment_set_loadstore_op(const GPUAttachmentType type,
                                                    const GPULoadStore action)
{
  loadstore_actions_[type] = action;
  pending_attachment_loads_ |= action.load_action != GPU_LOADACTION_LOAD;
}

void WebGPUFramebuffer::finish_loadstore(WebGPUContext &context)
{
  /* Store across internal pass breaks. Discard only when the Blender framebuffer
   * scope ends, so copies, compute and uploads can interrupt a pass safely. */
  const bool needs_final_pass =
      pending_attachment_loads_ ||
      std::any_of(loadstore_actions_.begin(), loadstore_actions_.end(), [](const auto &action) {
        return action && action->store_action == GPU_STOREACTION_DONT_CARE;
      });
  if (valid_ && needs_final_pass) {
    begin_pass(context.encoder(), {}, {}, {}, true).End();
  }
  pending_attachment_loads_ = false;
  loadstore_actions_ = {};
}

wgpu::RenderPassEncoder WebGPUFramebuffer::begin_pass(
    const wgpu::CommandEncoder &encoder,
    const Span<std::optional<double4>> clear_colors,
    const std::optional<float> clear_depth,
    const std::optional<uint32_t> clear_stencil,
    const bool finish_loadstore,
    const wgpu::QuerySet &occlusion_queries)
{
  /* At scope exit use the cached views from the scope being closed. A
   * framebuffer reconfiguration may already have changed attachments_. */
  if (!finish_loadstore && !update_attachments()) {
    return nullptr;
  }
  auto clear_value = [&](const GPUAttachmentType type) -> std::optional<double4> {
    const auto &action = loadstore_actions_[type];
    if (!pending_attachment_loads_ || !action || action->load_action == GPU_LOADACTION_LOAD) {
      return {};
    }
    if (action->load_action == GPU_LOADACTION_DONT_CARE) {
      return double4(0.0);
    }
    const double *value = action->clear_value;
    return double4(value[0], value[1], value[2], value[3]);
  };
  auto store_action = [&](const GPUAttachmentType type) {
    const auto &action = loadstore_actions_[type];
    return finish_loadstore && action && action->store_action == GPU_STOREACTION_DONT_CARE ?
               wgpu::StoreOp::Discard :
               wgpu::StoreOp::Store;
  };
  std::array<wgpu::RenderPassColorAttachment, GPU_FB_MAX_COLOR_ATTACHMENT> colors{};
  uint32_t color_count = 0;
  for (const int i : IndexRange(GPU_FB_MAX_COLOR_ATTACHMENT)) {
    if (targets_.colors[i] == wgpu::TextureFormat::Undefined ||
        (subpass_colors_ && !(*subpass_colors_ & (1u << i))))
    {
      continue;
    }
    const auto type = GPU_FB_COLOR_ATTACHMENT0 + i;
    auto &color = colors[i];
    color.view = is_canvas() ? window_context_->ghost.drawable().CreateView() : views_[type];
    if (raster_target_) {
      color.loadOp = wgpu::LoadOp::Clear;
      color.storeOp = wgpu::StoreOp::Discard;
      color_count = i + 1;
      continue;
    }
    color.loadOp = wgpu::LoadOp::Load;
    color.storeOp = store_action(type);
    const auto value = i < clear_colors.size() && clear_colors[i] ? clear_colors[i] :
                                                                    clear_value(type);
    if (value) {
      color.loadOp = wgpu::LoadOp::Clear;
      color.clearValue = {value->x, value->y, value->z, value->w};
    }
    color_count = i + 1;
  }
  wgpu::RenderPassDescriptor desc{};
  desc.label = name_get();
  desc.occlusionQuerySet = occlusion_queries;
  desc.colorAttachmentCount = color_count;
  desc.colorAttachments = colors.data();
  wgpu::RenderPassDepthStencilAttachment depth{};
  if (targets_.depth != wgpu::TextureFormat::Undefined && subpass_depth_) {
    const auto type = targets_.stencil ? GPU_FB_DEPTH_STENCIL_ATTACHMENT : GPU_FB_DEPTH_ATTACHMENT;
    const auto value = clear_value(type);
    depth.view = views_[type];
    depth.depthLoadOp = clear_depth || value ? wgpu::LoadOp::Clear : wgpu::LoadOp::Load;
    depth.depthStoreOp = store_action(type);
    /* EmdawnWebGPU forwards the numeric field even with loadOp set to Load. */
    depth.depthClearValue = clear_depth.value_or(value ? float(value->x) : 0.0f);
    if (targets_.stencil) {
      depth.stencilLoadOp = clear_stencil || value ? wgpu::LoadOp::Clear : wgpu::LoadOp::Load;
      depth.stencilStoreOp = store_action(type);
      depth.stencilClearValue = clear_stencil.value_or(value ? uint32_t(value->y) : 0);
    }
    desc.depthStencilAttachment = &depth;
  }
  pending_attachment_loads_ = false;
  return encoder.BeginRenderPass(&desc);
}

void WebGPUFramebuffer::clear_attachments(const Span<std::optional<double4>> colors,
                                          const std::optional<float> depth,
                                          const std::optional<uint32_t> stencil)
{
  if (!update_attachments()) {
    return;
  }
  const int4 scissor = scissor_rect();
  if (scissor.z == 0 || scissor.w == 0) {
    return;
  }
  auto &context = WebGPUContext::active();
  if (scissor.z == width_ && scissor.w == height_) {
    context.clear_framebuffer(*this, colors, depth, stencil);
    return;
  }

  context.end_render_pass();
  ClearPipelineKey key{targets_,
                       0,
                       depth.has_value() && targets_.depth != wgpu::TextureFormat::Undefined,
                       stencil.has_value() && targets_.stencil};
  std::array<std::array<uint32_t, 4>, GPU_FB_MAX_COLOR_ATTACHMENT + 1> values{};
  for (const int slot : IndexRange(std::min<int>(colors.size(), targets_.colors.size()))) {
    const auto format = targets_.colors[slot];
    if (!colors[slot] || format == wgpu::TextureFormat::Undefined) {
      continue;
    }
    key.colors |= 1u << slot;
    for (const int lane : IndexRange(4)) {
      const double value = (*colors[slot])[lane];
      values[slot][lane] = unsigned_color_format(format) ?
                               uint32_t(std::clamp(value, 0.0, double(UINT32_MAX))) :
                               std::bit_cast<uint32_t>(float(value));
    }
  }
  if (key.colors == 0 && !key.depth && !key.stencil) {
    return;
  }
  values.back()[0] = std::bit_cast<uint32_t>(float(depth.value_or(0.0f)));
  const auto &pipeline = context.clear_pipeline(key);
  const auto pass = begin_pass(context.encoder());
  pass.SetPipeline(pipeline);
  if (key.colors != 0 || key.depth) {
    const auto uniform = context.upload_draw_uniforms(std::as_bytes(std::span(values)));
    wgpu::BindGroupEntry binding{};
    binding.buffer = uniform.buffer;
    binding.offset = uniform.offset;
    binding.size = sizeof(values);
    wgpu::BindGroupDescriptor group{};
    group.layout = pipeline.GetBindGroupLayout(0);
    group.entryCount = 1;
    group.entries = &binding;
    pass.SetBindGroup(0, context.ghost.device().CreateBindGroup(&group));
  }
  if (key.stencil) {
    pass.SetStencilReference(*stencil & 0xff);
  }
  pass.SetScissorRect(scissor.x, scissor.y, scissor.z, scissor.w);
  pass.Draw(3, 1, 0, 0);
  pass.End();
}

void WebGPUFramebuffer::clear(const GPUFrameBufferBits buffers,
                              const double4 color,
                              const float depth,
                              const uint stencil)
{
  std::array<std::optional<double4>, GPU_FB_MAX_COLOR_ATTACHMENT> colors{};
  if (buffers & GPU_COLOR_BIT) {
    colors.fill(color);
  }
  clear_attachments(colors,
                    (buffers & GPU_DEPTH_BIT) ? std::optional(depth) : std::nullopt,
                    (buffers & GPU_STENCIL_BIT) ? std::optional(stencil) : std::nullopt);
}

void WebGPUFramebuffer::clear_multi(const Span<double4> values)
{
  std::array<std::optional<double4>, GPU_FB_MAX_COLOR_ATTACHMENT> colors{};
  for (const int i : IndexRange(std::min<int>(values.size(), colors.size()))) {
    colors[i] = values[i];
  }
  clear_attachments(colors);
}

void WebGPUFramebuffer::clear_attachment(const GPUAttachmentType type, const double4 value)
{
  std::array<std::optional<double4>, GPU_FB_MAX_COLOR_ATTACHMENT> colors{};
  if (type < GPU_FB_COLOR_ATTACHMENT0) {
    clear_attachments(colors,
                      float(value.x),
                      type == GPU_FB_DEPTH_STENCIL_ATTACHMENT ? std::optional(uint32_t(value.y)) :
                                                                std::nullopt);
    return;
  }
  colors[type - GPU_FB_COLOR_ATTACHMENT0] = value;
  clear_attachments(colors);
}

template<typename T> static T load_texture_component(const std::byte *pixel)
{
  T value;
  std::memcpy(&value, pixel, sizeof(value));
  return value;
}

static void unpack_texture_row(const std::byte *source,
                                   std::byte *destination,
                                   const uint32_t width,
                                   const TextureFormat texture_format,
                                   const TextureFormat transfer_format,
                                   const eGPUDataFormat data_format,
                                   const int channels)
{
  const int components = to_component_len(texture_format);
  const int transfer_components = to_component_len(transfer_format);
  const size_t pixel_size = to_bytesize(transfer_format);
  if (data_format == GPU_DATA_2_10_10_10_REV) {
    std::memcpy(destination, source, width * pixel_size);
    return;
  }

  const auto flags = to_format_flag(transfer_format);
  const bool integer = (flags & GPU_FORMAT_INTEGER) != 0;
  const bool normalized = (flags & GPU_FORMAT_NORMALIZED_INTEGER) != 0;
  const size_t component_size = pixel_size / transfer_components;
  const size_t output_size = to_bytesize(data_format);
  if (texture_format == transfer_format && texture_format != TextureFormat::UNORM_10_10_10_2 &&
      components == channels &&
      component_size == output_size)
  {
    std::memcpy(destination, source, width * pixel_size);
    return;
  }

  for (uint32_t x = 0; x < width; x++, source += pixel_size) {
    for (int channel = 0; channel < channels; channel++, destination += output_size) {
      const bool present = channel < components;
      const std::byte *component = source + (present ? channel * component_size : 0);
      if (integer) {
        uint32_t value = channel == 3;
        if (present) {
          switch (component_size) {
            case 1:
              value = load_texture_component<uint8_t>(component);
              break;
            case 2:
              value = load_texture_component<uint16_t>(component);
              break;
            default:
              value = load_texture_component<uint32_t>(component);
              break;
          }
        }
        if (ELEM(data_format, GPU_DATA_UINT, GPU_DATA_INT)) {
          std::memcpy(destination, &value, sizeof(value));
        }
        else {
          *destination = std::byte(value);
        }
        continue;
      }

      float value = float(channel == 3);
      if (present) {
        if (texture_format == TextureFormat::UNORM_10_10_10_2) {
          const uint32_t pixel = load_texture_component<uint32_t>(source);
          const uint32_t mask = channel == 3 ? 3 : 1023;
          value = float((pixel >> (10 * channel)) & mask) / float(mask);
        }
        else if (normalized) {
          value = load_texture_component<uint8_t>(component) * (1.0f / 255.0f);
        }
        else if (component_size == sizeof(uint16_t)) {
          value = math::half_to_float(load_texture_component<uint16_t>(component));
        }
        else {
          value = load_texture_component<float>(component);
        }
      }

      if (data_format == GPU_DATA_FLOAT) {
        std::memcpy(destination, &value, sizeof(value));
      }
      else if (data_format == GPU_DATA_HALF_FLOAT) {
        const uint16_t half = math::float_to_half(value);
        std::memcpy(destination, &half, sizeof(half));
      }
      else {
        const float clamped = std::clamp(std::isnan(value) ? 0.0f : value, 0.0f, 1.0f);
        *destination = std::byte(uint8_t(std::lround(clamped * 255.0f)));
      }
    }
  }
}

static void read_color_texture(WebGPUContext &context,
                               const WebGPUTexture &texture,
                               const uint32_t mip,
                               const wgpu::Origin3D origin,
                               const wgpu::Extent3D extent,
                               const eGPUDataFormat format,
                               const int channels,
                               void *data)
{
  const auto texture_format = texture.format_get();
  const auto transfer_format = texture.transfer_format();
  const auto flags = to_format_flag(texture_format);
  const bool integer = (flags & GPU_FORMAT_INTEGER) != 0;
  const bool packed = texture_format == TextureFormat::UNORM_10_10_10_2;
  const uint32_t pixel_size = to_bytesize(transfer_format);
  const int components = to_component_len(texture_format);
  const bool integer_format = flags & GPU_FORMAT_SIGNED ?
                                  format == GPU_DATA_INT && pixel_size == 4 * components :
                                  format == GPU_DATA_UINT ||
                                      (format == GPU_DATA_UBYTE && pixel_size == components);
  const bool color_format = ELEM(format, GPU_DATA_FLOAT, GPU_DATA_HALF_FLOAT, GPU_DATA_UBYTE) ||
                            (packed && format == GPU_DATA_2_10_10_10_REV && channels == 4);
  if (integer ? !integer_format : !color_format) {
    unsupported("converting this color texture readback format");
  }

  const uint32_t packed_row_bytes = extent.width * pixel_size;
  /* WebGPU texture-to-buffer rows require 256-byte alignment. */
  const uint32_t staging_stride = (packed_row_bytes + 255) & ~255u;
  const size_t rows = size_t(extent.height) * extent.depthOrArrayLayers;
  const uint64_t staging_size = (rows - 1) * staging_stride + ((packed_row_bytes + 3) & ~3u);
  const size_t output_stride = format == GPU_DATA_2_10_10_10_REV ?
                                   packed_row_bytes :
                                   size_t(extent.width) * channels * to_bytesize(format);

  const auto &device = context.ghost.device();
  wgpu::BufferDescriptor staging_desc{};
  staging_desc.label = "Blender color texture readback";
  staging_desc.size = staging_size;
  staging_desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
  const auto staging = device.CreateBuffer(&staging_desc);

  wgpu::TexelCopyTextureInfo source{};
  source.texture = texture.handle();
  source.mipLevel = mip;
  source.origin = origin;
  wgpu::TexelCopyBufferInfo destination{};
  destination.buffer = staging;
  destination.layout.bytesPerRow = staging_stride;
  destination.layout.rowsPerImage = extent.height;
  context.encoder().CopyTextureToBuffer(&source, &destination, &extent);
  context.flush();

  const auto completion = emscripten_promise_create();
  staging.MapAsync(
      wgpu::MapMode::Read,
      0,
      staging_size,
      wgpu::CallbackMode::AllowSpontaneous,
      [=](wgpu::MapAsyncStatus status, wgpu::StringView) {
        auto *output = static_cast<std::byte *>(data);
        const auto *input = status == wgpu::MapAsyncStatus::Success ?
                                static_cast<const std::byte *>(
                                    staging.GetConstMappedRange(0, staging_size)) :
                                nullptr;
        if (input) {
          for (size_t row = 0; row < rows; row++) {
            unpack_texture_row(input + row * staging_stride,
                               output + row * output_stride,
                               extent.width,
                               texture_format,
                               transfer_format,
                               format,
                               channels);
          }
        }
        else {
          std::fputs("WebGPU color texture readback mapping failed.\n", stderr);
          std::memset(output, 0, rows * output_stride);
        }
        if (status == wgpu::MapAsyncStatus::Success) {
          staging.Unmap();
        }
        emscripten_promise_resolve(completion, EM_PROMISE_FULFILL, nullptr);
      });
  (void)emscripten_promise_await_unchecked(completion);
  emscripten_promise_destroy(completion);
}

void WebGPUTexture::read(const int mip, const eGPUDataFormat format, void *data)
{
  if (!handle() || (format_flag_ & (GPU_FORMAT_DEPTH | GPU_FORMAT_STENCIL))) {
    unsupported("CPU reads from depth, stencil, or buffer-backed textures");
  }
  auto &context = WebGPUContext::active();
  context.end_render_pass();

  const int3 size = mip_size_get(mip);
  const wgpu::Extent3D extent = type_ == GPU_TEXTURE_1D_ARRAY ?
                                   wgpu::Extent3D{uint32_t(size.x), 1, uint32_t(size.y)} :
                                   wgpu::Extent3D{
                                       uint32_t(size.x), uint32_t(size.y), uint32_t(size.z)};
  read_color_texture(context,
                     *this,
                     backing_mip(mip),
                     {0, 0, backing_layer(0)},
                     extent,
                     format,
                     to_component_len(format_),
                     data);
}

void WebGPUFramebuffer::read(const GPUFrameBufferBits planes,
                             const eGPUDataFormat format,
                             const int rect[4],
                             const int channels,
                             const int slot,
                             void *data)
{
  if (planes != GPU_COLOR_BIT || is_canvas()) {
    unsupported("framebuffer reads from non-color or canvas attachments");
  }
  if (!update_attachments() || slot < 0 || slot >= GPU_FB_MAX_COLOR_ATTACHMENT) {
    unsupported("reading an invalid framebuffer attachment");
  }
  if (rect[2] == 0 || rect[3] == 0) {
    return;
  }
  if (rect[0] < 0 || rect[1] < 0 || rect[2] < 0 || rect[3] < 0 ||
      rect[0] > width_ - rect[2] || rect[1] > height_ - rect[3] || !data ||
      channels < 1 || channels > 4)
  {
    unsupported("reading outside framebuffer attachment bounds");
  }
  const GPUAttachment &attachment = attachments_[GPU_FB_COLOR_ATTACHMENT0 + slot];
  if (!attachment.tex) {
    unsupported("reading an unattached framebuffer slot");
  }

  auto &texture = static_cast<WebGPUTexture &>(*attachment.tex);
  auto &context = WebGPUContext::active();
  context.end_render_pass();
  if (pending_attachment_loads_) {
    begin_pass(context.encoder()).End();
  }
  /* Offscreen passes already map Blender's increasing Y to increasing texture rows. */
  read_color_texture(context,
                     texture,
                     texture.backing_mip(attachment.mip),
                     {uint32_t(rect[0]),
                      uint32_t(rect[1]),
                      texture.backing_layer(std::max(0, attachment.layer))},
                     {uint32_t(rect[2]), uint32_t(rect[3]), 1},
                     format,
                     channels,
                     data);
}

void WebGPUFramebuffer::blit_to(const GPUFrameBufferBits planes,
                                const int src_slot,
                                FrameBuffer *destination,
                                const int dst_slot,
                                const int dst_x,
                                const int dst_y)
{
  if (planes != GPU_COLOR_BIT || is_canvas()) {
    unsupported("depth/stencil blits or blits from the WebGPU canvas");
  }
  auto &target = static_cast<WebGPUFramebuffer &>(*destination);
  auto &context = WebGPUContext::active();
  context.end_render_pass();
  if (!update_attachments() || !target.update_attachments()) {
    return;
  }
  const GPUAttachment &source = attachments_[GPU_FB_COLOR_ATTACHMENT0 + src_slot];
  if (!source.tex || (!target.is_canvas() && !target.color_tex(dst_slot))) {
    return;
  }
  auto &texture = static_cast<WebGPUTexture &>(*source.tex);
  if (!(texture.usage_get() & GPU_TEXTURE_USAGE_SHADER_READ) ||
      (!target.is_canvas() && source.tex == target.color_tex(dst_slot)))
  {
    unsupported("color blits from unsampled or overlapping attachments");
  }
  const int width = texture.mip_width_get(source.mip);
  const int height = texture.type_get() == GPU_TEXTURE_1D_ARRAY ?
                         1 :
                         texture.mip_height_get(source.mip);
  auto mask = wgpu::ColorWriteMask(context.state_manager->state.write_mask & GPU_WRITE_COLOR);
  const auto &pipeline = context.blit_pipeline(
      target.targets_.colors[dst_slot], mask, target.is_canvas());
  wgpu::BindGroupEntry input{};
  input.textureView = texture.attachment_view(
      source.mip, source.layer, wgpu::TextureUsage::TextureBinding);
  wgpu::BindGroupDescriptor group{};
  group.layout = pipeline.GetBindGroupLayout(0);
  group.entryCount = 1;
  group.entries = &input;
  const auto bind_group = context.ghost.device().CreateBindGroup(&group);

  wgpu::RenderPassColorAttachment color{};
  color.view = target.is_canvas() ? target.window_context_->ghost.drawable().CreateView() :
                                    target.views_[GPU_FB_COLOR_ATTACHMENT0 + dst_slot];
  color.loadOp = wgpu::LoadOp::Load;
  color.storeOp = wgpu::StoreOp::Store;
  wgpu::RenderPassDescriptor desc{};
  desc.colorAttachmentCount = 1;
  desc.colorAttachments = &color;
  const auto pass = context.encoder().BeginRenderPass(&desc);
  pass.SetViewport(dst_x,
                   target.is_canvas() ? target.height_ - dst_y - height : dst_y,
                   width,
                   height,
                   0.0f,
                   1.0f);
  pass.SetPipeline(pipeline);
  pass.SetBindGroup(0, bind_group);
  pass.Draw(3, 1, 0, 0);
  pass.End();
}

void WebGPUStateManager::issue_barrier(const GPUBarrier bits)
{
  if (!images_.empty() && (bits & (GPU_BARRIER_SHADER_IMAGE_ACCESS | GPU_BARRIER_TEXTURE_FETCH |
                                   GPU_BARRIER_FRAMEBUFFER)))
  {
    /* Render passes use one resource-usage scope. End the pass before subsequent commands
     * read an image through a sampled binding or attachment. */
    WebGPUContext::active().end_render_pass();
  }
}

const wgpu::Device &webgpu_active_device()
{
  return WebGPUContext::active().ghost.device();
}

void webgpu_flush()
{
  WebGPUContext::active().flush();
}

void webgpu_generate_mipmaps(const wgpu::Texture &texture)
{
  WebGPUContext::active().generate_mipmaps(texture);
}

void webgpu_upload_depth(const wgpu::Texture &texture,
                         const uint32_t mip,
                         const wgpu::Origin3D origin,
                         const wgpu::Extent3D size,
                         const uint32_t row_length,
                         const float *data)
{
  WebGPUContext::active().upload_depth(texture, mip, origin, size, row_length, data);
}

WebGPUBufferTransfers &webgpu_geometry_transfers()
{
  WebGPUContext &context = WebGPUContext::active();
  context.flush();
  return context.transfers;
}

void webgpu_bind_geometry_as_storage(const unsigned int binding, const wgpu::Buffer &buffer)
{
  WebGPUContext::active().storage[binding] = buffer;
}

wgpu::Buffer webgpu_bound_storage(const int binding)
{
  const auto &storage = WebGPUContext::active().storage;
  const auto found = storage.find(binding);
  return found != storage.end() ? found->second : wgpu::Buffer{};
}

wgpu::Buffer webgpu_bound_uniform(const int binding)
{
  const auto &uniforms = WebGPUContext::active().uniforms;
  const auto found = uniforms.find(binding);
  return found != uniforms.end() ? found->second : wgpu::Buffer{};
}

WebGPUSampledTexture webgpu_bound_texture(const int binding)
{
  return static_cast<WebGPUStateManager &>(*WebGPUContext::active().state_manager)
      .bound_texture(binding);
}

uint64_t webgpu_emulated_sampler_mask()
{
  return static_cast<WebGPUStateManager &>(*WebGPUContext::active().state_manager)
      .emulated_sampler_mask();
}

uint8_t webgpu_emulated_sampler_parameters(const int binding)
{
  return static_cast<WebGPUStateManager &>(*WebGPUContext::active().state_manager)
      .emulated_sampler_parameters(binding);
}

wgpu::TextureView webgpu_bound_image(const int binding)
{
  return static_cast<WebGPUStateManager &>(*WebGPUContext::active().state_manager)
      .bound_image(binding);
}

wgpu::Buffer webgpu_bound_atomic_texture(const int binding, const bool image)
{
  return static_cast<WebGPUStateManager &>(*WebGPUContext::active().state_manager)
      .bound_atomic_texture(binding, image);
}

wgpu::TextureView webgpu_subpass_input(const int attachment)
{
  return static_cast<WebGPUFramebuffer &>(*WebGPUContext::active().active_fb)
      .subpass_input(attachment);
}

bool webgpu_supports_storage_texture_reads()
{
  /* This is a browser WGSL language feature. Device feature requests are separate. */
  static const bool supported = wgpu::CreateInstance().HasWGSLLanguageFeature(
      wgpu::WGSLLanguageFeatureName::ReadonlyAndReadwriteStorageTextures);
  return supported;
}

void webgpu_bind_texel_buffer(const int slot, const WebGPUTexelBuffer &texels)
{
  static_cast<WebGPUStateManager &>(*WebGPUContext::active().state_manager)
      .texel_bind(slot, texels);
}

WebGPUTexelBuffer webgpu_bound_texel_buffer(const int slot)
{
  return static_cast<WebGPUStateManager &>(*WebGPUContext::active().state_manager)
      .bound_texels(slot);
}

WebGPURenderTargets webgpu_render_targets()
{
  auto &context = WebGPUContext::active();
  auto &framebuffer = static_cast<WebGPUFramebuffer &>(*context.active_fb);
  if (framebuffer.needs_update()) {
    context.end_render_pass();
  }
  return framebuffer.targets();
}

wgpu::RenderPassEncoder webgpu_render_pass()
{
  return WebGPUContext::active().render_pass();
}

void webgpu_render_set_pipeline(const wgpu::RenderPipeline &pipeline)
{
  WebGPUContext::active().set_render_pipeline(pipeline);
}

void webgpu_render_set_bind_group(const uint32_t index,
                                  const wgpu::BindGroup &group,
                                  const std::span<const uint32_t> offsets)
{
  WebGPUContext::active().set_render_bind_group(index, group, offsets);
}

void webgpu_render_set_stencil_reference(const uint32_t reference)
{
  WebGPUContext::active().set_render_stencil_reference(reference);
}

const wgpu::Buffer &webgpu_default_vertex_buffer()
{
  return WebGPUContext::active().default_vertex_buffer();
}

const wgpu::Buffer &webgpu_triangle_fan_indices(const uint32_t index_count)
{
  return WebGPUContext::active().triangle_fan_indices(index_count);
}

wgpu::Buffer webgpu_gather_indexed_points(const WebGPUIndexedPoints &points,
                                          const wgpu::Buffer &source,
                                          const size_t offset,
                                          const uint32_t stride)
{
  if ((offset | stride) % 4 != 0 || offset / 4 > UINT32_MAX) {
    webgpu_draw_unsupported("an unaligned indexed point vertex stream");
  }
  return WebGPUContext::active().gather_indexed_points(points, source, offset, stride);
}

WebGPUPointCommands webgpu_expand_point_commands(const wgpu::Buffer &input,
                                                 const uint32_t first_word,
                                                 const uint32_t stride_words,
                                                 const uint32_t count,
                                                 const uint32_t point_capacity,
                                                 const bool indexed)
{
  return WebGPUContext::active().expand_point_commands(
      input, first_word, stride_words, count, point_capacity, indexed);
}

WebGPUUniformSlice webgpu_upload_draw_uniforms(std::span<const std::byte> bytes)
{
  return WebGPUContext::active().upload_draw_uniforms(bytes);
}

WebGPUUniformSlice webgpu_upload_immediate_vertices(std::span<const std::byte> bytes)
{
  return WebGPUContext::active().upload_immediate_vertices(bytes);
}

static void unbind_buffer(BufferBindings &bindings, const WebGPUBuffer &buffer)
{
  std::erase_if(bindings, [&buffer](const auto &item) {
    return item.second.Get() == buffer.handle().Get();
  });
}

class WebGPUStorageBuffer final : public StorageBuf {
 public:
  WebGPUStorageBuffer(size_t size, const char *name) : StorageBuf(size, name) {}

  void update(const void *data) override
  {
    auto &buffer = ensure_buffer();
    WebGPUContext::active().update_buffer(buffer, data, usage_size_in_bytes_);
  }
  void bind(int slot) override
  {
    WebGPUContext::active().storage[slot] = ensure_buffer().handle();
  }
  void unbind() override
  {
    if (buffer_) {
      unbind_buffer(WebGPUContext::active().storage, *buffer_);
    }
  }
  void clear(uint32_t value) override
  {
    auto &context = WebGPUContext::active();
    auto &buffer = ensure_buffer();
    if (value == 0) {
      context.clear_buffer(buffer);
    }
    else {
      context.flush();
      buffer.clear(value);
    }
  }
  void copy_sub(VertBuf *src, uint dst_offset, uint src_offset, uint size) override
  {
    auto &source = static_cast<WebGPUVertexBuffer &>(*src).buffer_for_use();
    auto &destination = ensure_buffer();
    webgpu_geometry_transfers().copy(source, src_offset, destination, dst_offset, size);
  }
  void read(void *data) override
  {
    async_flush_to_host();
    if (!readback_->ready) {
      /* MapAsync completes on the browser event loop. Suspend the current
       * Wasm frame while leaving the pending read and caller stack in place. */
      const em_promise_t completion = emscripten_promise_create();
      readback_->destination = data;
      readback_->completion = completion;
      (void)emscripten_promise_await_unchecked(completion);
      emscripten_promise_destroy(completion);
    }
    if (!readback_->success) {
      std::fputs("WebGPU storage-buffer mapping failed.\n", stderr);
      std::abort();
    }
    if (!readback_->data.empty()) {
      std::memcpy(data, readback_->data.data(), readback_->data.size());
    }
    readback_.reset();
  }
  void async_flush_to_host() override
  {
    if (readback_) {
      return;
    }
    WebGPUContext::active().flush();
    readback_ = std::make_shared<Readback>();
    ensure_buffer().read_async([readback = readback_](std::span<const std::byte> data) {
      readback->success = data.data() != nullptr;
      if (readback->success) {
        if (readback->destination) {
          if (!data.empty()) {
            std::memcpy(readback->destination, data.data(), data.size());
          }
        }
        else {
          readback->data.assign(data.begin(), data.end());
        }
      }
      readback->ready = true;
      if (readback->completion) {
        emscripten_promise_resolve(readback->completion, EM_PROMISE_FULFILL, nullptr);
      }
    });
  }
  void sync_as_indirect_buffer() override
  {
    /* Storage and indirect uses are declared when the buffer is created. */
    ensure_buffer();
  }

  const WebGPUBuffer &indirect_buffer()
  {
    return ensure_buffer();
  }

 private:
  struct Readback {
    std::vector<std::byte> data;
    void *destination = nullptr;
    em_promise_t completion = nullptr;
    bool ready = false;
    bool success = false;
  };

  std::optional<WebGPUBuffer> buffer_;
  std::shared_ptr<Readback> readback_;

  WebGPUBuffer &ensure_buffer()
  {
    if (!buffer_) {
      buffer_.emplace(WebGPUContext::active().ghost.device(),
                      size_in_bytes_,
                      wgpu::BufferUsage::Storage | wgpu::BufferUsage::Indirect,
                      name_);
    }
    return *buffer_;
  }
};

const WebGPUBuffer &webgpu_indirect_buffer(StorageBuf &buffer)
{
  return static_cast<WebGPUStorageBuffer &>(buffer).indirect_buffer();
}

class WebGPUUniformBuffer final : public UniformBuf {
 public:
  WebGPUUniformBuffer(size_t size, const char *name) : UniformBuf(size, name) {}

  void update(const void *data) override
  {
    auto &buffer = ensure_buffer();
    WebGPUContext::active().update_buffer(buffer, data, buffer.size());
    MEM_SAFE_DELETE_VOID(data_);
  }
  void clear_to_zero() override
  {
    auto &buffer = ensure_buffer();
    WebGPUContext::active().clear_buffer(buffer);
    MEM_SAFE_DELETE_VOID(data_);
  }
  void bind(int slot) override
  {
    if (data_) {
      update(data_);
    }
    WebGPUContext::active().uniforms[slot] = ensure_buffer().handle();
  }
  void bind_as_ssbo(int slot) override
  {
    if (data_) {
      update(data_);
    }
    WebGPUContext::active().storage[slot] = ensure_buffer().handle();
  }
  void unbind() override
  {
    if (buffer_) {
      WebGPUContext &context = WebGPUContext::active();
      unbind_buffer(context.uniforms, *buffer_);
      unbind_buffer(context.storage, *buffer_);
    }
  }

 private:
  std::optional<WebGPUBuffer> buffer_;

  WebGPUBuffer &ensure_buffer()
  {
    if (!buffer_) {
      buffer_.emplace(WebGPUContext::active().ghost.device(),
                      size_in_bytes_,
                      wgpu::BufferUsage::Uniform | wgpu::BufferUsage::Storage,
                      name_);
    }
    return *buffer_;
  }
};

WebGPUBackend::WebGPUBackend()
{
  compiler_ = nullptr;
}

WebGPUBackend::~WebGPUBackend()
{
  delete_resources();
  GPG.clear();
  GCaps = {};
}

bool WebGPUBackend::is_supported()
{
  const auto *system = dynamic_cast<const GHOST_SystemWeb *>(GHOST_ISystem::getSystem());
  return system && system->isWebGPUReady();
}

Context *WebGPUBackend::context_alloc(GHOST_IWindow *window, GHOST_IContext *context)
{
  auto *web_context = dynamic_cast<GHOST_ContextWebGPU *>(context);
  if (!web_context || !emscripten_is_main_runtime_thread()) {
    unsupported("context creation without a runtime-thread WebGPU device");
  }
  if (!GPG.initialized) {
    GPG.init(GPU_DEVICE_ANY,
             GPU_OS_ANY,
             GPU_DRIVER_ANY,
             GPU_SUPPORT_LEVEL_SUPPORTED,
             GPU_BACKEND_WEBGPU,
             "Browser",
             "WebGPU adapter",
             "WebGPU",
             GPU_ARCHITECTURE_IMR);

    /* EmdawnWebGPU handles belong to the canvas thread. */
    GCaps.use_main_context_workaround = true;
    wgpu::Limits limits;
    if (web_context->device().GetLimits(&limits) == wgpu::Status::Success) {
      GCaps.max_texture_size = limits.maxTextureDimension2D;
      GCaps.max_texture_3d_size = limits.maxTextureDimension3D;
      GCaps.max_texture_layers = limits.maxTextureArrayLayers;
      GCaps.max_textures = limits.maxSampledTexturesPerShaderStage;
      GCaps.max_images = limits.maxStorageTexturesPerShaderStage;
      GCaps.max_vertex_attribs = limits.maxVertexAttributes;
      GCaps.max_shader_storage_buffer_bindings = limits.maxStorageBuffersPerShaderStage;
      GCaps.max_compute_shader_storage_blocks = limits.maxStorageBuffersPerShaderStage;
      GCaps.max_storage_buffer_size = limits.maxStorageBufferBindingSize;
      /* Texel buffers use a storage binding with up to four 32-bit lanes per texel. */
      GCaps.max_buffer_texture_size = uint32_t(
          std::min<uint64_t>(limits.maxStorageBufferBindingSize / 16, UINT32_MAX));
      GCaps.max_uniform_buffer_size = limits.maxUniformBufferBindingSize;
      GCaps.storage_buffer_alignment = limits.minStorageBufferOffsetAlignment;
      GCaps.minimum_per_vertex_stride = 4;
      GCaps.max_work_group_size[0] = limits.maxComputeWorkgroupSizeX;
      GCaps.max_work_group_size[1] = limits.maxComputeWorkgroupSizeY;
      GCaps.max_work_group_size[2] = limits.maxComputeWorkgroupSizeZ;
      for (int &count : GCaps.max_work_group_count) {
        count = limits.maxComputeWorkgroupsPerDimension;
      }
    }
  }
  return new WebGPUContext(window, *web_context);
}

void WebGPUBackend::init_resources()
{
  compiler_ = MEM_new<ShaderCompiler>(__func__);
}
void WebGPUBackend::delete_resources()
{
  MEM_SAFE_DELETE(compiler_);
}
void WebGPUBackend::compute_dispatch(const int x, const int y, const int z)
{
  auto pass = WebGPUContext::active().begin_compute_pass();
  pass.DispatchWorkgroups(x, y, z);
  pass.End();
}
void WebGPUBackend::compute_dispatch_indirect(StorageBuf *indirect)
{
  auto pass = WebGPUContext::active().begin_compute_pass();
  pass.DispatchWorkgroupsIndirect(webgpu_indirect_buffer(*indirect).handle(), 0);
  pass.End();
}
Batch *WebGPUBackend::batch_alloc()
{
  return new WebGPUBatch;
}

static std::shared_ptr<WebGPUQueueCompletion> request_active_queue_completion()
{
  auto &context = WebGPUContext::active();
  context.flush();
  return WebGPUQueueCompletion::request(context.ghost.device().GetQueue());
}

class WebGPUFence final : public Fence {
 public:
  void signal() override
  {
    completion_ = request_active_queue_completion();
  }

  void wait() override
  {
    if (completion_) {
      completion_->wait();
    }
  }

 private:
  std::shared_ptr<WebGPUQueueCompletion> completion_;
};

class WebGPUWorkInFlight final : public WorkInFlight {
 public:
  explicit WebGPUWorkInFlight(const unsigned int max_in_flight)
      : completions_(std::max(1u, max_in_flight))
  {
  }

  void reset() override
  {
    std::fill(completions_.begin(), completions_.end(), nullptr);
    index_ = 0;
  }

  void begin_work() override
  {
    if (auto completion = std::move(completions_[index_])) {
      completion->wait();
    }
  }

  void end_work() override
  {
    completions_[index_] = request_active_queue_completion();
    index_ = (index_ + 1) % completions_.size();
  }

 private:
  std::vector<std::shared_ptr<WebGPUQueueCompletion>> completions_;
  size_t index_ = 0;
};

Fence *WebGPUBackend::fence_alloc()
{
  return new WebGPUFence;
}
WorkInFlight *WebGPUBackend::work_in_flight_alloc(const unsigned int max_in_flight)
{
  return new WebGPUWorkInFlight(max_in_flight);
}
FrameBuffer *WebGPUBackend::framebuffer_alloc(const char *name)
{
  return new WebGPUFramebuffer(name);
}
IndexBuf *WebGPUBackend::indexbuf_alloc()
{
  return new WebGPUIndexBuffer;
}
PixelBuffer *WebGPUBackend::pixelbuf_alloc(size_t)
{
  unsupported("pixel buffers");
}
QueryPool *WebGPUBackend::querypool_alloc()
{
  return new WebGPUQueryPool;
}
Shader *WebGPUBackend::shader_alloc(const char *name)
{
  return new WebGPUShader(name);
}

Texture *WebGPUBackend::texture_alloc(const char *name)
{
  return new WebGPUTexture(name);
}
TexturePool *WebGPUBackend::texturepool_alloc()
{
  return new TexturePoolImpl;
}
UniformBuf *WebGPUBackend::uniformbuf_alloc(size_t size, const char *name)
{
  return new WebGPUUniformBuffer(size, name);
}
StorageBuf *WebGPUBackend::storagebuf_alloc(size_t size, GPUUsageType, const char *name)
{
  return new WebGPUStorageBuffer(size, name);
}
VertBuf *WebGPUBackend::vertbuf_alloc()
{
  return new WebGPUVertexBuffer;
}
TopLevelAS *WebGPUBackend::tlas_alloc(const char *)
{
  unsupported("ray-tracing acceleration structures");
}
BottomLevelAS *WebGPUBackend::blas_alloc(const char *)
{
  unsupported("ray-tracing acceleration structures");
}
void WebGPUBackend::shader_cache_dir_clear_old() {}
void WebGPUBackend::render_begin() {}
void WebGPUBackend::render_end()
{
  /* Blender can close the render scope after deactivating its context, which already flushes. */
  if (Context::get()) {
    WebGPUContext::active().flush();
  }
}
void WebGPUBackend::render_step(bool force_resource_release)
{
  WebGPUContext &context = WebGPUContext::active();
  context.flush();
  context.texture_pool->reset(force_resource_release);
}

}  // namespace blender::gpu
