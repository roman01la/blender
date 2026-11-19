/* SPDX-License-Identifier: GPL-2.0-or-later */

#include "GHOST_WebGPUCanvas.hh"

#include <emscripten.h>

#include <array>
#include <utility>

// clang-format off
EM_JS(void, resize_offscreen_canvas, (uint32_t width, uint32_t height), {
  if (Module.canvas.width !== width) Module.canvas.width = width;
  if (Module.canvas.height !== height) Module.canvas.height = height;
});
// clang-format on

void GHOST_WebGPUCanvas::request(const char *selector, RequestCallback callback)
{
  auto canvas = std::unique_ptr<GHOST_WebGPUCanvas>(new GHOST_WebGPUCanvas());
  canvas->instance_ = wgpu::CreateInstance();
  if (!canvas->instance_) {
    callback(nullptr, "Could not initialize WebGPU.");
    return;
  }

  wgpu::EmscriptenSurfaceSourceCanvasHTMLSelector source;
  source.selector = selector;
  wgpu::SurfaceDescriptor surface_descriptor;
  surface_descriptor.nextInChain = &source;
  canvas->surface_ = canvas->instance_.CreateSurface(&surface_descriptor);
  if (!canvas->surface_) {
    callback(nullptr, "Could not create a WebGPU canvas surface.");
    return;
  }

  wgpu::RequestAdapterOptions options;
  options.compatibleSurface = canvas->surface_;
  options.powerPreference = wgpu::PowerPreference::HighPerformance;
  const auto instance = canvas->instance_;
  instance.RequestAdapter(
      &options,
      wgpu::CallbackMode::AllowSpontaneous,
      [canvas = std::move(canvas), callback = std::move(callback)](
          wgpu::RequestAdapterStatus status, wgpu::Adapter adapter, wgpu::StringView) mutable {
        if (status != wgpu::RequestAdapterStatus::Success) {
          callback(nullptr, "Could not acquire a WebGPU adapter.");
          return;
        }

        wgpu::SurfaceCapabilities capabilities;
        if (canvas->surface_.GetCapabilities(adapter, &capabilities) != wgpu::Status::Success ||
            capabilities.formatCount == 0)
        {
          callback(nullptr, "No WebGPU canvas format is available.");
          return;
        }
        canvas->format_ = capabilities.formats[0];

        constexpr std::array optional_features = {wgpu::FeatureName::Depth32FloatStencil8,
                                                  wgpu::FeatureName::IndirectFirstInstance,
                                                  wgpu::FeatureName::DualSourceBlending,
                                                  wgpu::FeatureName::TextureFormatsTier1,
                                                  wgpu::FeatureName::TextureFormatsTier2,
                                                  wgpu::FeatureName::TextureComponentSwizzle,
                                                  wgpu::FeatureName::Float32Filterable,
                                                  wgpu::FeatureName::Float32Blendable};
        std::array<wgpu::FeatureName, optional_features.size()> features;
        wgpu::DeviceDescriptor descriptor{};
        wgpu::Limits available{};
        wgpu::Limits required{};
        if (adapter.GetLimits(&available) == wgpu::Status::Success) {
          required.maxTextureDimension2D = available.maxTextureDimension2D;
          /* Hair refinement and EEVEE use more resource bindings than the WebGPU defaults. */
          required.maxStorageBuffersPerShaderStage = available.maxStorageBuffersPerShaderStage;
          required.maxStorageBufferBindingSize = available.maxStorageBufferBindingSize;
          required.maxBufferSize = available.maxBufferSize;
          required.maxStorageTexturesPerShaderStage = available.maxStorageTexturesPerShaderStage;
          required.maxSampledTexturesPerShaderStage = available.maxSampledTexturesPerShaderStage;
          required.maxSamplersPerShaderStage = available.maxSamplersPerShaderStage;
          required.maxUniformBuffersPerShaderStage = available.maxUniformBuffersPerShaderStage;
          required.maxColorAttachmentBytesPerSample = available.maxColorAttachmentBytesPerSample;
          required.maxComputeInvocationsPerWorkgroup = available.maxComputeInvocationsPerWorkgroup;
          required.maxComputeWorkgroupStorageSize = available.maxComputeWorkgroupStorageSize;
          required.maxComputeWorkgroupSizeX = available.maxComputeWorkgroupSizeX;
          required.maxComputeWorkgroupSizeY = available.maxComputeWorkgroupSizeY;
          required.maxComputeWorkgroupSizeZ = available.maxComputeWorkgroupSizeZ;
          descriptor.requiredLimits = &required;
        }
        descriptor.requiredFeatures = features.data();
        for (const auto feature : optional_features) {
          if (adapter.HasFeature(feature)) {
            features[descriptor.requiredFeatureCount++] = feature;
          }
        }
        adapter.RequestDevice(
            &descriptor,
            wgpu::CallbackMode::AllowSpontaneous,
            [canvas = std::move(canvas), callback = std::move(callback)](
                wgpu::RequestDeviceStatus status, wgpu::Device device, wgpu::StringView) mutable {
              if (status != wgpu::RequestDeviceStatus::Success) {
                callback(nullptr, "Could not acquire a WebGPU device.");
                return;
              }
              wgpu::Limits limits{};
              if (device.GetLimits(&limits) != wgpu::Status::Success ||
                  limits.maxTextureDimension2D == 0)
              {
                callback(nullptr, "Could not read the WebGPU canvas size limit.");
                return;
              }
              canvas->max_texture_dimension_ = limits.maxTextureDimension2D;
              canvas->device_ = std::move(device);
              callback(std::move(canvas), nullptr);
            });
      });
}

void GHOST_WebGPUCanvas::configure(uint32_t width, uint32_t height)
{
  if (width == width_ && height == height_) {
    return;
  }
  if (width > 0 && height > 0 && width_ > 0 && height_ > 0) {
    /* Changing the backing size expires the previous texture. The existing configuration
     * still applies to the next acquisition. */
    resize_offscreen_canvas(width, height);
  }
  else if (width > 0 && height > 0) {
    wgpu::SurfaceConfiguration configuration;
    configuration.device = device_;
    configuration.format = format_;
    configuration.width = width;
    configuration.height = height;
    surface_.Configure(&configuration);
  }
  else if (width_ > 0 && height_ > 0) {
    surface_.Unconfigure();
  }
  width_ = width;
  height_ = height;
}

wgpu::Texture GHOST_WebGPUCanvas::acquire_texture() const
{
  if (width_ == 0 || height_ == 0) {
    return {};
  }
  wgpu::SurfaceTexture drawable;
  surface_.GetCurrentTexture(&drawable);
  if (drawable.status == wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal ||
      drawable.status == wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal)
  {
    return drawable.texture;
  }
  return {};
}
