/* SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include <webgpu/webgpu_cpp.h>

#include <cstdint>
#include <functional>
#include <memory>

class GHOST_WebGPUCanvas {
 public:
  using RequestCallback = std::function<void(std::unique_ptr<GHOST_WebGPUCanvas>, const char *)>;

  /** Requests a device compatible with the canvas. Call on the canvas's owning thread. */
  static void request(const char *selector, RequestCallback callback);

  const wgpu::Device &device() const
  {
    return device_;
  }

  wgpu::TextureFormat format() const
  {
    return format_;
  }

  uint32_t max_texture_dimension() const
  {
    return max_texture_dimension_;
  }

  /** Configure on initial window creation and when its backing size changes. */
  void configure(uint32_t width, uint32_t height);

  /** Obtain the render target for this frame. The browser presents when the frame yields. */
  wgpu::Texture acquire_texture() const;

 private:
  GHOST_WebGPUCanvas() = default;

  wgpu::Instance instance_;
  wgpu::Surface surface_;
  wgpu::Device device_;
  wgpu::TextureFormat format_ = wgpu::TextureFormat::Undefined;
  uint32_t width_ = 0;
  uint32_t height_ = 0;
  uint32_t max_texture_dimension_ = 0;
};
