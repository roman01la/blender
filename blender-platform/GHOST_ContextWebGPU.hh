/* SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include "GHOST_Context.hh"
#include "GHOST_WebGPUCanvas.hh"

/** WebGPU drawing context. A window context also owns the canvas configuration. */
class GHOST_ContextWebGPU final : public GHOST_Context {
 public:
  GHOST_ContextWebGPU(const GHOST_ContextParams &params, GHOST_WebGPUCanvas &canvas);
  GHOST_ContextWebGPU(const GHOST_ContextParams &params, wgpu::Device device);
  ~GHOST_ContextWebGPU() override;

  void resize(uint32_t width, uint32_t height);
  const wgpu::Device &device() const;
  wgpu::TextureFormat format() const;

  /** Valid between successful swapBufferAcquire() and swapBufferRelease() calls. */
  const wgpu::Texture &drawable() const;

  GHOST_TSuccess initializeDrawingContext() override;
  GHOST_TSuccess activateDrawingContext() override;
  GHOST_TSuccess releaseDrawingContext() override;
  GHOST_TSuccess updateDrawingContext() override;
  GHOST_TSuccess releaseNativeHandles() override;
  GHOST_TSuccess swapBufferAcquire() override;
  GHOST_TSuccess swapBufferRelease() override;
  GHOST_TSuccess setSwapInterval(int interval) override;
  GHOST_TSuccess getSwapInterval(int &interval_out) override;
  GHOST_TVSyncModes getVSync() override;
  bool isUpsideDown() const override;

 private:
  GHOST_WebGPUCanvas *canvas_ = nullptr;
  wgpu::Device device_;
  wgpu::Texture drawable_;
  uint32_t width_ = 0;
  uint32_t height_ = 0;
  bool attached_ = true;
  bool presented_ = false;
  bool workspace_frame_submitted_ = false;
};
