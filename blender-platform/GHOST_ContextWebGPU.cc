/* SPDX-License-Identifier: GPL-2.0-or-later */

#include "GHOST_ContextWebGPU.hh"

#include <emscripten.h>

#include <utility>

GHOST_ContextWebGPU::GHOST_ContextWebGPU(const GHOST_ContextParams &params,
                                         GHOST_WebGPUCanvas &canvas)
    : GHOST_Context(params), canvas_(&canvas), device_(canvas.device())
{
}

GHOST_ContextWebGPU::GHOST_ContextWebGPU(const GHOST_ContextParams &params, wgpu::Device device)
    : GHOST_Context(params), device_(std::move(device))
{
}

GHOST_ContextWebGPU::~GHOST_ContextWebGPU()
{
  releaseNativeHandles();
}

void GHOST_ContextWebGPU::resize(uint32_t width, uint32_t height)
{
  width_ = width;
  height_ = height;
  if (width == 0 || height == 0) {
    updateDrawingContext();
  }
}

const wgpu::Device &GHOST_ContextWebGPU::device() const
{
  return device_;
}

wgpu::TextureFormat GHOST_ContextWebGPU::format() const
{
  return canvas_ ? canvas_->format() : wgpu::TextureFormat::Undefined;
}

const wgpu::Texture &GHOST_ContextWebGPU::drawable() const
{
  return drawable_;
}

GHOST_TSuccess GHOST_ContextWebGPU::initializeDrawingContext()
{
  return device_ ? activateDrawingContext() : GHOST_kFailure;
}

GHOST_TSuccess GHOST_ContextWebGPU::activateDrawingContext()
{
  if (!attached_) {
    return GHOST_kFailure;
  }
  active_context_ = this;
  return GHOST_kSuccess;
}

GHOST_TSuccess GHOST_ContextWebGPU::releaseDrawingContext()
{
  if (active_context_ == this) {
    active_context_ = nullptr;
  }
  return GHOST_kSuccess;
}

GHOST_TSuccess GHOST_ContextWebGPU::updateDrawingContext()
{
  if (!attached_) {
    return GHOST_kFailure;
  }
  if (canvas_ && !drawable_) {
    canvas_->configure(width_, height_);
  }
  return GHOST_kSuccess;
}

GHOST_TSuccess GHOST_ContextWebGPU::releaseNativeHandles()
{
  if (attached_) {
    drawable_ = nullptr;
    if (canvas_) {
      canvas_->configure(0, 0);
    }
    releaseDrawingContext();
    attached_ = false;
  }
  return GHOST_kSuccess;
}

GHOST_TSuccess GHOST_ContextWebGPU::swapBufferAcquire()
{
  if (!canvas_ || drawable_ || !updateDrawingContext()) {
    return GHOST_kFailure;
  }
  drawable_ = canvas_->acquire_texture();
  return drawable_ ? GHOST_kSuccess : GHOST_kFailure;
}

GHOST_TSuccess GHOST_ContextWebGPU::swapBufferRelease()
{
  if (!drawable_) {
    return GHOST_kFailure;
  }
  /* Submitted rendering is presented by the browser when the frame callback yields. */
  drawable_ = nullptr;
  EM_ASM({ Module["blenderFrameDrew"] = true; });
  if (!presented_) {
    presented_ = true;
    EM_ASM({ Module["blenderFirstFrame"]?.(); });
  }
  /* The window's initial theme clear also swaps. Wait for the first frame from the editor's
   * main loop, whose rendering (including any viewport) is already submitted at this point. */
  if (!workspace_frame_submitted_ &&
      EM_ASM_INT({ return Boolean(Module["blenderWorkspaceStarted"]); }))
  {
    workspace_frame_submitted_ = true;
    device_.GetQueue().OnSubmittedWorkDone(
        wgpu::CallbackMode::AllowSpontaneous, [](wgpu::QueueWorkDoneStatus status, wgpu::StringView) {
          EM_ASM({ Module["blenderViewportReady"]?.(Boolean($0)); },
                 status == wgpu::QueueWorkDoneStatus::Success);
        });
  }
  return GHOST_kSuccess;
}

GHOST_TSuccess GHOST_ContextWebGPU::setSwapInterval(int interval)
{
  return interval == 1 ? GHOST_kSuccess : GHOST_kFailure;
}

GHOST_TSuccess GHOST_ContextWebGPU::getSwapInterval(int &interval_out)
{
  interval_out = 1;
  return GHOST_kSuccess;
}

GHOST_TVSyncModes GHOST_ContextWebGPU::getVSync()
{
  return GHOST_kVSyncModeOn;
}

bool GHOST_ContextWebGPU::isUpsideDown() const
{
  return true;
}
