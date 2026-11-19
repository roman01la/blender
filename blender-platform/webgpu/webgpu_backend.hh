/* SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include "gpu_backend.hh"

namespace blender::gpu {

class WebGPUBackend final : public GPUBackend {
 public:
  WebGPUBackend();
  ~WebGPUBackend() override;

  static bool is_supported();

  void init_resources() override;
  void delete_resources() override;
  void compute_dispatch(int groups_x, int groups_y, int groups_z) override;
  void compute_dispatch_indirect(StorageBuf *indirect_buf) override;
  Context *context_alloc(GHOST_IWindow *window, GHOST_IContext *context) override;
  Batch *batch_alloc() override;
  Fence *fence_alloc() override;
  WorkInFlight *work_in_flight_alloc(unsigned int max_in_flight) override;
  FrameBuffer *framebuffer_alloc(const char *name) override;
  IndexBuf *indexbuf_alloc() override;
  PixelBuffer *pixelbuf_alloc(size_t size) override;
  QueryPool *querypool_alloc() override;
  Shader *shader_alloc(const char *name) override;
  Texture *texture_alloc(const char *name) override;
  TexturePool *texturepool_alloc() override;
  UniformBuf *uniformbuf_alloc(size_t size, const char *name) override;
  StorageBuf *storagebuf_alloc(size_t size, GPUUsageType usage, const char *name) override;
  VertBuf *vertbuf_alloc() override;
  TopLevelAS *tlas_alloc(const char *name) override;
  BottomLevelAS *blas_alloc(const char *name) override;
  void shader_cache_dir_clear_old() override;
  void render_begin() override;
  void render_end() override;
  void render_step(bool force_resource_release) override;
};

}  // namespace blender::gpu
