/* SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include <webgpu/webgpu_cpp.h>

#include "gpu_shader_private.hh"
#include "webgpu_batch.hh"

#include <array>
#include <cstddef>
#include <map>

namespace blender::gpu {

class WebGPUShader final : public Shader {
 public:
  explicit WebGPUShader(const char *name) : Shader(name) {}

  void init(const shader::ShaderCreateInfo &info, bool is_codegen_only) override;
  const shader::ShaderCreateInfo &patch_create_info(const shader::ShaderCreateInfo &info) override;

  void vertex_shader_from_glsl(const shader::ShaderCreateInfo &info,
                               MutableSpan<StringRefNull> sources) override;
  void geometry_shader_from_glsl(const shader::ShaderCreateInfo &info,
                                 MutableSpan<StringRefNull> sources) override;
  void fragment_shader_from_glsl(const shader::ShaderCreateInfo &info,
                                 MutableSpan<StringRefNull> sources) override;
  void compute_shader_from_glsl(const shader::ShaderCreateInfo &info,
                                MutableSpan<StringRefNull> sources) override;
  bool finalize(const shader::ShaderCreateInfo *info = nullptr) override;
  void warm_cache(int limit) override;

  void bind(const shader::SpecializationConstants *constants_state) override;
  void unbind() override;
  void uniform_float(int location, int comp_len, int array_size, const float *data) override;
  void uniform_int(int location, int comp_len, int array_size, const int *data) override;

  std::string resources_declare(const shader::ShaderCreateInfo &info) const override;
  std::string vertex_interface_declare(const shader::ShaderCreateInfo &info) const override;
  std::string fragment_interface_declare(const shader::ShaderCreateInfo &info) const override;
  std::string geometry_interface_declare(const shader::ShaderCreateInfo &info) const override;
  std::string geometry_layout_declare(const shader::ShaderCreateInfo &info) const override;
  std::string compute_layout_declare(const shader::ShaderCreateInfo &info) const override;

  const wgpu::ShaderModule &vertex_module() const
  {
    return vertex_;
  }
  const wgpu::ShaderModule &fragment_module() const
  {
    return fragment_;
  }
  const wgpu::ShaderModule &compute_module() const
  {
    return compute_;
  }

  const wgpu::RenderPipeline &render_pipeline(const WebGPURenderPipelineKey &key);
  void bind_compute_pipeline(const wgpu::ComputePassEncoder &pass);
  void bind_draw_resources(const wgpu::RenderPassEncoder &pass);
  bool uses_point_sprites() const
  {
    return point_sprites_;
  }
  bool uses_barycentric_coords() const
  {
    return barycentric_coords_;
  }
  uint32_t point_vertex_id_location() const
  {
    return point_vertex_id_location_;
  }
  void set_point_draw_state(int instance, bool indexed = false);
  void draw_pulled_vertices(const wgpu::RenderPassEncoder &pass,
                            const wgpu::RenderPipeline &pipeline,
                            const WebGPUVertexPullCommands &commands,
                            const wgpu::Buffer &indices,
                            std::span<const wgpu::Buffer> streams,
                            std::span<const uint32_t> stream_offsets,
                            uint32_t count,
                            uint32_t direct_vertex_count,
                            uint32_t direct_instance_count);

 private:
  struct Uniform {
    size_t offset;
    uint32_t array_len;
    uint32_t array_stride;
    uint32_t rows;
    uint32_t columns;
  };

  struct ResourceGroup {
    std::vector<wgpu::BindGroupLayoutEntry> bindings;
    wgpu::BindGroupLayout layout;
    std::vector<wgpu::BindGroupEntry> last_entries;
    wgpu::BindGroup last_group;
  };

  using SamplerOverrides = std::vector<std::pair<uint32_t, uint8_t>>;

  struct PipelineCache {
    std::map<WebGPURenderPipelineKey, wgpu::RenderPipeline> render_pipelines;
    wgpu::ComputePipeline compute_pipeline;
  };

  struct PipelineVariant {
    std::vector<wgpu::ConstantEntry> vertex_constants;
    std::vector<wgpu::ConstantEntry> fragment_constants;
    std::vector<wgpu::ConstantEntry> compute_constants;
    PipelineCache pipelines;
    std::map<SamplerOverrides, PipelineCache> sampler_pipelines;
  };

  static SamplerOverrides sampler_overrides(uint64_t mask);
  static void append_sampler_overrides(std::vector<wgpu::ConstantEntry> &constants,
                                       const SamplerOverrides &overrides,
                                       uint64_t mask);
  static uint64_t sampler_override_mask(const std::array<uint32_t, 18> &metadata);

  void init_draw_interface(const shader::ShaderCreateInfo &info);
  void ensure_pipeline_layout(const wgpu::Device &device);
  template<typename Pass> void bind_resources(const Pass &pass);
  template<typename Pass> void bind_resource_group(const Pass &pass, uint32_t index);
  template<typename Pass> void bind_uniforms(const Pass &pass);
  void update_draw_uniform(int location, int comp_len, int array_size, const void *data);
  const wgpu::ShaderModule &vertex_module(uint32_t packed_normals);
  const wgpu::ShaderModule &vertex_pull_module(const WebGPURenderPipelineKey &key);
  wgpu::PipelineLayout vertex_pull_layout(const WebGPURenderPipelineKey &key) const;

  std::vector<Uniform> uniforms_;
  std::vector<std::byte> uniform_data_;
  bool point_sprites_ = false;
  bool point_coord_ = false;
  bool barycentric_coords_ = false;
  uint32_t point_vertex_id_location_ = 0;
  size_t point_state_offset_ = 0;
  size_t pull_stream_offsets_ = 0;
  struct PullInput {
    uint32_t location;
    shader::Type type;
    std::string name;
  };
  std::vector<PullInput> pull_inputs_;
  std::string pull_vertex_source_;
  using PullModuleKey = std::pair<wgpu::IndexFormat, std::vector<WebGPUVertexLayout>>;
  std::map<PullModuleKey, wgpu::ShaderModule> vertex_pull_modules_;
  bool unsupported_draw_resources_ = false;
  uint64_t depth_bindings_ = 0;
  uint64_t atomic_sampler_slots_ = 0;
  bool uses_atomic_images_ = false;
  bool uses_uint_textures_ = false;
  std::array<ResourceGroup, 4> groups_;
  /* Four words per group of bindings, then two words for the emulated sampler overrides. */
  std::array<uint32_t, 18> vertex_bindings_{}, fragment_bindings_{}, compute_bindings_{};
  wgpu::PipelineLayout pipeline_layout_;
  std::map<WGPUBuffer, wgpu::BindGroup> draw_uniform_bindings_;
  std::vector<uint32_t> vertex_constant_ids_, fragment_constant_ids_, compute_constant_ids_;
  std::vector<std::string> constant_keys_;
  PipelineVariant default_pipelines_;
  std::map<std::vector<uint32_t>, PipelineVariant> specialized_pipelines_;
  PipelineVariant *bound_pipelines_ = &default_pipelines_;

  wgpu::ShaderModule vertex_;
  std::vector<uint32_t> vertex_spirv_;
  std::map<uint32_t, wgpu::ShaderModule> packed_vertex_modules_;
  wgpu::ShaderModule fragment_;
  wgpu::ShaderModule compute_;
  std::string error_;
};

}  // namespace blender::gpu
