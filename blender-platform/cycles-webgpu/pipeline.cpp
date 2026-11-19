/* SPDX-License-Identifier: Apache-2.0 */

#include "device/webgpu/pipeline.h"

#include "device/device.h"
#include "device/memory.h"
#include "device/queue.h"
#include "device/webgpu/device.h"
#include "device/webgpu/lighting_layout.h"
#include "device/webgpu/material_layout.h"
#include "device/webgpu/oidn.h"
#include "device/webgpu/path_state.h"
#include "device/webgpu/runtime.h"
#include "device/webgpu/scene.h"
#include "device/webgpu/wavefront_queue.h"
#include "session/buffers.h"
#include "util/log.h"
#include "util/vector.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <limits>
#include <vector>

CCL_NAMESPACE_BEGIN

namespace {

constexpr uint32_t workgroup_size = 64;
/* The stock scheduler takes one eighth of the path pool as its work tile limit. */
constexpr uint32_t min_path_capacity = 8;
constexpr size_t path_pool_budget_words = 32 * 1024 * 1024 / sizeof(uint32_t);

constexpr uint32_t pixel_words_per_path = 4;
constexpr uint32_t ray_header_words = 2;
constexpr uint32_t hit_words_per_path = 8;
constexpr uint32_t shading_words_per_path = 24;
constexpr uint32_t radiance_words_per_path = 8;
/* Denoising feature throughput, albedo sum and world normal sum. */
constexpr uint32_t denoising_words_per_path = 12;
constexpr uint32_t work_tile_header_words = 4;
constexpr uint32_t work_tile_words = 8;
/* Origin, direction, self-intersection IDs, and source triangle geometry. */
constexpr uint32_t shadow_words_per_path = 16;

constexpr uint32_t ray_words_per_path(const bool adaptive)
{
  /* The shared active/free queue is followed by one film class word per adaptive path. */
  return adaptive ? 14 : 13;
}

struct PathBufferSize {
  size_t words_per_path;
  size_t header_words = 0;
};

constexpr auto path_buffer_sizes(const bool adaptive,
                                 const bool deferred_shadow,
                                 const bool guided)
{
  return std::array{PathBufferSize{pixel_words_per_path},
                    PathBufferSize{ray_words_per_path(adaptive), ray_header_words},
                    PathBufferSize{hit_words_per_path},
                    PathBufferSize{deferred_shadow ? shadow_words_per_path : 0},
                    PathBufferSize{shading_words_per_path},
                    PathBufferSize{radiance_words_per_path},
                    PathBufferSize{guided ? denoising_words_per_path : 0},
                    PathBufferSize{CYCLES_WEBGPU_MATERIAL_WORDS},
                    PathBufferSize{CYCLES_WEBGPU_PATH_WORDS},
                    /* A batch cannot admit more than one work tile per path slot. */
                    PathBufferSize{work_tile_words, work_tile_header_words}};
}

constexpr uint32_t path_capacity_for_limits(const WebGPUComputeLimits &limits,
                                            const bool adaptive,
                                            const bool deferred_shadow,
                                            size_t &bytes_per_path,
                                            const bool guided = false)
{
  const size_t binding_words = limits.storage_buffer_bytes / sizeof(uint32_t);
  uint64_t capacity = std::min<uint64_t>(WebGPURenderPipeline::max_path_capacity,
                                         uint64_t(limits.workgroups_per_dimension) * workgroup_size);
  size_t words_per_path = 0;
  size_t header_words = 0;
  for (const auto &buffer : path_buffer_sizes(adaptive, deferred_shadow, guided)) {
    if (buffer.words_per_path == 0) {
      continue;
    }
    if (binding_words < buffer.header_words) {
      return 0;
    }
    capacity = std::min<uint64_t>(
        capacity, (binding_words - buffer.header_words) / buffer.words_per_path);
    words_per_path += buffer.words_per_path;
    header_words += buffer.header_words;
  }
  bytes_per_path = words_per_path * sizeof(uint32_t);
  return uint32_t(std::min<uint64_t>(capacity,
                                      (path_pool_budget_words - header_words) / words_per_path));
}

static_assert([] {
  size_t bytes_per_path = 0;
  return path_capacity_for_limits({128 * 1024 * 1024, 65535}, true, true, bytes_per_path) ==
             WebGPURenderPipeline::max_path_capacity &&
         path_capacity_for_limits({1536, 1}, true, true, bytes_per_path) == 16 &&
         path_capacity_for_limits({128 * 1024 * 1024, 1}, false, false, bytes_per_path) == 64;
}());

/* The queue compiles a few fused stages into private slots alongside the
 * DeviceKernel values the stock scheduler uses to select an iteration. */
constexpr DeviceKernel camera_kernel = DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA;
constexpr DeviceKernel adaptive_camera_slots_kernel = DEVICE_KERNEL_INTEGRATOR_INIT_FROM_BAKE;
constexpr DeviceKernel adaptive_camera_rays_kernel =
    DEVICE_KERNEL_INTEGRATOR_INTERSECT_VOLUME_STACK;
constexpr DeviceKernel adaptive_admission_kernel = DEVICE_KERNEL_INTEGRATOR_COMPACT_SHADOW_STATES;
constexpr DeviceKernel bvh_kernel = DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST;
constexpr DeviceKernel shadow_kernel = DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW;
constexpr DeviceKernel surface_kernel = DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE;
constexpr DeviceKernel background_kernel = DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND;
constexpr DeviceKernel tree_background_kernel = DEVICE_KERNEL_INTEGRATOR_SHADE_DEDICATED_LIGHT;
constexpr DeviceKernel tree_light_kernel = DEVICE_KERNEL_INTEGRATOR_INTERSECT_DEDICATED_LIGHT;
constexpr DeviceKernel init_state_kernel = DEVICE_KERNEL_INTEGRATOR_RESET;
constexpr DeviceKernel svm_kernel = DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE_RAYTRACE;
constexpr DeviceKernel wavefront_light_kernel = DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT_FORWARD;
constexpr DeviceKernel point_distribution_kernel = DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW;
constexpr DeviceKernel path_update_kernel = DEVICE_KERNEL_INTEGRATOR_MEGAKERNEL;
constexpr DeviceKernel reflection_light_kernel = DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT_NEE;
constexpr DeviceKernel reflection_update_kernel = DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME;
constexpr DeviceKernel accumulate_kernel = DEVICE_KERNEL_FILM_CONVERT_COMBINED;
constexpr DeviceKernel adaptive_accumulate_kernel = DEVICE_KERNEL_FILM_CONVERT_SAMPLE_COUNT;
constexpr DeviceKernel adaptive_count_kernel = DEVICE_KERNEL_PREFIX_SUM;
constexpr DeviceKernel guided_init_kernel = DEVICE_KERNEL_INTEGRATOR_INTERSECT_SUBSURFACE;
constexpr DeviceKernel denoising_surface_kernel = DEVICE_KERNEL_FILM_CONVERT_FLOAT3;
constexpr DeviceKernel guided_background_kernel = DEVICE_KERNEL_FILM_CONVERT_RGBE;
constexpr DeviceKernel guided_tree_background_kernel = DEVICE_KERNEL_FILM_CONVERT_FLOAT4;
constexpr DeviceKernel guided_accumulate_kernel = DEVICE_KERNEL_FILM_CONVERT_MOTION;

static_assert(DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST + 1 == 3);
static_assert(DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND + 1 == 9);
static_assert(DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE + 1 == 12);

constexpr auto storage = WebGPUKernelBinding::Type::Storage;
constexpr auto read_only = WebGPUKernelBinding::Type::ReadOnlyStorage;
constexpr auto uniform = WebGPUKernelBinding::Type::Uniform;
constexpr size_t material_bytes = CYCLES_WEBGPU_MATERIAL_WORDS * sizeof(float);
constexpr size_t denoising_bytes = denoising_words_per_path * sizeof(float);
constexpr size_t denoising_params_bytes = 16 * sizeof(uint32_t);

/* The translated WGSL uses read_write bindings even for its input buffers. */
constexpr WebGPUKernelBinding camera_bindings[] = {
    {0, storage, 32}, {1, storage, 128}, {2, storage, 16}, {3, storage, 52}, {4, storage, 28}};
constexpr WebGPUKernelBinding bvh_bindings[] = {
    {0, storage, 40}, {1, storage, 52}, {2, storage, 32}, {3, storage, 16}};
constexpr WebGPUKernelBinding surface_bindings[] = {
    {0, storage, 40}, {1, storage, 8}, {2, storage, 52}, {3, storage, 32}, {4, storage, 32}};
constexpr WebGPUKernelBinding background_bindings[] = {
    {0, storage, 52},
    {1, storage, 32},
    {2, storage, 48},
    {3, storage, 28},
    {4, storage, CYCLES_WEBGPU_PATH_WORDS * sizeof(uint32_t)}};
constexpr WebGPUKernelBinding init_state_bindings[] = {
    {0, read_only, 4},
    {1, storage, CYCLES_WEBGPU_PATH_WORDS * sizeof(uint32_t)},
    {2, storage, 16},
    {3, read_only, 16}};
constexpr WebGPUKernelBinding svm_bindings[] = {{0, storage, 8},
                                                {1, storage, 32},
                                                {2, storage, 52},
                                                {3, storage, material_bytes},
                                                {4, storage, 32}};
constexpr WebGPUKernelBinding wavefront_light_bindings[] = {
    {0, storage, 40},
    {1, storage, 52},
    {2, storage, 32},
    {3, storage, 32},
    {4, storage, 48},
    {5, storage, 16},
    {6, storage, material_bytes},
    {7, storage, CYCLES_WEBGPU_PATH_WORDS * sizeof(uint32_t)}};
constexpr auto deferred_light_bindings = [] {
  auto bindings = std::to_array(wavefront_light_bindings);
  bindings[0].minimum_size = shadow_words_per_path * sizeof(uint32_t);
  return bindings;
}();
constexpr WebGPUKernelBinding shadow_bindings[] = {
    {0, storage, 40},
    {1, storage, 52},
    {2, storage, shadow_words_per_path * sizeof(uint32_t)},
    {3, storage, 16}};
constexpr WebGPUKernelBinding path_update_bindings[] = {
    {0, storage, 52},
    {1, storage, 32},
    {2, storage, 32},
    {3, storage, 16},
    {4, storage, 28},
    {5, storage, material_bytes},
    {6, storage, 16},
    {7, storage, CYCLES_WEBGPU_PATH_WORDS * sizeof(uint32_t)}};
constexpr WebGPUKernelBinding accumulate_bindings[] = {
    {3, storage, 16},
    {5, read_only, CYCLES_WEBGPU_PATH_WORDS * sizeof(uint32_t)},
    {6, read_only, 4},
    {7, uniform, 48}};

constexpr auto guided_init_bindings = std::array{
    init_state_bindings[0], init_state_bindings[1], init_state_bindings[2], init_state_bindings[3],
    WebGPUKernelBinding{6, storage, denoising_bytes}};
constexpr WebGPUKernelBinding denoising_surface_bindings[] = {
    {0, storage, 52}, {1, storage, 32}, {2, storage, material_bytes}, {3, storage, 28},
    {4, storage, 16}, {5, storage, CYCLES_WEBGPU_PATH_WORDS * sizeof(uint32_t)},
    {6, storage, denoising_bytes}, {7, storage, denoising_params_bytes}};
constexpr auto guided_background_bindings = std::array{
    background_bindings[0], background_bindings[1], background_bindings[2],
    background_bindings[3], background_bindings[4],
    WebGPUKernelBinding{5, storage, denoising_bytes}};
constexpr auto guided_accumulate_bindings = std::array{
    accumulate_bindings[0], accumulate_bindings[1], accumulate_bindings[2], accumulate_bindings[3],
    WebGPUKernelBinding{9, read_only, denoising_bytes},
    WebGPUKernelBinding{10, read_only, denoising_params_bytes}};

constexpr WebGPUKernelBinding adaptive_camera_slots_bindings[] = {
    {0, storage, 32}, {1, storage, 128}, {2, storage, 16}, {3, storage, 52}};
constexpr WebGPUKernelBinding adaptive_admission_bindings[] = {
    {0, read_only, 4}, {2, storage, 16}, {3, read_only, 16}, {4, storage, 16}, {5, uniform, 48}};
constexpr WebGPUKernelBinding adaptive_count_bindings[] = {
    {0, read_only, 16}, {1, read_only, 52}, {2, storage, 4}};
constexpr WebGPUKernelBinding adaptive_filter_bindings[] = {{0, storage, 16}, {1, storage, 52}};

template<typename T, size_t Extent>
bool upload(Device &device, device_only_memory<T> &buffer, const std::span<const T, Extent> data)
{
  buffer.alloc_to_device(data.size(), true);
  return buffer.device_pointer && device_webgpu_upload(device, buffer, std::as_bytes(data));
}

bool finite(const float value)
{
  return (std::bit_cast<uint32_t>(value) & 0x7f800000u) != 0x7f800000u;
}

bool valid_offsets(std::span<const uint32_t> offsets, size_t header_words, size_t buffer_words)
{
  return std::is_sorted(offsets.begin(), offsets.end()) && offsets.front() >= header_words &&
         offsets.back() <= buffer_words;
}

bool valid_light_tree(const std::span<const uint32_t> light, const size_t prefix)
{
  if (prefix > light.size() || light.size() - prefix < CYCLES_WEBGPU_TREE_DESCRIPTOR_WORDS) {
    return false;
  }
  const auto tree = light.subspan(prefix);
  const uint64_t lamps = tree[CYCLES_WEBGPU_TREE_LAMP_COUNT];
  const uint64_t nodes = tree[CYCLES_WEBGPU_TREE_NODE_COUNT];
  const uint64_t emitters = tree[CYCLES_WEBGPU_TREE_EMITTER_COUNT];
  const uint64_t map_w = tree[CYCLES_WEBGPU_TREE_WORLD_MAP_WIDTH];
  const uint64_t map_h = tree[CYCLES_WEBGPU_TREE_WORLD_MAP_HEIGHT];
  const uint64_t tex_w = tree[CYCLES_WEBGPU_TREE_WORLD_WIDTH];
  const uint64_t tex_h = tree[CYCLES_WEBGPU_TREE_WORLD_HEIGHT];
  const uint32_t world_lamp = tree[CYCLES_WEBGPU_TREE_WORLD_LAMP];
  const uint32_t world_emitter = tree[CYCLES_WEBGPU_TREE_WORLD_EMITTER];
  const uint32_t area_lamp = tree[CYCLES_WEBGPU_TREE_AREA_LAMP];
  const bool sampled_world = world_lamp != CYCLES_WEBGPU_TREE_NO_LAMP;
  if (lamps == 0 || nodes == 0 || emitters != lamps ||
      (area_lamp != CYCLES_WEBGPU_TREE_NO_LAMP && area_lamp >= lamps) ||
      (sampled_world ? world_lamp >= lamps || world_emitter >= emitters || !map_w || !map_h :
                       world_emitter != CYCLES_WEBGPU_TREE_NO_LAMP || map_w || map_h) ||
      std::max({map_w, map_h, tex_w, tex_h}) > light.size())
  {
    return false;
  }
  const uint64_t lamp_offset = prefix + CYCLES_WEBGPU_TREE_DESCRIPTOR_WORDS;
  const uint64_t node_offset = lamp_offset + lamps * CYCLES_WEBGPU_TREE_LAMP_WORDS;
  const uint64_t emitter_offset = node_offset + nodes * CYCLES_WEBGPU_TREE_NODE_WORDS;
  const uint64_t marginal_offset = emitter_offset + emitters * CYCLES_WEBGPU_TREE_EMITTER_WORDS;
  const uint64_t conditional_offset = marginal_offset + (sampled_world ? 2 * (map_h + 1) : 0);
  const uint64_t pixel_offset = conditional_offset + 2 * (map_w + 1) * map_h;
  const uint64_t sampling_offset = pixel_offset + 3 * tex_w * tex_h;
  return tree[CYCLES_WEBGPU_TREE_LAMPS] == lamp_offset &&
         tree[CYCLES_WEBGPU_TREE_NODES] == node_offset &&
         tree[CYCLES_WEBGPU_TREE_EMITTERS] == emitter_offset &&
         tree[CYCLES_WEBGPU_TREE_WORLD_MARGINAL_CDF] == marginal_offset &&
         tree[CYCLES_WEBGPU_TREE_WORLD_CONDITIONAL_CDF] == conditional_offset &&
         tree[CYCLES_WEBGPU_TREE_WORLD_PIXELS] == pixel_offset &&
         tree[CYCLES_WEBGPU_TREE_SAMPLING] == sampling_offset &&
         sampling_offset <= light.size() &&
         light.size() - sampling_offset >= CYCLES_WEBGPU_LIGHT_SAMPLING_WORDS;
}

enum class LightSampling { Legacy, PointDistribution, Tree };

}  // namespace

struct WebGPURenderPipeline::Impl {
  Device &device;
  unique_ptr<DeviceQueue> queue;
  WebGPUSceneBuffers scene{device};
  device_only_memory<float> camera{&device, "WebGPU camera"};
  device_only_memory<uint32_t> sampling{&device, "WebGPU camera sampling"};
  device_only_memory<uint32_t> tile{&device, "WebGPU work tiles"};
  device_only_memory<uint32_t> params{&device, "WebGPU stock film parameters"};
  device_only_memory<uint32_t> pixels{&device, "WebGPU pixel samples"};
  device_only_memory<uint32_t> rays{&device, "WebGPU primary rays"};
  device_only_memory<uint32_t> hits{&device, "WebGPU primary intersections"};
  device_only_memory<uint32_t> shadow_rays{&device, "WebGPU shadow rays"};
  device_only_memory<uint32_t> points{&device, "WebGPU shading points"};
  WebGPUPathStateStorage path_state{device};
  WebGPUWavefrontQueue wavefront_queue{device};
  device_only_memory<float> materials{&device, "WebGPU evaluated materials"};
  device_only_memory<float> radiance{&device, "WebGPU sampled radiance"};
  device_only_memory<float> denoising{&device, "WebGPU denoising paths"};
  device_only_memory<uint32_t> denoising_params{&device, "WebGPU denoising parameters"};
  device_only_memory<uint32_t> film{&device, "WebGPU Cycles film"};
  device_only_memory<uint32_t> adaptive_params{&device, "WebGPU adaptive parameters"};
  device_only_memory<uint32_t> adaptive_active{&device, "WebGPU adaptive active pixel count"};
  std::vector<uint32_t> packed_tiles;

  string error;
  bool initialized = false;
  bool scene_uploaded = false;
  DeviceKernel light_kernel = wavefront_light_kernel;
  DeviceKernel update_kernel = path_update_kernel;
  DeviceKernel active_background_kernel = background_kernel;
  bool deferred_shadow = false;
  bool reflection_deferred = false;
  bool full_deferred = false;
  bool shadow_loaded = false;
  bool reflection_shading_loaded = false;
  bool reflection_update_loaded = false;
  bool point_distribution_loaded = false;
  bool light_tree_loaded = false;
  bool shading_loaded = false;
  bool background_loaded = false;
  std::string (*load_shader)(const char *name) = nullptr;
  IntegratorQueueCounter queue_counts{};
  int active_paths = 0;
  bool batch_active = false;
  bool film_ready = false;
  bool adaptive_sampling = false;
  bool adaptive_loaded = false;
  bool guided = false;
  bool guided_loaded = false;
  bool guided_background_loaded = false;
  bool guided_tree_background_loaded = false;
  uint32_t sampling_pattern = 0;
  std::array<uint32_t, 12> film_config{};
  uint32_t path_capacity = 0;
  uint32_t pixel_count = 0;
  uint32_t next_sample = 0;
  uint32_t batch_end_sample = 0;
  uint64_t admitted_paths = 0;
  std::array<uint32_t, 4> crop{};

  explicit Impl(Device &device) : device(device) {}

  bool fail(const char *message)
  {
    error = device_webgpu_error(device);
    if (error.empty()) {
      error = message;
    }
    return false;
  }

  bool register_kernel(const DeviceKernel kernel,
                       const std::string_view source,
                       const std::span<const WebGPUKernelBinding> bindings,
                       const std::string_view entry = "main")
  {
    return device_webgpu_register_kernel(device, kernel, source, workgroup_size, bindings, entry);
  }

  bool prepare_shadow()
  {
    if (!shadow_loaded) {
      const auto shadow = load_shader("intersect-shadow.wgsl");
      shadow_loaded = !shadow.empty() && register_kernel(shadow_kernel, shadow, shadow_bindings);
    }
    return shadow_loaded;
  }

  bool prepare_light(const DeviceKernel kernel, const bool reflection, bool &deferred)
  {
    const auto split_light = load_shader(reflection ? "wavefront-light-deferred-reflection.wgsl" :
                                                       "wavefront-light-deferred.wgsl");
    if (!split_light.empty()) {
      deferred = true;
      return prepare_shadow() && register_kernel(kernel, split_light, deferred_light_bindings);
    }
    const auto lighting = load_shader(reflection ? "wavefront-light-reflection.wgsl" :
                                                    "wavefront-light.wgsl");
    return !lighting.empty() && register_kernel(kernel, lighting, wavefront_light_bindings);
  }

  bool prepare_reflection_update(const std::string_view update)
  {
    if (!reflection_update_loaded) {
      reflection_update_loaded = !update.empty() &&
                                 register_kernel(reflection_update_kernel, update, path_update_bindings);
    }
    return reflection_update_loaded;
  }

  bool prepare_light_tree()
  {
    if (!light_tree_loaded) {
      const auto light = load_shader("wavefront-light-deferred-reflection-tree.wgsl");
      const auto background = load_shader("shade-background-tree.wgsl");
      const auto update = load_shader("path-update-reflection.wgsl");
      if (light.empty() || background.empty() || update.empty()) {
        return fail("Cycles WebGPU needs the Light Tree shader set");
      }
      LOG_INFO << "Cycles WebGPU compiling the stock Light Tree and world sampling shaders";
      if (!prepare_shadow() || !prepare_reflection_update(update) ||
          !register_kernel(tree_light_kernel, light, deferred_light_bindings) ||
          !register_kernel(tree_background_kernel, background, background_bindings))
      {
        return fail("Cycles WebGPU could not compile the Light Tree shaders");
      }
      light_tree_loaded = true;
    }
    light_kernel = tree_light_kernel;
    update_kernel = reflection_update_kernel;
    active_background_kernel = tree_background_kernel;
    deferred_shadow = true;
    return true;
  }

  bool prepare_shading(const bool has_transmission, const LightSampling sampling)
  {
    active_background_kernel = background_kernel;
    if (sampling == LightSampling::Tree) {
      return has_transmission ? fail("Cycles WebGPU Light Tree needs opaque materials") :
                                prepare_light_tree();
    }
    if (sampling == LightSampling::PointDistribution) {
      if (has_transmission) {
        return fail("Cycles WebGPU point-light distributions need opaque materials");
      }
      if (!point_distribution_loaded) {
        const auto light = load_shader("wavefront-light-deferred-multi-point-reflection.wgsl");
        const auto update = load_shader("path-update-reflection.wgsl");
        if (light.empty() || update.empty()) {
          return fail("Cycles WebGPU needs the point-light distribution shader set");
        }
        LOG_INFO << "Cycles WebGPU compiling the stock point-light distribution shader";
        if (!prepare_shadow() || !prepare_reflection_update(update) ||
            !register_kernel(point_distribution_kernel, light, deferred_light_bindings))
        {
          return fail("Cycles WebGPU could not compile the point-light distribution shader");
        }
        point_distribution_loaded = true;
      }
      light_kernel = point_distribution_kernel;
      update_kernel = reflection_update_kernel;
      deferred_shadow = true;
      return true;
    }
    if (!has_transmission) {
      if (!reflection_shading_loaded) {
        const auto update = load_shader("path-update-reflection.wgsl");
        if (!update.empty()) {
          LOG_INFO << "Cycles WebGPU compiling reflection-only wavefront shaders";
          if (!prepare_light(reflection_light_kernel, true, reflection_deferred) ||
              !prepare_reflection_update(update))
          {
            return fail("Cycles WebGPU could not compile the reflection-only wavefront shaders");
          }
          reflection_shading_loaded = true;
        }
      }
      if (reflection_shading_loaded) {
        light_kernel = reflection_light_kernel;
        update_kernel = reflection_update_kernel;
        deferred_shadow = reflection_deferred;
        return true;
      }
    }

    if (!shading_loaded) {
      const auto update = load_shader("path-update.wgsl");
      if (update.empty()) {
        return fail("Cycles WebGPU needs both wavefront shading shaders");
      }
      LOG_INFO << "Cycles WebGPU compiling transmission-capable wavefront shaders";
      if (!prepare_light(wavefront_light_kernel, false, full_deferred) ||
          !register_kernel(path_update_kernel, update, path_update_bindings))
      {
        return fail("Cycles WebGPU could not compile the transmission-capable wavefront shaders");
      }
      shading_loaded = true;
    }
    light_kernel = wavefront_light_kernel;
    update_kernel = path_update_kernel;
    deferred_shadow = full_deferred;
    return true;
  }

  bool prepare_guides()
  {
    if (!guided_loaded) {
      const auto init = load_shader("path-state.wgsl");
      const auto film_shader = load_shader("film.wgsl");
      const auto surface = load_shader("denoise-surface.wgsl");
      if (init.empty() || film_shader.empty() || surface.empty()) {
        return fail("Cycles WebGPU needs the denoising guide shader set");
      }
      if (!register_kernel(guided_init_kernel, init, guided_init_bindings, "initialize_guided") ||
          !register_kernel(denoising_surface_kernel, surface, denoising_surface_bindings) ||
          !register_kernel(guided_accumulate_kernel,
                           film_shader,
                           guided_accumulate_bindings,
                           "retire_guided_render_buffer"))
      {
        return fail("Cycles WebGPU could not compile the denoising guide shaders");
      }
      guided_loaded = true;
    }

    if (active_background_kernel == tree_background_kernel) {
      if (!guided_tree_background_loaded) {
        const auto background = load_shader("shade-background-tree-guides.wgsl");
        if (background.empty() ||
            !register_kernel(guided_tree_background_kernel, background, guided_background_bindings))
        {
          return fail("Cycles WebGPU needs the Light Tree denoising background shader");
        }
        guided_tree_background_loaded = true;
      }
    }
    else if (background_loaded && !guided_background_loaded) {
      const auto background = load_shader("shade-background-guides.wgsl");
      if (background.empty() ||
          !register_kernel(guided_background_kernel, background, guided_background_bindings))
      {
        return fail("Cycles WebGPU needs the denoising background shader");
      }
      guided_background_loaded = true;
    }
    return true;
  }

  bool prepare_adaptive()
  {
    if (adaptive_loaded) {
      return true;
    }
    const auto slots = load_shader("adaptive-camera-slots.wgsl");
    const auto rays = load_shader("adaptive-camera-rays-retire.wgsl");
    const auto init = load_shader("path-state.wgsl");
    const auto film = load_shader("film.wgsl");
    const auto check = load_shader("adaptive-check.wgsl");
    const auto count = load_shader("adaptive-active-count.wgsl");
    const auto filter_x = load_shader("adaptive-filter-x.wgsl");
    const auto filter_y = load_shader("adaptive-filter-y.wgsl");
    if (slots.empty() || rays.empty() || init.empty() || film.empty() ||
        check.empty() || count.empty() || filter_x.empty() || filter_y.empty())
    {
      return fail("Cycles WebGPU needs the adaptive sampling shader set");
    }
    if (!register_kernel(adaptive_camera_slots_kernel, slots, adaptive_camera_slots_bindings) ||
        !register_kernel(adaptive_camera_rays_kernel, rays, camera_bindings) ||
        !register_kernel(adaptive_admission_kernel,
                         init,
                         adaptive_admission_bindings,
                         "admit_adaptive_pixels") ||
        !register_kernel(adaptive_accumulate_kernel,
                         film,
                         accumulate_bindings,
                         "retire_adaptive_render_buffer") ||
        !register_kernel(
            DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_CHECK, check, adaptive_filter_bindings) ||
        !register_kernel(adaptive_count_kernel, count, adaptive_count_bindings) ||
        !register_kernel(DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_FILTER_X,
                         filter_x,
                         adaptive_filter_bindings) ||
        !register_kernel(DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_FILTER_Y,
                         filter_y,
                         adaptive_filter_bindings))
    {
      return fail("Cycles WebGPU could not compile adaptive sampling kernels");
    }
    adaptive_active.alloc_to_device(1);
    adaptive_loaded = adaptive_active.device_pointer != 0;
    return adaptive_loaded || fail("Cycles WebGPU could not allocate the adaptive pixel counter");
  }

  template<typename... Buffer> static DeviceKernelArguments bind(const Buffer &...buffer)
  {
    return DeviceKernelArguments(&buffer.device_pointer...);
  }
};

WebGPURenderPipeline::WebGPURenderPipeline(Device &device) : impl_(make_unique<Impl>(device)) {}

WebGPURenderPipeline::~WebGPURenderPipeline() = default;

bool WebGPURenderPipeline::initialize(const WebGPURenderShaders &shaders)
{
  Impl &p = *impl_;
  p.error.clear();
  if (p.initialized) {
    return p.fail("Cycles WebGPU render pipeline is already initialized");
  }
  if (!shaders.load_shader || shaders.camera.empty() || shaders.bvh2.empty() ||
      shaders.surface.empty() || shaders.film.empty() || shaders.state_init.empty() ||
      shaders.svm.empty() || shaders.wavefront_queue.empty())
  {
    return p.fail("Cycles WebGPU render pipeline needs the stock scheduler shader set");
  }

  if (!p.register_kernel(camera_kernel, shaders.camera, camera_bindings) ||
      !p.register_kernel(bvh_kernel, shaders.bvh2, bvh_bindings) ||
      !p.register_kernel(surface_kernel, shaders.surface, surface_bindings) ||
      !p.register_kernel(init_state_kernel, shaders.state_init, init_state_bindings) ||
      !p.register_kernel(svm_kernel, shaders.svm, svm_bindings) ||
      !p.register_kernel(
          accumulate_kernel, shaders.film, accumulate_bindings, "accumulate_render_buffer") ||
      !p.wavefront_queue.initialize(shaders.wavefront_queue))
  {
    return p.fail("Cycles WebGPU render pipeline failed to compile a kernel");
  }
  p.load_shader = shaders.load_shader;
  const auto background = p.load_shader("shade-background.wgsl");
  if (!background.empty()) {
    if (!p.register_kernel(background_kernel, background, background_bindings)) {
      return p.fail("Cycles WebGPU could not compile the background shader");
    }
    p.background_loaded = true;
  }

  p.queue = p.device.gpu_queue_create();
  if (!p.queue) {
    return p.fail("Cycles WebGPU render pipeline could not create a queue");
  }
  p.queue->init_execution();
  p.initialized = true;
  return true;
}

bool WebGPURenderPipeline::upload_scene(std::span<const uint32_t> scene,
                                        std::span<const uint32_t> surface,
                                        std::span<const uint32_t> light,
                                        std::span<const uint32_t> svm,
                                        std::span<const uint32_t> sampling)
{
  Impl &p = *impl_;
  p.error.clear();
  if (!p.initialized) {
    return p.fail("Cycles WebGPU render pipeline needs to be initialized");
  }
  if (scene.size() > max_scene_words || surface.size() > max_scene_words - scene.size() ||
      light.size() > max_scene_words - scene.size() - surface.size() ||
      svm.size() > max_scene_words - scene.size() - surface.size() - light.size())
  {
    return p.fail("Cycles WebGPU preview scene exceeds 32 MiB");
  }
  constexpr size_t light_header = CYCLES_WEBGPU_LIGHT_HEADER_WORDS;
  constexpr size_t reflective_lighting_words = light_header + CYCLES_WEBGPU_GGX_LOOKUP_WORDS;
  constexpr size_t glass_lighting_words = reflective_lighting_words +
                                          CYCLES_WEBGPU_GGX_GLASS_LOOKUP_WORDS;
  const uint32_t area = light.size() >= light_header ? light[CYCLES_WEBGPU_LIGHT_EXTENDED] : 0;
  const bool point_distribution = light.size() >= light_header &&
                                  light[0] == CYCLES_WEBGPU_POINT_LIGHTS_MARKER;
  const bool light_tree = light.size() >= light_header &&
                          light[0] == CYCLES_WEBGPU_LIGHT_TREE_MARKER;
  size_t light_prefix = area ? area : light.size();
  if (light_tree) {
    light_prefix = light[CYCLES_WEBGPU_LIGHT_TREE_OFFSET];
  }
  else if (point_distribution) {
    light_prefix = light[CYCLES_WEBGPU_POINT_LIGHTS_OFFSET];
  }
  constexpr size_t surface_offsets = 5;
  if (scene.size() < 10 || !valid_offsets(scene.subspan(1, 9), 10, scene.size()) ||
      surface.size() < surface_offsets ||
      !valid_offsets(surface.first(surface_offsets), surface_offsets, surface.size()) ||
      light.size() < light_header ||
      (light_prefix != light_header && light_prefix != reflective_lighting_words &&
       light_prefix != glass_lighting_words) ||
      !finite(std::bit_cast<float>(light[3])) || std::bit_cast<float>(light[3]) <= 0.0f ||
      svm.size() < 2 || svm[0] > svm.size() - 2 || sampling.empty())
  {
    return p.fail("Cycles WebGPU preview scene has invalid packed buffers");
  }
  if (area) {
    if (area > light.size() || light.size() - area < CYCLES_WEBGPU_EXTENDED_LIGHT_WORDS) {
      return p.fail("Cycles WebGPU needs a complete area light descriptor");
    }
    const uint64_t texel_words = 3 * uint64_t(light[area + CYCLES_WEBGPU_WORLD_WIDTH]) *
                                 light[area + CYCLES_WEBGPU_WORLD_HEIGHT];
    const size_t sampling = light[area + CYCLES_WEBGPU_LIGHT_SAMPLING];
    if (sampling != area + CYCLES_WEBGPU_EXTENDED_LIGHT_WORDS + texel_words ||
        sampling > light.size() || light.size() - sampling < CYCLES_WEBGPU_LIGHT_SAMPLING_WORDS)
    {
      return p.fail("Cycles WebGPU needs a complete world image and light sampling data");
    }
  }

  if (point_distribution) {
    if (area || light_prefix > light.size() ||
        light.size() - light_prefix < CYCLES_WEBGPU_POINT_DESCRIPTOR_WORDS)
    {
      return p.fail("Cycles WebGPU needs a complete point-light distribution descriptor");
    }
    const auto point = light.subspan(light_prefix);
    const uint64_t count = point[CYCLES_WEBGPU_POINT_COUNT];
    const float pdf = std::bit_cast<float>(point[CYCLES_WEBGPU_POINT_PDF]);
    const size_t cdf_offset = light_prefix + CYCLES_WEBGPU_POINT_DESCRIPTOR_WORDS;
    const uint64_t emitters_offset = cdf_offset + (count + 1) * CYCLES_WEBGPU_POINT_DISTRIBUTION_WORDS;
    const uint64_t sampling_offset = emitters_offset + count * CYCLES_WEBGPU_POINT_EMITTER_WORDS;
    if (count < 2 || !finite(pdf) || pdf <= 0.0f || pdf > 1.0f ||
        point[CYCLES_WEBGPU_POINT_CDF] != cdf_offset ||
        point[CYCLES_WEBGPU_POINT_EMITTERS] != emitters_offset ||
        point[CYCLES_WEBGPU_POINT_SAMPLING] != sampling_offset || sampling_offset > light.size() ||
        light.size() - sampling_offset < CYCLES_WEBGPU_LIGHT_SAMPLING_WORDS)
    {
      return p.fail("Cycles WebGPU needs complete point-light emitter and distribution buffers");
    }
  }

  if (light_tree && (area || !valid_light_tree(light, light_prefix))) {
    return p.fail("Cycles WebGPU needs complete Light Tree and world sampling buffers");
  }

  p.scene_uploaded = false;
  p.film_ready = false;
  p.batch_active = false;
  const LightSampling mode = light_tree ? LightSampling::Tree :
                               point_distribution ? LightSampling::PointDistribution :
                                                    LightSampling::Legacy;
  if (!p.prepare_shading(light_prefix == glass_lighting_words, mode)) {
    return false;
  }
  if (!p.scene.upload(scene, surface, light, svm) || !upload(p.device, p.sampling, sampling) ||
      !device_webgpu_error(p.device).empty())
  {
    return p.fail("Cycles WebGPU scene upload failed");
  }
  p.sampling_pattern = sampling.front();
  p.scene_uploaded = true;
  return true;
}

bool WebGPURenderPipeline::reset(std::span<const float, 32> camera,
                                 std::span<const uint32_t, 16> denoising_config,
                                 const BufferParams &params,
                                 const size_t film_words,
                                 uint32_t &path_capacity)
{
  Impl &p = *impl_;
  p.error.clear();
  if (!p.scene_uploaded) {
    return p.fail("Cycles WebGPU render pipeline needs a scene");
  }

  const int combined = params.get_pass_offset(PASS_COMBINED);
  const int sample_count = params.get_pass_offset(PASS_SAMPLE_COUNT);
  const int adaptive_aux = params.get_pass_offset(PASS_ADAPTIVE_AUX_BUFFER);
  const bool adaptive = adaptive_aux != PASS_UNUSED;
  const int denoising_albedo = params.get_pass_offset(PASS_DENOISING_ALBEDO);
  const int denoising_normal = params.get_pass_offset(PASS_DENOISING_NORMAL);
  const bool guided = denoising_albedo != PASS_UNUSED || denoising_normal != PASS_UNUSED;
  const auto valid_guide_offset = [&](const int offset) {
    return offset == PASS_UNUSED || (offset >= 0 && offset <= params.pass_stride - 3);
  };
  const uint64_t pixels = uint64_t(params.width) * params.height;
  if (params.width <= 0 || params.height <= 0 || params.full_x < 0 || params.full_y < 0 ||
      pixels > max_crop_pixels || params.stride < params.width || combined < 0 ||
      params.pass_stride < 4 ||
      combined > params.pass_stride - 4 ||
      (sample_count != PASS_UNUSED &&
       (sample_count < 0 || sample_count >= params.pass_stride ||
        (sample_count >= combined && sample_count < combined + 4))) ||
      (adaptive_aux != PASS_UNUSED &&
       (sample_count == PASS_UNUSED || adaptive_aux < 0 || adaptive_aux > params.pass_stride - 4 ||
        (adaptive_aux <= combined + 3 && combined <= adaptive_aux + 3) ||
        (sample_count >= adaptive_aux && sample_count <= adaptive_aux + 3) ||
        !finite(params.exposure))) ||
      !valid_guide_offset(denoising_albedo) || !valid_guide_offset(denoising_normal) ||
      film_words > std::numeric_limits<uint32_t>::max() ||
      film_words > std::numeric_limits<size_t>::max() / sizeof(uint32_t) ||
      !std::all_of(camera.begin(), camera.end(), finite))
  {
    return p.fail("Cycles WebGPU needs a valid camera and Combined render buffer layout");
  }

  const int64_t origin = int64_t(params.offset) + params.full_x +
                         int64_t(params.full_y) * params.stride;
  const int64_t last = origin + int64_t(params.height - 1) * params.stride + params.width - 1;
  if (origin < 0 || uint64_t(last) >= film_words / params.pass_stride) {
    return p.fail("Cycles WebGPU render slice falls outside its film buffer");
  }

  size_t bytes_per_path = 0;
  const auto limits = device_webgpu_compute_limits(p.device);
  if (film_words > limits.storage_buffer_bytes / sizeof(uint32_t)) {
    return p.fail("Cycles WebGPU render slice film exceeds the device buffer limit");
  }
  const uint32_t device_capacity = path_capacity_for_limits(
      limits, adaptive, p.deferred_shadow, bytes_per_path, guided);
  if (device_capacity < min_path_capacity) {
    return p.fail("Cycles WebGPU device limits cannot fit the minimum path pool");
  }
  const uint32_t requested = uint32_t(
      std::max(int(min_path_capacity), p.queue->num_concurrent_states(bytes_per_path)));
  const uint32_t capacity = uint32_t(std::min<uint64_t>(
      {std::max<uint64_t>(pixels * 2, min_path_capacity), requested, device_capacity}));

  p.film_ready = false;
  p.batch_active = false;
  p.next_sample = 0;
  p.path_capacity = capacity;
  p.pixel_count = uint32_t(pixels);
  p.crop = {uint32_t(params.full_x),
            uint32_t(params.full_y),
            uint32_t(params.width),
            uint32_t(params.height)};
  p.film_config = {
      p.crop[0],
      p.crop[1],
      p.crop[2],
      p.crop[3],
      uint32_t(origin),
      uint32_t(params.stride),
      uint32_t(params.pass_stride),
      uint32_t(combined),
      sample_count == PASS_UNUSED ? 0xffffffffu : uint32_t(sample_count),
      adaptive_aux == PASS_UNUSED ? 0xffffffffu : uint32_t(adaptive_aux),
      p.sampling_pattern,
      std::bit_cast<uint32_t>(params.exposure),
  };
  p.film.alloc_to_device(film_words, true);
  if (!p.film.device_pointer) {
    return p.fail("Cycles WebGPU could not allocate the render slice film");
  }

  p.guided = guided;
  if (p.guided) {
    if (!p.prepare_guides()) {
      return false;
    }
    auto guide_config = std::array<uint32_t, 16>{};
    std::copy(denoising_config.begin(), denoising_config.end(), guide_config.begin());
    guide_config[0] = denoising_albedo == PASS_UNUSED ? 0xffffffffu : uint32_t(denoising_albedo);
    guide_config[1] = denoising_normal == PASS_UNUSED ? 0xffffffffu : uint32_t(denoising_normal);
    p.denoising.alloc_to_device(denoising_words_per_path * capacity);
    if (!p.denoising.device_pointer ||
        !upload(p.device, p.denoising_params, std::span<const uint32_t, 16>(guide_config)))
    {
      return p.fail("Cycles WebGPU could not allocate denoising guide state");
    }
    LOG_INFO << "Cycles WebGPU writing denoising guides";
  }
  else {
    p.denoising.free();
    p.denoising_params.free();
  }

  p.adaptive_sampling = adaptive;
  if (p.adaptive_sampling && !p.prepare_adaptive()) {
    return false;
  }
  if (!upload(p.device, p.camera, camera) ||
      !upload(p.device, p.params, std::span<const uint32_t>(p.film_config)))
  {
    return p.fail("Cycles WebGPU camera or film parameters could not be uploaded");
  }
  p.pixels.alloc_to_device(pixel_words_per_path * capacity);
  p.rays.alloc_to_device(ray_header_words + ray_words_per_path(adaptive) * capacity);
  p.hits.alloc_to_device(hit_words_per_path * capacity);
  if (p.deferred_shadow) {
    p.shadow_rays.alloc_to_device(shadow_words_per_path * capacity);
    if (!p.shadow_rays.device_pointer) {
      return p.fail("Cycles WebGPU could not allocate shadow rays");
    }
  }
  else {
    p.shadow_rays.free();
  }
  p.points.alloc_to_device(shading_words_per_path * capacity);
  /* Area light hits accumulate separately from surface and world shading. */
  p.radiance.alloc_to_device(radiance_words_per_path * capacity);
  p.materials.alloc_to_device(CYCLES_WEBGPU_MATERIAL_WORDS * capacity);
  if (!p.pixels.device_pointer || !p.rays.device_pointer || !p.hits.device_pointer ||
      !p.points.device_pointer || !p.radiance.device_pointer || !p.materials.device_pointer ||
      !p.path_state.alloc(capacity) || !p.wavefront_queue.alloc()) {
    return p.fail("Cycles WebGPU could not allocate path state and queue counters");
  }
  p.queue->zero_to_device(p.path_state.words());
  p.queue->zero_to_device(p.film);
  if (!p.wavefront_queue.set_capacity(*p.queue, p.rays, p.path_state.words()) ||
      !device_webgpu_error(p.device).empty())
  {
    return p.fail("Cycles WebGPU could not initialize the path pool and film");
  }
  p.film_ready = true;
  path_capacity = capacity;
  LOG_INFO << "Cycles WebGPU path pool: " << capacity << " slots";
  return true;
}

bool WebGPURenderPipeline::begin_batch(const int start_sample, const int samples_num)
{
  Impl &p = *impl_;
  if (!p.film_ready || p.batch_active || start_sample < 0 || samples_num <= 0 ||
      samples_num > std::numeric_limits<int>::max() - start_sample ||
      uint32_t(start_sample) != p.next_sample)
  {
    return p.fail("Cycles WebGPU received a non-consecutive render batch");
  }
  p.batch_active = true;
  p.batch_end_sample = uint32_t(start_sample + samples_num);
  p.admitted_paths = 0;
  p.active_paths = 0;
  p.queue_counts = {};
  return true;
}

bool WebGPURenderPipeline::enqueue_camera(const std::span<const KernelWorkTile> tiles)
{
  Impl &p = *impl_;
  if (!p.batch_active || tiles.empty() || tiles.size() > p.path_capacity) {
    return p.fail("Cycles WebGPU cannot admit empty work outside a render batch");
  }

  p.packed_tiles.clear();
  p.packed_tiles.resize(work_tile_header_words);
  uint32_t largest_tile = 0;
  uint32_t next_slot = 0;
  const uint32_t free_slots = p.path_capacity - uint32_t(p.active_paths);
  const uint32_t max_x = p.crop[0] + p.crop[2];
  const uint32_t max_y = p.crop[1] + p.crop[3];
  for (const KernelWorkTile &tile : tiles) {
    if (tile.path_index_offset != p.active_paths + int(next_slot) || tile.work_size <= 0 ||
        tile.x < p.crop[0] || tile.x >= max_x || tile.w > max_x - tile.x || tile.y < p.crop[1] ||
        tile.y >= max_y || tile.h > max_y - tile.y || tile.num_samples == 0 ||
        tile.start_sample < p.next_sample || tile.start_sample >= p.batch_end_sample ||
        tile.num_samples > p.batch_end_sample - tile.start_sample ||
        uint64_t(tile.w) * tile.h * tile.num_samples != uint32_t(tile.work_size) ||
        tile.sample_offset != 0 || uint32_t(tile.work_size) > free_slots - next_slot)
    {
      return p.fail("Cycles WebGPU received an invalid stock camera work tile");
    }
    const uint32_t size = uint32_t(tile.work_size);
    p.packed_tiles.insert(
        p.packed_tiles.end(),
        {tile.x, tile.y, tile.w, tile.h, tile.start_sample, tile.num_samples, next_slot, size});
    next_slot += size;
    largest_tile = std::max(largest_tile, size);
  }
  const uint64_t launch_size = uint64_t(largest_tile) * tiles.size();
  if (launch_size > std::numeric_limits<int>::max()) {
    return p.fail("Cycles WebGPU camera work tiles exceed the admitted path capacity");
  }

  p.packed_tiles[0] = uint32_t(tiles.size());
  p.packed_tiles[1] = largest_tile;
  p.packed_tiles[2] = next_slot;
  p.packed_tiles[3] = PATH_RAY_VISIBILITY_CAMERA;
  if (!upload(p.device, p.tile, std::span<const uint32_t>(p.packed_tiles))) {
    return p.fail("Cycles WebGPU camera work tile upload failed");
  }
  auto &paths = p.path_state.words();
  const auto camera = p.bind(p.tile, p.camera, p.pixels, p.rays, p.sampling);
  const auto init = p.guided ? p.bind(p.rays, paths, p.pixels, p.tile, p.denoising) :
                               p.bind(p.rays, paths, p.pixels, p.tile);
  const WebGPUDispatch init_state{p.guided ? guided_init_kernel : init_state_kernel, init, next_slot};
  if (p.adaptive_sampling) {
    const auto slots = p.bind(p.tile, p.camera, p.pixels, p.rays);
    const auto claim = p.bind(p.rays, p.pixels, p.tile, p.film, p.params);
    const WebGPUDispatch stages[] = {{adaptive_camera_slots_kernel, slots, uint32_t(launch_size)},
                                     {adaptive_admission_kernel, claim, next_slot},
                                     {adaptive_camera_rays_kernel, camera, next_slot},
                                     init_state};
    if (!p.wavefront_queue.enqueue_admission(int(p.path_capacity), p.rays, paths, stages)) {
      return p.fail("Cycles WebGPU adaptive camera dispatch failed");
    }
  }
  else {
    const WebGPUDispatch stages[] = {{camera_kernel, camera, uint32_t(launch_size)},
                                     init_state};
    if (!p.wavefront_queue.enqueue_admission(int(p.path_capacity), p.rays, paths, stages)) {
      return p.fail("Cycles WebGPU camera dispatch failed");
    }
  }
  p.admitted_paths += next_slot;
  return true;
}

bool WebGPURenderPipeline::enqueue_stage(const DeviceKernel stage)
{
  Impl &p = *impl_;
  if (!p.batch_active || p.active_paths == 0) {
    return p.fail("Cycles WebGPU has no active render paths for this stage");
  }
  if (stage != DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST &&
      stage != DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE &&
      stage != DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND)
  {
    return p.fail("Cycles WebGPU received an unsupported scheduler stage");
  }
  const uint32_t count = uint32_t(p.queue_counts.num_queued[stage]);
  if (count == 0) {
    return p.fail("Cycles WebGPU received a scheduler stage with no queued paths");
  }
  const int capacity = int(p.path_capacity);
  auto &paths = p.path_state.words();
  if (stage == DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST) {
    const auto closest = p.bind(p.scene.bvh, p.rays, p.hits, p.points);
    const auto classify = p.bind(paths, p.rays, p.hits);
    const WebGPUDispatch stages[] = {{bvh_kernel, closest, count},
                                     {WebGPUWavefrontQueue::classify_kernel, classify, count}};
    if (!p.wavefront_queue.enqueue_stage(stage, capacity, p.rays, paths, stages)) {
      return p.fail("Cycles WebGPU closest intersection dispatch failed");
    }
    return true;
  }

  const auto film = p.guided ? p.bind(p.film, paths, p.rays, p.params, p.denoising,
                                       p.denoising_params) :
                               p.bind(p.film, paths, p.rays, p.params);
  const DeviceKernel film_kernel = p.guided ? guided_accumulate_kernel :
                                   p.adaptive_sampling ? adaptive_accumulate_kernel :
                                                         accumulate_kernel;
  const WebGPUDispatch accumulate{film_kernel, film, count};
  if (stage == DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND &&
      (p.background_loaded || p.active_background_kernel == tree_background_kernel))
  {
    const auto background = p.guided ? p.bind(p.rays, p.hits, p.scene.light, p.sampling, paths,
                                                p.denoising) :
                                       p.bind(p.rays, p.hits, p.scene.light, p.sampling, paths);
    const DeviceKernel kernel = !p.guided ? p.active_background_kernel :
                                p.active_background_kernel == tree_background_kernel ?
                                    guided_tree_background_kernel :
                                    guided_background_kernel;
    const WebGPUDispatch stages[] = {{kernel, background, count}, accumulate};
    if (!p.wavefront_queue.enqueue_stage(stage, capacity, p.rays, paths, stages)) {
      return p.fail("Cycles WebGPU background dispatch failed");
    }
    return true;
  }

  const auto surface = p.bind(p.scene.bvh, p.scene.surface, p.rays, p.hits, p.points);
  const auto svm = p.bind(p.scene.svm, p.points, p.rays, p.materials, p.hits);
  const auto light = p.bind(p.deferred_shadow ? p.shadow_rays : p.scene.bvh,
                            p.rays,
                            p.hits,
                            p.points,
                            p.scene.light,
                            p.radiance,
                            p.materials,
                            paths);
  const auto shadow = p.bind(p.scene.bvh, p.rays, p.shadow_rays, p.radiance);
  const WebGPUDispatch intersect_shadow{shadow_kernel, shadow, count};
  const auto guides = p.bind(p.rays, p.points, p.materials, p.sampling, p.radiance, paths,
                             p.denoising, p.denoising_params);
  const WebGPUDispatch shade_guides{denoising_surface_kernel, guides, count};
  const auto update = p.bind(
      p.rays, p.hits, p.points, p.pixels, p.sampling, p.materials, p.radiance, paths);
  const WebGPUDispatch shading[] = {
      {surface_kernel, surface, count},
      {svm_kernel, svm, count},
      {p.light_kernel, light, count},
      {p.update_kernel, update, count},
      accumulate};
  const auto stages = std::span(shading);
  /* Visibility traversal and BSDF continuation are the expensive stages.
   * Submit them separately to bound the work in each browser command buffer. */
  if (!p.wavefront_queue.enqueue_stage(stage, capacity, p.rays, paths, stages.first(2)) ||
      !device_webgpu_enqueue(p.device, count, stages.subspan(2, 1)) ||
      (p.deferred_shadow &&
       !device_webgpu_enqueue(p.device, count, {&intersect_shadow, 1})) ||
      (p.guided && !device_webgpu_enqueue(p.device, count, {&shade_guides, 1})) ||
      !device_webgpu_enqueue(p.device, count, stages.subspan(3)))
  {
    return p.fail("Cycles WebGPU shading dispatch failed");
  }
  return true;
}

bool WebGPURenderPipeline::read_queue(IntegratorQueueCounter &counts, bool &samples_overlap)
{
  Impl &p = *impl_;
  if (!p.batch_active || !p.wavefront_queue.read(int(p.path_capacity),
                                                 p.rays,
                                                 p.path_state.words(),
                                                 p.pixels,
                                                 counts,
                                                 samples_overlap))
  {
    return p.fail("Cycles WebGPU could not read the integrator queue");
  }
  int active = 0;
  for (const int count : counts.num_queued) {
    if (count < 0 || count > int(p.path_capacity) - active) {
      return p.fail("Cycles WebGPU returned invalid integrator queue counters");
    }
    active += count;
  }
  p.queue_counts = counts;
  p.active_paths = active;
  return true;
}

bool WebGPURenderPipeline::finish_batch()
{
  Impl &p = *impl_;
  const uint64_t expected_paths = uint64_t(p.pixel_count) * (p.batch_end_sample - p.next_sample);
  if (!p.batch_active || p.active_paths != 0 || p.admitted_paths != expected_paths) {
    return p.fail("Cycles WebGPU could not retire all paths in the render batch");
  }
  p.next_sample = p.batch_end_sample;
  p.batch_active = false;
  return true;
}

bool WebGPURenderPipeline::adaptive_filter(const float threshold,
                                           const bool reset,
                                           int &active_pixels)
{
  Impl &p = *impl_;
  if (!p.film_ready || p.batch_active || !p.adaptive_sampling || !finite(threshold)) {
    return p.fail("Cycles WebGPU adaptive filtering needs a completed adaptive film");
  }

  std::array<uint32_t, 13> config;
  std::copy_n(p.film_config.begin(), 10, config.begin());
  config[10] = p.film_config[11]; /* Exposure. */
  config[11] = std::bit_cast<uint32_t>(threshold);
  config[12] = uint32_t(reset);
  if (!upload(p.device, p.adaptive_params, std::span<const uint32_t>(config))) {
    return p.fail("Cycles WebGPU could not upload adaptive filter parameters");
  }
  p.queue->zero_to_device(p.adaptive_active);
  const auto count_active = p.bind(p.film, p.adaptive_params, p.adaptive_active);
  const auto filter = p.bind(p.film, p.adaptive_params);
  /* Cycles counts active pixels before dilating the convergence mask. */
  const WebGPUDispatch stages[] = {
      {DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_CHECK, filter, p.pixel_count},
      {adaptive_count_kernel, count_active, p.pixel_count},
      {DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_FILTER_X, filter, p.crop[3]},
      {DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_FILTER_Y, filter, p.crop[2]}};
  uint32_t count = 0;
  if (!device_webgpu_enqueue(p.device, int(p.pixel_count), stages) ||
      !device_webgpu_read(
          p.device, p.adaptive_active, std::as_writable_bytes(std::span(&count, 1))) ||
      count > p.pixel_count)
  {
    return p.fail("Cycles WebGPU adaptive filtering failed");
  }
  active_pixels = int(count);
  return true;
}

bool WebGPURenderPipeline::publish_preview(const BufferParams &params,
                                          const float exposure,
                                          const int samples,
                                          const uint32_t ticket)
{
  Impl &p = *impl_;
  if (!p.film_ready || p.batch_active || samples <= 0 || !finite(exposure) ||
      params.window_x < 0 || params.window_y < 0 ||
      params.window_width <= 0 || params.window_height <= 0 ||
      params.window_x + params.window_width > int(p.crop[2]) ||
      params.window_y + params.window_height > int(p.crop[3]))
  {
    return false;
  }

  const uint32_t stride = p.film_config[5];
  const uint32_t origin = p.film_config[4] + uint32_t(params.window_x) +
                          uint32_t(params.window_y) * stride;
  return cycles_webgpu_publish_preview(
      p.device,
      p.film,
      {ticket, uint32_t(params.window_width), uint32_t(params.window_height), origin, stride,
       p.film_config[6], p.film_config[7], p.film_config[8], uint32_t(samples), exposure});
}

bool WebGPURenderPipeline::read_film(const std::span<float> film)
{
  Impl &p = *impl_;
  if (!p.film_ready || p.batch_active || film.size() != p.film.data_size ||
      !device_webgpu_read(p.device, p.film, std::as_writable_bytes(film)))
  {
    return p.fail("Cycles WebGPU could not read the Cycles render buffer");
  }
  return true;
}

bool WebGPURenderPipeline::append_resident_film(const WebGPUOIDNFilmTile &tile)
{
  Impl &p = *impl_;
  return p.film_ready && !p.batch_active &&
         cycles_webgpu_oidn_append_tile(p.device, p.film, tile);
}

std::string_view WebGPURenderPipeline::error_message() const
{
  return impl_->error;
}

CCL_NAMESPACE_END
