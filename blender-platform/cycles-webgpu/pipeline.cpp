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
#include "device/webgpu/path_state_layout.h"
#include "device/webgpu/runtime.h"
#include "device/webgpu/scene.h"
#include "device/webgpu/volume_layout.h"
#include "device/webgpu/volume_grid_layout.h"
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
constexpr std::string_view triangle_light_tree_header =
    "// Cycles Light Tree finite point\n// Cycles Light Tree triangle emitters\n";
constexpr std::string_view textured_light_tree_marker = "// Cycles Light Tree textured emitters\n";
constexpr std::string_view shadow_target_header = "// Cycles WebGPU shadow targets\n";
constexpr std::string_view standalone_closure_marker =
    "// Cycles WebGPU standalone surface closures v2\n";
constexpr std::string_view aov_marker = "// Cycles WebGPU AOV output\n";
constexpr std::string_view projected_query_images_marker =
    "// Cycles WebGPU projected query images\n";
constexpr std::string_view advanced_query_marker = "// Cycles WebGPU advanced procedural queries\n";
constexpr std::string_view transparent_marker = "// Cycles WebGPU constant transparent surface\n";
constexpr std::string_view varying_transparent_marker =
    "// Cycles WebGPU varying transparent surface\n";
constexpr std::string_view absorption_marker = "// Cycles WebGPU homogeneous absorption bounds\n";
constexpr std::string_view volume_queue_marker = "// Cycles WebGPU volume scatter classification\n";
constexpr std::string_view volume_guide_surface_marker =
    "// Cycles WebGPU volume guide surface hits\n";
constexpr std::string_view portal_marker = "// Cycles WebGPU ray portals\n";
constexpr std::string_view differential_marker = "// Cycles WebGPU ray differentials\n";
constexpr std::string_view hair_marker = "// Cycles WebGPU hair BSDFs\n";
constexpr std::string_view subsurface_marker = "// Cycles WebGPU Christensen-Burley subsurface\n";
/* The stock scheduler takes one eighth of the path pool as its work tile limit. */
constexpr uint32_t min_path_capacity = 8;
/* 132 words for each of the 64K paths, including the emission material record. */
constexpr size_t path_pool_budget_words = 33 * 1024 * 1024 / sizeof(uint32_t);

constexpr uint32_t pixel_words_per_path = 4;
constexpr uint32_t ray_header_words = 2;
constexpr uint32_t hit_words_per_path = 8;
constexpr uint32_t shading_words_per_path = 24;
constexpr uint32_t shading_storage_words_per_path(const bool queries, const bool hair)
{
  return shading_words_per_path + (queries ? CYCLES_WEBGPU_QUERY_HIT_WORDS : 0) +
         (hair ? CYCLES_WEBGPU_SURFACE_HAIR_WORDS : 0);
}
constexpr uint32_t radiance_words_per_path = 8;
/* Denoising feature throughput, albedo sum and world normal sum. */
constexpr uint32_t denoising_words_per_path = 12;
constexpr uint32_t work_tile_header_words = 4;
constexpr uint32_t work_tile_words = 8;
/* Origin, direction, self-intersection IDs, and source triangle geometry. */
constexpr uint32_t shadow_words_per_path = 16;

constexpr uint32_t shadow_storage_words_per_path(const bool textured)
{
  return shadow_words_per_path + (textured ? CYCLES_WEBGPU_TEXTURED_LIGHT_SAMPLE_WORDS : 0);
}

constexpr uint32_t ray_words_per_path(const bool adaptive,
                                       const bool differentials,
                                       const bool scattering)
{
  /* The shared active/free queue is followed by one film class word per adaptive path. */
  return differentials ? 16 : (scattering ? 15 : (adaptive ? 14 : 13));
}

struct PathBufferSize {
  size_t words_per_path;
  size_t header_words = 0;
};

constexpr auto path_buffer_sizes(const bool adaptive,
                                 const bool deferred_shadow,
                                 const bool guided,
                                 const bool textured,
                                 const size_t aov_words,
                                 const bool queries,
                                 const bool differentials,
                                 const bool hair,
                                 const bool absorption,
                                 const bool scattering)
{
  return std::array{PathBufferSize{pixel_words_per_path},
                    PathBufferSize{
                        ray_words_per_path(adaptive, differentials, scattering), ray_header_words},
                    PathBufferSize{hit_words_per_path},
                    PathBufferSize{deferred_shadow ? shadow_storage_words_per_path(textured) : 0},
                    PathBufferSize{shading_storage_words_per_path(queries, hair)},
                    PathBufferSize{radiance_words_per_path},
                    PathBufferSize{guided ? denoising_words_per_path : 0},
                    PathBufferSize{CYCLES_WEBGPU_MATERIAL_WORDS + aov_words},
                    PathBufferSize{CYCLES_WEBGPU_PATH_WORDS},
                    PathBufferSize{scattering ? CYCLES_WEBGPU_VOLUME_STATE_WORDS :
                                                (absorption ? 1u : 0u)},
                    /* A batch cannot admit more than one work tile per path slot. */
                    PathBufferSize{work_tile_words, work_tile_header_words}};
}

constexpr uint32_t path_capacity_for_limits(const WebGPUComputeLimits &limits,
                                            const bool adaptive,
                                            const bool deferred_shadow,
                                            size_t &bytes_per_path,
                                            const bool guided = false,
                                            const bool textured = false,
                                            const size_t aov_words = 0,
                                            const bool queries = false,
                                            const bool differentials = false,
                                            const bool hair = false,
                                            const bool absorption = false,
                                            const bool scattering = false)
{
  const size_t binding_words = limits.storage_buffer_bytes / sizeof(uint32_t);
  uint64_t capacity = std::min<uint64_t>(WebGPURenderPipeline::max_path_capacity,
                                         uint64_t(limits.workgroups_per_dimension) * workgroup_size);
  size_t words_per_path = 0;
  size_t header_words = 0;
  for (const auto &buffer :
       path_buffer_sizes(adaptive, deferred_shadow, guided, textured, aov_words, queries,
                         differentials, hair, absorption, scattering)) {
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
constexpr DeviceKernel transparent_shadow_kernel = DEVICE_KERNEL_INTEGRATOR_TERMINATED_PATHS_ARRAY;
constexpr DeviceKernel absorption_kernel = DEVICE_KERNEL_SHADER_EVAL_CURVE_SHADOW_TRANSPARENCY;
constexpr DeviceKernel absorption_shadow_kernel = DEVICE_KERNEL_FILM_CONVERT_DEPTH;
constexpr DeviceKernel volume_interval_kernel = DEVICE_KERNEL_FILM_CONVERT_SAMPLE_COUNT_HALF_RGBA;
constexpr DeviceKernel volume_scatter_kernel = DEVICE_KERNEL_FILM_CONVERT_FLOAT4_HALF_RGBA;
constexpr DeviceKernel volume_scatter_shadow_kernel = DEVICE_KERNEL_FILM_CONVERT_RGBE_HALF_RGBA;
constexpr DeviceKernel volume_extinction_shadow_kernel = DEVICE_KERNEL_FILM_CONVERT_CRYPTOMATTE;
constexpr DeviceKernel grid_volume_scatter_kernel = DEVICE_KERNEL_FILM_CONVERT_CRYPTOMATTE_HALF_RGBA;
constexpr DeviceKernel grid_volume_scatter_shadow_kernel =
    DEVICE_KERNEL_FILM_CONVERT_SHADOW_CATCHER_MATTE_WITH_SHADOW;
constexpr DeviceKernel grid_volume_extinction_shadow_kernel =
    DEVICE_KERNEL_FILTER_GUIDING_SET_FAKE_ALBEDO;
constexpr DeviceKernel guided_grid_volume_scatter_kernel = DEVICE_KERNEL_FILTER_COLOR_PREPROCESS;
constexpr DeviceKernel guided_grid_volume_scatter_shadow_kernel =
    DEVICE_KERNEL_FILTER_COLOR_POSTPROCESS;
constexpr DeviceKernel volume_light_kernel = DEVICE_KERNEL_FILM_CONVERT_VOLUME_MAJORANT;
constexpr DeviceKernel volume_background_kernel = DEVICE_KERNEL_FILM_CONVERT_VOLUME_MAJORANT_HALF_RGBA;
constexpr DeviceKernel varying_shadow_kernel = DEVICE_KERNEL_FILM_CONVERT_MIST_HALF_RGBA;
constexpr DeviceKernel surface_kernel = DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE;
constexpr DeviceKernel surface_query_kernel =
    DEVICE_KERNEL_INTEGRATOR_SHADOW_CATCHER_COUNT_POSSIBLE_SPLITS;
constexpr DeviceKernel background_kernel = DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND;
constexpr DeviceKernel tree_background_kernel = DEVICE_KERNEL_INTEGRATOR_SHADE_DEDICATED_LIGHT;
constexpr DeviceKernel tree_light_kernel = DEVICE_KERNEL_INTEGRATOR_INTERSECT_DEDICATED_LIGHT;
constexpr DeviceKernel tree_transmission_kernel = DEVICE_KERNEL_INTEGRATOR_INTERSECT_MNEE;
constexpr DeviceKernel tree_hair_kernel = DEVICE_KERNEL_FILM_CONVERT_COMBINED_HALF_RGBA;
constexpr DeviceKernel point_hair_kernel = DEVICE_KERNEL_FILM_CONVERT_LIGHT_PATH_HALF_RGBA;
constexpr DeviceKernel legacy_hair_kernel = DEVICE_KERNEL_FILM_CONVERT_MOTION_HALF_RGBA;
constexpr DeviceKernel init_state_kernel = DEVICE_KERNEL_INTEGRATOR_RESET;
constexpr DeviceKernel svm_core_kernel = DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE_RAYTRACE;
constexpr DeviceKernel svm_extended_kernel = DEVICE_KERNEL_SHADER_EVAL_VOLUME_DENSITY;
constexpr DeviceKernel svm_query_kernel = DEVICE_KERNEL_SHADER_EVAL_BACKGROUND;
constexpr DeviceKernel svm_query_textures_kernel = DEVICE_KERNEL_FILM_CONVERT_MIST;
constexpr DeviceKernel svm_query_advanced_kernel =
    DEVICE_KERNEL_FILM_CONVERT_SHADOW_CATCHER_MATTE_WITH_SHADOW_HALF_RGBA;
constexpr DeviceKernel sampled_emission_kernel = DEVICE_KERNEL_SHADER_EVAL_DISPLACE;
constexpr DeviceKernel wavefront_light_kernel = DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT_FORWARD;
constexpr DeviceKernel point_distribution_kernel = DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW;
constexpr DeviceKernel point_transmission_kernel = DEVICE_KERNEL_INTEGRATOR_SHADOW_PATH_MNEE_PENDING;
constexpr DeviceKernel path_update_kernel = DEVICE_KERNEL_INTEGRATOR_MEGAKERNEL;
constexpr DeviceKernel reflection_light_kernel = DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT_NEE;
constexpr DeviceKernel reflection_update_kernel = DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME;
constexpr DeviceKernel closure_update_kernel = DEVICE_KERNEL_FILM_CONVERT_LIGHT_PATH;
constexpr DeviceKernel closure_reflection_update_kernel = DEVICE_KERNEL_FILM_CONVERT_DEPTH_HALF_RGBA;
constexpr DeviceKernel accumulate_kernel = DEVICE_KERNEL_FILM_CONVERT_COMBINED;
constexpr DeviceKernel accumulate_aov_kernel = DEVICE_KERNEL_FILM_CONVERT_FLOAT;
constexpr DeviceKernel adaptive_accumulate_kernel = DEVICE_KERNEL_FILM_CONVERT_SAMPLE_COUNT;
constexpr DeviceKernel adaptive_count_kernel = DEVICE_KERNEL_PREFIX_SUM;
constexpr DeviceKernel guided_init_kernel = DEVICE_KERNEL_INTEGRATOR_SORT_BUCKET_PASS;
constexpr DeviceKernel subsurface_kernel = DEVICE_KERNEL_INTEGRATOR_INTERSECT_SUBSURFACE;
constexpr DeviceKernel denoising_surface_kernel = DEVICE_KERNEL_FILM_CONVERT_FLOAT3;
constexpr DeviceKernel closure_denoising_surface_kernel = DEVICE_KERNEL_FILM_CONVERT_FLOAT3_HALF_RGBA;
constexpr DeviceKernel guided_background_kernel = DEVICE_KERNEL_FILM_CONVERT_RGBE;
constexpr DeviceKernel guided_tree_background_kernel = DEVICE_KERNEL_FILM_CONVERT_FLOAT4;
constexpr DeviceKernel guided_volume_background_kernel = DEVICE_KERNEL_FILM_CONVERT_FLOAT_HALF_RGBA;
constexpr DeviceKernel guided_volume_scatter_kernel = DEVICE_KERNEL_FILM_CONVERT_SHADOW_CATCHER;
constexpr DeviceKernel guided_volume_scatter_shadow_kernel =
    DEVICE_KERNEL_FILM_CONVERT_SHADOW_CATCHER_HALF_RGBA;
constexpr DeviceKernel guided_accumulate_kernel = DEVICE_KERNEL_FILM_CONVERT_MOTION;

static_assert(DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST + 1 == 3);
static_assert(DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND + 1 == 9);
static_assert(DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE + 1 == CYCLES_WEBGPU_PATH_SHADE_SURFACE);
static_assert(subsurface_kernel + 1 == CYCLES_WEBGPU_PATH_INTERSECT_SUBSURFACE);
static_assert(PRNG_BOUNCE_NUM == CYCLES_WEBGPU_PRNG_BOUNCE_NUM);
static_assert(PATH_RAY_VISIBILITY_CAMERA == CYCLES_WEBGPU_CAMERA_VISIBILITY);
static_assert(PATH_RAY_SUBSURFACE_RANDOM_WALK == CYCLES_WEBGPU_PATH_SUBSURFACE_RANDOM_WALK);

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
                                                {4, storage, 32},
                                                {5, storage, CYCLES_WEBGPU_PATH_WORDS * sizeof(uint32_t)},
                                                {6, storage, CYCLES_WEBGPU_CAMERA_WORDS * sizeof(uint32_t)},
                                                {7, storage, CYCLES_WEBGPU_SURFACE_OFFSETS * sizeof(uint32_t)}};
constexpr auto svm_query_bindings = [] {
  auto bindings = std::to_array(svm_bindings);
  bindings[4].minimum_size = 40;
  bindings[6].minimum_size = CYCLES_WEBGPU_QUERY_CAMERA_OFFSET * sizeof(uint32_t);
  return bindings;
}();
constexpr WebGPUKernelBinding sampled_emission_bindings[] = {
    {0, read_only, 8},
    {1, read_only, shadow_storage_words_per_path(true) * sizeof(uint32_t)},
    {2, read_only, 52},
    {3, storage, 16},
    {4, read_only, CYCLES_WEBGPU_PATH_WORDS * sizeof(uint32_t)},
    {5, read_only, CYCLES_WEBGPU_CAMERA_WORDS * sizeof(uint32_t)},
    {6, read_only, CYCLES_WEBGPU_SURFACE_OFFSETS * sizeof(uint32_t)},
    {7, read_only, 40}};
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
constexpr auto transparent_shadow_bindings = std::array{
    shadow_bindings[0], shadow_bindings[1], shadow_bindings[2], shadow_bindings[3],
    WebGPUKernelBinding{4, storage, CYCLES_WEBGPU_SURFACE_TRANSPARENT_OFFSETS * sizeof(uint32_t)},
    WebGPUKernelBinding{5, storage, CYCLES_WEBGPU_PATH_WORDS * sizeof(uint32_t)}};
constexpr WebGPUKernelBinding absorption_bindings[] = {
    {0, storage, 40}, {1, storage, 52}, {2, storage, 32},
    {3, storage, CYCLES_WEBGPU_SURFACE_ABSORPTION_OFFSETS * sizeof(uint32_t)},
    {4, storage, CYCLES_WEBGPU_PATH_WORDS * sizeof(uint32_t)}, {5, storage, sizeof(uint32_t)}};
constexpr auto absorption_shadow_bindings = std::array{
    shadow_bindings[0], shadow_bindings[1], shadow_bindings[2], shadow_bindings[3],
    WebGPUKernelBinding{4, storage, CYCLES_WEBGPU_SURFACE_ABSORPTION_OFFSETS * sizeof(uint32_t)},
    WebGPUKernelBinding{5, storage, sizeof(uint32_t)}};
constexpr auto volume_extinction_shadow_bindings = std::array{
    absorption_shadow_bindings[0], absorption_shadow_bindings[1], absorption_shadow_bindings[2],
    absorption_shadow_bindings[3], absorption_shadow_bindings[4], absorption_shadow_bindings[5],
    WebGPUKernelBinding{6, storage, CYCLES_WEBGPU_PATH_WORDS * sizeof(uint32_t)}};
constexpr WebGPUKernelBinding volume_scatter_bindings[] = {
    {0, storage, 52},
    {1, storage, 32},
    {2, storage, CYCLES_WEBGPU_SURFACE_ABSORPTION_OFFSETS * sizeof(uint32_t)},
    {3, storage, CYCLES_WEBGPU_PATH_WORDS * sizeof(uint32_t)},
    {4, storage, CYCLES_WEBGPU_VOLUME_STATE_WORDS * sizeof(uint32_t)},
    {5, storage, 28},
    {6, storage, 48},
    {7, storage, shadow_words_per_path * sizeof(uint32_t)}};
constexpr auto volume_scatter_shadow_bindings = std::array{
    absorption_shadow_bindings[0],
    absorption_shadow_bindings[1],
    absorption_shadow_bindings[2],
    WebGPUKernelBinding{3, storage, CYCLES_WEBGPU_PATH_WORDS * sizeof(uint32_t)},
    absorption_shadow_bindings[4],
    WebGPUKernelBinding{5, storage, CYCLES_WEBGPU_VOLUME_STATE_WORDS * sizeof(uint32_t)}};
constexpr auto varying_shadow_bindings = std::array{
    transparent_shadow_bindings[0], transparent_shadow_bindings[1],
    transparent_shadow_bindings[2], transparent_shadow_bindings[3],
    transparent_shadow_bindings[4], transparent_shadow_bindings[5],
    WebGPUKernelBinding{6, storage, 8},
    WebGPUKernelBinding{7, storage, CYCLES_WEBGPU_CAMERA_WORDS * sizeof(uint32_t)}};
constexpr WebGPUKernelBinding path_update_bindings[] = {
    {0, storage, 52},
    {1, storage, 32},
    {2, storage, 32},
    {3, storage, 16},
    {4, storage, 28},
    {5, storage, material_bytes},
    {6, storage, 16},
    {7, storage, CYCLES_WEBGPU_PATH_WORDS * sizeof(uint32_t)}};
constexpr WebGPUKernelBinding subsurface_bindings[] = {
    {0, storage, 40}, {1, storage, 32}, {2, storage, 28}, {3, storage, 32},
    {4, storage, material_bytes}, {5, storage, 32}, {6, storage, 52},
    {7, storage, CYCLES_WEBGPU_PATH_WORDS * sizeof(uint32_t)}};
constexpr WebGPUKernelBinding accumulate_bindings[] = {
    {3, storage, 16},
    {5, read_only, CYCLES_WEBGPU_PATH_WORDS * sizeof(uint32_t)},
    {6, read_only, 4},
    {7, uniform, 48}};
constexpr auto aov_bindings = std::array{
    WebGPUKernelBinding{1, read_only, material_bytes},
    accumulate_bindings[0], accumulate_bindings[1], accumulate_bindings[2], accumulate_bindings[3],
    WebGPUKernelBinding{11, read_only, 16}};

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
constexpr auto guided_volume_scatter_shadow_bindings = std::array{
    volume_scatter_shadow_bindings[0], volume_scatter_shadow_bindings[1],
    volume_scatter_shadow_bindings[2], volume_scatter_shadow_bindings[3],
    volume_scatter_shadow_bindings[4], volume_scatter_shadow_bindings[5],
    WebGPUKernelBinding{6, storage, denoising_bytes},
    WebGPUKernelBinding{7, storage, denoising_params_bytes}};
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
  const uint64_t objects = tree[CYCLES_WEBGPU_TREE_OBJECT_COUNT];
  const uint64_t triangle_indices = tree[CYCLES_WEBGPU_TREE_TRIANGLE_INDEX_COUNT];
  const uint64_t triangles = tree[CYCLES_WEBGPU_TREE_TRIANGLE_COUNT];
  const uint64_t textured = tree[CYCLES_WEBGPU_TREE_TEXTURED_EMITTERS];
  const uint64_t shaders = tree[CYCLES_WEBGPU_TREE_SHADER_COUNT];
  const uint64_t triangle_lookups = tree[CYCLES_WEBGPU_TREE_TRIANGLE_TO_TREE_COUNT];
  const uint64_t map_w = tree[CYCLES_WEBGPU_TREE_WORLD_MAP_WIDTH];
  const uint64_t map_h = tree[CYCLES_WEBGPU_TREE_WORLD_MAP_HEIGHT];
  const uint64_t tex_w = tree[CYCLES_WEBGPU_TREE_WORLD_WIDTH];
  const uint64_t tex_h = tree[CYCLES_WEBGPU_TREE_WORLD_HEIGHT];
  const uint32_t world_lamp = tree[CYCLES_WEBGPU_TREE_WORLD_LAMP];
  const uint32_t world_emitter = tree[CYCLES_WEBGPU_TREE_WORLD_EMITTER];
  const uint32_t forward_lamp = tree[CYCLES_WEBGPU_TREE_FIRST_FORWARD_LAMP];
  const bool sampled_world = world_lamp != CYCLES_WEBGPU_TREE_NO_LAMP;
  if (nodes == 0 || emitters == 0 || emitters < lamps + triangles || textured > triangles ||
      (triangles ?
           !objects || !shaders || triangles > triangle_indices || triangles > triangle_lookups :
           emitters != lamps || objects || triangle_indices || shaders || triangle_lookups) ||
      (forward_lamp != CYCLES_WEBGPU_TREE_NO_LAMP && forward_lamp >= lamps) ||
      (sampled_world ? world_lamp >= lamps || world_emitter >= emitters || !map_w || !map_h :
                       world_emitter != CYCLES_WEBGPU_TREE_NO_LAMP || map_w || map_h) ||
      std::max({map_w, map_h, tex_w, tex_h}) > light.size())
  {
    return false;
  }
  const uint64_t lamp_offset = prefix + CYCLES_WEBGPU_TREE_DESCRIPTOR_WORDS;
  const uint64_t node_offset = lamp_offset + lamps * CYCLES_WEBGPU_TREE_LAMP_WORDS;
  const uint64_t emitter_offset = node_offset + nodes * CYCLES_WEBGPU_TREE_NODE_WORDS;
  const uint64_t object_offset = emitter_offset + emitters * CYCLES_WEBGPU_TREE_EMITTER_WORDS;
  const uint64_t triangle_index_offset = object_offset + objects * CYCLES_WEBGPU_TREE_OBJECT_WORDS;
  const uint64_t triangle_offset = triangle_index_offset + triangle_indices;
  const uint64_t shader_offset = triangle_offset + triangles * CYCLES_WEBGPU_TREE_TRIANGLE_WORDS;
  const uint64_t triangle_lookup_offset = shader_offset +
                                          shaders * CYCLES_WEBGPU_TREE_SHADER_WORDS;
  const uint64_t marginal_offset = triangle_lookup_offset + triangle_lookups;
  const uint64_t conditional_offset = marginal_offset + (sampled_world ? 2 * (map_h + 1) : 0);
  const uint64_t pixel_offset = conditional_offset + 2 * (map_w + 1) * map_h;
  const uint64_t sampling_offset = pixel_offset + 3 * tex_w * tex_h;
  return tree[CYCLES_WEBGPU_TREE_LAMPS] == lamp_offset &&
         tree[CYCLES_WEBGPU_TREE_NODES] == node_offset &&
         tree[CYCLES_WEBGPU_TREE_EMITTERS] == emitter_offset &&
         tree[CYCLES_WEBGPU_TREE_OBJECTS] == object_offset &&
         tree[CYCLES_WEBGPU_TREE_TRIANGLE_INDICES] == triangle_index_offset &&
         tree[CYCLES_WEBGPU_TREE_TRIANGLES] == triangle_offset &&
         tree[CYCLES_WEBGPU_TREE_SHADERS] == shader_offset &&
         tree[CYCLES_WEBGPU_TREE_TRIANGLE_TO_TREE] == triangle_lookup_offset &&
         tree[CYCLES_WEBGPU_TREE_WORLD_MARGINAL_CDF] == marginal_offset &&
         tree[CYCLES_WEBGPU_TREE_WORLD_CONDITIONAL_CDF] == conditional_offset &&
         tree[CYCLES_WEBGPU_TREE_WORLD_PIXELS] == pixel_offset &&
         tree[CYCLES_WEBGPU_TREE_SAMPLING] == sampling_offset && sampling_offset <= light.size() &&
         light.size() - sampling_offset >= CYCLES_WEBGPU_LIGHT_SAMPLING_WORDS;
}

enum class LightSampling { Legacy, PointDistribution, Tree };

bool get_aov_layout(const BufferParams &params, std::array<uint32_t, 4> &layout)
{
  for (const BufferPass &pass : params.passes) {
    if (pass.mode != PassMode::NOISY || (pass.type != PASS_AOV_COLOR && pass.type != PASS_AOV_VALUE))
    {
      continue;
    }
    const bool color = pass.type == PASS_AOV_COLOR;
    const uint32_t components = color ? 4u : 1u;
    if (pass.offset < 0 || pass.offset > params.pass_stride - int(components)) {
      return false;
    }
    const uint32_t index = color ? 0u : 2u;
    uint32_t &offset = layout[index], &count = layout[index + 1];
    if (count == 0) {
      offset = uint32_t(pass.offset);
    }
    if (uint32_t(pass.offset) != offset + count * components) {
      return false;
    }
    ++count;
  }
  return true;
}

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
  device_only_memory<uint32_t> absorption_state{&device, "WebGPU homogeneous absorption paths"};
  WebGPUWavefrontQueue wavefront_queue{device};
  device_only_memory<float> materials{&device, "WebGPU evaluated materials"};
  device_only_memory<float> radiance{&device, "WebGPU sampled radiance"};
  device_only_memory<float> denoising{&device, "WebGPU denoising paths"};
  device_only_memory<uint32_t> denoising_params{&device, "WebGPU denoising parameters"};
  device_only_memory<uint32_t> film{&device, "WebGPU Cycles film"};
  device_only_memory<uint32_t> aov_params{&device, "WebGPU AOV film layout"};
  device_only_memory<uint32_t> adaptive_params{&device, "WebGPU adaptive parameters"};
  device_only_memory<uint32_t> adaptive_active{&device, "WebGPU adaptive active pixel count"};
  std::vector<uint32_t> packed_tiles;
  std::vector<uint32_t> query_sampling;

  string error;
  bool initialized = false;
  bool scene_uploaded = false;
  DeviceKernel light_kernel = wavefront_light_kernel;
  DeviceKernel update_kernel = path_update_kernel;
  DeviceKernel guide_kernel = denoising_surface_kernel;
  DeviceKernel active_background_kernel = background_kernel;
  DeviceKernel active_svm_kernel = svm_core_kernel;
  bool svm_core_loaded = false;
  bool svm_extended_loaded = false;
  bool sampled_emission_loaded = false;
  bool aov_loaded = false;
  bool aov_active = false;
  bool aov_materials = false;
  bool standalone_closures = false;
  bool hair_materials = false;
  std::array<bool, DEVICE_KERNEL_NUM> hair_shaders{};
  std::array<bool, DEVICE_KERNEL_NUM> shader_loaded{};
  std::array<bool, DEVICE_KERNEL_NUM> standalone_shaders{};
  std::array<bool, DEVICE_KERNEL_NUM> aov_shaders{};
  bool deferred_shadow = false;
  bool reflection_deferred = false;
  bool full_deferred = false;
  bool shadow_loaded = false;
  DeviceKernel active_shadow_kernel = shadow_kernel;
  bool transparent_materials = false;
  bool absorption = false;
  bool scattering = false;
  bool grid_volume = false;
  bool volume_queue_supported = false;
  std::array<bool, DEVICE_KERNEL_NUM> absorption_shaders{};
  std::array<bool, DEVICE_KERNEL_NUM> volume_guide_surface_shaders{};
  std::array<bool, DEVICE_KERNEL_NUM> transparent_shaders{};
  std::array<bool, DEVICE_KERNEL_NUM> varying_transparent_shaders{};
  bool mixed_transparent_materials = false;
  std::array<bool, DEVICE_KERNEL_NUM> mixed_transparent_shaders{};
  bool portal_materials = false;
  std::array<bool, DEVICE_KERNEL_NUM> portal_shaders{};
  bool pixel_wireframe = false;
  std::array<bool, DEVICE_KERNEL_NUM> differential_shaders{};
  bool reflection_shading_loaded = false;
  bool point_distribution_loaded = false;
  bool point_transmission_loaded = false;
  bool shadow_targets_supported = false;
  std::array<bool, DEVICE_KERNEL_NUM> tree_triangles_supported{};
  std::array<bool, DEVICE_KERNEL_NUM> tree_textured_supported{};
  std::array<bool, DEVICE_KERNEL_NUM> tree_background_triangles_supported{};
  bool guided_tree_background_triangles_supported = false;
  bool triangle_lights = false;
  bool textured_lights = false;
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
  bool subsurface_loaded = false;
  bool subsurface_materials = false;
  std::array<bool, DEVICE_KERNEL_NUM> subsurface_shaders{};
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
    if (!device_webgpu_register_kernel(device, kernel, source, workgroup_size, bindings, entry)) {
      return false;
    }
    standalone_shaders[kernel] = source.find(standalone_closure_marker) != source.npos;
    aov_shaders[kernel] = source.starts_with(aov_marker);
    shader_loaded[kernel] = true;
    transparent_shaders[kernel] = source.find(transparent_marker) != source.npos;
    varying_transparent_shaders[kernel] = source.find(varying_transparent_marker) != source.npos;
    mixed_transparent_shaders[kernel] =
        source.find("// Cycles WebGPU mixed Transparent surface closures v2\n") != source.npos;
    absorption_shaders[kernel] = source.find(absorption_marker) != source.npos;
    volume_guide_surface_shaders[kernel] = source.find(volume_guide_surface_marker) != source.npos;
    portal_shaders[kernel] = source.find(portal_marker) != source.npos;
    differential_shaders[kernel] = source.find(differential_marker) != source.npos;
    hair_shaders[kernel] = source.find(hair_marker) != source.npos;
    subsurface_shaders[kernel] = source.find(subsurface_marker) != source.npos;
    return true;
  }

  bool prepare_svm(const bool requires_extended_svm, const bool queries, const bool advanced)
  {
    if (queries) {
      if (!shader_loaded[surface_query_kernel]) {
        const auto surface = load_shader("surface-query.wgsl");
        if (surface.empty() || !register_kernel(surface_query_kernel, surface, surface_bindings)) {
          return fail("Cycles WebGPU could not load triangle surface setup for SVM ray queries");
        }
      }
      active_svm_kernel = advanced                 ? svm_query_advanced_kernel :
                          requires_extended_svm    ? svm_query_textures_kernel :
                                                     svm_query_kernel;
      if (!shader_loaded[active_svm_kernel]) {
        const auto source = load_shader(advanced                ? "svm-query-advanced.wgsl" :
                                        requires_extended_svm   ? "svm-query-textures.wgsl" :
                                                                  "svm-query.wgsl");
        if (source.empty() ||
            (requires_extended_svm && source.find(projected_query_images_marker) == source.npos) ||
            (advanced && source.find(advanced_query_marker) == source.npos) ||
            !register_kernel(active_svm_kernel, source, svm_query_bindings))
        {
          return fail("Cycles WebGPU could not load SVM ray queries for the scene materials");
        }
      }
      return true;
    }
    if (!requires_extended_svm) {
      if (!svm_core_loaded) {
        const auto source = load_shader("svm-compact-core.wgsl");
        if (!source.empty()) {
          svm_core_loaded = register_kernel(svm_core_kernel, source, svm_bindings);
          if (!svm_core_loaded) {
            return false;
          }
        }
      }
      if (svm_core_loaded) {
        active_svm_kernel = svm_core_kernel;
        return true;
      }
    }

    if (!svm_extended_loaded) {
      const auto source = load_shader("svm-compact.wgsl");
      svm_extended_loaded = !source.empty() &&
                            register_kernel(svm_extended_kernel, source, svm_bindings);
    }
    active_svm_kernel = svm_extended_kernel;
    return svm_extended_loaded;
  }

  bool prepare_shadow()
  {
    if (!shadow_loaded) {
      const auto shadow = load_shader("intersect-shadow.wgsl");
      shadow_targets_supported = shadow.starts_with(shadow_target_header);
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

  const char *path_update_shader(const bool has_transmission) const
  {
    if (standalone_closures) {
      return has_transmission ? "path-update-closures.wgsl" :
                                "path-update-reflection-closures.wgsl";
    }
    return has_transmission ? "path-update.wgsl" : "path-update-reflection.wgsl";
  }

  bool prepare_path_update(const bool has_transmission)
  {
    update_kernel = standalone_closures ?
                        (has_transmission ? closure_update_kernel : closure_reflection_update_kernel) :
                        (has_transmission ? path_update_kernel : reflection_update_kernel);
    if (!shader_loaded[update_kernel]) {
      const auto update = load_shader(path_update_shader(has_transmission));
      if (update.empty() || !register_kernel(update_kernel, update, path_update_bindings)) {
        return fail("Cycles WebGPU could not compile the path update shader");
      }
    }
    return true;
  }

  bool prepare_light_tree(const bool has_transmission)
  {
    const auto kernel = scattering ? volume_light_kernel :
                        hair_materials ?
                            tree_hair_kernel :
                            (has_transmission ? tree_transmission_kernel : tree_light_kernel);
    bool &triangles_supported = tree_triangles_supported[kernel];
    bool &textured_supported = tree_textured_supported[kernel];
    if (!shader_loaded[kernel]) {
      const char *shader_name = scattering ? "wavefront-light-deferred-reflection-tree-volume.wgsl" :
                                 hair_materials ? "wavefront-light-deferred-tree-hair.wgsl" :
                                 has_transmission ? "wavefront-light-deferred-tree.wgsl" :
                                                    "wavefront-light-deferred-reflection-tree.wgsl";
      const auto light = load_shader(shader_name);
      if (light.empty()) {
        return fail("Cycles WebGPU needs the Light Tree shader set");
      }
      triangles_supported = light.starts_with(triangle_light_tree_header);
      textured_supported = triangles_supported &&
                           std::string_view(light).substr(triangle_light_tree_header.size())
                               .starts_with(textured_light_tree_marker);
      if (triangle_lights && !triangles_supported) {
        return fail("Cycles WebGPU needs rebuilt Light Tree shaders for triangle emitters");
      }
      if (textured_lights && !textured_supported) {
        return fail("Cycles WebGPU needs rebuilt Light Tree shaders for textured emitters");
      }
      LOG_INFO << "Cycles WebGPU compiling the stock Light Tree and world sampling shaders";
      if (!prepare_shadow()) {
        return fail("Cycles WebGPU could not compile the Light Tree shaders");
      }
      if (triangle_lights && !shadow_targets_supported) {
        return fail("Cycles WebGPU needs a rebuilt shadow shader for triangle emitters");
      }
      if (!register_kernel(kernel, light, deferred_light_bindings)) {
        return fail("Cycles WebGPU could not compile the Light Tree shaders");
      }
    }
    if (triangle_lights && (!triangles_supported || !shadow_targets_supported)) {
      return fail(
          "Cycles WebGPU needs rebuilt Light Tree and shadow shaders for triangle emitters");
    }
    if (textured_lights && !textured_supported) {
      return fail("Cycles WebGPU needs rebuilt Light Tree shaders for textured emitters");
    }
    if (textured_lights && !sampled_emission_loaded) {
      const auto emission = load_shader("sampled-emission.wgsl");
      sampled_emission_loaded = !emission.empty() &&
                                register_kernel(sampled_emission_kernel,
                                                emission,
                                                sampled_emission_bindings);
      if (!sampled_emission_loaded) {
        return fail("Cycles WebGPU could not compile the sampled mesh emission shader");
      }
    }
    const DeviceKernel background_stage = scattering ? volume_background_kernel :
                                                       tree_background_kernel;
    bool &background_triangles_supported = tree_background_triangles_supported[background_stage];
    if (!shader_loaded[background_stage]) {
      const auto background = load_shader(scattering ? "shade-background-tree-volume.wgsl" :
                                                       "shade-background-tree.wgsl");
      background_triangles_supported = background.starts_with(triangle_light_tree_header);
      if (triangle_lights && !background_triangles_supported) {
        return fail(
            "Cycles WebGPU needs a rebuilt Light Tree background shader for triangle emitters");
      }
      if (background.empty() ||
          !register_kernel(background_stage, background, background_bindings))
      {
        return fail("Cycles WebGPU needs the Light Tree background shader");
      }
    }
    if (triangle_lights && !background_triangles_supported) {
      return fail(
          "Cycles WebGPU needs a rebuilt Light Tree background shader for triangle emitters");
    }
    light_kernel = kernel;
    active_background_kernel = background_stage;
    deferred_shadow = true;
    return prepare_path_update(has_transmission);
  }

  bool prepare_shading(const bool has_transmission, const LightSampling sampling)
  {
    active_background_kernel = background_kernel;
    if (sampling == LightSampling::Tree) {
      return prepare_light_tree(has_transmission);
    }
    if (hair_materials) {
      const bool distribution = sampling == LightSampling::PointDistribution;
      light_kernel = distribution ? point_hair_kernel : legacy_hair_kernel;
      if (!shader_loaded[light_kernel]) {
        const auto light = load_shader(distribution ?
                                           "wavefront-light-deferred-multi-point-hair.wgsl" :
                                           "wavefront-light-deferred-hair.wgsl");
        if (light.empty() || !prepare_shadow() ||
            !register_kernel(light_kernel, light, deferred_light_bindings))
        {
          return fail("Cycles WebGPU could not compile the Hair direct light shader");
        }
      }
      deferred_shadow = true;
      return prepare_path_update(has_transmission);
    }
    if (sampling == LightSampling::PointDistribution) {
      const auto kernel = has_transmission ? point_transmission_kernel : point_distribution_kernel;
      bool &loaded = has_transmission ? point_transmission_loaded : point_distribution_loaded;
      if (!loaded) {
        const auto light = load_shader(has_transmission ? "wavefront-light-deferred-multi-point.wgsl" :
                                                          "wavefront-light-deferred-multi-point-reflection.wgsl");
        if (light.empty()) {
          return fail("Cycles WebGPU needs the point-light distribution shader set");
        }
        LOG_INFO << "Cycles WebGPU compiling the stock point-light distribution shader";
        if (!prepare_shadow() || !register_kernel(kernel, light, deferred_light_bindings))
        {
          return fail("Cycles WebGPU could not compile the point-light distribution shader");
        }
        loaded = true;
      }
      light_kernel = kernel;
      deferred_shadow = true;
      return prepare_path_update(has_transmission);
    }
    if (!has_transmission) {
      if (!reflection_shading_loaded) {
        const auto update = load_shader(path_update_shader(false));
        if (!update.empty()) {
          LOG_INFO << "Cycles WebGPU compiling reflection-only wavefront shaders";
          if (!prepare_light(reflection_light_kernel, true, reflection_deferred))
          {
            return fail("Cycles WebGPU could not compile the reflection-only wavefront shaders");
          }
          reflection_shading_loaded = true;
        }
      }
      if (reflection_shading_loaded) {
        light_kernel = reflection_light_kernel;
        deferred_shadow = reflection_deferred;
        return prepare_path_update(false);
      }
    }

    if (!shading_loaded) {
      LOG_INFO << "Cycles WebGPU compiling transmission-capable wavefront shaders";
      if (!prepare_light(wavefront_light_kernel, false, full_deferred))
      {
        return fail("Cycles WebGPU could not compile the transmission-capable wavefront shaders");
      }
      shading_loaded = true;
    }
    light_kernel = wavefront_light_kernel;
    deferred_shadow = full_deferred;
    return prepare_path_update(true);
  }

  DeviceKernel volume_scatter_stage(const bool guides) const
  {
    if (grid_volume) {
      return guides ? guided_grid_volume_scatter_kernel : grid_volume_scatter_kernel;
    }
    return guides ? guided_volume_scatter_kernel : volume_scatter_kernel;
  }

  DeviceKernel volume_scatter_shadow_stage(const bool guides) const
  {
    if (grid_volume) {
      return guides ? guided_grid_volume_scatter_shadow_kernel : grid_volume_scatter_shadow_kernel;
    }
    return guides ? guided_volume_scatter_shadow_kernel : volume_scatter_shadow_kernel;
  }

  DeviceKernel guided_background_stage() const
  {
    return active_background_kernel == volume_background_kernel ? guided_volume_background_kernel :
           active_background_kernel == tree_background_kernel ? guided_tree_background_kernel :
                                                                 guided_background_kernel;
  }

  bool prepare_guides()
  {
    guide_kernel = standalone_closures ? closure_denoising_surface_kernel : denoising_surface_kernel;
    if (!shader_loaded[guide_kernel]) {
      const auto surface = load_shader(standalone_closures ? "denoise-surface-closures.wgsl" :
                                                           "denoise-surface.wgsl");
      if (surface.empty() || !register_kernel(guide_kernel, surface, denoising_surface_bindings)) {
        return fail("Cycles WebGPU could not compile the denoising surface shader");
      }
    }
    if (absorption &&
        (!volume_guide_surface_shaders[surface_kernel] || !volume_guide_surface_shaders[guide_kernel]))
    {
      return fail("Cycles WebGPU needs rebuilt surface shaders for volume denoising guides");
    }
    if (scattering) {
      const DeviceKernel scatter_stage = volume_scatter_stage(true);
      if (!shader_loaded[scatter_stage]) {
        const auto scatter = load_shader(grid_volume ? "volume-scatter-grid-guides.wgsl" :
                                                      "volume-scatter-guides.wgsl");
        if (scatter.empty() ||
            !register_kernel(scatter_stage, scatter, volume_scatter_bindings)) {
          return fail("Cycles WebGPU needs the volume scattering denoising shader");
        }
      }
      const DeviceKernel shadow_stage = volume_scatter_shadow_stage(true);
      if (!shader_loaded[shadow_stage]) {
        const auto shadow = load_shader(grid_volume ? "volume-scatter-shadow-grid-guides.wgsl" :
                                                     "volume-scatter-shadow-guides.wgsl");
        if (shadow.empty() ||
            !register_kernel(shadow_stage, shadow, guided_volume_scatter_shadow_bindings))
        {
          return fail("Cycles WebGPU needs the volume denoising feature shader");
        }
      }
    }
    if (!guided_loaded) {
      const auto init = load_shader("path-state.wgsl");
      const auto film_shader = load_shader("film.wgsl");
      if (init.empty() || film_shader.empty()) {
        return fail("Cycles WebGPU needs the denoising guide shader set");
      }
      if (!register_kernel(guided_init_kernel, init, guided_init_bindings, "initialize_guided") ||
          !register_kernel(guided_accumulate_kernel,
                           film_shader,
                           guided_accumulate_bindings,
                           "retire_guided_render_buffer"))
      {
        return fail("Cycles WebGPU could not compile the denoising guide shaders");
      }
      guided_loaded = true;
    }

    if (active_background_kernel == volume_background_kernel) {
      if (!shader_loaded[guided_volume_background_kernel]) {
        const auto background = load_shader("shade-background-tree-guides-volume.wgsl");
        if (background.empty() ||
            !register_kernel(guided_volume_background_kernel, background, guided_background_bindings))
        {
          return fail("Cycles WebGPU needs the volume denoising background shader");
        }
      }
    }
    else if (active_background_kernel == tree_background_kernel) {
      if (!guided_tree_background_loaded) {
        const auto background = load_shader("shade-background-tree-guides.wgsl");
        guided_tree_background_triangles_supported = background.starts_with(
            triangle_light_tree_header);
        if (triangle_lights && !guided_tree_background_triangles_supported) {
          return fail(
              "Cycles WebGPU needs a rebuilt Light Tree guide shader for triangle emitters");
        }
        if (background.empty() ||
            !register_kernel(guided_tree_background_kernel, background, guided_background_bindings))
        {
          return fail("Cycles WebGPU needs the Light Tree denoising background shader");
        }
        guided_tree_background_loaded = true;
      }
      if (triangle_lights && !guided_tree_background_triangles_supported) {
        return fail("Cycles WebGPU needs a rebuilt Light Tree guide shader for triangle emitters");
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
      shaders.wavefront_queue.empty())
  {
    return p.fail("Cycles WebGPU render pipeline needs the stock scheduler shader set");
  }

  if (!p.register_kernel(camera_kernel, shaders.camera, camera_bindings) ||
      !p.register_kernel(bvh_kernel, shaders.bvh2, bvh_bindings) ||
      !p.register_kernel(surface_kernel, shaders.surface, surface_bindings) ||
      !p.register_kernel(init_state_kernel, shaders.state_init, init_state_bindings) ||
      !p.register_kernel(
          accumulate_kernel, shaders.film, accumulate_bindings, "accumulate_render_buffer") ||
      !p.wavefront_queue.initialize(shaders.wavefront_queue))
  {
    return p.fail("Cycles WebGPU render pipeline failed to compile a kernel");
  }
  p.load_shader = shaders.load_shader;
  p.volume_queue_supported = shaders.wavefront_queue.find(volume_queue_marker) !=
                             shaders.wavefront_queue.npos;
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
                                        std::span<const uint32_t> sampling,
                                        bool has_subsurface,
                                        bool advanced_query_svm)
{
  Impl &p = *impl_;
  p.error.clear();
  if (!p.initialized) {
    return p.fail("Cycles WebGPU render pipeline needs to be initialized");
  }
  const size_t binding_words = device_webgpu_compute_limits(p.device).storage_buffer_words();
  if (std::max({scene.size(), surface.size(), light.size(), svm.size(), sampling.size()}) >
      binding_words)
  {
    return p.fail("Cycles WebGPU scene buffer exceeds the adapter's storage binding limit");
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
  constexpr size_t surface_offsets = CYCLES_WEBGPU_SURFACE_OFFSETS;
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
  p.transparent_materials = CYCLES_WEBGPU_SURFACE_HAS_TRANSPARENCY(surface.front());
  p.active_shadow_kernel = shadow_kernel;
  p.absorption = surface.front() == CYCLES_WEBGPU_SURFACE_ABSORPTION_OFFSETS;
  p.scattering = false;
  p.grid_volume = false;
  if (p.absorption) {
    if (surface.size() < CYCLES_WEBGPU_SURFACE_ABSORPTION_OFFSETS) {
      return p.fail("Cycles WebGPU needs homogeneous absorption metadata");
    }
    const size_t offset = surface[CYCLES_WEBGPU_SURFACE_ABSORPTION_OFFSET];
    if (offset < CYCLES_WEBGPU_SURFACE_ABSORPTION_OFFSETS || offset > surface.size())
    {
      return p.fail("Cycles WebGPU has invalid homogeneous absorption metadata");
    }
    const size_t words = surface.size() - offset;
    const size_t count = surface[CYCLES_WEBGPU_SURFACE_VOLUME_COUNT];
    p.scattering = count != 0;
    if ((!p.scattering && words != CYCLES_WEBGPU_ABSORPTION_WORDS) ||
        (p.scattering && (!light_tree || !p.volume_queue_supported ||
                          light_prefix == glass_lighting_words ||
                          count > CYCLES_WEBGPU_MAX_VOLUME_MEDIA ||
                          count * CYCLES_WEBGPU_VOLUME_SCATTER_WORDS > words)))
    {
      return p.fail("Cycles WebGPU needs compatible volume data and Light Tree sampling");
    }
    constexpr size_t grid_header_words = CYCLES_WEBGPU_GRID_ATTRIBUTES +
                                         CYCLES_WEBGPU_GRID_ATTRIBUTE_COUNT *
                                             CYCLES_WEBGPU_GRID_ATTRIBUTE_WORDS;
    const size_t end = offset + count * CYCLES_WEBGPU_VOLUME_SCATTER_WORDS;
    for (size_t record = offset; record < end; record += CYCLES_WEBGPU_VOLUME_SCATTER_WORDS) {
      const size_t grid = surface[record + CYCLES_WEBGPU_VOLUME_GRID_OFFSET];
      if (grid) {
        if (grid < end || grid > surface.size() || grid_header_words > surface.size() - grid) {
          return p.fail("Cycles WebGPU has invalid volume grid metadata");
        }
        p.grid_volume = true;
      }
    }
    p.active_shadow_kernel = p.grid_volume ? grid_volume_extinction_shadow_kernel :
                              p.scattering ? volume_extinction_shadow_kernel :
                                             absorption_shadow_kernel;
  }
  if (p.transparent_materials) {
    if (surface.size() < CYCLES_WEBGPU_SURFACE_TRANSPARENT_OFFSETS) {
      return p.fail("Cycles WebGPU needs transparent shadow metadata");
    }
    const size_t offset = surface[CYCLES_WEBGPU_SURFACE_TRANSPARENT_OFFSET];
    const size_t shader_count = svm[1] & CYCLES_WEBGPU_SVM_SHADER_COUNT;
    if (offset < surface[CYCLES_WEBGPU_SURFACE_SHADER_PASS_OFFSET] ||
        offset > surface.size() ||
        surface.size() - offset != CYCLES_WEBGPU_TRANSPARENT_SHADOW_HEADER_WORDS + 3 * shader_count)
    {
      return p.fail("Cycles WebGPU has invalid transparent shadow metadata");
    }
    const uint32_t shadow_flags = surface[offset + 1];
    p.active_shadow_kernel = (shadow_flags & CYCLES_WEBGPU_TRANSPARENT_SHADOW_VARYING) ?
                                 varying_shadow_kernel : transparent_shadow_kernel;
  }
  p.aov_materials = (svm[1] & CYCLES_WEBGPU_SVM_AOV) != 0;
  p.portal_materials = (svm[1] & CYCLES_WEBGPU_SVM_RAY_PORTAL) != 0;
  p.pixel_wireframe = (svm[1] & CYCLES_WEBGPU_SVM_PIXEL_WIREFRAME) != 0;
  p.standalone_closures = (svm[1] & CYCLES_WEBGPU_SVM_STANDALONE) != 0;
  p.hair_materials = (svm[1] & CYCLES_WEBGPU_SVM_HAIR) != 0;
  p.mixed_transparent_materials = (svm[1] & CYCLES_WEBGPU_SVM_MIXED_TRANSPARENT) != 0;
  p.subsurface_materials = has_subsurface;
  const bool queries = (svm[1] & CYCLES_WEBGPU_SVM_QUERIES) != 0;
  if (p.hair_materials != CYCLES_WEBGPU_SURFACE_HAS_HAIR(surface.front())) {
    return p.fail("Cycles WebGPU has invalid triangle tangent metadata for Hair");
  }
  if (!p.prepare_svm((svm[1] & CYCLES_WEBGPU_SVM_EXTENDED) != 0, queries, advanced_query_svm)) {
    return p.fail("Cycles WebGPU could not load the SVM material shader");
  }
  p.query_sampling.clear();
  if (queries) {
    /* The camera packing supplies the AO radius and the length up through
     * scrambling_distance. GGX and other path settings stay in p.sampling. */
    constexpr size_t trailer = 2;
    const size_t prefix = sampling.back();
    if (sampling.size() < trailer || prefix < 10 || prefix > sampling.size() - trailer ||
        CYCLES_WEBGPU_QUERY_CAMERA_OFFSET + prefix + 1 > binding_words)
    {
      return p.fail("Cycles WebGPU needs camera sampling data for SVM ray queries");
    }
    p.query_sampling.assign(sampling.begin(), sampling.begin() + prefix);
    p.query_sampling.push_back(sampling[sampling.size() - trailer]);
  }
  if (p.aov_materials && !p.aov_shaders[p.active_svm_kernel]) {
    return p.fail("Cycles WebGPU needs rebuilt SVM shaders for AOV output");
  }
  const LightSampling mode = light_tree ? LightSampling::Tree :
                               point_distribution ? LightSampling::PointDistribution :
                                                    LightSampling::Legacy;
  p.triangle_lights = light_tree && light[light_prefix + CYCLES_WEBGPU_TREE_TRIANGLE_COUNT] != 0;
  p.textured_lights = light_tree &&
                      light[light_prefix + CYCLES_WEBGPU_TREE_TEXTURED_EMITTERS] != 0;
  if (has_subsurface && !p.subsurface_loaded) {
    const auto shader = p.load_shader("subsurface-compact.wgsl");
    p.subsurface_loaded = !shader.empty() &&
                          p.register_kernel(subsurface_kernel, shader, subsurface_bindings);
    if (!p.subsurface_loaded) {
      return p.fail("Cycles WebGPU could not load the subsurface scattering kernel");
    }
  }
  if (!p.prepare_shading(light_prefix == glass_lighting_words, mode)) {
    return false;
  }
  if (p.subsurface_materials &&
      (!p.subsurface_shaders[p.active_svm_kernel] || !p.subsurface_shaders[p.update_kernel] ||
       !p.subsurface_shaders[subsurface_kernel]))
  {
    return p.fail("Cycles WebGPU needs rebuilt shaders for subsurface scattering");
  }
  if (p.hair_materials &&
      (!p.hair_shaders[p.active_svm_kernel] || !p.hair_shaders[p.light_kernel] ||
       !p.hair_shaders[p.update_kernel] ||
       !p.hair_shaders[queries ? surface_query_kernel : surface_kernel]))
  {
    return p.fail("Cycles WebGPU needs rebuilt SVM, surface, path and light shaders for Hair");
  }
  if (p.absorption) {
    if (!p.deferred_shadow || !p.absorption_shaders[p.update_kernel] || p.pixel_wireframe ||
        p.transparent_materials || p.portal_materials || p.hair_materials || p.aov_materials)
    {
      return p.fail("Cycles WebGPU needs compatible path and shadow shaders for volumes");
    }
    const auto load_volume_stage = [&](const DeviceKernel kernel,
                                       const char *name,
                                       const std::span<const WebGPUKernelBinding> bindings) {
      if (p.shader_loaded[kernel]) {
        return true;
      }
      const auto source = p.load_shader(name);
      return !source.empty() && p.register_kernel(kernel, source, bindings);
    };
    const bool prepared = p.scattering ?
        load_volume_stage(volume_interval_kernel, "volume-interval.wgsl", absorption_bindings) &&
        load_volume_stage(p.volume_scatter_stage(false),
                          p.grid_volume ? "volume-scatter-grid.wgsl" : "volume-scatter.wgsl",
                          volume_scatter_bindings) &&
        load_volume_stage(p.volume_scatter_shadow_stage(false),
                          p.grid_volume ? "volume-scatter-shadow-grid.wgsl" :
                                          "volume-scatter-shadow.wgsl",
                          volume_scatter_shadow_bindings) &&
        load_volume_stage(p.active_shadow_kernel,
                          p.grid_volume ? "volume-extinction-shadow-grid.wgsl" :
                                          "volume-extinction-shadow.wgsl",
                          volume_extinction_shadow_bindings) :
        load_volume_stage(absorption_kernel, "absorption-compact.wgsl", absorption_bindings) &&
        load_volume_stage(absorption_shadow_kernel, "absorption-shadow.wgsl", absorption_shadow_bindings);
    if (!prepared) {
      return p.fail("Cycles WebGPU could not compile volume integration and shadows");
    }
  }
  if (p.pixel_wireframe &&
      (p.textured_lights || !p.differential_shaders[camera_kernel] ||
       !p.differential_shaders[p.active_svm_kernel] || !p.differential_shaders[p.update_kernel]))
  {
    return p.fail("Cycles WebGPU needs ray differentials in camera, path and sampled emitter shaders for pixel Wireframe");
  }
  if (p.transparent_materials) {
    if (!p.deferred_shadow || !p.transparent_shaders[p.active_svm_kernel] ||
        !p.transparent_shaders[p.update_kernel] ||
        !p.transparent_shaders[p.active_background_kernel])
    {
      return p.fail("Cycles WebGPU needs rebuilt SVM, path, and background shaders for transparency");
    }
    const bool varying = p.active_shadow_kernel == varying_shadow_kernel;
    if (varying && (!p.varying_transparent_shaders[p.active_svm_kernel] ||
                    !p.varying_transparent_shaders[p.update_kernel]))
    {
      return p.fail("Cycles WebGPU needs rebuilt SVM and path shaders for varying transparency");
    }
    if (!p.shader_loaded[p.active_shadow_kernel]) {
      const auto source = p.load_shader(varying ? "intersect-shadow-transparent-materials.wgsl" :
                                                 "intersect-shadow-transparent.wgsl");
      const auto bindings = varying ? std::span<const WebGPUKernelBinding>(varying_shadow_bindings) :
                                      std::span<const WebGPUKernelBinding>(transparent_shadow_bindings);
      if (source.empty() || !p.register_kernel(p.active_shadow_kernel, source, bindings)) {
        return p.fail("Cycles WebGPU could not compile the transparent shadow shader");
      }
    }
  }
  if (p.portal_materials &&
      (!p.portal_shaders[p.active_svm_kernel] || !p.portal_shaders[p.update_kernel] ||
       !p.portal_shaders[p.light_kernel] || !p.portal_shaders[p.active_background_kernel] ||
       !p.portal_shaders[p.active_shadow_kernel]))
  {
    return p.fail("Cycles WebGPU needs rebuilt shaders for Ray Portal");
  }
  if (p.hair_materials && p.active_shadow_kernel == varying_shadow_kernel &&
      !p.hair_shaders[varying_shadow_kernel])
  {
    return p.fail("Cycles WebGPU needs a rebuilt varying shadow shader for Hair");
  }
  if (p.mixed_transparent_materials &&
      (!p.mixed_transparent_shaders[p.active_svm_kernel] ||
       !p.mixed_transparent_shaders[p.update_kernel] ||
       !p.mixed_transparent_shaders[p.light_kernel] ||
       (p.textured_lights && !p.mixed_transparent_shaders[sampled_emission_kernel]) ||
       (p.active_shadow_kernel == varying_shadow_kernel &&
        !p.mixed_transparent_shaders[varying_shadow_kernel])))
  {
    return p.fail("Cycles WebGPU needs rebuilt shaders for mixed Transparent surface closures");
  }
  if (p.standalone_closures &&
      (!p.standalone_shaders[p.active_svm_kernel] || !p.standalone_shaders[p.light_kernel] ||
       !p.standalone_shaders[p.update_kernel] ||
       (p.textured_lights && !p.standalone_shaders[sampled_emission_kernel])))
  {
    return p.fail("Cycles WebGPU needs rebuilt shaders for standalone surface closures");
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

bool WebGPURenderPipeline::reset(std::span<const float, CYCLES_WEBGPU_CAMERA_WORDS> camera,
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
  std::array<uint32_t, 4> aov_layout{};
  if (!get_aov_layout(params, aov_layout)) {
    return p.fail("Cycles WebGPU needs valid contiguous film AOV passes");
  }
  const uint32_t aov_slots = p.aov_materials ? aov_layout[1] + aov_layout[3] : 0;
  const size_t aov_words = 4 * size_t(aov_slots);
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
      limits, adaptive, p.deferred_shadow, bytes_per_path, guided, p.textured_lights, aov_words,
      !p.query_sampling.empty(), p.pixel_wireframe, p.hair_materials, p.absorption, p.scattering);
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
    if (p.transparent_materials &&
        (!p.transparent_shaders[p.guide_kernel] ||
         !p.transparent_shaders[p.guided_background_stage()]))
    {
      return p.fail("Cycles WebGPU needs rebuilt denoising shaders for transparency");
    }
    if (p.portal_materials &&
        (!p.portal_shaders[p.guide_kernel] ||
         !p.portal_shaders[p.guided_background_stage()]))
    {
      return p.fail("Cycles WebGPU needs rebuilt denoising shaders for Ray Portal");
    }
    if (p.mixed_transparent_materials && !p.mixed_transparent_shaders[p.guide_kernel]) {
      return p.fail("Cycles WebGPU needs rebuilt denoising shaders for mixed Transparent surface closures");
    }
    if (p.standalone_closures && !p.standalone_shaders[p.guide_kernel]) {
      return p.fail("Cycles WebGPU needs a rebuilt denoising shader for standalone surface closures");
    }
    if (p.hair_materials && !p.hair_shaders[p.guide_kernel]) {
      return p.fail("Cycles WebGPU needs a rebuilt denoising shader for Hair");
    }
    if (p.subsurface_materials && !p.subsurface_shaders[p.guide_kernel]) {
      return p.fail("Cycles WebGPU needs a rebuilt denoising shader for subsurface scattering");
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
    if (p.absorption) {
      LOG_INFO << "Cycles WebGPU writing volume denoising guides";
    }
  }
  else {
    p.denoising.free();
    p.denoising_params.free();
  }

  p.adaptive_sampling = adaptive;
  if (p.adaptive_sampling && !p.prepare_adaptive()) {
    return false;
  }
  if (p.pixel_wireframe && p.adaptive_sampling &&
      !p.differential_shaders[adaptive_camera_rays_kernel])
  {
    return p.fail("Cycles WebGPU needs a rebuilt adaptive camera shader for pixel Wireframe");
  }
  p.aov_active = aov_slots != 0;
  if (p.aov_active) {
    if (!p.aov_loaded) {
      const auto shader = p.load_shader("film.wgsl");
      p.aov_loaded = !shader.empty() &&
                     p.register_kernel(accumulate_aov_kernel, shader, aov_bindings, "accumulate_aov");
    }
    if (!p.aov_loaded ||
        !upload(p.device, p.aov_params, std::span<const uint32_t>(aov_layout)))
    {
      return p.fail("Cycles WebGPU could not initialize film AOV passes");
    }
  }
  else {
    p.aov_params.free();
  }
  std::span<const float> camera_data = camera;
  std::vector<float> camera_words;
  if (p.aov_materials || !p.query_sampling.empty()) {
    camera_words.assign(camera.begin(), camera.end());
    camera_words.resize(CYCLES_WEBGPU_CAMERA_WORDS + CYCLES_WEBGPU_CAMERA_SVM_CONFIG_WORDS);
    camera_words[CYCLES_WEBGPU_CAMERA_AOV_COLORS] = std::bit_cast<float>(aov_layout[1]);
    camera_words[CYCLES_WEBGPU_CAMERA_AOV_VALUES] = std::bit_cast<float>(aov_layout[3]);
    for (const uint32_t word : p.query_sampling) {
      camera_words.push_back(std::bit_cast<float>(word));
    }
    camera_data = camera_words;
  }
  if (!upload(p.device, p.camera, camera_data) ||
      !upload(p.device, p.params, std::span<const uint32_t>(p.film_config)))
  {
    return p.fail("Cycles WebGPU camera or film parameters could not be uploaded");
  }
  p.pixels.alloc_to_device(pixel_words_per_path * capacity);
  p.rays.alloc_to_device(ray_header_words +
                         ray_words_per_path(adaptive, p.pixel_wireframe, p.scattering) * capacity);
  p.hits.alloc_to_device(hit_words_per_path * capacity);
  if (p.absorption) {
    p.absorption_state.alloc_to_device(capacity *
                                       (p.scattering ? CYCLES_WEBGPU_VOLUME_STATE_WORDS : 1u));
    if (!p.absorption_state.device_pointer) {
      return p.fail("Cycles WebGPU could not allocate homogeneous absorption paths");
    }
  }
  else {
    p.absorption_state.free();
  }
  if (p.deferred_shadow) {
    p.shadow_rays.alloc_to_device(shadow_storage_words_per_path(p.textured_lights) * capacity);
    if (!p.shadow_rays.device_pointer) {
      return p.fail("Cycles WebGPU could not allocate shadow rays");
    }
  }
  else {
    p.shadow_rays.free();
  }
  p.points.alloc_to_device(
      shading_storage_words_per_path(!p.query_sampling.empty(), p.hair_materials) * capacity);
  /* Area light hits accumulate separately from surface and world shading. */
  p.radiance.alloc_to_device(radiance_words_per_path * capacity);
  p.materials.alloc_to_device((CYCLES_WEBGPU_MATERIAL_WORDS + aov_words) * capacity);
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
      stage != DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND && stage != subsurface_kernel)
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
    const auto absorption = p.bind(
        p.scene.bvh, p.rays, p.hits, p.scene.surface, paths, p.absorption_state);
    const WebGPUDispatch classify_dispatch{WebGPUWavefrontQueue::classify_kernel, classify, count};
    if (p.scattering) {
      const auto scatter = p.bind(
          p.rays, p.hits, p.scene.surface, paths, p.absorption_state, p.sampling, p.scene.light,
          p.shadow_rays);
      const auto shadow = p.guided ?
          p.bind(p.scene.bvh, p.rays, p.shadow_rays, paths, p.scene.surface, p.absorption_state,
                 p.denoising, p.denoising_params) :
          p.bind(p.scene.bvh, p.rays, p.shadow_rays, paths, p.scene.surface, p.absorption_state);
      const WebGPUDispatch segment[] = {
          {bvh_kernel, closest, count}, {volume_interval_kernel, absorption, count}};
      const WebGPUDispatch integrate{
          p.volume_scatter_stage(p.guided), scatter, count};
      const WebGPUDispatch visibility[] = {
          {p.volume_scatter_shadow_stage(p.guided), shadow, count},
          classify_dispatch};
      if (!p.wavefront_queue.enqueue_stage(stage, capacity, p.rays, paths, segment) ||
          !device_webgpu_enqueue(p.device, count, {&integrate, 1}) ||
          !device_webgpu_enqueue(p.device, count, visibility))
      {
        return p.fail("Cycles WebGPU volume intersection dispatch failed");
      }
      return true;
    }
    const WebGPUDispatch stages[] = {
        {bvh_kernel, closest, count},
        p.absorption ? WebGPUDispatch{absorption_kernel, absorption, count} : classify_dispatch,
        classify_dispatch,
    };
    if (!p.wavefront_queue.enqueue_stage(stage, capacity, p.rays, paths,
                                         std::span(stages).first(p.absorption ? 3 : 2))) {
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
  if (stage == subsurface_kernel) {
    const auto walk = p.bind(p.scene.bvh, p.scene.surface, p.sampling, p.points, p.materials,
                             p.hits, p.rays, paths);
    const WebGPUDispatch stages[] = {{subsurface_kernel, walk, count}, accumulate};
    if (!p.wavefront_queue.enqueue_stage(stage, capacity, p.rays, paths, stages)) {
      return p.fail("Cycles WebGPU subsurface dispatch failed");
    }
    return true;
  }
  if (stage == DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND &&
      p.shader_loaded[p.active_background_kernel])
  {
    const auto background = p.guided ? p.bind(p.rays, p.hits, p.scene.light, p.sampling, paths,
                                                p.denoising) :
                                       p.bind(p.rays, p.hits, p.scene.light, p.sampling, paths);
    const DeviceKernel kernel = p.guided ? p.guided_background_stage() : p.active_background_kernel;
    const WebGPUDispatch stages[] = {{kernel, background, count}, accumulate};
    if (!p.wavefront_queue.enqueue_stage(stage, capacity, p.rays, paths, stages)) {
      return p.fail("Cycles WebGPU background dispatch failed");
    }
    return true;
  }

  const auto surface = p.bind(p.scene.bvh, p.scene.surface, p.rays, p.hits, p.points);
  const auto aovs = p.bind(p.materials, p.film, paths, p.rays, p.params, p.aov_params);
  const WebGPUDispatch accumulate_aovs{accumulate_aov_kernel, aovs, count};
  const bool queries = !p.query_sampling.empty();
  const auto svm = p.bind(p.scene.svm, p.points, p.rays, p.materials,
                          queries ? p.scene.bvh : p.hits, paths, p.camera, p.scene.surface);
  const auto light = p.bind(p.deferred_shadow ? p.shadow_rays : p.scene.bvh,
                            p.rays,
                            p.hits,
                            p.points,
                            p.scene.light,
                            p.radiance,
                            p.materials,
                            paths);
  const auto shadow = p.scattering ?
                          p.bind(p.scene.bvh, p.rays, p.shadow_rays, p.radiance, p.scene.surface,
                                 p.absorption_state, paths) :
                      p.absorption ?
                          p.bind(p.scene.bvh, p.rays, p.shadow_rays, p.radiance, p.scene.surface,
                                 p.absorption_state) :
                      p.active_shadow_kernel == varying_shadow_kernel ?
                          p.bind(p.scene.bvh, p.rays, p.shadow_rays, p.radiance, p.scene.surface, paths,
                                 p.scene.svm, p.camera) :
                      p.transparent_materials ?
                          p.bind(p.scene.bvh, p.rays, p.shadow_rays, p.radiance, p.scene.surface, paths) :
                          p.bind(p.scene.bvh, p.rays, p.shadow_rays, p.radiance);
  const WebGPUDispatch intersect_shadow{p.active_shadow_kernel, shadow, count};
  const auto emission = p.bind(
      p.scene.svm, p.shadow_rays, p.rays, p.radiance, paths, p.camera, p.scene.surface, p.scene.bvh);
  const WebGPUDispatch shade_emission{sampled_emission_kernel, emission, count};
  const auto guides = p.bind(p.rays, p.points, p.materials, p.sampling, p.radiance, paths,
                             p.denoising, p.denoising_params);
  const WebGPUDispatch shade_guides{p.guide_kernel, guides, count};
  const auto update = p.bind(
      p.rays, p.hits, p.points, p.pixels, p.sampling, p.materials, p.radiance, paths);
  const WebGPUDispatch shading[] = {
      {queries ? surface_query_kernel : surface_kernel, surface, count},
      {p.active_svm_kernel, svm, count},
      {p.light_kernel, light, count},
      {p.update_kernel, update, count},
      accumulate};
  const auto stages = std::span(shading);
  /* Visibility traversal and BSDF continuation are the expensive stages.
   * Submit them separately to bound the work in each browser command buffer. */
  if (!p.wavefront_queue.enqueue_stage(stage, capacity, p.rays, paths, stages.first(2)) ||
      (p.aov_active && !device_webgpu_enqueue(p.device, count, {&accumulate_aovs, 1})) ||
      !device_webgpu_enqueue(p.device, count, stages.subspan(2, 1)) ||
      (p.textured_lights && !device_webgpu_enqueue(p.device, count, {&shade_emission, 1})) ||
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
