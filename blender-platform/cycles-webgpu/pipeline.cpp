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
using ShaderFeatures = std::array<bool, size_t(webgpu_svm_hardware_textures_kernel) + 1>;
constexpr std::string_view validated_svm_marker = "// Cycles WebGPU validated SVM node bounds\n";
constexpr std::string_view hardware_svm_marker = "// Cycles WebGPU hardware texture atlas\n";
constexpr std::string_view triangle_light_tree_header =
    "// Cycles Light Tree finite point\n// Cycles Light Tree triangle emitters\n";
constexpr std::string_view textured_light_tree_marker = "// Cycles Light Tree textured emitters\n";
constexpr std::string_view shadow_target_header = "// Cycles WebGPU shadow targets\n";
constexpr std::string_view standalone_closure_marker =
    "// Cycles WebGPU standalone surface closures v2\n";
constexpr std::string_view mixed_diffuse_marker = "// Cycles WebGPU mixed Diffuse closures v4\n";
constexpr std::string_view independent_mixed_bump_marker =
    "// Cycles WebGPU independent mixed Bump normals\n";
constexpr std::string_view ordered_closures_marker = "// Cycles WebGPU ordered closures\n";
constexpr std::string_view integer_ordered_tail_marker =
    "// Cycles WebGPU integer ordered tail indices\n";
constexpr std::string_view ordered_transparency_marker =
    "// Cycles WebGPU ordered Transparent and Translucent closures\n";
constexpr std::string_view glossy_displacement_bump_marker =
    "// Cycles WebGPU Glossy displacement Bump\n";
constexpr std::string_view mixed_displacement_bump_marker =
    "// Cycles WebGPU Diffuse and Glossy displacement Bump\n";
constexpr std::string_view residual_displacement_bump_marker =
    "// Cycles WebGPU residual displacement bump\n";
constexpr std::string_view glossy_transparent_marker = "// Cycles WebGPU Glossy and Transparent\n";
constexpr std::string_view transparent_emission_marker = "// Cycles WebGPU Transparent emission\n";
constexpr std::string_view closure_compositions_marker = "// Cycles WebGPU closure compositions\n";
constexpr std::string_view paired_principled_layers_marker =
    "// Cycles WebGPU paired Principled layers with metallic\n";
constexpr std::string_view single_principled_coat_marker =
    "// Cycles WebGPU single Principled Coat\n";
constexpr std::string_view coated_principled_emission_marker =
    "// Cycles WebGPU coated Principled emission\n";
constexpr std::string_view cubic_transparent_shadow_marker =
    "// Cycles WebGPU cubic transparent shadows\n";
constexpr std::string_view shadow_extended_svm_marker =
    "// Cycles WebGPU extended shadow SVM\n";
constexpr uint32_t principled_compositions = CYCLES_WEBGPU_SHADER_ORDERED_PRINCIPLED |
                                            CYCLES_WEBGPU_SHADER_PRINCIPLED_TRANSLUCENT;
constexpr uint32_t closure_composition_mask = principled_compositions |
                                               CYCLES_WEBGPU_SHADER_ORDERED_GLOSSY_MIX;
constexpr std::string_view lamp_sampling_flags_marker = "// Cycles WebGPU lamp sampling flags\n";
constexpr std::string_view lamp_falloff_marker = "// Cycles WebGPU analytic Light Falloff\n";
constexpr std::string_view ray_length_lamp_marker = "// Cycles WebGPU ray-length lamp emission\n";
constexpr std::string_view aov_marker = "// Cycles WebGPU AOV output\n";
constexpr std::string_view reflective_caustics_marker =
    "// Cycles WebGPU reflective caustics visibility\n";
constexpr std::string_view compact_voronoi_distance_marker =
    "// Cycles WebGPU compact Voronoi Distance\n";
constexpr std::string_view glass_caustics_marker = "// Cycles WebGPU glass caustics visibility\n";
constexpr std::string_view diffuse_principled_mix_marker =
    "// Cycles WebGPU weighted diffuse Principled mixtures\n";
constexpr std::string_view projected_query_images_marker =
    "// Cycles WebGPU projected query images\n";
constexpr std::string_view advanced_query_marker = "// Cycles WebGPU advanced procedural queries\n";
constexpr std::string_view transparent_marker = "// Cycles WebGPU constant transparent surface\n";
constexpr std::string_view varying_transparent_marker =
    "// Cycles WebGPU varying transparent surface\n";
constexpr std::string_view absorption_marker = "// Cycles WebGPU homogeneous absorption bounds\n";
constexpr std::string_view volume_queue_marker = "// Cycles WebGPU volume scatter classification\n";
constexpr std::string_view mixed_volume_marker = "// Cycles WebGPU mixed surface volume boundaries\n";
constexpr std::string_view additive_volume_marker = "// Cycles WebGPU literal additive volume\n";
constexpr std::string_view extended_volume_stack_marker = "// Cycles WebGPU extended volume stack\n";
constexpr std::string_view transparent_volume_marker = "// Cycles WebGPU transparent volume shadows\n";
constexpr std::string_view varying_volume_dispatch_marker =
    "// Cycles WebGPU varying volume shadow dispatch\n";
constexpr std::string_view compact_volume_shadow_marker =
    "// Cycles WebGPU compact material shadows\n";
constexpr std::string_view volume_differential_marker = "// Cycles WebGPU volume ray differentials\n";
constexpr std::string_view principled_alpha_marker = "// Cycles WebGPU Principled Alpha\n";
constexpr std::string_view volume_guide_surface_marker =
    "// Cycles WebGPU volume guide surface hits\n";
constexpr std::string_view portal_marker = "// Cycles WebGPU ray portals\n";
constexpr std::string_view differential_marker = "// Cycles WebGPU ray differentials\n";
constexpr std::string_view camera_lens_marker = "// Cycles WebGPU camera lens\n";
constexpr std::string_view mixed_lights_marker = "// Cycles WebGPU stock mixed-light distribution\n";
constexpr std::string_view sky_world_marker = "// Cycles WebGPU procedural sky world\n";
constexpr std::string_view mapped_world_marker = "// Cycles WebGPU mapped world SVM\n";
constexpr std::string_view hair_marker = "// Cycles WebGPU hair BSDFs\n";
constexpr std::string_view subsurface_marker = "// Cycles WebGPU Christensen-Burley subsurface\n";
constexpr std::string_view subsurface_bump_marker = "// Cycles WebGPU subsurface bump normals\n";
/* The stock scheduler takes one eighth of the path pool as its work tile limit. */
constexpr uint32_t min_path_capacity = 8;

constexpr uint32_t pixel_words_per_path = 4;
constexpr uint32_t ray_header_words = 2;
constexpr uint32_t max_volume_ray_words_per_path = 17;
constexpr uint32_t hit_words_per_path = 8;
constexpr uint32_t shading_words_per_path = 24;
constexpr uint32_t shading_storage_words_per_path(const bool queries, const bool hair)
{
  return shading_words_per_path + (queries ? CYCLES_WEBGPU_QUERY_HIT_WORDS : 0) +
         (hair ? CYCLES_WEBGPU_SURFACE_HAIR_WORDS : 0);
}
constexpr uint32_t radiance_words_per_path = 8;
constexpr uint32_t emission_words_per_path = 4;
constexpr uint32_t light_pass_words_per_path = 24;
constexpr uint32_t surface_pass_words_per_path = 24;
constexpr uint32_t radiance_header_words = 4;
/* Denoising feature throughput, albedo sum and world normal sum. */
constexpr uint32_t denoising_words_per_path = 12;
constexpr uint32_t work_tile_header_words = 4;
constexpr uint32_t work_tile_words = 8;
/* Keep the original and filtered surface indices until SVM is complete. */
constexpr uint32_t surface_roulette_queue_words_per_path = 2;
constexpr uint32_t surface_roulette_queue_header_words = 2;
constexpr uint32_t surface_roulette_dispatch_words = 5;
/* Origin, direction, self-intersection IDs, and source triangle geometry. */
constexpr uint32_t shadow_words_per_path = 16;

constexpr uint32_t shadow_storage_words_per_path(const bool textured)
{
  return shadow_words_per_path + (textured ? CYCLES_WEBGPU_TEXTURED_LIGHT_SAMPLE_WORDS : 0);
}

constexpr uint32_t ray_words_per_path(const bool adaptive,
                                       const bool differentials,
                                       const bool scattering,
                                       const bool compact_volume_shadows = false)
{
  if (compact_volume_shadows) {
    /* The shadow indices follow the longest volume ray layout. */
    return max_volume_ray_words_per_path + 1;
  }
  /* The shared active/free queue is followed by one film class word per adaptive path. */
  return differentials ? 16 + uint32_t(scattering) : (scattering ? 15 : (adaptive ? 14 : 13));
}

struct PathBufferSize {
  size_t words_per_path;
  size_t header_words = 0;
};

constexpr auto path_buffer_sizes(const bool adaptive,
                                 const bool deferred_shadow,
                                 const bool guided = false,
                                 const bool textured = false,
                                 const size_t extra_material_words = 0,
                                 const bool queries = false,
                                 const bool differentials = false,
                                 const bool hair = false,
                                 const bool absorption = false,
                                 const bool scattering = false,
                                 const bool emission = false,
                                 const bool surface_passes = false,
                                 const bool light_passes = false,
                                 const bool compact_volume_shadows = false,
                                 const bool compact_surface_roulette = false)
{
  return std::array{PathBufferSize{pixel_words_per_path},
                    PathBufferSize{
                        ray_words_per_path(adaptive, differentials, scattering, compact_volume_shadows),
                        ray_header_words + uint32_t(compact_volume_shadows)},
                    PathBufferSize{hit_words_per_path},
                    PathBufferSize{deferred_shadow ? shadow_storage_words_per_path(textured) : 0},
                    PathBufferSize{shading_storage_words_per_path(queries, hair)},
                    PathBufferSize{radiance_words_per_path +
                                       (emission ? emission_words_per_path : 0) +
                                       (light_passes ? light_pass_words_per_path : 0),
                                   radiance_header_words},
                    PathBufferSize{guided ? denoising_words_per_path : 0},
                    PathBufferSize{surface_passes ? surface_pass_words_per_path : 0},
                    PathBufferSize{CYCLES_WEBGPU_MATERIAL_WORDS + extra_material_words},
                    PathBufferSize{CYCLES_WEBGPU_PATH_WORDS},
                    PathBufferSize{scattering ? CYCLES_WEBGPU_VOLUME_STATE_WORDS :
                                                (absorption ? 1u : 0u)},
                    PathBufferSize{
                        compact_surface_roulette ? surface_roulette_queue_words_per_path : 0u,
                        compact_surface_roulette ? surface_roulette_queue_header_words : 0u},
                    PathBufferSize{0, compact_surface_roulette ? surface_roulette_dispatch_words : 0u},
                    /* A batch cannot admit more than one work tile per path slot. */
                    PathBufferSize{work_tile_words, work_tile_header_words}};
}

constexpr uint32_t path_capacity_for_limits(const WebGPUComputeLimits &limits,
                                            const std::span<const PathBufferSize> buffers,
                                            size_t &bytes_per_path)
{
  const size_t binding_words = std::min(limits.storage_buffer_bytes / sizeof(uint32_t),
                                        size_t(UINT32_MAX));
  uint64_t capacity = std::numeric_limits<int>::max();
  size_t words_per_path = 0;
  for (const auto &buffer : buffers) {
    if (binding_words < buffer.header_words) {
      return 0;
    }
    if (buffer.words_per_path == 0) {
      continue;
    }
    capacity = std::min<uint64_t>(
        capacity, (binding_words - buffer.header_words) / buffer.words_per_path);
    words_per_path += buffer.words_per_path;
  }
  bytes_per_path = words_per_path * sizeof(uint32_t);
  return uint32_t(capacity);
}

static_assert([] {
  size_t bytes_per_path = 0;
  constexpr auto buffers = path_buffer_sizes(true, true);
  return path_capacity_for_limits({128 * 1024 * 1024, 65535}, buffers, bytes_per_path) ==
             (128 * 1024 * 1024) / (CYCLES_WEBGPU_MATERIAL_WORDS * sizeof(uint32_t)) &&
         path_capacity_for_limits({1536, 1}, buffers, bytes_per_path) ==
             1536 / (CYCLES_WEBGPU_MATERIAL_WORDS * sizeof(uint32_t)) &&
         path_capacity_for_limits({std::numeric_limits<size_t>::max(), 65535},
                                  buffers, bytes_per_path) ==
             std::min(std::numeric_limits<size_t>::max() / sizeof(uint32_t), size_t(UINT32_MAX)) /
                 CYCLES_WEBGPU_MATERIAL_WORDS;
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
constexpr DeviceKernel varying_volume_shadow_kernel = DEVICE_KERNEL_CRYPTOMATTE_POSTPROCESS;
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
constexpr DeviceKernel volume_transition_kernel = DEVICE_KERNEL_VOLUME_GUIDING_FILTER_X;
constexpr DeviceKernel volume_transmission_light_kernel = DEVICE_KERNEL_VOLUME_GUIDING_FILTER_Y;
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
constexpr DeviceKernel svm_textures_kernel = DEVICE_KERNEL_INTEGRATOR_SHADE_VOLUME_RAY_MARCHING;
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
constexpr DeviceKernel write_geometry_passes_kernel = DEVICE_KERNEL_FILTER_GUIDING_PREPROCESS;
constexpr DeviceKernel accumulate_emission_kernel = DEVICE_KERNEL_FILTER_GUIDING_PREPROCESS_TO_SURFACE;
constexpr DeviceKernel shade_surface_passes_kernel =
    DEVICE_KERNEL_FILTER_COLOR_PREPROCESS_TO_SURFACE;
constexpr DeviceKernel accumulate_surface_passes_kernel =
    DEVICE_KERNEL_FILTER_COLOR_POSTPROCESS_FROM_SURFACE;
constexpr DeviceKernel shade_ao_pass_kernel = DEVICE_KERNEL_FILTER_COLOR_FLIP_Y;
/* The private film dispatch never enters the stock integrator queues. */
constexpr DeviceKernel accumulate_light_passes_kernel = DEVICE_KERNEL_NUM;
constexpr std::string_view emission_delta_marker = "// Cycles WebGPU Emission pass deltas\n";
constexpr std::string_view film_light_marker = "// Cycles WebGPU per-lobe film lighting\n";
constexpr std::string_view film_surface_marker = "// Cycles WebGPU filtered film layout 6\n";
constexpr std::string_view volume_geometry_marker = "// Cycles WebGPU volume geometry passes\n";
constexpr std::string_view varying_ao_shadow_marker =
    "// Cycles WebGPU varying Ambient Occlusion shadows\n";
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
static_assert(UINT32_MAX / CYCLES_WEBGPU_VOLUME_BOUNDARY_INCREMENT(CYCLES_WEBGPU_MAX_VOLUME_MEDIA) >
              VOLUME_BOUNDS_MAX);
/* These constants mirror volume-transition.wgsl's compact dispatch layout. */
static_assert(PATH_RAY_VISIBILITY_TRANSMIT == 2 && CYCLES_WEBGPU_MIXED_VOLUME_BOUNDARY == 5);
static_assert(CYCLES_WEBGPU_PATH_WORDS == 24 && CYCLES_WEBGPU_PATH_QUEUED_KERNEL == 7 &&
              CYCLES_WEBGPU_PATH_INTERSECT_CLOSEST == 3 && CYCLES_WEBGPU_HIT_SPECIAL == 7 &&
              CYCLES_WEBGPU_VOLUME_NEXT_BOUNDARY == 1 && CYCLES_WEBGPU_PATH_FLAGS == 14 &&
              PATH_RAY_TRANSPARENT == 4);
static_assert(PATH_RAY_SUBSURFACE_RANDOM_WALK == CYCLES_WEBGPU_PATH_SUBSURFACE_RANDOM_WALK);
static_assert(SD_HAS_BSSRDF_BUMP == CYCLES_WEBGPU_SHADER_BSSRDF_BUMP);

constexpr auto storage = WebGPUKernelBinding::Type::Storage;
constexpr auto read_only = WebGPUKernelBinding::Type::ReadOnlyStorage;
constexpr auto uniform = WebGPUKernelBinding::Type::Uniform;
constexpr size_t material_bytes = CYCLES_WEBGPU_MATERIAL_WORDS * sizeof(float);
constexpr size_t denoising_bytes = denoising_words_per_path * sizeof(float);
constexpr size_t denoising_params_bytes = 16 * sizeof(uint32_t);

/* The translated WGSL uses read_write bindings even for its input buffers. */
constexpr WebGPUKernelBinding camera_bindings[] = {
    {0, storage, 32},
    {1, storage, CYCLES_WEBGPU_CAMERA_RAY_WORDS * sizeof(float)},
    {2, storage, 16},
    {3, storage, 52},
    {4, storage, 28}};
constexpr WebGPUKernelBinding bvh_bindings[] = {
    {0, storage, 40}, {1, storage, 52}, {2, storage, 32}, {3, storage, 16}};
constexpr WebGPUKernelBinding surface_bindings[] = {
    {0, storage, 40}, {1, storage, 8}, {2, storage, 52}, {3, storage, 32}, {4, storage, 32}};
constexpr WebGPUKernelBinding background_bindings[] = {
    {0, storage, 52},
    {1, storage, 32},
    {2, storage, 48},
    {3, storage, 28},
    {4, storage, CYCLES_WEBGPU_PATH_WORDS * sizeof(uint32_t)},
    {5, storage, 16}};
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
constexpr auto svm_hardware_bindings = [] {
  std::array<WebGPUKernelBinding, std::size(svm_bindings) + 2> bindings{};
  std::copy(std::begin(svm_bindings), std::end(svm_bindings), bindings.begin());
  bindings[8] = {8, WebGPUKernelBinding::Type::FloatTexture, 0};
  bindings[9] = {9, WebGPUKernelBinding::Type::FilteringSampler, 0};
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
constexpr WebGPUKernelBinding volume_transition_bindings[] = {
    {0, read_only, 52},
    {1, read_only, 32},
    {2, read_only, CYCLES_WEBGPU_PATH_WORDS * sizeof(uint32_t)},
    {3, storage, CYCLES_WEBGPU_VOLUME_STATE_WORDS * sizeof(uint32_t)}};
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
constexpr auto geometry_pass_bindings = std::array{
    WebGPUKernelBinding{1, read_only, material_bytes},
    accumulate_bindings[0], accumulate_bindings[1], accumulate_bindings[2], accumulate_bindings[3],
    WebGPUKernelBinding{12, read_only, 32},
    WebGPUKernelBinding{13, read_only, 32},
    WebGPUKernelBinding{14, read_only, CYCLES_WEBGPU_SURFACE_OFFSETS * sizeof(uint32_t)},
    WebGPUKernelBinding{15, uniform, 32}};
constexpr auto emission_pass_bindings = std::array{
    WebGPUKernelBinding{2, read_only, 16},
    accumulate_bindings[0], accumulate_bindings[1], accumulate_bindings[2], accumulate_bindings[3],
    WebGPUKernelBinding{16, uniform, 16}};

constexpr WebGPUKernelBinding surface_pass_shader_bindings[] = {
    {0, storage, 52}, {1, storage, 32}, {2, storage, material_bytes}, {3, storage, 28},
    {4, storage, 16}, {5, storage, CYCLES_WEBGPU_PATH_WORDS * sizeof(uint32_t)},
    {6, storage, surface_pass_words_per_path * sizeof(uint32_t)}, {7, storage, 16}};
constexpr WebGPUKernelBinding ao_pass_shader_bindings[] = {
    {0, storage, 40}, {1, storage, 52}, {2, storage, 32}, {3, storage, 32},
    {4, storage, CYCLES_WEBGPU_PATH_WORDS * sizeof(uint32_t)}, {5, storage, 28},
    {6, storage, surface_pass_words_per_path * sizeof(uint32_t)},
    {7, storage, CYCLES_WEBGPU_SURFACE_OFFSETS * sizeof(uint32_t)}};
constexpr WebGPUKernelBinding ao_prepare_shader_bindings[] = {
    {0, storage, shadow_words_per_path * sizeof(uint32_t)},
    ao_pass_shader_bindings[1], ao_pass_shader_bindings[2], ao_pass_shader_bindings[3],
    ao_pass_shader_bindings[4], ao_pass_shader_bindings[5], ao_pass_shader_bindings[6],
    {7, storage, material_bytes}};
constexpr auto surface_pass_film_bindings = std::array{
    WebGPUKernelBinding{2, read_only, surface_pass_words_per_path * sizeof(uint32_t)},
    accumulate_bindings[0], accumulate_bindings[1], accumulate_bindings[2], accumulate_bindings[3],
    WebGPUKernelBinding{17, uniform, 32}};
constexpr auto light_pass_film_bindings = std::array{
    WebGPUKernelBinding{2, read_only, 16},
    accumulate_bindings[0], accumulate_bindings[1], accumulate_bindings[2], accumulate_bindings[3],
    WebGPUKernelBinding{18, uniform, 32}};

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
    WebGPUKernelBinding{5, storage, denoising_bytes}, WebGPUKernelBinding{6, storage, 16}};
constexpr auto guided_volume_scatter_shadow_bindings = std::array{
    volume_scatter_shadow_bindings[0], volume_scatter_shadow_bindings[1],
    volume_scatter_shadow_bindings[2], volume_scatter_shadow_bindings[3],
    volume_scatter_shadow_bindings[4], volume_scatter_shadow_bindings[5],
    WebGPUKernelBinding{6, storage, denoising_bytes},
    WebGPUKernelBinding{7, storage, denoising_params_bytes}};
constexpr auto varying_volume_shadow_bindings = std::array{
    volume_scatter_shadow_bindings[0], volume_scatter_shadow_bindings[1],
    volume_scatter_shadow_bindings[2], volume_scatter_shadow_bindings[3],
    volume_scatter_shadow_bindings[4], volume_scatter_shadow_bindings[5],
    WebGPUKernelBinding{6, storage, 16},
    WebGPUKernelBinding{7, storage, 8}};
constexpr WebGPUKernelBinding reset_volume_shadow_bindings[] = {
    {0, storage, sizeof(uint32_t)}, {2, storage, 3 * sizeof(uint32_t)}};
constexpr auto compact_volume_shadow_bindings = std::array{
    reset_volume_shadow_bindings[0],
    WebGPUKernelBinding{1, read_only, shadow_words_per_path * sizeof(uint32_t)},
    reset_volume_shadow_bindings[1]};
constexpr auto compact_guided_volume_shadow_bindings = std::array{
    compact_volume_shadow_bindings[0], compact_volume_shadow_bindings[1],
    compact_volume_shadow_bindings[2], WebGPUKernelBinding{3, storage, denoising_bytes},
    WebGPUKernelBinding{4, read_only, CYCLES_WEBGPU_SURFACE_VOLUME_SHADOW_OFFSETS * sizeof(uint32_t)}};
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

bool valid_mixed_lights(const std::span<const uint32_t> light, const size_t prefix)
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
  const bool distribution = nodes == 0;
  const uint64_t descriptor_words = distribution ? CYCLES_WEBGPU_DISTRIBUTION_DESCRIPTOR_WORDS :
                                                   CYCLES_WEBGPU_TREE_DESCRIPTOR_WORDS;
  if (tree.size() < descriptor_words) {
    return false;
  }
  const uint64_t distribution_count = distribution ? tree[CYCLES_WEBGPU_DISTRIBUTION_COUNT] : 0;
  if ((distribution ? (emitters != 0 || distribution_count == 0 ||
                       distribution_count != lamps + triangles || triangle_lookups != 0) :
                      (emitters == 0 || emitters < lamps + triangles ||
                       (triangles ? triangles > triangle_indices || triangles > triangle_lookups :
                                    emitters != lamps))) ||
      textured > triangles ||
      (triangles ? (!objects || !shaders || !triangle_indices) :
                   (objects || triangle_indices || shaders || triangle_lookups)) ||
      (forward_lamp != CYCLES_WEBGPU_TREE_NO_LAMP && forward_lamp >= lamps) ||
      (sampled_world ? (world_lamp >= lamps ||
                        world_emitter >= (distribution ? distribution_count : emitters) ||
                        !map_w || !map_h) :
                       (world_emitter != CYCLES_WEBGPU_TREE_NO_LAMP || map_w || map_h)) ||
      std::max({map_w, map_h, tex_w, tex_h}) > light.size())
  {
    return false;
  }
  const uint64_t lamp_offset = prefix + descriptor_words;
  const uint64_t node_offset = lamp_offset + lamps * CYCLES_WEBGPU_TREE_LAMP_WORDS;
  const uint64_t emitter_offset = node_offset + nodes * CYCLES_WEBGPU_TREE_NODE_WORDS;
  const uint64_t object_offset = emitter_offset + emitters * CYCLES_WEBGPU_TREE_EMITTER_WORDS;
  const uint64_t triangle_index_offset = object_offset + objects * CYCLES_WEBGPU_TREE_OBJECT_WORDS;
  const uint64_t triangle_offset = triangle_index_offset + triangle_indices;
  const uint64_t shader_offset = triangle_offset + triangles * CYCLES_WEBGPU_TREE_TRIANGLE_WORDS;
  const uint64_t triangle_lookup_offset = shader_offset +
                                          shaders * CYCLES_WEBGPU_TREE_SHADER_WORDS;
  const uint64_t distribution_offset = triangle_lookup_offset + triangle_lookups;
  const uint64_t marginal_offset = distribution_offset +
      (distribution ? (distribution_count + 1) * CYCLES_WEBGPU_POINT_DISTRIBUTION_WORDS : 0);
  const uint64_t conditional_offset = marginal_offset + (sampled_world ? 2 * (map_h + 1) : 0);
  const uint64_t pixel_offset = conditional_offset + 2 * (map_w + 1) * map_h;
  const uint32_t world_mode = tree[CYCLES_WEBGPU_TREE_WORLD_INTERPOLATION];
  const bool sky = world_mode & CYCLES_WEBGPU_WORLD_SKY_FLAG;
  const bool mapped = world_mode & CYCLES_WEBGPU_WORLD_SVM_FLAG;
  if (pixel_offset > light.size()) {
    return false;
  }
  const uint64_t available = light.size() - pixel_offset;
  uint64_t program_words = sky ? CYCLES_WEBGPU_SKY_HEADER_WORDS : 0;
  if (mapped) {
    if (sky || !tex_w || !tex_h || available < CYCLES_WEBGPU_WORLD_SVM_HEADER_WORDS) {
      return false;
    }
    const uint32_t count = light[pixel_offset + CYCLES_WEBGPU_WORLD_SVM_WORD_COUNT];
    if (!count) {
      return false;
    }
    program_words = uint64_t(CYCLES_WEBGPU_WORLD_SVM_HEADER_WORDS) + count;
  }
  if (program_words > available ||
      (tex_w && tex_h > (available - program_words) / 3 / tex_w))
  {
    return false;
  }
  const uint64_t sampling_offset = pixel_offset + program_words + 3 * tex_w * tex_h;
  const uint64_t falloff_offset = light[CYCLES_WEBGPU_TREE_LAMP_FALLOFF_OFFSET];
  if (falloff_offset != 0 &&
      (!lamps || falloff_offset < sampling_offset + CYCLES_WEBGPU_LIGHT_SAMPLING_WORDS ||
       falloff_offset + lamps * CYCLES_WEBGPU_LAMP_FALLOFF_WORDS > light.size()))
  {
    return false;
  }
  return tree[CYCLES_WEBGPU_TREE_LAMPS] == lamp_offset &&
         tree[CYCLES_WEBGPU_TREE_NODES] == node_offset &&
         tree[CYCLES_WEBGPU_TREE_EMITTERS] == emitter_offset &&
         tree[CYCLES_WEBGPU_TREE_OBJECTS] == object_offset &&
         tree[CYCLES_WEBGPU_TREE_TRIANGLE_INDICES] == triangle_index_offset &&
         tree[CYCLES_WEBGPU_TREE_TRIANGLES] == triangle_offset &&
         tree[CYCLES_WEBGPU_TREE_SHADERS] == shader_offset &&
         tree[CYCLES_WEBGPU_TREE_TRIANGLE_TO_TREE] == triangle_lookup_offset &&
         (!distribution || tree[CYCLES_WEBGPU_DISTRIBUTION_OFFSET] == distribution_offset) &&
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
  device_only_memory<float> camera_rays{&device, "WebGPU camera ray parameters"};
  device_only_memory<uint32_t> sampling{&device, "WebGPU camera sampling"};
  device_only_memory<uint32_t> tile{&device, "WebGPU work tiles"};
  device_only_memory<uint32_t> params{&device, "WebGPU stock film parameters"};
  device_only_memory<uint32_t> pixels{&device, "WebGPU pixel samples"};
  device_only_memory<uint32_t> rays{&device, "WebGPU primary rays"};
  device_only_memory<uint32_t> hits{&device, "WebGPU primary intersections"};
  device_only_memory<uint32_t> shadow_rays{&device, "WebGPU shadow rays"};
  device_only_memory<uint32_t> volume_shadow_indirect{&device, "WebGPU volume shadow indirect dispatch"};
  device_only_memory<uint32_t> points{&device, "WebGPU shading points"};
  WebGPUPathStateStorage path_state{device};
  device_only_memory<uint32_t> absorption_state{&device, "WebGPU homogeneous absorption paths"};
  WebGPUWavefrontQueue wavefront_queue{device};
  device_only_memory<uint32_t> surface_roulette_queue{&device, "WebGPU surface roulette queue"};
  device_only_memory<uint32_t> surface_roulette_dispatch{&device, "WebGPU SVM indirect dispatch"};
  std::array<uint32_t, size_t(webgpu_svm_hardware_textures_kernel) + 1> svm_workgroup_widths{};
  bool surface_roulette_loaded = false;
  device_only_memory<float> materials{&device, "WebGPU evaluated materials"};
  device_only_memory<float> radiance{&device, "WebGPU sampled radiance"};
  device_only_memory<float> denoising{&device, "WebGPU denoising paths"};
  device_only_memory<uint32_t> denoising_params{&device, "WebGPU denoising parameters"};
  device_only_memory<uint32_t> film{&device, "WebGPU Cycles film"};
  device_only_memory<uint32_t> aov_params{&device, "WebGPU AOV film layout"};
  device_only_memory<uint32_t> geometry_pass_params{&device, "WebGPU geometry film passes"};
  device_only_memory<uint32_t> emission_pass_params{&device, "WebGPU emission film offset"};
  device_only_memory<uint32_t> light_pass_params{&device, "WebGPU light pass offsets"};
  device_only_memory<uint32_t> surface_pass_params{&device, "WebGPU surface pass offsets"};
  device_only_memory<float> surface_pass_values{&device, "WebGPU filtered surface pass values"};
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
  device_ptr svm_image_atlas = 0;
  bool svm_core_loaded = false;
  bool svm_extended_loaded = false;
  bool sort_surface_shaders = false;
  bool sampled_emission_loaded = false;
  bool aov_loaded = false;
  bool aov_active = false;
  bool aov_materials = false;
  bool geometry_passes_active = false;
  bool emission_pass_active = false;
  bool light_passes_active = false;
  bool light_passes_loaded = false;
  bool surface_passes_active = false;
  bool ao_pass_active = false;
  bool ao_prepare_loaded = false;
  bool varying_ao_shadow_supported = false;
  bool standalone_closures = false;
  bool mixed_diffuse = false;
  bool independent_mixed_bump = false;
  bool ordered_closures = false;
  bool ordered_transparency = false;
  uint32_t closure_compositions = 0u;
  bool paired_principled_layers = false;
  bool single_principled_coat = false;
  bool glossy_displacement_bump = false;
  bool mixed_displacement_bump = false;
  bool glossy_transparent = false;
  bool hair_materials = false;
  ShaderFeatures hair_shaders{};
  ShaderFeatures shader_loaded{};
  ShaderFeatures emission_delta_shaders{};
  ShaderFeatures film_light_shaders{};
  ShaderFeatures sky_world_shaders{};
  bool sky_world = false;
  ShaderFeatures mapped_world_shaders{};
  bool mapped_world = false;
  ShaderFeatures standalone_shaders{};
  ShaderFeatures mixed_diffuse_shaders{};
  ShaderFeatures independent_mixed_bump_shaders{};
  ShaderFeatures ordered_closure_shaders{};
  ShaderFeatures integer_ordered_tail_shaders{};
  ShaderFeatures ordered_transparency_shaders{};
  ShaderFeatures closure_composition_shaders{};
  ShaderFeatures paired_principled_layer_shaders{};
  ShaderFeatures single_principled_coat_shaders{};
  ShaderFeatures coated_principled_emission_shaders{};
  ShaderFeatures cubic_transparent_shadow_shaders{};
  ShaderFeatures shadow_extended_svm_shaders{};
  ShaderFeatures glossy_displacement_bump_shaders{};
  ShaderFeatures mixed_displacement_bump_shaders{};
  ShaderFeatures residual_displacement_bump_shaders{};
  ShaderFeatures glossy_transparent_shaders{};
  ShaderFeatures transparent_emission_shaders{};
  ShaderFeatures lamp_sampling_flag_shaders{};
  ShaderFeatures lamp_falloff_shaders{};
  ShaderFeatures ray_length_lamp_shaders{};
  ShaderFeatures aov_shaders{};
  ShaderFeatures reflective_caustics_shaders{};
  ShaderFeatures volume_geometry_shaders{};
  ShaderFeatures compact_voronoi_distance_shaders{};
  ShaderFeatures glass_caustics_shaders{};
  ShaderFeatures diffuse_principled_mix_shaders{};
  bool deferred_shadow = false;
  bool reflection_deferred = false;
  bool full_deferred = false;
  bool shadow_loaded = false;
  DeviceKernel active_shadow_kernel = shadow_kernel;
  bool transparent_materials = false;
  bool absorption = false;
  bool scattering = false;
  bool mixed_volume = false;
  bool additive_volume = false;
  bool extended_volume_stack = false;
  size_t volume_shadow_params_offset = 0;
  bool compact_volume_shadow_shader = false;
  bool compact_volume_shadow_queue_loaded = false;
  bool compact_volume_shadow_linear_grid = false;
  bool compact_volume_shadows = false;
  bool grid_volume = false;
  bool volume_queue_supported = false;
  ShaderFeatures absorption_shaders{};
  ShaderFeatures mixed_volume_shaders{};
  ShaderFeatures additive_volume_shaders{};
  ShaderFeatures extended_volume_stack_shaders{};
  ShaderFeatures transparent_volume_shaders{};
  ShaderFeatures varying_volume_dispatch_shaders{};
  ShaderFeatures volume_differential_shaders{};
  ShaderFeatures principled_alpha_shaders{};
  ShaderFeatures volume_guide_surface_shaders{};
  ShaderFeatures transparent_shaders{};
  ShaderFeatures varying_transparent_shaders{};
  bool mixed_transparent_materials = false;
  ShaderFeatures mixed_transparent_shaders{};
  bool portal_materials = false;
  ShaderFeatures portal_shaders{};
  bool ray_differentials = false;
  bool surface_bump = false;
  bool triangle_tangents = false;
  ShaderFeatures differential_shaders{};
  ShaderFeatures camera_lens_shaders{};
  ShaderFeatures mixed_lights_shaders{};
  bool reflection_shading_loaded = false;
  bool point_distribution_loaded = false;
  bool point_transmission_loaded = false;
  bool shadow_targets_supported = false;
  ShaderFeatures tree_triangles_supported{};
  ShaderFeatures tree_textured_supported{};
  ShaderFeatures tree_background_triangles_supported{};
  bool guided_tree_background_triangles_supported = false;
  bool triangle_lights = false;
  bool light_distribution = false;
  bool lamp_falloff = false;
  bool ray_length_lamps = false;
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
  ShaderFeatures subsurface_shaders{};
  ShaderFeatures subsurface_bump_shaders{};
  bool guided_background_loaded = false;
  bool guided_tree_background_loaded = false;
  uint32_t sampling_pattern = 0;
  std::array<uint32_t, 12> film_config{};
  size_t material_words_per_path = 0;
  uint32_t path_pool_limit = 0;
  uint32_t path_capacity = 0;
  uint32_t pixel_count = 0;
  uint32_t next_sample = 0;
  uint32_t batch_end_sample = 0;
  uint64_t admitted_paths = 0;
  std::array<uint32_t, 4> crop{};

  explicit Impl(Device &device) : device(device) {}

  bool reserve_paths(uint32_t capacity);

  ~Impl()
  {
    device_webgpu_free_texture_atlas(device, svm_image_atlas);
  }

  bool supports_lamp_emission(const DeviceKernel kernel) const
  {
    return (!lamp_falloff || lamp_falloff_shaders[kernel]) &&
           (!ray_length_lamps || ray_length_lamp_shaders[kernel]);
  }

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
    const bool svm = kernel == svm_core_kernel || kernel == svm_textures_kernel ||
                     kernel == svm_extended_kernel || kernel == svm_query_kernel ||
                     kernel == svm_query_textures_kernel || kernel == svm_query_advanced_kernel ||
                     kernel == webgpu_svm_ao_textures_kernel ||
                     kernel == webgpu_svm_validated_textures_kernel ||
                     kernel == webgpu_svm_hardware_textures_kernel;
    uint32_t compiled_width = workgroup_size;
    bool registered = false;
    const uint32_t max_svm_width = svm ? device_webgpu_max_1d_workgroup_size(device) : 0;
    if (svm) {
      /* Generated SVM entry points use global IDs and no workgroup storage. Keep
       * the original entry for older assets and any other shader shape. */
      constexpr std::string_view original = "@workgroup_size(64, 1, 1)";
      const size_t attribute = source.find(original);
      if (max_svm_width > workgroup_size && attribute != source.npos &&
          source.find("@workgroup_size(") == attribute &&
          source.find("@workgroup_size(", attribute + original.size()) == source.npos &&
          source.find("var<workgroup>") == source.npos &&
          source.find("@builtin(local_invocation_") == source.npos &&
          source.find("@builtin(workgroup_id)") == source.npos &&
          source.find("@builtin(num_workgroups)") == source.npos)
      {
        /* Try the previously verified width if a larger group exceeds the
         * compiler's resource budget. The packaged shader is the final retry. */
        constexpr uint32_t fallback_width = 576;
        for (const uint32_t width : {max_svm_width, max_svm_width > fallback_width ? fallback_width : 0u}) {
          if (!width) {
            break;
          }
          std::string resized = "@workgroup_size(" + std::to_string(width) + ",1,1)";
          /* Preserve source length for the browser's shader profiler. */
          if (resized.size() < original.size()) {
            resized.append(original.size() - resized.size(), ' ');
          }
          std::string tuned(source);
          tuned.replace(attribute, original.size(), resized);
          std::string compile_error;
          registered = device_webgpu_try_register_kernel(
              device, kernel, tuned, width, bindings, entry, compile_error);
          if (registered) {
            compiled_width = width;
            break;
          }
          if (compile_error.empty()) {
            break;
          }
          LOG_WARNING << "Cycles WebGPU SVM workgroup width " << width << " was rejected: "
                      << compile_error;
        }
      }
    }
    if (!registered &&
        !device_webgpu_register_kernel(device, kernel, source, workgroup_size, bindings, entry))
    {
      const std::string_view name = kernel == webgpu_prepare_ao_pass_kernel ? "prepare_ao_pass" :
                                    kernel == webgpu_svm_ao_textures_kernel ? "svm_ao_textures" :
                                    kernel == webgpu_svm_validated_textures_kernel ? "svm_validated_textures" :
                                    kernel == webgpu_svm_hardware_textures_kernel ? "svm_hardware_textures" :
                                    kernel >= DEVICE_KERNEL_NUM ? entry :
                                                                  device_kernel_as_string(kernel);
      LOG_ERROR << "Cycles WebGPU could not register compute pipeline " << name;
      return false;
    }
    if (svm) {
      svm_workgroup_widths[kernel] = compiled_width;
      LOG_INFO << "Cycles WebGPU SVM workgroup width " << compiled_width << ", negotiated maximum "
               << max_svm_width;
    }
    if (kernel == varying_volume_shadow_kernel) {
      compact_volume_shadow_shader = source.find(compact_volume_shadow_marker) != source.npos;
    }
    standalone_shaders[kernel] = source.find(standalone_closure_marker) != source.npos;
    mixed_diffuse_shaders[kernel] = source.find(mixed_diffuse_marker) != source.npos;
    independent_mixed_bump_shaders[kernel] =
        source.find(independent_mixed_bump_marker) != source.npos;
    ordered_closure_shaders[kernel] = source.find(ordered_closures_marker) != source.npos;
    integer_ordered_tail_shaders[kernel] = source.find(integer_ordered_tail_marker) != source.npos;
    ordered_transparency_shaders[kernel] = source.find(ordered_transparency_marker) != source.npos;
    closure_composition_shaders[kernel] = source.find(closure_compositions_marker) != source.npos;
    paired_principled_layer_shaders[kernel] = source.find(paired_principled_layers_marker) !=
                                               source.npos;
    single_principled_coat_shaders[kernel] = source.find(single_principled_coat_marker) != source.npos;
    coated_principled_emission_shaders[kernel] = source.find(coated_principled_emission_marker) !=
                                                  source.npos;
    cubic_transparent_shadow_shaders[kernel] =
        source.find(cubic_transparent_shadow_marker) != source.npos;
    shadow_extended_svm_shaders[kernel] =
        source.find(shadow_extended_svm_marker) != source.npos;
    glossy_displacement_bump_shaders[kernel] =
        source.find(glossy_displacement_bump_marker) != source.npos;
    mixed_displacement_bump_shaders[kernel] =
        source.find(mixed_displacement_bump_marker) != source.npos;
    residual_displacement_bump_shaders[kernel] =
        source.find(residual_displacement_bump_marker) != source.npos;
    glossy_transparent_shaders[kernel] = source.find(glossy_transparent_marker) != source.npos;
    transparent_emission_shaders[kernel] = source.find(transparent_emission_marker) != source.npos;
    lamp_sampling_flag_shaders[kernel] = source.find(lamp_sampling_flags_marker) != source.npos;
    lamp_falloff_shaders[kernel] = source.find(lamp_falloff_marker) != source.npos;
    ray_length_lamp_shaders[kernel] = source.find(ray_length_lamp_marker) != source.npos;
    aov_shaders[kernel] = source.starts_with(aov_marker);
    reflective_caustics_shaders[kernel] =
        source.find(reflective_caustics_marker) != source.npos;
    volume_geometry_shaders[kernel] = source.find(volume_geometry_marker) != source.npos;
    compact_voronoi_distance_shaders[kernel] =
        source.find(compact_voronoi_distance_marker) != source.npos;
    glass_caustics_shaders[kernel] = source.find(glass_caustics_marker) != source.npos;
    diffuse_principled_mix_shaders[kernel] =
        source.find(diffuse_principled_mix_marker) != source.npos;
    shader_loaded[kernel] = true;
    emission_delta_shaders[kernel] = source.find(emission_delta_marker) != source.npos;
    film_light_shaders[kernel] = source.find(film_light_marker) != source.npos;
    sky_world_shaders[kernel] = source.find(sky_world_marker) != source.npos;
    mapped_world_shaders[kernel] = source.find(mapped_world_marker) != source.npos;
    transparent_shaders[kernel] = source.find(transparent_marker) != source.npos;
    varying_transparent_shaders[kernel] = source.find(varying_transparent_marker) != source.npos;
    mixed_transparent_shaders[kernel] =
        source.find("// Cycles WebGPU mixed Transparent surface closures v2\n") != source.npos;
    absorption_shaders[kernel] = source.find(absorption_marker) != source.npos;
    mixed_volume_shaders[kernel] = source.find(mixed_volume_marker) != source.npos;
    additive_volume_shaders[kernel] = source.find(additive_volume_marker) != source.npos;
    extended_volume_stack_shaders[kernel] = source.find(extended_volume_stack_marker) != source.npos;
    transparent_volume_shaders[kernel] = source.find(transparent_volume_marker) != source.npos;
    varying_volume_dispatch_shaders[kernel] =
        source.find(varying_volume_dispatch_marker) != source.npos;
    volume_differential_shaders[kernel] = source.find(volume_differential_marker) != source.npos;
    principled_alpha_shaders[kernel] = source.find(principled_alpha_marker) != source.npos;
    volume_guide_surface_shaders[kernel] = source.find(volume_guide_surface_marker) != source.npos;
    portal_shaders[kernel] = source.find(portal_marker) != source.npos;
    differential_shaders[kernel] = source.find(differential_marker) != source.npos;
    camera_lens_shaders[kernel] = source.find(camera_lens_marker) != source.npos;
    mixed_lights_shaders[kernel] = source.find(mixed_lights_marker) != source.npos;
    hair_shaders[kernel] = source.find(hair_marker) != source.npos;
    subsurface_shaders[kernel] = source.find(subsurface_marker) != source.npos;
    subsurface_bump_shaders[kernel] = source.find(subsurface_bump_marker) != source.npos;
    return true;
  }

  bool supports_integer_ordered_indices(const bool guided, const bool surface_passes) const
  {
    return integer_ordered_tail_shaders[active_svm_kernel] &&
           integer_ordered_tail_shaders[light_kernel] &&
           integer_ordered_tail_shaders[update_kernel] &&
           (!guided || integer_ordered_tail_shaders[guide_kernel]) &&
           (!surface_passes || integer_ordered_tail_shaders[shade_surface_passes_kernel]);
  }

  bool prepare_svm(const bool requires_extended_svm,
                   const bool queries,
                   const bool advanced,
                   const bool ao_only,
                   const std::span<const uint32_t> packed_svm)
  {
    if (queries) {
      if (!shader_loaded[surface_query_kernel]) {
        const auto surface = load_shader("surface-query.wgsl");
        if (surface.empty() || !register_kernel(surface_query_kernel, surface, surface_bindings)) {
          return fail("Cycles WebGPU could not load triangle surface setup for SVM ray queries");
        }
      }
      if (ao_only && requires_extended_svm && !advanced) {
        if (shader_loaded[webgpu_svm_ao_textures_kernel]) {
          active_svm_kernel = webgpu_svm_ao_textures_kernel;
          return true;
        }
        const auto source = load_shader("svm-query-ao-textures.wgsl");
        if (!source.empty()) {
          if (source.find(projected_query_images_marker) == source.npos) {
            return fail("Cycles WebGPU needs a rebuilt textured Ambient Occlusion shader");
          }
          active_svm_kernel = webgpu_svm_ao_textures_kernel;
          LOG_INFO << "Cycles WebGPU compiling the textured Ambient Occlusion shader";
          return register_kernel(active_svm_kernel, source, svm_query_bindings);
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

    /* The full module includes the Gabor and advanced Voronoi kernels.
     * Skip their compile cost for ordinary procedural and image textures. */
    if (!advanced) {
      /* Scene packing validated each non-volume material in its selected closure mode.
       * Leave other scene and shader consumers on checked bytecode reads. */
      if (!absorption) {
        if (device_webgpu_filterable_texture_limit(device)) {
          const auto source = load_shader("svm-compact-textures-validated-hardware.wgsl");
          if (!source.empty() && source.find(validated_svm_marker) != source.npos &&
              source.find(hardware_svm_marker) != source.npos)
          {
            svm_image_atlas = cycles_webgpu_create_svm_image_atlas(device, packed_svm);
            if (svm_image_atlas) {
              active_svm_kernel = webgpu_svm_hardware_textures_kernel;
              LOG_INFO << "Cycles WebGPU using hardware cubic filtering and validated surface SVM";
              return shader_loaded[active_svm_kernel] ||
                     register_kernel(active_svm_kernel, source, svm_hardware_bindings);
            }
          }
        }
        const auto source = load_shader("svm-compact-textures-validated.wgsl");
        if (!source.empty() && source.find(validated_svm_marker) != source.npos) {
          active_svm_kernel = webgpu_svm_validated_textures_kernel;
          LOG_INFO << "Cycles WebGPU using validated surface SVM";
          return shader_loaded[active_svm_kernel] ||
                 register_kernel(active_svm_kernel, source, svm_bindings);
        }
      }
      if (!shader_loaded[svm_textures_kernel]) {
        const auto source = load_shader("svm-compact-textures.wgsl");
        if (!source.empty() && !register_kernel(svm_textures_kernel, source, svm_bindings)) {
          return false;
        }
      }
      if (shader_loaded[svm_textures_kernel]) {
        active_svm_kernel = svm_textures_kernel;
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
    DeviceKernel kernel;
    const char *shader_name;
    if (scattering) {
      kernel = has_transmission ? volume_transmission_light_kernel : volume_light_kernel;
      shader_name = has_transmission ? "wavefront-light-deferred-tree-volume.wgsl" :
                                       "wavefront-light-deferred-reflection-tree-volume.wgsl";
    }
    else if (hair_materials) {
      kernel = tree_hair_kernel;
      shader_name = "wavefront-light-deferred-tree-hair.wgsl";
    }
    else {
      kernel = has_transmission ? tree_transmission_kernel : tree_light_kernel;
      shader_name = has_transmission ? "wavefront-light-deferred-tree.wgsl" :
                                       "wavefront-light-deferred-reflection-tree.wgsl";
    }
    bool &triangles_supported = tree_triangles_supported[kernel];
    bool &textured_supported = tree_textured_supported[kernel];
    if (!shader_loaded[kernel]) {
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
      LOG_INFO << (light_distribution ? "Cycles WebGPU compiling the stock mixed-light distribution" :
                                       "Cycles WebGPU compiling the stock Light Tree")
               << " and world sampling shaders";
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
    if (light_distribution &&
        (!mixed_lights_shaders[kernel] || !mixed_lights_shaders[background_stage]))
    {
      return fail("Cycles WebGPU needs rebuilt direct and background shaders for the light distribution");
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
    if (volume_shadow_params_offset) {
      return varying_volume_shadow_kernel;
    }
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
      if (light_distribution && !mixed_lights_shaders[scatter_stage]) {
        return fail("Cycles WebGPU needs rebuilt guided volume shaders for the light distribution");
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
      if (light_distribution && !mixed_lights_shaders[guided_volume_background_kernel]) {
        return fail("Cycles WebGPU needs a rebuilt volume guide background for the light distribution");
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
      if (light_distribution && !mixed_lights_shaders[guided_tree_background_kernel]) {
        return fail("Cycles WebGPU needs rebuilt denoising guides for the light distribution");
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

  bool prepare_surface_roulette()
  {
    const auto source = load_shader("surface-roulette-queue.wgsl");
    if (source.empty()) {
      return true;
    }
    constexpr auto storage = WebGPUKernelBinding::Type::Storage;
    constexpr auto read_only = WebGPUKernelBinding::Type::ReadOnlyStorage;
    constexpr WebGPUKernelBinding ray = {0, storage, sizeof(uint32_t)};
    constexpr WebGPUKernelBinding queue = {
        4, storage, surface_roulette_queue_header_words * sizeof(uint32_t)};
    constexpr WebGPUKernelBinding reset[] = {ray, queue};
    constexpr WebGPUKernelBinding filter[] = {
        ray, {1, read_only, CYCLES_WEBGPU_PATH_WORDS * sizeof(uint32_t)},
        {2, read_only, 16 * sizeof(uint32_t)}, {3, read_only, 5 * sizeof(uint32_t)}, queue,
        {5, storage, CYCLES_WEBGPU_MATERIAL_WORDS * sizeof(uint32_t)}};
    constexpr WebGPUKernelBinding publish[] = {
        ray, queue, {6, storage, surface_roulette_dispatch_words * sizeof(uint32_t)}};
    surface_roulette_loaded =
        device_webgpu_register_kernel(device, webgpu_reset_surface_roulette_kernel, source, 1,
                                      reset, "reset_surface_roulette") &&
        device_webgpu_register_kernel(device, webgpu_filter_surface_roulette_kernel, source,
                                      workgroup_size, filter, "filter_surface_roulette") &&
        device_webgpu_register_kernel(device, webgpu_publish_surface_roulette_kernel, source,
                                      workgroup_size, publish, "publish_surface_roulette") &&
        device_webgpu_register_kernel(device, webgpu_restore_surface_roulette_kernel, source,
                                      workgroup_size, reset, "restore_surface_roulette");
    return surface_roulette_loaded;
  }

  bool surface_roulette_enabled() const
  {
    return surface_roulette_loaded && !portal_materials && !absorption;
  }

  bool reserve_surface_roulette(const uint32_t capacity)
  {
    if (!surface_roulette_enabled()) {
      surface_roulette_queue.free();
      surface_roulette_dispatch.free();
      return true;
    }

    const auto limits = device_webgpu_compute_limits(device);
    const uint32_t width = svm_workgroup_widths[active_svm_kernel];
    if (limits.workgroups_per_dimension == 0 || width == 0) {
      return fail("Cycles WebGPU could not initialize the surface roulette dispatch dimensions");
    }
    surface_roulette_queue.alloc_to_device(
        surface_roulette_queue_header_words + surface_roulette_queue_words_per_path * size_t(capacity));
    if (!surface_roulette_queue.device_pointer) {
      return fail("Cycles WebGPU could not allocate the surface roulette queue");
    }

    const std::array<uint32_t, surface_roulette_dispatch_words> grid = {
        0, 1, 1, limits.workgroups_per_dimension, width};
    if (!upload(device, surface_roulette_dispatch, std::span(grid))) {
      return fail("Cycles WebGPU could not initialize the surface roulette indirect dispatch");
    }
    LOG_INFO << "Cycles WebGPU compacts roulette-rejected material lanes";
    return true;
  }

  bool enqueue_compact_surface_roulette(const uint32_t count,
                                        const WebGPUDispatch &setup,
                                        const WebGPUDispatch &evaluate)
  {
    auto &paths = path_state.words();
    const auto reset = bind(rays, surface_roulette_queue);
    const auto filter = bind(rays, paths, points, scene.surface, surface_roulette_queue, materials);
    const auto publish = bind(rays, surface_roulette_queue, surface_roulette_dispatch);
    const auto sort = bind(rays, points, scene.svm);

    const WebGPUDispatch before[] = {
        setup,
        {webgpu_reset_surface_roulette_kernel, reset, 1},
        {webgpu_filter_surface_roulette_kernel, filter, count},
        {webgpu_publish_surface_roulette_kernel, publish, count},
        {webgpu_sort_surface_paths_kernel, sort, WebGPUWavefrontQueue::surface_sort_work_size(count)},
    };
    const size_t before_count = sort_surface_shaders && count > 1 ? 5 : 4;
    if (!wavefront_queue.enqueue_stage(DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE,
                                       int(path_capacity), rays, paths,
                                       std::span(before).first(before_count))) {
      return false;
    }

    // The published indirect grid and filtered queue belong only to SVM.
    // Restore the complete shading queue before forward lighting and retirement.
    const WebGPUDispatch after[] = {
        {evaluate.kernel, evaluate.args, count, surface_roulette_dispatch.device_pointer,
         1, evaluate.texture_atlas},
        {webgpu_restore_surface_roulette_kernel, reset, count},
    };
    return device_webgpu_enqueue(device, count, after);
  }

  bool prepare_compact_volume_shadow_queue()
  {
    if (compact_volume_shadow_queue_loaded) {
      return true;
    }
    const auto source = load_shader("volume-shadow-queue.wgsl");
    constexpr std::string_view linear_grid_marker = "// Cycles WebGPU linear workgroup grid\n";
    compact_volume_shadow_linear_grid = source.find(linear_grid_marker) != source.npos;
    auto reset_bindings = std::to_array(reset_volume_shadow_bindings);
    auto compact_bindings = compact_volume_shadow_bindings;
    auto guided_bindings = compact_guided_volume_shadow_bindings;
    if (compact_volume_shadow_linear_grid) {
      reset_bindings[1].minimum_size = 4 * sizeof(uint32_t);
      compact_bindings[2] = reset_bindings[1];
      guided_bindings[2] = reset_bindings[1];
    }
    compact_volume_shadow_queue_loaded =
        !source.empty() &&
        device_webgpu_register_kernel(device, webgpu_reset_volume_shadow_queue,
                                      source, 1, reset_bindings,
                                      "reset_shadow_queue") &&
        device_webgpu_register_kernel(device, webgpu_compact_volume_shadow_queue,
                                      source, workgroup_size, compact_bindings,
                                      "compact_shadow_rays") &&
        device_webgpu_register_kernel(device, webgpu_compact_guided_volume_shadow_queue,
                                      source, workgroup_size, guided_bindings,
                                      "compact_guided_volume_shadow_rays") &&
        (!compact_volume_shadow_linear_grid ||
         device_webgpu_register_kernel(device, webgpu_finish_volume_shadow_grid_kernel,
                                       source, 1, reset_bindings, "finish_shadow_grid"));
    return compact_volume_shadow_queue_loaded;
  }

  bool enqueue_compact_volume_shadow(const uint32_t count,
                                     const DeviceKernelArguments &shadow,
                                     const bool accumulate_guides,
                                     const WebGPUDispatch *after = nullptr)
  {
    const auto reset = bind(rays, volume_shadow_indirect);
    const auto compact = accumulate_guides ?
                             bind(rays, shadow_rays, volume_shadow_indirect,
                                  denoising, scene.surface) :
                             bind(rays, shadow_rays, volume_shadow_indirect);
    const WebGPUDispatch prepare[] = {
        {webgpu_reset_volume_shadow_queue, reset, 1},
        {accumulate_guides ? webgpu_compact_guided_volume_shadow_queue :
                             webgpu_compact_volume_shadow_queue, compact, count},
        {webgpu_finish_volume_shadow_grid_kernel, reset, 1},
    };
    const WebGPUDispatch trace{
        varying_volume_shadow_kernel, shadow, count, volume_shadow_indirect.device_pointer};
    const size_t prepare_count = compact_volume_shadow_linear_grid ? 3 : 2;
    if (!device_webgpu_enqueue(device, count, std::span(prepare).first(prepare_count))) {
      return false;
    }
    if (after) {
      const WebGPUDispatch stages[] = {trace, *after};
      return device_webgpu_enqueue(device, count, stages);
    }
    return device_webgpu_enqueue(device, count, {&trace, 1});
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
  if (!p.prepare_surface_roulette()) {
    return p.fail("Cycles WebGPU could not prepare surface roulette compaction");
  }
  if (!p.wavefront_queue.prepare_surface_sort(p.load_shader("wavefront-sort.wgsl"))) {
    return p.fail("Cycles WebGPU could not compile the shader sorting kernel");
  }
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
                                        bool advanced_svm,
                                        bool ao_only_queries,
                                        bool residual_displacement_bump,
                                        bool paired_principled_layers,
                                        bool single_principled_coat,
                                        bool coated_principled_emission,
                                        bool compact_voronoi_distance,
                                        bool has_mixed_diffuse)
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
  const uint32_t extended = light.size() >= light_header ? light[CYCLES_WEBGPU_LIGHT_EXTENDED] : 0;
  const uint32_t area = extended & ~CYCLES_WEBGPU_LIGHT_VOLUME_DIFFERENTIALS;
  const bool point_distribution = light.size() >= light_header &&
                                  light[0] == CYCLES_WEBGPU_POINT_LIGHTS_MARKER;
  const bool light_tree = light.size() >= light_header &&
                          light[0] == CYCLES_WEBGPU_LIGHT_TREE_MARKER;
  const size_t rr_sampling = light.size() >= light_header ?
                                 light[CYCLES_WEBGPU_LIGHT_RR_SAMPLING] : 0;
  size_t light_prefix = area ? area : (rr_sampling ? rr_sampling : light.size());
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
      (rr_sampling != 0 &&
       (rr_sampling < light_prefix || rr_sampling > light.size() ||
        light.size() - rr_sampling < CYCLES_WEBGPU_LIGHT_SAMPLING_WORDS)) ||
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
    const uint32_t world_mode = light[area + CYCLES_WEBGPU_WORLD_INTERPOLATION];
    if (world_mode & CYCLES_WEBGPU_WORLD_SVM_FLAG) {
      return p.fail("Cycles WebGPU needs mixed-light scheduling for mapped world SVM");
    }
    const bool sky = world_mode & CYCLES_WEBGPU_WORLD_SKY_FLAG;
    const uint64_t world_words = 3 * uint64_t(light[area + CYCLES_WEBGPU_WORLD_WIDTH]) *
                                     light[area + CYCLES_WEBGPU_WORLD_HEIGHT] +
                                 (sky ? CYCLES_WEBGPU_SKY_HEADER_WORDS : 0);
    const size_t sampling = light[area + CYCLES_WEBGPU_LIGHT_SAMPLING];
    if (sampling != area + CYCLES_WEBGPU_EXTENDED_LIGHT_WORDS + world_words ||
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

  if (light_tree && (area || !valid_mixed_lights(light, light_prefix))) {
    return p.fail("Cycles WebGPU needs complete light selection and world sampling buffers");
  }

  p.scene_uploaded = false;
  p.film_ready = false;
  p.batch_active = false;
  device_webgpu_free_texture_atlas(p.device, p.svm_image_atlas);
  p.svm_image_atlas = 0;
  p.transparent_materials = CYCLES_WEBGPU_SURFACE_HAS_TRANSPARENCY(surface.front());
  p.active_shadow_kernel = shadow_kernel;
  const bool varying_volume_shadows =
      surface.front() == CYCLES_WEBGPU_SURFACE_VOLUME_SHADOW_OFFSETS ||
      surface.front() == CYCLES_WEBGPU_SURFACE_VOLUME_SHADOW_TANGENT_OFFSETS;
  p.absorption = CYCLES_WEBGPU_SURFACE_HAS_VOLUME(surface.front());
  p.scattering = false;
  p.mixed_volume = false;
  p.additive_volume = false;
  bool principled_alpha = false;
  bool cubic_transparent_shadows = false;
  bool shadow_extended_svm = false;
  p.extended_volume_stack = false;
  p.volume_shadow_params_offset = 0;
  p.compact_volume_shadows = false;
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
    size_t end = surface.size();
    if (varying_volume_shadows) {
      if (surface.size() < CYCLES_WEBGPU_SURFACE_VOLUME_SHADOW_OFFSETS) {
        return p.fail("Cycles WebGPU needs varying volume shadow metadata");
      }
      end = surface[CYCLES_WEBGPU_SURFACE_VOLUME_SHADOW_PARAMS];
      if (end < offset || end > surface.size() ||
          surface.size() - end != CYCLES_WEBGPU_VOLUME_SHADOW_PARAMS_WORDS)
      {
        return p.fail("Cycles WebGPU has invalid varying volume shadow metadata");
      }
      p.volume_shadow_params_offset = end;
    }
    const size_t words = end - offset;
    const size_t count = surface[CYCLES_WEBGPU_SURFACE_VOLUME_COUNT];
    p.scattering = count != 0;
    p.transparent_materials = surface[CYCLES_WEBGPU_SURFACE_TRANSPARENT_OFFSET] != 0;
    p.extended_volume_stack = count > CYCLES_WEBGPU_COMPACT_VOLUME_MEDIA;
    p.mixed_volume = surface[CYCLES_WEBGPU_SURFACE_MIXED_VOLUME] != 0;
    if ((!p.scattering && (p.mixed_volume || words != CYCLES_WEBGPU_ABSORPTION_WORDS)) ||
        (p.scattering && (!light_tree || !p.volume_queue_supported ||
                          count > CYCLES_WEBGPU_MAX_VOLUME_MEDIA ||
                          count * CYCLES_WEBGPU_VOLUME_SCATTER_WORDS > words)))
    {
      return p.fail("Cycles WebGPU needs compatible volume data and light sampling");
    }
    constexpr size_t grid_header_words = CYCLES_WEBGPU_GRID_ATTRIBUTES +
                                         CYCLES_WEBGPU_GRID_ATTRIBUTE_COUNT *
                                             CYCLES_WEBGPU_GRID_ATTRIBUTE_WORDS;
    const size_t records_end = offset + count * CYCLES_WEBGPU_VOLUME_SCATTER_WORDS;
    for (size_t record = offset; record < records_end; record += CYCLES_WEBGPU_VOLUME_SCATTER_WORDS) {
      p.additive_volume |=
          (surface[record + CYCLES_WEBGPU_VOLUME_OPCODE] & CYCLES_WEBGPU_VOLUME_ADD_ABSORPTION) != 0;
      const size_t grid = surface[record + CYCLES_WEBGPU_VOLUME_GRID_OFFSET];
      if (grid) {
        if (grid < records_end || grid > end || grid_header_words > end - grid) {
          return p.fail("Cycles WebGPU has invalid volume grid metadata");
        }
        p.grid_volume = true;
      }
    }
    p.active_shadow_kernel = varying_volume_shadows ? varying_volume_shadow_kernel :
                              p.grid_volume ? grid_volume_extinction_shadow_kernel :
                              p.scattering ? volume_extinction_shadow_kernel :
                                             absorption_shadow_kernel;
  }
  if (p.transparent_materials) {
    if (surface.size() < CYCLES_WEBGPU_SURFACE_TRANSPARENT_OFFSETS) {
      return p.fail("Cycles WebGPU needs transparent shadow metadata");
    }
    const size_t offset = surface[CYCLES_WEBGPU_SURFACE_TRANSPARENT_OFFSET];
    const size_t shader_count = svm[1] & CYCLES_WEBGPU_SVM_SHADER_COUNT;
    const size_t end = p.absorption ? surface[CYCLES_WEBGPU_SURFACE_ABSORPTION_OFFSET] :
                                     surface.size();
    if (offset < surface[CYCLES_WEBGPU_SURFACE_SHADER_PASS_OFFSET] ||
        offset > end ||
        end - offset != CYCLES_WEBGPU_TRANSPARENT_SHADOW_HEADER_WORDS + 3 * shader_count)
    {
      return p.fail("Cycles WebGPU has invalid transparent shadow metadata");
    }
    const uint32_t shadow_flags = surface[offset + 1];
    principled_alpha = (shadow_flags & CYCLES_WEBGPU_TRANSPARENT_SHADOW_PRINCIPLED_ALPHA) != 0;
    cubic_transparent_shadows = (shadow_flags & CYCLES_WEBGPU_TRANSPARENT_SHADOW_CUBIC) != 0;
    shadow_extended_svm = (shadow_flags & CYCLES_WEBGPU_TRANSPARENT_SHADOW_EXTENDED_SVM) != 0;
    if (p.absorption &&
        (bool(shadow_flags & CYCLES_WEBGPU_TRANSPARENT_SHADOW_VARYING) != varying_volume_shadows ||
         (varying_volume_shadows && p.grid_volume)))
    {
      return p.fail("Cycles WebGPU has unsupported varying volume shadow metadata");
    }
    if (!p.absorption) {
      p.active_shadow_kernel = (shadow_flags & CYCLES_WEBGPU_TRANSPARENT_SHADOW_VARYING) ?
                                   varying_shadow_kernel : transparent_shadow_kernel;
    }
  }
  p.aov_materials = (svm[1] & CYCLES_WEBGPU_SVM_AOV) != 0;
  p.portal_materials = (svm[1] & CYCLES_WEBGPU_SVM_RAY_PORTAL) != 0;
  p.ray_differentials = (svm[1] & CYCLES_WEBGPU_SVM_NEEDS_DIFFERENTIALS) != 0;
  p.surface_bump = (svm[1] & CYCLES_WEBGPU_SVM_BUMP) != 0;
  p.standalone_closures = (svm[1] & CYCLES_WEBGPU_SVM_STANDALONE) != 0;
  p.mixed_diffuse = has_mixed_diffuse;
  p.hair_materials = (svm[1] & CYCLES_WEBGPU_SVM_HAIR) != 0;
  p.mixed_transparent_materials = (svm[1] & CYCLES_WEBGPU_SVM_MIXED_TRANSPARENT) != 0;
  p.subsurface_materials = has_subsurface;
  const bool queries = (svm[1] & CYCLES_WEBGPU_SVM_QUERIES) != 0;
  p.triangle_tangents = p.hair_materials || p.surface_bump;
  if (p.triangle_tangents != CYCLES_WEBGPU_SURFACE_HAS_HAIR(surface.front())) {
    return p.fail("Cycles WebGPU has invalid triangle differential metadata");
  }
  if (!p.prepare_svm((svm[1] & CYCLES_WEBGPU_SVM_EXTENDED) != 0,
                     queries,
                     advanced_svm,
                     ao_only_queries,
                     svm))
  {
    return p.fail("Cycles WebGPU could not load the SVM material shader");
  }
  const size_t shader_flags = surface[4];
  const size_t shader_count = svm[1] & CYCLES_WEBGPU_SVM_SHADER_COUNT;
  if (shader_flags > surface.size() || shader_count > surface.size() - shader_flags) {
    return p.fail("Cycles WebGPU has invalid shader flag offsets");
  }
  p.sort_surface_shaders = p.wavefront_queue.use_surface_sort(shader_count);
  if (p.sort_surface_shaders) {
    LOG_INFO << "Cycles WebGPU sorting surface paths by shader";
  }
  p.ordered_closures = std::ranges::any_of(surface.subspan(shader_flags, shader_count),
                                           [](const uint32_t flags) {
    return (flags & CYCLES_WEBGPU_SHADER_ORDERED_CLOSURES) != 0u;
  });
  p.ordered_transparency = shader_count != 0 &&
      (surface[shader_flags] & CYCLES_WEBGPU_SHADER_ORDERED_TRANSPARENCY) != 0u;
  p.closure_compositions = shader_count != 0 ?
                               surface[shader_flags] & closure_composition_mask : 0u;
  p.paired_principled_layers = paired_principled_layers;
  p.single_principled_coat = single_principled_coat;
  if (p.paired_principled_layers && !p.paired_principled_layer_shaders[p.active_svm_kernel]) {
    return p.fail("Cycles WebGPU needs a rebuilt SVM shader for paired Principled layers");
  }
  if (p.single_principled_coat && !p.single_principled_coat_shaders[p.active_svm_kernel]) {
    return p.fail("Cycles WebGPU needs a rebuilt SVM shader for single Principled Coat");
  }
  if (coated_principled_emission && !p.coated_principled_emission_shaders[p.active_svm_kernel]) {
    return p.fail("Cycles WebGPU needs a rebuilt SVM shader for coated Principled emission");
  }
  if (p.closure_compositions && !p.closure_composition_shaders[p.active_svm_kernel]) {
    return p.fail("Cycles WebGPU needs a rebuilt SVM shader for these closure combinations");
  }
  p.independent_mixed_bump = shader_flags < surface.size() &&
      (surface[shader_flags] & CYCLES_WEBGPU_SHADER_INDEPENDENT_MIXED_BUMP) != 0u;
  p.glossy_displacement_bump = shader_count != 0 &&
      (surface[shader_flags] & CYCLES_WEBGPU_SHADER_GLOSSY_DISPLACEMENT_BUMP) != 0u;
  p.mixed_displacement_bump = shader_count != 0 &&
      (surface[shader_flags] & CYCLES_WEBGPU_SHADER_MIXED_DISPLACEMENT_BUMP) != 0u;
  p.glossy_transparent = shader_count != 0 &&
      (surface[shader_flags] & CYCLES_WEBGPU_SHADER_GLOSSY_TRANSPARENT) != 0u;
  const bool transparent_emission = shader_count != 0 &&
      (surface[shader_flags] & CYCLES_WEBGPU_SHADER_TRANSPARENT_EMISSION) != 0u;
  if (p.glossy_displacement_bump && !p.glossy_displacement_bump_shaders[p.active_svm_kernel]) {
    return p.fail("Cycles WebGPU needs a rebuilt SVM shader for Glossy displacement Bump");
  }
  if (p.mixed_displacement_bump && !p.mixed_displacement_bump_shaders[p.active_svm_kernel]) {
    return p.fail("Cycles WebGPU needs a rebuilt SVM shader for Diffuse and Glossy displacement Bump");
  }
  if (residual_displacement_bump && !p.residual_displacement_bump_shaders[p.active_svm_kernel]) {
    return p.fail("Cycles WebGPU needs a rebuilt SVM shader for residual displacement bump");
  }
  if (compact_voronoi_distance && !p.compact_voronoi_distance_shaders[p.active_svm_kernel]) {
    return p.fail("Cycles WebGPU needs a rebuilt SVM shader for compact Voronoi Distance");
  }
  if (shader_flags < surface.size() &&
      (surface[shader_flags] & CYCLES_WEBGPU_SHADER_NO_REFLECTIVE_CAUSTICS) != 0u)
  {
    if (!p.reflective_caustics_shaders[p.active_svm_kernel]) {
      return p.fail("Cycles WebGPU needs rebuilt SVM shaders for disabled reflective caustics");
    }
    if (light_prefix == glass_lighting_words && !p.glass_caustics_shaders[p.active_svm_kernel]) {
      return p.fail("Cycles WebGPU needs rebuilt SVM shaders for disabled glass caustics");
    }
  }
  if (shader_flags < surface.size() &&
      (surface[shader_flags] & CYCLES_WEBGPU_SHADER_DIFFUSE_PRINCIPLED_MIX) != 0u &&
      !p.diffuse_principled_mix_shaders[p.active_svm_kernel])
  {
    return p.fail("Cycles WebGPU needs rebuilt SVM shaders for diffuse Principled mixtures");
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
  p.light_distribution = light_tree && light[light_prefix + CYCLES_WEBGPU_TREE_NODE_COUNT] == 0;
  p.lamp_falloff = light_tree && light[CYCLES_WEBGPU_TREE_LAMP_FALLOFF_OFFSET] != 0;
  p.ray_length_lamps = false;
  if (p.lamp_falloff) {
    const size_t lamp_count = light[light_prefix + CYCLES_WEBGPU_TREE_LAMP_COUNT];
    const size_t records = light[CYCLES_WEBGPU_TREE_LAMP_FALLOFF_OFFSET];
    for (size_t lamp = 0; lamp < lamp_count; ++lamp) {
      p.ray_length_lamps |=
          light[records + lamp * CYCLES_WEBGPU_LAMP_FALLOFF_WORDS +
                CYCLES_WEBGPU_LAMP_FALLOFF_MODE] >= CYCLES_WEBGPU_LAMP_RAY_LENGTH;
    }
  }
  p.sky_world = light_tree ? (light[light_prefix + CYCLES_WEBGPU_TREE_WORLD_INTERPOLATION] &
                             CYCLES_WEBGPU_WORLD_SKY_FLAG) != 0 :
                            area && (light[area + CYCLES_WEBGPU_WORLD_INTERPOLATION] &
                                     CYCLES_WEBGPU_WORLD_SKY_FLAG);
  p.mapped_world = light_tree && (light[light_prefix + CYCLES_WEBGPU_TREE_WORLD_INTERPOLATION] &
                                  CYCLES_WEBGPU_WORLD_SVM_FLAG);
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
  if (coated_principled_emission && p.textured_lights &&
      !p.coated_principled_emission_shaders[sampled_emission_kernel])
  {
    return p.fail("Cycles WebGPU needs a rebuilt sampled-emission shader for Principled Coat");
  }
  if ((p.closure_compositions & principled_compositions) &&
      (!p.closure_composition_shaders[p.light_kernel] ||
       !p.closure_composition_shaders[p.update_kernel]))
  {
    return p.fail("Cycles WebGPU needs rebuilt lighting and path shaders for Principled mixtures");
  }
  if (p.paired_principled_layers &&
      (!p.paired_principled_layer_shaders[p.light_kernel] ||
       !p.paired_principled_layer_shaders[p.update_kernel]))
  {
    return p.fail("Cycles WebGPU needs rebuilt lighting and path shaders for paired Principled layers");
  }
  if (p.single_principled_coat &&
      (!p.single_principled_coat_shaders[p.light_kernel] ||
       !p.single_principled_coat_shaders[p.update_kernel]))
  {
    return p.fail("Cycles WebGPU needs rebuilt lighting and path shaders for single Principled Coat");
  }
  if (light_tree && !p.lamp_sampling_flag_shaders[p.light_kernel]) {
    const size_t lamps = light[light_prefix + CYCLES_WEBGPU_TREE_LAMP_COUNT];
    const size_t lamp_offset = light[light_prefix + CYCLES_WEBGPU_TREE_LAMPS];
    for (size_t lamp = 0; lamp < lamps; ++lamp) {
      const size_t record = lamp_offset + lamp * CYCLES_WEBGPU_TREE_LAMP_WORDS;
      const uint32_t shader = light[record + CYCLES_WEBGPU_TREE_LAMP_SHADER];
      const bool area_lamp = light[record + CYCLES_WEBGPU_TREE_LAMP_TYPE] == LIGHT_AREA;
      if (!(shader & SHADER_CAST_SHADOW) || (area_lamp && !(shader & SHADER_USE_MIS))) {
        return p.fail("Cycles WebGPU needs a rebuilt direct lighting shader for lamp sampling flags");
      }
    }
  }
  if (!p.supports_lamp_emission(p.light_kernel) ||
      !p.supports_lamp_emission(p.active_background_kernel))
  {
    return p.fail("Cycles WebGPU needs rebuilt direct and forward shaders for analytic lamp emission");
  }
  if (p.sky_world &&
      (!p.sky_world_shaders[p.light_kernel] || !p.sky_world_shaders[p.active_background_kernel]))
  {
    return p.fail("Cycles WebGPU needs rebuilt direct and background shaders for the procedural sky");
  }
  if (p.mapped_world &&
      (!p.mapped_world_shaders[p.light_kernel] ||
       !p.mapped_world_shaders[p.active_background_kernel]))
  {
    return p.fail("Cycles WebGPU needs rebuilt direct and background shaders for mapped world SVM");
  }
  if (p.subsurface_materials &&
      (!p.subsurface_shaders[p.active_svm_kernel] || !p.subsurface_shaders[p.update_kernel] ||
       !p.subsurface_shaders[subsurface_kernel]))
  {
    return p.fail("Cycles WebGPU needs rebuilt shaders for subsurface scattering");
  }
  if (p.subsurface_materials && p.surface_bump &&
      (!p.subsurface_bump_shaders[p.active_svm_kernel] ||
       !p.subsurface_bump_shaders[subsurface_kernel]))
  {
    return p.fail("Cycles WebGPU needs rebuilt shaders for subsurface bump normals");
  }
  if (p.hair_materials &&
      (!p.hair_shaders[p.active_svm_kernel] || !p.hair_shaders[p.light_kernel] ||
       !p.hair_shaders[p.update_kernel] ||
       !p.hair_shaders[queries ? surface_query_kernel : surface_kernel]))
  {
    return p.fail("Cycles WebGPU needs rebuilt SVM, surface, path and light shaders for Hair");
  }
  if (p.absorption) {
    if (!p.deferred_shadow || !p.absorption_shaders[p.update_kernel] ||
        (p.ray_differentials && !p.scattering) ||
        (p.transparent_materials && !p.scattering) ||
        p.portal_materials || p.hair_materials || p.aov_materials)
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
    bool prepared;
    if (p.scattering) {
      prepared = load_volume_stage(volume_interval_kernel, "volume-interval.wgsl", absorption_bindings) &&
                 load_volume_stage(p.volume_scatter_stage(false),
                                   p.grid_volume ? "volume-scatter-grid.wgsl" : "volume-scatter.wgsl",
                                   volume_scatter_bindings);
      if (varying_volume_shadows) {
        prepared = prepared && load_volume_stage(varying_volume_shadow_kernel,
                                                 "volume-scatter-shadow-guides-materials.wgsl",
                                                 varying_volume_shadow_bindings);
        if (prepared && p.compact_volume_shadow_shader) {
          prepared = p.prepare_compact_volume_shadow_queue();
          p.compact_volume_shadows = prepared;
        }
      }
      else {
        prepared = prepared && load_volume_stage(
            p.volume_scatter_shadow_stage(false),
            p.grid_volume ? "volume-scatter-shadow-grid.wgsl" : "volume-scatter-shadow.wgsl",
            volume_scatter_shadow_bindings) &&
            load_volume_stage(p.active_shadow_kernel,
                              p.grid_volume ? "volume-extinction-shadow-grid.wgsl" :
                                              "volume-extinction-shadow.wgsl",
                              volume_extinction_shadow_bindings);
      }
    }
    else {
      prepared = load_volume_stage(absorption_kernel, "absorption-compact.wgsl", absorption_bindings) &&
                 load_volume_stage(absorption_shadow_kernel, "absorption-shadow.wgsl",
                                   absorption_shadow_bindings);
    }
    if (!prepared) {
      return p.fail("Cycles WebGPU could not compile volume integration and shadows");
    }
    if (p.extended_volume_stack &&
        (!p.extended_volume_stack_shaders[volume_interval_kernel] ||
         !p.extended_volume_stack_shaders[p.volume_scatter_stage(false)] ||
         !p.extended_volume_stack_shaders[p.volume_scatter_shadow_stage(false)] ||
         !p.extended_volume_stack_shaders[p.active_shadow_kernel]))
    {
      return p.fail("Cycles WebGPU needs rebuilt shaders for the extended volume stack");
    }
    if (varying_volume_shadows &&
        !p.varying_volume_dispatch_shaders[p.volume_scatter_stage(false)])
    {
      return p.fail("Cycles WebGPU needs rebuilt shaders for varying volume shadows");
    }
    if (p.ray_differentials &&
        (!(extended & CYCLES_WEBGPU_LIGHT_VOLUME_DIFFERENTIALS) ||
         !p.volume_differential_shaders[queries ? surface_query_kernel : surface_kernel] ||
         !p.volume_differential_shaders[p.light_kernel] ||
         !p.volume_differential_shaders[p.active_background_kernel] ||
         !p.volume_differential_shaders[p.volume_scatter_stage(false)]))
    {
      return p.fail("Cycles WebGPU needs rebuilt shaders for volume ray differentials");
    }
    if (p.transparent_materials &&
        (!p.transparent_volume_shaders[p.volume_scatter_shadow_stage(false)] ||
         !p.transparent_volume_shaders[p.active_shadow_kernel]))
    {
      return p.fail("Cycles WebGPU needs rebuilt shaders for transparent volume shadows");
    }
    if (p.additive_volume &&
        (!p.additive_volume_shaders[p.volume_scatter_stage(false)] ||
         !p.additive_volume_shaders[p.volume_scatter_shadow_stage(false)] ||
         !p.additive_volume_shaders[p.active_shadow_kernel]))
    {
      return p.fail("Cycles WebGPU needs rebuilt shaders for literal additive volumes");
    }
    if (p.mixed_volume &&
        (!p.mixed_volume_shaders[volume_interval_kernel] ||
         !p.mixed_volume_shaders[p.volume_scatter_stage(false)] ||
         !p.mixed_volume_shaders[p.active_shadow_kernel] ||
         !p.mixed_volume_shaders[p.light_kernel] ||
         !p.mixed_volume_shaders[p.active_svm_kernel] ||
         !load_volume_stage(volume_transition_kernel, "volume-transition.wgsl",
                            volume_transition_bindings)))
    {
      return p.fail("Cycles WebGPU needs rebuilt shaders for mixed surface volume boundaries");
    }
    if (p.scattering && p.light_distribution &&
        !p.mixed_lights_shaders[p.volume_scatter_stage(false)])
    {
      return p.fail("Cycles WebGPU needs rebuilt volume shaders for the light distribution");
    }
    if (p.scattering && !p.supports_lamp_emission(p.volume_scatter_stage(false))) {
      return p.fail("Cycles WebGPU needs a rebuilt volume shader for analytic lamp emission");
    }
    if (p.sky_world && p.scattering && !p.sky_world_shaders[p.volume_scatter_stage(false)]) {
      return p.fail("Cycles WebGPU needs a rebuilt volume shader for the procedural sky");
    }
    if (p.mapped_world && p.scattering && !p.mapped_world_shaders[p.volume_scatter_stage(false)]) {
      return p.fail("Cycles WebGPU needs a rebuilt volume shader for mapped world SVM");
    }
  }
  if (p.ray_differentials &&
      (!p.differential_shaders[camera_kernel] || !p.differential_shaders[p.active_svm_kernel] ||
       !p.differential_shaders[p.update_kernel]))
  {
    return p.fail("Cycles WebGPU needs ray differentials in the camera, SVM and path shaders");
  }
  if (p.transparent_materials) {
    if (!p.deferred_shadow || !p.transparent_shaders[p.active_svm_kernel] ||
        !p.transparent_shaders[p.update_kernel] ||
        !p.transparent_shaders[p.active_background_kernel])
    {
      return p.fail("Cycles WebGPU needs rebuilt SVM, path, and background shaders for transparency");
    }
    const bool varying = p.active_shadow_kernel == varying_shadow_kernel || varying_volume_shadows;
    if (varying && (!p.varying_transparent_shaders[p.active_svm_kernel] ||
                    !p.varying_transparent_shaders[p.update_kernel]))
    {
      return p.fail("Cycles WebGPU needs rebuilt SVM and path shaders for varying transparency");
    }
    if (!p.absorption && !p.shader_loaded[p.active_shadow_kernel]) {
      const auto source = p.load_shader(varying ? "intersect-shadow-transparent-materials.wgsl" :
                                                 "intersect-shadow-transparent.wgsl");
      const auto bindings = varying ? std::span<const WebGPUKernelBinding>(varying_shadow_bindings) :
                                      std::span<const WebGPUKernelBinding>(transparent_shadow_bindings);
      if (source.empty() || !p.register_kernel(p.active_shadow_kernel, source, bindings)) {
        return p.fail("Cycles WebGPU could not compile the transparent shadow shader");
      }
      if (varying) {
        p.varying_ao_shadow_supported = source.find(varying_ao_shadow_marker) != source.npos;
      }
    }
    if (principled_alpha &&
        (!p.principled_alpha_shaders[p.active_svm_kernel] ||
         (varying && !p.principled_alpha_shaders[p.active_shadow_kernel])))
    {
      return p.fail("Cycles WebGPU needs rebuilt material and shadow shaders for Principled Alpha");
    }
    if (cubic_transparent_shadows && !p.cubic_transparent_shadow_shaders[p.active_shadow_kernel]) {
      return p.fail("Cycles WebGPU needs a rebuilt varying shadow shader for cubic image filtering");
    }
    if (shadow_extended_svm && !p.shadow_extended_svm_shaders[p.active_shadow_kernel]) {
      return p.fail("Cycles WebGPU needs a rebuilt varying shadow shader for material nodes");
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
  if (p.glossy_transparent &&
      (!p.glossy_transparent_shaders[p.active_svm_kernel] ||
       !p.glossy_transparent_shaders[p.update_kernel] ||
       !p.glossy_transparent_shaders[p.light_kernel] ||
       ((p.active_shadow_kernel == varying_shadow_kernel || varying_volume_shadows) &&
        !p.glossy_transparent_shaders[p.active_shadow_kernel])))
  {
    return p.fail("Cycles WebGPU needs rebuilt shaders for Glossy and Transparent");
  }
  if (transparent_emission &&
      (!p.transparent_emission_shaders[p.active_svm_kernel] ||
       (p.textured_lights && !p.transparent_emission_shaders[sampled_emission_kernel]) ||
       ((p.active_shadow_kernel == varying_shadow_kernel || varying_volume_shadows) &&
        !p.transparent_emission_shaders[p.active_shadow_kernel])))
  {
    return p.fail("Cycles WebGPU needs rebuilt shaders for Transparent emission");
  }
  constexpr uint32_t transparent_compositions = CYCLES_WEBGPU_SHADER_PRINCIPLED_TRANSLUCENT |
                                                CYCLES_WEBGPU_SHADER_ORDERED_GLOSSY_MIX;
  if ((p.closure_compositions & transparent_compositions) &&
      (p.active_shadow_kernel == varying_shadow_kernel || varying_volume_shadows) &&
      !p.closure_composition_shaders[p.active_shadow_kernel])
  {
    return p.fail("Cycles WebGPU needs a rebuilt varying shadow shader for these closure combinations");
  }
  if (p.standalone_closures &&
      (!p.standalone_shaders[p.active_svm_kernel] || !p.standalone_shaders[p.light_kernel] ||
       !p.standalone_shaders[p.update_kernel] ||
       (p.textured_lights && !p.standalone_shaders[sampled_emission_kernel])))
  {
    return p.fail("Cycles WebGPU needs rebuilt shaders for standalone surface closures");
  }
  if (p.mixed_diffuse &&
      (!p.mixed_diffuse_shaders[p.active_svm_kernel] ||
       !p.mixed_diffuse_shaders[p.light_kernel] || !p.mixed_diffuse_shaders[p.update_kernel] ||
       (p.textured_lights && !p.mixed_diffuse_shaders[sampled_emission_kernel])))
  {
    return p.fail("Cycles WebGPU needs rebuilt shaders for mixed Diffuse closures");
  }
  if (p.independent_mixed_bump &&
      (!p.independent_mixed_bump_shaders[p.active_svm_kernel] ||
       !p.independent_mixed_bump_shaders[p.light_kernel] ||
       !p.independent_mixed_bump_shaders[p.update_kernel] ||
       (p.textured_lights && !p.independent_mixed_bump_shaders[sampled_emission_kernel])))
  {
    return p.fail("Cycles WebGPU needs rebuilt shaders for independent mixed Bump normals");
  }
  if (p.ordered_closures &&
      (!p.ordered_closure_shaders[p.active_svm_kernel] ||
       !p.ordered_closure_shaders[p.light_kernel] ||
       !p.ordered_closure_shaders[p.update_kernel]))
  {
    return p.fail("Cycles WebGPU needs rebuilt shaders for ordered closures");
  }
  if (p.ordered_transparency &&
      (!p.ordered_transparency_shaders[p.active_svm_kernel] ||
       !p.ordered_transparency_shaders[p.update_kernel] ||
       !p.ordered_transparency_shaders[p.light_kernel] ||
       ((p.active_shadow_kernel == varying_shadow_kernel || varying_volume_shadows) &&
        !p.ordered_transparency_shaders[p.active_shadow_kernel])))
  {
    return p.fail("Cycles WebGPU needs rebuilt shaders for ordered Transparent and Translucent");
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
                                 std::span<const float, CYCLES_WEBGPU_CAMERA_LENS_WORDS> camera_lens,
                                 std::span<const uint32_t, 16> denoising_config,
                                 const BufferParams &params,
                                 const size_t film_words)
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
  const int emission_offset = params.get_pass_offset(PASS_EMISSION);
  p.emission_pass_active = emission_offset != PASS_UNUSED;
  if (p.emission_pass_active && p.absorption) {
    return p.fail("Cycles WebGPU Emission needs volume contribution pass transport");
  }
  constexpr std::array surface_passes = {
      PASS_NORMAL, PASS_DIFFUSE_COLOR, PASS_GLOSSY_COLOR, PASS_TRANSMISSION_COLOR, PASS_AO};
  constexpr std::array geometry_passes = {PASS_DEPTH, PASS_OBJECT_ID, PASS_MATERIAL_ID, PASS_POSITION};
  std::array<uint32_t, 8> surface_offsets{};
  p.ao_pass_active = params.get_pass_offset(PASS_AO) != PASS_UNUSED;
  surface_offsets[5] = denoising_config[3]; /* Ambient occlusion distance. */
  surface_offsets[6] = denoising_config[7]; /* Alpha threshold. */
  surface_offsets[7] = uint32_t(p.transparent_materials);
  p.surface_passes_active = p.transparent_materials &&
                            std::ranges::any_of(geometry_passes, [&](const PassType type) {
                              return params.get_pass_offset(type) != PASS_UNUSED;
                            });
  for (size_t i = 0; i < surface_passes.size(); i++) {
    const int offset = params.get_pass_offset(surface_passes[i]);
    if (offset != PASS_UNUSED) {
      if (offset < 0 || offset > params.pass_stride - 3) {
        return p.fail("Cycles WebGPU received invalid filtered surface pass offsets");
      }
      if (p.absorption) {
        return p.fail(
            "Cycles WebGPU filtered Normal, BSDF Color, and AO passes need surface-only transport");
      }
      p.surface_passes_active = true;
    }
    surface_offsets[i] = uint32_t(offset);
  }
  if (p.surface_passes_active && (p.portal_materials || p.hair_materials))
  {
    return p.fail("Cycles WebGPU filtered surface passes need surfaces without Ray Portal or Hair");
  }
  const int denoising_albedo = params.get_pass_offset(PASS_DENOISING_ALBEDO);
  constexpr std::array light_passes = {
      PASS_DIFFUSE_DIRECT, PASS_DIFFUSE_INDIRECT, PASS_GLOSSY_DIRECT, PASS_GLOSSY_INDIRECT,
      PASS_TRANSMISSION_DIRECT, PASS_TRANSMISSION_INDIRECT};
  std::array<uint32_t, 8> light_offsets{};
  p.light_passes_active = false;
  for (size_t i = 0; i < light_passes.size(); i++) {
    const int offset = params.get_pass_offset(light_passes[i]);
    if (offset != PASS_UNUSED) {
      if (offset < 0 || offset > params.pass_stride - 3) {
        return p.fail("Cycles WebGPU received invalid lighting pass offsets");
      }
      p.light_passes_active = true;
    }
    light_offsets[i] = uint32_t(offset);
  }
  if (p.light_passes_active && (p.absorption || p.hair_materials || p.subsurface_materials)) {
    return p.fail("Cycles WebGPU lighting passes need surface-only transport without Hair or SSS");
  }
  const int denoising_normal = params.get_pass_offset(PASS_DENOISING_NORMAL);
  const bool guided = denoising_albedo != PASS_UNUSED || denoising_normal != PASS_UNUSED;
  std::array<uint32_t, 4> aov_layout{};
  if (!get_aov_layout(params, aov_layout)) {
    return p.fail("Cycles WebGPU needs valid contiguous film AOV passes");
  }
  const uint32_t aov_slots = p.aov_materials ? aov_layout[1] + aov_layout[3] : 0;
  const size_t extra_material_words = 4 * (size_t(aov_slots) +
      (p.ordered_closures ? CYCLES_WEBGPU_ORDERED_PATH_VECTORS : 0u));
  const auto valid_guide_offset = [&](const int offset) {
    return offset == PASS_UNUSED || (offset >= 0 && offset <= params.pass_stride - 3);
  };
  const uint64_t pixels = uint64_t(params.width) * params.height;
  if (params.width <= 0 || params.height <= 0 || params.full_x < 0 || params.full_y < 0 ||
      pixels > std::numeric_limits<int>::max() || params.stride < params.width || combined < 0 ||
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
      !std::ranges::all_of(camera, finite) || !std::ranges::all_of(camera_lens, finite))
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
  if (adaptive &&
      (uint64_t(params.width) > uint64_t(limits.workgroups_per_dimension) * workgroup_size ||
       uint32_t(params.height) > limits.workgroups_per_dimension))
  {
    return p.fail("Cycles WebGPU adaptive film exceeds the device dispatch dimensions");
  }
  if (film_words > limits.storage_buffer_bytes / sizeof(uint32_t)) {
    return p.fail("Cycles WebGPU render slice film exceeds the device buffer limit");
  }
  const auto path_buffers = path_buffer_sizes(
      adaptive, p.deferred_shadow, guided, p.textured_lights, extra_material_words,
      !p.query_sampling.empty(), p.ray_differentials, p.triangle_tangents, p.absorption,
      p.scattering, p.emission_pass_active, p.surface_passes_active, p.light_passes_active,
      p.compact_volume_shadows, p.surface_roulette_enabled());
  const uint32_t device_capacity = path_capacity_for_limits(
      limits, path_buffers, bytes_per_path);
  if (device_capacity < min_path_capacity) {
    return p.fail("Cycles WebGPU device limits cannot fit the minimum path pool");
  }
  const uint32_t requested = uint32_t(
      std::max(int(min_path_capacity), p.queue->num_concurrent_states(bytes_per_path)));
  p.film_ready = false;
  p.batch_active = false;
  p.next_sample = 0;
  p.path_pool_limit = std::min(requested, device_capacity);
  p.material_words_per_path = CYCLES_WEBGPU_MATERIAL_WORDS + extra_material_words;
  p.path_capacity = 0;
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
    if ((p.closure_compositions & principled_compositions) &&
        !p.closure_composition_shaders[p.guide_kernel])
    {
      return p.fail("Cycles WebGPU needs rebuilt denoising guides for Principled mixtures");
    }
    if (p.paired_principled_layers && !p.paired_principled_layer_shaders[p.guide_kernel]) {
      return p.fail("Cycles WebGPU needs rebuilt denoising guides for paired Principled layers");
    }
    if (p.single_principled_coat && !p.single_principled_coat_shaders[p.guide_kernel]) {
      return p.fail("Cycles WebGPU needs rebuilt denoising guides for single Principled Coat");
    }
    if (p.volume_shadow_params_offset &&
        !p.varying_volume_dispatch_shaders[p.volume_scatter_stage(true)])
    {
      return p.fail("Cycles WebGPU needs rebuilt denoising shaders for varying volume shadows");
    }
    if (p.scattering && p.ray_differentials &&
        (!p.volume_differential_shaders[p.guided_background_stage()] ||
         !p.volume_differential_shaders[p.volume_scatter_stage(true)]))
    {
      return p.fail("Cycles WebGPU needs rebuilt guide shaders for volume ray differentials");
    }
    if (p.extended_volume_stack &&
        (!p.extended_volume_stack_shaders[p.volume_scatter_stage(true)] ||
         !p.extended_volume_stack_shaders[p.volume_scatter_shadow_stage(true)]))
    {
      return p.fail("Cycles WebGPU needs rebuilt denoising shaders for the extended volume stack");
    }
    if (p.scattering && p.transparent_materials &&
        !p.transparent_volume_shaders[p.volume_scatter_shadow_stage(true)])
    {
      return p.fail("Cycles WebGPU needs rebuilt denoising shaders for transparent volume shadows");
    }
    if (p.additive_volume &&
        (!p.additive_volume_shaders[p.volume_scatter_stage(true)] ||
         !p.additive_volume_shaders[p.volume_scatter_shadow_stage(true)]))
    {
      return p.fail("Cycles WebGPU needs rebuilt denoising shaders for literal additive volumes");
    }
    if (p.mixed_volume && !p.mixed_volume_shaders[p.volume_scatter_stage(true)]) {
      return p.fail("Cycles WebGPU needs a rebuilt mixed volume guide shader");
    }
    if (!p.supports_lamp_emission(p.guided_background_stage()) ||
        (p.scattering && !p.supports_lamp_emission(p.volume_scatter_stage(true))))
    {
      return p.fail("Cycles WebGPU needs rebuilt guide shaders for analytic lamp emission");
    }
    if (p.sky_world &&
        (!p.sky_world_shaders[p.guided_background_stage()] ||
         (p.scattering && !p.sky_world_shaders[p.volume_scatter_stage(true)])))
    {
      return p.fail("Cycles WebGPU needs rebuilt denoising guide shaders for the procedural sky");
    }
    if (p.mapped_world &&
        (!p.mapped_world_shaders[p.guided_background_stage()] ||
         (p.scattering && !p.mapped_world_shaders[p.volume_scatter_stage(true)])))
    {
      return p.fail("Cycles WebGPU needs rebuilt denoising guide shaders for mapped world SVM");
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
    if (p.glossy_transparent && !p.glossy_transparent_shaders[p.guide_kernel]) {
      return p.fail("Cycles WebGPU needs a rebuilt denoising shader for Glossy and Transparent");
    }
    if (p.standalone_closures && !p.standalone_shaders[p.guide_kernel]) {
      return p.fail("Cycles WebGPU needs a rebuilt denoising shader for standalone surface closures");
    }
    if (p.mixed_diffuse && !p.mixed_diffuse_shaders[p.guide_kernel]) {
      return p.fail("Cycles WebGPU needs rebuilt denoising shaders for mixed Diffuse closures");
    }
    if (p.independent_mixed_bump && !p.independent_mixed_bump_shaders[p.guide_kernel]) {
      return p.fail("Cycles WebGPU needs rebuilt denoising shaders for independent mixed Bump normals");
    }
    if (p.ordered_closures && !p.ordered_closure_shaders[p.guide_kernel]) {
      return p.fail("Cycles WebGPU needs a rebuilt denoising shader for ordered closures");
    }
    if (p.ordered_transparency && !p.ordered_transparency_shaders[p.guide_kernel]) {
      return p.fail("Cycles WebGPU needs a rebuilt denoising shader for ordered transparency");
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
    if (!upload(p.device, p.denoising_params, std::span<const uint32_t, 16>(guide_config)))
    {
      return p.fail("Cycles WebGPU could not allocate denoising guide state");
    }
    LOG_INFO << "Cycles WebGPU writing denoising guides";
    if (p.absorption) {
      LOG_INFO << "Cycles WebGPU writing volume denoising guides";
    }
  }
  else {
    p.denoising_params.free();
  }

  p.adaptive_sampling = adaptive;
  if (p.adaptive_sampling && !p.prepare_adaptive()) {
    return false;
  }
  const bool uses_camera_lens = camera_lens[0] != float(CAMERA_PERSPECTIVE) || camera_lens[1] > 0.0f;
  if (uses_camera_lens &&
      (!p.camera_lens_shaders[camera_kernel] ||
       (adaptive && !p.camera_lens_shaders[adaptive_camera_rays_kernel])))
  {
    return p.fail("Cycles WebGPU needs rebuilt camera shaders for orthographic or depth-of-field rays");
  }
  if (p.ray_differentials && p.adaptive_sampling &&
      !p.differential_shaders[adaptive_camera_rays_kernel])
  {
    return p.fail("Cycles WebGPU needs a rebuilt adaptive camera shader for ray differentials");
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
  std::array<uint32_t, 8> geometry_layout{};
  bool geometry_passes_active = false;
  for (size_t i = 0; i < geometry_passes.size(); i++) {
    const PassType type = geometry_passes[i];
    const int offset = params.get_pass_offset(type);
    geometry_layout[i] = uint32_t(offset);
    if (offset != PASS_UNUSED) {
      const int components = Pass::get_info(type).num_components;
      if (offset < 0 || offset > params.pass_stride - components) {
        return p.fail("Cycles WebGPU received invalid geometry pass offsets");
      }
      geometry_passes_active = true;
    }
  }
  p.geometry_passes_active = geometry_passes_active;
  if (p.geometry_passes_active) {
    if (p.portal_materials) {
      return p.fail("Cycles WebGPU geometry passes need surfaces without Ray Portal");
    }
    const auto camera_z = camera.subspan<CYCLES_WEBGPU_CAMERA_SVM_OFFSET + 8, 4>();
    std::ranges::transform(camera_z, geometry_layout.begin() + 4, [](const float value) {
      return std::bit_cast<uint32_t>(value);
    });
    if (!p.shader_loaded[write_geometry_passes_kernel]) {
      const auto shader = p.load_shader("film.wgsl");
      if (shader.empty() ||
          !p.register_kernel(
              write_geometry_passes_kernel, shader, geometry_pass_bindings, "write_geometry_passes"))
      {
        return p.fail("Cycles WebGPU could not initialize geometry film passes");
      }
    }
    if (p.absorption && !p.volume_geometry_shaders[write_geometry_passes_kernel]) {
      return p.fail("Cycles WebGPU needs a rebuilt film shader for geometry passes with volume");
    }
    if (!upload(p.device, p.geometry_pass_params, std::span<const uint32_t>(geometry_layout))) {
      return p.fail("Cycles WebGPU could not upload geometry film offsets");
    }
  }
  else {
    p.geometry_pass_params.free();
  }
  if (p.surface_passes_active) {
    if (!p.shader_loaded[shade_surface_passes_kernel]) {
      const auto shader = p.load_shader("film-surface.wgsl");
      if (shader.find(film_surface_marker) == shader.npos ||
          !p.register_kernel(shade_surface_passes_kernel, shader, surface_pass_shader_bindings))
      {
        return p.fail("Cycles WebGPU could not initialize filtered surface shading");
      }
    }
    if (p.absorption && !p.volume_geometry_shaders[shade_surface_passes_kernel]) {
      return p.fail(
          "Cycles WebGPU needs a rebuilt surface shader for transparent geometry passes with volume");
    }
    if (p.mixed_diffuse && !p.mixed_diffuse_shaders[shade_surface_passes_kernel]) {
      return p.fail("Cycles WebGPU needs a rebuilt film shader for mixed Diffuse closures");
    }
    if (p.independent_mixed_bump && !p.independent_mixed_bump_shaders[shade_surface_passes_kernel]) {
      return p.fail("Cycles WebGPU needs a rebuilt film shader for independent mixed Bump normals");
    }
    if (p.ordered_closures && !p.ordered_closure_shaders[shade_surface_passes_kernel]) {
      return p.fail("Cycles WebGPU needs a rebuilt film shader for ordered closures");
    }
    if (p.ordered_transparency && !p.ordered_transparency_shaders[shade_surface_passes_kernel]) {
      return p.fail("Cycles WebGPU needs a rebuilt film shader for ordered transparency");
    }
    if (p.glossy_displacement_bump &&
        !p.glossy_displacement_bump_shaders[shade_surface_passes_kernel])
    {
      return p.fail("Cycles WebGPU needs a rebuilt film shader for Glossy displacement Bump");
    }
    if (p.mixed_displacement_bump &&
        !p.mixed_displacement_bump_shaders[shade_surface_passes_kernel])
    {
      return p.fail("Cycles WebGPU needs a rebuilt film shader for Diffuse and Glossy displacement Bump");
    }
    if (p.glossy_transparent && !p.glossy_transparent_shaders[shade_surface_passes_kernel]) {
      return p.fail("Cycles WebGPU needs a rebuilt film shader for Glossy and Transparent");
    }
    if ((p.closure_compositions & principled_compositions) &&
        !p.closure_composition_shaders[shade_surface_passes_kernel])
    {
      return p.fail("Cycles WebGPU needs a rebuilt film shader for Principled mixtures");
    }
    if (p.paired_principled_layers &&
        !p.paired_principled_layer_shaders[shade_surface_passes_kernel])
    {
      return p.fail("Cycles WebGPU needs a rebuilt film shader for paired Principled layers");
    }
    if (p.single_principled_coat && !p.single_principled_coat_shaders[shade_surface_passes_kernel]) {
      return p.fail("Cycles WebGPU needs a rebuilt film shader for single Principled Coat");
    }
    if (p.ao_pass_active && p.active_shadow_kernel == varying_shadow_kernel) {
      if (!p.varying_ao_shadow_supported) {
        return p.fail("Cycles WebGPU needs a rebuilt varying shadow shader for Ambient Occlusion");
      }
      if (!p.ao_prepare_loaded) {
        const auto shader = p.load_shader("film-ao-prepare.wgsl");
        p.ao_prepare_loaded = shader.find(film_surface_marker) != shader.npos &&
                              device_webgpu_register_kernel(p.device, webgpu_prepare_ao_pass_kernel,
                                                            shader, workgroup_size,
                                                            ao_prepare_shader_bindings);
        if (!p.ao_prepare_loaded) {
          return p.fail("Cycles WebGPU could not prepare Ambient Occlusion shadow rays");
        }
      }
    }
    else if (p.ao_pass_active && !p.shader_loaded[shade_ao_pass_kernel]) {
      const auto shader = p.load_shader("film-ao.wgsl");
      if (shader.find(film_surface_marker) == shader.npos ||
          !p.register_kernel(shade_ao_pass_kernel, shader, ao_pass_shader_bindings))
      {
        return p.fail("Cycles WebGPU could not initialize the Ambient Occlusion pass");
      }
    }
    if (!p.shader_loaded[accumulate_surface_passes_kernel]) {
      const auto shader = p.load_shader("film.wgsl");
      if (shader.empty() ||
          !p.register_kernel(accumulate_surface_passes_kernel, shader, surface_pass_film_bindings,
                             "accumulate_surface_passes"))
      {
        return p.fail("Cycles WebGPU could not initialize filtered surface film writing");
      }
    }
    if (!upload(p.device, p.surface_pass_params, std::span<const uint32_t>(surface_offsets)))
    {
      return p.fail("Cycles WebGPU could not allocate filtered surface passes");
    }
  }
  else {
    p.surface_pass_params.free();
  }
  if (p.emission_pass_active) {
    if (emission_offset < 0 || emission_offset > params.pass_stride - 3) {
      return p.fail("Cycles WebGPU received an invalid Emission pass offset");
    }
    const DeviceKernel background_stage = p.guided ? p.guided_background_stage() :
                                                     p.active_background_kernel;
    if (!p.emission_delta_shaders[p.update_kernel] ||
        (p.shader_loaded[p.active_background_kernel] && !p.emission_delta_shaders[background_stage]))
    {
      return p.fail("Cycles WebGPU Emission needs rebuilt surface and background kernels");
    }
    if (!p.shader_loaded[accumulate_emission_kernel]) {
      const auto shader = p.load_shader("film.wgsl");
      if (shader.empty() ||
          !p.register_kernel(
              accumulate_emission_kernel, shader, emission_pass_bindings, "accumulate_emission"))
      {
        return p.fail("Cycles WebGPU could not initialize the Emission pass");
      }
    }
    const std::array<uint32_t, 4> layout{uint32_t(emission_offset), 0, 0, 0};
    if (!upload(p.device, p.emission_pass_params, std::span<const uint32_t>(layout))) {
      return p.fail("Cycles WebGPU could not upload the Emission offset");
    }
  }
  else {
    p.emission_pass_params.free();
  }
  if (p.light_passes_active) {
    const DeviceKernel background_stage = p.guided ? p.guided_background_stage() :
                                                     p.active_background_kernel;
    if (!p.film_light_shaders[p.light_kernel] || !p.film_light_shaders[p.update_kernel] ||
        (p.shader_loaded[p.active_background_kernel] && !p.film_light_shaders[background_stage]))
    {
      return p.fail("Cycles WebGPU lighting passes need rebuilt surface and background kernels");
    }
    if (!p.light_passes_loaded) {
      const auto shader = p.load_shader("film.wgsl");
      p.light_passes_loaded = !shader.empty() &&
                             device_webgpu_register_kernel(p.device,
                                                           accumulate_light_passes_kernel,
                                                           shader,
                                                           workgroup_size,
                                                           light_pass_film_bindings,
                                                           "accumulate_light_passes");
      if (!p.light_passes_loaded) {
        return p.fail("Cycles WebGPU could not initialize the lighting passes");
      }
    }
    if (!upload(p.device, p.light_pass_params, std::span<const uint32_t>(light_offsets))) {
      return p.fail("Cycles WebGPU could not upload the lighting pass offsets");
    }
  }
  else {
    p.light_pass_params.free();
  }
  std::span<const float> camera_data = camera;
  std::vector<float> camera_words;
  if (p.aov_materials || p.ordered_closures || !p.query_sampling.empty()) {
    camera_words.assign(camera.begin(), camera.end());
    camera_words.resize(CYCLES_WEBGPU_CAMERA_WORDS + CYCLES_WEBGPU_CAMERA_SVM_CONFIG_WORDS);
    camera_words[CYCLES_WEBGPU_CAMERA_AOV_COLORS] =
        std::bit_cast<float>(p.aov_materials ? aov_layout[1] : 0u);
    camera_words[CYCLES_WEBGPU_CAMERA_AOV_VALUES] =
        std::bit_cast<float>(p.aov_materials ? aov_layout[3] : 0u);
    for (const uint32_t word : p.query_sampling) {
      camera_words.push_back(std::bit_cast<float>(word));
    }
    camera_data = camera_words;
  }
  std::array<float, CYCLES_WEBGPU_CAMERA_RAY_WORDS> camera_rays;
  std::ranges::copy(camera.first<CYCLES_WEBGPU_CAMERA_SVM_OFFSET>(), camera_rays.begin());
  std::ranges::copy(camera_lens, camera_rays.begin() + CYCLES_WEBGPU_CAMERA_SVM_OFFSET);
  if (!upload(p.device, p.camera_rays, std::span<const float>(camera_rays)) ||
      !upload(p.device, p.camera, camera_data) ||
      !upload(p.device, p.params, std::span<const uint32_t>(p.film_config)))
  {
    return p.fail("Cycles WebGPU camera or film parameters could not be uploaded");
  }
  if (p.volume_shadow_params_offset) {
    std::array<uint32_t, CYCLES_WEBGPU_VOLUME_SHADOW_PARAMS_WORDS> shadow_params;
    for (size_t i = 0; i < camera.size(); ++i) {
      shadow_params[i] = std::bit_cast<uint32_t>(camera[i]);
    }
    shadow_params[CYCLES_WEBGPU_CAMERA_WORDS] = uint32_t(guided ? denoising_albedo : PASS_UNUSED);
    shadow_params[CYCLES_WEBGPU_CAMERA_WORDS + 1] = uint32_t(guided ? denoising_normal : PASS_UNUSED);
    if (!device_webgpu_upload(p.device, p.scene.surface,
                              p.volume_shadow_params_offset * sizeof(uint32_t),
                              std::as_bytes(std::span(shadow_params))))
    {
      return p.fail("Cycles WebGPU varying volume shadow parameters could not be uploaded");
    }
  }
  p.queue->zero_to_device(p.film);
  if (!device_webgpu_error(p.device).empty()) {
    return p.fail("Cycles WebGPU could not initialize the film");
  }
  p.film_ready = true;
  return true;
}

bool WebGPURenderPipeline::Impl::reserve_paths(const uint32_t capacity)
{
  denoising.alloc_to_device(guided ? size_t(denoising_words_per_path) * capacity : 0);
  if (guided && !denoising.device_pointer) {
    return fail("Cycles WebGPU could not allocate denoising guide state");
  }
  surface_pass_values.alloc_to_device(
      surface_passes_active ? size_t(surface_pass_words_per_path) * capacity : 0);
  if (surface_passes_active && !surface_pass_values.device_pointer) {
    return fail("Cycles WebGPU could not allocate filtered surface pass storage");
  }
  pixels.alloc_to_device(pixel_words_per_path * capacity);
  rays.alloc_to_device(ray_header_words + uint32_t(compact_volume_shadows) +
                       ray_words_per_path(adaptive_sampling, ray_differentials, scattering,
                                          compact_volume_shadows) * capacity);
  if (compact_volume_shadows) {
    if (!volume_shadow_indirect.device_pointer) {
      if (compact_volume_shadow_linear_grid) {
        const auto limits = device_webgpu_compute_limits(device);
        const std::array<uint32_t, 4> initial_grid = {0, 1, 1, limits.workgroups_per_dimension};
        if (!upload(device, volume_shadow_indirect, std::span<const uint32_t, 4>(initial_grid))) {
          return fail("Cycles WebGPU could not initialize volume shadow dispatch limits");
        }
      }
      else {
        volume_shadow_indirect.alloc_to_device(3);
      }
    }
    if (!volume_shadow_indirect.device_pointer) {
      return fail("Cycles WebGPU could not allocate volume shadow dispatch arguments");
    }
  }
  else {
    volume_shadow_indirect.free();
  }
  hits.alloc_to_device(hit_words_per_path * capacity);
  if (absorption) {
    absorption_state.alloc_to_device(
        capacity * (scattering ? CYCLES_WEBGPU_VOLUME_STATE_WORDS : 1u));
    if (!absorption_state.device_pointer) {
      return fail("Cycles WebGPU could not allocate homogeneous absorption paths");
    }
  }
  else {
    absorption_state.free();
  }
  if (deferred_shadow) {
    shadow_rays.alloc_to_device(shadow_storage_words_per_path(textured_lights) * capacity);
    if (!shadow_rays.device_pointer) {
      return fail("Cycles WebGPU could not allocate shadow rays");
    }
  }
  else {
    shadow_rays.free();
  }
  points.alloc_to_device(
      shading_storage_words_per_path(!query_sampling.empty(), triangle_tangents) * capacity);
  /* Area light hits accumulate separately from surface and world shading. */
  const size_t radiance_words = radiance_words_per_path * capacity;
  radiance.alloc_to_device(radiance_words + radiance_header_words +
                           (emission_pass_active ? emission_words_per_path * capacity : 0) +
                           (light_passes_active ? light_pass_words_per_path * capacity : 0));
  const std::array<float, radiance_header_words> radiance_header = {
      float(light_passes_active), 0.0f, 0.0f, float(emission_pass_active)};
  if (!device_webgpu_upload(device,
                            radiance,
                            radiance_words * sizeof(float),
                            std::as_bytes(std::span(radiance_header))))
  {
    return fail("Cycles WebGPU could not initialize the radiance layout");
  }
  materials.alloc_to_device(material_words_per_path * capacity);
  if (!pixels.device_pointer || !rays.device_pointer || !hits.device_pointer ||
      !points.device_pointer || !radiance.device_pointer || !materials.device_pointer ||
      !path_state.alloc(capacity) || !wavefront_queue.alloc()) {
    return fail("Cycles WebGPU could not allocate path state and queue counters");
  }
  if (!reserve_surface_roulette(capacity)) {
    return false;
  }
  queue->zero_to_device(path_state.words());
  if (!wavefront_queue.set_capacity(*queue, rays, path_state.words()) ||
      !device_webgpu_error(device).empty())
  {
    return fail("Cycles WebGPU could not initialize the path pool");
  }
  path_capacity = capacity;
  const device_memory *allocations[] = {
      &pixels, &rays, &hits, &shadow_rays, &points, &radiance, &denoising,
      &surface_pass_values, &materials, &path_state.words(), &absorption_state,
      &volume_shadow_indirect, &surface_roulette_queue, &surface_roulette_dispatch};
  uint64_t allocated_bytes = 0;
  size_t largest_binding_bytes = 0;
  for (const device_memory *buffer : allocations) {
    if (buffer->device_pointer) {
      allocated_bytes += buffer->device_size;
      largest_binding_bytes = std::max(largest_binding_bytes, buffer->device_size);
    }
  }
  LOG_INFO << "Cycles WebGPU path pool: " << capacity << " slots, " << allocated_bytes
           << " allocated path bytes, largest binding " << largest_binding_bytes
           << " bytes, limit " << path_pool_limit << " slots";
  return true;
}

bool WebGPURenderPipeline::begin_batch(const int start_sample,
                                      const int samples_num,
                                      uint32_t &path_capacity)
{
  Impl &p = *impl_;
  if (!p.film_ready || p.batch_active || start_sample < 0 || samples_num <= 0 ||
      samples_num > std::numeric_limits<int>::max() - start_sample ||
      uint32_t(start_sample) != p.next_sample)
  {
    return p.fail("Cycles WebGPU received a non-consecutive render batch");
  }
  if (!p.path_capacity) {
    p.path_pool_limit = std::min(p.path_pool_limit, device_webgpu_path_dispatch_capacity(p.device));
    if (p.ordered_closures &&
        !p.supports_integer_ordered_indices(p.guided, p.surface_passes_active))
    {
      /* Legacy ordered-closure shaders store the tail index in a float. */
      constexpr size_t max_exact_index = size_t(1) << std::numeric_limits<float>::digits;
      const size_t vectors_per_path = p.material_words_per_path / 4;
      p.path_pool_limit = uint32_t(std::min<size_t>(p.path_pool_limit,
                                                  max_exact_index / vectors_per_path));
    }
    if (p.path_pool_limit < min_path_capacity) {
      return p.fail("Cycles WebGPU dispatch limits cannot fit the minimum path pool");
    }
  }

  /* The stock scheduler assigns actual overlap in its sample batches. Grow only
   * while paths are retired, and preserve film data from previous batches. */
  const uint64_t batch_paths = uint64_t(p.pixel_count) * samples_num;
  const uint32_t capacity = uint32_t(std::min<uint64_t>(
      p.path_pool_limit, std::max<uint64_t>(min_path_capacity, batch_paths)));
  if (capacity > p.path_capacity && !p.reserve_paths(capacity)) {
    return false;
  }
  path_capacity = p.path_capacity;
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
  const auto camera = p.bind(p.tile, p.camera_rays, p.pixels, p.rays, p.sampling);
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
      const auto shadow = p.volume_shadow_params_offset ?
          p.bind(p.scene.bvh, p.rays, p.shadow_rays, paths, p.scene.surface, p.absorption_state,
                 p.guided ? p.denoising : p.radiance, p.scene.svm) :
          p.guided ?
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
          (p.compact_volume_shadows ?
               !p.enqueue_compact_volume_shadow(count, shadow, p.guided, &classify_dispatch) :
               !device_webgpu_enqueue(p.device, count, visibility)))
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
  const auto emission_pass = p.bind(p.radiance, p.film, paths, p.rays, p.params,
                                    p.emission_pass_params);
  const WebGPUDispatch accumulate_emission{accumulate_emission_kernel, emission_pass, count};
  const auto light_pass = p.bind(p.radiance, p.film, paths, p.rays, p.params,
                                 p.light_pass_params);
  const WebGPUDispatch accumulate_lighting{accumulate_light_passes_kernel, light_pass, count};
  const auto geometry_passes = p.bind(
      p.materials, p.film, paths, p.rays, p.params, p.points, p.hits, p.scene.surface,
      p.geometry_pass_params);
  const WebGPUDispatch geometry{write_geometry_passes_kernel, geometry_passes, count};
  if (stage == DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND &&
      p.shader_loaded[p.active_background_kernel])
  {
    const auto background = p.guided ?
                                p.bind(p.rays, p.hits, p.scene.light, p.sampling, paths, p.denoising,
                                       p.radiance) :
                                p.bind(p.rays, p.hits, p.scene.light, p.sampling, paths, p.radiance);
    const DeviceKernel kernel = p.guided ? p.guided_background_stage() : p.active_background_kernel;
    std::vector<WebGPUDispatch> stages;
    stages.reserve(5);
    if (p.geometry_passes_active && p.transparent_materials && p.next_sample == 0) {
      stages.push_back(geometry);
    }
    stages.push_back({kernel, background, count});
    if (p.emission_pass_active) {
      stages.push_back(accumulate_emission);
    }
    if (p.light_passes_active) {
      stages.push_back(accumulate_lighting);
    }
    stages.push_back(accumulate);
    if (!p.wavefront_queue.enqueue_stage(
            stage, capacity, p.rays, paths, stages))
    {
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
  const auto shadow = p.volume_shadow_params_offset ?
                          p.bind(p.scene.bvh, p.rays, p.shadow_rays, paths, p.scene.surface,
                                 p.absorption_state, p.radiance, p.scene.svm) :
                      p.scattering ?
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
  const auto surface_pass_inputs = p.bind(p.rays, p.points, p.materials, p.sampling,
                                           p.radiance, paths, p.surface_pass_values,
                                           p.surface_pass_params);
  const auto surface_pass_film = p.bind(p.surface_pass_values, p.film, paths, p.rays, p.params,
                                         p.surface_pass_params);
  const bool varying_ao = p.ao_pass_active && p.active_shadow_kernel == varying_shadow_kernel;
  const auto ao_inputs = varying_ao ?
      p.bind(p.shadow_rays, p.rays, p.hits, p.points, paths, p.sampling, p.surface_pass_values,
             p.materials) :
      p.bind(p.scene.bvh, p.rays, p.hits, p.points, paths, p.sampling, p.surface_pass_values,
             p.scene.surface);
  const auto ao_shadow = p.bind(p.scene.bvh, p.rays, p.shadow_rays, p.surface_pass_values,
                                p.scene.surface, paths, p.scene.svm, p.camera);
  const WebGPUDispatch intersect_ao{varying_shadow_kernel, ao_shadow, count};
  const WebGPUDispatch accumulate_surface{accumulate_surface_passes_kernel, surface_pass_film,
                                           count};
  const WebGPUDispatch surface_pass_dispatches[] = {
      {shade_surface_passes_kernel, surface_pass_inputs, count},
      p.ao_pass_active ? WebGPUDispatch{varying_ao ? webgpu_prepare_ao_pass_kernel :
                                                   shade_ao_pass_kernel,
                                      ao_inputs, count} :
                         accumulate_surface,
      accumulate_surface};

  const auto update = p.bind(
      p.rays, p.hits, p.points, p.pixels, p.sampling, p.materials, p.radiance, paths);
  const auto transition = p.bind(p.rays, p.hits, paths, p.absorption_state);
  const WebGPUDispatch cross_volume_boundary{volume_transition_kernel, transition, count};
  const WebGPUDispatch setup_surface{queries ? surface_query_kernel : surface_kernel,
                                    surface, count};
  const WebGPUDispatch evaluate_material{p.active_svm_kernel, svm, count, 0, 1, p.svm_image_atlas};
  const WebGPUDispatch sample_light{p.light_kernel, light, count};
  std::vector<WebGPUDispatch> continuation;
  continuation.reserve(5);
  continuation.push_back({p.update_kernel, update, count});
  if (p.mixed_volume) {
    continuation.push_back(cross_volume_boundary);
  }
  if (p.emission_pass_active) {
    continuation.push_back(accumulate_emission);
  }
  if (p.light_passes_active) {
    continuation.push_back(accumulate_lighting);
  }
  continuation.push_back(accumulate);
  /* Visibility traversal and BSDF continuation are the expensive stages.
   * Submit them separately to bound the work in each browser command buffer. */
  const bool material_ready = p.surface_roulette_enabled() &&
                                      stage == DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE ?
                                  p.enqueue_compact_surface_roulette(count, setup_surface,
                                                                     evaluate_material) :
                                  p.wavefront_queue.enqueue_material_stage(
                                      stage, capacity, p.rays, paths, p.points, p.scene.svm,
                                      p.sort_surface_shaders, setup_surface, evaluate_material);
  if (!material_ready ||
      (p.aov_active && !device_webgpu_enqueue(p.device, count, {&accumulate_aovs, 1})) ||
      !device_webgpu_enqueue(p.device, count, {&sample_light, 1}) ||
      (p.textured_lights && !device_webgpu_enqueue(p.device, count, {&shade_emission, 1})) ||
      (p.deferred_shadow &&
       (p.compact_volume_shadows ?
            !p.enqueue_compact_volume_shadow(count, shadow, false) :
            !device_webgpu_enqueue(p.device, count, {&intersect_shadow, 1}))) ||
      (p.guided && !device_webgpu_enqueue(p.device, count, {&shade_guides, 1})) ||
      (p.geometry_passes_active && p.next_sample == 0 &&
       !device_webgpu_enqueue(p.device, count, {&geometry, 1})) ||
      (p.surface_passes_active &&
       (!device_webgpu_enqueue(p.device, count, std::span(surface_pass_dispatches)
                                                   .first(p.ao_pass_active && !varying_ao ? 3 : 2)) ||
        (varying_ao &&
         (!device_webgpu_enqueue(p.device, count, {&intersect_ao, 1}) ||
          !device_webgpu_enqueue(p.device, count, {&accumulate_surface, 1}))))) ||
      !device_webgpu_enqueue(
          p.device, count, continuation))
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
      {DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_CHECK, filter, p.crop[2], 0, p.crop[3]},
      {adaptive_count_kernel, count_active, p.crop[2], 0, p.crop[3]},
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
