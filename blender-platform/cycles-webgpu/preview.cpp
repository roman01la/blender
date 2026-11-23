/* SPDX-License-Identifier: Apache-2.0 */

#include "device/webgpu/preview.h"
#include "device/webgpu/lighting_layout.h"
#include "device/webgpu/material_layout.h"
#include "device/webgpu/path_state_layout.h"
#include "device/webgpu/volume_layout.h"

#include "device/device.h"
#include "device/webgpu/device.h"
#include "device/webgpu/pipeline.h"
#include "scene/integrator.h"
#include "scene/pass.h"
#include "scene/scene.h"
#include "session/buffers.h"
#include "util/log.h"

#include <algorithm>
#include <bit>
#include <cfloat>
#include <fstream>
#include <iterator>
#include <string>

CCL_NAMESPACE_BEGIN

namespace {

std::string read_shader(const char *name)
{
  std::ifstream shader(std::string("/cycles-webgpu/") + name, std::ios::binary);
  return {std::istreambuf_iterator<char>(shader), std::istreambuf_iterator<char>()};
}

std::array<float, CYCLES_WEBGPU_CAMERA_WORDS> pack_camera(const KernelCamera &cam,
                                                         const float scrambling_distance,
                                                         const bool ray_differentials)
{
  const std::array rows = {cam.rastertocamera.x,
                           cam.rastertocamera.y,
                           cam.rastertocamera.z,
                           cam.rastertocamera.w,
                           cam.cameratoworld.x,
                           cam.cameratoworld.y,
                           cam.cameratoworld.z,
                           make_float4(cam.nearclip,
                                       cam.cliplength,
                                       scrambling_distance,
                                       ray_differentials ? cam.differential_scale : 0.0f),
                           cam.worldtocamera.x,
                           cam.worldtocamera.y,
                           cam.worldtocamera.z,
                           cam.worldtondc.x,
                           cam.worldtondc.y,
                           cam.worldtondc.z,
                           cam.worldtondc.w};
  static_assert(rows.size() * 4 == CYCLES_WEBGPU_CAMERA_WORDS);
  std::array<float, CYCLES_WEBGPU_CAMERA_WORDS> camera;
  for (size_t i = 0; i < rows.size(); ++i) {
    const auto &row = rows[i];
    camera[i * 4] = row.x;
    camera[i * 4 + 1] = row.y;
    camera[i * 4 + 2] = row.z;
    camera[i * 4 + 3] = row.w;
  }
  return camera;
}

bool pack_camera_sampling(const DeviceScene &scene,
                          const CyclesWebGPUScene &encoded,
                          std::vector<uint32_t> &words)
{
  const KernelIntegrator &integrator = scene.data.integrator;
  const int filter_offset = scene.data.tables.filter_table_offset;
  if (filter_offset < 0 || size_t(filter_offset) + FILTER_TABLE_SIZE > scene.lookup_table.size()) {
    return false;
  }

  size_t sobol_size = 0;
  if (integrator.sampling_pattern == SAMPLING_PATTERN_TABULATED_SOBOL) {
    const int sequence_size = integrator.tabulated_sobol_sequence_size;
    if (sequence_size < MIN_TAB_SOBOL_SAMPLES || sequence_size > MAX_TAB_SOBOL_SAMPLES ||
        !std::has_single_bit(uint32_t(sequence_size)))
    {
      return false;
    }
    sobol_size = size_t(sequence_size) * NUM_TAB_SOBOL_PATTERNS * NUM_TAB_SOBOL_DIMENSIONS;
    if (sobol_size != scene.sample_pattern_lut.size()) {
      return false;
    }
  }

  /* Header, pixel-filter lookup, then the sample pattern lookup when needed. */
  words = {uint32_t(integrator.sampling_pattern),
           uint32_t(integrator.seed),
           uint32_t(integrator.sobol_index_mask),
           uint32_t(integrator.blue_noise_sequence_length),
           uint32_t(integrator.tabulated_sobol_sequence_size),
           std::bit_cast<uint32_t>(integrator.pixel_jitter.x),
           std::bit_cast<uint32_t>(integrator.pixel_jitter.y)};
  size_t lookup_words = 0;
  const bool multiple_points = encoded.lighting[0] == CYCLES_WEBGPU_POINT_LIGHTS_MARKER;
  const bool light_tree = encoded.lighting[0] == CYCLES_WEBGPU_LIGHT_TREE_MARKER;
  if (encoded.has_ggx) {
    size_t descriptor = encoded.lighting[CYCLES_WEBGPU_LIGHT_EXTENDED];
    if (light_tree) {
      descriptor = encoded.lighting[CYCLES_WEBGPU_LIGHT_TREE_OFFSET];
    }
    else if (multiple_points) {
      descriptor = encoded.lighting[CYCLES_WEBGPU_POINT_LIGHTS_OFFSET];
    }
    if (descriptor == 0) {
      descriptor = encoded.lighting[CYCLES_WEBGPU_LIGHT_RR_SAMPLING];
    }
    lookup_words = (descriptor ? descriptor : encoded.lighting.size()) -
                   CYCLES_WEBGPU_LIGHT_HEADER_WORDS;
  }
  words.reserve(words.size() + FILTER_TABLE_SIZE + sobol_size + CYCLES_WEBGPU_PATH_SETTINGS_WORDS +
                lookup_words);
  const float *filter = scene.lookup_table.data() + filter_offset;
  for (int i = 0; i < FILTER_TABLE_SIZE; ++i) {
    words.push_back(std::bit_cast<uint32_t>(filter[i]));
  }
  for (size_t i = 0; i < sobol_size; ++i) {
    words.push_back(std::bit_cast<uint32_t>(scene.sample_pattern_lut.data()[i]));
  }
  const bool ray_differentials = encoded.svm[1] & CYCLES_WEBGPU_SVM_NEEDS_DIFFERENTIALS;
  words.push_back((encoded.max_bounces + 1) |
                  (ray_differentials ? CYCLES_WEBGPU_PATH_SETTINGS_DIFFERENTIALS : 0u));
  words.push_back(uint32_t(integrator.min_bounce));
  words.push_back(std::bit_cast<uint32_t>(integrator.scrambling_distance));
  const size_t query_sampling_prefix = words.size();
  /* With multiple lights, apply the bounce limit after selecting the light. */
  const float light_max_bounces = (light_tree || multiple_points) ?
                                      FLT_MAX :
                                      scene.lights.data()[0].max_bounces;
  words.push_back(std::bit_cast<uint32_t>(light_max_bounces));
  words.push_back(uint32_t(integrator.max_diffuse_bounce));
  words.push_back(uint32_t(integrator.max_glossy_bounce));
  words.push_back(uint32_t(integrator.max_transmission_bounce));
  words.push_back(std::bit_cast<uint32_t>(integrator.filter_glossy));
  const uint32_t surface_header = encoded.surface.front();
  const bool transparent = CYCLES_WEBGPU_SURFACE_HAS_TRANSPARENCY(surface_header) ||
                           (CYCLES_WEBGPU_SURFACE_HAS_VOLUME(surface_header) &&
                            encoded.surface[CYCLES_WEBGPU_SURFACE_TRANSPARENT_OFFSET] != 0);
  const uint32_t transparent_min = transparent ? uint32_t(integrator.transparent_min_bounce) : 0;
  words.push_back(uint32_t(integrator.transparent_max_bounce) | (transparent_min << 16));
  words.push_back(uint32_t(scene.data.background.transparent));
  words.push_back(std::bit_cast<uint32_t>(integrator.sample_clamp_direct));
  if (lookup_words) {
    const auto first = encoded.lighting.begin() + CYCLES_WEBGPU_LIGHT_HEADER_WORDS;
    words.insert(words.end(), first, first + lookup_words);
  }
  if (encoded.svm[1] & CYCLES_WEBGPU_SVM_QUERIES) {
    words.push_back(std::bit_cast<uint32_t>(integrator.ao_bounces_distance));
    words.push_back(uint32_t(query_sampling_prefix));
  }
  return true;
}

bool volume_auxiliary_pass(const PassType type)
{
  return is_volume_guiding_pass(type) || type == PASS_VOLUME_MAJORANT ||
         type == PASS_VOLUME_MAJORANT_SAMPLE_COUNT;
}

bool supports_passes(const BufferParams &params, const bool volume)
{
  return std::ranges::all_of(params.passes, [volume](const BufferPass &pass) {
    /* The stock denoiser writes a separate Combined pass into the host film. */
    return pass.type == PASS_COMBINED ||
           (volume && volume_auxiliary_pass(pass.type)) ||
           (pass.mode == PassMode::NOISY &&
            (pass.type == PASS_SAMPLE_COUNT || pass.type == PASS_ADAPTIVE_AUX_BUFFER ||
             pass.type == PASS_DENOISING_ALBEDO || pass.type == PASS_DENOISING_NORMAL ||
             pass.type == PASS_AOV_COLOR || pass.type == PASS_AOV_VALUE ||
             pass.type == PASS_DEPTH || pass.type == PASS_OBJECT_ID ||
             pass.type == PASS_MATERIAL_ID || pass.type == PASS_POSITION ||
             pass.type == PASS_EMISSION || pass.type == PASS_NORMAL ||
             pass.type == PASS_DIFFUSE_COLOR || pass.type == PASS_GLOSSY_COLOR ||
             pass.type == PASS_TRANSMISSION_COLOR || pass.type == PASS_AO ||
             pass.type == PASS_DIFFUSE_DIRECT || pass.type == PASS_DIFFUSE_INDIRECT ||
             pass.type == PASS_GLOSSY_DIRECT || pass.type == PASS_GLOSSY_INDIRECT ||
             pass.type == PASS_TRANSMISSION_DIRECT || pass.type == PASS_TRANSMISSION_INDIRECT));
  });
}

}  // namespace

WebGPUPreview::WebGPUPreview(Device &device) : device_(device) {}
WebGPUPreview::~WebGPUPreview() = default;

void WebGPUPreview::prepare(const Scene &scene, bool scene_changed)
{
  if (failed_) {
    return;
  }

  const KernelCamera &cam = scene.dscene.data.cam;
  if ((cam.type != CAMERA_PERSPECTIVE && cam.type != CAMERA_ORTHOGRAPHIC) ||
      cam.shuttertime != -1.0f || cam.num_motion_steps || cam.have_perspective_motion ||
      cam.interocular_offset != 0.0f || scene.dscene.data.bake.use)
  {
    if (scene_changed) {
      LOG_INFO << "Cycles WebGPU using CPU: camera projection, motion blur, stereo, or baking";
    }
    scene_checked_ = false;
    scene_supported_ = false;
    volume_ = false;
    scene_pending_ = false;
    scene_ = {};
    std::vector<uint32_t>().swap(sampling_);
    return;
  }

  // Scene upload frees the encoded data, but later batches still sync the camera.
  bool ray_differentials = camera_[CYCLES_WEBGPU_CAMERA_DIFFERENTIAL_SCALE] != 0.0f;
  if (scene_changed || !scene_checked_) {
    std::string reason;
    const auto limits = device_webgpu_prepare_compute_limits(device_);
    if (limits.storage_buffer_bytes) {
      scene_supported_ = cycles_webgpu_encode_scene(
          scene, limits.storage_buffer_words(), scene_, reason);
    }
    else {
      reason = device_webgpu_error(device_);
      scene_supported_ = false;
    }
    volume_ = scene_supported_ && !scene_.surface.empty() &&
              CYCLES_WEBGPU_SURFACE_HAS_VOLUME(scene_.surface.front());
    if (volume_) {
      for (const Pass *pass : scene.passes) {
        if (volume_auxiliary_pass(pass->get_type()) && !pass->is_auto()) {
          reason = "WebGPU Cycles supports automatic volume auxiliary passes";
          scene_supported_ = volume_ = false;
          break;
        }
      }
    }
    if (scene_supported_ && !pack_camera_sampling(scene.dscene, scene_, sampling_)) {
      reason = "missing Cycles camera sampling tables";
      scene_supported_ = false;
    }
    scene_checked_ = true;
    if (!scene_supported_) {
      scene_pending_ = false;
      scene_ = {};
      std::vector<uint32_t>().swap(sampling_);
      if (scene_changed) {
        LOG_INFO << "Cycles WebGPU using CPU: " << reason;
      }
      return;
    }
    scene_pending_ = true;
    reset_pending_ = true;
    ray_differentials = (scene_.svm[1] & CYCLES_WEBGPU_SVM_NEEDS_DIFFERENTIALS) != 0;
  }

  const auto camera = pack_camera(cam,
                                  scene.dscene.data.integrator.scrambling_distance,
                                  ray_differentials);
  /* dx/dy come from Camera::update(), including the full-resolution raster
   * and the different derivative definitions of each projection. */
  const std::array<float, CYCLES_WEBGPU_CAMERA_LENS_WORDS> camera_lens = {
      float(cam.type),
      cam.aperturesize,
      cam.focaldistance,
      cam.inv_aperture_ratio,
      cam.blades,
      cam.bladesrotation,
      0.0f,
      0.0f,
      cam.dx.x,
      cam.dx.y,
      cam.dx.z,
      0.0f,
      cam.dy.x,
      cam.dy.y,
      cam.dy.z,
      0.0f};
  const auto &world_to_camera = cam.worldtocamera;
  std::array<uint32_t, 16> denoising_config = {
      0, 0, uint32_t(scene.dscene.data.film.denoising_pass_options_flag),
      std::bit_cast<uint32_t>(scene.dscene.data.integrator.ao_bounces_distance)};
  /* Normal transforms use only the XYZ components of these rows. */
  denoising_config[7] = std::bit_cast<uint32_t>(scene.dscene.data.film.pass_alpha_threshold);
  const std::array rows = {world_to_camera.x, world_to_camera.y, world_to_camera.z};
  for (size_t row = 0; row < rows.size(); ++row) {
    const auto &value = rows[row];
    denoising_config[4 + row * 4] = std::bit_cast<uint32_t>(value.x);
    denoising_config[5 + row * 4] = std::bit_cast<uint32_t>(value.y);
    denoising_config[6 + row * 4] = std::bit_cast<uint32_t>(value.z);
  }
  if (camera != camera_ || camera_lens != camera_lens_ || denoising_config != denoising_config_) {
    camera_ = camera;
    camera_lens_ = camera_lens;
    denoising_config_ = denoising_config;
    reset_pending_ = true;
  }
}

void WebGPUPreview::reset()
{
  reset_pending_ = true;
}

bool WebGPUPreview::initialize()
{
  if (pipeline_) {
    return true;
  }

  const auto camera = read_shader("stock-camera.wgsl");
  const auto bvh = read_shader("bvh2-compact.wgsl");
  const auto surface = read_shader("surface-compact.wgsl");
  const auto film = read_shader("film.wgsl");
  const auto state_init = read_shader("path-state.wgsl");
  const auto queue = read_shader("wavefront-queue.wgsl");
  pipeline_ = make_unique<WebGPURenderPipeline>(device_);
  return pipeline_->initialize(
             {camera, bvh, surface, film, state_init, queue, read_shader}) ||
         report_error();
}

bool WebGPUPreview::report_error()
{
  LOG_WARNING << "Cycles WebGPU using CPU: " << pipeline_->error_message();
  failed_ = true;
  pipeline_.reset();
  scene_ = {};
  std::vector<uint32_t>().swap(sampling_);
  return false;
}

bool WebGPUPreview::begin_batch(const BufferParams &params,
                                const int start_sample,
                                const int samples_num,
                                uint32_t &path_capacity,
                                const size_t film_words)
{
  if (failed_ || !scene_supported_ || !supports_passes(params, volume_) ||
      params.get_pass_offset(PASS_COMBINED) == PASS_UNUSED || params.full_x < 0 ||
      params.full_y < 0 || params.width <= 0 || params.height <= 0 ||
      start_sample < 0 || (reset_pending_ && start_sample != 0))
  {
    return false;
  }

  if (!initialize()) {
    return false;
  }
  if (scene_pending_) {
    if (!pipeline_->upload_scene(scene_.bvh,
                                 scene_.surface,
                                 scene_.lighting,
                                 scene_.svm,
                                 sampling_,
                                 scene_.has_subsurface,
                                 scene_.advanced_svm,
                                 scene_.ao_only_queries,
                                 scene_.residual_displacement_bump,
                                 scene_.paired_principled_layers,
                                 scene_.single_principled_coat,
                                 scene_.coated_principled_emission,
                                 scene_.compact_voronoi_distance,
                                 scene_.has_mixed_diffuse))
    {
      return report_error();
    }
    scene_pending_ = false;
    scene_ = {};
    std::vector<uint32_t>().swap(sampling_);
  }
  if (reset_pending_) {
    if (!pipeline_->reset(
            camera_, camera_lens_, denoising_config_, params, film_words))
    {
      return report_error();
    }
    reset_pending_ = false;
  }
  return pipeline_->begin_batch(start_sample, samples_num, path_capacity) || report_error();
}

bool WebGPUPreview::enqueue_camera(const std::span<const KernelWorkTile> tiles)
{
  return pipeline_->enqueue_camera(tiles) || report_error();
}

bool WebGPUPreview::enqueue_stage(const DeviceKernel stage)
{
  return pipeline_->enqueue_stage(stage) || report_error();
}

bool WebGPUPreview::read_queue(IntegratorQueueCounter &counts, bool &samples_overlap)
{
  return pipeline_->read_queue(counts, samples_overlap) || report_error();
}

bool WebGPUPreview::finish_batch()
{
  return pipeline_->finish_batch() || report_error();
}

bool WebGPUPreview::adaptive_filter(const float threshold, const bool reset, int &active_pixels)
{
  return pipeline_ &&
         (pipeline_->adaptive_filter(threshold, reset, active_pixels) || report_error());
}

bool WebGPUPreview::append_resident_film(const WebGPUOIDNFilmTile &tile)
{
  return pipeline_ && pipeline_->append_resident_film(tile);
}

bool WebGPUPreview::publish_preview(const BufferParams &params,
                                   const float exposure,
                                   const int samples,
                                   const uint32_t ticket)
{
  return pipeline_ && pipeline_->publish_preview(params, exposure, samples, ticket);
}

bool WebGPUPreview::read_film(RenderBuffers &buffers, const BufferParams & /*params*/)
{
  return pipeline_ &&
         (pipeline_->read_film(std::span(buffers.buffer.data(), buffers.buffer.data_size)) ||
          report_error());
}

CCL_NAMESPACE_END
