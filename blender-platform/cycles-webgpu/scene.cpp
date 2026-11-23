/* SPDX-License-Identifier: Apache-2.0 */

#include "device/webgpu/scene.h"

#include "bvh/bvh2.h"
#include "device/webgpu/device.h"
#include "device/webgpu/lighting_layout.h"
#include "device/webgpu/material_layout.h"
#include "device/webgpu/svm_materials.h"
#include "device/webgpu/svm_world_eval.h"
#include "device/webgpu/volume_grids.h"
#include "device/webgpu/volume_layout.h"
#include "scene/object.h"
#include "scene/scene.h"
#include "scene/shader.h"
#include "scene/shader_nodes.h"

#include <algorithm>
#include <bit>
#include <cfloat>
#include <cmath>
#include <fstream>
#include <new>
#include <span>
#include <utility>

CCL_NAMESPACE_BEGIN

namespace {

template<typename T> std::span<const T> host_data(const device_vector<T> &data)
{
  return {data.data(), data.size()};
}

uint32_t float_word(const float value)
{
  return std::bit_cast<uint32_t>(value);
}

void write_rgb(uint32_t *words, const float3 value)
{
  words[0] = float_word(value.x);
  words[1] = float_word(value.y);
  words[2] = float_word(value.z);
}

struct PackedWords {
  std::vector<uint32_t> words;

  void append(const int4 value)
  {
    words.insert(words.end(),
                 {uint32_t(value.x), uint32_t(value.y), uint32_t(value.z), uint32_t(value.w)});
  }

  void append(const Transform &transform)
  {
    for (const float4 row : {transform.x, transform.y, transform.z}) {
      words.insert(words.end(),
                   {float_word(row.x), float_word(row.y), float_word(row.z), float_word(row.w)});
    }
  }

  template<typename T> void append(const size_t offset, const device_vector<T> &data)
  {
    words[offset] = uint32_t(words.size());
    const std::span<const T> values = host_data(data);
    if (!values.empty()) {
      words.insert(words.end(), values.begin(), values.end());
    }
  }
};

bool finite_rgb(const float3 value)
{
  return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

bool constant_emission(const DeviceScene &dscene, const int shader, float3 &emission)
{
  const uint index = shader & SHADER_MASK;
  if (index >= dscene.shaders.size()) {
    return false;
  }
  const KernelShader &kernel_shader = dscene.shaders.data()[index];
  if (!(kernel_shader.flags & SD_HAS_CONSTANT_EMISSION)) {
    return false;
  }
  const float *rgb = kernel_shader.constant_emission;
  emission = make_float3(rgb[0], rgb[1], rgb[2]);
  return finite_rgb(emission);
}

const char *scene_incompatibility(const Scene &scene)
{
  const DeviceScene &dscene = scene.dscene;
  const KernelBVH &bvh = dscene.data.bvh;
  if (bvh.bvh_layout != BVH_LAYOUT_BVH2 || bvh.have_motion || bvh.have_curves || bvh.have_points ||
      scene.params.shadingsystem != SHADINGSYSTEM_SVM)
  {
    return "WebGPU Cycles needs a static triangle BVH2 and SVM materials";
  }

  if (!dscene.tri_shader.size() || dscene.tri_shader.size() != dscene.tri_vindex.size() ||
      !dscene.tri_verts.size() || !dscene.bvh_leaf_nodes.size() ||
      dscene.bvh_nodes.size() % BVH_NODE_SIZE ||
      dscene.object_flag.size() != dscene.objects.size() ||
      dscene.object_node.size() != dscene.objects.size() ||
      dscene.prim_index.size() != dscene.prim_type.size() ||
      dscene.prim_index.size() != dscene.prim_visibility.size() ||
      dscene.prim_index.size() != dscene.prim_object.size())
  {
    return "WebGPU Cycles needs synchronized mesh and BVH2 buffers";
  }

  for (size_t index = 0; index < dscene.bvh_nodes.size(); index += BVH_NODE_SIZE) {
    const int4 node = dscene.bvh_nodes.data()[index];
    if ((node.x | node.y) & PATH_RAY_VISIBILITY_NODE_UNALIGNED) {
      return "WebGPU Cycles needs aligned BVH2 nodes";
    }
  }
  for (size_t index = 0; index < dscene.prim_index.size(); ++index) {
    /* A primitive index of -1 identifies an instance node. */
    if (dscene.prim_index.data()[index] != -1 &&
        dscene.prim_type.data()[index] != PRIMITIVE_TRIANGLE)
    {
      return "WebGPU Cycles supports static mesh triangles";
    }
  }

  constexpr uint object_flags = SD_OBJECT_HOLDOUT_MASK | SD_OBJECT_SHADOW_CATCHER |
                                SD_OBJECT_HAS_VERTEX_MOTION | SD_OBJECT_MOTION;
  constexpr uint64_t features = KERNEL_FEATURE_LIGHT_LINKING | KERNEL_FEATURE_SHADOW_LINKING;
  if (dscene.data.kernel_features & features) {
    return "WebGPU Cycles does not support light or shadow linking";
  }
  for (size_t index = 0; index < dscene.objects.size(); ++index) {
    /* Cycles marks the world light as a shadow catcher even in ordinary opaque renders. */
    if (dscene.objects.data()[index].primitive_type != PRIMITIVE_LAMP &&
        (dscene.object_flag.data()[index] & object_flags))
    {
      return "WebGPU Cycles does not support mesh motion, holdouts, or shadow catchers";
    }
  }

  if (dscene.data.background.volume_shader != SHADER_NONE ||
      dscene.data.background.transparent_roughness_squared_threshold >= 0.0f)
  {
    return "WebGPU Cycles does not support world volume or transparent glass";
  }
  const KernelIntegrator &integrator = dscene.data.integrator;
  if (integrator.ao_bounces || integrator.ao_additive_factor != 0.0f) {
    return "WebGPU Cycles needs Fast GI ambient occlusion disabled";
  }
  if (integrator.filter_closures &
      (FILTER_CLOSURE_DIFFUSE | FILTER_CLOSURE_EMISSION | FILTER_CLOSURE_DIRECT_LIGHT))
  {
    return "WebGPU Cycles needs diffuse, emission, and direct light closures enabled";
  }
  if (!std::isfinite(integrator.light_inv_rr_threshold) || integrator.light_inv_rr_threshold < 0.0f) {
    return "WebGPU Cycles needs a valid light sampling threshold";
  }
  return nullptr;
}

bool lamp_uses_varying_emission(const DeviceScene &dscene, const KernelLight &lamp)
{
  const uint shader = lamp.shader_id & SHADER_MASK;
  return (lamp.type == LIGHT_POINT || lamp.type == LIGHT_AREA || lamp.type == LIGHT_SUN) &&
         shader < dscene.shaders.size() &&
         !(dscene.shaders.data()[shader].flags & SD_HAS_CONSTANT_EMISSION);
}

bool pack_lamp_ray_length(MixClosureNode &mix, uint32_t *record)
{
  const ShaderOutput *first = mix.input("Closure1")->link;
  const ShaderOutput *second = mix.input("Closure2")->link;
  const ShaderOutput *emission_output = first ? first : second;
  if (bool(first) == bool(second) || emission_output->parent->type != EmissionNode::get_node_type()) {
    return false;
  }
  auto &emission = static_cast<EmissionNode &>(*emission_output->parent);
  if (emission.input("Color")->link || emission.input("Strength")->link ||
      !finite_rgb(emission.get_color()) || !std::isfinite(emission.get_strength()))
  {
    return false;
  }

  /* The graph expands a clamped Map Range into Map Range followed by Clamp. */
  const ShaderOutput *factor = mix.input("Fac")->link;
  if (factor && factor->parent->type == ClampNode::get_node_type()) {
    auto &clamp_node = static_cast<ClampNode &>(*factor->parent);
    if (clamp_node.input("Min")->link || clamp_node.input("Max")->link ||
        clamp_node.get_min() != 0.0f || clamp_node.get_max() != 1.0f)
    {
      return false;
    }
    factor = clamp_node.input("Value")->link;
  }
  if (!factor || factor->parent->type != MapRangeNode::get_node_type()) {
    return false;
  }
  auto &range = static_cast<MapRangeNode &>(*factor->parent);
  if (range.get_range_type() != NODE_MAP_RANGE_SMOOTHERSTEP || range.input("From Min")->link ||
      range.input("From Max")->link || range.input("To Min")->link || range.input("To Max")->link ||
      range.get_from_min() != 0.0f || range.get_from_max() != 1.0f ||
      range.get_to_min() != 0.0f || range.get_to_max() != 1.0f)
  {
    return false;
  }

  const ShaderOutput *distance = range.input("Value")->link;
  float divisor = 1.0f;
  if (distance && distance->parent->type == MathNode::get_node_type()) {
    auto &math = static_cast<MathNode &>(*distance->parent);
    if (math.get_math_type() != NODE_MATH_DIVIDE || math.input("Value2")->link ||
        !std::isfinite(math.get_value2()))
    {
      return false;
    }
    divisor = math.get_value2();
    distance = math.input("Value1")->link;
  }
  if (!distance || distance->parent->type != LightPathNode::get_node_type() ||
      distance != distance->parent->output("Ray Length"))
  {
    return false;
  }

  record[CYCLES_WEBGPU_LAMP_FALLOFF_MODE] = first ? CYCLES_WEBGPU_LAMP_RAY_LENGTH_INVERTED :
                                                  CYCLES_WEBGPU_LAMP_RAY_LENGTH;
  write_rgb(record + CYCLES_WEBGPU_LAMP_FALLOFF_COLOR, emission.get_color());
  record[CYCLES_WEBGPU_LAMP_FALLOFF_STRENGTH] = float_word(emission.get_strength());
  record[CYCLES_WEBGPU_LAMP_RAY_LENGTH_DIVISOR] = float_word(divisor);
  return true;
}

bool pack_lamp_falloff(const Scene &scene, const KernelLight &lamp, uint32_t *record)
{
  const uint shader = lamp.shader_id & SHADER_MASK;
  if (shader >= scene.shaders.size()) {
    return false;
  }
  const ShaderOutput *surface = scene.shaders[shader]->graph->output()->input("Surface")->link;
  if (!surface) {
    return false;
  }
  if (surface->parent->type == MixClosureNode::get_node_type()) {
    return pack_lamp_ray_length(static_cast<MixClosureNode &>(*surface->parent), record);
  }
  if (surface->parent->type != EmissionNode::get_node_type()) {
    return false;
  }
  auto &emission = static_cast<EmissionNode &>(*surface->parent);
  const ShaderOutput *strength = emission.input("Strength")->link;
  if (emission.input("Color")->link || !strength ||
      strength->parent->type != LightFalloffNode::get_node_type())
  {
    return false;
  }
  auto &falloff = static_cast<LightFalloffNode &>(*strength->parent);
  if (falloff.input("Strength")->link || falloff.input("Smooth")->link ||
      !finite_rgb(emission.get_color()) || !std::isfinite(falloff.get_strength()) ||
      !std::isfinite(falloff.get_smooth()))
  {
    return false;
  }
  NodeLightFalloff mode;
  if (strength == falloff.output("Quadratic")) {
    mode = NODE_LIGHT_FALLOFF_QUADRATIC;
  }
  else if (strength == falloff.output("Linear")) {
    mode = NODE_LIGHT_FALLOFF_LINEAR;
  }
  else if (strength == falloff.output("Constant")) {
    mode = NODE_LIGHT_FALLOFF_CONSTANT;
  }
  else {
    return false;
  }
  record[CYCLES_WEBGPU_LAMP_FALLOFF_MODE] = uint32_t(mode) + 1;
  write_rgb(record + CYCLES_WEBGPU_LAMP_FALLOFF_COLOR, emission.get_color());
  record[CYCLES_WEBGPU_LAMP_FALLOFF_STRENGTH] = float_word(falloff.get_strength());
  record[CYCLES_WEBGPU_LAMP_FALLOFF_SMOOTH] = float_word(falloff.get_smooth());
  return true;
}

const char *encode_lamp(const DeviceScene &dscene,
                        const KernelLight &light,
                        const bool allow_area,
                        uint32_t *words,
                        const bool allow_finite_point = false,
                        const bool allow_sun = false,
                        const bool varying_emission = false)
{
  const bool point = light.type == LIGHT_POINT &&
                     ((light.spot.radius == 0.0f && !light.spot.is_sphere &&
                       light.spot.eval_fac == 0.25f * M_1_PI_F) ||
                      (allow_finite_point && std::isfinite(light.spot.radius) &&
                       light.spot.radius > 0.0f && std::isfinite(light.spot.eval_fac) &&
                       light.spot.eval_fac > 0.0f));
  const bool area = allow_area && light.type == LIGHT_AREA;
  const bool sun = allow_sun && light.type == LIGHT_SUN;
  if ((!point && !area && !sun) ||
      (light.shader_id & SHADER_EXCLUDE_DIFFUSE) || !std::isfinite(light.max_bounces) ||
      light.max_bounces < 0.0f)
  {
    return "WebGPU Cycles needs a supported lamp with diffuse lighting";
  }

  float3 light_emission = one_float3();
  if (!varying_emission && !constant_emission(dscene, light.shader_id, light_emission)) {
    return "WebGPU Cycles needs constant light emission";
  }

  const float3 position = make_float3(light.co);
  const float3 strength = make_float3(light.strength[0], light.strength[1], light.strength[2]) *
                          light_emission;
  if (!finite_rgb(position) || !finite_rgb(strength)) {
    return "WebGPU Cycles needs finite light position and strength";
  }
  write_rgb(words, position);
  words[3] = float_word(light.max_bounces);
  write_rgb(words + 4, strength);
  words[7] = uint32_t(light.shader_id);
  return nullptr;
}

bool uses_mixed_lighting(const DeviceScene &scene)
{
  const KernelIntegrator &integrator = scene.data.integrator;
  const auto lights = host_data(scene.lights);
  if (integrator.num_background_lights > 0 ||
      (integrator.use_light_tree ? scene.light_tree_emitters.size() > lights.size() :
                                   scene.light_distribution.size() > lights.size() + 1))
  {
    return true;
  }
  if (lights.empty()) {
    return false;
  }
  if (integrator.use_light_tree && lights.size() > 1) {
    return true;
  }
  for (const KernelLight &light : lights) {
    if (lamp_uses_varying_emission(scene, light) || !(light.shader_id & SHADER_CAST_SHADOW) ||
        light.type == LIGHT_SUN ||
        (light.type == LIGHT_POINT && light.spot.radius > 0.0f) ||
        (light.type == LIGHT_AREA &&
         (!(light.shader_id & SHADER_USE_MIS) || lights.size() > 1 || light.area.invarea < 0.0f ||
          light.area.tan_half_spread != FLT_MAX)))
    {
      return true;
    }
  }
  if (lights.front().type != LIGHT_POINT) {
    return false;
  }
  /* The compact point paths do not carry indirect environment lighting. */
  constexpr int indirect = SHADER_EXCLUDE_DIFFUSE | SHADER_EXCLUDE_GLOSSY;
  const int world_shader = scene.data.background.surface_shader;
  float3 emission;
  return (world_shader & indirect) != indirect && constant_emission(scene, world_shader, emission) &&
         !is_zero(emission);
}

const char *encode_lighting(const Scene &scene,
                            const bool mixed_lighting,
                            std::vector<uint32_t> &lighting)
{
  const DeviceScene &dscene = scene.dscene;
  const KernelIntegrator &integrator = dscene.data.integrator;
  const auto lights = host_data(dscene.lights);
  if ((!mixed_lighting && lights.empty()) || lights.size() != size_t(integrator.num_lights) ||
      integrator.num_portals || !integrator.use_direct_light)
  {
    return "WebGPU Cycles needs supported lights without portals";
  }
  lighting.assign(CYCLES_WEBGPU_LIGHT_HEADER_WORDS, 0);
  lighting[CYCLES_WEBGPU_LIGHT_DIRECT_CLAMP] = float_word(integrator.sample_clamp_direct);
  lighting[3] = float_word(integrator.sample_clamp_indirect);
  lighting[CYCLES_WEBGPU_LIGHT_FILTER_GLOSSY] = float_word(integrator.filter_glossy);
  if (mixed_lighting) {
    return nullptr;
  }
  const bool area = lights.size() == 1 && lights.front().type == LIGHT_AREA;
  float3 background = zero_float3();
  const int background_shader = dscene.data.background.surface_shader;
  const bool camera_background = !(background_shader & SHADER_EXCLUDE_CAMERA);
  const bool diffuse_background = !(background_shader & SHADER_EXCLUDE_DIFFUSE);
  if (!area && (camera_background || diffuse_background) &&
      !constant_emission(dscene, background_shader, background))
  {
    return "WebGPU Cycles needs a constant world color";
  }
  if (!area && diffuse_background &&
      (background.x != 0.0f || background.y != 0.0f || background.z != 0.0f))
  {
    return "WebGPU Cycles needs a black world or diffuse world ray visibility disabled";
  }
  if (!camera_background) {
    background = zero_float3();
  }

  if (lights.size() == 1) {
    if (const char *error = encode_lamp(dscene, lights.front(), true, lighting.data())) {
      return error;
    }
    lighting[CYCLES_WEBGPU_LIGHT_EXTENDED] = 0;
  }
  lighting[3] = float_word(integrator.sample_clamp_indirect);
  write_rgb(lighting.data() + 8, background);
  return nullptr;
}

std::span<const float> light_sampling_table(const DeviceScene &dscene)
{
  return dscene.data.integrator.sampling_pattern == SAMPLING_PATTERN_TABULATED_SOBOL ?
             host_data(dscene.sample_pattern_lut) :
             std::span<const float>{};
}

void append_light_sampling(const KernelIntegrator &integrator,
                           const std::span<const float> lut,
                           std::vector<uint32_t> &lighting)
{
  lighting.insert(lighting.end(),
                  {uint32_t(integrator.sampling_pattern),
                   uint32_t(integrator.seed),
                   uint32_t(integrator.sobol_index_mask),
                   uint32_t(integrator.blue_noise_sequence_length),
                   uint32_t(integrator.tabulated_sobol_sequence_size),
                   float_word(integrator.pixel_jitter.x),
                   float_word(integrator.pixel_jitter.y),
                   float_word(integrator.scrambling_distance)});
  for (const float word : lut) {
    lighting.push_back(float_word(word));
  }
}

const char *encode_point_lighting(const DeviceScene &dscene,
                                  const size_t remaining_words,
                                  std::vector<uint32_t> &lighting)
{
  const KernelIntegrator &integrator = dscene.data.integrator;
  const auto lights = host_data(dscene.lights);
  const auto distribution = host_data(dscene.light_distribution);
  const float pdf = integrator.distribution_pdf_lights;
  if (integrator.use_light_tree || integrator.num_distribution != integrator.num_lights ||
      distribution.size() != lights.size() + 1 || distribution.front().totarea != 0.0f ||
      distribution.back().totarea != 1.0f || !std::isfinite(pdf) || pdf <= 0.0f ||
      (dscene.data.kernel_features & KERNEL_FEATURE_MNEE))
  {
    return "WebGPU Cycles multiple point lights need the stock lamp distribution without Light Tree";
  }

  const auto lut = light_sampling_table(dscene);
  const uint64_t distribution_words = uint64_t(distribution.size()) *
                                      CYCLES_WEBGPU_POINT_DISTRIBUTION_WORDS;
  const uint64_t lamp_words = uint64_t(lights.size()) * CYCLES_WEBGPU_POINT_EMITTER_WORDS;
  const uint64_t total_words = CYCLES_WEBGPU_POINT_DESCRIPTOR_WORDS + distribution_words +
                               lamp_words + CYCLES_WEBGPU_LIGHT_SAMPLING_WORDS + lut.size();
  if (total_words > remaining_words) {
    return "WebGPU Cycles point lighting exceeds the scene upload budget";
  }

  const size_t descriptor = lighting.size();
  const size_t cdf = descriptor + CYCLES_WEBGPU_POINT_DESCRIPTOR_WORDS;
  const size_t emitters = cdf + size_t(distribution_words);
  const size_t sampling = emitters + size_t(lamp_words);
  lighting[0] = CYCLES_WEBGPU_POINT_LIGHTS_MARKER;
  lighting[CYCLES_WEBGPU_POINT_LIGHTS_OFFSET] = uint32_t(descriptor);
  lighting.resize(cdf);
  lighting[descriptor + CYCLES_WEBGPU_POINT_COUNT] = uint32_t(lights.size());
  lighting[descriptor + CYCLES_WEBGPU_POINT_PDF] = float_word(pdf);
  lighting[descriptor + CYCLES_WEBGPU_POINT_SAMPLING] = uint32_t(sampling);
  lighting[descriptor + CYCLES_WEBGPU_POINT_CDF] = uint32_t(cdf);
  lighting[descriptor + CYCLES_WEBGPU_POINT_EMITTERS] = uint32_t(emitters);

  for (size_t index = 0; index < distribution.size(); ++index) {
    const KernelLightDistribution &entry = distribution[index];
    if (index < lights.size() &&
        (entry.prim != ~int(index) || !(entry.totarea < distribution[index + 1].totarea)))
    {
      return "WebGPU Cycles needs a synchronized distribution of point lights";
    }
    lighting.insert(lighting.end(),
                    {float_word(entry.totarea),
                     uint32_t(entry.prim),
                     uint32_t(entry.visibility_flag),
                     uint32_t(entry.object_id)});
  }
  lighting.resize(sampling);
  for (size_t index = 0; index < lights.size(); ++index) {
    uint32_t *words = lighting.data() + emitters + index * CYCLES_WEBGPU_POINT_EMITTER_WORDS;
    if (const char *error = encode_lamp(dscene, lights[index], false, words)) {
      return error;
    }
  }
  append_light_sampling(integrator, lut, lighting);
  return nullptr;
}

const char *area_light_incompatibility(const DeviceScene &dscene,
                                       const KernelLight &light,
                                       const bool full_rectangle = true)
{
  const KernelAreaLight &area = light.area;
  const KernelIntegrator &integrator = dscene.data.integrator;
  if (integrator.direct_light_sampling_type != DIRECT_LIGHT_SAMPLING_MIS ||
      (full_rectangle && (area.tan_half_spread != FLT_MAX || area.invarea < 0.0f)) ||
      !std::isfinite(area.tan_half_spread) || area.tan_half_spread < 0.0f ||
      !std::isfinite(area.normalize_spread) || area.normalize_spread < 0.0f ||
      !std::isfinite(area.len_u) || !std::isfinite(area.len_v) || !std::isfinite(area.invarea) ||
      area.len_u <= 0.0f || area.len_v <= 0.0f || area.invarea == 0.0f ||
      !finite_rgb(make_float3(area.axis_u)) || !finite_rgb(make_float3(area.axis_v)) ||
      !finite_rgb(make_float3(area.dir)) ||
      (dscene.data.kernel_features & KERNEL_FEATURE_MNEE))
  {
    return full_rectangle ? "WebGPU Cycles needs a valid rectangular area light with full spread" :
                            "WebGPU Cycles needs a valid area light";
  }
  return nullptr;
}

struct WorldEnvironment {
  KernelImageInfo image{};
  float strength = 1.0f;
  float3 background = zero_float3();
  std::span<const int> sky_parameters;
  std::span<const int> program;

  uint32_t interpolation() const
  {
    return (image.width ? uint32_t(image.interpolation) : 0u) |
           (sky_parameters.empty() ? 0u : CYCLES_WEBGPU_WORLD_SKY_FLAG) |
           (program.empty() ? 0u : CYCLES_WEBGPU_WORLD_SVM_FLAG);
  }

  uint64_t word_count() const
  {
    const size_t sky_words = sky_parameters.empty() ? 0 : CYCLES_WEBGPU_SKY_HEADER_WORDS;
    const size_t program_words = program.empty() ? 0 : CYCLES_WEBGPU_WORLD_SVM_HEADER_WORDS +
                                                        program.size();
    return sky_words + program_words + 3 * uint64_t(image.width) * image.height;
  }
};

std::span<const int> world_surface_code(const Scene &scene, const uint shader)
{
  const auto code = host_data(scene.dscene.svm_nodes);
  const uint count = uint(code.size());
  const uint shader_count = uint(scene.dscene.shaders.size());
  if (shader >= shader_count || shader_count > count / webgpu_svm_jump_words) {
    return {};
  }

  const auto fetch = [&](const uint index) { return uint(code[index]); };
  uint offset = shader * webgpu_svm_jump_words;
  SVMNodeShaderJump jump;
  if (fetch(offset++) != NODE_SHADER_JUMP || !webgpu_svm_read(fetch, offset, count, jump) ||
      uint(jump.offset_surface) < shader_count * webgpu_svm_jump_words ||
      uint(jump.offset_surface) >= uint(jump.offset_volume) || uint(jump.offset_volume) > count)
  {
    return {};
  }

  /* In an untransformed world, stock Generated compiles to Geometry Position.
   * shader_setup_from_background() sets that position to the ray direction. */
  return code.subspan(uint(jump.offset_surface),
                      uint(jump.offset_volume) - uint(jump.offset_surface));
}

bool load_world_sky(const Scene &scene, const uint shader, WorldEnvironment &environment)
{
  const auto code = world_surface_code(scene, shader);
  if (code.empty()) {
    return false;
  }
  const auto fetch = [&](const uint index) { return uint(code[index]); };
  uint offset = 0;
  const uint end = uint(code.size());
  SVMNodeGeometry direction;
  SVMNodeTexSky sky;
  if (fetch(offset++) != NODE_GEOMETRY || !webgpu_svm_read(fetch, offset, end, direction) ||
      direction.geom_type != NODE_GEOM_P || offset >= end || fetch(offset++) != NODE_TEX_SKY ||
      !webgpu_svm_read(fetch, offset, end, sky) || sky.dir_offset != direction.out_offset ||
      (sky.sky_type != NODE_SKY_SINGLE_SCATTERING && sky.sky_type != NODE_SKY_MULTIPLE_SCATTERING))
  {
    return false;
  }
  const uint parameters = offset;
  SVMNodeTexSkyNishitaData data;
  static_assert(sizeof(data) / sizeof(uint32_t) ==
                CYCLES_WEBGPU_SKY_HEADER_WORDS - CYCLES_WEBGPU_SKY_PARAMETERS);
  if (!webgpu_svm_read(fetch, offset, end, data) ||
      !cycles_webgpu_svm_image_info(scene, int(data.texture_id), environment.image) ||
      environment.image.data_type != IMAGE_DATA_TYPE_FLOAT4 ||
      environment.image.extension != EXTENSION_EXTEND ||
      environment.image.interpolation != INTERPOLATION_LINEAR)
  {
    return false;
  }
  environment.sky_parameters = code.subspan(parameters, offset - parameters);
  return true;
}

bool load_world_image(const Scene &scene, const int id, KernelImageInfo &image)
{
  return cycles_webgpu_svm_image_info(scene, id, image) &&
         image.data_type == IMAGE_DATA_TYPE_FLOAT4 && image.extension == EXTENSION_REPEAT &&
         (image.interpolation == INTERPOLATION_CLOSEST || image.interpolation == INTERPOLATION_LINEAR);
}

bool load_world_program(const Scene &scene, const uint shader, WorldEnvironment &environment)
{
  const auto code = world_surface_code(scene, shader);
  const auto fetch = [&](const uint index) { return uint(code[index]); };
  const auto image = [&](const int id, const float2, const bool alpha, float4 &color) {
    color = make_float4(1.0f);
    return !alpha && load_world_image(scene, id, environment.image);
  };
  Spectrum radiance;
  if (!webgpu_svm_eval_world(fetch,
                             image,
                             0,
                             uint(code.size()),
                             make_float3(0.3f, 0.4f, 0.8660254f),
                             make_float3(scene.dscene.data.film.rgb_to_y),
                             radiance))
  {
    return false;
  }
  environment.program = code;
  return true;
}

const char *find_world_environment(const Scene &scene, WorldEnvironment &environment)
{
  const DeviceScene &dscene = scene.dscene;
  const int world = dscene.data.background.surface_shader;
  if ((world & SHADER_EXCLUDE_ANY) == SHADER_EXCLUDE_ANY ||
      constant_emission(dscene, world, environment.background))
  {
    return nullptr;
  }

  const uint shader = uint(world) & SHADER_MASK;
  if (shader >= scene.shaders.size() || !scene.shaders[shader]->graph) {
    return "WebGPU Cycles needs a synchronized world shader";
  }
  const ShaderOutput *surface = scene.shaders[shader]->graph->output()->input("Surface")->link;
  if (!surface || surface->parent->type != BackgroundNode::get_node_type()) {
    return "WebGPU Cycles needs a constant world or one Background closure";
  }
  auto &node = static_cast<BackgroundNode &>(*surface->parent);
  const ShaderOutput *color = node.input("Color")->link;
  environment.strength = node.get_strength();
  const bool constant_strength = !node.input("Strength")->link &&
                                 std::isfinite(environment.strength) &&
                                 environment.strength >= 0.0f;
  if (constant_strength && color && color->name() == "Color") {
    if (color->parent->type == SkyTextureNode::get_node_type()) {
      return load_world_sky(scene, shader, environment) ?
                 nullptr : "WebGPU Cycles needs an untransformed Nishita Sky Texture world";
    }
    if (color->parent->type == EnvironmentTextureNode::get_node_type()) {
      auto &texture = static_cast<EnvironmentTextureNode &>(*color->parent);
      const ShaderOutput *vector = texture.input("Vector")->link;
      /* ShaderGraph::default_inputs wires the background direction through Geometry Position. */
      if (!texture.get_animated() && texture.tex_mapping.skip() &&
          texture.get_projection() == NODE_ENVIRONMENT_EQUIRECTANGULAR &&
          (!vector ||
           (vector->parent->type == GeometryNode::get_node_type() && vector->name() == "Position")) &&
          !texture.handle.empty() &&
          load_world_image(scene, texture.handle.kernel_id(), environment.image))
      {
        return nullptr;
      }
    }
  }

  return load_world_program(scene, shader, environment) ?
             nullptr : "WebGPU Cycles needs a supported world texture and Background shader graph";
}

const char *append_world_environment(const DeviceScene &scene,
                                     const WorldEnvironment &environment,
                                     std::vector<uint32_t> &lighting)
{
  const bool sky = !environment.sky_parameters.empty();
  if (sky) {
    const size_t base = lighting.size();
    lighting.resize(base + CYCLES_WEBGPU_SKY_PARAMETERS);
    uint32_t *header = lighting.data() + base;
    const auto &film = scene.data.film;
    const auto &world = scene.data.background;
    header[CYCLES_WEBGPU_SKY_STRENGTH] = float_word(environment.strength);
    write_rgb(header + CYCLES_WEBGPU_SKY_XYZ_TO_RGB, make_float3(film.xyz_to_r));
    write_rgb(header + CYCLES_WEBGPU_SKY_XYZ_TO_RGB + 3, make_float3(film.xyz_to_g));
    write_rgb(header + CYCLES_WEBGPU_SKY_XYZ_TO_RGB + 6, make_float3(film.xyz_to_b));
    write_rgb(header + CYCLES_WEBGPU_SKY_SUN, make_float3(world.sun));
    header[CYCLES_WEBGPU_SKY_SUN + 3] = float_word(world.sun.w);
    header[CYCLES_WEBGPU_SKY_SUN_WEIGHT] = float_word(world.sun_weight);
    lighting.insert(lighting.end(), environment.sky_parameters.begin(), environment.sky_parameters.end());
  }
  const bool program = !environment.program.empty();
  if (program) {
    const size_t base = lighting.size();
    lighting.resize(base + CYCLES_WEBGPU_WORLD_SVM_HEADER_WORDS);
    uint32_t *header = lighting.data() + base;
    header[CYCLES_WEBGPU_WORLD_SVM_WORD_COUNT] = uint32_t(environment.program.size());
    write_rgb(header + CYCLES_WEBGPU_WORLD_SVM_RGB_TO_Y, make_float3(scene.data.film.rgb_to_y));
    lighting.insert(lighting.end(), environment.program.begin(), environment.program.end());
  }
  const KernelImageInfo &image = environment.image;
  const float scale = sky || program ? 1.0f : environment.strength;
  const size_t texels = size_t(image.width) * image.height;
  const auto *pixels = reinterpret_cast<const float4 *>(uintptr_t(image.data));
  for (size_t i = 0; i < texels; ++i) {
    const float3 rgb = scale * make_float3(pixels[i]);
    if (!finite_rgb(rgb)) {
      return "WebGPU Cycles needs finite world texture colors";
    }
    lighting.insert(lighting.end(), {float_word(rgb.x), float_word(rgb.y), float_word(rgb.z)});
  }
  return nullptr;
}

/* Append a sampled emitter and optional environment after the shared GGX tables.
 * The environment contributes on ray misses, so this subset uses World sampling None. */
const char *encode_area_lighting(const Scene &scene,
                                 const WorldEnvironment &environment,
                                 const size_t remaining_words,
                                 std::vector<uint32_t> &lighting)
{
  const DeviceScene &dscene = scene.dscene;
  const KernelLight &light = dscene.lights.data()[0];
  const KernelAreaLight &area = light.area;
  const KernelIntegrator &integrator = dscene.data.integrator;
  if (const char *error = area_light_incompatibility(dscene, light)) {
    return error;
  }
  const int world = dscene.data.background.surface_shader;
  constexpr size_t descriptor_words = CYCLES_WEBGPU_EXTENDED_LIGHT_WORDS;
  constexpr size_t sampling_words = CYCLES_WEBGPU_LIGHT_SAMPLING_WORDS;
  const auto lut = light_sampling_table(dscene);
  if (descriptor_words + sampling_words + lut.size() > remaining_words ||
      environment.word_count() > remaining_words - descriptor_words - sampling_words - lut.size())
  {
    return "WebGPU Cycles area lighting exceeds the scene upload budget";
  }

  const size_t base = lighting.size();
  lighting[CYCLES_WEBGPU_LIGHT_EXTENDED] = uint32_t(base);
  write_rgb(lighting.data() + 8, environment.background);
  lighting.resize(base + descriptor_words);
  write_rgb(lighting.data() + base + CYCLES_WEBGPU_AREA_AXIS_U, make_float3(area.axis_u));
  lighting[base + CYCLES_WEBGPU_AREA_AXIS_U + 3] = float_word(area.len_u);
  write_rgb(lighting.data() + base + CYCLES_WEBGPU_AREA_AXIS_V, make_float3(area.axis_v));
  lighting[base + CYCLES_WEBGPU_AREA_AXIS_V + 3] = float_word(area.len_v);
  write_rgb(lighting.data() + base + CYCLES_WEBGPU_AREA_DIRECTION, make_float3(area.dir));
  lighting[base + CYCLES_WEBGPU_AREA_DIRECTION + 3] = float_word(area.invarea);
  lighting[base + CYCLES_WEBGPU_WORLD_WIDTH] = environment.image.width;
  lighting[base + CYCLES_WEBGPU_WORLD_HEIGHT] = environment.image.height;
  lighting[base + CYCLES_WEBGPU_WORLD_SHADER] = uint32_t(world);
  lighting[base + CYCLES_WEBGPU_AREA_SHADER] = uint32_t(light.shader_id);
  lighting[base + CYCLES_WEBGPU_WORLD_INTERPOLATION] = environment.interpolation();
  if (const char *error = append_world_environment(dscene, environment, lighting)) {
    return error;
  }
  lighting[base + CYCLES_WEBGPU_LIGHT_SAMPLING] = uint32_t(lighting.size());
  append_light_sampling(integrator, lut, lighting);
  return nullptr;
}

bool pack_light_tree_nodes(const std::span<const KernelLightTreeNode> nodes,
                           const size_t emitter_count,
                           const std::span<uint32_t> words)
{
  for (size_t index = 0; index < nodes.size(); ++index) {
    const KernelLightTreeNode &node = nodes[index];
    uint32_t *record = words.data() + index * CYCLES_WEBGPU_TREE_NODE_WORDS;
    write_rgb(record + CYCLES_WEBGPU_TREE_NODE_MIN, make_float3(node.bbox.min));
    write_rgb(record + CYCLES_WEBGPU_TREE_NODE_MAX, make_float3(node.bbox.max));
    write_rgb(record + CYCLES_WEBGPU_TREE_NODE_AXIS, make_float3(node.bcone.axis));
    record[CYCLES_WEBGPU_TREE_NODE_THETA_O] = float_word(node.bcone.theta_o);
    record[CYCLES_WEBGPU_TREE_NODE_THETA_E] = float_word(node.bcone.theta_e);
    record[CYCLES_WEBGPU_TREE_NODE_ENERGY] = float_word(node.energy);
    record[CYCLES_WEBGPU_TREE_NODE_TYPE] = uint32_t(node.type);
    record[CYCLES_WEBGPU_TREE_NODE_EMITTER_COUNT] = uint32_t(node.num_emitters);
    record[CYCLES_WEBGPU_TREE_NODE_BIT_TRAIL] = node.bit_trail;
    record[CYCLES_WEBGPU_TREE_NODE_BIT_SKIP] = node.bit_skip;

    if (node.type == LIGHT_TREE_INNER) {
      const size_t left = size_t(node.inner.left_child);
      const size_t right = size_t(node.inner.right_child);
      if (left <= index || right <= index || left >= nodes.size() || right >= nodes.size() ||
          left == right)
      {
        return false;
      }
      record[CYCLES_WEBGPU_TREE_NODE_LEFT_OR_FIRST] = uint32_t(left);
      record[CYCLES_WEBGPU_TREE_NODE_RIGHT] = uint32_t(right);
    }
    else if (node.type == LIGHT_TREE_LEAF || node.type == LIGHT_TREE_DISTANT) {
      const size_t first = size_t(node.leaf.first_emitter);
      if (first > emitter_count || size_t(node.num_emitters) > emitter_count - first) {
        return false;
      }
      record[CYCLES_WEBGPU_TREE_NODE_LEFT_OR_FIRST] = uint32_t(first);
    }
    else if (node.type == LIGHT_TREE_INSTANCE) {
      const size_t reference = size_t(node.instance.reference);
      if (reference >= index ||
          (nodes[reference].type != LIGHT_TREE_INNER && nodes[reference].type != LIGHT_TREE_LEAF))
      {
        return false;
      }
      record[CYCLES_WEBGPU_TREE_NODE_LEFT_OR_FIRST] = uint32_t(reference);
    }
    else {
      return false;
    }
  }
  return true;
}

const char *pack_mixed_lamps(const Scene &scene,
                            const std::span<const KernelLightTreeEmitter> emitters,
                            const std::span<const KernelLightDistribution> distribution,
                            const std::span<uint32_t> words,
                            const std::span<uint32_t> falloff_words,
                            uint32_t &first_forward_index,
                            uint32_t &world_index,
                            uint32_t &world_emitter)
{
  const DeviceScene &dscene = scene.dscene;
  const auto lights = host_data(dscene.lights);
  const auto light_to_tree = host_data(dscene.light_to_tree);
  const bool use_tree = distribution.empty();
  for (size_t index = 0; index < lights.size(); ++index) {
    const KernelLight &light = lights[index];
    uint32_t emitter_index;
    if (use_tree) {
      if (size_t(light.object_id) >= light_to_tree.size()) {
        return "WebGPU Cycles needs synchronized Light Tree lamp objects";
      }
      emitter_index = light_to_tree[light.object_id];
      if (emitter_index >= emitters.size() || emitters[emitter_index].light.id != ~int(index) ||
          emitters[emitter_index].object_id != light.object_id)
      {
        return "WebGPU Cycles needs synchronized Light Tree lamp emitters";
      }
    }
    else {
      /* LightManager::device_update_distribution appends lamps after triangles. */
      emitter_index = uint32_t(distribution.size() - 1 - lights.size() + index);
      const KernelLightDistribution &entry = distribution[emitter_index];
      if (entry.prim != ~int(index) || entry.object_id != light.object_id) {
        return "WebGPU Cycles needs synchronized lamp distribution entries";
      }
    }

    uint32_t *record = words.data() + index * CYCLES_WEBGPU_TREE_LAMP_WORDS;
    record[CYCLES_WEBGPU_TREE_LAMP_TYPE] = uint32_t(light.type);
    record[CYCLES_WEBGPU_TREE_LAMP_SHADER] = uint32_t(light.shader_id);
    record[CYCLES_WEBGPU_TREE_LAMP_MAX_BOUNCES] = float_word(light.max_bounces);
    record[CYCLES_WEBGPU_TREE_LAMP_EMITTER] = emitter_index;

    if (light.type == LIGHT_BACKGROUND) {
      if (world_index != CYCLES_WEBGPU_TREE_NO_LAMP ||
          light.object_id != dscene.data.background.object_index ||
          !(light.shader_id & SHADER_CAST_SHADOW) || !(light.shader_id & SHADER_USE_MIS) ||
          (light.shader_id & (SHADER_EXCLUDE_DIFFUSE | SHADER_EXCLUDE_GLOSSY)) ||
          !std::isfinite(light.max_bounces) || light.max_bounces < 0.0f)
      {
        return "WebGPU Cycles needs one sampled world visible to diffuse and glossy light rays";
      }
      world_index = uint32_t(index);
      world_emitter = emitter_index;
      continue;
    }

    const bool varying_emission = lamp_uses_varying_emission(dscene, light);
    if (varying_emission &&
        !pack_lamp_falloff(scene, light, falloff_words.data() + index * CYCLES_WEBGPU_LAMP_FALLOFF_WORDS))
    {
      return "WebGPU Cycles needs a constant light shader or supported analytic light falloff";
    }
    uint32_t lamp[CYCLES_WEBGPU_POINT_EMITTER_WORDS]{};
    if (const char *error = encode_lamp(dscene, light, true, lamp, true, !use_tree, varying_emission)) {
      return error;
    }
    std::copy_n(lamp, 3, record + CYCLES_WEBGPU_TREE_LAMP_POSITION);
    std::copy_n(lamp + 4, 3, record + CYCLES_WEBGPU_TREE_LAMP_STRENGTH);
    if (light.type == LIGHT_SUN) {
      const KernelSunLight &sun = light.sun;
      if (!std::isfinite(sun.angle) || sun.angle < 0.0f ||
          !std::isfinite(sun.one_minus_cosangle) || sun.one_minus_cosangle < 0.0f ||
          !std::isfinite(sun.pdf) || sun.pdf <= 0.0f ||
          !std::isfinite(sun.eval_fac) || sun.eval_fac <= 0.0f)
      {
        return "WebGPU Cycles needs finite sun angles and sampling weights";
      }
      record[CYCLES_WEBGPU_LAMP_SUN_ANGLE] = float_word(sun.angle);
      record[CYCLES_WEBGPU_LAMP_SUN_ONE_MINUS_COS] = float_word(sun.one_minus_cosangle);
      record[CYCLES_WEBGPU_LAMP_SUN_PDF] = float_word(sun.pdf);
      record[CYCLES_WEBGPU_LAMP_SUN_EVAL] = float_word(sun.eval_fac);
      continue;
    }
    const bool finite_point = light.type == LIGHT_POINT && light.spot.radius > 0.0f;
    if (light.type == LIGHT_POINT) {
      /* Point and area lights share the same space for shape parameters. */
      record[CYCLES_WEBGPU_TREE_LAMP_AXIS_U] = float_word(light.spot.radius);
      record[CYCLES_WEBGPU_TREE_LAMP_AXIS_U + 1] = float_word(light.spot.eval_fac);
      record[CYCLES_WEBGPU_TREE_LAMP_AXIS_U + 2] = uint32_t(light.spot.is_sphere);
    }
    if (light.type != LIGHT_AREA && !finite_point) {
      continue;
    }
    if (dscene.data.integrator.use_light_mis &&
        first_forward_index == CYCLES_WEBGPU_TREE_NO_LAMP)
    {
      first_forward_index = uint32_t(index);
    }
    if (finite_point) {
      std::ifstream shader("/cycles-webgpu/wavefront-light-deferred-reflection-tree.wgsl");
      std::string version;
      if (!std::getline(shader, version) || version != "// Cycles Light Tree finite point") {
        return "WebGPU Cycles needs rebuilt Light Tree shaders for finite-radius point lights";
      }
      continue;
    }
    if (const char *error = area_light_incompatibility(dscene, light, use_tree)) {
      return error;
    }
    const KernelAreaLight &area = light.area;
    write_rgb(record + CYCLES_WEBGPU_TREE_LAMP_AXIS_U, make_float3(area.axis_u));
    record[CYCLES_WEBGPU_TREE_LAMP_AXIS_U + 3] = float_word(area.len_u);
    write_rgb(record + CYCLES_WEBGPU_TREE_LAMP_AXIS_V, make_float3(area.axis_v));
    record[CYCLES_WEBGPU_TREE_LAMP_AXIS_V + 3] = float_word(area.len_v);
    write_rgb(record + CYCLES_WEBGPU_TREE_LAMP_DIRECTION, make_float3(area.dir));
    record[CYCLES_WEBGPU_TREE_LAMP_DIRECTION + 3] = float_word(area.invarea);
    record[CYCLES_WEBGPU_TREE_LAMP_SPREAD] = float_word(area.tan_half_spread);
    record[CYCLES_WEBGPU_TREE_LAMP_NORMALIZE_SPREAD] = float_word(area.normalize_spread);
  }
  return nullptr;
}

const char *pack_mixed_meshes(const DeviceScene &dscene,
                              const std::span<const KernelLightTreeNode> nodes,
                              const std::span<const KernelLightTreeEmitter> emitters,
                              const std::span<const KernelLightDistribution> distribution_triangles,
                              uint32_t *header,
                              const std::span<uint32_t> lighting)
{
  const auto objects = host_data(dscene.objects);
  const auto object_flags = host_data(dscene.object_flag);
  const auto object_to_tree = host_data(dscene.light_to_tree);
  const auto object_prims = host_data(dscene.object_prim_offset);
  const auto object_lookup = host_data(dscene.object_lookup_offset);
  const auto triangle_to_tree = host_data(dscene.triangle_to_tree);
  const bool use_tree = distribution_triangles.empty();
  if (use_tree &&
      (object_to_tree.size() != objects.size() || object_prims.size() != objects.size() ||
       object_lookup.size() != objects.size() || triangle_to_tree.empty()))
  {
    return "WebGPU Cycles needs synchronized Light Tree mesh lookups";
  }

  const auto object_records = lighting.subspan(header[CYCLES_WEBGPU_TREE_OBJECTS],
                                                objects.size() * CYCLES_WEBGPU_TREE_OBJECT_WORDS);
  const auto triangle_indices = lighting.subspan(header[CYCLES_WEBGPU_TREE_TRIANGLE_INDICES],
                                                  header[CYCLES_WEBGPU_TREE_TRIANGLE_INDEX_COUNT]);
  const auto lookup_records = lighting.subspan(header[CYCLES_WEBGPU_TREE_TRIANGLE_TO_TREE],
                                                header[CYCLES_WEBGPU_TREE_TRIANGLE_TO_TREE_COUNT]);
  std::ranges::fill(triangle_indices, CYCLES_WEBGPU_TREE_NO_LAMP);
  std::ranges::fill(lookup_records, CYCLES_WEBGPU_TREE_NO_LAMP);
  for (size_t i = 0; i < objects.size(); ++i) {
    object_records[i * CYCLES_WEBGPU_TREE_OBJECT_WORDS + CYCLES_WEBGPU_TREE_OBJECT_TO_TREE] =
        CYCLES_WEBGPU_TREE_NO_LAMP;
  }

  const auto pack_object = [&](const size_t object_index, const uint32_t emitter_index) {
    if (object_index >= objects.size()) {
      return false;
    }
    const KernelObject &object = objects[object_index];
    uint32_t *record = object_records.data() + object_index * CYCLES_WEBGPU_TREE_OBJECT_WORDS;
    if (!use_tree && record[CYCLES_WEBGPU_TREE_OBJECT_TO_TREE] != CYCLES_WEBGPU_TREE_NO_LAMP) {
      return true;
    }
    record[CYCLES_WEBGPU_TREE_OBJECT_FLAGS] = object_flags[object_index];
    record[CYCLES_WEBGPU_TREE_OBJECT_TO_TREE] = emitter_index;
    record[CYCLES_WEBGPU_TREE_OBJECT_PRIM_OFFSET] = use_tree ? object_prims[object_index] : 0;
    record[CYCLES_WEBGPU_TREE_OBJECT_LOOKUP_OFFSET] = use_tree ? object_lookup[object_index] : 0;
    record[CYCLES_WEBGPU_TREE_OBJECT_LIGHTGROUP] = uint32_t(object.lightgroup);
    const auto write_transform = [](const Transform &transform, uint32_t *data) {
      for (const float4 row : {transform.x, transform.y, transform.z}) {
        if (!finite_rgb(make_float3(row)) || !std::isfinite(row.w)) {
          return false;
        }
        write_rgb(data, make_float3(row));
        data[3] = float_word(row.w);
        data += 4;
      }
      return true;
    };
    return write_transform(object.tfm, record + CYCLES_WEBGPU_TREE_OBJECT_TRANSFORM) &&
           write_transform(object.itfm, record + CYCLES_WEBGPU_TREE_OBJECT_INVERSE_TRANSFORM);
  };

  if (use_tree) {
    /* Blender only initializes lookups for emissive objects and faces. */
    for (size_t i = 0; i < emitters.size(); ++i) {
      const KernelLightTreeEmitter &emitter = emitters[i];
      if (emitter.light.id < 0 || emitter.object_id != OBJECT_NONE) {
        continue;
      }
      const size_t object_index = size_t(emitter.mesh.object_id);
      if (object_index >= objects.size() || size_t(emitter.mesh.node_id) >= nodes.size() ||
          object_to_tree[object_index] != i ||
          object_lookup[object_index] >= triangle_to_tree.size() || !pack_object(object_index, uint32_t(i)))
      {
        return "WebGPU Cycles needs synchronized Light Tree mesh emitters and transforms";
      }
    }
  }
  else {
    for (const auto &entry : distribution_triangles) {
      if (!pack_object(size_t(entry.object_id), 0)) {
        return "WebGPU Cycles needs synchronized light distribution mesh objects and transforms";
      }
    }
  }

  const auto indices = host_data(dscene.tri_vindex);
  const auto vertices = host_data(dscene.tri_verts);
  const auto triangle_shaders = host_data(dscene.tri_shader);
  const auto shaders = host_data(dscene.shaders);
  uint32_t *shader_records = lighting.data() + header[CYCLES_WEBGPU_TREE_SHADERS];
  for (size_t i = 0; i < shaders.size(); ++i) {
    shader_records[i * CYCLES_WEBGPU_TREE_SHADER_WORDS + CYCLES_WEBGPU_TREE_SHADER_FLAGS] =
        uint32_t(shaders[i].flags);
  }

  uint32_t compact_index = 0;
  const auto pack_triangle = [&](const size_t object_index,
                                 const size_t prim,
                                 const int visibility,
                                 const size_t i) -> const char * {
    if (object_index >= objects.size() || prim >= indices.size() ||
        object_records[object_index * CYCLES_WEBGPU_TREE_OBJECT_WORDS +
                       CYCLES_WEBGPU_TREE_OBJECT_TO_TREE] == CYCLES_WEBGPU_TREE_NO_LAMP)
    {
      return "WebGPU Cycles needs synchronized triangle light emitters";
    }
    if (use_tree) {
      if (triangle_indices[prim] != CYCLES_WEBGPU_TREE_NO_LAMP || prim < object_prims[object_index])
      {
        return "WebGPU Cycles needs synchronized Light Tree triangle emitters";
      }
      const size_t lookup = prim - object_prims[object_index] + object_lookup[object_index];
      if (lookup >= triangle_to_tree.size() || triangle_to_tree[lookup] != i) {
        return "WebGPU Cycles needs the stock Light Tree triangle lookup";
      }
      lookup_records[lookup] = uint32_t(i);
    }
    if (visibility & (SHADER_EXCLUDE_DIFFUSE | SHADER_EXCLUDE_GLOSSY | SHADER_EXCLUDE_TRANSMIT)) {
      return "WebGPU Cycles needs diffuse, glossy and transmission lighting on emissive meshes";
    }
    if (triangle_indices[prim] != CYCLES_WEBGPU_TREE_NO_LAMP) {
      /* The flat distribution has one entry per instance. Shared mesh vertices
       * are packed once, and stock sampling applies the selected transform. */
      return nullptr;
    }
    const uint32_t shader = triangle_shaders[prim];
    const uint32_t shader_id = shader & SHADER_MASK;
    if (shader_id >= shaders.size()) {
      return "WebGPU Cycles needs a shader for each Light Tree triangle emitter";
    }
    if (shaders[shader_id].flags & SD_HAS_CONSTANT_EMISSION) {
      float3 emission;
      if (!constant_emission(dscene, shader, emission)) {
        return "WebGPU Cycles needs finite constant emission on mesh lights";
      }
      write_rgb(shader_records + size_t(shader_id) * CYCLES_WEBGPU_TREE_SHADER_WORDS +
                    CYCLES_WEBGPU_TREE_SHADER_EMISSION,
                emission);
    }
    else {
      ++header[CYCLES_WEBGPU_TREE_TEXTURED_EMITTERS];
    }

    triangle_indices[prim] = compact_index;
    uint32_t *record = lighting.data() + header[CYCLES_WEBGPU_TREE_TRIANGLES] +
                       size_t(compact_index) * CYCLES_WEBGPU_TREE_TRIANGLE_WORDS;
    ++compact_index;
    record[CYCLES_WEBGPU_TREE_TRIANGLE_SHADER] = shader;
    uint32_t *positions = record + CYCLES_WEBGPU_TREE_TRIANGLE_VERTICES;
    const size_t position_offset = size_t(objects[object_index].position_offset);
    const packed_uint3 index = indices[prim];
    for (const uint vertex : {index.x, index.y, index.z}) {
      if (position_offset >= vertices.size() || vertex >= vertices.size() - position_offset) {
        return "WebGPU Cycles needs synchronized Light Tree vertex positions";
      }
      const float3 position = make_float3(vertices[position_offset + vertex]);
      if (!finite_rgb(position)) {
        return "WebGPU Cycles needs finite Light Tree vertex positions";
      }
      write_rgb(positions, position);
      positions += 3;
    }
    return nullptr;
  };

  if (use_tree) {
    for (size_t i = 0; i < emitters.size(); ++i) {
      const auto &emitter = emitters[i];
      if (emitter.light.id >= 0 && emitter.object_id != OBJECT_NONE) {
        if (const char *error = pack_triangle(size_t(emitter.object_id),
                                              size_t(emitter.triangle.id),
                                              emitter.visibility_flag,
                                              i))
        {
          return error;
        }
      }
    }
  }
  else {
    for (const auto &entry : distribution_triangles) {
      if (const char *error = pack_triangle(size_t(entry.object_id),
                                            size_t(entry.prim),
                                            entry.visibility_flag,
                                            0))
      {
        return error;
      }
    }
  }
  return nullptr;
}

const char *encode_stock_mixed_lighting(const Scene &scene,
                                        const WorldEnvironment &environment,
                                        const bool volume_scattering,
                                        const size_t remaining_words,
                                        std::vector<uint32_t> &lighting)
{
  const DeviceScene &dscene = scene.dscene;
  const KernelIntegrator &integrator = dscene.data.integrator;
  const KernelBackground &world = dscene.data.background;
  const auto lights = host_data(dscene.lights);
  const auto nodes = host_data(dscene.light_tree_nodes);
  const auto emitters = host_data(dscene.light_tree_emitters);
  const bool use_tree = integrator.use_light_tree;
  const auto distribution = use_tree ? std::span<const KernelLightDistribution>{} :
                                       host_data(dscene.light_distribution);
  size_t triangle_count;
  if (use_tree) {
    triangle_count = std::ranges::count_if(emitters, [](const KernelLightTreeEmitter &entry) {
      return entry.light.id >= 0 && entry.object_id != OBJECT_NONE;
    });
    if (nodes.empty() || emitters.empty() || emitters.size() < lights.size() ||
        (triangle_count == 0 && lights.size() != emitters.size()))
    {
      return "WebGPU Cycles needs synchronized Light Tree emitters and nodes";
    }
  }
  else {
    if (distribution.size() <= lights.size() || !nodes.empty() || !emitters.empty() ||
        distribution.size() != uint64_t(integrator.num_distribution) + 1 ||
        distribution.front().totarea != 0.0f || distribution.back().totarea != 1.0f ||
        !std::isfinite(integrator.distribution_pdf_lights) ||
        integrator.distribution_pdf_lights < 0.0f ||
        !std::isfinite(integrator.distribution_pdf_triangles) ||
        integrator.distribution_pdf_triangles < 0.0f)
    {
      return "WebGPU Cycles needs a synchronized light distribution";
    }
    triangle_count = distribution.size() - lights.size() - 1;
  }
  if (integrator.num_background_lights > 1 ||
      integrator.direct_light_sampling_type != DIRECT_LIGHT_SAMPLING_MIS ||
      (dscene.data.kernel_features & KERNEL_FEATURE_MNEE))
  {
    return "WebGPU Cycles needs supported lights with MIS enabled";
  }

  const bool sampled_world = integrator.num_background_lights != 0;
  const auto marginal = sampled_world ? host_data(dscene.light_background_marginal_cdf) :
                                        std::span<const float2>{};
  const auto conditional = sampled_world ? host_data(dscene.light_background_conditional_cdf) :
                                           std::span<const float2>{};
  if (sampled_world &&
      (!world.use_mis || world.map_weight != 1.0f || world.portal_weight != 0.0f ||
       world.map_res_x <= 0 || world.map_res_y <= 0 ||
       marginal.size() != uint64_t(world.map_res_y) + 1 ||
       conditional.size() != (uint64_t(world.map_res_x) + 1) * world.map_res_y ||
       !std::isfinite(marginal.back().x) || marginal.back().x <= 0.0f))
  {
    return "WebGPU Cycles needs a synchronized world importance map without portals";
  }

  if (sampled_world && world.sun_weight != 0.0f &&
      (environment.sky_parameters.empty() || !std::isfinite(world.sun_weight) ||
       world.sun_weight < 0.0f || !finite_rgb(make_float3(world.sun)) ||
       !std::isfinite(world.sun.w) || world.sun.w <= 0.0f))
  {
    return "WebGPU Cycles needs a synchronized sky sun for world importance sampling";
  }

  const auto lut = light_sampling_table(dscene);
  const uint64_t lamp_words = uint64_t(lights.size()) * CYCLES_WEBGPU_TREE_LAMP_WORDS;
  const bool varying_lamps = std::ranges::any_of(lights, [&](const KernelLight &light) {
    return lamp_uses_varying_emission(dscene, light);
  });
  if (varying_lamps && volume_scattering && integrator.light_inv_rr_threshold > 0.0f) {
    return "WebGPU Cycles Light Falloff with volume scattering needs the light threshold disabled";
  }
  const uint64_t falloff_words = varying_lamps ?
                                     uint64_t(lights.size()) * CYCLES_WEBGPU_LAMP_FALLOFF_WORDS :
                                     0;
  const uint64_t node_words = uint64_t(nodes.size()) * CYCLES_WEBGPU_TREE_NODE_WORDS;
  const uint64_t emitter_words = uint64_t(emitters.size()) * CYCLES_WEBGPU_TREE_EMITTER_WORDS;
  const uint64_t object_count = triangle_count ? dscene.objects.size() : 0;
  const uint64_t triangle_index_count = triangle_count ? dscene.tri_shader.size() : 0;
  const uint64_t shader_count = triangle_count ? dscene.shaders.size() : 0;
  const uint64_t triangle_lookup_count = use_tree && triangle_count ?
                                             dscene.triangle_to_tree.size() : 0;
  const uint64_t object_words = object_count * CYCLES_WEBGPU_TREE_OBJECT_WORDS;
  const uint64_t triangle_words = uint64_t(triangle_count) * CYCLES_WEBGPU_TREE_TRIANGLE_WORDS;
  const uint64_t shader_words = shader_count * CYCLES_WEBGPU_TREE_SHADER_WORDS;
  const uint64_t world_words = environment.word_count();
  const uint64_t descriptor_words = use_tree ? CYCLES_WEBGPU_TREE_DESCRIPTOR_WORDS :
                                               CYCLES_WEBGPU_DISTRIBUTION_DESCRIPTOR_WORDS;
  const uint64_t distribution_words = uint64_t(distribution.size()) *
                                      CYCLES_WEBGPU_POINT_DISTRIBUTION_WORDS;
  const uint64_t total_words = descriptor_words + lamp_words + falloff_words + node_words +
                               distribution_words + emitter_words + object_words + triangle_index_count +
                               triangle_words + shader_words + triangle_lookup_count +
                               2 * uint64_t(marginal.size()) + 2 * uint64_t(conditional.size()) +
                               world_words + CYCLES_WEBGPU_LIGHT_SAMPLING_WORDS + lut.size();
  if (total_words > remaining_words) {
    return "WebGPU Cycles light data and world map exceed the scene upload budget";
  }

  const size_t descriptor = lighting.size();
  const size_t lamp_offset = descriptor + size_t(descriptor_words);
  const size_t node_offset = lamp_offset + size_t(lamp_words);
  const size_t emitter_offset = node_offset + size_t(node_words);
  const size_t object_offset = emitter_offset + size_t(emitter_words);
  const size_t triangle_index_offset = object_offset + size_t(object_words);
  const size_t triangle_offset = triangle_index_offset + size_t(triangle_index_count);
  const size_t shader_offset = triangle_offset + size_t(triangle_words);
  const size_t triangle_lookup_offset = shader_offset + size_t(shader_words);
  const size_t distribution_offset = triangle_lookup_offset + size_t(triangle_lookup_count);
  const size_t marginal_offset = distribution_offset + size_t(distribution_words);
  const size_t conditional_offset = marginal_offset + 2 * marginal.size();
  const size_t world_offset = conditional_offset + 2 * conditional.size();
  const size_t sampling_offset = world_offset + size_t(world_words);

  lighting.reserve(descriptor + size_t(total_words));
  lighting.resize(marginal_offset);
  lighting[0] = CYCLES_WEBGPU_LIGHT_TREE_MARKER;
  lighting[CYCLES_WEBGPU_LIGHT_TREE_OFFSET] = uint32_t(descriptor);
  write_rgb(lighting.data() + 8, environment.background);

  uint32_t *header = lighting.data() + descriptor;
  header[CYCLES_WEBGPU_TREE_LAMP_COUNT] = uint32_t(lights.size());
  header[CYCLES_WEBGPU_TREE_LAMPS] = uint32_t(lamp_offset);
  header[CYCLES_WEBGPU_TREE_NODE_COUNT] = uint32_t(nodes.size());
  header[CYCLES_WEBGPU_TREE_NODES] = uint32_t(node_offset);
  header[CYCLES_WEBGPU_TREE_EMITTER_COUNT] = uint32_t(emitters.size());
  header[CYCLES_WEBGPU_TREE_EMITTERS] = uint32_t(emitter_offset);
  header[CYCLES_WEBGPU_TREE_OBJECT_COUNT] = uint32_t(object_count);
  header[CYCLES_WEBGPU_TREE_OBJECTS] = uint32_t(object_offset);
  header[CYCLES_WEBGPU_TREE_TRIANGLE_INDEX_COUNT] = uint32_t(triangle_index_count);
  header[CYCLES_WEBGPU_TREE_TRIANGLE_INDICES] = uint32_t(triangle_index_offset);
  header[CYCLES_WEBGPU_TREE_TRIANGLE_COUNT] = uint32_t(triangle_count);
  header[CYCLES_WEBGPU_TREE_TEXTURED_EMITTERS] = 0;
  header[CYCLES_WEBGPU_TREE_TRIANGLES] = uint32_t(triangle_offset);
  header[CYCLES_WEBGPU_TREE_SHADER_COUNT] = uint32_t(shader_count);
  header[CYCLES_WEBGPU_TREE_SHADERS] = uint32_t(shader_offset);
  header[CYCLES_WEBGPU_TREE_TRIANGLE_TO_TREE_COUNT] = uint32_t(triangle_lookup_count);
  header[CYCLES_WEBGPU_TREE_TRIANGLE_TO_TREE] = uint32_t(triangle_lookup_offset);
  header[CYCLES_WEBGPU_TREE_FIRST_FORWARD_LAMP] = CYCLES_WEBGPU_TREE_NO_LAMP;
  header[CYCLES_WEBGPU_TREE_TRANSPARENT_MAX_BOUNCE] = uint32_t(integrator.transparent_max_bounce);
  header[CYCLES_WEBGPU_TREE_WORLD_LAMP] = CYCLES_WEBGPU_TREE_NO_LAMP;
  header[CYCLES_WEBGPU_TREE_WORLD_EMITTER] = CYCLES_WEBGPU_TREE_NO_LAMP;
  header[CYCLES_WEBGPU_TREE_WORLD_MAP_WIDTH] = sampled_world ? uint32_t(world.map_res_x) : 0;
  header[CYCLES_WEBGPU_TREE_WORLD_MAP_HEIGHT] = sampled_world ? uint32_t(world.map_res_y) : 0;
  header[CYCLES_WEBGPU_TREE_WORLD_MARGINAL_CDF] = uint32_t(marginal_offset);
  header[CYCLES_WEBGPU_TREE_WORLD_CONDITIONAL_CDF] = uint32_t(conditional_offset);
  header[CYCLES_WEBGPU_TREE_WORLD_WIDTH] = environment.image.width;
  header[CYCLES_WEBGPU_TREE_WORLD_HEIGHT] = environment.image.height;
  header[CYCLES_WEBGPU_TREE_WORLD_INTERPOLATION] = environment.interpolation();
  header[CYCLES_WEBGPU_TREE_WORLD_SHADER] = uint32_t(world.surface_shader);
  header[CYCLES_WEBGPU_TREE_WORLD_PIXELS] = uint32_t(world_offset);
  header[CYCLES_WEBGPU_TREE_SAMPLING] = uint32_t(sampling_offset);

  std::vector<uint32_t> falloff_records(size_t(falloff_words), 0);
  if (const char *error = pack_mixed_lamps(
          scene,
          emitters,
          distribution,
          std::span(lighting).subspan(lamp_offset, size_t(lamp_words)),
          falloff_records,
          header[CYCLES_WEBGPU_TREE_FIRST_FORWARD_LAMP],
          header[CYCLES_WEBGPU_TREE_WORLD_LAMP],
          header[CYCLES_WEBGPU_TREE_WORLD_EMITTER]))
  {
    return error;
  }
  if (sampled_world != (header[CYCLES_WEBGPU_TREE_WORLD_LAMP] != CYCLES_WEBGPU_TREE_NO_LAMP) ||
      (use_tree && !pack_light_tree_nodes(nodes,
                                          emitters.size(),
                                          std::span(lighting).subspan(node_offset, size_t(node_words)))))
  {
    return "WebGPU Cycles needs synchronized Light Tree nodes";
  }
  for (size_t i = 0; i < emitters.size(); ++i) {
    const KernelLightTreeEmitter &emitter = emitters[i];
    uint32_t *record = lighting.data() + emitter_offset + i * CYCLES_WEBGPU_TREE_EMITTER_WORDS;
    record[CYCLES_WEBGPU_TREE_EMITTER_THETA_O] = float_word(emitter.theta_o);
    record[CYCLES_WEBGPU_TREE_EMITTER_THETA_E] = float_word(emitter.theta_e);
    record[CYCLES_WEBGPU_TREE_EMITTER_ENERGY] = float_word(emitter.energy);
    record[CYCLES_WEBGPU_TREE_EMITTER_LIGHT] = uint32_t(emitter.light.id);
    if (emitter.light.id >= 0) {
      record[CYCLES_WEBGPU_TREE_EMITTER_DETAIL] = emitter.object_id == OBJECT_NONE ?
                                                      uint32_t(emitter.mesh.node_id) :
                                                      uint32_t(emitter.triangle.emission_sampling);
    }
    record[CYCLES_WEBGPU_TREE_EMITTER_OBJECT] = uint32_t(emitter.object_id);
    record[CYCLES_WEBGPU_TREE_EMITTER_VISIBILITY] = uint32_t(emitter.visibility_flag);
    record[CYCLES_WEBGPU_TREE_EMITTER_BIT_TRAIL] = uint32_t(emitter.bit_trail);
  }
  if (!use_tree) {
    header[CYCLES_WEBGPU_DISTRIBUTION_OFFSET] = uint32_t(distribution_offset);
    header[CYCLES_WEBGPU_DISTRIBUTION_COUNT] = uint32_t(integrator.num_distribution);
    header[CYCLES_WEBGPU_DISTRIBUTION_LAMP_PDF] = float_word(integrator.distribution_pdf_lights);
    header[CYCLES_WEBGPU_DISTRIBUTION_TRIANGLE_PDF] = float_word(integrator.distribution_pdf_triangles);
    float previous = 0.0f;
    for (size_t i = 0; i < distribution.size(); ++i) {
      const auto &entry = distribution[i];
      if (!std::isfinite(entry.totarea) || entry.totarea < previous || entry.totarea > 1.0f ||
          (i < triangle_count && entry.prim < 0))
      {
        return "WebGPU Cycles needs finite, ordered light distribution entries";
      }
      previous = entry.totarea;
      uint32_t *words = lighting.data() + distribution_offset +
                        i * CYCLES_WEBGPU_POINT_DISTRIBUTION_WORDS;
      words[0] = float_word(entry.totarea);
      words[1] = uint32_t(entry.prim);
      words[2] = uint32_t(entry.visibility_flag);
      words[3] = uint32_t(entry.object_id);
    }
  }
  if (triangle_count) {
    const auto distribution_triangles = use_tree ? std::span<const KernelLightDistribution>{} :
                                                   distribution.first(triangle_count);
    if (const char *error = pack_mixed_meshes(
            dscene, nodes, emitters, distribution_triangles, header, lighting))
 {
      return error;
    }
  }

  for (const auto cdf : {marginal, conditional}) {
    for (const float2 entry : cdf) {
      if (!std::isfinite(entry.x) || !std::isfinite(entry.y)) {
        return "WebGPU Cycles needs a finite world importance map";
      }
      lighting.push_back(float_word(entry.x));
      lighting.push_back(float_word(entry.y));
    }
  }
  if (const char *error = append_world_environment(dscene, environment, lighting)) {
    return error;
  }
  append_light_sampling(integrator, lut, lighting);
  if (varying_lamps) {
    lighting[CYCLES_WEBGPU_TREE_LAMP_FALLOFF_OFFSET] = uint32_t(lighting.size());
    lighting.insert(lighting.end(), falloff_records.begin(), falloff_records.end());
  }
  return nullptr;
}

const char *ggx_transmission_incompatibility(const DeviceScene &scene,
                                             const bool mixed_lighting,
                                             const bool transparent_shadows,
                                             const std::span<const uint> volume_shaders)
{
  const KernelIntegrator &integrator = scene.data.integrator;
  if (integrator.caustics_refractive != integrator.caustics_reflective) {
    return "WebGPU Cycles glass needs matching reflective and refractive caustics settings";
  }
  if (integrator.filter_closures & FILTER_CLOSURE_TRANSMISSION) {
    return "WebGPU Cycles glass needs transmission closures enabled";
  }
  if (scene.data.kernel_features & KERNEL_FEATURE_MNEE) {
    return "WebGPU Cycles glass needs shadow caustics turned off";
  }
  if (std::ranges::any_of(host_data(scene.lights), [](const KernelLight &light) {
        return (light.shader_id & SHADER_EXCLUDE_TRANSMIT) != 0;
      }))
  {
    return "WebGPU Cycles glass needs transmission lighting enabled on the lights";
  }
  const int world_shader = scene.data.background.surface_shader;
  const bool has_area_light = scene.lights.size() == 1 && scene.lights.data()[0].type == LIGHT_AREA;
  if (!mixed_lighting && !has_area_light && !(world_shader & SHADER_EXCLUDE_TRANSMIT))
  {
    float3 emission;
    if (!constant_emission(scene, world_shader, emission) || emission.x != 0.0f ||
        emission.y != 0.0f || emission.z != 0.0f)
    {
      return "WebGPU Cycles glass needs a black world or transmission world ray visibility "
             "disabled";
    }
  }
  if (!transparent_shadows) {
    for (const uint packed_shader : host_data(scene.tri_shader)) {
      const uint shader = packed_shader & SHADER_MASK;
      /* Cycles sets the transparent-shadow flag for all volume materials.
       * The volume shadow stages handle their boundaries and extinction. */
      if ((scene.shaders.data()[shader].flags & SD_HAS_TRANSPARENT_SHADOW) &&
          std::ranges::find(volume_shaders, shader) == volume_shaders.end())
      {
        return "WebGPU Cycles glass needs supported shadow casters";
      }
    }
  }
  return nullptr;
}

const char *encode_ggx_tables(const DeviceScene &scene,
                              const bool mixed_lighting,
                              const bool has_transmission,
                              const bool transparent_shadows,
                              const std::span<const uint> volume_shaders,
                              std::vector<uint32_t> &lighting)
{
  const KernelIntegrator &integrator = scene.data.integrator;
  if (integrator.filter_closures & FILTER_CLOSURE_GLOSSY) {
    return "WebGPU Cycles GGX needs glossy closures enabled";
  }
  if (std::ranges::any_of(host_data(scene.lights), [](const KernelLight &light) {
        return (light.shader_id & SHADER_EXCLUDE_GLOSSY) != 0;
      }))
  {
    return "WebGPU Cycles GGX needs glossy light contributions";
  }
  const int world_shader = scene.data.background.surface_shader;
  if (!mixed_lighting && scene.lights.data()[0].type != LIGHT_AREA &&
      !(world_shader & SHADER_EXCLUDE_GLOSSY))
  {
    float3 emission;
    if (!constant_emission(scene, world_shader, emission) || emission.x != 0.0f ||
        emission.y != 0.0f || emission.z != 0.0f)
    {
      return "WebGPU Cycles GGX needs a black world or glossy world ray visibility disabled";
    }
  }
  for (const KernelObject &object : host_data(scene.objects)) {
    if (object.primitive_type != PRIMITIVE_LAMP && object.shadow_terminator_shading_offset > 1.0f)
    {
      return "WebGPU Cycles GGX needs the default object shadow terminator shading offset";
    }
  }
  if (has_transmission) {
    if (const char *error = ggx_transmission_incompatibility(
            scene, mixed_lighting, transparent_shadows, volume_shaders))
    {
      return error;
    }
  }

  const auto &tables = scene.data.tables;
  const auto table_words = host_data(scene.lookup_table);
  const std::pair<int, size_t> packed_tables[] = {
      {tables.ggx_gen_schlick_ior_s, CYCLES_WEBGPU_GGX_TABLE_WORDS},
      {tables.ggx_gen_schlick_s, CYCLES_WEBGPU_GGX_TABLE_WORDS},
      {tables.ggx_E, CYCLES_WEBGPU_GGX_ENERGY_WORDS},
      {tables.ggx_Eavg, CYCLES_WEBGPU_GGX_AVERAGE_ENERGY_WORDS},
      {tables.ggx_glass_E, CYCLES_WEBGPU_GGX_GLASS_ENERGY_WORDS},
      {tables.ggx_glass_Eavg, CYCLES_WEBGPU_GGX_GLASS_AVERAGE_ENERGY_WORDS},
      {tables.ggx_glass_inv_E, CYCLES_WEBGPU_GGX_GLASS_ENERGY_WORDS},
      {tables.ggx_glass_inv_Eavg, CYCLES_WEBGPU_GGX_GLASS_AVERAGE_ENERGY_WORDS},
  };
  const auto offsets = std::span(packed_tables);
  for (const auto &[offset, table_size] : has_transmission ? offsets : offsets.first(4)) {
    if (offset < 0 || size_t(offset) > table_words.size() ||
        table_size > table_words.size() - size_t(offset))
    {
      return "WebGPU Cycles needs synchronized GGX lookup tables";
    }
    for (const float value : table_words.subspan(size_t(offset), table_size)) {
      lighting.push_back(float_word(value));
    }
  }
  return nullptr;
}

using VolumeWords = std::array<uint32_t, CYCLES_WEBGPU_VOLUME_SCATTER_WORDS>;

struct VolumeMedium {
  VolumeWords words{};
  std::vector<uint32_t> grids;
  BoundBox bounds = BoundBox::empty;
  uint phase_slots = 0;
  bool absorption = false;
};

bool volume_phase_capacity(const std::span<const VolumeMedium> media)
{
  uint total_slots = 0;
  for (const VolumeMedium &medium : media) {
    total_slots += medium.phase_slots;
  }
  if (total_slots <= MAX_VOLUME_CLOSURE) {
    return true;
  }

  /* Any common overlap of bounded media contains the point obtained by taking
   * the largest lower bound on each axis. Check those candidate points to
   * bound the number of stock phase closures evaluated at once. */
  for (const VolumeMedium &x : media) {
    for (const VolumeMedium &y : media) {
      for (const VolumeMedium &z : media) {
        const float3 point = make_float3(x.bounds.min.x, y.bounds.min.y, z.bounds.min.z);
        uint slots = 0;
        for (const VolumeMedium &medium : media) {
          const BoundBox &bounds = medium.bounds;
          if (point.x >= bounds.min.x && point.x <= bounds.max.x &&
              point.y >= bounds.min.y && point.y <= bounds.max.y &&
              point.z >= bounds.min.z && point.z <= bounds.max.z)
          {
            slots += medium.phase_slots;
          }
        }
        if (slots > MAX_VOLUME_CLOSURE) {
          return false;
        }
      }
    }
  }
  return true;
}

bool needs_volume_intervals(const std::span<const VolumeMedium> media)
{
  return media.size() != 1 || !media.front().absorption ||
         !(media.front().words[CYCLES_WEBGPU_VOLUME_SHADER_FLAGS] & SD_HAS_ONLY_VOLUME);
}

const char *encode_volumes(const Scene &scene,
                           const size_t grid_budget,
                           std::vector<uint> &shaders,
                           std::vector<VolumeMedium> &media)
{
  const DeviceScene &dscene = scene.dscene;
  if (scene.params.use_bvh_spatial_split ||
      dscene.light_tree_emitters.size() > dscene.lights.size() ||
      dscene.light_distribution.size() > dscene.lights.size() + 1)
  {
    return "WebGPU Cycles volumes need a BVH without spatial splits and analytic lighting";
  }
  size_t total_grid_words = 0;
  for (const Object *object : scene.objects) {
    const uint index = uint(object->get_device_index());
    if (index >= dscene.objects.size()) {
      return "WebGPU Cycles needs a synchronized volume object";
    }
    if (!(dscene.object_flag.data()[index] & SD_OBJECT_HAS_VOLUME)) {
      continue;
    }
    const Geometry *geometry = object->get_geometry();
    if (media.size() >= CYCLES_WEBGPU_MAX_VOLUME_MEDIA) {
      return "WebGPU Cycles volume count exceeds its active-stack capacity";
    }
    if (!geometry || (!geometry->is_mesh() && !geometry->is_volume()) ||
        geometry->get_used_shaders().size() != 1 ||
        !object->bounds.valid() ||
        (object->visibility_for_tracing() & PATH_RAY_VISIBILITY_OBJECT_ALL) !=
            PATH_RAY_VISIBILITY_OBJECT_ALL)
    {
      return "WebGPU Cycles needs closed volume bounds with one material and full ray visibility";
    }
    const uint shader = static_cast<const Shader *>(geometry->get_used_shaders()[0])->id;
    if (shader >= dscene.shaders.size()) {
      return "WebGPU Cycles needs a synchronized volume shader";
    }
    const uint flags = uint(dscene.shaders.data()[shader].flags);
    if (!(flags & SD_HAS_ONLY_VOLUME) && (!geometry->is_mesh() || !(flags & SD_HAS_VOLUME))) {
      return "WebGPU Cycles mixed volume boundaries need a mesh surface and volume closure";
    }

    /* The independent analytic-light forward pass must never finish inside
     * a medium. Compare the world-space bounds of the synchronized emitters. */
    BoundBox medium_bounds = object->bounds;
    for (const KernelLight &light : host_data(dscene.lights)) {
      if (light.type == LIGHT_BACKGROUND ||
          (light.type == LIGHT_SUN && !dscene.data.integrator.use_light_tree))
      {
        continue;
      }
      if (light.type != LIGHT_POINT && light.type != LIGHT_AREA) {
        return "WebGPU Cycles volumes need supported analytic or world lighting";
      }
      const float3 center = make_float3(light.co);
      const float3 extent =
          light.type == LIGHT_AREA ?
              0.5f * (fabs(make_float3(light.area.axis_u)) * light.area.len_u +
                      fabs(make_float3(light.area.axis_v)) * light.area.len_v) :
              make_float3(light.spot.radius);
      const BoundBox emitter(center - extent, center + extent);
      if (!emitter.valid() || medium_bounds.intersects(emitter)) {
        return "WebGPU Cycles needs analytic emitters outside the volume bounds";
      }
    }

    VolumeMedium medium;
    bool has_voxels = false;
    const float density = dscene.objects.data()[index].mesh_volume.volume_density;
    const bool can_read_grids = (dscene.object_flag.data()[index] &
                                 SD_OBJECT_HAS_VOLUME_ATTRIBUTES) != 0;
    if (!std::isfinite(density) || density < 0.0f ||
        !webgpu_svm_literal_volume(dscene,
                                  shader,
                                  int(index),
                                  medium.words.data(),
                                  medium.absorption,
                                  has_voxels,
                                  medium.phase_slots,
                                  can_read_grids) ||
        ((flags & SD_HETEROGENEOUS_VOLUME) && !has_voxels))
    {
      return "WebGPU Cycles needs supported volume closures and grid attribute inputs";
    }
    medium.bounds = medium_bounds;
    medium.words[CYCLES_WEBGPU_ABSORPTION_OBJECT] = index;
    medium.words[CYCLES_WEBGPU_ABSORPTION_OBJECT_DENSITY] = float_word(density);
    medium.words[CYCLES_WEBGPU_VOLUME_SHADER_FLAGS] = flags;
    if (has_voxels) {
      if (const char *error = webgpu_pack_volume_grids(
              scene, *object, medium.words, grid_budget - total_grid_words, medium.grids))
      {
        return error;
      }
    }
    total_grid_words += medium.grids.size();
    shaders.push_back(shader);
    media.push_back(std::move(medium));
  }

  if (media.empty()) {
    return "WebGPU Cycles could not find a synchronized volume";
  }
  if (!volume_phase_capacity(media)) {
    return "WebGPU Cycles overlapping volumes exceed the stock phase closure capacity";
  }
  if (needs_volume_intervals(media)) {
    const auto &integrator = dscene.data.integrator;
    const bool has_selection = integrator.use_light_tree ? dscene.light_tree_nodes.size() != 0 :
                                                            dscene.light_distribution.size() > 1;
    if (!has_selection || integrator.max_volume_bounce < 0 ||
        integrator.max_volume_bounce > 65535)
    {
      return "WebGPU Cycles volume scattering needs a light distribution and valid bounce settings";
    }
    for (VolumeMedium &medium : media) {
      medium.words[CYCLES_WEBGPU_VOLUME_DIRECT_CLAMP] = float_word(integrator.sample_clamp_direct);
      medium.words[CYCLES_WEBGPU_VOLUME_INDIRECT_CLAMP] = float_word(integrator.sample_clamp_indirect);
      medium.words[CYCLES_WEBGPU_VOLUME_MAX_BOUNCES] = uint32_t(integrator.max_volume_bounce);
    }
  }
  return nullptr;
}

}  // namespace

device_ptr cycles_webgpu_create_svm_image_atlas(Device &device, const std::span<const uint32_t> svm)
{
  /* Keep the packed image data for other kernels and devices without float filtering. Only
   * duplicate cubic/smart images, and bound the extra memory and upload independently. */
  constexpr uint32_t padding = 2;
  constexpr uint32_t max_dimension = 2048;
  constexpr size_t max_atlas_bytes = 64 * 1024 * 1024;
  const uint32_t limit = std::min(device_webgpu_filterable_texture_limit(device), max_dimension);
  if (limit <= 2 * padding || svm.size() < webgpu_svm_header_words) {
    return 0;
  }
  const size_t texture = webgpu_svm_header_words + size_t(svm[0]);
  if (texture > svm.size() || svm.size() - texture < webgpu_svm_texture_header_words ||
      svm[texture + 3] > webgpu_svm_attribute_slots)
  {
    return 0;
  }
  const size_t image_table = texture + webgpu_svm_texture_header_words +
                             size_t(svm[texture]) * webgpu_svm_triangle_uv_words +
                             size_t(svm[texture + 2]) * webgpu_svm_triangle_generated_words +
                             size_t(svm[texture + 3]) * svm[texture + 4] * webgpu_svm_attribute_words;
  const uint32_t slots = svm[texture + 1];
  if (image_table > svm.size() || slots > (svm.size() - image_table) / webgpu_svm_image_words) {
    return 0;
  }

  struct Image {
    uint32_t id, width, height, extension, start;
    bool alpha;
    uint32_t x = 0, y = 0;
  };
  std::vector<Image> images;
  uint32_t minimum_width = 0;
  size_t total_width = 0;
  for (uint32_t id = 0; id < std::min(slots, limit); ++id) {
    const size_t record = image_table + size_t(id) * webgpu_svm_image_words;
    const uint32_t width = svm[record];
    const uint32_t height = svm[record + 1];
    const uint32_t extension = svm[record + 2];
    const uint32_t interpolation = svm[record + 3];
    const uint32_t start = svm[record + 4];
    const bool alpha = (interpolation & webgpu_svm_image_alpha) != 0;
    if (!webgpu_svm_uses_cubic_filter(interpolation & ~webgpu_svm_image_alpha) ||
        width == 0 || height == 0 || width > limit - 2 * padding ||
        height >= limit - 2 * padding || start > svm.size() ||
        size_t(width) * height * (alpha ? 4 : 3) > svm.size() - start ||
        (extension != EXTENSION_REPEAT && extension != EXTENSION_MIRROR &&
         extension != EXTENSION_EXTEND && extension != EXTENSION_CLIP))
    {
      continue;
    }
    images.push_back({id, width, height, extension, start, alpha});
    minimum_width = std::max({minimum_width, width + 2 * padding, id + 1});
    total_width += width + 2 * padding;
  }
  if (images.empty()) {
    return 0;
  }
  const uint32_t atlas_width = std::max(minimum_width, uint32_t(std::min<size_t>(total_width, limit)));
  const uint32_t height_limit = uint32_t(std::min<size_t>(
      limit, max_atlas_bytes / (size_t(atlas_width) * 4 * sizeof(float))));
  /* Pack taller images first. The reserved top row maps original image IDs to their
   * origins. Zero entries keep using the packed buffer. */
  std::stable_sort(images.begin(), images.end(), [](const Image &a, const Image &b) {
    return a.height > b.height;
  });
  uint32_t left = 0, top = 1, shelf_height = 0, atlas_height = 1;
  for (Image &image : images) {
    const uint32_t width = image.width + 2 * padding;
    const uint32_t height = image.height + 2 * padding;
    if (left + width > atlas_width) {
      left = 0;
      top += shelf_height;
      shelf_height = 0;
    }
    if (top + height > height_limit) {
      continue;
    }
    image.x = left + padding;
    image.y = top + padding;
    left += width;
    shelf_height = std::max(shelf_height, height);
    atlas_height = std::max(atlas_height, top + shelf_height);
  }
  if (atlas_height == 1) {
    return 0;
  }
  std::vector<float> atlas;
  try {
    atlas.resize(size_t(atlas_width) * atlas_height * 4);
  }
  catch (const std::bad_alloc &) {
    return 0;
  }
  uint32_t uploaded = 0;
  const auto address = [](const int coord, const uint32_t size, const uint32_t extension) {
    return webgpu_svm_address_texel(size, extension, uint32_t(std::abs(coord)), coord < 0);
  };
  for (const Image &image : images) {
    if (!image.x) {
      continue;
    }
    ++uploaded;
    atlas[size_t(image.id) * 4] = float(image.x);
    atlas[size_t(image.id) * 4 + 1] = float(image.y);
    const size_t alpha_start = image.start + size_t(image.width) * image.height * 3;
    for (int y = -int(padding); y < int(image.height + padding); ++y) {
      const uint32_t row = address(y, image.height, image.extension);
      if (row >= image.height) {
        continue;
      }
      for (int x = -int(padding); x < int(image.width + padding); ++x) {
        const uint32_t col = address(x, image.width, image.extension);
        if (col >= image.width) {
          continue;
        }
        const size_t src = size_t(row) * image.width + col;
        const size_t dst = (size_t(int(image.y) + y) * atlas_width + int(image.x) + x) * 4;
        for (int channel = 0; channel < 3; ++channel) {
          atlas[dst + channel] = std::bit_cast<float>(svm[image.start + 3 * src + channel]);
        }
        atlas[dst + 3] = image.alpha ? std::bit_cast<float>(svm[alpha_start + src]) : 1.0f;
      }
    }
  }
  const device_ptr handle = device_webgpu_create_texture_atlas(device, atlas_width, atlas_height, atlas);
  if (handle) {
    LOG_INFO << "Cycles WebGPU cubic image atlas: " << uploaded << " images, " << atlas_width
             << " x " << atlas_height;
  }
  return handle;
}

WebGPUSceneBuffers::WebGPUSceneBuffers(Device &device)
    : bvh(&device, "WebGPU BVH2"),
      surface(&device, "WebGPU surface metadata"),
      light(&device, "WebGPU direct light"),
      svm(&device, "WebGPU SVM bytecode")
{
}

bool WebGPUSceneBuffers::upload(const std::span<const uint32_t> bvh_words,
                                const std::span<const uint32_t> surface_words,
                                const std::span<const uint32_t> light_words,
                                const std::span<const uint32_t> svm_words)
{
  const auto upload_words = [](device_only_memory<uint32_t> &buffer,
                               const std::span<const uint32_t> words) {
    buffer.alloc_to_device(words.size());
    return buffer.device_pointer &&
           device_webgpu_upload(*buffer.device, buffer, std::as_bytes(words));
  };
  return upload_words(bvh, bvh_words) && upload_words(surface, surface_words) &&
         upload_words(light, light_words) && upload_words(svm, svm_words);
}

bool cycles_webgpu_encode_scene(const Scene &scene,
                                const size_t binding_words,
                                CyclesWebGPUScene &out,
                                std::string &reason)
{
  if (const char *error = scene_incompatibility(scene)) {
    reason = error;
    return false;
  }
  const DeviceScene &dscene = scene.dscene;

  const bool volume = std::ranges::any_of(host_data(dscene.object_flag), [](const uint flag) {
    return (flag & SD_OBJECT_HAS_VOLUME) != 0;
  });
  /* Check the upload budget before allocating the packed buffers. */
  const uint64_t bvh_words = 10 + 4 * uint64_t(dscene.bvh_nodes.size()) +
                             4 * uint64_t(dscene.bvh_leaf_nodes.size()) +
                             3 * uint64_t(dscene.prim_index.size()) +
                             14 * uint64_t(dscene.objects.size()) +
                             3 * uint64_t(dscene.tri_vindex.size()) +
                             3 * uint64_t(dscene.tri_verts.size());
  const bool have_smooth = std::ranges::any_of(
      host_data(dscene.tri_shader),
      [](const uint shader) { return (shader & SHADER_SMOOTH_NORMAL) != 0; });
  if (have_smooth && dscene.attributes_normal.size() == 0) {
    reason = "WebGPU Cycles needs synchronized normals for smooth triangles";
    return false;
  }
  const uint64_t surface_words = CYCLES_WEBGPU_SURFACE_OFFSETS +
                                 (15 + CYCLES_WEBGPU_SURFACE_OBJECT_WORDS) *
                                     uint64_t(dscene.objects.size()) +
                                 uint64_t(dscene.tri_shader.size()) + 2 * dscene.shaders.size() +
                                 (have_smooth ? dscene.attributes_normal.size() : 0);
  if (bvh_words > binding_words || surface_words > binding_words ||
      CYCLES_WEBGPU_LIGHT_HEADER_WORDS > binding_words)
  {
    reason = "WebGPU Cycles geometry or lighting exceeds the adapter storage binding limit";
    return false;
  }
  const size_t surface_extra_words = binding_words - size_t(surface_words);
  std::vector<uint> volume_shaders;
  std::vector<VolumeMedium> volume_media;
  if (const char *error = volume ? encode_volumes(scene, surface_extra_words, volume_shaders,
                                                volume_media) : nullptr)
  {
    reason = error;
    return false;
  }
  const bool mixed_volume = std::ranges::any_of(volume_media, [](const VolumeMedium &medium) {
    return !(medium.words[CYCLES_WEBGPU_VOLUME_SHADER_FLAGS] & SD_HAS_ONLY_VOLUME);
  });
  const bool volume_scattering = volume && needs_volume_intervals(volume_media);
  size_t volume_record_words = volume_scattering ?
                                   volume_media.size() * CYCLES_WEBGPU_VOLUME_SCATTER_WORDS :
                                   CYCLES_WEBGPU_ABSORPTION_WORDS;
  for (const VolumeMedium &medium : volume_media) {
    volume_record_words += medium.grids.size();
  }
  WorldEnvironment environment;
  const char *world_error = find_world_environment(scene, environment);
  /* Scattering and world bytecode use the mixed layout with the stock CDF or tree. */
  const bool light_tree = volume_scattering || uses_mixed_lighting(dscene) ||
                          !environment.program.empty();
  const auto &integrator = dscene.data.integrator;
  const int depth = integrator.max_bounce;

  CyclesWebGPUScene encoded;
  if (depth < 1 || integrator.min_bounce < 1 || integrator.max_diffuse_bounce < 1 ||
      integrator.max_glossy_bounce < 1 || integrator.max_transmission_bounce < 1 ||
      !std::isfinite(integrator.scrambling_distance) || integrator.scrambling_distance < 0.0f ||
      (dscene.data.kernel_features & KERNEL_FEATURE_PATH_GUIDING))
  {
    reason = "WebGPU Cycles needs valid bounce settings and path guiding turned off";
    return false;
  }
  if (const char *error = encode_lighting(scene, light_tree, encoded.lighting)) {
    reason = error;
    return false;
  }
  std::vector<uint32_t> transparent_shadows;
  bool varying_transparency = false;
  bool cubic_transparent_shadows = false;
  bool shadow_extended_svm = false;
  bool has_principled_alpha = false;
  bool has_diffuse_principled_mix = false;
  bool has_independent_mixed_bump = false;
  bool has_glossy_displacement_bump = false;
  bool has_mixed_displacement_bump = false;
  bool has_glossy_transparent = false;
  bool has_transparent_emission = false;
  bool has_ordered_transparency = false;
  uint32_t closure_compositions = 0u;
  std::vector<bool> ordered_closure_shaders;
  if (!cycles_webgpu_encode_svm_materials(scene,
                                         encoded.svm,
                                         transparent_shadows,
                                         varying_transparency,
                                         cubic_transparent_shadows,
                                         shadow_extended_svm,
                                         has_principled_alpha,
                                         encoded.has_ggx,
                                         encoded.has_transmission,
                                         encoded.has_subsurface,
                                         encoded.advanced_svm,
                                         encoded.ao_only_queries,
                                         encoded.residual_displacement_bump,
                                         encoded.paired_principled_layers,
                                         encoded.single_principled_coat,
                                         encoded.coated_principled_emission,
                                         encoded.compact_voronoi_distance,
                                         encoded.has_mixed_diffuse,
                                         has_diffuse_principled_mix,
                                         has_independent_mixed_bump,
                                         has_glossy_displacement_bump,
                                         has_mixed_displacement_bump,
                                         has_glossy_transparent,
                                         has_transparent_emission,
                                         has_ordered_transparency,
                                         closure_compositions,
                                         ordered_closure_shaders,
                                         volume_shaders,
                                         reason,
                                         binding_words))
  {
    return false;
  }
  const bool transparent = !transparent_shadows.empty();
  const bool varying_volume_shadows = volume && varying_transparency;
  const size_t tangent_words = (encoded.svm[1] & (CYCLES_WEBGPU_SVM_HAIR | CYCLES_WEBGPU_SVM_BUMP)) ?
                                   CYCLES_WEBGPU_SURFACE_HAIR_HEADER_WORDS : 0;
  const bool ray_differentials = encoded.svm[1] & CYCLES_WEBGPU_SVM_NEEDS_DIFFERENTIALS;
  const uint unhandled_volume_materials = CYCLES_WEBGPU_SVM_AOV | CYCLES_WEBGPU_SVM_PIXEL_WIREFRAME |
                                          CYCLES_WEBGPU_SVM_RAY_PORTAL | CYCLES_WEBGPU_SVM_HAIR;
  if (volume &&
      (encoded.has_subsurface ||
       ((transparent || encoded.has_transmission || ray_differentials) && !volume_scattering) ||
       (encoded.svm[1] & unhandled_volume_materials)))
  {
    reason = "WebGPU Cycles volumes need compatible surface shading and Combined film";
    return false;
  }
  const uint32_t volume_offsets = varying_volume_shadows ?
                                      CYCLES_WEBGPU_SURFACE_VOLUME_SHADOW_OFFSETS :
                                      CYCLES_WEBGPU_SURFACE_ABSORPTION_OFFSETS;
  const size_t volume_words = volume ?
      volume_offsets - CYCLES_WEBGPU_SURFACE_OFFSETS + volume_record_words +
          (varying_volume_shadows ? CYCLES_WEBGPU_VOLUME_SHADOW_PARAMS_WORDS : 0u) : 0;
  if ((encoded.svm[1] & CYCLES_WEBGPU_SVM_PIXEL_WIREFRAME) && encoded.has_subsurface) {
    reason = "WebGPU Cycles needs ray differentials through subsurface scattering for pixel Wireframe";
    return false;
  }
  if (transparent && (encoded.svm[1] & CYCLES_WEBGPU_SVM_AOV)) {
    reason = "WebGPU Cycles needs transparent film data pass accumulation for AOVs";
    return false;
  }
  const size_t transparent_words = transparent ?
                                       1 + CYCLES_WEBGPU_TRANSPARENT_SHADOW_HEADER_WORDS +
                                           transparent_shadows.size() :
                                       0;
  if (transparent &&
      (scene.params.use_bvh_spatial_split || integrator.transparent_min_bounce < 0 ||
       integrator.transparent_max_bounce < 0 ||
       uint(integrator.transparent_min_bounce) > CYCLES_WEBGPU_TRANSPARENT_BOUNCE_MASK ||
       uint(integrator.transparent_max_bounce) > CYCLES_WEBGPU_TRANSPARENT_BOUNCE_MASK))
  {
    reason = "WebGPU Cycles constant transparent shadows need a BVH without spatial splitting and valid bounce limits";
    return false;
  }
  const bool multiple_points = !light_tree && dscene.lights.size() > 1;
  const bool area = !light_tree && !multiple_points &&
                    dscene.lights.data()[0].type == LIGHT_AREA;
  if (encoded.has_ggx) {
    const size_t lookup_words = CYCLES_WEBGPU_GGX_LOOKUP_WORDS +
                                (encoded.has_transmission ? CYCLES_WEBGPU_GGX_GLASS_LOOKUP_WORDS :
                                                            0);
    if (lookup_words > binding_words - encoded.lighting.size()) {
      reason = "WebGPU Cycles GGX lookup tables exceed the lighting storage binding limit";
      return false;
    }
    if (const char *error = encode_ggx_tables(
            dscene, light_tree, encoded.has_transmission, transparent, volume_shaders,
            encoded.lighting))
    {
      reason = error;
      return false;
    }
  }
  if (transparent_words + tangent_words + volume_words > surface_extra_words) {
    reason = "WebGPU Cycles surface metadata exceeds the adapter storage binding limit";
    return false;
  }
  const size_t light_extra_words = binding_words - encoded.lighting.size();
  const char *lighting_error = nullptr;
  if (light_tree) {
    lighting_error = world_error ? world_error :
                                    encode_stock_mixed_lighting(scene, environment, volume_scattering,
                                                                light_extra_words, encoded.lighting);
  }
  else if (multiple_points) {
    lighting_error = encode_point_lighting(dscene, light_extra_words, encoded.lighting);
  }
  else if (area) {
    lighting_error = world_error ? world_error :
                                    encode_area_lighting(scene, environment, light_extra_words,
                                                         encoded.lighting);
  }
  if (lighting_error) {
    reason = lighting_error;
    return false;
  }
  if (volume_scattering && ray_differentials) {
    encoded.lighting[CYCLES_WEBGPU_LIGHT_EXTENDED] |= CYCLES_WEBGPU_LIGHT_VOLUME_DIFFERENTIALS;
  }
  if (integrator.light_inv_rr_threshold > 0.0f) {
    const auto lut = light_sampling_table(dscene);
    if (CYCLES_WEBGPU_LIGHT_SAMPLING_WORDS + lut.size() >
        binding_words - encoded.lighting.size())
    {
      reason = "WebGPU Cycles light termination sampling exceeds the lighting storage binding limit";
      return false;
    }
    encoded.lighting[CYCLES_WEBGPU_LIGHT_INV_RR_THRESHOLD] = float_word(
        integrator.light_inv_rr_threshold);
    encoded.lighting[CYCLES_WEBGPU_LIGHT_RR_SAMPLING] = uint32_t(encoded.lighting.size());
    append_light_sampling(integrator, lut, encoded.lighting);
  }
  encoded.max_bounces = uint32_t(depth - 1);

  PackedWords bvh;
  bvh.words.reserve(size_t(bvh_words));
  bvh.words.resize(10);
  bvh.words[0] = uint32_t(dscene.data.bvh.root);
  bvh.words[1] = uint32_t(bvh.words.size());
  for (const int4 node : host_data(dscene.bvh_nodes)) {
    bvh.append(node);
  }
  bvh.words[2] = uint32_t(bvh.words.size());
  for (const int4 node : host_data(dscene.bvh_leaf_nodes)) {
    bvh.append(node);
  }
  bvh.append(3, dscene.prim_visibility);
  bvh.append(4, dscene.prim_index);
  bvh.append(5, dscene.prim_object);
  bvh.append(6, dscene.object_node);
  bvh.words[7] = uint32_t(bvh.words.size());
  for (const KernelObject &object : host_data(dscene.objects)) {
    bvh.words.push_back(uint32_t(object.position_offset));
    bvh.append(object.itfm);
  }
  bvh.words[8] = uint32_t(bvh.words.size());
  for (const packed_uint3 index : host_data(dscene.tri_vindex)) {
    bvh.words.insert(bvh.words.end(), {index.x, index.y, index.z});
  }
  bvh.words[9] = uint32_t(bvh.words.size());
  for (const packed_float3 vertex : host_data(dscene.tri_verts)) {
    bvh.words.insert(bvh.words.end(),
                     {float_word(vertex.x), float_word(vertex.y), float_word(vertex.z)});
  }

  PackedWords surface;
  surface.words.reserve(size_t(surface_words) + transparent_words + tangent_words + volume_words);
  surface.words.resize((volume ? volume_offsets :
                                  (transparent ? CYCLES_WEBGPU_SURFACE_TRANSPARENT_OFFSETS :
                                                 CYCLES_WEBGPU_SURFACE_OFFSETS)) + tangent_words);
  surface.words[0] = uint32_t(surface.words.size());
  for (size_t index = 0; index < dscene.objects.size(); ++index) {
    surface.words.push_back(dscene.object_flag.data()[index]);
    surface.append(dscene.objects.data()[index].tfm);
  }
  surface.append(1, dscene.tri_shader);
  surface.words[2] = uint32_t(surface.words.size());
  for (const KernelObject &object : host_data(dscene.objects)) {
    surface.words.push_back(uint32_t(object.mesh_volume.normal_offset));
    surface.words.push_back(float_word(object.shadow_terminator_geometry_offset));
  }
  surface.words[3] = uint32_t(surface.words.size());
  if (have_smooth) {
    for (const packed_normal normal : host_data(dscene.attributes_normal)) {
      surface.words.push_back(normal.value);
    }
  }
  surface.words[4] = uint32_t(surface.words.size());
  const uint32_t caustics_flag = integrator.caustics_reflective ?
                                      0u : CYCLES_WEBGPU_SHADER_NO_REFLECTIVE_CAUSTICS;
  const uint32_t diffuse_principled_flag = has_diffuse_principled_mix ?
                                               CYCLES_WEBGPU_SHADER_DIFFUSE_PRINCIPLED_MIX : 0u;
  const uint32_t independent_bump_flag = has_independent_mixed_bump ?
                                             CYCLES_WEBGPU_SHADER_INDEPENDENT_MIXED_BUMP : 0u;
  const uint32_t glossy_displacement_bump_flag = has_glossy_displacement_bump ?
      CYCLES_WEBGPU_SHADER_GLOSSY_DISPLACEMENT_BUMP : 0u;
  const uint32_t mixed_displacement_bump_flag = has_mixed_displacement_bump ?
      CYCLES_WEBGPU_SHADER_MIXED_DISPLACEMENT_BUMP : 0u;
  const uint32_t glossy_transparent_flag = has_glossy_transparent ?
                                              CYCLES_WEBGPU_SHADER_GLOSSY_TRANSPARENT : 0u;
  const uint32_t transparent_emission_flag = has_transparent_emission ?
      CYCLES_WEBGPU_SHADER_TRANSPARENT_EMISSION : 0u;
  const uint32_t ordered_transparency_flag = has_ordered_transparency ?
                                                 CYCLES_WEBGPU_SHADER_ORDERED_TRANSPARENCY : 0u;
  for (size_t i = 0; i < dscene.shaders.size(); ++i) {
    const uint32_t ordered_flag = ordered_closure_shaders[i] ?
                                      CYCLES_WEBGPU_SHADER_ORDERED_CLOSURES : 0u;
    surface.words.push_back(uint32_t(dscene.shaders.data()[i].flags) | caustics_flag |
                            diffuse_principled_flag | independent_bump_flag |
                            glossy_displacement_bump_flag | mixed_displacement_bump_flag |
                            glossy_transparent_flag | transparent_emission_flag |
                            ordered_transparency_flag | closure_compositions | ordered_flag);
  }

  surface.words[CYCLES_WEBGPU_SURFACE_OBJECT_SVM_OFFSET] = uint32_t(surface.words.size());
  for (const KernelObject &object : host_data(dscene.objects)) {
    surface.append(object.itfm);
    for (const float value : {object.color[0],
                              object.color[1],
                              object.color[2],
                              object.alpha,
                              object.pass_id,
                              object.random_number,
                              object.dupli_generated[0],
                              object.dupli_generated[1],
                              object.dupli_generated[2],
                              object.dupli_uv[0],
                              object.dupli_uv[1]})
    {
      surface.words.push_back(float_word(value));
    }
  }
  surface.words[CYCLES_WEBGPU_SURFACE_SHADER_PASS_OFFSET] = uint32_t(surface.words.size());
  for (const KernelShader &shader : host_data(dscene.shaders)) {
    surface.words.push_back(uint32_t(shader.pass_id));
  }
  if (transparent) {
    surface.words[CYCLES_WEBGPU_SURFACE_TRANSPARENT_OFFSET] = uint32_t(surface.words.size());
    surface.words.push_back(uint32_t(integrator.transparent_max_bounce));
    const uint32_t shadow_flags = (integrator.transparent_shadows ?
                                       CYCLES_WEBGPU_TRANSPARENT_SHADOW_ENABLE :
                                       0) |
                                  (varying_transparency ?
                                       CYCLES_WEBGPU_TRANSPARENT_SHADOW_VARYING :
                                       0) |
                                  (has_principled_alpha ?
                                       CYCLES_WEBGPU_TRANSPARENT_SHADOW_PRINCIPLED_ALPHA :
                                       0) |
                                  (cubic_transparent_shadows ?
                                       CYCLES_WEBGPU_TRANSPARENT_SHADOW_CUBIC :
                                       0) |
                                  (shadow_extended_svm ?
                                       CYCLES_WEBGPU_TRANSPARENT_SHADOW_EXTENDED_SVM :
                                       0);
    surface.words.push_back(shadow_flags);
    surface.words.insert(surface.words.end(), transparent_shadows.begin(), transparent_shadows.end());
  }

  if (volume) {
    const size_t offset = surface.words.size();
    surface.words[CYCLES_WEBGPU_SURFACE_ABSORPTION_OFFSET] = uint32_t(offset);
    if (volume_scattering) {
      surface.words[CYCLES_WEBGPU_SURFACE_MIXED_VOLUME] = uint32_t(mixed_volume);
      surface.words[CYCLES_WEBGPU_SURFACE_VOLUME_COUNT] = uint32_t(volume_media.size());
      size_t grid_offset = offset + volume_media.size() * CYCLES_WEBGPU_VOLUME_SCATTER_WORDS;
      for (VolumeMedium &medium : volume_media) {
        if (!medium.grids.empty()) {
          medium.words[CYCLES_WEBGPU_VOLUME_GRID_OFFSET] = uint32_t(grid_offset);
          grid_offset += medium.grids.size();
        }
        surface.words.insert(surface.words.end(), medium.words.begin(), medium.words.end());
      }
      for (const VolumeMedium &medium : volume_media) {
        surface.words.insert(surface.words.end(), medium.grids.begin(), medium.grids.end());
      }
    }
    else {
      const VolumeWords &words = volume_media.front().words;
      surface.words.insert(surface.words.end(), words.begin(), words.begin() + CYCLES_WEBGPU_ABSORPTION_WORDS);
    }
  }
  if (varying_volume_shadows) {
    surface.words[CYCLES_WEBGPU_SURFACE_VOLUME_SHADOW_PARAMS] = uint32_t(surface.words.size());
    surface.words.resize(surface.words.size() + CYCLES_WEBGPU_VOLUME_SHADOW_PARAMS_WORDS);
  }

  encoded.bvh = std::move(bvh.words);
  encoded.surface = std::move(surface.words);
  out = std::move(encoded);
  reason.clear();
  return true;
}

CCL_NAMESPACE_END
