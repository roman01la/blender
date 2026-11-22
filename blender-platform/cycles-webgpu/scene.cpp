/* SPDX-License-Identifier: Apache-2.0 */

#include "device/webgpu/scene.h"

#include "bvh/bvh2.h"
#include "device/webgpu/device.h"
#include "device/webgpu/lighting_layout.h"
#include "device/webgpu/material_layout.h"
#include "device/webgpu/pipeline.h"
#include "device/webgpu/svm_materials.h"
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
  if (integrator.ao_bounces || integrator.ao_additive_factor != 0.0f ||
      integrator.sample_clamp_direct != FLT_MAX || integrator.light_inv_rr_threshold != 0.0f ||
      (integrator.filter_closures &
       (FILTER_CLOSURE_DIFFUSE | FILTER_CLOSURE_EMISSION | FILTER_CLOSURE_DIRECT_LIGHT)))
  {
    return "WebGPU Cycles needs unclamped direct light without ambient occlusion";
  }
  return nullptr;
}

const char *encode_lamp(const DeviceScene &dscene,
                        const KernelLight &light,
                        const bool allow_area,
                        uint32_t *words,
                        const bool allow_finite_point = false)
{
  const bool point = light.type == LIGHT_POINT &&
                     ((light.spot.radius == 0.0f && !light.spot.is_sphere &&
                       light.spot.eval_fac == 0.25f * M_1_PI_F) ||
                      (allow_finite_point && std::isfinite(light.spot.radius) &&
                       light.spot.radius > 0.0f && std::isfinite(light.spot.eval_fac) &&
                       light.spot.eval_fac > 0.0f));
  const bool area = allow_area && light.type == LIGHT_AREA;
  if ((!point && !area) || !(light.shader_id & SHADER_CAST_SHADOW) ||
      (light.shader_id & SHADER_EXCLUDE_DIFFUSE) || !std::isfinite(light.max_bounces) ||
      light.max_bounces < 0.0f)
  {
    return "WebGPU Cycles needs a supported point or area light with diffuse lighting and shadows";
  }

  float3 light_emission;
  if (!constant_emission(dscene, light.shader_id, light_emission)) {
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

bool uses_stock_light_tree(const DeviceScene &scene)
{
  const KernelIntegrator &integrator = scene.data.integrator;
  if (!integrator.use_light_tree) {
    return false;
  }
  if (scene.light_tree_emitters.size() > scene.lights.size()) {
    return true;
  }
  if (integrator.num_lights > 1 || integrator.num_background_lights > 0) {
    return true;
  }

  if (scene.lights.size() != 1 || scene.lights.data()[0].type != LIGHT_POINT) {
    return false;
  }
  if (scene.lights.data()[0].spot.radius > 0.0f) {
    return true;
  }

  /* The single-point shader misses indirect world lighting. Reuse the stock
   * Light Tree path for a point light with a visible constant world. */
  constexpr int indirect = SHADER_EXCLUDE_DIFFUSE | SHADER_EXCLUDE_GLOSSY;
  const int world_shader = scene.data.background.surface_shader;
  float3 emission;
  return (world_shader & indirect) != indirect && constant_emission(scene, world_shader, emission) &&
         (emission.x != 0.0f || emission.y != 0.0f || emission.z != 0.0f);
}

const char *encode_lighting(const Scene &scene,
                            const bool light_tree,
                            std::vector<uint32_t> &lighting)
{
  const DeviceScene &dscene = scene.dscene;
  const KernelIntegrator &integrator = dscene.data.integrator;
  const auto lights = host_data(dscene.lights);
  if ((!light_tree && lights.empty()) || lights.size() != size_t(integrator.num_lights) ||
      integrator.num_portals || (integrator.num_background_lights && !light_tree) ||
      !integrator.use_direct_light)
  {
    return "WebGPU Cycles needs supported lights without portals, and Light Tree for sampled worlds";
  }
  if (!integrator.use_light_tree && dscene.light_distribution.size() > lights.size() + 1) {
    return "WebGPU Cycles needs the Light Tree to sample emissive meshes";
  }
  lighting.assign(CYCLES_WEBGPU_LIGHT_HEADER_WORDS, 0);
  lighting[3] = float_word(integrator.sample_clamp_indirect);
  lighting[CYCLES_WEBGPU_LIGHT_FILTER_GLOSSY] = float_word(integrator.filter_glossy);
  if (light_tree) {
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

const char *area_light_incompatibility(const DeviceScene &dscene, const KernelLight &light)
{
  const KernelAreaLight &area = light.area;
  const KernelIntegrator &integrator = dscene.data.integrator;
  if (integrator.direct_light_sampling_type != DIRECT_LIGHT_SAMPLING_MIS ||
      !(light.shader_id & SHADER_USE_MIS) || area.tan_half_spread != FLT_MAX ||
      !std::isfinite(area.len_u) || !std::isfinite(area.len_v) || !std::isfinite(area.invarea) ||
      area.len_u <= 0.0f || area.len_v <= 0.0f || area.invarea <= 0.0f ||
      !finite_rgb(make_float3(area.axis_u)) || !finite_rgb(make_float3(area.axis_v)) ||
      !finite_rgb(make_float3(area.dir)) ||
      (dscene.data.kernel_features & KERNEL_FEATURE_MNEE))
  {
    return "WebGPU Cycles needs a rectangular area light with full spread and MIS enabled";
  }
  return nullptr;
}

const char *find_environment_image(const Scene &scene,
                                   KernelImageInfo &image,
                                   float &strength,
                                   float3 &background)
{
  const DeviceScene &dscene = scene.dscene;
  const int world = dscene.data.background.surface_shader;
  if ((world & SHADER_EXCLUDE_ANY) != SHADER_EXCLUDE_ANY &&
      !constant_emission(dscene, world, background))
  {
    const size_t shader = uint(world) & SHADER_MASK;
    if (shader >= scene.shaders.size() || !scene.shaders[shader]->graph) {
      return "WebGPU Cycles needs a synchronized world shader";
    }
    const ShaderOutput *surface = scene.shaders[shader]->graph->output()->input("Surface")->link;
    if (!surface || surface->parent->type != BackgroundNode::get_node_type()) {
      return "WebGPU Cycles needs a constant world or a Background with one environment image";
    }
    auto &node = static_cast<BackgroundNode &>(*surface->parent);
    const ShaderOutput *color = node.input("Color")->link;
    strength = node.get_strength();
    if (!color || color->name() != "Color" ||
        color->parent->type != EnvironmentTextureNode::get_node_type() ||
        node.input("Strength")->link || !std::isfinite(strength) || strength < 0.0f)
    {
      return "WebGPU Cycles needs an environment image with constant world strength";
    }
    auto &environment = static_cast<EnvironmentTextureNode &>(*color->parent);
    auto mapping = environment.tex_mapping;
    const ShaderOutput *vector = environment.input("Vector")->link;
    /* ShaderGraph::default_inputs wires the background direction through Geometry Position. */
    if (environment.get_animated() || !mapping.skip() ||
        environment.get_projection() != NODE_ENVIRONMENT_EQUIRECTANGULAR ||
        (vector &&
         (vector->parent->type != GeometryNode::get_node_type() || vector->name() != "Position")))
    {
      return "WebGPU Cycles needs an untransformed equirectangular environment";
    }
    if (environment.handle.empty() ||
        !cycles_webgpu_svm_image_info(scene, environment.handle.kernel_id(), image) ||
        image.data_type != IMAGE_DATA_TYPE_FLOAT4 || image.extension != EXTENSION_REPEAT ||
        (image.interpolation != INTERPOLATION_CLOSEST && image.interpolation != INTERPOLATION_LINEAR))
    {
      return "WebGPU Cycles needs a synchronized floating-point environment image with linear or "
             "closest interpolation";
    }
  }
  return nullptr;
}

const char *append_environment_pixels(const KernelImageInfo &image,
                                      const float strength,
                                      std::vector<uint32_t> &lighting)
{
  const size_t texels = size_t(image.width) * image.height;
  for (size_t i = 0; i < texels; ++i) {
    const float3 rgb = strength *
                       make_float3(reinterpret_cast<const float4 *>(uintptr_t(image.data))[i]);
    if (!finite_rgb(rgb)) {
      return "WebGPU Cycles needs finite HDRI colors";
    }
    lighting.insert(lighting.end(), {float_word(rgb.x), float_word(rgb.y), float_word(rgb.z)});
  }
  return nullptr;
}

/* Append a sampled emitter and optional environment after the shared GGX tables.
 * The environment contributes on ray misses, so this subset uses World sampling None. */
const char *encode_area_lighting(const Scene &scene,
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
  KernelImageInfo image{};
  float strength = 1.0f;
  float3 background = zero_float3();
  if (const char *error = find_environment_image(scene, image, strength, background)) {
    return error;
  }

  constexpr size_t descriptor_words = CYCLES_WEBGPU_EXTENDED_LIGHT_WORDS;
  constexpr size_t sampling_words = CYCLES_WEBGPU_LIGHT_SAMPLING_WORDS;
  const auto lut = light_sampling_table(dscene);
  const size_t texels = size_t(image.width) * image.height;
  if (descriptor_words + sampling_words + lut.size() > remaining_words ||
      texels > (remaining_words - descriptor_words - sampling_words - lut.size()) / 3)
  {
    return "WebGPU Cycles area lighting exceeds the scene upload budget";
  }

  const size_t base = lighting.size();
  lighting[CYCLES_WEBGPU_LIGHT_EXTENDED] = uint32_t(base);
  write_rgb(lighting.data() + 8, background);
  lighting.resize(base + descriptor_words);
  write_rgb(lighting.data() + base + CYCLES_WEBGPU_AREA_AXIS_U, make_float3(area.axis_u));
  lighting[base + CYCLES_WEBGPU_AREA_AXIS_U + 3] = float_word(area.len_u);
  write_rgb(lighting.data() + base + CYCLES_WEBGPU_AREA_AXIS_V, make_float3(area.axis_v));
  lighting[base + CYCLES_WEBGPU_AREA_AXIS_V + 3] = float_word(area.len_v);
  write_rgb(lighting.data() + base + CYCLES_WEBGPU_AREA_DIRECTION, make_float3(area.dir));
  lighting[base + CYCLES_WEBGPU_AREA_DIRECTION + 3] = float_word(area.invarea);
  lighting[base + CYCLES_WEBGPU_WORLD_WIDTH] = image.width;
  lighting[base + CYCLES_WEBGPU_WORLD_HEIGHT] = image.height;
  lighting[base + CYCLES_WEBGPU_WORLD_SHADER] = uint32_t(world);
  lighting[base + CYCLES_WEBGPU_AREA_SHADER] = uint32_t(light.shader_id);
  lighting[base + CYCLES_WEBGPU_WORLD_INTERPOLATION] = image.interpolation;
  if (const char *error = append_environment_pixels(image, strength, lighting)) {
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

const char *pack_light_tree_lamps(const DeviceScene &dscene,
                                 const std::span<const KernelLightTreeEmitter> emitters,
                                 const std::span<uint32_t> words,
                                 uint32_t &first_forward_index,
                                 uint32_t &world_index,
                                 uint32_t &world_emitter)
{
  const auto lights = host_data(dscene.lights);
  const auto light_to_tree = host_data(dscene.light_to_tree);
  for (size_t index = 0; index < lights.size(); ++index) {
    const KernelLight &light = lights[index];
    if (size_t(light.object_id) >= light_to_tree.size()) {
      return "WebGPU Cycles needs synchronized Light Tree lamp objects";
    }
    const uint32_t emitter_index = light_to_tree[light.object_id];
    if (emitter_index >= emitters.size() || emitters[emitter_index].light.id != ~int(index) ||
        emitters[emitter_index].object_id != light.object_id)
    {
      return "WebGPU Cycles needs synchronized Light Tree lamp emitters";
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

    uint32_t lamp[CYCLES_WEBGPU_POINT_EMITTER_WORDS]{};
    if (const char *error = encode_lamp(dscene, light, true, lamp, true)) {
      return error;
    }
    std::copy_n(lamp, 3, record + CYCLES_WEBGPU_TREE_LAMP_POSITION);
    std::copy_n(lamp + 4, 3, record + CYCLES_WEBGPU_TREE_LAMP_STRENGTH);
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
    if (first_forward_index == CYCLES_WEBGPU_TREE_NO_LAMP) {
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
    if (const char *error = area_light_incompatibility(dscene, light)) {
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

const char *pack_light_tree_meshes(const DeviceScene &dscene,
                                   const std::span<const KernelLightTreeNode> nodes,
                                   const std::span<const KernelLightTreeEmitter> emitters,
                                   uint32_t *header,
                                   const std::span<uint32_t> lighting)
{
  const auto objects = host_data(dscene.objects);
  const auto object_flags = host_data(dscene.object_flag);
  const auto object_to_tree = host_data(dscene.light_to_tree);
  const auto object_prims = host_data(dscene.object_prim_offset);
  const auto object_lookup = host_data(dscene.object_lookup_offset);
  const auto triangle_to_tree = host_data(dscene.triangle_to_tree);
  if (object_to_tree.size() != objects.size() || object_prims.size() != objects.size() ||
      object_lookup.size() != objects.size() || triangle_to_tree.empty())
  {
    return "WebGPU Cycles needs synchronized Light Tree mesh lookups";
  }

  const auto object_records = lighting.subspan(header[CYCLES_WEBGPU_TREE_OBJECTS],
                                               objects.size() * CYCLES_WEBGPU_TREE_OBJECT_WORDS);
  const auto triangle_indices = lighting.subspan(header[CYCLES_WEBGPU_TREE_TRIANGLE_INDICES],
                                                 header[CYCLES_WEBGPU_TREE_TRIANGLE_INDEX_COUNT]);
  const auto lookup_records = lighting.subspan(header[CYCLES_WEBGPU_TREE_TRIANGLE_TO_TREE],
                                               triangle_to_tree.size());
  std::ranges::fill(triangle_indices, CYCLES_WEBGPU_TREE_NO_LAMP);
  std::ranges::fill(lookup_records, CYCLES_WEBGPU_TREE_NO_LAMP);
  for (size_t i = 0; i < objects.size(); ++i) {
    object_records[i * CYCLES_WEBGPU_TREE_OBJECT_WORDS + CYCLES_WEBGPU_TREE_OBJECT_TO_TREE] =
        CYCLES_WEBGPU_TREE_NO_LAMP;
  }

  /* Fill only mapped mesh objects. Blender's lookup arrays do not initialize
   * non-emissive objects and non-emissive faces. */
  for (size_t i = 0; i < emitters.size(); ++i) {
    const KernelLightTreeEmitter &emitter = emitters[i];
    if (emitter.light.id < 0 || emitter.object_id != OBJECT_NONE) {
      continue;
    }
    const size_t object_index = size_t(emitter.mesh.object_id);
    if (object_index >= objects.size() || size_t(emitter.mesh.node_id) >= nodes.size() ||
        object_to_tree[object_index] != i ||
        object_lookup[object_index] >= triangle_to_tree.size())
    {
      return "WebGPU Cycles needs synchronized Light Tree mesh emitters";
    }
    const KernelObject &object = objects[object_index];
    uint32_t *record = object_records.data() + object_index * CYCLES_WEBGPU_TREE_OBJECT_WORDS;
    record[CYCLES_WEBGPU_TREE_OBJECT_FLAGS] = object_flags[object_index];
    record[CYCLES_WEBGPU_TREE_OBJECT_TO_TREE] = uint32_t(i);
    record[CYCLES_WEBGPU_TREE_OBJECT_PRIM_OFFSET] = object_prims[object_index];
    record[CYCLES_WEBGPU_TREE_OBJECT_LOOKUP_OFFSET] = object_lookup[object_index];
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
    if (!write_transform(object.tfm, record + CYCLES_WEBGPU_TREE_OBJECT_TRANSFORM) ||
        !write_transform(object.itfm, record + CYCLES_WEBGPU_TREE_OBJECT_INVERSE_TRANSFORM))
    {
      return "WebGPU Cycles needs finite Light Tree mesh transforms";
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
  for (size_t i = 0; i < emitters.size(); ++i) {
    const KernelLightTreeEmitter &emitter = emitters[i];
    if (emitter.light.id < 0 || emitter.object_id == OBJECT_NONE) {
      continue;
    }
    const size_t object_index = size_t(emitter.object_id);
    const size_t prim = size_t(emitter.triangle.id);
    if (object_index >= objects.size() || prim >= indices.size() ||
        triangle_indices[prim] != CYCLES_WEBGPU_TREE_NO_LAMP ||
        object_records[object_index * CYCLES_WEBGPU_TREE_OBJECT_WORDS +
                       CYCLES_WEBGPU_TREE_OBJECT_TO_TREE] == CYCLES_WEBGPU_TREE_NO_LAMP ||
        prim < object_prims[object_index])
    {
      return "WebGPU Cycles needs synchronized Light Tree triangle emitters";
    }
    const size_t lookup = prim - object_prims[object_index] + object_lookup[object_index];
    if (lookup >= triangle_to_tree.size() || triangle_to_tree[lookup] != i) {
      return "WebGPU Cycles needs the stock Light Tree triangle lookup";
    }
    if (emitter.visibility_flag &
        (SHADER_EXCLUDE_DIFFUSE | SHADER_EXCLUDE_GLOSSY | SHADER_EXCLUDE_TRANSMIT))
    {
      return "WebGPU Cycles needs diffuse, glossy and transmission lighting on emissive meshes";
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

    lookup_records[lookup] = uint32_t(i);
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
  }
  return nullptr;
}

const char *encode_stock_light_tree(const Scene &scene,
                                   const size_t remaining_words,
                                   std::vector<uint32_t> &lighting)
{
  const DeviceScene &dscene = scene.dscene;
  const KernelIntegrator &integrator = dscene.data.integrator;
  const KernelBackground &world = dscene.data.background;
  const auto lights = host_data(dscene.lights);
  const auto nodes = host_data(dscene.light_tree_nodes);
  const auto emitters = host_data(dscene.light_tree_emitters);
  const size_t triangle_count = std::ranges::count_if(
      emitters, [](const KernelLightTreeEmitter &e) {
        return e.light.id >= 0 && e.object_id != OBJECT_NONE;
      });
  if (nodes.empty() || emitters.empty() || emitters.size() < lights.size() ||
      (triangle_count == 0 && lights.size() != emitters.size()) ||
      integrator.num_background_lights > 1 ||
      integrator.direct_light_sampling_type != DIRECT_LIGHT_SAMPLING_MIS ||
      (dscene.data.kernel_features & KERNEL_FEATURE_MNEE))
  {
    return "WebGPU Cycles needs a Light Tree of supported lights with MIS";
  }

  const bool sampled_world = integrator.num_background_lights != 0;
  const auto marginal = sampled_world ? host_data(dscene.light_background_marginal_cdf) :
                                        std::span<const float2>{};
  const auto conditional = sampled_world ? host_data(dscene.light_background_conditional_cdf) :
                                           std::span<const float2>{};
  if (sampled_world &&
      (!world.use_mis || world.map_weight != 1.0f || world.portal_weight != 0.0f ||
       world.sun_weight != 0.0f || world.map_res_x <= 0 || world.map_res_y <= 0 ||
       marginal.size() != uint64_t(world.map_res_y) + 1 ||
       conditional.size() != (uint64_t(world.map_res_x) + 1) * world.map_res_y ||
       !std::isfinite(marginal.back().x) || marginal.back().x <= 0.0f))
  {
    return "WebGPU Cycles needs a synchronized world importance map without portals or a sun";
  }

  KernelImageInfo image{};
  float strength = 1.0f;
  float3 background = zero_float3();
  if (const char *error = find_environment_image(scene, image, strength, background)) {
    return error;
  }

  const auto lut = light_sampling_table(dscene);
  const uint64_t lamp_words = uint64_t(lights.size()) * CYCLES_WEBGPU_TREE_LAMP_WORDS;
  const uint64_t node_words = uint64_t(nodes.size()) * CYCLES_WEBGPU_TREE_NODE_WORDS;
  const uint64_t emitter_words = uint64_t(emitters.size()) * CYCLES_WEBGPU_TREE_EMITTER_WORDS;
  const uint64_t object_count = triangle_count ? dscene.objects.size() : 0;
  const uint64_t triangle_index_count = triangle_count ? dscene.tri_shader.size() : 0;
  const uint64_t shader_count = triangle_count ? dscene.shaders.size() : 0;
  const uint64_t triangle_lookup_count = triangle_count ? dscene.triangle_to_tree.size() : 0;
  const uint64_t object_words = object_count * CYCLES_WEBGPU_TREE_OBJECT_WORDS;
  const uint64_t triangle_words = uint64_t(triangle_count) * CYCLES_WEBGPU_TREE_TRIANGLE_WORDS;
  const uint64_t shader_words = shader_count * CYCLES_WEBGPU_TREE_SHADER_WORDS;
  const uint64_t texel_words = 3 * uint64_t(image.width) * image.height;
  const uint64_t total_words = CYCLES_WEBGPU_TREE_DESCRIPTOR_WORDS + lamp_words + node_words +
                               emitter_words + object_words + triangle_index_count +
                               triangle_words + shader_words + triangle_lookup_count +
                               2 * uint64_t(marginal.size()) + 2 * uint64_t(conditional.size()) +
                               texel_words + CYCLES_WEBGPU_LIGHT_SAMPLING_WORDS + lut.size();
  if (total_words > remaining_words) {
    return "WebGPU Cycles Light Tree and world map exceed the scene upload budget";
  }

  const size_t descriptor = lighting.size();
  const size_t lamp_offset = descriptor + CYCLES_WEBGPU_TREE_DESCRIPTOR_WORDS;
  const size_t node_offset = lamp_offset + size_t(lamp_words);
  const size_t emitter_offset = node_offset + size_t(node_words);
  const size_t object_offset = emitter_offset + size_t(emitter_words);
  const size_t triangle_index_offset = object_offset + size_t(object_words);
  const size_t triangle_offset = triangle_index_offset + size_t(triangle_index_count);
  const size_t shader_offset = triangle_offset + size_t(triangle_words);
  const size_t triangle_lookup_offset = shader_offset + size_t(shader_words);
  const size_t marginal_offset = triangle_lookup_offset + size_t(triangle_lookup_count);
  const size_t conditional_offset = marginal_offset + 2 * marginal.size();
  const size_t world_offset = conditional_offset + 2 * conditional.size();
  const size_t sampling_offset = world_offset + size_t(texel_words);

  lighting.reserve(descriptor + size_t(total_words));
  lighting.resize(marginal_offset);
  lighting[0] = CYCLES_WEBGPU_LIGHT_TREE_MARKER;
  lighting[CYCLES_WEBGPU_LIGHT_TREE_OFFSET] = uint32_t(descriptor);
  write_rgb(lighting.data() + 8, background);

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
  header[CYCLES_WEBGPU_TREE_WORLD_WIDTH] = image.width;
  header[CYCLES_WEBGPU_TREE_WORLD_HEIGHT] = image.height;
  header[CYCLES_WEBGPU_TREE_WORLD_INTERPOLATION] = image.interpolation;
  header[CYCLES_WEBGPU_TREE_WORLD_SHADER] = uint32_t(world.surface_shader);
  header[CYCLES_WEBGPU_TREE_WORLD_PIXELS] = uint32_t(world_offset);
  header[CYCLES_WEBGPU_TREE_SAMPLING] = uint32_t(sampling_offset);

  if (const char *error = pack_light_tree_lamps(
          dscene,
          emitters,
          std::span(lighting).subspan(lamp_offset, size_t(lamp_words)),
          header[CYCLES_WEBGPU_TREE_FIRST_FORWARD_LAMP],
          header[CYCLES_WEBGPU_TREE_WORLD_LAMP],
          header[CYCLES_WEBGPU_TREE_WORLD_EMITTER]))
  {
    return error;
  }
  if (sampled_world != (header[CYCLES_WEBGPU_TREE_WORLD_LAMP] != CYCLES_WEBGPU_TREE_NO_LAMP) ||
      !pack_light_tree_nodes(nodes,
                             emitters.size(),
                             std::span(lighting).subspan(node_offset, size_t(node_words))))
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
  if (triangle_count) {
    if (const char *error = pack_light_tree_meshes(dscene, nodes, emitters, header, lighting)) {
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
  if (const char *error = append_environment_pixels(image, strength, lighting)) {
    return error;
  }
  append_light_sampling(integrator, lut, lighting);
  return nullptr;
}

const char *ggx_transmission_incompatibility(const DeviceScene &scene, const bool transparent_shadows)
{
  const KernelIntegrator &integrator = scene.data.integrator;
  if (!integrator.caustics_refractive ||
      (integrator.filter_closures & FILTER_CLOSURE_TRANSMISSION))
  {
    return "WebGPU Cycles glass needs refractive caustics and transmission closures enabled";
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
  if (!uses_stock_light_tree(scene) && !has_area_light && !(world_shader & SHADER_EXCLUDE_TRANSMIT))
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
      if (scene.shaders.data()[packed_shader & SHADER_MASK].flags & SD_HAS_TRANSPARENT_SHADOW) {
        return "WebGPU Cycles glass needs supported shadow casters";
      }
    }
  }
  return nullptr;
}

const char *encode_ggx_tables(const DeviceScene &scene,
                              const bool has_transmission,
                              const bool transparent_shadows,
                              std::vector<uint32_t> &lighting)
{
  const KernelIntegrator &integrator = scene.data.integrator;
  if (!integrator.caustics_reflective || (integrator.filter_closures & FILTER_CLOSURE_GLOSSY)) {
    return "WebGPU Cycles GGX needs reflective caustics and glossy closures enabled";
  }
  if (std::ranges::any_of(host_data(scene.lights), [](const KernelLight &light) {
        return (light.shader_id & SHADER_EXCLUDE_GLOSSY) != 0;
      }))
  {
    return "WebGPU Cycles GGX needs glossy light contributions";
  }
  const int world_shader = scene.data.background.surface_shader;
  if (!uses_stock_light_tree(scene) && scene.lights.data()[0].type != LIGHT_AREA &&
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
    if (const char *error = ggx_transmission_incompatibility(scene, transparent_shadows)) {
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
  bool absorption = false;
};

const char *encode_volumes(const Scene &scene,
                           const size_t grid_budget,
                           std::vector<uint> &shaders,
                           std::vector<VolumeMedium> &media)
{
  const DeviceScene &dscene = scene.dscene;
  if (scene.params.use_bvh_spatial_split ||
      dscene.light_tree_emitters.size() > dscene.lights.size())
  {
    return "WebGPU Cycles volumes need a BVH without spatial splits and analytic lighting";
  }
  uint total_phase_slots = 0;
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
    if (!(flags & SD_HAS_ONLY_VOLUME)) {
      return "WebGPU Cycles volume boundaries need a volume-only material";
    }

    /* The independent analytic-light forward pass must never finish inside
     * a medium. Compare the world-space bounds of the synchronized emitters. */
    BoundBox medium_bounds = object->bounds;
    for (const KernelLight &light : host_data(dscene.lights)) {
      if (light.type == LIGHT_BACKGROUND) {
        continue;
      }
      if (light.type != LIGHT_POINT && light.type != LIGHT_AREA) {
        return "WebGPU Cycles volumes need supported point, area, or world lighting";
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
    uint phase_slots = 0;
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
                                  phase_slots,
                                  can_read_grids) ||
        ((flags & SD_HETEROGENEOUS_VOLUME) && !has_voxels))
    {
      return "WebGPU Cycles needs supported volume closures and grid attribute inputs";
    }
    total_phase_slots += phase_slots;
    if (total_phase_slots > MAX_VOLUME_CLOSURE) {
      return "WebGPU Cycles volume phase closures exceed the supported stack capacity";
    }
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
  if (media.size() != 1 || !media.front().absorption) {
    const auto &integrator = dscene.data.integrator;
    if (!integrator.use_light_tree || dscene.light_tree_nodes.size() == 0 ||
        dscene.data.background.transparent ||
        integrator.max_volume_bounce < 0 || integrator.max_volume_bounce > 65535)
    {
      return "WebGPU Cycles volume scattering needs Light Tree, opaque film, and valid bounce settings";
    }
    for (VolumeMedium &medium : media) {
      medium.words[CYCLES_WEBGPU_VOLUME_INDIRECT_CLAMP] = float_word(integrator.sample_clamp_indirect);
      medium.words[CYCLES_WEBGPU_VOLUME_MAX_BOUNCES] = uint32_t(integrator.max_volume_bounce);
    }
  }
  return nullptr;
}

}  // namespace

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

bool cycles_webgpu_encode_scene(const Scene &scene, CyclesWebGPUScene &out, std::string &reason)
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
  const uint64_t base_words = bvh_words + surface_words + CYCLES_WEBGPU_LIGHT_HEADER_WORDS;
  if (base_words > WebGPURenderPipeline::max_scene_words) {
    reason = "WebGPU Cycles scene exceeds the upload budget";
    return false;
  }
  const size_t extra_words = WebGPURenderPipeline::max_scene_words - size_t(base_words);
  std::vector<uint> volume_shaders;
  std::vector<VolumeMedium> volume_media;
  if (const char *error = volume ? encode_volumes(scene, extra_words, volume_shaders,
                                                volume_media) : nullptr)
  {
    reason = error;
    return false;
  }
  const bool volume_scattering = volume &&
                                 (volume_media.size() != 1 || !volume_media.front().absorption);
  size_t volume_record_words = volume_scattering ?
                                   volume_media.size() * CYCLES_WEBGPU_VOLUME_SCATTER_WORDS :
                                   CYCLES_WEBGPU_ABSORPTION_WORDS;
  for (const VolumeMedium &medium : volume_media) {
    volume_record_words += medium.grids.size();
  }
  /* Live volume sampling uses the stock segment-conditioned Light Tree. */
  const bool light_tree = volume_scattering || uses_stock_light_tree(dscene);
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
  if (!cycles_webgpu_encode_svm_materials(scene,
                                         encoded.svm,
                                         transparent_shadows,
                                         varying_transparency,
                                         encoded.has_ggx,
                                         encoded.has_transmission,
                                         encoded.has_subsurface,
                                         encoded.advanced_query_svm,
                                         volume_shaders,
                                         reason,
                                         extra_words))
  {
    return false;
  }
  const bool transparent = !transparent_shadows.empty();
  const size_t hair_words = (encoded.svm[1] & CYCLES_WEBGPU_SVM_HAIR) ?
                               CYCLES_WEBGPU_SURFACE_HAIR_HEADER_WORDS : 0;
  const bool pixel_wireframe = encoded.svm[1] & CYCLES_WEBGPU_SVM_PIXEL_WIREFRAME;
  const uint unhandled_volume_materials = CYCLES_WEBGPU_SVM_AOV | CYCLES_WEBGPU_SVM_QUERIES |
                                          CYCLES_WEBGPU_SVM_PIXEL_WIREFRAME;
  if (volume &&
      (transparent || hair_words || encoded.has_subsurface || encoded.has_transmission ||
       (encoded.svm[1] & unhandled_volume_materials)))
  {
    reason = "WebGPU Cycles homogeneous volumes need opaque surface shaders and Combined film";
    return false;
  }
  const size_t volume_words = volume ? CYCLES_WEBGPU_SURFACE_ABSORPTION_OFFSETS -
                                          CYCLES_WEBGPU_SURFACE_OFFSETS + volume_record_words : 0;
  if (pixel_wireframe && varying_transparency) {
    reason = "WebGPU Cycles needs ray differentials in varying transparent shadows for pixel Wireframe";
    return false;
  }
  if (pixel_wireframe && encoded.has_subsurface) {
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
    if (lookup_words > extra_words - encoded.svm.size()) {
      reason = "WebGPU Cycles GGX lookup tables exceed the scene upload budget";
      return false;
    }
    if (const char *error = encode_ggx_tables(
            dscene, encoded.has_transmission, transparent, encoded.lighting))
    {
      reason = error;
      return false;
    }
  }
  const size_t used = encoded.svm.size() + encoded.lighting.size() -
                      CYCLES_WEBGPU_LIGHT_HEADER_WORDS + transparent_words + hair_words +
                      volume_words;
  if (used > extra_words) {
    reason = "WebGPU Cycles scene metadata exceeds the upload budget";
    return false;
  }
  const char *lighting_error = nullptr;
  if (light_tree) {
    lighting_error = encode_stock_light_tree(scene, extra_words - used, encoded.lighting);
  }
  else if (multiple_points) {
    lighting_error = encode_point_lighting(dscene, extra_words - used, encoded.lighting);
  }
  else if (area) {
    lighting_error = encode_area_lighting(scene, extra_words - used, encoded.lighting);
  }
  if (lighting_error) {
    reason = lighting_error;
    return false;
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
  surface.words.reserve(size_t(surface_words) + transparent_words + hair_words + volume_words);
  surface.words.resize(volume ? CYCLES_WEBGPU_SURFACE_ABSORPTION_OFFSETS :
                                    (transparent ? CYCLES_WEBGPU_SURFACE_TRANSPARENT_OFFSETS :
                                                   CYCLES_WEBGPU_SURFACE_OFFSETS) + hair_words);
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
  for (const KernelShader &shader : host_data(dscene.shaders)) {
    surface.words.push_back(uint32_t(shader.flags));
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
                                       0);
    surface.words.push_back(shadow_flags);
    surface.words.insert(surface.words.end(), transparent_shadows.begin(), transparent_shadows.end());
  }

  if (volume) {
    const size_t offset = surface.words.size();
    surface.words[CYCLES_WEBGPU_SURFACE_ABSORPTION_OFFSET] = uint32_t(offset);
    if (volume_scattering) {
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

  encoded.bvh = std::move(bvh.words);
  encoded.surface = std::move(surface.words);
  out = std::move(encoded);
  reason.clear();
  return true;
}

CCL_NAMESPACE_END
