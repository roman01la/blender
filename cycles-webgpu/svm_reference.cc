/* SPDX-License-Identifier: Apache-2.0 */

#include "../blender-platform/cycles-webgpu/material_layout.h"
#include "../blender-platform/cycles-webgpu/path_state_layout.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <span>
#include <vector>

#ifdef __aarch64__
#  include <arm_neon.h>
#endif

#define CCL_NAMESPACE_BEGIN namespace ccl {
#define CCL_NAMESPACE_END }
#include "kernel/svm/node_types.h"
#include "kernel/types.h"
#include "util/color.h"
#include "util/types_image.h"

namespace ccl {
/* Provide only the globals required to compile the stock image interpolator.
 * The reference calls ImageInterpolator directly without accessing these arrays. */
struct SvmReferenceGlobals {
  KernelData data;
#define KERNEL_DATA_ARRAY(type, name) const type *name;
#define KERNEL_DATA_ARRAY_WRITABLE(type, name) type *name;
#include "kernel/data_arrays.h"
#undef KERNEL_DATA_ARRAY_WRITABLE
#undef KERNEL_DATA_ARRAY
};
using KernelGlobals = const SvmReferenceGlobals *;
}  // namespace ccl

#define __KERNEL_GPU__
#define kernel_data (kg->data)
#define kernel_data_fetch(name, index) (kg->name[index])
#define kernel_data_array(name) (kg->name)
#include "kernel/camera/projection.h"
#include "kernel/svm/brightness.h"
#include "kernel/svm/camera.h"
#include "kernel/svm/fresnel.h"
#include "kernel/svm/clamp.h"
#include "kernel/svm/gamma.h"
#include "kernel/svm/hsv.h"
#include "kernel/svm/invert.h"
#include "kernel/svm/map_range.h"
#include "kernel/svm/mix.h"
#include "kernel/svm/normal.h"
#include "kernel/svm/ramp.h"
#include "kernel/svm/sepcomb_color.h"
#include "kernel/svm/sepcomb_vector.h"
#include "kernel/svm/vector_rotate.h"
#include "kernel/svm/brick.h"
#include "kernel/svm/checker.h"
#include "kernel/svm/gabor.h"
#include "kernel/svm/gradient.h"
#include "kernel/svm/magic.h"
#include "kernel/svm/mapping.h"
#include "kernel/svm/math.h"
#include "kernel/svm/noisetex.h"
#include "kernel/svm/voronoi.h"
#include "kernel/svm/wave.h"
#include "kernel/svm/white_noise.h"
#include "kernel/svm/vector_transform.h"
#include "kernel/svm/wavelength.h"
#include "kernel/svm/ies.h"
namespace ccl {
/* GPU builds supply kernel_image_interp(). The reference mirrors Nishita's LUT
 * lookup with ImageInterpolator and only calls the analytic sky models. */
template<typename... Args> float4 kernel_image_interp(Args...)
{
  std::abort();
}
}  // namespace ccl
#include "kernel/svm/sky.h"
#include "kernel/util/image.h"
#undef kernel_data_array
#undef kernel_data_fetch
#undef kernel_data
#undef __KERNEL_GPU__

using Float4 = float __attribute__((ext_vector_type(4)));
using ContextWords = unsigned int __attribute__((ext_vector_type(4)));

void cycles_svm_material(unsigned int shader,
                         unsigned int primitive,
                         float u,
                         float v,
                         unsigned int lane,
                         Float4 &base,
                         Float4 &specular,
                         Float4 &layers,
                         Float4 &transmission,
                         Float4 &subsurface,
                         Float4 &emission);

namespace {

using namespace ccl;

struct ImageMaterial {
  int image;
  uint8_t flags = 0;
  bool alpha_output = false;
  unsigned int projection = NODE_IMAGE_PROJ_FLAT;
  bool environment = false;
};

constexpr unsigned int first_image_shader = 9;
constexpr std::array image_materials{
    ImageMaterial{2},
    ImageMaterial{3},
    ImageMaterial{4},
    ImageMaterial{4, NODE_IMAGE_COMPRESS_AS_SRGB},
    ImageMaterial{5, NODE_IMAGE_COMPRESS_AS_SRGB},
    ImageMaterial{6, NODE_IMAGE_COMPRESS_AS_SRGB},
    ImageMaterial{4, NODE_IMAGE_ALPHA_UNASSOCIATE},
    ImageMaterial{4, NODE_IMAGE_COMPRESS_AS_SRGB, true},
    ImageMaterial{2, 0, true},
    ImageMaterial{9, NODE_IMAGE_ALPHA_UNASSOCIATE},
    ImageMaterial{10, 0, true},
    ImageMaterial{12, 0, true},
    ImageMaterial{13},
    ImageMaterial{2, 0, false, NODE_IMAGE_PROJ_SPHERE},
    ImageMaterial{9, 0, true, NODE_IMAGE_PROJ_SPHERE},
    ImageMaterial{3, 0, false, NODE_IMAGE_PROJ_TUBE},
    ImageMaterial{10, 0, true, NODE_IMAGE_PROJ_TUBE},
    ImageMaterial{4, 0, false, NODE_ENVIRONMENT_EQUIRECTANGULAR, true},
    ImageMaterial{4, 0, false, NODE_ENVIRONMENT_MIRROR_BALL, true},
    ImageMaterial{14},
    ImageMaterial{15, NODE_IMAGE_ALPHA_UNASSOCIATE},
    ImageMaterial{16, 0, true},
    ImageMaterial{17},
    ImageMaterial{18},
};
constexpr unsigned int first_principled_shader = first_image_shader + image_materials.size();
constexpr unsigned int first_ggx_shader = first_principled_shader + 4;
constexpr unsigned int first_multiscatter_shader = first_ggx_shader + 6;
constexpr unsigned int first_transmission_shader = first_multiscatter_shader + 4;
constexpr unsigned int first_glass_shader = first_transmission_shader + 6;
constexpr unsigned int first_refraction_shader = first_glass_shader + 4;
constexpr unsigned int first_mix_shader = first_refraction_shader + 2;
constexpr unsigned int procedural_shader = first_mix_shader + 7;
constexpr unsigned int additional_procedural_shader = procedural_shader + 1;
constexpr unsigned int additional_texture_fixtures = 44 + 6 + 20 + 12;
constexpr unsigned int color_fixtures = 25;
constexpr unsigned int first_context_shader = additional_procedural_shader + 17 +
                                              additional_texture_fixtures + color_fixtures;
constexpr std::array geometry_outputs{NODE_GEOM_P, NODE_GEOM_N, NODE_GEOM_Ng, NODE_GEOM_I,
                                       NODE_GEOM_uv};
constexpr unsigned int light_path_outputs = NODE_LP_ray_portal + 1;
constexpr unsigned int first_light_path_shader = first_context_shader + geometry_outputs.size();
constexpr unsigned int first_light_falloff_shader = first_light_path_shader + light_path_outputs;
constexpr unsigned int two_image_shader = first_light_falloff_shader + 3;
constexpr unsigned int first_view_shader = two_image_shader + 1;
constexpr unsigned int first_view_coordinate = first_view_shader + 6;
constexpr std::array view_coordinates{NODE_TEXCO_CAMERA, NODE_TEXCO_WINDOW,
                                      NODE_TEXCO_REFLECTION, NODE_TEXCO_OBJECT_WITH_TRANSFORM,
                                      NODE_TEXCO_OBJECT, NODE_TEXCO_NORMAL};
constexpr unsigned int first_object_info_shader = first_view_coordinate + view_coordinates.size();
constexpr unsigned int first_attribute_shader = first_object_info_shader + NODE_INFO_OB_RANDOM + 1;
constexpr std::array attribute_outputs{NODE_ATTR_OUTPUT_FLOAT3, NODE_ATTR_OUTPUT_FLOAT,
                                       NODE_ATTR_OUTPUT_FLOAT_ALPHA};
constexpr unsigned int first_hair_info_shader = first_attribute_shader + attribute_outputs.size();
constexpr std::array hair_outputs{NODE_INFO_CURVE_IS_STRAND, NODE_INFO_CURVE_THICKNESS,
                                  NODE_INFO_CURVE_TANGENT_NORMAL};
constexpr unsigned int first_point_info_shader = first_hair_info_shader + hair_outputs.size();
constexpr std::array point_outputs{NODE_INFO_POINT_POSITION, NODE_INFO_POINT_RADIUS};
constexpr unsigned int first_particle_info_shader = first_point_info_shader + point_outputs.size();
constexpr unsigned int particle_outputs = NODE_INFO_PAR_ANGULAR_VELOCITY + 1;
constexpr unsigned int first_wavelength_shader = first_particle_info_shader + particle_outputs;
constexpr std::array wavelengths{380.0f, 452.5f, 555.0f, 610.25f, 779.0f};
constexpr unsigned int first_vector_transform_shader = first_wavelength_shader + wavelengths.size();
/* Every vector type for every pair of World, Object and Camera spaces. */
constexpr unsigned int vector_transform_fixtures = 3 * 3 * 3;
constexpr unsigned int first_found_attribute_shader = first_vector_transform_shader +
                                                      vector_transform_fixtures;
/* Packed attribute slots on the two UV triangles: a mode and NodeAttributeType
 * word followed by three float4 corners, as the host converts them. */
struct AttributeRecord {
  unsigned int mode;
  NodeAttributeType type;
  std::array<float4, 3> corners;
  unsigned int flags = 0;
};
const std::array<std::array<AttributeRecord, 2>, 3> attribute_records{{
    {{{1, NODE_ATTR_RGBA, {make_float4(0.1f, 0.2f, 0.3f, 0.4f), make_float4(0.5f, 0.6f, 0.7f, 0.8f),
                           make_float4(0.9f, 0.25f, 0.125f, 0.0625f)}},
      {0, NODE_ATTR_FLOAT, {}}}},
    {{{2, NODE_ATTR_FLOAT, {make_float4(0.625f, 0, 0, 0)}},
      {1, NODE_ATTR_FLOAT2, {make_float4(0.1f, 0.2f, 0, 0), make_float4(0.3f, 0.4f, 0, 0),
                             make_float4(0.5f, 0.75f, 0, 0)}}}},
    {{{1, NODE_ATTR_FLOAT3, {make_float4(0.2f, 0.4f, 0.6f, 0), make_float4(0.8f, 0.1f, 0.3f, 0),
                             make_float4(0.5f, 0.5f, 0.5f, 0)}},
      {2, NODE_ATTR_FLOAT4, {make_float4(0.3f, 0.6f, 0.9f, 0.45f)}}}},
}};
/* Slots 3-6 hold a UV tangent, its sign, smooth object-space normals and
 * triangle positions, with the object transform pending on the first triangle. */
const std::array<std::array<AttributeRecord, 2>, 4> surface_records{{
    {{{1, NODE_ATTR_FLOAT3, {make_float4(0.8f, 0.6f, 0.1f, 0), make_float4(0.7f, 0.7f, 0.2f, 0),
                             make_float4(0.9f, 0.3f, 0.3f, 0)}},
      {1, NODE_ATTR_FLOAT3, {make_float4(0.1f, 0.9f, 0.2f, 0), make_float4(0.3f, 0.8f, 0.1f, 0),
                             make_float4(0.2f, 0.7f, 0.4f, 0)}}}},
    {{{2, NODE_ATTR_FLOAT, {make_float4(1, 0, 0, 0)}},
      {1, NODE_ATTR_FLOAT, {make_float4(-1, 0, 0, 0), make_float4(-1, 0, 0, 0),
                            make_float4(-1, 0, 0, 0)}}}},
    {{{1, NODE_ATTR_FLOAT3, {make_float4(0.1f, 0.2f, 0.9f, 0), make_float4(-0.2f, 0.1f, 0.95f, 0),
                             make_float4(0.3f, -0.1f, 0.85f, 0)}},
      {1, NODE_ATTR_FLOAT3, {make_float4(0.6f, 0, 0.8f, 0), make_float4(0, 0.6f, 0.8f, 0),
                             make_float4(0, 0, 1, 0)}}}},
    {{{2, NODE_ATTR_FLOAT3, {make_float4(0, 0, 0, 0), make_float4(1, 0, 0, 0),
                             make_float4(0, 1, 0, 0)}, 1},
      {2, NODE_ATTR_FLOAT3, {make_float4(0.5f, 0.2f, 0.125f, 0), make_float4(0.9f, 0.3f, 0.125f, 0),
                             make_float4(0.4f, 0.6f, 0.125f, 0)}}}},
}};
constexpr unsigned int tangent_slot = 3;
struct NormalMapFixture {
  NodeNormalMapSpace space;
  bool smooth;
  float strength;
  bool invert_green = false;
  unsigned int primitive = 0;
  bool missing_tangent = false;
  unsigned int surface = 1;
};
const std::array normal_map_fixtures{
    NormalMapFixture{NODE_NORMAL_MAP_TANGENT, true, 1.0f},
    NormalMapFixture{NODE_NORMAL_MAP_TANGENT, false, 0.5f},
    NormalMapFixture{NODE_NORMAL_MAP_TANGENT, true, 2.0f, true, 1},
    NormalMapFixture{NODE_NORMAL_MAP_TANGENT, false, 1.0f, false, 0, false, 2},
    NormalMapFixture{NODE_NORMAL_MAP_TANGENT, true, 1.0f, false, 0, true},
    NormalMapFixture{NODE_NORMAL_MAP_OBJECT, true, 1.0f},
    NormalMapFixture{NODE_NORMAL_MAP_WORLD, true, 0.5f},
    NormalMapFixture{NODE_NORMAL_MAP_BLENDER_OBJECT, false, 1.5f},
    NormalMapFixture{NODE_NORMAL_MAP_BLENDER_WORLD, true, 1.0f, false, 0, false, 2},
};
const float3 normal_map_color = make_float3(0.7f, 0.4f, 0.9f);
constexpr std::array wireframe_sizes{0.01f, 0.4f, 5.0f};
constexpr unsigned int first_normal_map_shader = first_found_attribute_shader +
                                                 3 * attribute_records.size();
constexpr unsigned int first_wireframe_shader = first_normal_map_shader +
                                                normal_map_fixtures.size();
/* IES directions, including both poles, against slot 0 and the empty slot 1. */
const std::array ies_directions{make_float3(0.3f, -0.5f, 0.2f), make_float3(-0.7f, 0.1f, -0.4f),
                                make_float3(0, 0, 1), make_float3(0, 0, -1),
                                make_float3(0.2f, 0.6f, -0.1f)};
constexpr unsigned int first_ies_shader = first_wireframe_shader + wireframe_sizes.size();
constexpr unsigned int first_sky_shader = first_ies_shader + ies_directions.size() + 1;
/* Sky directions above and below the horizon. Nishita adds one at the Sun. */
const std::array sky_directions{make_float3(0.3f, 0.4f, 0.6f), make_float3(0.5f, -0.2f, 0.1f),
                                make_float3(-0.2f, 0.7f, -0.3f)};
constexpr unsigned int sky_fixtures = 3 * 4;
constexpr unsigned int first_tangent_shader = first_sky_shader + sky_fixtures;
const SVMNodeTexSkyPreethamData sky_preetham{
    0.7f, 0.9f, 1.2f, 0.3f, 0.32f,
    {-1.0f, -0.3f, 2.0f, -2.5f, 0.4f, 0, 0, 0, 0},
    {-1.1f, -0.25f, 1.5f, -2.0f, 0.3f, 0, 0, 0, 0},
    {-0.9f, -0.35f, 2.5f, -3.0f, 0.5f, 0, 0, 0, 0}};
const SVMNodeTexSkyPreethamData sky_hosek{
    0.4f, 1.1f, 5.0f, 5.5f, 6.0f,
    {-1.2f, -0.25f, 0.5f, 2.0f, -3.0f, 0.1f, 0.7f, 0.5f, 0.4f},
    {-1.1f, -0.2f, 0.4f, 1.8f, -2.5f, 0.12f, 0.6f, 0.45f, 0.5f},
    {-1.0f, -0.15f, 0.3f, 1.5f, -2.0f, 0.15f, 0.5f, 0.4f, 0.6f}};
const SVMNodeTexSkyNishitaData sky_nishita{0.9f, 0.8f, 0.7f, 1.1f, 1.0f, 0.9f,
                                           0.5f, 1.0f, 0.2f, 2.0f, -0.05f, 3};
float3 sky_direction(const unsigned int i)
{
  if (i < sky_directions.size()) {
    return sky_directions[i];
  }
  /* Slightly off the Sun's center, inside its disc. */
  return spherical_to_direction(sky_nishita.sun_elevation - M_PI_2_F + 0.02f,
                                sky_nishita.sun_rotation - M_PI_2_F);
}
/* LightManager's IES layout: slot offsets, then per slot the horizontal and
 * vertical counts, their angles and intensities by horizontal angle. */
std::vector<float> ies_table()
{
  std::vector<float> table{std::bit_cast<float>(2), std::bit_cast<float>(-1)};
  table.insert(table.end(), {std::bit_cast<float>(5), std::bit_cast<float>(4)});
  for (int h = 0; h < 5; h++) {
    table.push_back(h * M_PI_2_F);
  }
  /* The last angle passes pi so the upward pole interpolates instead of
   * depending on the last bit of acosf(-1) at the table's upper bound. */
  for (int v = 0; v < 4; v++) {
    table.push_back(v == 3 ? M_PI_F * 1.0001f : v * M_PI_F / 3.0f);
  }
  for (int h = 0; h < 5; h++) {
    for (int v = 0; v < 4; v++) {
      table.push_back(10.0f + 3.0f * float((h % 4) * 4 + v) - 0.5f * float(v * v));
    }
  }
  return table;
}
/* Geometry Tangent, radial X, Y and Z, then UV tangents from slot 2 on the
 * first triangle, slot 1 on the second, a missing attribute and slot 0's
 * missing record on the second triangle. */
constexpr unsigned int tangent_fixtures = 8;
constexpr unsigned int first_holdout_shader = first_tangent_shader + tangent_fixtures;
constexpr unsigned int first_emission_shader = first_holdout_shader + 3;
constexpr unsigned int shader_count = first_emission_shader + 6;
constexpr float indirect_clamp = 10.0f;
/* XYZ to linear Rec. 709 rows, as synchronized in KernelFilm. */
const std::array xyz_to_rgb_rows{make_float3(3.2404542f, -1.5371385f, -0.4985314f),
                                 make_float3(-0.9692660f, 1.8760108f, 0.0415560f),
                                 make_float3(0.0556434f, -0.2040259f, 1.0572252f)};
const float3 vector_transform_input = make_float3(0.375f, 0.625f, 0.8125f);
const Transform world_to_camera = make_transform(1, 0, 0, 0.25f,
                                                  0, 1, 0, 0.5f,
                                                  0, 0, 1, 1.25f);
const Transform coordinates_from_object = make_transform(0, 2, 0, 0.25f,
                                                           2, 0, 0, 0.5f,
                                                           0, 0, 2, 1.25f);
const std::array world_to_ndc_rows{make_float4(2, 0, 0, 0),
                                    make_float4(0, 1, 0, 0.5f),
                                    make_float4(0, 0, 1, 0),
                                    make_float4(0, 0, 0.5f, 1)};
const Transform camera_to_world = make_transform(1, 0, 0, -0.25f,
                                                 0, 1, 0, -0.5f,
                                                 0, 0, 1, -1.25f);
const std::array camera_rows{world_to_camera.x, world_to_camera.y, world_to_camera.z,
                               world_to_ndc_rows[0], world_to_ndc_rows[1],
                               world_to_ndc_rows[2], world_to_ndc_rows[3]};
const Transform fixture_object = make_transform(0, 0.5f, 0, 0.125f,
                                                0.5f, 0, 0, 0.0625f,
                                                0, 0, 0.25f, 0.03125f);
const Transform fixture_object_inverse = make_transform(0, 2, 0, -0.125f,
                                                        2, 0, 0, -0.25f,
                                                        0, 0, 4, -0.125f);
const std::array object_rows{fixture_object.x, fixture_object.y, fixture_object.z,
                               fixture_object_inverse.x, fixture_object_inverse.y,
                               fixture_object_inverse.z,
                               make_float4(0.125f, 0.25f, 0.5f, 0.625f)};
constexpr uint material_pass = 3;
constexpr float object_pass = 5.0f;
constexpr float object_random = 0.125f;
constexpr unsigned int header_words = 2;
constexpr unsigned int jump_words = 1 + sizeof(SVMNodeShaderJump) / sizeof(unsigned int);
std::vector<unsigned int> bytecode(header_words + shader_count * jump_words, 0);
using FixtureSurface = std::array<ContextWords, CYCLES_WEBGPU_SVM_PATH + 1>;
std::vector<FixtureSurface> fixture_surfaces;

template<typename T> void push_node(ShaderNodeType opcode, const T &value)
{
  bytecode.push_back(opcode);
  const auto words = std::bit_cast<std::array<unsigned int, sizeof(T) / sizeof(unsigned int)>>(value);
  bytecode.insert(bytecode.end(), words.begin(), words.end());
}

template<typename Build> void push_shader(unsigned int shader, Build build)
{
  const auto jump = header_words + shader * jump_words;
  bytecode[jump] = NODE_SHADER_JUMP;
  bytecode[jump + 1] = bytecode.size() - header_words;
  build();
  bytecode.push_back(NODE_END);
  bytecode[jump + 2] = bytecode.size() - header_words;
  bytecode.push_back(NODE_END);
  bytecode[jump + 3] = bytecode.size() - header_words;
  bytecode.push_back(NODE_END);
}

void push_diffuse(float roughness = 0,
                  SVMStackOffset normal = SVM_STACK_INVALID,
                  SVMStackOffset roughness_offset = SVM_STACK_INVALID,
                  SVMStackOffset mix_weight = SVM_STACK_INVALID)
{
  push_node(NODE_CLOSURE_BSDF,
            SVMNodeClosureBsdf{.closure_type = CLOSURE_BSDF_DIFFUSE_ID,
                               .mix_weight_offset = mix_weight});
  const unsigned int roughness_bits = roughness_offset == SVM_STACK_INVALID ?
                                          std::bit_cast<unsigned int>(roughness) :
                                          SVM_INPUT_STACK_OFFSET_MASK | roughness_offset;
  const SVMNodeDiffuseBsdfData diffuse{.roughness = {roughness_bits}, .normal_offset = normal};
  const auto words = std::bit_cast<std::array<unsigned int, sizeof(diffuse) / sizeof(unsigned int)>>(
      diffuse);
  bytecode.insert(bytecode.end(), words.begin(), words.end());
}

void push_weighted_diffuse(float3 color, SVMStackOffset weight = SVM_STACK_INVALID,
                           float roughness = 0.0f)
{
  push_node(NODE_CLOSURE_SET_WEIGHT, SVMNodeClosureSetWeight{.rgb = color});
  push_diffuse(roughness, SVM_STACK_INVALID, SVM_STACK_INVALID, weight);
}

template<typename Jump, typename Branch>
void push_closure_branch(ShaderNodeType opcode, SVMStackOffset factor, Branch branch)
{
  const size_t start = bytecode.size();
  push_node(opcode, Jump{.jump_offset = 0, .stack_offset = factor});
  const size_t branch_start = bytecode.size();
  branch();
  bytecode[start + 1] = unsigned(bytecode.size() - branch_start);
}

void push_image_sample(int id, uint8_t flags, bool alpha_output = false,
                       unsigned int projection = NODE_IMAGE_PROJ_FLAT, bool environment = false)
{
  push_node(NODE_ATTR,
            SVMNodeAttr{.attr = ATTR_STD_UV,
                        .out_offset = 20,
                        .output_type = NODE_ATTR_OUTPUT_FLOAT3});
  const auto color = SVMStackOffset(alpha_output ? SVM_STACK_INVALID : 30);
  const auto alpha = SVMStackOffset(alpha_output ? 30 : SVM_STACK_INVALID);
  if (environment) {
    push_node(NODE_TEX_ENVIRONMENT,
              SVMNodeTexEnvironment{.id = id,
                                    .projection = NodeEnvironmentProjection(projection),
                                    .flags = flags, .co = 20,
                                    .out_offset = color, .alpha_offset = alpha});
  }
  else {
    push_node(NODE_TEX_IMAGE,
              SVMNodeTexImage{.id = id, .projection = projection, .flags = flags, .co = 20,
                              .out_offset = color, .alpha_offset = alpha});
  }
  if (alpha_output) {
    push_node(NODE_CONVERT, SVMNodeConvert{.convert_type = NODE_CONVERT_FV,
                                           .from_offset = 30,
                                           .to_offset = 30});
  }
}

void push_image(int id, bool varying_roughness = false, uint8_t flags = 0, bool alpha_output = false,
                unsigned int projection = NODE_IMAGE_PROJ_FLAT, bool environment = false)
{
  push_image_sample(id, flags, alpha_output, projection, environment);
  push_node(NODE_CLOSURE_WEIGHT, SVMNodeClosureWeight{.weight_offset = 30});
  push_diffuse(0, SVM_STACK_INVALID, varying_roughness ? 31 : SVM_STACK_INVALID);
}

struct PrincipledMaterial {
  float specular;
  float metallic;
  bool textured = false;
  ClosureType distribution = CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID;
  float roughness = 0.5f;
  float anisotropic = 0.0f;
  float ior = 1.5f;
  float coat_weight = 0.0f;
  std::array<float, 3> base_color{0.33f, -0.2f, 1.2f};
  std::array<float, 3> specular_tint{0.9f, 0.75f, 1.2f};
  float transmission = 0.0f;
  float dispersion_scale = 0.0f;
  bool thin_wall = false;
  std::array<float, 3> emission_color{};
  float emission_strength = 0.0f;
};

void push_principled(const PrincipledMaterial &material)
{
  if (material.textured) {
    push_image_sample(6, NODE_IMAGE_COMPRESS_AS_SRGB);
  }
  push_node(NODE_CLOSURE_BSDF,
            SVMNodeClosureBsdf{.closure_type = CLOSURE_BSDF_PRINCIPLED_ID,
                               .mix_weight_offset = SVM_STACK_INVALID});
  const auto bits = [](float value) { return std::bit_cast<unsigned int>(value); };
  const auto color_bits = [&](const std::array<float, 3> &color) {
    return SVMInputFloat3{{bits(color[0])}, {bits(color[1])}, {bits(color[2])}};
  };
  const SVMNodePrincipledBsdfData data{
      .distribution = material.distribution,
      .ior = {bits(material.ior)},
      .roughness = {bits(material.roughness)},
      .coat_weight = {bits(material.coat_weight)},
      .metallic = {bits(material.metallic)},
      .transmission_weight = {bits(material.transmission)},
      .base_color = material.textured ?
                        SVMInputFloat3{{SVM_INPUT_STACK_OFFSET_MASK | 30u}, {}, {}} :
                        color_bits(material.base_color),
      .alpha = {bits(1.0f)},
      .normal_offset = SVM_STACK_INVALID,
      .tangent_offset = SVM_STACK_INVALID,
      .coat_normal_offset = SVM_STACK_INVALID,
      .specular_tint = color_bits(material.specular_tint),
      .specular_ior_level = {bits(material.specular)},
      .anisotropic = {bits(material.anisotropic)},
      .transmission_dispersion_scale = {bits(material.dispersion_scale)},
      .thin_wall = {.value = material.thin_wall, .offset = SVM_STACK_INVALID},
      .emission_color = color_bits(material.emission_color),
      .emission_strength = {bits(material.emission_strength)},
  };
  const auto words = std::bit_cast<std::array<unsigned int, sizeof(data) / sizeof(unsigned int)>>(
      data);
  bytecode.insert(bytecode.end(), words.begin(), words.end());
}

void push_glass(ClosureType distribution, float roughness, bool textured = false)
{
  const float3 color = make_float3(0.72f, -0.05f, 1.1f);
  const auto bits = [](float value) { return std::bit_cast<unsigned int>(value); };
  if (textured) {
    push_image_sample(6, NODE_IMAGE_COMPRESS_AS_SRGB);
    push_node(NODE_CLOSURE_WEIGHT, SVMNodeClosureWeight{.weight_offset = 30});
  }
  else {
    push_node(NODE_CLOSURE_SET_WEIGHT, SVMNodeClosureSetWeight{.rgb = color});
  }
  push_node(NODE_CLOSURE_BSDF,
            SVMNodeClosureBsdf{.closure_type = distribution,
                               .mix_weight_offset = SVM_STACK_INVALID});
  const SVMNodeGlassBsdfData data{
      .color = textured ? SVMInputFloat3{{SVM_INPUT_STACK_OFFSET_MASK | 30u}, {}, {}} :
                          SVMInputFloat3{{bits(color.x)}, {bits(color.y)}, {bits(color.z)}},
      .roughness = {bits(roughness)},
      .ior = {bits(1.45f)},
      .normal_offset = SVM_STACK_INVALID,
      .tangent_offset = SVM_STACK_INVALID,
  };
  const auto words = std::bit_cast<std::array<unsigned int, sizeof(data) / sizeof(unsigned int)>>(
      data);
  bytecode.insert(bytecode.end(), words.begin(), words.end());
}

template<typename Range> void print_array(const Range &values)
{
  bool first = true;
  std::cout << '[';
  for (const auto &value : values) {
    if (!first) {
      std::cout << ',';
    }
    first = false;
    std::cout << value;
  }
  std::cout << ']';
}

}  // namespace

unsigned int cycles_svm_word(unsigned int index)
{
  return bytecode.at(index);
}

ContextWords cycles_svm_context(unsigned int lane, unsigned int field)
{
  if (field < CYCLES_WEBGPU_SVM_WORLD_TO_CAMERA) {
    return fixture_surfaces.at(lane).at(field);
  }
  if (field >= CYCLES_WEBGPU_SVM_CAMERA_TO_WORLD && field < CYCLES_WEBGPU_SVM_CAMERA_TO_WORLD_END) {
    const std::array rows{camera_to_world.x, camera_to_world.y, camera_to_world.z};
    const float4 row = rows.at(field - CYCLES_WEBGPU_SVM_CAMERA_TO_WORLD);
    return std::bit_cast<ContextWords>(Float4{row.x, row.y, row.z, row.w});
  }
  if (field >= CYCLES_WEBGPU_SVM_OBJECT_INDEX) {
    const std::array values{object_pass, float(material_pass), object_random};
    return {0, 0, 0, std::bit_cast<uint>(values.at(field - CYCLES_WEBGPU_SVM_OBJECT_INDEX))};
  }
  const auto &row = field < CYCLES_WEBGPU_SVM_OBJECT_TRANSFORM ?
                        camera_rows.at(field - CYCLES_WEBGPU_SVM_WORLD_TO_CAMERA) :
                        object_rows.at(field - CYCLES_WEBGPU_SVM_OBJECT_TRANSFORM);
  return std::bit_cast<ContextWords>(Float4{row.x, row.y, row.z, row.w});
}

int main()
{
  push_shader(0, [] {
    push_node(NODE_CLOSURE_SET_WEIGHT,
              ccl::SVMNodeClosureSetWeight{.rgb = ccl::make_float3(0.8f, 0.4f, -0.2f)});
    push_diffuse();
  });
  push_shader(1, [] {
    push_node(NODE_GEOMETRY,
              ccl::SVMNodeGeometry{.geom_type = ccl::NODE_GEOM_N, .out_offset = 130});
    push_node(NODE_VALUE_V,
              ccl::SVMNodeValueV{.out_offset = 127, .value = ccl::make_float3(0.2f, 0.9f, 0.7f)});
    push_node(NODE_VALUE_F, ccl::SVMNodeValueF{.value = 0.0f, .out_offset = 200});
    push_node(NODE_CLOSURE_WEIGHT, ccl::SVMNodeClosureWeight{.weight_offset = 127});
    push_diffuse(0, 130, 200);
  });
  push_shader(2, [] {
    push_node(NODE_CLOSURE_SET_WEIGHT,
              ccl::SVMNodeClosureSetWeight{.rgb = ccl::make_float3(1e-10f, 1e-10f, 1e-10f)});
    push_diffuse();
  });
  push_shader(3, [] {
    push_node(NODE_CLOSURE_SET_WEIGHT,
              ccl::SVMNodeClosureSetWeight{.rgb = ccl::make_float3(0.8f, 0.4f, 0.2f)});
    push_diffuse(0.1f);
  });
  push_shader(4, [] {
    push_node(NODE_CLOSURE_WEIGHT, ccl::SVMNodeClosureWeight{.weight_offset = 17});
    push_diffuse();
  });
  push_shader(5, [] {
    push_node(NODE_CLOSURE_BSDF,
              ccl::SVMNodeClosureBsdf{.closure_type = ccl::CLOSURE_BSDF_TRANSLUCENT_ID,
                                     .mix_weight_offset = ccl::SVM_STACK_INVALID});
  });
  push_shader(6, [] { push_image(11); });
  push_shader(7, [] { push_image(22); });
  push_shader(8, [] { push_image(11, true); });
  for (unsigned int i = 0; i < image_materials.size(); ++i) {
    push_shader(first_image_shader + i, [material = image_materials[i]] {
      push_image(material.image, false, material.flags, material.alpha_output,
                 material.projection, material.environment);
    });
  }
  push_shader(first_principled_shader, [] { push_principled({0.0f, 0.0f}); });
  push_shader(first_principled_shader + 1, [] { push_principled({0.0f, 0.0f, true}); });
  push_shader(first_principled_shader + 2, [] {
    push_principled({.specular = 0.5f,
                     .metallic = 0.0f,
                     .base_color = {0.8f, 0.8f, 0.8f},
                     .specular_tint = {1.0f, 1.0f, 1.0f}});
  });
  push_shader(first_principled_shader + 3, [] {
    push_principled({.specular = 0.5f, .metallic = 0.0f, .coat_weight = 0.2f});
  });
  constexpr auto ggx = CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID;
  push_shader(first_ggx_shader, [=] { push_principled({0.5f, 0.0f, false, ggx}); });
  push_shader(first_ggx_shader + 1, [=] { push_principled({0.0f, 1.0f, false, ggx}); });
  push_shader(first_ggx_shader + 2, [=] { push_principled({0.25f, 0.4f, true, ggx}); });
  push_shader(first_ggx_shader + 3, [=] { push_principled({0.5f, 0.0f, false, ggx, 0.0f}); });
  push_shader(first_ggx_shader + 4, [=] { push_principled({0.5f, 0.0f, false, ggx, 0.5f, 0.5f}); });
  push_shader(first_ggx_shader + 5, [=] {
    push_principled({0.25f, 0.0f, false, ggx, 0.5f, 0.0f, 0.5f});
  });
  push_shader(first_multiscatter_shader, [] {
    push_principled({.specular = 0.5f, .metallic = 1.0f, .roughness = 0.75f});
  });
  push_shader(first_multiscatter_shader + 1, [] { push_principled({0.25f, 0.4f, true}); });
  push_shader(first_multiscatter_shader + 2, [] {
    push_principled({.specular = 0.5f, .metallic = 0.0f, .roughness = 0.0f});
  });
  push_shader(first_multiscatter_shader + 3, [] {
    push_principled({.specular = 0.5f, .metallic = 0.0f, .anisotropic = 0.5f});
  });
  push_shader(first_transmission_shader, [] {
    push_principled({.specular = 0.5f, .metallic = 0.0f, .transmission = 1.0f});
  });
  push_shader(first_transmission_shader + 1, [=] {
    push_principled({.specular = 0.25f,
                     .metallic = 0.0f,
                     .textured = true,
                     .distribution = ggx,
                     .transmission = 0.4f});
  });
  push_shader(first_transmission_shader + 2, [] {
    push_principled({.specular = 0.5f, .metallic = 0.0f, .ior = 1.0f, .transmission = 1.0f});
  });
  push_shader(first_transmission_shader + 3, [] {
    push_principled({.specular = 0.5f, .metallic = 0.2f, .transmission = 1.0f});
  });
  push_shader(first_transmission_shader + 4, [] {
    push_principled({.specular = 0.5f, .metallic = 0.0f, .transmission = 1.0f, .thin_wall = true});
  });
  push_shader(first_transmission_shader + 5, [] {
    push_principled(
        {.specular = 0.5f, .metallic = 0.0f, .transmission = 1.0f, .dispersion_scale = 0.2f});
  });
  push_shader(first_glass_shader, [=] { push_glass(ggx, 0.4f); });
  push_shader(first_glass_shader + 1, [] {
    push_glass(CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID, 0.6f, true);
  });
  push_shader(first_glass_shader + 2, [=] { push_glass(ggx, 0.0f); });
  push_shader(first_glass_shader + 3, [] {
    push_glass(CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID, 0.4f);
  });
  for (const ClosureType type : {CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID,
                                 CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID})
  {
    const bool beckmann = type == CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID;
    push_shader(first_refraction_shader + unsigned(beckmann), [=] {
      push_node(NODE_CLOSURE_SET_WEIGHT,
                SVMNodeClosureSetWeight{.rgb = make_float3(0.2f, 0.5f, 0.7f)});
      push_node(NODE_CLOSURE_BSDF,
                SVMNodeClosureBsdf{.closure_type = type, .mix_weight_offset = SVM_STACK_INVALID});
      const SVMNodeRefractionBsdfData data{
          .roughness = {std::bit_cast<unsigned int>(beckmann ? 0.4f : 0.0f)},
          .ior = {std::bit_cast<unsigned int>(1.45f)},
          .normal_offset = SVM_STACK_INVALID,
      };
      const auto words = std::bit_cast<std::array<unsigned int, sizeof(data) / sizeof(unsigned int)>>(
          data);
      bytecode.insert(bytecode.end(), words.begin(), words.end());
    });
  }
  constexpr float scale = 14.25f;
  const std::array generated{make_float3(-0.21875f, -0.71875f, 0.125f),
                             make_float3(0.5f, 0.25f, 0.375f),
                             make_float3(0.25f, 0.75f, 0.125f)};
  const float3 mix_first = make_float3(0.625f, 0.125f, 0.25f);
  const float3 mix_second = make_float3(0.125f, 0.5f, 0.75f);
  const auto push_mix_weights = [](SVMInputFloat factor, SVMStackOffset parent_weight,
                                   SVMStackOffset first, SVMStackOffset second) {
    push_node(NODE_MIX_CLOSURE,
              SVMNodeMixClosure{.fac = factor,
                                  .in_weight_offset = parent_weight,
                                  .weight1_offset = first,
                                  .weight2_offset = second});
  };
  push_shader(first_mix_shader, [&] {
    push_weighted_diffuse(mix_first);
    push_weighted_diffuse(mix_second);
  });
  push_shader(first_mix_shader + 1, [&] {
    push_mix_weights({std::bit_cast<unsigned int>(0.25f)}, SVM_STACK_INVALID, 5, 6);
    push_weighted_diffuse(mix_first, 5);
    push_weighted_diffuse(mix_second, 6);
  });
  for (unsigned int mode = 0; mode < 4; ++mode) {
    push_shader(first_mix_shader + 2 + mode, [&] {
      if (mode == 0) {
        push_node(NODE_ATTR,
                  SVMNodeAttr{.attr = ATTR_STD_GENERATED,
                              .out_offset = 0,
                              .output_type = NODE_ATTR_OUTPUT_FLOAT3});
        push_node(NODE_CONVERT,
                  SVMNodeConvert{.convert_type = NODE_CONVERT_VF, .from_offset = 0, .to_offset = 3});
      }
      else {
        push_node(NODE_VALUE_F, SVMNodeValueF{.value = mode == 2 ? 1.0f : 0.0f, .out_offset = 3});
      }
      push_mix_weights({SVM_INPUT_STACK_OFFSET_MASK | 3u}, SVM_STACK_INVALID, 5, 6);
      push_closure_branch<SVMNodeJumpIfOne>(
          NODE_JUMP_IF_ONE, 3, [&] { push_weighted_diffuse(mix_first, 5); });
      push_closure_branch<SVMNodeJumpIfZero>(NODE_JUMP_IF_ZERO, 3, [&] {
        push_weighted_diffuse(mix_second, 6, mode == 3 ? 0.75f : 0.0f);
      });
    });
  }
  push_shader(first_mix_shader + 6, [&] {
    push_mix_weights({std::bit_cast<unsigned int>(0.75f)}, SVM_STACK_INVALID, 5, 6);
    push_mix_weights({std::bit_cast<unsigned int>(0.5f)}, 5, 7, 8);
    push_weighted_diffuse(mix_first, 7);
    push_weighted_diffuse(one_float3(), 8);
    push_weighted_diffuse(mix_second, 6);
  });
  push_shader(procedural_shader, [=] {
    push_node(NODE_ATTR,
              SVMNodeAttr{.attr = ATTR_STD_GENERATED,
                          .out_offset = 0,
                          .output_type = NODE_ATTR_OUTPUT_FLOAT3});
    push_node(NODE_TEX_VORONOI,
              SVMNodeTexVoronoi{.dimensions = 3,
                                .feature = NODE_VORONOI_F1,
                                .metric = NODE_VORONOI_EUCLIDEAN,
                                .scale = {std::bit_cast<uint>(scale)},
                                .randomness = {std::bit_cast<uint>(1.0f)},
                                .coord = 0,
                                .distance_offset = SVM_STACK_INVALID,
                                .color_offset = 6,
                                .position_offset = SVM_STACK_INVALID,
                                .w_out_offset = SVM_STACK_INVALID,
                                .radius_offset = SVM_STACK_INVALID});
    push_node(NODE_CLOSURE_WEIGHT, SVMNodeClosureWeight{.weight_offset = 6});
    push_diffuse();
  });
  /* White Noise hashes the coordinates' float words. Use exact binary
   * fractions so contraction during interpolation cannot change the input. */
  constexpr std::array procedural_uvs{std::array{0.125f, 0.375f},
                                      std::array{0.625f, 0.125f},
                                      std::array{0.75f, 0.25f},
                                      std::array{0.0f, 0.0f}};
  using ProceduralSamples = std::array<float3, procedural_uvs.size()>;
  struct ProceduralFixture {
    ProceduralSamples samples;
    bool supported;
  };
  std::vector<ProceduralFixture> procedural_samples;
  const auto linked_fixture = [&](const auto emit,
                                  const auto evaluate,
                                  const bool scalar,
                                  const bool supported = true) {
    push_shader(additional_procedural_shader + procedural_samples.size(), [&] {
      push_node(NODE_ATTR,
                SVMNodeAttr{.attr = ATTR_STD_GENERATED,
                            .out_offset = 0,
                            .output_type = NODE_ATTR_OUTPUT_FLOAT3});
      emit();
      if (scalar) {
        push_node(NODE_CONVERT,
                  SVMNodeConvert{.convert_type = NODE_CONVERT_FV, .from_offset = 4, .to_offset = 6});
      }
      push_node(NODE_CLOSURE_WEIGHT, SVMNodeClosureWeight{.weight_offset = 6});
      push_diffuse();
    });
    ProceduralSamples samples;
    for (size_t i = 0; i < procedural_uvs.size(); i++) {
      const auto [u, v] = procedural_uvs[i];
      float stack[SVM_STACK_SIZE]{};
      stack_store_float3(stack, 0, (1.0f - u - v) * generated[0] + u * generated[1] + v * generated[2]);
      evaluate(stack);
      const float factor = stack[4];
      samples[i] = scalar ? make_float3(factor, factor, factor) : stack_load_float3(stack, 6);
    }
    procedural_samples.push_back({samples, supported});
  };
  const auto procedural = [&](const ShaderNodeType opcode,
                              const auto &node,
                              const auto evaluate,
                              const bool scalar = false,
                              const std::span<const float> table = {},
                              const bool supported = true) {
    linked_fixture(
        [&] {
          push_node(opcode, node);
          for (const float value : table) {
            bytecode.push_back(std::bit_cast<uint>(value));
          }
        },
        [&](float *stack) { evaluate(stack, node); },
        scalar,
        supported);
  };
  const auto param = [](const float value) { return SVMInputFloat{std::bit_cast<uint>(value)}; };
  const auto rgb = [&](const float r, const float g, const float b) {
    return SVMInputFloat3{param(r), param(g), param(b)};
  };
  const SVMInputFloat3 coordinates{{SVM_INPUT_STACK_OFFSET_MASK}, {}, {}};
  procedural(NODE_TEX_BRICK,
             SVMNodeTexBrick{.color1 = rgb(0.8f, 0.2f, 0.1f),
                             .color2 = rgb(0.2f, 0.6f, 0.4f),
                             .mortar = rgb(0.1f, 0.1f, 0.1f),
                             .scale = param(4.0f),
                             .mortar_size = param(0.05f),
                             .brick_width = param(0.5f),
                             .row_height = param(0.25f),
                             .mortar_smooth = param(0.1f),
                             .offset_amount = 0.5f,
                             .squash_amount = 1.25f,
                             .offset_frequency = 3,
                             .squash_frequency = 3,
                             .color_offset = 6,
                             .fac_offset = SVM_STACK_INVALID},
             svm_node_tex_brick);
  procedural(NODE_TEX_CHECKER,
             SVMNodeTexChecker{.color1 = rgb(0.2f, 0.4f, 0.6f),
                               .color2 = rgb(0.8f, 0.6f, 0.4f),
                               .scale = param(5.0f),
                               .color_offset = 6,
                               .fac_offset = SVM_STACK_INVALID},
             svm_node_tex_checker);
  procedural(NODE_TEX_GRADIENT,
             SVMNodeTexGradient{.gradient_type = NODE_BLEND_DIAGONAL,
                                .fac_offset = SVM_STACK_INVALID,
                                .color_offset = 6},
             svm_node_tex_gradient);
  procedural(NODE_TEX_MAGIC,
             SVMNodeTexMagic{.scale = param(4.0f),
                             .distortion = param(0.5f),
                             .depth = 4,
                             .color_offset = 6,
                             .fac_offset = SVM_STACK_INVALID},
             svm_node_tex_magic);
  procedural(NODE_TEX_WAVE,
             SVMNodeTexWave{.wave_type = NODE_WAVE_RINGS,
                            .rings_direction = NODE_WAVE_RINGS_DIRECTION_Z,
                            .profile = NODE_WAVE_PROFILE_TRI,
                            .scale = param(2.0f),
                            .distortion = param(0.8f),
                            .detail = param(2.0f),
                            .dscale = param(1.1f),
                            .droughness = param(0.5f),
                            .phase = param(0.2f),
                            .color_offset = 6,
                            .fac_offset = SVM_STACK_INVALID},
             svm_node_tex_wave);
  procedural(NODE_TEX_WHITE_NOISE,
             SVMNodeTexWhiteNoise{.dimensions = 4,
                                  .vector = coordinates,
                                  .w = param(1.5f),
                                  .value_offset = SVM_STACK_INVALID,
                                  .color_offset = 6},
             svm_node_tex_white_noise);
  procedural(NODE_MATH,
             SVMNodeMath{.math_type = NODE_MATH_MULTIPLY_ADD,
                         .value1 = coordinates.x,
                         .value2 = param(0.4f),
                         .value3 = param(0.5f),
                         .result_offset = 4},
             svm_node_math,
             true);
  procedural(NODE_VECTOR_MATH,
             SVMNodeVectorMath{.math_type = NODE_VECTOR_MATH_ADD,
                               .a = coordinates,
                               .b = rgb(0.5f, 0.1f, 0.6f),
                               .value_offset = SVM_STACK_INVALID,
                               .vector_offset = 6},
             svm_node_vector_math<float3>);
  procedural(NODE_MATH,
             SVMNodeMath{.math_type = NODE_MATH_MODULO,
                         .value1 = coordinates.x,
                         .value2 = param(0.375f),
                         .result_offset = 4},
             svm_node_math,
             true);
  procedural(NODE_MATH,
             SVMNodeMath{.math_type = NODE_MATH_FLOORED_MODULO,
                         .value1 = coordinates.x,
                         .value2 = param(0.375f),
                         .result_offset = 4},
             svm_node_math,
             true);
  procedural(NODE_VECTOR_MATH,
             SVMNodeVectorMath{.math_type = NODE_VECTOR_MATH_MODULO,
                               .a = coordinates,
                               .b = rgb(0.375f, 0.5f, 0.75f),
                               .value_offset = SVM_STACK_INVALID,
                               .vector_offset = 6},
             svm_node_vector_math<float3>);
  procedural(NODE_TEXTURE_MAPPING,
             SVMNodeTextureMapping{.vec_offset = 0,
                                   .out_offset = 6,
                                   .tfm = make_transform(0, 0, 2, 0.25f,
                                                         0, 0, 0, 0.125f,
                                                         -0.5f, 0, 0, 0.75f)},
             svm_node_texture_mapping<float3>);
  procedural(NODE_MIN_MAX,
             SVMNodeMinMax{.vec_offset = 0,
                           .out_offset = 6,
                           .mn = make_float3(0.125f, 0.25f, 0.0f),
                           .mx = make_float3(0.25f, 0.625f, 1.0f)},
             svm_node_min_max);
  const auto noise = [&](uint dimensions, NodeNoiseType type, bool scalar, float detail) {
    procedural(NODE_TEX_NOISE,
               SVMNodeTexNoise{.dimensions = dimensions,
                               .noise_type = type,
                               .normalize = uint(scalar),
                               .w = coordinates.x,
                               .scale = param(1.5f),
                               .detail = param(detail),
                               .roughness = param(0.375f),
                               .lacunarity = param(2.0f),
                               .offset = param(1.0f),
                               .gain = param(1.5f),
                               .distortion = param(0.25f),
                               .value_offset = scalar ? SVMStackOffset(4) : SVM_STACK_INVALID,
                               .color_offset = scalar ? SVM_STACK_INVALID : SVMStackOffset(6)},
               svm_node_tex_noise,
               scalar);
  };
  for (uint dimensions = 1; dimensions <= 4; dimensions++) {
    for (uint type = 0; type <= NODE_NOISE_HETERO_TERRAIN; type++) {
      for (const bool scalar : {true, false}) {
        noise(dimensions, NodeNoiseType(type), scalar, 1.25f);
      }
    }
  }
  /* Cover the fractional tail with and without multiple full octaves. */
  for (const float detail : {0.375f, 6.625f}) {
    for (const bool scalar : {true, false}) {
      noise(3, NODE_NOISE_HYBRID_MULTIFRACTAL, scalar, detail);
    }
  }
  for (auto type : {NODE_GABOR_TYPE_2D, NODE_GABOR_TYPE_3D}) {
    for (int output = 0; output < 3; output++) {
      procedural(NODE_TEX_GABOR,
                 SVMNodeTexGabor{.gabor_type = type,
                                 .orientation_3d = coordinates,
                                 .scale = param(1.25f),
                                 .frequency = param(1.75f),
                                 .anisotropy = param(0.375f),
                                 .orientation_2d = param(-0.25f),
                                 .value_offset = output == 0 ? SVMStackOffset(4) : SVM_STACK_INVALID,
                                 .phase_offset = output == 1 ? SVMStackOffset(4) : SVM_STACK_INVALID,
                                 .intensity_offset = output == 2 ? SVMStackOffset(4) :
                                                                  SVM_STACK_INVALID},
                 svm_node_tex_gabor,
                 true);
    }
  }
  for (uint dimensions = 1; dimensions <= 4; dimensions++) {
    SVMNodeTexVoronoi node{.dimensions = dimensions,
                           .w = coordinates.x,
                           .scale = param(1.5f),
                           .detail = param(1.25f),
                           .roughness = param(0.5f),
                           .lacunarity = param(2.0f),
                           .smoothness = param(0.75f),
                           .exponent = param(1.75f),
                           .randomness = param(0.625f),
                           .normalize = uint8_t(dimensions % 2),
                           .distance_offset = SVM_STACK_INVALID,
                           .color_offset = SVM_STACK_INVALID,
                           .position_offset = SVM_STACK_INVALID,
                           .w_out_offset = SVM_STACK_INVALID,
                           .radius_offset = SVM_STACK_INVALID};
    for (uint feature = 0; feature <= NODE_VORONOI_N_SPHERE_RADIUS; feature++) {
      auto sample = node;
      sample.feature = NodeVoronoiFeature(feature);
      sample.metric = NodeVoronoiDistanceMetric((dimensions + feature) % (NODE_VORONOI_MINKOWSKI + 1));
      const bool radius = feature == NODE_VORONOI_N_SPHERE_RADIUS;
      const bool distance = !radius && (feature == NODE_VORONOI_DISTANCE_TO_EDGE ||
                                        (dimensions + feature) % 2 == 0);
      const bool scalar = radius || distance;
      sample.distance_offset = distance ? 4 : SVM_STACK_INVALID;
      sample.color_offset = scalar ? SVM_STACK_INVALID : 6;
      sample.radius_offset = radius ? 4 : SVM_STACK_INVALID;
      procedural(NODE_TEX_VORONOI, sample, svm_node_tex_voronoi<KERNEL_FEATURE_NODE_VORONOI_EXTRA>,
                 scalar);
    }
    for (auto feature : {NODE_VORONOI_F1, NODE_VORONOI_F2, NODE_VORONOI_SMOOTH_F1}) {
      auto sample = node;
      sample.feature = feature;
      sample.metric = NODE_VORONOI_EUCLIDEAN;
      const bool scalar = dimensions == 1;
      sample.position_offset = scalar ? SVM_STACK_INVALID : 6;
      sample.w_out_offset = scalar ? 4 : SVM_STACK_INVALID;
      procedural(NODE_TEX_VORONOI, sample, svm_node_tex_voronoi<KERNEL_FEATURE_NODE_VORONOI_EXTRA>,
                 scalar);
    }
  }
  for (const auto type : {NODE_MAPPING_TYPE_POINT,
                         NODE_MAPPING_TYPE_TEXTURE,
                         NODE_MAPPING_TYPE_VECTOR,
                         NODE_MAPPING_TYPE_NORMAL})
  {
    procedural(NODE_MAPPING,
               SVMNodeMapping{.mapping_type = type,
                              .vector = coordinates,
                              .location = rgb(0.0625f, 0.125f, -0.125f),
                              .rotation = rgb(-0.25f, 0.5f, 0.125f),
                              .scale = rgb(1.5f, 0.5f, 1.25f),
                              .result_offset = 6},
               svm_node_mapping<float3>);
  }
  procedural(NODE_HSV,
             SVMNodeHSV{.color = coordinates,
                        .hue = param(0.6875f),
                        .sat = param(0.75f),
                        .val = param(1.25f),
                        .fac = param(0.875f),
                        .out_color_offset = 6},
             svm_node_hsv);
  procedural(NODE_GAMMA,
             SVMNodeGamma{.color = coordinates, .gamma = param(2.0f), .out_offset = 6},
             svm_node_gamma);
  procedural(NODE_BRIGHTCONTRAST,
             SVMNodeBrightContrast{.color = coordinates,
                                   .bright = param(0.25f),
                                   .contrast = param(0.125f),
                                   .out_offset = 6},
             svm_node_brightness);
  procedural(NODE_INVERT,
             SVMNodeInvert{.color = coordinates, .fac = param(0.75f), .out_offset = 6},
             svm_node_invert);
  procedural(NODE_CLAMP,
             SVMNodeClamp{.clamp_type = NODE_CLAMP_RANGE,
                          .min = param(0.75f),
                          .max = param(0.25f),
                          .value = coordinates.x,
                          .result_offset = 4},
             svm_node_clamp,
             true);
  procedural(NODE_MIX,
             SVMNodeMix{.mix_type = NODE_MIX_CLAMP,
                        .c1 = coordinates,
                        .result_offset = 6},
             svm_node_mix);
  procedural(NODE_MIX_FLOAT,
             SVMNodeMixFloat{.fac = coordinates.x,
                             .a = param(0.125f),
                             .b = param(0.875f),
                             .use_clamp = 1,
                             .result_offset = 4},
             svm_node_mix_float,
             true);
  for (const auto mode : {NODE_MIX_BLEND, NODE_MIX_OVERLAY, NODE_MIX_HUE, NODE_MIX_EXCLUSION}) {
    procedural(NODE_MIX_COLOR,
               SVMNodeMixColor{.blend_type = mode,
                               .a = rgb(0.25f, 0.75f, 1.125f),
                               .b = coordinates,
                               .fac = param(0.625f),
                               .use_clamp = 1,
                               .use_clamp_result = 1,
                               .result_offset = 6},
               svm_node_mix_color);
  }
  procedural(NODE_MIX_VECTOR,
             SVMNodeMixVector{.a = rgb(-0.25f, 0.5f, 1.25f),
                              .b = coordinates,
                              .fac = param(1.25f),
                              .result_offset = 6},
             svm_node_mix_vector);
  procedural(NODE_MIX_VECTOR_NON_UNIFORM,
             SVMNodeMixVectorNonUniform{.a = rgb(0.25f, 0.625f, 0.75f),
                                        .b = rgb(0.75f, 0.125f, 0.5f),
                                        .fac = coordinates,
                                        .use_clamp = 1,
                                        .result_offset = 6},
             svm_node_mix_vector_non_uniform);
  procedural(NODE_COMBINE_COLOR,
             SVMNodeCombineColor{.color_type = NODE_COMBSEP_COLOR_HSL,
                                 .red = coordinates.x,
                                 .green = param(0.5f),
                                 .blue = param(0.625f),
                                 .color_offset = 6},
             svm_node_combine_color);
  procedural(NODE_SEPARATE_COLOR,
             SVMNodeSeparateColor{.color_type = NODE_COMBSEP_COLOR_HSV,
                                  .color = coordinates,
                                  .red_offset = SVM_STACK_INVALID,
                                  .green_offset = 4,
                                  .blue_offset = SVM_STACK_INVALID},
             svm_node_separate_color,
             true);
  procedural(NODE_SEPARATE_VECTOR,
             SVMNodeSeparateVector{.vector = coordinates,
                                   .vector_index = 2,
                                   .out_offset = 4},
             svm_node_separate_vector<float3>,
             true);
  const std::array vector_components{
      SVMNodeCombineVector{.in = coordinates.x, .vector_index = 0, .out_offset = 6},
      SVMNodeCombineVector{.in = param(0.25f), .vector_index = 1, .out_offset = 6},
      SVMNodeCombineVector{.in = param(0.875f), .vector_index = 2, .out_offset = 6},
  };
  linked_fixture(
      [&] {
        for (const auto &node : vector_components) {
          push_node(NODE_COMBINE_VECTOR, node);
        }
      },
      [&](float *stack) {
        for (const auto &node : vector_components) {
          svm_node_combine_vector<float3>(stack, node);
        }
      },
      false);
  procedural(NODE_MAP_RANGE,
             SVMNodeMapRange{.range_type = NODE_MAP_RANGE_STEPPED,
                             .value = coordinates.x,
                             .from_min = param(-0.25f),
                             .from_max = param(1.25f),
                             .to_min = param(0.125f),
                             .to_max = param(0.75f),
                             .steps = param(3.0f),
                             .result_offset = 4},
             svm_node_map_range,
             true);
  procedural(NODE_VECTOR_MAP_RANGE,
             SVMNodeVectorMapRange{.range_type = NODE_MAP_RANGE_SMOOTHERSTEP,
                                   .value = coordinates,
                                   .from_min = rgb(0.0f, 1.0f, 0.25f),
                                   .from_max = rgb(1.0f, 0.0f, 0.75f),
                                   .to_min = rgb(0.125f, 0.25f, 0.375f),
                                   .to_max = rgb(0.875f, 0.75f, 0.625f),
                                   .result_offset = 6},
             svm_node_vector_map_range);
  procedural(NODE_VECTOR_ROTATE,
             SVMNodeVectorRotate{.rotate_type = NODE_VECTOR_ROTATE_TYPE_AXIS,
                                 .vector = coordinates,
                                 .center = rgb(0.25f, 0.5f, 0.75f),
                                 .axis = rgb(1.0f, 2.0f, 3.0f),
                                 .angle = param(0.25f),
                                 .invert = 1,
                                 .result_offset = 6},
             svm_node_vector_rotate);
  procedural(NODE_NORMAL,
             SVMNodeNormal{.in_normal = coordinates,
                           .out_normal_offset = SVM_STACK_INVALID,
                           .out_dot_offset = 4,
                           .direction_x = 1.0f,
                           .direction_y = 0.5f,
                           .direction_z = 0.25f},
             svm_node_normal,
             true);

  const std::array color_table{0.125f, 0.25f, 0.375f, 0.25f,
                               0.625f, 0.375f, 0.75f, 0.875f,
                               1.0f, 0.75f, 0.5f, 0.375f};
  const std::array curve_table{0.25f, 0.625f, 0.875f};
  const auto append_stock_table = [&]<typename Node>(const ShaderNodeType opcode,
                                                       const Node &node,
                                                       const auto evaluate,
                                                       const bool scalar,
                                                       const std::span<const float> table) {
    std::vector<uint> stock_table;
    for (const float value : table) {
      stock_table.push_back(std::bit_cast<uint>(value));
    }
    ccl::SvmReferenceGlobals globals{};
    globals.svm_nodes = stock_table.data();
    procedural(opcode, node, [&globals, evaluate](float *stack, const Node &data) {
      evaluate(&globals, stack, data, 0);
    }, scalar, table);
  };
  for (const bool interpolate : {false, true}) {
    append_stock_table(NODE_RGB_RAMP,
                       SVMNodeRGBRamp{.table_size = 3,
                                      .fac = coordinates.x,
                                      .interpolate = interpolate,
                                      .color_offset = SVM_STACK_INVALID,
                                      .alpha_offset = 4},
                       svm_node_rgb_ramp,
                       true,
                       color_table);
  }
  append_stock_table(NODE_CURVES,
                     SVMNodeCurves{.color = coordinates,
                                   .fac = param(0.625f),
                                   .min_x = 0.125f,
                                   .max_x = 0.625f,
                                   .table_size = 3,
                                   .extrapolate = 1,
                                   .out_offset = 6},
                     svm_node_curves,
                     false,
                     color_table);
  append_stock_table(NODE_FLOAT_CURVE,
                     SVMNodeFloatCurve{.fac = param(0.75f),
                                       .value_in = coordinates.x,
                                       .min_x = 0.125f,
                                       .max_x = 0.625f,
                                       .table_size = 3,
                                       .extrapolate = 1,
                                       .out_offset = 4},
                     svm_node_curve,
                     true,
                     curve_table);
  for (unsigned int i = 0; i < geometry_outputs.size(); ++i) {
    push_shader(first_context_shader + i, [=] {
      push_node(NODE_GEOMETRY, SVMNodeGeometry{.geom_type = geometry_outputs[i], .out_offset = 3});
      push_node(NODE_CLOSURE_WEIGHT, SVMNodeClosureWeight{.weight_offset = 3});
      push_diffuse(0);
    });
  }
  auto push_scalar_result = [] {
    push_node(NODE_CONVERT, SVMNodeConvert{.convert_type = NODE_CONVERT_FV,
                                           .from_offset = 0,
                                           .to_offset = 3});
    push_node(NODE_CLOSURE_WEIGHT, SVMNodeClosureWeight{.weight_offset = 3});
    push_diffuse(0);
  };
  for (unsigned int i = 0; i < light_path_outputs; ++i) {
    push_shader(first_light_path_shader + i, [=] {
      push_node(NODE_LIGHT_PATH, SVMNodeLightPath{.path_type = NodeLightPath(i), .out_offset = 0});
      push_scalar_result();
    });
  }
  for (unsigned int i = 0; i <= NODE_LIGHT_FALLOFF_CONSTANT; ++i) {
    push_shader(first_light_falloff_shader + i, [=] {
      push_node(NODE_LIGHT_FALLOFF, SVMNodeLightFalloff{
          .falloff_type = NodeLightFalloff(i), .strength = param(0.25f),
          .smooth = param(0.5f), .out_offset = 0});
      push_scalar_result();
    });
  }
  push_shader(two_image_shader, [] {
    push_image_sample(11, 0);
    push_node(NODE_TEX_IMAGE, SVMNodeTexImage{.id = 4,
                                             .projection = NODE_IMAGE_PROJ_FLAT,
                                             .co = 20,
                                             .out_offset = 33,
                                             .alpha_offset = SVM_STACK_INVALID});
    push_node(NODE_VECTOR_MATH, SVMNodeVectorMath{.math_type = NODE_VECTOR_MATH_ADD,
                                                 .a = {{SVM_INPUT_STACK_OFFSET_MASK | 30u}, {}, {}},
                                                 .b = {{SVM_INPUT_STACK_OFFSET_MASK | 33u}, {}, {}},
                                                 .value_offset = SVM_STACK_INVALID,
                                                 .vector_offset = 36});
    push_node(NODE_CLOSURE_WEIGHT, SVMNodeClosureWeight{.weight_offset = 36});
    push_diffuse(0);
  });
  const SVMNodeFresnel fresnel{.ior = param(1.45f),
                                  .normal_offset = SVM_STACK_INVALID, .out_offset = 0};
  const SVMNodeLayerWeight layer_fresnel{.weight_type = NODE_LAYER_WEIGHT_FRESNEL,
                                         .blend = param(0.7f),
                                         .normal_offset = SVM_STACK_INVALID, .out_offset = 0};
  const SVMNodeLayerWeight layer_facing{.weight_type = NODE_LAYER_WEIGHT_FACING,
                                        .blend = param(0.3f),
                                        .normal_offset = SVM_STACK_INVALID, .out_offset = 0};
  push_shader(first_view_shader, [&] {
    push_node(NODE_FRESNEL, fresnel);
    push_scalar_result();
  });
  push_shader(first_view_shader + 1, [&] {
    push_node(NODE_LAYER_WEIGHT, layer_fresnel);
    push_scalar_result();
  });
  push_shader(first_view_shader + 2, [&] {
    push_node(NODE_LAYER_WEIGHT, layer_facing);
    push_scalar_result();
  });
  const std::array camera_nodes{
      SVMNodeCamera{.vector_offset = 0, .zdepth_offset = SVM_STACK_INVALID,
                    .distance_offset = SVM_STACK_INVALID},
      SVMNodeCamera{.vector_offset = SVM_STACK_INVALID, .zdepth_offset = 0,
                    .distance_offset = SVM_STACK_INVALID},
      SVMNodeCamera{.vector_offset = SVM_STACK_INVALID, .zdepth_offset = SVM_STACK_INVALID,
                    .distance_offset = 0},
  };
  for (unsigned int i = 0; i < camera_nodes.size(); ++i) {
    push_shader(first_view_shader + 3 + i, [&] {
      push_node(NODE_CAMERA, camera_nodes[i]);
      if (i == 0) {
        push_node(NODE_CLOSURE_WEIGHT, SVMNodeClosureWeight{.weight_offset = 0});
        push_diffuse(0);
      }
      else {
        push_scalar_result();
      }
    });
  }
  for (unsigned int i = 0; i < view_coordinates.size(); ++i) {
    push_shader(first_view_coordinate + i, [&] {
      push_node(NODE_TEX_COORD, SVMNodeTexCoord{.texco_type = view_coordinates[i], .out_offset = 0});
      if (view_coordinates[i] == NODE_TEXCO_OBJECT_WITH_TRANSFORM) {
        const auto words = std::bit_cast<std::array<uint, sizeof(PackedTransform) / sizeof(uint)>>(
            PackedTransform(coordinates_from_object));
        bytecode.insert(bytecode.end(), words.begin(), words.end());
      }
      push_node(NODE_CLOSURE_WEIGHT, SVMNodeClosureWeight{.weight_offset = 0});
      push_diffuse(0);
    });
  }
  for (unsigned int i = 0; i <= NODE_INFO_OB_RANDOM; ++i) {
    push_shader(first_object_info_shader + i, [&] {
      push_node(NODE_OBJECT_INFO, SVMNodeObjectInfo{.info_type = NodeObjectInfo(i), .out_offset = 0});
      if (i == NODE_INFO_OB_LOCATION || i == NODE_INFO_OB_COLOR) {
        push_node(NODE_CLOSURE_WEIGHT, SVMNodeClosureWeight{.weight_offset = 0});
        push_diffuse(0);
      }
      else {
        push_scalar_result();
      }
    });
  }
  const auto f3 = [](const float3 value) {
    return SVMInputFloat3{{std::bit_cast<uint>(value.x)},
                          {std::bit_cast<uint>(value.y)},
                          {std::bit_cast<uint>(value.z)}};
  };
  const auto push_result = [&](const bool vector) {
    if (vector) {
      push_node(NODE_CLOSURE_WEIGHT, SVMNodeClosureWeight{.weight_offset = 0});
      push_diffuse(0);
    }
    else {
      push_scalar_result();
    }
  };
  const auto attribute_node = [&](const int attr, const NodeAttributeOutputType type) {
    return SVMNodeAttr{.attr = attr,
                       .missing = f3(make_float3(0.25f, 0.5f, 0.875f)),
                       .missing_alpha = {std::bit_cast<uint>(0.375f)},
                       .out_offset = 0,
                       .output_type = type};
  };
  for (unsigned int i = 0; i < attribute_outputs.size(); ++i) {
    push_shader(first_attribute_shader + i, [&] {
      /* The host marks attributes absent on every object as not found. */
      push_node(NODE_ATTR, attribute_node(int(ATTR_STD_NOT_FOUND), attribute_outputs[i]));
      push_result(attribute_outputs[i] == NODE_ATTR_OUTPUT_FLOAT3);
    });
  }
  for (unsigned int i = 0; i < hair_outputs.size(); ++i) {
    push_shader(first_hair_info_shader + i, [&] {
      push_node(NODE_HAIR_INFO, SVMNodeHairInfo{.info_type = hair_outputs[i], .out_offset = 0});
      push_result(hair_outputs[i] == NODE_INFO_CURVE_TANGENT_NORMAL);
    });
  }
  for (unsigned int i = 0; i < point_outputs.size(); ++i) {
    push_shader(first_point_info_shader + i, [&] {
      push_node(NODE_POINT_INFO, SVMNodePointInfo{.info_type = point_outputs[i], .out_offset = 0});
      push_result(point_outputs[i] == NODE_INFO_POINT_POSITION);
    });
  }
  const auto particle_vector = [](const unsigned int i) {
    return i == NODE_INFO_PAR_LOCATION || i == NODE_INFO_PAR_VELOCITY ||
           i == NODE_INFO_PAR_ANGULAR_VELOCITY;
  };
  for (unsigned int i = 0; i < particle_outputs; ++i) {
    push_shader(first_particle_info_shader + i, [&] {
      push_node(NODE_PARTICLE_INFO,
                SVMNodeParticleInfo{.info_type = NodeParticleInfo(i), .out_offset = 0});
      push_result(particle_vector(i));
    });
  }
  for (unsigned int i = 0; i < wavelengths.size(); ++i) {
    push_shader(first_wavelength_shader + i, [&] {
      push_node(NODE_WAVELENGTH,
                SVMNodeWavelength{.wavelength = {std::bit_cast<uint>(wavelengths[i])},
                                  .color_offset = 0});
      push_result(true);
    });
  }
  const auto vector_transform_node = [&](const unsigned int i) {
    return SVMNodeVectorTransform{.transform_type = NodeVectorTransformType(i / 9),
                                  .convert_from = NodeVectorTransformConvertSpace((i / 3) % 3),
                                  .convert_to = NodeVectorTransformConvertSpace(i % 3),
                                  .vector_in = f3(vector_transform_input),
                                  .vector_out_offset = 0};
  };
  for (unsigned int i = 0; i < vector_transform_fixtures; ++i) {
    push_shader(first_vector_transform_shader + i, [&] {
      push_node(NODE_VECTOR_TRANSFORM, vector_transform_node(i));
      push_result(true);
    });
  }
  for (unsigned int slot = 0; slot < attribute_records.size(); ++slot) {
    for (unsigned int i = 0; i < attribute_outputs.size(); ++i) {
      push_shader(first_found_attribute_shader + 3 * slot + i, [&] {
        push_node(NODE_ATTR, attribute_node(int(CYCLES_WEBGPU_SVM_ATTRIBUTE_SLOT | slot),
                                            attribute_outputs[i]));
        push_result(attribute_outputs[i] == NODE_ATTR_OUTPUT_FLOAT3);
      });
    }
  }
  for (unsigned int i = 0; i < normal_map_fixtures.size(); ++i) {
    push_shader(first_normal_map_shader + i, [&] {
      const NormalMapFixture &fixture = normal_map_fixtures[i];
      const auto slot = [](const unsigned int index) {
        return int(CYCLES_WEBGPU_SVM_ATTRIBUTE_SLOT | (tangent_slot + index));
      };
      push_node(NODE_NORMAL_MAP,
                SVMNodeNormalMap{.space = fixture.space,
                                 .invert_green = fixture.invert_green,
                                 .use_original_base = slot(2),
                                 .attr = fixture.missing_tangent ? int(ATTR_STD_NOT_FOUND) : slot(0),
                                 .attr_sign = slot(1),
                                 .color = f3(normal_map_color),
                                 .strength = {std::bit_cast<uint>(fixture.strength)},
                                 .normal_offset = 0});
      push_result(true);
    });
  }
  for (unsigned int i = 0; i < wireframe_sizes.size(); ++i) {
    push_shader(first_wireframe_shader + i, [&] {
      push_node(NODE_WIREFRAME,
                SVMNodeWireframe{.in_size = {std::bit_cast<uint>(wireframe_sizes[i])},
                                 .bump_filter_width = std::bit_cast<float>(
                                     CYCLES_WEBGPU_SVM_ATTRIBUTE_SLOT | (tangent_slot + 3)),
                                 .out_fac_offset = 0});
      push_scalar_result();
    });
  }
  for (unsigned int i = 0; i <= ies_directions.size(); ++i) {
    push_shader(first_ies_shader + i, [&] {
      const float3 direction = ies_directions[i % ies_directions.size()];
      push_node(NODE_VALUE_V, SVMNodeValueV{.value = direction, .out_offset = 3});
      push_node(NODE_IES, SVMNodeIES{.strength = {std::bit_cast<uint>(1.5f)},
                                     .slot = i == ies_directions.size() ? 1u : 0u,
                                     .vector_offset = 3,
                                     .fac_offset = 0});
      push_scalar_result();
    });
  }
  for (unsigned int i = 0; i < sky_fixtures; ++i) {
    push_shader(first_sky_shader + i, [&] {
      const NodeSkyType type = NodeSkyType(i / 4 == 0 ? NODE_SKY_PREETHAM :
                                           i / 4 == 1 ? NODE_SKY_HOSEK :
                                                        NODE_SKY_MULTIPLE_SCATTERING);
      push_node(NODE_VALUE_V, SVMNodeValueV{.value = sky_direction(i % 4), .out_offset = 3});
      push_node(NODE_TEX_SKY, SVMNodeTexSky{.sky_type = type, .dir_offset = 3, .out_offset = 0});
      const auto data = [&](const auto &value) {
        const auto words = std::bit_cast<std::array<uint, sizeof(value) / sizeof(uint)>>(value);
        bytecode.insert(bytecode.end(), words.begin(), words.end());
      };
      if (type == NODE_SKY_PREETHAM) {
        data(sky_preetham);
      }
      else if (type == NODE_SKY_HOSEK) {
        data(sky_hosek);
      }
      else {
        data(sky_nishita);
      }
      push_result(true);
    });
  }
  push_shader(first_tangent_shader, [&] {
    push_node(NODE_GEOMETRY, SVMNodeGeometry{.geom_type = NODE_GEOM_T, .out_offset = 0});
    push_result(true);
  });
  const std::array<int, 4> uv_tangents{int(CYCLES_WEBGPU_SVM_ATTRIBUTE_SLOT | 2),
                                       int(CYCLES_WEBGPU_SVM_ATTRIBUTE_SLOT | 1),
                                       int(ATTR_STD_NOT_FOUND),
                                       int(CYCLES_WEBGPU_SVM_ATTRIBUTE_SLOT | 0)};
  for (unsigned int i = 1; i < tangent_fixtures; ++i) {
    push_shader(first_tangent_shader + i, [&] {
      const bool radial = i <= 3;
      push_node(NODE_TANGENT,
                SVMNodeTangent{.direction_type = radial ? NODE_TANGENT_RADIAL : NODE_TANGENT_UVMAP,
                               .axis = radial ? NodeTangentAxis(i - 1) : NODE_TANGENT_AXIS_X,
                               .attr = radial ? int(ATTR_STD_GENERATED) : uv_tangents[i - 4],
                               .tangent_offset = 0});
      push_result(true);
    });
  }
  /* A lone Holdout, one weighted by a mix input, and Holdout followed by a
   * diffuse closure, which stays on CPU Cycles. */
  for (unsigned int i = 0; i < 3; ++i) {
    push_shader(first_holdout_shader + i, [&] {
      if (i == 1) {
        push_node(NODE_VALUE_F, SVMNodeValueF{.value = 0.5f, .out_offset = 0});
      }
      push_node(NODE_CLOSURE_SET_WEIGHT, SVMNodeClosureSetWeight{.rgb = make_float3(1, 1, 1)});
      push_node(NODE_CLOSURE_HOLDOUT,
                SVMNodeClosureHoldout{.mix_weight_offset = i == 1 ? SVMStackOffset(0) :
                                                                    SVM_STACK_INVALID});
      if (i == 2) {
        push_weighted_diffuse(make_float3(0.5f, 0.5f, 0.5f));
      }
    });
  }
  /* Emission alone, Emission Strength with a mix weight next to a diffuse
   * closure, a zero mix weight, and Emission next to Holdout, which stays on
   * CPU Cycles. */
  const auto emission_weight = [&] {
    push_node(NODE_EMISSION_WEIGHT,
              SVMNodeEmissionWeight{.color = f3(make_float3(1.0f, 0.5f, 0.25f)),
                                    .strength = {std::bit_cast<uint>(3.0f)}});
  };
  push_shader(first_emission_shader, [&] {
    push_node(NODE_CLOSURE_SET_WEIGHT, SVMNodeClosureSetWeight{.rgb = make_float3(0.5f, 0.25f, 2.0f)});
    push_node(NODE_CLOSURE_EMISSION, SVMNodeClosureEmission{.mix_weight_offset = SVM_STACK_INVALID});
  });
  push_shader(first_emission_shader + 1, [&] {
    push_node(NODE_VALUE_F, SVMNodeValueF{.value = 0.25f, .out_offset = 0});
    emission_weight();
    push_node(NODE_CLOSURE_EMISSION, SVMNodeClosureEmission{.mix_weight_offset = 0});
    push_weighted_diffuse(make_float3(0.8f, 0.6f, 0.4f));
  });
  push_shader(first_emission_shader + 2, [&] {
    push_node(NODE_VALUE_F, SVMNodeValueF{.value = 0.0f, .out_offset = 0});
    emission_weight();
    push_node(NODE_CLOSURE_EMISSION, SVMNodeClosureEmission{.mix_weight_offset = 0});
  });
  push_shader(first_emission_shader + 4, [] {
    push_principled({.specular = 0.0f, .metallic = 0.0f, .emission_color = {1.0f, 0.5f, 0.25f},
                     .emission_strength = 2.0f});
  });
  /* Black or zero-strength emission compiles to no closure nodes. */
  push_shader(first_emission_shader + 5, [] {});
  push_shader(first_emission_shader + 3, [&] {
    emission_weight();
    push_node(NODE_CLOSURE_EMISSION, SVMNodeClosureEmission{.mix_weight_offset = SVM_STACK_INVALID});
    push_node(NODE_CLOSURE_SET_WEIGHT, SVMNodeClosureSetWeight{.rgb = make_float3(1, 1, 1)});
    push_node(NODE_CLOSURE_HOLDOUT, SVMNodeClosureHoldout{.mix_weight_offset = SVM_STACK_INVALID});
  });
  bytecode[0] = bytecode.size() - header_words;
  bytecode[1] = shader_count;
  const std::array triangle_uv{0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 2.0f,
                               -0.25f, -0.25f, -0.75f, -1.25f, -0.5f, -0.5f};
  const std::array float_pixels{make_float4(1, 0, 0, 0),
                                make_float4(0, 1, 0, 0.25f),
                                make_float4(0, 0, 1, 0.75f),
                                make_float4(1, 1, 1, 1)};
  const std::array byte_pixels{make_uchar4(10, 128, 224, 0),
                               make_uchar4(11, 151, 92, 64),
                               make_uchar4(212, 244, 29, 192),
                               make_uchar4(75, 9, 189, 255)};
  struct Image {
    int id;
    unsigned int extension;
    unsigned int interpolation;
    bool is_byte = false;
  };
  const std::array images{
      Image{11, EXTENSION_REPEAT, INTERPOLATION_CLOSEST},
      Image{22, EXTENSION_EXTEND, INTERPOLATION_CLOSEST},
      Image{2, EXTENSION_REPEAT, INTERPOLATION_LINEAR},
      Image{3, EXTENSION_EXTEND, INTERPOLATION_LINEAR},
      Image{4, EXTENSION_REPEAT, INTERPOLATION_LINEAR, true},
      Image{5, EXTENSION_EXTEND, INTERPOLATION_LINEAR, true},
      Image{6, EXTENSION_REPEAT, INTERPOLATION_CLOSEST, true},
      Image{9, EXTENSION_MIRROR, INTERPOLATION_LINEAR, true},
      Image{10, EXTENSION_CLIP, INTERPOLATION_LINEAR},
      Image{12, EXTENSION_CLIP, INTERPOLATION_CLOSEST, true},
      Image{13, EXTENSION_MIRROR, INTERPOLATION_CLOSEST},
      Image{14, EXTENSION_REPEAT, INTERPOLATION_CUBIC, true},
      Image{15, EXTENSION_EXTEND, INTERPOLATION_CUBIC},
      Image{16, EXTENSION_CLIP, INTERPOLATION_CUBIC},
      Image{17, EXTENSION_MIRROR, INTERPOLATION_CUBIC, true},
      Image{18, EXTENSION_MIRROR, INTERPOLATION_SMART},
  };
  constexpr unsigned int image_slots = 23;
  constexpr unsigned int image_words = 5;
  const size_t texture_header = bytecode.size();
  bytecode.insert(bytecode.end(), {2, image_slots, 1,
                                   unsigned(attribute_records.size() + surface_records.size()), 2,
                                   0, 0, 0, 0});
  for (const float3 row : xyz_to_rgb_rows) {
    bytecode.insert(bytecode.end(), {std::bit_cast<uint>(row.x),
                                     std::bit_cast<uint>(row.y),
                                     std::bit_cast<uint>(row.z)});
  }
  for (unsigned int i = 0; i < (WAVELENGTH_RESOLUTION + 1) * 3; ++i) {
    bytecode.push_back(std::bit_cast<uint>(cie_color_match[i / 3][i % 3]));
  }
  bytecode.push_back(std::bit_cast<uint>(indirect_clamp));
  for (float uv : triangle_uv) {
    bytecode.push_back(std::bit_cast<unsigned>(uv));
  }
  for (const float3 value : generated) {
    bytecode.insert(bytecode.end(), {std::bit_cast<uint>(value.x),
                                     std::bit_cast<uint>(value.y),
                                     std::bit_cast<uint>(value.z)});
  }
  std::vector<std::array<AttributeRecord, 2>> packed_records(attribute_records.begin(),
                                                             attribute_records.end());
  packed_records.insert(packed_records.end(), surface_records.begin(), surface_records.end());
  for (const auto &slot : packed_records) {
    for (const AttributeRecord &record : slot) {
      bytecode.push_back(record.mode | (unsigned(record.type) << 8) | (record.flags << 16));
      for (const float4 corner : record.corners) {
        bytecode.insert(bytecode.end(), {std::bit_cast<uint>(corner.x), std::bit_cast<uint>(corner.y),
                                         std::bit_cast<uint>(corner.z), std::bit_cast<uint>(corner.w)});
      }
    }
  }
  const size_t image_table = bytecode.size();
  bytecode.resize(image_table + image_words * image_slots);
  const auto float_offset = unsigned(bytecode.size());
  const auto byte_offset = float_offset + 4 * float_pixels.size();
  constexpr unsigned int alpha_plane = 0x80000000u;
  for (const auto &image : images) {
    const bool stores_alpha = std::any_of(image_materials.begin(), image_materials.end(),
                                            [&](const ImageMaterial &material) {
      return material.image == image.id &&
             (material.alpha_output || (material.flags & NODE_IMAGE_ALPHA_UNASSOCIATE));
    });
    const std::array record{2u, 2u, image.extension,
                            image.interpolation | (stores_alpha ? alpha_plane : 0u),
                            image.is_byte ? unsigned(byte_offset) : float_offset};
    std::copy(record.begin(), record.end(), bytecode.begin() + image_table + image_words * image.id);
  }
  const auto push_pixel = [](const float3 rgb) {
    bytecode.insert(bytecode.end(), {std::bit_cast<unsigned>(rgb.x),
                                     std::bit_cast<unsigned>(rgb.y),
                                     std::bit_cast<unsigned>(rgb.z)});
  };
  for (const float4 rgba : float_pixels) {
    push_pixel(make_float3(rgba));
  }
  for (const float4 rgba : float_pixels) {
    bytecode.push_back(std::bit_cast<unsigned int>(rgba.w));
  }
  for (const uchar4 rgba : byte_pixels) {
    push_pixel(color_byte_to_float(rgba));
  }
  for (const uchar4 rgba : byte_pixels) {
    bytecode.push_back(std::bit_cast<unsigned int>(color_uchar4_to_float4(rgba).w));
  }
  const std::vector<float> ies = ies_table();
  bytecode[texture_header + 5] = unsigned(bytecode.size() - texture_header);
  for (const float value : ies) {
    bytecode.push_back(std::bit_cast<unsigned int>(value));
  }

  constexpr std::array expected{
      std::array{0.8f, 0.4f, 0.0f, 1.0f},
      std::array{0.2f, 0.9f, 0.7f, 1.0f},
      std::array{0.0f, 0.0f, 0.0f, 1.0f},
      std::array{0.0f, 0.0f, 0.0f, 2.0f},
      std::array{0.0f, 0.0f, 0.0f, 2.0f},
      std::array{0.0f, 0.0f, 0.0f, 2.0f},
      std::array{0.0f, 0.0f, 0.0f, 2.0f},
      std::array{0.0f, 0.0f, 0.0f, 2.0f},
      std::array{0.0f, 0.0f, 0.0f, 0.0f},
      std::array{1.0f, 0.0f, 0.0f, 1.0f},
      std::array{1.0f, 0.0f, 0.0f, 1.0f},
      std::array{1.0f, 1.0f, 1.0f, 1.0f},
      std::array{0.0f, 1.0f, 0.0f, 1.0f},
      std::array{1.0f, 1.0f, 1.0f, 1.0f},
      std::array{1.0f, 0.0f, 0.0f, 1.0f},
      std::array{0.0f, 0.0f, 1.0f, 1.0f},
      std::array{1.0f, 0.0f, 0.0f, 1.0f},
      /* This sampled texel has zero green, but other texels make roughness vary. */
      std::array{0.0f, 0.0f, 0.0f, 2.0f},
  };
  const std::array surfaces{1u, 2u, 1u, 1u, 1u, 1u, 1u, 3u, 0u,
                            1u, 1u, 1u, 2u, 1u, 1u, 1u, 1u, 1u};
  const std::array shaders{0u, 1u, 2u, 3u, 4u, 5u, shader_count, 0u, 0u,
                           6u, 6u, 6u, 7u, 6u, 7u, 6u, 6u, 8u};
  const std::array u{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                     0.125f, 0.625f, 0.375f, 0.625f, 0.0f, 0.0f, 1.0f, 0.0f, 0.125f};
  const std::array v{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                     0.125f, 0.125f, 0.375f, 0.125f, 0.0f, 0.0f, 0.0f, 1.0f, 0.125f};
  std::vector<unsigned int> surface_words, geometric_words, hit_words, ray_slots, path_words;
  std::vector<float> reference;
  /* Relative GPU tolerance per lane. Transcendental-heavy models widen it. */
  std::vector<float> gpu_tolerances;
  float gpu_tolerance = 3e-6f;
  /* Emission expected on evaluated lanes, which also carry the indirect clamp. */
  float3 expected_emission = zero_float3();
  FixtureSurface fixture{};
  using MaterialVector = std::array<float, 4>;
  const auto check = [&](unsigned int surface, unsigned int shader, unsigned int primitive,
                         float u, float v, const MaterialVector &expected, float tolerance,
                         const MaterialVector &specular = {}, const MaterialVector &layers = {},
                         const MaterialVector &transmission = {}) {
    shader |= SHADER_CAST_SHADOW;
    const auto lane = unsigned(fixture_surfaces.size());
    auto &context = fixture_surfaces.emplace_back(fixture);
    auto &hit = context[CYCLES_WEBGPU_SVM_INTERSECTION];
    hit.y = std::bit_cast<unsigned>(u);
    hit.z = std::bit_cast<unsigned>(v);
    hit.w = surface;
    hit_words.insert(hit_words.end(), {1, primitive, 0, PRIMITIVE_TRIANGLE, hit.x, hit.y, hit.z, 0});
    const auto P = context[CYCLES_WEBGPU_SVM_POSITION];
    const auto N = context[CYCLES_WEBGPU_SVM_NORMAL];
    const auto Ng = context[CYCLES_WEBGPU_SVM_GEOMETRIC];
    const auto direction = context[CYCLES_WEBGPU_SVM_DIRECTION];
    const auto path = context[CYCLES_WEBGPU_SVM_PATH];
    const auto bounces = context[CYCLES_WEBGPU_SVM_BOUNCES];
    surface_words.insert(surface_words.end(), {P.x, P.y, P.z, surface, N.x, N.y, N.z, shader});
    geometric_words.insert(geometric_words.end(), {Ng.x, Ng.y, Ng.z, 0, 0, 0, 0, 0});
    ray_slots.insert(ray_slots.end(), {0, 0, 0, 0, direction.x, direction.y, direction.z,
                                      0, path.x, 0, 0, 0});
    path_words.resize((lane + 1u) * CYCLES_WEBGPU_PATH_WORDS);
    auto *slot = path_words.data() + lane * CYCLES_WEBGPU_PATH_WORDS;
    slot[CYCLES_WEBGPU_PATH_BOUNCE] = bounces.x;
    slot[CYCLES_WEBGPU_PATH_DIFFUSE_BOUNCE] = bounces.y;
    slot[CYCLES_WEBGPU_PATH_GLOSSY_BOUNCE] = bounces.z;
    slot[CYCLES_WEBGPU_PATH_TRANSMISSION_BOUNCE] = bounces.w;
    slot[CYCLES_WEBGPU_PATH_FLAGS] = path.y;
    slot[CYCLES_WEBGPU_PATH_TRANSPARENT_BOUNCE] = path.z;
    std::array<Float4, CYCLES_WEBGPU_MATERIAL_VECTORS> result{};
    if (surface == 1 || surface == 2) {
      cycles_svm_material(
          shader, primitive, u, v, lane, result[0], result[1], result[2], result[3], result[4],
          result[5]);
    }
    else if (surface != 0) {
      result[0].w = 2;
    }
    const bool evaluated = (surface == 1 || surface == 2) && expected[3] != 2.0f;
    const MaterialVector emission = evaluated ? MaterialVector{expected_emission.x,
                                                               expected_emission.y,
                                                               expected_emission.z,
                                                               indirect_clamp} :
                                                MaterialVector{};
    const std::array expected_material{expected, specular, layers, transmission, MaterialVector{},
                                       emission};
    for (int entry = 0; entry < CYCLES_WEBGPU_MATERIAL_VECTORS; ++entry) {
      for (int channel = 0; channel < 4; ++channel) {
        const float value = expected_material[entry][channel];
        if (!(std::abs(result[entry][channel] - value) <= tolerance)) {
          std::cerr << "Unexpected material sample " << surface_words.size() / 8 - 1
                    << " (shader " << (shader & SHADER_MASK) << ")"
                    << " vector " << entry << " channel " << channel
                    << ": " << result[entry][channel] << " expected " << value
                    << " (result " << result[entry][0] << ' ' << result[entry][1] << ' '
                    << result[entry][2] << ' ' << result[entry][3] << ")\n";
          return false;
        }
      }
      reference.insert(reference.end(), expected_material[entry].begin(),
                       expected_material[entry].end());
    }
    gpu_tolerances.push_back(gpu_tolerance);
    return true;
  };
  for (unsigned int i = 0; i < surfaces.size(); ++i) {
    const unsigned int primitive = i >= 13 && i <= 16 ? 1 : 0;
    if (!check(surfaces[i], shaders[i], primitive, u[i], v[i], expected[i], 0.0f)) {
      return 1;
    }
  }
  for (unsigned int i = 0; i < image_materials.size(); ++i) {
    const auto &material = image_materials[i];
    const auto &image = *std::find_if(images.begin(), images.end(), [&](const Image &item) {
      return item.id == material.image;
    });
    KernelImageInfo info{};
    info.data = image.is_byte ? uintptr_t(byte_pixels.data()) : uintptr_t(float_pixels.data());
    info.width = info.height = 2;
    info.interpolation = image.interpolation;
    info.extension = image.extension;
    struct Hit {
      unsigned int primitive;
      float u, v;
    };
    for (const auto hit : {Hit{0, 0.25f, 0.25f}, Hit{0, 0.0f, 0.0f}, Hit{0, 0.0625f, 0.09375f},
                           Hit{0, 0.5f, 0.125f}, Hit{1, 0.0f, 0.0f}, Hit{1, 0.25f, 0.5f}}) {
      const auto coord = [&](unsigned int axis) {
        const unsigned int first = hit.primitive * 6 + axis;
        return (1.0f - hit.u - hit.v) * triangle_uv[first] +
               hit.u * triangle_uv[first + 2] + hit.v * triangle_uv[first + 4];
      };
      const float3 co = make_float3(coord(0), coord(1), 0.0f);
      float2 uv = make_float2(co);
      if (material.environment) {
        const float3 direction = safe_normalize(co);
        uv = material.projection == NODE_ENVIRONMENT_EQUIRECTANGULAR ?
                 direction_to_equirectangular(direction) : direction_to_mirrorball(direction);
      }
      else if (material.projection == NODE_IMAGE_PROJ_SPHERE) {
        uv = map_to_sphere((co - make_float3(0.5f)) * 2.0f);
      }
      else if (material.projection == NODE_IMAGE_PROJ_TUBE) {
        uv = map_to_tube((co - make_float3(0.5f)) * 2.0f);
      }
      float4 rgba = image.is_byte ? ImageInterpolator<uchar4>::interp(info, 2.0f * uv.x, 2.0f * uv.y) :
                                    ImageInterpolator<float4>::interp(info, 2.0f * uv.x, 2.0f * uv.y);
      if ((material.flags & NODE_IMAGE_ALPHA_UNASSOCIATE) && rgba.w != 0.0f && rgba.w != 1.0f) {
        const float alpha = rgba.w;
        rgba /= alpha;
        rgba.w = alpha;
      }
      if (material.flags & NODE_IMAGE_COMPRESS_AS_SRGB) {
        rgba = color_srgb_to_linear_v4(rgba);
      }
      const MaterialVector expected = material.alpha_output ? MaterialVector{rgba.w, rgba.w, rgba.w, 1} :
                                                            MaterialVector{rgba.x, rgba.y, rgba.z, 1};
      const float tolerance = 1.0e-6f * std::max(1.0f, *std::max_element(expected.begin(),
                                                                         expected.end()));
      if (!check(1, first_image_shader + i, hit.primitive, hit.u, hit.v, expected, tolerance)) {
        return 1;
      }
    }
  }

  VoronoiParams voronoi{};
  voronoi.metric = NODE_VORONOI_EUCLIDEAN;
  voronoi.randomness = 1.0f;
  constexpr float voronoi_u = 0.12f, voronoi_v = 0.34f;
  const float3 coord = ((1.0f - voronoi_u - voronoi_v) * generated[0] +
                        voronoi_u * generated[1] + voronoi_v * generated[2]) * scale;
  const float3 procedural_color = voronoi_f1(voronoi, coord).color;
  if (!check(1, procedural_shader, 0, voronoi_u, voronoi_v,
             {procedural_color.x, procedural_color.y, procedural_color.z, 1.0f}, 1.0e-6f))
  {
    return 1;
  }
  for (size_t fixture = 0; fixture < procedural_samples.size(); fixture++) {
    const auto &[samples, supported] = procedural_samples[fixture];
    for (size_t i = 0; i < procedural_uvs.size(); i++) {
      const auto [u, v] = procedural_uvs[i];
      const float3 color = supported ? max(samples[i], zero_float3()) : zero_float3();
      if (!check(1, additional_procedural_shader + fixture, 0, u, v,
                 {color.x, color.y, color.z, supported ? 1.0f : 2.0f}, 1e-5f))
      {
        return 1;
      }
    }
  }

  const float3 principled_color = color_srgb_to_linear_v3(color_byte_to_float(byte_pixels[0]));
  if (!check(1, first_principled_shader, 0, 0, 0, {0.33f, 0.0f, 1.2f, 1.0f}, 0.0f) ||
      !check(1, first_principled_shader + 1, 0, 0.125f, 0.125f,
             {principled_color.x, principled_color.y, principled_color.z, 1.0f}, 1.0e-6f) ||
      !check(1, first_principled_shader + 2, 0, 0, 0,
             {0.8f, 0.8f, 0.8f, CYCLES_WEBGPU_MATERIAL_MULTISCATTER_GGX}, 1.0e-6f,
             {1, 1, 1, 0.25f}, {0, 1.5f, 0.04f, 1}) ||
      !check(1, first_principled_shader + 3, 0, 0, 0, {0, 0, 0, 2}, 0.0f))
  {
    return 1;
  }

  const MaterialVector ggx_color{0.33f, 0, 1.2f, CYCLES_WEBGPU_MATERIAL_GGX};
  const MaterialVector specular{0.9f, 0.75f, 1.2f, 0.25f};
  const float eta_mixed = (1.0f + std::sqrt(0.02f)) / (1.0f - std::sqrt(0.02f));
  const float f0_inside = 1.0f / 18.0f;
  const float eta_inside = (1.0f - std::sqrt(f0_inside)) / (1.0f + std::sqrt(f0_inside));
  if (!check(1, first_ggx_shader, 0, 0, 0, ggx_color, 1.0e-6f, specular,
             {0, 1.5f, 0.04f, 1}) ||
      !check(1, first_ggx_shader + 1, 0, 0, 0, ggx_color, 1.0e-6f, specular,
             {1, 1, 0, 1}) ||
      !check(1, first_ggx_shader + 2, 0, 0.125f, 0.125f,
             {principled_color.x, principled_color.y, principled_color.z, CYCLES_WEBGPU_MATERIAL_GGX},
             1.0e-6f,
             specular, {0.4f, eta_mixed, 0.02f, 1}) ||
      !check(1, first_ggx_shader + 3, 0, 0, 0, {0, 0, 0, 2}, 0.0f) ||
      !check(1, first_ggx_shader + 4, 0, 0, 0, {0, 0, 0, 2}, 0.0f) ||
      !check(1, first_ggx_shader + 5, 0, 0, 0, ggx_color, 1.0e-6f, specular,
             {0, eta_inside, f0_inside, 1}))
  {
    return 1;
  }

  const MaterialVector multiscatter_color{0.33f, 0, 1.2f, CYCLES_WEBGPU_MATERIAL_MULTISCATTER_GGX};
  if (!check(1, first_multiscatter_shader, 0, 0, 0, multiscatter_color, 1.0e-6f,
             {0.9f, 0.75f, 1.2f, 0.5625f}, {1, 1.5f, 0.04f, 1}) ||
      !check(1, first_multiscatter_shader + 1, 0, 0.125f, 0.125f,
             {principled_color.x, principled_color.y, principled_color.z,
              CYCLES_WEBGPU_MATERIAL_MULTISCATTER_GGX},
             1.0e-6f, specular, {0.4f, eta_mixed, 0.02f, 1}) ||
      !check(1, first_multiscatter_shader + 2, 0, 0, 0, {0, 0, 0, 2}, 0.0f) ||
      !check(1, first_multiscatter_shader + 3, 0, 0, 0, {0, 0, 0, 2}, 0.0f))
  {
    return 1;
  }

  const MaterialVector textured_ggx{principled_color.x, principled_color.y,
                                    principled_color.z, CYCLES_WEBGPU_MATERIAL_GGX};
  const MaterialVector textured_multiscatter{principled_color.x, principled_color.y,
                                             principled_color.z,
                                             CYCLES_WEBGPU_MATERIAL_MULTISCATTER_GGX};
  const MaterialVector unsupported{0, 0, 0, 2};
  if (!check(1, first_transmission_shader, 0, 0, 0, multiscatter_color, 1.0e-6f,
             specular, {0, 1.5f, 0.04f, 1}, {1, 1.5f, 0, 0}) ||
      !check(1, first_transmission_shader + 1, 0, 0.125f, 0.125f, textured_ggx, 1.0e-6f,
             specular, {0, eta_mixed, 0.02f, 1}, {0.4f, 1.5f, 0, 0}) ||
      !check(1, first_transmission_shader + 2, 0, 0, 0, unsupported, 0) ||
      !check(1, first_transmission_shader + 3, 0, 0, 0, unsupported, 0) ||
      !check(1, first_transmission_shader + 4, 0, 0, 0, unsupported, 0) ||
      !check(1, first_transmission_shader + 5, 0, 0, 0, unsupported, 0) ||
      !check(1, first_glass_shader, 0, 0, 0, {0.72f, 0, 1.1f, CYCLES_WEBGPU_MATERIAL_GGX}, 1.0e-6f,
             {1, 1, 1, 0.16f}, {0, 1, 0, 1}, {1, 1.45f, 1, 0}) ||
      !check(1, first_glass_shader + 1, 0, 0.125f, 0.125f, textured_multiscatter, 1.0e-6f,
             {1, 1, 1, 0.36f}, {0, 1, 0, 1}, {1, 1.45f, 1, 0}) ||
      !check(1, first_glass_shader + 2, 0, 0, 0, {0.72f, 0, 1.1f, CYCLES_WEBGPU_MATERIAL_GGX}, 1.0e-6f,
             {1, 1, 1, 0}, {0, 1, 0, 1}, {1, 1.45f, CYCLES_WEBGPU_GLASS_GGX, 0}) ||
      !check(1, first_glass_shader + 3, 0, 0, 0, {0.72f, 0, 1.1f, CYCLES_WEBGPU_MATERIAL_GGX}, 1.0e-6f,
             {1, 1, 1, 0.16f}, {0, 1, 0, 1}, {1, 1.45f, CYCLES_WEBGPU_GLASS_BECKMANN, 0}) ||
      !check(1, first_refraction_shader, 0, 0, 0, {0.2f, 0.5f, 0.7f, CYCLES_WEBGPU_MATERIAL_GGX},
             1.0e-6f, {1, 1, 1, 0}, {0, 1, 0, 1}, {1, 1.45f, CYCLES_WEBGPU_REFRACTION_GGX, 0}) ||
      !check(1, first_refraction_shader + 1, 0, 0, 0, {0.2f, 0.5f, 0.7f, CYCLES_WEBGPU_MATERIAL_GGX},
             1.0e-6f, {1, 1, 1, 0.16f}, {0, 1, 0, 1}, {1, 1.45f, CYCLES_WEBGPU_REFRACTION_BECKMANN, 0}))
  {
    return 1;
  }

  const auto check_mix = [&](unsigned int mode, float3 expected, float u = 0, float v = 0) {
    return check(1, first_mix_shader + mode, 0, u, v,
                 {expected.x, expected.y, expected.z, 1}, 1e-6f);
  };
  if (!check_mix(0, mix_first + mix_second) ||
      !check_mix(1, mix(mix_first, mix_second, 0.25f)) ||
      !check_mix(3, mix_first) || !check_mix(4, mix_second) ||
      !check(1, first_mix_shader + 5, 0, 0, 0, unsupported, 0) ||
      !check_mix(6, mix(mix(mix_first, one_float3(), 0.5f), mix_second, 0.75f)))
  {
    return 1;
  }
  for (const auto [u, v] : procedural_uvs) {
    const float3 coord = (1.0f - u - v) * generated[0] + u * generated[1] + v * generated[2];
    if (!check_mix(2, mix(mix_first, mix_second, saturatef(average(coord))), u, v)) {
      return 1;
    }
  }

  const auto pack_xyz = [](float x, float y, float z) {
    return std::bit_cast<ContextWords>(Float4{x, y, z, 0});
  };
  fixture[CYCLES_WEBGPU_SVM_POSITION] = pack_xyz(0.5f, 0.25f, 0.125f);
  fixture[CYCLES_WEBGPU_SVM_NORMAL] = pack_xyz(0, 0.5f, 0.75f);
  fixture[CYCLES_WEBGPU_SVM_GEOMETRIC] = pack_xyz(0.25f, 0, 0.75f);
  fixture[CYCLES_WEBGPU_SVM_DIRECTION] = pack_xyz(-0.125f, -0.25f, -0.5f);
  fixture[CYCLES_WEBGPU_SVM_INTERSECTION].x = std::bit_cast<unsigned int>(2.0f);
  fixture[CYCLES_WEBGPU_SVM_BOUNCES] = {4, 1, 2, 3};
  fixture[CYCLES_WEBGPU_SVM_PATH] = {PATH_RAY_VISIBILITY_GLOSSY | PATH_RAY_VISIBILITY_TRANSMIT,
                                    PATH_RAY_SINGULAR, 5, 0};
  const std::array<MaterialVector, 5> expected_geometry{{
      {0.5f, 0.25f, 0.125f, 1}, {0, 0.5f, 0.75f, 1}, {0.25f, 0, 0.75f, 1},
      {0.125f, 0.25f, 0.5f, 1}, {0.375f, 0.25f, 0, 1}}};
  for (unsigned int i = 0; i < expected_geometry.size(); ++i) {
    if (!check(1, first_context_shader + i, 0, 0.25f, 0.375f, expected_geometry[i], 0)) {
      return 1;
    }
  }
  const std::array<float, light_path_outputs> expected_path{
      0, 0, 0, 1, 1, 0, 1, 0, 1, 2, 4, 1, 2, 5, 3, 0};
  for (unsigned int i = 0; i < expected_path.size(); ++i) {
    const float expected = expected_path[i];
    if (!check(2, first_light_path_shader + i, 0, 0, 0, {expected, expected, expected, 1}, 0)) {
      return 1;
    }
  }
  fixture[CYCLES_WEBGPU_SVM_PATH] = {PATH_RAY_VISIBILITY_CAMERA, PATH_RAY_REFLECT, 0, 0};
  if (!check(1, first_light_path_shader + NODE_LP_camera, 0, 0, 0, {1, 1, 1, 1}, 0) ||
      !check(1, first_light_path_shader + NODE_LP_reflection, 0, 0, 0, {1, 1, 1, 1}, 0))
  {
    return 1;
  }
  for (unsigned int i = 0; i <= NODE_LIGHT_FALLOFF_CONSTANT; ++i) {
    const float expected = float(1u << i) / 4.5f;
    if (!check(1, first_light_falloff_shader + i, 0, 0, 0, {expected, expected, expected, 1}, 1e-6f)) {
      return 1;
    }
  }
  const float3 first_pixel = color_byte_to_float(byte_pixels[0]);
  if (!check(1, two_image_shader, 0, 0.125f, 0.125f,
             {1 + first_pixel.x, first_pixel.y, first_pixel.z, 1}, 1e-6f))
  {
    return 1;
  }
  ShaderData sd;
  sd.P = make_float3(0.5f, 0.25f, 0.125f);
  sd.N = make_float3(0, 0, 1);
  sd.wi = normalize(make_float3(-0.3f, -0.4f, 0.9f));
  fixture[CYCLES_WEBGPU_SVM_POSITION] = pack_xyz(sd.P.x, sd.P.y, sd.P.z);
  fixture[CYCLES_WEBGPU_SVM_NORMAL] = pack_xyz(sd.N.x, sd.N.y, sd.N.z);
  fixture[CYCLES_WEBGPU_SVM_DIRECTION] = pack_xyz(-sd.wi.x, -sd.wi.y, -sd.wi.z);
  float view_stack[SVM_STACK_SIZE];
  const auto check_view_scalar = [&](unsigned int shader, const unsigned int status = 1) {
    const float value = view_stack[0];
    return check(status, shader, 0, 0, 0, {value, value, value, 1}, 1e-6f);
  };
  const auto check_view_vector = [&](unsigned int shader, const float3 value) {
    return check(1, shader, 0, 0, 0, {value.x, value.y, value.z, 1}, 1e-6f);
  };
  for (unsigned int status : {1u, 2u}) {
    sd.runtime_flag = status == 2 ? SR_BACKFACING : 0;
    svm_node_fresnel(&sd, view_stack, fresnel);
    if (!check_view_scalar(first_view_shader, status)) {
      return 1;
    }
    svm_node_layer_weight(&sd, view_stack, layer_fresnel);
    if (!check_view_scalar(first_view_shader + 1, status)) {
      return 1;
    }
    svm_node_layer_weight(&sd, view_stack, layer_facing);
    if (!check_view_scalar(first_view_shader + 2, status)) {
      return 1;
    }
  }
  SvmReferenceGlobals globals{};
  globals.data.cam.worldtocamera = world_to_camera;
  svm_node_camera(&globals, &sd, view_stack, camera_nodes[0]);
  if (!check_view_vector(first_view_shader + 3, stack_load_float3(view_stack, 0))) {
    return 1;
  }
  for (unsigned int i = 1; i < camera_nodes.size(); ++i) {
    svm_node_camera(&globals, &sd, view_stack, camera_nodes[i]);
    if (!check_view_scalar(first_view_shader + 3 + i)) {
      return 1;
    }
  }
  ProjectionTransform world_to_ndc;
  world_to_ndc.x = world_to_ndc_rows[0];
  world_to_ndc.y = world_to_ndc_rows[1];
  world_to_ndc.z = world_to_ndc_rows[2];
  world_to_ndc.w = world_to_ndc_rows[3];
  float3 window = transform_perspective(&world_to_ndc, sd.P);
  window.z = 0.0f;
  if (!check_view_vector(first_view_coordinate, transform_point(&world_to_camera, sd.P)) ||
      !check_view_vector(first_view_coordinate + 1, window) ||
      !check_view_vector(first_view_coordinate + 2, -reflect(sd.wi, sd.N)) ||
      !check_view_vector(first_view_coordinate + 3, transform_point(&coordinates_from_object, sd.P)) ||
      !check_view_vector(first_view_coordinate + 4, transform_point(&fixture_object_inverse, sd.P)) ||
      !check_view_vector(first_view_coordinate + 5,
                         safe_normalize(transform_direction_transposed(&fixture_object, sd.N))) ||
      !check_view_vector(first_object_info_shader + NODE_INFO_OB_LOCATION,
                         make_float3(fixture_object.x.w, fixture_object.y.w, fixture_object.z.w)) ||
      !check_view_vector(first_object_info_shader + NODE_INFO_OB_COLOR, make_float3(0.125f, 0.25f, 0.5f)))
  {
    return 1;
  }
  for (const auto [property, expected] : {
           std::pair{NODE_INFO_OB_ALPHA, 0.625f},
           std::pair{NODE_INFO_OB_INDEX, object_pass},
           std::pair{NODE_INFO_MAT_INDEX, float(material_pass)},
           std::pair{NODE_INFO_OB_RANDOM, object_random}}) {
    view_stack[0] = expected;
    if (!check_view_scalar(first_object_info_shader + property)) {
      return 1;
    }
  }
  /* Stock SVM nodes on a static triangle of a fixture object. */
  std::array<KernelObject, 1> stock_objects{};
  stock_objects[0].tfm = fixture_object;
  stock_objects[0].itfm = fixture_object_inverse;
  globals.objects = stock_objects.data();
  globals.data.cam.cameratoworld = camera_to_world;
  globals.data.film.xyz_to_r = make_float4(xyz_to_rgb_rows[0]);
  globals.data.film.xyz_to_g = make_float4(xyz_to_rgb_rows[1]);
  globals.data.film.xyz_to_b = make_float4(xyz_to_rgb_rows[2]);
  ShaderData stock_sd = sd;
  stock_sd.object = 0;
  stock_sd.prim = 0;
  stock_sd.type = PRIMITIVE_TRIANGLE;
  stock_sd.object_flag = 0;
  /* The fixtures weight a diffuse closure, which closure_sample_weight()
   * clamps to non-negative values and drops below CLOSURE_WEIGHT_CUTOFF. */
  const auto closure_color = [](float3 color) {
    color = max(color, zero_float3());
    return fabsf(average(color)) >= CLOSURE_WEIGHT_CUTOFF ? color : zero_float3();
  };
  const auto check_stock = [&](const unsigned int shader, const bool vector) {
    if (vector) {
      return check_view_vector(shader, closure_color(stack_load_float3(view_stack, 0)));
    }
    return check_view_scalar(shader);
  };
  /* Missing attributes follow svm_node_attr_surface_eval(): the Missing input
   * for vectors, its average for Fac, and Missing Alpha for Alpha. */
  const float3 missing = make_float3(0.25f, 0.5f, 0.875f);
  if (!check_view_vector(first_attribute_shader, missing)) {
    return 1;
  }
  view_stack[0] = average(missing);
  if (!check_stock(first_attribute_shader + 1, false)) {
    return 1;
  }
  view_stack[0] = 0.375f;
  if (!check_stock(first_attribute_shader + 2, false)) {
    return 1;
  }
  /* Curve and point info outputs on a triangle are zero. */
  view_stack[0] = 0.0f;
  if (!check_stock(first_hair_info_shader, false) || !check_stock(first_hair_info_shader + 1, false) ||
      !check_view_vector(first_hair_info_shader + 2, zero_float3()) ||
      !check_view_vector(first_point_info_shader, zero_float3()) ||
      !check_stock(first_point_info_shader + 1, false))
  {
    return 1;
  }
  /* svm_node_particle_info() reading Cycles' zeroed dummy particle. */
  for (unsigned int i = 0; i < particle_outputs; ++i) {
    view_stack[0] = i == NODE_INFO_PAR_RANDOM ? hash_uint2_to_float(0, 0) : 0.0f;
    const bool passed = particle_vector(i) ?
                            check_view_vector(first_particle_info_shader + i, zero_float3()) :
                            check_stock(first_particle_info_shader + i, false);
    if (!passed) {
      return 1;
    }
  }
  for (unsigned int i = 0; i < wavelengths.size(); ++i) {
    svm_node_wavelength(&globals, view_stack,
                        SVMNodeWavelength{.wavelength = {std::bit_cast<uint>(wavelengths[i])},
                                          .color_offset = 0});
    if (!check_stock(first_wavelength_shader + i, true)) {
      return 1;
    }
  }
  for (unsigned int i = 0; i < vector_transform_fixtures; ++i) {
    svm_node_vector_transform(&globals, &stock_sd, view_stack, vector_transform_node(i));
    if (!check_stock(first_vector_transform_shader + i, true)) {
      return 1;
    }
  }
  /* svm_node_attr_surface_eval() and svm_node_attr_store() for the packed
   * slots, interpolated like triangle_interpolate(). */
  const float attribute_u = 0.25f;
  const float attribute_v = 0.375f;
  for (unsigned int slot = 0; slot < attribute_records.size(); ++slot) {
    for (unsigned int primitive = 0; primitive < 2; ++primitive) {
      const AttributeRecord &record = attribute_records[slot][primitive];
      const auto &c = record.corners;
      const float4 f = record.mode == 1 ? (1.0f - attribute_u - attribute_v) * c[0] +
                                              attribute_u * c[1] + attribute_v * c[2] :
                                          c[0];
      for (unsigned int i = 0; i < attribute_outputs.size(); ++i) {
        const NodeAttributeOutputType type = attribute_outputs[i];
        float3 value;
        if (record.mode == 0) {
          value = type == NODE_ATTR_OUTPUT_FLOAT_ALPHA ? make_float3(0.375f) : missing;
        }
        else if (record.type == NODE_ATTR_FLOAT) {
          value = type == NODE_ATTR_OUTPUT_FLOAT_ALPHA ? make_float3(1.0f) : make_float3(f.x, f.x, f.x);
        }
        else if (record.type == NODE_ATTR_FLOAT2) {
          value = type == NODE_ATTR_OUTPUT_FLOAT       ? make_float3(f.x) :
                  type == NODE_ATTR_OUTPUT_FLOAT_ALPHA ? make_float3(1.0f) :
                                                         make_float3(make_float2(f.x, f.y));
        }
        else if (record.type == NODE_ATTR_FLOAT3) {
          value = type == NODE_ATTR_OUTPUT_FLOAT       ? make_float3(average(make_float3(f))) :
                  type == NODE_ATTR_OUTPUT_FLOAT_ALPHA ? make_float3(1.0f) :
                                                         make_float3(f);
        }
        else {
          value = type == NODE_ATTR_OUTPUT_FLOAT       ? make_float3(average(make_float3(f))) :
                  type == NODE_ATTR_OUTPUT_FLOAT_ALPHA ? make_float3(f.w) :
                                                         make_float3(f);
        }
        const unsigned int shader = first_found_attribute_shader + 3 * slot + i;
        const float3 expected = type == NODE_ATTR_OUTPUT_FLOAT3 ? closure_color(value) :
                                                                  make_float3(average(value));
        if (!check(1, shader, primitive, attribute_u, attribute_v,
                   {expected.x, expected.y, expected.z, 1}, 1e-6f))
        {
          return 1;
        }
      }
    }
  }
  /* svm_node_normal_map() */
  {
    const float u = 0.25f;
    const float v = 0.375f;
    const float3 Ng = make_float3(0.25f, 0, 0.75f);
    const auto record_value = [&](const AttributeRecord &record) {
      const auto &c = record.corners;
      return record.mode == 1 ? (1.0f - u - v) * c[0] + u * c[1] + v * c[2] : c[0];
    };
    for (unsigned int i = 0; i < normal_map_fixtures.size(); ++i) {
      const NormalMapFixture &fixture = normal_map_fixtures[i];
      const bool is_backfacing = fixture.surface == 2;
      float3 color = 2.0f * make_float3(normal_map_color.x - 0.5f, normal_map_color.y - 0.5f,
                                        normal_map_color.z - 0.5f);
      if (fixture.invert_green) {
        color.y = -color.y;
      }
      float strength = fixture.strength;
      bool linear = false;
      float3 N;
      if (fixture.space == NODE_NORMAL_MAP_TANGENT) {
        if (fixture.missing_tangent) {
          N = sd.N;
        }
        else {
          const float3 tangent = make_float3(record_value(surface_records[0][fixture.primitive]));
          const float sign = record_value(surface_records[1][fixture.primitive]).x;
          float3 normal;
          if (fixture.smooth) {
            normal = safe_normalize(make_float3(record_value(surface_records[2][fixture.primitive])));
          }
          else {
            normal = is_backfacing ? -Ng : Ng;
            normal = safe_normalize(transform_direction_transposed(&fixture_object, normal));
          }
          color.x *= strength;
          color.y *= strength;
          color.z = mix(1.0f, color.z, saturatef(strength));
          const float3 B = sign * cross(normal, tangent);
          N = safe_normalize(to_global(color, tangent, B, normal));
          N = normalize(transform_direction_transposed(&fixture_object_inverse, N));
          if (is_backfacing) {
            N = -N;
          }
        }
      }
      else {
        linear = true;
        if (fixture.space == NODE_NORMAL_MAP_BLENDER_OBJECT ||
            fixture.space == NODE_NORMAL_MAP_BLENDER_WORLD)
        {
          color.y = -color.y;
          color.z = -color.z;
        }
        N = color;
        if (fixture.space == NODE_NORMAL_MAP_OBJECT ||
            fixture.space == NODE_NORMAL_MAP_BLENDER_OBJECT)
        {
          N = normalize(transform_direction_transposed(&fixture_object_inverse, N));
        }
        else {
          N = safe_normalize(N);
        }
        if (is_backfacing) {
          N = -N;
        }
      }
      if (!(fixture.space == NODE_NORMAL_MAP_TANGENT && fixture.missing_tangent)) {
        if (linear && strength != 1.0f) {
          strength = max(strength, 0.0f);
          N = safe_normalize(sd.N + (N - sd.N) * strength);
        }
        if (is_zero(N) || !isfinite_safe(N)) {
          N = sd.N;
        }
      }
      const float3 expected = closure_color(N);
      const unsigned int shader = (first_normal_map_shader + i) |
                                  (fixture.smooth ? SHADER_SMOOTH_NORMAL : 0);
      if (!check(fixture.surface, shader, fixture.primitive, u, v,
                 {expected.x, expected.y, expected.z, 1}, 1e-6f))
      {
        return 1;
      }
    }
    /* svm_node_ies() with the stock interpolation. */
    globals.ies = ies.data();
    for (unsigned int i = 0; i <= ies_directions.size(); ++i) {
      const float3 direction = ies_directions[i % ies_directions.size()];
      float stack[SVM_STACK_SIZE];
      stack_store_float3(stack, 3, direction);
      svm_node_ies(&globals, nullptr, stack,
                   SVMNodeIES{.strength = {std::bit_cast<uint>(1.5f)},
                              .slot = i == ies_directions.size() ? 1u : 0u,
                              .vector_offset = 3,
                              .fac_offset = 0});
      const float fac = stack[0];
      if (!check(1, first_ies_shader + i, 0, u, v, {fac, fac, fac, 1}, 2e-6f)) {
        return 1;
      }
    }
    /* Stock Preetham and Hosek radiance, and sky_radiance_nishita() with its
     * Linear, Extend LUT sampled by ImageInterpolator. */
    /* Chained expf() and powf() in the sky models differ by a few ulps on GPUs. */
    gpu_tolerance = 2e-5f;
    for (unsigned int i = 0; i < sky_fixtures; ++i) {
      const float3 dir = sky_direction(i % 4);
      float3 rgb;
      if (i / 4 < 2) {
        SVMNodeTexSkyPreethamData data = i / 4 == 0 ? sky_preetham : sky_hosek;
        rgb = i / 4 == 0 ? sky_radiance_preetham(&globals, dir, data.phi, data.theta,
                                                 data.radiance_x, data.radiance_y,
                                                 data.radiance_z, data.config_x, data.config_y,
                                                 data.config_z) :
                           sky_radiance_hosek(&globals, dir, data.phi, data.theta, data.radiance_x,
                                              data.radiance_y, data.radiance_z, data.config_x,
                                              data.config_y, data.config_z);
      }
      else {
        const SVMNodeTexSkyNishitaData &data = sky_nishita;
        float3 xyz = zero_float3();
        const float2 direction = direction_to_spherical(dir);
        const float3 sun_dir = spherical_to_direction(data.sun_elevation - M_PI_2_F,
                                                      data.sun_rotation - M_PI_2_F);
        const float sun_dir_angle = precise_angle(dir, sun_dir);
        const float half_angular = data.angular_diameter * 0.5f;
        const float dir_elevation = M_PI_2_F - direction.x;
        if (sun_dir_angle < half_angular && dir_elevation > data.earth_intersection_angle) {
          const float y = ((dir_elevation - data.sun_elevation) / data.angular_diameter) + 0.5f;
          const float limb = 1.0f - 0.6f * (1.0f - sqrtf(1.0f - sqr(sun_dir_angle / half_angular)));
          xyz = mix(make_float3(0.9f, 0.8f, 0.7f), make_float3(1.1f, 1.0f, 0.9f), y) *
                data.sun_intensity * limb;
        }
        const float x = fractf((-direction.y - M_PI_2_F + data.sun_rotation) * M_1_2PI_F);
        const float y = copysignf(sqrtf(fabsf(dir_elevation) * M_2_PI_F), dir_elevation) * 0.5f +
                        0.5f;
        KernelImageInfo info{};
        info.data = uintptr_t(float_pixels.data());
        info.width = info.height = 2;
        info.interpolation = INTERPOLATION_LINEAR;
        info.extension = EXTENSION_EXTEND;
        xyz += make_float3(ImageInterpolator<float4>::interp(info, 2.0f * x, 2.0f * y));
        rgb = xyz_to_rgb_clamped(&globals, xyz);
      }
      const float3 expected = closure_color(rgb);
      const float tolerance = 2e-6f * std::max(1.0f, reduce_max(expected));
      if (!check(1, first_sky_shader + i, 0, u, v, {expected.x, expected.y, expected.z, 1},
                 tolerance))
      {
        return 1;
      }
    }
    gpu_tolerance = 3e-6f;
    /* svm_node_wireframe() without pixel size. */
    for (unsigned int primitive = 0; primitive < 2; ++primitive) {
      const AttributeRecord &record = surface_records[3][primitive];
      float3 Co[3];
      for (int i = 0; i < 3; i++) {
        Co[i] = make_float3(record.corners[i]);
        if (record.flags) {
          Co[i] = transform_point(&fixture_object, Co[i]);
        }
      }
      for (unsigned int size = 0; size < wireframe_sizes.size(); ++size) {
        float pixelwidth = 0.5f * wireframe_sizes[size];
        pixelwidth *= pixelwidth;
        float f = 0.0f;
        for (int i = 0; i < 3; i++) {
          const int i2 = i ? i - 1 : 2;
          const float3 crs = cross(Co[i] - Co[i2], sd.P - Co[i]);
          if (dot(crs, crs) < dot(Co[i] - Co[i2], Co[i] - Co[i2]) * pixelwidth) {
            f = 1.0f;
            break;
          }
        }
        if (!check(1, first_wireframe_shader + size, primitive, u, v, {f, f, f, 1}, 0.0f)) {
          return 1;
        }
      }
    }
  }
  /* object_normal_transform() and the projection in primitive_tangent() and
   * svm_node_tangent(). */
  const auto surface_tangent = [&](float3 tangent) {
    tangent = normalize(transform_direction_transposed(&fixture_object_inverse, tangent));
    return cross(sd.N, normalize(cross(tangent, sd.N)));
  };
  {
    const float u = 0.25f;
    const float v = 0.375f;
    const float3 g = (1.0f - u - v) * generated[0] + u * generated[1] + v * generated[2];
    const std::array radial{make_float3(-(g.y - 0.5f), (g.x - 0.5f), 0.0f),
                            make_float3(0.0f, -(g.z - 0.5f), (g.y - 0.5f)),
                            make_float3(-(g.z - 0.5f), 0.0f, (g.x - 0.5f)),
                            make_float3(-(g.y - 0.5f), (g.x - 0.5f), 0.0f)};
    for (unsigned int i = 0; i < radial.size(); ++i) {
      const float3 expected = closure_color(surface_tangent(radial[i]));
      if (!check(1, first_tangent_shader + i, 0, u, v, {expected.x, expected.y, expected.z, 1}, 1e-6f)) {
        return 1;
      }
    }
    const auto interpolated = [&](const AttributeRecord &record) {
      const auto &c = record.corners;
      return record.mode == 1 ? (1.0f - u - v) * c[0] + u * c[1] + v * c[2] : c[0];
    };
    const float4 slot2 = interpolated(attribute_records[2][0]);
    const float4 slot1 = interpolated(attribute_records[1][1]);
    const std::array<std::pair<unsigned int, float3>, 4> uv{{
        {0, closure_color(surface_tangent(make_float3(slot2)))},
        {1, closure_color(surface_tangent(make_float3(slot1.x, slot1.y, 0.0f)))},
        {0, zero_float3()},
        {1, zero_float3()},
    }};
    for (unsigned int i = 0; i < uv.size(); ++i) {
      const float3 expected = uv[i].second;
      if (!check(1, first_tangent_shader + 4 + i, uv[i].first, u, v,
                 {expected.x, expected.y, expected.z, 1}, 1e-6f))
      {
        return 1;
      }
    }
  }
  for (unsigned int i = 0; i < 3; ++i) {
    const float holdout = i == 1 ? 0.5f : 1.0f;
    const MaterialVector expected = i == 2 ? MaterialVector{0, 0, 0, 2} :
                                             MaterialVector{holdout, holdout, holdout,
                                                            CYCLES_WEBGPU_MATERIAL_HOLDOUT};
    if (!check(1, first_holdout_shader + i, 0, 0.25f, 0.25f, expected, 0.0f)) {
      return 1;
    }
  }
  /* emission_setup() accumulates closure weight times mix weight. */
  {
    const std::array<std::pair<float3, MaterialVector>, 6> emissions{{
        {make_float3(0.5f, 0.25f, 2.0f), {0, 0, 0, 1}},
        {make_float3(3.0f, 1.5f, 0.75f) * 0.25f, {0.8f, 0.6f, 0.4f, 1}},
        {zero_float3(), {0, 0, 0, 1}},
        {zero_float3(), {0, 0, 0, 2}},
        /* principled_bsdf_emission() on the matte Principled fixture. */
        {make_float3(2.0f, 1.0f, 0.5f), {0.33f, 0.0f, 1.2f, 1}},
        {zero_float3(), {0, 0, 0, 1}},
    }};
    for (unsigned int i = 0; i < emissions.size(); ++i) {
      expected_emission = emissions[i].first;
      if (!check(1, first_emission_shader + i, 0, 0.25f, 0.25f, emissions[i].second, 1e-6f)) {
        return 1;
      }
    }
    expected_emission = zero_float3();
  }
  std::array<uint, CYCLES_WEBGPU_CAMERA_WORDS> camera_words{};
  for (uint i = 0; i < 3; ++i) {
    const auto row = cycles_svm_context(0, CYCLES_WEBGPU_SVM_CAMERA_TO_WORLD + i);
    for (uint channel = 0; channel < 4; ++channel) {
      camera_words[CYCLES_WEBGPU_CAMERA_TO_WORLD_OFFSET + 4 * i + channel] = row[channel];
    }
  }
  for (uint i = 0; i < camera_rows.size(); ++i) {
    const auto row = cycles_svm_context(0, CYCLES_WEBGPU_SVM_WORLD_TO_CAMERA + i);
    for (uint channel = 0; channel < 4; ++channel) {
      camera_words[CYCLES_WEBGPU_CAMERA_SVM_OFFSET + 4 * i + channel] = row[channel];
    }
  }

  std::vector<uint> metadata(CYCLES_WEBGPU_SURFACE_OFFSETS);
  metadata[0] = uint(metadata.size());
  metadata.push_back(0);
  for (unsigned int i = 0; i < object_rows.size(); ++i) {
    if (i == 3) {
      metadata[CYCLES_WEBGPU_SURFACE_OBJECT_SVM_OFFSET] = uint(metadata.size());
    }
    const auto row = cycles_svm_context(0, CYCLES_WEBGPU_SVM_OBJECT_TRANSFORM + i);
    metadata.insert(metadata.end(), {row.x, row.y, row.z, row.w});
  }
  metadata.insert(metadata.end(), {std::bit_cast<uint>(object_pass),
                                   std::bit_cast<uint>(object_random)});
  metadata[CYCLES_WEBGPU_SURFACE_SHADER_PASS_OFFSET] = uint(metadata.size());
  for (uint shader = 0; shader < shader_count; ++shader) {
    metadata.push_back(shader >= first_object_info_shader ? material_pass : shader);
  }

  surface_words.insert(surface_words.end(), geometric_words.begin(), geometric_words.end());

  std::cout << "{\"bytecode\":";
  print_array(bytecode);
  std::cout << ",\"surfaces\":";
  print_array(surface_words);
  std::cout << ",\"hits\":";
  print_array(hit_words);
  std::cout << ",\"raySlots\":";
  print_array(ray_slots);
  std::cout << ",\"pathWords\":";
  print_array(path_words);
  std::cout << ",\"cameraWords\":";
  print_array(camera_words);
  std::cout << ",\"surfaceMetadata\":";
  print_array(metadata);
  std::cout << ",\"materials\":" << std::setprecision(9);
  print_array(reference);
  std::cout << ",\"tolerances\":";
  print_array(gpu_tolerances);
  std::cout << "}\n";
}
