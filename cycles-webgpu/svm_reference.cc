/* SPDX-License-Identifier: Apache-2.0 */

#include "../blender-platform/cycles-webgpu/material_layout.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <iomanip>
#include <iostream>
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
  const KernelImageUDIM *image_texture_udims;
  const KernelImageTexture *image_textures;
  const KernelTileDescriptor *image_texture_tile_descriptors;
  uchar *image_texture_tile_access_state;
  const KernelImageInfo *image_info;
  const uint *svm_nodes;
  const float *lookup_table;
};
using KernelGlobals = const SvmReferenceGlobals *;
}  // namespace ccl

#define __KERNEL_GPU__
#define kernel_data (kg->data)
#define kernel_data_fetch(name, index) (kg->name[index])
#define kernel_data_array(name) (kg->name)
#include "kernel/svm/voronoi.h"
#include "kernel/util/image.h"
#undef kernel_data_array
#undef kernel_data_fetch
#undef kernel_data
#undef __KERNEL_GPU__

using Float4 = float __attribute__((ext_vector_type(4)));

void cycles_svm_material(unsigned int shader,
                         unsigned int primitive,
                         float u,
                         float v,
                         Float4 &base,
                         Float4 &specular,
                         Float4 &layers,
                         Float4 &transmission,
                         Float4 &subsurface);

namespace {

using namespace ccl;

struct ImageMaterial {
  int image;
  uint8_t flags = 0;
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
};
constexpr unsigned int first_principled_shader = first_image_shader + image_materials.size();
constexpr unsigned int first_ggx_shader = first_principled_shader + 4;
constexpr unsigned int first_multiscatter_shader = first_ggx_shader + 6;
constexpr unsigned int first_transmission_shader = first_multiscatter_shader + 4;
constexpr unsigned int first_glass_shader = first_transmission_shader + 6;
constexpr unsigned int procedural_shader = first_glass_shader + 4;
constexpr unsigned int shader_count = procedural_shader + 1;
constexpr unsigned int header_words = 2;
constexpr unsigned int jump_words = 1 + sizeof(SVMNodeShaderJump) / sizeof(unsigned int);
std::vector<unsigned int> bytecode(header_words + shader_count * jump_words, 0);

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
                  SVMStackOffset roughness_offset = SVM_STACK_INVALID)
{
  push_node(NODE_CLOSURE_BSDF,
            SVMNodeClosureBsdf{.closure_type = CLOSURE_BSDF_DIFFUSE_ID,
                               .mix_weight_offset = SVM_STACK_INVALID});
  const unsigned int roughness_bits = roughness_offset == SVM_STACK_INVALID ?
                                          std::bit_cast<unsigned int>(roughness) :
                                          SVM_INPUT_STACK_OFFSET_MASK | roughness_offset;
  const SVMNodeDiffuseBsdfData diffuse{.roughness = {roughness_bits}, .normal_offset = normal};
  const auto words = std::bit_cast<std::array<unsigned int, sizeof(diffuse) / sizeof(unsigned int)>>(
      diffuse);
  bytecode.insert(bytecode.end(), words.begin(), words.end());
}

void push_image_sample(int id, uint8_t flags)
{
  push_node(NODE_ATTR,
            SVMNodeAttr{.attr = ATTR_STD_UV,
                        .out_offset = 20,
                        .output_type = NODE_ATTR_OUTPUT_FLOAT3});
  push_node(NODE_TEX_IMAGE,
            SVMNodeTexImage{.id = id,
                            .projection = NODE_IMAGE_PROJ_FLAT,
                            .flags = flags,
                            .co = 20,
                            .out_offset = 30,
                            .alpha_offset = SVM_STACK_INVALID});
}

void push_image(int id, bool varying_roughness = false, uint8_t flags = 0)
{
  push_image_sample(id, flags);
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
      push_image(material.image, false, material.flags);
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
  constexpr float scale = 14.25f;
  const std::array generated{make_float3(-0.3f, 0.8f, 0.1f),
                             make_float3(0.5f, 0.2f, 0.4f),
                             make_float3(0.3f, 0.8f, 0.1f)};
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
  bytecode[0] = bytecode.size() - header_words;
  bytecode[1] = shader_count;
  const std::array triangle_uv{0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 2.0f,
                               -0.25f, -0.25f, -0.75f, -1.25f, -0.5f, -0.5f};
  const std::array float_pixels{make_float4(1, 0, 0, 1),
                                make_float4(0, 1, 0, 1),
                                make_float4(0, 0, 1, 1),
                                make_float4(1, 1, 1, 1)};
  const std::array byte_pixels{make_uchar4(10, 128, 224, 255),
                               make_uchar4(11, 151, 92, 255),
                               make_uchar4(212, 244, 29, 255),
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
  };
  constexpr unsigned int image_slots = 23;
  constexpr unsigned int image_words = 5;
  bytecode.insert(bytecode.end(), {2, image_slots, 1, 0, 0, 0});
  for (float uv : triangle_uv) {
    bytecode.push_back(std::bit_cast<unsigned>(uv));
  }
  for (const float3 value : generated) {
    bytecode.insert(bytecode.end(), {std::bit_cast<uint>(value.x),
                                     std::bit_cast<uint>(value.y),
                                     std::bit_cast<uint>(value.z)});
  }
  const size_t image_table = bytecode.size();
  bytecode.resize(image_table + image_words * image_slots);
  const auto float_offset = unsigned(bytecode.size());
  const auto byte_offset = float_offset + 3 * float_pixels.size();
  for (const auto &image : images) {
    const std::array record{2u, 2u, image.extension, image.interpolation,
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
  for (const uchar4 rgba : byte_pixels) {
    push_pixel(color_byte_to_float(rgba));
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
  std::vector<unsigned int> surface_words, hit_words;
  std::vector<float> reference;
  using MaterialVector = std::array<float, 4>;
  const auto check = [&](unsigned int surface, unsigned int shader, unsigned int primitive,
                         float u, float v, const MaterialVector &expected, float tolerance,
                         const MaterialVector &specular = {}, const MaterialVector &layers = {},
                         const MaterialVector &transmission = {}) {
    shader |= SHADER_CAST_SHADOW;
    hit_words.insert(hit_words.end(), {1, primitive, 0, PRIMITIVE_TRIANGLE, 0,
                                       std::bit_cast<unsigned>(u), std::bit_cast<unsigned>(v), 0});
    surface_words.insert(surface_words.end(), {0, 0, 0, surface, 0, 0, 0, shader});
    std::array<Float4, CYCLES_WEBGPU_MATERIAL_VECTORS> result{};
    if (surface == 1 || surface == 2) {
      cycles_svm_material(shader, primitive, u, v, result[0], result[1], result[2], result[3], result[4]);
    }
    else if (surface != 0) {
      result[0].w = 2;
    }
    const std::array expected_material{expected, specular, layers, transmission, MaterialVector{}};
    for (int entry = 0; entry < CYCLES_WEBGPU_MATERIAL_VECTORS; ++entry) {
      for (int channel = 0; channel < 4; ++channel) {
        const float value = expected_material[entry][channel];
        if (!(std::abs(result[entry][channel] - value) <= tolerance)) {
          std::cerr << "Unexpected material sample " << surface_words.size() / 8 - 1
                    << " vector " << entry << " channel " << channel
                    << ": " << result[entry][channel] << " expected " << value << '\n';
          return false;
        }
      }
      reference.insert(reference.end(), expected_material[entry].begin(),
                       expected_material[entry].end());
    }
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
    if (material.flags == NODE_IMAGE_ALPHA_UNASSOCIATE) {
      if (!check(1, first_image_shader + i, 0, 0.25f, 0.25f, {0, 0, 0, 2}, 0.0f)) {
        return 1;
      }
      continue;
    }
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
        return 2.0f * ((1.0f - hit.u - hit.v) * triangle_uv[first] +
                       hit.u * triangle_uv[first + 2] + hit.v * triangle_uv[first + 4]);
      };
      float4 rgba = image.is_byte ? ImageInterpolator<uchar4>::interp(info, coord(0), coord(1)) :
                                    ImageInterpolator<float4>::interp(info, coord(0), coord(1));
      if (material.flags == NODE_IMAGE_COMPRESS_AS_SRGB) {
        rgba = color_srgb_to_linear_v4(rgba);
      }
      if (!check(1, first_image_shader + i, hit.primitive, hit.u, hit.v,
                 {rgba.x, rgba.y, rgba.z, 1.0f}, 1.0e-6f)) {
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
      !check(1, first_glass_shader + 2, 0, 0, 0, unsupported, 0) ||
      !check(1, first_glass_shader + 3, 0, 0, 0, unsupported, 0))
  {
    return 1;
  }

  std::cout << "{\"bytecode\":";
  print_array(bytecode);
  std::cout << ",\"surfaces\":";
  print_array(surface_words);
  std::cout << ",\"hits\":";
  print_array(hit_words);
  std::cout << ",\"materials\":" << std::setprecision(9);
  print_array(reference);
  std::cout << "}\n";
}
