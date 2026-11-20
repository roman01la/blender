/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/svm/value.h"
#include "kernel/svm/voronoi.h"

#include "kernel/closure/bsdf_oren_nayar.h"
#include "kernel/closure/bsdf_util.h"
#include "kernel/util/colorspace.h"
#include "util/color.h"
#include "util/types_image.h"

#include <array>
#include <bit>
#include <utility>

CCL_NAMESPACE_BEGIN

/* Two host-supplied words preceding the stock SVM bytecode. Jump offsets in the
 * bytecode remain relative to its first word. */
constexpr uint webgpu_svm_header_words = 2;
constexpr uint webgpu_svm_jump_words = 1 + sizeof(SVMNodeShaderJump) / sizeof(uint);
/* After the bytecode: UV triangle count, image slots, Generated triangle
 * count, and the working space's luminance coefficients. Attribute records
 * precede the image records and their texels. */
constexpr uint webgpu_svm_texture_header_words = 6;
constexpr uint webgpu_svm_triangle_uv_words = 6;
constexpr uint webgpu_svm_triangle_generated_words = 9;
constexpr uint webgpu_svm_image_words = 5;
using WebGPUSvmUv = float __attribute__((ext_vector_type(2)));

template<typename Fetch>
ccl_device_inline WebGPUSvmUv webgpu_svm_triangle_uv(
    Fetch word, const uint texture, const uint primitive, const float u, const float v)
{
  if (primitive >= word(texture)) {
    return {};
  }
  const uint base = texture + webgpu_svm_texture_header_words +
                    primitive * webgpu_svm_triangle_uv_words;
  /* A native two-lane vector avoids the packed i64 ABI for float2. */
  const WebGPUSvmUv f0{__uint_as_float(word(base)), __uint_as_float(word(base + 1))};
  const WebGPUSvmUv f1{__uint_as_float(word(base + 2)), __uint_as_float(word(base + 3))};
  const WebGPUSvmUv f2{__uint_as_float(word(base + 4)), __uint_as_float(word(base + 5))};
  /* Same barycentric interpolation as triangle_interpolate(). */
  return (1.0f - u - v) * f0 + u * f1 + v * f2;
}

template<typename Fetch>
ccl_device_inline float3 webgpu_svm_read_rgb(Fetch word, const uint offset)
{
  return make_float3(__uint_as_float(word(offset)),
                     __uint_as_float(word(offset + 1)),
                     __uint_as_float(word(offset + 2)));
}

template<typename Fetch>
ccl_device_inline float3 webgpu_svm_triangle_generated(
    Fetch word, const uint texture, const uint primitive, const float u, const float v)
{
  if (primitive >= word(texture + 2)) {
    return zero_float3();
  }
  const uint base = texture + webgpu_svm_texture_header_words +
                    word(texture) * webgpu_svm_triangle_uv_words +
                    primitive * webgpu_svm_triangle_generated_words;
  return (1.0f - u - v) * webgpu_svm_read_rgb(word, base) +
         u * webgpu_svm_read_rgb(word, base + 3) +
         v * webgpu_svm_read_rgb(word, base + 6);
}

ccl_device_inline uint webgpu_svm_texel_axis(const float coordinate,
                                             const uint size,
                                             const uint extension,
                                             const uint neighbor)
{
  /* Address int(coordinate) - (coordinate < 0) + neighbor by its unsigned
   * magnitude so SPIR-V to WGSL preserves negative coordinates. */
  const bool negative = coordinate < 0.0f;
  const uint magnitude = uint(fabsf(coordinate)) + (negative ? 1u - neighbor : neighbor);
  if (extension == EXTENSION_REPEAT) {
    const uint remainder = magnitude % size;
    return negative && remainder != 0 ? size - remainder : remainder;
  }
  return negative ? 0 : min(magnitude, size - 1);
}

ccl_device_inline float webgpu_svm_texel_fraction(const float coordinate)
{
  /* Match ImageInterpolator::frac(), including its negative integer endpoint. */
  const uint magnitude = uint(fabsf(coordinate));
  return coordinate < 0.0f ? coordinate + float(magnitude + 1) : coordinate - float(magnitude);
}

template<typename FetchTexel>
ccl_device_inline float3 webgpu_svm_sample_image(FetchTexel texel,
                                                 const uint width,
                                                 const uint height,
                                                 const uint extension,
                                                 const uint interpolation,
                                                 const float2 uv)
{
  const float x = uv.x * float(width);
  const float y = uv.y * float(height);
  if (interpolation == INTERPOLATION_CLOSEST) {
    const uint ix = webgpu_svm_texel_axis(x, width, extension, 0);
    const uint iy = webgpu_svm_texel_axis(y, height, extension, 0);
    return texel(iy * width + ix);
  }

  /* Match ImageInterpolator::interp_linear() with storage-word fetches in
   * place of native texel pointers. Apply the extension to each neighbor. */
  const float sx = x - 0.5f;
  const float sy = y - 0.5f;
  const float tx = webgpu_svm_texel_fraction(sx);
  const float ty = webgpu_svm_texel_fraction(sy);
  const uint ix = webgpu_svm_texel_axis(sx, width, extension, 0);
  const uint nix = webgpu_svm_texel_axis(sx, width, extension, 1);
  const uint iy = webgpu_svm_texel_axis(sy, height, extension, 0);
  const uint niy = webgpu_svm_texel_axis(sy, height, extension, 1);
  return (1.0f - ty) * (1.0f - tx) * texel(iy * width + ix) +
         (1.0f - ty) * tx * texel(iy * width + nix) +
         ty * (1.0f - tx) * texel(niy * width + ix) + ty * tx * texel(niy * width + nix);
}

template<typename Fetch>
ccl_device_inline bool webgpu_svm_image(
    Fetch word, const uint texture, const int image, const float2 uv, float3 &color)
{
  if (uint(image) >= word(texture + 1)) {
    return false;
  }
  const uint table = texture + webgpu_svm_texture_header_words +
                     word(texture) * webgpu_svm_triangle_uv_words +
                     word(texture + 2) * webgpu_svm_triangle_generated_words;
  /* Index the synchronized image slot directly in the SVM loop. */
  const uint base = table + uint(image) * webgpu_svm_image_words;
  const uint width = word(base);
  if (width == 0) {
    return false;
  }
  const uint height = word(base + 1);
  const uint extension = word(base + 2);
  const uint interpolation = word(base + 3);
  const uint start = word(base + 4);
  const auto texel = [&](const uint index) {
    return webgpu_svm_read_rgb(word, start + 3u * index);
  };
  color = webgpu_svm_sample_image(texel, width, height, extension, interpolation, uv);
  return true;
}

template<typename T, typename Fetch, size_t... I>
ccl_device_inline T webgpu_svm_decode(Fetch fetch, const uint offset, std::index_sequence<I...>)
{
  return std::bit_cast<T>(std::array<uint, sizeof...(I)>{fetch(offset + uint(I))...});
}

template<typename T, typename Fetch>
ccl_device_inline bool webgpu_svm_read(Fetch fetch, uint &offset, const uint end, T &node)
{
  static_assert(sizeof(T) % sizeof(uint) == 0);
  constexpr uint words = sizeof(T) / sizeof(uint);
  if (words > end - offset) {
    return false;
  }
  node = webgpu_svm_decode<T>(fetch, offset, std::make_index_sequence<words>());
  offset += words;
  return true;
}

ccl_device_inline bool webgpu_svm_finite(const float value)
{
  return (std::bit_cast<uint>(value) & 0x7fffffffu) < 0x7f800000u;
}

/* Two vectors of bitsets avoid private memory memset when initializing the
 * stack bookkeeping in the shader. The float stack itself is written lazily. */
using WebGPUSvmMask = uint __attribute__((ext_vector_type(4)));
struct WebGPUSvmStackUsage {
  WebGPUSvmMask low{};
  WebGPUSvmMask high{};
};

ccl_device_inline bool webgpu_svm_stack_has(const WebGPUSvmStackUsage &written, const uint offset)
{
  const WebGPUSvmMask mask = offset < 128 ? written.low : written.high;
  return offset < SVM_STACK_SIZE && (mask[(offset / 32) % 4] & (1u << (offset % 32))) != 0;
}

ccl_device_inline bool webgpu_svm_stack_has3(const WebGPUSvmStackUsage &written, const uint offset)
{
  return webgpu_svm_stack_has(written, offset) && webgpu_svm_stack_has(written, offset + 1) &&
         webgpu_svm_stack_has(written, offset + 2);
}

ccl_device_inline void webgpu_svm_stack_mark(WebGPUSvmStackUsage &written,
                                             const uint offset,
                                             const bool valid = true)
{
  WebGPUSvmMask &mask = offset < 128 ? written.low : written.high;
  const uint bit = 1u << (offset % 32);
  const uint lane = (offset / 32) % 4;
  mask[lane] = valid ? mask[lane] | bit : mask[lane] & ~bit;
}

ccl_device_inline void webgpu_svm_stack_mark3(WebGPUSvmStackUsage &written,
                                              const uint offset,
                                              const bool valid = true)
{
  webgpu_svm_stack_mark(written, offset, valid);
  webgpu_svm_stack_mark(written, offset + 1, valid);
  webgpu_svm_stack_mark(written, offset + 2, valid);
}

ccl_device_inline bool webgpu_svm_input_on_stack(const SVMInputFloat input)
{
  return (input.bits >> 8) == (SVM_INPUT_STACK_OFFSET_MASK >> 8);
}

ccl_device_inline bool webgpu_svm_surface_normal(const uint normal, const uint surface_normal)
{
  return normal == SVM_STACK_INVALID || normal == surface_normal;
}

struct WebGPUSvmMaterial {
  float3 color;
  float3 specular_tint;
  float alpha;
  float metallic;
  float eta;
  float f0;
  float mix_weight;
  float transmission;
  float glass_ior;
  float subsurface;
  float3 subsurface_radius;
  float subsurface_anisotropy;
  ClosureType subsurface_method;
  bool ggx;
  bool multiscatter;
  bool standalone_glass;
};

ccl_device_inline bool webgpu_svm_surface_color(const SVMInputFloat3 input,
                                                const WebGPUSvmStackUsage &written,
                                                const float *stack,
                                                float3 &color)
{
  if (webgpu_svm_input_on_stack(input.x)) {
    const uint base = input.x.bits & 0xffu;
    if (!webgpu_svm_stack_has3(written, base)) {
      return false;
    }
    color = make_float3(stack_load_float(stack, base),
                        stack_load_float(stack, base + 1),
                        stack_load_float(stack, base + 2));
  }
  else {
    color = stack_load(stack, input);
  }
  return webgpu_svm_finite(color.x) && webgpu_svm_finite(color.y) && webgpu_svm_finite(color.z);
}

ccl_device_inline bool webgpu_svm_rough_ggx(const float alpha)
{
  /* Match alpha_is_almost_specular() for isotropic GGX. Delta lobes need a
   * different direct-light and continuation path. */
  return sqr(alpha) > 2e-10f;
}

ccl_device_inline bool webgpu_svm_rough_glass_ior(const float ior)
{
  /* The stock glass sampler treats either relative IOR near 1 as singular. */
  return fabsf(ior - 1.0f) >= 1e-4f && fabsf(1.0f / ior - 1.0f) >= 1e-4f;
}

ccl_device_inline bool webgpu_svm_subsurface(const SVMInputFloat3 input_radius,
                                             const SVMInputFloat input_scale,
                                             const float ior,
                                             const SVMInputFloat input_anisotropy,
                                             const ClosureType method,
                                             WebGPUSvmMaterial &material)
{
  const bool random_walk = method == CLOSURE_BSSRDF_RANDOM_WALK_ID;
  const float anisotropy = __uint_as_float(input_anisotropy.bits);
  const float scale = __uint_as_float(input_scale.bits);
  const float3 radius = make_float3(__uint_as_float(input_radius.x.bits),
                                    __uint_as_float(input_radius.y.bits),
                                    __uint_as_float(input_radius.z.bits)) * scale;
  if ((!random_walk && method != CLOSURE_BSSRDF_RANDOM_WALK_SKIN_ID &&
       method != CLOSURE_BSSRDF_RANDOM_WALK_LEGACY_ID) ||
      !webgpu_svm_finite(scale) || !(scale > 0.0f) || !webgpu_svm_finite(anisotropy) ||
      !webgpu_svm_finite(ior) || !webgpu_svm_finite(radius.x) ||
      !webgpu_svm_finite(radius.y) || !webgpu_svm_finite(radius.z) ||
      reduce_min(radius) < BSSRDF_MIN_RADIUS)
  {
    return false;
  }
  material.subsurface_method = method;
  material.subsurface_radius = radius;
  material.subsurface_anisotropy = clamp(anisotropy, -0.99f, random_walk ? 0.99f : 0.9f) +
                                   (random_walk ? 0.0f : 2.0f);
  material.glass_ior = clamp(ior, 1.01f, 3.8f);
  return true;
}

ccl_device_inline bool webgpu_svm_principled(const SVMNodePrincipledBsdfData &data,
                                            const WebGPUSvmStackUsage &written,
                                            const float *stack,
                                            const uint surface_normal,
                                            const float mix_weight,
                                            WebGPUSvmMaterial &material)
{
  /* Pack the supported Principled layers with uniform layer controls. */
  const uint layer_bits = data.sheen_weight.bits | data.coat_weight.bits |
                          data.thin_film_thickness.bits | data.emission_strength.bits |
                          data.diffuse_roughness.bits;
  const float ior_input = __uint_as_float(data.ior.bits);
  const bool varying_metallic = webgpu_svm_input_on_stack(data.metallic);
  if (varying_metallic && !webgpu_svm_stack_has(written, data.metallic.bits & 0xffu)) {
    return false;
  }
  const float metallic = saturatef(stack_load(stack, data.metallic));
  const float ior_level = __uint_as_float(data.specular_ior_level.bits);
  const float sss_input = __uint_as_float(data.subsurface_weight.bits);
  const float transmission_input = __uint_as_float(data.transmission_weight.bits);
  if ((layer_bits & 0x7fffffffu) != 0 || data.alpha.bits != std::bit_cast<uint>(1.0f) ||
      !webgpu_svm_finite(ior_input) || !webgpu_svm_finite(metallic) ||
      !webgpu_svm_finite(ior_level) || !webgpu_svm_finite(transmission_input) ||
      !webgpu_svm_finite(sss_input) || transmission_input < 0.0f || transmission_input > 1.0f ||
      !webgpu_svm_finite(__uint_as_float(data.emission_color.x.bits)) ||
      !webgpu_svm_finite(__uint_as_float(data.emission_color.y.bits)) ||
      !webgpu_svm_finite(__uint_as_float(data.emission_color.z.bits)) ||
      !webgpu_svm_surface_normal(data.normal_offset, surface_normal))
  {
    return false;
  }

  float3 color;
  if (!webgpu_svm_surface_color(data.base_color, written, stack, color)) {
    return false;
  }
  material.color = max(color, zero_float3());
  const float transmission = transmission_input > CLOSURE_WEIGHT_CUTOFF ? transmission_input : 0.0f;
  const float subsurface = saturatef(sss_input);
  material.subsurface = subsurface > CLOSURE_WEIGHT_CUTOFF ? subsurface : 0.0f;
  if (material.subsurface > 0.0f) {
    const bool skin = data.subsurface_method == CLOSURE_BSSRDF_RANDOM_WALK_SKIN_ID;
    if (transmission > 0.0f || data.thin_wall.value != 0 ||
        data.thin_wall.offset != SVM_STACK_INVALID ||
        !webgpu_svm_subsurface(data.subsurface_radius,
                              data.subsurface_scale,
                              skin ? __uint_as_float(data.subsurface_ior.bits) : ior_input,
                              data.subsurface_anisotropy,
                              data.subsurface_method,
                              material))
    {
      return false;
    }
  }
  else if (metallic == 0.0f && ior_level == 0.0f && transmission == 0.0f) {
    material.color *= mix_weight;
    return true;
  }

  const float roughness = __uint_as_float(data.roughness.bits);
  const float3 tint = make_float3(__uint_as_float(data.specular_tint.x.bits),
                                  __uint_as_float(data.specular_tint.y.bits),
                                  __uint_as_float(data.specular_tint.z.bits));
  const bool multiscatter = data.distribution == CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID;
  if ((!multiscatter && data.distribution != CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID) ||
      (data.anisotropic.bits & 0x7fffffffu) != 0 || !webgpu_svm_finite(roughness) ||
      !webgpu_svm_finite(tint.x) || !webgpu_svm_finite(tint.y) || !webgpu_svm_finite(tint.z) ||
      !webgpu_svm_finite(mix_weight) || mix_weight < 0.0f || mix_weight > 1.0f ||
      ior_input <= 0.0f || metallic < 0.0f || metallic > 1.0f ||
      ior_level < 0.0f || ior_level > 1.0f)
  {
    return false;
  }

  const float alpha = sqr(saturatef(roughness));
  const float ior = fmaxf(ior_input, 1e-5f);
  if (!webgpu_svm_rough_ggx(alpha) ||
      (transmission > 0.0f &&
       (varying_metallic || metallic != 0.0f || !webgpu_svm_rough_glass_ior(ior) ||
        data.thin_wall.value != 0 ||
        data.thin_wall.offset != SVM_STACK_INVALID ||
        (data.transmission_dispersion_scale.bits & 0x7fffffffu) != 0)))
  {
    return false;
  }

  float eta = ior;
  float f0 = F0_from_ior(eta);
  if (ior_level != 0.5f) {
    f0 *= 2.0f * ior_level;
    eta = ior_from_F0(f0);
    if (ior < 1.0f) {
      eta = 1.0f / eta;
    }
  }
  material.ggx = true;
  material.multiscatter = multiscatter;
  material.specular_tint = max(tint, zero_float3());
  material.alpha = alpha;
  material.metallic = metallic;
  material.eta = eta;
  material.f0 = f0;
  material.mix_weight = mix_weight;
  material.transmission = transmission;
  if (material.subsurface == 0.0f) {
    material.glass_ior = transmission > 0.0f ? ior : 0.0f;
  }
  else if (material.subsurface_method != CLOSURE_BSSRDF_RANDOM_WALK_SKIN_ID) {
    material.glass_ior = clamp(eta, 1.01f, 3.8f);
  }
  return true;
}

ccl_device_inline bool webgpu_svm_glass(const SVMNodeGlassBsdfData &data,
                                        const WebGPUSvmStackUsage &written,
                                        const float *stack,
                                        const uint surface_normal,
                                        const float mix_weight,
                                        const bool multiscatter,
                                        WebGPUSvmMaterial &material)
{
  const float roughness = __uint_as_float(data.roughness.bits);
  const float ior_input = __uint_as_float(data.ior.bits);
  if (!webgpu_svm_finite(roughness) || !webgpu_svm_finite(ior_input) || ior_input <= 0.0f ||
      !webgpu_svm_finite(mix_weight) || mix_weight < 0.0f || mix_weight > 1.0f ||
      ((data.anisotropy.bits | data.thin_film_thickness.bits) & 0x7fffffffu) != 0 ||
      !webgpu_svm_surface_normal(data.normal_offset, surface_normal))
  {
    return false;
  }

  const float alpha = sqr(saturatef(roughness));
  const float ior = fmaxf(ior_input, 1e-5f);
  float3 color;
  if (!webgpu_svm_rough_ggx(alpha) || !webgpu_svm_rough_glass_ior(ior) ||
      !webgpu_svm_surface_color(data.color, written, stack, color))
  {
    return false;
  }
  material.color = max(color, zero_float3());
  material.specular_tint = one_float3();
  material.alpha = alpha;
  material.metallic = 0.0f;
  material.eta = 1.0f;
  material.f0 = 0.0f;
  material.mix_weight = mix_weight;
  material.transmission = 1.0f;
  material.glass_ior = ior;
  material.ggx = true;
  material.multiscatter = multiscatter;
  material.standalone_glass = true;
  return true;
}

/* Interpret the supported Lambert, Principled and Glass GGX SVM surface nodes.
 * Read typed Blender records word by word because WebGPU storage bindings do not
 * expose the native pointer used by svm_node_get(). Host eligibility checks and
 * GPU evaluation share the same opcode support and operand validation. */
template<typename Fetch, typename FetchImage>
ccl_device_inline bool webgpu_svm_eval_material(Fetch fetch,
                                               FetchImage image,
                                               const WebGPUSvmUv uv,
                                               const float3 generated,
                                               const float3 rgb_to_y,
                                               bool &uses_uv,
                                               bool &uses_generated,
                                               const uint word_count,
                                               const uint shader_count,
                                               const uint packed_shader,
                                               WebGPUSvmMaterial &material)
{
  const uint shader = packed_shader & SHADER_MASK;
  if (shader >= shader_count || shader_count > word_count / webgpu_svm_jump_words) {
    return false;
  }

  uint offset = shader * webgpu_svm_jump_words;
  SVMNodeShaderJump jump;
  if (fetch(offset++) != NODE_SHADER_JUMP ||
      !webgpu_svm_read(fetch, offset, word_count, jump) ||
      uint(jump.offset_surface) < shader_count * webgpu_svm_jump_words ||
      uint(jump.offset_surface) >= uint(jump.offset_volume) ||
      uint(jump.offset_volume) > word_count)
  {
    return false;
  }
  offset = uint(jump.offset_surface);
  const uint end = uint(jump.offset_volume);
  float stack[SVM_STACK_SIZE];
  WebGPUSvmStackUsage written;
  WebGPUSvmStackUsage varying;
  Spectrum weight = zero_spectrum();
  uint surface_normal_offset = uint(SVM_STACK_INVALID);
  bool has_closure = false;
  bool has_image = false;

  while (offset < end) {
    const uint type = fetch(offset++);
    switch (type) {
      case NODE_END:
        return has_closure && offset == end;
      case NODE_GEOMETRY: {
        SVMNodeGeometry node;
        if (!webgpu_svm_read(fetch, offset, end, node) || node.geom_type != NODE_GEOM_N ||
            node.bump_offset != NODE_BUMP_OFFSET_CENTER || node.store_derivatives ||
            uint(node.out_offset) + 2 >= SVM_STACK_SIZE)
        {
          return false;
        }
        surface_normal_offset = node.out_offset;
        /* Only the normal input of the diffuse closure may read this value. */
        webgpu_svm_stack_mark3(written, node.out_offset, false);
        break;
      }
      case NODE_VALUE_F: {
        SVMNodeValueF node;
        if (!webgpu_svm_read(fetch, offset, end, node) || node.out_offset >= SVM_STACK_SIZE) {
          return false;
        }
        svm_node_value_f<float>(stack, node);
        webgpu_svm_stack_mark(written, node.out_offset);
        webgpu_svm_stack_mark(varying, node.out_offset, false);
        if (uint(node.out_offset) >= surface_normal_offset &&
            uint(node.out_offset) < surface_normal_offset + 3)
        {
          surface_normal_offset = uint(SVM_STACK_INVALID);
        }
        break;
      }
      case NODE_VALUE_V: {
        SVMNodeValueV node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            uint(node.out_offset) + 2 >= SVM_STACK_SIZE)
        {
          return false;
        }
        /* Index scalar slots explicitly to preserve a typed array access for SPIR-V. */
        stack_store_float(stack, node.out_offset, node.value.x);
        stack_store_float(stack, uint(node.out_offset) + 1, node.value.y);
        stack_store_float(stack, uint(node.out_offset) + 2, node.value.z);
        webgpu_svm_stack_mark3(written, node.out_offset);
        webgpu_svm_stack_mark3(varying, node.out_offset, false);
        if (uint(node.out_offset) < surface_normal_offset + 3 &&
            uint(node.out_offset) + 2 >= surface_normal_offset)
        {
          surface_normal_offset = uint(SVM_STACK_INVALID);
        }
        break;
      }
      case NODE_ATTR: {
        SVMNodeAttr node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            (node.attr != ATTR_STD_UV && node.attr != ATTR_STD_GENERATED) ||
            node.output_type != NODE_ATTR_OUTPUT_FLOAT3 ||
            node.bump_offset != NODE_BUMP_OFFSET_CENTER || node.store_derivatives ||
            uint(node.out_offset) + 2 >= SVM_STACK_SIZE)
        {
          return false;
        }
        const bool uv_attribute = node.attr == ATTR_STD_UV;
        uses_uv |= uv_attribute;
        uses_generated |= !uv_attribute;
        const float3 values = uv_attribute ? make_float3(uv.x, uv.y, 0.0f) : generated;
        stack_store_float(stack, node.out_offset, values.x);
        stack_store_float(stack, uint(node.out_offset) + 1, values.y);
        stack_store_float(stack, uint(node.out_offset) + 2, values.z);
        webgpu_svm_stack_mark3(written, node.out_offset);
        webgpu_svm_stack_mark3(varying, node.out_offset);
        if (uint(node.out_offset) < surface_normal_offset + 3 &&
            uint(node.out_offset) + 2 >= surface_normal_offset)
        {
          surface_normal_offset = uint(SVM_STACK_INVALID);
        }
        break;
      }
      case NODE_TEX_VORONOI: {
        SVMNodeTexVoronoi node;
        if (!webgpu_svm_read(fetch, offset, end, node)) {
          return false;
        }
        if (node.dimensions != 3 || node.feature != NODE_VORONOI_F1 ||
            node.metric != NODE_VORONOI_EUCLIDEAN || webgpu_svm_input_on_stack(node.detail) ||
            __uint_as_float(node.detail.bits) > 0.0f)
        {
          return false;
        }
        if (webgpu_svm_input_on_stack(node.scale) ||
            webgpu_svm_input_on_stack(node.randomness) ||
            !webgpu_svm_finite(__uint_as_float(node.scale.bits)) ||
            !webgpu_svm_finite(__uint_as_float(node.randomness.bits)))
        {
          return false;
        }
        if (node.distance_offset != SVM_STACK_INVALID ||
            node.position_offset != SVM_STACK_INVALID ||
            node.w_out_offset != SVM_STACK_INVALID || node.radius_offset != SVM_STACK_INVALID ||
            uint(node.color_offset) + 2 >= SVM_STACK_SIZE ||
            !webgpu_svm_stack_has3(written, node.coord))
        {
          return false;
        }
        VoronoiParams params{};
        params.metric = NODE_VORONOI_EUCLIDEAN;
        params.randomness = saturatef(stack_load(stack, node.randomness));
        const float3 coord = make_float3(stack_load_float(stack, node.coord),
                                         stack_load_float(stack, uint(node.coord) + 1),
                                         stack_load_float(stack, uint(node.coord) + 2)) *
                             stack_load(stack, node.scale);
        const float3 color = voronoi_f1(params, coord).color;
        stack_store_float(stack, node.color_offset, color.x);
        stack_store_float(stack, uint(node.color_offset) + 1, color.y);
        stack_store_float(stack, uint(node.color_offset) + 2, color.z);
        webgpu_svm_stack_mark3(written, node.color_offset);
        webgpu_svm_stack_mark3(varying, node.color_offset);
        if (uint(node.color_offset) < surface_normal_offset + 3 &&
            uint(node.color_offset) + 2 >= surface_normal_offset) {
          surface_normal_offset = uint(SVM_STACK_INVALID);
        }
        break;
      }
      case NODE_CONVERT: {
        SVMNodeConvert node;
        if (!webgpu_svm_read(fetch, offset, end, node) || node.convert_type != NODE_CONVERT_CF ||
            node.to_offset >= SVM_STACK_SIZE || !webgpu_svm_stack_has3(written, node.from_offset))
        {
          return false;
        }
        const float3 color = make_float3(stack_load_float(stack, node.from_offset),
                                         stack_load_float(stack, uint(node.from_offset) + 1),
                                         stack_load_float(stack, uint(node.from_offset) + 2));
        stack_store_float(stack, node.to_offset, dot(color, rgb_to_y));
        webgpu_svm_stack_mark(written, node.to_offset);
        webgpu_svm_stack_mark(varying, node.to_offset);
        if (uint(node.to_offset) >= surface_normal_offset &&
            uint(node.to_offset) < surface_normal_offset + 3) {
          surface_normal_offset = uint(SVM_STACK_INVALID);
        }
        break;
      }
      case NODE_RGB_RAMP: {
        SVMNodeRGBRamp node;
        if (!webgpu_svm_read(fetch, offset, end, node) || !node.table_size ||
            node.table_size > (end - offset) / 4u || uint(node.color_offset) + 2 >= SVM_STACK_SIZE ||
            node.alpha_offset != SVM_STACK_INVALID ||
            (webgpu_svm_input_on_stack(node.fac) &&
             !webgpu_svm_stack_has(written, node.fac.bits & 0xffu)))
        {
          return false;
        }
        /* Use the stock SVM ramp table and its non-extrapolating interpolation. */
        const float f = saturatef(stack_load(stack, node.fac)) * float(node.table_size - 1);
        const uint index = min(uint(f), node.table_size - 1);
        const float fraction = f - float(index);
        float3 color = webgpu_svm_read_rgb(fetch, offset + 4u * index);
        if (node.interpolate && fraction > 0.0f && index + 1 < node.table_size) {
          color = (1.0f - fraction) * color +
                  fraction * webgpu_svm_read_rgb(fetch, offset + 4u * (index + 1u));
        }
        offset += 4u * node.table_size;
        stack_store_float(stack, node.color_offset, color.x);
        stack_store_float(stack, uint(node.color_offset) + 1, color.y);
        stack_store_float(stack, uint(node.color_offset) + 2, color.z);
        webgpu_svm_stack_mark3(written, node.color_offset);
        webgpu_svm_stack_mark3(varying, node.color_offset);
        if (uint(node.color_offset) < surface_normal_offset + 3 &&
            uint(node.color_offset) + 2 >= surface_normal_offset) {
          surface_normal_offset = uint(SVM_STACK_INVALID);
        }
        break;
      }
      case NODE_TEX_IMAGE: {
        SVMNodeTexImage node;
        /* The first supported path samples one image from UVs or uniform coordinates. */
        if (has_image || !webgpu_svm_read(fetch, offset, end, node) || node.id < 0 ||
            node.projection != NODE_IMAGE_PROJ_FLAT ||
            (node.flags & ~uint(NODE_IMAGE_COMPRESS_AS_SRGB)) ||
            uint(node.out_offset) + 2 >= SVM_STACK_SIZE ||
            node.alpha_offset != SVM_STACK_INVALID ||
            !webgpu_svm_stack_has3(written, node.co))
        {
          return false;
        }
        const float2 co = make_float2(stack_load_float(stack, node.co),
                                      stack_load_float(stack, uint(node.co) + 1));
        float3 color;
        if (!(fabsf(co.x) <= 8194.0f && fabsf(co.y) <= 8194.0f) || !image(node.id, co, color)) {
          return false;
        }
        if (node.flags & NODE_IMAGE_COMPRESS_AS_SRGB) {
          /* Decode each channel after sampling, as Cycles does. Scalar calls
           * also avoid the SPIR-V backend's unsupported float3 aggregate copy. */
          color.x = color_srgb_to_linear(color.x);
          color.y = color_srgb_to_linear(color.y);
          color.z = color_srgb_to_linear(color.z);
        }
        has_image = true;
        stack_store_float(stack, node.out_offset, color.x);
        stack_store_float(stack, uint(node.out_offset) + 1, color.y);
        stack_store_float(stack, uint(node.out_offset) + 2, color.z);
        webgpu_svm_stack_mark3(written, node.out_offset);
        webgpu_svm_stack_mark3(varying, node.out_offset);
        if (uint(node.out_offset) < surface_normal_offset + 3 &&
            uint(node.out_offset) + 2 >= surface_normal_offset)
        {
          surface_normal_offset = uint(SVM_STACK_INVALID);
        }
        break;
      }
      case NODE_CLOSURE_SET_WEIGHT: {
        SVMNodeClosureSetWeight node;
        if (!webgpu_svm_read(fetch, offset, end, node)) {
          return false;
        }
        weight = rgb_to_spectrum(node.rgb);
        break;
      }
      case NODE_CLOSURE_WEIGHT: {
        SVMNodeClosureWeight node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            !webgpu_svm_stack_has3(written, node.weight_offset))
        {
          return false;
        }
        weight = rgb_to_spectrum(make_float3(stack_load_float(stack, node.weight_offset),
                                             stack_load_float(stack, uint(node.weight_offset) + 1),
                                             stack_load_float(stack, uint(node.weight_offset) + 2)));
        break;
      }
      case NODE_CLOSURE_BSDF: {
        SVMNodeClosureBsdf closure;
        if (has_closure || !webgpu_svm_read(fetch, offset, end, closure) ||
            (closure.mix_weight_offset != SVM_STACK_INVALID &&
             (!webgpu_svm_stack_has(written, closure.mix_weight_offset) ||
              webgpu_svm_stack_has(varying, closure.mix_weight_offset))))
        {
          return false;
        }
        const float mix_weight = stack_load_float_default(stack, closure.mix_weight_offset, 1.0f);
        material.ggx = false;
        material.standalone_glass = false;
        material.transmission = 0.0f;
        material.subsurface = 0.0f;
        Spectrum closure_weight;
        if (closure.closure_type == CLOSURE_BSDF_DIFFUSE_ID) {
          SVMNodeDiffuseBsdfData diffuse;
          if (!webgpu_svm_read(fetch, offset, end, diffuse) ||
              !webgpu_svm_surface_normal(diffuse.normal_offset, surface_normal_offset) ||
              (webgpu_svm_input_on_stack(diffuse.roughness) &&
               (!webgpu_svm_stack_has(written, diffuse.roughness.bits & 0xffu) ||
                webgpu_svm_stack_has(varying, diffuse.roughness.bits & 0xffu))))
          {
            return false;
          }
          const float roughness = stack_load(stack, diffuse.roughness);
          if (!webgpu_svm_finite(roughness) || !diffuse_roughness_is_almost_zero(roughness)) {
            return false;
          }
          closure_weight = weight * mix_weight;
        }
        else if (closure.closure_type == CLOSURE_BSDF_PRINCIPLED_ID) {
          SVMNodePrincipledBsdfData principled;
          if (!webgpu_svm_read(fetch, offset, end, principled) ||
              !webgpu_svm_principled(
                  principled, written, stack, surface_normal_offset, mix_weight, material))
          {
            return false;
          }
          if (material.ggx) {
            has_closure = true;
            break;
          }
          closure_weight = material.color;
        }
        else if (CLOSURE_IS_BSSRDF(closure.closure_type)) {
          SVMNodeBssrdfData data;
          if (!webgpu_svm_read(fetch, offset, end, data) ||
              !webgpu_svm_surface_normal(data.normal_offset, surface_normal_offset) ||
              !webgpu_svm_finite(weight.x) || !webgpu_svm_finite(weight.y) ||
              !webgpu_svm_finite(weight.z) || reduce_min(weight) < 0.0f ||
              reduce_max(weight) > 1.0f ||
              !webgpu_svm_finite(mix_weight) || mix_weight < 0.0f || mix_weight > 1.0f ||
              !webgpu_svm_finite(__uint_as_float(data.roughness.bits)) ||
              !webgpu_svm_subsurface(data.radius,
                                    data.scale,
                                    __uint_as_float(data.ior.bits),
                                    data.anisotropy,
                                    closure.closure_type,
                                    material))
          {
            return false;
          }
          material.color = weight;
          material.ggx = true;
          material.multiscatter = false;
          material.specular_tint = zero_float3();
          material.alpha = saturatef(__uint_as_float(data.roughness.bits));
          material.metallic = 0.0f;
          material.eta = 1.0f;
          material.f0 = 0.0f;
          material.mix_weight = mix_weight;
          material.subsurface = 1.0f;
          has_closure = true;
          break;
        }
        else if (closure.closure_type == CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID ||
                 closure.closure_type == CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID)
        {
          SVMNodeGlassBsdfData glass;
          if (!webgpu_svm_read(fetch, offset, end, glass) ||
              !webgpu_svm_glass(glass,
                                written,
                                stack,
                                surface_normal_offset,
                                mix_weight,
                                closure.closure_type == CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID,
                                material))
          {
            return false;
          }
          has_closure = true;
          break;
        }
        else {
          return false;
        }
        if (!webgpu_svm_finite(closure_weight.x) || !webgpu_svm_finite(closure_weight.y) ||
            !webgpu_svm_finite(closure_weight.z))
        {
          return false;
        }
        material.color = closure_sample_weight(0, closure_weight) > 0.0f ? closure_weight :
                                                                         zero_float3();
        if (!webgpu_svm_finite(average(material.color))) {
          return false;
        }
        has_closure = true;
        break;
      }
      default:
        return false;
    }
  }
  return false;
}

CCL_NAMESPACE_END
