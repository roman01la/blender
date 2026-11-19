/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "kernel/svm/value.h"

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
/* Immediately after the bytecode: triangle count, image count, six UV words per
 * triangle and six words per image (ID, dimensions, extension, interpolation,
 * pixel offset). */
constexpr uint webgpu_svm_texture_header_words = 2;
constexpr uint webgpu_svm_triangle_uv_words = 6;
constexpr uint webgpu_svm_image_words = 6;
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
  const uint images = word(texture + 1);
  const uint table = texture + webgpu_svm_texture_header_words +
                     word(texture) * webgpu_svm_triangle_uv_words;
  for (uint i = 0; i < images; ++i) {
    const uint base = table + i * webgpu_svm_image_words;
    if (word(base) == uint(image)) {
      const uint width = word(base + 1);
      const uint height = word(base + 2);
      const uint extension = word(base + 3);
      const uint interpolation = word(base + 4);
      const uint start = word(base + 5);
      const auto texel = [&](const uint index) {
        const uint pixel = start + 3 * index;
        return make_float3(__uint_as_float(word(pixel)),
                           __uint_as_float(word(pixel + 1)),
                           __uint_as_float(word(pixel + 2)));
      };
      color = webgpu_svm_sample_image(texel, width, height, extension, interpolation, uv);
      return true;
    }
  }
  return false;
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

ccl_device_inline bool webgpu_svm_principled(const SVMNodePrincipledBsdfData &data,
                                            const WebGPUSvmStackUsage &written,
                                            const float *stack,
                                            const uint surface_normal,
                                            const float mix_weight,
                                            WebGPUSvmMaterial &material)
{
  /* Accept the opaque Lambert, dielectric, metallic and rough transmission
   * layers, with uniform layer controls. */
  const uint layer_bits = data.subsurface_weight.bits | data.sheen_weight.bits | data.coat_weight.bits |
                          data.thin_film_thickness.bits | data.emission_strength.bits |
                          data.diffuse_roughness.bits;
  const float ior_input = __uint_as_float(data.ior.bits);
  const float metallic = __uint_as_float(data.metallic.bits);
  const float ior_level = __uint_as_float(data.specular_ior_level.bits);
  const float transmission_input = __uint_as_float(data.transmission_weight.bits);
  if ((layer_bits & 0x7fffffffu) != 0 || data.alpha.bits != std::bit_cast<uint>(1.0f) ||
      !webgpu_svm_finite(ior_input) || !webgpu_svm_finite(metallic) ||
      !webgpu_svm_finite(ior_level) || !webgpu_svm_finite(transmission_input) ||
      transmission_input < 0.0f || transmission_input > 1.0f ||
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
  if (metallic == 0.0f && ior_level == 0.0f && transmission == 0.0f) {
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
       (metallic != 0.0f || !webgpu_svm_rough_glass_ior(ior) || data.thin_wall.value != 0 ||
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
  material.glass_ior = transmission > 0.0f ? ior : 0.0f;
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
                                               bool &uses_uv,
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
        if (!webgpu_svm_read(fetch, offset, end, node) || node.attr != ATTR_STD_UV ||
            node.output_type != NODE_ATTR_OUTPUT_FLOAT3 ||
            node.bump_offset != NODE_BUMP_OFFSET_CENTER || node.store_derivatives ||
            uint(node.out_offset) + 2 >= SVM_STACK_SIZE)
        {
          return false;
        }
        uses_uv = true;
        stack_store_float(stack, node.out_offset, uv.x);
        stack_store_float(stack, uint(node.out_offset) + 1, uv.y);
        stack_store_float(stack, uint(node.out_offset) + 2, 0.0f);
        webgpu_svm_stack_mark3(written, node.out_offset);
        webgpu_svm_stack_mark3(varying, node.out_offset);
        if (uint(node.out_offset) < surface_normal_offset + 3 &&
            uint(node.out_offset) + 2 >= surface_normal_offset)
        {
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
          /* Cycles decodes sRGB after sampling the stored texels. */
          color = color_srgb_to_linear_v3(color);
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
