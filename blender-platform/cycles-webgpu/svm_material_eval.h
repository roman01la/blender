/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "material_layout.h"
#include "svm_surface_context.h"

#include "kernel/camera/projection.h"
#include "kernel/svm/brightness.h"
#include "kernel/svm/clamp.h"
#include "kernel/svm/gamma.h"
#include "kernel/svm/hsv.h"
#include "kernel/svm/invert.h"
#include "kernel/svm/map_range.h"
#include "kernel/svm/mix.h"
#include "kernel/svm/normal.h"
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
#include "kernel/svm/value.h"
#include "kernel/svm/voronoi.h"
#include "kernel/svm/wave.h"
#include "kernel/svm/white_noise.h"

#include "kernel/closure/bsdf_oren_nayar.h"
#include "kernel/closure/bsdf_util.h"
#include "kernel/util/colorspace.h"
#include "util/color.h"
#include "util/types_image.h"

#include <array>
#include <bit>
#include <type_traits>
#include <utility>

CCL_NAMESPACE_BEGIN

static_assert(SHADER_MASK == CYCLES_WEBGPU_SVM_SHADER_ID_MASK);

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
/* The interpolation word also records an optional alpha plane after the RGB texels. */
constexpr uint webgpu_svm_image_alpha = 0x80000000u;
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

ccl_device_inline uint webgpu_svm_address_texel(const uint size,
                                                const uint extension,
                                                const uint magnitude,
                                                const bool negative)
{
  if (extension == EXTENSION_REPEAT) {
    const uint remainder = magnitude % size;
    return negative && remainder != 0 ? size - remainder : remainder;
  }
  if (extension == EXTENSION_MIRROR) {
    /* Match ImageInterpolator::wrap_mirror(), which repeats the edge texel. */
    const uint mirrored = (magnitude - uint(negative && magnitude > 0)) % (2 * size);
    return mirrored < size ? mirrored : 2 * size - mirrored - 1;
  }
  if (extension == EXTENSION_CLIP) {
    return (negative && magnitude != 0) || magnitude >= size ? size : magnitude;
  }
  return negative ? 0 : min(magnitude, size - 1);
}

template<int Neighbor>
ccl_device_inline uint webgpu_svm_texel_axis(const float coordinate,
                                             const uint size,
                                             const uint extension)
{
  /* Work in unsigned magnitudes so lowering preserves negative coordinates.
   * ImageInterpolator::frac() uses the preceding cell at negative integers. */
  const bool negative = coordinate < 0.0f;
  const uint whole = uint(fabsf(coordinate));
  if constexpr (Neighbor == -1) {
    if (negative || whole == 0) {
      return webgpu_svm_address_texel(size, extension, negative ? whole + 2 : 1, true);
    }
    return webgpu_svm_address_texel(size, extension, whole - 1, false);
  }
  else if constexpr (Neighbor == 2) {
    if (negative && whole == 0) {
      return webgpu_svm_address_texel(size, extension, 1, false);
    }
    return webgpu_svm_address_texel(size, extension, negative ? whole - 1 : whole + 2, negative);
  }
  else {
    static_assert(Neighbor == 0 || Neighbor == 1);
    const uint magnitude = whole + (negative ? 1u - Neighbor : Neighbor);
    return webgpu_svm_address_texel(size, extension, magnitude, negative);
  }
}

ccl_device_inline float webgpu_svm_texel_fraction(const float coordinate)
{
  /* Match ImageInterpolator::frac(), including its negative integer endpoint. */
  const uint magnitude = uint(fabsf(coordinate));
  return coordinate < 0.0f ? coordinate + float(magnitude + 1) : coordinate - float(magnitude);
}

ccl_device_inline bool webgpu_svm_uses_cubic_filter(const uint interpolation)
{
  return interpolation == INTERPOLATION_CUBIC || interpolation == INTERPOLATION_SMART;
}

#ifndef CYCLES_WEBGPU_SVM_CORE
ccl_device_inline float4 webgpu_svm_cubic_weights(const float t)
{
  /* B-spline weights from ImageInterpolator::interp_cubic(). */
  return make_float4((((-1.0f / 6.0f) * t + 0.5f) * t - 0.5f) * t + (1.0f / 6.0f),
                     ((0.5f * t - 1.0f) * t) * t + (2.0f / 3.0f),
                     ((-0.5f * t + 0.5f) * t + 0.5f) * t + (1.0f / 6.0f),
                     (1.0f / 6.0f) * t * t * t);
}
#endif

template<bool AllowCubic = false, typename FetchTexel>
ccl_device_inline auto webgpu_svm_sample_image(FetchTexel texel,
                                                 const uint width,
                                                 const uint height,
                                                 const uint extension,
                                                 const uint interpolation,
                                                 const float2 uv)
{
  const auto read = [&](const uint x, const uint y) {
    using Color = decltype(texel(0));
    return x < width && y < height ? texel(y * width + x) : Color{};
  };
  const float x = uv.x * float(width);
  const float y = uv.y * float(height);
  if (interpolation == INTERPOLATION_CLOSEST) {
    const uint ix = webgpu_svm_texel_axis<0>(x, width, extension);
    const uint iy = webgpu_svm_texel_axis<0>(y, height, extension);
    return read(ix, iy);
  }

  /* Sample on either side of the centers, using storage-word fetches in
   * place of native texel pointers. Apply the extension to each neighbor. */
  const float sx = x - 0.5f;
  const float sy = y - 0.5f;
  const float tx = webgpu_svm_texel_fraction(sx);
  const float ty = webgpu_svm_texel_fraction(sy);
  const uint ix = webgpu_svm_texel_axis<0>(sx, width, extension);
  const uint nix = webgpu_svm_texel_axis<1>(sx, width, extension);
  const uint iy = webgpu_svm_texel_axis<0>(sy, height, extension);
  const uint niy = webgpu_svm_texel_axis<1>(sy, height, extension);
#ifndef CYCLES_WEBGPU_SVM_CORE
  if constexpr (AllowCubic) {
    if (webgpu_svm_uses_cubic_filter(interpolation)) {
      const uint px = webgpu_svm_texel_axis<-1>(sx, width, extension);
      const uint qx = webgpu_svm_texel_axis<2>(sx, width, extension);
      const uint py = webgpu_svm_texel_axis<-1>(sy, height, extension);
      const uint qy = webgpu_svm_texel_axis<2>(sy, height, extension);
      const float4 u = webgpu_svm_cubic_weights(tx);
      const float4 v = webgpu_svm_cubic_weights(ty);
      /* Keep the captured weights live by inlining before shader IR lowering. */
      const auto cubic_texel = [&](const uint i, const uint j) __attribute__((always_inline)) {
        const auto value = texel(min(j, height - 1) * width + min(i, width - 1));
        return float(i < width && j < height) * value;
      };
      const auto row = [&](const uint j) __attribute__((always_inline)) {
        return u.x * cubic_texel(px, j) + u.y * cubic_texel(ix, j) +
               u.z * cubic_texel(nix, j) + u.w * cubic_texel(qx, j);
      };
      return v.x * row(py) + v.y * row(iy) + v.z * row(niy) + v.w * row(qy);
    }
  }
#endif
  return (1.0f - ty) * (1.0f - tx) * read(ix, iy) +
         (1.0f - ty) * tx * read(nix, iy) +
         ty * (1.0f - tx) * read(ix, niy) + ty * tx * read(nix, niy);
}

template<typename Fetch>
ccl_device_inline bool webgpu_svm_image(
    Fetch word, const uint texture, const int image, const float2 uv, const bool alpha, float4 &color)
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
#ifdef CYCLES_WEBGPU_SVM_CORE
  if (webgpu_svm_uses_cubic_filter(interpolation & ~webgpu_svm_image_alpha)) {
    return false;
  }
#endif
  if (alpha && !(interpolation & webgpu_svm_image_alpha)) {
    return false;
  }
  const uint start = word(base + 4);
  const auto texel = [&](const uint index) {
    const float3 rgb = webgpu_svm_read_rgb(word, start + 3u * index);
    const float a = alpha ? __uint_as_float(word(start + 3u * width * height + index)) : 1.0f;
    return make_float4(rgb.x, rgb.y, rgb.z, a);
  };
  color = webgpu_svm_sample_image<true>(
      texel, width, height, extension, interpolation & ~webgpu_svm_image_alpha, uv);
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

ccl_device_inline bool webgpu_svm_input_valid(const WebGPUSvmStackUsage &written,
                                             const SVMInputFloat input)
{
  return webgpu_svm_input_on_stack(input) ? webgpu_svm_stack_has(written, input.bits & 0xffu) :
                                          webgpu_svm_finite(__uint_as_float(input.bits));
}

ccl_device_inline bool webgpu_svm_input_valid(const WebGPUSvmStackUsage &written,
                                             const SVMInputFloat3 input)
{
  return webgpu_svm_input_on_stack(input.x) ?
             webgpu_svm_stack_has3(written, input.x.bits & 0xffu) :
             webgpu_svm_finite(__uint_as_float(input.x.bits)) &&
                 webgpu_svm_finite(__uint_as_float(input.y.bits)) &&
                 webgpu_svm_finite(__uint_as_float(input.z.bits));
}

template<typename... Inputs>
ccl_device_inline bool webgpu_svm_inputs_valid(const WebGPUSvmStackUsage &written,
                                              const Inputs... inputs)
{
  return (webgpu_svm_input_valid(written, inputs) && ...);
}

ccl_device_inline bool webgpu_svm_outputs_valid(const uint scalar, const uint vector)
{
  return (scalar == SVM_STACK_INVALID || scalar < SVM_STACK_SIZE) &&
         (vector == SVM_STACK_INVALID || vector + 2 < SVM_STACK_SIZE);
}

/* A procedural result can vary even when host eligibility happens to sample a
 * constant-looking point. Do not accept it where a uniform closure is required. */
ccl_device_inline void webgpu_svm_mark_outputs(WebGPUSvmStackUsage &written,
                                               WebGPUSvmStackUsage &varying,
                                               uint &surface_normal_offset,
                                               const uint scalar,
                                               const uint vector)
{
  if (stack_valid(scalar)) {
    webgpu_svm_stack_mark(written, scalar);
    webgpu_svm_stack_mark(varying, scalar);
    if (scalar >= surface_normal_offset && scalar < surface_normal_offset + 3) {
      surface_normal_offset = SVM_STACK_INVALID;
    }
  }
  if (stack_valid(vector)) {
    webgpu_svm_stack_mark3(written, vector);
    webgpu_svm_stack_mark3(varying, vector);
    if (vector < surface_normal_offset + 3 && vector + 2 >= surface_normal_offset) {
      surface_normal_offset = SVM_STACK_INVALID;
    }
  }
}

ccl_device_inline bool webgpu_svm_surface_normal(const uint normal, const uint surface_normal)
{
  return normal == SVM_STACK_INVALID || normal == surface_normal;
}

template<typename Image, typename Node>
ccl_device_inline bool webgpu_svm_eval_image(Image image,
                                             const Node &node,
                                             ccl_private float *stack,
                                             WebGPUSvmStackUsage &written,
                                             WebGPUSvmStackUsage &varying,
                                             uint &surface_normal_offset)
{
  if (node.id < 0 ||
      (node.flags & ~uint(NODE_IMAGE_COMPRESS_AS_SRGB | NODE_IMAGE_ALPHA_UNASSOCIATE)) ||
      !webgpu_svm_outputs_valid(node.alpha_offset, node.out_offset) ||
      !webgpu_svm_stack_has3(written, node.co))
  {
    return false;
  }
  const float3 co = stack_load_float3(stack, node.co);
  if (!(fabsf(co.x) <= 8194.0f && fabsf(co.y) <= 8194.0f)) {
    return false;
  }
  float2 uv = make_float2(co);
#ifndef CYCLES_WEBGPU_SVM_CORE
  if constexpr (std::is_same_v<Node, SVMNodeTexEnvironment>) {
    if (!(fabsf(co.z) <= 8194.0f)) {
      return false;
    }
    const float3 direction = safe_normalize(co);
    if (node.projection == NODE_ENVIRONMENT_EQUIRECTANGULAR) {
      if (direction.x == 0.0f && direction.y == 0.0f && direction.z != 0.0f) {
        /* WGSL atan2 is undefined at the poles. Equirectangular images repeat. */
        const bool negative_x = (__float_as_uint(direction.x) >> 31) != 0;
        uv = make_float2(negative_x ? 0.0f : 0.5f, direction.z > 0.0f ? 1.0f : 0.0f);
      }
      else {
        uv = direction_to_equirectangular(direction);
      }
    }
    else {
      uv = direction_to_mirrorball(direction);
    }
  }
  else if (node.projection != NODE_IMAGE_PROJ_FLAT) {
    if (!(fabsf(co.z) <= 8194.0f)) {
      return false;
    }
    const float3 square = (co - make_float3(0.5f)) * 2.0f;
    uv = node.projection == NODE_IMAGE_PROJ_SPHERE ? map_to_sphere(square) : map_to_tube(square);
  }
#endif
  const bool unassociate = node.flags & NODE_IMAGE_ALPHA_UNASSOCIATE;
  const bool alpha = unassociate || stack_valid(node.alpha_offset);
  float4 color;
  if (!image(node.id, uv, alpha, color)) {
    return false;
  }
  if (stack_valid(node.out_offset)) {
    if (unassociate && color.w != 0.0f && color.w != 1.0f) {
      color.x /= color.w;
      color.y /= color.w;
      color.z /= color.w;
    }
    if (node.flags & NODE_IMAGE_COMPRESS_AS_SRGB) {
      /* Stock SVM decodes RGB after interpolation and unassociation. */
      color.x = color_srgb_to_linear(color.x);
      color.y = color_srgb_to_linear(color.y);
      color.z = color_srgb_to_linear(color.z);
    }
    stack_store_float3(stack, node.out_offset, make_float3(color));
  }
  if (stack_valid(node.alpha_offset)) {
    stack_store_float(stack, node.alpha_offset, color.w);
  }
  webgpu_svm_mark_outputs(
      written, varying, surface_normal_offset, node.alpha_offset, node.out_offset);
  return true;
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
  float standalone_transmission;
  bool requires_extended_svm;
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
  /* The fused Principled reflection path evaluates non-singular GGX lobes. */
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

ccl_device_inline void webgpu_svm_set_standalone_transmission(const float3 color,
                                                            const float alpha,
                                                            const float ior,
                                                            const float mix_weight,
                                                            const float closure,
                                                            const bool multiscatter,
                                                            WebGPUSvmMaterial &material)
{
  material.color = color;
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
  material.standalone_transmission = closure;
}

ccl_device_inline bool webgpu_svm_glass(const SVMNodeGlassBsdfData &data,
                                        const WebGPUSvmStackUsage &written,
                                        const float *stack,
                                        const uint surface_normal,
                                        const float mix_weight,
                                        const ClosureType distribution,
                                        WebGPUSvmMaterial &material)
{
  const uint roughness_offset = data.roughness.bits & 0xffu;
  if (webgpu_svm_input_on_stack(data.roughness) && !webgpu_svm_stack_has(written, roughness_offset)) {
    return false;
  }
  const float roughness = stack_load(stack, data.roughness);
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
  if (!webgpu_svm_rough_glass_ior(ior) ||
      !webgpu_svm_surface_color(data.color, written, stack, color))
  {
    return false;
  }
  const float closure = distribution == CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID ?
                            CYCLES_WEBGPU_GLASS_BECKMANN :
                            CYCLES_WEBGPU_GLASS_GGX;
  webgpu_svm_set_standalone_transmission(max(color, zero_float3()),
                                        alpha,
                                        ior,
                                        mix_weight,
                                        closure,
                                        distribution == CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID,
                                        material);
  return true;
}

ccl_device_inline bool webgpu_svm_refraction(const SVMNodeRefractionBsdfData &data,
                                            const WebGPUSvmStackUsage &written,
                                            const float *stack,
                                            const uint surface_normal,
                                            const float3 color,
                                            const float mix_weight,
                                            const bool beckmann,
                                            WebGPUSvmMaterial &material)
{
  const float ior_input = __uint_as_float(data.ior.bits);
  const float ior = fmaxf(ior_input, 1e-5f);
  if (!webgpu_svm_input_valid(written, data.roughness) ||
      !webgpu_svm_surface_normal(data.normal_offset, surface_normal) ||
      !webgpu_svm_finite(ior_input) || ior_input <= 0.0f || !webgpu_svm_rough_glass_ior(ior) ||
      !webgpu_svm_finite(mix_weight) || mix_weight < 0.0f || mix_weight > 1.0f ||
      !webgpu_svm_finite(color.x) || !webgpu_svm_finite(color.y) ||
      !webgpu_svm_finite(color.z) || reduce_min(color) < 0.0f)
  {
    return false;
  }
  webgpu_svm_set_standalone_transmission(
      color,
      saturatef(sqr(stack_load(stack, data.roughness))),
      ior,
      mix_weight,
      beckmann ? CYCLES_WEBGPU_REFRACTION_BECKMANN : CYCLES_WEBGPU_REFRACTION_GGX,
      false,
      material);
  return true;
}

/* Follow Cycles' ramp interpolation and extrapolation with word-based table
 * reads. The bytecode table stays in scene storage. */
template<typename Sample>
ccl_device_inline auto webgpu_svm_ramp_lookup(const Sample sample,
                                               const uint count,
                                               const float factor,
                                               const bool interpolate,
                                               const bool extrapolate)
{
  if (extrapolate && (factor < 0.0f || factor > 1.0f)) {
    const bool below = factor < 0.0f;
    const auto endpoint = sample(below ? 0u : count - 1);
    const auto adjacent = sample(below ? 1u : count - 2);
    const float distance = below ? -factor : factor - 1.0f;
    return endpoint + (endpoint - adjacent) * distance * float(count - 1);
  }

  const float position = saturatef(factor) * float(count - 1);
  const uint index = min(uint(position), count - 1);
  const float fraction = position - float(index);
  const auto value = sample(index);
  if (interpolate && fraction > 0.0f && index + 1 < count) {
    return (1.0f - fraction) * value + fraction * sample(index + 1);
  }
  return value;
}

/* Interpret the supported Lambert, Principled and transmission SVM surface nodes.
 * Read typed Blender records word by word because WebGPU storage bindings do not
 * expose the native pointer used by svm_node_get(). Host eligibility checks and
 * GPU evaluation share the same opcode support and operand validation. */
template<typename Fetch, typename FetchImage, typename FetchSurface = WebGPUSvmNoSurface>
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
                                               WebGPUSvmMaterial &material,
                                               FetchSurface surface = {})
{
  material.requires_extended_svm = false;
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
  uint lambert_closures = 0;
  bool mixes_closures = false;

  while (offset < end) {
    const uint type = fetch(offset++);
    switch (type) {
      case NODE_END:
        return has_closure && offset == end;
      case NODE_GEOMETRY: {
        SVMNodeGeometry node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            node.bump_offset != NODE_BUMP_OFFSET_CENTER || node.store_derivatives ||
            uint(node.out_offset) + 2 >= SVM_STACK_SIZE)
        {
          return false;
        }
        float3 value;
        switch (node.geom_type) {
          case NODE_GEOM_P:
            value = webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_POSITION));
            break;
          case NODE_GEOM_N:
            value = webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_NORMAL));
            break;
          case NODE_GEOM_Ng:
            value = webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_GEOMETRIC));
            break;
          case NODE_GEOM_I:
            value = -webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_DIRECTION));
            break;
          case NODE_GEOM_uv: {
            const auto hit = surface(CYCLES_WEBGPU_SVM_INTERSECTION);
            const float u = __uint_as_float(hit.y);
            const float v = __uint_as_float(hit.z);
            /* Geometry's Parametric output uses the stock triangle barycentrics. */
            value = make_float3(1.0f - u - v, u, 0.0f);
            break;
          }
          default:
            return false;
        }
        stack_store_float(stack, node.out_offset, value.x);
        stack_store_float(stack, uint(node.out_offset) + 1, value.y);
        stack_store_float(stack, uint(node.out_offset) + 2, value.z);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, SVM_STACK_INVALID, node.out_offset);
        if (node.geom_type == NODE_GEOM_N) {
          surface_normal_offset = node.out_offset;
        }
        break;
      }
      case NODE_CAMERA: {
        SVMNodeCamera node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            !webgpu_svm_outputs_valid(node.zdepth_offset, node.vector_offset) ||
            !webgpu_svm_outputs_valid(node.distance_offset, SVM_STACK_INVALID))
        {
          return false;
        }
        /* Match the stock Camera node using its synchronized world-to-camera transform. */
        const float3 view = webgpu_svm_camera_position(surface);
        if (stack_valid(node.vector_offset)) {
          const float3 direction = normalize(view);
          stack_store_float(stack, node.vector_offset, direction.x);
          stack_store_float(stack, uint(node.vector_offset) + 1, direction.y);
          stack_store_float(stack, uint(node.vector_offset) + 2, direction.z);
        }
        if (stack_valid(node.zdepth_offset)) {
          stack_store_float(stack, node.zdepth_offset, view.z);
        }
        if (stack_valid(node.distance_offset)) {
          stack_store_float(stack, node.distance_offset, len(view));
        }
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, node.zdepth_offset, node.vector_offset);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, node.distance_offset, SVM_STACK_INVALID);
        break;
      }
      case NODE_TEX_COORD:
      case NODE_TEX_COORD_DERIVATIVE: {
        SVMNodeTexCoord node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            node.bump_offset != NODE_BUMP_OFFSET_CENTER || node.store_derivatives ||
            uint(node.out_offset) + 2 >= SVM_STACK_SIZE)
        {
          return false;
        }
        float3 value;
        switch (node.texco_type) {
          case NODE_TEXCO_CAMERA:
            value = webgpu_svm_camera_position(surface);
            break;
          case NODE_TEXCO_WINDOW:
            value = webgpu_svm_window_position(surface);
            break;
          case NODE_TEXCO_REFLECTION: {
            const float3 wi = -webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_DIRECTION));
            const float3 normal = webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_NORMAL));
            value = -reflect(wi, normal);
            break;
          }
          case NODE_TEXCO_OBJECT: {
            const Transform inverse = webgpu_svm_context_transform(
                surface, CYCLES_WEBGPU_SVM_OBJECT_INVERSE);
            value = transform_point(
                &inverse, webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_POSITION)));
            break;
          }
          case NODE_TEXCO_NORMAL: {
            if (type == NODE_TEX_COORD_DERIVATIVE) {
              return false;
            }
            const Transform transform = webgpu_svm_context_transform(
                surface, CYCLES_WEBGPU_SVM_OBJECT_TRANSFORM);
            const float3 world_normal = webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_NORMAL));
            value = safe_normalize(transform_direction_transposed(&transform, world_normal));
            break;
          }
          case NODE_TEXCO_OBJECT_WITH_TRANSFORM: {
            PackedTransform inverse;
            if (!webgpu_svm_read(fetch, offset, end, inverse)) {
              return false;
            }
            const Transform transform = make_transform(inverse);
            value = transform_point(
                &transform, webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_POSITION)));
            break;
          }
          default:
            return false;
        }
        stack_store_float(stack, node.out_offset, value.x);
        stack_store_float(stack, uint(node.out_offset) + 1, value.y);
        stack_store_float(stack, uint(node.out_offset) + 2, value.z);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, SVM_STACK_INVALID, node.out_offset);
        break;
      }
      case NODE_OBJECT_INFO: {
        SVMNodeObjectInfo node;
        if (!webgpu_svm_read(fetch, offset, end, node) || node.info_type > NODE_INFO_OB_RANDOM ||
            node.out_offset >= SVM_STACK_SIZE)
        {
          return false;
        }
        if (node.info_type == NODE_INFO_OB_LOCATION || node.info_type == NODE_INFO_OB_COLOR) {
          if (uint(node.out_offset) + 2 >= SVM_STACK_SIZE) {
            return false;
          }
          float3 result;
          if (node.info_type == NODE_INFO_OB_LOCATION) {
            const Transform transform = webgpu_svm_context_transform(
                surface, CYCLES_WEBGPU_SVM_OBJECT_TRANSFORM);
            result = make_float3(transform.x.w, transform.y.w, transform.z.w);
          }
          else {
            result = webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_OBJECT_COLOR));
          }
          stack_store_float(stack, node.out_offset, result.x);
          stack_store_float(stack, uint(node.out_offset) + 1, result.y);
          stack_store_float(stack, uint(node.out_offset) + 2, result.z);
          webgpu_svm_mark_outputs(
              written, varying, surface_normal_offset, SVM_STACK_INVALID, node.out_offset);
        }
        else {
          float value;
          if (node.info_type == NODE_INFO_OB_ALPHA) {
            value = __uint_as_float(surface(CYCLES_WEBGPU_SVM_OBJECT_COLOR).w);
          }
          else {
            /* Object Index, Material Index, and Random are consecutive SVM outputs. */
            static_assert(NODE_INFO_MAT_INDEX == NODE_INFO_OB_INDEX + 1 &&
                          NODE_INFO_OB_RANDOM == NODE_INFO_OB_INDEX + 2);
            const uint field = CYCLES_WEBGPU_SVM_OBJECT_INDEX +
                               (node.info_type - NODE_INFO_OB_INDEX);
            value = __uint_as_float(surface(field).w);
          }
          stack_store_float(stack, node.out_offset, value);
          webgpu_svm_mark_outputs(
              written, varying, surface_normal_offset, node.out_offset, SVM_STACK_INVALID);
        }
        break;
      }
      case NODE_FRESNEL: {
        SVMNodeFresnel node;
        if (!webgpu_svm_read(fetch, offset, end, node) || node.out_offset >= SVM_STACK_SIZE ||
            !webgpu_svm_input_valid(written, node.ior) ||
            (stack_valid(node.normal_offset) && !webgpu_svm_stack_has3(written, node.normal_offset)))
        {
          return false;
        }
        ShaderData sd;
        webgpu_svm_shader_view(surface, sd);
        svm_node_fresnel(&sd, stack, node);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, node.out_offset, SVM_STACK_INVALID);
        break;
      }
      case NODE_LAYER_WEIGHT: {
        SVMNodeLayerWeight node;
        if (!webgpu_svm_read(fetch, offset, end, node) || node.out_offset >= SVM_STACK_SIZE ||
            node.weight_type > NODE_LAYER_WEIGHT_FACING ||
            !webgpu_svm_input_valid(written, node.blend) ||
            (stack_valid(node.normal_offset) && !webgpu_svm_stack_has3(written, node.normal_offset)))
        {
          return false;
        }
        ShaderData sd;
        webgpu_svm_shader_view(surface, sd);
        svm_node_layer_weight(&sd, stack, node);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, node.out_offset, SVM_STACK_INVALID);
        break;
      }
      case NODE_LIGHT_PATH: {
        SVMNodeLightPath node;
        if (!webgpu_svm_read(fetch, offset, end, node) || node.path_type > NODE_LP_ray_portal ||
            node.out_offset >= SVM_STACK_SIZE)
        {
          return false;
        }
        const auto path = surface(CYCLES_WEBGPU_SVM_PATH);
        const auto hit = surface(CYCLES_WEBGPU_SVM_INTERSECTION);
        ShaderData sd;
        sd.runtime_flag = hit.w == 2 ? SR_BACKFACING : 0;
        sd.ray_length = __uint_as_float(hit.x);
        const WebGPUSvmPath state{surface(CYCLES_WEBGPU_SVM_BOUNCES), path.z};
        svm_node_light_path<KERNEL_FEATURE_NODE_LIGHT_PATH>(
            nullptr, state, &sd, stack, node, PathRayVisibility(path.x), path.y);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, node.out_offset, SVM_STACK_INVALID);
        break;
      }
      case NODE_LIGHT_FALLOFF: {
        SVMNodeLightFalloff node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            node.falloff_type > NODE_LIGHT_FALLOFF_CONSTANT || node.out_offset >= SVM_STACK_SIZE ||
            !webgpu_svm_inputs_valid(written, node.strength, node.smooth))
        {
          return false;
        }
        ShaderData sd;
        sd.ray_length = __uint_as_float(surface(CYCLES_WEBGPU_SVM_INTERSECTION).x);
        svm_node_light_falloff(&sd, stack, node);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, node.out_offset, SVM_STACK_INVALID);
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
      case NODE_MAPPING: {
        SVMNodeMapping node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            node.mapping_type > NODE_MAPPING_TYPE_NORMAL ||
            !webgpu_svm_inputs_valid(written, node.vector, node.location, node.rotation, node.scale) ||
            uint(node.result_offset) + 2 >= SVM_STACK_SIZE)
        {
          return false;
        }
        svm_node_mapping<float3>(stack, node);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, SVM_STACK_INVALID, node.result_offset);
        break;
      }
      case NODE_TEXTURE_MAPPING: {
        SVMNodeTextureMapping node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            !webgpu_svm_stack_has3(written, node.vec_offset) ||
            uint(node.out_offset) + 2 >= SVM_STACK_SIZE)
        {
          return false;
        }
        svm_node_texture_mapping<float3>(stack, node);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, SVM_STACK_INVALID, node.out_offset);
        break;
      }
      case NODE_MIN_MAX: {
        SVMNodeMinMax node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            !webgpu_svm_stack_has3(written, node.vec_offset) ||
            uint(node.out_offset) + 2 >= SVM_STACK_SIZE)
        {
          return false;
        }
        svm_node_min_max(stack, node);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, SVM_STACK_INVALID, node.out_offset);
        break;
      }
#ifndef CYCLES_WEBGPU_SVM_CORE
      case NODE_TEX_NOISE: {
        material.requires_extended_svm = true;
        SVMNodeTexNoise node;
        if (!webgpu_svm_read(fetch, offset, end, node) || node.dimensions < 1 ||
            node.dimensions > 4 || node.noise_type > NODE_NOISE_HETERO_TERRAIN ||
            !webgpu_svm_stack_has3(written, node.vector) ||
            !webgpu_svm_inputs_valid(written, node.w, node.scale, node.detail, node.roughness,
                                     node.lacunarity, node.offset, node.gain, node.distortion) ||
            !webgpu_svm_outputs_valid(node.value_offset, node.color_offset))
        {
          return false;
        }
        svm_node_tex_noise(stack, node);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, node.value_offset, node.color_offset);
        break;
      }
      case NODE_TEX_GABOR: {
        material.requires_extended_svm = true;
        SVMNodeTexGabor node;
        if (!webgpu_svm_read(fetch, offset, end, node) || node.gabor_type > NODE_GABOR_TYPE_3D ||
            !webgpu_svm_stack_has3(written, node.coordinates) ||
            !webgpu_svm_inputs_valid(written, node.scale, node.frequency, node.anisotropy,
                                     node.orientation_2d, node.orientation_3d) ||
            !webgpu_svm_outputs_valid(node.value_offset, SVM_STACK_INVALID) ||
            !webgpu_svm_outputs_valid(node.phase_offset, SVM_STACK_INVALID) ||
            !webgpu_svm_outputs_valid(node.intensity_offset, SVM_STACK_INVALID))
        {
          return false;
        }
        /* Evaluate Blender's Gabor kernels with scalar stack accesses. The native
         * combined helper can merge the linked and inline orientation pointers. */
        const float3 co = stack_load_float3(stack, node.coordinates) * stack_load(stack, node.scale);
        const float frequency = max(0.001f, stack_load(stack, node.frequency));
        const float isotropy = 1.0f - saturatef(stack_load(stack, node.anisotropy));
        float2 phasor;
        float standard_deviation;
        if (node.gabor_type == NODE_GABOR_TYPE_2D) {
          phasor = compute_2d_gabor_noise(make_float2(co.x, co.y),
                                          frequency,
                                          isotropy,
                                          stack_load(stack, node.orientation_2d));
          standard_deviation = compute_2d_gabor_standard_deviation();
        }
        else {
          float3 orientation;
          /* Keep stack and inline constants separate so SPIR-V can lower the loads. */
          if (!webgpu_svm_surface_color(node.orientation_3d, written, stack, orientation)) {
            return false;
          }
          phasor = compute_3d_gabor_noise(co, frequency, isotropy, normalize(orientation));
          standard_deviation = compute_3d_gabor_standard_deviation();
        }
        const float normalization_factor = 6.0f * standard_deviation;
        if (stack_valid(node.value_offset)) {
          stack_store_float(stack, node.value_offset,
                             (phasor.y / normalization_factor) * 0.5f + 0.5f);
        }
        if (stack_valid(node.phase_offset)) {
          stack_store_float(stack,
                             node.phase_offset,
                             (atan2f(phasor.y, phasor.x) + M_PI_F) / (2.0f * M_PI_F));
        }
        if (stack_valid(node.intensity_offset)) {
          stack_store_float(stack, node.intensity_offset, len(phasor) / normalization_factor);
        }
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, node.value_offset, SVM_STACK_INVALID);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, node.phase_offset, SVM_STACK_INVALID);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, node.intensity_offset, SVM_STACK_INVALID);
        break;
      }
#endif
      case NODE_TEX_VORONOI: {
        SVMNodeTexVoronoi node;
        if (!webgpu_svm_read(fetch, offset, end, node)) {
          return false;
        }
        if (node.dimensions == 3 && node.feature == NODE_VORONOI_F1 &&
            node.metric == NODE_VORONOI_EUCLIDEAN && !webgpu_svm_input_on_stack(node.detail) &&
            __uint_as_float(node.detail.bits) <= 0.0f && !webgpu_svm_input_on_stack(node.scale) &&
            !webgpu_svm_input_on_stack(node.randomness) &&
            webgpu_svm_finite(__uint_as_float(node.scale.bits)) &&
            webgpu_svm_finite(__uint_as_float(node.randomness.bits)) &&
            node.distance_offset == SVM_STACK_INVALID && node.position_offset == SVM_STACK_INVALID &&
            node.w_out_offset == SVM_STACK_INVALID && node.radius_offset == SVM_STACK_INVALID &&
            uint(node.color_offset) + 2 < SVM_STACK_SIZE &&
            webgpu_svm_stack_has3(written, node.coord))
        {
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
          webgpu_svm_mark_outputs(
              written, varying, surface_normal_offset, SVM_STACK_INVALID, node.color_offset);
          break;
        }
#ifndef CYCLES_WEBGPU_SVM_CORE
        if (node.dimensions < 1 || node.dimensions > 4 ||
            node.feature > NODE_VORONOI_N_SPHERE_RADIUS ||
            node.metric > NODE_VORONOI_MINKOWSKI || !webgpu_svm_stack_has3(written, node.coord) ||
            !webgpu_svm_inputs_valid(written, node.w, node.scale, node.detail, node.roughness,
                                     node.lacunarity, node.smoothness, node.exponent, node.randomness) ||
            !webgpu_svm_outputs_valid(node.distance_offset, node.color_offset) ||
            !webgpu_svm_outputs_valid(node.w_out_offset, node.position_offset) ||
            !webgpu_svm_outputs_valid(node.radius_offset, SVM_STACK_INVALID))
        {
          return false;
        }
        material.requires_extended_svm = true;
        svm_node_tex_voronoi<KERNEL_FEATURE_NODE_VORONOI_EXTRA>(stack, node);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, node.distance_offset, node.color_offset);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, node.w_out_offset, node.position_offset);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, node.radius_offset, SVM_STACK_INVALID);
        break;
#else
        return false;
#endif
      }
      case NODE_TEX_BRICK: {
        SVMNodeTexBrick node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            !webgpu_svm_stack_has3(written, node.co) ||
            !webgpu_svm_inputs_valid(written,
                                     node.color1,
                                     node.color2,
                                     node.mortar,
                                     node.scale,
                                     node.mortar_size,
                                     node.mortar_smooth,
                                     node.bias,
                                     node.brick_width,
                                     node.row_height) ||
            !webgpu_svm_outputs_valid(node.fac_offset, node.color_offset))
        {
          return false;
        }
        svm_node_tex_brick(stack, node);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, node.fac_offset, node.color_offset);
        break;
      }
      case NODE_TEX_CHECKER: {
        SVMNodeTexChecker node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            !webgpu_svm_stack_has3(written, node.co) ||
            !webgpu_svm_inputs_valid(written, node.color1, node.color2, node.scale) ||
            !webgpu_svm_outputs_valid(node.fac_offset, node.color_offset))
        {
          return false;
        }
        svm_node_tex_checker(stack, node);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, node.fac_offset, node.color_offset);
        break;
      }
      case NODE_TEX_GRADIENT: {
        SVMNodeTexGradient node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            node.gradient_type > NODE_BLEND_SPHERICAL ||
            !webgpu_svm_stack_has3(written, node.co) ||
            !webgpu_svm_outputs_valid(node.fac_offset, node.color_offset))
        {
          return false;
        }
        /* Use word indices for the adjacent vector slots. The native stack
         * store combines them into a pointer cast for this packed node. */
        const float factor = saturatef(svm_gradient(stack_load_float3(stack, node.co),
                                                   node.gradient_type));
        if (stack_valid(node.fac_offset)) {
          stack_store_float(stack, node.fac_offset, factor);
        }
        if (stack_valid(node.color_offset)) {
          stack_store_float(stack, node.color_offset, factor);
          stack_store_float(stack, uint(node.color_offset) + 1, factor);
          stack_store_float(stack, uint(node.color_offset) + 2, factor);
        }
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, node.fac_offset, node.color_offset);
        break;
      }
      case NODE_TEX_MAGIC: {
        SVMNodeTexMagic node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            !webgpu_svm_stack_has3(written, node.co) ||
            !webgpu_svm_inputs_valid(written, node.scale, node.distortion) ||
            !webgpu_svm_outputs_valid(node.fac_offset, node.color_offset))
        {
          return false;
        }
        svm_node_tex_magic(stack, node);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, node.fac_offset, node.color_offset);
        break;
      }
      case NODE_TEX_WAVE: {
        SVMNodeTexWave node;
        if (!webgpu_svm_read(fetch, offset, end, node) || node.wave_type > NODE_WAVE_RINGS ||
            node.bands_direction > NODE_WAVE_BANDS_DIRECTION_DIAGONAL ||
            node.rings_direction > NODE_WAVE_RINGS_DIRECTION_SPHERICAL ||
            node.profile > NODE_WAVE_PROFILE_TRI ||
            !webgpu_svm_stack_has3(written, node.co) ||
            !webgpu_svm_inputs_valid(written,
                                     node.scale,
                                     node.distortion,
                                     node.detail,
                                     node.dscale,
                                     node.droughness,
                                     node.phase) ||
            !webgpu_svm_outputs_valid(node.fac_offset, node.color_offset))
        {
          return false;
        }
        svm_node_tex_wave(stack, node);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, node.fac_offset, node.color_offset);
        break;
      }
      case NODE_TEX_WHITE_NOISE: {
        SVMNodeTexWhiteNoise node;
        if (!webgpu_svm_read(fetch, offset, end, node) || node.dimensions < 1 ||
            node.dimensions > 4 || !webgpu_svm_inputs_valid(written, node.vector, node.w) ||
            !webgpu_svm_outputs_valid(node.value_offset, node.color_offset))
        {
          return false;
        }
        svm_node_tex_white_noise(stack, node);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, node.value_offset, node.color_offset);
        break;
      }
      case NODE_MATH: {
        SVMNodeMath node;
        if (!webgpu_svm_read(fetch, offset, end, node) || node.result_offset >= SVM_STACK_SIZE ||
            node.math_type > NODE_MATH_FLOORED_MODULO ||
            !webgpu_svm_inputs_valid(written, node.value1, node.value2, node.value3))
        {
          return false;
        }
        svm_node_math(stack, node);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, node.result_offset, SVM_STACK_INVALID);
        break;
      }
      case NODE_VECTOR_MATH: {
        SVMNodeVectorMath node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            node.math_type > NODE_VECTOR_MATH_ROUND ||
            !webgpu_svm_inputs_valid(written, node.a, node.b, node.c, node.param1) ||
            !webgpu_svm_outputs_valid(node.value_offset, node.vector_offset))
        {
          return false;
        }
        svm_node_vector_math<float3>(stack, node);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, node.value_offset, node.vector_offset);
        break;
      }
#ifndef CYCLES_WEBGPU_SVM_CORE
      case NODE_CLAMP: {
        material.requires_extended_svm = true;
        SVMNodeClamp node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            node.clamp_type > NODE_CLAMP_RANGE ||
            !webgpu_svm_inputs_valid(written, node.value, node.min, node.max) ||
            node.result_offset >= SVM_STACK_SIZE)
        {
          return false;
        }
        svm_node_clamp(stack, node);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, node.result_offset, SVM_STACK_INVALID);
        break;
      }
      case NODE_GAMMA: {
        material.requires_extended_svm = true;
        SVMNodeGamma node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            !webgpu_svm_inputs_valid(written, node.color, node.gamma) ||
            !webgpu_svm_outputs_valid(SVM_STACK_INVALID, node.out_offset))
        {
          return false;
        }
        svm_node_gamma(stack, node);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, SVM_STACK_INVALID, node.out_offset);
        break;
      }
      case NODE_BRIGHTCONTRAST: {
        material.requires_extended_svm = true;
        SVMNodeBrightContrast node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            !webgpu_svm_inputs_valid(written, node.color, node.bright, node.contrast) ||
            !webgpu_svm_outputs_valid(SVM_STACK_INVALID, node.out_offset))
        {
          return false;
        }
        svm_node_brightness(stack, node);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, SVM_STACK_INVALID, node.out_offset);
        break;
      }
      case NODE_HSV: {
        material.requires_extended_svm = true;
        SVMNodeHSV node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            !webgpu_svm_inputs_valid(written, node.color, node.hue, node.sat, node.val, node.fac) ||
            !webgpu_svm_outputs_valid(SVM_STACK_INVALID, node.out_color_offset))
        {
          return false;
        }
        svm_node_hsv(stack, node);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, SVM_STACK_INVALID, node.out_color_offset);
        break;
      }
      case NODE_INVERT: {
        material.requires_extended_svm = true;
        SVMNodeInvert node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            !webgpu_svm_inputs_valid(written, node.color, node.fac) ||
            !webgpu_svm_outputs_valid(SVM_STACK_INVALID, node.out_offset))
        {
          return false;
        }
        /* Keep the inline color and stack loads as values. LLVM otherwise
         * merges their pointers before the SPIR-V backend. */
        if (stack_valid(node.out_offset)) {
          const uint base = node.color.x.bits & 0xffu;
          const float factor = stack_load(stack, node.fac);
          float3 color;
          if (webgpu_svm_input_on_stack(node.color.x)) {
            color = make_float3(stack_load_float(stack, base),
                                stack_load_float(stack, base + 1),
                                stack_load_float(stack, base + 2));
          }
          else {
            color = make_float3(__uint_as_float(node.color.x.bits),
                                __uint_as_float(node.color.y.bits),
                                __uint_as_float(node.color.z.bits));
          }
          stack_store_float(stack, node.out_offset, invert(color.x, factor));
          stack_store_float(stack, uint(node.out_offset) + 1, invert(color.y, factor));
          stack_store_float(stack, uint(node.out_offset) + 2, invert(color.z, factor));
        }
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, SVM_STACK_INVALID, node.out_offset);
        break;
      }
      case NODE_MIX: {
        material.requires_extended_svm = true;
        SVMNodeMix node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            node.mix_type > NODE_MIX_CLAMP ||
            !webgpu_svm_inputs_valid(written, node.c1, node.c2, node.fac) ||
            uint(node.result_offset) + 2 >= SVM_STACK_SIZE)
        {
          return false;
        }
        svm_node_mix(stack, node);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, SVM_STACK_INVALID, node.result_offset);
        break;
      }
      case NODE_MIX_COLOR: {
        material.requires_extended_svm = true;
        SVMNodeMixColor node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            node.blend_type > NODE_MIX_EXCLUSION ||
            !webgpu_svm_inputs_valid(written, node.a, node.b, node.fac) ||
            uint(node.result_offset) + 2 >= SVM_STACK_SIZE)
        {
          return false;
        }
        svm_node_mix_color(stack, node);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, SVM_STACK_INVALID, node.result_offset);
        break;
      }
      case NODE_MIX_FLOAT: {
        material.requires_extended_svm = true;
        SVMNodeMixFloat node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            !webgpu_svm_inputs_valid(written, node.a, node.b, node.fac) ||
            node.result_offset >= SVM_STACK_SIZE)
        {
          return false;
        }
        svm_node_mix_float(stack, node);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, node.result_offset, SVM_STACK_INVALID);
        break;
      }
      case NODE_MIX_VECTOR: {
        material.requires_extended_svm = true;
        SVMNodeMixVector node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            !webgpu_svm_inputs_valid(written, node.a, node.b, node.fac) ||
            uint(node.result_offset) + 2 >= SVM_STACK_SIZE)
        {
          return false;
        }
        svm_node_mix_vector(stack, node);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, SVM_STACK_INVALID, node.result_offset);
        break;
      }
      case NODE_MIX_VECTOR_NON_UNIFORM: {
        material.requires_extended_svm = true;
        SVMNodeMixVectorNonUniform node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            !webgpu_svm_inputs_valid(written, node.a, node.b, node.fac) ||
            uint(node.result_offset) + 2 >= SVM_STACK_SIZE)
        {
          return false;
        }
        svm_node_mix_vector_non_uniform(stack, node);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, SVM_STACK_INVALID, node.result_offset);
        break;
      }
      case NODE_COMBINE_COLOR: {
        material.requires_extended_svm = true;
        SVMNodeCombineColor node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            node.color_type > NODE_COMBSEP_COLOR_HSL ||
            !webgpu_svm_inputs_valid(written, node.red, node.green, node.blue) ||
            !webgpu_svm_outputs_valid(SVM_STACK_INVALID, node.color_offset))
        {
          return false;
        }
        svm_node_combine_color(stack, node);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, SVM_STACK_INVALID, node.color_offset);
        break;
      }
      case NODE_SEPARATE_VECTOR: {
        material.requires_extended_svm = true;
        SVMNodeSeparateVector node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            node.vector_index > 2 ||
            !webgpu_svm_inputs_valid(written, node.vector) ||
            !webgpu_svm_outputs_valid(node.out_offset, SVM_STACK_INVALID))
        {
          return false;
        }
        svm_node_separate_vector<float3>(stack, node);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, node.out_offset, SVM_STACK_INVALID);
        break;
      }
      case NODE_MAP_RANGE: {
        material.requires_extended_svm = true;
        SVMNodeMapRange node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            node.range_type > NODE_MAP_RANGE_SMOOTHERSTEP ||
            !webgpu_svm_inputs_valid(written,
                                     node.value,
                                     node.from_min,
                                     node.from_max,
                                     node.to_min,
                                     node.to_max,
                                     node.steps) ||
            node.result_offset >= SVM_STACK_SIZE)
        {
          return false;
        }
        svm_node_map_range(stack, node);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, node.result_offset, SVM_STACK_INVALID);
        break;
      }
      case NODE_VECTOR_MAP_RANGE: {
        material.requires_extended_svm = true;
        SVMNodeVectorMapRange node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            node.range_type > NODE_MAP_RANGE_SMOOTHERSTEP ||
            !webgpu_svm_inputs_valid(written,
                                     node.value,
                                     node.from_min,
                                     node.from_max,
                                     node.to_min,
                                     node.to_max,
                                     node.steps) ||
            uint(node.result_offset) + 2 >= SVM_STACK_SIZE)
        {
          return false;
        }
        svm_node_vector_map_range(stack, node);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, SVM_STACK_INVALID, node.result_offset);
        break;
      }
      case NODE_VECTOR_ROTATE: {
        material.requires_extended_svm = true;
        SVMNodeVectorRotate node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            node.rotate_type > NODE_VECTOR_ROTATE_TYPE_EULER_XYZ ||
            !webgpu_svm_inputs_valid(written, node.vector, node.center, node.axis, node.rotation, node.angle) ||
            !webgpu_svm_outputs_valid(SVM_STACK_INVALID, node.result_offset))
        {
          return false;
        }
        svm_node_vector_rotate(stack, node);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, SVM_STACK_INVALID, node.result_offset);
        break;
      }
      case NODE_NORMAL: {
        material.requires_extended_svm = true;
        SVMNodeNormal node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            !webgpu_svm_input_valid(written, node.in_normal) ||
            !webgpu_svm_outputs_valid(node.out_dot_offset, node.out_normal_offset))
        {
          return false;
        }
        svm_node_normal(stack, node);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, node.out_dot_offset, node.out_normal_offset);
        break;
      }
      case NODE_SEPARATE_COLOR: {
        material.requires_extended_svm = true;
        SVMNodeSeparateColor node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            node.color_type > NODE_COMBSEP_COLOR_HSL ||
            !webgpu_svm_input_valid(written, node.color) ||
            !webgpu_svm_outputs_valid(node.red_offset, SVM_STACK_INVALID) ||
            !webgpu_svm_outputs_valid(node.green_offset, SVM_STACK_INVALID) ||
            !webgpu_svm_outputs_valid(node.blue_offset, SVM_STACK_INVALID))
        {
          return false;
        }
        svm_node_separate_color(stack, node);
        for (const uint channel : {node.red_offset, node.green_offset, node.blue_offset}) {
          webgpu_svm_mark_outputs(written, varying, surface_normal_offset, channel, SVM_STACK_INVALID);
        }
        break;
      }
      case NODE_COMBINE_VECTOR: {
        material.requires_extended_svm = true;
        SVMNodeCombineVector node;
        if (!webgpu_svm_read(fetch, offset, end, node) || node.vector_index > 2 ||
            !webgpu_svm_input_valid(written, node.in) ||
            !webgpu_svm_outputs_valid(SVM_STACK_INVALID, node.out_offset))
        {
          return false;
        }
        /* The stock compiler emits one instruction per component. Keep
         * stack reads and writes on the same scalar array index. */
        if (stack_valid(node.out_offset)) {
          const uint component = uint(node.out_offset) + uint(node.vector_index);
          stack_store_float(stack, component, stack_load(stack, node.in));
          webgpu_svm_mark_outputs(
              written, varying, surface_normal_offset, component, SVM_STACK_INVALID);
        }
        break;
      }
#endif
      case NODE_CONVERT: {
        SVMNodeConvert node;
        if (!webgpu_svm_read(fetch, offset, end, node)) {
          return false;
        }
        if (node.convert_type == NODE_CONVERT_FV) {
          if (uint(node.to_offset) + 2 >= SVM_STACK_SIZE ||
              !webgpu_svm_stack_has(written, node.from_offset))
          {
            return false;
          }
          const float value = stack_load_float(stack, node.from_offset);
          stack_store_float(stack, node.to_offset, value);
          stack_store_float(stack, uint(node.to_offset) + 1, value);
          stack_store_float(stack, uint(node.to_offset) + 2, value);
          webgpu_svm_mark_outputs(
              written, varying, surface_normal_offset, SVM_STACK_INVALID, node.to_offset);
        }
        else {
          if ((node.convert_type != NODE_CONVERT_CF && node.convert_type != NODE_CONVERT_VF) ||
              node.to_offset >= SVM_STACK_SIZE || !webgpu_svm_stack_has3(written, node.from_offset))
          {
            return false;
          }
          const float3 value = stack_load_float3(stack, node.from_offset);
          /* The working-space coefficients are serialized from kernel data. */
          const float scalar = node.convert_type == NODE_CONVERT_CF ? dot(value, rgb_to_y) :
                                                                     average(value);
          stack_store_float(stack, node.to_offset, scalar);
          webgpu_svm_mark_outputs(
              written, varying, surface_normal_offset, node.to_offset, SVM_STACK_INVALID);
        }
        break;
      }
#ifndef CYCLES_WEBGPU_SVM_CORE
      case NODE_RGB_RAMP: {
        material.requires_extended_svm = true;
        SVMNodeRGBRamp node;
        if (!webgpu_svm_read(fetch, offset, end, node) || node.table_size == 0 ||
            node.table_size > (end - offset) / 4u ||
            !webgpu_svm_outputs_valid(node.alpha_offset, node.color_offset) ||
            !webgpu_svm_input_valid(written, node.fac))
        {
          return false;
        }
        const auto sample = [&](const uint index) {
          const uint word = offset + 4u * index;
          return make_float4(__uint_as_float(fetch(word)),
                             __uint_as_float(fetch(word + 1u)),
                             __uint_as_float(fetch(word + 2u)),
                             __uint_as_float(fetch(word + 3u)));
        };
        const float4 color = webgpu_svm_ramp_lookup(
            sample, node.table_size, stack_load(stack, node.fac), node.interpolate, false);
        if (stack_valid(node.color_offset)) {
          stack_store_float(stack, node.color_offset, color.x);
          stack_store_float(stack, uint(node.color_offset) + 1, color.y);
          stack_store_float(stack, uint(node.color_offset) + 2, color.z);
        }
        if (stack_valid(node.alpha_offset)) {
          stack_store_float(stack, node.alpha_offset, color.w);
        }
        offset += 4u * node.table_size;
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, node.alpha_offset, node.color_offset);
        break;
      }
      case NODE_CURVES: {
        material.requires_extended_svm = true;
        SVMNodeCurves node;
        if (!webgpu_svm_read(fetch, offset, end, node) || node.table_size < 2 ||
            node.table_size > (end - offset) / 4u ||
            !webgpu_svm_inputs_valid(written, node.color, node.fac) ||
            uint(node.out_offset) + 2 >= SVM_STACK_SIZE)
        {
          return false;
        }
        const float3 color = stack_load(stack, node.color);
        const float fraction = stack_load(stack, node.fac);
        const float3 position = (color - make_float3(node.min_x)) / (node.max_x - node.min_x);
        const auto evaluate = [&](const uint channel, const float value) {
          const auto sample = [&](const uint index) {
            return __uint_as_float(fetch(offset + 4u * index + channel));
          };
          return webgpu_svm_ramp_lookup(sample, node.table_size, value, true, node.extrapolate);
        };
        const float3 result = (1.0f - fraction) * color +
                              fraction * make_float3(evaluate(0, position.x),
                                                     evaluate(1, position.y),
                                                     evaluate(2, position.z));
        stack_store_float(stack, node.out_offset, result.x);
        stack_store_float(stack, uint(node.out_offset) + 1, result.y);
        stack_store_float(stack, uint(node.out_offset) + 2, result.z);
        offset += 4u * node.table_size;
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, SVM_STACK_INVALID, node.out_offset);
        break;
      }
      case NODE_FLOAT_CURVE: {
        material.requires_extended_svm = true;
        SVMNodeFloatCurve node;
        if (!webgpu_svm_read(fetch, offset, end, node) || node.table_size < 2 ||
            node.table_size > end - offset || node.out_offset >= SVM_STACK_SIZE ||
            !webgpu_svm_inputs_valid(written, node.value_in, node.fac))
        {
          return false;
        }
        const float value = stack_load(stack, node.value_in);
        const float fraction = stack_load(stack, node.fac);
        const float position = (value - node.min_x) / (node.max_x - node.min_x);
        const auto sample = [&](const uint index) {
          return __uint_as_float(fetch(offset + index));
        };
        const float curve = webgpu_svm_ramp_lookup(
            sample, node.table_size, position, true, node.extrapolate);
        stack_store_float(stack, node.out_offset, (1.0f - fraction) * value + fraction * curve);
        offset += node.table_size;
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, node.out_offset, SVM_STACK_INVALID);
        break;
      }
#endif
      case NODE_TEX_IMAGE: {
        SVMNodeTexImage node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            (node.projection != NODE_IMAGE_PROJ_FLAT
#ifndef CYCLES_WEBGPU_SVM_CORE
             && node.projection != NODE_IMAGE_PROJ_SPHERE && node.projection != NODE_IMAGE_PROJ_TUBE
#endif
             ) ||
            !webgpu_svm_eval_image(image, node, stack, written, varying, surface_normal_offset))
        {
          return false;
        }
        material.requires_extended_svm |= node.projection != NODE_IMAGE_PROJ_FLAT;
        break;
      }
#ifndef CYCLES_WEBGPU_SVM_CORE
      case NODE_TEX_ENVIRONMENT: {
        SVMNodeTexEnvironment node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            (node.projection != NODE_ENVIRONMENT_EQUIRECTANGULAR &&
             node.projection != NODE_ENVIRONMENT_MIRROR_BALL) ||
            !webgpu_svm_eval_image(image, node, stack, written, varying, surface_normal_offset))
        {
          return false;
        }
        material.requires_extended_svm = true;
        break;
      }
#endif
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
      case NODE_MIX_CLOSURE: {
        SVMNodeMixClosure node;
        if ((has_closure && lambert_closures == 0) ||
            !webgpu_svm_read(fetch, offset, end, node) ||
            !webgpu_svm_input_valid(written, node.fac) ||
            (stack_valid(node.in_weight_offset) &&
             !webgpu_svm_stack_has(written, node.in_weight_offset)) ||
            !webgpu_svm_outputs_valid(node.weight1_offset, SVM_STACK_INVALID) ||
            !webgpu_svm_outputs_valid(node.weight2_offset, SVM_STACK_INVALID))
        {
          return false;
        }
        /* Evaluate the scalar part of svm_node_mix_closure() on the packed stack. */
        const float fac = saturatef(stack_load(stack, node.fac));
        const float weight_in = stack_load_float_default(stack, node.in_weight_offset, 1.0f);
        if (stack_valid(node.weight1_offset)) {
          stack_store_float(stack, node.weight1_offset, weight_in * (1.0f - fac));
        }
        if (stack_valid(node.weight2_offset)) {
          stack_store_float(stack, node.weight2_offset, weight_in * fac);
        }
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, node.weight1_offset, SVM_STACK_INVALID);
        webgpu_svm_mark_outputs(
            written, varying, surface_normal_offset, node.weight2_offset, SVM_STACK_INVALID);
        mixes_closures = true;
        break;
      }
      case NODE_JUMP_IF_ZERO:
      case NODE_JUMP_IF_ONE: {
        static_assert(sizeof(SVMNodeJumpIfZero) == sizeof(SVMNodeJumpIfOne));
        static_assert(offsetof(SVMNodeJumpIfZero, jump_offset) ==
                      offsetof(SVMNodeJumpIfOne, jump_offset));
        static_assert(offsetof(SVMNodeJumpIfZero, stack_offset) ==
                      offsetof(SVMNodeJumpIfOne, stack_offset));
        SVMNodeJumpIfZero jump;
        if ((has_closure && lambert_closures == 0) ||
            !webgpu_svm_read(fetch, offset, end, jump) ||
            !webgpu_svm_stack_has(written, jump.stack_offset) || jump.jump_offset < 0 ||
            uint(jump.jump_offset) > end - offset)
        {
          return false;
        }
        /* These stock branches skip zero-weight closures. Visit both so host
         * eligibility validates every closure, including those outside the probe point. */
        mixes_closures = true;
        break;
      }
      case NODE_CLOSURE_BSDF: {
        SVMNodeClosureBsdf closure;
        if (!webgpu_svm_read(fetch, offset, end, closure)) {
          return false;
        }
        const bool lambert = closure.closure_type == CLOSURE_BSDF_DIFFUSE_ID;
        if ((has_closure && lambert_closures == 0) ||
            ((has_closure || mixes_closures) && !lambert) || lambert_closures == MAX_CLOSURE ||
            (closure.mix_weight_offset != SVM_STACK_INVALID &&
             (!webgpu_svm_stack_has(written, closure.mix_weight_offset) ||
              (!lambert && webgpu_svm_stack_has(varying, closure.mix_weight_offset)))))
        {
          return false;
        }
        const float mix_weight = stack_load_float_default(stack, closure.mix_weight_offset, 1.0f);
        material.ggx = false;
        material.standalone_transmission = 0.0f;
        material.transmission = 0.0f;
        material.subsurface = 0.0f;
        Spectrum closure_weight;
        const Spectrum previous = lambert_closures != 0 ? material.color : zero_spectrum();
        if (lambert) {
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
        else if (closure.closure_type == CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID ||
                 closure.closure_type == CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID)
        {
          SVMNodeRefractionBsdfData refraction;
          if (!webgpu_svm_read(fetch, offset, end, refraction) ||
              !webgpu_svm_refraction(
                  refraction, written, stack, surface_normal_offset, weight, mix_weight,
                  closure.closure_type == CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID, material))
          {
            return false;
          }
          has_closure = true;
          break;
        }
        else if (closure.closure_type == CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID ||
                 closure.closure_type == CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID ||
                 closure.closure_type == CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID)
        {
          SVMNodeGlassBsdfData glass;
          if (!webgpu_svm_read(fetch, offset, end, glass) ||
              !webgpu_svm_glass(glass,
                                written,
                                stack,
                                surface_normal_offset,
                                mix_weight,
                                ClosureType(closure.closure_type),
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
        /* Lambert closures with the same surface normal have the same cosine PDF. */
        material.color = previous;
        if (closure_sample_weight(0, closure_weight) > 0.0f) {
          material.color += closure_weight;
        }
        if (!webgpu_svm_finite(average(material.color))) {
          return false;
        }
        has_closure = true;
        lambert_closures += uint(lambert);
        break;
      }
      default:
        return false;
    }
  }
  return false;
}

CCL_NAMESPACE_END
