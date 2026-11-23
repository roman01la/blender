/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "material_layout.h"
#include "svm_image_projection.h"
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
#include "kernel/closure/bsdf_transparent.h"
#include "kernel/closure/bsdf_ray_portal.h"
#include "kernel/closure/bsdf_util.h"
#ifdef CYCLES_WEBGPU_SAMPLED_EMISSION
#  include "kernel/svm/webgpu_sampled_closure_skip.h"
#endif
#include "kernel/util/colorspace.h"
#include "kernel/util/differential.h"
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
 * count, attribute slot and triangle counts, the IES table offset (or the
 * lookup descriptor in scenes with Sheen or coated emission), the luminance coefficients, three XYZ to
 * RGB rows and the stock CIE color matching table. UV, Generated and attribute
 * records precede the image records and their texels. The IES table follows. */
constexpr uint webgpu_svm_ies_offset = 5;
constexpr uint webgpu_svm_lookup_ies = 0;
constexpr uint webgpu_svm_lookup_sheen = 1;
constexpr uint webgpu_svm_lookup_words = 2;
constexpr uint webgpu_svm_color_space_offset = 6;
constexpr uint webgpu_svm_cie_table_offset = 18;
constexpr uint webgpu_svm_cie_table_words = (WAVELENGTH_RESOLUTION + 1) * 3;
/* KernelIntegrator::sample_clamp_indirect, which surface emission uses after
 * the first bounce. */
constexpr uint webgpu_svm_indirect_clamp_offset = webgpu_svm_cie_table_offset +
                                                  webgpu_svm_cie_table_words;
constexpr uint webgpu_svm_texture_header_words = webgpu_svm_indirect_clamp_offset + 1;
constexpr uint webgpu_svm_triangle_uv_words = 6;
constexpr uint webgpu_svm_triangle_generated_words = 9;
constexpr uint webgpu_svm_image_words = 5;
/* Per attribute slot and triangle: mode and NodeAttributeType, then three
 * float4 corner values converted as attribute_data_fetch() returns them. */
constexpr uint webgpu_svm_attribute_words = CYCLES_WEBGPU_SVM_ATTRIBUTE_WORDS;
constexpr uint webgpu_svm_attribute_slots = 16;
/* Attribute IDs stay below the slot flag. */
constexpr uint webgpu_svm_attribute_slot = CYCLES_WEBGPU_SVM_ATTRIBUTE_SLOT;
enum WebGPUSvmAttributeMode : uint {
  WEBGPU_SVM_ATTRIBUTE_MISSING = 0,
  WEBGPU_SVM_ATTRIBUTE_INTERPOLATE = 1,
  WEBGPU_SVM_ATTRIBUTE_CONSTANT = 2,
};
/* Host-built slots for kernel data other than attributes. Smooth normals hold
 * the object-space corner normals triangle_smooth_normal_unnormalized_object_space()
 * interpolates. Positions hold triangle_vertices() and flag objects whose
 * transform is not applied. */
enum WebGPUSvmSyntheticSlot : uint {
  WEBGPU_SVM_SLOT_SMOOTH_NORMAL = 1,
  WEBGPU_SVM_SLOT_POSITIONS = 2,
  WEBGPU_SVM_SLOT_ORIGINAL_NORMAL = 3,
};
constexpr uint webgpu_svm_slot_object_transform = 1u << 16;
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

/* KernelFilm's working-space conversions used by the stock color nodes. */
struct WebGPUSvmColorSpace {
  float3 rgb_to_y;
  float3 xyz_to_r;
  float3 xyz_to_g;
  float3 xyz_to_b;
};

template<typename Fetch>
ccl_device_inline WebGPUSvmColorSpace webgpu_svm_read_color_space(Fetch word, const uint texture)
{
  const uint offset = texture + webgpu_svm_color_space_offset;
  return {webgpu_svm_read_rgb(word, offset),
          webgpu_svm_read_rgb(word, offset + 3),
          webgpu_svm_read_rgb(word, offset + 6),
          webgpu_svm_read_rgb(word, offset + 9)};
}

struct WebGPUSvmAttributeSample {
  uint mode;
  uint type;
  float4 value;
};

struct WebGPUSvmTriangleCorners {
  uint flags;
  float3 corners[3];
};

template<typename Fetch>
ccl_device_inline uint webgpu_svm_attribute_table(Fetch word, const uint texture)
{
  return texture + webgpu_svm_texture_header_words + word(texture) * webgpu_svm_triangle_uv_words +
         word(texture + 2) * webgpu_svm_triangle_generated_words;
}

template<typename Fetch>
ccl_device_inline uint webgpu_svm_attribute_record(Fetch word,
                                                   const uint texture,
                                                   const uint slot,
                                                   const uint primitive)
{
  const uint triangles = word(texture + 4);
  if (slot >= word(texture + 3) || primitive >= triangles) {
    return 0;
  }
  return webgpu_svm_attribute_table(word, texture) +
         (slot * triangles + primitive) * webgpu_svm_attribute_words;
}

/* Raw corners of a host-built slot. */
template<typename Fetch>
ccl_device_inline WebGPUSvmTriangleCorners webgpu_svm_triangle_corners(Fetch word,
                                                                       const uint texture,
                                                                       const uint slot,
                                                                       const uint primitive)
{
  const uint base = webgpu_svm_attribute_record(word, texture, slot, primitive);
  if (base == 0) {
    return {0, {zero_float3(), zero_float3(), zero_float3()}};
  }
  return {word(base) >> 16,
          {webgpu_svm_read_rgb(word, base + 1),
           webgpu_svm_read_rgb(word, base + 5),
           webgpu_svm_read_rgb(word, base + 9)}};
}

/* Interpolate one packed attribute slot like triangle_attribute(). */
template<typename Fetch>
ccl_device_inline WebGPUSvmAttributeSample webgpu_svm_triangle_attribute(Fetch word,
                                                                         const uint texture,
                                                                         const uint slot,
                                                                         const uint primitive,
                                                                         const float u,
                                                                         const float v,
                                                                         const float du = 0.0f,
                                                                         const float dv = 0.0f,
                                                                         const float filter_width = 0.0f)
{
  const uint triangles = word(texture + 4);
  if (slot >= word(texture + 3) || primitive >= triangles) {
    return {WEBGPU_SVM_ATTRIBUTE_MISSING, 0, zero_float4()};
  }
  const uint base = webgpu_svm_attribute_table(word, texture) +
                    (slot * triangles + primitive) * webgpu_svm_attribute_words;
  const uint header = word(base);
  const auto corner = [&](const uint i) {
    const uint offset = base + 1 + 4 * i;
    return make_float4(__uint_as_float(word(offset)),
                       __uint_as_float(word(offset + 1)),
                       __uint_as_float(word(offset + 2)),
                       __uint_as_float(word(offset + 3)));
  };
  const uint mode = header & 0xffu;
  const uint type = (header >> 8) & 0xffu;
  const float4 f0 = corner(0);
  if (mode != WEBGPU_SVM_ATTRIBUTE_INTERPOLATE) {
    return {mode, type, f0};
  }
  /* Match triangle_interpolate() and triangle_attribute() derivatives. */
  const float4 f1 = corner(1);
  const float4 f2 = corner(2);
  const float4 value = (1.0f - u - v) * f0 + u * f1 + v * f2;
  return {mode, type, value + filter_width * ((f1 - f0) * du + (f2 - f0) * dv)};
}

/* svm_node_attr_surface_eval() conversions for a found surface attribute. */
ccl_device_inline bool webgpu_svm_attribute_value(const WebGPUSvmAttributeSample &sample,
                                                  const NodeAttributeOutputType type,
                                                  float3 &value)
{
  const float4 f = sample.value;
  switch (sample.type) {
    case NODE_ATTR_FLOAT:
      value = type == NODE_ATTR_OUTPUT_FLOAT_ALPHA ? make_float3(1.0f) :
                                                     make_float3(f.x, f.x, f.x);
      return true;
    case NODE_ATTR_FLOAT2:
      value = type == NODE_ATTR_OUTPUT_FLOAT       ? make_float3(f.x) :
              type == NODE_ATTR_OUTPUT_FLOAT_ALPHA ? make_float3(1.0f) :
                                                     make_float3(f.x, f.y, 0.0f);
      return true;
    case NODE_ATTR_FLOAT4:
    case NODE_ATTR_RGBA:
      value = type == NODE_ATTR_OUTPUT_FLOAT       ? make_float3(average(make_float3(f))) :
              type == NODE_ATTR_OUTPUT_FLOAT_ALPHA ? make_float3(f.w) :
                                                     make_float3(f);
      return true;
    case NODE_ATTR_FLOAT3:
      value = type == NODE_ATTR_OUTPUT_FLOAT       ? make_float3(average(make_float3(f))) :
              type == NODE_ATTR_OUTPUT_FLOAT_ALPHA ? make_float3(1.0f) :
                                                     make_float3(f);
      return true;
    default:
      return false;
  }
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

#if !defined(CYCLES_WEBGPU_SVM_CORE) || defined(CYCLES_WEBGPU_SVM_PROJECTED_IMAGES) || \
    defined(CYCLES_WEBGPU_SHADOW_SVM)
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
#if !defined(CYCLES_WEBGPU_SVM_CORE) || defined(CYCLES_WEBGPU_SVM_PROJECTED_IMAGES) || \
    defined(CYCLES_WEBGPU_SHADOW_SVM)
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
  const uint table = webgpu_svm_attribute_table(word, texture) +
                     word(texture + 3) * word(texture + 4) * webgpu_svm_attribute_words;
  /* Index the synchronized image slot directly in the SVM loop. */
  const uint base = table + uint(image) * webgpu_svm_image_words;
  const uint width = word(base);
  if (width == 0) {
    return false;
  }
  const uint height = word(base + 1);
  const uint extension = word(base + 2);
  const uint interpolation = word(base + 3);
#if defined(CYCLES_WEBGPU_SVM_CORE) && !defined(CYCLES_WEBGPU_SVM_PROJECTED_IMAGES) && \
    !defined(CYCLES_WEBGPU_SHADOW_SVM)
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

struct WebGPUSvmNormalSlots {
  uint active = SVM_STACK_INVALID;
  uint base = SVM_STACK_INVALID;
  uint bump = SVM_STACK_INVALID;
  uint previous_bump = SVM_STACK_INVALID;
  uint bump_word = 0;
  uint previous_bump_word = 0;

  ccl_device_inline_method void invalidate(const uint first, const uint width)
  {
    if (first < active + 3 && first + width > active) {
      active = SVM_STACK_INVALID;
    }
    if (first < base + 3 && first + width > base) {
      base = SVM_STACK_INVALID;
    }
    if (first < bump + 3 && first + width > bump) {
      bump = SVM_STACK_INVALID;
    }
    if (first < previous_bump + 3 && first + width > previous_bump) {
      previous_bump = SVM_STACK_INVALID;
    }
  }

  ccl_device_inline_method void set_bump(const uint slot, const uint word)
  {
    /* The SVM compiler can recycle a stack address across branch closures.
     * Identify its value by the defining bytecode instruction. */
    if (stack_valid(bump)) {
      previous_bump = bump;
      previous_bump_word = bump_word;
    }
    active = bump = slot;
    bump_word = word;
  }

  ccl_device_inline_method uint find_bump(const uint slot) const
  {
    if (!stack_valid(slot)) {
      return 0;
    }
    return slot == bump ? bump_word : slot == previous_bump ? previous_bump_word : 0;
  }
};

/* A procedural result can vary even when host eligibility happens to sample a
 * constant-looking point. Do not accept it where a uniform closure is required. */
ccl_device_inline void webgpu_svm_mark_outputs(WebGPUSvmStackUsage &written,
                                               WebGPUSvmStackUsage &varying,
                                               WebGPUSvmNormalSlots &surface_normals,
                                               const uint scalar,
                                               const uint vector)
{
  if (stack_valid(scalar)) {
    webgpu_svm_stack_mark(written, scalar);
    webgpu_svm_stack_mark(varying, scalar);
    surface_normals.invalidate(scalar, 1);
  }
  if (stack_valid(vector)) {
    webgpu_svm_stack_mark3(written, vector);
    webgpu_svm_stack_mark3(varying, vector);
    surface_normals.invalidate(vector, 3);
  }
}

ccl_device_inline bool webgpu_svm_surface_normal(const uint normal, const uint surface_normal)
{
  return normal == SVM_STACK_INVALID || normal == surface_normal;
}

template<typename Image, typename Node>
ccl_device_inline bool webgpu_svm_fetch_image(Image image,
                                              const Node &node,
                                              const float2 uv,
                                              float4 &color)
{
  const bool unassociate = node.flags & NODE_IMAGE_ALPHA_UNASSOCIATE;
  if (!image(node.id, uv, unassociate || stack_valid(node.alpha_offset), color)) {
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
  }
  return true;
}

ccl_device_inline float2 webgpu_svm_environment_coordinates(const float3 co,
                                                                 const NodeEnvironmentProjection projection)
{
  const float3 direction = safe_normalize(co);
  if (projection == NODE_ENVIRONMENT_MIRROR_BALL) {
    return direction_to_mirrorball(direction);
  }
  if (direction.x == 0.0f && direction.y == 0.0f && direction.z != 0.0f) {
    /* WGSL atan2 is undefined at the poles. Equirectangular images repeat. */
    const bool negative_x = (__float_as_uint(direction.x) >> 31) != 0;
    return make_float2(negative_x ? 0.0f : 0.5f, direction.z > 0.0f ? 1.0f : 0.0f);
  }
  return direction_to_equirectangular(direction);
}

template<typename Image, typename Node, typename Surface>
ccl_device_inline bool webgpu_svm_eval_image(Image image,
                                             const Node &node,
                                             Surface surface,
                                             ccl_private float *stack,
                                             WebGPUSvmStackUsage &written,
                                             WebGPUSvmStackUsage &varying,
                                             WebGPUSvmNormalSlots &surface_normals)
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
  float4 color;
#if !defined(CYCLES_WEBGPU_SVM_CORE) || defined(CYCLES_WEBGPU_SVM_PROJECTED_IMAGES)
  if constexpr (std::is_same_v<Node, SVMNodeTexImageBox>) {
    if (!(fabsf(co.z) <= 8194.0f && node.blend >= 0.0f && node.blend <= 1.0f)) {
      return false;
    }
    /* object_inverse_normal_transform() for a static triangle. */
    const Transform tfm = webgpu_svm_context_transform(surface,
                                                       CYCLES_WEBGPU_SVM_OBJECT_TRANSFORM);
    const float3 normal = webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_NORMAL));
    const float3 object_normal = safe_normalize(transform_direction_transposed(&tfm, normal));
    const float3 weights = webgpu_svm_image_box_weights(object_normal, node.blend);
    color = zero_float4();
    /* Keep the texture lookup shared by all three faces. */
#  pragma clang loop unroll(disable)
    for (uint axis = 0; axis < 3; ++axis) {
      const float weight = axis == 0 ? weights.x : (axis == 1 ? weights.y : weights.z);
      if (weight > 0.0f) {
        float4 face;
        if (!webgpu_svm_fetch_image(
                image, node, webgpu_svm_image_box_uv(co, object_normal, axis), face))
        {
          return false;
        }
        color += weight * face;
      }
    }
  }
  else
#endif
  {
    float2 uv = make_float2(co);
#if !defined(CYCLES_WEBGPU_SVM_CORE) || defined(CYCLES_WEBGPU_SVM_PROJECTED_IMAGES)
    if constexpr (std::is_same_v<Node, SVMNodeTexEnvironment>) {
      if (!(fabsf(co.z) <= 8194.0f)) {
        return false;
      }
      uv = webgpu_svm_environment_coordinates(co, node.projection);
    }
    else if (node.projection != NODE_IMAGE_PROJ_FLAT) {
      if (!(fabsf(co.z) <= 8194.0f)) {
        return false;
      }
      const float3 square = (co - make_float3(0.5f)) * 2.0f;
      uv = node.projection == NODE_IMAGE_PROJ_SPHERE ? map_to_sphere(square) : map_to_tube(square);
    }
#endif
    if (!webgpu_svm_fetch_image(image, node, uv, color)) {
      return false;
    }
  }
  if (stack_valid(node.out_offset)) {
    stack_store_float3(stack, node.out_offset, make_float3(color));
  }
  if (stack_valid(node.alpha_offset)) {
    stack_store_float(stack, node.alpha_offset, color.w);
  }
  webgpu_svm_mark_outputs(
      written, varying, surface_normals, node.alpha_offset, node.out_offset);
  return true;
}

struct WebGPUSvmRaycastHit {
  float distance;
  float is_hit;
  float is_self_hit;
  float3 normal;
};

/* Scene facts resolved by the host while packing. The packed bytecode marks an
 * attribute that no object using the material provides as ATTR_STD_NOT_FOUND
 * and other attributes with their packed slot. The host only accepts Particle
 * Info on objects using the dummy particle. */
struct WebGPUSvmPackedHost {
  /* Light and shadow evaluation omit KERNEL_FEATURE_NODE_RAYTRACE. Host
   * eligibility also uses these defaults when validating the bytecode. */
  static constexpr bool uses_ray_queries = false;
  ccl_device_inline_method float ambient_occlusion(float3, float, uint, uint) const
  {
    return 1.0f;
  }
  ccl_device_inline_method WebGPUSvmRaycastHit raycast(float3, float3, float distance, bool) const
  {
    return {distance, 0.0f, 0.0f, zero_float3()};
  }
  ccl_device_inline_method float3 bevel(float3 normal, float, uint) const
  {
    return normal;
  }
  ccl_device_inline_method bool evaluate_aov() const
  {
    return true;
  }
  ccl_device_inline_method void write_aov(const int /*offset*/,
                                          const bool /*color*/,
                                          const float4 /*value*/) const
  {
  }
  ccl_device_inline_method bool write_ordered_lobe(const uint /*index*/,
                                                   const uint /*type*/,
                                                   const Spectrum /*weight*/,
                                                   const float3 /*normal*/,
                                                   const float /*alpha*/) const
  {
    return true;
  }
  ccl_device_inline_method float4 setup_paired_sheen(const float3 /*normal*/,
                                                     const float /*weight*/,
                                                     const float /*roughness*/) const
  {
    /* Eligibility checks the sockets and arranges the stock LTC table upload. */
    return zero_float4();
  }

  ccl_device_inline_method int attribute(const uint /*word*/, const int attr) const
  {
    return attr;
  }
  ccl_device_inline_method bool dummy_particle() const
  {
    return true;
  }
  /* Packed attribute slot values at the shaded point. */
  ccl_device_inline_method bool attribute_sample(const uint /*slot*/,
                                                 WebGPUSvmAttributeSample & /*sample*/,
                                                 const float /*du*/ = 0.0f,
                                                 const float /*dv*/ = 0.0f,
                                                 const float /*filter_width*/ = 0.0f) const
  {
    return false;
  }
  /* kernel_ies_interp() for an IES slot. */
  ccl_device_inline_method bool ies(const uint /*slot*/,
                                    const float /*h_angle*/,
                                    const float /*v_angle*/,
                                    float & /*value*/) const
  {
    return false;
  }
  /* Packed reference to a host-built slot stored in an otherwise unused field. */
  ccl_device_inline_method int synthetic(const uint /*word*/,
                                         const int packed,
                                         const WebGPUSvmSyntheticSlot /*kind*/) const
  {
    return packed;
  }
  ccl_device_inline_method uint displacement_bump_slots(const uint /*word*/,
                                                         const uint packed) const
  {
    return packed;
  }
  ccl_device_inline_method bool triangle_corners(const uint /*slot*/,
                                                 WebGPUSvmTriangleCorners & /*corners*/) const
  {
    return false;
  }
  /* Called with each opcode so the host can report where evaluation stopped. */
  ccl_device_inline_method void visit(const uint /*type*/, const uint /*word*/) const {}
  /* Row of cie_color_match. WebGPU reads the packed copy because WGSL cannot
   * index the constant table dynamically. */
  ccl_device_inline_method float3 cie_color_match_row(const int row) const
  {
    return make_float3(cie_color_match[row][0], cie_color_match[row][1], cie_color_match[row][2]);
  }
};

/* First-order screen derivatives of the synchronized triangle inputs. */
struct WebGPUSvmBumpInputs {
  differential3 position;
  differential3 uv;
  differential3 generated;
  differential u;
  differential v;
};

#include "displacement_bump_svm.h"

ccl_device_inline float3 webgpu_svm_bump_offset(const float3 value,
                                                const differential3 derivatives,
                                                const NodeBumpOffset offset,
                                                const float width)
{
  if (offset == NODE_BUMP_OFFSET_DX) {
    return value + derivatives.dx * width;
  }
  if (offset == NODE_BUMP_OFFSET_DY) {
    return value + derivatives.dy * width;
  }
  return value;
}

template<typename FetchSurface>
ccl_device_inline float3 webgpu_svm_set_bump(const SVMNodeSetBump &node,
                                             float *stack,
                                             const FetchSurface surface,
                                             differential3 position)
{
  /* Stock svm_node_set_bump() with the triangle differential supplied by
   * the wavefront shading record. */
  const float3 shading_normal = webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_NORMAL));
#ifdef CYCLES_WEBGPU_SAMPLED_EMISSION
  /* Keep the default and linked normals in the same local array. SPIR-V
   * disallows a select between two different private pointers. */
  stack[SVM_STACK_SIZE] = shading_normal.x;
  stack[SVM_STACK_SIZE + 1] = shading_normal.y;
  stack[SVM_STACK_SIZE + 2] = shading_normal.z;
  const uint normal_offset = node.normal_offset;
  float3 normal = make_float3(stack[normal_offset], stack[normal_offset + 1],
                              stack[normal_offset + 2]);
#else
  float3 normal = stack_load_float3_default(stack, node.normal_offset, shading_normal);
#endif
  if (node.use_object_space) {
    const Transform transform = webgpu_svm_context_transform(
        surface, CYCLES_WEBGPU_SVM_OBJECT_TRANSFORM);
    const Transform inverse = webgpu_svm_context_transform(surface, CYCLES_WEBGPU_SVM_OBJECT_INVERSE);
    normal = normalize(transform_direction_transposed(&transform, normal));
    position.dx = transform_direction(&inverse, position.dx);
    position.dy = transform_direction(&inverse, position.dy);
  }

  const float3 Rx = cross(position.dy, normal);
  const float3 Ry = cross(normal, position.dx);
  const float center = stack_load_float(stack, node.center_offset);
  const float3 gradient = (stack_load_float(stack, node.dx_offset) - center) * Rx +
                          (stack_load_float(stack, node.dy_offset) - center) * Ry;
  const float determinant = dot(position.dx, Rx);
  const float strength = max(stack_load(stack, node.strength), 0.0f);
  const float scale = (node.invert ? -1.0f : 1.0f) * stack_load(stack, node.scale);
  float3 result = safe_normalize(node.bump_filter_width * fabsf(determinant) * normal -
                                 scale * signf(determinant) * gradient);
  result = is_zero(result) ? normal : normalize(strength * result + (1.0f - strength) * normal);
  if (node.use_object_space) {
    const Transform inverse = webgpu_svm_context_transform(surface, CYCLES_WEBGPU_SVM_OBJECT_INVERSE);
    result = normalize(transform_direction_transposed(&inverse, result));
  }
  return result;
}

struct WebGPUSvmCoat {
  float weight;
  float ior;
  float alpha;
  bool possible;
};

struct WebGPUSvmMaterial {
  float3 color;
  float3 specular_tint;
  float alpha;
  float anisotropy_aspect;
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
  bool requires_full_svm;
  bool requires_advanced_procedural_svm;
  bool uses_compact_voronoi_distance;
  bool varying_transparency;
  bool principled_alpha;
  bool mixed_transparency;
  Spectrum transparency;
  float transparent_sample_weight;
  float diffuse_sample_weight;
  /* A word-sized flag keeps WGSL lowering from packing it into a bool vector. */
  uint transparent_first;
  uint transparent_emission;
  uint principled_translucent;
  Spectrum translucent_color;
  float translucent_sample_weight;
  bool translucent_first;
  Spectrum glossy_color;
  float3 mixed_glossy_normal;
  float glossy_sample_weight;
  uint mixed_glossy_flags;
  bool glossy_isotropic;
  bool pixel_wireframe;
  bool surface_bump;
  bool ordered_principled_normal;
  bool ordered_principled_layers;
  bool ordered_principled_sheen;
  uint ordered_principled_specular_candidates;
  uint ordered_principled_metal_candidates;
  float4 ordered_principled_tangent_0;
  float4 ordered_principled_tangent_1;
  float4 ordered_principled_coat;
  bool displacement_bump;
  float3 displacement_normal;
  float3 closure_normal;
  bool closure_normal_overridden;
  /* A lone Holdout closure, with its weight in color. Camera paths write the
   * weighted transparency and end without scattering or emission. */
  bool holdout;
  /* Standalone surface closures reuse the specular, layer and transmission records. */
  float surface_closure;
  float2 surface_parameters;
  float3 surface_tangent;
  float3 surface_fresnel_a;
  float3 surface_fresnel_b;
  WebGPUSvmCoat coat;
  bool single_principled_specular_candidate;
  bool coated_principled_emission;
  Spectrum coat_emission;
  uint surface_model;
  float surface_hair_ior;
  float surface_hair_aspect;
  /* Accumulated emission closure weight, as emission_setup() stores it. */
  float3 emission;
  /* Keep the host eligibility count out of the shader's packed bool fields. */
  uint diffuse_principled_lobes;
  uint ordered_lobe_candidates;
  uint ordered_lobe_count;
  /* Candidate types are validated even when the probe's weights are zero. */
  uint ordered_lobe_types;
  uint ordered_lobe_features;
  bool ordered_glossy_mix;
};

ccl_device_inline void webgpu_svm_save_paired_normal(WebGPUSvmMaterial &material,
                                                     const float *stack,
                                                     const uint slot,
                                                     const uint bump_word,
                                                     uint &closure_bump_word,
                                                     const bool diffuse,
                                                     const bool paired)
{
  const uint first_bump_word = closure_bump_word;
  closure_bump_word = bump_word;
  if (paired && bump_word != first_bump_word) {
    if (bump_word != 0u && first_bump_word != 0u) {
      material.mixed_glossy_flags |= CYCLES_WEBGPU_MIXED_BUMP_INDEPENDENT;
      if (!diffuse) {
        material.mixed_glossy_normal = stack_load_float3(stack, slot);
        return;
      }
      /* Diffuse uses the main closure normal record. The Glossy normal was
       * copied at its closure, before its bytecode operand could be overwritten. */
      material.mixed_glossy_normal = material.closure_normal;
    }
    else {
      material.mixed_glossy_flags |= diffuse == (bump_word != 0u) ?
                                         CYCLES_WEBGPU_MIXED_BUMP_DIFFUSE_ONLY :
                                         CYCLES_WEBGPU_MIXED_BUMP_GLOSSY_ONLY;
    }
  }
  if (bump_word != 0u) {
    material.closure_normal = stack_load_float3(stack, slot);
    material.closure_normal_overridden = true;
  }
}

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

/* In the stock Principled setup, these constant inputs leave exactly the
 * Lambert Diffuse branch. In particular, zero specular level makes eta one,
 * so there is no dielectric closure or attenuation of the diffuse weight. */
ccl_device_inline bool webgpu_svm_is_diffuse_principled(const SVMNodePrincipledBsdfData &data)
{
  const uint other_layers = data.metallic.bits | data.diffuse_roughness.bits |
                            data.specular_ior_level.bits | data.transmission_weight.bits |
                            data.subsurface_weight.bits | data.coat_weight.bits |
                            data.sheen_weight.bits | data.thin_film_thickness.bits;
  const uint emission_color = data.emission_color.x.bits | data.emission_color.y.bits |
                              data.emission_color.z.bits;
  const bool emits = (data.emission_strength.bits & 0x7fffffffu) != 0 &&
                     (emission_color & 0x7fffffffu) != 0;
  return data.alpha.bits == __float_as_uint(1.0f) && (other_layers & 0x7fffffffu) == 0 &&
         !webgpu_svm_input_on_stack(data.ior) && webgpu_svm_finite(__uint_as_float(data.ior.bits)) &&
         !emits;
}

/* Apply the stock closure allocation cutoff before reducing equivalent lobes.
 * Identical Lambert normals share the cosine distribution and diffuse label.
 * Stock MIS then reduces to that common PDF, and only the RGB weights add. */
ccl_device_inline bool webgpu_svm_add_diffuse_lobe(WebGPUSvmMaterial &material,
                                                   Spectrum weight,
                                                   const uint preceding_lobes)
{
  if (!webgpu_svm_finite(weight.x) || !webgpu_svm_finite(weight.y) ||
      !webgpu_svm_finite(weight.z))
  {
    return false;
  }
  const float sample_weight = closure_sample_weight(0, weight);
  if (preceding_lobes == 0) {
    material.color = zero_spectrum();
  }
  if (sample_weight > 0.0f) {
    material.color += weight;
    material.diffuse_sample_weight += sample_weight;
  }
  return webgpu_svm_finite(average(material.color));
}

ccl_device_inline bool webgpu_svm_principled_hair(const SVMNodePrincipledHairBsdfData &data,
                                                 const bool huang,
                                                 const WebGPUSvmStackUsage &written,
                                                 const float *stack,
                                                 const float3 tangent,
                                                 WebGPUSvmMaterial &material)
{
  if (!webgpu_svm_inputs_valid(written, data.roughness, data.random_roughness, data.offset,
                               data.ior, data.random))
  {
    return false;
  }
  /* The static-triangle adapter has no per-curve random attribute. Use the
   * fallback socket, as svm_node_closure_bsdf() does on mesh geometry. */
  const float random = stack_load(stack, data.random);
  const float factor = 1.0f + 2.0f * (random - 0.5f) * stack_load(stack, data.random_roughness);
  const float roughness = stack_load(stack, data.roughness) * factor;
  const float ior = stack_load(stack, data.ior);
  if (!webgpu_svm_finite(roughness) || !webgpu_svm_finite(ior) || ior <= 1.0f ||
      (!huang && !webgpu_svm_inputs_valid(written, data.radial_roughness, data.coat)) ||
      (huang && !webgpu_svm_inputs_valid(written, data.R, data.TT, data.TRT, data.aspect_ratio)))
  {
    return false;
  }
  const float radial = huang ? roughness : stack_load(stack, data.radial_roughness) * factor;
  Spectrum sigma;
  if (data.parametrization == NODE_PRINCIPLED_HAIR_DIRECT_ABSORPTION) {
    if (!webgpu_svm_surface_color(data.absorption_coefficient, written, stack, sigma)) {
      return false;
    }
    sigma = rgb_to_spectrum(sigma);
  }
  else if (data.parametrization == NODE_PRINCIPLED_HAIR_PIGMENT_CONCENTRATION) {
    float3 tint;
    if (!webgpu_svm_surface_color(data.tint, written, stack, tint) ||
        !webgpu_svm_inputs_valid(written, data.melanin, data.melanin_redness, data.random_color))
    {
      return false;
    }
    const float factor_color = 1.0f + 2.0f * (random - 0.5f) *
                                        saturatef(stack_load(stack, data.random_color));
    const float melanin = -logf(fmaxf(1.0f - stack_load(stack, data.melanin) * factor_color,
                                      0.0001f));
    const float redness = stack_load(stack, data.melanin_redness);
    sigma = bsdf_principled_hair_sigma_from_concentration(melanin * (1.0f - redness),
                                                         melanin * redness) +
            bsdf_principled_hair_sigma_from_reflectance(rgb_to_spectrum(tint), radial);
  }
  else if (data.parametrization == NODE_PRINCIPLED_HAIR_REFLECTANCE) {
    float3 color;
    if (!webgpu_svm_surface_color(data.color, written, stack, color)) {
      return false;
    }
    sigma = bsdf_principled_hair_sigma_from_reflectance(rgb_to_spectrum(color), radial);
  }
  else {
    return false;
  }
  material.surface_tangent = tangent;
  material.surface_parameters = make_float2(stack_load(stack, data.offset), roughness);
  material.surface_fresnel_a = sigma;
  material.surface_hair_ior = ior;
  if (huang) {
    material.surface_hair_aspect = stack_load(stack, data.aspect_ratio);
    material.surface_fresnel_b = max(make_float3(stack_load(stack, data.R),
                                                 stack_load(stack, data.TT),
                                                 stack_load(stack, data.TRT)),
                                     zero_float3());
    return webgpu_svm_finite(material.surface_hair_aspect) && material.surface_hair_aspect > 0.0f;
  }
  material.surface_fresnel_b = make_float3(radial,
                                           1.0f - saturatef(stack_load(stack, data.coat)),
                                           0.0f);
  return webgpu_svm_finite(radial);
}

ccl_device_inline bool webgpu_svm_rough_ggx(const float alpha)
{
  /* The fused Principled reflection path evaluates non-singular GGX lobes. */
  return sqr(alpha) > 2e-10f;
}

template<typename Fetch, typename Host>
ccl_device_inline bool webgpu_svm_capture_ordered_lobe(
    Fetch fetch,
    uint &offset,
    const uint end,
    const SVMNodeClosureBsdf &closure,
    const WebGPUSvmStackUsage &written,
    const WebGPUSvmStackUsage &varying,
    const WebGPUSvmNormalSlots &surface_normals,
    const float *stack,
    const Spectrum weight,
    const Host &host,
    WebGPUSvmMaterial &material,
    uint &active)
{
  if (stack_valid(closure.mix_weight_offset) &&
      !webgpu_svm_stack_has(written, closure.mix_weight_offset))
  {
    return false;
  }
  const ClosureType type = closure.closure_type;
  uint normal_slot = SVM_STACK_INVALID;
  float alpha = 1.0f;
  if (type == CLOSURE_BSDF_DIFFUSE_ID) {
    SVMNodeDiffuseBsdfData data;
    if (!webgpu_svm_read(fetch, offset, end, data) ||
        !webgpu_svm_input_valid(written, data.roughness) ||
        (webgpu_svm_input_on_stack(data.roughness) &&
         webgpu_svm_stack_has(varying, data.roughness.bits & 0xffu)))
    {
      return false;
    }
    const float roughness = stack_load(stack, data.roughness);
    if (!webgpu_svm_finite(roughness) || !diffuse_roughness_is_almost_zero(roughness))
    {
      return false;
    }
    normal_slot = data.normal_offset;
  }
  else if (type == CLOSURE_BSDF_TRANSPARENT_ID || type == CLOSURE_BSDF_TRANSLUCENT_ID) {
    SVMNodeSimpleBsdfData data;
    if (!webgpu_svm_read(fetch, offset, end, data)) {
      return false;
    }
    if (type == CLOSURE_BSDF_TRANSLUCENT_ID) {
      normal_slot = data.normal_offset;
    }
  }
  else if (type == CLOSURE_BSDF_MICROFACET_GGX_ID ||
           type == CLOSURE_BSDF_MICROFACET_BECKMANN_ID)
  {
    SVMNodeGlossyBsdfData data;
    if (!webgpu_svm_read(fetch, offset, end, data) ||
        !webgpu_svm_inputs_valid(written, data.roughness, data.anisotropy) ||
        (webgpu_svm_input_on_stack(data.roughness) &&
         webgpu_svm_stack_has(varying, data.roughness.bits & 0xffu)))
    {
      return false;
    }
    const float roughness = stack_load(stack, data.roughness);
    const float anisotropy = stack_load(stack, data.anisotropy);
    alpha = sqr(saturatef(roughness));
    /* The stock microfacet routines sample the singular case directly and
     * contribute zero directional density for it during MIS evaluation. */
    if (!webgpu_svm_finite(roughness) || !webgpu_svm_finite(anisotropy) ||
        (stack_valid(data.tangent_offset) &&
         (webgpu_svm_input_on_stack(data.anisotropy) || fabsf(anisotropy) > 1e-4f)))
    {
      return false;
    }
    material.ordered_glossy_mix |= !webgpu_svm_rough_ggx(alpha);
    normal_slot = data.normal_offset;
  }
  else {
    return false;
  }

  const bool bump = surface_normals.find_bump(normal_slot) != 0u;
  if (!bump && !webgpu_svm_surface_normal(normal_slot, surface_normals.base)) {
    return false;
  }
  Spectrum color = weight * stack_load_float_default(stack, closure.mix_weight_offset, 1.0f);
  if (!webgpu_svm_finite(color.x) || !webgpu_svm_finite(color.y) || !webgpu_svm_finite(color.z)) {
    return false;
  }
  /* bsdf_transparent_setup() retains the input tint and tests its absolute
   * average. The other ordered lobes use the stock nonnegative allocation weight. */
  const bool transparent = type == CLOSURE_BSDF_TRANSPARENT_ID;
  if (transparent && !material.varying_transparency &&
      (reduce_min(color) < 0.0f || reduce_max(color) > 1.0f))
  {
    return false;
  }
  const float sample_weight = transparent ? fabsf(average(color)) :
                                            closure_sample_weight(0, color);
  if (!webgpu_svm_finite(sample_weight)) {
    return false;
  }
  if (sample_weight >= CLOSURE_WEIGHT_CUTOFF) {
    const float3 normal = bump ? stack_load_float3(stack, normal_slot) : zero_float3();
    const uint stored_type = uint(type) | (bump ? CYCLES_WEBGPU_ORDERED_LOBE_BUMP : 0u);
    /* Snapshot the normal before the next opcode can replace this stack slot. */
    if (host.write_ordered_lobe(active, stored_type, color, normal, alpha)) {
      ++active;
      if (transparent) {
        material.transparency = color;
        material.ordered_lobe_features |= CYCLES_WEBGPU_ORDERED_HAS_TRANSPARENT;
      }
      if (type == CLOSURE_BSDF_TRANSLUCENT_ID) {
        material.ordered_lobe_features |= CYCLES_WEBGPU_ORDERED_HAS_TRANSLUCENT;
      }
    }
  }
  return true;
}

/* Match the unlayered glossy and metallic cases of svm_node_closure_bsdf().
 * The GPU applies energy compensation and blur after it knows the view direction. */
ccl_device_inline bool webgpu_svm_glossy_geometry(const SVMInputFloat roughness_input,
                                                  const SVMInputFloat anisotropy_input,
                                                  const SVMInputFloat rotation_input,
                                                  const SVMStackOffset tangent_offset,
                                                  const bool metallic,
                                                  const WebGPUSvmStackUsage &written,
                                                  const float *stack,
                                                  const float3 N,
                                                  WebGPUSvmMaterial &material)
{
  if (!webgpu_svm_inputs_valid(written, roughness_input, anisotropy_input, rotation_input)) {
    return false;
  }
  const float roughness = stack_load(stack, roughness_input);
  const float anisotropy_value = stack_load(stack, anisotropy_input);
  const float rotation = stack_load(stack, rotation_input);
  if (!webgpu_svm_finite(roughness) || !webgpu_svm_finite(anisotropy_value) ||
      !webgpu_svm_finite(rotation))
  {
    return false;
  }
  const float alpha = sqr(saturatef(roughness));
  material.surface_parameters = make_float2(alpha, alpha);
  material.surface_tangent = zero_float3();
  const float anisotropy = metallic ? saturatef(anisotropy_value) :
                                     clamp(anisotropy_value, -0.99f, 0.99f);
  if (!stack_valid(tangent_offset) ||
      (metallic ? anisotropy == 0.0f : fabsf(anisotropy) <= 1e-4f))
  {
    return true;
  }
  if (!webgpu_svm_stack_has3(written, tangent_offset)) {
    return false;
  }
  material.surface_tangent = stack_load_float3(stack, tangent_offset);
  if (metallic) {
    const float aspect = sqrtf(1.0f - anisotropy * 0.9f);
    material.surface_parameters = make_float2(alpha / aspect, alpha * aspect);
    if (rotation != 0.0f) {
      material.surface_tangent = rotate_around_axis(
          material.surface_tangent, N, rotation * M_2PI_F);
    }
  }
  else {
    const float aspect = 1.0f - fabsf(anisotropy);
    material.surface_parameters = anisotropy < 0.0f ? make_float2(alpha / aspect, alpha * aspect) :
                                                     make_float2(alpha * aspect, alpha / aspect);
    /* Glossy rotates about the corrected specular normal, available during closure setup. */
    material.surface_fresnel_b.x = rotation;
  }
  return true;
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
       method != CLOSURE_BSSRDF_RANDOM_WALK_LEGACY_ID && method != CLOSURE_BSSRDF_BURLEY_ID) ||
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

ccl_device_inline bool webgpu_svm_read_white_coat(const SVMNodePrincipledBsdfData &data,
                                                   const WebGPUSvmStackUsage &written,
                                                   const float *stack,
                                                   const uint shader_normal,
                                                   WebGPUSvmCoat &coat)
{
  if (!webgpu_svm_input_valid(written, data.coat_weight)) {
    return false;
  }
  coat.weight = stack_load(stack, data.coat_weight);
  if (!webgpu_svm_finite(coat.weight)) {
    return false;
  }
  coat.possible = webgpu_svm_input_on_stack(data.coat_weight) ||
                  coat.weight > CLOSURE_WEIGHT_CUTOFF;
  if (coat.possible) {
    const uint white = __float_as_uint(1.0f);
    if (!webgpu_svm_surface_normal(data.coat_normal_offset, shader_normal) ||
        data.coat_tint.x.bits != white || data.coat_tint.y.bits != white ||
        data.coat_tint.z.bits != white ||
        !webgpu_svm_inputs_valid(written, data.coat_ior, data.coat_roughness))
    {
      return false;
    }
    const float ior = stack_load(stack, data.coat_ior);
    const float roughness = stack_load(stack, data.coat_roughness);
    if (!webgpu_svm_finite(ior) || !webgpu_svm_finite(roughness)) {
      return false;
    }
    /* A connected Geometry Normal uses the shader normal. An absent SVM
     * offset instead uses the branch's own Normal through the stock default. */
    const float ior_magnitude = fmaxf(ior, 1.0f);
    coat.ior = stack_valid(data.coat_normal_offset) ? -ior_magnitude : ior_magnitude;
    coat.alpha = sqr(saturatef(roughness));
  }
  return true;
}

template<bool ordered = false>
ccl_device_inline bool webgpu_svm_principled(const SVMNodePrincipledBsdfData &data,
                                            const WebGPUSvmStackUsage &written,
                                            const float *stack,
                                            const uint surface_normal,
                                            const float mix_weight,
                                            const float3 shading_N,
                                            WebGPUSvmMaterial &material,
                                            const bool allow_coat = false)
{
  /* Pack the supported Principled layers with uniform layer controls. */
  const uint layer_bits = (ordered ? 0u : data.sheen_weight.bits) |
                          (ordered || allow_coat ? 0u : data.coat_weight.bits) |
                          data.thin_film_thickness.bits | data.diffuse_roughness.bits;
  const float ior_input = __uint_as_float(data.ior.bits);
  const bool varying_metallic = webgpu_svm_input_on_stack(data.metallic);
  const bool varying_specular = webgpu_svm_input_on_stack(data.specular_ior_level);
  if (!webgpu_svm_inputs_valid(written, data.metallic,
                               data.specular_ior_level, data.roughness,
                               data.anisotropic, data.anisotropic_rotation,
                               data.emission_color, data.emission_strength) ||
      (stack_valid(data.tangent_offset) && !webgpu_svm_stack_has3(written, data.tangent_offset)))
  {
    return false;
  }
  const float metallic = saturatef(stack_load(stack, data.metallic));
  const float ior_level = max(stack_load(stack, data.specular_ior_level), 0.0f);
  const float sss_input = __uint_as_float(data.subsurface_weight.bits);
  const float transmission_input = __uint_as_float(data.transmission_weight.bits);
  if ((layer_bits & 0x7fffffffu) != 0 ||
      !webgpu_svm_finite(ior_input) || !webgpu_svm_finite(metallic) ||
      !webgpu_svm_finite(ior_level) || !webgpu_svm_finite(transmission_input) ||
      !webgpu_svm_finite(sss_input) || transmission_input < 0.0f || transmission_input > 1.0f ||
      !webgpu_svm_surface_normal(data.normal_offset, surface_normal))
  {
    return false;
  }
  /* principled_bsdf_emission(): the input mix weight already includes Alpha. */
  const float3 emission = stack_load(stack, data.emission_color) *
                          stack_load(stack, data.emission_strength);
  if (allow_coat) {
    material.coat_emission = emission;
  }
  else if (!is_zero(emission)) {
    material.emission += emission * mix_weight;
  }
  if (!webgpu_svm_finite(emission.x) || !webgpu_svm_finite(emission.y) ||
      !webgpu_svm_finite(emission.z) ||
      !webgpu_svm_finite(material.emission.x) || !webgpu_svm_finite(material.emission.y) ||
      !webgpu_svm_finite(material.emission.z))
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
  else if (!allow_coat && !varying_metallic && !varying_specular && metallic == 0.0f && ior_level == 0.0f &&
           transmission == 0.0f)
  {
    if constexpr (!ordered) {
      material.color *= mix_weight;
    }
    return true;
  }

  const float roughness = stack_load(stack, data.roughness);
  const float anisotropic = saturatef(stack_load(stack, data.anisotropic));
  const float rotation = stack_load(stack, data.anisotropic_rotation);
  const bool has_tangent = stack_valid(data.tangent_offset);
  const bool may_anisotropic = has_tangent &&
                               (anisotropic > 0.0f || webgpu_svm_input_on_stack(data.anisotropic));
  const float3 tint = make_float3(__uint_as_float(data.specular_tint.x.bits),
                                  __uint_as_float(data.specular_tint.y.bits),
                                  __uint_as_float(data.specular_tint.z.bits));
  const bool multiscatter = data.distribution == CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID;
  if ((!multiscatter && data.distribution != CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID) ||
      !webgpu_svm_finite(roughness) || !webgpu_svm_finite(anisotropic) ||
      !webgpu_svm_finite(rotation) || (may_anisotropic && material.subsurface > 0.0f) ||
      !webgpu_svm_finite(tint.x) || !webgpu_svm_finite(tint.y) || !webgpu_svm_finite(tint.z) ||
      !webgpu_svm_finite(mix_weight) || mix_weight < 0.0f || mix_weight > 1.0f ||
      ior_input <= 0.0f)
  {
    return false;
  }

  const float alpha = sqr(saturatef(roughness));
  const float ior = fmaxf(ior_input, 1e-5f);
  if (transmission > 0.0f &&
      (varying_metallic || metallic != 0.0f || !webgpu_svm_rough_glass_ior(ior) ||
       data.thin_wall.value != 0 || data.thin_wall.offset != SVM_STACK_INVALID ||
       (data.transmission_dispersion_scale.bits & 0x7fffffffu) != 0))
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
  /* Principled uses the tangent in the closure's original normal frame. */
  if (has_tangent && anisotropic > 0.0f) {
    material.anisotropy_aspect = sqrtf(1.0f - anisotropic * 0.9f);
    float3 tangent = stack_load_float3(stack, data.tangent_offset);
    if (rotation != 0.0f) {
      const float3 N = safe_normalize_fallback(
          stack_load_float3_default(stack, data.normal_offset, shading_N), shading_N);
      tangent = rotate_around_axis(tangent, N, rotation * M_2PI_F);
    }
    material.surface_tangent = tangent;
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

#include "ordered_principled_svm.h"

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

ccl_device_inline bool webgpu_svm_is_slot(const int resolved)
{
  return resolved != int(ATTR_STD_NOT_FOUND) && (uint(resolved) & webgpu_svm_attribute_slot) != 0;
}

/* Sample a resolved attribute, leaving a missing record for ATTR_STD_NOT_FOUND. */
template<typename Host>
ccl_device_inline bool webgpu_svm_sample_resolved(const Host &host,
                                                  const int resolved,
                                                  WebGPUSvmAttributeSample &sample)
{
  sample = {WEBGPU_SVM_ATTRIBUTE_MISSING, 0, zero_float4()};
  if (resolved == int(ATTR_STD_NOT_FOUND)) {
    return true;
  }
  return webgpu_svm_is_slot(resolved) &&
         host.attribute_sample(uint(resolved) & ~webgpu_svm_attribute_slot, sample);
}

/* object_normal_transform() followed by the projection onto the shading normal
 * shared by primitive_tangent() and svm_node_tangent(). */
template<typename FetchSurface>
ccl_device_inline float3 webgpu_svm_surface_tangent(FetchSurface surface, float3 tangent)
{
  const Transform inverse = webgpu_svm_context_transform(surface, CYCLES_WEBGPU_SVM_OBJECT_INVERSE);
  tangent = normalize(transform_direction_transposed(&inverse, tangent));
  const float3 N = webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_NORMAL));
  return cross(N, normalize(cross(tangent, N)));
}

/* Helpers from kernel/svm/sky.h, which also includes the kernel image lookup. */
ccl_device_inline float webgpu_svm_sky_angle_between(const float thetav,
                                                     const float phiv,
                                                     const float theta,
                                                     const float phi)
{
  const float cospsi = sinf(thetav) * sinf(theta) * cosf(phi - phiv) + cosf(thetav) * cosf(theta);
  return safe_acosf(cospsi);
}

ccl_device_inline float webgpu_svm_sky_perez_function(const float *lam,
                                                      const float theta,
                                                      const float gamma)
{
  const float ctheta = cosf(theta);
  const float cgamma = cosf(gamma);
  return (1.0f + lam[0] * expf(lam[1] / ctheta)) *
         (1.0f + lam[2] * expf(lam[3] * gamma) + lam[4] * cgamma * cgamma);
}

ccl_device_inline float webgpu_svm_sky_radiance_internal(const float *configuration,
                                                         const float theta,
                                                         const float gamma)
{
  const float ctheta = cosf(theta);
  const float cgamma = cosf(gamma);
  const float expM = expf(configuration[4] * gamma);
  const float rayM = cgamma * cgamma;
  const float mieM = (1.0f + rayM) / powf((1.0f + configuration[8] * configuration[8] -
                                           2.0f * configuration[8] * cgamma),
                                          1.5f);
  const float zenith = sqrtf(ctheta);
  return (1.0f + configuration[0] * expf(configuration[1] / (ctheta + 0.01f))) *
         (configuration[2] + configuration[3] * expM + configuration[5] * rayM +
          configuration[6] * mieM + configuration[7] * zenith);
}

/* xyz_to_rgb_clamped() with the packed film rows. */
ccl_device_inline float3 webgpu_svm_xyz_to_rgb_clamped(const WebGPUSvmColorSpace &color_space,
                                                       const float3 xyz)
{
  return max(make_float3(dot(color_space.xyz_to_r, xyz),
                         dot(color_space.xyz_to_g, xyz),
                         dot(color_space.xyz_to_b, xyz)),
             zero_float3());
}

/* sky_radiance_preetham() and sky_radiance_hosek(). */
ccl_device_inline float3 webgpu_svm_sky_analytic(const WebGPUSvmColorSpace &color_space,
                                                 const NodeSkyType type,
                                                 const float3 dir,
                                                 const SVMNodeTexSkyPreethamData &data)
{
  const float2 spherical = direction_to_spherical(dir);
  float theta = spherical.x;
  const float phi = -spherical.y + M_PI_2_F;
  const float gamma = webgpu_svm_sky_angle_between(theta, phi, data.theta, data.phi);
  theta = min(theta, M_PI_2_F - 0.001f);
  if (type == NODE_SKY_PREETHAM) {
    const float x = data.radiance_y * webgpu_svm_sky_perez_function(data.config_y, theta, gamma);
    const float y = data.radiance_z * webgpu_svm_sky_perez_function(data.config_z, theta, gamma);
    const float Y = data.radiance_x * webgpu_svm_sky_perez_function(data.config_x, theta, gamma);
    return webgpu_svm_xyz_to_rgb_clamped(color_space, xyY_to_xyz(x, y, Y));
  }
  const float x = webgpu_svm_sky_radiance_internal(data.config_x, theta, gamma) * data.radiance_x;
  const float y = webgpu_svm_sky_radiance_internal(data.config_y, theta, gamma) * data.radiance_y;
  const float z = webgpu_svm_sky_radiance_internal(data.config_z, theta, gamma) * data.radiance_z;
  return webgpu_svm_xyz_to_rgb_clamped(color_space, make_float3(x, y, z)) * (M_2PI_F / 683);
}

/* sky_radiance_nishita() shared by surface and world textures. The caller
 * supplies the stock LUT and working RGB space from its packed scene data. */
template<typename FetchImage>
ccl_device_inline bool webgpu_svm_sky_nishita(const WebGPUSvmColorSpace &color_space,
                                             const float3 dir,
                                             const SVMNodeTexSkyNishitaData &data,
                                             FetchImage image,
                                             float3 &rgb)
{
  const float3 pixel_bottom = make_float3(
      data.pixel_bottom_x, data.pixel_bottom_y, data.pixel_bottom_z);
  const float3 pixel_top = make_float3(data.pixel_top_x, data.pixel_top_y, data.pixel_top_z);
  const bool sun_disc = (data.angular_diameter >= 0.0f);
  float3 xyz = zero_float3();
  const float2 direction = (dir.x == 0.0f && dir.y == 0.0f) ?
                               make_float2(safe_acosf(dir.z), 0.0f) :
                               direction_to_spherical(dir);
  const float3 sun_dir = spherical_to_direction(data.sun_elevation - M_PI_2_F,
                                                data.sun_rotation - M_PI_2_F);
  const float sun_dir_angle = precise_angle(dir, sun_dir);
  const float half_angular = data.angular_diameter * 0.5f;
  const float dir_elevation = M_PI_2_F - direction.x;
  if (sun_disc && sun_dir_angle < half_angular &&
      dir_elevation > data.earth_intersection_angle)
  {
    const float y = ((dir_elevation - data.sun_elevation) / data.angular_diameter) + 0.5f;
    const float limb_darkening = (1.0f - 0.6f * (1.0f - sqrtf(1.0f - sqr(sun_dir_angle /
                                                                        half_angular))));
    xyz = mix(pixel_bottom, pixel_top, y) * data.sun_intensity * limb_darkening;
  }
  const float x = fractf((-direction.y - M_PI_2_F + data.sun_rotation) * M_1_2PI_F);
  const float y = copysignf(sqrtf(fabsf(dir_elevation) * M_2_PI_F), dir_elevation) * 0.5f +
                  0.5f;
  float4 sky;
  if (!image(int(data.texture_id), make_float2(x, y), false, sky)) {
    return false;
  }
  xyz += make_float3(sky);
  rgb = webgpu_svm_xyz_to_rgb_clamped(color_space, xyz);
  return true;
}

/* svm_node_vector_transform() for a static triangle object, reading the
 * synchronized object and camera transforms from the surface context. */
template<typename FetchSurface>
ccl_device_inline float3 webgpu_svm_vector_transform(FetchSurface surface,
                                                     const SVMNodeVectorTransform &node,
                                                     float3 in)
{
  const NodeVectorTransformType type = node.transform_type;
  const NodeVectorTransformConvertSpace from = node.convert_from;
  const NodeVectorTransformConvertSpace to = node.convert_to;
  const bool is_normal = (type == NODE_VECTOR_TRANSFORM_TYPE_NORMAL);
  const bool is_direction = (type == NODE_VECTOR_TRANSFORM_TYPE_VECTOR);

  const auto world_to_camera = [&]() {
    if (is_normal) {
      const Transform tfm = webgpu_svm_context_transform(surface,
                                                         CYCLES_WEBGPU_SVM_CAMERA_TO_WORLD);
      return normalize(transform_direction_transposed(&tfm, in));
    }
    const Transform tfm = webgpu_svm_context_transform(surface, CYCLES_WEBGPU_SVM_WORLD_TO_CAMERA);
    return is_direction ? transform_direction(&tfm, in) : transform_point(&tfm, in);
  };
  const auto camera_to_world = [&]() {
    if (is_normal) {
      const Transform tfm = webgpu_svm_context_transform(surface, CYCLES_WEBGPU_SVM_WORLD_TO_CAMERA);
      return normalize(transform_direction_transposed(&tfm, in));
    }
    const Transform tfm = webgpu_svm_context_transform(surface, CYCLES_WEBGPU_SVM_CAMERA_TO_WORLD);
    return is_direction ? transform_direction(&tfm, in) : transform_point(&tfm, in);
  };
  /* object_inverse_*_transform() */
  const auto world_to_object = [&]() {
    if (is_normal) {
      const Transform tfm = webgpu_svm_context_transform(surface,
                                                         CYCLES_WEBGPU_SVM_OBJECT_TRANSFORM);
      return safe_normalize(transform_direction_transposed(&tfm, in));
    }
    const Transform tfm = webgpu_svm_context_transform(surface, CYCLES_WEBGPU_SVM_OBJECT_INVERSE);
    return is_direction ? transform_direction(&tfm, in) : transform_point(&tfm, in);
  };
  /* object_*_transform() */
  const auto object_to_world = [&]() {
    if (is_normal) {
      const Transform tfm = webgpu_svm_context_transform(surface, CYCLES_WEBGPU_SVM_OBJECT_INVERSE);
      return normalize(transform_direction_transposed(&tfm, in));
    }
    const Transform tfm = webgpu_svm_context_transform(surface,
                                                       CYCLES_WEBGPU_SVM_OBJECT_TRANSFORM);
    return is_direction ? transform_direction(&tfm, in) : transform_point(&tfm, in);
  };

  if (from == NODE_VECTOR_TRANSFORM_CONVERT_SPACE_WORLD) {
    if (to == NODE_VECTOR_TRANSFORM_CONVERT_SPACE_CAMERA) {
      in = world_to_camera();
    }
    else if (to == NODE_VECTOR_TRANSFORM_CONVERT_SPACE_OBJECT) {
      in = world_to_object();
    }
  }
  else if (from == NODE_VECTOR_TRANSFORM_CONVERT_SPACE_CAMERA) {
    if (to == NODE_VECTOR_TRANSFORM_CONVERT_SPACE_WORLD ||
        to == NODE_VECTOR_TRANSFORM_CONVERT_SPACE_OBJECT)
    {
      in = camera_to_world();
    }
    if (to == NODE_VECTOR_TRANSFORM_CONVERT_SPACE_OBJECT) {
      in = world_to_object();
    }
  }
  else if (from == NODE_VECTOR_TRANSFORM_CONVERT_SPACE_OBJECT) {
    if (to == NODE_VECTOR_TRANSFORM_CONVERT_SPACE_WORLD ||
        to == NODE_VECTOR_TRANSFORM_CONVERT_SPACE_CAMERA)
    {
      in = object_to_world();
    }
    if (to == NODE_VECTOR_TRANSFORM_CONVERT_SPACE_CAMERA) {
      in = world_to_camera();
    }
  }
  return in;
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
template<typename Fetch,
         typename FetchImage,
         typename FetchSurface = WebGPUSvmNoSurface,
         typename Host = WebGPUSvmPackedHost>
ccl_device_inline bool webgpu_svm_eval_material(Fetch fetch,
                                               FetchImage image,
                                               const WebGPUSvmUv uv,
                                               const float3 generated,
                                               const WebGPUSvmColorSpace &color_space,
                                               bool &uses_uv,
                                               bool &uses_generated,
                                               const uint word_count,
                                               const uint shader_count,
                                               const uint packed_shader,
                                               WebGPUSvmMaterial &material,
                                               FetchSurface raw_surface = {},
                                               const Host host = {},
                                               const WebGPUSvmBumpInputs bump = {},
                                               const bool ordered_closures = false)
{
  material.requires_extended_svm = false;
  material.requires_full_svm = false;
  material.requires_advanced_procedural_svm = false;
  material.uses_compact_voronoi_distance = false;
  material.varying_transparency = false;
  material.principled_alpha = false;
  material.mixed_transparency = false;
  material.ggx = false;
  material.anisotropy_aspect = 0.0f;
  material.standalone_transmission = 0.0f;
  material.transmission = 0.0f;
  material.subsurface = 0.0f;
  material.transparency = zero_spectrum();
  material.transparent_sample_weight = 0.0f;
  material.diffuse_sample_weight = 0.0f;
  material.diffuse_principled_lobes = 0;
  material.ordered_lobe_candidates = 0;
  material.ordered_lobe_count = 0;
  material.ordered_lobe_types = 0u;
  material.ordered_lobe_features = 0u;
  material.ordered_glossy_mix = false;
  material.mixed_glossy_flags = 0u;
  material.transparent_first = 0;
  material.transparent_emission = 0;
  material.principled_translucent = 0;
  material.pixel_wireframe = false;
  material.surface_bump = false;
  material.ordered_principled_normal = false;
  material.ordered_principled_layers = false;
  material.ordered_principled_sheen = false;
  material.ordered_principled_specular_candidates = 0;
  material.ordered_principled_metal_candidates = 0;
  material.ordered_principled_tangent_0 = zero_float4();
  material.ordered_principled_tangent_1 = zero_float4();
  material.ordered_principled_coat = zero_float4();
  material.coat = {};
  material.single_principled_specular_candidate = false;
  material.coated_principled_emission = false;
  material.coat_emission = zero_spectrum();
  material.displacement_bump = false;
  material.displacement_normal = zero_float3();
  WebGPUSvmDisplacementBump displacement;
  const auto surface = [&](const uint field) -> WebGPUSvmContextWords {
    if (field == CYCLES_WEBGPU_SVM_NORMAL &&
        (material.displacement_bump || displacement.offset != SVM_STACK_INVALID))
    {
      const float3 normal = material.displacement_bump ? material.displacement_normal :
                                                        displacement.normal;
      return {__float_as_uint(normal.x), __float_as_uint(normal.y), __float_as_uint(normal.z), 0};
    }
    if (field == CYCLES_WEBGPU_SVM_POSITION && displacement.offset != SVM_STACK_INVALID) {
      const float3 P = displacement.position;
      return {__float_as_uint(P.x), __float_as_uint(P.y), __float_as_uint(P.z), raw_surface(field).w};
    }
    return raw_surface(field);
  };
  const auto position_differential = [&]() {
    return displacement.offset == SVM_STACK_INVALID ? bump.position : displacement.compact;
  };
  material.closure_normal = zero_float3();
  material.closure_normal_overridden = false;
  material.holdout = false;
  material.surface_closure = 0.0f;
  material.emission = zero_float3();
  bool has_emission = false;
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
#ifdef CYCLES_WEBGPU_SAMPLED_EMISSION
  static_assert(SVM_STACK_SIZE == SVM_STACK_INVALID);
  float stack[SVM_STACK_SIZE + 3]; /* Default normal for Bump. */
#else
  float stack[SVM_STACK_SIZE];
#endif
  WebGPUSvmStackUsage written;
  WebGPUSvmStackUsage varying;
  Spectrum weight = zero_spectrum();
  bool weight_varying = false;
  uint closure_bump_word = 0;
  WebGPUSvmNormalSlots surface_normals;
  bool has_closure = false;
  bool has_transparent = false;
  bool single_explicit_transparent = false;
  bool has_explicit_transparent = false;
  bool has_translucent = false;
  bool has_glossy = false;
  bool glossy_nonsingular = false;
  bool has_principled = false;
  bool has_diffuse_principled = false;
  bool mixable_principled = false;
  bool translucent_mixable_principled = false;
  uint lambert_closures = 0;
  bool mixes_closures = false;
  uint ordered_candidates = 0;
  uint ordered_count = 0;
  uint ordered_principled_branches = 0;
  bool ordered_has_glossy = false;

  while (offset < end) {
    const uint type = fetch(offset++);
    host.visit(type, offset - 1);
    switch (type) {
      case NODE_AOV_START:
        if (ordered_closures) {
          return false;
        }
        /* Match stock SVM's early exit when this path cannot write film AOVs. */
        if (!host.evaluate_aov()) {
          offset = end - 1;
        }
        break;
      case NODE_AOV_COLOR: {
        SVMNodeAOVColor node;
        if (!webgpu_svm_read(fetch, offset, end, node) || node.aov_offset < 0 ||
            node.aov_offset % 4 != 0 || !webgpu_svm_inputs_valid(written, node.color))
        {
          return false;
        }
        host.write_aov(node.aov_offset, true, make_float4(stack_load(stack, node.color), 1.0f));
        break;
      }
      case NODE_AOV_VALUE: {
        SVMNodeAOVValue node;
        if (!webgpu_svm_read(fetch, offset, end, node) || node.aov_offset < 0 ||
            !webgpu_svm_inputs_valid(written, node.value))
        {
          return false;
        }
        host.write_aov(node.aov_offset, false, make_float4(stack_load(stack, node.value), 0, 0, 0));
        break;
      }
      case NODE_END: {
#ifdef CYCLES_WEBGPU_SAMPLED_EMISSION
        return offset == end;
#endif
        if (ordered_closures) {
          if (ordered_principled_branches != 0u) {
            material.ordered_lobe_candidates = ordered_candidates;
            material.ordered_lobe_count = ordered_principled_branches;
            material.surface_closure = CYCLES_WEBGPU_MATERIAL_ORDERED_PRINCIPLED;
            return offset == end && ordered_principled_branches == 2u;
          }
          const bool transparent =
              (material.ordered_lobe_types & CYCLES_WEBGPU_ORDERED_HAS_TRANSPARENT) != 0u;
          const bool translucent =
              (material.ordered_lobe_types & CYCLES_WEBGPU_ORDERED_HAS_TRANSLUCENT) != 0u;
          if ((ordered_has_glossy && translucent) ||
              (transparent && !translucent && !ordered_has_glossy))
          {
            return false;
          }
          material.ordered_lobe_candidates = ordered_candidates;
          material.ordered_lobe_count = ordered_count;
          material.ordered_glossy_mix |= ordered_has_glossy && transparent;
          material.mixed_transparency = transparent;
          material.surface_closure = CYCLES_WEBGPU_MATERIAL_ORDERED_CLOSURES;
          return offset == end && ordered_candidates >= 2;
        }
        const bool diffuse_principled_mix = has_diffuse_principled && lambert_closures > 1;
        if (diffuse_principled_mix && has_emission) {
          return false;
        }
        if (material.coat.possible && (has_transparent || has_translucent || has_emission)) {
          return false;
        }
        if (material.coated_principled_emission && material.surface_bump) {
          return false;
        }
        /* The host validates enough closure slots even when some of these branches
         * happen to fall below the stock weight cutoff at its probe point. */
        material.diffuse_principled_lobes = diffuse_principled_mix ? lambert_closures : 0u;
        if (has_emission &&
            (material.surface_closure == CYCLES_WEBGPU_MATERIAL_RAY_PORTAL ||
             (has_transparent &&
              (!single_explicit_transparent || lambert_closures != 0 || has_principled ||
               has_glossy || has_translucent))))
        {
          return false;
        }
        material.transparent_emission = uint(has_emission && has_transparent);
        const bool principled_translucent = has_principled && has_translucent;
        if (principled_translucent) {
          if (has_explicit_transparent || has_emission || material.surface_bump ||
              material.displacement_bump || material.closure_normal_overridden ||
              !translucent_mixable_principled)
          {
            return false;
          }
          material.principled_translucent = 1u;
        }
        if (has_principled && ((has_transparent && !mixable_principled) ||
                               (!has_transparent && mixes_closures && !principled_translucent))) {
          return false;
        }
        if (has_transparent) {
          if (lambert_closures != 0 || has_principled || has_glossy) {
            material.mixed_transparency = true;
            if (!material.ggx) {
              material.surface_closure = CYCLES_WEBGPU_MATERIAL_MIXED_TRANSPARENT;
            }
          }
          else {
            material.surface_closure = CYCLES_WEBGPU_MATERIAL_TRANSPARENT;
            material.color = material.transparency;
          }
        }
        /* A single isotropic Glossy uses the same two normal records as Glass.
         * The shader-wide normal remains sd.N; the closure keeps its local Bump. */
        const bool standalone_glossy = has_glossy && lambert_closures == 0 &&
                                       material.glossy_isotropic;
        /* A Lambert closure can also use these records. A Lambert and Glossy
         * pair must share the local Bump, since both records are occupied. */
        const uint separate_normals = CYCLES_WEBGPU_MIXED_BUMP_INDEPENDENT |
                                      CYCLES_WEBGPU_MIXED_BUMP_DIFFUSE_ONLY |
                                      CYCLES_WEBGPU_MIXED_BUMP_GLOSSY_ONLY;
        const bool lambert_bump = lambert_closures == 1 && !has_translucent &&
            (!has_glossy || (material.glossy_isotropic && glossy_nonsingular &&
                            (material.mixed_glossy_flags & separate_normals) == 0u));
        /* Opaque isotropic Principled also has an unused auxiliary record for
         * the shader-wide normal while its closures use a linked normal. */
        const bool principled_bump = has_principled && material.ggx &&
                                     material.transmission == 0.0f && material.subsurface == 0.0f &&
                                     material.anisotropy_aspect == 0.0f;
        if (material.displacement_bump &&
            ((material.closure_normal_overridden && material.standalone_transmission == 0.0f &&
              !standalone_glossy && !lambert_bump && !principled_bump) ||
             has_transparent || has_emission))
        {
          return false;
        }
        if (has_translucent && !principled_translucent) {
          if (lambert_closures != 0) {
            if (material.closure_normal_overridden) {
              return false;
            }
            material.surface_closure = CYCLES_WEBGPU_MATERIAL_DIFFUSE_TRANSLUCENT;
          }
          else {
            material.surface_closure = CYCLES_WEBGPU_MATERIAL_TRANSLUCENT;
            material.color = material.translucent_color;
            material.surface_parameters = zero_float2();
          }
        }
        if (has_glossy) {
          if (lambert_closures != 0) {
            if (!material.glossy_isotropic) {
              return false;
            }
            material.surface_closure = CYCLES_WEBGPU_MATERIAL_DIFFUSE_GLOSSY;
          }
          else {
            material.color = material.glossy_color;
            if (has_transparent) {
              if (!single_explicit_transparent || !material.glossy_isotropic ||
                  !glossy_nonsingular || material.surface_bump)
              {
                return false;
              }
              material.surface_closure = CYCLES_WEBGPU_MATERIAL_GLOSSY_TRANSPARENT;
            }
            else {
              material.surface_closure = CYCLES_WEBGPU_MATERIAL_GLOSSY;
            }
          }
        }
        if (!has_closure) {
          /* Emission alone, or a shader the compiler reduced to no closures
           * (for example black or zero-strength emission): no scattering,
           * light sampling or bounce. */
          material.color = zero_float3();
          material.ggx = false;
          material.transmission = 0.0f;
          material.standalone_transmission = 0.0f;
          material.subsurface = 0.0f;
        }
        return offset == end;
      }
      case NODE_GEOMETRY:
      case NODE_GEOMETRY_DERIVATIVE: {
        SVMNodeGeometry node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            node.bump_offset > NODE_BUMP_OFFSET_DY || node.store_derivatives ||
            (node.bump_offset != NODE_BUMP_OFFSET_CENTER &&
             node.geom_type != NODE_GEOM_P && node.geom_type != NODE_GEOM_N &&
             node.geom_type != NODE_GEOM_Ng && node.geom_type != NODE_GEOM_uv) ||
            uint(node.out_offset) + 2 >= SVM_STACK_SIZE)
        {
          return false;
        }
        float3 value;
        switch (node.geom_type) {
          case NODE_GEOM_P:
            value = webgpu_svm_bump_offset(
                webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_POSITION)),
                position_differential(), node.bump_offset, node.bump_filter_width);
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
          case NODE_GEOM_T:
            /* primitive_tangent() from the synchronized Generated coordinates. */
            uses_generated = true;
            value = webgpu_svm_surface_tangent(
                surface, make_float3(-(generated.y - 0.5f), (generated.x - 0.5f), 0.0f));
            break;
          case NODE_GEOM_uv: {
            const auto hit = surface(CYCLES_WEBGPU_SVM_INTERSECTION);
            const float u = __uint_as_float(hit.y);
            const float v = __uint_as_float(hit.z);
            /* Geometry's Parametric output uses the stock triangle barycentrics. */
            const differential3 derivatives{
                make_float3(-bump.u.dx - bump.v.dx, bump.u.dx, 0.0f),
                make_float3(-bump.u.dy - bump.v.dy, bump.u.dy, 0.0f)};
            value = webgpu_svm_bump_offset(make_float3(1.0f - u - v, u, 0.0f),
                                           derivatives, node.bump_offset, node.bump_filter_width);
            break;
          }
          default:
            return false;
        }
        stack_store_float(stack, node.out_offset, value.x);
        stack_store_float(stack, uint(node.out_offset) + 1, value.y);
        stack_store_float(stack, uint(node.out_offset) + 2, value.z);
        webgpu_svm_mark_outputs(
            written, varying, surface_normals, SVM_STACK_INVALID, node.out_offset);
        if (node.geom_type == NODE_GEOM_N) {
          surface_normals.base = surface_normals.active = node.out_offset;
        }
        break;
      }
#if !defined(CYCLES_WEBGPU_RUNTIME_SVM) || defined(CYCLES_WEBGPU_SCENE_QUERIES) || \
    defined(CYCLES_WEBGPU_SAMPLED_EMISSION) || defined(CYCLES_WEBGPU_SHADOW_SVM)
      case NODE_AMBIENT_OCCLUSION: {
        SVMNodeAmbientOcclusion node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            !webgpu_svm_inputs_valid(written, node.dist, node.color) ||
            (stack_valid(node.normal_offset) &&
             !webgpu_svm_stack_has3(written, node.normal_offset)) ||
            !webgpu_svm_outputs_valid(node.out_ao_offset, node.out_color_offset))
        {
          return false;
        }
        const float3 N = webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_NORMAL));
        const float3 normal = safe_normalize(stack_load_float3_default(stack, node.normal_offset, N));
        const float ao = host.ambient_occlusion(normal, stack_load(stack, node.dist),
                                                node.samples, node.flags);
        if (stack_valid(node.out_ao_offset)) {
          stack_store_float(stack, node.out_ao_offset, ao);
        }
        if (stack_valid(node.out_color_offset)) {
          stack_store_float3(stack, node.out_color_offset, ao * stack_load(stack, node.color));
        }
        webgpu_svm_mark_outputs(
            written, varying, surface_normals, node.out_ao_offset, node.out_color_offset);
        break;
      }
#  if !defined(CYCLES_WEBGPU_AO_ONLY_QUERIES)
      case NODE_BEVEL: {
        SVMNodeBevel node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            !webgpu_svm_inputs_valid(written, node.radius) ||
            (stack_valid(node.normal_offset) &&
             !webgpu_svm_stack_has3(written, node.normal_offset)) ||
            uint(node.out_offset) + 2 >= SVM_STACK_SIZE)
        {
          return false;
        }
        const float3 N = webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_NORMAL));
        float3 bevel_N = host.bevel(N, stack_load(stack, node.radius), node.num_samples);
        if constexpr (Host::uses_ray_queries) {
          if (stack_valid(node.normal_offset)) {
            /* Preserve the linked input normal, as svm_node_bevel() does. */
            bevel_N = normalize(stack_load_float3(stack, node.normal_offset) + (bevel_N - N));
          }
        }
        stack_store_float3(stack, node.out_offset, bevel_N);
        webgpu_svm_mark_outputs(
            written, varying, surface_normals, SVM_STACK_INVALID, node.out_offset);
        break;
      }
      case NODE_RAYCAST: {
        SVMNodeRaycast node;
        if (!webgpu_svm_read(fetch, offset, end, node) || node.bump_filter_width != 0.0f ||
            node.num_attributes != 0 ||
            !webgpu_svm_inputs_valid(written, node.position, node.direction, node.distance) ||
            !webgpu_svm_outputs_valid(node.is_hit_offset, node.hit_position_offset) ||
            !webgpu_svm_outputs_valid(node.is_self_hit_offset, node.hit_normal_offset) ||
            !webgpu_svm_outputs_valid(node.hit_distance_offset, SVM_STACK_INVALID))
        {
          return false;
        }
        const float3 position = stack_load(stack, node.position);
        const float3 direction = stack_load(stack, node.direction);
        const auto hit = host.raycast(position, direction, stack_load(stack, node.distance),
                                      node.only_local);
        if (stack_valid(node.is_hit_offset)) {
          stack_store_float(stack, node.is_hit_offset, hit.is_hit);
        }
        if (stack_valid(node.is_self_hit_offset)) {
          stack_store_float(stack, node.is_self_hit_offset, hit.is_self_hit);
        }
        if (stack_valid(node.hit_distance_offset)) {
          stack_store_float(stack, node.hit_distance_offset, hit.distance);
        }
        if (stack_valid(node.hit_position_offset)) {
          stack_store_float3(stack, node.hit_position_offset,
                             hit.is_hit != 0.0f ? position + direction * hit.distance : zero_float3());
        }
        if (stack_valid(node.hit_normal_offset)) {
          stack_store_float3(stack, node.hit_normal_offset, hit.normal);
        }
        webgpu_svm_mark_outputs(
            written, varying, surface_normals, node.is_hit_offset, node.hit_position_offset);
        webgpu_svm_mark_outputs(
            written, varying, surface_normals, node.is_self_hit_offset, node.hit_normal_offset);
        webgpu_svm_mark_outputs(
            written, varying, surface_normals, node.hit_distance_offset, SVM_STACK_INVALID);
        break;
      }
#  endif
#endif
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
            written, varying, surface_normals, node.zdepth_offset, node.vector_offset);
        webgpu_svm_mark_outputs(
            written, varying, surface_normals, node.distance_offset, SVM_STACK_INVALID);
        break;
      }
      case NODE_TEX_COORD:
      case NODE_TEX_COORD_DERIVATIVE: {
        /* Read the packed fields as words so the shader compiler does not
         * widen this two-word record to a 64-bit integer. */
        static_assert(sizeof(SVMNodeTexCoord) == 2 * sizeof(uint));
        if (end - offset < 2) {
          return false;
        }
        const uint flags = fetch(offset++);
        const float filter_width = std::bit_cast<float>(fetch(offset++));
        const uint coordinate = flags & 0xffu;
        const uint bump_offset = (flags >> 8) & 0xffu;
        const uint out_offset = flags >> 24;
        if (bump_offset > NODE_BUMP_OFFSET_DY || (flags & 0xff0000u) != 0 ||
            (bump_offset != NODE_BUMP_OFFSET_CENTER && coordinate != NODE_TEXCO_OBJECT &&
             coordinate != NODE_TEXCO_OBJECT_WITH_TRANSFORM) ||
            out_offset + 2 >= SVM_STACK_SIZE)
        {
          return false;
        }
        const auto object_coordinate = [&](const Transform &inverse) {
          /* Match svm_node_tex_coord_derivative(): translate P, transform
           * the differential as a direction, then apply the filter width. */
          const float3 position = transform_point(
              &inverse, webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_POSITION)));
          if (bump_offset == NODE_BUMP_OFFSET_CENTER) {
            return position;
          }
          const differential3 position_diff = position_differential();
          const float3 differential = bump_offset == NODE_BUMP_OFFSET_DX ? position_diff.dx :
                                                                         position_diff.dy;
          return position + transform_direction(&inverse, differential) * filter_width;
        };
        float3 value;
        switch (coordinate) {
          case NODE_TEXCO_CAMERA:
            value = webgpu_svm_camera_position(surface);
            break;
          case NODE_TEXCO_WINDOW:
            value = webgpu_svm_window_position(surface);
            break;
          case NODE_TEXCO_DUPLI_GENERATED:
            value = webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_DUPLI_GENERATED));
            break;
          case NODE_TEXCO_DUPLI_UV:
            value = webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_DUPLI_UV));
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
            value = object_coordinate(inverse);
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
            value = object_coordinate(transform);
            break;
          }
          default:
            return false;
        }
        stack_store_float(stack, out_offset, value.x);
        stack_store_float(stack, out_offset + 1, value.y);
        stack_store_float(stack, out_offset + 2, value.z);
        webgpu_svm_mark_outputs(
            written, varying, surface_normals, SVM_STACK_INVALID, out_offset);
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
              written, varying, surface_normals, SVM_STACK_INVALID, node.out_offset);
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
              written, varying, surface_normals, node.out_offset, SVM_STACK_INVALID);
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
            written, varying, surface_normals, node.out_offset, SVM_STACK_INVALID);
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
            written, varying, surface_normals, node.out_offset, SVM_STACK_INVALID);
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
            written, varying, surface_normals, node.out_offset, SVM_STACK_INVALID);
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
            written, varying, surface_normals, node.out_offset, SVM_STACK_INVALID);
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
        surface_normals.invalidate(node.out_offset, 1);
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
        surface_normals.invalidate(node.out_offset, 3);
        break;
      }
      case NODE_DISPLACEMENT: {
        SVMNodeDisplacement node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            (node.space != NODE_NORMAL_MAP_OBJECT && node.space != NODE_NORMAL_MAP_WORLD) ||
            !webgpu_svm_inputs_valid(written, node.height, node.midlevel, node.scale) ||
            (stack_valid(node.normal_offset) && !webgpu_svm_stack_has3(written, node.normal_offset)) ||
            !stack_valid(node.out_offset) ||
            !webgpu_svm_outputs_valid(SVM_STACK_INVALID, node.out_offset))
        {
          return false;
        }
        const float scale = (stack_load(stack, node.height) - stack_load(stack, node.midlevel)) *
                            stack_load(stack, node.scale);
        float3 displacement = stack_load_float3_default(
            stack, node.normal_offset, webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_NORMAL)));
        /* svm_node_displacement(): the synchronized static transform replaces
         * the stock object table and its identity transform for baked geometry. */
        if (node.space == NODE_NORMAL_MAP_OBJECT) {
          const Transform tfm = webgpu_svm_context_transform(surface,
                                                            CYCLES_WEBGPU_SVM_OBJECT_TRANSFORM);
          displacement = transform_direction(
              &tfm, safe_normalize(transform_direction_transposed(&tfm, displacement)) * scale);
        }
        else {
          displacement *= scale;
        }
        stack_store_float3(stack, node.out_offset, displacement);
        webgpu_svm_mark_outputs(
            written, varying, surface_normals, SVM_STACK_INVALID, node.out_offset);
        break;
      }
      case NODE_ENTER_BUMP_EVAL: {
        static_assert(sizeof(SVMNodeEnterBumpEval) == sizeof(uint));
        if (offset == end || ordered_closures || has_closure || material.displacement_bump) {
          return false;
        }
        const uint packed = host.displacement_bump_slots(offset, fetch(offset));
        ++offset;
        if (!webgpu_svm_enter_displacement_bump(
                host, raw_surface, bump, packed, (packed_shader & SHADER_SMOOTH_NORMAL) != 0,
                displacement))
        {
          return false;
        }
        break;
      }
      case NODE_LEAVE_BUMP_EVAL: {
        SVMNodeLeaveBumpEval node;
        if (!webgpu_svm_read(fetch, offset, end, node) || !material.displacement_bump ||
            displacement.offset == SVM_STACK_INVALID || node.state_offset != displacement.offset)
        {
          return false;
        }
        displacement.offset = SVM_STACK_INVALID;
        break;
      }
      case NODE_CLOSURE_SET_NORMAL: {
        SVMNodeClosureSetNormal node;
        if (ordered_closures || !webgpu_svm_read(fetch, offset, end, node) || !material.surface_bump ||
            material.displacement_bump || has_closure ||
            surface_normals.find_bump(node.direction_offset) == 0u ||
            !webgpu_svm_stack_has3(written, node.direction_offset) ||
            !webgpu_svm_outputs_valid(SVM_STACK_INVALID, node.normal_offset))
        {
          return false;
        }
        /* Stock svm_node_set_normal() sets ShaderData.N for the following
         * surface program, including unconnected closure normals. */
        material.displacement_normal = stack_load_float3(stack, node.direction_offset);
        material.displacement_bump = true;
        if (stack_valid(node.normal_offset)) {
          stack_store_float3(stack, node.normal_offset, material.displacement_normal);
          webgpu_svm_mark_outputs(
              written, varying, surface_normals, SVM_STACK_INVALID, node.normal_offset);
        }
        surface_normals.base = surface_normals.active = node.normal_offset;
        surface_normals.bump = surface_normals.previous_bump = SVM_STACK_INVALID;
        break;
      }
      case NODE_SET_BUMP: {
        SVMNodeSetBump node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            (node.bump_state_offset != SVM_STACK_INVALID &&
             node.bump_state_offset != displacement.offset) ||
            !webgpu_svm_inputs_valid(written, node.scale, node.strength) ||
            (stack_valid(node.normal_offset) && !webgpu_svm_stack_has3(written, node.normal_offset)) ||
            !webgpu_svm_stack_has(written, node.center_offset) ||
            !webgpu_svm_stack_has(written, node.dx_offset) ||
            !webgpu_svm_stack_has(written, node.dy_offset) ||
            !stack_valid(node.out_offset) ||
            !webgpu_svm_outputs_valid(SVM_STACK_INVALID, node.out_offset))
        {
          return false;
        }
        const differential3 dP = node.bump_state_offset == SVM_STACK_INVALID ?
                                     position_differential() : displacement.full;
        const float3 normal = webgpu_svm_set_bump(node, stack, surface, dP);
        stack_store_float3(stack, node.out_offset, normal);
        webgpu_svm_mark_outputs(
            written, varying, surface_normals, SVM_STACK_INVALID, node.out_offset);
        surface_normals.set_bump(node.out_offset, offset);
        material.surface_bump = true;
        break;
      }
      case NODE_ATTR:
      case NODE_ATTR_DERIVATIVE: {
        const uint attribute_word = offset;
        SVMNodeAttr node;
        if (!webgpu_svm_read(fetch, offset, end, node)) {
          return false;
        }
        if (node.attr != ATTR_STD_UV && node.attr != ATTR_STD_GENERATED) {
          /* Stock svm_node_attr_surface_eval() for a packed attribute slot, or
           * for an attribute that find_attribute() does not locate. */
          const bool vector = node.output_type == NODE_ATTR_OUTPUT_FLOAT3;
          const int resolved = host.attribute(attribute_word, node.attr);
          const bool slot = resolved != int(ATTR_STD_NOT_FOUND) &&
                            (uint(resolved) & webgpu_svm_attribute_slot) != 0;
          if ((resolved != int(ATTR_STD_NOT_FOUND) && !slot) ||
              node.output_type > NODE_ATTR_OUTPUT_FLOAT_ALPHA ||
              node.bump_offset > NODE_BUMP_OFFSET_DY || node.store_derivatives ||
              !webgpu_svm_inputs_valid(written, node.missing, node.missing_alpha) ||
              !webgpu_svm_outputs_valid(vector ? SVM_STACK_INVALID : uint(node.out_offset),
                                        vector ? uint(node.out_offset) : SVM_STACK_INVALID))
          {
            return false;
          }
          WebGPUSvmAttributeSample sample{WEBGPU_SVM_ATTRIBUTE_MISSING, 0, zero_float4()};
          float du = 0.0f;
          float dv = 0.0f;
          if (node.bump_offset == NODE_BUMP_OFFSET_DX) {
            du = bump.u.dx;
            dv = bump.v.dx;
          }
          else if (node.bump_offset == NODE_BUMP_OFFSET_DY) {
            du = bump.u.dy;
            dv = bump.v.dy;
          }
          if (slot && !host.attribute_sample(uint(resolved) & ~webgpu_svm_attribute_slot,
                                             sample, du, dv, node.bump_filter_width))
          {
            return false;
          }
          float3 value;
          if (sample.mode == WEBGPU_SVM_ATTRIBUTE_MISSING) {
            value = node.output_type == NODE_ATTR_OUTPUT_FLOAT_ALPHA ?
                        make_float3(stack_load(stack, node.missing_alpha)) :
                        stack_load(stack, node.missing);
          }
          else if (!webgpu_svm_attribute_value(sample, node.output_type, value)) {
            return false;
          }
          if (vector) {
            stack_store_float3(stack, node.out_offset, value);
          }
          else {
            /* svm_node_attr_store() averages the scalar outputs. */
            stack_store_float(stack, node.out_offset, average(value));
          }
          webgpu_svm_mark_outputs(written,
                                  varying,
                                  surface_normals,
                                  vector ? SVM_STACK_INVALID : uint(node.out_offset),
                                  vector ? uint(node.out_offset) : SVM_STACK_INVALID);
          break;
        }
        if (node.output_type != NODE_ATTR_OUTPUT_FLOAT3 ||
            node.bump_offset > NODE_BUMP_OFFSET_DY || node.store_derivatives ||
            uint(node.out_offset) + 2 >= SVM_STACK_SIZE)
        {
          return false;
        }
        const bool uv_attribute = node.attr == ATTR_STD_UV;
        uses_uv |= uv_attribute;
        uses_generated |= !uv_attribute;
        const float3 values = uv_attribute ?
                                  webgpu_svm_bump_offset(make_float3(uv.x, uv.y, 0.0f),
                                                         bump.uv, node.bump_offset,
                                                         node.bump_filter_width) :
                                  webgpu_svm_bump_offset(generated, bump.generated, node.bump_offset,
                                                         node.bump_filter_width);
        stack_store_float(stack, node.out_offset, values.x);
        stack_store_float(stack, uint(node.out_offset) + 1, values.y);
        stack_store_float(stack, uint(node.out_offset) + 2, values.z);
        webgpu_svm_stack_mark3(written, node.out_offset);
        webgpu_svm_stack_mark3(varying, node.out_offset);
        surface_normals.invalidate(node.out_offset, 3);
        break;
      }
      case NODE_NORMAL_MAP: {
        /* svm_node_normal_map() with packed tangent, sign and smooth normal slots. */
        constexpr uint words = sizeof(uint);
        const uint attribute_word = offset + offsetof(SVMNodeNormalMap, attr) / words;
        const uint sign_word = offset + offsetof(SVMNodeNormalMap, attr_sign) / words;
        const uint base_word = offset + offsetof(SVMNodeNormalMap, use_original_base) / words;
        SVMNodeNormalMap node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            node.space > NODE_NORMAL_MAP_BLENDER_WORLD ||
            !webgpu_svm_inputs_valid(written, node.color, node.strength) ||
            !stack_valid(node.normal_offset) ||
            !webgpu_svm_outputs_valid(SVM_STACK_INVALID, node.normal_offset))
        {
          return false;
        }
        const float3 shading_N = webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_NORMAL));
        const bool is_backfacing = surface(CYCLES_WEBGPU_SVM_INTERSECTION).w == 2;
        float3 color = stack_load(stack, node.color);
        color = 2.0f * make_float3(color.x - 0.5f, color.y - 0.5f, color.z - 0.5f);
        if (node.invert_green) {
          color.y = -color.y;
        }
        float3 N;
        float strength = stack_load(stack, node.strength);
        bool linear_interpolate_strength = false;
        bool unperturbed = false;
        if (node.space == NODE_NORMAL_MAP_TANGENT) {
          WebGPUSvmAttributeSample tangent_sample;
          WebGPUSvmAttributeSample sign_sample;
          if (!webgpu_svm_sample_resolved(
                  host, host.attribute(attribute_word, node.attr), tangent_sample) ||
              !webgpu_svm_sample_resolved(
                  host, host.attribute(sign_word, node.attr_sign), sign_sample))
          {
            return false;
          }
          const int smooth_slot = host.synthetic(
              base_word, node.use_original_base, WEBGPU_SVM_SLOT_SMOOTH_NORMAL);
          if (!webgpu_svm_is_slot(smooth_slot)) {
            return false;
          }
          unperturbed = tangent_sample.mode == WEBGPU_SVM_ATTRIBUTE_MISSING ||
                        sign_sample.mode == WEBGPU_SVM_ATTRIBUTE_MISSING;
          if (!unperturbed) {
            if (tangent_sample.type != NODE_ATTR_FLOAT3 || sign_sample.type != NODE_ATTR_FLOAT) {
              return false;
            }
            const float3 tangent = make_float3(tangent_sample.value);
            const float sign = sign_sample.value.x;
            float3 normal;
            if (packed_shader & SHADER_SMOOTH_NORMAL) {
              WebGPUSvmAttributeSample smooth;
              if (!host.attribute_sample(uint(smooth_slot) & ~webgpu_svm_attribute_slot, smooth) ||
                  smooth.mode == WEBGPU_SVM_ATTRIBUTE_MISSING)
              {
                return false;
              }
              /* Original normals use linear strength in world space, and the
               * tangent frame interpolates their unnormalized corner values. */
              linear_interpolate_strength = smooth.value.w != 0.0f;
              normal = linear_interpolate_strength ? make_float3(smooth.value) :
                                                     safe_normalize(make_float3(smooth.value));
              if (is_zero(normal)) {
                normal = webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_GEOMETRIC));
              }
            }
            else {
              normal = webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_GEOMETRIC));
              /* The normal is already inverted, which is too soon for the math here. */
              if (is_backfacing) {
                normal = -normal;
              }
              /* object_inverse_normal_transform() */
              const Transform tfm = webgpu_svm_context_transform(
                  surface, CYCLES_WEBGPU_SVM_OBJECT_TRANSFORM);
              normal = safe_normalize(transform_direction_transposed(&tfm, normal));
            }
            if (!linear_interpolate_strength) {
              color.x *= strength;
              color.y *= strength;
              color.z = mix(1.0f, color.z, saturatef(strength));
            }
            const float3 B = sign * cross(normal, tangent);
            N = safe_normalize(to_global(color, tangent, B, normal));
            /* object_normal_transform() */
            const Transform itfm = webgpu_svm_context_transform(surface,
                                                                CYCLES_WEBGPU_SVM_OBJECT_INVERSE);
            N = normalize(transform_direction_transposed(&itfm, N));
            if (is_backfacing) {
              N = -N;
            }
          }
        }
        else {
          linear_interpolate_strength = true;
          if (node.space == NODE_NORMAL_MAP_BLENDER_OBJECT ||
              node.space == NODE_NORMAL_MAP_BLENDER_WORLD)
          {
            color.y = -color.y;
            color.z = -color.z;
          }
          N = color;
          if (node.space == NODE_NORMAL_MAP_OBJECT || node.space == NODE_NORMAL_MAP_BLENDER_OBJECT)
          {
            const Transform itfm = webgpu_svm_context_transform(surface,
                                                                CYCLES_WEBGPU_SVM_OBJECT_INVERSE);
            N = normalize(transform_direction_transposed(&itfm, N));
          }
          else {
            N = safe_normalize(N);
          }
          if (is_backfacing) {
            N = -N;
          }
        }
        if (unperturbed) {
          N = shading_N;
        }
        else {
          if (linear_interpolate_strength && strength != 1.0f) {
            strength = max(strength, 0.0f);
            N = safe_normalize(shading_N + (N - shading_N) * strength);
          }
          if (is_zero(N) || !isfinite_safe(N)) {
            N = shading_N;
          }
        }
        stack_store_float3(stack, node.normal_offset, N);
        webgpu_svm_mark_outputs(
            written, varying, surface_normals, SVM_STACK_INVALID, node.normal_offset);
        /* Retain the mapped normal until its closure snapshots it. The map
         * itself has no position differential requirement. */
        surface_normals.set_bump(node.normal_offset, attribute_word);
        break;
      }
      case NODE_TEX_SKY: {
        /* svm_node_tex_sky() for a surface material. */
        SVMNodeTexSky node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            node.sky_type > NODE_SKY_MULTIPLE_SCATTERING ||
            !webgpu_svm_stack_has3(written, node.dir_offset) || !stack_valid(node.out_offset) ||
            !webgpu_svm_outputs_valid(SVM_STACK_INVALID, node.out_offset))
        {
          return false;
        }
        const float3 dir = stack_load_float3(stack, node.dir_offset);
        float3 rgb;
        if (node.sky_type == NODE_SKY_PREETHAM || node.sky_type == NODE_SKY_HOSEK) {
          SVMNodeTexSkyPreethamData data;
          if (!webgpu_svm_read(fetch, offset, end, data)) {
            return false;
          }
          rgb = webgpu_svm_sky_analytic(color_space, node.sky_type, dir, data);
        }
        else {
          /* sky_radiance_nishita() without the Sun guiding bake. */
          SVMNodeTexSkyNishitaData data;
          if (!webgpu_svm_read(fetch, offset, end, data)) {
            return false;
          }
          if (!webgpu_svm_sky_nishita(color_space, dir, data, image, rgb)) {
            return false;
          }
        }
        stack_store_float3(stack, node.out_offset, rgb);
        webgpu_svm_mark_outputs(
            written, varying, surface_normals, SVM_STACK_INVALID, node.out_offset);
        break;
      }
      case NODE_IES: {
        SVMNodeIES node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            !webgpu_svm_stack_has3(written, node.vector_offset) ||
            !webgpu_svm_input_valid(written, node.strength) ||
            !webgpu_svm_outputs_valid(node.fac_offset, SVM_STACK_INVALID))
        {
          return false;
        }
        /* svm_node_ies() */
        float3 vector = stack_load_float3(stack, node.vector_offset);
        const float strength = stack_load(stack, node.strength);
        vector = normalize(vector);
        const float v_angle = safe_acosf(-vector.z);
        float h_angle;
        if (vector.x == 0.0f && vector.y == 0.0f) {
          /* C atan2f() of signed zeros. WGSL leaves atan2(0, 0) undefined. */
          const bool negative_x = (__float_as_uint(vector.x) >> 31) != 0;
          const bool negative_y = (__float_as_uint(vector.y) >> 31) != 0;
          h_angle = (negative_y ? (negative_x ? -M_PI_F : M_PI_F) : 0.0f) + M_PI_F;
        }
        else {
          h_angle = atan2f(vector.x, vector.y) + M_PI_F;
        }
        float ies;
        if (!host.ies(node.slot, h_angle, v_angle, ies)) {
          return false;
        }
        if (stack_valid(node.fac_offset)) {
          stack_store_float(stack, node.fac_offset, strength * ies);
          webgpu_svm_mark_outputs(
              written, varying, surface_normals, node.fac_offset, SVM_STACK_INVALID);
        }
        break;
      }
      case NODE_WIREFRAME: {
        /* Match the stock wireframe projection and triangle edge test. */
        const uint slot_word = offset + offsetof(SVMNodeWireframe, bump_filter_width) / sizeof(uint);
        SVMNodeWireframe node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            node.bump_offset != NODE_BUMP_OFFSET_CENTER ||
            !webgpu_svm_input_valid(written, node.in_size) ||
            !webgpu_svm_outputs_valid(node.out_fac_offset, SVM_STACK_INVALID))
        {
          return false;
        }
        const int slot = host.synthetic(
            slot_word, __float_as_int(node.bump_filter_width), WEBGPU_SVM_SLOT_POSITIONS);
        WebGPUSvmTriangleCorners triangle;
        if (!webgpu_svm_is_slot(slot) ||
            !host.triangle_corners(uint(slot) & ~webgpu_svm_attribute_slot, triangle))
        {
          return false;
        }
        float3 Co[3] = {triangle.corners[0], triangle.corners[1], triangle.corners[2]};
        if (triangle.flags & (webgpu_svm_slot_object_transform >> 16)) {
          /* object_position_transform() */
          const Transform tfm = webgpu_svm_context_transform(surface,
                                                             CYCLES_WEBGPU_SVM_OBJECT_TRANSFORM);
          Co[0] = transform_point(&tfm, Co[0]);
          Co[1] = transform_point(&tfm, Co[1]);
          Co[2] = transform_point(&tfm, Co[2]);
        }
        const float3 P = webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_POSITION));
        float pixelwidth = 1.0f;
        if (node.use_pixel_size) {
          const auto ray = surface(CYCLES_WEBGPU_SVM_DIFFERENTIAL);
          const float3 Ng = webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_GEOMETRIC));
          const float3 wi = -webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_DIRECTION));
          const float t = __uint_as_float(surface(CYCLES_WEBGPU_SVM_INTERSECTION).x);
          const float radius = differential_transfer_compact(
              __uint_as_float(ray.x), -wi, __uint_as_float(ray.y), t);
          const differential3 dP = differential_from_compact(Ng, radius);
          const float pixelwidth_x = len(dP.dx - dot(dP.dx, wi) * wi);
          const float pixelwidth_y = len(dP.dy - dot(dP.dy, wi) * wi);
          pixelwidth = (pixelwidth_x + pixelwidth_y) * 0.5f;
          material.pixel_wireframe = true;
        }
        pixelwidth *= 0.5f * stack_load(stack, node.in_size);
        pixelwidth *= pixelwidth;
        float f = 0.0f;
        for (int i = 0; i < 3; i++) {
          const int i2 = i ? i - 1 : 2;
          const float3 dir = P - Co[i];
          const float3 edge = Co[i] - Co[i2];
          const float3 crs = cross(edge, dir);
          if (dot(crs, crs) < (dot(edge, edge) * pixelwidth)) {
            f = 1.0f;
            break;
          }
        }
        if (stack_valid(node.out_fac_offset)) {
          stack_store_float(stack, node.out_fac_offset, f);
          webgpu_svm_mark_outputs(
              written, varying, surface_normals, node.out_fac_offset, SVM_STACK_INVALID);
        }
        break;
      }
      case NODE_TANGENT: {
        /* svm_node_tangent() with a packed UV tangent slot or the synchronized
         * Generated coordinates. */
        const uint attribute_word = offset + offsetof(SVMNodeTangent, attr) / sizeof(uint);
        SVMNodeTangent node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            node.direction_type > NODE_TANGENT_UVMAP || node.axis > NODE_TANGENT_AXIS_Z ||
            !stack_valid(node.tangent_offset) ||
            !webgpu_svm_outputs_valid(SVM_STACK_INVALID, node.tangent_offset))
        {
          return false;
        }
        float3 tangent = zero_float3();
        bool found = true;
        if (node.direction_type == NODE_TANGENT_UVMAP) {
          const int resolved = host.attribute(attribute_word, node.attr);
          WebGPUSvmAttributeSample sample{WEBGPU_SVM_ATTRIBUTE_MISSING, 0, zero_float4()};
          if (resolved != int(ATTR_STD_NOT_FOUND) &&
              ((uint(resolved) & webgpu_svm_attribute_slot) == 0 ||
               !host.attribute_sample(uint(resolved) & ~webgpu_svm_attribute_slot, sample)))
          {
            return false;
          }
          found = sample.mode != WEBGPU_SVM_ATTRIBUTE_MISSING;
          if (found) {
            if (sample.type != NODE_ATTR_FLOAT2 && sample.type != NODE_ATTR_FLOAT3) {
              return false;
            }
            tangent = sample.type == NODE_ATTR_FLOAT2 ?
                          make_float3(sample.value.x, sample.value.y, 0.0f) :
                          make_float3(sample.value);
          }
        }
        else {
          if (node.attr != ATTR_STD_GENERATED) {
            return false;
          }
          uses_generated = true;
          if (node.axis == NODE_TANGENT_AXIS_X) {
            tangent = make_float3(0.0f, -(generated.z - 0.5f), (generated.y - 0.5f));
          }
          else if (node.axis == NODE_TANGENT_AXIS_Y) {
            tangent = make_float3(-(generated.z - 0.5f), 0.0f, (generated.x - 0.5f));
          }
          else {
            tangent = make_float3(-(generated.y - 0.5f), (generated.x - 0.5f), 0.0f);
          }
        }
        /* A missing UV tangent stores zero without the normal projection. */
        stack_store_float3(
            stack, node.tangent_offset, found ? webgpu_svm_surface_tangent(surface, tangent) : tangent);
        webgpu_svm_mark_outputs(
            written, varying, surface_normals, SVM_STACK_INVALID, node.tangent_offset);
        break;
      }
      case NODE_HAIR_INFO: {
        /* svm_node_hair_info() on a triangle. Intercept, Length and Random
         * compile to attribute nodes. */
        SVMNodeHairInfo node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            node.info_type > NODE_INFO_CURVE_RANDOM)
        {
          return false;
        }
        if (node.info_type == NODE_INFO_CURVE_IS_STRAND ||
            node.info_type == NODE_INFO_CURVE_THICKNESS)
        {
          if (!webgpu_svm_outputs_valid(node.out_offset, SVM_STACK_INVALID)) {
            return false;
          }
          stack_store_float(stack, node.out_offset, 0.0f);
          webgpu_svm_mark_outputs(
              written, varying, surface_normals, node.out_offset, SVM_STACK_INVALID);
        }
        else if (node.info_type == NODE_INFO_CURVE_TANGENT_NORMAL) {
          if (!webgpu_svm_outputs_valid(SVM_STACK_INVALID, node.out_offset)) {
            return false;
          }
          stack_store_float3(stack, node.out_offset, zero_float3());
          webgpu_svm_mark_outputs(
              written, varying, surface_normals, SVM_STACK_INVALID, node.out_offset);
        }
        break;
      }
      case NODE_POINT_INFO: {
        /* svm_node_point_info() on a triangle. Random compiles to an attribute. */
        SVMNodePointInfo node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            node.info_type > NODE_INFO_POINT_RANDOM)
        {
          return false;
        }
        if (node.info_type == NODE_INFO_POINT_POSITION) {
          if (!webgpu_svm_outputs_valid(SVM_STACK_INVALID, node.out_offset)) {
            return false;
          }
          stack_store_float3(stack, node.out_offset, zero_float3());
          webgpu_svm_mark_outputs(
              written, varying, surface_normals, SVM_STACK_INVALID, node.out_offset);
        }
        else if (node.info_type == NODE_INFO_POINT_RADIUS) {
          if (!webgpu_svm_outputs_valid(node.out_offset, SVM_STACK_INVALID)) {
            return false;
          }
          stack_store_float(stack, node.out_offset, 0.0f);
          webgpu_svm_mark_outputs(
              written, varying, surface_normals, node.out_offset, SVM_STACK_INVALID);
        }
        break;
      }
      case NODE_PARTICLE_INFO: {
        /* svm_node_particle_info() reading the dummy particle Cycles stores
         * first, as it does for objects without particle instancing. */
        SVMNodeParticleInfo node;
        if (!webgpu_svm_read(fetch, offset, end, node) || !host.dummy_particle() ||
            node.info_type > NODE_INFO_PAR_ANGULAR_VELOCITY)
        {
          return false;
        }
        const KernelParticle particle{};
        const bool vector = node.info_type == NODE_INFO_PAR_LOCATION ||
                            node.info_type == NODE_INFO_PAR_VELOCITY ||
                            node.info_type == NODE_INFO_PAR_ANGULAR_VELOCITY;
        if (!webgpu_svm_outputs_valid(vector ? SVM_STACK_INVALID : uint(node.out_offset),
                                      vector ? uint(node.out_offset) : SVM_STACK_INVALID))
        {
          return false;
        }
        switch (node.info_type) {
          case NODE_INFO_PAR_INDEX:
            stack_store_float(stack, node.out_offset, float(uint(particle.index)));
            break;
          case NODE_INFO_PAR_RANDOM:
            stack_store_float(stack, node.out_offset, hash_uint2_to_float(uint(particle.index), 0));
            break;
          case NODE_INFO_PAR_AGE:
            stack_store_float(stack, node.out_offset, particle.age);
            break;
          case NODE_INFO_PAR_LIFETIME:
            stack_store_float(stack, node.out_offset, particle.lifetime);
            break;
          case NODE_INFO_PAR_LOCATION:
            stack_store_float3(stack, node.out_offset, make_float3(particle.location));
            break;
          case NODE_INFO_PAR_SIZE:
            stack_store_float(stack, node.out_offset, particle.size);
            break;
          case NODE_INFO_PAR_VELOCITY:
            stack_store_float3(stack, node.out_offset, make_float3(particle.velocity));
            break;
          case NODE_INFO_PAR_ANGULAR_VELOCITY:
            stack_store_float3(stack, node.out_offset, make_float3(particle.angular_velocity));
            break;
        }
        webgpu_svm_mark_outputs(written,
                                varying,
                                surface_normals,
                                vector ? SVM_STACK_INVALID : uint(node.out_offset),
                                vector ? uint(node.out_offset) : SVM_STACK_INVALID);
        break;
      }
      case NODE_WAVELENGTH: {
        SVMNodeWavelength node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            !webgpu_svm_input_valid(written, node.wavelength) ||
            !stack_valid(node.color_offset) ||
            !webgpu_svm_outputs_valid(SVM_STACK_INVALID, node.color_offset))
        {
          return false;
        }
        /* svm_node_wavelength() and svm_math_wavelength_color_xyz(), with
         * xyz_to_rgb() reading the packed film rows. */
        float ii = (stack_load(stack, node.wavelength) - 380.0f) * (1.0f / 5.0f);
        const int i = float_to_int(ii);
        float3 xyz = make_float3(0.0f, 0.0f, 0.0f);
        if (!(i < 0 || i >= WAVELENGTH_RESOLUTION)) {
          ii -= i;
          xyz = interp(host.cie_color_match_row(i), host.cie_color_match_row(i + 1), ii);
        }
        float3 color = make_float3(dot(color_space.xyz_to_r, xyz),
                                   dot(color_space.xyz_to_g, xyz),
                                   dot(color_space.xyz_to_b, xyz));
        color *= 1.0f / 2.52f;
        color = max(color, make_float3(0.0f, 0.0f, 0.0f));
        stack_store_float3(stack, node.color_offset, color);
        webgpu_svm_mark_outputs(
            written, varying, surface_normals, SVM_STACK_INVALID, node.color_offset);
        break;
      }
      case NODE_VECTOR_TRANSFORM: {
        SVMNodeVectorTransform node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            node.transform_type > NODE_VECTOR_TRANSFORM_TYPE_NORMAL ||
            node.convert_from > NODE_VECTOR_TRANSFORM_CONVERT_SPACE_CAMERA ||
            node.convert_to > NODE_VECTOR_TRANSFORM_CONVERT_SPACE_CAMERA ||
            !webgpu_svm_input_valid(written, node.vector_in) ||
            !webgpu_svm_outputs_valid(SVM_STACK_INVALID, node.vector_out_offset))
        {
          return false;
        }
        if (stack_valid(node.vector_out_offset)) {
          stack_store_float3(stack,
                             node.vector_out_offset,
                             webgpu_svm_vector_transform(surface, node, stack_load(stack, node.vector_in)));
          webgpu_svm_mark_outputs(
              written, varying, surface_normals, SVM_STACK_INVALID, node.vector_out_offset);
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
            written, varying, surface_normals, SVM_STACK_INVALID, node.result_offset);
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
            written, varying, surface_normals, SVM_STACK_INVALID, node.out_offset);
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
            written, varying, surface_normals, SVM_STACK_INVALID, node.out_offset);
        break;
      }
#if !defined(CYCLES_WEBGPU_SVM_CORE) || defined(CYCLES_WEBGPU_SVM_TEXTURES)
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
            written, varying, surface_normals, node.value_offset, node.color_offset);
        break;
      }
#endif
#if !defined(CYCLES_WEBGPU_SVM_CORE) || defined(CYCLES_WEBGPU_SVM_ADVANCED_QUERIES)
      case NODE_TEX_GABOR: {
        material.requires_extended_svm = true;
        material.requires_full_svm = true;
        material.requires_advanced_procedural_svm = true;
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
            written, varying, surface_normals, node.value_offset, SVM_STACK_INVALID);
        webgpu_svm_mark_outputs(
            written, varying, surface_normals, node.phase_offset, SVM_STACK_INVALID);
        webgpu_svm_mark_outputs(
            written, varying, surface_normals, node.intensity_offset, SVM_STACK_INVALID);
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
            node.position_offset == SVM_STACK_INVALID &&
            node.w_out_offset == SVM_STACK_INVALID && node.radius_offset == SVM_STACK_INVALID &&
            webgpu_svm_outputs_valid(node.distance_offset, node.color_offset) &&
            webgpu_svm_stack_has3(written, node.coord))
        {
          VoronoiParams params{};
          params.metric = NODE_VORONOI_EUCLIDEAN;
          params.randomness = saturatef(stack_load(stack, node.randomness));
          const float3 coord = make_float3(stack_load_float(stack, node.coord),
                                           stack_load_float(stack, uint(node.coord) + 1),
                                           stack_load_float(stack, uint(node.coord) + 2)) *
                               stack_load(stack, node.scale);
          const VoronoiOutput output = voronoi_f1(params, coord);
          const float bound = 0.5f + 0.5f * params.randomness;
          const float distance = node.normalize ?
                                     output.distance / voronoi_distance(
                                                           zero_float3(), make_float3(bound), params) :
                                     output.distance;
          svm_voronoi_output(stack, node, distance, output.color, zero_float3(), 0.0f, 0.0f);
          material.uses_compact_voronoi_distance |= stack_valid(node.distance_offset);
          webgpu_svm_mark_outputs(
              written, varying, surface_normals, node.distance_offset, node.color_offset);
          break;
        }
#if !defined(CYCLES_WEBGPU_SVM_CORE) || defined(CYCLES_WEBGPU_SVM_ADVANCED_QUERIES)
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
        material.requires_full_svm = true;
        material.requires_advanced_procedural_svm = true;
        svm_node_tex_voronoi<KERNEL_FEATURE_NODE_VORONOI_EXTRA>(stack, node);
        webgpu_svm_mark_outputs(
            written, varying, surface_normals, node.distance_offset, node.color_offset);
        webgpu_svm_mark_outputs(
            written, varying, surface_normals, node.w_out_offset, node.position_offset);
        webgpu_svm_mark_outputs(
            written, varying, surface_normals, node.radius_offset, SVM_STACK_INVALID);
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
            written, varying, surface_normals, node.fac_offset, node.color_offset);
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
            written, varying, surface_normals, node.fac_offset, node.color_offset);
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
            written, varying, surface_normals, node.fac_offset, node.color_offset);
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
            written, varying, surface_normals, node.fac_offset, node.color_offset);
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
            written, varying, surface_normals, node.fac_offset, node.color_offset);
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
            written, varying, surface_normals, node.value_offset, node.color_offset);
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
            written, varying, surface_normals, node.result_offset, SVM_STACK_INVALID);
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
            written, varying, surface_normals, node.value_offset, node.vector_offset);
        break;
      }
#if !defined(CYCLES_WEBGPU_SVM_CORE) || defined(CYCLES_WEBGPU_SVM_TEXTURES)
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
            written, varying, surface_normals, node.result_offset, SVM_STACK_INVALID);
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
            written, varying, surface_normals, SVM_STACK_INVALID, node.out_offset);
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
            written, varying, surface_normals, SVM_STACK_INVALID, node.out_offset);
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
            written, varying, surface_normals, SVM_STACK_INVALID, node.out_color_offset);
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
            written, varying, surface_normals, SVM_STACK_INVALID, node.out_offset);
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
            written, varying, surface_normals, SVM_STACK_INVALID, node.result_offset);
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
            written, varying, surface_normals, SVM_STACK_INVALID, node.result_offset);
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
            written, varying, surface_normals, node.result_offset, SVM_STACK_INVALID);
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
            written, varying, surface_normals, SVM_STACK_INVALID, node.result_offset);
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
            written, varying, surface_normals, SVM_STACK_INVALID, node.result_offset);
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
            written, varying, surface_normals, SVM_STACK_INVALID, node.color_offset);
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
            written, varying, surface_normals, node.out_offset, SVM_STACK_INVALID);
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
            written, varying, surface_normals, node.result_offset, SVM_STACK_INVALID);
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
            written, varying, surface_normals, SVM_STACK_INVALID, node.result_offset);
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
            written, varying, surface_normals, SVM_STACK_INVALID, node.result_offset);
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
            written, varying, surface_normals, node.out_dot_offset, node.out_normal_offset);
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
          webgpu_svm_mark_outputs(written, varying, surface_normals, channel, SVM_STACK_INVALID);
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
              written, varying, surface_normals, component, SVM_STACK_INVALID);
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
              written, varying, surface_normals, SVM_STACK_INVALID, node.to_offset);
        }
        else {
          if ((node.convert_type != NODE_CONVERT_CF && node.convert_type != NODE_CONVERT_VF) ||
              node.to_offset >= SVM_STACK_SIZE || !webgpu_svm_stack_has3(written, node.from_offset))
          {
            return false;
          }
          const float3 value = stack_load_float3(stack, node.from_offset);
          /* The working-space coefficients are serialized from kernel data. */
          const float scalar = node.convert_type == NODE_CONVERT_CF ? dot(value, color_space.rgb_to_y) :
                                                                     average(value);
          stack_store_float(stack, node.to_offset, scalar);
          webgpu_svm_mark_outputs(
              written, varying, surface_normals, node.to_offset, SVM_STACK_INVALID);
        }
        break;
      }
#if !defined(CYCLES_WEBGPU_SVM_CORE) || defined(CYCLES_WEBGPU_SVM_TEXTURES)
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
            written, varying, surface_normals, node.alpha_offset, node.color_offset);
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
            written, varying, surface_normals, SVM_STACK_INVALID, node.out_offset);
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
            written, varying, surface_normals, node.out_offset, SVM_STACK_INVALID);
        break;
      }
#endif
      case NODE_TEX_IMAGE: {
        SVMNodeTexImage node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            (node.projection != NODE_IMAGE_PROJ_FLAT
#if !defined(CYCLES_WEBGPU_SVM_CORE) || defined(CYCLES_WEBGPU_SVM_PROJECTED_IMAGES)
             && node.projection != NODE_IMAGE_PROJ_SPHERE && node.projection != NODE_IMAGE_PROJ_TUBE
#endif
             ) ||
            !webgpu_svm_eval_image(
                image, node, surface, stack, written, varying, surface_normals))
        {
          return false;
        }
        material.requires_extended_svm |= node.projection != NODE_IMAGE_PROJ_FLAT;
        material.requires_full_svm |= node.projection != NODE_IMAGE_PROJ_FLAT;
        break;
      }
#if !defined(CYCLES_WEBGPU_SVM_CORE) || defined(CYCLES_WEBGPU_SVM_PROJECTED_IMAGES)
      case NODE_TEX_IMAGE_BOX: {
        material.requires_full_svm = true;
        SVMNodeTexImageBox node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            !webgpu_svm_eval_image(
                image, node, surface, stack, written, varying, surface_normals))
        {
          return false;
        }
        material.requires_extended_svm = true;
        break;
      }
      case NODE_TEX_ENVIRONMENT: {
        material.requires_full_svm = true;
        SVMNodeTexEnvironment node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            (node.projection != NODE_ENVIRONMENT_EQUIRECTANGULAR &&
             node.projection != NODE_ENVIRONMENT_MIRROR_BALL) ||
            !webgpu_svm_eval_image(
                image, node, surface, stack, written, varying, surface_normals))
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
        weight_varying = false;
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
        weight_varying = webgpu_svm_stack_has(varying, node.weight_offset) ||
                         webgpu_svm_stack_has(varying, uint(node.weight_offset) + 1) ||
                         webgpu_svm_stack_has(varying, uint(node.weight_offset) + 2);
        break;
      }
      case NODE_EMISSION_WEIGHT: {
        /* svm_node_emission_weight() */
        SVMNodeEmissionWeight node;
        if (!webgpu_svm_read(fetch, offset, end, node) ||
            !webgpu_svm_inputs_valid(written, node.color, node.strength))
        {
          return false;
        }
        weight = rgb_to_spectrum(stack_load(stack, node.color)) * stack_load(stack, node.strength);
        weight_varying = true;
        break;
      }
      case NODE_CLOSURE_BACKGROUND:
      case NODE_CLOSURE_EMISSION: {
        /* Both stock closures add the mixed weight to closure_emission_background. */
        static_assert(sizeof(SVMNodeClosureEmission) == sizeof(SVMNodeClosureBackground));
        static_assert(offsetof(SVMNodeClosureEmission, mix_weight_offset) ==
                      offsetof(SVMNodeClosureBackground, mix_weight_offset));
        SVMNodeClosureEmission node;
        if (ordered_closures || material.holdout || !webgpu_svm_read(fetch, offset, end, node) ||
            (stack_valid(node.mix_weight_offset) &&
             !webgpu_svm_stack_has(written, node.mix_weight_offset)))
        {
          return false;
        }
        /* A zero mix weight adds nothing, but the surface still has a valid
         * (black) shader wherever the weight varies. */
        has_emission = true;
        Spectrum emission = weight;
        if (stack_valid(node.mix_weight_offset)) {
          const float mix_weight = stack_load_float(stack, node.mix_weight_offset);
          if (mix_weight == 0.0f) {
            break;
          }
          emission *= mix_weight;
        }
        material.emission += emission;
        if (!webgpu_svm_finite(material.emission.x) || !webgpu_svm_finite(material.emission.y) ||
            !webgpu_svm_finite(material.emission.z))
        {
          return false;
        }
        break;
      }
      case NODE_CLOSURE_HOLDOUT: {
        SVMNodeClosureHoldout node;
        if (ordered_closures || has_closure || mixes_closures || has_emission ||
            !webgpu_svm_read(fetch, offset, end, node) ||
            (stack_valid(node.mix_weight_offset) &&
             !webgpu_svm_stack_has(written, node.mix_weight_offset)))
        {
          return false;
        }
        /* svm_node_closure_holdout() drops a zero mix weight. */
        const float mix_weight = stack_load_float_default(stack, node.mix_weight_offset, 1.0f);
        material.color = mix_weight == 0.0f ? zero_float3() : weight * mix_weight;
        if (!webgpu_svm_finite(material.color.x) || !webgpu_svm_finite(material.color.y) ||
            !webgpu_svm_finite(material.color.z))
        {
          return false;
        }
        material.ggx = false;
        material.transmission = 0.0f;
        material.standalone_transmission = 0.0f;
        material.subsurface = 0.0f;
        material.holdout = true;
        has_closure = true;
        break;
      }
      case NODE_MIX_CLOSURE: {
        SVMNodeMixClosure node;
        if ((has_closure && lambert_closures == 0 && !has_transparent && !has_principled &&
             !has_translucent && !has_glossy) ||
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
            written, varying, surface_normals, node.weight1_offset, SVM_STACK_INVALID);
        webgpu_svm_mark_outputs(
            written, varying, surface_normals, node.weight2_offset, SVM_STACK_INVALID);
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
        if ((has_closure && lambert_closures == 0 && !has_transparent && !has_principled &&
             !has_translucent && !has_glossy) ||
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
#ifdef CYCLES_WEBGPU_SAMPLED_EMISSION
        /* Stock SURFACE_LIGHT evaluates only the Principled emission among
         * BSDF nodes. Advance the others using the stock bytecode layout. */
        if (closure.closure_type != CLOSURE_BSDF_PRINCIPLED_ID) {
          offset = uint(svm_node_closure_bsdf_skip(int(offset), closure.closure_type));
          if (offset > end) {
            return false;
          }
          break;
        }

        SVMNodePrincipledBsdfData emission_data;
        if (!webgpu_svm_read(fetch, offset, end, emission_data) ||
            !webgpu_svm_inputs_valid(written, emission_data.alpha, emission_data.emission_color,
                                     emission_data.emission_strength) ||
            (stack_valid(closure.mix_weight_offset) &&
             !webgpu_svm_stack_has(written, closure.mix_weight_offset)))
        {
          return false;
        }
        const float mix = stack_load_float_default(stack, closure.mix_weight_offset, 1.0f);
        const float emission_alpha = saturatef(stack_load(stack, emission_data.alpha));
        const Spectrum emission = stack_load(stack, emission_data.emission_color) *
                                  stack_load(stack, emission_data.emission_strength);
        if (!is_zero(emission) && mix != 0.0f && emission_alpha != 0.0f) {
          WebGPUSvmCoat coat{};
          if (!webgpu_svm_read_white_coat(
                  emission_data, written, stack, surface_normals.base, coat))
          {
            return false;
          }
          if (coat.possible) {
            /* Main-path eligibility admits opaque Coat with a single emissive
             * Principled component and the unmodified surface normal. */
            material.coat = coat;
            material.coat_emission = emission;
            material.mix_weight = mix * emission_alpha;
          }
          else {
            material.emission += emission * (mix * emission_alpha);
          }
        }
        if (!webgpu_svm_finite(emission.x) || !webgpu_svm_finite(emission.y) ||
            !webgpu_svm_finite(emission.z) ||
            !webgpu_svm_finite(material.emission.x) || !webgpu_svm_finite(material.emission.y) ||
            !webgpu_svm_finite(material.emission.z))
        {
          return false;
        }
        break;
#endif
        if (ordered_closures) {
          const ClosureType closure_type = closure.closure_type;
          if (closure_type == CLOSURE_BSDF_PRINCIPLED_ID) {
            if ((ordered_principled_branches == 0u && ordered_candidates != 0u) ||
                ordered_principled_branches >= 2u ||
                !webgpu_svm_capture_principled_branch(
                    fetch, offset, end, closure, written, surface_normals, stack,
                    webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_NORMAL)),
                    surface(CYCLES_WEBGPU_SVM_PATH), host, ordered_principled_branches, material,
                    ordered_candidates))
            {
              return false;
            }
            ++ordered_principled_branches;
            break;
          }
          if (ordered_principled_branches != 0u) {
            return false;
          }
          if (closure_type == CLOSURE_BSDF_TRANSPARENT_ID) {
            /* Stock Cycles merges repeated Transparent closures into the first
             * allocation. Admit one until the ordered list can replace a lobe. */
            if (material.ordered_lobe_types & CYCLES_WEBGPU_ORDERED_HAS_TRANSPARENT) {
              return false;
            }
            material.ordered_lobe_types |= CYCLES_WEBGPU_ORDERED_HAS_TRANSPARENT;
            material.varying_transparency = weight_varying ||
                (stack_valid(closure.mix_weight_offset) &&
                 webgpu_svm_stack_has(varying, closure.mix_weight_offset));
          }
          if (closure_type == CLOSURE_BSDF_TRANSLUCENT_ID) {
            material.ordered_lobe_types |= CYCLES_WEBGPU_ORDERED_HAS_TRANSLUCENT;
          }
          ordered_has_glossy |= closure_type == CLOSURE_BSDF_MICROFACET_GGX_ID ||
                                closure_type == CLOSURE_BSDF_MICROFACET_BECKMANN_ID;
          if (++ordered_candidates > CYCLES_WEBGPU_MAX_ORDERED_LOBES ||
              !webgpu_svm_capture_ordered_lobe(fetch, offset, end, closure, written, varying,
                                               surface_normals, stack, weight, host, material,
                                               ordered_count))
          {
            return false;
          }
          break;
        }
        const bool diffuse = closure.closure_type == CLOSURE_BSDF_DIFFUSE_ID;
        const bool principled_node = closure.closure_type == CLOSURE_BSDF_PRINCIPLED_ID;
        SVMNodePrincipledBsdfData principled_data;
        if (principled_node && !webgpu_svm_read(fetch, offset, end, principled_data)) {
          return false;
        }
        /* principled_bsdf_emission() installs a Transparent closure before
         * multiplying every remaining lobe and emission by Alpha. */
        float alpha = 1.0f;
        bool linked_alpha = false;
        if (principled_node) {
          if (!webgpu_svm_input_valid(written, principled_data.alpha)) {
            return false;
          }
          const float value = stack_load(stack, principled_data.alpha);
          if (!webgpu_svm_finite(value)) {
            return false;
          }
          alpha = saturatef(value);
          linked_alpha = webgpu_svm_input_on_stack(principled_data.alpha);
        }
        const bool zero_alpha = principled_node && !linked_alpha && alpha == 0.0f;
        const bool explicit_transparent = closure.closure_type == CLOSURE_BSDF_TRANSPARENT_ID;
        const bool transparent = explicit_transparent || zero_alpha;
        const bool alpha_transparent = principled_node && (linked_alpha || alpha < 1.0f);
        const bool diffuse_principled = principled_node &&
            webgpu_svm_surface_normal(principled_data.normal_offset, surface_normals.base) &&
            webgpu_svm_is_diffuse_principled(principled_data);
        const bool lambert = diffuse || diffuse_principled;
        const bool principled = principled_node && !zero_alpha && !diffuse_principled;
        const bool translucent = closure.closure_type == CLOSURE_BSDF_TRANSLUCENT_ID;
        const bool glossy = closure.closure_type == CLOSURE_BSDF_MICROFACET_BECKMANN_ID ||
                            closure.closure_type == CLOSURE_BSDF_MICROFACET_GGX_ID;
        const bool glossy_follows_transparent = glossy && single_explicit_transparent &&
                                                lambert_closures == 0;
        const bool transparent_follows_glossy = explicit_transparent && has_glossy &&
                                                !has_transparent && lambert_closures == 0;
        const bool principled_translucent_pair = lambert_closures == 0 && !has_glossy &&
            ((translucent && has_principled && !has_translucent) ||
             (principled && has_translucent && !has_principled));
        if ((has_closure && lambert_closures == 0 && !has_transparent && !has_principled &&
             !has_translucent && !has_glossy) ||
            ((has_closure || mixes_closures) && !lambert && !transparent && !principled &&
             !translucent && !glossy) ||
            (!principled_translucent_pair && (translucent || glossy) &&
             (has_translucent || has_glossy || (has_transparent && !glossy_follows_transparent) ||
              has_principled || has_diffuse_principled || lambert_closures > 1)) ||
            (!principled_translucent_pair && (has_translucent || has_glossy) &&
             !transparent_follows_glossy &&
             (!diffuse || lambert_closures != 0 || (has_glossy && has_transparent))) ||
            lambert_closures == MAX_CLOSURE || (transparent && lambert_closures > 1) ||
            (lambert && has_transparent && lambert_closures != 0) ||
            (principled && (has_principled || lambert_closures != 0)) ||
            (lambert && has_principled) ||
            (closure.mix_weight_offset != SVM_STACK_INVALID &&
             (!webgpu_svm_stack_has(written, closure.mix_weight_offset) ||
              (!lambert && !transparent && !principled && !translucent && !glossy &&
               webgpu_svm_stack_has(varying, closure.mix_weight_offset)))))
        {
          return false;
        }
        const float mix_weight = stack_load_float_default(stack, closure.mix_weight_offset, 1.0f);
        if (transparent || alpha_transparent) {
          SVMNodeSimpleBsdfData data;
          const Spectrum tint = alpha_transparent ? make_spectrum(mix_weight * (1.0f - alpha)) :
                                                     weight * mix_weight;
          const bool varying_tint = linked_alpha || (explicit_transparent && weight_varying) ||
              (stack_valid(closure.mix_weight_offset) &&
               webgpu_svm_stack_has(varying, closure.mix_weight_offset));
          if ((explicit_transparent && !webgpu_svm_read(fetch, offset, end, data)) ||
              !webgpu_svm_finite(tint.x) || !webgpu_svm_finite(tint.y) ||
              !webgpu_svm_finite(tint.z) ||
              (!varying_tint && (reduce_min(tint) < 0.0f || reduce_max(tint) > 1.0f)))
          {
            return false;
          }
          ShaderData sd{};
          sd.num_closure_left = 1;
          bsdf_transparent_setup(&sd, tint, 0);
          if (sd.num_closure != 0) {
            if (material.transparent_sample_weight == 0.0f) {
              material.transparent_first = uint(material.diffuse_sample_weight == 0.0f &&
                                                !has_principled && !has_glossy);
            }
            material.transparency += sd.closure_transparent_extinction;
            material.transparent_sample_weight += sd.closure[0].sample_weight;
          }
          material.varying_transparency |= varying_tint;
          material.principled_alpha |= alpha_transparent && !zero_alpha;
          single_explicit_transparent = explicit_transparent && !has_transparent;
          has_explicit_transparent |= explicit_transparent;
          has_transparent = true;
          if (transparent) {
            has_closure = true;
            break;
          }
        }
        if (translucent) {
          SVMNodeSimpleBsdfData data;
          Spectrum color = weight * mix_weight;
          if (!webgpu_svm_read(fetch, offset, end, data) ||
              !webgpu_svm_surface_normal(data.normal_offset, surface_normals.active) ||
              surface_normals.find_bump(data.normal_offset) != 0u ||
              !webgpu_svm_finite(color.x) || !webgpu_svm_finite(color.y) ||
              !webgpu_svm_finite(color.z))
          {
            return false;
          }
          material.translucent_sample_weight = closure_sample_weight(0, color);
          material.translucent_color = material.translucent_sample_weight > 0.0f ?
                                           color : zero_spectrum();
          material.translucent_first = lambert_closures == 0 && !has_principled;
          has_translucent = has_closure = true;
          break;
        }
        if (glossy) {
          SVMNodeGlossyBsdfData data;
          Spectrum color = weight * mix_weight;
          material.surface_fresnel_a = zero_float3();
          material.surface_fresnel_b = zero_float3();
          if (!webgpu_svm_read(fetch, offset, end, data)) {
            return false;
          }
          const uint bump_word = surface_normals.find_bump(data.normal_offset);
          if ((bump_word == 0u &&
               !webgpu_svm_surface_normal(data.normal_offset, surface_normals.base)) ||
              !webgpu_svm_finite(color.x) || !webgpu_svm_finite(color.y) ||
              !webgpu_svm_finite(color.z) ||
              !webgpu_svm_glossy_geometry(data.roughness, data.anisotropy, data.rotation,
                                         data.tangent_offset, false, written, stack,
                                         zero_float3(), material))
          {
            return false;
          }
          webgpu_svm_save_paired_normal(material, stack, data.normal_offset, bump_word,
                                        closure_bump_word, false, lambert_closures != 0);
          material.glossy_sample_weight = closure_sample_weight(0, color);
          material.glossy_color = material.glossy_sample_weight > 0.0f ? color : zero_spectrum();
          if (lambert_closures == 0) {
            material.mixed_glossy_flags |= CYCLES_WEBGPU_MIXED_GLOSSY_FIRST;
          }
          material.glossy_isotropic = !stack_valid(data.tangent_offset) ||
              (material.surface_parameters.x == material.surface_parameters.y &&
               !webgpu_svm_input_on_stack(data.anisotropy));
          glossy_nonsingular = webgpu_svm_rough_ggx(material.surface_parameters.x) &&
              (!webgpu_svm_input_on_stack(data.roughness) ||
               !webgpu_svm_stack_has(varying, data.roughness.bits & 0xffu));
          material.surface_model = uint(closure.closure_type);
          has_glossy = has_closure = true;
          break;
        }
        if (lambert) {
          uint normal_offset;
          Spectrum closure_weight;
          if (diffuse_principled) {
            const SVMNodePrincipledBsdfData &data = principled_data;
            normal_offset = data.normal_offset;
            Spectrum color;
            if (!webgpu_svm_surface_color(data.base_color, written, stack, color)) {
              return false;
            }
            closure_weight = max(color, zero_spectrum()) * mix_weight;
            has_diffuse_principled = true;
          }
          else {
            SVMNodeDiffuseBsdfData data;
            if (!webgpu_svm_read(fetch, offset, end, data)) {
              return false;
            }
            if ((surface_normals.find_bump(data.normal_offset) == 0u &&
                 !webgpu_svm_surface_normal(data.normal_offset, surface_normals.base)) ||
                (webgpu_svm_input_on_stack(data.roughness) &&
                 (!webgpu_svm_stack_has(written, data.roughness.bits & 0xffu) ||
                  webgpu_svm_stack_has(varying, data.roughness.bits & 0xffu))))
            {
              return false;
            }
            const float roughness = stack_load(stack, data.roughness);
            if (!webgpu_svm_finite(roughness) || !diffuse_roughness_is_almost_zero(roughness)) {
              return false;
            }
            normal_offset = data.normal_offset;
            closure_weight = weight * mix_weight;
          }
          const uint bump_word = surface_normals.find_bump(normal_offset);
          if (lambert_closures != 0 && (bump_word != 0u || closure_bump_word != 0u))
          {
            return false;
          }
          webgpu_svm_save_paired_normal(material, stack, normal_offset, bump_word,
                                        closure_bump_word, true, has_glossy);
          if (!webgpu_svm_add_diffuse_lobe(material, closure_weight, lambert_closures)) {
            return false;
          }
          lambert_closures++;
          has_closure = true;
          break;
        }
#ifdef CYCLES_WEBGPU_SHADOW_SVM
        if (principled) {
          /* Host eligibility already validated the opaque Principled lobes.
           * Stock shadow evaluation only uses the Transparent extinction. */
          has_principled = has_closure = mixable_principled = true;
          translucent_mixable_principled = true;
          break;
        }
        if (closure.closure_type == CLOSURE_BSDF_HAIR_HUANG_ID) {
          SVMNodePrincipledHairBsdfData hair;
          if (!webgpu_svm_read(fetch, offset, end, hair) ||
              !webgpu_svm_inputs_valid(written, hair.aspect_ratio, hair.R, hair.TT, hair.TRT))
          {
            return false;
          }
          material.color = weight * mix_weight;
          if (!(stack_load(stack, hair.R) > 0.0f || stack_load(stack, hair.TT) > 0.0f ||
                stack_load(stack, hair.TRT) > 0.0f))
          {
            material.color = zero_spectrum();
          }
          material.surface_hair_aspect = stack_load(stack, hair.aspect_ratio);
          material.surface_tangent = webgpu_svm_context_xyz(
              surface(CYCLES_WEBGPU_SVM_TRIANGLE_DPDU));
          material.surface_closure = CYCLES_WEBGPU_MATERIAL_HAIR_HUANG;
          has_closure = true;
          break;
        }
        return false;
#else
        Spectrum closure_weight;
        if (principled) {
          const SVMNodePrincipledBsdfData &data = principled_data;
          const bool linked_normal = surface_normals.find_bump(data.normal_offset) != 0u;
          const uint normal_slot = linked_normal ? uint(data.normal_offset) : surface_normals.active;
          if (!webgpu_svm_read_white_coat(data, written, stack, surface_normals.base, material.coat)) {
            return false;
          }
          const uint emission_color_bits = data.emission_color.x.bits |
                                           data.emission_color.y.bits |
                                           data.emission_color.z.bits;
          const bool emits = (data.emission_strength.bits & 0x7fffffffu) != 0 &&
                             (emission_color_bits & 0x7fffffffu) != 0;
          if (material.coat.possible) {
            const uint white = __float_as_uint(1.0f);
            if (alpha_transparent || material.displacement_bump || (emits && linked_normal) ||
                ((data.metallic.bits | data.transmission_weight.bits) & 0x7fffffffu) != 0u ||
                (stack_valid(data.tangent_offset) && (data.anisotropic.bits & 0x7fffffffu) != 0u) ||
                data.specular_tint.x.bits != white || data.specular_tint.y.bits != white ||
                data.specular_tint.z.bits != white)
            {
              return false;
            }
          }
          material.coated_principled_emission = material.coat.possible && emits;
          if (!webgpu_svm_principled(data,
                                    written,
                                    stack,
                                    normal_slot,
                                    mix_weight * alpha,
                                    webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_NORMAL)),
                                    material,
                                    material.coat.possible))
          {
            return false;
          }
          if (linked_normal) {
            material.closure_normal = stack_load_float3(stack, data.normal_offset);
            material.closure_normal_overridden = true;
          }
          has_principled = true;
          /* Subsurface and anisotropic Principled occupy the auxiliary record
           * used by Transparent extinction. Isotropic GGX transmission does not. */
          material.single_principled_specular_candidate = material.coat.possible &&
              (material.eta != 1.0f || webgpu_svm_input_on_stack(data.specular_ior_level));
          mixable_principled = material.subsurface == 0.0f && !material.coat.possible && !emits &&
                               !(stack_valid(data.tangent_offset) &&
                                 (webgpu_svm_input_on_stack(data.anisotropic) ||
                                  __uint_as_float(data.anisotropic.bits) > 0.0f));
          translucent_mixable_principled = mixable_principled && material.ggx &&
              material.transmission == 0.0f && (data.metallic.bits & 0x7fffffffu) == 0u;
          if (material.ggx) {
            has_closure = true;
            break;
          }
          closure_weight = material.color;
        }
        else if (material.surface_bump && !material.displacement_bump &&
                 !CLOSURE_IS_GLASS(closure.closure_type) &&
                 closure.closure_type != CLOSURE_BSDF_RAY_PORTAL_ID) {
          return false;
        }
        else if (closure.closure_type == CLOSURE_BSDF_RAY_PORTAL_ID) {
          SVMNodeRayPortalBsdfData portal;
          if (!webgpu_svm_read(fetch, offset, end, portal) ||
              !webgpu_svm_inputs_valid(written, portal.direction) ||
              (stack_valid(portal.position_offset) &&
               !webgpu_svm_stack_has3(written, portal.position_offset)))
          {
            return false;
          }
          ShaderData sd{};
          sd.num_closure_left = 1;
          sd.P = webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_POSITION));
          sd.N = webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_NORMAL));
          sd.wi = -webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_DIRECTION));
          const float3 P = stack_load_float3_default(stack, portal.position_offset, sd.P);
          bsdf_ray_portal_setup(&sd, weight * mix_weight, P, stack_load(stack, portal.direction));
          material.color = sd.closure_transparent_extinction;
          material.surface_closure = CYCLES_WEBGPU_MATERIAL_RAY_PORTAL;
          material.surface_fresnel_a = P;
          material.surface_fresnel_b = sd.num_closure > 0 ?
                                           reinterpret_cast<const RayPortalClosure *>(&sd.closure[0])
                                               ->D :
                                           zero_float3();
          has_closure = true;
          break;
        }
        else if (closure.closure_type == CLOSURE_BSDF_DIFFUSE_TOON_ID ||
                 closure.closure_type == CLOSURE_BSDF_GLOSSY_TOON_ID)
        {
          SVMNodeToonBsdfData toon;
          if (!webgpu_svm_read(fetch, offset, end, toon) ||
              !webgpu_svm_surface_normal(toon.normal_offset, surface_normals.active) ||
              !webgpu_svm_inputs_valid(written, toon.size, toon.smooth))
          {
            return false;
          }
          const float size = stack_load(stack, toon.size);
          const float smooth = stack_load(stack, toon.smooth);
          if (!webgpu_svm_finite(size) || !webgpu_svm_finite(smooth)) {
            return false;
          }
          material.surface_closure = closure.closure_type == CLOSURE_BSDF_DIFFUSE_TOON_ID ?
                                         CYCLES_WEBGPU_MATERIAL_DIFFUSE_TOON :
                                         CYCLES_WEBGPU_MATERIAL_GLOSSY_TOON;
          material.surface_parameters = make_float2(size, smooth);
          closure_weight = weight * mix_weight;
        }
        else if (closure.closure_type == CLOSURE_BSDF_ASHIKHMIN_VELVET_ID ||
                 closure.closure_type == CLOSURE_BSDF_SHEEN_ID)
        {
          SVMNodeSimpleBsdfData sheen;
          if (!webgpu_svm_read(fetch, offset, end, sheen) ||
              !webgpu_svm_surface_normal(sheen.normal_offset, surface_normals.active) ||
              !webgpu_svm_inputs_valid(written, sheen.param1))
          {
            return false;
          }
          const float roughness = stack_load(stack, sheen.param1);
          if (!webgpu_svm_finite(roughness)) {
            return false;
          }
          material.surface_closure = closure.closure_type == CLOSURE_BSDF_SHEEN_ID ?
                                         CYCLES_WEBGPU_MATERIAL_SHEEN :
                                         CYCLES_WEBGPU_MATERIAL_VELVET;
          material.surface_parameters = make_float2(saturatef(roughness), 0.0f);
          closure_weight = weight * mix_weight;
        }
        else if (closure.closure_type == CLOSURE_BSDF_MICROFACET_MULTI_GGX_ID ||
                 closure.closure_type == CLOSURE_BSDF_ASHIKHMIN_SHIRLEY_ID)
        {
          SVMNodeGlossyBsdfData glossy;
          material.surface_fresnel_a = zero_float3();
          material.surface_fresnel_b = zero_float3();
          if (!webgpu_svm_read(fetch, offset, end, glossy) ||
              !webgpu_svm_surface_normal(glossy.normal_offset, surface_normals.active) ||
              !webgpu_svm_glossy_geometry(glossy.roughness, glossy.anisotropy, glossy.rotation,
                                         glossy.tangent_offset, false, written, stack,
                                         zero_float3(), material))
          {
            return false;
          }
          if (closure.closure_type == CLOSURE_BSDF_MICROFACET_MULTI_GGX_ID) {
            if (!webgpu_svm_surface_color(glossy.color, written, stack, material.surface_fresnel_a)) {
              return false;
            }
            material.surface_fresnel_a = max(material.surface_fresnel_a, zero_float3());
          }
          material.surface_closure = CYCLES_WEBGPU_MATERIAL_GLOSSY;
          material.surface_model = uint(closure.closure_type);
          closure_weight = weight * mix_weight;
        }
        else if (closure.closure_type == CLOSURE_BSDF_PHYSICAL_CONDUCTOR ||
                 closure.closure_type == CLOSURE_BSDF_F82_CONDUCTOR)
        {
          SVMNodeMetallicBsdfData metallic;
          if (!webgpu_svm_read(fetch, offset, end, metallic) ||
              !webgpu_svm_surface_normal(metallic.normal_offset, surface_normals.active) ||
              !webgpu_svm_inputs_valid(written, metallic.thin_film_thickness) ||
              (webgpu_svm_input_on_stack(metallic.thin_film_thickness) &&
               webgpu_svm_stack_has(varying, metallic.thin_film_thickness.bits & 0xffu)) ||
              !webgpu_svm_surface_color(metallic.base_ior, written, stack, material.surface_fresnel_a) ||
              !webgpu_svm_surface_color(metallic.edge_tint_k, written, stack, material.surface_fresnel_b))
          {
            return false;
          }
          const float film_thickness = stack_load(stack, metallic.thin_film_thickness);
          if (!webgpu_svm_finite(film_thickness) || film_thickness > THINFILM_THICKNESS_CUTOFF ||
              (metallic.distribution != CLOSURE_BSDF_MICROFACET_GGX_ID &&
               metallic.distribution != CLOSURE_BSDF_MICROFACET_BECKMANN_ID &&
               metallic.distribution != CLOSURE_BSDF_MICROFACET_MULTI_GGX_ID) ||
              !webgpu_svm_glossy_geometry(metallic.roughness, metallic.anisotropy, metallic.rotation,
                                         metallic.tangent_offset, true, written, stack,
                                         webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_NORMAL)),
                                         material))
          {
            return false;
          }
          material.surface_closure = CYCLES_WEBGPU_MATERIAL_GLOSSY;
          material.surface_model = uint(metallic.distribution) | (uint(closure.closure_type) << 8u);
          closure_weight = make_spectrum(mix_weight);
        }
        else if (closure.closure_type == CLOSURE_BSDF_HAIR_REFLECTION_ID ||
                 closure.closure_type == CLOSURE_BSDF_HAIR_TRANSMISSION_ID)
        {
          SVMNodeHairBsdfData hair;
          if (!webgpu_svm_read(fetch, offset, end, hair) ||
              !webgpu_svm_inputs_valid(written, hair.roughness1, hair.roughness2, hair.offset) ||
              (stack_valid(hair.tangent_offset) &&
               !webgpu_svm_stack_has3(written, hair.tangent_offset)))
          {
            return false;
          }
          const bool custom_tangent = stack_valid(hair.tangent_offset);
          material.surface_tangent = safe_normalize(stack_load_float3_default(
              stack, hair.tangent_offset,
              webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_TRIANGLE_DPDV))));
          material.surface_parameters = make_float2(
              custom_tangent ? -stack_load(stack, hair.offset) : 0.0f,
              stack_load(stack, hair.roughness1));
          material.surface_fresnel_b.x = stack_load(stack, hair.roughness2);
          material.surface_closure = closure.closure_type == CLOSURE_BSDF_HAIR_REFLECTION_ID ?
                                         CYCLES_WEBGPU_MATERIAL_HAIR_REFLECTION :
                                         CYCLES_WEBGPU_MATERIAL_HAIR_TRANSMISSION;
          closure_weight = weight * mix_weight;
        }
        else if (closure.closure_type == CLOSURE_BSDF_HAIR_CHIANG_ID ||
                 closure.closure_type == CLOSURE_BSDF_HAIR_HUANG_ID)
        {
          SVMNodePrincipledHairBsdfData hair;
          const bool huang = closure.closure_type == CLOSURE_BSDF_HAIR_HUANG_ID;
          if (!webgpu_svm_read(fetch, offset, end, hair) ||
              !webgpu_svm_principled_hair(hair, huang, written, stack,
                  webgpu_svm_context_xyz(surface(CYCLES_WEBGPU_SVM_TRIANGLE_DPDU)), material))
          {
            return false;
          }
          material.surface_closure = huang ? CYCLES_WEBGPU_MATERIAL_HAIR_HUANG :
                                            CYCLES_WEBGPU_MATERIAL_HAIR_CHIANG;
          closure_weight = weight * mix_weight;
          if (huang && is_zero(material.surface_fresnel_b)) {
            closure_weight = zero_spectrum();
          }
        }
        else if (CLOSURE_IS_BSSRDF(closure.closure_type)) {
          SVMNodeBssrdfData data;
          if (!webgpu_svm_read(fetch, offset, end, data) ||
              !webgpu_svm_surface_normal(data.normal_offset, surface_normals.active) ||
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
                  refraction, written, stack, surface_normals.active, weight, mix_weight,
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
                                surface_normals.active,
                                mix_weight,
                                ClosureType(closure.closure_type),
                                material))
          {
            return false;
          }
          if (surface_normals.find_bump(glass.normal_offset) != 0u) {
            material.closure_normal = stack_load_float3(stack, glass.normal_offset);
            material.closure_normal_overridden = true;
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
        const float sample_weight = closure_sample_weight(0, closure_weight);
        material.color = sample_weight > 0.0f ? closure_weight : zero_spectrum();
        if (principled) {
          material.diffuse_sample_weight = sample_weight;
        }
        if (!webgpu_svm_finite(average(material.color))) {
          return false;
        }
        has_closure = true;
        break;
#endif
      }
      default:
        return false;
    }
  }
  return false;
}

CCL_NAMESPACE_END
