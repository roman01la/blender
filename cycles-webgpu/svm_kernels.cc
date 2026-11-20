/* SPDX-License-Identifier: Apache-2.0 */

#include "../blender-platform/cycles-webgpu/material_layout.h"

#ifdef __aarch64__
#  include <arm_neon.h>
#endif

#define CCL_NAMESPACE_BEGIN namespace ccl {
#define CCL_NAMESPACE_END }
#include "kernel/types.h"

#define __KERNEL_GPU__
#include "kernel/globals.h"
#include "util/types_image.h"
#undef __KERNEL_GPU__

__attribute__((pure)) unsigned int cycles_svm_word(unsigned int index);

namespace ccl {
/* kernel_data_fetch(ies, i) reads the IES table packed after the SVM records. */
struct SvmIesWords {
  unsigned int offset;
  float operator[](const int index) const
  {
    return __uint_as_float(cycles_svm_word(offset + unsigned(index)));
  }
};
struct SvmGlobals {
  KernelData data;
  const uint *svm_nodes;
  const float *lookup_table;
  SvmIesWords ies;
};
using KernelGlobals = const SvmGlobals *;
}  // namespace ccl

#define kernel_data (kg->data)
#define kernel_data_fetch(name, index) (kg->name[index])
#include "../blender-platform/cycles-webgpu/svm_material_eval.h"
#include "kernel/util/ies.h"

using Float4 = float __attribute__((ext_vector_type(4)));

__attribute__((pure)) ccl::WebGPUSvmContextWords cycles_svm_context(unsigned int lane,
                                                                  unsigned int field);

namespace {
/* The host already resolved attributes and particles while packing. */
struct PackedWordsHost : ccl::WebGPUSvmPackedHost {
  unsigned int texture;
  unsigned int primitive;
  float u;
  float v;
  bool attribute_sample(const unsigned int slot, ccl::WebGPUSvmAttributeSample &sample) const
  {
    sample = ccl::webgpu_svm_triangle_attribute(cycles_svm_word, texture, slot, primitive, u, v);
    return true;
  }
  bool triangle_corners(const unsigned int slot, ccl::WebGPUSvmTriangleCorners &corners) const
  {
    corners = ccl::webgpu_svm_triangle_corners(cycles_svm_word, texture, slot, primitive);
    return true;
  }
  bool ies(const unsigned int slot, const float h_angle, const float v_angle, float &value) const
  {
    const unsigned int table = cycles_svm_word(texture + ccl::webgpu_svm_ies_offset);
    if (table == 0) {
      return false;
    }
    /* Only the IES accessor is read, so leave the kernel data uninitialized. */
    ccl::SvmGlobals globals;
    globals.ies = {texture + table};
    value = ccl::kernel_ies_interp(&globals, int(slot), h_angle, v_angle);
    return true;
  }
  ccl::float3 cie_color_match_row(const int row) const
  {
    return ccl::webgpu_svm_read_rgb(cycles_svm_word,
                                    texture + ccl::webgpu_svm_cie_table_offset + 3u * unsigned(row));
  }
};
}  // namespace

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
                         Float4 &emission)
{
  specular = {};
  layers = {};
  transmission = {};
  subsurface = {};
  emission = {};
  const auto fetch = [](const unsigned int index) {
    return cycles_svm_word(ccl::webgpu_svm_header_words + index);
  };
  const unsigned int word_count = cycles_svm_word(0);
  const unsigned int texture = ccl::webgpu_svm_header_words + word_count;
  const auto uv = ccl::webgpu_svm_triangle_uv(cycles_svm_word, texture, primitive, u, v);
  const auto generated = ccl::webgpu_svm_triangle_generated(cycles_svm_word, texture, primitive, u, v);
  const auto color_space = ccl::webgpu_svm_read_color_space(cycles_svm_word, texture);
  const auto image = [texture](int id, ccl::float2 co, bool alpha, ccl::float4 &color) {
    return ccl::webgpu_svm_image(cycles_svm_word, texture, id, co, alpha, color);
  };
  ccl::WebGPUSvmMaterial material;
  bool uses_uv = false;
  bool uses_generated = false;
  const auto surface = [lane](const unsigned int field) { return cycles_svm_context(lane, field); };
  if (!ccl::webgpu_svm_eval_material(
          fetch, image, uv, generated, color_space, uses_uv, uses_generated, word_count,
          cycles_svm_word(1) & CYCLES_WEBGPU_SVM_SHADER_COUNT, shader, material, surface,
          PackedWordsHost{{}, texture, primitive, u, v}))
  {
    base = {0, 0, 0, 2};
    return;
  }

  base = {material.color.x, material.color.y, material.color.z, 1.0f};
  /* The indirect clamp rides along for the path update's film_clamp_light(). */
  emission = {material.emission.x,
              material.emission.y,
              material.emission.z,
              ccl::__uint_as_float(cycles_svm_word(texture + ccl::webgpu_svm_indirect_clamp_offset))};
  if (material.holdout) {
    base.w = CYCLES_WEBGPU_MATERIAL_HOLDOUT;
  }
  else if (material.ggx) {
    base.w = material.multiscatter ? CYCLES_WEBGPU_MATERIAL_MULTISCATTER_GGX :
                                     CYCLES_WEBGPU_MATERIAL_GGX;
    specular = {material.specular_tint.x,
                material.specular_tint.y,
                material.specular_tint.z,
                material.alpha};
    layers = {material.metallic, material.eta, material.f0, material.mix_weight};
    transmission = {material.transmission,
                    material.glass_ior,
                    material.subsurface > 0.0f ? float(material.subsurface_method) :
                                                material.standalone_transmission,
                    material.subsurface};
    if (material.subsurface > 0.0f) {
      subsurface = {material.subsurface_radius.x,
                    material.subsurface_radius.y,
                    material.subsurface_radius.z,
                    material.subsurface_anisotropy};
    }
  }
}
