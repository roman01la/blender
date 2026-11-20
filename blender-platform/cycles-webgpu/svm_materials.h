/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "device/cpu/device_impl.h"
#include "device/webgpu/svm_material_eval.h"
#include "scene/mesh.h"
#include "scene/scene.h"
#include "scene/shader.h"
#include "util/color.h"

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

CCL_NAMESPACE_BEGIN

struct WebGPUSvmImage {
  int id;
  KernelImageInfo info;
  bool alpha = false;

  float4 pixel(size_t index) const
  {
    if (info.data_type == IMAGE_DATA_TYPE_BYTE4) {
      return color_uchar4_to_float4(reinterpret_cast<const uchar4 *>(uintptr_t(info.data))[index]);
    }
    return reinterpret_cast<const float4 *>(uintptr_t(info.data))[index];
  }

  size_t pixel_count() const
  {
    return size_t(info.width) * info.height;
  }
};

inline bool cycles_webgpu_svm_image_info(const Scene &scene, int id, KernelImageInfo &result)
{
  if ((scene.device->info.type != DEVICE_CPU && scene.device->info.type != DEVICE_WEBGPU) || id < 0 ||
      size_t(id) >= scene.dscene.image_textures.size())
  {
    return false;
  }
  const KernelImageTexture &texture = scene.dscene.image_textures.data()[id];
  const auto &images = *static_cast<const CPUDevice *>(scene.device)->image_info;
  if (texture.tile_descriptor_offset != KERNEL_TILE_LOAD_NONE ||
      texture.image_info_id >= images.size())
  {
    return false;
  }

  result = images.data()[texture.image_info_id];
  return result.data &&
         (result.data_type == IMAGE_DATA_TYPE_FLOAT4 || result.data_type == IMAGE_DATA_TYPE_BYTE4) &&
         (result.interpolation == INTERPOLATION_CLOSEST ||
          result.interpolation == INTERPOLATION_LINEAR ||
          webgpu_svm_uses_cubic_filter(result.interpolation)) &&
         (result.extension == EXTENSION_REPEAT || result.extension == EXTENSION_EXTEND ||
          result.extension == EXTENSION_CLIP || result.extension == EXTENSION_MIRROR) &&
         result.width > 0 && result.height > 0 && result.width <= 32768 && result.height <= 32768;
}

/* The compiled active UV attribute is stored per triangle corner. Mesh prim_offset
 * indexes the same triangle array used by the synchronized BVH and shader IDs. */
inline bool cycles_webgpu_svm_triangle_attributes(const Scene &scene,
                                                  const std::vector<bool> &shaders,
                                                  uint32_t *destination)
{
  const auto &tri_shader = scene.dscene.tri_shader;
  size_t required = 0;
  for (size_t i = 0; i < tri_shader.size(); ++i) {
    required += shaders[tri_shader.data()[i] & SHADER_MASK];
  }
  for (const Geometry *geom : scene.geometry) {
    if (!geom->is_mesh()) {
      continue;
    }
    const auto &mesh = *static_cast<const Mesh *>(geom);
    if (mesh.prim_offset > tri_shader.size() ||
        mesh.num_triangles() > tri_shader.size() - mesh.prim_offset)
    {
      return false;
    }
    const Attribute *attr = mesh.attributes.find(ATTR_STD_UV);
    const bool have_uv = attr && attr->element == ATTR_ELEMENT_CORNER &&
                         attr->type == TypeFloat2 && attr->size >= 0 &&
                         size_t(attr->size) == mesh.num_triangles() * 3;
    const float2 *uv = have_uv ? attr->data<float2>() : nullptr;
    for (size_t i = 0; i < mesh.num_triangles(); ++i) {
      const size_t triangle = mesh.prim_offset + i;
      if (!shaders[tri_shader.data()[triangle] & SHADER_MASK]) {
        continue;
      }
      if (!uv || required == 0) {
        return false;
      }
      for (size_t corner = 0; corner < 3; ++corner) {
        const float2 value = uv[i * 3 + corner];
        if (!(fabsf(value.x) <= 8192.0f && fabsf(value.y) <= 8192.0f)) {
          return false;
        }
        const size_t word = triangle * webgpu_svm_triangle_uv_words + 2 * corner;
        destination[word] = std::bit_cast<uint32_t>(value.x);
        destination[word + 1] = std::bit_cast<uint32_t>(value.y);
      }
      --required;
    }
  }
  return required == 0;
}

inline bool cycles_webgpu_svm_generated_attributes(const Scene &scene,
                                                   const std::vector<bool> &shaders,
                                                   uint32_t *destination)
{
  const auto &triangle_shaders = scene.dscene.tri_shader;
  size_t pending = 0;
  for (size_t i = 0; i < triangle_shaders.size(); ++i) {
    pending += shaders[triangle_shaders.data()[i] & SHADER_MASK];
  }

  for (const Geometry *geometry : scene.geometry) {
    if (!geometry->is_mesh()) {
      continue;
    }
    const auto &mesh = static_cast<const Mesh &>(*geometry);
    if (mesh.prim_offset > triangle_shaders.size() ||
        mesh.num_triangles() > triangle_shaders.size() - mesh.prim_offset)
    {
      return false;
    }
    const Attribute *attribute = mesh.attributes.find(ATTR_STD_GENERATED);
    const bool valid = attribute && attribute->element == ATTR_ELEMENT_VERTEX &&
                       attribute->type == TypePoint && attribute->size >= 0 &&
                       size_t(attribute->size) == mesh.num_verts();
    const packed_float3 *generated = valid ? attribute->data<packed_float3>() : nullptr;
    for (size_t i = 0; i < mesh.num_triangles(); ++i) {
      const size_t triangle = mesh.prim_offset + i;
      if (!shaders[triangle_shaders.data()[triangle] & SHADER_MASK]) {
        continue;
      }
      if (!generated || pending == 0) {
        return false;
      }
      const Mesh::Triangle indices = mesh.get_triangle(i);
      for (size_t corner = 0; corner < 3; ++corner) {
        if (size_t(indices.v[corner]) >= mesh.num_verts()) {
          return false;
        }
        const packed_float3 value = generated[indices.v[corner]];
        if (!(fabsf(value.x) <= 8192.0f && fabsf(value.y) <= 8192.0f &&
              fabsf(value.z) <= 8192.0f))
        {
          return false;
        }
        const size_t word = triangle * webgpu_svm_triangle_generated_words + 3 * corner;
        destination[word] = std::bit_cast<uint32_t>(value.x);
        destination[word + 1] = std::bit_cast<uint32_t>(value.y);
        destination[word + 2] = std::bit_cast<uint32_t>(value.z);
      }
      --pending;
    }
  }
  return pending == 0;
}

/* Keep the synchronized scene bytecode verbatim and snapshot the texture inputs
 * used by triangle materials. Eval and eligibility share the supported opcodes. */
inline bool cycles_webgpu_encode_svm_materials(const Scene &scene,
                                               std::vector<uint32_t> &words,
                                               bool &has_ggx,
                                               bool &has_transmission,
                                               bool &has_subsurface,
                                               std::string &reason,
                                               size_t max_words)
{
  const DeviceScene &dscene = scene.dscene;
  max_words = std::min(max_words, size_t(std::numeric_limits<uint32_t>::max()));
  const size_t count = dscene.svm_nodes.size();
  const size_t shader_count = dscene.shaders.size();
  const size_t header = webgpu_svm_header_words + webgpu_svm_texture_header_words;
  if (!count || max_words < header || count > max_words - header ||
      shader_count > CYCLES_WEBGPU_SVM_SHADER_COUNT ||
      shader_count > count / webgpu_svm_jump_words || shader_count != scene.shaders.size())
  {
    reason = "WebGPU Cycles needs synchronized SVM bytecode within the scene budget";
    return false;
  }

  std::vector<WebGPUSvmImage> images;
  size_t image_slots = 0;
  size_t remaining = max_words - header - count;
  bool requires_extended_svm = false;
  const auto fetch = [&](const uint index) { return uint(dscene.svm_nodes.data()[index]); };
  const auto image = [&](int id, float2 uv, bool alpha, float4 &color) {
    auto found = std::find_if(images.begin(), images.end(), [id](const WebGPUSvmImage &item) {
      return item.id == id;
    });
    if (found == images.end()) {
      WebGPUSvmImage item{id, {}};
      if (!cycles_webgpu_svm_image_info(scene, id, item.info)) {
        return false;
      }
      const size_t slots = std::max(image_slots, size_t(id) + 1);
      const size_t added_slots = slots - image_slots;
      if (added_slots > remaining / webgpu_svm_image_words) {
        return false;
      }
      const size_t required = added_slots * webgpu_svm_image_words + item.pixel_count() * 3;
      if (required > remaining) {
        return false;
      }
      remaining -= required;
      image_slots = slots;
      requires_extended_svm |= webgpu_svm_uses_cubic_filter(item.info.interpolation);
      images.push_back(item);
      found = images.end() - 1;
    }
    if (alpha && !found->alpha) {
      if (found->pixel_count() > remaining) {
        return false;
      }
      remaining -= found->pixel_count();
      found->alpha = true;
    }
    const auto &info = found->info;
    color = webgpu_svm_sample_image<true>(
        [&](const uint index) { return found->pixel(index); },
        info.width,
        info.height,
        info.extension,
        info.interpolation,
        uv);
    return true;
  };

  std::vector<bool> checked(shader_count, false);
  std::vector<bool> uv_shaders(shader_count, false);
  std::vector<bool> generated_shaders(shader_count, false);
  bool have_uv = false;
  bool have_generated = false;
  bool have_ggx = false;
  bool have_transmission = false;
  bool have_subsurface = false;
  const float3 rgb_to_y = make_float3(dscene.data.film.rgb_to_y);
  for (size_t i = 0; i < dscene.tri_shader.size(); ++i) {
    const uint shader = dscene.tri_shader.data()[i] & SHADER_MASK;
    if (shader >= shader_count) {
      reason = "WebGPU Cycles found an invalid SVM material";
      return false;
    }
    if (checked[shader]) {
      continue;
    }

    const Shader &material = *scene.shaders[shader];
    WebGPUSvmMaterial decoded;
    bool uses_uv = false;
    bool uses_generated = false;
    if (material.id != shader || material.has_volume_connected || material.has_volume ||
        material.has_displacement || material.has_bump_from_surface ||
        material.has_bump_from_displacement || material.has_aov_output_node ||
        !webgpu_svm_eval_material(fetch,
                                 image,
                                 WebGPUSvmUv{0.25f, 0.25f},
                                 make_float3(0.25f, 0.25f, 0.25f),
                                 rgb_to_y,
                                 uses_uv,
                                 uses_generated,
                                 uint(count),
                                 uint(shader_count),
                                 shader,
                                 decoded))
    {
      reason = "WebGPU Cycles needs supported Lambert, Principled or Glass SVM materials";
      return false;
    }
    checked[shader] = true;
    uv_shaders[shader] = uses_uv;
    generated_shaders[shader] = uses_generated;
    have_uv |= uses_uv;
    have_generated |= uses_generated;
    have_ggx |= decoded.ggx;
    have_transmission |= decoded.ggx && decoded.transmission > 0.0f;
    have_subsurface |= decoded.subsurface > 0.0f;
    requires_extended_svm |= decoded.requires_extended_svm;
  }

  const size_t triangles = have_uv ? dscene.tri_shader.size() : 0;
  if (triangles > remaining / webgpu_svm_triangle_uv_words) {
    reason = "WebGPU Cycles triangle UVs exceed the scene budget";
    return false;
  }
  remaining -= triangles * webgpu_svm_triangle_uv_words;
  const size_t generated_triangles = have_generated ? dscene.tri_shader.size() : 0;
  if (generated_triangles > remaining / webgpu_svm_triangle_generated_words) {
    reason = "WebGPU Cycles Generated coordinates exceed the scene budget";
    return false;
  }
  remaining -= generated_triangles * webgpu_svm_triangle_generated_words;
  std::vector<uint32_t> packed(max_words - remaining);
  packed[0] = uint32_t(count);
  packed[1] = uint32_t(shader_count) | (requires_extended_svm ? CYCLES_WEBGPU_SVM_EXTENDED : 0);
  for (size_t i = 0; i < count; ++i) {
    packed[webgpu_svm_header_words + i] = uint32_t(dscene.svm_nodes.data()[i]);
  }
  const size_t texture = webgpu_svm_header_words + count;
  packed[texture] = uint32_t(triangles);
  packed[texture + 1] = uint32_t(image_slots);
  packed[texture + 2] = uint32_t(generated_triangles);
  packed[texture + 3] = std::bit_cast<uint32_t>(rgb_to_y.x);
  packed[texture + 4] = std::bit_cast<uint32_t>(rgb_to_y.y);
  packed[texture + 5] = std::bit_cast<uint32_t>(rgb_to_y.z);
  const size_t uv = texture + webgpu_svm_texture_header_words;
  if (have_uv && !cycles_webgpu_svm_triangle_attributes(scene, uv_shaders, packed.data() + uv)) {
    reason = "WebGPU Cycles needs an active corner UV layer on textured triangles";
    return false;
  }

  const size_t generated = uv + triangles * webgpu_svm_triangle_uv_words;
  if (have_generated &&
      !cycles_webgpu_svm_generated_attributes(scene, generated_shaders, packed.data() + generated))
  {
    reason = "WebGPU Cycles needs synchronized vertex Generated coordinates";
    return false;
  }
  const size_t image_table = generated + generated_triangles * webgpu_svm_triangle_generated_words;
  size_t texel = image_table + image_slots * webgpu_svm_image_words;
  for (const WebGPUSvmImage &item : images) {
    size_t record = image_table + size_t(item.id) * webgpu_svm_image_words;
    packed[record++] = item.info.width;
    packed[record++] = item.info.height;
    packed[record++] = item.info.extension;
    packed[record++] = uint32_t(item.info.interpolation) |
                       (item.alpha ? webgpu_svm_image_alpha : 0);
    packed[record++] = uint32_t(texel);
    const size_t alpha_offset = texel + item.pixel_count() * 3;
    for (size_t i = 0; i < item.pixel_count(); ++i) {
      const float4 rgba = item.pixel(i);
      if (!(fabsf(rgba.x) <= 1.0e6f && fabsf(rgba.y) <= 1.0e6f && fabsf(rgba.z) <= 1.0e6f) ||
          (item.alpha && !(fabsf(rgba.w) <= 1.0e6f)))
      {
        reason = "WebGPU Cycles needs finite texture colors";
        return false;
      }
      packed[texel++] = std::bit_cast<uint32_t>(rgba.x);
      packed[texel++] = std::bit_cast<uint32_t>(rgba.y);
      packed[texel++] = std::bit_cast<uint32_t>(rgba.z);
      if (item.alpha) {
        packed[alpha_offset + i] = std::bit_cast<uint32_t>(rgba.w);
      }
    }
    if (item.alpha) {
      texel += item.pixel_count();
    }
  }

  words = std::move(packed);
  has_ggx = have_ggx;
  has_transmission = have_transmission;
  has_subsurface = have_subsurface;
  reason.clear();
  return true;
}

CCL_NAMESPACE_END
