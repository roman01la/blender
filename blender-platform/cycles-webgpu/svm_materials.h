/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "device/cpu/device_impl.h"
#include "device/webgpu/svm_material_eval.h"
#include "device/webgpu/volume_layout.h"
#include "kernel/constants.h"
#include "scene/mesh.h"
#include "scene/object.h"
#include "scene/scene.h"
#include "scene/shader.h"
#include "scene/shader_nodes.h"
#include "util/color.h"

#include <algorithm>
#include <limits>
#include <span>
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

enum class WebGPUSvmAttributeLookup { Missing, Found, Unreadable };

/* Mirror find_attr_offset() and find_attribute() for a triangle on the host. */
inline WebGPUSvmAttributeLookup cycles_webgpu_svm_find_attribute(const DeviceScene &dscene,
                                                                 const int object,
                                                                 const uint64_t id,
                                                                 AttributeMap &result)
{
  if (object < 0 || size_t(object) >= dscene.objects.size()) {
    return WebGPUSvmAttributeLookup::Unreadable;
  }
  const AttributeMap *map = dscene.attributes_map.data();
  const size_t size = dscene.attributes_map.size();
  uint offset = dscene.objects.data()[object].attribute_map_offset;
  for (size_t step = 0; step <= size && offset < size; ++step) {
    const AttributeMap &entry = map[offset];
    if (entry.id == id) {
      result = entry;
      return entry.element == ATTR_ELEMENT_NONE ? WebGPUSvmAttributeLookup::Missing :
                                                  WebGPUSvmAttributeLookup::Found;
    }
    if (entry.id == ATTR_STD_NONE) {
      if (entry.element == 0) {
        return WebGPUSvmAttributeLookup::Missing;
      }
      offset = uint(entry.offset);
    }
    else {
      offset += ATTR_PRIM_TYPES;
    }
  }
  return WebGPUSvmAttributeLookup::Unreadable;
}

/* attribute_data_fetch() for the value type primitive_surface_attribute()
 * reads for each NodeAttributeType, widened to float4. */
inline bool cycles_webgpu_svm_fetch_attribute(const DeviceScene &dscene,
                                              const AttributeMap &desc,
                                              const int64_t index,
                                              float4 &value)
{
  const auto valid = [index](const auto &array) {
    return index >= 0 && size_t(index) < array.size();
  };
  switch (desc.type) {
    case NODE_ATTR_FLOAT:
      if (!valid(dscene.attributes_float)) {
        return false;
      }
      value = make_float4(dscene.attributes_float.data()[index], 0.0f, 0.0f, 0.0f);
      break;
    case NODE_ATTR_FLOAT2: {
      if (!valid(dscene.attributes_float2)) {
        return false;
      }
      const float2 f = dscene.attributes_float2.data()[index];
      value = make_float4(f.x, f.y, 0.0f, 0.0f);
      break;
    }
    case NODE_ATTR_FLOAT3: {
      float3 f;
      if (desc.element & ATTR_ELEMENT_IS_NORMAL) {
        if (!valid(dscene.attributes_normal)) {
          return false;
        }
        f = dscene.attributes_normal.data()[index].decode();
      }
      else {
        if (!valid(dscene.attributes_float3)) {
          return false;
        }
        f = make_float3(dscene.attributes_float3.data()[index]);
      }
      value = make_float4(f.x, f.y, f.z, 0.0f);
      break;
    }
    case NODE_ATTR_FLOAT4:
    case NODE_ATTR_RGBA:
      if (desc.element & ATTR_ELEMENT_IS_BYTE) {
        if (!valid(dscene.attributes_uchar4)) {
          return false;
        }
        const KernelFilm &film = dscene.data.film;
        const float4 rec709 = color_srgb_to_linear_v4(
            color_uchar4_to_float4(dscene.attributes_uchar4.data()[index]));
        const float3 rgb = film.is_rec709 ?
                               make_float3(rec709) :
                               make_float3(dot(make_float3(film.rec709_to_r), make_float3(rec709)),
                                           dot(make_float3(film.rec709_to_g), make_float3(rec709)),
                                           dot(make_float3(film.rec709_to_b), make_float3(rec709)));
        value = make_float4(rgb.x, rgb.y, rgb.z, rec709.w);
      }
      else {
        if (!valid(dscene.attributes_float4)) {
          return false;
        }
        value = dscene.attributes_float4.data()[index];
      }
      break;
    default:
      return false;
  }
  return isfinite_safe(value.x) && isfinite_safe(value.y) && isfinite_safe(value.z) &&
         isfinite_safe(value.w);
}

/* Slot IDs above every attribute ID name host-built data. */
constexpr uint64_t webgpu_svm_synthetic_id = 0xffffffff00000000ull;

/* Corner data for a host-built slot on one triangle, or false when the stock
 * kernel would read outside the synchronized arrays. */
inline bool cycles_webgpu_svm_synthetic_record(const DeviceScene &dscene,
                                               const int object,
                                               const size_t prim,
                                               const WebGPUSvmSyntheticSlot kind,
                                               std::array<uint32_t, webgpu_svm_attribute_words> &record)
{
  if (object < 0 || size_t(object) >= dscene.objects.size() ||
      size_t(object) >= dscene.object_flag.size() || prim >= dscene.tri_vindex.size())
  {
    return false;
  }
  const KernelObject &kobject = dscene.objects.data()[object];
  const uint object_flag = dscene.object_flag.data()[object];
  const packed_uint3 vertices = dscene.tri_vindex.data()[prim];
  std::array<float3, 3> corners;
  uint header;
  if (kind == WEBGPU_SVM_SLOT_SMOOTH_NORMAL) {
    /* triangle_smooth_normal_unnormalized_object_space() */
    const std::array<int64_t, 3> indices =
        (object_flag & SD_OBJECT_HAS_CORNER_NORMALS) ?
            std::array<int64_t, 3>{int64_t(prim) * 3, int64_t(prim) * 3 + 1, int64_t(prim) * 3 + 2} :
            std::array<int64_t, 3>{vertices.x, vertices.y, vertices.z};
    for (size_t corner = 0; corner < 3; ++corner) {
      const int64_t index = int64_t(kobject.mesh_volume.normal_offset) + indices[corner];
      if (index < 0 || size_t(index) >= dscene.attributes_normal.size()) {
        return false;
      }
      corners[corner] = dscene.attributes_normal.data()[index].decode();
      if (object_flag & SD_OBJECT_TRANSFORM_APPLIED) {
        /* object_inverse_normal_transform() */
        corners[corner] = safe_normalize(transform_direction_transposed(&kobject.tfm,
                                                                        corners[corner]));
      }
    }
    header = WEBGPU_SVM_ATTRIBUTE_INTERPOLATE | (uint(NODE_ATTR_FLOAT3) << 8);
  }
  else {
    /* triangle_vertices() */
    for (size_t corner = 0; corner < 3; ++corner) {
      const int64_t index = int64_t(kobject.position_offset) +
                            int64_t(corner == 0 ? vertices.x : corner == 1 ? vertices.y : vertices.z);
      if (index < 0 || size_t(index) >= dscene.tri_verts.size()) {
        return false;
      }
      corners[corner] = make_float3(dscene.tri_verts.data()[index]);
    }
    header = WEBGPU_SVM_ATTRIBUTE_CONSTANT | (uint(NODE_ATTR_FLOAT3) << 8) |
             ((object_flag & SD_OBJECT_TRANSFORM_APPLIED) ? 0u : webgpu_svm_slot_object_transform);
  }
  record.fill(0);
  record[0] = header;
  for (size_t corner = 0; corner < 3; ++corner) {
    if (!isfinite_safe(corners[corner].x) || !isfinite_safe(corners[corner].y) ||
        !isfinite_safe(corners[corner].z))
    {
      return false;
    }
    record[1 + 4 * corner] = std::bit_cast<uint32_t>(corners[corner].x);
    record[2 + 4 * corner] = std::bit_cast<uint32_t>(corners[corner].y);
    record[3 + 4 * corner] = std::bit_cast<uint32_t>(corners[corner].z);
  }
  return true;
}

/* Fill each slot's per-triangle record following primitive_surface_attribute()
 * and triangle_attribute(). Triangles of objects without the attribute keep
 * the zero missing record, so the shader uses the node's Missing inputs. */
inline bool cycles_webgpu_svm_attribute_records(const Scene &scene,
                                                const std::vector<uint64_t> &ids,
                                                uint32_t *destination,
                                                std::string &reason)
{
  const DeviceScene &dscene = scene.dscene;
  const size_t triangles = dscene.tri_shader.size();
  std::vector<bool> written(ids.size() * triangles, false);
  reason = "WebGPU Cycles needs supported triangle attributes";
  for (const Object *object : scene.objects) {
    const Geometry *geometry = object->get_geometry();
    if (!geometry || !geometry->is_mesh()) {
      continue;
    }
    const auto &mesh = static_cast<const Mesh &>(*geometry);
    if (mesh.prim_offset > triangles || mesh.num_triangles() > triangles - mesh.prim_offset) {
      return false;
    }
    for (size_t slot = 0; slot < ids.size(); ++slot) {
      if (ids[slot] >= webgpu_svm_synthetic_id) {
        const auto kind = WebGPUSvmSyntheticSlot(ids[slot] - webgpu_svm_synthetic_id);
        for (size_t i = 0; i < mesh.num_triangles(); ++i) {
          const size_t prim = mesh.prim_offset + i;
          std::array<uint32_t, webgpu_svm_attribute_words> record;
          if (!cycles_webgpu_svm_synthetic_record(
                  dscene, object->get_device_index(), prim, kind, record))
          {
            /* Flat triangles do not read the smooth normals. */
            if (kind != WEBGPU_SVM_SLOT_SMOOTH_NORMAL ||
                (dscene.tri_shader.data()[prim] & SHADER_SMOOTH_NORMAL))
            {
              return false;
            }
            record.fill(0);
          }
          const size_t entry = slot * triangles + prim;
          uint32_t *target = destination + entry * webgpu_svm_attribute_words;
          if (written[entry] && !std::equal(record.begin(), record.end(), target)) {
            reason = "WebGPU Cycles needs matching attribute values on instanced triangles";
            return false;
          }
          std::copy(record.begin(), record.end(), target);
          written[entry] = true;
        }
        continue;
      }
      AttributeMap desc{};
      const WebGPUSvmAttributeLookup lookup = cycles_webgpu_svm_find_attribute(
          dscene, object->get_device_index(), ids[slot], desc);
      if (lookup == WebGPUSvmAttributeLookup::Unreadable ||
          (lookup == WebGPUSvmAttributeLookup::Found &&
           (desc.element & ATTR_ELEMENT_IS_MOTION) != 0))
      {
        return false;
      }
      const bool found = lookup == WebGPUSvmAttributeLookup::Found;
      for (size_t i = 0; i < mesh.num_triangles(); ++i) {
        const size_t prim = mesh.prim_offset + i;
        std::array<uint32_t, webgpu_svm_attribute_words> record{};
        if (found) {
          uint mode = WEBGPU_SVM_ATTRIBUTE_CONSTANT;
          std::array<float4, 3> corners{zero_float4(), zero_float4(), zero_float4()};
          if (desc.element & (ATTR_ELEMENT_OBJECT | ATTR_ELEMENT_MESH)) {
            if (!cycles_webgpu_svm_fetch_attribute(dscene, desc, desc.offset, corners[0])) {
              return false;
            }
          }
          else if (desc.element & (ATTR_ELEMENT_VERTEX | ATTR_ELEMENT_CORNER)) {
            mode = WEBGPU_SVM_ATTRIBUTE_INTERPOLATE;
            if (prim >= dscene.tri_vindex.size()) {
              return false;
            }
            const packed_uint3 vertices = dscene.tri_vindex.data()[prim];
            const std::array<int64_t, 3> indices =
                (desc.element & ATTR_ELEMENT_VERTEX) ?
                    std::array<int64_t, 3>{vertices.x, vertices.y, vertices.z} :
                    std::array<int64_t, 3>{int64_t(prim) * 3, int64_t(prim) * 3 + 1,
                                           int64_t(prim) * 3 + 2};
            for (size_t corner = 0; corner < 3; ++corner) {
              if (!cycles_webgpu_svm_fetch_attribute(
                      dscene, desc, int64_t(desc.offset) + indices[corner], corners[corner]))
              {
                return false;
              }
            }
          }
          else if (desc.element & ATTR_ELEMENT_FACE) {
            if (!cycles_webgpu_svm_fetch_attribute(
                    dscene, desc, int64_t(desc.offset) + int64_t(prim), corners[0]))
            {
              return false;
            }
          }
          else if (desc.type > NODE_ATTR_RGBA) {
            /* triangle_attribute() returns zero for other elements. */
            return false;
          }
          record[0] = mode | (uint(desc.type) << 8);
          for (size_t corner = 0; corner < 3; ++corner) {
            record[1 + 4 * corner] = std::bit_cast<uint32_t>(corners[corner].x);
            record[2 + 4 * corner] = std::bit_cast<uint32_t>(corners[corner].y);
            record[3 + 4 * corner] = std::bit_cast<uint32_t>(corners[corner].z);
            record[4 + 4 * corner] = std::bit_cast<uint32_t>(corners[corner].w);
          }
        }
        uint32_t *target = destination + (slot * triangles + prim) * webgpu_svm_attribute_words;
        const size_t entry = slot * triangles + prim;
        if (written[entry]) {
          /* Instances share triangles, so per-object values must agree. */
          if (!std::equal(record.begin(), record.end(), target)) {
            reason = "WebGPU Cycles needs matching attribute values on instanced triangles";
            return false;
          }
          continue;
        }
        std::copy(record.begin(), record.end(), target);
        written[entry] = true;
      }
    }
  }
  reason.clear();
  return true;
}

/* Host answers for one material: the objects using it, the bytecode words of
 * attributes none of them provide, which packing marks as missing, and the
 * packed slots of the other attributes. */
struct WebGPUSvmSceneHost : WebGPUSvmPackedHost {
  const DeviceScene *dscene;
  const std::vector<int> *objects;
  std::vector<uint> *missing;
  std::vector<uint64_t> *slots;
  std::vector<uint> *slot_types;
  std::vector<uint2> *slot_words;
  uint2 *last_node;
  bool *uses_ies;
  bool *uses_shadow_path_context;

  bool ies(const uint slot, const float /*h_angle*/, const float /*v_angle*/, float &value) const
  {
    /* Packing uploads the synchronized IES table when a material reads it. */
    if (slot >= dscene->ies_lights.size()) {
      return false;
    }
    *uses_ies = true;
    value = 1.0f;
    return true;
  }

  /* The slot index for an ID, appended with its stored type when new. */
  bool slot_index(const uint64_t id, const uint type, uint &index) const
  {
    auto slot = std::find(slots->begin(), slots->end(), id);
    if (slot == slots->end()) {
      if (slots->size() == webgpu_svm_attribute_slots) {
        return false;
      }
      slot = slots->insert(slots->end(), id);
      slot_types->push_back(type);
    }
    index = uint(slot - slots->begin());
    return true;
  }

  void visit(const uint type, const uint word) const
  {
    *last_node = make_uint2(type, word);
    *uses_shadow_path_context |= type == NODE_LIGHT_PATH || type == NODE_AMBIENT_OCCLUSION ||
                                 type == NODE_RAYCAST || type == NODE_BEVEL;
  }

  int attribute(const uint word, const int attr) const
  {
    if (objects->empty()) {
      return attr;
    }
    bool found = false;
    uint type = 0;
    for (const int object : *objects) {
      AttributeMap desc;
      const WebGPUSvmAttributeLookup lookup = cycles_webgpu_svm_find_attribute(
          *dscene, object, uint64_t(attr), desc);
      if (lookup == WebGPUSvmAttributeLookup::Unreadable) {
        return attr;
      }
      if (lookup == WebGPUSvmAttributeLookup::Found && !found) {
        type = desc.type;
      }
      found |= lookup == WebGPUSvmAttributeLookup::Found;
    }
    if (!found) {
      missing->push_back(word);
      return int(ATTR_STD_NOT_FOUND);
    }
    uint index;
    if (!slot_index(uint64_t(attr), type, index)) {
      return attr;
    }
    slot_words->push_back(make_uint2(word, index));
    return int(webgpu_svm_attribute_slot | index);
  }

  int synthetic(const uint word, const int packed, const WebGPUSvmSyntheticSlot kind) const
  {
    if (objects->empty()) {
      return -1;
    }
    if (kind == WEBGPU_SVM_SLOT_SMOOTH_NORMAL && packed) {
      /* Normal Map's Original base reads undisplaced normals when present. */
      for (const int object : *objects) {
        AttributeMap desc;
        if (cycles_webgpu_svm_find_attribute(
                *dscene, object, ATTR_STD_NORMAL_UNDISPLACED, desc) !=
            WebGPUSvmAttributeLookup::Missing)
        {
          return -1;
        }
      }
    }
    uint index;
    if (!slot_index(webgpu_svm_synthetic_id + kind, NODE_ATTR_FLOAT3, index)) {
      return -1;
    }
    slot_words->push_back(make_uint2(word, index));
    return int(webgpu_svm_attribute_slot | index);
  }

  bool attribute_sample(const uint slot, WebGPUSvmAttributeSample &sample) const
  {
    /* Packing validates each stored value. Values vary across the surface, and a
     * unit value keeps tangent projections finite during eligibility. */
    sample = {WEBGPU_SVM_ATTRIBUTE_INTERPOLATE,
              slot < slot_types->size() ? (*slot_types)[slot] : 0u,
              make_float4(1.0f, 0.0f, 0.0f, 1.0f)};
    return true;
  }

  bool triangle_corners(const uint /*slot*/, WebGPUSvmTriangleCorners &corners) const
  {
    corners = {0, {make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 0.0f, 0.0f),
                   make_float3(0.0f, 1.0f, 0.0f)}};
    return true;
  }

  bool dummy_particle() const
  {
    return !objects->empty() &&
           std::all_of(objects->begin(), objects->end(), [this](const int object) {
             return object >= 0 && size_t(object) < dscene->objects.size() &&
                    dscene->objects.data()[object].particle_index == 0;
           });
  }
};

/* Keep one stock SVM volume closure in its typed encoding. Scene eligibility
 * guarantees the kernel can evaluate it without a stack or grid attributes. */
inline bool webgpu_svm_literal_volume(const DeviceScene &scene,
                                     const uint shader,
                                     const int object,
                                     uint32_t *words,
                                     bool &absorption,
                                     bool &has_voxels,
                                     uint &phase_slots,
                                     const bool allow_voxel_attributes)
{
  const uint count = uint(scene.svm_nodes.size());
  const auto fetch = [&](const uint index) { return uint(scene.svm_nodes.data()[index]); };
  const uint shader_count = uint(scene.shaders.size());
  if (shader_count > count / webgpu_svm_jump_words || shader >= shader_count) {
    return false;
  }
  uint offset = shader * webgpu_svm_jump_words;
  SVMNodeShaderJump jump;
  if (fetch(offset++) != NODE_SHADER_JUMP || !webgpu_svm_read(fetch, offset, count, jump) ||
      uint(jump.offset_volume) < shader_count * webgpu_svm_jump_words ||
      uint(jump.offset_displacement) > count || jump.offset_displacement <= jump.offset_volume)
  {
    return false;
  }
  offset = uint(jump.offset_volume);
  const uint end = uint(jump.offset_displacement);
  SVMNodeClosureSetWeight weight;
  if (fetch(offset++) != NODE_CLOSURE_SET_WEIGHT || !webgpu_svm_read(fetch, offset, end, weight) ||
      offset == end)
  {
    return false;
  }
  const float3 color = make_float3(weight.rgb);
  if (!webgpu_svm_finite(color.x) || !webgpu_svm_finite(color.y) || !webgpu_svm_finite(color.z) ||
      reduce_min(color) < 0.0f)
  {
    return false;
  }

  const uint type = fetch(offset++);
  const uint start = offset;
  const WebGPUSvmStackUsage empty_stack;
  const auto valid_phase = [](const ClosureType phase) {
    return phase >= CLOSURE_VOLUME_HENYEY_GREENSTEIN_ID && phase <= CLOSURE_VOLUME_DRAINE_ID;
  };
  const auto phase_count = [](const ClosureType phase) {
    return phase == CLOSURE_VOLUME_MIE_ID ? 2u : 1u;
  };
  switch (type) {
    case NODE_CLOSURE_VOLUME: {
      SVMNodeClosureVolume node;
      if (!webgpu_svm_read(fetch, offset, end, node) ||
          node.mix_weight_offset != SVM_STACK_INVALID ||
          !webgpu_svm_inputs_valid(empty_stack, node.density, node.param1, node.param_extra))
      {
        return false;
      }
      absorption = node.closure_type == CLOSURE_VOLUME_ABSORPTION_ID;
      if (absorption) {
        if (reduce_max(color) > 1.0f || __uint_as_float(node.density.bits) < 0.0f) {
          return false;
        }
        words[CYCLES_WEBGPU_ABSORPTION_COLOR] = std::bit_cast<uint32_t>(color.x);
        words[CYCLES_WEBGPU_ABSORPTION_COLOR + 1] = std::bit_cast<uint32_t>(color.y);
        words[CYCLES_WEBGPU_ABSORPTION_COLOR + 2] = std::bit_cast<uint32_t>(color.z);
        words[CYCLES_WEBGPU_ABSORPTION_DENSITY] = node.density.bits;
      }
      else if (!valid_phase(node.closure_type)) {
        return false;
      }
      phase_slots = absorption ? 0u : phase_count(node.closure_type);
      break;
    }
    case NODE_VOLUME_COEFFICIENTS: {
      SVMNodeVolumeCoefficients node;
      if (!webgpu_svm_read(fetch, offset, end, node) ||
          node.mix_weight_offset != SVM_STACK_INVALID || !valid_phase(node.closure_type) ||
          !webgpu_svm_inputs_valid(
              empty_stack, node.absorption_coeffs, node.emission_coeffs, node.param1, node.param_extra))
      {
        return false;
      }
      phase_slots = phase_count(node.closure_type);
      break;
    }
    case NODE_PRINCIPLED_VOLUME: {
      SVMNodePrincipledVolume node;
      if (!webgpu_svm_read(fetch, offset, end, node) || node.mix_weight_offset != SVM_STACK_INVALID ||
          !webgpu_svm_inputs_valid(empty_stack, node.absorption_color, node.emission_color,
                                   node.blackbody_tint, node.density, node.anisotropy, node.emission,
                                   node.blackbody, node.temperature) ||
          (__uint_as_float(node.blackbody.bits) > 0.0f && !scene.data.film.is_rec709))
      {
        return false;
      }
      phase_slots = 1;
      for (const int id : {node.attr_density, node.attr_color, node.attr_temperature}) {
        if (id == ATTR_STD_NOT_FOUND || id == ATTR_STD_NONE) {
          continue;
        }
        AttributeMap attribute;
        const auto lookup = cycles_webgpu_svm_find_attribute(
            scene, object, uint64_t(id), attribute);
        if (lookup == WebGPUSvmAttributeLookup::Missing) {
          continue;
        }
        if (!allow_voxel_attributes || lookup != WebGPUSvmAttributeLookup::Found ||
            attribute.element != ATTR_ELEMENT_VOXEL) {
          return false;
        }
        has_voxels = true;
      }
      break;
    }
    default:
      return false;
  }
  if (offset + 1 != end || fetch(offset) != NODE_END ||
      offset - start > CYCLES_WEBGPU_VOLUME_MAX_BOUNCES - CYCLES_WEBGPU_VOLUME_DATA)
  {
    return false;
  }
  words[CYCLES_WEBGPU_VOLUME_OPCODE] = type;
  words[CYCLES_WEBGPU_VOLUME_WEIGHT] = std::bit_cast<uint32_t>(color.x);
  words[CYCLES_WEBGPU_VOLUME_WEIGHT + 1] = std::bit_cast<uint32_t>(color.y);
  words[CYCLES_WEBGPU_VOLUME_WEIGHT + 2] = std::bit_cast<uint32_t>(color.z);
  for (uint index = start; index < offset; ++index) {
    words[CYCLES_WEBGPU_VOLUME_DATA + index - start] = fetch(index);
  }
  return true;
}

/* Keep the synchronized scene bytecode verbatim and snapshot the texture inputs
 * used by triangle materials. Eval and eligibility share the supported opcodes. */
inline bool cycles_webgpu_encode_svm_materials(const Scene &scene,
                                               std::vector<uint32_t> &words,
                                               std::vector<uint32_t> &transparent_shadows,
                                               bool &varying_transparency,
                                               bool &has_ggx,
                                               bool &has_transmission,
                                               bool &has_subsurface,
                                               bool &advanced_query_svm,
                                               std::span<const uint> volume_shaders,
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
  bool requires_advanced_procedural_svm = false;
  bool shader_uses_cubic = false;
  bool has_aov = false;
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
    const bool cubic = webgpu_svm_uses_cubic_filter(info.interpolation);
    requires_extended_svm |= cubic;
    shader_uses_cubic |= cubic;
    color = webgpu_svm_sample_image<true>(
        [&](const uint index) { return found->pixel(index); },
        info.width,
        info.height,
        info.extension,
        info.interpolation,
        uv);
    return true;
  };

  std::vector<std::vector<int>> shader_objects(shader_count);
  for (const Object *object : scene.objects) {
    const Geometry *geometry = object->get_geometry();
    if (!geometry || !geometry->is_mesh()) {
      continue;
    }
    for (const Node *node : geometry->get_used_shaders()) {
      const uint id = static_cast<const Shader *>(node)->id;
      if (id < shader_count) {
        shader_objects[id].push_back(object->get_device_index());
      }
    }
  }
  std::vector<uint> missing_attributes;
  std::vector<uint64_t> attribute_slots;
  std::vector<uint> attribute_slot_types;
  std::vector<uint2> attribute_slot_words;
  uint2 last_node = make_uint2(0, 0);
  bool uses_ies = false;

  std::vector<bool> checked(shader_count, false);
  std::vector<bool> uv_shaders(shader_count, false);
  std::vector<bool> generated_shaders(shader_count, false);
  bool have_uv = false;
  bool have_generated = false;
  bool have_ggx = false;
  bool have_transmission = false;
  bool have_subsurface = false;
  bool have_standalone = false;
  bool have_hair = false;
  bool have_portal = false;
  bool have_mixed_transparent = false;
  bool pixel_wireframe = false;
  bool have_sheen = false;
  bool uses_sheen_ltc = false;
  const WebGPUSvmColorSpace color_space{make_float3(dscene.data.film.rgb_to_y),
                                        make_float3(dscene.data.film.xyz_to_r),
                                        make_float3(dscene.data.film.xyz_to_g),
                                        make_float3(dscene.data.film.xyz_to_b)};
  for (size_t i = 0; i < dscene.tri_shader.size(); ++i) {
    const uint shader = dscene.tri_shader.data()[i] & SHADER_MASK;
    if (shader >= shader_count) {
      reason = "WebGPU Cycles found an invalid SVM material";
      return false;
    }
    if (checked[shader] ||
        std::ranges::find(volume_shaders, shader) != volume_shaders.end()) {
      continue;
    }

    const Shader &material = *scene.shaders[shader];
    last_node = make_uint2(0, 0);
    WebGPUSvmMaterial decoded;
    bool uses_shadow_path_context = false;
    shader_uses_cubic = false;
    bool uses_uv = false;
    bool uses_generated = false;
    /* GeometryManager has already evaluated true displacement on the host and
     * synchronized the displaced BVH. Bump-only materials need surface bump
     * evaluation only if the finalized graph still has a bump output. */
    const auto displacement_method = material.get_displacement_method();
    const bool unsupported_displacement =
        displacement_method != DISPLACE_TRUE &&
        (displacement_method == DISPLACE_BUMP ?
             material.has_bump_from_displacement && material.graph->output()->input("Normal")->link :
             material.has_displacement || material.has_bump_from_displacement);
    if (material.id != shader || material.has_volume_connected || material.has_volume ||
        unsupported_displacement || material.has_bump_from_surface ||
        !webgpu_svm_eval_material(fetch,
                                 image,
                                 WebGPUSvmUv{0.25f, 0.25f},
                                 make_float3(0.25f, 0.25f, 0.25f),
                                 color_space,
                                 uses_uv,
                                 uses_generated,
                                 uint(count),
                                 uint(shader_count),
                                 shader,
                                 decoded,
                                 WebGPUSvmNoSurface{},
                                 WebGPUSvmSceneHost{{}, &dscene, &shader_objects[shader],
                                                    &missing_attributes, &attribute_slots,
                                                    &attribute_slot_types, &attribute_slot_words,
                                                    &last_node, &uses_ies,
                                                    &uses_shadow_path_context}))
    {
      reason = "WebGPU Cycles needs supported Lambert, Principled or Glass SVM materials (" +
               material.name.string() + " stops at SVM node " + std::to_string(last_node.x) +
               ", word " + std::to_string(last_node.y) + ")";
      return false;
    }
    checked[shader] = true;
    uv_shaders[shader] = uses_uv;
    generated_shaders[shader] = uses_generated;
    have_uv |= uses_uv;
    have_generated |= uses_generated;
    const bool huang_hair = decoded.surface_closure == CYCLES_WEBGPU_MATERIAL_HAIR_HUANG;
    have_ggx |= decoded.ggx || huang_hair || decoded.surface_closure == CYCLES_WEBGPU_MATERIAL_GLOSSY;
    have_transmission |= huang_hair || (decoded.ggx && decoded.transmission > 0.0f);
    have_subsurface |= decoded.subsurface > 0.0f;
    requires_extended_svm |= decoded.requires_extended_svm;
    requires_advanced_procedural_svm |= decoded.requires_advanced_procedural_svm;
    pixel_wireframe |= decoded.pixel_wireframe;
    uses_sheen_ltc |= decoded.surface_closure == CYCLES_WEBGPU_MATERIAL_SHEEN;
    have_sheen |= decoded.surface_closure == CYCLES_WEBGPU_MATERIAL_SHEEN ||
                  decoded.surface_closure == CYCLES_WEBGPU_MATERIAL_VELVET;
    const bool portal = decoded.surface_closure == CYCLES_WEBGPU_MATERIAL_RAY_PORTAL;
    have_portal |= portal;
    const bool mixed = decoded.mixed_transparency;
    have_mixed_transparent |= mixed;
    if (mixed && (dscene.data.integrator.filter_closures & FILTER_CLOSURE_TRANSPARENT)) {
      reason = "WebGPU Cycles needs unfiltered Transparent closures for mixed materials";
      return false;
    }
    const bool shader_has_transparent_shadows =
        (dscene.shaders.data()[shader].flags & SD_HAS_TRANSPARENT_SHADOW) != 0;
    const bool varying_hair_shadows = huang_hair && shader_has_transparent_shadows;
    if (portal || mixed || varying_hair_shadows ||
        decoded.surface_closure == CYCLES_WEBGPU_MATERIAL_TRANSPARENT)
    {
      if (transparent_shadows.empty()) {
        if (shader_count > remaining / 3) {
          reason = "WebGPU Cycles transparent shadow colors exceed the scene budget";
          return false;
        }
        transparent_shadows.resize(3 * shader_count);
      }
      const bool evaluate_shadow = decoded.varying_transparency || varying_hair_shadows;
      varying_transparency |= evaluate_shadow;
      /* Stock shade_shadow treats portals as opaque even though main rays teleport. */
      if (!portal && shader_has_transparent_shadows) {
        if (evaluate_shadow) {
          if (uses_shadow_path_context) {
            reason = "WebGPU Cycles transparent shadow material needs unsupported ray context";
            return false;
          }
          if (decoded.requires_full_svm || shader_uses_cubic) {
            reason = "WebGPU Cycles varying transparent shadows need supported texture projections and filters";
            return false;
          }
        }
        const Spectrum tint = mixed ? decoded.transparency : decoded.color;
        transparent_shadows[3 * shader] = evaluate_shadow ?
                                               CYCLES_WEBGPU_TRANSPARENT_SHADOW_EVAL :
                                               std::bit_cast<uint32_t>(tint.x);
        transparent_shadows[3 * shader + 1] = std::bit_cast<uint32_t>(tint.y);
        transparent_shadows[3 * shader + 2] = std::bit_cast<uint32_t>(tint.z);
      }
    }
    have_hair |= CYCLES_WEBGPU_MATERIAL_IS_HAIR(decoded.surface_closure);
    if (CYCLES_WEBGPU_MATERIAL_IS_STANDALONE(decoded.surface_closure)) {
      have_standalone = true;
      const auto &integrator = dscene.data.integrator;
      if (((decoded.surface_closure == CYCLES_WEBGPU_MATERIAL_GLOSSY_TOON ||
            decoded.surface_closure == CYCLES_WEBGPU_MATERIAL_GLOSSY) &&
           (!integrator.caustics_reflective ||
            (integrator.filter_closures & FILTER_CLOSURE_GLOSSY))) ||
          (decoded.surface_closure == CYCLES_WEBGPU_MATERIAL_TRANSLUCENT &&
           (integrator.filter_closures & FILTER_CLOSURE_TRANSMISSION)))
      {
        reason = "WebGPU Cycles needs unfiltered standalone BSDFs";
        return false;
      }
    }
    has_aov |= material.has_aov_output_node;
  }

  if (have_standalone && std::ranges::any_of(
                            std::span(dscene.objects.data(), dscene.objects.size()),
                            [](const KernelObject &object) {
                              return object.primitive_type != PRIMITIVE_LAMP &&
                                     object.shadow_terminator_shading_offset > 1.0f;
                            }))
  {
    reason = "WebGPU Cycles standalone BSDFs need the default shadow terminator shading offset";
    return false;
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
  const size_t attribute_triangles = attribute_slots.empty() ? 0 : dscene.tri_shader.size();
  if (attribute_triangles &&
      attribute_triangles > remaining / (attribute_slots.size() * webgpu_svm_attribute_words))
  {
    reason = "WebGPU Cycles attributes exceed the scene budget";
    return false;
  }
  remaining -= attribute_slots.size() * attribute_triangles * webgpu_svm_attribute_words;
  const size_t ies_words = uses_ies ? dscene.ies_lights.size() : 0;
  if (ies_words > remaining) {
    reason = "WebGPU Cycles IES data exceeds the scene budget";
    return false;
  }
  remaining -= ies_words;
  static_assert(CYCLES_WEBGPU_SHEEN_LTC_WORDS == 3 * SHEEN_RES_MU * SHEEN_RES_ROUGH);
  const size_t lookup_words = have_sheen ? webgpu_svm_lookup_words : 0;
  const size_t sheen_words = uses_sheen_ltc ? CYCLES_WEBGPU_SHEEN_LTC_WORDS : 0;
  const int sheen_offset = dscene.data.tables.sheen_ltc;
  if (lookup_words > remaining || sheen_words > remaining - lookup_words ||
      (sheen_words && (sheen_offset < 0 ||
       size_t(sheen_offset) > dscene.lookup_table.size() ||
       sheen_words > dscene.lookup_table.size() - size_t(sheen_offset))))
  {
    reason = "WebGPU Cycles needs the synchronized Sheen lookup table within the scene budget";
    return false;
  }
  remaining -= lookup_words + sheen_words;
  const bool queries = (dscene.data.kernel_features & KERNEL_FEATURE_NODE_RAYTRACE) != 0;
  std::vector<uint32_t> packed(max_words - remaining);
  packed[0] = uint32_t(count);
  packed[1] = uint32_t(shader_count) | (requires_extended_svm ? CYCLES_WEBGPU_SVM_EXTENDED : 0);
  if (have_standalone) {
    packed[1] |= CYCLES_WEBGPU_SVM_STANDALONE;
  }
  if (have_hair) {
    packed[1] |= CYCLES_WEBGPU_SVM_HAIR;
  }
  if (have_sheen) {
    packed[1] |= CYCLES_WEBGPU_SVM_SHEEN;
  }
  if (has_aov) {
    packed[1] |= CYCLES_WEBGPU_SVM_AOV;
  }
  if (have_portal) {
    packed[1] |= CYCLES_WEBGPU_SVM_RAY_PORTAL;
  }
  if (have_mixed_transparent) {
    packed[1] |= CYCLES_WEBGPU_SVM_MIXED_TRANSPARENT;
  }
  if (pixel_wireframe) {
    packed[1] |= CYCLES_WEBGPU_SVM_PIXEL_WIREFRAME;
  }
  if (dscene.data.kernel_features & KERNEL_FEATURE_NODE_RAYTRACE) {
    packed[1] |= CYCLES_WEBGPU_SVM_QUERIES;
  }
  for (size_t i = 0; i < count; ++i) {
    packed[webgpu_svm_header_words + i] = uint32_t(dscene.svm_nodes.data()[i]);
  }
  for (const uint word : missing_attributes) {
    packed[webgpu_svm_header_words + word] = uint32_t(int(ATTR_STD_NOT_FOUND));
  }
  for (const uint2 slot : attribute_slot_words) {
    packed[webgpu_svm_header_words + slot.x] = webgpu_svm_attribute_slot | slot.y;
  }
  const size_t texture = webgpu_svm_header_words + count;
  packed[texture] = uint32_t(triangles);
  packed[texture + 1] = uint32_t(image_slots);
  packed[texture + 2] = uint32_t(generated_triangles);
  packed[texture + 3] = uint32_t(attribute_slots.size());
  packed[texture + 4] = uint32_t(attribute_triangles);
  const std::array color_rows{
      color_space.rgb_to_y, color_space.xyz_to_r, color_space.xyz_to_g, color_space.xyz_to_b};
  for (size_t row = 0; row < color_rows.size(); ++row) {
    const size_t offset = texture + webgpu_svm_color_space_offset + 3 * row;
    packed[offset] = std::bit_cast<uint32_t>(color_rows[row].x);
    packed[offset + 1] = std::bit_cast<uint32_t>(color_rows[row].y);
    packed[offset + 2] = std::bit_cast<uint32_t>(color_rows[row].z);
  }
  packed[texture + webgpu_svm_indirect_clamp_offset] = std::bit_cast<uint32_t>(
      dscene.data.integrator.sample_clamp_indirect);
  for (uint i = 0; i < webgpu_svm_cie_table_words; ++i) {
    packed[texture + webgpu_svm_cie_table_offset + i] = std::bit_cast<uint32_t>(
        cie_color_match[i / 3][i % 3]);
  }
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
  const size_t attributes = generated + generated_triangles * webgpu_svm_triangle_generated_words;
  if (!attribute_slots.empty() &&
      !cycles_webgpu_svm_attribute_records(scene, attribute_slots, packed.data() + attributes, reason))
  {
    return false;
  }
  const size_t image_table = attributes +
                             attribute_slots.size() * attribute_triangles * webgpu_svm_attribute_words;
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
  if (ies_words) {
    packed[texture + webgpu_svm_ies_offset] = uint32_t(texel - texture);
    for (size_t i = 0; i < ies_words; ++i) {
      packed[texel++] = std::bit_cast<uint32_t>(dscene.ies_lights.data()[i]);
    }
  }
  if (have_sheen) {
    packed[texel + webgpu_svm_lookup_ies] = packed[texture + webgpu_svm_ies_offset];
    packed[texel + webgpu_svm_lookup_sheen] = sheen_words ?
                                                uint32_t(texel + lookup_words - texture) : 0;
    packed[texture + webgpu_svm_ies_offset] = uint32_t(texel - texture);
    texel += lookup_words;
    for (size_t i = 0; i < sheen_words; ++i) {
      packed[texel++] = std::bit_cast<uint32_t>(dscene.lookup_table.data()[size_t(sheen_offset) + i]);
    }
  }

  words = std::move(packed);
  has_ggx = have_ggx;
  has_transmission = have_transmission;
  has_subsurface = have_subsurface;
  advanced_query_svm = queries && requires_advanced_procedural_svm;
  reason.clear();
  return true;
}

CCL_NAMESPACE_END
