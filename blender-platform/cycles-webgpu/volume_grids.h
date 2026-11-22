/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "device/webgpu/svm_materials.h"
#include "device/webgpu/volume_grid_layout.h"
#include "device/webgpu/volume_layout.h"
#include "scene/geometry.h"
#include "scene/image.h"
#include "scene/object.h"
#include "util/log.h"

#include <bit>
#include <climits>
#include <cmath>
#include <cstring>

#if defined(WITH_OPENVDB) && defined(WITH_NANOVDB)
#  include "util/nanovdb.h"
#endif

CCL_NAMESPACE_BEGIN

#if defined(WITH_OPENVDB) && defined(WITH_NANOVDB)

/* Unpack the bounded part of a synchronized NanoVDB grid. The shader keeps the
 * stock filtering and addresses the payload with 32-bit storage indices. */
template<typename BuildT>
bool webgpu_pack_nanovdb(device_image &image,
                         const uint descriptor,
                         const size_t word_budget,
                         std::vector<uint32_t> &packed,
                         float &maximum)
{
  using Grid = ::nanovdb::NanoGrid<BuildT>;
  using Value = typename Grid::ValueType;
  constexpr uint channels =
      std::is_same_v<Value, float> ? 1 : std::is_same_v<Value, ::nanovdb::Vec3f> ? 3 : 4;

  const size_t bytes = image.memory_size();
  if (!image.host_pointer || bytes < sizeof(Grid)) {
    LOG_INFO << "WebGPU Cycles NanoVDB source has " << bytes
             << " bytes, expected at least " << sizeof(Grid)
             << ", host data " << (image.host_pointer ? "present" : "missing");
    return false;
  }
  /* Use the same non-owning buffer view as Cycles' volume mesh builder. */
  ::nanovdb::GridHandle handle(::nanovdb::HostBuffer::createFull(bytes, image.host_pointer));
  const Grid *source = handle.template grid<BuildT>(0);
  if (!source || source->gridSize() > bytes || source->isEmpty()) {
    LOG_INFO << "WebGPU Cycles could not read NanoVDB format " << image.info.data_type
             << " from " << bytes << " bytes";
    if (source) {
      LOG_INFO << "WebGPU Cycles NanoVDB declares " << source->gridSize()
               << " bytes, empty " << source->isEmpty();
    }
    return false;
  }
  const Grid &grid = *source;

  auto bbox = grid.tree().bbox();
  if (bbox.empty()) {
    /* Cycles can omit the NanoVDB bounding-box statistics. Recover the active
     * bounds using the same topology mask as the stock volume mesh builder. */
    const auto mask = nanovdb_to_openvdb_mask(handle);
    if (!mask) {
      return false;
    }
    const auto active = mask->evalActiveVoxelBoundingBox();
    if (active.empty()) {
      return false;
    }
    bbox = ::nanovdb::CoordBBox({active.min().x(), active.min().y(), active.min().z()},
                               {active.max().x(), active.max().y(), active.max().z()});
  }
  uint dimensions[3];
  int origin[3];
  size_t words = channels;
  const size_t remaining = word_budget - packed.size();
  for (int axis = 0; axis < 3; ++axis) {
    /* Include the inactive values that the stock cubic filter can reach. */
    const int64_t min = int64_t(bbox.min()[axis]) - 2;
    const int64_t max = int64_t(bbox.max()[axis]) + 2;
    const int64_t size = max - min + 1;
    if (min < INT_MIN || max > INT_MAX || size <= 0 || size > INT_MAX ||
        size_t(size) > remaining / words)
    {
      LOG_INFO << "WebGPU Cycles NanoVDB axis " << axis << " spans " << min << " to " << max
               << " with " << remaining << " storage words available at stride " << words;
      return false;
    }
    origin[axis] = int(min);
    dimensions[axis] = uint(size);
    packed[descriptor + CYCLES_WEBGPU_GRID_ORIGIN + axis] = uint(origin[axis]);
    packed[descriptor + CYCLES_WEBGPU_GRID_SIZE + axis] = dimensions[axis];
    words *= size_t(size);
  }

  packed[descriptor + CYCLES_WEBGPU_GRID_CHANNELS] = channels;
  packed[descriptor + CYCLES_WEBGPU_GRID_VOXELS] = uint(packed.size());
  packed.reserve(packed.size() + words);
  bool finite = true;
  maximum = 0.0f;
  const auto encode = [&](const Value &value, uint channel) {
    float result;
    if constexpr (channels == 1) {
      result = value;
    }
    else {
      result = value[channel];
    }
    finite &= std::isfinite(result);
    maximum = std::max(maximum, result);
    return std::bit_cast<uint32_t>(result);
  };
  const Value &background = grid.tree().background();
  for (uint channel = 0; channel < channels; ++channel) {
    packed[descriptor + CYCLES_WEBGPU_GRID_BACKGROUND + channel] = encode(background, channel);
  }
  auto accessor = grid.getAccessor();
  for (uint z = 0; z < dimensions[2]; ++z) {
    for (uint y = 0; y < dimensions[1]; ++y) {
      for (uint x = 0; x < dimensions[0]; ++x) {
        const Value value =
            accessor.getValue({origin[0] + int(x), origin[1] + int(y), origin[2] + int(z)});
        for (uint channel = 0; channel < channels; ++channel) {
          packed.push_back(encode(value, channel));
        }
      }
    }
  }
  if (!finite) {
    LOG_INFO << "WebGPU Cycles NanoVDB contains a nonfinite voxel value";
  }
  return finite;
}

#endif

inline const char *webgpu_pack_volume_grids(const Scene &scene,
                                            const Object &object,
                                            const std::span<const uint32_t> record,
                                            const size_t word_budget,
                                            std::vector<uint32_t> &packed)
{
#if defined(WITH_OPENVDB) && defined(WITH_NANOVDB)
  constexpr uint header_words = CYCLES_WEBGPU_GRID_ATTRIBUTES +
                                CYCLES_WEBGPU_GRID_ATTRIBUTE_COUNT *
                                    CYCLES_WEBGPU_GRID_ATTRIBUTE_WORDS;
  if (record.size() < CYCLES_WEBGPU_VOLUME_DATA + sizeof(SVMNodePrincipledVolume) / sizeof(uint) ||
      record[CYCLES_WEBGPU_VOLUME_OPCODE] != NODE_PRINCIPLED_VOLUME)
  {
    return "WebGPU Cycles needs a packed Principled Volume grid material";
  }
  if (word_budget < header_words || word_budget > UINT32_MAX) {
    return "WebGPU Cycles lacks storage for the volume grid header";
  }
  SVMNodePrincipledVolume node;
  std::memcpy(&node, record.data() + CYCLES_WEBGPU_VOLUME_DATA, sizeof(node));
  const auto &dscene = scene.dscene;
  const int index = object.get_device_index();
  const Geometry &geometry = *object.get_geometry();
  if (dscene.data.integrator.volume_ray_marching) {
    return "WebGPU Cycles grid volumes require the stock unbiased volume integrator";
  }
  if (index < 0 || size_t(index) >= dscene.objects.size()) {
    return "WebGPU Cycles could not find the synchronized volume grid object";
  }
  if (dscene.object_flag.data()[index] & SD_OBJECT_HAS_VOLUME_MOTION) {
    return "WebGPU Cycles grid volumes require static voxel motion";
  }
  if (size_t(index) >= dscene.volume_tree_root_ids.size())
  {
    return "WebGPU Cycles could not find the synchronized volume grid root list";
  }
  const int root = dscene.volume_tree_root_ids.data()[index];
  if (root < 0 || size_t(root) >= dscene.volume_tree_roots.size()) {
    return "WebGPU Cycles could not find the synchronized volume grid octree root";
  }
  const int octree = dscene.volume_tree_roots.data()[root].id;
  if (octree < 0 || size_t(octree) >= dscene.volume_tree_nodes.size()) {
    return "WebGPU Cycles could not find the synchronized volume grid octree majorant";
  }

  std::vector<uint32_t> payload(header_words, 0);
  const int ids[] = {node.attr_density, node.attr_color, node.attr_temperature};
  float maxima[] = {1.0f, 1.0f, 1.0f};
  for (uint slot = 0; slot < CYCLES_WEBGPU_GRID_ATTRIBUTE_COUNT; ++slot) {
    AttributeMap desc{};
    const int id = ids[slot];
    if (id == ATTR_STD_NONE || id == ATTR_STD_NOT_FOUND) {
      continue;
    }
    const auto lookup = cycles_webgpu_svm_find_attribute(dscene, index, uint64_t(id), desc);
    if (lookup == WebGPUSvmAttributeLookup::Missing) {
      continue;
    }
    if (lookup != WebGPUSvmAttributeLookup::Found || desc.element != ATTR_ELEMENT_VOXEL ||
        desc.offset < 0 || size_t(desc.offset) >= dscene.image_textures.size())
    {
      return "WebGPU Cycles could not resolve a named volume grid texture";
    }

    const KernelImageTexture &texture = dscene.image_textures.data()[desc.offset];
    device_image *image = nullptr;
    for (const Attribute &attribute : geometry.attributes.attributes) {
      if (attribute.element == ATTR_ELEMENT_VOXEL &&
          attribute.data_voxel().kernel_id() == int(desc.offset))
      {
        image = attribute.data_voxel().vdb_image_memory();
        break;
      }
    }
    if (image == nullptr) {
      return "WebGPU Cycles could not find host data for a named NanoVDB grid";
    }
    const uint descriptor =
        CYCLES_WEBGPU_GRID_ATTRIBUTES + slot * CYCLES_WEBGPU_GRID_ATTRIBUTE_WORDS;
    payload[descriptor + CYCLES_WEBGPU_GRID_ATTRIBUTE_ID] = uint(id);
    const uint interpolation = (record[CYCLES_WEBGPU_VOLUME_SHADER_FLAGS] & SD_VOLUME_CUBIC) ?
                                   INTERPOLATION_CUBIC :
                                   image->info.interpolation;
    if (interpolation != INTERPOLATION_CLOSEST && interpolation != INTERPOLATION_LINEAR &&
        interpolation != INTERPOLATION_CUBIC)
    {
      return "WebGPU Cycles needs nearest, linear, or cubic volume grid interpolation";
    }
    payload[descriptor + CYCLES_WEBGPU_GRID_INTERPOLATION] = interpolation;
    const Transform &inverse = dscene.objects.data()[index].itfm;
    /* primitive_volume_attribute always applies the inverse object transform. */
    const Transform world_to_index = texture.use_transform_3d ?
                                         texture.transform_3d * inverse :
                                         inverse;
    uint row = descriptor + CYCLES_WEBGPU_GRID_TRANSFORM;
    for (const float4 vector : {world_to_index.x, world_to_index.y, world_to_index.z}) {
      for (const float value : {vector.x, vector.y, vector.z, vector.w}) {
        if (!std::isfinite(value)) {
          return "WebGPU Cycles needs a finite transform for each volume grid";
        }
        payload[row++] = std::bit_cast<uint32_t>(value);
      }
    }
    float &maximum = maxima[slot];
    bool valid;
    switch (image->info.data_type) {
      case IMAGE_DATA_TYPE_NANOVDB_EMPTY:
        payload[descriptor + CYCLES_WEBGPU_GRID_CHANNELS] = 1;
        maximum = 0.0f;
        valid = true;
        break;
      case IMAGE_DATA_TYPE_NANOVDB_FLOAT:
        valid = webgpu_pack_nanovdb<float>(*image, descriptor, word_budget, payload, maximum);
        break;
      case IMAGE_DATA_TYPE_NANOVDB_FPN:
        valid = webgpu_pack_nanovdb<::nanovdb::FpN>(
            *image, descriptor, word_budget, payload, maximum);
        break;
      case IMAGE_DATA_TYPE_NANOVDB_FP16:
        valid = webgpu_pack_nanovdb<::nanovdb::Fp16>(
            *image, descriptor, word_budget, payload, maximum);
        break;
      case IMAGE_DATA_TYPE_NANOVDB_FLOAT3:
        valid = webgpu_pack_nanovdb<::nanovdb::Vec3f>(
            *image, descriptor, word_budget, payload, maximum);
        break;
      case IMAGE_DATA_TYPE_NANOVDB_FLOAT4:
        valid = webgpu_pack_nanovdb<::nanovdb::Vec4f>(
            *image, descriptor, word_budget, payload, maximum);
        break;
      default:
        return "WebGPU Cycles needs a float or vector NanoVDB volume grid";
    }
    if (!valid) {
      return "WebGPU Cycles could not expand the NanoVDB voxels within the upload budget";
    }
  }

  /* Smooth nonnegative filter weights cannot increase the voxel maxima.
   * Bound extinction conservatively alongside Cycles' baked octree maximum. */
  const float density_scale = std::bit_cast<float>(record[CYCLES_WEBGPU_ABSORPTION_OBJECT_DENSITY]);
  const float color = std::max({std::bit_cast<float>(record[CYCLES_WEBGPU_VOLUME_WEIGHT]),
                                std::bit_cast<float>(record[CYCLES_WEBGPU_VOLUME_WEIGHT + 1]),
                                std::bit_cast<float>(record[CYCLES_WEBGPU_VOLUME_WEIGHT + 2])});
  const float extinction = density_scale * fmaxf(0.0f, std::bit_cast<float>(node.density.bits)) *
                           maxima[0] * fmaxf(1.0f, color * maxima[1]);
  const float baked = dscene.volume_tree_nodes.data()[octree].sigma.max * density_scale;
  const float majorant = fmaxf(baked, extinction);
  if (!std::isfinite(majorant) || majorant < 0.0f) {
    return "WebGPU Cycles needs a finite extinction majorant for each volume grid";
  }
  payload[CYCLES_WEBGPU_GRID_MAJORANT] = std::bit_cast<uint32_t>(majorant);
  packed = std::move(payload);
  return nullptr;
#else
  (void)scene;
  (void)object;
  (void)record;
  (void)word_budget;
  (void)packed;
  return "WebGPU Cycles grid volumes require OpenVDB and NanoVDB support";
#endif
}

CCL_NAMESPACE_END
