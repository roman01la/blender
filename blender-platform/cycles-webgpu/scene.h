/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "device/memory.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

CCL_NAMESPACE_BEGIN

class Scene;

/* Immutable shader inputs for an uploaded scene. Keep the packing vectors only until the
 * upload completes. Cycles' original DeviceScene stays available to synchronize and render on
 * the CPU. */
class WebGPUSceneBuffers {
 public:
  explicit WebGPUSceneBuffers(Device &device);

  bool upload(std::span<const uint32_t> bvh_words,
              std::span<const uint32_t> surface_words,
              std::span<const uint32_t> light_words,
              std::span<const uint32_t> svm_words);

  device_only_memory<uint32_t> bvh;
  device_only_memory<uint32_t> surface;
  device_only_memory<uint32_t> light;
  device_only_memory<uint32_t> svm;
};

struct CyclesWebGPUScene {
  std::vector<uint32_t> bvh;
  std::vector<uint32_t> surface;
  std::vector<uint32_t> lighting;
  std::vector<uint32_t> svm;
  uint32_t max_bounces = 0;
  bool has_ggx = false;
  bool has_transmission = false;
  bool has_subsurface = false;
};

/* Read the synchronized host DeviceScene under the Scene mutex. Preserve the previous output
 * if the scene requires a kernel outside the experimental static surface renderer. */
bool cycles_webgpu_encode_scene(const Scene &scene, CyclesWebGPUScene &out, std::string &reason);

CCL_NAMESPACE_END
