/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "device/kernel.h"
#include "device/memory.h"
#include "device/webgpu/path_state_layout.h"

#include <algorithm>
#include <limits>

CCL_NAMESPACE_BEGIN

static_assert(CYCLES_WEBGPU_PATH_INTERSECT_CLOSEST ==
              DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST + 1);

/* Device-local state retained between wavefront stages. Ray and intersection
 * records are separate bindings using the existing traversal ABI. */
class WebGPUPathStateStorage {
 public:
  explicit WebGPUPathStateStorage(Device &device) : words_(&device, "WebGPU path states") {}

  /* Camera admission initializes the chosen free slots before traversal. */
  bool alloc(const uint32_t capacity)
  {
    constexpr uint32_t max_capacity = std::min<uint64_t>(
        std::numeric_limits<uint32_t>::max() / CYCLES_WEBGPU_PATH_WORDS,
        std::numeric_limits<size_t>::max() / bytes_per_path);
    if (!capacity || capacity > max_capacity) {
      return false;
    }

    words_.alloc_to_device(size_t(capacity) * CYCLES_WEBGPU_PATH_WORDS, true);
    if (!words_.device_pointer) {
      words_.free();
      return false;
    }
    return true;
  }

  size_t capacity() const
  {
    return words_.data_size / CYCLES_WEBGPU_PATH_WORDS;
  }

  device_only_memory<uint32_t> &words()
  {
    return words_;
  }

 private:
  static constexpr size_t bytes_per_path = CYCLES_WEBGPU_PATH_WORDS * sizeof(uint32_t);

  device_only_memory<uint32_t> words_;
};

CCL_NAMESPACE_END
