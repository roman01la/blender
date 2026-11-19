/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "util/defines.h"

#include <cstdint>
#include <functional>

namespace wgpu {
class Device;
}

CCL_NAMESPACE_BEGIN

class Device;
class device_memory;

struct WebGPUDisplayPreview {
  uint32_t ticket, width, height, origin, stride, pass_stride, combined, sample_count, samples;
  float exposure;
};

bool cycles_webgpu_requested();
bool cycles_webgpu_available();
/** Call on a render pthread. The callback runs on the compute owner. */
bool cycles_webgpu_on_compute_device(const std::function<void(const wgpu::Device &)> &operation);
bool cycles_webgpu_publish_preview(Device &device,
                                   const device_memory &buffer,
                                   const WebGPUDisplayPreview &preview);

CCL_NAMESPACE_END
