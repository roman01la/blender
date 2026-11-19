/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include <cstdint>

CCL_NAMESPACE_BEGIN

/* Cycles' scene-facing bounce limit excludes the primary surface visit. */
inline constexpr uint32_t webgpu_max_wavefront_bounces = 1024;

CCL_NAMESPACE_END
