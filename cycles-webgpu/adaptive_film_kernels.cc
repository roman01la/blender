/* SPDX-License-Identifier: Apache-2.0 */

#ifdef __aarch64__
#  include <arm_neon.h>
#endif

#define CCL_NAMESPACE_BEGIN namespace ccl {
#define CCL_NAMESPACE_END }
#include "kernel/types.h"

#define __KERNEL_GPU__
#include "kernel/globals.h"
#undef __KERNEL_GPU__

namespace ccl {
struct FilmGlobals {
  KernelData data{};
  const float *sample_pattern_lut = nullptr;
};
using KernelGlobals = const FilmGlobals *;
}

#define kernel_data (kg->data)
#define kernel_data_fetch(name, index) (kg->name[index])
#undef ccl_device
#define ccl_device static inline __attribute__((always_inline))
#include "kernel/film/light_passes.h"

using Float4 = float __attribute__((ext_vector_type(4)));

void cycles_adaptive_film_delta(const unsigned int sample,
                               const unsigned int pattern,
                               const Float4 rgba,
                               Float4 &combined,
                               Float4 &adaptive)
{
  using ccl::PASS_COMBINED;
  ccl::FilmGlobals globals;
  globals.data.film.light_pass_flag = PASSMASK(COMBINED);
  globals.data.film.pass_combined = 0;
  globals.data.film.pass_adaptive_aux_buffer = 4;
  globals.data.film.pass_volume_scatter = PASS_UNUSED;
  globals.data.film.pass_volume_transmit = PASS_UNUSED;
  globals.data.integrator.sampling_pattern = pattern;
  /* Fold the already-clamped contributions of this path through stock Cycles' pass writers.
   * The dispatch publishes these local deltas with WebGPU-compatible atomics. */
  float pixel[8] = {};
  ccl::film_write_combined_transparent_pass(
      &globals, ccl::PATH_RAY_VISIBILITY_CAMERA, 0, int(sample),
      ccl::make_float3(rgba.x, rgba.y, rgba.z), 1.0f - rgba.w, pixel);
  combined = {pixel[0], pixel[1], pixel[2], pixel[3]};
  adaptive = {pixel[4], pixel[5], pixel[6], pixel[7]};
}
