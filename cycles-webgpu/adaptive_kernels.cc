/* SPDX-License-Identifier: Apache-2.0 */

#define CCL_NAMESPACE_BEGIN namespace ccl {
#define CCL_NAMESPACE_END }
#include "kernel/types.h"

#define __KERNEL_GPU__
#include "kernel/globals.h"
#undef __KERNEL_GPU__

namespace ccl {
struct AdaptiveGlobals {
  struct {
    KernelFilm film;
  } data;
};
using KernelGlobals = const AdaptiveGlobals *;
}  // namespace ccl

#define kernel_data (kg->data)
#undef ccl_global
#define ccl_global __attribute__((address_space(31)))
#undef ccl_device
#define ccl_device static inline __attribute__((always_inline))

#include "kernel/film/adaptive_sampling.h"

/* This symbol marks offset zero of the HLSL storage buffer. The shader compiler
 * lowers stock film pointer accesses in this address space to buffer loads/stores. */
extern ccl_global float cycles_adaptive_film[];

using UInt4 = unsigned int __attribute__((ext_vector_type(4)));

#if defined(CYCLES_WEBGPU_ADAPTIVE_CHECK)
unsigned int cycles_adaptive_check(UInt4 pixel,
                                   UInt4 layout,
                                   unsigned int sample_count_offset,
                                   float exposure,
                                   float threshold)
{
  ccl::AdaptiveGlobals globals;
  globals.data.film.pass_stride = layout.y;
  globals.data.film.pass_combined = layout.z;
  globals.data.film.pass_adaptive_aux_buffer = layout.w;
  globals.data.film.pass_sample_count = sample_count_offset;
  globals.data.film.exposure = exposure;
  const bool converged = ccl::film_adaptive_sampling_convergence_check(
      &globals, cycles_adaptive_film, pixel.x, pixel.y, threshold, pixel.z, pixel.w, layout.x);
  return !converged;
}
#else
void cycles_adaptive_filter(UInt4 line, UInt4 layout)
{
  ccl::AdaptiveGlobals globals;
  globals.data.film.pass_stride = layout.y;
  globals.data.film.pass_adaptive_aux_buffer = layout.w;
#  if defined(CYCLES_WEBGPU_ADAPTIVE_FILTER_X)
  ccl::film_adaptive_sampling_filter_x(
      &globals, cycles_adaptive_film, line.x, 0, line.y, line.w, layout.x);
#  else
  ccl::film_adaptive_sampling_filter_y(
      &globals, cycles_adaptive_film, line.x, 0, line.y, line.w, layout.x);
#  endif
}
#endif
