/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

/* The last word of each live wavefront ray carries the camera's RNG seed.
 * It belongs to the path and is preserved as the ray advances. */
#define CYCLES_WEBGPU_RAY_RNG_PIXEL 11u

/* One dense wavefront slot. Keep these word offsets in sync with path-state.wgsl.
 * Floating-point components are stored as raw IEEE 754 words. */
#define CYCLES_WEBGPU_PATH_THROUGHPUT 0u
#define CYCLES_WEBGPU_PATH_RADIANCE 3u
#define CYCLES_WEBGPU_PATH_QUEUED_KERNEL 7u
#define CYCLES_WEBGPU_PATH_BOUNCE 8u
#define CYCLES_WEBGPU_PATH_DIFFUSE_BOUNCE 9u
#define CYCLES_WEBGPU_PATH_GLOSSY_BOUNCE 10u
#define CYCLES_WEBGPU_PATH_TRANSMISSION_BOUNCE 11u
#define CYCLES_WEBGPU_PATH_MIS_RAY_PDF 12u
#define CYCLES_WEBGPU_PATH_MIN_RAY_PDF 13u
#define CYCLES_WEBGPU_PATH_FLAGS 14u
#define CYCLES_WEBGPU_PATH_MIS_ORIGIN_N 15u
#define CYCLES_WEBGPU_PATH_PIXEL 18u
/* Positive values divide throughput by the continuation probability at a
 * surface. Zero rejects the surface but can still hit an area light or world. */
#define CYCLES_WEBGPU_PATH_SURFACE_RR 21u
#define CYCLES_WEBGPU_PATH_TRANSPARENT_BOUNCE 22u
#define CYCLES_WEBGPU_PATH_WORDS 23u

#define CYCLES_WEBGPU_PATH_MIS_HAD_TRANSMISSION (1u << 6u)
#define CYCLES_WEBGPU_PATH_MIS_SKIP (1u << 7u)
#define CYCLES_WEBGPU_PATH_TRANSPARENT_BACKGROUND (1u << 9u)
#define CYCLES_WEBGPU_PATH_TERMINATE_AFTER_TRANSPARENT (1u << 12u)

/* Zero marks a terminated path. Live queues use the upstream DeviceKernel + 1. */
#define CYCLES_WEBGPU_PATH_TERMINATED 0u
#define CYCLES_WEBGPU_PATH_INTERSECT_CLOSEST 3u
