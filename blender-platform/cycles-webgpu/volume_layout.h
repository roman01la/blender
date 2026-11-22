/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

/* Optional surface metadata for homogeneous and grid volume bounds. */
#define CYCLES_WEBGPU_SURFACE_ABSORPTION_OFFSET 8u
#define CYCLES_WEBGPU_SURFACE_VOLUME_COUNT 9u
#define CYCLES_WEBGPU_SURFACE_ABSORPTION_OFFSETS 11u
#define CYCLES_WEBGPU_ABSORPTION_WORDS 6u
#define CYCLES_WEBGPU_ABSORPTION_OBJECT 0u
#define CYCLES_WEBGPU_ABSORPTION_COLOR 1u
#define CYCLES_WEBGPU_ABSORPTION_DENSITY 4u
#define CYCLES_WEBGPU_ABSORPTION_OBJECT_DENSITY 5u

/* The homogeneous scattering record adds a typed SVM volume instruction to the
 * absorption prefix. The stock closure evaluator derives the optical properties. */
#define CYCLES_WEBGPU_VOLUME_INDIRECT_CLAMP 4u
#define CYCLES_WEBGPU_VOLUME_OPCODE 6u
#define CYCLES_WEBGPU_VOLUME_WEIGHT 7u
#define CYCLES_WEBGPU_VOLUME_DATA 10u
#define CYCLES_WEBGPU_VOLUME_NODE_WORDS 20u
#define CYCLES_WEBGPU_VOLUME_MAX_BOUNCES 30u
#define CYCLES_WEBGPU_VOLUME_SHADER_FLAGS 31u
#define CYCLES_WEBGPU_VOLUME_GRID_OFFSET 32u
#define CYCLES_WEBGPU_VOLUME_SCATTER_WORDS 33u
#define CYCLES_WEBGPU_MAX_VOLUME_MEDIA 8u
#define CYCLES_WEBGPU_VOLUME_MEDIA_MASK ((1u << CYCLES_WEBGPU_MAX_VOLUME_MEDIA) - 1u)
#define CYCLES_WEBGPU_VOLUME_BOUNDARY_INCREMENT (1u << CYCLES_WEBGPU_MAX_VOLUME_MEDIA)

/* Per-path storage is arranged in planes so absorption's original stack
 * remains at the start of the buffer. */
#define CYCLES_WEBGPU_VOLUME_STATE_WORDS 7u
#define CYCLES_WEBGPU_VOLUME_NEXT_BOUNDARY 1u
#define CYCLES_WEBGPU_VOLUME_BOUNCE 2u
#define CYCLES_WEBGPU_VOLUME_DIRECT 3u
#define CYCLES_WEBGPU_VOLUME_SHADOW_STACK 6u

/* In the compact absorption-only mode this bit stores inside/outside state.
 * The live stack uses the low bits for its active media. */
#define CYCLES_WEBGPU_ABSORPTION_INSIDE (1u << 31u)
/* Values in CYCLES_WEBGPU_HIT_SPECIAL after CYCLES_WEBGPU_HIT_SUBSURFACE_EXIT. */
#define CYCLES_WEBGPU_ABSORPTION_BOUNDARY 2u
#define CYCLES_WEBGPU_ABSORPTION_TERMINATE 3u
#define CYCLES_WEBGPU_VOLUME_SCATTER_HIT 4u

/* Fourth spare word of the specular normal in the compact surface record. */
#define CYCLES_WEBGPU_SURFACE_VOLUME_HIT 7u
