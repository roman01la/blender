/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

/* A volume grid record contains the stock extinction majorant and the three
 * named Principled input descriptors. All voxel offsets are record-relative. */
#define CYCLES_WEBGPU_GRID_MAJORANT 0u
#define CYCLES_WEBGPU_GRID_ATTRIBUTES 1u
#define CYCLES_WEBGPU_GRID_ATTRIBUTE_COUNT 3u
#define CYCLES_WEBGPU_GRID_ATTRIBUTE_WORDS 26u
#define CYCLES_WEBGPU_GRID_ATTRIBUTE_ID 0u
#define CYCLES_WEBGPU_GRID_VOXELS 1u
#define CYCLES_WEBGPU_GRID_CHANNELS 2u
#define CYCLES_WEBGPU_GRID_INTERPOLATION 3u
#define CYCLES_WEBGPU_GRID_ORIGIN 4u
#define CYCLES_WEBGPU_GRID_SIZE 7u
#define CYCLES_WEBGPU_GRID_TRANSFORM 10u
#define CYCLES_WEBGPU_GRID_BACKGROUND 22u
