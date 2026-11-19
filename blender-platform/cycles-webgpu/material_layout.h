/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

/* Four float4 records per path: base color and status, specular tint and
 * squared roughness, metallic/effective IOR/dielectric F0/mix weight, then
 * transmission weight, original IOR, and the standalone Glass BSDF flag. */
#define CYCLES_WEBGPU_MATERIAL_VECTORS 4
#define CYCLES_WEBGPU_MATERIAL_WORDS (4 * CYCLES_WEBGPU_MATERIAL_VECTORS)
#define CYCLES_WEBGPU_MATERIAL_GGX 3.0f
#define CYCLES_WEBGPU_MATERIAL_MULTISCATTER_GGX 4.0f

/* Cycles' scalar GGX tables: IOR Schlick, Schlick, directional energy, mean energy. */
#define CYCLES_WEBGPU_GGX_TABLE_WORDS (32 * 32 * 32)
#define CYCLES_WEBGPU_GGX_ENERGY_WORDS (32 * 32)
#define CYCLES_WEBGPU_GGX_AVERAGE_ENERGY_WORDS 32
#define CYCLES_WEBGPU_GGX_LOOKUP_WORDS \
  (2 * CYCLES_WEBGPU_GGX_TABLE_WORDS + CYCLES_WEBGPU_GGX_ENERGY_WORDS + \
   CYCLES_WEBGPU_GGX_AVERAGE_ENERGY_WORDS)
/* Glass adds directional and mean energy in regular, then inverse IOR order. */
#define CYCLES_WEBGPU_GGX_GLASS_ENERGY_WORDS (32 * 32 * 32)
#define CYCLES_WEBGPU_GGX_GLASS_AVERAGE_ENERGY_WORDS (32 * 32)
#define CYCLES_WEBGPU_GGX_GLASS_LOOKUP_WORDS \
  (2 * (CYCLES_WEBGPU_GGX_GLASS_ENERGY_WORDS + CYCLES_WEBGPU_GGX_GLASS_AVERAGE_ENERGY_WORDS))
#define CYCLES_WEBGPU_LIGHT_HEADER_WORDS 12
#define CYCLES_WEBGPU_PATH_SETTINGS_WORDS 9
