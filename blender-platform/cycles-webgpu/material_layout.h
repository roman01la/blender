/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

/* Six float4 records per path: base color and status, specular tint and
 * squared roughness, metallic/effective IOR/dielectric F0/mix weight, then
 * transmission weight, original IOR, the standalone transmission closure and
 * subsurface weight, subsurface RGB radius and encoded anisotropy, and the
 * surface emission with the indirect clamp. Subsurface stores its entry IOR
 * and BSSRDF type in the two Glass slots. */
#define CYCLES_WEBGPU_MATERIAL_VECTORS 6
#define CYCLES_WEBGPU_MATERIAL_WORDS (4 * CYCLES_WEBGPU_MATERIAL_VECTORS)
#define CYCLES_WEBGPU_GLASS_GGX 1.0f
#define CYCLES_WEBGPU_GLASS_BECKMANN 2.0f
#define CYCLES_WEBGPU_REFRACTION_GGX 3.0f
#define CYCLES_WEBGPU_REFRACTION_BECKMANN 4.0f
#define CYCLES_WEBGPU_MATERIAL_GGX 3.0f
#define CYCLES_WEBGPU_MATERIAL_MULTISCATTER_GGX 4.0f
/* Holdout weight in RGB. Camera paths write its transparency, and the path
 * ends without scattering, emission or light sampling. */
#define CYCLES_WEBGPU_MATERIAL_HOLDOUT 5.0f
/* The shader count word also selects the extended SVM interpreter. */
#define CYCLES_WEBGPU_SVM_EXTENDED 0x80000000u
#define CYCLES_WEBGPU_SVM_SHADER_COUNT 0x7fffffffu
#define CYCLES_WEBGPU_SVM_SHADER_ID_MASK 0x203fffffu
/* Packed NODE_ATTR records reference an attribute slot with this flag. Each
 * slot stores a mode and type word and three float4 corners per triangle. */
#define CYCLES_WEBGPU_SVM_ATTRIBUTE_SLOT 0x40000000u
#define CYCLES_WEBGPU_SVM_ATTRIBUTE_WORDS 13u

/* Read-only SVM inputs from existing wavefront surface and path slots. */
#define CYCLES_WEBGPU_SVM_POSITION 0u
#define CYCLES_WEBGPU_SVM_NORMAL 1u
#define CYCLES_WEBGPU_SVM_GEOMETRIC 2u
#define CYCLES_WEBGPU_SVM_DIRECTION 3u
#define CYCLES_WEBGPU_SVM_INTERSECTION 4u
#define CYCLES_WEBGPU_SVM_BOUNCES 5u
#define CYCLES_WEBGPU_SVM_PATH 6u
#define CYCLES_WEBGPU_SVM_WORLD_TO_CAMERA 7u
#define CYCLES_WEBGPU_SVM_WORLD_TO_NDC 10u
#define CYCLES_WEBGPU_SVM_WORLD_TO_NDC_END 14u
#define CYCLES_WEBGPU_SVM_OBJECT_TRANSFORM 14u
#define CYCLES_WEBGPU_SVM_OBJECT_INVERSE 17u
#define CYCLES_WEBGPU_SVM_OBJECT_COLOR 20u
#define CYCLES_WEBGPU_SVM_OBJECT_INDEX 21u
#define CYCLES_WEBGPU_SVM_MATERIAL_INDEX 22u
#define CYCLES_WEBGPU_SVM_OBJECT_RANDOM 23u
/* Camera rows 4-6 hold the synchronized camera-to-world transform. */
#define CYCLES_WEBGPU_SVM_CAMERA_TO_WORLD 24u
#define CYCLES_WEBGPU_SVM_CAMERA_TO_WORLD_END 27u
#define CYCLES_WEBGPU_CAMERA_TO_WORLD_OFFSET 16u

/* Additional offsets in the existing surface metadata. The forward transforms
 * are already packed for triangle shading. */
#define CYCLES_WEBGPU_SURFACE_OBJECT_SVM_OFFSET 5u
#define CYCLES_WEBGPU_SURFACE_SHADER_PASS_OFFSET 6u
#define CYCLES_WEBGPU_SURFACE_OFFSETS 7u
#define CYCLES_WEBGPU_SURFACE_OBJECT_WORDS 18u

/* The SVM transforms follow the eight float4 records used by camera rays. */
#define CYCLES_WEBGPU_CAMERA_SVM_OFFSET 32u
#define CYCLES_WEBGPU_CAMERA_WORDS 60u

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
#define CYCLES_WEBGPU_PATH_SETTINGS_WORDS 10
