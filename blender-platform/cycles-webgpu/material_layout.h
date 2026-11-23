/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include "volume_layout.h"

/* The first six float4 records per path: base color and status, specular tint and
 * squared roughness, metallic/effective IOR/dielectric F0/mix weight, then
 * transmission weight, original IOR, the standalone transmission closure and
 * subsurface weight, subsurface RGB radius and encoded anisotropy, and the
 * surface emission with the indirect clamp. Subsurface stores its entry IOR
 * and BSSRDF type in the two Glass slots. Hair uses the middle four records
 * for tangents, lobe parameters and luminance or ellipse and RNG data.
 * Anisotropic Principled stores the negative aspect in the closure model slot
 * and its tangent in the otherwise unused subsurface RGB. */
/* The final vector holds either a per-closure normal or the shader-wide
 * shading normal produced by displacement bump. Odd modes omit the stock
 * bump-map correction, and the next mode enables it. When eligible closures
 * also supply a separate normal, their shader-wide normal uses the free
 * subsurface record. */
#define CYCLES_WEBGPU_NORMAL_CLOSURE 1.0f
#define CYCLES_WEBGPU_NORMAL_SHADER 3.0f
#define CYCLES_WEBGPU_NORMAL_BOTH 5.0f
#define CYCLES_WEBGPU_MATERIAL_VECTORS 7
#define CYCLES_WEBGPU_MATERIAL_NORMAL_SLOT 6u
/* Private bits in the packed ShaderDataFlag record. Stock flags start at bit 12. */
#define CYCLES_WEBGPU_SHADER_NO_REFLECTIVE_CAUSTICS (1u << 0u)
#define CYCLES_WEBGPU_SHADER_DIFFUSE_PRINCIPLED_MIX (1u << 1u)
#define CYCLES_WEBGPU_SHADER_INDEPENDENT_MIXED_BUMP (1u << 2u)
#define CYCLES_WEBGPU_SHADER_ORDERED_CLOSURES (1u << 3u)
#define CYCLES_WEBGPU_SHADER_GLOSSY_DISPLACEMENT_BUMP (1u << 4u)
#define CYCLES_WEBGPU_SHADER_GLOSSY_TRANSPARENT (1u << 5u)
#define CYCLES_WEBGPU_SHADER_ORDERED_TRANSPARENCY (1u << 6u)
#define CYCLES_WEBGPU_SHADER_MIXED_DISPLACEMENT_BUMP (1u << 7u)
#define CYCLES_WEBGPU_SHADER_TRANSPARENT_EMISSION (1u << 8u)
#define CYCLES_WEBGPU_SHADER_ORDERED_PRINCIPLED (1u << 9u)
#define CYCLES_WEBGPU_SHADER_PRINCIPLED_TRANSLUCENT (1u << 10u)
#define CYCLES_WEBGPU_SHADER_ORDERED_GLOSSY_MIX (1u << 11u)
#define CYCLES_WEBGPU_SHADER_BUMP_CORRECTION (1u << 15u)
#define CYCLES_WEBGPU_SHADER_BSSRDF_BUMP (1u << 21u)
#define CYCLES_WEBGPU_SVM_EVAL_AOV 1u
#define CYCLES_WEBGPU_SVM_EVAL_SUBSURFACE_EXIT 2u
#define CYCLES_WEBGPU_SVM_EVAL_BUMP_CORRECTION 4u
#define CYCLES_WEBGPU_MATERIAL_WORDS (4 * CYCLES_WEBGPU_MATERIAL_VECTORS)
#define CYCLES_WEBGPU_MATERIAL_EMISSION_SLOT 5u
#define CYCLES_WEBGPU_GLASS_GGX 1.0f
#define CYCLES_WEBGPU_GLASS_BECKMANN 2.0f
#define CYCLES_WEBGPU_REFRACTION_GGX 3.0f
#define CYCLES_WEBGPU_REFRACTION_BECKMANN 4.0f
#define CYCLES_WEBGPU_MATERIAL_UNSHADED 2.0f
#define CYCLES_WEBGPU_MATERIAL_GGX 3.0f
#define CYCLES_WEBGPU_MATERIAL_MULTISCATTER_GGX 4.0f
/* Principled with a separate Translucent lobe uses the otherwise empty Glass
 * record for its weighted RGB and its order relative to the Principled lobes. */
#define CYCLES_WEBGPU_GGX_TRANSLUCENT 1u
#define CYCLES_WEBGPU_GGX_TRANSLUCENT_FIRST 2u
#define CYCLES_WEBGPU_GGX_TRANSLUCENT_CORRECTION 4u
#define CYCLES_WEBGPU_GGX_HAS_TRANSLUCENT(glass) ((glass).w < 0.0f)
#define CYCLES_WEBGPU_GGX_CAN_BE_TRANSPARENT(glass) \
  (CYCLES_WEBGPU_GGX_HAS_TRANSLUCENT(glass) || ((glass).w == 0.0f && (glass).z >= 0.0f))
/* Holdout weight in RGB. Camera paths write its transparency, and the path
 * ends without scattering, emission or light sampling. */
#define CYCLES_WEBGPU_MATERIAL_HOLDOUT 5.0f
#define CYCLES_WEBGPU_MATERIAL_TRANSLUCENT 6.0f
#define CYCLES_WEBGPU_MATERIAL_DIFFUSE_TOON 7.0f
#define CYCLES_WEBGPU_MATERIAL_GLOSSY_TOON 8.0f
#define CYCLES_WEBGPU_MATERIAL_GLOSSY 9.0f
#define CYCLES_WEBGPU_MATERIAL_VELVET 10.0f
#define CYCLES_WEBGPU_MATERIAL_SHEEN 11.0f
#define CYCLES_WEBGPU_MATERIAL_TRANSPARENT 12.0f
#define CYCLES_WEBGPU_MATERIAL_HAIR_REFLECTION 13.0f
#define CYCLES_WEBGPU_MATERIAL_HAIR_TRANSMISSION 14.0f
#define CYCLES_WEBGPU_MATERIAL_HAIR_CHIANG 15.0f
#define CYCLES_WEBGPU_MATERIAL_HAIR_HUANG 16.0f
#define CYCLES_WEBGPU_MATERIAL_IS_HAIR(status) \
  ((status) >= CYCLES_WEBGPU_MATERIAL_HAIR_REFLECTION && (status) <= CYCLES_WEBGPU_MATERIAL_HAIR_HUANG)
#define CYCLES_WEBGPU_MATERIAL_HAIR_TRANSMITS(status) \
  ((status) >= CYCLES_WEBGPU_MATERIAL_HAIR_TRANSMISSION && (status) <= CYCLES_WEBGPU_MATERIAL_HAIR_HUANG)
#define CYCLES_WEBGPU_MATERIAL_RAY_PORTAL 17.0f
/* Mixed Lambert and Transparent use the base RGB for Lambert, the specular
 * record for extinction and its sample weight, and the layer record for the
 * Lambert sample weight and closure order. */
#define CYCLES_WEBGPU_MATERIAL_MIXED_TRANSPARENT 18.0f
/* Base and specular RGB are the weighted Diffuse and Translucent colors.
 * Layers store both closure sample weights, Translucent-first order and the
 * stock bump correction mode for the adjusted Translucent normal. */
#define CYCLES_WEBGPU_MATERIAL_DIFFUSE_TRANSLUCENT 19.0f
/* A Diffuse and isotropic Glossy mixture uses specular RGB for weighted
 * Glossy color, specular W for squared roughness, and layers W for the
 * stock GGX or Beckmann closure type. */
#define CYCLES_WEBGPU_MATERIAL_DIFFUSE_GLOSSY 20.0f
/* Mixed Diffuse + Glossy ordering and per-closure use of the packed Bump normal. */
#define CYCLES_WEBGPU_MIXED_GLOSSY_FIRST 1u
#define CYCLES_WEBGPU_MIXED_BUMP_DIFFUSE_ONLY 2u
#define CYCLES_WEBGPU_MIXED_BUMP_GLOSSY_ONLY 4u
/* Slot 6 holds the Diffuse Bump normal and correction mode. Slot 4 RGB
 * carries the separate Glossy Bump normal. */
#define CYCLES_WEBGPU_MIXED_BUMP_INDEPENDENT 8u
/* The base record holds the tail vector index, live closure count, and Bump
 * correction flag. Each ordered record holds RGB weight and type, then the
 * raw normal and squared roughness. The per-ray tail follows film AOVs.
 * Auxiliary RGB holds Transparent extinction and W holds live lobe flags. */
#define CYCLES_WEBGPU_MATERIAL_ORDERED_CLOSURES 21.0f
#define CYCLES_WEBGPU_ORDERED_HAS_TRANSPARENT 1u
#define CYCLES_WEBGPU_ORDERED_HAS_TRANSLUCENT 2u
/* Standalone Glossy records plus weighted Transparent RGB and its signed
 * sample weight in the auxiliary record. The sign retains closure order. */
#define CYCLES_WEBGPU_MATERIAL_GLOSSY_TRANSPARENT 22.0f
#define CYCLES_WEBGPU_MATERIAL_ORDERED_PRINCIPLED 23.0f
#define CYCLES_WEBGPU_ORDERED_PRINCIPLED_MAPPED_NORMAL 1u
#define CYCLES_WEBGPU_ORDERED_PRINCIPLED_SKIP_METAL_REFLECTION 2u
#define CYCLES_WEBGPU_MATERIAL_IS_GLOSSY(status) \
  ((status) == CYCLES_WEBGPU_MATERIAL_GLOSSY || (status) == CYCLES_WEBGPU_MATERIAL_GLOSSY_TRANSPARENT)
#define CYCLES_WEBGPU_ORDERED_LOBE_BUMP 256u
#define CYCLES_WEBGPU_MAX_ORDERED_LOBES 4u
#define CYCLES_WEBGPU_ORDERED_LOBE_VECTORS 2u
#define CYCLES_WEBGPU_ORDERED_PATH_VECTORS \
  (CYCLES_WEBGPU_MAX_ORDERED_LOBES * CYCLES_WEBGPU_ORDERED_LOBE_VECTORS)
#ifdef CYCLES_WEBGPU_ORDERED_CLOSURES
#  define CYCLES_WEBGPU_MATERIAL_HAS_ORDERED_CLOSURES(status) \
    ((status) == CYCLES_WEBGPU_MATERIAL_ORDERED_CLOSURES || \
     (status) == CYCLES_WEBGPU_MATERIAL_ORDERED_PRINCIPLED)
#else
#  define CYCLES_WEBGPU_MATERIAL_HAS_ORDERED_CLOSURES(status) false
#endif
/* Returned by the GGX sampler so the scheduler reuses the transparent
 * ray transition without replacing the previous vertex or its MIS PDF. */
#define CYCLES_WEBGPU_MIXED_TRANSPARENT_PENDING (1u << 30u)
#define CYCLES_WEBGPU_SHEEN_LTC_WORDS (3u * 32u * 32u)
#ifdef CYCLES_WEBGPU_EXCLUDE_STANDALONE
#  define CYCLES_WEBGPU_MATERIAL_IS_STANDALONE(status) false
#else
#  define CYCLES_WEBGPU_MATERIAL_IS_STANDALONE(status) \
  (((status) >= CYCLES_WEBGPU_MATERIAL_TRANSLUCENT && (status) <= CYCLES_WEBGPU_MATERIAL_SHEEN) || \
   CYCLES_WEBGPU_MATERIAL_IS_HAIR(status) || \
   (status) == CYCLES_WEBGPU_MATERIAL_DIFFUSE_TRANSLUCENT || \
   (status) == CYCLES_WEBGPU_MATERIAL_DIFFUSE_GLOSSY || \
   (status) == CYCLES_WEBGPU_MATERIAL_GLOSSY_TRANSPARENT || \
   CYCLES_WEBGPU_MATERIAL_HAS_ORDERED_CLOSURES(status))
#endif
#define CYCLES_WEBGPU_MATERIAL_USES_SPECULAR_NORMAL(status) \
  ((status) == CYCLES_WEBGPU_MATERIAL_TRANSLUCENT || CYCLES_WEBGPU_MATERIAL_IS_GLOSSY(status) || \
   (status) == CYCLES_WEBGPU_MATERIAL_HAIR_REFLECTION || \
   (status) == CYCLES_WEBGPU_MATERIAL_HAIR_TRANSMISSION)
/* The shader count word also selects the extended SVM interpreter. */
#define CYCLES_WEBGPU_SVM_EXTENDED 0x80000000u
#define CYCLES_WEBGPU_SVM_QUERIES 0x10000000u
#define CYCLES_WEBGPU_SVM_STANDALONE 0x40000000u
#define CYCLES_WEBGPU_SVM_LOOKUP_TABLES 0x08000000u
#define CYCLES_WEBGPU_SVM_AOV 0x20000000u
#define CYCLES_WEBGPU_SVM_RAY_PORTAL 0x04000000u
#define CYCLES_WEBGPU_SVM_PIXEL_WIREFRAME 0x02000000u
#define CYCLES_WEBGPU_SVM_HAIR 0x01000000u
#define CYCLES_WEBGPU_SVM_MIXED_TRANSPARENT 0x00800000u
#define CYCLES_WEBGPU_SVM_BUMP 0x00400000u
#define CYCLES_WEBGPU_SVM_SHADER_COUNT 0x003fffffu
#define CYCLES_WEBGPU_SVM_NEEDS_DIFFERENTIALS \
  (CYCLES_WEBGPU_SVM_PIXEL_WIREFRAME | CYCLES_WEBGPU_SVM_BUMP)
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
#define CYCLES_WEBGPU_SVM_DUPLI_GENERATED 27u
#define CYCLES_WEBGPU_SVM_DUPLI_UV 28u

/* Additional offsets in the existing surface metadata. The forward transforms
 * are already packed for triangle shading. */
#define CYCLES_WEBGPU_SURFACE_OBJECT_SVM_OFFSET 5u
#define CYCLES_WEBGPU_SURFACE_SHADER_PASS_OFFSET 6u
#define CYCLES_WEBGPU_SURFACE_OFFSETS 7u
#define CYCLES_WEBGPU_SURFACE_TRANSPARENT_OFFSET 7u
#define CYCLES_WEBGPU_SURFACE_TRANSPARENT_OFFSETS 8u
/* A larger surface prefix independently selects the optional tangent output,
 * leaving the existing seven offsets and transparent shadow slot intact. */
#define CYCLES_WEBGPU_SURFACE_HAIR_HEADER_WORDS 2u
#define CYCLES_WEBGPU_SURFACE_HAS_HAIR(header) \
  ((header) == CYCLES_WEBGPU_SURFACE_OFFSETS + CYCLES_WEBGPU_SURFACE_HAIR_HEADER_WORDS || \
   (header) == CYCLES_WEBGPU_SURFACE_TRANSPARENT_OFFSETS + CYCLES_WEBGPU_SURFACE_HAIR_HEADER_WORDS || \
   (header) == CYCLES_WEBGPU_SURFACE_VOLUME_TANGENT_OFFSETS || \
   (header) == CYCLES_WEBGPU_SURFACE_VOLUME_SHADOW_TANGENT_OFFSETS)
#define CYCLES_WEBGPU_SURFACE_HAS_TRANSPARENCY(header) \
  ((header) == CYCLES_WEBGPU_SURFACE_TRANSPARENT_OFFSETS || \
   (header) == CYCLES_WEBGPU_SURFACE_TRANSPARENT_OFFSETS + CYCLES_WEBGPU_SURFACE_HAIR_HEADER_WORDS)
#define CYCLES_WEBGPU_TRANSPARENT_SHADOW_HEADER_WORDS 2u
#define CYCLES_WEBGPU_TRANSPARENT_SHADOW_ENABLE 1u
#define CYCLES_WEBGPU_TRANSPARENT_SHADOW_VARYING 2u
#define CYCLES_WEBGPU_TRANSPARENT_SHADOW_PRINCIPLED_ALPHA 4u
#define CYCLES_WEBGPU_TRANSPARENT_SHADOW_CUBIC 8u
#define CYCLES_WEBGPU_TRANSPARENT_SHADOW_EXTENDED_SVM 16u
/* A non-color word identifying a shadow hit that needs SVM evaluation. */
#define CYCLES_WEBGPU_TRANSPARENT_SHADOW_EVAL 0x7fc00000u
#define CYCLES_WEBGPU_SURFACE_OBJECT_WORDS 23u

/* The SVM transforms follow the eight float4 records used by camera rays. */
#define CYCLES_WEBGPU_CAMERA_SVM_OFFSET 32u
#define CYCLES_WEBGPU_CAMERA_WORDS 60u
/* Camera ray input extends the shared raster-to-world prefix with lens and differential records. */
#define CYCLES_WEBGPU_CAMERA_LENS_WORDS 16u
#define CYCLES_WEBGPU_CAMERA_RAY_WORDS \
  (CYCLES_WEBGPU_CAMERA_SVM_OFFSET + CYCLES_WEBGPU_CAMERA_LENS_WORDS)
/* The ray-query SVM variant appends Cycles' camera sampling after the AOV header. */
#define CYCLES_WEBGPU_QUERY_CAMERA_OFFSET \
  (CYCLES_WEBGPU_CAMERA_WORDS + CYCLES_WEBGPU_CAMERA_SVM_CONFIG_WORDS)
#define CYCLES_WEBGPU_QUERY_HIT_WORDS 8u
#define CYCLES_WEBGPU_QUERY_RNG 29u
#define CYCLES_WEBGPU_QUERY_HIT 30u
#define CYCLES_WEBGPU_SVM_TRIANGLE_DPDU 31u
#define CYCLES_WEBGPU_SVM_TRIANGLE_DPDV 32u
#define CYCLES_WEBGPU_SURFACE_HAIR_WORDS 8u
#define CYCLES_WEBGPU_SVM_DIFFERENTIAL 33u
#define CYCLES_WEBGPU_CAMERA_DIFFERENTIAL_SCALE 31u
/* Optional SVM configuration between the camera and the shader-query sampling tables. */
#define CYCLES_WEBGPU_CAMERA_SVM_CONFIG_WORDS 4u
#define CYCLES_WEBGPU_CAMERA_AOV_COLORS CYCLES_WEBGPU_CAMERA_WORDS
#define CYCLES_WEBGPU_CAMERA_AOV_VALUES (CYCLES_WEBGPU_CAMERA_WORDS + 1u)

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
#define CYCLES_WEBGPU_LIGHT_HEADER_WORDS 15
#define CYCLES_WEBGPU_PATH_SETTINGS_WORDS 11
