/* SPDX-License-Identifier: Apache-2.0 */

#ifndef __CYCLES_WEBGPU_WAVEFRONT_LIGHT_DECLARATION__
#define __CYCLES_WEBGPU_WAVEFRONT_LIGHT_DECLARATION__

extern void cycles_wavefront_light(uint4 position, uint4 normal, uint4 spec_normal,
                                   uint4 geometric, uint4 shadow_along, uint4 shadow_opposite,
                                   uint4 hit, uint4 ray, uint4 origin, uint4 path, uint4 sampling,
                                   uint subsurface_steps,
                                   float4 material, float4 specular, float4 parameters,
                                   float4 glass, float4 hair,
                                   out float4 result, out float4 area_result, out uint hair_lcg
#ifdef CYCLES_WEBGPU_LIGHT_TREE
                                   , uint4 mis_origin_normal, float4 throughput
#endif
#ifdef CYCLES_WEBGPU_DEFER_SHADOW
                                   , out uint4 shadow_origin, out uint4 shadow_direction,
                                   out uint4 shadow_options
#  ifdef CYCLES_WEBGPU_LIGHT_TREE
                                   , out uint4 sampled_emission
#  endif
#endif
                                   );

#endif
