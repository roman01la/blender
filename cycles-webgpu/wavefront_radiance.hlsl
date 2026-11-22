/* SPDX-License-Identifier: Apache-2.0 */

#ifndef __CYCLES_WEBGPU_WAVEFRONT_RADIANCE__
#define __CYCLES_WEBGPU_WAVEFRONT_RADIANCE__

extern uint cycles_path_transparent_max_bounce();
extern uint cycles_film_transparent();

bool cycles_trace_past_area_limit(uint transparent_bounce)
{
  return transparent_bounce < cycles_path_transparent_max_bounce();
}

uint cycles_forward_light_hits(float4 light)
{
  return light.w < 0.0f ? uint(-light.w) : uint(light.w > 0.0f);
}

#if !defined(CYCLES_WEBGPU_DENOISE_SURFACE) && !defined(CYCLES_WEBGPU_TRANSPARENT_SHADOW)
bool cycles_trace_past_area(uint path, uint hits)
{
  /* Cycles counts a light hit as transparent without advancing the RNG. */
  const uint counter_word = path + CYCLES_WEBGPU_PATH_TRANSPARENT_BOUNCE;
  const uint packed = path_words[counter_word];
  const uint transparent_bounce = (packed & CYCLES_WEBGPU_TRANSPARENT_BOUNCE_MASK) + hits;
  path_words[counter_word] = (packed & ~CYCLES_WEBGPU_TRANSPARENT_BOUNCE_MASK) |
                             __builtin_elementwise_min(
                                 transparent_bounce, CYCLES_WEBGPU_TRANSPARENT_BOUNCE_MASK);
  return cycles_trace_past_area_limit(transparent_bounce);
}

bool cycles_write_transparent_background(uint path, uint flags, float3 throughput)
{
  if (cycles_film_transparent() == 0u ||
      (flags & CYCLES_WEBGPU_PATH_TRANSPARENT_BACKGROUND) == 0u) {
    return false;
  }
  // Match Cycles' background integration. Path state stores alpha and film stores transparency.
  const float transparent = (throughput.x + throughput.y + throughput.z) * (1.0f / 3.0f);
  path_words[path + CYCLES_WEBGPU_PATH_RADIANCE + 3u] = __builtin_bit_cast(uint, 1.0f - transparent);
  return true;
}
#endif

float3 clamp_light(float3 value, float limit)
{
  // Match the per-contribution finite handling and sum in Cycles' film_clamp_light.
  const uint3 magnitude = __builtin_bit_cast(uint3, value) & uint3(0x7fffffff, 0x7fffffff, 0x7fffffff);
  const float3 contribution = float3(magnitude.x < 0x7f800000 ? value.x : 0.0f,
                                      magnitude.y < 0x7f800000 ? value.y : 0.0f,
                                      magnitude.z < 0x7f800000 ? value.z : 0.0f);
  const float3 absolute = __builtin_elementwise_abs(contribution);
  const float sum = absolute.x + absolute.y + absolute.z;
  return sum > limit ? contribution * (limit / sum) : contribution;
}

float3 cycles_forward_emission(float4 light, float3 throughput)
{
  return light.w < 0.0f ? light.xyz : clamp_light(throughput * light.xyz, light.w);
}

#endif
