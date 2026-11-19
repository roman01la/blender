/* SPDX-License-Identifier: Apache-2.0 */

#ifndef __CYCLES_WEBGPU_WAVEFRONT_RADIANCE__
#define __CYCLES_WEBGPU_WAVEFRONT_RADIANCE__

extern uint cycles_path_transparent_max_bounce();

bool cycles_trace_past_area_limit(uint transparent_bounce)
{
  return transparent_bounce < cycles_path_transparent_max_bounce();
}

#ifndef CYCLES_WEBGPU_DENOISE_SURFACE
bool cycles_trace_past_area(uint path)
{
  /* Cycles counts a light hit as transparent without advancing the RNG. */
  const uint counter_word = path + CYCLES_WEBGPU_PATH_TRANSPARENT_BOUNCE;
  const uint transparent_bounce = path_words[counter_word] + 1;
  path_words[counter_word] = transparent_bounce;
  return cycles_trace_past_area_limit(transparent_bounce);
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

#endif
