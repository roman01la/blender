/* SPDX-License-Identifier: Apache-2.0 */

#pragma once

namespace ccl {

/* Use Cycles' shadow_ray_offset() with the surface stage's cached triangle
 * offsets. Do not change the direction or endpoint of an infinite ray. */
ccl_device_inline bool webgpu_shadow_ray_offset(float3 &P,
                                               const float3 D,
                                               const float3 N,
                                               const float3 Ng,
                                               const float3 along,
                                               const float3 opposite,
                                               const float cutoff)
{
  if (!(cutoff > 0.0f)) {
    return true;
  }
  const float NL = dot(N, D);
  const bool transmit = NL < 0.0f;
  const float NgL = dot(transmit ? -Ng : Ng, D);
  const float abs_NL = fabsf(NL);
  const float amount = abs_NL < cutoff ? clamp(2.0f - (NgL + abs_NL) / cutoff, 0.0f, 1.0f) :
                                        clamp(1.0f - NgL / cutoff, 0.0f, 1.0f);
  if (amount > 0.0f) {
    P += (transmit ? opposite : along) * amount;
    return NgL > 0.0f;
  }
  return true;
}

}  // namespace ccl
