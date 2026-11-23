/* SPDX-License-Identifier: Apache-2.0 */

#ifndef __CYCLES_WEBGPU_LIGHT_PASSES__
#define __CYCLES_WEBGPU_LIGHT_PASSES__

/* The optional radiance tail keeps the first surface's BSDF weights alive
 * across bounces. A second pair holds the current light sample's evaluation.
 * See Cycles' bsdf_eval_pass_*_weight and film_write_direct_light. */
uint cycles_light_pass_record(uint capacity, uint lane)
{
  const float4 layout = CYCLES_WEBGPU_LIGHT_PASS_BUFFER[2u * capacity];
  return 2u * capacity + 1u + uint(layout.w) * capacity + 6u * lane;
}

bool cycles_light_pass_enabled(uint capacity)
{
  return CYCLES_WEBGPU_LIGHT_PASS_BUFFER[2u * capacity].x != 0.0f;
}

void cycles_light_pass_deltas(uint capacity, uint lane, uint bounce,
                              float3 forward, float3 nee, bool has_nee)
{
  const uint base = cycles_light_pass_record(capacity, lane);
  CYCLES_WEBGPU_LIGHT_PASS_BUFFER[base + 4u] = float4(
      forward, float(__builtin_elementwise_min(bounce, 2u)));
  CYCLES_WEBGPU_LIGHT_PASS_BUFFER[base + 5u] = float4(nee, has_nee ? float(1u + uint(bounce > 0u)) : 0.0f);
}

#endif
