/* SPDX-License-Identifier: Apache-2.0 */
#pragma once

namespace ccl {

ccl_device_inline void webgpu_sample_cos_hemisphere(const float3 N,
                                                    const float2 random,
                                                    float3 *wo,
                                                    float *pdf)
{
  const float2 disk = sample_uniform_disk(random);
  /* The concentric mapping preserves this radius. Compute the height before
   * trigonometric rounding can magnify the error near the horizon. */
  const float radius = max(fabsf(2.0f * random.x - 1.0f), fabsf(2.0f * random.y - 1.0f));
  const float costheta = safe_sqrtf((1.0f - radius) * (1.0f + radius));
  float3 T, B;
  make_orthonormals(N, &T, &B);
  *wo = disk.x * T + disk.y * B + costheta * N;
  *pdf = costheta * M_1_PI_F;
}

}  // namespace ccl
