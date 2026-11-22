/* SPDX-License-Identifier: Apache-2.0 */

StructuredBuffer<uint> entries : register(t0);
RWStructuredBuffer<float4> results : register(u1);

extern void cycles_volume_coefficients(uint4 options,
                                       float4 weight,
                                       uint4 data0,
                                       uint4 data1,
                                       uint4 data2,
                                       uint4 data3,
                                       uint4 data4,
                                       float4 incoming,
                                       float4 random,
                                       out float4 extinction,
                                       out float4 scattering,
                                       out float4 transmittance,
                                       out float4 outgoing,
                                       out float4 emission);

uint4 entry_words(uint index)
{
  const uint word = index * 4;
  return uint4(entries[word], entries[word + 1], entries[word + 2], entries[word + 3]);
}

[numthreads(32, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
  if (id.x >= entries[0]) {
    return;
  }
  const uint input = 1 + 9 * id.x;
  float4 extinction, scattering, transmittance, outgoing, emission;
  cycles_volume_coefficients(entry_words(input),
                              __builtin_bit_cast(float4, entry_words(input + 1)),
                              entry_words(input + 2), entry_words(input + 3), entry_words(input + 4),
                              entry_words(input + 5), entry_words(input + 6),
                              __builtin_bit_cast(float4, entry_words(input + 7)),
                              __builtin_bit_cast(float4, entry_words(input + 8)),
                              extinction, scattering, transmittance, outgoing, emission);
  const uint result = 5 * id.x;
  results[result] = extinction;
  results[result + 1] = scattering;
  results[result + 2] = transmittance;
  results[result + 3] = outgoing;
  results[result + 4] = emission;
}
