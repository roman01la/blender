/* SPDX-License-Identifier: Apache-2.0 */

StructuredBuffer<uint> scene_words : register(t0);
StructuredBuffer<uint> surface_words : register(t1);
StructuredBuffer<uint> camera_sampling : register(t2);
StructuredBuffer<uint4> entries : register(t3);
RWStructuredBuffer<uint4> exits : register(u4);

uint cycles_scene_word(uint index)
{
  return scene_words[index];
}

uint cycles_surface_word(uint index)
{
  return surface_words[index];
}

uint cycles_camera_sample_word(uint index)
{
  return camera_sampling[index];
}

extern void cycles_subsurface_walk(float4 entry, float4 direction, float4 normal,
                                   float4 albedo, float4 radius, float4 weight,
                                   uint4 rng, uint4 surface, out uint4 hit, out float4 coordinates,
                                   out float4 exit_direction, out float4 throughput);

[numthreads(32, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
  if (id.x >= entries[0].x) {
    return;
  }

  const uint entry = 1 + id.x * 8;
  uint4 hit;
  float4 coordinates, direction, throughput;
  cycles_subsurface_walk(__builtin_bit_cast(float4, entries[entry]),
                         __builtin_bit_cast(float4, entries[entry + 1]),
                         __builtin_bit_cast(float4, entries[entry + 2]),
                         __builtin_bit_cast(float4, entries[entry + 3]),
                         __builtin_bit_cast(float4, entries[entry + 4]),
                         __builtin_bit_cast(float4, entries[entry + 5]),
                         entries[entry + 6], entries[entry + 7], hit, coordinates, direction,
                         throughput);
  const uint result = id.x * 4;
  exits[result] = hit;
  exits[result + 1] = __builtin_bit_cast(uint4, coordinates);
  exits[result + 2] = __builtin_bit_cast(uint4, direction);
  exits[result + 3] = __builtin_bit_cast(uint4, throughput);
}
