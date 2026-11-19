/* SPDX-License-Identifier: Apache-2.0 */

RWStructuredBuffer<uint> film_words : register(u0);
StructuredBuffer<uint> params : register(t1);

uint cycles_adaptive_film_read(uint index)
{
  return film_words[index];
}

void cycles_adaptive_film_write(uint index, uint value)
{
  film_words[index] = value;
}

#ifdef CYCLES_WEBGPU_ADAPTIVE_CHECK
extern uint cycles_adaptive_check(uint4 pixel,
                                  uint4 layout,
                                  uint sample_count_offset,
                                  float exposure,
                                  float threshold);
#else
extern void cycles_adaptive_filter(uint4 line, uint4 layout);
#endif

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
  const uint4 layout = uint4(params[5], params[6], params[7], params[9]);
#if defined(CYCLES_WEBGPU_ADAPTIVE_CHECK)
  const uint width = params[2];
  const uint y = id.x / width;
  if (y < params[3]) {
    const uint4 pixel = uint4(id.x - y * width, y, params[12], params[4]);
    cycles_adaptive_check(
        pixel, layout, params[8], __builtin_bit_cast(float, params[10]),
        __builtin_bit_cast(float, params[11]));
  }
#elif defined(CYCLES_WEBGPU_ADAPTIVE_FILTER_X)
  if (id.x < params[3]) {
    cycles_adaptive_filter(uint4(id.x, params[2], 0, params[4]), layout);
  }
#else
  if (id.x < params[2]) {
    cycles_adaptive_filter(uint4(id.x, params[3], 0, params[4]), layout);
  }
#endif
}
