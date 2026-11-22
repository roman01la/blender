/* SPDX-License-Identifier: Apache-2.0 */

#include "../blender-platform/cycles-webgpu/path_state_layout.h"
#include "../blender-platform/cycles-webgpu/volume_layout.h"

StructuredBuffer<uint> scene_words : register(t0);
StructuredBuffer<uint> ray_words : register(t1);
#ifdef CYCLES_WEBGPU_ABSORPTION_SHADOW
StructuredBuffer<uint4> shadow_words : register(t2);
#  ifdef CYCLES_WEBGPU_VOLUME_SCATTER_SHADOW
RWStructuredBuffer<uint> path_words : register(u3);
#  else
RWStructuredBuffer<float4> radiance : register(u3);
#    ifdef CYCLES_WEBGPU_VOLUME_EXTINCTION_SHADOW
StructuredBuffer<uint> path_words : register(t6);
#    endif
#  endif
StructuredBuffer<uint> metadata_words : register(t4);
StructuredBuffer<uint> volume_state : register(t5);
#  ifdef CYCLES_WEBGPU_VOLUME_GUIDES
RWStructuredBuffer<float4> guide_state : register(u6);
StructuredBuffer<uint> guide_params : register(t7);
#  endif

extern float4 cycles_absorption_shadow(uint4 origin, uint4 direction, uint4 options,
                                       uint4 geometric, uint state
#  ifdef CYCLES_WEBGPU_VOLUME_EXTINCTION_SHADOW
                                       , uint4 rng
#  endif
                                       );
#else
RWStructuredBuffer<uint> hit_words : register(u2);
StructuredBuffer<uint> metadata_words : register(t3);
RWStructuredBuffer<uint> path_words : register(u4);
RWStructuredBuffer<uint> volume_state : register(u5);

extern float4 cycles_absorption_segment(uint4 origin, uint4 direction, uint4 hit,
                                        uint hit_t, uint init, uint state, out uint next_state
#  ifdef CYCLES_WEBGPU_VOLUME_INTERVAL
                                        , out uint prior_state
#  endif
                                        );
#endif

#include "wavefront_queue.hlsl"

typedef float float3 __attribute__((ext_vector_type(3)));
#ifdef CYCLES_WEBGPU_VOLUME_GUIDES
#  include "volume-denoising.hlsl"
#endif

uint cycles_scene_word(uint index)
{
  return scene_words[index];
}

uint cycles_volume_word(uint index)
{
  return metadata_words[index];
}

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
  uint lane;
  if (!cycles_queue_slot(id.x, lane)) {
    return;
  }
#ifdef CYCLES_WEBGPU_ABSORPTION_SHADOW
  const uint shadow = 4 * lane;
#  ifdef CYCLES_WEBGPU_VOLUME_GUIDES
  const float4 guide = __builtin_bit_cast(float4, shadow_words[shadow + 3]);
  const uint feature = 3 * lane;
  float4 feature_throughput = guide_state[feature];
  if ((guide.x != 0.0f || guide.y != 0.0f || guide.z != 0.0f || guide.w != 0.0f) &&
      (feature_throughput.x != 0.0f || feature_throughput.y != 0.0f ||
       feature_throughput.z != 0.0f))
  {
    float4 albedo = guide_state[feature + 1];
    float4 normal = guide_state[feature + 2];
    accumulate_volume_features(guide.xyz, guide.w != 0.0f, guide_params[0] != 0xffffffffu,
                               guide_params[1] != 0xffffffffu, feature_throughput, albedo, normal);
    guide_state[feature] = feature_throughput;
    guide_state[feature + 1] = albedo;
    guide_state[feature + 2] = normal;
  }
#  endif
  const uint4 options = shadow_words[shadow + 2];
  if (options.x != 0u) {
#  ifdef CYCLES_WEBGPU_VOLUME_EXTINCTION_SHADOW
    const uint capacity = ray_words[0];
    const uint ray = 1 + 12 * lane;
    const uint path = CYCLES_WEBGPU_PATH_WORDS * lane;
    const uint bounce = path_words[path + CYCLES_WEBGPU_PATH_BOUNCE];
    const uint steps = path_words[path + CYCLES_WEBGPU_PATH_EXTRA_RNG_STEPS];
    const uint4 rng = uint4(ray_words[ray + CYCLES_WEBGPU_RAY_RNG_PIXEL],
                           path_words[path + CYCLES_WEBGPU_PATH_PIXEL + 2],
                           (bounce + steps + 1u) * CYCLES_WEBGPU_PRNG_BOUNCE_NUM, 0u);
#  endif
#  ifdef CYCLES_WEBGPU_VOLUME_SCATTER_SHADOW
    const uint stack = volume_state[CYCLES_WEBGPU_VOLUME_SHADOW_STACK * capacity + lane];
    const float3 T = cycles_absorption_shadow(shadow_words[shadow], shadow_words[shadow + 1],
                                              options, uint4(0), stack, rng).xyz;
    const uint direct = CYCLES_WEBGPU_VOLUME_DIRECT * capacity + lane;
    float3 sample = T * __builtin_bit_cast(
        float3, uint3(volume_state[direct], volume_state[direct + capacity],
                      volume_state[direct + 2 * capacity]));
    const uint medium = cycles_volume_word(CYCLES_WEBGPU_SURFACE_ABSORPTION_OFFSET);
    const float limit = options.w == 0u ? 3.402823466e38f : __builtin_bit_cast(
        float, cycles_volume_word(medium + CYCLES_WEBGPU_VOLUME_INDIRECT_CLAMP));
    const float abs_value = __builtin_elementwise_abs(sample.x) +
                            __builtin_elementwise_abs(sample.y) +
                            __builtin_elementwise_abs(sample.z);
    if (abs_value > limit) {
      sample *= limit / abs_value;
    }
    const uint film = lane * CYCLES_WEBGPU_PATH_WORDS + CYCLES_WEBGPU_PATH_RADIANCE;
    path_words[film] = __builtin_bit_cast(uint, __builtin_bit_cast(float, path_words[film]) + sample.x);
    path_words[film + 1] = __builtin_bit_cast(
        uint, __builtin_bit_cast(float, path_words[film + 1]) + sample.y);
    path_words[film + 2] = __builtin_bit_cast(
        uint, __builtin_bit_cast(float, path_words[film + 2]) + sample.z);
#  else
    const float3 T = cycles_absorption_shadow(shadow_words[shadow], shadow_words[shadow + 1],
                                              options, shadow_words[shadow + 3],
                                              volume_state[lane]
#    ifdef CYCLES_WEBGPU_VOLUME_EXTINCTION_SHADOW
                                              , rng
#    endif
                                              ).xyz;
    const float4 sample = radiance[lane];
    radiance[lane] = float4(T * sample.xyz, sample.w);
#  endif
  }
#else
  const uint ray = 1 + 12 * lane;
  const uint4 origin = uint4(ray_words[ray], ray_words[ray + 1], ray_words[ray + 2], ray_words[ray + 3]);
  const uint4 direction = uint4(ray_words[ray + 4], ray_words[ray + 5], ray_words[ray + 6], ray_words[ray + 7]);
  const uint h = 8 * lane;
  const uint4 hit = uint4(hit_words[h], hit_words[h + 1], hit_words[h + 2], hit_words[h + 3]);
  const uint path = lane * CYCLES_WEBGPU_PATH_WORDS;
  const bool init = path_words[path + CYCLES_WEBGPU_PATH_BOUNCE] == 0u &&
                    path_words[path + CYCLES_WEBGPU_PATH_EXTRA_RNG_STEPS] == 0u;
  const uint state = init ? 0u : volume_state[lane];
  uint next_state;
#  ifdef CYCLES_WEBGPU_VOLUME_INTERVAL
  uint prior_state;
#  endif
  const float4 segment = cycles_absorption_segment(origin, direction, hit, hit_words[h + 4],
                                                    uint(init), state, next_state
#  ifdef CYCLES_WEBGPU_VOLUME_INTERVAL
                                                    , prior_state
#  endif
                                                    );
#  ifdef CYCLES_WEBGPU_VOLUME_INTERVAL
  const uint capacity = ray_words[0];
  volume_state[lane] = prior_state;
  volume_state[CYCLES_WEBGPU_VOLUME_NEXT_BOUNDARY * capacity + lane] = next_state;
  if (init) {
    volume_state[CYCLES_WEBGPU_VOLUME_BOUNCE * capacity + lane] = 0u;
  }
#  else
  volume_state[lane] = next_state;
  const uint weight = path + CYCLES_WEBGPU_PATH_THROUGHPUT;
  const float3 throughput = __builtin_bit_cast(
      float3, uint3(path_words[weight], path_words[weight + 1], path_words[weight + 2]));
  const uint3 next = __builtin_bit_cast(uint3, throughput * segment.xyz);
  path_words[weight] = next.x;
  path_words[weight + 1] = next.y;
  path_words[weight + 2] = next.z;
#  endif
  hit_words[h + CYCLES_WEBGPU_HIT_SPECIAL] = uint(segment.w);
#endif
}
