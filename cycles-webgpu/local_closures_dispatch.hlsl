/* SPDX-License-Identifier: Apache-2.0 */

StructuredBuffer<uint> svm_words : register(t0);
StructuredBuffer<uint> surface_words : register(t1);
RWStructuredBuffer<uint> ray_words : register(u2);
StructuredBuffer<uint> hit_words : register(t3);
RWStructuredBuffer<uint> path_words : register(u4);
cbuffer CameraRows : register(b5) {
  uint4 camera_rows[15];
};
StructuredBuffer<uint> metadata_words : register(t6);
#ifdef CYCLES_WEBGPU_DEFER_SHADOW
RWStructuredBuffer<uint4> shadow_words : register(u7);
#else
StructuredBuffer<uint> scene_words : register(t7);
#endif
StructuredBuffer<uint> light_words : register(t8);
RWStructuredBuffer<float4> radiance : register(u9);
StructuredBuffer<uint> camera_sampling : register(t10);

static float4 local_closure_result;
static float4 local_closure_specular;
static float4 local_closure_layers;
static float4 local_closure_glass;
static float4 local_closure_subsurface;
static float4 local_closure_emission;
static float4 local_closure_closure_normal;

#include "svm_dispatch.hlsl"
#define samples radiance
#include "direct_light_dispatch.hlsl"
#undef samples
#undef CYCLES_WEBGPU_LIGHT_PASS_BUFFER
#include "path_update_dispatch.hlsl"

[numthreads(576, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
  uint lane;
  if (cycles_queue_slot(id.x, lane)) {
    cycles_local_material(lane);
    cycles_local_light(lane);
    cycles_local_continue(lane);
  }
}
