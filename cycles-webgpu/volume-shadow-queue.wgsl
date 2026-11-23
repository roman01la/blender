// SPDX-License-Identifier: Apache-2.0
// Cycles WebGPU linear workgroup grid

@group(0) @binding(0) var<storage, read_write> rays: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> shadow_words: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> indirect: array<atomic<u32>, 4>;
@group(0) @binding(3) var<storage, read_write> guides: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> surface: array<u32>;

const group_size = 64u;
const from_volume_flag = 1u << 30u;
const volume_shadow_params = 11u;
const camera_words = 60u;
const pass_unused = 0xffffffffu;

var<workgroup> compact_count: atomic<u32>;
var<workgroup> compact_base: u32;

fn main_queue_base() -> u32 {
  return 1u + 12u * atomicLoad(&rays[0]);
}

// The second queue follows the longest primary ray layout (queue, optional
// differentials and volume MIS distances). Keep this in sync with the host.
fn shadow_queue_base() -> u32 {
  return 2u + 17u * atomicLoad(&rays[0]);
}

@compute @workgroup_size(1)
fn reset_shadow_queue() {
  atomicStore(&rays[shadow_queue_base()], 0u);
  atomicStore(&indirect[0], 0u);
  atomicStore(&indirect[1], 1u);
  atomicStore(&indirect[2], 1u);
}

// The shadow producer updates guides even when it generates no light ray.
// Keep the stock Cycles film_write_denoising_features_volume policy here,
// using the same path-local throughput and output layout as absorption_dispatch.
fn accumulate_volume_guide(lane: u32) {
  let params = surface[volume_shadow_params] + camera_words;
  let write_albedo = surface[params] != pass_unused;
  let write_normal = surface[params + 1u] != pass_unused;
  if (!write_albedo && !write_normal) {
    return;
  }

  let guide = bitcast<vec4<f32>>(shadow_words[4u * lane + 3u]);
  let index = 3u * lane;
  let throughput = guides[index].xyz;
  if (!any(guide != vec4<f32>(0.0)) || !any(throughput != vec3<f32>(0.0))) {
    return;
  }

  if (guide.w != 0.0 && write_normal) {
    guides[index] = vec4<f32>(0.0);
    let normal = guides[index + 2u];
    guides[index + 2u] = vec4<f32>(normal.xyz, normal.w - 1.0);
  }
  if (write_albedo) {
    let rgb = throughput * guide.xyz;
    let finite = (bitcast<vec3<u32>>(rgb) & vec3<u32>(0x7f800000u)) != vec3<u32>(0x7f800000u);
    let albedo = guides[index + 1u];
    guides[index + 1u] = vec4<f32>(albedo.xyz + select(vec3<f32>(0.0), rgb, finite), albedo.w);
  }
}

fn compact(candidate: u32, local: u32) {
  let main_queue = main_queue_base();
  var lane = 0u;
  var index = 0u;
  var pending = false;
  if (candidate < atomicLoad(&rays[main_queue])) {
    lane = atomicLoad(&rays[main_queue + 1u + candidate]);
    let options = shadow_words[4u * lane + 2u].x;
    pending = (options & ~from_volume_flag) != 0u;
    if (pending) {
      index = atomicAdd(&compact_count, 1u);
    }
  }
  workgroupBarrier();

  // As in Cycles' active-path compaction, reserve one contiguous range per
  // nonempty workgroup and preserve the physical path slot for every ray.
  let queue = shadow_queue_base();
  if (local == 0u) {
    let count = atomicLoad(&compact_count);
    if (count != 0u) {
      compact_base = atomicAdd(&rays[queue], count);
    }
  }
  workgroupBarrier();

  if (pending) {
    atomicStore(&rays[queue + 1u + compact_base + index], lane);
  }
}

// The fourth word is the negotiated limit and is not part of the indirect tuple.
@compute @workgroup_size(1)
fn finish_shadow_grid() {
  let count = atomicLoad(&rays[shadow_queue_base()]);
  let groups = (count + group_size - 1u) / group_size;
  let limit = max(atomicLoad(&indirect[3]), 1u);
  let rows = max((groups + limit - 1u) / limit, 1u);
  atomicStore(&indirect[0], (groups + rows - 1u) / rows);
  atomicStore(&indirect[1], rows);
}

@compute @workgroup_size(64)
fn compact_shadow_rays(@builtin(global_invocation_id) global_id: vec3<u32>,
                       @builtin(num_workgroups) groups: vec3<u32>,
                       @builtin(local_invocation_index) local: u32) {
  let index = global_id.x + global_id.y * groups.x * 64u;
  compact(index, local);
}

@compute @workgroup_size(64)
fn compact_guided_volume_shadow_rays(@builtin(global_invocation_id) global_id: vec3<u32>,
                                     @builtin(num_workgroups) groups: vec3<u32>,
                                     @builtin(local_invocation_index) local: u32) {
  let index = global_id.x + global_id.y * groups.x * 64u;
  let main_queue = main_queue_base();
  if (index < atomicLoad(&rays[main_queue])) {
    let lane = atomicLoad(&rays[main_queue + 1u + index]);
    if ((shadow_words[4u * lane + 2u].x & from_volume_flag) != 0u) {
      accumulate_volume_guide(lane);
    }
  }
  compact(index, local);
}

// Cycles WebGPU render buffer ABI 2
