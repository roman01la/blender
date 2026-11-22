// SPDX-License-Identifier: Apache-2.0
// Cycles WebGPU volume scatter classification

@group(0) @binding(0) var<storage, read_write> paths: array<u32>;
@group(0) @binding(1) var<storage, read_write> rays: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> hits: array<u32>;
@group(0) @binding(4) var<storage, read> pixels: array<vec4<u32>>;

// Path-state entries store the Cycles DeviceKernel index plus one. Zero means terminated.
const path_words = 24u;
const queued_kernel = 7u;
const intersect_closest = 3u;
const intersect_subsurface = 5u;
const shade_background = 9u;
const shade_surface = 12u;
// CYCLES_WEBGPU_HIT_SPECIAL and CYCLES_WEBGPU_VOLUME_SCATTER_HIT.
const hit_special = 7u;
const volume_scatter_hit = 4u;

// Match the C++ IntegratorQueueCounter, then extend it with the live sample range.
const queue_kernels = 18u;
const min_sample_word = 19u;
const max_sample_word = 20u;

var<workgroup> group_counts: array<atomic<u32>, queue_kernels>;
var<workgroup> group_min_sample: atomic<u32>;
var<workgroup> group_max_sample: atomic<u32>;
var<workgroup> compact_count: atomic<u32>;
var<workgroup> compact_base: u32;

fn capacity() -> u32 {
  return atomicLoad(&rays[0]);
}

fn queue_offset() -> u32 {
  return 1u + 12u * capacity();
}

@compute @workgroup_size(1)
fn set_capacity() {
  atomicStore(&rays[0], arrayLength(&paths) / path_words);
}

@compute @workgroup_size(1)
fn reset_queue() {
  atomicStore(&rays[queue_offset()], 0u);
}

fn compact(lane: u32, kernel: u32, local: u32) {
  let queued = lane < capacity() && paths[lane * path_words + queued_kernel] == kernel;
  var index = 0u;
  if (queued) {
    index = atomicAdd(&compact_count, 1u);
  }
  workgroupBarrier();

  // Reserve one contiguous range per nonempty workgroup in the device queue.
  let queue = queue_offset();
  if (local == 0u) {
    let count = atomicLoad(&compact_count);
    if (count != 0u) {
      compact_base = atomicAdd(&rays[queue], count);
    }
  }
  workgroupBarrier();

  if (queued) {
    atomicStore(&rays[queue + 1u + compact_base + index], lane);
  }
}

@compute @workgroup_size(64)
fn compact_free(@builtin(global_invocation_id) id: vec3<u32>,
                @builtin(local_invocation_index) local: u32) {
  compact(id.x, 0u, local);
}

@compute @workgroup_size(64)
fn compact_intersect(@builtin(global_invocation_id) id: vec3<u32>,
                     @builtin(local_invocation_index) local: u32) {
  compact(id.x, intersect_closest, local);
}

@compute @workgroup_size(64)
fn compact_surface(@builtin(global_invocation_id) id: vec3<u32>,
                   @builtin(local_invocation_index) local: u32) {
  compact(id.x, shade_surface, local);
}

@compute @workgroup_size(64)
fn compact_subsurface(@builtin(global_invocation_id) id: vec3<u32>,
                      @builtin(local_invocation_index) local: u32) {
  compact(id.x, intersect_subsurface, local);
}

@compute @workgroup_size(64)
fn compact_background(@builtin(global_invocation_id) id: vec3<u32>,
                      @builtin(local_invocation_index) local: u32) {
  compact(id.x, shade_background, local);
}

@compute @workgroup_size(64)
fn classify_intersection(@builtin(global_invocation_id) id: vec3<u32>) {
  let queue = queue_offset();
  if (id.x < atomicLoad(&rays[queue])) {
    let lane = atomicLoad(&rays[queue + 1u + id.x]);
    let hit = 8u * lane;
    if (hits[hit + hit_special] != volume_scatter_hit) {
      paths[lane * path_words + queued_kernel] =
          select(shade_background, shade_surface, hits[hit] != 0u);
    }
  }
}

@compute @workgroup_size(64)
fn count_queues(@builtin(global_invocation_id) id: vec3<u32>,
                @builtin(local_invocation_index) local: u32) {
  // Reduce within the workgroup to avoid one global atomic per live path.
  if (id.x < capacity()) {
    let base = id.x * path_words;
    let kernel = paths[base + queued_kernel];
    if (kernel != 0u && kernel <= queue_kernels) {
      let sample = pixels[id.x].w;
      atomicAdd(&group_counts[kernel - 1u], 1u);
      atomicMax(&group_min_sample, ~sample);
      atomicMax(&group_max_sample, sample);
    }
  }
  workgroupBarrier();
  if (local < queue_kernels) {
    let count = atomicLoad(&group_counts[local]);
    if (count != 0u) {
      atomicAdd(&counts[local], count);
    }
  }
  if (local == 0u) {
    atomicMax(&counts[min_sample_word], atomicLoad(&group_min_sample));
    atomicMax(&counts[max_sample_word], atomicLoad(&group_max_sample));
  }
}
