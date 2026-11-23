// SPDX-License-Identifier: Apache-2.0
// Cycles WebGPU volume scatter classification
// Cycles WebGPU linear workgroup grid
// Cycles WebGPU queue transitions

@group(0) @binding(0) var<storage, read_write> paths: array<u32>;
@group(0) @binding(1) var<storage, read_write> rays: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> hits: array<u32>;
@group(0) @binding(4) var<storage, read> pixels: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read> work: array<u32>;

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
// Keep the delta snapshot in sync with WebGPUWavefrontQueue. Slot 18
// is cache_miss, and words 19 and 20 are used by the full-scan fallback.
const sample_delta_base = 21u;
const sample_bins = 64u;
const delta_work_word = sample_delta_base + sample_bins;
const delta_error_word = delta_work_word + 1u;

var<workgroup> group_counts: array<atomic<u32>, queue_kernels>;
var<workgroup> group_min_sample: atomic<u32>;
var<workgroup> group_sample_delta: array<atomic<u32>, sample_bins>;
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
fn compact_free(@builtin(global_invocation_id) global_id: vec3<u32>,
                @builtin(num_workgroups) groups: vec3<u32>,
                @builtin(local_invocation_index) local: u32) {
  let index = global_id.x + global_id.y * groups.x * 64u;
  compact(index, 0u, local);
}

@compute @workgroup_size(64)
fn compact_intersect(@builtin(global_invocation_id) global_id: vec3<u32>,
                     @builtin(num_workgroups) groups: vec3<u32>,
                     @builtin(local_invocation_index) local: u32) {
  let index = global_id.x + global_id.y * groups.x * 64u;
  compact(index, intersect_closest, local);
}

@compute @workgroup_size(64)
fn compact_surface(@builtin(global_invocation_id) global_id: vec3<u32>,
                   @builtin(num_workgroups) groups: vec3<u32>,
                   @builtin(local_invocation_index) local: u32) {
  let index = global_id.x + global_id.y * groups.x * 64u;
  compact(index, shade_surface, local);
}

@compute @workgroup_size(64)
fn compact_subsurface(@builtin(global_invocation_id) global_id: vec3<u32>,
                      @builtin(num_workgroups) groups: vec3<u32>,
                      @builtin(local_invocation_index) local: u32) {
  let index = global_id.x + global_id.y * groups.x * 64u;
  compact(index, intersect_subsurface, local);
}

@compute @workgroup_size(64)
fn compact_background(@builtin(global_invocation_id) global_id: vec3<u32>,
                      @builtin(num_workgroups) groups: vec3<u32>,
                      @builtin(local_invocation_index) local: u32) {
  let index = global_id.x + global_id.y * groups.x * 64u;
  compact(index, shade_background, local);
}

@compute @workgroup_size(64)
fn classify_intersection(@builtin(global_invocation_id) global_id: vec3<u32>,
                         @builtin(num_workgroups) groups: vec3<u32>) {
  let index = global_id.x + global_id.y * groups.x * 64u;
  let queue = queue_offset();
  if (index < atomicLoad(&rays[queue])) {
    let lane = atomicLoad(&rays[queue + 1u + index]);
    let hit = 8u * lane;
    if (hits[hit + hit_special] != volume_scatter_hit) {
      paths[lane * path_words + queued_kernel] =
          select(shade_background, shade_surface, hits[hit] != 0u);
    }
  }
}

@compute @workgroup_size(64)
fn count_queues(@builtin(global_invocation_id) global_id: vec3<u32>,
                @builtin(num_workgroups) groups: vec3<u32>,
                @builtin(local_invocation_index) local: u32) {
  let index = global_id.x + global_id.y * groups.x * 64u;
  // Reduce within the workgroup to avoid one global atomic per live path.
  if (index < capacity()) {
    let base = index * path_words;
    let kernel = paths[base + queued_kernel];
    if (kernel != 0u && kernel <= queue_kernels) {
      let sample = pixels[index].w;
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

// The completed stage operated on this list after compaction. Other paths
// retain their previously counted state, as in Cycles' path transition counters.
fn count_changed_paths(index: u32, local: u32, count: u32, admission: bool) {
  if (index < count) {
    let queue = queue_offset();
    let lane = atomicLoad(&rays[queue + 1u + index]);
    if (lane < capacity()) {
      let kernel = paths[lane * path_words + queued_kernel];
      if (kernel != 0u && kernel <= queue_kernels) {
        atomicAdd(&group_counts[kernel - 1u], 1u);
      } else if (kernel > queue_kernels) {
        atomicStore(&counts[delta_error_word], 1u);
      }
      // Adaptive admission can leave a pixel unclaimed. Only a newly live
      // or a retired path changes the per-sample occupancy.
      if (admission == (kernel != 0u)) {
        let sample = pixels[lane].w;
        if (sample != 0xffffffffu) {
          atomicAdd(&group_sample_delta[sample % sample_bins], 1u);
        } else {
          atomicStore(&counts[delta_error_word], 1u);
        }
      }
    } else {
      atomicStore(&counts[delta_error_word], 1u);
    }
  }
  workgroupBarrier();

  if (local < queue_kernels) {
    let queued = atomicLoad(&group_counts[local]);
    if (queued != 0u) {
      atomicAdd(&counts[local], queued);
    }
  }
  let changed = atomicLoad(&group_sample_delta[local]);
  if (changed != 0u) {
    atomicAdd(&counts[sample_delta_base + local], changed);
  }
  if (index == 0u) {
    atomicStore(&counts[delta_work_word], count);
  }
}

@compute @workgroup_size(64)
fn count_queue_delta(@builtin(global_invocation_id) global_id: vec3<u32>,
                     @builtin(num_workgroups) groups: vec3<u32>,
                     @builtin(local_invocation_index) local: u32) {
  let index = global_id.x + global_id.y * groups.x * 64u;
  count_changed_paths(index, local, atomicLoad(&rays[queue_offset()]), false);
}

@compute @workgroup_size(64)
fn count_admitted_delta(@builtin(global_invocation_id) global_id: vec3<u32>,
                        @builtin(num_workgroups) groups: vec3<u32>,
                        @builtin(local_invocation_index) local: u32) {
  let index = global_id.x + global_id.y * groups.x * 64u;
  let admitted = min(work[2u], atomicLoad(&rays[queue_offset()]));
  count_changed_paths(index, local, admitted, true);
}
