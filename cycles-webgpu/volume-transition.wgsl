// SPDX-License-Identifier: Apache-2.0
// Cycles WebGPU linear workgroup grid

@group(0) @binding(0) var<storage, read> rays: array<u32>;
@group(0) @binding(1) var<storage, read> hits: array<u32>;
@group(0) @binding(2) var<storage, read> paths: array<u32>;
@group(0) @binding(3) var<storage, read_write> volume_state: array<u32>;

// Match path_state_layout.h, volume_layout.h and the stock ray visibility.
const path_words = 24u;
const queued_kernel = 7u;
const path_flags = 14u;
const intersect_closest = 3u;
const hit_special = 7u;
const mixed_volume_boundary = 5u;
const transmit_visibility = 2u;
const transparent_flag = 4u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
  let index = global_id.x + global_id.y * groups.x * 64u;
  let capacity = rays[0];
  let queue = 1u + 12u * capacity;
  if (index >= rays[queue]) {
    return;
  }

  let lane = rays[queue + 1u + index];
  let visibility = rays[1u + 12u * lane + 8u];
  let path = lane * path_words;
  if (hits[8u * lane + hit_special] == mixed_volume_boundary &&
      paths[path + queued_kernel] == intersect_closest &&
      ((visibility & transmit_visibility) != 0u ||
       (paths[path + path_flags] & transparent_flag) != 0u)) {
    // The interval stage prepared this candidate from the incident stack.
    // The stock path update exposes LABEL_TRANSMIT as ray visibility.
    volume_state[lane] = volume_state[capacity + lane];
  }
}

// Cycles WebGPU render buffer ABI 2
