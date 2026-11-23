// SPDX-FileCopyrightText: 2021-2022 Blender Foundation
// SPDX-License-Identifier: Apache-2.0
// Cycles WebGPU linear workgroup grid

// Adapt the partition bucket counts, prefix offsets and atomic scatter from
// Cycles' kernel/device/gpu/parallel_sorted_index.h. The input is already a
// dense queue of active surface paths, so its partition offset is just its
// start index. Keep a local copy while rewriting the original queue.
@group(0) @binding(0) var<storage, read_write> rays: array<u32>;
@group(0) @binding(1) var<storage, read> surfaces: array<u32>;
@group(0) @binding(2) var<storage, read> svm: array<u32>;

const partition_size = 2048u;
const group_size = 256u;
const max_buckets = 1024u;

var<workgroup> queue_paths: array<u32, partition_size>;
var<workgroup> key_offsets: array<atomic<u32>, max_buckets>;

fn shader_key(path: u32, num_keys: u32) -> u32 {
  return (surfaces[8u * path + 7u] & 0x203fffffu) % num_keys;
}

@compute @workgroup_size(group_size)
fn sort_surface_paths(@builtin(workgroup_id) group: vec3<u32>,
                     @builtin(num_workgroups) groups: vec3<u32>,
                      @builtin(local_invocation_index) local: u32) {
  let queue = 1u + 12u * rays[0];
  let count = rays[queue];
  let num_keys = clamp(svm[1] & 0x003fffffu, 1u, max_buckets);
  let start = partition_size * (group.x + group.y * groups.x);
  let size = min(count - min(count, start), partition_size);

  // Count each shader in this partition. Only the positions in the queue move.
  for (var index = local; index < size; index += group_size) {
    let path = rays[queue + 1u + start + index];
    queue_paths[index] = path;
    atomicAdd(&key_offsets[shader_key(path, num_keys)], 1u);
  }
  workgroupBarrier();

  if (local == 0u) {
    var offset = 0u;
    for (var key = 0u; key < num_keys; key += 1u) {
      let bucket = atomicLoad(&key_offsets[key]);
      atomicStore(&key_offsets[key], offset);
      offset += bucket;
    }
  }
  workgroupBarrier();

  for (var index = local; index < size; index += group_size) {
    let path = queue_paths[index];
    let sorted = atomicAdd(&key_offsets[shader_key(path, num_keys)], 1u);
    rays[queue + 1u + start + sorted] = path;
  }
}
