// SPDX-License-Identifier: Apache-2.0
// Cycles WebGPU linear workgroup grid

@group(0) @binding(0) var<storage, read_write> rays: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> paths: array<u32>;
@group(0) @binding(2) var<storage, read> points: array<u32>;
@group(0) @binding(3) var<storage, read> metadata: array<u32>;
@group(0) @binding(4) var<storage, read_write> scratch: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> materials: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> dispatch_grid: array<u32>;

const path_words = 24u;
const path_bounce = 8u;
const path_flags = 14u;
const path_surface_rr = 21u;
const path_camera_or_subsurface = (1u << 9u) | (7u << 14u);
// SD_HAS_TRANSPARENT_SHADOW, SD_HAS_VOLUME, SD_HAS_ONLY_VOLUME and SD_HAS_EMISSION.
// SD_HAS_CONSTANT_EMISSION also describes shaders whose emitted color is zero.
const preserve_terminal_surface = (7u << 17u) | (1u << 29u);
const shader_id_mask = 0x203fffffu;
const material_vectors = 7u;
const material_unshaded = 2.0;
const workgroup_size_x = 64u;
const kept_count = 0u;
const original_count = 1u;
const original_paths = 2u;

var<workgroup> group_count: atomic<u32>;
var<workgroup> group_offset: u32;

fn ray_capacity() -> u32 {
  return atomicLoad(&rays[0]);
}

fn queue_offset(capacity: u32) -> u32 {
  return 1u + 12u * capacity;
}

fn work_index(id: vec3<u32>, groups: vec3<u32>) -> u32 {
  return id.x + id.y * groups.x * workgroup_size_x;
}

fn skip_surface(lane: u32, capacity: u32) -> bool {
  let path = lane * path_words;
  if (bitcast<f32>(paths[path + path_surface_rr]) != 0.0 ||
      paths[path + path_bounce] == 0u ||
      (paths[path + path_flags] & path_camera_or_subsurface) != 0u) {
    return false;
  }
  let hit = 8u * lane;
  let status = points[hit + 3u];
  let special = points[8u * capacity + hit + 7u];
  if ((status != 1u && status != 2u) || special != 0u) {
    return false;
  }
  let shader = points[hit + 7u] & shader_id_mask;
  return (metadata[metadata[4u] + shader] & preserve_terminal_surface) == 0u;
}

@compute @workgroup_size(1)
fn reset_surface_roulette() {
  atomicStore(&scratch[kept_count], 0u);
  atomicStore(&scratch[original_count], atomicLoad(&rays[queue_offset(ray_capacity())]));
}

@compute @workgroup_size(workgroup_size_x)
fn filter_surface_roulette(@builtin(global_invocation_id) id: vec3<u32>,
                           @builtin(num_workgroups) groups: vec3<u32>,
                           @builtin(local_invocation_index) local: u32) {
  let capacity = ray_capacity();
  let queue = queue_offset(capacity);
  let index = work_index(id, groups);
  var keep = false;
  var lane = 0u;
  var rank = 0u;
  if (index < atomicLoad(&scratch[original_count])) {
    lane = atomicLoad(&rays[queue + 1u + index]);
    atomicStore(&scratch[original_paths + index], lane);
    keep = !skip_surface(lane, capacity);
    if (keep) {
      rank = atomicAdd(&group_count, 1u);
    } else {
      // Downstream forward lamps, light passes, and film still visit these paths.
      let material = material_vectors * lane;
      materials[material] = vec4<f32>(0.0, 0.0, 0.0, material_unshaded);
      for (var i = 1u; i < material_vectors; i++) {
        materials[material + i] = vec4<f32>(0.0);
      }
    }
  }

  workgroupBarrier();
  if (local == 0u) {
    let count = atomicLoad(&group_count);
    if (count != 0u) {
      group_offset = atomicAdd(&scratch[kept_count], count);
    }
  }
  workgroupBarrier();
  if (keep) {
    atomicStore(&scratch[original_paths + capacity + group_offset + rank], lane);
  }
}

@compute @workgroup_size(workgroup_size_x)
fn publish_surface_roulette(@builtin(global_invocation_id) id: vec3<u32>,
                            @builtin(num_workgroups) groups: vec3<u32>) {
  let capacity = ray_capacity();
  let queue = queue_offset(capacity);
  let index = work_index(id, groups);
  let count = atomicLoad(&scratch[kept_count]);
  if (index < count) {
    let lane = atomicLoad(&scratch[original_paths + capacity + index]);
    atomicStore(&rays[queue + 1u + index], lane);
  }
  if (index == 0u) {
    atomicStore(&rays[queue], count);
    // Same balanced 2D grid used by the device queue.
    let width = dispatch_grid[4];
    let limit = dispatch_grid[3];
    let total = (count + width - 1u) / width;
    let rows = max(1u, (total + limit - 1u) / limit);
    dispatch_grid[0] = (total + rows - 1u) / rows;
    dispatch_grid[1] = rows;
    dispatch_grid[2] = 1u;
  }
}

@compute @workgroup_size(workgroup_size_x)
fn restore_surface_roulette(@builtin(global_invocation_id) id: vec3<u32>,
                            @builtin(num_workgroups) groups: vec3<u32>) {
  let queue = queue_offset(ray_capacity());
  let index = work_index(id, groups);
  let count = atomicLoad(&scratch[original_count]);
  if (index < count) {
    atomicStore(&rays[queue + 1u + index], atomicLoad(&scratch[original_paths + index]));
  }
  if (index == 0u) {
    atomicStore(&rays[queue], count);
  }
}
