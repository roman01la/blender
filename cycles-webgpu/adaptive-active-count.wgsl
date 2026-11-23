// SPDX-License-Identifier: Apache-2.0

@group(0) @binding(0) var<storage, read> film: array<u32>;
@group(0) @binding(1) var<storage, read> params: array<u32>;
@group(0) @binding(2) var<storage, read_write> active_pixels: atomic<u32>;

var<workgroup> workgroup_count: atomic<u32>;

// The stock convergence kernel returns the number of unconverged pixels
// before the X/Y filters reactivate their neighbors.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) local: u32) {
  if (id.x < params[2] && id.y < params[3]) {
    let index = (params[4] + id.x + id.y * params[5]) * params[6] + params[9] + 3u;
    if (bitcast<f32>(film[index]) == 0.0) {
      atomicAdd(&workgroup_count, 1u);
    }
  }
  workgroupBarrier();
  if (local == 0u) {
    atomicAdd(&active_pixels, atomicLoad(&workgroup_count));
  }
}
