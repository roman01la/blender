// SPDX-License-Identifier: Apache-2.0
// Cycles WebGPU linear workgroup grid

@group(0) @binding(0) var<storage, read> rays: array<u32>;
@group(0) @binding(1) var<storage, read_write> path_state: array<u32>;
@group(0) @binding(2) var<storage, read_write> pixels: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> work: array<u32>;
@group(0) @binding(4) var<storage, read_write> film: array<atomic<u32>>;

struct RenderBuffer {
  bounds: vec4<u32>,
  placement: vec4<u32>,
  sample_count_offset: u32,
  adaptive_aux_offset: u32,
  sampling_pattern: u32,
  exposure: f32,
}

@group(0) @binding(5) var<uniform> render_buffer: RenderBuffer;
@group(0) @binding(6) var<storage, read_write> denoising: array<vec4<f32>>;

// Keep the slot layout in sync with path_state_layout.h.
const path_words = 24u;
const path_throughput = 0u;
const path_radiance = 3u;
const path_queued_kernel = 7u;
const path_bounce = 8u;
const path_min_ray_pdf = 13u;
const path_flags = 14u;
const path_pixel = 18u;
const path_surface_rr = 21u;
const intersect_closest = 3u; // DeviceKernel index + 1

fn initialize_path(index: u32) -> u32 {
  let queue = 1u + 12u * rays[0];
  // The last workgroup may extend into free slots that were not admitted.
  if (index >= work[2] || index >= rays[queue]) {
    return 0xffffffffu;
  }

  let lane = rays[queue + 1u + index];
  let pixel = pixels[lane];
  if (pixel.w == 0xffffffffu) {
    return 0xffffffffu;
  }
  let base = lane * path_words;
  let one = bitcast<u32>(1.0f);
  path_state[base + path_throughput] = one;
  path_state[base + path_throughput + 1u] = one;
  path_state[base + path_throughput + 2u] = one;
  path_state[base + path_radiance] = 0u;
  path_state[base + path_radiance + 1u] = 0u;
  path_state[base + path_radiance + 2u] = 0u;
  path_state[base + path_radiance + 3u] = one;
  path_state[base + path_queued_kernel] = intersect_closest;
  for (var i = path_bounce; i < path_words; i += 1u) {
    path_state[base + i] = 0u;
  }
  path_state[base + path_min_ray_pdf] = 0x7f7fffffu; // FLT_MAX
  path_state[base + path_flags] = (1u << 7u) | (1u << 9u); // MIS_SKIP | TRANSPARENT_BACKGROUND
  path_state[base + path_pixel] = pixel.x;
  path_state[base + path_pixel + 1u] = pixel.y;
  path_state[base + path_pixel + 2u] = pixel.z;
  path_state[base + path_surface_rr] = one;
  return lane;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
  let index = global_id.x + global_id.y * groups.x * 64u;
  _ = initialize_path(index);
}

@compute @workgroup_size(64)
fn initialize_guided(@builtin(global_invocation_id) global_id: vec3<u32>,
                     @builtin(num_workgroups) groups: vec3<u32>) {
  let index = global_id.x + global_id.y * groups.x * 64u;
  let lane = initialize_path(index);
  if (lane == 0xffffffffu) {
    return;
  }
  // Cycles' camera seeds feature throughput independently of light throughput.
  let feature = lane * 3u;
  denoising[feature] = vec4<f32>(1.0);
  denoising[feature + 1u] = vec4<f32>(0.0);
  denoising[feature + 2u] = vec4<f32>(0.0);
}

// The storage adapter for Cycles' camera film_need_sample_pixel/film_write_sample.
// Claim a contiguous effective sample ID per pixel before generating the ray.
@compute @workgroup_size(64)
fn admit_adaptive_pixels(@builtin(global_invocation_id) global_id: vec3<u32>,
                         @builtin(num_workgroups) groups: vec3<u32>) {
  let index = global_id.x + global_id.y * groups.x * 64u;
  let queue = 1u + 12u * rays[0];
  if (index >= work[2] || index >= rays[queue]) {
    return;
  }

  let lane = rays[queue + 1u + index];
  let pixel = pixels[lane];
  let local = pixel.xy - render_buffer.bounds.xy;
  let pixel_index = render_buffer.placement.x + local.y * render_buffer.placement.y + local.x;
  let base = pixel_index * render_buffer.placement.z;
  if (bitcast<f32>(atomicLoad(&film[base + render_buffer.adaptive_aux_offset + 3u])) != 0.0) {
    pixels[lane] = vec4<u32>(pixel.xyz, 0xffffffffu);
    return;
  }

  let sample = atomicAdd(&film[base + render_buffer.sample_count_offset], 1u);
  pixels[lane] = vec4<u32>(pixel.xy, sample, pixel.w);
}
