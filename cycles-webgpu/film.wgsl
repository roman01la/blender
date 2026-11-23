// SPDX-License-Identifier: Apache-2.0
// Cycles WebGPU volume geometry passes
// Cycles WebGPU linear workgroup grid

struct FilmParameters {
  bounds: vec4<u32>, // Absolute raster origin XY, width, height.
  ray_count: u32,
  exposure_multiplier: f32,
  reserved: vec2<u32>,
}

struct RenderBufferParameters {
  bounds: vec4<u32>, // Absolute raster origin XY, width, height of this tile.
  placement: vec4<u32>, // Pixel base, row stride, pass stride, Combined offset.
  sample_count_offset: u32,
  adaptive_aux_offset: u32,
  sampling_pattern: u32,
  exposure: f32,
}

@group(0) @binding(0) var<uniform> params: FilmParameters;
@group(0) @binding(1) var<storage, read> pixel_samples: array<vec4<u32>>;
// Each ray contributes one premultiplied scene-linear RGBA sample.
@group(0) @binding(2) var<storage, read> samples: array<vec4<f32>>;
// Each pixel stores four floating-point sums as u32 bits and an integer sample count.
// Allocate or clear this buffer before the first submission for a film.
@group(0) @binding(3) var<storage, read_write> film: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> resolved: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> completed_paths: array<u32>;
@group(0) @binding(6) var<storage, read> retire_rays: array<u32>;
@group(0) @binding(7) var<uniform> render_buffer: RenderBufferParameters;
@group(0) @binding(8) var<storage, read> adaptive_deltas: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read> denoising: array<vec4<f32>>;

struct DenoisingParameters {
  albedo_offset: u32,
  normal_offset: u32,
  options: u32,
  reserved: u32,
  world_to_camera_x: vec4<f32>,
  world_to_camera_y: vec4<f32>,
  world_to_camera_z: vec4<f32>,
}
@group(0) @binding(10) var<storage, read> denoising_params: DenoisingParameters;

struct AOVParameters {
  color: vec2<u32>, // First film offset and number of color passes.
  value: vec2<u32>, // First film offset and number of value passes.
}
@group(0) @binding(11) var<storage, read> aov_params: AOVParameters;

struct GeometryPassParameters {
  offsets: vec4<u32>, // Depth, Object Index, Material Index, Position.
  world_to_camera_z: vec4<f32>,
}
@group(0) @binding(12) var<storage, read> shaded_points: array<vec4<u32>>;
@group(0) @binding(13) var<storage, read> intersections: array<vec4<u32>>;
@group(0) @binding(14) var<storage, read> surface_metadata: array<u32>;
@group(0) @binding(15) var<uniform> geometry_params: GeometryPassParameters;
@group(0) @binding(16) var<uniform> emission_pass: vec4<u32>; // Emission offset in X.
struct SurfacePassParameters {
  offsets: vec4<u32>, // Normal, Diffuse Color, Glossy Color, Transmission Color.
  ao: vec4<u32>, // Pass offset and integration distance.
}
@group(0) @binding(17) var<uniform> surface_pass_params: SurfacePassParameters;
struct LightPassParameters {
  reflection: vec4<u32>, // Diffuse direct/indirect, glossy direct/indirect.
  transmission: vec4<u32>, // Transmission direct/indirect.
}
@group(0) @binding(18) var<uniform> light_pass_params: LightPassParameters;

const workgroup_size_x = 64u;
const words_per_pixel = 5u;
// Match path_state_layout.h and the compact ray queue layout.
const path_words = 24u;
const path_radiance = 3u;
const path_queued_kernel = 7u;
const path_bounce = 8u;
const path_flags = 14u;
const path_pixel = 18u;
const path_surface_rr = 21u;
const path_terminated = 0u;
const path_shade_background = 9u;
const single_pass_done = 1u << 8u;
const ray_words = 12u;
const material_vectors = 7u;
const transparent_background = 1u << 9u;
const terminate_on_next_surface = 1u << 10u;
// CYCLES_WEBGPU_MIXED_VOLUME_BOUNDARY in volume_layout.h.
const mixed_volume_boundary = 5u;
const pass_unused = 0xffffffffu;

// Match the material layouts packed by scene.cpp.
const material_holdout = 5.0;
const object_table_offset_word = 5u;
const shader_pass_table_offset_word = 6u;
const object_metadata_words = 23u;
const object_pass_id_word = 16u;
const packed_shader_id_mask = 0x203fffffu;

fn add_component(index: u32, value: f32) {
  // Cycles filters non-finite light contributions before writing passes. The bit test
  // also works on implementations that assume finite values in floating-point comparisons.
  if (value == 0.0 || (bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u) {
    return;
  }

  var old_bits = atomicLoad(&film[index]);
  loop {
    let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + value);
    let result = atomicCompareExchangeWeak(&film[index], old_bits, new_bits);
    if (result.exchanged) {
      break;
    }
    old_bits = result.old_value;
  }
}

// Dispatch uses a 1D grid for small films and may extend along Y when the X limit is reached.
// Keep the Z workgroup count at one.
fn work_index(id: vec3<u32>, groups: vec3<u32>) -> u32 {
  return id.y * groups.x * workgroup_size_x + id.x;
}

@compute @workgroup_size(workgroup_size_x)
fn accumulate(@builtin(global_invocation_id) id: vec3<u32>,
              @builtin(num_workgroups) groups: vec3<u32>) {
  let ray = work_index(id, groups);
  if (ray >= params.ray_count || ray >= arrayLength(&pixel_samples) ||
      ray >= arrayLength(&samples)) {
    return;
  }

  let raster = pixel_samples[ray].xy;
  let pixel = raster - params.bounds.xy;
  if (any(raster < params.bounds.xy) || any(pixel >= params.bounds.zw)) {
    return;
  }

  let index = pixel.y * params.bounds.z + pixel.x;
  if (index >= arrayLength(&film) / words_per_pixel) {
    return;
  }

  let offset = index * words_per_pixel;
  let rgba = samples[ray];
  add_component(offset, rgba.r);
  add_component(offset + 1u, rgba.g);
  add_component(offset + 2u, rgba.b);
  add_component(offset + 3u, rgba.a);
  // The sample ID is used upstream for RNG. Each completed ray here adds one sample.
  atomicAdd(&film[offset + 4u], 1u);
}

@compute @workgroup_size(workgroup_size_x)
fn resolve(@builtin(global_invocation_id) id: vec3<u32>,
           @builtin(num_workgroups) groups: vec3<u32>) {
  let pixel = work_index(id, groups);
  if (pixel >= params.bounds.z * params.bounds.w || pixel >= arrayLength(&resolved) ||
      pixel >= arrayLength(&film) / words_per_pixel) {
    return;
  }

  let offset = pixel * words_per_pixel;
  let sample_count = atomicLoad(&film[offset + 4u]);
  if (sample_count == 0u) {
    resolved[pixel] = vec4<f32>(0.0);
    return;
  }

  let sum = bitcast<vec4<f32>>(vec4<u32>(atomicLoad(&film[offset]),
                                        atomicLoad(&film[offset + 1u]),
                                        atomicLoad(&film[offset + 2u]),
                                        atomicLoad(&film[offset + 3u])));
  let mean = sum / f32(sample_count);
  // Exposure applies to RGB only. Blender can apply its view transform to the linear result.
  resolved[pixel] = vec4<f32>(mean.rgb * params.exposure_multiplier, clamp(mean.a, 0.0, 1.0));
}

fn queued_lane(index: u32) -> u32 {
  let capacity = retire_rays[0];
  let queue_base = 1u + ray_words * capacity;
  if (index >= capacity || index >= retire_rays[queue_base]) {
    return pass_unused;
  }
  let lane = retire_rays[queue_base + 1u + index];
  return select(lane, pass_unused, lane >= arrayLength(&completed_paths) / path_words);
}

fn film_pixel_offset(lane: u32) -> u32 {
  let path = lane * path_words;
  let raster = vec2<u32>(completed_paths[path + path_pixel],
                         completed_paths[path + path_pixel + 1u]);
  let local_pixel = raster - render_buffer.bounds.xy;
  if (any(raster < render_buffer.bounds.xy) || any(local_pixel >= render_buffer.bounds.zw)) {
    return pass_unused;
  }

  let pixel = render_buffer.placement.x + local_pixel.y * render_buffer.placement.y +
              local_pixel.x;
  return pixel * render_buffer.placement.z;
}

// The compact queue is kept until all newly terminated paths have retired.
// Return the physical lane and the native Cycles film word for its pixel.
fn retirement(index: u32) -> vec2<u32> {
  let lane = queued_lane(index);
  if (lane == pass_unused ||
      completed_paths[lane * path_words + path_queued_kernel] != path_terminated) {
    return vec2<u32>(pass_unused);
  }
  let pixel = film_pixel_offset(lane);
  return select(vec2<u32>(lane, pixel), vec2<u32>(pass_unused), pixel == pass_unused);
}

@compute @workgroup_size(workgroup_size_x)
fn accumulate_aov(@builtin(global_invocation_id) id: vec3<u32>,
                  @builtin(num_workgroups) groups: vec3<u32>) {
  let lane = queued_lane(work_index(id, groups));
  if (lane == pass_unused) {
    return;
  }
  let path = lane * path_words;
  if (completed_paths[path + path_bounce] != 0u ||
      (completed_paths[path + path_flags] & transparent_background) == 0u) {
    return;
  }
  let pixel = film_pixel_offset(lane);
  if (pixel == pass_unused) {
    return;
  }
  // Binding 1 is the SVM material buffer for this entry point.
  let colors = aov_params.color.y;
  let values = aov_params.value.y;
  let first = material_vectors * retire_rays[0] + lane * (colors + values);
  for (var i = 0u; i < colors; i++) {
    let dest = pixel + aov_params.color.x + 4u * i;
    let rgba = bitcast<vec4<f32>>(pixel_samples[first + i]);
    add_component(dest, rgba.x);
    add_component(dest + 1u, rgba.y);
    add_component(dest + 2u, rgba.z);
    add_component(dest + 3u, rgba.w);
  }
  for (var i = 0u; i < values; i++) {
    let dest = pixel + aov_params.value.x + i;
    add_component(dest, bitcast<f32>(pixel_samples[first + colors + i].x));
  }
}

// Cycles' film_write_data_passes overwrites these unfiltered data passes once
// from sample zero. A background miss leaves the cleared pass values intact.
@compute @workgroup_size(workgroup_size_x)
fn write_geometry_passes(@builtin(global_invocation_id) id: vec3<u32>,
                          @builtin(num_workgroups) groups: vec3<u32>) {
  let lane = queued_lane(work_index(id, groups));
  if (lane == pass_unused) {
    return;
  }
  let path = lane * path_words;
  if (completed_paths[path + path_pixel + 2u] != 0u ||
      completed_paths[path + path_bounce] != 0u ||
      (completed_paths[path + path_flags] & (transparent_background | single_pass_done)) !=
          transparent_background) {
    return;
  }

  let point = shaded_points[2u * lane];
  let hit = intersections[2u * lane];
  let special_hit = intersections[2u * lane + 1u].w;
  let pixel = film_pixel_offset(lane);
  if (pixel == pass_unused) {
    return;
  }
  let offsets = geometry_params.offsets;
  if (completed_paths[path + path_queued_kernel] == path_shade_background) {
    // Stock background data passes clear a sample-zero hit rejected by the alpha threshold.
    for (var i = 0u; i < 4u; i++) {
      if (offsets[i] != pass_unused) {
        atomicStore(&film[pixel + offsets[i]], 0u);
      }
    }
    if (offsets.w != pass_unused) {
      atomicStore(&film[pixel + offsets.w + 1u], 0u);
      atomicStore(&film[pixel + offsets.w + 2u], 0u);
    }
    return;
  }
  // The material buffer shares binding 1 with the AOV kernel. Stock Cycles
  // terminates a full Holdout before it can write data passes.
  let material = bitcast<vec4<f32>>(pixel_samples[material_vectors * lane]);
  // Mixed surface/volume boundaries still shade the surface. Volume-only boundaries
  // and actual scattering events do not invoke stock film_write_data_passes().
  if ((point.w != 1u && point.w != 2u) ||
      (special_hit != 0u && special_hit != mixed_volume_boundary) ||
      material.w == material_holdout ||
      (completed_paths[path + path_flags] & terminate_on_next_surface) != 0u ||
      !(bitcast<f32>(completed_paths[path + path_surface_rr]) > 0.0)) {
    return;
  }

  if (offsets.x != pass_unused) {
    // camera_z_depth() uses the world-to-camera Z row for perspective and ortho.
    let depth = dot(geometry_params.world_to_camera_z, vec4<f32>(bitcast<vec3<f32>>(point.xyz), 1.0));
    atomicStore(&film[pixel + offsets.x], bitcast<u32>(depth));
  }
  if (offsets.y != pass_unused) {
    // Reuse the same packed KernelObject pass_id as the SVM Object Info node.
    let object = surface_metadata[object_table_offset_word] + object_metadata_words * hit.z;
    atomicStore(&film[pixel + offsets.y], surface_metadata[object + object_pass_id_word]);
  }
  if (offsets.z != pass_unused) {
    let shader = shaded_points[2u * lane + 1u].w & packed_shader_id_mask;
    let pass_id = f32(bitcast<i32>(
        surface_metadata[surface_metadata[shader_pass_table_offset_word] + shader]));
    atomicStore(&film[pixel + offsets.z], bitcast<u32>(pass_id));
  }
  if (offsets.w != pass_unused) {
    atomicStore(&film[pixel + offsets.w], point.x);
    atomicStore(&film[pixel + offsets.w + 1u], point.y);
    atomicStore(&film[pixel + offsets.w + 2u], point.z);
  }
}

// The surface and background stages publish only the same clamped, directly
// visible emission contribution that they added to Combined on this iteration.
@compute @workgroup_size(workgroup_size_x)
fn accumulate_emission(@builtin(global_invocation_id) id: vec3<u32>,
                       @builtin(num_workgroups) groups: vec3<u32>) {
  let lane = queued_lane(work_index(id, groups));
  if (lane == pass_unused) {
    return;
  }
  let pixel = film_pixel_offset(lane);
  if (pixel != pass_unused) {
    let contribution = samples[2u * retire_rays[0] + 1u + lane].xyz;
    add_rgb(pixel + emission_pass.x, contribution);
  }
}

@compute @workgroup_size(workgroup_size_x)
fn accumulate_surface_passes(@builtin(global_invocation_id) id: vec3<u32>,
                              @builtin(num_workgroups) groups: vec3<u32>) {
  let lane = queued_lane(work_index(id, groups));
  if (lane == pass_unused) {
    return;
  }
  let pixel = film_pixel_offset(lane);
  if (pixel == pass_unused) {
    return;
  }
  for (var i = 0u; i < 4u; i++) {
    let offset = surface_pass_params.offsets[i];
    if (offset != pass_unused) {
      add_rgb(pixel + offset, samples[6u * lane + i].xyz);
    }
  }
  if (surface_pass_params.ao.x != pass_unused) {
    add_rgb(pixel + surface_pass_params.ao.x, samples[6u * lane + 4u].xyz);
  }
}

fn add_light_passes(pixel: u32, delta: vec4<f32>, diffuse: vec3<f32>, glossy: vec3<f32>) {
  let index = u32(delta.w) - 1u;
  let diffuse_offset = light_pass_params.reflection[index];
  let glossy_offset = light_pass_params.reflection[index + 2u];
  let transmission_offset = light_pass_params.transmission[index];
  if (diffuse_offset != pass_unused) {
    add_rgb(pixel + diffuse_offset, delta.xyz * diffuse);
  }
  if (glossy_offset != pass_unused) {
    add_rgb(pixel + glossy_offset, delta.xyz * glossy);
  }
  if (transmission_offset != pass_unused) {
    add_rgb(pixel + transmission_offset, delta.xyz * (vec3<f32>(1.0) - diffuse - glossy));
  }
}

@compute @workgroup_size(workgroup_size_x)
fn accumulate_light_passes(@builtin(global_invocation_id) id: vec3<u32>,
                           @builtin(num_workgroups) groups: vec3<u32>) {
  let lane = queued_lane(work_index(id, groups));
  if (lane == pass_unused) {
    return;
  }
  let pixel = film_pixel_offset(lane);
  if (pixel == pass_unused) {
    return;
  }
  let capacity = retire_rays[0];
  let header = 2u * capacity;
  let record = header + 1u + u32(samples[header].w) * capacity + 6u * lane;
  let forward = samples[record + 4u];
  let nee = samples[record + 5u];
  if (forward.w > 0.0) {
    add_light_passes(pixel, forward, samples[record].xyz, samples[record + 1u].xyz);
  }
  if (nee.w > 0.0) {
    let weights = select(record, record + 2u, nee.w == 1.0);
    add_light_passes(pixel, nee, samples[weights].xyz, samples[weights + 1u].xyz);
  }
}

fn retired_contribution(lane: u32) -> vec4<f32> {
  let sum = lane * path_words + path_radiance;
  return vec4<f32>(bitcast<f32>(completed_paths[sum]),
                   bitcast<f32>(completed_paths[sum + 1u]),
                   bitcast<f32>(completed_paths[sum + 2u]),
                   1.0 - bitcast<f32>(completed_paths[sum + 3u]));
}

fn add_combined(base: u32, rgba: vec4<f32>) {
  let combined = base + render_buffer.placement.w;
  add_component(combined, rgba.x);
  add_component(combined + 1u, rgba.y);
  add_component(combined + 2u, rgba.z);
  add_component(combined + 3u, rgba.w);
}

@compute @workgroup_size(workgroup_size_x)
fn accumulate_render_buffer(@builtin(global_invocation_id) id: vec3<u32>,
                            @builtin(num_workgroups) groups: vec3<u32>) {
  let retired = retirement(work_index(id, groups));
  if (retired.x == pass_unused) {
    return;
  }

  retire_fixed(retired);
}

fn retire_fixed(retired: vec2<u32>) {
  let count_offset = render_buffer.sample_count_offset;
  // Same layout as Cycles' film_write_combined_transparent_pass and
  // film_write_sample. Combined stores sums, including transparency in W.
  add_combined(retired.y, retired_contribution(retired.x));
  if (count_offset != pass_unused) {
    atomicAdd(&film[retired.y + count_offset], 1u);
  }
}

@compute @workgroup_size(workgroup_size_x)
fn retire_adaptive_render_buffer(@builtin(global_invocation_id) id: vec3<u32>,
                                 @builtin(num_workgroups) groups: vec3<u32>) {
  let retired = retirement(work_index(id, groups));
  if (retired.x == pass_unused || render_buffer.adaptive_aux_offset == pass_unused) {
    return;
  }
  retire_adaptive(retired);
}

fn retire_adaptive(retired: vec2<u32>) {
  let rgba = retired_contribution(retired.x);
  add_combined(retired.y, rgba);
  // Camera admission ran Cycles' sample_is_class_A on the effective sample.
  let capacity = retire_rays[0];
  let class_base = 2u + (ray_words + 1u) * capacity;
  if (retire_rays[class_base + retired.x] != 0u) {
    let aux = retired.y + render_buffer.adaptive_aux_offset;
    add_component(aux, 2.0 * rgba.x);
    add_component(aux + 1u, 2.0 * rgba.y);
    add_component(aux + 2u, 2.0 * rgba.z);
  }
}


fn add_rgb(offset: u32, value: vec3<f32>) {
  add_component(offset, value.x);
  add_component(offset + 1u, value.y);
  add_component(offset + 2u, value.z);
}

@compute @workgroup_size(workgroup_size_x)
fn retire_guided_render_buffer(@builtin(global_invocation_id) id: vec3<u32>,
                               @builtin(num_workgroups) groups: vec3<u32>) {
  let retired = retirement(work_index(id, groups));
  if (retired.x == pass_unused) {
    return;
  }
  if (render_buffer.adaptive_aux_offset == pass_unused) {
    retire_fixed(retired);
  } else {
    retire_adaptive(retired);
  }

  let feature = retired.x * 3u;
  if (denoising_params.albedo_offset != pass_unused) {
    add_rgb(retired.y + denoising_params.albedo_offset, denoising[feature + 1u].xyz);
  }
  if (denoising_params.normal_offset != pass_unused) {
    // Surface features are in world space. Volume scatter contributes directly
    // along camera Z, as in Cycles' film_write_denoising_features_volume.
    let feature_normal = denoising[feature + 2u];
    let normal = feature_normal.xyz;
    let camera_normal = vec3<f32>(dot(denoising_params.world_to_camera_x.xyz, normal),
                                  dot(denoising_params.world_to_camera_y.xyz, normal),
                                  dot(denoising_params.world_to_camera_z.xyz, normal) +
                                    feature_normal.w);
    add_rgb(retired.y + denoising_params.normal_offset, camera_normal);
  }
}

@compute @workgroup_size(workgroup_size_x)
fn accumulate_adaptive_render_buffer(@builtin(global_invocation_id) id: vec3<u32>,
                                     @builtin(num_workgroups) groups: vec3<u32>) {
  let retired = retirement(work_index(id, groups));
  let aux_offset = render_buffer.adaptive_aux_offset;
  if (retired.x == pass_unused || aux_offset == pass_unused) {
    return;
  }

  // The translated stock writers produced the per-path pass deltas.
  add_combined(retired.y, adaptive_deltas[retired.x]);

  // Camera admission already counted this sample. Keep AUX.W for convergence.
  let adaptive = adaptive_deltas[retire_rays[0] + retired.x];
  let aux = retired.y + aux_offset;
  add_component(aux, adaptive.x);
  add_component(aux + 1u, adaptive.y);
  add_component(aux + 2u, adaptive.z);
}
