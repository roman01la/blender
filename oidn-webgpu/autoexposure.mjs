// SPDX-License-Identifier: Apache-2.0
// Derived from Intel Open Image Denoise's devices/gpu/gpu_autoexposure.h.
// Copyright 2022 Intel Corporation

const shader = /* wgsl */ `
struct Image {
  width: u32,
  height: u32,
  pixel_stride: u32,
  row_stride: u32,
}

struct Reduction {
  sum: f32,
  count: u32,
}

@group(0) @binding(0) var<storage, read> pixels: array<f32>;
@group(0) @binding(1) var<uniform> image: Image;
@group(0) @binding(2) var<storage, read_write> bins: array<f32>;
@group(0) @binding(3) var<storage, read_write> reductions: array<Reduction>;
@group(0) @binding(4) var<storage, read_write> scale: f32;

var<workgroup> local_sum: array<f32, 256>;
var<workgroup> local_count: array<u32, 256>;

fn finite_positive(value: f32) -> f32 {
  if ((bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u) {
    return 0;
  }
  return clamp(value, 0.0, 3.402823466e+38);
}

@compute @workgroup_size(16, 16)
fn downsample(@builtin(workgroup_id) bin: vec3u,
              @builtin(num_workgroups) bins_size: vec3u,
              @builtin(local_invocation_id) local: vec3u,
              @builtin(local_invocation_index) lane: u32) {
  let size = vec2u(image.width, image.height);
  let first = bin.xy * size / bins_size.xy;
  let end = (bin.xy + vec2u(1)) * size / bins_size.xy;
  let coord = first + local.xy;
  var luminance = 0.0;
  if (all(coord < end)) {
    let index = coord.y * image.row_stride + coord.x * image.pixel_stride;
    luminance = dot(vec3f(finite_positive(pixels[index]),
                         finite_positive(pixels[index + 1u]),
                         finite_positive(pixels[index + 2u])),
                    vec3f(0.212671, 0.715160, 0.072169));
  }
  local_sum[lane] = luminance;
  for (var stride = 128u; stride > 0; stride >>= 1u) {
    workgroupBarrier();
    if (lane < stride) {
      local_sum[lane] += local_sum[lane + stride];
    }
  }
  if (lane == 0) {
    let extent = end - first;
    bins[bin.y * bins_size.x + bin.x] = local_sum[0] / f32(extent.x * extent.y);
  }
}

fn reduce_group(lane: u32, value: Reduction) {
  local_sum[lane] = value.sum;
  local_count[lane] = value.count;
  for (var stride = 128u; stride > 0; stride >>= 1u) {
    workgroupBarrier();
    if (lane < stride) {
      local_sum[lane] += local_sum[lane + stride];
      local_count[lane] += local_count[lane + stride];
    }
  }
}

@compute @workgroup_size(256)
fn reduce(@builtin(global_invocation_id) id: vec3u,
          @builtin(workgroup_id) group: vec3u,
          @builtin(num_workgroups) groups: vec3u,
          @builtin(local_invocation_index) lane: u32) {
  var total = Reduction(0, 0u);
  for (var index = id.x; index < arrayLength(&bins); index += 256u * groups.x) {
    let value = bins[index];
    if (value > 1e-8) {
      total.sum += log2(value);
      total.count++;
    }
  }
  reduce_group(lane, total);
  if (lane == 0) {
    reductions[group.x] = Reduction(local_sum[0], local_count[0]);
  }
}

@compute @workgroup_size(256)
fn finalize(@builtin(local_invocation_index) lane: u32) {
  var total = Reduction(0, 0u);
  if (lane < arrayLength(&reductions)) {
    total = reductions[lane];
  }
  reduce_group(lane, total);
  if (lane == 0) {
    scale = 1.0;
    if (local_count[0] > 0) {
      scale = 0.18 / exp2(local_sum[0] / f32(local_count[0]));
    }
  }
}
`;

const pipelines = new WeakMap();

export function encodeOIDNAutoexposure(device, encoder, pixels, width, height,
                                       pixelStride = 4, rowStride = width * pixelStride) {
  let stages = pipelines.get(device);
  if (!stages) {
    const module = device.createShaderModule({label: "OIDN autoexposure", code: shader});
    stages = ["downsample", "reduce", "finalize"].map(entryPoint =>
      device.createComputePipeline({
        label: `OIDN autoexposure ${entryPoint}`,
        layout: "auto",
        compute: {module, entryPoint},
      }));
    pipelines.set(device, stages);
  }

  const binsWidth = Math.ceil(width / 16);
  const binsHeight = Math.ceil(height / 16);
  const groupCount = Math.min(256, Math.ceil(binsWidth * binsHeight / 256));
  const image = device.createBuffer({
    label: "OIDN exposure image",
    size: 16,
    usage: GPUBufferUsage.UNIFORM,
    mappedAtCreation: true,
  });
  new Uint32Array(image.getMappedRange()).set([width, height, pixelStride, rowStride]);
  image.unmap();
  const bins = device.createBuffer({
    label: "OIDN exposure bins",
    size: binsWidth * binsHeight * 4,
    usage: GPUBufferUsage.STORAGE,
  });
  const reductions = device.createBuffer({
    label: "OIDN exposure reductions",
    size: groupCount * 8,
    usage: GPUBufferUsage.STORAGE,
  });
  const scale = device.createBuffer({
    label: "OIDN input scale",
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const pass = encoder.beginComputePass({label: "OIDN autoexposure"});
  const dispatch = (stage, bindings, x, y = 1) => {
    const pipeline = stages[stage];
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: bindings.map(([binding, buffer]) => ({binding, resource: {buffer}})),
    }));
    pass.dispatchWorkgroups(x, y);
  };
  dispatch(0, [[0, pixels], [1, image], [2, bins]], binsWidth, binsHeight);
  dispatch(1, [[2, bins], [3, reductions]], groupCount);
  dispatch(2, [[3, reductions], [4, scale]], 1);
  pass.end();

  return {
    scale,
    destroy() {
      for (const buffer of [image, bins, reductions, scale]) buffer.destroy();
    },
  };
}
