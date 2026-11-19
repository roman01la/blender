/* SPDX-License-Identifier: Apache-2.0 */

import { blenderOIDNWebGPU } from "./oidn-webgpu-launcher.mjs";

const devices = new WeakMap();
const unused = 0xffffffff;

const shaders = {
  compose: `
struct Tile {
  width: u32, height: u32, passStride: u32, sourceWidth: u32,
  sourceX: u32, sourceY: u32, targetWidth: u32, targetX: u32, targetY: u32,
}
@group(0) @binding(0) var<uniform> tile: Tile;
@group(0) @binding(1) var<storage, read> source: array<u32>;
@group(0) @binding(2) var<storage, read_write> destination: array<u32>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= tile.width || id.y >= tile.height) { return; }
  let src = ((tile.sourceY + id.y) * tile.sourceWidth + tile.sourceX + id.x) * tile.passStride;
  let dst = ((tile.targetY + id.y) * tile.targetWidth + tile.targetX + id.x) * tile.passStride;
  for (var p = 0u; p < tile.passStride; p++) {
    destination[dst + p] = source[src + p];
  }
}
`,
  unpack: `
struct Film {
  width: u32, height: u32, startY: u32, passStride: u32,
  samples: u32, sampleCount: u32, noisy: u32, denoised: u32, albedo: u32, normal: u32,
}
@group(0) @binding(0) var<uniform> film: Film;
@group(0) @binding(1) var<storage, read> source: array<u32>;
@group(0) @binding(2) var<storage, read_write> color: array<f32>;
@group(0) @binding(3) var<storage, read_write> albedo: array<f32>;
@group(0) @binding(4) var<storage, read_write> normal: array<f32>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= film.width || id.y >= film.height) { return; }
  let src = (id.y * film.width + id.x) * film.passStride;
  let dst = ((film.startY + id.y) * film.width + id.x) * 3u;

  // Cycles film_get_scale_and_scale_exposure() and OIDNDenoiseContext::is_pass_scale_needed().
  // Without a per-pixel sample count OIDN reads the HDR sum and scales the guides.
  var colorScale = 1.0;
  var guideScale = 1.0 / f32(max(1u, film.samples));
  if (film.sampleCount != 0xffffffffu) {
    let count = source[src + film.sampleCount];
    colorScale = select(0.0, 1.0 / f32(count), count != 0u);
    guideScale = colorScale;
  }
  for (var c = 0u; c < 3u; c++) {
    color[dst + c] = bitcast<f32>(source[src + film.noisy + c]) * colorScale;
    if (film.albedo != 0xffffffffu) {
      albedo[dst + c] = bitcast<f32>(source[src + film.albedo + c]) * guideScale;
    }
    if (film.normal != 0xffffffffu) {
      normal[dst + c] = bitcast<f32>(source[src + film.normal + c]) * guideScale;
    }
  }
}
`,
  restore: `
struct Film {
  width: u32, height: u32, startY: u32, passStride: u32,
  samples: u32, sampleCount: u32, noisy: u32, denoised: u32, albedo: u32, normal: u32,
}
@group(0) @binding(0) var<uniform> film: Film;
@group(0) @binding(1) var<storage, read_write> destination: array<u32>;
@group(0) @binding(2) var<storage, read> color: array<f32>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= film.width || id.y >= film.height) { return; }
  let src = ((film.startY + id.y) * film.width + id.x) * 3u;
  let dst = (id.y * film.width + id.x) * film.passStride;
  // OIDNDenoiseContext::postprocess_output() restores raw sample sums and transparency.
  var scale = 1.0;
  if (film.sampleCount != 0xffffffffu) { scale = f32(destination[dst + film.sampleCount]); }
  for (var c = 0u; c < 3u; c++) {
    destination[dst + film.denoised + c] = bitcast<u32>(color[src + c] * scale);
  }
  destination[dst + film.denoised + 3u] = destination[dst + film.noisy + 3u];
}
`,
};

function state(device) {
  if (!devices.has(device)) {
    devices.set(device, { frames: new Map(), pipelines: new Map() });
  }
  return devices.get(device);
}

function pipeline(device, name) {
  const pipelines = state(device).pipelines;
  if (!pipelines.has(name)) {
    const label = `oidn/film/${name}`;
    const module = device.createShaderModule({ label, code: shaders[name] });
    pipelines.set(name, device.createComputePipelineAsync({
      label, layout: "auto", compute: { module, entryPoint: "main" },
    }));
  }
  return pipelines.get(name);
}

async function checked(device, action) {
  device.pushErrorScope("internal");
  device.pushErrorScope("out-of-memory");
  device.pushErrorScope("validation");
  try {
    return await action();
  } finally {
    const errors = await Promise.all([
      device.popErrorScope(), device.popErrorScope(), device.popErrorScope(),
    ]);
    const error = errors.find(Boolean);
    if (error) {
      throw new Error(error.message);
    }
  }
}

function dispatch(device, encoder, program, name, width, height, values, buffers, scratch) {
  const label = `oidn/film/${name}`;
  const params = device.createBuffer({
    label: `${label}/params`, size: Math.ceil(values.length / 4) * 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  scratch.push(params);
  device.queue.writeBuffer(params, 0, Uint32Array.from(values));
  const bindGroup = device.createBindGroup({
    label, layout: program.getBindGroupLayout(0),
    entries: [params, ...buffers].map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
  const pass = encoder.beginComputePass({ label });
  pass.setPipeline(program);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
  pass.end();
}

function storageLimit(device) {
  return Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
}

function createFrame(device, width, height, passStride, id) {
  const rowBytes = width * passStride * 4;
  const rows = Math.floor(storageLimit(device) / rowBytes);
  if (rows < 1 || width * height * 12 > storageLimit(device)) {
    throw new Error("Cycles film or OIDN input exceeds WebGPU storage limits");
  }
  const frame = { width, height, passStride, shards: [] };
  state(device).frames.set(id, frame);
  for (let y = 0; y < height; y += rows) {
    const count = Math.min(rows, height - y);
    const buffer = device.createBuffer({
      label: "oidn/film/shard", size: count * rowBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    frame.shards.push({ y, count, buffer });
  }
  return frame;
}

export function dropFilm(device, id) {
  const frames = state(device).frames;
  const frame = frames.get(id);
  if (frame) {
    for (const shard of frame.shards) {
      shard.buffer.destroy();
    }
    frames.delete(id);
  }
}

export async function composeFilmTile(device, source, layout) {
  const [id, width, height, stride, sourceWidth, sourceX, sourceY,
    targetX, targetY, tileWidth, tileHeight] = layout;
  const scratch = [];
  try {
    await checked(device, async () => {
      const frame = state(device).frames.get(id) ?? createFrame(device, width, height, stride, id);
      const program = await pipeline(device, "compose");
      const encoder = device.createCommandEncoder({ label: "oidn/film/compose" });
      for (const { y, count, buffer } of frame.shards) {
        const first = Math.max(y, targetY);
        const rows = Math.min(y + count, targetY + tileHeight) - first;
        if (rows > 0) {
          dispatch(device, encoder, program, "compose", tileWidth, rows, [
            tileWidth, rows, stride, sourceWidth, sourceX, sourceY + first - targetY,
            width, targetX, first - y,
          ], [source, buffer], scratch);
        }
      }
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
    });
  } catch (error) {
    dropFilm(device, id);
    throw error;
  } finally {
    for (const buffer of scratch) {
      buffer.destroy();
    }
  }
}

export async function finishFilm(device, layout, models, writeWords, reportTimes) {
  const [id, samples, sampleCount, noisy, denoised, albedoOffset, normalOffset] = layout;
  const frame = state(device).frames.get(id);
  if (!frame) {
    throw new Error("Missing Cycles resident film");
  }
  const { width, height, passStride, shards } = frame;
  const scratch = [];
  const times = [0, 0, 0];
  function plane(present) {
    const data = device.createBuffer({
      label: "oidn/film/normalized", size: present ? width * height * 12 : 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    scratch.push(data);
    return { data, offset: 0, pixelStride: 3, rowStride: width * 3 };
  }
  function filmPass(program, name, boundBuffers) {
    const encoder = device.createCommandEncoder({ label: `oidn/film/${name}` });
    for (const { y, count, buffer } of shards) {
      dispatch(device, encoder, program, name, width, count, [
        width, count, y, passStride, samples, sampleCount, noisy, denoised, albedoOffset, normalOffset,
      ], [buffer, ...boundBuffers], scratch);
    }
    device.queue.submit([encoder.finish()]);
  }
  async function filter(index, inputs, output, hdr) {
    if (models[index]) {
      const t = performance.now();
      await blenderOIDNWebGPU(device, {
        width, height, model: models[index], hdr, srgb: false, ...inputs, output,
      });
      times[index] = (performance.now() - t) / 1000;
    }
  }

  try {
    if (models[2]) {
      await checked(device, async () => {
        const color = plane(true);
        const albedo = plane(albedoOffset !== unused);
        const normal = plane(normalOffset !== unused);
        filmPass(await pipeline(device, "unpack"), "unpack", [color.data, albedo.data, normal.data]);

        await filter(0, { albedo }, albedo, false);
        await filter(1, { normal }, normal, false);
        await filter(2, {
          color,
          albedo: albedoOffset !== unused ? albedo : undefined,
          normal: normalOffset !== unused ? normal : undefined,
        }, color, true);

        filmPass(await pipeline(device, "restore"), "restore", [color.data]);
        await device.queue.onSubmittedWorkDone();
      });
    }

    await checked(device, async () => {
      const label = "oidn/film/export-readback";
      const readback = device.createBuffer({
        label, size: shards[0].buffer.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      scratch.push(readback);
      for (const { y, buffer } of shards) {
        const encoder = device.createCommandEncoder({ label });
        encoder.copyBufferToBuffer(buffer, 0, readback, 0, buffer.size);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ, 0, buffer.size);
        writeWords(new Uint32Array(readback.getMappedRange(0, buffer.size)), y * width * passStride);
        readback.unmap();
      }
    });
    reportTimes(times);
  } finally {
    for (const buffer of scratch) {
      buffer.destroy();
    }
    dropFilm(device, id);
  }
}
