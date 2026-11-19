// SPDX-License-Identifier: Apache-2.0
// Copyright 2018-2022 Intel Corporation
// The input, output, autoexposure and tile coordinates follow the OIDN RT
// filter in core/unet_filter.cpp and devices/gpu/gpu_*_process.h.
import {NativeUNetExecutor} from "oidn-webgpu/lib/nativeUNet.js";
import {detectUNetModelSpec, validateUNetModel} from "oidn-webgpu/lib/modelSpec.js";
import {planModelExecution} from "oidn-webgpu/lib/graphOptimizer.js";
import {parseTZA} from "oidn-webgpu/lib/tza.js";
import {hdrForwardWGSL, hdrInverseWGSL} from "oidn-webgpu/lib/hdrTransfer.js";
import {sanitizeWGSL} from "oidn-webgpu/lib/sanitize.js";
import {encodeOIDNAutoexposure} from "./autoexposure.mjs";

const alignment = 16;
const maxActivationBytes = 128 * 1024 * 1024;
const maxTileEdge = 704;
const imageWorkgroupSize = 8;

const srgbWGSL = /* wgsl */ `
fn forward(value: f32) -> f32 {
  if (value <= 0.0031308) {
    return 12.92 * value;
  }
  return 1.055 * pow(value, 1.0 / 2.4) - 0.055;
}
fn inverse(value: f32) -> f32 {
  if (value <= 0.04045) {
    return value / 12.92;
  }
  return pow((value + 0.055) / 1.055, 2.4);
}`;

function transfers(hdr, signedNormal, srgb, output) {
  if (srgb || signedNormal) {
    return {code: "", name: ""};
  }
  if (hdr) {
    return output ?
      {code: hdrInverseWGSL("pu"), name: "HDRInverse"} :
      {code: hdrForwardWGSL("pu"), name: "HDRForward"};
  }
  return {code: srgbWGSL, name: output ? "inverse" : "forward"};
}

function inputShader(sourceCount, hdr, signedNormal, srgb) {
  const transfer = transfers(hdr, signedNormal, srgb, false);
  const inputBindings = Array.from({length: sourceCount}, (_, index) =>
    `@group(0) @binding(${index}) var<storage, read> input${index}: array<f32>;`).join("\n");
  const blocks = Math.ceil(sourceCount * 3 / 4);
  const channels = Array.from({length: sourceCount}, (_, index) =>
    ["r", "g", "b"].map(channel => `value${index}.${channel}`)).flat();
  const stores = Array.from({length: blocks}, (_, index) =>
    `packed[outPixel * ${blocks}u + ${index}u] = vec4f(${Array.from({length: 4}, (_, lane) =>
      channels[index * 4 + lane] ?? "0.0").join(", ")});`).join("\n  ");
  const secondary = Array.from({length: sourceCount - 1}, (_, index) =>
    `value${index + 1} = sanitize(vec3f(input${index + 1}[src],
        input${index + 1}[src + 1u], input${index + 1}[src + 2u]),
        ${index === 0 ? "0.0, 1.0)" : "-1.0, 1.0) * 0.5 + 0.5"};`).join("\n    ");
  const primaryLower = signedNormal ? "-1.0" : "0.0";
  const primaryUpper = hdr ? "FLT_MAX" : "1.0";
  return /* wgsl */ `
${sanitizeWGSL}
${transfer.code}
struct InputTile {
  imageWidth: u32,
  tileWidth: u32,
  tileHeight: u32,
  padding: u32,
  sourceOrigin: vec2u,
  inputSize: vec2u,
  destinationOrigin: vec2u,
}
${inputBindings}
@group(0) @binding(${sourceCount}) var<storage, read_write> packed: array<vec4f>;
@group(0) @binding(${sourceCount + 1}) var<uniform> tile: InputTile;
${hdr ? `@group(0) @binding(${sourceCount + 2}) var<storage, read> exposure: f32;` : ""}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= tile.tileWidth || id.y >= tile.tileHeight) { return; }
  ${Array.from({length: sourceCount}, (_, index) => `var value${index} = vec3f(0.0);`).join("\n  ")}
  if (all(id.xy >= tile.destinationOrigin) &&
      all(id.xy - tile.destinationOrigin < tile.inputSize)) {
    let inputTile = id.xy - tile.destinationOrigin + tile.sourceOrigin;
    let src = (inputTile.y * tile.imageWidth + inputTile.x) * 3u;
    value0 = sanitize(vec3f(input0[src], input0[src + 1u], input0[src + 2u])
        ${hdr ? "* exposure" : ""}, ${primaryLower}, ${primaryUpper});
    ${signedNormal ? "value0 = value0 * 0.5 + 0.5;" : ""}
    ${transfer.name ? `value0 = vec3f(${transfer.name}(value0.r), ${transfer.name}(value0.g),
        ${transfer.name}(value0.b));` : ""}
    ${secondary}
  }
  let outPixel = id.y * tile.tileWidth + id.x;
  ${stores}
}`;
}

function outputShader(hdr, signedNormal, srgb) {
  const transfer = transfers(hdr, signedNormal, srgb, true);
  return /* wgsl */ `
${sanitizeWGSL}
${transfer.code}
struct OutputTile {
  imageWidth: u32,
  tileWidth: u32,
  sourceOrigin: vec2u,
  outputSize: vec2u,
  destinationOrigin: vec2u,
}
@group(0) @binding(0) var<storage, read> network: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> outputImage: array<f32>;
@group(0) @binding(2) var<uniform> tile: OutputTile;
${hdr ? "@group(0) @binding(3) var<storage, read> exposure: f32;" : ""}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= tile.outputSize)) { return; }
  let source = id.xy + tile.sourceOrigin;
  let destination = id.xy + tile.destinationOrigin;
  var value = sanitize(network[source.y * tile.tileWidth + source.x].rgb, 0.0, FLT_MAX);
  ${transfer.name ? `value = vec3f(${transfer.name}(value.r), ${transfer.name}(value.g),
      ${transfer.name}(value.b));` : ""}
  ${signedNormal ? "value = max(value * 2.0 - 1.0, vec3f(-1.0));" : ""}
  ${hdr ? "value = value / exposure;" : "value = min(value, vec3f(1.0));"}
  let outIndex = (destination.y * tile.imageWidth + destination.x) * 3u;
  outputImage[outIndex] = value.r;
  outputImage[outIndex + 1u] = value.g;
  outputImage[outIndex + 2u] = value.b;
}`;
}

function roundUp(value, block) {
  return Math.ceil(value / block) * block;
}

function selectTileSize(model, width, height, device) {
  const maxBytes = Math.min(device.limits.maxStorageBufferBindingSize,
                            device.limits.maxBufferSize, maxActivationBytes);
  const overlap = roundUp(model.spec.receptiveField / 2, alignment);
  let tileWidth = Math.min(roundUp(width, alignment), maxTileEdge);
  let tileHeight = Math.min(roundUp(height, alignment), maxTileEdge);
  while (true) {
    const {valueShapes} = planModelExecution(model, tileWidth, tileHeight);
    const bytes = Math.max(...[...valueShapes.values()].map(shape =>
      shape.width * shape.height * roundUp(shape.channels, 4) * 4));
    if (bytes <= maxBytes) {
      return {tileWidth, tileHeight, overlap};
    }
    if (tileWidth >= tileHeight && tileWidth > 2 * overlap + alignment) {
      tileWidth -= alignment;
    }
    else if (tileHeight > 2 * overlap + alignment) {
      tileHeight -= alignment;
    }
    else {
      throw new Error("OIDN activations exceed the WebGPU storage limit");
    }
  }
}

function tileAxis(dimension, tileSize, overlap) {
  if (dimension <= tileSize) {
    return [{input: 0, inputSize: dimension, pad: 0, output: 0, outputSize: dimension, start: 0}];
  }
  const stride = tileSize - 2 * overlap;
  const count = Math.ceil((dimension - 2 * overlap) / stride);
  return Array.from({length: count}, (_, index) => {
    const input = index * stride;
    const inputSize = Math.min(dimension - input, tileSize);
    const pad = tileSize - roundUp(inputSize, alignment);
    const overlapBefore = index > 0 ? overlap : 0;
    const overlapAfter = index + 1 < count ? overlap : 0;
    return {input, inputSize, pad, output: input + overlapBefore,
            outputSize: inputSize - overlapBefore - overlapAfter, start: pad + overlapBefore};
  });
}

function isGPUView(view) {
  return view.data instanceof GPUBuffer;
}

function validateView(view, width, height) {
  const {data, offset = 0, pixelStride, rowStride} = view;
  const gpu = isGPUView(view);
  const length = gpu ? data.size / 4 : data?.length;
  if ((!gpu && !(data instanceof Float32Array)) ||
      !Number.isSafeInteger(offset) || offset < 0 ||
      !Number.isSafeInteger(pixelStride) || !Number.isSafeInteger(rowStride) ||
      pixelStride < 3 || rowStride < (width - 1) * pixelStride + 3 ||
      offset + (height - 1) * rowStride + (width - 1) * pixelStride + 3 > length) {
    throw new Error("Invalid OIDN Float32 image view");
  }
}

function isPackedView(view, width) {
  return view.pixelStride === 3 && view.rowStride === width * 3;
}

function viewTransferShader(toPacked) {
  return /* wgsl */ `
struct Params {
  width: u32,
  height: u32,
  firstRow: u32,
  offset: u32,
  pixelStride: u32,
  rowStride: u32,
}
@group(0) @binding(0) var<storage, read${toPacked ? "" : "_write"}> view: array<f32>;
@group(0) @binding(1) var<storage, read${toPacked ? "_write" : ""}> packed: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= params.width || id.y >= params.height) { return; }
  let viewIndex = params.offset + id.y * params.rowStride + id.x * params.pixelStride;
  let packedIndex = ((params.firstRow + id.y) * params.width + id.x) * 3u;
  ${toPacked ?
    `packed[packedIndex] = view[viewIndex];
  packed[packedIndex + 1u] = view[viewIndex + 1u];
  packed[packedIndex + 2u] = view[viewIndex + 2u];` :
    `view[viewIndex] = packed[packedIndex];
  view[viewIndex + 1u] = packed[packedIndex + 1u];
  view[viewIndex + 2u] = packed[packedIndex + 2u];`}
}`;
}

function packPixels(view, width, height, destination) {
  const {data, offset = 0, pixelStride, rowStride} = view;
  if (isPackedView(view, width)) {
    destination.set(data.subarray(offset, offset + width * height * 3));
    return;
  }
  let pixel = 0;
  for (let y = 0; y < height; y++) {
    let source = offset + y * rowStride;
    for (let x = 0; x < width; x++, source += pixelStride) {
      destination[pixel++] = data[source];
      destination[pixel++] = data[source + 1];
      destination[pixel++] = data[source + 2];
    }
  }
}

function unpackPixels(view, width, height, source) {
  const {data, offset = 0, pixelStride, rowStride} = view;
  if (isPackedView(view, width)) {
    data.set(source, offset);
    return;
  }
  let pixel = 0;
  for (let y = 0; y < height; y++) {
    let destination = offset + y * rowStride;
    for (let x = 0; x < width; x++, destination += pixelStride) {
      data[destination] = source[pixel++];
      data[destination + 1] = source[pixel++];
      data[destination + 2] = source[pixel++];
    }
  }
}

function encodePass(device, command, pipeline, buffers, label, workgroups) {
  const bindings = device.createBindGroup({
    label,
    layout: pipeline.getBindGroupLayout(0),
    entries: buffers.map(([binding, resource]) =>
      ({binding, resource: resource instanceof GPUBuffer ? {buffer: resource} : resource})),
  });
  const pass = command.beginComputePass({label});
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindings);
  pass.dispatchWorkgroups(...workgroups);
  pass.end();
}

function imageWorkgroups(width, height) {
  return [Math.ceil(width / imageWorkgroupSize), Math.ceil(height / imageWorkgroupSize)];
}

async function collectErrors(device, count) {
  const errors = [];
  for (let i = 0; i < count; i++) {
    const error = await device.popErrorScope();
    if (error) {
      errors.push(error.message);
    }
  }
  if (errors.length) {
    throw new Error(`OIDN WebGPU failed: ${errors.join(" | ")}`);
  }
}

export function createOIDNDenoiser({device, loadWeights}) {
  const models = new Map();
  const pipelines = new Map();
  let active = false;
  let lost = null;
  void device.lost.then(reason => { lost = reason; });

  function pipeline(label, code, entryPoint = "main") {
    const key = `${label}/${entryPoint}`;
    let cached = pipelines.get(key);
    if (!cached) {
      cached = device.createComputePipelineAsync({
        label: key,
        layout: "auto",
        compute: {module: device.createShaderModule({label, code}), entryPoint},
      });
      pipelines.set(key, cached);
      void cached.catch(() => pipelines.delete(key));
    }
    return cached;
  }

  const scopeTypes = ["out-of-memory", "internal", "validation"];

  async function withScopes(callback) {
    for (const type of scopeTypes) {
      device.pushErrorScope(type);
    }
    try {
      return await callback();
    }
    finally {
      await collectErrors(device, scopeTypes.length);
    }
  }

  async function transferGPUView(view, packed, width, height, toPacked) {
    const {data, offset = 0, pixelStride, rowStride} = view;
    const operation = toPacked ? "oidn/input" : "oidn/output";
    if (isPackedView(view, width) &&
        (data.usage & (toPacked ? GPUBufferUsage.COPY_SRC : GPUBufferUsage.COPY_DST))) {
      const command = device.createCommandEncoder({label: operation + "/copy"});
      const src = toPacked ? data : packed;
      const dst = toPacked ? packed : data;
      command.copyBufferToBuffer(src, toPacked ? offset * 4 : 0, dst, toPacked ? 0 : offset * 4,
                                 width * height * 12);
      device.queue.submit([command.finish()]);
      await device.queue.onSubmittedWorkDone();
      return;
    }
    if (!(data.usage & GPUBufferUsage.STORAGE)) {
      throw new Error("OIDN strided GPU views require storage binding usage");
    }
    const label = operation + (toPacked ? "/gather" : "/scatter");
    const command = device.createCommandEncoder({label});
    const bindingAlignment = device.limits.minStorageBufferOffsetAlignment / 4;
    const bindingLimit = Math.floor(device.limits.maxStorageBufferBindingSize / 4);
    const rowSize = (width - 1) * pixelStride + 3;
    const workgroups = device.limits.maxComputeWorkgroupsPerDimension;
    const stage = await pipeline(label, viewTransferShader(toPacked));
    const uniforms = [];
    try {
      for (let row = 0; row < height;) {
        const firstPixel = offset + row * rowStride;
        const aligned = firstPixel - firstPixel % bindingAlignment;
        const relative = firstPixel - aligned;
        const maxRows = Math.floor((bindingLimit - relative - rowSize) / rowStride) + 1;
        const rows = Math.min(maxRows, height - row, workgroups * imageWorkgroupSize);
        if (rows < 1) {
          throw new Error("OIDN image row exceeds the WebGPU storage binding limit");
        }
        const size = (relative + (rows - 1) * rowStride + rowSize) * 4;
        const params = device.createBuffer({
          label: `${label}/parameters`,
          size: 32,
          usage: GPUBufferUsage.UNIFORM,
          mappedAtCreation: true,
        });
        uniforms.push(params);
        new Uint32Array(params.getMappedRange()).set([width, rows, row, relative, pixelStride, rowStride]);
        params.unmap();
        encodePass(device, command, stage, [
          [0, {buffer: data, offset: aligned * 4, size}],
          [1, packed],
          [2, params],
        ], label, imageWorkgroups(width, rows));
        row += rows;
      }
      device.queue.submit([command.finish()]);
      await device.queue.onSubmittedWorkDone();
    }
    finally {
      for (const params of uniforms) {
        params.destroy();
      }
    }
  }

  async function modelFromName(name) {
    let model = models.get(name);
    if (!model) {
      const tensors = parseTZA(await loadWeights(name));
      model = validateUNetModel(tensors, detectUNetModelSpec(tensors));
      models.set(name, model);
    }
    return model;
  }

  async function run({width, height, model: name, color, albedo, normal, output, hdr, srgb = false}) {
    const model = await modelFromName(name);
    const signedNormal = !color && !albedo && Boolean(normal);
    hdr ??= Boolean(color) && name.startsWith("rt_hdr");
    const sources = [color, albedo, normal].filter(Boolean);
    if (sources.length !== model.inputChannels / 3 ||
        (color && normal && !albedo) || (!color && sources.length !== 1) ||
        (hdr && !color)) {
      throw new Error(`OIDN inputs do not match ${name}`);
    }
    const {tileWidth, tileHeight, overlap} = selectTileSize(model, width, height, device);
    const tilesX = tileAxis(width, tileWidth, overlap);
    const tilesY = tileAxis(height, tileHeight, overlap);
    const imageBytes = width * height * 12;
    if (imageBytes > Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize)) {
      throw new Error("OIDN image exceeds the WebGPU storage limit");
    }

    const buffers = [];
    let executor;
    let autoexposure;
    let keepOutput = false;
    function buffer(label, size, usage, mappedAtCreation = false) {
      const result = device.createBuffer({label, size, usage, mappedAtCreation});
      buffers.push(result);
      return result;
    }
    function uniform(label, values) {
      const result = buffer(label, roundUp(values.length * 4, 16), GPUBufferUsage.UNIFORM, true);
      new Uint32Array(result.getMappedRange()).set(values);
      result.unmap();
      return result;
    }

    const imageOutput = device.createBuffer({
      label: "oidn/output",
      size: imageBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    try {
      const imageInputs = [];
      for (const [index, image] of sources.entries()) {
        const gpu = isGPUView(image);
        if (gpu && isPackedView(image, width) && !(image.offset ?? 0) &&
            image.data.size <= device.limits.maxStorageBufferBindingSize &&
            (image.data.usage & GPUBufferUsage.STORAGE)) {
          imageInputs.push(image.data);
          continue;
        }
        const upload = buffer(`oidn/input/${index}`, imageBytes,
                              GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, !gpu);
        if (gpu) {
          await transferGPUView(image, upload, width, height, true);
        }
        else {
          packPixels(image, width, height, new Float32Array(upload.getMappedRange()));
          upload.unmap();
        }
        imageInputs.push(upload);
      }

      const shaderVariant = `${sources.length}/${hdr}/${signedNormal}/${srgb}`;
      const [inputPipeline, outputPipeline] = await Promise.all([
        pipeline(`oidn/input/${shaderVariant}`, inputShader(sources.length, hdr, signedNormal, srgb)),
        pipeline(`oidn/output/${shaderVariant}`, outputShader(hdr, signedNormal, srgb)),
      ]);
      executor = new NativeUNetExecutor(device, model, {precision: "fp32"});
      await executor.prepare();
      const packedInput = executor.packedInputBuffer(tileWidth, tileHeight);

      if (hdr) {
        const command = device.createCommandEncoder({label: "oidn/exposure"});
        autoexposure = encodeOIDNAutoexposure(device, command, imageInputs[0], width, height, 3);
        device.queue.submit([command.finish()]);
      }
      const exposure = autoexposure?.scale;

      const inputEntries = imageInputs.map((input, binding) => [binding, input]);
      inputEntries.push([sources.length, packedInput]);
      for (const y of tilesY) {
        for (const x of tilesX) {
          const inputParams = uniform("oidn/input/tile", [
            width, tileWidth, tileHeight, 0, x.input, y.input, x.inputSize, y.inputSize, x.pad, y.pad,
          ]);
          const outputParams = uniform("oidn/output/tile", [
            width, tileWidth, x.start, y.start, x.outputSize, y.outputSize, x.output, y.output,
          ]);
          const inputCommand = device.createCommandEncoder({label: "oidn/input"});
          encodePass(device, inputCommand, inputPipeline,
            [...inputEntries, [sources.length + 1, inputParams],
             ...(exposure ? [[sources.length + 2, exposure]] : [])],
            "oidn/input", imageWorkgroups(tileWidth, tileHeight));
          device.queue.submit([inputCommand.finish()]);

          const network = executor.executePacked(tileWidth, tileHeight);

          const outputCommand = device.createCommandEncoder({label: "oidn/output"});
          encodePass(device, outputCommand, outputPipeline,
            [[0, network], [1, imageOutput], [2, outputParams], ...(exposure ? [[3, exposure]] : [])],
            "oidn/output", imageWorkgroups(x.outputSize, y.outputSize));
          device.queue.submit([outputCommand.finish()]);
          // Bound queued work and keep the browser responsive between OIDN tiles.
          await device.queue.onSubmittedWorkDone();
          if (lost) {
            throw new Error(`OIDN WebGPU device lost: ${lost.message}`);
          }
        }
      }
      const tiles = tilesX.length * tilesY.length;
      if (isGPUView(output)) {
        keepOutput = true;
        return {buffer: imageOutput, tiles};
      }
      const readback = buffer("oidn/readback", imageBytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
      const outputCommand = device.createCommandEncoder({label: "oidn/readback"});
      outputCommand.copyBufferToBuffer(imageOutput, 0, readback, 0, imageBytes);
      device.queue.submit([outputCommand.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const data = new Float32Array(readback.getMappedRange().slice(0));
      readback.unmap();
      await device.queue.onSubmittedWorkDone();
      if (lost) {
        throw new Error(`OIDN WebGPU device lost: ${lost.message}`);
      }
      return {data, tiles};
    }
    finally {
      executor?.dispose();
      autoexposure?.destroy();
      if (!keepOutput) {
        imageOutput.destroy();
      }
      for (const resource of buffers) {
        resource.destroy();
      }
    }
  }

  return {
    async denoise(options) {
      const {width, height, output, color, albedo, normal, model} = options;
      if (active || lost) {
        throw new Error(lost ? `OIDN WebGPU device lost: ${lost.message}` : "OIDN WebGPU is busy");
      }
      if (!Number.isSafeInteger(width) || width <= 0 || !Number.isSafeInteger(height) || height <= 0 ||
          !Number.isSafeInteger(width * height * 12) || !output) {
        throw new Error("Invalid OIDN dimensions or output image");
      }
      for (const view of [color, albedo, normal, output]) {
        if (view) {
          validateView(view, width, height);
        }
      }
      const startTime = performance.now();
      active = true;
      let result;
      try {
        await withScopes(async () => {
          result = await run(options);
        });
        // Delay publishing until inference scopes settle so an in-place guide
        // remains available for fallback whenever inference fails.
        if (result.buffer) {
          await withScopes(() => transferGPUView(output, result.buffer, width, height, false));
        }
        else {
          unpackPixels(output, width, height, result.data);
        }
        if (lost) {
          throw new Error(`OIDN WebGPU device lost: ${lost.message}`);
        }
        const milliseconds = performance.now() - startTime;
        const resident = [color, albedo, normal, output].filter(Boolean).every(isGPUView);
        console.info(`Cycles WebGPU OIDN dispatched ${model}: ${result.tiles} tiles`,
                     `in ${milliseconds.toFixed(1)} ms${resident ? " GPU-resident" : ""}`);
        return {tiles: result.tiles, milliseconds, resident};
      }
      finally {
        result?.buffer?.destroy();
        active = false;
      }
    },
  };
}
