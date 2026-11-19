// SPDX-License-Identifier: Apache-2.0

import { execFile, spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { promisify } from 'node:util';

const args = process.argv.slice(2);
const experimentModes = ['--bvh2', '--primary', '--surface', '--lit', '--film', '--svm', '--svm-compact'];
const mode = experimentModes.includes(args[0]) ? args.shift().slice(2) : 'camera';
const argCount = { camera: 2, bvh2: 4, primary: 5, surface: 7, lit: 10, film: 1, svm: 2, 'svm-compact': 2 };
if (args.length !== argCount[mode] && !(mode === 'film' && args.length === 2)) {
  throw Error(
    'Usage: node verify.mjs camera.wgsl reference.json\n' +
    '       node verify.mjs --bvh2 traversal.wgsl scene.bin rays.bin expected-hits.bin\n' +
    '       node verify.mjs --primary primary-camera.wgsl bvh2.wgsl scene.bin primary-reference.json bvh2-reference\n' +
    '       node verify.mjs --surface primary-camera.wgsl bvh2.wgsl scene.bin primary-reference.json bvh2-reference surface.wgsl surface.bin\n' +
    '       node verify.mjs --lit primary-camera.wgsl bvh2.wgsl scene.bin primary-reference.json bvh2-reference surface.wgsl surface.bin direct-light.wgsl direct-light-reference film.wgsl\n' +
    '       node verify.mjs --film film.wgsl [adaptive-film.wgsl]\n' +
    '       node verify.mjs --svm svm.wgsl svm-reference.json\n' +
    '       node verify.mjs --svm-compact svm-compact.wgsl svm-reference.json',
  );
}
const chromePath = process.env.CHROME ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const testOrigin = process.env.BLENDER_LOCAL_ORIGIN ?? 'http://127.0.0.1:8002';
const origin = new URL(testOrigin);
if (!['http:', 'https:'].includes(origin.protocol) ||
    !['127.0.0.1', 'localhost', '[::1]'].includes(origin.hostname)) {
  throw Error('Use a loopback origin from the existing Blender server');
}
const code = await readFile(args[0], 'utf8');
const payload = { mode };
if (mode === 'film' && args[1]) payload.adaptiveFilm = await readFile(args[1], 'utf8');
const verifyFilm = mode === 'film' ? (await import('./film-test.mjs')).verifyFilm : null;
if (mode === 'primary' || mode === 'surface' || mode === 'lit') {
  const [, traversalPath, scenePath, casesPath, cpuReference, surfacePath, metadataPath,
    directLightPath, directLightReference, filmPath] = args;
  payload.traversal = await readFile(traversalPath, 'utf8');
  payload.scene = await readFile(scenePath, 'base64');
  payload.cases = JSON.parse(await readFile(casesPath, 'utf8'));
  if (surfacePath) {
    payload.surface = await readFile(surfacePath, 'utf8');
    payload.surfaceMetadata = await readFile(metadataPath, 'base64');
  }
  const temporary = await mkdtemp(join(tmpdir(), 'cycles-primary-reference-'));
  const traceCpu = promisify(execFile);
  try {
    const lightingPath = join(temporary, 'lighting.bin');
    if (mode === 'lit') {
      payload.directLight = await readFile(directLightPath, 'utf8');
      payload.film = await readFile(filmPath, 'utf8');
      const bytes = Buffer.from(payload.surfaceMetadata, 'base64');
      const shadersStart = bytes.readUInt32LE(4);
      if (bytes.length % 4 || shadersStart < 2 || shadersStart >= bytes.length / 4) {
        throw Error('Invalid surface shader metadata');
      }
      const shaderMask = 0x203fffff;
      let maxShader = 0;
      for (let i = shadersStart * 4; i < bytes.length; i += 4) {
        maxShader = Math.max(maxShader, bytes.readUInt32LE(i) & shaderMask);
      }
      if (maxShader > 65535) throw Error('Surface fixture shader index is out of range');
      const palette = [[0.78, 0.2, 0.05, 0], [0.05, 0.4, 0.73, 0], [0.1, 0.55, 0.2, 0]];
      const lighting = new Float32Array(12 + 4 * (maxShader + 1));
      lighting.set([3.31, -2.71, 6.7, 0, 256, 199, 141, 0, 0.03, 0.05, 0.09, 0]);
      new Uint32Array(lighting.buffer)[3] = maxShader + 1;
      for (let shader = 0; shader <= maxShader; ++shader) {
        lighting.set(palette[shader % palette.length], 12 + 4 * shader);
      }
      const lightingBytes = Buffer.from(lighting.buffer);
      await writeFile(lightingPath, lightingBytes);
      payload.lighting = lightingBytes.toString('base64');
    }
    for (const testCase of payload.cases) {
      const { tile, expected } = testCase;
      const rays = Buffer.alloc(4 * (1 + expected.length * 12));
      rays.writeUInt32LE(expected.length);
      expected.forEach(([, ray], index) => {
        const offset = 4 + index * 48;
        ray.forEach((component, word) => rays.writeFloatLE(component, offset + word * 4));
        rays.writeUInt32LE(tile[7], offset + 32);
        rays.writeUInt32LE(0xffffffff, offset + 36);
        rays.writeUInt32LE(0xffffffff, offset + 40);
      });
      const raysPath = join(temporary, 'rays.bin');
      const hitsPath = join(temporary, 'hits.bin');
      await writeFile(raysPath, rays);
      await traceCpu(resolve(cpuReference), [resolve(scenePath), raysPath, hitsPath]);
      testCase.expectedHits = await readFile(hitsPath, 'base64');
      if (mode === 'lit') {
        const samplesPath = join(temporary, 'samples.bin');
        const { stdout } = await traceCpu(resolve(directLightReference), [
          resolve(scenePath), raysPath, hitsPath, resolve(metadataPath), lightingPath, samplesPath,
        ]);
        testCase.lightingCoverage = JSON.parse(stdout);
        testCase.expectedSamples = await readFile(samplesPath, 'base64');
      }
    }
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
} else if (mode === 'bvh2') {
  payload.scene = await readFile(args[1], 'base64');
  payload.rays = await readFile(args[2], 'base64');
  payload.expected = await readFile(args[3], 'base64');
} else if (mode === 'svm' || mode === 'svm-compact') {
  payload.svm = JSON.parse(await readFile(args[1], 'utf8'));
} else if (mode === 'camera') {
  payload.cases = JSON.parse(await readFile(args[1], 'utf8'));
}
const profile = await mkdtemp(join(tmpdir(), 'blender-cycles-webgpu-'));
const chrome = spawn(chromePath, [
  '--headless', '--no-first-run', '--no-default-browser-check',
  '--remote-debugging-pipe', '--user-data-dir=' + profile, 'about:blank',
], { stdio: ['ignore', 'ignore', 'inherit', 'pipe', 'pipe'] });

let messageId = 0;
let pending = '';
const requests = new Map();
function rejectRequests(error) {
  for (const request of requests.values()) request.reject(error);
  requests.clear();
}
chrome.once('error', rejectRequests);
chrome.once('exit', () => rejectRequests(Error('Chrome exited before verification finished')));
chrome.stdio[4].setEncoding('utf8').on('data', chunk => {
  pending += chunk;
  const messages = pending.split('\0');
  pending = messages.pop();
  for (const item of messages) {
    const msg = JSON.parse(item);
    const request = requests.get(msg.id);
    if (!request) continue;
    requests.delete(msg.id);
    if (msg.error) request.reject(Error(msg.error.message));
    else request.resolve(msg.result);
  }
});

function send(method, params = {}, sessionId) {
  return new Promise((resolve, reject) => {
    const id = ++messageId;
    requests.set(id, { resolve, reject });
    chrome.stdio[3].write(JSON.stringify({ id, method, params, sessionId }) + '\0');
  });
}

async function runGpuProbe({ code, payload }, verifyFilm) {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw Error('WebGPU adapter unavailable');
  if (adapter.info.isFallbackAdapter) throw Error('WebGPU selected a fallback adapter');
  const device = await adapter.requestDevice();
  const errors = [];
  device.addEventListener('uncapturederror', event => errors.push(event.error.message));
  async function compilePipeline(code, label, entryPoint = 'main') {
    const shader = device.createShaderModule({ label, code });
    const messages = await shader.getCompilationInfo();
    for (const message of messages.messages) {
      if (message.type === 'error') throw Error(message.message);
    }
    return device.createComputePipelineAsync({
      label,
      layout: 'auto',
      compute: { module: shader, entryPoint },
    });
  }
  const pipeline = verifyFilm ? null : await compilePipeline(code, 'Cycles ' + payload.mode);
  function storage(data, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC) {
    const buffer = device.createBuffer({
      size: data.byteLength,
      usage,
      mappedAtCreation: true,
    });
    new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer));
    buffer.unmap();
    return buffer;
  }

  function encodeDispatch(pass, pipeline, buffers, workgroups) {
    const entries = buffers.flatMap((buffer, binding) => buffer ? [{ binding, resource: { buffer } }] : []);
    const resources = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries,
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, resources);
    pass.dispatchWorkgroups(workgroups);
  }

  async function dispatch(inputs, outputs, workgroups, record = encodeDispatch, repeat = 1) {
    const buffers = [...inputs, ...outputs].map(array => storage(array));
    const readbacks = outputs.map(array => device.createBuffer({
      size: array.byteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    }));
    try {
      for (let iteration = 0; iteration < repeat; ++iteration) {
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        record(pass, pipeline, buffers, workgroups, iteration);
        pass.end();
        if (iteration === repeat - 1) {
          readbacks.forEach((readback, i) => {
            encoder.copyBufferToBuffer(buffers[inputs.length + i], 0, readback, 0, readback.size);
          });
        }
        device.queue.submit([encoder.finish()]);
      }
      await Promise.all(readbacks.map(buffer => buffer.mapAsync(GPUMapMode.READ)));
      return readbacks.map((buffer, i) => {
        return new outputs[i].constructor(buffer.getMappedRange().slice(0));
      });
    } finally {
      readbacks.forEach(buffer => buffer.destroy());
      buffers.forEach(buffer => buffer.destroy());
    }
  }

  function checkFloat(actual, expected, label, tolerance = 3e-6) {
    const error = Math.abs(actual - expected);
    if (!Number.isFinite(actual) || !Number.isFinite(expected) ||
        error > tolerance * Math.max(1, Math.abs(expected))) {
      throw Error(label + ' = ' + actual + ', expected ' + expected);
    }
    return error;
  }

  async function svmTest({ bytecode, surfaces, hits, materials }, compact) {
    const rayCount = surfaces.length / 8;
    const materialWords = 16;
    if (!Number.isInteger(rayCount) || rayCount === 0 || materials.length !== rayCount * materialWords ||
        hits.length !== surfaces.length) {
      throw Error('SVM fixture has inconsistent surface and material records');
    }
    const workgroups = Math.ceil(rayCount / 64);
    const padding = 0xBADC0FFE;
    const rays = new Uint32Array(compact ? 2 + rayCount * 13 : 1);
    rays[0] = rayCount;
    const omitted = compact ? 0 : -1;
    if (compact) {
      const base = 1 + rayCount * 12;
      rays[base] = rayCount - 1;
      let next = base + 1;
      for (let lane = rayCount - 1; lane >= 0; --lane) {
        if (lane !== omitted) rays[next++] = lane;
      }
    }
    const [result] = await dispatch(
      [new Uint32Array(bytecode), new Uint32Array(surfaces), rays, new Uint32Array(hits)],
      [new Uint32Array(workgroups * 64 * materialWords).fill(padding)], workgroups,
      (pass, pipeline, buffers, groups) => {
        return encodeDispatch(pass, pipeline, [...buffers.slice(0, 3), buffers[4], buffers[3]], groups);
      },
    );
    const floats = new Float32Array(result.buffer);
    let maxAbsoluteError = 0;
    let supported = 0;
    let principledGGX = 0;
    let principledMultiscatterGGX = 0;
    let principledTransmission = 0;
    let roughGlass = 0;
    let unsupported = 0;
    let misses = 0;
    for (let i = 0; i < rayCount; ++i) {
      const offset = i * materialWords;
      if (i === omitted) {
        if (!result.subarray(offset, offset + materialWords).every(word => word === padding)) {
          throw Error('SVM dispatch wrote into a lane omitted from the queue');
        }
        continue;
      }
      for (let word = 0; word < materialWords; ++word) {
        const error = checkFloat(floats[offset + word], materials[offset + word],
                                 `SVM material ${i} word ${word}`);
        maxAbsoluteError = Math.max(maxAbsoluteError, error);
      }
      const status = floats[offset + 3];
      if (status !== materials[offset + 3]) {
        throw Error(`SVM material ${i} status ${status}, expected ${materials[offset + 3]}`);
      }
      if (status === 3 || status === 4) {
        ++supported;
        const standalone = floats[offset + 14] === 1;
        roughGlass += Number(standalone);
        principledGGX += Number(!standalone && status === 3);
        principledMultiscatterGGX += Number(!standalone && status === 4);
        principledTransmission += Number(!standalone && floats[offset + 12] > 0);
      }
      else if (status === 1) ++supported;
      else if (status === 2) ++unsupported;
      else ++misses;
    }
    if (!result.subarray(materials.length).every(word => word === padding)) {
      throw Error('SVM dispatch wrote into the padded lanes');
    }
    return {
      raysChecked: rayCount - (compact ? 1 : 0),
      supported, principledGGX, principledMultiscatterGGX, principledTransmission, roughGlass,
      unsupported, misses, maxAbsoluteError,
    };
  }

  async function cameraTest(cases) {
    const sentinel = 44021;
    let raysChecked = 0;
    let maxAbsoluteError = 0;
    for (const { tile, camera, expected } of cases) {
      const workgroups = Math.ceil(expected.length / 64);
      const lanes = workgroups * 64;
      const inputs = [new Uint32Array(tile), new Float32Array(camera.flat())];
      const outputs = [new Uint32Array(lanes * 4), new Float32Array(lanes * 8)];
      outputs.forEach(array => array.fill(sentinel));
      const [pixels, rays] = await dispatch(inputs, outputs, workgroups);
      expected.forEach(([pixel, ray], i) => {
        for (let c = 0; c < pixel.length; c++) {
          const value = pixels[i * 4 + c];
          if (value !== pixel[c]) throw Error(`Tile result ${i}:${c} = ${value}, expected ${pixel[c]}`);
        }
        for (let c = 0; c < ray.length; c++) {
          const error = checkFloat(rays[i * 8 + c], ray[c], `Ray result ${i}:${c}`);
          maxAbsoluteError = Math.max(maxAbsoluteError, error);
        }
      });
      const checked = expected.length;
      if (!pixels.subarray(checked * 4).every(value => value === sentinel) ||
          !rays.subarray(checked * 8).every(value => value === sentinel)) {
        throw Error('Dispatch wrote outside the tile work size');
      }
      raysChecked += checked;
    }
    return { raysChecked, maxAbsoluteError };
  }

  function decodeWords(data) {
    const bytes = Uint8Array.fromBase64(data);
    if (!bytes.length || bytes.length % 4 !== 0) {
      throw Error('BVH fixture needs a nonempty sequence of 32-bit words');
    }
    return new Uint32Array(bytes.buffer);
  }

  const sentinel = 0xBADC0FFE;
  const blankWords = count => new Uint32Array(count).fill(sentinel);

  function compareHits(actual, expected, rayCount, barycentricTolerance = 3e-6) {
    if (expected.length !== rayCount * 8 || actual.length < expected.length) {
      throw Error('Invalid BVH hit buffer length');
    }
    const actualFloat = new Float32Array(actual.buffer);
    const expectedFloat = new Float32Array(expected.buffer);
    const maxAbsoluteError = { t: 0, u: 0, v: 0 };
    const components = Object.keys(maxAbsoluteError);
    const primitives = new Set();
    const objects = new Set();
    let hits = 0;
    for (let ray = 0; ray < rayCount; ray++) {
      const start = ray * 8;
      for (const word of [0, 1, 2, 3, 7]) {
        if (actual[start + word] !== expected[start + word]) {
          throw Error(`BVH ray ${ray} word ${word} = ${actual[start + word]}, expected ${expected[start + word]}`);
        }
      }
      for (const [index, name] of components.entries()) {
        const word = start + 4 + index;
        if (actual[word] === expected[word]) continue;
        const tolerance = name === 't' ? 3e-6 : barycentricTolerance;
        const error = checkFloat(actualFloat[word], expectedFloat[word], `BVH ray ${ray} ${name}`,
                                 tolerance);
        maxAbsoluteError[name] = Math.max(maxAbsoluteError[name], error);
      }
      if (actual[start] === 1) {
        hits++;
        primitives.add(actual[start + 1]);
        objects.add(actual[start + 2]);
      }
    }
    if (!actual.subarray(rayCount * 8).every(value => value === sentinel)) {
      throw Error('Dispatch wrote beyond the specified ray count');
    }
    return {
      raysChecked: rayCount,
      hits,
      misses: rayCount - hits,
      distinctPrimitives: primitives.size,
      distinctObjects: objects.size,
      maxAbsoluteError,
    };
  }

  async function bvhTest(scene, rays, expected) {
    const rayCount = rays[0];
    if (scene.length < 10 || rays.length !== 1 + rayCount * 12 || expected.length !== rayCount * 8) {
      throw Error('Invalid BVH fixture layout');
    }
    const workgroups = Math.ceil(rayCount / 64);
    const [actual] = await dispatch([scene, rays], [blankWords(workgroups * 64 * 8)], workgroups);
    return compareHits(actual, expected, rayCount);
  }

  function compareSurfacePoints(points, scene, metadata, expected, cameraRays) {
    const positions = new Float32Array(points.buffer);
    const sceneFloats = new Float32Array(scene.buffer);
    const hitFloats = new Float32Array(expected.buffer);
    const transformApplied = 1 << 2;
    const negativeScale = 1 << 3;
    const smoothNormal = 0x80000000;
    let frontFaces = 0;
    let backFaces = 0;
    let unsupported = 0;
    const shaders = new Set();
    let maxPositionError = 0;
    let maxNormalError = 0;
    for (let i = 0; i < cameraRays.length; ++i) {
      const base = i * 8;
      const [, ray] = cameraRays[i];
      const [hit, primitive, object, type] = expected.subarray(base, base + 4);
      const shader = hit ? metadata[metadata[1] + primitive] : 0xffffffff;
      const unsupportedHit = hit && (type !== 1 || (shader & smoothNormal));
      if (!hit || unsupportedHit) {
        const status = unsupportedHit ? 3 : 0;
        const actual = points.subarray(base, base + 8);
        if (!actual.every((word, index) => word === (index === 3 ? status :
                                                      index === 7 ? 0xffffffff : 0))) {
          throw Error(`Unexpected surface data for ray ${i}`);
        }
        unsupported += Number(Boolean(unsupportedHit));
        continue;
      }

      const objectRecord = scene[7] + object * 13;
      const flags = metadata[metadata[0] + object * 13];
      const vertex = corner => {
        const index = scene[scene[8] + primitive * 3 + corner];
        const offset = scene[9] + ((scene[objectRecord] | 0) + index) * 3;
        return sceneFloats.subarray(offset, offset + 3);
      };
      const [a, b, c] = [vertex(0), vertex(1), vertex(2)];
      const [ab, ac] = [b, c].map(v => [...v].map((value, axis) => value - a[axis]));
      let normal = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
      ];
      if (!(flags & transformApplied)) {
        const inverse = objectRecord + 1;
        normal = normal.map((_, axis) => normal[0] * sceneFloats[inverse + axis] +
                                         normal[1] * sceneFloats[inverse + 4 + axis] +
                                         normal[2] * sceneFloats[inverse + 8 + axis]);
      } else if (flags & negativeScale) {
        normal = normal.map(value => -value);
      }
      const backface = normal.reduce((sum, value, axis) => sum + value * ray[4 + axis], 0) > 0;
      const scale = (backface ? -1 : 1) / Math.hypot(...normal);
      if (points[base + 3] !== (backface ? 2 : 1) || points[base + 7] !== shader) {
        throw Error(`Surface orientation or packed shader differs for ray ${i}`);
      }
      if (backface) backFaces++;
      else frontFaces++;
      shaders.add(shader);
      for (let axis = 0; axis < 3; ++axis) {
        const world = ray[axis] + hitFloats[base + 4] * ray[4 + axis];
        maxPositionError = Math.max(maxPositionError,
          checkFloat(positions[base + axis], world, `Surface position ray ${i} axis ${axis}`, 2e-5));
        maxNormalError = Math.max(maxNormalError,
          checkFloat(positions[base + 4 + axis], normal[axis] * scale, `Surface normal ray ${i} axis ${axis}`));
      }
    }
    if (!points.subarray(cameraRays.length * 8).every(value => value === sentinel)) {
      throw Error('Surface dispatch wrote beyond the ray count');
    }
    return { frontFaces, backFaces, unsupported, distinctShaders: shaders.size,
             maxPositionError, maxNormalError };
  }

  function compareFilm(samples, expectedSamples, cameraRays, tile, film, firstImage, image) {
    const cpuSamples = new Float32Array(decodeWords(expectedSamples).buffer);
    if (cpuSamples.length !== cameraRays.length * 4) throw Error('Invalid CPU sample buffer');
    const gpuSamples = new Float32Array(samples.buffer);
    const filmValues = new Float32Array(film.buffer);
    const [x, y, width, height, , sampleCount] = tile;
    const sums = new Float64Array(width * height * 4);
    let maxRadianceError = 0;
    for (const [i, [[pixelX, pixelY]]] of cameraRays.entries()) {
      const offset = 4 * ((pixelY - y) * width + pixelX - x);
      for (let component = 0; component < 4; ++component) {
        const rayWord = i * 4 + component;
        const cpu = cpuSamples[rayWord];
        maxRadianceError = Math.max(maxRadianceError,
          checkFloat(gpuSamples[rayWord], cpu, `Sample ${i} channel ${component}`, 2e-5));
        sums[offset + component] += cpu;
      }
    }
    if (!samples.subarray(cpuSamples.length).every(word => word === sentinel)) {
      throw Error('Lighting dispatched outside the ray count');
    }
    let maxFilmError = 0;
    for (let pixel = 0; pixel < width * height; pixel++) {
      if (film[pixel * 5 + 4] !== 2 * sampleCount) {
        throw Error(`Film pixel ${pixel} has the wrong sample count`);
      }
      for (let component = 0; component < 4; ++component) {
        const offset = pixel * 4 + component;
        const sum = sums[offset];
        checkFloat(filmValues[pixel * 5 + component], sum * 2,
                   `Film pixel ${pixel} sum channel ${component}`, 2e-5);
        const expected = sum / sampleCount;
        checkFloat(firstImage[offset], expected, `First resolve channel ${offset}`, 2e-5);
        maxFilmError = Math.max(maxFilmError,
          checkFloat(image[offset], expected, `Final resolve channel ${offset}`, 2e-5));
      }
    }
    return { filmPixels: width * height, maxRadianceError, maxFilmError };
  }

  async function primaryTest(scene, cases) {
    const traversal = await compilePipeline(payload.traversal, 'Cycles primary BVH2');
    const surface = payload.surface ? await compilePipeline(payload.surface, 'Cycles triangle surface') : null;
    const lighting = payload.lighting ? decodeWords(payload.lighting) : null;
    const directLight = lighting ? await compilePipeline(payload.directLight, 'Cycles direct light and shadows') : null;
    const accumulate = lighting ? await compilePipeline(payload.film, 'Cycles film accumulation', 'accumulate') : null;
    const resolveFilm = lighting ? await compilePipeline(payload.film, 'Cycles film resolve', 'resolve') : null;
    const metadata = surface ? decodeWords(payload.surfaceMetadata) : null;
    if (surface && (metadata.length < 2 || metadata[0] !== 2 || metadata[1] > metadata.length)) {
      throw Error('Invalid surface metadata');
    }
    const results = [];
    for (const { tile, camera, sampling, expected, expectedHits, expectedSamples, lightingCoverage } of cases) {
      const rayCount = expected.length;
      if (tile.length !== 8 || tile[6] !== rayCount || rayCount === 0) {
        throw Error('Invalid primary camera tile layout');
      }
      const workgroups = Math.ceil(rayCount / 64);
      const lanes = workgroups * 64;
      if (lighting && sampling) {
        throw Error('The lighting fixture needs unchanged rays between submissions');
      }
      const inputs = [new Uint32Array(tile), new Float32Array(camera.flat()),
                      new Uint32Array(sampling ?? [0xffffffff]), scene];
      const outputs = [blankWords(lanes * 4), blankWords(1 + lanes * 12), blankWords(lanes * 8)];
      if (surface) {
        inputs.push(metadata);
        outputs.push(blankWords(lanes * 8));
      }
      let filmUniform;
      const filmPixels = tile[2] * tile[3];
      if (lighting) {
        const secondTile = new Uint32Array(tile);
        // The center-ray fixture produces the same radiance for the next sample range.
        secondTile[4] += tile[5];
        const filmConfig = new Uint32Array(8);
        filmConfig.set(tile.slice(0, 4));
        filmConfig[4] = rayCount;
        new Float32Array(filmConfig.buffer)[5] = 1;
        filmUniform = storage(filmConfig, GPUBufferUsage.UNIFORM);
        inputs.push(lighting, secondTile);
        outputs.push(blankWords(lanes * 4), new Uint32Array(filmPixels * 5),
                     new Float32Array(filmPixels * 4), new Float32Array(filmPixels * 4));
      }
      let output;
      try {
        output = await dispatch(inputs, outputs, workgroups,
          (pass, cameraPipeline, buffers, groups, submission) => {
            const [tile, camera, sampling, scene, metadata, lighting, secondTile] = buffers;
            const [pixels, rays, hits, points, samples, film, firstImage, image] = buffers.slice(inputs.length);
            encodeDispatch(pass, cameraPipeline, [submission === 0 ? tile : secondTile, camera, pixels, rays, sampling], groups);
            encodeDispatch(pass, traversal, [scene, rays, hits], groups);
            if (surface) encodeDispatch(pass, surface, [scene, metadata, rays, hits, points], groups);
            if (directLight) {
              encodeDispatch(pass, directLight, [scene, rays, hits, points, lighting, samples], groups);
              encodeDispatch(pass, accumulate, [filmUniform, pixels, samples, film], groups);
              encodeDispatch(pass, resolveFilm, [filmUniform, , , film, submission === 0 ? firstImage : image],
                             Math.ceil(filmPixels / 64));
            }
          }, lighting ? 2 : 1);
      } finally {
        filmUniform?.destroy();
      }
      const [pixels, rays, hits, points, samples, film, firstImage, image] = output;
      if (rays[0] !== rayCount || !pixels.subarray(rayCount * 4).every(value => value === sentinel) ||
          !rays.subarray(1 + rayCount * 12).every(value => value === sentinel)) {
        throw Error('The camera dispatch wrote an invalid ray count or touched the dispatch tail');
      }
      const rayFloats = new Float32Array(rays.buffer);
      expected.forEach(([pixel, ray], i) => {
        for (let word = 0; word < 4; ++word) {
          const sampleOffset = lighting && word === 2 ? tile[5] : 0;
          if (pixels[i * 4 + word] !== pixel[word] + sampleOffset) {
            throw Error(`Primary ray ${i} has incorrect tile coordinates`);
          }
        }
        const offset = 1 + i * 12;
        for (let word = 0; word < 8; ++word) {
          checkFloat(rayFloats[offset + word], ray[word], `Primary ray ${i} word ${word}`);
        }
        if (rays[offset + 8] !== tile[7] || rays[offset + 9] !== 0xffffffff ||
            rays[offset + 10] !== 0xffffffff || rays[offset + 11] !== 0) {
          throw Error(`Primary ray ${i} has incorrect ray options`);
        }
      });
      // This comparison includes rounding from generating independent CPU and GPU rays.
      const cpuHits = decodeWords(expectedHits);
      results.push({
        visibility: tile[7],
        ...compareHits(hits, cpuHits, rayCount, 1e-5),
        ...(surface && compareSurfacePoints(points, scene, metadata, cpuHits, expected)),
        ...(lighting && {
          ...compareFilm(samples, expectedSamples, expected, tile, film, firstImage, image),
          lightingCoverage,
        }),
      });
    }
    return {
      raysChecked: results.reduce((sum, result) => sum + result.raysChecked, 0),
      hits: results.reduce((sum, result) => sum + result.hits, 0),
      misses: results.reduce((sum, result) => sum + result.misses, 0),
      cases: results,
    };
  }

  let result;
  if (verifyFilm) {
    result = await verifyFilm(device, code, payload.adaptiveFilm);
  } else if (payload.mode === 'primary' || payload.mode === 'surface' || payload.mode === 'lit') {
    result = await primaryTest(decodeWords(payload.scene), payload.cases);
    if (payload.mode === 'lit') {
      for (const kind of ['misses', 'backlit', 'shadowed', 'lit', 'unsupported']) {
        if (!result.cases.some(({ lightingCoverage }) => lightingCoverage[kind] > 0)) {
          throw Error(`Lighting fixture did not exercise ${kind}`);
        }
      }
    }
  } else if (payload.mode === 'bvh2') {
    result = await bvhTest(decodeWords(payload.scene), decodeWords(payload.rays),
                           decodeWords(payload.expected));
  } else if (payload.mode === 'svm' || payload.mode === 'svm-compact') {
    result = await svmTest(payload.svm, payload.mode === 'svm-compact');
  } else {
    result = await cameraTest(payload.cases);
  }
  await device.queue.onSubmittedWorkDone();
  if (errors.length) throw Error(errors.join('\n'));
  return {
    adapter: {
      vendor: adapter.info.vendor,
      architecture: adapter.info.architecture,
      fallback: adapter.info.isFallbackAdapter,
    },
    dedicatedWorker: globalThis instanceof DedicatedWorkerGlobalScope,
    ...result,
  };
}

const timer = setTimeout(() => chrome.kill(), 30000);
try {
  const { targetId } = await send('Target.createTarget', { url: 'about:blank' });
  const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true });
  await send('Page.enable', {}, sessionId);
  const { errorText } = await send('Page.navigate', { url: origin.origin + '/favicon.ico' }, sessionId);
  if (errorText) throw Error(errorText);
  const expression = `(async () => {
    const source = ${JSON.stringify(`onmessage = async ({ data }) => {
      try { postMessage({ result: await (${runGpuProbe})(data, ${verifyFilm}) }); }
      catch (error) { postMessage({error: error.stack}); }
    }`)};
    const blobUrl = URL.createObjectURL(new Blob([source], { type: 'text/javascript' }));
    const worker = new Worker(blobUrl);
    try {
      return await new Promise((resolve, reject) => {
        worker.onmessage = ({ data }) => data.error ? reject(Error(data.error)) : resolve(data.result);
        worker.onerror = event => reject(Error(event.message));
        worker.postMessage(${JSON.stringify({ code, payload })});
      });
    } finally {
      worker.terminate();
      URL.revokeObjectURL(blobUrl);
    }
  })()`;
  const { result, exceptionDetails } = await send('Runtime.evaluate', {
    expression, awaitPromise: true, returnByValue: true,
  }, sessionId);
  if (exceptionDetails) throw Error(exceptionDetails.exception?.description ?? exceptionDetails.text);
  if (!result.value?.dedicatedWorker) throw Error('Kernel did not run in a dedicated worker');
  const resultTag = {
    camera: 'CYCLES_WEBGPU_PRIMARY_RAYS_PASS',
    bvh2: 'CYCLES_WEBGPU_BVH2_PASS',
    primary: 'CYCLES_WEBGPU_PRIMARY_INTERSECTION_PASS',
    surface: 'CYCLES_WEBGPU_SURFACE_PASS',
    lit: 'CYCLES_WEBGPU_LIT_FILM_PASS',
    film: 'CYCLES_WEBGPU_FILM_PASS',
    svm: 'CYCLES_WEBGPU_SVM_PASS',
    'svm-compact': 'CYCLES_WEBGPU_SVM_COMPACT_PASS',
  };
  console.log(resultTag[mode], JSON.stringify(result.value));
} finally {
  clearTimeout(timer);
  if (chrome.exitCode === null && chrome.signalCode === null) {
    const exited = once(chrome, 'exit');
    chrome.kill();
    await exited;
  }
  await rm(profile, { recursive: true, force: true, maxRetries: 5 });
}
