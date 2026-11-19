// SPDX-License-Identifier: Apache-2.0

// This function is serialized into the verifier's dedicated worker.
export async function verifyFilm(device, code, adaptiveCode) {
  const shader = device.createShaderModule({ label: 'Cycles film', code });
  for (const message of (await shader.getCompilationInfo()).messages) {
    if (message.type === 'error') throw Error(message.message);
  }
  const [accumulate, resolve, accumulateRenderBuffer, retireAdaptive] = await Promise.all(
    ['accumulate', 'resolve', 'accumulate_render_buffer', 'retire_adaptive_render_buffer'].map(entryPoint =>
      device.createComputePipelineAsync({ layout: 'auto', compute: { module: shader, entryPoint } }),
    ),
  );

  let adaptiveDelta, accumulateAdaptive;
  if (adaptiveCode) {
    const deltaShader = device.createShaderModule({label: 'Cycles adaptive film delta', code: adaptiveCode});
    for (const message of (await deltaShader.getCompilationInfo()).messages) {
      if (message.type === 'error') throw Error(message.message);
    }
    [adaptiveDelta, accumulateAdaptive] = await Promise.all([
      device.createComputePipelineAsync({ layout: 'auto', compute: {module: deltaShader, entryPoint: 'main'} }),
      device.createComputePipelineAsync({
        layout: 'auto', compute: {module: shader, entryPoint: 'accumulate_adaptive_render_buffer'},
      }),
    ]);
  }

  const workgroupSize = 64;
  const padding = 0xBADC0FFE;
  let submissions = 0;
  let samplesChecked = 0;
  let pixelsChecked = 0;
  let maxAbsoluteError = 0;

  function buffer(data, usage) {
    const result = device.createBuffer({ size: data.byteLength, usage, mappedAtCreation: true });
    new Uint8Array(result.getMappedRange()).set(new Uint8Array(data.buffer));
    result.unmap();
    return result;
  }

  function createFilm(bounds) {
    const [, , width, height] = bounds;
    const pixelCount = width * height;
    return {
      bounds,
      film: device.createBuffer({
        size: pixelCount * 20,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      }),
      resolved: buffer(new Uint32Array((pixelCount + 1) * 4).fill(padding),
                       GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC),
      sums: new Float64Array(pixelCount * 4),
      counts: new Uint32Array(pixelCount),
    };
  }

  function encodeDispatch(encoder, pipeline, bindings, count, use2D) {
    const groupCount = Math.ceil(count / workgroupSize);
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: bindings.map(([binding, buffer]) => ({ binding, resource: { buffer } })),
    }));
    pass.dispatchWorkgroups(use2D ? 1 : groupCount, use2D ? groupCount : 1);
    pass.end();
  }

  function checkFloat(actual, expected, label) {
    const error = Math.abs(actual - expected);
    if (!Number.isFinite(actual) || error > 2e-6 * Math.max(1, Math.abs(expected))) {
      throw Error(`${label} = ${actual}, expected ${expected}`);
    }
    maxAbsoluteError = Math.max(maxAbsoluteError, error);
  }

  async function dispatchAndCheck(state, paths, exposure, { clear = false, use2D = false } = {}) {
    const { bounds, film, resolved, sums, counts } = state;
    const [left, top, width, height] = bounds;
    const params = new Uint32Array(8);
    params.set(bounds);
    params[4] = paths.length;
    new Float32Array(params.buffer)[5] = exposure;
    const uniform = buffer(params, GPUBufferUsage.UNIFORM);
    const readback = [film, resolved].map(source => device.createBuffer({
      size: source.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    }));
    const temporary = [uniform, ...readback];
    try {
      const encoder = device.createCommandEncoder();
      if (clear) {
        encoder.clearBuffer(film);
        sums.fill(0);
        counts.fill(0);
      }
      if (paths.length) {
        // One extra valid-looking path tests the explicit ray count on a partial group.
        const inputs = paths.concat({ coords: [left, top, 0], rgba: [4, 4, 4, 1] });
        const coords = buffer(new Uint32Array(inputs.flatMap(path => [...path.coords, 0])),
                              GPUBufferUsage.STORAGE);
        const radiance = buffer(new Float32Array(inputs.flatMap(path => path.rgba)),
                                GPUBufferUsage.STORAGE);
        temporary.push(coords, radiance);
        encodeDispatch(encoder, accumulate, [[0, uniform], [1, coords], [2, radiance], [3, film]],
                       paths.length, use2D);
      }
      encodeDispatch(encoder, resolve, [[0, uniform], [3, film], [4, resolved]], counts.length, use2D);
      [film, resolved].forEach((source, index) => {
        encoder.copyBufferToBuffer(source, 0, readback[index], 0, source.size);
      });
      device.queue.submit([encoder.finish()]);
      submissions++;

      for (const { coords: [x, y], rgba } of paths) {
        if (x < left || x >= left + width || y < top || y >= top + height) continue;
        const pixel = (y - top) * width + x - left;
        for (let channel = 0; channel < 4; channel++) {
          sums[4 * pixel + channel] += Number.isFinite(rgba[channel]) ? rgba[channel] : 0;
        }
        counts[pixel]++;
        samplesChecked++;
      }

      await Promise.all(readback.map(result => result.mapAsync(GPUMapMode.READ)));
      const filmWords = new Uint32Array(readback[0].getMappedRange());
      const filmFloats = new Float32Array(filmWords.buffer);
      const rgbaWords = new Uint32Array(readback[1].getMappedRange());
      const rgbaFloats = new Float32Array(rgbaWords.buffer);
      counts.forEach((count, pixel) => {
        if (filmWords[5 * pixel + 4] !== count) throw Error(`Pixel ${pixel} sample count mismatch`);
        for (let channel = 0; channel < 4; channel++) {
          const sum = sums[4 * pixel + channel];
          checkFloat(filmFloats[5 * pixel + channel], sum, `Pixel ${pixel}:${channel} sum`);
          const mean = count ? sum / count : 0;
          const expected = channel === 3 ? Math.min(Math.max(mean, 0), 1) : mean * exposure;
          checkFloat(rgbaFloats[4 * pixel + channel], expected, `Pixel ${pixel}:${channel} resolve`);
        }
      });
      if (!rgbaWords.subarray(4 * counts.length).every(word => word === padding)) {
        throw Error('Film resolve wrote outside the region');
      }
      pixelsChecked += counts.length;
    } finally {
      temporary.forEach(resource => resource.destroy());
    }
  }

  async function checkRenderBuffer(withSampleCount, adaptiveMode) {
    const adaptive = adaptiveMode !== undefined;
    const bounds = [90, 110, 7, 4];
    const [left, top, width, height] = bounds;
    const [firstPixel, rowStride, passStride, combinedOffset] = [6, 11, adaptive ? 12 : 9, 3];
    const countOffset = withSampleCount ? 1 : 0xffffffff;
    const auxOffset = adaptive ? 7 : 0xffffffff;
    const flagWord = 0x3f800000; // 1.0 preserves a prior convergence test's marker.
    const params = new Uint32Array(12);
    params.set([...bounds, firstPixel, rowStride, passStride, combinedOffset, countOffset, auxOffset]);
    const words = new Uint32Array((firstPixel + rowStride * (height + 1)) * passStride).fill(padding);
    const sums = new Float64Array(words.length);
    const expectedCounts = new Uint32Array(words.length);
    const passWords = new Set();
    function pixelBase(x, y) {
      return (firstPixel + (y - top) * rowStride + x - left) * passStride;
    }
    for (let y = top; y < top + height; y++) {
      for (let x = left; x < left + width; x++) {
        const pixel = pixelBase(x, y);
        for (let channel = 0; channel < 4; channel++) {
          passWords.add(pixel + combinedOffset + channel);
        }
        if (adaptive) {
          for (let channel = 0; channel < 4; channel++) {
            passWords.add(pixel + auxOffset + channel);
          }
        }
        if (withSampleCount) {
          passWords.add(pixel + countOffset);
          expectedCounts[pixel + countOffset] = 1;
        }
      }
    }
    for (const word of passWords) {
      words[word] = adaptive && word % passStride === auxOffset + 3 ? flagWord : expectedCounts[word];
    }

    const film = buffer(words, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const uniform = buffer(params, GPUBufferUsage.UNIFORM | GPUBufferUsage.STORAGE);
    const readback = device.createBuffer({
      size: words.byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const resources = [film, uniform, readback];
    let retired = 0;
    try {
      const encoder = device.createCommandEncoder();
      const capacity = 263;
      const queueBase = 1 + 12 * capacity;
      const classBase = queueBase + 1 + capacity;
      const pathStride = 23;
      // Contend on the same pixel from independent sample lanes across
      // workgroups, then reuse those physical slots with different fragments.
      for (let wave = 0; wave < 2; wave++) {
        const queues = new Uint32Array(classBase + (adaptiveMode === 'retire' ? capacity : 0));
        const paths = new Uint32Array(pathStride * capacity);
        const floats = new Float32Array(paths.buffer);
        const count = 131 - wave * 62;
        queues[0] = capacity;
        queues[queueBase] = count;
        for (let i = 0; i <= count; i++) {
          const lane = (i * 11 + 7) % capacity;
          const path = lane * pathStride;
          const x = left + (i % 5 === 0 ? -1 : (i % 3 === 0 ? i % width : 1));
          const y = top + (i % 13 === 0 ? height : 2);
          const rgba = [i % 17 * 0.125, wave - 0.5, i % 2 === 0 ? 7 : -0.25, i % 4 * 0.25];
          if (i === 2) rgba[0] = Infinity;
          // Leave one terminated ray beyond the queue tail and queued
          // intersections among traced paths to catch premature retirement.
          const live = i % 7 === 0;
          queues[queueBase + 1 + i] = lane;
          const sample = i + wave * 193;
          let bits = sample & 0xaaaaaaaa;
          let classA = false;
          for (; bits !== 0; bits &= bits - 1) classA = !classA;
          if (adaptiveMode === 'retire') queues[classBase + lane] = Number(classA);
          paths.set([x, y, sample], path + 18);
          paths[path + 7] = live ? 3 : 0;
          floats.set(rgba, path + 3);
          if (i === count || live || x < left || y >= top + height) continue;
          const pixel = pixelBase(x, y);
          const combined = pixel + combinedOffset;
          rgba[3] = 1 - rgba[3];
          rgba.forEach((value, channel) => {
            sums[combined + channel] += Number.isFinite(value) ? value : 0;
          });
          if (adaptive) {
            if (classA) {
              for (let channel = 0; channel < 3; channel++) {
                sums[pixel + auxOffset + channel] += Number.isFinite(rgba[channel]) ? 2 * rgba[channel] : 0;
              }
            }
          } else if (withSampleCount) {
            expectedCounts[pixel + countOffset]++;
          }
          retired++;
        }
        const pathBuffer = buffer(paths, GPUBufferUsage.STORAGE);
        const rayBuffer = buffer(queues, GPUBufferUsage.STORAGE);
        resources.push(pathBuffer, rayBuffer);
        const bindings = [[3, film], [5, pathBuffer], [6, rayBuffer], [7, uniform]];
        if (adaptiveMode === 'retire') {
          encodeDispatch(encoder, retireAdaptive, bindings, count + 1, false);
        } else if (adaptive) {
          const deltas = device.createBuffer({size: capacity * 32, usage: GPUBufferUsage.STORAGE});
          resources.push(deltas);
          encodeDispatch(encoder, adaptiveDelta,
                         [[0, pathBuffer], [1, rayBuffer], [2, uniform], [3, deltas]], count + 1, false);
          encodeDispatch(encoder, accumulateAdaptive, [...bindings, [8, deltas]], count + 1, false);
        } else {
          encodeDispatch(encoder, accumulateRenderBuffer, bindings, count + 1, wave === 0);
        }
      }
      encoder.copyBufferToBuffer(film, 0, readback, 0, film.size);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const result = new Uint32Array(readback.getMappedRange());
      const values = new Float32Array(result.buffer);
      for (const [word, value] of result.entries()) {
        if (!passWords.has(word)) {
          if (value !== padding) throw Error(`Changed padding or unrelated pass word ${word}`);
        } else if (adaptive && word % passStride === auxOffset + 3) {
          if (value !== flagWord) throw Error(`Changed convergence flag word ${word}`);
        } else if (word % passStride === countOffset) {
          if (value !== expectedCounts[word]) throw Error(`Sample count word ${word} mismatch`);
        } else {
          checkFloat(values[word], sums[word], `Combined word ${word}`);
        }
      }
    } finally {
      resources.forEach(resource => resource.destroy());
    }
    return retired;
  }

  const state = createFilm([100, 200, 13, 7]);
  const resized = createFilm([400, 500, 3, 2]);
  try {
    // Repeated pixels across workgroups contend on the same film records. Preserve empty pixels.
    const paths = Array.from({ length: 259 }, (_, i) => {
      const index = i % 3 === 0 ? 7 : (19 * i + 5) % 73;
      const coords = [100 + (index % 13), 200 + Math.floor(index / 13), 2000 + i * 3];
      return { coords, rgba: [i % 8 / 8, i % 7 - 3, i % 11, i % 4 / 4] };
    });
    paths.push({ coords: [99, 200, 0], rgba: [4, 4, 4, 1] },
               { coords: [112, 207, 0], rgba: [4, 4, 4, 1] },
               { coords: [112, 206, 77], rgba: [Infinity, NaN, -Infinity, 1.5] });
    await dispatchAndCheck(state, [], 1);
    await dispatchAndCheck(state, paths, 1, { use2D: true });
    await dispatchAndCheck(state, paths.slice(64, 165), 2);
    await dispatchAndCheck(state, [], 0.5);
    await dispatchAndCheck(state, [], 1, { clear: true });
    await dispatchAndCheck(state, paths.slice(0, 3), 1);
    await dispatchAndCheck(resized, [{ coords: [402, 501, 9], rgba: [2, -1, 0.5, 1] }], 1);
  } finally {
    for (const { film, resolved } of [state, resized]) {
      film.destroy();
      resolved.destroy();
    }
  }
  const nativeFilmPathsChecked = await checkRenderBuffer(true) + await checkRenderBuffer(false);
  const adaptiveFilmPathsChecked = adaptiveDelta ? await checkRenderBuffer(true, 'stock') : 0;
  const adaptiveRetirementPathsChecked = await checkRenderBuffer(true, 'retire');
  return { submissions, samplesChecked, pixelsChecked, nativeFilmPathsChecked, adaptiveFilmPathsChecked,
    adaptiveRetirementPathsChecked, maxAbsoluteError };

}
