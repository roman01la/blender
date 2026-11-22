// SPDX-License-Identifier: GPL-2.0-or-later

// Injected into the render worker before it creates the device. Count only
// dispatches and transfers on command buffers submitted to an actual GPUQueue.
export function observeVolumeGpu() {
  if (!globalThis.GPUQueue || globalThis.volumeGpuObserver) return;
  globalThis.volumeGpuObserver = true;
  const encoderWork = new WeakMap();
  const passWork = new WeakMap();
  const commandWork = new WeakMap();
  const readSources = new WeakMap();
  let queues = new Map();
  let otherFilmReads = [];
  const exportLabel = 'oidn/film/export-readback';
  const isFilm = label => label === 'WebGPU Cycles film' ||
    /^oidn\/(?:film\/|input\/\d+$|output$|readback$)/.test(label);

  const timestamp = () => performance.timeOrigin + performance.now();

  const track = (prototype, method, measure) => {
    const invoke = prototype[method];
    prototype[method] = function (...args) {
      const result = invoke.apply(this, args);
      measure(this, result, args);
      return result;
    };
  };
  const encoderState = encoder => {
    if (!encoderWork.has(encoder)) {
      encoderWork.set(encoder, { trace: 0, oidn: 0, copies: [] });
    }
    return encoderWork.get(encoder);
  };
  track(GPUCommandEncoder.prototype, 'beginComputePass', (encoder, pass) => {
    passWork.set(pass, { work: encoderState(encoder), label: '' });
  });
  track(GPUComputePassEncoder.prototype, 'setPipeline', (pass, _, [pipeline]) => {
    passWork.get(pass).label = pipeline.label;
  });
  const recordDispatch = pass => {
    const { label, work } = passWork.get(pass);
    if (/^integrator_(?:init_from_camera|intersect_closest|shade_surface)$/.test(label)) {
      work.trace++;
    } else if (/\boidn\b/i.test(label)) {
      work.oidn++;
    }
  };
  track(GPUComputePassEncoder.prototype, 'dispatchWorkgroups', (pass, _, dimensions) => {
    if (dimensions.every(size => size > 0)) recordDispatch(pass);
  });
  track(GPUComputePassEncoder.prototype, 'dispatchWorkgroupsIndirect', recordDispatch);
  track(GPUCommandEncoder.prototype, 'copyBufferToBuffer',
    (encoder, _, [source, , destination]) => {
      if (destination.usage & GPUBufferUsage.MAP_READ) {
        encoderState(encoder).copies.push([source.label, destination]);
      }
    });
  track(GPUCommandEncoder.prototype, 'finish', (encoder, command) => {
    if (encoderWork.has(encoder)) commandWork.set(command, encoderWork.get(encoder));
  });
  track(GPUQueue.prototype, 'submit', (queue, _, [commands]) => {
    let trace = 0;
    let oidn = 0;
    for (const command of commands) {
      const work = commandWork.get(command);
      if (!work) continue;
      trace += work.trace;
      oidn += work.oidn;
      for (const [source, destination] of work.copies) {
        readSources.set(destination, { source, queue });
      }
    }
    if (trace || oidn) {
      if (!queues.has(queue)) queues.set(queue, []);
      queues.get(queue).push({ time: timestamp(), trace, oidn });
    }
  });
  track(GPUBuffer.prototype, 'getMappedRange', (buffer, data) => {
    const copy = readSources.get(buffer);
    if (buffer.label === exportLabel) {
      console.log('VOLUME_WEBGPU_PROOF ' + JSON.stringify({
        submissions: queues.get(copy?.queue) ?? [],
        exportBytes: data.byteLength,
        exportSource: copy?.source,
        otherFilmReads,
      }));
      queues = new Map();
      otherFilmReads = [];
    } else if (copy && (isFilm(copy.source) || isFilm(buffer.label))) {
      otherFilmReads.push({
        time: timestamp(), source: copy.source, destination: buffer.label, bytes: data.byteLength,
      });
    }
  });
}
