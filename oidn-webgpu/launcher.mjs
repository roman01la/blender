// SPDX-License-Identifier: Apache-2.0

import {createOIDNDenoiser} from "./oidn-webgpu.mjs";

const denoisers = new WeakMap();

async function loadWeights(name) {
  if (!/^rt_[a-z0-9_]+\.tza$/.test(name)) {
    throw new Error(`Invalid OIDN model name: ${name}`);
  }
  const response = await fetch(new URL(`./oidn-weights/${name}`, import.meta.url));
  if (!response.ok) {
    throw new Error(`Couldn't load OIDN model ${name}: ${response.status}`);
  }
  return response.arrayBuffer();
}

export async function blenderOIDNWebGPU(device, request) {
  if (!device) throw new Error("No WebGPU device is available for OIDN.");
  let denoiser = denoisers.get(device);
  if (!denoiser) {
    denoiser = createOIDNDenoiser({device, loadWeights});
    denoisers.set(device, denoiser);
  }
  return denoiser.denoise(request);
}
