/* SPDX-License-Identifier: Apache-2.0 */

const devices = new WeakMap();

const shader = /* wgsl */ `
struct Film {
  width: u32, height: u32, origin: u32, stride: u32, passStride: u32,
  combined: u32, sampleCount: u32, samples: u32, exposure: f32,
}
@group(0) @binding(0) var<storage, read> pixels: array<u32>;
@group(0) @binding(1) var<uniform> film: Film;

@vertex fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  let position = array(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));
  return vec4f(position[index], 0, 1);
}

@fragment fn combined(@builtin(position) position: vec4f) -> @location(0) vec4f {
  let pixel = vec2u(position.xy);
  let base = (film.origin + pixel.y * film.stride + pixel.x) * film.passStride;
  var samples = film.samples;
  if (film.sampleCount != 0xffffffffu) {
    samples = pixels[base + film.sampleCount];
  }
  if (samples == 0u) {
    return vec4f(0);
  }
  // Cycles film_get_pass_pixel_combined(), with scene-linear RGB and stock alpha.
  let scale = 1.0 / f32(samples);
  let offset = base + film.combined;
  let rgb = vec3f(bitcast<f32>(pixels[offset]), bitcast<f32>(pixels[offset + 1u]),
                  bitcast<f32>(pixels[offset + 2u]));
  let alpha = clamp(1.0 - bitcast<f32>(pixels[offset + 3u]) * scale, 0.0, 1.0);
  return vec4f(rgb * (scale * film.exposure), alpha);
}
`;

async function createPreview(device, width, height) {
  const label = "cycles/preview/combined";
  const module = device.createShaderModule({ label, code: shader });
  const pipeline = await device.createRenderPipelineAsync({
    label,
    layout: "auto",
    vertex: { module, entryPoint: "vertex" },
    fragment: { module, entryPoint: "combined", targets: [{ format: "rgba16float" }] },
  });
  const parameters = device.createBuffer({
    label: "cycles/preview/parameters",
    size: 48,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const canvas = new OffscreenCanvas(width, height);
  const context = canvas.getContext("webgpu");
  context.configure({
    device,
    format: "rgba16float",
    colorSpace: "srgb",
    alphaMode: "premultiplied",
    toneMapping: { mode: "extended" },
  });
  return { canvas, context, pipeline, parameters };
}

export async function publishRenderPreview(device, source, layout) {
  const [ticket, width, height] = layout;
  if (!devices.has(device)) {
    devices.set(device, createPreview(device, width, height));
  }
  const { canvas, context, pipeline, parameters } = await devices.get(device);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  device.queue.writeBuffer(parameters, 0, layout.subarray(1));
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [source, parameters].map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
  const commands = device.createCommandEncoder({ label: "cycles/preview" });
  const pass = commands.beginRenderPass({
    label: "cycles/preview/combined",
    colorAttachments: [{
      view: context.getCurrentTexture().createView(),
      loadOp: "clear",
      storeOp: "store",
    }],
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.draw(3);
  pass.end();
  device.queue.submit([commands.finish()]);

  // The next trace batch can reuse the film after the preview commands are submitted.
  // Keep the bitmap in the same row order as the stock half4 display upload.
  const bitmap = canvas.transferToImageBitmap();
  try {
    self.postMessage({ type: "cycles-webgpu-preview", ticket, bitmap }, [bitmap]);
  }
  catch (error) {
    bitmap.close();
    throw error;
  }
}
