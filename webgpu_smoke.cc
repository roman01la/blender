#include <emscripten.h>
#include <emscripten/html5.h>
#include "GHOST_WebGPUCanvas.hh"
#include "webgpu_buffer.hh"

#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>

#ifdef __EMSCRIPTEN_PTHREADS__
#  include <atomic>
#  include <thread>
#endif

namespace {

constexpr char canvas_selector[] = "#gpu";
constexpr char shader_source[] = R"(
@group(0) @binding(0) var<storage, read_write> vertices: array<vec4<f32>>;

@compute @workgroup_size(3)
fn generate(@builtin(local_invocation_index) index: u32) {
  let data = array(
    vec4f( 0.0,  0.7, 1.0, 0.0),
    vec4f(-0.7, -0.6, 0.0, 1.0),
    vec4f( 0.7, -0.6, 0.0, 0.0),
  );
  vertices[index] = data[index];
}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) color: vec3f,
}

@vertex
fn vertex_main(@location(0) vertex: vec4f) -> VertexOutput {
  return VertexOutput(vec4f(vertex.xy, 0.0, 1.0),
                      vec3f(vertex.zw, 1.0 - vertex.z - vertex.w));
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4f {
  return vec4f(input.color, 1.0);
}
)";

EM_JS(void, show_status, (const char *status), {
  Module.setGpuStatus(UTF8ToString(Number(status)));
});

std::unique_ptr<GHOST_WebGPUCanvas> canvas;

void verify_buffer_transfers(const wgpu::Device &device)
{
  using blender::gpu::WebGPUBuffer;
  const std::array<uint8_t, 11> contents = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const WebGPUBuffer source(device, contents.size(), wgpu::BufferUsage::Storage, "Source");
  source.write(contents.data());
  source.read_async([source, contents](std::span<const std::byte> data) {
    if (!std::ranges::equal(data, std::as_bytes(std::span(contents)))) {
      show_status("WebGPU upload or unaligned tail readback failed.");
      return;
    }

    const WebGPUBuffer target(
        canvas->device(), 16, wgpu::BufferUsage::Uniform, "Copy destination");
    target.clear(0x22222222);
    target.copy_from(source, 4, 8, 4);
    target.read_async([target](std::span<const std::byte> data) {
      const std::array<uint8_t, 16> expected = {
          34, 34, 34, 34, 34, 34, 34, 34, 4, 5, 6, 7, 34, 34, 34, 34};
      if (!std::ranges::equal(data, std::as_bytes(std::span(expected)))) {
        show_status("WebGPU fill or buffer copy failed.");
        return;
      }
      target.clear(0);
      target.read_async([](std::span<const std::byte> data) {
        const std::array<std::byte, 16> expected{};
        show_status(std::ranges::equal(data, expected) ? "WebGPU commands completed." :
                                                         "WebGPU buffer zero clear failed.");
      });
    });
  });
}

void draw()
{
  int width, height;
  if (emscripten_get_canvas_element_size(canvas_selector, &width, &height) !=
          EMSCRIPTEN_RESULT_SUCCESS ||
      width <= 0 || height <= 0) {
    show_status("Could not read the canvas dimensions.");
    return;
  }

  canvas->configure(width, height);
  const auto &device = canvas->device();
  const auto drawable = canvas->acquire_texture();
  if (!drawable) {
    show_status("Could not acquire a WebGPU canvas texture.");
    return;
  }

  device.PushErrorScope(wgpu::ErrorFilter::Validation);

  wgpu::ShaderSourceWGSL source;
  source.code = shader_source;
  wgpu::ShaderModuleDescriptor shader_descriptor;
  shader_descriptor.nextInChain = &source;
  const auto shader = device.CreateShaderModule(&shader_descriptor);

  constexpr uint64_t vertex_bytes = 4 * sizeof(float);
  constexpr uint32_t vertex_count = 3;
  const blender::gpu::WebGPUBuffer vertices(
      device, vertex_count * vertex_bytes, wgpu::BufferUsage::Storage | wgpu::BufferUsage::Vertex,
      "Triangle vertices");

  wgpu::ComputePipelineDescriptor compute_descriptor;
  compute_descriptor.compute.module = shader;
  compute_descriptor.compute.entryPoint = "generate";
  const auto compute_pipeline = device.CreateComputePipeline(&compute_descriptor);

  wgpu::BindGroupEntry binding{};
  binding.buffer = vertices.handle();
  binding.size = vertices.size();
  wgpu::BindGroupDescriptor group_descriptor;
  group_descriptor.layout = compute_pipeline.GetBindGroupLayout(0);
  group_descriptor.entryCount = 1;
  group_descriptor.entries = &binding;
  const auto group = device.CreateBindGroup(&group_descriptor);

  wgpu::VertexAttribute attribute{};
  attribute.format = wgpu::VertexFormat::Float32x4;
  wgpu::VertexBufferLayout vertex_layout;
  vertex_layout.arrayStride = vertex_bytes;
  vertex_layout.attributeCount = 1;
  vertex_layout.attributes = &attribute;

  wgpu::ColorTargetState target;
  target.format = canvas->format();
  wgpu::FragmentState fragment;
  fragment.module = shader;
  fragment.entryPoint = "fragment_main";
  fragment.targetCount = 1;
  fragment.targets = &target;

  wgpu::RenderPipelineDescriptor render_descriptor;
  render_descriptor.vertex.module = shader;
  render_descriptor.vertex.entryPoint = "vertex_main";
  render_descriptor.vertex.bufferCount = 1;
  render_descriptor.vertex.buffers = &vertex_layout;
  render_descriptor.fragment = &fragment;
  const auto render_pipeline = device.CreateRenderPipeline(&render_descriptor);

  const auto encoder = device.CreateCommandEncoder();
  const auto compute = encoder.BeginComputePass();
  compute.SetPipeline(compute_pipeline);
  compute.SetBindGroup(0, group);
  compute.DispatchWorkgroups(1);
  compute.End();

  wgpu::RenderPassColorAttachment attachment;
  attachment.view = drawable.CreateView();
  attachment.loadOp = wgpu::LoadOp::Clear;
  attachment.storeOp = wgpu::StoreOp::Store;
  attachment.clearValue = {0.04, 0.06, 0.09, 1.0};
  wgpu::RenderPassDescriptor pass_descriptor;
  pass_descriptor.colorAttachmentCount = 1;
  pass_descriptor.colorAttachments = &attachment;

  const auto render = encoder.BeginRenderPass(&pass_descriptor);
  render.SetPipeline(render_pipeline);
  render.SetVertexBuffer(0, vertices.handle());
  render.Draw(vertex_count);
  render.End();

  const auto commands = encoder.Finish();
  const auto queue = device.GetQueue();
  queue.Submit(1, &commands);
  device.PopErrorScope(
      wgpu::CallbackMode::AllowSpontaneous,
      [](wgpu::PopErrorScopeStatus status, wgpu::ErrorType error, wgpu::StringView) {
        if (status != wgpu::PopErrorScopeStatus::Success || error != wgpu::ErrorType::NoError) {
          show_status("WebGPU command validation failed.");
          return;
        }
        verify_buffer_transfers(canvas->device());
      });
}

void start_gpu()
{
  GHOST_WebGPUCanvas::request(
      canvas_selector, [](std::unique_ptr<GHOST_WebGPUCanvas> ready_canvas, const char *error) {
        if (error) {
          show_status(error);
          return;
        }
        canvas = std::move(ready_canvas);
        draw();
      });
}

#ifdef __EMSCRIPTEN_PTHREADS__
std::atomic<bool> worker_ready = false;

void start_gpu_when_worker_is_ready()
{
  if (worker_ready.load(std::memory_order_acquire)) {
    emscripten_cancel_main_loop();
    start_gpu();
  }
}
#endif

}  // namespace

int main()
{
#ifdef __EMSCRIPTEN_PTHREADS__
  std::thread([] { worker_ready.store(true, std::memory_order_release); }).detach();
  emscripten_set_main_loop(start_gpu_when_worker_is_ready, 0, 0);
#else
  start_gpu();
#endif
}
