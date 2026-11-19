/* SPDX-License-Identifier: Apache-2.0 */

#include "device/webgpu/device.h"

#include "device/cpu/device_impl.h"
#include "device/device.h"
#include "device/queue.h"
#include "device/webgpu/oidn.h"
#include "device/webgpu/runtime.h"

#include "webgpu_buffer.hh"

#include <emscripten/proxying.h>
#include <emscripten/threading.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

CCL_NAMESPACE_BEGIN

namespace {

using blender::gpu::WebGPUBuffer;

emscripten::ProxyingQueue &webgpu_owner_queue()
{
  static emscripten::ProxyingQueue queue;
  return queue;
}

DeviceInfo experimental_device_info()
{
  DeviceInfo info;
  info.type = DEVICE_NONE;
  info.id = "WEBGPU_EXPERIMENTAL";
  info.description = "WebGPU experimental compute device";
  info.has_gpu_queue = true;
  return info;
}

string webgpu_error(const char *message, wgpu::StringView detail)
{
  string result(message);
  if (detail.data && detail.length) {
    result += ": ";
    result += std::string_view(detail);
  }
  return result;
}

struct KernelPipeline {
  wgpu::ComputePipeline pipeline;
  wgpu::BindGroupLayout layout;
  std::vector<WebGPUKernelBinding> bindings;
  uint32_t workgroup_size;
};

/* WebGPU objects and the opaque handles never leave the owning worker. */
struct OwnerState {
  explicit OwnerState(const wgpu::Device &device) : device(device), queue(device.GetQueue()) {}

  const wgpu::Device device;
  const wgpu::Queue queue;
  std::map<device_ptr, unique_ptr<WebGPUBuffer>> buffers;
  std::map<DeviceKernel, KernelPipeline> kernels;
  wgpu::Buffer counter_readback;

  WebGPUBuffer *get_buffer(device_ptr handle)
  {
    const auto it = buffers.find(handle);
    return it == buffers.end() ? nullptr : it->second.get();
  }
};

class WebGPUDevice final : public CPUDevice {
 public:
  WebGPUDevice(const DeviceInfo &info, Stats &stats, Profiler &profiler, bool headless)
      : CPUDevice(info, stats, profiler, headless), host_scene_(info.type == DEVICE_WEBGPU)
  {
  }

  WebGPUDevice(const wgpu::Device &gpu,
               const wgpu::Limits &limits,
               Stats &stats,
               Profiler &profiler)
      : WebGPUDevice(experimental_device_info(), stats, profiler, true)
  {
    attach_owner(gpu, limits);
  }

  ~WebGPUDevice() override
  {
    if (!state_) {
      return;
    }
    if (on_owner_thread()) {
      delete state_;
    }
    else if (!webgpu_owner_queue().proxyAsync(owner_, [state = state_] { delete state; })) {
      /* A stopped runtime worker owns no live browser resources to release. */
      LOG_ERROR << "Cycles WebGPU owner stopped before device cleanup.";
    }
  }

  unique_ptr<DeviceQueue> gpu_queue_create() override;

  BVHLayoutMask get_bvh_layout_mask(uint64_t /*kernel_features*/) const override
  {
    return BVH_LAYOUT_BVH2;
  }

  bool prepare_gpu();
  bool register_kernel(DeviceKernel kernel,
                       std::string_view wgsl,
                       uint32_t workgroup_size,
                       std::span<const WebGPUKernelBinding> bindings,
                       std::string_view entry_point);

  bool enqueue(int work_size, std::span<const WebGPUDispatch> dispatches);
  bool synchronize();
  bool with_buffer(const device_memory &mem,
                   const std::function<void(const wgpu::Device &, const wgpu::Buffer &)> &operation);
  bool upload(const device_memory &mem, std::span<const std::byte> bytes);
  bool read(const device_memory &mem, size_t offset, size_t size, void *destination);
  bool read_queue_counters(int work_size,
                           const WebGPUDispatch &dispatch,
                           const device_memory &counters,
                           std::span<std::byte> bytes);

  size_t storage_limit() const
  {
    const size_t storage_bytes = compute_limits().storage_buffer_bytes;
    return storage_bytes ? storage_bytes : 32 * 1024 * 1024;
  }

  WebGPUComputeLimits compute_limits() const
  {
    std::lock_guard lock(owner_mutex_);
    if (!state_) {
      return {};
    }
    return {size_t(std::min(limits_.maxBufferSize, limits_.maxStorageBufferBindingSize)),
            limits_.maxComputeWorkgroupsPerDimension};
  }

  string gpu_error() const
  {
    std::lock_guard lock(gpu_error_mutex_);
    return gpu_error_;
  }

  void mem_copy_to(device_memory &mem) override;
  void mem_zero(device_memory &mem) override;

  bool gpu_memory(const device_memory &mem) const
  {
    /* Keep synchronized BVH2, image metadata and render results CPU-addressable.
     * The queue and translated kernels explicitly allocate device-only storage. */
    if (!host_scene_) {
      return true;
    }
    if (mem.type == MEM_GLOBAL || mem.type == MEM_IMAGE_TEXTURE) {
      return false;
    }
    return mem.type == MEM_DEVICE_ONLY || (mem.flags & MEM_FLAG_NO_HOST_FALLBACK);
  }

 private:
  using OwnerTask = std::function<void(OwnerState &, string &)>;
  using Finish = std::function<void(string)>;
  using AsyncOwnerTask = std::function<void(OwnerState &, Finish)>;

  bool on_owner_thread() const
  {
    return pthread_equal(owner_, pthread_self()) != 0;
  }

  void attach_owner(const wgpu::Device &gpu, const wgpu::Limits &limits)
  {
    limits_ = limits;
    owner_ = pthread_self();
    state_ = new OwnerState(gpu);
  }

  void set_gpu_error(const string &error)
  {
    {
      std::lock_guard lock(gpu_error_mutex_);
      if (gpu_error_.empty()) {
        gpu_error_ = error;
      }
    }
    if (host_scene_) {
      LOG_ERROR << error;
    }
    else {
      set_error(error);
    }
  }

  bool gpu_failed() const
  {
    std::lock_guard lock(gpu_error_mutex_);
    return !gpu_error_.empty();
  }

  string record_dispatch(OwnerState &state,
                         const wgpu::ComputePassEncoder &pass,
                         int work_size,
                         const WebGPUDispatch &dispatch) const;

  bool with_owner(const OwnerTask &operation);
  bool await_owner(const AsyncOwnerTask &operation);

  void mem_alloc(device_memory &mem) override;
  void mem_free(device_memory &mem) override;
  void mem_move_to_host(device_memory &mem) override;
  void mem_copy_from(
      device_memory &mem, size_t y, size_t width, size_t height, size_t pixel_size) override;

  const bool host_scene_;
  mutable std::mutex owner_mutex_;
  wgpu::Limits limits_{};
  pthread_t owner_{};
  OwnerState *state_ = nullptr;
  mutable std::mutex gpu_error_mutex_;
  string gpu_error_;
};

bool WebGPUDevice::prepare_gpu()
{
  std::lock_guard lock(owner_mutex_);
  if (state_) {
    return true;
  }
  if (gpu_failed()) {
    return false;
  }
  cycles_webgpu_on_compute_device([&](const wgpu::Device &gpu) {
    wgpu::Limits limits{};
    if (gpu.GetLimits(&limits) == wgpu::Status::Success) {
      attach_owner(gpu, limits);
    }
  });
  if (!state_) {
    set_gpu_error("Cycles could not initialize the WebGPU compute device");
  }
  return state_ != nullptr;
}

bool WebGPUDevice::with_owner(const OwnerTask &operation)
{
  if (!prepare_gpu()) {
    return false;
  }
  string error;
  const auto task = [&] { operation(*state_, error); };
  if (on_owner_thread()) {
    task();
  }
  else if (!webgpu_owner_queue().proxySync(owner_, task)) {
    error = "Cycles WebGPU could not reach the owning worker";
  }
  if (!error.empty()) {
    set_gpu_error(error);
  }
  return error.empty();
}

bool WebGPUDevice::await_owner(const AsyncOwnerTask &operation)
{
  if (!prepare_gpu()) {
    return false;
  }
  if (on_owner_thread()) {
    set_gpu_error("Cycles WebGPU completion needs a Cycles pthread");
    return false;
  }

  string error;
  if (!webgpu_owner_queue().proxySyncWithCtx(owner_,
                                             [&](auto context) {
                                               operation(*state_,
                                                         [&error, context](string result) mutable {
                                                           error = std::move(result);
                                                           context.finish();
                                                         });
                                             }))
  {
    error = "Cycles WebGPU could not reach the owning worker";
  }
  if (!error.empty()) {
    set_gpu_error(error);
  }
  return error.empty();
}

bool WebGPUDevice::register_kernel(DeviceKernel kernel,
                                   std::string_view code,
                                   uint32_t workgroup_size,
                                   std::span<const WebGPUKernelBinding> bindings,
                                   std::string_view entry_point)
{
  if (gpu_failed() || !prepare_gpu()) {
    return false;
  }
  if (kernel < 0 || kernel >= DEVICE_KERNEL_NUM || code.empty() || entry_point.empty() ||
      workgroup_size == 0 || workgroup_size > limits_.maxComputeWorkgroupSizeX ||
      workgroup_size > limits_.maxComputeInvocationsPerWorkgroup || bindings.empty() ||
      bindings.size() > DeviceKernelArguments::MAX_ARGS ||
      bindings.size() > limits_.maxBindingsPerBindGroup)
  {
    set_gpu_error("Invalid Cycles WebGPU kernel limits");
    return false;
  }

  std::vector<wgpu::BindGroupLayoutEntry> entries;
  entries.reserve(bindings.size());
  uint32_t uniforms = 0;
  for (const auto &binding : bindings) {
    const bool uniform = binding.type == WebGPUKernelBinding::Type::Uniform;
    uniforms += uniform;
    const size_t limit = uniform ? limits_.maxUniformBufferBindingSize : storage_limit();
    if (binding.binding >= limits_.maxBindingsPerBindGroup || binding.minimum_size == 0 ||
        binding.minimum_size > limit ||
        std::ranges::any_of(entries,
                            [&](const auto &entry) { return entry.binding == binding.binding; }))
    {
      set_gpu_error("Invalid Cycles WebGPU binding number or buffer size");
      return false;
    }
    auto &entry = entries.emplace_back();
    entry.binding = binding.binding;
    entry.visibility = wgpu::ShaderStage::Compute;
    if (uniform) {
      entry.buffer.type = wgpu::BufferBindingType::Uniform;
    }
    else if (binding.type == WebGPUKernelBinding::Type::ReadOnlyStorage) {
      entry.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
    }
    else {
      entry.buffer.type = wgpu::BufferBindingType::Storage;
    }
    entry.buffer.minBindingSize = binding.minimum_size;
  }
  if (uniforms > limits_.maxUniformBuffersPerShaderStage ||
      bindings.size() - uniforms > limits_.maxStorageBuffersPerShaderStage)
  {
    set_gpu_error("Cycles WebGPU kernel exceeds the buffer binding limits");
    return false;
  }

  return await_owner([&](OwnerState &state, Finish finish) {
    if (state.kernels.contains(kernel)) {
      finish("Cycles WebGPU kernel is already registered");
      return;
    }

    wgpu::BindGroupLayoutDescriptor binding_layout{};
    binding_layout.entryCount = entries.size();
    binding_layout.entries = entries.data();
    auto group = state.device.CreateBindGroupLayout(&binding_layout);
    wgpu::PipelineLayoutDescriptor pipeline_layout{};
    pipeline_layout.bindGroupLayoutCount = 1;
    pipeline_layout.bindGroupLayouts = &group;

    wgpu::ShaderSourceWGSL source{};
    source.code = wgpu::StringView(code);
    wgpu::ShaderModuleDescriptor module{};
    module.nextInChain = &source;
    wgpu::ComputePipelineDescriptor descriptor{};
    descriptor.label = device_kernel_as_string(kernel);
    descriptor.layout = state.device.CreatePipelineLayout(&pipeline_layout);
    descriptor.compute.module = state.device.CreateShaderModule(&module);
    descriptor.compute.entryPoint = wgpu::StringView(entry_point);
    state.device.CreateComputePipelineAsync(
        &descriptor,
        wgpu::CallbackMode::AllowSpontaneous,
        [&state, kernel, workgroup_size, bindings, group = std::move(group), finish](
            wgpu::CreatePipelineAsyncStatus status,
            wgpu::ComputePipeline pipeline,
            wgpu::StringView detail) mutable {
          if (status != wgpu::CreatePipelineAsyncStatus::Success) {
            finish(webgpu_error("Cannot compile Cycles WebGPU compute pipeline", detail));
            return;
          }
          state.kernels.emplace(kernel,
                                KernelPipeline{std::move(pipeline),
                                               std::move(group),
                                               {bindings.begin(), bindings.end()},
                                               workgroup_size});
          finish({});
        });
  });
}

string WebGPUDevice::record_dispatch(OwnerState &state,
                                    const wgpu::ComputePassEncoder &pass,
                                    const int work_size,
                                    const WebGPUDispatch &dispatch) const
{
  const auto found = state.kernels.find(dispatch.kernel);
  if (found == state.kernels.end()) {
    return "Cycles WebGPU kernel has no registered pipeline";
  }
  const auto &args = dispatch.args;
  const KernelPipeline &kernel_pipeline = found->second;
  const uint32_t items = dispatch.work_size ? dispatch.work_size : uint32_t(work_size);
  const auto workgroups = (uint64_t(items) + kernel_pipeline.workgroup_size - 1) /
                          kernel_pipeline.workgroup_size;
  const WebGPUBuffer *indirect = dispatch.indirect ? state.get_buffer(dispatch.indirect) : nullptr;
  if (dispatch.indirect && (!indirect || indirect->size() < 3 * sizeof(uint32_t))) {
    return "Cycles WebGPU dispatch has an invalid indirect buffer";
  }
  if ((!indirect && workgroups > limits_.maxComputeWorkgroupsPerDimension) ||
      args.count != kernel_pipeline.bindings.size())
  {
    return "Cycles WebGPU dispatch exceeds the pipeline limits";
  }

  std::array<wgpu::BindGroupEntry, DeviceKernelArguments::MAX_ARGS> entries;
  for (size_t index = 0; index < args.count; ++index) {
    if (args.types[index] != DeviceKernelArguments::POINTER ||
        args.sizes[index] != sizeof(device_ptr) || !args.values[index])
    {
      return "Cycles WebGPU storage binding needs a buffer argument";
    }
    const auto handle = *static_cast<const device_ptr *>(args.values[index]);
    const WebGPUBuffer *buffer = state.get_buffer(handle);
    const auto &binding = kernel_pipeline.bindings[index];
    if (!buffer || buffer->size() < binding.minimum_size ||
        (binding.type == WebGPUKernelBinding::Type::Uniform &&
         buffer->allocation_size() > limits_.maxUniformBufferBindingSize))
    {
      return "Cycles WebGPU dispatch has an invalid storage-buffer handle or size";
    }
    auto &entry = entries[index];
    entry.binding = binding.binding;
    entry.buffer = buffer->handle();
    entry.size = buffer->allocation_size();
  }

  wgpu::BindGroupDescriptor descriptor{};
  descriptor.layout = kernel_pipeline.layout;
  descriptor.entries = entries.data();
  descriptor.entryCount = args.count;
  const auto group = state.device.CreateBindGroup(&descriptor);
  pass.SetPipeline(kernel_pipeline.pipeline);
  pass.SetBindGroup(0, group);
  if (indirect) {
    pass.DispatchWorkgroupsIndirect(indirect->handle(), 0);
  }
  else {
    pass.DispatchWorkgroups(uint32_t(workgroups));
  }
  return {};
}

bool WebGPUDevice::enqueue(const int work_size, std::span<const WebGPUDispatch> dispatches)
{
  if (gpu_failed()) {
    return false;
  }
  if (work_size < 0) {
    set_gpu_error("Negative Cycles WebGPU dispatch size");
    return false;
  }
  if (work_size == 0 || dispatches.empty()) {
    return true;
  }

  return await_owner([&](OwnerState &state, Finish finish) {
    state.device.PushErrorScope(wgpu::ErrorFilter::Validation);
    const auto encoder = state.device.CreateCommandEncoder();
    const auto pass = encoder.BeginComputePass();
    string error;
    for (const WebGPUDispatch &dispatch : dispatches) {
      error = record_dispatch(state, pass, work_size, dispatch);
      if (!error.empty()) {
        break;
      }
    }
    pass.End();
    if (error.empty()) {
      const auto commands = encoder.Finish();
      state.queue.Submit(1, &commands);
    }
    state.device.PopErrorScope(
        wgpu::CallbackMode::AllowSpontaneous,
        [finish, error = std::move(error)](
            wgpu::PopErrorScopeStatus status, wgpu::ErrorType type, wgpu::StringView detail) {
          if (!error.empty()) {
            finish(error);
          }
          else if (status != wgpu::PopErrorScopeStatus::Success || type != wgpu::ErrorType::NoError) {
            finish(webgpu_error("Invalid Cycles WebGPU dispatch", detail));
          }
          else {
            finish({});
          }
        });
  });
}

bool WebGPUDevice::synchronize()
{
  if (gpu_failed()) {
    return false;
  }
  return await_owner([](OwnerState &state, Finish finish) {
    state.queue.OnSubmittedWorkDone(
        wgpu::CallbackMode::AllowSpontaneous,
        [finish](wgpu::QueueWorkDoneStatus status, wgpu::StringView detail) {
          finish(status == wgpu::QueueWorkDoneStatus::Success ?
                     string() :
                     webgpu_error("Cycles WebGPU queue failed", detail));
        });
  });
}

bool WebGPUDevice::read_queue_counters(const int work_size,
                                       const WebGPUDispatch &dispatch,
                                       const device_memory &counters,
                                       const std::span<std::byte> bytes)
{
  if (gpu_failed()) {
    return false;
  }
  if (work_size <= 0 || counters.device != this || !gpu_memory(counters) ||
      !counters.device_pointer || bytes.empty() || bytes.size() % sizeof(uint32_t) != 0 ||
      bytes.size() != counters.device_size)
  {
    set_gpu_error("Invalid Cycles WebGPU queue counter readback");
    return false;
  }

  /* Both callbacks run on the owning worker. The render pthread waits until both
   * have finished, keeping this state and the destination span alive. */
  int pending = 1;
  string dispatch_error;
  string map_error;
  return await_owner([&](OwnerState &state, Finish finish) {
    const auto *buffer = state.get_buffer(counters.device_pointer);
    if (!buffer || buffer->size() != bytes.size()) {
      finish("Invalid Cycles WebGPU queue counter buffer");
      return;
    }

    state.device.PushErrorScope(wgpu::ErrorFilter::Validation);
    /* Lease the buffer so another caller cannot reuse it while its map is pending. */
    auto staging = std::exchange(state.counter_readback, wgpu::Buffer{});
    if (!staging || staging.GetSize() != bytes.size()) {
      wgpu::BufferDescriptor desc;
      desc.label = "Cycles WebGPU queue counter readback";
      desc.size = bytes.size();
      desc.usage = wgpu::BufferUsage::MapRead | wgpu::BufferUsage::CopyDst;
      staging = state.device.CreateBuffer(&desc);
    }

    if (!staging) {
      dispatch_error = "Could not allocate Cycles WebGPU queue counter staging";
    }
    else {
      const auto encoder = state.device.CreateCommandEncoder();
      encoder.ClearBuffer(buffer->handle(), 0, bytes.size());
      const auto pass = encoder.BeginComputePass();
      dispatch_error = record_dispatch(state, pass, work_size, dispatch);
      pass.End();
      if (dispatch_error.empty()) {
        encoder.CopyBufferToBuffer(buffer->handle(), 0, staging, 0, bytes.size());
        const auto commands = encoder.Finish();
        state.queue.Submit(1, &commands);
        pending = 2;
      }
    }

    const bool map_submitted = pending == 2;
    const auto complete = [&, finish, staging]() {
      if (--pending != 0) {
        return;
      }
      /* The map callback has already unmapped the buffer. Leave the cache alone
       * if another completed read returned a buffer first. */
      if (dispatch_error.empty() && map_error.empty() && !state.counter_readback) {
        state.counter_readback = staging;
      }
      finish(dispatch_error.empty() ? std::move(map_error) : std::move(dispatch_error));
    };
    state.device.PopErrorScope(
        wgpu::CallbackMode::AllowSpontaneous,
        [&, complete](wgpu::PopErrorScopeStatus status, wgpu::ErrorType type, wgpu::StringView detail) {
          if (dispatch_error.empty() &&
              (status != wgpu::PopErrorScopeStatus::Success || type != wgpu::ErrorType::NoError))
          {
            dispatch_error = webgpu_error("Invalid Cycles WebGPU queue counter dispatch", detail);
          }
          complete();
        });
    if (map_submitted) {
      staging.MapAsync(
          wgpu::MapMode::Read,
          0,
          bytes.size(),
          wgpu::CallbackMode::AllowSpontaneous,
          [&map_error, bytes, staging, complete](wgpu::MapAsyncStatus status,
                                                 wgpu::StringView detail) {
            if (status != wgpu::MapAsyncStatus::Success) {
              map_error = webgpu_error("Cycles WebGPU queue counter mapping failed", detail);
            }
            else {
              const auto *data = staging.GetConstMappedRange(0, bytes.size());
              if (data) {
                std::memcpy(bytes.data(), data, bytes.size());
              }
              else {
                map_error = "Cycles WebGPU queue counter mapping returned no data";
              }
              staging.Unmap();
            }
            complete();
          });
    }
  });
}

void WebGPUDevice::mem_alloc(device_memory &mem)
{
  if (!gpu_memory(mem)) {
    CPUDevice::mem_alloc(mem);
    return;
  }
  if (gpu_failed() || mem.memory_size() == 0 || mem.device_pointer || !prepare_gpu()) {
    return;
  }
  const size_t size = mem.memory_size();
  if (mem.device != this || mem.type == MEM_GLOBAL || mem.type == MEM_IMAGE_TEXTURE) {
    set_gpu_error("Cycles WebGPU allocation needs device-local storage memory");
    return;
  }

  /* Binding and transfer sizes round to four bytes. */
  if (size > (storage_limit() & ~size_t(3))) {
    set_gpu_error("Cycles WebGPU allocation exceeds the storage-buffer limit");
    return;
  }

  device_ptr handle = 0;
  size_t allocation_size = 0;
  if (with_owner([&](OwnerState &state, string &error) {
        static device_ptr next_handle = 1;
        if (!next_handle) {
          error = "Cycles WebGPU buffer handle space exhausted";
          return;
        }
        auto buffer = make_unique<WebGPUBuffer>(state.device,
                                                size,
                                                wgpu::BufferUsage::Storage |
                                                    wgpu::BufferUsage::Uniform |
                                                    wgpu::BufferUsage::Indirect,
                                                mem.log_name().c_str());
        if (!buffer->handle()) {
          error = "Cycles WebGPU storage allocation failed";
          return;
        }
        allocation_size = buffer->allocation_size();
        handle = next_handle++;
        state.buffers.emplace(handle, std::move(buffer));
      }))
  {
    mem.device_pointer = handle;
    mem.device_size = size;
    stats.mem_alloc(allocation_size);
  }
}

void WebGPUDevice::mem_free(device_memory &mem)
{
  if (!gpu_memory(mem)) {
    CPUDevice::mem_free(mem);
    return;
  }
  if (!mem.device_pointer) {
    return;
  }
  size_t allocation_size = 0;
  if (with_owner([&](OwnerState &state, string &error) {
        const auto it = state.buffers.find(mem.device_pointer);
        if (it == state.buffers.end()) {
          error = "Invalid Cycles WebGPU buffer handle";
          return;
        }
        allocation_size = it->second->allocation_size();
        state.buffers.erase(it);
      }))
  {
    mem.device_pointer = 0;
    mem.device_size = 0;
    stats.mem_free(allocation_size);
  }
}

void WebGPUDevice::mem_copy_to(device_memory &mem)
{
  if (!gpu_memory(mem)) {
    CPUDevice::mem_copy_to(mem);
    return;
  }
  const size_t size = mem.memory_size();
  if (gpu_failed() || size == 0) {
    return;
  }
  if (mem.device != this || !mem.host_pointer) {
    set_gpu_error("Cycles WebGPU upload needs host storage");
    return;
  }
  if (mem.device_pointer && mem.device_size != size) {
    mem_free(mem);
  }
  mem_alloc(mem);
  if (!mem.device_pointer) {
    return;
  }
  upload(mem, {static_cast<const std::byte *>(mem.host_pointer), size});
}

bool WebGPUDevice::upload(const device_memory &mem, const std::span<const std::byte> bytes)
{
  if (gpu_failed()) {
    return false;
  }
  if (mem.device != this || !gpu_memory(mem) || !mem.device_pointer ||
      bytes.size() != mem.device_size)
  {
    set_gpu_error("Cycles WebGPU upload must fill an allocated GPU buffer");
    return false;
  }
  return with_owner([&](OwnerState &state, string &error) {
    if (const auto *buffer = state.get_buffer(mem.device_pointer)) {
      buffer->write(bytes.data());
    }
    else {
      error = "Invalid Cycles WebGPU upload buffer";
    }
  });
}

bool WebGPUDevice::with_buffer(
    const device_memory &mem,
    const std::function<void(const wgpu::Device &, const wgpu::Buffer &)> &operation)
{
  if (gpu_failed() || mem.device != this || !mem.device_pointer) {
    return false;
  }
  bool invoked = false;
  return with_owner([&](OwnerState &state, string & /*error*/) {
           if (const auto *buffer = state.get_buffer(mem.device_pointer)) {
             operation(state.device, buffer->handle());
             invoked = true;
           }
         }) &&
         invoked;
}

void WebGPUDevice::mem_move_to_host(device_memory &mem)
{
  if (gpu_memory(mem)) {
    set_gpu_error("Cycles WebGPU allocation cannot become host-resident");
  }
}

void WebGPUDevice::mem_zero(device_memory &mem)
{
  if (!gpu_memory(mem)) {
    CPUDevice::mem_zero(mem);
    return;
  }
  if (gpu_failed() || mem.memory_size() == 0) {
    return;
  }
  if (mem.device != this) {
    set_gpu_error("Invalid Cycles WebGPU clear buffer");
    return;
  }
  mem_alloc(mem);
  if (!mem.device_pointer) {
    return;
  }
  with_owner([&](OwnerState &state, string &error) {
    if (const auto *buffer = state.get_buffer(mem.device_pointer)) {
      buffer->clear(0);
    }
    else {
      error = "Invalid Cycles WebGPU clear buffer";
    }
  });
}

bool WebGPUDevice::read(const device_memory &mem, size_t offset, size_t size, void *destination)
{
  if (gpu_failed() || size == 0) {
    return !gpu_failed();
  }
  if (!destination || mem.device != this || !gpu_memory(mem) || !mem.device_pointer ||
      offset > mem.device_size || size > mem.device_size - offset)
  {
    set_gpu_error("Invalid Cycles WebGPU download range");
    return false;
  }

  return await_owner([&](OwnerState &state, Finish finish) {
    const auto *buffer = state.get_buffer(mem.device_pointer);
    if (!buffer) {
      finish("Invalid Cycles WebGPU download buffer");
      return;
    }
    buffer->read_async(offset, size, [destination, size, finish](std::span<const std::byte> data) {
      if (data.size() != size) {
        finish("Cycles WebGPU buffer mapping failed");
        return;
      }
      std::memcpy(destination, data.data(), size);
      finish({});
    });
  });
}

void WebGPUDevice::mem_copy_from(
    device_memory &mem, size_t y, size_t width, size_t height, size_t pixel_size)
{
  const size_t limit = std::numeric_limits<size_t>::max();
  if ((width && pixel_size > limit / width)) {
    set_gpu_error("Cycles WebGPU row size overflow");
    return;
  }
  if (!gpu_memory(mem)) {
    return;
  }
  const size_t row = width * pixel_size;
  if ((row && (y > limit / row || height > limit / row)) || y * row > mem.memory_size() ||
      height * row > mem.memory_size() - y * row || (!mem.host_pointer && height && row))
  {
    set_gpu_error("Invalid Cycles WebGPU host download range");
    return;
  }
  if (height && row) {
    const size_t offset = y * row;
    read(mem, offset, height * row, static_cast<char *>(mem.host_pointer) + offset);
  }
}

class WebGPUDeviceQueue final : public DeviceQueue {
 public:
  explicit WebGPUDeviceQueue(WebGPUDevice *device) : DeviceQueue(device), webgpu_(*device) {}

  void init_execution() override
  {
    debug_init_execution();
  }
  void load_image_info() override
  {
    /* The WebGPU kernels bind the packed image data explicitly. */
  }
  bool enqueue(DeviceKernel kernel, int work_size, const DeviceKernelArguments &args) override
  {
    debug_enqueue_begin(kernel, work_size);
    const WebGPUDispatch dispatch{kernel, args};
    const bool success = webgpu_.enqueue(work_size, {&dispatch, 1});
    debug_enqueue_end();
    return success;
  }
  bool synchronize() override
  {
    const bool success = webgpu_.synchronize();
    debug_synchronize();
    return success;
  }
  void zero_to_device(device_memory &mem) override
  {
    webgpu_.mem_zero(mem);
  }
  void copy_to_device(device_memory &mem) override
  {
    webgpu_.mem_copy_to(mem);
  }
  void copy_from_device(device_memory &mem) override
  {
    if (webgpu_.gpu_memory(mem)) {
      webgpu_.read(mem, 0, mem.memory_size(), mem.host_pointer);
    }
  }
  void *copy_from_device_synchronized(device_memory &mem, vector<uint8_t> &storage) override
  {
    if (!webgpu_.gpu_memory(mem)) {
      return mem.host_pointer;
    }
    storage.resize(mem.memory_size());
    return webgpu_.read(mem, 0, storage.size(), storage.data()) ? storage.data() : nullptr;
  }

 protected:
  ConcurrentStatesParams concurrent_states_params() const override
  {
    /* WebGPU exposes buffer limits, not free device memory. Use the tested path count as
     * the stock default. The render pipeline applies resource limits after environment
     * overrides, when the actual path storage layout is known. */
    constexpr int default_concurrent_paths = 64 * 1024;
    ConcurrentStatesParams params;
    params.min = default_concurrent_paths;
    params.baseline = default_concurrent_paths;
    params.max = default_concurrent_paths;
    return params;
  }
  void get_memory_info(size_t &total, size_t &free) const override
  {
    /* Use the per-binding limit as the scheduling budget. WebGPU does not expose device memory. */
    total = free = webgpu_.storage_limit();
  }

 private:
  WebGPUDevice &webgpu_;
};

unique_ptr<DeviceQueue> WebGPUDevice::gpu_queue_create()
{
  return make_unique<WebGPUDeviceQueue>(this);
}

}  // namespace

DeviceInfo device_webgpu_info()
{
  DeviceInfo info;
  info.type = DEVICE_WEBGPU;
  info.id = "WEBGPU";
  info.description = "WebGPU";
  info.display_device = true;
  info.has_mnee_ = false;
  info.has_gpu_queue = true;
#ifdef WITH_OPENIMAGEDENOISE
  if (cycles_webgpu_oidn_available()) {
    info.denoisers = DENOISER_OPENIMAGEDENOISE;
  }
#endif
  return info;
}

unique_ptr<Device> device_webgpu_create(const DeviceInfo &info,
                                       Stats &stats,
                                       Profiler &profiler,
                                       bool headless)
{
  return make_unique<WebGPUDevice>(info, stats, profiler, headless);
}

std::string device_webgpu_error(const Device &device)
{
  const auto *webgpu = dynamic_cast<const WebGPUDevice *>(&device);
  return webgpu ? webgpu->gpu_error() : "Expected a Cycles WebGPU device";
}

WebGPUComputeLimits device_webgpu_compute_limits(const Device &device)
{
  const auto *webgpu = dynamic_cast<const WebGPUDevice *>(&device);
  return webgpu ? webgpu->compute_limits() : WebGPUComputeLimits{};
}

WebGPUComputeLimits device_webgpu_prepare_compute_limits(Device &device)
{
  auto *webgpu = dynamic_cast<WebGPUDevice *>(&device);
  return webgpu && webgpu->prepare_gpu() ? webgpu->compute_limits() : WebGPUComputeLimits{};
}

bool device_webgpu_upload(Device &device,
                          const device_memory &buffer,
                          const std::span<const std::byte> bytes)
{
  auto *webgpu = dynamic_cast<WebGPUDevice *>(&device);
  return webgpu && webgpu->upload(buffer, bytes);
}

bool device_webgpu_read(Device &device,
                        const device_memory &buffer,
                        const std::span<std::byte> bytes)
{
  auto *webgpu = dynamic_cast<WebGPUDevice *>(&device);
  return webgpu && bytes.size() == buffer.device_size &&
         webgpu->read(buffer, 0, bytes.size(), bytes.data());
}

bool device_webgpu_with_buffer(
    Device &device,
    const device_memory &buffer,
    const std::function<void(const wgpu::Device &, const wgpu::Buffer &)> &operation)
{
  auto *webgpu = dynamic_cast<WebGPUDevice *>(&device);
  return webgpu && webgpu->with_buffer(buffer, operation);
}

unique_ptr<Device> device_webgpu_create_experimental(const wgpu::Device &device,
                                                     Stats &stats,
                                                     Profiler &profiler)
{
  wgpu::Limits limits{};
  if (emscripten_is_main_browser_thread() || !device ||
      device.GetLimits(&limits) != wgpu::Status::Success)
  {
    return nullptr;
  }
  return make_unique<WebGPUDevice>(device, limits, stats, profiler);
}

bool device_webgpu_register_kernel(Device &device,
                                   DeviceKernel kernel,
                                   std::string_view wgsl,
                                   uint32_t workgroup_size,
                                   std::span<const WebGPUKernelBinding> bindings,
                                   std::string_view entry_point)
{
  auto *webgpu = dynamic_cast<WebGPUDevice *>(&device);
  return webgpu && webgpu->register_kernel(kernel, wgsl, workgroup_size, bindings, entry_point);
}

bool device_webgpu_enqueue(Device &device,
                           const int work_size,
                           std::span<const WebGPUDispatch> dispatches)
{
  auto *webgpu = dynamic_cast<WebGPUDevice *>(&device);
  return webgpu && webgpu->enqueue(work_size, dispatches);
}

bool device_webgpu_read_queue_counters(Device &device,
                                      const int work_size,
                                      const WebGPUDispatch &dispatch,
                                      const device_memory &counters,
                                      const std::span<std::byte> bytes)
{
  auto *webgpu = dynamic_cast<WebGPUDevice *>(&device);
  return webgpu && webgpu->read_queue_counters(work_size, dispatch, counters, bytes);
}

CCL_NAMESPACE_END
