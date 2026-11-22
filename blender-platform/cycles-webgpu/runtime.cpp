/* SPDX-License-Identifier: Apache-2.0 */

#include "device/webgpu/runtime.h"

#include "device/device.h"
#include "device/webgpu/device.h"
#include "device/webgpu/oidn.h"

#include <emscripten.h>
#include <emscripten/proxying.h>
#include <emscripten/threading.h>
#include <webgpu/webgpu_cpp.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <mutex>
#include <pthread.h>
#include <unistd.h>
#include <utility>

CCL_NAMESPACE_BEGIN
namespace {

EM_JS_DEPS(cycles_oidn_runtime, "$WebGPU,$UTF8ToString");
EM_JS_DEPS(cycles_webgpu_preview_transport, "$PThread");
EM_JS(void, cycles_webgpu_attach_preview_transport, (pthread_t thread), {
  PThread.pthreads[thread]?.addEventListener("message", ({data}) => {
    if (data?.type === "cycles-webgpu-preview") {
      if (Module.ghostBrowserWindow) {
        Module.ghostBrowserWindow.submitCyclesPreview(data.ticket, data.bitmap);
      } else {
        data.bitmap.close();
      }
    }
  });
});
EM_JS(void,
      cycles_webgpu_launch_oidn,
      (WGPUDevice device,
       const char *model,
       int width,
       int height,
       int hdr,
       int srgb,
       float *color,
       size_t color_pixel_stride,
       size_t color_row_stride,
       float *albedo,
       size_t albedo_pixel_stride,
       size_t albedo_row_stride,
       float *normal,
       size_t normal_pixel_stride,
       size_t normal_row_stride,
       float *output,
       size_t output_pixel_stride,
       size_t output_row_stride,
       void *completion),
      {
        function image(pointer, pixelStride, rowStride) {
          const offset = Number(pointer);
          if (offset === 0) {
            return undefined;
          }
          pixelStride = Number(pixelStride);
          rowStride = Number(rowStride);
          const length = (height - 1) * rowStride + (width - 1) * pixelStride + 3;
          const data = new Float32Array(wasmMemory.buffer, offset, length);
          return { data, pixelStride, rowStride };
        }

        function finish(status) {
          const words = new Int32Array(wasmMemory.buffer);
          const word = Number(completion) / 4;
          Atomics.store(words, word, status);
          Atomics.notify(words, word);
        }

        async function filter() {
          const gpuDevice = WebGPU.getJsObject(device);
          const module = await import(new URL("oidn-webgpu-launcher.mjs", self.location.href));
          const request = {
            width,
            height,
            model: UTF8ToString(Number(model)),
            hdr: hdr !== 0,
            srgb: srgb !== 0,
            color: image(color, color_pixel_stride, color_row_stride),
            albedo: image(albedo, albedo_pixel_stride, albedo_row_stride),
            normal: image(normal, normal_pixel_stride, normal_row_stride),
            output: image(output, output_pixel_stride, output_row_stride),
          };
          await module.blenderOIDNWebGPU(gpuDevice, request);
        }

        filter().then(
          () => finish(1),
          (error) => {
            console.error("Cycles OIDN WebGPU", error);
            finish(-1);
          }
        );
      });

EM_JS(void,
      cycles_webgpu_launch_resident_film,
      (WGPUDevice device,
       WGPUBuffer source,
       const void *layout,
       const char *albedo,
       const char *normal,
       const char *color,
       float *pixels,
       double *times,
       void *completion),
      {
        function finish(status) {
          const words = new Int32Array(wasmMemory.buffer);
          const word = Number(completion) / 4;
          Atomics.store(words, word, status);
          Atomics.notify(words, word);
        }

        async function run() {
          const gpuDevice = WebGPU.getJsObject(device);
          const module = await import(new URL("oidn-webgpu-film.mjs", self.location.href));
          if (source) {
            await module.composeFilmTile(
              gpuDevice,
              WebGPU.getJsObject(source),
              new Uint32Array(wasmMemory.buffer, Number(layout), 11)
            );
          } else {
            const models = [albedo, normal, color].map(
              (value) => value ? UTF8ToString(Number(value)) : undefined
            );
            await module.finishFilm(
              gpuDevice,
              new Uint32Array(wasmMemory.buffer, Number(layout), 7),
              models,
              (words, offset) => {
                new Uint32Array(wasmMemory.buffer, Number(pixels) + offset * 4, words.length)
                  .set(words);
              },
              (elapsed) => {
                new Float64Array(wasmMemory.buffer, Number(times), 3).set(elapsed);
              }
            );
          }
        }

        run().then(
          () => finish(1),
          (error) => {
            console.error("Cycles WebGPU resident film", error);
            finish(-1);
          }
        );
      });

EM_JS(void, cycles_webgpu_discard_resident_film, (WGPUDevice device, uint32_t id), {
  const gpuDevice = WebGPU.getJsObject(device);
  import(new URL("oidn-webgpu-film.mjs", self.location.href)).then(
    (module) => module.dropFilm(gpuDevice, id),
    (error) => console.error("Cycles WebGPU resident film", error)
  );
});

EM_JS(void,
      cycles_webgpu_launch_preview,
      (WGPUDevice device, WGPUBuffer source, const void *layout, void *completion),
      {
        const gpuDevice = WebGPU.getJsObject(device);
        const gpuFilm = WebGPU.getJsObject(source);
        const parameters = new Uint32Array(wasmMemory.buffer, Number(layout), 10).slice();
        function finish(status) {
          const words = new Int32Array(wasmMemory.buffer);
          const word = Number(completion) / 4;
          Atomics.store(words, word, status);
          Atomics.notify(words, word);
        }

        import(new URL("cycles-webgpu-render-preview.mjs", self.location.href)).then(
          (module) => module.publishRenderPreview(gpuDevice, gpuFilm, parameters)
        ).then(
          () => finish(1),
          (error) => {
            console.error("Cycles WebGPU preview", error);
            finish(-1);
          }
        );
      });

bool wait_for_compute_worker(std::atomic<int32_t> &completion)
{
  /* The compute owner must remain free to service promises and GPU callbacks. */
  while (completion.load(std::memory_order_acquire) == 0) {
    emscripten_futex_wait(&completion, 0, std::numeric_limits<double>::infinity());
  }
  return completion.load(std::memory_order_acquire) == 1;
}

struct ComputeWorker {
  enum class Status { Idle, Starting, Ready, Lost, Failed };

  std::mutex mutex;
  std::condition_variable ready;
  Status status = Status::Idle;
  uint64_t generation = 0;
  pthread_t thread{};
  emscripten::ProxyingQueue queue;

  /* Only the compute worker touches these handles. */
  wgpu::Instance instance;
  wgpu::Device device;

  void finish(const uint64_t request, wgpu::Device acquired = {})
  {
    {
      std::lock_guard lock(mutex);
      if (request != generation || status != Status::Starting) {
        return;
      }
      if (acquired) {
        std::swap(device, acquired);
        status = Status::Ready;
      }
      else {
        status = device ? Status::Lost : Status::Failed;
      }
    }
    ready.notify_all();
  }

  void device_lost(const uint64_t request)
  {
    {
      std::lock_guard lock(mutex);
      if (request != generation || (status != Status::Starting && status != Status::Ready)) {
        return;
      }
      status = Status::Lost;
    }
    ready.notify_all();
  }

  void request_device(const uint64_t request)
  {
    const wgpu::RequestAdapterOptions options{};
    instance.RequestAdapter(
        &options,
        wgpu::CallbackMode::AllowSpontaneous,
        [this, request](
            wgpu::RequestAdapterStatus status, wgpu::Adapter adapter, wgpu::StringView) {
          if (status != wgpu::RequestAdapterStatus::Success) {
            finish(request);
            return;
          }
          wgpu::DeviceDescriptor descriptor{};
          wgpu::Limits available{};
          wgpu::Limits required{};
          if (adapter.GetLimits(&available) == wgpu::Status::Success) {
            required.maxStorageBufferBindingSize = available.maxStorageBufferBindingSize;
            required.maxBufferSize = available.maxBufferSize;
            descriptor.requiredLimits = &required;
          }
          descriptor.SetDeviceLostCallback(
              wgpu::CallbackMode::AllowSpontaneous,
              [this, request](const wgpu::Device &, wgpu::DeviceLostReason reason, wgpu::StringView) {
                if (reason != wgpu::DeviceLostReason::CallbackCancelled &&
                    reason != wgpu::DeviceLostReason::FailedCreation)
                {
                  device_lost(request);
                }
              });
          adapter.RequestDevice(
              &descriptor,
              wgpu::CallbackMode::AllowSpontaneous,
              [this, request](
                  wgpu::RequestDeviceStatus status, wgpu::Device device, wgpu::StringView) {
                finish(request,
                       status == wgpu::RequestDeviceStatus::Success ? std::move(device) :
                                                                     wgpu::Device());
              });
        });
  }

  static void *run(void *data)
  {
    auto &worker = *static_cast<ComputeWorker *>(data);
    uint64_t request;
    {
      std::lock_guard lock(worker.mutex);
      request = worker.generation;
    }
    worker.instance = wgpu::CreateInstance();
    if (!worker.instance) {
      worker.finish(request);
      return nullptr;
    }
    worker.request_device(request);
    emscripten_exit_with_live_runtime();
  }
};

ComputeWorker &compute_worker()
{
  /* The worker lives for the browser runtime, independently of render sessions. */
  static auto *worker = new ComputeWorker();
  return *worker;
}

/* The caller holds worker.mutex while starting an acquisition. */
void start_compute_worker(ComputeWorker &worker)
{
  if (worker.status == ComputeWorker::Status::Lost) {
    worker.status = ComputeWorker::Status::Starting;
    const uint64_t request = ++worker.generation;
    if (!worker.queue.proxyAsync(worker.thread,
                                 [&worker, request] { worker.request_device(request); }))
    {
      worker.status = ComputeWorker::Status::Failed;
    }
    return;
  }
  if (worker.status != ComputeWorker::Status::Idle) {
    return;
  }
  worker.status = ComputeWorker::Status::Starting;
  ++worker.generation;
  if (pthread_create(&worker.thread, nullptr, ComputeWorker::run, &worker) == 0) {
    const pthread_t thread = worker.thread;
    worker.queue.proxyAsync(emscripten_main_runtime_thread_id(),
                           [thread] { cycles_webgpu_attach_preview_transport(thread); });
    pthread_detach(worker.thread);
  }
  else {
    worker.status = ComputeWorker::Status::Failed;
  }
}

}  // namespace

bool cycles_webgpu_requested()
{
  return access("/cycles-webgpu/stock-camera.wgsl", R_OK) == 0;
}

bool cycles_webgpu_available()
{
  if (!cycles_webgpu_requested()) {
    return false;
  }
  ComputeWorker &worker = compute_worker();
  std::lock_guard lock(worker.mutex);
  return worker.status != ComputeWorker::Status::Failed;
}

bool cycles_webgpu_on_compute_device(const std::function<void(const wgpu::Device &)> &operation)
{
  if (emscripten_is_main_runtime_thread() || emscripten_is_main_browser_thread()) {
    return false;
  }
  ComputeWorker &worker = compute_worker();
  pthread_t owner;
  uint64_t generation;
  {
    std::unique_lock lock(worker.mutex);
    start_compute_worker(worker);
    worker.ready.wait_for(lock, std::chrono::seconds(2), [&] {
      return worker.status != ComputeWorker::Status::Starting;
    });
    if (worker.status != ComputeWorker::Status::Ready) {
      return false;
    }
    owner = worker.thread;
    generation = worker.generation;
  }
  bool invoked = false;
  const auto use_device = [&] {
    {
      std::lock_guard lock(worker.mutex);
      if (worker.status != ComputeWorker::Status::Ready || worker.generation != generation) {
        return;
      }
    }
    operation(worker.device);
    invoked = true;
  };
  return worker.queue.proxySync(owner, use_device) && invoked;
}

bool cycles_webgpu_oidn_available()
{
  return access("/cycles-webgpu/oidn-webgpu.enabled", R_OK) == 0;
}

bool cycles_webgpu_oidn_filter(const WebGPUOIDNRequest &request)
{
  if (!request.model || !request.output.pixels || request.width <= 0 || request.height <= 0) {
    return false;
  }

  std::atomic<int32_t> completion(0);
  if (!cycles_webgpu_on_compute_device([&](const wgpu::Device &device) {
        cycles_webgpu_launch_oidn(device.Get(),
                                 request.model,
                                 request.width,
                                 request.height,
                                 request.hdr,
                                 request.srgb,
                                 request.color.pixels,
                                 request.color.pixel_stride,
                                 request.color.row_stride,
                                 request.albedo.pixels,
                                 request.albedo.pixel_stride,
                                 request.albedo.row_stride,
                                 request.normal.pixels,
                                 request.normal.pixel_stride,
                                 request.normal.row_stride,
                                 request.output.pixels,
                                 request.output.pixel_stride,
                                 request.output.row_stride,
                                 &completion);
      }))
  {
    return false;
  }

  return wait_for_compute_worker(completion);
}

bool cycles_webgpu_publish_preview(Device &device,
                                   const device_memory &buffer,
                                   const WebGPUDisplayPreview &preview)
{
  static_assert(sizeof(preview) == 10 * sizeof(uint32_t));
  std::atomic<int32_t> completion(0);
  return device_webgpu_with_buffer(
             device,
             buffer,
             [&](const wgpu::Device &gpu, const wgpu::Buffer &source) {
               cycles_webgpu_launch_preview(gpu.Get(), source.Get(), &preview, &completion);
             }) &&
         wait_for_compute_worker(completion);
}

bool cycles_webgpu_oidn_append_tile(Device &device,
                                   const device_memory &buffer,
                                   const WebGPUOIDNFilmTile &tile)
{
  static_assert(sizeof(tile) == 11 * sizeof(uint32_t));
  std::atomic<int32_t> completion(0);
  return device_webgpu_with_buffer(
             device,
             buffer,
             [&](const wgpu::Device &gpu, const wgpu::Buffer &source) {
               cycles_webgpu_launch_resident_film(gpu.Get(),
                                                  source.Get(),
                                                  &tile,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  nullptr,
                                                  &completion);
             }) &&
         wait_for_compute_worker(completion);
}

bool cycles_webgpu_oidn_finish_film(const WebGPUOIDNFilmOutput &film,
                                    const std::array<const char *, 3> &models,
                                    float *pixels,
                                    std::array<double, 3> &times)
{
  static_assert(sizeof(film) == 7 * sizeof(uint32_t));
  std::atomic<int32_t> completion(0);
  return cycles_webgpu_on_compute_device([&](const wgpu::Device &gpu) {
           cycles_webgpu_launch_resident_film(gpu.Get(),
                                              nullptr,
                                              &film,
                                              models[0],
                                              models[1],
                                              models[2],
                                              pixels,
                                              times.data(),
                                              &completion);
         }) &&
         wait_for_compute_worker(completion);
}

void cycles_webgpu_oidn_drop_film(const uint32_t id)
{
  ComputeWorker &worker = compute_worker();
  std::lock_guard lock(worker.mutex);
  if (worker.status == ComputeWorker::Status::Ready) {
    const uint64_t generation = worker.generation;
    worker.queue.proxyAsync(worker.thread, [&worker, generation, id] {
      std::lock_guard lock(worker.mutex);
      if (worker.status == ComputeWorker::Status::Ready && worker.generation == generation) {
        cycles_webgpu_discard_resident_film(worker.device.Get(), id);
      }
    });
  }
}

const char *cycles_webgpu_oidn_model(const bool color,
                                     const bool albedo,
                                     const bool normal,
                                     const bool clean_aux,
                                     const bool high,
                                     const bool fast)
{
  if (color) {
    if (normal) {
      if (clean_aux) {
        return high ? "rt_hdr_calb_cnrm_large.tza" :
                      fast ? "rt_hdr_calb_cnrm_small.tza" : "rt_hdr_calb_cnrm.tza";
      }
      return fast ? "rt_hdr_alb_nrm_small.tza" : "rt_hdr_alb_nrm.tza";
    }
    if (albedo) {
      return fast ? "rt_hdr_alb_small.tza" : "rt_hdr_alb.tza";
    }
    return fast ? "rt_hdr_small.tza" : "rt_hdr.tza";
  }
  if (normal) {
    return high ? "rt_nrm_large.tza" : "rt_nrm.tza";
  }
  return high ? "rt_alb_large.tza" : "rt_alb.tza";
}

CCL_NAMESPACE_END
