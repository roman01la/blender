// SPDX-License-Identifier: GPL-2.0-or-later

/** Run Python source in a fresh browser Worker with binary input and output. */
export function startPythonProcess(source, input = new Uint8Array(), channelRequest, runtime) {
  if (typeof source !== "string" || !(input instanceof Uint8Array)) {
    throw new TypeError("Expected Python source and a Uint8Array");
  }
  if (!self.crossOriginIsolated) {
    throw new Error("The Wasm Python build requires cross-origin isolation");
  }

  const worker = new Worker(new URL("./worker.js", import.meta.url));
  const pthreads = new Map();
  let shared;
  let arenaAddress;
  if (runtime) {
    const size = 64n * 1024n * 1024n;
    try {
      arenaAddress = runtime.allocate(size);
    } catch (error) {
      worker.terminate();
      throw error;
    }
    if (typeof arenaAddress !== "bigint" || arenaAddress <= 0n) {
      worker.terminate();
      throw new Error("Could not allocate shared Wasm Python worker memory");
    }
    shared = {memory: runtime.memory, arena: {address: arenaAddress, size}};
  }

  let canReleaseSharedMemory = !runtime;
  function releaseWorker(safeToRecycle = false) {
    worker.terminate();
    for (const [port, pthread] of pthreads) {
      pthread.terminate();
      port.close();
    }
    pthreads.clear();
    // Browser hard termination provides no acknowledgement that Wasm execution stopped.
    // Retain an arena until the session ends when the Worker could not confirm quiescence.
    if (arenaAddress && safeToRecycle) {
      runtime.free(arenaAddress);
      arenaAddress = 0n;
      canReleaseSharedMemory = true;
    }
  }

  function startSharedPThread({port, ready}) {
    let status = -1;
    try {
      if (!runtime) return;
      const script = new URL("./blender-python-child-shared.js", import.meta.url);
      const pthread = new Worker(script, {name: "em-pthread"});
      pthreads.set(port, pthread);
      pthread.onmessage = ({data}) => port.postMessage({message: data});
      pthread.onerror = ({message, filename, lineno}) => {
        port.postMessage({error: {message, filename, lineno}});
      };
      port.onmessage = ({data}) => {
        if (data.terminate) {
          pthread.terminate();
          pthreads.delete(port);
          port.close();
        } else {
          pthread.postMessage(data.message, data.transfer);
        }
      };
      status = 1;
    } catch {
      status = -1;
    } finally {
      Atomics.store(ready, 0, status);
      Atomics.notify(ready, 0);
    }
  }

  function handleSharedMemory({operation, name, size, track, lease, response}) {
    const header = new Int32Array(response, 0, 2);
    let status = -4;
    try {
      const result = runtime.request(operation, name, size, track, lease);
      if (result.status === 0 && operation < 2) {
        const output = new DataView(response);
        output.setBigUint64(8, result.pointer, true);
        output.setBigUint64(16, BigInt(result.size), true);
        output.setUint32(24, result.lease, true);
      }
      status = result.status;
    } catch {
      status = -4;
    } finally {
      Atomics.store(header, 1, status);
      Atomics.store(header, 0, 1);
      Atomics.notify(header, 0);
    }
  }

  let deferredOutput;
  let rejectResult;
  let settled = false;
  async function handleChannel({operation, id, argument, bytes, seconds, response}) {
    const header = new Int32Array(response, 0, 3);
    const output = new Uint8Array(response, 12);
    let result;
    try {
      if (operation === "take") {
        result = deferredOutput ?? -4;
        deferredOutput = undefined;
      } else {
        result = channelRequest?.(operation, id, argument, bytes, seconds) ?? -4;
        if (result instanceof Promise) result = await result;
      }
      if (result instanceof Uint8Array) {
        if (result.length > output.length) deferredOutput = result;
        else output.set(result);
        Atomics.store(header, 2, result.length);
        result = 0;
      }
    } catch {
      result = -4;
    }
    Atomics.store(header, 1, result);
    Atomics.store(header, 0, 1);
    Atomics.notify(header, 0);
  }
  const result = new Promise((resolve, reject) => {
    rejectResult = reject;
    worker.onmessage = ({data}) => {
      if (settled) return;
      if (data.channel) {
        void handleChannel(data.channel);
        return;
      }
      if (data.sharedMemory) {
        handleSharedMemory(data.sharedMemory);
        return;
      }
      if (data.sharedPThread) {
        startSharedPThread(data.sharedPThread);
        return;
      }
      settled = true;
      releaseWorker(data.canReleaseSharedMemory === true);
      if (data.error) {
        reject(new Error(data.error));
      } else {
        resolve(data);
      }
    };
    const onFailure = (event) => {
      if (settled) return;
      settled = true;
      releaseWorker();
      reject(new Error(event.message || "Python worker failed"));
    };
    worker.onerror = onFailure;
    worker.onmessageerror = onFailure;
  });
  const bytes = input.slice();
  try {
    worker.postMessage({source, input: bytes, shared}, [bytes.buffer]);
  } catch (error) {
    settled = true;
    releaseWorker(true);
    rejectResult(error);
  }

  return {
    result,
    get canReleaseSharedMemory() {
      return canReleaseSharedMemory;
    },
    terminate() {
      if (!settled) {
        settled = true;
        releaseWorker();
        rejectResult(new DOMException("Python worker terminated", "AbortError"));
      }
    },
  };
}
