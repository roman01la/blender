// SPDX-License-Identifier: GPL-2.0-or-later

// This file is embedded in the Emscripten factory's scope with --pre-js.
Module["sharedArenaIdle"] = () => Object.keys(PThread.pthreads).length === 0;

Module["allocateSharedPThread"] = () => {
  // Chrome starts a newly constructed Worker after its creating task yields.
  // Python can wait for a pthread during that task, so create it on the owner page.
  const {port1, port2} = new MessageChannel();
  const ready = new Int32Array(new SharedArrayBuffer(4));
  self.postMessage({sharedPThread: {port: port2, ready}}, [port2]);
  Atomics.wait(ready, 0, 0);
  if (Atomics.load(ready, 0) !== 1) {
    port1.close();
    throw new Error("Could not create a shared Python pthread");
  }

  const proxy = {
    postMessage(message, transfer = []) {
      port1.postMessage({message, transfer}, transfer);
    },
    terminate() {
      port1.postMessage({terminate: true});
      port1.close();
    },
  };
  port1.onmessage = ({data}) => {
    if (data.error) proxy.onerror?.(data.error);
    else proxy.onmessage?.({data: data.message});
  };
  return proxy;
};

Module["instantiateSharedPython"] = (binary, imports) => {
  const {address, size} = Module["sharedArena"];
  const metadata = getDylinkMetadata(binary);
  const alignment = 1n << BigInt(metadata.memoryAlign);
  const base = (address + alignment - 1n) & ~(alignment - 1n);
  const dataSize = BigInt(metadata.memorySize);
  const stackLow = (base + dataSize + 15n) & ~15n;
  const stackHigh = stackLow + 2097152n;
  const end = address + size;
  if (stackHigh >= end) {
    throw new RangeError("The shared Python worker arena is too small");
  }

  const __stack_pointer = new WebAssembly.Global({value: "i64", mutable: true}, stackHigh);
  const __heap_base = new WebAssembly.Global({value: "i64"}, stackHigh);
  const tableBase = 1n << BigInt(metadata.tableAlign);
  const __indirect_function_table = new WebAssembly.Table({
    element: "anyfunc",
    address: "i64",
    initial: BigInt(metadata.tableSize) + tableBase,
  });
  Object.assign(imports.env, {
    __memory_base: base,
    __table_base: tableBase,
    __table_base32: Number(tableBase),
    __stack_pointer,
    __indirect_function_table,
  });
  imports["GOT.mem"].__heap_base.value = stackHigh;
  imports["GOT.mem"].__stack_low.value = stackLow;
  imports["GOT.mem"].__stack_high.value = stackHigh;

  if (!ENVIRONMENT_IS_PTHREAD) {
    // The linker places its one-time initialization flag in static storage.
    new Uint8Array(wasmMemory.buffer, Number(base), Number(stackLow - base)).fill(0);
  }
  const instance = new WebAssembly.Instance(binary, imports);
  const exports = {
    ...relocateExports(instance.exports, Number(base)),
    __stack_pointer,
    __heap_base,
    __indirect_function_table,
  };
  if (!ENVIRONMENT_IS_PTHREAD) {
    exports.__wasm_apply_data_relocs?.();
    exports.blender_web_configure_child_arena(end);
  }
  return {exports};
};
