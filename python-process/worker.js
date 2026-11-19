// SPDX-License-Identifier: GPL-2.0-or-later

function createChannelRequest() {
  const mailbox = new SharedArrayBuffer(12 + 64 * 1024);

  function exchange(request, response, transfer = []) {
    const header = new Int32Array(response, 0, 3);
    Atomics.store(header, 0, 0);
    Atomics.store(header, 2, 0);
    self.postMessage({channel: {...request, response}}, transfer);
    Atomics.wait(header, 0, 0);
    return {status: Atomics.load(header, 1), length: Atomics.load(header, 2)};
  }

  return (operation, id, argument, bytes, seconds) => {
    let response = mailbox;
    let {status, length} = exchange({operation, id, argument, bytes, seconds}, response,
                                    bytes?.length ? [bytes.buffer] : []);
    if (status < 0 || operation !== 2) return status;
    if (length > response.byteLength - 12) {
      response = new SharedArrayBuffer(12 + length);
      ({status, length} = exchange({operation: "take"}, response));
      if (status < 0) return status;
    }
    return new Uint8Array(response, 12, length);
  };
}

function createSharedMemoryRequest() {
  const response = new SharedArrayBuffer(32);
  const header = new Int32Array(response, 0, 2);
  const output = new DataView(response);
  return (operation, name, size, track, lease) => {
    Atomics.store(header, 0, 0);
    self.postMessage({sharedMemory: {operation, name, size, track, lease, response}});
    Atomics.wait(header, 0, 0);
    const status = Atomics.load(header, 1);
    if (status !== 0 || operation >= 2) return {status};
    return {
      status,
      pointer: output.getBigUint64(8, true),
      size: Number(output.getBigUint64(16, true)),
      lease: output.getUint32(24, true),
    };
  };
}

self.onmessage = async ({data: {source, input, shared}}) => {
  // One Worker owns exactly one interpreter and its virtual filesystem.
  self.onmessage = null;
  try {
    const runtimeFile = shared ? "blender-python-child-shared.js" : "blender-python-child.js";
    const mainScriptUrlOrBlob = new URL(runtimeFile, self.location.href).href;
    importScripts(mainScriptUrlOrBlob);
    const options = {channelRequest: createChannelRequest(), mainScriptUrlOrBlob};
    if (shared) {
      options.wasmMemory = shared.memory;
      options.sharedArena = shared.arena;
      options.sharedMemoryRequest = createSharedMemoryRequest();
    }
    const python = await createBlenderPythonChild(options);
    if (shared) {
      // The standalone data archive consists of this one zip file.
      const response = await fetch("blender-python-child.data");
      if (!response.ok) throw new Error("Could not load the Python standard library");
      python.FS.mkdir("/lib");
      python.FS.writeFile("/lib/python313.zip", new Uint8Array(await response.arrayBuffer()),
                          {canOwn: true});
    }
    python.FS.mkdir("/process");
    python.FS.writeFile("/process/job.py", source);
    python.FS.writeFile("/process/stdin", input);
    let exitCode;
    try {
      exitCode = python._blender_python_child_run();
    } catch (status) {
      // Emscripten represents a C exit() as a JavaScript exception.
      if (status?.name !== "ExitStatus" || !Number.isInteger(status.status)) {
        throw status;
      }
      exitCode = status.status;
    }
    const readOutput = (path) => python.FS.analyzePath(path).exists
      ? python.FS.readFile(path)
      : new Uint8Array();
    const stdout = readOutput("/process/stdout");
    const stderr = readOutput("/process/stderr");
    // Let completed pthreads report cleanup before deciding whether the arena can be recycled.
    if (shared) await new Promise(resolve => setTimeout(resolve, 0));
    const canReleaseSharedMemory = !shared || python.sharedArenaIdle();
    self.postMessage({exitCode, stdout, stderr, canReleaseSharedMemory},
                     [stdout.buffer, stderr.buffer]);
  } catch (error) {
    self.postMessage({error: error?.stack || error?.message || String(error)});
  }
  self.close();
};
