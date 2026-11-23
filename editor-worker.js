"use strict";

importScripts("editor-local-files.js", "editor-recent-files.js");

self.addEventListener("message", function start(event) {
  const { canvas, viewport, display, assets } = event.data;
  const send = (type, ...args) => self.postMessage({ type, args });
  const reportError = error => send("error", String(error));
  const input = [];
  let handlers = null;
  let size = viewport;
  let focused = false;
  let runFrame = null;
  let animationFrame = null;
  let idleTimer = null;
  let framePending = false;
  let fileRequest = null;
  const fileHandles = new Map();
  const recentFiles = new self.BlenderRecentFiles();
  const cyclesPreviews = new Map();
  const downloadSize = new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 });

  function requestFile(type, ...args) {
    if (fileRequest) return Promise.reject(new Error("A file picker is already open."));
    return new Promise((resolve, reject) => {
      fileRequest = { resolve, reject };
      send(type, ...args);
    });
  }

  function requestFrame() {
    if (idleTimer !== null) {
      clearTimeout(idleTimer);
      idleTimer = null;
    }
    if (runFrame && animationFrame === null && !framePending) {
      animationFrame = requestAnimationFrame(() => {
        animationFrame = null;
        framePending = true;
        runFrame();
      });
    }
  }

  const bridge = {
    display,
    send,
    submitCyclesPreview(ticket, bitmap) {
      const old = cyclesPreviews.get(ticket);
      old?.close();
      cyclesPreviews.set(ticket, bitmap);
      if (cyclesPreviews.size > 8) {
        const [first, discarded] = cyclesPreviews.entries().next().value;
        cyclesPreviews.delete(first);
        discarded.close();
      }
      setTimeout(() => {
        if (cyclesPreviews.get(ticket) === bitmap) {
          cyclesPreviews.delete(ticket);
          bitmap.close();
        }
      }, 2000);
      requestFrame();
    },
    hasCyclesPreview(ticket) {
      return cyclesPreviews.has(ticket);
    },
    takeCyclesPreview(ticket) {
      const bitmap = cyclesPreviews.get(ticket);
      cyclesPreviews.delete(ticket);
      return bitmap;
    },
    chooseBlendFile() {
      return requestFile("chooseBlendFile");
    },
    chooseBlendSaveFile(suggestedName) {
      return requestFile("chooseBlendSaveFile", suggestedName);
    },
    chooseLocalDirectory(mode = "read") {
      return requestFile("chooseLocalDirectory", mode);
    },
    async createBlendPath(name, capacity, handle) {
      if (!name || /[\/\\\0]/.test(name) || name === "." || name === "..") {
        throw new Error("The selected project has an invalid file name.");
      }
      const parent = "/home/web_user";
      const previous = await recentFiles.pathForHandle(handle);
      const path = previous ?? parent + "/Browser Files/" + crypto.randomUUID() + "/" +
        name.replace(/[\r\n]/g, " ");
      if (new TextEncoder().encode(path).length >= capacity) {
        throw new Error("The selected project file name is too long.");
      }
      const directory = path.slice(parent.length + 1, path.lastIndexOf("/"));
      Module.FS_createPath(parent, directory, true, true);
      return path;
    },
    getBlendFileHandle(path) {
      return fileHandles.get(path) ?? null;
    },
    rememberBlendFile(path, handle, file) {
      if (handle) fileHandles.set(path, handle);
      recentFiles.remember(path, handle, file);
    },
    localFiles(fs) {
      return Module.blenderLocalFiles ??= new self.BlenderLocalFiles(
        fs, bridge.writeBlendFile, bridge.requestBlendFileWrite);
    },
    addBlendFile(fs, path, file) {
      fs.mkdirTree(path.slice(0, path.lastIndexOf("/")));
      if (fs.analyzePath(path).exists) fs.unlink(path);
      bridge.localFiles(fs).addFile(path, file);
    },
    restoreRecentFiles(fs, path) {
      recentFiles.restore(fs, path, bridge.localFiles(fs));
    },
    saveRecentFiles(history, opening) {
      return recentFiles.save(history, opening);
    },
    hasRecentFile(path) {
      return Boolean(recentFiles.get(path));
    },
    async reopenRecentFile(fs, path) {
      const recent = recentFiles.get(path);
      if (!recent) return true;
      const { handle } = recent;
      if (handle && await handle.queryPermission({ mode: "read" }) !== "granted" &&
          !await requestFile("requestBlendFileRead", handle)) {
        return false;
      }
      let file;
      try {
        file = handle ? await handle.getFile() : recent.file;
      } catch (error) {
        if (error.name === "NotFoundError") {
          recentFiles.forget(path);
          fileHandles.delete(path);
          if (fs.analyzePath(path).exists) fs.unlink(path);
        }
        throw error;
      }
      bridge.addBlendFile(fs, path, file);
      if (handle) fileHandles.set(path, handle);
      return true;
    },
    async requestBlendFileWrite(handle) {
      if (await handle.queryPermission({ mode: "readwrite" }) === "granted") return true;
      return requestFile("requestBlendFileWrite", handle);
    },
    async writeBlendFile(handle, contents) {
      const stream = await handle.createWritable();
      try {
        if (ArrayBuffer.isView(contents) || contents instanceof ArrayBuffer) {
          await stream.write(contents);
        } else {
          for await (const chunk of contents) await stream.write(chunk);
        }
        await stream.close();
      } catch (error) {
        await stream.abort().catch(() => {});
        throw error;
      }
    },
    attach(callbacks) {
      handlers = callbacks;
      handlers.resize(...size);
      handlers.focus(focused);
    },
    detach() {
      handlers = null;
      input.length = 0;
    },
    drain() {
      for (const { type, args } of input.splice(0)) {
        handlers?.[type === "pointerMotion" ? "pointer" : type](...args);
      }
    },
    startFrames(callback) {
      runFrame = callback;
      requestFrame();
    },
    frameDone(delay) {
      framePending = false;
      if (!runFrame) return;
      if (input.length || delay <= 16) {
        requestFrame();
      } else {
        idleTimer = setTimeout(requestFrame, delay);
      }
    },
    stopFrames() {
      if (!runFrame) return false;
      runFrame = null;
      if (animationFrame !== null) cancelAnimationFrame(animationFrame);
      if (idleTimer !== null) clearTimeout(idleTimer);
      animationFrame = idleTimer = null;
      return true;
    },
  };

  self.addEventListener("message", ({ data: { type, args } }) => {
    if (type === "filePickerResult") {
      // The current Blender frame is suspended while the picker is open.
      const request = fileRequest;
      fileRequest = null;
      if (args[1]) request?.reject(new Error(args[1]));
      else request?.resolve(args[0]);
      return;
    }
    if (type === "resize") {
      if (args.every((value, index) => value === size[index])) return;
      size = args;
    }
    if (type === "focus") focused = args[0];
    if (handlers) {
      const previous = input.at(-1);
      if (previous?.type === type &&
          (type === "resize" ||
           (type === "pointerMotion" && previous.args[2] === 0 && args[2] === 0))) {
        previous.args = args;
      } else {
        input.push({ type, args });
      }
      requestFrame();
    }
  });
  self.addEventListener("error", event => reportError(event.message));
  self.addEventListener("unhandledrejection", event => reportError(event.reason));

  const prepare = [() => {
    // Expose the keyboard platform separately from Python's Emscripten runtime.
    const platforms = [
      [/^mac/i, "darwin"],
      [/^win/i, "win32"],
      [/^(?:iOS|iPhone|iPad|iPod)/i, "ios"],
      [/^android/i, "android"],
      [/^(?:chrome os|cros)/i, "chromeos"],
      [/^linux/i, "linux"],
    ];
    for (const value of [navigator.userAgentData?.platform, navigator.platform]) {
      const match = value && platforms.find(([pattern]) => pattern.test(value));
      if (match) {
        ENV.BLENDER_WEB_HOST_PLATFORM = match[1];
        break;
      }
    }
  }];
  self.Module = {
    canvas,
    thisProgram: "/blender",
    // Emscripten's nested pthread workers load the runtime script, not this launcher.
    mainScriptUrlOrBlob: new URL("blender.js", self.location.href).href,
    ghostBrowserWindow: bridge,
    preRun: prepare,
    setStatus(message) {
      if (!message) return;
      const download = /^Downloading data\.\.\.(?: \((\d+)\/\d+\))?$/.exec(message);
      if (!download) {
        send("status", message);
        return;
      }
      const loaded = Number(download[1] ?? 0);
      // Fetch streams decoded bytes. A gzip response's Content-Length describes
      // the encoded transfer, so use the uncompressed archive sizes from the build.
      const total = assets.preloadBytes >= loaded ? assets.preloadBytes : 0;
      const scale = Math.max(loaded, total);
      const [divisor, unit] = scale >= 1e6 ? [1e6, "MB"] : scale >= 1e3 ? [1e3, "kB"] : [1, "B"];
      const amount = downloadSize.format(loaded / divisor);
      const received = total > 0 ? `${amount} of ${downloadSize.format(total / divisor)}` : amount;
      send("status", loaded || total ? `Loading Blender data… ${received} ${unit}` : "Loading Blender data…");
    },
    onRuntimeInitialized() {
      send("status", "Starting the Blender editor…");
    },
    blenderFirstFrame() {
      send("status", "Preparing the Blender viewport…");
    },
    blenderViewportReady(success) {
      send("ready", success);
    },
    onAbort: reportError,
  };

  function addDependency(name, promise, install = () => {}) {
    // Register rejection handling while importScripts is still fetching blender.js.
    const ready = promise.catch(error => {
      reportError(error);
      return null;
    });
    prepare.push(() => {
      Module.addRunDependency(name);
      ready.then(value => {
        if (value) {
          install(value);
          Module.removeRunDependency(name);
        }
      });
    });
  }

  addDependency("recent-files", recentFiles.load().then(() => true));

  if (assets.translator) {
    addDependency(
      "shader-translator",
      WebAssembly.instantiateStreaming(fetch(assets.translator), {}),
      result => { Module.shaderTranslator = result.instance.exports; },
    );
  }
  if (assets.pythonProcess) {
    addDependency(
      "python-process-client",
      Promise.all([import(assets.pythonProcess), import("./python-process/bridge.mjs")])
        .then(([client, bridge]) => bridge.createPythonProcessBridge(
          client.startPythonProcess, console.log.bind(console), console.error.bind(console),
        )),
      value => { Module.blenderPythonProcessBridge = value; },
    );
  }

  importScripts(...assets.preloads, "blender.js");
}, { once: true });
