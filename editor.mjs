export function startEditor(assets) {
  const canvas = document.getElementById("gpu");
  const loading = document.getElementById("loading");
  const status = document.getElementById("status");
  const controller = new AbortController();
  const { signal } = controller;
  let worker;
  let failed = false;

  function reportError(error) {
    if (failed) return;
    failed = true;
    controller.abort();
    worker?.terminate();
    canvas.inert = true;
    loading.hidden = false;
    status.hidden = false;
    status.textContent = String(error);
    loading.querySelector("progress").hidden = true;
  }

  if (!navigator.gpu || !canvas.transferControlToOffscreen) {
    reportError("Blender requires WebGPU and OffscreenCanvas in a browser worker.");
    return;
  }

  const viewport = () => [canvas.clientWidth, canvas.clientHeight, devicePixelRatio || 1];
  const send = (type, ...args) => worker.postMessage({ type, args });
  const measure = () => send("resize", ...viewport());
  const listen = (target, type, handler) =>
    target.addEventListener(type, handler, { signal, passive: false });

  let normalSize;
  let capturedPointer = null;
  let pointerX = 0;
  let pointerY = 0;
  let grabRequested = false;
  let lockPending = false;
  let wasLocked = false;
  const hasRawMouse = isSecureContext && "onpointerrawupdate" in window;
  const blendPickerOptions = {
    id: "blender-project",
    types: [{ description: "Blender Projects", accept: { "application/x-blender": [".blend"] } }],
  };

  function chooseBlendFromInput() {
    return new Promise((resolve, reject) => {
      const picker = document.createElement("input");
      picker.type = "file";
      picker.accept = ".blend";
      picker.hidden = true;
      function finish(value) {
        picker.remove();
        resolve(value);
      }
      picker.addEventListener("change", () => {
        const file = picker.files[0];
        finish(file ? { file, handle: null } : null);
      }, { once: true });
      picker.addEventListener("cancel", () => finish(null), { once: true });
      document.body.append(picker);
      try {
        if (picker.showPicker) picker.showPicker();
        else picker.click();
      } catch (error) {
        picker.remove();
        reject(error);
      }
    });
  }

  function requestFilePicker(title, action, choose) {
    let dialog;
    let settled = false;
    const cleanup = () => dialog?.remove();
    const finish = (value, error) => {
      if (settled) return;
      settled = true;
      signal.removeEventListener("abort", cleanup);
      cleanup();
      if (failed) return;
      canvas.focus({ preventScroll: true });
      try {
        send("filePickerResult", value, error);
      } catch (error) {
        send("filePickerResult", null, error.message || String(error));
      }
    };
    const run = async () => {
      try {
        finish(await choose());
      } catch (error) {
        if (error.name === "AbortError") {
          finish(null);
        } else if (!dialog && (error.name === "SecurityError" || error.name === "NotAllowedError")) {
          promptForClick();
        } else {
          finish(null, error.message || String(error));
        }
      }
    };
    function promptForClick() {
      dialog = document.createElement("dialog");
      dialog.style.colorScheme = "dark";
      dialog.setAttribute("aria-label", title);
      const confirm = document.createElement("button");
      confirm.textContent = action;
      const cancel = document.createElement("button");
      cancel.textContent = "Cancel";
      confirm.addEventListener("click", () => {
        confirm.disabled = true;
        void run();
      }, { once: true });
      cancel.addEventListener("click", () => finish(null), { once: true });
      dialog.addEventListener("cancel", event => {
        event.preventDefault();
        finish(null);
      }, { once: true });
      dialog.append(confirm, " ", cancel);
      document.body.append(dialog);
      dialog.showModal();
    }

    signal.addEventListener("abort", cleanup, { once: true });
    if (navigator.userActivation?.isActive !== false) void run();
    else promptForClick();
  }

  async function setCursorGrab(active) {
    grabRequested = active;
    if (!active) {
      if (document.pointerLockElement === canvas) document.exitPointerLock();
      return;
    }
    if (lockPending || document.pointerLockElement || !canvas.requestPointerLock) return;

    lockPending = true;
    try {
      await canvas.requestPointerLock({ unadjustedMovement: true });
    } catch (error) {
      if (grabRequested && !failed && error.name === "NotSupportedError") {
        try {
          await canvas.requestPointerLock();
        } catch {
          // Absolute pointer events remain available when locking is denied.
        }
      }
    } finally {
      lockPending = false;
      if ((!grabRequested || failed) && document.pointerLockElement === canvas) {
        document.exitPointerLock();
      }
    }
  }

  const pointer = event => {
    if (!event.isPrimary || (capturedPointer !== null && capturedPointer !== event.pointerId)) return;
    const mouse = event.pointerType === "mouse";
    const motion = event.type === "pointermove" || event.type === "pointerrawupdate";
    if (event.type === "pointerrawupdate" && !mouse) return;
    if (event.type === "pointermove" && mouse && hasRawMouse && event.isTrusted) return;
    const locked = mouse && document.pointerLockElement === canvas;
    if (event.type === "pointerdown") {
      canvas.focus({ preventScroll: true });
      if (event.isTrusted && !locked) {
        canvas.setPointerCapture(event.pointerId);
        capturedPointer = event.pointerId;
      }
      event.preventDefault();
    }
    const eraser = event.pointerType === "pen" && (event.button === 5 || (event.buttons & 32));
    const tablet = eraser ? 2 : event.pointerType === "pen" ? 1 : 0;
    const buttons = eraser ? ((event.buttons & ~32) | ((event.buttons & 32) ? 1 : 0)) : event.buttons;
    const samples = motion && mouse && (locked || buttons) && event.getCoalescedEvents?.();
    for (const sample of samples?.length ? samples : [event]) {
      if (locked && motion) {
        pointerX += sample.movementX;
        pointerY += sample.movementY;
      } else if (!locked) {
        pointerX = event.offsetX + (sample.clientX - event.clientX);
        pointerY = event.offsetY + (sample.clientY - event.clientY);
      }
      send(motion && mouse ? "pointerMotion" : "pointer", pointerX, pointerY, buttons, tablet,
           sample.pressure, sample.tiltX, sample.tiltY);
    }
  };
  const cancelPointer = event => {
    if (!event.isPrimary || (capturedPointer !== null && event.pointerId !== capturedPointer)) return;
    capturedPointer = null;
    if (event.type === "lostpointercapture" && document.pointerLockElement === canvas) return;
    send("releaseButtons");
  };
  const key = event => {
    if (event.isComposing) return;
    const modifiers = (event.ctrlKey ? 1 : 0) | (event.altKey ? 2 : 0) | (event.metaKey ? 4 : 0);
    send("key", event.type === "keydown", event.code, event.key, modifiers, event.repeat);
    event.preventDefault();
  };

  try {
    worker = new Worker(new URL("editor-worker.js", import.meta.url), { name: "Blender" });
    listen(worker, "error", event => reportError(event.message || "The Blender worker failed to load."));
    listen(worker, "messageerror", () => reportError("Could not read a message from the Blender worker."));
    listen(worker, "message", ({ data: { type, args } }) => {
      switch (type) {
        case "status":
          status.textContent = args[0];
          break;
        case "ready":
          if (!args[0]) {
            reportError("WebGPU could not finish drawing the Blender viewport.");
            break;
          }
          requestAnimationFrame(() => {
            if (failed) return;
            loading.hidden = true;
            status.hidden = true;
            canvas.inert = false;
          });
          break;
        case "error":
          reportError(args[0]);
          break;
        case "cursor":
          canvas.style.cursor = args[0];
          break;
        case "cursorGrab":
          void setCursorGrab(args[0]);
          break;
        case "chooseBlendFile":
          requestFilePicker("Open Blender Project", "Choose .blend File", async () => {
            if (!window.showOpenFilePicker) return chooseBlendFromInput();
            const [handle] = await window.showOpenFilePicker(blendPickerOptions);
            return { file: await handle.getFile(), handle };
          });
          break;
        case "chooseBlendSaveFile":
          if (!window.showSaveFilePicker) {
            send("filePickerResult", null, "This browser does not support saving to a local file.");
            break;
          }
          requestFilePicker("Save Blender Project", "Choose Save Location", () => (
            window.showSaveFilePicker({ ...blendPickerOptions, suggestedName: args[0] })
          ));
          break;
        case "chooseLocalDirectory":
          if (!window.showDirectoryPicker) {
            send("filePickerResult", null, "This browser does not support selecting a local folder.");
            break;
          }
          requestFilePicker("Select a Local Folder", "Choose Folder", () => (
            window.showDirectoryPicker({ mode: args[0] })
          ));
          break;
        case "requestBlendFileWrite":
          requestFilePicker("Allow Blender to Write", "Allow Write Access", async () => (
            await args[0].requestPermission({ mode: "readwrite" }) === "granted"
          ));
          break;
        case "title":
          document.title = args[0];
          break;
        case "resize":
          canvas.style.width = args[0] + "px";
          canvas.style.height = args[1] + "px";
          measure();
          break;
        case "maximize":
          if (args[0]) {
            normalSize ??= [canvas.style.width, canvas.style.height];
            canvas.style.width = "100vw";
            canvas.style.height = "100vh";
          } else if (normalSize) {
            [canvas.style.width, canvas.style.height] = normalSize;
            normalSize = undefined;
          }
          measure();
          break;
      }
    });

    const offscreen = canvas.transferControlToOffscreen();
    const loadCyclesShaders = !!assets.cyclesWebGPU;
    worker.postMessage({
      canvas: offscreen,
      viewport: viewport(),
      display: [screen.width, screen.height],
      assets: loadCyclesShaders
        ? { ...assets, preloads: [...assets.preloads, assets.cyclesWebGPU] }
        : assets,
    }, [offscreen]);
  } catch (error) {
    reportError(error);
    return;
  }

  const observer = new ResizeObserver(measure);
  observer.observe(canvas);
  signal.addEventListener("abort", () => observer.disconnect(), { once: true });
  function watchScale() {
    const monitor = matchMedia(`(resolution: ${devicePixelRatio}dppx)`);
    monitor.addEventListener("change", () => {
      measure();
      watchScale();
    }, { once: true, signal });
  }
  watchScale();
  listen(window, "resize", measure);
  for (const type of ["pointerdown", "pointermove", "pointerup"]) listen(canvas, type, pointer);
  if (hasRawMouse) listen(canvas, "pointerrawupdate", pointer);
  listen(document, "pointerlockchange", () => {
    const locked = document.pointerLockElement === canvas;
    if (wasLocked && !locked && grabRequested) {
      grabRequested = false;
      send("releaseButtons");
      // The browser consumes Escape when it uses the key to release the pointer.
      send("key", true, "Escape", "Escape", 0, false);
      send("key", false, "Escape", "Escape", 0, false);
    } else if (locked && !grabRequested) {
      document.exitPointerLock();
    }
    wasLocked = locked;
  });
  signal.addEventListener("abort", () => {
    if (document.pointerLockElement === canvas) document.exitPointerLock();
  }, { once: true });
  listen(canvas, "pointercancel", cancelPointer);
  listen(canvas, "lostpointercapture", cancelPointer);
  listen(canvas, "contextmenu", event => event.preventDefault());
  listen(canvas, "wheel", event => {
    const mode = event.deltaMode;
    const { deltaX, deltaY } = event;
    if (mode !== WheelEvent.DOM_DELTA_PIXEL) {
      send("wheel", deltaX, deltaY);
    } else {
      // Chrome exposes discrete mouse detents through the legacy wheelDelta fields.
      // Its trusted event paths can express these in device or CSS pixels.
      const detent = (legacy, delta, scale) => {
        if (delta === 0) return 0;
        const value = -legacy * scale / 120;
        const rounded = Math.round(value);
        return rounded && Math.abs(rounded - value) <= scale / 120 ? rounded : null;
      };
      let ticks;
      if (event.isTrusted) {
        for (const scale of [devicePixelRatio || 1, 1]) {
          const x = detent(event.wheelDeltaX, deltaX, scale);
          const y = detent(event.wheelDeltaY, deltaY, scale);
          if (x !== null && y !== null && (x || y)) {
            ticks = [x, y];
            break;
          }
        }
      }
      if (ticks) send("wheel", ...ticks);
      else send("scroll", deltaX, deltaY, event.ctrlKey);
    }
    event.preventDefault();
  });
  listen(canvas, "focus", () => send("focus", true));
  listen(canvas, "blur", () => send("focus", false));
  listen(window, "blur", () => send("focus", false));
  listen(window, "focus", () => {
    if (document.activeElement === canvas) send("focus", true);
  });
  listen(canvas, "keydown", key);
  listen(canvas, "keyup", key);
}
