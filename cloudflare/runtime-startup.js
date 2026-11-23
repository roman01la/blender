// SPDX-License-Identifier: GPL-2.0-or-later

async function enableRuntimeCache(release) {
  const workers = navigator.serviceWorker;
  if (!workers || !navigator.storage?.getDirectory) return;
  const worker = new URL("./runtime-cache.js", import.meta.url);
  worker.searchParams.set("release", release);
  const controlled = () => workers.controller?.scriptURL === worker.href;
  const listener = new AbortController();
  let timer;
  try {
    await Promise.race([
      (async () => {
        await workers.register(worker, { updateViaCache: "none" });
        if (controlled()) return;
        await new Promise(resolve => {
          workers.addEventListener("controllerchange", () => {
            if (controlled()) resolve();
          }, { signal: listener.signal });
        });
      })(),
      // The versioned runtime URLs also work when storage or worker activation is blocked.
      new Promise(resolve => { timer = setTimeout(resolve, 3000); }),
    ]);
  } catch {
    // Continue with ordinary network requests.
  } finally {
    clearTimeout(timer);
    listener.abort();
  }
}
