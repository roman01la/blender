// SPDX-License-Identifier: GPL-2.0-or-later

/* runtime manifest */

const scope = new URL(self.registration.scope);

async function openCache() {
  try {
    const root = await navigator.storage.getDirectory();
    return await root.getDirectoryHandle("blender-runtime", { create: true });
  } catch {
    return null;
  }
}

self.addEventListener("install", event => event.waitUntil(self.skipWaiting()));
self.addEventListener("activate", event => {
  event.waitUntil(self.clients.claim());
  event.waitUntil((async () => {
    const cache = await openCache();
    if (!cache) return;
    const current = new Set(Object.values(assets).map(asset => asset.digest));
    for await (const name of cache.keys()) {
      if (!current.has(name)) await cache.removeEntry(name).catch(() => {});
    }
  })().catch(() => {}));
});

async function serve(event, asset) {
  const cache = await openCache();
  if (!cache) return fetch(event.request);

  const { digest, size, type } = asset;
  try {
    const handle = await cache.getFileHandle(digest);
    const file = await handle.getFile();
    if (file.size === size) {
      return new Response(file, { headers: {
        "Content-Type": type,
        "Content-Length": String(size),
        "Cross-Origin-Embedder-Policy": "require-corp",
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Resource-Policy": "same-origin",
      } });
    }
  } catch {
    // A missing or unreadable file is a cache miss.
  }

  const response = await fetch(event.request);
  if (response.status !== 200 || response.redirected || !response.body) return response;
  let writer;
  try {
    const handle = await cache.getFileHandle(digest, { create: true });
    writer = await handle.createWritable();
  } catch {
    return response;
  }
  async function discard() {
    if (!writer) return;
    const writable = writer;
    writer = null;
    await writable.abort().catch(() => {});
  }

  let bytes = 0;
  // Backpressure keeps the download from buffering a second copy of the archive.
  // createWritable publishes the replacement only when it closes successfully.
  const stream = new TransformStream({
    async transform(chunk, controller) {
      bytes += chunk.byteLength;
      if (writer) {
        try {
          await writer.write(chunk);
        } catch {
          await discard();
        }
      }
      controller.enqueue(chunk);
    },
    async flush() {
      if (writer && bytes === size) await writer.close().catch(() => {});
      else await discard();
    },
  });
  event.waitUntil(response.body.pipeTo(stream.writable).catch(discard));

  // Fetch has already decoded the response body, including gzip from R2.
  const headers = new Headers(response.headers);
  headers.delete("Content-Encoding");
  headers.set("Content-Length", String(size));
  return new Response(stream.readable, { headers });
}

self.addEventListener("fetch", event => {
  const { request } = event;
  if (request.method !== "GET" || request.mode === "navigate" || request.headers.has("Range")) return;
  const url = new URL(request.url);
  if (url.origin !== scope.origin || !url.pathname.startsWith(scope.pathname) || url.search) return;
  const name = url.pathname.slice(scope.pathname.length);
  if (Object.hasOwn(assets, name)) event.respondWith(serve(event, assets[name]));
});
