import largeAssets from "./dist/large-assets.json";

async function streamParts(assets, origin, paths, writable) {
  try {
    for (const path of paths) {
      const part = await assets.fetch(new URL(path, origin));
      if (!part.ok || !part.body) throw new Error(`Missing runtime part ${path}`);
      await part.body.pipeTo(writable, { preventClose: true });
    }
    await writable.close();
  } catch (error) {
    await writable.abort(error);
    throw error;
  }
}

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const release = largeAssets[url.pathname];
    if (!release) return new Response("Not found", { status: 404 });
    if (request.method !== "GET" && request.method !== "HEAD") {
      return new Response(null, { status: 405, headers: { Allow: "GET, HEAD" } });
    }

    const etag = `"${release.sha256}"`;
    const headers = new Headers({
      "Cache-Control": "public, no-cache",
      "Content-Type": release.contentType,
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Resource-Policy": "same-origin",
      ETag: etag,
    });
    const validators = request.headers.get("If-None-Match")?.split(/,\s*/) ?? [];
    if (validators.some(value => value === "*" || value.replace(/^W\//, "") === etag)) {
      return new Response(null, { status: 304, headers });
    }

    headers.set("Content-Encoding", "gzip");
    headers.set("Content-Length", String(release.size));
    if (request.method === "HEAD") return new Response(null, { headers, encodeBody: "manual" });

    const { readable, writable } = new FixedLengthStream(release.size);
    ctx.waitUntil(streamParts(env.ASSETS, url.origin, release.parts, writable));
    return new Response(readable, { headers, encodeBody: "manual" });
  },
};
