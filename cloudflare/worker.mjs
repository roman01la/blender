import objects from "./dist/objects.json";

export default {
  async fetch(request, env) {
    const path = new URL(request.url).pathname;
    const release = objects[path];
    if (!release) return new Response("Not found", { status: 404 });
    if (request.method !== "GET" && request.method !== "HEAD") {
      return new Response(null, { status: 405, headers: { Allow: "GET, HEAD" } });
    }

    const headers = new Headers({
      "Cache-Control": "public, no-cache",
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Resource-Policy": "same-origin",
    });
    const object = request.method === "HEAD"
      ? await env.RELEASES.head(release)
      : await env.RELEASES.get(release);
    if (!object) return new Response("Release asset unavailable", { status: 503, headers });

    object.writeHttpMetadata(headers);
    headers.set("ETag", object.httpEtag);
    const validators = request.headers.get("If-None-Match")?.split(/,\s*/) ?? [];
    if (validators.some(value => value === "*" || value.replace(/^W\//, "") === object.httpEtag)) {
      if (object.body) await object.body.cancel();
      return new Response(null, { status: 304, headers });
    }
    headers.set("Content-Length", String(object.size));
    return new Response(object.body ?? null, { headers, encodeBody: "manual" });
  },
};
