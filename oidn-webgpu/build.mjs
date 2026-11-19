// SPDX-License-Identifier: GPL-2.0-or-later
import {build} from "esbuild";
import {mkdir, readFile, writeFile} from "node:fs/promises";
import {dirname, resolve} from "node:path";
import {fileURLToPath} from "node:url";

if (process.argv.length !== 3) {
  throw new Error("Usage: node web/oidn-webgpu/build.mjs SHADER_OUTPUT_DIR");
}

const source = dirname(fileURLToPath(import.meta.url));
const out = resolve(process.argv[2]);
await mkdir(out, {recursive: true});
await build({
  absWorkingDir: source,
  bundle: true,
  entryPoints: ["runtime.mjs"],
  format: "esm",
  minify: true,
  outfile: resolve(out, "oidn-webgpu.mjs"),
  platform: "browser",
  target: ["es2022"],
});
const executorLicense = await readFile(resolve(source, "node_modules/oidn-webgpu/LICENSE"), "utf8");
const float16License = await readFile(resolve(source, "node_modules/@petamoriken/float16/LICENSE"), "utf8");
const apacheLicense = await readFile(resolve(source, "../../blender/release/license/spdx/Apache-2.0.txt"), "utf8");
await writeFile(resolve(out, "oidn-webgpu-LICENSE.txt"),
  `oidn-webgpu 0.6.0 — https://github.com/pissang/oidn-webgpu\n${executorLicense}\n` +
  `@petamoriken/float16 — https://github.com/petamoriken/float16\n${float16License}\n` +
  "This bundle adapts OIDN input, output and exposure stages.\n" +
  `Copyright 2018-2022 Intel Corporation. Licensed under Apache-2.0.\n${apacheLicense}\n`);
