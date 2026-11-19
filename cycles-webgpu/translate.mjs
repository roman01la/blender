// SPDX-License-Identifier: Apache-2.0

import { readFile, writeFile } from 'node:fs/promises';

const [translatorPath, inputPath, outputPath] = process.argv.slice(2);
const { instance: { exports } } = await WebAssembly.instantiate(await readFile(translatorPath));
const input = await readFile(inputPath);
const offset = exports.naga_alloc(input.length);
new Uint8Array(exports.memory.buffer, offset, input.length).set(input);
const status = exports.naga_translate(offset, input.length, 0, 0, 0);
const result = new Uint8Array(exports.memory.buffer, exports.naga_result_ptr(), exports.naga_result_size());
if (status !== 0) {
  throw Error(new TextDecoder().decode(result));
}
await writeFile(outputPath, result);
