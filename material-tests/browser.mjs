// SPDX-License-Identifier: GPL-2.0-or-later

import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { createReadStream } from 'node:fs';
import { mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { extname, join, resolve, sep } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

const [buildPath, requestPath, catalogPath, timeout = '600'] = process.argv.slice(2);
const build = resolve(buildPath);
const request = JSON.parse(await readFile(requestPath, 'utf8'));
const scriptPaths = ['catalog.py', 'probe.py'];
const files = await Promise.all(scriptPaths.map(name => readFile(new URL(name, import.meta.url), 'utf8')));
const editor = await readFile(join(build, 'editor-worker.js'), 'utf8');
const anchor = '    thisProgram: "/blender",';
if (editor.split(anchor).length !== 2) throw Error('Cannot enable Cycles and GPU logging in this build');
const testEditor = editor.replace(anchor, anchor + '\n    arguments: ["--log", "cycles,gpu", "--log-level", "info"],');
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.wasm': 'application/wasm', '.png': 'image/png' };
const server = createServer(async (incoming, response) => {
  try {
    const path = decodeURIComponent(new URL(incoming.url, 'http://localhost').pathname);
    const file = resolve(build, '.' + (path === '/' ? '/index.html' : path));
    if (!file.startsWith(build + sep)) throw Error('Outside build');
    const info = await stat(file);
    if (!info.isFile()) throw Error('Not a file');
    response.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    response.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    response.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
    response.setHeader('Content-Type', mime[extname(file)] ?? 'application/octet-stream');
    if (path === '/editor-worker.js') response.end(testEditor);
    else {
      response.setHeader('Content-Length', info.size);
      const stream = createReadStream(file);
      response.on('close', () => stream.destroy());
      stream.on('error', () => response.destroy());
      stream.pipe(response);
    }
  } catch {
    response.writeHead(404).end();
  }
});
server.listen(0, '127.0.0.1');
await once(server, 'listening');

const profile = await mkdtemp(join(tmpdir(), 'blender-material-tests-'));
const chrome = spawn(process.env.CHROME ?? (process.platform === 'darwin'
  ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' : 'google-chrome'), [
  '--headless', '--remote-debugging-pipe', '--no-first-run', '--no-default-browser-check',
  '--user-data-dir=' + profile, 'about:blank',
], { stdio: ['ignore', 'ignore', 'inherit', 'pipe', 'pipe'] });

const pending = new Map();
let sequence = 0;
let input = '';
let editorWorker;
let outcome;
let failure;

function send(method, params = {}, sessionId) {
  return new Promise((resolve, reject) => {
    const id = ++sequence;
    pending.set(id, { resolve, reject });
    chrome.stdio[3].write(JSON.stringify({ id, method, params, sessionId }) + '\0', error => {
      if (!error) return;
      pending.delete(id);
      reject(error);
    });
  });
}

chrome.once('error', error => { failure = error; });
chrome.stdio[4].setEncoding('utf8').on('data', chunk => {
  const packets = (input + chunk).split('\0');
  input = packets.pop();
  for (const packet of packets) {
    const event = JSON.parse(packet);
    if (event.id) {
      const waiter = pending.get(event.id);
      pending.delete(event.id);
      if (event.error) waiter?.reject(Error(event.error.message));
      else waiter?.resolve(event.result);
    } else if (event.method === 'Target.attachedToTarget') {
      const { sessionId, targetInfo } = event.params;
      if (targetInfo.url.endsWith('/editor-worker.js')) editorWorker = sessionId;
      if (targetInfo.type === 'worker') {
        send('Runtime.enable', {}, sessionId).catch(() => {});
        autoAttach(sessionId).catch(() => {});
      }
    } else if (event.method === 'Runtime.consoleAPICalled' || event.method === 'Log.entryAdded') {
      const message = event.params.entry?.text ??
        event.params.args.map(arg => arg.value ?? arg.description ?? '').join(' ');
      console.log(message);
      if (message.startsWith('MATERIAL_TEST ')) {
        const result = JSON.parse(message.slice('MATERIAL_TEST '.length));
        if (['done', 'inventory', 'fatal'].includes(result.event)) outcome = result;
      }
    } else if (event.method === 'Inspector.targetCrashed') {
      failure = Error('Chrome target crashed');
    } else if (event.method === 'Runtime.exceptionThrown') {
      console.log(event.params.exceptionDetails.exception?.description ?? event.params.exceptionDetails.text);
    }
  }
});
function autoAttach(sessionId) {
  return send('Target.setAutoAttach', { autoAttach: true, waitForDebuggerOnStart: false,
    flatten: true, filter: [{ type: 'worker', exclude: false }, { exclude: true }] }, sessionId);
}

async function evaluate(expression, target) {
  const result = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true }, target);
  if (result.exceptionDetails) throw Error(result.exceptionDetails.exception?.description ?? result.exceptionDetails.text);
  return result.result.value;
}

async function waitFor(check, seconds) {
  const end = Date.now() + seconds * 1000;
  do {
    if (failure) throw failure;
    if (await check()) return;
    await delay(100);
  } while (Date.now() < end);
  throw Error('Blender test timed out');
}

const watchdog = setTimeout(() => {
  const error = Error('Chrome material batch exceeded the deadline');
  failure = error;
  for (const waiter of pending.values()) waiter.reject(error);
  pending.clear();
  chrome.kill();
}, (Number(timeout) + 70) * 1000);
try {
  const { targetId } = await send('Target.createTarget', { url: 'about:blank' });
  const { sessionId: tab } = await send('Target.attachToTarget', { targetId, flatten: true });
  await send('Page.enable', {}, tab);
  await send('Log.enable', {}, tab);
  await autoAttach(tab);
  await send('Emulation.setDeviceMetricsOverride', { width: 960, height: 680, deviceScaleFactor: 1, mobile: false }, tab);
  const navigation = await send('Page.navigate', { url: `http://127.0.0.1:${server.address().port}/` }, tab);
  if (navigation.errorText) throw Error(navigation.errorText);
  await waitFor(() => editorWorker && evaluate('document.querySelector("#loading")?.hidden === true && !document.querySelector("canvas").inert', tab), 55);

  for (const [i, filename] of scriptPaths.entries()) {
    await evaluate(`FS.writeFile("/tmp/${filename}", ${JSON.stringify(files[i])}); true`, editorWorker);
  }
  const wire = { ...request, image: '/tmp/node-probe.png' };
  if (wire.inventory) wire.inventory = '/tmp/node-catalog.json';
  await evaluate(`FS.writeFile("/tmp/matrix-request.json", ${JSON.stringify(JSON.stringify(wire))}); true`, editorWorker);

  const key = (type, key, code, windowsVirtualKeyCode, modifiers = 0) =>
    send('Input.dispatchKeyEvent', { type, key, code, windowsVirtualKeyCode, modifiers }, tab);
  await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: 440, y: 315 }, tab);
  for (const type of ['mousePressed', 'mouseReleased']) {
    await send('Input.dispatchMouseEvent', { type, x: 440, y: 315, button: 'left', clickCount: 1 }, tab);
  }
  await key('keyDown', 'Shift', 'ShiftLeft', 16, 8);
  await key('keyDown', 'F4', 'F4', 115, 8);
  await key('keyUp', 'F4', 'F4', 115, 8);
  await key('keyUp', 'Shift', 'ShiftLeft', 16);
  await delay(700);
  const command = "import sys,json;sys.path.insert(0,'/tmp');import probe;probe.run(json.load(open('/tmp/matrix-request.json')))";
  await evaluate(`(() => {
    const canvas = document.querySelector("canvas");
    for (const key of ${JSON.stringify(command)}) {
      const code = /[a-z]/i.test(key) ? 'Key' + key.toUpperCase() : 'Unidentified';
      for (const type of ['keydown', 'keyup']) {
        canvas.dispatchEvent(new KeyboardEvent(type, {key, code, bubbles:true, cancelable:true}));
      }
    }
    return true;
  })()`, tab);
  await key('keyDown', 'Enter', 'Enter', 13);
  await key('keyUp', 'Enter', 'Enter', 13);
  await waitFor(() => outcome, Number(timeout));
  if (outcome.event === 'fatal') throw Error(outcome.reason);
  if (wire.inventory) {
    const catalog = await evaluate('FS.readFile("/tmp/node-catalog.json", {encoding:"utf8"})', editorWorker);
    await writeFile(catalogPath, catalog + '\n');
  }
} finally {
  clearTimeout(watchdog);
  if (chrome.exitCode === null && chrome.signalCode === null && chrome.pid) {
    const exit = once(chrome, 'exit');
    chrome.kill();
    await exit;
  }
  for (const pipe of [chrome.stdio[3], chrome.stdio[4]]) pipe.destroy();
  server.closeAllConnections();
  server.close();
  await rm(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
}
