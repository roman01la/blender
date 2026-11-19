// SPDX-License-Identifier: GPL-2.0-or-later

import {createPythonChannels} from './channels.mjs';
import {createSharedMemoryRegistry} from './shared-memory.mjs';

const RUNNING = -2147483648;

export function createPythonProcessBridge(startProcess, write, writeError) {
  const records = new Map();
  const channels = createPythonChannels();
  const decoder = new TextDecoder();
  const pools = new Map();
  const readyPools = new Set();
  let sharedMemory;
  let memoryRuntime;
  let wakePools = () => {};
  let nextId = 1;

  function notifyPool(channel) {
    const pool = pools.get(channel);
    if (!pool) return;
    if (pool.waiter) pool.waiter(true);
    else {
      readyPools.add(channel);
      wakePools();
    }
  }

  function childRequest(client, operation, channel, ...args) {
    const result = channels.request(client, operation, channel, ...args);
    if (operation !== 1 && operation !== 'put') return result;
    const notify = status => {
      if (status === 1) notifyPool(channel);
      return status;
    };
    return result instanceof Promise ? result.then(notify) : notify(result);
  }

  return {
    start(source, bytes, tickets = []) {
      if (nextId > 2147483647) return 0;
      try {
        const id = nextId++;
        const runtime = memoryRuntime && {
          ...memoryRuntime,
          request: (...args) => sharedMemory.request(id, ...args),
        };
        const child = startProcess(source, bytes, (...request) => childRequest(id, ...request), runtime);
        const record = {child, code: RUNNING, signal: 0};
        record.done = child.result.then(({exitCode, stdout, stderr}) => {
          if (stdout.length) write(decoder.decode(stdout));
          if (stderr.length) writeError(decoder.decode(stderr));
          record.code = exitCode;
        }).catch(error => {
          record.code = record.signal ? -record.signal : 1;
          if (!record.signal) writeError(String(error));
        }).finally(() => {
          channels.disconnect(id);
          if (child.canReleaseSharedMemory) sharedMemory?.disconnect(id);
          for (const ticket of tickets) channels.request(0, 'cancel', ticket);
          notifyPool(record.pool);
        });
        records.set(id, record);
        return id;
      } catch (error) {
        writeError(String(error));
        return 0;
      }
    },
    poll(id) {
      return records.get(id)?.code ?? 1;
    },
    async wait(id, seconds) {
      const record = records.get(id);
      if (!record) return 1;
      if (seconds < 0 || seconds === Infinity) {
        await record.done;
      } else {
        let timer;
        try {
          await Promise.race([
            record.done,
            new Promise(resolve => {
              timer = setTimeout(resolve, Math.min(seconds * 1000, 2147483647));
            }),
          ]);
        } finally {
          clearTimeout(timer);
        }
      }
      return record.code;
    },
    stop(id, signal) {
      const record = records.get(id);
      if (record?.code === RUNNING && !record.signal) {
        record.signal = signal;
        record.child.terminate();
      }
    },
    close(id) {
      records.delete(id);
    },
    channel(...request) {
      return channels.request(0, ...request);
    },
    configureSharedMemory(runtime) {
      if (!startProcess.supportsSharedMemory) return false;
      if (!memoryRuntime) {
        memoryRuntime = runtime;
        sharedMemory = createSharedMemoryRegistry(runtime.allocateZeroed, runtime.free);
      }
      return true;
    },
    sharedMemoryRequest(...request) {
      return sharedMemory?.request(0, ...request) ?? {status: -4};
    },
    setPoolWake(wake) {
      wakePools = wake;
    },
    watchPool(channel, enabled) {
      if (enabled) pools.set(channel, {waiter: null});
      else {
        pools.get(channel)?.waiter?.(false);
        pools.delete(channel);
        readyPools.delete(channel);
      }
    },
    watchPoolWorker(pid, channel) {
      records.get(pid).pool = channel;
    },
    hasPoolEvents() {
      return readyPools.size !== 0;
    },
    nextPoolEvent() {
      const channel = readyPools.values().next().value ?? 0;
      readyPools.delete(channel);
      return channel;
    },
    waitPoolEvent(channel, seconds) {
      if (readyPools.delete(channel)) return true;
      const pool = pools.get(channel);
      if (!pool || seconds === 0) return false;
      return new Promise(resolve => {
        let timer;
        const finish = ready => {
          clearTimeout(timer);
          pool.waiter = null;
          resolve(ready);
        };
        pool.waiter = finish;
        if (seconds > 0 && Number.isFinite(seconds)) {
          const deadline = performance.now() + seconds * 1000;
          const onTimeout = () => {
            const delay = deadline - performance.now();
            if (delay <= 0) finish(false);
            else timer = setTimeout(onTimeout, Math.min(delay, 2147483647));
          };
          onTimeout();
        }
      });
    },
  };
}
