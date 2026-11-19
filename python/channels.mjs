// SPDX-License-Identifier: GPL-2.0-or-later

const TIMEOUT = -1;
const EOF = -2;
const BROKEN_PIPE = -3;
const INVALID_CHANNEL = -4;
const INVALID_TIMEOUT = -5;
const INVALID_ARGUMENT = -6;
const MAX_ID = 2147483647;
const MAX_TIMER_MS = 2147483647;

const READER = 0;
const WRITER = 1;
const QUEUE = 2;

// These indexes match BlenderWebChannelOperation in channel_api.h.
const operations = [
  'create', 'put', 'get', 'poll', 'qsize', 'close', 'retain', 'release', 'export', 'import', 'cancel',
];

class Channel {
  constructor(maxsize) {
    this.maxsize = maxsize || Infinity;
    this.frames = [];
    this.head = 0;
    this.putters = new Set();
    this.getters = new Set();
    this.pollers = new Set();
    this.explicitlyClosed = false;
    this.mode = null;
    this.leases = new Map();
    this.totals = [0, 0, 0];
  }

  get size() {
    return this.frames.length - this.head;
  }

  get eof() {
    return this.explicitlyClosed || (this.mode === 'pipe' && this.totals[WRITER] === 0);
  }

  get broken() {
    return this.explicitlyClosed || (this.mode === 'pipe' && this.totals[READER] === 0);
  }

  pop() {
    const frame = this.frames[this.head];
    this.frames[this.head++] = undefined;
    if (this.head >= 64 && this.head * 2 >= this.frames.length) {
      this.frames = this.frames.slice(this.head);
      this.head = 0;
    }
    return frame;
  }
}

function settle(pending, waiter, result) {
  pending.delete(waiter);
  clearTimeout(waiter.timer);
  waiter.resolve(result);
}

function settleAll(pending, result) {
  for (const waiter of pending) settle(pending, waiter, result);
}

function cancelClientWaiters(pending, client) {
  for (const waiter of pending) {
    if (waiter.client === client) settle(pending, waiter, INVALID_CHANNEL);
  }
}

function awaitChannel(pending, client, bytes, seconds, timeoutResult) {
  if (seconds === 0) return timeoutResult;
  return new Promise(resolve => {
    const waiter = {client, bytes, resolve, timer: undefined};
    pending.add(waiter);
    const deadline = performance.now() + (seconds < 0 ? Infinity : seconds * 1000);
    if (deadline === Infinity) return;
    const schedule = () => {
      const remaining = deadline - performance.now();
      if (remaining <= 0) {
        settle(pending, waiter, timeoutResult);
      } else {
        waiter.timer = setTimeout(schedule, Math.min(remaining, MAX_TIMER_MS));
      }
    };
    schedule();
  });
}

function advance(channel) {
  if (channel.mode === 'pipe' && channel.totals[READER] === 0) {
    channel.frames = [];
    channel.head = 0;
  }
  if (channel.broken) settleAll(channel.putters, BROKEN_PIPE);
  if (channel.eof) settleAll(channel.pollers, 1);

  while (true) {
    if (channel.size > 0) {
      settleAll(channel.pollers, 1);
      const reader = channel.getters.values().next().value;
      if (reader) {
        settle(channel.getters, reader, channel.pop());
        continue;
      }
    }
    const writer = channel.putters.values().next().value;
    if (!writer || channel.size >= channel.maxsize) break;
    channel.frames.push(writer.bytes);
    settle(channel.putters, writer, 1);
  }
  if (channel.eof && channel.size === 0) settleAll(channel.getters, EOF);
}

function grant(channel, client, role) {
  let leases = channel.leases.get(client);
  if (!leases) {
    leases = [0, 0, 0];
    channel.leases.set(client, leases);
  }
  ++leases[role];
}

function canUse(channel, client, role) {
  if (!channel.mode) return true;
  const leases = channel.leases.get(client);
  return Boolean(leases && (leases[role] || leases[QUEUE]));
}

export function createPythonChannels() {
  const channels = new Map();
  const tickets = new Map();
  let nextId = 1;

  function update(channel, id) {
    advance(channel);
    if (channel.mode && !channel.totals.some(Boolean)) {
      channels.delete(id);
    }
  }

  function request(client, operation, id, argument, bytes, seconds = 0) {
    const op = typeof operation === 'number' ? operations[operation] : operation;
    if (op === 'create') {
      if (!Number.isInteger(argument) || argument < 0 || argument > MAX_ID) return INVALID_ARGUMENT;
      if (nextId > MAX_ID) return INVALID_CHANNEL;
      const handle = nextId++;
      channels.set(handle, new Channel(argument));
      return handle;
    }
    if (op === 'import' || op === 'cancel') {
      const ticket = tickets.get(id);
      if (!ticket || (op === 'cancel' && client !== ticket.client)) return INVALID_CHANNEL;
      tickets.delete(id);
      const {channel, handle, role} = ticket;
      if (op === 'import') {
        grant(channel, client, role);
        return handle;
      }
      --channel.totals[role];
      update(channel, handle);
      return 0;
    }

    const channel = channels.get(id);
    if (!channel) return INVALID_CHANNEL;
    if ((op === 'put' || op === 'get' || op === 'poll') &&
        !(seconds === -1 || seconds >= 0)) {
      return INVALID_TIMEOUT;
    }
    switch (op) {
      case 'retain': {
        const role = argument;
        if (role !== READER && role !== WRITER && role !== QUEUE) return INVALID_ARGUMENT;
        const mode = role === QUEUE ? 'queue' : 'pipe';
        if (channel.mode && channel.mode !== mode) return INVALID_CHANNEL;
        channel.mode = mode;
        grant(channel, client, role);
        ++channel.totals[role];
        return 0;
      }
      case 'release': {
        const role = argument;
        const leases = channel.leases.get(client);
        if (!leases?.[role]) return INVALID_CHANNEL;
        --leases[role];
        --channel.totals[role];
        if (!leases.some(Boolean)) channel.leases.delete(client);
        if (!canUse(channel, client, READER)) {
          cancelClientWaiters(channel.getters, client);
          cancelClientWaiters(channel.pollers, client);
        }
        if (!canUse(channel, client, WRITER)) cancelClientWaiters(channel.putters, client);
        update(channel, id);
        return 0;
      }
      case 'export': {
        const role = argument;
        if (!channel.leases.get(client)?.[role] || nextId > MAX_ID) return INVALID_CHANNEL;
        const token = nextId++;
        tickets.set(token, {client, channel, handle: id, role});
        ++channel.totals[role];
        return token;
      }
      case 'close':
        channel.explicitlyClosed = true;
        update(channel, id);
        return 0;
      case 'put':
        if (!canUse(channel, client, WRITER)) return INVALID_CHANNEL;
        if (channel.broken) return BROKEN_PIPE;
        if (channel.size >= channel.maxsize) {
          return awaitChannel(channel.putters, client, bytes, seconds, TIMEOUT);
        }
        channel.frames.push(bytes);
        advance(channel);
        return 1;
      case 'get':
        if (!canUse(channel, client, READER)) return INVALID_CHANNEL;
        if (channel.size > 0) {
          const result = channel.pop();
          advance(channel);
          return result;
        }
        if (channel.eof) return EOF;
        return awaitChannel(channel.getters, client, undefined, seconds, TIMEOUT);
      case 'poll':
        if (!canUse(channel, client, READER)) return INVALID_CHANNEL;
        if (channel.size > 0 || channel.eof) return 1;
        return awaitChannel(channel.pollers, client, undefined, seconds, 0);
      case 'qsize':
        if (!canUse(channel, client, READER)) return INVALID_CHANNEL;
        return channel.size;
      default:
        return INVALID_CHANNEL;
    }
  }

  function disconnect(client) {
    for (const [id, channel] of channels) {
      cancelClientWaiters(channel.getters, client);
      cancelClientWaiters(channel.pollers, client);
      cancelClientWaiters(channel.putters, client);
      const leases = channel.leases.get(client);
      if (leases) {
        for (const role of [READER, WRITER, QUEUE]) channel.totals[role] -= leases[role];
        channel.leases.delete(client);
        update(channel, id);
      }
    }
  }

  return {request, disconnect};
}
