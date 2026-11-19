// SPDX-License-Identifier: GPL-2.0-or-later

const CREATE = 0;
const ATTACH = 1;
const RELEASE = 2;
const UNLINK = 3;

/** Own named segments and mapping leases in the common Wasm linear memory. */
export function createSharedMemoryRegistry(allocateZeroed, free) {
  const names = new Map();
  const owners = new Map();
  let nextLease = 1;

  function releaseSegment(segment) {
    if (!segment.linked && segment.references === 0) {
      free(segment.pointer);
    }
  }

  function disconnect(owner) {
    const leases = owners.get(owner);
    if (!leases) return;
    owners.delete(owner);
    for (const segment of leases.values()) {
      segment.references--;
      releaseSegment(segment);
    }
  }

  function request(owner, operation, name, size, _track, lease) {
    if (operation === RELEASE) {
      const leases = owners.get(owner);
      const segment = leases?.get(lease);
      if (!segment) return {status: -4};
      leases.delete(lease);
      if (leases.size === 0) owners.delete(owner);
      segment.references--;
      releaseSegment(segment);
      return {status: 0};
    }

    if (typeof name !== "string" || name.length === 0 || name.includes("\0")) {
      return {status: -5};
    }
    let segment = names.get(name);
    if (operation === UNLINK) {
      if (!segment) return {status: -2};
      names.delete(name);
      segment.linked = false;
      releaseSegment(segment);
      return {status: 0};
    }

    if (nextLease > 0xFFFFFFFF) return {status: -7};
    if (operation === CREATE) {
      if (segment) return {status: -3};
      if (!Number.isSafeInteger(size) || size < 1) return {status: -5};
      let pointer;
      try {
        pointer = allocateZeroed(BigInt(size));
      } catch {
        return {status: -7};
      }
      if (typeof pointer !== "bigint" || pointer <= 0n) return {status: -7};
      segment = {pointer, size, linked: true, references: 0};
      names.set(name, segment);
    } else if (operation === ATTACH) {
      if (!segment) return {status: -2};
    } else {
      return {status: -4};
    }

    const id = nextLease++;
    let leases = owners.get(owner);
    if (!leases) owners.set(owner, leases = new Map());
    leases.set(id, segment);
    segment.references++;
    return {status: 0, pointer: segment.pointer, size: segment.size, lease: id};
  }

  return {request, disconnect};
}
