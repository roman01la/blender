"use strict";

// Loaded inside the editor worker. MEMFS exposes normal paths to Blender's importers
// and exporters. Host file contents are read only when Blender requests them.
self.BlenderLocalFiles = class {
  constructor(fs, writeFile, requestWrite) {
    this.fs = fs;
    this.writeFile = writeFile;
    this.requestWrite = requestWrite;
    this.mounts = new Map();
    this.files = new WeakMap();
    this.exports = new Map();
    this.reader = new FileReaderSync();
  }

  watch(node, hostFile = null) {
    let state = this.files.get(node);
    if (state) return state;
    state = { source: hostFile, revision: 0, page: null, pageStart: 0 };
    this.files.set(node, state);
    const streamOps = node.stream_ops;
    const nodeOps = node.node_ops;
    const read = (start, end) => {
      try {
        return new Uint8Array(this.reader.readAsArrayBuffer(state.source.slice(start, end)));
      } catch {
        throw new this.fs.ErrnoError(29); // EIO
      }
    };
    const materialize = (size = node.usedBytes) => {
      if (state.source) {
        node.contents = read(0, Math.min(size, node.usedBytes));
        node.usedBytes = node.contents.length;
        state.source = state.page = null;
      }
    };
    node.node_ops = {
      ...nodeOps,
      setattr(node, attributes) {
        if (attributes.size != null) {
          materialize(attributes.size);
          state.revision++;
        }
        return nodeOps.setattr(node, attributes);
      },
    };
    node.stream_ops = {
      ...streamOps,
      read(stream, output, offset, length, position) {
        if (!state.source) return streamOps.read(stream, output, offset, length, position);
        const count = Math.min(length, node.usedBytes - position);
        if (count <= 0) return 0;
        const pageStart = Math.floor(position / (1024 * 1024)) * (1024 * 1024);
        const end = position + count;
        if (end <= pageStart + 1024 * 1024) {
          if (!state.page || state.pageStart !== pageStart) {
            state.page = read(pageStart, pageStart + 1024 * 1024);
            state.pageStart = pageStart;
          }
          output.set(state.page.subarray(position - pageStart, end - pageStart), offset);
        } else {
          output.set(read(position, end), offset);
        }
        return count;
      },
      write(...args) {
        materialize();
        state.revision++;
        return streamOps.write(...args);
      },
      mmap(...args) {
        materialize();
        return streamOps.mmap(...args);
      },
      msync(...args) {
        materialize();
        state.revision++;
        return streamOps.msync(...args);
      },
    };
    return state;
  }

  addFile(path, file) {
    const node = this.fs.create(path, 0o666);
    node.usedBytes = file.size;
    node.mtime = file.lastModified;
    this.watch(node, file);
  }

  async mount(handle, writable) {
    const root = "/home/web_user/Local Folders/" + crypto.randomUUID();
    if (Array.isArray(handle)) {
      if (writable) throw new Error("Choose a folder for write access.");
      const names = new Set(handle.map(file => file.name));
      if (names.size !== handle.length ||
          [...names].some(name => !name || name === "." || name === ".." || /[/\\]/.test(name))) {
        throw new Error("The selected files must have distinct, valid names.");
      }
      const path = root + "/Selected Files";
      this.fs.mkdirTree(path);
      for (const file of handle) this.addFile(path + "/" + file.name, file);
      return path;
    }
    this.fs.mkdirTree(root);
    const entries = new Map();
    const populate = async (directory, path) => {
      for await (const entry of directory.values()) {
        const child = path + "/" + entry.name;
        if (entry.kind === "directory") {
          this.fs.mkdir(child);
          await populate(entry, child);
        } else {
          const file = await entry.getFile();
          this.addFile(child, file);
          entries.set(child, file);
        }
      }
    };
    const path = root + "/" + handle.name;
    this.fs.mkdir(path);
    await populate(handle, path);
    this.mounts.set(path, { root: path, handle, writable, entries });
    return path;
  }

  versions(path, versions = new Map()) {
    for (const name of this.fs.readdir(path)) {
      if (name === "." || name === "..") continue;
      const child = path + "/" + name;
      const { node } = this.fs.lookupPath(child, { follow: false });
      if (this.fs.isDir(node.mode)) this.versions(child, versions);
      else if (this.fs.isFile(node.mode)) {
        versions.set(child, { node, revision: this.watch(node).revision });
      }
    }
    return versions;
  }

  findMount(path) {
    return [...this.mounts.values()].find(
      volume => path === volume.root || path.startsWith(volume.root + "/"));
  }

  beginExport(path) {
    const mount = this.findMount(path);
    if (!mount) return 0;
    if (!mount.writable) return -1;
    this.exports.set(path, { mount, before: this.versions(mount.root) });
    return 1;
  }

  async finishExport(path, success) {
    const transaction = this.exports.get(path);
    if (!transaction) return;
    this.exports.delete(path);
    if (!success) return;
    const { mount, before } = transaction;
    for (const [filePath, current] of this.versions(mount.root)) {
      const previous = before.get(filePath);
      if (previous?.node === current.node && previous.revision === current.revision) continue;
      const relative = filePath.slice(mount.root.length + 1).split("/");
      const name = relative.pop();
      let directory = mount.handle;
      for (const part of relative) directory = await directory.getDirectoryHandle(part, { create: true });

      const expected = mount.entries.get(filePath);
      let handle;
      try {
        handle = await directory.getFileHandle(name);
      } catch (error) {
        if (error.name !== "NotFoundError") throw error;
      }
      if (handle) {
        const existing = await handle.getFile();
        if (!expected || existing.lastModified !== expected.lastModified || existing.size !== expected.size) {
          throw new Error("The local file changed since the folder was selected: " + name);
        }
      } else {
        if (expected) throw new Error("The local file was moved or deleted: " + name);
        handle = await directory.getFileHandle(name, { create: true });
      }
      const fs = this.fs;
      function* chunks() {
        const stream = fs.open(filePath, "r");
        try {
          const bytes = new Uint8Array(256 * 1024);
          let size;
          while ((size = fs.read(stream, bytes, 0, bytes.length)) > 0) {
            yield bytes.subarray(0, size);
          }
        } finally {
          fs.close(stream);
        }
      }
      await this.writeFile(handle, chunks());
      mount.entries.set(filePath, await handle.getFile());
    }
  }

  async moveToTrash(path) {
    const name = path.slice(path.lastIndexOf("/") + 1);
    const trash = "/home/web_user/.Trash";
    if (!name || name === "." || name === ".." || path === trash || path.startsWith(trash + "/")) {
      throw new Error("This path cannot be moved to Trash.");
    }
    const mount = this.findMount(path);
    if (mount) await this.trashHostItem(mount, path, name);

    const destination = trash + "/" + crypto.randomUUID();
    this.fs.mkdirTree(destination);
    this.fs.rename(path, destination + "/" + name);
  }

  async trashHostItem(mount, path, name) {
    const trashName = ".blender-trash";
    const relative = path.slice(mount.root.length + 1).split("/");
    if (path === mount.root || relative[0] === trashName || relative.includes("..")) {
      throw new Error("This location cannot be moved to Trash.");
    }
    relative.pop();
    let sourceParent = mount.handle;
    let item;
    try {
      for (const part of relative) sourceParent = await sourceParent.getDirectoryHandle(part);
      const isDirectory = this.fs.isDir(this.fs.lstat(path).mode);
      item = isDirectory ? await sourceParent.getDirectoryHandle(name) :
                           await sourceParent.getFileHandle(name);
    } catch (error) {
      if (error.name !== "NotFoundError") throw error;
      // A session-only item has no host file to move.
      return;
    }
    if (!await this.requestWrite(mount.handle)) {
      throw new Error("Write access to the local folder was not granted.");
    }

    const trash = await mount.handle.getDirectoryHandle(trashName, { create: true });
    let destination = await trash.getDirectoryHandle(crypto.randomUUID(), { create: true });
    for (const part of relative) {
      destination = await destination.getDirectoryHandle(part, { create: true });
    }
    const verify = [];
    const copy = async (source, directory) => {
      if (source.kind === "directory") {
        const folder = await directory.getDirectoryHandle(source.name, { create: true });
        const names = new Set();
        for await (const entry of source.values()) {
          names.add(entry.name);
          await copy(entry, folder);
        }
        verify.push(async () => {
          let count = 0;
          for await (const name of source.keys()) {
            if (!names.has(name)) return false;
            count++;
          }
          return count === names.size;
        });
      } else {
        const original = await source.getFile();
        const target = await directory.getFileHandle(source.name, { create: true });
        await this.writeFile(target, original.stream());
        verify.push(async () => {
          const current = await source.getFile();
          return current.size === original.size && current.lastModified === original.lastModified;
        });
      }
    };
    await copy(item, destination);
    for (const unchanged of verify) {
      if (!await unchanged()) throw new Error("The local item changed while moving it to Trash.");
    }
    await sourceParent.removeEntry(name, { recursive: item.kind === "directory" });
    for (const entry of mount.entries.keys()) {
      if (entry === path || entry.startsWith(path + "/")) mount.entries.delete(entry);
    }
  }
};
