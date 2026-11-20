"use strict";

self.BlenderRecentFiles = class {
  #database;
  #files = new Map();
  #available = new Map();

  async load() {
    try {
      this.#database = await new Promise((resolve, reject) => {
        const request = indexedDB.open("blender-web-projects", 1);
        request.onupgradeneeded = () => request.result.createObjectStore("history");
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
      });
      this.#database.onversionchange = () => {
        this.#database.close();
        this.#database = null;
      };
      const entries = await new Promise((resolve, reject) => {
        const request = this.#database.transaction("history").objectStore("history").get("recent");
        request.onsuccess = () => resolve(request.result ?? []);
        request.onerror = () => reject(request.error);
      });
      for (const { path, handle, file } of entries) {
        if (/^\/home\/web_user\/Browser Files\/[a-f0-9-]{36}\/[^/\\\0\r\n]+$/.test(path) &&
            (handle || file)) {
          this.#files.set(path, { path, handle, file });
        }
      }
      await Promise.all([...this.#files.values()].map(async ({ path, handle, file }) => {
        try {
          if (handle && await handle.queryPermission({ mode: "read" }) === "granted") {
            file = await handle.getFile();
          }
          if (file) this.#available.set(path, file);
        } catch (error) {
          if (error.name === "NotFoundError") this.#files.delete(path);
        }
      }));
    } catch (error) {
      console.warn("Could not load Blender's recent files:", error);
    }
  }

  restore(fs, historyPath, localFiles) {
    if (!this.#available) return;
    for (const [path, file] of this.#available) {
      fs.mkdirTree(path.slice(0, path.lastIndexOf("/")));
      localFiles.addFile(path, file);
    }
    this.#available = null;
    if (this.#files.size) fs.writeFile(historyPath, [...this.#files.keys()].join("\n") + "\n");
  }

  get(path) {
    return this.#files.get(path);
  }

  forget(path) {
    this.#files.delete(path);
  }

  remember(path, handle, file) {
    this.#files.set(path, { path, handle, file: handle ? null : file });
  }

  async pathForHandle(handle) {
    if (!handle) return null;
    for (const entry of this.#files.values()) {
      if (entry.handle && entry.handle.name === handle.name && await handle.isSameEntry(entry.handle)) {
        return entry.path;
      }
    }
    return null;
  }

  async save(history, opening) {
    const entries = history.split("\n").flatMap(path => {
      if (opening?.path === path) {
        return [{ path, handle: opening.handle, file: opening.handle ? null : opening.file }];
      }
      const entry = this.#files.get(path);
      return entry ? [entry] : [];
    });
    this.#files = new Map(entries.map(entry => [entry.path, entry]));
    if (!this.#database) return;
    try {
      await new Promise((resolve, reject) => {
        const transaction = this.#database.transaction("history", "readwrite");
        transaction.objectStore("history").put(entries, "recent");
        transaction.oncomplete = resolve;
        transaction.onabort = () => reject(transaction.error);
      });
    } catch (error) {
      console.warn("Could not save Blender's recent files:", error);
    }
  }
};
