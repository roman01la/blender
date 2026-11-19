import argparse
import fcntl
from functools import partial
from hashlib import sha256
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
from tempfile import gettempdir


class BrowserBuildHandler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".mjs": "text/javascript"}

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve a local Blender web build.")
    parser.add_argument("build_directory", type=Path)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if not (args.build_directory / "index.html").is_file():
        parser.error("the build directory must contain index.html")

    project = Path(__file__).resolve().parent.parent
    project_id = sha256(os.fsencode(project)).hexdigest()[:16]
    lock_path = Path(gettempdir()) / f"blender-web-{os.getuid()}-{project_id}.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.seek(0)
            url = lock.read().strip()
            detail = f" at {url}" if url else ""
            parser.error(f"this project already has a Blender server{detail}")

        handler = partial(BrowserBuildHandler, directory=str(args.build_directory))
        with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
            url = f"http://127.0.0.1:{server.server_port}/"
            lock.seek(0)
            lock.truncate()
            lock.write(url)
            lock.flush()
            print(url, flush=True)
            server.serve_forever()
