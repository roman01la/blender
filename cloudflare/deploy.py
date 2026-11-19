"""Stage and deploy an existing Blender browser package using Wrangler."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile


PORT = Path(__file__).resolve().parents[1]
CLOUDFLARE = PORT / "cloudflare"
UPSTREAM = "4831affec6ebefd54061c41d60ab26a0189def42"
BUCKET = "blender-rocks-releases"
WRANGLER = ["npx", "--yes", "wrangler@4.160.0"]
R2_FILES = {
    "blender.wasm": "application/wasm",
    "blender-assets.data": "application/octet-stream",
    "blender-source.tar.gz": "application/gzip",
}
STATIC_LIMIT = 25 * 1024**2
WRANGLER_R2_LIMIT = 300 * 1024**2


def add_git_archive(target, checkout, revision, prefix):
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    command = ["git", "-C", checkout, "archive", f"--prefix={prefix}/", revision]
    with subprocess.Popen(command, stdout=subprocess.PIPE, env=env) as process:
        with tarfile.open(fileobj=process.stdout, mode="r|") as source:
            for member in source:
                target.addfile(member, source.extractfile(member) if member.isfile() else None)
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, command)


def stage(package, output):
    for name in ("index.html", "blender.js", "blender.wasm", "blender-assets.data",
                 "blender-cycles-webgpu.data", "oidn-webgpu.mjs",
                 "oidn-weights/rt_hdr_calb_cnrm_large.tza"):
        if not (package / name).is_file():
            raise ValueError(f"missing browser package asset: {name}")

    public = output / "public"
    shutil.copytree(package, public, ignore=shutil.ignore_patterns(*R2_FILES))
    shutil.copyfile(PORT / "favicon.png", public / "favicon.png")
    launcher = public / "index.html"
    html = launcher.read_text()
    html, favicon_count = re.subn(
        r'<link rel="icon"[^>]*>', '<link rel="icon" type="image/png" href="./favicon.png">', html
    )
    indicator = '  <progress aria-label="Blender startup"></progress>'
    if favicon_count != 1 or html.count("</style>") != 1 or html.count(indicator) != 1:
        raise ValueError("browser package has an unexpected editor launcher")
    html = html.replace(
        "</style>",
        "  #source { color: #ddd; font-size: 0.75rem; margin: 1rem auto 0; }\n</style>",
    ).replace(indicator, indicator + '\n  <a id="source" href="/blender-source.tar.gz">Source code</a>')
    launcher.write_text(html)
    (public / "_headers").write_text(
        "/*\n"
        "  Cross-Origin-Opener-Policy: same-origin\n"
        "  Cross-Origin-Embedder-Policy: require-corp\n"
        "  Cross-Origin-Resource-Policy: same-origin\n"
    )
    for path in public.rglob("*"):
        if path.is_file() and path.stat().st_size > STATIC_LIMIT:
            raise ValueError(f"static asset exceeds Cloudflare's 25 MiB limit: {path}")

    with tarfile.open(output / "blender-source.tar.gz", "w:gz", compresslevel=4) as source:
        add_git_archive(source, PORT.parent / "blender", UPSTREAM, "blender")
        add_git_archive(source, PORT, "HEAD", "web")

    files = {
        name: (output if name == "blender-source.tar.gz" else package) / name
        for name in R2_FILES
    }
    objects = {}
    for name, path in files.items():
        if path.stat().st_size > WRANGLER_R2_LIMIT:
            raise ValueError(f"object exceeds Wrangler's 300 MiB single-upload limit: {path}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        objects[f"/{name}"] = f"{digest}/{name}"
    (output / "objects.json").write_text(json.dumps(objects, indent=2) + "\n")
    return files, objects


def deploy(files, objects):
    for name, path in files.items():
        subprocess.run(
            [*WRANGLER, "r2", "object", "put", f"{BUCKET}/{objects[f'/{name}']}",
             "--file", path, "--content-type", R2_FILES[name], "--remote"],
            cwd=CLOUDFLARE,
            check=True,
        )
    subprocess.run([*WRANGLER, "deploy"], cwd=CLOUDFLARE, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("browser_package", type=Path)
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()
    try:
        output = CLOUDFLARE / "dist"
        with tempfile.TemporaryDirectory(dir=CLOUDFLARE, prefix=".stage-") as directory:
            staged = Path(directory) / "dist"
            files, objects = stage(args.browser_package.resolve(), staged)
            if output.is_dir():
                shutil.rmtree(output)
            staged.rename(output)
        files["blender-source.tar.gz"] = output / "blender-source.tar.gz"
        if not args.stage_only:
            deploy(files, objects)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")
