"""Stage and deploy an existing Blender browser package using Wrangler."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from urllib.request import urlopen


PORT = Path(__file__).resolve().parents[1]
CLOUDFLARE = PORT / "cloudflare"
BUCKET = "blender-rocks-releases"
WRANGLER = ["npx", "--yes", "wrangler@4.160.0"]
R2_FILES = {
    "blender.wasm": ("application/wasm", "-6"),
    "blender-assets.data": ("application/octet-stream", "-3"),
}
STATIC_LIMIT = 25 * 1024**2
WRANGLER_R2_LIMIT = 300 * 1024**2


def source_url(repository, verify):
    match = re.fullmatch(r"https://github\.com/([\w.-]+/[\w.-]+?)(?:\.git)?/?", repository)
    if not match:
        raise ValueError("origin must point to a public GitHub repository over HTTPS")
    path = match[1]
    commit = subprocess.check_output(
        ["git", "-C", PORT, "rev-parse", "HEAD"], text=True
    ).strip()
    if verify:
        raw_readme = f"https://raw.githubusercontent.com/{path}/{commit}/README.md"
        try:
            with urlopen(raw_readme, timeout=15) as response:
                response.read(1)
        except OSError as error:
            raise ValueError(f"GitHub source commit {commit} must be publicly accessible") from error
    return f"https://github.com/{path}/tree/{commit}"


def stage(package, output, source):
    for name in ("index.html", "blender.js", "blender.wasm", "blender-assets.data",
                 "blender-cycles-webgpu.data", "oidn-webgpu.mjs",
                 "oidn-weights/rt_hdr_calb_cnrm_large.tza"):
        if not (package / name).is_file():
            raise ValueError(f"missing browser package asset: {name}")

    public = output / "public"
    shutil.copytree(package, public, ignore=shutil.ignore_patterns(*R2_FILES))
    shutil.copyfile(PORT / "favicon.png", public / "favicon.png")
    shutil.copyfile(PORT / "LICENSE", public / "license.txt")
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
        "  #source { font-size: 0.75rem; margin: 1rem auto 0; }\n"
        "  #source a { color: #ddd; }\n</style>",
    ).replace(
        indicator,
        f'{indicator}\n  <p id="source"><a href="{source}">Source code</a> · '
        '<a href="./license.txt">License</a></p>',
    )
    launcher.write_text(html)
    (public / "_headers").write_text(
        "/*\n"
        "  Cross-Origin-Opener-Policy: same-origin\n"
        "  Cross-Origin-Embedder-Policy: require-corp\n"
        "  Cross-Origin-Resource-Policy: same-origin\n"
        "/\n"
        f'  Link: <{source}>; rel="alternate"; title="Source code"\n'
    )
    for path in public.rglob("*"):
        if path.is_file() and path.stat().st_size > STATIC_LIMIT:
            raise ValueError(f"static asset exceeds Cloudflare's 25 MiB limit: {path}")

    runtime = output / "runtime"
    runtime.mkdir()
    objects = {}
    for name, (_, level) in R2_FILES.items():
        path = runtime / name
        with path.open("wb") as compressed:
            subprocess.run(
                ["gzip", "-n", level, "-c", package / name], stdout=compressed, check=True
            )
        if path.stat().st_size > WRANGLER_R2_LIMIT:
            raise ValueError(f"object exceeds Wrangler's 300 MiB single-upload limit: {path}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        objects[f"/{name}"] = f"{digest}/{name}"
    (output / "objects.json").write_text(json.dumps(objects, indent=2) + "\n")
    return objects


def deploy(objects, output):
    for name, (content_type, _) in R2_FILES.items():
        subprocess.run(
            [*WRANGLER, "r2", "object", "put", f"{BUCKET}/{objects[f'/{name}']}",
             "--file", output / "runtime" / name, "--content-type", content_type,
             "--content-encoding", "gzip", "--remote"],
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
        origin = subprocess.check_output(
            ["git", "-C", PORT, "remote", "get-url", "origin"], text=True
        ).strip()
        source = source_url(origin, verify=not args.stage_only)
        output = CLOUDFLARE / "dist"
        with tempfile.TemporaryDirectory(dir=CLOUDFLARE, prefix=".stage-") as directory:
            staged = Path(directory) / "dist"
            objects = stage(args.browser_package.resolve(), staged, source)
            if output.is_dir():
                shutil.rmtree(output)
            staged.rename(output)
        if not args.stage_only:
            deploy(objects, output)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")
