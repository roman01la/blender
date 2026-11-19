import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def stage(source: Path, destination: Path, port: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"staging directory already exists: {destination}")
    if destination.is_relative_to(source):
        raise ValueError("stage Blender outside the upstream source tree")

    patch = port / "blender-platform" / "emscripten.patch"
    subprocess.run(["git", "-C", source, "apply", "--check", patch], check=True)
    lfs_storage = subprocess.check_output(
        ["git", "-C", source, "rev-parse", "--path-format=absolute", "--git-path", "lfs"],
        text=True,
    ).strip()
    files = {
        "platform_emscripten.cmake": "build_files/cmake/platform",
        "GHOST_ContextWebGPU.cc": "intern/ghost/intern",
        "GHOST_ContextWebGPU.hh": "intern/ghost/intern",
        "GHOST_SystemWeb.cc": "intern/ghost/intern",
        "GHOST_SystemWeb.hh": "intern/ghost/intern",
        "GHOST_WebGPUCanvas.cc": "intern/ghost/intern",
        "GHOST_WebGPUCanvas.hh": "intern/ghost/intern",
        "bpy_web_process.hh": "source/blender/python/intern",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        tree = Path(temporary) / "blender"
        subprocess.run(
            ["git", "clone", "--quiet", "--shared", "--", source, tree],
            check=True,
            env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"},
        )
        subprocess.run(
            ["git", "-C", tree, "-c", f"lfs.storage={lfs_storage}", "lfs", "checkout"],
            check=True,
        )
        subprocess.run(["git", "-C", tree, "apply", patch], check=True)
        defaults = port / "blender-platform" / "cycles-defaults.patch"
        subprocess.run(["git", "-C", tree, "apply", defaults], check=True)
        for name, directory in files.items():
            shutil.copyfile(port / "blender-platform" / name, tree / directory / name)
        for name in ("channel_api.h", "shared_memory_api.h"):
            shutil.copyfile(port / "python-process" / name,
                            tree / "source/blender/python/intern" / name)
        shutil.copytree(port / "blender-platform" / "webgpu", tree / "source/blender/gpu/webgpu")
        shutil.copytree(port / "blender-platform" / "cycles-webgpu",
                        tree / "intern/cycles/device/webgpu")
        tree.rename(destination)


if __name__ == "__main__":
    port = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Stage Blender with the browser platform overlay.")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        destination = args.destination.parent.resolve() / args.destination.name
        stage(port.parent / "blender", destination, port)
    except ValueError as error:
        parser.error(str(error))
    except subprocess.CalledProcessError as error:
        parser.exit(error.returncode)
