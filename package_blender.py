import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


def pack_preloads(file_packager: Path, bundle: Path, basename: str,
                  files: list[tuple[Path, str]]) -> None:
    subprocess.run(
        [str(file_packager), f"{basename}.data", "--preload",
         *(f"{str(path).replace('@', '@@')}@{destination}" for path, destination in files),
         f"--js-output={basename}.js", "--no-node", "--no-force"],
        cwd=bundle,
        check=True,
    )


def package(installation: Path, output: Path, python_process: Path | None = None,
            cycles_webgpu: Path | None = None) -> None:
    if output.exists() or output.is_symlink():
        raise ValueError(f"output already exists: {output}")
    if output.is_relative_to(installation):
        raise ValueError("put the browser package outside the Blender installation")

    binary_dir = installation
    resources_parent = installation
    if not (binary_dir / "blender.js").is_file():
        binary_dir = installation / "bin"
        resources_parent = installation / "share" / "blender"
    binaries = [binary_dir / name for name in ("blender.js", "blender.wasm")]
    for path in binaries:
        if not path.is_file():
            raise ValueError(f"missing installed file: {path}")
    worker = binary_dir / "blender.worker.js"
    if worker.is_file():
        binaries.append(worker)
    translator = binary_dir / "blender-shader-translator.wasm"
    has_translator = translator.is_file()
    if has_translator:
        binaries.append(translator)

    versions = [
        path for path in resources_parent.glob("*.*")
        if path.is_dir() and re.fullmatch(r"\d+\.\d+", path.name)
    ]
    if len(versions) != 1:
        raise ValueError(f"expected one versioned resource directory under {resources_parent}")
    resources = versions[0]

    port = Path(__file__).resolve().parent
    launcher = (port / "editor.html").read_text(encoding="utf-8")
    cycle_preloads = []
    if cycles_webgpu is not None:
        shaders = [cycles_webgpu.resolve() / name for name in (
            "stock-camera.wgsl", "bvh2-compact.wgsl", "surface-compact.wgsl",
            "svm-compact.wgsl", "wavefront-light.wgsl", "path-update.wgsl",
        )]
        shaders.extend(port / "cycles-webgpu" / name for name in (
            "film.wgsl", "path-state.wgsl", "wavefront-queue.wgsl",
        ))
        background = cycles_webgpu.resolve() / "shade-background.wgsl"
        if background.exists():
            shaders.append(background)
        reflection_light = cycles_webgpu.resolve() / "wavefront-light-reflection.wgsl"
        reflection_update = cycles_webgpu.resolve() / "path-update-reflection.wgsl"
        multi_point = cycles_webgpu.resolve() / "wavefront-light-deferred-multi-point-reflection.wgsl"
        light_tree = [cycles_webgpu.resolve() / name for name in (
            "wavefront-light-deferred-reflection-tree.wgsl", "shade-background-tree.wgsl",
        )]
        want_light_tree = any(shader.exists() for shader in light_tree)
        if reflection_light.exists() or (reflection_update.exists() and not
                                         (multi_point.exists() or want_light_tree)):
            shaders.append(reflection_light)
        if reflection_light.exists() or reflection_update.exists() or multi_point.exists() or want_light_tree:
            shaders.append(reflection_update)
        deferred_lighting = [cycles_webgpu.resolve() / name for name in (
            "wavefront-light-deferred.wgsl", "wavefront-light-deferred-reflection.wgsl",
        )] + [multi_point]
        if want_light_tree:
            shaders.extend(light_tree)
        if want_light_tree or any(shader.exists() for shader in deferred_lighting):
            shaders.append(cycles_webgpu.resolve() / "intersect-shadow.wgsl")
            shaders.extend(shader for shader in deferred_lighting if shader.exists())
        adaptive_shaders = [cycles_webgpu.resolve() / name for name in (
            "adaptive-camera-slots.wgsl", "adaptive-camera-rays-retire.wgsl",
            "adaptive-check.wgsl", "adaptive-filter-x.wgsl", "adaptive-filter-y.wgsl",
        )]
        if any(shader.exists() for shader in adaptive_shaders):
            shaders.extend(adaptive_shaders)
            shaders.append(port / "cycles-webgpu" / "adaptive-active-count.wgsl")
        denoise_surface = cycles_webgpu.resolve() / "denoise-surface.wgsl"
        if denoise_surface.exists():
            shaders.append(denoise_surface)
            if background.exists():
                shaders.append(cycles_webgpu.resolve() / "shade-background-guides.wgsl")
            if want_light_tree:
                shaders.append(cycles_webgpu.resolve() / "shade-background-tree-guides.wgsl")
        for shader in shaders:
            if not shader.is_file() or not shader.stat().st_size:
                raise ValueError(f"missing or empty Cycles WebGPU shader: {shader}")
            cycle_preloads.append((shader, f"/cycles-webgpu/{shader.name}"))

    oidn_assets = None
    if cycles_webgpu is not None:
        oidn_runtime = cycles_webgpu / "oidn-webgpu.mjs"
        if oidn_runtime.is_file():
            oidn_license = cycles_webgpu / "oidn-webgpu-LICENSE.txt"
            oidn_weights = binary_dir / "oidn-weights"
            for required in (oidn_license, oidn_weights / "LICENSE.txt",
                             oidn_weights / "rt_alb_large.tza", oidn_weights / "rt_nrm_large.tza",
                             oidn_weights / "rt_hdr_calb_cnrm_large.tza"):
                if not required.is_file() or not required.stat().st_size:
                    raise ValueError(f"missing or empty WebGPU OIDN asset: {required}")
            if not oidn_runtime.stat().st_size:
                raise ValueError(f"empty WebGPU OIDN runtime: {oidn_runtime}")
            oidn_assets = (oidn_runtime, oidn_license, oidn_weights)

    sdk = Path(subprocess.check_output(["em-config", "EMSCRIPTEN_ROOT"], text=True).strip())
    file_packager = sdk / "tools" / "file_packager"

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        bundle = Path(temporary) / "browser"
        bundle.mkdir()
        for binary in binaries:
            shutil.copyfile(binary, bundle / binary.name)
        for name in ("editor.mjs", "editor-worker.js", "editor-local-files.js", "favicon.png"):
            shutil.copyfile(port / name, bundle / name)
        if python_process is not None:
            process_dir = bundle / "python-process"
            process_dir.mkdir()
            for name in ("client.mjs", "worker.js", "blender-python-child.js",
                         "blender-python-child.wasm", "blender-python-child.data"):
                source = python_process / name
                if not source.is_file():
                    raise ValueError(f"missing Python worker artifact: {source}")
                shutil.copyfile(source, process_dir / name)
            shared_artifacts = [python_process / f"blender-python-child-shared.{suffix}"
                                for suffix in ("js", "wasm")]
            if any(path.exists() for path in shared_artifacts):
                for path in shared_artifacts:
                    if not path.is_file():
                        raise ValueError(f"missing shared Python worker artifact: {path}")
                    shutil.copyfile(path, process_dir / path.name)
            for name in ("bridge.mjs", "channels.mjs", "shared-memory.mjs"):
                shutil.copyfile(port / "python" / name, process_dir / name)
        # Both install layouts use Blender's portable lookup in the browser.
        if binary_dir == installation:
            side_modules = (installation / "lib").glob("*.so")
        else:
            addon_dir = f"*/blender/{resources.name}/scripts/addons_core/io_scene_gltf2"
            side_modules = installation.glob(f"{addon_dir}/*.so")
        preloads = [(resources, f"/{resources.name}")]
        preloads.extend((path, f"/lib/{path.name}") for path in sorted(side_modules))
        if python_process is not None:
            stdlibs = [
                path for path in (resources / "python").glob("lib*/python3.*")
                if (path / "multiprocessing/__init__.py").is_file()
            ]
            if len(stdlibs) != 1:
                raise ValueError("expected one embedded Python standard library for browser spawn")
            destination = f"/{resources.name}/{stdlibs[0].relative_to(resources)}"
            preloads.extend((module, f"{destination}/{module.name}")
                            for module in sorted((port / "python").glob("_blender_web_*.py")))
        pack_preloads(file_packager, bundle, "blender-assets", preloads)
        preload_scripts = sorted(bundle.glob("blender-assets*.js"))
        if not preload_scripts:
            raise ValueError(f"no files to package under {resources}")
        assets = {
            "preloads": [path.name for path in preload_scripts],
            "translator": translator.name if has_translator else None,
            "pythonProcess": "./python-process/client.mjs" if python_process is not None else None,
        }
        if oidn_assets is not None:
            runtime, license_file, weights = oidn_assets
            shutil.copyfile(runtime, bundle / runtime.name)
            shutil.copyfile(license_file, bundle / license_file.name)
            shutil.copyfile(port / "oidn-webgpu/launcher.mjs", bundle / "oidn-webgpu-launcher.mjs")
            shutil.copyfile(port / "oidn-webgpu/film.mjs", bundle / "oidn-webgpu-film.mjs")
            shutil.copytree(weights, bundle / weights.name)
            marker = Path(temporary) / "oidn-webgpu.enabled"
            marker.write_text("WebGPU OIDN\n", encoding="utf-8")
            cycle_preloads.append((marker, "/cycles-webgpu/oidn-webgpu.enabled"))
        if cycle_preloads:
            shutil.copyfile(port / "cycles-webgpu/render-preview.mjs",
                            bundle / "cycles-webgpu-render-preview.mjs")
            pack_preloads(file_packager, bundle, "blender-cycles-webgpu", cycle_preloads)
            assets["cyclesWebGPU"] = "blender-cycles-webgpu.js"
        (bundle / "index.html").write_text(
            launcher.replace("/* editor assets */ null", json.dumps(assets)), encoding="utf-8"
        )
        bundle.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package an installed Wasm Blender build for the browser.")
    parser.add_argument("installation", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--python-process", type=Path,
                        help="include the Python worker runtime built by web/python-process/build.py")
    parser.add_argument("--cycles-webgpu", type=Path,
                        help="include Cycles WGSL shaders from web/cycles-webgpu/build.py")
    args = parser.parse_args()
    try:
        output = args.output.parent.resolve() / args.output.name
        package(args.installation.resolve(), output, args.python_process, args.cycles_webgpu)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    except subprocess.CalledProcessError as error:
        parser.exit(error.returncode)
