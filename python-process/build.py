# SPDX-License-Identifier: GPL-2.0-or-later

import argparse
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile


def _replace_runtime_hook(code: str, target: str, replacement: str) -> str:
    if code.count(target) != 1:
        raise ValueError("unexpected Emscripten shared Python worker glue")
    return code.replace(target, replacement)


def _build_shared_worker(source: Path, sysroot: Path, output: Path, temporary: Path,
                         exception_flags: tuple[str, ...]) -> None:
    # Emscripten links its main module with absolute data addresses, even when
    # its object files are PIC. Reuse the generated runtime and link its exact
    # inputs as a PIE so the Worker can place statics in its assigned arena.
    name = "blender-python-child-shared"
    replay = temporary / "shared-worker-link.tar"
    libraries = ("libpython3.13.a", "libexpat.a", "libbz2.a", "libz.a")
    subprocess.run(
        [
            "emcc", "-m64", "-pthread", "-msimd128", *exception_flags, "-fPIC", "-O2", "-g2",
            "--no-entry", "-sENVIRONMENT=worker", "-sMODULARIZE=1",
            "-sEXPORT_NAME=createBlenderPythonChild", "-sMAIN_MODULE=2", "-sAUTOLOAD_DYLIBS=0",
            "-sEXPORTED_RUNTIME_METHODS=FS", "-sALLOW_MEMORY_GROWTH=1", "-sFORCE_FILESYSTEM=1",
            "-sSTACK_SIZE=2MB", "--pre-js", str(source / "shared_child_pre.js"),
            "-I", str(sysroot / "include/python3.13"), str(source / "child.c"),
            str(source / "shared_child_arena.c"),
            *(str(sysroot / "lib" / library) for library in libraries),
            "-Wl,--undefined=__cancel",
            f"-Wl,--reproduce={replay}", "-o", f"{name}.js",
        ],
        cwd=temporary,
        check=True,
    )
    with tarfile.open(replay) as archive:
        archive.extractall(temporary, filter="data")
    link_root = temporary / "shared-worker-link"
    replay_args = iter(shlex.split((link_root / "response.txt").read_text(encoding="utf-8")))
    args = []
    for arg in replay_args:
        if arg == "--export-table" or arg.startswith(("--table-base=", "--global-base=")):
            continue
        if arg == "--export":
            symbol = next(replay_args)
            if symbol not in {"__heap_base", "__stack_pointer"}:
                args.extend((arg, symbol))
        else:
            args.append(arg)
    llvm = Path(subprocess.check_output(["em-config", "LLVM_ROOT"], text=True).strip())
    subprocess.run(
        [str(llvm / "wasm-ld"), *args, "--pie",
         "--export-if-defined=__wasm_apply_data_relocs"],
        cwd=link_root,
        check=True,
    )
    wasm = link_root / f"{name}.wasm"
    subprocess.run([str(llvm / "llvm-strip"), "--strip-debug", str(wasm)], check=True)

    glue = (temporary / f"{name}.js").read_text(encoding="utf-8")
    glue = _replace_runtime_hook(
        glue,
        "        wasmModule = msgData.wasmModule;",
        '        wasmModule = msgData.wasmModule;\n        Module["sharedArena"] = msgData.sharedArena;',
    )
    glue = _replace_runtime_hook(
        glue,
        "      wasmMemory,\n      wasmModule,",
        '      wasmMemory,\n      wasmModule,\n      sharedArena: Module["sharedArena"],',
    )
    glue = _replace_runtime_hook(
        glue,
        "    worker = new Worker(pthreadMainJs, {",
        '    worker = Module["sharedArena"] ? Module["allocateSharedPThread"]() : '
        "new Worker(pthreadMainJs, {",
    )
    glue = _replace_runtime_hook(
        glue,
        "  var info = getWasmImports();",
        """  var info = getWasmImports();
  if (Module["sharedArena"]) {
    wasmBinaryFile ??= findWasmBinary();
    var binary = wasmModule ?? await WebAssembly.compile(await getWasmBinary(wasmBinaryFile));
    return receiveInstance(Module["instantiateSharedPython"](binary, info), binary);
  }""",
    )
    (output / f"{name}.js").write_text(glue, encoding="utf-8")
    shutil.copyfile(wasm, output / f"{name}.wasm")


def build(sysroot: Path, output: Path, shared_memory_sysroot: Path | None = None, *,
          native_exceptions: bool = False) -> None:
    source = Path(__file__).resolve().parent
    sysroot = sysroot.resolve()
    output = output.resolve()
    stdlib = sysroot / "lib/python3.13"
    if not (stdlib / "encodings/__init__.py").is_file():
        raise ValueError(f"missing CPython 3.13 standard library under {stdlib}")
    exception_flags = (("-fwasm-exceptions", "-sSUPPORT_LONGJMP=wasm") if native_exceptions
                       else ("-fexceptions",))
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as temporary:
        archive = Path(temporary) / "python313.zip"
        with ZipFile(archive, "w", ZIP_DEFLATED) as package:
            for file in sorted(stdlib.rglob("*")):
                relative = file.relative_to(stdlib)
                if file.is_file() and "__pycache__" not in relative.parts:
                    package.write(file, arcname=relative.as_posix())
            for module in sorted((source.parent / "python").glob("_blender_web_*.py")):
                package.write(module, arcname=module.name)
        subprocess.run(
            [
                "emcc", "-m64", "-pthread", "-msimd128", *exception_flags, "-O3", "--no-entry",
                "-sENVIRONMENT=worker", "-sMODULARIZE=1", "-sEXPORT_NAME=createBlenderPythonChild",
                "-sEXPORTED_RUNTIME_METHODS=FS", "-sALLOW_MEMORY_GROWTH=1", "-sFORCE_FILESYSTEM=1",
                "-sSTACK_SIZE=2MB",
                "-I", str(sysroot / "include/python3.13"), str(source / "child.c"),
                *(str(sysroot / "lib" / library) for library in
                  ("libpython3.13.a", "libexpat.a", "libbz2.a", "libz.a")),
                "--preload-file", f"{archive}@/lib/python313.zip",
                "-o", "blender-python-child.js",
            ],
            cwd=output,
            check=True,
        )
        if shared_memory_sysroot is not None:
            _build_shared_worker(source, shared_memory_sysroot.resolve(), output, Path(temporary),
                                 exception_flags)
    for name in ("client.mjs", "worker.js"):
        shutil.copyfile(source / name, output / name)
    if shared_memory_sysroot is not None:
        with (output / "client.mjs").open("a", encoding="utf-8") as client:
            client.write("\nstartPythonProcess.supportsSharedMemory = true;\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the Blender browser Python child runtime.")
    parser.add_argument("sysroot", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--shared-memory-sysroot", type=Path,
                        help="also build a shared-memory worker from PIC Python and libraries")
    parser.add_argument("--native-exceptions", action="store_true",
                        help="match a sysroot built with native Wasm exceptions")
    args = parser.parse_args()
    build(args.sysroot, args.output, args.shared_memory_sysroot,
          native_exceptions=args.native_exceptions)
