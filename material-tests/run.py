#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Run and compare the Blender material-node GPU matrix."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from catalog import cases, json_key


ROOT = Path(__file__).resolve().parent
MARKER = "MATERIAL_TEST "
CPU_FALLBACK = re.compile(r"Cycles WebGPU.*(?:using CPU|fall(?:ing)? back to CPU|CPU replay)", re.I)
SHADER_FAILURE = re.compile(
    r"shader compilation (?:failed|error)|(?:could not|failed to) compile (?:the )?(?:.*shader|.*pipeline)|"
    r"error while (?:parsing|validating) (?:WGSL|shader)|invalid (?:shader ?module|render ?pipeline|"
    r"compute ?pipeline|command ?buffer|bind ?group)|GPUValidationError|Validation Error|"
    r"Binding doesn't exist in \[BindGroupLayout|WebGPU batch backend does not implement|"
    r"(?:GPU|WebGPU|Metal) device (?:was )?lost|\bWGSL.*error:|WebGPU shader '[^']+':", re.I)
TRACE_DEVICE = re.compile(r"Path tracing on:.*?\[([A-Z0-9]+)(?=[_\]])")
NON_HARDWARE = re.compile("swiftshader|llvmpipe|softpipe|software rasterizer", re.I)


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def case_id(case):
    digest = hashlib.sha256(json_key(case).encode()).hexdigest()[:16]
    return f"{case['node']}/{digest}"


def parse_log(lines, engine, target):
    results = {}
    active = None
    evidence = []
    fingerprint = None
    fatal = None
    materials = {}
    shader_diagnostics = {}

    def incomplete():
        result = {"status": "incomplete", "evidence": list(dict.fromkeys(evidence)),
                  "fingerprint": fingerprint}
        result["execution"] = execution(result, engine, target)
        return result

    for line in lines:
        if line.startswith(MARKER):
            event = json.loads(line[len(MARKER):])
            match event["event"]:
                case "begin":
                    if event["id"] != active:
                        if active is not None:
                            results[active] = incomplete()
                        active, evidence, fingerprint = event["id"], [], None
                case "connected":
                    if event["id"] == active:
                        fingerprint = event["fingerprint"]
                        if event.get("material"):
                            materials[event["material"]] = active
                case "result":
                    if event["id"] == active:
                        event.pop("event")
                        event["evidence"] = list(dict.fromkeys(evidence))
                        event.setdefault("fingerprint", fingerprint)
                        event["execution"] = execution(event, engine, target)
                        results[active] = event
                        active, evidence = None, []
                case "fatal":
                    fatal = event["reason"]
        elif SHADER_FAILURE.search(line):
            # Native warnings can arrive just after the case that submitted the shader.
            tags = re.findall(r"Probe_(?:[a-f0-9]{16}|baseline)", line)
            if tags:
                for tag in tags:
                    shader_diagnostics.setdefault(tag, []).append(line.strip())
            elif active:
                evidence.append(line.strip())
        elif active and (CPU_FALLBACK.search(line) or TRACE_DEVICE.search(line) or
                         "Cycles WebGPU stock scheduler completed" in line or
                         "Denoising on:" in line or
                         "Cycles WebGPU OIDN dispatched" in line or
                         "Cycles WebGPU writing volume denoising guides" in line or
                         "Cycles WebGPU resident film export" in line or
                         line.startswith("VOLUME_WEBGPU_PROOF ")):
            evidence.append(line.strip())
    if active is not None:
        results[active] = incomplete()
    for tag, messages in shader_diagnostics.items():
        identifier = materials.get(tag)
        if identifier in results:
            event = results[identifier]
            event["evidence"] = list(dict.fromkeys([*event["evidence"], *messages]))
            event["execution"] = execution(event, engine, target)
    return results, fatal


def execution(result, engine, target):
    evidence = result["evidence"]
    if engine == "cycles" and any(CPU_FALLBACK.search(line) for line in evidence):
        return "cpu_fallback"
    if any(SHADER_FAILURE.search(line) for line in evidence):
        return "gpu_failure"
    if result["status"] != "completed":
        return result["status"]
    if engine == "cycles":
        devices = [match.group(1) for line in evidence if (match := TRACE_DEVICE.search(line))]
        if "CPU" in devices:
            return "cpu_fallback" if target != "CPU" else "cpu_control"
        if target == "WEBGPU":
            return ("gpu" if any("Cycles WebGPU stock scheduler completed" in line for line in evidence)
                    else "unproven")
        return "gpu" if target in devices else "unproven"
    info = result["gpu"]
    if NON_HARDWARE.search(info["renderer"]):
        return "software_gpu"
    return "gpu" if info["backend"] in {"METAL", "VULKAN", "OPENGL", "WEBGPU"} else "unproven"


def compare(native, web):
    if native.get("execution") != "gpu":
        return "native_gpu_unproven"
    if native.get("fingerprint") and web.get("fingerprint") and native["fingerprint"] != web["fingerprint"]:
        return "different_graph"
    if web.get("execution") == "cpu_fallback":
        return "browser_cpu_fallback" if web.get("fingerprint") else "browser_unproven"
    if web.get("execution") in {"gpu_failure", "software_gpu"}:
        return "browser_gpu_failure"
    if web.get("execution") != "gpu":
        return "browser_unproven"
    if web["image"]["magenta"] > 0.95 and native["image"]["magenta"] < 0.05:
        return "browser_shader_output_suspect"
    return "both_gpu"


def run_logged(command, logfile, timeout):
    with logfile.open("w") as log:
        try:
            return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=timeout, check=False).returncode
        except subprocess.TimeoutExpired:
            log.write("\nTest runner timed out\n")
            return 124


def browser_command(args, request, catalog="-"):
    return ["node", ROOT / "browser.mjs", args.browser_build, request, catalog, str(args.timeout)]


def native_command(args, request):
    return [args.native, "-b", "--factory-startup", "--log", "cycles,gpu", "--log-level", "info",
            "--python-exit-code", "1", "--python", ROOT / "probe.py", "--", request]


def inventory(args):
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = destination.with_suffix(".request.json")
    log = destination.with_suffix(".log")
    dump(request, {"inventory": str(destination)})
    if run_logged(browser_command(args, request, destination), log, args.timeout + 85):
        raise SystemExit(f"Catalog extraction failed. See {log}")
    data = json.loads(destination.read_text())
    if args.source_revision:
        data["blender"]["source_revision"] = args.source_revision
        destination.write_text(json_key(data) + "\n")
    print(f"{len(data['nodes'])} registered shader nodes in {destination}")


def report(path, run, rows):
    summary = Counter((row["engine"], row["verdict"]) for row in rows)
    lines = ["# Material node GPU report", "",
             f"- Browser Blender {run['browser'].get('source_revision', run['browser']['hash'])} "
             f"({run['browser']['version']})",
             f"- Native Blender {run['native']} with Cycles {run['native_device']}",
             f"- Strategy {run['strategy']} with {run['lighting']} lighting", "",
             "| Engine | Result | Cases |", "| --- | --- | ---: |"]
    lines.extend(f"| {engine} | {verdict} | {count} |"
                 for (engine, verdict), count in sorted(summary.items()))
    lines.extend(["", "## Browser CPU fallbacks compared with a native GPU render", "",
                  "| Engine | Node | Cases |", "| --- | --- | ---: |"])
    for (engine, node), count in sorted(Counter(
            (row["engine"], row["case"]["node"]) for row in rows
            if row["verdict"] == "browser_cpu_fallback").items()):
        lines.append(f"| {engine} | {node} | {count} |")
    lines.extend(["", "Shader failures, skipped cases, graph differences and missing GPU evidence "
                  "are recorded separately. See results.json for case settings and exact diagnostics."])
    path.write_text("\n".join(lines) + "\n")


def run_side(args, output, engine, side, device, start, batch):
    baseline = {"id": "baseline", "node": "ShaderNodeBsdfDiffuse",
                "settings": {}, "output": "BSDF"}
    remaining = [baseline, *batch]
    collected = {}
    attempt = 0
    while remaining:
        stem = f"{engine}-{start:05d}-{side}" + (f"-retry-{attempt}" if attempt else "")
        request = output / f"{stem}.json"
        log = output / f"{stem}.log"
        options = {"engine": engine, "device": device, "cases": remaining,
                   "lighting": args.lighting, "image": str(output / f"{stem}.png")}
        if grid := getattr(args, "volume_grid", None):
            options["volume_grid"] = str(grid.resolve())
        dump(request, options)
        command = native_command(args, request) if side == "native" else browser_command(args, request)
        code = run_logged(command, log, args.timeout + 85)
        with log.open() as stream:
            results, fatal = parse_log(stream, engine, device)
        collected.update(results)
        if code or fatal:
            print(f"{side} {fatal or f'batch exited {code}'} ({log.name})", flush=True)
        if not results or fatal:
            break
        remaining = [case for case in remaining if case["id"] not in collected]
        attempt += 1
    return collected


def matrix(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    catalog = json.loads(args.catalog.read_text())
    version = subprocess.check_output([args.native, "--version"], text=True,
                                      stderr=subprocess.DEVNULL, timeout=30)
    native_hash = re.search(r"build hash:\s*(\S+)", version)
    native_hash = native_hash[1] if native_hash else "unknown"
    browser_hash = catalog["blender"].get("source_revision", catalog["blender"]["hash"])
    if not browser_hash.startswith(native_hash) and not args.allow_different_revision:
        raise SystemExit("Native revision differs from the browser catalog. "
                         "Pass --allow-different-revision to compare matching graph fingerprints.")
    visible = []
    seen = set()
    for case in cases(catalog, args.strategy):
        identifier = case_id(case)
        if identifier not in seen and re.search(args.match, json_key(case)):
            visible.append({"id": identifier, **case})
            seen.add(identifier)
    if args.limit:
        visible = visible[:args.limit]
    if not visible:
        raise SystemExit("No material cases matched")
    selected = visible[args.shard_index::args.shards]
    if not selected:
        raise SystemExit("This shard has no cases")
    run = {"browser": catalog["blender"], "native": native_hash,
           "native_device": args.native_device, "strategy": args.strategy, "lighting": args.lighting}
    rows = []
    for engine in args.engines:
        for start in range(0, len(selected), args.batch_size):
            batch = selected[start:start + args.batch_size]
            print(f"{engine} {start + 1}–{start + len(batch)} / {len(selected)}", flush=True)
            results = {side: run_side(args, output, engine, side, device, start, batch)
                       for side, device in (("native", args.native_device), ("web", "WEBGPU"))}
            controls = {side: results[side].get("baseline", {}).get("execution", "missing")
                        for side in results}
            for case in batch:
                native, web = (results[side].get(case["id"], {"status": "missing"})
                               for side in ("native", "web"))
                verdict = compare(native, web)
                if verdict == "browser_cpu_fallback" and controls["web"] != "gpu":
                    verdict = "browser_control_unproven"
                rows.append({"engine": engine, "case": case, "native": native, "web": web,
                             "controls": controls, "verdict": verdict})
            dump(output / "results.json", {"run": run, "cases": rows})
            report(output / "report.md", run, rows)
    for result, count in sorted(Counter(row["verdict"] for row in rows).items()):
        print(f"{result} {count}")
    if (args.fail_on_fallback and any(row["verdict"] in {
            "browser_cpu_fallback", "browser_gpu_failure", "browser_shader_output_suspect"}
            for row in rows)):
        return 1
    if not any(row["native"].get("execution") == "gpu" for row in rows):
        return 2
    return 0


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return value


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="action", required=True)
    generate = commands.add_parser("inventory")
    compare_runs = commands.add_parser("run")
    for command in (generate, compare_runs):
        command.add_argument("--browser-build", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--timeout", type=positive, default=300)
    generate.add_argument("--source-revision", help="Known Blender source revision of the browser package")
    compare_runs.add_argument("--native", type=Path, required=True)
    compare_runs.add_argument("--native-device", default="METAL")
    compare_runs.add_argument("--catalog", type=Path, required=True)
    compare_runs.add_argument("--engines", nargs="+", choices=["cycles", "eevee"],
                              default=["cycles", "eevee"])
    compare_runs.add_argument("--strategy", choices=["default", "modes", "options", "pairwise"],
                              default="options")
    compare_runs.add_argument("--match", default=".", help="Regex matched against the case JSON")
    compare_runs.add_argument("--lighting",
                              choices=["area-world", "point", "two-points", "area-hdri", "mixed-hdri"],
                              default="area-world")
    compare_runs.add_argument("--limit", type=positive)
    compare_runs.add_argument("--shards", type=positive, default=1)
    compare_runs.add_argument("--shard-index", type=int, default=0)
    compare_runs.add_argument("--batch-size", type=positive, default=72)
    compare_runs.add_argument("--allow-different-revision", action="store_true")
    compare_runs.add_argument("--fail-on-fallback", action="store_true")
    args = cli.parse_args()
    if args.action == "inventory":
        inventory(args)
        return 0
    if not 0 <= args.shard_index < args.shards:
        cli.error("--shard-index must be between zero and --shards minus one")
    return matrix(args)


if __name__ == "__main__":
    sys.exit(main())
