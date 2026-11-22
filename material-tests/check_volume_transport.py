#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Check WebGPU volume integration and resident OIDN against desktop Cycles."""

import argparse
import json
from pathlib import Path
import re

from check_volume_scattering import image_errors
from run import case_id, dump, positive, run_side


MODES = (
    ("surface guided", "surface", True, "LINEAR"),
    ("homogeneous raw", "homogeneous", False, "LINEAR"),
    ("homogeneous guided", "homogeneous", True, "LINEAR"),
    ("overlap raw", "overlap", False, "LINEAR"),
    ("overlap guided", "overlap", True, "LINEAR"),
    ("overlap inside raw", "overlap-inside", False, "LINEAR"),
    ("overlap inside guided", "overlap-inside", True, "LINEAR"),
    ("overlap absorption raw", "overlap-absorption", False, "LINEAR"),
    ("overlap absorption guided", "overlap-absorption", True, "LINEAR"),
    ("grid raw", "grid", False, "LINEAR"),
    ("grid guided", "grid", True, "LINEAR"),
    ("cubic grid guided", "grid", True, "CUBIC"),
    ("grid and homogeneous overlap guided", "grid-overlap", True, "LINEAR"),
)


def make_cases(args):
    for name, media, denoise, interpolation in MODES:
        if not re.search(args.match, name):
            continue
        surface = media == "surface"
        case = {
            "node": "ShaderNodeBsdfDiffuse" if surface else "ShaderNodeVolumeScatter",
            "output": "BSDF" if surface else "Volume",
            "settings": {},
            "volume_transport": {"media": media, "denoise": denoise, "interpolation": interpolation},
            "cycles_samples": args.samples,
            "cycles_volume_bounces": args.volume_bounces,
            "resolution": args.resolution,
            "include_pixels": True,
        }
        yield {"id": case_id(case), "name": name, **case}


def denoising_proof(web, case):
    evidence = web.get("evidence", [])
    marker = "VOLUME_WEBGPU_PROOF "
    traces = [json.loads(line[len(marker):]) for line in evidence if line.startswith(marker)]
    models = {match[1] for line in evidence if (match := re.search(
        r"Cycles WebGPU OIDN dispatched (\S+): [1-9]\d* tiles .*GPU-resident", line))}
    guides = case["volume_transport"]["media"] == "surface" or any(
        "Cycles WebGPU writing volume denoising guides" in line for line in evidence)
    export = any(f"Cycles WebGPU resident film export {case['resolution']} x {case['resolution']}" in line
                 for line in evidence)
    return (
        guides and export and models == {
            "rt_alb_large.tza", "rt_nrm_large.tza", "rt_hdr_calb_cnrm_large.tza",
        } and len(traces) == 1 and all(
            trace.get("caseId") == case["id"] and
            trace["traceDispatches"] > 0 and trace["denoiseDispatches"] > 0 and
            trace["exportBytes"] >= case["resolution"] ** 2 * 4 * 4 and
            trace.get("exportSource") == "oidn/film/shard" and not trace["otherFilmReads"]
            for trace in traces
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-build", type=Path)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--volume-grid", type=Path, default=Path(__file__).resolve().parents[2] /
                        "blender/tests/files/render/openvdb/smoke_low_res.vdb")
    parser.add_argument("--samples", type=positive, default=256)
    parser.add_argument("--resolution", type=positive, default=64)
    parser.add_argument("--volume-bounces", type=positive, default=4)
    parser.add_argument("--timeout", type=positive, default=420)
    parser.add_argument("--match", default=".")
    args = parser.parse_args()
    args.lighting = "mixed-hdri"
    cases = list(make_cases(args))
    if not cases:
        parser.error("--match did not select any volume transport cases")
    if args.resolution < 8:
        parser.error("--resolution must be at least 8")
    if any(case["volume_transport"]["media"].startswith("grid") for case in cases):
        if not args.volume_grid.is_file():
            parser.error(f"grid fixture missing: {args.volume_grid}")
    else:
        args.volume_grid = None

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cpu = run_side(args, output, "cycles", "native", "CPU", 0, cases)
    web = run_side(args, output, "cycles", "web", "WEBGPU", 0, cases) if args.browser_build else {}
    results = []
    for case in cases:
        reference = cpu.get(case["id"], {})
        browser = web.get(case["id"], {})
        verdict = "desktop_cpu_unproven"
        rmse = center = None
        proof = False
        cpu_oidn = any("Denoising on:" in line and "[CPU]" in line
                       for line in reference.get("evidence", []))
        if reference.get("execution") == "cpu_control" and (
                cpu_oidn or not case["volume_transport"]["denoise"]):
            verdict = "desktop_only" if not args.browser_build else browser.get(
                "execution", "browser_unproven")
            if verdict == "gpu":
                verdict = "different_fixture"
                if browser["fingerprint"] == reference["fingerprint"]:
                    rmse, center = image_errors(reference, browser)
                    proof = not case["volume_transport"]["denoise"] or denoising_proof(browser, case)
                    verdict = ("denoising_unproven" if not proof else
                               "pass" if rmse <= 0.035 and center <= 0.02 else "image_mismatch")
        results.append({"case": case, "verdict": verdict, "rgb_rmse": rmse,
                        "max_mean_rgb_error": center, "gpu_proof": proof,
                        "native": reference, "web": browser})
        print(case["name"], verdict, rmse, center, flush=True)
    dump(output / "results.json", {"cases": results})
    return int(any(row["verdict"] not in {"pass", "desktop_only"} for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
