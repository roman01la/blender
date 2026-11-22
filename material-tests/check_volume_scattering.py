#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Check homogeneous volume scattering and direct lighting against desktop CPU Cycles."""

import argparse
from math import fsum, sqrt
from pathlib import Path
import re

from run import case_id, dump, positive, run_side


def volume_cases(samples, volume_bounces, match):
    phases = ("HENYEY_GREENSTEIN", "FOURNIER_FORAND", "DRAINE", "RAYLEIGH", "MIE")
    for node in ("VolumeScatter", "VolumeCoefficients", "VolumePrincipled"):
        for phase in phases if node != "VolumePrincipled" else phases[:1]:
            name = f"{node} {phase}" if node != "VolumePrincipled" else node
            if not re.search(match, name):
                continue
            settings = {} if phase == phases[0] else {"phase": phase}
            case = {"node": f"ShaderNode{node}", "output": "Volume", "settings": settings,
                    "cycles_samples": samples, "cycles_volume_bounces": volume_bounces,
                    "include_pixels": True}
            yield {"id": case_id(case), "name": name, **case}


def image_errors(cpu, browser):
    reference = bytes.fromhex(cpu["image"]["rgba_hex"])
    actual = bytes.fromhex(browser["image"]["rgba_hex"])
    squared_rgb = fsum((a - b) ** 2 for i, (a, b) in enumerate(zip(reference, actual, strict=True))
                       if i % 4 != 3)
    rgb_rmse = sqrt(squared_rgb / (3 * len(reference) / 4)) / 255
    center_error = max(abs(a - b) for a, b in zip(cpu["image"]["mean_rgb"],
                                                  browser["image"]["mean_rgb"], strict=True))
    return rgb_rmse, center_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-build", type=Path)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=positive, default=512)
    parser.add_argument("--volume-bounces", type=int, default=4)
    parser.add_argument("--lighting", choices=("area-world", "mixed-hdri"), default="mixed-hdri")
    parser.add_argument("--match", default=".")
    parser.add_argument("--timeout", type=positive, default=360)
    args = parser.parse_args()
    if not 0 <= args.volume_bounces <= 1024:
        parser.error("--volume-bounces must be between 0 and 1024")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = list(volume_cases(args.samples, args.volume_bounces, args.match))
    if not cases:
        parser.error("--match did not select any volume cases")

    cpu = run_side(args, output, "cycles", "native", "CPU", 0, cases)
    web = run_side(args, output, "cycles", "web", "WEBGPU", 0, cases) if args.browser_build else {}
    rows = []
    for case in cases:
        reference = cpu.get(case["id"], {})
        actual = web.get(case["id"], {})
        verdict = "desktop_cpu_unproven"
        rmse = center = None
        if reference.get("execution") == "cpu_control":
            verdict = "desktop_only" if not args.browser_build else actual.get(
                "execution", "browser_unproven")
            if verdict == "gpu":
                verdict = "different_fixture"
                if actual["fingerprint"] == reference["fingerprint"]:
                    rmse, center = image_errors(reference, actual)
                    # Sampling with and without the stock contribution guides has different variance.
                    verdict = "pass" if rmse <= 0.025 and center <= 0.015 else "image_mismatch"
        rows.append({"case": case, "verdict": verdict, "rgb_rmse": rmse,
                     "max_mean_rgb_error": center, "native": reference, "web": actual})
        print(case["name"], verdict, rmse, center, flush=True)
    dump(output / "results.json", {"samples": args.samples, "volume_bounces": args.volume_bounces,
                                   "lighting": args.lighting,
                                   "cases": rows})
    return int(any(row["verdict"] not in {"pass", "desktop_only"} for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
