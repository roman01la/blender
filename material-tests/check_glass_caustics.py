#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Compare Cycles Glass visibility with both caustics switches enabled or disabled."""

import argparse
from pathlib import Path
import re

from check_surface_scattering import compare, rgb_rmse
from run import case_id, dump, positive, run_side


def cases(samples):
    for material in ("Principled", "Glass", "Refraction"):
        for disabled in (True, False):
            for diffuse_probe in (True, False):
                case = {
                    "node": "ShaderNodeBsdf" + material, "settings": {}, "output": "BSDF",
                    "cycles_samples": samples, "include_pixels": True, "resolution": 64,
                    "glass_caustics": {"caustics": not disabled, "diffuse_probe": diffuse_probe},
                }
                context = "diffuse" if diffuse_probe else "camera"
                state = "disabled" if disabled else "enabled"
                yield {"id": case_id(case), "name": f"{material}-{context}-{state}", **case}


def signals(rows, backend):
    distances = {}
    for material in ("Principled", "Glass", "Refraction"):
        for context in ("diffuse", "camera"):
            name = f"{material}-{context}"
            matched = [rows.get(f"{name}-{state}") for state in ("disabled", "enabled")]
            if all(matched):
                images = [case[backend].get("image") for case in matched]
                distances[name] = rgb_rmse(*images) if all(images) else None
    return distances


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-build", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=positive, default=128)
    parser.add_argument("--timeout", type=positive, default=420)
    parser.add_argument("--match", default=".")
    args = parser.parse_args()
    args.lighting = "area-hdri"
    args.output.mkdir(parents=True, exist_ok=True)
    fixtures = [case for case in cases(args.samples) if re.search(args.match, case["name"])]
    if not fixtures:
        parser.error("--match did not select any caustics cases")
    native = run_side(args, args.output, "cycles", "native", "CPU", 0, fixtures)
    web = run_side(args, args.output, "cycles", "web", "WEBGPU", 0, fixtures)
    rows = {}
    for fixture in fixtures:
        desktop = native.get(fixture["id"], {})
        browser = web.get(fixture["id"], {})
        verdict, error = compare(desktop, browser)
        name = fixture["name"]
        rows[name] = {"case": fixture, "verdict": verdict, "rgb_rmse": error,
                      "native": desktop, "web": browser}
        print(name, verdict, error, flush=True)

    distances = {side: signals(rows, side) for side in ("native", "web")}
    for side, values in distances.items():
        print(side, values, flush=True)
    signal_pass = all(
        value is not None and
        (value > 0.002 if name.endswith("-diffuse") or name == "Principled-camera" else value < 0.002)
        for values in distances.values() for name, value in values.items()
    )
    dump(args.output / "results.json", {"cases": rows, "signal_pass": signal_pass,
                                       "signals": distances})
    return int(not signal_pass or any(row["verdict"] != "pass" for row in rows.values()))


if __name__ == "__main__":
    raise SystemExit(main())
