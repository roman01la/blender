#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Compare true displacement with residual bump against desktop CPU Cycles."""

import argparse
from pathlib import Path

from check_surface_scattering import compare, rgb_rmse
from run import case_id, dump, positive, run_side


def fixtures(samples):
    settings = {
        "both-combined": {"method": "BOTH"},
        "both-smooth": {"normal_pass": True},
        "true-smooth": {"method": "DISPLACEMENT", "normal_pass": True},
        "bump-smooth": {"method": "BUMP", "normal_pass": True},
        "both-flat": {"flat": True, "normal_pass": True},
        "both-original-map": {"map_base": "ORIGINAL", "normal_pass": True, "transformed": True},
        "both-displaced-map": {"map_base": "DISPLACED", "normal_pass": True, "transformed": True},
        "true-original-map": {"method": "DISPLACEMENT", "map_base": "ORIGINAL", "normal_pass": True},
    }
    for name, graph in settings.items():
        case = {"node": "ShaderNodeOutputMaterial", "settings": {}, "output": "Surface",
                "cycles_samples": samples, "include_pixels": True, "resolution": 64,
                "both_displacement": graph}
        yield {"id": case_id(case), "name": name, **case}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-build", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=positive, default=128)
    parser.add_argument("--timeout", type=positive, default=360)
    args = parser.parse_args()
    args.lighting = "area-hdri"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    cases = list(fixtures(args.samples))
    cpu = run_side(args, output, "cycles", "native", "CPU", 0, cases)
    browser = run_side(args, output, "cycles", "web", "WEBGPU", 0, cases)
    rows = {}
    for case in cases:
        native, web = (result.get(case["id"], {}) for result in (cpu, browser))
        verdict, error = compare(native, web)
        name = case["name"]
        rows[name] = {"case": case, "verdict": verdict, "rgb_rmse": error, "native": native, "web": web}
        print(name, verdict, error, flush=True)

    pairs = (("both-smooth", "true-smooth"), ("both-smooth", "bump-smooth"),
             ("both-original-map", "both-displaced-map"))
    signals = {}
    for first, second in pairs:
        for side in ("native", "web"):
            a, b = (rows[name][side].get("image") for name in (first, second))
            signals[f"{side} {first} / {second}"] = rgb_rmse(a, b) if a and b else None
    print("Fixture distances", signals, flush=True)
    dump(output / "results.json", {"signals": signals, "cases": rows})
    return int(any(error is None or error <= 0.002 for error in signals.values()) or
               any(row["verdict"] != "pass" for row in rows.values()))


if __name__ == "__main__":
    raise SystemExit(main())
