#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Compare mapped, color-processed HDRI World shaders with desktop Cycles."""

import argparse
from pathlib import Path

from check_mixed_glossy import guided_denoise_proven
from check_surface_scattering import compare, rgb_rmse
from run import case_id, dump, positive, run_side


def cases(samples):
    mapped = {"mapped": True}
    graph = {**mapped, "color": True}
    settings = {
        "control": {"mapped": False},
        "mapping": mapped,
        "color-tree": graph,
        "color-cdf": {**graph, "light_tree": False},
        "color-without-mis": {**graph, "sampled": False},
        "color-single-area": {**graph, "sampled": False, "single_area": True},
        "color-illumination": {**graph, "camera": False},
        "mapping-illumination": {**mapped, "camera": False},
        "color-guided": {**graph, "denoise": True},
    }
    for name, world in settings.items():
        case = {"node": "ShaderNodeBsdfDiffuse", "settings": {}, "output": "BSDF",
                "cycles_samples": samples, "include_pixels": True, "resolution": 64,
                "world_graph": world}
        yield {"id": case_id(case), "name": name, **case}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-build", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=positive, default=128)
    parser.add_argument("--timeout", type=positive, default=420)
    args = parser.parse_args()
    args.lighting = "mixed-hdri"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fixtures = list(cases(args.samples))
    cpu = run_side(args, output, "cycles", "native", "CPU", 0, fixtures)
    browser = run_side(args, output, "cycles", "web", "WEBGPU", 0, fixtures)
    rows = {}
    for case in fixtures:
        desktop, web = (results.get(case["id"], {}) for results in (cpu, browser))
        verdict, error = compare(desktop, web)
        name = case["name"]
        if verdict == "pass" and case["world_graph"].get("denoise") and not guided_denoise_proven(
                desktop, web):
            verdict = "denoising_unproven"
        rows[name] = {"case": case, "verdict": verdict, "rgb_rmse": error,
                      "native": desktop, "web": web}
        print(name, verdict, error, flush=True)

    signals = {}
    for side in ("native", "web"):
        for label, first, second in (
                ("rotation", "control", "mapping"),
                ("color", "mapping", "color-tree"),
                ("illumination", "mapping-illumination", "color-illumination")):
            a, b = (rows[name][side].get("image") for name in (first, second))
            signals[f"{side} {label}"] = rgb_rmse(a, b) if a and b else 0.0
    print("Fixture signals", signals, flush=True)
    dump(output / "results.json", {"signals": signals, "cases": rows})
    return int(any(value <= 0.005 for value in signals.values()) or
               any(row["verdict"] != "pass" for row in rows.values()))


if __name__ == "__main__":
    raise SystemExit(main())
