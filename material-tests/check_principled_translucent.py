#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Compare Principled Alpha with an independent Translucent BSDF against desktop Cycles."""

import argparse
from pathlib import Path

from check_glossy_transparency import alpha_rmse
from check_mixed_glossy import guided_denoise_proven
from check_surface_scattering import compare, rgb_rmse
from run import case_id, dump, positive, run_side


def settings():
    linked = {"image": True, "alpha": True, "principled_translucent": True}
    yield "principled-only", {**linked, "only_principled": True}
    yield "add-linked-alpha", linked
    yield "add-linked-alpha-reverse", {**linked, "order": "reverse"}
    yield "mix-linked-alpha", {**linked, "mix": 0.37}
    yield "mix-linked-alpha-reverse", {**linked, "mix": 0.37, "order": "reverse"}
    yield "constant-alpha", {**linked, "fixed_alpha": 0.5}
    yield "opaque", {**linked, "fixed_alpha": 1}
    yield "multiscatter", {**linked, "distribution": "MULTI_GGX"}
    yield "normal", {**linked, "normal_pass": True, "alpha": False}
    yield "guided", {**linked, "denoise": True, "alpha": False}
    yield "varying-shadow", {**linked, "shadow": "mixed", "alpha": False}
    yield "opaque-shadow", {"shadow": "opaque"}


def cases(samples):
    for name, graph in settings():
        case = {"node": "ShaderNodeOutputMaterial", "settings": {}, "output": "Surface",
                "cycles_samples": samples, "include_pixels": True, "resolution": 64,
                "paper_closures": graph}
        yield {"id": case_id(case), "name": name, **case}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-build", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=positive, default=128)
    parser.add_argument("--timeout", type=positive, default=420)
    args = parser.parse_args()
    args.lighting = "area-hdri"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    fixtures = list(cases(args.samples))
    cpu = run_side(args, output, "cycles", "native", "CPU", 0, fixtures)
    web = run_side(args, output, "cycles", "web", "WEBGPU", 0, fixtures)
    rows = {}
    for case in fixtures:
        desktop, browser = (results.get(case["id"], {}) for results in (cpu, web))
        verdict, rgb_error = compare(desktop, browser)
        alpha_error = None
        graph = case["paper_closures"]
        if verdict == "pass":
            if graph.get("alpha"):
                alpha_error = alpha_rmse(desktop["image"], browser["image"])
                if alpha_error > 0.005:
                    verdict = "alpha_mismatch"
            if graph.get("normal_pass") and rgb_error > 0.001:
                verdict = "normal_mismatch"
            if graph.get("denoise") and not guided_denoise_proven(desktop, browser):
                verdict = "denoising_unproven"
        name = case["name"]
        print(name, verdict, rgb_error, alpha_error, flush=True)
        rows[name] = {"case": case, "verdict": verdict, "rgb_rmse": rgb_error,
                      "alpha_rmse": alpha_error, "native": desktop, "web": browser}

    signals = {}
    for side in ("native", "web"):
        for label, first, second in (
                ("translucent", "add-linked-alpha", "principled-only"),
                ("shadow", "varying-shadow", "opaque-shadow")):
            a, b = (rows[name][side].get("image") for name in (first, second))
            signals[f"{side} {label}"] = rgb_rmse(a, b) if a and b else 0.0
    print("Fixture signals", signals, flush=True)
    dump(output / "results.json", {"signals": signals, "cases": rows})
    return int(any(value <= 0.008 for value in signals.values()) or
               any(row["verdict"] != "pass" for row in rows.values()))


if __name__ == "__main__":
    raise SystemExit(main())
