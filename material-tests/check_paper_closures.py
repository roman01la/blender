#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Compare ordered Transparent, Diffuse, and Translucent closures with CPU Cycles."""

import argparse
from pathlib import Path

from check_glossy_transparency import alpha_rmse
from check_mixed_glossy import guided_denoise_proven
from check_surface_scattering import compare, rgb_rmse
from run import case_id, dump, positive, run_side


def settings():
    linked = {"image": True, "alpha": True}
    bumped = {**linked, "bump": True}
    paper = {**bumped, "random_images": True}
    yield "diffuse-translucent-control", {"image": True, "pair": "translucent"}
    yield "diffuse-translucent-bump", {"image": True, "bump": True, "pair": "translucent"}
    yield "diffuse-transparent-control", {**linked, "pair": "transparent"}
    for order in ("forward", "reverse", "middle"):
        yield f"constant-{order}", {"order": order, "alpha": True}
    yield "linked-flat", linked
    yield "linked-off-camera-bump", {**linked, "global_bump": True}
    yield "linked-bump", bumped
    yield "linked-reverse", {**bumped, "order": "reverse"}
    yield "random-image-paper", paper
    yield "normal-flat", {**paper, "bump": False, "normal_pass": True, "alpha": False}
    yield "normal-bump", {**paper, "normal_pass": True, "alpha": False}
    yield "guided-oidn", {**paper, "denoise": True, "alpha": False}
    yield "mixed-shadow", {"image": True, "bump": True, "shadow": "mixed"}
    yield "opaque-shadow", {"shadow": "opaque"}


def cases(samples):
    for name, graph in settings():
        case = {"node": "ShaderNodeOutputMaterial", "settings": {}, "output": "Surface",
                "cycles_samples": samples, "include_pixels": True, "resolution": 64,
                "paper_closures": graph}
        yield {"id": case_id(case), "name": name, **case}


def signals(rows, side):
    def image(name):
        return rows[name][side].get("image")

    def distance(a, b):
        first, second = image(a), image(b)
        return rgb_rmse(first, second) if first and second else 0.0

    mixed, opaque = image("mixed-shadow"), image("opaque-shadow")
    paper = image("linked-bump")
    flat, extra_bump = image("linked-flat"), image("linked-off-camera-bump")
    return {
        "translucent_lobe_rmse": distance("linked-flat", "diffuse-transparent-control"),
        "shared_bump_rmse": distance("linked-bump", "linked-flat"),
        "signed_normal_rmse": distance("normal-bump", "normal-flat"),
        "shadow_lift": sum(a - b for a, b in zip(mixed["mean_rgb"], opaque["mean_rgb"], strict=True)) / 3
                       if mixed and opaque else 0.0,
        "paper_mean_alpha": paper["mean_alpha"] if paper else 0.0,
        "off_camera_bump_alpha_rmse": alpha_rmse(flat, extra_bump) if flat and extra_bump else 1.0,
    }


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

    measures = {side: signals(rows, side) for side in ("native", "web")}
    for side, values in measures.items():
        print(side, values, flush=True)
    visible_effects = ("translucent_lobe_rmse", "shared_bump_rmse", "signed_normal_rmse", "shadow_lift")
    signal_pass = all(0.1 < value["paper_mean_alpha"] < 0.95 and
                      value["off_camera_bump_alpha_rmse"] <= 0.001 and
                      all(value[label] > 0.008 for label in visible_effects)
                      for value in measures.values())
    dump(output / "results.json", {"signals": measures, "signal_pass": signal_pass, "cases": rows})
    return int(not signal_pass or any(row["verdict"] != "pass" for row in rows.values()))


if __name__ == "__main__":
    raise SystemExit(main())
