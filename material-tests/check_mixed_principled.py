#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Compare mixed diffuse-only Principled shaders with desktop CPU Cycles."""

import argparse
from pathlib import Path

from check_surface_scattering import compare, rgb_rmse
from run import case_id, dump, positive, run_side


def cases(samples, nested=False):
    if nested:
        fixtures = {
            "nested-forward": {"mode": "nested"},
            "nested-reverse": {"mode": "nested", "reverse": True},
            "nested-explicit-normal": {"mode": "nested", "explicit_normal": True},
            "nested-reference": {"mode": "nested-reference"},
            "nested-uniform-reference": {"mode": "nested-reference", "varying": False},
        }
    else:
        fixtures = {
            "single-principled": {"mode": "single-principled"},
            "varying-diffuse-reference": {"mode": "varying-diffuse-reference"},
            "independent-bump": {"mode": "independent-bump", "factor": 0.37},
            "below-cutoff": {"mode": "cutoff", "factor": 0.5, "scale": 1},
            "above-cutoff": {"mode": "cutoff", "factor": 0.5, "scale": 2},
        }
        for reverse in (False, True):
            order = "reverse" if reverse else "forward"
            for name, factor in (("zero", 0), ("intermediate", 0.37), ("one", 1)):
                fixtures[f"{order}-{name}"] = {"mode": "constant", "factor": factor,
                                               "reverse": reverse}
            fixtures[f"{order}-varying"] = {"mode": "varying", "reverse": reverse}
            fixtures[f"{order}-varying-diffuse"] = {"mode": "varying", "reverse": reverse,
                                                   "with_diffuse": True}
    for name, settings in fixtures.items():
        fixture = {"node": "ShaderNodeOutputMaterial", "settings": {}, "output": "Surface",
                   "cycles_samples": samples, "include_pixels": True, "resolution": 64,
                   "mixed_principled": settings}
        if settings["mode"] == "cutoff":
            fixture["cycles_exposure"] = 14
        yield {"id": case_id(fixture), "name": name, **fixture}


def check_guard(cpu, browser):
    if (cpu.get("execution") != "cpu_control" or browser.get("execution") != "cpu_fallback" or
            browser.get("status") != "completed" or
            cpu["fingerprint"] != browser["fingerprint"] or
            not any("rejected material" in line for line in browser["evidence"])):
        return compare(cpu, browser)
    error = rgb_rmse(cpu["image"], browser["image"])
    return ("guarded" if error <= 0.005 else "image_mismatch"), error


def center_brightness(image):
    width, height = image["size"]
    pixels = bytes.fromhex(image["rgba_hex"])
    center = [pixels[4 * (y * width + x) + channel]
              for y in range(3 * height // 8, 5 * height // 8)
              for x in range(3 * width // 8, 5 * width // 8)
              for channel in range(3)]
    return sum(center) / (255 * len(center))


def check_nested_signals(rows):
    signals = {}
    for side in ("native", "web"):
        forward, reverse, normal, reference, uniform = (
            rows[name][side].get("image") for name in (
                "nested-forward", "nested-reverse", "nested-explicit-normal",
                "nested-reference", "nested-uniform-reference"))
        signals[f"{side} nested branch order"] = bool(
            forward and reverse and rgb_rmse(forward, reverse) <= 0.005)
        signals[f"{side} nested explicit shading normal"] = bool(
            forward and normal and rgb_rmse(forward, normal) <= 0.005)
        signals[f"{side} nested weighted color"] = bool(
            forward and reference and rgb_rmse(forward, reference) <= 0.005)
        signals[f"{side} nested variation"] = bool(
            forward and uniform and rgb_rmse(forward, uniform) > 0.015)
    return signals


def check_signals(rows):
    comparisons = [(f"forward-{factor}", f"reverse-{factor}")
                   for factor in ("zero", "intermediate", "one", "varying", "varying-diffuse")]
    comparisons += [("forward-zero", "single-principled"),
                    ("forward-varying", "varying-diffuse-reference"),
                    ("forward-varying-diffuse", "varying-diffuse-reference")]
    signals = {}
    for side in ("native", "web"):
        def image(name):
            return rows[name][side].get("image")

        for first, second in comparisons:
            a, b = image(first), image(second)
            signals[f"{side} {first} / {second}"] = bool(a and b and rgb_rmse(a, b) <= 0.005)
        a, b, c = (image(f"forward-{factor}") for factor in ("zero", "intermediate", "one"))
        signals[f"{side} intermediate weights"] = bool(
            a and b and c and a["mean_rgb"][0] > b["mean_rgb"][0] > c["mean_rgb"][0] and
            a["mean_rgb"][2] < b["mean_rgb"][2] < c["mean_rgb"][2])
        bump = image("independent-bump")
        signals[f"{side} independent normals"] = bool(b and bump and rgb_rmse(b, bump) > 0.005)
        below, above = image("below-cutoff"), image("above-cutoff")
        signals[f"{side} per-closure cutoff"] = bool(
            below and above and center_brightness(below) < 0.002 and center_brightness(above) > 0.3)
    return signals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-build", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=positive, default=96)
    parser.add_argument("--timeout", type=positive, default=240)
    parser.add_argument("--nested", action="store_true",
                        help="Check four-branch nested mixes instead of two-branch mixes")
    args = parser.parse_args()
    args.lighting = "area-world"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    fixtures = list(cases(args.samples, args.nested))
    cpu = run_side(args, output, "cycles", "native", "CPU", 0, fixtures)
    browser = run_side(args, output, "cycles", "web", "WEBGPU", 0, fixtures)
    rows = {}
    for case in fixtures:
        desktop = cpu.get(case["id"], {})
        web = browser.get(case["id"], {})
        name = case["name"]
        check = check_guard if name == "independent-bump" else compare
        verdict, error = check(desktop, web)
        rows[name] = {"case": case, "verdict": verdict, "rgb_rmse": error,
                      "native": desktop, "web": web}
        print(name, verdict, error, flush=True)

    signals = check_nested_signals(rows) if args.nested else check_signals(rows)
    for signal, passed in signals.items():
        if not passed:
            print(f"Failed signal: {signal}", flush=True)
    dump(output / "results.json", {"signals": signals, "cases": rows})
    return int(not all(signals.values()) or
               any(row["verdict"] not in {"pass", "guarded"} for row in rows.values()))


if __name__ == "__main__":
    raise SystemExit(main())
