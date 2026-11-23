#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Compare Diffuse and Glossy closures with desktop CPU Cycles."""

import argparse
from pathlib import Path
import re

from check_mixed_principled import check_guard
from check_surface_scattering import compare, rgb_rmse
from run import case_id, dump, positive, run_side


def pair_settings():
    for distribution in ("GGX", "BECKMANN"):
        for reverse, order in ((False, "diffuse-first"), (True, "glossy-first")):
            yield f"{distribution.lower()}-{order}", {
                "distribution": distribution, "bump": "both", "reverse": reverse}
    for mode in ("flat", "diffuse", "glossy", "shared"):
        yield f"ggx-{mode}", {"distribution": "GGX", "bump": mode}
    for mode in ("flat", "both"):
        yield f"ggx-normal-{mode}", {"distribution": "GGX", "bump": mode, "normal_pass": True}


def ordered_settings(guided=False):
    common = {"ordered": True, "distribution": "GGX", "bump": "shared"}
    for distribution in ("GGX", "BECKMANN"):
        for reverse, order in ((False, "forward"), (True, "reverse")):
            yield f"ordered-{distribution.lower()}-{order}", {
                **common, "distribution": distribution, "reverse": reverse}
    for index, label in ((0, "narrow"), (1, "wide")):
        yield f"ordered-ggx-{label}", {**common, "collapse": index}
    yield "ordered-ggx-independent-bump", {**common, "bump": "both"}
    yield "ordered-ggx-explicit-normal", {**common, "bump": "flat", "explicit_normal": True}
    for mode in ("flat", "shared"):
        yield f"ordered-ggx-normal-{mode}", {**common, "bump": mode, "normal_pass": True}
    for mode in ("anisotropic", "rough-diffuse"):
        yield f"ordered-guard-{mode}", {**common, "outside": mode}
    if guided:
        yield "ordered-ggx-denoised", {**common, "denoise": True}


def displacement_settings(guided=False):
    common = {"displacement_bump": True, "distribution": "GGX", "bump": "independent"}
    for distribution in ("GGX", "BECKMANN"):
        for mode in ("independent", "shared-height"):
            yield f"displacement-{distribution.lower()}-{mode}", {
                **common, "distribution": distribution, "bump": mode}
    for mode in ("flat", "local", "displacement"):
        yield f"displacement-ggx-{mode}", {**common, "bump": mode}
    for mode in ("flat", "independent", "shared-height"):
        yield f"displacement-ggx-normal-{mode}", {**common, "bump": mode, "normal_pass": True}
    if guided:
        yield "displacement-ggx-denoised", {**common, "bump": "shared-height", "denoise": True}


def mixed_displacement_settings(guided=False):
    common = {"mixed_displacement_bump": True, "distribution": "GGX", "bump": "shared-height"}
    for distribution in ("GGX", "BECKMANN"):
        for reverse, order in ((False, "diffuse-first"), (True, "glossy-first")):
            yield f"mixed-displacement-{distribution.lower()}-{order}", {
                **common, "distribution": distribution, "reverse": reverse}
    yield "mixed-displacement-desk-weight", {**common, "distribution": "BECKMANN", "factor": 0.06}
    yield "mixed-displacement-varying-weight", {**common, "factor": "Z"}
    for factor in (0, 1):
        yield f"mixed-displacement-factor-{factor}", {**common, "factor": factor}
    for bump in ("independent-height", "local", "displacement"):
        yield f"mixed-displacement-{bump}", {**common, "bump": bump}
    for bump in ("flat", "shared-height", "independent-height"):
        yield f"mixed-displacement-normal-{bump}", {**common, "bump": bump, "normal_pass": True}
    yield "mixed-displacement-glossy-color", {**common, "glossy_color_pass": True}
    yield "mixed-displacement-guard-independent-normals", {
        **common, "bump": "independent-normals", "outside": True}
    if guided:
        yield "mixed-displacement-guided", {**common, "denoise": True}


def cases(samples, mode="pair", guided=False):
    settings_for_mode = {
        "ordered": ordered_settings,
        "displacement": displacement_settings,
        "mixed-displacement": mixed_displacement_settings,
    }
    fixtures = pair_settings() if mode == "pair" else settings_for_mode[mode](guided)
    for name, settings in fixtures:
        fixture = {"node": "ShaderNodeOutputMaterial", "settings": {}, "output": "Surface",
                   "cycles_samples": samples, "include_pixels": True, "resolution": 64,
                   "mixed_glossy": settings}
        yield {"id": case_id(fixture), "name": name, **fixture}


def check_signals(rows, mode="pair"):
    if mode == "mixed-displacement":
        images = [("mixed-displacement-independent-height", f"mixed-displacement-{control}")
                  for control in ("local", "displacement")]
        images.extend([(f"mixed-displacement-normal-{bump}", "mixed-displacement-normal-flat")
                       for bump in ("shared-height", "independent-height")])
    elif mode == "displacement":
        images = [("displacement-ggx-independent", f"displacement-ggx-{control}")
                  for control in ("local", "displacement")]
        images.extend([("displacement-ggx-normal-independent", "displacement-ggx-normal-flat"),
                       ("displacement-ggx-normal-shared-height", "displacement-ggx-normal-flat")])
    elif mode == "ordered":
        images = [("ordered-ggx-forward", f"ordered-ggx-{mode}")
                  for mode in ("narrow", "wide", "independent-bump")]
        images.append(("ordered-ggx-normal-flat", "ordered-ggx-normal-shared"))
    else:
        images = [(f"ggx-{mode}", "ggx-diffuse-first") for mode in ("flat", "diffuse", "glossy", "shared")]
        images.append(("ggx-normal-flat", "ggx-normal-both"))
    distances = {}
    for first, second in images:
        for side in ("native", "web"):
            a, b = (rows[name][side].get("image") for name in (first, second))
            distances[f"{side} {first} / {second}"] = rgb_rmse(a, b) if a and b else None
    return distances


def guided_denoise_proven(cpu, web):
    evidence = web["evidence"]
    models = {match[1] for line in evidence if (match := re.search(
        r"Cycles WebGPU OIDN dispatched (\S+): [1-9]\d* tiles .*GPU-resident", line))}
    return (
        any("Denoising on:" in line and "[CPU]" in line for line in cpu["evidence"]) and
        any("Cycles WebGPU writing denoising guides" in line for line in evidence) and
        any("Denoising on:" in line and "[WEBGPU]" in line for line in evidence) and
        models == {"rt_alb_large.tza", "rt_nrm_large.tza", "rt_hdr_calb_cnrm_large.tza"}
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-build", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=positive, default=128)
    parser.add_argument("--timeout", type=positive, default=240)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--ordered", dest="mode", action="store_const", const="ordered", default="pair",
                       help="compare four-closure nested mixes")
    modes.add_argument("--displacement-bump", dest="mode", action="store_const", const="displacement",
                       help="compare a Glossy closure with local and shader-wide bump")
    modes.add_argument("--mixed-displacement-bump", dest="mode", action="store_const",
                       const="mixed-displacement",
                       help="compare Diffuse and Glossy sharing a local normal with shader-wide bump")
    parser.add_argument("--guided", action="store_true",
                        help="also compare a guided OIDN result")
    parser.add_argument("--lighting", choices=("area-world", "two-points", "area-hdri"))
    args = parser.parse_args()
    if args.guided and args.mode == "pair":
        parser.error("--guided requires a closure ordering or displacement mode")
    if args.lighting is None:
        args.lighting = "two-points" if args.mode in {"displacement", "mixed-displacement"} else "area-world"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    fixtures = list(cases(args.samples, args.mode, args.guided))
    cpu = run_side(args, output, "cycles", "native", "CPU", 0, fixtures)
    browser = run_side(args, output, "cycles", "web", "WEBGPU", 0, fixtures)
    rows = {}
    for fixture in fixtures:
        desktop = cpu.get(fixture["id"], {})
        web = browser.get(fixture["id"], {})
        settings = fixture["mixed_glossy"]
        guard = bool(settings.get("outside"))
        verdict, error = (check_guard if guard else compare)(desktop, web)
        # The camera-hit data passes have much less sampling variation than Combined.
        if (verdict == "pass" and error > 0.001 and
                (settings.get("normal_pass") or settings.get("glossy_color_pass"))):
            verdict = "image_mismatch"
        if verdict == "pass" and settings.get("glossy_color_pass"):
            colors = (result["image"]["mean_rgb"] for result in (desktop, web))
            if not all(blue > red + 0.05 for red, _, blue in colors):
                verdict = "glossy_color_unproven"
        if guard and verdict == "pass":
            verdict = "unexpected_gpu_support"
        if verdict == "pass" and settings.get("denoise") and not guided_denoise_proven(desktop, web):
            verdict = "denoising_unproven"
        name = fixture["name"]
        rows[name] = {"case": fixture, "verdict": verdict, "rgb_rmse": error,
                      "native": desktop, "web": web}
        print(name, verdict, error, flush=True)

    signals = check_signals(rows, args.mode)
    failed_signals = [name for name, error in signals.items() if error is None or error <= 0.005]
    for name in failed_signals:
        print(f"Inconclusive fixture signal: {name}", flush=True)
    signal_key = {"pair": "bump_signal_rmse", "ordered": "ordered_signal_rmse",
                  "displacement": "displacement_signal_rmse",
                  "mixed-displacement": "mixed_displacement_signal_rmse"}[args.mode]
    dump(output / "results.json", {signal_key: signals, "cases": rows})
    return int(bool(failed_signals) or
               any(row["verdict"] not in {"pass", "guarded"} for row in rows.values()))


if __name__ == "__main__":
    raise SystemExit(main())
