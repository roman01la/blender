#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Compare Cycles WebGPU transparent camera rays and shadows with desktop Metal."""

import argparse
from pathlib import Path

from run import case_id, compare, dump, positive, run_side


def transparency_cases():
    modes = ("image-tinted", "image-alpha", "noise-tinted", "unoccluded",
             "image-colored-shadow", "noise-colored-shadow", "image-opaque-shadow",
             "colored-shadow", "query-noise-tinted", "query-noise-colored-shadow")
    cases = []
    for mode in modes:
        node = ("ShaderNodeBsdfTransparent" if mode.endswith(("tinted", "alpha"))
                else "ShaderNodeBsdfDiffuse")
        case = {"node": node, "settings": {}, "output": "BSDF", "transparent_fixture": mode}
        cases.append({"id": case_id(case), **case})
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-build", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=positive, default=200)
    args = parser.parse_args()
    args.lighting = "area-world"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    cases = transparency_cases()
    native = run_side(args, output, "cycles", "native", "METAL", 0, cases)
    browser = run_side(args, output, "cycles", "web", "WEBGPU", 0, cases)

    rows = []
    for case in cases:
        desktop = native.get(case["id"], {})
        web = browser.get(case["id"], {})
        verdict = compare(desktop, web)
        error = None
        if verdict == "both_gpu":
            desktop_image, web_image = desktop["image"], web["image"]
            values = zip(desktop_image["mean_rgb"], web_image["mean_rgb"], strict=True)
            error = max(abs(a - b) for a, b in values)
            alpha_error = abs(desktop_image["mean_alpha"] - web_image["mean_alpha"])
            verdict = "pass" if max(error, alpha_error) <= 0.005 else "image_mismatch"
        print(case["transparent_fixture"], verdict, error, flush=True)
        rows.append({"case": case, "verdict": verdict, "max_mean_rgb_error": error,
                     "native": desktop, "web": web})

    images = {case["transparent_fixture"]: native.get(case["id"], {}).get("image", {})
              for case in cases}
    alpha = images["image-alpha"].get("mean_alpha", 0)
    shadow = (images["unoccluded"].get("mean_rgb", [0])[0] -
              images["image-colored-shadow"].get("mean_rgb", [0])[0])
    signal_pass = 0 < alpha < 1 and shadow > 0.05
    dump(output / "results.json", {"signal_pass": signal_pass, "cases": rows})
    return int(not signal_pass or any(row["verdict"] != "pass" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
