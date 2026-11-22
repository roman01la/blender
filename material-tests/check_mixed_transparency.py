#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Compare mixed Transparent paths and shadows with desktop CPU Cycles."""

import argparse
from pathlib import Path

from check_surface_scattering import compare
from run import case_id, dump, positive, run_side


def cases(samples):
    modes = ("constant", "reverse", "linked", "alpha", "shadow",
             "opaque-shadow", "add", "limited", "principled-constant", "principled-reverse",
             "principled-linked", "principled-shadow", "principled-limited")
    for mode in modes:
        node = "BsdfDiffuse" if mode.endswith("shadow") else "BsdfTransparent"
        case = {"node": f"ShaderNode{node}", "settings": {}, "output": "BSDF",
                "transparent_fixture": f"mixed-{mode}", "cycles_samples": samples,
                "include_pixels": True}
        yield {"id": case_id(case), **case}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-build", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=positive, default=128)
    parser.add_argument("--timeout", type=positive, default=240)
    args = parser.parse_args()
    args.lighting = "area-world"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    fixtures = list(cases(args.samples))
    cpu = run_side(args, output, "cycles", "native", "CPU", 0, fixtures)
    browser = run_side(args, output, "cycles", "web", "WEBGPU", 0, fixtures)
    rows = {}
    for case in fixtures:
        desktop = cpu.get(case["id"], {})
        web = browser.get(case["id"], {})
        verdict, error = compare(desktop, web)
        alpha_error = None
        if verdict == "pass":
            alpha_error = abs(desktop["image"]["mean_alpha"] - web["image"]["mean_alpha"])
            if alpha_error > 0.005:
                verdict = "image_mismatch"
        mode = case["transparent_fixture"]
        print(mode, verdict, error, alpha_error, flush=True)
        rows[mode] = {"case": case, "verdict": verdict, "rgb_rmse": error,
                      "mean_alpha_error": alpha_error, "native": desktop, "web": web}

    def native_image(mode):
        return rows[f"mixed-{mode}"]["native"].get("image", {})

    alpha = native_image("alpha").get("mean_alpha", 0)
    shadow = (native_image("shadow").get("mean_rgb", [0])[0] -
              native_image("opaque-shadow").get("mean_rgb", [0])[0])
    signal_pass = 0.1 < alpha < 0.99 and shadow > 0.025
    dump(output / "results.json", {"signal_pass": signal_pass, "cases": rows})
    return int(not signal_pass or any(row["verdict"] != "pass" for row in rows.values()))


if __name__ == "__main__":
    raise SystemExit(main())
