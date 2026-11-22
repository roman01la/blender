#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Compare Cycles Ray Portal and pixel-size Wireframe with desktop Blender."""

import argparse
from pathlib import Path

from run import case_id, compare, dump, positive, run_side


def make_case(mode):
    wireframe = mode.startswith("wireframe")
    case = {
        "node": "ShaderNodeWireframe" if wireframe else "ShaderNodeBsdfRayPortal",
        "output": "Fac" if wireframe else "BSDF",
        "settings": {"use_pixel_size": True} if wireframe else {},
        "ray_transport_fixture": mode,
    }
    return {"id": case_id(case), **case}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-build", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=positive, default=180)
    parser.add_argument("--only", choices=("portal", "wireframe"))
    args = parser.parse_args()
    args.lighting = "area-world"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    modes = (
        "portal-stationary", "portal-linked", "portal-away",
        "portal-shadow-clear", "portal-shadow",
        "wireframe-zero", "wireframe-near", "wireframe-far", "wireframe-adaptive",
        "wireframe-reflection-zero", "wireframe-reflection",
    )
    cases = [make_case(mode) for mode in modes if not args.only or mode.startswith(args.only)]
    native = run_side(args, output, "cycles", "native", "METAL", 0, cases)
    browser = run_side(args, output, "cycles", "web", "WEBGPU", 0, cases)
    rows = {}
    for case in cases:
        name = case["ray_transport_fixture"]
        desktop = native.get(case["id"], {})
        web = browser.get(case["id"], {})
        verdict = compare(desktop, web)
        error = None
        if verdict == "both_gpu":
            error = max(abs(a - b) for a, b in zip(desktop["image"]["mean_rgb"],
                                                  web["image"]["mean_rgb"], strict=True))
            verdict = "pass" if error <= 0.005 else "image_mismatch"
        print(f"{name} {verdict} {error}")
        rows[name] = {"case": case, "verdict": verdict, "max_mean_rgb_error": error,
                      "native": desktop, "web": web}

    def red(name):
        return rows[name]["native"].get("image", {}).get("mean_rgb", (0,))[0]

    signals = {}
    if args.only != "wireframe":
        signals["portal redirection and depth"] = red("portal-linked") - red("portal-away")
        signals["portal default direction"] = red("portal-stationary") - red("portal-away")
        signals["portal shadow"] = red("portal-shadow-clear") - red("portal-shadow")
    if args.only != "portal":
        signals["pixel-size line"] = red("wireframe-near") - red("wireframe-zero")
        signals["distant pixel-size line"] = red("wireframe-far") - red("wireframe-zero")
        signals["reflected pixel-size line"] = red("wireframe-reflection") - red(
            "wireframe-reflection-zero")
    minimum_signals = {"portal shadow": 0.05, "reflected pixel-size line": 0.025}
    meaningful = all(value > minimum_signals.get(name, 0.1) for name, value in signals.items())
    print(f"Desktop fixture signal {signals}")
    dump(output / "results.json", {"signals": signals, "signal_pass": meaningful, "cases": rows})
    return int(not meaningful or any(row["verdict"] != "pass" for row in rows.values()))


if __name__ == "__main__":
    raise SystemExit(main())
