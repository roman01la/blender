#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Compare tiled Image Texture shading on desktop and browser Eevee."""

import argparse
from pathlib import Path

from run import case_id, compare, dump, positive, run_side


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-build", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--native-device", default="METAL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=positive, default=120)
    args = parser.parse_args()
    args.lighting = "area-world"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    cases = []
    for interpolation in ("Closest", "Linear", "Cubic", "Smart"):
        for socket in ("Color", "Alpha"):
            case = {"node": "ShaderNodeTexImage", "settings": {"interpolation": interpolation},
                    "output": socket, "variant": "udim"}
            cases.append({"id": case_id(case), "name": f"{interpolation} {socket}", **case})

    native = run_side(args, output, "eevee", "native", args.native_device, 0, cases)
    web = run_side(args, output, "eevee", "web", "WEBGPU", 0, cases)
    results = []
    for case in cases:
        desktop = native.get(case["id"], {})
        browser = web.get(case["id"], {})
        verdict = compare(desktop, browser)
        error = None
        if verdict == "both_gpu":
            error = max(abs(a - b) for a, b in zip(desktop["image"]["mean_rgb"],
                                                   browser["image"]["mean_rgb"]))
            if error > 0.015:
                verdict = "udim_image_mismatch"
        print(f"{case['name']} {verdict}" + (f" ({error:.6f})" if error is not None else ""))
        results.append({"case": case, "verdict": verdict, "max_mean_rgb_error": error,
                        "native": desktop, "web": browser})
    dump(output / "results.json", results)
    return int(not all(result["verdict"] == "both_gpu" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
