#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Compare homogeneous absorption with desktop CPU Cycles."""

import argparse
import math
from pathlib import Path

from run import case_id, dump, positive, run_side


def make_case(mode):
    case = {"node": "ShaderNodeVolumeAbsorption", "output": "Volume", "settings": {},
            "absorption_fixture": mode}
    return {"id": case_id(case), **case}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-build", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=positive, default=180)
    parser.add_argument("--native-only", action="store_true")
    args = parser.parse_args()
    args.lighting = "area-world"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    modes = ("outside-clear", "outside-tint", "inside-clear", "inside-tint", "shadow-clear", "shadow-tint",
             "requested-majorant")
    cases = [make_case(mode) for mode in modes]
    native = run_side(args, output, "cycles", "native", "CPU", 0, cases)
    web = {} if args.native_only else run_side(args, output, "cycles", "web", "WEBGPU", 0, cases)
    rows = {}
    for mode, case in zip(modes, cases, strict=True):
        cpu = native.get(case["id"], {})
        browser = web.get(case["id"], {})
        valid = cpu.get("status") == "completed" and cpu.get("execution") == "cpu_control"
        verdict = "pass" if valid and args.native_only else "browser_unproven"
        error = None
        rmse = None
        if mode == "requested-majorant" and valid and browser.get("execution") == "cpu_fallback":
            verdict = "pass"
        elif (mode != "requested-majorant" and valid and browser.get("execution") == "gpu" and
              browser.get("fingerprint") == cpu.get("fingerprint")):
            error = max(abs(a - b) for a, b in zip(cpu["image"]["mean_rgb"],
                                                  browser["image"]["mean_rgb"], strict=True))
            cpu_pixels = bytes.fromhex(cpu["image"]["rgba_hex"])
            browser_pixels = bytes.fromhex(browser["image"]["rgba_hex"])
            squares = [(a - b) ** 2 for i, (a, b) in enumerate(
                zip(cpu_pixels, browser_pixels, strict=True)) if i % 4 != 3]
            rmse = math.sqrt(math.fsum(squares) / len(squares)) / 255
            verdict = "pass" if error <= 0.005 and rmse <= 0.01 else "image_mismatch"
        rows[mode] = {"case": case, "verdict": verdict, "max_mean_rgb_error": error,
                      "rmse": rmse, "native": cpu, "web": browser}
        print(mode, verdict, error, rmse, flush=True)

    def red(mode):
        return rows[mode]["native"].get("image", {}).get("mean_rgb", [0])[0]

    signals = {mode: red(f"{mode}-clear") - red(f"{mode}-tint")
               for mode in ("outside", "inside", "shadow")}
    signal_pass = all(value > 0.01 for value in signals.values())
    print(f"Desktop absorption signal {signals}")
    dump(output / "results.json", {"signals": signals, "signal_pass": signal_pass, "cases": rows})
    return int(not signal_pass or any(record["verdict"] != "pass" for record in rows.values()))


if __name__ == "__main__":
    raise SystemExit(main())
