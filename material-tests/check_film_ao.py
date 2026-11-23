#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Check the stock Cycles AO film pass through transparent shadow materials."""

import argparse
from pathlib import Path

from check_surface_scattering import compare, rgb_rmse
from run import case_id, dump, positive, run_side


def cases(samples):
    settings = {mode: {"mode": mode} for mode in (
        "clear", "opaque", "tinted", "varying-scalar", "varying-tinted", "backface-open", "backface-closed")}
    for output in ("combined-with-ao", "combined-only"):
        settings[output] = {"mode": "varying-tinted", "output": output}
    for name, graph in settings.items():
        case = {"node": "ShaderNodeBsdfDiffuse", "settings": {}, "output": "BSDF",
                "cycles_samples": samples, "include_pixels": True, "resolution": 64,
                "film_ao": graph}
        yield {"id": case_id(case), "name": name, **case}


def signals(results, side):
    comparisons = {
        "opaque_occlusion": ("clear", "opaque"),
        "constant_tint": ("tinted", "opaque"),
        "varying_scalar": ("varying-scalar", "opaque"),
        "varying_tint": ("varying-tinted", "varying-scalar"),
        "backface_occlusion": ("backface-open", "backface-closed"),
        "combined_delta": ("combined-with-ao", "combined-only"),
    }
    measures = {}
    for label, (first, second) in comparisons.items():
        a, b = (results[name][side].get("image") for name in (first, second))
        measures[label] = rgb_rmse(a, b) if a and b else None
    return measures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-build", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=positive, default=128)
    parser.add_argument("--timeout", type=positive, default=300)
    args = parser.parse_args()
    args.lighting = "area-world"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    fixtures = list(cases(args.samples))
    cpu = run_side(args, output, "cycles", "native", "CPU", 0, fixtures)
    browser = run_side(args, output, "cycles", "web", "WEBGPU", 0, fixtures)
    rows = {}
    for case in fixtures:
        desktop, web = (results.get(case["id"], {}) for results in (cpu, browser))
        verdict, error = compare(desktop, web)
        rows[case["name"]] = {"case": case, "verdict": verdict, "rgb_rmse": error,
                              "native": desktop, "web": web}
        print(case["name"], verdict, error, flush=True)
    measures = {side: signals(rows, side) for side in ("native", "web")}
    signal_pass = all(value is not None and (value <= 0.001 if label == "combined_delta" else value > 0.005)
                      for distances in measures.values() for label, value in distances.items())
    print("Film AO fixture distances", measures, flush=True)
    dump(output / "results.json", {"signals": measures, "signal_pass": signal_pass, "cases": rows})
    return int(not signal_pass or any(row["verdict"] != "pass" for row in rows.values()))


if __name__ == "__main__":
    raise SystemExit(main())
