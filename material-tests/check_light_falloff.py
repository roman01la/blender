#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Compare stock analytic Light Falloff outputs with desktop CPU Cycles."""

import argparse
from pathlib import Path

from check_surface_scattering import compare, rgb_rmse
from run import case_id, dump, positive, run_side


def light_settings():
    for kind in ("POINT", "AREA", "SUN"):
        yield f"{kind.lower()}-control", {"kind": kind}
    for output, smooth in (("Quadratic", 0), ("Quadratic", 8), ("Linear", 8), ("Constant", 8)):
        yield f"point-{output.lower()}-{smooth}", {"kind": "POINT", "falloff": output, "smooth": smooth}
    yield "area-quadratic", {"kind": "AREA", "falloff": "Quadratic", "smooth": 8}
    for output in ("Linear", "Constant"):
        yield f"sun-{output.lower()}", {"kind": "SUN", "falloff": output, "smooth": 8}
    for kind in ("POINT", "AREA"):
        camera = {"kind": kind, "camera_visible": True, "light_tree": kind == "AREA"}
        yield f"{kind.lower()}-camera-control", camera
        yield f"{kind.lower()}-camera-quadratic", {**camera, "falloff": "Quadratic", "smooth": 8}
    yield "point-colored-threshold", {"kind": "POINT", "energy": 6, "color": [0.36, 0.68, 0.44],
                                      "falloff": "Quadratic", "smooth": 8, "light_threshold": 0.04}


def shadow_settings():
    for kind in ("POINT", "AREA"):
        for shadows in (True, False):
            yield f"{kind.lower()}-shadows-{'on' if shadows else 'off'}", {
                "kind": kind, "cast_shadow": shadows, "light_tree": kind == "AREA"}


def mis_settings():
    area = {"kind": "AREA", "light_tree": True}
    for camera, label in ((False, "receiver"), (True, "camera")):
        for mis in (True, False):
            mode = "on" if mis else "off"
            yield f"area-mis-{label}-{mode}", {**area, "camera_visible": camera, "light_mis": mis}
    for mis in (True, False):
        mode = "on" if mis else "off"
        yield f"area-mis-shared-camera-{mode}", {**area, "camera_visible": True,
                                                "light_mis": mis, "fill_radius": 0.15}


def ray_length_settings():
    point = {"kind": "POINT", "energy": 9, "emission_strength": 2.5}
    area = {"kind": "AREA", "light_tree": True, "temperature": 4500, "emission_strength": 3}
    camera = {**area, "camera_visible": True}
    yield "length-point-control", point
    yield "length-point-near", {**point, "ray_length": 4.5, "inverted": True}
    yield "length-point-far", {**point, "ray_length": 4.5, "clamp": False}
    yield "length-point-unit", {**point, "ray_length": None}
    yield "length-area-control", area
    yield "length-area-near", {**area, "ray_length": 5.5, "inverted": True}
    yield "length-area-camera-control", camera
    yield "length-area-camera-far", {**camera, "ray_length": 6.4}


def cases(samples, settings_for_mode):
    for name, settings in settings_for_mode():
        case = {"node": "ShaderNodeBsdfDiffuse", "settings": {}, "output": "BSDF",
                "cycles_samples": samples, "include_pixels": True, "resolution": 64,
                "light_falloff": settings}
        yield {"id": case_id(case), "name": name, **case}


def signal_distances(rows, side, shadows=False, mis=False, ray_length=False):
    if shadows:
        comparisons = (("point-shadows-on", "point-shadows-off"), ("area-shadows-on", "area-shadows-off"))
    elif mis:
        comparisons = (("area-mis-camera-on", "area-mis-camera-off"),
                       ("area-mis-shared-camera-on", "area-mis-shared-camera-off"))
    elif ray_length:
        comparisons = (("length-point-control", "length-point-near"),
                       ("length-point-control", "length-point-far"),
                       ("length-point-control", "length-point-unit"),
                       ("length-area-control", "length-area-near"),
                       ("length-area-camera-control", "length-area-camera-far"))
    else:
        comparisons = (("point-control", "point-quadratic-0"), ("point-control", "point-quadratic-8"),
                       ("point-quadratic-8", "point-linear-8"), ("point-linear-8", "point-constant-8"),
                       ("area-control", "area-quadratic"), ("sun-control", "sun-linear"),
                       ("sun-control", "sun-constant"), ("point-camera-control", "point-camera-quadratic"),
                       ("area-camera-control", "area-camera-quadratic"))
    result = {}
    for first, second in comparisons:
        a, b = (rows[name][side].get("image") for name in (first, second))
        result[second] = rgb_rmse(a, b) if a and b else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-build", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=positive, default=128)
    parser.add_argument("--timeout", type=positive, default=240)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--shadow-toggle", action="store_true", help="compare lamps with disabled shadows")
    mode.add_argument("--lamp-mis", action="store_true", help="compare Area lights with disabled MIS")
    mode.add_argument("--ray-length", action="store_true", help="compare live Ray Length falloff")
    args = parser.parse_args()
    args.lighting = "point"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    settings_for_mode = light_settings
    if args.shadow_toggle:
        settings_for_mode = shadow_settings
    elif args.lamp_mis:
        settings_for_mode = mis_settings
    elif args.ray_length:
        settings_for_mode = ray_length_settings
    fixtures = list(cases(args.samples, settings_for_mode))
    cpu = run_side(args, output, "cycles", "native", "CPU", 0, fixtures)
    browser = run_side(args, output, "cycles", "web", "WEBGPU", 0, fixtures)
    rows = {}
    for case in fixtures:
        desktop, web = (results.get(case["id"], {}) for results in (cpu, browser))
        verdict, error = compare(desktop, web)
        rows[case["name"]] = {"case": case, "verdict": verdict, "rgb_rmse": error,
                              "native": desktop, "web": web}
        print(case["name"], verdict, error, flush=True)
    signals = {side: signal_distances(rows, side, args.shadow_toggle, args.lamp_mis, args.ray_length)
               for side in ("native", "web")}
    matches = {"point-quadratic-0", "sun-linear", "sun-constant", "area-mis-shared-camera-off",
               "length-point-unit"}
    signal_pass = all(error is not None and (error <= 0.001 if label in matches else error > 0.01)
                      for distances in signals.values() for label, error in distances.items())
    if args.lamp_mis:
        for name, row in rows.items():
            reds = (row[side].get("image", {}).get("mean_rgb", [0])[0] for side in ("native", "web"))
            signal_pass &= (all(red < 0.01 for red in reds) if name == "area-mis-camera-off"
                            else all(red > 0.04 for red in reds))
    print("Fixture distances", signals, flush=True)
    dump(output / "results.json", {"signals": signals, "signal_pass": signal_pass, "cases": rows})
    return int(not signal_pass or any(row["verdict"] != "pass" for row in rows.values()))


if __name__ == "__main__":
    raise SystemExit(main())
