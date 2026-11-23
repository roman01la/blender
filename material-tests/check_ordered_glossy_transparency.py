#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Compare ordered sharp Glossy, rough Glossy, and Transparent with CPU Cycles."""

import argparse
from math import sqrt
from pathlib import Path

from check_mixed_glossy import guided_denoise_proven
from check_surface_scattering import compare, rgb_rmse
from run import case_id, dump, positive, run_side


def cases(samples, guided=False):
    settings = (
        ("opaque-sharp-pair", {"opaque": True}),
        ("opaque-rough-control", {"opaque": True, "rough_only": True}),
        ("camera-select", {"camera": True}),
        ("camera-reverse", {"camera": True, "reverse": True}),
        ("simultaneous", {"alpha": True}),
        ("simultaneous-reverse", {"alpha": True, "reverse": True}),
        ("varying", {"alpha": True, "factor": "Z", "sharp": "BECKMANN", "rough": "GGX"}),
        ("glossy-color", {"camera": True, "glossy_color_pass": True}),
        ("tinted-shadow", {"camera": True, "shadow": "tinted"}),
        ("opaque-shadow-control", {"camera": True, "shadow": "opaque"}),
        ("linear-image-shadow", {"shadow": "tinted", "image_alpha": True,
                                 "depth": 16, "interpolation": "Linear"}),
        ("cubic-image-shadow", {"shadow": "tinted", "image_alpha": True,
                                "depth": 16, "interpolation": "Cubic"}),
        ("cubic-image-shadow-ao", {"shadow": "tinted", "image_alpha": True, "shadow_ao": True,
                                    "depth": 16, "interpolation": "Cubic"}),
        ("cubic-image-shadow-voronoi", {"shadow": "tinted", "image_alpha": True,
                                        "shadow_voronoi": True, "depth": 16,
                                        "interpolation": "Cubic"}),
        ("cubic-image-shadow-voronoi-normalized", {"shadow": "tinted", "image_alpha": True,
                                                   "shadow_voronoi": True, "normalize": True,
                                                   "depth": 16, "interpolation": "Cubic"}),
    )
    if guided:
        settings += (("guided", {"camera": True, "denoise": True}),)
    for name, options in settings:
        node = "ShaderNodeBsdfDiffuse" if options.get("shadow") else "ShaderNodeOutputMaterial"
        fixture = {"node": node, "settings": {}, "output": "BSDF" if options.get("shadow") else "Surface",
                   "cycles_samples": samples, "include_pixels": True, "resolution": 64,
                   "glossy_transparency": {"ordered_glossy_mix": True, **options}}
        yield {"id": case_id(fixture), "name": name, **fixture}


def alpha_rmse(cpu, web):
    native_alpha = bytes.fromhex(cpu["rgba_hex"])[3::4]
    browser_alpha = bytes.fromhex(web["rgba_hex"])[3::4]
    squared_error = sum(((a - b) / 255) ** 2
                        for a, b in zip(native_alpha, browser_alpha, strict=True))
    return sqrt(squared_error / len(native_alpha))


def check_signals(rows, side):
    def distance(first, second):
        a, b = (rows[name][side].get("image") for name in (first, second))
        return rgb_rmse(a, b) if a and b else 0.0

    return {
        "sharp_lobe": distance("opaque-sharp-pair", "opaque-rough-control"),
        "tinted_shadow": distance("tinted-shadow", "opaque-shadow-control"),
        "cubic_shadow": distance("cubic-image-shadow", "linear-image-shadow"),
        "shadow_ao_default": distance("cubic-image-shadow", "cubic-image-shadow-ao"),
        "shadow_voronoi": distance("cubic-image-shadow", "cubic-image-shadow-voronoi"),
        "shadow_voronoi_normalized": distance("cubic-image-shadow-voronoi",
                                              "cubic-image-shadow-voronoi-normalized"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-build", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=positive, default=128)
    parser.add_argument("--timeout", type=positive, default=240)
    parser.add_argument("--guided", action="store_true")
    args = parser.parse_args()
    args.lighting = "area-hdri"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    fixtures = list(cases(args.samples, args.guided))
    cpu = run_side(args, output, "cycles", "native", "CPU", 0, fixtures)
    browser = run_side(args, output, "cycles", "web", "WEBGPU", 0, fixtures)
    rows = {}
    for fixture in fixtures:
        desktop = cpu.get(fixture["id"], {})
        web = browser.get(fixture["id"], {})
        verdict, error = compare(desktop, web)
        alpha_error = None
        if verdict == "pass":
            alpha_error = alpha_rmse(desktop["image"], web["image"])
            if alpha_error > 0.005:
                verdict = "alpha_mismatch"
        if (verdict == "pass" and fixture["glossy_transparency"].get("denoise") and
                not guided_denoise_proven(desktop, web)):
            verdict = "denoising_unproven"
        name = fixture["name"]
        rows[name] = {"case": fixture, "verdict": verdict, "rgb_rmse": error,
                      "alpha_rmse": alpha_error, "native": desktop, "web": web}
        print(name, verdict, error, alpha_error, flush=True)

    signals = {side: check_signals(rows, side) for side in ("native", "web")}
    signal_pass = all(
        distances["shadow_ao_default"] < 0.0005 and
        all(distances[name] > 0.005 for name in
            ("sharp_lobe", "tinted_shadow", "cubic_shadow", "shadow_voronoi")) and
        distances["shadow_voronoi_normalized"] > 0.002
        for distances in signals.values()
    )
    print("Fixture distances", signals, flush=True)
    dump(output / "results.json", {"signals": signals, "signal_pass": signal_pass, "cases": rows})
    return int(not signal_pass or any(row["verdict"] != "pass" for row in rows.values()))


if __name__ == "__main__":
    raise SystemExit(main())
