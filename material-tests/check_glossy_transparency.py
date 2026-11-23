#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Verify external RGBA16 textures and Glossy mixed with Transparent against CPU Cycles."""

import argparse
from math import sqrt
from pathlib import Path

from check_mixed_glossy import guided_denoise_proven
from check_surface_scattering import compare, rgb_rmse
from run import case_id, dump, positive, run_side


def cases(samples):
    for channel in ("color", "alpha"):
        for depth in (8, 16):
            yield fixture_case(
                f"precision-{channel}-rgba{depth}", samples, {"precision": channel, "depth": depth})
    yield fixture_case("precision-color-nearest", samples,
                       {"precision": "color", "depth": 16, "interpolation": "Closest"})

    for distribution in ("GGX", "BECKMANN"):
        yield fixture_case(f"{distribution.lower()}-constant", samples, {
            "distribution": distribution, "reverse": distribution == "BECKMANN", "alpha": True})
    yield fixture_case("beckmann-image-reverse", samples,
                       {"distribution": "BECKMANN", "reverse": True, "depth": 8, "alpha": True})

    classroom = {"classroom": True, "distribution": "GGX", "depth": 16}
    for depth in (8, 16):
        yield fixture_case(f"classroom-rgba{depth}", samples, {**classroom, "depth": depth, "alpha": True})
    yield fixture_case("classroom-normal", samples, {**classroom, "normal_pass": True})
    yield fixture_case("classroom-guided", samples, {**classroom, "denoise": True})
    yield fixture_case("rgba16-shadow", samples,
                       {"distribution": "GGX", "depth": 16, "shadow": "mixed", "tinted": True})
    yield fixture_case("opaque-shadow", samples, {"distribution": "GGX", "shadow": "opaque"})


def fixture_case(name, samples, settings):
    case = {"node": "ShaderNodeOutputMaterial", "settings": {}, "output": "Surface",
            "cycles_samples": samples, "include_pixels": True, "resolution": 64,
            "glossy_transparency": settings}
    return {"id": case_id(case), "name": name, **case}


def alpha_rmse(first, second):
    a = bytes.fromhex(first["rgba_hex"])[3::4]
    b = bytes.fromhex(second["rgba_hex"])[3::4]
    return sqrt(sum(((x - y) / 255) ** 2 for x, y in zip(a, b, strict=True)) / len(a))


def signals(rows, side):
    def image(name):
        return rows[name][side].get("image")

    def distance(a, b):
        first, second = image(a), image(b)
        return rgb_rmse(first, second) if first and second else 0.0

    mixed, opaque = image("rgba16-shadow"), image("opaque-shadow")
    classroom = image("classroom-rgba16")
    return {
        "red_low_bits_rmse": distance("precision-color-rgba8", "precision-color-rgba16"),
        "alpha_low_bits_rmse": distance("precision-alpha-rgba8", "precision-alpha-rgba16"),
        "filter_rmse": distance("precision-color-rgba16", "precision-color-nearest"),
        "shadow_lift": sum(a - b for a, b in zip(mixed["mean_rgb"], opaque["mean_rgb"], strict=True)) / 3
                       if mixed and opaque else 0.0,
        "classroom_mean_alpha": classroom["mean_alpha"] if classroom else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-build", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=positive, default=128)
    parser.add_argument("--timeout", type=positive, default=420)
    args = parser.parse_args()
    args.lighting = "area-hdri"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    fixtures = list(cases(args.samples))
    cpu = run_side(args, output, "cycles", "native", "CPU", 0, fixtures)
    web = run_side(args, output, "cycles", "web", "WEBGPU", 0, fixtures)
    rows = {}
    for case in fixtures:
        desktop, browser = (results.get(case["id"], {}) for results in (cpu, web))
        verdict, rgb_error = compare(desktop, browser)
        settings = case["glossy_transparency"]
        alpha_error = None
        if verdict == "pass":
            if settings.get("alpha"):
                alpha_error = alpha_rmse(desktop["image"], browser["image"])
                if alpha_error > 0.005:
                    verdict = "alpha_mismatch"
            if settings.get("normal_pass") and rgb_error > 0.001:
                verdict = "normal_mismatch"
            if settings.get("denoise") and not guided_denoise_proven(desktop, browser):
                verdict = "denoising_unproven"
        name = case["name"]
        print(name, verdict, rgb_error, alpha_error, flush=True)
        rows[name] = {"case": case, "verdict": verdict, "rgb_rmse": rgb_error,
                      "alpha_rmse": alpha_error, "native": desktop, "web": browser}

    measures = {side: signals(rows, side) for side in ("native", "web")}
    for side, values in measures.items():
        print(side, values, flush=True)
    signal_pass = all(0.1 < value["classroom_mean_alpha"] < 0.95 and
                      all(signal > 0.01 for label, signal in value.items()
                          if label != "classroom_mean_alpha") for value in measures.values())
    dump(output / "results.json", {"signals": measures, "signal_pass": signal_pass, "cases": rows})
    return int(not signal_pass or any(row["verdict"] != "pass" for row in rows.values()))


if __name__ == "__main__":
    raise SystemExit(main())
