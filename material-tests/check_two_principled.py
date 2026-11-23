#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Compare two opaque Principled branches against desktop Cycles."""

import argparse
from pathlib import Path

from check_mixed_glossy import guided_denoise_proven
from check_mixed_principled import check_guard
from check_surface_scattering import compare, rgb_rmse
from run import case_id, dump, positive, run_side


def settings(guided):
    yield "mixed-flat", {"normal": "flat"}
    yield "mixed-mapped", {"normal": "mapped"}
    yield "mixed-reverse-order", {"reverse": True}
    yield "mixed-independent-normals", {"normal": "independent"}
    yield "mixed-stock-ggx", {"distribution": "GGX"}
    yield "mixed-linked-parameters", {"linked_lobes": True, "factor": "Z"}
    yield "mixed-factor-0", {"normal": "flat", "factor": 0}
    yield "mixed-factor-1", {"normal": "flat", "factor": 1}
    yield "mixed-normal-map-pass", {"normal_pass": True}
    yield "mixed-normal-flat-pass", {"normal": "flat", "normal_pass": True}
    yield "mixed-glossy-color", {"glossy_color_pass": True}
    # The browser request serializes whole-number settings as integers.
    yield "mixed-coat-first", {"normal": "flat", "coat_weights": (0.54, 0)}
    yield "mixed-coat-second", {"normal": "flat", "coat_weights": (0, 0.54)}
    yield "mixed-zero-specular-first", {"normal": "flat", "zero_specular_first": True}
    yield "mixed-zero-specular-first-coat", {
        "normal": "flat", "zero_specular_first": True, "coat_weights": (0.54, 0)}
    layered = {"bump_second": True, "linked_lobes": True, "strong_specular": True}
    yield "mixed-textured-bump", layered
    anisotropic = {**layered, "anisotropy": True}
    yield "mixed-anisotropy", anisotropic
    yield "mixed-anisotropy-coat", {**anisotropic, "coat_weights": (0, 0.35)}
    yield "mixed-anisotropy-coat-glossy-color", {
        **anisotropic, "coat_weights": (0, 0.54), "glossy_color_pass": True}
    yield "mixed-guard-colored-coat", {
        "normal": "flat", "coat_weights": (0, 0.54), "outside": "coat-tint"}
    if guided:
        yield "mixed-guided-oidn", {"normal": "independent", "denoise": True}
        yield "mixed-anisotropy-coat-guided-oidn", {
            **anisotropic, "coat_weights": (0, 0.54), "denoise": True}


def sheen_settings(guided):
    fabric = {"linked_lobes": True, "factor": 0.5}
    zero = {**fabric, "sheen_weights": (0, 0)}
    second = {**fabric, "sheen_weights": (0, 0.2)}
    first = {**fabric, "sheen_weights": (0.2, 0)}
    yield "sheen-zero", zero
    yield "sheen-second", second
    yield "sheen-second-reverse-order", {**second, "reverse": True}
    yield "sheen-first-mapped", first
    yield "sheen-second-roughness", {**second, "sheen_roughness": 0.22}
    yield "sheen-zero-normal-pass", {**zero, "normal_pass": True}
    yield "sheen-first-normal-pass", {**first, "normal_pass": True}
    if guided:
        yield "sheen-guided-oidn", {**second, "denoise": True}


def metallic_settings(guided):
    surface = {"linked_lobes": True, "factor": 0.5}
    paired = {**surface, "metallic_pair": True, "linked_metallic": True}
    coated = {**paired, "coat_weights": (0, 0.125)}
    yield "metal-zero", surface
    yield "metal-paired", paired
    yield "metal-paired-coat", coated
    yield "metal-paired-coat-strong", {**paired, "coat_weights": (0, 0.5)}
    yield "metal-paired-coat-reverse-order", {**coated, "reverse": True}
    yield "metal-literal-coat-flat", {**coated, "linked_metallic": False, "normal": "flat"}
    yield "metal-paired-coat-glossy-color", {**coated, "glossy_color_pass": True}
    if guided:
        yield "metal-paired-coat-guided-oidn", {**coated, "denoise": True}


def cases(samples, guided=False, sheen_only=False, metallic_only=False):
    if metallic_only:
        graphs = metallic_settings(guided)
    elif sheen_only:
        graphs = sheen_settings(guided)
    else:
        graphs = settings(guided)
    for name, graph in graphs:
        case = {"node": "ShaderNodeOutputMaterial", "settings": {}, "output": "Surface",
                "cycles_samples": samples, "include_pixels": True, "resolution": 64,
                "two_principled": graph}
        yield {"id": case_id(case), "name": name, **case}


def check_signals(rows, sheen_only=False, metallic_only=False):
    if metallic_only:
        targets = (("metal-paired", "metal-zero"),
                   ("metal-paired-coat-strong", "metal-paired"))
    elif sheen_only:
        targets = (("sheen-second", "sheen-zero"),
                   ("sheen-first-mapped", "sheen-zero"),
                   ("sheen-second-roughness", "sheen-second"))
    else:
        targets = (("mixed-mapped", "mixed-flat"),
                   ("mixed-independent-normals", "mixed-mapped"),
                   ("mixed-normal-map-pass", "mixed-normal-flat-pass"),
                   ("mixed-linked-parameters", "mixed-mapped"),
                   ("mixed-factor-0", "mixed-factor-1"),
                   ("mixed-coat-first", "mixed-flat"),
                   ("mixed-coat-second", "mixed-flat"),
                   ("mixed-zero-specular-first-coat", "mixed-zero-specular-first"),
                   ("mixed-anisotropy", "mixed-textured-bump"),
                   ("mixed-anisotropy-coat", "mixed-anisotropy"))
    signals = {}
    for first, second in targets:
        for side in ("native", "web"):
            a, b = (rows[name][side].get("image") for name in (first, second))
            signals[f"{side} {first} / {second}"] = rgb_rmse(a, b) if a and b else None
    return signals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-build", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=positive, default=128)
    parser.add_argument("--timeout", type=positive, default=360)
    parser.add_argument("--guided", action="store_true")
    suite = parser.add_mutually_exclusive_group()
    suite.add_argument("--sheen-only", action="store_true",
                       help="compare paired Principled Sheen with desktop CPU Cycles")
    suite.add_argument("--metallic-only", action="store_true",
                       help="compare paired metallic Principled with desktop CPU Cycles")
    args = parser.parse_args()
    args.lighting = "area-hdri"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    fixtures = list(cases(args.samples, args.guided, args.sheen_only, args.metallic_only))
    cpu = run_side(args, output, "cycles", "native", "CPU", 0, fixtures)
    browser = run_side(args, output, "cycles", "web", "WEBGPU", 0, fixtures)
    rows = {}
    for fixture in fixtures:
        desktop, web = (results.get(fixture["id"], {}) for results in (cpu, browser))
        graph = fixture["two_principled"]
        guard = bool(graph.get("outside"))
        verdict, error = (check_guard if guard else compare)(desktop, web)
        if guard and verdict == "pass":
            verdict = "unexpected_gpu_support"
        if (verdict == "pass" and (graph.get("normal_pass") or graph.get("glossy_color_pass")) and
                error > 0.001):
            verdict = "image_mismatch"
        if verdict == "pass" and graph.get("denoise") and not guided_denoise_proven(desktop, web):
            verdict = "denoising_unproven"
        name = fixture["name"]
        rows[name] = {"case": fixture, "verdict": verdict, "rgb_rmse": error,
                      "native": desktop, "web": web}
        print(name, verdict, error, flush=True)

    signals = check_signals(rows, args.sheen_only, args.metallic_only)
    inconclusive = [label for label, value in signals.items() if value is None or value <= 0.002]
    print("Fixture distances", signals, flush=True)
    dump(output / "results.json", {"signals": signals, "cases": rows})
    return int(bool(inconclusive) or any(row["verdict"] not in {"pass", "guarded"}
                                        for row in rows.values()))


if __name__ == "__main__":
    raise SystemExit(main())
