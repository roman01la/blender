#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Check Cycles Hair and Christensen–Burley against desktop CPU Cycles."""

import argparse
from math import sqrt
from pathlib import Path

from run import case_id, dump, positive, run_side


def cases(samples):
    fixtures = (
        ("Hair Reflection", "BsdfHair", "BSDF", {}),
        ("Hair Transmission", "BsdfHair", "BSDF", {"component": "Transmission"}),
        ("Principled Hair Chiang", "BsdfHairPrincipled", "BSDF", {}),
        ("Principled Hair Huang", "BsdfHairPrincipled", "BSDF", {"model": "HUANG"}),
        ("Principled Hair Absorption", "BsdfHairPrincipled", "BSDF", {"parametrization": "ABSORPTION"}),
        ("Principled Hair Melanin", "BsdfHairPrincipled", "BSDF", {"parametrization": "MELANIN"}),
        ("Principled Burley", "BsdfPrincipled", "BSDF", {"subsurface_method": "BURLEY"}),
        ("Subsurface Burley", "SubsurfaceScattering", "BSSRDF", {"falloff": "BURLEY"}),
    )
    for name, node, output, settings in fixtures:
        fixture = {"node": f"ShaderNode{node}", "settings": settings, "output": output,
                   "cycles_samples": samples, "include_pixels": True}
        yield {"id": case_id(fixture), "name": name, **fixture}


def compare(cpu, browser):
    if cpu.get("execution") != "cpu_control":
        return "desktop_cpu_unproven", None
    if browser.get("execution") != "gpu":
        return browser.get("execution", "browser_unproven"), None
    if cpu["fingerprint"] != browser["fingerprint"]:
        return "different_fixture", None
    cpu_pixels = bytes.fromhex(cpu["image"]["rgba_hex"])
    web_pixels = bytes.fromhex(browser["image"]["rgba_hex"])
    squared_rgb_error = sum(
        ((a - b) / 255) ** 2
        for i, (a, b) in enumerate(zip(cpu_pixels, web_pixels, strict=True))
        if i % 4 != 3
    )
    rgb_rmse = sqrt(squared_rgb_error / (3 * len(cpu_pixels) / 4))
    return ("pass" if rgb_rmse <= 0.005 else "image_mismatch"), rgb_rmse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-build", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=positive, default=128)
    parser.add_argument("--timeout", type=positive, default=240)
    args = parser.parse_args()
    args.lighting = "mixed-hdri"

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fixtures = list(cases(args.samples))
    cpu = run_side(args, output, "cycles", "native", "CPU", 0, fixtures)
    browser = run_side(args, output, "cycles", "web", "WEBGPU", 0, fixtures)
    rows = []
    for fixture in fixtures:
        reference = cpu.get(fixture["id"], {})
        web = browser.get(fixture["id"], {})
        verdict, error = compare(reference, web)
        print(f'{fixture["name"]} {verdict}' +
              (f" (RGB RMSE {error:.6f})" if error is not None else ""))
        rows.append({"case": fixture, "verdict": verdict, "rgb_rmse": error,
                     "native": reference, "web": web})

    dump(output / "results.json", {"samples": args.samples, "cases": rows})
    return int(any(row["verdict"] != "pass" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
