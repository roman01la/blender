#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Check Cycles' WebGPU shader ray queries against desktop Metal."""

import argparse
import math
from pathlib import Path

from run import case_id, compare, dump, positive, run_side


def make_case(name, node, output, fixture, **settings):
    case = {"node": f"ShaderNode{node}", "settings": settings, "output": output,
            "query_fixture": fixture, "include_pixels": True}
    return {"id": case_id(case), "name": name, **case}


def query_cases():
    return [
        make_case("Raycast hit", "Raycast", "Is Hit", "other-hit"),
        make_case("Raycast miss", "Raycast", "Is Hit", "miss"),
        make_case("Raycast self hit", "Raycast", "Self Hit", "self-hit"),
        make_case("Raycast other hit", "Raycast", "Self Hit", "other-hit"),
        make_case("Raycast only local", "Raycast", "Self Hit", "other-hit", only_local=True),
        make_case("Raycast distance", "Raycast", "Hit Distance", "other-hit"),
        make_case("Raycast position", "Raycast", "Hit Position", "other-hit"),
        make_case("Raycast normal", "Raycast", "Hit Normal", "other-hit"),
        make_case("AO unoccluded", "AmbientOcclusion", "AO", "clear"),
        make_case("AO occluded", "AmbientOcclusion", "AO", "occluded"),
        make_case("AO only local", "AmbientOcclusion", "AO", "occluded", only_local=True),
        make_case("AO inside color", "AmbientOcclusion", "Color", "clear", inside=True),
        make_case("Bevel zero radius", "Bevel", "Normal", "zero-radius"),
        make_case("Bevel smoothed normal", "Bevel", "Normal", "smoothed"),
        make_case("Noise with AO", "AmbientOcclusion", "AO", "extended-occluded"),
        make_case("Noise with Raycast hit", "Raycast", "Is Hit", "extended-other-hit"),
        make_case("Noise with Raycast miss", "Raycast", "Is Hit", "extended-miss"),
        make_case("Noise with Bevel", "Bevel", "Normal", "extended-smoothed"),
        make_case("Cubic image with Bevel", "Bevel", "Normal", "extended-cubic-smoothed"),
        make_case("Smart image with Bevel", "Bevel", "Normal", "extended-smart-smoothed"),
        make_case("Tube projection with AO", "AmbientOcclusion", "AO", "extended-tube-occluded"),
        make_case("Sphere projection with AO", "AmbientOcclusion", "AO",
                  "extended-sphere-occluded"),
        make_case("Box projection with AO", "AmbientOcclusion", "AO", "extended-box-occluded"),
        make_case("Equirectangular projection with AO", "AmbientOcclusion", "AO",
                  "extended-environment-occluded"),
        make_case("Mirror Ball projection with AO", "AmbientOcclusion", "AO",
                  "extended-mirror-occluded"),
        make_case("Gabor 2D with Raycast hit", "Raycast", "Is Hit", "extended-gabor2d-other-hit"),
        make_case("Gabor 2D with Raycast miss", "Raycast", "Is Hit", "extended-gabor2d-miss"),
        make_case("Gabor 3D with AO", "AmbientOcclusion", "AO", "extended-gabor3d-clear"),
        make_case("Gabor Phase with AO", "AmbientOcclusion", "AO", "extended-gaborphase-clear"),
        make_case("Gabor Intensity with Bevel", "Bevel", "Normal",
                  "extended-gaborintensity-smoothed"),
        make_case("Voronoi 1D F2 with Raycast", "Raycast", "Is Hit",
                  "extended-voronoi1d-other-hit"),
        make_case("Voronoi 2D Smooth F1 with AO", "AmbientOcclusion", "AO",
                  "extended-voronoi2d-clear"),
        make_case("Voronoi 3D Chebychev with Bevel", "Bevel", "Normal",
                  "extended-voronoi3d-smoothed"),
        make_case("Voronoi 4D Manhattan with Raycast", "Raycast", "Is Hit",
                  "extended-voronoi4d-other-hit"),
        make_case("Voronoi Distance to Edge with Raycast", "Raycast", "Is Hit",
                  "extended-voronoiedge-other-hit"),
        make_case("Voronoi N-Sphere Radius with Raycast", "Raycast", "Is Hit",
                  "extended-voronoiradius-other-hit"),
        make_case("Gabor and Voronoi with Raycast", "Raycast", "Is Hit",
                  "extended-combined-other-hit"),
        make_case("Gabor and cubic Box with Raycast", "Raycast", "Is Hit",
                  "extended-gaborimage-other-hit"),
        make_case("Voronoi and cubic Box with Raycast", "Raycast", "Is Hit",
                  "extended-voronoiimage-other-hit"),
        make_case("1D Multifractal Noise and Gabor with Raycast", "Raycast", "Is Hit",
                  "extended-noise1-gabor2d-other-hit"),
        make_case("2D Hybrid Multifractal Noise and Voronoi with AO", "AmbientOcclusion", "AO",
                  "extended-noise2-voronoi2d-clear"),
        make_case("3D Ridged Multifractal Noise, Gabor and Voronoi with Raycast", "Raycast", "Is Hit",
                  "extended-noise3-combined-other-hit"),
        make_case("4D Hetero Terrain Noise and Voronoi with Bevel", "Bevel", "Normal",
                  "extended-noise4-voronoi4d-smoothed"),
        make_case("4D Hetero Terrain Noise, Gabor and cubic Box with Raycast", "Raycast", "Is Hit",
                  "extended-noise4-gaborimage-other-hit"),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-build", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=positive, default=200)
    args = parser.parse_args()
    args.lighting = "area-world"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = query_cases()
    native = run_side(args, output, "cycles", "native", "METAL", 0, cases)
    browser = run_side(args, output, "cycles", "web", "WEBGPU", 0, cases)
    rows = []
    for case in cases:
        desktop = native.get(case["id"], {})
        gpu = browser.get(case["id"], {})
        verdict = compare(desktop, gpu)
        delta = rmse = None
        if verdict == "both_gpu":
            delta = max(abs(a - b) for a, b in zip(desktop["image"]["mean_rgb"],
                                                  gpu["image"]["mean_rgb"], strict=True))
            reference, actual = (
                bytes.fromhex(render["image"]["rgba_hex"]) for render in (desktop, gpu)
            )
            rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(reference, actual, strict=True))
                             / len(reference)) / 255
            verdict = "pass" if delta <= 0.005 and rmse <= 0.01 else "image_mismatch"
        print(case["name"], verdict, delta, rmse)
        rows.append({"case": case, "verdict": verdict, "max_mean_rgb_error": delta,
                     "rgba_rmse": rmse, "native": desktop, "web": gpu})

    def red(name):
        return next(row["native"] for row in rows if row["case"]["name"] == name).get(
            "image", {}).get("mean_rgb", [0])[0]

    signal = {
        "hit": red("Raycast hit") - red("Raycast miss"),
        "self_hit": red("Raycast self hit") - red("Raycast other hit"),
        "only_local_hit": red("Raycast only local") - red("Raycast other hit"),
        "ao": red("AO unoccluded") - red("AO occluded"),
        "ao_only_local": red("AO only local") - red("AO occluded"),
        "gabor_hit": red("Gabor 2D with Raycast hit") - red("Gabor 2D with Raycast miss"),
        "gabor_texture": red("Raycast hit") - red("Gabor 2D with Raycast hit"),
    }
    combined_signal = red("Voronoi 4D Manhattan with Raycast") - red("Gabor and Voronoi with Raycast")
    noise_signal = (red("Gabor and Voronoi with Raycast") -
                    red("3D Ridged Multifractal Noise, Gabor and Voronoi with Raycast"))
    meaningful = min(signal.values()) > 0.1 and combined_signal > 0.02 and noise_signal > 0.02
    signal["combined_textures"] = combined_signal
    signal["noise_with_combined_textures"] = noise_signal
    print("Desktop fixture signal", signal)
    dump(output / "results.json", {"signal": signal, "signal_pass": meaningful, "cases": rows})
    return int(not meaningful or any(row["verdict"] != "pass" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
