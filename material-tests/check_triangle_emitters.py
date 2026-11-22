#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Check Cycles WebGPU triangle emitters against desktop CPU Cycles."""

import argparse
from pathlib import Path

from run import case_id, dump, positive, run_side


def make_case(name, sampling="FRONT_BACK", **settings):
    case = {"node": ("ShaderNodeBsdfPrincipled" if settings.get("glossy") else
                     "ShaderNodeBsdfDiffuse"),
            "settings": {}, "output": "BSDF",
            "triangle_emitter": {"sampling": sampling, **settings}}
    return {"id": case_id(case), "name": name, **case}


def compare(reference, browser):
    if reference.get("execution") != "cpu_control":
        return "cpu_unproven", None
    if browser.get("execution") != "gpu":
        return browser.get("execution", "browser_unproven"), None
    if reference["fingerprint"] != browser["fingerprint"]:
        return "different_fixture", None
    delta = max(abs(a - b) for a, b in zip(reference["image"]["mean_rgb"],
                                          browser["image"]["mean_rgb"], strict=True))
    # Compare the mean of the center pixels to tolerate individual Monte Carlo outliers.
    return ("pass" if delta <= 0.005 else "image_mismatch"), delta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-build", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=positive, default=150)
    args = parser.parse_args()
    args.lighting = "area-world"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    unlit = make_case("Unlit control", strength=0)
    two_sided = make_case("Two-sided sampling")
    glossy_hit = make_case("Glossy hit with sampling disabled", "NONE", glossy=True)
    glossy_unlit = make_case("Glossy unlit control", "NONE", glossy=True, strength=0)
    image = make_case("Camera-hidden UV image emitter", texture="image", camera_hidden=True)
    flipped = make_case("Flipped emitter UVs", texture="image", camera_hidden=True, flip_uv=True)
    noise = make_case("Camera-hidden Generated Noise emitter", texture="noise", camera_hidden=True)
    cases = [
        unlit,
        make_case("Front sampling", "FRONT"),
        make_case("Back sampling", "BACK"),
        two_sided,
        make_case("Transformed instances", instanced=True),
        make_case("Mixed analytic and mesh lights", point_fill=True),
        make_case("Triangle-only Light Tree", mesh_only=True),
        glossy_unlit,
        glossy_hit,
        make_case("Glossy hit with MIS", glossy=True),
        image,
        flipped,
        noise,
        make_case("Textured light back sampling", "BACK", texture="image", camera_hidden=True),
        make_case("Textured light instances", texture="image", camera_hidden=True, instanced=True),
        make_case("Mesh-only textured Light Tree", texture="image", camera_hidden=True,
                  mesh_only=True),
        make_case("Textured glossy hit with MIS", texture="noise", glossy=True),
    ]
    native = run_side(args, output, "cycles", "native", "CPU", 0, cases)
    browser = run_side(args, output, "cycles", "web", "WEBGPU", 0, cases)
    rows = []
    for case in cases:
        cpu_case = native.get(case["id"], {})
        gpu_case = browser.get(case["id"], {})
        verdict, delta = compare(cpu_case, gpu_case)
        print(f'{case["name"]} {verdict}' + (f" ({delta:.6f})" if delta is not None else ""))
        rows.append({"case": case, "verdict": verdict, "max_mean_rgb_error": delta,
                     "native": cpu_case, "web": gpu_case})

    def red(case):
        return native.get(case["id"], {}).get("image", {}).get("mean_rgb", [0])[0]

    signal = {
        "sampled_emitter_red_delta": red(two_sided) - red(unlit),
        "unsampled_glossy_emitter_red_delta": red(glossy_hit) - red(glossy_unlit),
        "uv_image_red_delta": red(image) - red(unlit),
        "flip_uv_red_delta": abs(red(image) - red(flipped)),
        "generated_noise_red_delta": red(noise) - red(unlit),
    }
    signal_pass = (signal["sampled_emitter_red_delta"] > 0.08 and
                   signal["unsampled_glossy_emitter_red_delta"] > 0.08 and
                   signal["uv_image_red_delta"] > 0.03 and signal["flip_uv_red_delta"] > 0.005 and
                   signal["generated_noise_red_delta"] > 0.03)
    print(f"CPU fixture signal {'pass' if signal_pass else 'failed'} {signal}")
    dump(output / "results.json", {"signal": signal, "signal_pass": signal_pass, "cases": rows})
    return int(not signal_pass or any(row["verdict"] != "pass" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
