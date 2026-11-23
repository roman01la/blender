#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Check Cycles WebGPU triangle emitters against desktop CPU Cycles."""

import argparse
from pathlib import Path

from check_glossy_transparency import alpha_rmse
from check_surface_scattering import rgb_rmse
from run import case_id, dump, positive, run_side


def make_case(name, sampling="FRONT_BACK", **settings):
    case = {"node": ("ShaderNodeBsdfPrincipled" if settings.get("glossy") else
                     "ShaderNodeBsdfDiffuse"),
            "settings": {}, "output": "BSDF",
            "triangle_emitter": {"sampling": sampling, **settings}}
    if settings.get("camera_hit") or settings.get("texture") == "pixel-wireframe":
        case["include_pixels"] = True
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


def transparent_emitter_cases():
    common = {"transparent_emission": "backface", "camera_hidden": True}
    yield make_case("Front sampled transparent", **common)
    yield make_case("Back sampled emission", **common, reverse_facing=True)
    yield make_case("Back emission with flat light distribution", "AUTO", **common,
                    reverse_facing=True, light_tree=False)
    yield make_case("Back-facing directional sky emission", "AUTO", **common, reverse_facing=True,
                    texture="sky", light_tree=False, strength=20)
    yield make_case("Simultaneous Transparent and Emission", transparent_emission="partial",
                    camera_hidden=True)
    yield make_case("Textured transparent emitter beside Bump", **common, reverse_facing=True,
                    texture="image", receiver_bump=True)
    yield make_case("Pixel Wireframe sampled-emitter guard", **common, reverse_facing=True,
                    texture="pixel-wireframe")
    yield make_case("Camera-only Wireframe emission", texture="pixel-wireframe", camera_hit=True,
                    camera_only=True)
    camera = {"transparent_emission": "backface", "camera_hit": True}
    yield make_case("Camera-transparent control", "NONE", **camera, camera_hidden=True)
    yield make_case("Front camera transparent", "NONE", **camera)
    yield make_case("Back camera emission", "NONE", **camera, reverse_facing=True)


def transparent_emitter_signals(rows, side):
    def image(name):
        return rows[name][side].get("image")

    front, back, partial, flat, sky, hidden, visible, backlit = [image(name) for name in (
        "Front sampled transparent", "Back sampled emission", "Simultaneous Transparent and Emission",
        "Back emission with flat light distribution", "Back-facing directional sky emission",
        "Camera-transparent control", "Front camera transparent", "Back camera emission")]
    if not all((front, back, partial, flat, sky, hidden, visible, backlit)):
        return {}
    return {"back_light_red_delta": back["mean_rgb"][0] - front["mean_rgb"][0],
            "partial_light_red_delta": partial["mean_rgb"][0] - front["mean_rgb"][0],
            "flat_light_red_delta": flat["mean_rgb"][0] - front["mean_rgb"][0],
            "sky_light_differs": max(abs(a - b) for a, b in zip(sky["mean_rgb"], front["mean_rgb"], strict=True)),
            "front_camera_alpha_rmse": alpha_rmse(hidden, visible),
            "back_camera_alpha_rmse": alpha_rmse(hidden, backlit)}


def coated_emitter_cases():
    camera = {"camera_hit": True, "camera_only": True, "strength": 1.1}
    sampled = {"camera_hidden": True, "strength": 8}
    for weight, label in ((0, "uncoated"), (1, "coated")):
        yield make_case(f"Camera {label}", "NONE", principled_coat=weight, **camera)
        yield make_case(f"Light Tree {label}", principled_coat=weight, **sampled)
        yield make_case(f"Flat distribution {label}", "AUTO", principled_coat=weight,
                        light_tree=False, **sampled)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-build", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=positive, default=150)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--transparent-emission", action="store_true",
                      help="compare directional emission mixed with Transparent")
    mode.add_argument("--coated-emission", action="store_true",
                      help="compare camera and sampled Principled Coat attenuation")
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
    if args.transparent_emission:
        cases = list(transparent_emitter_cases())
    if args.coated_emission:
        cases = list(coated_emitter_cases())
    native = run_side(args, output, "cycles", "native", "CPU", 0, cases)
    browser = run_side(args, output, "cycles", "web", "WEBGPU", 0, cases)
    rows = []
    for case in cases:
        cpu_case = native.get(case["id"], {})
        gpu_case = browser.get(case["id"], {})
        verdict, delta = compare(cpu_case, gpu_case)
        settings = case["triangle_emitter"]
        if settings.get("texture") == "pixel-wireframe" and not settings.get("camera_only"):
            if (verdict == "cpu_fallback" and cpu_case["fingerprint"] == gpu_case["fingerprint"] and
                    gpu_case["status"] == "completed" and
                    any("ray differentials for sampled mesh emission on Probe triangle" in line
                        for line in gpu_case["evidence"])):
                error = rgb_rmse(cpu_case["image"], gpu_case["image"])
                verdict = "guarded" if error <= 0.005 else "image_mismatch"
            elif verdict == "pass":
                verdict = "guard_unproven"
        alpha_error = None
        if verdict == "pass" and settings.get("camera_hit"):
            alpha_error = alpha_rmse(cpu_case["image"], gpu_case["image"])
            if alpha_error > 0.005:
                verdict = "alpha_mismatch"
        print(f'{case["name"]} {verdict}' + (f" ({delta:.6f})" if delta is not None else ""))
        rows.append({"case": case, "verdict": verdict, "max_mean_rgb_error": delta, "alpha_rmse": alpha_error,
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
    if args.transparent_emission:
        by_name = {row["case"]["name"]: row for row in rows}
        signal = {side: transparent_emitter_signals(by_name, side) for side in ("native", "web")}
        signal_pass = all(values and values["back_light_red_delta"] > 0.06 and
                          values["partial_light_red_delta"] > 0.03 and values["flat_light_red_delta"] > 0.06 and
                          values["sky_light_differs"] > 0.02 and values["front_camera_alpha_rmse"] <= 0.001 and
                          values["back_camera_alpha_rmse"] > 0.04 for values in signal.values())
    if args.coated_emission:
        by_name = {row["case"]["name"]: row for row in rows}

        def red(name, side):
            return by_name[name][side].get("image", {}).get("mean_rgb", [0])[0]

        signal = {
            side: {route: red(f"{route} uncoated", side) - red(f"{route} coated", side)
                   for route in ("Camera", "Light Tree", "Flat distribution")}
            for side in ("native", "web")
        }
        signal_pass = all(value > 0.001 for values in signal.values() for value in values.values())
    print(f"CPU fixture signal {'pass' if signal_pass else 'failed'} {signal}")
    dump(output / "results.json", {"signal": signal, "signal_pass": signal_pass, "cases": rows})
    return int(not signal_pass or any(row["verdict"] not in {"pass", "guarded"} for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
