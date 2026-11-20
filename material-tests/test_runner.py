# SPDX-License-Identifier: GPL-2.0-or-later

"""Regression checks for the device evidence used by the material matrix."""

import json
import unittest

from run import compare, parse_log


def marker(event, **kwargs):
    return "MATERIAL_TEST " + json.dumps({"event": event, **kwargs})


def finished(case, backend):
    return marker("result", id=case, status="completed", fingerprint=case,
                  gpu={"backend": backend, "renderer": "hardware"},
                  image={"magenta": 0})


class EvidenceTest(unittest.TestCase):
    def test_backend_abort_reports_gpu_failure_without_render_result(self):
        evidence = (
            "Blender's WebGPU batch backend does not implement using a shader "
            "with a missing storage texture binding yet."
        )
        results, _ = parse_log([marker("begin", id="volume"), evidence], "eevee", "WEBGPU")
        self.assertEqual(results["volume"]["execution"], "gpu_failure")

    def test_reported_webgpu_cannot_hide_cpu_fallback(self):
        lines = [marker("begin", id="a"), marker("connected", id="a", fingerprint="a"),
                 "Cycles WebGPU using CPU: unsupported material",
                 "Path tracing on: WebGPU (WebGPU) (display) [WEBGPU]",
                 finished("a", "WEBGPU")]
        result, _ = parse_log(lines, "cycles", "WEBGPU")
        native = {"execution": "gpu", "fingerprint": "a"}
        self.assertEqual(compare(native, result["a"]), "browser_cpu_fallback")

    def test_native_hardware_trace(self):
        lines = [marker("begin", id="a"),
                 "Path tracing on: Apple GPU (Metal) (display) [METAL_Apple GPU]",
                 finished("a", "METAL")]
        result, _ = parse_log(lines, "cycles", "METAL")
        self.assertEqual(result["a"]["execution"], "gpu")

    def test_webgpu_requires_a_real_dispatch(self):
        lines = [marker("begin", id="a"), "Path tracing on: WebGPU [WEBGPU]",
                 finished("a", "WEBGPU")]
        result, _ = parse_log(lines, "cycles", "WEBGPU")
        self.assertEqual(result["a"]["execution"], "unproven")
        lines.insert(1, "Cycles WebGPU stock scheduler completed a tile")
        result, _ = parse_log(lines, "cycles", "WEBGPU")
        self.assertEqual(result["a"]["execution"], "gpu")

    def test_delayed_shader_failure_is_attributed_to_its_material(self):
        tag = "Probe_0123456789abcdef"
        lines = [marker("begin", id="a"),
                 marker("connected", id="a", fingerprint="a", material=tag),
                 finished("a", "WEBGPU"), marker("begin", id="b"),
                 f"Error while parsing WGSL for MA{tag}_shadow_mesh",
                 finished("b", "WEBGPU")]
        result, _ = parse_log(lines, "eevee", "WEBGPU")
        self.assertEqual(result["a"]["execution"], "gpu_failure")
        self.assertEqual(result["b"]["execution"], "gpu")

    def test_dawn_render_pipeline_and_command_buffer_failures(self):
        tag = "Probe_0123456789abcdef"
        lines = [marker("begin", id="a"),
                 marker("connected", id="a", fingerprint="a", material=tag),
                 finished("a", "WEBGPU"), marker("begin", id="b"),
                 f'GPUDevice: "[Invalid RenderPipeline "MA{tag}_deferred_mesh"] is invalid"',
                 'GPUDevice: "[Invalid CommandBuffer] is invalid due to a previous error."',
                 finished("b", "WEBGPU")]
        result, _ = parse_log(lines, "eevee", "WEBGPU")
        self.assertEqual(result["a"]["execution"], "gpu_failure")
        self.assertEqual(result["b"]["execution"], "gpu_failure")

    def test_changed_graph_cannot_confirm_regression(self):
        native = {"execution": "gpu", "fingerprint": "desktop"}
        web = {"execution": "cpu_fallback", "fingerprint": "browser"}
        self.assertEqual(compare(native, web), "different_graph")


if __name__ == "__main__":
    unittest.main()
