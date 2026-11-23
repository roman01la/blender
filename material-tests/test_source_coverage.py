# SPDX-License-Identifier: GPL-2.0-or-later

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from source_coverage import BLENDER, LEDGER, coverage


class SourceCoverageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ledger = json.loads(LEDGER.read_text())
        cls.baseline = coverage(BLENDER, cls.ledger)

    def test_every_pinned_entry_has_a_status_and_stock_index(self):
        self.assertEqual([], self.baseline["errors"])
        registries = self.baseline["registries"]
        svm = {row["name"]: row for row in registries["ShaderNodeType"]["members"]}
        self.assertEqual(svm["NODE_TEX_COORD"]["value"] + 1, svm["NODE_TEX_COORD_DERIVATIVE"]["value"])
        kernels = {row["name"]: row for row in registries["DeviceKernel"]["members"]}
        self.assertEqual("unsupported", kernels["DEVICE_KERNEL_INTEGRATOR_INTERSECT_MNEE"]["status"])
        self.assertEqual(kernels["DEVICE_KERNEL_FILM_CONVERT_DEPTH"]["value"] + 1,
                         kernels["DEVICE_KERNEL_FILM_CONVERT_DEPTH_HALF_RGBA"]["value"])

    def test_added_stock_opcode_and_changed_enum_value_are_drift(self):
        paths = {member["source"] for details in self.baseline["registries"].values()
                 for member in details["members"]}
        for registry in self.ledger["registries"].values():
            for group in registry["groups"]:
                for reference in group.get("evidence", []):
                    if reference.startswith("blender/"):
                        paths.add(reference.removeprefix("blender/").partition("#")[0])
        with tempfile.TemporaryDirectory() as directory:
            blender = Path(directory)
            for relative in paths:
                output = blender / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(BLENDER / relative, output)
            template = blender / "intern/cycles/kernel/svm/node_types_template.h"
            old = "SHADER_NODE_TYPE(NODE_SCENE_TIME)"
            template.write_text(template.read_text().replace(
                old, old + "\nSHADER_NODE_TYPE_DERIVATIVE(NODE_COVERAGE_PROBE)"))
            types = blender / "intern/cycles/kernel/types.h"
            types.write_text(types.read_text().replace("PASS_DEPTH = 32", "PASS_DEPTH = 33"))
            report = coverage(blender, self.ledger)

        errors = report["errors"]
        self.assertIn("ShaderNodeType: untracked NODE_COVERAGE_PROBE", errors)
        self.assertIn("ShaderNodeType: untracked NODE_COVERAGE_PROBE_DERIVATIVE", errors)
        self.assertTrue(any(error.startswith("PassType: stock names, ordering, or values changed")
                            for error in errors))
        self.assertFalse(any(error.startswith("PassType: untracked") for error in errors))

    def test_stale_evidence_and_duplicate_rules_are_drift(self):
        ledger = copy.deepcopy(self.ledger)
        group = ledger["registries"]["DeviceKernel"]["groups"][0]
        group["evidence"][0] += "_does_not_exist"
        group["members"].append(group["members"][0])
        errors = coverage(BLENDER, ledger)["errors"]
        self.assertIn(f"Missing coverage evidence: {group['evidence'][0]}", errors)
        self.assertIn(f"DeviceKernel: duplicate ledger entry {group['members'][0]}", errors)
