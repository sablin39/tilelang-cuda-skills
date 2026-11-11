"""Calibration evidence, reuse boundaries, held-out selection and failure records."""

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "skills/tilelang/profiling-tilelang-programs/scripts"
sys.path.insert(0, str(SCRIPTS))
import calibration as cal
import measure_calibration as collector
sys.path.pop(0)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "kernel.py").write_text("# preserved candidate\n")
        self.device = self.make_device()

    def make_device(self, name="device.json", samples=None, **changes):
        profile = {
            "kind": "device", "created_at": name, "command": "actual-probe-command",
            "context": {"device": {"name": "test device", "uuid": "GPU-test", "compute_capability": [10, 3]},
                        "software": {"driver": "test-driver", "compiler": "test-compiler"},
                        "operating": {"clock_policy": "uncontrolled", "isolation": "exclusive"}},
            "protocol": {"backend": "cuda_events", "cache": "repeated inputs"},
            "measurements": [{"id": "copy", "status": "measured", "correctness": "pass",
                              "quantity": "copy_effective_bandwidth", "unit": "byte/s",
                              "scope": {"working_set_bytes": 1024}, "samples": samples or [99., 100., 101.]}],
        }
        profile.update(changes)
        return cal.seal(profile, self.root / name, self.root)

    def case(self, name, size, family, predicted, actual, partition, group="memory"):
        return {"id": name, "partition": partition, "group": group, "workload": {"elements": size},
                "configuration": {"family": family}, "predicted_ns": predicted,
                "status": "measured", "correctness": "pass", "samples_ns": [actual * .999, actual, actual * 1.001]}

    def make_module(self, cases=None, name="module.json", **changes):
        profile = {"kind": "module", "module": "test-module", "regime": {"elements": [1, 2]},
                   "context": deepcopy(self.device["context"]), "protocol": {"backend": "cuda_events"},
                   "command": "actual-module-command", "refresh_reason": "initial",
                   "base_profile": cal.fingerprint(self.root, "device.json"),
                   "inputs": [cal.snapshot_input(self.root, "kernel.py")],
                   "cases": cases or [self.case("fit", 1, "a", 50, 100, "fit"),
                                      self.case("held-a", 2, "a", 100, 200, "holdout"),
                                      self.case("held-b", 2, "b", 150, 300, "holdout")]}
        profile.update(changes)
        return cal.seal(profile, self.root / name, self.root)

    def assess(self, profile, **kwargs):
        return cal.assess_module(profile, self.root, .1, .05, .1, **kwargs)

    def reseal_in_memory(self, profile):
        profile["profile_id"] = cal.profile_id(profile)
        return profile

    def test_profile_hash_summary_and_exclusive_write(self):
        self.assertEqual(cal.validate(self.device, self.root), self.device)
        with self.assertRaises(FileExistsError):
            self.make_device()
        changed = deepcopy(self.device)
        changed["context"]["device"]["name"] = "different"
        with self.assertRaisesRegex(ValueError, "content hash"):
            cal.validate(changed, self.root)
        changed = deepcopy(self.device)
        changed["measurements"][0]["summary"]["median"] = 999
        with self.assertRaisesRegex(ValueError, "summary differs"):
            cal.validate(self.reseal_in_memory(changed), self.root)

    def test_unavailable_is_not_zero_or_a_usable_measurement(self):
        unavailable = {"id": "latency", "status": "unavailable", "reason": "no probe for this path",
                       "quantity": "dependent_latency", "unit": "ns", "scope": {"path": "unknown"}}
        result = self.make_device("missing.json", measurements=[unavailable])
        self.assertEqual(result["measurements"][0]["status"], "unavailable")
        unavailable["samples"] = [0]
        with self.assertRaisesRegex(ValueError, "eligible rate"):
            self.make_device("invalid.json", measurements=[unavailable])
        for sample in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                cal.summarize([sample])

    def test_source_snapshot_survives_live_edit_and_detects_tampering(self):
        profile = self.make_module()
        original = profile["inputs"][0]
        (self.root / "kernel.py").write_text("# next candidate\n")
        cal.validate(profile, self.root)
        self.assertEqual(self.assess(profile, current_inputs=True)["status"], "stale")
        self.assertEqual((self.root / original["artifact"]["path"]).read_text(), "# preserved candidate\n")
        (self.root / original["artifact"]["path"]).write_text("tampered")
        with self.assertRaisesRegex(ValueError, "changed artifact"):
            cal.validate(profile, self.root)

    def test_concurrent_workers_share_complete_immutable_source_snapshots(self):
        (self.root / "kernel.py").write_text("# shared source\n" * 4096)
        with ThreadPoolExecutor(max_workers=8) as pool:
            refs = list(pool.map(lambda _: cal.snapshot_input(self.root, "kernel.py"), range(32)))
        self.assertTrue(all(ref == refs[0] for ref in refs))
        self.assertEqual((self.root / refs[0]["artifact"]["path"]).read_bytes(), (self.root / "kernel.py").read_bytes())
        self.assertFalse(list(self.root.rglob(".calibration-*")))

    def test_references_cannot_escape_task_root(self):
        with tempfile.TemporaryDirectory() as elsewhere:
            outside = Path(elsewhere) / "external.py"
            outside.write_text("outside")
            (self.root / "alias.py").symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "outside"):
                cal.snapshot_input(self.root, "alias.py")

    def test_module_base_and_parent_are_verified(self):
        profile = self.make_module()
        child = self.make_module(name="child.json", parent=cal.fingerprint(self.root, "module.json"), refresh_reason="staging changed")
        self.assertEqual(cal.validate(child, self.root)["parent"]["path"], "module.json")
        other_context = deepcopy(self.device["context"])
        other_context["software"]["compiler"] = "new-compiler"
        with self.assertRaisesRegex(ValueError, "context differs"):
            self.make_module(name="new.json", context=other_context)
        (self.root / "device.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "changed artifact"):
            cal.validate(profile, self.root)

    def test_reuse_matches_only_checked_scopes_and_environment(self):
        fresh = self.make_device("fresh.json", samples=[100., 101., 102.])
        result = cal.reuse(self.device, fresh, .05, .1)
        self.assertEqual(result["measurements"][0]["status"], "reusable")
        fresh["measurements"][0]["scope"]["working_set_bytes"] = 2048
        self.assertEqual(cal.reuse(self.device, fresh, .05, .1)["measurements"][0]["status"], "unverified")
        fresh["context"]["device"]["uuid"] = "another-GPU"
        self.assertFalse(cal.reuse(self.device, fresh, .05, .1)["context_compatible"])
        self.assertFalse(cal.reuse(self.device, self.device, .05, .1)["context_compatible"])

    def test_reuse_rejects_unknown_identity_noise_and_drift(self):
        drifted = self.make_device("drift.json", samples=[79., 80., 81.])
        self.assertEqual(cal.reuse(self.device, drifted, .05, .1)["measurements"][0]["status"], "drifted")
        noisy = self.make_device("noisy.json", samples=[50., 100., 150.])
        self.assertEqual(cal.reuse(self.device, noisy, .05, .1)["measurements"][0]["status"], "inconclusive")
        noisy["context"]["device"]["uuid"] = None
        self.assertFalse(cal.reuse(self.device, noisy, .05, .1)["context_compatible"])

    def test_duplicate_scopes_and_holdout_leakage_are_rejected(self):
        rows = deepcopy(self.device["measurements"] * 2)
        rows[1] = deepcopy(rows[1])
        rows[1]["id"] = "different-id-same-scope"
        with self.assertRaisesRegex(ValueError, "duplicate measurement scope"):
            self.make_device("duplicate.json", measurements=rows)
        rows = [self.case("fit", 1, "a", 50, 100, "fit"), self.case("held", 1, "a", 50, 101, "holdout")]
        with self.assertRaisesRegex(ValueError, "hold out configurations"):
            self.make_module(cases=rows)

    def test_fit_uses_only_fit_rows_and_checks_selection(self):
        profile = self.make_module()
        report = self.assess(profile, require_ranking=True)
        self.assertEqual(report["status"], "validated")
        self.assertEqual(report["groups"][0]["median_correction"], 2)
        self.assertEqual(report["selections"][0]["nominal_choice"], "held-a")
        profile["cases"][1]["summary"]["median"] = 400
        report = self.assess(profile, require_ranking=True)
        self.assertEqual(report["groups"][0]["median_correction"], 2)
        self.assertEqual(report["status"], "inconclusive")
        self.assertGreater(report["selections"][0]["regret"], .3)

    def test_cross_family_selection_uses_fitted_predictions(self):
        profile = self.make_module(cases=[
            self.case("fit-a", 1, "a", 10, 100, "fit", "a"),
            self.case("fit-b", 1, "b", 50, 50, "fit", "b"),
            self.case("held-a", 2, "a", 20, 200, "holdout", "a"),
            self.case("held-b", 2, "b", 100, 100, "holdout", "b")])
        report = self.assess(profile, require_ranking=True)
        self.assertEqual(report["status"], "validated")
        self.assertEqual(report["selections"][0]["nominal_choice"], "held-b")

    def test_overlapping_timing_and_missing_ranking_are_inconclusive(self):
        profile = self.make_module(cases=[
            self.case("fit", 1, "a", 50, 100, "fit"),
            self.case("held-a", 2, "a", 100, 200, "holdout"),
            self.case("held-b", 2, "b", 100, 200.1, "holdout")])
        self.assertEqual(self.assess(profile, require_ranking=True)["status"], "inconclusive")
        profile["cases"].pop()
        self.assertEqual(self.assess(profile)["status"], "validated")
        self.assertEqual(self.assess(profile, require_ranking=True)["ranking_status"], "unassessed")
        self.assertEqual(self.assess(profile, require_ranking=True)["status"], "inconclusive")

    def test_noisy_fit_cannot_qualify_a_ranking(self):
        profile = self.make_module()
        profile["cases"][0]["summary"] = cal.summarize([50, 100, 150])
        report = self.assess(profile, require_ranking=True)
        self.assertEqual(report["status"], "inconclusive")
        self.assertEqual(report["ranking_status"], "inconclusive")

    def test_failed_cases_survive_collection_and_cannot_validate_a_model(self):
        (self.root / "factory.py").write_text('''
from dataclasses import dataclass
@dataclass
class Shape:
    n: int = 1
def calibration_cases(base):
    def setup():
        return {"run": lambda: None, "check": lambda: False}
    return {"module": "failing-module", "regime": {"test": True}, "cases": [
        {"id": "wrong", "group": "all", "partition": "fit", "workload": {"n": Shape().n},
         "configuration": {"a": 1}, "predicted_ns": 1, "setup": setup}]}
''')
        args = SimpleNamespace(root=self.root, base="device.json", factory="factory.py", input=[], parent=None,
                               reason="test failure", warmup=1, iterations=1, repeats=3, device=0)
        profile = collector.collect_module(None, args, self.device["context"], [])
        profile["command"] = "collector test"
        sealed = cal.seal(profile, self.root / "failed.json", self.root)
        self.assertEqual(sealed["cases"][0]["failure_stage"], "correctness")
        self.assertNotIn("samples_ns", sealed["cases"][0])
        self.assertEqual(self.assess(sealed)["status"], "inconclusive")

    def test_target_coverage_keeps_partial_results_and_proxy_roles_explicit(self):
        target = {"spec": {"id": "test-target"}, "facts": {
            "bandwidth": {"dram": {"value": 1e12, "unit": "byte/s", "origin": "vendor", "source": "spec", "conditions": "nominal"}},
            "throughput": {"bf16": {"status": "unavailable", "conditions": "not supplied"}}}}
        required = [{"fact": "bandwidth.dram", "work": 1000, "unit": "byte/s"},
                    {"fact": "throughput.bf16", "work": 1000, "unit": "flop/s"}]
        report = cal.target_coverage(target, required, "proxy")
        self.assertEqual(report["coverage"], "partial")
        self.assertEqual(len(report["available"]), 1)
        self.assertFalse(report["prediction_qualified"])
        required[1]["work"] = 0
        self.assertEqual(cal.target_coverage(target, required, "reference")["coverage"], "complete_for_requested_work")
        required[0]["unit"] = "GB/s"
        self.assertEqual(cal.target_coverage(target, required, "local")["missing"][0]["reason"], "rate unit differs")
        target["facts"]["bandwidth"]["dram"]["value"] = 0
        self.assertEqual(cal.target_coverage(target, required, "local")["coverage"], "partial")

    def test_cli_fails_for_partial_reuse_and_checks_without_gpu_import(self):
        command = [sys.executable, str(SCRIPTS / "calibration.py"), "--root", str(self.root)]
        result = subprocess.run(command + ["check", "device.json"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

        self.assertEqual(json.loads(result.stdout)["scope"], "structure_and_artifact_identity")
        result = subprocess.run(command + ["reuse", str(self.root / "device.json"), str(self.root / "device.json"),
                                           "--max-relative-drift", ".1", "--max-relative-spread", ".1"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        result = subprocess.run([sys.executable, str(SCRIPTS / "measure_calibration.py"), "module", "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_telemetry_matches_bare_cuda_uuid_but_not_mig_identity(self):
        response = SimpleNamespace(stdout="GPU-abc, 600.0, 1500, 4000, 1000, 200, 45, 0, 12\n")
        with patch.object(collector.subprocess, "run", return_value=response):
            self.assertEqual(collector.telemetry("abc")["values"]["driver_version"], "600.0")
            self.assertEqual(collector.telemetry("MIG-abc")["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
