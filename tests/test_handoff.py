import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("handoff", Path(__file__).resolve().parents[1] / "skills/tilelang-optimization-loop/scripts/check_handoff.py")
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for file in ("candidate.py", "context.json", "correctness.log", "timing.json"):
            (self.root / file).write_text("fixture")
        self.report = {
            "schema_version": 1, "stage": "candidate", "candidate": "c01", "status": "complete",
            "inputs": [handoff.artifact(self.root, f) for f in ("candidate.py", "context.json")],
            "skills": [{"name": "profiling-tilelang-programs", "status": "used", "question": "Stable win?",
                        "finding": "Improvement exceeds variation on all required cases.", "evidence": ["time"]}],
            "checks": [{"id": cid, "kind": kind, "status": "pass", "required": True,
                        "command": command, "exit_code": 0, "summary": "Pass fixture.",
                        "artifacts": [handoff.artifact(self.root, file)]}
                       for cid, kind, command, file in (
                           ("correct", "correctness", "python check.py", "correctness.log"),
                           ("time", "benchmark", "python bench.py", "timing.json"))],
            "decision": {"action": "promote", "reason": "Meets objective and regression limits.",
                         "evidence": ["correct", "time"], "next": None}}

    def test_valid_promotion_and_reuse(self):
        handoff.validate(self.report, self.root)
        self.report["skills"][0]["status"] = "reused"
        handoff.validate(self.report, self.root)

    def test_stale_source_context_and_evidence(self):
        for file in ("candidate.py", "context.json", "timing.json"):
            with self.subTest(file=file):
                path = self.root / file
                path.write_text("changed")
                with self.assertRaises(ValueError):
                    handoff.validate(self.report, self.root)
                path.write_text("fixture")

    def test_noisy_unrun_failed_checks_block_promotion(self):
        for status in ("inconclusive", "unrun", "fail", "unavailable"):
            with self.subTest(status=status):
                report = copy.deepcopy(self.report)
                report["checks"][1].update(status=status, exit_code=None)
                with self.assertRaises(ValueError):
                    handoff.validate(report, self.root)

    def test_rejected_failure_is_completed_stage(self):
        self.report["checks"][0].update(status="fail", exit_code=1)
        self.report["checks"][1].update(status="unrun", command=None, exit_code=None, artifacts=[])
        self.report["decision"].update(action="reject", evidence=["correct"])
        handoff.validate(self.report, self.root)

    def test_success_exit_and_consumed_evidence_required(self):
        self.report["checks"][0]["exit_code"] = 1
        with self.assertRaises(ValueError):
            handoff.validate(self.report, self.root)
        self.report["checks"][0]["exit_code"] = 0
        self.report["decision"]["evidence"] = ["time"]
        with self.assertRaises(ValueError):
            handoff.validate(self.report, self.root)

    def test_unknown_reference_and_escape_refused(self):
        self.report["skills"][0]["evidence"] = ["invented"]
        with self.assertRaises(ValueError):
            handoff.validate(self.report, self.root)
        with self.assertRaises(ValueError):
            handoff.artifact(self.root, "../outside")

    def test_continue_routes_next_question(self):
        self.report["decision"]["action"] = "continue"
        with self.assertRaises(ValueError):
            handoff.validate(self.report, self.root)
        self.report["decision"]["next"] = {
            "stage": "final", "question": "Reproduce exact winner?", "skills": ["tilelang-optimization-loop"]}
        handoff.validate(self.report, self.root)


if __name__ == "__main__":
    unittest.main()
