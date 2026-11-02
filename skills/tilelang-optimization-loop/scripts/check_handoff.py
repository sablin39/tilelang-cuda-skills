#!/usr/bin/env python3
"""Validate stage handovers and file identity, not the truth of reported results."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

STAGES = {"baseline", "design", "candidate", "diagnosis", "final", "maintenance"}
KINDS = {"correctness", "benchmark", "lowering", "model", "research", "reproduction"}
STATUSES = {"pass", "fail", "unrun", "unavailable", "inconclusive", "not_applicable"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def artifact(root, path):
    require(isinstance(path, str) and path and not Path(path).is_absolute(),
            "artifact paths must be task-relative")
    resolved = (root / path).resolve()
    require(resolved.is_relative_to(root.resolve()), f"artifact escapes task root: {path}")
    require(resolved.is_file(), f"missing artifact: {path}")
    return {"path": path, "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest()}


def validate(report, root):
    require(report.get("schema_version") == 1, "unsupported schema_version")
    require(report.get("stage") in STAGES, "unknown stage")
    require(report.get("status") in {"complete", "incomplete", "blocked", "inconclusive"},
            "unknown stage status")
    require(isinstance(report.get("candidate"), str) and report["candidate"], "missing candidate")
    inputs = report.get("inputs")
    require(isinstance(inputs, list) and inputs, "inputs must include source and task context")
    for item in inputs:
        require(item == artifact(root, item["path"]), f"stale input: {item['path']}")
    require(isinstance(report.get("checks"), list), "checks must be a list")
    checks = {}
    for check in report["checks"]:
        cid = check.get("id")
        require(isinstance(cid, str) and cid and cid not in checks, "missing/duplicate check id")
        checks[cid] = check
        require(check.get("kind") in KINDS, f"{cid}: unknown check kind")
        require(check.get("status") in STATUSES, f"{cid}: unknown check status")
        require(type(check.get("required")) is bool, f"{cid}: required must be boolean")
        require(isinstance(check.get("summary"), str) and check["summary"].strip(),
                f"{cid}: explain result or missing evidence")
        status = check["status"]
        command, code = check.get("command"), check.get("exit_code")
        require(command is None or isinstance(command, str) and command.strip(), f"{cid}: invalid command")
        require(code is None or type(code) is int, f"{cid}: invalid exit_code")
        require(isinstance(check.get("artifacts"), list), f"{cid}: artifacts must be a list")
        for item in check["artifacts"]:
            require(item == artifact(root, item["path"]), f"{cid}: stale artifact {item['path']}")
        if status == "pass":
            require(check["artifacts"], f"{cid}: pass needs evidence artifacts")
            require(command is None or code == 0, f"{cid}: passing command needs exit_code 0")
            if check["kind"] in {"correctness", "benchmark", "reproduction"}:
                require(command is not None and code == 0, f"{cid}: execution check needs successful command")
        if status in {"unrun", "not_applicable"}:
            require(code is None, f"{cid}: unrun/not_applicable has an exit code")
        if check["required"]:
            require(status != "not_applicable", f"{cid}: a required check cannot be not_applicable")
    def references(ids):
        require(isinstance(ids, list) and all(isinstance(i, str) and i in checks for i in ids),
                "evidence references must name existing check ids")
    skills = report.get("skills")
    require(isinstance(skills, list) and skills, "record the selected skills or why unavailable")
    for skill in skills:
        require(all(isinstance(skill.get(k), str) and skill[k].strip()
                    for k in ("name", "question", "finding")), "skill needs name, question and finding")
        require(skill.get("status") in {"used", "reused", "unavailable", "not_applicable"},
                "unknown skill status")
        references(skill.get("evidence"))
        if skill["status"] in {"used", "reused"}:
            require(skill["evidence"], "used/reused skill needs an evidence reference")
    decision = report.get("decision", {})
    action = decision.get("action")
    require(action in {"continue", "promote", "retain", "reject", "blocked"}, "unknown decision")
    require(isinstance(decision.get("reason"), str) and decision["reason"].strip(), "decision needs reason")
    references(decision.get("evidence"))
    next_step = decision.get("next")
    if next_step is not None:
        require(next_step.get("stage") in STAGES and isinstance(next_step.get("question"), str)
                and next_step["question"].strip(), "next needs stage and decision question")
        require(isinstance(next_step.get("skills"), list) and next_step["skills"]
                and all(isinstance(s, str) and s for s in next_step["skills"]), "next needs routed skills")
    if action == "continue":
        require(next_step is not None, "continue needs a next stage")
    if action == "promote":
        require(report["status"] == "complete", "promotion needs a completed stage")
        require(all(c["status"] == "pass" for c in checks.values() if c["required"]),
                "promotion has unresolved required checks")
        consumed = [checks[cid] for cid in decision["evidence"]]
        for kind in ("correctness", "benchmark"):
            require(any(c["kind"] == kind and c["status"] == "pass" for c in consumed),
                    f"promotion must consume passing {kind} evidence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fingerprint", "check"))
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        if args.action == "fingerprint":
            print(json.dumps([artifact(args.root, p) for p in args.paths], indent=2))
        else:
            for path in args.paths:
                validate(json.loads(Path(path).read_text()), args.root)
                print(f"PASS: {path} (structure and file identity; results need review)")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
