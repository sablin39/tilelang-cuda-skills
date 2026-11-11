#!/usr/bin/env python3
"""Seal calibration evidence, check reuse, and assess module predictions.

Standard-library only. These checks establish identity and numerical coverage;
they do not prove a supplied command ran or turn attained rates into bounds.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def number(value, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def text(value):
    return isinstance(value, str) and bool(value.strip())


def relative_file(root, path):
    root = Path(root).resolve()
    require(text(path) and not Path(path).is_absolute(), "file references must be task-relative")
    resolved = (root / path).resolve()
    require(resolved.is_relative_to(root) and resolved.is_file(), f"missing/outside task file: {path}")
    return resolved


def fingerprint(root, path):
    file = relative_file(root, path)
    return {"path": path, "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}


def check_ref(root, ref):
    require(isinstance(ref, dict) and ref == fingerprint(root, ref.get("path")),
            "missing or changed artifact")


def write_immutable(path, data, allow_existing=False):
    """Publish complete bytes atomically without replacing existing evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".calibration-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not allow_existing:
                raise
            require(path.read_bytes() == data, "existing snapshot differs from its digest")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def snapshot_input(root, path):
    """Keep the old source bytes when the next experiment edits the live file."""
    file = relative_file(root, path)
    data = file.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    relative = f"calibration/sources/{sha}/{file.name}"
    dest = Path(root) / relative
    write_immutable(dest, data, allow_existing=True)
    return {"source_path": path, "artifact": fingerprint(root, relative)}


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def summarize(samples):
    require(isinstance(samples, list) and samples and all(number(x, True) for x in samples),
            "samples must contain finite positive numbers")
    return {"count": len(samples), "median": statistics.median(samples),
            "p10": percentile(samples, .1), "p90": percentile(samples, .9),
            "interval_kind": "sample_quantiles_not_confidence_interval"}


def profile_id(profile):
    return digest({key: value for key, value in profile.items() if key != "profile_id"})


def validate(profile, root, seen=None):
    require(isinstance(profile, dict), "profile must be an object")
    require(profile.get("schema_version") == 1, "unsupported calibration schema")
    require(profile.get("kind") in {"device", "module"}, "unknown profile kind")
    require(profile.get("profile_id") == profile_id(profile), "profile content hash differs")
    context = profile.get("context", {})
    for name in ("device", "software", "operating"):
        require(isinstance(context.get(name), dict) and context[name], f"missing context.{name}")
    require(text(context["device"].get("name")), "missing device name")
    require(isinstance(profile.get("protocol"), dict) and profile["protocol"], "missing timing protocol")
    require(text(profile.get("command")), "missing measurement command")
    require(text(profile.get("created_at")), "missing creation time")
    for item in profile.get("inputs", []):
        require(text(item.get("source_path")), "missing input source path")
        check_ref(root, item["artifact"])
    for ref in profile.get("artifacts", []):
        check_ref(root, ref)
    if profile["kind"] == "device":
        records = profile.get("measurements")
        require(isinstance(records, list), "measurements must be a list")
        for fact in profile.get("facts", {}).values():
            require(isinstance(fact, dict), "fact must be an object")
            if fact.get("status") == "unavailable":
                require("value" not in fact and text(fact.get("conditions")), "unavailable fact needs a reason, not a value")
            else:
                require("value" in fact and "status" not in fact, "fact needs value or unavailable status")
                require(fact.get("origin") in {"queried", "vendor", "reference", "measured", "derived", "estimated"}, "unknown fact origin")
                require(all(text(fact.get(k)) for k in ("unit", "source", "conditions")), "fact needs unit, source and conditions")
    else:
        require(text(profile.get("module")) and isinstance(profile.get("regime"), dict) and profile["regime"], "module needs an identity and regime")
        require(text(profile.get("refresh_reason")), "module needs a refresh reason")
        require(profile.get("inputs"), "module needs preserved source inputs")
        check_ref(root, profile.get("base_profile"))
        base = load(relative_file(root, profile["base_profile"]["path"]))
        seen = set() if seen is None else set(seen)
        require(profile["profile_id"] not in seen, "cyclic calibration references")
        seen.add(profile["profile_id"])
        validate(base, root, seen)
        require(base["kind"] == "device", "module base must be a device profile")
        require(base["context"] == context, "module context differs from its base; collect/check the current device profile")
        if profile.get("parent"):
            check_ref(root, profile["parent"])
            parent = load(relative_file(root, profile["parent"]["path"]))
            validate(parent, root, seen)
            require(parent["kind"] == "module" and parent["module"] == profile["module"], "parent belongs to another module")
        records = profile.get("cases")
        require(isinstance(records, list) and records, "module needs cases")
    ids, configurations = set(), set()
    for record in records:
        rid = record.get("id")
        require(text(rid) and rid not in ids, "missing or duplicate measurement id")
        ids.add(rid)
        require(record.get("status") in {"measured", "failed", "unavailable"}, f"{rid}: invalid status")
        if profile["kind"] == "device":
            require(all(text(record.get(k)) for k in ("quantity", "unit")), f"{rid}: missing quantity/unit")
            require(isinstance(record.get("scope"), dict) and record["scope"], f"{rid}: missing scope")
            key = measurement_key(record)
            require(key not in configurations, "duplicate measurement scope")
            configurations.add(key)
        else:
            require(record.get("partition") in {"fit", "holdout"}, f"{rid}: missing partition")
            require(text(record.get("group")), f"{rid}: missing fit group")
            for key in ("workload", "configuration"):
                require(isinstance(record.get(key), dict) and record[key], f"{rid}: missing {key}")
            key = digest([record["workload"], record["configuration"]])
            require(key not in configurations, "duplicate configuration; hold out configurations, not repeated samples")
            configurations.add(key)
            require(number(record.get("predicted_ns"), True), f"{rid}: prediction must be frozen and positive")
        samples_key = "samples" if profile["kind"] == "device" else "samples_ns"
        if record["status"] == "measured":
            require(record.get("correctness") == "pass", f"{rid}: measured result needs a passing check")
            summary = summarize(record.get(samples_key))
            require(record.get("summary") == summary, f"{rid}: summary differs from raw samples")
        else:
            require(text(record.get("reason")), f"{rid}: failed/unavailable result needs a reason")
            require(not record.get(samples_key) and "summary" not in record, f"{rid}: unusable result cannot carry an eligible rate")
    return profile


def seal(profile, output, root):
    profile = deepcopy(profile)
    profile.setdefault("schema_version", 1)
    profile.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    records = profile.get("measurements", []) if profile["kind"] == "device" else profile.get("cases", [])
    for record in records:
        if record["status"] == "measured":
            record["summary"] = summarize(record["samples" if profile["kind"] == "device" else "samples_ns"])
    profile["profile_id"] = profile_id(profile)
    validate(profile, root)
    output = Path(output)
    write_immutable(output, (json.dumps(profile, indent=2, allow_nan=False) + "\n").encode())
    return profile


def load(path):
    return json.loads(Path(path).read_text())


def measurement_key(record):
    return digest([record["quantity"], record["unit"], record["scope"]])


def relative_spread(row):
    summary = row["summary"]
    return (summary["p90"] - summary["p10"]) / summary["median"]


def reuse(previous, current, tolerance, max_spread):
    require(number(tolerance), "drift tolerance must be nonnegative")
    require(number(max_spread), "spread tolerance must be nonnegative")
    require(previous["kind"] == current["kind"] == "device", "reuse requires device profiles")
    reasons = []
    if not previous["context"]["device"].get("uuid") or not current["context"]["device"].get("uuid"):
        reasons.append("device UUID unavailable; identity not established")
    if not previous["context"]["software"].get("driver") or not current["context"]["software"].get("driver"):
        reasons.append("driver version unavailable; environment identity incomplete")
    if previous["context"] != current["context"]:
        reasons.append("device, software or operating context changed")
    if previous["protocol"] != current["protocol"]:
        reasons.append("timing protocol changed")
    if previous.get("inputs", []) != current.get("inputs", []):
        reasons.append("declared source/context inputs changed")
    if previous["profile_id"] == current["profile_id"]:
        reasons.append("a profile cannot validate its own reuse")
    anchors = {measurement_key(r): r for r in current["measurements"] if r["status"] == "measured"}
    decisions = []
    for row in previous["measurements"]:
        anchor = anchors.get(measurement_key(row))
        result = {"id": row["id"], "status": "unverified"}
        if reasons:
            result.update(status="context_changed", reasons=reasons)
        elif row["status"] != "measured" or anchor is None:
            result["reason"] = "no matching current measured anchor"
        else:
            drift = abs(anchor["summary"]["median"] / row["summary"]["median"] - 1)
            noisy = any(r["summary"]["count"] < 3 or relative_spread(r) > max_spread for r in (row, anchor))
            result.update(status="inconclusive" if noisy else "reusable" if drift <= tolerance else "drifted", relative_drift=drift,
                          previous_relative_spread=relative_spread(row), current_relative_spread=relative_spread(anchor),
                          current_id=anchor["id"])
            if noisy:
                result["reason"] = "too few samples or spread exceeds the declared tolerance"
        decisions.append(result)
    return {"context_compatible": not reasons, "reasons": reasons, "max_relative_drift": tolerance,
            "max_relative_spread": max_spread,
            "previous_profile": previous["profile_id"], "current_profile": current["profile_id"],
            "measurements": decisions,
            "note": "Unverified rows are not refreshed by an unrelated anchor; drift is not automatically a hardware change."}


def assess_module(profile, root, max_error, max_regret, max_spread, current_inputs=False, require_ranking=False):
    require(profile["kind"] == "module", "assessment requires a module profile")
    require(all(number(x) for x in (max_error, max_regret, max_spread)), "assessment thresholds must be nonnegative")
    changed = []
    if current_inputs:
        for item in profile.get("inputs", []):
            try:
                same = fingerprint(root, item["source_path"])["sha256"] == item["artifact"]["sha256"]
            except (OSError, ValueError):
                same = False
            if not same:
                changed.append(item["source_path"])
    groups, factors = [], {}
    for group in sorted({r["group"] for r in profile["cases"]}):
        rows = [r for r in profile["cases"] if r["group"] == group]
        fit = [r for r in rows if r["partition"] == "fit" and r["status"] == "measured"]
        held = [r for r in rows if r["partition"] == "holdout" and r["status"] == "measured"]
        result = {"group": group, "fit_count": len(fit), "holdout_count": len(held), "status": "incomplete"}
        if not fit or not held:
            result["reason"] = "need measured fit and distinct holdout configurations"
            groups.append(result)
            continue
        factor = statistics.median(r["summary"]["median"] / r["predicted_ns"] for r in fit)
        factors[group] = factor
        residuals = [{"id": r["id"], "prediction_ns": factor * r["predicted_ns"],
                      "uncorrected_prediction_ns": r["predicted_ns"],
                      "observed_ns": r["summary"]["median"],
                      "relative_error": abs(factor * r["predicted_ns"] / r["summary"]["median"] - 1)} for r in held]
        noisy = [r["id"] for r in fit + held if r["summary"]["count"] < 3 or relative_spread(r) > max_spread]
        passed = (all(r["status"] == "measured" for r in rows)
                  and max(r["relative_error"] for r in residuals) <= max_error
                  and not noisy)
        result.update(status="validated" if passed else "inconclusive", median_correction=factor,
                      residuals=residuals, noisy_cases=noisy)
        groups.append(result)
    # Compare corrected predictions across families only for the same workload.
    held = [r for r in profile["cases"] if r["partition"] == "holdout" and r["status"] == "measured" and r["group"] in factors]
    selections = []
    for workload in sorted({digest(r["workload"]) for r in held}):
        candidates = [r for r in held if digest(r["workload"]) == workload]
        if len(candidates) < 2:
            continue
        chosen = min(candidates, key=lambda r: factors[r["group"]] * r["predicted_ns"])
        best = min(candidates, key=lambda r: r["summary"]["median"])
        overlap = any(r["id"] != best["id"] and r["summary"]["p10"] <= best["summary"]["p90"] for r in candidates)
        noisy = [r["id"] for r in candidates if r["summary"]["count"] < 3 or relative_spread(r) > max_spread]
        noisy_groups = [g["group"] for g in groups if g.get("noisy_cases") and any(r["group"] == g["group"] for r in candidates)]
        regret = chosen["summary"]["median"] / best["summary"]["median"] - 1
        selections.append({"workload": chosen["workload"], "nominal_choice": chosen["id"], "measured_best_median": best["id"],
                           "regret": regret, "timing_ranges_overlap": overlap, "noisy_cases": noisy, "noisy_groups": noisy_groups,
                           "status": "inconclusive" if overlap or noisy or noisy_groups or regret > max_regret else "validated"})
    ranking_ok = bool(selections) and all(s["status"] == "validated" for s in selections)
    passed = groups and all(g["status"] == "validated" for g in groups)
    if require_ranking:
        passed = passed and ranking_ok
    return {"module": profile["module"], "profile_id": profile["profile_id"],
            "base_profile": profile["base_profile"], "regime": profile["regime"],
            "changed_inputs": changed, "max_relative_error": max_error, "max_selection_regret": max_regret,
            "max_relative_spread": max_spread, "require_ranking": require_ranking,
            "status": "stale" if changed else "validated" if passed else "inconclusive",
            "groups": groups, "selections": selections,
            "ranking_status": "unassessed" if not selections else "validated" if ranking_ok else "inconclusive",
            "note": "A per-group empirical correction, not a lower bound or a proof of global optimality. Outside the declared regime is unknown; candidate promotion still needs direct measurement."}


def target_coverage(document, requirements, role):
    """Check active-work facts without interpreting a partial bound as a prediction."""
    require(role in {"reference", "proxy", "local"}, "invalid target role")
    require(isinstance(requirements, list) and requirements, "requirements must be a nonempty list")
    missing, available, zero_work = [], [], []
    for request in requirements:
        path, work, unit = request.get("fact"), request.get("work"), request.get("unit")
        require(text(path) and number(work) and text(unit), "require fact, nonnegative work and expected rate unit")
        if work == 0:
            zero_work.append(path)
            continue
        fact = document.get("facts", {})
        for part in path.split("."):
            fact = fact.get(part, {}) if isinstance(fact, dict) else {}
        if not isinstance(fact, dict) or fact.get("status") == "unavailable" or not number(fact.get("value"), True):
            missing.append({"fact": path, "reason": "required rate missing, unavailable or nonpositive"})
        elif fact.get("unit") != unit:
            missing.append({"fact": path, "reason": "rate unit differs", "expected": unit, "observed": fact.get("unit")})
        else:
            available.append({"fact": path, **fact})
    return {"target": document.get("spec", {}).get("id"), "role": role,
            "coverage": "partial" if missing else "complete_for_requested_work",
            "missing": missing, "available": available, "zero_work": zero_work,
            "prediction_qualified": False,
            "note": "Coverage is not calibration. A partial ideal reference may remain useful; its largest priced term does not establish the actual bottleneck. Proxy/reference rates are not native measured performance."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="action", required=True)
    check = sub.add_parser("check")
    check.add_argument("profile", type=Path)
    compare = sub.add_parser("reuse")
    compare.add_argument("previous", type=Path)
    compare.add_argument("current", type=Path)
    compare.add_argument("--max-relative-drift", type=float, required=True)
    compare.add_argument("--max-relative-spread", type=float, required=True)
    fit = sub.add_parser("assess-module")
    fit.add_argument("profile", type=Path)
    fit.add_argument("--max-relative-error", type=float, required=True)
    fit.add_argument("--max-selection-regret", type=float, required=True)
    fit.add_argument("--max-relative-spread", type=float, required=True)
    fit.add_argument("--current-inputs", action="store_true")
    fit.add_argument("--require-ranking", action="store_true")
    coverage = sub.add_parser("target-check")
    coverage.add_argument("target", type=Path)
    coverage.add_argument("requirements", type=Path)
    coverage.add_argument("--role", choices=("reference", "proxy", "local"), required=True)
    args = parser.parse_args(argv)
    args.root = args.root.resolve()
    for name in ("profile", "previous", "current", "target", "requirements"):
        path = getattr(args, name, None)
        if path is not None and not path.is_absolute():
            setattr(args, name, args.root / path)
    try:
        if args.action == "target-check":
            if args.target.suffix == ".toml":
                try:
                    import tomllib
                except ImportError:
                    raise ValueError("TOML input requires Python 3.11+; use the suite's Python environment") from None
                document = tomllib.loads(args.target.read_text())
            else:
                document = load(args.target)
            result = target_coverage(document, load(args.requirements), args.role)
            code = 2 if result["missing"] else 0
        elif args.action == "reuse":
            previous, current = (validate(load(p), args.root) for p in (args.previous, args.current))
            result = reuse(previous, current, args.max_relative_drift, args.max_relative_spread)
            code = 0 if result["measurements"] and all(r["status"] == "reusable" for r in result["measurements"]) else 2
        else:
            profile = validate(load(args.profile), args.root)
            if args.action == "check":
                result = {"status": "valid", "profile_id": profile["profile_id"], "kind": profile["kind"], "scope": "structure_and_artifact_identity"}
                code = 0
            else:
                result = assess_module(profile, args.root, args.max_relative_error, args.max_selection_regret,
                                       args.max_relative_spread, args.current_inputs, args.require_ranking)
                code = 0 if result["status"] == "validated" else 2
        print(json.dumps(result, indent=2, allow_nan=False))
        return code
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print(json.dumps({"status": "error", "reason": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
