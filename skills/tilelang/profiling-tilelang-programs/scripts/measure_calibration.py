#!/usr/bin/env python3
"""Collect scoped CUDA device probes or a task's iterative module calibration.

PyTorch is imported only for collection. No clocks, power limits or compiler
settings are changed. Device probes are complete library operations, not
isolated hardware instructions or a comprehensive hardware specification.
"""

import argparse
from copy import deepcopy
import csv
from contextlib import nullcontext
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys

import calibration as cal


CONTEXT_ENV = ("CUDA_MODULE_LOADING", "CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE",
               "TORCH_CUDA_ARCH_LIST", "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "CUDA_MPS_PIPE_DIRECTORY")


class ModuleJournal:
    """Single-writer immutable case receipts, separate from a sealed profile."""
    def __init__(self, args, manifest, base):
        self.root = args.root.resolve()
        self.directory = (args.root / args.journal).resolve()
        cal.require(not Path(args.journal).is_absolute() and self.directory.is_relative_to(self.root),
                    "journal must stay inside the task root")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / "writer.lock").open("a+")
        import fcntl
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.manifest = manifest
            identity = cal.digest(manifest)
            self.identity = identity
            path = self.directory / "manifest.json"
            if path.exists():
                cal.require(getattr(args, "resume_journal", False), "existing journal requires --resume-journal")
                cal.require(cal.load(path) == manifest, "journal source, case design or measurement context changed")
                cal.require(getattr(args, "resume_anchor", None), "resume requires a fresh --resume-anchor device profile")
                current = cal.validate(cal.load(cal.relative_file(args.root, args.resume_anchor)), args.root)
                assessment = cal.reuse(base, current, args.max_anchor_drift, args.max_anchor_spread)
                required = {row["id"] for row in base["measurements"] if row["status"] == "measured"}
                eligible = {row["id"] for row in assessment["measurements"] if row["status"] == "reusable"}
                cal.require(required and assessment["context_compatible"] and required <= eligible,
                            "resume anchors are incomplete, noisy, drifted or incompatible")
                self.anchor = cal.fingerprint(args.root, args.resume_anchor)
            else:
                cal.require(not getattr(args, "resume_journal", False), "no existing journal to resume")
                cal.write_immutable(path, cal.canonical(manifest))
                self.anchor = None
        except BaseException:
            self.lock.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.lock.close()

    def case_path(self, record):
        return self.directory / (cal.digest(record) + ".json")

    def read(self, record):
        path = self.case_path(record)
        if not path.exists():
            return None
        value = cal.load(path)
        cal.require(value["manifest_hash"] == self.identity and value["contract"] == record, "case receipt identity differs")
        observed = value["record"]
        cal.require(value["record_hash"] == cal.digest(observed),
                    "case receipt content hash differs")
        return deepcopy(observed)

    def write(self, contract, record):
        value = {"manifest_hash": self.identity, "contract": contract, "record": record,
                 "record_hash": cal.digest(record)}
        cal.write_immutable(self.case_path(contract), cal.canonical(value))

    def references(self):
        return [cal.fingerprint(self.root, str(path.relative_to(self.root)))
                for path in sorted(self.directory.glob("*.json"))]


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def telemetry(uuid):
    fields = ["uuid", "driver_version", "clocks.current.sm", "clocks.current.memory",
              "power.limit", "power.draw", "temperature.gpu", "utilization.gpu", "memory.used"]
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=" + ",".join(fields),
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=True)
        for row in csv.reader(io.StringIO(result.stdout)):
            item = dict(zip(fields, (x.strip() for x in row)))
            # PyTorch may expose the bare CUDA UUID while nvidia-smi adds GPU-.
            # Keep MIG identities distinct; never map a visible ordinal by guess.
            key = lambda value: str(value).removeprefix("GPU-").lower() if value else None
            if uuid and key(item.get("uuid")) == key(uuid):
                return {"status": "queried", "source": "nvidia-smi", "values": item,
                        "units": {"clocks": "MHz", "power": "W", "temperature": "C", "utilization": "%", "memory": "MiB"}}
        return {"status": "unavailable", "reason": "device UUID not matched; ordinal mapping was not guessed"}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"status": "unavailable", "reason": str(exc)}


def identity(torch, args):
    props = torch.cuda.get_device_properties(args.device)
    uuid = getattr(props, "uuid", None)
    uuid = str(uuid) if uuid is not None else None
    observed = telemetry(uuid)
    values = observed.get("values", {})
    context = {
        "device": {"uuid": uuid, "name": props.name, "compute_capability": [props.major, props.minor],
                   "sm_count": props.multi_processor_count, "total_memory_bytes": props.total_memory},
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "host_arch": platform.machine(), "intraop_threads": torch.get_num_threads(),
                     "interop_threads": torch.get_num_interop_threads(),
                     "cuda_runtime": torch.version.cuda, "driver": values.get("driver_version"),
                     "tilelang": package_version("tilelang"), "triton": package_version("triton"),
                     "tilefoundry": package_version("tilefoundry"),
                     "matmul_precision": torch.get_float32_matmul_precision(),
                     "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                     "allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                     "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                     "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                     "compiler_environment": {k: os.environ.get(k) for k in CONTEXT_ENV}},
        "operating": {"clock_policy": args.clock_policy, "power_policy": args.power_policy,
                      "power_limit_w": values.get("power.limit"), "isolation": args.isolation},
    }
    facts = {}
    for key, attr, unit in [("sm_count", "multi_processor_count", "count"),
                            ("memory_capacity", "total_memory", "byte"),
                            ("shared_memory_per_block", "shared_memory_per_block", "byte"),
                            ("shared_memory_per_sm", "shared_memory_per_multiprocessor", "byte"),
                            ("registers_per_sm", "regs_per_multiprocessor", "count")]:
        value = getattr(props, attr, None)
        facts[key] = ({"status": "unavailable", "conditions": "not exposed by this runtime"} if value is None else
                      {"value": value, "unit": unit, "origin": "queried", "source": "torch.cuda.get_device_properties." + attr,
                       "conditions": "visible device; capacity, not measured throughput"})
    return context, facts, observed


def tool_sources(root):
    refs = []
    for path in (Path(__file__).resolve(), Path(cal.__file__).resolve()):
        data = path.read_bytes()
        relative = f"calibration/tools/{hashlib.sha256(data).hexdigest()}/{path.name}"
        destination = root / relative
        cal.write_immutable(destination, data, allow_existing=True)
        refs.append(cal.fingerprint(root, relative))
    return refs


def protocol(args):
    return {"backend": "cuda_events", "scope": "current_stream_batched_calls",
            "warmup_calls": args.warmup, "calls_per_sample": args.iterations, "samples": args.repeats,
            "summary": "median_and_sample_p10_p90", "allocation": "setup excluded; allocations inside run included",
            "cache": "repeated inputs; no explicit cache flush", "cuda_graph": False,
            "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def time_calls(torch, run, args):
    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize(args.device)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(args.repeats):
        start.record()
        for _ in range(args.iterations):
            run()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e6 / args.iterations)
    cal.summarize(samples)
    return samples


def copy_probe(torch, args, mib):
    size = mib * 1024 * 1024
    a = torch.randint(0, 256, (size,), dtype=torch.uint8, device=args.device)
    b = torch.empty_like(a)

    def run():
        b.copy_(a)

    run()
    cal.require(torch.equal(a, b), "copy mismatch")
    samples_ns = time_calls(torch, run, args)
    return {"samples": [2 * size / (ns * 1e-9) for ns in samples_ns], "samples_ns": samples_ns,
            "work": {"logical_read_write_bytes": 2 * size}, "correctness": "pass"}


def gemm_probe(torch, args, size):
    dtype = getattr(torch, args.dtype)
    a = torch.rand((size, size), dtype=dtype, device=args.device) - .5
    b = torch.rand_like(a) - .5
    c = torch.empty_like(a)

    def run():
        torch.mm(a, b, out=c)

    run()
    # Independent CPU FP64 check of evenly spaced rows/columns. This is a
    # library-probe sanity check, not the oracle for arbitrary task kernels.
    indices = torch.linspace(0, size - 1, min(8, size), device=args.device).long()
    ref = a.index_select(0, indices).cpu().double() @ b.index_select(1, indices).cpu().double()
    got = c.index_select(0, indices).index_select(1, indices).cpu().double()
    rtol, atol = {"bfloat16": (.02, .02), "float16": (.002, .005), "float32": (.0001, .0001)}[args.dtype]
    torch.testing.assert_close(got, ref, rtol=rtol, atol=atol)
    samples_ns = time_calls(torch, run, args)
    return {"samples": [2 * size ** 3 / (ns * 1e-9) for ns in samples_ns], "samples_ns": samples_ns,
            "work": {"dense_flop": 2 * size ** 3}, "correctness": "pass",
            "check": {"kind": "sampled_cpu_fp64", "rows_and_columns": min(8, size), "rtol": rtol, "atol": atol}}


def collect_device(torch, args, context, facts, artifacts):
    inputs = [cal.snapshot_input(args.root, p) for p in args.input]
    if args.reference_facts:
        inputs.append(cal.snapshot_input(args.root, args.reference_facts))
        references = cal.load(cal.relative_file(args.root, args.reference_facts))
        cal.require(isinstance(references, dict), "reference facts must be an object")
        for name, fact in references.items():
            cal.require(name not in facts, "reference facts cannot overwrite runtime facts")
            cal.require(fact.get("status") == "unavailable" or fact.get("origin") in {"vendor", "reference", "derived", "estimated"},
                        "reference import cannot claim a local measurement")
            facts[name] = fact
    records = []
    probes = []
    if "copy" in args.probes:
        for mib in args.copy_mib:
            probes.append(({"id": f"copy-{mib}MiB", "quantity": "copy_effective_bandwidth", "unit": "byte/s",
                            "scope": {"implementation": "torch.Tensor.copy_", "dtype": "uint8", "contiguous": True,
                                      "elements": mib * 1024 * 1024, "working_set_bytes": 2 * mib * 1024 * 1024,
                                      "traffic": "logical read+write; cache/DRAM traffic unverified"}},
                           lambda mib=mib: copy_probe(torch, args, mib)))
    if "gemm" in args.probes:
        for size in args.gemm_size:
            probes.append(({"id": f"gemm-{args.dtype}-{size}", "quantity": "dense_gemm_effective_throughput", "unit": "flop/s",
                            "scope": {"implementation": "torch.mm", "dtype": args.dtype, "m": size, "n": size, "k": size,
                                      "layout": "contiguous_row_major", "sparsity": "dense", "traffic": "complete library GEMM"}},
                           lambda size=size: gemm_probe(torch, args, size)))
    for record, run in probes:
        try:
            record.update(run(), status="measured")
        except Exception as exc:
            record.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        records.append(record)
        print(f"{record['id']}: {record['status']}", file=sys.stderr, flush=True)
    return {"kind": "device", "context": context, "protocol": protocol(args), "facts": facts,
            "measurements": records, "inputs": inputs, "artifacts": artifacts}


def collect_module(torch, args, context, artifacts):
    base_ref = cal.fingerprint(args.root, args.base)
    base = cal.validate(cal.load(cal.relative_file(args.root, args.base)), args.root)
    cal.require(base["kind"] == "device" and context == base["context"],
                "current context differs from the device profile; collect a current device profile first")
    factory_path = cal.relative_file(args.root, args.factory)
    inputs = [cal.snapshot_input(args.root, p) for p in dict.fromkeys([args.factory] + args.input)]
    spec = importlib.util.spec_from_file_location("task_calibration_factory", factory_path)
    module = importlib.util.module_from_spec(spec)
    sys.path[:0] = [str(factory_path.parent), str(args.root)]
    sys.modules[spec.name] = module
    # Execute the preserved factory bytes, avoiding a timestamp-based .pyc
    # cache hit for a same-size source edit within one filesystem clock tick.
    factory_source = cal.relative_file(args.root, inputs[0]["artifact"]["path"]).read_bytes()
    exec(compile(factory_source, str(factory_path), "exec"), module.__dict__)
    definition = module.calibration_cases(deepcopy(base))
    cal.require(isinstance(definition, dict) and isinstance(definition.get("cases"), list) and definition["cases"],
                "factory must return module, regime, and a nonempty cases list")
    cases = []
    # Freeze metadata/predictions for every case before any timing occurs.
    for case in definition["cases"]:
        record = {k: deepcopy(case[k]) for k in ("id", "group", "partition", "workload", "configuration", "predicted_ns")}
        cal.canonical(record)
        cal.require(callable(case.get("setup")), "case needs a setup callable")
        cases.append((record, case["setup"]))
    profile = {"kind": "module", "module": definition["module"], "regime": deepcopy(definition["regime"]),
               "context": context, "protocol": protocol(args), "base_profile": base_ref,
               "refresh_reason": args.reason, "inputs": inputs, "artifacts": artifacts, "cases": []}
    if args.parent:
        profile["parent"] = cal.fingerprint(args.root, args.parent)
    # Reject a malformed/overlapping fit design before spending GPU time.
    preflight = deepcopy(profile)
    preflight.update(schema_version=1, created_at="preflight", command="preflight")
    preflight["cases"] = [dict(record, status="unavailable", reason="not run yet") for record, _ in cases]
    preflight["profile_id"] = cal.profile_id(preflight)
    cal.validate(preflight, args.root)
    manifest = {"module": profile["module"], "regime": profile["regime"], "context": context,
                "protocol": profile["protocol"], "base_profile": base_ref, "inputs": inputs,
                "tools": artifacts, "cases": [record for record, _ in cases],
                "resume_safe_cases": definition.get("resume_safe_cases") is True}
    journal_context = ModuleJournal(args, manifest, base) if getattr(args, "journal", None) else nullcontext(None)
    with journal_context as journal:
        if journal and getattr(args, "resume_journal", False):
            cal.require(manifest["resume_safe_cases"], "factory must declare independently reconstructible resume_safe_cases")
        for record, setup in cases:
            contract = deepcopy(record)
            cached = journal.read(contract) if journal else None
            if cached is not None:
                profile["cases"].append(cached)
                print(json.dumps({"kind": "case_reused", "id": record["id"], "status": cached["status"]}), file=sys.stderr, flush=True)
                continue
            stage, prepared = "setup", None
            try:
                prepared = setup()
                cal.require(callable(prepared.get("run")) and callable(prepared.get("check")), "setup needs run and check callables")
                stage = "correctness"
                cal.require(prepared["check"]() is True, "check must assert the contract and return True")
                torch.cuda.synchronize(args.device)
                record["correctness"] = "pass"
                stage = "timing"
                samples = time_calls(torch, prepared["run"], args)
                record.update(status="measured", correctness="pass", samples_ns=samples)
            except Exception as exc:
                record.update(status="failed", failure_stage=stage, reason=f"{type(exc).__name__}: {exc}")
            finally:
                del prepared
            profile["cases"].append(record)
            if journal:
                journal.write(contract, record)
            print(json.dumps({"kind": "case_completed", "id": record["id"], "status": record["status"]}), file=sys.stderr, flush=True)
        if journal:
            profile["journal"] = {"manifest_hash": journal.identity, "receipts": journal.references(),
                                  "resume_anchor": journal.anchor}
            profile["artifacts"] = list(artifacts) + journal.references() + ([journal.anchor] if journal.anchor else [])
    changed = []
    for item in inputs:
        try:
            same = cal.fingerprint(args.root, item["source_path"])["sha256"] == item["artifact"]["sha256"]
        except (OSError, ValueError):
            same = False
        if not same:
            changed.append(item["source_path"])
    if changed:
        for record in profile["cases"]:
            if record["status"] == "measured":
                record["discarded_samples_ns"] = record.pop("samples_ns")
                record.update(status="failed", failure_stage="source_identity", reason="inputs changed during collection: " + ", ".join(changed))
    return profile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    device, module = sub.add_parser("device"), sub.add_parser("module")
    for command in (device, module):
        command.add_argument("--root", type=Path, default=Path.cwd())
        command.add_argument("--output", required=True, help="new task-relative JSON path; existing evidence is never overwritten")
        command.add_argument("--device", type=int, default=0, help="visible CUDA ordinal")
        command.add_argument("--warmup", type=int, default=5)
        command.add_argument("--iterations", type=int, default=20)
        command.add_argument("--repeats", type=int, default=9)
        command.add_argument("--clock-policy", default="uncontrolled")
        command.add_argument("--power-policy", default="uncontrolled")
        command.add_argument("--isolation", choices=("exclusive", "shared", "unknown"), required=True)
        command.add_argument("--input", action="append", default=[], help="additional task-relative source/context input to preserve")
    device.add_argument("--probes", choices=("copy", "gemm"), nargs="+", default=["copy", "gemm"])
    device.add_argument("--copy-mib", type=int, nargs="+", default=[16, 64, 256])
    device.add_argument("--gemm-size", type=int, nargs="+", default=[1024, 4096])
    device.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    device.add_argument("--reference-facts", help="task-relative JSON facts from specs/wiki, with source and applicability")
    module.add_argument("--factory", required=True, help="task-relative Python file exporting calibration_cases(device_profile)")
    module.add_argument("--base", required=True, help="task-relative device profile")
    module.add_argument("--parent", help="task-relative previous module version")
    module.add_argument("--reason", required=True, help="initial calibration or the exact change triggering recalibration")
    module.add_argument("--journal", help="task-relative directory for immutable completed-case receipts")
    module.add_argument("--resume-journal", action="store_true", help="reuse completed cases after identity and anchor checks")
    module.add_argument("--resume-anchor", help="fresh device profile with matching measured anchors")
    module.add_argument("--max-anchor-drift", type=float, default=0.05)
    module.add_argument("--max-anchor-spread", type=float, default=0.10)
    args = parser.parse_args(argv)
    try:
        args.root = args.root.resolve()
        output = (args.root / args.output).resolve()
        cal.require(not Path(args.output).is_absolute() and output.is_relative_to(args.root), "output must stay inside the task root")
        cal.require(not output.exists(), "output already exists; choose a new version")
        cal.require(args.warmup >= 0 and args.iterations > 0 and args.repeats >= 3, "need nonnegative warmup, positive iterations and at least three samples")
        if args.action == "device":
            cal.require(all(x > 0 for x in args.copy_mib + args.gemm_size), "probe sizes must be positive")
            cal.require(len(set(args.probes)) == len(args.probes) and len(set(args.copy_mib)) == len(args.copy_mib)
                        and len(set(args.gemm_size)) == len(args.gemm_size), "duplicate probes/sizes")
        else:
            cal.require(not args.resume_journal or args.journal, "--resume-journal requires --journal")
            cal.require(not args.resume_anchor or args.resume_journal, "--resume-anchor requires --resume-journal")
        import torch
        cal.require(torch.cuda.is_available(), "CUDA runtime unavailable")
        torch.cuda.set_device(args.device)
        context, facts, before = identity(torch, args)
        artifacts = tool_sources(args.root)
        profile = (collect_device(torch, args, context, facts, artifacts) if args.action == "device" else
                   collect_module(torch, args, context, artifacts))
        profile["telemetry"] = {"before": before, "after": telemetry(context["device"]["uuid"])}
        launch_env = {key: os.environ[key] for key in CONTEXT_ENV + ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "NVIDIA_VISIBLE_DEVICES") if key in os.environ}
        profile["launch_environment"] = launch_env
        profile["command"] = shlex.join(["env"] + [f"{k}={v}" for k, v in launch_env.items()] +
                                       [sys.executable, str(Path(__file__).resolve())] + (sys.argv[1:] if argv is None else argv))
        sealed = cal.seal(profile, output, args.root)
        records = sealed["measurements"] if sealed["kind"] == "device" else sealed["cases"]
        failed = [r["id"] for r in records if r["status"] != "measured"]
        print(json.dumps({"profile": str(output), "profile_id": sealed["profile_id"], "failed": failed}, indent=2))
        return 2 if failed else 0
    except (OSError, ValueError, KeyError, TypeError, AttributeError, ImportError, RuntimeError) as exc:
        print(json.dumps({"status": "error", "reason": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
