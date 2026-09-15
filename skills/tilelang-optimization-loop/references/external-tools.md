# Working with KernelWiki and ncu-report-skill

KernelWiki is included as a supplementary submodule at `external/KernelWiki`.
NCU report tooling remains an optional external integration. These supply
research and report-analysis capabilities; the task workspace
still owns the executable correctness and benchmark harness. Keep this suite's
contract, TileLang programming/lowering guidance, and candidate selection loop.

## Responsibilities

| Component | Responsibility |
| --- | --- |
| `tilelang-optimization-loop` | Task contract, PyTorch baseline analysis, stage selection, experiment records, and acceptance criteria |
| TileLang specialists | Implementation, compiler inspection, correctness/autograd, timing, and optimization |
| Minimal CUDA workflow | Native compilation, launch/stream contracts, Compute Sanitizer, Nsight Systems, and NVTX |
| KernelWiki | On-demand lookup of implementation patterns and upstream evidence, especially Hopper/Blackwell |
| ncu-report-skill helpers | Extract, compare, and summarize selected `.ncu-rep` evidence |
| Task harness | Generate inputs, call reference/candidate, validate, reset state, measure, and write results |

KernelWiki can supply the optional architecture/implementation reference layer.
It does not replace generic CUDA host/debugging workflows or dedicated TileLang
guidance. Its scripts query and validate a knowledge corpus; they do not run a
candidate correctness/performance evaluator.

The related [Kernel Design Agents workflow](https://github.com/mit-han-lab/kernel-design-agents/blob/4806866492d5c7cad05e07918790bd1e94e976a1/docs/agent-flow.md)
also delegates validation/evaluation commands to the task workspace. Its
`docs/draft.md`, `docs/plan.md`, `benchmark.csv`, `candidates.jsonl`, and `profile/`
conventions fit this suite. Use one task plan and one set of results when
combining the workflows.

## KernelWiki lookup

Use the shared [TileLang interface](kernelwiki.md)
for checkout initialization, direct upstream queries, and the chain from
TileLang intent through lowering to hardware hypotheses and local evidence.
The suite maintains this interface; the submodule maintains the knowledge.

## Use NCU analysis behind the existing measurement workflow

Preserve the original Python/TileLang launcher when it reproduces the selected
kernel. The inspected TileLang NVCC path already includes `-lineinfo`; check the
active backend and preserve matching generated source. A standalone C++ harness
is useful when isolation is needed, but recreating a JIT launch can change
dispatch, descriptors, streams, output resets, or the measured scope. Validate
equivalence before treating an extracted harness as evidence for the original.

Choose a matching invocation from the trace and discover sections with
`ncu --list-sets` / `ncu --list-sections`. Collect a bounded report with the
[local collection workflow](../../cuda_skill/references/ncu-guide.md).
Add `SourceCounters` or PM sampling when they answer the current question and
are supported. Keep ordinary latency comparisons in the original benchmark.

Set `NCU_REPORT_SKILL_ROOT` to its checkout and `NCU_PYTHON_DIR` to the installed
Nsight Compute `extras/python` directory containing `ncu_report.py`:

```bash
# Verified location on the reviewed machine; discover it on other systems.
NCU_PYTHON_DIR=/opt/nvidia/nsight-compute/2025.3.1/extras/python
env PYTHONPATH="$NCU_PYTHON_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  conda run --no-capture-output -n qla python \
  "$NCU_REPORT_SKILL_ROOT/helpers/analyze_reports.py" \
  --run-dir "$PROFILE_RUN_DIR" --report "$NCU_REPORT_PATH" --tag baseline
```

`ncu_report` ships with Nsight Compute; it is not a required pip dependency of
this suite. Archive the original report and selected invocation identity with
the derived metrics. Missing counters are unavailable evidence, not zeros.
Pass an observed limitation and a testable hypothesis to optimization, then
rerun correctness and the original benchmark after changing the kernel.

## Adaptation required before making upstream NCU the default

The inspected upstream skill is oriented toward B200 and Nsight Compute 2026.1.
Reuse its helpers selectively while addressing these compatibility issues:

- **Tool discovery:** its automatic search misses this machine's
  `/opt/nvidia/nsight-compute/2025.3.1/extras/python` layout. Supply the path
  explicitly. Choose metric names from the report/device rather than assuming
  the `B200_KEY_METRICS` list applies to every target.
- **Collection policy:** upstream requires full and source captures, including
  `--set source`. That set does not exist in local NCU 2025.3.1; `SourceCounters`
  is a supported section. Make collection proportional to the question.
- **Report selection:** `load_report()` chooses range 0, action 0. Require a
  verified single-launch report, or add explicit range/action selection and
  record the kernel identity. Keep the report object alive during analysis.
- **Failure status:** `analyze_reports.py` skips nonexistent reports and can
  exit successfully with no analysis. The harness must verify that the expected
  report/action and outputs exist, and fail the stage when they do not.
- **Timeline preservation:** `ascii_plot()` uses floor-sized buckets and can
  discard the last samples: 101 positive samples at width 80 keep only the
  first 80. Fix and test complete sample coverage before using its plots for
  tail diagnosis; retain timestamps and distinguish missing values from zero.
- **Diagnosis:** upstream's single-cause and mandatory-playbook rules are too
  strong for automatic optimization. Treat stalls and rule-engine speedup
  estimates as evidence for hypotheses; confirm effects with experiments.

Keep our event/graph benchmarking, PyTorch/Inductor inspection, allocation
analysis, Nsight Systems, sanitizers, and lowering tools. NCU report analysis
can replace duplicated NCU interpretation material once adapted; it does not
cover those other profiling/debugging responsibilities.

## Repository and harness integration

Prefer pinned external checkouts or submodules with thin local routing, so
upstream content stays maintainable. Keep task outputs in the task workspace:

```text
skill-suite/
  skills/tilelang-optimization-loop/
  skills/tilelang/
  skills/cuda_skill/             # minimal native workflow
  external/KernelWiki/           # included pinned submodule
  external/ncu-report-skill/     # proposed pinned dependency

task-workspace/
  docs/                         # contract and shared plan
  reference.py
  candidates/
  harness/                      # validation / benchmark / profile entrypoints
  candidates.jsonl
  benchmark.csv
  profile/<candidate-and-case>/
```

The harness should expose validation, benchmark, and profile-target commands
against the same candidate entrypoint and case definition. Return candidate
revision/configuration, case ID, correctness status, timing scope/units,
environment, and artifact paths. Distinguish failed, unrun, and unavailable
stages from passes; serialize shared-GPU measurements. Start with the existing
task's harness and result schema rather than imposing a second evaluator.

For a lighter worker, provide one bounded stage plus the contract, exact
commands, current evidence packet, and expected artifacts. Query results and
profiler findings feed that task; they do not change its acceptance criteria.

## Review snapshot

Inspected on 2026-09-11:

- [KernelWiki `b6b4301`](https://github.com/mit-han-lab/KernelWiki/tree/b6b4301f15e8ce6955a56776690643ce5db369e6):
  query/page tools ran in `qla`. The README scopes dedicated language guides
  to CUDA C++, CuTe DSL, PTX, and Triton; TileLang has source coverage.
- [ncu-report-skill `74a1291`](https://github.com/mit-han-lab/ncu-report-skill/tree/74a12918e9f64d78036f14da5f8765e435b949a4):
  helper import required the explicit NCU Python path; the timeline sample-loss
  issue was reproduced independently of GPU profiling. A direct Python/TileLang
  add kernel was profiled with NCU 2025.3.1 `--set basic` on an NVIDIA L20Z;
  `analyze_reports.py` exported 563 metrics for the selected `kernel_kernel`
  action. This verifies basic collection/extraction, not every B200 diagnostic
  or source/PM-sampling helper.
- [KDA `4806866`](https://github.com/mit-han-lab/kernel-design-agents/tree/4806866492d5c7cad05e07918790bd1e94e976a1):
  workflow documents plus pinned companion skills; no shared executable
  correctness/benchmark runner in the inspected first-party tree. Its pins
  differ from the standalone repositories reviewed above.

Preserve upstream notices and artifact provenance when incorporating files.
KernelWiki is included; the NCU dependency and task-harness layout above remain
integration guidance. Reuse an existing task harness when available.
