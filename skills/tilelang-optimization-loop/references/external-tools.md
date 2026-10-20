# Working with TileFoundry, KernelWiki, and ncu-report-skill

TileFoundry and NCU report tooling are optional external integrations.
KernelWiki is a standalone skill dependency, installed next to this suite's
skills. Together they supply authored-model analysis, research, and
report-analysis capabilities; the task workspace still owns the executable
correctness and benchmark harness. Keep this suite's contract, TileLang
programming/lowering guidance, and candidate selection loop.

## Responsibilities

| Component | Responsibility |
| --- | --- |
| `tilelang-optimization-loop` | Task contract, PyTorch baseline analysis, stage selection, experiment records, and acceptance criteria |
| TileFoundry | Authored model/operator HIR, topology and placement, static cost analysis, evaluator/runtime-twin checking, and HIR visualization |
| TileLang specialists | Implementation, compiler inspection, correctness/autograd, timing, and optimization |
| Minimal CUDA workflow | Native compilation, launch/stream contracts, Compute Sanitizer, Nsight Systems, and NVTX |
| KernelWiki | On-demand lookup of implementation patterns and upstream evidence, especially Hopper/Blackwell |
| ncu-report-skill helpers | Extract, compare, and summarize selected `.ncu-rep` evidence |
| Task harness | Generate inputs, call reference/candidate, validate, reset state, measure, and write results |

KernelWiki can supply the optional architecture/implementation reference layer.
It does not replace generic CUDA host/debugging workflows or dedicated TileLang
guidance. Its scripts query and validate a knowledge corpus; they do not run a
candidate correctness/performance evaluator.

TileFoundry supplies an earlier design layer when a task spans fused operations,
logical sharding, topology, or storage placement. Follow the
[TileFoundry design workflow](../../designing-with-tilefoundry/SKILL.md), and
keep its static nanosecond estimates distinct from measured kernel latency. It
is installed separately rather than included as a submodule or default Python
dependency because its interpreter and package constraints may conflict with
an established TileLang environment.

The related [Kernel Design Agents workflow](https://github.com/mit-han-lab/kernel-design-agents/blob/4806866492d5c7cad05e07918790bd1e94e976a1/docs/agent-flow.md)
also delegates validation/evaluation commands to the task workspace. Its
`docs/draft.md`, `docs/plan.md`, `benchmark.csv`, `candidates.jsonl`, and `profile/`
conventions fit this suite. Use one task plan and one set of results when
combining the workflows.

## KernelWiki lookup

Use the shared [TileLang interface](kernelwiki.md)
for locating the installed skill, direct upstream queries, and the chain from
TileLang intent through lowering to hardware hypotheses and local evidence.
The suite maintains this interface; the upstream KernelWiki project maintains
the knowledge.

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
Nsight Compute `extras/python` directory containing `ncu_report.py`. The
ncu-report-skill checkout is a proposed pinned dependency and is **not** present
in this suite, so guard on it and fail the stage explicitly when it is absent:

```bash
# Verified location on the reviewed machine; discover it on other systems.
NCU_PYTHON_DIR=/opt/nvidia/nsight-compute/2025.3.1/extras/python
if [ ! -f "$NCU_REPORT_SKILL_ROOT/helpers/analyze_reports.py" ]; then
  echo "ncu-report-skill checkout missing at '$NCU_REPORT_SKILL_ROOT'; skipping report analysis" >&2
  exit 1
fi
env PYTHONPATH="$NCU_PYTHON_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  python "$NCU_REPORT_SKILL_ROOT/helpers/analyze_reports.py" \
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
  external/TileFoundry/          # vendored pinned submodule
  external/ncu-report-skill/     # proposed pinned dependency
  # KernelWiki is a standalone skill, installed outside this tree at
  # ~/.agents/skills/KernelWiki and linked into each skills directory.

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

TileFoundry was inspected separately on 2026-09-15 at
[tile-ai/TileFoundry `2af2364`](https://github.com/tile-ai/TileFoundry/tree/2af2364d53f262946b7f07643583262f3abd2d76).
In an isolated Python 3.12 environment, its CLI catalog/spec/tutorial, H200
RMSNorm compute/memory/roofline/performance analysis, CPU evaluator/runtime-twin
check, DOT builder, and local interactive viewer endpoints were exercised. The
viewer fetched and verified its pinned Graphviz-WASM and D3 assets. After eight
NVIDIA L20Z devices became visible, the targeted 34-test subset passed and a
synchronized PyTorch FP16 CUDA matrix multiplication completed on device 0. No
TileFoundry-generated CUDA kernel was compiled or timed in this review.

A full cold start was exercised on 2026-10-08 (tilefoundry 0.0.2, tilelang
0.1.15, L20Z): a fused-vs-split residual-add+RMSNorm pair was taken from a
PyTorch reference through authored HIR variants, `analyze` diffs, pure-torch
and TileLang runtime-twin checks, and `do_bench` measurement. The static
traffic diff (1.67×) predicted the direction and memory-bound diagnosis; the
measured latency ratio was 1.31× (torch.compile) and 1.42× (TileLang), and the
fused TileLang kernel reached 1.55× the compiled baseline. The `compile()`
HIR→CUDA backend was found unusable as a measured floor in 0.0.2 (no MeshScope
lowering, runtime headers missing from the wheel, demo-quality codegen).

Upstream main was re-checked on the same day ([`f71ed0e`](https://github.com/tile-ai/TileFoundry/tree/f71ed0e7789648ce408af623383b8f039ed90605),
0.0.3.dev28): the new `schedule candidates/facts/finalize` surface lowers
fully-scheduled HIR to verified TIR and handwritten TIR compiles to real CUDA,
but lowered-HIR compile remains broken (two emitter gaps), the wheel still
ships no runtime headers, and `analyze`'s report format changed. The skills
stay pinned to 0.0.2 pending the next release tag; evidence lives in the task
workspace's `artifacts/upstream-main/`.

Re-checked again at [`5f346a2`](https://github.com/tile-ai/TileFoundry/tree/5f346a285158b47e92d29d2c19e76b5dc3ff48a6)
(0.0.3.dev38, 2026-10-15) with an end-to-end `compile()` drive. Mesh-capable
lowering now works (the f71ed0e-era blocker is gone; upstream suite 474/474,
including with four small local emitter patches applied). Still open: the
wheel ships no `runtime.h`; `T.copy_async_tensor` has no emitter by design, so
the tutorial's "optimal" GEMM schedule cannot compile; same-day commit
`a64ec82` introduced four emitter bugs (slice-coordinate rendering, MeshCoord
coordinates, missing `Cast` emitter, `Fill` on kernel params — all patched and
validated locally) and removed the `rmem_layout=` attribute, breaking the
two-day-old flagship `examples/qwen3_8_27b_prefill-tilelang` artifacts on main
(pin that example to `f8fd71e`). With the patches plus a hand-added `T.reduce`
smem workspace, a scheduled RMSNorm compiles but the binary reads smem out of
bounds — the runtime reduce dispatch mis-counts `warps_per_group` for the
composed cta×thread mesh that multi-CTA finalize always produces. The Qwen
example itself confirms the supported integration: TileFoundry for
HIR/check/analyze/schedule/finalize, TileLang kernels implementing the
finalized TIR — `compile()` unused.

Later on 2026-10-15 the suite vendored TileFoundry as the
`external/TileFoundry` submodule, pinned at `5f346a2`, and installed it
editable into the suite `.venv` (with `cmake`/`ninja` and the
`third_party/cutlass` submodule initialized). That resolves the header problem
for this suite — op-level `compile()` builds and runs pass on the suite GPU —
and the `designing-with-tilefoundry` skill was re-verified against the pin
(CLI surface, tutorial extraction, described models, targets, predicates, DOT
and Viewer imports, the TileLang runtime-twin check PASS with tilelang
0.1.15). Two 0.0.2-era DSL notes were dropped as obsolete (`tf.local` is
removed; the `ShardMesh` verbose shard form no longer imports), and the
shared-weight shard gap persists in a new form with the scheduled-copy idiom
as its answer. The pin policy: bump the submodule deliberately, re-run the
skill's probes at each new pin, and never track main blindly — upstream broke
its own flagship example within two days.

Bumped to `25fbbaa` (#246) on 2026-10-20. The four local emitter patches are
now upstream (merged as `a5501c9`, PR tile-ai/TileFoundry#240), and the
opaque finalize error is fixed (#241, closing #237): `schedule finalize` now
names the unscheduled site, lists its instruction candidates, and points at
`tf.schedule` as the remedy. Still open: #239 (the `insert_slice`/`tf.zeros`
per-position billing and lane under-fill — predicted-ns numbers are
byte-identical at the new pin, so the ranking-inversion warning stands), the
TMA/WGMMA/`Where` emitter gaps, and the `warps_per_group` reduce-dispatch
bug. Probe harness 12/12 PASS at the new pin; trust-table spot checks
byte-identical.

Preserve upstream notices and artifact provenance when incorporating files.
KernelWiki ships as a standalone skill; the NCU dependency and task-harness
layout above remain integration guidance. Reuse an existing task harness when
available.
