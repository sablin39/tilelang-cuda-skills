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
ships as a pinned submodule, while installation into a task Python environment
is explicit. Install core and the matching vendor package together; interpreter
and dependency constraints may conflict with an established TileLang environment.

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
ncu-report-skill checkout is **not** bundled
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

Inspect the selected helper revision before adopting it. Check these integration
boundaries rather than assuming its metric/collection defaults fit the task:

- **Tool discovery:** automatic searches may miss layouts such as
  `/opt/nvidia/nsight-compute/2025.3.1/extras/python` layout. Supply the path
  explicitly. Choose metric names from the report/device rather than assuming
  the `B200_KEY_METRICS` list applies to every target.
- **Collection policy:** discover supported sets/sections before requesting
  source captures. Do not assume `--set source` exists; check `SourceCounters`
  availability. Make collection proportional to the question.
- **Report selection:** helpers may default to range 0, action 0. Require a
  verified single-launch report, or add explicit range/action selection and
  record the kernel identity. Keep the report object alive during analysis.
- **Failure status:** a helper can skip nonexistent reports and
  exit successfully with no analysis. The harness must verify that the expected
  report/action and outputs exist, and fail the stage when they do not.
- **Timeline preservation:** floor-sized downsampling buckets can
  discard the last samples: 101 positive samples at width 80 keep only the
  first 80. Fix and test complete sample coverage before using its plots for
  tail diagnosis; retain timestamps and distinguish missing values from zero.
- **Diagnosis:** do not force single-cause or mandatory-playbook conclusions. Treat stalls and rule-engine speedup
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

## Current TileFoundry boundary

Use the [TileFoundry skill](../../designing-with-tilefoundry/SKILL.md) for the
active pin's package layout, schedule/facts interfaces, shared emission rules,
and opt-in synchronization analysis. Keep that contract in one place. Static
analysis and evaluator agreement are separate from generated-code correctness
and measured performance. Runtime headers ship with their owning packages;
CUTLASS and the CUDA toolchain remain additional build requirements.

Preserve upstream notices and provenance. KernelWiki ships as a standalone
skill; NCU integration remains optional. Return tool findings through the
[stage report](stage-handoff.md) and feed reusable gaps into
[suite maintenance](improving-the-suite.md).
