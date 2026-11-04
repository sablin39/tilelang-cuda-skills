---
name: designing-with-tilefoundry
description: Author TileFoundry HIR, compare logical work and placement, inspect instruction schedules, and check runtime twins against the evaluator. Use for explicit TileFoundry work or fusion/sharding/storage decisions that benefit from an independent model; use TileLang specialists directly for an isolated kernel tuning question.
---

# Designing with TileFoundry

Use an authored HIR model when it answers a concrete design question: a fusion
boundary, storage placement, ownership transition, or independent semantic
check. State that question before building the model. Static reports rank
hypotheses; independent numerical tests and GPU measurements select the final
kernel. Do not introduce a second model for a local tuning question the existing
TileLang lowering and benchmark already answer.

## Locate and install the pinned source

The suite vendors `external/TileFoundry` and its CUTLASS dependency at recorded
gitlinks. Use `$TILEFOUNDRY_ROOT` if set; otherwise locate the suite from this
symlinked skill's resolved directory or `$SUITE_CACHE` (default
`~/.local/share/tilelang-cuda-skills`). For copied skills, retain the suite
checkout or set `TILEFOUNDRY_ROOT` explicitly. Source is outside the skill copy.
The suite installer installs core and bundled vendor packages into
`$SUITE/.venv` by default. Use that interpreter, or pass `--python` to install
into an existing task venv. Rerun installation after a pin update; checking out
new source alone does not refresh package metadata. `--check-only` verifies the
selected environment and its recorded installation pin without modifying it.

```bash
SUITE=...  # suite checkout/cache
TF="$SUITE/external/TileFoundry"
test "$(git -C "$TF" rev-parse HEAD)" = "$(git -C "$SUITE" rev-parse HEAD:external/TileFoundry)"
# If missing/mismatched, preserve local edits before restoring the recorded pin:
git -C "$SUITE" submodule update --init --recursive -- external/TileFoundry

# Python 3.12+; use the environment that will run the task.
TASK_ENV=/path/to/venv
"$SUITE/scripts/install-skills.sh" --python "$TASK_ENV/bin/python" --scope project
"$SUITE/scripts/install-skills.sh" --python "$TASK_ENV/bin/python" --scope project --check-only
```

Core, NVIDIA and Apple are separate distributions. Core-only installation
supplies CPU support. Released extras such as `tilefoundry[nvidia]` select the
matching vendor version; for vendored source install core and vendor projects
together. A `PYTHONPATH=.../src` override neither registers vendor entry points
nor switches their code. Compare pins in separate environments. When migrating
an old checkout, remove obsolete repository-root `tilefoundry.egg-info` before
reinstalling, as upstream's developer guide directs.

Runtime headers ship with their owning packages. CUDA builds additionally need
`nvcc`, CMake/Ninja on `PATH`, and CUTLASS: set `TILEFOUNDRY_CUTLASS_INCLUDE`
to its `include` directory or initialize the vendored `third_party/cutlass`.
Record the git SHA as well as versions; shallow metadata can mislabel releases.
Probe the selected interpreter/CLI with `--version`, `target list` and
`analyze --help`. Check dependency resolution before changing a torch/TileLang
environment. Invoke TileFoundry when its independent analysis answers the task's
design question; ordinary TileLang work can use the TileLang tools directly.

## Author against the installed interface

Start from executable tutorials and normative specs:

```bash
tilefoundry tutorial
tilefoundry tutorial optimize > optimize.md
tilefoundry spec analysis
tilefoundry spec
tilefoundry models
tilefoundry check --help
tilefoundry schedule --help
```

Use each tutorial's own opening extraction command in its own directory;
pages reuse filenames. Use `migrate` for semantic HIR, `optimize` for placement,
`schedule` for instruction selection/finalization, and `showcase` for a larger
placement sequence. Rerun commands instead of trusting captured reports or
generated example TIR. Vendor specs have qualified topics; list before choosing.

`models NAME --source` exports resources owned by installed targets. Shared
`run.py`/`generation.py` orchestration comes separately through `tutorial
orchestrator`; it is not included in copied model directories.

Author a `@module` resolving its target, topology, entry function, tensor types,
meshes and storage scopes. Import CUDA targets from `tilefoundry_nvidia` and
Apple targets from `tilefoundry_apple`. The default target is CPU; specify the
device target. Bind open dimensions with `--dim NAME=VALUE`. Keep independent
PyTorch semantics separate from HIR and implementation. For layouts, partial
reductions, scheduling, ring buffers and compiler extensions, read
[layout and lowering contracts](references/layout-and-lowering.md).

## Compare authored designs

Write alternatives as distinct selections and analyze each with identical flags:

```bash
tilefoundry analyze model.py:Variant artifacts/analysis.txt \
  --compute-cost --memory --roofline --operands
```

`analyze` does not search layouts, select a parallelism strategy, or insert a
distributed resharding implementation. Keep logical shape/ownership, local
storage layout and actual communication distinct.

| Evidence | Interpretation |
| --- | --- |
| Compute work | Logical arithmetic under registered cost rules |
| Traffic/footprint | Modeled bytes/peaks for authored layout/storage; not measured transactions or network traffic |
| Capacity/reuse advisories | Structural diagnostics; absence does not prove cache fit or reuse |
| Roofline `ideal-ns` | Ideal bound using the named target's rates and modeled work/bytes |
| `--performance` | Nominal CTA-local timeline, not achieved occupancy or latency |

Record the target identity and reason in SM terms (`sm_90`, etc.). Same
architecture does not imply identical unit counts, bandwidth or cache capacity.
Inspect `target show IDENTITY` and instruction capabilities.

For hardware questions that affect the design—memory hierarchy, instruction
contracts, async copies, synchronization or pipeline choices—use the
[KernelWiki interface](../tilelang-optimization-loop/references/kernelwiki.md)
and read the installed KernelWiki skill. Query for the target architecture,
follow the cited sources, and record which finding informs the HIR choice.
Check applicability to the named target; hardware support and TileFoundry
backend support are separate questions. Before a kernel exists, treat these
findings as design evidence to validate after implementation.

KernelWiki hardware numbers are not calibration data. Verify device facts and
use [matched microbenchmarks](../tilelang/profiling-tilelang-programs/references/microbenchmark-calibration.md)
when rates will rank candidates. Keep specification, measured and inferred
profiles distinct, with sources and applicable regimes.

Use specification rates for a toy/ideal model when sufficient; label assumptions and expose logical
bytes. Do not calibrate NVLink/NVSwitch or inter-node costs merely to establish
an HIR design. Use measurements when a decision requires runtime performance.
For detailed kernel latency diagnosis, see the loop's
[cost model](../tilelang-optimization-loop/references/cost-model.md).

For tensor/layout obligations and large tiling domains, use
[constraint pruning](../tilelang-optimization-loop/references/constraint-pruning.md).
Reuse integer relations and installed solver libraries; preserve proof,
counterexample and unknown statuses. A calibrated estimate can defer a
candidate but does not prove that its layout is illegal or runtime cannot win.

The performance model prices the running mesh (`wave=active/physical`) and
footprint share, with a per-CTA Little's-law cap over stated in-flight bytes
when latency is supplied. It does not measure resident CTAs, register pressure,
dependent-chain latency or cross-trip software-pipeline overlap. Lower modeled
traffic can still lose at runtime. Treat disagreement as a diagnosis question.

## Check the runtime twin

Bind `@runtime_module`/`@runtime_func` implementations to the authored Module's
full function tree. Arguments are plain tensors in authored order; outputs are
positional (`output`, `output[0]`, etc.). See the
[TileLang twin template](references/tilelang-twin.md).

```bash
tilefoundry check model.py:RuntimeTwin --inputs random --weights random \
  --out output --fn allclose --atol 1e-5 --rtol 1e-4 \
  --json artifacts/runtime-check.json
```

Derive tolerances from dtype and rounding requirements; those example values
are not universal. State a predicate for every output; discrete outputs require
exact comparison. Repeat `--expected` per output file. For correlated inputs,
generate realistic files and use `--inputs files:...` instead of independent
random draws. A passing twin covers only exercised cases. Keep independent
tests for tails, strides, mutation, repeated calls and gradients. If evaluator
coverage refuses a construct, preserve the error and validate directly against
the reference; a refusal does not establish a kernel failure.

## Finalize and implement

Use `schedule candidates` for accept/refuse reasons and `schedule facts
INSTRUCTION --target IDENTITY` for operand/execution contracts. Choose explicitly
with `tf.schedule`, bind dimensions, then run `schedule finalize`. It writes a
Module, including a single-PrimFunction result. Analyze HIR for memory
diagnostics; finalized TIR has no printed HIR analysis header.

Translate the schedule into TileLang and record intentional changes in buffers,
launches and synchronization. For an explicitly requested compiler experiment,
validate `tilefoundry.compile()` on the exact program. Some scheduled/TIR paths
compile and run; unsupported emitters and missing synchronization lowering
still limit others. A compile success or instruction evaluator pass is not a
correct program binary. Inspect generated code and validate before timing.

The opt-in shared-memory `sync` family plans ordering; finalization does not
automatically emit its commits/waits/barriers. It is not a collective planner
or proof of synchronization in a TileLang implementation. Read the lowering
reference before relying on it.

## Review and hand over

Export authored HIR with `module_entry_to_dot` from `tilefoundry.inspection`
or inspect it with `Viewer(Model).serve(open_browser=True)`. Label selection,
dimensions and target, and retain a static export for review. Ownership diagrams
show declarations; they do not verify numerical equality or communication.

Use [design documents](../tilelang/writing-tilelang-design-docs/SKILL.md) to explain
implementation mapping and evidence. For optimization tasks return a
[stage report](../tilelang-optimization-loop/references/stage-handoff.md) with the
question, HIR/twin identities, flags, artifacts, limitations and next experiment.
Preserve one task contract, ledger and promotion system. Upstream source:
[tile-ai/TileFoundry](https://github.com/tile-ai/TileFoundry).
