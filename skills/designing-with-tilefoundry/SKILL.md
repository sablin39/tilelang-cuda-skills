---
name: designing-with-tilefoundry
description: Describe model or operator HIR with TileFoundry, inspect topology and placement, run static cost analysis, visualize authored dataflow, and check a runtime implementation against its evaluator. Use for TileFoundry requests and high-level fusion, sharding, or memory-placement design before or alongside TileLang kernel work. Machine-checked HIR and static analysis; for human-facing kernel design write-ups use writing-tilelang-design-docs.
---

# Designing with TileFoundry

Use TileFoundry as an authored model and operator layer. It can describe logical
HIR, topology and placement, evaluate the description, estimate work and data
movement, and compare a runtime twin with that evaluator. Carry the selected
design into TileLang for compiler lowering, GPU validation, measurement, and
profiling.

This workflow is useful when the decision spans several operators or kernels,
changes a fusion boundary, places values across a topology, or needs an
independent semantic model. For a focused question about an existing TileLang
`PrimFunc`, start with the relevant TileLang writing, debugging, profiling, or
optimization skill.

Know what it costs (single-operator ablation, measured 2026-10-20): on a
coldstart whose deliverable is just a fast kernel plus a pragmatic fusion
decision, a TileLang-skills-only arm finished 6-8× cheaper and produced the
*faster* kernel. This layer pays when variants must be priced before any kernel
exists, when an independent semantic anchor is required (model-scale work),
or when the static evidence itself is the deliverable — not as a
kernel-speed tool.

## Use the vendored checkout

This suite vendors TileFoundry as a submodule at
[external/TileFoundry](../../../external/TileFoundry), pinned to a recorded
upstream commit (currently main @ `25fbbaa`), and installs it
editable into the suite's `.venv`:

```bash
git submodule update --init external/TileFoundry
git -C external/TileFoundry submodule update --init third_party/cutlass
uv pip install --python .venv/bin/python -e external/TileFoundry cmake ninja
```

Editable-from-source is required, not a convenience: the CUDA backend resolves
`include/tilefoundry/runtime.h` and CUTLASS relative to a source checkout, and
the PyPI wheel ships neither (verified 2026-10-15 on 0.0.3.dev38), so
`tilefoundry.compile()` cannot build a kernel from a wheel install. The
`third_party/cutlass` submodule must be initialized or every kernel build fails
on `cute/tensor.hpp`. `cmake`/`ninja` must be on `PATH` for the same reason.

The editable install's version string comes from git metadata; a shallow
submodule clone reports `0.1.dev1+g<sha>` instead of the true release line.
Record the submodule's pinned commit, not the string.

TileFoundry requires Python 3.12+ and pulls a substantial dependency set
(ISL, OR-Tools, Transformers, Graphviz, Apache TVM FFI). On 2026-10-15 the
install into this suite's venv added `transformers 5.14.1`, `ortools 9.15`,
and `isl-python 0.1.11` without touching torch or tilelang; re-check that
combinations still hold whenever the submodule pin moves, and prefer
`uv pip install --index-url <mirror>` when PyPI reachability is poor.

Probe the active interpreter and CLI before relying on them:

```bash
python --version
tilefoundry --version
tilefoundry
```

Do not add TileFoundry to this suite's default requirements merely to perform
ordinary TileLang work. Record the pinned commit, Python version, and target
named by the authored module.

## Use the CLI as the primary interface

Start with the commands shipped by the installed version instead of assuming a
copied example is current:

```bash
tilefoundry tutorial
tilefoundry spec analysis
tilefoundry models
tilefoundry analyze --help
tilefoundry check --help
tilefoundry schedule --help
```

Use `tilefoundry models NAME --source` when a shipped model is close to the
task. Treat it as a reference for the active release and adapt it to the actual
operation, target, shapes, and topology. The registry contents change between
releases: list what the installed release ships with `tilefoundry models` at
the pinned checkout instead of assuming a particular architecture is there,
and when one matches the task, its HIR is the fastest correct starting point.

Author a `@module` with its target, topology hierarchy, entry function, tensor
types, meshes, sharding, and storage placement. Let the parser/type checker
reject inconsistent descriptions. Keep this HIR separate from the PyTorch
oracle and the TileLang implementation so agreement between them supplies
useful independent evidence.

## Learn the authoring DSL from executable tutorial sources

The authoring DSL (`@module`/`@func`, `Mesh` and `Topology`, `tf.reshard`,
shard annotations such as `ROWS @ m.x`, storage scopes) is learned from the
installed release, not from prose about it. Each tutorial page ships its
programs embedded in the page itself and opens with the command that extracts
them; extract and run, never retype:

```bash
tilefoundry tutorial optimize > optimize.md
# the page's own opening awk block extracts its programs into .py files
```

Extract each page into its own directory: pages reuse program filenames, so
extracting two pages side by side overwrites the first. The awk blocks work
verbatim at the pinned commit (re-verified on the `optimize` page, 2026-10-15).

Work the pages in order: `migrate` describes a published model's step as HIR,
`optimize` prices a placement change before any kernel exists, `schedule`
chooses schedules and instructions for a backend and lowers them with
`schedule finalize`, and `showcase` takes one kernel through six analyze-driven
placement decisions. `tutorial orchestrator` lists shipped causal-LM decode
sources. Use `spec` for the normative grammar behind all of it, and
`check --help` for the predicate list and per-dtype tolerance derivation.

Authoring notes verified at the pinned commit (2026-10-15):

- A small weight shared against activations sharded over a composed
  (cta, thread) mesh still cannot be placed by `tf.reshard` in any form: the
  binary op then fails verification with a precise
  `incompatible output shard Split(axis=0) vs Split(axis=1)` — broadcast and
  split on one mesh axis do not compose in an authored elementwise op, and the
  0.0.2-era verbose workaround is gone with `ShardMesh`. The current idiom
  brings the shared value to a matching shard with an explicit scheduled copy
  into a thread-mapped destination (see the `schedule` tutorial page and the
  shipped examples' gamma handling), not with a shard annotation.
- `tf.local` no longer exists in the DSL; the old crash report is moot.
- `Topology("thread", N)` counts per-CTA threads (limit 1024), so a
  `ROWS * warps * 32`-style pattern breaks as ROWS grows; size the thread
  topology per CTA, not per launch.
- A `tf.tile` window is a window only as a *direct* slice index; any
  arithmetic on it (`f = cw * NC + i`) folds it to its scalar start and the
  slice silently loses a rank (scan case, 2026-10-19). Index with the two
  windows directly and reshape flat↔3D at pass boundaries instead.
- A scalar index drops the axis: `cs[f, 0, :]` on a `[TILES,1,N]` tensor
  yields rank-2. Size-1 axes need an explicit `0:1` range slice.
- One mesh layout extent per module: mixed lane counts across passes are
  rejected (`logical axis 0 size 40 is not divisible by mesh extent product
  128`). Lane counts are divisibility-driven wave widths, not launch
  configs — and `tf.tile` needs exact division, so sweep dimensions must be
  multiples of the wave width.

Never trust captured output in a page, README, or copied example: rendered
reports drift between releases and the DSL itself has broken shipped examples
within days (the `rmem_layout=` schedule attribute was removed two days after
the flagship example that used it was verified). The commands are the
contract — rerun them and read what the pinned checkout actually prints.
This workflow is verified against the vendored submodule at `25fbbaa`
(verified 2026-10-20).

## Select the target by SM capability

Choose the target by its architecture, not by a product name. List what the
installed release can construct, then inspect the facts behind one:

```bash
tilefoundry target list
tilefoundry target show IDENTITY
```

A CUDA target value is a product document layered on an architecture document;
`target show` reports both. The architecture layer records the compilation
identity (`sm_90`, `sm_100`, ...), `instruction.capabilities` (the instruction
families that SM generation's code path may select, such as the tensor-core
MMA variant and async copy engines), `instruction.compute_dtypes`, and storage
capacities per topology owner. Reason and record in these SM capability
terms — sm_90, sm_100, sm_120 — rather than individual product names, so the
design evidence transfers across products of one generation and states exactly
which features it relies on. Architecture coverage depends on the checkout — enumerate it with
`tilefoundry target list` at the active pin rather than assuming a product or
architecture exists; a missing one is added explicitly with
`tilefoundry target add`, never assumed.

Registering a target for the attached GPU (recipe verified 2026-10-16 at the
pinned commit):

1. Read the device in SM terms: `tvm.target.Target.from_device("cuda")` (or
   torch's `get_device_capability`) for `arch=sm_*`; SM count and L2 capacity
   from the runtime's device properties; max clocks from
   `nvidia-smi -q -d CLOCK`.
2. If `target list` already has a product on the same SM architecture,
   topology-dependent results (wave counts, per-wave L2 advisories,
   instruction facts) already transfer — only rates may differ. Clone that
   product's device TOML (shipped under the package's `target/*/hardware/`)
   as the starting document.
3. Edit the rates: HBM bandwidth = max memory clock × 2 × bus width / 8 (bus
   width = HBM stack count × 1024 bits) or the vendor sheet; compute
   throughputs from SM count × clock — and if the product might have cut-down
   tensor cores, measure the dense bf16 peak with one cuBLAS GEMM instead of
   assuming the full die rate. Keep every fact's `origin`/`source`/`conditions`
   honest: mark derived numbers with their derivation and measured ones with
   the measurement.
4. `tilefoundry target add --document your-device.toml` (add the architecture
   document first if the SM generation itself is new). Registration persists
   in `<sys.prefix>/share/tilefoundry/registry.toml`, digest-checked on
   replay; `--registry PATH` overrides per invocation.
5. Verify: name the new target in the HIR, re-run one `analyze --roofline`,
   and check the floor scales by exactly the bandwidth ratio. With the
   device's true HBM rate, a bandwidth-bound kernel's roofline floor should
   land near its measured time (observed within ~4% on a kernel measured at
   79% of DRAM peak, 2026-10-16).

For what a capability means for kernel design — MMA shapes, async-copy
mechanics, warp-specialization idioms under a given SM generation — query the
[KernelWiki interface](../tilelang-optimization-loop/references/kernelwiki.md)
with a matching architecture filter instead of guessing from the facts list.

`analyze` is static: it prices the authored program against the named target's
published rates and runs without a GPU, so the target named in the module need
not match the attached device — but its numbers describe that named target,
and a cross-architecture comparison is meaningless. `check` executes on a real
device (the selection's target by default, or `--device`); there the
architecture the implementation was written for and the device must agree.

## Compare design variants with analyze

`analyze` never searches: no candidate enumeration, layout search, or
automatic resharding. The design loop is therefore authored, not launched —
write each alternative as its own `@module` (or boundary placement) in one
source file, analyze each selection with identical flags, and diff the
reports. The fusion-boundary question is the canonical case, exercised on the
optimize tutorial's step (the page and its analyze blocks are current at the
pinned commit): the variant split at a `@func`
boundary hands the middle rows across it (28672 B written and read back
through gmem); the fused variant removes exactly that round trip, the
roofline bound falls with it, and `bound-by` stays `memory` — so cutting
arithmetic would buy nothing in either variant. The diff *is* the decision,
priced before any kernel exists.

Rank candidate directions in the optimization-loop draft by these diffs —
fusion boundaries, storage placement across rmem/smem/gmem, sharding and
topology — and reject statically dominated designs before compiling anything.
A variant that wins on paper still faces the normal TileLang correctness and
benchmark gates; a variant that loses on paper needs an argument for why the
model is wrong before it deserves a kernel.

Calibrate the numbers against the attached GPU: relative diffs transfer to
measurement, absolute nanoseconds do not. Verified on 0.0.2 (fused vs split
residual-add+RMSNorm, [4096,4096] bf16 on an L20Z): the 1.67× gmem-traffic
diff predicted the direction and the `bound-by: memory` diagnosis, and the
measured latency ratio landed at 1.3-1.4× for both TileLang and
torch.compile — the streaming split kernels run closer to copy peak, so the
time ratio comes in under the traffic ratio. When absolute floors matter,
register a target document carrying the device's actual rates (recipe in the
target section above) instead of mentally scaling a fleet product's numbers —
verified 2026-10-16: with the device's true HBM rate the roofline floor of a
kernel measured at 79% of DRAM peak lands within ~4% of its measured time.

## Analyze the authored design

Write analysis to a named artifact:

```bash
tilefoundry analyze path/to/model.py:ModuleName artifacts/model-analysis.txt \
  --compute-cost --memory --roofline --operands
```

Bind every open dimension needed by the selection with `--dim NAME=VALUE`.
Read the report at both levels: its header summarizes the selected function and
target; annotations on each HIR operation show where work, traffic, footprint,
or predicted scheduling arises.

Interpret each family by what it is actually worth (trust table verified
2026-10-16 at the pinned commit, cross-checked against ncu and torch.profiler
measurements of the analyzed variants):

| Output | Trust | What it supports |
| --- | --- | --- |
| Compute cost | exact | Logical FLOPs by dtype and topology — arithmetic intensity and recompute-tax diffs between variants |
| Memory traffic/footprint | exact | Logical bytes and per-level peaks — fusion-boundary and placement diffs; this is the ranking axis |
| Capacity errors, L2 reuse-window advisories | trust, within their coverage | They fire on the buffer stream measurement later blames — but coverage is structural, not semantic: verified 2026-10-19 that the window machinery keys on staging buffers reused *within one loop nest* and misses capacity overflow across *sibling top-level loops* (a two-pass scan's chunk-state stream crossed 2× L2 with no advisory) |
| Roofline bound/ideal-ns | trust with a rate-correct target | The floor and its limiting resource; with the device's true HBM rate registered, the floor lands near a bandwidth-bound kernel's measured time — and near a compute-bound one: a 4096³ GEMM floor came within 0.5% of measured cuBLAS once the target document's derating note and cold-L2 compulsory reads were priced in (2026-10-19) |
| Performance predicted-ns/waves | do not rank with it | See the warning below |

Two sharpening principles, both verified against measurement (2026-10-19/20):

- **Traffic/roofline ratios price a decision's *ceiling*, not its guaranteed
  margin.** A static fused-by-1.47× ratio materialized as 1.48× measured only
  with an equally-well-optimized implementation pair; a fidelity-constrained
  pair measured a tie on the same question. When a static margin matters,
  budget the optimization effort to realize it.
- **Whether the occupancy blind spot bites is kernel-structure ×
  op-structure dependent, not a function of lane count.** The DPLR inversion
  (below) needed dependent MMA chains without pipelining; a pipelined
  elementwise scan at the same 40/132 lane fill was issue-rate-bound and the
  static ranking's *direction* held (its magnitude did not).

`--performance` places operations on a CTA-local nominal timeline with fixed
per-CTA service rates: compute and movement overlap *within* one op
(duration = max of the two), occurrences with disjoint buffers may overlap,
but loop trips run in lockstep (`trips × stride_ns`) — cross-trip software
pipelining (the TMA-ring/WGMMA prefetch idiom) is invisible to it by
construction ("what overlaps is what the program's own placement made
independent"). Two further artifacts make its predicted nanoseconds unusable
for ranking variants, both verified against measurement: it bills
`insert_slice`/`tf.zeros` at whole-tensor-copy cost per update (a streaming
output write of 40 MB was billed gigabytes, inflating one kernel's prediction
27× over its measured time; re-confirmed on a second operator family
2026-10-19, 55-60× over measured on both scan variants), and it cannot see
mesh-lane under-fill — a 40-of-132-lane scan costs the same per lane as a
full launch, `waves=1` and `wave=132/132` on every header, so in
occupancy-gated regimes it names the wrong winner (it preferred a fused scan
over the split that measurement showed 1.3× faster). The spec itself states
performance "MUST NOT be read as a guarantee about physical occupancy".
Occupancy-, latency-, and pipeline-overlap-gated questions belong to ncu and
torch.profiler sweeps; use `--performance` output only to inspect per-op
scheduling structure, never as the variant verdict.

These are static results for the authored HIR. They do not show which TileLang
TIR or CUDA was emitted. Use them to compare design alternatives and form
implementation hypotheses; select an implementation only after the normal
TileLang correctness and benchmark gates.

## The compile() backend works for ops, not yet for programs

The package ships a Python-only HIR/TIR→CUDA compiler (`tilefoundry.compile`)
with no CLI verb. Status at the pinned commit (2026-10-15, all verified on the
vendored checkout unless said otherwise):

- **The environment is solved by the vendored editable install** — kernel
  builds find `runtime.h` and CUTLASS under the source checkout; op-level
  compile-and-run passes on the suite's GPU (`tests/ops/tir` copy and
  elementwise). The PyPI wheel still ships no headers, so this never works
  from a wheel.
- Mesh-capable lowering works: `schedule finalize` lowers cta/thread-mesh HIR
  and the emitter handles `MeshScope`; upstream's own mesh op tests pass.
- Handwritten TIR compiles to respectable CUDA (1.11× `torch.compile` on the
  fused RMSNorm case, still 1.39× behind the TileLang kernel; measured
  2026-10-08).
- But there is still no supported path from scheduled HIR to a correct
  program binary. Known gaps found in one end-to-end drive: the TMA op
  `T.copy_async_tensor` has no emitter by design, so the schedule tutorial's
  "optimal" GEMM cannot compile at all; four emitter bugs broke ordinary
  lowered programs (slice coordinates on Dim arithmetic, `MeshCoord`
  coordinates, missing `Cast` emitter, `Fill` on kernel params — small fixes
  validated against upstream's own test suite, submitted upstream as PR
  tile-ai/TileFoundry#240); and with
  those fixed plus a hand-added reduce workspace, the compiled RMSNorm reads
  shared memory out of bounds because the runtime reduce dispatch mis-counts
  `warps_per_group` for the composed cta×thread mesh that every multi-CTA
  finalize produces.

Treat `compile()` as experimental: fine for op-level probes, not a measured
floor and not the implementation route. Upstream's own flagship example
(`examples/qwen3_8_27b_prefill-tilelang`, a KDA/gated-delta-net prefill
reported 1–10% faster than SGLang 0.5.21) does not use it either — TileFoundry
authors/checks/analyzes/schedules/finalizes, and TileLang kernels implement
the finalized TIR. That is the integration shape to copy. The example's own
artifacts were verified against `f8fd71e` and no longer parse on later main
(the `rmem_layout=` attribute was removed); read them at our pin, run them at
theirs.

## Check the runtime twin

Bind a runtime implementation to the semantic module with TileFoundry's
`@runtime_module` and `@runtime_func` surface, then compare it with the HIR
evaluator. A runtime twin may call a TileLang kernel or another backend while
preserving the same semantic function tree; a verified TileLang twin template
lives in [references/tilelang-twin.md](references/tilelang-twin.md) (PASS at
the pinned commit with tilelang 0.1.15, including `equal` on an fp8 output,
2026-10-15).

The twin contract:

- A twin is written per Module and must cover all of that Module's functions
  at once. `check` names the twin class; the twin states which authored Module
  it stands for, so naming the implementation reaches what it is judged
  against.
- Each `@runtime_func` takes plain torch tensors in the authored parameter
  order and returns outputs positionally: one tensor is `output`; a tuple's
  tensors are `output[0]`, `output[1]`, and so on in return order.
- If the Module declares weights, state their source with `--weights random`
  or `--weights ckpt:DIR`. A weight reached with no source is refused where it
  is first asked; weights come from the same draw on both sides of the
  comparison.

State at least one predicate for every output; `check` deliberately provides no
default tolerance and no default predicate:

```bash
tilefoundry check path/to/model.py:RuntimeTwin \
  --inputs random --weights random \
  --out output --fn allclose --atol 1e-5 --rtol 1e-4 \
  --json artifacts/runtime-check.json
```

`--inputs random` draws each activation independently. When a downstream
segment consumes a deterministic transform of an upstream activation (a
cumsum, a gate, a decay chain), random draws break the correlation and the
twin can FAIL for non-semantic reasons — generate those inputs from the real
implementation and pass `--inputs files:...` instead. A failed run prints
this warning after the fact; better to choose `files:` up front when
activations are coupled.

Derive tolerances from the operation, dtype, and serial rounding points
(`check --help` gives the per-dtype derivation). On a discrete output (fp8,
integer indices), one wrong value is a total failure: aggregate predicates are
refused there, so use exact comparison and expect it to be demanding — a
different reduction order in the implementation can move a bf16 rounding and
flip a code. Add representative input files, dimension bindings, devices, and
output predicates as required by the contract. A passing twin establishes
agreement with the authored evaluator for those cases; run the suite's
TileLang tests for the independent reference, tails, mutation, gradients, and
repeated-call state.

Evaluator coverage, measured 2026-10-19: an 8192-step recurrence whose
slices carry loop *windows* (`g[cw, t, :]`) checks fine (both scan twins
PASS, max_violation 0), but a persistent kernel whose loop bound carries a
mesh coordinate is refused — `evaluator: LoopRegion start: resolve_dim:
non-dim Call target MeshCoord`, same family as the documented TF-1. When
that appears, fall back to direct validation against the PyTorch reference
and record it; it is an evaluator gap, not a kernel failure.

## Visualize HIR for review

For a portable source artifact, serialize the entry graph as DOT:

```python
from pathlib import Path

from tilefoundry.inspection import module_entry_to_dot
from model import Model

Path("artifacts/model-hir.dot").write_text(
    module_entry_to_dot(Model), encoding="utf-8"
)
```

For interactive inspection, use the viewer:

```python
from tilefoundry.inspection import Viewer
from model import Model

Viewer(Model).serve(open_browser=True)
```

The viewer serves a local application with pan/zoom, search, collapsible
function regions, data-dependency highlighting, and per-expression details.
It downloads checksum-pinned Graphviz-WASM and D3 assets on a cold cache. For
offline work, pre-populate a cache and set `TILEFOUNDRY_VIEWER_ASSET_DIR`.
The viewer needs its local server; it is not a standalone HTML report.

In a human-facing design document, label this as an **authored-model** graph and
name the TileFoundry version, source selector, target architecture (in SM
capability terms), and dimension bindings.
Include a static export or focused screenshot for durable review, and link the
DOT, authored HIR, analysis report, and viewer launch command. Follow the
[TileLang design-document skill](../tilelang/writing-tilelang-design-docs/SKILL.md)
for the implementation mapping and compiler/runtime figures.

## Hand the design to TileLang

Keep the evidence chain explicit:

```text
PyTorch/reference semantics
  -> TileFoundry authored HIR, target, topology, and placement
  -> static analysis of each design variant and evaluator/runtime-twin check
  -> (optional) tf.schedule + schedule finalize: the finalized TIR as the
     implementation specification
  -> TileLang implementation and specialized lowering
  -> independent correctness, GPU timing, and focused profiling
```

The schedule step is how upstream itself connects the layers: scheduled HIR
records instruction and placement choices, `schedule finalize` lowers it to
verified TIR annotated with the memory analysis, and the TileLang kernel is
written against that TIR — see the `schedule` tutorial page and, at full
scale, `external/TileFoundry/examples/qwen3_8_27b_prefill-tilelang` (per-op
scheduled HIR, finalized TIR, and the TileLang kernels implementing them).
The finalized TIR is a specification to implement, not compiler input — do
not feed it to `compile()` expecting a working binary (see above).

The suite has now driven this leg itself (verified 2026-10-19): a
suite-authored scheduled 4096³ GEMM finalized first try (tile shapes are
shape-independent; only the grid/swizzle arithmetic is re-derived per shape —
note K/BK ≡ 1 (mod 3) shifts the ring rotation one slot per tile, which
finalize's index arithmetic already encodes), and the TileLang kernel
implementing the TIR reached 94.8% of cuBLAS, bitwise-equal to it. The
mapping is mechanical except where the TIR is deliberately silent: the
synchronization protocol (mbarriers, parities, `wait_wgmma`, `set_max_nreg`)
is the kernel author's to add, and that is where most of the implementation
effort goes — log every deliberate deviation from the TIR with its reason.

Getting a program finalize-ready (verified 2026-10-20 at the pinned commit):

- `schedule candidates prog.py out.txt` lists, per unscheduled site, which of
  the target's instructions accept or refuse, with reasons. Read refusals as
  diagnostics: every instruction refusing with `mesh ... does not match
  Mesh(('thread',), ...)` means the HIR places data at CTA level only, and
  instruction selection additionally needs a thread-level placement.
- `schedule finalize` rejects unscheduled programs with an actionable error
  naming the site, its instruction candidates, and the remedy, e.g.
  `aqk_c has instruction candidates [T.copy, ...] but no automatic selection;
  write tf.schedule to choose the instruction and its attributes`. When the
  candidate list is empty or surprising, go to the candidates report for the
  refusal reasons. Finalize also needs every open dimension bound with
  `--dim` and `tf.schedule(...)` choices recorded in the HIR; the `schedule`
  tutorial's own `optimal.py` (extract it with the page's awk block)
  finalizes cleanly at the pin and is the working reference for the
  annotation idiom.
- `schedule facts INSTRUCTION --target IDENTITY` prints one instruction's
  contract (execution mesh, parameter and operand constraints) — read it
  before choosing an atom by name.

Record which HIR operations and storage transitions map to each TileLang
kernel, buffer, launch, and synchronization point. If TileLang lowering differs
from the authored design, update the explanation or the design instead of
presenting the HIR estimate as evidence for code that did not materialize.

Use the [TileLang optimization loop](../tilelang-optimization-loop/SKILL.md) for
candidate tracking and promotion. Preserve one task contract, plan, result
history, and benchmark harness; TileFoundry reports are additional evidence,
not a second promotion system.

The vendored source lives at
[external/TileFoundry](../../../external/TileFoundry); upstream is
[tile-ai/TileFoundry](https://github.com/tile-ai/TileFoundry). The project
moves fast and breaks its own examples within days, so verify the active CLI
and DSL against the pinned checkout after every submodule bump, and re-run
the probes in this file (tutorial extraction, twin check, one `tests/ops/tir`
test) before trusting the workflow at a new pin.
