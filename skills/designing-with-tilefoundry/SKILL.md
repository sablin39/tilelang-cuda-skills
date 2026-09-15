---
name: designing-with-tilefoundry
description: Describe model or operator HIR with TileFoundry, inspect topology and placement, run static cost analysis, visualize authored dataflow, and check a runtime implementation against its evaluator. Use for TileFoundry requests and high-level fusion, sharding, or memory-placement design before or alongside TileLang kernel work.
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

## Check the tool before changing the environment

Probe the active interpreter and CLI first:

```bash
python --version
tilefoundry --version
tilefoundry
```

TileFoundry currently requires Python 3.12 or newer and has a substantial
dependency set, including PyTorch, ISL, Apache TVM FFI, Graphviz, Transformers,
and OR-Tools. Its version constraints can downgrade an existing environment's
packages (observed on 0.0.2: transformers, tokenizers, and protobuf were
replaced with older releases). Inspect the selected release's current package
metadata before installing because its version constraints can conflict with
an existing TileLang or model environment.

If TileFoundry or a required dependency is missing, explain what is needed and
ask the user before installing or changing their environment. Prefer an
isolated environment unless the existing environment is demonstrably
compatible. The upstream installation command is:

```bash
python -m pip install tilefoundry
```

Do not add TileFoundry to this suite's default requirements merely to perform
ordinary TileLang work. Record the `tilefoundry --version`, Python version,
source revision when using a checkout, and target named by the authored module.

## Use the CLI as the primary interface

Start with the commands shipped by the installed version instead of assuming a
copied example is current:

```bash
tilefoundry tutorial
tilefoundry spec analysis
tilefoundry models
tilefoundry analyze --help
tilefoundry check --help
```

Use `tilefoundry models NAME --source` when a shipped model is close to the
task. Treat it as a reference for the active release and adapt it to the actual
operation, target, shapes, and topology.

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

Work the pages in order: `migrate` describes a published model's step as HIR,
`optimize` prices a placement change before any kernel exists, and `showcase`
takes one kernel through six analyze-driven placement decisions.
`tutorial orchestrator` lists shipped causal-LM decode sources. Use `spec` for
the normative grammar behind all of it, and `check --help` for the predicate
list and per-dtype tolerance derivation.

Never trust captured output in a page, README, or copied example: rendered
reports drift between releases (0.0.2's `analyze` rendering already differs
from the output its own tutorial page captured). The commands are the
contract — rerun them and read what the installed version actually prints.
This workflow was verified against tilefoundry 0.0.2.

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
which features it relies on. Architecture coverage depends on the installed
release (0.0.2 ships sm_90 and sm_100 architecture documents); a missing
architecture is added explicitly with `tilefoundry target add --document`,
never assumed.

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
reports. The fusion-boundary question is the canonical case, verified against
0.0.2 on the optimize tutorial's step: the variant split at a `@func`
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

## Analyze the authored design

Write analysis to a named artifact:

```bash
tilefoundry analyze path/to/model.py:ModuleName artifacts/model-analysis.txt \
  --compute-cost --memory --roofline --performance --operands
```

Bind every open dimension needed by the selection with `--dim NAME=VALUE`.
Read the report at both levels: its header summarizes the selected function and
target; annotations on each HIR operation show where work, traffic, footprint,
or predicted scheduling arises.

Interpret the analysis according to what it establishes:

| Output | What it supports |
| --- | --- |
| Compute cost | Logical operations and service demand by dtype and topology |
| Memory | Modeled traffic and peak footprint by storage scope |
| Roofline | A target-based lower bound and its modeled limiting resource |
| Performance | A predicted operation timeline, placement, fit, and wave count |

These are static results for the authored HIR. They do not show which TileLang
TIR or CUDA was emitted, and predicted nanoseconds are not measured GPU
latency (the predicted schedule's nanoseconds can be orders of magnitude above
the roofline bound; neither is a benchmark). Use them to compare design
alternatives and form implementation hypotheses. Select an implementation only
after the normal TileLang correctness and benchmark gates.

## Check the runtime twin

Bind a runtime implementation to the semantic module with TileFoundry's
`@runtime_module` and `@runtime_func` surface, then compare it with the HIR
evaluator. A runtime twin may call a TileLang kernel or another backend while
preserving the same semantic function tree; a verified TileLang twin template
lives in [references/tilelang-twin.md](references/tilelang-twin.md).

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
  -> TileLang implementation and specialized lowering
  -> independent correctness, GPU timing, and focused profiling
```

Record which HIR operations and storage transitions map to each TileLang
kernel, buffer, launch, and synchronization point. If TileLang lowering differs
from the authored design, update the explanation or the design instead of
presenting the HIR estimate as evidence for code that did not materialize.

Use the [TileLang optimization loop](../tilelang-optimization-loop/SKILL.md) for
candidate tracking and promotion. Preserve one task contract, plan, result
history, and benchmark harness; TileFoundry reports are additional evidence,
not a second promotion system.

The upstream source and current documentation live in
[tile-ai/TileFoundry](https://github.com/tile-ai/TileFoundry). Verify its active
CLI and specifications because the project is evolving quickly.
