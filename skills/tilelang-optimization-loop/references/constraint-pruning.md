# Tensor constraints and bounded candidate pruning

Use constraints to check tensor metadata and generate a bounded set of legal
implementation choices. Keep legality, performance prediction and experimental
selection separate. Read [cost models](cost-model.md) for calibration and
uncertainty; benchmark survivors through the existing task harness.

## Reuse the available solvers

| Component | Existing role | Appropriate reuse |
| --- | --- | --- |
| TileFoundry `isl-python` (`import isl`) | Integer iteration/access relations and footprint analysis | Affine/Presburger domains, map composition, overlap, coverage and dependence relations |
| TileFoundry OR-Tools CP-SAT | Memory placement with interference/alignment; unresolved pattern predicates | Finite choices, packing, bounded arithmetic and explicit resource/schedule objectives |
| TileLang `z3-solver` | Compiler reasoning, parallel-loop/synchronization checks, and target-specific scheduling | Integer/Boolean/bit-vector proof obligations, witnesses, unsat cores and finite tiling constraints |
| TileLang autotuner | Compiles, validates and benchmarks supplied configurations | Give `AutoTuner.from_kernel(factory, configs)` a filtered concrete config list; callable configs can specialize the domain |

Check imports/versions in the task interpreter before using them:

```python
import importlib.metadata as metadata
for package in ("tilelang", "tilefoundry", "z3-solver", "ortools", "isl-python"):
    try:
        print(package, metadata.version(package))
    except metadata.PackageNotFoundError:
        print(package, "unavailable")
```

Use Z3 directly for a task-local SMT query; do not call backend-specific
compiler scheduling internals as a generic CUDA performance API. CP-SAT works
with bounded integer variables, with Boolean choices and arithmetic/table
constraints; it is not Z3's array/bit-vector/floating-point theory interface.
ISL is particularly useful for linear integer relations and constant-divisor
floor/mod. Products of unknown dimensions and variable divisors require
specialization, another encoding, or an explicit unsupported result.
Keep arithmetic semantics explicit: CP-SAT integer division truncates toward
zero, which differs from Python floor division for negative operands. Restrict
the supported domain or encode the intended operation; do not silently substitute.

TileFoundry's allocation solver seeks the first feasible placement with a
heuristic seed, not minimum memory or maximum occupancy. A feasible footprint
above a budget does not prove no better packing exists. Ask a separate capacity
feasibility question, or use a sound lower bound, before rejecting a candidate.
Do not generalize specialized layout predicates: an inverse-projectability
test can demand compact order in addition to injectivity.

## Model a tensor through separate maps

For each relevant value, preserve:

1. **Domain:** element type, rank/extents, valid-index predicates, masks and
   specialization assumptions.
2. **Logical view:** view coordinates to canonical logical coordinates,
   including slice/transpose/split/fuse/tile composition and preconditions.
3. **Placement relation:** logical coordinates to storage participants and local
   coordinates. Replication needs a relation, not a unique-owner function.
4. **Storage:** allocation/version identity, memory space, participant instance,
   byte offset, access width, capacity and alignment.
5. **Effects and order:** read/write/reduction roles, dependencies, lifetimes,
   completion events, synchronization scopes and resource requirements.

Logical composition should produce a map and obligations without choosing a
CUDA address format. Storage lowering consumes the map and must discharge
those obligations. Preserve the relation between layers so a failed check can
name a logical element, participant, address and originating operation.

A view is not necessarily injective or surjective: broadcasts intentionally
repeat elements; slices intentionally omit them. Replicas at the same local
offset in different storage instances do not collide. A reduction may allow
multiple contributions but still needs an explicit combination/completion
contract. Do not infer replica equality from placement metadata alone.

## Ask for a counterexample

Prove a property by searching for its violation under explicit assumptions:

| Obligation | Counterexample query |
| --- | --- |
| Bounds | A valid logical index maps outside the permitted storage interval |
| Required injectivity | Distinct relevant logical indices map to overlapping bytes in the same storage instance |
| View equivalence | A valid index maps to different canonical coordinates under the two compositions |
| Coverage | A required logical element has no valid implementing participant/access |
| Race freedom under a chosen execution model | Two conflicting, un-ordered accesses have overlapping byte ranges and satisfiable guards |
| Legal reuse | A buffer version is overwritten before all required consumers complete |
| Schedule feasibility | A selected schedule violates a dependence, completion or actual resource constraint |

For an exact encoding: `unsat` proves this bounded obligation under the listed
assumptions, `sat` returns a witness, and `unknown`/timeout leaves it unresolved.
An over-approximate access relation can produce a spurious witness; validate
it before rejecting the real program. Never equate `unknown` with infeasible.

For example, this complete CPU query checks a fixed 4x4 mesh storing a 16x16
tensor in 4x4 local tiles. It asks for a collision between distinct elements:

```python
import z3

i, j, p, q = z3.Ints("i j p q")
solver = z3.Solver()
solver.set(timeout=5000)
solver.add(*[z3.And(0 <= x, x < 16) for x in (i, j, p, q)])
def location(row, col):
    return row / 4, col / 4, (row % 4) * 4 + col % 4
solver.add(z3.Or(i != p, j != q))
solver.add(*[a == b for a, b in zip(location(i, j), location(p, q))])
status = solver.check()
print(status)  # unsat for this fixed mapping
if status == z3.sat:
    print(solver.model())
elif status == z3.unknown:
    print(solver.reason_unknown())
```

Offsets here are single-element positions of one dtype; real alias checks need
byte intervals, allocation/version IDs and storage instances. Changing local
row stride from 4 to 3 creates a collision. This is not a proof of floating-point
results, GPU scheduling or synchronization. Start with known rank, concrete
candidate tiles, affine maps and constant divisors. Arbitrary tensor-value
equivalence and floating-point reductions need substantially stronger semantics.
For packed sub-byte values, model bit ranges or report unsupported; rounding
each element to a separate byte changes aliasing semantics.

## Make rejection reasons precise

- **Proven infeasible:** the exact implementation template contradicts a hard
  contract. Examples include unsupported instruction shape, an actual launch
  limit, or mandatory storage exceeding capacity. Exact divisibility is hard
  only for a template without a valid masking/padding path. Low occupancy is
  not infeasibility; estimated registers are not a verified register count.
- **Dominated by a justified bound:** even the candidate's sound optimistic
  runtime bound cannot satisfy the acceptance objective. State the rate/work
  assumptions proving the bound; empirical coefficients normally do not qualify.
- **Deferred by budget:** a fitted cost/ranking model predicts poor performance.
  Keep the candidate available for later exploration; this is not a proof.
- **Unresolved:** unsupported encoding, unknown hardware fact, timeout or
  solver `unknown`. Keep or separately investigate it, rather than discard it.

Enumerate finite tile/stage/warp domains and precompute products, rounded
footprints and applicable instruction tables when possible. This avoids putting
unrestricted nonlinear equations in the solver. Incremental checks and named
assumptions make a failed choice explainable. An unsat core is one sufficient
conflicting subset, not necessarily the smallest or sole explanation.

Before compiling, use only sound mandatory-resource facts for hard rejection.
For survivors, compile enough candidates to obtain real shared allocation,
registers, spills, instruction selection and launch attributes. Refine the
model from these facts, then run numerical validation and timing on the chosen
subset. Grouped compilation can amortize this middle stage without timing every
candidate. Candidate metadata keys must be stripped before passing configs to
the kernel factory; its arguments and supported pass-config keys define that API.

## Evaluate pruning as a search policy

Score complete candidate descriptions, including staging and lowering choices.
When they are not fixed, retain plausible completions instead of condemning a
partial schedule. Use predicted winners plus structural diversity and an
explicit exploration budget. Statistical lower/upper time envelopes can justify
deferral, but marginal confidence intervals do not guarantee retention of the
adaptively selected winner. Audit deferred choices and reopen a regime on failure.

On small representative workloads, measure the feasible domain once and replay
selection policies at equal GPU-trial budgets. Compare legality-only search,
cost-ranked selection and cost-plus-exploration. Report epsilon-best hit rate,
best/top-k recall, missed-winner rate, latency regret, compile/calibration/GPU
cost, and results across seeds and regimes. Use tie/noise rules. Do not claim
global optimality from a shortlist or only average prediction error.

Store solver version/options, IR and calibration hashes, domains/assumptions,
status, witness/core, bound provenance and candidate classification with the
task ledger. Attach these artifacts to a `model` check in the
[stage report](stage-handoff.md); the usual correctness/benchmark gates still
control promotion.

## Design references

- [CuTe layout algebra](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html): compositional indexing vocabulary and operation preconditions.
- [MLIR shape inference](https://mlir.llvm.org/docs/ShapeInference/): separate element type, rank and extent contracts; includes design requirements as well as implemented interfaces.
- [ISL manual](https://libisl.sourceforge.io/user.html#Scheduling): integer relations and separation of validity dependences from optimization objectives.
- [Shardy representation](https://openxla.org/shardy/sharding_representation): explicit replication, mesh-axis structure and non-divisible sharding policies.
- [GPUVerify](https://www.doc.ic.ac.uk/~afd/papers/2014/AVoCS.pdf): pairwise access reasoning under a stated concurrency abstraction; do not assume it covers arbitrary async GPU execution.
- [Programming Z3](https://z3prover.github.io/papers/programmingz3.html) and [CP-SAT](https://developers.google.com/optimization/cp/cp_solver): solver interfaces, supported domains and result statuses.
- [ROLLER](https://www.usenix.org/system/files/osdi22-zhu.pdf): hardware-aware tile construction with measured final choices.
- [Ansor](https://www.usenix.org/system/files/osdi20-zheng.pdf): search across complete program structures and ranking validation.
- [Learning to Optimize Tensor Programs](https://proceedings.neurips.cc/paper/2018/file/8b5700012be65c9da25f49408d959ca0-Paper.pdf): learned ranking with exploration; uncertainty-based acquisition is a design to evaluate, not an assumed improvement.
