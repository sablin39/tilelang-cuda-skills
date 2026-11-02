# Layout, scheduling and lowering contracts

Read the pinned source's `docs/spec/analysis.md`, `docs/spec/passes.md` and the
vendor package's `cuda/docs/spec/` for compiler changes. Backend code and facts
live under `targets/nvidia` or `targets/apple`; shared IR/analysis remains core.

## Layout and ownership

- `Layout(shape)` makes concrete row-major defaults; static unit leaves have
  zero stride, symbolic strides remain symbolic, and rank zero is
  `Layout((), ())`. Authored parameter/return annotations constrain signatures.
- `ShardLayout` storage ownership compacts owner-level and coarser splits
  into per-instance storage while retaining local residual extents. Preserve
  authored affine maps for slicing, local views and instruction projection.
- Plain/sharded slices do not invent a composed offset. Existing composed
  layouts retain/update their offset; alias decisions compare absolute
  addresses from slice starts. Test addresses as well as printed layouts.
- A `Reshard` changing any mesh axis's `Partial` state performs a reduction
  and incurs traffic. It is not a view even when allocation reuses the buffer.
  Splits over extent-one mesh axes do not change alias boxes.

Use explicit slice windows. Arithmetic on a `tf.tile` window can turn it into
a scalar start; check inferred rank before reshaping. Scalar indexing and range
slicing have different rank effects. Inspect topology extents, divisibility and
instruction mesh requirements rather than treating meshes as arbitrary launch
tuning knobs.

## One emission contract

Core `analysis/emission.py` is shared by lowering and analyses that need to
know what it will issue. It classifies calls, selects instructions, maps operand
types/roles, decides alias-dependent copies/fills, determines issuing meshes and
cuts instruction windows. Reuse it instead of reconstructing a second schedule.

Finalization takes the operand's local view before cutting an issue window.
A statically zero-start window spanning the whole operand uses the operand
itself (or its pointer retyped for the instruction). Partial/dynamic windows
still need their own representation and backend support.

Ring stages are tuples of bound TensorViews; the selected stage is bound once
per consuming scope. Compile, finalize and scaffold share `default_pipeline`.
Full HIR lowering needs memory-analysis metadata. Python APIs are
`finalize(analyzed, *, authored)` and `candidates(module, entry, *, source)`;
the latter expects a checked program. Prefer the CLI loader for files and dims
over removed private loaders. `Module.entry` is the host starting point;
`build` requires a CPU host root, and linked CPU hosts follow the call paths.

Backend extensions use `Target.resolve`/`bind` and pass creation hooks, replacing
the monolithic code-generator API. `InsertHostEntryPass.create` and
`ConvertHIRToTIR.create` admit target-owned implementations. Op registration
needs explicit dialect/category, effects, capabilities and applicable
type/eval/cost/schedule/emission contracts. Evaluation does not supply codegen.

## Shared-memory ordering analysis

Request `sync` explicitly through the analysis API; it is not in default
finalization. It depends on memory placement and attaches `SyncPlanMetadata`
with ordered `commit`, `wait`, `barrier` steps keyed by liveness positions.
Inspect the API/spec and vendor `analysis/test_sync.py` for invocation; do not
assume an `analyze --sync` CLI flag exists.

The planner shares emission decisions, `IterationScope` and allocated byte
regions with lowering. It orders overlapping read/write accesses, handles loop
backedges to a fixed point and commits/waits pending `cp.async` writes before
conflicts. It orders whole allocation regions, not particular windows/stages.
Unknown regions overlap everything and may require conservative barriers.

The completion registry defaults undeclared ops to `UNKNOWN`. The planner
refuses conflicting accesses after unknown completion, unknown work still in
flight at return, thread-divergent barriers, and retained callees holding shared
memory. TMA/WGMMA completion is not declared for this analysis, so those paths
can be refused. Preserve the refusal; an empty/default plan is not proof of safety.

**The plan is not yet consumed to emit CUDA synchronization.** Inspect actual
waits/barriers separately. This is within-kernel shared-memory ordering, not
intra-node/inter-node collectives or distributed resharding.

## CUDA launch and instruction boundaries

Synthesized TIR hosts request dynamic shared memory reached by device functions
and variants/ring stages using per-CTA storage accounting. Lowering also carries
analyzed shared-memory peaks into launches. Authored hosts keep authored sizes.
Requests above 48 KiB raise the kernel's dynamic shared-memory limit; static
plus dynamic storage may still exceed capacity. Launch refusals raise runtime
errors; synchronize to detect asynchronous execution faults.

CUDA emits a used dynamic shared-memory base once at kernel scope. The TIR
printer preserves non-default launch cluster, stream, dynamic shared memory and
attributes. Round-trip inspection must preserve them; compilation alone does
not validate their runtime effect.

`StMatrix` and additive `CopyReduceAsyncTensor` have semantics, evaluators,
costs and scheduling coverage, but CUDA emission remains unsupported. TMA
tensor-map encoding, WGMMA and unsupported views/instructions need exact-path
checks. An instruction fact or op test does not prove whole-program coverage.

## Facts and other targets

Facts queries return declaration classes; rendering consumes declarations and
the Target. Inventory JSON uses `ops`, not the former `instructions` key.
Registered names/aliases come from op registration; the separate alias API is
removed. Adapt consumers rather than silently treating an absent key as empty.

Apple AMX uses `T.amx`, typed static op versions and a thread topology with
private logical register state. Thread count is not AMX engine count. Evaluator
and original-C comparisons do not establish native AMX codegen/performance.
The ARM-specific numerical oracle requires an ARM host; report skipped coverage
explicitly on CUDA/Linux.
