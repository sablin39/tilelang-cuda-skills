# Pipeline Scheduling and Layout Inspection

Use these tools to explain or change how a drafted algorithm is lowered.
Begin with inferred scheduling; introduce explicit annotations for a concrete
dependency/order problem and validate the resulting program.

## Inferred pipeline

```python
for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
    T.copy(A[by * BM, ko * BK], A_shared)
    T.copy(B[ko * BK, bx * BN], B_shared)
    T.gemm(A_shared, B_shared, C_local)
```

Changing pipeline depth can change temporary storage and scheduling. Compare
lowered IR/source and measured latency at the same shape after validating
outputs. For diagnosis, a non-pipelined variant can isolate scheduling issues.

## Explicit stage/order arrays

`stage` and `order` describe executable statements: copies, fills, GEMMs,
reductions, stores, atomics, and explicit synchronization operations.
Replayable scalar aliases do not consume annotation slots:

```python
for ko in T.Pipelined(T.ceildiv(K, BK),
                     stage=[0, 0, 1], order=[0, 1, 2]):
    base: T.int32 = ko * BK
    T.copy(A[by * BM, base], A_shared)
    T.copy(B[base, bx * BN], B_shared)
    T.gemm(A_shared, B_shared, C_local)
```

There are three scheduled operations, not four. The compiler replays `base`
at the logical iteration of each consumer. A scalar binding that reads a
buffer written inside the pipeline is different: it is scheduled and needs
an entry, in the same stage as consumers of that value. Materialize storage
when a value must be computed once and shared across later stages.

Use unique order values and keep producers before consumers according to the
dependency rules. Legacy arrays that count replayable binds can be accepted
for compatibility; omit those slots in new code. Inspect `TL_LOWER_TRACE` to
understand which statements the pass actually schedules.

## Automatic warp specialization (v0.1.15)

An opt-in role-based scheduler partitions producer/consumer warps
automatically instead of relying on the inferred pipeline alone:

```python
@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_ENABLE_AUTO_WARP_SPECIALIZATION: "role_based",
})
```

It targets persistent-style kernels (the pass was formerly `AutoSchedule`;
worked examples live in `examples/aws/gemm.py` and
`examples/aws/flash_attention.py` in a matching checkout). Treat it as a
candidate configuration, not a default: validate outputs and measure against
the inferred pipeline at the same shape before promoting it.

## Copy and layout contracts

`T.copy` has synchronous semantics; its lowering can choose different transfer
mechanisms while preserving availability of the destination. `T.async_copy`
is an explicit async operation and requires completion before consumption plus
cross-thread visibility where appropriate. Prefer the inferred pipeline when
manual synchronization is not the task.

`T.Parallel(..., loop_layout=fragment)` and `T.annotate_layout` can express
specific thread/data mappings. Use layout visualization to understand the
inferred mapping before overriding it. A valid-looking plot does not prove
correct indexing or better performance. Re-run tails and supported stride
cases after changing a mapping.

See [compiler-tools.md](../../debugging-tilelang-programs/references/compiler-tools.md)
for lowering traces and layout visualization. Sources:
`docs/programming_guides/software_pipeline.md`, `control_flow.md`, and
`instructions.md` in the matching TileLang checkout.
Upstream: [software pipeline annotations](https://github.com/tile-ai/tilelang/blob/main/docs/programming_guides/software_pipeline.md).
