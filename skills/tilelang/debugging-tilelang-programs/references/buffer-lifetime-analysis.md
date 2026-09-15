# Buffer Lifetime and Storage Analysis

Use this workflow to explain when a kernel value is needed, why temporaries
overlap, or how shared storage is reused. It derives compiler-event intervals;
it does not turn TIR statements into GPU timestamps.

## State which lifetime is being reported

| View | Meaning | Evidence |
| --- | --- | --- |
| Logical access | First and last compiler statements that may touch a buffer | TIR reads, writes, and opaque accesses at a named pass boundary |
| Storage allocation | Scope and storage that the lowered kernel reserves or reuses | Allocation nodes, `StorageRewrite`, shared-memory arena plan, generated source |
| Runtime | When GPU work actually executes | IKET, Nsight Systems, or another runtime trace |

Do not combine these axes. A global parameter normally exists before and after
the kernel, so a kernel analysis can report only its access span. A logical
fragment is distributed across threads; its TIR lifetime is not a physical
CUDA-register interval. PTXAS can coalesce values, extend physical live ranges,
or spill them to local memory.

## Choose an IR boundary

Capture the real lowering with `TL_LOWER_TRACE` and a cache miss. Interpret
buffers at a deliberate boundary:

- After pipeline injection and layout inference, inspect logical fragments,
  shared buffers, pipeline versions, layouts, and high-level access regions.
- Around `LowerTileOp`, connect tile operations to the lower-level loads,
  stores, MMA operations, waits, and barriers that implement them.
- Around `StorageRewrite`, inspect changes to local allocations and reuse. This
  is still compiler storage, not PTXAS's final register allocation.
- After `MergeSharedMemoryAllocations`, use the emitted shared arena and aliases
  as the physical shared-memory plan for that lowering.
- Use final CUDA plus PTXAS or Nsight Compute resource evidence for register
  count and spills. Do not infer those solely from fragment element counts.

Keep the target, specialization arguments, pass configuration, and selected
before/after artifacts with the result. Pipeline depth or target changes can
change both buffer versions and access order.

## Inspect the current shared-memory plan

The current TileLang allocator can emit its own liveness and packing decisions:

```python
import tilelang

pass_configs = {
    tilelang.PassConfigKey.TL_DEBUG_MERGE_SHARED_MEMORY_ALLOCATIONS: True,
}
kernel = tilelang.compile(program, target="cuda", pass_configs=pass_configs)
```

Compile on a cache miss, for example with `TILELANG_DISABLE_CACHE=1`. The report
is written to stderr and includes `GEN`/`KILL` events and, when reuse planning
runs, each buffer's `start`, `end`, byte size, required alignment, arena offset,
and the total merged size. In the current implementation `end` is the
one-past-last position in the allocator's flattened statement sequence. These
positions are internal ordering indices rather than source lines or elapsed
time.

This flag directly enables the allocator's gated `LOG(DEBUG)` records in the
reviewed version. `TVM_LOG_DEBUG` is not needed for this particular report.
If no plan appears, verify that lowering ran, more than one applicable shared
allocation survived, and the expected device function reached
`MergeSharedMemoryAllocations`.

For an A/B diagnosis, use
`PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: True` to keep dedicated regions
inside the merged allocation. Treat that as an experiment: revalidate output,
inspect the resulting shared footprint, and remeasure before attributing a bug
or speed change to reuse.

## Use the bundled TVM analyses

TileLang 0.1.14 in the reviewed checkout bundles TVM 0.25.dev0 and exposes the
relevant APIs under `tvm.s_tir` and `tvm.tirx`. Check the installed namespaces;
examples written for upstream `tvm.tir` may not match this fork.

```python
from tilelang import tvm
from tvm.s_tir import analysis as s_analysis

func = kernel_factory.get_tir(...)
allocated = s_analysis.calculate_allocated_bytes(func)
access_lca = s_analysis.detect_buffer_access_lca(func)
```

`calculate_allocated_bytes` groups explicit TIR allocations by storage scope.
It does not include caller-owned global tensors and does not compute peak
liveness. `detect_buffer_access_lca` returns the lowest statement enclosing all
known accesses for each buffer; that lexical scope is useful context but is not
a first/last-use interval.

For a selected `SBlock`,
`s_analysis.get_sblock_access_region(block, buffer_var_map)` returns read,
write, and opaque regions. Build `buffer_var_map` from buffers visible outside
that block. Opaque accesses need conservative handling because a pointer passed
to an intrinsic may hide the exact region or completion point.

For a custom interval extractor, traverse a selected PrimFunc with
`tvm.tirx.stmt_functor.pre_order_visit` or `post_order_visit`. Assign stable
event indices and record, per buffer, its name, scope, dtype, shape/bytes,
first read, first write, last touch, and access kind. Include `BufferLoad`,
`BufferStore`, declared block regions, and opaque pointer/intrinsic accesses.
Preserve symbolic sizes when the arithmetic analyzer cannot prove a constant.

## Account for asynchronous operations

A lexical last access can be too early for a pipelined kernel. Extend or mark an
interval through the operation that establishes completion:

- `cp.async` groups through the corresponding wait and required thread sync;
- TMA transfers through their barrier phase and consumer synchronization;
- asynchronous MMA/WGMMA operations through the matching wait before reuse;
- producer/consumer warp-specialized storage through both participants' use.

Prefer the shared-memory allocator's final interval and offset report when the
question is whether two shared buffers physically reuse bytes. Use the higher
level interval to explain the logical producer/consumer relationship.

## Present the result

Use one row per logical buffer, grouped by `local`/fragment, `shared`, and
global scope. Put compiler event indices on the horizontal axis and distinguish
reads, writes, waits, and barriers. Add a separate shared-arena panel with byte
offset and size; overlapping byte ranges are valid only when the allocator's
live intervals do not overlap.

Report uncertainty explicitly for opaque calls, symbolic bounds, asynchronous
completion, and fragments that have already been scalarized. Corroborate high
fragment pressure with generated CUDA and PTXAS/Nsight resource data before
calling it register pressure or a spill.
