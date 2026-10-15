# Core Programming and Inspection APIs

Use this lookup after choosing an invocation model in
[language-docs.md](language-docs.md). Check installed signatures when changing
versions.

## Definition and launch

| API | Contract |
| --- | --- |
| `@tilelang.jit(out_idx=[-1])` | Compile a PrimFunc factory and return its last buffer as an allocated output |
| `@T.prim_func` | Define the computation returned by a factory |
| `tilelang.compile(func, target=..., execution_backend=...)` | Explicitly compile a PrimFunc |
| `T.Tensor(shape, dtype)` | Declare a buffer's shape and dtype; dense layout is the default |
| `T.StridedTensor(shape, strides, dtype)` | Declare explicit strides |
| `T.empty(shape, dtype)` | Declare a returned eager-JIT output; eager JIT does not use `out_idx` |
| `T.Kernel(grid_x, grid_y, threads=N)` | Define launch dimensions and bind block coordinates |
| `T.dynamic("M")`, `T.dyn["M"]` | Declare a symbolic dimension resolved from runtime arguments |
| `T.const("M")` | Bind an eager-JIT compile-time constant, such as a specialized input dimension |

Dynamic dimensions do not imply arbitrary stride support. Target selection
controls code generation; the execution backend controls compilation/launch
integration.

## Storage and operations

| API | Meaning |
| --- | --- |
| `T.alloc_shared(shape, dtype)` | Block-shared storage for cooperation and reuse |
| `T.alloc_fragment(shape, dtype)` | Logical tile distributed across threads by a fragment layout |
| `T.alloc_local(shape, dtype)` | Thread-private storage |
| `T.alloc_var(dtype)` | Mutable scalar storage |
| `T.copy(src, dst)` | Tile copy with synchronous semantics; inspect inferred extents and tails |
| `T.clear(buf)`, `T.fill(buf, value)` | Initialize a buffer; clear accumulators before accumulation |
| `T.gemm(A, B, C, transpose_A=..., transpose_B=...)` | Accumulate a tile product into C; check reduction dimensions after transposes |
| `T.gemm_blockscaled(A, B, C, SFA, SFB, ...)` | Block-scaled tile product with per-block scale factors (MXFP8/FP4-class dtypes, v0.1.15); requires `k_start` and `sf_a_granularity_k` |
| `T.reduce_sum(src, dst, dim=...)` | Reduce along an axis; `reduce_max` and `reduce_min` use different identities |
| `T.Parallel(dim0, dim1, ...)` | Distribute elementwise iterations across threads |
| `T.serial(start, stop)`, `T.unroll(start, stop)` | Sequential or explicitly unrolled iteration |
| `T.Pipelined(iters, num_stages=N)` | Request software pipeline scheduling; verify the resulting dependencies |
| `T.atomic_add(dst, value)` | Accumulate atomically into initialized storage; floating-point order can affect results |
| `T.cast(value, dtype)` | Explicit conversion; choose accumulator precision separately from input/output precision |
| `T.if_then_else(cond, x, y)` | Conditional value selection; masks still need the correct reduction identity |
| `T.ceildiv(a, b)` | Ceiling division for launch and tile counts |

Common math: `T.exp`, `T.exp2`, `T.log`, `T.log1p`, `T.log2`, `T.rsqrt`,
`T.abs`, `T.max`, `T.min`. Stable softplus is
`T.max(x, 0) + T.log1p(T.exp(-T.abs(x)))`.

Use inferred layouts and pipelines first. Explicit `T.async_copy`,
`T.annotate_layout`, and pipeline stage/order arrays require the contracts in
[pipeline-and-layouts.md](../../optimizing-tilelang-programs/references/pipeline-and-layouts.md).

## Inspection and validation

| API | Evidence it provides |
| --- | --- |
| `factory.get_tir(...)` | Specialized TIR before device compilation |
| `kernel.get_host_source()` | Generated host checks and launch integration |
| `kernel.get_kernel_source()` | Generated device source |
| `T.print(...)` | Device values; behavior depends on the object, so bound the diagnostic scope |
| `T.device_assert(condition, message)` | A device-side invariant check |
| `kernel.get_profiler()` | Input generation, reference comparison, and benchmark helpers |
| `profiler.assert_allclose(ref_prog, rtol=..., atol=...)` | Agreement for the supplied/generated cases and tolerances |
| `profiler.assert_consistent(repeat=N)` | Repeated-output agreement; it cannot prove race freedom |
| `profiler.do_bench(...)` | Latency in milliseconds under the selected measurement policy |

Choose generated inputs appropriate to the domain, or pass explicit tensors.
Integer-valued floating-point inputs can simplify debugging, but do not ensure
exact arithmetic for all reductions or magnitudes. For timing semantics, use
[profiling](../../profiling-tilelang-programs/SKILL.md); for pass traces and
compile-only emission, use [compiler tools](../../debugging-tilelang-programs/references/compiler-tools.md).
