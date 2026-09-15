---
name: writing-tilelang-kernels
description: Draft TileLang operators using JIT, tensor contracts, tiled computation, and reusable kernel patterns. Use for new kernels, ports, dynamic shapes, and choosing between eager and PrimFunc-factory programming styles.
---

# Writing TileLang Kernels

Identify the input/output contract: mathematical operation, shapes, dtypes,
strides, device, mutation, and accuracy. Use the user's active environment and
inspect the installed TileLang version before selecting newer APIs.

For a supplied PyTorch baseline, first use
[baseline inspection](../torch-profiling-tilelang-programs/references/torch-compile-inspection.md)
to connect its source, compiled lowering, and profile to the region being
implemented. Reuse its reference and measurements when drafting.

## Choose the programming model

| Style | Define | Invoke |
| --- | --- | --- |
| PrimFunc factory (lazy) | Outer Python function returns `@T.prim_func`; `@tilelang.jit` compiles it | `kernel = factory(shape, config)` then `out = kernel(inputs)` |
| Eager JIT | `@tilelang.jit` function annotates inputs, uses `T.empty`, and returns output buffers | `out = operator(inputs, config)` |

In the factory style, `out_idx=[-1]` allocates and returns the last buffer;
omit that output argument at launch. Without `out_idx`, pass every buffer,
including preallocated mutable outputs. In eager style, declare outputs with
`T.empty` and return them; do not supply `out_idx`.

Read [language-docs.md](references/language-docs.md) for runnable versions of
both models, symbolic dimensions, explicit targets, and inspection APIs. Keep
the existing style when extending a working kernel unless changing it helps
the task.

## Draft the computation

1. Reuse the supplied PyTorch reference and its contract; create a reference
   only when none is available.
2. Use the baseline findings to select the closest pattern from
   [kernel-templates.md](references/kernel-templates.md): vector scale, stable
   softplus, row reduction, GEMM, fused GEMM, dynamic GEMM, or transposed GEMM.
3. Define the grid with `T.Kernel`, tile bounds, and memory scopes. Fragments
   distribute logical tile elements across threads; local buffers are private
   to each thread. Shared buffers support cooperation/reuse when needed.
4. Use `T.copy`, `T.Parallel`, and tile reductions/GEMM for the main algorithm.
   Clear accumulators before accumulation and write every required output.
5. Validate a small case, representative sizes, tails, and supported strides.
   For exp/log and normalization, include extreme finite inputs and masks.

Automatic safe-memory lowering handles many ordinary out-of-bounds accesses.
Algorithmic masks still matter: padding with zero is unsuitable for max or
softmax when excluded elements must contribute negative infinity. Test the
actual reduction identity and tail behavior.

`T.copy` has synchronous semantics even when lowered using asynchronous
instructions. Prefer inferred `T.Pipelined` scheduling for copy/compute loops.
Explicit async operations require their documented completion and visibility
protocol; see the optimization skill's
[pipeline guide](../optimizing-tilelang-programs/references/pipeline-and-layouts.md).

## Inspect before adding complexity

For an unfamiliar implementation pattern, use the supplementary
[KernelWiki interface](../../tilelang-optimization-loop/references/kernelwiki.md). Query the operation or
emitted mechanism, check target/version requirements, and connect the proposed
TileLang design to its actual lowering before claiming a hardware benefit.

- `factory.get_tir(...)`: inspect specialized TIR without device compilation.
- `kernel.get_kernel_source()`: inspect generated device code.
- `kernel.get_host_source()`: inspect generated argument checks and launch code.
- Compile-only and lowering traces: use the debugging skill's
  [compiler tools](../debugging-tilelang-programs/references/compiler-tools.md).

When the implementation needs a human-facing proposal or implementation
explanation, use the
[design-document skill](../writing-tilelang-design-docs/SKILL.md). Tie its
diagrams to the specialized TIR or generated artifacts and label conceptual
views separately from compiler-derived or measured evidence.

See [api-quick-reference.md](references/api-quick-reference.md) for core DSL
operations and [tilelang-examples.md](references/tilelang-examples.md) for
additional algorithm patterns. Load only the relevant example.

Use [debugging](../debugging-tilelang-programs/SKILL.md) for failures,
[profiling](../profiling-tilelang-programs/SKILL.md) to establish a baseline,
and [optimization](../optimizing-tilelang-programs/SKILL.md) for measured
improvements. Differentiable operators also need
[forward/backward validation](../testing-fwd-bwd-kernels/SKILL.md).

## Completion evidence

Return the kernel and reference paths, exact validation command, supported
input/output contract, and cases actually checked. Distinguish emitted source,
successful compilation, and validated execution. Carry this evidence into
profiling; use the suite's [workflow](../../tilelang-optimization-loop/SKILL.md) when
continuing through optimization or handing off the implementation.
