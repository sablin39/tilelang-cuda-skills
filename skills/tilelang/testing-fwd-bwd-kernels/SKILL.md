---
name: testing-fwd-bwd-kernels
description: Implement and validate forward and backward TileLang kernels integrated with torch.autograd.Function. Use for custom differentiable operators and gradient mismatches; use the writing or debugging skills for forward-only kernels.
---

# Testing Forward and Backward Kernels

For performance work on both phases, use the
[joint optimization loop](../../tilelang-optimization-loop/references/joint-fwd-bwd.md).
Validate the forward/backward pair and saved-state contract together; select
using the task's combined objective and record per-phase tradeoffs.

## Define the contract

Identify differentiable inputs, output structure, supported shapes, strides,
dtypes, and whether higher-order gradients are required. Match the reference's
precision and mathematical operation, including scaling and masks.

Keep the wrapper proportional to the operator. Matmul can use two backward
GEMMs; attention may need preprocess, accumulation, and postprocess kernels.
Read [gradient-testing-patterns.md](references/gradient-testing-patterns.md) for
a complete matmul example and reusable comparison helper. Read
[fwd-bwd-examples.md](references/fwd-bwd-examples.md) only for attention-specific
intermediates, ownership of gradient tiles, and atomic accumulation.

## Integrate with autograd

- Save tensors needed by backward with `ctx.save_for_backward`; save scalar
  configuration on `ctx`. Return one gradient per forward input, using `None`
  for nondifferentiable inputs.
- Respect the kernel's stride contract. Upstream gradients may be expanded or
  non-contiguous even when forward inputs were contiguous. Materialize a
  contiguous gradient if the compiled kernel requires it.
- Allocate atomic accumulation buffers as zeros **for every invocation**.
  Preallocated mutable outputs usually omit `out_idx`; returned outputs use it.
- Use float32 accumulation when the error budget requires it, then convert
  returned gradients to the intended dtype. Half-precision atomics exist on
  supported targets; their precision and availability are separate questions.
- A raw TileLang launch does not build a differentiable backward graph.
  Implement higher-order derivatives explicitly if required; otherwise document
  the limitation and consider `once_differentiable` on backward.

## Validate independently

1. Run custom and reference paths on detached input clones with matching
   `requires_grad` flags. Compare forward outputs before gradients.
2. Use the same random upstream gradient for both paths. `torch.autograd.grad`
   avoids accumulating into `.grad`; otherwise clear `.grad` between runs.
   Independent forward graphs do not require `retain_graph=True`.
3. Compare each differentiable input's gradient, including missing gradients.
   Report maximum absolute error and tolerances derived from dtype, reduction
   length, and the operator's accuracy requirements.
4. Cover rectangular and tail shapes, supported dtypes, masks/causal modes, and
   supported strides. Include repeated calls to catch stale accumulation state.
   Avoid exercising unsupported layouts as though the kernel accepts them.

For fp16/bf16 tensor-core kernels, reference-based gradient comparisons are the
main check. `gradcheck` is useful for small kernels with a numerically suitable
precision path, normally float64. Quantization and overly loose tolerances can
invalidate a finite-difference check; they do not make the tool universally
ineffective. Use `gradgradcheck` only when higher-order gradients are supported.

## Localize a mismatch

For `C = A @ B`, check `dA = dC @ B.T` and `dB = A.T @ dC` independently.
The shared tile shapes must account for each GEMM transpose flag.

For attention, validate forward output, the exact log-sum-exp convention, and
`Delta = rowsum(O * dO)` before dQ/dK/dV. A cosine similarity near one with the
wrong magnitude suggests a scale error; it does not prove correctness.
Floating-point atomics can introduce nondeterminism without a data race.

Use [debugging](../debugging-tilelang-programs/SKILL.md) for launch, shape, or
memory failures. Once gradients pass, use
[profiling](../profiling-tilelang-programs/SKILL.md) on the complete backward
path, including zeroing and conversion costs when they occur in production.

## Completion evidence

Return the wrapper/kernel paths, supported forward/backward contract, exact
comparison command, checked cases, and per-output/gradient results. Include
repeated-call and upstream-stride checks when those behaviors are supported.
State higher-order derivative support explicitly. A matching forward result
does not establish backward correctness.
