---
name: debugging-tilelang-programs
description: Diagnose TileLang compilation, host argument, memory-liveness, runtime, and numerical failures. Use for failing kernels, buffer lifetime or shared-memory reuse analysis, generated-code and TVM-pass inspection, lowering traces, layout diagnostics, and minimal reproducers.
---

# Debugging TileLang Programs

Reproduce with the user's interpreter, command, input contract, and target.
Record TileLang/PyTorch/toolkit versions and the full error. Use a fixed seed
for numerical comparisons; a seed alone does not eliminate races.

## Select the failing layer

| Failure | First checks |
| --- | --- |
| Import or toolchain | Active environment, editable checkout, backend availability, writable cache |
| Parsing/lowering/codegen | DSL construct, operand shapes, target, first failing pass |
| Host argument validation | Argument count, dtype, shape, strides, device, symbolic constraints |
| Storage/liveness | Buffer access span, pipeline versions, storage scope, shared-memory reuse, physical compiler resources |
| Device crash/hang | Bounds, resource usage, synchronization, async wait/barrier protocol |
| Wrong values | Accumulator initialization, writeback, masks, indexing, accumulation precision |

## Compilation and host failures

- For transposed GEMM operands, compare the **logical** reduction dimensions:
  a transposed B tile has physical shape `(BN, BK)`.
- Inspect `kernel.get_host_source()` for host argument checks, and
  `kernel.get_kernel_source()` for generated device source after a successful
  compilation. Neither accessor can recover an object whose construction failed.
- Use `get_tir()` for specialized TIR before device compilation. Use the
  compile-only tool when inspecting source without a GPU or driver.
- Use `TL_LOWER_TRACE` for pass-by-pass IR and codegen inspection. The legacy
  Pass Diff workflow is superseded. See
  [compiler-tools.md](references/compiler-tools.md) for commands, layout plots,
  and the backend restrictions on editing generated source.
- For fragment/local, shared, or global buffer lifetimes, distinguish logical
  access intervals from physical allocation and runtime duration. Read
  [buffer-lifetime-analysis.md](references/buffer-lifetime-analysis.md) for the
  pass boundaries, current `tvm.s_tir` helpers, and the shared-memory allocator's
  liveness report.

Host stride errors can come from transposes, slices, or expanded gradients.
Either supply a contiguous tensor if that is the contract, or implement explicit
stride support. Keep shape/dtype/device validation active while debugging.

When an emitted operation or synchronization protocol is unfamiliar, use the
[KernelWiki interface](../../tilelang-optimization-loop/references/kernelwiki.md) to search its symbol or
mechanism. Compare the cited requirements with the actual lowering and target;
retain the failing artifact and the source that supports the diagnosis.

## Runtime failures

```bash
CUDA_LAUNCH_BLOCKING=1 python script.py
compute-sanitizer --tool memcheck python script.py
# Select for the suspected failure:
compute-sanitizer --tool racecheck python script.py
compute-sanitizer --tool synccheck python script.py
```

Synchronize after each suspected launch in a small reproducer before reporting
a pass; a later allocation or launch can surface an earlier asynchronous fault.
Verify that the reproducer preserves the failing call order and repetitions.
After a timeout or device fault, terminate that owned experiment and retry the reduced case in a fresh process with a finite
timeout and unbuffered logs.

Reduce to one tile, then test a tail tile. Check generated allocation and
launch requirements against the selected target. For a transfer descriptor
error, inspect the tensor's stride/alignment contract and the chosen lowering
path. Preserve the user's problem dimensions when selecting a supported path.

`T.copy` preserves synchronous semantics; explicit `T.async_copy` needs a
wait before consumption and the required cross-thread synchronization. For
manual pipelines, inspect the lowered waits and barriers.

## Numerical failures

Check `T.clear` before accumulation and final writeback. Use `T.print` on
small buffers or selected blocks to compare intermediate values. Preserve
collective participation when placing diagnostics inside thread-level control
flow. `T.device_assert` can check device-side invariants.

Match the reference's dtypes, scaling, masking, and accumulation behavior.
Use tolerances justified by the operator, and test extreme finite inputs for
exp/log/normalization kernels. Do not loosen tolerances or permit mismatched
elements solely to make a failure disappear.

`profiler.assert_consistent(repeat=10)` compares repeated outputs with
`torch.allclose`. A failure suggests nondeterminism, uninitialized state, or a
race; floating-point atomic order is another possibility. Passing it does not
prove race freedom.

## Minimize a stable failure

```bash
python -m tilelang.autodd script.py \
  --err-msg "specific failure substring" -o minimized.py --timeout 60 -j 1
```

AutoDD matches text in stdout/stderr, not necessarily a nonzero exit. Re-run
the minimized result to confirm the intended failure. Freeze essential setup
with `# autodd: freeze` or `freeze-start` / `end-freeze` annotations. Use
`--backend subproc` when persistent worker state affects the result; its
`python3` must resolve to the intended environment. Increase concurrency only
when workers will not interfere on the same GPU.

For additional examples, see [error-catalog.md](references/error-catalog.md),
[interpreting-debug-output.md](references/interpreting-debug-output.md), and
[debug-tools-doc.md](references/debug-tools-doc.md). Gradient mismatches route
to [testing-fwd-bwd-kernels](../testing-fwd-bwd-kernels/SKILL.md).

## Completion evidence

Retain a reproducer, exact failing command, relevant error/pass/trace artifact,
diagnosis, and smallest justified fix. Re-run the failing case plus cases
affected by the change. Report an unresolved failure explicitly; a tool that
did not execute or a missing device is not a passing check. For automation,
add `--error-exitcode 1` to sanitizer commands and retain their logs.
