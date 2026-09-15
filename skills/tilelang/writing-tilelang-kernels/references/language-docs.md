# Programming Models and Execution Contracts

Use this guide when choosing an invocation style or diagnosing the boundary
between Python, TIR, compilation, and a kernel launch. It follows the local
TileLang docs reviewed with the `qla` 0.1.14 checkout. Inspect installed APIs
when targeting another version.

## PrimFunc factory: specialize, compile, launch

```python
import torch
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1])
def add_factory(N, block=128):
    @T.prim_func
    def kernel(A: T.Tensor((N,), T.float32),
               B: T.Tensor((N,), T.float32),
               C: T.Tensor((N,), T.float32)):
        with T.Kernel(T.ceildiv(N, block), threads=128) as bx:
            for i in T.Parallel(block):
                index = bx * block + i
                if index < N:
                    C[index] = A[index] + B[index]
    return kernel

if __name__ == "__main__":
    a = torch.randn(273, device="cuda")
    b = torch.randn_like(a)
    tir = add_factory.get_tir(273)
    kernel = add_factory(273)
    torch.testing.assert_close(kernel(a, b), a + b)
    print(tir.script())
    print(kernel.get_kernel_source())
    print(kernel.get_host_source())
```

The outer parameters specialize the program. The inner tensor parameters are
launch arguments. `out_idx` removes returned outputs from the launch argument
list. Omit `out_idx` when the caller passes all output buffers explicitly.

## Eager JIT: tensors in, outputs returned

```python
import torch
import tilelang
import tilelang.language as T

@tilelang.jit
def add_eager(A, B, block=128):
    N = T.dynamic("N")
    A: T.Tensor((N,), T.float32)
    B: T.Tensor((N,), T.float32)
    C = T.empty((N,), T.float32)
    with T.Kernel(T.ceildiv(N, block), threads=128) as bx:
        for i in T.Parallel(block):
            index = bx * block + i
            if index < N:
                C[index] = A[index] + B[index]
    return C

if __name__ == "__main__":
    for n in (128, 273):
        a = torch.randn(n, device="cuda")
        b = torch.randn_like(a)
        torch.testing.assert_close(add_eager(a, b), a + b)
    print(add_eager.get_tir(block=128).script())
```

Do not pass `out_idx` to eager JIT. `T.empty` declares an output whose values
must be written; it does not initialize the data. The decorator translates the
function into a kernel rather than running its body as ordinary tensor Python.

## Shape, dtype, and stride contracts

- Python integer parameters and `T.const` represent specialization choices;
  `T.dynamic` / `T.dyn` represent symbolic dimensions bound from input shapes.
  `T.symbolic` is a deprecated alias of `T.dynamic`.
- A dynamic dimension does not automatically make strides dynamic. A dense
  `T.Tensor` declaration can reject a sliced/transposed input. Use an explicit
  supported buffer stride contract or materialize contiguous input.
- Output dimensions must be derivable from inputs or supplied scalar values.
  Keep shared/fragment extents suitable for compilation even when the grid or
  iteration count is symbolic.
- Strings, TileLang dtype objects, and supported framework dtype objects can
  be normalized by the DSL. Operation support still depends on dtype/backend.
- `T.copy` preserves synchronous semantics. Explicit async operations require
  their completion/visibility protocol; do not remove waits as a cosmetic edit.

For ordinary edge accesses, safe-memory lowering inserts guards when needed.
For reductions and softmax, explicitly choose the padding identity and semantic
mask. Safe addressing alone cannot make padded values mathematically correct.

## Python inside the DSL

Use `T.serial`, `T.unroll`, `T.Parallel`, `T.Pipelined`, and supported TIR
conditionals for kernel control flow. Host-side lists, configuration generation,
and framework operations belong outside the kernel body. Use `@T.macro` for
reusable inlined DSL blocks. Use `T.print` / `T.device_assert` for diagnostics.
Check compatibility before translating arbitrary Python containers or builtins.

## Targets and execution backends

`target` selects code generation; `execution_backend` selects the compilation/
launch integration. They are distinct. Use `target="cuda"` for an explicitly
CUDA task, or the configured automatic target when portability is intended.
Use target dictionaries for additional deployment attributes. Do not infer the
deployment target from the build machine when cross-compiling.

The default target can come from `TILELANG_DEFAULT_TARGET`; otherwise it is
automatic. Record an override with the experiment. Source editing during a
lowering trace needs a backend that recompiles source, such as `nvrtc` or
`cython`; a precompiled `tvm_ffi` binary is not changed by editing a source dump.

## Inspect one boundary at a time

| Artifact | Obtain it | Question |
| --- | --- | --- |
| Specialized TIR | `factory.get_tir(...)` | Did shapes, loops, and constants elaborate correctly? |
| Lowered source without launch | `tilelang.tools.compile_only` | What did lowering emit? |
| Host source | `kernel.get_host_source()` | Which argument checks and launch calls were generated? |
| Device source | `kernel.get_kernel_source()` | Which copies, loops, and helper calls survived lowering? |
| Pass history | `TL_LOWER_TRACE` | Which transformation introduced a change or failure? |

Read [compiler-tools.md](../../debugging-tilelang-programs/references/compiler-tools.md)
for commands and the distinction between source emission and runnable binaries.

Sources in a matching TileLang checkout: `docs/get_started/targets.md`,
`docs/programming_guides/{language_basics,control_flow,python_compatibility,type_system}.md`,
`docs/compiler_internals/tensor_checks.md`, and `docs/tools/compile_only.md`.
Upstream: [programming guides](https://github.com/tile-ai/tilelang/tree/main/docs/programming_guides)
and [targets](https://github.com/tile-ai/tilelang/blob/main/docs/get_started/targets.md).
