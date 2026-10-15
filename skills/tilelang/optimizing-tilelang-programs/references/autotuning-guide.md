# Autotuning a Validated Operator

Expose configuration choices as factory arguments, validate candidates against
a reference returning tensors, and benchmark a bounded search on one selected
device. This guide follows the current checkout's programming guides and
checks their call contracts against the implementation.

## Decorator workflow

This complete example tunes two block sizes. It uses fixed input data and a
tail size so correctness includes boundary handling.

```python
import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs

def reference(A, B):
    return A + B

@tilelang.autotune(
    configs=[{"block": 128}, {"block": 256}],
    ref_prog=reference, warmup=3, rep=10,
    rtol=1e-5, atol=1e-6, max_mismatched_ratio=0.0,
)
@tilelang.jit(out_idx=[-1])
def add(N, block=128):
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
    with set_autotune_inputs(a, b):
        best = add(a.numel())
    torch.testing.assert_close(best(a, b), reference(a, b))
    print("autotuned add: PASS")
```

The decorator is stacked above `@tilelang.jit`. Config keys must match tunable
arguments. Callable config spaces should return a concrete list for the chosen
problem. With `out_idx=[-1]`, both the supplied input list and the best-kernel
launch omit C. Do not copy examples that pass all buffers to an output-returning
kernel. `ref_prog` returns expected tensors; a boolean predicate is not the
reference output contract.

## Programmatic workflow

Use a **plain factory** returning a PrimFunc, with fixed problem dimensions
captured in its closure. Do not pass a JITImpl or already-compiled JITKernel
into `AutoTuner.from_kernel`.

```python
import torch
import tilelang.language as T
from tilelang.autotuner import AutoTuner

def make_tuner(N):
    def factory(block=128):
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

    return (AutoTuner.from_kernel(factory, [{"block": 128}, {"block": 256}])
            .set_compile_args(out_idx=[-1], target="cuda")
            .set_profile_args(ref_prog=lambda a, b: a + b,
                              rtol=1e-5, atol=1e-6, max_mismatched_ratio=0.0))

if __name__ == "__main__":
    result = make_tuner(273).run(warmup=3, rep=10)
    a = torch.randn(273, device="cuda")
    b = torch.randn_like(a)
    torch.testing.assert_close(result.kernel(a, b), a + b)
    print(result.config, result.latency)
```

Separate compilation options (`out_idx`, target/backend, pass configs) from
profiling options (reference, input supply, tolerances, benchmark backend).
Keep settings in the appropriate call; the installed `.run` can override
warmup/rep passed earlier, so set measurement budgets at that call explicitly.

## Reproducibility and resource controls

- Prefer representative fixed inputs with `set_autotune_inputs`. Dynamic-shape
  factories need concrete inputs or a custom supplier; generated tensor supply
  cannot guess every symbolic size.
- Reset mutated inputs or accumulated outputs between trials. Capturing tensors
  alone does not guarantee the correct initial state for an in-place kernel.
- Record chosen configuration, shape/dtype, input distribution, target/backend,
  and benchmark method. Retest the winner outside the tuning loop.
- Bound compilation workers with `TILELANG_AUTO_TUNING_CPU_COUNTS` or
  `TILELANG_AUTO_TUNING_MAX_CPU_COUNT`; select the GPU through the environment.
- Per-candidate timeout is a wait limit: it cannot forcibly stop a hung native
  or CUDA call. Diagnose hangs with isolated reproducers before a broad search.

## Cache and manual searches

`TILELANG_CACHE_DIR` controls artifact placement. Use a writable experiment
directory when needed. `TILELANG_DISABLE_CACHE=1` disables kernel caching;
`TILELANG_AUTO_TUNING_DISABLE_CACHE=1` disables autotune disk caching. A fresh
process may still be needed to avoid in-process reuse during inspection.

Since v0.1.15, cache artifact hash verification is always on
(`TILELANG_CACHE_VERIFY_HASH` was removed), cached kernel parameters are stored
as JSON rather than cloudpickle, and cache directories are published atomically;
legacy caches rebuild automatically on first miss.

For manual compilation sweeps, `JITImpl.par_compile` accepts dictionaries or
tuples of factory arguments. Supply **all** required fixed dimensions along
with the varying parameters, then benchmark each compiled result with the
same harness. Do not mistake compilation throughput for kernel performance.

Source: [autotuning programming guide](https://github.com/tile-ai/tilelang/blob/main/docs/programming_guides/autotuning.md).
The local docs contain older examples with mismatched output/reference or
factory contracts; the runnable examples here use the inspected `qla` APIs.
