# TileLang runtime twin template

A verified pattern for making a TileLang kernel the runtime twin of an authored
TileFoundry Module, so `tilefoundry check` judges it against the HIR evaluator.
Verified against tilefoundry 0.0.2 with tilelang 0.1.14, and again against the
vendored main checkout (`25fbbaa`, 2026-10-20) with tilelang 0.1.15 — `equal`
agreement on the fp8 output and `allclose` on the f32 scale, both times.

The example twins the fused step from `tilefoundry tutorial optimize`: RMSNorm
followed by per-block FP8 quantization. The authored module (`Fused`, in the
extracted `rms_norm_quant.py`) declares one entry function

```python
rms_norm_quant(a: Tensor[(ROWS, H), "bf16"], gamma: ConstTensor[(1, H), "bf16"])
  -> (Tensor[(ROWS, H), "fp8e4m3"], Tensor[(ROWS, BLOCKS), "f32"])
```

The twin is one file beside it. The TileLang kernel is ordinary tilelang code;
the only TileFoundry surface is the two decorators:

```python
#!/usr/bin/env python3
"""Runtime twin of the fused step, as one TileLang kernel."""

import torch
import tilelang
import tilelang.language as T
from rms_norm_quant import BLOCK, BLOCKS, EPS, FP8_MAX, H, ROWS, Fused

from tilefoundry.runtime import runtime_func, runtime_module


@tilelang.jit
def _build():
    @T.prim_func
    def main(
        A: T.Tensor((ROWS, H), "bfloat16"),
        G: T.Tensor((1, H), "bfloat16"),
        Q: T.Tensor((ROWS, H), "float8_e4m3"),
        S: T.Tensor((ROWS, BLOCKS), "float32"),
    ):
        with T.Kernel(ROWS, threads=128) as bx:
            a_f = T.alloc_fragment((H,), "float32")
            g_bf = T.alloc_fragment((H,), "bfloat16")
            sq = T.alloc_fragment((H,), "float32")
            red = T.alloc_fragment((1,), "float32")
            n_b = T.alloc_fragment((BLOCKS, BLOCK), "float32")
            am = T.alloc_fragment((BLOCKS,), "float32")

            for i in T.Parallel(H):
                a_f[i] = T.cast(A[bx, i], "float32")
                g_bf[i] = G[0, i]
            for i in T.Parallel(H):
                sq[i] = a_f[i] * a_f[i]
            T.reduce_sum(sq, red, dim=0)
            inv = T.rsqrt(red[0] / H + EPS)
            for b, i in T.Parallel(BLOCKS, BLOCK):
                normed = T.cast(a_f[b * BLOCK + i] * inv, "bfloat16") * g_bf[b * BLOCK + i]
                n_b[b, i] = T.cast(normed, "float32")
            T.reduce_absmax(n_b, am, dim=1)
            for b in T.Parallel(BLOCKS):
                S[bx, b] = am[b] * (1.0 / FP8_MAX)
            for b, i in T.Parallel(BLOCKS, BLOCK):
                val = T.clamp(n_b[b, i] / (am[b] * (1.0 / FP8_MAX)), -FP8_MAX, FP8_MAX)
                Q[bx, b * BLOCK + i] = T.cast(val, "float8_e4m3")

    return main


_kernel = _build()


@runtime_module(Fused)
class FastTileLang:
    @runtime_func
    def rms_norm_quant(self, a, gamma):
        quant = torch.empty((ROWS, H), dtype=torch.float8_e4m3fn, device=a.device)
        scale = torch.empty((ROWS, BLOCKS), dtype=torch.float32, device=a.device)
        _kernel(a, gamma, quant, scale)
        return quant, scale
```

Check it exactly as any other twin (the quantized output is discrete, so
`equal` is the honest predicate; the f32 scale takes a tolerance):

```bash
tilefoundry check twin_tilelang.py:FastTileLang --inputs random --weights random \
  --out 'output[0]' --fn equal \
  --out 'output[1]' --fn allclose --atol 1e-6 --rtol 1e-6 \
  --json artifacts/runtime-check.json
```

For bf16 elementwise outputs whose only disagreement is reduction order, a
single-ulp flip at a binade bottom needs `rtol >= 2**-7` to pass
`torch.allclose`. The sharper predicate is `--fn ulp --max 1`, which admits a
one-landing-point difference without widening tolerances everywhere else.

## What to copy and what to re-derive

- **Match the authored semantics operation for operation.** This kernel passes
  `equal` on fp8 because it reproduces the authored rounding chain exactly:
  mean of squares in f32, cast to bf16 *before* multiplying by the bf16 gamma,
  cast back to f32 for the absmax scale. A "numerically equivalent" shortcut
  (keeping the normalized value in f32 through the gamma multiply) is a
  different computation and can fail exact comparison.
- **The twin method owns allocation and launch.** Outputs are fresh torch
  tensors on the input's device; the kernel handle is built once at module
  load (`@tilelang.jit` caches compilation), not per call.
- **Weights come from `check`, not from the twin.** `gamma` is a declared
  `ConstTensor`, so the run needs `--weights random` (or `ckpt:DIR`); the
  twin receives it as an ordinary argument.
- **One twin class per Module, all of its functions.** This Module has one
  function, so one `@runtime_func` suffices; a Module with more must cover
  every one.
- **Tile choice answers "is it the same computation", not "how fast".** The
  shipped TileFoundry examples twin with tiles chosen to divide dimensions
  cleanly and keep the fast engine elsewhere. Optimize only after the twin
  agrees; then keep the agreeing twin as the correctness gate for each
  optimization-loop candidate, or re-twin the optimized kernel when its
  structure moves past what the twin wraps.

The environment caveat that applies to all `check` runs: the implementation
executes on a real device (the module target's default, or `--device`), so the
SM capability the TileLang kernel targets and the device must agree, even
though `analyze` itself is static.
