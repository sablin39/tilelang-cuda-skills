# Bottleneck Classification with torch.profiler

How to classify a TileLang kernel as IO-bound / CUDA-core bound / Tensor-core
bound / latency-bound using only `torch.profiler` plus a peek at the generated
CUDA source. This is the lightweight first-pass diagnosis; ncu is the
heavyweight confirmation.

## Table of contents

1. [The roofline-with-corroboration method](#1-the-roofline-with-corroboration-method)
2. [Step-by-step procedure](#2-step-by-step-procedure)
3. [Per-class signatures (with measured numbers)](#3-per-class-signatures-with-measured-numbers)
4. [Source-inspection checklist](#4-source-inspection-checklist)
5. [Hardware peaks worth memorizing](#5-hardware-peaks-worth-memorizing)
6. [When to escalate to ncu](#6-when-to-escalate-to-ncu)

---

## 1. The roofline-with-corroboration method

`torch.profiler` gives you kernel time. Roofline turns time into achieved
throughput. Comparing achieved throughput against the GPU's per-pipe peak
tells you which roofline you're under — i.e., which hardware resource limits
the kernel. Because the roofline is an upper bound, **closeness to a peak
implies you're bound by that resource**.

The corroboration step exists because TileLang names every kernel after its
`prim_func`. `gemm_kernel` could be tensor-core MMA, scalar FMA, or
register-spilling MMA — the name doesn't say. Inspecting
`kernel.get_kernel_source()` lets you confirm what instructions the kernel
actually issues.

## 2. Step-by-step procedure

### 2a. Measure per-call GPU time

```python
def gpu_us_per_call(prof, label):
    """Return microseconds per call for a record_function range."""
    for evt in prof.key_averages():
        if evt.key == label and evt.self_device_time_total > 0 and evt.count:
            return evt.self_device_time_total / evt.count
    return None
```

Why `self_device_time_total > 0`: `record_function` ranges appear twice in
`key_averages`, once as a CPU-side aggregator (self_device == 0) and once
attributed to the GPU side. You want the GPU one.

### 2b. Compute achieved throughput

For GEMM-shaped work:

```python
flops  = 2 * M * N * K              # one mul + one add per output element
tflops = flops / (us_per_call * 1e-6) / 1e12
```

For memory-bound work, count every distinct global-memory byte transferred
once per launch (read or write):

```python
bytes_moved = elements_read * bytes_per_in + elements_written * bytes_per_out
gb_s        = bytes_moved / (us_per_call * 1e-6) / 1e9
```

For mixed work (e.g., attention), do both. The closer one to its respective
peak is the bottleneck.

### 2c. Compare to peaks

Use `torch.cuda.get_device_properties(0)` for memory bandwidth (×1024 to
convert from `total_memory` GB sizing, but actual HBM bandwidth requires
looking up the SKU). For compute peaks, see [§5](#5-hardware-peaks-worth-memorizing).

Decision table:

| Achieved is near | Then the kernel is | What torch.profiler also tends to show |
|------------------|---------------------|----------------------------------------|
| DRAM peak | **IO-bound** | One CUDA kernel dominates; few launches; achieved GB/s ≈ peak |
| Tensor-core peak | **Tensor-core bound** | One kernel dominates; achieved TFLOPS ≈ tensor peak; generated source has `mma`/`wgmma` |
| Scalar FMA peak | **CUDA-core bound** | One kernel dominates; achieved TFLOPS ≈ scalar peak (10–30× below tensor peak); source has only scalar FMAs |
| Nothing — far below every peak, single kernel | **Latency-bound** (occupancy, warp stalls, spill) | Escalate to ncu |
| Nothing — far below every peak, many small kernels | **Launch-overhead bound** | `cuLaunchKernel` CPU total ≈ kernel CUDA total |

### 2d. Corroborate with the generated source

**Important caveat.** TileLang inlines its MMA emission into a template
header (`tl_templates/cuda/instruction/mma.h` and friends), so the raw
`mma.sync` / `wgmma` PTX tokens typically DO NOT appear in the C++ string
returned by `kernel.get_kernel_source()`. Don't conclude "no tensor cores"
just because a regex for `wgmma` came back empty. Use the TileLang-aware
indicators instead, and drop to PTX/SASS when you need to see real
instructions:

```python
src = kernel.get_kernel_source()
# TileLang-aware tensor-core indicators (look at the high-level source)
indicators = {
    "tl_mma_template":     r"tl_templates/[^\"]*mma[^\"]*\.h",  # MMA header included
    "cutensormap":         r"\bCUtensorMap\b",                   # TMA descriptors
    "mbarrier":            r"\bmbarrier\b",                       # async barriers
    "cp_async_bulk":       r"cp\.async\.bulk",                    # TMA loads
    # These rarely appear in the top-level source for TileLang Hopper/Blackwell paths:
    "raw_wgmma":           r"\bwgmma\b",
    "raw_mma_sync":        r"mma\.sync\b",
    "raw_wmma":            r"wmma::mma_sync",
}
import re
hits = {k: len(re.findall(v, src)) for k, v in indicators.items()}
print(hits)

# If you must see real instructions, drop to PTX or SASS:
# 1. Find the kernel artifact in TileLang's cache (default ~/.tilelang/cache).
# 2. cuobjdump --dump-ptx  <cubin>   # look for wgmma., mma.sync, .local
# 3. cuobjdump --dump-sass <cubin>   # look for HMMA/QMMA, STL/LDL (spill)
```

Decision table for the high-level source:

| You see... | Inference |
|------------|-----------|
| `tl_templates/.../mma.h` included AND (`CUtensorMap` OR `mbarrier`) | Tensor-core async path is wired up. Confirm at PTX/SASS only if achieved TFLOPS looks wrong. |
| `tl_templates/.../mma.h` included, no `CUtensorMap` | Older WMMA-style path, still tensor-core. |
| Neither MMA header nor TMA tokens | Scalar/FMA path — `T.gemm` either didn't lower (check shapes/dtypes) or the kernel doesn't use it. |
| MMA header included but achieved TFLOPS << peak | Likely spill / low occupancy / bank conflicts. Drop to SASS and grep `STL|LDL` for spill, or escalate to ncu. |

## 3. Per-class signatures (with measured numbers)

These signatures were measured on an NVIDIA RTX PRO 6000 Blackwell (sm_120)
under CUDA 13.1 / PyTorch 2.11 / TileLang 0.1.9. Numbers will vary by GPU but
the **shape** of the signature is portable.

### Tensor-core bound (the goal state for GEMM)

```
fp16 4096^3 GEMM, block 128x128x64 with T.gemm
us/call: 376    achieved: 366 TFLOPS   peak ~700 TFLOPS (~52% — healthy)
top CUDA kernel: gemm_kernel
generated source contains: wgmma instructions
cuLaunchKernel:  << kernel time (negligible)
```

This is what success looks like. Achieved TFLOPS is on the same order as
tensor-core peak. Room remains, but you're firmly in tensor-core territory.

### CUDA-core bound (you forgot or skipped `T.gemm`)

```
fp16 1024^3 GEMM, manual scalar FMA loop (T.Parallel, no T.gemm)
us/call: 74     achieved: 29 TFLOPS    peak FMA ~30-50 TFLOPS  (close to scalar peak)
top CUDA kernel: gemm_kernel  (same name -- doesn't help)
generated source: only scalar f32 muls/adds, no mma
cuLaunchKernel: << kernel time
```

Achieved is 10–30× below tensor-core peak — that's the giveaway. Same kernel
name, totally different instruction mix.

### IO-bound (memory-bound elementwise)

```
fp16 8192x8192 elementwise add
us/call: 261    achieved: 1541 GB/s    peak HBM ~1700 GB/s  (~90%)
top CUDA kernel: add_kernel
generated source: vector loads (uint4) + adds + vector stores, no mma
```

90% of HBM peak — saturated. There is no compute speedup possible; only
algorithmic changes (fusion to reduce trips through DRAM) help.

### Register-spill (a tensor-core kernel pretending to be tensor-core bound but isn't)

```
fp16 4096^3 GEMM, block 256x256x32, only 128 threads
us/call: 3689   achieved: 37 TFLOPS    peak ~700 TFLOPS  (5% -- terrible)
top CUDA kernel: gemm_kernel  (same name as the good case!)
generated source: still has wgmma  (T.gemm lowered fine)
```

`torch.profiler` cannot directly tell you it's spilling. Symptom is:
"my tile sizes are big, I'm using T.gemm, the source has tensor-core
instructions, but achieved TFLOPS is 5× below the well-tuned config".
That's the cue to:

1. Check the source for `.local` memory references (`STL`/`LDL` in SASS).
2. Run ncu — `launch__registers_per_thread` and
   `smsp__inst_executed_pipe_lsu_op_st.sum` will tell the story.
3. Decrease tile sizes or increase thread count.

### Launch-overhead bound

```
fp16 128^3 GEMM in a tight loop (200 reps)
us/call: 3.8    cuLaunchKernel CPU: 4.3us/launch
ratio: 53% of wallclock is launch overhead
```

The diagnostic: `cuLaunchKernel`'s `Self CPU Total` rivals `kernel`'s
`Self CUDA Total`. Per-launch CPU cost (~4–6us on most hosts) is fixed; if
your kernel is only a few microseconds, every launch is mostly host
dispatch. Fix by increasing per-launch work or using CUDA graphs.

### CPU bound (the kernel is fine, the host isn't)

You'll see a wide gap on the timeline between the end of one CUDA kernel and
the start of the next, filled with Python / aten ops on the CPU row.
`Self CPU` dominates the table. The fix is upstream of the kernel — usually
dataloading, host-side preprocessing, or a chatty Python loop.

## 4. Source-inspection checklist

When `torch.profiler` says a kernel is slow but doesn't say why, dump the
generated source and look for these signals:

```python
src = kernel.get_kernel_source()
print(f"Source length: {len(src)} chars")

import re
patterns = {
    # TileLang-aware tensor-core indicators (these reliably appear in the .cu source)
    "tl_mma_template_include": r"tl_templates/[^\"]*mma[^\"]*\.h",
    "cutensormap":             r"\bCUtensorMap\b",
    "mbarrier":                r"\bmbarrier\b",
    # Raw PTX-level tokens (often absent because MMA is in an included header)
    "raw_wgmma":               r"\bwgmma\b",
    "raw_mma_sync":            r"mma\.sync\b",
    "raw_wmma":                r"wmma::mma_sync",
    # Memory / launch / unroll signals
    "cp_async":                r"cp\.async",
    "vectorized_load_uint4":   r"\buint4\b",
    "vectorized_load_float4":  r"\bfloat4\b",
    "shared_memory_decl":      r"__shared__",
    "launch_bounds":           r"__launch_bounds__",
    "unroll":                  r"#pragma\s+unroll",
}
for name, pat in patterns.items():
    hits = len(re.findall(pat, src))
    if hits:
        print(f"  {name}: {hits}")
```

If the raw-PTX rows are zero but the TileLang-aware rows are non-zero, you
are on the tensor-core path; the actual MMA instructions are inside the
included template header. To see them you have to drop to PTX/SASS via
`cuobjdump --dump-ptx`/`--dump-sass` against the cubin in TileLang's cache
directory (default `~/.tilelang/cache`).

Decision table:

| You see... | Implication |
|------------|-------------|
| No `mma`/`wgmma`/`wmma` anywhere | CUDA-core path; if you wanted tensor cores, your `T.gemm` didn't lower — check shapes/dtypes |
| `mma`/`wgmma` present but achieved TFLOPS << peak | Likely spill, occupancy, or bank conflicts → ncu |
| Many `__shared__` declarations summing to a huge number | Possibly limiting occupancy; check `cudaDeviceProp::sharedMemPerMultiprocessor` |
| No vectorized loads on a memory-bound kernel | DRAM bound but not at peak → tighten copies (`T.copy` should be picking vectorized paths) |
| `__launch_bounds__(N, M)` with small N | Few threads per block — fine for small kernels, bad for large tiles (drives spill) |

To go below source level (SASS, occupancy), invoke `cuobjdump`:

```bash
cuobjdump --dump-sass /tmp/tilelang_cache/*.fatbin | grep -E "STL|LDL"
# STL/LDL = local-memory store/load = register spill
```

The `cuda_skill` covers the binary-inspection workflow in depth.

## 5. Hardware peaks worth memorizing

Approximate per-SKU peaks; check the spec sheet for your card. For sm_120
Blackwell (RTX PRO 6000 / B100 / B200) the relevant rooflines are:

| Class | Peak (approximate) | Use as ceiling for... |
|-------|--------------------|------------------------|
| HBM bandwidth (RTX PRO 6000) | ~1700 GB/s | bytes/sec — IO-bound kernels |
| HBM bandwidth (B200) | ~8 TB/s | same |
| FP16/BF16 tensor TFLOPS (dense, sm_120) | ~700 TFLOPS | GEMM with T.gemm |
| FP8 tensor TFLOPS | ~1400 TFLOPS | quantized GEMM |
| FP32 CUDA-core peak | ~50 TFLOPS | scalar T.Parallel math |

For other GPUs (A100, H100, etc.) the **method** is the same — only the
roofline numbers change. Query `torch.cuda.get_device_name(0)` and consult
the datasheet; for compute peaks NVIDIA publishes spec PDFs per SKU.

## 6. When to escalate to ncu

Roofline + source inspection can mis-classify. Escalate to the
`profiling-tilelang-programs` skill (ncu) when:

- Source contains tensor-core instructions but TFLOPS is < 25% of peak — likely spill, occupancy, or stall; ncu's warp-stall reasons identify which.
- A kernel is far below every roofline (latency-bound) and changing tile sizes doesn't help — ncu's `smsp__warp_issue_stalled_*` metrics point at the cause (barrier, memory, MIO, etc.).
- Suspect shared-memory bank conflicts — torch.profiler can't see them; ncu's `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm` can.
- Need to confirm "this kernel uses N registers" — torch.profiler doesn't expose that; ncu's `launch__registers_per_thread` does.

The lightweight loop is: torch.profiler tells you something is slow and
roughly what class. Source inspection narrows it. ncu confirms the
mechanism. You typically only need the last step a fraction of the time.
