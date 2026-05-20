---
name: torch-profiling-tilelang-programs
description: >
  Use torch.profiler as a lightweight, always-available alternative to
  Nsight Compute/Systems for profiling, debugging, and optimizing TileLang
  GPU kernels. Trigger this skill whenever the user wants to profile a
  TileLang kernel without installing ncu/nsys, get an operator/kernel
  breakdown of a forward pass that includes TileLang custom ops, hunt
  launch-overhead in a multi-kernel pipeline, classify whether a TileLang
  kernel is IO-bound / CUDA-core bound / Tensor-core bound, generate a
  Chrome trace, profile GPU memory allocations, or compare a TileLang
  kernel against a torch/cuBLAS reference inside one trace. Also trigger
  on phrases like "profile this kernel", "why is this slow", "how do I see
  the kernel timeline", "lightweight profiling", "torch profiler",
  "chrome trace", or "perfetto", even if the user does not mention
  TileLang explicitly but TileLang code is on the table.
---

# Profiling TileLang Programs with `torch.profiler`

## When to reach for this skill

`torch.profiler` (the Kineto-based one in `torch.profiler`, **not** the legacy
`torch.autograd.profiler`) is the right tool when:

- ncu / nsys aren't installed (cluster without admin, containerized env).
- You want the full forward (or fwd+bwd) view, not a single kernel.
- You need a per-op breakdown that includes PyTorch ops and TileLang custom kernels in the same timeline.
- You're hunting CPU↔GPU overlap, kernel launch overhead, or which step in a model is slowest.

When this skill is **not** enough:

- Register-spill, bank-conflict, occupancy, warp-stall analysis → use the
  `profiling-tilelang-programs` skill (ncu) instead. `torch.profiler` shows
  *what* is slow; ncu shows *why*.
- Single-kernel deep-dive at the SM/DRAM level → ncu.

This skill is the lightweight first pass; escalate to ncu when symptoms point
to a hardware-level issue.

## Honest scope: what torch.profiler can and cannot tell you

`torch.profiler` reports **time** (per op, per kernel, per launch) and **memory
allocations**. From that alone you can directly identify:

- **Memory-bound kernels** — achieved GB/s near peak HBM bandwidth.
- **Launch-overhead-bound code** — `cuLaunchKernel` CPU time approaching the kernel's GPU time.
- **CPU-bound code paths** — long CPU intervals with little or no concurrent GPU work.
- **Which kernel dominates** the timeline.

But for TileLang specifically, **the profiler cannot, on its own**, distinguish:

- Tensor-core vs CUDA-core use — TileLang names every kernel after its `prim_func`, e.g. `gemm_kernel`. You won't see `hmma`, `wgmma`, `cutlass_tensorop` in the name. Inspect the generated CUDA source instead (`kernel.get_kernel_source()`).
- Register spill, shared-memory bank conflicts, low occupancy — these show up only as "this kernel runs slowly compared to its theoretical peak". You need ncu to confirm the mechanism.

So the workflow is: **measure with torch.profiler → roofline-classify → if the kernel underperforms, escalate to source inspection or ncu**.

## Quick start: profile a single TileLang kernel

A reusable template is bundled at `scripts/profile_template.py`. Copy it next
to your kernel script and adapt the marked section. The essentials are:

```python
import torch
from torch.profiler import profile, ProfilerActivity, record_function

# Build + warm up the kernel so JIT compile cost doesn't leak into the trace.
kernel = my_tilelang_kernel(...)
for _ in range(10):
    kernel(a, b)
torch.cuda.synchronize()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    for _ in range(20):
        with record_function("tilelang_kernel"):
            kernel(a, b)
        with record_function("torch_ref"):
            a @ b
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
prof.export_chrome_trace("trace.json")
```

Why each part matters:

- `warmup loop + cuda.synchronize()` — TileLang JIT-compiles on first call
  (you'll see `TileLang begins to compile kernel ...` in the log). Without
  warmup that compile time pollutes your first iteration. Aim for **at least
  10 warmup calls** for autotuned kernels; the compile log line stops printing
  once it's done.
- `record_function("...")` — names a region so it appears as a single row in
  the table and as a labeled band in the Chrome trace. Wrap each thing you
  want to compare. Without it you only see the underlying CUDA kernel name.
- `ProfilerActivity.CUDA` — required to capture GPU events. Without it you
  only get CPU-side launches.
- `record_shapes=True` — keeps the input shapes so you can group by
  `key_averages(group_by_input_shape=True)`. Costs a little overhead; turn
  off in production-style benchmarking.
- The `synchronize()` after the loop ensures the last kernel's events make
  it into the trace before the context exits.

## Scheduled profiling for longer runs

For training loops or multi-step models, replace the simple `with profile(...)` block with the scheduled form so you sample a stable window rather than including the cold first step:

```python
from torch.profiler import schedule, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(skip_first=10, wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=tensorboard_trace_handler("./tb_log"),
) as prof:
    for step in range(N):
        train_step(...)
        prof.step()
```

Cycle: skip 10 → wait 1 → warmup 1 → record 3 → fire `on_trace_ready` → done. View later with `tensorboard --logdir tb_log` (requires `pip install torch_tb_profiler`). The scheduled API is preferred for long-running jobs because it bounds profiler overhead.

## Reading the `key_averages` table

The table has many columns; for TileLang triage the load-bearing ones are:

| Column | What it means |
|--------|---------------|
| `Self CUDA` | GPU time spent in this op *excluding* nested kernels — this is the one that ranks "real" cost |
| `CUDA total` | GPU time including nested calls (use for `record_function` ranges) |
| `Self CPU` | CPU time spent inside this op |
| `# of Calls` | how many times the event fired |

Sort by `self_cuda_time_total` to find the biggest GPU cost; sort by
`self_cpu_time_total` to find CPU bottlenecks (often dataloader, host-side
preprocessing, or chatty Python).

A TileLang kernel always shows up twice in the table — once as your
`record_function` name (the CPU-side range) and once as the underlying CUDA
kernel name (whatever the `prim_func` is, usually `gemm_kernel`, `add_kernel`,
etc.). They report the same `Self CUDA` value. Use the kernel name when you
need to identify the exact GPU work, and the `record_function` name when you
want grouping that you control.

`cuLaunchKernel` / `cudaLaunchKernel` rows show **per-launch CPU dispatch
cost**. Their total CPU time vs the total GPU time is the launch-overhead
diagnostic (see below).

## Identifying the bottleneck: IO / CUDA core / Tensor core

There is no single metric in torch.profiler that says "tensor-core bound". You
classify by **roofline analysis** on the numbers you do measure, then confirm
with a glance at the generated CUDA source. See
`references/bottleneck-classification.md` for the full decision tree and
worked examples; the short version is:

### Step 1 — compute achieved throughput

From the per-call GPU time (`evt.self_device_time_total / evt.count`, in
microseconds), derive:

```python
tflops    = (2 * M * N * K) / (us_per_call * 1e-6) / 1e12     # for GEMM-like
gb_per_s  = total_bytes      / (us_per_call * 1e-6) / 1e9     # for memory-bound
```

For GEMM-like kernels with non-trivial epilogues, also add the epilogue work
to the numerator (usually negligible). For memory-bound kernels,
`total_bytes` should count every distinct global-memory byte read or written
exactly once.

### Step 2 — compare against hardware peak

Look up the device's peak using `torch.cuda.get_device_properties(0)` and the
GPU's datasheet. Roughly:

| Class | If achieved is near peak of... | Likely bound |
|-------|--------------------------------|--------------|
| HBM bandwidth | DRAM peak (e.g., ~1700 GB/s on RTX PRO 6000) | **IO-bound** (memory) |
| Tensor-core peak (fp16/bf16/fp8) | Tensor TFLOPS peak | **Tensor-core bound** |
| FP32 FMA peak (CUDA cores) | Scalar TFLOPS peak | **CUDA-core bound** |
| None of the above | far below every peak | **Latency-bound** (launch overhead, low occupancy, stalls — escalate to ncu) |

### Step 3 — confirm with the generated CUDA source

`kernel.get_kernel_source()` returns the generated CUDA. Grep for:

- **Tensor cores in use** — TileLang wraps MMA in a template header, so the raw `mma.sync` / `wgmma` tokens often DO NOT appear in the top-level source even when the kernel is tensor-core based. The reliable indicators in TileLang-emitted CUDA are:
  - `#include "tl_templates/cuda/instruction/mma.h"` (or similar `tl_templates/.../mma*.h`) — TileLang's MMA template header is pulled in
  - `CUtensorMap` declarations — TMA descriptors, the Hopper/Blackwell async tensor-core data path
  - `mbarrier` / `cp.async.bulk.tensor` — the producer/consumer barriers and TMA loads that flank a wgmma pipeline
  - `__hmma_*`, `wmma::mma_sync` — older WMMA path (rare in newly generated TileLang)
  
  If the C++ source doesn't show raw MMA tokens, dump the PTX (`cuobjdump --dump-ptx ...`) or SASS to confirm at the instruction level. The TileLang cache typically lives under `~/.tilelang/cache/` — each compiled kernel has its `.cu`, `.ptx`, and `.fatbin` artifacts there.
- **CUDA cores only** — no `tl_templates/.../mma.h` include, no `CUtensorMap`, just plain scalar `+`/`*`/`fma` on fp/int. This is what `T.Parallel` elementwise math compiles to.
- **Possible register spill** — search for `.local` references in the PTX (`cuobjdump --dump-ptx`) or `STL`/`LDL` in SASS (`cuobjdump --dump-sass`). Symptoms in torch.profiler: a kernel that gets nowhere near its theoretical peak despite a generous tile size. The top-level `.cu` source from `kernel.get_kernel_source()` does NOT show this directly — you must drop to PTX or SASS.

### Worked example (numbers from the experiment in `references/bottleneck-classification.md`)

On an RTX PRO 6000 Blackwell, fp16 4096³ GEMM:

| Kernel | us/call | Achieved | Classification |
|--------|---------|----------|----------------|
| Good GEMM (block 128×128×64, T.gemm) | 376 | **366 TFLOPS** | Tensor-core bound (~half of fp16 peak, headroom remains) |
| Spill GEMM (oversized 256×256×32 tile) | 3689 | 37 TFLOPS | Same kernel name, same tile API — but 10× slower; **escalate to ncu / source inspection** to confirm spill |
| FMA-only GEMM (no `T.gemm`, scalar loops, 1024³) | 73 | 29 TFLOPS | **CUDA-core bound** — well below tensor-core peak, achieved peak matches scalar FMA |
| Memory-bound add (8192² fp16) | 261 | **1541 GB/s** | **IO-bound** — ~90% of HBM peak |
| Tiny GEMM (128³, 200 launches) | 3.8 | n/a | **Launch-overhead-bound** — cuLaunchKernel CPU total ≈ kernel CPU total |

The first three all show as `gemm_kernel` in the trace. **You cannot
distinguish them from the kernel name alone** — only from achieved TFLOPS
relative to peak plus a peek at the generated source.

## Diagnosing launch overhead

If `cuLaunchKernel`'s total CPU time approaches the kernel's total GPU time,
the workload is launch-bound: the CPU side is the bottleneck, not the GPU.

```
cuLaunchKernel  Self CPU:  862us across 200 calls   (~4.3us/launch)
gemm_kernel     Self CUDA: 760us across 200 calls   (~3.8us/kernel)
```

When per-launch CPU cost rivals per-kernel GPU cost, every option below is
worth trying:

- Increase the per-launch work (larger batch / tile / fused kernel) so the
  GPU side dominates again.
- Use CUDA graphs (or `do_bench(backend="cudagraph")` for measurement). The
  TileLang `profiling-tilelang-programs` skill covers the cudagraph backend.
- Fuse adjacent kernels (epilogue fusion, kernel fusion across small ops).

## Memory profiling

For OOM hunts and allocation-pattern questions, enable `profile_memory` and
export a timeline. The current path is via the (still working but deprecated)
`export_memory_timeline`:

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True,
) as prof:
    forward_pass()

prof.export_memory_timeline("memory.html", device="cuda:0")
```

The HTML embeds a stacked-area plot of allocations by category over time.
Use `.json.gz` for raw data, `.raw.json.gz` for per-event allocator actions.

The PyTorch docs recommend the newer `torch.cuda.memory._record_memory_history`
/ `_export_memory_snapshot` pair for deeper allocator analysis:

```python
torch.cuda.memory._record_memory_history(max_entries=100_000)
forward_pass()
torch.cuda.memory._export_memory_snapshot("snapshot.pickle")
torch.cuda.memory._record_memory_history(enabled=None)
```

Open `snapshot.pickle` at https://pytorch.org/memory_viz to see allocations,
frees, and stack traces. This is the right tool for "where did my N GB go".

## Comparing TileLang vs a reference in one trace

Put both inside the same profile, each in its own `record_function`:

```python
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(20):
        with record_function("tilelang_matmul"):
            kernel(a, b)
        with record_function("torch_matmul"):
            a @ b
    torch.cuda.synchronize()
```

The Chrome trace then shows both kernel bands side by side, and the
`key_averages` table reports per-call GPU time for each label, ready for a
TFLOPS-vs-TFLOPS comparison. This is the lightweight equivalent of running
two `do_bench` calls back-to-back, but with the bonus of seeing exactly which
underlying CUDA kernels each path produces (helpful when you suspect torch is
falling back to a non-tensor-core path).

## Visualizing the trace

`prof.export_chrome_trace("trace.json")` writes a Chrome-format trace.
Two ways to view it:

- **chrome://tracing** — open Chrome, paste `chrome://tracing` in the URL bar, click Load, pick the file.
- **Perfetto** — https://ui.perfetto.dev/ → "Open trace file". Perfetto handles large traces better and has a nicer query UI.

What to look for in the timeline:

- **Gaps between kernels** with no concurrent CPU work → launch overhead or sync.
- **CPU work between kernels** but no GPU activity → CPU-bound preprocessing.
- **Overlapping kernels on different streams** → good multi-stream pipelining.
- **A wide gap then a burst** → likely a `cudaMalloc` or first-touch JIT compile slipping past your warmup; widen the warmup loop or freeze allocator behavior with `torch.cuda.memory.set_per_process_memory_fraction`.

## Common pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| No warmup before the profile block | First iteration is 10× slower; JIT log lines inside the trace | Run ≥10 warmup calls + `torch.cuda.synchronize()` before entering the `with profile(...)` |
| Forgot `ProfilerActivity.CUDA` | `Self CUDA` column is all zeros | Add `ProfilerActivity.CUDA` to `activities` |
| Forgot `cuda.synchronize()` inside the block | Last few kernels missing from the trace | Call `torch.cuda.synchronize()` right before exiting `with profile(...)` |
| `record_shapes=True` left on for benchmarking | Wall time and table inflated by 10–20% | Turn off when you care about absolute timing |
| Reading `Self CPU` to judge GPU work | "Why is my kernel only 0us of CPU?" | TileLang kernel = GPU work; look at `Self CUDA` |
| Using kernel name to infer tensor-core use | "It's called gemm_kernel so it must use tensor cores" | Always confirm with `kernel.get_kernel_source()` — TileLang names kernels after the prim_func, not after what they lower to |
| Comparing tiny kernels with the `event` backend | Numbers vary 30% run-to-run | Sort by median over ≥20 reps; for very small kernels use `do_bench(backend="cudagraph")` |
| Profiling a freshly created `@tilelang.jit` kernel | First call includes minutes of autotune | Either warm up first or autotune separately and cache before profiling |

## Escalation

| Symptom from torch.profiler | Next skill / tool |
|----------------------------|-------------------|
| Kernel is slow but achieved TFLOPS far below peak; cause unclear | `profiling-tilelang-programs` (ncu) — pipe util, warp stalls, register spill |
| Wrong output / NaNs | `debugging-tilelang-programs` |
| Want to actually fix the slowness (tile sizes, pipeline, fusion) | `optimizing-tilelang-programs` |
| Need fwd+bwd timeline together | This skill, scheduled mode + `tensorboard_trace_handler` |
| Allocator fragmentation / OOM | `_record_memory_history` + https://pytorch.org/memory_viz |

For the detailed roofline-classification decision tree and source-inspection
checklists, read `references/bottleneck-classification.md`. For deeper
guidance on Chrome trace / Perfetto navigation, read
`references/trace-reading-guide.md`.
