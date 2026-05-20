# Reading torch.profiler Output

Two complementary views: the **summary table** (good for triage) and the
**Chrome trace** (good for understanding timing and overlap).

## The `key_averages().table()` summary

```python
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
```

### What each column actually measures

| Column | Meaning | When to look at it |
|--------|---------|--------------------|
| `Self CPU` | CPU time spent *inside this op only* (not its children) | finding host-side hotspots: dataloader, Python overhead, `cudaMalloc` |
| `Self CPU %` | self CPU as percentage of total CPU time captured | quick ranking — sort by this column |
| `CPU total` | CPU time inside this op + children | the wall-clock cost of a `record_function` range |
| `CPU time avg` | `CPU total / # of calls` | per-iteration cost |
| `Self CUDA` | GPU time spent *inside this op only* (children excluded) | **the column for "which kernel is the GPU spending time on"** |
| `Self CUDA %` | self CUDA as percentage of total CUDA time | quick ranking — usually the column to sort by |
| `CUDA total` | GPU time including child kernels | per-call GPU cost of a `record_function` range |
| `# of Calls` | how many times this event fired | divide CUDA total by this for per-call cost |

The actual CUDA kernel rows (e.g., `gemm_kernel`, `add_kernel`) have
`Self CPU = 0` because the kernel itself is GPU work — the CPU side is the
launch, which lives under `cuLaunchKernel`.

### Useful sort keys

- `sort_by="self_cuda_time_total"` → biggest GPU consumers first. Default for GPU work.
- `sort_by="self_cpu_time_total"` → CPU bottlenecks (dataloader, host preprocessing).
- `sort_by="cuda_memory_usage"` → memory allocators, when `profile_memory=True`.

### Grouping options

```python
prof.key_averages(group_by_input_shape=True)    # requires record_shapes=True
prof.key_averages(group_by_stack_n=5)           # requires with_stack=True
```

Group by shape when you suspect shape-dependent slowdowns (e.g., one
batch size is mysteriously slow). Group by stack when you have the same
op called from multiple Python sites and want to know which one is hot.

### Common patterns in the table

- **Two rows for the same TileLang kernel** — e.g., a row for your
  `record_function("tilelang_matmul")` *and* a row for the underlying
  `gemm_kernel`. Same `Self CUDA`. Use whichever name is more meaningful
  to you; they're describing the same GPU work.
- **`Runtime Triggered Module Loading` / `Lazy Function Loading` with non-zero CUDA time** — first launch is loading the cubin. Increase warmup to push it out of the measured window.
- **`Activity Buffer Request`** — profiler internal; ignore unless it dominates (then your profile window is too short).

## The Chrome trace

```python
prof.export_chrome_trace("trace.json")
```

Two viewers:

- **chrome://tracing** — in any Chrome/Chromium tab, paste `chrome://tracing`, click Load. Old but built-in.
- **Perfetto** — https://ui.perfetto.dev/ → "Open trace file". Handles huge traces, better navigation, can run SQL.

### What the timeline shows

The view has multiple horizontal "tracks":

- **CPU thread tracks** — Python op stack on a per-thread basis. A typical PyTorch script has one main Python thread; multi-worker dataloaders add more.
- **CUDA stream tracks** — one per CUDA stream. Default-stream work goes on stream 7 in the trace (Kineto convention). Other streams (NCCL, custom) get their own rows.
- **`runtime`** track — `cudaLaunchKernel`, `cudaMemcpyAsync`, `cudaStreamSynchronize`, etc. This is the host-side CUDA API.

### What to look for

- **Gaps on the CUDA stream with no concurrent CPU work** → launch overhead between two short kernels. Mitigation: bigger work per launch, CUDA graphs, fusion.
- **Gaps on the CUDA stream filled with CPU activity on the main thread** → CPU-bound. Optimize host-side code.
- **`cudaDeviceSynchronize` near the end of every iteration** → expected; this is `torch.cuda.synchronize()`. Long syncs in the middle of an iteration are suspicious.
- **Wide first kernel on the CUDA stream** → JIT compile or cubin load slipped into the measured window. Insert more warmup.
- **Two streams running kernels in parallel** → good multi-stream pipelining. Common for cuBLAS / NCCL overlap.

### Selecting and measuring intervals

In both viewers, click+drag selects a range and shows total durations.
Useful for "how long is the gap between the last MLP kernel and the first
attention kernel".

In Perfetto, you can run SQL queries — for example, total time spent in
GPU kernels named `gemm_kernel`:

```sql
SELECT name, SUM(dur) AS total_ns
FROM slice
WHERE name = 'gemm_kernel'
GROUP BY name;
```

## How to read launch overhead from the trace

Two indicators converge for launch-overhead-bound code:

1. **Table**: `cuLaunchKernel`'s `Self CPU Total` ≥ ~50% of all `Self CUDA Total`.
2. **Timeline**: CUDA stream is "spiky" — short kernels with consistent gaps between them. Zoom in: each gap should match the host-side `cuLaunchKernel` slot on the runtime track.

When per-kernel duration drops below ~5us, launch overhead dominates on
almost any host. CUDA graphs (`do_bench(backend="cudagraph")` for
measurement, or `torch.cuda.graph` for production) collapse a sequence of
launches into one.

## How to read memory-bound from the trace

The trace tells you "one kernel dominates", but the *bandwidth math* is
what classifies it as IO-bound. From the table:

```
add_kernel:  Self CUDA = 5.226ms total, 20 calls = 261us per call
```

then compute `bytes_moved / per_call_seconds` (see `bottleneck-classification.md`,
§2b). If that's near HBM peak, you're IO-bound regardless of how pretty the
timeline looks.

## How to verify a kernel is tensor-core-using

The trace alone cannot tell you. Two ways:

1. **From the table**: a `gemm_kernel` whose achieved TFLOPS approaches the
   tensor-core peak (e.g., > 100 TFLOPS on a Blackwell SKU at fp16) is
   probably using tensor cores. Anything sitting in the 10–30 TFLOPS range
   on the same GPU is almost certainly on the scalar/FMA pipe.
2. **From the source**: `kernel.get_kernel_source()` then grep for
   `mma.sync`, `wmma::mma_sync`, `wgmma`, `cp.async.bulk`, or `__hmma_*`.
   Presence means tensor-core lowering succeeded.

## Tensorboard plugin (alternative view)

If you prefer Tensorboard's UI:

```python
from torch.profiler import tensorboard_trace_handler
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=tensorboard_trace_handler("./tb_log"),
) as prof:
    for step in range(N):
        train_step(...)
        prof.step()
```

Then `pip install torch_tb_profiler && tensorboard --logdir tb_log`. The
plugin adds an "Operator" view (≈ key_averages table), a "Kernel" view
(per-CUDA-kernel breakdown), a "Trace" view (Chrome-style timeline embedded
in the browser), and a "Memory" view (allocator timeline if
`profile_memory=True`).

The Tensorboard plugin is great for sharing results with teammates or
keeping a record of profiles across a series of experiments.

## When the table and the trace disagree

They shouldn't, but if they do:

- `Self CUDA` in the table missing kernels you see in the trace → the
  trace exited mid-flight. Add `torch.cuda.synchronize()` before exiting `with profile(...)`.
- A kernel in the table with 0us GPU time → the activity was attributed
  to a CPU range whose child kernel was already counted elsewhere
  (i.e., it's the CPU-side row of a `record_function`).
- Numbers that change run-to-run → use `do_bench` for repeatable timing;
  `torch.profiler` is designed for *attribution*, not high-precision
  microbenchmarking. For sub-microsecond comparisons, use `do_bench`
  with the `cudagraph` backend.
