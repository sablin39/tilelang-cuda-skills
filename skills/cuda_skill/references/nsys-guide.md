# Nsight Systems: Workflow Timelines

Use nsys to locate expensive phases, host launch gaps, synchronization,
allocations, and transfers. Establish a warmed baseline separately; trace
collection changes execution costs. Verify `nsys --version` and installed
command help before selecting version-specific options or reports.

## Capture and summarize

```bash
nsys profile --trace=cuda,nvtx --sample=none -o timeline ./program
nsys stats --report cuda_gpu_kern_sum timeline.nsys-rep
nsys stats --report cuda_api_sum timeline.nsys-rep
nsys stats --report cuda_gpu_mem_time_sum timeline.nsys-rep
nsys stats --report nvtx_sum timeline.nsys-rep
```

For Python, replace `./program` with the intended interpreter and script.
Select a short representative workload, warm JIT/library paths before the
region of interest, and use NVTX names to distinguish implementations or phases.

| Report | Use it to locate |
| --- | --- |
| `cuda_gpu_kern_sum` | Frequent or expensive device kernels |
| `cuda_api_sum` | Host CUDA API calls, including allocation and synchronization |
| `cuda_gpu_mem_time_sum` | Device transfer time by operation |
| `cuda_gpu_mem_size_sum` | Transfer volume |
| `nvtx_sum` | Annotated host regions |

Summary tables aggregate events. Inspect the timeline for overlap and causal
ordering before adding durations or interpreting a gap. A long synchronization
API call may be waiting for earlier device work. A host NVTX range around an
asynchronous launch measures its host scope, not the kernel duration.

## Narrow capture and export

```bash
# Use only if the application calls cudaProfilerStart/cudaProfilerStop.
nsys profile --trace=cuda,nvtx --sample=none \
  --capture-range=cudaProfilerApi -o focused ./program

nsys stats --report cuda_gpu_kern_sum --format csv timeline.nsys-rep > kernels.csv
nsys export --type sqlite --output timeline.sqlite timeline.nsys-rep
```

Keep the `.nsys-rep` for timeline inspection. CSV is useful for tabular
comparisons; SQLite supports queries against the collected events. Check the
installed report list and schema before automating extraction.

## Turn an observation into an experiment

| Observation | Investigate next |
| --- | --- |
| Device gaps | Host work, stream dependencies, synchronization, launch cadence |
| Many small launches | Their contribution to operation latency; then test batching/fusion/graphs |
| Frequent copies or allocations | Tensor conversion, output allocation, wrapper code, reuse lifetime |
| One expensive kernel | Filter that kernel in [ncu](ncu-guide.md) and inspect its generated source |
| Unexpected extra work | Link NVTX/operator regions to the actual launches and host code |

Do not diagnose launch overhead solely from a kernel-duration threshold.
Record the selected region and supporting events, propose a specific change,
validate results, and compare again with the baseline timing method.

If CUDA events are absent, check that the chosen workload actually launches
CUDA work, the capture range includes it, and the tracing tool supports the
runtime. An empty trace is not evidence of zero cost. For environments without
nsys, [torch.profiler](../../tilelang/torch-profiling-tilelang-programs/SKILL.md)
can provide operator/device attribution with a different collection scope.
