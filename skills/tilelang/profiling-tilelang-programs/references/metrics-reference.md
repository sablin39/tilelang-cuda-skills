# Measurement Contracts and Static Estimates

## Choose what a number includes

| Measurement | Include and report |
| --- | --- |
| Compiled kernel launch | Inputs already prepared; output allocation policy specified |
| User-visible operator | Required allocation, conversion, zeroing, and all launches |
| Graph replay | Capture performed before timing; compatible state/reset behavior |
| Complete workflow | Host work, synchronization, copies, forward/backward phases |
| Compilation/autotuning | Measure separately from steady-state execution |

Use a selected, available GPU and the same stream, shapes, dtypes, layouts,
input state, and benchmark backend for comparisons. Warm both implementations.
Include reset cost when it belongs to the operation; do not repeatedly time
atomic accumulation into an already-accumulated output as if each run were fresh.

`do_bench` uses millisecond budgets and returns milliseconds. Profiler events
expose microsecond durations; Chrome trace and other formats may use different
units. Check the format before converting or aggregating.

## Derived quantities

```python
gemm_tflops = (2 * M * N * K) / latency_ms * 1e-9
effective_gb_s = logical_bytes / latency_ms * 1e-6
```

Count input/output dtypes separately. For row reduction with fp16 input and
fp32 output, logical bytes are `M*N*2 + M*4`. For GEMM, reading each input
once and writing each output once is a logical lower bound, not a measurement
of memory traffic: tile reuse, caching, and redundant reads change actual
transfers. State what the numerator models.

Summed kernel durations differ from elapsed time when work overlaps. Ranges
and their child kernels can be duplicate representations of the same work.
Use the timeline's dependency path for workflow latency.

## Static TIR analysis

For a drafted tiled operator, inspect recognized work before benchmarking:

```python
from tilelang.tools import Analyzer
from tilelang.carver.arch import CUDA

tir = matmul.get_tir(M, N, K, block_M=128, block_N=128, block_K=32)
estimate = Analyzer.analysis(tir, CUDA("cuda"))
print(estimate.total_flops, estimate.total_global_bytes)
print(estimate.estimated_time)
```

Use the actual factory signature. `CUDA("cuda")` queries a visible device, so
this particular model setup is not a GPU-free workflow even though analysis
does not launch the kernel.

The current analyzer recognizes FLOPs from `T.gemm` and global traffic from
selected `T.copy` regions; direct loads/stores and elementwise arithmetic are
not fully counted. Grid scaling covers x/y but not z, and compute models are
available for only some capabilities. Treat missing peak values or incomplete
coverage as model limitations. An estimate cannot diagnose actual scheduling,
cache behavior, or runtime bottlenecks.

Use compile-only/source inspection when the question is about emitted work,
then validate the estimate against measurement. Source:
[Analyzer docs](https://github.com/tile-ai/tilelang/blob/main/docs/tools/analyzer.md).

## Unstable or surprising measurements

- Startup in the trace: warm custom and reference paths, including backward
  when measured. Exclude compilation and library initialization deliberately.
- Missing device events: verify CUDA profiler support; do not substitute CPU
  dispatch duration for device execution.
- High variance: check competing work, measurement duration, synchronization,
  and state changes. Repeat only enough to answer the comparison reliably.
- Apparent speedup after changing a trace option: remeasure with the same
  uninstrumented benchmark harness before claiming an improvement.
