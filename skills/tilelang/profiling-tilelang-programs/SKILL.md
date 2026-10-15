---
name: profiling-tilelang-programs
description: Measure TileLang kernel and workflow performance, choose benchmark or timeline tools, and compare implementations fairly. Use for latency, throughput, profiler selection, and establishing optimization baselines on TileLang-generated code; use the cuda skill for hand-written CUDA C++ sources.
---

# Profiling TileLang Programs

Define what is being measured: an isolated launch, an operation including output
allocation, a graph replay, or a complete forward/backward pipeline. Validate
correctness before interpreting speed. Record shapes, dtypes, strides, selected
device, versions, warmup, repetition settings, and cache policy.

## Choose the measurement

| Question | Tool | Meaning |
| --- | --- | --- |
| How long does this callable take on the device? | `do_bench(backend="event")` | Event timing; may include gaps between GPU work |
| What is replay performance? | `do_bench(backend="cudagraph")` | CUDA graph replay; requires capture-compatible work |
| Which operators, launches, or copies consume time? | `torch.profiler` or `nsys` | Timeline and attribution; instrumentation adds overhead |
| Which phases execute inside a CUDA kernel? | IKET markers/ranges | Experimental warp-level events; requires the external IKET runtime |
| Which kernel deserves deeper inspection? | `ncu` filtered to that kernel | Selected diagnostic sections; separate from baseline timing |
| What work does this TIR express? | `tilelang.tools.Analyzer` | Static estimate with limited operator coverage |

Use [torch-profiling-tilelang-programs](../torch-profiling-tilelang-programs/SKILL.md)
for a ready-to-run timeline and allocation workflow. `do_bench(backend="cupti")`
uses profiler infrastructure but returns a timing result; use a trace to inspect
individual kernels.

For compiler-derived fragment/local, shared, or global access spans, use the
debugging skill's
[buffer-lifetime workflow](../debugging-tilelang-programs/references/buffer-lifetime-analysis.md).
Those intervals use compiler event order; use a runtime trace when elapsed time
or overlap on the GPU is the question.

For named markers, ranges, and optional scalar payloads inside one CUDA kernel,
read [iket-profiling.md](references/iket-profiling.md). Use an ordinary system
timeline first when launch, copy, CPU/GPU overlap, or cross-kernel attribution
can answer the question.

Before choosing IKET, run the dependency gate in that reference. The external
IKET profiler is not installed by TileLang. If a required component is missing,
identify the active Python environment and ask the user whether to install it
there; do not modify the environment without their approval. PyTorch is needed
only by examples or harnesses that import it.

## Benchmark a compiled factory-style kernel

```python
# kernel is compiled; a and b are inputs; output is returned via out_idx.
from tilelang.profiler import do_bench
import torch

ref = lambda: a @ b
torch.testing.assert_close(kernel(a, b), ref(), rtol=1e-2, atol=1e-2)
for _ in range(10):
    kernel(a, b)
    ref()
torch.cuda.synchronize()

latency_ms = do_bench(lambda: kernel(a, b), warmup=25, rep=100,
                     backend="event", return_mode="median")
reference_ms = do_bench(ref, warmup=25, rep=100,
                       backend="event", return_mode="median")
print(latency_ms, reference_ms, reference_ms / latency_ms)
```

The tolerances above illustrate fp16 GEMM; choose tolerances for the actual
operator. Warm **both** implementations so JIT and reference-library startup
stay outside the measurement.

`warmup` and `rep` are time budgets in milliseconds. The profiler object's
`do_bench` accepts `n_warmup` / `n_repeat` for iteration overrides; standalone
`do_bench` uses `_n_warmup` / `_n_repeat` in the inspected version. Check the
signature before supplying version-dependent keywords.

Backend selection (v0.1.15): `event` (CUDA events, default), `cupti`,
`cudagraph`, and `wall` — host wall-clock per sample including launch and
synchronization overhead. Use `wall` when event timing is distorted (e.g.
under MPS) or when the launch path is part of what you are measuring; keep
kernel-only comparisons on `event` and never mix backends within one table.

The current benchmark implementation flushes a cache buffer between samples.
This is a measurement policy, not a requirement for every workload. Compare
implementations under the same policy and separately measure realistic reuse
when it matters. Keep allocation and preallocated-output choices comparable.

## Profiler inputs and dynamic shapes

```python
profiler = kernel.get_profiler()
latency_ms = profiler.do_bench(input_tensors=[a, b], return_mode="median")
# Alternatively, for symbolic kernels with generated inputs:
latency_ms = profiler.do_bench(
    dynamic_symbolic_constraints={"M": M, "N": N, "K": K})
```

Use the symbolic names actually declared by the kernel. For mutable/atomic
outputs, ensure each iteration starts from the required state; include reset
cost if it belongs to the operation, or explicitly separate it in the harness.

For independent static workload estimates and formulas, read
[metrics-reference.md](references/metrics-reference.md). Estimates cannot
replace measured performance.

## Trace and inspect a focused region

```bash
nsys profile --trace=cuda,nvtx -o timeline python script.py
nsys stats --report cuda_gpu_kern_sum timeline.nsys-rep
ncu --list-sets
ncu --list-sections
ncu --kernel-name 'regex:.*kernel.*' --launch-skip 10 --launch-count 1 \
    --set basic -o kernel_report python script.py
```

Adapt the filter and skip count to the actual trace. Use
[ncu-bottleneck-guide.md](references/ncu-bottleneck-guide.md) for the collection
workflow. Inspect the selected kernel's generated source or lowering trace to
connect the measurement to a concrete change; remeasure after validation.

For a hardware explanation of a measured symptom, query through the
[KernelWiki interface](../../tilelang-optimization-loop/references/kernelwiki.md). Match the selected
kernel's generated mechanism and target to the cited source, then state a
testable hypothesis. A matching pattern or counter alone does not establish
the cause.

## Completion evidence

Return the benchmark command, source/configuration, input cases, measurement
scope/backend, units, reset/allocation policy, and comparison results. Include
observed variation when selecting between close candidates. For a trace,
identify the relevant region/kernel and artifact path; distinguish the observed
event from a proposed explanation. Hand optimization a specific measurable
hypothesis, not an unsupported bottleneck label.
