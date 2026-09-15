# Inspect the Existing PyTorch Baseline Before Drafting

Start from the supplied PyTorch implementation and representative inputs.
Keep eager execution as the correctness reference. Inspect compiled execution
to understand its fusion, temporary storage, layout assumptions, and launch
structure before choosing what to implement in TileLang or CUDA.

## Compile and validate the baseline

Extend the existing harness. For a stateless forward callable, with
`torch_baseline`, `args`, `rtol`, and `atol` supplied by that harness:

```python
import torch

compiled_baseline = torch.compile(torch_baseline, backend="inductor")
expected = torch_baseline(*args)
actual = compiled_baseline(*args)  # First invocation triggers compilation.
torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

for _ in range(10):
    torch_baseline(*args)
    compiled_baseline(*args)
torch.cuda.synchronize()
```

Preserve existing compile settings when they are part of the workload; record
backend, mode, dynamic/fullgraph options, and shape cases. Do not silently
turn a partially compiled workflow into a different operation by changing
`fullgraph` or suppressing errors. For mutation, RNG, or saved state, restore
equivalent state and use independent inputs for each path. For training, also
validate and warm backward with matching upstream gradients: backward can
compile lazily on its first execution.

## Capture graphs, lowering artifacts, and generated code

Run the harness in the selected environment:

```bash
TORCH_LOGS="graph_code,graph_breaks,output_code" \
  TORCH_COMPILE_DEBUG=1 TORCH_COMPILE_DEBUG_DIR=/tmp/baseline-debug \
  python inspect_baseline.py 2> /tmp/baseline-compile.log
```

Use a fresh process. If a cache hit omits the needed compiler artifacts, give
the inspection run a fresh `TORCHINDUCTOR_CACHE_DIR`; do not delete a shared
cache. Bound compilation workers with `TORCHINDUCTOR_COMPILE_THREADS` when
resources are shared. Follow the artifact paths printed by the installed
version rather than hardcoding its run-directory naming convention.

| Evidence | Inspect for |
| --- | --- |
| `graph_code` / FX graph | Captured operators, dimensions, dtypes, reductions, and graph boundaries |
| `graph_breaks` | Eager regions and host boundaries that affect actual execution |
| `fx_graph_transformed.py` | Graph after compiler transformations/decompositions |
| `ir_pre_fusion.txt`, `ir_post_fusion.txt` | Inductor's buffer dependencies and fusion groups |
| `output_code.py` or emitted source | Actual launches/library calls, intermediates, allocation, strides, and indexing |
| `guards`, `recompiles` logs | Shape/stride/scalar assumptions and specialization changes |

The debug directory can include `fx_graph_runnable.py` for reproducing a graph.
Artifact availability and names depend on the version, backend, and cache.
For a focused question, add only the relevant logs, for example
`TORCH_LOGS="guards,recompiles"` while exercising the intended shape cases.
Inspect available names with `TORCH_LOGS=help python -c 'import torch'`.

These artifacts show selected compiler boundaries, not every lowering pass.
Generated code may call external libraries or emit Triton kernels. Read it to
explain dataflow, fusion, and launch structure; choose the custom TileLang/CUDA
schedule based on the requested contract and measurements.

## Correlate with the warmed execution profile

Use the [torch profiling workflow](../SKILL.md) with distinct
`baseline.eager` and `baseline.compiled` ranges. Capture after compilation and
warmup, and inspect which device kernels/copies belong to each range. Keep a
separate uninstrumented [benchmark](../../profiling-tilelang-programs/SKILL.md)
for comparable latency. Re-run it without compiler-debug logging when timing.

Connect an expensive region to its graph operators and generated calls.
Check whether a temporary/copy actually survives lowering and whether a fused
region already avoids the launches seen in eager execution. For backward,
include required zeroing/conversion and all gradient kernels in the scope.
Graph breaks, library calls, and host work can matter as much as generated
kernel count; do not infer operation latency by counting source operators.

## Output for the draft stage

Carry forward the baseline entrypoint and invocation, input/output contract,
validated eager/compiled outputs, compile settings, separate baseline timings,
and paths to the relevant graph/code/profile artifacts. State:

- Which operator or region the custom kernel will replace, including boundaries.
- Which intermediates, layout conversions, reductions, and state must be preserved.
- Which observed cost motivates the implementation and how improvement will be measured.

If compilation fails or the backend is unavailable, retain the validated eager
baseline and profile it. Record the compiler failure and missing evidence;
do not report an eager fallback as a compiled baseline. Continue drafting from
the available source/profile evidence when that is sufficient for the task.

The inspected PyTorch implementation exposes these controls in
`torch/_logging/_registrations.py`, `torch/_inductor/debug.py`, and
`torch/_dynamo/config.py`. Check the active installation when upgrading.
