---
name: optimizing-tilelang-programs
description: Tune a correct TileLang kernel using measured bottlenecks, tile sizes, layouts, pipelining, fusion, and autotuning. Use when improving kernel performance; use profiling to establish measurements and debugging to resolve incorrect output.
---

# Optimizing TileLang Programs

Establish correctness and a baseline at the user's production shapes, dtypes,
target, and execution backend. Record the measurement backend and whether
allocation, cache flushing, and launch overhead are included. See
[profiling](../profiling-tilelang-programs/SKILL.md).

## Choose a change from evidence

| Evidence | Candidate changes |
| --- | --- |
| Excess global-memory traffic | Fuse an epilogue, reuse tiles, remove redundant copies |
| Unexpected layout conversions | Inspect tensor strides, fragment mappings, and generated copies |
| Excess temporary storage or repeated work | Reduce live intermediates or restructure the loop |
| Copy/compute stalls | Tune pipeline depth and reduction tile size; inspect dependencies |
| Small-workload launch gaps | Fuse launches, batch work, or use CUDA graphs when applicable |
| Computation differs from the intended schedule | Compare specialized TIR and lowering passes |

Compare one hypothesis at a time when diagnosing a bottleneck; a controlled
autotune search can vary interacting parameters together. Revalidate correctness
after changes and keep measurements for successful and unsuccessful variants.
When excess temporary storage, shared-memory reuse, or fragment pressure is the
evidence, use the debugging skill's
[buffer-lifetime workflow](../debugging-tilelang-programs/references/buffer-lifetime-analysis.md)
to separate logical access spans from the final shared arena and physical
register/spill evidence.

Use the supplementary [KernelWiki interface](../../tilelang-optimization-loop/references/kernelwiki.md) to
research a mechanism suggested by the lowering or profile. Link the retrieved
source to an expected code/resource change and measurable effect. Inspect the
new lowering to check that the intended change materialized, then validate and
remeasure; keep this evidence with the candidate.

## Separate the algorithm from its schedule

Expose tile sizes, loop ordering, and pipeline stages as configuration choices.
Keep the mathematical reference and input/output contract stable while tuning.
Inspect lowered loops, copies, temporary buffers, and launch configuration to
check that a candidate expresses the intended schedule.

Shared memory helps reuse or layout conversion. Direct global-to-fragment
copies or direct `T.Parallel` accesses can be preferable for elementwise work.

## Autotuning and pipelines

Read [autotuning-guide.md](references/autotuning-guide.md) for a runnable
decorator example, plain-factory programmatic tuning, fixed inputs, and cache
controls. Keep search spaces bounded by available CPU/GPU resources.

Start with inferred `T.Pipelined(..., num_stages=...)` scheduling. Read
[pipeline-and-layouts.md](references/pipeline-and-layouts.md) only when explicit
stage/order arrays, async copies, or layout choices are relevant. It explains
which scalar bindings consume pipeline annotation slots.

For a detailed investigation, use
[optimization-checklist.md](references/optimization-checklist.md). For ncu
interpretation, use the profiling skill's
[bottleneck guide](../profiling-tilelang-programs/references/ncu-bottleneck-guide.md).

## Completion evidence

Preserve the correct baseline and choose a finite search budget. Return the
selected source/configuration, validation results, baseline and candidate
timings under the same conditions, and rejected hypotheses. Select using the
requested workload objective; do not treat one faster shape as an improvement
for every shape. If results are within observed variation or the search finds
no improvement, retain the baseline. Use the suite's
[experiment record](../../tilelang-optimization-loop/references/experiment-record.md)
for a longer task or handoff.
