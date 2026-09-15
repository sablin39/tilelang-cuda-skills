# A Repeatable Optimization Experiment

## Record the baseline

Keep the operation contract, representative input shapes/dtypes/strides,
configuration, target/backend, input/output state, and benchmark settings.
Separate compilation and warmup from steady-state timing. Validate the custom
operator against its reference before investigating performance.

## Follow the observed work

1. Use a named timeline region to identify expensive launches, preparation,
   copies, synchronization, and output handling.
2. Retrieve specialized TIR and generated host/device source for that exact
   configuration. Inspect whether they express the intended operation.
3. Use `TL_LOWER_TRACE` when a transformation or pipeline decision is unclear;
   use layout visualization when an inferred mapping needs explanation.
4. Propose a specific change: remove redundant work, fuse an epilogue, adjust
   a configuration, or change scheduling. Avoid guessing from a kernel name.
5. Revalidate correctness, then repeat the same benchmark. Keep the change
   only when its benefit is meaningful for the intended workload.

## Bound a search

Use a small set of valid configurations first. Tune interacting parameters
together when that is the hypothesis, and use representative inputs for every
candidate. Limit workers to available resources. A cached best result is not
a fresh comparison after changing the measurement environment.

## Report the result

| Variant | Shape/dtype | Measurement scope | Latency | Correctness | Observation |
| --- | --- | --- | --- | --- | --- |
| Baseline | Record | Kernel/operator/workflow | Measure | Check | Reason for change |
| Candidate | Same | Same | Measure | Check | Observed effect |

Keep estimates distinct from measurements and explain excluded work such as
allocation or conversion. Validate representative tails and repeated calls
after changes to indexing, output ownership, or accumulation.

Use [autotuning-guide.md](autotuning-guide.md) for executable tuning patterns,
[pipeline-and-layouts.md](pipeline-and-layouts.md) for scheduling semantics, and
the debugging skill's [compiler tools](../../debugging-tilelang-programs/references/compiler-tools.md)
for inspection commands.
