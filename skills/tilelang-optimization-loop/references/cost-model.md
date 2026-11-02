# Calibrated cost models for kernel search

Use a model when its calibration cost can be amortized over a broad tiling or
schedule search, or when measurements contradict a traffic-only ranking. Keep
the mathematical contract fixed. A model proposes which candidates to compile
and measure; correctness and the workload benchmark decide promotion.

Read [microbenchmark calibration](../../tilelang/profiling-tilelang-programs/references/microbenchmark-calibration.md)
to collect coefficients and [constraint pruning](constraint-pruning.md) to use
them alongside exact tensor/layout constraints. No hardware rate from this
document, KernelWiki or a previous task is a calibrated constant for this task.

Two bundled starting probes are [stream_sweep.py](cost-model/stream_sweep.py)
and [gemm_fit.py](cost-model/gemm_fit.py). Run them from their containing
directory in the task environment:

```bash
python stream_sweep.py --size-mib 512 --ctas-per-sm 1,2,4,8 --batches 5
python gemm_fit.py --shapes 64x64x64,128x128x64 --dtype float16 --batches 5
```

The first reports whole-device copy GB/s versus offered grid work, not achieved
residency or per-SM bandwidth. The second times a complete single-CTA
load/GEMM/store kernel, not isolated dot latency. Both emit repeated-batch times
using ordinary launches with buffer reuse and no explicit cache flush. They
are starting probes, not a complete calibration profile: use the measurement
contract below and add matched dependent-latency or async-access probes when
those terms drive the decision.

## Separate facts, bounds and predictions

| Quantity | Source and use |
| --- | --- |
| Shapes, operation semantics, required bytes/FLOPs | Authored IR plus validated access/counting rules; record logical versus physical traffic |
| Device limits, instruction availability | Active-device queries and applicable compiler/vendor contracts; corroborate Wiki summaries |
| Ideal reference time | Named target's specification rates and explicit cache/traffic assumptions |
| Calibrated time interval | Repeated matched microbenchmarks plus error on held-out configurations |
| Actual candidate time | Correct candidate, fixed benchmark scope, variation and interference recorded |

KernelWiki is a discovery source for mechanisms and references, not a hardware
measurement database. Even vendor rates need the right SKU, clock conditions,
precision, sparsity mode and byte convention. Keep specification and measured
profiles separate; do not silently replace a target's hardware facts with a
fit for one kernel family.

With justified upper service-rate bounds and mandatory work/traffic, a simple
optimistic bound is `max(F / P_max, B_level / BW_max)` across resource levels.
A measured copy maximum is a rate attained by that probe, not a proven maximum
for every possible kernel. A prediction interval or empirical roofline is not
a mathematical lower bound. If a candidate beats a purported bound, revisit
its units, traffic, cache, concurrency and applicability before using it again.

TileFoundry's `ideal-ns` is the authored model's ideal reference;
`bound-by` identifies its limiting resource under those assumptions. It does
not certify a globally safe pruning bound. Its performance analysis prices
modeled in-flight bytes and the running mesh, not measured register allocation,
resident CTAs or achieved software-pipeline overlap.

## Preserve per-access staging

For each operand access site, record stable site/tensor ID, bytes per CTA per
iteration, storage path, dtype/layout/alignment, synchronous or asynchronous
issue, buffer slots, wait/commit position, consumer dependence and reuse scope.
Mark whether each field is authored, extracted from lowering, or measured.

The current TileFoundry performance path estimates in-flight bytes per Call:
async categories use modeled transfer bytes and declared buffer slots;
synchronous accesses use a thread-count-based default. Reuse available facts
from `analysis/compute_cost.py`, `analysis/emission.py`, memory metadata and
iteration scopes. Those defaults do not prove actual outstanding requests.
Correlate them with the generated TileLang loads, waits and resource use.

A loop with seven prefetched operands and one synchronous load is a mixed
pipeline. Identify whether that load lies on the consumer's critical path;
do not average it into a stage-wide in-flight value or assume every synchronous
load is exposed. Increasing every buffer may consume shared memory without
moving that load earlier. Compare moving the specific access, adding stages,
and preserving the original pipeline as distinct hypotheses.

For a matched access class, Little's law gives a model term
`rate <= outstanding_bytes / latency`. Outstanding bytes are concurrent bytes
on that path, not all allocated buffer bytes. Shared paths also have a common
service cap; adding per-operand peak rates cannot exceed it. Pointer-chase
latency, async completion latency and consumer stall time are different terms.
Recalibrate affected terms when copy mechanism, wait placement or staging changes.

## Residency, waves and compute service

Use the compiled kernel's occupancy API/resource report when available. Resource
ceilings include registers and their allocation granularity, shared memory,
threads/warps, block limits and relevant cluster constraints. Logical fragment
elements are not physical register counts. A hand estimate can bound a known
mandatory allocation; it cannot prove the compiler's final residency.

Achieved concurrency is a distribution over time and SMs. Use an appropriate
trace or occupancy measurement, with uninstrumented timing for selection.
Grid sweeps reveal underfill/saturation and tail behavior; a throughput plateau
alone does not establish resident CTAs/SM or a register count. If concurrency
is unknown, evaluate plausible regimes and retain candidates whose rank changes.

For uniform independent CTAs and an integer resident-slot model,
`slots = SMs * r_slots`, `waves = ceil(ctas / slots)`. Distinguish full waves
and the final partial wave: its compute contention and bandwidth can differ.
Do not insert average achieved occupancy into an integer wave formula or assume
every block has the same duration. Persistent, clustered and heterogeneous
workloads need a matching schedule model.

Measure dependent MMA latency separately from throughput with independent
accumulators. Match instruction shape, precision, K length, accumulator layout
and number of issuing warps. A fitted intercept per small dot is not necessarily
call overhead. Merging dots earns a model speedup only when changed instruction
work or a matched measurement supports it. Do not force linear serialization
by multiplying every dot cost by resident CTA count; measure sharing regimes.

## Compose a model without double counting

Name the unit for every coefficient: device, SM, CTA, warp or instruction;
bytes/s versus bytes/ns; elapsed time versus service time. Use per-CTA traffic
with per-CTA service rates. Dividing a whole-device result by all SMs does not
measure an active SM's rate on an underfilled grid.

Model issue/completion/consumer dependencies and shared service resources.
Serialize exposed chain segments; permit overlap only where dependencies and
resources allow it. A single `max(compute, memory, chain)` is a useful coarse
estimate only within a validated regime, not a general pipeline composition
rule. Do not charge an exposed load in both its path and a fitted residual
chain term. Preserve prologue, steady-state and drain costs for short loops.

Prefer measured tables or piecewise fits over shape/staging/concurrency classes.
Keep dimensions finite for solver search and precompute nonlinear features
where practical. Return an interval and coverage flag for each prediction;
outside the calibrated domain return unknown, not a confident extrapolation.

## Validate the search decision

Hold out shapes and configuration families, including near residency and cache
transitions, ragged tails and a different staging path. Report prediction error,
rank agreement, recall of measured best/top candidates, and final selection
regret under an equal GPU-trial budget. Validate complete kernels as well as
isolated probes. Small average timing error can still discard the winner.

Keep the best validated candidate, diverse near-optimal choices, boundary cases
and an exploration sample of deferred candidates. A counterexample invalidates
the affected fit/regime and can reopen previously deferred configurations.
Track compile count, GPU trials and calibration cost to show whether pruning
actually saves work.

In the candidate/stage report, record the calibration artifact/hash, covered
regime, access-site table, prediction interval, held-out residuals and reason
for retain/defer/reject. Report `measured / ideal_reference` and
`measured / predicted_median` with distinct labels; neither proves remaining
headroom. Benchmark results override the fitted explanation.
