# Microbenchmarks for static analysis and candidate selection

Collect a small, reusable calibration profile for the actual kernel family and
device. Start with the uncertain terms capable of changing the next decision;
avoid measuring an entire architecture before the first experiment. Use the
task harness and the skill's existing timing tools. The recipes below describe
probes to implement there. The [cost-model guide](../../../tilelang-optimization-loop/references/cost-model.md)
links the bundled copy and single-CTA GEMM starting probes and states their
measurement scope; they do not cover the entire matrix below.

## Establish the measurement contract

1. Query device identity/UUID, SM architecture and count, memory/cache capacity,
   limits and any partitioning. Record driver/toolchain/compiler, clocks/power
   policy, temperature, ECC/MIG/MPS state when applicable, and competing work.
   Hardware specs and Wiki tables are reference inputs, not measured rates.
2. Fix dtype, instruction family, tensor strides/alignment, sizes, byte-count
   convention, cache state, grid/block configuration, stream, allocations and
   resets. Choose kernel-only or operation scope and retain it across probes.
3. Compile and inspect the exact specialization. Verify the intended load,
   copy, barrier or MMA survives lowering and optimization. Make results
   observable and numerically check them. Save source, generated code and
   compiler/resource reports with hashes.
4. Warm compilation and runtime separately. Time uninstrumented execution;
   collect NCU/IKET/source traces in separate diagnostic runs. For short work,
   amortize timing overhead without accidentally changing dependence or cache
   behavior. Compare several inner-loop lengths to identify slope/intercept.
5. Repeat independent batches, interleave/randomize configuration order and
   retain raw samples. Report median, spread and an uncertainty interval with
   its method. Use independent-batch resampling when samples are correlated;
   repeated launches in one graph are not independent experiments.

Do not silently change clocks or evict other users. Interference makes a rate
uncertain; it does not establish a new hardware limitation. If the requested
device is busy, record the condition and defer calibration that requires isolation.

## Select probes by the model term

| Term | Probe and controlled sweep | Check before using the result |
| --- | --- | --- |
| DRAM streaming service | Read, write and copy variants; aligned/strided paths; several working sets larger than L2; grid, bytes per thread and vector width | Count read plus write bytes for copy; distinguish logical bytes from measured DRAM sectors; verify cache regime |
| L2/cache service | Repeated working sets below/near/above cache size; cache policy and reuse distance | A fitting allocation does not prove residency; separate cold and reused cases |
| Dependent-load latency | Random permutation pointer chain, serialized loads, multiple working sets and chain lengths | Keep a true dependence, consume final result, inspect code, separate cache/TLB regimes and loop overhead |
| Async global-to-shared path | Match cp.async/TMA/vector path, per-site bytes, slots, issuing warps, commit/wait placement and consumer gap | Allocated stages are not measured in-flight requests; distinguish launch/setup, fill, steady state and drain |
| Shared-memory service | Match lane layout, bank/stride pattern, vector width and read/write mix | Broadcasting a read is different from conflict serialization; distinguish service from barriers |
| Tensor-core compute | Actual MMA family/dtypes/layout and K-loop; dependent accumulator chain and independent accumulator groups | Distinguish latency from throughput, preserve output, verify instruction lowering and register/spill use |
| Conversion/reduction/SFU | Actual operand shape, dependency chain, conversion dtype and participating warps | Avoid folding constants or substituting another instruction; keep numerical contract |
| Concurrency and waves | Sweep grid around predicted resident slots; correlate occupancy API/resource reports with traces/counters | Grid size is offered work, not achieved residency; the last wave and uneven CTAs need separate treatment |
| Launch/synchronization | Empty/minimal work and matched useful work, event versus graph scope | Estimate an intercept only in the matching harness; do not subtract unlike launch paths |

For effective bandwidth, use `(read_bytes + write_bytes) / seconds`, with a
declared decimal/binary convention. A read+write copy rate is not a one-way
read rate. Sparse/FP8 advertised throughput is not dense BF16 throughput.
Do not divide saturated-device bandwidth by an underfilled grid's nominal SM
count and call that a measured per-SM service curve.

For each access site, sweep alignment, transaction size, cache regime, active
warps, issue mechanism and buffer depth only as needed by the candidate family.
Include an intentionally mixed pipeline: leave one consumer-critical operand
synchronous while the others are prefetched, then stage that operand. This
tests whether the model can name the exposed access rather than recommending
more buffers for everything.

## Timing implementation

Use `tilelang.profiler.do_bench` or the task's equivalent with verified units
and policy. Its warmup/repetition budgets are time budgets in the inspected
API, not counts. Check cache-flush and graph behavior rather than inheriting
defaults unknowingly. For true hot-cache microbenchmarks, use an explicit
matching harness; a cache-flushing benchmark answers a different question.

CUDA events measure a device-stream interval. A repeated inner loop can reduce
launch overhead, but it can also introduce reuse, a serial chain, or different
instruction scheduling. Demonstrate the intended steady state and sweep work
amount. An empty-kernel subtraction is not universally valid. Graph replay
changes dispatch overhead and can retain data; label it as its own regime.

Use the [IKET reference](iket-profiling.md) when installed tooling supports the
needed trace. Warp-lifetime tracing can estimate per-SM active intervals;
body markers and external cubin tracing are distinct capabilities. Verify the
installed profiler's mode, timestamp/SM identity and perturbation before
interpreting concurrency. Missing IKET does not justify inventing residency:
use compiled-resource/occupancy APIs and available counters, or mark it unknown.

## Store data for a solver

Keep raw samples and a versioned manifest, not just fitted coefficients:

| Record | Fields |
| --- | --- |
| Identity | Device UUID/partition, SM capability, versions, clocks/power policy, source/helper/generated-code hashes |
| Probe | Mechanism and stable access-site ID; shape/dtype/layout/alignment; per-site bytes; instruction/copy path; slots and waits |
| Execution | Grid, threads, allocated smem/registers/spills with provenance; offered and measured concurrency; cache/stream policy |
| Observation | Commands, raw times and units, correctness result, diagnostic artifacts, interference, derived rate and its counting formula |
| Fit | Feature/domain definition, train/holdout assignment, coefficient/table version, residual distribution and uncertainty method |
| Applicability | Valid shape/staging/concurrency/cache regimes, invalidation conditions, source kind (`queried`, `vendor`, `wiki`, `measured`, `inferred`) |

Use integer nanoseconds or explicit rational units at the solver boundary.
Keep lower/median/upper predictions and coverage flags; quantization must not
round an optimistic lower bound upward. Do not serialize unavailable values as
zero, or encode empirical confidence as a proven constraint.

Training and holdout rows must not be merely different repetitions of the
same configuration. Hold out tile families, sizes around cache/occupancy
transitions, short/long K loops and tail cases. Test paired full kernels that
share bytes/FLOPs but differ in dependence or staging. Report top-candidate
recall and selection regret, not only fit residuals.

When the staging path, compiler, device partition or resource usage changes,
invalidate the affected rows/fits. When a cheaper probe explains a new regime,
add it deliberately and reevaluate deferred candidates. Reference the profile
and checks from the shared
[stage report](../../../tilelang-optimization-loop/references/stage-handoff.md).

The [cost-model guide](../../../tilelang-optimization-loop/references/cost-model.md)
defines how to compose these measurements. For the timing and byte-count
contracts, see NVIDIA's
[CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#performance-metrics).
