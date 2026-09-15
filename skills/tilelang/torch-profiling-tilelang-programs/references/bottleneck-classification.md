# From Trace Observations to Optimization Experiments

Torch profiling provides attribution and a timeline. Use those observations
to choose a change and verify it with an independent benchmark; do not classify
an instruction-level bottleneck from kernel names or a fixed throughput cutoff.

| Observation | Inspect next | Candidate experiment |
| --- | --- | --- |
| Long first call only | Compilation, lazy module loading, allocator startup | Warm both paths and profile a stable window |
| CPU activity while the GPU has gaps | Python/framework work and dependencies | Move repeated setup out of the loop or batch work |
| Many short dependent launches | Operation boundaries and launch sequence | Fuse stages or test capture-compatible graph replay |
| Repeated copies or conversions | Input/stride contracts and host source | Remove redundant preparation or change the interface |
| One device kernel dominates | Matching TIR, generated source, focused ncu report | Change a relevant configuration and compare |
| Time in backward preparation | Zeroing, materialization, conversion, intermediates | Measure complete backward variants |
| Allocation growth | Profiler memory timeline or allocator snapshot | Identify tensors retaining storage and their lifetimes |

CPU launch cost comparable to kernel duration suggests launch overhead may
matter. Confirm gaps and dependencies in the timeline; CPU and GPU work can
overlap, so comparing two aggregate totals is not a proof.

Label custom and reference ranges. Sum each logical work item once: a GPU
annotation and the kernel it surrounds can both appear in the table. For
concurrent streams, elapsed workflow time is not the sum of all kernel times.

When source inspection is needed, compare before/after TIR and generated code
from the same shape/configuration. Header inclusion or transfer helper names
alone do not prove which compute instructions execute. Defer instruction-level
claims until the corresponding evidence is available.

Retest the candidate with the same `do_bench` backend and input/output state.
Keep the observed benefit specific to the measured workload and report what
the experiment includes: allocation, reset, conversion, cache policy, and
launch/capture behavior.
