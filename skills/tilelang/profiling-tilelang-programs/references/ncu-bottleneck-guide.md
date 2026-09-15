# Focused Nsight Collection for Optimization

Use a timeline to identify the operation that matters before collecting a
kernel report. Keep device-specific metric interpretation separate from the
collection workflow.

If ncu-report-skill is available, its report extraction helpers can support this
stage. Read the [integration and compatibility notes](../../../tilelang-optimization-loop/references/external-tools.md)
before using its B200 defaults or selecting an action from a multi-launch report.

## Select the region

```bash
nsys profile --trace=cuda,nvtx -o timeline python benchmark.py
nsys stats --report cuda_gpu_kern_sum timeline.nsys-rep
```

Match observed kernel names to the compiled TileLang function and host ranges.
A single operator can launch multiple kernels, including conversions or
pre/postprocessing. Decide whether to optimize the complete path or one launch.

## Collect a bounded kernel sample

```bash
ncu --list-sets
ncu --list-sections
ncu --kernel-name 'regex:.*gemm.*' --launch-skip 10 --launch-count 1 \
  --set basic -o selected_kernel python benchmark.py
```

Use `--kernel-name` for regex filters. `--kernel-name-base` selects the name
representation, not a regex. Adapt the skip count and filter to the observed
launch sequence; do not blindly collect startup, warmup, or every autotune
candidate. Add `--target-processes all` when the workload launches child
processes and they are part of the intended collection.

Discover section identifiers from the installed version. Common identifiers
include `LaunchStats`, `Occupancy`, `ComputeWorkloadAnalysis`,
`MemoryWorkloadAnalysis`, `SchedulerStats`, `WarpStateStats`, and `SourceCounters`.
The section filename/display name is not necessarily the CLI identifier.
Do not invent sets such as `--set memory` or `--set source`.

## Connect the report to the program

Record shape/configuration and the matching generated host/device source.
Compare resource usage, work distribution, and relevant reported limitations.
Use a lowering trace to inspect where loops, copies, allocations, or scheduling
changed. Form a concrete hypothesis; a profiler recommendation alone does not
establish the cause.

After the edit, rerun correctness and the original benchmark. Nsight collection
can replay/serialize execution and adds overhead, so its instrumented wall
time is not the before/after baseline.

If permissions or tool availability prevent collection, use torch.profiler
and source/IR inspection for the questions they can answer, and report the
missing evidence. Hardware counters and instruction-specific tuning can be
introduced when the investigation requires them.
