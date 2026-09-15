# Focused Nsight Compute Collection

Use a timeline first to choose the kernel and invocation worth inspecting.
Keep ordinary before/after timing separate from instrumented collection.

For ncu-report-skill's structured extraction helpers, see the suite's
[integration and compatibility notes](../../tilelang-optimization-loop/references/external-tools.md).
Use supported local sections and preserve the selected kernel/invocation.

```bash
ncu --list-sets
ncu --list-sections
ncu --kernel-name 'regex:.*myKernel.*' --launch-skip 10 --launch-count 1 \
  --set basic -o report ./program
```

`--kernel-name` accepts a regex filter. `--kernel-name-base` selects the name
representation (for example `demangled`); it does not accept a regex. Match
skip/count to the observed launch sequence and use `--target-processes all`
only when the relevant work runs in child processes.

## Select supported sections

Use identifiers from the installed `--list-sections` output. Common examples:

```bash
ncu --section LaunchStats --section Occupancy ./program
ncu --section MemoryWorkloadAnalysis --section ComputeWorkloadAnalysis ./program
ncu --section SourceCounters --kernel-name 'regex:.*myKernel.*' ./program
```

Display names and section filenames may differ from CLI identifiers. Common
identifiers include `SchedulerStats` and `WarpStateStats`, not their longer
display names. Sets such as `memory`, `compute`, `launch`, or `source` are not
portable predefined sets; do not assume they exist.

## Preserve useful evidence

Record shape, dtype, configuration, target, selected invocation, and report
path. Build with `-lineinfo` when source correlation is needed. For a generated
kernel, retain the corresponding TIR and host/device source. Use CSV or the
saved report when comparing specific sections across revisions.

```bash
ncu --csv --set basic --launch-count 1 ./program
ncu --import report.ncu-rep --page details
```

Profiler collection can serialize or replay work and adds overhead. Its wall
time is not the application's baseline latency. Treat suggestions as
hypotheses, validate outputs after a change, and repeat the original timing
harness. Hardware-specific counter interpretation belongs to a separate
investigation when needed.
