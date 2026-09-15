# Reading a Workflow Trace

Use the aggregate table to find expensive work and the timeline to understand
ordering, overlap, and gaps.

## Event attribution

| Field | Meaning |
| --- | --- |
| Self CPU | Host duration excluding nested CPU operations |
| CPU total | Host range duration including children; not GPU completion time |
| Self device | Attributed device duration excluding nested device work |
| Device total | Attributed device duration including nested work |
| Count | Number of events aggregated into the row |

PyTorch may label device columns `CUDA` in rendered tables. Programmatic fields
include `self_device_time_total`, `device_time_total`, and `count` in the
inspected build. Use a device event's duration for kernel timing and an
appropriately attributed range for grouped work. Some builds emit GPU
annotations with the same names as CPU ranges. Inspect `device_type` and the
trace rather than assuming each name has exactly one or two rows.

Never sum a range and its nested kernel as independent work. A CPU range with
zero self device time can still have device work attributed elsewhere. If
CUDA events are absent, report that limitation instead of dividing CPU time
into FLOPs and labeling the result GPU throughput.

## Useful views

```python
print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=20))
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
prof.export_chrome_trace("trace.json")
```

Enable `record_shapes=True` only when shape grouping is needed, and
`with_stack=True` for stack grouping. Open the trace in Perfetto. Stream IDs
are trace-specific; do not infer the default stream from a fixed numeric ID.

## Read the critical path

- Locate the named workflow region and its correlated GPU work.
- Inspect gaps for CPU preparation, synchronization, transfers, or dependencies
  on another stream. A gap alone does not identify its cause.
- Check whether multiple kernels overlap. Use elapsed region time for workflow
  latency and kernel durations for attribution.
- Identify first-call compilation/module loading/allocation. Warm the relevant
  path before a steady-state trace; keep cold-start costs separate if needed.
- Inspect synchronizations that serialize otherwise independent work. Avoid
  adding synchronization after every operation just to simplify the trace.

For scheduled profiling, ensure `prof.step()` advances the schedule and that
the loop reaches the active window. Export from `on_trace_ready` when retaining
multiple windows; choose a unique filename per window if repeat is enabled.

## Compare implementations

Warm and validate both custom and reference paths before measurement. Match
shape, dtype, layout, output ownership, and included setup. A reference can
launch helper kernels in addition to its main compute kernel; those are part
of its operator cost when they occur in normal use.

Use [bottleneck-classification.md](bottleneck-classification.md) to choose the
next experiment. Confirm the result with a controlled benchmark; trace
instrumentation, shapes, and stacks can alter the timings.
