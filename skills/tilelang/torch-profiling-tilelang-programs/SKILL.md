---
name: torch-profiling-tilelang-programs
description: Inspect a PyTorch baseline with torch.compile graphs/code and torch.profiler before drafting a custom kernel, then compare TileLang and PyTorch timelines, forward/backward regions, and allocations. Use for baseline analysis and attribution; use do_bench for controlled latency comparisons.
---

# Profiling TileLang Workflows with PyTorch

Use `torch.profiler` to attribute CPU/device work and inspect scheduling,
copies, allocations, and synchronization. It does not directly establish an
instruction-level bottleneck. Verify CUDA profiling support in the installed
build; a CPU-only trace is insufficient for GPU timing.

## Understand the baseline before drafting

When a PyTorch implementation is available, start with
[torch-compile-inspection.md](references/torch-compile-inspection.md). Validate
its eager/compiled paths, inspect graphs and generated code, and profile the
warmed operation. Identify a concrete region, layout contract, and cost before
choosing the custom kernel's scope. Keep eager and compiled timing results
separate and retain the existing PyTorch implementation as the reference.

## Profile a warmed workflow

Adapt [scripts/profile_template.py](scripts/profile_template.py). It validates
the example, warms both paths, labels them, and exports a trace. For your own
operation:

```python
import torch
from torch.profiler import profile, ProfilerActivity, record_function

for _ in range(10):
    custom_call()
    reference_call()
torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(20):
        with record_function("custom"):
            custom_call()
        with record_function("reference"):
            reference_call()
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=20))
prof.export_chrome_trace("trace.json")
```

Open the trace in Perfetto (`https://ui.perfetto.dev`). Read
[trace-reading-guide.md](references/trace-reading-guide.md) for event
attribution, overlapping work, and comparison pitfalls. A range and its nested
kernel can describe the same work; do not add both into an operation total.

Keep `record_shapes` and `with_stack` off unless needed for grouping or memory
analysis. Final performance comparisons belong in
[controlled benchmarks](../profiling-tilelang-programs/SKILL.md), because
profiling changes execution costs.

## Training or multi-step workflows

```python
from torch.profiler import schedule

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=lambda p: p.export_chrome_trace("training_trace.json"),
) as prof:
    for step in range(5):
        train_step()
        prof.step()
```

Place `record_function` ranges around forward, backward, and optimizer work
when comparing their contribution. Keep synchronization representative of the
real workflow; do not insert a device-wide sync after every operation merely
to simplify attribution.

## Allocation inspection

For a profiler memory timeline, all three options are required:

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True, record_shapes=True, with_stack=True,
) as prof:
    forward_pass()
    torch.cuda.synchronize()
prof.export_memory_timeline("memory.json", device="cuda:0")
```

JSON export needs no plotting library; HTML export additionally uses Matplotlib.
Timeline export is deprecated in the inspected PyTorch build. For allocator
history, use the API actually present in the selected environment:

```python
torch.cuda.memory._record_memory_history(max_entries=100_000)
try:
    forward_pass()
    torch.cuda.synchronize()
    torch.cuda.memory._dump_snapshot("snapshot.pickle")
finally:
    torch.cuda.memory._record_memory_history(enabled=None)
```

These are private APIs; inspect their signatures when changing PyTorch versions.
View snapshots at `https://pytorch.org/memory_viz`. They describe the PyTorch
allocator, so external allocations may be absent. Limiting allocator capacity
does not freeze allocation behavior or replace warmup.

## Move from observation to a change

Use [bottleneck-classification.md](references/bottleneck-classification.md) to
turn trace observations into hypotheses and a controlled experiment. Follow
unexpected launches or copies back to host/device source and lowering traces.
Use nsys for a broader system timeline or ncu for a selected kernel's diagnostic
sections. Optional in-kernel instrumentation is described in TileLang's
[IKET workflow](../profiling-tilelang-programs/references/iket-profiling.md); it
requires an external runtime and should only be used when ordinary traces cannot
answer the question.

## Completion evidence

Return the trace/snapshot path, capture command and scope, selected ranges and
device events, and the observation that motivates the next experiment. If GPU
events are absent, report the missing attribution. Remeasure a proposed change
with the same controlled benchmark before claiming a performance improvement.
