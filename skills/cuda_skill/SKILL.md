---
name: cuda
description: Draft, debug, and measure native CUDA C++ kernels and host launch workflows. Use for nvcc compilation, launch contracts, compute-sanitizer, Nsight timelines, focused kernel profiling, and generated-code inspection of hand-written CUDA; for TileLang programs use the TileLang debugging/profiling skills instead.
---

# CUDA Development Workflow

Record the active toolchain and the operation's shapes, dtypes, strides,
ownership, stream, and output initialization contract. Choose the compilation
target for the deployment environment. Start with a reference and a small
reproducer that checks results and CUDA API errors.

## Draft and validate

Separate host setup/allocation from the kernel's indexing and computation.
Make bounds, output ownership, and synchronization explicit. An asynchronous
launch returning successfully does not establish that execution succeeded;
synchronize at a meaningful validation boundary and check its error.

```bash
# Build with source correlation for profiling and sanitizer reports.
nvcc -O3 -lineinfo program.cu -o program
# Use device debug compilation only when stepping through device code.
nvcc -g -G program.cu -o program_debug
```

Validate tails and repeated calls as well as the main shape. Confirm resets for
outputs that accumulate. Use representative precision and tolerances.

## Debug the failing layer

```bash
compute-sanitizer --tool memcheck ./program
compute-sanitizer --tool racecheck ./program
compute-sanitizer --tool initcheck ./program
compute-sanitizer --tool synccheck ./program
cuda-gdb -batch -ex run -ex bt ./program_debug
```

Select the tool that matches the symptom. Memcheck reports accesses, initcheck
uninitialized reads, racecheck shared-memory hazards, and synccheck invalid
synchronization. Racecheck is not a general proof that all global-memory
communication is correct. Bounded device `printf` can help inspect values;
flush it through synchronization before reading the output. Instrumentation
can change timing, so rerun without it after fixing the problem.

For tool options, see [debugging-tools.md](references/debugging-tools.md).
For generated TileLang code, use the TileLang debugging skill's
[lowering inspection workflow](../tilelang/debugging-tilelang-programs/references/compiler-tools.md).

## Measure before and after a change

Warm all compared paths, keep initialization and allocation costs consistent,
and distinguish kernel timing from complete operation latency. Use CUDA events
for stream timing and synchronized wall time for the full host/device workflow.

```bash
nsys profile --trace=cuda,nvtx -o timeline ./program
nsys stats --report cuda_gpu_kern_sum timeline.nsys-rep
ncu --list-sets
ncu --list-sections
ncu --kernel-name 'regex:.*myKernel.*' --launch-skip 10 --launch-count 1 \
    --set basic -o kernel_report ./program
```

Use nsys to select the important region and ncu to inspect a filtered kernel.
Profiler collection overhead is not the baseline latency. Use available
section identifiers from the installed version; see
[ncu-guide.md](references/ncu-guide.md) and [nsys-guide.md](references/nsys-guide.md).

Label host regions with NVTX when they help connect application work to the
timeline. NVTX 3 uses `#include <nvtx3/nvToolsExt.h>` and is header-only; ensure
the headers and platform linker dependencies are available. See
[nvtx-patterns.md](references/nvtx-patterns.md).

Form a hypothesis from the trace or source, change the relevant code, validate
outputs, and repeat the same measurement. Keep configuration and results with
the experiment. Architecture and instruction-level tuning are separate,
optional follow-up work.

## Versioned documentation snapshots

[references/cuda-driver.md](references/cuda-driver.md),
[references/cuda-runtime.md](references/cuda-runtime.md), and
[references/ptx-isa.md](references/ptx-isa.md) index local snapshots of the
CUDA 13.1 driver/runtime APIs and PTX ISA 9.1. They are newer than the reviewed
CUDA 13.0 toolkit: confirm an API or instruction exists in the installed
toolchain before relying on a snapshot-only entry.

## Completion evidence

Return source/build commands, launch and input/output contracts, reference
comparison results, and any sanitizer logs. For performance changes, retain
the correct baseline and record timing scope, conditions, and before/after
results. For a task involving TileLang, carry those artifacts into the suite's
[optimization loop](../tilelang-optimization-loop/SKILL.md).
