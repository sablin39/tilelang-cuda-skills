# CUDA Debugging Tools

Keep a small reproducer with a reference, fixed inputs, checked CUDA API
returns, and a synchronization boundary that exposes asynchronous failures.
Preserve the failing shape/stride relationship when minimizing it.

## Choose the check from the failure

| Tool | Useful for | Limit |
| --- | --- | --- |
| `memcheck` | Invalid or misaligned accesses; optional leak detection | Does not validate the mathematical result |
| `racecheck` | Shared-memory access hazards | Does not establish correctness of every global-memory protocol |
| `initcheck` | Uninitialized device-memory reads | Passing does not establish correct initialization values |
| `synccheck` | Invalid synchronization usage | Does not prove an arbitrary producer/consumer protocol correct |
| `cuda-gdb` | Breakpoints, device state, and backtraces | Debug compilation and stepping can change execution behavior |

Run memcheck first for a device-memory failure, then the tool matching the
remaining symptom. Restrict expensive checks to a reproducer.

```bash
nvcc -O3 -lineinfo program.cu -o program
compute-sanitizer --tool memcheck --error-exitcode 1 \
  --show-backtrace yes --log-file memcheck.log ./program
compute-sanitizer --tool racecheck --error-exitcode 1 ./program
compute-sanitizer --tool initcheck --error-exitcode 1 ./program
compute-sanitizer --tool synccheck --error-exitcode 1 ./program
```

`--error-exitcode 1` gives automation a failure status when the application
itself succeeds but the sanitizer finds errors. It does not control whether
execution continues after an error. Keep the log and application exit status;
a timeout, missing tool, or failed launch is not a clean sanitizer result.

## Useful options

Check `compute-sanitizer --help` for the installed version. In CUDA 13.0:

| Option | Meaning |
| --- | --- |
| `--leak-check full` | Enable memcheck leak reporting |
| `--print-limit 20` | Limit printed errors; zero means unlimited |
| `--show-backtrace yes` | Request host and device backtraces; line information improves attribution |
| `--destroy-on-device-error kernel` | End the affected kernel; `context` destroys the context and is the default |
| `--track-unused-memory` | An initcheck flag for unused allocations, without a `yes` argument |

Treat the first invalid access as the first investigation target. Correlate its
source line, block/thread coordinates, address, access size, allocation bounds,
and index calculation. An out-of-bounds shared write can come from indexing or
allocation size; the memory space alone does not identify the cause.

## Device stepping and diagnostics

```bash
# Separate build for device stepping; do not benchmark this binary.
nvcc -g -G program.cu -o program_debug
cuda-gdb -batch -ex run -ex bt -ex 'info cuda threads' ./program_debug
cuda-gdb -batch -ex 'break myKernel' -ex run \
  -ex 'info cuda threads' -ex 'info locals' ./program_debug
```

Use bounded `printf` for selected values or write small intermediate buffers
for comparison with the reference. Synchronize before consuming diagnostics.
Keep required collective operations outside divergent diagnostic guards.
Instrumentation can hide timing-sensitive failures: repeat correctness and
the relevant sanitizer check after removing it.

For generated TileLang kernels, first inspect host validation, specialized
TIR, and lowering with [compiler tools](../../tilelang/debugging-tilelang-programs/references/compiler-tools.md).
Retain the reproducer, failing command, diagnosis, fix, and checks that now pass.
