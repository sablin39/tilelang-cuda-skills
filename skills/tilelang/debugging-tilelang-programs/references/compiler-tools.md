# Inspecting TIR, Lowering, and Generated Source

Use the first boundary that can answer the question. Source generation does
not establish runtime correctness or performance. These workflows follow the
matching checkout's `docs/tools/` and `docs/compiler_internals/tensor_checks.md`.

The reviewed TileLang 0.1.14 checkout bundles TVM 0.25.dev0. Its TIR APIs are
split across `tvm.tirx` and `tvm.s_tir`; confirm those namespaces before adapting
examples written for a different TVM release.

## Emit source without a GPU

Create a file containing a kernel definition with all specialization parameters
bound or defaulted. Guard host allocation and launches under
`if __name__ == "__main__"` because the tool imports the file.

```bash
python -I -m tilelang.tools.compile_only \
  --target cuda --output_file /tmp/kernel.cu kernel_definition.py
```

The tool selects the first JIT function or PrimFunc in module order and stops
after lowering with device compilation disabled. It does not call nvcc or
launch a kernel. Use an explicit target; `auto` is rejected. Bare `cuda` uses
a pinned target for inspection, which need not be the deployment target. The
default `c` target emits C source; CUDA emission requires CUDA codegen support
in the installed build, even though no GPU/driver is needed.

The output path must be distinct from the input. A failed run removes stale
output at that destination. Use a dedicated artifact path. Isolated Python mode
does not prevent the input file's top-level code from executing.

Programmatic equivalent:

```python
from tilelang.tools.compile_only import compile_kernel_source
source = compile_kernel_source(factory.get_tir(...), target="cuda")
```

## Trace a real lowering pipeline

```bash
TL_LOWER_TRACE=html TL_LOWER_TRACE_DIR=/tmp/tilelang-lowering \
  TILELANG_DISABLE_CACHE=1 python repro.py
```

Use a fresh process/cache miss so lowering actually runs. `terminal`, `html`,
and `both` select report modes. `TL_LOWER_TRACE=1` selects HTML. Trace artifacts
are grouped by script and run; `report.html` points to the latest report.

For inspection without editable code injection:

```python
from tilelang.tools import lower_trace as lt
lt.enable(mode="html", trace_dir="/tmp/tilelang-lowering", codegen_output=None)
try:
    kernel = factory(...)
finally:
    lt.disable()
```

The report shows before/after TIR, pass ordering, failures, and final codegen.
Find the first unexpected transformation and compare its input and output.
A textual diff is evidence to inspect, not proof of changed semantics.
Prefer this workflow to the superseded `TILELANG_PASS_DIFF` hook.

For a selected transformation rather than a full compile:

```python
records = lt.lower_trace(tir, tilelang.transform.Simplify(), mode="terminal")
```

When line diffs obscure a structural change, use the current CUDA Pass
Visualizer:

```bash
python -m tilelang.tools.pass_visualizer.viewer \
  kernel_definition.py --out /tmp/kernel-passes.html
```

The input must expose a `@tilelang.jit` factory. Bind required factory
arguments with repeated `--set name=value` options, and use `--factory name`
when the file contains more than one candidate. The tool emits a self-contained
HTML structure-tree view and a sibling text dump. It follows the focused CUDA
lowering prologue; use Lower Trace when the full backend pipeline or final
codegen is the question.

## Editing generated source

The default HTML workflow also maintains `codegen.cpp`, `.original`, and
`.latest`. `.original` is the baseline; `.latest` is current generated output;
the working copy can contain edits that are reinjected on a compatible backend.
Do not edit the baseline. If both the program and working copy changed, inspect
the reported conflict and backup before reapplying edits.

For CUDA edit-and-recompile experiments, use a source-compiling backend such
as `execution_backend="nvrtc"` or `"cython"`. With `tvm_ffi`, edits can appear
in the report without recompiling the device binary. Confirm the trace's
`PATCHED`/backend messages and verify the resulting behavior. Move a successful
algorithmic fix back into the TileLang source and validate it there.

## Host checks and layout inspection

`kernel.get_host_source()` exposes argument checks and launch code for the
selected backend. Inspect mismatched dtype, rank, shape, strides, device, or
symbol binding before changing the computation. A host rejection is different
from an out-of-bounds device access.

For inferred thread/data mappings, compile with:

```python
pass_configs = {
    tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE: True,
    tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_FORMATS: "txt",
}
```

To inspect a layout you constructed:

```python
import tilelang.language as T
from tilelang.tools import plot_layout
layout = T.Layout([4, 4], lambda i, j: (j, i))
plot_layout(layout, save_directory="/tmp/layouts", name="transpose", formats="svg")
```

Text output needs no plotting library; image output requires Matplotlib.
The suite's `requirements.txt` records this extra dependency; PyTorch and
TileLang are assumed to be installed already.
Large or non-bijective layouts can produce misleading/expensive plots, so
use small mappings and corroborate them with indexing and correctness checks.

For access intervals, allocation scopes, and shared-memory packing, read
[buffer-lifetime-analysis.md](buffer-lifetime-analysis.md).

## Pass instruments and compiler checks

For whole-kernel lowering, prefer TileLang's integrated instruments:

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ENABLE_DUMP_IR: True,
    tilelang.PassConfigKey.TL_DUMP_IR_DIR: "/tmp/tilelang-ir",
    tilelang.PassConfigKey.TL_PASS_PROFILE: True,
    tilelang.PassConfigKey.TL_PASS_PROFILE_THRESHOLD_MS: 1.0,
}
```

`TL_ENABLE_DUMP_IR` attaches the bundled `tvm.ir.instrument.DumpIR` to the JIT
lowering context. Pass profiling reports inclusive and self time after the
compilation; `TILELANG_PASS_PROFILE=1` and
`TILELANG_PASS_PROFILE_THRESHOLD_MS=<ms>` provide the environment form. These
are compiler timings and say nothing about kernel runtime.

For a selected pass or custom pipeline, the bundled `tvm.ir.instrument` module
also supplies `PassInstrument`, `PassTimingInstrument`,
`PassPrintingInstrument`, `PrintBeforeAll`, `PrintAfterAll`, and `DumpIR`.
Attach them to the `tvm.transform.PassContext` that actually runs the pass.
TileLang's own JIT constructs its lowering context, so a surrounding generic
PassContext is not a substitute for the integrated flags or TileLang's
instrumentation lifecycle.

Use the checks that answer the suspected failure:

- `tvm.tirx.analysis.verify_ssa`, `verify_well_formed`, and `undefined_vars`
  inspect IR invariants.
- `tvm.s_tir.analysis.OOBChecker()` performs static out-of-bounds analysis;
  `tvm.s_tir.transform.InstrumentBoundCheckers()` changes generated execution
  and is a separate runtime-instrumentation pass.
- `tvm.s_tir.analysis.verify_gpu_code(func, constraints)` checks specified GPU
  resource constraints. It does not discover the device limits for the caller.
- TileLang's `VerifyBufferInit` is enabled in the normal pipeline and warns on
  non-global buffers read before a write. Do not disable it to hide a warning.
- `TILELANG_ENABLE_DATA_RACE_CHECK=1` opts into TileLang's static parallel-loop
  race check. It is disabled by default because address independence can be
  difficult to prove and false positives are possible.
- `tvm.arith.Analyzer` simplifies index expressions, derives integer sets and
  constant bounds, and proves conditions. Failure to prove safety is not proof
  of an out-of-bounds access.

Current eager parsing preserves source spans by default. For CUDA source
correlation, `PassConfigKey.TL_EMIT_LINE_DIRECTIVES: True` emits `#line`
directives when the installed build supports the key; the compile-only tool
enables it automatically. Keep the source file available when consuming those
locations.

## Logs and minimization

`TVM_LOG_DEBUG=1` selects runtime debug logging only for statements present
in the compiled build; release builds may omit DLOG/DCHECK. Lowering traces
often provide useful pass evidence without rebuilding TileLang.

Use `TILELANG_JIT_DIAGNOSTICS=1` for start, completion, failure, and elapsed
time of TileLang's JIT lowering and adapter phases. For DLOG/VLOG, a debug-enabled
native build is still required; `TVM_LOG_DEBUG=1` enables DLOG, while a
comma-separated `file.cc=level` specification selects VLOG levels. Ordinary
`LOG`, Python logging, JIT diagnostics, and pass reports have separate gates.

Use AutoDD only after obtaining a stable, distinctive error signature. Preserve
essential setup and synchronization with freeze annotations, use one worker
initially, and re-run the minimized file under the same environment.

Upstream sources: [compile-only](https://github.com/tile-ai/tilelang/blob/main/docs/tools/compile_only.md),
[lower trace](https://github.com/tile-ai/tilelang/blob/main/docs/tools/lower_trace.md),
[layout visualization](https://github.com/tile-ai/tilelang/blob/main/docs/tools/layout_visualization.md),
and [AutoDD](https://github.com/tile-ai/tilelang/blob/main/docs/tools/autodd.md).
