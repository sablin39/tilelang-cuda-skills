# IKET In-Kernel Profiling

Use IKET when a CUDA timeline identifies the kernel but cannot distinguish its
internal phases. IKET adds events to generated CUDA and the external IKET
profiler exports traces for Perfetto. It is experimental instrumentation, so
collect performance baselines without it.

## Check the environment

Treat IKET as an optional dependency. TileLang supplies the instrumentation
frontend and CUDA code-generation hook, but it does not install the external
IKET profiler or runtime. Trace collection needs all of the following:

- TileLang with CUDA support;
- a CUDA-capable NVIDIA GPU and compatible driver;
- the external IKET Python package, CLI, and runtime;
- PyTorch only when the selected example or profiling harness imports it.

Probe the same interpreter that will run the target program before changing the
kernel or the environment:

```bash
python -c "import sys, tilelang; from tilelang.tools.cuda import iket as tl_iket; print(sys.executable); print(tilelang.__version__)"
python -m iket.cli.main --help
# Run this only when the selected program imports torch:
python -c "import torch; print(torch.__version__)"
```

If a required probe fails, report which component is missing, show the active
interpreter or environment name, and ask the user whether to install the
dependency into that environment. Do not run `pip`, `uv`, `conda`, a system
package manager, or an IKET installer until the user approves that environment
change. If the user declines, continue with generated-source inspection,
TileLang timing, Nsight, or compiler-lifetime analysis as appropriate.

The reviewed TileLang documentation names the external package and CLI but does
not specify a distribution URL or a supported IKET version. In the reviewed
`qla` environment, `import iket` resolves from the `nvidia-cutlass-dsl` package,
which ships `iket.cli.main` and a `profiler/` directory; this suite's
`requirements.txt` carries that package on this basis. It remains
version-unpinned, so treat the CLI probe above as the acceptance check in
whichever interpreter will run the target program, and do not assume that an
unrelated package named `iket` from a package index is the profiler. If the
probe fails, ask for the project's intended IKET source/version, or follow
installation instructions already present in the user's environment, before
proposing an exact install command. After installation, rerun the CLI probe
above.

IKET is a CUDA tool rather than a TileLang language namespace:

```python
import tilelang
import tilelang.language as T
from tilelang.tools.cuda import iket
```

## Instrument and compile in one session

```python
def instrumented_add(n: int, threads: int = 128):
    @T.prim_func
    def main(
        A: T.Tensor((n,), T.float32),
        B: T.Tensor((n,), T.float32),
        C: T.Tensor((n,), T.float32),
    ):
        with T.Kernel(T.ceildiv(n, threads), threads=threads) as bx, \
             iket.range("block_total"):
            for tx in T.Parallel(threads):
                i = bx * threads + tx
                if i < n:
                    iket.mark("before_store")
                    C[i] = A[i] + B[i]
                    iket.mark("after_store")

    return main


with iket.session(output_dir="/tmp/tilelang_iket"):
    kernel = tilelang.compile(
        instrumented_add(1024),
        out_idx=-1,
        target="cuda",
        execution_backend="cython",
    )
```

The session must be active while CUDA source is generated. Constructing the
PrimFunc inside it resets frontend event allocation cleanly, although event
metadata embedded in a prebuilt PrimFunc survives a later registry reset.
`iket.session()` disables TileLang's kernel cache by default because callback
activation and payload mode are host-side compilation state. Leave that default
unless the caller controls how every reused binary was compiled.

Use `iket.range(name)` for a lexical range and `iket.mark(name)` for an instant
event. `range_push`/`range_pop` and their `range_start`/`range_end` aliases are
available when a context manager is unsuitable. Ranges are recorded per warp,
so one block can emit several ranges for one lexical scope. Event names are
limited to 32 UTF-8 bytes.

## Capture scalar payloads deliberately

Markers and range starts accept one `int32`, `uint32`, or `float32` value:

```python
iket.mark("store_index", payload=iket.payload(i, dtype="int32"))
```

Compile with runtime payloads enabled:

```python
with iket.session(output_dir="/tmp/tilelang_iket", runtime_payloads=True):
    kernel = tilelang.compile(program, target="cuda")
```

Without `runtime_payloads=True`, the payload schema remains in TIR metadata but
the generated event writes no runtime value. A captured value reflects IKET's
warp-level dump selection; it is not one value for every lane. Reusing a name
with incompatible payload dtypes in one frontend registry is rejected.

## Collect and inspect a trace

The TileLang helper can construct the current external-profiler command:

```python
command = iket.profile_command(
    ["python", "profile_target.py", "--iket-runtime-payloads"],
    directory="/tmp/tilelang_iket_profile",
)
print(command)
```

It returns a quoted string and does not execute it. The equivalent command is:

```bash
python -m iket.cli.main \
  --output-dir /tmp/tilelang_iket_profile \
  --clobber \
  profile \
  --postprocess all \
  -- \
  python profile_target.py --iket-runtime-payloads
```

Keep the profiler's output directory and the program's IKET output directory
the same. Depending on postprocessing, it can contain `.pftrace`, compressed
trace, JSON, and HTML files. Serve the directory so generated HTML can load its
adjacent trace:

```bash
cd /tmp/tilelang_iket_profile
python3 -m http.server 8080
```

Open the exact generated HTML filename. If it only opens the Perfetto landing
page, import the matching `.pftrace` manually. `iket.trace_files()` returns only
`.trace.json` files, ordered largest first; it does not collect or convert a
trace.

## Preserve the evidence boundary

`iket.event_table()` inspects recently registered frontend events, but CUDA
codegen recovers the authoritative event metadata from TIR. The callback,
runtime-payload mode, frontend registry, and cache switch are process-global.
Use scoped sessions and coordinate concurrent compilation.

Report the capture command, target, instrumentation sites, payload mode, exact
trace path, and relevant warp/range events. IKET supports only the TileLang CUDA
backend. It does not provide source-location tables, tensor lifetimes, or a
replacement for uninstrumented latency measurements.
