# NVTX Regions for Kernel Workflows

NVTX labels host work so a timeline can connect application phases with CUDA
API calls and device launches. A host range around an asynchronous launch can
end before its device kernel finishes. Read the correlated GPU events or use
CUDA events for device latency.

## Minimal C++ integration

NVTX 3 is header-only. Include `nvtx3/nvToolsExt.h`; add its include path and
platform linker dependencies such as `-ldl` where required. Do not add the
legacy `-lnvToolsExt` library merely to use NVTX 3.

```cpp
#include <nvtx3/nvToolsExt.h>

class NvtxRange {
public:
    explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
    ~NvtxRange() { nvtxRangePop(); }
    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};

// Inside the application's existing host workflow:
// {
//     NvtxRange range("custom.forward");
//     launch_forward(...);  // Keep the normal stream/synchronization behavior.
// }
```

Push/pop ranges belong to the same host thread and must be balanced. Scope
guards preserve balance across returns and exceptions. Use stable phase names
such as `custom.forward`, `custom.backward`, and `reference.forward`, with
nested ranges only when they clarify a specific investigation.

## Collect and interpret

```bash
nsys profile --trace=cuda,nvtx --sample=none -o timeline ./program
nsys stats --report nvtx_sum timeline.nsys-rep
nsys stats --report cuda_gpu_kern_sum timeline.nsys-rep
```

Warm both paths before the measured region. Inspect the trace to associate
ranges with launches; do not add a parent range and its children as independent
work. Avoid inserting `cudaDeviceSynchronize()` after every launch to make
range lengths resemble GPU time: that changes scheduling and overlap. If a
synchronized end-to-end region is the intended measurement, document it and
apply the same boundary to every compared implementation.

Use `nvtxMarkA("checkpoint")` for an instant event. Avoid per-element marks
and unbounded dynamic names. Remove or disable diagnostic instrumentation for
the final controlled performance comparison.
