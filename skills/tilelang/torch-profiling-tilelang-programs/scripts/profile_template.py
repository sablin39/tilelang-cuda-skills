"""Reusable torch.profiler template for TileLang kernels.

Copy this file next to your kernel script and adapt the marked sections.
Run with:  CUDA_VISIBLE_DEVICES=<id> python profile_template.py
Outputs:   ./trace.json (open in chrome://tracing or https://ui.perfetto.dev/)
           A summary table is printed to stdout.

For long-running models, replace the simple `with profile(...)` block with the
scheduled variant near the bottom of this file.
"""

import torch
from torch.profiler import profile, ProfilerActivity, record_function

import tilelang
import tilelang.language as T


# ---------------------------------------------------------------------------
# 1. Define the kernel under study.  Replace this with your own.
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def gemm(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])
    return gemm


def main():
    M, N, K = 4096, 4096, 4096
    kernel = matmul(M, N, K, 128, 128, 32)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    # --- Warm up so JIT compile and autotune costs do not contaminate the trace ---
    for _ in range(5):
        _ = kernel(a, b)
    torch.cuda.synchronize()

    # --- Profile ---
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,        # input shapes per op
        with_stack=False,          # set True if you also want Python stacks (more overhead)
        profile_memory=False,      # set True for allocator timeline
    ) as prof:
        for _ in range(20):
            with record_function("tilelang_matmul"):
                c = kernel(a, b)
            # If comparing to a reference, wrap it in its own record_function:
            with record_function("torch_matmul"):
                ref = a @ b
        torch.cuda.synchronize()

    # --- Aggregated table (sort by GPU self-time) ---
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))

    # --- Save chrome trace for visual inspection ---
    prof.export_chrome_trace("trace.json")
    print("Wrote trace.json (open in chrome://tracing or https://ui.perfetto.dev/)")

    # --- Roofline math: turn the measured kernel time into achieved TFLOPS ---
    # record_function ranges appear twice in key_averages: once as the CPU range
    # (self_device_time_total == 0) and once as the GPU range that owns the
    # kernel time (self_device_time_total > 0).  Pick the GPU-side one.
    def lookup_gpu_us_per_call(name):
        for evt in prof.key_averages():
            if evt.key == name and evt.self_device_time_total > 0 and evt.count:
                return evt.self_device_time_total / evt.count  # microseconds
        return None

    for label, flops in (("tilelang_matmul", 2 * M * N * K),
                          ("torch_matmul", 2 * M * N * K)):
        us = lookup_gpu_us_per_call(label)
        if us:
            tflops = flops / (us * 1e-6) / 1e12
            print(f"{label}: {us/1e3:.3f} ms/call, {tflops:.1f} TFLOPS")


# ---------------------------------------------------------------------------
# Scheduled variant: for training loops or multi-step models, prefer this so
# the profiler only captures a stable window.  Cycle: skip 1, warmup 1, record 3.
# ---------------------------------------------------------------------------
def main_scheduled():
    # ... build kernel/inputs as above ...
    pass  # left as a stub on purpose, see SKILL.md "Scheduled profiling" section

if __name__ == "__main__":
    main()
