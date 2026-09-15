"""Reusable torch.profiler template for TileLang kernels.

Copy this file next to your kernel script and adapt the marked sections.
Run with:  CUDA_VISIBLE_DEVICES=<id> python profile_template.py
Outputs:   ./trace.json (open in chrome://tracing or https://ui.perfetto.dev/)
           A summary table is printed to stdout.

For long-running models, replace the simple `with profile(...)` block with the
scheduled variant in SKILL.md.
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

    torch.testing.assert_close(kernel(a, b), a @ b, rtol=1e-2, atol=1e-2)

    # --- Warm both paths before collecting steady-state work ---
    for _ in range(10):
        _ = kernel(a, b)
        _ = a @ b
    torch.cuda.synchronize()

    # --- Profile ---
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,       # enable only for shape grouping
        with_stack=False,          # set True if you also want Python stacks (more overhead)
        profile_memory=False,      # timeline also needs record_shapes and with_stack
    ) as prof:
        for _ in range(20):
            with record_function("tilelang_matmul"):
                c = kernel(a, b)
            # If comparing to a reference, wrap it in its own record_function:
            with record_function("torch_matmul"):
                ref = a @ b
        torch.cuda.synchronize()

    # --- Aggregated table (sort by GPU self-time) ---
    print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=20))

    # --- Save chrome trace for visual inspection ---
    prof.export_chrome_trace("trace.json")
    print("Wrote trace.json (open in chrome://tracing or https://ui.perfetto.dev/)")

    # --- Derived throughput from attributed GPU time (diagnostic, not a benchmark) ---
    # Prefer a GPU annotation when available; otherwise use attributed CPU
    # range totals. Never add the annotation and its nested kernel together.
    def lookup_gpu_us_per_call(name):
        events = [evt for evt in prof.key_averages() if evt.key == name and evt.count]
        for evt in events:
            if evt.device_type == torch.autograd.DeviceType.CUDA and evt.self_device_time_total > 0:
                return evt.self_device_time_total / evt.count
        for evt in events:
            if evt.device_time_total > 0:
                return evt.device_time_total / evt.count
        return None

    for label, flops in (("tilelang_matmul", 2 * M * N * K),
                          ("torch_matmul", 2 * M * N * K)):
        us = lookup_gpu_us_per_call(label)
        if us:
            tflops = flops / (us * 1e-6) / 1e12
            print(f"{label}: {us/1e3:.3f} ms/call, {tflops:.1f} TFLOPS")
        else:
            print(f"{label}: no attributed GPU timing; inspect CUDA profiler support and trace")


if __name__ == "__main__":
    main()
