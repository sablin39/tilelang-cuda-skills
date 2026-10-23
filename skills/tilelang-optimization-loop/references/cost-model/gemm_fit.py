"""Time one barrier-separated GEMM on one SM at the shapes you will use.

The model in cost-model.md prices a stage's dots as a sum of per-shape costs,
never as one merged call. This template measures those costs on the attached
device; it embeds no coefficient. One CTA runs the whole GEMM, so the reported
time is one dot's time.

Usage:
    python gemm_fit.py --shapes 64x64x64,128x128x64 --threads 256 --dtype float16

Shapes are `MxNxK` and must be multiples of 16 with `K` divisible by
`--block-k`. With two or more shapes the script also prints a least-squares
line through them; use it only inside the measured shape range, and prefer the
per-shape numbers when they do not fit a line.
"""

import argparse

import torch

import tilelang
import tilelang.language as T

_TORCH_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@tilelang.jit
def _one_sm_gemm(M: int, N: int, K: int, threads: int, dtype: str, accum: str,
                 block_k: int, stages: int):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum),
    ):
        with T.Kernel(1, threads=threads):
            A_s = T.alloc_shared((M, block_k), dtype)
            B_s = T.alloc_shared((block_k, N), dtype)
            C_l = T.alloc_fragment((M, N), accum)
            T.clear(C_l)
            for ko in T.Pipelined(T.ceildiv(K, block_k), num_stages=stages):
                T.copy(A[0, ko * block_k], A_s)
                T.copy(B[ko * block_k, 0], B_s)
                T.gemm(A_s, B_s, C_l)
            T.copy(C_l, C[0, 0])

    return main


def _time(call, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        call()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1e3  # microseconds


def _parse_shape(text: str) -> tuple[int, int, int]:
    m, n, k = (int(x) for x in text.lower().split("x"))
    return m, n, k


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", required=True, help="MxNxK[,MxNxK...]")
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--dtype", default="float16", choices=sorted(_TORCH_DTYPES))
    parser.add_argument("--accum", default="float32")
    parser.add_argument("--block-k", type=int, default=32)
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    torch_dtype = _TORCH_DTYPES[args.dtype]
    print(f"# device={props.name} sms={props.multi_processor_count} "
          f"dtype={args.dtype} threads={args.threads}")
    print("# m,n,k,macs,us")

    points = []
    for text in args.shapes.split(","):
        M, N, K = _parse_shape(text)
        if M % 16 or N % 16 or K % args.block_k:
            raise SystemExit(f"{text}: M and N must be multiples of 16, K of block-k")
        kernel = _one_sm_gemm(M, N, K, args.threads, args.dtype, args.accum,
                              args.block_k, args.stages)
        a = torch.randn(M, K, device="cuda", dtype=torch_dtype)
        b = torch.randn(K, N, device="cuda", dtype=torch_dtype)
        c = torch.empty(M, N, device="cuda", dtype=torch.float32)
        expected = (a.float() @ b.float()).to(torch_dtype)
        kernel(a, b, c)
        if not torch.allclose(c, expected.float(), rtol=1e-2, atol=1e-1):
            raise SystemExit(f"{text}: result does not match torch matmul; fix the shape before timing")
        us = _time(lambda: kernel(a, b, c), args.warmup, args.iters)
        macs = M * N * K
        points.append((macs, us))
        print(f"{M},{N},{K},{macs},{us:.3f}")

    if len(points) >= 2:
        n = len(points)
        sum_x = sum(x for x, _ in points)
        sum_y = sum(y for _, y in points)
        sum_xx = sum(x * x for x, _ in points)
        sum_xy = sum(x * y for x, y in points)
        denom = n * sum_xx - sum_x * sum_x
        slope = 0.0 if denom == 0 else (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n
        print(f"# local fit: us = {intercept:.3f} + {slope:.6f} * macs "
              f"(in-range only; per-shape numbers win when they disagree)")


if __name__ == "__main__":
    main()
