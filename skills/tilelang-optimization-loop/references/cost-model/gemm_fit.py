"""Sample a complete single-CTA load/GEMM/store kernel at selected shapes.

This is not isolated MMA latency or saturated SM throughput. The measurement
includes loads, pipeline fill/drain, computation, stores and dispatch gaps
between repeated ordinary launches. Use it as a matched composition probe;
do not sum these times as per-instruction dot costs.

Usage:
    python gemm_fit.py --shapes 64x64x64,128x128x64 --threads 256 --dtype float16

Shapes are `MxNxK`; M/N must be multiples of 16 and K divisible by
`--block-k`. Output contains raw repeated-batch times, without assuming one
linear fit transfers across shapes. Reused buffers have no explicit cache
flush. Inspect the actual instruction family and resource usage before reuse.
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
    parser.add_argument("--accum", default="float32", choices=("float32",))
    parser.add_argument("--block-k", type=int, default=32)
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--batches", type=int, default=5)
    args = parser.parse_args()
    if min(args.threads, args.block_k, args.iters, args.batches) <= 0 or min(args.stages, args.warmup) < 0:
        parser.error("threads, block-k and iteration/batch counts must be positive; stages/warmup must be nonnegative")
    try:
        shapes = [_parse_shape(text) for text in args.shapes.split(",")]
    except ValueError:
        parser.error("--shapes must contain MxNxK triples of positive integers")
    for M, N, K in shapes:
        if min(M, N, K) <= 0 or M % 16 or N % 16 or K % args.block_k:
            parser.error(f"{M}x{N}x{K}: positive M/N multiples of 16 and K multiple of block-k required")

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch_dtype = _TORCH_DTYPES[args.dtype]
    print(f"# sm={props.major}{props.minor} sms={props.multi_processor_count} "
          f"dtype={args.dtype} accum={args.accum} threads={args.threads} block_k={args.block_k} stages={args.stages} "
          f"scope=whole_kernel_repeated_launch warmup={args.warmup} iters={args.iters}")
    print("# m,n,k,macs,batch,us")

    for M, N, K in shapes:
        kernel = _one_sm_gemm(M, N, K, args.threads, args.dtype, args.accum,
                              args.block_k, args.stages)
        a = torch.randn(M, K, device="cuda", dtype=torch_dtype)
        b = torch.randn(K, N, device="cuda", dtype=torch_dtype)
        c = torch.empty(M, N, device="cuda", dtype=torch.float32)
        expected = a.float() @ b.float()
        kernel(a, b, c)
        if not torch.allclose(c, expected, rtol=1e-2, atol=1e-1):
            raise SystemExit(f"{M}x{N}x{K}: result does not match torch matmul; fix the shape before timing")
        macs = M * N * K
        for batch in range(args.batches):
            us = _time(lambda: kernel(a, b, c), args.warmup, args.iters)
            print(f"{M},{N},{K},{macs},{batch},{us:.6f}")


if __name__ == "__main__":
    main()
