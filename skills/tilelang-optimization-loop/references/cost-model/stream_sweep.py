"""Sweep whole-device copy bandwidth against offered grid work.

Grid CTAs divided by SM count is not measured residency. This template does
not measure per-SM bandwidth or guarantee a DRAM-only path. It prints raw
repeated-batch kernel times; inspect cache behavior and compiled resources
before fitting a model.

Usage:
    python stream_sweep.py --size-mib 512 --threads 256 --ctas-per-sm 1,2,4,8,16

The kernel is a plain copy (read + write). Each buffer's `--size-mib` should
exceed the last-level cache for a DRAM experiment; verify the resulting cache
regime. Timing includes repeated ordinary launches and can include dispatch
gaps. Retain the command, device/software identity and cache/clock policy.
"""

import argparse

import torch

import tilelang
import tilelang.language as T


@tilelang.jit
def _stream(n: int, grid: int, threads: int, vec: int, dtype: str = "float32"):
    step = threads * vec
    per_cta = (n + grid - 1) // grid
    chunks = max(1, (per_cta + step - 1) // step)

    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype)):
        with T.Kernel(grid, threads=threads) as bx:
            for c in T.serial(chunks):
                for i in T.Parallel(step):
                    idx = bx * chunks * step + c * step + i
                    if idx < n:
                        B[idx] = A[idx]

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
    return start.elapsed_time(end) / iters / 1e3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size-mib", type=int, default=512)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--vec", type=int, default=4, help="elements per thread in a chunk; inspect emitted vectorization")
    parser.add_argument("--ctas-per-sm", default="1,2,4,8,16", help="grid CTAs / SM count, not measured residency")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--dtype", default="float32", choices=("float16", "bfloat16", "float32"))
    args = parser.parse_args()
    try:
        offered = [int(x) for x in args.ctas_per_sm.split(",")]
    except ValueError:
        parser.error("--ctas-per-sm must be comma-separated positive integers")
    if min(args.size_mib, args.threads, args.vec, args.iters, args.batches, *offered) <= 0 or args.warmup < 0:
        parser.error("sizes, grid ratios and iteration/batch counts must be positive; warmup must be nonnegative")

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    sms = props.multi_processor_count
    n = args.size_mib * 1024 * 1024 // (torch.finfo(getattr(torch, args.dtype)).bits // 8)
    a = torch.randn(n, device="cuda", dtype=getattr(torch, args.dtype))
    b = torch.empty_like(a)
    itemsize = a.element_size()

    print(f"# sm={props.major}{props.minor} sms={sms} size_mib_per_buffer={args.size_mib} dtype={args.dtype} "
          f"scope=whole_kernel_repeated_launch warmup={args.warmup} iters={args.iters}")
    print("# grid_ctas_per_sm,grid,batch,seconds,gb_per_s")
    for grid_ratio in offered:
        grid = sms * grid_ratio
        kernel = _stream(n, grid, args.threads, args.vec, args.dtype)
        kernel(a, b)
        if not torch.equal(a, b):
            raise SystemExit(f"grid={grid}: copy correctness check failed")
        gb = 2 * n * itemsize / 1e9  # read + write, decimal GB
        for batch in range(args.batches):
            seconds = _time(lambda: kernel(a, b), args.warmup, args.iters)
            print(f"{grid_ratio},{grid},{batch},{seconds:.9g},{gb / seconds:.3f}")


if __name__ == "__main__":
    main()
