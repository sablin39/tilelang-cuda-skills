"""Sweep achieved streaming bandwidth against resident CTAs per SM.

The model in cost-model.md needs `bw_sm(r)`: what one SM sustains when `r`
CTAs share it. This template measures it on the attached device; no rate is
embedded. It queries the device at runtime and prints one row per point.

Usage:
    python stream_sweep.py --size-mib 512 --threads 256 --resident 1,2,4,8,16

The kernel is a plain copy (read + write), the widest streaming pattern most
kernels approach. `--size-mib` should exceed the last-level cache so the sweep
measures memory rather than cache. Report the device name and SM count printed
in the header with every constant you record.
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
    parser.add_argument("--vec", type=int, default=4, help="elements per thread (vector width)")
    parser.add_argument("--resident", default="1,2,4,8,16", help="CTAs per SM to sweep")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--dtype", default="float32")
    args = parser.parse_args()

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    sms = props.multi_processor_count
    n = args.size_mib * 1024 * 1024 // torch.finfo(getattr(torch, args.dtype)).bits * 8
    a = torch.randn(n, device="cuda", dtype=getattr(torch, args.dtype))
    b = torch.empty_like(a)
    itemsize = a.element_size()

    print(f"# device={props.name} sms={sms} size_mib={args.size_mib} dtype={args.dtype}")
    print("# resident_per_sm,grid,gib_per_s,gib_per_s_per_sm")
    for resident in (int(x) for x in args.resident.split(",")):
        grid = sms * resident
        kernel = _stream(n, grid, args.threads, args.vec, args.dtype)
        seconds = _time(lambda: kernel(a, b), args.warmup, args.iters)
        gib = 2 * n * itemsize / 1e9  # read + write, decimal GB
        print(f"{resident},{grid},{gib / seconds:.3f},{gib / seconds / sms:.3f}")


if __name__ == "__main__":
    main()
