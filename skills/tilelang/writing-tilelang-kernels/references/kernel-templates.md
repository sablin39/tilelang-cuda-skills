# TileLang Kernel Templates

Complete kernel templates using the PrimFunc-factory style.
Each template includes imports, kernel definition, host code, validation, and profiling.

---

## 1. 1D Elementwise -- Vector Scale

Multiply every element of a vector by a scalar constant.

```python
import torch
import tilelang
import tilelang.language as T

# ---------- kernel definition ----------
@tilelang.jit(out_idx=[-1])
def vector_scale(N, block_size, scale_val, dtype=T.float16):
    @T.prim_func
    def kernel(
        X: T.Tensor((N,), dtype),
        Y: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_size), threads=128) as (bx,):
            X_shared = T.alloc_shared((block_size,), dtype)
            Y_local = T.alloc_fragment((block_size,), dtype)

            T.copy(X[bx * block_size], X_shared)
            for i in T.Parallel(block_size):
                Y_local[i] = X_shared[i] * T.cast(scale_val, dtype)
            T.copy(Y_local, Y[bx * block_size])
    return kernel

# ---------- parameters ----------
N = 4096
block_size = 256
scale_val = 2.0

# ---------- host code ----------
x = torch.randn(N, dtype=torch.float16, device="cuda")
kernel = vector_scale(N, block_size, scale_val)
y = kernel(x)

# ---------- validation ----------
ref = x * scale_val
torch.testing.assert_close(y, ref, rtol=1e-2, atol=1e-2)
print("vector_scale: PASSED")

# ---------- profiling ----------
profiler = kernel.get_profiler()
latency = profiler.do_bench()
print(f"vector_scale latency: {latency:.4f} ms")
```

---

## 2. 2D Elementwise -- Softplus

Compute `softplus(x) = log(1 + exp(x))` over a 2D tensor.

```python
import torch
import tilelang
import tilelang.language as T

# ---------- kernel definition ----------
@tilelang.jit(out_idx=[-1])
def softplus(M, N, block_M, block_N, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def kernel(
        X: T.Tensor((M, N), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            X_shared = T.alloc_shared((block_M, block_N), dtype)
            Y_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.copy(X[by * block_M, bx * block_N], X_shared)
            for i, j in T.Parallel(block_M, block_N):
                val = T.cast(X_shared[i, j], accum_dtype)
                Y_local[i, j] = T.max(val, T.cast(0.0, accum_dtype)) + T.log1p(T.exp(-T.abs(val)))
            T.copy(Y_local, Y[by * block_M, bx * block_N])
    return kernel

# ---------- parameters ----------
M, N = 1024, 2048
block_M, block_N = 64, 64

# ---------- host code ----------
x = torch.randn(M, N, dtype=torch.float16, device="cuda")
x[0, :5] = torch.tensor([-100, -20, 0, 20, 100], dtype=x.dtype, device=x.device)
kernel = softplus(M, N, block_M, block_N)
y = kernel(x)

# ---------- validation ----------
ref = torch.nn.functional.softplus(x.float()).half()
torch.testing.assert_close(y, ref, rtol=5e-2, atol=5e-2)
print("softplus: PASSED")

# ---------- profiling ----------
profiler = kernel.get_profiler()
latency = profiler.do_bench()
print(f"softplus latency: {latency:.4f} ms")
```

---

## 3. Row Reduction -- Row Sum

Sum each row of an (M, N) matrix, producing an (M,) vector.

```python
import torch
import tilelang
import tilelang.language as T

# ---------- kernel definition ----------
@tilelang.jit(out_idx=[-1])
def row_sum(M, N, block_M, block_N, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, N), dtype),
        Out: T.Tensor((M,), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), threads=128) as (bx,):
            A_shared = T.alloc_shared((block_M, block_N), dtype)
            acc = T.alloc_fragment((block_M,), accum_dtype)
            local_sum = T.alloc_fragment((block_M,), accum_dtype)

            T.clear(acc)
            for ko in T.serial(T.ceildiv(N, block_N)):
                T.copy(A[bx * block_M, ko * block_N], A_shared)
                T.reduce_sum(A_shared, local_sum, dim=1)
                for i in T.Parallel(block_M):
                    acc[i] = acc[i] + local_sum[i]
            T.copy(acc, Out[bx * block_M])
    return kernel

# ---------- parameters ----------
M, N = 512, 1024
block_M, block_N = 64, 128

# ---------- host code ----------
a = torch.randn(M, N, dtype=torch.float16, device="cuda")
kernel = row_sum(M, N, block_M, block_N)
out = kernel(a)

# ---------- validation ----------
ref = a.float().sum(dim=1)
torch.testing.assert_close(out, ref, rtol=5e-2, atol=5e-2)
print("row_sum: PASSED")

# ---------- profiling ----------
profiler = kernel.get_profiler()
latency = profiler.do_bench()
print(f"row_sum latency: {latency:.4f} ms")
```

---

## 4. GEMM -- Minimal

Standard tiled matrix multiplication: C = A @ B.
Uses shared memory staging, software pipelining, and tensor core GEMM.

```python
import torch
import tilelang
import tilelang.language as T

# ---------- kernel definition ----------
@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, num_stages=3, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])
    return kernel

# ---------- parameters ----------
M, N, K = 1024, 1024, 1024
block_M, block_N, block_K = 128, 128, 32
num_stages = 3

# ---------- host code ----------
a = torch.randn(M, K, dtype=torch.float16, device="cuda")
b = torch.randn(K, N, dtype=torch.float16, device="cuda")
kernel = matmul(M, N, K, block_M, block_N, block_K, num_stages)
c = kernel(a, b)

# ---------- validation ----------
ref = a @ b
torch.testing.assert_close(c, ref.to(torch.float16), rtol=1e-2, atol=1e-2)
print("matmul: PASSED")

# ---------- profiling ----------
profiler = kernel.get_profiler()
latency = profiler.do_bench()
tflops = 2 * M * N * K / latency * 1e-9
print(f"matmul latency: {latency:.4f} ms, {tflops:.2f} TFLOPS")
```

---

## 5. GEMM + Fusion -- Sigmoid Epilogue

Matrix multiplication followed by elementwise sigmoid: C = sigmoid(A @ B).
Demonstrates fusing a post-GEMM operation before the writeback copy.

```python
import torch
import tilelang
import tilelang.language as T

# ---------- kernel definition ----------
@tilelang.jit(out_idx=[-1])
def matmul_sigmoid(M, N, K, block_M, block_N, block_K, num_stages=3, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            # Fused sigmoid epilogue -- applied in-register before writeback
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = T.sigmoid(C_local[i, j])

            T.copy(C_local, C[by * block_M, bx * block_N])
    return kernel

# ---------- parameters ----------
M, N, K = 512, 512, 256
block_M, block_N, block_K = 128, 128, 32
num_stages = 3

# ---------- host code ----------
a = torch.randn(M, K, dtype=torch.float16, device="cuda")
b = torch.randn(K, N, dtype=torch.float16, device="cuda")
kernel = matmul_sigmoid(M, N, K, block_M, block_N, block_K, num_stages)
c = kernel(a, b)

# ---------- validation ----------
ref = torch.sigmoid((a @ b).float()).half()
torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
print("matmul_sigmoid: PASSED")

# ---------- profiling ----------
profiler = kernel.get_profiler()
latency = profiler.do_bench()
tflops = 2 * M * N * K / latency * 1e-9
print(f"matmul_sigmoid latency: {latency:.4f} ms, {tflops:.2f} TFLOPS")
```

---

## 6. Dynamic Shape GEMM

Uses `T.dynamic()` for symbolic dimensions so you can compile once and run with
different (M, N, K) sizes without recompilation.

```python
import torch
import tilelang
import tilelang.language as T

# ---------- kernel definition ----------
@tilelang.jit(out_idx=[-1])
def dynamic_matmul(block_M, block_N, block_K, num_stages=2, dtype=T.float16, accum_dtype=T.float32):
    M = T.dynamic("M")
    N = T.dynamic("N")
    K = T.dynamic("K")

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])
    return kernel

# ---------- parameters ----------
block_M, block_N, block_K = 128, 128, 32

# ---------- compile once ----------
kernel = dynamic_matmul(block_M, block_N, block_K)

# ---------- run with three different sizes ----------
test_shapes = [
    (256, 256, 256),
    (512, 1024, 256),
    (1024, 1024, 1024),
]

for (M, N, K) in test_shapes:
    a = torch.randn(M, K, dtype=torch.float16, device="cuda")
    b = torch.randn(K, N, dtype=torch.float16, device="cuda")
    c = kernel(a, b)

    # validation
    ref = a @ b
    torch.testing.assert_close(c, ref.to(torch.float16), rtol=1e-2, atol=1e-2)
    print(f"dynamic_matmul ({M}x{N}x{K}): PASSED")

# ---------- profiling (pick one representative size) ----------
profiler = kernel.get_profiler()
M, N, K = 1024, 1024, 1024
latency = profiler.do_bench(dynamic_symbolic_constraints={"M": M, "N": N, "K": K})
tflops = 2 * M * N * K / latency * 1e-9
print(f"dynamic_matmul ({M}x{N}x{K}) latency: {latency:.4f} ms, {tflops:.2f} TFLOPS")
```

---

## 7. GEMM with Transpose B

Computes C = A @ B^T where B has shape (N, K).
This is the common pattern in attention (Q @ K^T) and weight-transposed linear layers.
Uses `transpose_B=True` in `T.gemm`.

```python
import torch
import tilelang
import tilelang.language as T

# ---------- kernel definition ----------
@tilelang.jit(out_idx=[-1])
def matmul_nt(M, N, K, block_M, block_N, block_K, num_stages=3, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),       # B stored as (N, K), transposed logically
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])
    return kernel

# ---------- parameters ----------
M, N, K = 1024, 1024, 1024
block_M, block_N, block_K = 128, 128, 32
num_stages = 3

# ---------- host code ----------
a = torch.randn(M, K, dtype=torch.float16, device="cuda")
b = torch.randn(N, K, dtype=torch.float16, device="cuda")   # shape (N, K), not (K, N)
kernel = matmul_nt(M, N, K, block_M, block_N, block_K, num_stages)
c = kernel(a, b)

# ---------- validation ----------
ref = a @ b.t()
torch.testing.assert_close(c, ref.to(torch.float16), rtol=1e-2, atol=1e-2)
print("matmul_nt (A @ B^T): PASSED")

# ---------- profiling ----------
profiler = kernel.get_profiler()
latency = profiler.do_bench()
tflops = 2 * M * N * K / latency * 1e-9
print(f"matmul_nt latency: {latency:.4f} ms, {tflops:.2f} TFLOPS")
```

---

## Quick Reference -- API Patterns Used in These Templates

| API | Purpose | Used In |
|-----|---------|---------|
| `@tilelang.jit(out_idx=[-1])` | JIT compile; last param is output, returned automatically | All templates |
| `@T.prim_func` | Declare TIR kernel function | All templates |
| `T.Tensor((shape), dtype)` | Declare buffer in kernel signature | All templates |
| `T.Kernel(grid_x, grid_y, threads=N)` | Launch configuration | All templates |
| `T.alloc_shared((shape), dtype)` | Shared memory (on-chip SRAM) | All templates |
| `T.alloc_fragment((shape), dtype)` | Register fragment (per-thread, layout-inferred) | All templates |
| `T.copy(src[offset], dst)` | Data movement (global/shared/fragment) | All templates |
| `T.clear(buf)` | Zero-initialize a buffer | Templates 3-7 |
| `T.Parallel(dim0)` / `T.Parallel(dim0, dim1)` | Thread-parallel loop | Templates 1-3, 5 |
| `T.serial(iters)` | Sequential loop | Template 3 |
| `T.Pipelined(iters, num_stages=N)` | Software-pipelined loop | Templates 4-7 |
| `T.gemm(A, B, C)` | Tensor core matrix multiply-accumulate | Templates 4-7 |
| `T.gemm(..., transpose_B=True)` | GEMM with transposed B operand | Template 7 |
| `T.reduce_sum(src, dst, dim=1)` | Tile-level row reduction | Template 3 |
| `T.cast(value, dtype)` | Explicit type cast | Templates 1-2 |
| `T.log(x)`, `T.exp(x)`, `T.sigmoid(x)` | Elementwise math | Templates 2, 5 |
| `T.ceildiv(a, b)` | Ceiling division for grid dims | All templates |
| `T.dynamic("name")` | Symbolic dynamic dimension | Template 6 |
| `kernel.get_profiler()` | Create a Profiler for benchmarking | All templates |
| `profiler.do_bench()` | Benchmark latency (ms) with warmup | All templates |
| `profiler.do_bench(dynamic_symbolic_constraints={...})` | Benchmark dynamic-shape kernel at a specific size | Template 6 |

## Tolerance Guidelines

| Operation Type | Recommended rtol | Recommended atol | Reason |
|---------------|-----------------|-----------------|--------|
| fp16 GEMM | 1e-2 | 1e-2 | Tensor core accumulation rounding |
| fp16 GEMM + fusion | 1e-2 | 1e-2 | Same as GEMM plus epilogue rounding |
| Elementwise (transcendental) | 5e-2 | 5e-2 | exp/log/sigmoid approximation in fp16 |
| Reduction (row sum) | 5e-2 | 5e-2 | Accumulation order differs from PyTorch |
| Simple elementwise (scale) | 1e-2 | 1e-2 | Minimal rounding |

## TFLOPS Calculation

For GEMM-class kernels, TFLOPS is computed as:

```
tflops = 2 * M * N * K / latency_ms * 1e-9
```

The factor of 2 accounts for one multiply and one add per output element per K step.
`latency_ms` is in milliseconds, so `* 1e-9` converts to TFLOPS
(= 1e12 FLOP/s, and ms->s is 1e-3, so 1e-12 * 1e3 = 1e-9).
