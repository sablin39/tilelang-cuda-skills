# Gradient Testing Patterns

## 1. Simple Matmul Fwd+Bwd

Minimal complete example: a matmul forward kernel and its backward kernel, wrapped in `torch.autograd.Function`, with gradient validation.

### Forward Kernel

```python
import tilelang
import tilelang.language as T
import torch

@tilelang.jit(out_idx=[-1])
def matmul_fwd(M, N, K, block_M=128, block_N=128, block_K=32,
               dtype=T.float16, accum_dtype=T.float32):
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
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])
    return kernel
```

### Backward Kernel

For C = A @ B, the gradients are:
- dA = dC @ B^T  (shape: M x K)
- dB = A^T @ dC  (shape: K x N)

```python
@tilelang.jit(out_idx=[-1])
def matmul_dA(M, K, N, block_M=128, block_N=128, block_K=32,
              dtype=T.float16, accum_dtype=T.float32):
    """dA = dC @ B^T. dC is (M, N), B is (K, N), dA is (M, K)."""
    @T.prim_func
    def kernel(
        dC: T.Tensor((M, N), dtype),
        B: T.Tensor((K, N), dtype),
        dA: T.Tensor((M, K), dtype),
    ):
        with T.Kernel(T.ceildiv(K, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            dC_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            dA_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(dA_local)
            for ko in T.Pipelined(T.ceildiv(N, block_K), num_stages=2):
                T.copy(dC[by * block_M, ko * block_K], dC_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                T.gemm(dC_shared, B_shared, dA_local, transpose_B=True)
            T.copy(dA_local, dA[by * block_M, bx * block_N])
    return kernel

@tilelang.jit(out_idx=[-1])
def matmul_dB(K, N, M, block_M=128, block_N=128, block_K=32,
              dtype=T.float16, accum_dtype=T.float32):
    """dB = A^T @ dC. A is (M, K), dC is (M, N), dB is (K, N)."""
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        dC: T.Tensor((M, N), dtype),
        dB: T.Tensor((K, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(K, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_K, block_M), dtype)
            dC_shared = T.alloc_shared((block_K, block_N), dtype)
            dB_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(dB_local)
            for ko in T.Pipelined(T.ceildiv(M, block_K), num_stages=2):
                T.copy(A[ko * block_K, by * block_M], A_shared)
                T.copy(dC[ko * block_K, bx * block_N], dC_shared)
                T.gemm(A_shared, dC_shared, dB_local, transpose_A=True)
            T.copy(dB_local, dB[by * block_M, bx * block_N])
    return kernel
```

### autograd.Function Wrapper

```python
class TileLangMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        M, K = A.shape
        K2, N = B.shape
        assert K == K2

        fwd_kernel = matmul_fwd(M, N, K)
        C = fwd_kernel(A, B)

        ctx.save_for_backward(A, B)
        ctx.M, ctx.N, ctx.K = M, N, K
        return C

    @staticmethod
    def backward(ctx, dC):
        A, B = ctx.saved_tensors
        M, N, K = ctx.M, ctx.N, ctx.K

        dC = dC.contiguous()
        dA_kernel = matmul_dA(M, K, N)
        dB_kernel = matmul_dB(K, N, M)

        dA = dA_kernel(dC, B)
        dB = dB_kernel(A, dC)
        return dA, dB
```

### Test

```python
M, N, K = 256, 256, 256
A = torch.randn(M, K, device="cuda", dtype=torch.float16, requires_grad=True)
B = torch.randn(K, N, device="cuda", dtype=torch.float16, requires_grad=True)
dO = torch.randn(M, N, device="cuda", dtype=torch.float16)

# Reference
A_ref = A.detach().clone().requires_grad_(True)
B_ref = B.detach().clone().requires_grad_(True)
C_ref = A_ref @ B_ref
C_ref.backward(dO)

# Custom
C_custom = TileLangMatmul.apply(A, B)
C_custom.backward(dO)

# Compare
torch.testing.assert_close(C_custom, C_ref, rtol=1e-2, atol=1e-2)
torch.testing.assert_close(A.grad, A_ref.grad, rtol=1e-2, atol=1e-2)
torch.testing.assert_close(B.grad, B_ref.grad, rtol=1e-2, atol=1e-2)
print("All gradients match!")
```


## 2. Reference-Based Comparison Helper

The helper below handles a tensor or tuple of differentiable tensor outputs.
For auxiliary nondifferentiable outputs, check them separately and pass only
the differentiable outputs to this helper. Each path receives independent
input tensors so gradient accumulation cannot contaminate the other path.

```python
def compare_backward(custom, reference, args, rtol=1e-2, atol=1e-2):
    def clone_args():
        return tuple(x.detach().clone().requires_grad_(x.requires_grad)
                     if isinstance(x, torch.Tensor) else x for x in args)

    custom_args, reference_args = clone_args(), clone_args()
    actual, expected = custom(*custom_args), reference(*reference_args)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    actual = (actual,) if isinstance(actual, torch.Tensor) else tuple(actual)
    expected = (expected,) if isinstance(expected, torch.Tensor) else tuple(expected)
    upstream = tuple(torch.randn_like(x) for x in actual)
    custom_inputs = tuple(x for x in custom_args
                          if isinstance(x, torch.Tensor) and x.requires_grad)
    reference_inputs = tuple(x for x in reference_args
                             if isinstance(x, torch.Tensor) and x.requires_grad)
    actual_grads = torch.autograd.grad(actual, custom_inputs, upstream, allow_unused=True)
    expected_grads = torch.autograd.grad(expected, reference_inputs, upstream, allow_unused=True)
    for index, (got, want) in enumerate(zip(actual_grads, expected_grads)):
        if got is None or want is None:
            assert got is want, f"gradient {index}: missing on only one path"
        else:
            torch.testing.assert_close(got, want, rtol=rtol, atol=atol)
```

Use nonsquare inputs to expose accidental gradient swaps and transpose errors.
Use random upstream gradients rather than testing only a sum loss. Also test
the wrapper with expanded/non-contiguous upstream gradients if supported.

Finite-difference `gradcheck` complements reference comparisons when the
kernel supports a numerically suitable precision path, normally float64.
Low-precision tensor-core kernels generally need reference-based checks.
Do not loosen tolerances until deliberately wrong gradients pass. Higher-order
checks require an explicitly differentiable backward implementation.
