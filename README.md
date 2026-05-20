# GPU Development Skills

Agent skills for GPU kernel development — writing, debugging, profiling, and optimizing CUDA and [TileLang](https://github.com/tile-ai/tilelang) GPU kernels.

The `.vscode` directory is intentionally tracked as a reference to configure highlighting for humans.

## TileLang Skills

TileLang skills created by Claude Opus 4.6 (probably) in Claude Code with the `skill-creator` plugin. Based on TileLang `v0.1.9` docs and examples, validated on RTX PRO 6000 Blackwell GPUs (sm_120) with CUDA 13.1 and PyTorch 2.11.

| Skill | Description | Key Topics |
|-------|-------------|------------|
| [writing-tilelang-kernels](skills/tilelang/writing-tilelang-kernels/SKILL.md) | Write TileLang GPU kernels from scratch or by adapting patterns | Kernel anatomy, templates (GEMM, elementwise, reduction), memory scopes, T.copy/T.gemm, dynamic shapes |
| [debugging-tilelang-programs](skills/tilelang/debugging-tilelang-programs/SKILL.md) | Diagnose and fix errors in TileLang programs | Failure taxonomy, T.print, AutoDD, compute-sanitizer, numerical drift, race detection |
| [profiling-tilelang-programs](skills/tilelang/profiling-tilelang-programs/SKILL.md) | Benchmark and profile TileLang kernels | do_bench backends, TFLOPS/bandwidth, ncu bottleneck diagnosis (pipe utilization, warp stalls), roofline |
| [torch-profiling-tilelang-programs](skills/tilelang/torch-profiling-tilelang-programs/SKILL.md) | Lightweight torch.profiler alternative to ncu/nsys for TileLang | torch.profiler setup, key_averages, chrome trace, roofline classification (IO / CUDA-core / tensor-core), launch overhead, memory profiling |
| [optimizing-tilelang-programs](skills/tilelang/optimizing-tilelang-programs/SKILL.md) | Optimize TileLang kernels for performance | Tile sizes, pipeline stages, threads, AutoTuner, epilogue fusion, swizzle, ncu-guided tuning |
| [testing-fwd-bwd-kernels](skills/tilelang/testing-fwd-bwd-kernels/SKILL.md) | Test kernels with forward and backward passes | torch.autograd.Function, compare_backward (not gradcheck), mixed-precision, atomicAdd, attention bwd |

### Workflow

The skills form a natural progression:

```
writing → debugging → profiling → optimizing
                          ↓           ↓
            torch-profiling   testing-fwd-bwd (for differentiable ops)
```

1. **Write** a kernel using templates from the writing skill
2. **Debug** if it fails to compile or produces wrong results
3. **Profile** to measure baseline performance — use `profiling-tilelang-programs` for the ncu/do_bench workflow, or `torch-profiling-tilelang-programs` when ncu/nsys aren't available and you want a quick `torch.profiler`-based pass
4. **Optimize** to improve performance
5. **Test fwd+bwd** if the kernel needs gradients

## CUDA Skills

General-purpose CUDA development skill for debugging, profiling, and optimizing GPU kernels — independent of any specific framework. Originally from [`technillogue/ptx-isa-markdown`](https://github.com/technillogue/ptx-isa-markdown). Includes scraped PTX ISA 9.1, CUDA Runtime API 13.1, and CUDA Driver API 13.1 documentation (640+ markdown files) for grep-based lookup.

| Skill | Description | Key Topics |
|-------|-------------|------------|
| [cuda-programming](skills/cuda_skill/SKILL.md) | Debug, profile, and optimize CUDA kernels | compute-sanitizer, cuda-gdb, ncu/nsys profiling, NVTX, PTX ISA, coalescing, bank conflicts, inline PTX |