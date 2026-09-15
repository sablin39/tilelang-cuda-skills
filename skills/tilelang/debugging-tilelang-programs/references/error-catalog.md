# Failure Patterns and Next Checks

| Failure | Inspect | Typical correction |
| --- | --- | --- |
| GEMM reduction dimensions disagree | Physical tile shapes and transpose flags | Match the logical reduction dimensions |
| Host dtype/rank/shape assertion | Tensor signature and `get_host_source()` | Supply the declared contract or change it deliberately |
| Host stride assertion | Slice, transpose, or expanded tensor | Support those strides or materialize contiguous data |
| Host device assertion | Devices of every argument and compiled target | Use one compatible device |
| Missing output/argument | `out_idx` or eager `T.empty`/return convention | Pass only the required launch inputs |
| Codegen failure before a kernel exists | `get_tir`, compile-only, lower trace | Find the first failing transformation or source construct |
| Resource-related launch failure | Generated allocation/launch configuration | Reduce required resources or select a supported configuration |
| Invalid transfer descriptor | Declared strides/alignment and selected transfer path | Satisfy that path's contract or use a supported alternative |
| Partial output corruption | Tail masks, indexing, writeback, initialization | Isolate the first differing tile |
| NaN/inf | Uninitialized accumulation, invalid arithmetic, masking, overflow | Compare intermediates and extreme finite inputs |
| Run-to-run differences | Output resets, state, atomics, synchronization | Distinguish expected numeric variation from incorrect ordering |

For `A @ B.T`, physical tiles are A `(BM, BK)`, B `(BN, BK)`, and accumulator
`(BM, BN)`. The non-transposed B convention `(BK, BN)` is not interchangeable.

Do not infer a unique cause from a symptom such as NaN or a slow kernel.
Reduce to a small valid configuration and compare the relevant intermediate
against a reference. A tail-only failure points to a boundary hypothesis;
test it rather than assuming all other indexing is correct.

For runtime memory errors, synchronize at a meaningful boundary or use
`CUDA_LAUNCH_BLOCKING=1`, then run the appropriate sanitizer. Inspect host
argument checks before attributing a failure to the device computation.
