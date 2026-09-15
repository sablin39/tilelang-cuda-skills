# Attention as a Multi-Kernel Autograd Workflow

Read this only when the requested operator needs attention-style intermediates
and scattered gradient accumulation. It illustrates a programming model; it
does not require every differentiable operator to use the same phases.

## Forward and saved state

Forward can return `(output, lse)` while the public autograd operation returns
only `output`. Save inputs and backward intermediates with
`ctx.save_for_backward`. Record causal mode, scaling, and other non-tensor
configuration on the context.

The backward must use the exact forward log-sum-exp convention. Natural-log
and base-two implementations are not interchangeable without scale conversion.

## Backward phases and ownership

| Phase | Purpose | State contract |
| --- | --- | --- |
| Preprocess | `Delta = rowsum(output * grad_output)` | Return one value per query row |
| Main backward | Recompute probabilities and accumulate dQ/dK/dV | Determine which block owns each output tile |
| Postprocess, if needed | Convert/rearrange accumulation buffers | Match the public gradient dtype/layout |

When blocks partition K/V tiles, each block can own its dK/dV tile while
contributing partial dQ to multiple query tiles. An atomic dQ buffer must be
zeroed for every invocation. Other decompositions can change which outputs
need atomics; derive the ownership from the actual algorithm.

The wrapper can allocate dQ in float32 for accumulation accuracy and dK/dV in
the working dtype for direct writes. A separate conversion kernel or framework
conversion is an implementation choice whose cost belongs to the backward
measurement when it is part of the real path.

## Reference for testing

```python
import math
import torch

def attention_reference(q, k, v, causal=False):
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(q.shape[-1])
    if causal:
        rows = torch.arange(q.shape[-2], device=q.device)
        cols = torch.arange(k.shape[-2], device=q.device)
        mask = rows[:, None] >= cols[None, :]
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.einsum("bhqk,bhkd->bhqd", scores.softmax(-1), v)
```

This reference uses top-left causal alignment. Match the requested operator's
alignment, precision, mask, and scaling rather than assuming this convention
applies to every attention variant. Compare forward, lse, Delta, then gradients
with the same upstream gradient. Test masked tails and repeated backward calls.

## Read a matching complete implementation

Use source from the installed/matching TileLang checkout, since full attention
kernels evolve with compiler APIs:

- [Flash attention forward/backward](https://github.com/tile-ai/tilelang/blob/main/examples/flash_attention/example_mha_bwd_bhsd.py)
  for the preprocess/main/postprocess decomposition.
- [Linear attention backward](https://github.com/tile-ai/tilelang/blob/main/examples/linear_attention/example_linear_attn_bwd.py)
  only when the operator actually uses linear attention and recurrent state.
- [Attention sink backward](https://github.com/tile-ai/tilelang/blob/main/examples/attention_sink/example_mha_sink_bwd_bhsd.py)
  only when learnable sink parameters require additional gradients.

The general [gradient-testing-patterns.md](gradient-testing-patterns.md) contains
a small executable matmul example and comparison helper. Keep specialized
attention source out of an unrelated operator's starting template.
