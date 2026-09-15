# Selecting a Drafting Pattern

Start with the smallest pattern that expresses the operator. Complete local
examples are in [kernel-templates.md](kernel-templates.md); use them to establish
correctness before adding scheduling or fusion. The table below routes more
specialized work to matching docs or source instead of duplicating full kernels.

| Operator / programming pattern | Starting point | Contract to preserve |
| --- | --- | --- |
| Elementwise | Local vector-scale template or eager add in [language-docs.md](language-docs.md) | Shape, strides, tail writes, output ownership |
| Reduction | Local row-sum template | Reduction dimension, accumulator dtype, padding identity |
| GEMM | Local matmul or transposed-B template | Logical operand dimensions and transpose flags |
| Fusion | Local sigmoid epilogue template | Reference order, precision, and output conversion |
| Dynamic shape | Local dynamic GEMM or eager symbolic add | Shape binding and the separate stride contract |
| Normalization / softmax | Matching current example | Stable exponentials, reduction identity, masked tails |
| Attention | Matching forward/backward example | Scaling, mask convention, saved intermediates, gradient ownership |
| Stateful operator | Matching recurrent/chunked example | State dependency and ordering across chunks |

For current patterns, inspect the installed TileLang checkout first:

- `docs/deeplearning_operators/elementwise.md`
- `docs/deeplearning_operators/matmul.md`
- `docs/deeplearning_operators/gemv.md`
- `examples/norm/`, `examples/online_softmax/`, and `examples/flash_attention/`

Upstream: [operator docs](https://github.com/tile-ai/tilelang/tree/main/docs/deeplearning_operators)
and [examples](https://github.com/tile-ai/tilelang/tree/main/examples).
Some documentation snippets omit setup or use older call conventions; check
out_idx, reference return values, and the installed signatures before reuse.

Avoid copying an entire attention or normalization implementation into a
simple operator merely because it includes a useful loop. Extract the needed
programming pattern, validate the new contract, and retain an inspection path
from factory arguments through TIR to generated source.
