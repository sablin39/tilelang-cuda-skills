# Optimize forward and backward as one candidate

Use this mode when a training operation or explicit forward/backward pair is
required. Read the [testing skill](../../tilelang/testing-fwd-bwd-kernels/SKILL.md)
for gradient comparison mechanics. The loop's candidate identity includes both
implementations, their configurations, and the saved-state schema.

## Fix the paired contract and objective

Record forward inputs/outputs, every required input/parameter gradient,
upstream-gradient dtype/stride support, saved tensors, aliases/lifetimes, and
any recomputation. State whether higher-order derivatives or accumulation into
existing buffers are required. Preserve the caller's API or autograd adapter.

Measure the operation the caller needs: typically forward plus backward,
including saved-state allocation, required conversions/resets, and all gradient
kernels. Use the task's weighted workload objective and any per-phase regression
or memory limits. Record forward and backward timings as diagnostics alongside
an actual combined measurement. Their sum alone can miss allocation, replay,
or state-lifetime effects in the combined path.

Do not promote a forward-only gain when the required combined objective fails.
A slower forward may be useful if it produces saved state that improves the
complete training step within the task's constraints.

## Establish alignment before tuning

Locate or assemble the caller that connects the complete forward/backward pair,
including saved intermediates. Component tests help diagnose this path; they
do not establish whole-operation correctness or performance.

Use independent reference inputs/graphs and the **same supplied upstream
gradient** for reference and candidate. Compare forward outputs and each
requested gradient. Include nonuniform upstream gradients; a summed loss alone
can hide transposes and scaling errors. Exercise supported noncontiguous or
expanded upstream gradients and count required conversion costs.

Check state as well as numbers: interleave two forward calls with different
inputs, then run backward using each matching context. Repeat backward when
the contract permits it, verify earlier outputs/gradients remain valid, and
reset accumulating buffers at the required boundary. Compile success or a
correct forward does not establish a correct backward.

For nonlinear functions, follow the reference's rounding, masks, scaling,
accumulation dtype, and derivative convention. A changed forward intermediate
can change the appropriate saved state and backward computation. Keep all
changes to that interface in the same candidate.

## Plan an experiment across both phases

Use baseline graph/code/profile findings to choose a joint hypothesis. Examples
of questions to investigate, rather than mandatory optimizations:

- Would saving an intermediate remove backward work, and what storage/traffic
  does that add to forward and the complete step?
- Would recomputation lower saved-state cost enough to offset extra arithmetic?
- Can forward layout or tiling make the transposed/gradient operations cheaper
  without adding an expensive conversion?
- Does fusing a derivative, reduction, or epilogue reduce total launches/traffic,
  and does it preserve output ownership and synchronization?

Inspect the relevant forward **and** backward lowering. Use the
[KernelWiki interface](kernelwiki.md) for unfamiliar emitted mechanisms and
target-specific explanations, then test that explanation on the local pair.
Configuration search may use different tile sizes for different gradients;
select the complete pair using the recorded workload objective.

## Record and promote the pair

For each candidate, retain:

| Evidence | Content |
| --- | --- |
| Identity | Forward/backward source hashes/configurations and parent candidate |
| State | Saved tensor shapes/dtypes/bytes, lifetimes, and recomputation policy |
| Correctness | Forward and every required gradient, matched upstream gradients, state/stride cases |
| Mechanism | Expected and observed changes in both phases; lowering/profile/source locators |
| Performance | Forward, backward, and combined per-case timing under stated scopes; combined objective and variation |
| Decision | Promotion/rejection against the complete objective, phase/memory constraints, and remaining uncertainty |

Validate and remeasure the final exact pair. Report phase tradeoffs explicitly;
separate numerical success from missing profiling or compiler evidence. Keep
the best validated pair if the search finds no eligible improvement.
