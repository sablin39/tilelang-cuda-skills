---
name: tilelang-optimization-loop
description: Run a bounded TileLang implementation and optimization task from a PyTorch reference or existing kernel through baseline analysis, a written plan, candidate edits, lowering inspection, validation, measurement, and promotion. Use for iterative kernel work and reproducible handoffs; use a specialist directly for a single API, debugging, or profiling question.
---

# TileLang Optimization Loop

Follow a repeatable experiment cycle in the **task workspace**. Keep the skill
suite and its upstream knowledge separate from generated kernels and results.
Resume from evidence that still matches the source, inputs, and environment.
This adapts the [KDA workflow](https://github.com/mit-han-lab/kernel-design-agents/blob/4806866492d5c7cad05e07918790bd1e94e976a1/docs/agent-flow.md)
to the chain from TileLang source through compiler lowering to GPU behavior.

## 1. Establish the contract and baseline

Read the reference, current candidate, caller, and harness. Record the following
in the existing task record, or in `docs/draft.md`:

- Operation and input/output structure, shapes, dtypes, strides, accuracy,
  mutation/aliasing, initialization, stream, and gradient requirements.
- Supported cases and specialization boundaries. Preserve the supplied
  reference, validation cases, tolerances, and benchmark scope.
- Interpreter/environment, TileLang checkout/version, target GPU and execution
  backend, plus exact validation and benchmark commands.
- Objective across workload cases, promotion threshold/regression limits, and
  a finite candidate or time budget. Use the task's objective; if unspecified,
  choose and record a reasonable one before measuring candidates.

When forward and backward are both required, treat their implementations,
saved-state contract, and configurations as one candidate. Use the
[joint forward/backward workflow](references/joint-fwd-bwd.md) to define the
combined objective and paired correctness checks before optimizing either side.

For legacy examples, separate migration from tuning: choose one small supported
case and one configuration before launching an inherited autotuner. Preserve
the original source, record each failure/fix, and synchronize after execution.
Compilation or shape checks establish smoke status only. Establish numerical
correctness of the complete requested operation against an independent
reference before tuning; then preserve the correct baseline and warmed timings.

For a supplied PyTorch baseline, inspect its validated `torch.compile` graph,
generated code, and warmed profile using [baseline inspection](../tilelang/torch-profiling-tilelang-programs/references/torch-compile-inspection.md).
Record eager and compiled timings separately, outside compilation cost. Reuse
existing artifacts when applicable. If compilation is unavailable or fails,
record the missing evidence and use the validated eager path as described in
that reference. Keep the mathematical oracle independent of the candidate.

When the task changes a multi-operation fusion boundary, logical sharding,
topology, or high-level memory placement, or already uses TileFoundry, apply the
[TileFoundry design workflow](../designing-with-tilefoundry/SKILL.md) between
the contract and the plan:

1. Author each design direction as an HIR variant (fusion boundaries, storage
   placement, sharding) and diff the `analyze` reports to rank them before any
   kernel work. Verified against 0.0.2: relative diffs transfer to measurement
   — a 1.67× gmem-traffic diff predicted the fused-vs-split direction and the
   `bound-by: memory` diagnosis, with the measured latency ratio landing at
   1.3-1.4× — while absolute predicted nanoseconds do not transfer, being
   priced at the named target's published rates rather than the attached GPU.
2. Run `tilefoundry check` on a plain-PyTorch runtime twin against the HIR
   evaluator as a pre-codegen semantic oracle, so rounding-chain and semantics
   errors surface before a GPU kernel exists.
3. Record the target in SM capability terms (sm_90, sm_100, ...) so the
   evidence states which architecture features it relies on.

Record these as modeled design evidence; the emitted TileLang lowering and
measured GPU result remain separate acceptance gates. Rank HIR variants by
logical traffic/FLOP diffs, capacity advisories, and roofline floors.
`analyze --performance` is a modeled plan, not a measured floor: at `bb3b726`
it prices the mesh that runs (`wave=active/physical`) and the footprint's bytes
at the bandwidth level, with a per-CTA Little's-law cap over the in-flight
bytes the program states — the per-position billing and invisible lane
under-fill that made older pins untrustworthy are gone. What it still cannot
see is achieved residency, dependent-chain latency, and pipeline overlap; when
a decision turns on those, use the measured two-bound recipe in
[the cost model](references/cost-model.md) and keep the benchmark as the
verdict. Historical: at `5f346a2` and `25fbbaa` the timeline billed
`insert_slice`/`tf.zeros` per position and read a partially filled launch as a
full one, so it named the wrong winner on an occupancy-gated scan (verified
2026-10-16). Do not introduce TileFoundry solely to tune an isolated,
already-defined `PrimFunc`, and do not rely on TileFoundry's own `compile()`
CUDA backend as a measured floor — it remains experimental (TMA/WGMMA emitter
gaps); see the TileFoundry design skill for its status at the active pin.

## 2. Write the draft, then an executable plan

Before editing a candidate, write a compact `docs/draft.md`: the selected region,
baseline findings, unknowns, and a few candidate directions ranked by expected
benefit and risk. Research only what those decisions need.

Turn it into `docs/plan.md` with the first experiment, files to change, expected
lowering/runtime effect, commands to run, acceptance rule, and search budget.
The agent can do this directly. Reuse equivalent existing planning files;
retain one plan and one result history in the task workspace.

The draft should link the baseline validation, timing, and compiled-baseline
inspection artifacts, or name the concrete unavailable/failed stage. A plan
that only describes the original source has not established those results.

Choose the smallest experiment that can test the current explanation. A
controlled parameter search can cover interacting variables, but give it a
mechanism, bounded configurations, and a selection rule before launching it.

## 3. Implement and inspect one candidate

Assign an ID and parent (the current best correct candidate); preserve source
and configuration in a named file or revision so the result can be reproduced.
Use [writing](../tilelang/writing-tilelang-kernels/SKILL.md) for the implementation
and [optimization](../tilelang/optimizing-tilelang-programs/SKILL.md) for scheduling
and autotuning details. Keep required resets, allocations, conversions, and
launches in the operation's contract.

Inspect the relevant boundary with [compiler tools](../tilelang/debugging-tilelang-programs/references/compiler-tools.md):
specialized TIR, changed lowering passes, generated host/device code, or layout.
Record whether the intended change materialized. A stage count or tile size
in Python is not evidence that the compiler produced the expected schedule.
Capture artifacts from the exact compiled specialization under investigation;
record its workload/configuration and emitted launch geometry. A cache file
from another shape cannot establish what the measured candidate executed.

For a human-facing proposal, implementation explanation, or final review, use
the [design-document workflow](../tilelang/writing-tilelang-design-docs/SKILL.md)
to combine a clear overview with the relevant layout, lowering, lifetime, or
runtime artifact. Keep this document separate from the operational plan and
candidate record.

When a mechanism needs explanation, use the supplementary
[KernelWiki interface](references/kernelwiki.md). Retrieve selected pages and
their concrete sources, check applicability to this target/version, and connect
them to the actual emitted implementation. Keep the Wiki knowledge upstream.
For instruction-specific claims, inspect target PTX/SASS or source-correlated
profiling when needed; generated CUDA alone does not establish the final
machine instruction sequence.

## 4. Validate, then measure

Run the fixed correctness command on every candidate considered for promotion,
including relevant tails, repeated calls, and state/gradient cases. Use
[debugging](../tilelang/debugging-tilelang-programs/SKILL.md) for failures and
[forward/backward testing](../tilelang/testing-fwd-bwd-kernels/SKILL.md) when needed.
Record compile, runtime, and numerical failures separately. Failed or untested
candidates have no eligible performance result.

When a TileFoundry runtime twin exists for the candidate, also run
`tilefoundry check` against the authored evaluator and keep its JSON report
with the candidate evidence. It is a second, independent agreement gate; it
never replaces the fixed correctness command.

For passing candidates, use the fixed [profiling workflow](../tilelang/profiling-tilelang-programs/SKILL.md).
Warm all compared paths and keep shapes, timing backend, allocation/reset/cache
policy, and units consistent. Record variation; serialize measurements on a
shared GPU and report interference. Use traces/NCU for focused diagnostic
questions, then use the ordinary benchmark to select a winner.

If the observed lowering or timing contradicts the hypothesis, record that
result and revise the explanation. Avoid repeating an unchanged failing
command or expanding a search without new evidence.

## 5. Record a decision and choose the next experiment

Keep one compact candidate record in `candidates.jsonl` (or the task's format)
and measurements in `benchmark.csv` or existing structured results:

| Record | Required content |
| --- | --- |
| Identity | Candidate ID, parent, source revision/hash/path, configuration, workload cases |
| Hypothesis | Observed cost, intended change, expected lowering and runtime effect; Wiki/source locator if used |
| Evidence | Actual lowering artifact/finding; correctness status, command, tolerances, and log |
| Measurement | Command, scope/backend/units, baseline and candidate per-case values, variation, artifact paths; for a modeled stage, `measured / floor` and the constants it used |
| Decision | Promote, reject, inconclusive, or blocked; reason and next experiment if any |

When a stage's gap to its floor needs an explanation, the
[cost model](references/cost-model.md) separates work, bandwidth, residency,
and chain terms with constants measured on the attached device. It explains a
measurement; it never replaces the benchmark.

Promote only when the candidate satisfies correctness and the recorded workload
objective/regression limits. Compare with the current best and preserve the
original baseline for the final comparison. A faster single case does not
establish a win across the workload; improvement within variation is inconclusive.
The [worked record](references/experiment-record.md) shows the expected level
of detail without prescribing a new evaluator.

Update the plan from the evidence and return to step 3. Stop at the objective,
search budget, or an external blocker that prevents the next useful experiment.
Retain the best validated candidate if nothing improves it. Explain exactly
which claims remain unvalidated when execution or tools are unavailable.

## 6. Reproduce and hand off

Re-run the selected **exact source/configuration** with the final correctness
and benchmark commands. A source hash ties the final file to its evidence.
Check that baseline analysis, changed lowering, validation, measurement, and
promotion each have an artifact/result or an explicit unresolved status.
For joint work, this includes the forward/backward pair and saved-state costs.
Report the selected candidate, baseline comparison, supported cases, rejected
hypotheses, and remaining limitations; preserve the draft, plan, results, and
relevant lowering/profile artifacts in the task workspace.

Give another worker one bounded stage, the contract, allowed files, active
environment/device, exact commands, current best/results, and expected outputs.
Have it return files, commands actually run, evidence, and a decision. Check
those artifacts before advancing. For lighter workers, use the
[checkpoint handoff](references/worker-handoff.md) to separate baseline analysis,
one candidate experiment, and the final audit. A successful timing run does not
complete missing analysis stages. Use [external tool integration](references/external-tools.md)
for optional report tools and KDA conventions; the task harness owns execution.
