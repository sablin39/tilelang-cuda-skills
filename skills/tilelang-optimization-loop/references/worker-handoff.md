# Checkpoint handoffs for lighter workers

Use this pattern when work is delegated, or as a self-check between stages.
Give a worker one checkpoint plus the existing contract and evidence. The
coordinator checks the returned artifacts before advancing; a worker's prose
claim that a stage passed is not sufficient. Keep the same task plan, candidate
history, oracle, workload objective, and environment across checkpoints.
For unattended runs, enforce resource limits and truthful command status in the
[execution harness](execution-harness.md), outside the model's task prompt.

## Baseline checkpoint

```text
Use tilelang-optimization-loop, steps 1–2, in <task workspace>.
Read <task contract> and the supplied reference/current pair/harness.
Establish validation, warmed timing, and compiled-baseline inspection evidence.
Write the draft and executable first-experiment plan, with exact artifact paths.
Return the proposed experiment and a status for each baseline requirement.
Do not edit a candidate during this checkpoint. Report unavailable/failed
inspection with its command and result instead of describing it as complete.
```

Accept a concrete plan supported by evidence, or explicitly decide whether
the available evidence is sufficient to proceed despite a missing tool/stage.

## Candidate checkpoint

```text
Use tilelang-optimization-loop, steps 3–5, to run <one hypothesis or bounded search>.
Inputs: <accepted plan>, <current best>, <baseline evidence>, <fixed commands>.
Allowed edits: <candidate files>; budget/device: <limits>.
Return the exact candidate identity, matched lowering artifacts, correctness,
per-case measurements/variation, and a proposed decision against the objective.
For joint work include both phases, all gradients, and saved-state costs.
Record the result before starting another experiment.
```

Accept promotion only when the evidence supports the contract and objective.
If measurements are dominated by contention or variation, retain the current
best and label the comparison inconclusive. A slower sample alone does not
establish a cause such as occupancy, memory traffic, or launch overhead.

## Final audit checkpoint

```text
Use tilelang-optimization-loop step 6 to audit <selected candidate and records>.
Check the matrix below using concrete artifacts and commands actually run.
Reproduce missing evidence within <allowed resources/time>; otherwise mark
the requirement unrun, failed, or inconclusive with the reason.
Correct unsupported claims. Return a selected implementation and an honest
evidence matrix. Do not start a new optimization search during this checkpoint.
```

| Requirement | Evidence to check |
| --- | --- |
| Identity | Selected source/configuration and imported helper hashes match measured files |
| Baseline | Reference validation, baseline timings, compiled graph/code/profile artifacts or explicit missing status |
| Lowering | Exact file paths with candidate, phase, shape, target/backend, and actual emitted launch geometry; a cache directory alone is insufficient |
| Research, if used | Actual query result, page ID/revision, cited source, and applicability; a failed query supplies no supporting knowledge |
| Correctness | Exact selected forward/backward pair, every required gradient, state/stride checks, command/results |
| Performance | Combined objective and per-phase tradeoffs, variation/interference, scope, units, and unchanged harness |
| Decision | Final reproduction and supported promotion/retention; unresolved claims identified |

The audit may confirm correct kernels while leaving a performance mechanism
or baseline analysis unresolved. Report these statuses separately. Keep
unavailable evidence visible in the handoff instead of inferring it from a
candidate's source or a timing improvement.
