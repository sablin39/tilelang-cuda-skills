# Execution harness for unattended workers

Keep runtime guarantees outside the model's prompt. Native agent tools can run
behind a thin execution boundary without replacing the agent's model/tool loop.
The task workspace still owns its mathematical oracle and benchmark.

## Before starting the agent budget

Verify the chosen interpreter and actual GPU execution inside the same boundary
as the worker: allocate, compute a known result, and synchronize. Enumeration
alone is insufficient. Record the visible device UUID and local CUDA ordinal.
Hide other GPU device nodes when isolation is requested; an environment variable
alone is a selection convention. Preserve common NVIDIA control/UVM nodes.

Use the intended OS identity, drop elevated capabilities, and prevent privilege
gain. Verify signal permissions with signal 0, without sending a terminating
signal. Document the actual boundary: Linux ownership checks protect other
users' processes, while same-user processes remain reachable unless separately
isolated. Keep inference and experiment resources explicit.

## During execution

- Bound commands and the complete run independently; allow deliberate larger
  compilation/profiling budgets. Preserve full unbuffered logs and the actual
  exit status, including pipeline errors, signals, and timeouts.
- Supervise the command's descendants. Nested timeout/session tools can escape
  a single process-group kill. Verify ownership and process identity before
  cleanup, and check for survivors when the run ends.
- Keep source changes, native session/events, environment, and skill revisions.
  Distinguish infrastructure interventions from model-authored fixes.
- Check artifacts before advancing stages: smoke, numerical whole-operation
  validation, and eligible measurements are different results. A printed PASS,
  output shape, or successful log-filter command cannot substitute for them.

## For a compact-prior stress test

Register the loop skill and give the operation, correctness/optimization goal,
workspace, and environment/device. Let the worker discover linked specialists.
Record initial prompt usage and all later interventions; preserve the tested
skill revision until the run ends. Keep any independent evaluator outside the
worker's view, and judge against the requested contract rather than unstated
hidden requirements. Apply trace-driven changes only after that trial.
