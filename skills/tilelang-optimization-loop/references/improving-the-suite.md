# Feed findings back into the skills

At final handoff, distinguish a task-specific tuning result from a reusable
instruction gap. A winning tile size for one shape is normally task evidence.
A changed API, missing routing condition, broken helper, or reproducible
compiler limitation may justify updating the suite. It is fine to record that
there is no reusable finding; do not manufacture a skill edit for every task.

For each proposed update, keep a short maintenance record in the task workspace:

- Decision or failure that existing guidance did not handle, and its owner
  skill/reference.
- Source/tool versions, target, shape/dtype scope and exact reproducer.
- Expected versus observed behavior and supporting artifacts; distinguish a
  single observation from a general rule.
- Proposed current-behavior wording, competing/obsolete advice to remove,
  and one counterexample or nearby case that should not activate this advice.
- Validation result on the affected case and that nearby case, plus any
  untested boundary.

When suite maintenance is in scope, patch the smallest owning reference and
its routing link, test changed helpers and installation paths, then replay the
affected case. Use a `maintenance` stage report to connect findings, checks,
and the accept/reject decision. This is a suite-edit decision, not a kernel
performance promotion. Keep raw traces, benchmark outputs and provenance in
the task record; ship concise instructions about current behavior.

When maintenance is outside the task scope, return the concrete proposal with
the final handoff. Do not silently modify installed/global skills or publish
findings from a kernel task. A later maintainer should be able to reproduce
the observation without reconstructing the conversation.
