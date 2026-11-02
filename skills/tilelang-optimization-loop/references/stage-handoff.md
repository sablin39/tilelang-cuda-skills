# Stage reports and evidence reuse

Use one JSON report after a bounded baseline, design, candidate, diagnosis or
final stage when the task spans iterations, workers or sessions. A single API
question does not need a report. Link reports from the existing candidate ledger;
do not create a second experiment history. A stage may finish by rejecting a
candidate. `complete` means the stage finished, not that every check passed.

## Before doing the work

Name the decision question and read the selected specialist's `SKILL.md` and
only the references needed to answer it. Reuse an earlier finding when its
source, helpers, case/configuration, contract and environment still match.
Record a reason for an unavailable tool or an inapplicable check. A missing
tool never counts as a passing check; an optional missing diagnosis need not
block a numerically correct measured improvement, but its mechanism stays
unproven. Do not invoke all specialists just to fill a checklist.

## Report contract

Use schema version 1 with these fields. All paths in fingerprints are relative
to the task root; files may not resolve outside it. Put tool versions, target,
workload, tolerances, benchmark policy and acceptance limits in a task-context
file and fingerprint it alongside the candidate, imported helpers and harness.
Record the full command, including relevant environment overrides, actually run.

| Field | Content |
| --- | --- |
| `schema_version` | `1` |
| `stage` | `baseline`, `design`, `candidate`, `diagnosis`, `final`, `maintenance` |
| `candidate` | Stable candidate ID; use a baseline ID before any edits |
| `status` | `complete`, `incomplete`, `blocked`, `inconclusive` |
| `inputs` | Nonempty list of `{ "path": "…", "sha256": "…" }` fingerprints |
| `skills` | List of `{ "name", "status", "question", "finding", "evidence" }`; status is `used`, `reused`, `unavailable`, or `not_applicable`; evidence contains check IDs |
| `checks` | List of the check records below |
| `decision` | `{ "action", "reason", "evidence", "next" }`; action is `continue`, `promote`, `retain`, `reject`, or `blocked`; evidence contains check IDs consumed by this decision |
| `decision.next` | `null` at completion, otherwise `{ "stage", "question", "skills": ["skill-name"] }` |

Each check has a unique `id`, a `kind` (`correctness`, `benchmark`, `lowering`,
`model`, `research`, `reproduction`), a boolean `required`, and `status` (`pass`,
`fail`, `unrun`, `unavailable`, `inconclusive`, `not_applicable`). Include
`command` and `exit_code` (both nullable for non-execution evidence), a `summary`
of the finding or missing evidence, and `artifacts` with file fingerprints.
A passing execution check needs a successful command and artifacts. Mark noisy
timing `inconclusive` even if the process exited successfully. Set `required`
from the contract before executing, not to excuse a failed promotion gate.

Promotion must consume passing correctness and benchmark checks and resolve
every required check. The decision reason must compare the workload objective,
regression limits and variation; a zero exit code cannot establish a speedup.
Rejected candidates may have failed checks. Preserve their evidence too.

## Check and resume

`<loop-skill>` below is the installed directory containing this `SKILL.md`,
not a path relative to the task's working directory:

```bash
python <loop-skill>/scripts/check_handoff.py fingerprint \
  candidate.py harness.py docs/task-context.json --root <task-root>
python <loop-skill>/scripts/check_handoff.py check \
  <task-root>/reports/c02-candidate.json --root <task-root>
```

Generate fingerprints from existing files after the run; never invent hashes.
On resume, run the checker and review the cited artifacts before consuming
the report. It checks structure, references, file identity and promotion gates;
it cannot prove the reported commands ran, the findings are true, or the
context file still describes the current machine. Check the active environment
against that file. A new source/helper/context hash invalidates dependent
checks. Rerun those checks; retain unrelated valid research as scoped evidence.
Archive prior candidate sources to keep rejected experiments reproducible.

The next stage consumes the finding, not merely the fact that a skill was read:

| Finding | Next action |
| --- | --- |
| Wrong output or gradient | Testing/debugging isolates the first mismatch before timing |
| Intended pipeline absent from emitted code | Debugging inspects the lowering boundary; revise the candidate |
| Timing dominated by variation/contention | Profiling remeasures under the fixed policy; retain current best |
| Model predicts less traffic but timing is worse | Inspect actual lowering and residency; revise the explanation |
| Correct, stable improvement across the objective | Promote, then reproduce the exact source/configuration |
| Reusable tool/API failure or missing instruction | Prepare a scoped suite update using the maintenance reference |

Update the shared plan with this decision question and route before starting
the next experiment. Reports are handovers for the same agent too; they do
not authorize delegation or additional external actions.

For calibrated searches, attach the profile manifest/raw data, fit and held-out
results, candidate-domain hash, per-access staging table, solver statuses and
selection audit as `model` check artifacts. Keep rejected versus deferred
candidate classifications in the existing ledger. The report checker verifies
file identity and references, not the soundness of solver encodings or timing
bounds. A performance claim still needs the fixed benchmark gate.
