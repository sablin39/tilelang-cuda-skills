# A candidate record that explains the decision

Use the task's existing schema when available. The example below is a
**hypothetical rejected experiment**, not a measured kernel result. Store
machine-readable fields alongside artifact paths so another worker can check
the decision without reconstructing the conversation.

```json
{
  "id": "c02",
  "parent": "c01",
  "source": "candidates/c02.py",
  "source_sha256": "<hash of the measured source>",
  "config": {"num_stages": 3},
  "hypothesis": "An additional pipeline stage will reduce copy/compute waiting.",
  "expected_lowering": "Additional shared buffers with the corresponding waits/reuse protocol.",
  "observed_lowering": "Buffering increased; see runs/c02/device.cu and lowering.txt.",
  "research": {
    "wiki_revision": "<queried checkout commit>",
    "page_id": "technique-pipeline-stages",
    "source_locator": "<actual upstream file or documentation used>"
  },
  "correctness": {"status": "pass", "command": "<exact validation command>", "log": "runs/c02/check.log"},
  "measurement": {
    "command": "<exact benchmark command>",
    "scope": "complete operator with output allocation",
    "backend": "event",
    "unit": "ms",
    "results": "runs/c02/timings.json"
  },
  "model": {
    "ideal": "<saturated bound: analyze ideal-ns or measured rates>",
    "real": "<measured two-bound recipe>",
    "measured_over_floor": "<measured / real>"
  },
  "decision": "reject",
  "reason": "Validation passed, but the aggregate workload objective regressed beyond measured variation.",
  "next": "Keep c01; investigate whether the added storage reduced residency before increasing stages again."
}
```

Use observed values in the referenced per-case timing results. Keep baseline,
current best, and candidate measurements distinguishable. Record compile/runtime
errors as failures, and missing execution as unrun/blocked; do not write dummy
latency values or a passing status for those cases. An unchanged source file
with a different configuration is still a distinct candidate identity. When a
stage was modeled, record both bounds and `measured / floor`; the
[cost model](cost-model.md) defines them.

The plan answers **what to try next and why**. The candidate record answers
**what actually happened and why it was kept or rejected**. A successful
mechanism change may still lose on complete-operator latency or workload mix.
