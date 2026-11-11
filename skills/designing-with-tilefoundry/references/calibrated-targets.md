# Hardware targets and local calibration

TileFoundry targets provide instruction/capacity contracts and sourced hardware
facts. Keep these separate from attainable device measurements and
implementation-specific corrections. Use the profiling skill's
[two-level calibration workflow](../../tilelang/profiling-tilelang-programs/references/calibration-profiles.md)
when numerical rates affect a runtime decision, even if a shipped profile has
the same architecture name. Pure semantic work or a labeled ideal-reference
comparison can use specification facts without a local GPU run.

## Check identity and active-work coverage

1. Inspect `tilefoundry target show IDENTITY`, the selected target's hardware
   document and the attached device's queried UUID/SM capability/capacities.
   Label the target `local`, `reference` or `proxy`; a neighboring SKU is a
   proxy even when its instructions are useful. A matching capability does
   not establish matching throughput, SM count, cache or power behavior.
2. Inspect modeled work/bytes and enumerate the rates actually needed, by
   dtype and resource. Include every active term; the helper does not derive
   this list from arbitrary HIR. Separate a genuinely zero work term from a
   nonzero term with an unknown rate.
3. Check coverage before interpreting `ideal-ns`, `bound-by` or a timeline.
   Keep raw reports and surface missing/proxy terms in the decision record.
   If missing rates could change the ordering, measure matched cases or
   return an unresolved ranking.

For example, save the active-work requirements as task-local JSON. Replace
the work counts with those of the authored model:

```json
[
  {"fact": "memory.hbm.bandwidth", "work": 4096, "unit": "byte/s"},
  {"fact": "throughput.bf16", "work": 8192, "unit": "flop/s"}
]
```

Fact paths are relative to the hardware document's `facts` table. Use its
actual names/units; the checker does not convert them. Inspect the document
instead of assuming the example paths exist.

```bash
# <profiling-skill> is the resolved installed profiling skill directory.
python <profiling-skill>/scripts/calibration.py target-check \
  path/to/hardware.toml requirements.json --role reference > target-coverage.json
```

TOML requires Python 3.11+; JSON input works with the suite's standard-library
checker too. The command preserves fact provenance, distinguishes zero work,
and returns exit 2 for missing/unavailable/nonpositive rates or unit mismatches.
It checks required-rate coverage, not the complete upstream target schema or
measured predictive accuracy. `prediction_qualified` is always false: even
complete coverage is not local calibration.

The roofline implementation can skip a dtype whose throughput is unavailable
while pricing known memory traffic. Such a report is a **partial ideal
reference**. A printed zero compute time need not mean zero compute work; a
printed memory bottleneck identifies the largest priced term, not necessarily
the actual limiting resource. Preserve the useful partial result together with
its omitted terms. Do not promote it to a complete latency prediction or a safe
pruning bound.

## Use measured data without corrupting target facts

Keep the versioned device profile and module correction alongside the authored
model. Preserve vendor/reference sources, units and conditions. If authoring a
custom hardware document, use the installed vendor schema and documented fact
envelope (`value`, `unit`, `origin`, `source`, `conditions`, or unavailable status);
validate/load it through the supported target interface. Do not edit a shipped
target in place or assume a sidecar is automatically consumed by TileFoundry.

The NVIDIA hardware schema requires L2 bandwidth to remain unavailable. A
measured cache-service curve belongs in the empirical sidecar and a matching
task-level cost model; inserting a numeric L2 bandwidth into that hardware
document is not a supported way to supply it. Likewise, a whole-device
library GEMM rate is not automatically the rate for a scheduled MMA instruction,
and a copy curve is not a per-CTA async-access service rate.

Use TileFoundry for supported work/traffic/placement facts. Compose those facts
with applicable local measurements in the task's
[cost model](../../tilelang-optimization-loop/references/cost-model.md), then
check held-out module cases. Preserve nominal `ideal_reference`, empirical
`predicted_median` and actual timing as distinct fields. If the emitted
implementation changes staging, fusion, instruction family or resource use,
refresh the affected module fit; do not encode that residual as a new hardware
fact. Final candidate selection still uses correctness and the workload's
measured objective.
