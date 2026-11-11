# Reusable device profiles and iterative module calibration

Read [initial device collection](#collect-the-initial-device-evidence),
[reuse checks](#reuse-without-silently-accepting-drift),
[module iteration](#calibrate-a-module-then-refresh-only-what-changed), or
[the evidence contract](#evidence-format-and-handoff) for the current decision.

At the first performance pass, query the attached device and look for matching
calibration evidence. Measure locally before using an uncertain rate to rank
candidates. A shipped target, a GPU name, or an architecture match is insufficient:
unit counts, runtime, clocks, partitioning and attainable rates can differ.
Calibrate only terms that can change the next decision. An API/semantic question
or an explicitly ideal comparison does not need GPU measurements.

Keep two levels of evidence in the task workspace:

| Level | Reusable content | Applicability |
| --- | --- | --- |
| Device/runtime | Queried capacities; sourced reference facts; repeated attainable rates for declared probe regimes | Exact device/environment and matching dtype, path, size, cache, concurrency and timing policy |
| Module/implementation family | Base-profile reference, exact sources, workload/configuration, fitted corrections and held-out observations | Declared module regime; refresh affected terms when its implementation or workload changes |

Never replace a shared device rate with a correction for a slow module. Store
raw samples, not just coefficients. A fast measured copy or GEMM is an attained
rate for that implementation, not a universal hardware ceiling or a proven
latency lower bound.

## Collect the initial device evidence

`<profiling-skill>` is the installed directory containing this skill's
`SKILL.md`; run commands in the task root using the task's Python environment.
The JSON checker is standard-library only; collecting these starter probes
needs PyTorch with CUDA. No extra package is required by these helpers.

```bash
python <profiling-skill>/scripts/measure_calibration.py device \
  --root . --output calibration/device-v1.json --isolation exclusive \
  --probes copy gemm --copy-mib 16 64 256 --gemm-size 1024 4096 \
  --dtype bfloat16 --warmup 5 --iterations 20 --repeats 9
python <profiling-skill>/scripts/calibration.py --root . \
  check calibration/device-v1.json
```

These sizes are examples, not recommended saturation points. Select the visible
ordinal with `--device`; retain device UUID and compute capability from queries.
Use `--isolation shared` or `unknown` when appropriate; the flag records a claim,
it does not reserve the GPU. Serialize GPU work through the task's scheduler.
Record `--clock-policy` and `--power-policy` when controlled; defaults are
`uncontrolled`. The collector does not alter clocks, power or competing work.

The collector saves device/runtime identity, observable telemetry before/after,
tool source snapshots, timing policy, logical work and raw repeated samples.
Missing telemetry remains unavailable. Add task/compiler/build manifests using
repeatable `--input task-relative-path`; installed package versions alone do
not identify a dirty checkout. Record partition/MPS/ECC or other relevant
conditions there when runtime queries do not expose them.

| Starter probe | What it measures | What remains unknown |
| --- | --- | --- |
| Copy | Contiguous `torch.copy_`, logical read+write bytes per elapsed second, across sizes | Actual DRAM sectors, cache residency, TMA/cp.async service, achieved occupancy |
| Dense GEMM | Complete `torch.mm` by shape and dtype, counting `2*M*N*K` FLOPs | Isolated MMA latency, chosen machine instruction, one-CTA service, all other shapes/dtypes |

Both use warmed ordinary launches, repeated buffers, batched CUDA-event samples
and no explicit cache flush. Setup allocation is excluded. Copy checks the
whole output; GEMM checks sampled output rows/columns against CPU FP64. These
library sanity checks do not replace task-kernel correctness. Small working
sets may measure cache service; **do not rename the copy result “HBM bandwidth”**
without checking that regime. GEMM may be underfilled. Add focused probes using
the [measurement recipes](microbenchmark-calibration.md) when those gaps matter.
The existing [copy/grid and single-CTA GEMM probes](../../../tilelang-optimization-loop/references/cost-model.md)
answer different questions and remain useful.

For spec/KernelWiki inputs, follow the cited source and retain URL/section,
SKU/architecture, precision, sparsity, clocks and byte convention. A task-local
JSON object can be passed as `--reference-facts references.json`:

```json
{
  "reference_dense_bf16_rate": {
    "status": "unavailable",
    "conditions": "No applicable source or matching local probe yet"
  }
}
```

An available fact instead has `value`, `unit`, `origin`, `source`, `conditions`.
Reference import accepts `vendor`, `reference`, `derived`, `estimated`; it cannot
claim a local measurement or overwrite runtime facts. Wiki is a discovery
source, recorded as `reference` with the underlying citation. Use `unavailable`
with a reason when a number is absent. Do not fill it with zero, a neighboring
SKU's rate, or a percentage of an advertised peak. Such proxies can inform
sensitivity analysis when explicitly labeled; directly measure candidates
whose ranking depends on the unresolved value.

## Reuse without silently accepting drift

Share the device JSON together with its referenced files, retaining relative
paths, and check it after copying into another task root. A JSON hash validates
identity, not the original observation or current machine. Query the current
context and collect fresh anchors for the **needed scopes** with the same
protocol; choose a new output path rather than overwriting evidence.

```bash
# Same command/policy and matching probe sizes as the earlier profile:
python <profiling-skill>/scripts/measure_calibration.py device \
  --root . --output calibration/device-v2.json --isolation exclusive \
  --probes copy gemm --copy-mib 16 64 256 --gemm-size 1024 4096 \
  --dtype bfloat16 --warmup 5 --iterations 20 --repeats 9
python <profiling-skill>/scripts/calibration.py --root . reuse \
  calibration/device-v1.json calibration/device-v2.json \
  --max-relative-drift 0.05 --max-relative-spread 0.10 \
  > calibration/device-reuse-v1-v2.json
```

Set drift and spread tolerances before running, based on decision sensitivity;
the example values are not hardware constants. Spread is `(p90-p10)/median`,
using sample quantiles, **not** a confidence interval. Matching UUID, driver,
context, declared inputs and timing protocol are required. Results are per row:
`reusable`, `drifted`, `inconclusive`, `unverified` or `context_changed`. At least
three samples and acceptable spread are needed. A matching copy anchor does
not refresh an unmeasured compute or latency row. Exit 2 can mean partial reuse:
read the individual decisions and use only the eligible scopes.

On drift, first inspect contention, clocks, thermal state and protocol. Repeat
under the declared conditions before interpreting it as a device change. New
driver/compiler, partition, power policy or source manifests require current
evidence; a new module schedule alone does not invalidate all device probes.

## Calibrate a module, then refresh only what changed

1. Name the decision and module regime: workload distribution, dtype/layout,
   instruction/copy path, fusion and saved-state boundaries, cache reuse,
   resources, staging/waits and timing scope. Reference the applicable device
   profile. Record unknowns rather than borrowing an incompatible rate.
2. Freeze predictions, fit/holdout assignments and acceptance limits before
   measuring. Hold out **workload/configuration pairs**, not repetitions of
   identical pairs. Include boundaries and alternative implementations; compare
   candidates on the same workload. For packed sequences, include layout
   distribution/seeds and fresh layouts in correctness and holdout checks.
3. Validate each implementation, collect raw timing samples, and inspect the
   lowering/resources that explain a discrepancy. Include reset, conversion,
   state and allocation costs belonging to the chosen scope.
4. Fit only the fit partition and assess held-out errors and selection regret.
   If evidence is noisy or ranking is unresolved, measure more or retain the
   candidate as unknown. A model never replaces the final workload benchmark.
5. After a change, create a new module version with `parent` and
   `refresh_reason`; retain the old sources and results. If a holdout influenced
   the next model, select new holdouts for its validation.

| Change or finding | Required response |
| --- | --- |
| Same source/context and covered workload | Reuse matching module evidence; check anchors when conditions may have drifted |
| Different fusion, instruction family, copy/wait/staging, resource use or saved-state policy | Refresh affected module cases/terms; preserve still-applicable device measurements |
| New shape/layout distribution, cache transition, underfill or tail regime | Add boundary probes and held-out full-module cases; do not extrapolate confidently |
| Held-out error or selection regret exceeds limits | Inspect work counting, lowering, dependencies and scope; revise the model and reconsider deferred candidates |
| Device/runtime identity or anchor drift changes | Establish the current base profile, then revalidate dependent module fits |

### Runnable module example and factory interface

Copy the [pointwise example](pointwise-calibration.py) into the task root. It
compares fused and separate implementations using copy bandwidth as an initial
hypothesis; this model may fail validation on the attached device.

```bash
cp <profiling-skill>/references/pointwise-calibration.py ./module_calibration.py
python <profiling-skill>/scripts/measure_calibration.py module \
  --root . --factory module_calibration.py --base calibration/device-v2.json \
  --output calibration/pointwise-v1.json --reason initial --isolation exclusive
python <profiling-skill>/scripts/calibration.py --root . assess-module \
  calibration/pointwise-v1.json --current-inputs --require-ranking \
  --max-relative-error 0.15 --max-selection-regret 0.05 --max-relative-spread 0.10 \
  > calibration/pointwise-v1-assessment.json
```

The factory exports `calibration_cases(device_profile)` returning `module`,
`regime` and `cases`. Each case supplies `id`, `group`, `partition` (`fit` or
`holdout`), nonempty `workload`/`configuration` objects, positive `predicted_ns`
and `setup()`. Setup compiles/allocates outside timing and returns `run()` and
`check()`. Check verifies the contract, exercises the implementation and returns
`True` or raises. Run performs the measured operation on the current stream;
it must join any additional streams and include required resets. Arrange case
order deliberately and repeat with counterbalanced orders to check drift.
Predictions/metadata are copied before timing starts. Include imported kernel,
model and harness sources with repeatable `--input`; automatic dependency
discovery is not claimed. Sources are preserved by content hash.

`assess-module` fits one multiplicative correction per declared group using
the median of `observed/predicted` on fit rows only. It reports held-out relative
errors and compares corrected predictions across groups for identical workloads.
Regret is `chosen_time/best_observed_time - 1` among those candidates. Overlapping
sample ranges leave selection inconclusive. This small diagnostic does not
replace a domain-specific model, prove causal attribution, estimate top-K recall,
or establish global optimality. `--require-ranking` prevents a latency-only
check with no comparable candidates from qualifying a search decision. Inspect
coverage and thresholds, not just exit status.

To refresh, change the factory/inputs and collect to a new path with
`--parent calibration/pointwise-v1.json --reason 'changed the staging path'`.
Use a truthful reason. The parent is provenance, not automatically merged data;
the new profile declares its own current cases and validated regime. The device
base remains unchanged if still applicable. `--current-inputs` detects later
source changes while historical checks continue to use preserved snapshots.

For wall-clock training steps, graph replay, cold-cache probes or other timing
policies, use the task's matching benchmark and emit the same records through
the adjacent `calibration.py` functions (`snapshot_input`, `fingerprint`,
`seal`). The starter collector measures current-stream CUDA events; it cannot
establish host/distributed end-to-end throughput. Keep prediction checking and
the task's promotion benchmark separate.

## Evidence format and handoff

Both profile kinds use `schema_version: 1`, a canonical-content `profile_id`,
creation time, executed command, `context` (`device`, `software`, `operating`),
`protocol`, source `inputs` and optional artifact fingerprints. A device profile
has `facts` and `measurements`; a module profile adds `module`, `regime`,
`refresh_reason`, `base_profile`, optional `parent`, and `cases`. The runnable
collector/example shows the full format. References are task-relative
`{path, sha256}` objects; source inputs retain both the live `source_path` and
an immutable `artifact` snapshot. Hashes are generated, never invented.

Measured rows require passing correctness, raw positive samples and recomputed
summaries. Device rows use `quantity`, `unit`, `scope`, `samples`; module rows
use `samples_ns` plus the frozen design fields above. Failed/unavailable rows
retain a reason and no eligible rate. The collector retains per-case setup,
correctness or timing failures and exits 2; fix and create a new version.
Setup errors before a case manifest exists remain command failures and need
their log attached to the existing stage report. Process termination cannot
guarantee a final profile, so retain command logs in the task harness too.

Checker exit codes are 0 for the requested successful check, 2 for incomplete
coverage/inconclusive results, and 1 for invalid evidence or command errors.
`check` verifies structure/content/files, not execution, statistical independence,
truth of provenance or applicability to an untested regime. Link profiles,
assessment JSON and raw/lowering artifacts as `model` evidence in the existing
[stage report](../../../tilelang-optimization-loop/references/stage-handoff.md).
Do not introduce a second experiment ledger. For TileFoundry inputs, also use
the [target coverage check](../../../designing-with-tilefoundry/references/calibrated-targets.md).
