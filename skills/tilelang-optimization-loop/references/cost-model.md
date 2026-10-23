# A cost model for occupancy-gated decisions

`analyze` ranks designs on logical traffic, capacity advisories, and a roofline
floor. Its `ideal-ns` is a saturated-machine bound: footprint bytes at the
target's published rate, arithmetic at the published peak, and a per-CTA
bandwidth cap from the in-flight bytes the authored program states. That is the
right bound for what a program could reach. When a decision turns on what the
attached device actually sustains — resident CTAs per SM, a dependent chain, a
staging path — build a small per-stage model on measured constants. The
benchmark stays the verdict.

## When to build one

- Two candidates' static margin is inside measurement variation, or the static
  ranking and a measurement disagree.
- Candidates differ in residency (smem, registers, threads), pipeline depth, or
  staging path, not only in logical bytes.
- A stage measures far from its floor and you need to know which term binds:
  work, bandwidth, residency, or a serial chain.

Do not build one for an isolated, already-defined primitive; measure that
directly. Do not let the model decide a promotion; it explains a measurement.

## Two bounds per stage

| Symbol | Meaning |
| --- | --- |
| `SM` | Streaming multiprocessors of the attached device |
| `smem_budget`, `reg_budget` | Shared-memory and register capacity one SM makes available |
| `smem_per_cta`, `threads`, `regs_per_thread` | The candidate's request |
| `r` | Residency the resources admit: `min(smem_budget / smem_per_cta, reg_budget / (threads * regs_per_thread))` |
| `r_eff` | Resident CTAs per SM actually achieved (`r_eff <= r`) |
| `share` | `max(1, r_eff)`; resident CTAs serialize on the SM's compute |
| `ctas`, `waves` | Grid size and `ceil(ctas / (SM * r_eff))` |
| `bw_sm` | Global-memory bytes/s one SM sustains at `r_eff` |
| `bw_sm_saturated` | The most global-memory bandwidth one SM sustains, from the sweep's plateau |
| `inflight_per_warp` | Bytes one warp keeps in flight for the access pattern |
| `warps_per_cta` | Warps one CTA runs |
| `latency` | End-to-end global-memory latency seen by that pattern |
| `dots` | One CTA's serialized dot time in one step: `sum(t_dot(shape))`, or the step's analytical compute time when no per-dot measurement exists |
| `chain` | Dependent-latency cost in one step beyond the dots: barriers, dependent loads. A serialized dot belongs in `dots`, not here |
| `t_dot(shape)` | One small GEMM's time on one SM at that shape |
| `steps` | Serial steps each CTA runs at this stage |
| `bytes` | Logical bytes the stage moves at the bandwidth level |

```
ideal: t_stage = waves * steps * max( share * dots , bytes / bw_sm_saturated )
real:  t_stage = waves * steps * max( share * dots , chain , bytes / (bw_sm / r_eff) )
real:  bw_sm    = min( bw_sm_saturated , r_eff * warps_per_cta * inflight_per_warp / latency )
```

`share * dots` is the per-step compute: `dots` measured with one CTA on one SM,
multiplied by the CTAs that share that SM. `waves` divides the grid among
resident slots, so a compute-bound stage stays invariant to residency, as it
should. Report both bounds; `real / ideal` is the part of the gap the static
bound cannot see.

Apply the model to a fusion boundary by comparing the products, not the
traffic: a direction that serializes what the other spreads across waves shows
up as `steps` multiplying `dots` on one side and `waves` dividing it on the
other. The case that motivates this recipe is exactly that inversion — a fused
stage with a long dependent chain against a split stage with many waves — and
traffic totals alone cannot see it.

Before a candidate exists, `r_eff` is not measurable. Run the model with the
resource-admitted `r` (smem usually sets the upper bound; registers can lower
it) and with `t_dot` at the candidate's shapes, and label the result an
estimate. Replace `r` with the measured `r_eff` once the kernel exists. A
ranking that flips between the estimate and the measurement is a signal to
measure the binding term, not evidence for either side.

## Measure every constant on the attached device

No value from a document, a previous task, or this skill is valid until it is
measured on the device the candidate will run on. Record the command, date,
device identity, and shape or configuration with every constant.

| Quantity | How to obtain |
| --- | --- |
| `SM`, `smem_budget`, `reg_budget`, published rates | Device query plus the architecture's hardware pages in KernelWiki or the vendor documentation; label these ideal |
| `bw_sm_saturated`, `bw_sm(r)` | `stream_sweep.py`: a copy kernel at several grid sizes, one point per resident CTAs/SM |
| `latency` | A dependent-load chain over a buffer larger than the last-level cache, one warp, pointers shuffled so each load depends on the previous; report the plateau. Or use the published latency for the ideal bound and say so |
| `inflight_per_warp` | The emitted staging depth (stages x bytes per stage / warps) for an asynchronous pipeline; the widest per-thread access for a synchronous one. Corroborate against the knee of the stream sweep |
| `t_dot(shape)` | `gemm_fit.py`: one barrier-separated small GEMM on one SM at the candidate's actual dot shapes |
| `dots` | Sum `t_dot` over the dots one step runs; when no per-dot measurement exists, use the step's analytical compute time |
| `r_eff` | Per-CTA intervals from IKET warp-lifetime tracing, or a grid sweep of the candidate where wall time stops improving |
| `chain` | Dependent cost in one step beyond the dots, measured on a stage whose staging is already good; re-derive it after any staging change. A serialized dot stays in `dots` |

Rules that keep the model honest:

- **Never credit a merged call.** A per-dot constant is a shape cost, not a
  per-call overhead; a merged dot is the sum of its parts until a measurement
  says otherwise.
- **Re-derive the chain term after a staging change.** Better staging can put
  the measured per-CTA time below the term that absorbed the old latency.
- **Do not infer registers from the binary when the tool is absent.** TileLang
  may leave its kernel resource fields empty; infer residency from the sweep or
  the trace instead of guessing a per-thread register count.
- **State which latency is which.** A published pointer-chase latency and the
  achieved latency behind a staging path are different quantities; the model
  uses the one that matches the access pattern.

## Templates

`stream_sweep.py` — measures achieved streaming bandwidth against resident
CTAs per SM. Query the device at runtime; no rate is embedded.

```bash
python stream_sweep.py --size-mib 512 --threads 256 --resident 1,2,4,8,16
```

`gemm_fit.py` — times one barrier-separated small GEMM on one SM at each
requested shape and prints the per-shape cost.

```bash
python gemm_fit.py --shapes 64x64x64,128x128x64 --threads 256 --dtype float16
```

**Occupancy.** Use the external IKET profiler's warp-lifetime mode when it is
available: it patches kernel entry and exit in the cubin, so per-warp start and
end timestamps yield per-CTA intervals and achieved concurrency for any
compiled kernel, TileLang included. Body tracing (`iket.mark`,
`range_start`) is a different feature: it needs source instrumentation and
exists only for CuTe-DSL sources, and an uninstrumented cubin yields an empty
body trace. The suite's
[IKET reference](../../tilelang/profiling-tilelang-programs/references/iket-profiling.md)
covers the TileLang frontend; warp-lifetime is the external profiler's mode,
not a TileLang namespace. Without the profiler, sweep the candidate's grid
below and above `SM * r` and take the plateau where more CTAs stop improving
wall time as `r_eff`; report which method produced the number.

The two scripts print measurements; neither ships a default rate, latency, or
shape. If a template's output disagrees with the generated code, the generated
code wins — inspect it before trusting the number.

## Validate before ranking

- Fit on one set of shapes or configurations and report **held-out residuals**
  on another. A model fitted on the measurements it reproduces is not
  validated; the residuals are the evidence.
- Record `measured / floor` per stage in the candidate record. It separates
  "this shape is slow" from "this kernel leaves performance on the table", and
  it is how a stage that regressed against its own floor is found.
- State the regime. The constants are calibrated for one staging path and one
  residency regime; crossing either invalidates them, and the model is
  in-regime evidence, not a universal law.
- When the model and the benchmark disagree, the benchmark wins. Report the
  model's error; do not adjust the measurement.

## Tie the measured bound to the static report

- `ideal-ns` is the saturated-machine bound; `bound-by` names the resource
  class. A `bound-by: memory` floor says which term binds, not what the kernel
  achieves.
- `wave=active/physical` states the static fill. Price the wave, not the grid:
  `waves = ceil(ctas / (SM * r_eff))`, so a partially filled tail wave appears
  before it is measured. KernelWiki's `patterns/tail-effect` states the rule.
- The report's per-CTA bandwidth cap is Little's law over the in-flight bytes
  the authored program states, at a published latency. It does not measure
  residency; use the calibrated layer above when residency is the question.
