# AGENTS.md — maintaining the TileLang/CUDA skills suite

This repo is the source of a skills suite for TileLang/CUDA kernel work. The
**deliverable is `skills/`** (plus `README.md`, `scripts/`, helper `tests/`,
`requirements.txt`, and the `external/TileFoundry` submodule pin). Everything under `evals/` is
local-only evidence: gitignored, never shipped, existing only in this working
copy.

## Non-negotiable rules

1. **Delivery boundary.** Tracked files — above all anything in `skills/` —
   must never reference `evals/` paths. Skill docs state conclusions as
   current behavior; provenance (dates, pins, issue/PR numbers) lives in this
   file and the local records. Before every commit and push:
   `git grep -n "evals/" -- . ':!evals' ':!AGENTS.md'` must print nothing.
   (This AGENTS.md is maintainer-facing and is the one allowed exception:
   it names `evals/` paths because its reader has the full checkout.)
2. **Staging discipline.** Stage explicit paths only, and read `git status`
   first — a careless `git add` of an evals path once half-staged `.gitignore`.
   A TileFoundry pin bump commits exactly: the `external/TileFoundry` gitlink
   and the touched skill docs.
3. **Never track TileFoundry main blindly.** The pin is the gitlink
   (`.gitmodules` cannot hold a commit hash; do not add `branch =` — that
   enables `--remote` tip-tracking, the opposite of pinning). Bump
   deliberately, re-verify, then commit. Upstream has broken its own flagship
   example within two days.
4. **No changelog in the suite.** Delivered skills state current behavior only.
   Do not add commit hashes, PR/issue numbers, "bumped on" paragraphs, or
   verification-date walls to `skills/`. Write the delta in the commit message
   instead: what changed upstream, what was re-verified, and what still
   reproduces. This file keeps the current pin and the tracked-issue status.
   Keep unmerged work and branch-specific exclusions here or in local audit
   records; they do not belong in shipped skill instructions.

## Environment facts (save yourself rediscovery)

- CLI is `.venv/bin/tilefoundry`. Install the pinned core and NVIDIA package
  together: `uv pip install --python .venv/bin/python -e external/TileFoundry
  -e external/TileFoundry/targets/nvidia`; add `targets/apple` for all-target
  upstream tests. Runtime headers ship with the owning package; CUDA builds
  additionally need CUTLASS and nvcc. Initialize the submodule recursively.
  Old repository-root `tilefoundry.egg-info` must be removed before migrating
  to the split editable packages. Keep cmake/ninja on `PATH`.
- `scripts/install-skills.sh` also ships TileFoundry: it checks out the pinned
  submodule (recursive, including CUTLASS) in checkout and cache modes and
  prints the source path and install commands. It never updates TileFoundry to
  upstream main; the gitlink is the pin.
- Put `.venv/bin` on `PATH` when a test or CLI shells out to the build tools:
  `PATH=$PWD/.venv/bin:$PATH .venv/bin/python -m pytest ...`. Without it,
  `link_modules` fails with `'cmake' not on PATH` even though cmake is
  installed in the venv.
- Record the git SHA and package versions. Shallow git metadata can mislabel
  the release; all editable vendor packages must match the core checkout.
- The first CLI invocation after a submodule checkout can stall for minutes
  (import/cache warm-up). Known behavior, not a regression — run long commands
  in the background instead of aborting them.
- GPU runs: `CUDA_VISIBLE_DEVICES=0`.
- To test a different TileFoundry checkout, use a separate environment with
  matching core/vendor editable installs. A src-only `PYTHONPATH` leaves the
  installed vendor entry points and code active and is not a valid pin switch.
- `tilefoundry check`: repeat `--expected` per output file — the comma form
  silently parses as one path.

## Updating the TileFoundry pin (the ritual)

Current pin: `3a6a4c0` (audited 2026-11-02). Run every step; skip nothing.

1. **Survey the delta.** `git -C external/TileFoundry fetch origin`, then
   `git log --oneline <old>..<new>` and `git diff <old>..<new> -- pyproject.toml
   .gitmodules` for dependency/submodule changes. Check what happened to the
   tracked upstream issues/PRs. At this pin #260 closes MLA/layout gaps;
   #265/#268 split core/vendor packages; #269 changes CLI/finalize loading;
   #271/#273/#277 fix launch sizing, full windows and ring bindings;
   #279 changes AMX/facts contracts; #280/#282 centralize emission decisions;
   #283 adds opt-in sync planning, without CUDA synchronization emission.
   #249 (WGMMA emission), #278 (all-Broadcast normalization), and our #284
   (FSDP/HSDP HIR modeling) remain open. Do not teach #284 as pinned main.

2. **Checkout the new SHA** (detached) in `external/TileFoundry`, then
   `git -C external/TileFoundry submodule update --init --recursive
   third_party/cutlass` so the CUDA backend has its headers.
3. **Probe harness:** `./evals/tilefoundry-capability-map/run_probes.sh` →
   expect `12/12 PASS` (exits non-zero on any DIFF). If a DIFF is an upstream
   *improvement* (e.g. better error text), read the new output, update the
   fingerprint in `evals/tilefoundry-capability-map/probes/expected/`, rerun.
   Never edit fingerprints to silence a regression. On API changes first
   migrate local fixtures with backups. The current compiler probe loads a
   freshly finalized Module; the obsolete bare-PrimFunction error is no longer
   useful. It now reaches an unsupported TupleGetItem index expression.
   Numerical cost fingerprints remain unchanged.
4. **Pin ritual** (proves the user-facing workflow still works):
   - Extract the tutorial programs with the awk block from
     `external/TileFoundry/docs/tutorial/optimize.md`, then
     `tilefoundry analyze rms_norm_quant.py:Fused out.txt --compute-cost
     --memory --roofline` → expect `ideal-ns=3755` (the tutorial now runs 132
     CTAs; the old 13 was the pre-#258 costing).
   - One runtime-twin check, e.g. from
     `evals/tilefoundry-coldstart-20261019/cases/scan_occupancy`:
     `tilefoundry check hir/twin_tilelang.py:FastTileLang --dim C=40
     --inputs files:refdata/in0.pt,refdata/in1.pt --expected refdata/expected0.pt
     --expected refdata/expected1.pt --out 'output[0]' --fn allclose --atol 0.01
     --rtol 0.0078 --out 'output[1]' --fn allclose --atol 0.01 --rtol 0.0078`
     → PASS both outputs.
   - One upstream test:
     `PATH=$PWD/.venv/bin:$PATH .venv/bin/python -m pytest
     external/TileFoundry/tests/nvidia/cuda/ops/tir/test_copy.py -x -q`
     (cmake/ninja live in `.venv/bin`; without them on PATH the test fails with
     `link_modules: 'cmake' not on PATH`).
5. **Trust-table spot checks** (re-derive after any costing change; at
   `3a6a4c0` both are unchanged): scan fused (`scan_variants.py:ScanFused`,
   `--dim C=40`, `--performance`) → `ideal-ns=125063`; GEMM scheduled
   (`gemm_compute/hir/scheduled.py:GEMM_4096X4096X4096_OPTIMAL`) →
   `ideal-ns=143256 bound-by=compute`.
6. **Sweep the skills for staleness.** `grep -rn <old-sha> skills/`; also hunt
   behavioral claims the release changed (error texts, emitter status, CLI
   flags). Skills state current behavior only: rewrite the claim and drop any
   commit hash, PR number, or bump paragraph. The pin and its verification
   dates live in this file and the local eval records; the delta belongs in
   the commit message.
7. **Refresh the eval docs** (untracked local records):
   `evals/tilefoundry-capability-map/{ISSUES,REPRO,CAPABILITIES}.md` headers —
   what got fixed, what still reproduces, at which SHA, on what date.
8. **Commit** the gitlink + skill docs. The commit message is the changelog:
   upstream delta (commits, merged PRs), what changed in behavior, what still
   reproduces (issue numbers), and the verification evidence (probe count,
   ritual results, trust-table numbers).

## Suite validation and stage evidence

- Run `python3 -m unittest discover -s tests -v`, shell syntax checks, and
  `python3 scripts/skill_layout.py check skills` after helper/routing edits.
- Exercise a disposable flattened copy and symlink install. Check-only must
  remain offline/read-only and return nonzero for missing or stale files.
- Use the loop's stage report for bounded handovers, referencing the existing
  candidate ledger. Validate hashes and references, then review the findings;
  the helper cannot prove execution or measure agent skill-use behavior.
- Maintenance findings need a reproducer, owning reference, applicability scope
  and a nearby case that should not trigger the advice. Keep raw evidence local.
- Current audit/probe logs and fixture backups are in
  `evals/suite-update-20261102`; capability-map docs retain the source history.
- Calibration/solver audit (2026-11-02): repo issue #1 and its comments motivate
  per-access staging, conditional microbenchmarks and held-out search recall.
  The current pin already has per-Call in-flight estimates; the older claim
  that staging facts are absent is stale. Existing stream/GEMM helpers measure
  offered grid work and complete kernels, not achieved residency/isolated dots.
  TileFoundry uses CP-SAT for finite predicates and first-feasible allocation,
  and ISL for integer relations; installed TileLang supplies Z3 4.15.4.
  CPU probes cover ownership collisions, finite-domain agreement and a separate
  optimal packing example. No general tensor solver or calibration dataset is
  claimed. Sources, probes and raw results live in
  `evals/calibration-solver-20261102`.

## Updating skills from eval findings

- Findings originate in `evals/` task workspaces; fold them into skills as
  *conclusions* (mechanism, command, current behavior) — never as paths.
- Provenance (verification date, pin, upstream issue/PR numbers) lives in this
  file and the local eval records, not in delivered skills. Skills state what
  the tool does now; they do not carry a changelog. Write the changelog in the
  commit message.
- `scripts/install-skills.sh` symlinks skills into `~/.agents/skills` (and
  mirrors for other CLIs), so content edits go live immediately; rerun the
  installer only after adding or renaming a skill directory. KernelWiki is a
  standalone skill at `~/.agents/skills/KernelWiki`, invoked by name — the
  suite's interface to it is
  `skills/tilelang-optimization-loop/references/kernelwiki.md`.

## Contributing fixes to upstream projects

Upstream work is a model negotiation, not a patch delivery. Maintainers own
the cost model, the output contract, and the vocabulary; most of what a
contributor gets wrong is not arithmetic but the unit of account. Rules that
survive a take-over:

1. **Diagnose the model, not the number.** Ask what quantity is priced, at
   which level, against whose rates, and what one unit of it means. A patch
   that divides a shared cost by a position count fixes the reported instance;
   a change that makes the model's unit of account truthful fixes the class.
   Prefer the second and say in the PR which one it chose.
2. **Read the normative spec before the code.** Where the project states MUSTs
   (formats, formulas, rates), the fix must satisfy them. If it cannot without
   changing a MUST, that is a design proposal: state the old contract, the new
   one, and why, and expect the maintainer to decide. Keeping "make the code
   match the spec" separate from "change the spec" is what keeps a change
   reviewable.
3. **Grep the vocabulary and conventions first.** Before adding a field, flag,
   wrapper, or unit suffix, find how the project already spells it: record
   forms, help text versus argument parser, binary prefixes, report layouts. A
   new field on a normative line is a format change, not a bug fix.
4. **Audit duplicate paths.** When one concept is bound, rendered, or formatted
   in two places, unify at the seam (one instantiation, one formatter, one
   renderer) instead of patching the path that failed and threading a new
   parameter through its callers.
5. **Reproduce at deployment scale, and sweep the dimension.** Size repros to
   what the hardware runs (one wave of physical units, the real mesh
   hierarchy), and vary the dimension the bug lives in — participation, waves,
   tile counts — not only the reported instance. Latent bugs surface at scale.
6. **Keep changes separable.** One concern per commit and per PR; model
   changes apart from plumbing; a test that fails before and passes after. A
   maintainer may take over a model-level change and keep part of the branch;
   separable commits survive that, a monolith does not.
7. **Read the merge, not just the approval.** When a change is taken over or
   rewritten, diff the branch against the merged state and extract every
   model, contract, and format decision the maintainer made. That delta is the
   most valuable review the work will get; record it.
8. **Verification is part of the contribution.** Scoped tests, sanitizer on
   the touched kernels, the full suite compared against the base failure set,
   project hooks, and an explicit list of environment-only failures. Re-run
   after every rebase; never present an unverified tree. In the PR text, state
   before/after evidence and stay within the project's template.
9. **Match the layout to the medium.** A commit message is read in `git log`,
   email, and patches: hard-wrap its prose at the conventional column and keep
   a blank line after the subject. A PR description, issue, or review reply is
   rendered by the forge, which does its own wrapping: write each paragraph as
   one unbroken line, with blank lines between paragraphs and list markers for
   parallel items. Manual hard wraps there surface as ragged short lines, and
   extra breaks read as structure the author did not mean.
10. **Follow the project's review mechanics.** Rebase and force-push with a
    lease, answer each thread with evidence, resolve only threads actually
    addressed, and never merge your own change.

## Folding upstream rewrites into this suite

- After a take-over or a model rewrite, sweep every claim in `skills/` that
  asserted the old behavior. A merged fix flips a "known bug" invariant into
  "fixed"; a changed model flips a cost claim. Update the conclusion to the
  current behavior; the pin, date, and delta stay in this file and the local
  records.
- Record the boundary the new model states, not just its headline: what it
  prices, at which level, against whose rates, and what it still cannot see.
  A conclusion without its model becomes the next stale claim.
- Where the upstream solution differs from what this suite teaches, teach the
  upstream model and state the rejected alternative in one line. The gap is
  the lesson; do not keep our version as if it had won.
- Never cite local evidence paths in the delivered suite; the conclusion
  stands on its own, and its provenance is recorded here.

## Trust invariants the skills assert — keep them true

- **predicted-ns is a model, never a verdict.** At the current pin it prices
  the running mesh (`wave=active/physical`) and the footprint's bytes, with a
  per-CTA Little's-law cap over the in-flight bytes the program states; the
  #239 billing and under-fill artifacts are fixed. It still does not measure
  achieved residency, dependent-chain latency, or pipeline overlap. Rank HIR
  variants by logical traffic/FLOP diffs and ideal references; occupancy-gated
  questions need matched calibration plus resource reports and ncu/torch.profiler
  or IKET warp-lifetime. Empirical predictions are not proven runtime bounds.
- **Commands are the contract.** Captured output in docs/READMEs/examples
  drifts between releases; rerun at the pinned checkout and read what it
  actually prints.
- **TileFoundry `compile()` is not a measured floor.** Current tutorial
  finalization succeeds, but compilation reaches an unsupported TupleGetItem
  index expression. Other emitter gaps remain; `sync` plans ordering without
  emitting CUDA waits/barriers. Copy/launch GPU tests do not prove an entire
  program correct. Keep HIR, finalization, runtime correctness and timing as
  distinct evidence gates.
- **The TileLang runtime-twin pattern** (equal/allclose predicates per output;
  `--fn ulp --max 1` for bf16 reduction-order disagreement) is the correctness
  gate for optimization candidates — keep `tilefoundry check` green after each
  edit.
