# AGENTS.md — maintaining the TileLang/CUDA skills suite

This repo is the source of a skills suite for TileLang/CUDA kernel work. The
**deliverable is `skills/`** (plus `README.md`, `scripts/`, `requirements.txt`,
and the `external/TileFoundry` submodule pin). Everything under `evals/` is
local-only evidence: gitignored, never shipped, existing only in this working
copy.

## Non-negotiable rules

1. **Delivery boundary.** Tracked files — above all anything in `skills/` —
   must never reference `evals/` paths. Skill docs state conclusions with a
   `(verified YYYY-MM-DD, pin/version)` tag; the evidence behind them lives in
   `evals/` as local records. Before every commit and push:
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

## Environment facts (save yourself rediscovery)

- CLI is `.venv/bin/tilefoundry`; TileFoundry is installed **editable** from
  `external/TileFoundry` (its CUDA backend resolves `runtime.h` and CUTLASS
  relative to a source checkout; the wheel ships neither). cmake/ninja come
  from `requirements.txt`. Init:
  `git submodule update --init external/TileFoundry` then
  `git -C external/TileFoundry submodule update --init third_party/cutlass`.
- Put `.venv/bin` on `PATH` when a test or CLI shells out to the build tools:
  `PATH=$PWD/.venv/bin:$PATH .venv/bin/python -m pytest ...`. Without it,
  `link_modules` fails with `'cmake' not on PATH` even though cmake is
  installed in the venv.
- `tilefoundry --version` prints a bogus shallow-clone artifact
  (`0.1.dev1+g...`). Record the **commit SHA**
  (`git -C external/TileFoundry rev-parse --short HEAD`), never that string.
- The first CLI invocation after a submodule checkout can stall for minutes
  (import/cache warm-up). Known behavior, not a regression — run long commands
  in the background instead of aborting them.
- GPU runs: `CUDA_VISIBLE_DEVICES=0`.
- To test a *different* TileFoundry checkout without reinstalling:
  `PYTHONPATH=<checkout>/src .venv/bin/python -m tilefoundry.cli <args...>`.
- `tilefoundry check`: repeat `--expected` per output file — the comma form
  silently parses as one path.

## Updating the TileFoundry pin (the ritual)

Current pin: `bb3b726` (set 2026-10-23). Run every step; skip nothing.

1. **Survey the delta.** `git -C external/TileFoundry fetch origin`, then
   `git log --oneline <old>..<new>` and `git diff <old>..<new> -- pyproject.toml
   .gitmodules` for dependency/submodule changes. Check what happened to the
   tracked upstream issues/PRs (at this pin: #239, #252, #253 and the
   `warps_per_group` reduce-dispatch bug are fixed by #258/#257; #240 merged;
   #241/#237 closed; the TMA/WGMMA/`Where` emitter gaps remain, tracked as
   #249).
2. **Checkout the new SHA** (detached) in `external/TileFoundry`.
3. **Probe harness:** `./evals/tilefoundry-capability-map/run_probes.sh` →
   expect `12/12 PASS` (exits non-zero on any DIFF). If a DIFF is an upstream
   *improvement* (e.g. better error text), read the new output, update the
   fingerprint in `evals/tilefoundry-capability-map/probes/expected/`, rerun.
   Never edit fingerprints to silence a regression. At `bb3b726` the five
   I-1/I-2 fingerprints were re-derived for the new costing model (wave states
   active/physical, the timeline prices the footprint share, the floor reads
   footprint bytes).
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
     external/TileFoundry/tests/ops/tir/test_copy.py -x -q`
     (cmake/ninja live in `.venv/bin`; without them on PATH the test fails with
     `link_modules: 'cmake' not on PATH`).
5. **Trust-table spot checks** (re-derive after any costing change; at
   `bb3b726` both are unchanged): scan fused (`scan_variants.py:ScanFused`,
   `--dim C=40`, `--performance`) → `ideal-ns=125063`; GEMM scheduled
   (`gemm_compute/hir/scheduled.py:GEMM_4096X4096X4096_OPTIMAL`) →
   `ideal-ns=143256 bound-by=compute`.
6. **Sweep the skills for staleness.** `grep -rn <old-sha> skills/`; also hunt
   behavioral claims the release changed (error texts, emitter status, CLI
   flags). Update pin lines and add the verification date. Keep historical
   entries historical — append a dated paragraph, don't rewrite the record.
7. **Refresh the eval docs** (untracked local records):
   `evals/tilefoundry-capability-map/{ISSUES,REPRO,CAPABILITIES}.md` headers —
   what got fixed, what still reproduces, at which SHA, on what date.
8. **Commit** the gitlink + skill docs. The message records: upstream delta
   (commits, merged PRs), what still reproduces (issue numbers), and the
   verification evidence (probe count, ritual results, trust-table numbers).

## Updating skills from eval findings

- Findings originate in `evals/` task workspaces; fold them into skills as
  *conclusions* (mechanism, command, verification tag) — never as paths.
- Every verification claim carries its date and pin/version. When re-verifying,
  update both; when behavior changes upstream, say what changed and when.
- Cite upstream issues/PRs by number (`tile-ai/TileFoundry#239`) so the next
  bump can check their status quickly.
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
  "fixed at pin X"; a changed model flips a cost claim. Update conclusion,
  date, and pin together, and keep the historical note.
- Record the boundary the new model states, not just its headline: what it
  prices, at which level, against whose rates, and what it still cannot see.
  A conclusion without its model becomes the next stale claim.
- Where the upstream solution differs from what this suite teaches, teach the
  upstream model and state the rejected alternative in one line. The gap is
  the lesson; do not keep our version as if it had won.
- Never cite local evidence paths in the delivered suite; the conclusion, its
  date, and the pin are the citation.

## Trust invariants the skills assert — keep them true

- **predicted-ns is a model, never a verdict.** At the current pin it prices
  the running mesh (`wave=active/physical`) and the footprint's bytes, with a
  per-CTA Little's-law cap over the in-flight bytes the program states; the
  #239 billing and under-fill artifacts are fixed. It still does not measure
  achieved residency, dependent-chain latency, or pipeline overlap. Rank HIR
  variants by logical traffic/FLOP diffs and roofline floors; occupancy-gated
  questions need the measured two-bound recipe plus ncu/torch.profiler or IKET
  warp-lifetime.
- **Commands are the contract.** Captured output in docs/READMEs/examples
  drifts between releases; rerun at the pinned checkout and read what it
  actually prints.
- **TileFoundry `compile()` is not a measured floor.** Emitter gaps remain
  (TMA / WGMMA / `Where`); the `warps_per_group` reduce-dispatch bug is fixed
  at this pin. The supported integration is TileFoundry for
  HIR/check/analyze/schedule/finalize with TileLang kernels implementing the
  finalized TIR.
- **The TileLang runtime-twin pattern** (equal/allclose predicates per output;
  `--fn ulp --max 1` for bf16 reduction-order disagreement) is the correctness
  gate for optimization candidates — keep `tilefoundry check` green after each
  edit.
