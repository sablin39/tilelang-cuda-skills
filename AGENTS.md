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

Current pin: `25fbbaa` (set 2026-10-20). Run every step; skip nothing.

1. **Survey the delta.** `git -C external/TileFoundry fetch origin`, then
   `git log --oneline <old>..<new>` and `git diff <old>..<new> -- pyproject.toml
   .gitmodules` for dependency/submodule changes. Check what happened to the
   tracked upstream issues/PRs (currently: #239 open = the predicted-ns billing
   artifacts; #240 merged = our emitter fixes; #241/#237 = actionable finalize
   errors).
2. **Checkout the new SHA** (detached) in `external/TileFoundry`.
3. **Probe harness:** `./evals/tilefoundry-capability-map/run_probes.sh` →
   expect `12/12 PASS` (exits non-zero on any DIFF). If a DIFF is an upstream
   *improvement* (e.g. better error text), read the new output, update the
   fingerprint in `evals/tilefoundry-capability-map/probes/expected/`, rerun.
   Never edit fingerprints to silence a regression.
4. **Pin ritual** (proves the user-facing workflow still works):
   - Extract the tutorial programs with the awk block from
     `external/TileFoundry/docs/tutorial/optimize.md`, then
     `tilefoundry analyze rms_norm_quant.py:Fused out.txt --compute-cost
     --memory --roofline` → expect `ideal-ns=13`.
   - One runtime-twin check, e.g. from
     `evals/tilefoundry-coldstart-20261019/cases/scan_occupancy`:
     `tilefoundry check hir/twin_tilelang.py:FastTileLang --dim C=40
     --inputs files:refdata/in0.pt,refdata/in1.pt --expected refdata/expected0.pt
     --expected refdata/expected1.pt --out 'output[0]' --fn allclose --atol 0.01
     --rtol 0.0078 --out 'output[1]' --fn allclose --atol 0.01 --rtol 0.0078`
     → PASS both outputs.
   - One upstream test:
     `.venv/bin/python -m pytest external/TileFoundry/tests/ops/tir/test_copy.py -x -q`.
5. **Trust-table spot checks** (byte-identical unless upstream changed costing):
   scan fused (`scan_variants.py:ScanFused`, `--dim C=40`, `--performance`) →
   `ideal-ns=125063`; GEMM scheduled (`gemm_compute/hir/scheduled.py:Scheduled`)
   → `ideal-ns=143256 bound-by=compute`.
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

## Trust invariants the skills assert — keep them true

- **predicted-ns is never a verdict.** `analyze --performance` bills
  `insert_slice`/`tf.zeros` at whole-window cost per mesh position and cannot
  see lane under-fill (#239); it has named the wrong fusion winner. Rank HIR
  variants by logical traffic/FLOP diffs and roofline floors; occupancy and
  latency questions belong to ncu/torch.profiler.
- **Commands are the contract.** Captured output in docs/READMEs/examples
  drifts between releases; rerun at the pinned checkout and read what it
  actually prints.
- **TileFoundry `compile()` is not a measured floor.** Emitter gaps remain
  (TMA / WGMMA / `Where`) plus the `warps_per_group` reduce-dispatch bug; the
  supported integration is TileFoundry for HIR/check/analyze/schedule/finalize
  with TileLang kernels implementing the finalized TIR.
- **The TileLang runtime-twin pattern** (equal/allclose predicates per output;
  `--fn ulp --max 1` for bf16 reduction-order disagreement) is the correctness
  gate for optimization candidates — keep `tilefoundry check` green after each
  edit.
