# GPU Development Skills

Agent skills for drafting, inspecting, debugging, measuring, and optimizing
TileLang and CUDA programs. The workflows focus on programming models and
evidence from source, lowering, tests, and profilers.

Start with [tilelang-optimization-loop](skills/tilelang-optimization-loop/SKILL.md)
for an iterative TileLang task. It adapts [KDA](https://github.com/NVlabs/kda)'s contract, draft, executable plan,
candidate, validation, measurement, and promotion cycle to TileLang lowering.
Keep the plan and candidate history in the task workspace. Analyze the supplied
PyTorch baseline through `torch.compile` graphs/generated code and profiling,
then connect each candidate's intended mechanism to its emitted implementation
and measured result. Use a specialist directly for a focused question.

For training operations, the loop's [joint forward/backward mode](skills/tilelang-optimization-loop/references/joint-fwd-bwd.md)
keeps kernels and saved state aligned, validates all gradients, and selects
the pair using complete forward-plus-backward measurements.
Use the [checkpoint handoffs](skills/tilelang-optimization-loop/references/worker-handoff.md)
to give lighter workers a bounded baseline, candidate, or final-audit stage
and check the returned evidence before continuing.

The supplementary [KernelWiki](external/KernelWiki/SKILL.md) submodule supplies
upstream implementation and hardware knowledge. TileLang skills use a
[lightweight interface](skills/tilelang-optimization-loop/references/kernelwiki.md) to connect
source/configuration, lowering artifacts, hardware explanations, and measured
results. Query the upstream corpus as needed and keep experiment evidence in
the task workspace.

For model- or operator-level design, the optional
[TileFoundry workflow](skills/designing-with-tilefoundry/SKILL.md) adds authored
HIR, topology and placement, static cost analysis, evaluator/runtime-twin
checking, and an interactive HIR viewer. Use it before or alongside TileLang
when decisions span fusion, sharding, or memory-placement boundaries. It is a
separate installation rather than a default suite dependency.

## Install the suite

[scripts/install-skills.sh](scripts/install-skills.sh) installs the skills for
Kimi Code, Claude Code, and Codex, verifies KernelWiki, and sets up the
[humanize2](https://github.com/humanfia/humanize) orchestrator (`hmz`) that can
drive all three CLIs in one reviewed workflow.

Quick install, no clone needed — the script fetches the suite into
`~/.local/share/tilelang-cuda-skills` and symlinks from there:

```bash
curl -fsSL https://raw.githubusercontent.com/sablin39/tilelang-cuda-skills/main/scripts/install-skills.sh | bash -s -- --scope global
```

Rerunning the same command updates the cached suite and every installed CLI at
once. Override the source with `$SUITE_URL`, `$SUITE_REF`, and `$SUITE_CACHE`.

Global install from a checkout — canonical location `~/.agents/skills`,
mirrored with symlinks into `~/.claude/skills` and `~/.codex/skills`:

```bash
./scripts/install-skills.sh --scope global --target all
```

Project-local install — run from the task project's root; installs into
`./.agents/skills`, `./.claude/skills`, and `./.codex/skills`:

```bash
/path/to/suite/scripts/install-skills.sh --scope project --target all
```

Skills are symlinked by default, so a `git pull` of this suite updates every
installed CLI at once; pass `--copy` for a snapshot. The script repairs what it
finds: broken symlinks, pre-installed directories conflicting with symlink
mode, a KernelWiki checkout with no readable commit or one behind the remote
(removed and reinstalled), and an `hmz` that does not run (reinstalled with
`--force`). Audit without changing anything via `--check-only`.

KernelWiki is verified with its official smoke test
(`scripts/query.py --tag nvfp4 --type kernel --compact`). In global scope it is
installed per its official instructions — a standalone clone whose root is the
skill directory — at `~/.agents/skills/KernelWiki`. In project scope the script
prefers, in order: `--kernelwiki PATH`, `$KERNELWIKI_ROOT`, this suite's
`external/KernelWiki` submodule, then `~/.agents/skills/KernelWiki`.

Unless `--no-hmz` is given, `hmz` is installed with `uv` (a hard requirement;
the script stops immediately without it) with dependencies resolved through
`--pypi-mirror URL` (default `https://mirrors.aliyun.com/pypi/simple`, or set
`$PYPI_MIRROR`). The script warns when any of the `kimi`, `claude`, or `codex`
CLIs is not on `PATH`. Network access, proxy variables included, is inherited
from the caller's environment — set or unset `ALL_PROXY`/`all_proxy` before
running as your network requires.

## A short guide to hmz

`hmz` runs *flows* — Python loops that drive one or more agent CLIs — using the
CLI you are already logged into; it holds no API keys itself. Everything below
is the whole of what you need from the hmz docs for this suite.

Two ways to run anything: the TUI (`hmz`, then `$<flow> <task>` and answer
`/flow` once), or headless (`hmz exec`, as below). Agents are named
`cli/model:effort`, where `model` is the alias the CLI itself registers — Kimi
Code's aliases are namespaced, so `kimi/kimi-code/k3:high` names the Kimi Code
CLI running its `kimi-code/k3` alias (not a bare `k3`), and the 256k-context
variant is a separate alias: `kimi/kimi-code/k3-256k:high`. The models offered
in `/flow` are the ones your account may name, asked of the CLI itself.

### Single agent: `ralph_loop`

```bash
hmz exec -f ralph_loop -a kimi/kimi-code/k3:high "Fix the bug in kernel.py"
```

One agent gets the same task over and over, each round in a **fresh
conversation** — it restarts from the task and the repository rather than from a
context window full of its own earlier attempts. Simple, cheap, and the right
default for one bounded experiment at a time.

### Multiple agents: `flame_chase`

```bash
hmz exec -f flame_chase -a kimi/kimi-code/k3-256k:high -a claude/glm-5.3:high "Fix the bug in kernel.py"
```

Two implementers — here Kimi Code's Kimi K3 and GLM-5.3 backed by Claude Code,
one `-a` per agent in turn order — **alternate turns on the same tree**. Neither
is told what the other said; the repository is the only account of the last
turn. The difference from `ralph_loop`: a loop over one agent compounds that
model's blind spot, while alternation gives every round to a model that did not
write what it is looking at — two models fail differently, so each round is also
an implicit review of the last one. (The number of `-a` flags must match the
flow: `ralph_loop` takes one, `flame_chase` takes two.)

Either way: keep the plan and candidate history in files, as the
optimization-loop skill prescribes, since no round remembers the last. Loops do
not stop on their own — a run's **allowance** (hours, tokens, or dollars, set
via `-c budget.yaml` or the budget row in `/flow`) caps them — and every round
is logged, so stopping loses nothing.

Steer and stop. While a turn runs, just type a line — it goes *into* the
running turn (Kimi Code, Claude Code, and Codex all accept mid-turn input);
with no turn open the line is held for the next one, never dropped. Stop with
**ctrl+c twice** in the TUI (once for `hmz exec`) or `/stop`; `/resume` picks
the run up where it stopped, and `/epics` lists past runs.

Review with all three CLIs. A custom flow in `~/.hmz/flows/<name>/`
(global) or `<task>/.hmz/flows/<name>/` (project) can assign roles across
model families — e.g. Kimi implements a candidate, Claude reviews the evidence
chain, Codex re-runs the exact validation and benchmark commands — driven by
one `-a` per CLI:

```bash
hmz exec -f reviewed_opt_loop -a kimi/kimi-code/k3:high \
  -a claude/claude-opus-4-8:high -a codex/gpt-5.6-sol:high "Optimize per docs/plan.md"
```

Where flows live. The classic loops — `chat`, `ralph_loop`, `stateful_ralph`,
`continue_loop`, `goal`, `flame_chase`, `rlar` — ship inside the `hmz` package
and need no setup. Everything else (`humanize1`'s idea/plan/rlcr phases,
`flame_chasoid`, and other experiments) is a separately versioned release in the
official flowverse: open `/flowverses` in the TUI and fetch, then install the
release you want. hmz is published on PyPI as pre-releases, so
`uv tool install 'hmz[all]>=0.1.0b1'` also works; this suite's installer tracks
the git source instead so upgrades follow upstream `main`. Note hmz's state
home is `~/.hmz` — runs, settings, and flowverses from the old `~/.humanize`
layout are not carried over, so per-directory flow/agent setups are chosen
once again on first use.

See the [humanize user guide](https://github.com/humanfia/humanize/tree/main/docs)
for flows, steering, permissions, and allowances.

## Initialize KernelWiki manually

Without the script, run from this suite's root:

```bash
git submodule update --init -- external/KernelWiki
python external/KernelWiki/scripts/query.py pipeline --type technique --limit 3 --compact
```

Git records the Wiki revision as a submodule pin. Initialize that revision for
reproducible tasks; update the pin deliberately when adopting upstream changes.
Query scripts work without extra pip dependencies. Keep the full suite checkout
available when using its skills so their relative links resolve.

For optional ncu-report-skill integration and Kernel Design Agents conventions,
see [external tool integration](skills/tilelang-optimization-loop/references/external-tools.md).

## TileLang skills

| Skill | Use it for |
| --- | --- |
| [designing-with-tilefoundry](skills/designing-with-tilefoundry/SKILL.md) | Authored model/operator HIR, topology and placement, static analysis, runtime-twin checking, and HIR visualization |
| [writing-tilelang-kernels](skills/tilelang/writing-tilelang-kernels/SKILL.md) | PrimFunc factories, eager JIT, tensor contracts, and kernel templates |
| [writing-tilelang-design-docs](skills/tilelang/writing-tilelang-design-docs/SKILL.md) | Human-facing implementation proposals and reports with data-flow, pipeline, lifetime, compiler, and runtime visuals |
| [debugging-tilelang-programs](skills/tilelang/debugging-tilelang-programs/SKILL.md) | Runtime failures, host checks, buffer lifetime/storage analysis, lower traces, and TVM diagnostics |
| [profiling-tilelang-programs](skills/tilelang/profiling-tilelang-programs/SKILL.md) | Benchmark scope, event/graph timing, IKET instrumentation, static analysis, and focused Nsight collection |
| [torch-profiling-tilelang-programs](skills/tilelang/torch-profiling-tilelang-programs/SKILL.md) | PyTorch baseline/torch.compile inspection, operator timelines, and allocation history |
| [optimizing-tilelang-programs](skills/tilelang/optimizing-tilelang-programs/SKILL.md) | Measured experiments, autotuning, pipeline annotations, and layout changes |
| [testing-fwd-bwd-kernels](skills/tilelang/testing-fwd-bwd-kernels/SKILL.md) | Autograd contracts, independent forward/gradient checks, and accumulation state |

Analyze the existing PyTorch baseline, draft a kernel, inspect its lowering
when needed, validate against the reference, compare the relevant operation,
and optimize from that evidence. Load details only for the current task.

The workflows were reviewed against `../tilelang/docs` and the `qla` environment
on 2026-09-11: TileLang `0.1.14+cuda.git021592f4` from an editable checkout,
PyTorch `2.13.0+cu130`, and CUDA toolkit 13.0. Upstream docs and installed APIs
can differ; verify the active version rather than assuming every copied
snippet is current. The original templates were based on TileLang 0.1.9.
The buffer-lifetime and compiler-instrumentation APIs were rechecked against
the same TileLang commit and bundled TVM `0.25.dev0` on 2026-09-15.

## Extra Python dependencies

Assume PyTorch and TileLang are installed. Install the extra visualization
dependency into the selected environment:

```bash
python -m pip install -r requirements.txt
```

[requirements.txt](requirements.txt) includes Matplotlib for layout images and
HTML memory timelines. Text layout inspection and JSON memory export work
without it. The tensor-lifetime SVG renderer uses only the Python standard
library. CUDA toolkit tools, Compute Sanitizer, Nsight Compute, and Nsight
Systems are external installations, outside these pip requirements.
TileFoundry is also optional and intentionally omitted: follow its skill's
compatibility check and dependency gate before installing it into a separate or
existing environment.

## CUDA skill

[CUDA development](skills/cuda_skill/SKILL.md) covers compilation, launch
contracts, sanitizers, Nsight collection, and NVTX regions. The skill originated
from [technillogue/ptx-isa-markdown](https://github.com/technillogue/ptx-isa-markdown).
Existing versioned CUDA/PTX documentation snapshots remain in its references;
use the supplementary KernelWiki for targeted architecture research and connect
its explanations to the actual TileLang lowering and measurements.

The tracked `.vscode` directory provides editor highlighting configuration.
