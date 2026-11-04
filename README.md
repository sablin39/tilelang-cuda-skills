# GPU Development Skills

Agent skills for drafting, inspecting, debugging, measuring, and optimizing
TileLang and CUDA programs. The workflows focus on programming models and
evidence from source, lowering, tests, and profilers.

Start with [tilelang-optimization-loop](skills/tilelang-optimization-loop/SKILL.md)
for an iterative TileLang task. It adapts [KDA](https://github.com/mit-han-lab/kernel-design-agents)'s (mirrored as [NVlabs/kda](https://github.com/NVlabs/kda)) contract, draft, executable plan,
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

The standalone [KernelWiki](https://github.com/mit-han-lab/KernelWiki) skill
supplies upstream implementation and hardware knowledge; the installer sets it
up next to this suite's skills. TileLang skills use a
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
Kimi Code, Claude Code, and Codex, verifies KernelWiki, and installs pinned
TileFoundry core, NVIDIA and Apple packages with CUTLASS source and cmake/ninja.
It creates/reuses a Python 3.12+ virtual environment at `<suite>/.venv` by
default. Python 3.12+ must be available on the host.

Quick install, no clone needed — the script fetches the suite into
`~/.local/share/tilelang-cuda-skills` and symlinks from there:

```bash
curl -fsSL https://raw.githubusercontent.com/sablin39/tilelang-cuda-skills/main/scripts/install-skills.sh | bash -s -- --scope global
```

Rerunning the same command updates the cached suite and every installed CLI at
once, and moves TileFoundry to the commit the suite records. Override the
source with `$SUITE_URL`, `$SUITE_REF`, and `$SUITE_CACHE`. The installer never
updates TileFoundry to upstream main: it checks out the submodule gitlink and
reinstalls matching editable packages. A failed cache update stops installation.
When running from an existing checkout, update that checkout first.

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

To use an existing virtual environment instead of `<suite>/.venv`, pass
`--python /path/to/venv/bin/python` to installation and subsequent checks.
The script bootstraps pip with `ensurepip` when needed, including in a venv
created without pip. It does not install into the system Python.

Skills are symlinked by default, so a `git pull` of this suite updates every
installed CLI at once; pass `--copy` for a snapshot. The script repairs what it
finds: broken symlinks, pre-installed directories conflicting with symlink
mode, and a KernelWiki checkout with no readable commit or one behind the
remote (removed and reinstalled). Audit without changing anything via
`--check-only`.

KernelWiki is verified with its official smoke test
(`scripts/query.py --tag nvfp4 --type kernel --compact`). It is installed per
its official instructions — a standalone clone whose root is the skill
directory — at `~/.agents/skills/KernelWiki` in both scopes; project-scope
installs link it into each project skills directory from there.

Network access, proxy variables included, is inherited from the caller's
environment — set or unset `ALL_PROXY`/`all_proxy` before running as your
network requires.

## Initialize KernelWiki manually

Without the script, clone it per its official instructions — the clone root is
the skill directory:

```bash
git clone --depth 1 https://github.com/mit-han-lab/KernelWiki.git ~/.agents/skills/KernelWiki
python3 ~/.agents/skills/KernelWiki/scripts/query.py pipeline --type technique --limit 3 --compact
```

Record the Wiki revision (`git -C ~/.agents/skills/KernelWiki rev-parse HEAD`)
in task evidence for reproducible runs; update the clone deliberately when
adopting upstream changes. Query scripts work without extra pip dependencies.
Copied installs rewrite cross-skill Markdown links for the flat layout.
Keep the full suite checkout available for vendored tools; copied skills contain
their own documentation and helpers.

## TileFoundry

TileFoundry ships as a pinned submodule at `external/TileFoundry`. The installer
initializes its recursive dependencies even when the core is already at the
pin but CUTLASS is missing. It installs core and both bundled vendor packages
together, replaces older installations in the selected environment, and removes
obsolete root `tilefoundry.egg-info` metadata from the old monolithic layout.

Installation verifies matching package versions, editable source/import paths,
CLI and target entry points, cmake/ninja executables, and a successful
`tilefoundry target list` with CPU, CUDA and AMX registration. A receipt in the
environment records the checked source pin. `--check-only` verifies that receipt
and the installed metadata without fetching, installing, or importing target
implementations. An environment installed manually has no receipt until the
installer verifies it. These checks do not compile or execute a GPU kernel.

For example, install and check a task environment:

```bash
SUITE=...   # checkout/cache root
TASK_ENV=/path/to/venv
"$SUITE/scripts/install-skills.sh" --python "$TASK_ENV/bin/python" --scope project
"$SUITE/scripts/install-skills.sh" --python "$TASK_ENV/bin/python" --scope project --check-only
"$TASK_ENV/bin/tilefoundry" target list
```

Runtime headers ship with their owning packages. CUDA execution also needs a
compatible driver and host CUDA toolkit (`nvcc`); the installer does not install
drivers or the CUDA toolkit. Activate the selected environment (including its
`bin` directory on `PATH`) when using TileFoundry. A src-only `PYTHONPATH`
override does not load a different vendor package.

After updating a suite checkout, rerun its installer to update the pinned source
and Python packages together. For the default environment:

```bash
"$SUITE/scripts/install-skills.sh" --scope global
"$SUITE/scripts/install-skills.sh" --scope global --check-only
```

Bump the pin deliberately and re-verify the workflow at each new commit — the
probes are in [designing-with-tilefoundry](skills/designing-with-tilefoundry/SKILL.md).

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
on 2026-09-11 (TileLang `0.1.14+cuda.git021592f4`), and re-reviewed on 2026-10-08
against TileLang `0.1.15+cuda.git29b4fdd2`, PyTorch `2.13.0+cu130`, and CUDA
toolkit 13.0, with compile-and-run spot checks of the v0.1.15 surface
(`T.gemm_blockscaled`, role-based auto warp specialization, eager compile-time
iteration, `do_bench` wall backend). Upstream docs and installed APIs can
differ; verify the active version rather than assuming every copied snippet is
current. The original templates were based on TileLang 0.1.9.
The buffer-lifetime and compiler-instrumentation APIs were rechecked against
the 0.1.14 commit and bundled TVM `0.25.dev0` on 2026-09-15; the bundled TVM
remains `0.25.dev0` at 0.1.15.

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
The suite installer also installs TileFoundry into its default virtual
environment or the environment selected by `--python` (see
[TileFoundry](#tilefoundry)). Use that environment for its compatibility checks.

## CUDA skill

[CUDA development](skills/cuda_skill/SKILL.md) covers compilation, launch
contracts, sanitizers, Nsight collection, and NVTX regions. The skill originated
from [technillogue/ptx-isa-markdown](https://github.com/technillogue/ptx-isa-markdown).
Existing versioned CUDA/PTX documentation snapshots remain in its references;
use the supplementary KernelWiki for targeted architecture research and connect
its explanations to the actual TileLang lowering and measurements.

The tracked `.vscode` directory provides editor highlighting configuration.

## Stage handovers and maintenance

The loop routes by the current decision question and consumes specialist
findings before the next experiment. Its [stage report](skills/tilelang-optimization-loop/references/stage-handoff.md)
records matching source/context, findings, checks, artifacts and the next route.
The checker detects stale evidence and unresolved promotion gates; it does not
verify the truth of reported results. Reports link into the existing ledger;
single questions do not require this machinery.

Reusable findings feed a scoped [maintenance loop](skills/tilelang-optimization-loop/references/improving-the-suite.md)
with reproducer, owning reference, affected and nearby cases, and an accept/reject
decision. Raw task evidence stays outside delivered skills.

`--check-only` is offline and read-only. It checks selected installation files,
reachable links, missing/stale copies, KernelWiki structure, vendored
pin/CUTLASS presence and the selected TileFoundry environment, returning nonzero
on failure. It cannot establish that the active agent discovered those files;
verify its advertised skill list.
Source-only checks and helper tests:

```bash
python3 scripts/skill_layout.py check skills
python3 -m unittest discover -s tests -v
```

For broad tiling searches, use the [calibrated cost model](skills/tilelang-optimization-loop/references/cost-model.md),
[measurement recipes](skills/tilelang/profiling-tilelang-programs/references/microbenchmark-calibration.md)
and [tensor constraint guidance](skills/tilelang-optimization-loop/references/constraint-pruning.md).
They distinguish hardware reference data, measured service profiles, proven
infeasibility and budget-based deferral. Z3, CP-SAT and ISL integration uses
the task environment's existing libraries; no new default dependency is added.
