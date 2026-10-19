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
Kimi Code, Claude Code, and Codex, and verifies KernelWiki.

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
Keep the full suite checkout available when using its skills so their relative
links resolve.

## Initialize TileFoundry manually

TileFoundry is vendored the same way at `external/TileFoundry`, but it also
needs an editable install and its CUTLASS submodule (its CUDA backend resolves
headers relative to the source checkout; the PyPI wheel ships none):

```bash
git submodule update --init -- external/TileFoundry
git -C external/TileFoundry submodule update --init third_party/cutlass
uv pip install --python .venv/bin/python -e external/TileFoundry cmake ninja
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
