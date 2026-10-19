# KernelWiki interface for TileLang work

Use the standalone KernelWiki skill when a design choice, lowering result, or
profile needs an implementation or hardware explanation. The upstream
repository owns the knowledge and query tools; the suite's installer keeps a
usable checkout next to this suite's skills. Keep local notes about the
current kernel and experiment.

## Locate and query

Find the installed KernelWiki skill directory: `~/.agents/skills/KernelWiki`
in a global install, or the `KernelWiki` entry under the task project's
`.agents/skills`, `.claude/skills`, or `.codex/skills` in a project install.
If it is missing, install it per its official instructions — a standalone
clone whose root is the skill directory:

```bash
git clone --depth 1 https://github.com/mit-han-lab/KernelWiki.git ~/.agents/skills/KernelWiki
```

Its scripts resolve their corpus from their own location, so they run from
anywhere with the task's active `python`, whatever environment manager
provides it (venv, uv, conda, ...):

```bash
kw=~/.agents/skills/KernelWiki   # the installed KernelWiki skill directory
# Find a mechanism or a TileLang implementation change.
python3 "$kw/scripts/query.py" pipeline --type technique --limit 3 --compact
python3 "$kw/scripts/query.py" --repo tilelang --limit 3 --compact
# Fetch a selected page and the excerpts of its cited sources.
python3 "$kw/scripts/get_page.py" technique-pipeline-stages --follow-sources
# Search an unfamiliar symbol found in generated code.
python3 "$kw/scripts/grep_wiki.py" mbarrier --only wiki --limit 3 --files-only
```

Choose query words from the actual operation, emitted symbol, or observed
symptom. The upstream interface also supports `--architecture`, `--language`,
`--symptom`, `--has-code`, and `--paths-only`; see the skill's own `SKILL.md`
for navigation and `scripts/query.py --help` for filters. Use an architecture
filter matching the actual target. A cross-language example can explain
emitted CUDA even when a TileLang-only search has no result.

Read selected pages, follow their `sources` and `prerequisites` as needed, and
inspect concrete upstream files with their `PROVENANCE.yaml`. `--follow-sources`
prints excerpts; fetch a source ID separately for its full page. If a search
returns no applicable evidence, refine it or use the installed TileLang source
and authoritative target documentation; an empty result does not prove a
mechanism is unsupported. PyYAML is optional; upstream includes a fallback.

## Connect the explanation to the actual kernel

1. **TileLang intent:** identify the relevant source/configuration and contract.
   For example, a `T.Pipelined` stage count expresses a scheduling request.
2. **Lowered implementation:** use specialized TIR, lowering traces, generated
   host/device code, and layouts from [compiler inspection](../../tilelang/debugging-tilelang-programs/references/compiler-tools.md).
   Identify the actual copies, buffers, synchronization, compute operations,
   and launch configuration before choosing a Wiki explanation.
   When the claim depends on specific machine instructions, corroborate
   generated CUDA with target PTX/SASS or a source-correlated profile.
3. **Hardware hypothesis:** use the selected Wiki page and its sources to
   explain how that implementation could produce the observed behavior on
   this target. State the expected code change and measurable effect. For
   a stage-count experiment, check the resulting buffers/waits and resource
   use; the parameter value alone does not demonstrate overlap.
4. **Check the mechanism:** inspect the changed lowering, validate correctness,
   and compare under the same benchmark conditions. Collect focused profiler
   evidence when needed to distinguish explanations. If the expected lowering
   change did not occur, investigate that boundary before expanding the search.

Keep a short chain with the experiment:
`TileLang source/config -> lowering artifact -> Wiki page/source -> hypothesis -> validation/measurement -> decision`.
Record the Wiki revision (`git -C "$kw" rev-parse HEAD`), page
ID/path, upstream locator, actual target, and confidence. Distinguish facts
observed locally from explanations still to be tested.

KernelWiki primarily covers Hopper/Blackwell. Verify target and tool-version
requirements before adapting a pattern. Its `verified`/`snippet` labels describe
corpus evidence and structure; they do not establish local compilation,
correctness, or speed. Treat source-reported performance as evidence for its
stated workload. Retrieve architecture details as needed for this investigation;
preserve upstream attribution instead of copying pages into a second local KB.
