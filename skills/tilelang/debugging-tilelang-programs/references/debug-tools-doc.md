# Selecting a Debugging Tool

| Question | Tool | Evidence and limits |
| --- | --- | --- |
| What did specialization produce? | `factory.get_tir(...)` | Shapes, constants, and high-level TIR |
| What source would lowering emit? | Compile-only | Source without device compilation or launch |
| Which transformation changed the program? | `TL_LOWER_TRACE` | Before/after IR and codegen, including failures |
| How did block/operator structure change? | CUDA Pass Visualizer | Per-pass structure tree and text dump for a focused lowering prologue |
| Why was an argument rejected? | `kernel.get_host_source()` | Shape, stride, dtype, device, and launch checks |
| What value did a kernel compute? | `T.print` / `T.device_assert` | Runtime values/invariants with diagnostic overhead |
| Where is invalid access/synchronization? | compute-sanitizer | Tool-specific reports; not proof of complete correctness |
| Which mapping did layout inference choose? | Layout text/plots | Thread/data mapping for a small inspectable case |
| When is storage live or reused? | TIR access analysis plus shared allocator report | Compiler event intervals and arena offsets, not GPU time or physical register allocation |
| Which compiler phase is slow? | TileLang pass profile | Inclusive/self compiler-pass duration, not kernel runtime |
| Can a stable failure be simplified? | AutoDD | Minimal Python reproducer preserving a specific signature |

Read [compiler-tools.md](compiler-tools.md) for compile-only, lowering traces,
source editing, pass instruments, and visualization. Read
[buffer-lifetime-analysis.md](buffer-lifetime-analysis.md) for fragment/local,
shared, and global access/storage lifetimes. Choose the smallest tool that
answers the current question instead of enabling every diagnostic at once.

## T.print

```python
T.print(A_shared, msg="after copy")
T.print(C_local, msg="after accumulation")
```

Use small tiles and selected blocks. Shared/fragment printing may involve
collective cooperation; do not put such diagnostics inside a branch that
prevents participating threads from reaching them. Interpret values using the
logical buffer shape and the reported mapping. Remove prints before timing.

## AutoDD

```bash
python -m tilelang.autodd repro.py --err-msg "distinctive error" \
  -o minimized.py --timeout 60 -j 1
```

The interestingness test matches text in stdout/stderr and can succeed even
with exit code zero. Confirm the original signature and re-run the result.
Keep required initialization intact using `# autodd: freeze` or paired
`# autodd: freeze-start` / `# autodd: end-freeze` comments. Limit workers when
candidate programs share a GPU. `subproc` uses a fresh `python3`, so activate
the correct environment before selecting that backend.

## Logging

`TVM_LOG_DEBUG` controls runtime verbosity for available statements, but cannot
restore DLOG/DCHECK code removed by a release build. Inspect lowering traces
before deciding whether a debug rebuild is necessary.

Sources in the matching checkout: `docs/tutorials/debug_tools_for_tilelang.md`,
`docs/tutorials/logging.md`, and `docs/tools/autodd.md`.
