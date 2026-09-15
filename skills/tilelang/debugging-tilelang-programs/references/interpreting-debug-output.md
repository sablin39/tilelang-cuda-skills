# Interpreting Debug Evidence

## Compare the earliest differing value

Run a small case with a known reference and inspect values after load,
accumulation/reduction, and writeback. Match logical buffer indices and dtype
when reading `T.print`; layout transformations can distribute a fragment
across threads. Avoid diagnosing a transpose or scale error from a single
scalar alone.

| Observation | Hypothesis to test |
| --- | --- |
| Inputs already differ after load | Addressing, strides, masks, or dtype |
| Load matches, accumulation differs | Initialization, transpose, reduction, scale |
| Accumulation matches, output differs | Writeback, conversion, ownership |
| Only tail values differ | Bounds, padding identity, mask convention |
| Extreme finite values produce inf/NaN | Unstable exp/log/division or accumulation overflow |
| Repeated calls differ | State reset, numerical atomic order, or synchronization |

## Read a sanitizer report

Record the tool, access kind, address/size, kernel, source location, and
reported thread/block. Use a build with source correlation and a small
reproducer. Multiple reports can originate from the same first error; fix
that access before interpreting downstream corruption.

Memcheck, racecheck, initcheck, and synccheck answer different questions.
Shared-memory race detection does not cover every global-memory protocol.
Diagnostic success is evidence for the checked case, not a universal proof.

## Read a lowering trace

Start from specialized TIR and find the first pass whose output no longer
matches the intended contract. Compare loops, regions, output initialization,
and producer/consumer relationships. A failed pass's input is useful even
when no compiled kernel object exists.

Keep a matching record of the factory arguments, target/backend, and pass
configs. Source dumps from a cached or differently specialized kernel can
explain the wrong program. A generated-source edit must actually be recompiled
before it can explain changed runtime behavior.

## Validate a minimized reproducer

Run it directly under the intended environment and verify the same failure.
AutoDD's text signature can accidentally match a warning or print statement;
inspect the reduced program before treating it as a kernel/compiler bug.
