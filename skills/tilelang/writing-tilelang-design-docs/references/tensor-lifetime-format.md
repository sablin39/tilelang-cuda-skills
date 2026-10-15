# Tensor Lifetime SVG Input for Design Documents

The lifetime renderer consumes JSON and writes a standalone SVG. It expects
already-derived intervals and access events; use it after inspecting a named
TileLang compiler boundary or a runtime trace.

## Schema

```json
{
  "title": "Pipelined GEMM buffer access spans",
  "subtitle": "M=N=K=1024, block=128x128x32, after pipeline injection",
  "axis": {
    "kind": "compiler-event",
    "label": "lowered statement event index",
    "end_semantics": "exclusive",
    "min": 0,
    "max": 12
  },
  "events": [
    {"at": 3, "label": "async copy issued"},
    {"at": 7, "label": "wait + barrier"}
  ],
  "buffers": [
    {
      "name": "A",
      "scope": "global",
      "start": 1,
      "end": 4,
      "interval": "access",
      "note": "caller-owned; reads only",
      "accesses": [
        {"at": 1, "kind": "read"},
        {"at": 3, "kind": "read"}
      ]
    },
    {
      "name": "A_shared_0",
      "scope": "shared",
      "start": 2,
      "end": 8,
      "interval": "allocator-live",
      "offset_bytes": 0,
      "size_bytes": 8192,
      "accesses": [
        {"at": 2, "kind": "write"},
        {"at": 6, "kind": "read"},
        {"at": 7, "kind": "wait"}
      ]
    },
    {
      "name": "C_local",
      "scope": "fragment",
      "start": 4,
      "end": 11,
      "interval": "logical-access",
      "uncertain": true,
      "note": "not a physical register interval",
      "accesses": [
        {"at": 4, "kind": "write"},
        {"at": 9, "kind": "read"}
      ]
    },
    {
      "name": "C",
      "scope": "global",
      "start": 9,
      "end": 12,
      "interval": "access",
      "note": "caller-owned; writes only",
      "accesses": [
        {"at": 10, "kind": "write"}
      ]
    }
  ]
}
```

Required fields are `axis.kind`, `axis.label`, and a nonempty `buffers` list.
Every buffer requires `name`, `scope`, `start`, and `end`. Supported axis kinds
are `compiler-event`, `source-order`, and `runtime`. The label supplies the
units; for a runtime axis, use a label such as `GPU timestamp (µs)`.

`interval` and `note` are descriptive. `offset_bytes` and `size_bytes` record a
shared-arena assignment without claiming a source-level allocation address.
Set `uncertain` to render a dashed outline. Access kinds use `read`, `write`,
`wait`, `barrier`, or `other`.

Top-level `events` draw vertical reference lines. Event positions and access
markers must lie inside the axis range. When `axis.min` or `axis.max` is omitted,
the renderer derives it from buffer intervals and events.

## Command

```bash
# From the suite root, matching the invocation in the design-docs skill:
python skills/tilelang/writing-tilelang-design-docs/scripts/render_tensor_lifetimes.py \
  lifetimes.json -o lifetimes.svg
```

The output uses a responsive SVG `viewBox`, tooltips, scope colors, access
markers, and the declared axis semantics. Commit or attach both files when the
figure supports an implementation claim.
