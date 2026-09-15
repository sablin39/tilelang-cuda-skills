#!/usr/bin/env python3
"""Render reviewed tensor-lifetime evidence for a design document."""

from __future__ import annotations

import argparse
import json
import math
from html import escape
from pathlib import Path
from typing import Any


SCOPE_COLORS = {
    "global": "#2563eb",
    "shared": "#d97706",
    "fragment": "#7c3aed",
    "local": "#059669",
}
SCOPE_ORDER = {"global": 0, "shared": 1, "fragment": 2, "local": 3}
ACCESS_COLORS = {
    "read": "#0ea5e9",
    "write": "#ef4444",
    "wait": "#111827",
    "barrier": "#f59e0b",
    "other": "#64748b",
}
AXIS_KINDS = {"compiler-event", "source-order", "runtime"}


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value.strip()


def _format_number(value: float) -> str:
    return f"{value:.4g}"


def _load(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("top-level JSON value must be an object")

    axis = data.get("axis")
    if not isinstance(axis, dict):
        raise ValueError("axis must be an object")
    kind = _text(axis.get("kind"), "axis.kind")
    if kind not in AXIS_KINDS:
        raise ValueError(f"axis.kind must be one of {sorted(AXIS_KINDS)}")
    _text(axis.get("label"), "axis.label")

    raw_buffers = data.get("buffers")
    if not isinstance(raw_buffers, list) or not raw_buffers:
        raise ValueError("buffers must be a nonempty list")

    buffers: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_buffers):
        if not isinstance(raw, dict):
            raise ValueError(f"buffers[{index}] must be an object")
        item = dict(raw)
        item["name"] = _text(item.get("name"), f"buffers[{index}].name")
        item["scope"] = _text(item.get("scope"), f"buffers[{index}].scope").lower()
        item["start"] = _number(item.get("start"), f"buffers[{index}].start")
        item["end"] = _number(item.get("end"), f"buffers[{index}].end")
        if item["end"] < item["start"]:
            raise ValueError(f"buffers[{index}].end precedes start")
        accesses = item.get("accesses", [])
        if not isinstance(accesses, list):
            raise ValueError(f"buffers[{index}].accesses must be a list")
        normalized_accesses = []
        for access_index, access in enumerate(accesses):
            if not isinstance(access, dict):
                raise ValueError(f"buffers[{index}].accesses[{access_index}] must be an object")
            at = _number(access.get("at"), f"buffers[{index}].accesses[{access_index}].at")
            access_kind = _text(
                access.get("kind", "other"),
                f"buffers[{index}].accesses[{access_index}].kind",
            ).lower()
            normalized_accesses.append({"at": at, "kind": access_kind})
        item["accesses"] = normalized_accesses
        buffers.append(item)

    raw_events = data.get("events", [])
    if not isinstance(raw_events, list):
        raise ValueError("events must be a list")
    events = []
    for index, raw in enumerate(raw_events):
        if not isinstance(raw, dict):
            raise ValueError(f"events[{index}] must be an object")
        events.append(
            {
                "at": _number(raw.get("at"), f"events[{index}].at"),
                "label": _text(raw.get("label"), f"events[{index}].label"),
            }
        )

    return data, buffers, events


def _render(data: dict[str, Any], buffers: list[dict[str, Any]], events: list[dict[str, Any]]) -> str:
    axis = data["axis"]
    values = [value for item in buffers for value in (item["start"], item["end"])]
    values.extend(event["at"] for event in events)
    values.extend(access["at"] for item in buffers for access in item["accesses"])
    axis_min = _number(axis.get("min", min(values)), "axis.min")
    axis_max = _number(axis.get("max", max(values)), "axis.max")
    if axis_max <= axis_min:
        raise ValueError("axis.max must be greater than axis.min")
    for value in values:
        if value < axis_min or value > axis_max:
            raise ValueError(f"event position {value:g} lies outside the axis range")

    indexed = list(enumerate(buffers))
    indexed.sort(key=lambda pair: (SCOPE_ORDER.get(pair[1]["scope"], 99), pair[0]))
    ordered = [item for _, item in indexed]

    width = 1440
    left = 330
    right = 50
    top = 145
    row_height = 52
    plot_width = width - left - right
    height = top + row_height * len(ordered) + 130

    def x(value: float) -> float:
        return left + (value - axis_min) / (axis_max - axis_min) * plot_width

    title = escape(str(data.get("title", "Tensor lifetime view")))
    subtitle = escape(str(data.get("subtitle", "")))
    axis_label = escape(str(axis["label"]))
    end_semantics = escape(str(axis.get("end_semantics", "unspecified")))
    axis_kind = escape(str(axis["kind"]))

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
        f"<title>{title}</title>",
        "<style>",
        "text { font-family: ui-sans-serif, system-ui, sans-serif; fill: #172033; }",
        ".title { font-size: 26px; font-weight: 700; }",
        ".subtitle { font-size: 14px; fill: #526176; }",
        ".axis { font-size: 12px; fill: #526176; }",
        ".label { font-size: 14px; font-weight: 600; }",
        ".detail { font-size: 11px; fill: #526176; }",
        ".event { stroke: #94a3b8; stroke-width: 1; stroke-dasharray: 4 4; }",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="36" y="42" class="title">{title}</text>',
    ]
    if subtitle:
        lines.append(f'<text x="36" y="68" class="subtitle">{subtitle}</text>')
    lines.append(
        f'<text x="36" y="96" class="subtitle">Evidence axis: {axis_kind} · end: {end_semantics}</text>'
    )

    for tick in range(6):
        value = axis_min + (axis_max - axis_min) * tick / 5
        tick_x = x(value)
        lines.extend(
            [
                f'<line x1="{tick_x:.2f}" y1="{top - 12}" x2="{tick_x:.2f}" y2="{height - 86}" stroke="#e2e8f0"/>',
                f'<text x="{tick_x:.2f}" y="{top - 22}" text-anchor="middle" class="axis">{escape(_format_number(value))}</text>',
            ]
        )

    for event in events:
        event_x = x(event["at"])
        lines.extend(
            [
                f'<line x1="{event_x:.2f}" y1="{top - 7}" x2="{event_x:.2f}" y2="{height - 86}" class="event"/>',
                f'<text x="{event_x + 4:.2f}" y="{top - 42}" class="axis" transform="rotate(-20 {event_x + 4:.2f} {top - 42})">{escape(event["label"])}</text>',
            ]
        )

    previous_scope = None
    for row, item in enumerate(ordered):
        y = top + row * row_height
        scope = item["scope"]
        if scope != previous_scope:
            if row:
                lines.append(
                    f'<line x1="36" y1="{y - 4}" x2="{width - 36}" y2="{y - 4}" stroke="#cbd5e1" stroke-width="1.5"/>'
                )
            lines.append(
                f'<text x="36" y="{y + 18}" class="axis" text-transform="uppercase">{escape(scope.upper())}</text>'
            )
            previous_scope = scope

        name = escape(item["name"])
        interval = escape(str(item.get("interval", "interval")))
        details = [interval]
        if "offset_bytes" in item:
            details.append(f'offset {item["offset_bytes"]} B')
        if "size_bytes" in item:
            details.append(f'{item["size_bytes"]} B')
        note = str(item.get("note", ""))
        tooltip = escape(" · ".join([item["name"], *details, note] if note else [item["name"], *details]))
        color = SCOPE_COLORS.get(scope, "#64748b")
        start_x = x(item["start"])
        end_x = x(item["end"])
        bar_width = max(3.0, end_x - start_x)
        dash = ' stroke-dasharray="6 4"' if item.get("uncertain", False) else ""

        lines.extend(
            [
                f'<rect x="36" y="{y + 23}" width="{width - 72}" height="28" rx="6" fill="#f8fafc"/>',
                f'<text x="120" y="{y + 41}" class="label">{name}</text>',
                f'<text x="{left - 12}" y="{y + 41}" text-anchor="end" class="detail">{escape(" · ".join(details))}</text>',
                f'<g><title>{tooltip}</title><rect x="{start_x:.2f}" y="{y + 28}" width="{bar_width:.2f}" height="18" rx="5" fill="{color}" fill-opacity="0.82" stroke="{color}" stroke-width="2"{dash}/></g>',
                f'<text x="{start_x:.2f}" y="{y + 22}" text-anchor="middle" class="axis">{escape(_format_number(item["start"]))}</text>',
                f'<text x="{end_x:.2f}" y="{y + 22}" text-anchor="middle" class="axis">{escape(_format_number(item["end"]))}</text>',
            ]
        )
        for access in item["accesses"]:
            access_x = x(access["at"])
            access_kind = access["kind"]
            access_color = ACCESS_COLORS.get(access_kind, ACCESS_COLORS["other"])
            lines.append(
                f'<g><title>{escape(access_kind)} at {escape(_format_number(access["at"]))}</title><circle cx="{access_x:.2f}" cy="{y + 37}" r="5" fill="{access_color}" stroke="#ffffff" stroke-width="1.5"/></g>'
            )

    axis_y = height - 70
    lines.extend(
        [
            f'<line x1="{left}" y1="{axis_y}" x2="{width - right}" y2="{axis_y}" stroke="#334155" stroke-width="1.5"/>',
            f'<text x="{left + plot_width / 2:.2f}" y="{axis_y + 30}" text-anchor="middle" class="label">{axis_label}</text>',
        ]
    )

    legend_x = 36
    legend_y = height - 28
    for scope in ("global", "shared", "fragment", "local"):
        color = SCOPE_COLORS[scope]
        lines.extend(
            [
                f'<rect x="{legend_x}" y="{legend_y - 12}" width="14" height="14" rx="3" fill="{color}"/>',
                f'<text x="{legend_x + 20}" y="{legend_y}" class="axis">{scope}</text>',
            ]
        )
        legend_x += 100
    for kind in ("read", "write", "wait", "barrier"):
        color = ACCESS_COLORS[kind]
        lines.extend(
            [
                f'<circle cx="{legend_x + 7}" cy="{legend_y - 5}" r="5" fill="{color}"/>',
                f'<text x="{legend_x + 18}" y="{legend_y}" class="axis">{kind}</text>',
            ]
        )
        legend_x += 92
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON file containing lifetime evidence")
    parser.add_argument("-o", "--output", type=Path, required=True, help="destination SVG path")
    args = parser.parse_args()

    data, buffers, events = _load(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_render(data, buffers, events), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
