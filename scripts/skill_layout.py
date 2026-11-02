#!/usr/bin/env python3
"""Check reachable skill links and relocate cross-skill links for flat copies."""

import argparse
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit
import os

LINK = re.compile(r"(!?\[[^\]\n]*\]\()(<[^>\n]+>|[^\s)]+)(\))")


def roots(source):
    candidates = sorted(source.resolve().rglob("SKILL.md"))
    result = []
    for entry in candidates:
        if not any(parent in result for parent in entry.parent.parents):
            result.append(entry.parent)
    names = [p.name for p in result]
    if not result or len(names) != len(set(names)):
        raise ValueError("no skills found or duplicate installation directory names")
    return result


def links(text, callback):
    """Process inline Markdown links outside fenced code; leave URLs/anchors alone."""
    fence = None
    lines = []
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            token = marker[1]
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            lines.append(line)
            continue
        def replace(match):
            raw = match[2].strip("<>")
            parsed = urlsplit(raw)
            if parsed.scheme or parsed.netloc or not parsed.path or raw.startswith("/"):
                return match[0]
            replacement = callback(unquote(parsed.path), raw[len(parsed.path):])
            if replacement is None:
                return match[0]
            if match[2].startswith("<"):
                replacement = "<" + replacement + ">"
            return match[1] + replacement + match[3]
        lines.append(line if fence else LINK.sub(replace, line))
    return "".join(lines)


def relocated(path, skill_roots, destination):
    owner = next(root for root in skill_roots if path.is_relative_to(root))
    out = destination / owner.name / path.relative_to(owner)
    def redirect(relative, suffix):
        target = (path.parent / relative).resolve()
        target_owner = next((r for r in skill_roots if target.is_relative_to(r)), None)
        if target_owner is None or target_owner == owner:
            return None
        new_target = destination / target_owner.name / target.relative_to(target_owner)
        return Path(os.path.relpath(new_target, out.parent)).as_posix() + suffix
    return links(path.read_text(), redirect)


def reachable_errors(entries):
    queue, seen, errors = list(entries), set(), []
    while queue:
        path = queue.pop().resolve()
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            errors.append(f"missing entry/reference: {path}")
            continue
        def visit(relative, suffix):
            target = (path.parent / relative).resolve()
            if not target.exists():
                errors.append(f"broken link: {path} -> {relative}{suffix}")
            elif target.is_file() and target.suffix == ".md":
                queue.append(target)
        links(path.read_text(), visit)
    return errors


def check(source, destination=None):
    skill_roots = roots(source)
    errors = []
    for root in skill_roots:
        text = (root / "SKILL.md").read_text()
        front = text.split("---", 2)
        if len(front) < 3 or front[0].strip() or not all(
            re.search(rf"^{key}:\s*\S", front[1], re.M) for key in ("name", "description")
        ):
            errors.append(f"missing name/description frontmatter: {root}")
        if destination is not None:
            installed = destination / root.name
            if installed.resolve() == root.resolve():
                continue
            for path in root.rglob("*"):
                if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                    continue
                target = installed / path.relative_to(root)
                expected = (relocated(path, skill_roots, destination).encode()
                            if path.suffix == ".md" else path.read_bytes())
                if not target.is_file() or target.read_bytes() != expected:
                    errors.append(f"missing/stale installed file: {target}")
    entries = [(destination / r.name if destination else r) / "SKILL.md" for r in skill_roots]
    return errors + reachable_errors(entries)


def relocate(source, destination):
    skill_roots = roots(source)
    for root in skill_roots:
        installed = destination / root.name
        if installed.is_symlink() or installed.resolve() == root.resolve():
            raise ValueError(f"copy relocation refuses a symlink/source destination: {installed}")
        for path in root.rglob("*.md"):
            target = installed / path.relative_to(root)
            if target.is_symlink() or any(p.is_symlink() for p in target.parents if p.is_relative_to(installed)):
                raise ValueError(f"copy relocation refuses symlink path: {target}")
            target.write_text(relocated(path, skill_roots, destination))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "relocate"))
    parser.add_argument("source", type=Path)
    parser.add_argument("--installed", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    destination = args.installed.absolute() if args.installed else None
    try:
        if args.action == "relocate":
            if destination is None:
                parser.error("relocate requires --installed")
            relocate(source, destination)
        errors = check(source, destination)
    except (OSError, ValueError) as exc:
        errors = [str(exc)]
    for error in errors:
        print(error, file=sys.stderr)
    print(f"Skill layout: {'FAIL' if errors else 'PASS'}")
    return bool(errors)


if __name__ == "__main__":
    sys.exit(main())
