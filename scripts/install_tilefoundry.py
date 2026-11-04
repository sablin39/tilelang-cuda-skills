#!/usr/bin/env python3
"""Install and verify the suite's pinned TileFoundry in the invoking venv."""

import argparse
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlsplit


PACKAGES = (
    ("tilefoundry", "", "tilefoundry", "src/tilefoundry", "console_scripts", "tilefoundry"),
    ("tilefoundry-nvidia", "targets/nvidia", "tilefoundry_nvidia", "tilefoundry_nvidia", "tilefoundry.targets", "cuda"),
    ("tilefoundry-apple", "targets/apple", "tilefoundry_apple", "tilefoundry_apple", "tilefoundry.targets", "amx"),
)
RECEIPT = ".tilefoundry-install.json"


def check_source(source, pin):
    if not (source / ".git").exists():
        raise ValueError(f"TileFoundry checkout missing: {source}")
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if not pin or head != pin:
        raise ValueError("TileFoundry source does not match the suite pin")
    if not (source / "third_party/cutlass/include/cute/tensor.hpp").is_file():
        raise ValueError("CUTLASS headers missing; rerun install-skills.sh")


def inspect_packages(source):
    """Read metadata and module locations without importing target packages."""
    versions = {}
    for name, relative, module, module_dir, group, entry in PACKAGES:
        project = source / relative
        dist = metadata.distribution(name)
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        url = urlsplit(direct.get("url", ""))
        if (url.scheme != "file" or url.netloc not in ("", "localhost")
                or Path(unquote(url.path)).resolve() != project.resolve()
                or not direct.get("dir_info", {}).get("editable")):
            raise ValueError(f"{name} is not installed editable from {project}")
        spec = importlib.util.find_spec(module)
        expected = (project / module_dir).resolve()
        if spec is None or spec.origin is None or Path(spec.origin).resolve().parent != expected:
            raise ValueError(f"{module} imports from a different checkout; check PYTHONPATH")
        if not any(ep.group == group and ep.name == entry for ep in dist.entry_points):
            raise ValueError(f"{name} is missing its {entry} entry point")
        versions[name] = dist.version
    if len(set(versions.values())) != 1:
        raise ValueError(f"TileFoundry core/vendor versions disagree: {versions}")
    for name in ("cmake", "ninja"):
        metadata.distribution(name)
        if not os.access(Path(sys.prefix) / "bin" / name, os.X_OK):
            raise ValueError(f"{name} executable missing from {sys.prefix}/bin")
    if not os.access(Path(sys.prefix) / "bin/tilefoundry", os.X_OK):
        raise ValueError("TileFoundry CLI executable missing")
    return versions


def record(source, pin, versions):
    return {"schema_version": 1, "source": str(source), "pin": pin,
            "python": os.path.abspath(sys.executable), "packages": versions}


def check(source, pin):
    check_source(source, pin)
    if (source / "tilefoundry.egg-info").exists():
        raise ValueError("Obsolete root tilefoundry.egg-info remains; rerun installation")
    versions = inspect_packages(source)
    path = Path(sys.prefix) / RECEIPT
    if not path.is_file() or json.loads(path.read_text()) != record(source, pin, versions):
        raise ValueError("TileFoundry installation is unverified or stale for this pin; rerun installation")
    print(f"TileFoundry environment PASS: {sys.executable} @ {pin[:12]}")


def install(source, pin):
    check_source(source, pin)
    if sys.prefix == sys.base_prefix:
        raise ValueError("Select a virtual environment with --python, or use the suite's default .venv")
    receipt = Path(sys.prefix) / RECEIPT
    receipt.unlink(missing_ok=True)
    # This root metadata directory belongs to the old monolithic package.
    # The current core's src/tilefoundry.egg-info remains valid and is preserved.
    legacy = source / "tilefoundry.egg-info"
    if legacy.is_symlink() or legacy.is_file():
        legacy.unlink()
    elif legacy.is_dir():
        shutil.rmtree(legacy)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    if subprocess.run([sys.executable, "-m", "pip", "--version"], env=env,
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], env=env, check=True)
    command = [sys.executable, "-m", "pip", "install"]
    for _, relative, *_ in PACKAGES:
        command.extend(["-e", str(source / relative)])
    subprocess.run(command + ["cmake", "ninja"], env=env, check=True)
    # A fresh interpreter must process newly installed editable .pth files.
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--source", str(source),
                    "--pin", pin, "--verify-install"], env=env, check=True)


def verify_install(source, pin):
    check_source(source, pin)
    versions = inspect_packages(source)
    cli = Path(sys.prefix) / "bin/tilefoundry"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    env["PATH"] = str(cli.parent) + os.pathsep + env.get("PATH", "")
    with tempfile.TemporaryDirectory(prefix="tilefoundry-smoke-") as tmp:
        output = subprocess.check_output([sys.executable, str(cli), "target", "list"],
                                         cwd=tmp, env=env, text=True)
    if not all(name in output for name in ("CpuTarget", "CudaTarget", "AmxTarget")):
        raise ValueError(f"TileFoundry CLI did not register all bundled targets:\n{output}")
    with tempfile.NamedTemporaryFile(mode="w", dir=sys.prefix, delete=False) as tmp:
        json.dump(record(source, pin, versions), tmp, indent=2)
        tmp.write("\n")
    os.replace(tmp.name, Path(sys.prefix) / RECEIPT)
    check(source, pin)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pin", required=True)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check-only", action="store_true")
    actions.add_argument("--verify-install", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if sys.version_info < (3, 12):
            raise ValueError("TileFoundry requires Python 3.12+; select it with --python")
        action = check if args.check_only else verify_install if args.verify_install else install
        action(args.source.resolve(), args.pin)
    except (OSError, ValueError, metadata.PackageNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: TileFoundry installation: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
