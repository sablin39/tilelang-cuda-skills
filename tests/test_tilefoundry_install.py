import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tf_install", ROOT / "scripts/install_tilefoundry.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class PackageChecks(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root / "source"
        self.venv = self.root / "venv"
        self.dists, self.specs = {}, {}
        for name, relative, module, module_dir, group, entry in installer.PACKAGES:
            project = self.source / relative
            origin = project / module_dir / "__init__.py"
            origin.parent.mkdir(parents=True, exist_ok=True)
            origin.write_text("")
            direct = json.dumps({"url": project.as_uri(), "dir_info": {"editable": True}})
            self.dists[name] = SimpleNamespace(version="1.0", entry_points=[
                SimpleNamespace(group=group, name=entry)], read_text=lambda _, data=direct: data)
            self.specs[module] = SimpleNamespace(origin=str(origin))
        (self.venv / "bin").mkdir(parents=True)
        for name in ("tilefoundry", "cmake", "ninja"):
            path = self.venv / "bin" / name
            path.write_text("#!/bin/sh\nexit 0\n")
            path.chmod(0o755)
            if name != "tilefoundry":
                self.dists[name] = SimpleNamespace(version="1.0")
        def distribution(name):
            if name not in self.dists:
                raise installer.metadata.PackageNotFoundError(name)
            return self.dists[name]
        for mocker in (patch.object(installer.metadata, "distribution", side_effect=distribution),
                       patch.object(installer.importlib.util, "find_spec", side_effect=self.specs.get),
                       patch.object(installer.sys, "prefix", str(self.venv))):
            mocker.start()
            self.addCleanup(mocker.stop)

    def test_matching_editable_packages(self):
        self.assertEqual(set(installer.inspect_packages(self.source).values()), {"1.0"})

    def test_missing_vendor_and_mismatched_version_rejected(self):
        self.dists["tilefoundry-nvidia"].version = "0.9"
        with self.assertRaisesRegex(ValueError, "versions disagree"):
            installer.inspect_packages(self.source)
        del self.dists["tilefoundry-nvidia"]
        with self.assertRaises(installer.metadata.PackageNotFoundError):
            installer.inspect_packages(self.source)

    def test_same_version_wrong_checkout_and_shadowed_import_rejected(self):
        dist = self.dists["tilefoundry-nvidia"]
        original = dist.read_text
        dist.read_text = lambda _: json.dumps({"url": (self.root / "old").as_uri(),
                                              "dir_info": {"editable": True}})
        with self.assertRaisesRegex(ValueError, "not installed editable"):
            installer.inspect_packages(self.source)
        dist.read_text = original
        self.specs["tilefoundry_nvidia"].origin = str(self.root / "old/__init__.py")
        with self.assertRaisesRegex(ValueError, "different checkout"):
            installer.inspect_packages(self.source)

    def test_missing_registration_and_build_tool_rejected(self):
        dist = self.dists["tilefoundry-nvidia"]
        entry_points = dist.entry_points
        dist.entry_points = []
        with self.assertRaisesRegex(ValueError, "entry point"):
            installer.inspect_packages(self.source)
        dist.entry_points = entry_points
        (self.venv / "bin/ninja").unlink()
        with self.assertRaisesRegex(ValueError, "ninja executable"):
            installer.inspect_packages(self.source)

    def test_receipt_detects_pin_change_and_check_is_read_only(self):
        path = self.venv / installer.RECEIPT
        versions = installer.inspect_packages(self.source)
        with patch.object(installer, "check_source"), patch.object(installer.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "unverified or stale"):
                installer.check(self.source, "pin-a")
            self.assertFalse(path.exists())
            path.write_text(json.dumps(installer.record(self.source, "pin-a", versions)))
            before = path.stat().st_mtime_ns
            installer.check(self.source, "pin-a")
            self.assertEqual(before, path.stat().st_mtime_ns)
            with self.assertRaisesRegex(ValueError, "unverified or stale"):
                installer.check(self.source, "pin-b")
            run.assert_not_called()

    def test_failed_install_invalidates_receipt_and_removes_only_legacy_metadata(self):
        legacy = self.source / "tilefoundry.egg-info"
        legacy.mkdir()
        (legacy / "PKG-INFO").write_text("old")
        current = self.source / "src/tilefoundry.egg-info"
        current.mkdir()
        receipt = self.venv / installer.RECEIPT
        receipt.write_text("{}")
        def run(command, **kwargs):
            if command[2:4] == ["pip", "--version"]:
                return SimpleNamespace(returncode=0)
            raise subprocess.CalledProcessError(1, command)
        with patch.object(installer, "check_source"), patch.object(installer.subprocess, "run", side_effect=run):
            with self.assertRaises(subprocess.CalledProcessError):
                installer.install(self.source, "pin-a")
        self.assertFalse(legacy.exists())
        self.assertTrue(current.is_dir())
        self.assertFalse(receipt.exists())

    def test_missing_pip_bootstraps_and_verification_uses_fresh_interpreter(self):
        commands = []
        def run(command, **kwargs):
            commands.append(command)
            return SimpleNamespace(returncode=int(command[2:4] == ["pip", "--version"]))
        with patch.object(installer, "check_source"), patch.object(installer.subprocess, "run", side_effect=run):
            installer.install(self.source, "pin-a")
        self.assertTrue(any(command[2:4] == ["ensurepip", "--upgrade"] for command in commands))
        install = next(command for command in commands if command[2:4] == ["pip", "install"])
        self.assertEqual(install.count("-e"), 3)
        self.assertEqual(commands[-1][0], sys.executable)
        self.assertIn("--verify-install", commands[-1])
        self.assertFalse((self.venv / installer.RECEIPT).exists())

    def test_failed_cli_registration_does_not_record_success(self):
        with patch.object(installer, "check_source"), \
             patch.object(installer.subprocess, "check_output", return_value="CpuTarget()\n"):
            with self.assertRaisesRegex(ValueError, "did not register"):
                installer.verify_install(self.source, "pin-a")
        self.assertFalse((self.venv / installer.RECEIPT).exists())


class SourceUpgradeTests(unittest.TestCase):
    """Exercise source repair with real local Git submodules, without a network."""
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.env = dict(os.environ, GIT_ALLOW_PROTOCOL="file", PYTHONDONTWRITEBYTECODE="1")
        self.cutlass = self.root / "cutlass"
        self.tf = self.root / "tilefoundry"
        self.suite = self.root / "suite"
        for repo in (self.cutlass, self.tf, self.suite):
            repo.mkdir()
            self.git(repo, "init", "-q")
            self.git(repo, "config", "user.name", "Installer test")
            self.git(repo, "config", "user.email", "installer@example.invalid")
        header = self.cutlass / "include/cute/tensor.hpp"
        header.parent.mkdir(parents=True)
        header.write_text("// fixture\n")
        self.commit(self.cutlass)
        for relative in ("src/tilefoundry/__init__.py", "targets/nvidia/pyproject.toml", "targets/apple/pyproject.toml"):
            p = self.tf / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("")
        self.git(self.tf, "submodule", "add", "-q", str(self.cutlass), "third_party/cutlass")
        self.commit(self.tf)
        self.git(self.suite, "submodule", "add", "-q", str(self.tf), "external/TileFoundry")
        (self.suite / "scripts").mkdir()
        (self.suite / "skills").mkdir()
        shutil.copy2(ROOT / "scripts/install-skills.sh", self.suite / "scripts/install-skills.sh")
        shutil.copy2(ROOT / "scripts/skill_layout.py", self.suite / "scripts/skill_layout.py")
        # Stop at the package-install boundary; do not touch host skills or packages.
        (self.suite / "scripts/install_tilefoundry.py").write_text(
            "import pathlib, sys\n"
            "if '--check-only' in sys.argv: sys.exit(0)\n"
            "pathlib.Path('package-install-reached').write_text(sys.executable)\n"
            "sys.exit(42)\n")
        self.commit(self.suite)
        self.checkout = self.suite / "external/TileFoundry"

    def git(self, repo, *args):
        return subprocess.check_output(["git", "-C", str(repo), *args], env=self.env,
                                       stderr=subprocess.PIPE, text=True).strip()

    def commit(self, repo):
        self.git(repo, "add", ".")
        self.git(repo, "commit", "-qm", "fixture")

    def run_installer(self, *args):
        return subprocess.run(["bash", str(self.suite / "scripts/install-skills.sh"),
                               "--python", sys.executable, "--target", "codex", *args],
                              cwd=self.root, env=self.env, capture_output=True, text=True)

    def test_repairs_missing_nested_dependency_at_matching_pin(self):
        self.assertEqual(self.git(self.tf, "rev-parse", "HEAD"), self.git(self.checkout, "rev-parse", "HEAD"))
        header = self.checkout / "third_party/cutlass/include/cute/tensor.hpp"
        self.assertFalse(header.exists())
        result = self.run_installer()
        self.assertEqual(result.returncode, 42, result.stdout + result.stderr)
        self.assertTrue(header.is_file())
        self.assertTrue((self.root / "package-install-reached").exists())

    def test_mismatched_source_restored_to_suite_pin(self):
        self.git(self.checkout, "config", "user.name", "Installer test")
        self.git(self.checkout, "config", "user.email", "installer@example.invalid")
        (self.checkout / "new.txt").write_text("different revision")
        self.commit(self.checkout)
        result = self.run_installer()
        self.assertEqual(result.returncode, 42, result.stdout + result.stderr)
        self.assertEqual(self.git(self.checkout, "rev-parse", "HEAD"),
                         self.git(self.suite, "rev-parse", "HEAD:external/TileFoundry"))

    def test_check_only_does_not_initialize_missing_dependency(self):
        result = self.run_installer("--check-only")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.checkout / "third_party/cutlass/include/cute/tensor.hpp").exists())
        self.assertFalse((self.root / "package-install-reached").exists())
        self.assertIn("TileFoundry or CUTLASS", result.stderr)

    def test_cached_update_failure_stops_before_install(self):
        launcher = self.root / "standalone/scripts/install-skills.sh"
        launcher.parent.mkdir(parents=True)
        shutil.copy2(ROOT / "scripts/install-skills.sh", launcher)
        self.git(self.suite, "remote", "add", "origin", str(self.root / "does-not-exist"))
        env = dict(self.env, SUITE_CACHE=str(self.suite))
        result = subprocess.run(["bash", str(launcher), "--python", sys.executable],
                                cwd=self.root, env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("installation stopped", result.stderr)
        self.assertFalse((self.root / "package-install-reached").exists())

    @unittest.skipUnless(sys.version_info >= (3, 12), "default environment needs Python 3.12+")
    def test_default_environment_is_created(self):
        result = subprocess.run(["bash", str(self.suite / "scripts/install-skills.sh"),
                                 "--target", "codex"], cwd=self.root, env=self.env,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 42, result.stdout + result.stderr)
        python = self.suite / ".venv/bin/python"
        self.assertTrue(python.is_file())
        self.assertEqual((self.root / "package-install-reached").read_text(), str(python))

    def test_successful_check_only_leaves_files_unchanged(self):
        self.git(self.suite, "submodule", "update", "--init", "--recursive")
        skill = self.suite / "skills/example"
        skill.mkdir()
        (skill / "SKILL.md").write_text("---\nname: example\ndescription: Fixture.\n---\n")
        dest = self.root / ".codex/skills"
        dest.mkdir(parents=True)
        (dest / "example").symlink_to(skill)
        for relative in ("SKILL.md", "scripts/query.py", "wiki/entry.md"):
            p = dest / "KernelWiki" / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("")
        def snapshot():
            return {str(p): (p.read_bytes(), p.stat().st_mtime_ns)
                    for p in self.root.rglob("*") if p.is_file() and ".git" not in p.parts}
        before = snapshot()
        result = self.run_installer("--check-only")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(snapshot(), before)


if __name__ == "__main__":
    unittest.main()
