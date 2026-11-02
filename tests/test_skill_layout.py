import importlib.util
from pathlib import Path
import shutil
import os
import subprocess
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("skill_layout", Path(__file__).resolve().parents[1] / "scripts/skill_layout.py")
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)


class SkillLayoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source, self.dest = self.root / "source", self.root / "installed"
        self.a, self.b = self.source / "loop", self.source / "nested/specialist"
        for p in (self.a, self.b):
            p.mkdir(parents=True)
            (p / "SKILL.md").write_text(f"---\nname: {p.name}\ndescription: Test.\n---\n")
        with (self.a / "SKILL.md").open("a") as f:
            f.write("[route](../nested/specialist/SKILL.md#evidence)\n```text\n[example](missing.md)\n```\n")
        self.dest.mkdir()

    def copies(self):
        for p in (self.a, self.b):
            shutil.copytree(p, self.dest / p.name)

    def test_source_and_symlink_install(self):
        self.assertEqual(layout.check(self.source), [])
        for p in (self.a, self.b):
            (self.dest / p.name).symlink_to(p, target_is_directory=True)
        self.assertEqual(layout.check(self.source, self.dest), [])
        with self.assertRaises(ValueError):
            layout.relocate(self.source, self.dest)

    def test_flat_copy_and_anchor(self):
        self.copies()
        self.assertTrue(layout.check(self.source, self.dest))
        layout.relocate(self.source, self.dest)
        self.assertEqual(layout.check(self.source, self.dest), [])
        text = (self.dest / "loop/SKILL.md").read_text()
        self.assertIn("../specialist/SKILL.md#evidence", text)
        self.assertIn("[example](missing.md)", text)

    def test_missing_and_stale_are_read_only_failures(self):
        self.copies()
        layout.relocate(self.source, self.dest)
        entry = self.dest / "specialist/SKILL.md"
        entry.write_text("stale")
        self.assertTrue(layout.check(self.source, self.dest))
        self.assertEqual(entry.read_text(), "stale")
        entry.unlink()
        self.assertTrue(layout.check(self.source, self.dest))

    def test_reachable_reference_checked(self):
        ref = self.a / "guide.md"
        ref.write_text("[missing](absent.py)\n")
        with (self.a / "SKILL.md").open("a") as f:
            f.write("[guide](guide.md)\n")
        self.assertTrue(any("absent.py" in e for e in layout.check(self.source)))

    def test_standalone_check_does_not_create_cache(self):
        script = Path(__file__).resolve().parents[1] / "scripts/install-skills.sh"
        launcher = self.root / "standalone/install-skills.sh"
        launcher.parent.mkdir()
        shutil.copy2(script, launcher)
        cache = self.root / "absent-cache"
        env = dict(os.environ, SUITE_CACHE=str(cache))
        result = subprocess.run(["bash", str(launcher), "--check-only"],
                                cwd=self.root, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("no cached suite", result.stderr)
        self.assertFalse(cache.exists())


if __name__ == "__main__":
    unittest.main()
