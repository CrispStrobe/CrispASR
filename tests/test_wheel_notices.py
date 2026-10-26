"""License notice loss must stop platform-wheel packaging."""

import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "wheel_notices", Path(__file__).resolve().parents[1] / "python/tools/stage_wheel_notices.py")
notices = importlib.util.module_from_spec(spec)
spec.loader.exec_module(notices)


class WheelNoticesTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.bundle = root / "bundle"
        self.bundle.mkdir()
        self.package = root / "crispasr"
        self.package.mkdir()
        (self.package / "_binding.py").write_text("")
        (self.bundle / "LICENSE").write_text("original native license\n")
        (self.bundle / "THIRD_PARTY_NOTICES.txt").write_text("original third-party notices\n")
        self.project = root / "pyproject.toml"
        self.project.write_text('[project]\nversion = "0.8.45"\n'
                                '[tool.setuptools.package-data]\ncrispasr = [\n    "*.so",\n]\n')

    def stage(self):
        # Keep these controls portable and independent of the runner's apt DB.
        with patch.object(notices.platform, "system", return_value="Windows"):
            return notices.stage(self.bundle, self.package, self.project)

    def test_preserves_notice_bytes_and_nested_native_licenses(self):
        source = self.bundle / "licenses/component/COPYING"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"component copyright\n\xff")
        manifest = self.stage()
        destination = self.package / "licenses"
        for name in ["LICENSE", "THIRD_PARTY_NOTICES.txt"]:
            self.assertEqual((destination / name).read_bytes(), (self.bundle / name).read_bytes())
        self.assertEqual((destination / "native/component/COPYING").read_bytes(), source.read_bytes())
        self.assertEqual(manifest["files"]["native/component/COPYING"], notices.digest(source))
        text = self.project.read_text()
        self.assertIn('version = "0.8.45"', text)
        self.assertEqual(text.count('"licenses/**/*"'), 1)
        self.assertIn('"*.so"', text)

    def test_missing_notice_fails_before_modifying_package_metadata(self):
        before = self.project.read_bytes()
        (self.bundle / "THIRD_PARTY_NOTICES.txt").unlink()
        with self.assertRaisesRegex(ValueError, "Missing or empty"):
            self.stage()
        self.assertEqual(self.project.read_bytes(), before)
        self.assertFalse((self.package / "licenses").exists())

    def test_empty_notice_fails(self):
        (self.bundle / "LICENSE").write_text("\n")
        with self.assertRaisesRegex(ValueError, "Missing or empty"):
            self.stage()

    def test_unrecognized_packaging_metadata_fails_closed(self):
        self.project.write_text('[project]\nversion = "0.8.45"\n')
        with self.assertRaisesRegex(ValueError, "package-data"):
            self.stage()
        self.assertFalse((self.package / "licenses").exists())

    def test_stale_notices_are_not_silently_merged(self):
        (self.package / "licenses").mkdir()
        with self.assertRaisesRegex(ValueError, "stale"):
            self.stage()

    def wheel(self, omit=None, change=None):
        result = self.package.parent / "test.whl"
        with zipfile.ZipFile(result, "w") as archive:
            for path in (self.package / "licenses").rglob("*"):
                if path.is_file() and path.name != omit:
                    name = "crispasr/" + str(path.relative_to(self.package))
                    archive.writestr(name, b"changed" if path.name == change else path.read_bytes())
        return result

    def test_finished_wheel_preserves_every_notice(self):
        self.stage()
        notices.verify_wheel(self.wheel(), self.package)

    def test_finished_wheel_cannot_omit_notices(self):
        self.stage()
        with self.assertRaisesRegex(ValueError, "omitted"):
            notices.verify_wheel(self.wheel(omit="THIRD_PARTY_NOTICES.txt"), self.package)

    def test_finished_wheel_cannot_change_notice_bytes(self):
        self.stage()
        with self.assertRaisesRegex(ValueError, "changed"):
            notices.verify_wheel(self.wheel(change="LICENSE"), self.package)


if __name__ == "__main__":
    unittest.main()
