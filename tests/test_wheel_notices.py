"""License notice loss must stop platform-wheel packaging."""

import importlib.util
import json
import tempfile
from types import SimpleNamespace
import unittest
import zipfile
from pathlib import Path, PureWindowsPath
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "wheel_notices", Path(__file__).resolve().parents[1] / "python/tools/stage_wheel_notices.py")
notices = importlib.util.module_from_spec(spec)
spec.loader.exec_module(notices)
stager_spec = importlib.util.spec_from_file_location(
    "wheel_stager", Path(__file__).resolve().parents[1] / "python/tools/stage_libs.py")
stager = importlib.util.module_from_spec(stager_spec)
stager_spec.loader.exec_module(stager)


class HostProvidedRuntimeTest(unittest.TestCase):
    """#557: ROCm's libomp is host-provided for +hip only, never globally."""

    LDD = ("\tlibomp.so => not found\n"
           "\tlibamdhip64.so.6 => not found\n"
           "\tlibvulkan.so.1 => not found\n")

    def run_vendor(self, extra):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp)
            (pkg / "libcrispasr.so").write_bytes(b"\x7fELF")
            ldd = SimpleNamespace(stdout=self.LDD)
            with patch.object(stager.shutil, "which", return_value="/usr/bin/ldd"), \
                 patch.object(stager.subprocess, "run", return_value=ldd):
                return stager.vendor_linux_deps(pkg, {}, extra)

    def test_libomp_unresolved_without_variant_opt_in(self):
        vendored, unresolved, host = self.run_vendor(())
        self.assertEqual((vendored, unresolved), ([], ["libomp.so"]))
        self.assertEqual(sorted(host), ["libamdhip64.so.6", "libvulkan.so.1"])

    def test_variant_host_runtime_is_neither_vendored_nor_missing(self):
        vendored, unresolved, host = self.run_vendor(("libomp.so",))
        self.assertEqual((vendored, unresolved), ([], []))
        self.assertEqual(sorted(host), ["libamdhip64.so.6", "libomp.so", "libvulkan.so.1"])


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

    def stage(self, provenance=None):
        # Keep these controls portable and independent of the runner's apt DB.
        with patch.object(notices.platform, "system", return_value="Windows"):
            return notices.stage(self.bundle, self.package, self.project, provenance)

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

    def test_windows_notice_paths_use_zip_separators(self):
        root = PureWindowsPath("C:/package/licenses")
        path = root / "native" / "component" / "COPYING"
        self.assertEqual(notices.archive_path(path, root), "native/component/COPYING")

    def test_unrecorded_native_payload_is_rejected(self):
        (self.package / "unrecorded.dll").write_bytes(b"native bytes")
        with self.assertRaisesRegex(ValueError, "requires original source provenance"):
            self.stage()

    def test_added_runtime_uses_its_recorded_source_even_outside_ldconfig(self):
        source = self.package.parent / "unregistered-runtime.so.1"
        source.write_bytes(b"exact installed runtime")
        library = self.package / "liboutside.so.1"
        library.write_bytes(source.read_bytes())
        copyright_file = self.package.parent / "copyright"
        copyright_file.write_bytes(b"the actual runtime copyright")
        provenance = self.package.parent / "sources.json"
        provenance.write_text(json.dumps({"libraries": {library.name: dict(
            kind="installed-runtime", source_path=str(source), source_sha256=notices.digest(source),
            staged_sha256=notices.digest(library))}}))
        with patch.object(notices.platform, "system", return_value="Linux"), \
                patch.object(notices, "installed_package", return_value=("outside:amd64", "source identity", copyright_file)) as lookup:
            manifest = notices.stage(self.bundle, self.package, self.project, provenance)
        lookup.assert_called_once_with(source)
        self.assertEqual((self.package / "licenses/packages/outside_amd64/copyright").read_bytes(), copyright_file.read_bytes())
        self.assertNotIn("source_path", manifest["native_libraries"][library.name])
        self.assertEqual(manifest["native_libraries"][library.name]["copyright_package"], "outside:amd64")

    def test_provenance_cannot_hide_changed_library_bytes(self):
        library = self.package / "wrong.dll"
        library.write_bytes(b"changed")
        provenance = self.package.parent / "sources.json"
        provenance.write_text(json.dumps({"libraries": {library.name: dict(
            kind="release-bundle", source_sha256="0" * 64, staged_sha256="0" * 64)}}))
        with self.assertRaisesRegex(ValueError, "library changed"):
            self.stage(provenance)

    def test_stager_records_actual_ldd_source_path_without_loader_cache(self):
        core = self.package / "libcrispasr.so"
        core.write_bytes(b"core")
        source = self.package.parent / "unregistered.so.1"
        source.write_bytes(b"external runtime")
        provenance = {}

        def command(args, **kwargs):
            if args[0] == "ldd" and Path(args[1]) == core:
                return SimpleNamespace(stdout=f"liboutside.so.1 => {source} (0x0)\n")
            return SimpleNamespace(stdout="")

        with patch.object(stager.shutil, "which", return_value="/tools/tool"), \
                patch.object(stager.subprocess, "run", side_effect=command) as calls:
            copied, missing, host = stager.vendor_linux_deps(self.package, provenance)
        self.assertEqual(copied, ["liboutside.so.1"])
        self.assertEqual((missing, host), ([], []))
        self.assertEqual(provenance[copied[0]]["source_path"], str(source.resolve()))
        self.assertEqual(provenance[copied[0]]["source_sha256"], notices.digest(source))
        self.assertFalse(any(call.args[0][0] == "ldconfig" for call in calls.call_args_list))

    def wheel(self, omit=None, change=None):
        result = self.package.parent / "test.whl"
        with zipfile.ZipFile(result, "w") as archive:
            for path in (self.package / "licenses").rglob("*"):
                if path.is_file() and path.name != omit:
                    name = "crispasr/" + path.relative_to(self.package).as_posix()
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
