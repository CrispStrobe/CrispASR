"""Notice-only repair must preserve native payloads and framework symlinks."""

import argparse
import importlib.util
import io
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile


class PublicationAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        root = Path(cls.temporary.name)
        with patch.dict(os.environ, HEAVY_OUT=str(root / "out"), HEAVY_SCRATCH=str(root / "scratch")):
            spec = importlib.util.spec_from_file_location(
                "publication_audit", Path(__file__).resolve().parents[1] /
                "tools/ci-heavy/release_publication_audit.py")
            cls.audit = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.audit)

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.notices = {"LICENSE": b"project license\n", "THIRD_PARTY_NOTICES.txt": b"third-party notices\n"}

    def tar(self, name="original.tar.gz", notice=None):
        path = self.root / name
        with tarfile.open(path, "w:gz") as archive:
            item = tarfile.TarInfo("root/library")
            item.mode = 0o755
            payload = b"\x00unchanged native binary\xff"
            item.size = len(payload)
            archive.addfile(item, io.BytesIO(payload))
            link = tarfile.TarInfo("root/library-alias")
            link.type, link.linkname, link.mode = tarfile.SYMTYPE, "library", 0o777
            archive.addfile(link)
            if notice is not None:
                item = tarfile.TarInfo("root/LICENSE")
                item.size = len(notice)
                archive.addfile(item, io.BytesIO(notice))
        return path

    def test_tar_preserves_binary_mode_and_link(self):
        path = self.tar()
        before = self.audit.inventory(path)
        repaired, after, added = self.audit.repair_notices(path, before, self.notices)
        self.assertEqual(self.audit.inventory(repaired), after)
        self.assertTrue(all(after[name] == item for name, item in before.items()))
        self.assertEqual(set(added), {"root/LICENSE", "root/THIRD_PARTY_NOTICES.txt"})
        self.assertEqual(after["root/library"]["mode"], 0o755)
        self.assertEqual(after["root/library-alias"]["link"], "library")

    def test_zip_preserves_framework_symlink_type_and_bytes(self):
        path = self.root / "framework.zip"
        with zipfile.ZipFile(path, "w") as archive:
            binary = zipfile.ZipInfo("root/Versions/A/library")
            binary.create_system, binary.external_attr = 3, 0o100755 << 16
            archive.writestr(binary, b"signed Mach-O payload")
            link = zipfile.ZipInfo("root/Versions/Current")
            link.create_system, link.external_attr = 3, 0o120777 << 16
            archive.writestr(link, b"A")
        before = self.audit.inventory(path)
        repaired, after, _ = self.audit.repair_notices(path, before, self.notices)
        self.assertEqual(self.audit.inventory(repaired), after)
        self.assertTrue(all(after[name] == item for name, item in before.items()))
        with zipfile.ZipFile(repaired) as archive:
            self.assertEqual(archive.getinfo("root/Versions/Current").external_attr, link.external_attr)
            self.assertEqual(archive.read("root/Versions/Current"), b"A")

    def test_conflicting_project_license_is_never_overwritten(self):
        path = self.tar(notice=b"different license")
        before = self.audit.sha_file(path)
        with self.assertRaisesRegex(ValueError, "replace an existing notice"):
            self.audit.repair_notices(path, self.audit.inventory(path), self.notices)
        self.assertEqual(self.audit.sha_file(path), before)

    def test_traversal_member_rejected(self):
        path = self.root / "unsafe.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("../outside", b"no")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            self.audit.inventory(path)

    def test_duplicate_members_rejected(self):
        path = self.root / "duplicate.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            for _ in range(2):
                item = tarfile.TarInfo("root/binary")
                item.size = 1
                archive.addfile(item, io.BytesIO(b"x"))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.audit.inventory(path)

    def test_unknown_archive_family_cannot_be_published_by_repair(self):
        path = self.tar()
        identity = self.audit.sha_file(path)
        asset = dict(name="crispasr-unknown.tar.gz", id=1, size=identity["bytes"],
                     digest="sha256:" + identity["sha256"], browser_download_url="https://example.invalid/archive")

        def api(name):
            if name.startswith("actions/runs/"):
                return dict(status="completed", conclusion="success", head_sha="pinned",
                            path=".github/workflows/release.yml")
            return {"assets": [asset]}

        def download(url, destination):
            shutil.copyfile(path, destination)
            return identity

        args = argparse.Namespace(release_run=1, source="pinned", tag="v0.8.45", repair_notices=True)
        with patch.object(self.audit, "api", side_effect=api), \
                patch.object(self.audit, "tag_file", side_effect=lambda tag, name: self.notices[name]), \
                patch.object(self.audit, "download", side_effect=download), \
                patch.object(self.audit.subprocess, "run") as publish:
            with self.assertRaisesRegex(ValueError, "notices missing"):
                self.audit.native(args, {})
            publish.assert_not_called()


if __name__ == "__main__":
    unittest.main()
