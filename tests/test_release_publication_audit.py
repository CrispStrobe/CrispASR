"""Notice-only repair must preserve native payloads and framework symlinks."""

import argparse
import copy
import hashlib
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
                            path=".github/workflows/release.yml", event="push", head_branch="v0.8.45")
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

    def test_manual_or_wrong_ref_producer_cannot_authorize_native_repair(self):
        args = argparse.Namespace(release_run=1, source="pinned", tag="v0.8.45", repair_notices=True)
        for event, branch in [("workflow_dispatch", "v0.8.45"), ("push", "main")]:
            run = dict(status="completed", conclusion="success", head_sha="pinned",
                       path=".github/workflows/release.yml", event=event, head_branch=branch)
            with self.subTest(event=event, branch=branch), \
                    patch.object(self.audit, "api", return_value=run), \
                    patch.object(self.audit.subprocess, "run") as publish:
                with self.assertRaises(AssertionError):
                    self.audit.native(args, {})
                publish.assert_not_called()

    def test_nested_notices_do_not_mask_missing_project_root_notices(self):
        path = self.root / "nested.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("root/program", b"binary")
            for name, body in self.notices.items():
                archive.writestr("root/third_party/" + name, body)
        before = self.audit.inventory(path)
        self.assertFalse(self.audit.root_notices_present(before, self.notices))
        _, after, _ = self.audit.repair_notices(path, before, self.notices)
        self.assertTrue(self.audit.root_notices_present(after, self.notices))
        self.assertTrue(all(after[n] == value for n, value in before.items()))

    def test_windows_notice_newlines_are_accepted_without_rewriting(self):
        path = self.root / "windows.zip"
        with zipfile.ZipFile(path, "w") as archive:
            for name, body in self.notices.items():
                archive.writestr("root/" + name, body.replace(b"\n", b"\r\n"))
        members = self.audit.inventory(path)
        self.assertTrue(self.audit.root_notices_present(members, self.notices))
        self.assertIsNone(self.audit.repair_notices(path, members, self.notices))

    def test_cuda_runtime_checksums_require_exact_complete_set(self):
        names = ["cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"]
        rows = ["a" * 64 + "  " + name for name in names]
        self.assertEqual(set(self.audit.parse_runtime_checksums("\n".join(rows), "cuda126")), set(names))
        for broken in ["", "\n".join(rows[:2]), "\n".join(rows + [rows[0]]),
                       "\n".join(rows + ["b" * 64 + "  unknown.dll"])]:
            with self.subTest(broken=broken), self.assertRaises(ValueError):
                self.audit.parse_runtime_checksums(broken, "cuda")

    def test_cuda_split_and_runtime_payload_sets_are_complete(self):
        records = {}
        for flavor in ["cuda", "cuda126", "cuda13"]:
            major = "13" if flavor == "cuda13" else "12"
            runtime = {}
            for filename in [f"cudart64_{major}.dll", f"cublas64_{major}.dll", f"cublasLt64_{major}.dll"]:
                body = (flavor + filename).encode()
                runtime[filename] = dict(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
                if flavor != "cuda126":
                    records[filename] = runtime[filename].copy()
            prefix = f"crispasr-windows-x86_64-{flavor}"
            project = dict(bytes=4, sha256=hashlib.sha256(b"core").hexdigest())
            full = {"full/" + name: row for name, row in runtime.items()}
            full["full/crispasr.exe"] = project
            records[prefix + ".zip"] = {"members": full}
            records[prefix + "-non-cuda.zip"] = {"members": {"split/crispasr.exe": project}}
            records[prefix + "-runtime-sha256.txt"] = {"runtime_checksums": "\n".join(
                row["sha256"] + "  " + name for name, row in runtime.items())}
            if flavor != "cuda13":
                records[flavor + "-runtime.zip"] = {"members": runtime}
        self.audit.validate_cuda_payloads(records)
        for name, member in [("cuda-runtime.zip", "cudart64_12.dll"),
                             ("crispasr-windows-x86_64-cuda-non-cuda.zip", "split/crispasr.exe")]:
            broken = copy.deepcopy(records)
            del broken[name]["members"][member]
            with self.subTest(asset=name), self.assertRaises(AssertionError):
                self.audit.validate_cuda_payloads(broken)

    def test_native_aliases_are_resolved_and_wrong_wheel_payloads_fail(self):
        payload = b"qualified native library"
        digest = hashlib.sha256(payload).hexdigest()
        members = {"root/lib/libcrispasr.so.1": dict(bytes=len(payload), sha256=digest, mode=0o755),
                   "root/lib/libcrispasr.so": dict(link="libcrispasr.so.1", link_type="symlink", mode=0o777)}
        expected = self.audit.bundle_libraries(members)
        self.assertEqual(set(expected), {"libcrispasr.so", "libcrispasr.so.1"})
        for wrong in [False, True]:
            path = self.root / "native.whl"
            with zipfile.ZipFile(path, "w") as archive:
                for name in expected:
                    archive.writestr("crispasr/" + name, b"foreign bytes" if wrong else payload)
            with zipfile.ZipFile(path) as archive:
                if wrong:
                    with self.assertRaises(AssertionError):
                        self.audit.verify_native_wheel(archive, expected, {})
                else:
                    self.assertEqual(self.audit.verify_native_wheel(archive, expected, {}), 2)

    def test_native_asset_set_requires_both_new_bundles_and_rejects_extras(self):
        baseline = {"crispasr-v0.8.44-xcframework.zip", "libcrispasr-linux-x86_64.tar.gz", "old.whl"}
        old = {"crispasr-v0.8.46-xcframework.zip", "libcrispasr-linux-x86_64.tar.gz", "new.whl"}
        self.audit.require_native_asset_set(baseline, old, "0.8.46")
        current = {"crispasr-v0.8.47-xcframework.zip", "libcrispasr-linux-x86_64.tar.gz",
                   "libcrispasr-linux-x86_64-vulkan.tar.gz", "libcrispasr-linux-arm64-cuda.tar.gz"}
        self.audit.require_native_asset_set(baseline, current, "0.8.47")
        for bundle in ["libcrispasr-linux-x86_64-vulkan.tar.gz", "libcrispasr-linux-arm64-cuda.tar.gz"]:
            with self.subTest(missing=bundle), self.assertRaisesRegex(ValueError, "missing=.*" + bundle):
                self.audit.require_native_asset_set(baseline, current - {bundle}, "0.8.47")
        with self.assertRaisesRegex(ValueError, "unexpected=.*unreviewed"):
            self.audit.require_native_asset_set(baseline, current | {"unreviewed.tar.gz"}, "0.8.47")

    def test_linux_gpu_wheel_set_is_version_gated_and_flavors_exact(self):
        # #557: v0.8.46 and older keep their original exact wheel set.
        self.assertFalse(self.audit.linux_gpu_wheels_expected("0.8.46"))
        self.assertTrue(self.audit.linux_gpu_wheels_expected("0.8.47"))
        self.assertTrue(self.audit.linux_gpu_wheels_expected("0.9.0"))
        self.assertEqual(self.audit.wheel_local_suffix("cpu-wheel-linux-x86_64"), "")
        self.assertEqual(self.audit.wheel_local_suffix("gpu-wheel-linux-x86_64-hip"), "+hip")
        self.assertEqual(self.audit.wheel_local_suffix("gpu-wheel-linux-x86_64-vulkan"), "+vulkan")
        self.assertEqual(self.audit.wheel_local_suffix("gpu-wheel-windows-x86_64-cuda"), "+cuda")
        with self.assertRaises(ValueError):
            self.audit.wheel_local_suffix("gpu-wheel-linux-aarch64-unknown")
        self.assertEqual(set(self.audit.LINUX_GPU_WHEEL_ASSETS.values()),
                         {"libcrispasr-linux-x86_64-vulkan.tar.gz", "libcrispasr-linux-x86_64-hip.tar.gz",
                          "libcrispasr-linux-arm64-cuda.tar.gz"})
        self.assertEqual(self.audit.wheel_local_suffix("gpu-wheel-linux-arm64-cuda"), "+cuda")
        # v0.8.46 has no whl/hip index; requiring it 404'd the real audit.
        self.assertEqual(self.audit.gpu_index_flavors("0.8.46"), ("cuda", "vulkan"))
        self.assertEqual(self.audit.gpu_index_flavors("0.8.47"), ("cuda", "vulkan", "hip"))

    def test_missing_native_wheel_libraries_fail(self):
        path = self.root / "empty.whl"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("crispasr/_binding.py", b"correct Python does not prove native payload")
        with zipfile.ZipFile(path) as archive, self.assertRaises(AssertionError):
            self.audit.verify_native_wheel(archive, {"crispasr.dll": {}}, {})

    def test_failed_asset_promotion_restores_original_without_deleting_it(self):
        self.asset_transaction(fail_readback=True)

    def test_old_asset_removed_only_after_new_public_bytes_verify(self):
        self.asset_transaction(fail_readback=False)

    def test_transient_rollback_rename_errors_are_retried(self):
        self.asset_transaction(fail_readback=True, rollback_failures=2)

    def test_lost_promotion_response_is_recovered_by_asset_id(self):
        self.asset_transaction(fail_readback=False, lost_promotion_response=True)

    def asset_transaction(self, fail_readback, rollback_failures=0, lost_promotion_response=False):
        original = self.tar()
        original_bytes = original.read_bytes()
        old_digest = self.audit.sha_file(original)
        repaired, _, _ = self.audit.repair_notices(original, self.audit.inventory(original), self.notices)
        proof = dict(original=old_digest, repaired=self.audit.sha_file(repaired), published=False)
        old = dict(id=1, name=original.name, browser_download_url="https://example.invalid/" + original.name,
                   digest="sha256:" + old_digest["sha256"])
        assets = {1: old.copy()}
        data = {1: original_bytes}
        events = []
        failures_left = rollback_failures
        lose_response = lost_promotion_response

        def upload(command, **kwargs):
            self.assertNotIn("--clobber", command)
            path = Path(command[4])
            data[2] = path.read_bytes()
            assets[2] = dict(id=2, name=path.name, browser_download_url="https://example.invalid/" + path.name,
                             digest="sha256:" + hashlib.sha256(data[2]).hexdigest())
            events.append("upload-staged")

        def rename(asset_id, name):
            nonlocal failures_left, lose_response
            if asset_id == 1 and name == old["name"] and failures_left:
                failures_left -= 1
                raise OSError("transient rollback rename failure")
            assets[asset_id].update(name=name, browser_download_url="https://example.invalid/" + name)
            events.append(("rename", asset_id, name))
            if asset_id == 2 and name == old["name"] and lose_response:
                lose_response = False
                raise OSError("promotion committed but response lost")
            return assets[asset_id].copy()

        def api(path):
            if path.startswith("releases/assets/"):
                return assets[int(path.rsplit("/", 1)[1])].copy()
            return {"assets": list(assets.values())}

        def download(url, destination):
            name = url.split("?", 1)[0].rsplit("/", 1)[1]
            asset_id = next(i for i, value in assets.items() if value["name"] == name)
            if asset_id == 2 and name == old["name"] and fail_readback:
                raise OSError("public readback failed")
            destination.write_bytes(data[asset_id])
            events.append(("verified", asset_id, name))
            return self.audit.sha_file(destination)

        def delete(asset_id):
            events.append(("delete", asset_id))
            del assets[asset_id]

        args = argparse.Namespace(tag="v0.8.45")
        with patch.object(self.audit.subprocess, "run", side_effect=upload), \
                patch.object(self.audit, "api", side_effect=api), \
                patch.object(self.audit, "rename_asset", side_effect=rename), \
                patch.object(self.audit, "delete_asset", side_effect=delete), \
                patch.object(self.audit, "download", side_effect=download), \
                patch.object(self.audit.time, "sleep"):
            if fail_readback:
                with self.assertRaisesRegex(OSError, "readback failed"):
                    self.audit.publish_repaired(args, old, repaired, original, proof, self.root / "proof.json")
                self.assertEqual(assets[1]["name"], old["name"])
                self.assertEqual(data[1], original_bytes)
                self.assertFalse(any(e[0] == "delete" for e in events if isinstance(e, tuple)))
                self.assertTrue(proof["original_restored"])
                self.assertFalse(proof["published"])
                self.assertEqual(proof["recovery_asset_state"]["original"]["name"], old["name"])
            else:
                self.audit.publish_repaired(args, old, repaired, original, proof, self.root / "proof.json")
                self.assertNotIn(1, assets)
                self.assertEqual(assets[2]["name"], old["name"])
                self.assertEqual(events[-1], ("delete", 1))
                self.assertTrue(proof["original_backup_removed_after_verification"])


if __name__ == "__main__":
    unittest.main()
