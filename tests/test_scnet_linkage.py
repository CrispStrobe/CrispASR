"""Rejection controls for Linux SCNet dependency evidence; no model qualification.

Optional --build-dir/--output-dir also exercises real loader snapshots and the
packager's fresh recheck against the five actual production entrypoints.
"""

import argparse
import copy
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "ci-heavy"))
import scnet_qualify as Q
import scnet_package as P


class LinkageControls(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="scnet-linkage-controls-", dir=ROOT.parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.build = self.root / "build"
        self.build.mkdir()
        (self.build / "CMakeCache.txt").write_text("GGML_BACKEND_DL:BOOL=OFF\n")
        self.logs = self.root / "logs"
        self.environment = dict(os.environ)
        self.binaries = {}
        for name in sorted(Q.LINKAGE_ENTRYPOINTS):
            path = self.build / name
            path.write_bytes(("synthetic entrypoint " + name).encode())
            self.binaries[name] = path
        runtime = self.root / "runtime"
        runtime.mkdir()
        self.dependencies = {}
        for name in sorted(Q.LINKAGE_GGML | {"libopenblas.so.0", "ld-linux-x86-64.so.2"}):
            path = runtime / name
            path.write_bytes(("synthetic dependency " + name).encode())
            self.dependencies[name] = path
        self.resolver_suffix = ""
        self.resolver_code = 0

    def resolver(self, argv, **kwargs):
        self.assertEqual(argv[0], "ldd")
        self.assertEqual(kwargs["env"]["LC_ALL"], "C")
        lines = ["\tlinux-vdso.so.1 (0x1234)"]
        for name, path in sorted(self.dependencies.items()):
            lines.append(f"\t{path} (0x5678)" if name.startswith("ld-linux") else
                         f"\t{name} => {path} (0x5678)")
        return subprocess.CompletedProcess(argv, self.resolver_code, "\n".join(lines) + self.resolver_suffix + "\n", "")

    def mocked_resolver(self):
        stack = ExitStack()
        stack.enter_context(mock.patch.object(Q.platform, "system", return_value="Linux"))
        stack.enter_context(mock.patch.object(Q.subprocess, "run", side_effect=self.resolver))
        return stack

    def snapshot(self, epoch):
        with self.mocked_resolver():
            return Q.snapshot_runtime_linkage(self.binaries, self.build, self.environment, self.logs, epoch)

    def receipt(self):
        before, after = self.snapshot("before"), self.snapshot("after")
        return {"runtime_linkage": Q.compare_runtime_linkage(before, after),
                "binaries": {name: {key: record[key] for key in ("path", "sha256")}
                             for name, record in before["entrypoints"].items()}}

    def test_success_binds_all_entrypoints_dependencies_and_portable_logs(self):
        environment = dict(self.environment)
        qualification = self.receipt()
        after = Q.verify_runtime_linkage_receipt(qualification["runtime_linkage"], qualification["binaries"])
        self.assertEqual(set(after["entrypoints"]), Q.LINKAGE_ENTRYPOINTS)
        self.assertEqual(len(after["dependencies"]), 6)
        self.assertEqual(self.environment, environment)
        self.assertNotIn(str(self.root), json.dumps(qualification))
        for path in self.logs.iterdir():
            self.assertNotIn(str(self.root), path.read_text())
        with self.mocked_resolver():
            current = P.verify_packaging_linkage(qualification, self.binaries, self.build, self.logs)
        self.assertEqual(current["entrypoints"], after["entrypoints"])
        self.assertEqual(current["dependencies"], after["dependencies"])

    def test_changed_dependency_bytes_reject_qualification_and_packaging(self):
        qualification = self.receipt()
        self.dependencies["libopenblas.so.0"].write_bytes(b"different provider bytes")
        changed = self.snapshot("changed")
        proof = Q.compare_runtime_linkage(qualification["runtime_linkage"]["before"], changed)
        self.assertFalse(proof["dependencies_unchanged"])
        with self.assertRaisesRegex(ValueError, "runtime linkage"):
            Q.verify_runtime_linkage_receipt(proof, qualification["binaries"])
        with self.mocked_resolver(), self.assertRaisesRegex(ValueError, "runtime linkage"):
            P.verify_packaging_linkage(qualification, self.binaries, self.build, self.logs)

    def test_changed_entrypoint_bytes_reject(self):
        qualification = self.receipt()
        self.binaries["production_library"].write_bytes(b"changed Session library")
        changed = self.snapshot("changed")
        proof = Q.compare_runtime_linkage(qualification["runtime_linkage"]["before"], changed)
        self.assertFalse(proof["entrypoints_unchanged"])
        with self.assertRaisesRegex(ValueError, "runtime linkage"):
            Q.verify_runtime_linkage_receipt(proof, qualification["binaries"])

    def test_same_name_and_bytes_at_a_different_resolved_path_reject(self):
        qualification = self.receipt()
        original = self.dependencies["libopenblas.so.0"]
        alternate = self.root / "alternate" / original.name
        alternate.parent.mkdir()
        shutil.copyfile(original, alternate)
        self.dependencies[original.name] = alternate
        changed = self.snapshot("changed")
        proof = Q.compare_runtime_linkage(qualification["runtime_linkage"]["before"], changed)
        self.assertFalse(proof["resolution_unchanged"])
        original_record = next(record for record in proof["before"]["dependencies"].values()
                               if original.name in record["sonames"])
        alternate_record = next(record for record in changed["dependencies"].values()
                                if original.name in record["sonames"])
        self.assertEqual(original_record["path"], alternate_record["path"])
        self.assertEqual(original_record["sha256"], alternate_record["sha256"])
        self.assertNotEqual(original_record["resolved_path_sha256"], alternate_record["resolved_path_sha256"])
        with self.assertRaisesRegex(ValueError, "runtime linkage"):
            Q.verify_runtime_linkage_receipt(proof, qualification["binaries"])

    def test_missing_file_unresolved_omitted_and_duplicate_dependencies_reject(self):
        provider = self.dependencies["libopenblas.so.0"]
        content = provider.read_bytes()
        provider.unlink()
        with self.assertRaisesRegex(ValueError, "missing/unreadable"):
            self.snapshot("missing-file")
        provider.write_bytes(content)
        self.resolver_suffix = "\n\tlibmissing.so => not found"
        with self.assertRaisesRegex(ValueError, "unresolved"):
            self.snapshot("unresolved")
        self.resolver_suffix = f"\n\tlibopenblas.so.0 => {provider} (0x4321)"
        with self.assertRaisesRegex(ValueError, "duplicated"):
            self.snapshot("duplicate")
        self.resolver_suffix = ""
        del self.dependencies["libggml-blas.so.0"]
        with self.assertRaisesRegex(ValueError, "all four ggml"):
            self.snapshot("omitted")

    def test_nonzero_or_unrecognized_resolver_output_rejects_and_preserves_logs(self):
        self.resolver_code = 1
        with self.assertRaisesRegex(ValueError, "resolution failed"):
            self.snapshot("nonzero")
        self.assertTrue(list(self.logs.glob("linkage-nonzero-*.log")))
        self.resolver_code = 0
        self.resolver_suffix = "\nunknown dependency format"
        with self.assertRaisesRegex(ValueError, "unrecognized"):
            self.snapshot("malformed")

    def test_timed_out_resolver_preserves_partial_output_and_rejects(self):
        failure = subprocess.TimeoutExpired(["ldd"], 30, output=b"partial loader output\n", stderr=b"partial error\n")
        with mock.patch.object(Q.platform, "system", return_value="Linux"), \
                mock.patch.object(Q.subprocess, "run", side_effect=failure), \
                self.assertRaisesRegex(ValueError, "timed out"):
            Q.snapshot_runtime_linkage(self.binaries, self.build, self.environment, self.logs, "timeout")
        self.assertEqual(next(self.logs.glob("*.stdout.log")).read_text(), "partial loader output\n")
        self.assertEqual(next(self.logs.glob("*.stderr.log")).read_text(), "partial error\n")

    def test_dynamic_backends_missing_entrypoints_or_wrong_platform_reject(self):
        for option in ("ON", ""):
            (self.build / "CMakeCache.txt").write_text("GGML_BACKEND_DL:BOOL=" + option + "\n")
            with self.assertRaisesRegex(ValueError, "GGML_BACKEND_DL=OFF"):
                self.snapshot("backend-" + (option.lower() or "missing"))
        (self.build / "CMakeCache.txt").write_text("GGML_BACKEND_DL:BOOL=OFF\n")
        with mock.patch.object(Q.platform, "system", return_value="Darwin"), self.assertRaisesRegex(ValueError, "Linux"):
            Q.snapshot_runtime_linkage(self.binaries, self.build, self.environment, self.logs, "platform")
        del self.binaries["crispasr"]
        with self.assertRaisesRegex(ValueError, "all five"):
            self.snapshot("entrypoint")

    def test_changed_loader_environment_rejects_without_mutating_it(self):
        qualification = self.receipt()
        self.environment["LD_LIBRARY_PATH"] = "/synthetic/changed/loader/path"
        changed = self.snapshot("changed")
        proof = Q.compare_runtime_linkage(qualification["runtime_linkage"]["before"], changed)
        self.assertFalse(proof["resolution_unchanged"])
        self.assertNotIn("/synthetic/", json.dumps(proof))
        self.assertEqual(self.environment["LD_LIBRARY_PATH"], "/synthetic/changed/loader/path")
        with self.assertRaisesRegex(ValueError, "runtime linkage"):
            Q.verify_runtime_linkage_receipt(proof, qualification["binaries"])

    def test_incomplete_flags_and_forged_success_flags_reject(self):
        qualification = self.receipt()
        mutations = [
            lambda p: p.pop("before"), lambda p: p.pop("after"),
            lambda p: p.update(complete=False), lambda p: p.pop("complete"),
            lambda p: p.update(entrypoints_unchanged=False), lambda p: p.pop("dependencies_unchanged"),
            lambda p: p.update(resolution_unchanged=False),
            lambda p: p["after"]["entrypoints"]["crispasr"].update(sha256="f" * 64),
            lambda p: next(iter(p["after"]["dependencies"].values())).update(sha256="f" * 64),
            lambda p: p["after"]["resolutions"].pop("validation_library"),
            lambda p: p["after"]["resolver_logs"].pop("crispasr"),
        ]
        for index, mutate in enumerate(mutations):
            proof = copy.deepcopy(qualification["runtime_linkage"])
            mutate(proof)
            with self.subTest(control=index), self.assertRaisesRegex(ValueError, "runtime linkage"):
                Q.verify_runtime_linkage_receipt(proof, qualification["binaries"])

    def test_packaging_rejects_changed_resolver_log_before_fresh_lookup(self):
        qualification = self.receipt()
        log = qualification["runtime_linkage"]["before"]["resolver_logs"]["crispasr"]["stdout"]["file"]
        (self.logs / log).write_text("replaced resolver evidence")
        with mock.patch.object(P, "snapshot_runtime_linkage") as snapshot:
            with self.assertRaisesRegex(ValueError, "resolver log changed"):
                P.verify_packaging_linkage(qualification, self.binaries, self.build, self.logs)
            snapshot.assert_not_called()

    def test_historical_receipt_without_linkage_proof_cannot_be_packaged(self):
        # Only the gate's minimum fields are synthetic; no model bundle is made.
        receipt = {"schema": 1, "complete": True, "acceptance": True, "status": "passed", "errors": []}
        with self.assertRaisesRegex(ValueError, "runtime linkage before/after"):
            P.verify_receipts(receipt, receipt, "scnet", "1" * 40)

    def test_existing_resolver_logs_are_never_replaced(self):
        self.snapshot("once")
        before = {path.name: Q.sha256(path) for path in self.logs.iterdir()}
        with self.assertRaises(FileExistsError):
            self.snapshot("once")
        self.assertEqual(before, {path.name: Q.sha256(path) for path in self.logs.iterdir()})


def actual_build_control(build, output):
    output.mkdir(parents=True, exist_ok=False)
    binaries = Q.resolve_qualification_binaries(build)
    before = Q.snapshot_runtime_linkage(binaries, build, dict(os.environ), output / "logs", "before")
    after = Q.snapshot_runtime_linkage(binaries, build, dict(os.environ), output / "logs", "after")
    qualification = {"binaries": {name: {key: row[key] for key in ("path", "sha256")}
                                  for name, row in before["entrypoints"].items()},
                     "runtime_linkage": Q.compare_runtime_linkage(before, after)}
    Q.verify_runtime_linkage_receipt(qualification["runtime_linkage"], qualification["binaries"])
    packaging = P.verify_packaging_linkage(qualification, binaries, build, output / "logs")
    receipt = {"schema": "scnet-linkage-controls-v1", "complete": True, "control_pass": True,
               "acceptance": False, "qualification": False, "performance_qualification": False,
               "scope": "Actual Linux loader/byte identity control only; no model inference or accepted bundle.",
               "source_sha256": {name: Q.sha256(ROOT / name) for name in
                   ("tools/ci-heavy/scnet_qualify.py", "tools/ci-heavy/scnet_package.py", "tests/test_scnet_linkage.py")},
               **qualification, "packaging_snapshot": packaging}
    Q.write_json(output / "receipt.json", receipt)
    print(json.dumps({"actual_linkage_control_pass": True, "entrypoints": len(binaries),
                      "dependencies": len(before["dependencies"]), "qualification": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    arguments, remaining = parser.parse_known_args()
    if bool(arguments.build_dir) != bool(arguments.output_dir):
        parser.error("--build-dir and --output-dir must be supplied together")
    program = unittest.main(argv=[sys.argv[0], *remaining], exit=False)
    if not program.result.wasSuccessful():
        raise SystemExit(1)
    if arguments.build_dir:
        actual_build_control(arguments.build_dir.resolve(), arguments.output_dir.resolve())
