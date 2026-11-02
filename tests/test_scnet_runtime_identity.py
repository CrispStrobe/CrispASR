"""Loaded-runtime rejection controls; synthetic contracts are not model qualification.

Optional --library/--output runs a genuine load-only check with Torch imported
first. It does not initialize an SCNet model or run inference/timings.
"""

import argparse
import copy
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import scnet_runtime_identity as R


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def identity_fixture(process_id=101, native_sha256="4" * 64):
    """An in-memory test contract; no success receipt is written from this fixture."""
    files, names = {}, {}
    for name in ("libscnet-validation.so", "libopenblas.so.0", "libtorch_cpu.so", "libgomp.so.1",
                 *("lib" + name + ".so.0" for name in R.GGML_NAMES), "liboptional.so"):
        identity = digest("synthetic resolved path " + name)
        names[name] = identity
        files[identity] = {"file": "{runtime}/" + name, "name": name, "resolved_path_sha256": identity,
                           "sha256": native_sha256 if name == "libscnet-validation.so" else digest(name),
                           "bytes": 100, "mappings": 1}
    snapshot = {"schema": 1, "collector": R.COLLECTOR, "scope": R.SCOPE, "process_id": process_id,
        "require_torch": True, "cmake_cache_sha256": "c" * 64, "files": files,
        "virtual_objects_without_file_bytes": ["linux-vdso.so.1"],
        "roles": {"native_library": names["libscnet-validation.so"], "configured_openblas": names["libopenblas.so.0"],
                  "torch_cpu": names["libtorch_cpu.so"], "openmp_candidates": [names["libgomp.so.1"]],
                  "ggml": {name: names["lib" + name + ".so.0"] for name in R.GGML_NAMES}}}
    evidence = {"schema": 1, "collector": R.COLLECTOR, "complete": True,
                "scope": "after_warmups_through_all_timing_pairs",
                **{key: copy.deepcopy(snapshot) for key in ("initialization", "timed_before", "timed_after")}}
    refresh_transitions(evidence)
    return evidence


def refresh_transitions(evidence):
    evidence["initialization_transition"] = R.compare_snapshots(evidence["initialization"], evidence["timed_before"])
    evidence["timed_transition"] = R.compare_snapshots(evidence["timed_before"], evidence["timed_after"])


class RuntimeIdentityControls(unittest.TestCase):
    def test_stable_proof_and_first_use_addition_are_accepted(self):
        evidence = identity_fixture()
        R.verify_timed_identity(evidence, native_sha256="4" * 64)
        optional = next(key for key, value in evidence["initialization"]["files"].items()
                        if value["name"] == "liboptional.so")
        del evidence["initialization"]["files"][optional]
        refresh_transitions(evidence)
        self.assertEqual(evidence["initialization_transition"]["added_objects"], [optional])
        R.verify_timed_identity(evidence)

    def test_initialization_cannot_change_common_file_bytes(self):
        evidence = identity_fixture()
        provider = evidence["initialization"]["roles"]["configured_openblas"]
        evidence["initialization"]["files"][provider]["sha256"] = "f" * 64
        refresh_transitions(evidence)
        with self.assertRaisesRegex(ValueError, "identity changed"):
            R.verify_timed_identity(evidence)

    def test_timed_file_bytes_additions_removals_and_map_counts_reject(self):
        def mutate(evidence, operation):
            files = evidence["timed_after"]["files"]
            optional = next(key for key, value in files.items() if value["name"] == "liboptional.so")
            if operation == "remove":
                del files[optional]
            elif operation == "add":
                new = copy.deepcopy(files[optional])
                new.update(resolved_path_sha256=digest("new path"), name="libnew.so", file="{runtime}/libnew.so")
                files[new["resolved_path_sha256"]] = new
            else:
                files[optional][operation] = {"sha256": "f" * 64, "bytes": 101, "mappings": 2}[operation]
        for operation in ("sha256", "bytes", "mappings", "add", "remove"):
            with self.subTest(operation=operation):
                evidence = identity_fixture()
                mutate(evidence, operation)
                refresh_transitions(evidence)
                with self.assertRaises(ValueError):
                    R.verify_timed_identity(evidence)

    def test_missing_required_roles_or_objects_reject(self):
        for role in ("native_library", "configured_openblas", "torch_cpu", "openmp_candidates", "ggml"):
            with self.subTest(role=role):
                snapshot = identity_fixture()["timed_before"]
                del snapshot["roles"][role]
                with self.assertRaises(ValueError):
                    R.validate_snapshot(snapshot)
        for name in ("libscnet-validation.so", "libopenblas.so.0", "libtorch_cpu.so", "libgomp.so.1",
                     "libggml-cpu.so.0"):
            with self.subTest(object=name):
                snapshot = identity_fixture()["timed_before"]
                identity = next(key for key, value in snapshot["files"].items() if value["name"] == name)
                del snapshot["files"][identity]
                with self.assertRaises(ValueError):
                    R.validate_snapshot(snapshot)

    def test_incomplete_forged_success_wrong_scope_and_process_reject(self):
        changes = [lambda e: e.update(complete=False), lambda e: e.pop("timed_after"),
                   lambda e: e.update(scope="all possible symbol bindings"),
                   lambda e: e["timed_after"].update(process_id=102),
                   lambda e: e["timed_after"].update(cmake_cache_sha256="d" * 64),
                   lambda e: e["timed_after"].update(scope="relocation proof"),
                   lambda e: e["timed_transition"].update(mapping_set_unchanged=False)]
        for index, change in enumerate(changes):
            with self.subTest(control=index):
                evidence = identity_fixture()
                change(evidence)
                with self.assertRaises(ValueError):
                    R.verify_timed_identity(evidence)

    def test_malformed_role_ids_raise_value_error(self):
        for role, value in (("native_library", []), ("configured_openblas", {}),
                            ("torch_cpu", []), ("openmp_candidates", [[]])):
            with self.subTest(role=role):
                snapshot = identity_fixture()["timed_before"]
                snapshot["roles"][role] = value
                with self.assertRaises(ValueError):
                    R.validate_snapshot(snapshot)

    def test_native_artifacts_bind_but_standalone_gomp_is_not_a_loaded_gomp_requirement(self):
        evidence = identity_fixture()
        before = evidence["timed_before"]
        native_files = copy.deepcopy(before["files"])
        for identity in [before["roles"]["torch_cpu"], *before["roles"]["openmp_candidates"]]:
            del native_files[identity]
        # Standalone ldd may name system GOMP, while this worker mapped Torch's.
        native_files[digest("system gomp")] = {"sha256": digest("system bytes"), "bytes": 77,
                                               "resolved_path_sha256": digest("system gomp")}
        R.verify_timed_identity(evidence, "4" * 64, native_files)
        for role in ("native_library", "configured_openblas"):
            with self.subTest(role=role):
                wrong = copy.deepcopy(native_files)
                wrong[before["roles"][role]]["sha256"] = "f" * 64
                with self.assertRaises(ValueError):
                    R.verify_timed_identity(evidence, "4" * 64, wrong)
        wrong = copy.deepcopy(native_files)
        del wrong[before["roles"]["ggml"]["ggml-blas"]]
        with self.assertRaises(ValueError):
            R.verify_timed_identity(evidence, "4" * 64, wrong)
        with self.assertRaises(ValueError):
            R.verify_timed_identity(evidence, "f" * 64, native_files)

    def test_real_file_hashes_roles_and_private_path_scrubbing(self):
        with tempfile.TemporaryDirectory(prefix="scnet-mapped-control-") as directory:
            root = Path(directory)
            build, torch_dir = root / "build", root / "torch"
            (build / "src").mkdir(parents=True)
            (torch_dir / "lib").mkdir(parents=True)
            library = build / "src/libscnet-validation.so"
            provider = root / "libopenblas.so.0"
            paths = [library, provider, torch_dir / "lib/libtorch_cpu.so", torch_dir / "lib/libgomp.so.1",
                     *(build / "src" / ("lib" + name + ".so.0") for name in R.GGML_NAMES)]
            for path in paths:
                path.write_bytes(("Synthetic file, not executable: " + path.name).encode())
            (build / "CMakeCache.txt").write_text("GGML_BACKEND_DL:BOOL=OFF\nBLAS_openblas_LIBRARY:FILEPATH=" +
                                                  str(provider) + "\n")
            with mock.patch.object(R, "mapped_paths", return_value=(paths, ["linux-vdso.so.1"])), \
                    mock.patch.dict(sys.modules, {"torch": SimpleNamespace(__file__=str(torch_dir / "__init__.py"))}):
                snapshot = R.capture_loaded_dsos(library, require_torch=True)
            self.assertNotIn(str(root), json.dumps(snapshot))
            self.assertEqual(snapshot["files"][snapshot["roles"]["native_library"]]["sha256"], R.sha256(library))
            gomp = snapshot["files"][snapshot["roles"]["openmp_candidates"][0]]
            self.assertEqual(gomp["file"], "{torch}/lib/libgomp.so.1")
            for missing in (provider, torch_dir / "lib/libtorch_cpu.so"):
                with mock.patch.object(R, "mapped_paths", return_value=([p for p in paths if p != missing], [])), \
                        self.assertRaises(ValueError):
                    R.capture_loaded_dsos(library, require_torch=True)

    def test_loader_callback_exceptions_fail_closed(self):
        class Iterator:
            def __call__(self, callback, unused):
                prefix = self.argtypes[0]._argtypes_[0]._type_
                info = prefix()
                info.name = b"relative/libambiguous.so"
                return callback(ctypes.pointer(info), ctypes.sizeof(prefix), None)
        with mock.patch.object(R.ctypes, "CDLL", return_value=SimpleNamespace(dl_iterate_phdr=Iterator())), \
                self.assertRaisesRegex(ValueError, "enumeration failed"):
            R.mapped_paths()

    def test_tracker_keeps_failed_timings_unaccepted(self):
        good = identity_fixture()
        changed = copy.deepcopy(good["timed_after"])
        provider = changed["roles"]["configured_openblas"]
        changed["files"][provider]["sha256"] = "f" * 64
        with mock.patch.object(R, "capture_loaded_dsos", side_effect=[good["initialization"], good["timed_before"], changed]):
            tracker = R.LoadedDSOTracker("synthetic")
            tracker.before_timing()
            with self.assertRaises(ValueError):
                tracker.after_timing()
        self.assertFalse(tracker.evidence["complete"])
        self.assertIn("timed_after", tracker.evidence)


class QualificationIdentityControls(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(ROOT / "tools/ci-heavy"))
        import scnet_qualify
        import scnet_package
        self.Q, self.P = scnet_qualify, scnet_package
        evidence = identity_fixture()
        before = evidence["timed_before"]
        native = before["files"][before["roles"]["native_library"]]
        dependencies = {identity: record for identity, record in before["files"].items()
                        if identity != before["roles"]["native_library"]}
        # Standalone linkage-schema controls live in test_scnet_linkage.py;
        # these isolate the new actual-worker cross-binding boundary.
        self.snapshot = {"entrypoints": {"validation_library": native}, "dependencies": dependencies}
        self.validation = {"library_sha256": native["sha256"], "benchmark": {
            "runtime": {"cmake_cache_sha256": before["cmake_cache_sha256"],
                        "loaded_dso_initialization": copy.deepcopy(evidence["initialization"])},
            "models": {precision: {"loaded_dso_identity": copy.deepcopy(evidence)}
                       for precision in ("f16", "q8_0", "q4_k")}}}
        self.valid_snapshot = mock.patch.object(self.Q, "validate_linkage_snapshot", return_value=self.snapshot)
        self.valid_snapshot.start()
        self.addCleanup(self.valid_snapshot.stop)

    def test_all_three_precision_phases_are_cross_bound(self):
        result = self.Q.verify_validation_loaded_identities(self.validation, self.snapshot)
        self.assertTrue(result["complete"])
        self.assertEqual(set(result["models"]), {"f16", "q8_0", "q4_k"})
        for precision in result["models"]:
            with self.subTest(precision=precision):
                missing = copy.deepcopy(self.validation)
                del missing["benchmark"]["models"][precision]["loaded_dso_identity"]
                with self.assertRaises(ValueError):
                    self.Q.verify_validation_loaded_identities(missing, self.snapshot)

    def test_different_worker_or_cache_cannot_be_combined_into_one_validation(self):
        for key in ("process", "cache", "initialization"):
            with self.subTest(change=key):
                changed = copy.deepcopy(self.validation)
                if key == "process":
                    changed["benchmark"]["models"]["q8_0"]["loaded_dso_identity"] = identity_fixture(process_id=102)
                elif key == "cache":
                    changed["benchmark"]["runtime"]["cmake_cache_sha256"] = "e" * 64
                else:
                    changed["benchmark"]["runtime"].pop("loaded_dso_initialization")
                with self.assertRaises(ValueError):
                    self.Q.verify_validation_loaded_identities(changed, self.snapshot)

    def test_packaging_rejects_success_flags_without_actual_worker_identity(self):
        accepted_flags = {"schema": 1, "complete": True, "acceptance": True, "status": "passed", "errors": []}
        validation = {**copy.deepcopy(self.validation), **accepted_flags}
        qualification = {**accepted_flags, "runtime_linkage": {}, "binaries": {}}
        with mock.patch.object(self.P, "verify_runtime_linkage_receipt", return_value=self.snapshot):
            with self.assertRaisesRegex(ValueError, "actual timing-worker"):
                self.P.verify_receipts(qualification, validation, "scnet", "0" * 40)
            del validation["benchmark"]["models"]["q4_k"]["loaded_dso_identity"]
            with self.assertRaisesRegex(ValueError, "loaded-DSO"):
                self.P.verify_receipts(qualification, validation, "scnet", "0" * 40)


def real_load_only(library, output):
    if output.exists():
        raise ValueError("refusing to overwrite the load-only control receipt")
    import torch

    native = ctypes.CDLL(str(library.resolve()))
    tracker = R.LoadedDSOTracker(library)
    tracker.before_timing()
    tracker.after_timing()
    # The helper's phase contract is exercised without pretending model pairs ran.
    receipt = {"schema": 1, "scope": "Genuine load-only identity control; no model initialization, inference or timings",
        "complete": True, "acceptance": False, "qualification": False, "model_inference_performed": False,
        "torch_version": torch.__version__, "library_sha256": R.sha256(library),
        "identity_control": tracker.evidence}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return native  # Keep the loaded handle alive through the final snapshot.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--output", type=Path)
    args, rest = parser.parse_known_args()
    if bool(args.library) != bool(args.output):
        parser.error("--library and --output must be supplied together")
    result = unittest.main(argv=[sys.argv[0], *rest], exit=False)
    if not result.result.wasSuccessful():
        raise SystemExit(1)
    if args.library:
        real_load_only(args.library, args.output)
