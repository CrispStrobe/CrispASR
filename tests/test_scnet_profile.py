"""Model-free rejection controls for the optional SCNet wait-policy diagnostic."""

import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
from test_scnet_runtime_identity import identity_fixture


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "ci-heavy"))
SPEC = importlib.util.spec_from_file_location("scnet_profile", ROOT / "tools" / "ci-heavy" / "scnet_profile.py")
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)
V = sys.modules["validate_scnet"]


def fixture():
    """Synthetic receipt for contract tests, never a model qualification result."""
    row = {"complete": True, "waveform_acceptance": True, "acceptance": False, "qualification": False,
           "required_stages_checked": 0, "threads": 2, "torch_intraop_threads": 2, "torch_interop_threads": 1,
           "repetitions": 3, "warmups_per_implementation": 1, "worker_process_id": 101, "errors": [],
           "profile": dict(P.PROFILE), "source_commit": "1" * 40, "upstream_source_revision": "2" * 40,
           "upstream_source_hashes": {"test-source": "3" * 64}, "original_provenance": {"synthetic_test": True},
           "native_backend": "CPU+BLAS/CPU-LSTM-FFT", "native_compute_precision": "F32 test fixture",
           "torch_version": "2.8.0+cpu", "original_dtype": "float32",
           "waveform_policy": {"strict_cosine_gt": .999, "norm_ratio_bounds": [.99, 1.01], "relative_l2_max": .05},
           "runtime": {"thread_environment": dict.fromkeys(P.THREAD_VARIABLES, "2"),
                       "wait_policy_environment": dict.fromkeys(P.WAIT_POLICY_VARIABLES)},
           "native_blas_thread_queries": {"openblas_get_num_threads": 2},
           "execution_controls": {**dict.fromkeys(P.CONTROLS), "CRISPASR_SCNET_BENCH": "0", "CRISPASR_SCNET_GPU": "0"}}
    for key in ("model_sha256", "checkpoint_sha256", "audio_sha256", "library_sha256",
                "input_pcm_interleaved_sha256", "input_pcm_planar_sha256"):
        row[key] = "4" * 64
    row["loaded_dso_identity"] = identity_fixture(process_id=101)

    def call(iteration, name):
        return {"iteration": iteration, "implementation": name, "complete": True,
                "quality_acceptance": True, "input_unchanged": True,
                "seconds": 10.0 if name == "original" else 12.0,
                "pcm_sha256": ("a" if name == "original" else "b") * 64,
                "stems": {stem: {"acceptance": True, "elements": 2 * 44100 * 24,
                                  "cos_global": 1.0, "norm_ratio": 1.0, "relative_l2": 0.0,
                                  "max_abs": 0.0, "rms_error": 0.0} for stem in P.STEMS}}
    row["warmups"] = [call(0, name) for name in ("original", "native")]
    row["calls"] = [call(iteration, name) for iteration in range(3)
                    for name in (("original", "native") if iteration % 2 == 0 else ("native", "original"))]
    row["reference_pcm_sha256"] = "a" * 64
    row.update(P.summarize_pairs(row["calls"], 3))
    passive = copy.deepcopy(row)
    passive["worker_process_id"] = 102
    passive["loaded_dso_identity"] = identity_fixture(process_id=102)
    passive["runtime"]["wait_policy_environment"].update(P.PASSIVE_WAIT_POLICY)
    return row, passive


class SCNetProfileControls(unittest.TestCase):
    def test_packed_recurrence_timers_retain_worker_zero_scope(self):
        def telemetry(mode, scope="worker_zero"):
            return (
                f"scnet_bench_lstm: name=synthetic mode={mode} scope={scope} hidden=16 batch=2 "
                "steps=3 gemm_calls=3 gate_steps=3 gemm_ms=1.0 gates_ms=2.0 reset_ms=0.1\n"
                "scnet_bench_op: stage=synthetic op=scnet_cpu_lstm backend=CPU frames=3 calls=1 ms=4.0\n"
                "scnet_bench_ops: scope=per_node_synchronized frames=3 nodes=1 groups=1 "
                "unmapped_nodes=0 measured_ms=4.0 unmapped_ms=0.0\n"
                "scnet_bench: separate 5.0 ms\n"
            )

        for mode in ("packed_avx512_fma", "packed_avx2_fma"):
            with self.subTest(mode=mode):
                parsed = P.parse_timers(telemetry(mode))
                self.assertEqual(parsed["lstm"][0]["mode"], mode)
                self.assertEqual(parsed["lstm"][0]["scope"], "worker_zero")
                self.assertIn("not whole-node", parsed["interpretation"])
                with self.assertRaisesRegex(ValueError, "timer scope"):
                    P.parse_timers(telemetry(mode, "whole_node"))
        with self.assertRaisesRegex(ValueError, "timer scope"):
            P.parse_timers(telemetry("packed_unknown"))

    def test_loaded_runtime_proof_is_required_for_both_workers(self):
        for label in (0, 1):
            with self.subTest(worker=label):
                rows = list(fixture())
                del rows[label]["loaded_dso_identity"]
                with self.assertRaisesRegex(ValueError, "loaded-DSO"):
                    P.summarize_wait_policy_comparison(*rows)
        primary, passive = fixture()
        passive["loaded_dso_identity"] = identity_fixture(process_id=999)
        with self.assertRaisesRegex(ValueError, "another worker"):
            P.summarize_wait_policy_comparison(primary, passive)

    def test_identity_snapshots_enclose_only_the_existing_measured_pairs(self):
        import numpy as np
        events = []
        output = np.arange(64, dtype=np.float32).reshape(4, 2, 8) / 64

        def run(name):
            events.append(name)
            return output.copy()

        identity = SimpleNamespace(before_timing=lambda: events.append("mapped-before"),
                                   after_timing=lambda: events.append("mapped-after"))
        receipt = {"warmups": [], "calls": []}
        P.paired_calls(lambda: run("original"), lambda: run("native"), receipt, lambda: None,
                       lambda: True, identity)
        self.assertEqual(events, ["original", "native", "mapped-before", "original", "native",
                                  "native", "original", "original", "native", "mapped-after"])
        self.assertEqual(len(receipt["calls"]), 6)
        self.assertTrue(receipt["waveform_acceptance"])

    def test_slower_passive_result_remains_complete_and_unqualified(self):
        primary, passive = fixture()
        for call in passive["calls"]:
            if call["implementation"] == "native":
                call["seconds"] = 14.0
        passive.update(P.summarize_pairs(passive["calls"], 3))
        original = copy.deepcopy((primary, passive))
        result = P.summarize_wait_policy_comparison(primary, passive)
        self.assertTrue(result["complete"])
        self.assertFalse(result["acceptance"])
        self.assertFalse(result["qualification"])
        self.assertFalse(result["passive_results"]["performance_acceptance"])
        self.assertEqual(result["passive_minus_default_seconds"], {"original": 0.0, "native": 2.0})
        self.assertEqual((primary, passive), original)

    def test_faster_passive_result_does_not_replace_primary_or_claim_qualification(self):
        primary, passive = fixture()
        for call in passive["calls"]:
            if call["implementation"] == "native":
                call["seconds"] = 8.0
        passive.update(P.summarize_pairs(passive["calls"], 3))
        result = P.summarize_wait_policy_comparison(primary, passive)
        self.assertTrue(result["passive_results"]["performance_acceptance"])
        self.assertFalse(result["default_results"]["performance_acceptance"])
        self.assertFalse(result["acceptance"])
        self.assertFalse(result["qualification"])

    def test_rejects_incomplete_failed_or_selectively_omitted_calls(self):
        mutations = [
            lambda r: r.update(complete=False),
            lambda r: r.update(waveform_acceptance=False),
            lambda r: r.update(acceptance=True),
            lambda r: r.update(qualification=True),
            lambda r: r.update(errors=["worker failed"]),
            lambda r: r["warmups"].pop(),
            lambda r: r["warmups"][0].update(seconds=float("nan")),
            lambda r: r["calls"].pop(),
            lambda r: r["calls"].reverse(),
            lambda r: r["calls"][0].update(complete=False),
            lambda r: r["calls"][0].update(quality_acceptance=False),
            lambda r: r["calls"][0].update(input_unchanged=False),
            lambda r: r["calls"][0]["stems"].pop("bass"),
            lambda r: r["calls"][0]["stems"]["bass"].update(acceptance=False),
            lambda r: r.update(native_median_seconds=0.001),
        ]
        for index, change in enumerate(mutations):
            with self.subTest(control=index):
                primary, passive = fixture()
                change(passive)
                with self.assertRaises(ValueError):
                    P.summarize_wait_policy_comparison(primary, passive)

    def test_rejects_nonfresh_workers_wrong_threads_or_instrumentation(self):
        mutations = [
            lambda r: r.update(worker_process_id=101),
            lambda r: r.pop("worker_process_id"),
            lambda r: r.update(torch_intraop_threads=1),
            lambda r: r.update(torch_interop_threads=2),
            lambda r: r["native_blas_thread_queries"].update(openblas_get_num_threads=3),
            lambda r: r.update(native_blas_thread_queries={}),
            lambda r: r["runtime"]["thread_environment"].update(OMP_NUM_THREADS="1"),
            lambda r: r["execution_controls"].update(CRISPASR_SCNET_BENCH="1"),
            lambda r: r["execution_controls"].update(CRISPASR_SCNET_CAPTURE_GGUF="capture.gguf"),
            lambda r: r["execution_controls"].update(CRISPASR_SCNET_SCALAR_GATES="1"),
            lambda r: r["execution_controls"].pop("CRISPASR_SCNET_CAPTURE_GGUF"),
        ]
        for index, change in enumerate(mutations):
            with self.subTest(control=index):
                primary, passive = fixture()
                change(passive)
                with self.assertRaises(ValueError):
                    P.summarize_wait_policy_comparison(primary, passive)

    def test_rejects_identity_output_or_wait_policy_changes(self):
        for key in ("source_commit", "model_sha256", "checkpoint_sha256", "audio_sha256", "library_sha256",
                    "input_pcm_interleaved_sha256", "input_pcm_planar_sha256", "original_provenance", "waveform_policy"):
            with self.subTest(identity=key):
                primary, passive = fixture()
                passive[key] = "altered"
                with self.assertRaises(ValueError):
                    P.summarize_wait_policy_comparison(primary, passive)
        for key, value in (("OMP_WAIT_POLICY", None), ("GOMP_SPINCOUNT", "300000"),
                           ("KMP_BLOCKTIME", "0"), ("OPENBLAS_THREAD_TIMEOUT", "1")):
            with self.subTest(environment=key):
                primary, passive = fixture()
                passive["runtime"]["wait_policy_environment"][key] = value
                with self.assertRaises(ValueError):
                    P.summarize_wait_policy_comparison(primary, passive)
        primary, passive = fixture()
        for call in passive["warmups"] + passive["calls"]:
            if call["implementation"] == "native":
                call["pcm_sha256"] = "c" * 64
        with self.assertRaises(ValueError):
            P.summarize_wait_policy_comparison(primary, passive)

    def test_runtime_provenance_records_wait_variables_without_changing_them(self):
        values = {"OMP_WAIT_POLICY": "PASSIVE", "GOMP_SPINCOUNT": "0",
                  "KMP_BLOCKTIME": "17", "OPENBLAS_THREAD_TIMEOUT": "4"}
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "synthetic-library"
            library.write_bytes(b"not a shared library; CDLL is mocked for this metadata control")
            with mock.patch.dict(V.os.environ, values), mock.patch.object(V.ctypes, "CDLL", return_value=SimpleNamespace()):
                result = V.runtime_provenance(library)
                self.assertEqual(result["wait_policy_environment"], values)
                self.assertEqual({key: V.os.environ.get(key) for key in values}, values)


if __name__ == "__main__":
    unittest.main()
