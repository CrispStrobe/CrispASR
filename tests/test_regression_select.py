"""Model-free tests for tools/regression_select.py (the Regression matrix picker)."""
import json
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import regression_select as rs  # noqa: E402

NIGHTLY = json.loads((ROOT / "tests/regression/nightly_matrix.json").read_text())["nightly"]


class TestGranite5Select(unittest.TestCase):
    def test_relevant_paths_select_acceptance(self):
        paths = (
            "src/core/attention.h", "src/granite_speech.cpp", "src/granite_speech.h",
            "src/crispasr_model_registry.cpp", "src/crispasr_c_api.cpp",
            "examples/cli/cli.cpp", "examples/cli/crispasr_backend_granite.cpp",
            "examples/crispasr-quantize/main.cpp", "models/granite_rope.py",
            "models/convert-granite-speech-to-gguf.py", "tools/ci-heavy/granite5_execution.py",
            "tests/regression/granite5.json", "tests/test_granite_rope_converter.py",
            "tests/test-registry.cpp", "tests/test-registry-c-abi.cpp",
            ".github/workflows/regression.yml", "CMakeLists.txt", "src/CMakeLists.txt",
            "ggml", "ggml/src/ggml.c", "cmake/FindBLAS.cmake", "tests/CMakeLists.txt",
            "examples/cli/CMakeLists.txt", "tools/regression_select.py", "python/crispasr/_binding.py",
        )
        for path in paths:
            with self.subTest(path=path):
                self.assertTrue(rs.select_granite5("push", [path]))

    def test_unrelated_and_empty_changes_skip(self):
        for changed in ([], ["README.md", "docs/granite.md"],
                        ["src/canary.cpp", "tests/regression/manifest.json"],
                        ["models/granite_rope.py.bak", "examples/cli/cli.cpp.bak"]):
            with self.subTest(changed=changed):
                self.assertFalse(rs.select_granite5("pull_request", changed))

    def test_full_runs_override_changed_list(self):
        for event in ("schedule", "workflow_dispatch", "dispatch"):
            self.assertTrue(rs.select_granite5(event, ["README.md"]))

    def test_unknown_base_and_diff_failures_run_acceptance(self):
        import subprocess
        for base in ("", "0" * 40):
            with patch.object(rs.subprocess, "run") as git:
                self.assertTrue(rs.select_granite5("push", base=base))
                git.assert_not_called()
        for outcome in (subprocess.CompletedProcess([], 128, stdout=""), OSError("git unavailable")):
            with self.subTest(outcome=outcome):
                kwargs = {"side_effect": outcome} if isinstance(outcome, OSError) else {"return_value": outcome}
                with patch.object(rs.subprocess, "run", **kwargs):
                    self.assertTrue(rs.select_granite5("push", base="base"))

    def test_git_diff_is_used_for_selection(self):
        import subprocess
        for path, selected in (("src/granite_speech.cpp", True), ("README.md", False), ("", False)):
            with patch.object(rs.subprocess, "run", return_value=subprocess.CompletedProcess(
                    [], 0, stdout=path + "\n")) as git:
                self.assertEqual(rs.select_granite5("push", base="old", head="new"), selected)
                self.assertEqual(git.call_args.args[0], ["git", "diff", "--name-only", "old", "new"])

    def test_cli_boolean_and_legacy_matrix_outputs(self):
        cases = [(["--granite5", "--event", "push", "--changed", "README.md"], "false\n"),
                 (["--granite5", "--event", "schedule"], "true\n"),
                 (["--event", "push", "--changed", "README.md"], "[]\n"),
                 (["--event", "schedule"], json.dumps(NIGHTLY) + "\n")]
        for arguments, expected in cases:
            with self.subTest(arguments=arguments), patch.object(sys, "argv", ["regression_select.py", *arguments]):
                output = StringIO()
                with redirect_stdout(output):
                    rs.main()
                self.assertEqual(output.getvalue(), expected)


class TestRegressionSelect(unittest.TestCase):
    def test_nightly_names_exist_in_manifest(self):
        # A renamed/removed manifest entry must not silently drop out of the nightly.
        ids = rs.manifest_ids()
        self.assertEqual([n for n in NIGHTLY if n not in ids], [])
        self.assertEqual(len(NIGHTLY), len(set(NIGHTLY)))

    def test_core_is_subset_of_nightly(self):
        self.assertEqual([n for n in rs.CORE if n not in NIGHTLY], [])

    def test_docs_only_selects_nothing(self):
        self.assertEqual(rs.select(["README.md", "docs/foo.md"], NIGHTLY), [])

    def test_own_source_selects_backend(self):
        self.assertEqual(rs.select(["src/canary.cpp"], NIGHTLY), ["canary-1b-v2"])

    def test_alias_fanout(self):
        got = rs.select(["src/wav2vec2.cpp"], NIGHTLY)
        for n in ("wav2vec2-xlsr-en", "hubert-large", "data2vec-base"):
            self.assertIn(n, got)

    def test_phonon_importer_and_reference_select_the_model(self):
        for source in ("models/phonon2_container.py", "tools/reference_backends/phonon2/fermion_container.py",
                       "tools/reference_backends/parakeet_hf.py"):
            with self.subTest(source=source):
                self.assertEqual(rs.select([source], NIGHTLY), ["phonon2"])

    def test_shared_code_selects_core(self):
        got = rs.select(["src/core/beam_decode.h"], NIGHTLY)
        self.assertEqual(sorted(got), sorted(rs.CORE))

    def test_manifest_transcript_and_hash_changes_select_the_model(self):
        import copy
        before = {"backends": [{"name": "data2vec-base", "expected_transcript": "OLD",
                                "transcript_reference": {"sha256": "old"}}]}
        for field in ("expected_transcript", "transcript_reference"):
            after = copy.deepcopy(before)
            after["backends"][0][field] = "changed"
            changed = rs.changed_entries(before, after)
            selected = rs.select(["tests/regression/manifest.json"], NIGHTLY, changed)
            self.assertEqual(set(selected), set(rs.CORE) | {"data2vec-base"})

    def test_manifest_addition_removal_and_tts_changes(self):
        before = {"backends": [{"name": "data2vec-base", "model": "old"}],
                  "tts_backends": [{"name": "kokoro-82m-en", "reference": "old"}]}
        after = {"backends": [{"name": "hubert-large", "model": "new"}],
                 "tts_backends": [{"name": "kokoro-82m-en", "reference": "new"}]}
        changed = rs.changed_entries(before, after)
        self.assertEqual(changed, {"data2vec-base", "hubert-large", "kokoro-82m-en"})
        selected = rs.select(["tests/regression/manifest.json"], NIGHTLY, changed)
        self.assertTrue(changed.issubset(selected))

    def test_manifest_reordering_does_not_change_entries(self):
        entries = [{"name": "data2vec-base"}, {"name": "hubert-large"}]
        self.assertEqual(rs.changed_entries({"backends": entries},
                                             {"backends": list(reversed(entries))}), set())
        self.assertEqual(set(rs.select(["tests/regression/manifest.json"], NIGHTLY, set())),
                         set(rs.CORE))

    def test_manifest_comparison_failure_runs_all_nightly(self):
        from unittest.mock import patch
        import subprocess
        with patch.object(rs.subprocess, "run", side_effect=subprocess.CalledProcessError(128, "git")):
            self.assertIsNone(rs.manifest_changes("unavailable", "HEAD"))
        self.assertEqual(rs.select(["tests/regression/manifest.json"], NIGHTLY), NIGHTLY)

    def test_invalid_manifest_snapshot_falls_back(self):
        from unittest.mock import patch
        import subprocess
        for payload in ("not json", "[]", '{"backends": [null]}'):
            with self.subTest(payload=payload):
                result = subprocess.CompletedProcess([], 0, stdout=payload)
                with patch.object(rs.subprocess, "run", return_value=result):
                    self.assertIsNone(rs.manifest_changes("base", "head"))

    def test_nightly_list_change_runs_all_nightly(self):
        self.assertEqual(rs.select(["tests/regression/nightly_matrix.json"], NIGHTLY), NIGHTLY)

    def test_every_nightly_backend_is_reachable_from_some_source(self):
        # Each nightly entry must be selectable by at least one of its own stems,
        # otherwise a change to its sources would never run it before the nightly.
        ids = rs.manifest_ids()
        unreachable = []
        for n in NIGHTLY:
            stems = rs.stems_for(ids.get(n, n))
            if not any(n in rs.select([f"src/{s}.cpp"], NIGHTLY) for s in stems):
                unreachable.append(n)
        self.assertEqual(unreachable, [])


if __name__ == "__main__":
    unittest.main()
