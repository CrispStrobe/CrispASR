"""Model-free tests for tools/regression_select.py (the Regression matrix picker)."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import regression_select as rs  # noqa: E402

NIGHTLY = json.loads((ROOT / "tests/regression/nightly_matrix.json").read_text())["nightly"]


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

    def test_shared_code_selects_core(self):
        got = rs.select(["src/core/beam_decode.h"], NIGHTLY)
        self.assertEqual(sorted(got), sorted(rs.CORE))

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
