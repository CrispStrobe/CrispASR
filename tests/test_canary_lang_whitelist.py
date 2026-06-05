"""Guard test: canary backend language whitelist must match the v2 model card.

Issue #140 — the whitelist was stuck at 4 v1-era languages while v2 supports 25.
This test extracts the kSupportedLangs array from the source and asserts it matches
the canonical set from the NVIDIA canary-1b-v2 model card.
"""

import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Canonical 25-language set from the nvidia/canary-1b-v2 HuggingFace model card.
CANARY_V2_LANGS = frozenset({
    "en", "bg", "hr", "cs", "da", "nl", "et", "fi", "fr",
    "de", "el", "hu", "it", "lv", "lt", "mt", "pl", "pt",
    "ro", "sk", "sl", "es", "sv", "ru", "uk",
})


class TestCanaryLangWhitelist(unittest.TestCase):
    def test_whitelist_matches_v2_model_card(self) -> None:
        src = (ROOT / "examples/cli/crispasr_backend_canary.cpp").read_text(encoding="utf-8")
        m = re.search(
            r'static\s+const\s+char\*\s+kSupportedLangs\[\]\s*=\s*\{([^}]+)\}',
            src,
        )
        self.assertIsNotNone(m, "kSupportedLangs[] not found in canary backend")
        langs = set(re.findall(r'"([a-z]{2})"', m.group(1)))
        self.assertEqual(
            langs,
            CANARY_V2_LANGS,
            f"Whitelist drift — missing: {CANARY_V2_LANGS - langs}, "
            f"extra: {langs - CANARY_V2_LANGS}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
