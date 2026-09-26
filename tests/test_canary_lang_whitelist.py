"""Source-contract guards for legacy and metadata-driven Canary variants.

Issue #140 — the whitelist was stuck at 4 v1-era languages while v2 supports 25.
This test extracts the kSupportedLangs array from the source and asserts it matches
the canonical set from the NVIDIA canary-1b-v2 model card. Canary 180M Flash uses
the newer transcribe.cpp schema, whose language and translation capabilities must
come from GGUF metadata and whose canary2 prompt includes the no-ITN control.
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


def function_body(source: str, signature: str) -> str:
    start = source.find(signature)
    if start < 0:
        raise AssertionError(f"{signature} not found")
    brace = source.find("{", start)
    if brace < 0:
        raise AssertionError(f"{signature} has no body")
    depth = 0
    for pos in range(brace, len(source)):
        if source[pos] == "{":
            depth += 1
        elif source[pos] == "}":
            depth -= 1
            if depth == 0:
                return source[brace + 1:pos]
    raise AssertionError(f"{signature} has an unterminated body")


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

    def test_new_schema_capabilities_come_from_gguf_metadata(self) -> None:
        src = (ROOT / "src/canary.cpp").read_text(encoding="utf-8")
        loader = function_body(src, "static bool canary_load_new_hparams(")
        self.assertIn(
            'canary_read_required_string_array(gctx, "general.languages"',
            loader,
        )
        self.assertIn(
            'canary_read_required_string_array(gctx, "stt.translation.pairs"',
            loader,
        )

    def test_canary2_prompt_includes_noitn(self) -> None:
        src = (ROOT / "src/canary.cpp").read_text(encoding="utf-8")
        prompt = function_body(src, "static std::vector<int> canary_build_prompt(")
        pnc = prompt.index("prompt.push_back(punctuation ? hp.pnc_id : hp.nopnc_id);")
        noitn = prompt.index("prompt.push_back(hp.noitn_id);")
        timestamp = prompt.index("prompt.push_back(hp.notimestamp_id);")
        self.assertLess(pnc, noitn)
        self.assertLess(noitn, timestamp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
