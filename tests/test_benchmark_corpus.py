"""Unit tests for benchmark_corpus.py — no audio or models needed."""

import json
import os
import tempfile
from pathlib import Path

import pytest

# Allow import from tests/ dir
import sys
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_corpus import FLEURS_LANGS, DURATIONS, EXISTING_AUDIO


class TestCorpusConfig:
    """Validate the corpus configuration constants."""

    def test_fleurs_langs_cover_required(self):
        """en, de, ja, zh must all be present."""
        for lang in ["en", "de", "ja", "zh"]:
            assert lang in FLEURS_LANGS, f"missing FLEURS lang: {lang}"

    def test_fleurs_codes_are_valid(self):
        """FLEURS codes should match Google's naming convention."""
        for lang, code in FLEURS_LANGS.items():
            assert "_" in code or len(code) == 2, f"unexpected FLEURS code format: {code}"

    def test_durations_ascending(self):
        assert DURATIONS == sorted(DURATIONS)
        assert DURATIONS[0] >= 5  # at least 5s
        assert DURATIONS[-1] <= 1200  # at most 20 min

    def test_existing_audio_keys(self):
        """Existing audio should reference known languages."""
        for lang in EXISTING_AUDIO:
            assert lang in FLEURS_LANGS or lang in ("en", "de", "ja", "zh"), \
                f"unknown lang in EXISTING_AUDIO: {lang}"


class TestCorpusManifest:
    """Test corpus.json manifest format if it exists."""

    @pytest.fixture
    def corpus_path(self):
        p = Path("/mnt/storage/test-audio/corpus.json")
        if not p.exists():
            pytest.skip("corpus.json not found (run benchmark_corpus.py first)")
        return p

    def test_manifest_is_valid_json(self, corpus_path):
        with open(corpus_path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) > 0

    def test_manifest_entries_have_required_fields(self, corpus_path):
        with open(corpus_path) as f:
            data = json.load(f)
        for entry in data:
            assert "path" in entry
            assert "language" in entry
            assert "duration_s" in entry
            assert entry["duration_s"] > 0

    def test_manifest_audio_files_exist(self, corpus_path):
        corpus_dir = corpus_path.parent
        with open(corpus_path) as f:
            data = json.load(f)
        missing = []
        for entry in data:
            p = corpus_dir / entry["path"]
            if not p.exists():
                missing.append(entry["path"])
        assert len(missing) == 0, f"missing audio files: {missing}"

    def test_manifest_covers_multiple_languages(self, corpus_path):
        with open(corpus_path) as f:
            data = json.load(f)
        langs = {e["language"] for e in data}
        assert len(langs) >= 2, f"only {langs} languages in corpus"

    def test_manifest_covers_multiple_durations(self, corpus_path):
        with open(corpus_path) as f:
            data = json.load(f)
        durations = sorted(set(int(e["duration_s"]) for e in data))
        assert len(durations) >= 2, f"only {durations}s durations in corpus"
