"""Unit tests for benchmark_metrics.py — no binary or audio needed."""

import pytest
from benchmark_metrics import (
    Segment,
    CoverageMetrics,
    parse_segments_from_dict,
    compute_coverage,
)


# ---------------------------------------------------------------------------
# Fixtures: canned crispasr JSON transcription arrays
# ---------------------------------------------------------------------------

def _make_json(segments: list[tuple[int, int, str]]) -> dict:
    """Build a minimal crispasr JSON dict from (from_ms, to_ms, text) tuples."""
    return {
        "transcription": [
            {"offsets": {"from": f, "to": t}, "text": txt}
            for f, t, txt in segments
        ]
    }


GOOD_60S = _make_json([
    (160, 4800, "皆さん、こんにちは"),
    (5000, 12000, "このチャンネルでは日本語について"),
    (13200, 26000, "年末年始とお正月です"),
    (27000, 42000, "年の終わりから年の初めにかけて"),
    (43000, 58000, "基本的にはお正月を迎えるためのこと"),
])

# Issue #89 failure: 35 tokens in first 5s of 300s audio
ISSUE89_BAD = _make_json([
    (160, 480, "皆さん、"),
    (480, 640, "こ"),
    (640, 720, "ん"),
    (720, 880, "に"),
    (880, 960, "ち"),
    (960, 1200, "は、"),
    (4400, 4880, "です。"),
])

EMPTY = _make_json([])

SINGLE_SEG = _make_json([(500, 3000, "hello world")])


# ---------------------------------------------------------------------------
# Tests: parse_segments_from_dict
# ---------------------------------------------------------------------------

class TestParsing:
    def test_good_json(self):
        segs = parse_segments_from_dict(GOOD_60S)
        assert len(segs) == 5
        assert segs[0].from_ms == 160
        assert segs[0].text == "皆さん、こんにちは"

    def test_empty(self):
        segs = parse_segments_from_dict(EMPTY)
        assert segs == []

    def test_missing_transcription_key(self):
        segs = parse_segments_from_dict({"other": "data"})
        assert segs == []


# ---------------------------------------------------------------------------
# Tests: compute_coverage
# ---------------------------------------------------------------------------

class TestCoverage:
    def test_good_60s_coverage(self):
        segs = parse_segments_from_dict(GOOD_60S)
        m = compute_coverage(segs, audio_duration_s=60.0)
        assert m.word_count > 0
        assert m.char_count > 0
        assert m.first_ts_s == pytest.approx(0.16, abs=0.01)
        assert m.last_ts_s == pytest.approx(58.0, abs=0.01)
        assert m.coverage_pct > 80.0  # most of 60s covered
        assert m.gap_count == 0  # all gaps < 2s

    def test_issue89_failure_detected(self):
        """The issue #89 signature: tiny coverage on long audio."""
        segs = parse_segments_from_dict(ISSUE89_BAD)
        m = compute_coverage(segs, audio_duration_s=300.0)
        assert m.word_count < 10
        assert m.coverage_pct < 5.0  # ~1.6% — catastrophic loss
        assert m.last_ts_s < 5.0     # nothing past 5s

    def test_empty_segments(self):
        m = compute_coverage([], audio_duration_s=60.0)
        assert m.word_count == 0
        assert m.coverage_pct == 0.0
        assert m.gap_count == 0

    def test_single_segment(self):
        segs = parse_segments_from_dict(SINGLE_SEG)
        m = compute_coverage(segs, audio_duration_s=10.0)
        assert m.word_count == 2
        assert m.first_ts_s == pytest.approx(0.5)
        assert m.last_ts_s == pytest.approx(3.0)
        assert m.gap_count == 0

    def test_gap_detection(self):
        """Gaps > 2s between segments should be flagged."""
        data = _make_json([
            (0, 5000, "first"),
            (5100, 10000, "close"),     # 100ms gap — not flagged
            (15000, 20000, "after gap"), # 5000ms gap — flagged
            (30000, 35000, "another"),   # 10000ms gap — flagged
        ])
        segs = parse_segments_from_dict(data)
        m = compute_coverage(segs, audio_duration_s=60.0)
        assert m.gap_count == 2
        assert m.max_gap_s == pytest.approx(10.0)
        assert m.total_gap_s == pytest.approx(15.0)

    def test_overlapping_segments_coverage(self):
        """Overlapping segments should not double-count coverage."""
        data = _make_json([
            (0, 10000, "first"),
            (5000, 15000, "overlapping"),
        ])
        segs = parse_segments_from_dict(data)
        m = compute_coverage(segs, audio_duration_s=20.0)
        # Merged interval: 0-15000ms = 15s out of 20s = 75%
        assert m.coverage_pct == pytest.approx(75.0, abs=0.1)

    def test_zero_duration_audio(self):
        segs = parse_segments_from_dict(SINGLE_SEG)
        m = compute_coverage(segs, audio_duration_s=0.0)
        assert m.coverage_pct == 0.0  # no division by zero

    def test_custom_gap_threshold(self):
        data = _make_json([
            (0, 5000, "first"),
            (6500, 10000, "second"),  # 1500ms gap
        ])
        segs = parse_segments_from_dict(data)
        # Default threshold 2000ms — no gap
        m1 = compute_coverage(segs, audio_duration_s=10.0)
        assert m1.gap_count == 0
        # Lower threshold 1000ms — gap detected
        m2 = compute_coverage(segs, audio_duration_s=10.0, gap_threshold_ms=1000)
        assert m2.gap_count == 1

    def test_to_dict(self):
        m = CoverageMetrics(word_count=42, coverage_pct=95.0)
        d = m.to_dict()
        assert d["word_count"] == 42
        assert d["coverage_pct"] == 95.0
        assert isinstance(d, dict)


# ---------------------------------------------------------------------------
# Tests: reporter's table metrics
# ---------------------------------------------------------------------------

class TestReporterMetrics:
    """Verify we can reproduce the reporter's table columns."""

    def test_300s_good_run(self):
        """A good 300s run should have high coverage."""
        segs = []
        # Simulate 487 words spread across 300s
        for i in range(0, 300000, 600):
            segs.append(Segment(from_ms=i + 120, to_ms=i + 500, text=f"w{i}"))
        m = compute_coverage(segs, audio_duration_s=300.0)
        assert m.word_count == 500
        assert m.first_ts_s < 1.0
        assert m.last_ts_s > 299.0
        assert m.coverage_pct > 50.0

    def test_300s_issue89_regression(self):
        """The issue #89 failure: 4 words in first 60s of 300s."""
        segs = [
            Segment(58160, 58480, "なので"),
            Segment(58080, 58400, "なので"),
            Segment(59280, 59600, "年"),
            Segment(59600, 59920, "末"),
        ]
        m = compute_coverage(segs, audio_duration_s=300.0)
        assert m.word_count == 4
        assert m.first_ts_s > 58.0
        assert m.coverage_pct < 1.0  # catastrophic
