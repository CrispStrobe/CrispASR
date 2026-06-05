"""benchmark_metrics.py — parse crispasr JSON output and compute coverage metrics.

The crispasr CLI with ``-ojf`` writes a JSON file whose ``transcription``
array contains per-segment entries with millisecond offsets::

    {"transcription": [
        {"offsets": {"from": 1200, "to": 4800}, "text": "..."},
        ...
    ]}

This module extracts those entries and computes the coverage metrics used
by the ASR benchmark framework (issue #89 regression guard):

    word_count, char_count, first_ts_s, last_ts_s,
    coverage_pct, gap_count, max_gap_s, total_gap_s
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class Segment:
    """One transcription segment with millisecond offsets."""
    from_ms: int
    to_ms: int
    text: str


@dataclass
class CoverageMetrics:
    """Coverage metrics computed from a transcription."""
    word_count: int = 0
    char_count: int = 0
    first_ts_s: float = 0.0
    last_ts_s: float = 0.0
    coverage_pct: float = 0.0
    gap_count: int = 0
    max_gap_s: float = 0.0
    total_gap_s: float = 0.0
    transcript_head: str = ""  # first 80 chars
    transcript_tail: str = ""  # last 80 chars

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_segments(json_path: str | Path) -> list[Segment]:
    """Parse crispasr ``-ojf`` JSON into a list of Segments."""
    path = Path(json_path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    segments = []
    for entry in data.get("transcription", []):
        offsets = entry.get("offsets", {})
        from_ms = offsets.get("from", 0)
        to_ms = offsets.get("to", 0)
        text = entry.get("text", "")
        segments.append(Segment(from_ms=from_ms, to_ms=to_ms, text=text))
    return segments


def parse_segments_from_dict(data: dict) -> list[Segment]:
    """Parse from an already-loaded dict (for testing)."""
    segments = []
    for entry in data.get("transcription", []):
        offsets = entry.get("offsets", {})
        segments.append(Segment(
            from_ms=offsets.get("from", 0),
            to_ms=offsets.get("to", 0),
            text=entry.get("text", ""),
        ))
    return segments


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

GAP_THRESHOLD_MS = 2000  # gaps > 2s are flagged


def compute_coverage(
    segments: list[Segment],
    audio_duration_s: float,
    gap_threshold_ms: int = GAP_THRESHOLD_MS,
) -> CoverageMetrics:
    """Compute coverage metrics from parsed segments.

    Args:
        segments: parsed transcription segments (need not be sorted).
        audio_duration_s: total audio duration in seconds.
        gap_threshold_ms: minimum gap (ms) between segments to flag.

    Returns:
        CoverageMetrics dataclass.
    """
    m = CoverageMetrics()
    if not segments:
        return m

    # Sort by start time
    segs = sorted(segments, key=lambda s: s.from_ms)

    # Word / char counts
    full_text = " ".join(s.text.strip() for s in segs if s.text.strip())
    m.char_count = len(full_text.replace(" ", ""))
    m.word_count = len(full_text.split()) if full_text else 0
    m.transcript_head = full_text[:80]
    m.transcript_tail = full_text[-80:] if len(full_text) > 80 else full_text

    # Timestamp range
    m.first_ts_s = segs[0].from_ms / 1000.0
    m.last_ts_s = segs[-1].to_ms / 1000.0

    # Gap detection: walk sorted segments, find gaps between consecutive
    # segment.to and next segment.from that exceed the threshold.
    gaps = []
    for i in range(1, len(segs)):
        gap_ms = segs[i].from_ms - segs[i - 1].to_ms
        if gap_ms > gap_threshold_ms:
            gaps.append(gap_ms)

    m.gap_count = len(gaps)
    m.total_gap_s = sum(gaps) / 1000.0
    m.max_gap_s = max(gaps) / 1000.0 if gaps else 0.0

    # Coverage: fraction of audio duration that has transcription
    if audio_duration_s > 0:
        # Compute covered time by merging overlapping intervals
        covered_ms = _merge_covered(segs)
        m.coverage_pct = round(covered_ms / (audio_duration_s * 1000.0) * 100.0, 1)

    return m


def _merge_covered(segs: list[Segment]) -> float:
    """Merge overlapping segment intervals and return total covered ms."""
    if not segs:
        return 0.0
    intervals = [(s.from_ms, s.to_ms) for s in segs]
    intervals.sort()
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return sum(end - start for start, end in merged)


# ---------------------------------------------------------------------------
# Convenience: parse + compute in one call
# ---------------------------------------------------------------------------

def metrics_from_json(
    json_path: str | Path,
    audio_duration_s: float,
    gap_threshold_ms: int = GAP_THRESHOLD_MS,
) -> CoverageMetrics:
    """Parse a crispasr JSON output file and return coverage metrics."""
    segs = parse_segments(json_path)
    return compute_coverage(segs, audio_duration_s, gap_threshold_ms)
