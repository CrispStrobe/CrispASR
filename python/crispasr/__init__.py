"""CrispASR — lightweight speech recognition via ggml."""

from ._binding import (
    AlignedWord,
    CrispASR,
    DiarizeMethod,
    DiarizeSegment,
    LidMethod,
    LidResult,
    Segment,
    Session,
    SessionSegment,
    SessionWord,
    align_words,
    detect_language_pcm,
    diarize_segments,
)

__all__ = [
    "AlignedWord",
    "CrispASR",
    "DiarizeMethod",
    "DiarizeSegment",
    "LidMethod",
    "LidResult",
    "Segment",
    "Session",
    "SessionSegment",
    "SessionWord",
    "align_words",
    "detect_language_pcm",
    "diarize_segments",
]
__version__ = "0.4.7"
