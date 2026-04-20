"""CrispASR — lightweight speech recognition via ggml."""

from ._binding import (
    AlignedWord,
    CrispASR,
    DiarizeMethod,
    DiarizeSegment,
    LidMethod,
    LidResult,
    RegistryEntry,
    Segment,
    Session,
    SessionSegment,
    SessionWord,
    align_words,
    cache_dir,
    cache_ensure_file,
    detect_language_pcm,
    diarize_segments,
    registry_lookup,
    registry_lookup_by_filename,
)

__all__ = [
    "AlignedWord",
    "CrispASR",
    "DiarizeMethod",
    "DiarizeSegment",
    "LidMethod",
    "LidResult",
    "RegistryEntry",
    "Segment",
    "Session",
    "SessionSegment",
    "SessionWord",
    "align_words",
    "cache_dir",
    "cache_ensure_file",
    "detect_language_pcm",
    "diarize_segments",
    "registry_lookup",
    "registry_lookup_by_filename",
]
__version__ = "0.4.9"
