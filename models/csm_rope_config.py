"""Validate and write CSM RoPE config metadata without deriving frequencies."""

from dataclasses import dataclass
import math
import struct


@dataclass(frozen=True)
class RopeScaling:
    kind: str
    factor: float
    low_freq_factor: float
    high_freq_factor: float
    orig_max_pos: int


def _u32_positive(value, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 0xFFFFFFFF:
        raise ValueError(f"{field} must be a positive uint32")
    return value


def _f32_positive(value, field: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field} does not fit float32") from exc
    if not math.isfinite(number) or number <= 0 or number < minimum:
        raise ValueError(f"{field} must be finite, positive, and >= {minimum:g}")
    try:
        stored = struct.unpack("<f", struct.pack("<f", number))[0]
    except OverflowError as exc:
        raise ValueError(f"{field} does not fit float32") from exc
    if not math.isfinite(stored) or stored <= 0:
        raise ValueError(f"{field} does not remain positive float32")
    return stored


def parse_rope_scaling(config: dict, *, default_max_pos: int, label: str) -> RopeScaling:
    """Transport HF's raw Llama-3 scaling fields, or an explicit unscaled state."""
    max_pos = _u32_positive(default_max_pos, f"{label}.max_position_embeddings")
    raw = config.get("rope_scaling")
    if raw is None or raw == {}:
        return RopeScaling("none", 1.0, 1.0, 4.0, max_pos)
    if not isinstance(raw, dict):
        raise ValueError(f"{label}.rope_scaling must be an object or null")
    kind = raw.get("rope_type")
    legacy = raw.get("type")
    if any(value is not None and not isinstance(value, str) for value in (kind, legacy)):
        raise ValueError(f"{label}.rope_scaling type must be a string")
    if kind not in (None, "") and legacy not in (None, "") and kind != legacy:
        raise ValueError(f"{label}.rope_scaling has conflicting rope_type and type")
    kind = kind if kind not in (None, "") else legacy
    if kind is None or kind == "":
        if any(key not in ("rope_type", "type") for key in raw):
            raise ValueError(f"{label}.rope_scaling has parameters without a rope type")
        return RopeScaling("none", 1.0, 1.0, 4.0, max_pos)
    if kind in ("none", "default"):
        if any(key not in ("rope_type", "type") for key in raw):
            raise ValueError(f"{label}.rope_scaling none cannot carry scaling parameters")
        return RopeScaling("none", 1.0, 1.0, 4.0, max_pos)
    if kind != "llama3":
        raise ValueError(f"{label}.rope_scaling type {kind!r} is unsupported")
    required = ("factor", "low_freq_factor", "high_freq_factor", "original_max_position_embeddings")
    missing = [field for field in required if field not in raw]
    if missing:
        raise ValueError(f"{label}.rope_scaling missing {', '.join(missing)}")
    factor = _f32_positive(raw["factor"], f"{label}.factor", minimum=1.0)
    low = _f32_positive(raw["low_freq_factor"], f"{label}.low_freq_factor")
    high = _f32_positive(raw["high_freq_factor"], f"{label}.high_freq_factor")
    if high <= low:
        raise ValueError(f"{label}.high_freq_factor must exceed low_freq_factor")
    original = _u32_positive(raw["original_max_position_embeddings"], f"{label}.original_max_position_embeddings")
    return RopeScaling("llama3", factor, low, high, original)


def write_csm_rope_metadata(writer, config: dict, *, bb_max_pos: int,
                            dd_max_pos: int, bb_theta: float) -> tuple[RopeScaling, RopeScaling]:
    """Write both source configs and depth theta using GGUF scalar types."""
    depth = config.get("depth_decoder_config")
    if depth is None:
        depth = {}
    if not isinstance(depth, dict):
        raise ValueError("depth_decoder_config must be an object")
    bb = parse_rope_scaling(config, default_max_pos=bb_max_pos, label="backbone")
    dd = parse_rope_scaling(depth, default_max_pos=dd_max_pos, label="depth")
    dd_theta = _f32_positive(depth.get("rope_theta", bb_theta), "depth.rope_theta")
    for prefix, scaling in (("csm.backbone", bb), ("csm.depth", dd)):
        writer.add_string(f"{prefix}.rope_scaling_type", scaling.kind)
        writer.add_float32(f"{prefix}.rope_scaling_factor", scaling.factor)
        writer.add_float32(f"{prefix}.rope_low_freq_factor", scaling.low_freq_factor)
        writer.add_float32(f"{prefix}.rope_high_freq_factor", scaling.high_freq_factor)
        writer.add_uint32(f"{prefix}.rope_orig_max_pos", scaling.orig_max_pos)
    writer.add_float32("csm.depth.rope_theta", dd_theta)
    return bb, dd
