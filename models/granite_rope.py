"""Granite YaRN metadata, matching the original Transformer implementation."""
import math


def rope_scaling_metadata(text_cfg):
    """Preserve original YaRN semantics; fail instead of silently dropping scaling."""
    rope = text_cfg.get("rope_scaling") or {}
    kind = rope.get("rope_type", rope.get("type", "default"))
    if kind == "default":
        return {}
    if kind != "yarn" or not rope.get("truncate", True):
        raise ValueError(f"Unsupported Granite RoPE scaling: {rope}")
    if float(text_cfg.get("partial_rotary_factor", 1.0)) != 1.0:
        raise ValueError("Granite YaRN requires full-head rotary dimensions")
    factor = float(rope["factor"])
    original = int(rope.get("original_max_position_embeddings") or text_cfg["max_position_embeddings"])
    fast = float(rope.get("beta_fast") or 32)
    slow = float(rope.get("beta_slow") or 1)
    if not all(math.isfinite(v) for v in (factor, fast, slow)) or factor < 1 or original <= 0 or not fast >= slow > 0:
        raise ValueError(f"Invalid Granite YaRN parameters: {rope}")

    def mscale(multiplier=1):
        return 1.0 if factor <= 1 else 1.0 + 0.1 * multiplier * math.log(factor)

    attention = rope.get("attention_factor")
    if attention is None:
        attention = (mscale(float(rope["mscale"])) / mscale(float(rope["mscale_all_dim"]))
                     if rope.get("mscale") and rope.get("mscale_all_dim") else mscale())
    attention = float(attention)
    if not math.isfinite(attention) or attention <= 0:
        raise ValueError("Invalid Granite YaRN attention factor")
    return {"type": "yarn", "factor": factor, "original_context": original,
            "attention_factor": attention, "beta_fast": fast, "beta_slow": slow}
