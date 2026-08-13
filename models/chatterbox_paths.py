from __future__ import annotations

from pathlib import Path


def select_chatterbox_t3_checkpoint(model_dir: Path) -> Path:
    """Select the best available Chatterbox T3 checkpoint."""

    for name in ("t3_mtl23ls_v3.safetensors", "t3_mtl23ls_v2.safetensors"):
        candidate = model_dir / name
        if candidate.exists():
            return candidate

    for candidate in sorted(model_dir.glob("t3_mtl*.safetensors")):
        return candidate

    return model_dir / "t3_cfg.safetensors"


def select_chatterbox_s3gen_checkpoint(model_dir: Path) -> Path:
    """Select the S3Gen checkpoint used by upstream multilingual V3.

    Chatterbox's V3 release changes the multilingual T3 checkpoint.  The
    production driver still pairs it with the original ``s3gen.pt`` weights
    (the equivalent converter input is ``s3gen.safetensors``).  In particular,
    ``s3gen_v3`` is *not* the currently deployed V3 pairing.
    """

    for name in ("s3gen.safetensors", "s3gen_v3.safetensors"):
        candidate = model_dir / name
        if candidate.exists():
            return candidate

    return model_dir / "s3gen.safetensors"


def select_chatterbox_tokenizer(model_dir: Path) -> Path:
    """Select the tokenizer that matches the selected Chatterbox T3 checkpoint."""

    for name in ("grapheme_mtl_merged_expanded_v1.json", "mtl_tokenizer.json", "tokenizer.json"):
        candidate = model_dir / name
        if candidate.exists():
            return candidate

    return model_dir / "tokenizer.json"
