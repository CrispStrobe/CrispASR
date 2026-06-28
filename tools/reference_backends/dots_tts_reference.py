"""dots.tts reference dump backend for the crispasr-diff harness.

Driven by ``tools/dump_reference.py --backend dots-tts``; the harness
serializes the returned dict to a GGUF the C++ ``crispasr-diff dots-tts``
branch reads.

MEMORY DISCIPLINE (see crispasr-crispembed-dev.md): never load the full
dots.tts model in PyTorch — that is ~11 GB in float32 and OOM-crashes a
16 GB machine. Each stage instead loads ONLY the sub-model under test via
``safe_open`` (lazy, per-tensor bf16->f32) and runs it in isolation:

  PatchEncoder (VAESemanticEncoder, ~128 M / ~0.5 GB f32) — decode_patch on
  a fixed, seeded "patch 0" latent (zero conv tail, empty KV). This is the
  cleanest isolation point and the critical stage to validate first.

Stages dumped (all f32; GGUF column-major == numpy row-major flat):
  penc_in_patch0        (patch_size, latent_dim)   latent fed to decode_patch
  penc_positions_patch0 (out_ds_rate,)             decode positions, as f32
  penc_conv_tail_patch0 (latent_dim, left_pad)     conv tail (zeros for patch 0)
  penc_out_patch0       (1, out_dim=1536)          decode_patch llm_embedding

The ``--model-dir`` is the source safetensors snapshot
(rednote-hilab/dots.tts-soar). ``--audio`` is ignored (dots.tts is TTS).

Env:
  DOTS_SEED   deterministic latent seed (default 0).

Future stages (DiT velocity, BigVGAN) follow the same isolated, lazy pattern.
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "penc_in_patch0",
    "penc_positions_patch0",
    "penc_conv_tail_patch0",
    "penc_out_patch0",
]


def required_packages() -> list[str]:
    return ["dots_tts @ git+https://github.com/rednote-hilab/dots.tts.git"]


class _Cfg(dict):
    """dict with attribute access + .get (duck-types dots_tts ConfigBase)."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:  # pragma: no cover
            raise AttributeError(k) from e


def _install_import_stubs() -> None:
    """Stub heavy text-frontend deps (lingua / WeTextProcessing-pynini) that
    dots_tts.utils.text imports at module load. We only import the backbone
    module and never run text normalization, so inert stubs suffice."""

    class _Dummy:
        def __init__(self, *a, **k):
            pass

        def __call__(self, *a, **k):
            return self

        def __getattr__(self, _n):
            return _Dummy()

    def _mod(name, **attrs):
        if name in sys.modules:
            return
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m

    try:
        import lingua  # noqa: F401
    except ImportError:
        _mod("lingua", Language=_Dummy, LanguageDetectorBuilder=_Dummy)
    try:
        import tn  # noqa: F401
    except ImportError:
        _mod("tn")
        _mod("tn.chinese")
        _mod("tn.chinese.normalizer", Normalizer=_Dummy)
        _mod("tn.english")
        _mod("tn.english.normalizer", Normalizer=_Dummy)


def _npf(t) -> np.ndarray:
    return np.ascontiguousarray(t.detach().cpu().float().numpy())


def _dump_patch_encoder(model_dir: Path, seed: int) -> Dict[str, np.ndarray]:
    """Isolated PatchEncoder (VAESemanticEncoder) decode_patch reference."""
    import torch
    from safetensors import safe_open
    from dots_tts.modules.backbone.semantic_encoder import VAESemanticEncoder

    torch.set_num_threads(2)
    torch.set_grad_enabled(False)

    cj = json.loads((model_dir / "config.json").read_text())
    latent_dim = int(cj["latent_dim"])
    patch_size = int(cj["patch_size"])
    top_cfg = _Cfg({
        "patch_size": patch_size,
        "latent_dim": latent_dim,
        "PatchEncoder": _Cfg(cj["PatchEncoder"]),
    })

    st_path = model_dir / "model.safetensors"
    with safe_open(str(st_path), framework="pt") as f:
        out_dim = int(list(f.get_slice("patch_encoder.out_proj.weight").get_shape())[0])
        penc_sd = {
            k[len("patch_encoder."):]: f.get_tensor(k).float()
            for k in f.keys() if k.startswith("patch_encoder.")
        }
    print(f"[dots-ref/penc] out_dim={out_dim} penc tensors={len(penc_sd)}")

    enc = VAESemanticEncoder(in_dim=latent_dim, out_dim=out_dim, config=top_cfg)
    missing, unexpected = enc.load_state_dict(penc_sd, strict=False)
    if missing:
        print(f"[dots-ref/penc] WARNING missing: {missing}")
    if unexpected:
        print(f"[dots-ref/penc] WARNING unexpected: {unexpected}")
    enc.eval().float()

    g = torch.Generator().manual_seed(seed)
    latent_patch = torch.randn(1, patch_size, latent_dim, generator=g, dtype=torch.float32)
    state = enc.init_decode_state(
        max_audio_patch_count=4, batch_size=1,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    positions = torch.arange(enc.out_ds_rate, dtype=torch.long)  # patch 0: [0, 1]
    embedding, _conv_tail = enc.decode_patch(
        latent_patch, state.conv_tail, state.layer_caches, positions,
    )

    return {
        "penc_in_patch0": _npf(latent_patch[0]),               # (patch_size, latent_dim)
        "penc_positions_patch0": positions.float().numpy(),    # (out_ds_rate,)
        "penc_conv_tail_patch0": _npf(state.conv_tail[0]),     # (latent_dim, left_pad)
        "penc_out_patch0": _npf(embedding[0]),                 # (1, out_dim)
    }


def dump(
    model_dir: "Path | str | None" = None,
    audio: np.ndarray | None = None,
    stages: Set[str] | None = None,
    max_new_tokens: int = 20,
    **kwargs: Any,
) -> Dict[str, np.ndarray]:
    """Return per-stage dots.tts reference activations (isolated, lazy load).

    The harness serializes the returned dict to GGUF. ``audio`` is ignored.
    """
    if stages is None:
        stages = set(DEFAULT_STAGES)
    if model_dir is None:
        model_dir = os.environ.get("DOTS_MODEL_DIR",
                                   "/Volumes/backups/ai/crispasr-gguf/dots-tts-soar-src")
    model_dir = Path(model_dir)
    seed = int(os.environ.get("DOTS_SEED", "0"))

    _install_import_stubs()

    results: Dict[str, np.ndarray] = {}

    if any(s.startswith("penc_") for s in stages):
        results.update(_dump_patch_encoder(model_dir, seed))

    # Keep only requested stages (plus always-present positions/conv_tail).
    out = {k: np.ascontiguousarray(v.astype(np.float32))
           for k, v in results.items()
           if k in stages or k.startswith("penc_")}
    print(f"[dots-ref] returning {len(out)} tensors: {sorted(out)}")
    return out
