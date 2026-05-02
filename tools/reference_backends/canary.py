"""NeMo Canary reference dump backend.

Loads `nvidia/canary-1b-v2` (or any compatible NeMo .nemo Canary
checkpoint) via the NeMo toolkit and captures the mel features and the
final encoder output for crispasr-diff comparison against the C++
runtime.

Stages:

  raw_audio        (N,)            input PCM
  mel_spectrogram  (T_mel, n_mels) NeMo preprocessor output, batch-stripped
                                   and transposed to match the C++ runtime's
                                   TimeMels layout (n_mels is the fast axis,
                                   T_mel is the slow axis — ne[0]=n_mels,
                                   ne[1]=T_mel in GGUF).
  encoder_output   (T_enc, d_model) FastConformer encoder output, transposed
                                    from NeMo's (d_model, T_enc) convention
                                    to match `canary_run_encoder`.

Optional per-layer encoder captures (`pre_encode_output`, `encoder_layer_K`)
are emitted when present in `--stages`. The C++ diff harness today only
compares mel + encoder_output, but the per-layer tensors are useful for
diagnostic when chasing a divergence.

Usage:

  python tools/dump_reference.py --backend canary \\
      --model-dir nvidia/canary-1b-v2 \\
      --audio samples/jfk.wav \\
      --output /tmp/canary-ref.gguf
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "raw_audio",
    "mel_spectrogram",
    "pre_encode_output",
    "encoder_output",
] + [f"encoder_layer_{i}" for i in range(32)]


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Run NeMo Canary reference forward and return stage captures.

    `model_dir` may be either a local path to an extracted .nemo or a
    HuggingFace pretrained name (e.g. "nvidia/canary-1b-v2").
    """
    import torch
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError as e:
        raise SystemExit(
            "NeMo toolkit required.\n"
            "Install: pip install 'nemo_toolkit[asr]'\n"
            f"(import error: {e})")

    pretrained = str(model_dir)
    print(f"  loading NeMo Canary model from {pretrained}")
    if pretrained.startswith("nvidia/") or "/" not in pretrained:
        model = nemo_asr.models.ASRModel.from_pretrained(pretrained)
    else:
        model = nemo_asr.models.ASRModel.restore_from(pretrained)
    model.eval()
    dev = next(model.parameters()).device

    sig = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(dev)
    sig_len = torch.tensor([audio.shape[0]], device=dev)

    out: Dict[str, np.ndarray] = {}

    if "raw_audio" in stages:
        out["raw_audio"] = audio.astype(np.float32)

    # Per-layer hooks via the shared helper. Same pattern as parakeet.py.
    from . import _hooks
    captured: Dict[str, torch.Tensor] = {}

    enc = model.encoder
    stage_modules = []
    if "pre_encode_output" in stages and hasattr(enc, "pre_encode"):
        stage_modules.append(("pre_encode_output", enc.pre_encode))
    layers = getattr(enc, "layers", None)
    if layers is not None:
        for i in range(len(layers)):
            stage = f"encoder_layer_{i}"
            if stage in stages:
                stage_modules.append((stage, layers[i]))
    handles = _hooks.capture_modules(captured, stage_modules)

    with torch.no_grad():
        feats, feat_len = model.preprocessor(input_signal=sig, length=sig_len)
        # feats: (B=1, n_mels, T_mel). The C++ runtime's `canary_compute_mel`
        # uses core_mel::Layout::TimeMels — flat layout (T_mel, n_mels) with
        # n_mels as the fast axis. Transpose so the numpy flat ordering
        # matches. Same convention as parakeet.py.
        if "mel_spectrogram" in stages:
            m = feats[0].transpose(0, 1).contiguous()
            out["mel_spectrogram"] = m.detach().cpu().float().numpy()

        encf, enc_len = model.encoder(audio_signal=feats, length=feat_len)
        # encf: (B=1, d_model, T_enc) in NeMo's convention. The C++ side
        # `canary_run_encoder` returns (T_enc, d_model), so transpose to
        # match.
        if "encoder_output" in stages:
            T_enc = int(enc_len.item())
            e = encf[0, :, :T_enc].transpose(0, 1).contiguous()
            out["encoder_output"] = e.detach().cpu().float().numpy()

    _hooks.drop_hooks(handles)
    out.update(_hooks.finalize(captured, T_max=int(enc_len.item())))
    return out
