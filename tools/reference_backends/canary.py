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
    # Intermediate pre-encoder conv stages for bug isolation.
    # Named pre_enc_c{idx} where idx is the nn.Sequential index in
    # enc.pre_encode.conv (0=first conv, 2=dw, 3=pw, 5=dw, 6=pw).
    "pre_enc_c0",
    "pre_enc_c2",
    "pre_enc_c3",
    "pre_enc_c5",
    "pre_enc_c6",
] + [f"encoder_layer_{i}" for i in range(32)]


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Run NeMo Canary reference forward and return stage captures.

    `model_dir` may be either a local path to an extracted .nemo or a
    HuggingFace pretrained name (e.g. "nvidia/canary-1b-v2").
    """
    import sys, torch

    # overrides 7.x enforces covariance + EnforceOverridesMeta at class-definition
    # time, which breaks nv_one_logger (transitive NeMo dep). Swap in lenient
    # versions for the duration of the NeMo import, then restore originals.
    import overrides as _real_overrides
    import overrides.enforce as _real_enforce
    from abc import ABCMeta
    _lenient_override = lambda f=None, **kw: (f if callable(f) else lambda fn: fn)
    class _LenientMeta(ABCMeta): pass
    class _LenientEnforceOverrides(metaclass=_LenientMeta): pass
    _lenient_mod = type(sys)("overrides")
    _lenient_mod.__dict__.update(_real_overrides.__dict__)
    _lenient_mod.override = _lenient_override
    _lenient_mod.overrides = _lenient_override
    _lenient_mod.EnforceOverrides = _LenientEnforceOverrides
    _real_meta = _real_enforce.EnforceOverridesMeta
    _real_enforce.EnforceOverridesMeta = _LenientMeta
    sys.modules["overrides"] = _lenient_mod
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError as e:
        raise SystemExit(
            "NeMo toolkit required.\n"
            "Install: pip install 'nemo_toolkit[asr]'\n"
            f"(import error: {e})")
    finally:
        sys.modules["overrides"] = _real_overrides
        sys.modules["overrides.enforce"] = _real_enforce
        _real_enforce.EnforceOverridesMeta = _real_meta

    pretrained = str(model_dir)
    print(f"  loading NeMo Canary model from {pretrained}")
    if pretrained.startswith("nvidia/") or "/" not in pretrained:
        model = nemo_asr.models.ASRModel.from_pretrained(pretrained)
    else:
        model = nemo_asr.models.ASRModel.restore_from(pretrained)
    model.eval()

    # Disable dither for deterministic mel comparison against C++.
    if hasattr(model, "preprocessor") and hasattr(model.preprocessor, "featurizer"):
        model.preprocessor.featurizer.dither = 0.0
        model.preprocessor.featurizer.pad_to = 0

    # Free sub-models and caches that are not needed for encoder-only
    # inference to reduce peak RAM. Canary loads an auxiliary CTC model
    # during restoration; we can drop it after loading.
    import gc
    for attr in ("ctc_decoder", "ctc", "decoding", "wer", "loss"):
        if hasattr(model, attr):
            try:
                setattr(model, attr, None)
            except Exception:
                pass
    gc.collect()

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
    # Intermediate conv sub-module hooks (4D outputs, handled separately below).
    _pre_conv_indices = {
        "pre_enc_c0": 0,
        "pre_enc_c2": 2,
        "pre_enc_c3": 3,
        "pre_enc_c5": 5,
        "pre_enc_c6": 6,
    }
    pre_conv_handles = []
    pre_conv_captured: Dict[str, torch.Tensor] = {}
    if hasattr(enc, "pre_encode") and hasattr(enc.pre_encode, "conv"):
        conv_seq = enc.pre_encode.conv
        for stage_name, idx in _pre_conv_indices.items():
            if stage_name in stages and idx < len(conv_seq):
                def _make_hook(name):
                    def hook(_m, _inp, out):
                        t = out[0] if isinstance(out, (tuple, list)) else out
                        pre_conv_captured[name] = t.detach().cpu().float()
                    return hook
                pre_conv_handles.append(
                    conv_seq[idx].register_forward_hook(_make_hook(stage_name)))
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
            T_valid = int(feat_len.item())  # NeMo valid frame count
            m = feats[0, :, :T_valid].transpose(0, 1).contiguous()
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
    for h in pre_conv_handles:
        h.remove()
    out.update(_hooks.finalize(captured, T_max=int(enc_len.item())))

    # Reshape 4D conv outputs (B=1, C, H=T, W=Freq) → (T, C*Freq).
    # C*Freq is the "feature" axis (fastest in C++), T is the time axis.
    # Ordering: feature k = channel*(Freq) + freq_bin  (channel outermost).
    # Do NOT clip to T_enc here — each conv stage has its own time dimension
    # (conv0 OH≈T_mel//2, conv2 OH≈T_mel//4, conv5/6 OH=T_enc).
    for stage_name, t4 in pre_conv_captured.items():
        # t4: (B=1, C, H=T_stage, W=Freq)
        t = t4[0]              # (C, T_stage, Freq)
        C, T_stage, Freq = t.shape
        # Permute to (T, C, Freq), then flatten C*Freq → feature axis.
        t = t.permute(1, 0, 2).contiguous().reshape(T_stage, C * Freq)
        out[stage_name] = t.numpy()

    return out
