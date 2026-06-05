"""NeMo Parakeet-TDT reference dump backend.

Loads `nvidia/parakeet-tdt_ctc-0.6b-ja` (or v3) via NeMo and captures the
mel features and final encoder output for crispasr-diff comparison
against the C++ runtime. Intended for diagnosing the JA TDT decoder bug
(emits 1 token then collapses to blanks) where the C++ encoder
diverged from the NeMo reference at the FIRST output frame.

Stages:

  raw_audio        (N,)            input PCM
  mel_spectrogram  (n_mels, T_mel) NeMo preprocessor output, batch-stripped
  encoder_output   (T_enc, d_model) FastConformer encoder output

Captures match the names that `examples/cli/crispasr_diff_main.cpp`
already looks up for the "parakeet" backend.

Usage:

  python tools/dump_reference.py --backend parakeet \\
      --model-dir nvidia/parakeet-tdt_ctc-0.6b-ja \\
      --audio /tmp/parakeet-ja/ja16k.wav \\
      --output /tmp/parakeet-ja/ref.gguf
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
] + [f"encoder_layer_{i}" for i in range(24)]


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Run NeMo Parakeet-TDT reference forward and return stage captures.

    `model_dir` may be either a local path to an extracted .nemo or a
    HuggingFace pretrained name (e.g. "nvidia/parakeet-tdt_ctc-0.6b-ja").

    Per-layer captures (`pre_encode_output`, `encoder_layer_K`) use
    forward hooks on `model.encoder.pre_encode` and
    `model.encoder.layers[K]` so we get the exact tensor each module
    produces — no manual reconstruction. All captures are transposed
    to (T, d_model) row-major to match crispasr's flat layout.
    """
    import sys, torch

    # overrides 7.x enforces two things that break nv_one_logger (which is a
    # transitive NeMo dependency):
    #
    #  1. The @override decorator checks covariance / signature compatibility.
    #  2. EnforceOverridesMeta.__new__ raises TypeError when a subclass method
    #     shadows a base-class method without @override.
    #
    # nv_one_logger.exporter.exporter.BaseExporter(Exporter) uses
    # EnforceOverridesMeta as its metaclass and defines `initialize` without
    # decorating it, which triggers (2) at class-body execution time — before
    # our swap of sys.modules["overrides"] has any effect, because
    # overrides.enforce is already imported separately as
    # sys.modules["overrides.enforce"].
    #
    # Fix: also replace EnforceOverridesMeta with a plain ABCMeta subclass
    # (no strict checking) inside overrides.enforce, and restore both after
    # the NeMo import is complete.
    import overrides as _real_overrides
    import overrides.enforce as _real_enforce
    from abc import ABCMeta

    # Lenient decorator: just sets __override__ and returns the function.
    _lenient_override = lambda f=None, **kw: (f if callable(f) else lambda fn: fn)

    # Lenient metaclass: skips the "must have @override" check entirely.
    class _LenientEnforcesMeta(ABCMeta):
        pass

    class _LenientEnforceOverrides(metaclass=_LenientEnforcesMeta):
        pass

    # Patch the top-level overrides module.
    _lenient_mod = type(sys)("overrides")
    _lenient_mod.__dict__.update(_real_overrides.__dict__)
    _lenient_mod.override = _lenient_override
    _lenient_mod.overrides = _lenient_override
    _lenient_mod.EnforceOverrides = _LenientEnforceOverrides

    # Patch the enforce sub-module so that any code that does
    #   from overrides.enforce import EnforceOverridesMeta
    # or uses the already-imported metaclass also gets the lenient version.
    _real_meta = _real_enforce.EnforceOverridesMeta
    _real_enforce.EnforceOverridesMeta = _LenientEnforcesMeta

    sys.modules["overrides"] = _lenient_mod
    sys.modules["overrides.enforce"] = _real_enforce  # keep sub-module entry, patched
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError as e:
        raise SystemExit(
            "NeMo toolkit required.\n"
            "Install: pip install 'nemo_toolkit[asr]'\n"
            f"(import error: {e})")
    finally:
        # Restore originals.
        sys.modules["overrides"] = _real_overrides
        sys.modules["overrides.enforce"] = _real_enforce
        _real_enforce.EnforceOverridesMeta = _real_meta

    pretrained = str(model_dir)
    print(f"  loading NeMo Parakeet-TDT model from {pretrained}")
    # NeMo 1.x dispatched ASRModel.from_pretrained to the concrete class via the
    # `target` field in model_config; NeMo 2.x dropped that path and now raises
    # `TypeError: Can't instantiate abstract class ASRModel`. Fall back to the
    # concrete TDT+CTC hybrid class explicitly. Other parakeet variants
    # (rnnt-only / ctc-only) would need their own concrete class — extend here
    # as we add backends for them.
    HybridCls = getattr(nemo_asr.models, "EncDecHybridRNNTCTCBPEModel", None)
    try:
        if pretrained.startswith("nvidia/") or "/" not in pretrained:
            model = nemo_asr.models.ASRModel.from_pretrained(pretrained)
        else:
            model = nemo_asr.models.ASRModel.restore_from(pretrained)
    except TypeError as e:
        if "abstract class ASRModel" not in str(e) or HybridCls is None:
            raise
        print(f"  NeMo 2.x dispatch dropped; falling back to {HybridCls.__name__}")
        if pretrained.startswith("nvidia/") or "/" not in pretrained:
            model = HybridCls.from_pretrained(pretrained)
        else:
            model = HybridCls.restore_from(pretrained)
    model.eval()

    # Disable dither for deterministic mel comparison against C++.
    if hasattr(model, "preprocessor") and hasattr(model.preprocessor, "featurizer"):
        model.preprocessor.featurizer.dither = 0.0
        model.preprocessor.featurizer.pad_to = 0

    dev = next(model.parameters()).device

    sig = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(dev)
    sig_len = torch.tensor([audio.shape[0]], device=dev)

    out: Dict[str, np.ndarray] = {}

    if "raw_audio" in stages:
        out["raw_audio"] = audio.astype(np.float32)

    # ---- Forward hooks: capture per-layer encoder activations ----
    # Shared helper handles registration + (T, D) normalisation. Each
    # backend just declares its (stage_name, nn.Module) list.
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
        # feats: (B=1, n_mels, T_mel). The C++ runtime stores mel in the
        # TimeMels layout (T_mel, n_mels) — ne[0]=n_mels is the fast axis.
        # Transpose so flat-element ordering matches what
        # parakeet_compute_mel returns.
        if "mel_spectrogram" in stages:
            T_valid = int(feat_len.item())  # NeMo valid frame count
            m = feats[0, :, :T_valid].transpose(0, 1).contiguous()
            out["mel_spectrogram"] = m.detach().cpu().float().numpy()

        encf, enc_len = model.encoder(audio_signal=feats, length=feat_len)
        # encf: (B=1, d_model, T_enc) in NeMo's convention. crispasr-diff's
        # parakeet_encoder_r returns (T_enc, d_model), so transpose to match.
        if "encoder_output" in stages:
            T_enc = int(enc_len.item())
            e = encf[0, :, :T_enc].transpose(0, 1).contiguous()
            out["encoder_output"] = e.detach().cpu().float().numpy()

    _hooks.drop_hooks(handles)
    out.update(_hooks.finalize(captured, T_max=int(enc_len.item())))
    return out
