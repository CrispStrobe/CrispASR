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
    "ctc_logits",
] + [f"encoder_layer_{i}" for i in range(24)]


def _attach_fluid_speech(model, nemo_path: str) -> None:
    """FluidVoice-Labs/fluid-speech-pro (#595): mirror fluid_speech.load().

    The .nemo carries `encoder.fv_insert.*` (a FrameCleaner block run as a
    forward hook on encoder.pre_encode) and an optional `clean_mode.*`
    adapter. A plain restore_from(strict=False) silently drops both, so add
    the block and load every tensor strictly like upstream. Clean mode is
    folded when CRISPASR_REF_FLUID_CLEAN=1. No-op for other checkpoints.
    """
    import os, tarfile, tempfile
    import torch
    import torch.nn as nn

    spec = model.cfg.get("fluid_speech") if hasattr(model, "cfg") else None
    if not spec:
        return

    class FrameCleaner(nn.Module):  # verbatim structure of fluid_speech.FrameCleaner
        def __init__(self, width=1024, dim=1024, kernel=15):
            super().__init__()
            self.ln = nn.LayerNorm(width)
            self.down = nn.Linear(width, dim)
            self.mix = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
            self.up = nn.Linear(dim, width)

        def forward(self, x):
            h = self.down(self.ln(x)).transpose(1, 2)
            return x + self.up(nn.functional.gelu(self.mix(h)).transpose(1, 2))

    enc = model.encoder
    enc.fv_insert = FrameCleaner(enc.d_model, int(spec.insert_dim), int(spec.insert_kernel)).to(
        next(enc.parameters()).device)

    def clean_frames(module, args, out):
        if isinstance(out, tuple):
            return (enc.fv_insert(out[0]),) + tuple(out[1:])
        return enc.fv_insert(out)

    enc.pre_encode.register_forward_hook(clean_frames)

    with tarfile.open(nemo_path) as tar, tempfile.TemporaryDirectory(
            dir=os.path.dirname(os.path.abspath(nemo_path))) as tmp:
        member = next(m for m in tar.getmembers() if os.path.basename(m.name) == "model_weights.ckpt")
        tar.extract(member, tmp, filter="data")
        # mmap: the state dict is only read while copying into the model.
        state = torch.load(os.path.join(tmp, member.name), map_location="cpu", weights_only=True, mmap=True)
        _load_fluid_state(model, spec, state)
        del state  # unmap before the temp dir is removed (CIFS/Windows refuse otherwise)
        import gc
        gc.collect()
    print(f"  fluid-speech: FrameCleaner attached (dim={spec.insert_dim} kernel={spec.insert_kernel})")


def _load_fluid_state(model, spec, state) -> None:
    """Strict load, then fold clean mode in place (same sums as
    fluid_speech.fold_clean_mode, without a second copy of the state)."""
    import os
    import torch
    pre = "clean_mode."
    adapter = {k[len(pre):]: state.pop(k) for k in [k for k in state if k.startswith(pre)]}
    model.load_state_dict(state, strict=True)
    if os.environ.get("CRISPASR_REF_FLUID_CLEAN") != "1":
        return
    scale = float(spec.clean_mode.alpha) / int(spec.clean_mode.rank)
    params = dict(model.named_parameters())
    di, jh = adapter.pop("decoder_input"), adapter.pop("joint_hidden")
    lstm = "decoder.prediction.dec_rnn.lstm"
    with torch.no_grad():
        params[f"{lstm}.bias_ih_l0"].add_(params[f"{lstm}.weight_ih_l0"] @ di)
        params["joint.pred.bias"].add_(jh)
        for name in sorted({k.rsplit(".", 1)[0] for k in adapter}):
            w = params[f"encoder.{name}.weight"]
            w.add_((scale * (adapter[f"{name}.B"] @ adapter[f"{name}.A"])).view_as(w).to(w.dtype))
    print("  fluid-speech: clean mode folded")


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
            try:
                model = nemo_asr.models.ASRModel.restore_from(pretrained)
            except RuntimeError as e:
                # Fluid Speech (#595): extra encoder.fv_insert.* / clean_mode.*
                # keys; _attach_fluid_speech() re-loads every tensor strictly.
                if "fv_insert" not in str(e) and "clean_mode." not in str(e):
                    raise
                model = nemo_asr.models.ASRModel.restore_from(pretrained, strict=False)
    except TypeError as e:
        if "abstract class ASRModel" not in str(e) or HybridCls is None:
            raise
        print(f"  NeMo 2.x dispatch dropped; falling back to {HybridCls.__name__}")
        if pretrained.startswith("nvidia/") or "/" not in pretrained:
            model = HybridCls.from_pretrained(pretrained)
        else:
            model = HybridCls.restore_from(pretrained)
    _attach_fluid_speech(model, pretrained)
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

        # CTC head: NeMo's ConvASRDecoder returns log-probs (B, T_enc, V+1,
        # blank last) - the same log P(v | t) grid canary_ctc_compute_logits()
        # returns. A hybrid TDT/CTC model keeps it as ctc_decoder; a pure
        # CTC model as decoder. Pure TDT/RNNT models have none (their
        # `decoder` is the prediction net), so the stage is skipped there -
        # this file is shared with fastconformer_ctc.py via a symlink.
        dec = getattr(model, "ctc_decoder", None)
        if dec is None and type(getattr(model, "decoder", None)).__name__ == "ConvASRDecoder":
            dec = model.decoder
        if "ctc_logits" in stages and dec is not None:
            lp = dec(encoder_output=encf)
            T_enc = int(enc_len.item())
            out["ctc_logits"] = lp[0, :T_enc].detach().cpu().float().numpy().copy()

    _hooks.drop_hooks(handles)
    out.update(_hooks.finalize(captured, T_max=int(enc_len.item())))
    return out
