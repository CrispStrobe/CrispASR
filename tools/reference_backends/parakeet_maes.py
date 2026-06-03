"""NeMo Parakeet-TDT MAES decoding reference dump backend.

Loads a NeMo Parakeet-TDT model (e.g. nvidia/parakeet-tdt-0.6b-v2) and
captures intermediate activations from the transducer decoding path —
prediction network, joint network, and MAES beam search — so
``crispasr-diff`` can validate the C++ MAES implementation step by step.

This backend builds on the encoder-only ``parakeet.py`` backend by adding
the transducer decoding components that greedy/MAES search needs:

Component-level stages (validate each C++ module independently):

  encoder_output            (T_enc, D_enc)     FastConformer encoder output
  encoder_output_projected  (T_enc, D_joint)   joint.project_encoder(enc)
  decoder_initial           (1, D_pred)        prediction net for blank/SOS
  decoder_initial_projected (1, D_joint)       joint.project_prednet(dec)
  joint_t0                  (V+1+N_dur,)       raw joint logits at frame 0
  joint_t0_token_logp       (V+1,)             log_softmax token head, t=0
  joint_t0_dur_logp         (N_dur,)           log_softmax duration head, t=0

Decode-level stages (validate end-to-end MAES output):

  greedy_tokens             (T_out,)           greedy decode token IDs (I32→F32)
  maes_tokens               (T_out,)           MAES best-hyp token IDs (I32→F32)
  maes_score                (1,)               MAES best hypothesis log-prob

Metadata (GGUF key-value, not tensors):

  greedy_text               greedy decoded text
  generated_text            MAES decoded text (best hypothesis)
  maes_beam_size            beam size used
  maes_num_steps            expansion steps per frame
  maes_gamma                pruning threshold

Usage:

  python tools/dump_reference.py --backend parakeet-maes \\
      --model-dir nvidia/parakeet-tdt-0.6b-v2 \\
      --audio samples/jfk.wav \\
      --output /tmp/parakeet-maes-ref.gguf

The intermediate stages allow crispasr-diff to pinpoint bugs before
running the full MAES loop:
  1. encoder_output_projected → validates joint.project_encoder
  2. decoder_initial → validates StatelessTransducerDecoder for blank
  3. joint_t0 → validates joint forward (enc_proj + dec_proj → logits)
  4. maes_tokens vs greedy_tokens → validates MAES search improvement
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Dict, Set

import numpy as np

# MAES configuration — override via env vars for experimentation.
_BEAM_SIZE = int(os.environ.get("MAES_BEAM_SIZE", "4"))
_NUM_STEPS = int(os.environ.get("MAES_NUM_STEPS", "2"))
_GAMMA = float(os.environ.get("MAES_GAMMA", "2.3"))
_BETA = int(os.environ.get("MAES_BETA", "2"))
_PREFIX_ALPHA = int(os.environ.get("MAES_PREFIX_ALPHA", "1"))

DEFAULT_STAGES = [
    "raw_audio",
    "mel_spectrogram",
    "encoder_output",
    "encoder_output_projected",
    "decoder_initial",
    "decoder_initial_projected",
    "joint_t0",
    "joint_t0_token_logp",
    "joint_t0_dur_logp",
    "greedy_tokens",
    "maes_tokens",
    "maes_score",
]


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Run NeMo Parakeet-TDT with MAES decoding and return stage captures.

    Captures transducer component intermediates (prediction net, joint net)
    plus full MAES decode output for end-to-end validation.
    """
    import sys
    import torch

    # ----------------------------------------------------------------
    # NeMo import with overrides monkey-patch (same as parakeet.py)
    # ----------------------------------------------------------------
    import overrides as _real_overrides
    import overrides.enforce as _real_enforce
    from abc import ABCMeta

    _lenient_override = lambda f=None, **kw: (f if callable(f) else lambda fn: fn)

    class _LenientEnforcesMeta(ABCMeta):
        pass

    class _LenientEnforceOverrides(metaclass=_LenientEnforcesMeta):
        pass

    _lenient_mod = type(sys)("overrides")
    _lenient_mod.__dict__.update(_real_overrides.__dict__)
    _lenient_mod.override = _lenient_override
    _lenient_mod.overrides = _lenient_override
    _lenient_mod.EnforceOverrides = _LenientEnforceOverrides

    _real_meta = _real_enforce.EnforceOverridesMeta
    _real_enforce.EnforceOverridesMeta = _LenientEnforcesMeta

    sys.modules["overrides"] = _lenient_mod
    sys.modules["overrides.enforce"] = _real_enforce
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

    # ----------------------------------------------------------------
    # Load model
    # ----------------------------------------------------------------
    pretrained = str(model_dir)
    print(f"  loading NeMo Parakeet-TDT model from {pretrained}")

    HybridCls = getattr(nemo_asr.models, "EncDecHybridRNNTCTCBPEModel", None)
    try:
        if pretrained.startswith("nvidia/") or "/" not in pretrained:
            model = nemo_asr.models.ASRModel.from_pretrained(pretrained)
        else:
            model = nemo_asr.models.ASRModel.restore_from(pretrained)
    except TypeError as e:
        if "abstract class ASRModel" not in str(e) or HybridCls is None:
            raise
        print(f"  NeMo 2.x dispatch; falling back to {HybridCls.__name__}")
        if pretrained.startswith("nvidia/") or "/" not in pretrained:
            model = HybridCls.from_pretrained(pretrained)
        else:
            model = HybridCls.restore_from(pretrained)
    model.eval()

    # Disable dither for deterministic mel.
    if hasattr(model, "preprocessor") and hasattr(model.preprocessor, "featurizer"):
        model.preprocessor.featurizer.dither = 0.0
        model.preprocessor.featurizer.pad_to = 0

    dev = next(model.parameters()).device
    out: Dict[str, np.ndarray] = {}

    # ----------------------------------------------------------------
    # Audio → mel → encoder
    # ----------------------------------------------------------------
    sig = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0).to(dev)
    sig_len = torch.tensor([audio.shape[0]], device=dev)

    if "raw_audio" in stages:
        out["raw_audio"] = audio.astype(np.float32)

    with torch.no_grad():
        feats, feat_len = model.preprocessor(input_signal=sig, length=sig_len)
        T_mel = int(feat_len.item())

        if "mel_spectrogram" in stages:
            m = feats[0, :, :T_mel].transpose(0, 1).contiguous()
            out["mel_spectrogram"] = m.cpu().float().numpy()

        enc_out, enc_len = model.encoder(audio_signal=feats, length=feat_len)
        T_enc = int(enc_len.item())

        if "encoder_output" in stages:
            # enc_out: (B=1, D_enc, T_enc) → (T_enc, D_enc)
            e = enc_out[0, :, :T_enc].transpose(0, 1).contiguous()
            out["encoder_output"] = e.cpu().float().numpy()

    # ----------------------------------------------------------------
    # Transducer component captures (joint + prediction net)
    # ----------------------------------------------------------------
    # NeMo encoder convention: (B, D, T) → reshape to (B, T, D) for joint
    enc_for_joint = enc_out[:, :, :T_enc].transpose(1, 2).contiguous()
    # enc_for_joint: (1, T_enc, D_enc)

    joint = model.joint
    decoder = model.decoder

    with torch.no_grad():
        # --- encoder projection ---
        # joint.project_encoder() applies self.enc Linear(D_enc → D_joint)
        enc_proj = joint.project_encoder(enc_for_joint)  # (1, T_enc, D_joint)
        if "encoder_output_projected" in stages:
            out["encoder_output_projected"] = (
                enc_proj[0].cpu().float().numpy()  # (T_enc, D_joint)
            )

        # --- prediction network for initial state (blank/SOS) ---
        blank_idx = model.decoder.blank_idx if hasattr(model.decoder, "blank_idx") \
            else getattr(model, '_blank_index',
                         model.joint.num_classes_with_blank - 1)
        print(f"  blank index: {blank_idx}")

        # Initial prediction: feed blank/SOS as the "previous token".
        # RNNTDecoder.predict(y=None, state=None, add_sos=True, batch_size=1)
        # returns the SOS embedding output.
        dec_out_init, dec_state = decoder.predict(
            y=None, state=None, add_sos=True, batch_size=1)
        # dec_out_init: (B, U, D_pred) — for LSTM, U=1 (single step output)
        # but the batch dim may carry extra info. Take [:, -1:, :] for last step.
        print(f"  decoder_initial shape: {dec_out_init.shape}")

        if "decoder_initial" in stages:
            # Store (1, D_pred) — last timestep, no batch
            out["decoder_initial"] = (
                dec_out_init[0, -1:, :].cpu().float().numpy()  # (1, D_pred)
            )

        # --- decoder projection ---
        # Take only last timestep for projection: (B, 1, D_pred)
        dec_out_last = dec_out_init[:, -1:, :]  # (1, 1, D_pred)
        dec_proj_init = joint.project_prednet(dec_out_last)  # (1, 1, D_joint)
        if "decoder_initial_projected" in stages:
            out["decoder_initial_projected"] = (
                dec_proj_init[0].cpu().float().numpy()  # (1, D_joint)
            )

        # --- joint output at frame 0 (raw logits, no log_softmax) ---
        # joint_after_projection applies log_softmax on CPU (NeMo auto mode).
        # We want raw logits so C++ joint_step (which also returns raw logits)
        # can be compared directly. Use joint.joint_net on the summed
        # projections instead.
        enc_t0_proj = enc_proj[:, 0:1, :]  # (1, 1, D_joint)
        dec_u0_proj = dec_proj_init         # (1, 1, D_joint)
        joint_logits = joint.joint_net(enc_t0_proj + dec_u0_proj)  # (1, 1, V+1+N_dur)
        joint_logits = joint_logits.view(-1)  # (V+1+N_dur,) — flatten all dims

        if "joint_t0" in stages:
            out["joint_t0"] = joint_logits.cpu().float().numpy()

        # Split into token and duration heads for TDT.
        # num_extra_outputs is the number of duration classes.
        n_dur = getattr(joint, 'num_extra_outputs', 0)
        n_vocab_blank = joint_logits.shape[0] - n_dur
        print(f"  joint output dim: {joint_logits.shape[0]} "
              f"(vocab+blank={n_vocab_blank}, durations={n_dur})")

        token_logits = joint_logits[:n_vocab_blank]
        dur_logits = joint_logits[n_vocab_blank:] if n_dur > 0 else None

        if "joint_t0_token_logp" in stages:
            token_logp = torch.log_softmax(token_logits, dim=-1)
            out["joint_t0_token_logp"] = token_logp.cpu().float().numpy()

        if "joint_t0_dur_logp" in stages and dur_logits is not None:
            dur_logp = torch.log_softmax(dur_logits, dim=-1)
            out["joint_t0_dur_logp"] = dur_logp.cpu().float().numpy()

    # ----------------------------------------------------------------
    # Greedy decode (baseline for comparison)
    # ----------------------------------------------------------------
    print("  running greedy decode...")
    greedy_cfg = copy.deepcopy(model.cfg.decoding)
    greedy_cfg.strategy = "greedy_batch"
    # EncDecRNNTBPEModel doesn't take decoder_type; the Hybrid variant does.
    try:
        model.change_decoding_strategy(greedy_cfg, decoder_type='rnnt')
    except TypeError:
        model.change_decoding_strategy(greedy_cfg)

    with torch.no_grad():
        greedy_result = model.transcribe(
            [audio.astype(np.float32)],
            batch_size=1,
            return_hypotheses=True,
        )
    # transcribe returns list of Hypothesis objects (or list of lists).
    if isinstance(greedy_result, tuple):
        greedy_result = greedy_result[0]  # some NeMo versions return (hyps, ...)
    greedy_hyp = greedy_result[0]
    if hasattr(greedy_hyp, 'text'):
        greedy_text = greedy_hyp.text
    else:
        greedy_text = str(greedy_hyp)
    print(f"  greedy: {greedy_text!r}")

    if "greedy_tokens" in stages:
        if hasattr(greedy_hyp, 'y_sequence') and greedy_hyp.y_sequence is not None:
            tok = greedy_hyp.y_sequence
            if isinstance(tok, torch.Tensor):
                tok = tok.cpu().numpy()
            out["greedy_tokens"] = np.array(tok, dtype=np.float32)

    # ----------------------------------------------------------------
    # MAES decode
    # ----------------------------------------------------------------
    print(f"  running MAES decode (beam={_BEAM_SIZE}, steps={_NUM_STEPS}, "
          f"gamma={_GAMMA}, beta={_BETA})...")
    maes_cfg = copy.deepcopy(model.cfg.decoding)
    maes_cfg.strategy = "beam"
    maes_cfg.beam = maes_cfg.get("beam", {})
    # OmegaConf DictConfig — use attribute access for nested keys.
    if hasattr(maes_cfg.beam, '__setattr__'):
        maes_cfg.beam.beam_size = _BEAM_SIZE
        maes_cfg.beam.search_type = "maes"
        maes_cfg.beam.maes_num_steps = _NUM_STEPS
        maes_cfg.beam.maes_expansion_gamma = _GAMMA
        maes_cfg.beam.maes_expansion_beta = _BETA
        maes_cfg.beam.maes_prefix_alpha = _PREFIX_ALPHA
        maes_cfg.beam.score_norm = True
        maes_cfg.beam.return_best_hypothesis = False  # get all beams
    else:
        maes_cfg.beam["beam_size"] = _BEAM_SIZE
        maes_cfg.beam["search_type"] = "maes"
        maes_cfg.beam["maes_num_steps"] = _NUM_STEPS
        maes_cfg.beam["maes_expansion_gamma"] = _GAMMA
        maes_cfg.beam["maes_expansion_beta"] = _BETA
        maes_cfg.beam["maes_prefix_alpha"] = _PREFIX_ALPHA
        maes_cfg.beam["score_norm"] = True
        maes_cfg.beam["return_best_hypothesis"] = False

    try:
        model.change_decoding_strategy(maes_cfg, decoder_type='rnnt')
    except TypeError:
        model.change_decoding_strategy(maes_cfg)

    with torch.no_grad():
        maes_result = model.transcribe(
            [audio.astype(np.float32)],
            batch_size=1,
            return_hypotheses=True,
        )
    if isinstance(maes_result, tuple):
        maes_result = maes_result[0]

    # maes_result is a list of NBestHypotheses (one per audio in batch).
    # Each NBestHypotheses has .n_best_hypotheses list sorted by score.
    maes_hyps = maes_result[0]
    if hasattr(maes_hyps, 'n_best_hypotheses'):
        best = maes_hyps.n_best_hypotheses[0]
    elif isinstance(maes_hyps, (list, tuple)):
        best = maes_hyps[0]
    else:
        best = maes_hyps

    maes_text = best.text if hasattr(best, 'text') else str(best)
    print(f"  MAES:   {maes_text!r}")

    if "maes_tokens" in stages:
        if hasattr(best, 'y_sequence') and best.y_sequence is not None:
            tok = best.y_sequence
            if isinstance(tok, torch.Tensor):
                tok = tok.cpu().numpy()
            out["maes_tokens"] = np.array(tok, dtype=np.float32)

    if "maes_score" in stages:
        score = best.score if hasattr(best, 'score') else 0.0
        out["maes_score"] = np.array([float(score)], dtype=np.float32)

    # ----------------------------------------------------------------
    # Metadata — string-typed captures route to GGUF kv automatically
    # ----------------------------------------------------------------
    # dump_reference.py moves str-valued captures into the GGUF header
    # as key-value metadata, accessible via ref.meta("key") in C++.
    out["generated_text"] = maes_text
    out["greedy_text"] = greedy_text
    out["maes_beam_size"] = str(_BEAM_SIZE)
    out["maes_num_steps"] = str(_NUM_STEPS)
    out["maes_gamma"] = str(_GAMMA)
    out["maes_beta"] = str(_BETA)
    out["blank_index"] = str(blank_idx)
    out["n_durations"] = str(n_dur)
    out["vocab_plus_blank"] = str(n_vocab_blank)

    return out
