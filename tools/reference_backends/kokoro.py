"""Kokoro / StyleTTS2 (iSTFTNet) reference dump backend.

Captures stage-by-stage activations from the official `kokoro` PyTorch
package (`pip install kokoro`, or unpacked at /tmp/kokoro_pkg/unpacked)
so the C++ runtime in `src/kokoro.cpp` can be diffed bit-true. The test
phoneme string and voice are embedded in the GGUF metadata so both
sides see exactly the same inputs.

Stages dumped (subset selectable via tools/dump_reference.py --stages):

  token_ids         — raw phoneme ids (post drop-unknown, pre pad-wrap),
                      cast int32 → float32 to fit GGUF's tensor types.
  bert_pooler_out   — CustomAlbert.last_hidden_state, (L_padded, 768).
                      "pooler" is legacy nomenclature; ABI-stable name.
  bert_proj_out     — bert_encoder Linear(768→512), (L_padded, 512).
  text_enc_out      — TextEncoder output (with bidir LSTM), (L_padded, 512).
  dur_enc_out       — DurationEncoder output (with cat-style),
                      (L_padded, 640).
  pred_lstm_out     — pred.lstm output, (L_padded, 512).
  durations         — per-token integer durations, (L_padded,).
  align_out         — d.transpose(-1, -2) @ pred_aln_trg, (T_frames, 640).
  f0_curve          — F0_proj output, (2*T_frames,).
  n_curve           — N_proj output, (2*T_frames,).
  dec_encode_out    — decoder.encode AdainResBlk1d output,
                      (T_frames, 1024).
  dec_decode_3_out  — decoder.decode[3] output (last decode block),
                      (2*T_frames, 512).
  gen_pre_post_out  — input to generator.conv_post (= last LeakyReLU
                      output before conv_post), (T_har, 128).
  mag               — torch.exp(conv_post[:, :11, :]), (T_har, 11).
  phase             — torch.sin(conv_post[:, 11:, :]), (T_har, 11).
  audio_out         — final iSTFT output, (T_samples,) at 24 kHz.

Layout convention: every (L, D) numpy shape mirrors C++ ggml `ne=(D, L)`
(ggml is column-major, ne[0] = innermost). For 2D PyTorch tensors of
shape (B, D, T) we transpose to (T, D); for (B, L, D) we keep as (L, D)
since flat memory already matches.

The "audio" arg from tools/dump_reference.py is repurposed for this
backend as a placeholder (Kokoro doesn't consume it). Phonemes and
voice come from env vars:

  KOKORO_PHONEMES   default: "həˈloʊ wɝld"
  KOKORO_VOICE      default: "af_heart"
  KOKORO_VOICE_PT   path to a voices/<name>.pt; default
                    <model_dir>/voices/<KOKORO_VOICE>.pt
  KOKORO_MODEL_PT   override .pth path; default <model_dir>/kokoro-v*.pth
  KOKORO_SEED       seed for torch.manual_seed; default 0x12345 to
                    mirror the C++ default (note: PyTorch's RNG
                    sequence cannot match std::mt19937, so generator
                    stages 12–15 only diff at looser tolerance).

Critical reference-side behaviours that must match the C++ runtime
(see LEARNINGS.md "Kokoro / StyleTTS2 lessons" for the full
catalogue of plan/source divergences caught during M3-M9):

  1. Pad-wrap input ids `[0, *raw, 0]` — KModel.forward does this.
  2. Drop unknown phonemes — KModel.forward filters None vocab lookups.
  3. AdaIN1d's InstanceNorm1d affine weights are NOT in the GGUF (the
     converter dropped them per the kokoro author's ONNX-export
     workaround); the C++ treats it as `affine=False`. We force the
     reference's InstanceNorm to behave as affine=False by setting
     weight=1, bias=0 on every AdaIN1d.norm before the forward pass.
  4. Banker's rounding for durations — torch.round uses round-half-to-
     even on float32, matching nearbyintf with FE_TONEAREST in C++.
     `torch.set_default_dtype(torch.float32)` before calling forward.
  5. Voice-pack split — pack[L_raw - 1] is (1, 256) F32; the predictor
     uses [:, 128:256] and the decoder uses [:, 0:128].
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Set

import numpy as np


DEFAULT_STAGES = [
    "token_ids",
    "bert_pooler_out",
    "bert_proj_out",
    "text_enc_out",
    "dur_enc_out",
    "pred_lstm_out",
    "durations",
    "align_out",
    "f0_curve",
    "n_curve",
    "dec_encode_out",
    "dec_decode_3_out",
    "gen_pre_post_out",
    "mag",
    "phase",
    "audio_out",
]

_DEFAULT_PHONEMES = "həˈloʊ wɝld"
_DEFAULT_VOICE = "af_heart"
_KOKORO_UNPACKED = Path("/tmp/kokoro_pkg/unpacked")


def _ensure_kokoro_importable() -> None:
    """Make `kokoro.model` importable.

    Bypasses kokoro/__init__.py — that file imports KPipeline → misaki →
    espeak-ng, which fails on conda envs whose phonemizer is out of
    sync with kokoro's. We only need the bare nn.Module classes.
    """
    try:
        import kokoro.model  # noqa: F401
        return
    except (ImportError, AttributeError):
        pass
    if not _KOKORO_UNPACKED.exists():
        raise SystemExit(
            "kokoro package not importable and {p} missing. Install with:\n"
            "  pip download kokoro --no-deps --dest /tmp/kokoro_pkg\n"
            "  unzip -o /tmp/kokoro_pkg/kokoro-*.whl -d /tmp/kokoro_pkg/unpacked"
            .format(p=_KOKORO_UNPACKED))
    if str(_KOKORO_UNPACKED) not in sys.path:
        sys.path.insert(0, str(_KOKORO_UNPACKED))
    # Side-step __init__.py by registering an empty module object before
    # the submodule imports try to bind into it. The submodules use
    # relative imports (`from .modules import ...`), so we just need
    # `kokoro` to exist as a package object in sys.modules.
    import importlib.util
    import types
    pkg = types.ModuleType("kokoro")
    pkg.__path__ = [str(_KOKORO_UNPACKED / "kokoro")]
    sys.modules["kokoro"] = pkg
    import kokoro.model  # noqa: F401  - retry, must succeed now


def _override_adain_affine(kmodel) -> None:
    import torch
    import torch.nn as nn

    n_overridden = 0
    for m in kmodel.modules():
        if isinstance(m, nn.InstanceNorm1d) and m.affine:
            with torch.no_grad():
                if m.weight is not None:
                    m.weight.fill_(1.0)
                if m.bias is not None:
                    m.bias.fill_(0.0)
            n_overridden += 1
    print(f"  forced affine=False on {n_overridden} AdaIN1d.InstanceNorm1d modules")


def _resolve_voice_pack(model_dir: Path) -> tuple[Path, str]:
    voice_name = os.environ.get("KOKORO_VOICE", _DEFAULT_VOICE)
    env_pt = os.environ.get("KOKORO_VOICE_PT", "")
    if env_pt:
        p = Path(env_pt)
    else:
        p = model_dir / "voices" / f"{voice_name}.pt"
    if not p.exists():
        raise SystemExit(
            f"voice pack not found: {p}\n"
            f"  set KOKORO_VOICE_PT=<path> or place {voice_name}.pt under "
            f"{model_dir}/voices/")
    return p, voice_name


def _resolve_model_pth(model_dir: Path) -> Path:
    env_pt = os.environ.get("KOKORO_MODEL_PT", "")
    if env_pt:
        p = Path(env_pt)
        if not p.exists():
            raise SystemExit(f"KOKORO_MODEL_PT={p} not found")
        return p
    candidates = sorted(model_dir.glob("kokoro-v*.pth"))
    if not candidates:
        raise SystemExit(
            f"no kokoro-v*.pth in {model_dir}; set KOKORO_MODEL_PT=<path>")
    return candidates[0]


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Run Kokoro forward, return captured stage tensors keyed by name."""
    _ = audio  # unused — Kokoro is text-driven
    _ = max_new_tokens

    _ensure_kokoro_importable()

    import torch

    # Banker's rounding alignment: torch.round on float32 is round-half-
    # to-even, matching the C++ side's nearbyintf with FE_TONEAREST.
    # Setting the default dtype ensures the duration tensor stays
    # float32 through the sigmoid+sum+round pipeline.
    torch.set_default_dtype(torch.float32)

    from kokoro.model import KModel

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"config.json missing in {model_dir}")
    model_pth = _resolve_model_pth(model_dir)
    voice_pt, voice_name = _resolve_voice_pack(model_dir)
    phonemes = os.environ.get("KOKORO_PHONEMES", _DEFAULT_PHONEMES)
    seed_env = os.environ.get("KOKORO_SEED", "0x12345")
    seed = int(seed_env, 0)

    print(f"  loading Kokoro from {model_pth} (config={config_path.name})")
    kmodel = KModel(repo_id="hexgrad/Kokoro-82M",
                    config=str(config_path),
                    model=str(model_pth),
                    disable_complex=False).eval()
    _override_adain_affine(kmodel)

    # L_raw (post-drop-unknown, pre-pad-wrap) — same logic as
    # KModel.forward + kokoro_phonemes_to_ids on the C++ side.
    raw_ids = [kmodel.vocab.get(p) for p in phonemes]
    raw_ids = [i for i in raw_ids if i is not None]
    L_raw = len(raw_ids)
    if L_raw == 0:
        raise SystemExit(f"no recognised phonemes in '{phonemes}'")

    # Voice pack: shape (max_phon, 1, 256). Index by clamp(L_raw, 1, max).
    # ref_s shape becomes (1, 256). KModel.forward_with_tokens splits
    # this as ref_s[:, 128:] for the predictor and ref_s[:, :128] for
    # the decoder.
    pack = torch.load(str(voice_pt), map_location="cpu", weights_only=True)
    if pack.ndim != 3 or pack.shape[1] != 1:
        raise SystemExit(f"unexpected voice shape {tuple(pack.shape)}")
    if pack.shape[2] != 256:
        raise SystemExit(
            f"voice {voice_name} has style_dim={pack.shape[2]} (need 256)")
    max_phon = pack.shape[0]
    idx = max(0, min(L_raw - 1, max_phon - 1))
    ref_s = pack[idx].to(torch.float32)  # (1, 256)

    print(f"  phonemes={phonemes!r}  L_raw={L_raw}  voice={voice_name}  pack_idx={idx}")

    # Reproducible noise. PyTorch's RNG sequence cannot match
    # std::mt19937 + uniform_real / normal exactly, so the generator
    # noise component still differs — the diff harness applies looser
    # tolerance to those stages. Same seed across runs is still useful
    # so re-running the dumper produces identical bytes.
    torch.manual_seed(seed)

    out: Dict[str, np.ndarray] = {}

    if "token_ids" in stages:
        # int32 cast to float32 to fit GGUF's tensor types — matches the
        # C++ extractor's "token_ids" stage, which also stores raw ids
        # as float for the diff harness.
        out["token_ids"] = np.asarray(raw_ids, dtype=np.int32).astype(np.float32)

    captures: Dict[str, "torch.Tensor"] = {}
    handles = []

    def post_hook(name):
        def h(_mod, _inp, output):
            t = output
            if isinstance(t, tuple):
                t = t[0]
            if hasattr(t, "last_hidden_state"):
                t = t.last_hidden_state
            if isinstance(t, torch.Tensor):
                captures[name] = t.detach().cpu().float()
        return h

    def pre_hook(name):
        def h(_mod, args):
            if not args:
                return
            x = args[0]
            if isinstance(x, torch.Tensor):
                captures[name] = x.detach().cpu().float()
        return h

    # ---- Hook registrations ----
    if "bert_pooler_out" in stages:
        handles.append(kmodel.bert.register_forward_hook(post_hook("bert_pooler_out")))
    if "bert_proj_out" in stages:
        handles.append(kmodel.bert_encoder.register_forward_hook(post_hook("bert_proj_out")))
    if "text_enc_out" in stages:
        handles.append(kmodel.text_encoder.register_forward_hook(post_hook("text_enc_out")))
    if "dur_enc_out" in stages or "align_out" in stages:
        # align_out depends on dur_enc_out (= the `d` in forward_with_tokens)
        handles.append(kmodel.predictor.text_encoder.register_forward_hook(post_hook("dur_enc_out")))
    if "pred_lstm_out" in stages:
        handles.append(kmodel.predictor.lstm.register_forward_hook(post_hook("pred_lstm_out")))
    if "dec_encode_out" in stages:
        handles.append(kmodel.decoder.encode.register_forward_hook(post_hook("dec_encode_out")))
    if "dec_decode_3_out" in stages:
        handles.append(kmodel.decoder.decode[3].register_forward_hook(post_hook("dec_decode_3_out")))
    if "f0_curve" in stages:
        handles.append(kmodel.predictor.F0_proj.register_forward_hook(post_hook("f0_curve")))
    if "n_curve" in stages:
        handles.append(kmodel.predictor.N_proj.register_forward_hook(post_hook("n_curve")))
    # gen_pre_post_out is the input to generator.conv_post (= the last
    # LeakyReLU(0.01) output). Pre-hook captures it before conv_post runs.
    if "gen_pre_post_out" in stages:
        handles.append(
            kmodel.decoder.generator.conv_post.register_forward_pre_hook(
                pre_hook("gen_pre_post_out")))
    # mag/phase: split the conv_post output (B, 22, T_har) and apply
    # exp/sin per the istftnet reference (lines 323-324).
    if "mag" in stages or "phase" in stages:
        handles.append(
            kmodel.decoder.generator.conv_post.register_forward_hook(
                post_hook("__conv_post_out")))

    # ---- Forward ----
    with torch.no_grad():
        out_obj = kmodel.forward(phonemes, ref_s, speed=1, return_output=True)
    audio_t = out_obj.audio
    pred_dur = out_obj.pred_dur

    for h in handles:
        h.remove()

    # ---- Per-stage post-processing into ggml-flat numpy arrays ----

    def first_batch(t: "torch.Tensor") -> "torch.Tensor":
        return t[0] if t.dim() >= 3 else t

    # bert_pooler_out: ALBERT (B, L, 768) → (L, 768) ✓
    if "bert_pooler_out" in captures:
        out["bert_pooler_out"] = first_batch(captures["bert_pooler_out"]).contiguous().numpy().astype(np.float32)
    # bert_proj_out: Linear (B, L, 512) → (L, 512) ✓
    if "bert_proj_out" in captures:
        out["bert_proj_out"] = first_batch(captures["bert_proj_out"]).contiguous().numpy().astype(np.float32)
    # text_enc_out: TextEncoder returns (B, D=512, L) → transpose to (L, D)
    if "text_enc_out" in captures:
        t = first_batch(captures["text_enc_out"]).transpose(0, 1).contiguous().numpy().astype(np.float32)
        out["text_enc_out"] = t
    # dur_enc_out: DurationEncoder returns (B, L, D=640) → (L, D) ✓
    if "dur_enc_out" in captures and "dur_enc_out" in stages:
        out["dur_enc_out"] = first_batch(captures["dur_enc_out"]).contiguous().numpy().astype(np.float32)
    # pred_lstm_out: LSTM output (B, L, 512) → (L, 512) ✓
    if "pred_lstm_out" in captures:
        out["pred_lstm_out"] = first_batch(captures["pred_lstm_out"]).contiguous().numpy().astype(np.float32)
    # decoder body: (B, C, T) → (T, C)
    if "dec_encode_out" in captures:
        t = first_batch(captures["dec_encode_out"]).transpose(0, 1).contiguous().numpy().astype(np.float32)
        out["dec_encode_out"] = t
    if "dec_decode_3_out" in captures:
        t = first_batch(captures["dec_decode_3_out"]).transpose(0, 1).contiguous().numpy().astype(np.float32)
        out["dec_decode_3_out"] = t
    # F0_proj / N_proj: Conv1d output (B, 1, 2T) → squeeze to (2T,)
    if "f0_curve" in captures:
        out["f0_curve"] = captures["f0_curve"][0, 0].contiguous().numpy().astype(np.float32)
    if "n_curve" in captures:
        out["n_curve"] = captures["n_curve"][0, 0].contiguous().numpy().astype(np.float32)
    # gen_pre_post_out: pre-hook gives (B, 128, T_har) → (T_har, 128)
    if "gen_pre_post_out" in captures:
        t = first_batch(captures["gen_pre_post_out"]).transpose(0, 1).contiguous().numpy().astype(np.float32)
        out["gen_pre_post_out"] = t
    # mag / phase: from __conv_post_out (B, 22, T_har)
    if "__conv_post_out" in captures:
        cp = first_batch(captures["__conv_post_out"])  # (22, T_har)
        post_n_fft = 20
        n_bins = post_n_fft // 2 + 1  # 11
        if "mag" in stages:
            mag = torch.exp(cp[:n_bins])
            out["mag"] = mag.transpose(0, 1).contiguous().numpy().astype(np.float32)
        if "phase" in stages:
            phase = torch.sin(cp[n_bins:])
            out["phase"] = phase.transpose(0, 1).contiguous().numpy().astype(np.float32)

    # durations: pred_dur from KModel.forward (already post-round, post-clamp(min=1)).
    if "durations" in stages and pred_dur is not None:
        d = pred_dur.detach().cpu().long().reshape(-1)
        out["durations"] = d.numpy().astype(np.int32).astype(np.float32)

    # align_out: en = d.transpose(-1, -2) @ pred_aln_trg.
    # We build pred_aln_trg the same way KModel.forward_with_tokens does
    # (one-hot at indices = repeat_interleave(arange(L_padded), pred_dur)).
    if "align_out" in stages and "dur_enc_out" in captures and pred_dur is not None:
        d = captures["dur_enc_out"]  # (B, L, 640)
        L_padded = d.shape[1]
        durs = pred_dur.detach().cpu().long().reshape(-1)
        T_frames = int(durs.sum().item())
        if T_frames > 0:
            indices = torch.repeat_interleave(torch.arange(L_padded), durs)
            aln = torch.zeros(L_padded, T_frames)
            aln[indices, torch.arange(T_frames)] = 1
            en = d.transpose(-1, -2) @ aln.unsqueeze(0)  # (B, 640, T_frames)
            out["align_out"] = en[0].transpose(0, 1).contiguous().numpy().astype(np.float32)

    # audio_out: KModel.forward squeezes to (T_audio,).
    if "audio_out" in stages and audio_t is not None:
        out["audio_out"] = audio_t.detach().cpu().contiguous().numpy().astype(np.float32).reshape(-1)

    print(f"  L_padded={L_raw + 2}  pred_dur={pred_dur.detach().cpu().tolist() if pred_dur is not None else 'N/A'}")
    if audio_t is not None:
        print(f"  audio_out: {audio_t.numel()} samples ({audio_t.numel() / 24000.0:.3f}s @ 24 kHz)")

    return out
