"""Granite-Speech 4.1-2b NAR (NLENARDecoder) reference dump backend.

Captures the input mel + encoder output (concatenated 4-layer hidden
states per `encoder_layer_indices = [4, 8, 12, -1]`) for the NAR
variant. Mirrors what the C++ runtime exposes through
`granite_nle_compute_mel` and `granite_nle_run_encoder`.

The HF model under `ibm-granite/granite-speech-4.1-2b-nar` ships custom
modeling code, so we load with `trust_remote_code=True`. The encoder
outputs `last_hidden_state`, `logits`, `logits_bpe` (or None) and a
tuple `all_hidden_states` of length `n_layers + 1` where index 0 is
the post-input_linear hidden state and index N is the output of the
N-th conformer block (including the self-conditioning residual at
N == self_conditioning_layer).

Stages exposed:

  raw_audio          (N,)             F32 PCM samples
  mel_spectrogram    (T, 160)         F32 stacked log-mel input_features
  encoder_output     (T, 4*D)         F32 concatenated 4-layer hidden state
  encoder_logits     (T, ctc_vocab)   F32 char-level CTC logits
  projector_output   (T_out, llm_dim) F32 windowed Q-Former output
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Set

import numpy as np


DEFAULT_STAGES = [
    "raw_audio",
    "mel_spectrogram",
    "encoder_output",
    "encoder_logits",
    "projector_output",
]


def _resolve_local_snapshot(model_dir: Path) -> Path:
    """Resolve `ibm-granite/granite-speech-4.1-2b-nar` to its local HF cache
    directory if --model-dir was given as a hub repo id. Required because
    the AutoModel path tries to fetch the LLM tokenizer over the network
    even when we don't need the LLM, so we sidestep the full constructor
    by importing the modeling code directly from the local snapshot.
    """
    s = str(model_dir)
    if Path(s).is_dir():
        return Path(s)
    import os
    base = Path(os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
                 or Path.home() / ".cache" / "huggingface" / "hub")
    repo_dir = base / f"models--{s.replace('/', '--')}"
    snaps = repo_dir / "snapshots"
    if not snaps.is_dir():
        raise SystemExit(f"could not resolve '{s}' to a local snapshot under {snaps}")
    latest = sorted(snaps.iterdir())[-1]
    return latest


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    import importlib.util
    import sys
    import torch

    snap = _resolve_local_snapshot(model_dir)
    print(f"  loading granite-speech-4.1-2b-nar (encoder only) from {snap}")

    # Bypass AutoModel: the upstream NLENARDecoder constructor tries to
    # fetch the LLM tokenizer over the network, which trips up offline
    # runs. Instead we load just the encoder by importing the modeling
    # code directly from the local snapshot.
    def _load_local(name: str):
        spec = importlib.util.spec_from_file_location(
            f"_granite_nle_local.{name}", snap / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        # The modeling files import each other via relative imports; make
        # them resolve to the same _granite_nle_local package.
        spec.loader.exec_module(mod)
        return mod

    # Order matters: configuration_nle defines the configs the others use.
    pkg_root = "_granite_nle_local"
    if pkg_root not in sys.modules:
        sys.modules[pkg_root] = importlib.util.module_from_spec(
            importlib.util.spec_from_loader(pkg_root, loader=None))
    cfg_mod = _load_local("configuration_nle")
    sys.modules[f"{pkg_root}.configuration_nle"] = cfg_mod
    conf_mod = _load_local("modeling_conformer")
    sys.modules[f"{pkg_root}.modeling_conformer"] = conf_mod
    ctc_mod = _load_local("modeling_ctc")
    sys.modules[f"{pkg_root}.modeling_ctc"] = ctc_mod
    proj_mod = _load_local("modeling_projector")
    sys.modules[f"{pkg_root}.modeling_projector"] = proj_mod
    feat_mod = _load_local("feature_extraction_nle")

    # Build encoder config from the JSON config. NLEConfig stores its
    # encoder sub-config as a dict; instantiate NLEEncoderConfig from it.
    import json
    with open(snap / "config.json") as fh:
        full_cfg = json.load(fh)
    enc_cfg = cfg_mod.NLEEncoderConfig(**full_cfg["encoder_config"])
    enc_layer_indices = list(full_cfg.get("encoder_layer_indices", [-1]))

    encoder = ctc_mod.NLECTCEncoder(enc_cfg).to(torch.float32).eval()

    # Load encoder weights from the safetensors file. Keys in the
    # state-dict are prefixed with "encoder." in the full model.
    from safetensors.torch import load_file
    sd_full = load_file(str(snap / "model.safetensors"))
    sd_enc = {k[len("encoder."):]: v for k, v in sd_full.items()
              if k.startswith("encoder.")}
    missing, unexpected = encoder.load_state_dict(sd_enc, strict=False)
    if missing:
        print(f"  warning: missing keys: {missing[:5]} ({len(missing)} total)")
    if unexpected:
        print(f"  warning: unexpected keys: {unexpected[:5]} ({len(unexpected)} total)")

    feat_ext = feat_mod.NLEFeatureExtractor()
    wav = torch.from_numpy(audio.astype(np.float32))
    inputs = feat_ext(wav)

    out: Dict[str, np.ndarray] = {}

    if "mel_spectrogram" in stages and "input_features" in inputs:
        feats = inputs["input_features"]
        if feats.ndim == 3:
            feats = feats[0]
        out["mel_spectrogram"] = feats.detach().cpu().float().numpy()

    indices = enc_layer_indices
    print(f"  encoder_layer_indices={indices}")

    with torch.no_grad():
        enc_out = encoder(
            input_features=inputs["input_features"],
            attention_mask=inputs.get("attention_mask"),
            output_hidden_states=True,
        )

    if "encoder_logits" in stages and enc_out.logits is not None:
        out["encoder_logits"] = enc_out.logits[0].detach().cpu().float().numpy()

    enc_concat = None
    if "encoder_output" in stages or "projector_output" in stages:
        all_h = enc_out.all_hidden_states
        if all_h is None:
            raise RuntimeError(
                "encoder didn't return all_hidden_states even with "
                "output_hidden_states=True — model code mismatch")
        # HF semantics: index 0 = post-input_linear, index N = output of
        # block N. Negative indices count from the end of the tuple.
        sel = [all_h[idx] for idx in indices]
        # cat along feature dim → (B, T, K * D)
        enc_concat = torch.cat(sel, dim=-1)

    if "encoder_output" in stages:
        out["encoder_output"] = enc_concat[0].detach().cpu().float().numpy()
        print(f"  encoder_output shape={tuple(out['encoder_output'].shape)} "
              f"(K={len(indices)} layers)")

    if "projector_output" in stages:
        # Build the projector and load its safetensors weights from the same
        # snapshot. Same trick as the encoder: bypass the full NLE constructor.
        proj_cfg = cfg_mod.NLEProjectorConfig(**full_cfg.get("projector_config", {}))
        projector = proj_mod.EncoderProjectorQFormer(proj_cfg).to(torch.float32).eval()
        sd_proj = {k[len("projector."):]: v for k, v in sd_full.items()
                   if k.startswith("projector.")}
        miss_p, unex_p = projector.load_state_dict(sd_proj, strict=False)
        if miss_p:
            print(f"  warning: projector missing keys: {miss_p[:5]} ({len(miss_p)} total)")
        if unex_p:
            print(f"  warning: projector unexpected keys: {unex_p[:5]} ({len(unex_p)} total)")
        with torch.no_grad():
            proj_out = projector(enc_concat)
        out["projector_output"] = proj_out[0].detach().cpu().float().numpy()
        print(f"  projector_output shape={tuple(out['projector_output'].shape)}")

    return out
