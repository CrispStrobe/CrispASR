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

  raw_audio           (N,)              F32 PCM samples
  mel_spectrogram     (T, 160)          F32 stacked log-mel input_features
  encoder_output      (T, 4*D)          F32 concatenated 4-layer hidden state
  encoder_logits      (T, ctc_vocab)    F32 char-level CTC logits
  projector_output    (T_out, llm_dim)  F32 windowed Q-Former output
  audio_embs_for_llm  (T_out, llm_dim)  F32 projector / embedding_multiplier
  text_ids_with_slots (n_text,)         I32 LLM IDs after add_insertion_slots
  editing_logits      (n_text, vocab)   F32 slot-position LLM logits
  ctc_text            ()                str BPE-CTC text init prediction
  final_text          ()                str argmax + uniq + drop EOS
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
    "audio_embs_for_llm",
    "text_ids_with_slots",
    "editing_logits",
    "ctc_text",
    "final_text",
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

    proj_out = None
    proj_cfg = None
    projector = None
    needs_projector = any(
        s in stages
        for s in (
            "projector_output", "audio_embs_for_llm",
            "text_ids_with_slots", "editing_logits",
            "ctc_text", "final_text",
        )
    )
    if needs_projector:
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

    if "projector_output" in stages and proj_out is not None:
        out["projector_output"] = proj_out[0].detach().cpu().float().numpy()
        print(f"  projector_output shape={tuple(out['projector_output'].shape)}")

    needs_llm = any(
        s in stages
        for s in (
            "audio_embs_for_llm", "text_ids_with_slots",
            "editing_logits", "ctc_text", "final_text",
        )
    )
    if not needs_llm:
        return out

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    # LLM tokenizer: load from the snapshot directly — the upstream
    # NLENARDecoder ctor would try to fetch `ibm-granite/granite-4.0-1b-base`
    # over the network. The snapshot already ships a tokenizer.json /
    # vocab.json / merges.txt that's identical to that base model's
    # tokenizer.
    llm_tokenizer = AutoTokenizer.from_pretrained(snap)

    # LLM model: instantiate from the embedded `llm_config` so we never
    # touch the network. The upstream `flash_attention_2` assertion is
    # NOT paranoia — `GraniteModel.forward` always calls
    # `create_causal_mask` and passes the resulting upper-triangular
    # mask into SDPA, which then enforces causality regardless of
    # `self_attn.is_causal=False`. FA2 reads is_causal directly and
    # ignores the mask, so it's the only backend that produces actually
    # non-causal attention.
    #
    # We sidestep this by monkey-patching `create_causal_mask` to return
    # None whenever it's invoked from the loaded model. With None mask
    # AND is_causal=False, SDPA gives true non-causal attention — which
    # is what the C++ `ggml_flash_attn_ext(..., mask=nullptr)` path also
    # produces.
    import transformers.models.granite.modeling_granite as _granite_mod
    _orig_create_causal_mask = _granite_mod.create_causal_mask
    _granite_mod.create_causal_mask = lambda **kwargs: None

    llm_cfg = AutoConfig.for_model(**full_cfg["llm_config"])
    llm_cfg._attn_implementation = "sdpa"
    llm = AutoModelForCausalLM.from_config(llm_cfg).to(torch.float32).eval()
    sd_llm = {k[len("llm."):]: v for k, v in sd_full.items() if k.startswith("llm.")}
    miss_l, unex_l = llm.load_state_dict(sd_llm, strict=False)
    if miss_l:
        print(f"  warning: llm missing keys: {miss_l[:5]} ({len(miss_l)} total)")
    if unex_l:
        print(f"  warning: llm unexpected keys: {unex_l[:5]} ({len(unex_l)} total)")
    for layer in llm.model.layers:
        layer.self_attn.is_causal = False

    eos_id = llm_cfg.eos_token_id

    # Encoder lengths and downsampled "projected" lengths (same logic as
    # NLENARDecoder.forward).
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(enc_out.logits[..., 0], dtype=torch.bool)
    encoder_lengths = attention_mask.sum(dim=1)
    downsample_rate = proj_cfg.downsample_rate
    projected_lengths = (encoder_lengths // downsample_rate).cpu().tolist()

    # BPE-CTC text init.
    pool_window = enc_cfg.bpe_pooling_window
    if enc_out.logits_bpe is not None:
        bpe_lengths = (-(encoder_lengths // -pool_window)).tolist()
        preds_flat = enc_out.logits_bpe.argmax(dim=-1)
        text_ctc_preds = []
        offset = 0
        for length in bpe_lengths:
            seq = preds_flat[offset:offset + length]
            offset += length
            collapsed = torch.unique_consecutive(seq)
            collapsed = collapsed[collapsed != 0] - 1
            if collapsed.numel() > 0:
                text = llm_tokenizer.decode(collapsed.tolist(), skip_special_tokens=True)
            else:
                text = " "
            text = text.strip().lower() or " "
            text_ctc_preds.append(text)
    else:
        # Char-level CTC fallback (no BPE aux head).
        ctc_preds = torch.where(attention_mask, enc_out.logits.argmax(dim=-1), 0).cpu().numpy()
        # Stub: char tokenizer not loaded here — rely on upstream's ctc_tokenizer
        ctc_mod_local = sys.modules.get(f"{pkg_root}.modeling_ctc")
        ctc_tokenizer = ctc_mod_local.Tokenizer(**(full_cfg.get("ctc_tokenizer_config") or {}))
        text_ctc_preds = [ctc_tokenizer.decode(p).strip() or " " for p in ctc_preds]

    print(f"  text_ctc_preds={text_ctc_preds!r}")

    if "ctc_text" in stages:
        out["ctc_text"] = text_ctc_preds[0]

    # add_insertion_slots: [eos, t0, eos, t1, eos, ...] of length
    # max(2*n+1, 8). Mirrors NLENARDecoder.add_insertion_slots.
    def add_insertion_slots(x):
        n = len(x)
        total_len = max(2 * n + 1, 8)
        result = [eos_id] * total_len
        for i, v in enumerate(x):
            result[2 * i + 1] = int(v)
        return result

    pred_text_llm_tokens = llm_tokenizer(text_ctc_preds)
    text_ids_with_slots = [add_insertion_slots(x) for x in pred_text_llm_tokens.input_ids]

    if "text_ids_with_slots" in stages:
        # Stored as F32 so the existing Ref::get_f32 loader can read it.
        # Vocab fits in 17 bits so the F32 round-trip is bit-exact.
        out["text_ids_with_slots"] = np.array(text_ids_with_slots[0], dtype=np.float32)
        print(f"  text_ids_with_slots shape={out['text_ids_with_slots'].shape}")

    # Build flat embeds matching NLENARDecoder._build_flat_llm_inputs.
    embedding_multiplier = float(getattr(llm_cfg, "embedding_multiplier", 1.0))
    scale_proj = bool(full_cfg.get("scale_projected_embeddings", True))
    audio_for_llm = proj_out
    if scale_proj and embedding_multiplier != 0.0:
        audio_for_llm = proj_out / embedding_multiplier

    if "audio_embs_for_llm" in stages:
        # Slice to projected_lengths[0] frames (match flat-input layout).
        n_audio_kept = projected_lengths[0]
        out["audio_embs_for_llm"] = audio_for_llm[0, :n_audio_kept].detach().cpu().float().numpy()
        print(f"  audio_embs_for_llm shape={out['audio_embs_for_llm'].shape}")

    # Run the LLM editing forward (single sample → trivial flat layout).
    embed_tokens = llm.model.embed_tokens
    text_tensor = torch.tensor(text_ids_with_slots[0])
    text_embs = embed_tokens(text_tensor)
    audio_keep = audio_for_llm[0, :projected_lengths[0]].to(text_embs.dtype)
    flat_embeds = torch.cat([audio_keep, text_embs], dim=0).unsqueeze(0)
    flat_position_ids = torch.arange(flat_embeds.shape[1]).unsqueeze(0)

    with torch.no_grad():
        llm_out = llm.model(
            inputs_embeds=flat_embeds,
            position_ids=flat_position_ids,
            use_cache=False,
        )
    llm_hidden = llm_out.last_hidden_state.squeeze(0)
    text_hidden = llm_hidden[projected_lengths[0]:projected_lengths[0] + len(text_ids_with_slots[0])]
    text_logits = llm.lm_head(text_hidden)

    if "editing_logits" in stages:
        out["editing_logits"] = text_logits.detach().cpu().float().numpy()
        print(f"  editing_logits shape={tuple(out['editing_logits'].shape)}")

    if "final_text" in stages:
        cur_pred = torch.unique_consecutive(text_logits.argmax(-1))
        cur_pred = cur_pred[cur_pred != eos_id]
        final_text = llm_tokenizer.decode(cur_pred.tolist(), skip_special_tokens=True)
        out["final_text"] = final_text
        out["generated_text"] = final_text  # also expose under the standard key
        print(f"  final_text={final_text!r}")

    return out
