"""LFM2.5-Audio reference dump backend.

Loads LiquidAI/LFM2.5-Audio-1.5B-JP (or any LFM2-Audio variant) via the
liquid-audio package and captures intermediates at every architectural
boundary for crispasr-diff comparison against the C++ runtime.

Stages:

  raw_audio           (N,)              input PCM @ 16 kHz
  mel_spectrogram     (T_mel, n_mels)   log-mel features (128 bins)
  encoder_output      (T_enc, d_model)  FastConformer encoder output (512-dim)
  adapter_output      (T_enc, hidden)   after audio_adapter MLP (→2048-dim)
  lfm_layer_K         (1, T, hidden)    LFM2 backbone after block K
  lfm_output          (1, T, hidden)    LFM2 backbone final (after embedding_norm)
  generated_text      (string)          greedy-decoded transcript (metadata)

For ASR, we use sequential generation mode with the ASR system prompt.

Usage:

  python tools/dump_reference.py --backend lfm2-audio \\
      --model-dir LiquidAI/LFM2.5-Audio-1.5B-JP \\
      --audio samples/jfk.wav \\
      --output /tmp/lfm2-audio-ref.gguf

Env vars:
  LFM2_PROMPT     system prompt (default: "Perform ASR in japanese.")
  LFM2_MAX_TOKENS max new tokens for generation (default: 512)
"""

from __future__ import annotations
try:
    from reference_backends._safe_capture import own as _own
except ImportError:  # run as a standalone script from this directory
    from _safe_capture import own as _own

import os
from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "raw_audio",
    "mel_spectrogram",
    "pre_encode_output",
    "encoder_output",
    "adapter_output",
    "lfm_output",
    "lfm_audio_only_output",
    "generated_text",
] + [f"encoder_layer_{i}" for i in range(17)] + [f"lfm_layer_{i}" for i in range(16)]


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Run LFM2-Audio reference forward and return stage captures."""
    import torch

    prompt = os.environ.get("LFM2_PROMPT", "Perform ASR in japanese.")
    max_tok = int(os.environ.get("LFM2_MAX_TOKENS", str(max_new_tokens or 512)))

    print(f"  Loading LFM2-Audio from {model_dir}")
    print(f"  System prompt: {prompt!r}")
    print(f"  Max tokens: {max_tok}")

    # --- Load model + processor ---
    from liquid_audio import LFM2AudioModel, LFM2AudioProcessor
    from liquid_audio.processor import ChatState

    device = "cpu"
    dtype = torch.float32  # F32 for reference accuracy

    processor = LFM2AudioProcessor.from_pretrained(str(model_dir), device=device)
    processor.audio_processor.featurizer.dither = 0.0  # deterministic mel

    model = LFM2AudioModel.from_pretrained(str(model_dir), dtype=dtype, device=device)
    model.eval()

    d_model = model.conformer._feat_out  # encoder output dim (512)

    out: Dict[str, np.ndarray | str] = {}

    if "raw_audio" in stages:
        out["raw_audio"] = audio.astype(np.float32)

    # --- Hook: mel spectrogram ---
    # Must be registered BEFORE building ChatState, because add_audio()
    # calls the preprocessor during construction.
    mel_captured = {}

    if "mel_spectrogram" in stages:
        def mel_hook(module, inp, output):
            feats, feat_len = output
            # feats: (B=1, n_mels, T_mel)
            T_valid = int(feat_len[0].item())
            m = feats[0, :, :T_valid].transpose(0, 1).contiguous()
            mel_captured["mel_spectrogram"] = _own(m.detach().float())
        h_mel = processor.audio_processor.register_forward_hook(mel_hook)

    # --- Build chat state (triggers preprocessor via add_audio) ---
    chat = ChatState(processor, dtype=dtype)
    chat.new_turn("system")
    chat.add_text(prompt)
    chat.end_turn()
    chat.new_turn("user")

    pcm = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    chat.add_audio(pcm, 16000)
    chat.end_turn()

    # --- Hook: pre-encoder output ---
    pre_enc_captured = {}

    if "pre_encode_output" in stages and hasattr(model.conformer, 'pre_encode'):
        def pre_enc_hook(module, inp, output):
            # pre_encode returns (d_model, T_enc) tensor
            t = output if isinstance(output, torch.Tensor) else output[0]
            pre_enc_captured["pre_encode_output"] = _own(t.detach().float())
        h_pre_enc = model.conformer.pre_encode.register_forward_hook(pre_enc_hook)

    # --- Hook: per-encoder-layer ---
    enc_layer_captured = {}
    enc_layer_handles = []

    if hasattr(model.conformer, 'layers'):
        for i, layer in enumerate(model.conformer.layers):
            stage_name = f"encoder_layer_{i}"
            if stage_name in stages:
                def make_enc_hook(name):
                    def hook(module, inp, output):
                        t = output if isinstance(output, torch.Tensor) else output[0]
                        enc_layer_captured[name] = _own(t.detach().float())
                    return hook
                enc_layer_handles.append(layer.register_forward_hook(make_enc_hook(stage_name)))

    # --- Hook: conformer encoder (full) ---
    enc_captured = {}

    if "encoder_output" in stages:
        def enc_hook(module, inp, output):
            enc, enc_len = output
            # enc: (B, D, T) — transpose to (T, D)
            T_enc = int(enc_len[0].item())
            e = enc[0, :, :T_enc].transpose(0, 1).contiguous()
            enc_captured["encoder_output"] = _own(e.detach().float())
        h_enc = model.conformer.register_forward_hook(enc_hook)

    # --- Hook: audio adapter ---
    adap_captured = {}

    if "adapter_output" in stages:
        def adap_hook(module, inp, output):
            adap_captured["adapter_output"] = _own(output.detach().float())
        h_adap = model.audio_adapter.register_forward_hook(adap_hook)

    # --- Hook: LFM2 per-layer ---
    lfm_captured = {}
    lfm_handles = []

    for i, layer in enumerate(model.lfm.layers):
        stage_name = f"lfm_layer_{i}"
        if stage_name in stages:
            def make_hook(name):
                def hook(module, inp, output):
                    t = output if isinstance(output, torch.Tensor) else output[0]
                    lfm_captured[name] = _own(t.detach().float())
                return hook
            lfm_handles.append(layer.register_forward_hook(make_hook(stage_name)))

    # --- Hook: LFM2 final output ---
    lfm_out_captured = {}

    if "lfm_output" in stages:
        def lfm_out_hook(module, inp, output):
            lfm_out_captured["lfm_output"] = _own(output.last_hidden_state.detach().float())
        h_lfm = model.lfm.register_forward_hook(lfm_out_hook)

    # --- Run prefill to capture encoder + adapter + LFM backbone ---
    print("  Running prefill (hooks capture intermediates)...")
    with torch.no_grad():
        # _prefill internally calls conformer + adapter, then assembles
        # the embedding sequence. We then run lfm forward on it.
        in_emb = model._prefill(
            text=chat["text"],
            audio_in=chat["audio_in"],
            audio_in_lens=chat["audio_in_lens"],
            audio_out=chat["audio_out"],
            modality_flag=chat["modality_flag"],
        )
        # Run LFM to trigger per-layer and output hooks
        _ = model.lfm(inputs_embeds=in_emb, use_cache=False)

    # --- Audio-only LFM pass for C++ comparison ---
    # The C++ run_lfm currently only runs on audio embeddings (no text tokens).
    # Run a separate LFM forward on just the adapter output to create a
    # comparable reference, with per-layer hooks for staged comparison.
    lfm_ao_stages = {s for s in stages if s.startswith("lfm_ao_layer_")}
    want_ao = "lfm_audio_only_output" in stages or lfm_ao_stages
    if want_ao and "adapter_output" in adap_captured:
        print("  Running LFM on audio-only embeddings (with per-layer hooks)...")

        ao_layer_captured = {}
        ao_handles = []
        for i, layer in enumerate(model.lfm.layers):
            stage_name = f"lfm_ao_layer_{i}"
            if stage_name in stages:
                def make_ao_hook(name):
                    def hook(module, inp, output):
                        t = output if isinstance(output, torch.Tensor) else output[0]
                        ao_layer_captured[name] = _own(t.detach().float())
                    return hook
                ao_handles.append(layer.register_forward_hook(make_ao_hook(stage_name)))

        with torch.no_grad():
            audio_emb = adap_captured["adapter_output"].to(model.lfm.embed_tokens.weight.dtype)
            if audio_emb.dim() == 2:
                audio_emb = audio_emb.unsqueeze(0)  # (1, T_enc, hidden)
            lfm_audio = model.lfm(inputs_embeds=audio_emb, use_cache=False)

            if "lfm_audio_only_output" in stages:
                out["lfm_audio_only_output"] = lfm_audio.last_hidden_state.detach().float().numpy()
                print(f"    lfm_audio_only_output: {out['lfm_audio_only_output'].shape}")

        for h in ao_handles:
            h.remove()

        for name, tensor in sorted(ao_layer_captured.items()):
            out[name] = tensor.numpy()
            print(f"    {name}: {tensor.shape}")

    # --- Collect hook outputs ---
    if "mel_spectrogram" in stages and "mel_spectrogram" in mel_captured:
        h_mel.remove()
        out["mel_spectrogram"] = mel_captured["mel_spectrogram"].numpy()
        print(f"    mel_spectrogram: {out['mel_spectrogram'].shape}")

    if "pre_encode_output" in stages and "pre_encode_output" in pre_enc_captured:
        h_pre_enc.remove()
        t = pre_enc_captured["pre_encode_output"]
        if t.dim() == 3:
            t = t[0]
        # NeMo conformer internal layout is (d_model, T_enc).
        # Transpose to (T_enc, d_model) to match encoder_output convention.
        if t.shape[0] == d_model:
            t = t.transpose(0, 1).contiguous()
        out["pre_encode_output"] = t.numpy()
        print(f"    pre_encode_output: {t.shape}")

    for name, tensor in sorted(enc_layer_captured.items()):
        if name in stages:
            t = tensor
            if t.dim() == 3:
                t = t[0]
            # Same (d, T) → (T, d) transpose
            if t.shape[0] == d_model:
                t = t.transpose(0, 1).contiguous()
            out[name] = t.numpy()
            print(f"    {name}: {t.shape}")
    for h in enc_layer_handles:
        h.remove()

    if "encoder_output" in stages:
        h_enc.remove()
        if "encoder_output" in enc_captured:
            out["encoder_output"] = enc_captured["encoder_output"].numpy()
            print(f"    encoder_output: {out['encoder_output'].shape}")

    if "adapter_output" in stages:
        h_adap.remove()
        if "adapter_output" in adap_captured:
            out["adapter_output"] = adap_captured["adapter_output"].numpy()
            print(f"    adapter_output: {out['adapter_output'].shape}")

    for name, tensor in sorted(lfm_captured.items()):
        if name in stages:
            out[name] = tensor.numpy()
            print(f"    {name}: {tensor.shape}")

    for h in lfm_handles:
        h.remove()

    if "lfm_output" in stages:
        h_lfm.remove()
        if "lfm_output" in lfm_out_captured:
            out["lfm_output"] = lfm_out_captured["lfm_output"].numpy()
            print(f"    lfm_output: {out['lfm_output'].shape}")

    # --- Sequential generation (ASR) ---
    if "generated_text" in stages:
        print("  Running sequential generation for ASR...")
        generated_tokens = []
        generated_text_str = ""

        with torch.no_grad():
            for token in model.generate_sequential(
                text=chat["text"],
                audio_in=chat["audio_in"],
                audio_in_lens=chat["audio_in_lens"],
                audio_out=chat["audio_out"],
                modality_flag=chat["modality_flag"],
                max_new_tokens=max_tok,
            ):
                if token.numel() == 1:
                    tok_id = token.item()
                    generated_tokens.append(tok_id)
                    piece = processor.text.decode([tok_id])
                    generated_text_str += piece
                # Audio tokens are ignored for ASR

        print(f"    Generated: {generated_text_str!r}")
        print(f"    Tokens: {len(generated_tokens)}")

        out["generated_text"] = generated_text_str
        if generated_tokens:
            out["generated_tokens"] = np.array(generated_tokens, dtype=np.int32)

    return out
