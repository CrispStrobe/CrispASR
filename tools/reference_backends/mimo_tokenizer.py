"""MiMo-Audio-Tokenizer encoder reference dump backend.

Captures stage-by-stage activations from the official PyTorch
`MiMoAudioTokenizer.encoder` so the C++ runtime in src/mimo_tokenizer.cpp
can be compared element-wise via crispasr-diff.

Pipeline (see ref/mimo/github/src/mimo_audio_tokenizer/modeling_audio_tokenizer.py):

  PCM 24 kHz
    → MelSpectrogram(n_fft=960, hop=240, win=960, n_mels=128, power=1.0,
                     center=True) → log(clip(.,1e-7))            (T_mel, 128)
    → conv1 (Conv1d 128→1280, k=3, p=1)        + GELU            (1280, T_mel)
    → conv2 (Conv1d 1280→1280, k=3, s=2, p=1)  + GELU            (1280, T_mel/2)
    → permute (1280, T) → (T, 1280)
    → 32× TransformerLayer (LN-pre, RoPE θ=10000, head_dim=64,
        q/v/o have biases, k has NO bias, GELU FFN 1280→5120→1280)
        — capture skip after layer (encoder_skip_layer_id-1) = layer 2
    → hidden += skip_connect_hidden_states
    → final LayerNorm                                            (T_mel/2, 1280)
    → down_sample_layer = Conv1d(1280, 1280, k=2, s=2, no bias) + GELU
    → down_sample_norm  = LayerNorm                              (T_mel/4, 1280)
    → ResidualVectorQuantizer.encode (8 stages, codebook lookup) (T_mel/4, 8) int

Stages dumped (all (T, D) row-major except codes):
  tok_mel              (T_mel,    128)   F32 log-mel features
  tok_conv1_out        (T_mel,   1280)   F32 after conv1+GELU (channels-last)
  tok_conv2_out        (T_mel/2, 1280)   F32 after conv2+GELU
  tok_xfmr_out         (T_mel/2, 1280)   F32 after 32 layers + skip + final LN
  tok_pool_out         (T_mel/4, 1280)   F32 after down_sample + GELU + LN
  tok_codes            (T_mel/4, 8)      I32 8-channel RVQ codes (saved as F32)

Audio is loaded at 16 kHz mono by the shared loader; this backend resamples
to 24 kHz to match the tokenizer's expected input rate.

Environment:
  MIMO_TOKENIZER_DIR — path to the MiMo-Audio-Tokenizer HF snapshot
                       (defaults to upstream auto-download).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Set

import numpy as np


DEFAULT_STAGES = [
    "tok_mel",
    "tok_conv1_out",
    "tok_conv2_out",
    "tok_xfmr_out",
    "tok_pool_out",
    "tok_codes",
]


def _stub_flash_attn():
    """Pre-register a fake `flash_attn` module so the upstream imports work.

    The tokenizer source does `from flash_attn import flash_attn_varlen_func`
    at module level, which hard-fails on macOS / CPU-only builds. We replace
    the real call site via _patch_attention_for_no_flash_attn() so the stub
    function never actually runs.

    Transformers also probes `is_flash_attn_2_available()` via
    `importlib.util.find_spec("flash_attn")`, which raises if the module's
    `__spec__` attribute is `None`. We attach a synthetic ModuleSpec so the
    probe returns truthy without trying to actually import flash_attn — the
    monkey-patched Attention.forward never reaches the symbol anyway.
    """
    import sys, types
    from importlib.machinery import ModuleSpec
    if "flash_attn" in sys.modules:
        return
    mod = types.ModuleType("flash_attn")
    mod.__spec__ = ModuleSpec("flash_attn", loader=None)
    mod.__version__ = "0.0.0-stub"
    def _stub(*_a, **_kw):
        raise RuntimeError("flash_attn stub called — Attention.forward should be patched")
    mod.flash_attn_varlen_func = _stub
    sys.modules["flash_attn"] = mod


def _patch_attention_for_no_flash_attn():
    """Replace the upstream Attention.forward to use torch SDPA.

    The upstream uses `flash_attn_varlen_func`, which isn't available on
    macOS / CPU-only builds. For batch=1 the variable-length packing is a
    no-op, so plain scaled_dot_product_attention is bit-equivalent.
    """
    import torch
    from torch.nn import functional as F
    from mimo_audio_tokenizer.modeling_audio_tokenizer import Attention
    from mimo_audio_tokenizer.modeling_rope_utils import apply_rotary_pos_emb

    def forward(self, hidden_states, seq_len, rope_position_embeddings=None):
        # hidden_states: (sum_T, D) packed across the batch.
        bsz = hidden_states.shape[0]  # really sum_T
        q = self.q_proj(hidden_states).view(bsz, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, self.num_heads, self.head_dim)
        if rope_position_embeddings is not None:
            cos, sin = rope_position_embeddings
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)
        # Reconstruct per-sample attention from cu_len. For batch=1 the
        # whole packed sequence IS one sample, which is the only case we
        # exercise from the diff harness.
        if seq_len.numel() != 1:
            raise NotImplementedError(
                "MiMo tokenizer ref dumper only supports batch=1 input")
        T = int(seq_len[0].item())
        # (T, H, D) → (1, H, T, D) for SDPA
        q4 = q.view(1, T, self.num_heads, self.head_dim).transpose(1, 2)
        k4 = k.view(1, T, self.num_heads, self.head_dim).transpose(1, 2)
        v4 = v.view(1, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q4, k4, v4, attn_mask=None, dropout_p=0.0, is_causal=self.causal)
        # (1, H, T, D) → (T, H*D)
        out = attn.transpose(1, 2).contiguous().view(T, self.embed_dim)
        return self.out_proj(out)

    Attention.forward = forward


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Run the tokenizer encoder forward and return captured stage tensors.

    `audio` is 16 kHz mono float32 from the shared loader.
    """
    import torch
    import torchaudio

    # Make `from mimo_audio_tokenizer ...` resolve to the local upstream tree.
    ref_path = Path(__file__).resolve().parents[2] / "ref" / "mimo" / "github" / "src"
    if ref_path.is_dir() and str(ref_path) not in sys.path:
        sys.path.insert(0, str(ref_path))

    _stub_flash_attn()
    _patch_attention_for_no_flash_attn()
    from mimo_audio_tokenizer.modeling_audio_tokenizer import MiMoAudioTokenizer

    src_dir = Path(os.environ.get("MIMO_TOKENIZER_DIR", model_dir))
    print(f"  loading MiMoAudioTokenizer from {src_dir} (CPU, fp32)")
    tok = MiMoAudioTokenizer.from_pretrained(str(src_dir), dtype=torch.float32)
    tok.eval()
    enc = tok.encoder
    cfg = tok.config

    # 16 kHz → 24 kHz (tokenizer's native sampling rate).
    pcm = torch.tensor(audio, dtype=torch.float32)
    if cfg.sampling_rate != 16000:
        pcm = torchaudio.functional.resample(pcm, 16000, cfg.sampling_rate)

    # Match upstream MimoAudio.wav2mel: power=1.0 magnitude → log(clip,1e-7).
    mel_xform = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.sampling_rate, n_fft=cfg.nfft, hop_length=cfg.hop_length,
        win_length=cfg.window_size, f_min=cfg.fmin, f_max=cfg.fmax,
        n_mels=cfg.n_mels, power=1.0, center=True)
    spec = mel_xform(pcm[None, :])  # (1, n_mels, T_mel)
    log_mel = torch.log(torch.clip(spec, min=1e-7))  # (1, 128, T_mel)

    out: Dict[str, np.ndarray] = {}
    if "tok_mel" in stages:
        # Store as (T_mel, 128) row-major to match the C++ ggml layout.
        out["tok_mel"] = log_mel.squeeze(0).T.contiguous().numpy().astype(np.float32)

    # Forward-hook captures. The packed-sequence flow makes mid-encoder
    # captures awkward, so we capture conv stem outputs explicitly (before
    # packing) and instrument the encoder_layer trail via a thin wrapper.
    # We monkey-patch get_features to capture along the way without
    # rewriting the whole method.
    captures: Dict[str, torch.Tensor] = {}

    orig_get_features = enc.get_features

    def get_features_with_capture(input_features, output_length):
        # Reproduce upstream get_features verbatim, capturing per stage.
        x = input_features.to(enc.conv1.weight)
        a = torch.nn.functional.gelu(enc.conv1(x))
        if "tok_conv1_out" in stages:
            captures["tok_conv1_out"] = a.detach().clone()  # (1, 1280, T_mel)
        b = torch.nn.functional.gelu(enc.conv2(a))
        if "tok_conv2_out" in stages:
            captures["tok_conv2_out"] = b.detach().clone()  # (1, 1280, T_mel/2)
        h = b.permute(0, 2, 1)  # (1, T_mel/2, 1280)
        bsz, tgt_len, _ = h.size()

        from mimo_audio_tokenizer.modeling_audio_tokenizer import (
            get_position_ids, get_sequence_mask)
        position_ids = get_position_ids(output_length).long().to(x.device)
        rope_pos = enc.position_embedding(x, position_ids)
        attn_mask, unpacking_index = get_sequence_mask(h, output_length)
        h = torch.masked_select(h, attn_mask).view(
            torch.sum(output_length), enc.config.d_model)

        skip = 0.0
        for idx, layer in enumerate(enc.layers):
            h = layer(h, output_length, rope_position_embeddings=rope_pos)
            if (enc.skip_layer_idx is not None) and idx == enc.skip_layer_idx - 1:
                skip = h.clone()
        h = h + skip
        h = enc.layer_norm(h)
        if "tok_xfmr_out" in stages:
            captures["tok_xfmr_out"] = h.detach().clone()  # (T_mel/2, 1280) packed

        if enc.down_sample_layer is not None:
            h_unpacked = torch.index_select(h, 0, unpacking_index).view(
                bsz, tgt_len, enc.config.d_model)
            local_tgt_len = tgt_len
            if h_unpacked.size(1) % enc.config.avg_pooler:
                pad = enc.config.avg_pooler - h_unpacked.size(1) % enc.config.avg_pooler
                h_unpacked = torch.nn.functional.pad(
                    h_unpacked, (0, 0, 0, pad), mode="constant", value=0.0)
                local_tgt_len += pad
            local_tgt_len = local_tgt_len // enc.config.avg_pooler
            h_pooled = enc.down_sample_layer(h_unpacked.transpose(1, 2))
            new_len = (output_length // enc.config.avg_pooler
                       + (output_length % enc.config.avg_pooler != 0).int())
            h_pooled = h_pooled.transpose(1, 2)
            attn_mask2, unpacking_index2 = get_sequence_mask(h_pooled, new_len)
            h_packed = torch.masked_select(h_pooled, attn_mask2).view(
                torch.sum(new_len), enc.config.d_model)
            h_packed = enc.down_sample_norm(h_packed)
            if "tok_pool_out" in stages:
                captures["tok_pool_out"] = h_packed.detach().clone()
            return (h_packed, new_len, attn_mask2, unpacking_index2,
                    local_tgt_len, bsz)
        return h, output_length, attn_mask, unpacking_index, tgt_len, bsz

    enc.get_features = get_features_with_capture
    try:
        # AudioEncoder.encode() expects PACKED input: shape (sum_T, n_mels)
        # — `unpack_hidden_states(input_features, input_lens)` turns it back
        # into (B, T, n_mels). For batch=1 the "sum_T" is just T_mel and the
        # packed view is mel_packed.squeeze(0). The misleading variable name
        # in the upstream signature is `input_features`; it must already be
        # packed.
        T_mel = log_mel.shape[-1]
        input_lens = torch.tensor([T_mel])
        mel_packed = log_mel.transpose(1, 2).squeeze(0)  # (T_mel, 128)
        with torch.no_grad():
            # `return_codes_only=True` returns (codes, output_length); the
            # full-output 4-tuple is (hidden_states, hidden_states_packed,
            # output_length, codes).
            codes, output_length = enc.encode(
                mel_packed, input_lens=input_lens, return_codes_only=True)
    finally:
        enc.get_features = orig_get_features

    # Convert captures to (T, D) row-major numpy.
    for name, t in list(captures.items()):
        if t.dim() == 3:  # (1, D, T) channels-first → (T, D)
            arr = t[0].T.contiguous().cpu().numpy().astype(np.float32)
        elif t.dim() == 2:  # (T, D) packed
            arr = t.contiguous().cpu().numpy().astype(np.float32)
        else:
            continue
        out[name] = arr

    if "tok_codes" in stages:
        # codes shape: (n_q=20, T_codes). Slice 8 channels for ASR + transpose.
        codes_np = codes[:8, :].T.contiguous().cpu().numpy().astype(np.int32)
        # diff harness loads as F32; store as float for cosine compare.
        out["tok_codes"] = codes_np.astype(np.float32)

    return out
