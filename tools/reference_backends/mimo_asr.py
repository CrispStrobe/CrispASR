"""MiMo-V2.5-ASR LM-half reference dump backend.

Captures stage-by-stage activations from the official PyTorch
`MiMoAudioForCausalLM` forward pass on the assembled `asr_sft` prompt
so the C++ runtime in src/mimo_asr.cpp can be compared element-wise via
crispasr-diff.

Pipeline (see ref/mimo/github/src/mimo_audio/modeling_mimo_audio.py):

  audio PCM (16 kHz)
    → MiMo-Audio-Tokenizer encoder → 8-channel RVQ codes [T_a, 8] @ 25 fps
    → pad-to-multiple-of-group_size
    → asr_sft prompt template builder (process_speechdata.InputSegment):
      [<|im_start|>user\\n] [audio segment] [asr_en_template] [<|im_end|>\\n]
      [<|im_start|>assistant\\n] [<think>\\n\\n</think>\\n<english>]
      → input_ids [9, T_total] = (channel 0: text mostly <|empty|>;
                                  channels 1..8: audio codes per channel)
    → _prepare_input_embeds:
        per-channel speech embedding lookup, sum, mask zero on empty rows
        → [B, T_groups, group_size=4, hidden=1024]
        → input_local_transformer (6L Qwen2 1024d, 64×16, full attention,
                                   RoPE θ=640000, FFN=4096) per-group
        → reshape [B, T_groups, group_size*1024=4096]
        → speech_group_downcast Linear(4096→4096)
        → mask non-speech positions
        → text_embeds = embed_tokens(text_input_ids) (mask zero on empty)
        → inputs_embeds = text_embeds + speech_grouped_embeds  [B, T_groups, 4096]
    → 36L Qwen2 LM (4096d, 32×8 GQA, head_dim=128, RoPE θ=640000,
                    intermediate=11008, RMSNorm eps=1e-6, q/k/v biases)
    → final RMSNorm + lm_head
    → text_logits at last group position → argmax → next text token

Stages dumped (all (T, D) row-major where applicable):
  prefill_input_ids       (9, T_total)         I32 — [9, T] (text + 8 audio)
  prefill_audio_codes     (T_a_padded, 8)      I32 — for diff against C++
                                                     mimo_tokenizer codes
  prefill_audio_features  (T_groups, 4096)     F32 — post group_proj (the
                                                     "speech_grouped_embeds"
                                                     inside _prepare_input_embeds,
                                                     before adding text_embeds)
  prefill_inputs_embeds   (T_groups, 4096)     F32 — final inputs_embeds
                                                     fed to the LM
  prefill_last_hidden     (hidden=4096,)       F32 — LM last-position hidden
                                                     state after final norm
                                                     (input to lm_head)
  prefill_text_logits_step0 (vocab=151680,)    F32 — lm_head output at last
                                                     position
  prefill_text_argmax_step0 (1,)               I32 — argmax of the above
  generated_text          str                  — full decoded transcript

Audio is loaded at 16 kHz mono by the shared loader; this backend pulls
the tokenizer from MIMO_TOKENIZER_DIR (or downloads via HF) and the LM
from MIMO_ASR_DIR (or `model_dir`).

Environment:
  MIMO_TOKENIZER_DIR  — MiMo-Audio-Tokenizer HF snapshot (if not auto)
  MIMO_ASR_DIR        — MiMo-V2.5-ASR HF snapshot (defaults to model_dir)
  MIMO_ASR_AUDIO_TAG  — language tag, default "<english>"
  MIMO_ASR_MAX_NEW    — generated_text max_new_tokens (default 64)
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "prefill_input_ids",
    "prefill_audio_codes",
    "prefill_audio_features",
    "prefill_text_embeds",
    "prefill_inputs_embeds",
    "prefill_last_hidden",
    "prefill_text_logits_step0",
    "prefill_text_argmax_step0",
    "generated_text",
]


def _ensure_upstream_on_path():
    """Make `from src.mimo_audio ...` and `from src.mimo_audio_tokenizer ...`
    resolve to the local upstream tree under ref/mimo/github/.

    We need `src` itself to be importable as a (namespace) package — not
    just have its children on sys.path — because mimo_audio.mimo_audio
    does `from ..mimo_audio_tokenizer import MiMoAudioTokenizer`, which
    requires `mimo_audio` to live two levels deep inside an importable
    parent (`src`). So we put the parent of `src` on sys.path and import
    via the `src.` prefix throughout."""
    ref_path = Path(__file__).resolve().parents[2] / "ref" / "mimo" / "github"
    if ref_path.is_dir() and str(ref_path) not in sys.path:
        sys.path.insert(0, str(ref_path))


def _stub_flash_attn():
    """Pre-register a fake `flash_attn` so the upstream tokenizer imports
    work on macOS / CPU-only hosts. Mirrors the trick in mimo_tokenizer.py."""
    import sys, types
    from importlib.machinery import ModuleSpec
    if "flash_attn" in sys.modules:
        return
    mod = types.ModuleType("flash_attn")
    mod.__spec__ = ModuleSpec("flash_attn", loader=None)
    mod.__version__ = "0.0.0-stub"

    def _stub(*_a, **_kw):
        raise RuntimeError("flash_attn stub called")

    mod.flash_attn_varlen_func = _stub
    sys.modules["flash_attn"] = mod


def _patch_tokenizer_attention_for_no_flash_attn():
    """Same patch as in mimo_tokenizer.py — replace
    `flash_attn_varlen_func` calls with torch SDPA so the audio
    tokenizer can run on macOS / CPU."""
    import torch
    from torch.nn import functional as F
    from src.mimo_audio_tokenizer.modeling_audio_tokenizer import Attention
    from src.mimo_audio_tokenizer.modeling_rope_utils import apply_rotary_pos_emb

    def forward(self, hidden_states, seq_len, rope_position_embeddings=None):
        bsz = hidden_states.shape[0]
        q = self.q_proj(hidden_states).view(bsz, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, self.num_heads, self.head_dim)
        if rope_position_embeddings is not None:
            cos, sin = rope_position_embeddings
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)
        if seq_len.numel() != 1:
            raise NotImplementedError("only batch=1 supported in this dumper")
        T = int(seq_len[0].item())
        q4 = q.view(1, T, self.num_heads, self.head_dim).transpose(1, 2)
        k4 = k.view(1, T, self.num_heads, self.head_dim).transpose(1, 2)
        v4 = v.view(1, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q4, k4, v4, attn_mask=None, dropout_p=0.0, is_causal=self.causal)
        out = attn.transpose(1, 2).contiguous().view(T, self.embed_dim)
        return self.out_proj(out)

    Attention.forward = forward


class _LowMemMimoAudio:
    """Two-phase replacement for upstream `MimoAudio` that keeps peak
    memory under 16 GB on the dump path.

    The upstream class loads BOTH the LM (~16 GB bf16) AND the audio
    tokenizer (~3.6 GB bf16) in `__init__`, peaking at ~20 GB co-resident
    — kernel SIGKILLs on a 16 GB Mac. This shim splits the loads:

      Phase A:  __init__               — text tokenizer + AutoConfig only (~few MB)
      Phase B:  load_audio_tokenizer() — MiMoAudioTokenizer + mel transform (~3.6 GB)
                preprocess_input(...)  — produce audio_tokenized
                free_audio_tokenizer() — drop, gc.collect
      Phase C:  load_lm()              — MiMoAudioForCausalLM (~16 GB)
                ...forward...

    Peak: max(3.6 GB tokenizer phase, 16 GB LM phase) ≈ 16 GB.
    Mirrors `ref/mimo/github/src/mimo_audio/mimo_audio.py:__init__`
    byte-for-byte aside from the load split.
    """

    def __init__(self, model_path: str, mimo_audio_tokenizer_path: str, device: str = "cpu"):
        import torch
        from transformers import AutoConfig, AutoTokenizer

        self.device = device
        self.path = model_path
        self.mimo_audio_tokenizer_path = mimo_audio_tokenizer_path

        # Text tokenizer (small).
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.padding_idx = int(self.tokenizer.pad_token_id)
        for tok in ("<|sosp|>", "<|eosp|>", "<|empty|>", "<|Human|>", "<|SpeechLM|>", "<|sostm|>", "<|eostm|>",
                    "<|eot|>"):
            if tok not in self.tokenizer.get_vocab():
                self.tokenizer.add_tokens([tok], special_tokens=True)
        cti = self.tokenizer.convert_tokens_to_ids
        self.sosp_idx = cti("<|sosp|>")
        self.eosp_idx = cti("<|eosp|>")
        self.empty_token = cti("<|empty|>")
        self.sostm_idx = cti("<|sostm|>")
        self.eostm_idx = cti("<|eostm|>")
        self.eot_idx = cti("<|eot|>")
        self.im_start_idx = cti("<|im_start|>")
        self.im_end_idx = cti("<|im_end|>")

        # LM scalar metadata from config (no weight load). We read the
        # config rather than instantiating the model because the model
        # weights are 16 GB and we're trying to avoid loading them in
        # this phase. Fallback defaults match MiMo-V2.5-ASR's published
        # config (and src/mimo_asr.cpp::mimo_asr_hp).
        cfg = AutoConfig.from_pretrained(self.path, trust_remote_code=True)
        self.group_size = getattr(cfg, "group_size", 4)
        self.audio_channels = getattr(cfg, "audio_channels", 8)
        self.vocab_size = getattr(cfg, "vocab_size", 151680)
        # `model.speech_empty_ids` per channel — set by
        # MiMoAudioForCausalLM.__init__ from config.speech_empty_ids when
        # present, else from the canonical MiMo-V2.5-ASR values (matches
        # src/mimo_asr.cpp::speech_zeroemb_idx).
        self.speech_zeroemb_idx = list(getattr(cfg, "speech_empty_ids",
                                               [1024, 1024, 128, 128, 128, 128, 128, 128]))

        # Lazy fields — populated by load_audio_tokenizer() / load_lm().
        self.mimo_audio_tokenizer = None
        self.mel_transform = None
        self.model = None

    def load_audio_tokenizer(self):
        import torch
        from torchaudio.transforms import MelSpectrogram

        from src.mimo_audio_tokenizer.modeling_audio_tokenizer import MiMoAudioTokenizer
        self.mimo_audio_tokenizer = MiMoAudioTokenizer.from_pretrained(self.mimo_audio_tokenizer_path)
        self.mimo_audio_tokenizer.eval().bfloat16().to(self.device)
        cfg = self.mimo_audio_tokenizer.config
        self.mel_transform = MelSpectrogram(
            sample_rate=cfg.sampling_rate, n_fft=cfg.nfft, hop_length=cfg.hop_length, win_length=cfg.window_size,
            f_min=cfg.fmin, f_max=cfg.fmax, n_mels=cfg.n_mels, power=1.0, center=True,
        ).to(self.device)

    def free_audio_tokenizer(self):
        self.mimo_audio_tokenizer = None
        self.mel_transform = None
        gc.collect()

    def load_lm(self, dtype):
        from src.mimo_audio.mimo_audio import MiMoAudioArguments
        from src.mimo_audio.modeling_mimo_audio import MiMoAudioForCausalLM

        args = MiMoAudioArguments(
            model_name_or_path=self.path,
            sosp_idx=self.sosp_idx,
            eosp_idx=self.eosp_idx,
            empty_idx=self.empty_token,
            sostm_idx=self.sostm_idx,
            eostm_idx=self.eostm_idx,
            eot_idx=self.eot_idx,
        )
        self.model = MiMoAudioForCausalLM.from_pretrained(
            self.path, args=args, torch_dtype=dtype, device_map={"": self.device},
        )
        self.model.eval()

    # --- Methods mirroring the upstream MimoAudio surface used by the dumper ---

    def wav2mel(self, wav):
        import torch
        spec = self.mel_transform(wav[None, :])
        return torch.log(torch.clip(spec, min=1e-7)).squeeze()

    def resample_audio_if_needed(self, wav_tensor, original_sr: int):
        import torchaudio
        target_sr = self.mimo_audio_tokenizer.config.sampling_rate
        if original_sr != target_sr:
            wav_tensor = torchaudio.functional.resample(wav_tensor, original_sr, target_sr)
        return wav_tensor

    def preprocess_input(self, audio):
        """Inline copy of `MimoAudio.preprocess_input` (and its
        `encode_batch` / `group_by_length` helpers, simplified for
        single-batch). Mirrors ref/mimo/github/src/mimo_audio/mimo_audio.py
        byte-for-byte for the no-streaming case."""
        import torch
        wav = audio.to(self.device)
        target_sr = self.mimo_audio_tokenizer.config.sampling_rate
        chunk_samples = 30 * target_sr
        n_fft = self.mimo_audio_tokenizer.config.nfft

        total_samples = wav.shape[-1]
        code_parts = []
        start = 0
        while start < total_samples:
            end = min(start + chunk_samples, total_samples)
            if 0 < total_samples - end < n_fft:
                end = total_samples
            chunk = wav[start:end]
            if chunk.shape[-1] < n_fft:
                chunk = torch.nn.functional.pad(chunk, (0, n_fft - chunk.shape[-1]))
            mel = self.wav2mel(chunk).transpose(0, 1)  # [seq_len, n_mels]
            with torch.no_grad():
                codes_chunk, _ = self.mimo_audio_tokenizer.encoder.encode(
                    input_features=mel.to(self.device),
                    input_lens=torch.tensor([mel.size(0)]).to(self.device),
                    return_codes_only=True,
                )
            code_parts.append(codes_chunk)
            start = end

        codes_packed = torch.cat(code_parts, dim=-1)
        codes = codes_packed.transpose(0, 1).detach().cpu()
        audio_codes = codes[:, :self.audio_channels]

        # Pad to multiple of group_size by repeating the last frame.
        n = audio_codes.shape[0]
        if n % self.group_size != 0:
            need = self.group_size - (n % self.group_size)
            audio_codes = torch.cat([audio_codes, audio_codes[-1:, :].repeat(need, 1)], dim=0)
        return audio_codes.flatten()

    def get_input_ids(self, prompt):
        import torch
        ids = [seg.to_input_id(self.tokenizer, self.group_size, self.audio_channels) for seg in prompt]
        return torch.cat(ids, dim=1).to(self.device)


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Run MiMo-V2.5-ASR reference forward, return captured stage tensors.

    Memory-defensive: the upstream `MimoAudio` class loads the 16 GB LM
    + 3.6 GB audio tokenizer co-resident, peaking at ~20 GB and OOMing
    on a 16 GB Mac. We use `_LowMemMimoAudio` (defined above) which
    splits the loads:

      1. Init: text tokenizer + AutoConfig only.
      2. Load audio tokenizer (3.6 GB) → preprocess_input → free.
      3. Load LM (16 GB) → forward + capture stages.

    Peak memory: max(3.6 GB, 16 GB) ≈ 16 GB. Plus three smaller knobs:
      - `torch.set_num_threads(1)` — lower per-thread allocator pressure.
      - `torch.inference_mode()` — drops autograd metadata vs `no_grad`.
      - `gc.collect()` between stages — frees intermediate activations.

    Set `MIMO_REF_KEEP_TOKENIZER=1` to keep the audio tokenizer resident
    through the LM forward (only useful when "generated_text" is in
    stages, i.e. the wrap.asr_sft path needs it).
    """
    import torch

    # Single-threaded torch to reduce peak allocator footprint. The forward
    # is memory-bound, not compute-bound, so this costs little wall-clock.
    torch.set_num_threads(1)

    _ensure_upstream_on_path()
    _stub_flash_attn()
    _patch_tokenizer_attention_for_no_flash_attn()

    asr_dir = Path(os.environ.get("MIMO_ASR_DIR", model_dir))
    tok_dir_env = os.environ.get("MIMO_TOKENIZER_DIR")
    if tok_dir_env:
        tok_dir = Path(tok_dir_env)
    else:
        # Try the same parent dir; fall back to MIMO_ASR_DIR if it bundles
        # the tokenizer (it does not for the official release, hence the
        # explicit env var preferred).
        tok_dir = Path(asr_dir).parent / "MiMo-Audio-Tokenizer"
        if not tok_dir.is_dir():
            tok_dir = asr_dir

    audio_tag = os.environ.get("MIMO_ASR_AUDIO_TAG", "<english>")
    max_new = int(os.environ.get("MIMO_ASR_MAX_NEW", max_new_tokens or 64))

    print(f"  loading MiMo-V2.5-ASR LM from {asr_dir}")
    print(f"  loading MiMo-Audio-Tokenizer from {tok_dir}")

    # The upstream MimoAudio still has to be importable for InputSegment —
    # we don't construct it (we use _LowMemMimoAudio defined above to keep
    # peak memory sane), but the dataclass-y InputSegment lives there.
    from src.mimo_audio.process_speechdata import InputSegment

    # Dtype: bf16 matches upstream and fits on 16 GB. fp32 doubles to ~28 GB
    # and only fits on a larger box. The C++ runtime is fp32, so bf16 ref
    # vs fp32 C++ will not bit-match — expect cos≥0.99 (not 0.999) on the
    # deeper LM stages.
    ref_dtype_str = os.environ.get("MIMO_ASR_REF_DTYPE", "bf16").lower()
    if ref_dtype_str in ("fp32", "float32", "float"):
        ref_dtype = torch.float32
    elif ref_dtype_str in ("bf16", "bfloat16"):
        ref_dtype = torch.bfloat16
    else:
        raise ValueError(f"MIMO_ASR_REF_DTYPE must be bf16 or fp32, got {ref_dtype_str!r}")
    print(f"  ref dtype: {ref_dtype} (override via MIMO_ASR_REF_DTYPE)")

    # Phase A: tokenizer + scalar metadata only (~few MB).
    wrap = _LowMemMimoAudio(
        model_path=str(asr_dir),
        mimo_audio_tokenizer_path=str(tok_dir),
        device="cpu",
    )

    out: Dict[str, np.ndarray] = {}

    # ---- 1. Tokenize audio → [T_a_padded, 8] codes ----
    # Phase B: load audio tokenizer (~3.6 GB), run preprocess, free.
    print("  phase B: loading audio tokenizer (~3.6 GB)")
    wrap.load_audio_tokenizer()
    if ref_dtype == torch.float32:
        wrap.mimo_audio_tokenizer = wrap.mimo_audio_tokenizer.float()
    wrap.mimo_audio_tokenizer.eval()

    audio_t = torch.tensor(audio, dtype=torch.float32)
    with torch.inference_mode():
        audio_tokenized = wrap.preprocess_input(audio_t)
    # audio_tokenized is [T_a_padded * 8] flat. Reshape for capture.
    n_total = int(audio_tokenized.shape[0])
    assert n_total % wrap.audio_channels == 0, \
        f"n_total {n_total} not divisible by audio_channels {wrap.audio_channels}"
    T_audio = n_total // wrap.audio_channels
    if "prefill_audio_codes" in stages:
        codes_t = audio_tokenized.view(-1, wrap.audio_channels).to(torch.int32)
        out["prefill_audio_codes"] = codes_t.numpy().astype(np.float32)

    # Free the audio tokenizer BEFORE loading the LM. This is the
    # critical peak-memory mitigation — keeps total resident under 16 GB.
    # MIMO_REF_KEEP_TOKENIZER=1 disables this (only useful when chasing
    # the generated_text stage's wrap.asr_sft path, which we don't run
    # here — see comment further down where generated_text is skipped
    # when the audio tokenizer was freed).
    keep_tok = os.environ.get("MIMO_REF_KEEP_TOKENIZER", "0") == "1"
    if not keep_tok:
        wrap.free_audio_tokenizer()
        print("  freed audio tokenizer (peak-memory mitigation)")

    # Phase C: load LM (~16 GB).
    print("  phase C: loading LM (~16 GB)")
    wrap.load_lm(dtype=ref_dtype)

    # ---- 2. Build the asr_sft prompt and capture input_ids ----
    # Mirror the upstream `get_asr_sft_prompt` exactly. The only difference:
    # we pin the template to `asr_en_templates[0]` for determinism (the
    # upstream picks one at random, which would change the input_ids every
    # run). The C++ harness uses the same fixed template.
    from src.mimo_audio.templates import asr_en_templates
    template_str = asr_en_templates[0]

    lm_prompt = [
        InputSegment(
            text="<|im_start|>user\n",
            speech_zeroemb_idx=wrap.speech_zeroemb_idx,
            text_zeroemb_idx=wrap.empty_token,
        ),
        InputSegment(
            audio=audio_tokenized,
            speech_zeroemb_idx=wrap.speech_zeroemb_idx,
            text_zeroemb_idx=wrap.empty_token,
        ),
        InputSegment(
            text=template_str,
            speech_zeroemb_idx=wrap.speech_zeroemb_idx,
            text_zeroemb_idx=wrap.empty_token,
        ),
        InputSegment(
            text="<|im_end|>\n",
            speech_zeroemb_idx=wrap.speech_zeroemb_idx,
            text_zeroemb_idx=wrap.empty_token,
        ),
        InputSegment(
            text="<|im_start|>assistant\n",
            speech_zeroemb_idx=wrap.speech_zeroemb_idx,
            text_zeroemb_idx=wrap.empty_token,
        ),
        InputSegment(
            text=f"<think>\n\n</think>\n{audio_tag}",
            speech_zeroemb_idx=wrap.speech_zeroemb_idx,
            text_zeroemb_idx=wrap.empty_token,
        ),
    ]
    input_ids = wrap.get_input_ids(lm_prompt)  # [9, T_total]
    if "prefill_input_ids" in stages:
        out["prefill_input_ids"] = input_ids.detach().cpu().numpy().astype(np.int32).astype(np.float32)
    print(f"  built asr_sft prompt input_ids shape={tuple(input_ids.shape)} "
          f"(audio T={T_audio}, group_size={wrap.group_size})")

    # The model's forward path expects input_ids reshaped via
    # `prepare_inputs_for_generation`. For prefill, it ends up as
    # [B, audio_channels+1=9, T_groups*group_size], which equals input_ids
    # transposed: [9, T_total] with T_total = T_groups * group_size.
    # We feed input_ids[None] which is [1, 9, T_total] and the wrapper
    # handles the reshape internally.
    # -> But for our diff dump we run the explicit pipeline:
    model = wrap.model
    B = 1
    input_ids_b = input_ids.unsqueeze(0).int()  # [1, 9, T_total]

    # ---- 3. Capture inputs_embeds and audio_features via hook on the
    # internal `_prepare_input_embeds` ----
    captures: Dict[str, np.ndarray] = {}

    # Re-implement _prepare_input_embeds inline so we can capture the
    # post-input_local_transformer / post-group_proj signal. Mirrors the
    # upstream code byte-for-byte. inference_mode is stronger than
    # no_grad — it disables autograd metadata too, which matters for
    # the 36-layer LM forward where every intermediate carries the
    # version-counter overhead.
    with torch.inference_mode():
        ids = input_ids_b.int()
        gs = model.config.group_size
        text_input_ids = ids[:, 0, ::gs]                            # [B, T_groups]
        speech_input_ids = (
            ids[:, 1:, :].view(B, model.audio_channels, -1, gs).transpose(1, 2)
        )                                                           # [B, T_groups, 8, gs]
        is_speech = text_input_ids == model.args.empty_idx          # [B, T_groups]

        # All intermediate activations live in `ref_dtype` so they match the
        # bf16/fp32 weights of the underlying nn.Linear modules. Capturing
        # always casts to fp32 at the end.
        speech_embeds = torch.zeros(
            (B, is_speech.shape[1], gs, model.input_local_config.hidden_size),
            dtype=ref_dtype,
        )
        for idx in range(model.audio_channels):
            cur_empty = model.speech_empty_ids[idx]
            cur_embed = model.speech_embeddings[idx]
            cur_speech_ids = speech_input_ids[:, :, idx, :]
            cur_speech_embeds = cur_embed(cur_speech_ids).to(ref_dtype)
            cur_mask = cur_speech_ids == cur_empty
            cur_speech_embeds.masked_fill_(cur_mask.unsqueeze(-1), 0.0)
            speech_embeds = speech_embeds + cur_speech_embeds

        speech_embeds = speech_embeds * is_speech.unsqueeze(-1).unsqueeze(-1)
        # Run the input_local_transformer.
        speech_embeds = model.apply_input_local_transformer(speech_embeds.to(ref_dtype))
        speech_embeds = speech_embeds * is_speech.unsqueeze(-1).unsqueeze(-1)

        T_groups = speech_embeds.shape[1]
        speech_grouped = model.speech_group_downcast(
            speech_embeds.view(B, T_groups, -1).to(ref_dtype)
        )  # [B, T_groups, hidden=4096]

        if "prefill_audio_features" in stages:
            captures["prefill_audio_features"] = (
                speech_grouped[0].detach().cpu().float().numpy())
        # Drop the [B, T_groups, gs, 1024] speech_embeds — we already
        # downcast it into speech_grouped, no later stage uses it.
        del speech_embeds
        gc.collect()

        text_embeds = model.model.embed_tokens(text_input_ids).to(ref_dtype)
        text_zero_mask = text_input_ids == model.args.empty_idx
        text_embeds = text_embeds.masked_fill(text_zero_mask.unsqueeze(-1), 0.0)

        if "prefill_text_embeds" in stages:
            captures["prefill_text_embeds"] = (
                text_embeds[0].detach().cpu().float().numpy())

        inputs_embeds = text_embeds + speech_grouped  # [B, T_groups, hidden]
        del text_embeds, speech_grouped
        gc.collect()

        if "prefill_inputs_embeds" in stages:
            captures["prefill_inputs_embeds"] = (
                inputs_embeds[0].detach().cpu().float().numpy())

        # ---- 4. Run the 36L Qwen2 LM forward + lm_head ----
        # Use the lower-level model (without generate's KV cache mechanics)
        # so we get the full hidden state sequence.
        from transformers.modeling_outputs import BaseModelOutputWithPast
        attention_mask = torch.ones((B, T_groups), dtype=torch.bool)
        position_ids = torch.arange(T_groups).unsqueeze(0)
        lm_out: BaseModelOutputWithPast = model.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            return_dict=True,
        )
        hidden = lm_out.last_hidden_state  # [B, T_groups, hidden]
        last_hidden = hidden[:, -1:, :]    # [B, 1, hidden]
        # Drop the BaseModelOutputWithPast and the upstream activations —
        # last_hidden is a view into `hidden` so we keep that alive too.
        del lm_out, inputs_embeds
        gc.collect()

        if "prefill_last_hidden" in stages:
            captures["prefill_last_hidden"] = (
                last_hidden[0, 0].detach().cpu().float().numpy())

        text_logits = model.lm_head(last_hidden)  # [B, 1, vocab]
        if "prefill_text_logits_step0" in stages:
            captures["prefill_text_logits_step0"] = (
                text_logits[0, 0].detach().cpu().float().numpy())
        if "prefill_text_argmax_step0" in stages:
            argmax = int(torch.argmax(text_logits[0, 0]).item())
            captures["prefill_text_argmax_step0"] = (
                np.array([argmax], dtype=np.int32).astype(np.float32))
            print(f"  text_logits argmax_step0 = {argmax} "
                  f"(token={wrap.tokenizer.decode([argmax]) if argmax < wrap.vocab_size else '<oob>'!r})")

    out.update(captures)

    # ---- 5. End-to-end transcript via wrapper.asr_sft (greedy) ----
    # _LowMemMimoAudio is a tokenizer-only / model-only shim; it does NOT
    # implement asr_sft (which would require both the audio tokenizer AND
    # the LM resident, defeating the peak-memory mitigation). The C++
    # runtime's transcribe path covers this end-to-end path independently
    # — JFK transcript byte-equality against the gold quote is the gate.
    if "generated_text" in stages:
        out["generated_text"] = (
            "<skipped: _LowMemMimoAudio shim doesn't implement asr_sft; "
            "use the C++ runtime for end-to-end transcript validation>"
        )

    return out
