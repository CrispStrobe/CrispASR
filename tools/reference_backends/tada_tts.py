"""TADA-3B-ML TTS reference dump backend for crispasr-diff.

Captures per-stage activations from the HumeAI/tada-3b-ml TTS model
so the C++ implementation can be validated tensor-by-tensor.

TADA architecture:
  1. Llama-3.2-3B backbone with added acoustic/time embeddings
  2. Per-token flow-matching (VibeVoiceDiffusionHead) ODE solver
  3. TADA codec decoder (LocalAttentionEncoder + DACDecoder) → 24 kHz PCM

Pipeline stages captured:
  text_tokens          — BPE token IDs (int32 as float32)
  llm_embed            — input embeddings at step 0 (before LLM)
  llm_hidden_0         — LLM hidden state at step 0 (after forward)
  fm_noise_0           — initial noise for FM solver at step 0
  fm_step_0_0          — FM state after ODE step 0 at token 0
  fm_step_0_5          — FM state after ODE step 5 at token 0
  fm_output_0          — FM output (speech vector) at token 0
  acoustic_features    — all generated acoustic features (N, 512)
  time_before          — all time_before values (N,) as float32
  codec_input          — expanded features going into codec decoder
  codec_proj           — after decoder linear projection
  codec_attn_out       — after local attention encoder
  codec_pcm            — final 24 kHz waveform

Usage:
    python tools/dump_reference.py --backend tada-tts \\
        --model-dir HumeAI/tada-3b-ml \\
        --audio samples/jfk.wav \\
        --output /mnt/storage/tada-3b-ml/tada-ref.gguf \\
        --stages text_tokens,fm_output_0,acoustic_features,codec_pcm

    Environment variables:
      TADA_SYN_TEXT       — text to synthesize (default: "Hello world.")
      TADA_NUM_FM_STEPS   — flow matching ODE steps (default: 10)
      TADA_CFG_SCALE      — acoustic CFG scale (default: 1.0 = no CFG, deterministic)
      TADA_NOISE_TEMP     — noise temperature (default: 0.0 = zero noise, deterministic)
      TADA_SEED           — random seed (default: 42)
      TADA_DEVICE         — "cpu" or "cuda" (default: "cpu")
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "text_tokens",
    "llm_embed",
    "llm_hidden_0",
    "fm_noise_0",
    "fm_step_0_0",
    "fm_step_0_5",
    "fm_output_0",
    "acoustic_features",
    "time_before",
    "codec_input",
    "codec_proj",
    "codec_attn_out",
    "codec_pcm",
]

DEFAULT_SYN_TEXT = "Hello world."
DEFAULT_NUM_FM_STEPS = 10
DEFAULT_CFG_SCALE = 1.0       # no CFG → deterministic velocity
DEFAULT_NOISE_TEMP = 0.0      # zero noise → deterministic init
DEFAULT_SEED = 42


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int = 64) -> Dict[str, np.ndarray]:
    import torch
    torch.set_grad_enabled(False)

    out: Dict[str, np.ndarray] = {}

    device = os.environ.get("TADA_DEVICE", "cpu")
    syn_text = os.environ.get("TADA_SYN_TEXT", DEFAULT_SYN_TEXT)
    num_fm_steps = int(os.environ.get("TADA_NUM_FM_STEPS", DEFAULT_NUM_FM_STEPS))
    cfg_scale = float(os.environ.get("TADA_CFG_SCALE", DEFAULT_CFG_SCALE))
    noise_temp = float(os.environ.get("TADA_NOISE_TEMP", DEFAULT_NOISE_TEMP))
    seed = int(os.environ.get("TADA_SEED", DEFAULT_SEED))

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── Load model ──
    print(f"  loading TADA from {model_dir}")
    from transformers import AutoTokenizer, LlamaForCausalLM
    from tada.modules.tada import TadaForCausalLM, InferenceOptions, TadaConfig
    from tada.modules.decoder import Decoder
    from tada.modules.encoder import EncoderOutput
    from tada.utils.gray_code import decode_gray_code_to_time

    # Monkey-patch from_pretrained to use the local tokenizer.json
    # (avoids gated-repo 401 on meta-llama/Llama-3.2-1B)
    _orig_from_pretrained = TadaForCausalLM.from_pretrained.__func__

    @classmethod
    def _patched_from_pretrained(cls, path, *args, **kwargs):
        self = LlamaForCausalLM.from_pretrained(path, *args, **kwargs)
        self.__class__ = cls
        cls.__init__(self, self.config)
        # Load decoder from codec dir if available, else from HF
        codec_dir = os.environ.get("TADA_CODEC_DIR")
        if codec_dir:
            self._decoder = Decoder.from_pretrained(codec_dir, subfolder="decoder")
        else:
            self._decoder = Decoder.from_pretrained("HumeAI/tada-codec", subfolder="decoder")
        # Load tokenizer from the model dir itself (has tokenizer.json)
        self._tokenizer = AutoTokenizer.from_pretrained(str(path))
        return self

    TadaForCausalLM.from_pretrained = _patched_from_pretrained

    model = TadaForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    tokenizer = model.tokenizer
    decoder = model.decoder

    # ── Config params ──
    acoustic_dim = model.config.acoustic_dim     # 512
    num_time_bits = model.num_time_bits           # 10
    time_dim = model.time_dim                     # 20
    shift_acoustic = model.config.shift_acoustic  # 5
    total_dim = acoustic_dim + time_dim           # 532

    print(f"  acoustic_dim={acoustic_dim}, time_dim={time_dim}, "
          f"shift_acoustic={shift_acoustic}")
    print(f"  syn_text={syn_text!r}, num_fm_steps={num_fm_steps}, "
          f"cfg_scale={cfg_scale}, noise_temp={noise_temp}, seed={seed}")

    # ── Tokenize ──
    # Build input_ids the same way model.generate() does:
    # text_tokens = tokenizer.encode(text, add_special_tokens=False)
    # Then wrap with BOS + prefix + EOS tokens
    text_tokens_raw = tokenizer.encode(syn_text, add_special_tokens=False)

    if "text_tokens" in stages:
        out["text_tokens"] = np.array(text_tokens_raw, dtype=np.float32)

    # ── Build empty prompt (no voice cloning) ──
    prompt = EncoderOutput.empty(device=torch.device(device),
                                 token_dim=acoustic_dim)
    prompt.text = [syn_text]

    # ── Inference options: deterministic for reproducibility ──
    opts = InferenceOptions(
        text_do_sample=False,           # greedy decoding
        acoustic_cfg_scale=cfg_scale,   # 1.0 = no CFG
        duration_cfg_scale=1.0,
        noise_temperature=noise_temp,   # 0.0 = zero initial noise
        num_flow_matching_steps=num_fm_steps,
        num_acoustic_candidates=1,
        cfg_schedule="constant",
        time_schedule="uniform",        # simplest for validation
    )

    # ── Manual step-by-step generation with intermediate capture ──
    # We replicate the _generate() logic to capture intermediates.
    # For the reference dump we do a simple no-prompt generation.

    # Build input_ids like model.generate() does
    text_token_ids = tokenizer.encode(syn_text, add_special_tokens=False)
    input_ids = torch.tensor([text_token_ids], device=device, dtype=torch.long)

    # Add BOS + prefix + EOS
    bos = tokenizer.bos_token_id
    eos = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    prefix = "<|start_header_id|>assistant<|end_header_id|>"
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)

    # Full sequence: [BOS] + prefix + text + [EOS]*shift_acoustic
    full_ids = [bos] + prefix_ids + text_token_ids + [eos] * shift_acoustic
    input_ids = torch.tensor([full_ids], device=device, dtype=torch.long)
    num_tokens = input_ids.shape[1]

    print(f"  input_ids shape: {input_ids.shape}")
    print(f"  tokens: {tokenizer.decode(input_ids[0])}")

    # ── Step-by-step AR + FM loop ──
    # Initialize state
    acoustic_features_list = []
    time_before_list = []

    acoustic_feat = torch.zeros(1, 1, acoustic_dim, device=device,
                                dtype=torch.bfloat16)
    acoustic_mask = torch.zeros(1, 1, device=device, dtype=torch.long)
    time_before_val = torch.zeros(1, 1, device=device, dtype=torch.long)
    time_after_val = torch.zeros(1, 1, device=device, dtype=torch.long)

    # Prepare generation
    from transformers.generation.configuration_utils import GenerationConfig
    gen_config = GenerationConfig(
        bos_token_id=bos,
        eos_token_id=eos,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen_config, model_kwargs = model._prepare_generation_config(gen_config, True)
    model._prepare_cache_for_generation(gen_config, model_kwargs, None, 1, num_tokens + max_new_tokens)
    model_kwargs["cache_position"] = torch.arange(1, device=device, dtype=torch.long)

    # Capture intermediates for first generated token
    captured_fm_steps = {}

    for step in range(num_tokens + max_new_tokens):
        if step < num_tokens:
            input_slice = input_ids[:, step:step+1]
        else:
            input_slice = input_ids[:, -1:]

        # ── LLM forward step ──
        model_inputs = model.prepare_inputs_for_generation(input_slice, **model_kwargs)
        outputs = model.forward_one_step(
            **model_inputs,
            acoustic_features=acoustic_feat,
            acoustic_masks=acoustic_mask,
            time_len_before=time_before_val,
            time_len_after=time_after_val,
            compute_logits=(step >= num_tokens - 1),
        )

        hidden_states = outputs.hidden_states[-1]  # (1, 1, hidden_size)

        # Capture LLM intermediates at first step
        is_first_gen = (step == 0)
        if is_first_gen:
            if "llm_embed" in stages:
                # Re-compute embedding for capture
                emb = (model.model.embed_tokens(input_slice)
                       + model.acoustic_proj(acoustic_feat)
                       + model.acoustic_mask_emb(acoustic_mask)
                       + model.time_start_embed(time_before_val)
                       + model.time_end_embed(time_after_val))
                out["llm_embed"] = emb[0].cpu().float().numpy()

            if "llm_hidden_0" in stages:
                out["llm_hidden_0"] = hidden_states[0].cpu().float().numpy()

        # ── Flow matching solver (per token) ──
        cond = hidden_states  # (1, 1, hidden_size)

        # Generate initial noise
        torch.manual_seed(seed + step)  # deterministic per-step
        speech = torch.randn(1, total_dim, device=device,
                             dtype=torch.bfloat16) * noise_temp

        if is_first_gen and "fm_noise_0" in stages:
            out["fm_noise_0"] = speech[0].cpu().float().numpy()

        # Euler ODE solve
        neg_cond = torch.zeros_like(cond[:, 0, :])  # (1, hidden)
        t_span = torch.linspace(0, 1, num_fm_steps + 1, device=device)

        for i in range(1, len(t_span)):
            dt = t_span[i] - t_span[i-1]
            t_curr = t_span[i-1]

            velocity = model._compute_velocity(
                speech, t_curr, cond,
                neg_cond.unsqueeze(1),
                cfg_scale, 1.0  # duration_cfg = 1.0
            )
            speech = speech + dt * velocity

            # Capture FM steps for first generated token
            if is_first_gen:
                fm_name = f"fm_step_0_{i-1}"
                if fm_name in stages:
                    captured_fm_steps[fm_name] = speech[0].cpu().float().numpy()

        # FM output
        if is_first_gen and "fm_output_0" in stages:
            out["fm_output_0"] = speech[0].cpu().float().numpy()

        # Extract time from gray code
        time_gray = speech[..., -time_dim:]
        pred_time_before = decode_gray_code_to_time(
            time_gray[..., :num_time_bits], num_time_bits
        )
        pred_time_after = decode_gray_code_to_time(
            time_gray[..., num_time_bits:], num_time_bits
        )

        # Next-token prediction (greedy)
        if step >= num_tokens - 1:
            logits = outputs.logits[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token.long()], dim=1)

            # Stop on EOS
            if next_token.item() == eos:
                print(f"  EOS at step {step}")
                break

        # Update state for next step
        if step >= shift_acoustic:
            acoustic_feat = speech[:, :acoustic_dim].unsqueeze(1)
            acoustic_mask = torch.ones(1, 1, device=device, dtype=torch.long)
            acoustic_features_list.append(
                speech[0, :acoustic_dim].cpu().float().numpy()
            )
            time_before_list.append(pred_time_before[0].item())
        else:
            acoustic_feat = torch.zeros(1, 1, acoustic_dim, device=device,
                                        dtype=torch.bfloat16)
            acoustic_mask = torch.zeros(1, 1, device=device, dtype=torch.long)

        time_before_val = pred_time_before.unsqueeze(0)
        time_after_val = pred_time_after.unsqueeze(0)

        model_kwargs = model._update_model_kwargs_for_generation(outputs, model_kwargs)

    # Store captured FM steps
    out.update(captured_fm_steps)

    # ── Aggregate acoustic features ──
    if acoustic_features_list:
        all_feats = np.stack(acoustic_features_list, axis=0)
        if "acoustic_features" in stages:
            out["acoustic_features"] = all_feats  # (N, 512)

        all_times = np.array(time_before_list, dtype=np.float32)
        if "time_before" in stages:
            out["time_before"] = all_times  # (N,)

        # ── Codec decoder ──
        # Denormalize: features * std + mean
        acoustic_std = model.config.acoustic_std   # 1.5
        acoustic_mean = model.config.acoustic_mean  # 0.0
        feats_denorm = torch.from_numpy(all_feats).to(device).to(torch.bfloat16)
        feats_denorm = feats_denorm * acoustic_std + acoustic_mean

        # Expand with time_before durations (same as model._decode_wav)
        time_arr = torch.tensor([0] + time_before_list, dtype=torch.long,
                                device=device)
        encoded_expanded = []
        for pos in range(feats_denorm.shape[0]):
            # Insert (time_before - 1) zero frames before each acoustic vector
            n_zeros = max(0, int(time_arr[pos].item()) - 1)
            if n_zeros > 0:
                encoded_expanded.append(
                    torch.zeros(n_zeros, acoustic_dim, device=device,
                                dtype=torch.bfloat16)
                )
            encoded_expanded.append(feats_denorm[pos:pos+1])

        # Trailing zeros from last time value
        if len(time_arr) > feats_denorm.shape[0]:
            n_trail = int(time_arr[-1].item())
            if n_trail > 0:
                encoded_expanded.append(
                    torch.zeros(n_trail, acoustic_dim, device=device,
                                dtype=torch.bfloat16)
                )

        if encoded_expanded:
            expanded = torch.cat(encoded_expanded, dim=0).unsqueeze(0)
            token_masks = (torch.norm(expanded, dim=-1) != 0).long()

            if "codec_input" in stages:
                out["codec_input"] = expanded[0].cpu().float().numpy()

            # Run codec decoder
            from . import _hooks
            captured = {}
            handles = _hooks.capture_modules(captured, [
                ("codec_proj", decoder.decoder_proj),
                ("codec_attn_out", decoder.local_attention_decoder),
            ])

            pcm = decoder.generate(
                expanded.float(),
                token_masks=token_masks,
            )

            _hooks.drop_hooks(handles)

            if "codec_proj" in stages and "codec_proj" in captured:
                out["codec_proj"] = captured["codec_proj"][0].numpy()
            if "codec_attn_out" in stages and "codec_attn_out" in captured:
                out["codec_attn_out"] = captured["codec_attn_out"][0].numpy()

            if "codec_pcm" in stages:
                out["codec_pcm"] = pcm[0, 0].cpu().float().numpy()

    generated_text = tokenizer.decode(input_ids[0, num_tokens:], skip_special_tokens=True)
    print(f"  generated text: {generated_text!r}")
    print(f"  {len(acoustic_features_list)} acoustic frames generated")
    if acoustic_features_list:
        print(f"  total time_before frames: {sum(time_before_list)}")
        if "codec_pcm" in out:
            pcm_samples = out["codec_pcm"].shape[0]
            print(f"  PCM samples: {pcm_samples} ({pcm_samples/24000:.2f}s @ 24kHz)")

    return out
