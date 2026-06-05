"""Parler TTS reference dump backend — T5 encoder + MusicGen decoder + DAC.

Captures per-stage activations from the HuggingFace parler-tts model for
diff-testing the C++ port.  Text inputs come from env vars:

  PARLER_TEXT       default: "Hello, this is a test of Parler TTS."
  PARLER_DESC       default: "A female speaker with a warm, natural voice
                     delivers her words at a moderate pace in a quiet
                     environment."
  PARLER_SEED       default: 42

The audio arg from tools/dump_reference.py is unused (Parler is text-driven).

Stages dumped (subset selectable via --stages):

  description_ids     — T5 encoder input token IDs cast to F32
  prompt_ids          — decoder prompt token IDs cast to F32
  t5_encoder_output   — T5 encoder final hidden state (T_desc, D)
  prefill_input       — decoder prefill input hidden (T_prefill, D)
                        = cat(prompt_embed, bos_cb_sum) + pos_embed
  prefill_kv_k_0      — self-attn K cache layer 0 after prefill (D, T)
                        in ggml (D-fast) layout
  prefill_kv_k_23     — same for layer 23
  step1_input         — incremental step 1 input hidden (D,)
  step1_logits_cb0    — step 1 codebook-0 logits (vocab_size,)
  gen_codes_20        — first 20 greedy-decoded code steps (9, 20) as F32
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "description_ids",
    "prompt_ids",
    "t5_encoder_output",
    "prefill_input",
    "prefill_kv_k_0",
    "prefill_kv_k_23",
    "step1_input",
    "step1_logits_cb0",
    "gen_codes_20",
]


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Run Parler TTS forward, return captured stage tensors keyed by name."""
    _ = audio  # unused — Parler is text-driven
    import torch

    os.environ.setdefault("HF_HOME", "/mnt/storage/huggingface")

    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    text = os.environ.get("PARLER_TEXT",
                          "Hello, this is a test of Parler TTS.")
    desc = os.environ.get("PARLER_DESC",
                          "A female speaker with a warm, natural voice delivers "
                          "her words at a moderate pace in a quiet environment.")
    seed = int(os.environ.get("PARLER_SEED", "42"))
    n_steps = min(max_new_tokens if max_new_tokens > 0 else 20, 20)

    print(f"parler_tts ref: loading {model_dir}", file=sys.stderr)
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        str(model_dir), torch_dtype=torch.float32).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    decoder = model.decoder

    desc_ids = tokenizer(desc, return_tensors="pt").input_ids
    prompt_ids = tokenizer(text, return_tensors="pt").input_ids
    num_cb = decoder.config.num_codebooks
    bos = decoder.config.bos_token_id
    D = decoder.config.hidden_size

    captures: Dict[str, Any] = {}

    if "description_ids" in stages:
        captures["description_ids"] = desc_ids[0].numpy().astype(np.float32)
    if "prompt_ids" in stages:
        captures["prompt_ids"] = prompt_ids[0].numpy().astype(np.float32)

    with torch.no_grad():
        # ── T5 encoder ──
        enc_out = model.text_encoder(input_ids=desc_ids).last_hidden_state
        if "t5_encoder_output" in stages:
            captures["t5_encoder_output"] = enc_out[0].numpy()

        # ── Prefill input construction (mirrors C++) ──
        prompt_embed = model.embed_prompts(prompt_ids)     # (1, T_prompt, D)
        bos_ids_3d = torch.full((1, num_cb, 1), bos, dtype=torch.long)
        cb_sum = sum([decoder.model.decoder.embed_tokens[k](bos_ids_3d[:, k])
                      for k in range(num_cb)])             # (1, 1, D)
        full_embed = torch.cat([prompt_embed, cb_sum], dim=1)  # (1, T_pf, D)
        T_pf = full_embed.shape[1]
        pos_w = decoder.model.decoder.embed_positions.weights[:T_pf]
        prefill_hidden = full_embed[0] + pos_w             # (T_pf, D)

        if "prefill_input" in stages:
            captures["prefill_input"] = prefill_hidden.numpy()

        # ── Decoder prefill (get KV cache + first logits) ──
        bos_ids = torch.full((num_cb, 1), bos, dtype=torch.long)
        result = decoder(
            input_ids=bos_ids,
            encoder_hidden_states=enc_out,
            prompt_hidden_states=prompt_embed,
            use_cache=True,
        )
        past_kv = result.past_key_values

        # Capture KV cache — convert to ggml (D, T) layout
        # past_kv format: EncoderDecoderCache or legacy tuple
        # For legacy tuple: past_kv[layer] = ((self_k, self_v), (cross_k, cross_v))
        # For EncoderDecoderCache: .self_attention_cache.key_cache[layer]
        for li in [0, 23]:
            stage = f"prefill_kv_k_{li}"
            if stage in stages:
                # Legacy tuple: past_kv[layer] = (self_k, self_v, cross_k, cross_v)
                # self_k shape: (batch, num_heads, T, head_dim)
                k = past_kv[li][0][0]  # (nh, T, hd)
                # To ggml (D, T): permute to (nh, hd, T) then reshape
                k_flat = k.permute(0, 2, 1).reshape(-1, k.shape[1])  # (D, T)
                captures[stage] = k_flat.numpy()

        # First token (greedy)
        first_logits = result.logits[:, -1, :]         # (num_cb, vocab)
        first_tokens = first_logits.argmax(dim=-1).tolist()
        for k in range(1, num_cb):
            first_tokens[k] = bos
        all_tokens = [first_tokens]

        print(f"parler_tts ref: step 0 tokens={first_tokens}", file=sys.stderr)

        # ── Incremental decode ──
        torch.manual_seed(seed)
        for step in range(1, n_steps):
            cur_ids = torch.tensor(all_tokens[-1], dtype=torch.long).unsqueeze(1)
            out = decoder(
                input_ids=cur_ids,
                encoder_hidden_states=enc_out,
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv = out.past_key_values
            next_logits = out.logits[:, -1, :]
            next_tokens = next_logits.argmax(dim=-1).tolist()
            for k in range(num_cb):
                if step < k:
                    next_tokens[k] = bos

            if step == 1:
                if "step1_logits_cb0" in stages:
                    captures["step1_logits_cb0"] = next_logits[0].numpy()
                # Build the step-1 input for comparison
                if "step1_input" in stages:
                    prev = torch.tensor(all_tokens[-1], dtype=torch.long)
                    inp_3d = prev.reshape(1, num_cb, 1)
                    emb_sum = sum([decoder.model.decoder.embed_tokens[k](inp_3d[:, k])
                                   for k in range(num_cb)])  # (1, 1, D)
                    pos = decoder.model.decoder.embed_positions.weights[T_pf]
                    step1_inp = emb_sum[0, 0] + pos
                    captures["step1_input"] = step1_inp.numpy()

            print(f"parler_tts ref: step {step} tokens={next_tokens}", file=sys.stderr)
            all_tokens.append(next_tokens)

        if "gen_codes_20" in stages:
            n_s = len(all_tokens)
            T_audio = max(0, n_s - num_cb + 1)
            aligned = []
            for t in range(T_audio):
                frame = []
                for k in range(num_cb):
                    step = t + k
                    tok = all_tokens[step][k] if step < n_s else 1024
                    if tok == bos:
                        tok = 1024
                    frame.append(tok)
                aligned.append(frame)
            captures["gen_codes_20"] = np.array(aligned, dtype=np.float32)

    # Store metadata as string captures (dump_reference.py handles these)
    captures["parler_text"] = text
    captures["parler_desc"] = desc
    captures["parler_seed"] = str(seed)

    return captures
