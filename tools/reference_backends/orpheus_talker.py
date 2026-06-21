"""Orpheus talker (Llama-3.2-3B-FT) reference dump backend.

Captures the greedy codec-token stream the Orpheus talker LM emits for a
given prompt, so `crispasr-diff orpheus-talker` can verify the C++ AR
decode (the §176b Lk-bucketed graph + device KV) against PyTorch ground
truth — the part the SNAC-only `orpheus` diff does not cover.

The prompt is built verbatim from canopylabs/Orpheus-TTS:_format_prompt:

    [128259]  +  tokenizer("{speaker}: {text}")  +  [128009,128260,128261,128257]

where the HF tokenizer prepends the Llama-3 BOS (128000). The full prompt
IDs are dumped so the C++ side uses them via ORPHEUS_PROMPT_IDS (no BPE
mismatch). Greedy decoding then runs N steps; tokens in the codec block
[custom_token_offset, +count) are dumped as `gen_codes` — the same raw IDs
`orpheus_synthesize_codes` returns.

Inputs via env:
    ORPHEUS_TEXT     (default "Hey there, my name is Tara.")
    ORPHEUS_SPEAKER  (default "tara")

Stages:
    prompt_ids   — (T_prompt,)  int32 as float32, full talker prompt
    gen_codes    — (N_codes,)   int32 as float32, greedy codec-block tokens
    step0_logits — (vocab,)     float32, logits at the first decode position
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Set

import numpy as np

DEFAULT_STAGES = [
    "prompt_ids",
    "gen_codes",
    "step0_logits",
]

# Orpheus / canopylabs control tokens (fixed).
AUDIO_START = 128259
EOT = 128009
AUDIO_EOT = 128260
AUDIO_EOM = 128261
AUDIO_END = 128257
# Codec block range — MUST match the C++ GGUF's orpheus.custom_token_offset /
# .custom_token_count (model-specific; e.g. canonical 128266/28672, kartoffel
# 128256/28683). Set from the GGUF metadata when generating the reference.
CUSTOM_TOKEN_OFFSET = int(os.environ.get("ORPHEUS_CUSTOM_OFFSET", "128266"))
CUSTOM_TOKEN_COUNT = int(os.environ.get("ORPHEUS_CUSTOM_COUNT", str(4096 * 7)))


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    _ = audio  # text-driven
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    text = os.environ.get("ORPHEUS_TEXT", "Hey there, my name is Tara.")
    speaker = os.environ.get("ORPHEUS_SPEAKER", "tara")
    n_steps = max_new_tokens if max_new_tokens > 0 else 64

    print(f"orpheus_talker ref: loading {model_dir}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    # 3B model: float32 (~12 GB) risks OOM on a 16 GB box; bf16 (~6 GB) is
    # ample ground truth for a greedy-argmax token comparison (cf. F16 GGUF).
    dtype = getattr(torch, os.environ.get("ORPHEUS_REF_DTYPE", "bfloat16"))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=dtype, low_cpu_mem_usage=True).eval()

    body = tok(f"{speaker}: {text}", return_tensors="pt").input_ids[0]  # incl. BOS 128000
    prompt = ([AUDIO_START] + body.tolist() + [EOT, AUDIO_EOT, AUDIO_EOM, AUDIO_END])
    prompt_ids = torch.tensor([prompt], dtype=torch.long)

    captures: Dict[str, Any] = {}
    if "prompt_ids" in stages:
        captures["prompt_ids"] = prompt_ids[0].numpy().astype(np.float32)

    gen_codes = []
    with torch.no_grad():
        out = model(input_ids=prompt_ids, use_cache=True)
        logits = out.logits[:, -1, :]
        past = out.past_key_values
        if "step0_logits" in stages:
            captures["step0_logits"] = logits[0].float().numpy()
        for step in range(n_steps):
            nxt = int(torch.argmax(logits[0]).item())  # greedy
            if nxt == AUDIO_END:
                break
            if CUSTOM_TOKEN_OFFSET <= nxt < CUSTOM_TOKEN_OFFSET + CUSTOM_TOKEN_COUNT:
                gen_codes.append(nxt)
            step_in = torch.tensor([[nxt]], dtype=torch.long)
            out = model(input_ids=step_in, past_key_values=past, use_cache=True)
            logits = out.logits[:, -1, :]
            past = out.past_key_values
        print(f"orpheus_talker ref: {len(gen_codes)} codec tokens in {step+1} steps",
              file=sys.stderr)

    if "gen_codes" in stages:
        captures["gen_codes"] = np.asarray(gen_codes, dtype=np.float32)

    captures["orpheus_text"] = text
    captures["orpheus_speaker"] = speaker
    return captures
