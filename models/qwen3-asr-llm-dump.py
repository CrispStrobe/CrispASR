#!/usr/bin/env python3
"""Dump Qwen3-ASR LLM (text-only) ground-truth logits for diff testing.

Feeds a fixed token sequence into the thinker text model and the lm_head,
captures intermediate activations and final logits.

Outputs:
  llm_input_ids.npy        (T,) int32        input token IDs
  llm_post_blk00.npy       (T, 1024) f32     hidden state after layer 0
  llm_post_blk27.npy       (T, 1024) f32     hidden state after last layer
  llm_post_norm.npy        (T, 1024) f32     after model.norm
  llm_logits.npy           (T, 151936) f32   final logits
  llm_topk.npy             (T, 5) int32      top-5 token ids per position
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--out-dir",   required=True, type=Path)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from qwen_asr import Qwen3ASRModel

    print(f"Loading {args.model_dir} ...")
    wrapper = Qwen3ASRModel.from_pretrained(args.model_dir, dtype="float32", device_map="cpu")
    model = wrapper.model
    model.eval()
    thinker = model.thinker
    text_model = thinker.model
    lm_head = thinker.lm_head

    # Fixed input — short prompt that exercises ChatML special tokens.
    # Tokens chosen by tokenization, dumped here for reproducibility.
    tok = wrapper.processor.tokenizer
    text = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
    print(f"input text: {text!r}")
    print(f"input ids:  {ids[0].tolist()}")
    T = ids.shape[1]
    np.save(args.out_dir / "llm_input_ids.npy", ids[0].numpy().astype(np.int32))

    # Hooks
    captures = {}
    def cap(name):
        def hook(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            captures[name] = t.detach().cpu().float().numpy()
        return hook
    text_model.layers[0].register_forward_hook(cap("llm_post_blk00"))
    text_model.layers[-1].register_forward_hook(cap("llm_post_blk27"))
    text_model.norm.register_forward_hook(cap("llm_post_norm"))

    # Forward — no audio, no KV cache
    with torch.no_grad():
        out = text_model(input_ids=ids, use_cache=False)
        h = out.last_hidden_state  # (1, T, 1024)
        logits = lm_head(h)  # (1, T, 151936)
    print(f"hidden: {tuple(h.shape)}, logits: {tuple(logits.shape)}")

    captures["llm_logits"] = logits[0].detach().cpu().float().numpy()
    for k, v in captures.items():
        # Squeeze leading batch dim if present
        if v.ndim == 3 and v.shape[0] == 1:
            v = v[0]
        np.save(args.out_dir / f"{k}.npy", v)
        print(f"  {k}: {v.shape}")

    # Top-5 per position for sanity
    topk = np.argsort(-captures["llm_logits"], axis=-1)[:, :5].astype(np.int32)
    np.save(args.out_dir / "llm_topk.npy", topk)
    print(f"  llm_topk: {topk.shape}")
    for i, row in enumerate(topk):
        toks = [tok.decode([int(t)]) for t in row]
        print(f"  pos {i}: {toks}")

    print(f"\nDone: {args.out_dir}")


if __name__ == "__main__":
    main()
