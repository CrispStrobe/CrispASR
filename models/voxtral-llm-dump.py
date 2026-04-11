#!/usr/bin/env python3

# LEGACY: kept to feed the examples/{qwen3,voxtral}-test-*/main.cpp
# differential-test drivers which still consume .npy files with the
# specific names this script produces. New backend ports should use
# tools/dump_reference.py + tools/reference_backends/<name>.py, which
# write a single GGUF tensor archive that crispasr-diff can load via
# core_gguf::load_weights. See ARCHITECTURE.md for the overall picture.

"""Dump Voxtral LLM (text-only, no audio injection) ground-truth logits.

Feeds a fixed token sequence into the language_model branch and the lm_head,
captures intermediate activations and final logits. Used as the reference for
the C++ voxtral LLM forward diff test.

Outputs:
  voxtral_input_ids.npy        (T,) int32        input token IDs
  voxtral_post_blk00.npy       (T, 3072) f32     hidden state after layer 0
  voxtral_post_blk29.npy       (T, 3072) f32     hidden state after last layer
  voxtral_post_norm.npy        (T, 3072) f32     after model.norm
  voxtral_logits.npy           (T, 131072) f32   final logits
  voxtral_topk.npy             (T, 5) int32      top-5 token ids per position
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

    print(f"Loading {args.model_dir} ...")
    from transformers import VoxtralForConditionalGeneration, AutoProcessor
    # bfloat16 — F32 OOMs on a 7.6 GB host. The C++ side will diff against
    # bf16 reference with ~1e-3 tolerance which is fine since our GGUF is F16.
    model = VoxtralForConditionalGeneration.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_dir)
    text_model = model.language_model
    lm_head = (model.language_model.lm_head if hasattr(model.language_model, "lm_head")
               else model.lm_head)
    text_inner = text_model.model if hasattr(text_model, "model") else text_model
    print(f"  text_model: {type(text_model).__name__}")
    print(f"  text_inner: {type(text_inner).__name__}")

    # Use the proper Voxtral chat template — the README shows that text-only
    # prompts go through processor.apply_chat_template, NOT raw [INST] strings.
    # Voxtral wraps the message in extra control tokens the model was trained on.
    conversation = [{
        "role": "user",
        "content": [{"type": "text", "text": "Why should AI models be open-sourced?"}],
    }]
    inputs = processor.apply_chat_template(conversation)
    ids = inputs.input_ids  # (1, T)
    print(f"input ids ({ids.shape[1]}):  {ids[0].tolist()[:20]}{'...' if ids.shape[1] > 20 else ''}")
    np.save(args.out_dir / "voxtral_input_ids.npy", ids[0].numpy().astype(np.int32))

    # Hooks
    captures = {}
    def cap(name):
        def hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            captures[name] = t.detach().cpu().float().numpy()
        return hook
    text_inner.layers[0].register_forward_hook(cap("voxtral_post_blk00"))
    text_inner.layers[-1].register_forward_hook(cap("voxtral_post_blk29"))
    text_inner.norm.register_forward_hook(cap("voxtral_post_norm"))

    # Forward — no audio, no KV cache
    with torch.no_grad():
        out = text_inner(input_ids=ids, use_cache=False)
        h = out.last_hidden_state
        logits = lm_head(h)
    print(f"hidden: {tuple(h.shape)}, logits: {tuple(logits.shape)}")

    captures["voxtral_logits"] = logits[0].detach().cpu().float().numpy()
    for k, v in captures.items():
        if v.ndim == 3 and v.shape[0] == 1:
            v = v[0]
        np.save(args.out_dir / f"{k}.npy", v)
        print(f"  {k}: {v.shape}")

    topk = np.argsort(-captures["voxtral_logits"], axis=-1)[:, :5].astype(np.int32)
    np.save(args.out_dir / "voxtral_topk.npy", topk)
    print(f"  voxtral_topk: {topk.shape}")
    tok = processor.tokenizer
    last_n = min(5, topk.shape[0])
    for i in range(topk.shape[0] - last_n, topk.shape[0]):
        toks = [tok.decode([int(t)]) for t in topk[i]]
        print(f"  pos {i}: {toks}")

    print(f"\nDone: {args.out_dir}")


if __name__ == "__main__":
    main()
