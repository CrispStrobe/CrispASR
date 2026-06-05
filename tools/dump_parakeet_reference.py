#!/usr/bin/env python3
"""Dump NeMo parakeet-tdt predictor + joint outputs step-by-step.

The C++ runtime emits one token then collapses to all-blanks on the
Japanese model. Encoder output and predictor-at-SOS already match
NeMo to 4 decimals, so the bug must be downstream: the predictor
state after the first non-blank emission, the joint head's response
to that state, or both.

This script reproduces the NeMo TDT greedy decode with verbose dumps
so the C++ PARAKEET_DEBUG output can be diffed line by line.

Usage on Kaggle:
    pip install -q 'numpy<2.2' 'scipy<1.15'
    pip install -q nemo_toolkit[asr]
    python dump_parakeet_reference.py \\
        --model nvidia/parakeet-tdt_ctc-0.6b-ja \\
        --audio /tmp/ja_speech.wav \\
        --steps 10
"""

import argparse
import sys

import numpy as np
import torch


def vec_stats(name: str, x: torch.Tensor) -> str:
    f = x.flatten().float()
    return (
        f"{name}: shape={tuple(x.shape)} mean={f.mean().item():+.4f} "
        f"std={f.std().item():.4f} min={f.min().item():+.3f} "
        f"max={f.max().item():+.3f} [0..3]={f[:4].tolist()}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/parakeet-tdt_ctc-0.6b-ja")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--steps", type=int, default=10,
                    help="how many emission steps to trace before stopping")
    args = ap.parse_args()

    import nemo.collections.asr as nemo_asr
    import soundfile as sf

    print(f"Loading: {args.model}")
    model = nemo_asr.models.ASRModel.from_pretrained(args.model).eval()
    if torch.cuda.is_available():
        model = model.cuda()
    dev = next(model.parameters()).device

    decoder = model.decoder        # RNNTDecoder (predictor)
    joint = model.joint            # RNNTJoint (or RNNTJointTDT)

    # ---- 1. Encoder pass on the audio ----------------------------------
    audio, sr = sf.read(args.audio)
    assert sr == 16000, f"expected 16k mono, got {sr}"
    sig = torch.tensor(audio, dtype=torch.float32, device=dev).unsqueeze(0)
    sig_len = torch.tensor([sig.shape[1]], device=dev)
    with torch.no_grad():
        feats, feat_len = model.preprocessor(input_signal=sig, length=sig_len)
        enc, enc_len = model.encoder(audio_signal=feats, length=feat_len)
    enc_b = enc.transpose(1, 2).contiguous()  # (B, T, D) for joint convenience
    T = int(enc_len.item())
    print(f"encoder: T={T} frames, d_model={enc_b.shape[-1]}")
    print("  " + vec_stats("enc[0,0]", enc_b[0, 0]))
    print("  " + vec_stats("enc[0,T//2]", enc_b[0, T // 2]))

    # ---- 2. Predictor + joint inventory --------------------------------
    print("\ndecoder.prediction modules:")
    for n, m in decoder.named_modules():
        if not n:
            continue
        cls = m.__class__.__name__
        # bail out of leaf-of-leaf so we get the interesting layers
        if cls in ("Sequential", "ModuleList"):
            continue
        print(f"  {n}: {cls}")
    print("\njoint modules:")
    for n, m in joint.named_modules():
        if not n:
            continue
        cls = m.__class__.__name__
        print(f"  {n}: {cls}")

    blank_id = decoder.blank_idx if hasattr(decoder, "blank_idx") else decoder.vocab_size
    print(f"\nblank_id={blank_id}  vocab_size={getattr(decoder, 'vocab_size', '?')}")

    # ---- 3. Predictor at SOS (verifies biases) -------------------------
    # This is what we already know matches the C++ runtime. We dump it
    # again so the next run has a complete trace.
    state = decoder.initialize_state(enc[:1])  # zero state
    sos = torch.full((1, 1), blank_id, dtype=torch.long, device=dev)
    with torch.no_grad():
        # NeMo's prediction() returns (output, hidden) where output is the
        # LSTM hidden post-projection; matches our pred_out.
        pred_out, new_state = decoder.predict(sos, state, add_sos=False, batch_size=1)
    pred_sos = pred_out.squeeze()  # (1, 1, H) -> (H,)
    print("\nSOS predictor output (input = blank, state = zero):")
    print("  " + vec_stats("pred_out[SOS]", pred_sos))

    # ---- 4. Greedy TDT decode with dumps -------------------------------
    # Reproduce the standard NeMo greedy-TDT loop with explicit logging.
    n_durations = len(getattr(joint, "durations", [0, 1, 2, 3, 4]))
    durations = list(getattr(joint, "durations", [0, 1, 2, 3, 4]))
    print(f"\nTDT durations = {durations}")

    state = decoder.initialize_state(enc[:1])
    last_label = torch.full((1, 1), blank_id, dtype=torch.long, device=dev)
    with torch.no_grad():
        pred_out, state = decoder.predict(last_label, state, add_sos=False, batch_size=1)
    pred_u = pred_out.squeeze(1)  # (1, 1, H) -> (1, H)

    t = 0
    emitted = []
    n_inner = 0
    print("\n--- greedy decode trace ---")
    while t < T and len(emitted) < args.steps:
        f_t = enc_b[:, t : t + 1, :]            # (1, 1, D)
        with torch.no_grad():
            log = joint.joint(f_t, pred_out)    # (1, 1, 1, V)
        logits = log.squeeze().float()           # (V,)
        n_classes = logits.shape[0]
        # split: vocab+blank   |   durations
        n_vocab_blk = n_classes - n_durations
        vocab_logits = logits[:n_vocab_blk]
        dur_logits = logits[n_vocab_blk:]
        tok = int(vocab_logits.argmax().item())
        dur_id = int(dur_logits.argmax().item())
        dur_skip = durations[dur_id]
        print(f"t={t:3d}  inner={n_inner}  "
              f"argmax_tok={tok}({'BLANK' if tok == blank_id else 'TOK'})  "
              f"vocab_lp={vocab_logits[tok].item():+.3f}  blank_lp={vocab_logits[blank_id].item():+.3f}  "
              f"dur={dur_skip}")
        if tok == blank_id:
            t += max(1, dur_skip)
            n_inner = 0
            continue
        # emit + advance predictor
        emitted.append(tok)
        last_label = torch.tensor([[tok]], device=dev)
        with torch.no_grad():
            pred_out, state = decoder.predict(last_label, state, add_sos=False, batch_size=1)
        pred_u = pred_out.squeeze(1)
        print("  emit#{} tok={}  ".format(len(emitted), tok)
              + vec_stats("pred_out", pred_u.squeeze()))
        if dur_skip > 0:
            t += dur_skip
            n_inner = 0
        else:
            n_inner += 1
            if n_inner >= 10:
                t += 1
                n_inner = 0

    # ---- 5. Final transcript -------------------------------------------
    print(f"\nemitted token ids: {emitted[:20]}...")
    if hasattr(model, "tokenizer") and emitted:
        try:
            text = model.tokenizer.ids_to_text(emitted)
        except Exception:
            text = model.tokenizer.tokenizer.decode(emitted)
        print(f"transcript (first {len(emitted)} tokens): {text!r}")

    # ---- 6. NeMo's own transcribe (for sanity) -------------------------
    with torch.no_grad():
        out = model.transcribe([args.audio])
    print(f"\nNeMo transcribe(): {out[0].text!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
