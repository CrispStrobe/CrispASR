#!/usr/bin/env python3
"""Dump per-stage intermediates of the OmniVoice audio-tokenizer ENCODE path
(HiggsAudioV2TokenizerModel.encode) → ref.gguf for the crispasr-diff harness.

This is the reference for the C++ voice-clone `higgs_encode` port (#254). It runs
the REAL transformers `encode()` with forward hooks so the captured stages are
exactly what the model computes (not a hand-reimplementation).

Requires transformers >= 5.3 (HiggsAudioV2TokenizerModel) — use the isolated venv,
NOT the shared conda base (which is 4.57.6). Also needs torch, torchaudio, gguf,
soundfile.

Usage:
    <venv>/bin/python tools/dump_omnivoice_encode_reference.py \
        --model /Volumes/backups/ai/crispasr-gguf/omnivoice-ref-src/audio_tokenizer \
        --audio samples/jfk.wav \
        --output /Volumes/backups/ai/crispasr-gguf/omnivoice-encode-ref.gguf

Stages dumped (all F32, ggml column-major = numpy row-major of (T, C) → ne=[C,T]):
    input_wav24k        (T_samp,)     mono 24 kHz, clipped to a multiple of 960
    sem_hidden_mean     (T50, 768)    mean over all 13 HuBERT hidden states
    sem_ds              (T25, 768)    after [::2] downsample
    e_semantic          (T25, 768)    encoder_semantic bridge output
    e_acoustic          (T25, 256)    DAC acoustic encoder output
    emb_fc              (T25, 1024)   after cat([acoustic,semantic]) + fc
    codes               (8,  T25)     int32 RVQ codes (0..1023)
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="audio_tokenizer dir (HF)")
    ap.add_argument("--audio", required=True, help="reference WAV")
    ap.add_argument("--output", required=True, help="ref.gguf path")
    args = ap.parse_args()

    import torch
    import torchaudio
    from transformers import AutoModel

    print(f"loading tokenizer from {args.model} …", file=sys.stderr)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
    model.eval()

    # --- Locate the encode-path submodules by attribute name (map_tensor_name
    #     in the converter proves these HF names). Fail loudly if the modeling
    #     changed so we never silently dump the wrong tensor. ---
    def get(mod, name):
        m = getattr(mod, name, None)
        if m is None:
            raise SystemExit(f"module '{name}' not found on {type(mod).__name__}; "
                             f"available: {[n for n,_ in mod.named_children()]}")
        return m

    semantic_model = get(model, "semantic_model")
    encoder_semantic = get(model, "encoder_semantic")
    acoustic_encoder = get(model, "acoustic_encoder")
    fc = get(model, "fc")

    caps: dict = {}
    hooks = []

    def cap(name):
        def hook(_m, _inp, out):
            caps[name] = out
        return hook

    # semantic_model returns a BaseModelOutput with .hidden_states (13 tensors)
    hooks.append(semantic_model.register_forward_hook(cap("_sem_out")))
    hooks.append(encoder_semantic.register_forward_hook(cap("e_semantic")))
    hooks.append(acoustic_encoder.register_forward_hook(cap("e_acoustic")))
    hooks.append(fc.register_forward_hook(cap("emb_fc")))

    # --- Load + prep audio: mono, 24 kHz, clip to a multiple of hop_length(960) ---
    # soundfile (not torchaudio.load, which now needs torchcodec).
    import soundfile as sf
    data, sr = sf.read(args.audio, dtype="float32", always_2d=True)  # (T, C)
    wav = torch.from_numpy(data.T)  # (C, T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 24000:
        wav = torchaudio.functional.resample(wav, sr, 24000)
    n = wav.shape[-1]
    clip = n % 960
    if clip:
        wav = wav[:, : n - clip]
    print(f"audio: {wav.shape[-1]} samples @ 24kHz ({wav.shape[-1]/24000:.2f}s)", file=sys.stderr)

    inp = wav.unsqueeze(0)  # (1,1,T)
    with torch.no_grad():
        enc = model.encode(inp)
    codes = getattr(enc, "audio_codes", enc)
    codes = codes.squeeze(0).cpu().numpy().astype(np.int32)  # (8, T25)

    for h in hooks:
        h.remove()

    # --- Reduce the semantic hidden states exactly as encode() does ---
    sem_out = caps["_sem_out"]
    hidden_states = getattr(sem_out, "hidden_states", None)
    if hidden_states is None:
        raise SystemExit("semantic_model did not return hidden_states — "
                         "encode() may not pass output_hidden_states=True")
    stacked = torch.stack(list(hidden_states), dim=1)  # (1, 13, T50, 768)
    sem_hidden_mean = stacked.mean(dim=1)[0]           # (T50, 768)
    sem_ds = sem_hidden_mean[::2]                      # (T25, 768)

    def np2d(t):  # (…, T, C) torch → numpy (T, C) f32
        a = t
        if a.dim() == 3:
            a = a[0]
        a = a.detach().cpu().float().numpy()
        return np.ascontiguousarray(a)

    e_semantic = np2d(caps["e_semantic"].transpose(-1, -2))  # (768,T)→(T,768)
    e_acoustic = np2d(caps["e_acoustic"].transpose(-1, -2))  # (256,T)→(T,256)
    emb_fc = np2d(caps["emb_fc"])                            # fc acts on (…,T,1024)

    stages = {
        "input_wav24k": wav[0].cpu().numpy().astype(np.float32),
        "sem_hidden_mean": sem_hidden_mean.cpu().numpy().astype(np.float32),
        "sem_ds": sem_ds.cpu().numpy().astype(np.float32),
        "e_semantic": e_semantic.astype(np.float32),
        "e_acoustic": e_acoustic.astype(np.float32),
        "emb_fc": emb_fc.astype(np.float32),
        "codes": codes,  # int32
    }
    for k, v in stages.items():
        print(f"  {k:18s} shape={v.shape} dtype={v.dtype}", file=sys.stderr)

    # --- Write ref.gguf (ggml column-major: a numpy (T,C) row-major array maps to
    #     ne=[C,T]; gguf writer takes numpy arrays and stores them as-is). ---
    from gguf import GGUFWriter
    w = GGUFWriter(args.output, "omnivoice_encode_ref")
    for k, v in stages.items():
        w.add_tensor(k, np.ascontiguousarray(v))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
