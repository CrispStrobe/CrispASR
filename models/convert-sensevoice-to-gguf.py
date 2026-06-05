#!/usr/bin/env python3
"""Convert FunAudioLLM/SenseVoiceSmall to GGUF.

SenseVoiceSmall is the encoder-only multi-task ASR sibling of Fun-ASR-Nano:
same SenseVoiceEncoderSmall topology (1 encoders0 entry block in_size=560
→ size=512, 49 encoders main blocks, 20 tp_encoders blocks), but instead
of an LLM decoder it emits its predictions through a single CTC head
(vocab=25055) and uses 4 "control query" embeddings prepended to the
LFR fbank features for language / event / emotion / textnorm output.

State-dict layout (the only three top-level prefixes):
  encoder.encoders0.0.*          (560→512 SANM entry block)
  encoder.encoders.{0..48}.*     (49 SANM blocks, 512→512)
  encoder.tp_encoders.{0..19}.*  (20 SANM blocks, 512→512)
  encoder.after_norm.{w,b}       (between main + tp)
  encoder.tp_norm.{w,b}          (after tp)
  embed.weight                   (16, 560) — query embeddings
  ctc.ctc_lo.{weight,bias}       (25055, 512) — CTC head + bias

Inference (matches funasr/models/sense_voice/model.py inference()):
  audio(16 kHz)
    → WavFrontend (hamming kaldi-fbank + LFR(m=7, n=6))   (T_lfr, 560)
    → prepend 4 query embeds:
        idx 0: language     {0=auto, 3=zh, 4=en, 7=yue, 11=ja, 12=ko, 13=nospeech}
        idx 1: event        always 1
        idx 2: emotion      always 2
        idx 3: textnorm     {14=withitn, 15=woitn}
        sequence: [lang_q, event_q, emo_q, textnorm_q, fbank_lfr]
                                                              (T_lfr+4, 560)
    → SenseVoiceEncoderSmall (70 SANM blocks)            (T_lfr+4, 512)
    → CTC head: (Linear 512→25055) + bias                (T_lfr+4, 25055)
    → log_softmax → argmax → unique_consecutive → drop blank_id=0
    → SentencePiece decode (chn_jpn_yue_eng_ko_spectok.bpe.model)

Tensor naming in GGUF:
  sensevoice.enc.blk.K.norm1.{w,b}      (K=0 has dim 560, all others 512)
  sensevoice.enc.blk.K.norm2.{w,b}      (always 512)
  sensevoice.enc.blk.K.attn.qkv.{w,b}   (w: in_size×1536, b: 1536)
  sensevoice.enc.blk.K.attn.out.{w,b}   (512×512, 512)
  sensevoice.enc.blk.K.attn.fsmn.w      (11×512 after squeeze)
  sensevoice.enc.blk.K.ffn.l1.{w,b}     (512×2048, 2048)
  sensevoice.enc.blk.K.ffn.l2.{w,b}     (2048×512, 512)
  sensevoice.enc.after_norm.{w,b}       (512)
  sensevoice.enc.tp_norm.{w,b}          (512)
  sensevoice.query_embed.w              (16, 560)
  sensevoice.ctc.{w,b}                  (25055×512, 25055)

Plus `tokenizer.ggml.tokens` carrying the 25055 SentencePiece pieces so
the runtime can detokenise without linking libsentencepiece.

Usage:
  python models/convert-sensevoice-to-gguf.py \\
      --input FunAudioLLM/SenseVoiceSmall \\
      --bpemodel /path/to/chn_jpn_yue_eng_ko_spectok.bpe.model \\
      --output /Volumes/backups/ai/crispasr-models/sensevoice-small/sensevoice-small-f16.gguf
"""

import argparse
import os
import sys

import numpy as np

try:
    import gguf
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ggml", "python"))
    import gguf


def enc_block_path(layer_idx: int) -> str:
    """Upstream state-dict prefix for SANM block at flat index 0..69."""
    if layer_idx == 0:
        return "encoder.encoders0.0"
    if layer_idx < 50:
        return f"encoder.encoders.{layer_idx - 1}"
    return f"encoder.tp_encoders.{layer_idx - 50}"


def to_np(t):
    import torch
    t = t.detach()
    if t.dtype == torch.bfloat16:
        return t.float().numpy()
    return t.numpy()


def add_tensor(writer, name, t, *, force_f32=False):
    assert len(name) < 64, f"GGUF name too long ({len(name)}): {name}"
    if not force_f32 and t.ndim >= 2:
        data = np.ascontiguousarray(t.astype(np.float16))
        dtype = "F16"
    else:
        data = np.ascontiguousarray(t.astype(np.float32))
        dtype = "F32"
    writer.add_tensor(name, data)
    return dtype


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="HF model ID or local snapshot dir")
    ap.add_argument("--bpemodel", required=True,
                    help="Path to chn_jpn_yue_eng_ko_spectok.bpe.model (download "
                         "via funasr AutoModel — the file is not in the HF snapshot, "
                         "it's downloaded from ModelScope at runtime)")
    ap.add_argument("--output", required=True, help="Output GGUF path")
    args = ap.parse_args()

    import torch
    import sentencepiece as spm

    if os.path.isdir(args.input):
        base = args.input
    else:
        from huggingface_hub import snapshot_download
        print(f"Downloading {args.input}")
        base = snapshot_download(repo_id=args.input, allow_patterns=["*.pt", "*.yaml"])

    model_pt = os.path.join(base, "model.pt")
    print(f"  Loading state dict from {model_pt}")
    raw = torch.load(model_pt, map_location="cpu", weights_only=False)
    sd = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    print(f"  {len(sd)} tensors")

    # ---- SentencePiece pieces ----
    print(f"  Loading SentencePiece model from {args.bpemodel}")
    sp = spm.SentencePieceProcessor()
    sp.load(args.bpemodel)
    vocab_size = sp.GetPieceSize()
    print(f"  vocab_size = {vocab_size}")
    pieces = [sp.id_to_piece(i) for i in range(vocab_size)]

    # ---- Hparams ----
    n_enc_base = 50
    n_enc_tp = 20
    hp = dict(
        n_mels=80, lfr_m=7, lfr_n=6, sample_rate=16000,
        frame_length_ms=25, frame_shift_ms=10,
        d_model=512, n_heads=4, ffn_dim=2048,
        n_blocks_base=n_enc_base, n_blocks_tp=n_enc_tp, sanm_kernel=11,
        vocab_size=vocab_size,
        blank_id=0, sos_id=1, eos_id=2,
        # Query token ids — match SenseVoiceSmall.lid_dict / textnorm_dict /
        # emo_dict in funasr/models/sense_voice/model.py
        lang_auto=0, lang_zh=3, lang_en=4, lang_yue=7, lang_ja=11, lang_ko=12,
        lang_nospeech=13, event_query=1, emo_query=2,
        textnorm_withitn=14, textnorm_woitn=15,
    )

    # ---- Open writer ----
    writer = gguf.GGUFWriter(args.output, "sensevoice")
    writer.add_name("SenseVoiceSmall")
    for k, v in hp.items():
        writer.add_uint32(f"sensevoice.{k}", int(v))

    writer.add_tokenizer_model("sentencepiece")
    writer.add_token_list(pieces)

    n_written = 0
    n_f16 = 0
    n_f32 = 0

    def write(name, tensor, *, force_f32=False):
        nonlocal n_written, n_f16, n_f32
        t = to_np(tensor)
        dtype = add_tensor(writer, name, t, force_f32=force_f32)
        if dtype == "F16":
            n_f16 += 1
        else:
            n_f32 += 1
        n_written += 1

    # ---- Encoder blocks ----
    for L in range(n_enc_base + n_enc_tp):
        p = enc_block_path(L)
        o = f"sensevoice.enc.blk.{L}"
        write(f"{o}.norm1.w", sd[f"{p}.norm1.weight"], force_f32=True)
        write(f"{o}.norm1.b", sd[f"{p}.norm1.bias"], force_f32=True)
        write(f"{o}.norm2.w", sd[f"{p}.norm2.weight"], force_f32=True)
        write(f"{o}.norm2.b", sd[f"{p}.norm2.bias"], force_f32=True)
        write(f"{o}.attn.qkv.w", sd[f"{p}.self_attn.linear_q_k_v.weight"])
        write(f"{o}.attn.qkv.b", sd[f"{p}.self_attn.linear_q_k_v.bias"], force_f32=True)
        write(f"{o}.attn.out.w", sd[f"{p}.self_attn.linear_out.weight"])
        write(f"{o}.attn.out.b", sd[f"{p}.self_attn.linear_out.bias"], force_f32=True)
        # Squeeze the middle 1-dim of the depthwise conv weight: (n_feat, 1, K) → (n_feat, K).
        fsmn = to_np(sd[f"{p}.self_attn.fsmn_block.weight"])
        if fsmn.ndim == 3 and fsmn.shape[1] == 1:
            fsmn = fsmn.squeeze(1)
        add_tensor(writer, f"{o}.attn.fsmn.w", fsmn)
        n_written += 1
        n_f16 += 1
        write(f"{o}.ffn.l1.w", sd[f"{p}.feed_forward.w_1.weight"])
        write(f"{o}.ffn.l1.b", sd[f"{p}.feed_forward.w_1.bias"], force_f32=True)
        write(f"{o}.ffn.l2.w", sd[f"{p}.feed_forward.w_2.weight"])
        write(f"{o}.ffn.l2.b", sd[f"{p}.feed_forward.w_2.bias"], force_f32=True)

    write("sensevoice.enc.after_norm.w", sd["encoder.after_norm.weight"], force_f32=True)
    write("sensevoice.enc.after_norm.b", sd["encoder.after_norm.bias"], force_f32=True)
    write("sensevoice.enc.tp_norm.w", sd["encoder.tp_norm.weight"], force_f32=True)
    write("sensevoice.enc.tp_norm.b", sd["encoder.tp_norm.bias"], force_f32=True)

    # ---- Query embedding table (16, 560) ----
    write("sensevoice.query_embed.w", sd["embed.weight"])

    # ---- CTC head ----
    write("sensevoice.ctc.w", sd["ctc.ctc_lo.weight"])
    write("sensevoice.ctc.b", sd["ctc.ctc_lo.bias"], force_f32=True)

    print()
    print(f"  Total {n_written} tensors  (F16: {n_f16}, F32: {n_f32})")
    print(f"  Vocab pieces: {vocab_size}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size = os.path.getsize(args.output)
    print(f"\nDone: {args.output} ({size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
