"""
Export Cohere Transcribe to GGUF (for whisper.cpp fork).

Reads model.safetensors directly (no transformers model load needed).
All weights stored as F16 (converted from BF16). Vocab embedded in metadata.

Usage:
    python export_gguf.py --model-dir ./local_cohere_model --output cohere-transcribe.gguf

Output tensor naming convention  (used by cohere_whisper.cpp):
    fe.*           feature extraction (mel filterbank, window)
    enc.pre.*      Conv2D subsampling
    enc.blk.N.*    Conformer layer N (ff1, attn, conv, ff2, out_norm)
    enc.proj.*     encoder→decoder projection
    dec.emb.*      token + position embeddings
    dec.blk.N.*    decoder transformer layer N
    dec.out_ln.*   decoder final layer norm
    dec.head.*     output projection (logits)
"""

import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# safetensors lazy reader (no torch/transformers needed)
# ---------------------------------------------------------------------------

def open_safetensors(path):
    """Return (header_dict, mmap_array) for lazy tensor access."""
    import mmap
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    data_offset = 8 + n
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    return header, mm, data_offset, f


def load_tensor(header, mm, data_offset, key):
    """Load a single tensor as a numpy array (bf16 → float32 for safety)."""
    info = header[key]
    dtype_map = {
        "BF16": (np.uint16, 2),
        "F16":  (np.float16, 2),
        "F32":  (np.float32, 4),
        "I8":   (np.int8,   1),
        "I32":  (np.int32,  4),
        "I64":  (np.int64,  8),
    }
    np_dtype, itemsize = dtype_map[info["dtype"]]
    start, end = info["data_offsets"]
    raw = np.frombuffer(mm[data_offset + start: data_offset + end], dtype=np_dtype)
    shape = info["shape"]
    arr = raw.reshape(shape)
    if info["dtype"] == "BF16":
        # BF16 → F32: insert 16 zero bits at the low end
        u32 = arr.astype(np.uint32) << 16
        arr = u32.view(np.float32)
    return arr


def bf16_to_f16(arr):
    """Convert a float32 array (from bf16 expansion) to float16."""
    return arr.astype(np.float16)


# ---------------------------------------------------------------------------
# GGUF writer wrapper
# ---------------------------------------------------------------------------

def get_writer(output_path, arch):
    from gguf import GGUFWriter, GGMLQuantizationType
    return GGUFWriter(output_path, arch), GGMLQuantizationType


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def load_vocab(model_dir):
    """Return list of token strings indexed by token id."""
    import sentencepiece as spm

    tok_cfg_path = os.path.join(model_dir, "tokenizer_config.json")
    tok_model_path = os.path.join(model_dir, "tokenizer.model")

    sp = spm.SentencePieceProcessor()
    sp.Load(tok_model_path)

    vocab = {i: sp.id_to_piece(i) for i in range(sp.get_piece_size())}

    with open(tok_cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    for idx_str, info in cfg.get("added_tokens_decoder", {}).items():
        vocab[int(idx_str)] = info["content"]

    max_id = max(vocab)
    return [vocab.get(i, f"<unused_{i}>") for i in range(max_id + 1)]


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="./local_cohere_model")
    parser.add_argument("--output", default="cohere-transcribe.gguf")
    parser.add_argument("--f32", action="store_true",
                        help="Store ALL weights as F32 instead of F16")
    parser.add_argument("--f32-encoder", action="store_true",
                        help="Store encoder weight matrices as F32, decoder as F16")
    args = parser.parse_args()

    sf_path = os.path.join(args.model_dir, "model.safetensors")
    if not os.path.exists(sf_path):
        print(f"ERROR: {sf_path} not found"); sys.exit(1)

    target_dtype = np.float32 if args.f32 else np.float16
    dtype_label  = "F32" if args.f32 else "F16"
    enc_dtype    = np.float32 if (args.f32 or args.f32_encoder) else np.float16
    enc_label    = "F32" if (args.f32 or args.f32_encoder) else "F16"

    print(f"Loading safetensors metadata from {sf_path}...")
    header, mm, data_offset, sf_file = open_safetensors(sf_path)

    def get(key):
        t = load_tensor(header, mm, data_offset, key)
        return t.astype(target_dtype)

    def get_f32(key):
        return load_tensor(header, mm, data_offset, key).astype(np.float32)

    # --- Read config ----------------------------------------------------------
    with open(os.path.join(args.model_dir, "config.json")) as f:
        cfg = json.load(f)

    dec_cfg = cfg["transf_decoder"]["config_dict"]
    n_dec_layers = dec_cfg["num_layers"]           # 8
    dec_d        = dec_cfg["hidden_size"]          # 1024
    dec_heads    = dec_cfg["num_attention_heads"]  # 8
    dec_ffn      = dec_cfg.get("ffn_dim", dec_d * 4)
    dec_max_ctx  = dec_cfg["max_sequence_length"]  # 1024
    dec_head_dim = dec_d // dec_heads              # 128

    n_enc_layers = sum(1 for k in header if k.startswith("encoder.layers.") and k.endswith(".norm_out.weight"))
    enc_d        = 1280
    enc_heads    = 8
    enc_head_dim = enc_d // enc_heads  # 160
    enc_ffn      = 5120
    enc_conv_k   = header["encoder.layers.0.conv.depthwise_conv.weight"]["shape"][2]  # 9

    vocab_size = header["transf_decoder._embedding.token_embedding.weight"]["shape"][0]  # 16384

    print(f"  Encoder: {n_enc_layers} Conformer layers, d={enc_d}, heads={enc_heads}")
    print(f"  Decoder: {n_dec_layers} transformer layers, d={dec_d}, heads={dec_heads}")
    print(f"  Vocab:   {vocab_size}")

    # --- GGUF writer ----------------------------------------------------------
    writer, GGMLQt = get_writer(args.output, "cohere-transcribe")

    # Architecture metadata
    writer.add_string ("general.architecture",             "cohere-transcribe")
    writer.add_string ("general.name",                     "Cohere Transcribe 03-2026")
    writer.add_uint32 ("cohere_transcribe.vocab_size",     vocab_size)
    # encoder
    writer.add_uint32 ("cohere_transcribe.encoder.n_layers",   n_enc_layers)
    writer.add_uint32 ("cohere_transcribe.encoder.d_model",    enc_d)
    writer.add_uint32 ("cohere_transcribe.encoder.n_heads",    enc_heads)
    writer.add_uint32 ("cohere_transcribe.encoder.head_dim",   enc_head_dim)
    writer.add_uint32 ("cohere_transcribe.encoder.ffn_dim",    enc_ffn)
    writer.add_uint32 ("cohere_transcribe.encoder.conv_kernel",enc_conv_k)
    # decoder
    writer.add_uint32 ("cohere_transcribe.decoder.n_layers",   n_dec_layers)
    writer.add_uint32 ("cohere_transcribe.decoder.d_model",    dec_d)
    writer.add_uint32 ("cohere_transcribe.decoder.n_heads",    dec_heads)
    writer.add_uint32 ("cohere_transcribe.decoder.head_dim",   dec_head_dim)
    writer.add_uint32 ("cohere_transcribe.decoder.ffn_dim",    dec_ffn)
    writer.add_uint32 ("cohere_transcribe.decoder.max_ctx",    dec_max_ctx)
    # audio
    writer.add_uint32 ("cohere_transcribe.audio.sample_rate",  16000)
    writer.add_uint32 ("cohere_transcribe.audio.n_mels",       128)
    writer.add_uint32 ("cohere_transcribe.audio.n_fft",        512)
    writer.add_uint32 ("cohere_transcribe.audio.hop_length",   160)
    writer.add_uint32 ("cohere_transcribe.audio.win_length",   400)

    # Tokenizer vocab
    print("Embedding tokenizer vocab...")
    vocab_list = load_vocab(args.model_dir)
    writer.add_token_list(vocab_list)
    writer.add_tokenizer_model("llama")  # SPM-based

    def add(name, key, f32=False, enc=False):
        if f32:
            arr = get_f32(key)
        elif enc:
            arr = load_tensor(header, mm, data_offset, key).astype(enc_dtype)
        else:
            arr = get(key)
        writer.add_tensor(name, arr)
        print(f"  {name:50s} {str(list(arr.shape)):25s} {arr.dtype}")

    # -------------------------------------------------------------------------
    # Feature extraction
    # -------------------------------------------------------------------------
    print(f"\nWriting feature extraction ({enc_label})...")
    add("fe.mel_fb", "preprocessor.featurizer.fb", enc=True)   # [1, 128, 257]
    add("fe.window", "preprocessor.featurizer.window", f32=True)  # [400] keep f32

    # -------------------------------------------------------------------------
    # Encoder pre-encode (Conv2D subsampling)
    # -------------------------------------------------------------------------
    print(f"\nWriting pre-encode subsampling ({enc_label})...")
    for idx in [0, 2, 3, 5, 6]:
        for suffix in ["weight", "bias"]:
            k = f"encoder.pre_encode.conv.{idx}.{suffix}"
            if k in header:
                sf_is_bias = (suffix == "bias")
                add(f"enc.pre.conv.{idx}.{suffix}", k, f32=sf_is_bias, enc=(not sf_is_bias))
    add("enc.pre.out.weight", "encoder.pre_encode.out.weight", enc=True)
    add("enc.pre.out.bias",   "encoder.pre_encode.out.bias", f32=True)

    # -------------------------------------------------------------------------
    # Conformer encoder layers
    # -------------------------------------------------------------------------
    print(f"\nWriting {n_enc_layers} Conformer encoder layers ({enc_label})...")
    for i in range(n_enc_layers):
        prefix = f"encoder.layers.{i}"
        bp     = f"enc.blk.{i}"
        if i % 8 == 0:
            print(f"  layer {i}/{n_enc_layers}...")

        # Feed-forward 1 (Macaron pre-attention)
        add(f"{bp}.ff1.norm.weight", f"{prefix}.norm_feed_forward1.weight", f32=True)
        add(f"{bp}.ff1.norm.bias",   f"{prefix}.norm_feed_forward1.bias",   f32=True)
        add(f"{bp}.ff1.up.weight",   f"{prefix}.feed_forward1.linear1.weight",  enc=True)
        add(f"{bp}.ff1.up.bias",     f"{prefix}.feed_forward1.linear1.bias",    f32=True)
        add(f"{bp}.ff1.down.weight", f"{prefix}.feed_forward1.linear2.weight",  enc=True)
        add(f"{bp}.ff1.down.bias",   f"{prefix}.feed_forward1.linear2.bias",    f32=True)

        # Self-attention (Transformer-XL relative position)
        add(f"{bp}.attn.norm.weight",    f"{prefix}.norm_self_att.weight",          f32=True)
        add(f"{bp}.attn.norm.bias",      f"{prefix}.norm_self_att.bias",            f32=True)
        add(f"{bp}.attn.q.weight",       f"{prefix}.self_attn.linear_q.weight",     enc=True)
        add(f"{bp}.attn.q.bias",         f"{prefix}.self_attn.linear_q.bias",       f32=True)
        add(f"{bp}.attn.k.weight",       f"{prefix}.self_attn.linear_k.weight",     enc=True)
        add(f"{bp}.attn.k.bias",         f"{prefix}.self_attn.linear_k.bias",       f32=True)
        add(f"{bp}.attn.v.weight",       f"{prefix}.self_attn.linear_v.weight",     enc=True)
        add(f"{bp}.attn.v.bias",         f"{prefix}.self_attn.linear_v.bias",       f32=True)
        add(f"{bp}.attn.out.weight",     f"{prefix}.self_attn.linear_out.weight",   enc=True)
        add(f"{bp}.attn.out.bias",       f"{prefix}.self_attn.linear_out.bias",     f32=True)
        add(f"{bp}.attn.pos.weight",     f"{prefix}.self_attn.linear_pos.weight",   enc=True)
        add(f"{bp}.attn.pos_bias_u",     f"{prefix}.self_attn.pos_bias_u",          f32=True)
        add(f"{bp}.attn.pos_bias_v",     f"{prefix}.self_attn.pos_bias_v",          f32=True)

        # Convolution module
        add(f"{bp}.conv.norm.weight",    f"{prefix}.norm_conv.weight",              f32=True)
        add(f"{bp}.conv.norm.bias",      f"{prefix}.norm_conv.bias",                f32=True)
        add(f"{bp}.conv.pw1.weight",     f"{prefix}.conv.pointwise_conv1.weight",   enc=True)
        add(f"{bp}.conv.pw1.bias",       f"{prefix}.conv.pointwise_conv1.bias",     f32=True)
        add(f"{bp}.conv.dw.weight",      f"{prefix}.conv.depthwise_conv.weight",    enc=True)
        add(f"{bp}.conv.dw.bias",        f"{prefix}.conv.depthwise_conv.bias",      f32=True)
        # BatchNorm stored as scale/bias/mean/var (folded at inference)
        add(f"{bp}.conv.bn.weight",      f"{prefix}.conv.batch_norm.weight",        f32=True)
        add(f"{bp}.conv.bn.bias",        f"{prefix}.conv.batch_norm.bias",          f32=True)
        add(f"{bp}.conv.bn.mean",        f"{prefix}.conv.batch_norm.running_mean",  f32=True)
        add(f"{bp}.conv.bn.var",         f"{prefix}.conv.batch_norm.running_var",   f32=True)
        add(f"{bp}.conv.pw2.weight",     f"{prefix}.conv.pointwise_conv2.weight",   enc=True)
        add(f"{bp}.conv.pw2.bias",       f"{prefix}.conv.pointwise_conv2.bias",     f32=True)

        # Feed-forward 2 (Macaron post-conv)
        add(f"{bp}.ff2.norm.weight", f"{prefix}.norm_feed_forward2.weight",         f32=True)
        add(f"{bp}.ff2.norm.bias",   f"{prefix}.norm_feed_forward2.bias",           f32=True)
        add(f"{bp}.ff2.up.weight",   f"{prefix}.feed_forward2.linear1.weight",      enc=True)
        add(f"{bp}.ff2.up.bias",     f"{prefix}.feed_forward2.linear1.bias",        f32=True)
        add(f"{bp}.ff2.down.weight", f"{prefix}.feed_forward2.linear2.weight",      enc=True)
        add(f"{bp}.ff2.down.bias",   f"{prefix}.feed_forward2.linear2.bias",        f32=True)

        # Output norm
        add(f"{bp}.out_norm.weight", f"{prefix}.norm_out.weight", f32=True)
        add(f"{bp}.out_norm.bias",   f"{prefix}.norm_out.bias",   f32=True)

    # -------------------------------------------------------------------------
    # Encoder→decoder projection
    # -------------------------------------------------------------------------
    print(f"\nWriting enc→dec projection ({enc_label})...")
    add("enc.proj.weight", "encoder_decoder_proj.weight", enc=True)
    add("enc.proj.bias",   "encoder_decoder_proj.bias", f32=True)

    # -------------------------------------------------------------------------
    # Decoder embeddings
    # -------------------------------------------------------------------------
    print(f"\nWriting decoder embeddings ({dtype_label} weight matrices)...")
    add("dec.emb.weight",    "transf_decoder._embedding.token_embedding.weight")
    add("dec.pos.weight",    "transf_decoder._embedding.position_embedding.pos_enc")
    add("dec.emb_ln.weight", "transf_decoder._embedding.layer_norm.weight", f32=True)
    add("dec.emb_ln.bias",   "transf_decoder._embedding.layer_norm.bias",   f32=True)

    # -------------------------------------------------------------------------
    # Decoder transformer layers
    # -------------------------------------------------------------------------
    print(f"\nWriting {n_dec_layers} decoder layers ({dtype_label} weight matrices)...")
    for i in range(n_dec_layers):
        prefix = f"transf_decoder._decoder.layers.{i}"
        bp     = f"dec.blk.{i}"

        # Self-attention
        add(f"{bp}.attn_ln.weight",  f"{prefix}.layer_norm_1.weight",               f32=True)
        add(f"{bp}.attn_ln.bias",    f"{prefix}.layer_norm_1.bias",                 f32=True)
        add(f"{bp}.attn_q.weight",   f"{prefix}.first_sub_layer.query_net.weight")
        add(f"{bp}.attn_q.bias",     f"{prefix}.first_sub_layer.query_net.bias",    f32=True)
        add(f"{bp}.attn_k.weight",   f"{prefix}.first_sub_layer.key_net.weight")
        add(f"{bp}.attn_k.bias",     f"{prefix}.first_sub_layer.key_net.bias",      f32=True)
        add(f"{bp}.attn_v.weight",   f"{prefix}.first_sub_layer.value_net.weight")
        add(f"{bp}.attn_v.bias",     f"{prefix}.first_sub_layer.value_net.bias",    f32=True)
        add(f"{bp}.attn_o.weight",   f"{prefix}.first_sub_layer.out_projection.weight")
        add(f"{bp}.attn_o.bias",     f"{prefix}.first_sub_layer.out_projection.bias", f32=True)

        # Cross-attention (K/V projections live in decoder, computed once per utterance)
        add(f"{bp}.cross_ln.weight", f"{prefix}.layer_norm_2.weight",               f32=True)
        add(f"{bp}.cross_ln.bias",   f"{prefix}.layer_norm_2.bias",                 f32=True)
        add(f"{bp}.cross_q.weight",  f"{prefix}.second_sub_layer.query_net.weight")
        add(f"{bp}.cross_q.bias",    f"{prefix}.second_sub_layer.query_net.bias",   f32=True)
        add(f"{bp}.cross_k.weight",  f"{prefix}.second_sub_layer.key_net.weight")
        add(f"{bp}.cross_k.bias",    f"{prefix}.second_sub_layer.key_net.bias",     f32=True)
        add(f"{bp}.cross_v.weight",  f"{prefix}.second_sub_layer.value_net.weight")
        add(f"{bp}.cross_v.bias",    f"{prefix}.second_sub_layer.value_net.bias",   f32=True)
        add(f"{bp}.cross_o.weight",  f"{prefix}.second_sub_layer.out_projection.weight")
        add(f"{bp}.cross_o.bias",    f"{prefix}.second_sub_layer.out_projection.bias", f32=True)

        # FFN
        add(f"{bp}.ffn_ln.weight",   f"{prefix}.layer_norm_3.weight",               f32=True)
        add(f"{bp}.ffn_ln.bias",     f"{prefix}.layer_norm_3.bias",                 f32=True)
        add(f"{bp}.ffn_up.weight",   f"{prefix}.third_sub_layer.dense_in.weight")
        add(f"{bp}.ffn_up.bias",     f"{prefix}.third_sub_layer.dense_in.bias",     f32=True)
        add(f"{bp}.ffn_down.weight", f"{prefix}.third_sub_layer.dense_out.weight")
        add(f"{bp}.ffn_down.bias",   f"{prefix}.third_sub_layer.dense_out.bias",    f32=True)

    # -------------------------------------------------------------------------
    # Decoder output
    # -------------------------------------------------------------------------
    print(f"\nWriting decoder output ({dtype_label} weight matrices)...")
    add("dec.out_ln.weight", "transf_decoder._decoder.final_layer_norm.weight", f32=True)
    add("dec.out_ln.bias",   "transf_decoder._decoder.final_layer_norm.bias",   f32=True)
    add("dec.head.weight",   "log_softmax.mlp.layer0.weight")
    add("dec.head.bias",     "log_softmax.mlp.layer0.bias",  f32=True)

    # -------------------------------------------------------------------------
    # Write GGUF
    # -------------------------------------------------------------------------
    sf_file.close()
    print(f"\nWriting GGUF to {args.output}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_gb = os.path.getsize(args.output) / 1e9
    print(f"Done. {args.output}  ({size_gb:.2f} GB)")


if __name__ == "__main__":
    main()
