#!/ usr / bin / env python3
""
    "Convert Facebook OmniASR-LLM to GGUF.

    Architecture : wav2vec2 encoder(24L, d = 1024) +
                   Linear(1024→4096) + 12 - layer LLaMA decoder(d = 4096, 8 heads, SwiGLU, RMSNorm, RoPE) +
                   LM head(4096→9812).

                   The encoder is identical to OmniASR
                   -
                   CTC -
                   300M. The decoder is a standard LLaMA model with no attention bias and SwiGLU
                       FFN(gate + inner→output)
                           .

                       Usage
    : python models / convert
                   -
                   omniasr - llm - to - gguf.py-- input facebook / omniASR - LLM - 300M --output omniasr - llm -
                   300m.gguf ""
                             "

                   import argparse import os import sys

                   import numpy as np

                   try : import gguf except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ggml", "python"))
    import gguf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="HF model ID (e.g. facebook/omniASR-LLM-300M)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import torch
    import sentencepiece as spm
    from huggingface_hub import hf_hub_download

#Download or locate model + tokenizer
    if os.path.isdir(args.input):
#Local directory
        model_name = os.path.basename(args.input.rstrip("/"))
#Find.pt file
        pt_files = [f for f in os.listdir(args.input) if f.endswith(".pt")]
        if not pt_files:
            raise FileNotFoundError(f"No .pt file in {args.input}")
        pt_path = os.path.join(args.input, pt_files[0])
#Search for tokenizer : prefer v2, fall back to v1
        tok_candidates = [
            os.path.join(args.input, "omniASR_tokenizer_written_v2.model"),
            os.path.join(args.input, "omniASR_tokenizer.model"),
            os.path.join(args.input, "..", "omniASR_tokenizer_written_v2.model"),
            os.path.join(args.input, "..", "omniASR_tokenizer.model"),
        ]
        tok_path = None
        for c in tok_candidates:
            if os.path.exists(c):
                tok_path = c
                break
        if not tok_path:
            raise FileNotFoundError(f"No tokenizer found in {args.input}")
    else:
        model_name = args.input.split("/")[-1]
        pt_path = hf_hub_download(args.input, f"{model_name}.pt")
        tok_path = hf_hub_download(args.input, "omniASR_tokenizer.model")

    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    print(f"Loaded {len(sd)} tensors from {model_name}")

#Infer encoder architecture(same as CTC)
    n_enc = max(int(k.split('.')[2]) for k in sd if k.startswith("encoder.layers.")) + 1
    d_enc = sd["encoder.layers.0.self_attn.q_proj.weight"].shape[0]
    d_ffn_enc = sd["encoder.layers.0.ffn.inner_proj.weight"].shape[0]
    n_heads_enc = d_enc // 64  # head_dim=64
    n_cnn = max(int(k.split('.')[3]) for k in sd
                if k.startswith("encoder_frontend.feature_extractor.layers.")) + 1

#Infer decoder architecture
    n_dec = max(int(k.split('.')[2]) for k in sd if k.startswith("llama_decoder.layers.")) + 1
    d_dec = sd["llama_decoder.layers.0.self_attn.q_proj.weight"].shape[0]
    d_ffn_dec = sd["llama_decoder.layers.0.ffn.gate_proj.weight"].shape[0]
#head_dim for decoder : Q is[d_dec, d_dec], so n_heads = d_dec / head_dim
#LLaMA - 300M uses 8 heads → head_dim = 4096 / 8 = 512
    n_heads_dec = 8  # Standard for this model
    head_dim_dec = d_dec // n_heads_dec

    vocab_size = sd["final_proj.weight"].shape[0]
    n_langs = sd["lang_embeddings.weight"].shape[0]

    print(f"  Encoder: {n_enc}L, d={d_enc}, ffn={d_ffn_enc}, heads={n_heads_enc}, cnn={n_cnn}")
    print(f"  Decoder: {n_dec}L, d={d_dec}, ffn={d_ffn_dec}, heads={n_heads_dec}, head_dim={head_dim_dec}")
    print(f"  Vocab: {vocab_size}, Languages: {n_langs}")

#CNN strides
    strides = [5] + [2] * (n_cnn - 1)

#Tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load(tok_path)
    vocab = [sp.IdToPiece(i) for i in range(sp.GetPieceSize())]

#Create GGUF
    writer = gguf.GGUFWriter(args.output, "omniasr-llm")
    writer.add_name(f"OmniASR-LLM-{model_name.split('-')[-1]}")

#Encoder params(same as CTC)
    writer.add_uint32("omniasr.d_model", d_enc)
    writer.add_uint32("omniasr.d_ffn", d_ffn_enc)
    writer.add_uint32("omniasr.n_heads", n_heads_enc)
    writer.add_uint32("omniasr.n_enc_layers", n_enc)
    writer.add_uint32("omniasr.n_cnn_layers", n_cnn)
    writer.add_uint32("omniasr.vocab_size", vocab_size)
    writer.add_uint32("omniasr.bos_id", sp.bos_id())
    writer.add_uint32("omniasr.eos_id", sp.eos_id())
    writer.add_uint32("omniasr.pad_id", sp.pad_id())
    writer.add_uint32("omniasr.unk_id", sp.unk_id())
    writer.add_array("omniasr.cnn_strides", strides)

#Decoder params
    writer.add_uint32("omniasr.d_dec", d_dec)
    writer.add_uint32("omniasr.d_ffn_dec", d_ffn_dec)
    writer.add_uint32("omniasr.n_heads_dec", n_heads_dec)
    writer.add_uint32("omniasr.n_dec_layers", n_dec)
    writer.add_uint32("omniasr.head_dim_dec", head_dim_dec)
    writer.add_uint32("omniasr.n_langs", n_langs)
    writer.add_uint32("omniasr.model_type", 1)  # 0=CTC, 1=LLM

    writer.add_array("tokenizer.ggml.tokens", vocab)

#Embed language mapping from parquet for dynamic lang selection
    try:
        import pyarrow.parquet as pq
        parquet_path = os.path.join(os.path.dirname(__file__), "..", "tools", "languges_lookup_table.parquet")
        if not os.path.exists(parquet_path):
#Try downloading from GitHub
            import requests
            url = "https://github.com/facebookresearch/omnilingual-asr/raw/main/src/omnilingual_asr/models/wav2vec2_llama/languges_lookup_table.parquet"
            r = requests.get(url, allow_redirects=True)
            if r.status_code == 200:
                os.makedirs(os.path.dirname(parquet_path) or ".", exist_ok=True)
                with open(parquet_path, "wb") as f:
                    f.write(r.content)
        if os.path.exists(parquet_path):
            table = pq.read_table(parquet_path)
            rows = sorted(table.to_pylist(), key=lambda r: r["index"])
            lang_codes = [r["lang"] for r in rows]
            writer.add_array("omniasr.lang_codes", lang_codes)
            print(f"  Language mapping: {len(lang_codes)} codes embedded")
    except Exception as e:
        print(f"  Language mapping not embedded: {e}")

    def f16(t):
        return t.astype(np.float16) if t.dtype == np.float32 else t

    def f32(t):
        return t.astype(np.float32)

#Shorten tensor names
    def shorten(name):
        name = name.replace("encoder_frontend.feature_extractor.layers.", "cnn.")
        name = name.replace("encoder_frontend.model_dim_proj.", "proj.")
        name = name.replace("encoder.layers.", "enc.")
        name = name.replace("encoder.layer_norm.", "enc_ln.")
        name = name.replace("llama_decoder.layers.", "dec.")
        name = name.replace("llama_decoder.layer_norm.", "dec_ln.")
        name = name.replace("self_attn.", "attn.")
        name = name.replace("self_attn_layer_norm.", "attn_ln.")
        name = name.replace("ffn_layer_norm.", "ffn_ln.")
        name = name.replace("ffn.inner_proj.", "ffn.up.")
        name = name.replace("ffn.output_proj.", "ffn.down.")
        name = name.replace("ffn.gate_proj.", "ffn.gate.")
        name = name.replace("layer_norm.", "ln.")
        name = name.replace("output_proj.", "out.")
        name = name.replace("final_proj.", "lm_head.")
        name = name.replace("encoder_proj.", "enc_proj.")
        name = name.replace("text_frontend.", "tok_emb.")
        name = name.replace("lang_embeddings.", "lang_emb.")
        name = name.replace("encoder_frontend.post_extract_layer_norm.", "encoder_frontend.post_extract_ln.")
        name = name.replace("encoder_frontend.post_extract_ln.", "post_extract_ln.")
        return name

#Pre - compute pos_encoder weight normalization(same as CTC)
    if "encoder_frontend.pos_encoder.conv.weight_v" in sd:
        wv = sd["encoder_frontend.pos_encoder.conv.weight_v"]
        wg = sd["encoder_frontend.pos_encoder.conv.weight_g"]
        v_norm = wv.reshape(wv.shape[0], -1).norm(dim=1).reshape(-1, 1, 1)
        w_combined = (wg / (v_norm + 1e-12)) * wv
        sd["pos_conv.weight"] = w_combined
        sd["pos_conv.bias"] = sd["encoder_frontend.pos_encoder.conv.bias"]
        print(f"  Pre-computed pos_conv weight: {w_combined.shape}")
        del sd["encoder_frontend.pos_encoder.conv.weight_g"]
        del sd["encoder_frontend.pos_encoder.conv.weight_v"]
        del sd["encoder_frontend.pos_encoder.conv.bias"]

    tensor_count = 0
    for name in sorted(sd.keys()):
        t = sd[name].float().numpy()
        gguf_name = shorten(name)

        if len(gguf_name) >= 64:
            print(f"  WARNING: name too long ({len(gguf_name)}): {gguf_name}")
            continue

#Store norms / biases as F32, weights as F16
        if "norm" in name or "ln." in gguf_name or name.endswith(".bias") or len(t.shape) <= 1:
            data = f32(t)
        else:
            data = f16(t)

        writer.add_tensor(gguf_name, data)
        tensor_count += 1
        if tensor_count <= 5 or tensor_count % 50 == 0:
            print(f"  [{tensor_count}] {gguf_name:50s} {str(data.shape):20s}")

    print(f"  total: {tensor_count} tensors")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    sz = os.path.getsize(args.output)
    print(f"\nDone: {args.output} ({sz / 1e9:.2f} GB, {tensor_count} tensors)")


if __name__ == "__main__":
    main()
