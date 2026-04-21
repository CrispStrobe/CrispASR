#!/usr/bin/env python3
"""Convert SpeechBrain ECAPA-TDNN LID to GGUF.

Model: speechbrain/lang-id-voxlingua107-ecapa (Apache-2.0)
Architecture: ECAPA-TDNN with SE-Res2Net blocks + attentive statistical pooling
  - 60-dim mel fbank input
  - channels=[1024,1024,1024,1024,3072], kernels=[5,3,3,3,1], dilations=[1,2,3,4,1]
  - 107 language classes
  - 21M params, ~84 MB F32

Usage:
  python models/convert-ecapa-tdnn-lid-to-gguf.py \
      --input speechbrain/lang-id-voxlingua107-ecapa \
      --output ecapa-lid-107.gguf
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


def main():
    parser = argparse.ArgumentParser(description="Convert ECAPA-TDNN LID to GGUF")
    parser.add_argument("--input", required=True, help="HF model ID or local path")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    args = parser.parse_args()

    import torch
    from huggingface_hub import hf_hub_download

    # Download model files
    if os.path.isdir(args.input):
        base = args.input
    else:
        for f in ["embedding_model.ckpt", "classifier.ckpt", "label_encoder.txt"]:
            hf_hub_download(args.input, f)
        base = os.path.dirname(hf_hub_download(args.input, "embedding_model.ckpt"))

    emb_ckpt = torch.load(os.path.join(base, "embedding_model.ckpt"), map_location="cpu", weights_only=False)
    cls_ckpt = torch.load(os.path.join(base, "classifier.ckpt"), map_location="cpu", weights_only=False)

    # Read language labels — two formats:
    # VoxLingua107: 'en: English' => 20
    # CommonLanguage: 'English' => 5
    labels = []
    label_path = os.path.join(base, "label_encoder.txt")
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=>" not in line or line.startswith("="):
                continue
            label_str = line.split("=>")[0].strip().strip("'")
            if label_str == "starting_index":
                continue
            if ": " in label_str:
                # VoxLingua107 format: 'en: English' → extract ISO code
                code = label_str.split(":")[0].strip()
            else:
                # CommonLanguage format: 'English' → use as-is
                code = label_str
            labels.append(code)

    # Detect n_mels from hyperparams
    n_mels = 60  # VoxLingua107 default
    hp_path = os.path.join(base, "hyperparams.yaml")
    if os.path.exists(hp_path):
        with open(hp_path) as f:
            for line in f:
                if line.strip().startswith("n_mels:") and not line.strip().startswith("n_mels: !"):
                    n_mels = int(line.split(":")[1].strip())
                    break

    # Detect classifier type
    cls_type = "cosine" if "weight" in cls_ckpt and len(cls_ckpt) <= 2 else "dnn"

    # Detect lin_neurons from embedding model FC layer
    fc_key = [k for k in emb_ckpt if "fc.conv" in k and "weight" in k]
    lin_neurons = 256
    if fc_key:
        lin_neurons = emb_ckpt[fc_key[0]].shape[0]

    print(f"ECAPA-TDNN LID: {len(emb_ckpt)} emb tensors, {len(cls_ckpt)} cls tensors, {len(labels)} labels")
    print(f"  n_mels={n_mels}, lin_neurons={lin_neurons}, cls_type={cls_type}")

    # Compute the exact SpeechBrain filterbank matrix for embedding in GGUF.
    # This avoids fbank mismatch at runtime.
    try:
        import sys, types
        ext = types.ModuleType('torchaudio._extension')
        ext._IS_TORCHAUDIO_EXT_AVAILABLE = False
        for attr in ['fail_if_no_align','fail_if_no_rnnt','fail_if_no_sox','fail_if_no_kaldi',
                      '_check_cuda_version','_init_dll_path']:
            setattr(ext, attr, lambda *a,**kw: None)
        sys.modules['torchaudio._extension'] = ext
        import torchaudio
        from speechbrain.processing.features import Filterbank
        # Extract the LINEAR filterbank matrix by running with log_mel=False and identity input
        sb_fb = Filterbank(n_mels=n_mels, n_fft=400, sample_rate=16000, freeze=True, log_mel=False)
        identity = torch.eye(201).unsqueeze(0)
        fb_out = sb_fb(identity)  # [1, 201, 60]
        fb_matrix = fb_out[0].detach().numpy().T.astype(np.float32)  # [60, 201]
        print(f"  SpeechBrain filterbank matrix: {fb_matrix.shape}")
    except Exception as e:
        print(f"  WARNING: could not compute SpeechBrain filterbank: {e}")
        fb_matrix = None

    # Create GGUF
    model_name = os.path.basename(args.input) if os.path.isdir(args.input) else args.input.split("/")[-1]
    writer = gguf.GGUFWriter(args.output, "ecapa-tdnn-lid")
    writer.add_name(f"ECAPA-TDNN-LID-{model_name}")

    # Hyperparameters
    writer.add_uint32("ecapa.n_mels", n_mels)
    writer.add_uint32("ecapa.n_classes", len(labels))
    writer.add_uint32("ecapa.channels_0", 1024)
    writer.add_uint32("ecapa.channels_mfa", 3072)
    writer.add_uint32("ecapa.attention_channels", 128)
    writer.add_uint32("ecapa.lin_neurons", lin_neurons)
    writer.add_uint32("ecapa.n_se_res2net_blocks", 3)
    writer.add_uint32("ecapa.res2net_scale", 8)
    writer.add_uint32("ecapa.cls_type", 0 if cls_type == "dnn" else 1)  # 0=DNN, 1=cosine

    # Tokenizer (language labels)
    writer.add_array("tokenizer.ggml.tokens", labels)

    # Embed the SpeechBrain filterbank matrix so C++ runtime doesn't need to recompute it
    if fb_matrix is not None:
        writer.add_tensor("mel_filterbank", fb_matrix)
        writer.add_uint32("ecapa.n_fft", 400)
        writer.add_uint32("ecapa.fbank_bins", fb_matrix.shape[1])  # 201

    def f32(t):
        return t.float().numpy()

    def f16(t):
        return t.float().numpy().astype(np.float16)

    # Write embedding model tensors
    tensor_count = 0
    for name, tensor in sorted(emb_ckpt.items()):
        if "num_batches_tracked" in name:
            continue  # Skip BN tracking counters
        t = tensor.float().numpy()

        # Shorten names
        gguf_name = "emb." + name
        gguf_name = gguf_name.replace(".conv.conv.", ".conv.")
        gguf_name = gguf_name.replace(".norm.norm.", ".bn.")

        if len(gguf_name) >= 64:
            print(f"  WARNING: name too long ({len(gguf_name)}): {gguf_name}")
            # Further shorten
            gguf_name = gguf_name.replace("res2net_block.blocks.", "r2n.")
            gguf_name = gguf_name.replace("se_block.", "se.")

        # Squeeze kernel=1 conv weights
        if len(t.shape) == 3 and t.shape[-1] == 1:
            t = t.squeeze(-1)

        # Store norms/biases as F32, weights as F16
        if "bn." in gguf_name or "bias" in name or len(t.shape) <= 1:
            data = f32(tensor)
        else:
            data = f16(tensor)

        writer.add_tensor(gguf_name, data)
        tensor_count += 1
        if tensor_count <= 5 or tensor_count % 20 == 0:
            print(f"  [{tensor_count}] {gguf_name:55s} {str(data.shape):20s}")

    # Write classifier tensors
    for name, tensor in sorted(cls_ckpt.items()):
        if "num_batches_tracked" in name:
            continue
        t = tensor.float().numpy()
        gguf_name = "cls." + name
        gguf_name = gguf_name.replace(".norm.norm.", ".bn.")
        gguf_name = gguf_name.replace(".linear.w.", ".linear.")

        if "bn." in gguf_name or "bias" in name or len(t.shape) <= 1:
            data = f32(tensor)
        else:
            data = f16(tensor)

        writer.add_tensor(gguf_name, data)
        tensor_count += 1
        print(f"  [{tensor_count}] {gguf_name:55s} {str(data.shape):20s}")

    print(f"  ... total: {tensor_count} tensors")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    file_size = os.path.getsize(args.output)
    print(f"\nDone: {args.output} ({file_size / 1e6:.1f} MB, {tensor_count} tensors)")


if __name__ == "__main__":
    main()
