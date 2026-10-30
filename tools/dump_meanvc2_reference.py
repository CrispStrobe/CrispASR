#!/usr/bin/env python3
"""
MeanVC2 reference activation dumper for CrispASR diff harness.

Runs the MeanVC2 PyTorch pipeline (WeNet Fast-U2++ ASR, GTM, TVT, DiT, Vocos)
on a known audio sample with deterministic random seed, capturing intermediate
tensors at every architectural stage and writing them to a reference GGUF.

Usage:
    python tools/dump_meanvc2_reference.py \\
        --model-dir /path/to/meanvc2_dir \\
        --audio samples/jfk.wav \\
        --output /path/to/meanvc2-ref.gguf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFWriter
except ImportError:
    sys.exit("pip install gguf")

try:
    import torch
    import torchaudio.compliance.kaldi as kaldi
    import soundfile as sf
    from safetensors.torch import load_file
except ImportError:
    sys.exit("pip install torch torchaudio soundfile safetensors")

# Include MeanVC2 runtime
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, "/tmp/meanvc2_repo/runtime")
from src.dit import DiT


def write_f32_tensor(writer: GGUFWriter, name: str, data: Any):
    """Write float32 tensor to reference GGUF."""
    if isinstance(data, torch.Tensor):
        data = data.detach().float().cpu().numpy()
    writer.add_tensor(name, np.ascontiguousarray(data, dtype=np.float32))


def main():
    parser = argparse.ArgumentParser(description="MeanVC2 reference dumper for CrispASR")
    parser.add_argument("--model-dir", type=Path, required=True, help="Path containing meanvc2 checkpoints")
    parser.add_argument("--audio", type=Path, default=Path("samples/jfk.wav"), help="Input source audio")
    parser.add_argument("--samples", type=int, default=25600, help="Number of samples to process (1.6s @ 16kHz)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--output", type=Path, required=True, help="Output GGUF path")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading audio from {args.audio}...")
    wav, sr = sf.read(str(args.audio), dtype="int16")
    wav = wav.astype("float32") / 32768.0
    if len(wav) > args.samples:
        wav = wav[:args.samples]

    print(f"Audio slice length: {len(wav)} samples")

    # 1. Kaldi fbank
    fbanks = kaldi.fbank(
        torch.from_numpy(wav * 32768.0).unsqueeze(0),
        snip_edges=True,
        num_mel_bins=80,
        energy_floor=0.0,
        dither=0.0,
        sample_frequency=16000,
    ).squeeze(0).float()
    print(f"Kaldi fbanks: shape={fbanks.shape}")

    # 2. Fast-U2++ ASR BNF extraction
    asr_path = args.model_dir / "fastu2pp_160ms.pt"
    print(f"Loading Fast-U2++ ASR from {asr_path}...")
    asr = torch.jit.load(str(asr_path), map_location="cpu")
    asr.eval()

    offset = 8
    req_cache = 8
    att_cache = torch.zeros(6, 4, req_cache, 128, dtype=torch.float32)
    cnn_cache = torch.zeros(6, 1, 256, 8, dtype=torch.float32)

    window = 19
    stride = 16
    bns = []
    i = 0
    T = fbanks.shape[0]
    with torch.no_grad():
        while i + window <= T:
            fb = fbanks[i : i + window].unsqueeze(0).float()
            out, att_cache, cnn_cache = asr(
                fb,
                torch.tensor(offset, dtype=torch.int64),
                torch.tensor(req_cache, dtype=torch.int64),
                att_cache,
                cnn_cache,
            )
            bns.append(out.squeeze(0).detach())
            offset += 4
            i += stride

    bn_cat = torch.cat(bns, dim=0)  # [T_bn, 256]
    print(f"BN features: shape={bn_cat.shape}")

    # 3. 4x linear interpolation of BNF
    bn_t = bn_cat.unsqueeze(0).transpose(1, 2)
    bn_interp = torch.nn.functional.interpolate(
        bn_t, size=int(bn_t.shape[2] * 4), mode="linear", align_corners=True
    ).transpose(1, 2)  # [1, T_mel, 256]
    print(f"BN interpolated: shape={bn_interp.shape}")

    # 4. DiT model
    config_path = args.model_dir / "config_120ms_40ms.json"
    dit_ckpt = args.model_dir / "meanvc2_120ms_40ms.safetensors"
    print(f"Loading DiT from {dit_ckpt} with config {config_path}...")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    dit_model = DiT(**cfg["model"])
    state_dict = load_file(str(dit_ckpt))
    dit_model.load_state_dict(state_dict, strict=False)
    dit_model.eval()

    # Deterministic speaker embedding for reproducibility
    torch.manual_seed(args.seed)
    spks = torch.randn(1, 256)
    spks = spks / torch.norm(spks)

    with torch.no_grad():
        k_mem, v_mem = dit_model.gtm(spks)
    print(f"GTM memory: k={k_mem.shape}, v={v_mem.shape}")

    # 5. DiT 1-NFE Euler solve
    seq_len = bn_interp.shape[1]
    chunk_size = 12
    block_size = 4
    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    x_pred_collect = []
    offset_val = 0
    kv_cache = None

    torch.manual_seed(args.seed + 1)
    noise = torch.randn(1, seq_len, 80)

    with torch.no_grad():
        for chunk_id in range(num_chunks):
            start = chunk_id * chunk_size
            mid = min(start + chunk_size, seq_len)
            end = min(mid + block_size, seq_len)
            curr_len = mid - start
            cond_in = torch.cat([bn_interp[:, start:mid], bn_interp[:, mid:end]], dim=1)
            x = noise[:, start:end]
            t = torch.tensor([1.0])
            r = torch.tensor([0.0])
            u, tmp_kv = dit_model(
                x,
                t,
                r,
                cond=cond_in,
                spks=spks,
                offset=offset_val,
                is_inference=True,
                kv_cache=kv_cache,
            )
            x = x - (1.0 - 0.0) * u
            kv_cache = tmp_kv
            x_pred_collect.append(x[:, :curr_len])
            offset_val += curr_len

    x_pred = torch.cat(x_pred_collect, dim=1)
    mel = (x_pred.transpose(1, 2) + 1.0) / 2.0  # [1, 80, T_mel]
    print(f"Generated mel: shape={mel.shape}")

    # 6. Vocos decode
    vocos_path = args.model_dir / "vocos.pt"
    print(f"Loading Vocos from {vocos_path}...")
    vocos = torch.jit.load(str(vocos_path), map_location="cpu")
    vocos.eval()
    with torch.no_grad():
        wav_out = vocos.decode(mel)
    print(f"Vocos audio: shape={wav_out.shape}")

    # 7. Write reference GGUF
    print(f"Writing reference GGUF to {args.output}...")
    writer = GGUFWriter(args.output, arch="crispasr.reference")
    writer.add_string("general.name", "meanvc2_reference")

    write_f32_tensor(writer, "raw_audio", wav)
    write_f32_tensor(writer, "kaldi_fbank", fbanks)
    write_f32_tensor(writer, "asr_bn", bn_cat)
    write_f32_tensor(writer, "asr_bn_interp", bn_interp.squeeze(0))
    write_f32_tensor(writer, "spk_emb", spks.squeeze(0))
    write_f32_tensor(writer, "gtm_k", k_mem.squeeze(0))
    write_f32_tensor(writer, "gtm_v", v_mem.squeeze(0))
    write_f32_tensor(writer, "noise", noise.squeeze(0))
    write_f32_tensor(writer, "dit_mel", mel.squeeze(0))
    write_f32_tensor(writer, "vocos_pcm", wav_out.squeeze(0))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Successfully dumped MeanVC2 reference to {args.output}!")


if __name__ == "__main__":
    main()
