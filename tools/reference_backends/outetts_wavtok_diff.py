#!/usr/bin/env python3
"""
WavTokenizer decoder diff harness -- dump per-stage intermediates from
the Python reference decoder and compare against C++ output.

Usage:
  # 1. Dump Python reference stages:
  python tools/reference_backends/outetts_wavtok_diff.py dump \
      --decoder-dir /path/to/wavtokenizer-large-75token-interface/decoder \
      --codes "0,1,2,3,4,5,6,7,8,9" \
      --out /mnt/storage/outetts/ref_stages/

  # 2. Run C++ decoder with --dump (not yet implemented):
  # [would dump matching .bin files at each pipeline stage]

  # 3. Compare:
  python tools/reference_backends/outetts_wavtok_diff.py compare \
      --ref /mnt/storage/outetts/ref_stages/ \
      --cpp /mnt/storage/outetts/cpp_stages/
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    sys.exit("pip install torch")


def dump_stages(args):
    decoder_dir = Path(args.decoder_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    codes = [int(c.strip()) for c in args.codes.split(",")]
    codes_t = torch.tensor([codes], dtype=torch.long)
    print(f"Input codes: {codes} (T={len(codes)})")

    # Load checkpoint
    pt_path = decoder_dir / "decoder_model.pt"
    raw = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    sd = raw["model_state_dict"]
    codebook = raw["codebook_weights"]  # (4096, 512)

    # Stage 1: codebook lookup
    features = codebook[codes_t.squeeze()]  # (T, 512)
    features = features.T.unsqueeze(0)       # (1, 512, T)
    save(out_dir / "01_codebook.npy", features)

    # Stage 2: conv_pre
    conv_w = sd["backbone.embed.weight"]  # (768, 512, 7)
    conv_b = sd["backbone.embed.bias"]    # (768,)
    x = F.conv1d(features.float(), conv_w.float(), conv_b.float(), padding=3)
    save(out_dir / "02_conv_pre.npy", x)

    # Stage 4: pos_net blocks (BEFORE AdaNorm — matches VocosBackbone.forward order)
    # 0,1: ResNet; 2: SelfAttn; 3,4: ResNet; 5: GroupNorm
    resnet_indices = [0, 1, 3, 4]
    for ri in resnet_indices:
        residual = x
        n1w = sd[f"backbone.pos_net.{ri}.norm1.weight"]
        n1b = sd[f"backbone.pos_net.{ri}.norm1.bias"]
        n2w = sd[f"backbone.pos_net.{ri}.norm2.weight"]
        n2b = sd[f"backbone.pos_net.{ri}.norm2.bias"]
        c1w = sd[f"backbone.pos_net.{ri}.conv1.weight"]
        c1b = sd[f"backbone.pos_net.{ri}.conv1.bias"]
        c2w = sd[f"backbone.pos_net.{ri}.conv2.weight"]
        c2b = sd[f"backbone.pos_net.{ri}.conv2.bias"]

        h = F.layer_norm(x.squeeze(0).T, [768], n1w, n1b).T.unsqueeze(0)
        h = F.conv1d(h.float(), c1w.float(), c1b.float(), padding=1)
        h = F.gelu(h)
        h = F.layer_norm(h.squeeze(0).T, [768], n2w, n2b).T.unsqueeze(0)
        h = F.conv1d(h.float(), c2w.float(), c2b.float(), padding=1)
        h = F.gelu(h)
        x = h + residual
        save(out_dir / f"04_posnet_resnet{ri}.npy", x)

        if ri == 1:
            # Self-attention at index 2
            residual = x
            nw = sd["backbone.pos_net.2.norm.weight"]
            nb = sd["backbone.pos_net.2.norm.bias"]
            h = F.layer_norm(x.squeeze(0).T, [768], nw, nb).T.unsqueeze(0)
            qw = sd["backbone.pos_net.2.q.weight"].float()
            qb = sd["backbone.pos_net.2.q.bias"].float()
            kw = sd["backbone.pos_net.2.k.weight"].float()
            kb = sd["backbone.pos_net.2.k.bias"].float()
            vw = sd["backbone.pos_net.2.v.weight"].float()
            vb = sd["backbone.pos_net.2.v.bias"].float()
            pw = sd["backbone.pos_net.2.proj_out.weight"].float()
            pb = sd["backbone.pos_net.2.proj_out.bias"].float()

            q = F.conv1d(h.float(), qw, qb)
            k = F.conv1d(h.float(), kw, kb)
            v = F.conv1d(h.float(), vw, vb)

            C = q.shape[1]
            attn = torch.bmm(q.transpose(1, 2), k) / (C ** 0.5)
            attn = F.softmax(attn, dim=-1)
            out = torch.bmm(v, attn.transpose(1, 2))
            out = F.conv1d(out, pw, pb)
            x = out + residual
            save(out_dir / "04_posnet_selfattn2.npy", x)

    # GroupNorm at pos_net.5
    gnw = sd["backbone.pos_net.5.weight"]
    gnb = sd["backbone.pos_net.5.bias"]
    x_n = F.layer_norm(x.squeeze(0).T, [768], gnw, gnb).T.unsqueeze(0)
    x = x_n
    save(out_dir / "05_posnet_gnorm.npy", x)

    # Stage 3: backbone AdaNorm (AFTER pos_net — matches VocosBackbone.forward)
    scale = sd["backbone.norm.scale.weight"][0]  # (768,)
    shift = sd["backbone.norm.shift.weight"][0]
    x_n = F.layer_norm(x.squeeze(0).T, [768]).T.unsqueeze(0)
    x = x_n * scale.view(1, -1, 1) + shift.view(1, -1, 1)
    save(out_dir / "03_backbone_adanorm.npy", x)

    # Stage 5: ConvNeXt blocks
    for i in range(12):
        residual = x
        # DW conv
        dw_w = sd[f"backbone.convnext.{i}.dwconv.weight"].float()  # (768, 1, 7)
        dw_b = sd[f"backbone.convnext.{i}.dwconv.bias"].float()
        h = F.conv1d(x.float(), dw_w, dw_b, padding=3, groups=768)

        # AdaNorm (bw_id=0)
        sc = sd[f"backbone.convnext.{i}.norm.scale.weight"][0].float()
        sh = sd[f"backbone.convnext.{i}.norm.shift.weight"][0].float()
        h = F.layer_norm(h.squeeze(0).T, [768]).T.unsqueeze(0)
        h = h * sc.view(1, -1, 1) + sh.view(1, -1, 1)

        # PW up + GELU
        up_w = sd[f"backbone.convnext.{i}.pwconv1.weight"].float()
        up_b = sd[f"backbone.convnext.{i}.pwconv1.bias"].float()
        h = h.squeeze(0).T  # (T, 768)
        h = F.linear(h, up_w, up_b)  # (T, 2304)
        h = F.gelu(h)

        # PW down
        dn_w = sd[f"backbone.convnext.{i}.pwconv2.weight"].float()
        dn_b = sd[f"backbone.convnext.{i}.pwconv2.bias"].float()
        h = F.linear(h, dn_w, dn_b)  # (T, 768)

        # Gamma gate
        gamma = sd[f"backbone.convnext.{i}.gamma"].float()
        h = h * gamma

        h = h.T.unsqueeze(0)  # (1, 768, T)
        x = h + residual

        if i == 0 or i == 11:
            save(out_dir / f"06_convnext_{i}.npy", x)

    # Final LayerNorm
    fnw = sd["backbone.final_layer_norm.weight"]
    fnb = sd["backbone.final_layer_norm.bias"]
    x = F.layer_norm(x.squeeze(0).T, [768], fnw, fnb).T.unsqueeze(0)
    save(out_dir / "07_final_norm.npy", x)

    # ISTFT head
    hw = sd["head.out.weight"].float()  # (1282, 768)
    hb = sd["head.out.bias"].float()
    stft = F.linear(x.squeeze(0).T, hw, hb)  # (T, 1282)
    save(out_dir / "08_istft_head.npy", stft)

    # Split mag/phase and reconstruct
    n_freq = 641
    mag = torch.exp(stft[:, :n_freq]).clamp(max=1e2)  # ISTFTHead safeguard
    phase = stft[:, n_freq:]
    save(out_dir / "09_mag.npy", mag)
    save(out_dir / "09_phase.npy", phase)

    # iSTFT using WavTokenizer's padding="same" (not center=True)
    # padding="same" trims (win_length - hop_length) // 2 from each end
    n_fft_val, hop_val = 1280, 320
    window = sd.get("head.istft.window", torch.hann_window(n_fft_val)).float()
    S = mag * torch.exp(1j * phase)  # (T, n_freq) -> need (n_freq, T)
    S_3d = S.T.unsqueeze(0)  # (1, n_freq, T)
    # Use torch.fft.irfft + manual overlap-add to match WavTokenizer's custom ISTFT
    ifft = torch.fft.irfft(S_3d, n_fft_val, dim=1, norm="backward")  # (1, n_fft, T)
    ifft = ifft * window[None, :, None]
    T_frames = ifft.shape[2]
    output_size = (T_frames - 1) * hop_val + n_fft_val
    y = torch.nn.functional.fold(
        ifft, output_size=(1, output_size),
        kernel_size=(1, n_fft_val), stride=(1, hop_val)
    )[:, 0, 0, :]
    pad = (n_fft_val - hop_val) // 2
    y = y[:, pad:-pad]
    # Window envelope normalization
    window_sq = window.square().expand(1, T_frames, -1).transpose(1, 2)
    window_envelope = torch.nn.functional.fold(
        window_sq, output_size=(1, output_size),
        kernel_size=(1, n_fft_val), stride=(1, hop_val)
    ).squeeze()[pad:-pad]
    audio = y / window_envelope
    save(out_dir / "10_audio.npy", audio)

    print(f"\nAll stages dumped to {out_dir}")
    print(f"Audio shape: {audio.shape}, range [{audio.min():.4f}, {audio.max():.4f}]")


def save(path, t):
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().float().numpy()
    np.save(str(path), t)
    print(f"  {path.name}: {t.shape} range [{t.min():.6f}, {t.max():.6f}]")


def compare(args):
    ref_dir = Path(args.ref)
    cpp_dir = Path(args.cpp)

    for ref_path in sorted(ref_dir.glob("*.npy")):
        cpp_path = cpp_dir / ref_path.name.replace(".npy", ".bin")
        if not cpp_path.exists():
            cpp_path = cpp_dir / ref_path.name
        if not cpp_path.exists():
            print(f"  {ref_path.name}: MISSING in C++ output")
            continue

        ref = np.load(str(ref_path)).flatten()
        if cpp_path.suffix == ".bin":
            cpp = np.fromfile(str(cpp_path), dtype=np.float32)
        else:
            cpp = np.load(str(cpp_path)).flatten()

        if len(ref) != len(cpp):
            print(f"  {ref_path.name}: SIZE MISMATCH ref={len(ref)} cpp={len(cpp)}")
            continue

        cos = np.dot(ref, cpp) / (np.linalg.norm(ref) * np.linalg.norm(cpp) + 1e-12)
        mae = np.abs(ref - cpp).max()
        print(f"  {ref_path.name}: cos={cos:.6f}  max_abs_err={mae:.6f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    dp = sub.add_parser("dump", help="Dump reference stages")
    dp.add_argument("--decoder-dir", required=True)
    dp.add_argument("--codes", default="0,1,2,3,4,5,6,7,8,9")
    dp.add_argument("--out", required=True)

    cp = sub.add_parser("compare", help="Compare ref vs C++ stages")
    cp.add_argument("--ref", required=True)
    cp.add_argument("--cpp", required=True)

    args = ap.parse_args()
    if args.cmd == "dump":
        dump_stages(args)
    elif args.cmd == "compare":
        compare(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
