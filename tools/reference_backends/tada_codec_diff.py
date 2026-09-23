"""Standalone TADA codec diff harness — VPS-friendly (codec only, not the 3B LLM).

Loads only HumeAI/tada-codec/decoder (~200 MB), runs a forward pass on
acoustic features dumped by the C++ binary, and emits per-stage stats for
comparison against C++'s dump_* tensors printed to stderr.

Usage
-----
# 1. Build crispasr with TADA enabled and run to dump features:
#    TADA_DUMP_FEATURES=/mnt/volume1/tmp-overflow/tada_features.bin \
#      ./build/crispasr-cli --model tada-tts-3b-ml-q4_k.gguf \
#        --model-codec tada-codec-f16.gguf --synth "Please call Stella."
#
# 2. Run this script on the VPS:
#    TADA_CODEC_DIR=/mnt/storage/tada-codec \
#      python tools/reference_backends/tada_codec_diff.py \
#        --features /mnt/volume1/tmp-overflow/tada_features.bin \
#        [--no-denorm]   # if C++ features are already denormalized
"""

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np


def load_features(path: str) -> np.ndarray:
    """Load C++-dumped features from a simple binary: int32 n_frames, int32 dim, float32 data."""
    with open(path, "rb") as f:
        n_frames, dim = struct.unpack("<II", f.read(8))
        data = np.frombuffer(f.read(n_frames * dim * 4), dtype=np.float32).reshape(n_frames, dim)
    print(f"  features: {n_frames} frames × {dim} dims  rms={np.sqrt(np.mean(data**2)):.4f}")
    return data


def run_python_decoder(features_np: np.ndarray, codec_dir: str) -> dict:
    """Run the TADA Python decoder and return per-stage activations."""
    import torch
    torch.set_grad_enabled(False)

    from tada.modules.decoder import Decoder

    print(f"  loading Decoder from {codec_dir}")
    dec = Decoder.from_pretrained(codec_dir, subfolder="decoder")
    dec.eval()

    # features_np: (T, 512) float32
    feat = torch.from_numpy(features_np).float().unsqueeze(0)  # (1, T, 512)

    stages = {}

    # ── Hook intermediates ──
    handles = []

    def _hook(name):
        def fn(module, inp, out):
            arr = out[0] if isinstance(out, (tuple, list)) else out
            stages[name] = arr.detach().cpu().float().squeeze(0).numpy()
        return fn

    # Try to hook into the decoder layers. The exact names depend on the
    # hume-tada package version — check with `print(dec)` if unsure.
    # Common layout:
    #   dec.proj                    (decoder_proj)
    #   dec.attn_dec.layers[0..5]   (local attention encoder layers)
    #   dec.attn_dec.final_norm     (final norm)
    #   dec.wav_dec                 (DACDecoder)
    try:
        if hasattr(dec, "proj"):
            handles.append(dec.proj.register_forward_hook(_hook("proj_out")))
        if hasattr(dec, "attn_dec"):
            attn = dec.attn_dec
            if hasattr(attn, "layers"):
                for i, layer in enumerate(attn.layers):
                    if i == 0:
                        handles.append(layer.register_forward_hook(_hook("attn_layer0")))
                if hasattr(attn, "final_norm"):
                    handles.append(attn.final_norm.register_forward_hook(_hook("attn_final_norm")))
        if hasattr(dec, "wav_dec"):
            handles.append(dec.wav_dec.register_forward_hook(_hook("wav_dec_out")))
    except Exception as e:
        print(f"  WARN: hook registration failed: {e}")

    with torch.no_grad():
        try:
            pcm = dec(feat)  # try positional
        except TypeError:
            try:
                pcm = dec.decode(feat)
            except Exception:
                pcm = dec.forward(feat)

    for h in handles:
        h.remove()

    if isinstance(pcm, (tuple, list)):
        pcm = pcm[0]
    pcm_np = pcm.detach().cpu().float().squeeze().numpy()
    stages["pcm"] = pcm_np

    return stages


def compare(cpp_label: str, py_arr: np.ndarray, ref_rms: float | None = None) -> None:
    rms = float(np.sqrt(np.mean(py_arr.astype(float) ** 2)))
    mn, mx = float(py_arr.min()), float(py_arr.max())
    print(f"  PY  {cpp_label:24s}: n={py_arr.size:7d}  rms={rms:.4f}  range=[{mn:.4f}, {mx:.4f}]")
    if ref_rms is not None:
        print(f"       {'→ expect':24s}: rms≈{ref_rms:.4f}  (from C++ dump_)")


def main() -> None:
    ap = argparse.ArgumentParser(description="TADA codec diff harness")
    ap.add_argument("--features", required=True, help="Binary file of expanded acoustic features from C++")
    ap.add_argument("--no-denorm", action="store_true",
                    help="Features are already denormalized (C++ applied acoustic_std/mean); "
                         "pass raw to Python without re-scaling")
    ap.add_argument("--acoustic-std", type=float, default=1.5,
                    help="acoustic_std from the GGUF metadata (default 1.5)")
    ap.add_argument("--acoustic-mean", type=float, default=0.0,
                    help="acoustic_mean from the GGUF metadata (default 0.0)")
    args = ap.parse_args()

    codec_dir = os.environ.get("TADA_CODEC_DIR", "HumeAI/tada-codec")
    print(f"TADA codec diff harness")
    print(f"  features file : {args.features}")
    print(f"  codec dir     : {codec_dir}")

    features = load_features(args.features)

    # The Python decoder may expect normalized OR denormalized features.
    # C++ currently denormalizes before passing to the codec.
    # Try denormalized first (what C++ sends); if output is noise, try normalized.
    if args.no_denorm:
        # Reverse the C++ denorm to get back to normalized space
        feats_for_py = (features - args.acoustic_mean) / args.acoustic_std
        print(f"  reversing denorm: features / {args.acoustic_std}")
    else:
        feats_for_py = features
        print(f"  passing features as-is (denormalized by C++)")

    print()
    try:
        stages = run_python_decoder(feats_for_py, codec_dir)
    except Exception as e:
        print(f"ERROR: Python decoder failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    print()
    print("── Python codec intermediate stats ──")
    print("(compare against C++ 'DUMP <name>:' lines in stderr)")
    print()

    mapping = [
        ("proj_out",         "dump_proj"),
        ("attn_layer0",      "dump_layer0"),
        ("attn_final_norm",  "dump_attn"),
        ("wav_dec_out",      "dump_dac_out"),
        ("pcm",              "pcm"),
    ]
    for py_key, cpp_label in mapping:
        if py_key in stages:
            compare(cpp_label, stages[py_key])

    pcm = stages.get("pcm")
    if pcm is not None:
        rms = float(np.sqrt(np.mean(pcm.astype(float) ** 2)))
        print()
        print(f"Final PCM: {len(pcm)} samples  ({len(pcm)/24000:.2f}s @ 24kHz)  rms={rms:.6f}")
        if rms < 1e-4:
            print("  ⚠ RMS near zero — silence or degenerate output")
        elif rms > 0.3:
            print("  ⚠ RMS very high — possible noise/clipping")
        else:
            print("  ✓ RMS in normal speech range")

        # Save WAV for ASR roundtrip
        wav_path = os.environ.get("TADA_WAV_OUTPUT", "/mnt/volume1/tmp-overflow/tada_codec_diff.wav")
        try:
            import wave
            pcm16 = (pcm * 32767).clip(-32767, 32767).astype(np.int16)
            with wave.open(wav_path, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(pcm16.tobytes())
            print(f"  WAV: {wav_path}")
        except Exception as e:
            print(f"  WARN: could not write WAV: {e}")


if __name__ == "__main__":
    main()
