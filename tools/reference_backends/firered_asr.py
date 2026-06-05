"""FireRedASR-AED reference dump backend.

Uses the official `fireredasr` Python library and `kaldi_native_fbank` to
capture the Kaldi log-mel fbank features + CMVN and the full encoder output
for stage-by-stage comparison against the C++ runtime.

Stages:
  raw_audio       (N,)            input PCM
  mel_spectrogram (n_frames, 80)  Kaldi fbank + CMVN — matches firered_asr_compute_fbank
  encoder_output  (T_enc, 1280)   Conformer encoder — matches firered_asr_run_encoder

`model_dir` may be:
  - A HuggingFace id:  "FireRedTeam/FireRedASR2-AED"
  - A local directory: "/path/to/FireRedASR2-AED" (must contain model.pth.tar
    and cmvn.ark in the same layout as the HF snapshot)

Usage:
  python tools/dump_reference.py --backend firered-asr \\
      --model-dir FireRedTeam/FireRedASR2-AED \\
      --audio samples/jfk.wav \\
      --output /mnt/storage/ref/firered-asr2-aed-ref.gguf
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Dict, Set

import numpy as np

DEFAULT_STAGES = ["raw_audio", "mel_spectrogram", "encoder_output"]


def _parse_cmvn_ark(path: Path) -> tuple:
    """Parse a Kaldi binary CMVN .ark file; returns (mean, std) as float32 arrays."""
    with open(path, "rb") as f:
        data = f.read()
    for i in range(len(data) - 3):
        if data[i : i + 3] in (b"BDM", b"BFM"):
            mtype = data[i : i + 3]
            idx = i + 3
            if data[idx] == 0x20:
                idx += 1
            assert data[idx] == 4
            idx += 1
            rows = struct.unpack("<i", data[idx : idx + 4])[0]
            idx += 4
            assert data[idx] == 4
            idx += 1
            cols = struct.unpack("<i", data[idx : idx + 4])[0]
            idx += 4
            elem_size = 8 if mtype == b"BDM" else 4
            dtype = "<f8" if mtype == b"BDM" else "<f4"
            vals = np.frombuffer(data[idx : idx + rows * cols * elem_size],
                                  dtype=dtype).reshape(rows, cols)
            count = vals[0, -1]
            mean = (vals[0, :-1] / count).astype(np.float32)
            var = vals[1, :-1] / count - mean.astype(np.float64) ** 2
            std = np.sqrt(np.maximum(var, 1e-10)).astype(np.float32)
            return mean, std
    raise ValueError(f"could not parse CMVN ark: {path}")


def dump(*, model_dir: Path, audio: np.ndarray, stages: Set[str],
         max_new_tokens: int) -> Dict[str, np.ndarray]:
    """Run FireRedASR-AED reference forward pass and return stage captures."""
    import torch

    # Resolve model directory
    model_dir_s = str(model_dir)
    if os.path.isdir(model_dir_s):
        base = model_dir_s
    else:
        # HF id — download via hub
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(model_dir_s, "model.pth.tar")
        base = os.path.dirname(p)
        for fname in ["dict.txt", "train_bpe1000.model", "cmvn.ark"]:
            hf_hub_download(model_dir_s, fname)

    print(f"  loading FireRedASR model from {base}")
    try:
        from fireredasr.models.fireredasr_aed import FireRedAsrAed
    except ImportError as e:
        raise SystemExit(
            "fireredasr library required.\n"
            "Install: pip install fireredasr\n"
            f"(import error: {e})")
    try:
        import kaldi_native_fbank as knf
    except ImportError as e:
        raise SystemExit(
            "kaldi_native_fbank required.\n"
            "Install: pip install kaldi_native_fbank\n"
            f"(import error: {e})")

    ckpt = torch.load(os.path.join(base, "model.pth.tar"),
                      map_location="cpu", weights_only=False)
    model = FireRedAsrAed(ckpt["args"])
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    out: Dict[str, np.ndarray] = {}

    if "raw_audio" in stages:
        out["raw_audio"] = audio.astype(np.float32)

    # --- Kaldi fbank features ---
    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = 16000
    opts.frame_opts.frame_length_ms = 25.0
    opts.frame_opts.frame_shift_ms = 10.0
    opts.mel_opts.num_bins = 80
    fbank_comp = knf.OnlineFbank(opts)
    # Scale to int16 range: FireRedASR CMVN was trained on int16-scaled fbank.
    fbank_comp.accept_waveform(16000, (audio * 32768.0).tolist())
    fbank_comp.input_finished()
    n_frames = fbank_comp.num_frames_ready
    features = np.zeros((n_frames, 80), dtype=np.float32)
    for i in range(n_frames):
        features[i] = fbank_comp.get_frame(i)

    # CMVN
    cmvn_path = os.path.join(base, "cmvn.ark")
    mean, std = _parse_cmvn_ark(Path(cmvn_path))
    features = (features - mean) / std

    if "mel_spectrogram" in stages:
        # (n_frames, 80) matching firered_asr_compute_fbank
        out["mel_spectrogram"] = features.astype(np.float32)

    if "encoder_output" in stages:
        padded = torch.from_numpy(features).unsqueeze(0)  # (1, T, 80)
        lengths = torch.tensor([n_frames], dtype=torch.long)
        with torch.no_grad():
            enc_out, _, _ = model.encoder(padded, lengths)
        # enc_out: (1, T_enc, 1280). Squeeze batch → (T_enc, 1280).
        out["encoder_output"] = enc_out[0].float().numpy()

    return out
