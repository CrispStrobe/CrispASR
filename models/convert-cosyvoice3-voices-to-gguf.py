#!/usr/bin/env python3
"""Bake CosyVoice3 voice-clone artifacts into a single GGUF blob.

Per voice, the runtime needs four prebaked tensors:

  prompt_speech_tokens   int32[T_prompt_tok]   speech_tokenizer_v3 output
  prompt_text            UTF-8 string          tokenised in C++ at synth time
  spk_emb                f32[192]              CAMPPlus embedding
  ref_mel                f32[T_ref_mel, 80]    matcha-style log-mel @ 24 kHz

Plus a top-level `voice.names` string array listing all voices in the file.
The runtime opens this file with `cosyvoice3_tts_init_voices_from_file`.

Manifest format (JSON):
  [
    {"name": "zero_shot",
     "wav":  "/path/to/zero_shot_prompt.wav",
     "prompt_text": "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"}
  ]

If --manifest is omitted, a built-in default with the upstream
`asset/zero_shot_prompt.wav` voice is used.

Usage:
  python models/convert-cosyvoice3-voices-to-gguf.py \\
      --manifest voices.json \\
      --upstream-base /Volumes/backups/code/cosyvoice3-stash/CosyVoice-upstream \\
      --output /Volumes/backups/ai/crispasr-models/cosyvoice3-0.5b-2512/cosyvoice3-voices.gguf
"""

import argparse
import json
import os
import sys

import numpy as np

try:
    import gguf
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ggml", "python"))
    import gguf


# ---------------------------------------------------------------------------
# matcha mel_spectrogram — inlined port of
# third_party/Matcha-TTS/matcha/utils/audio.py::mel_spectrogram
# (CV3 hparams: n_fft=1920, num_mels=80, sr=24000, hop=480, win=1920,
#  fmin=0, fmax=None, center=False, log compression w/ clip 1e-5)
# ---------------------------------------------------------------------------

_mel_basis_cache = {}
_hann_cache = {}


def _librosa_mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Hand-rolled equivalent of librosa.filters.mel(htk=False, norm='slaney').

    Inlined because importing librosa triggers numba — broken on numpy >= 2.4
    in this env. Matches librosa exactly to f64 precision on the slaney path.
    """
    def hz_to_mel(hz):
        # Slaney: linear below 1 kHz, log above.
        f_min = 0.0
        f_sp = 200.0 / 3
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = np.log(6.4) / 27.0
        hz = np.asarray(hz, dtype=np.float64)
        mels = (hz - f_min) / f_sp
        mask = hz >= min_log_hz
        mels = np.where(mask, min_log_mel + np.log(hz / min_log_hz) / logstep, mels)
        return mels

    def mel_to_hz(mels):
        f_min = 0.0
        f_sp = 200.0 / 3
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = np.log(6.4) / 27.0
        mels = np.asarray(mels, dtype=np.float64)
        hz = f_min + f_sp * mels
        mask = mels >= min_log_mel
        hz = np.where(mask, min_log_hz * np.exp(logstep * (mels - min_log_mel)), hz)
        return hz

    if fmax is None:
        fmax = sr / 2
    fftfreqs = np.linspace(0, sr / 2, n_fft // 2 + 1, dtype=np.float64)
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    fdiff = np.diff(hz_pts)
    ramps = np.subtract.outer(hz_pts, fftfreqs)
    weights = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))
    enorm = 2.0 / (hz_pts[2 : n_mels + 2] - hz_pts[:n_mels])
    weights *= enorm[:, np.newaxis]
    return weights.astype(np.float32)


def _matcha_mel(y_np: np.ndarray) -> np.ndarray:
    """y_np: (T_samples,) f32 mono @ 24 kHz, range [-1, 1].

    Returns (T_frames, 80) f32 log-mel matching matcha + CV3 yaml settings.
    """
    import torch

    n_fft = 1920
    num_mels = 80
    sr = 24000
    hop = 480
    win = 1920
    fmin = 0
    fmax = None
    center = False

    key = ("default",)
    if key not in _mel_basis_cache:
        mel_fb = _librosa_mel_filterbank(sr=sr, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        _mel_basis_cache[key] = torch.from_numpy(mel_fb).float()
        _hann_cache[key] = torch.hann_window(win)
    mel_basis = _mel_basis_cache[key]
    hann = _hann_cache[key]

    y = torch.from_numpy(np.ascontiguousarray(y_np, dtype=np.float32))
    y = y.unsqueeze(0)  # (1, T)
    pad = int((n_fft - hop) / 2)
    y = torch.nn.functional.pad(y.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop,
        win_length=win,
        window=hann,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.view_as_real(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    spec = torch.matmul(mel_basis, spec)
    spec = torch.log(torch.clamp(spec, min=1e-5))
    return spec.squeeze(0).transpose(0, 1).contiguous().numpy()  # (T_frames, 80)


def _load_wav(path: str, target_sr: int) -> np.ndarray:
    """Load mono wav resampled to target_sr. Returns (T,) f32 in [-1, 1]."""
    import torchaudio

    wav, sr = torchaudio.load(path, backend="soundfile")
    wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
    return wav.squeeze(0).numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# CAMPPlus 192-D speaker embedding via campplus.onnx (CV2-shared)
# ---------------------------------------------------------------------------

def _extract_spk_embedding(campplus_session, wav_16k: np.ndarray) -> np.ndarray:
    import torch
    import torchaudio.compliance.kaldi as kaldi

    wav = torch.from_numpy(wav_16k).unsqueeze(0)  # (1, T)
    feat = kaldi.fbank(wav, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    feat_np = feat.unsqueeze(0).cpu().numpy()  # (1, T_frames, 80)
    emb = campplus_session.run(
        None,
        {campplus_session.get_inputs()[0].name: feat_np},
    )[0].flatten()
    return emb.astype(np.float32)


# ---------------------------------------------------------------------------
# speech_tokenizer_v3.onnx → discrete speech tokens
# ---------------------------------------------------------------------------

def _whisper_log_mel_128(wav_16k: np.ndarray):
    """Inlined whisper.log_mel_spectrogram(audio, n_mels=128) — avoids
    pulling in `whisper.transcribe`, which imports numba and breaks on
    numpy >=2.4. We need only the deterministic 128-mel front-end."""
    import os
    import torch

    n_fft = 400
    hop = 160
    audio = torch.from_numpy(wav_16k.astype(np.float32))
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)  # (1, T) — to match upstream call shape
    window = torch.hann_window(n_fft)
    stft = torch.stft(audio, n_fft, hop, window=window, return_complex=True)
    mags = stft[..., :-1].abs() ** 2

    # Find whisper's mel_filters.npz without importing `whisper` (which
    # transitively imports numba, broken on numpy >=2.4). We only need
    # the path to the bundled .npz asset.
    import importlib.util as _ilu
    spec = _ilu.find_spec("whisper")
    if spec is None or spec.origin is None:
        raise ImportError("openai-whisper package not installed")
    whisper_assets = os.path.join(os.path.dirname(spec.origin), "assets", "mel_filters.npz")
    with np.load(whisper_assets, allow_pickle=False) as f:
        mel_128 = torch.from_numpy(f["mel_128"])
    mel_spec = mel_128 @ mags

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


def _extract_speech_tokens(tok_session, wav_16k: np.ndarray) -> np.ndarray:
    assert wav_16k.shape[0] / 16000 <= 30.0, "speech tokenizer caps prompts at 30s"
    feat = _whisper_log_mel_128(wav_16k)  # (1, 128, T_frames)
    feat_np = feat.detach().cpu().numpy()
    tok = tok_session.run(
        None,
        {
            tok_session.get_inputs()[0].name: feat_np,
            tok_session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32),
        },
    )[0].flatten()
    return tok.astype(np.int32)


# ---------------------------------------------------------------------------
# ONNX session bootstrap
# ---------------------------------------------------------------------------

def _resolve_onnx(name: str, upstream_base: str, cache_dir: str) -> str:
    """Return a path to the named onnx, downloading from HF if needed."""
    candidates = [
        os.path.join(upstream_base, name),
        os.path.join(cache_dir, name),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    from huggingface_hub import hf_hub_download

    print(f"  downloading {name} from HF...")
    # campplus.onnx lives in FunAudioLLM/CosyVoice2-0.5B
    # speech_tokenizer_v3.onnx lives in FunAudioLLM/Fun-CosyVoice3-0.5B-2512
    if name == "campplus.onnx":
        repo = "FunAudioLLM/CosyVoice2-0.5B"
    elif name == "speech_tokenizer_v3.onnx":
        repo = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
    else:
        raise ValueError(f"don't know which HF repo hosts {name}")
    os.makedirs(cache_dir, exist_ok=True)
    return hf_hub_download(repo_id=repo, filename=name, cache_dir=cache_dir, local_dir=cache_dir)


def _open_onnx_sessions(upstream_base: str, cache_dir: str):
    import onnxruntime

    opts = onnxruntime.SessionOptions()
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 1

    campplus_path = _resolve_onnx("campplus.onnx", upstream_base, cache_dir)
    tok_path = _resolve_onnx("speech_tokenizer_v3.onnx", upstream_base, cache_dir)
    campplus = onnxruntime.InferenceSession(campplus_path, sess_options=opts, providers=["CPUExecutionProvider"])
    tok = onnxruntime.InferenceSession(tok_path, sess_options=opts, providers=["CPUExecutionProvider"])
    return campplus, tok


# ---------------------------------------------------------------------------
# Manifest + writer
# ---------------------------------------------------------------------------

DEFAULT_MANIFEST = [
    {
        "name": "zero_shot",
        "wav": "{upstream}/asset/zero_shot_prompt.wav",
        "prompt_text": "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
    },
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", help="JSON list of voice entries; omit for built-in default")
    ap.add_argument("--upstream-base", required=True,
                    help="Path to CosyVoice upstream clone (provides asset/*.wav and on-disk onnxes)")
    ap.add_argument("--onnx-cache", default=os.path.expanduser("~/.cache/crispasr/cosyvoice3-onnx"),
                    help="Where to stash campplus.onnx + speech_tokenizer_v3.onnx if not on disk")
    ap.add_argument("--output", required=True, help="Output voices.gguf path")
    args = ap.parse_args()

    if args.manifest:
        with open(args.manifest, encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = [
            {**v, "wav": v["wav"].format(upstream=args.upstream_base)} for v in DEFAULT_MANIFEST
        ]

    print(f"Voices to bake: {[v['name'] for v in manifest]}")
    print("Opening onnx sessions...")
    campplus, tok = _open_onnx_sessions(args.upstream_base, args.onnx_cache)

    writer = gguf.GGUFWriter(args.output, "cosyvoice3-voices")
    writer.add_name("Fun-CosyVoice3-0.5B-2512-voices")
    names = [v["name"] for v in manifest]
    writer.add_array("voice.names", names)

    for v in manifest:
        name = v["name"]
        wav_path = v["wav"]
        prompt_text = v["prompt_text"]
        print(f"\n=== {name} ({wav_path}) ===")
        if not os.path.exists(wav_path):
            raise FileNotFoundError(wav_path)
        wav_16k = _load_wav(wav_path, 16000)
        wav_24k = _load_wav(wav_path, 24000)
        print(f"  wav: {wav_16k.shape[0]/16000:.2f}s @ 16kHz, {wav_24k.shape[0]/24000:.2f}s @ 24kHz")

        spk_emb = _extract_spk_embedding(campplus, wav_16k)
        print(f"  spk_emb: shape={spk_emb.shape}, norm={np.linalg.norm(spk_emb):.4f}")

        speech_tokens = _extract_speech_tokens(tok, wav_16k)
        print(f"  speech_tokens: shape={speech_tokens.shape}, range=[{speech_tokens.min()}, {speech_tokens.max()}]")

        ref_mel = _matcha_mel(wav_24k)
        # Match upstream's `force speech_feat % speech_token = 2` clamp from
        # frontend.py — at 24kHz CV3 expects T_ref_mel = 2 * T_speech_tokens.
        ratio = 2
        target_len = ratio * speech_tokens.shape[0]
        if ref_mel.shape[0] > target_len:
            ref_mel = ref_mel[:target_len]
        elif ref_mel.shape[0] < target_len:
            speech_tokens = speech_tokens[: ref_mel.shape[0] // ratio]
            ref_mel = ref_mel[: ratio * speech_tokens.shape[0]]
        print(f"  ref_mel: shape={ref_mel.shape} (post-align)")
        print(f"  speech_tokens (aligned): shape={speech_tokens.shape}")

        prefix = f"voice.{name}"
        writer.add_string(f"{prefix}.prompt_text", prompt_text)
        writer.add_tensor(f"{prefix}.prompt_speech_tokens", speech_tokens.astype(np.int32))
        writer.add_tensor(f"{prefix}.spk_emb", spk_emb.astype(np.float32))
        writer.add_tensor(f"{prefix}.ref_mel", np.ascontiguousarray(ref_mel.astype(np.float32)))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    sz = os.path.getsize(args.output)
    print(f"\n→ {args.output} ({sz/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
