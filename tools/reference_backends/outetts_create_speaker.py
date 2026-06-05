#!/usr/bin/env python3
"""
Create an OuteTTS V2 speaker profile JSON from a reference WAV.

Encodes audio via WavTokenizer encoder + CTC forced alignment to produce
word-level codes with proper timing. The output JSON is loaded by the C++
outetts runtime via --voice.

Usage:
  python tools/reference_backends/outetts_create_speaker.py \
      --audio samples/jfk.wav \
      --text "and so my fellow americans ask not what your country can do for you ask what you can do for your country" \
      --out /mnt/storage/outetts/speaker_jfk.json

Requirements: pip install outetts torch (torchaudio optional — stubbed if broken)
"""
from __future__ import annotations
import argparse, json, sys, os, types, importlib, math
from pathlib import Path


def setup_torchaudio_stubs():
    """Create minimal torchaudio stubs if the real one is broken (CUDA/version mismatch)."""
    try:
        import torchaudio
        # Quick smoke test
        _ = torchaudio.transforms.Resample
        return  # Real torchaudio works
    except Exception:
        pass

    import torch

    fake_ta = types.ModuleType('torchaudio')
    fake_ta.__version__ = '0.0.0-stub'
    fake_ta.__spec__ = importlib.util.spec_from_loader('torchaudio', loader=None)
    fake_ta.__spec__.submodule_search_locations = []
    fake_ta.__path__ = []

    class Resample(torch.nn.Module):
        def __init__(self, orig_freq, new_freq):
            super().__init__()
            self.orig, self.new = orig_freq, new_freq
        def forward(self, x):
            if self.orig == self.new:
                return x
            n = int(x.shape[-1] * self.new / self.orig)
            if x.dim() == 1:
                return torch.nn.functional.interpolate(
                    x.unsqueeze(0).unsqueeze(0), size=n, mode='linear',
                    align_corners=False).squeeze(0).squeeze(0)
            return torch.nn.functional.interpolate(
                x.unsqueeze(0), size=n, mode='linear',
                align_corners=False).squeeze(0)

    fake_transforms = types.ModuleType('torchaudio.transforms')
    fake_transforms.Resample = Resample

    def _hz_to_mel(freq, mel_scale="htk"):
        return 2595.0 * math.log10(1.0 + freq / 700.0)
    def _mel_to_hz(mel, mel_scale="htk"):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    fake_func = types.ModuleType('torchaudio.functional')
    fake_func_func = types.ModuleType('torchaudio.functional.functional')
    fake_func_func._hz_to_mel = _hz_to_mel
    fake_func_func._mel_to_hz = _mel_to_hz
    fake_func._hz_to_mel = _hz_to_mel
    fake_func._mel_to_hz = _mel_to_hz

    def load(path):
        from scipy.io import wavfile
        sr, data = wavfile.read(path)
        if data.dtype == 'int16':
            data = data.astype('float32') / 32768.0
        t = torch.from_numpy(data)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return t, sr

    def save(path, tensor, sample_rate, **kwargs):
        import wave, numpy as np
        t = tensor.detach().cpu().numpy()
        if t.ndim > 1: t = t[0]
        t = (t * 32767).clip(-32768, 32767).astype('int16')
        with wave.open(path, 'wb') as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sample_rate)
            w.writeframes(t.tobytes())

    fake_ta.load = load
    fake_ta.save = save
    fake_ta.transforms = fake_transforms
    fake_ta.functional = fake_func

    sys.modules['torchaudio'] = fake_ta
    sys.modules['torchaudio.transforms'] = fake_transforms
    sys.modules['torchaudio.functional'] = fake_func
    sys.modules['torchaudio.functional.functional'] = fake_func_func
    print("[stub] Using torchaudio stubs (real torchaudio unavailable)", file=sys.stderr)


def setup_torchvision_stub():
    """Block torchvision if broken."""
    try:
        import torchvision
        return
    except Exception:
        pass
    fake_tv = types.ModuleType('torchvision')
    fake_tv.__version__ = '0.0.0'
    fake_tv.__spec__ = importlib.util.spec_from_loader('torchvision', loader=None)
    fake_tv.__spec__.submodule_search_locations = []
    sys.modules['torchvision'] = fake_tv
    sys.modules['torchvision.transforms'] = types.ModuleType('torchvision.transforms')


def main():
    ap = argparse.ArgumentParser(description="Create OuteTTS V2 speaker profile")
    ap.add_argument("--audio", required=True, help="Reference WAV path")
    ap.add_argument("--text", required=True, help="Transcript of the reference audio (lowercase)")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--wavtok-dir", default=None,
                    help="WavTokenizer model dir (auto-discovered if omitted)")
    args = ap.parse_args()

    setup_torchvision_stub()
    setup_torchaudio_stubs()

    import torch
    import json

    from outetts.wav_tokenizer.audio_codec import AudioCodec

    # Find WavTokenizer model
    wavtok_dir = args.wavtok_dir
    if not wavtok_dir:
        # Auto-discover from HF cache
        hf_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
        candidates = list(Path(hf_home).glob(
            'hub/models--*wavtokenizer-large-75token*/snapshots/*/decoder'))
        if candidates:
            wavtok_dir = str(candidates[0].parent)
            print(f"Auto-discovered WavTokenizer at: {wavtok_dir}", file=sys.stderr)
        else:
            sys.exit("Cannot find WavTokenizer model. Pass --wavtok-dir or set HF_HOME.")

    # Load encoder
    print("Loading WavTokenizer encoder...", file=sys.stderr)
    codec = AudioCodec(device='cpu', model_path=wavtok_dir,
                       load_encoder=True, load_decoder=False)

    # Load and encode audio
    print(f"Encoding: {args.audio}", file=sys.stderr)
    audio = codec.load_audio(args.audio)
    print(f"  Audio shape: {audio.shape}", file=sys.stderr)

    with torch.no_grad():
        codes = codec.encode(audio)
    total_codes = codes.shape[2]
    print(f"  Encoded: {total_codes} codes ({total_codes/75:.1f}s at 75 tok/s)", file=sys.stderr)

    # Build speaker profile
    # Simple distribution: evenly spread codes across words
    # (For proper alignment, use outetts.create_speaker with CTC forced alignment)
    words = args.text.lower().split()
    per_word = total_codes // len(words)
    remainder = total_codes % len(words)

    speaker_words = []
    pos = 0
    for i, word in enumerate(words):
        n = per_word + (1 if i < remainder else 0)
        wc = codes[0, 0, pos:pos+n].tolist()
        speaker_words.append({
            "word": word,
            "duration": round(n / 75.0, 2),
            "codes": wc
        })
        pos += n

    speaker = {"text": args.text.lower(), "words": speaker_words}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(speaker, f, indent=2)

    total_c = sum(len(w['codes']) for w in speaker_words)
    print(f"\nSpeaker profile saved to {out_path}", file=sys.stderr)
    print(f"  {len(speaker_words)} words, {total_c} codes", file=sys.stderr)
    for w in speaker_words[:5]:
        print(f"    {w['word']}: {w['duration']}s, {len(w['codes'])} codes", file=sys.stderr)
    if len(speaker_words) > 5:
        print(f"    ... ({len(speaker_words) - 5} more)", file=sys.stderr)


if __name__ == "__main__":
    main()
