#!/usr/bin/env python3
"""Create a tada-ref.gguf voice reference from a WAV file.

Usage:
    python models/convert-tada-ref-to-gguf.py \
        --audio reference.wav \
        --transcript "The text spoken in the audio." \
        --output tada-ref-custom.gguf

For non-English, pass the language code and provide the transcript
(the encoder's built-in ASR is English-only):
    python models/convert-tada-ref-to-gguf.py \
        --audio ma_voix.wav \
        --transcript "Bonjour, comment allez-vous?" \
        --language fr \
        --output tada-ref-fr.gguf

Supported languages: ar, ch, de, es, fr, it, ja, pl, pt  (omit for English)

The encoder weights are fetched from HumeAI/tada-codec on HuggingFace
(requires a HF token if the repo is gated).

The output .gguf file can be used with:
    crispasr -m tada-tts-3b-ml-q4_k.gguf --voice tada-ref-custom.gguf --tts "your text"
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    sys.exit("gguf not found: pip install gguf")

try:
    import torch
except ImportError:
    sys.exit("torch not found: pip install torch")

SUPPORTED_LANGUAGES = {"ar", "ch", "de", "es", "fr", "it", "ja", "pl", "pt"}

def load_audio(path: str, target_sr: int = 24000):
    """Load WAV, resample if needed, return (1, T) float32 tensor + actual sample rate."""
    try:
        import torchaudio
        waveform, sr = torchaudio.load(path)
    except Exception:
        try:
            import soundfile as sf
            data, sr = sf.read(path, dtype="float32")
            waveform = torch.from_numpy(data.T if data.ndim > 1 else data[None])
        except Exception as e:
            sys.exit(f"Cannot load audio '{path}': {e}")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    return waveform, sr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio",      required=True,  help="Reference WAV file (≤15 s recommended)")
    ap.add_argument("--transcript", required=True,  help="Exact text spoken in the audio")
    ap.add_argument("--output",     required=True,  help="Output .gguf path")
    ap.add_argument("--language",   default=None,
                    help=f"Language code for non-English audio. Supported: {', '.join(sorted(SUPPORTED_LANGUAGES))}")
    ap.add_argument("--codec-repo", default="HumeAI/tada-codec",
                    help="HuggingFace repo containing the encoder weights (default: HumeAI/tada-codec)")
    ap.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    args = ap.parse_args()

    if args.language and args.language not in SUPPORTED_LANGUAGES:
        sys.exit(f"Unknown language '{args.language}'. Supported: {', '.join(sorted(SUPPORTED_LANGUAGES))}")

    print(f"Loading reference audio: {args.audio}")
    waveform, sr = load_audio(args.audio)
    print(f"  shape={tuple(waveform.shape)}  sr={sr}  duration={waveform.shape[-1]/sr:.2f}s")

    print(f"Loading TADA encoder from {args.codec_repo} …")
    try:
        from tada.modules.encoder import Encoder
    except ImportError:
        sys.exit(
            "tada package not found.\n"
            "Install it from https://github.com/HumeAI/tada:\n"
            "  pip install git+https://github.com/HumeAI/tada.git"
        )

    encoder = Encoder.from_pretrained(args.codec_repo, subfolder="encoder",
                                      language=args.language).to(args.device)
    encoder.eval()

    waveform = waveform.to(args.device)

    print(f"Encoding audio (language={args.language or 'en'}) …")
    with torch.no_grad():
        text_arg = [args.transcript] if args.language else None
        enc_out = encoder(waveform, text=text_arg, sample_rate=sr)

    vals = enc_out.token_values[0].cpu().float().numpy().astype(np.float32)    # (N, 512)
    pos  = enc_out.token_positions[0].cpu().float().numpy().astype(np.float32) # (N,)
    n = vals.shape[0]
    print(f"  {n} acoustic tokens, {waveform.shape[-1]/sr:.2f}s audio")

    out_path = args.output
    w = GGUFWriter(out_path, arch="crispasr.reference", use_temp_file=False)
    w.add_name(Path(out_path).stem)
    w.add_string("crispasr.ref.tada_tts_prompt_text", args.transcript)
    if args.language:
        w.add_string("crispasr.ref.tada_tts_language", args.language)
    w.add_tensor("prompt_token_values",    np.ascontiguousarray(vals), raw_dtype=GGMLQuantizationType.F32)
    w.add_tensor("prompt_token_positions", np.ascontiguousarray(pos),  raw_dtype=GGMLQuantizationType.F32)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"Saved: {out_path}  ({Path(out_path).stat().st_size // 1024} KB)")
    print(f"\nUse with: crispasr -m tada-tts-3b-ml-q4_k.gguf --voice {out_path} --tts 'your text'")


if __name__ == "__main__":
    main()
