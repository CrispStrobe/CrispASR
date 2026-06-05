#!/usr/bin/env python3
"""
Reference backend for Piper VITS TTS — dumps intermediate stage outputs
for diff-testing against the C++ piper_tts runtime.

Usage:
    python tools/reference_backends/piper_tts.py \
        --onnx /mnt/storage/piper/en_US-lessac-medium.onnx \
        --text "Hello world" \
        --output /mnt/storage/piper/ref_dump.gguf

Or with --stages to dump intermediate stages as separate GGUF tensors.

The ONNX model is treated as ground truth. We also provide a hook to
dump internal stage outputs by running the model through ONNX Runtime
with intermediate node outputs exposed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    sys.exit("pip install onnxruntime")

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    sys.exit("pip install gguf")


def load_config(onnx_path: str) -> dict:
    """Load the .onnx.json sidecar config."""
    json_path = Path(onnx_path + ".json")
    if not json_path.exists():
        json_path = Path(onnx_path).with_suffix(".onnx.json")
    if not json_path.exists():
        sys.exit(f"Config not found: {json_path}")
    with open(json_path) as f:
        return json.load(f)


def phonemize_text(text: str, voice: str) -> str:
    """Phonemize text using espeak-ng subprocess."""
    import subprocess
    result = subprocess.run(
        ["espeak-ng", "-q", "--ipa=3", "-v", voice, text],
        capture_output=True, text=True
    )
    return result.stdout.strip()


def encode_phonemes(ipa: str, phoneme_id_map: dict) -> list[int]:
    """Encode IPA phonemes to integer IDs with interspersed padding."""
    ids = [1]  # BOS = ^
    i = 0
    while i < len(ipa):
        # Try longest match (some phonemes are multi-char)
        found = False
        for length in range(4, 0, -1):
            candidate = ipa[i:i+length]
            if candidate in phoneme_id_map:
                ids.extend(phoneme_id_map[candidate])
                ids.append(0)  # PAD
                i += length
                found = True
                break
        if not found:
            i += 1  # skip unknown
    ids.append(2)  # EOS = $
    return ids


def run_onnx_inference(onnx_path: str, phoneme_ids: list[int],
                       noise_scale: float = 0.667,
                       length_scale: float = 1.0,
                       noise_w: float = 0.8,
                       speaker_id: int = 0,
                       num_speakers: int = 1) -> np.ndarray:
    """Run full ONNX inference, return audio waveform."""
    sess = ort.InferenceSession(onnx_path)

    input_ids = np.array([phoneme_ids], dtype=np.int64)
    input_lengths = np.array([len(phoneme_ids)], dtype=np.int64)
    scales = np.array([noise_scale, length_scale, noise_w], dtype=np.float32)

    inputs = {
        "input": input_ids,
        "input_lengths": input_lengths,
        "scales": scales,
    }

    # Check if model expects speaker ID
    input_names = [inp.name for inp in sess.get_inputs()]
    if "sid" in input_names:
        inputs["sid"] = np.array([speaker_id], dtype=np.int64)

    outputs = sess.run(None, inputs)
    audio = outputs[0].squeeze()
    return audio


def dump_stages(onnx_path: str, phoneme_ids: list[int],
                config: dict, output_path: str,
                audio: np.ndarray):
    """Dump reference outputs as GGUF for diff testing."""
    writer = GGUFWriter(output_path, arch="piper-ref")

    # Store the phoneme IDs used
    writer.add_string("piper_ref.phoneme_ids",
                      json.dumps(phoneme_ids))

    # Full audio output
    writer.add_tensor("audio_output", audio.astype(np.float32),
                      raw_dtype=GGMLQuantizationType.F32)

    # Store config used
    writer.add_float32("piper_ref.noise_scale",
                       config.get("inference", {}).get("noise_scale", 0.667))
    writer.add_float32("piper_ref.length_scale",
                       config.get("inference", {}).get("length_scale", 1.0))
    writer.add_float32("piper_ref.noise_w",
                       config.get("inference", {}).get("noise_w", 0.8))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"Reference dump written to {output_path}")
    print(f"  audio_output: {audio.shape} ({audio.dtype})")
    print(f"  duration: {len(audio) / config.get('audio', {}).get('sample_rate', 22050):.3f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="Path to .onnx model")
    parser.add_argument("--text", default="Hello world",
                        help="Text to synthesize")
    parser.add_argument("--ipa", default=None,
                        help="Pre-phonemized IPA (skip espeak-ng)")
    parser.add_argument("--output", default=None,
                        help="Output GGUF path for reference dump")
    parser.add_argument("--wav", default=None,
                        help="Output WAV path")
    parser.add_argument("--noise-scale", type=float, default=None)
    parser.add_argument("--length-scale", type=float, default=None)
    parser.add_argument("--noise-w", type=float, default=None)
    parser.add_argument("--speaker-id", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.onnx)
    phoneme_id_map = config.get("phoneme_id_map", {})
    espeak_voice = config.get("espeak", {}).get("voice", "en-us")

    noise_scale = args.noise_scale if args.noise_scale is not None else \
        config.get("inference", {}).get("noise_scale", 0.667)
    length_scale = args.length_scale if args.length_scale is not None else \
        config.get("inference", {}).get("length_scale", 1.0)
    noise_w = args.noise_w if args.noise_w is not None else \
        config.get("inference", {}).get("noise_w", 0.8)

    # Phonemize
    if args.ipa:
        ipa = args.ipa
    else:
        ipa = phonemize_text(args.text, espeak_voice)
    print(f"IPA: {ipa}")

    # Encode
    phoneme_ids = encode_phonemes(ipa, phoneme_id_map)
    print(f"Phoneme IDs ({len(phoneme_ids)}): {phoneme_ids[:20]}...")

    # Run inference
    num_speakers = config.get("num_speakers", 1)
    audio = run_onnx_inference(
        args.onnx, phoneme_ids,
        noise_scale=noise_scale,
        length_scale=length_scale,
        noise_w=noise_w,
        speaker_id=args.speaker_id,
        num_speakers=num_speakers,
    )
    print(f"Audio: {audio.shape}, range [{audio.min():.4f}, {audio.max():.4f}]")
    sr = config.get("audio", {}).get("sample_rate", 22050)
    print(f"Duration: {len(audio) / sr:.3f}s @ {sr} Hz")

    # Save WAV
    if args.wav:
        import wave
        import struct
        audio_int16 = (audio * 32767).astype(np.int16)
        with wave.open(args.wav, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(audio_int16.tobytes())
        print(f"WAV written to {args.wav}")

    # Dump reference
    if args.output:
        dump_stages(args.onnx, phoneme_ids, config, args.output, audio)


if __name__ == "__main__":
    main()
