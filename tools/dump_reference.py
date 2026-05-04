#!/usr/bin/env python3
"""
CrispASR — unified reference activation dumper.

Loads a HuggingFace ASR model in PyTorch, runs it on an audio file, captures
intermediate activations at every architectural boundary via forward hooks,
and writes the collection to a single **GGUF tensor archive**. The C++ diff
harness (`crispasr-diff`) then loads that GGUF via `core_gguf::load_weights`
and compares each captured tensor against what the ggml forward pass
produces — element-wise, with cosine similarity, max-abs diff, and top-1
argmax match for logits.

Replaces the per-model one-off `models/*-reference-dump.py` scripts by
providing:

  1. A consistent CLI across all backends:
     `python tools/dump_reference.py --backend voxtral --model-dir /hf/dir
        --audio samples/jfk.wav --output /tmp/voxtral-ref.gguf`

  2. A shared WAV loader (16 kHz mono, stdlib only).

  3. A shared GGUF writer that handles the float serialization + tensor
     metadata (using the `gguf` Python package that ships with llama.cpp).

  4. A plug-in registry so each backend's PyTorch hooks live in its own
     small module (`tools/reference_backends/<name>.py`), and adding a
     new backend is a ~60-line file instead of a ~250-line script.

Stages exposed by every backend (adjust per backend as needed):

  raw_audio           (N,)            F32 PCM samples
  mel_spectrogram     (B, n_mels, T)  F32 log-mel features
  encoder_output      (B, T_enc, D)   F32 encoder hidden state
  encoder_layer_K     (B, T_enc, D)   F32 after encoder block K
  projector_output    (B, N, D_llm)   F32 audio tokens for the LLM
  llm_block_K         (B, T, D)       F32 after LLM block K
  llm_logits          (B, T, V)       F32 language-model logits
  llm_argmax          (B, T)          I32 greedy-decoded token IDs
  generated_text      (string)        text decoded from argmax

Backends are free to emit additional stage names (see each module's
`DEFAULT_STAGES`). Unused stages are skipped without erroring.

Usage:

  python tools/dump_reference.py --list-backends
  python tools/dump_reference.py --backend qwen3   \\
      --model-dir /hf/qwen3-asr-0.6b               \\
      --audio samples/jfk.wav                      \\
      --output /tmp/qwen3-ref.gguf
  python tools/dump_reference.py --backend voxtral \\
      --model-dir /hf/voxtral-mini-3b-2507          \\
      --audio samples/jfk.wav                      \\
      --stages mel_spectrogram,encoder_output,llm_logits \\
      --output /tmp/voxtral-ref.gguf

The GGUF archive stores each activation as a named F32 tensor. Load it
from C++ with `core_gguf::load_weights(path, backend, "ref", wl)` and
then `wl.tensors["mel_spectrogram"]` etc.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import wave
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

# Each entry maps a user-facing --backend name to a Python module under
# tools/reference_backends/ that exposes a dump() function:
#
#   def dump(model_dir: Path, audio: np.ndarray, stages: set[str]) -> dict[str, np.ndarray]:
#       """Run the HF model on `audio`, capture activations for the stages
#       listed in `stages`, and return {name: ndarray}. Raise KeyError or
#       NotImplementedError for stages this backend doesn't support."""
#
# For per-layer encoder/LLM captures, use `reference_backends/_hooks.py`
# (capture_modules / drop_hooks / finalize) — it handles forward-hook
# bookkeeping and normalises (B, T, D) → (T, D) row-major to match
# crispasr's flat layout. See parakeet.py for a worked example.
#
# Adding a new backend is:
#   1. tools/reference_backends/<name>.py  with dump() + DEFAULT_STAGES
#   2. one line here.
REGISTERED_BACKENDS: Dict[str, str] = {
    "qwen3":      "reference_backends.qwen3",
    "voxtral":    "reference_backends.voxtral",
    "voxtral4b":  "reference_backends.voxtral4b",
    "granite":    "reference_backends.granite",
    "granite-4.1": "reference_backends.granite",
    # granite-speech 4.1-2b NAR: non-autoregressive variant. Custom
    # modeling code in the HF snapshot — needs trust_remote_code=True.
    "granite-nle": "reference_backends.granite_nle",
    # Encoder-decoder (NeMo + Cohere) reference backends. These capture
    # encoder activations via forward hooks and run generate() for a
    # greedy transcript check — no per-token logits, because the decoder
    # autoregresses with a KV cache that doesn't have a clean "per-step
    # logits" entry point the way the speech-LLMs do.
    "cohere":     "reference_backends.cohere",
    "parakeet":   "reference_backends.parakeet",
    # NeMo Canary (FastConformer + Transformer decoder). model_dir may be
    # the HF id "nvidia/canary-1b-v2" or a local .nemo path. The C++ diff
    # branch ("canary") compares mel_spectrogram + encoder_output; the
    # per-layer captures listed in DEFAULT_STAGES are diagnostic-only.
    "canary":     "reference_backends.canary",
    "gemma4":     "reference_backends.gemma4",
    # Qwen3-TTS-12Hz Base. The audio arg is the voice-clone reference WAV
    # (16 kHz mono); synth text + ref text come from env vars. See
    # reference_backends/qwen3_tts.py for the full prompt contract.
    "qwen3-tts":  "reference_backends.qwen3_tts",
    # Qwen3-TTS-Tokenizer-12Hz codec decoder only (codes → PCM).
    # model_dir = the Tokenizer-12Hz HF snapshot; audio arg is unused.
    "qwen3-tts-codec": "reference_backends.qwen3_tts_codec",
    # Qwen3-TTS ECAPA speaker encoder only.
    # model_dir = the talker HF snapshot (contains speaker_encoder.*).
    "qwen3-tts-spk":   "reference_backends.qwen3_tts_spk",
    # Qwen3-TTS-Tokenizer-12Hz codec ENCODER (audio → codes).
    # model_dir = the Tokenizer-12Hz HF snapshot. audio is unused.
    "qwen3-tts-cenc":  "reference_backends.qwen3_tts_cenc",
    # VibeVoice-ASR 7B: two σ-VAE encoders + connectors + Qwen2 decoder.
    # NOTE: audio must be 16 kHz on entry (shared loader); the backend
    # resamples to 24 kHz internally.
    "vibevoice":  "reference_backends.vibevoice",
    # MiMo-Audio-Tokenizer encoder (PCM → 8-channel RVQ codes).
    # model_dir = the MiMo-Audio-Tokenizer HF snapshot. 16 kHz mono PCM is
    # resampled to 24 kHz internally.
    "mimo-tokenizer": "reference_backends.mimo_tokenizer",
    # MiMo-V2.5-ASR LM-half: input_local_transformer + 36L Qwen2 LLM.
    # model_dir = the MiMo-V2.5-ASR HF snapshot. The audio-tokenizer path
    # is read from MIMO_TOKENIZER_DIR (or auto-derived from a sibling dir).
    "mimo-asr":   "reference_backends.mimo_asr",
    # Kokoro / StyleTTS2 (iSTFTNet). Text-driven; the audio arg is a
    # placeholder. Phonemes + voice come from KOKORO_PHONEMES / KOKORO_VOICE
    # env vars (see reference_backends/kokoro.py for the full list).
    "kokoro":     "reference_backends.kokoro",
    # Orpheus-3B SNAC 24 kHz codec decoder only (codes → PCM).
    # model_dir = hubertsiuzdak/snac_24khz HF snapshot; audio arg is unused.
    # Driven by ORPHEUS_SNAC_T_SUPER (default 4) + ORPHEUS_SNAC_CODE
    # (default 0); see reference_backends/orpheus_snac.py.
    "orpheus":    "reference_backends.orpheus_snac",
    # Chatterbox TTS: T3 (Llama AR) → S3Gen (CFM) → HiFTGenerator.
    # model_dir = ResembleAI/chatterbox snapshot (or local with
    # t3_cfg.safetensors + s3gen.safetensors + ve.safetensors + conds.pt).
    # audio arg is a reference voice WAV for cloning (16 kHz); when conds.pt
    # exists the built-in voice is used and audio is ignored.
    "chatterbox": "reference_backends.chatterbox",
    # Chatterbox Turbo (GPT-2 T3 + meanflow S3Gen). Per-layer encoder
    # dumps for element-wise conformer attention validation.
    "chatterbox_turbo": "reference_backends.chatterbox_turbo",
}

DEFAULT_STAGES_BY_BACKEND: Dict[str, List[str]] = {}  # populated at import


# ---------------------------------------------------------------------------
# Shared WAV loader (stdlib only, no torchaudio / librosa)
# ---------------------------------------------------------------------------

def load_audio_16k_mono(path: Path) -> np.ndarray:
    """Load 16 kHz mono PCM audio into a float32 numpy array in [-1, 1].

    Accepts any 16-bit PCM WAV at 16 kHz. Multi-channel input is averaged
    to mono. Raises SystemExit with a clear message for unsupported inputs.
    """
    with wave.open(str(path), "rb") as w:
        sr     = w.getframerate()
        nchan  = w.getnchannels()
        sampw  = w.getsampwidth()
        nframe = w.getnframes()
        raw    = w.readframes(nframe)
    if sampw != 2:
        raise SystemExit(f"{path}: only 16-bit PCM supported, got {sampw*8}-bit. "
                         f"Pre-convert with: ffmpeg -i in.X -ar 16000 -ac 1 -c:a pcm_s16le out.wav")
    if sr != 16000:
        raise SystemExit(f"{path}: expected 16 kHz, got {sr} Hz. "
                         f"Pre-convert with: ffmpeg -i in.X -ar 16000 -ac 1 -c:a pcm_s16le out.wav")
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if nchan > 1:
        pcm = pcm.reshape(-1, nchan).mean(axis=1)
    return np.ascontiguousarray(pcm)


# ---------------------------------------------------------------------------
# GGUF writer — ONE tensor archive per dump
# ---------------------------------------------------------------------------

def _to_contig_f32(arr: np.ndarray) -> np.ndarray:
    """Squeeze sentinel axes that GGUF's max-4D tensor limit can't accept,
    convert to float32, and make it C-contiguous."""
    if not np.issubdtype(arr.dtype, np.floating) and not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"unsupported ndarray dtype: {arr.dtype}")
    a = arr
    # GGUF tensors are up to 4D. If the caller captured a 5D or 6D tensor
    # (rare — happens for some multi-head layouts), squeeze unit axes first.
    while a.ndim > 4:
        # Squeeze the LEFTMOST unit axis we can find.
        squeezable = [i for i in range(a.ndim) if a.shape[i] == 1]
        if not squeezable:
            raise ValueError(f"tensor has {a.ndim} dims and no unit axes to squeeze: {a.shape}")
        a = np.squeeze(a, axis=squeezable[0])
    if a.dtype != np.float32 and a.dtype != np.int32:
        a = a.astype(np.float32)
    if not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a)
    return a


def write_gguf_archive(captures: Dict[str, np.ndarray],
                       meta: Dict[str, Any],
                       output_path: Path) -> None:
    """Serialize a dict of captured activations to a GGUF tensor archive.

    The resulting file is loadable via core_gguf::load_weights on the C++
    side, which returns a `WeightLoad` whose `tensors` map is keyed by the
    names used here. Scalar metadata (backend name, model path, audio
    path, generated text) is stored as GGUF key/value pairs in the header.
    """
    try:
        import gguf
    except ImportError as e:
        raise SystemExit(
            "gguf Python package not found. Install with:  pip install gguf\n"
            "(it ships with llama.cpp and ggml; installs quickly).") from e

    output_path.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(str(output_path), arch="crispasr.reference")
    w.add_description("CrispASR reference activation dump")

    # Metadata
    for k, v in meta.items():
        if isinstance(v, bool):
            w.add_bool(f"crispasr.ref.{k}", v)
        elif isinstance(v, int):
            w.add_int32(f"crispasr.ref.{k}", v)
        elif isinstance(v, float):
            w.add_float32(f"crispasr.ref.{k}", v)
        elif isinstance(v, str):
            w.add_string(f"crispasr.ref.{k}", v)
        # silently skip other types

    # Tensors. GGUF orders them as the caller adds them.
    for name, arr in sorted(captures.items()):
        a = _to_contig_f32(arr)
        w.add_tensor(name, a)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _resolve_backend(name: str):
    if name not in REGISTERED_BACKENDS:
        raise SystemExit(
            f"unknown backend '{name}'. Available: {sorted(REGISTERED_BACKENDS)}")
    module_name = REGISTERED_BACKENDS[name]
    # tools/ is on sys.path because we run dump_reference.py from that dir,
    # OR the caller has added it. Fall back to importing via relative package.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        mod = importlib.import_module(module_name)
    except ImportError as e:
        raise SystemExit(
            f"failed to import backend module '{module_name}': {e}\n"
            f"Make sure tools/{module_name.replace('.', '/')}.py exists.")
    return mod


def main() -> None:
    p = argparse.ArgumentParser(
        description="CrispASR unified reference activation dumper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--backend", help="backend name (see --list-backends)")
    p.add_argument("--model-dir", type=Path, help="HF model directory")
    p.add_argument("--audio", type=Path, help="input WAV (16 kHz mono)")
    p.add_argument("--output", type=Path, help="output GGUF archive path")
    p.add_argument("--stages", default="",
                   help="comma-separated stage names to capture; empty = backend default")
    p.add_argument("--max-new-tokens", type=int, default=20,
                   help="number of tokens to greedy-decode for logits capture")
    p.add_argument("--list-backends", action="store_true",
                   help="print available backends and exit")
    args = p.parse_args()

    if args.list_backends:
        print("Available backends:")
        for name, mod_path in sorted(REGISTERED_BACKENDS.items()):
            print(f"  {name:10s}  -> tools/{mod_path.replace('.', '/')}.py")
        return

    if not (args.backend and args.model_dir and args.audio and args.output):
        p.error("--backend, --model-dir, --audio, --output are all required "
                "(unless --list-backends is set)")

    mod = _resolve_backend(args.backend)

    # Resolve stage list
    default_stages = getattr(mod, "DEFAULT_STAGES", [])
    if args.stages:
        stages = set(s.strip() for s in args.stages.split(",") if s.strip())
    else:
        stages = set(default_stages)
    if not stages:
        raise SystemExit(f"no stages to capture (backend '{args.backend}' "
                         f"has empty DEFAULT_STAGES and --stages is empty)")

    # Load audio
    print(f"Loading audio: {args.audio}")
    audio = load_audio_16k_mono(args.audio)
    print(f"  samples: {len(audio)}  ({len(audio)/16000:.2f} s)")

    # Run backend dump
    print(f"Running {args.backend} reference forward pass ...")
    captures = mod.dump(
        model_dir=args.model_dir,
        audio=audio,
        stages=stages,
        max_new_tokens=args.max_new_tokens,
    )

    # Always include raw audio so C++ tests can feed it in without
    # re-reading the WAV.
    if "raw_audio" in stages:
        captures.setdefault("raw_audio", audio.astype(np.float32))

    print(f"Captured {len(captures)} tensors:")
    for name in sorted(captures):
        a = captures[name]
        if isinstance(a, str):
            # String-typed captures (e.g. granite/cohere generated_text)
            # move into metadata below — skip them in the tensor listing.
            preview = a if len(a) <= 60 else a[:57] + "..."
            print(f"  {name:28s}  str        {preview!r}")
            continue
        print(f"  {name:28s}  {tuple(a.shape)}  {a.dtype}")

    # Serialize
    meta = {
        "backend":  args.backend,
        "model_dir": str(args.model_dir.resolve()),
        "audio":    str(args.audio.resolve()),
        "n_samples": int(len(audio)),
        "sample_rate": 16000,
        "generated_text": str(captures.pop("generated_text", "")) if "generated_text" in captures else "",
    }
    # Any other str-typed captures get routed into metadata too so the
    # C++ diff harness can read them via the GGUF kv table without
    # tripping `_to_contig_f32`'s "must be numeric ndarray" check.
    for name in list(captures.keys()):
        if isinstance(captures[name], str):
            meta[name] = captures.pop(name)
    # Pass through env-configurable prompt/text/voice metadata so diff
    # harnesses on the C++ side can replay the exact synthesis context.
    for env_key in ("QWEN3_TTS_SYN_TEXT", "QWEN3_TTS_REF_TEXT", "QWEN3_TTS_LANG", "QWEN3_TTS_VOICE",
                    "KOKORO_PHONEMES", "KOKORO_VOICE", "KOKORO_SEED"):
        val = os.environ.get(env_key)
        if val is not None:
            meta[env_key.lower()] = val
    write_gguf_archive(captures, meta, args.output)
    print(f"Wrote GGUF archive: {args.output}  "
          f"({args.output.stat().st_size/1024:.1f} KiB)")


if __name__ == "__main__":
    main()
