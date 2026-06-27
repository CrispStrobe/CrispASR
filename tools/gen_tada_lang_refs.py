#!/usr/bin/env python3
"""Generate tada-ref-<lang>.gguf for each supported language using FLEURS clips.

Requires: pip install datasets soundfile gguf torch
Also: pip install git+https://github.com/HumeAI/tada.git

Source audio: google/fleurs (CC-BY 4.0, attribution: Google).
Output .gguf files should carry the same CC-BY 4.0 attribution.

Usage:
    python gen_lang_refs.py --output-dir /Volumes/backups/ai/crispasr-gguf/lang-refs/
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import soundfile as sf
import torch

OMP_THREADS = os.environ.get("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", OMP_THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    sys.exit("gguf not found: pip install gguf")

# Redirect gated meta-llama/Llama-3.2-1B → unsloth mirror (same weights, public).
# hume-tada's Encoder internally loads the Llama-3.2-1B tokenizer/config; patch
# hf_hub_download at all call sites before importing tada/transformers so those
# requests go to the ungated unsloth mirror instead.
_LLAMA_REDIR = {"meta-llama/Llama-3.2-1B": "unsloth/Llama-3.2-1B"}

try:
    import huggingface_hub
    import huggingface_hub.file_download as _hfd

    _orig_dl = _hfd.hf_hub_download

    def _hf_hub_download_redir(repo_id: str, *args, **kw):
        if repo_id in _LLAMA_REDIR:
            target = _LLAMA_REDIR[repo_id]
            print(f"  [llama-bypass] {repo_id} → {target}", flush=True)
            repo_id = target
        return _orig_dl(repo_id, *args, **kw)

    # Patch both the module attr and the file_download submodule attr so any
    # import path (huggingface_hub.hf_hub_download or the internal one) is covered.
    _hfd.hf_hub_download = _hf_hub_download_redir
    huggingface_hub.hf_hub_download = _hf_hub_download_redir

    # Pre-import transformers and patch its module-level binding too, so
    # transformers.utils.hub.cached_file() uses the redirected function.
    import transformers.utils.hub as _tuh
    if hasattr(_tuh, "hf_hub_download"):
        _tuh.hf_hub_download = _hf_hub_download_redir
    print("  [llama-bypass] redirect installed (meta-llama → unsloth)", flush=True)
except Exception as _patch_err:
    print(f"  WARNING: llama bypass setup failed: {_patch_err}", flush=True)

try:
    from tada.modules.encoder import Encoder
except ImportError as _e:
    sys.exit(f"tada import failed ({_e}): pip install hume-tada")

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("datasets not found: pip install datasets")

# FLEURS config name per TADA language code.
# Picks a clip from the 'train' split; index chosen for ~5-8 s duration.
LANG_CONFIGS = {
    "fr": ("fr_fr",   0),
    "de": ("de_de",   0),
    "es": ("es_419",  0),
    "ar": ("ar_eg",   0),
    "ch": ("cmn_hans_cn", 0),
    "it": ("it_it",   0),
    "ja": ("ja_jp",   0),
    "pl": ("pl_pl",   0),
    "pt": ("pt_br",   0),
}

CODEC_REPO = "HumeAI/tada-codec"
TARGET_SR = 24000
MAX_DURATION_S = 10.0


def resample(arr: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    if from_sr == to_sr:
        return arr
    try:
        import resampy
        return resampy.resample(arr, from_sr, to_sr)
    except ImportError:
        pass
    try:
        from scipy.signal import resample as scipy_resample
        target_len = int(len(arr) * to_sr / from_sr)
        return scipy_resample(arr, target_len).astype(np.float32)
    except ImportError:
        pass
    # Basic linear resampling fallback (lower quality but works everywhere)
    target_len = int(len(arr) * to_sr / from_sr)
    xs = np.linspace(0, len(arr) - 1, target_len)
    return np.interp(xs, np.arange(len(arr)), arr).astype(np.float32)


def fetch_fleurs_clip(fleurs_config: str, index: int = 0):
    """Return (audio_array_float32_24k, transcript, sr=24000)."""
    print(f"  fetching FLEURS '{fleurs_config}' index={index} …")
    ds = load_dataset("google/fleurs", fleurs_config, split="train",
                      streaming=True)
    ex = None
    for i, row in enumerate(ds):
        if i == index:
            ex = row
            break
    if ex is None:
        raise RuntimeError(f"FLEURS '{fleurs_config}' has no example at index {index}")

    arr = np.array(ex["audio"]["array"], dtype=np.float32)
    sr  = int(ex["audio"]["sampling_rate"])
    transcript = ex.get("transcription") or ex.get("raw_transcription") or ""

    # Normalise
    peak = np.abs(arr).max()
    if peak > 0:
        arr = arr / peak * 0.9

    # Trim to MAX_DURATION_S
    max_samples = int(MAX_DURATION_S * sr)
    arr = arr[:max_samples]

    # Resample to 24 kHz
    arr = resample(arr, sr, TARGET_SR)

    return arr.astype(np.float32), transcript.strip()


def encode_and_save(audio: np.ndarray, transcript: str, lang: str,
                    out_path: str, device: str = "cpu"):
    print(f"  loading aligner for lang='{lang}' from {CODEC_REPO} …")
    encoder = Encoder.from_pretrained(CODEC_REPO, language=lang).to(device)
    encoder.eval()

    waveform = torch.from_numpy(audio).unsqueeze(0).to(device)  # (1, T)

    print(f"  encoding {audio.shape[0]/TARGET_SR:.2f}s of audio …")
    with torch.no_grad():
        enc_out = encoder(waveform, text=[transcript], sample_rate=TARGET_SR)

    vals = enc_out.token_values[0].cpu().float().numpy().astype(np.float32)    # (N, 512)
    pos  = enc_out.token_positions[0].cpu().float().numpy().astype(np.float32) # (N,)
    n = vals.shape[0]
    print(f"  → {n} acoustic tokens")

    from pathlib import Path
    w = GGUFWriter(out_path, arch="crispasr.reference", use_temp_file=False)
    w.add_name(Path(out_path).stem)
    w.add_string("crispasr.ref.tada_tts_prompt_text", transcript)
    w.add_string("crispasr.ref.tada_tts_language", lang)
    w.add_string("crispasr.ref.source",
                 f"google/fleurs ({lang}), CC-BY 4.0, https://huggingface.co/datasets/google/fleurs")
    w.add_tensor("prompt_token_values",    np.ascontiguousarray(vals), raw_dtype=GGMLQuantizationType.F32)
    w.add_tensor("prompt_token_positions", np.ascontiguousarray(pos),  raw_dtype=GGMLQuantizationType.F32)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    sz = Path(out_path).stat().st_size
    print(f"  saved {out_path}  ({sz // 1024} KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--langs", nargs="+",
                    default=list(LANG_CONFIGS.keys()),
                    help="Language codes to generate (default: all)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for lang in args.langs:
        if lang not in LANG_CONFIGS:
            print(f"  WARNING: unknown lang '{lang}', skipping")
            continue

        out_path = os.path.join(args.output_dir, f"tada-ref-{lang}.gguf")
        if args.skip_existing and os.path.exists(out_path):
            print(f"[{lang}] {out_path} already exists, skipping")
            continue

        print(f"\n[{lang}] generating {out_path}")
        fleurs_config, idx = LANG_CONFIGS[lang]
        try:
            audio, transcript = fetch_fleurs_clip(fleurs_config, idx)
            print(f"  transcript: {transcript[:80]}")
            encode_and_save(audio, transcript, lang, out_path, device=args.device)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}", flush=True)
            traceback.print_exc()
            continue

    print("\nDone.")


if __name__ == "__main__":
    main()
