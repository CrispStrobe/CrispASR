"""benchmark_corpus.py — build and manage the ASR benchmark audio corpus.

Creates test audio in /mnt/storage/test-audio/ organized by language and
duration from two sources:

1. **FLEURS** (Google, CC-BY-4.0) — short clips (~10s) concatenated to
   target durations. Covers en, de, ja, zh.
2. **Existing files** — reporter's YouTube audio (ja), JFK sample (en).

Usage::

    python tests/benchmark_corpus.py                    # build all
    python tests/benchmark_corpus.py --lang ja          # just Japanese
    python tests/benchmark_corpus.py --lang de --force  # rebuild German
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

CORPUS_DIR = Path("/mnt/storage/test-audio")
FLEURS_CACHE = Path("/tmp/hf_cache")

# FLEURS language codes → our short codes
FLEURS_LANGS = {
    "en": "en_us",
    "de": "de_de",
    "ja": "ja_jp",
    "zh": "cmn_hans_cn",
}

# Target durations in seconds
DURATIONS = [10, 60, 300, 600]

# Existing audio that doesn't need FLEURS
EXISTING_AUDIO = {
    "en": [
        # JFK 11s — already in samples/
        {"src": "samples/jfk.wav", "name": "jfk_11s.wav", "duration": 11},
    ],
    "ja": [
        # Reporter's YouTube audio
        {"src": "/mnt/storage/samples/o_9dWkRPYC0.mp3", "duration": None},
    ],
}


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def _ffprobe_duration(path: str | Path) -> float:
    """Get audio duration in seconds via ffprobe."""
    r = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
              "-of", "csv=p=0", str(path)])
    return float(r.stdout.strip()) if r.returncode == 0 else 0.0


def _ffmpeg_convert(src: str, dst: str, duration: int | None = None) -> bool:
    """Convert audio to 16kHz mono WAV, optionally truncating."""
    cmd = ["ffmpeg", "-y", "-i", str(src), "-ar", "16000", "-ac", "1"]
    if duration:
        cmd += ["-t", str(duration)]
    cmd.append(str(dst))
    r = _run(cmd)
    return r.returncode == 0


def _ffmpeg_concat(wav_files: list[str], output: str, target_duration_s: int) -> bool:
    """Concatenate WAV files with 0.3s silence gaps up to target duration."""
    # Build a concat filter input list
    inputs = []
    total = 0.0
    for f in wav_files:
        if total >= target_duration_s:
            break
        dur = _ffprobe_duration(f)
        if dur <= 0:
            continue
        inputs.append(f)
        total += dur + 0.3  # account for silence gap

    if not inputs:
        return False

    # Use ffmpeg concat demuxer via a temp file list
    list_path = output + ".list"
    silence = output + ".silence.wav"
    # Generate 0.3s silence
    _run(["ffmpeg", "-y", "-f", "lavfi", "-i",
          "anullsrc=r=16000:cl=mono", "-t", "0.3", silence])

    with open(list_path, "w") as f:
        for i, inp in enumerate(inputs):
            f.write(f"file '{inp}'\n")
            if i < len(inputs) - 1:
                f.write(f"file '{silence}'\n")

    r = _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
              "-i", list_path, "-t", str(target_duration_s),
              "-ar", "16000", "-ac", "1", str(output)])

    # Cleanup temp files
    for p in [list_path, silence]:
        try:
            os.unlink(p)
        except OSError:
            pass

    return r.returncode == 0


def download_fleurs(lang: str) -> Path | None:
    """Download FLEURS test audio for a language. Returns extract dir."""
    fleurs_code = FLEURS_LANGS.get(lang)
    if not fleurs_code:
        print(f"  SKIP: no FLEURS code for '{lang}'")
        return None

    extract_dir = Path(f"/tmp/fleurs_{lang}")
    if extract_dir.exists() and any(extract_dir.rglob("*.wav")):
        print(f"  FLEURS {lang}: using cached {extract_dir}")
        return extract_dir

    try:
        from huggingface_hub import hf_hub_download
        tar_path = hf_hub_download(
            repo_id="google/fleurs",
            filename=f"data/{fleurs_code}/audio/test.tar.gz",
            repo_type="dataset",
            cache_dir=str(FLEURS_CACHE),
        )
        print(f"  FLEURS {lang}: downloaded {tar_path}")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path) as tf:
            tf.extractall(extract_dir, filter="data")
        return extract_dir
    except Exception as e:
        print(f"  FLEURS {lang}: download failed: {e}")
        return None


def build_language(lang: str, force: bool = False) -> list[dict]:
    """Build all duration variants for one language. Returns corpus entries."""
    lang_dir = CORPUS_DIR / lang
    lang_dir.mkdir(parents=True, exist_ok=True)
    entries = []

    # --- Existing audio (ja: reporter's YT, en: jfk) ---
    for item in EXISTING_AUDIO.get(lang, []):
        src = item["src"]
        if not Path(src).exists():
            # Try relative to repo root
            repo = Path(__file__).parent.parent
            src = str(repo / item["src"])
        if not Path(src).exists():
            print(f"  SKIP existing: {item['src']} not found")
            continue

        if "name" in item:
            dst = lang_dir / item["name"]
            if not dst.exists() or force:
                _ffmpeg_convert(src, str(dst))
            dur = _ffprobe_duration(dst)
            entries.append({
                "path": f"{lang}/{item['name']}",
                "language": lang,
                "duration_s": round(dur, 1),
                "source": item["src"],
            })
        else:
            # Create duration variants from long source
            src_dur = _ffprobe_duration(src)
            for target in DURATIONS:
                if target > src_dur:
                    continue
                name = f"yt_{target}s.wav"
                dst = lang_dir / name
                if not dst.exists() or force:
                    print(f"  {lang}/{name}: extracting {target}s from {Path(src).name}")
                    _ffmpeg_convert(src, str(dst), duration=target)
                dur = _ffprobe_duration(dst)
                entries.append({
                    "path": f"{lang}/{name}",
                    "language": lang,
                    "duration_s": round(dur, 1),
                    "source": f"{Path(src).name} first {target}s",
                })

    # --- FLEURS concatenations ---
    fleurs_dir = download_fleurs(lang)
    if fleurs_dir:
        wavs = sorted(str(p) for p in fleurs_dir.rglob("*.wav"))
        if wavs:
            # Single clip for <30s test
            name_short = "fleurs_10s.wav"
            dst_short = lang_dir / name_short
            if not dst_short.exists() or force:
                _ffmpeg_convert(wavs[0], str(dst_short))
            dur = _ffprobe_duration(dst_short)
            entries.append({
                "path": f"{lang}/{name_short}",
                "language": lang,
                "duration_s": round(dur, 1),
                "source": "FLEURS test clip",
            })

            # Concatenated durations
            for target in DURATIONS:
                if target <= 10:
                    continue
                name = f"fleurs_{target}s.wav"
                dst = lang_dir / name
                if not dst.exists() or force:
                    print(f"  {lang}/{name}: concatenating FLEURS clips to {target}s")
                    _ffmpeg_concat(wavs, str(dst), target)
                if dst.exists():
                    dur = _ffprobe_duration(dst)
                    entries.append({
                        "path": f"{lang}/{name}",
                        "language": lang,
                        "duration_s": round(dur, 1),
                        "source": f"FLEURS test concat {target}s",
                    })

    return entries


def build_corpus(langs: list[str] | None = None, force: bool = False) -> None:
    """Build the full audio corpus."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    Path("/mnt/storage/benchmark-results").mkdir(parents=True, exist_ok=True)

    if langs is None:
        langs = list(FLEURS_LANGS.keys())

    all_entries = []
    for lang in langs:
        print(f"\n=== Building {lang} ===")
        entries = build_language(lang, force=force)
        all_entries.extend(entries)
        print(f"  → {len(entries)} audio files")

    # Write corpus manifest
    manifest = CORPUS_DIR / "corpus.json"
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)
    print(f"\nCorpus manifest: {manifest} ({len(all_entries)} entries)")


def main():
    parser = argparse.ArgumentParser(description="Build ASR benchmark audio corpus")
    parser.add_argument("--lang", type=str, help="Build only this language (en/de/ja/zh)")
    parser.add_argument("--force", action="store_true", help="Rebuild existing files")
    args = parser.parse_args()

    langs = [args.lang] if args.lang else None
    build_corpus(langs=langs, force=args.force)


if __name__ == "__main__":
    main()
