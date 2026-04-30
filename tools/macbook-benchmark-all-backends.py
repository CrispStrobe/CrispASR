#!/usr/bin/env python3
"""
CrispASR — Comprehensive Backend Benchmark on macOS / Apple Silicon

Adaptation of tools/kaggle-benchmark-all-backends.py for local M1/M2/M3 use.

Differences from the Kaggle script:
  - Doesn't clone or build CrispASR — assumes ./build/bin/crispasr already
    built via `scripts/dev-build.sh` (Ninja + ccache + libomp + Metal).
  - Models live on the attached SSD (default /Volumes/backups/ai/). Each
    backend's model is checked there first; if missing, downloaded once
    via huggingface_hub and stored on the same SSD for future runs (no
    cleanup — there's space).
  - Default backend is Metal. Pass --cpu to also measure CPU-only times
    side-by-side (useful for spotting Metal regressions or driver issues).
  - No Kaggle-secret machinery. HF token via env (HF_TOKEN) or
    `huggingface-cli login`. Optional gist upload via GH_GIST_TOKEN env.

Run from repo root:
  python tools/macbook-benchmark-all-backends.py
  python tools/macbook-benchmark-all-backends.py --cpu          # also CPU times
  python tools/macbook-benchmark-all-backends.py --slow         # incl. 3-4B models
  python tools/macbook-benchmark-all-backends.py --models /Volumes/backups/ai
  python tools/macbook-benchmark-all-backends.py --gist         # upload to gist
  python tools/macbook-benchmark-all-backends.py --backends whisper,parakeet
"""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CRISPASR = REPO_ROOT / "build" / "bin" / "crispasr"
JFK_WAV = REPO_ROOT / "samples" / "jfk.wav"

JFK_REF = (
    "and so my fellow americans ask not what your country can do for you "
    "ask what you can do for your country"
)

# Backend → (display_name, timeout_s, notes)
BACKENDS = [
    ("firered-asr",       "FireRed ASR2 AED",          90, "Q4_K, 900M params"),
    ("whisper",           "Whisper (base)",            60, "ggml-base.bin"),
    ("parakeet",          "Parakeet TDT 0.6B",         60, "Q4_K"),
    ("moonshine",         "Moonshine Tiny",            30, "Q4_K, 27M params"),
    ("wav2vec2",          "Wav2Vec2 XLSR-EN",          60, "Q4_K, 300M params"),
    ("fastconformer-ctc", "FastConformer CTC Large",   30, "Q4_K, 120M params"),
    ("data2vec",          "Data2Vec Base",             30, "Q4_K, 95M params"),
    ("hubert",            "HuBERT Large",              60, "Q4_K, 300M params"),
    ("canary",            "Canary 1B",                120, "Q4_K, 1B params"),
    ("cohere",            "Cohere Transcribe",        120, "Q4_K, 2B params"),
    ("qwen3",             "Qwen3 ASR 0.6B",            60, "Q4_K"),
    ("omniasr",           "OmniASR CTC 1B v2",        120, "Q4_K, 975M params"),
    ("omniasr-llm",       "OmniASR LLM 300M",         120, "Q4_K, 300M+1.3B params"),
    ("glm-asr",           "GLM ASR Nano",              90, "Q4_K, 1.3B params"),
    ("kyutai-stt",        "Kyutai STT 1B",             90, "Q4_K, 1B params"),
    ("vibevoice",         "VibeVoice ASR",             90, "Q4_K, 4.5B params"),
]

SLOW_BACKENDS = [
    ("voxtral",   "Voxtral Mini 3B",      300, "Q4_K, 3B params"),
    ("voxtral4b", "Voxtral 4B Realtime",  300, "Q4_K, 4B params"),
    ("granite",   "Granite Speech 1B",    300, "Q4_K, 2.9B params"),
]

# Per-backend model registry: (local_filename, hf_repo, hf_filename)
# local_filename is what we look for on disk; download saves under same name.
MODEL_REGISTRY = {
    "whisper":           ("ggml-base.bin",                       "ggerganov/crispasr",                     "ggml-base.bin"),
    "parakeet":          ("parakeet-tdt-0.6b-v3-q4_k.gguf",      "cstr/parakeet-tdt-0.6b-v3-GGUF",            "parakeet-tdt-0.6b-v3-q4_k.gguf"),
    "moonshine":         ("moonshine-tiny-q4_k.gguf",            "cstr/moonshine-tiny-GGUF",                  "moonshine-tiny-q4_k.gguf"),
    "wav2vec2":          ("wav2vec2-xlsr-en-q4_k.gguf",          "cstr/wav2vec2-large-xlsr-53-english-GGUF",  "wav2vec2-xlsr-en-q4_k.gguf"),
    "fastconformer-ctc": ("stt-en-fastconformer-ctc-large-q4_k.gguf", "cstr/stt-en-fastconformer-ctc-large-GGUF", "stt-en-fastconformer-ctc-large-q4_k.gguf"),
    "data2vec":          ("data2vec-audio-base-960h-q4_k.gguf",  "cstr/data2vec-audio-960h-GGUF",             "data2vec-audio-base-960h-q4_k.gguf"),
    "hubert":            ("hubert-large-ls960-ft-q4_k.gguf",     "cstr/hubert-large-ls960-ft-GGUF",           "hubert-large-ls960-ft-q4_k.gguf"),
    "canary":            ("canary-1b-v2-q4_k.gguf",              "cstr/canary-1b-v2-GGUF",                    "canary-1b-v2-q4_k.gguf"),
    "cohere":            ("cohere-transcribe-q4_k.gguf",         "cstr/cohere-transcribe-03-2026-GGUF",       "cohere-transcribe-q4_k.gguf"),
    "qwen3":             ("qwen3-asr-0.6b-q4_k.gguf",            "cstr/qwen3-asr-0.6b-GGUF",                  "qwen3-asr-0.6b-q4_k.gguf"),
    "omniasr":           ("omniasr-ctc-1b-v2-q4_k.gguf",         "cstr/omniASR-CTC-1B-v2-GGUF",               "omniasr-ctc-1b-v2-q4_k.gguf"),
    "omniasr-llm":       ("omniasr-llm-300m-v2-q4_k.gguf",       "cstr/omniasr-llm-300m-v2-GGUF",             "omniasr-llm-300m-v2-q4_k.gguf"),
    "glm-asr":           ("glm-asr-nano-q4_k.gguf",              "cstr/glm-asr-nano-GGUF",                    "glm-asr-nano-q4_k.gguf"),
    "firered-asr":       ("firered-asr2-aed-q4_k.gguf",          "cstr/firered-asr2-aed-GGUF",                "firered-asr2-aed-q4_k.gguf"),
    "kyutai-stt":        ("kyutai-stt-1b-q4_k.gguf",             "cstr/kyutai-stt-1b-GGUF",                   "kyutai-stt-1b-q4_k.gguf"),
    "vibevoice":         ("vibevoice-asr-7b-q4_k-fixed.gguf",    "cstr/vibevoice-asr-GGUF",                   "vibevoice-asr-q4_k.gguf"),  # use fixed GGUF (original had tokenizer + lm_head bugs)
    "voxtral":           ("voxtral-mini-3b-2507-q4_k.gguf",      "cstr/voxtral-mini-3b-2507-GGUF",            "voxtral-mini-3b-2507-q4_k.gguf"),
    "voxtral4b":         ("voxtral-mini-4b-realtime-q4_k.gguf",  "cstr/voxtral-mini-4b-realtime-GGUF",        "voxtral-mini-4b-realtime-q4_k.gguf"),
    "granite":           ("granite-speech-4.0-1b-q4_k.gguf",     "cstr/granite-speech-4.0-1b-GGUF",           "granite-speech-4.0-1b-q4_k.gguf"),
}

# Some backends need extra files (tokenizer, voice, alt-name fallbacks).
EXTRA_FILES = {
    "moonshine": [("tokenizer.bin", "cstr/moonshine-tiny-GGUF", "tokenizer.bin")],
}


def find_local_model(models_dir: Path, local_name: str, hf_filename: str) -> Path | None:
    """Look for the model under models_dir, accepting either the local
    convention name or the HF-canonical name."""
    for candidate in (local_name, hf_filename):
        p = models_dir / candidate
        if p.is_file():
            return p
    return None


def ensure_model(backend: str, models_dir: Path) -> tuple[Path | None, float]:
    """Return (local_path, download_seconds). Tries SSD first, then HF download
    into models_dir. Skip if backend isn't in registry."""
    if backend not in MODEL_REGISTRY:
        return None, 0.0
    local_name, repo, hf_name = MODEL_REGISTRY[backend]
    p = find_local_model(models_dir, local_name, hf_name)
    if p:
        return p, 0.0

    print(f"    {local_name} not on SSD — downloading from HF…")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("    ! huggingface_hub not installed — pip install huggingface_hub hf_xet")
        return None, 0.0

    t0 = time.time()
    try:
        downloaded = hf_hub_download(repo, hf_name, local_dir=str(models_dir))
        elapsed = time.time() - t0
        sz_mb = os.path.getsize(downloaded) / 1024 / 1024
        print(f"    ✓ {hf_name} → {downloaded} ({sz_mb:.0f} MB in {elapsed:.1f}s)")
        # Extras (tokenizer etc.)
        for extra_local, extra_repo, extra_name in EXTRA_FILES.get(backend, []):
            if not (models_dir / extra_local).is_file():
                try:
                    hf_hub_download(extra_repo, extra_name, local_dir=str(models_dir))
                except Exception:
                    pass
        return Path(downloaded), elapsed
    except Exception as e:
        print(f"    ✗ download failed: {e}")
        return None, time.time() - t0


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", text.lower())).strip()


def calc_wer(ref: str, hyp: str) -> float | None:
    try:
        from jiwer import wer as compute_wer
    except ImportError:
        return None
    r, h = normalize(ref), normalize(hyp)
    if not r or not h:
        return 1.0
    return compute_wer(r, h)


def run_one(crispasr: Path, model: Path, backend: str, audio: Path, use_gpu: bool,
            timeout: int, audio_duration: float, verbose: bool = True) -> dict:
    """Run a single inference, return parsed result dict.

    `realtime_factor` is wall-clock RT (audio / wall). For one-shot CLI
    runs this includes process startup + model load + (cold) Metal kernel
    JIT — see the `--warmup` flag for steady-state numbers."""
    cmd = [
        str(crispasr),
        "--backend", backend,
        "-m", str(model),
        "-f", str(audio),
        "--no-prints",
        "-bs", "1",
    ]
    if verbose:
        cmd.append("-v")
    if not use_gpu:
        cmd.append("-ng")
    env = {**os.environ, "CRISPASR_VERBOSE": "1"}
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, env=env)
        elapsed = time.time() - t0
        ok = r.returncode == 0
        out, err = r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "wall_s": timeout, "transcript": ""}

    transcript = re.sub(r"\[[\d:.]+\s*-->\s*[\d:.]+\]\s*", "", out.strip()).strip()
    rt = audio_duration / elapsed if elapsed > 0 else None

    # Try to extract a model-load hint from stderr (best-effort; many
    # backends don't emit anything useful).
    load_s = None
    m = re.search(r"library_init:\s+loaded in\s+([\d.]+)\s+sec", err or "")
    if m:
        load_s = float(m.group(1))

    if not ok:
        return {"status": "CRASH", "wall_s": round(elapsed, 2), "transcript": "",
                "stderr_tail": (err or "")[-400:]}
    if not transcript:
        return {"status": "EMPTY", "wall_s": round(elapsed, 2), "transcript": "",
                "stderr_tail": (err or "")[-200:]}

    return {
        "status": "OK",
        "wall_s": round(elapsed, 2),
        "inference_s": round(elapsed, 2),
        "metal_lib_load_s": load_s,
        "realtime_factor": round(rt, 2) if rt else None,
        "transcript": transcript,
    }


def benchmark_backend(backend: str, display: str, timeout: int, notes: str,
                      models_dir: Path, audio: Path, audio_duration: float,
                      run_cpu: bool, warmup: bool, verbose: bool) -> dict:
    print(f"\n{'='*60}\n  {display}  (--backend {backend})\n{'='*60}")

    result = {
        "backend": backend, "display_name": display, "notes": notes,
        "model_path": None, "download_s": 0.0, "model_size_mb": None,
        "metal": None, "cpu": None,
    }

    model, dl = ensure_model(backend, models_dir)
    result["download_s"] = round(dl, 1)
    if not model:
        result["status"] = "NO_MODEL"
        print(f"  ✗ no model available")
        return result
    result["model_path"] = str(model)
    result["model_size_mb"] = round(os.path.getsize(model) / 1024 / 1024, 1)

    # Metal run (optional warmup first to prime kernel cache + page cache)
    if warmup:
        print("  → Metal warmup…", flush=True)
        w0 = run_one(CRISPASR, model, backend, audio, use_gpu=True,
                     timeout=timeout, audio_duration=audio_duration, verbose=verbose)
        if w0["status"] == "OK":
            print(f"    cold: {w0.get('realtime_factor', '?')}x RT  ({w0.get('wall_s', '?')}s wall)")
        result["metal_cold"] = w0

    print("  → Metal…", flush=True)
    metal = run_one(CRISPASR, model, backend, audio, use_gpu=True,
                    timeout=timeout, audio_duration=audio_duration, verbose=verbose)
    if metal.get("transcript"):
        w = calc_wer(JFK_REF, metal["transcript"])
        if w is not None:
            metal["wer"] = round(w, 4)
    result["metal"] = metal
    if metal["status"] == "OK":
        print(f"    {metal.get('realtime_factor', '?')}x RT  "
              f"WER={metal.get('wer', '?')}  wall={metal.get('wall_s', '?')}s")
        print(f"    out: {metal['transcript'][:80]}")
    else:
        print(f"    ✗ {metal['status']} — {metal.get('stderr_tail', '')[:200]}")

    # Optional CPU comparison
    if run_cpu:
        if warmup:
            print("  → CPU warmup…", flush=True)
            run_one(CRISPASR, model, backend, audio, use_gpu=False,
                    timeout=timeout, audio_duration=audio_duration, verbose=verbose)
        print("  → CPU (-ng)…", flush=True)
        cpu = run_one(CRISPASR, model, backend, audio, use_gpu=False,
                      timeout=timeout, audio_duration=audio_duration, verbose=verbose)
        if cpu.get("transcript"):
            w = calc_wer(JFK_REF, cpu["transcript"])
            if w is not None:
                cpu["wer"] = round(w, 4)
        result["cpu"] = cpu
        if cpu["status"] == "OK":
            print(f"    {cpu.get('realtime_factor', '?')}x RT  "
                  f"WER={cpu.get('wer', '?')}  wall={cpu.get('wall_s', '?')}s")

    return result


def emit_markdown(results: list[dict], sysinfo: dict, run_cpu: bool, warmup: bool) -> str:
    md = []
    md.append("# CrispASR Backend Benchmark — macOS\n")
    md.append(f"**Date:** {sysinfo['date']}  ")
    md.append(f"**Platform:** {sysinfo['platform']}  ")
    md.append(f"**Audio:** {sysinfo['audio']}  ")
    md.append(f"**Reference:** _{JFK_REF}_  ")
    if warmup:
        md.append("**Mode:** warmup + measured (RT = audio / measured-wall, with kernel/page cache hot)\n")
    else:
        md.append("**Mode:** cold (RT = audio / wall, includes process startup + model load + Metal kernel JIT)\n")

    head = ["#", "Backend", "Status", "Warm RT", "Warm wall (s)", "WER"]
    if warmup:
        head += ["Cold RT", "Cold wall (s)"]
    if run_cpu:
        head += ["CPU RT", "CPU wall (s)", "CPU WER"]
    head += ["Model (MB)", "Notes"]
    md.append("| " + " | ".join(head) + " |")
    md.append("|" + "|".join(["---"] * len(head)) + "|")

    def col(d, k):
        if d is None or d.get("status") != "OK":
            return d.get("status", "—") if d else "—"
        v = d.get(k)
        if k == "realtime_factor":
            return f"{v:.1f}×" if v else "—"
        if k == "wer":
            return f"{v:.1%}" if v is not None else "—"
        if k in ("inference_s", "wall_s"):
            return f"{v:.1f}" if v else "—"
        return str(v or "—")

    for i, r in enumerate(results, 1):
        m = r.get("metal") or {}
        cold = r.get("metal_cold") or {}
        c = r.get("cpu") or {}
        status = m.get("status", r.get("status", "?"))
        size = f"{r['model_size_mb']:.0f}" if r.get("model_size_mb") else "—"
        row = [str(i), f"**{r['display_name']}**", status,
               col(m, "realtime_factor"), col(m, "wall_s"), col(m, "wer")]
        if warmup:
            row += [col(cold, "realtime_factor"), col(cold, "wall_s")]
        if run_cpu:
            row += [col(c, "realtime_factor"), col(c, "wall_s"), col(c, "wer")]
        row += [size, r.get("notes", "")]
        md.append("| " + " | ".join(row) + " |")

    # Speed ranking on Metal
    speed = [r for r in results if (r.get("metal") or {}).get("realtime_factor")]
    if speed:
        speed.sort(key=lambda r: r["metal"]["realtime_factor"], reverse=True)
        md.append("\n## Metal speed ranking (fastest first)\n")
        for i, r in enumerate(speed, 1):
            mr = r["metal"]
            md.append(f"{i}. **{r['display_name']}** — "
                      f"{mr['realtime_factor']:.1f}× RT, WER {mr.get('wer', 0):.1%}")

    return "\n".join(md)


def maybe_gist(report_md: str, payload: dict) -> str | None:
    token = os.environ.get("GH_GIST_TOKEN")
    if not token:
        return None
    import urllib.request
    body = {
        "description": f"CrispASR macOS benchmark — {payload['system']['date']}",
        "public": True,
        "files": {
            "benchmark_results.md": {"content": report_md},
            "benchmark_results.json": {"content": json.dumps(payload, indent=2)},
        },
    }
    req = urllib.request.Request(
        "https://api.github.com/gists",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"token {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())["html_url"]
    except Exception as e:
        print(f"gist upload failed: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="/Volumes/backups/ai",
                    help="Directory holding model files (also used as HF download target)")
    ap.add_argument("--audio", default=str(JFK_WAV),
                    help="Audio file to transcribe")
    ap.add_argument("--cpu", action="store_true",
                    help="Also run each backend with -ng (CPU) for side-by-side comparison")
    ap.add_argument("--slow", action="store_true",
                    help="Include voxtral / voxtral4b / granite (3-4B params, slow)")
    ap.add_argument("--backends", default=None,
                    help="Comma-separated subset of backends to run (default: all fast)")
    ap.add_argument("--results", default=str(REPO_ROOT / "build" / "benchmark"),
                    help="Where to write results .md and .json")
    ap.add_argument("--gist", action="store_true",
                    help="Upload to GitHub gist (uses GH_GIST_TOKEN env)")
    ap.add_argument("--no-warmup", dest="warmup", action="store_false",
                    help="Skip the warmup pass; report cold-start wall RT instead "
                         "(default: warmup ON for steady-state numbers)")
    ap.add_argument("--quiet", dest="verbose", action="store_false",
                    help="Don't pass -v to crispasr (default: verbose ON)")
    args = ap.parse_args()

    if not CRISPASR.is_file():
        print(f"ERROR: {CRISPASR} not built. Run scripts/dev-build.sh first.")
        sys.exit(1)
    if not Path(args.audio).is_file():
        print(f"ERROR: audio file not found: {args.audio}")
        sys.exit(1)

    models_dir = Path(args.models)
    if not models_dir.is_dir():
        print(f"ERROR: models dir not found: {models_dir} "
              "(plug in the SSD or pass --models PATH)")
        sys.exit(1)

    # Test set
    backend_set = BACKENDS + (SLOW_BACKENDS if args.slow else [])
    if args.backends:
        wanted = {b.strip() for b in args.backends.split(",")}
        backend_set = [b for b in backend_set if b[0] in wanted]
        if not backend_set:
            print(f"ERROR: no matching backends in --backends={args.backends}")
            sys.exit(1)

    with wave.open(str(JFK_WAV)) as wf:
        audio_duration = wf.getnframes() / wf.getframerate()

    sysinfo = {
        "date":      datetime.now().strftime("%Y-%m-%d %H:%M %Z"),
        "platform":  platform.platform(),
        "machine":   platform.machine(),
        "audio":     f"{Path(args.audio).name} ({audio_duration:.1f}s)",
        "models_dir": str(models_dir),
        "git":       subprocess.run(["git", "log", "--oneline", "-1"],
                                    capture_output=True, text=True,
                                    cwd=REPO_ROOT).stdout.strip(),
    }
    print(f"CrispASR macOS benchmark — {sysinfo['date']}")
    print(f"  HEAD: {sysinfo['git']}")
    print(f"  Models: {models_dir}  ({len(backend_set)} backends, "
          f"{'Metal+CPU' if args.cpu else 'Metal only'}, "
          f"{'warmup' if args.warmup else 'cold'}, "
          f"{'verbose' if args.verbose else 'quiet'})")

    sysinfo["warmup"] = args.warmup
    sysinfo["verbose"] = args.verbose

    results = []
    for backend, display, timeout, notes in backend_set:
        results.append(benchmark_backend(
            backend, display, timeout, notes,
            models_dir, Path(args.audio), audio_duration,
            args.cpu, args.warmup, args.verbose))

    # Output
    Path(args.results).mkdir(parents=True, exist_ok=True)
    payload = {"system": sysinfo, "results": results}
    json_path = Path(args.results) / "macbook_benchmark.json"
    json_path.write_text(json.dumps(payload, indent=2))

    md = emit_markdown(results, sysinfo, args.cpu, args.warmup)
    md_path = Path(args.results) / "macbook_benchmark.md"
    md_path.write_text(md)

    print("\n" + "=" * 60)
    print("  Done")
    print("=" * 60)
    print(f"  JSON:     {json_path}")
    print(f"  Markdown: {md_path}")
    print()
    print(md)

    if args.gist:
        url = maybe_gist(md, payload)
        if url:
            print(f"\nGist: {url}")


if __name__ == "__main__":
    main()
