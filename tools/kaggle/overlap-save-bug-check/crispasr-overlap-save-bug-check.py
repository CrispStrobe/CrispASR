"""
CrispASR — overlap-save A/B harness on Kaggle CPU for the inconclusive backends

PLAN #114 audit follow-up. The local-M1 run of `tools/check-overlap-save-bug.sh`
on 2026-05-25 came back inconclusive for granite-4.1 and omniasr-llm because
the per-run wallclock budget (~20 min on M1) wasn't enough for those two LLM-AR
backends to finish a 5-min clip. Kaggle CPU has 4 vCPUs + 30 GB RAM + a 9-hour
wall budget — fine for two backends × two runs (default overlap vs nooverlap)
× ~90 s audio.

Scope:
  - Build crispasr CPU-only from current main.
  - Download granite-4.1 + omniasr-llm GGUFs.
  - Synthesize a ~90 s audio by concatenating samples/jfk.wav eight times.
  - Run each backend twice (default --chunk-overlap 3.0 vs --chunk-overlap 0).
  - Compare SRT outputs: last timestamp, segment count, char count.
  - If nooverlap produces materially more output → SUSPECTED-BUG, add backend
    to kBlocked in examples/cli/crispasr_chunk_context_gate.h.

Patterns lifted from tools/kaggle/mimo-asr-cpu-vs-gpu/ (HISTORY 2026-05-26).
"""

import os
import re
import subprocess
import sys
import wave
from pathlib import Path

# ── Working dirs ────────────────────────────────────────────────────────
ROOT = Path("/kaggle/working")
REPO = ROOT / "CrispASR"
BUILD = ROOT / "build"
MODELS = ROOT / "models"
AUDIO_DIR = ROOT / "audio"
OUT_DIR = ROOT / "results"

for d in (BUILD, MODELS, AUDIO_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Step 1: clone (minimal inline; shared harness only importable after) ─
# The harness lives in the repo we are about to clone, so the clone itself
# uses a tiny inline runner. Everything after the import goes through `kh`.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

CRISPASR_REF = os.environ.get("CRISPASR_REF", "main")
print(f"[clone] ref={CRISPASR_REF}", flush=True)
if not REPO.exists():
    _clone_cmd = (
        f"git clone --depth 1 --branch {CRISPASR_REF} "
        f"https://github.com/CrispStrobe/CrispASR.git {REPO}"
    )
else:
    _clone_cmd = f"git -C {REPO} pull --ff-only"
print(f"$ {_clone_cmd}", flush=True)
if subprocess.run(_clone_cmd, shell=True).returncode != 0:
    raise SystemExit(f"clone failed: {_clone_cmd}")
sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()

# ── Shared harness: line-buffered I/O, progress JSONL, heartbeat, ccache/mold,
#    3-tier HF auth — replaces the previously copy-pasted local helpers. ──
sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress(progress_path=str(ROOT / "progress.jsonl"))
kh.resolve_hf_token()
kh.step("script.start")
kh.step("clone.done", sha=sha)

# ── Step 2: CPU-only build (ccache + mold via shared harness) ───────────

kh.step("build.begin")
BUILD.mkdir(exist_ok=True)
kh.install_build_toolchain()  # apt ninja + ccache + mold, primes CCACHE_DIR
cmake_flags = [
    f"cmake {REPO} -B{BUILD} -GNinja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_SHARED_LIBS=ON",
    "-DCRISPASR_BUILD_TESTS=OFF",
] + kh.cache_and_link_flags()  # ccache launchers + mold linker flags
cmake_cmd = " ".join(cmake_flags)
njobs = kh.safe_build_jobs(gpu=False)
with kh.build_heartbeat("cmake-configure"):
    kh.sh_with_progress(cmake_cmd)
kh.step("build.configured")
with kh.build_heartbeat("cmake-build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli -- -j{njobs}"
    )
CRISPASR = BUILD / "bin" / "crispasr"
assert CRISPASR.is_file(), f"crispasr binary missing at {CRISPASR}"
kh.step("build.done", binary=str(CRISPASR))

# ── Step 3: download GGUFs (granite-4.1 + omniasr-llm) ──────────────────

kh.step("download.begin")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
kh.sh_with_progress("pip install -q huggingface_hub hf_transfer")
from huggingface_hub import hf_hub_download

# The two inconclusive backends from the M1 harness run on 2026-05-25.
# Add more here to widen the audit later.
BACKEND_GGUFS = [
    ("granite-4.1", "cstr/granite-speech-4.1-2b-GGUF", "granite-speech-4.1-2b-q4_k.gguf"),
    ("omniasr-llm", "cstr/omniasr-llm-300m-v2-GGUF", "omniasr-llm-300m-v2-q4_k.gguf"),
]
backend_model = {}
for backend, repo_id, fname in BACKEND_GGUFS:
    kh.step(f"download.{backend}.begin", repo=repo_id, file=fname)
    p = hf_hub_download(repo_id=repo_id, filename=fname, local_dir=str(MODELS), local_dir_use_symlinks=False)
    backend_model[backend] = Path(p)
    kh.step(f"download.{backend}.done", path=p, size_mib=Path(p).stat().st_size // (1 << 20))
kh.step("download.done")

# ── Step 4: build a ~88 s audio by concatenating samples/jfk.wav 8x ─────

kh.step("audio.begin")
SRC_WAV = REPO / "samples" / "jfk.wav"
LONG_WAV = AUDIO_DIR / "jfk_x8.wav"
assert SRC_WAV.is_file(), f"missing jfk.wav at {SRC_WAV}"
with wave.open(str(SRC_WAV), "rb") as src:
    params = src.getparams()
    data = src.readframes(src.getnframes())
n_copies = 8
with wave.open(str(LONG_WAV), "wb") as dst:
    dst.setparams(params)
    for _ in range(n_copies):
        dst.writeframes(data)
src_secs = params.nframes / params.framerate
dst_secs = src_secs * n_copies
kh.step("audio.done", file=str(LONG_WAV), secs=round(dst_secs, 1))

# ── Step 5: A/B sweep ───────────────────────────────────────────────────


def parse_srt(srt_path: Path) -> dict:
    """Mirror parse_srt() in tools/check-overlap-save-bug.sh."""
    if not srt_path.is_file() or srt_path.stat().st_size == 0:
        return {"last_s": 0, "segs": 0, "chars": 0}
    text = srt_path.read_text(errors="replace")
    times = re.findall(r"\d{2}:\d{2}:\d{2}[,.]\d{3} --> (\d{2}:\d{2}:\d{2})[,.]\d{3}", text)
    if not times:
        return {"last_s": 0, "segs": 0, "chars": len(text)}
    last_s = max(
        int(h) * 3600 + int(m) * 60 + int(s) for h, m, s in (t.split(":") for t in times)
    )
    segs = text.count(" --> ")
    chars = sum(
        len(line) for line in text.splitlines() if line and not line[0].isdigit() and " --> " not in line
    )
    return {"last_s": last_s, "segs": segs, "chars": chars}


def run_one(backend: str, model: Path, label: str, extra: list[str]) -> dict:
    out_stem = OUT_DIR / f"{backend}-{label}"
    for ext in (".txt", ".srt", ".json"):
        f = out_stem.with_suffix(ext)
        if f.exists():
            f.unlink()
    cmd = [
        str(CRISPASR),
        "-m", str(model),
        "--backend", backend,
        "-f", str(LONG_WAV),
        "-of", str(out_stem),
        "-osrt",
        "-np",
    ] + extra
    kh.step(f"run.{backend}.{label}.begin", cmd=" ".join(cmd))
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        rc = r.returncode
        log = (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired as e:
        rc = -1
        log = f"TIMEOUT after {round(time.time() - t0, 1)}s\n" + (e.stdout or "") + (e.stderr or "")
    dt = time.time() - t0
    srt_stats = parse_srt(out_stem.with_suffix(".srt"))
    kh.step(f"run.{backend}.{label}.done", rc=rc, wall_s=round(dt, 1), **srt_stats)
    if rc != 0:
        print(f"--- last 30 lines of log for {backend}/{label} ---", flush=True)
        for line in log.splitlines()[-30:]:
            print(line, flush=True)
    return {"rc": rc, "wall_s": round(dt, 1), **srt_stats}


verdicts = {}
for backend, _, _ in BACKEND_GGUFS:
    model = backend_model[backend]
    kh.step(f"sweep.{backend}.begin")
    default = run_one(backend, model, "default", ["--chunk-overlap", "3.0"])
    nooverlap = run_one(backend, model, "nooverlap", ["--chunk-overlap", "0"])
    # Verdict (mirror tools/check-overlap-save-bug.sh logic): if nooverlap
    # produces materially more output (later last_s or larger chars), the
    # backend has the bug. Threshold: 25 % more chars or >= 10 s later.
    chars_d, chars_n = default["chars"], nooverlap["chars"]
    last_d, last_n = default["last_s"], nooverlap["last_s"]
    chars_delta = chars_n - chars_d
    last_delta = last_n - last_d
    rel_chars = (chars_delta / chars_d) if chars_d > 0 else 0.0
    if rel_chars > 0.25 or last_delta >= 10:
        verdict = "SUSPECTED-BUG"
    elif default["rc"] != 0 and nooverlap["rc"] == 0:
        verdict = "DEFAULT-FAILS"
    elif default["rc"] != 0 and nooverlap["rc"] != 0:
        verdict = "BOTH-FAIL"
    else:
        verdict = "OK"
    verdicts[backend] = {
        "verdict": verdict,
        "default": default,
        "nooverlap": nooverlap,
        "rel_chars": round(rel_chars, 2),
        "last_delta_s": last_delta,
    }
    kh.step(
        f"sweep.{backend}.done",
        verdict=verdict,
        rel_chars=round(rel_chars, 2),
        last_delta_s=last_delta,
    )

# ── Summary ─────────────────────────────────────────────────────────────

kh.step("summary", verdicts=verdicts, sha=sha)
print("\n" + "=" * 76)
print(f"SUMMARY — overlap-save A/B on HEAD ({sha[:8]}) — Kaggle CPU")
print("=" * 76)
print(f"  audio: {LONG_WAV.name} ({round(dst_secs, 1)} s)")
for backend, v in verdicts.items():
    print()
    print(f"  {backend}: {v['verdict']}")
    print(f"    default  rc={v['default']['rc']:>3}  wall={v['default']['wall_s']:>6}s  "
          f"chars={v['default']['chars']:>6}  last_s={v['default']['last_s']:>4}")
    print(f"    nooverlap rc={v['nooverlap']['rc']:>3}  wall={v['nooverlap']['wall_s']:>6}s  "
          f"chars={v['nooverlap']['chars']:>6}  last_s={v['nooverlap']['last_s']:>4}")
    print(f"    rel_chars_delta={v['rel_chars']:+.2f}  last_delta_s={v['last_delta_s']:+d}")

print()
print("Interpretation:")
for backend, v in verdicts.items():
    if v["verdict"] == "SUSPECTED-BUG":
        print(f"  {backend}: add to kBlocked in examples/cli/crispasr_chunk_context_gate.h")
    elif v["verdict"] == "OK":
        print(f"  {backend}: opt-out NOT needed; leave default chunking on")
    elif v["verdict"] == "BOTH-FAIL":
        print(f"  {backend}: both arms failed — backend is broken at this length, separate bug")
    elif v["verdict"] == "DEFAULT-FAILS":
        print(f"  {backend}: default fails but nooverlap works — add to kBlocked + investigate")

kh._push_progress_to_hf(force=True)
kh.step("script.end")
