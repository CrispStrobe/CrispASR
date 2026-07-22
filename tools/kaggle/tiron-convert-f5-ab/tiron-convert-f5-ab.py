# ─────────────────────────── cell 0 (markdown) ───────────────────────────
# # CrispASR — Tiron convert + validate  &  F5-TTS perf A/B  (clean CUDA GPU)
#
# Two independent phases on one uncontended CUDA box (T4/P100). Each is guarded
# so a failure in one does not waste the other.
#
# PHASE A — Tiron (#295): convert Trelis/tiron (Whisper large-v3 + inline
#   <|speakerN|> tokens, vocab 51904) to legacy GGML, quantize q8_0/q4_k, and
#   VALIDATE the new "tiron" decode mode on the CUDA build:
#     * multispeaker.wav must transcribe AND emit <|speakerN|> markers inline
#     * CRISPASR_WHISPER_TIRON=0 (stock path) as an A/B negative control
#     * jfk.wav single-speaker sanity (text must be recognizable)
#   Then upload f16/q8_0/q4_k + an apache-2.0 model card to cstr/tiron-GGML.
#
# PHASE B — F5-TTS (#294): settle on CUDA whether the opt-in perf gates are
#   wins (M1 Metal timing is dispatch-bound/noisy — needs a clean GPU verdict).
#   F16 only (flow-matching: q8 error compounds). Matrix, seed 42, fixed jfk
#   ref + fixed gen text, 1 warmup + 3 measured, median ode_solve ms + ASR
#   roundtrip proof-of-work (a perf win is a lie without proof — kaggle_usage #24):
#     A baseline (32 steps)          F CRISPASR_F5_BATCH_CFG=1
#     B --tts-steps 7 (EPSS)         G CRISPASR_F5_F16_ACT=1  (CUDA-only question)
#     C 7 + CFG_INTERVAL=2           H CRISPASR_F5_DIT_SKIP=2
#     D CRISPASR_F5_EMBED_GPU=1      I recommended CUDA stack (EMBED_GPU+BATCH_CFG+7+int2)
#     E EMBED_GPU + 7 steps
#
# Regime (kaggle_usage.md): repo/build/models staged on the ~70 GB ephemeral
# layer (/kaggle/temp), NOT the 20 GB /kaggle/working output mount (#18/#21/#22);
# only small logs + summary JSON land in /kaggle/working. ccache auto-warmed by
# the harness onto /kaggle/temp; ccache.tar exported at the end for refresh.
#
# Requirements: Kaggle GPU, Internet ON. Datasets (chr1s4, same account as id):
#   chr1s4/crispasr-hf-token, chr1s4/crispasr-ccache.

# ─────────────────────────── cell 1 (code) — config ──────────────────────
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

# Stage heavy work on the ~70 GB ephemeral layer; keep /kaggle/working tiny
# (it is the 20 GB output mount, page-capped at 500 files — kaggle_usage #22).
SCRATCH = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
WORK = Path("/kaggle/working")
REPO = SCRATCH / "CrispASR"
BUILD = SCRATCH / "build"
MODELS = SCRATCH / "models"
OUT = SCRATCH / "tiron-out"
AUDIO = SCRATCH / "audio"      # throwaway synthesized wavs (kept off /kaggle/working)
RESULTS = WORK / "results"     # retained output: logs + summary JSON only
for d in (MODELS, OUT, AUDIO, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

CRISPASR_REPO = os.environ.get("CRISPASR_REPO", "https://github.com/CrispStrobe/CrispASR.git")
CRISPASR_REF = os.environ.get("CRISPASR_REF", "feat/tiron-asr")

# Phase A
TIRON_SRC = "Trelis/tiron"
TIRON_HF_OUT = "cstr/tiron-GGML"
TIRON_NAME = "tiron"
TIRON_FILES = [
    "config.json", "generation_config.json", "preprocessor_config.json",
    "tokenizer_config.json", "special_tokens_map.json", "normalizer.json",
    "tokenizer.json", "vocab.json", "merges.txt", "added_tokens.json",
    "model.safetensors",
]

# Phase B
F5_HF_REPO = "cstr/f5-tts-GGUF"
F5_HF_FILE = "f5-tts-v1-base-f16.gguf"
F5_REF_TEXT = ("And so, my fellow Americans, ask not what your country can do for you, "
               "ask what you can do for your country.")
F5_GEN_TEXT = "The quick brown fox jumps over the lazy dog and then it ran away."
F5_GEN_WORDS = ["quick", "brown", "fox", "lazy", "dog", "ran"]
SEED = 42
N_WARMUP = 1
N_MEASURED = 3

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # hf_transfer wedges multi-GB on Kaggle (#-download)
os.environ["USE_TF"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"


def step(name, **kv):
    print(f"[{time.strftime('%H:%M:%S')}] STEP {name} " + json.dumps(kv), flush=True)


def run(cmd, check=True, env=None, cwd=None, timeout=None):
    e = {**os.environ, **(env or {})}
    r = subprocess.run(cmd, env=e, cwd=cwd, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.stdout:
        print(r.stdout, flush=True)
    # RuntimeError (not SystemExit): SystemExit derives from BaseException and
    # would bypass the Phase A/B `except Exception` guards, killing the whole
    # kernel on a single-phase failure instead of isolating it.
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(map(str, cmd))}")
    return r


# ─────────────────────────── cell 2 (code) — clone + build ───────────────
step("start", ref=CRISPASR_REF, scratch=str(SCRATCH))
if REPO.exists():
    shutil.rmtree(REPO)

# Robust clone + import (kaggle_usage "Auth via kaggle_harness"): prefer the
# cloned repo's harness, fall back to the copy bundled beside this script.
try:
    run(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF, "--recursive",
         CRISPASR_REPO, str(REPO)])
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
except Exception as ex:  # noqa: BLE001
    print(f"clone failed: {ex}", flush=True)
if str(REPO / "tools" / "kaggle") not in sys.path or "kaggle_harness" not in sys.modules:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh  # noqa: E402

# Live progress mirror to a public HF dataset so the run is watchable while it
# runs (kaggle_usage: never hallucinate progress). Resolve the token EARLY so
# the mirror + heartbeat are active from the build onward, not just at upload.
kh.init_progress(hf_progress_repo="cstr/crispasr-kaggle-progress")
kh._HF_PUSH_INTERVAL_S = 20.0
if not REPO.exists():
    raise SystemExit("repo clone missing — cannot build (need internet + GPU worker)")
step("cloned", sha=subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip())

token = kh.resolve_hf_token()  # exports HF_TOKEN (+ enables the progress mirror)
# ⚠ resolve_hf_token() flips HF_HUB_ENABLE_HF_TRANSFER back to "1"; hf_transfer
# wedges multi-GB Kaggle downloads with no resume — force it OFF for the Tiron
# safetensors (3 GB) + F5 f16 (~1 GB) pulls.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
step("hf_token", have=bool(token), hf_transfer=os.environ.get("HF_HUB_ENABLE_HF_TRANSFER"))

run(["nvidia-smi", "-L"])
gpu_name = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
step("gpu", gpu=gpu_name)

kh.install_build_toolchain()  # warms ccache onto /kaggle/temp/.ccache
arch = kh.detect_cuda_arch()
step("cuda_arch", arch=arch)

BUILD.mkdir(parents=True, exist_ok=True)
cmake_args = [
    "cmake", "-S", str(REPO), "-B", str(BUILD),
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_SHARED_LIBS=ON",
] + kh.cuda_build_flags(arch) + kh.cache_and_link_flags()  # cache_and_link_flags folds in NO_C2PA
run(cmake_args)
step("cmake_done")

with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli crispasr-quantize "
        f"-j{kh.safe_build_jobs(gpu=True)}")
step("build_done")


def _find(name):
    p = BUILD / "examples" / "cli" / name
    if p.exists():
        return p
    cands = [c for c in BUILD.rglob(name) if c.is_file() and os.access(c, os.X_OK)]
    if not cands:
        raise SystemExit(f"{name} binary not found after build")
    return cands[0]


CLI = _find("crispasr")
QUANT = _find("crispasr-quantize")
os.environ["LD_LIBRARY_PATH"] = f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"
SAMPLES = REPO / "samples"
step("bins", cli=str(CLI), quant=str(QUANT))


# ─────────────────────────── cell 3 (code) — shared helpers ──────────────
def wav_dur(path: Path) -> float:
    if not path.exists():
        return 0.0
    try:
        with wave.open(str(path), "rb") as w:
            return round(w.getnframes() / w.getframerate(), 3)
    except Exception:
        return 0.0


def md5(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def asr_roundtrip(wav: Path, timeout=600) -> str:
    """Transcribe with stock whisper (auto-download tiny/base) for proof-of-work."""
    if not wav.exists():
        return ""
    cmd = [str(CLI), "--backend", "whisper", "-m", "auto", "--auto-download", "-f", str(wav), "-np"]
    try:
        r = subprocess.run(cmd, timeout=timeout, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
        body = " ".join(ln for ln in lines
                        if not ln.startswith(("[", "whisper", "ggml", "load", "crispasr")))
        return body[-300:]
    except Exception as ex:  # noqa: BLE001
        return f"<asr-error: {type(ex).__name__}>"


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else round((xs[n // 2 - 1] + xs[n // 2]) / 2, 2)


# ═══════════════════════════ PHASE A — Tiron ═════════════════════════════
tiron_summary = {"phase": "tiron", "src": TIRON_SRC}
try:
    # transformers is needed by convert-h5-to-ggml.py; torch is pre-installed on
    # Kaggle (kaggle_usage #11 — never pip install torch).
    kh.sh_with_progress("pip install -q transformers safetensors")
    from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # defensive: no wedge on the 3 GB pull
    step("tiron.download.begin", free_gb=kh.free_gb(str(MODELS)))
    src_dir = MODELS / "tiron-src"
    src_dir.mkdir(parents=True, exist_ok=True)
    for fn in TIRON_FILES:
        p = hf_hub_download(repo_id=TIRON_SRC, filename=fn, local_dir=str(src_dir), token=token or None)
        step("tiron.download.file", file=fn, mb=round(os.path.getsize(p) / 1e6, 1))

    # OpenAI Whisper repo for the mel_128 filter assets the converter needs.
    whisper_repo = MODELS / "openai-whisper"
    if not whisper_repo.exists():
        run(["git", "clone", "--depth", "1", "https://github.com/openai/whisper.git", str(whisper_repo)])

    f16 = OUT / "ggml-model.bin"
    if f16.exists():
        f16.unlink()
    step("tiron.convert.begin", free_gb=kh.free_gb(str(OUT)))
    with kh.build_heartbeat("tiron.convert", interval_s=60):
        run(["python", "models/convert-h5-to-ggml.py", str(src_dir), str(whisper_repo), str(OUT)],
            cwd=str(REPO), timeout=3600)
    f16_named = OUT / f"{TIRON_NAME}-f16.bin"
    f16.rename(f16_named)
    step("tiron.convert.done", gb=round(f16_named.stat().st_size / 1e9, 3))

    # quantize (f16 + quant coexist on disk — staged on /kaggle/temp per #18)
    quants = {}
    for q in ("q8_0", "q4_k"):
        outq = OUT / f"{TIRON_NAME}-{q}.bin"
        step(f"tiron.quant.{q}.begin", free_gb=kh.free_gb(str(OUT)))
        with kh.build_heartbeat(f"tiron.quant.{q}", interval_s=60):
            run([str(QUANT), str(f16_named), str(outq), q], timeout=1800)
        quants[q] = outq
        step(f"tiron.quant.{q}.done", gb=round(outq.stat().st_size / 1e9, 3))

    # ── validate the tiron decode mode on the CUDA build ──
    def tiron_transcribe(wav, extra_env=None, print_special=True, timeout=600):
        cmd = [str(CLI), "--backend", "whisper", "-m", str(f16_named), "-f", str(wav), "-l", "en"]
        if print_special:
            cmd.append("-ps")
        r = subprocess.run(cmd, env={**os.environ, **(extra_env or {})}, timeout=timeout,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return r.returncode, r.stdout or ""

    ms_wav = SAMPLES / "multispeaker.wav"
    step("tiron.validate.begin", multispeaker=str(ms_wav), exists=ms_wav.exists())

    rc_on, out_on = tiron_transcribe(ms_wav)
    (RESULTS / "tiron_multispeaker_ON.log").write_text(out_on)
    n_speaker_markers = len(re.findall(r"<\|speaker\d\|>", out_on))

    rc_off, out_off = tiron_transcribe(ms_wav, extra_env={"CRISPASR_WHISPER_TIRON": "0"})
    (RESULTS / "tiron_multispeaker_OFF.log").write_text(out_off)
    n_speaker_off = len(re.findall(r"<\|speaker\d\|>", out_off))

    rc_jfk, out_jfk = tiron_transcribe(SAMPLES / "jfk.wav", print_special=False)
    (RESULTS / "tiron_jfk.log").write_text(out_jfk)
    jfk_ok = "country" in out_jfk.lower()

    tiron_summary.update({
        "convert_ok": True,
        "files": {"f16": str(f16_named), **{k: str(v) for k, v in quants.items()}},
        "validate": {
            "multispeaker_rc": rc_on,
            "speaker_markers_ON": n_speaker_markers,
            "speaker_markers_OFF": n_speaker_off,
            "jfk_rc": rc_jfk,
            "jfk_recognizable": jfk_ok,
            "PASS": rc_on == 0 and n_speaker_markers > 0 and jfk_ok,
        },
    })
    step("tiron.validate.done", **tiron_summary["validate"])

    # ── model card + upload ──
    if token:
        readme = OUT / "README.md"
        readme.write_text(
            f"""---
license: apache-2.0
base_model: {TIRON_SRC}
tags:
- automatic-speech-recognition
- whisper
- speaker-diarization
- crispasr
- ggml
---

# {TIRON_NAME} — GGML for CrispASR

Converted from [`{TIRON_SRC}`](https://huggingface.co/{TIRON_SRC}) (Apache-2.0).

Tiron is a Whisper large-v3 checkpoint that jointly transcribes and attributes
speech to speakers in one decode pass, emitting inline `<|speakerN|>` turn markers
(up to 8 speakers / 30 s window) and 20 ms timestamps.

Requires a CrispASR build with the **tiron decode mode** (issue #295): the extended
vocab (51904) is serialized into the GGML tokenizer table and the decoder passes the
speaker tokens through inline instead of treating them as timestamps.

| file | quant |
|------|-------|
| `{TIRON_NAME}-f16.bin`  | F16 (reference) |
| `{TIRON_NAME}-q8_0.bin` | q8_0 |
| `{TIRON_NAME}-q4_k.bin` | q4_k |

Speaker indices are local to each window; cross-window meeting-level linking is a
harness feature (see the upstream Trelis/tiron harness).
""",
            encoding="utf-8",
        )
        api = HfApi(token=token)
        api.create_repo(repo_id=TIRON_HF_OUT, repo_type="model", private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=str(readme), path_in_repo="README.md",
                        repo_id=TIRON_HF_OUT, repo_type="model",
                        commit_message="model card")
        for path in [f16_named, *quants.values()]:
            step("tiron.upload.begin", file=path.name, gb=round(path.stat().st_size / 1e9, 3))
            api.upload_file(path_or_fileobj=str(path), path_in_repo=path.name,
                            repo_id=TIRON_HF_OUT, repo_type="model",
                            commit_message=f"add {path.name}")
            step("tiron.upload.done", file=path.name)
        tiron_summary["uploaded"] = TIRON_HF_OUT
    else:
        step("tiron.upload.skip", reason="no HF token")
        tiron_summary["uploaded"] = None
except Exception as ex:  # noqa: BLE001
    import traceback
    tiron_summary["error"] = f"{type(ex).__name__}: {ex}"
    traceback.print_exc()
    step("tiron.ERROR", error=tiron_summary["error"])

(RESULTS / "tiron_summary.json").write_text(json.dumps(tiron_summary, indent=2))


# ═══════════════════════════ PHASE B — F5 A/B ════════════════════════════
_ODE_RE = re.compile(r"f5_bench:\s+ode_solve\s+([\d.]+)\s+ms")
_SPLIT_RE = re.compile(r"host_embed=([\d.]+)\s+ms\s+dit_graph=([\d.]+)\s+ms")

f5_summary = {"phase": "f5", "gpu": gpu_name, "cuda_arch": arch,
              "gen_text": F5_GEN_TEXT, "seed": SEED, "configs": {}}
F5_CONFIGS = []
try:
    from huggingface_hub import hf_hub_download  # noqa: E402

    step("f5.download.begin", free_gb=kh.free_gb(str(MODELS)))
    f5_model = Path(hf_hub_download(repo_id=F5_HF_REPO, filename=F5_HF_FILE,
                                    local_dir=str(MODELS), token=token or None))
    step("f5.model", path=str(f5_model), mb=round(f5_model.stat().st_size / 1e6, 1))

    F5_REF = SAMPLES / "jfk.wav"

    def f5_synth(tag, env, extra_args, out_wav, timeout=1200):
        cmd = [str(CLI), "--backend", "f5-tts", "-m", str(f5_model),
               "--voice", str(F5_REF), "--ref-text", F5_REF_TEXT,
               "--i-have-rights", "--no-spoken-disclaimer",
               "--tts", F5_GEN_TEXT, "--tts-output", str(out_wav),
               "--seed", str(SEED)] + list(extra_args)
        t0 = time.time()
        r = subprocess.run(cmd, env={**os.environ, "CRISPASR_F5_BENCH": "1", **env},
                           timeout=timeout, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        dt = round(time.time() - t0, 2)
        out = r.stdout or ""
        (RESULTS / f"f5_{tag}.log").write_text(out)
        ode = _ODE_RE.search(out)
        sp = _SPLIT_RE.search(out)
        return {
            "rc": r.returncode, "wall_s": dt,
            "ode_ms": float(ode.group(1)) if ode else None,
            "host_embed_ms": float(sp.group(1)) if sp else None,
            "dit_graph_ms": float(sp.group(2)) if sp else None,
            "dur_s": wav_dur(out_wav),
        }

    F5_CONFIGS = [
        ("A_base", {}, []),
        ("B_steps7", {}, ["--tts-steps", "7"]),
        ("C_s7_int2", {"CRISPASR_F5_CFG_INTERVAL": "2"}, ["--tts-steps", "7"]),
        ("D_embedgpu", {"CRISPASR_F5_EMBED_GPU": "1"}, []),
        ("E_embedgpu_s7", {"CRISPASR_F5_EMBED_GPU": "1"}, ["--tts-steps", "7"]),
        ("F_batchcfg", {"CRISPASR_F5_BATCH_CFG": "1"}, []),
        ("G_f16act", {"CRISPASR_F5_F16_ACT": "1"}, []),
        ("H_ditskip2", {"CRISPASR_F5_DIT_SKIP": "2"}, []),
        ("I_stack", {"CRISPASR_F5_EMBED_GPU": "1", "CRISPASR_F5_BATCH_CFG": "1",
                     "CRISPASR_F5_CFG_INTERVAL": "2"}, ["--tts-steps", "7"]),
    ]

    for tag, env, args in F5_CONFIGS:
        step("f5.config_start", tag=tag, env=env, args=args)
        for _ in range(N_WARMUP):
            f5_synth(tag, env, args, AUDIO / f"f5_{tag}_warm.wav")
        runs = [f5_synth(tag, env, args, AUDIO / f"f5_{tag}_{i}.wav") for i in range(N_MEASURED)]
        wav0 = AUDIO / f"f5_{tag}_0.wav"
        rt = asr_roundtrip(wav0)
        # proof-of-work: a fast crash mints a fake win, so a config only counts
        # if the audio actually reproduces the sentence (kaggle_usage #24).
        rt_hits = sum(w in rt.lower() for w in F5_GEN_WORDS)
        roundtrip_ok = rt_hits >= 3
        ok = [r for r in runs if r["rc"] == 0 and r["ode_ms"] is not None]
        f5_summary["configs"][tag] = {
            "env": env, "args": args,
            "ode_ms_median": median([r["ode_ms"] for r in ok]),
            "ode_ms_runs": [r["ode_ms"] for r in ok],
            "host_embed_ms": ok[0]["host_embed_ms"] if ok else None,
            "dit_graph_ms": ok[0]["dit_graph_ms"] if ok else None,
            "wall_s_median": median([r["wall_s"] for r in ok]),
            "dur_s": ok[0]["dur_s"] if ok else None,
            "n_ok": len(ok),
            "md5": md5(wav0),
            "roundtrip": rt,
            "roundtrip_ok": roundtrip_ok,
        }
        step("f5.config_done", tag=tag, **f5_summary["configs"][tag])
        (RESULTS / "f5_summary.json").write_text(json.dumps(f5_summary, indent=2))
except Exception as ex:  # noqa: BLE001
    import traceback
    f5_summary["error"] = f"{type(ex).__name__}: {ex}"
    traceback.print_exc()
    step("f5.ERROR", error=f5_summary["error"])

(RESULTS / "f5_summary.json").write_text(json.dumps(f5_summary, indent=2))


# ─────────────────────────── cell 4 (code) — tables ──────────────────────
print("\n" + "=" * 80, flush=True)
print("PHASE A — TIRON", flush=True)
print("=" * 80, flush=True)
print(json.dumps(tiron_summary, indent=2), flush=True)

print("\n" + "=" * 80, flush=True)
print(f"PHASE B — F5 perf on {gpu_name} (arch {arch}) — median of {N_MEASURED} runs", flush=True)
print(f"gen: {F5_GEN_TEXT!r}  seed {SEED}", flush=True)
print("=" * 80, flush=True)
base = f5_summary["configs"].get("A_base", {}).get("ode_ms_median")
hdr = f"{'config':16} {'ode ms':>10} {'speedup':>8} {'dur s':>6} {'rt_ok':>6}  roundtrip"
print(hdr, flush=True)
print("-" * 80, flush=True)
for tag, _, _ in F5_CONFIGS:
    c = f5_summary["configs"].get(tag)
    if not c:
        continue
    spd = f"{base / c['ode_ms_median']:.2f}x" if base and c["ode_ms_median"] else "-"
    print(f"{tag:16} {str(c['ode_ms_median']):>10} {spd:>8} {str(c['dur_s']):>6} "
          f"{str(c['roundtrip_ok']):>6}  {(c['roundtrip'] or '')[:40]}", flush=True)
print("=" * 80, flush=True)

# Refresh the ccache seed (single tar in /kaggle/working so `kaggle kernels
# output` gets it on page 1 — kaggle_usage #22). Upload to BOTH account copies.
try:
    kh.export_ccache_tar()
except Exception as ex:  # noqa: BLE001
    print(f"ccache export skipped: {ex}", flush=True)

step("DONE")
