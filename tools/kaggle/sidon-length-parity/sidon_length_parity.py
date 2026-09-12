#!/usr/bin/env python3
"""#431: does SIDON'S QUALITY FALL OFF WITH LENGTH IN UPSTREAM TOO?

The question this exists to answer, and why it could not be answered on the VPS:

CrispASR refuses sidon input past a 3000-frame attention cap. Windowing the
predictor makes long clips work, and an ASR roundtrip said windowed output was
BETTER than the whole-utterance path at 50 s — but "better to moonshine-tiny" is
not "faithful to upstream", and those can point in opposite directions. If the
upstream model also degrades at 50 s, then our whole-utterance path is CORRECT
and windowing is a deviation that merely sounds nicer.

So compare against the reference, per stage, at several lengths:

  for L in 11, 30, 50, 62 s:
      REF   = upstream TorchScript (feature_extractor_cpu.pt) predictor feats
      OURS_W = our C++, whole-utterance   (CRISPASR_SIDON_WINDOW_FRAMES=0)
      OURS_N = our C++, windowed          (default window)
      cos(OURS_W, REF)   vs   cos(OURS_N, REF)

READ THE VERDICT LIKE THIS — both directions are meaningful:
  * cos(OURS_W, REF) stays high at every L  -> upstream does NOT degrade; the
    falloff is ours, and windowing is a fix only if cos(OURS_N,REF) is higher.
  * cos(OURS_W, REF) FALLS with L           -> our whole-utterance path diverges
    from upstream as L grows; check whether windowing recovers it.
  * cos(OURS_N, REF) < cos(OURS_W, REF)     -> WINDOWING IS A DEVIATION. It would
    make us differ from the reference no matter how it sounds. Do not ship it.

The 11 s row is the control: a reference already exists for it and our handoff
matches at cos 0.994, so if this kernel does not reproduce ~0.994 at 11 s its
own plumbing is wrong and every other row is uninterpretable.
"""
import json, os, subprocess, sys, time, wave
from pathlib import Path

WORK = Path("/kaggle/working")
SCRATCH = Path("/tmp")               # 70 GB here vs ~20 GB in working
CLONE = SCRATCH / "CrispASR"
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
SCRIPT_VERSION = "2026-09-12-sidon-length-parity-3-jobs"
RESULTS = WORK / "results.json"
LENGTHS = [11, 30, 50, 62]

def log(m):
    print(m, flush=True)
    try:
        with open(WORK / "progress.txt", "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {m}\n")
    except Exception:
        pass

if not CLONE.exists():
    subprocess.check_call(["git", "clone", "--depth", "1", "--recurse-submodules",
                           "--shallow-submodules", CRISPASR_URL, str(CLONE)])
sys.path.insert(0, str(CLONE / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402
kh.init_progress()

sha = subprocess.run(["git", "-C", str(CLONE), "rev-parse", "--short", "HEAD"],
                     capture_output=True, text=True).stdout.strip()
# Gotcha #24: the C++ is fresh from this clone while THIS SCRIPT is frozen at the
# last `kaggle kernels push`. Print both so a verdict is attributable.
log(f"[parity] script_version={SCRIPT_VERSION} clone={sha}")

HF_TOKEN = kh.resolve_hf_token()
os.environ.setdefault("HF_TOKEN", HF_TOKEN or "")

# ── build (CPU is enough; sidon's CUDA path is not what is under test) ────────
kh.install_build_toolchain()
BUILD = SCRATCH / "build"
cfg = ["cmake", "-S", str(CLONE), "-B", str(BUILD), "-G", "Ninja",
       "-DCMAKE_BUILD_TYPE=Release"] + kh.cache_and_link_flags()
r = subprocess.run(cfg, capture_output=True, text=True)
if r.returncode != 0:
    log(f"build.configure FAILED rc={r.returncode}")
    log("---- configure stdout ----"); log((r.stdout or "<empty>")[-3000:])
    log("---- configure stderr ----"); log((r.stderr or "<empty>")[-3000:])
    raise SystemExit(1)
log("build.configure ok; tail: " + (r.stdout or "")[-600:])
# safe_build_jobs returns a SHELL SNIPPET ("$(nproc)"), not a number — every
# working kernel interpolates it into a shell string. v2 passed it to a
# list-form subprocess.run, where nothing expands it, and the build died with
#   '-j' invalid number '$(nproc)' given.
# Run it through a shell, the way the callers that demonstrably work do.
jobs = kh.safe_build_jobs(gpu=False)
with kh.build_heartbeat("build.crispasr"):
    r = subprocess.run(f"cmake --build {BUILD} --target crispasr -j{jobs}",
                       shell=True, capture_output=True, text=True)
if r.returncode != 0:
    # BOTH streams. v1 logged stdout only and ninja writes its errors to stderr,
    # so the failure line said "build FAILED" followed by nothing — a diagnostic
    # that cannot show the failure it exists to report.
    log(f"build FAILED rc={r.returncode}")
    log("---- build stdout (tail) ----"); log((r.stdout or "<empty>")[-4000:])
    log("---- build stderr (tail) ----"); log((r.stderr or "<empty>")[-4000:])
    raise SystemExit(1)
CRISPASR = BUILD / "bin" / "crispasr"
# HARD RULE #8: judge by proof-of-work, not an exit code.
if not CRISPASR.is_file():
    log("build reported success but produced no binary"); raise SystemExit(1)
log(f"[parity] built {CRISPASR} ({CRISPASR.stat().st_size} bytes)")

# ── models: our GGUF + the upstream TorchScript modules ──────────────────────
from huggingface_hub import hf_hub_download, list_repo_files
GGUF = hf_hub_download("cstr/Sidon-GGUF", "sidon-v0.1-f16.gguf", local_dir=str(SCRATCH/"m"))
files = list_repo_files("sarulab-speech/sidon_raw_weight")
log(f"[parity] upstream repo files: {files[:20]}")
def pick(sub):
    c = [f for f in files if sub in f.lower() and f.endswith(".pt")]
    return c[0] if c else None
fe_f, dec_f = pick("feature_extractor"), pick("decoder")
if not fe_f or not dec_f:
    log(f"[parity] could NOT find TorchScript modules (fe={fe_f} dec={dec_f}) — cannot build a reference")
    RESULTS.write_text(json.dumps({"conclusive": False, "reason": "torchscript modules not found",
                                   "files": files}, indent=2))
    raise SystemExit(0)
FE = hf_hub_download("sarulab-speech/sidon_raw_weight", fe_f, local_dir=str(SCRATCH/"m"))
DEC = hf_hub_download("sarulab-speech/sidon_raw_weight", dec_f, local_dir=str(SCRATCH/"m"))
log(f"[parity] fe={FE}\n[parity] dec={DEC}")

import numpy as np, torch
from transformers import SeamlessM4TFeatureExtractor
fx = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
fe = torch.jit.load(FE, map_location="cpu").eval()

def read_wav(p):
    w = wave.open(str(p), "rb")
    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return a, w.getframerate()

def make_clip(seconds, out):
    subprocess.check_call(["ffmpeg", "-loglevel", "error", "-y", "-stream_loop", "9",
                           "-i", str(CLONE / "samples" / "jfk.wav"), "-t", str(seconds),
                           "-ar", "16000", "-ac", "1", str(out)])

def ref_feats(wav_path):
    pcm, sr = read_wav(wav_path)
    with torch.no_grad():
        f = fx(pcm, sampling_rate=sr, return_tensors="pt")
        out = fe(f["input_features"], f["attention_mask"]) if "attention_mask" in f else fe(f["input_features"])
    t = out[0] if isinstance(out, (tuple, list)) else out
    return t.squeeze(0).float().numpy()          # [T, 1024]

def ours_feats(wav_path, windowed):
    dump = SCRATCH / ("h_%s.f32" % ("win" if windowed else "whole"))
    env = dict(os.environ, CRISPASR_SIDON_DUMP_HANDOFF=str(dump))
    if windowed:
        env["CRISPASR_SIDON_WINDOW_FRAMES"] = "1500"
    else:
        env["CRISPASR_SIDON_WINDOW_FRAMES"] = "0"      # whole-utterance
        env["CRISPASR_SIDON_MAX_FRAMES"] = "100000"    # and do not refuse
    out = SCRATCH / ("o_%s.wav" % ("win" if windowed else "whole"))
    r = subprocess.run([str(CRISPASR), "-m", GGUF, "-f", str(wav_path),
                        "--s2s", "--s2s-output", str(out)],
                       capture_output=True, text=True, env=env, timeout=5400)
    if not dump.is_file():
        return None, (r.stderr or "")[-600:]
    # The windowed arm must ACTUALLY window. If the binary predates the env var
    # (it clones main at runtime — gotcha #24) the flag is ignored, both arms run
    # the same path, and the comparison silently becomes A-vs-A. Demand the proof
    # line the windowed path prints.
    if windowed and "windows with" not in (r.stderr or ""):
        return None, "WINDOWED ARM DID NOT WINDOW — env var ignored by this binary: " + (r.stderr or "")[-400:]
    a = np.fromfile(dump, dtype=np.float32)
    return a.reshape(-1, 1024), None

def cos_aligned(ours, ref):
    """Best frame offset then cosine. Ours carries lead+lookahead padding that
    the reference does not, so an unaligned compare would report a divergence
    that is really an offset."""
    if ours is None or ref is None: return None, None, None
    T = min(ref.shape[0], ours.shape[0])
    best = (None, -9.0)
    for off in range(0, min(200, max(1, ours.shape[0] - T + 1))):
        a = ours[off:off + T].ravel(); b = ref[:T].ravel()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        if d == 0: continue
        c = float(a @ b / d)
        if c > best[1]: best = (off, c)
    off, c = best
    a = ours[off:off + T]
    return c, off, float(np.sqrt((a**2).mean()) / max(1e-9, np.sqrt((ref[:T]**2).mean())))

results = {"script_version": SCRIPT_VERSION, "clone": sha, "rows": []}
for L in LENGTHS:
    clip = SCRATCH / f"clip{L}.wav"
    make_clip(L, clip)
    log(f"[parity] === {L}s ===")
    try:
        ref = ref_feats(clip)
    except Exception as e:
        log(f"[parity] {L}s REFERENCE FAILED: {type(e).__name__}: {str(e)[:200]}")
        results["rows"].append({"seconds": L, "reference_error": f"{type(e).__name__}: {str(e)[:200]}"})
        RESULTS.write_text(json.dumps(results, indent=2)); continue
    row = {"seconds": L, "ref_frames": int(ref.shape[0])}
    for tag, windowed in (("whole", False), ("windowed", True)):
        ours, err = ours_feats(clip, windowed)
        if ours is None:
            row[tag] = {"error": err}
        else:
            c, off, mag = cos_aligned(ours, ref)
            row[tag] = {"frames": int(ours.shape[0]), "cos_vs_ref": c, "offset": off, "mag_ratio": mag}
            log(f"[parity] {L}s {tag:8} frames={ours.shape[0]} cos_vs_ref={c:.6f} off={off} mag={mag:.4f}")
    results["rows"].append(row)
    RESULTS.write_text(json.dumps(results, indent=2))

log("[parity] DONE")
log(json.dumps(results, indent=2))
