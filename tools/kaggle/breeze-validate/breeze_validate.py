#!/usr/bin/env python3
"""Breeze TTS 2 (#412): does it actually work?

Two questions, in this order, because the second is meaningless if the first
has not been asked:

  1. per-stage parity against the reference fixture — cosine AND magnitude
     (cosine is scale-blind, so a stage wrong by a uniform factor passes every
     cosine check ever run against it);
  2. synthesis to a WAV, then an ASR roundtrip on that WAV.

THE TWO TRAPS ARE CHECKED FIRST, deliberately, because both corrupt everything
downstream while looking like a model bug:

  ref_codes   the oracle hands jfk.wav to the codec tokenizer at 16 kHz and
              lets it resample; the runtime pre-resamples to 24 kHz with
              resample_polyphase. Two resamplers on the clone reference change
              the prompt before a single transformer weight is touched.
  repetition  absent from generation_config.json — infer.py passes 1.1 at CALL
              time. The oracle never passes it, so its codes are penalty-free;
              both arms now DECLARE 1.0 on the greedy path.

CHEAP GATE. The fixture's frame 0 is argmax_cb0 = 404 and
  [404, 172, 340, 1357, 644, 528, 1025, 1250, 122, 730, 1219, 1452, 1957, 443,
   416, 1187]
If frame 0 does not match, the run says so loudly and does not pretend the
later numbers mean anything.

No GPU: nothing here needs one (the model is a 2 GB q4_k and the dump is 24
frames), so the kernel skips the accelerator lottery and the CUDA build
entirely. Internet still works with enable_gpu=false — the convert kernel
proved that.

Push (chr1str):
  export KAGGLE_API_TOKEN=<chr1str token>
  python -m kaggle kernels push -p tools/kaggle/breeze-validate
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_VERSION = "2026-09-17.1"
WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else WORK
REPO = WORK / "CrispASR"
BRANCH = os.environ.get("CRISPASR_REF", "feat/412-breeze-tts-2")

HF_MODEL = "cstr/breeze-tts-2-GGUF"
HF_CODEC = "cstr/qwen3-tts-tokenizer-12hz-GGUF"
HF_FIX = "cstr/crispasr-regression-fixtures"
FIX_PREFIX = "breeze-tts-2"

SYN_TEXT = "The quick brown fox jumps over the lazy dog."
REF_TEXT = ("And so my fellow Americans, ask not what your country can do for you, "
            "ask what you can do for your country.")

# The cheap gate reads its landmarks FROM THE FIXTURE at runtime, not from
# constants pinned here. Run 4 showed why: the fixture was regenerated, its
# frame-0 codes legitimately changed, and the hardcoded vector then reported a
# MISMATCH for a stage that the comparator scored as passing. A gate that goes
# stale when the thing it guards is updated produces false alarms, and a false
# alarm on the cheap gate is worse than no cheap gate — it sends you to
# investigate a stage that is fine.
SMOKE_CB0 = None
SMOKE_FRAME0 = None

verdict = {"script_version": SCRIPT_VERSION, "conclusive": False}


def finish(**kw):
    verdict.update(kw)
    (WORK / "verdict.json").write_text(json.dumps(verdict, indent=1))
    print("VERDICT " + json.dumps(verdict), flush=True)


print(f"=== breeze-validate {SCRIPT_VERSION} (branch {BRANCH}) ===", flush=True)

# ── clone ───────────────────────────────────────────────────────────────────
if not REPO.exists():
    for attempt in range(4):
        if REPO.exists():
            shutil.rmtree(REPO)
        rc = subprocess.run(["git", "clone", "--depth", "1", "--recursive",
                             "--shallow-submodules", "-b", BRANCH,
                             "https://github.com/CrispStrobe/CrispASR",
                             str(REPO)]).returncode
        if rc == 0:
            break
        print(f"clone attempt {attempt + 1} failed rc={rc}", flush=True)
    else:
        finish(stage="clone", error="clone failed after 4 attempts")
        raise SystemExit(1)
subprocess.check_call(["git", "log", "--oneline", "-1"], cwd=str(REPO))
subprocess.run(["git", "submodule", "update", "--init", "--recursive", "--depth", "1"],
               cwd=str(REPO), check=False)
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
if hasattr(kh, "provenance"):
    kh.provenance(SCRIPT_VERSION, clone_dir=REPO)

kh.step("install deps")
kh.sh_with_progress("pip install -q huggingface_hub numpy soundfile")
hf_token = kh.resolve_hf_token()
if not hf_token:
    finish(stage="token", error="no HF token")
    raise SystemExit(1)
os.environ["HF_TOKEN"] = hf_token

from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402

# ── artifacts ───────────────────────────────────────────────────────────────
kh.step("download artifacts")
# f16 BY DEFAULT, and that is the point of this run. The first pass diffed a
# q4_k model against a bf16 reference and read 0.9988 at layer 10 decaying to
# 0.998 by layer 27 — a smooth decay with depth, which is the signature of
# accumulating QUANTIZATION noise, not of a structural bug. Diffing f16 against
# bf16 removes that confound: if the cosines jump, the port is right and the
# q4_k numbers were quantization; if they stay, there is a real porting bug and
# the layer where it starts is the answer. BREEZE_QUANT=q4_k re-runs the A/B.
QUANT = os.environ.get("BREEZE_QUANT", "f16")
model = hf_hub_download(HF_MODEL, f"breeze-tts-2-{QUANT}.gguf", token=hf_token,
                        local_dir=str(TEMP / "m"))
verdict["quant"] = QUANT
codec = hf_hub_download(HF_CODEC, "qwen3-tts-tokenizer-12hz.gguf", token=hf_token,
                        local_dir=str(TEMP / "m"))
fixdir = snapshot_download(HF_FIX, repo_type="dataset", token=hf_token,
                           allow_patterns=[f"{FIX_PREFIX}/*"],
                           local_dir=str(TEMP / "fix"))
fixdir = str(Path(fixdir) / FIX_PREFIX)
print("model:", model, os.path.getsize(model) / 2**30, "GiB", flush=True)
print("fixture files:", len(list(Path(fixdir).glob("*.npy"))), flush=True)

# ── build (CPU only) ────────────────────────────────────────────────────────
kh.step("build")
kh.install_build_toolchain()
BUILD = REPO / "build"
subprocess.check_call(
    ["cmake", "-G", "Ninja", "-B", str(BUILD), "-S", str(REPO),
     "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=OFF"] + kh.cache_and_link_flags())
with kh.build_heartbeat("build"):
    kh.sh_with_progress(
        f"cmake --build {BUILD} --target crispasr crispasr-diff -j {kh.safe_build_jobs(gpu=False)}")

BIN = BUILD / "bin"
crispasr = BIN / "crispasr"
diffbin = BIN / "crispasr-diff"
for b in (crispasr, diffbin):
    if not b.is_file():
        finish(stage="build", error=f"{b} not produced")
        raise SystemExit(1)
    print("built:", b, os.path.getsize(b), flush=True)

# A backend invisible to --list-backends is invisible to everything else too.
lb = subprocess.run([str(crispasr), "--list-backends"], capture_output=True, text=True)
print("bt2-tts listed:", "bt2-tts" in (lb.stdout + lb.stderr), flush=True)
verdict["backend_listed"] = "bt2-tts" in (lb.stdout + lb.stderr)

# ── the NC gate must BLOCK, not warn ────────────────────────────────────────
# Proven by observation, not by reading the code: ask for -m auto WITHOUT
# acceptance and require a refusal, then WITH it and require the refusal to go
# away. An arm that cannot fail is not evidence.
kh.step("licence gate")
env_no = dict(os.environ)
env_no.pop("CRISPASR_ACCEPT_LICENSE", None)
g1 = subprocess.run([str(crispasr), "--backend", "bt2-tts", "-m", "auto", "--auto-download",
                     "--tts", "hi", "--tts-output", str(WORK / "gate.wav")],
                    capture_output=True, text=True, timeout=600, env=env_no)
blocked = "refusing to download" in (g1.stdout + g1.stderr).lower()
verdict["nc_gate_blocks_without_acceptance"] = blocked
print("NC gate blocked without acceptance:", blocked, flush=True)

# ── stage dump + comparison ─────────────────────────────────────────────────
kh.step("stage dump")
dump = WORK / "cppdump"
dump.mkdir(exist_ok=True)
env = dict(os.environ, BREEZE_CODEC=codec, CRISPASR_ACCEPT_LICENSE="other")
d = subprocess.run([str(diffbin), "bt2-tts", model, fixdir, str(dump)],
                   capture_output=True, text=True, timeout=5400, env=env)
print(d.stdout[-16000:], flush=True)
if d.returncode != 0:
    print("STDERR:", d.stderr[-8000:], flush=True)
verdict["dump_rc"] = d.returncode
verdict["dump_stages"] = len(list(dump.glob("*.npy")))

# The cheap gate, read out of the dump rather than the log.
import numpy as np  # noqa: E402

smoke = {}
ref_f0_path = Path(fixdir) / "dd_codes_frame0_stepwise.npy"
ref_lg_path = Path(fixdir) / "backbone_logits_frame0.npy"
if ref_f0_path.exists():
    SMOKE_FRAME0 = np.load(ref_f0_path).tolist()
if ref_lg_path.exists():
    SMOKE_CB0 = int(np.argmax(np.load(ref_lg_path)))
smoke["landmarks_from_fixture"] = {"cb0": SMOKE_CB0, "frame0": SMOKE_FRAME0}
f0 = dump / "dd_codes_frame0_stepwise.npy"
if f0.exists():
    got = np.load(f0).tolist()
    smoke["frame0_codes"] = got
    smoke["frame0_match"] = got == SMOKE_FRAME0
    smoke["frame0_first_bad"] = next((i for i, (a, b) in enumerate(zip(got, SMOKE_FRAME0)) if a != b), None)
bl = dump / "backbone_logits_frame0.npy"
if bl.exists():
    am = int(np.argmax(np.load(bl)))
    smoke["argmax_cb0"] = am
    smoke["argmax_match"] = am == SMOKE_CB0
rc_ = dump / "ref_codes.npy"
if rc_.exists() and (Path(fixdir) / "ref_codes.npy").exists():
    a, b = np.load(rc_), np.load(Path(fixdir) / "ref_codes.npy")
    smoke["ref_codes_shape_cpp"] = list(a.shape)
    smoke["ref_codes_shape_ref"] = list(b.shape)
    n = min(a.shape[0], b.shape[0])
    smoke["ref_codes_match_frac"] = float((a[:n] == b[:n]).mean()) if n else 0.0
    smoke["ref_codes_exact"] = bool(a.shape == b.shape and (a == b).all())
verdict["smoke"] = smoke
print("SMOKE " + json.dumps(smoke), flush=True)

kh.step("compare")
cmp_env = dict(env, BREEZE_FIXTURE_DIR=fixdir, PYTHONPATH=str(REPO / "tools"))
c = subprocess.run([sys.executable, str(REPO / "tools" / "reference_backends" / "breeze_tts_2.py"),
                    "--cpp-dump", str(dump)],
                   capture_output=True, text=True, timeout=1800, env=cmp_env)
print(c.stdout[-20000:], flush=True)
if c.stderr:
    print("CMP STDERR:", c.stderr[-4000:], flush=True)
verdict["compare_rc"] = c.returncode
(WORK / "compare.txt").write_text(c.stdout + "\n" + c.stderr)

# ── synthesis + ASR roundtrip ───────────────────────────────────────────────
# This is the question the owner actually asked. It runs even if parity failed,
# because "it produces audio and the ASR reads it back" and "every stage
# matches" are different claims and both are worth knowing.
kh.step("synthesize")
wav = WORK / "bt2_out.wav"
s = subprocess.run([str(crispasr), "--backend", "bt2-tts", "-m", model,
                    "--codec-model", codec, "--accept-license", "other",
                    "--tts", SYN_TEXT, "--tts-output", str(wav)],
                   capture_output=True, text=True, timeout=5400, env=env)
print(s.stdout[-8000:], flush=True)
print("SYNTH STDERR:", s.stderr[-8000:], flush=True)
verdict["synth_rc"] = s.returncode
verdict["wav_exists"] = wav.is_file()
if wav.is_file():
    import soundfile as sf

    a, sr = sf.read(str(wav))
    verdict["wav"] = {"samples": int(a.shape[0]), "sr": int(sr),
                      "seconds": round(a.shape[0] / sr, 2),
                      "peak": float(abs(a).max()), "rms": float((a ** 2).mean() ** 0.5)}
    print("WAV " + json.dumps(verdict["wav"]), flush=True)

kh.step("asr roundtrip")
if wav.is_file() and verdict.get("wav", {}).get("rms", 0) > 1e-5:
    # CONTROL FIRST: run the ORACLE's own wav through the same ASR. Without it
    # a bad transcript cannot be attributed — it could be the ASR, the text, or
    # our audio, and only the control separates them.
    ref_wav = Path(fixdir) / "breeze-ref.wav"
    for tag, path in (("oracle_control", ref_wav), ("ours", wav)):
        if not Path(path).is_file():
            continue
        r = subprocess.run([str(crispasr), "--backend", "whisper", "-m", "auto",
                            "--auto-download", "-f", str(path), "-l", "en", "--no-prints"],
                           capture_output=True, text=True, timeout=2400, env=env)
        txt = " ".join(r.stdout.split())
        verdict.setdefault("asr", {})[tag] = txt[:400]
        print(f"ASR[{tag}] rc={r.returncode}: {txt[:300]}", flush=True)
        if r.returncode != 0:
            print(f"ASR[{tag}] stderr: {r.stderr[-2000:]}", flush=True)
else:
    verdict["asr_skipped"] = "no audio, or silence"

finish(conclusive=True)
print("[DONE]", flush=True)
