#!/usr/bin/env python3
"""Kaggle kernel: FireRedTTS3 (#377) — build (CPU), per-stage diff, quantize,
TTS-to-ASR roundtrip.

CPU-only kernel (no GPU quota):
  1. clone branch feat/377-fireredtts3 (+ggml submodule), build crispasr-cli,
     crispasr-diff, crispasr-quantize
  2. download f16 GGUFs from cstr/fireredtts3-GGUF + the jfk ref.gguf from
     cstr/crispasr-regression-fixtures
  3. crispasr-diff fireredtts3 (per-stage cos + |mine| + |ref|, noise replay)
  4. quantize base f16 → q4_k, upload
  5. roundtrip on F16 and Q4_K: --tts (baked default jfk prompt) → wav →
     whisper-base ASR (the SAME instrument the Python control arm used —
     control transcript was 'All there, how are you today?' vs target
     'Hello there, how are you today?', overlap 0.83)
  6. verdict: non-silent audio + word overlap vs target, judged against the
     control arm's own 0.83

Push (chr1s4):
  export KAGGLE_API_TOKEN=<chr1s4 token>
  python -m kaggle kernels push -p tools/kaggle/fireredtts3-validate
"""

import os
import subprocess
import sys
from pathlib import Path

SCRIPT_VERSION = "v3-diffonly"
WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
BRANCH = "feat/377-fireredtts3"
HF_REPO = "cstr/fireredtts3-GGUF"
FIXTURES = "cstr/crispasr-regression-fixtures"
SYN_TEXT = "Hello there, how are you today?"
CONTROL_OVERLAP = 0.83  # python reference arm, whisper-base, same text

print(f"=== fireredtts3-validate {SCRIPT_VERSION} ===", flush=True)

if not REPO.exists():
    subprocess.check_call(["git", "clone", "--depth", "1", "-b", BRANCH,
                           "https://github.com/CrispStrobe/CrispASR", str(REPO)])
subprocess.check_call(["git", "log", "--oneline", "-1"], cwd=str(REPO))
subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"], cwd=str(REPO))
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()

kh.step("install deps")
kh.sh_with_progress("pip install -q huggingface_hub hf_transfer openai-whisper")
tool = kh.install_build_toolchain()
print(f"  toolchain: {tool}")

hf_token = kh.resolve_hf_token()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

# ── build (CPU) ─────────────────────────────────────────────────────────────
kh.step("cmake configure")
flags = ["-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=OFF",
         "-DCRISPASR_BUILD_TESTS=OFF", "-DCRISPASR_BUILD_SERVER=OFF"]
flags += kh.cache_and_link_flags()
gen = ["-G", "Ninja"] if tool.get("ninja") else []
kh.sh_with_progress("cmake -B build " + " ".join(gen + flags), cwd=str(REPO))

kh.step("build crispasr-cli + crispasr-diff + crispasr-quantize")
with kh.build_heartbeat("build"):
    kh.sh_with_progress(
        "cmake --build build --target crispasr-cli crispasr-diff crispasr-quantize -j $(nproc)",
        cwd=str(REPO))
BIN = REPO / "build" / "bin"
for exe in ["crispasr", "crispasr-diff", "crispasr-quantize"]:
    p = BIN / exe
    assert p.exists(), f"missing {p} — build did not produce the binary"
    print(f"  built: {p} ({p.stat().st_size} bytes)")

# ── models ──────────────────────────────────────────────────────────────────
kh.step("download GGUFs + ref")
from huggingface_hub import hf_hub_download, HfApi  # noqa: E402

MD = TEMP / "frt-models"
MD.mkdir(parents=True, exist_ok=True)
base_f16 = hf_hub_download(HF_REPO, "fireredtts3-base-f16.gguf", local_dir=str(MD), token=hf_token)
redae_f16 = hf_hub_download(HF_REPO, "fireredtts3-redae-f16.gguf", local_dir=str(MD), token=hf_token)
ref_gguf = hf_hub_download(FIXTURES, "fireredtts3/jfk_11s/ref.gguf", local_dir=str(MD),
                           repo_type="dataset", token=hf_token)
print(f"  base={base_f16}\n  redae={redae_f16}\n  ref={ref_gguf}")

env = dict(os.environ)
env["FIREREDTTS3_REDAE"] = redae_f16
env["OMP_NUM_THREADS"] = "4"

# ── per-stage diff ──────────────────────────────────────────────────────────
kh.step("crispasr-diff fireredtts3 (f16)")
diff_log = WORK / "diff_f16.log"
with kh.build_heartbeat("diff"):
    r = subprocess.run([str(BIN / "crispasr-diff"), "fireredtts3", base_f16, ref_gguf,
                        str(REPO / "samples" / "jfk.wav")],
                       env=env, capture_output=True, text=True, timeout=7200)
diff_log.write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
print(r.stderr[-4000:])
print(f"  diff rc={r.returncode}")
diff_pass = (r.returncode == 0)

DO_HEAVY = False  # v3: diff-only iteration — quant + roundtrips already done
                  # (f16 overlap 1.00, q4k 0.83 vs control 0.83, q4_k on HF)

# ── wiring audit + backends.json (for feature-matrix regen) ────────────────
kh.step("wiring audit")
ra = subprocess.run([sys.executable, str(REPO / "tools" / "check-backend-wiring.py"),
                     "--crispasr", str(BIN / "crispasr")], capture_output=True, text=True)
print(ra.stdout[-2500:])
print(ra.stderr[-500:])
rb = subprocess.run([str(BIN / "crispasr"), "--list-backends-json"], capture_output=True, text=True)
(WORK / "backends.json").write_text(rb.stdout)

# ── quantize ────────────────────────────────────────────────────────────────
kh.step("quantize q4_k + upload")
base_q4 = str(MD / "fireredtts3-base-q4_k.gguf")
if not DO_HEAVY:
    print("  skipped (diff-only mode)")
    (WORK / "verdict.txt").write_text(f"diff_pass={diff_pass} (diff-only run)\n")
    kh.step("verdict")
    print(f"VERDICT diff_pass={diff_pass} (diff-only)")
    print("FRT_VALIDATE_OK" if diff_pass else "FRT_VALIDATE_FAIL", flush=True)
    raise SystemExit(0)
rq = subprocess.run([str(BIN / "crispasr-quantize"), base_f16, base_q4, "q4_k"],
                    capture_output=True, text=True)
print(rq.stdout[-1500:])
print(rq.stderr[-1500:])
api = HfApi(token=hf_token)
if rq.returncode == 0 and Path(base_q4).exists():
    print(f"  q4_k: {Path(base_q4).stat().st_size/2**30:.2f} GiB")
    api.upload_file(path_or_fileobj=base_q4, path_in_repo="fireredtts3-base-q4_k.gguf",
                    repo_id=HF_REPO)
    print("  q4_k uploaded")
else:
    print("  QUANTIZE FAILED — check crispasr-quantize rules for arch fireredtts3")

# ── roundtrip ───────────────────────────────────────────────────────────────
kh.step("roundtrip")
import re
import numpy as np  # noqa: E402
import whisper  # noqa: E402

wm = whisper.load_model("base")


def roundtrip(model_path, tag):
    out_wav = str(WORK / f"frt-tts-{tag}.wav")
    cmd = [str(BIN / "crispasr"), "--backend", "fireredtts3", "-m", model_path,
           "--tts", SYN_TEXT, "--tts-output", out_wav, "-ng"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
    print(f"  [{tag}] rc={r.returncode}")
    print("  " + "\n  ".join((r.stderr or "").splitlines()[-12:]))
    if r.returncode != 0 or not Path(out_wav).exists():
        return False, "", 0.0
    import wave
    w = wave.open(out_wav)
    n, sr = w.getnframes(), w.getframerate()
    pcm = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    rms = float(np.sqrt((pcm ** 2).mean())) if n else 0.0
    dur = n / sr if sr else 0
    print(f"  [{tag}] {dur:.2f}s @ {sr} Hz rms={rms:.4f}")
    if n < sr // 2 or rms < 1e-3:
        print(f"  [{tag}] FAIL: silent or too short")
        return False, "", 0.0
    res = wm.transcribe(out_wav, language="en")
    tr = res["text"].strip()
    tgt = set(x.strip(".,?!'").lower() for x in SYN_TEXT.split())
    got = set(x.strip(".,?!'").lower() for x in tr.split())
    ov = len(tgt & got) / max(1, len(tgt))
    print(f"  [{tag}] TRANSCRIPT: {tr!r}  overlap={ov:.2f} (control arm: {CONTROL_OVERLAP})")
    return True, tr, ov


ok16, tr16, ov16 = roundtrip(base_f16, "f16")
okq4, trq4, ovq4 = (False, "", 0.0)
if Path(base_q4).exists():
    okq4, trq4, ovq4 = roundtrip(base_q4, "q4k")

(WORK / "verdict.txt").write_text(
    f"diff_pass={diff_pass}\nf16: ok={ok16} overlap={ov16:.2f} transcript={tr16!r}\n"
    f"q4k: ok={okq4} overlap={ovq4:.2f} transcript={trq4!r}\n")

kh.step("verdict")
print(f"VERDICT diff_pass={diff_pass} f16_overlap={ov16:.2f} q4k_overlap={ovq4:.2f}")
if diff_pass and ok16 and ov16 >= 0.6:
    print("FRT_VALIDATE_OK", flush=True)
else:
    print("FRT_VALIDATE_FAIL", flush=True)
