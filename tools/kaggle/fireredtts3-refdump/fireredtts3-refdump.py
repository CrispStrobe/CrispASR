#!/usr/bin/env python3
"""Kaggle kernel: FireRedTTS3 (#377) — reference dump + e2e Python control arm.

Runs the REAL upstream pipeline on CPU (fp32, ~13 GB — CPU kernel has ~30 GB)
via tools/dump_reference.py --backend fireredtts3, then:
  - saves the reference-generated audio as WAV
  - CONTROL ARM: ASR-transcribes that audio with openai-whisper (base) and
    prints the transcript — proves the reference pipeline itself produces
    intelligible speech BEFORE any C++ is judged against it
  - uploads ref.gguf to cstr/crispasr-regression-fixtures
    (fireredtts3/jfk_11s/ref.gguf) and the control WAV + transcript to
    cstr/fireredtts3-GGUF

Push (chr1s4):
  export KAGGLE_API_TOKEN=<chr1s4 token>
  python -m kaggle kernels push -p tools/kaggle/fireredtts3-refdump
"""

import os
import subprocess
import sys
from pathlib import Path

SCRIPT_VERSION = "v1"
WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")
BRANCH = "feat/377-fireredtts3"
SRC_REPO = "FireRedTeam/FireRedTTS3"
HF_REPO = "cstr/fireredtts3-GGUF"
FIXTURES_REPO = "cstr/crispasr-regression-fixtures"

SYN_TEXT = "Hello there, how are you today?"
JFK_TEXT = ("And so my fellow Americans ask not what your country can do "
            "for you, ask what you can do for your country.")

print(f"=== fireredtts3-refdump {SCRIPT_VERSION} ===", flush=True)

if not REPO.exists():
    subprocess.check_call(["git", "clone", "--depth", "1", "-b", BRANCH,
                           "https://github.com/CrispStrobe/CrispASR", str(REPO)])
subprocess.check_call(["git", "log", "--oneline", "-1"], cwd=str(REPO))
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()

kh.step("install deps")
kh.sh_with_progress("pip install -q gguf safetensors huggingface_hub hf_transfer openai-whisper")

kh.step("resolve HF token")
hf_token = kh.resolve_hf_token()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

kh.step("download FireRedTTS3")
from huggingface_hub import snapshot_download, HfApi  # noqa: E402

src = snapshot_download(
    repo_id=SRC_REPO, cache_dir=str(TEMP / "frt-src"), token=hf_token,
    allow_patterns=["fireredtts3_base/*", "redae/*", "campp/*", "text_tokenizer/*"],
)
print(f"  src: {src}")

kh.step("clone upstream")
UP = TEMP / "FireRedTTS3-upstream"
if not UP.exists():
    subprocess.check_call(["git", "clone", "--depth", "1",
                           "https://github.com/FireRedTeam/FireRedTTS3.git", str(UP)])

kh.step("reference dump (full upstream pipeline, CPU)")
env = dict(os.environ)
env.update({
    "FIREREDTTS3_UPSTREAM": str(UP),
    "FIREREDTTS3_SYN_TEXT": SYN_TEXT,
    "FIREREDTTS3_PROMPT_TEXT": JFK_TEXT,
    "FIREREDTTS3_LANG": "English",
    "FIREREDTTS3_SEED": "1234",
    "OMP_NUM_THREADS": "4",
})
ref_gguf = WORK / "fireredtts3-jfk-ref.gguf"
with kh.build_heartbeat("refdump"):
    subprocess.check_call([
        sys.executable, str(REPO / "tools" / "dump_reference.py"),
        "--backend", "fireredtts3",
        "--model-dir", src,
        "--audio", str(REPO / "samples" / "jfk.wav"),
        "--output", str(ref_gguf),
    ], env=env, cwd=str(REPO / "tools"))
print(f"  ref.gguf: {ref_gguf.stat().st_size/2**20:.1f} MiB")

kh.step("extract control WAV from ref.gguf")
import numpy as np  # noqa: E402
from gguf import GGUFReader  # noqa: E402

r = GGUFReader(str(ref_gguf))
gen = None
for t in r.tensors:
    if t.name == "gen_audio":
        gen = np.asarray(t.data, dtype=np.float32).reshape(-1)
if gen is None:
    raise SystemExit("gen_audio missing from ref.gguf")
print(f"  gen_audio: {len(gen)} samples = {len(gen)/24000:.2f}s "
      f"rms={float(np.sqrt((gen**2).mean())):.4f}")
assert len(gen) > 24000 * 0.5, "generated audio suspiciously short"
assert float(np.sqrt((gen**2).mean())) > 1e-3, "generated audio is near-silence"

import wave  # noqa: E402
wav_path = WORK / "fireredtts3-ref-control.wav"
with wave.open(str(wav_path), "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(24000)
    w.writeframes((np.clip(gen, -1, 1) * 32767).astype(np.int16).tobytes())

kh.step("CONTROL ARM: whisper ASR of reference audio")
import whisper  # noqa: E402

wm = whisper.load_model("base")
res = wm.transcribe(str(wav_path), language="en")
transcript = res["text"].strip()
print(f"  CONTROL_TRANSCRIPT: {transcript!r}")
(WORK / "control_transcript.txt").write_text(
    f"target: {SYN_TEXT}\nwhisper-base: {transcript}\n")
# word-overlap check (present-in, not exact-match)
tgt = set(w.strip(".,?!'").lower() for w in SYN_TEXT.split())
got = set(w.strip(".,?!'").lower() for w in transcript.split())
overlap = len(tgt & got) / max(1, len(tgt))
print(f"  word overlap vs target: {overlap:.2f}")
if overlap < 0.6:
    print("  WARNING: control arm word overlap below 0.6 — reference pipeline "
          "output may not be intelligible; investigate before blaming C++")

kh.step("upload artifacts")
api = HfApi(token=hf_token)
api.upload_file(path_or_fileobj=str(ref_gguf),
                path_in_repo="fireredtts3/jfk_11s/ref.gguf",
                repo_id=FIXTURES_REPO, repo_type="dataset")
print("  ref.gguf uploaded to fixtures")
api.upload_file(path_or_fileobj=str(wav_path),
                path_in_repo="reference/fireredtts3-ref-control.wav",
                repo_id=HF_REPO)
api.upload_file(path_or_fileobj=str(WORK / "control_transcript.txt"),
                path_in_repo="reference/control_transcript.txt",
                repo_id=HF_REPO)
print("  control wav + transcript uploaded")

kh.step("DONE")
print("REFDUMP_OK", flush=True)
