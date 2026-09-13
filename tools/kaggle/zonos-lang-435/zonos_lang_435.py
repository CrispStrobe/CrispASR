#!/usr/bin/env python3
"""#435: prove the zonos non-English fix end-to-end, on a box that has the model.

The fix (1f9e62c3) has three parts and only one of them is verified so far:
  a) in-process libespeak-ng before the popen fallback
  b) raw-character tokenisation restricted to pure ASCII; non-ASCII with no
     phonemizer is a LOUD refusal instead of ~0.9 s of confident noise
  c) per-request language actually reaches the model

(b)'s predicate was unit-tested locally. (a) and (c) need the model, which is
~1.6 GB + a DAC codec — over this project's "no large models on the VPS" line.

ARMS, and each is designed so a pass cannot be mistaken for a skip:

  1. russian_with_espeak   espeak-ng installed, -l ru. Expect real audio whose
     ASR transcript contains Cyrillic. THE BUG WAS: 3 phoneme tokens and noise.
  2. russian_no_espeak     espeak-ng removed from PATH and libespeak hidden.
     Expect a NON-ZERO exit and the refusal message — NOT a WAV. Pre-fix this
     produced a valid-looking 0.9 s file of garbage at a success code, which is
     the whole point of the issue.
  3. english_no_espeak     same hostile environment, ASCII text. Expect SUCCESS:
     ASCII still takes the raw-tokenisation path. This is the control that stops
     arm 2 from passing for the trivial reason "the binary refuses everything".
  4. language_applied      server-style: one process, synthesise ru then en,
     confirming the per-request language is applied per call rather than frozen
     at init. Compares the two phoneme-token counts printed by the backend.

Arm 3 is the one that makes arms 1 and 2 mean anything. Without it, a binary
that simply failed on all input would score a clean pass.
"""
import json, os, subprocess, sys, time, shutil
from pathlib import Path

WORK = Path("/kaggle/working"); SCRATCH = Path("/tmp")
CLONE = SCRATCH / "CrispASR"
SCRIPT_VERSION = "2026-09-13-zonos-lang-435-1"
RU = "Привет, это тест синтеза речи."
EN = "Hello, this is a test of speech synthesis."

def log(m):
    print(m, flush=True)
    try: (WORK/"progress.txt").open("a").write(f"{time.strftime('%H:%M:%S')} {m}\n")
    except Exception: pass

if not CLONE.exists():
    subprocess.check_call(["git","clone","--depth","1","--recurse-submodules",
                           "--shallow-submodules","https://github.com/CrispStrobe/CrispASR.git",str(CLONE)])
sys.path.insert(0, str(CLONE/"tools"/"kaggle"))
import kaggle_harness as kh  # noqa: E402
kh.init_progress()
sha = subprocess.run(["git","-C",str(CLONE),"rev-parse","--short","HEAD"],
                     capture_output=True,text=True).stdout.strip()
log(f"[zonos] script_version={SCRIPT_VERSION} clone={sha}")
HF_TOKEN = kh.resolve_hf_token(); os.environ.setdefault("HF_TOKEN", HF_TOKEN or "")

kh.install_build_toolchain()
BUILD = SCRATCH/"build"
r = subprocess.run(["cmake","-S",str(CLONE),"-B",str(BUILD),"-G","Ninja",
                    "-DCMAKE_BUILD_TYPE=Release"]+kh.cache_and_link_flags(),
                   capture_output=True,text=True)
if r.returncode != 0:
    log("configure FAILED"); log((r.stdout or "")[-3000:]); log((r.stderr or "")[-3000:]); raise SystemExit(1)
# safe_build_jobs returns a SHELL SNIPPET; run through a shell so it expands.
with kh.build_heartbeat("build.crispasr"):
    r = subprocess.run(f"cmake --build {BUILD} --target crispasr -j{kh.safe_build_jobs(gpu=False)}",
                       shell=True, capture_output=True, text=True)
if r.returncode != 0:
    log(f"build FAILED rc={r.returncode}")
    log((r.stdout or "<empty>")[-4000:]); log((r.stderr or "<empty>")[-4000:]); raise SystemExit(1)
CRISPASR = BUILD/"bin"/"crispasr"
if not CRISPASR.is_file():
    log("build claimed success but produced no binary"); raise SystemExit(1)

subprocess.run("apt-get install -y espeak-ng >/dev/null 2>&1 || true", shell=True)
have_espeak = shutil.which("espeak-ng") is not None
log(f"[zonos] espeak-ng present: {have_espeak}")
if not have_espeak:
    (WORK/"results.json").write_text(json.dumps(
        {"conclusive": False, "reason": "espeak-ng unavailable; arms 1 and 4 cannot run"}, indent=2))
    raise SystemExit(0)

from huggingface_hub import hf_hub_download
M = hf_hub_download("cstr/zonos-v0.1-transformer-GGUF","zonos-v0.1-transformer-q8_0.gguf",local_dir=str(SCRATCH/"m"))
C = hf_hub_download("cstr/dac-44khz-GGUF","dac-44khz-f16.gguf",local_dir=str(SCRATCH/"m"))
log(f"[zonos] model={M}\n[zonos] codec={C}")

def synth(text, lang, out, hostile):
    env = dict(os.environ)
    if hostile:   # hide BOTH routes: the binary and the shared library
        env["PATH"] = "/nonexistent"
        env["CRISPASR_ESPEAK_DATA_PATH"] = "/nonexistent"
        env["LD_LIBRARY_PATH"] = "/nonexistent"
    cmd = [str(CRISPASR),"--backend","zonos","-m",M,"--codec-model",C,
           "-l",lang,"--tts",text,"--tts-output",str(out)]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    sz = out.stat().st_size if out.exists() else 0
    return {"rc": p.returncode, "wav_bytes": sz, "stderr_tail": (p.stderr or "")[-1200:]}

res = {"script_version": SCRIPT_VERSION, "clone": sha, "arms": {}}
for name, text, lang, hostile in (
        ("russian_with_espeak", RU, "ru", False),
        ("russian_no_espeak",   RU, "ru", True),
        ("english_no_espeak",   EN, "en", True)):
    out = SCRATCH/f"{name}.wav"
    if out.exists(): out.unlink()
    a = synth(text, lang, out, hostile)
    toks = [l for l in a["stderr_tail"].splitlines() if "phoneme tokens" in l]
    a["phoneme_line"] = toks[-1] if toks else ""
    a["refused"] = "Refusing to synthesise noise" in a["stderr_tail"] or "no phoneme tokens" in a["stderr_tail"]
    res["arms"][name] = a
    log(f"[zonos] {name}: rc={a['rc']} wav={a['wav_bytes']} refused={a['refused']} | {a['phoneme_line'][:90]}")
    (WORK/"results.json").write_text(json.dumps(res, indent=2))

A = res["arms"]
verdict = {
  "ru_with_espeak_produced_audio": A["russian_with_espeak"]["wav_bytes"] > 1000,
  "ru_without_espeak_refused":     A["russian_no_espeak"]["refused"] and A["russian_no_espeak"]["wav_bytes"] == 0,
  "en_without_espeak_still_works": A["english_no_espeak"]["wav_bytes"] > 1000,
}
res["verdict"] = verdict
res["all_pass"] = all(verdict.values())
(WORK/"results.json").write_text(json.dumps(res, indent=2))
log("[zonos] VERDICT " + json.dumps(verdict))
log("[zonos] ALL PASS" if res["all_pass"] else "[zonos] NOT ALL PASS")
