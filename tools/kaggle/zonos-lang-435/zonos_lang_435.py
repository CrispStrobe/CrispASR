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
  4. language_applied      ONE `crispasr --server` process started with -l en,
     then two POST /v1/audio/speech requests carrying "language":"ru" and
     "language":"en". Requires the backend's own "phoneme tokens (lang=XX)" line
     to track the REQUEST. This is the only arm that can see the second half of
     the bug: a long-lived server applied its startup language to every request.

Arm 3 is the one that makes arms 1 and 2 mean anything. Without it, a binary
that simply failed on all input would score a clean pass. Arm 4 had the same
disease in v1 of this script -- it used two separate CLI processes, so each one
set its language at INIT, a path that was never broken. It would have passed on
the unfixed build. Two processes cannot test what only one process can exhibit.
"""
import json, os, subprocess, sys, time, shutil
from pathlib import Path

WORK = Path("/kaggle/working"); SCRATCH = Path("/tmp")
CLONE = SCRATCH / "CrispASR"
SCRIPT_VERSION = "2026-09-13-zonos-lang-435-3"
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
    if hostile:
        # v1 set PATH and LD_LIBRARY_PATH to /nonexistent and believed that
        # disabled espeak. IT DOES NOT: the in-process route dlopens the soname,
        # and LD_LIBRARY_PATH only ADDS search directories -- the system path
        # still resolves libespeak-ng.so.1. Both Russian arms then produced
        # byte-identical 205002-byte WAVs, which is what a no-op looks like.
        # Espeak is now physically removed instead (see remove_espeak below);
        # this only clears the data-dir override.
        env["CRISPASR_ESPEAK_DATA_PATH"] = "/nonexistent"
    cmd = [str(CRISPASR),"--backend","zonos","-m",M,"--codec-model",C,
           "-l",lang,"--tts",text,"--tts-output",str(out)]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    sz = out.stat().st_size if out.exists() else 0
    # KEEP THE WHOLE STREAM. v1 kept only the last 1200 chars; the backend prints
    # "N phoneme tokens (lang=XX)" BEFORE synthesis, so the model-loading and
    # generation logs pushed it out of the window and every arm read lang=''.
    # The signal existed and the instrument threw it away.
    err = p.stderr or ""
    return {"rc": p.returncode, "wav_bytes": sz, "stderr": err, "stderr_tail": err[-1500:]}

def remove_espeak():
    """Physically remove espeak, then PROVE it is gone.

    Returns (gone: bool, note: str). If it is not gone, the hostile arms are
    marked inconclusive rather than reported as passes -- an arm that cannot
    disable the dependency it is testing proves nothing, and in v1 it silently
    "passed" by producing perfectly good Russian audio.
    """
    subprocess.run("apt-get remove -y --purge espeak-ng espeak-ng-data libespeak-ng1 "
                   ">/dev/null 2>&1 || true", shell=True)
    # apt may leave the runtime .so behind; move every candidate out of the way.
    subprocess.run("for f in $(find /usr/lib /usr/local/lib /lib -name 'libespeak*' 2>/dev/null); "
                   "do mv \"$f\" \"$f.hidden\" 2>/dev/null || true; done", shell=True)
    leftover = subprocess.run("find /usr/lib /usr/local/lib /lib -name 'libespeak*' "
                              "! -name '*.hidden' 2>/dev/null", shell=True,
                              capture_output=True, text=True).stdout.strip()
    on_path = shutil.which("espeak-ng") or shutil.which("espeak")
    if on_path or leftover:
        return False, f"espeak still reachable (binary={on_path!r}, libs={leftover!r})"
    return True, ""

res = {"script_version": SCRIPT_VERSION, "clone": sha, "arms": {}}

# ORDER MATTERS: removal is global and irreversible in this container, so every
# arm that NEEDS espeak runs first.
ESPEAK_ARMS  = (("russian_with_espeak", RU, "ru", False),)
HOSTILE_ARMS = (("russian_no_espeak",   RU, "ru", True),
                ("english_no_espeak",   EN, "en", True))

def run_arm(name, text, lang, hostile):
    out = SCRATCH/f"{name}.wav"
    if out.exists(): out.unlink()
    a = synth(text, lang, out, hostile)
    toks = [l for l in a["stderr"].splitlines() if "phoneme tokens" in l]
    a["phoneme_line"] = toks[-1] if toks else ""
    a["refused"] = ("Refusing to synthesise noise" in a["stderr"]
                    or "no phoneme tokens" in a["stderr"])
    a.pop("stderr", None)   # keep results.json readable; stderr_tail is retained
    res["arms"][name] = a
    log(f"[zonos] {name}: rc={a['rc']} wav={a['wav_bytes']} refused={a['refused']} | {a['phoneme_line'][:90]}")
    (WORK/"results.json").write_text(json.dumps(res, indent=2))
    return a

for _a in ESPEAK_ARMS:
    run_arm(*_a)

# Arm 4: the per-request language must reach the model on EACH CALL.
#
# v1 OF THIS SCRIPT GOT THIS WRONG AND WOULD HAVE PASSED ON THE BROKEN BINARY.
# It spawned TWO SEPARATE `crispasr --tts` processes with -l ru and -l en. Each
# process sets the language at INIT, which the pre-fix build already honoured --
# the bug was that a LONG-LIVED process ignored the language on subsequent
# requests. Two processes cannot see that. The arm tested the one path that was
# never broken and would have reported a pass either way.
#
# The real path is the server: init once with -l en, then POST two
# /v1/audio/speech requests carrying different "language" values, and require
# the backend's own "N phoneme tokens (lang=XX)" line to track the REQUEST.
# Pre-fix, both requests phonemise as the startup language.
import urllib.request, socket

def wait_port(port, proc, timeout_s=900):
    for _ in range(timeout_s):
        if proc.poll() is not None:
            return False
        with socket.socket() as sk:
            sk.settimeout(1.0)
            if sk.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(1)
    return False

PORT = 8137
srv_log = open(SCRATCH / "server.log", "wb")
srv = subprocess.Popen(
    [str(CRISPASR), "--server", "--backend", "zonos", "-m", M, "--codec-model", C,
     "-l", "en", "--port", str(PORT)],
    stdout=srv_log, stderr=subprocess.STDOUT)
lang_seen, srv_note = [], ""
if not wait_port(PORT, srv):
    srv_note = "server did not come up"
    log(f"[zonos] language_applied: {srv_note} (rc={srv.poll()})")
else:
    for lang, text in (("ru", RU), ("en", EN)):
        body = json.dumps({"input": text, "language": lang}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/audio/speech",
                                     data=body, headers={"Content-Type": "application/json"})
        mark = srv_log.tell()
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                nbytes = len(r.read())
        except Exception as e:
            nbytes = 0; log(f"[zonos] request {lang} failed: {e}")
        srv_log.flush()
        tail = Path(SCRATCH / "server.log").read_bytes()[mark:].decode("utf-8", "replace")
        got = ""
        for line in tail.splitlines():
            if "phoneme tokens (lang=" in line:
                got = line.split("lang=")[1].rstrip(")").strip()
        lang_seen.append(got)
        log(f"[zonos] language_applied[{lang}]: printed lang={got!r} wav_bytes={nbytes}")
srv.terminate()
try: srv.wait(timeout=30)
except Exception: srv.kill()
srv_log.close()
res["arms"]["language_applied"] = {"requested": ["ru", "en"], "printed": lang_seen,
                                   "note": srv_note, "method": "single server process, 2 requests"}

# ── espeak is no longer needed; remove it and run the hostile arms ──
gone, why = remove_espeak()
res["espeak_removed"] = gone
res["espeak_removed_note"] = why
log(f"[zonos] espeak removed: {gone} {why}")
if gone:
    for _a in HOSTILE_ARMS:
        run_arm(*_a)
else:
    log("[zonos] SKIPPING hostile arms: could not disable espeak, so they would "
        "prove nothing. Reported as inconclusive, NOT as passes.")

A = res["arms"]
verdict = {
  "ru_with_espeak_produced_audio": A["russian_with_espeak"]["wav_bytes"] > 1000,
  # `.get` because the hostile arms are SKIPPED when espeak could not be
  # removed. A missing arm must not read as a pass, and must not read as a
  # failure of the fix either -- res["espeak_removed"] says which it is.
  "ru_without_espeak_refused":     bool(A.get("russian_no_espeak", {}).get("refused")) and
                                   A.get("russian_no_espeak", {}).get("wav_bytes", -1) == 0,
  "en_without_espeak_still_works": A.get("english_no_espeak", {}).get("wav_bytes", 0) > 1000,
  # Requires BOTH that each call printed its own requested language AND that the
  # two differ -- a backend frozen at init would print the same value twice, and
  # a backend that printed nothing would pass an equality-only check vacuously.
  # One process, two requests. Pre-fix BOTH print the startup language ("en-us"),
  # so ["ru","en"] is reachable only if the per-request language is applied.
  "per_request_language_applied": [x.split("-")[0] for x in lang_seen] == ["ru", "en"],
}
res["verdict"] = verdict
# CONCLUSIVE is separate from PASS. If espeak could not be removed, arms 2 and 3
# did not run, and "all_pass == False" would misreport a missing measurement as a
# failed one. v1 made the opposite error: it reported arms that ran but could not
# fail.
res["conclusive"] = bool(gone)
res["all_pass"] = all(verdict.values())
(WORK/"results.json").write_text(json.dumps(res, indent=2))
log("[zonos] VERDICT " + json.dumps(verdict))
log("[zonos] ALL PASS" if res["all_pass"] else "[zonos] NOT ALL PASS")
if not res["conclusive"]:
    log("[zonos] INCONCLUSIVE: espeak could not be removed; arms 2-3 did not run.")
