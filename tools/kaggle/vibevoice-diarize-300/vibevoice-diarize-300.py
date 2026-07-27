"""#300 follow-up: prove vibevoice now surfaces STRUCTURED speaker labels.

The original #300 kernel recorded vibevoice as a no-op: its adapter never set
`seg.speaker`, so the whole JSON answer came back as one segment's TEXT and the
structured-label change could not reach it. The reporter came back with "What
about vibevoice?" — this validates the fix for exactly that.

Under test: the vibevoice adapter now parses the model's answer
(`[{"Start":..,"End":..,"Speaker":N,"Content":".."}]`) into one segment per
utterance, with the speaker in the structured field.

The A/B runs on ONE binary, flipped by the env gate the fix ships with — so the
two arms cannot differ by anything except the code path:
    RAW  CRISPASR_VIBEVOICE_RAW_TRANSCRIPT=1  → pre-fix: one segment, raw blob
    NEW  (unset)                              → per-utterance segments + labels

Proof-of-work rules (harness gotcha #24): every arm must exit 0 AND produce
non-empty output; a label count is only meaningful next to those. `\\(Speaker N\\)`
matches ONLY the structured form the CLI prefixes — never the model's own
`"Speaker":0` JSON text, which is what the RAW arm emits.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

TEMP = Path("/kaggle/temp"); OUT = Path("/kaggle/working")
REPO = TEMP / "CrispASR"; MODELS = TEMP / "models"
for d in (TEMP, OUT, MODELS):
    d.mkdir(parents=True, exist_ok=True)

BRANCH = "fix/300-vibevoice-diarize"

SPK_RE = re.compile(r"\(Speaker \d+\)")   # the STRUCTURED label, CLI-prefixed
RAW_RE = re.compile(r'"Content"\s*:')     # the model's own JSON leaking into text


def _eh(et, ev, tb):
    try:
        (OUT / "error.txt").write_text("".join(traceback.format_exception(et, ev, tb)))
    except Exception:
        pass
    sys.__excepthook__(et, ev, tb)


sys.excepthook = _eh


def run(cmd, **kw):
    kw.setdefault("capture_output", True)
    kw.setdefault("text", True)
    return subprocess.run(cmd, **kw)


print(json.dumps({"step": "start", "branch": BRANCH}), flush=True)

# ── Clone the feature branch + submodules ────────────────────────────────────
if REPO.exists():
    shutil.rmtree(REPO)
r = run(["git", "clone", "--depth", "5", "--branch", BRANCH,
         "https://github.com/CrispStrobe/CrispASR.git", str(REPO)], capture_output=False)
if r.returncode:
    (OUT / "results.json").write_text(json.dumps({"error": f"clone of {BRANCH} failed"}))
    raise SystemExit(1)
run(["git", "-C", str(REPO), "submodule", "update", "--init", "--recursive", "--depth", "1"],
    capture_output=False, timeout=1800)
sys.path.insert(0, os.path.join(str(REPO), "tools", "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
HEAD = run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"]).stdout.strip()
kh.step("cloned", head=HEAD)

# The gate has to EXIST in this checkout, else the "RAW" arm silently runs the
# new path and the A/B compares a thing to itself.
gate_src = (REPO / "examples/cli/crispasr_backend_vibevoice.cpp").read_text()
if "CRISPASR_VIBEVOICE_RAW_TRANSCRIPT" not in gate_src:
    (OUT / "results.json").write_text(json.dumps({"error": "env gate missing from checkout"}))
    kh.step("gate.MISSING")
    raise SystemExit(1)
kh.step("gate.present")

kh.install_build_toolchain()
run(["apt-get", "install", "-y", "-q", "libopenblas-dev", "ffmpeg"], capture_output=False)

BDIR = TEMP / "build"
# CUDA needs the harness flags, not a bare -DGGML_CUDA=ON: without the stubs
# dir on LIBRARY_PATH cmake fails Generate with `Target "ggml-cuda" links to
# CUDA::cuda_driver but the target was not found`. cuda_build_flags() also
# pins ONE arch (T4 → 75), which is what keeps nvcc from OOMing the box.
CUDA = os.path.isfile("/usr/local/cuda/bin/nvcc") or shutil.which("nvcc") is not None
extra = kh.cuda_build_flags() if CUDA else []
JOBS = "2" if CUDA else str(min(4, os.cpu_count() or 2))  # nvcc TUs are RAM-heavy
cfg = ["cmake", "-G", "Ninja", "-B", str(BDIR), "-S", str(REPO),
       "-DCMAKE_BUILD_TYPE=Release"] + kh.cache_and_link_flags() + extra
kh.step("cfg.begin", cuda=CUDA, jobs=JOBS, extra=" ".join(extra))
r = run(cfg, capture_output=False)
if r.returncode and CUDA:
    # A CUDA-toolchain problem must not cost the whole run — the thing under
    # test is a text parse, which a CPU build validates just as well (slower).
    kh.step("cfg.FAIL.cuda.retry_cpu")
    shutil.rmtree(BDIR, ignore_errors=True)
    CUDA = False
    JOBS = str(min(4, os.cpu_count() or 2))
    cfg = ["cmake", "-G", "Ninja", "-B", str(BDIR), "-S", str(REPO),
           "-DCMAKE_BUILD_TYPE=Release"] + kh.cache_and_link_flags()
    r = run(cfg, capture_output=False)
if r.returncode:
    kh.step("cfg.FAIL")
    raise SystemExit(1)
with kh.build_heartbeat("build"):
    r = run(["cmake", "--build", str(BDIR), "--target", "crispasr-cli", "-j", JOBS],
            capture_output=False)
if r.returncode:
    kh.step("build.FAIL")
    raise SystemExit(1)
CLI = BDIR / "bin" / "crispasr"
if not CLI.exists():
    c = [p for p in BDIR.rglob("crispasr") if p.is_file() and os.access(p, os.X_OK)]
    if not c:
        kh.step("cli.MISSING")
        raise SystemExit(1)
    CLI = c[0]
kh.step("build.done", cli=str(CLI))

# Also build + run the hermetic parser unit test — it needs no model, so a
# failure here localizes the bug to the parse rather than the pipeline.
with kh.build_heartbeat("build.unit"):
    r = run(["cmake", "--build", str(BDIR), "--target", "test-vibevoice-transcript", "-j", JOBS],
            capture_output=False)
UNIT = {"built": r.returncode == 0}
if r.returncode == 0:
    tb = BDIR / "bin" / "test-vibevoice-transcript"
    ru = run([str(tb)])
    UNIT.update(exit=ru.returncode, tail=(ru.stdout or "")[-300:])
kh.step("unit", **{k: v for k, v in UNIT.items() if k != "tail"})

# ── Samples: multispeaker (the diarization clip) + jfk (content sanity) ──────
WAV_MS = REPO / "samples" / "multispeaker.wav"
WAV_JFK = REPO / "samples" / "jfk.wav"
assert WAV_MS.is_file() and WAV_JFK.is_file()
PCM = TEMP / "multispeaker.s16le"
run(["ffmpeg", "-y", "-i", str(WAV_MS), "-f", "s16le", "-ar", "16000", "-ac", "1", str(PCM)],
    capture_output=False)
assert PCM.is_file() and PCM.stat().st_size > 0, "ffmpeg produced no PCM"
kh.step("pcm.ready", bytes=PCM.stat().st_size)

from huggingface_hub import hf_hub_download  # noqa: E402

kh.step("dl.vibe.begin")
VIBE = Path(hf_hub_download(repo_id="cstr/vibevoice-asr-GGUF",
                            filename="vibevoice-asr-q4_k.gguf", local_dir=str(MODELS)))
kh.step("dl.vibe.done", gb=round(VIBE.stat().st_size / 1e9, 2))


def _env(raw_gate):
    e = {**os.environ, "LD_LIBRARY_PATH": f"{BDIR}/src:" + os.environ.get("LD_LIBRARY_PATH", "")}
    if raw_gate:
        e["CRISPASR_VIBEVOICE_RAW_TRANSCRIPT"] = "1"
    else:
        e.pop("CRISPASR_VIBEVOICE_RAW_TRANSCRIPT", None)
    return e


def _summ(out):
    lines = [ln for ln in out.splitlines() if ln.strip()]
    return {
        "labels": len(SPK_RE.findall(out)),        # structured "(Speaker N)"
        "raw_json_hits": len(RAW_RE.findall(out)), # the model's blob as text
        "chars": len(out),
        "nonempty_lines": len(lines),
        "head": "\n".join(lines[:6])[:900],
        "tail": "\n".join(lines[-4:])[-500:],
    }


def run_file(wav, raw_gate, timeout=3600):
    t0 = time.time()
    r = run([str(CLI), "-m", str(VIBE), "-f", str(wav), "--backend", "vibevoice"],
            env=_env(raw_gate), timeout=timeout)
    s = _summ(r.stdout or "")
    s.update(exit=r.returncode, wall=round(time.time() - t0, 1),
             stderr_tail=(r.stderr or "")[-400:] if r.returncode else "")
    return s


def run_stream(raw_gate, extra=None, timeout=5400):
    cmd = [str(CLI), "--stream", "--backend", "vibevoice", "-m", str(VIBE)] + (extra or [])
    t0 = time.time()
    with open(PCM, "rb") as fin:
        r = run(cmd, env=_env(raw_gate), stdin=fin, timeout=timeout)
    s = _summ(r.stdout or "")
    s.update(exit=r.returncode, wall=round(time.time() - t0, 1),
             stderr_tail=(r.stderr or "")[-400:] if r.returncode else "")
    finals = spk = 0
    for ln in (r.stdout or "").splitlines():
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            ev = json.loads(ln)
        except Exception:
            continue
        if ev.get("type") == "final":
            finals += 1
            if ev.get("speaker"):
                spk += 1
    s.update(json_finals=finals, json_finals_with_speaker=spk)
    return s


R = {"unit_test": UNIT}


def save():
    (OUT / "results.json").write_text(json.dumps({"branch": BRANCH, "head": HEAD, "results": R}, indent=2))


# T1 RAW file — the pre-fix behaviour: raw blob in the text, no structured label.
R["file_raw"] = run_file(WAV_MS, True); kh.step("t1.file_raw", **{k: R["file_raw"][k] for k in ("exit", "labels", "raw_json_hits", "wall")}); save()
# T2 NEW file — THE change: structured labels, no blob.
R["file_new"] = run_file(WAV_MS, False); kh.step("t2.file_new", **{k: R["file_new"][k] for k in ("exit", "labels", "raw_json_hits", "wall")}); save()
# T3 NEW plain --stream — inline labels, which is the reporter's ask.
R["stream_new"] = run_stream(False); kh.step("t3.stream_new", **{k: R["stream_new"][k] for k in ("exit", "labels", "raw_json_hits", "wall")}); save()
# T4 NEW --stream-json + VAD — a `final` event carrying a "speaker" field.
R["streamjson_new"] = run_stream(False, extra=["--stream-json", "--vad", "--stream-final-on-silence-ms", "800"])
kh.step("t4.streamjson_new", **{k: R["streamjson_new"][k] for k in ("exit", "labels", "json_finals", "json_finals_with_speaker", "wall")}); save()
# T5 NEW jfk — content survives the parse (matches the pinned regression text).
R["jfk_new"] = run_file(WAV_JFK, False); kh.step("t5.jfk_new", **{k: R["jfk_new"][k] for k in ("exit", "labels", "raw_json_hits", "wall")}); save()
# T6 RAW --stream — the pre-fix streaming arm, for the same-binary A/B.
try:
    R["stream_raw"] = run_stream(True); kh.step("t6.stream_raw", **{k: R["stream_raw"][k] for k in ("exit", "labels", "raw_json_hits", "wall")})
except Exception as e:
    R["stream_raw"] = {"error": f"{type(e).__name__}: {e}"}
    kh.step("t6.err", error=str(e))
save()


def _ok(k):
    v = R.get(k)
    return isinstance(v, dict) and v.get("exit") == 0 and v.get("nonempty_lines", 0) > 0


EXPECT_JFK = "ask not what your country can do for you"

verdict = {}
# Pre-fix arm really is pre-fix: the blob reaches the transcript, no labels.
verdict["raw_file_has_blob_no_labels"] = _ok("file_raw") and R["file_raw"]["labels"] == 0 \
    and R["file_raw"]["raw_json_hits"] > 0
# The fix: labels appear and the blob is gone.
verdict["new_file_has_labels"] = _ok("file_new") and R["file_new"]["labels"] > 0
verdict["new_file_no_blob"] = _ok("file_new") and R["file_new"]["raw_json_hits"] == 0
# Streaming — what the reporter asked for.
verdict["new_stream_has_labels"] = _ok("stream_new") and R["stream_new"]["labels"] > 0
verdict["new_streamjson_speaker_field"] = _ok("streamjson_new") \
    and R["streamjson_new"].get("json_finals_with_speaker", 0) > 0
# Content is not damaged by the parse.
verdict["jfk_content_intact"] = _ok("jfk_new") and EXPECT_JFK in (R["jfk_new"]["head"] + R["jfk_new"]["tail"]).lower()
verdict["unit_test_passed"] = UNIT.get("exit") == 0
verdict["raw_stream_no_labels"] = (not _ok("stream_raw")) or R["stream_raw"].get("labels", 0) == 0

verdict["PASS"] = bool(verdict["raw_file_has_blob_no_labels"] and verdict["new_file_has_labels"]
                       and verdict["new_file_no_blob"] and verdict["new_stream_has_labels"]
                       and verdict["new_streamjson_speaker_field"] and verdict["unit_test_passed"])

(OUT / "results.json").write_text(json.dumps(
    {"branch": BRANCH, "head": HEAD, "verdict": verdict, "results": R}, indent=2))

print("\n=== #300 VIBEVOICE STRUCTURED-DIARIZATION VALIDATION ===", flush=True)
for k in ("file_raw", "file_new", "stream_raw", "stream_new", "streamjson_new", "jfk_new"):
    v = R.get(k)
    if isinstance(v, dict):
        print(f"  {k:16s} exit={v.get('exit')} labels={v.get('labels')} blob={v.get('raw_json_hits')} "
              f"json_final_spk={v.get('json_finals_with_speaker')} lines={v.get('nonempty_lines')} "
              f"wall={v.get('wall')}s", flush=True)
print("\n  --- transcripts ---", flush=True)
for k in ("file_raw", "file_new"):
    if isinstance(R.get(k), dict):
        print(f"  [{k}]\n{R[k].get('head', '')}\n", flush=True)
print("  --- verdict ---", flush=True)
for k, val in verdict.items():
    print(f"  {k:32s} {val}", flush=True)
print(f"\n  OVERALL: {'PASS' if verdict['PASS'] else 'FAIL'}", flush=True)

kh.step("done", **{k: v for k, v in verdict.items()})
print(json.dumps({"step": "done", "PASS": verdict["PASS"]}), flush=True)
if not verdict["PASS"]:
    raise SystemExit(2)
