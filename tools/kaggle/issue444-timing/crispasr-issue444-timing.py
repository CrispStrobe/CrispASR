"""Issue #444: v0.8.34 whole-slice regression vs local-island repair."""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp")
REPO = WORK / "CrispASR"
MODELS = TEMP / "models"
REF = os.environ.get("CRISPASR_REF", "fix/444-timing")


def run(cmd, *, cwd=None, capture=False):
    print("$", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, cwd=cwd, check=True, text=True,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.STDOUT if capture else None)


run(["git", "clone", "--recursive", "--branch", REF, "--single-branch",
     "https://github.com/CrispStrobe/CrispASR", str(REPO)])
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
kh.resolve_hf_token()
kh.install_build_toolchain()
sha = run(["git", "rev-parse", "HEAD"], cwd=REPO, capture=True).stdout.strip()
kh.step("script.start", ref=REF, sha=sha)


def build(label, ref):
    run(["git", "checkout", "--detach", ref], cwd=REPO)
    run(["git", "submodule", "update", "--init", "--recursive"], cwd=REPO)
    out = TEMP / f"build-{label}"
    flags = kh.cuda_build_flags(kh.detect_cuda_arch())
    cache = kh.cache_and_link_flags()
    with kh.build_heartbeat(f"{label}.configure"):
        run(["cmake", "-S", str(REPO), "-B", str(out), "-GNinja",
             "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON", *flags, *cache])
    with kh.build_heartbeat(f"{label}.build"):
        kh.sh_with_progress(
            f"stdbuf -oL -eL cmake --build {out} --target crispasr-cli test-align-only "
            f"-- -j{kh.safe_build_jobs(gpu=True)}"
        )
    if label == "current":
        unit = run([str(out / "bin" / "test-align-only"), "[issue444]"], capture=True)
        (WORK / "current-unit.log").write_text(unit.stdout or "")
        kh.step("current.unit", passed=True)
    return out / "bin" / "crispasr"


current = build("current", sha)
run(["git", "fetch", "--depth", "1", "origin", "tag", "v0.8.34"], cwd=REPO)
baseline = build("v0834", "v0.8.34")

run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "hf_transfer"])
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import hf_hub_download  # noqa: E402

MODELS.mkdir(parents=True, exist_ok=True)


def hf(repo, filename):
    path = Path(hf_hub_download(repo_id=repo, filename=filename, local_dir=str(MODELS)))
    kh.step("download", file=path.name, bytes=path.stat().st_size)
    return path


qwen = hf("cstr/qwen3-asr-1.7b-GGUF", "qwen3-asr-1.7b-q4_k.gguf")
aligner = hf("cstr/qwen3-forced-aligner-0.6b-GGUF", "qwen3-forced-aligner-0.6b-q4_k.gguf")
audio = TEMP / "issue444.mp3"
run(["curl", "-L", "--fail", "--retry", "3",
     "https://github.com/user-attachments/files/32415984/1.mp3", "-o", str(audio)])
shutil.rmtree(REPO)


STAMP = re.compile(r"(\d+):(\d+):(\d+),(\d+)\s+-->\s+(\d+):(\d+):(\d+),(\d+)")


def ms(parts):
    h, m, s, milli = map(int, parts)
    return ((h * 60 + m) * 60 + s) * 1000 + milli


def execute(binary, label, *, align=True):
    prefix = WORK / label
    cmd = [str(binary), "--backend", "qwen3-1.7b", "-m", str(qwen), "--vad", "-vm", "firered",
           "-vmsd", "30"]
    if align:
        cmd += ["-am", str(aligner)]
    cmd += ["--split-on-punct", "-osrt", "-ojf", "-l", "zh",
            "-f", str(audio), "-of", str(prefix), "-t", "4", "-v"]
    started = time.time()
    with kh.build_heartbeat(f"{label}.inference", interval_s=30):
        p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           env={**os.environ, "CRISPASR_ALIGN_DEBUG": "1"})
    elapsed = round(time.time() - started, 2)
    (WORK / f"{label}.log").write_text(p.stdout or "")
    srt_path = prefix.with_suffix(".srt")
    text = srt_path.read_text() if srt_path.exists() else ""
    rows = []
    for cue in re.split(r"\n\s*\n", text.strip()):
        match = STAMP.search(cue)
        if not match:
            continue
        lines = cue.splitlines()
        rows.append({"start_ms": ms(match.groups()[:4]), "end_ms": ms(match.groups()[4:]),
                     "text": " ".join(lines[2:]).strip()})
    (WORK / f"{label}-parsed.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    kh.step(f"{label}.result", rc=p.returncode, elapsed_s=elapsed, cues=len(rows))
    return p.returncode, elapsed, rows


raw_rc, raw_s, raw_rows = execute(current, "current-raw", align=False)
baseline_rc, baseline_s, baseline_rows = execute(baseline, "v0834")
current_rc, current_s, current_rows = execute(current, "current")


def by_text(rows):
    return {r["text"].replace(" ", ""): r for r in rows}


bad = by_text(baseline_rows)
raw = by_text(raw_rows)
fixed = by_text(current_rows)
expected_starts = {
    "你找谁？": 28630,
    "找你。": 29910,
    "我不在。": 30870,
    "那我跟谁说话呢？": 31830,
    "跟小郭姐姐。": 33670,
}
anchor_errors = {text: abs(fixed[text]["start_ms"] - expected) for text, expected in expected_starts.items()
                 if text in fixed}
ordered = (bool(current_rows)
           and all(r["end_ms"] >= r["start_ms"] for r in current_rows)
           and all(current_rows[i]["start_ms"] >= current_rows[i - 1]["end_ms"]
                   for i in range(1, len(current_rows))))
baseline_reproduced = ("你找谁？" in bad and bad["你找谁？"]["start_ms"] <= 27500)
raw_interpolation_reproduced = ("你找谁？" in raw and raw["你找谁？"]["start_ms"] <= 27500)
anchors_pass = len(anchor_errors) == len(expected_starts) and max(anchor_errors.values()) <= 400
late_repair = ("你……" in fixed and "他们个个憨是憨，哥是哥的，凭什么要我连轴转？" in fixed
               and fixed["你……"]["end_ms"]
               <= fixed["他们个个憨是憨，哥是哥的，凭什么要我连轴转？"]["start_ms"])
passed = (raw_rc == 0 and baseline_rc == 0 and current_rc == 0 and raw_interpolation_reproduced
          and baseline_reproduced and ordered and anchors_pass and late_repair)

summary = {
    "sha": sha,
    "passed": passed,
    "raw": {"rc": raw_rc, "elapsed_s": raw_s, "cues": len(raw_rows)},
    "baseline": {"rc": baseline_rc, "elapsed_s": baseline_s, "cues": len(baseline_rows)},
    "current": {"rc": current_rc, "elapsed_s": current_s, "cues": len(current_rows)},
    "baseline_reproduced": baseline_reproduced,
    "raw_interpolation_reproduced": raw_interpolation_reproduced,
    "ordered": ordered,
    "anchor_errors_ms": anchor_errors,
    "late_repair": late_repair,
}
(WORK / "issue444-timing-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
kh.export_ccache_tar()
kh.step("script.end", passed=passed)
assert passed, summary
