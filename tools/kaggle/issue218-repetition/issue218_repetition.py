#!/usr/bin/env python3
"""CrispASR issue #218 repeated-phrase validation on Kaggle CUDA.

Builds the pushed feature branch, downloads the issue sample, then runs the
reported AR ASR backends with CRISPASR_NO_NGRAM_LOOPFIX=1 (raw) and default
(patched). Results are written to /kaggle/working/issue218_results.json.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

WORK = Path("/kaggle/working")
REPO = WORK / "CrispASR"
BUILD = WORK / "build"
CACHE = Path("/tmp/crispasr-model-cache")
AUDIO_ZIP = WORK / "t32-145s.wav.zip"
AUDIO = WORK / "t32-145s.wav"
RESULTS = WORK / "issue218_results.json"

CRISPASR_REPO = "https://github.com/CrispStrobe/CrispASR.git"
CRISPASR_REF = os.environ.get("CRISPASR_REF", "fix/issue218-repetition")
ISSUE_AUDIO_URL = "https://github.com/user-attachments/files/29652411/t32-145s.wav.zip"


def run(cmd: list[str], *, timeout: int = 1800, env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess:
    merged_env = {**os.environ, **(env or {})}
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout, env=merged_env)
    if proc.stdout:
        print(proc.stdout[-4096:], flush=True)
    if proc.stderr:
        print(proc.stderr[-4096:], flush=True)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout, proc.stderr)
    return proc


def extract_transcript(stdout: str) -> str:
    lines: list[str] = []
    for line in stdout.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("[") and "]" in s[:32]:
            # Drop timestamped display prefix, keep text after it.
            s = s.split("]", 1)[1].strip()
        if s.startswith("crispasr:") or s.startswith("system_info:"):
            continue
        lines.append(s)
    return " ".join(lines).strip()


def repetition_metrics(text: str) -> dict[str, int]:
    words = re.findall(r"[a-z0-9']+", text.lower())

    def max_run(n: int) -> int:
        best = 0
        i = 0
        while i + n <= len(words):
            gram = tuple(words[i:i + n])
            reps = 1
            j = i + n
            while j + n <= len(words) and tuple(words[j:j + n]) == gram:
                reps += 1
                j += n
            best = max(best, reps)
            i += max(n, 1)
        return best

    hey_count = sum(1 for w in words if w == "hey")
    come_on_count = 0
    for i in range(len(words) - 1):
        if words[i] == "come" and words[i + 1] == "on":
            come_on_count += 1
    return {
        "chars": len(text),
        "words": len(words),
        "max_unigram_run": max_run(1),
        "max_bigram_run": max_run(2),
        "max_trigram_run": max_run(3),
        "hey_count": hey_count,
        "come_on_count": come_on_count,
    }


def has_bad_loop(metrics: dict[str, int]) -> bool:
    return (
        metrics["max_unigram_run"] > 8
        or metrics["max_bigram_run"] > 5
        or metrics["max_trigram_run"] > 4
        or metrics["hey_count"] > 12
        or metrics["come_on_count"] > 8
    )


def write_ccache_artifact(kh) -> None:
    ccache_dir = WORK / ".ccache"
    if not ccache_dir.exists():
        kh.step("ccache_artifact_missing", path=str(ccache_dir))
        return
    artifact = WORK / "ccache.tar"
    proc = run(["tar", "-cf", str(artifact), "-C", str(WORK), ".ccache"], timeout=900, check=False)
    if proc.returncode == 0 and artifact.exists():
        kh.step("ccache_artifact_ready", path=str(artifact), size_mb=round(artifact.stat().st_size / 1e6, 1))
    else:
        kh.step("ccache_artifact_failed", rc=proc.returncode)


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    # Clone first so we can import the in-repo harness; fall back to bundled copy.
    if REPO.exists():
        shutil.rmtree(REPO)
    run(["git", "clone", "--depth", "1", "--branch", CRISPASR_REF, "--recursive", CRISPASR_REPO, str(REPO)], timeout=900)
    sys.path.insert(0, str(REPO / "tools" / "kaggle"))
    try:
        import kaggle_harness as kh
    except Exception:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import kaggle_harness as kh
    kh.init_progress()
    kh.step("start", ref=CRISPASR_REF)
    sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    kh.step("cloned", sha=sha)

    kh.resolve_hf_token()
    kh.install_build_toolchain()
    run(["nvidia-smi", "-L"], check=False, timeout=60)
    arch = kh.detect_cuda_arch()
    kh.step("cuda_arch", arch=arch)

    flags = kh.cuda_build_flags(arch) + kh.cache_and_link_flags()
    cmake_cmd = [
        "cmake", "-S", str(REPO), "-B", str(BUILD), "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCRISPASR_BUILD_TESTS=OFF",
        "-DCRISPASR_BUILD_SERVER=OFF",
        "-DCRISPASR_OPUS_FETCH=ON",
    ] + flags
    run(cmake_cmd, timeout=900)
    kh.step("cmake_done")

    with kh.build_heartbeat("cmake.build"):
        kh.sh_with_progress(
            f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli -j{kh.safe_build_jobs(gpu=True)}"
        )
    kh.step("build_done")

    cli = BUILD / "bin" / "crispasr"
    if not cli.exists():
        raise SystemExit(f"missing crispasr binary: {cli}")
    os.environ["LD_LIBRARY_PATH"] = f"{BUILD / 'src'}:{os.environ.get('LD_LIBRARY_PATH', '')}"

    kh.step("download_issue_audio")
    urllib.request.urlretrieve(ISSUE_AUDIO_URL, AUDIO_ZIP)
    with zipfile.ZipFile(AUDIO_ZIP) as zf:
        zf.extractall(WORK)
    if not AUDIO.exists():
        wavs = list(WORK.glob("*.wav"))
        if not wavs:
            raise SystemExit("issue audio zip did not contain a wav")
        wavs[0].rename(AUDIO)
    kh.step("audio_ready", size_mb=round(AUDIO.stat().st_size / 1e6, 1))

    from huggingface_hub import hf_hub_download

    kh.step("download_granite_plus")
    granite_plus_path = hf_hub_download(
        repo_id="cstr/granite-speech-4.1-2b-plus-GGUF",
        filename="granite-speech-4.1-2b-plus-q4_k.gguf",
        local_dir=str(CACHE),
    )
    kh.step("granite_plus_ready", path=granite_plus_path)
    cases = [
        {"name": "moss-transcribe", "backend": "moss-transcribe", "model": "auto", "args": ["--language", "en"]},
        {"name": "qwen3", "backend": "qwen3", "model": "auto", "args": ["--language", "en"]},
        {"name": "glm-asr", "backend": "glm-asr", "model": "auto", "args": ["--language", "en"]},
        {"name": "cohere", "backend": "cohere", "model": "auto", "args": ["--language", "en"]},
        {
            "name": "granite-4.1-plus",
            "backend": "granite",
            "model": granite_plus_path,
            "args": ["--language", "en", "--chunk-seconds", "10", "--chunk-overlap", "2"],
        },
        {
            "name": "canary-qwen",
            "backend": "canary-qwen",
            "model": "auto",
            "args": ["--chunk-seconds", "10", "--chunk-overlap", "2"],
        },
    ]

    results: list[dict] = []
    failures: list[str] = []
    for case in cases:
        kh.step("case_start", case=case["name"])
        for mode in ("raw", "patched"):
            env = {
                "CRISPASR_CACHE_DIR": str(CACHE),
                "CRISPASR_MODELS_DIR": str(CACHE),
            }
            if mode == "raw":
                env["CRISPASR_NO_NGRAM_LOOPFIX"] = "1"
                env["CRISPASR_MOSS_TRANSCRIBE_NO_LOOPFIX"] = "1"
            cmd = [
                str(cli),
                "--backend", case["backend"],
                "--gpu-backend", "cuda",
                "-m", case["model"],
                "--cache-dir", str(CACHE),
                "--no-prints",
                "--print-progress",
                "-f", str(AUDIO),
            ] + case["args"]
            t0 = time.time()
            proc = run(cmd, timeout=2400, env=env, check=False)
            elapsed = round(time.time() - t0, 2)
            text = extract_transcript(proc.stdout)
            metrics = repetition_metrics(text)
            rec = {
                "case": case["name"],
                "mode": mode,
                "returncode": proc.returncode,
                "elapsed_s": elapsed,
                "metrics": metrics,
                "bad_loop": has_bad_loop(metrics),
                "text_head": text[:500],
                "text_tail": text[-500:],
                "stderr_tail": proc.stderr[-2000:],
            }
            results.append(rec)
            kh.step("case_mode_done", case=case["name"], mode=mode, rc=proc.returncode, elapsed_s=elapsed, **metrics)
        patched = [r for r in results if r["case"] == case["name"] and r["mode"] == "patched"][-1]
        if patched["returncode"] != 0:
            failures.append(f"{case['name']}: patched run failed rc={patched['returncode']}")
        elif patched["bad_loop"]:
            failures.append(f"{case['name']}: patched transcript still has repetition loop {patched['metrics']}")
        kh.step("case_done", case=case["name"], failures=len(failures), free_tmp_gb=kh.free_gb("/tmp"))

    payload = {"ref": CRISPASR_REF, "sha": sha, "audio": str(AUDIO), "results": results, "failures": failures}
    RESULTS.write_text(json.dumps(payload, indent=2))
    kh.step("results_written", path=str(RESULTS), failures=len(failures))
    write_ccache_artifact(kh)
    print(json.dumps(payload, indent=2), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
