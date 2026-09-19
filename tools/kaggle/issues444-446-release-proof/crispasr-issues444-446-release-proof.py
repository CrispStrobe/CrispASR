"""Live release proof for issue #444 and #446 fixes."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp")
REPO = WORK / "CrispASR"
BUILD = TEMP / "build-issues444-446"
MODELS = TEMP / "models"
SHA = "a9c627a1d465a1de36290831a0e1ca20df0c7ec3"


def run(cmd, *, cwd=None, capture=False):
    print("$", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, cwd=cwd, check=True, text=True,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.STDOUT if capture else None)


run(["git", "clone", "--recursive", "https://github.com/CrispStrobe/CrispASR", str(REPO)])
run(["git", "checkout", "--detach", SHA], cwd=REPO)
run(["git", "submodule", "update", "--init", "--recursive"], cwd=REPO)
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
kh.resolve_hf_token()
kh.step("script.start", sha=SHA)
kh.install_build_toolchain()

flags = kh.cuda_build_flags(kh.detect_cuda_arch())
cache = kh.cache_and_link_flags()
with kh.build_heartbeat("configure"):
    run(["cmake", "-S", str(REPO), "-B", str(BUILD), "-GNinja",
         "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON", *flags, *cache])
with kh.build_heartbeat("build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target crispasr-cli crispasr-chat "
        f"-- -j{kh.safe_build_jobs(gpu=True)}"
    )

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
minicpm = hf("openbmb/MiniCPM5-2B-GGUF", "MiniCPM5-2B-Q4_K_M.gguf")
audio = TEMP / "issue444.mp3"
run(["curl", "-L", "--fail", "--retry", "3",
     "https://github.com/user-attachments/files/32415984/1.mp3", "-o", str(audio)])

cli = BUILD / "bin" / "crispasr"
prefix = WORK / "issue444-current"
cmd = [str(cli), "--backend", "qwen3-1.7b", "-m", str(qwen), "--vad", "-vm", "firered",
       "-vmsd", "30", "-am", str(aligner), "--split-on-punct", "-osrt", "-ojf", "-l", "zh",
       "-f", str(audio), "-of", str(prefix), "-t", "4", "-v"]
p = run(cmd, capture=True)
(WORK / "issue444-current.log").write_text(p.stdout)

srt = prefix.with_suffix(".srt")
assert srt.exists() and srt.stat().st_size > 0
stamps = []
for m in re.finditer(r"(\d+):(\d+):(\d+),(\d+)\s+-->\s+(\d+):(\d+):(\d+),(\d+)", srt.read_text()):
    v = list(map(int, m.groups()))
    start = ((v[0] * 60 + v[1]) * 60 + v[2]) * 1000 + v[3]
    end = ((v[4] * 60 + v[5]) * 60 + v[6]) * 1000 + v[7]
    stamps.append((start, end))
assert stamps and all(e >= s for s, e in stamps)
assert all(stamps[i][0] >= stamps[i - 1][1] for i in range(1, len(stamps))), stamps
kh.step("issue444.pass", cues=len(stamps), first_ms=stamps[0][0], last_ms=stamps[-1][1])

chat = BUILD / "bin" / "crispasr-chat"
p = subprocess.run([str(chat), "-m", str(minicpm), "-c", "512", "-t", "4",
                    "--max-tokens", "8", "--temp", "0", "--one-shot", "--no-color"],
                   input="Reply with OK.", text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
(WORK / "issue446-chat.stdout").write_text(p.stdout)
(WORK / "issue446-chat.stderr").write_text(p.stderr)
assert p.returncode == 0, p.stderr[-4000:]
assert p.stdout.strip(), "MiniCPM5 loaded but generated no output"
assert "unknown pre-tokenizer" not in p.stderr
kh.step("issue446.pass", output=p.stdout.strip()[:200])

summary = {
    "sha": SHA,
    "issue444": {"passed": True, "cues": len(stamps), "first_ms": stamps[0][0], "last_ms": stamps[-1][1]},
    "issue446": {"passed": True, "output": p.stdout.strip()[:200]},
}
(WORK / "issues444-446-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
kh.export_ccache_tar()
kh.step("script.end", passed=True)
