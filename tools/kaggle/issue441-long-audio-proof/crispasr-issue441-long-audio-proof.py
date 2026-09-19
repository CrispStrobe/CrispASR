"""Issue #441: old/new allocation A/B plus exact 47.5-minute CPU proof."""

import json
import os
import re
import resource
import selectors
import signal
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp")
REPO = WORK / "CrispASR"
CURRENT_BUILD = TEMP / "build-issue441-current"
OLD_BUILD = TEMP / "build-issue441-v0833"
MODELS = TEMP / "models"
REF = os.environ.get("CRISPASR_REF", "fix/441-proof")
N_SAMPLES = 45_602_304
LIMIT_BYTES = 12 * 1024**3


def sh(cmd: str) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh(f"git clone --depth 1 --branch {REF} --recursive https://github.com/CrispStrobe/CrispASR {REPO}")
sys.path.insert(0, str(REPO / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
kh.resolve_hf_token()
sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
kh.step("script.start", ref=REF, sha=sha)
kh.install_build_toolchain()
sh("apt-get update -qq && apt-get install -y -qq strace time")


def build(source: Path, build_dir: Path, label: str) -> Path:
    flags = " ".join(kh.cuda_build_flags(kh.detect_cuda_arch()))
    cache = " ".join(kh.cache_and_link_flags())
    with kh.build_heartbeat(f"{label}.configure"):
        kh.sh_with_progress(
            f"cmake {source} -B{build_dir} -GNinja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON {flags} {cache}"
        )
    with kh.build_heartbeat(f"{label}.build"):
        kh.sh_with_progress(
            f"stdbuf -oL -eL cmake --build {build_dir} --target crispasr-cli -- -j{kh.safe_build_jobs(gpu=True)}"
        )
    binary = build_dir / "bin" / "crispasr"
    kh.step(f"{label}.build.done", binary_bytes=binary.stat().st_size)
    return binary


CURRENT_BIN = build(REPO, CURRENT_BUILD, "current")
sh(f"git -C {REPO} fetch --depth 1 origin tag v0.8.33")
sh(f"git -C {REPO} checkout --detach v0.8.33")
sh(f"git -C {REPO} submodule update --init --recursive")
OLD_BIN = build(REPO, OLD_BUILD, "old")

MODELS.mkdir(exist_ok=True)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
kh.sh_with_progress("pip install -q huggingface_hub hf_transfer")
from huggingface_hub import hf_hub_download  # noqa: E402

model = Path(
    hf_hub_download(
        repo_id="cstr/parakeet-tdt-0.6b-v3-GGUF",
        filename="parakeet-tdt-0.6b-v3-q4_k.gguf",
        local_dir=str(MODELS),
        local_dir_use_symlinks=False,
    )
)
kh.step("download.done", model=model.name, bytes=model.stat().st_size)


def make_clip() -> Path:
    src = REPO / "samples" / "jfk.wav"
    out = TEMP / "issue441-2850s.wav"
    with wave.open(str(src), "rb") as r:
        assert r.getnchannels() == 1 and r.getsampwidth() == 2 and r.getframerate() == 16000
        frames = r.readframes(r.getnframes())
    frame_bytes = 2
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(frame_bytes)
        w.setframerate(16000)
        remaining = N_SAMPLES
        src_samples = len(frames) // frame_bytes
        while remaining:
            take = min(remaining, src_samples)
            w.writeframesraw(frames[: take * frame_bytes])
            remaining -= take
    with wave.open(str(out), "rb") as r:
        assert r.getnframes() == N_SAMPLES
    kh.step("clip.done", samples=N_SAMPLES, seconds=round(N_SAMPLES / 16000, 4), bytes=out.stat().st_size)
    return out


clip = make_clip()
# The seeded source path is temporary. Keep only final evidence in
# /kaggle/working so output retrieval never walks a checkout or build tree.
shutil.rmtree(REPO)


def parse_last_srt(path: Path) -> float:
    if not path.exists():
        return 0.0
    stamps = re.findall(r"-->\s*(\d+):(\d+):(\d+),(\d+)", path.read_text(errors="replace"))
    if not stamps:
        return 0.0
    h, m, s, ms = map(int, stamps[-1])
    return h * 3600 + m * 60 + s + ms / 1000.0


def full_run(binary: Path, label: str, rep: int, trace: bool) -> dict:
    stem = WORK / f"{label}-r{rep}"
    trace_path = WORK / f"{label}-r{rep}.strace"
    cmd = []
    if trace:
        cmd += ["strace", "-f", "-qq", "-e", "trace=mmap,mremap,brk", "-o", str(trace_path)]
    cmd += [
        "/usr/bin/time", "-v", str(binary), "--backend", "parakeet", "-m", str(model), "-f", str(clip),
        "-l", "en", "-sp", "-osrt", "-of", str(stem), "-t", "4", "--chunk-seconds", "419", "-v", "--no-gpu",
    ]
    env = {**os.environ, "CRISPASR_PARAKEET_AVAILABLE_MB": "12000"}
    t0 = time.time()
    p = subprocess.Popen(
        cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True
    )
    while True:
        try:
            log, _ = p.communicate(timeout=60)
            break
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            kh.step(f"run.{label}.r{rep}.heartbeat", elapsed_s=round(elapsed, 1))
            if elapsed >= 5400:
                os.killpg(p.pid, signal.SIGKILL)
                log, _ = p.communicate()
                break
    (WORK / f"{label}-r{rep}.log").write_text(log)
    rss = re.findall(r"Maximum resident set size \(kbytes\):\s*(\d+)", log)
    mmap_max = 0
    if trace_path.exists():
        for n in re.findall(r"mmap\([^,\n]*,\s*(\d+),", trace_path.read_text(errors="replace")):
            mmap_max = max(mmap_max, int(n))
    row = {
        "label": label,
        "rep": rep,
        "rc": p.returncode,
        "wall_s": round(time.time() - t0, 2),
        "max_rss_kib": int(rss[-1]) if rss else -1,
        "max_mmap_bytes": mmap_max,
        "last_srt_s": round(parse_last_srt(stem.with_suffix(".srt")), 3),
        "bounded_route": "route=chunk-segmented" in log,
        "refused_graph": "refusing encoder graph" in log,
    }
    kh.step(f"run.{label}.r{rep}", **row)
    print(json.dumps(row, sort_keys=True), flush=True)
    return row


def set_limit() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (LIMIT_BYTES, LIMIT_BYTES))


def unsafe_probe(binary: Path, label: str) -> dict:
    """Force the unchunked route under 12 GiB; stop after refusal/failure."""
    trace_path = WORK / f"probe-{label}.strace"
    cmd = [
        "strace", "-f", "-qq", "-e", "trace=mmap,mremap,brk", "-o", str(trace_path), str(binary),
        "--backend", "parakeet", "-m", str(model), "-f", str(clip), "-l", "en", "-sp", "-osrt", "-of",
        str(WORK / f"probe-{label}"), "-t", "4", "--chunk-seconds", "0", "-v", "--no-gpu",
    ]
    env = {
        **os.environ,
        "CRISPASR_PARAKEET_AVAILABLE_MB": "12000",
        # Put both revisions on the unsafe single-pass route. A zero explicit
        # chunk size alone still selects LONGFORM for a 47.5-minute clip and
        # would make this negative control test another bounded path.
        "CRISPASR_PARAKEET_STREAM_THRESHOLD": "99999",
        "CRISPASR_PARAKEET_LONGFORM": "0",
        "CRISPASR_PARAKEET_MEM_POLICY": "off",
    }
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         preexec_fn=set_limit, bufsize=1)
    lines = []
    deadline = time.time() + 600
    marker = False
    assert p.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(p.stdout, selectors.EVENT_READ)
    while time.time() < deadline:
        if not selector.select(timeout=min(1.0, max(0.0, deadline - time.time()))):
            if p.poll() is not None:
                break
            continue
        line = p.stdout.readline()
        if not line and p.poll() is not None:
            break
        lines.append(line)
        if "refusing encoder graph" in line:
            marker = True
            break
    selector.close()
    if p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            p.kill(); p.wait()
    log = "".join(lines)
    (WORK / f"probe-{label}.log").write_text(log)
    mmap_max = 0
    if trace_path.exists():
        for n in re.findall(r"mmap\([^,\n]*,\s*(\d+),", trace_path.read_text(errors="replace")):
            mmap_max = max(mmap_max, int(n))
    row = {"label": label, "rc": p.returncode, "refused_graph": marker, "max_mmap_bytes": mmap_max}
    kh.step(f"probe.{label}", **row)
    print(json.dumps(row, sort_keys=True), flush=True)
    return row


# Negative control first: v0.8.33 must expose the old unsafe request/failure;
# current must refuse at the allocation boundary even with routing policy off.
old_probe = unsafe_probe(OLD_BIN, "old")
new_probe = unsafe_probe(CURRENT_BIN, "current")

# The forced single-pass A/B above makes the intermittent allocator failure
# deterministic. One full run is therefore enough to prove that the reporter's
# exact bounded command completes; repeating it only repeats CPU inference.
new_exact = [full_run(CURRENT_BIN, "current-exact", 1, trace=True)]

passes = all(
    r["rc"] == 0 and r["last_srt_s"] >= 2700 and r["max_rss_kib"] > 0 and r["max_rss_kib"] < 8 * 1024 * 1024
    and r["bounded_route"]
    for r in new_exact
)
passes = passes and new_probe["refused_graph"] and not old_probe["refused_graph"]
# ggml reserves an 8 GiB virtual arena for this bounded graph (the largest
# individual mmap is a few pages above 8 GiB once allocator bookkeeping is
# included), while resident memory remains independently capped above.  The
# regression signature is the old 123.6 GB request, so keep the virtual-map
# ceiling at the same 12 GiB budget supplied to the policy/guard.
passes = passes and new_exact[0]["max_mmap_bytes"] < 12 * 1024**3

summary = {
    "sha": sha,
    "passed": passes,
    "old_probe": old_probe,
    "new_probe": new_probe,
    "new_exact": new_exact,
}
(WORK / "issue441-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
kh.step("result", passed=passes)
kh.export_ccache_tar()
kh._push_progress_to_hf(force=True)
assert passes, json.dumps(summary, indent=2)
kh.step("script.end", passed=True)
