"""Shared helpers for the 2026-11-19 extended-sweep control arms.

Imported by sweep_ja_nemo_control.py, sweep_omniasr_control.py and
sweep_lahgtna_control.py (each is a separate heavy-cpu.yml dispatch, so the
upstream Python stacks never share an interpreter). Not a job on its own.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("HEAVY_OUT", ROOT / "heavy-out"))
SCRATCH = Path(os.environ.get("HEAVY_SCRATCH", ROOT / "heavy-scratch"))
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH.mkdir(parents=True, exist_ok=True)


def run(cmd, log=None, check=True, env=None, timeout=None):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env=env, timeout=timeout)
    if log:
        Path(log).write_text(f"$ {' '.join(str(c) for c in cmd)}\nrc={r.returncode}\n--stdout--\n{r.stdout}\n--stderr--\n{r.stderr}")
    if check and r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
        raise SystemExit(f"command failed rc={r.returncode}: {cmd[0]}")
    return r


def build_crispasr() -> Path:
    bd = SCRATCH / "build"
    run(["cmake", "-S", ROOT, "-B", bd, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF",
         "-DBUILD_SHARED_LIBS=OFF"], OUT / "cmake.log")
    run(["cmake", "--build", bd, "-j", str(os.cpu_count() or 4), "--target", "crispasr"], OUT / "build.log")
    b = bd / "bin" / "crispasr"
    assert b.is_file(), b
    return b


def hf(repo, file, revision):
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, file, revision=revision, cache_dir=str(SCRATCH / "hf")))


def cli(bin_, args, log, env=None, timeout=3600):
    e = dict(os.environ)
    if env:
        e.update(env)
    r = run([bin_] + list(args), log=log, check=False, env=e, timeout=timeout)
    lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    return {"rc": r.returncode, "text": lines[-1] if lines else "", "stderr_tail": r.stderr[-1500:]}


def _lev(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a[i - 1] != b[j - 1]))
    return d[len(b)]


def _norm_chars(t):
    return [c for c in unicodedata.normalize("NFKC", t or "").lower() if c.isalnum()]


def cer(ref, hyp):
    r = _norm_chars(ref)
    return round(_lev(r, _norm_chars(hyp)) / max(1, len(r)), 4)


def wer(ref, hyp):
    def w(t):
        return "".join(c if (c.isalnum() or c.isspace()) else " " for c in unicodedata.normalize("NFKC", t or "").lower()).split()
    r = w(ref)
    return round(_lev(r, w(hyp)) / max(1, len(r)), 4)


def save(res, name="result.json"):
    (OUT / name).write_text(json.dumps(res, indent=1, ensure_ascii=False))


def summary(lines):
    with open(OUT / "summary.md", "a") as f:
        f.write("\n".join(lines) + "\n")


def venv(name, pkgs, extra=()):
    """A separate interpreter for an upstream stack (never this one)."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "uv"])
    v = SCRATCH / f"venv-{name}"
    subprocess.check_call([sys.executable, "-m", "uv", "venv", "--python", "3.11", str(v)])
    py = v / "bin" / "python"
    r = subprocess.run([sys.executable, "-m", "uv", "pip", "install", "--python", str(py)] + list(pkgs) + list(extra),
                       capture_output=True, text=True, timeout=3600)
    (OUT / f"venv-{name}.log").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"venv {name} install failed; see venv-{name}.log")
    return py
