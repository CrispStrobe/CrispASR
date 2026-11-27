#!/usr/bin/env python3
"""#609 follow-up: bt2-tts honours per-request sampling knobs on the server.

Before the fix the adapter applied temperature once at init and the native
sampler read top-k/top-p/repetition penalty from the checkpoint only, so every
request below produced the same audio for the same seed. Pass criteria:

  same      explicit model defaults (0.9/50/1.0/1.1) == no knobs   (defaults unchanged)
  same      no knobs after a temperature-0.3 request == first      (not sticky)
  differs   temperature 0.3 != no knobs
  differs   top_k 1 != no knobs
  differs   repetition_penalty 1.5 != no knobs

On a commit without the fix the three "differs" rows fail; that is the
negative control.

  gh workflow run heavy-cpu.yml --ref <sha> -f script=tools/ci-heavy/bt2_sampling_knobs.py -f timeout_minutes=120
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sweep_controls_common as C  # noqa: E402

NT = str(os.cpu_count() or 4)
TEXT = "The village of Chamounix lay at the foot of the mountain."
PORT = 18093


def start(b, breeze, codec):
    log = open(C.OUT / "server.log", "w")
    voices = C.OUT / "voices"
    voices.mkdir(exist_ok=True)
    p = subprocess.Popen(
        [str(b), "--server", "--backend", "bt2-tts", "-m", str(breeze), "--codec-model", str(codec), "--host",
         "127.0.0.1", "--port", str(PORT), "--voice-dir", str(voices), "--accept-license", "other",
         "--i-have-rights", "--speaker-identity", "synthetic", "--no-watermark", "--accept-marking-responsibility",
         "--no-spoken-disclaimer", "-t", NT, "--no-gpu"],
        stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    while time.time() - t0 < 900:
        if p.poll() is not None:
            raise SystemExit(f"server exited rc={p.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    return p
        except Exception:
            pass
        time.sleep(3)
    raise SystemExit("server not healthy")


def speech(name, extra):
    body = {"model": "bt2-tts", "input": TEXT, "seed": 7, "response_format": "wav", **extra}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/audio/speech", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        data = r.read()
    (C.OUT / f"{name}.wav").write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def main():
    head = subprocess.run(["git", "-C", str(C.ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    b = C.build_crispasr()
    breeze = C.hf("cstr/breeze-tts-2-GGUF", "breeze-tts-2-q4_k.gguf", None)
    codec = C.hf("cstr/qwen3-tts-tokenizer-12hz-GGUF", "qwen3-tts-tokenizer-12hz.gguf", None)
    srv = start(b, breeze, codec)
    try:
        h = {}
        h["default"] = speech("default", {})
        h["temp03"] = speech("temp03", {"temperature": 0.3})
        h["default_after"] = speech("default_after", {})
        h["topk1"] = speech("topk1", {"top_k": 1})
        h["rep15"] = speech("rep15", {"repetition_penalty": 1.5})
        h["explicit_defaults"] = speech("explicit_defaults",
                                        {"temperature": 0.9, "top_k": 50, "top_p": 1.0, "repetition_penalty": 1.1})
    finally:
        srv.terminate()
        srv.wait(60)
    checks = {
        "explicit defaults == default": h["explicit_defaults"] == h["default"],
        "default after temp 0.3 == default": h["default_after"] == h["default"],
        "temperature 0.3 differs": h["temp03"] != h["default"],
        "top_k 1 differs": h["topk1"] != h["default"],
        "repetition_penalty 1.5 differs": h["rep15"] != h["default"],
    }
    ok = all(checks.values())
    C.save({"head": head, "sha256": h, "checks": checks, "ok": ok})
    lines = [f"## bt2-tts per-request sampling knobs @ {head[:9]}", "", "| check | pass |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in checks.items()]
    lines += ["", "| request | sha256 |", "|---|---|"] + [f"| {k} | {v[:16]} |" for k, v in h.items()]
    lines.append(f"\noverall: {'PASS' if ok else 'FAIL'}")
    C.summary(lines)
    if not ok:
        raise SystemExit("sampling knob check failed")


if __name__ == "__main__":
    main()
