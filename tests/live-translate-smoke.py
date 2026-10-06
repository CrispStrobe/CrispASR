#!/usr/bin/env python3
"""End-to-end smoke test of `--stream` live transcription + translation.

Written for the platforms where that path is compiled but otherwise never
executed (Windows: binary stdin, the PeekNamedPipe catch-up probe, UTF-8
argv). It runs the real CLI on real models and checks what comes out:

  1. text -> text through the Opus-MT backend, with non-ASCII input on the
     command line;
  2. raw PCM on stdin from a FILE (redirect), English -> German;
  3. the same PCM through a PIPE, written in real time, with
     --stream-realtime (the catch-up probe only does anything on a pipe).

    live-translate-smoke.py --crispasr build/bin/crispasr --models DIR \
        [--wav samples/jfk.wav]

DIR must hold parakeet-tdt-0.6b-v3-q4_k.gguf, opus-mt-en-de-q8_0.gguf,
opus-mt-de-en-q8_0.gguf and ggml-silero-v6.2.0.bin.
"""
import argparse
import json
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path


def fail(msg: str, detail: str = "") -> None:
    print(f"FAIL: {msg}")
    if detail:
        print(detail[-4000:])
    sys.exit(1)


def events(stdout: bytes) -> list:
    out = []
    for line in stdout.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def check_stream(name: str, stdout: bytes, stderr: bytes, rc: int) -> None:
    err = stderr.decode("utf-8", errors="replace")
    if rc != 0:
        fail(f"{name}: exit code {rc}", err)
    tr = [e for e in events(stdout) if e.get("type") == "translation"]
    if not tr:
        fail(f"{name}: no translation event", stdout.decode("utf-8", errors="replace") + "\n" + err)
    src = " ".join(e.get("text", "") for e in tr).lower()
    dst = " ".join(e.get("translation", "") for e in tr)
    print(f"  {name}: {len(tr)} sentence(s)")
    for e in tr:
        print(f"    {e.get('text')!r} -> {e.get('translation')!r}  (mt {e.get('mt_ms')} ms)")
    # "...ask not what your country can do for you..." — the recogniser must
    # have heard the clip and the translator must have produced German.
    if "country" not in src:
        fail(f"{name}: transcript does not contain 'country': {src!r}")
    if "land" not in dst.lower():
        fail(f"{name}: translation does not contain 'Land': {dst!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crispasr", required=True)
    ap.add_argument("--models", required=True, type=Path)
    ap.add_argument("--wav", default="samples/jfk.wav")
    args = ap.parse_args()

    m = args.models
    need = ["parakeet-tdt-0.6b-v3-q4_k.gguf", "opus-mt-en-de-q8_0.gguf", "opus-mt-de-en-q8_0.gguf",
            "ggml-silero-v6.2.0.bin"]
    for f in need:
        if not (m / f).exists():
            fail(f"missing model {m / f}")

    # 1. Non-ASCII text on the command line.
    text = "Können Sie mir bitte sagen, wie spät es ist?"
    r = subprocess.run([args.crispasr, "--backend", "marian", "-m", str(m / "opus-mt-de-en-q8_0.gguf"), "-sl", "de",
                        "-tl", "en", "-bs", "1", "-np", "--text", text], capture_output=True, timeout=600)
    out = r.stdout.decode("utf-8", errors="replace").strip()
    print(f"  text: {text!r} -> {out!r}")
    if r.returncode != 0 or "time" not in out.lower():
        fail("text translation with non-ASCII argv", out + "\n" + r.stderr.decode("utf-8", errors="replace"))

    with wave.open(args.wav) as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (16000, 1, 2):
            fail(f"{args.wav} is not 16 kHz mono s16")
        pcm = w.readframes(w.getnframes())
    # Trailing silence so the last utterance is closed by the VAD, not by EOF.
    pcm += b"\x00" * (16000 * 2 * 2)
    raw = m / "smoke-input.s16le"
    raw.write_bytes(pcm)

    cmd = [args.crispasr, "--stream", "--stream-json", "-m", str(m / "parakeet-tdt-0.6b-v3-q4_k.gguf"), "--backend",
           "parakeet", "-l", "en", "--tr-tl", "de", "--translate-model", str(m / "opus-mt-en-de-q8_0.gguf"),
           "--vad-model", str(m / "ggml-silero-v6.2.0.bin")]

    # 2. stdin redirected from a file.
    with open(raw, "rb") as f:
        r = subprocess.run(cmd, stdin=f, capture_output=True, timeout=1200)
    check_stream("stdin from file", r.stdout, r.stderr, r.returncode)

    # 3. stdin as a pipe, fed in real time.
    p = subprocess.Popen(cmd + ["--stream-realtime"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)

    def feed() -> None:
        chunk = 16000 * 2 // 10  # 100 ms
        try:
            for i in range(0, len(pcm), chunk):
                p.stdin.write(pcm[i:i + chunk])
                p.stdin.flush()
                time.sleep(0.1)
            p.stdin.close()
        except (BrokenPipeError, OSError):
            pass

    t = threading.Thread(target=feed, daemon=True)
    t.start()
    try:
        stdout, stderr = b"", b""
        # communicate() would close stdin under the feeder; read the pipes directly.
        errbuf = []
        et = threading.Thread(target=lambda: errbuf.append(p.stderr.read()), daemon=True)
        et.start()
        stdout = p.stdout.read()
        rc = p.wait(timeout=1200)
        et.join(timeout=30)
        stderr = errbuf[0] if errbuf else b""
    except subprocess.TimeoutExpired:
        p.kill()
        fail("pipe run timed out")
    check_stream("stdin from pipe, real time", stdout, stderr, rc)

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
