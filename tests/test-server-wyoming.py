#!/usr/bin/env python
"""test-server-wyoming.py — integration test for the --wyoming-port Wyoming
protocol TCP server (issue #172).

Boots `crispasr --server --wyoming-port N` with a small whisper model, then
exercises the Wyoming peer-to-peer JSONL protocol over raw TCP:

  1. describe → info: verifies asr array present and type="info"
  2. ASR round-trip: streams jfk.wav as int16 PCM, asserts non-empty transcript
  3. (optional) disconnect mid-stream: server must not crash and must accept
     a subsequent connection

SKIPs (exit 0) if no whisper GGUF or binary found. No third-party deps.

Usage:
  python tests/test-server-wyoming.py [--cache-dir DIR]
  CRISPASR_TEST_CACHE=/path/to/models python tests/test-server-wyoming.py
"""
import json
import os
import socket
import subprocess
import sys
import time
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_binary():
    for rel in ["build/bin/crispasr", "build/bin/Release/crispasr.exe",
                "build-ninja-compile/bin/crispasr", "bin/crispasr", "bin/crispasr.exe"]:
        p = os.path.join(ROOT, rel)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def find_whisper(cache_dir):
    cands = []
    for d in [cache_dir,
              os.environ.get("CRISPASR_TEST_CACHE"),
              os.environ.get("CRISPASR_MODELS_DIR"),
              os.path.expanduser("~/.cache/crispasr")]:
        if d and os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.startswith("ggml-") and f.endswith(".bin"):
                    cands.append(os.path.join(d, f))
    return cands[0] if cands else None


def load_pcm_s16(path):
    """Return (raw_bytes, sample_rate, channels) for a WAV file."""
    with wave.open(path, "rb") as w:
        n, sr, ch = w.getnframes(), w.getframerate(), w.getnchannels()
        raw = w.readframes(n)
    return raw, sr, ch


def wyoming_send(sock, msg_type, data, payload=b""):
    # Frame exactly like Home Assistant's official `wyoming` library: non-empty
    # `data` is sent as a SEPARATE length-prefixed JSON blob (data_length) after
    # the header line — NOT inline. Sending it inline (as the old test did) hides
    # the data_length parsing path and let issue #172's transcribe desync slip
    # through. (regression guard)
    hdr = {"type": msg_type}
    data_bytes = b""
    if data:
        data_bytes = json.dumps(data, ensure_ascii=False).encode("utf-8")
        hdr["data_length"] = len(data_bytes)
    if payload:
        hdr["payload_length"] = len(payload)
    sock.sendall((json.dumps(hdr) + "\n").encode())
    if data_bytes:
        sock.sendall(data_bytes)
    if payload:
        sock.sendall(payload)


def wyoming_recv(sock, timeout=30.0):
    """Read one Wyoming message (header + optional payload). Returns (type, data, payload)."""
    sock.settimeout(timeout)
    buf = b""
    while True:
        c = sock.recv(1)
        if not c:
            return None, None, b""
        if c == b"\n":
            break
        buf += c
    hdr = json.loads(buf)
    payload_len = hdr.get("payload_length", 0)
    payload = b""
    if payload_len > 0:
        while len(payload) < payload_len:
            chunk = sock.recv(payload_len - len(payload))
            if not chunk:
                break
            payload += chunk
    return hdr.get("type"), hdr.get("data", {}), payload


def wait_tcp(host, port, seconds=120):
    for _ in range(seconds):
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(1)
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    http_port    = 10396
    wyoming_port = 10397
    cache_dir    = os.environ.get("CRISPASR_TEST_CACHE", "")

    for i, arg in enumerate(sys.argv[1:], 1):
        if arg in ("--cache-dir",) and i < len(sys.argv):
            cache_dir = sys.argv[i]
        elif arg.startswith("--cache-dir="):
            cache_dir = arg.split("=", 1)[1]

    binary = find_binary()
    if not binary:
        print("SKIP: crispasr binary not found")
        return 0
    model = find_whisper(cache_dir)
    if not model:
        print("SKIP: no whisper ggml-*.bin found "
              "(set CRISPASR_TEST_CACHE or CRISPASR_MODELS_DIR)")
        return 0
    sample = os.path.join(ROOT, "samples/jfk.wav")
    if not os.path.isfile(sample):
        print("SKIP: samples/jfk.wav not found")
        return 0

    cmd = [binary,
           "--server", "-m", model, "--backend", "whisper",
           "--host", "127.0.0.1", "--port", str(http_port),
           "--wyoming-port", str(wyoming_port),
           "--no-prints"]
    if cache_dir:
        cmd += ["--cache-dir", cache_dir]

    print(f"Model: {model}")
    print(f"Wyoming port: {wyoming_port}")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    passed = failed = 0

    try:
        # Wait for HTTP server (model load can take up to 2 min for large models)
        if not wait_tcp("127.0.0.1", http_port, seconds=120):
            print("ERROR: server did not start within 120 s")
            return 2
        time.sleep(0.5)  # let Wyoming listener bind too

        # ── Test 1: describe → info ──────────────────────────────────────────
        print("\n[1] describe → info")
        try:
            s = socket.create_connection(("127.0.0.1", wyoming_port), timeout=5)
            wyoming_send(s, "describe", {})
            msg_type, data, _ = wyoming_recv(s, timeout=10.0)
            s.close()
            if msg_type == "info" and isinstance(data.get("asr"), list) and data["asr"]:
                name = data["asr"][0].get("name", "?")
                langs = data["asr"][0].get("languages", [])
                print(f"  ✓ type=info, asr.name={name!r}, languages={langs}")
                passed += 1
            else:
                print(f"  ✗ got type={msg_type!r}, data keys={list(data.keys()) if data else None}")
                failed += 1
        except Exception as exc:
            print(f"  ✗ exception: {exc}")
            failed += 1

        # ── Test 2: info message has required keys ───────────────────────────
        print("\n[2] info structure")
        try:
            s = socket.create_connection(("127.0.0.1", wyoming_port), timeout=5)
            wyoming_send(s, "describe", {})
            msg_type, data, _ = wyoming_recv(s, timeout=10.0)
            s.close()
            has_asr = isinstance(data.get("asr"), list)
            has_tts = "tts" in data
            if msg_type == "info" and has_asr and has_tts:
                print(f"  ✓ info has both asr and tts keys")
                passed += 1
            else:
                print(f"  ✗ missing keys: asr={has_asr} tts={has_tts}")
                failed += 1
        except Exception as exc:
            print(f"  ✗ exception: {exc}")
            failed += 1

        # ── Test 3: ASR transcription (jfk.wav) ─────────────────────────────
        print("\n[3] ASR round-trip (jfk.wav → transcript)")
        try:
            pcm_raw, pcm_rate, pcm_ch = load_pcm_s16(sample)
            s = socket.create_connection(("127.0.0.1", wyoming_port), timeout=5)
            s.settimeout(120)
            wyoming_send(s, "transcribe", {"name": "whisper", "language": "en"})
            wyoming_send(s, "audio-start", {"rate": pcm_rate, "width": 2, "channels": pcm_ch})
            chunk_sz = 4096 * 2 * pcm_ch
            for i in range(0, len(pcm_raw), chunk_sz):
                wyoming_send(s, "audio-chunk",
                             {"rate": pcm_rate, "width": 2, "channels": pcm_ch},
                             pcm_raw[i:i + chunk_sz])
            wyoming_send(s, "audio-stop", {})
            msg_type, data, _ = wyoming_recv(s, timeout=120.0)
            s.close()
            if msg_type == "transcript" and isinstance(data.get("text"), str) and data["text"].strip():
                print(f"  ✓ transcript: {data['text'].strip()[:80]!r}")
                passed += 1
            else:
                print(f"  ✗ got type={msg_type!r} text={data.get('text')!r}")
                failed += 1
        except Exception as exc:
            print(f"  ✗ exception: {exc}")
            failed += 1

        # ── Test 4: disconnect before audio-stop → server survives ───────────
        print("\n[4] mid-stream disconnect resilience")
        try:
            pcm_raw, pcm_rate, pcm_ch = load_pcm_s16(sample)
            s = socket.create_connection(("127.0.0.1", wyoming_port), timeout=5)
            wyoming_send(s, "transcribe", {"name": "whisper", "language": "en"})
            wyoming_send(s, "audio-start", {"rate": pcm_rate, "width": 2, "channels": pcm_ch})
            # Send only the first chunk, then hard-close
            wyoming_send(s, "audio-chunk",
                         {"rate": pcm_rate, "width": 2, "channels": pcm_ch},
                         pcm_raw[:4096])
            s.close()
            time.sleep(0.5)

            # Server must still accept the next connection
            s2 = socket.create_connection(("127.0.0.1", wyoming_port), timeout=5)
            wyoming_send(s2, "describe", {})
            msg_type2, _, _ = wyoming_recv(s2, timeout=10.0)
            s2.close()
            if msg_type2 == "info":
                print("  ✓ server still responsive after client disconnect")
                passed += 1
            else:
                print(f"  ✗ server returned {msg_type2!r} after disconnect recovery")
                failed += 1
        except Exception as exc:
            print(f"  ✗ exception: {exc}")
            failed += 1

        # ── Test 5: unknown message type → server stays alive ─────────────────
        print("\n[5] unknown message type resilience")
        try:
            s = socket.create_connection(("127.0.0.1", wyoming_port), timeout=5)
            wyoming_send(s, "nonexistent-event", {"foo": "bar"})
            s.close()
            time.sleep(0.3)

            s2 = socket.create_connection(("127.0.0.1", wyoming_port), timeout=5)
            wyoming_send(s2, "describe", {})
            msg_type2, _, _ = wyoming_recv(s2, timeout=10.0)
            s2.close()
            if msg_type2 == "info":
                print("  ✓ server still responsive after unknown event")
                passed += 1
            else:
                print(f"  ✗ server returned {msg_type2!r} after unknown-event test")
                failed += 1
        except Exception as exc:
            print(f"  ✗ exception: {exc}")
            failed += 1

        print(f"\nRESULT: {passed} passed, {failed} failed")
        return 0 if failed == 0 else 1

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
