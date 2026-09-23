#!/usr/bin/env python3
"""Drive the Qwen3 realtime session (#445, Confucius4-R2T2 streaming) through
the real server and dump its per-call trace.

Starts `crispasr --server --ws-port P` with CRISPASR_QWEN3_STREAM=1 and
CRISPASR_QWEN3_STREAM_TRACE=1, connects to ws://127.0.0.1:P+1/v1/realtime,
appends the WAV in the reference driver's chunk sizes (step + lookahead first,
then step), commits, and writes {"calls": [...], "completed": "..."} where each
call is one QWEN3_STREAM trace line — the same fields
tools/kaggle/r2t2-stream-ref logs from the upstream Python.

  python tools/qwen3_stream_client.py -m r2t2-q4_k.gguf -f samples/jfk.wav -o trace.json
"""
import argparse
import base64
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location("rt", os.path.join(ROOT, "tests", "test-server-realtime-api.py"))
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("-f", "--file", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--bin", default=os.path.join(ROOT, "build", "bin", "crispasr"))
    ap.add_argument("--port", type=int, default=48300)
    ap.add_argument("--step-ms", type=int, default=160)
    ap.add_argument("--lookahead-ms", type=int, default=160)
    ap.add_argument("-t", "--threads", type=int, default=4)
    a = ap.parse_args()

    pcm, sr = rt.load_pcm_16_bytes(a.file)  # little-endian int16 bytes
    assert sr == 16000, f"{a.file}: need 16 kHz, got {sr}"
    env = dict(os.environ, CRISPASR_QWEN3_STREAM="1")
    env.setdefault("CRISPASR_QWEN3_STREAM_TRACE", "1")
    log_path = a.out + ".server.log"
    log = open(log_path, "w")
    proc = subprocess.Popen([a.bin, "--server", "-m", a.model, "--backend", "qwen3", "--host", "127.0.0.1",
                             "--port", str(a.port), "--ws-port", str(a.port + 1), "-t", str(a.threads)],
                            stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        for _ in range(600):
            try:
                socket.create_connection(("127.0.0.1", a.port + 2), timeout=1).close()
                break
            except OSError:
                time.sleep(1)
        s = socket.create_connection(("127.0.0.1", a.port + 2), timeout=30)
        s.sendall(("GET /v1/realtime HTTP/1.1\r\nHost: 127.0.0.1\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                   "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        hdr = b""
        while b"\r\n\r\n" not in hdr:
            hdr += s.recv(4096)
        step = a.step_ms * 16 * 2
        first = (a.step_ms + a.lookahead_ms) * 16 * 2
        pos, sizes = 0, []
        while pos < len(pcm):
            n = first if pos == 0 else step
            chunk = pcm[pos:pos + n]
            pos += len(chunk)
            sizes.append(len(chunk) // 2)
            msg = json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(chunk).decode()})
            s.sendall(rt.ws_frame(msg, opcode=0x1))
        s.sendall(rt.ws_frame(json.dumps({"type": "input_audio_buffer.commit"}), opcode=0x1))
        completed, deltas = None, []
        deadline = time.time() + 1800
        while completed is None and time.time() < deadline:
            for m in rt.read_server_frames(s, timeout=10.0):
                try:
                    j = json.loads(m)
                except Exception:
                    continue
                if j.get("type") == "conversation.item.input_audio_transcription.delta":
                    deltas.append(j.get("delta", ""))
                elif j.get("type") == "conversation.item.input_audio_transcription.completed":
                    completed = j.get("transcript", j.get("text"))
        s.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    calls = []
    for line in open(log_path, encoding="utf-8", errors="replace"):
        if line.startswith("QWEN3_STREAM "):
            calls.append(json.loads(line[len("QWEN3_STREAM "):]))
    json.dump({"calls": calls, "completed": completed, "deltas": deltas, "append_sizes": sizes},
              open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"{len(calls)} calls; completed: {completed}")


if __name__ == "__main__":
    sys.exit(main())
