#!/usr/bin/env python3
"""e2e: chunked (Transfer-Encoding: chunked) uploads to the HTTP server.

chunked bodies must work like any OpenAI SDK streaming client
(requests with fs.createReadStream / http.client chunked bodies) and
must still be bounded by the 512 MB payload cap:

  1. chunked multipart upload to /v1/audio/transcriptions -> 200 + text
  2. chunked upload past the 512 MB cap -> 413, server stays alive
  3. Content-Length upload still works (regression)
  4. oversized Content-Length still rejected before the body (regression)

run: python3 tests/test-server-chunked-upload.py
env: CRISPASR_TEST_MODEL  model file (skips when absent and not findable)
     CRISPASR_TEST_CACHE  cache dir scanned for ggml-*.bin
"""

import os
import socket
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOUNDARY = "crispasrchunkedboundary"
CAP = 512 * 1024 * 1024  # server-side set_payload_max_length


def find_binary():
    for c in ["build/bin/crispasr", "build-ninja-compile/bin/crispasr", "bin/crispasr"]:
        p = os.path.join(ROOT, c)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def find_model():
    env = os.environ.get("CRISPASR_TEST_MODEL")
    if env and os.path.isfile(env):
        return env
    for d in [
        os.environ.get("CRISPASR_TEST_CACHE"),
        os.path.expanduser("~/.cache/crispasr"),
    ]:
        if d and os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.startswith("ggml-") and f.endswith(".bin"):
                    return os.path.join(d, f)
    return None


def multipart_body(audio, content_type="audio/wav"):
    head = (
        '--%s\r\nContent-Disposition: form-data; name="file"; '
        'filename="jfk.wav"\r\nContent-Type: %s\r\n\r\n' % (BOUNDARY, content_type)
    ).encode()
    tail = ("\r\n--%s--\r\n" % BOUNDARY).encode()
    return head + audio + tail


def send_raw(port, head_lines, body_chunks):
    """open a connection, send headers + an iterable of body chunks, read the reply."""
    s = socket.create_connection(("127.0.0.1", port), timeout=30)
    s.sendall(("\r\n".join(head_lines) + "\r\n\r\n").encode())
    broken = False
    try:
        for chunk in body_chunks:
            if chunk:
                s.sendall(("%x\r\n" % len(chunk)).encode() + chunk + b"\r\n")
        s.sendall(b"0\r\n\r\n")
    except (BrokenPipeError, ConnectionResetError):
        broken = True
    resp = b""
    try:
        while b"\r\n\r\n" not in resp or not resp.endswith(b"\r\n\r\n"):
            b = s.recv(65536)
            if not b:
                break
            resp += b
            # stop after headers + a short body; these responses are small
            if len(resp) > 4096:
                break
    except OSError:
        pass
    s.close()
    return resp.decode("latin-1"), broken


def send_cl(port, method, path, content_length, body=b""):
    """request with an explicit Content-Length; body is a bytes object or an
    iterable of chunks totaling at most content_length."""
    s = socket.create_connection(("127.0.0.1", port), timeout=60)
    s.sendall(
        (
            "%s %s HTTP/1.1\r\nHost: 127.0.0.1:%d\r\n"
            "Content-Type: multipart/form-data; boundary=%s\r\n"
            "Content-Length: %d\r\n\r\n"
            % (method, path, port, BOUNDARY, content_length)
        ).encode()
    )
    try:
        if isinstance(body, (bytes, bytearray)):
            if body:
                s.sendall(body)
        else:
            sent = 0
            for chunk in body:
                s.sendall(chunk)
                sent += len(chunk)
    except (BrokenPipeError, ConnectionResetError):
        pass
    resp = b""
    try:
        while True:
            b = s.recv(65536)
            if not b:
                break
            resp += b
            if len(resp) > 4096:
                break
    except OSError:
        pass
    s.close()
    return resp.decode("latin-1")


def status_of(resp):
    try:
        return int(resp.split(" ", 2)[1])
    except (IndexError, ValueError):
        return -1


def chunked_repeater(total_bytes, chunk=1024 * 1024):
    sent = 0
    while sent < total_bytes:
        n = min(chunk, total_bytes - sent)
        yield b"Z" * n
        sent += n


def main():
    binary = find_binary()
    if not binary:
        print("SKIP: crispasr binary not found")
        return 0
    model = find_model()
    if not model:
        print("SKIP: no model found (set CRISPASR_TEST_MODEL or CRISPASR_TEST_CACHE)")
        return 0
    sample = os.path.join(ROOT, "samples/jfk.wav")
    if not os.path.isfile(sample):
        print("SKIP: samples/jfk.wav not found")
        return 0

    port = 48137
    cmd = [
        binary,
        "--server",
        "-m",
        model,
        "--backend",
        "whisper",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--language",
        "en",
        "--no-prints",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    passed = failed = 0
    try:
        ready = False
        for _ in range(120):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    ready = True
                    break
            except OSError:
                time.sleep(1)
        if not ready:
            print("ERROR: server did not start")
            return 2

        with open(sample, "rb") as f:
            audio = f.read()

        def check(name, ok, detail=""):
            nonlocal passed, failed
            if ok:
                passed += 1
                print("PASS: %s" % name)
            else:
                failed += 1
                print("FAIL: %s %s" % (name, detail))

        # 1. chunked multipart upload must be accepted and transcribed.
        head = [
            "POST /v1/audio/transcriptions HTTP/1.1",
            "Host: 127.0.0.1:%d" % port,
            "Content-Type: multipart/form-data; boundary=%s" % BOUNDARY,
            "Transfer-Encoding: chunked",
        ]
        resp, _ = send_raw(port, head, [multipart_body(audio)])
        check(
            "chunked upload is not refused (no 411/400)",
            status_of(resp) == 200,
            "status=%d body=%s" % (status_of(resp), resp[:200]),
        )
        check("chunked upload transcribes", "ask not" in resp.lower(), resp[:300])

        # 2. chunked upload past the cap -> 413, and the server stays alive.
        head2 = [
            "POST /inference HTTP/1.1",
            "Host: 127.0.0.1:%d" % port,
            "Content-Type: multipart/form-data; boundary=%s" % BOUNDARY,
            "Transfer-Encoding: chunked",
        ]
        resp, _ = send_raw(port, head2, chunked_repeater(CAP + 8 * 1024 * 1024))
        check(
            "chunked upload past the cap gets 413",
            status_of(resp) == 413,
            "status=%d" % status_of(resp),
        )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=5) as hs:
                hs.sendall(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n")
                health = hs.recv(256).decode()
            check(
                "server alive after oversized chunked upload",
                "200" in health.split("\r\n")[0],
                health.split("\r\n")[0],
            )
        except OSError as e:
            check("server alive after oversized chunked upload", False, str(e))

        # 3. Content-Length multipart upload still works.
        body = multipart_body(audio)
        resp3 = send_cl(port, "POST", "/v1/audio/transcriptions", len(body), body)
        check(
            "content-length upload still works",
            status_of(resp3) == 200,
            "status=%d" % status_of(resp3),
        )

        # 4. Content-Length upload past the cap also gets 413 (bounded memory).
        over = CAP + 8 * 1024 * 1024
        resp4 = send_cl(port, "POST", "/inference", over, chunked_repeater(over))
        check(
            "oversized content-length still 413",
            status_of(resp4) == 413,
            "status=%d" % status_of(resp4),
        )

        print("%d passed, %d failed" % (passed, failed))
        return 1 if failed else 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
