#!/usr/bin/env python
"""test-server-realtime-api.py — integration test for the server's --ws-port
vLLM Realtime API WebSocket streaming endpoint.

Boots `crispasr --server --ws-port 0` with a small whisper model, then with a
stdlib-only raw WebSocket client:
  1. verifies the Sec-WebSocket-Accept handshake
  2. streams samples/jfk.wav as base64 PCM16 JSON frames via input_audio_buffer.append
  3. asserts a non-empty transcription comes back as JSON text frames (transcription.delta).

SKIPs (exit 0) without a whisper GGUF. No third-party deps.

Usage: python tests/test-server-realtime-api.py [--port N] [--cache-dir DIR]
"""
import base64
import hashlib
import json
import os
import socket
import struct
import subprocess
import sys
import time
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def find_binary():
    for c in ["build/bin/crispasr", "build-ninja-compile/bin/crispasr", "bin/crispasr"]:
        p = os.path.join(ROOT, c)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def find_whisper(cache_dir):
    cands = []
    for d in [cache_dir, os.environ.get("CRISPASR_TEST_CACHE"), os.environ.get("CRISPASR_MODELS_DIR"),
              os.path.expanduser("~/.cache/crispasr")]:
        if d and os.path.isdir(d):
            for f in os.listdir(d):
                if f.startswith("ggml-") and f.endswith(".bin"):
                    cands.append(os.path.join(d, f))
    return cands[0] if cands else None


def load_pcm_16_bytes(path):
    w = wave.open(path, "rb")
    n, sr = w.getnframes(), w.getframerate()
    raw = w.readframes(n)
    w.close()
    return raw, sr


def ws_frame(payload, opcode=0x1):
    # Client->server frame, masked per RFC 6455.
    b = bytearray([0x80 | opcode])
    n = len(payload)
    mask_bit = 0x80
    if n < 126:
        b.append(mask_bit | n)
    elif n < 65536:
        b.append(mask_bit | 126)
        b += struct.pack(">H", n)
    else:
        b.append(mask_bit | 127)
        b += struct.pack(">Q", n)
    key = os.urandom(4)
    b += key
    if isinstance(payload, str):
        payload = payload.encode('utf-8')
    b += bytes(payload[i] ^ key[i % 4] for i in range(n))
    return bytes(b)


def read_one_frame(sock, timeout=4.0):
    """Return (opcode, payload) of the next small server frame (under 126 bytes, like a pong),
    or (None, b"") on timeout or a larger frame."""
    sock.settimeout(timeout)
    buf = b""
    try:
        while True:
            if len(buf) >= 2:
                ln = buf[1] & 0x7F
                if ln >= 126:
                    return None, b""
                if len(buf) >= 2 + ln:
                    return buf[0] & 0x0F, buf[2:2 + ln]
            chunk = sock.recv(1)
            if not chunk:
                return None, b""
            buf += chunk
    except socket.timeout:
        return None, b""


def handshake(port, key_header="Sec-WebSocket-Key"):
    key = base64.b64encode(os.urandom(16)).decode()
    req = ("GET /v1/realtime HTTP/1.1\r\nHost: 127.0.0.1:%d\r\nUpgrade: websocket\r\n"
           "Connection: Upgrade\r\n%s: %s\r\n"
           "Sec-WebSocket-Version: 13\r\n\r\n" % (port, key_header, key))
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    s.sendall(req.encode())
    resp = b""
    while b"\r\n\r\n" not in resp:
        chunk = s.recv(1)
        if not chunk:
            break
        resp += chunk
    expected = base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()
    ok = resp.startswith(b"HTTP/1.1 101") and ("Sec-WebSocket-Accept: " + expected).encode() in resp
    return s, ok


def read_server_frames(sock, timeout=4.0):
    sock.settimeout(timeout)
    buf = b""
    msgs = []
    try:
        while True:
            chunk = sock.recv(8192)
            if not chunk:
                break
            buf += chunk
            # Parse as many complete frames as present.
            while len(buf) >= 2:
                ln = buf[1] & 0x7F
                off = 2
                if ln == 126:
                    if len(buf) < 4:
                        break
                    ln = struct.unpack(">H", buf[2:4])[0]
                    off = 4
                elif ln == 127:
                    if len(buf) < 10:
                        break
                    ln = struct.unpack(">Q", buf[2:10])[0]
                    off = 10
                if len(buf) < off + ln:
                    break
                payload = buf[off:off + ln]
                buf = buf[off + ln:]
                opcode = buf and (buf[0] & 0x0F)  # noqa
                msgs.append(payload.decode("utf-8", "replace"))
    except socket.timeout:
        pass
    return msgs


def main():
    port = 11520
    cache_dir = os.environ.get("CRISPASR_TEST_CACHE", "")
    backend = "whisper"
    explicit_model = ""
    language = "en"
    server_vad = "--server-vad" in sys.argv[1:]
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
        elif a.startswith("--port="):
            port = int(a.split("=", 1)[1])
        elif a == "--cache-dir" and i + 1 < len(args):
            cache_dir = args[i + 1]
        elif a.startswith("--cache-dir="):
            cache_dir = a.split("=", 1)[1]
        elif a == "--backend" and i + 1 < len(args):
            backend = args[i + 1]
        elif a.startswith("--backend="):
            backend = a.split("=", 1)[1]
        elif a == "--model" and i + 1 < len(args):
            explicit_model = args[i + 1]
        elif a.startswith("--model="):
            explicit_model = a.split("=", 1)[1]
        elif a == "--language" and i + 1 < len(args):
            language = args[i + 1]
        elif a.startswith("--language="):
            language = a.split("=", 1)[1]

    binary = find_binary()
    if not binary:
        print("SKIP: crispasr binary not found")
        return 0
    model = explicit_model or find_whisper(cache_dir)
    if not model:
        print("SKIP: no model found (pass --model or set CRISPASR_TEST_CACHE / CRISPASR_MODELS_DIR)")
        return 0
    sample = os.path.join(ROOT, "samples/jfk.wav")
    if not os.path.isfile(sample):
        print("SKIP: samples/jfk.wav not found")
        return 0

    ws_port = port + 1
    rt_port = ws_port + 1
    cmd = [binary, "--server", "-m", model, "--backend", backend,
           "--host", "127.0.0.1", "--port", str(port), "--ws-port", "0",
           "--language", language, "--no-prints"]
    if server_vad:
        vad_model = os.environ.get("CRISPASR_TEST_VAD_MODEL", "webrtc")
        cmd += ["--vad", "--vad-model", vad_model, "--vad-min-silence-duration-ms", "300"]
    if cache_dir:
        cmd += ["--cache-dir", cache_dir]
    print("Model:", model)
    show_server_log = os.environ.get("CRISPASR_TEST_SERVER_LOG") == "1"
    proc = subprocess.Popen(cmd,
                            stdout=None if show_server_log else subprocess.DEVNULL,
                            stderr=None if show_server_log else subprocess.DEVNULL)
    try:
        # Wait for /health.
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
        time.sleep(1)  # let the WS listener bind

        passed = failed = 0

        # 1. Handshake + accept verification.
        key = base64.b64encode(os.urandom(16)).decode()
        req = ("GET /v1/realtime HTTP/1.1\r\nHost: 127.0.0.1:%d\r\nUpgrade: websocket\r\n"
               "Connection: Upgrade\r\nSec-WebSocket-Key: %s\r\n"
               "Sec-WebSocket-Version: 13\r\n\r\n" % (rt_port, key))
        s = socket.create_connection(("127.0.0.1", rt_port), timeout=5)
        s.sendall(req.encode())
        resp_bytes = b""
        while b"\r\n\r\n" not in resp_bytes:
            chunk = s.recv(1)
            if not chunk: break
            resp_bytes += chunk
        
        resp = resp_bytes.decode("utf-8", "replace")
        accept = ""
        for line in resp.split("\r\n"):
            if line.lower().startswith("sec-websocket-accept:"):
                accept = line.split(":", 1)[1].strip()
        expected = base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()
        if "101" in resp.split("\r\n")[0] and accept == expected:
            print("  ✓ handshake Sec-WebSocket-Accept correct (RFC 6455 GUID)")
            passed += 1
        else:
            print("  ✗ bad handshake: status=%r accept=%r expected=%r" % (resp.split(chr(13))[0], accept, expected))
            failed += 1
            s.close()
            print("\nRESULT: %d passed, %d failed" % (passed, failed))
            return 1

        # Read session.created
        msgs = read_server_frames(s, timeout=1.0)
        session_created = False
        session_contract = None
        for m in msgs:
            try:
                j = json.loads(m)
                if j.get("type") == "session.created":
                    session_created = True
                    session_contract = j
            except: pass
        if session_created:
            print("  ✓ received session.created")
            passed += 1
        else:
            print("  ✗ no session.created received")
            failed += 1

        expected_detection = "server_vad" if server_vad else "client_commit"
        if (session_contract and session_contract.get("turn_detection") == expected_detection
                and session_contract.get("server_vad") is server_vad
                and isinstance(session_contract.get("partial_transcription"), bool)):
            print("  ✓ session declares its turn/VAD/partial contract")
            passed += 1
        else:
            print("  ✗ session did not declare its turn/VAD contract: %r" % session_contract)
            failed += 1

        # A Ping must get a Pong with the same payload.
        s.sendall(ws_frame(b"abcd", opcode=0x9))
        opcode, payload = read_one_frame(s, timeout=5.0)
        if opcode == 0xA and payload == b"abcd":
            print("  ✓ ping answered with matching pong")
            passed += 1
        else:
            print("  ✗ no pong for ping: opcode=%r payload=%r" % (opcode, payload))
            failed += 1

        # Proxies may send the key header in another case.
        s2, ok = handshake(rt_port, "Sec-Websocket-Key")
        s2.close()
        if ok:
            print("  ✓ handshake accepts the key header spelled Sec-Websocket-Key")
            passed += 1
        else:
            print("  ✗ handshake rejected Sec-Websocket-Key")
            failed += 1

        # 2. Stream PCM, collect text frames.
        pcm, sr = load_pcm_16_bytes(sample)
        chunk = 16000 * 2  # ~1s of PCM16
        msgs = []
        for i in range(0, len(pcm), chunk):
            b64 = base64.b64encode(pcm[i:i+chunk]).decode('utf-8')
            msg = json.dumps({"type": "input_audio_buffer.append", "audio": b64})
            s.sendall(ws_frame(msg, opcode=0x1))
            msgs += read_server_frames(s, timeout=0.4)

        premature = []
        speech_started = False
        for m in msgs:
            try:
                event_type = json.loads(m).get("type", "")
                if event_type.startswith("conversation.item.input_audio_transcription"):
                    premature.append(m)
                if event_type == "input_audio_buffer.speech_started":
                    speech_started = True
            except Exception:
                pass
        if server_vad:
            if speech_started:
                print("  ✓ server VAD emitted speech_started and opened the ASR gate")
                passed += 1
            else:
                print("  ✗ server VAD did not emit speech_started")
                failed += 1
        expects_partials = bool(session_contract and session_contract.get("partial_transcription"))
        if expects_partials and premature:
            print("  ✓ stateful backend emitted transcription before commit")
            passed += 1
        elif not expects_partials and not premature:
            print("  ✓ fallback backend buffers without growing-prefix re-decodes")
            passed += 1
        else:
            print("  ✗ pre-commit behavior disagrees with session contract: %r" % premature[:2])
            failed += 1

        commit_msg = json.dumps({"type": "input_audio_buffer.commit"})
        auto_completed = False
        if server_vad:
            silence = bytes(16000 * 2)  # one second PCM16 silence
            silence_msg = json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(silence).decode("utf-8"),
            })
            s.sendall(ws_frame(silence_msg, opcode=0x1))
            msgs += read_server_frames(s, timeout=30.0)
            event_types = []
            for m in msgs:
                try: event_types.append(json.loads(m).get("type"))
                except Exception: pass
            auto_completed = "conversation.item.input_audio_transcription.completed" in event_types
            if "input_audio_buffer.speech_stopped" in event_types and auto_completed:
                print("  ✓ server VAD emitted speech_stopped and auto-committed")
                passed += 1
            else:
                print("  ✗ server VAD did not stop/auto-commit: %r" % event_types[-8:])
                failed += 1
        if not auto_completed:
            s.sendall(ws_frame(commit_msg, opcode=0x1))
            # CPU backends can take several seconds for a committed turn; this is a
            # correctness/live test, not the latency budget itself.
            msgs += read_server_frames(s, timeout=30.0)
        texts = []
        completed = None
        for m in msgs:
            try:
                j = json.loads(m)
                if j.get("type") == "conversation.item.input_audio_transcription.delta":
                    texts.append(j.get("delta", ""))
                elif j.get("type") == "conversation.item.input_audio_transcription.completed":
                    completed = j
            except Exception:
                pass
        joined = "".join(t for t in texts)
        print("  transcription:", joined[:160])
        if joined and len(joined) > 5:
            print("  ✓ streaming returned non-empty transcription")
            passed += 1
        else:
            print("  ✗ no transcription text received")
            failed += 1

        if (completed and completed.get("audio_duration_ms", 0) > 1000
                and completed.get("processing_ms", 0) > 0
                and completed.get("model_queue_wait_ms", -1) >= 0
                and completed.get("queue_backlog_duration_ms") == 0
                and "realtime_factor" in completed):
            print("  ✓ completion exposes audio and processing timing")
            passed += 1
        else:
            print("  ✗ completion timing metadata missing: %r" % completed)
            failed += 1

        # 3. Reuse the same WebSocket for another turn. Its duration must be
        # just this append, proving commit cleared the prior audio prefix.
        if server_vad:
            # A low silence threshold may split the sample at an internal pause
            # and leave a later fragment active after the first auto-completion.
            # Establish an explicit clean boundary for the reset assertion.
            s.sendall(ws_frame(commit_msg, opcode=0x1))
            read_server_frames(s, timeout=30.0)
        second_pcm = pcm[:16000 * 2]
        second_append = json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(second_pcm).decode("utf-8"),
        })
        s.sendall(ws_frame(second_append, opcode=0x1))
        s.sendall(ws_frame(commit_msg, opcode=0x1))
        second_msgs = read_server_frames(s, timeout=30.0)
        second_completed = None
        for m in second_msgs:
            try:
                j = json.loads(m)
                if j.get("type") == "conversation.item.input_audio_transcription.completed":
                    second_completed = j
            except Exception:
                pass
        if second_completed and 900 <= second_completed.get("audio_duration_ms", 0) <= 1100:
            print("  ✓ second turn is reset to its own 1.0 s audio buffer")
            passed += 1
        else:
            print("  ✗ second turn retained prior audio: %r" % second_completed)
            failed += 1
        s.close()

        print("\nRESULT: %d passed, %d failed" % (passed, failed))
        return 0 if failed == 0 else 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
