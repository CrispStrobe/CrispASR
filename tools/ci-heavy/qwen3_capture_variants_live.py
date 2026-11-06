#!/usr/bin/env python3
"""Model-backed Qwen3-TTS capture HTTP audit; runs on an isolated local server.

Requires a built CLI, its codec, CustomVoice and VoiceDesign GGUFs, and an ASR
model. Captures are written only under the supplied temporary output directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import time
import urllib.error
import urllib.request
import wave

TEXT = "Good evening, this is the voice assistant."
INSTRUCT = "A calm adult voice speaking clearly."


def wait_file(path, seconds=120):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if path.is_file():
            return json.loads(path.read_text())
        time.sleep(0.1)
    raise AssertionError(f"manifest not produced: {path}")


def request(base, payload):
    req = urllib.request.Request(base + "/v1/audio/speech", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as response:
        return response.headers["X-Crispasr-Request-Id"], response.read()


def assert_manifest(root, request_id, payload, variant, stream):
    response = wait_file(root / "requests" / request_id / "manifest.json")
    assert response["complete"] and response["body_bytes"] == len(payload), response
    assert response["body_sha256"] == hashlib.sha256(payload).hexdigest(), response
    samples = [json.loads(p.read_text()) for p in (root / "samples").glob("*/manifest.json")]
    matches = [sample for sample in samples if sample["request_id"] == request_id]
    assert len(matches) == 1, matches
    sample = matches[0]
    assert sample["capture_complete"] and sample["generated_codes"]["eos_observed"], sample
    assert sample["model_input_text"] == TEXT and sample["request_source_text"] == "original source text", sample
    assert sample["model"]["type"] == variant, sample["model"]
    assert sample["http_request_manifest"] == f"requests/{request_id}/manifest.json"
    assert sample["decoder_calls"] and all(
        call["windowed"] == stream and call["decoded_pcm_sha256"] for call in sample["decoder_calls"]
    ), sample
    if variant == "custom_voice":
        assert sample["voice"].lower() == "ryan" and sample["reference"]["speaker_embedding_sha256"], sample
    else:
        assert sample["instruction"] == INSTRUCT, sample
    sample_path = root / "samples" / sample["sample_id"]
    for item in [sample["generated_codes"], *sample["decoder_calls"]]:
        data = (sample_path / item["file"]).read_bytes()
        assert len(data) == 4 * item["shape"][0] * item["shape"][1]
        assert hashlib.sha256(data).hexdigest() == item["sha256"]
    return sample


def disconnect(port, root, payload):
    body = json.dumps(payload).encode()
    request_bytes = (f"POST /v1/audio/speech HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                     f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode() + body
    sock = socket.create_connection(("127.0.0.1", port), timeout=600)
    sock.settimeout(600)
    sock.sendall(request_bytes)
    received = b""
    while b"\r\n\r\n" not in received:
        chunk = sock.recv(4096)
        assert chunk, "server closed before headers"
        received += chunk
    headers, data = received.split(b"\r\n\r\n", 1)
    if not data:
        assert sock.recv(4096), "server closed before first chunk"
    request_id = next(line.split(b":", 1)[1].strip().decode() for line in headers.split(b"\r\n")
                      if line.lower().startswith(b"x-crispasr-request-id:"))
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    sock.close()
    result = wait_file(root / "requests" / request_id / "manifest.json", 600)
    assert not result["complete"] and result["capture_error"] in ("client_disconnected", "response_capture_abandoned"), result
    print("HTTP_DISCONNECT_PASS", request_id, result["body_bytes"], result["capture_error"], flush=True)


def run_variant(args, variant, port):
    root = (args.output / variant).resolve()
    root.mkdir(parents=True, exist_ok=True)
    model = args.custom if variant == "custom_voice" else args.design
    backend = "qwen3-tts-customvoice" if variant == "custom_voice" else "qwen3-tts-voicedesign"
    command = [str(args.cli), "--server", "--backend", backend, "-m", str(model), "--codec-model", str(args.codec),
               "--host", "127.0.0.1", "--port", str(port), "-ng"]
    env = dict(os.environ, CRISPASR_QWEN3_TTS_CAPTURE_DIR=str(root), CRISPASR_QWEN3_TTS_GREEDY="1",
               CRISPASR_TEST_STREAM_THROW="1")
    base = f"http://127.0.0.1:{port}"
    with (root / "server.log").open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(180):
                if process.poll() is not None:
                    raise RuntimeError(f"server exited: {process.returncode}; see {root / 'server.log'}")
                try:
                    urllib.request.urlopen(base + "/health", timeout=1).read()
                    break
                except (OSError, urllib.error.URLError):
                    time.sleep(1)
            else:
                raise RuntimeError("server readiness timed out")
            outputs = []
            for streaming in (False, True):
                payload = {"input": TEXT, "capture_source_text": "original source text", "response_format": "pcm",
                           "stream": streaming, "seed": 42, "instructions": INSTRUCT}
                if variant == "custom_voice":
                    payload["voice"] = "ryan"
                request_id, audio = request(base, payload)
                assert len(audio) > 1000
                sample = assert_manifest(root, request_id, audio, variant, streaming)
                print("VARIANT_CAPTURE_PASS", variant, "stream" if streaming else "buffered", len(audio),
                      len(sample["decoder_calls"]), flush=True)
                outputs.append(audio)
            wav = root / "speech.wav"
            with wave.open(str(wav), "wb") as out:
                out.setnchannels(1); out.setsampwidth(2); out.setframerate(24000); out.writeframes(outputs[0])
            result = subprocess.run([str(args.cli), "--backend", "parakeet", "-m", str(args.asr), "-f", str(wav),
                                     "-np", "-nt", "-ng"], check=True, capture_output=True, text=True)
            transcript = (result.stdout + result.stderr).lower()
            assert "good evening" in transcript and "voice assistant" in transcript, transcript
            print("VARIANT_ASR_PASS", variant, flush=True)
            if variant == "custom_voice":
                request_id, _ = request(base, {"input": "__throw_test__", "response_format": "pcm", "stream": True,
                                               "voice": "ryan"})
                failed = wait_file(root / "requests" / request_id / "manifest.json")
                assert not failed["complete"] and failed["capture_error"] == "stream_incomplete", failed
                print("HTTP_INJECTED_FAILURE_PASS", flush=True)
                disconnect(port, root, {"input": "Good evening, this is a long streaming test. " * 8,
                                        "response_format": "pcm", "stream": True, "voice": "ryan", "seed": 42})
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()


def main():
    parser = argparse.ArgumentParser()
    for option in ("cli", "codec", "custom", "design", "asr", "output"):
        parser.add_argument("--" + option, type=lambda value: Path(value).resolve(), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run_variant(args, "custom_voice", 18131)
    run_variant(args, "voice_design", 18132)


if __name__ == "__main__":
    main()
