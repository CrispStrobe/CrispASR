#!/usr/bin/env python3
"""e2e: bt2-tts honours the per-request `voice` on /v1/audio/speech.

the server reuses one backend and passes each request's `voice` in
params.tts_voice. a backend that reads it only at init() serves the startup
voice for every request (#371), which is what bt2-tts did:

  1. seeded request without a voice is reproducible (control)
  2. the same seeded request with an enrolled voice gives different audio
  3. a voice that names no file is an error, not a 200 in the default voice

run: python3 tests/test-server-bt2-voice.py
env: CRISPASR_TEST_SERVER_URL  test a running server (needs CRISPASR_TEST_VOICE)
     CRISPASR_TEST_VOICE       enrolled voice name on that server
     CRISPASR_TEST_MODEL_NAME  `model` field to send (llama-swap routing)
     CRISPASR_BT2_MODEL        model file, to boot a local server instead
     CRISPASR_BT2_CODEC        qwen3-tts-tokenizer-12hz gguf for the local server
skips when neither a server url nor the model files are given.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = int(os.environ.get("PORT", "11447"))
TEXT = "I found the file you asked about and opened it."
SEED = 7
CONSENT = "test fixture: public domain recording"
REF_WAV = os.path.join(ROOT, "samples", "jfk.wav")
REF_TEXT = (
    "And so my fellow Americans, ask not what your country can do for you, "
    "ask what you can do for your country."
)
LOCAL_VOICE = "jfk"
BOOT_TIMEOUT_S = 600
REQUEST_TIMEOUT_S = 900


def find_binary():
    for c in ["build/bin/crispasr", "build-ninja-compile/bin/crispasr", "bin/crispasr"]:
        p = os.path.join(ROOT, c)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def speech(url, **fields):
    """POST /v1/audio/speech, return (status, body)."""
    body = {"input": TEXT, "response_format": "pcm", "seed": SEED, "consent_attestation": CONSENT,
            "spoken_disclaimer": False, "marking_attestation": CONSENT}
    model = os.environ.get("CRISPASR_TEST_MODEL_NAME")
    if model:
        body["model"] = model
    body.update(fields)
    req = urllib.request.Request(url + "/v1/audio/speech", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def boot_local(binary, model, codec, voice_dir):
    shutil.copy(REF_WAV, os.path.join(voice_dir, LOCAL_VOICE + ".wav"))
    with open(os.path.join(voice_dir, LOCAL_VOICE + ".txt"), "w") as f:
        f.write(REF_TEXT + "\n")
    log = open(os.path.join(voice_dir, "server.log"), "w")
    proc = subprocess.Popen(
        [binary, "--server", "--port", str(PORT), "--backend", "bt2-tts", "--model", model,
         "--codec-model", codec, "--voice-dir", voice_dir, "--accept-marking-responsibility"],
        stdout=log, stderr=subprocess.STDOUT)
    url = "http://127.0.0.1:%d" % PORT
    deadline = time.time() + BOOT_TIMEOUT_S
    while time.time() < deadline:
        if proc.poll() is not None:
            sys.exit("FAIL: server exited during startup, see %s" % log.name)
        try:
            urllib.request.urlopen(url + "/health", timeout=2)
            return proc, url
        except (urllib.error.URLError, OSError):
            time.sleep(1)
    proc.kill()
    sys.exit("FAIL: server did not come up, see %s" % log.name)


def check(url, voice):
    failures = []
    runs = {}
    for label, fields in [("no voice", {}), ("no voice (repeat)", {}), ("voice " + voice, {"voice": voice})]:
        status, body = speech(url, **fields)
        runs[label] = (status, hashlib.sha1(body).hexdigest())
        print("%-28s %d %8dB sha1 %s" % (label, status, len(body), runs[label][1][:10]))
        if status != 200:
            failures.append("%s: expected 200, got %d: %s" % (label, status, body[:200]))
    if runs["no voice"] != runs["no voice (repeat)"]:
        failures.append("seeded request is not reproducible, the voice comparison proves nothing")
    if runs["no voice"][1] == runs["voice " + voice][1]:
        failures.append("voice '%s' gave the same audio as no voice: the reference was not used" % voice)
    status, body = speech(url, voice="no-such-voice-xyz")
    print("%-28s %d %8dB" % ("voice no-such-voice-xyz", status, len(body)))
    if status == 200:
        failures.append("unknown voice answered 200 instead of an error")
    return failures


def main():
    url = os.environ.get("CRISPASR_TEST_SERVER_URL")
    voice = os.environ.get("CRISPASR_TEST_VOICE", LOCAL_VOICE)
    proc = voice_dir = None
    if not url:
        model, codec = os.environ.get("CRISPASR_BT2_MODEL"), os.environ.get("CRISPASR_BT2_CODEC")
        binary = find_binary()
        if not (model and codec and os.path.isfile(model) and os.path.isfile(codec)):
            print("SKIP: set CRISPASR_TEST_SERVER_URL or CRISPASR_BT2_MODEL + CRISPASR_BT2_CODEC")
            return 0
        if not binary:
            sys.exit("ERROR: crispasr binary not found. build first.")
        voice_dir = tempfile.mkdtemp(prefix="crispasr-bt2-voices.")
        proc, url = boot_local(binary, model, codec, voice_dir)
    try:
        failures = check(url.rstrip("/"), voice)
    finally:
        if proc:
            proc.terminate()
            proc.wait()
    for f in failures:
        print("FAIL: " + f)
    if not failures:
        print("PASS")
        if voice_dir:
            shutil.rmtree(voice_dir)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
