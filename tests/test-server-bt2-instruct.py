#!/usr/bin/env python3
"""e2e: bt2-tts guidance branches in one reused server session.

the server keeps one backend for its lifetime, and a guided request
(`instructions` with `cfg_scale` other than 1) grows the backend from one
prompt branch to two. every request is seeded, so equal prompts must give
equal audio whatever ran before them:

  1. one -> two -> one -> two: plain and guided requests repeat exactly after
     the branch count changed in between
  2. cfg_scale 0 runs the instruction-free prompt alone, which IS the plain
     request, so the two are byte-identical
  3. cfg_scale 1 (instruction prompt alone) and 4 (guided) both differ from
     plain and from each other

run: python3 tests/test-server-bt2-instruct.py
env: CRISPASR_TEST_SERVER_URL  test a running server
     CRISPASR_TEST_MODEL_NAME  `model` field to send (llama-swap routing)
     CRISPASR_BT2_MODEL        model file, to boot a local server instead
     CRISPASR_BT2_CODEC        qwen3-tts-tokenizer-12hz gguf for the local server
skips when neither a server url nor the model files are given.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = int(os.environ.get("PORT", "11448"))
TEXT = "I found the file you asked about and opened it."
INSTRUCTIONS = "Whisper slowly, as if sharing a secret."
SEED = 7
ATTESTATION = "test fixture: synthetic speech for an automated test"
BOOT_TIMEOUT_S = 600
REQUEST_TIMEOUT_S = 900

# (label, extra request fields), in the order they are sent
SEQUENCE = [
    ("plain", {}),
    ("guided 4", {"instructions": INSTRUCTIONS, "cfg_scale": 4}),
    ("plain again", {}),
    ("guided 4 again", {"instructions": INSTRUCTIONS, "cfg_scale": 4}),
    ("instruction alone", {"instructions": INSTRUCTIONS, "cfg_scale": 1}),
    ("negative alone", {"instructions": INSTRUCTIONS, "cfg_scale": 0}),
]
MUST_MATCH = [("plain", "plain again"), ("guided 4", "guided 4 again"), ("plain", "negative alone")]
MUST_DIFFER = [("plain", "guided 4"), ("plain", "instruction alone"), ("guided 4", "instruction alone")]


def find_binary():
    for c in ["build/bin/crispasr", "build-ninja-compile/bin/crispasr", "bin/crispasr"]:
        p = os.path.join(ROOT, c)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def speech(url, **fields):
    """POST /v1/audio/speech, return (status, body)."""
    body = {"input": TEXT, "response_format": "pcm", "seed": SEED, "marking_attestation": ATTESTATION}
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


def boot_local(binary, model, codec, log_path):
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [binary, "--server", "--port", str(PORT), "--backend", "bt2-tts", "--model", model,
         "--codec-model", codec, "--accept-marking-responsibility"],
        stdout=log, stderr=subprocess.STDOUT)
    url = "http://127.0.0.1:%d" % PORT
    deadline = time.time() + BOOT_TIMEOUT_S
    while time.time() < deadline:
        if proc.poll() is not None:
            sys.exit("FAIL: server exited during startup, see %s" % log_path)
        try:
            urllib.request.urlopen(url + "/health", timeout=2)
            return proc, url
        except (urllib.error.URLError, OSError):
            time.sleep(1)
    proc.kill()
    sys.exit("FAIL: server did not come up, see %s" % log_path)


def check(url):
    failures = []
    digest = {}
    for label, fields in SEQUENCE:
        status, body = speech(url, **fields)
        digest[label] = hashlib.sha1(body).hexdigest()
        print("%-20s %d %8dB sha1 %s" % (label, status, len(body), digest[label][:10]))
        if status != 200:
            failures.append("%s: expected 200, got %d: %s" % (label, status, body[:200]))
    for a, b in MUST_MATCH:
        if digest[a] != digest[b]:
            failures.append("'%s' and '%s' differ: equal prompts did not give equal audio" % (a, b))
    for a, b in MUST_DIFFER:
        if digest[a] == digest[b]:
            failures.append("'%s' and '%s' are identical: the instruction or its scale had no effect" % (a, b))
    return failures


def main():
    url = os.environ.get("CRISPASR_TEST_SERVER_URL")
    proc = None
    if not url:
        model, codec = os.environ.get("CRISPASR_BT2_MODEL"), os.environ.get("CRISPASR_BT2_CODEC")
        binary = find_binary()
        if not (model and codec and os.path.isfile(model) and os.path.isfile(codec)):
            print("SKIP: set CRISPASR_TEST_SERVER_URL or CRISPASR_BT2_MODEL + CRISPASR_BT2_CODEC")
            return 0
        if not binary:
            sys.exit("ERROR: crispasr binary not found. build first.")
        log_path = os.path.join(tempfile.mkdtemp(prefix="crispasr-bt2-instruct."), "server.log")
        proc, url = boot_local(binary, model, codec, log_path)
    try:
        failures = check(url.rstrip("/"))
    finally:
        if proc:
            proc.terminate()
            proc.wait()
    for f in failures:
        print("FAIL: " + f)
    if not failures:
        print("PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
