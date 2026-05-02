"""Live captions from microphone via voxtral4b (PLAN #7 phase 3 demo).

Captures audio from the system microphone, feeds it to a voxtral4b
streaming session with live decode enabled, and prints text
progressively as words emerge.

Usage:
    python tools/voxtral4b_live_demo.py \\
        -m /Volumes/backups/ai/crispasr-models/voxtral-mini-4b-realtime-q4_k.gguf

    # Speak. Press Ctrl+C to stop and print the final transcript.

Sequential live decode is ~1.5x realtime on M1 Q4_K, so on a typical
mic feed the decoder will fall progressively behind the speech. For
true realtime, a decoder thread parallel to the encoder is needed
(PLAN #7 phase 4, deferred). This demo is most useful for:
- Validating the streaming API end-to-end
- Speak slowly / in short bursts and watch tokens emerge
- File-mode replay (point a `play` at a file → mic → demo)
"""
from __future__ import annotations

import argparse
import os
import queue
import signal
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

import crispasr  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-m", "--model", required=True, help="Path to voxtral-mini-4b-realtime GGUF")
    p.add_argument("--chunk-ms", type=int, default=80,
                   help="PCM chunk size pushed to the stream (ms). Mic callback fires more often than this; "
                        "the demo accumulates and feeds.")
    p.add_argument("--no-live", action="store_true",
                   help="Disable live decode — decode happens at flush only (PTT semantics).")
    args = p.parse_args()

    if not Path(args.model).exists():
        print(f"error: model not found: {args.model}", file=sys.stderr)
        return 2

    print(f"[demo] loading {args.model} ...", file=sys.stderr)
    sess = crispasr.Session(backend="voxtral4b", model_path=args.model)
    stream = sess.stream_open(step_ms=args.chunk_ms, length_ms=15000, live=not args.no_live)

    # Mic → queue → main-thread feed loop. Keeps the audio callback
    # short (just enqueue), main thread does the LLM work.
    audio_q: "queue.Queue[bytes]" = queue.Queue()
    stop_flag = {"stop": False}

    def mic_cb(pcm):
        try:
            audio_q.put_nowait(pcm.copy())
        except queue.Full:
            pass

    def on_sigint(signum, frame):
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, on_sigint)

    print("[demo] opening mic ... (speak now; Ctrl+C to stop)", file=sys.stderr)
    mic = crispasr.Mic(sample_rate=16000, channels=1, callback=mic_cb)
    mic.start()

    try:
        last_text = ""
        last_print_t = time.monotonic()
        while not stop_flag["stop"]:
            try:
                pcm = audio_q.get(timeout=0.1)
            except queue.Empty:
                continue
            stream.feed(pcm)
            now = time.monotonic()
            # Poll every 100ms — more frequent risks the get_text/feed
            # contention; less frequent makes the demo feel sluggish.
            if now - last_print_t > 0.1:
                out = stream.get_text()
                txt = out["text"]
                if txt:
                    sys.stdout.write(txt)
                    sys.stdout.flush()
                last_print_t = now
    finally:
        mic.stop()
        mic.close()

    print("\n[demo] flushing ...", file=sys.stderr)
    stream.flush()
    out = stream.get_text()
    if out["text"]:
        sys.stdout.write(out["text"])
    sys.stdout.write("\n")
    sys.stdout.flush()
    print("[demo] done.", file=sys.stderr)

    stream.close()
    sess.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
