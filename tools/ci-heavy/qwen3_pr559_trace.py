#!/usr/bin/env python3
"""Capture actual C ABI Qwen3 tokenizer inputs with a test-only ELF interposer."""
import json
import os
from pathlib import Path
import sys
import subprocess
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from crispasr import Session

lib, model, audio, out, cli = map(Path, sys.argv[1:])
with wave.open(str(audio), "rb") as wav:
    assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
    pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768

modes = [
    ("unset", None, "words", "en", ""),
    ("zero", "0", "words", "en", ""),
    ("raw", "1", "words", "en", ""),
    ("changed", "1", "different", "en", ""),
    ("clear", "1", "", "en", ""),
    ("auto", "1", "words", "auto", ""),
    ("question", "1", "words", "en", "What was said?"),
]
with Session(str(model), lib_path=str(lib), backend="qwen3", n_threads=4) as session:
    session.set_max_new_tokens(128)
    for mode, switch, hotwords, language, question in modes:
        path = out / ("prompt-trace-" + mode + ".txt")
        os.environ["CRISPASR_TEST_PROMPT_TRACE"] = str(path)
        if switch is None:
            os.environ.pop("CRISPASR_QWEN3_HOTWORDS_AS_CONTEXT", None)
        else:
            os.environ["CRISPASR_QWEN3_HOTWORDS_AS_CONTEXT"] = switch
        session.set_hotwords(hotwords)
        session.set_source_language(language)
        session.set_ask(question)
        list(session.transcribe(pcm))
        if not path.is_file():
            raise RuntimeError("Tokenizer interposer did not capture " + mode)
        trace = path.read_text()
        expected = ("The following words may appear in the audio: words." if mode in ("unset", "zero")
                    else hotwords)
        if "<|im_start|>system\n" + expected + "<|im_end|>" not in trace:
            raise AssertionError("Incorrect system context: " + mode)
        if mode == "clear" and ("words" in trace or "different" in trace):
            raise AssertionError("Cleared context persisted")
        if mode not in ("auto", "question") and "language English<asr_text>" not in trace:
            raise AssertionError("Missing forced language prefill: " + mode)
        if mode == "auto" and "language English<asr_text>" in trace:
            raise AssertionError("Automatic language unexpectedly forced")
        if mode == "question" and "What was said?" not in trace:
            raise AssertionError("Question was lost")
# Capture the CLI's actual tokenizer calls on a synthetic short clip and on
# explicit multi-chunk input. LD_PRELOAD was set before this process started.
long_audio = out / "qwen-cli-chunks.wav"
with wave.open(str(audio), "rb") as source, wave.open(str(long_audio), "wb") as target:
    target.setparams(source.getparams())
    frames = source.getnframes()
    if frames <= 0:
        raise RuntimeError("Synthetic input is empty")
    target.writeframes(source.readframes(frames) * (25 * 16000 // frames + 1))
for label, input_audio, context, extra, minimum in (
    ("raw", audio, "Acme context", [], 1),
    ("empty", audio, "", [], 1),
    ("chunks", long_audio, "Acme context", ["--chunk-seconds", "10"], 2),
):
    path = out / ("cli-prompt-trace-" + label + ".txt")
    env = dict(os.environ, CRISPASR_TEST_PROMPT_TRACE=str(path))
    command = [str(cli), "--backend", "qwen3", "-m", str(model), "-ng", "-t", "4", "-l", "en",
               "--context", context, "-f", str(input_audio), "-otxt", "-of", str(out / ("cli-" + label))] + extra
    with (out / ("cli-prompt-run-" + label + ".log")).open("w") as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800, check=True)
    if not path.is_file():
        raise RuntimeError("CLI tokenizer interposer did not capture " + label)
    prompts = [p.split("\n--- END ---\n", 1)[0] for p in path.read_text().split("\n--- PROMPT ---\n")[1:]]
    if len(prompts) < minimum:
        raise AssertionError("CLI did not produce enough chunk prompts: " + label)
    for prompt in prompts:
        if "<|im_start|>system\n" + context + "<|im_end|>" not in prompt:
            raise AssertionError("CLI context missing in " + label)
        if "language English<asr_text>" not in prompt:
            raise AssertionError("CLI forced language missing in " + label)
(out / "prompt-routing-result.json").write_text(json.dumps(
    {"passed": True, "modes": [m[0] for m in modes], "cli": ["raw", "empty", "chunks"]}) + "\n")
