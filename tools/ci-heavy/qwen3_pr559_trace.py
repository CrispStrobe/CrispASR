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


def captured_prompts(path):
    marker, ending = "\n--- PROMPT ---\n", "\n--- END ---\n"
    parts = path.read_text().split(marker)
    if parts[0] or len(parts) < 2 or any(not part.endswith(ending) for part in parts[1:]):
        raise AssertionError("Malformed tokenizer capture: " + str(path))
    return [part[:-len(ending)] for part in parts[1:]]


def assert_prompt(prompt, system, prefill, question=""):
    header = "<|im_start|>system\n" + system + "<|im_end|>\n<|im_start|>user\n<|audio_start|>"
    if not prompt.startswith(header):
        raise AssertionError("Incorrect system turn")
    audio, separator, tail = prompt[len(header):].partition("<|audio_end|>")
    if not separator or not audio or audio.replace("<|audio_pad|>", ""):
        raise AssertionError("Incorrect audio token span")
    expected_tail = ("\n" + question if question else "") + "<|im_end|>\n<|im_start|>assistant\n" + prefill
    if tail != expected_tail:
        raise AssertionError("Incorrect question or assistant prefill")


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
        path.unlink(missing_ok=True)
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
        prompts = captured_prompts(path)
        if len(prompts) != 1:
            raise AssertionError("Expected exactly one C ABI prompt: " + mode)
        system = ("The following words may appear in the audio: words." if mode in ("unset", "zero")
                  else hotwords)
        prefill = "" if mode in ("auto", "question") else "language English<asr_text>"
        assert_prompt(prompts[0], system, prefill, question)
# Capture the CLI's actual tokenizer calls on a synthetic short clip and on
# explicit multi-chunk input. LD_PRELOAD was set before this process started.
long_audio = out / "qwen-cli-chunks.wav"
with wave.open(str(audio), "rb") as source, wave.open(str(long_audio), "wb") as target:
    target.setparams(source.getparams())
    frames = source.getnframes()
    if frames <= 0:
        raise RuntimeError("Synthetic input is empty")
    needed = 25 * 16000
    target.writeframes((source.readframes(frames) * ((needed + frames - 1) // frames))[:needed * 2])
with wave.open(str(long_audio), "rb") as target:
    if target.getnframes() != 25 * 16000:
        raise AssertionError("Chunk fixture must span exactly 25 seconds")
cli_prompt_counts = {}
for label, input_audio, context, extra, minimum in (
    ("raw", audio, "Acme context", [], 1),
    ("empty", audio, "", [], 1),
    ("chunks", long_audio, "Acme context", ["--chunk-seconds", "10"], 2),
):
    path = out / ("cli-prompt-trace-" + label + ".txt")
    path.unlink(missing_ok=True)
    env = dict(os.environ, CRISPASR_TEST_PROMPT_TRACE=str(path))
    command = [str(cli), "--backend", "qwen3", "-m", str(model), "-ng", "-t", "4", "-l", "en",
               "--context", context, "-f", str(input_audio), "-otxt", "-of", str(out / ("cli-" + label))] + extra
    with (out / ("cli-prompt-run-" + label + ".log")).open("w") as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800, check=True)
    if not path.is_file():
        raise RuntimeError("CLI tokenizer interposer did not capture " + label)
    prompts = captured_prompts(path)
    if len(prompts) < minimum:
        raise AssertionError("CLI did not produce enough chunk prompts: " + label)
    if label != "chunks" and len(prompts) != 1:
        raise AssertionError("Expected exactly one short-clip CLI prompt: " + label)
    cli_prompt_counts[label] = len(prompts)
    for prompt in prompts:
        assert_prompt(prompt, context, "language English<asr_text>")
(out / "prompt-routing-result.json").write_text(json.dumps(
    {"passed": True, "modes": [m[0] for m in modes],
     "cli_prompt_counts": cli_prompt_counts}) + "\n")
