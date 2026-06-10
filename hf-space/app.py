"""Gradio UI wrapper for the CrispASR HTTP server.

Surfaces multiple capability areas of the C++ engine inside one Space:
  * Transcribe — 9 ASR backends, hot-swapped through POST /load.
  * Speak     — Kokoro TTS through POST /v1/audio/speech.
  * Detect    — text language identification via the crispasr-lid binary.
  * Backends  — capability snapshot from /backends + /health.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterable

import gradio as gr
import httpx
import requests
import uvicorn
from fastapi import FastAPI, Request, Response


SERVER_URL = os.environ.get("CRISPASR_SERVER_URL", "http://127.0.0.1:8080").rstrip("/")
SPACE_TITLE = os.environ.get("CRISPASR_SPACE_TITLE", "CrispASR")
DEFAULT_LANGUAGE = os.environ.get("CRISPASR_LANGUAGE", "auto")
API_KEY = next(
    (k.strip() for k in os.environ.get("CRISPASR_API_KEYS", "").split(",") if k.strip()),
    "",
)
CACHE_DIR = Path(os.environ.get("CRISPASR_CACHE_DIR", "/cache"))
SAMPLES_DIR = Path(os.environ.get("CRISPASR_SAMPLES_DIR", "/space/samples"))
CRISPASR_LID_BIN = shutil.which("crispasr-lid") or "crispasr-lid"

# (display, backend, model_arg, default_language, approx_size, blurb)
ASR_MODELS = [
    ("Whisper base — multilingual, balanced",                "whisper",            "auto", "auto", "~147 MB",
     "OpenAI Whisper. 99 langs. Native timestamps + speech translation."),
    ("Moonshine tiny — fastest CPU, EN",                     "moonshine",          "auto", "en",   "~37 MB",
     "Smallest model. ~16× realtime on CPU. English only."),
    ("Moonshine base DE — German fine-tune (CC-BY-NC-SA)",   "moonshine-de",       "auto", "de",   "~150 MB",
     "fidoriel German fine-tune of moonshine-base. 6.9% WER on CV22."),
    ("Parakeet TDT v3 — 25 EU langs, word timestamps",       "parakeet",           "auto", "auto", "~467 MB",
     "NVIDIA Parakeet. Multilingual, native word-level timestamps."),
    ("Wav2vec2 XLSR — English CTC",                          "wav2vec2",           "auto", "en",   "~212 MB",
     "Lightweight CTC. No punctuation/casing — pair with --punc-model locally."),
    ("Wav2vec2 XLSR — German CTC",                           "wav2vec2",
     "wav2vec2-large-xlsr-53-german-q4_k.gguf",                                     "de",   "~250 MB",
     "German fine-tune of wav2vec2-XLSR-53. CTC head."),
    ("Fast-conformer CTC 0.6B — English, 10× realtime",      "fastconformer-ctc",
     "parakeet-ctc-0.6b-q4_k.gguf",                                                 "en",   "~250 MB",
     "NeMo FastConformer + CTC. Fastest reasonable EN backend."),
    ("Cohere Transcribe — 13 langs, lowest EN WER",          "cohere",             "auto", "auto", "~550 MB",
     "Cohere Labs. Punctuation + casing. Slowest of the small set."),
    ("Qwen3 ASR 0.6B — 30 langs + 22 Chinese dialects",      "qwen3",              "auto", "auto", "~500 MB",
     "Speech-LLM (Whisper enc + Qwen3 0.6B). Native language ID."),
    ("Canary — multilingual + translation",                   "canary",             "auto", "auto", "~800 MB",
     "NVIDIA Canary 1B. Native speech translation. 25+ langs."),
    ("HuBERT CTC — English",                                  "hubert",             "auto", "en",   "~380 MB",
     "Self-supervised CTC. Lightweight encoder, no punctuation."),
    ("Data2Vec CTC — English",                                "data2vec",           "auto", "en",   "~380 MB",
     "Meta Data2Vec CTC. Similar to HuBERT."),
]

TTS_MODELS = [
    ("Kokoro 82M — multilingual StyleTTS2",                  "kokoro",             "auto", "en",   "~85 MB",
     "9 langs (EN/ES/FR/HI/IT/JA/PT/ZH/DE). Apache-2.0. Only TTS realistic on free-tier CPU."),
    ("VibeVoice 0.5B — EN/DE/ZH StyleTTS",                  "vibevoice",          "auto", "en",   "~200 MB",
     "VibeVoice encoder-decoder TTS. EN/DE/ZH. Apache-2.0."),
    ("Orpheus 0.5B — English expressive TTS",                "orpheus",            "auto", "en",   "~400 MB",
     "SNAC codec + LLM decoder. Expressive speech with emotion tags."),
    ("Chatterbox — English voice cloning",                   "chatterbox",         "auto", "en",   "~450 MB",
     "Resemble AI. Zero-shot voice cloning from a reference clip. Apache-2.0."),
    ("Chatterbox Turbo — faster voice cloning",              "chatterbox-turbo",   "auto", "en",   "~350 MB",
     "Faster Chatterbox variant. Same voice cloning, 2× speed."),
]

# (display, model arg passed to `-m`, blurb)
LID_MODELS = [
    ("CLD3 — 109 ISO-639-1 (default)",      "auto",                  "Google CLD3 in GGUF. ~440 KB, instant."),
    ("GlotLID-V3 — 2102 ISO-639-3 + script", "auto:glotlid",          "cis-lmu fastText. Max language coverage."),
    ("LID-176 — 176 ISO-639-1 (CC-BY-SA)",   "auto:lid-fasttext176",  "Facebook fastText. Output GGUF inherits CC-BY-SA-3.0."),
]

# (display, backend, model_arg, approx_size, blurb)
NMT_MODELS = [
    ("M2M-100 418M — 100 langs, any→any",    "m2m100",     "auto", "~800 MB",
     "Facebook M2M-100. 100 language pairs. Best general-purpose NMT."),
    ("WMT21 Dense — en↔X, 14 high-resource",  "m2m100-wmt21", "auto", "~800 MB",
     "WMT21 competition model. Higher quality for en↔de, en↔zh, etc."),
    ("MADLAD-400 — 419 langs (CC-BY-SA)",     "madlad",     "auto", "~500 MB",
     "Google T5-based. Widest language coverage. Output inherits CC-BY-SA."),
]

CRISPASR_CLI_BIN = shutil.which("crispasr") or "crispasr"


def log(msg: str) -> None:
    print(
        f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] hf-space-app: {msg}",
        flush=True,
    )


def _request(method: str, path: str, **kwargs):
    if API_KEY:
        h = dict(kwargs.pop("headers", {}) or {})
        h.setdefault("Authorization", f"Bearer {API_KEY}")
        kwargs["headers"] = h
    return requests.request(method, f"{SERVER_URL}{path}", timeout=900, **kwargs)


def _spec_by_label(table: Iterable[tuple], label: str):
    for entry in table:
        if entry[0] == label:
            return entry
    return None


def fetch_status() -> tuple[str, str, str, str]:
    try:
        h = _request("GET", "/health")
        m = _request("GET", "/v1/models")
        b = _request("GET", "/backends")
    except Exception as exc:
        return "starting", f"{type(exc).__name__}: {exc}", "", ""
    if h.status_code == 503:
        return "loading", "model loading", "", ""
    if not h.ok:
        return "error", f"/health -> {h.status_code}", "", ""
    h_json = h.json() if h.ok else {}
    m_json = m.json() if m.ok else {}
    b_json = b.json() if b.ok else {}
    backend = h_json.get("backend", "")
    model_ids = ", ".join(d.get("id", "") for d in m_json.get("data", []))
    backends = ", ".join(b_json.get("backends", []))
    info = f"backend: {backend or '(none)'}\nmodel:   {model_ids or '(none)'}"
    return "ready", info, model_ids, backends


def wait_for_server() -> tuple[str, str]:
    log("wait_for_server: start")
    for i in range(600):
        st, info, _, backends = fetch_status()
        if st == "ready":
            log(f"wait_for_server: ready after {i + 1} probe(s)")
            return f"{st}\n{info}", backends
        time.sleep(1)
    log("wait_for_server: timed out")
    return "timeout — server did not become ready", ""


def _load_via_endpoint(backend: str, model: str, language: str) -> None:
    log(f"load: backend={backend} model={model} language={language}")
    r = _request(
        "POST",
        "/load",
        files={
            "backend": (None, backend),
            "model": (None, model),
            "language": (None, language),
        },
    )
    if r.status_code >= 400:
        log(f"load: error status={r.status_code} body={r.text[:300]}")
        raise gr.Error(f"/load returned {r.status_code}: {r.text[:300]}")


def load_asr(choice: str):
    spec = _spec_by_label(ASR_MODELS, choice)
    if spec is None:
        raise gr.Error(f"Unknown ASR choice: {choice}")
    _, backend, model, lang, _, _ = spec
    _load_via_endpoint(backend, model, lang)
    _, info, _, backends = fetch_status()
    return f"ready\n{info}", backends, lang


def load_tts(choice: str):
    spec = _spec_by_label(TTS_MODELS, choice)
    if spec is None:
        raise gr.Error(f"Unknown TTS choice: {choice}")
    _, backend, model, lang, _, _ = spec
    _load_via_endpoint(backend, model, lang)
    _, info, _, backends = fetch_status()
    return f"ready\n{info}", backends


def transcribe(audio_path, language, prompt, temperature, response_format):
    if not audio_path:
        raise gr.Error("Upload, record, or pick a sample first.")
    fp = Path(audio_path)
    if not fp.exists():
        raise gr.Error("Audio file is no longer available.")
    data = {
        "model": "loaded-model",
        "response_format": response_format,
        "temperature": f"{float(temperature):.2f}",
    }
    if language and language != "auto":
        data["language"] = language
    if prompt:
        data["prompt"] = prompt
    log(
        f"transcribe: file={fp.name} language={language or 'default'} "
        f"format={response_format} temp={float(temperature):.2f}"
    )
    with fp.open("rb") as f:
        r = _request(
            "POST",
            "/v1/audio/transcriptions",
            files={"file": (fp.name, f, "application/octet-stream")},
            data=data,
        )
    if r.status_code >= 400:
        raise gr.Error(f"{r.status_code}: {r.text[:400]}")
    ct = r.headers.get("content-type", "")
    if response_format == "verbose_json" or "application/json" in ct:
        payload = r.json()
        text = payload.get("text", "") if isinstance(payload, dict) else ""
        return text, json.dumps(payload, indent=2, ensure_ascii=False)
    body = r.text.strip()
    return body, body


def synthesize(text, voice, speed):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Type some text first.")
    payload = {
        "input": text,
        "speed": float(speed),
        "response_format": "wav",
    }
    voice = (voice or "").strip()
    if voice:
        payload["voice"] = voice
    log(f"synthesize: chars={len(text)} voice={voice or '(default)'} speed={speed:.2f}")
    r = _request(
        "POST",
        "/v1/audio/speech",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
    )
    if r.status_code >= 400:
        try:
            err = r.json().get("error", {})
            msg = err.get("message") or r.text[:400]
        except Exception:
            msg = r.text[:400]
        if r.status_code == 400 and "CAP_TTS" in (r.text or ""):
            msg += " — Load a TTS backend first (Kokoro) on the Speak tab."
        raise gr.Error(f"{r.status_code}: {msg}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / f"tts_{int(time.time() * 1000)}.wav"
    out.write_bytes(r.content)
    return str(out)


def list_voices() -> str:
    try:
        r = _request("GET", "/v1/voices")
    except Exception as exc:
        return f"({type(exc).__name__}: {exc})"
    if not r.ok:
        return f"(server returned {r.status_code})"
    try:
        voices = r.json().get("voices", [])
    except Exception:
        return f"(invalid JSON: {r.text[:200]})"
    if not voices:
        return ("(no extra voice files in --voice-dir; built-in voices still work — "
                "Kokoro default is `af_heart`)")
    lines = []
    for v in voices:
        if isinstance(v, dict):
            lines.append(f"{v.get('name', '?')}\t{v.get('format', '')}")
        else:
            lines.append(str(v))
    return "\n".join(lines)


def detect_text_language(text, model_choice, top_k):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Paste some text first.")
    spec = _spec_by_label(LID_MODELS, model_choice) or LID_MODELS[0]
    _, model_arg, _ = spec
    cmd = [CRISPASR_LID_BIN, "-m", model_arg, "--text", text, "-k", str(int(top_k)), "--quiet"]
    log(f"lid: cmd={cmd[:5]} k={top_k}")
    env = {**os.environ, "CRISPASR_CACHE_DIR": str(CACHE_DIR)}
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )
    except FileNotFoundError:
        raise gr.Error(f"crispasr-lid not found at '{CRISPASR_LID_BIN}'.")
    if proc.returncode != 0:
        raise gr.Error(
            f"crispasr-lid exited {proc.returncode}: {(proc.stderr or '').strip()[:400]}"
        )
    rows = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        label = parts[0]
        try:
            score = float(parts[1])
        except ValueError:
            continue
        rows.append([label, round(score, 6)])
    if not rows:
        rows = [["(no prediction)", 0.0]]
    return rows, (proc.stdout or "") + ("\n--- stderr ---\n" + proc.stderr if proc.stderr else "")


def translate_text(text, model_choice, src_lang, tgt_lang):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Enter some text to translate.")
    spec = _spec_by_label(NMT_MODELS, model_choice) or NMT_MODELS[0]
    _, backend, model_arg, _, _ = spec
    src = (src_lang or "en").strip()
    tgt = (tgt_lang or "de").strip()
    cmd = [
        CRISPASR_CLI_BIN,
        "--backend", backend,
        "-m", model_arg,
        "--auto-download",
        "--cache-dir", str(CACHE_DIR),
        "--text", text,
        "-sl", src,
        "-tl", tgt,
    ]
    log(f"translate: backend={backend} {src}→{tgt} chars={len(text)}")
    env = {**os.environ, "CRISPASR_CACHE_DIR": str(CACHE_DIR)}
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
    except FileNotFoundError:
        raise gr.Error(f"crispasr not found at '{CRISPASR_CLI_BIN}'.")
    if proc.returncode != 0:
        raise gr.Error(
            f"Translation failed (exit {proc.returncode}): {(proc.stderr or '').strip()[:400]}"
        )
    result = proc.stdout.strip()
    if not result:
        raise gr.Error("Translation returned empty output.")
    return result


def select_nmt(choice):
    spec = _spec_by_label(NMT_MODELS, choice) or NMT_MODELS[0]
    return f"{spec[4]}\n\nApprox. download: {spec[3]}"


def list_sample_files() -> list[str]:
    if not SAMPLES_DIR.exists():
        return []
    out = []
    for p in sorted(SAMPLES_DIR.iterdir()):
        if p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}:
            out.append(str(p))
    return out


def select_asr(choice):
    spec = _spec_by_label(ASR_MODELS, choice) or ASR_MODELS[0]
    return f"{spec[5]}\n\nApprox. download: {spec[4]}", spec[3]


def select_tts(choice):
    spec = _spec_by_label(TTS_MODELS, choice) or TTS_MODELS[0]
    return f"{spec[5]}\n\nApprox. download: {spec[4]}"


def select_lid(choice):
    spec = _spec_by_label(LID_MODELS, choice) or LID_MODELS[0]
    return spec[2]


def refresh_status():
    st, info, _, backends = fetch_status()
    return f"{st}\n{info}", backends


def use_sample(path):
    return path


CAPABILITY_TABLE_MD = """### Free-tier ASR backends in this Space

| Backend | Native ts | LID | Speech translation | Best for |
|---|:-:|:-:|:-:|---|
| `whisper` | ✔ | ✔ | ✔ | All-rounder, multilingual translation |
| `parakeet` | ✔ | ✔ | | 25 EU langs + free word timestamps |
| `moonshine` | | | | Smallest English model (~37 MB) |
| `moonshine-de` | | | | German fine-tune (CC-BY-NC-SA) |
| `wav2vec2` | | | | Lightweight CTC (no punctuation) |
| `parakeet-ctc-0.6b` | | | | Fast English CTC |
| `cohere` | ✔ | LID | | Lowest English WER |
| `qwen3` | | ✔ | ✔ | 30 langs + 22 Chinese dialects |
| `canary` | ✔ | ✔ | ✔ | Multilingual + translation |
| `hubert` | | | | Self-supervised English CTC |
| `data2vec` | | | | Meta English CTC |

### TTS
* `kokoro` — 82M StyleTTS2, multilingual (9 langs). Apache-2.0. Only TTS realistic on free-tier CPU.
* `vibevoice` — 0.5B encoder-decoder. EN/DE/ZH. Apache-2.0. ~200 MB.
* `orpheus` — 0.5B SNAC + LLM. Expressive EN with emotion tags. ~400 MB.
* `chatterbox` / `chatterbox-turbo` — Zero-shot voice cloning from a reference clip. Apache-2.0.

### Why not the big speech-LLMs?
Voxtral (2.5 GB), MiMo-V2.5-ASR (4.5 GB), Granite-4.1 (3 GB), Qwen3-TTS (1.5 GB),
IndexTTS (2 GB), CosyVoice3 (1.5 GB), and VoxCPM2-TTS (2 GB) all run in CrispASR
but exceed the free-tier 16 GB ceiling. Run them locally:

```bash
docker build -f hf-space/Dockerfile -t crispasr-hf-space .
docker run --rm -p 7860:7860 -p 8080:8080 \\
    -e CRISPASR_BACKEND=voxtral -e CRISPASR_AUTO_DOWNLOAD=1 \\
    crispasr-hf-space
```

The full backend list, GPU options, and language bindings are documented in the [CrispASR README](https://github.com/CrispStrobe/CrispASR) and the live feature matrix at [`docs/feature-matrix.md`](https://github.com/CrispStrobe/CrispASR/blob/main/docs/feature-matrix.md).
"""


with gr.Blocks(title=SPACE_TITLE, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"""# {SPACE_TITLE}

CPU-only demo of [CrispASR](https://github.com/CrispStrobe/CrispASR) — one C++ binary, 24+ ASR backends and 8 TTS engines, no Python at inference time.

Each tab loads its own backend through the server's `/load` endpoint; the server holds **one** model in memory, so switching tabs may trigger a model hot-swap (download + load on first use, instant thereafter).

* Cache: `{CACHE_DIR}` &nbsp;·&nbsp; Server: `{SERVER_URL}` &nbsp;·&nbsp; Samples: `{SAMPLES_DIR}`
"""
    )

    with gr.Row():
        status_box = gr.Textbox(label="Server status / loaded model", interactive=False, lines=3, scale=3)
        backends_box = gr.Textbox(label="Available backends", interactive=False, lines=3, scale=2)
    refresh_btn = gr.Button("Refresh status", size="sm")

    with gr.Tabs():
        # --- Transcribe ------------------------------------------------
        with gr.Tab("Transcribe (ASR)"):
            gr.Markdown(
                "Speech → text. Pick a model, click **Load**, then upload, record, or pick a sample. "
                "The OpenAI-compatible `/v1/audio/transcriptions` endpoint is what carries the request."
            )
            with gr.Row():
                asr_choice = gr.Dropdown(
                    [e[0] for e in ASR_MODELS], value=ASR_MODELS[0][0], label="ASR backend / model", scale=3
                )
                asr_load_btn = gr.Button("Load model", variant="primary", scale=1)
            asr_info = gr.Textbox(
                value=f"{ASR_MODELS[0][5]}\n\nApprox. download: {ASR_MODELS[0][4]}",
                label="Notes",
                interactive=False,
                lines=2,
            )

            with gr.Row():
                with gr.Column():
                    audio = gr.Audio(label="Audio", type="filepath", sources=["upload", "microphone"])
                    sample_picker = gr.Dropdown(
                        choices=list_sample_files(),
                        label="…or load a bundled sample",
                        value=None,
                    )
                with gr.Column():
                    language = gr.Textbox(
                        value=DEFAULT_LANGUAGE,
                        label="Language",
                        placeholder="auto / en / de / fr / es / zh …",
                    )
                    response_format = gr.Dropdown(
                        ["verbose_json", "text", "srt", "vtt"],
                        value="verbose_json",
                        label="Response format",
                    )
                    temperature = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Temperature")
                    prompt = gr.Textbox(label="Prompt", placeholder="Optional context / initial prompt")

            asr_submit = gr.Button("Transcribe", variant="primary")
            transcript = gr.Textbox(label="Transcript", lines=8)
            asr_raw = gr.Code(label="Raw server response", language="json")

        # --- TTS -------------------------------------------------------
        with gr.Tab("Speak (TTS)"):
            gr.Markdown(
                "Text → speech via the OpenAI-compatible `/v1/audio/speech` endpoint. "
                "Load Kokoro before synthesizing — Kokoro is the only TTS engine that fits comfortably on free-tier CPU."
            )
            with gr.Row():
                tts_choice = gr.Dropdown(
                    [e[0] for e in TTS_MODELS], value=TTS_MODELS[0][0], label="TTS backend / model", scale=3
                )
                tts_load_btn = gr.Button("Load model", variant="primary", scale=1)
            tts_info = gr.Textbox(
                value=f"{TTS_MODELS[0][5]}\n\nApprox. download: {TTS_MODELS[0][4]}",
                label="Notes",
                interactive=False,
                lines=2,
            )

            with gr.Row():
                with gr.Column():
                    tts_text = gr.Textbox(
                        value=(
                            "Hello world. CrispASR is one binary, twenty-four ASR backends, "
                            "and eight TTS engines — running offline on this Space."
                        ),
                        label="Text to synthesize",
                        lines=4,
                    )
                    tts_voice = gr.Textbox(
                        label="Voice",
                        value="af_heart",
                        placeholder="Kokoro voices: af_heart, af_bella, am_michael, df_victoria (DE) …",
                    )
                with gr.Column():
                    tts_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed")
                    tts_voices_btn = gr.Button("List server-side voices (GET /v1/voices)")
                    tts_voices_list = gr.Textbox(label="/v1/voices", interactive=False, lines=4)

            tts_submit = gr.Button("Synthesize", variant="primary")
            tts_audio = gr.Audio(label="Output audio", interactive=False, type="filepath")

        # --- Text LID --------------------------------------------------
        with gr.Tab("Detect language (text)"):
            gr.Markdown(
                "Identify the language of pasted text via the standalone `crispasr-lid` binary "
                "(routes between CLD3, GlotLID-V3, and LID-176 by GGUF architecture)."
            )
            with gr.Row():
                lid_choice = gr.Dropdown(
                    [e[0] for e in LID_MODELS], value=LID_MODELS[0][0], label="LID model"
                )
                lid_topk = gr.Slider(1, 10, value=3, step=1, label="Top-K")
            lid_info = gr.Textbox(value=LID_MODELS[0][2], label="Notes", interactive=False, lines=1)
            lid_text = gr.Textbox(
                label="Text",
                value="Bonjour le monde, comment ça va aujourd'hui ?",
                lines=4,
            )
            lid_run = gr.Button("Detect", variant="primary")
            lid_table = gr.Dataframe(
                headers=["language", "confidence"],
                label="Top-K predictions",
                interactive=False,
            )
            lid_raw = gr.Code(label="Raw output")

        # --- Text translation ------------------------------------------
        with gr.Tab("Translate text (NMT)"):
            gr.Markdown(
                "Text → text translation via M2M-100, WMT21, or MADLAD-400 NMT backends. "
                "Uses the `crispasr` CLI's `--text` mode — the model downloads on first use."
            )
            with gr.Row():
                nmt_choice = gr.Dropdown(
                    [e[0] for e in NMT_MODELS], value=NMT_MODELS[0][0], label="NMT model", scale=3
                )
            nmt_info = gr.Textbox(
                value=f"{NMT_MODELS[0][4]}\n\nApprox. download: {NMT_MODELS[0][3]}",
                label="Notes",
                interactive=False,
                lines=2,
            )
            with gr.Row():
                nmt_src_lang = gr.Textbox(value="en", label="Source language", placeholder="en / de / fr / zh …")
                nmt_tgt_lang = gr.Textbox(value="de", label="Target language", placeholder="de / en / fr / zh …")
            nmt_input = gr.Textbox(
                label="Source text",
                value="The quick brown fox jumps over the lazy dog.",
                lines=4,
            )
            nmt_submit = gr.Button("Translate", variant="primary")
            nmt_output = gr.Textbox(label="Translation", lines=4)

        # --- Backends info --------------------------------------------
        with gr.Tab("About & backends"):
            gr.Markdown(CAPABILITY_TABLE_MD)

    # --- Wiring -------------------------------------------------------
    refresh_btn.click(refresh_status, outputs=[status_box, backends_box])

    asr_choice.change(select_asr, inputs=[asr_choice], outputs=[asr_info, language])
    asr_load_btn.click(load_asr, inputs=[asr_choice], outputs=[status_box, backends_box, language])

    tts_choice.change(select_tts, inputs=[tts_choice], outputs=[tts_info])
    tts_load_btn.click(load_tts, inputs=[tts_choice], outputs=[status_box, backends_box])
    nmt_choice.change(select_nmt, inputs=[nmt_choice], outputs=[nmt_info])
    tts_voices_btn.click(list_voices, outputs=[tts_voices_list])

    lid_choice.change(select_lid, inputs=[lid_choice], outputs=[lid_info])

    sample_picker.change(use_sample, inputs=[sample_picker], outputs=[audio])

    asr_submit.click(
        transcribe,
        inputs=[audio, language, prompt, temperature, response_format],
        outputs=[transcript, asr_raw],
    )
    tts_submit.click(synthesize, inputs=[tts_text, tts_voice, tts_speed], outputs=[tts_audio])
    lid_run.click(detect_text_language, inputs=[lid_text, lid_choice, lid_topk], outputs=[lid_table, lid_raw])
    nmt_submit.click(
        translate_text,
        inputs=[nmt_input, nmt_choice, nmt_src_lang, nmt_tgt_lang],
        outputs=[nmt_output],
    )

    demo.load(wait_for_server, outputs=[status_box, backends_box])


# ── OpenAI-compatible REST proxy in front of Gradio ──────────────────
# HF Spaces only route the public port (7860) to the Gradio app, so the
# CrispASR HTTP server's OpenAI-compatible API on :8080 was unreachable
# from outside the container — `/v1/*`, `/health`, `/backends`, `/load`
# all 404'd publicly. Mount a thin reverse proxy so those endpoints are
# served on the public URL (with CORS), for HTTP API consumers like the
# CrisperWeaver web/PWA app. The Gradio UI stays mounted at "/".

_PROXY_TIMEOUT = httpx.Timeout(900.0, connect=15.0)
_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-length",
    "host", "content-encoding",
}


def _cors(headers: dict) -> dict:
    headers["Access-Control-Allow-Origin"] = "*"
    headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    headers["Access-Control-Allow-Headers"] = "*"
    headers["Access-Control-Max-Age"] = "86400"
    return headers


def _build_app():
    """FastAPI reverse proxy for the CrispASR server, with Gradio at '/'."""
    api = FastAPI(title="CrispASR API proxy", docs_url=None, redoc_url=None)
    client = httpx.AsyncClient(timeout=_PROXY_TIMEOUT)

    async def _forward(request: Request, path: str) -> Response:
        # Preflight: answer here so the browser never reaches the backend.
        if request.method == "OPTIONS":
            return Response(status_code=204, headers=_cors({}))
        body = await request.body()
        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in _HOP_BY_HOP and k.lower() != "origin"
        }
        if API_KEY and "authorization" not in {k.lower() for k in headers}:
            headers["Authorization"] = f"Bearer {API_KEY}"
        try:
            upstream = await client.request(
                request.method, f"{SERVER_URL}{path}", content=body,
                headers=headers, params=request.query_params,
            )
        except httpx.HTTPError as exc:
            return Response(
                content=json.dumps({"error": f"upstream unavailable: {exc}"}),
                status_code=502, media_type="application/json",
                headers=_cors({}),
            )
        out = _cors({
            k: v for k, v in upstream.headers.items()
            if k.lower() not in _HOP_BY_HOP
        })
        return Response(
            content=upstream.content, status_code=upstream.status_code,
            headers=out, media_type=upstream.headers.get("content-type"),
        )

    @api.api_route("/v1/{path:path}",
                   methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    async def _proxy_v1(path: str, request: Request):
        return await _forward(request, f"/v1/{path}")

    @api.api_route("/health", methods=["GET", "OPTIONS"])
    async def _proxy_health(request: Request):
        return await _forward(request, "/health")

    @api.api_route("/backends", methods=["GET", "OPTIONS"])
    async def _proxy_backends(request: Request):
        return await _forward(request, "/backends")

    @api.api_route("/load", methods=["POST", "OPTIONS"])
    async def _proxy_load(request: Request):
        return await _forward(request, "/load")

    # Gradio UI at the root; the explicit API routes above take precedence.
    return gr.mount_gradio_app(api, demo, path="/")


app = _build_app()


if __name__ == "__main__":
    log(
        f"launch: server_url={SERVER_URL} samples={SAMPLES_DIR} "
        f"lid_bin={CRISPASR_LID_BIN} cache={CACHE_DIR}"
    )
    uvicorn.run(
        app,
        host=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    )
