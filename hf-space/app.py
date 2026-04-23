import json
import os
import time
from pathlib import Path

import gradio as gr
import requests


SERVER_URL = os.environ.get("CRISPASR_SERVER_URL", "http://127.0.0.1:8080").rstrip("/")
SPACE_TITLE = os.environ.get("CRISPASR_SPACE_TITLE", "CrispASR")
DEFAULT_LANGUAGE = os.environ.get("CRISPASR_LANGUAGE", "en")
DEFAULT_MODEL = os.environ.get("CRISPASR_MODEL", "auto")


def _request(method: str, path: str, **kwargs):
    return requests.request(method, f"{SERVER_URL}{path}", timeout=300, **kwargs)


def fetch_status():
    try:
        health = _request("GET", "/health")
        health.raise_for_status()
        models = _request("GET", "/v1/models")
        models.raise_for_status()
        health_json = health.json()
        models_json = models.json()
        model_names = [item.get("id", "") for item in models_json.get("data", [])]
        return (
            "ready",
            json.dumps(health_json, indent=2, ensure_ascii=False),
            "\n".join(model_names) if model_names else "(no models reported)",
        )
    except Exception as exc:
        return "starting", f"{type(exc).__name__}: {exc}", DEFAULT_MODEL


def wait_for_server():
    last_status = "starting"
    last_health = ""
    last_models = DEFAULT_MODEL
    for _ in range(300):
        last_status, last_health, last_models = fetch_status()
        if last_status == "ready":
            break
        time.sleep(1)
    return last_status, last_health, last_models


def transcribe(audio_path: str, language: str, prompt: str, temperature: float, response_format: str):
    if not audio_path:
        raise gr.Error("Upload or record audio first.")

    file_path = Path(audio_path)
    if not file_path.exists():
        raise gr.Error("Audio file is not available anymore.")

    data = {
        "model": "loaded-model",
        "response_format": response_format,
        "temperature": f"{temperature:.2f}",
    }

    if language and language != "auto":
        data["language"] = language
    if prompt:
        data["prompt"] = prompt
    with file_path.open("rb") as f:
        response = _request(
            "POST",
            "/v1/audio/transcriptions",
            files={"file": (file_path.name, f, "application/octet-stream")},
            data=data,
        )

    if response.status_code >= 400:
        raise gr.Error(f"{response.status_code}: {response.text}")

    content_type = response.headers.get("content-type", "")
    if response_format == "verbose_json" or "application/json" in content_type:
        payload = response.json()
        text = payload.get("text", "") if isinstance(payload, dict) else ""
        return text, json.dumps(payload, indent=2, ensure_ascii=False)

    text = response.text.strip()
    return text, text


with gr.Blocks(title=SPACE_TITLE) as demo:
    gr.Markdown(
        f"""# {SPACE_TITLE}

Offline speech transcription via CrispASR's OpenAI-compatible server.

- Server URL: `{SERVER_URL}`
- Model path: `{DEFAULT_MODEL}`
"""
    )

    with gr.Row():
        status = gr.Textbox(label="Server status", interactive=False)
        models = gr.Textbox(label="Loaded model(s)", interactive=False)
    health = gr.Code(label="/health", language="json", interactive=False)
    refresh = gr.Button("Refresh server status")

    with gr.Row():
        audio = gr.Audio(label="Audio", type="filepath", sources=["upload", "microphone"])
        with gr.Column():
            language = gr.Textbox(value=DEFAULT_LANGUAGE, label="Language", placeholder="auto or ISO-639-1 code")
            response_format = gr.Dropdown(
                ["text", "verbose_json"], value="verbose_json", label="Response format"
            )
            temperature = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Temperature")
            prompt = gr.Textbox(label="Prompt", placeholder="Optional prompt or context")
            submit = gr.Button("Transcribe", variant="primary")

    transcript = gr.Textbox(label="Transcript", lines=12)
    raw = gr.Code(label="Raw response", language="json")

    refresh.click(fetch_status, outputs=[status, health, models])
    submit.click(
        transcribe,
        inputs=[audio, language, prompt, temperature, response_format],
        outputs=[transcript, raw],
    )
    demo.load(wait_for_server, outputs=[status, health, models])


if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        show_api=False,
    )
