---
title: CrispASR
sdk: docker
app_port: 7860
pinned: false
---

# CrispASR Space

This folder contains a Hugging Face Docker Space wrapper for CrispASR:

- a Gradio UI for upload / microphone transcription
- a local CrispASR server started inside the same container
- OpenAI-compatible transcription requests routed to `http://127.0.0.1:8080/v1/audio/transcriptions`

## Environment variables

- `CRISPASR_MODEL=/models/model.gguf`
- `CRISPASR_BACKEND=whisper` or another backend name
- `CRISPASR_LANGUAGE=en`
- `CRISPASR_AUTO_DOWNLOAD=1`
- `CRISPASR_EXTRA_ARGS=`

## Local build

```bash
docker build -f hf-space/Dockerfile -t crispasr-hf-space .
docker run --rm -p 7860:7860 -p 8080:8080 \
  -e CRISPASR_MODEL=/models/ggml-base.en.bin \
  -v "$PWD/models:/models" \
  crispasr-hf-space
```
