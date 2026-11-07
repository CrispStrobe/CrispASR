# v0.8.51 — session controls, capture, and fixes

## Recognition and bindings

- Sessions now support per-session auto-chunking controls and a token budget proportional to each piece's duration, with an optional flat ceiling. These controls are available through the C ABI and language bindings (#574).

## Speech synthesis

- Qwen3-TTS can optionally capture generated codec tokens, decoder inputs, and HTTP response provenance for offline audit. Capture is disabled by default; captured data can contain speech-derived tokens and text. See `docs/qwen3-tts-codec-capture.md` for privacy, failure, and format details (#576, #578).
- VoxCPM2 avoids a costly VAE CPU fallback on long sentences (#164).

## Fixes and documentation

- Environment flags recognize false, off, and no, case-insensitively, as disabled (#572).
- Fixed formatting and the Breeze session control test harness (#575).
- Added the existing Haas-ja model to the README and recorded the remaining Nemotron 3.5 Transcribe 30B investigation (#577).
- Mini-BS-RoFormer retains its default CPU fallback on Metal; a virtual Apple Metal test did not reproduce the reported silence, but physical M1 and broader accuracy qualification remain open (#568).
