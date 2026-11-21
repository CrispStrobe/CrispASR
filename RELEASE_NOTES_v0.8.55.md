# v0.8.55 — new backends, Metal Q4_K, gcc 15

## New models

- **ZONOS3-Mini TTS** (`--backend zonos3`, #603) runs directly from Zyphra's own GGUFs. It is diff-harness checked against upstream (logits cos ≥ 0.9997) and passes the ASR roundtrip.
- **Reson8 Rumble-2-Flash ASR** (#602) runs on the qwen3-asr runtime with its 8 s encoder attention windows. GGUFs: `cstr/rumble-2-flash-GGUF`.
- **Inworld Voice Isolator v1** (#602) is a native separation backend. GGUFs: `cstr/inworld-voice-isolator-GGUF`.
- **Fluid Speech Pro** (#595) adds the FrameCleaner `--clean-mode`. **Musci-ASR** is in the registry.

## Fixes

- **bt2-tts** (#594) applies its repetition penalty once per distinct token. Text is packed into paragraph chunks, and a short final chunk is rebalanced, so instruction text no longer bleeds into the audio.
- **mini-bs-roformer runs on Metal by default again** (#568). The Q4_K/Q8_0 F32-precision Metal kernels are restored, and an Apple M1 test matches CPU output. `CRISPASR_MINI_BS_ROFORMER_METAL=0` still forces the CPU.
- **fastconformer-ctc punctuated models** (e.g. Russian) no longer lose their word spaces. There is no space before punctuation, matching NeMo byte for byte.
- **confucius4-r2t2** routes by the GGUF's architecture tag, not its filename.
- **gcc 15** builds no longer segfault in VAD/GPU backend selection.
- **voxcpm2** VAE setup uploads the upsampler weights once (#461). That is 158 MB less GPU memory and about a third less setup time.
- **VibeVoice** base TTS follows upstream prompts, CFG noise, diffusion configuration and generation budget.
- **Windows** model paths open as UTF-8.
- **German ordinals and abbreviations** no longer split TTS chunks or subtitles.
- **The pcs punctuation model** resolves to the mixed-precision GGUF (#604).
