# v0.8.50 — Mini-BS-RoFormer, SCNet, MeanVC2, and correctness fixes

## Source separation

- **Mini-BS-RoFormer V2.5** (HiDolen) native separation: four stems, F16/Q8_0/Q4_K, auto-download via `-m mini-bs-roformer` ([`cstr/mini-bs-roformer-GGUF`](https://huggingface.co/cstr/mini-bs-roformer-GGUF)). CLI `--separate`, HTTP server and Session (#562).
  - On Apple Metal it currently runs on the CPU, because the Metal path loses most of the signal (#568). Set `CRISPASR_MINI_BS_ROFORMER_METAL=1` to opt back in for debugging.
- **SCNet and SCNet-large** native separation (#566): strict 372-tensor conversion, F16/Q8_0/Q4_K, CLI/server/Session. Qualified against the original Python model on every stage, waveform and speed gate for both sizes. The upstream MIT notice and the author's permission for the checkpoints are retained.

## Recognition

- **MOSS-Transcribe-Diarize**: Session and CLI now forward the beam size and offer optional seeded sampling (temperature → top-k → top-p, in HF order); greedy stays the default (#564). New CLI aliases `--top-k` and `--top-p`.
- **moss-audio**: encodes only the real audio instead of padding every clip to 30 s, so time markers and event times on short clips are correct (#569). `CRISPASR_MOSS_AUDIO_PAD30=1` restores the old path.
- **Haas-ja-v1.0** with the KDA predictor in the parakeet runtime (#561).
- **parakeet-unified-en-0.6b**: the converter now takes the model's own tokenizer (#567). The published GGUFs were already corrected in place; re-download if your copy decodes to nonsense.

## Language identification

- LID models are no longer downloaded silently. A missing LID model is fetched only with `--auto-download` or after a TTY prompt; otherwise LID is skipped with a hint (#570). moss-audio with `--prompt` skips LID, since the result would be unused.

## Voice conversion and synthesis

- **MeanVC2** streaming zero-shot voice conversion, with C ABI, registry and bindings.
- ArkTTS Metal acceleration and codebook clamping.

## Performance

- CPU speedups across mel/fbank/LFR, conformer attention, CTC alignment and beam search, wav2vec2, SenseVoice, Paraformer, FireRed, F5-TTS, VoxCPM2, Mel-Band RoFormer, MeanVC2, MioCodec, diarization, CQT and RVQ/iSTFT. Each comes with a paired A/B speed and quality bench and a serial fallback switch.
- CosyVoice3 vocoder FASTCONV path.

## Fixes

- Lint Deep (cppcheck) is green again. Fixed: an ArkTTS virtual call in the destructor, a Mini-BS lambda, a parakeet null check, and the sampler comparator.
- Windows build fix for Mini-BS-RoFormer (`setenv`).
