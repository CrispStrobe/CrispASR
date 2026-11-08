# v0.8.52 — Ultra aligner, Qwen3-TTS split mode, HIP fallback

## Alignment

- The Qwen3-ForcedAligner-Ultra base model is published as an F16 GGUF and in the model registry: `crispasr --align-only -am qwen3-forced-aligner-ultra --auto-download ...`. Its JFK word timestamps match the upstream `qwen-asr` implementation exactly. Ultra checkpoints named after the upstream repo also select the Qwen aligner path (#579).
- Experimental: the optional Ultra local boundary-refinement head, which requires an explicit base model. See `docs/qwen3-ultra-local-head-port.md`; its precision is not yet qualified (#579).
- The nightly regression now covers forced aligners (Qwen3 0.6B, Ultra, canary-ctc), comparing word timestamps with pinned references.

## Speech synthesis

- Qwen3-TTS: `CRISPASR_QWEN3_TTS_VULKAN_CPU=1` together with `CRISPASR_QWEN3_TTS_CODEC_GPU=1` (or `CODEC_VULKAN=1`) runs the talker and code predictor on CPU and keeps only the codec on Vulkan. This suits shared iGPUs; voice-clone reference encoding works in this mode (#585).
- TTS: a bare `--voice-dir` voice name now resolves to the recording and its transcript for cloning backends in the CLI, HTTP server and Wyoming (#584).

## Separation and server

- SCNet: very short inputs are zero-padded instead of aborting. The server's separate endpoint returns 400 for empty audio and 413 above `CRISPASR_SERVER_SEPARATE_MAX_SECONDS` (default 1800; 0 disables).

## Packaging

- The Linux HIP tarball loads the GPU backend as a module. Without ROCm 6.x and hipBLAS it now runs on the CPU instead of failing with `libhipblas.so.2: cannot open shared object file` (#586).
