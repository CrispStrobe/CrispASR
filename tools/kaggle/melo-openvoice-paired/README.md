# Melo and OpenVoice original versus native timing pilot

The prepared private Kaggle pilot times **OpenVoice conversion** on the two
accepted deterministic `tau=0` source/reference pairs. Its inputs come from
the accepted conditioning run, whose source speech was produced with the
qualified F16 Melo model and F16 BERT companion. The exact current-source CUDA
application checkpoint is pinned by SHA-256. The original OpenVoice source is
checked out at `74a1d147b17a8c3092dd5430504bd83ef6c7eb23` and the
original model revision is read from the hashed reference archive, then its
checkpoint hash is checked against that archive. No new source speaker,
recording or identity is introduced.

Both arms receive the same 22,050 Hz source and reference PCM, source speaker,
`tau=0`, and two threads. The original PyTorch arm runs its reference encoder,
source encoder, flows and HiFi-GAN on CUDA. The native arm runs its reference
encoder and flows on **CPU**, with **CUDA HiFi-GAN**; that is the accepted native
architecture, not a full CUDA converter. One warmup and three measured calls
per arm are made after each model is loaded. Both arms cache the reference
embedding by its PCM after the first warmup, matching the native C API's
reference cache. The measured region includes source STFT, encoder, flows and
vocoder, with the cached reference encoder excluded. This is a warm, in-process
conversion comparison; load, Melo synthesis, resampling and CLI marking are
outside the timed region. The original checkpoint is F32 and the native
OpenVoice GGUF has F32-sensitive tensors. The F16 qualification applies to
the upstream Melo/BERT source-speech generation. OpenVoice conversion is not
autoregressive, so this pilot reports whole conversion calls and no invented
cross-stack per-step speedup.

Each arm's in-memory output must be finite, have the accepted sample count,
and match the accepted original waveform at cosine ≥0.99. The worker writes
only aggregate JSON; no input or generated audio, speaker embeddings, tokens,
or private model weights are copied into Kaggle output. The kernel remains
private. Do not publish the full output or raw logs without review.

**Melo full-original comparison is pending.** The existing pinned original
Melo controls (`melotts_inputs.py`, `melotts_duration.py`,
`melotts_upstream.py`, `melotts_decoder.py`) prove individual boundaries using
converted F16 weights and captured inputs. They do not execute the full
upstream Melo synthesis API with its original checkpoint and matched text,
speaker, BERT, noise, duration and vocoder settings. Native Melo CLI timings
and original boundary timings must not be reported as a full-model speedup.
The official MIT `myshell-ai/MeloTTS-English-v2` repository at
`a53e3509c4ee4ff16d79272feb2474ff864e18f3` supplies `checkpoint.pth`
(SHA-256 `794226eb7c1745f3ca281b290613d5f39aa5b0d3b16a117009966f4aaf184757`)
and `config.json`; the pinned original source at
`209145371cff8fc3bd60d7be902ea69cbdb7965a` has the full `melo.api.TTS`
path. This makes a hosted full-original run feasible in principle. Before
timing it, prove that checkpoint generated the qualified GGUF, and pin the
original BERT/tokenizer/G2P dependencies. Pass explicit native settings to
`TTS.tts_to_file`: speaker `EN-US`/native id 0, `sdp_ratio=0`,
`noise_scale=0.667`, `noise_scale_w=0.8`, `speed=1.0`, and seed 42; upstream
API defaults (`0.2`, `0.6`) differ. Check sample rate and independent speech
quality on both accepted phrases before reporting whole generation times.
Keep reference/generated speech private, and disclose original F32 versus
native F16 model/BERT precision and PyTorch versus ggml RNG differences.

The pilot has not yet been submitted. Before any submission, refresh the
metadata-aware Kaggle GPU slot check, review private attachments and source
hashes, and retain all terminal source/log/output evidence before a retry.
