# CrispASR v0.8.33

Two new TTS backends, and a release whose main theme is measurement finding
things that passing tests could not.

**Supertonic-3** (`--backend supertonic`, OpenRAIL-M) is non-autoregressive
flow-matching at 44.1 kHz with ten built-in voices. Every per-stage diff against
the ONNX reference is at or above cos 0.999996, and the TTS→ASR roundtrip scores
1.00 on CPU and 1.00 on GPU against a control arm that also scores 1.00.

**FireRedTTS3** (`--backend fireredtts3`, Apache-2.0) is zero-shot voice cloning
over continuous 64-d 25 Hz RedAE latents with a Qwen3 backbone. It closes the
last of the three models in #377.

The second theme is Sidon and Zonos, both reported broken and both fixed — but
neither in the way the first attempt assumed.

---

## New backends

- **supertonic** — Supertonic-3, 44.1 kHz, non-AR flow matching, 10 voices.
  OpenRAIL-M (attribution + use restrictions), gated in the registry.
- **fireredtts3** — Qwen3 backbone + RedAE autoencoder + CAM++ speaker encoder,
  24 kHz, zero-shot cloning. Apache-2.0.

## Fixed

- **#431 sidon — a 60 s file was refused.** The 3000-frame cap was a hardcoded
  number with nothing behind it; it now derives from a memory budget
  (`CRISPASR_SIDON_MEM_BUDGET_MB`), giving ~4000 frames. A first attempt
  *windowed* the predictor, which made ASR transcripts look better while moving
  measurably **away** from the reference (0.991/0.987/0.974 at 30/50/62 s
  against a flat 0.997 for the faithful path). That was retracted. Attention has
  no receptive field, so no amount of context makes a windowed core exact.

  For genuinely long audio, `CRISPASR_SIDON_SPLIT=1` restores it as N **exact**
  passes cut at energy minima, each fed real neighbouring audio as context.
  Verified on a 154 s file: output length exactly 3x the input sample count.

- **#435 zonos — garbage for non-Latin scripts.** Three defects, and the first
  is why the initial fix never fired: `phonemize_espeak()` returned a non-empty
  string when espeak was absent, because it appended trailing punctuation to an
  empty result, and the caller tested `!ipa.empty()`. Also: `set_language()`
  matched exactly, so a bare `"en"` never matched `"en-us"`; and the language
  was latched even when the switch failed. Russian now refuses loudly instead of
  emitting 0.9 s of confident noise.

- **#432 / #433 API** — `set_voice_samples()` (voice from an in-memory buffer),
  `backend_caps()` / `list_backends_with_caps()` (per-backend verbs), and
  `detect_backends()`, which names **every** backend that can open a file — a
  voxcpm2 GGUF opens as both `voxcpm2-tts` and `voxcpm2-vae`.

- **supertonic q4_k was unloadable.** `crispasr-quantize ... q4_k` emits Q4_0 for
  tensors that do not suit a K-quant block, and the loader accepted only
  F32/F16. Not slow or degraded — dead. All of #434's gates run F16, so the
  quantised artifact had never been loaded.

- **CAM++ `seg_pooling` divided the partial tail window by the kernel size**
  instead of by its own width, in code shared by five backends. For fireredtts3
  it was decisive: `spk_emb` went 0.268 → 0.999452.

  Confirmed for chatterbox, confucius4 and dots-tts as well, each against its
  own PyTorch upstream, with the fbank pinned identical so pooling is the only
  variable.

  **cosyvoice3 is tracked as unresolved.** Its upstream is `campplus.onnx`, and
  two runs on the same clip disagree — one onnxruntime build matches the new
  divisor, another matches the old one exactly. Independently of that, the eight
  speaker embeddings baked into the shipped voice bank match the OLD divisor, so
  the `--voice ref.wav` path and the baked bank now describe the same voice
  slightly differently (cos ~0.998). End-to-end passes 8/8 on both, and
  cosyvoice3 ships with the same convention as the other four for now. Closing
  it needs the onnxruntime version pinned on both sides and probably a re-bake
  of the voice bank.

- **zonos: built-in G2P is the default for en/de/fr**, so espeak-ng (GPL-3.0) is
  no longer needed for them. Flipped only on measured evidence — en 1.00 vs
  1.00, de 1.00 vs 1.00, fr **0.95 vs espeak's 0.84**. Spanish is deliberately
  not flipped: its espeak baseline was itself 0.57, so a higher score proves
  nothing.

- A built-in **Russian** G2P ships (812,953 entries, CC-BY-4.0) and is wired
  into piper's tier, but `ru` stays on espeak for zonos — zonos-v0.1 does not
  list Russian, so its roundtrip cannot certify a G2P for it in either
  direction.

- `voxcpm2-vae` is reachable by `-m auto`. Piper's Windows HTTP gate now scores
  word overlap over the whole prompt instead of matching one phrase.

## Tooling

- The shipped-library wiring check **had never executed in CI** — that job
  builds no shared library, so it printed `skipped` and went green. It now runs
  where a shared `libcrispasr` exists. It also resolved the library by a
  hardcoded path unrelated to the binary under audit, and used `nm -gU`, which
  on binutils 2.38 means `--unicode` and fabricates a missing-backend list.
- A staleness guard: a binary older than the sources it is judged against is now
  reported as such, instead of naming every backend added since.
- An expired HF token is dropped rather than sent, because HF answers 401 to a
  bad token even for a public repo.
