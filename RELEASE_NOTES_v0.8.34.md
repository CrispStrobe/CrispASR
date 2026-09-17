# CrispASR v0.8.34

One new TTS backend, the Linux CUDA tarballs that v0.8.33 was supposed to ship,
and a release whose theme is **capabilities that were only claims and guards
that were only comments**.

Several things in this tree announced that they did something and then did not:
a `--temperature` flag the decoder never read, a `--voice` flag with no speaker
encoder behind it, a memory budget that defaulted to "disabled" so the check it
guarded always passed, and a length check that was never called from anywhere.
None of these showed up as a failing test, because each one's failure mode is
silence — a flag accepted and ignored, a guard that returns "fine" for every
input. Two of them reached users as bug reports before anyone noticed.

The second theme is the one that keeps paying: an experiment is only worth its
controls. The step-graph cache in here is a **negative** result — measured,
written down, and deliberately not extended.

---

## New backend

- **bt2-tts** — MediaTek **Breeze-TTS-2** (#412). T5Gemma2 text encoder +
  Sesame-style backbone/depth AR decoder over the qwen3-tts-tokenizer-12hz codec
  (16 codebooks @ 12 Hz) → 24 kHz. Plain TTS, Voice Clone (`--voice ref.wav
  --ref-text <transcript> --i-have-rights`) and Voice Design (natural-language
  style instruction). Aliases: `bt2`, `breeze-tts-2`.

  **Non-commercial only** — BreezeBlue Research and Non-Commercial License
  v1.1. The grant in §2 is disjunctive (Research *or* Non-Commercial), §5(k)
  expressly permits quantization, and §1.2 excludes Apache-2.0 code, so the port
  itself is unrestricted; the GGUFs are Derivative Models under §1.3 and inherit
  the terms. Gated in the registry.

  The registry default is **q4_k**, decided by listening rather than by a
  code-level metric:

  | quant | size | normalised WER | frame-0 codes vs oracle |
  |---|---|---|---|
  | q4_k | 2.05 GiB | 0.0357 | 1/16 |
  | q8_0 | 3.19 GiB | 0.0278 | **16/16 exact** |
  | f16 | 5.32 GiB | 0.0000 | 16/16 |

  q8_0 reproduces the oracle's codes **exactly** and still makes a real word
  error; q4_k matches 1/16 and makes one too. Code exactness did not predict
  audio quality, so no code-level metric was promoted into a quality gate, and
  q8_0's extra 1.14 GiB buys exactness that never reaches the audio. Raw WER
  ranked the two quants the other way round — the ASR writes "seventeen" as
  "17" and Americanises British spellings — and normalising flips the order.
  n=4 separates f16 from both quants but **cannot** separate q4_k from q8_0,
  and that claim is not made.

## New artifacts

- **Voxtral Mini F16 GGUFs** (#437) — `voxtral-mini-3b-2507-f16.gguf` (8.72 GiB)
  and `voxtral-mini-4b-realtime-f16.gguf` (8.27 GiB), plus Q8_0 registry rows.
  Not a port: both converters always emitted F16, the publish step was simply
  never in the recipe.

  Verified end-to-end rather than by header read-back: a kernel resolved both
  files **through the registry**, downloaded, loaded and transcribed them at
  word-F1 1.000, with the shipped q4_k files as control arms. The F1 metric
  self-tests before it judges anything — empty output, repetition loops,
  wrong-language hallucination and a truncated half-decode all score below
  threshold, while punctuation- and case-variant transcripts still score 1.000
  (separation margin +0.375 against 0.8). Four arms all scoring 1.000 is also
  what a broken comparator looks like; this is how we know it is not that.

## The release fix v0.8.33 needed

- **Linux CUDA tarballs.** v0.8.33 published 41 assets, and the CLI CUDA builds
  were not among them: `crispasr-linux-x86_64-cuda.tar.gz` is absent (the
  library-only `libcrispasr-linux-x86_64-cuda.tar.gz` did ship), and there are
  no `cuda13` artifacts at all. The legs failed to link with `undefined
  reference to ggml_backend_is_cuda` — backend-module symbols are not linkable
  under `GGML_BACKEND_DL`. Replaced with `ggml_backend_name()` string matching
  at the two call sites (`src/mimo_tokenizer.cpp`, `src/parakeet.cpp`).

  The fix is in, but **this is the claim to verify against the actual v0.8.34
  assets before announcing it** — it is the reason this release exists, and a
  release that ships without those tarballs again would repeat v0.8.33 exactly.

## Fixed

- **#441 parakeet — intermittent SIGSEGV on a 47.5-minute input.** The kernel
  refused a 123.6 GB allocation and the result was dereferenced. The guard was
  already implemented and never armed: `resolve_strategy()` estimates the
  single-pass encoder's O(T²) relative-position bias and switches to the
  streamed encoder when it will not fit — but an unset budget meant "policy
  disabled", and the predicate returns "fits" for *any* input at budget 0. So
  the default path was "estimate it, then ignore the estimate and allocate
  anyway". The budget now defaults to half of `MemAvailable`; an explicit `0`
  still disables it.

  Why it was intermittent is what makes the diagnosis certain: the reporter saw
  RSS steady at ~1.0 GB *in a run that completed*. The buffer is never fully
  written, so Linux overcommit granted it lazily about half the time and refused
  it the rest. Their own correction — "it finished" — is what made the mechanism
  legible.

- **#431 sidon — a 60 s file was refused.** v0.8.33 raised the cap and added
  `CRISPASR_SIDON_SPLIT=1`; the reporter's one-minute clip still produced 3075
  frames against a 3000-frame cap, so it still failed, and the advice printed
  was "split the audio" — asking the user to do by hand what the runtime already
  does exactly. Splitting is now the **default**. The branch is only reachable
  when the frame count already exceeds the cap, so it can only turn a failure
  into a result; input that fit before takes the identical single-pass path.

- **#439 m2m100 / wmt21 — output collapses into repetition.** Two decode
  defaults disagreed with the checkpoints' own `config.json`. All three models
  in the family declare `num_beams: 5` and `max_length: 200`; we used beam 1
  (greedy) and 256. Greedy on a 4.7B translation model collapses exactly as
  reported, and 256 is why the report shows 258 tokens. Beam search is not a
  tuning preference for these checkpoints — it is the decode they were released
  with. `--beam-size 1` still selects greedy.

- **#435 zonos — non-Latin scripts.** The substance was fixed in v0.8.33; this
  release removes the remaining false promise (below). Still open pending
  confirmation from a release build.

- **nemotron GPU drift on P100-class hardware.** Conformer attention used fused
  `ggml_flash_attn_ext` with no `set_prec`, and ggml's fused flash accumulates
  the KQ product in F16 while `set_prec(F32)` is *silently ignored* on sm_60.
  Manual F32 SDPA is now the default (correct on every backend), with the fused
  path opt-in via `CRISPASR_NEMOTRON_FLASH=1`, read per call so it is A/B-able
  on a live context.

- **indextts `AA_SCALAR` was cached in a function-local `static`**, so the
  SIMD-vs-scalar A/B knob was fixed for the process: both arms of an in-process
  A/B silently ran the same path.

## Capabilities that were only claims

Declaring a capability **suppresses the CLI's own "unsupported by this backend"
warning**. That is what makes a false claim worse than no claim: the flag is
accepted in silence and ignored, and the user has to establish from outside that
nothing happened. #369 records a reporter spending real time proving that
`-tp 0.8` under different `--seed` values returned character-identical output.

- **voxtral `--temperature`** was declared and never read — the greedy branch was
  a hand-rolled argmax loop. Now **implemented**: it routes through
  `core_greedy_decode::sample_temp` with the session's temperature and seed, so
  the capability is restored on the condition #369 sets. Per-token confidence now
  comes from the token actually picked rather than the argmax; once sampling is
  on those differ.

- **The C API decode path never set `dec_cfg.temperature` at all.** Every
  backend decoding through `core_greedy_decode` from the library (bindings,
  server — not the CLI, which has its own loops) ran at a hardcoded 0 while
  advertising the knob. Plumbed for qwen3-asr and granite; a no-op unless a
  caller passes temperature.

- **zonos `--voice`** declared `CAP_VOICE_CLONING` with no route behind it:
  `zonos_tts_set_voice()` is a stub (the ResNet293 speaker encoder is not
  ported) and `set_speaker_embedding()` is called from nowhere. `--voice` was
  accepted, warned about once, and answered with a **random speaker**. Cap
  dropped, docs corrected, and the warning no longer says "failed to load voice
  from '<file>'" — which reads as a bad wav and is what sent #435's reporter to
  check their file.

One correction in the same sweep: **voxtral4b's `CAP_TEMPERATURE` was briefly
removed and restored.** It was removed on the evidence that `temperature`
appears nowhere in `src/voxtral4b.cpp` — worthless evidence, because that
backend's decode loop lives in its CLI *adapter*, which sets temperature, seed
and frequency penalty before calling `sample_temp`. The capability was true the
whole time. "Zero references in the backend .cpp" cannot distinguish "nothing
reads it" from "the reader is in another file".

## Measured and rejected

- **Bucketed cached decode-step graphs** (`CRISPASR_VOXTRAL_STEP_CACHE`,
  default **off**). Correct but not worth it here, and the numbers are recorded
  so the next person does not re-derive them:

  | arm | decode ms | ms/step |
  |---|---|---|
  | cache off | 4934 | 189.8 |
  | cache on, width 16 | 4993 | 192.0 — 1.2% **slower** |
  | cache on, width ≥ max_ctx | 17767 | 683.3 — **3.6× slower** |

  All three bit-identical. The cache saves ~2 ms of graph prep per step; a
  voxtral decode step costs ~190 ms, so prep is ~1% of it. funasr's win was real
  because its step is far cheaper. The question was never "which width" but "is
  prep a meaningful fraction of a step at all" — for a 3B/30-layer decoder it is
  not, so this is **not** wired into gemma4-e2b, granite or voxtral4b. The
  max_ctx row is the useful half: it confirms the fixed-Lk trap on a second
  model, far worse than the +69% funasr measured.

## Behaviour changes worth knowing

- **m2m100 / wmt21 now default to beam size 5.** Better output, roughly 5×
  slower. `--beam-size 1` restores the old behaviour.
- **sidon splits long input instead of refusing it.** `CRISPASR_SIDON_SPLIT=0`
  restores the refusal.
- **parakeet may now choose streamed encoding** on long audio where it
  previously attempted a single pass. `CRISPASR_PARAKEET_MEM_POLICY=single`
  forces the old path; `CRISPASR_PARAKEET_VRAM_BUDGET_MB=0` disables the policy.
- **nemotron GPU attention is manual F32 by default**, which is slower than
  fused flash. `CRISPASR_NEMOTRON_FLASH=1` opts back in.
- **zonos `--voice` is now refused** rather than silently ignored.

## Also

- The regression nightly had been failing 4 runs in 5: an **expired** `HF_TOKEN`
  made HuggingFace answer 401 for repositories that need no authentication at
  all. A stale credential is worse than none; rejected tokens are now dropped
  and the fetch retried anonymously.
- `bt2-tts` was merged reachable from the CLI and invisible to the library — the
  backend-wiring audit caught it, and the C API dispatch, `available_backends`
  roster and feature matrix are now wired.
- The feature matrix regenerates from a live `crispasr --list-backends-json`, so
  it is only as correct as the binary that produced it; a stale build silently
  deletes backends from it.
