# CrispASR — Pending work

## CLAIMED 2026-08-13 — PR #347 GGUF weight-mapping release review

Worktree: `.claude/worktrees/review-pr-352`.
Review PR #352 end to end, validate that its long-form routing and gap repair
do not regress any language path, add targeted unit/live coverage where needed,
and merge or improve the change after local/SSD validation.

## OPEN 2026-08-18 — stb_vorbis heap overflow on untrusted audio (security)

Found incidentally by `linux-fuzz-smoke` on PR #371, which does not touch that
code — fuzzing is stochastic and it happened to surface there. NOT that PR's
fault, and it reproduces from the audio fuzzer, not from anything in the diff.

    ==6684==ERROR: AddressSanitizer: heap-buffer-overflow
    WRITE of size 13174835200 at 0x7f29bad7d000
      #0 memset
      #1 start_decoder(stb_vorbis*)        examples/stb_vorbis.c:3683
      #2 stb_vorbis_open_memory            examples/stb_vorbis.c:5141
      #3 stb_vorbis_decode_memory          examples/stb_vorbis.c:5419
      #4 crispasr_webm_decode(...)         src/crispasr_audio.cpp:1405
      #5 crispasr_audio_load               src/crispasr_audio.cpp:2801
      #6 LLVMFuzzerTestOneInput            tests/fuzz/fuzz_audio_load.cpp:40

    0x7f29bad7d000 is located 0 bytes after a 289933312-byte region
    allocated by setup_malloc(stb_vorbis*, int)  examples/stb_vorbis.c:960

A 13 GB `memset` past a 289 MB allocation: the size computation in
`start_decoder` overflows or is not validated against the allocation
`setup_malloc` actually made. Reachable through `crispasr_audio_load`, i.e. on
ANY caller-supplied audio file — the CLI, the HTTP server's upload path, and
every binding. That makes it a memory-safety issue on untrusted input, not just
a fuzz curiosity.

Severity note, honestly: a 13 GB write will fault almost immediately in
practice, so the realistic outcome is a crash (DoS) rather than exploitable
corruption. Worth fixing regardless, and worth checking whether a smaller,
more controllable overflow is reachable from the same path.

Next steps: reproduce locally with a fuzz build
(`-DCRISPASR_FUZZ=ON -DCRISPASR_SANITIZE_ADDRESS=ON`), minimise the input, then
decide between bounds-checking `start_decoder` and pulling a newer stb_vorbis.
Check whether upstream stb has already fixed it before patching a vendored copy.

## Start here

Live work only. Completed threads move to `HISTORY.md`; technical deep-dives to
`LEARNINGS.md`.

**Before you pick something up:** re-read this section on `origin/main`, add a
`## CLAIMED <date> — <what>` block naming your worktree, and **push that claim
to main before you start**. Several agents run here at once; a claim that lands
with the work is a claim that did nothing. Delete it when the work lands, or if
it goes stale for more than a day.

## CLAIMED 2026-08-13 — #350 parakeet non-JA long-form drops whole spans

Worktree: `.claude/worktrees/crisp-asr-issue-filing-0ca183`.
Two defects behind one symptom on parakeet-tdt-0.6b-v3 (30-300 s, non-JA):
the unified dispatch read `chunk_seconds = 0` (documented "per-model defaults")
as "not chunked" and ran an explicitly chunked session call as one unbounded
pass; and the TDT decoder drops whole spans past ~30 s at any routing, so no
cap alone fixes it. Fix = a `chunked_requested` flag that caps such calls at
the reliable window, plus a shared gap-fill repair pass over holes in the word
timeline. Reporter's clip: 66 % → 95 % coverage (CLI), 66 % → 100 % (session
chunked API); jfk×21 regression guard added.

## CLAIMED 2026-08-13 — #344 MOSS valid-frame metadata review and validation

Worktree: `.claude/worktrees/fix-344-moss-valid-frame-metadata`.
Audit PR #345, verify the additive C ABI and failure contracts, and run the
hermetic plus available model-backed/live tests before deciding whether any
changes are needed.

## CLAIMED 2026-08-13 — #337 Qwen3-TTS HIP prefill divergence (second pass)

Worktree: `.claude/worktrees/fix-337-qwen3-tts-hip`.
The first-pass single-backend allocator fix was disproven on RX 7900 XTX:
the reporter confirms it is byte-identical to the scheduler path and CPU-vs-HIP
full-frame replay still diverges at prefill. Reproduce with the diff harness,
bisect the first divergent talker tensor/op under identical teacher-forced
history, then fix and validate on the real HIP path via the Kaggle regime.

## CLAIMED 2026-08-13 — #344 MOSS valid-frame metadata in stable C ABI

Worktree: `.claude/worktrees/fix-344-moss-valid-frame-metadata`
(branch `fix/344-moss-valid-frame-metadata`).
Additive MOSS encoder/tap/adapter valid-frame metadata for the downstream
MOSS-Music-8B-Thinking feature pipeline: `moss_audio_plan_chunks`,
`moss_audio_compute_mel_meta` (preserves pre-pad `T_mel_actual` — never inferred
from padded zeros/floor), and `moss_audio_run_encoder_meta` (existing chunk loop
reused, caller-allocated per-chunk valid counts, adapter output dim reported from
GGUF weights, fail-closed llm_hidden vs adapter-row check). Existing
`moss_audio_*` symbols unchanged. Hermetic CPU test + live differential test.

## LANDED 2026-08-10 — voxtral-tts pre-tokenizer parity (c69ac61b, from #338)

Carried out of #338 after it closed. Reporter measured 5/57 vs `mistral-common`;
reproduced and fixed. **0 mismatches across ~16k strings**, 0 out-of-range ids.

Three defects in `tekken_pre_tokenize`, now in `voxtral_tekken_vocab.h`:

1. A whitespace run swallowed its last character — the letter and punctuation
   alternatives each open with an optional leading slot, so `a  b` is
   `["a"," "," b"]`. The slots DIFFER (letter takes any non-alnum, punctuation
   takes a literal space only, `\p{N}` takes none), which is why this needed
   the reference rather than a reading of the pattern.
2. Every byte >= 0x80 counted as a letter. Bites on punctuation ADJACENCY, not
   on letters — BPE recovers the same ids for accented text, which is why my
   first guess at this was wrong. ASCII-only fuzz 0/4000; mixed 233/4000, 232
   containing multibyte punctuation.
3. `\p{N}` is one codepoint; digit runs were grouped. INVISIBLE to id-level
   parity on this checkpoint (no multi-digit piece, so BPE re-splits to the same
   ids). Only the generated split test caught it.

**Two checks, because they fail on different things** — the durable lesson:
- `tests/test-voxtral-pretokenize.cpp` — hermetic, pins the SPLIT from the
  published `config.pattern` via Python `regex`. Caught (3).
- `tools/check-voxtral-tokenizer-parity.py` — live, pins the IDS through BPE +
  the #338 bound vs `mistral-common`. Caught (1) and (2).

Reporter was told on the issue in case his in-flight PR overlaps; his harness is
still wanted (real corpora, other languages, mistral-common version drift).

## LANDED 2026-08-07 — #13273 omnivoice language knob, dead on three surfaces

Full write-up: `docs/omnivoice/PLAN.md` §LANDED 2026-08-07. Fixed the CLI
adapter (language applied only in `init()`, so a persistent server could never
change it per request), the session C-ABI arm (#329's bug one backend over), and
the runtime (no `_resolve_language()` mirror, so `de-DE` or a typo went into
`<|lang_start|>` verbatim). Guards: `tests/test-omnivoice-lang.cpp` — predicate
AND joins, all four join assertions watched red first.

**Two things to know before touching this again:**

- **A/B omnivoice on CODES, never on the WAV.** Output is watermarked and
  carries a spoken disclaimer, so `cmp` on audio differs for every render
  including two that should match. Use `CRISPASR_OMNIVOICE_DUMP_CODES` with
  `--no-spoken-disclaimer --accept-marking-responsibility`, and include a
  control arm whose expected answer is IDENTICAL.
- **The accent half of the report is NOT fixed and may not be fixable here.**
  whisper LID cannot separate tagged from untagged output (accent-robust by
  design; the one sentence that moved was noise), and OmniVoice has no
  cross-lingual drop-ref path to port #329 into. Needs a listener, not another
  metric.

**SE-side gap, no longer blocking:** `OmniVoiceCrispAsr.Speak()` accepts the
language and never puts it in the payload. Worked around on our side —
omnivoice guesses the language from the target text when nobody supplies one
(`CRISPASR_OMNIVOICE_AUTO_LANG`, default ON; explicit always wins), verified
byte-identical to an explicit `-l de` against the exact SE request shape.
Sending the field is still better (exact, covers languages the detector does
not); snippet is on the issue, but nothing here waits on it.

**Honest residual, closed cheaply:** the style-token live test pins ids from a
tokenizer run; a red test alone could not say which side moved. Now it can —
the test header records the upstream revision (`c5fdb5cc`) + tokenizer.json
sha256 and the one-command disambiguation, all 9 pins re-derived byte-identical
2026-08-07. Vendoring the 7 MB tokenizer.json remains not worth it.

### Ready to take — scoped, unblocked, nobody on them

| # | Task | Size | Where |
|---|---|---|---|
| 1 | **Delete the duplicated fallback copies** instead of keeping 14 files in sync | M | §"OPEN follow-ups from #300 / #308" item 3 |
| 2 | **`CAP_PUNCTUATION_NATIVE` audit** for `lfm2-audio`, `fastconformer-ctc`, `wav2vec2` | S | same section, item 2 |
| 3 | **#326 speaker-count estimator** — the last diarization accuracy item | M | §"NOW — #326" |
| 4 | **VAD + mel front-end parallelization** — remaining backends | M | §"NOW — VAD + mel" |
| 5 | **Diff harness: per-step talker logits** — validates the sampler over verified logits | M | §"Diff-harness extensions" |
| 6 | **Diff harness: replay-token dual-mode** — makes sampling-dependent stages diff deterministically | M | §"Diff-harness extensions" |

### Blocked, and on what

| Task | Blocked by |
|---|---|
| **Stamp the kartoffel repos** (§below) | Disk. `/Volumes/backups` has ~1.8 GB free of 1.9 TB; `ai/cache` alone is 347 GB. Needs ~15 GB. Everything else it needs is built and proven. |

### Machine state a newcomer will trip over

- **Disk is effectively full** (above). It is also a hazard: several agents build
  here concurrently and a full disk kills processes rather than failing cleanly.
- **Load spikes to 100+.** Time-based test assertions and tight ctest timeouts
  fail here for reasons that have nothing to do with the code. A `TIMEOUT` is a
  backstop against a hang, not an assertion about speed — size it for the worst
  machine, not the median one.
- **Worktrees need `git submodule update --init --recursive ggml
  third_party/c2pa-audio`.** `ggml` is a submodule, so a fresh worktree has it
  empty and cannot configure, let alone build.
- **`ls build/src/libcrispasr*.dylib | head -1` picks a STALE versioned dylib.**
  Use `ls -t`. Two binding-parity tests "failed" on this until the tell was
  noticed: another agent's brand-new symbol was missing too.
- **CI runs are cancelled constantly, and `cancelled` is not `success`.** With
  several agents pushing to `main` every few minutes, each push cancels the
  previous commit's in-flight Lint/CI/Ruby — a run can sit at `cancelled` all
  day and never verify anything. Do not chase green on HEAD. Verify locally
  (`ctest -L unit`, `tools/format.sh --check`, `tools/check-readme-langs.py`,
  `tools/check-kaggle-harness-sync.py`) and treat a LATER run that *contains*
  your commit as the real signal. Gotcha when scripting that check: `gh run
  list` returns a full 40-char `headSha`, so comparing an 8-char prefix matches
  nothing and makes a wait-loop exit instantly, looking like "it finished".

### Conventions worth knowing before you write a test

Earned the hard way; each cost a real bug getting through.

1. **A gate CI cannot run is a gate that ships wrong.** Compliance and policy
   logic belongs in weight-free headers with unit tests, not only behind a live
   server.
2. **Prove the gate can go red.** Re-introduce the thing it forbids and watch it
   fail. Every guard in `tests/test-compliance-wiring.cpp` and
   `tests/test-copies-in-sync.cpp` was verified this way — and that pass has
   caught more broken *tests* than broken code.
3. **Assert the token that only exists when the behaviour does.** Never one that
   also appears in prose, a help string, a comment, or an unrelated function.
   Four separate guards passed while the behaviour was gutted because they
   matched a substring that survived.
4. **Guard the joins, not just the predicate.** Every compliance failure in this
   repo's history happened while the pure predicate tests were green: missing
   call sites, unstamped bakers, ungated endpoints.
5. **A hand-maintained list needs a machine check that it is complete.** The
   copies-in-sync guard covered 1 of 14 files for months — not a wrong entry, a
   missing one.

## LANDED 2026-08-06 — #335 `Session::open()` could not open granite-speech

Root cause is NOT granite-specific. The `general.architecture` → backend table
existed **twice** — `examples/cli/crispasr_backend.cpp` pass 2 and
`crispasr_detect_backend_from_gguf()` in `src/crispasr_c_api.cpp` — and the two
had drifted by **113 architecture strings**. Every granite-speech GGUF carries
`general.architecture = "granite_speech"` (underscore; that is what
`models/convert-granite-speech-to-gguf.py` writes, confirmed by range-reading
the header of the published `granite-speech-4.1-2b-plus-q4_k.gguf`), and the
C-ABI copy only knew the hyphen spelling → detect returned `""` →
`crispasr_session_open` returned NULL for every binding. The CLI never noticed
because its **filename** pass matches `granite`+`speech` and short-circuits
pass 2 — so **auto-detect working in the CLI proves nothing about the
bindings**, and that is the durable lesson here.

granite was one of many: nemotron, moonshine, kokoro, piper, melotts,
sensevoice, funasr, paraformer, glm-asr, kyutai-stt, mini-omni2, csm, dia,
bark, speecht5, fastpitch, pocket-tts, gemma4-e2b, mimo-asr, voxtral-tts,
piano-transcription and more were CLI-only too — no binding could auto-detect
any of them.

Fix: one shared table, `src/core/arch_backend_map.h`, read by both surfaces.
`tests/test-arch-backend-map.cpp` pins it and drives the real C-ABI export over
a synthesised metadata-only GGUF (hermetic, no models) — deleting the
`granite_speech` row makes it fail with `"" == "granite"`, rc=0, which is the
reported bug exactly. Verified on the reporter's own artifact:
`crispasr_session_open()` returns a live handle reporting backend `granite` and
transcribes `samples/jfk.wav` correctly through the session ABI.

## LANDED 2026-08-06 — cohere: the language whitelist, + probe LID

Prompted by reading [bakrianoo/cohereX](https://github.com/bakrianoo/cohereX)
(a WhisperX-shaped Python wrapper around the same model). Nothing to take from
its runtime — we have the native port, CTC alignment and native diarization —
but it validates `-l` against the model's `config.json` and we did not.

**The bug.** Cohere Transcribe answers a wrong language *fluently* instead of
failing, and we had no whitelist, so an unsupported `-l` was accepted in
silence. Reachable without user error: `-l auto` is the CLI default and
whisper-tiny LID knows 99 languages against this model's 14 — or the Arabic
finetune's **two**.

**The thing I got wrong first, and the reason this needed metadata.** I assumed
`<|ru|>` was absent from the vocab and silently dropped by the prompt builder's
`remove_if`. Measured on the published Arabic GGUF: the tokenizer carries **183
`<|xx|>` tokens — the whole of ISO 639-1** — while the model supports two. So
every real code is well-formed and a vocab check catches nothing. On one 8 s
Arabic clip: `-l ru` added a hallucinated leading word, `-l ja` swapped the
quote marks for brackets, `-l de` changed the diacritics. All plausible, none
flagged. `config.json`'s `supported_languages` is the ONLY available signal —
which is why it now rides in the GGUF (`cohere_transcribe.supported_languages`)
rather than being inferred. (The `remove_if` backstop stays, but it only ever
fires for non-ISO input: `<|auto|>` genuinely is absent.)

Fixing it in `cohere_transcribe_ex` covers CLI + session ABI + server at once —
no three-surface edit needed.

**Probe LID** (`--lid-backend probe`, cohereX's `langid.py` idea): transcribe a
20 s clip once per supported language, score
`len × (1 + 3·text-LID-agreement) × distinct-token-ratio²`. Needs no second
model and **cannot return a language the model does not support**. Verified on
the Arabic finetune: Arabic clip → `ar` p=0.675, `jfk.wav` → `en` p=0.647.

Measured again on the **real 14-language base model** (`cohere-transcribe-q4_k`,
after republishing): the forced 14-way probe is **correct on both clips** —
`jfk.wav` → `en` (228, p=0.169), Arabic clip → `ar` (292, p=0.254). The ≤4
ceiling (`CRISPASR_COHERE_PROBE_MAX_LANGS`) is therefore a **cost** gate: 14
probes take 37 s on an M1 against ~1 s for whisper-tiny.

⚠ **I first documented the opposite and it was wrong.** The "forcing 14
candidates picks `fr`" result came from forcing a 14-language list onto the
**two-language Arabic finetune** — a model with no French, which *translates*
rather than degrading, and cld3 then confirms the translation at 1.00. The real
base model's `fr` probe code-switches instead ("Et so, my fellow Americans…",
agreement 0.00, score 57) and loses. I had flagged that extrapolation as
"suggestive, not conclusive" and shipped the conclusion anyway. **The lesson is
narrow and reusable: a forced/mismatched capability list does not simulate a
model that genuinely has that capability.** The scoring soft spot itself is real
and stays documented (a fluent translation *can* outscore repetitive truth); it
is just not what the candidate count controls.

### Closed since (2026-08-06, same day)

- **Metadata republished — all 10 GGUFs**, via `tools/gguf-add-cohere-langs.py`
  (tensors passed through, per-tensor sha256 verified) on Kaggle kernel
  `chr1str/cohere-langs-republish`. Confirmed independently by range-reading each
  live file's KV section: Arabic ×4 = `['en','ar']`, base ×6 = the 14. The fix is
  no longer inert; `CRISPASR_COHERE_LANGS` is now only for third-party GGUFs.
- **Converter runs end-to-end** (2104 tensors, 4.14 GB, key + `max_clip_s`
  present, output transcribes and enforces its own whitelist). Running it
  surfaced a pre-existing bug: it wrote `general.architecture` twice, which
  strict gguf-py versions reject.
- **Server and session C-ABI surfaces exercised**, not just compiled — probe
  fires and substitution warns on both.
- **UTF-8 truncation bug** in the probe log (`%.60s` cut mid-character, killed
  the Kaggle run via Python's stderr decode). Fixed + guarded.

- **`prefers_vad()` for cohere** — and it was a bigger deal than "a plausible
  quality win". The model transcribes SILENCE as speech: 10 s of pure digital
  silence returns *"And I'm going to go ahead and do that."*, and 20 s of
  trailing silence appends that same sentence to an otherwise perfect
  `jfk.wav` transcript. (Low-level noise and a 440 Hz tone produce nothing, so
  it is silence specifically.) A/B'd on real speech before flipping: on a 60 s
  FLEURS clip the un-VAD'd run also cut mid-sentence, garbled a clause and
  **dropped a whole sentence** the VAD run recovers — so it is content
  recovery, not just a silence guard. Uses the existing >30 s safeguard.
- **The cheaper probe** — encode once, decode per candidate. The two encode
  blocks were already brace-delimited, so it is two `{` → `if (!reuse_enc) {`
  with no re-indentation, and the cross-KV free/realloc lives *inside* the
  second block so skipping it keeps the allocation live. 14-candidate A/B,
  back-to-back and repeated in reverse order: **12 s → 4–5 s, byte-identical
  output**. Gated `CRISPASR_COHERE_PROBE_REUSE_ENC=0`.
- **`tools/gguf-add-merges.py`** — passed `sub_type=` to a `MetadataDetails`
  that the installed gguf-py does not have, so it raised TypeError and could
  not run at all. Now passes it only when the field exists; verified on a
  synthetic glm-asr GGUF (merges land ARRAY/STRING, both pair- and string-form
  inputs normalise, arch/vocab/tensors pass through).

### Open follow-ups

None tracked for cohere. Two things deliberately NOT done, with reasons:

1. **Fresh-conversion vs published-f16 tensor equality** was never checked (it
   would cost a 4.14 GB download to compare a converter run against an artifact
   produced by an older converter). The republished files are guaranteed
   byte-identical to their *pre-rewrite* selves by the tool's per-tensor sha256
   `--verify`, which is the property that actually matters.
2. ~~Silence-hallucination below the 30 s threshold~~ — **fixed** (99d8e60b),
   and not the way I expected. Measuring first showed the defect is narrower
   than "silence in a short clip": 23 s of speech + trailing silence is CLEAN.
   Only an all-silent span fabricates, which means it needed no VAD at all —
   just a digital-silence gate in `cohere_transcribe_ex`, free and with no
   extra model. Threshold sits below one int16 LSB so one non-zero sample
   disables it; the quietest real speech to hand peaks ~3800x higher.

## LANDED 2026-08-10 — #339 HIP: the bundler erased the RUNPATH it needed to read

v0.8.27 restored five of the six missing Linux tarballs and HIP still did not
package. Separate defect, and one the first fix could not have caught: the build
succeeds, then `check-bundled-deps.py` refuses the staged directory with
`crispasr needs libomp.so`.

`bundle-linux-runtime.sh` did its two jobs in the wrong order — rewrite RUNPATH
to `$ORIGIN`, THEN ask `ldd` what the binaries need. `ldd` resolves through the
binary's own RUNPATH, so erasing it first turns exactly those dependencies into
`=> not found`, and the copy loop's `grep '^/'` dropped them with the blank
lines. ROCm's clang links OpenMP against LLVM's `libomp.so` under
`/opt/rocm/lib/llvm/lib`, reachable only that way; gcc's `libgomp.so.1` is in
the default loader path, which is why six legs were unaffected and this
survived v0.8.27.

⚠ **The failing line printed the evidence and was read as progress.** The log
says `rpath crispasr: '$ORIGIN:…:/opt/rocm-6.3.0/lib/llvm/lib:…' -> '$ORIGIN'`
and then `rpaths normalised, 0 librar(ies) bundled`. The directory it needed
was in the string being discarded, and "0 bundled" was a count nobody had a
reason to expect to be non-zero.

⚠ **A green summary line over a dropped dependency.** The bundler reported how
many libraries it had copied and said nothing about the one it could not find;
`grep '^/'` filtered `=> not found` out with the blank lines. It is now fatal
there, naming the library, and consults the same exclusion list the copy loop
uses — otherwise `libcuda.so.1`, legitimately absent from a driverless CI
runner, would take down every CUDA leg.

⚠ **These scripts only ever ran inside a release job.** That is why two defects
in them shipped: there was no way to observe one without publishing a release.
`tests/test-bundle-linux-runtime.sh` now reproduces the whole thing with `cc`
and a private directory plus `-Wl,-rpath` — no ROCm, no GPU, no release — and
sits in the `unit` tier on every push. Red-verified against the v0.8.27 script
before being trusted. `patchelf` was added to the CI unit job for it, because a
SKIP reads exactly like a PASS in the ctest summary.

Dry runs of a single leg are now readable: `validate-version` compared VERSION
against the branch name when `tag` was empty and failed on every dry run, which
is the mode the input's own documentation recommends.

**Then I went looking for the same shape elsewhere, and found two more (#341).**
Both verified against the DOWNLOADED v0.8.27 assets, not inferred:
`libcrispasr-linux-x86_64-hip` needs `libomp.so` + `libhipblas.so.2`;
`crispasr-python-linux-{x86_64,arm64}` needs `libgomp.so.1` + `libblas.so.3`.
Neither carried them, and neither leg ran `check-bundled-deps.py` at all — the
python legs did the RUNPATH half inline with no closure step and no gate, so
`import crispasr` died in the loader on any host without OpenBLAS and gcc's
OpenMP.

⚠ **A tolerance that `exit(0)`s is not a tolerance, it is a stop.**
`verify-lib-bundle.sh` DID fail to dlopen the HIP bundle on `libomp.so` — and
`libomp` is on its EXTERNAL list, so it printed "dlopen deferred: external
driver absent in CI" and exited 0, skipping the rest of the verification too.
The list was written for gcc's versioned `libgomp.so.1` on the default loader
path and silently generalised to ROCm clang's unversioned `libomp.so`, which
exists only under `/opt/rocm/lib/llvm/lib`. When adding to an allowlist, ask
what a shipped artifact looks like if the entry is wrong.

⚠ **Two gates asking different questions is not two gates.**
`verify-lib-bundle.sh` gates INTRA-bundle resolvability — for every library the
bundle ships, every dependency whose soname is ALSO shipped must be reachable.
A dependency missing from the bundle entirely is outside the question, so it was
never asked in five years of libs bundles. `check-bundled-deps.py` now runs on
all five Linux libs legs and both python legs.

⚠ **Licence text was not travelling with the binaries.** Every Windows, Android
and libcrispasr artifact shipped LICENSE + THIRD_PARTY_NOTICES; no CLI tarball
did, on any platform — while bundling `libgomp` (GPLv3 + GCC Runtime Library
Exception) since #296. The RLE covers LINKING it into an MIT binary; shipping
the `.so` is separately a GPLv3 conveyance with a §6 source obligation, so the
notices now carry a source pointer and a written offer. `libomp` is Apache-2.0
WITH LLVM-exception — notice only. The notices file also claimed OpenBLAS was
"not bundled", false since #296.

## LANDED 2026-08-10 — #339 fallout: a red Release run silently killed every GPU wheel

The reported bug was six of seven Linux tarballs failing in v0.8.26 (two shell
bugs in `release.yml`, fixed by the parallel session in 43231d0d). Verifying the
asset SET rather than the count turned up a SECOND, unreported casualty:

`release-python-wheels.yml` triggers on `workflow_run: [Release]` and gated on
`github.event.workflow_run.conclusion == 'success'`. The six failing tarball
legs reddened the run, so the whole wheel pipeline **skipped** — and the GPU
wheels it publishes to the gh-pages PEP503 index
(`crispstrobe.github.io/CrispASR/whl/{cuda,vulkan}/`) have been stale at 0.8.25
since 2026-08-01. Nothing reported it: a skipped workflow is not a failure.

The wheels repackage `libcrispasr-<platform>[-cuda|-vulkan]`, and every one of
those assets built fine in v0.8.26 — so the inputs existed the whole time. The
gate now accepts `failure` as well as `success`; each matrix leg already fails
loudly if the asset it downloads is missing, which is the real precondition.

⚠ **Two process lessons, both now in the dev guide.** (1) "A red Release run is
often fine" is true for ASSETS and false for anything triggered by
`workflow_run` — check what a downstream consumer gates on before calling a red
run cosmetic. (2) Verify the asset SET, not the count: v0.8.26 published 21
assets and was approved on that basis; the six missing tarballs were an
absence, which a count cannot show. There is a normalised `comm -23` recipe in
the guide, and it was proved red against v0.8.26 before being trusted.

**Backfill — and the trap in it.** `release-python-wheels.yml` takes a `tag`
input, so an old release's wheels can be regenerated by `workflow_dispatch`
rather than re-tagging. ⚠ But its build jobs used `actions/checkout@v6` with NO
`ref:`, so they always built from the DEFAULT BRANCH while uploading to the tag
they were given. Dispatching `tag=v0.8.26` while main was 0.8.27 attached three
**0.8.27-labelled wheels to the v0.8.26 release** — a wrong artifact under a
version number, which is worse than a missing one. Assets removed; the three
build jobs now check out `${{ needs.setup.outputs.tag }}`. `publish-gpu-index`
keeps `ref: main` on purpose — it COMMITS the index there.

⚠ This also exposed a limit of the normalised asset-set check: stripping
version numbers to compare kinds is what makes the diff readable, and it is
exactly what hides a version MISMATCH. The set check answers "is anything
missing", not "is everything correctly labelled". Check both.

## LANDED 2026-08-09 — #337 qwen3-tts "GPU runaway" is NOT a miscompute

Reported on HIP: the talker picks a different token from CPU at frame 0, then
runs to the KV ceiling — 3796 frames / 303.76 s for one sentence — with exit
code 0 and a valid WAV. Reporter ruled out quantization, model size,
flash-attn, HIP graph capture and the voice reference, each with a paired test.

**Three measurements, and the second one overturned the first conclusion.**

1. His CPU-vs-GPU token table was confounded: the talker hardcoded top_k=50 /
   temp=0.9, and `qwen3_tts_set_temperature` reached the code predictor but NOT
   the talker. The RNG stream is identical across backends, but the pick is a
   multinomial draw over a softmax of 50 logits, so any float difference moves
   it. Added `CRISPASR_QWEN3_TTS_GREEDY=1` (top_k=1) and wired the temperature.
2. Under greedy on M1, CPU and Metal diverge — I first read that as "the bug
   reproduces on Metal, so it is a GPU-path defect". **Wrong.** Dumping the raw
   talker logits (`CRISPASR_QWEN3_TTS_DUMP_LOGITS`, the glm_ocr `*_DUMP_LOGITS`
   pattern) shows cos **0.99992** at frame 0 — better than the 0.998–0.999 band
   the guide calls normal for GPU-vs-CPU — decaying to 0.990 by frame 3 and
   0.84 by frame 5 as the AR loop amplifies it. Neither `--no-flash-attn`
   (bit-identical) nor `CRISPASR_KV_QUANT_{K,V}=f32` (5th decimal) moves it.
   This is the voxtral-tts pattern in the guide verbatim: an AR pipeline
   reproduces the reference at frame 0 then diverges as rounding amplifies —
   NOT a bug. The two backends simply follow different plausible trajectories
   after the argmax flips at frame 5; one of them didn't terminate.
3. `core_repeat::tail_is_repetition` looked like the fix — 3 backends use it,
   qwen3-tts never adopted it, and the degenerate output repeats. **Rejected by
   measurement**: healthy CPU output repeats a codec frame **7×** mid-utterance
   (period-1 `[1657]`) against the degenerate run's 8×. Structurally identical.
   The helper was written for TEXT tokens; at 12 Hz a held sound legitimately
   repeats, so it cannot discriminate here. Shipping the library default would
   have truncated good audio — a worse bug than the one being fixed.

**The actual defect** is that `max_frames` was the KV ceiling, so any input was
allowed 4096 frames (340 s). Now bounded by the text, the same
max_token_text_ratio idea upstream TTS models carry and that #334 ported for
cosyvoice3: `max(240, codepoints × 12)`. Sized from measurement — five
utterances ran 1.35–2.61 frames per codepoint, so 12 is ~5× the worst observed;
verified all five are untouched (79–124 frames against caps of 528–804) and the
cap branch is reachable. It only ever tightens the ceiling; `max_codec_steps`
and `CRISPASR_QWEN3_TTS_MAX_FRAMES` still override.

Hitting either bound now prints an explicit ERROR saying the output is a
runaway and should be discarded, and that a GPU-only reproduction is expected
arithmetic rather than a miscompute.

### Still open

- Making a runaway a non-zero exit rather than a log line. Deliberately not
  done: it changes the contract for callers who may be relying on truncated
  output.
- The reporter's HIP run diverged at frame 0 under SAMPLING; whether it also
  diverges at frame 0 under greedy is unknown and worth asking — a frame-0
  greedy divergence would point at the prefill and would be a different story
  from what Metal shows.

## LANDED 2026-08-05 — #334 cosyvoice3 WAV cloning (85d60ba9, 88c02788)

Reported as "long delay, pitch shifting and accent issues", blamed on sample
rate. **The sample rate was a red herring** and that is worth remembering: the
C++ speech tokens are byte-exact vs `speech_tokenizer_v3.onnx` (201/201) at
every reference rate, and our polyphase resampler tracks torchaudio to inside
the tokenizer's own sensitivity. What is real:

- **s3tok token ids are NOT stable across resamplers.** ONNX on a sox-16 kHz
  file vs the same audio resampled from 24 kHz agrees on only ~62% of tokens —
  and torchaudio scores the same 62%. The FSQ codes are near-ties, so a −51 dB
  difference flips a third of them. Any future "our tokens don't match" report
  must first ask *which 16 kHz signal*.
- **The talker had no minimum length.** Upstream masks the stop token while
  `i < min_len = (target text tokens) × 2`. Without it one unlucky step-0
  sample ends the decode ("AR decode produced 0 tokens", no audio); short of
  that the model spends fewer than 2 frames per text token, which is the
  reporter's chipmunk. Ported + gated `CRISPASR_COSYVOICE3_NO_MIN_LEN=1`.
- **The actual trigger is a `--ref-text` that doesn't transcribe the clip.**
  17.7 s of audio labelled with one sentence → collapse. Now warned against
  the same 2..20 speech-tokens-per-text-token band the decode uses.
- **The clone front-end re-ran per sentence chunk.** s3tok + CAMPPlus + prompt
  mel on every `synthesize()`; cached per (path, size, mtime, transcript).
  3-sentence `--tts` 66.0 s → 45.7 s, byte-identical.
- **`resample_polyphase` truncated its filter on every downsample** — the
  input window was `±num_zeros` where the filter spans `half_len/L` input
  samples. The pre-existing DC test had a 5e-4 margin over a 5.4e-4 defect.
- **RL talker published + wired**: `cosyvoice3-llm-rl-{f16,q4_k}.gguf` on
  `cstr/cosyvoice3-0.5b-2512-GGUF`, `--backend cosyvoice3-tts-rl`.

### OPEN follow-ups

1. ~~Auto-transcribe the reference when `--ref-text` is missing.~~ **DONE.**
   `examples/cli/crispasr_tts_ref_text.h` hoists f5-tts's transcriber + the
   ref-text cache; cosyvoice3 now auto-transcribes instead of hard-erroring
   (cached as `<voice>.cv3reftext`). Measured on a 17.7 s reference: a
   one-sentence guess lost the requested line entirely, auto-transcribed it
   came out in full. **Still CLI/server only** — `crispasr_session_set_voice`
   keeps returning -2 for a WAV with no transcript, because the session C-ABI
   cannot construct a second `CrispasrBackend` for ASR. That is the same
   limit f5-tts has always had, so bindings callers must still pass a
   transcript; lifting it means giving the library its own ASR entry point.
2. ~~The clone front-end is a harness-blind zone.~~ **DONE, and it found a
   real bug.** `clone_{spk_emb,prompt_feat_24k,speech_tokens}` stages now
   exist on both sides and give the three `cosyvoice3_tts_extract_*` APIs
   their first caller. First run: prompt mel 0.999948, speech tokens
   1.000000 — and **spk_emb 0.737** against `campplus.onnx`.
   Cause: CAMPPlus ends `transit3.linear(Conv1d, bias=False) →
   out_nonlinear(BN+ReLU) → StatsPool`, and the ONNX exporter FOLDED that
   trailing BN into the conv — the graph has a bare
   `/xvector/out_nonlinear/relu/Relu` with no BN parameters, while
   `/xvector/transit3/linear/Conv` gained a fused weight and a fused BIAS
   (transit1/2 keep bias-free named weights, no BN follows them).
   `bn_relu_conv1d` never applied a conv bias, so the fold was silently
   dropped. Zeroing that bias in the ONNX reference reproduces the old C++
   output at cos 0.999998 — that is what pins it. Fixed → 0.999997.
   ⚠ Every WAV clone had been conditioned on the wrong timbre while the baked
   voice bank was fine (its embeddings come from the ONNX model in Python) —
   which is why this read as "cloning quality" rather than as a bug.
   The bias is applied only when the checkpoint carries one.
   **Cross-backend blast radius — VERIFIED, not assumed** (the first commit
   message claimed "their converters emit no transit bias", which was
   reasoning, not a check; chatterbox's GGUF is not even produced by an
   in-tree converter). Listing the tensor names in each published GGUF over a
   ranged HTTP read of the header:
   | GGUF | transit tensors | transit `linear.bias` | out_nonlinear BN |
   |---|---|---|---|
   | chatterbox / -turbo s3gen | 15 | none | `s3.se.xv.out_nl.bn.*` present |
   | dots.tts spk | 18 | none | `…out_nonlinear.batchnorm.*` present |
   | cosyvoice3 campplus | — | **transit3 present** | **absent (folded)** |
   So chatterbox and dots.tts are bit-identical, and cosyvoice3 is the only
   folded export. The C++ was written against the un-folded shape and had
   never met the other one.
3. **Re-validation after the CAMPPlus fix — DONE.** Every #334 measurement
   before it was taken through a cos-0.737 speaker embedding, so the
   clone-quality claims needed re-running. Same reference at 8/16/22.05/24/
   32/44.1/48 kHz, matching `--ref-text`: all seven ASR-round-trip to the
   identical transcript, durations 7.62–8.94 s, and speaker similarity to the
   reference (Resemblyzer) is 0.744–0.776, spread **0.032**, mean 0.762.
   Timbre is now rate-invariant as well as content — the earlier conclusions
   hold, and the direct 24 kHz comparison moved 0.7245 → 0.7537. (Absolute
   values are dragged down in every arm by the prepended spoken disclaimer,
   which is in a different voice; the spread and the delta are the signal.)
4. **The 10 s prompt-mel cap is ours, not upstream's.** `compute_prompt_feat_24k`
   is called with `max_samples = 10 * 24000`, so a longer reference gives the
   flow a 10 s prompt while the LM keeps the full token set (deliberate, see
   the #310 comment). A 17.7 s reference round-trips fine, so this is not a
   bug — but it is an untested divergence worth an A/B.

## #333 madlad400 — DONE 2026-08-05 (F16 + Q8_0 published, port validated)

Reporter: only `q4_k` was on `cstr/madlad400-3b-mt-GGUF`, though the README
listed F16 and Q8_0 — and the card's own quickstart told you to download
`…-q8_0.gguf`, which 404'd. Both files are now published, and madlad went from
having **no diff-harness coverage at all** to a full per-stage table.

**Per-stage cosine vs the PyTorch blueprint** (`crispasr-diff madlad <gguf>
<ref>`; 14 stages, encoder → cross-attention → decoder → step-0 logits):

| | worst cosine | step-0 argmax |
|---|---|---|
| F16 | **1.000000** (every stage) | MATCH |
| Q8_0 | 0.999894 | MATCH |
| Q4_K | 0.993328 | MATCH |

F16 at 1.000000 on all 14 says the T5 port is faithful to HF's semantics; the
quants then degrade exactly in the expected order, and all three still pick the
same first token. `enc_pos_bias` is bit-exact (max_abs 0.00000) at every
precision, which is the relative-position-bucket logic — encoder-bidirectional
vs decoder-causal — confirming itself.

Shipped: `tools/reference_backends/madlad.py` (lazy per-tensor walk of the
11.76 GB fp32 checkpoint), `t5_translate_diff()` in `src/t5_translate.cpp`, the
`madlad`/`t5` arm in `crispasr-diff`, and `tools/kaggle/madlad-quants/` which
produced and validated everything. Reference archive:
`cstr/madlad400-3b-mt-GGUF/madlad400-3b-mt-ref.gguf` (1.5 MB).

**Follow-ups this left open:**

- ⚠ **The reference archive is in the MODEL repo, not
  `cstr/crispasr-regression-fixtures`**, which is where the convention puts them
  (`tests/regression/README.md`). It works where it is and the kernel points at
  it, but it should be moved or mirrored, and pinned by `fixture_ref_path` so CI
  can use it.
- ~~Greedy decode does not always terminate cleanly~~ **RESOLVED, and it was
  NOT a port bug.** The hypothesis was that the runaway pointed at the runtime,
  since the blueprint presumably stopped. Measured instead of assumed, and the
  answer is the opposite: **the PyTorch reference runs away identically** —
  60 tokens, no EOS, byte-identical string
  (`'Hello world! – 100000000000…'`). Ruled out along the way, each with its own
  arm: quantization (reproduces at F16, where parity is 1.000000), the KV cache
  (`CRISPASR_T5_NO_KV_REUSE=1` full re-forward gives the identical string), the
  tokenizer (runtime independently emits the same 11 ids ending in EOS=2, and
  `enc_embed` is cos=1.000000), the EOS id (config, GGUF and runtime all say 2),
  and the graph (14/14 stages, argmax MATCH). The port reproduces the model
  faithfully **including its failure mode**, which is itself evidence of
  fidelity.
  Shipped anyway as a decode-policy improvement: `core_repeat::tail_is_repetition`
  in the greedy loop, which trims the repeated tail and stops
  ("Hello world! – 10" instead of "…– 100000000000…"). **Gated
  `CRISPASR_T5_REPEAT_BREAK=0`, because it deliberately DEVIATES from the
  blueprint** — anyone diffing against HF needs the old behaviour back. Both
  sentences the reference terminates cleanly on are byte-identical with it on
  and off.
- **`t5` quantizer rule: written, MEASURED, and defaulted OFF because it loses.**
  The port pipeline's step 3 asks for a per-arch rule, so I added one keeping
  `shared.embed.weight` and `lm_head.weight` at source precision — reasoning
  that Q4_K's worst stages (`enc_embed` 0.9974, `enc_out` 0.9937) were
  embedding-driven. **That reasoning was wrong.** Re-quantized from the F16 and
  diffed against the reference archive:

  | | size | worst cosine |
  |---|---|---|
  | q8_0 | 3.38 → 3.62 GB | 0.999922 → 0.999920 |
  | q4_k | 2.04 → 2.41 GB | 0.992929 → 0.992606 |

  Bigger and no better, so neither was published — the kernel's
  "upload only if parity improves" gate held. The new worst stages say why:
  `cross_v_blk0`, `enc_out`, `cross_k_blk0`, i.e. the error is what ACCUMULATES
  through 32 encoder blocks, not what the embedding lookup rounds off. A wide
  embedding cannot repair a stack that has already drifted.
  Kept and inverted rather than deleted (`CRISPASR_T5_KEEP_EMBED=1`): a
  different T5 checkpoint — tied embeddings, smaller vocabulary, or an imatrix
  run — could land differently, and the lever costs nothing switched off.
  **The published q4_k/q8_0 are the generic-path files and remain the
  measured-best.**

## OPEN 2026-08-05 — miotts writes a 24 kHz WAV header for 44.1 kHz audio

Reported from another session as "docs say 44.1 kHz, adapter says 24 kHz".
**Verified, and it is not a docs typo — it is a shipping defect:**

- `models/convert-miotts-to-gguf.py:269` writes `miotts.codec.sample_rate =
  44100` (default, or parsed from the upstream config).
- `src/miotts.cpp:262` reads it correctly: `get_u32("miotts.codec.sample_rate",
  24000)` — so the runtime synthesises at whatever the GGUF says, i.e. 44.1 kHz.
- `examples/cli/crispasr_backend_miotts.cpp:35` **hardcodes**
  `tts_sample_rate() { return 24000; }`, and that is what stamps the WAV header.

So 44.1 kHz samples get a 24 kHz header: the file plays ~1.84× too slow and
about an octave low. README.md:161, README.md:296 and docs/tts.md:13 all say
44.1 kHz, so the docs are right and the adapter is wrong.

**The fix is not "change 24000 to 44100"** — the rate is per-model GGUF metadata,
so hardcoding the other constant just moves the bug. Add a
`miotts_get_sample_rate(ctx)` accessor (there is none: `grep sample_rate
src/miotts.h` is empty), have the adapter return it, and keep 24000 only as the
pre-init fallback. Then check the session C-ABI arm too — per the multi-surface
rule it reimplements the backend inline and will have its own copy.

⚠ NOT fixed here because it needs the model to verify end-to-end (502 MB, not on
this box) and an audible before/after is the only acceptance test that matters
for a rate bug. Everything above is from the source, not from listening.

## OPEN 2026-08-05 — carried out of the #316 round-2 work

These are the loose ends from `619e74b6..4b875be0`. None block anything.

1. **fr/es have not been checked for either German defect.** Both were found by
   method, and the method transfers:
   (a) *citation stress* — `espeak_fr.tsv`/`espeak_es.tsv` were generated the
   same way (one word at a time), so they bake in the same isolation stress
   German had. `tools/gen-g2p-de-unstressed.py` is the generator; pointing it at
   another language is a few lines.
   (b) *out-of-vocabulary symbols* — scan the G2P output against the model's own
   `tokenizer.ggml.tokens`. Three lines of Python, and in German it found we
   were deleting `ʏ` out of every München. Worth running for **every**
   non-English Kokoro model we ship, not just fr/es.
2. **Regenerate `espeak_de.tsv` with `--tie`.** Our dictionary has no tie marks,
   so `core_phoneme`'s German `ts`→`ʦ` is a blanket rewrite that cannot tell an
   affricate from a compound seam. espeak emits `t^s` only for the affricate, so
   a tied dictionary makes the collapse exact — and would let
   `CRISPASR_KOKORO_DE_MISAKI_ALPHABET` be judged on its merits instead of on an
   approximation. Regenerate + re-upload to `cstr/g2p-dicts`.
3. **The German tied-alphabet collapse needs a listening test, or a kikiri-tts
   model to test against.** It matches the published recipe and made the ASR
   round-trip worse on the hui base we ship; the hypothesis is that that model
   predates that part of the recipe. `kikiri-tts/kikiri-german-{victoria,martin}`
   are explicitly "misaki 0.9.4 + espeak-ng" and would settle it.
4. **Five `tools/kaggle/*/kaggle_harness.py` bundles are gitignored** —
   `mimo-cuda-rvq-309`, `moss-tts-quants`, `streaming-diarize-300`,
   `whisper-ja-760M-convert`, `whisper-punc-308`. They exist only on the machine
   that made them, so a fresh clone (and the CI checkout) has **no** bundled
   fallback harness for those kernels — which is the exact failure
   `check-kaggle-harness-sync.py` was written to prevent, and it cannot see it
   because it only compares copies that are present. Either `git add -f` them or
   teach the checker to fail on a kernel dir with no bundle.
5. **English `that`/`read`/`used`/`object`/`console`/`use` need a POS tagger** —
   already recorded below with the measurement that says a full spaCy port buys
   0.34%. The cheap slice is a closed-class rule for `that` alone (~24 of the
   ~32 residual tokens, no model).

## OPEN follow-ups from #332 (Rust/Dart diarize ABI, landed 2026-08-05)

The landed part: `DiarizeMethod::FoxNose` + options exposed in the Rust crate
and Dart, and the real bug behind the report — the hand-maintained Rust and
Dart mirrors of the APPEND-ONLY `crispasr_diarize_opts_abi` were never updated
when #324 appended the FoxNose fields, so every `diarize_segments` call from
those bindings had the C side read 24 bytes past the caller's allocation.
Both mirrors now carry the 48-byte layout; `crispasr-sys` has a size/offset
layout test, the flutter smoke test pins the `DiarizeMethod` indexes, and the
c_api struct comment now lists every hand-written mirror to update on the next
append.

Still open from the issue's asks:

1. ~~**`crispasr_session_output_sample_rate()` (+ channels getters).**~~
   **DONE** (second #332 landing): `crispasr_session_output_sample_rate` /
   `input_channels` / `output_channels` in the C ABI (per-backend rate table
   mirroring the CLI adapters' `tts_sample_rate()`, 0 = no audio output),
   wired through Rust / Ruby / JS / Java / C# (the surfaces that expose
   `input_sample_rate`), pinned by `tests/test-session-abi-nulls.cpp` and
   documented in `docs/bindings.md` §"Session audio-format getters".
   **A new TTS backend must add its ctx to the getter's table** —
   `docs/contributing.md` §5b.3 records the duty. Go / Dart / Python don't
   expose `input_sample_rate` either; extend all four getters together there
   if anyone asks.
2. ~~**Same-benchmark DER for the pyannote+embedder path**~~ — **already
   existed**; the note here originally claimed the pyannote path had no DER
   on the shared benchmark, which was wrong. The cross-method table lives in
   the #326 NOW section below and in `docs/diarization-speakers.md` "#326":
   pyannote+embedder **7.81 %** vs foxnose **7.32 %** mean DER on the same 8
   VoxConverse dev files (whisper-tiny segments, 0.25 s collar), with the
   3.18 %-vs-7.32 % foxnose discrepancy explained there (own turns vs ASR
   segments as speech regions). Nothing left to run for #332; the estimator
   under-count remains #326's open accuracy item.

## NOW — #326 diarization: the count estimator is the last accuracy item

Landed: parallel 60 s chunked pyannote inference (e517273d), the `SPK_MASK`
powerset transposition worth 15 DER points (15aad6f8), and the over-clustering
fix (a719c89d). See `docs/diarization-speakers.md` "#326".

Measured end to end on the 8 VoxConverse dev files, whisper-tiny segments,
0.25 s collar, `tools/der_voxconverse.py`:

| path | mean DER |
|---|---|
| raw posteriors, no clustering (the chunking A/B harness only) | 33.37% |
| `--diarize-method pyannote --diarize-embedder auto` | **7.81%** |
| `--diarize-method foxnose` (#324) | 7.32% |

Reproduce either arm in one command:

    python tools/der_voxconverse.py --prepare <voxconverse>/data --audio-dir /tmp/vox
    python tools/der_voxconverse.py --audio-dir /tmp/vox --model ggml-tiny.bin \
        --args "--diarize --diarize-method pyannote --diarize-embedder auto"

### 1. DONE (a719c89d) — over-clustering, 15.74% -> 7.81%

Archived to HISTORY. Independently re-measured 2026-08-03 from a clean corpus
extraction: mean 7.81%, per-file counts 5/6/3/4/5/4/3/3 — identical to the
commit's numbers, so both the fix and the harness are confirmed.

### 2. OPEN — the speaker-count estimator now UNDER-counts

The cap no longer decides the count, but the BIC estimator that replaced it is
wrong on half the shard in the other direction:

    file    GT  hyp   DER%
    esrit    5    5    3.29
    fsaal    7    6    3.80   under
    jyirt    4    3    9.33   under
    mesob    4    4   16.90   <- worst file, count is RIGHT
    nnqfq    5    5    3.59
    rcxzg    4    4    9.71
    tiams    5    3    9.61   under
    willh    2    3    6.23   over

Two separate threads, and it matters not to conflate them:

  a. **Count.** 4 of 8 wrong, 3 of those under. HISTORY's estimator survey
     ("the estimator is not FRAGILE, it is BIASED") measured the same shape
     with margins of 6-14%, so this is not a tie-breaking problem and a
     silhouette tweak will not fix it. NME-SC was already tried and LOST
     (archived). Next candidate is calibrating the BIC penalty against
     embedding-window count, since the errors concentrate on short files.
  b. **mesob has the RIGHT count and the WORST DER (16.90%).** Nothing to do
     with counting — it is confusion within 4 correctly-estimated speakers.
     Diagnose separately; a count fix cannot touch it.

GATE for any count change: mean DER must beat 7.81% AND mesob must not regress.

## OPEN 2026-08-03 — stamp the kartoffel repos (needs ~15 GB free; this box has 11)

Unclaimed. Everything needed is built and proven; this is blocked on disk, not
on work.

**Do NOT stamp parler or csm.** Checked: the verdict table resolves `parler-tts`
and `csm` by BACKEND NAME (`crispasr_speaker_identity_models.h`), which is
already independent of the filename — renaming one changes nothing, so stamping
them moves 14 GB for no behavioural gain.

`kartoffel-orpheus-de-{natural,synthetic}` is the opposite case: one `orpheus`
backend serves several checkpoints, so the verdict keys on a filename substring
and a rename drops `natural` from real_person to unknown — silently losing an
Art. 50(4) disclosure. That is the case the stamp exists for.

**Why it is not done.** `stamp-speaker-identity.py` rewrites rather than patches,
so it needs source AND output live: 13.2 GB peak for the f16, 7 GB for the q8_0.
This machine has 11 GB on `/` and 5.9 GB on `/Volumes/backups`, and other agents
build here — filling the disk would take them down with it. Not worth a
rename-robustness gain.

**When there is room**, stream it one file at a time rather than
`snapshot_download`-ing 24 GB:

    for f in kartoffel-orpheus-de-natural-{q4_k,q8_0,f16}.gguf; do
      hf download cstr/kartoffel-orpheus-3b-german-natural-GGUF "$f" --local-dir src/
      python models/stamp-speaker-identity.py --input src/$f --output out/$f \
        --speaker-identity real_person \
        --evidence "card: fine-tuned primarily on natural human speech recordings; 19 speakers extracted from podcasts/lectures/OER"
      # verify tensors with a RAW BYTE comparison, then upload, then rm both
    done

The synthetic sibling takes `--speaker-identity synthetic` and the evidence
"card: trained on synthetic German speech with emotion and outburst control".

⚠ Verify with `.tobytes()`, not `np.array_equal` — the latter returns False
whenever NaNs are present and will call a byte-identical file corrupt.

**Scope narrowed after checking where the stamp actually buys anything.** The
verdict table resolves `parler-tts` and `csm` by BACKEND NAME
(`crispasr_speaker_identity_models.h`), which is already independent of the
filename — renaming one of those checkpoints changes nothing. Stamping them
would move 14 GB for no behavioural gain.

`kartoffel-orpheus-de-{natural,synthetic}` is the opposite case: one `orpheus`
backend serves several checkpoints, so the verdict keys on a filename substring
and a rename drops `natural` from real_person to unknown — silently losing an
Art. 50(4) disclosure. That is exactly what the stamp exists to fix, so those
two repos (24 GB) get it and the others do not. Everything needed is built and proven, this is bandwidth.

Stamp `crispasr.voice.speaker_identity` into the repos whose verdict is
established but which were left for size: `cstr/parler-tts-mini-v1.1-GGUF`
(real_person), `cstr/csm-1b-GGUF` (synthetic),
`cstr/kartoffel-orpheus-3b-german-{natural,synthetic}-GGUF` (real_person /
synthetic). Upload only, no runtime code:

    ./models/stamp-published-voices.sh <downloaded-dir>

It asks `crispasr --print-speaker-identity` per file and skips unknowns, so no
verdict is restated. Verify tensors are byte-identical afterwards with a RAW
BYTE comparison — `np.array_equal` returns False whenever NaNs are present and
will report a good file as corrupt.

## Why not the hash-chained log the sibling projects built

CrispTTS hash-chains its consent log (SHA-256 per line + a sibling `.anchor`
file, since a chain cannot detect truncation of its own tail; Art. 17 erasure
handled by re-chaining survivors plus a `[CHAIN-REBUILT]` marker). It is well
built. Porting it is still the wrong first move here, for four reasons:

1. **We would be chaining the wrong layer.** Our record identifies the voice by
   NAME (`voice=alice.wav`), never by content — there is no hash of the
   reference anywhere in the tree. That file can be swapped a minute later and
   the record still "verifies". A chain protects the SEQUENCE of records; if
   each record is an unbound assertion, a perfectly chained log of them proves
   nothing. CrispTTS's log is worth chaining *because* it already ties to a
   hash.
2. **The threat model does not support it.** This is a self-attestation BY the
   operator, who controls the binary, the file and the anchor. A chain does not
   defend against the party it is recording. It defends against a third party
   editing the file without the tooling — the narrow case. CrispTTS says
   "tamper-evidence, not tamper-proofing" and is right; the risk is readers
   hearing the stronger claim.
3. **Persisting more creates the liability.** CrispTTS had to build erasure,
   retention pruning and rebuild records BECAUSE they persist identifying data,
   then prove re-chaining cannot launder later edits. Every field stored is a
   field that must be deletable on request.
4. **Library vs application.** CrispTTS and Susurrus are applications. CrispASR
   is a library + CLI embedded by others (four bindings, an HTTP server,
   Wyoming). The right shape for an embedded component is to emit a clean
   structured record and let the host own durability; a bespoke chained log
   duplicates journald/CloudWatch/SIEM and is weaker than any of them.

Real tamper-resistance is a STORAGE decision — append-only permissions,
object-lock/WORM, shipping off-box — and the library must not pretend to
substitute for it.

### The work, in priority order

1. **`ref_sha256=` in every `[CONSENT]` line.** SHA-256 of the file the backend
   will actually open (`resolve_voice_path()` output), via the header-only
   `crispasr::sha` already vendored for C2PA. This is the change that turns the
   record from an assertion into evidence, and a hash carries far less
   data-protection weight than the recording or a name.
2. **Correlate the record to the output it authorised.** A per-process `run_id`
   in the `[CONSENT]` line and in the post-synthesis audit line, so a disputed
   clip can be walked back to the attestation. On the server add a request id —
   today a consent line and its output are unlinkable.
3. **`--consent-log <path>`, JSON Lines, default off.** Today the only route is
   redirecting stderr, which interleaves the record with model-load noise and
   progress output. A separable sink is what lets operators route it into
   infrastructure that IS append-only.
4. **Document the division of responsibility** in docs/eu-ai-act.md: the
   operator is the controller, this is their artefact, tamper-resistance is
   theirs to provide, and here is the field to key erasure on.

### Below the line, probably never in-library

Hash chaining. The one real case is the Docker server run as an appliance where
no host audit infrastructure exists and the operator wants to show a regulator
the log was not casually edited. Revisit only then, and only after (1).

⚠ Legal note: as read here, Art. 50(4) is a DISCLOSURE duty, which CrispASR
already discharges unconditionally through marking. The consent attestation goes
to GDPR lawful basis, where the accountability duty sits with the operator as
controller, not with the tool. Engineering judgement, not legal advice — put it
to counsel before it appears in a compliance claim.

## OPEN follow-ups from #300 / #308 (landed 2026-07-27, see HISTORY)

1. **C# and WASM bindings are source-only-verified.** No `dotnet` or emsdk on the
   Mac, so the `IntPtr`/`PtrToUtf8` marshalling in
   `bindings/csharp/CrispASR/NativeMethods.cs` and the two `emscripten.cpp` sites
   were not compiled. `bindings-csharp.yml` is the real check — watch that job.
2. **Three ASR backends still unaudited for `CAP_PUNCTUATION_NATIVE`**:
   `lfm2-audio`, `fastconformer-ctc`, `wav2vec2` — no local GGUFs. The CTC pair
   almost certainly needs the pass (unpunctuated by construction); `lfm2-audio`
   is an LLM decoder and is the likely one to need the flag. Method:
   `FIREREDPUNC_DEBUG=1 … | grep PUNCDBG` and read `in=` — do NOT use
   `--no-punctuation`, it strips after the fact and inverts the answer.
3. **Delete the duplicated fallback copies** (READY — this is "Ready to take" #1).

   Fourteen `.cpp`/`.h` files exist twice: once in `crisp_punc/src`,
   `crisp_lid/src`, `crisp_truecase/src` (what the shared libraries build, and
   what CrispEmbed consumes via `add_subdirectory`) and once in `src/` (what
   `src/CMakeLists.txt` builds when those directories are absent from a
   checkout). They are the same implementations twice.

   `tests/test-copies-in-sync.cpp` now makes the duplication *safe* — it byte-
   compares all 14 pairs and asserts the pair list is exhaustive — but safe is
   not the same as gone. Two copies had already drifted before that test existed
   (see HISTORY 2026-08-03), and #308's capitalisation fix was dead code for
   months for the same reason.

   **The task:** make the fallbacks unnecessary, then remove them.

   - Find out whether a partial checkout without `crisp_punc/`, `crisp_lid/`,
     `crisp_truecase/` is still a real distribution mode. `src/CMakeLists.txt`
     branches on their absence — check whether anything actually ships that way,
     or whether the submodules are always present now.
   - If nobody needs it: delete the `src/` copies, drop the CMake fallback
     branches, and delete `tests/test-copies-in-sync.cpp` with them. A guard for
     a hazard that no longer exists is dead weight.
   - If someone does need it: the copies must stay, and so must the test.
     Say who needs it, in this file, so the next person does not re-derive it.

   **Do not** simply delete the `src/` copies and see if the build goes green —
   the fallback branch only compiles when the shared directories are *missing*,
   so a normal build will pass either way. Test by moving the three directories
   aside and configuring from clean.


## #316 Kokoro G2P — rounds 1 and 2 COMPLETE (2026-07-28 / 2026-08-05)

_Both rounds archived to HISTORY.md (PLAN compaction 2026-08-05). Round 2 landed
`619e74b6..4b875be0`, all six CI workflows green, reporter answered on the issue._

Headline: English phoneme agreement with misaki **67.0% → 95.7%** (99.0% over
aligned lines); German round-trip word accuracy **85.9% → 90.6%**; the reporter's
own paragraph now byte-identical to misaki's output.

**Still open** — carried into "OPEN 2026-08-05 — carried out of the #316 round-2
work" above (fr/es unchecked, `--tie` dictionary regen, the German tied-alphabet
listening test) plus the one long-standing English item:

- **`that` / `read` / `used` / `by` / `am` / `object` / `console` / `use` need a
  part-of-speech tag.** Measured rather than assumed: `python
  tools/check_misaki_g2p_agreement.py --corpus <prose> --tagger-value` runs
  misaki against itself with the tag withheld. The tagger moves 5.42% of its
  tokens — but 90% of that is `in`, `a` and `I`, which `core/g2p_ctxwords.h`
  already gets right with no tagger. **The genuinely tag-dependent remainder is
  0.34%**, which is the entire return on porting spaCy's `en_core_web_sm`
  (12 MB neural tok2vec + tagger). Everything else in misaki's `en.py` is
  already ported. The cheap slice is a closed-class rule for `that` alone
  (~24 of the ~32 tokens, no model).
- **misaki's reduced vowels `ᵊ` / `ᵻ` are still not modelled** — we emit plain
  `ə`/`ɪ` where misaki reduces. Context-dependent, so it needs the rule rather
  than a blanket substitution.

## NOW — VAD + mel front-end parallelization campaign (#305 → fleet-wide)

Started from #305 (reporter: whisper-vad-asmr + firered-vad slow / single-core).
The recurring pattern: the model compute is fine (ggml threads / Accelerate BLAS),
but the **audio FRONT-END (STFT/mel/fbank) is a scalar single-threaded per-frame
loop** — and for long audio the mel can out-cost the encoder (measured in
whisper-vad). Parallelizing over the independent frame axis (per-thread FFT
scratch, reductions keep order) is **bit-identical** and a large win.

**DONE (per-backend, std::thread, gated + bit-identical):**
- whisper-vad-encdec — GPU move (3.3×) + parallel mel (~30% long-audio) + requant
  (b5eb7556 / 9f867909 / 8324719e). `CRISPASR_VAD_ENCDEC_*`.
- firered-vad — parallel DFSMN convs + fbank FFT, ~3.7× (8c1d6a10).
  `CRISPASR_FIRERED_VAD_SERIAL=1`.
- marblenet-vad — parallel mel front-end, ~1.6× (7b2f11bc).
  `CRISPASR_MARBLENET_VAD_SERIAL=1`.
- Audited silero (whisper.cpp native ggml, already threaded) + webrtc (subband
  GMM, no FFT) — no change needed.

**STATUS: the CPU front-end parallelization vein is MINED (2026-07-26).** Full sweep
done. Nothing clean+validatable-locally remains — do NOT keep sprinkling std::thread:
- `core/istft.h` (shared by outetts_wavtok/kokoro/cosyvoice3 +8 vocoders) is
  single-threaded BUT it is **overlap-add** — adjacent output frames write
  overlapping samples (data race, unlike the mel's disjoint writes), and n_fft is
  tiny for most consumers (kokoro=20, cosyvoice3=16) so the per-frame IRFFT is
  already cheap. Poor risk/reward — SKIP (would need per-thread out buffers + merge
  or a stride-coloring scheme for a marginal win).
- Own-mel backends NOT on core_mel (f5_tts, gemma4_e2b, titanet, chatterbox_s3gen,
  outetts_wavtok, ecapa_lid): their FFT runs ONCE on the reference clip or on TTS
  output where the DiT/decoder dominates — marginal fractions, not the
  per-long-audio bottleneck the ASR/VAD mel was. Not worth the churn.
- moonshine uses a shared mel helper (no local scalar loop of its own).
The remaining lever is TIER 2 (GPU ports) only — see below + the handover prompt.

**KEY FINDING that reframes the fleet rollout:** `core/mel.h::compute` (used by
~28 ASR/TTS backends: parakeet, canary, canary_ctc, nemotron, qwen3_asr, cohere,
glm_asr, granite_*, higgs_stt, ark_asr, voxtral/4b, moss_*, mini_omni2,
lfm2_audio, qwen3_tts, cosyvoice3, chatterbox, indextts, mimo, piano_transcription
…) ALREADY has a §176f parallel-STFT path — but it is **`#ifdef _OPENMP` only**,
and this macOS/AppleClang build has **`OpenMP_CXX_FLAGS=NOTFOUND`** (no libomp), so
it is **compiled out** on macOS (and any non-libomp build). That is why ZERO
backends set `allow_parallel_stft=true` — the flag is a no-op on the dev box. The
mel *projection* matmul in `compute()` is serial too (not even OpenMP-gated).

**TIER 1 — port core_mel STFT to std::thread (PORTABLE) — DONE (528d672f).** One
change to `src/core/mel.cpp`: portable std::thread STFT path (OpenMP kept when
`_OPENMP`), DEFAULT parallel with `CRISPASR_MEL_SERIAL=1` opt-out, threshold T≥256.
Bit-identical (disjoint power[] rows, same reduction order). The mel projection was
already Accelerate/BLAS-threaded, so the STFT was the sole single-threaded piece.
Validated parakeet-tdt-0.6b-ja (M1): STFT 1070-frame chunk 20.19→5.52 ms (~3.7×),
transcript BIT-IDENTICAL serial vs parallel. Lifts all ~28 core_mel backends.
Neutral for heavy-LLM-decoder ASR (mel is a tiny fraction); real win for
encoder-bound ASR (parakeet/canary/nemotron) and long audio. `allow_parallel_stft`
per-backend flag now redundant (kept for back-compat).

**TIER 2 — CPU-hardcoded backends that could go GPU (per-model, MAJOR effort — NOT
started; needs models + Kaggle validation).** `ggml_backend_cpu_init()` with no GPU
path, conv-heavy enough to benefit: **mel_band_roformer** (source sep — STRONGEST,
already promised as a GPU port under #296 below; but 1284 lines already
BLAS-optimized, model not local, and #296 was Kaggle-validated → a full
transformer+iSTFT ggml-graph port is a focused multi-hour project that must be
diff-harness + Kaggle validated, NOT doable+trustworthy locally), piano_transcription
(cblas=0 — not even BLAS yet; a cheaper first step is BLAS/threads before GPU),
pyannote_seg (already uses threads), openvoice2, TTS codecs (miocodec has a scalar
FFT — VAD-style parallelize; miotts/tada_encoder). Small classifiers (ecapa_lid,
lid_fasttext, bert_encoder, marblenet) stay CPU (launch-bound). Each needs a
ggml-graph port + Metal/Vulkan landmine handling + diff-harness validation. RECOMMEND
as a dedicated session per model (get the model, port, validate on Kaggle), starting
with mel_band_roformer (highest impact, already promised).

**TIER 3 — mined.** §176 runtime-opt campaign is 18/20 DONE; the 2 open are
<2% (measure-first). Do not chase.

## #296 mel-band-roformer — follow-ups PROMISED on the issue (do not drop)

`--separate` (mel-band-roformer) was ~24 min for 11 s on Linux/Windows (fast only
on macOS/Accelerate). Root cause: CPU-only forward with an Apple-only BLAS gate +
naive O(N²) iSTFT + scalar attention + redundant weight de-quant. FIXED on main to
**24 min → 56 s (~26×), cos=1.0** via: portable OpenBLAS `linear()`, FFT iSTFT,
attention→SGEMM + BLAS-thread-pin, per-layer weight hoist. Validated on Kaggle
(cos + per-stage profile each round). Replied on the issue
(`#296` comments 5062263388 / 5062697259).

**Explicitly PROMISED on the issue — must follow through:**
1. **OpenBLAS in the shipped binaries — AUDITED against the published v0.8.25
   artifacts, not the workflow source (`b0c548e7`).**
   - **Windows: DELIVERED.** `crispasr-windows-x86_64-cpu.zip` ships
     `openblas.dll` (1.78 MB) beside the exe. The reporter's platform has the
     fast path; this bullet's old "Windows CLI jobs set up NO BLAS" is stale.
   - **Linux: was BROKEN, and worse than a slow fallback.** apt's OpenBLAS is
     dynamic, so the binary carries `NEEDED libopenblas.so.0` while the tarball
     shipped only crispasr, crispasr-quantize and libc2pa_c.so — on a host
     without OpenBLAS it does not start at all (`error while loading shared
     libraries`). Read off the real tarball with objdump. Fixed by
     `scripts/bundle-openblas.sh` (the OpenBLAS half of `bundle-c2pa.sh`; RUNPATH
     is already `$ORIGIN`), wired into the x86_64 + arm64 Package steps.
   - **Both platforms could ship a degraded binary on a GREEN run** — the Windows
     vcpkg step is `continue-on-error` and CMake falls back to scalar silently.
     Both Package steps now assert the ARTIFACT and fail with `::error::`.
   ⚠ **UNVALIDATED until the next release run**: the bundling copy branch cannot
   execute on macOS. The no-op branch, the detection branch against the real
   shipped Linux binary, and the YAML parse were verified locally.
2. **GPU / ggml-graph port** (the real long-term fix, promised as "a GPU path is
   tracked"). Rewrite the forward (transformer + iSTFT) as a ggml graph → SIMD +
   threads + GPU everywhere, eliminating the scalar/BLAS/OpenMP scaffolding. Same
   Apple-only-BLAS pattern also slows **htdemucs** (the other `--separate`
   backend) on Linux/Windows — the ggml port is the template that fixes both.
   Needed for full-song latency (56 s/11 s still extrapolates to ~15 min/song).
   Validate with the mel-band-roformer diff harness (per-stage cos≥0.9995).

## RELEASE follow-up (open, 2026-07-25): v0.8.22 shipped with NO Windows CPU CLI

`release.yml` for v0.8.22 **failed on `build-windows-cpu`**: the mel-band-roformer
`openblas_set_num_threads` thread-pin used `__attribute__((weak))`, which **MSVC
rejects** — and it only bit that job (the one Windows job my vcpkg step gives
`HAVE_BLAS`). 26 assets shipped, but the #296 reporter's Windows CPU binary is
missing. **Fixed on main** (`3f41a0a4`, gated on portable `CRISPASR_MBR_OPENBLAS`,
no weak). **Still needs delivery** — v0.8.22 tag has the broken code, so a **v0.8.23**
must ship it (+ the queued #298/#299 fixes; also realigns main with the 0.8.23
PyPI wrapper). The MSVC fix is UNVALIDATED locally — watch `build-windows-cpu` on
the v0.8.23 release run. LESSON: [[untested-release-workflow-broke-release]].

## #266 follow-up — hoist speaker orchestration into the library (PARKED, LOW)

The #266 rework (closed-roster, cluster-level speaker identification — DONE,
see `docs/speaker-db-clusters/PLAN.md` + HISTORY) left one parked item (F9):
the diarize → merge → global-cluster → identify orchestration lives in the CLI
(`crispasr_apply_global_speaker_stages()`, examples/cli/crispasr_run.cpp).
Session-ABI/bindings compose the primitives themselves; the server exposes no
speaker-db (deliberate). If a binding or the server ever needs the full named
pipeline, hoist the orchestration into src/ (the parakeet_orchestrate
pattern / multi-surface lesson) rather than duplicating it per surface. The
compliance invariants (consent + `--expect-speakers` closed roster + post-only)
must move with it — see `docs/diarization-speakers.md` §2.

Pending roadmap items. Each is self-contained with files, approach, and
effort estimate. Completed items have been moved to `HISTORY.md`.

> **Numbering convention:** `§N` refers to PLAN items (sections in this
> file). `#N` refers to GitHub issues on CrispStrobe/CrispASR. They are
> independent sequences and numbers may collide. When in doubt, PLAN
> items are always written as `§N` and GitHub issues as `#N`.

**Latest release: v0.8.20** (tag `v0.8.20`, + pub.dev `crispasr 0.8.20`). Release
notes live on each tag; per-version `RELEASE_NOTES_v0.8.*.md` at repo root.
The v0.8.18 → v0.8.20 train shipped in one session (2026-07-21): each patch was
a genuine fix that only surfaced on a real tag run — see "Recent completions"
and `LEARNINGS.md` "a green release job is not a shipped artifact".

**Recent completions (2026-07-22):**
- **#292 --max-new-tokens ignored** — SHIPPED. moss-diarize (and 9 more ASR
  backends) hardcoded the decode cap, so `--max-new-tokens` did nothing and a
  long single-pass truncated (reporter's 300 s file stopped at 164 s). Fixed
  across all 10: context `max_new_tokens` field defaulting to the old constant
  (no regression) + setter + cap/KV read it; CLI forwards only when
  `max_new_tokens_explicit`; session C-ABI forwards `s->max_new_tokens`.
  moonshine excluded (its 194 is an architectural short-form limit). Also added
  `chunk_id` to segments (part 2: diarize speaker labels are chunk-local).
  **CUDA-validated** on the reporter's exact backend (moss-diarize q4_k):
  `--max-new-tokens` 64→4096 raised output 216→366 words CPU / 232→382 CUDA,
  chunk_id 4 distinct, tabcnn CPU/CUDA parity 0 mismatches (6/6, ~18 min,
  `tools/kaggle/cuda-292-maxnewtokens/`). See `LEARNINGS.md` "a hardcoded decode
  cap" + memory [[kaggle-full-harness-regime]].

**Recent completions (2026-07-21):**
- **#290 canary-qwen long audio** — SHIPPED (v0.8.19). The backend declared
  `CAP_INTERNAL_CHUNKING` with no chunker (`src/canary_qwen.cpp` has zero
  chunking code vs 62 hits in parakeet.cpp), disabling BOTH dispatcher safety
  nets → one full-length encoder pass → O(T²) attention (384 MiB→10.2 GiB) and
  sparse output. Fix: drop the false capability. See `LEARNINGS.md`
  "a capability flag is a promise".
- **Lib-delivery bugs (CometBeat)** — SHIPPED (v0.8.19 rpath, v0.8.20 flat+gpu).
  6 of 7 lib bundles were unloadable as delivered: macOS baked the CI runner's
  build path into LC_RPATH; all 5 Linux bundles used `$ORIGIN/../../ggml/src`
  (one level too high). The old gate only checked deps were PRESENT, never that
  the loader could FIND them. New `tools/verify-lib-bundle.sh` relocates +
  dlopens; `tools/package-lib-bundle.sh` flattens to `lib/` + rewrites rpaths.
  See `LEARNINGS.md` "presence is not resolvability".
- **iOS shipping for the first time** — SHIPPED (v0.8.20). Added a
  `build-xcframework` job to release.yml (build.yml is tag-excluded by design);
  fixed `build-xcframework.sh` to include `libglint.a` in the combined archive.
- **#291 C# binding neglect** — SHIPPED (v0.8.20). VadSegments returned
  centiseconds while documented as seconds; added `CrispASR.Logging`; bound the
  7 task backends (tab/beats/chords/piano/pitch/separate/convert) in
  `SessionMusic.cs`; added the first-ever C# CI. C# was also missing from the
  contributing.md binding-parity list. See memory [[csharp-binding-neglect]].

**Recent completions (2026-07-17):**
- **Roadmap accuracy sweep** — audited the PLAN's OPEN/NOT-STARTED headers against
  the code; **~11 items were already shipped** but still marked open and are now
  corrected: §169 (qwen3-asr ChatML prompt), #128 (Piper TTS), §66 (pub.dev
  `crispasr 0.8.11`), python `_find_lib`, #60o (MTLBinaryArchive pipeline cache),
  §155 (CONV_TRANSPOSE_1D — all phases + Metal/Vulkan/CUDA col2im kernels), #58
  (MOSS-Audio-4B), #101 (OmniVoice), §229 (GGML_LLAMAFILE ON), plus §57/§106/§224/§247
  sub-items. See `LEARNINGS.md` "verify PLAN OPEN items against code".
- **#227 VAD boundary reuse** — SHIPPED (CLI `--vad-export`/`--vad-import` +
  server `vad_export`/`vad_import`); shared serializer, unit-tested.
- **#91 CLI parity** — `--offset-t`/`--duration` now honoured by every backend on
  both CLI + server (shared `core/audio_window.h`, unit-tested); `--print-confidence`
  fixed for non-whisper backends (was a parsed-but-dead flag).
- **#201 TADA on-the-fly voice cloning** — in-memory make-ref (no temp GGUF) on
  both the C-ABI/session and server/adapter surfaces, opt-in
  (`CRISPASR_TADA_WAV_CLONE=1`). Roundtrip gate pending (see below).
- **Docs** — new `docs/benchmarking.md` (§227 fair-measurement recipe); `-am`
  aligner aliases enumerated in `docs/cli.md` (§105).

**Recent completions (2026-07-11):**
- **#242 moss-diarize**: SHIPPED — joint ASR + diarization + timestamps, 0.9B model,
  diff harness 4/4 cos=1.0, GGUFs on HF, full 12-point checklist. See `HISTORY.md`.
- **#200 dots-tts PatchEncoder**: FIXED — added missing RoPE (theta=10K) + QK-norm;
  full pipeline now runs e2e (LLM → DiT → PEnc → vocoder → WAV), ASR roundtrip passes.
  Wired `--tts-steps` / `--tts-cfg-scale`. GPU already supported. See `HISTORY.md`.
- **#215e gallocr UAF audit**: DONE — all 19 `cached_*_gf` sites audited across
  the codebase. 7 backends fixed (canary, canary_ctc, kyutai_stt, moonshine_streaming,
  nemotron, paraformer, sensevoice). See `HISTORY.md`.
- **Generation-health gate**: DONE — `src/core/generation_health.h` with 5 checks
  + 16 unit tests. Non-breaking additive.
- **qwen3-tts-perf (#245)**: ANALYZED — profiled on CPU: dispatch overhead (build+
  reset+alloc = ~5ms) is <0.1% of per-frame cost (5000ms compute). The O15 path
  already caches the graph and uses a dedicated sched. Skip-realloc is broken on
  CUDA+Metal. The bottleneck is pure matmul compute; perf wins require GPU where
  the ~5ms overhead becomes significant. Handover removed.
- **Untrusted-input parser hardening**: SHIPPED — multi-agent security audit of the
  audio demuxers + GGUF loader found 6 DoS/OOB defects (MP4 stsz/stco/co64 count +
  co64 offset overflow, WebM lacing, WAV + AU size clamps, GGUF split mmap bounds),
  all fixed + ASan-validated. See `HISTORY.md` + `LEARNINGS.md`.

## Untrusted-input parser hardening — follow-ups

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** Upstream the ggml empty-key fix to ggml-org (outbound public PR, left for a human)

**FIXED 2026-08-10 — the stb_vorbis SEGV `linux-fuzz-smoke` found.**
`vorbis_deinit` walked `comment_list_length` entries of a NULL `comment_list`.
`comment_list_length` is read straight from the file at
`examples/stb_vorbis.c:3660` and set BEFORE the array is allocated, so the
allocation-failure return one line later (an attacker-sized count makes
`setup_malloc` return NULL) left the two out of step. Reachable from the public
`crispasr_audio_load` on untrusted input.

⚠ An earlier CrispASR patch had already hardened the SIBLING path here — the
partially-filled array whose unassigned slots were freed as if valid — by
zeroing the allocation. It could not help when the allocation never happened.
Fixing the case you can see and leaving its neighbour is the recurring shape:
the guard now lives in `vorbis_deinit` (defends every path in, present and
future) rather than at the Nth caller, plus the length is reset on the error
path so the struct's invariant holds.

**Reproduced deterministically rather than waiting for the fuzzer.** The crash
input is 102 hand-crafted bytes — an Ogg page carrying a Vorbis ident header
and a comment header declaring `comment_list_length = 0x3FFFFFFF`. Kept as
`tests/fuzz/regressions/ogg-huge-comment-count.ogg`, and the smoke-fuzz job now
copies `tests/fuzz/regressions/` into its corpus, so every fixed crash becomes a
deterministic gate instead of a coin flip. Before/after on matching builds: 2
SEGV lines → 0.

⚠ `libcrispasr` is a SHARED library — the fuzz harness picks it up by rpath, so
"rebuild the harness" tests the new code with an old-looking binary. Rebuild the
dylib when doing a before/after, or the control silently becomes a second copy
of the experiment. (Cost me one invalid control here.)

The job also now uploads the crashing input on failure. It previously kept only
the stack trace, which for a stochastic job means the reproduction is gone.

**Noted, not fixed:** on macOS the same input flows past stb_vorbis into the
AudioToolbox fallback (`crispasr_at_decode` → `ExtAudioFileOpenURL`) and
libFuzzer reports an out-of-memory there. That is a resource limit inside
Apple's decoder on malformed input, not memory corruption, and that path does
not exist on the Linux CI. Worth a size sanity-check before handing a file to
AudioToolbox if anyone wants it.

## Diff-harness extensions (detail for "Ready to take" #5 and #6)

Both validate the TTS port against the Python reference. Neither is claimed.

### Diff-harness extension: per-step talker logits (#1, GPU-preferred)

**What:** Dump talker LLM logits at each generation step in both Python reference and C++ runtime, compare — validates the text-decoder input so the sampler is a faithful port over verified logits.

**How:** (a) add `talker_logits_step_N` capture to the Python reference dumper (`tools/reference_backends/qwen3_tts.py`) via a `generate`-time hook; (b) add matching C++ stage to `crispasr_diff_main.cpp`; (c) run on the TTS diff harness.

**C++ SIDE DONE 2026-08-10 (#337), and it settled the report.** Three levers,
which only work together:
`CRISPASR_QWEN3_TTS_GREEDY=1` (argmax), `CRISPASR_QWEN3_TTS_REPLAY_CODES=<file>`
(16 codec ids per frame — teacher forcing), and
`CRISPASR_QWEN3_TTS_DUMP_LOGITS=<dir>` (raw per-frame logits, before the
repetition penalty and suppress mask).

**Result: CPU vs Metal, fully pinned, worst cos 0.999870, mean 0.999940,
0/49 argmax disagreements.** Given identical history the backends agree at
every step, so the free-running divergence is entirely trajectory divergence
seeded by ~1e-4 arithmetic. That is the rigorous version of the #337 verdict;
the earlier free-running comparison suggested it but could not separate the two.

⚠ **Partial teacher forcing is a trap.** Replaying codebook 0 alone leaves the
15 residual codebooks sampled (`code_pred_generate_15` MUST sample — greedy
there is documented to produce silent output), so the per-frame input embedding
still diverges and the "teacher-forced" diff bottoms out at cos 0.849 — pure
artefact. Pin all 16 or measure nothing.

⚠ **Every per-synthesis dump must be tagged.** One `--tts` run generates twice
(the utterance, then the spoken AI disclaimer). `talker_%04d.f32` and
`generated_codes` were both frame-only names, so the disclaimer OVERWROTE the
utterance's dumps and a directory held two utterances with no way to tell.
Both now carry `_s%02d`. This produced one entirely bogus cross-backend table
before it was spotted — the tell was `argmax_gpu[k] == argmax_cpu[k-2]`.

**Python reference side DONE 2026-08-10.** `_hooks.capture_per_call()` written
(the module referenced an `_iter_capture` that never existed) — one capture per
call instead of first-call-only. `tools/reference_backends/qwen3_tts.py` now
emits `talker_logits_step0..15` and, at last, `generated_codes`: that stage had
been in DEFAULT_STAGES since the backend was written and was NEVER produced,
because the ids only exist inside `generate_voice_clone` (the outer call returns
audio). Captured by wrapping `tts.model.generate`.

Verified by running it: 17 tensors, `generated_codes (19, 16) int32` — exactly
the layout `CRISPASR_QWEN3_TTS_REPLAY_CODES` consumes — and
`talker_logits_step0 (1, 147, 3072)` = the PREFILL, with steps 1+ the AR steps
at `(1, 1, 3072)`. Note that indexing when diffing: step0 is not frame 0.

⚠ **Env:** the shared conda base has transformers 5.x; upstream Qwen3-TTS pins
4.57.3 and importing `qwen_tts` against 5.x dies on `check_model_inputs()`.
Do NOT downgrade the base. `tools/reference_envs/qwen3-tts/requirements.txt`
now scaffolds it; a `--system-site-packages` venv inherits torch and shadows
only transformers. `qwen_tts` comes from the clone at `~/code/Qwen3-TTS` via
PYTHONPATH, not pip.

**Prep landed 2026-08-10:** `qwen3_tts_sum_frame_embed()` factors the 16-codebook
embedding sum out of the AR loop so a harness entry point can build the SAME
per-step talker input without a second copy — duplicating it is how a harness
drifts from the runtime it checks, which is what #338 was. Verified
output-neutral: PCM byte-identical before/after (the WAV bytes differ only in
the C2PA/watermark metadata, which carries a per-run id — compare PCM, not the
container, when checking a refactor here).

**Still open, and here is the actual blocker.** A per-step talker input is
NOT just the codec-embedding sum:

    next_emb[step] = sum_{cb=0..15} embed_cb(frame[cb]) + trailing_text_hidden[step]

The `trailing` term is prompt-derived state computed from the synth text inside
the generate path. So a harness stage cannot simply prefill from the
reference's `talker_inputs_embeds` and then step — it also needs `trailing`,
which the reference does not currently dump and the runtime does not expose.
That dependency is why this stage does not exist yet; it is not just wiring.

Two ways forward, pick one before writing code:
1. Dump `trailing_text_hidden` as a reference stage too, and pass it into a
   `qwen3_tts_talker_logits_replay(embeds, n_tokens, codes16, n_frames,
   trailing, n_trail)` entry point. Most faithful, and it makes the
   dependency explicit in the archive.
2. Have the runtime construct the whole prompt itself from the same text +
   voice wav the reference used, and use the reference's
   `talker_inputs_embeds` ONLY as a structural gate (cos ≈ 1 before trusting
   any logits — the guide's input-alignment rule). Less plumbing, but it
   assumes the two prompt builders agree, which is the thing being tested.

Option 2's gate is worth having either way.

The superseded note: `CRISPASR_QWEN3_TTS_DUMP_LOGITS=<dir>`
writes raw per-frame talker logits (f32), dumped BEFORE the repetition penalty
and the suppress mask so a diff isolates the forward from the sampling policy.
It already paid for itself — it is what proved the reported "GPU miscompute"
was cos 0.99992 at frame 0, i.e. ordinary backend arithmetic amplified by the
AR loop. What remains is (a): the Python reference hook, which turns a
cross-BACKEND comparison into a cross-IMPLEMENTATION one against the blueprint.

**Test:** needs a TTS model (qwen3-tts or tada). 0.6B Q8_0 (941 MB) fits on VPS; TTS gen slow on CPU (~105x RTF) — use short "Hi." input, 2-3 frames.

### Diff-harness extension: replay-token dual-mode (#3, VPS-doable)

**What:** Dump Python's *sampled* token IDs and replay them in C++ (instead of re-sampling) so sampling-enabled downstream stages diff deterministically despite torch-vs-mt19937 RNG mismatch.

**How:** (a) Python dumper captures `sampled_token_ids` as a 1D int32 tensor in the reference GGUF; (b) C++ diff harness reads them and feeds the backend's step function instead of sampling; (c) compare downstream stages (codec, vocoder) vs the Python reference that used those tokens.

**Files:** `tools/reference_backends/<tts_backend>.py` + `crispasr_diff_main.cpp`.

## Delivery bugs found by CometBeat against the released v0.8.17 dylib (2026-07-20)

**STATUS: all SHIPPED (v0.8.19 + v0.8.20).** They validated the real macOS arm64
release artifact — pitch/piano/separate all work (CREPE gave a clean 2-octave
scale, piano recognised the Für Elise motif) — and hit two packaging/teardown
bugs. Independent confirmation that those three backends function end to end on
a shipped build, which we did not have before.

> The DB1 static-build fix below UNCOVERED a deeper delivery bug (2026-07-21):
> forcing ogg/opus static also made them non-PIC (broke the Linux shared link),
> and separately, 6 of 7 lib bundles had a broken run-path and could not be
> loaded as delivered. The old @rpath gate only checked that dependencies were
> PRESENT, never that the loader could FIND them. Fixed with a relocate+dlopen
> gate (`tools/verify-lib-bundle.sh`) and a flatten+rpath packager
> (`tools/package-lib-bundle.sh`), shipped v0.8.19/v0.8.20. See the 2026-07-21
> completions block above and `LEARNINGS.md` "presence is not resolvability".

### DB1 — release tarball missing libogg/libopus — FIXED + SHIPPED (v0.8.19)

`libcrispasr-macos-arm64.tar.gz` links `@rpath/libogg.0.dylib` and
`@rpath/libopus.0.dylib` (CRISPASR_OPUS_FETCH=ON builds them) but the Package
step only collected `libcrispasr*`, `libwhisper`, and `libggml*`. The tarball
therefore does not `dlopen` standalone. CometBeat worked around it by copying
Homebrew's copies into a flat rpath dir and re-signing.

**Root cause was deeper than packaging, and the first fix was wrong.** The
option is documented as building ogg/opus/opusfile *statically*, and
THIRD_PARTY_NOTICES.txt states they are "statically compiled into libcrispasr
... all official release binaries". Neither was true: `opusfile` is an explicit
`add_library(... STATIC)`, but ogg and opus arrive via
`FetchContent_MakeAvailable`, which INHERITS `BUILD_SHARED_LIBS` — and every
shared-lib release job passes `-DBUILD_SHARED_LIBS=ON`. So they silently built
as dylibs, creating @rpath deps nobody packaged.

Fixed in `src/CMakeLists.txt` by forcing `BUILD_SHARED_LIBS=OFF` around the
FetchContent block (save/restore, mirroring the existing BUILD_TESTING
pattern). Verified under the exact release flags: `libopus.a` + `libogg.a`
static archives, and **0 ogg/opus @rpath deps** in libcrispasr.dylib. The
bundle now loads standalone with nothing extra shipped, and the
THIRD_PARTY_NOTICES claim becomes true.

Also in `.github/workflows/release.yml`: a gate that derives the requirement
from the binaries themselves — every
`@rpath` dep of every packaged dylib must exist in the bundle, or the release
job fails. A hand-maintained copy list is what failed here; the next new
dependency now breaks the RELEASE instead of the consumer.
Gate tested both directions locally under `set -euo pipefail`, including the
`grep`-returns-1 case that would otherwise fail a dependency-free dylib.

**Needs a 0.8.18 release to reach consumers.**

### DB2 — SIGABRT at process exit when a session is still open — FIXED

`GGML_ASSERT([rsets->data count] == 0)` in `ggml_metal_rsets_free`
(`ggml/src/ggml-metal/ggml-metal-device.m:690`), reached from
`ggml_metal_device_free`. Fires AFTER correct output, during teardown.

Reproduced and root-caused:

| case | result |
|---|---|
| open session, **close it**, exit | exit 0, no assert |
| open session, exit **without closing** | **exit 134 (SIGABRT)** + backtrace |

`ggml_metal_device_get` holds the device in a **function-local static**
(`static std::vector<ggml_metal_device_ptr> devs`, ggml-metal-device.cpp:21),
so its destructor runs at static-destruction time and asserts if any Metal
buffer is still alive. The assert comment says as much: "you haven't
deallocated all Metal resources before exiting."

**Immediate workaround for consumers: close every session before process exit**
(`crispasr_session_close`). That is a one-line finalizer on the Dart side and
makes the abort go away entirely.

**FIXED in the ggml fork** (`CrispStrobe/ggml` @ bfe8ea22, submodule bumped):
`ggml_metal_rsets_free` now warns instead of asserting. Releasing the array is
correct refcounting either way — a residency set a live buffer still references
stays alive by its own retain — so the abort bought nothing but a dead process.
Precedent: 1dc4cb93 ("gguf: reject empty keys instead of asserting").

Chosen over the alternative — a session registry closed from an `atexit`
registered after ggml's device static (LIFO ordering) — because that adds new
lifetime management to the session C ABI with double-free exposure, and would
REGRESS a consumer that closes correctly from its own late static destructor:
we would free the session first and they would then close a dangling pointer.
The ggml patch has no such failure mode.

Verified both ways on macOS/Metal: closing the session exits 0 silently;
leaking it exits 0 with one actionable warning, where it previously exited 134
with a backtrace. A real leak is still diagnosable — the message names the fix.

## CometBeat handoff — singing-voice-conversion vocoders (OPEN, NOT STARTED)

Requested by the CometBeat `opus` (voice-svc) agent via its `docs/PLAN.md`
coordination note (read 2026-07-20). CometBeat is splitting a singing-voice-
conversion stack into pure-Dart (lightweight/offline) and native-via-CrispASR
halves. **We own the real-time-critical heavy vocoders**; they keep the
HuBERT/ContentVec encoder, Harvest F0, and a lightweight DDSP-SVC synth as the
web/offline fallback.

### §CB1 — RVC voice conversion — PORTED AND VALIDATED (packaging left)

Converter -> numpy spec -> ggml graphs -> convert() -> session C ABI, every
step at cos 1.00000000 against torch. `crispasr-diff rvc <model> <ref> <wav>`
reports 48 stages, including `convert_e2e` which runs the real
`rvc_svc_convert()` and reproduces the reference audio from ContentVec + F0 +
speaker id + replayed noise alone. Live test: 4 cases / 12825 assertions.

Deliberately NO CLI verb — the input is ContentVec features, which we do not
produce. `docs/bindings.md` documents the session surface.

ARTIFACTS: stored in the **PRIVATE** repo `cstr/rvc-svc-GGUF` (verified
`private: True` server-side before upload) — `rvc-40k-f32.gguf` (105 MB) plus
`rvc-ref.gguf` (23 MB), the 60-stage parity reference. That reference is the
valuable half: `crispasr-diff rvc <model> <ref> <wav>` re-runs all 48
comparisons with **no torch, no checkpoint and no RVC repo**.

REMAINING (packaging, not correctness):
- Registry entries. BLOCKED, and deliberately so on two counts: the checkpoint's
  licence is unscoped (converted `--license other`), AND the repo is private, so
  a registry URL would fail auto-download for anyone without auth. Scoping the
  base checkpoint's terms and re-stamping the tag comes first.
- F32 ONLY. An f16 GGUF converts but the graph is F32-only today (ggml_scale is
  F32-only; an F16 operand reaching ggml_add trips
  `GGML_ASSERT(src1->type == GGML_TYPE_F32)`). It REFUSES rather than producing
  subtly wrong audio, and is noted as such in the header.
- Dart/Flutter wrapper for `crispasr_session_convert*` (CometBeat is the
  consumer; the C ABI they need is done).
- Two agreed parameters were CORRECTED rather than implemented — `protect` is
  provably inert without the FAISS index, and `rms_mix_rate` needs the source
  waveform our seam never receives. See SVC_RECORD_SHAPES §9b.

### §CB1-history — original framing and the two findings that reshaped it

Contract CONFIRMED (see `docs/music-transcription/SVC_RECORD_SHAPES.md`).
Source traced; see `docs/music-transcription/RVC_BLUEPRINT.md`. Two findings
change the job before any code:

1. **The ask is ~3x bigger than "the vocoder".** The seam CometBeat wants
   (features + F0 + speaker -> audio) is `SynthesizerTrnMs768NSFsid.infer()`
   (`models.py:664`), which runs a **transformer encoder** (`enc_p`, with a
   `nn.Embedding(256, ...)` on the coarse pitch), a **normalizing flow**
   (4x coupling + flip, reversed), AND the NSF vocoder. Only the third is
   HiFi-GAN. **Ask CometBeat whether they want all three or only `dec`** — if
   only `dec`, they must send `z`, not ContentVec features, which changes the
   contract we just froze.
2. **Inference is STOCHASTIC, at two independent sites**: the latent sample
   `z_p = m_p + exp(logs_p) * randn * 0.66666` (`models.py:684`), and SineGen's
   random initial phase `rand_ini` (`:325`) plus additive noise (`:358`). So
   output is not reproducible run-to-run, waveform correlation is an invalid
   acceptance test, and the diff harness MUST replay the reference's noise
   (same pattern as `input_feat` in btc / `input_audio` in mel-band-roformer).
   Settle this before the graph, not after.

Numerical hazards already spotted (phase `cumsum` drift in f32, `F.interpolate`
mode, `upp = prod(upsample_rates)` derived not configured, `noise_convs`
stride schedule) are itemised in the blueprint.

### §CB1-old — RVC NSF-HiFi-GAN generator (original framing)

**What:** port the RVC NSF-HiFi-GAN generator to ggml/native-FFI. CometBeat
feeds us ContentVec features (from their `hubert.dart`), F0 (from RMVPE —
already done on their side) and a speaker id; we return converted audio.

**Seam:** they expect a `CrispasrSession.convert(...)`-style entry point. That
is a THIRD task-shaped surface after `--separate`/`--pitch`/`--chords`, so it
follows the same pattern: its own session entry points rather than riding on
transcribe(), plus the CLI dispatcher and the wasm/Go arms. Note the
`crispasr_detect_backend_from_gguf` trap — register the arch there too, not just
in the CLI (see the BTC entry in `docs/music-transcription/PLAN.md`).

**Blocking coordination — DRAFTED, awaiting their reply:** see
`docs/music-transcription/SVC_RECORD_SHAPES.md`, a concrete proposal for every
record shape (layout, dims, frame rate, F0 units, unvoiced encoding, speaker id,
return PCM) with reasons, so it can be accepted or amended rather than discussed
in the abstract. Items we have not yet verified against the RVC reference are
marked [UNVERIFIED] and must not be built against. The sharpest open question is
§3: who resamples F0 onto the feature timebase. The feature/F0 record shapes
must be agreed with the opus agent BEFORE their API freeze. Pin down, in writing: ContentVec feature
rate + dimensionality + dtype, F0 units (Hz vs cents vs MIDI) and hop, whether
F0 is voiced-masked, speaker-id encoding, and the sample rate of the returned
audio. Do this first — it is cheap now and expensive after the freeze.

**Licence:** RVC's own code is MIT, but the WEIGHTS in circulation are a mess
(many community models are of unclear provenance, and some RVC forks carry
non-commercial terms). Scope licences per-checkpoint before shipping any
registry entry, exactly as the music-transcription scoping pass did.

### §CB2 — Beatrice v2 (low-latency voice conversion)

**What:** port Beatrice v2; its low-latency design suits the native path.

**Licence: MIT — this entry previously said "custom/NON-COMMERCIAL" and that was
wrong.** `fierce-cats/beatrice-trainer` ships `LICENSE` = MIT (Copyright (c) 2024
Project Beatrice), and its README states in terms that *"このリポジトリ内の
ソースコードおよび**学習済みモデル**は MIT License のもとで公開されています"* —
the source **and the trained models**. So **no acceptance gate is needed**; a
`cc-by-nc-sa-4.0`-style tag here would have withheld permission the licence
actually grants.

Two real non-MIT signals exist but neither applies to this port: `beatrice.lib`
(the closed inference engine used by the VST "under permission") is irrelevant
because we port from the MIT source, not that binary; and the "Beatrice JVS
Corpus Edition" carve-out is a *different distribution*, not this repo.

**Feasibility: CONFIRMED — architecture and weights are both published.** An
earlier WebFetch summary of this same repo claimed it held "training scripts
only, not inference code or the model architecture". That summary was false;
`beatrice_trainer/__main__.py` is 4519 lines and defines the entire path
(`PhoneExtractor`, `PitchEstimator`, `VectorQuantizer`, `ConverterNetwork`,
`Vocoder`). Another instance of [[blueprint-summary-is-not-the-source]] — the
file listing settled in one call what the summary got backwards.

Weights (`assets/pretrained/`), all MIT, are **split per component** — the
obvious "load the checkpoint" assumption fails:

| file | contents |
|---|---|
| `122_checkpoint_03000000.pt` (14.7 MB) | `phone_extractor` **only** |
| `104_3_checkpoint_00300000.pt` (7.1 MB) | `pitch_estimator` **only** |
| `151_checkpoint_libritts_r_200_02750000.pt.gz` (153 MB) | `net_g` (177 tensors, the multi-speaker LibriTTS-R base) + `net_d` (discriminator, training-only → skip) |

**Scope is larger than §CB1, and the wire contract is different.**
`ConverterNetwork.forward` takes **raw waveform** (`x: [batch, 1, wav_length]`,
plus `target_speaker_id`, `formant_shift_semitone`, optional
`pitch_shift_semitone`) and returns 24 kHz audio. Beatrice therefore owns phone
extraction and pitch estimation itself — unlike RVC, it needs **no ContentVec
from the caller**, which simplifies CometBeat's side but means porting three
networks rather than one.

Non-obvious details to verify while reading (each a silent bug if assumed):

* `CausalConv1d` / `WSConv1d` — **weight-standardised** causal convs, not
  standard `nn.Conv1d`. The standardisation is part of the forward pass.
* `VectorQuantizer` is injected into `phone_extractor.head` via a **forward
  hook** (`enable_hook`). This session already lost time twice to hooks that
  silently never fire; the numpy spec must confirm the hook is active by
  asserting the quantised path *changes* the output, not by trusting it ran.
* `embed_quantized_pitch` is a **fixed sinusoidal** table (built in `__init__`,
  `requires_grad_(False)`) — it may or may not be present in the checkpoint, so
  the converter must rebuild it rather than assume it was saved.
* `key_value_speaker_embedding` is initialised with every speaker row **copied
  from row 0**, so speakers look identical until trained — an untrained-looking
  A/B is not necessarily a port bug.
* `self.melspectrograms` is **loss-only**; it is not on the inference path.
* Output rate is hardcoded 24000 with `hop_length = 24000/100` — the same 100 Hz
  frame rate §CB1 uses, so the record shapes carry over.

**Progress:** two of three components ported and validated.

| component | state |
|---|---|
| `PitchEstimator` | **DONE** — `crispasr-diff beatrice`, 30 stages + e2e, 0 failed |
| `PhoneExtractor` | **DONE** — `crispasr-diff beatrice-phone`, 69 stages + e2e, 0 failed |
| `ConverterNetwork` + `Vocoder` | blueprint read done; converter and graph NOT started |

**Next step:** the ConverterNetwork/Vocoder port. It is the largest of the three
and differs in kind from the two frozen extractors:

* The vocoder is a **source-filter / impulse-response synthesiser**, not a
  HiFi-GAN — a 512-tap IR per frame plus aperiodicity and post-filter, driven by
  `overlap_add`. That is why Beatrice is real-time-cheap.
* It uses **`WSConv1d`/`WSLinear`**, which neither ported component does, so the
  unbiased-variance trap (`torch.var_mean` is ddof=1, numpy defaults to 0 →
  uniform ~20 % weight mis-scale) becomes live for the first time.
* Its attention is **`CrossAttention`** over the speaker embedding, not
  `nn.MultiheadAttention`; the PhoneExtractor's `mha_subsequence()` does not
  transfer.
* `overlap_add` draws a **random initial phase**, so this component needs §CB1's
  injectable-noise discipline — and the injector must patch `torch.rand`, not
  just `randn_like`.
* The **lookahead alignment** (energy shifted 1 frame, quantised pitch and pitch
  features 2, all reflect-padded) is the highest-risk silent bug: getting it
  wrong misaligns content against pitch by 10–20 ms and no input-aligned
  per-stage check can see it.

Details in `docs/music-transcription/BEATRICE_BLUEPRINT.md`.

**Effort:** unestimated until the blueprint read is done, but larger than §CB1
(three networks, custom conv variants). Do NOT start before §CB1's record shapes
are agreed.

## Gemma-4 12B (gemma4_unified) ASR support (OPEN)

The remaining open item for full 12B support (a new converter map + backend audio path for the 640-dim unified
encoder) is a larger port, scoped but not started.

## crispasr-diff harness extensions — catch decode-policy / quality bugs (OPEN)

Status: config/adapter-parity guards DONE (`tests/test-tada-params.cpp` defaults-audit
now covers library + CLI + c_api; full ~40-adapter sweep clean, cosyvoice3/f5-tts session-config
bugs fixed). Generation-health header + unit tests DONE. Three extensions + one generalisation remain.

TODO (partial — status verified 2026-07-17):
1. **Per-step talker logits in the diff — DONE for the qwen3_tts exemplar.**
   `tools/reference_backends/qwen3_tts.py` captures `talker_logits` + `cp_step{0..14}_logits`;
   `_iter_capture.py` documents per-step talker_logits. Generalise to other TTS backends if/when
   a second consumer needs it.
2. **Wire generation-health checks into backends' live tests — still OPEN.** Shared header
   `src/core/generation_health.h` (check_not_empty / duration_plausibility / no_ngram_loop /
   not_truncated / tts_duration / trailing_silence) + `tests/test-generation-health.cpp` are
   done; still need per-backend live-test integration (needs models).
3. **Replay-token dual-mode reference — PARTIAL.** Noise-replay infra exists
   (`_iter_capture.py` writes `noise.bin` for C++ to replay); the *sampled-token* dual-mode
   (replay Python's argmax picks instead of re-sampling) is the remaining piece.
4. **Generalise the defaults-audit pattern across backends.** Per-backend table of
   (param → upstream default) checked against the params struct, so "knob declared but dead /
   default diverges from upstream" fails CI everywhere.

---

## #201 follow-up — generate a TADA voice ref from audio+transcript at query time (C-ABI + server DONE gated; roundtrip pending)

Switch-voice, offline `--make-ref`, `--align`, and CLI query-time inline cloning
(`--tts "…" --voice sample.wav --ref-text "…"`) all shipped.

**C-ABI / session half — DONE (opt-in, `feat/tada-201-server-abi`).** In-memory
make-ref, no temp GGUF:
- `tada_set_prompt_values()` — in-memory counterpart of `tada_load_prompt`
  (the latter now reuses it, so file + in-memory paths are identical).
- `tada_make_ref_from_pcm()` in `src/tada_tts.{h,cpp}` — `tada_encoder_encode`
  (validated) → `tada_set_prompt_values`. Provably equivalent to
  `write_ref_gguf()+load_prompt()` (same tensors), so no new graph math.
- `crispasr_session_set_voice(s, "ref.wav", "<transcript>")` decodes to 24 kHz,
  resolves encoder + language-matched aligner (explicit → next-to-model → cache),
  and bakes the prompt. `crispasr_session_tada_set_makeref_models()` sets the
  GGUF paths; Python `Session.set_voice(path, ref_text)` already routes here.
- **Gated default-OFF: `CRISPASR_TADA_WAV_CLONE=1`** — without it a `.wav` voice
  keeps the historical `-2` reject, so default behaviour is byte-identical.

**Remaining before flipping the gate on by default:**
- **Decoded-output roundtrip (HARD RULE #3)** — synth reference → set_voice(wav,
  text) → synth clone → ASR + `speaker-cosine(clone,ref) > cosine(baseline,ref)`
  via the Python `Session` API on `tada-1b` (+ `tada-encoder-f16.gguf` +
  `tada-aligner-en.gguf`). Not run on the dev box (memory-pressured).

**Server / adapter half — DONE (opt-in, same gate).** The HTTP server uses the
backend *adapter* (`crispasr_backend_tada.cpp`), a distinct surface from the
session C-ABI:
- `TadaBackend::clone_from_wav()` — gated (`CRISPASR_TADA_WAV_CLONE=1`) helper
  called from both `init()` (first voice) and `apply_request_voice()` (the
  server's per-request switch). Resolves the WAV against `--voice-dir` (bare
  name → `<dir>/<name>.wav` + companion `<dir>/<name>.txt` for ref-text, the
  qwen3-tts convention), resolves encoder + aligner (explicit → next-to-model →
  cache → auto-download), decodes to 24 kHz, and applies via
  `tada_make_ref_from_pcm` — no temp GGUF. Off/failed → the historical reject.
- `/v1/audio/speech` gained a `ref_text` body field (→ `tts_ref_text`). The
  existing `consent_attestation` gate already fires for a `.wav` voice.
- Docs: `docs/server.md` (ref_text field + updated tada voice note).

**Still OPEN (optional):** cache baked ref keyed by (audio hash, transcript) to
skip re-running the aligner on repeat requests.

**Files:** `examples/cli/crispasr_backend_tada.cpp`, `examples/cli/crispasr_server.cpp`,
`src/tada_tts.{h,cpp}`, `src/tada_encoder.*`. Aligner is language-specific
(`tada-aligner-<lang>.gguf`) — must match audio language. CLI helper
`tada_run_aligner_pipeline` is the reference implementation.

**Effort:** Medium-large. Pipeline is CLI-proven; work is server lifecycle +
per-request wiring + caching + the ~1.3 GB memory gate. Lower priority than the
shipped switch-voice half. Tracked on #201.

## §192 follow-up — native-Vulkan TADA garbled output FIXED (codec on CPU)

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** GPU-native codec on RADV / chunked-decode design (deferred, conditional on a RADV user needing it)

## Cross-platform (Linux/x86 CPU) validation — DONE (green); follow-ups open

x86 CPU-only Linux validation of moss-transcribe / higgs-stt / ark-asr passed at
`dcc7e47b` (all verbatim, beam == greedy, Go LDFLAGS drift clean). Audit tool
`tools/check-backend-wiring.py` (`ccc04a02`) ships; 49 canonical PASS required.

Follow-ups (LOW, not blocking):
- [ ] Fix handover to `cmake --build build` (all targets) before `ctest -L unit` —
  VPS run only built `crispasr`/`crispasr-diff`, ran 2 unit tests. Or have VPS build all.
- [ ] Install Go toolchain on VPS (`root@168.119.190.252`) to close the one SKIPPED
  Go link check, or leave to CI.
- [ ] Optional: promote to a standing post-push Linux smoke (Routine/cron).

Multilingual + beam spot-checks (LOW, either machine):
- [ ] moss-transcribe is zh/en but only English (jfk) validated — run one German +
  one Chinese clip (de fixtures under `audio_samples/`).
- [ ] Run higgs/ark `-bs 4` on a noisy/accented clip to see if beam improves WER
  (only proven no-regression == greedy on easy JFK).

Backend-wiring coverage gaps (LOW cleanup; re-list via `python tools/check-backend-wiring.py`):
- [ ] **missing reference dumper**: `fastconformer-ctc`, `wav2vec2`, `m2m100`,
  `kyutai-stt`, `gemma4-e2b`. Mostly intentional (m2m100 text-only MT, gemma4-e2b
  shares gemma path, encoder components diff via host backends) — confirm per-backend
  before adding, not a blanket gap.

---

## moss-transcribe follow-ups (OPEN, LOW)

`moss-transcribe` backend (`OpenMOSS-Team/MOSS-Transcribe-preview-2B`) shipped
`9f3c5ede` — q4_k verbatim on jfk.wav, validated vs PyTorch ref via `crispasr-diff`.
#218 (greedy n-gram loop collapse + 30 s-seam dup) is fixed. Remaining optional work:

- **Publish f16 + q8_0** to `cstr/MOSS-Transcribe-preview-2B-GGUF` (q4_k + card live;
  f16/q8_0 held back for WLAN bandwidth). Re-stage from
  `/Volumes/backups/ai/moss-transcribe-preview-2b-{f16,q8_0}.gguf`, `hf upload-large-folder`
  into the existing repo. Both already produce the verbatim transcript locally.
- **GPU validation beyond Metal.** Metal (default) + CPU both verbatim; CUDA/Vulkan
  untested. LM reuses `core_attn::kv_self_attn` (covered by §192/#200 Vulkan F16-GQA guard);
  check the encoder's windowed `flash_attn_ext` + conv front-end on CUDA.
- **Multilingual eval.** Authors report 4.87 % avg WER; only English (jfk) validated here.
  Model is zh/en — spot-check a Chinese clip.

Note: encoder/adapter run F16 (cos ~0.98 vs f32 ref, byte-exact at layer 0 — pure F16
weight precision, not a bug). f32 encoder path not worth it since decode is verbatim.
Loop-fix opt-out: `CRISPASR_MOSS_TRANSCRIBE_NO_LOOPFIX=1`.

---

## #218 qwen3-asr long-audio root cause — quantized audio tower + prompt contract

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** Broader eval to flip windowed-attn default / declare CAP_UNBOUNDED_INPUT (mechanism ready)

## #218 glm-asr long-form blueprint parity — instruction + multi-window

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** Document that single-pass blueprint skips quiet leading audio (chunked covers more) in README

## #218 qwen3-family rebake + CUDA validation — kernel results

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** Investigate mega-asr long-form at 4-bit vs bf16 blueprint (candidate follow-up)

## #218 arc — remaining open threads (OPEN — only user-side calls left)

All technical threads resolved: mega-asr long-form loop is LoRA-induced +
model-inherent (port faithful, nothing to fix); windowed-encoder default flip
REJECTED (default stays FULL attention + 30 s chunks, `CRISP_AUDIO_WINDOWED_ATTN=1`
remains the >10-min escape hatch, pair with fix_loops ON); loop-metric hardening
done for new kernels.

**TO DO (user-side):**
- Reply/close GitHub issue #218 — all reported symptoms fixed on main + HF.
- Tag a release so binary users get the runtime half.

**Gate for any future loop guard:** must include the phrase-cycle metric (carried
by `tools/kaggle/mega-asr-blueprint-ref/`) — unigram-only gates miss 2-gram
degeneration.

## Priority ordering

| Priority | Item | Effort | Status |
|---|---|---|---|
| **HIGH** | [#221 Issue #89 hardening + v0.8.8](#221-issue-89-hardening--v088-release) | Medium | 5 steps: CI regression guard (a), server-path mirror (b), Vulkan sanity (c), q4_k registry/UX (d), release (e). |
| **DONE / LOW** | [§176 Runtime optimization pass](#176-runtime-optimization-pass--2026-06-20-audit) | Phased | 18/20 done. **2 OPEN (low-value, measure-first):** §176c device-resident KV (Dia measured ~1.2% of decode → DEFERRED; compute-bound decoders make this <2%), §176l Kyutai RVQ (genuinely scalar, but no local model to validate). Do not treat as HIGH. |
| **MEDIUM** | [#52 Qwen3-TTS](#52-qwen3-tts) — perf pass | Medium | talker + code_predictor + codec + ECAPA + codec_encoder done; step-4 perf pass open (~137 ms/frame → real-time). **O15 broken on CUDA and default-OFF** (`61c42bfb`) — main perf lever disabled; root cause is `ggml_set_rows` KV scatter or fixed-Lk causal mask on CUDA (crash on first code_pred call). Baseline O15=OFF: 27.4 ms/frame, WAV OK. |
| **MOSTLY DONE** | [#57 Commercial-friendly TTS expansion](#57-commercial-friendly-tts-backend-expansion) | Phased | Phases 1–3 + Turbo + native voice cloning shipped; #83 S3Gen fix landed. VoxCPM2, kugelaudio, gwen-tts, kartoffelbox-turbo, CosyVoice3 all shipped + registry-wired (verified 2026-07-17). **Remaining:** only the Darwin-TTS-1.7B-Cross / AMAImedia Qwen3-Darwin family unported. → HISTORY §82, upstream-prs/09–11. |
| **MEDIUM** | [#51c MiMo-V2.5-ASR F16 step decode](#51c-f16-step-decode) | Small | F16 step-decode validation blocked behind ≥32 GB box. Base runtime + Q4_K shipped → HISTORY §56. |
| **MOSTLY DONE** | [#58 MOSS-Audio-4B-Instruct](#58-moss-audio-4b-instruct) | Large | Runtime + GGUFs shipped, diff cos≥0.999, Kaggle P100 CUDA PASS. **Remaining:** flash-attn encoder, sweep transcript-extraction fix. → see HISTORY. |
| **MOSTLY DONE** | [§221 TADA encoder `--make-ref`](#221-tada-encoder---make-ref) | Medium | C++ encoder runtime + GGUF converters + diff harness shipped; GGUFs at `cstr/tada-encoder-GGUF`. **WIP:** cos_mean=0.94 parity (F16 precision), C++ BPE tokenizer for end-to-end `--make-ref`. → HISTORY §221. |
| **LOW** | [#95 IndexTTS Chinese TN binary alternative](#95-indextts-15-chinese-tn--binary-alternative-to-the-python-wetext-hook) | survey only | Python `INDEXTTS_TEXT_NORMALIZER` hook shipped. Hand-roll (#95a) is next *when* a user reports a digit/date prompt that breaks; OpenFST vendoring (#95b) only after #95a grows past ~5 cases. |
| **IN PROGRESS** | [#97 More Parakeet variants](#97-more-parakeet-variants) | Small per-variant | TDT/TDT+CTC + rnnt 0.6b/1.1b DONE. **parakeet-unified-en-0.6b** surveyed: 24L 600M Unified-FastConformer-RNNT, NOT converter-only — 8× subsampling (vs 4×) + Dynamic Chunked Convolutions are new; ~80% overlap with #81. Offline mode may work through existing converter; realtime-EOU blocked on #81 cache-aware streaming. |
| **LOW** | [#106 TEN-VAD](#106-ten-vad--low-latency-cross-platform-vad) | Small | Feasible VAD backend (C-compatible, 16 kHz / 10-16 ms frames, prebuilt libs + ONNX). License is the gate: Apache 2.0 plus extra no-compete / own-app-only conditions from Agora. |
| **MOSTLY DONE** | [#114 Long-form transcribe chunking-default ladder](#114-long-form-transcribe--make-chunkingstreamed-the-default-for-all-asr-backends-issue-89-follow-up) | Medium | Chunking/streamed-default shipped for all ASR backends + per-chunk AED re-injection + LCS dedup + word-snap. **Remaining:** EN-FLEURS retokenization artifacts (out of scope). → see HISTORY. |
| **MOSTLY DONE** | [#125 multi-backend bug sweep from montvid](#125-issue-125--multi-backend-bug-sweep-from-montvid-12-findings) | Medium | 12 findings; P1–P6b all DONE. **Remaining:** P0 Blackwell retest; mimo-asr `-np` empty-transcript retest. → see HISTORY. |
| **LOW** | [#127 Coverage gaps from 2026-05-26 sweep](#127-coverage-gaps-from-the-2026-05-26-overlap-save-sweep-close-out) | Small | (a) omniasr-llm DONE; (c) cohere-asr-ja DONE (repo `CKHO/cohere-asr-ja-GGUF`) — still needs JA fixture sweep for PERFORMANCE.md table. **(b) OPEN:** mimo-asr local test doesn't run in CI (4.2 GB Q4_K doesn't fit runner disk). |

**Open follow-ups from §79:**
- **#73 cohere long-form rerun.** flash_attn_ext shipped on canary + cohere (`193a736`). On JFK (~11 s)
  canary q8_0/q4_0 −17% under flash (win) but cohere q8_0/q4_0 is +11% (regress); F16 ties both.
  Before promoting flash as cohere's recommended path, validate on a multi-minute clip — if the
  crossover is workload-dependent, docs must recommend cast-on-read for short audio, flash for long.
  Until then PERFORMANCE.md notes flash as available-but-regresses-on-JFK for cohere.
- **encoder-decoder #69a** (canary, cohere, kyutai-stt). Cross-attention layout has no
  `<prefix><N>.*` block-tagged tensors; needs bespoke per-backend predicates. Own design problem.
- **#06 FA per-head mask (A1000 perf step).** Removes 72 CPU splits/chunk (per-head additive mask in
  `fattn.cu:423` + four kernel variants). ~300-500 LOC across `fattn.cu` / `fattn-common.cuh` /
  `fattn-mma-f16.cuh`. Expected ~10-15% wallclock on top of postsiglu. Follow the WDDM-warm bench
  protocol (LEARNINGS) for the new baseline before starting.

---

## §214 follow-up — chatterbox T3 batched-CFG (B=2) — deliver the win (OPEN)

Batched-CFG B=2 T3 decode shipped behind `CRISPASR_CHATTERBOX_T3_CFG_B2=1`
(default OFF); greedy-token bit-identical to legacy on CPU (all quants), GPU+F16,
GPU+quant. Path works — what's left is proving/delivering the speedup.
Files: `src/chatterbox.cpp` (`build_graph_t3_kv_b2`, `run_t3_kv_b2`,
`ensure_t3_b2_f16_weights`, decode loop ~§214). See HISTORY + PERFORMANCE §214.

TODO:
1. **Quiet-machine A/B + default-flip decision (HIGH, ~1 h).** CPU floor ~34 %
   was measured on a contended M1 — unreliable. Re-measure on a quiet host
   (alternating order, min-of-N, token-parity gate `CRISPASR_CHATTERBOX_TEMP=0`).
   **Flip default ON for the CFG path only if the win holds** (keep env + legacy
   path forever for bisection).
2. **Generalize B=2 to the other CFG backends — see §215.**

Closed/won't-pursue (do not reopen): cached/bucketed B=2 step graph (build+alloc
<1 % on both backends, and risks the §186 Metal `buffer is nil` crash); GPU B=2
as a speed win (GPU ~4× slower/step than CPU — its value is enabling T3-on-GPU +
GPU+quant F16-dequant, not beating the CPU default which stays production).

## §215 — batched-CFG (B=2) for the remaining TTS backends (OPEN)

Full-tree audit (§214, PERFORMANCE.md): only s3gen CFM + chatterbox T3 batch cond+uncond into one B=2 forward; every other CFG backend runs two sequential B=1 passes. Where the per-step forward is GPU-run, dispatch-bound, AND the dominant cost, apply the chatterbox-T3 B=2 pattern: batch over `ne[2]=2` for heavy GEMMs, split per-batch attention/KV-cache, tag `GGML_PREC_F32`, gate behind an env (default OFF), parity-gate vs the sequential path. **On GPU + quantized weights, dequant the batched weights q*→F16 GPU-resident** (s3gen `dequant_cfm_f16` / T3 `ensure_t3_b2_f16_weights` trick — Metal's batched `ne[2]=2` quant mat-vec misses the PREC_F32 exact-dot kernel and degenerates).

**General caveat (voxtral §93 lesson):** batched-CFG is a MODEST (~1.2–1.3×), GPU-dispatch-bound win. Before each port confirm the stage is GPU-run AND dispatch-bound AND dominant — measure, don't assume the T3 −42% transfers.

**Already resolved (see HISTORY):** §215a dia-*encoder* B=2 done; §215b tada — MEASURED NON-GOAL, do NOT port (talker only ~24% of loop, asymmetric graph paths); §215b bucket-floor follow-up SHIPPED backend-conditional default (`tada_default_bucket_min()`: Metal/CPU→64, CUDA/ROCm/Vulkan/WebGPU→512; `CRISPASR_TADA_BUCKET_MIN` overrides).

Open candidates, prioritized (high step count × dispatch-bound first):

1. **§215a dia decoder (HIGH).** `src/dia_tts.cpp run_dia_synth` — encoder already B=2, but the decoder AR loop still runs cond/uncond as two passes / two KV caches (`run_dia_decode_step`). Mirror chatterbox T3: B=2 decode-step graph, split per-batch KV write/read, F16-dequant on GPU+quant. Largest payoff — long AR loop, CFG every step.
2. **§215c zonos (MED).** `src/zonos_tts.cpp` keeps two KV caches (`kv_k`/`kv_k_uncond`), decodes sequentially (~L1740–1842). Batch AR backbone B=2, split dual-KV attention; keep dual-KV CFG + optional random speaker embed independent per batch.
3. **§215d voxcpm2 (MED).** `src/voxcpm2_tts.cpp` LocDiT runs `locdit_call` twice per ODE step (cond `mu` + zero-`mu`, ~L2752–2778). Diffusion (fewer steps) so lower payoff, but each LocDiT forward is heavy. Keep cfg-zero-star blend per batch.
4. **§215e f5 (MED, risky).** `src/f5_tts.cpp` runs `dit_forward` twice per CFG step (v_cond+v_uncond, ~L1563–1566). Same B=2-DiT shape as voxcpm2. §176h: a standalone B=2 DiT graph is correct but a runtime B=2 corrupted batch-1 (F5-runtime-specific, [[project_ggml_batched_fused_graph_alloc_bug]]) — run the parity gate especially carefully; may not be worth it.
5. **§215f cosyvoice3 (LOW — likely WON'T).** `src/cosyvoice3_tts.cpp` explicitly declined batching (~L3027–3030): 22-block diffusion twice/step, per-call overhead small vs forward. Only revisit if a profile shows dispatch overhead matters; else document as deliberate non-goal.
6. **§215g kugelaudio (BLOCKED).** `src/kugelaudio.cpp` CFG is a TODO (cfg_scale read but unused, negative path unimplemented). Implement sequential CFG first (correctness), then consider B=2.

**Shared infra:** factor the F16-dequant-of-matmul-weights helper (duplicated in `ensure_t3_b2_f16_weights` + s3gen `dequant_cfm_f16`) into a `core_*` helper only when a third backend needs it.

## §210 follow-up — shape-stable bucketed decode for remaining LLM/AR backends (CUDA-graph capture) (OPEN, CONDITIONAL)

Status: template landed in granite-speech (PR #207) — fixed-`Lk` KV bucket written via
`ggml_set_rows`, in-graph `ggml_argmax`, fused F16 embed. Unlocks CUDA-graph capture
(~9–13× decode on Ampere+, engages automatically in ggml-cuda for shape+pointer-stable graphs)
and Metal gallocr allocate-once. Survey done (LEARNINGS §210). **Done, no work:** granite_speech,
mimo_asr, dots_tts. irodori-tts DiT persistent graph implemented + byte-identical parity (default
OFF, gated `CRISPASR_IRODORI_PERSIST_GRAPH`).

### ⚠️ Cost/benefit — don't mass-port
- CUDA-graph win is gated to Ampere+ (sm_80+): `ggml-cuda.cu:4329`. Project's usual test GPUs
  **T4 (sm_75) / P100 (sm_60) are gated OUT** — only RTX 5090 / A100-class benefit.
- On Metal there is **no throughput win** (granite decode host-encode 1.8% / GPU 98%, GPU-bound
  Q4_K GEMVs); shape-stable rewrite buys only memory-pressure robustness, not speed.
- Each port is a MANUAL graph rewrite + byte-identical diff-harness validation — NOT delegable to
  agents (runtime graph code). Budget one focused session per backend.

**Port a backend ONLY when actually deployed on Ampere+ CUDA, and measure first**
(`CRISPASR_METAL_PROFILE` for host/GPU split; confirm no "disabling CUDA graphs" GGML_LOG_DEBUG line
on an Ampere+ GPU). Smaller decoders may have a larger host-encode fraction than granite's 1.8% — measure, don't assume.

### Templates (copy — already shape-stable)
- `src/granite_speech.cpp`: `granite_build_argmax_decode` (~L2474), `granite_dec_use_gallocr`
  (~L2362; gallocr on Metal/CPU, sched on CUDA so capture fires).
- `src/mimo_asr.cpp`: cached `step_t1_gf` + `step_t1_fixed_kv_len` (L211), set_rows scatter at
  runtime `kv_indices` (L958, L1026), skip-plan reuse (L1507–1522) — good set_rows reference.

### Candidates — growing-shape (`Lk = n_past + T`), naive per-step rebuild + sched_reset/alloc
ASR-LLM (prioritized by likely server deployment):
1. **voxtral** — `src/voxtral.cpp`, `Lk` L1019; rebuild + reset/alloc L1159/L1192/L1244.
2. **qwen3_asr** — `src/qwen3_asr.cpp`, `Lk` L1210.
3. **voxtral4b** — `src/voxtral4b.cpp`, `Lk` L1411 (decode via `core_greedy_decode`).
4. **gemma4_e2b** — `src/gemma4_e2b.cpp`, `Lk` L995 (`core_greedy_decode::run_with_probs_cb` ~L1631).
5. **glm_asr** — `src/glm_asr.cpp`, `Lk` L1322.
6. **higgs_stt** — `src/higgs_stt.cpp`, `Lk` L1061 (Qwen3-1.7B decoder).
7. **ark_asr** — `src/ark_asr.cpp`, `ark_build_decoder_graph` (L643) / `ark_run_decoder` (L729), `Lk` L674.
8. **lfm2_audio** — `src/lfm2_audio.cpp`, `Lk` L933; already on gallocr (L875) but growing-shape, not yet capturable.

TTS AR decoders (LOWER priority — heavier per-step compute = more GPU-bound, payoff diluted):
`csm_tts`, `indextts`, `bark_tts`, `moss_audio`, `mini_omni2` naive growing-shape. `qwen3_tts`,
`chatterbox` (T3), `tada_tts`, `parler_tts`, `vibevoice` already have per-backend perf work — audit individually.

### Per-backend recipe (mirror granite)
1. Allocate KV at `kv_max_ctx` once; pick fixed `bucket_len` (≤ cache cap).
2. Rewrite step graph to fixed `[0, bucket_len)` KV view; write new token K/V via `ggml_set_rows` at
   runtime index `n_past`. Mask `(bucket_len, 1)`, set host-side each step. Topology byte-identical across steps.
3. Move argmax in-graph (`ggml_argmax`); keep `logits` as graph output for callers.
4. Make embed capturable: if `token_embd` k-quant, in-graph `GET_ROWS` host-syncs and disables
   capture — fuse a F16 embed or pass a pre-computed F32 embed input (granite `fused_embed`).
5. Cache the cgraph; gate gallocr-vs-sched like `granite_dec_use_gallocr` (sched on CUDA/HIP so
   capture engages; force sched if a CPU layer split exists). Env opt-out.
6. Bound `n_past < bucket_len` (granite OOB guard `c5035969`).

### Validation gate (mandatory)
- Byte-identical transcript vs legacy path on jfk + fleurs_60s — no merge without it.
- Measure before/after: `CRISPASR_METAL_PROFILE` + per-step compute-µs accumulator like
  `CRISPASR_GRANITE_DEC_PROFILE`. On real Ampere+ confirm capture engages, A/B decode RTFx.
  M1 wall time is noise — gate on the instrumented per-step quantity.
- irodori DiT: flip its default ON (or gate `cc>=800`) only once a real Ampere A/B shows a win.

**Effort:** ~1 focused session per backend. Do highest-deployment ASR backend first; stop if its
measured Ampere+ CUDA A/B doesn't justify the next.

---

## §219 — more permissive audio input formats for crispasr_audio_load

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** Speex / WavPack decoder support (LOW, only if a corpus needs them)

## §166 follow-up — WASM `asr*` session surface needs a build-verify

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** Runtime asrTranscribe-equivalent smoke test in node/browser (LOW nicety)

## §175 Surgical DRY — share pure helpers across the CLI/library boundary

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** language-instruction prompt template helper (revisit if a 3rd consumer appears)

## §177 VibeVoice #171 — remaining layer: RDNA4/RADV quantized-matmul path (OPEN)

All other legs fixed (server/CLI chunking §176 prefix guard, KV stride leak at
71f0639, quant recipe b36248c1+5c8add40 — three HF repos regenerated). Reporter
retested regenerated q8_0 (sha256-confirmed current HF file): still broken. Signature:
q8_0 breaks, f16 clean, same RX 9070 XT (RADV GFX1201, `int dot: 1`); Metal + MoltenVK
clean on the exact q8_0. Suspicion: RADV GFX1201 shader miscompile on the quantized-
matmul paths (int-dot MMQ/MMVQ, KHR_coopmat dequant) — shader logic proven fine
(forced int-dot + MMVQ clean on M1). Vendored ggml base 2026-05-05, no matching
upstream fix found.

TO DO:
- [ ] Reporter runs the knob matrix one-at-a-time on a broken sample (env-only, no
  rebuild): `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1`, `GGML_VK_DISABLE_MMVQ=1`,
  `GGML_VK_DISABLE_COOPMAT=1`, `GGML_VK_DISABLE_F16=1`, anchor `CRISPASR_N_GPU_LAYERS=0`.
  Also Mesa upgrade / AMDVLK cross-check.
- [ ] Once one knob is confirmed → device-targeted safe default (RADV GFX12xx) in
  vendored `ggml-vulkan` + upstream report to ggml-org/llama.cpp with minimal repro.
- Note for thread: reporter's old `vibevoice-1.5b-bf16.gguf` predates `--include-decoder`;
  current f16 on `cstr/vibevoice-1.5b-GGUF` has the decoder tensors.

---

## 40. More Moonshine model variants

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** moonshine-streaming-tiny/small/medium need new streaming runtime

## 97. More Parakeet variants

Runtime (`src/parakeet.cpp`) dispatches TDT/CTC/RNNT via GGUF flags; converter
(`models/convert-parakeet-to-gguf.py`) reads hparams from `model_config.yaml` +
cross-checks tensor shapes. Most FastConformer-encoder + TDT/CTC/RNNT checkpoints
are converter-only, no new C++. Shipped: tdt-0.6b-v2, tdt-1.1b, tdt_ctc-110m,
tdt_ctc-1.1b, rnnt-0.6b, rnnt-1.1b (all on HF, registry-wired). See HISTORY.

**Still open:**
- **`nvidia/parakeet-unified-en-0.6b`** — direct-zip extraction + GGUF conversion
  already WORK (v4/v5 Kaggle kernels; 1181 MB F16, same arch as standard parakeet:
  d_model=1024, n_layers=24, vocab=1025, pred=640, joint=640). CrispASR SIGABRTs
  because runtime assumes 4× subsampling but this model uses **8×** (3 strided
  convs, not 2). **Fix:** make `parakeet_build_pre_encode` in `src/parakeet.cpp`
  (or `core/fastconformer.h`) handle `subsampling_factor=8` — 3 Conv2d layers with
  strides [1,2] instead of 2. Tensor shapes already in GGUF; graph builder just
  needs the extra conv layer.
- **`nvidia/parakeet_realtime_eou_120m-v1`** — streaming + end-of-utterance head.
  Needs cache-aware FastConformer streaming (cf. #81 Nemotron) + an EOU head. Not
  converter-only.

**Won't do (unless a user asks):** `parakeet-ctc-0.6b-Vietnamese` — already
runtime-supported (CTC), known gap not active work.

See also #98 Hotwords (orthogonal; lights up biasing on every Parakeet variant
once CTC-WS trie lands).

## 98. Hotwords / contextual biasing (OPEN)

User-supplied vocabulary the ASR prefers when in doubt (names, jargon, product/place
names). Helps only the biased subset, but lift there is large. Phased so each phase covers
a family of backends. ~1 week total covering 9 of 14 backends. (Full upstream-support
survey → HISTORY.)

**Phase A — generic CTC-WS phrase-boost trie** (2–3 days). Covers parakeet-ctc / -tdt /
fastconformer-ctc / omniasr in one shot (model-agnostic on the logit stream).
- New shared helper `src/core/asr_context_bias.{h,cpp}` — Aho-Corasick trie over piece-id
  sequences, configurable per-phrase boost; emits a per-frame log-prob bias vector the
  CTC/TDT decoder shallow-fuses into argmax/beam scoring. Pure CPU, no ggml graph.
- Wire-in: `parakeet_ctc_decode` + `parakeet_tdt_decode` (`src/parakeet.cpp:999+`,
  `parakeet.cpp:1670` dispatch). Phrase tokenisation via the backend's existing
  SentencePiece model so users pass human-readable strings.
- CLI: `--hotwords "Acme Corp,Sandra Berenz,GPU-PB"` and/or `--hotwords-file <path>`
  (one phrase/line, optional `^N` boost suffix); env `CRISPASR_HOTWORDS=...` for the
  OpenAI-server path. ~250–400 LOC incl. beam-rescoring.
- Reference to mirror: NeMo CTC-WS notebook + TurboBias `BoostingTree` C++
  (`arxiv.org/html/2508.07014v1`).

**Phase B — `--hotwords` → LLM prompt-prefix helper** (1 day). Covers funasr, granite-plus,
voxtral, qwen3-asr (each already has a system-prompt path).
- Tiny `src/core/` helper renders a hotword list into each backend's prompt template
  (granite `Keywords: …`; funasr hotword token block; qwen3 free-text context). One template
  registry, one call site per backend.
- Wire-in: `funasr_transcribe_ex`, `granite_nle_transcribe`, voxtral, qwen3-asr. ~150 LOC +
  per-backend template strings.

**Phase C — parakeet TDT joint-net boost (Transducer-native)** (1–2 days). Per-step bias on
joint-net output when the partial hyp matches a trie prefix (mirrors NeMo MBS hotwords).
- DECISION-GATE: defer until Phase A is shipped + benchmarked; only do it if Phase A on TDT
  undershoots NeMo's reference numbers.

Out of scope: whisper `initial_prompt` (already upstream via `--initial-prompt`), MiMo
PromptASR (no upstream flag — park), cohere/moonshine/kyutai-stt/glm-asr (no upstream hook).

Validation: `tests/test_hotwords.py` — synthetic clip with a rare name (e.g. "Berenz")
through each Phase-A backend with/without `--hotwords Berenz`; assert unbiased misspells,
biased nails it. Phase B: assert prompt-prefix matches upstream Python byte-for-byte.

---

## funasr — perf follow-ups (LOW priority, not blocking)

Port ships with `ggml_flash_attn_ext` on encoder+adaptor (`FUNASR_NO_FA=1` to opt
out), fused QKV (DONE), and single-token embed fast path (`CRISPASR_FUNASR_EMBED_FAST`,
default ON, DONE §180). None affect correctness — pure throughput. Remaining:

- [ ] **Per-step LLM decode graph cache.** JFK decode runs ~37.6 ms/tok vs ~6 ms/tok
  memory-bound floor (F16 Qwen3-0.6B on M1) → ~30 ms is graph-build/sched overhead.
  Build the step graph once at `funasr_kv_init` with `kv_indices` runtime input
  (K/V via `ggml_set_rows` to runtime slot, not static-offset `ggml_cpy`) and
  `fixed_kv_len = kv_max_ctx`; each step only writes positions/kv_indices/mask/inputs.
  Expected 5–10 ms/tok (15–25% decode). qwen3_asr could adopt same. Effort: ~1 bench session.
- [ ] **Encoder graph cache by T_lfr bucket.** Bucket to {128,256,512,1024,2048}
  (voxcpm2 TSLM pattern), pad inputs + static mask dropping trailing rows; first call
  per bucket pays build, rest reuse. Expected 10–20 ms/call warm. Effort: ~1 bench session.
- [ ] **Two-pass: CTC fast pass → Fun-ASR-Nano LLM rescore.** Upstream checkpoint has
  0 CTC tensors (LLM-style by choice); the only public trained CTC head is
  `csukuangfj/funasr-nano-with-ctc` (Apache-2.0, encoder+adaptor+CTC, no LLM, frozen
  encoder = upstream). Two patterns: (a) csukuangfj CTC head + upstream encoder/adaptor/LLM
  — single shared encoder forward, no vocab remap (cleaner, single-author trust);
  (b) fallback SenseVoice-Small fast pass (gold trust, extra encoder + vocab translate).
  - **Phase A (measure, no code):** tensor-list csukuangfj `model.pt` to confirm his
    encoder == upstream byte-identical; run his CTC head + upstream encoder+adaptor on
    a zh+en ground-truth set, measure CER/WER vs pure-LLM path. Need within ~3–5% rel
    for rescore net win. Write to LEARNINGS.md; proceed to B if in bounds, else use
    fallback (b).
  - **Phase B (impl):** grow `models/convert-funasr-to-gguf.py` to optionally pick up
    `ctc_decoder.*` from a with-ctc checkpoint → `funasr-nano-with-ctc-q4_k.gguf`,
    auto-download from `cstr/funasr-nano-with-ctc-GGUF` (mirror csukuangfj + attribution).
    Opt-in `CRISPASR_FUNASR_TWOPASS=1` (requires companion GGUF, else single-pass) +
    `--asr-rescore` CLI flag. One encoder forward forks into CTC + LLM heads: greedy
    CTC → per-frame probs; skip LLM if avg per-frame conf >0.95, else CTC top-K as
    LLM decode-prefix candidates. Expected 2–4× on high-conf clips, neutral on hard audio.

---

## §187 Cross-runtime embed fast path sweep

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** granite_speech embed fast path (VPS bench blocked) + deepen bench stages / encoder graph cache

## §201 Kaggle CUDA backend failures — full sweep 2026-06-20 (OPEN)

Status: the 2026-06-20 full Kaggle GPU sweep (`tools/kaggle-benchmark-all-backends.py`, streamed to
`cstr/crispasr-kaggle-progress/full-backend-sweep/`) ran 59 backends; 10 apparent failures reduced to
a handful of genuine CUDA-path bugs. Most resolved (vibevoice/lfm2-audio §206/kugelaudio §209/
fastpitch+speecht5 §204/chatterbox §205 — all in HISTORY). Remaining open:

TODO (open):
- [ ] **f5-tts** — runs once given a reference voice but TIMEOUT at 120 s in re-test. Bump smoke
  timeout (≥240 s) and re-run to settle pass-vs-stuck; passes on M1 Metal locally.
- [ ] **orpheus** (TTS) — fixed §215 (Metal + CPU bucket both ASR-roundtrip verbatim on M1), stays
  opt-in `CRISPASR_ORPHEUS_BUCKET=1` (~30% slower on M1 unified memory, may win on CUDA).
  **CUDA cross-check still pending** (Kaggle `chr1str/crispasr-orpheus-talker-cuda` end-to-end
  `orpheus_synthesize`).
- [ ] **chatterbox** (TTS) — 0-byte (~14 s) with `--voice <wav> --i-have-rights`; the #83 S3Gen GPU
  fix was Metal-validated — re-check the CUDA S3Gen path.
- [ ] **cosyvoice3** (TTS) — dies in 0.1 s even with a reference voice; passes on M1 Metal.
  Flow-matching + HiFT — cheapest to bisect. §205's mixed-radix FFT fix covers CosyVoice3's
  `n_fft=400` mel (same heap overflow as chatterbox) — likely resolved, needs Kaggle CUDA re-test.

**Method:** per-backend JSONs in the dataset have timing context; reproduce on a CUDA worker
(Kaggle T4/P100 or A1000) with `CRISPASR_VERBOSE=1` + `CRISPASR_<BACKEND>_DEBUG=1`. Several pass on
M1 Metal → CUDA-path-specific; cross-check Metal first. Small models also fit the 8 GB CPU-only VPS
where the diff harness can drive the fix if the bug reproduces on CPU (how §204/§206 were fixed).

---

## 51c. MiMo-V2.5-ASR F16 step decode — open (validation only)

Base runtime + Q4_K + fused-QKV shipped (HISTORY §56/§64); 51a mmap loader (§62)
and 51b step-decode KV reuse (§60) DONE. Only F16 step decode remains — **no code
change needed** (runtime is dtype-agnostic; mmap loader already wired). Blocked
behind ≥32 GB RAM: F16 working set (~16 GB) thrashes on this 16 GB box.

### TO DO
- On a 32+ GB box, validate JFK transcript byte-equality + decode speedup:
  ```
  CRISPASR_GGUF_MMAP=1 ./build-ninja-compile/bin/crispasr --backend mimo-asr \
    -m /path/to/mimo-asr-f16.gguf --codec-model /path/to/mimo-tokenizer-q4_k.gguf \
    -f samples/jfk.wav
  ```
- **Gate:** if F16 prefill hits ≥1× realtime as predicted, ship F16 as the
  recommended quant on `cstr/mimo-asr-GGUF` and demote Q4_K to memory-tight
  fallback. Until then both shipped, Q4_K default.

Effort: **0 LOC** (validation only).

## 52. Qwen3-TTS

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** perf pass — clean quiet-machine FUSED_QKV bench (F16/Q4_K), fusing 15 cp steps into one graph

## 54-follow-up. granite-speech-4.1 plus speaker labels + word timestamps — open

Variants 4.1 / 4.1-plus / 4.1-nar shipped bit-exact on JFK → HISTORY
§61. Remaining: speaker labels + word-level timestamps for the `plus`
variant via chat_template (~50 LOC, template-only).

---

## 56. Kokoro multilingual phonemizer (espeak-ng)

In-process libespeak-ng phonemization (behind CMake `CRISPASR_WITH_ESPEAK_NG`
AUTO/ON/OFF, `CRISPASR_HAVE_ESPEAK_NG=1`) with popen fallback + LRU cache SHIPPED.
German voice cascade (Option 1/2a/2b), phonemizer diff harness, and cache-clear
ABI all DONE (see HISTORY). Remaining open: Mandarin tone numbers, Japanese kanji,
optional German stage-2 fine-tune.

### Open TODO

1. **German stage-2 fine-tune (optional, out of scope of this item).** Native
   German path already ships (dida-80b backbone + kikiri voicepacks, auto-routes
   when both `kokoro-82m-f16.gguf` and `kokoro-de-hui-base-f16.gguf` are in the
   same dir). For deployable single-speaker production quality, run Stage-2
   fine-tune on one HUI speaker (~half-day A40). Track separately if needed.

2. **Mandarin tone numbers.** espeak-ng emits digit-suffixed tones (`ni2χˈɑu2`)
   not in the 178-symbol kokoro IPA vocab → dropped at tokenization, losing tone.
   Investigate `--ipa=2` (no tone numbers) + separate tone embedding, or switch
   Mandarin G2P (e.g. `pypinyin`). Symptom is auto-captured by
   `tools/check_kokoro_phonemizer_parity.py`. Effort: ~an afternoon.

3. **Japanese kanji.** espeak-ng falls back to English for kanji (日本語 →
   "Chinese letter") inserting non-IPA `(en)…(ja)` markers. Pre-process kanji→kana
   with a Japanese frontend before espeak. MIT-clean approach: MeCab (BSD-3) +
   unidic-lite (MIT) morphological analysis → reading extraction → feed kana to
   espeak `ja`. NO kakasi/pykakasi (GPL-3, viral). Impl: libmecab C API via dlopen
   (like espeak) OR a shipped kanji→kana flat-file dict (simpler, less accurate on
   rare/compound; generate offline with fugashi/cutlet, both MIT). Effort: ~an
   afternoon.

Files: `src/kokoro.{h,cpp}`, `examples/cli/crispasr_backend_kokoro.cpp`.

## 57. Commercial-friendly TTS backend expansion (OPEN — Phase 1 done)

May 2026 sweep of high-traffic HF TTS models. Filter: **permissive license + reusable
architecture + reasonable effort**. Sequenced so each phase unlocks a family of finetunes
(e.g. Phase 3 Chatterbox stack also unlocks Phase 5's CFM solver).

License triage that drives ordering (candidates for later phases):

| ✅ Permissive (commercial OK) | ⚠️ Llama-3.2 community (OK w/ attribution) | ❌ Non-commercial — defer |
|---|---|---|
| Qwen3-TTS-{Base,CustomVoice} (Apache 2.0) | Orpheus-3B family + Kartoffel_Orpheus (llama3.2) | SebastianBodza/Kartoffelbox-v0.1 (CC-BY-NC-ND) |
| ResembleAI/chatterbox base (MIT) | HumeAI/tada-3b-ml (llama3.2) | marduk-ra/F5-TTS-German (CC-BY-NC) |
| SebastianBodza/Kartoffelbox_Turbo (CC-BY-4.0, gated) | | mlx-community/fish-audio-s2-pro (Fish-Audio Research) |
| oddadmix/lahgtna-chatterbox-v0/v1 (MIT) | | amphion/Vevo1.5 (CC-BY-NC-ND) |
| openbmb/VoxCPM2 (Apache 2.0) | | mlx-community/Voxtral-4B-TTS-2603 (CC-BY-NC; upstream Mistral Apache OK) |
| FINAL-Bench/Darwin-TTS-1.7B-Cross (Apache 2.0) | | |
| AMAImedia Qwen3-1.7B-TTS-Cross-Darwin AWQ (Apache 2.0) | | |
| g-group-ai-lab/gwen-tts-0.6B (MIT) | | |
| kugelaudio/kugelaudio-0-open (MIT) | | |

TO DO:
- ~~Resolve license gap before depending on CosyVoice 3~~ — MOOT: `src/cosyvoice3_tts.cpp`
  + registry entry already shipped. Likewise VoxCPM2, kugelaudio, gwen-tts, and
  kartoffelbox-turbo (German) are all ported + registry-wired (verified 2026-07-17).
- Remaining unported from the permissive column: **Darwin-TTS-1.7B-Cross** and the
  **AMAImedia Qwen3-1.7B-TTS-Cross-Darwin** family (no `src/*darwin*`, no registry entry).
- Phase 2+ ports otherwise not detailed here — scope from the permissive column when picking
  the next family.

Phase 1 — DONE (see HISTORY.md + git log).

---

## Ecosystem expansion (lower priority)

### Candidate new backends (from PazaBench, see HISTORY #30)

| Model | License | Approach | Priority |
|---|---|---|---|
| Wav2Vec2 Conformer | Apache-2.0 | Conformer attention variant | Medium |
| Qwen2-Audio 7B | Apache-2.0 | Whisper encoder + Qwen2 LLM | Medium |
| OmniASR larger (1B/3B/7B) | Apache-2.0 | Same converter, bigger models | Medium |
| NeMo Canary-Qwen-2.5b | Apache-2.0 | FastConformer + Qwen2.5 decoder | Medium |
| Paza / Phi-4 | MIT | 14B multimodal, defer to llama.cpp | Low |
| XiaomiMiMo/MiMo-V2.5-ASR | TBD (check) | LLM-style multimodal speech (Qwen3-ASR pattern) | Medium — user-requested #35 |
| google/gemma-4-E2B | Gemma terms | Conformer + Gemma 4 decoder | Medium — user-requested #35 |

From llama.cpp (MIT): Ultravox (Whisper enc + Llama 3.2), Gemma 4 Audio (Conformer,
chunked attn, streaming), LFM2-Audio (Conformer variant, position embeddings).

### Post-processing — remaining candidates

- [ ] **CT-Transformer (FunASR)** Apache-2.0 — Medium. SANM 3-layer (vocab 272727),
  zh+en, FunASR/RapidPunc production default.
  `modelscope/punc_ct-transformer_zh-cn-common-vadrealtime-vocab272727-pytorch`. SANM
  primitives already in CrispASR (`src/core/sanm.h`). New alias `ct-punc`; VAD-realtime
  variant emits per-segment punc for streaming.
- [ ] bert-restore-punctuation (MIT, en) — Low.
- [ ] xashru/punctuation (Apache-2.0, XLM-R+BiLSTM-CRF, 40+ langs) — Low.
  (FireRedPunc, fullstop, punctuate-all, PCS, all truecasers — DONE, see HISTORY.)

### Optimizations still open

| # | Optimization | Applies to | Expected gain | Status |
|---|---|---|---|---|
| O2 | Fused QKV pre-merge | LLM decoders | ~10-15% attn (GPU) | API ready in core/attention.h; CPU gain <1%, defer to GPU |
| O5 | Pipelined mel+encode | LLM backends, CPU | ~15-20% | TODO |
| O6 | Batched encoder (GPU) | All + GPU | 3-5x | TODO |
| O7 | Speculative decoding | LLM backends | 2-4x decode | TODO |
| O4 | Beam search for LLMs | Audio-LLM backends | Quality | DONE except mimo-asr, blocked on #115 |

Guidance: only move LARGE, REUSED matmuls onto ggml/GPU; persistent subgraphs per
decode step > one-off graphs; never dequant (native Q4_K matmul 9.3× faster than F32
OpenMP). Candidate from CrispEmbed: SentencePiece Viterbi DP optimal tokenizer.

### Audio format support (open)

- [ ] `.m4a`, `.mp4`, `.webm` crash with upstream ffmpeg integration — needs fix or
  robust fallback.
- [ ] `.aiff`, `.wma`, raw PCM not supported without pre-conversion. Consider bundling
  a lightweight M4A/AAC decoder or improving the ffmpeg path.

### Other

- [ ] **OmniASR-LLM beam search** — beam=2+ with N hypothesis KV caches.

---

## Publish language wrappers to package registries

**Status:** Dart wrapper is **published** on pub.dev (`crispasr 0.8.11`, manual publish — see §66). Rust + Python wrappers have publishable metadata + passing dry-runs but are **not yet on crates.io / PyPI** (both 404, verified 2026-07-17) — blocked on the one-time registry creds bootstrap in §66. All are thin FFI/ctypes shims over the C ABI in `src/crispasr_c_api.cpp` — they do NOT bundle the native lib (user must have `libcrispasr.{so,dylib,dll}` installed).

### TO DO — one-time registry setup (must happen before first `v*` tag)

1. **PyPI** — at https://pypi.org/manage/account/publishing/ add a pending publisher: owner `CrispStrobe`, repo `CrispASR`, workflow `release-wrappers.yml`, environment `pypi`. Then push any `v*` tag. (OIDC trusted-publishing, no token.)
2. **crates.io** — generate a token at https://crates.io/me, add as `CARGO_REGISTRY_TOKEN` repo secret. Publish order: `crispasr-sys` then `crispasr`.
3. **pub.dev** — at https://pub.dev/packages/crispasr/admin (after first manual publish/claim) enable automated publishing, tag pattern `v{{version}}`. Or first-publish locally via `dart pub publish` with owner creds.

### TO DO — Python library discovery
Update `_find_lib()` in `python/crispasr/_binding.py` to probe in order: (1) `$CRISPASR_LIB_PATH`; (2) `sys.prefix/lib/`; (3) Homebrew/Linux paths (`/opt/homebrew/lib`, `/usr/local/lib`, `/usr/lib`); (4) existing repo-relative fallbacks. If none found, raise `RuntimeError` linking to install docs.

### TO DO — release automation
`.github/workflows/release-wrappers.yml`, tag-triggered (`v*` only, not every commit), runs in parallel: `python -m build && twine upload` (PyPI OIDC); `cargo publish -p crispasr-sys && cargo publish -p crispasr` (crates.io); `dart pub publish --force` (pub.dev OIDC). Version bumps stay manual — bump `pyproject.toml`/`Cargo.toml`/`pubspec.yaml` together in the tag commit.

**Effort:** Low per wrapper.

### Deferred — bundled wheels for Python
After the pure-Python release is out and stable, add a `cibuildwheel` pipeline (manylinux2014 + macOS arm64/x64 + Windows) bundling `libcrispasr.*` via `auditwheel`/`delocate`/`delvewheel`. Same optional path for Rust (`crispasr-sys` vendoring native build like `tch-rs`/`onnxruntime-sys`). Defer.

## 59. Cross-binding C-ABI parity

**Status:** Session TTS API (incl. qwen3-tts variant routing) is wrapped across all 7 bindings (commit `65e0a61` + Dart follow-up). The non-Session ABI (~80 of the 136+ `crispasr_*` exports in `src/crispasr_c_api.cpp`) is still C-ABI-only or partially wrapped on most bindings. Rust + Python are canonical full-coverage; the diarize surface (segment-level + #107 P6 embedder/clustering/cache) also landed in Dart/Flutter + Go.

### When to do this
**Not now.** Open a capability×binding only when a concrete consumer asks (e.g. "Java VAD", "Go streaming"). Reference commits for the pattern: `4f476c3` (TTS surface sweep), `65e0a61` (variant detect).

### TO DO — capabilities reachable only from C-ABI / Rust / Python
Each = ~3-12 exports + an idiomatic result type per binding:
- **Forced alignment** — `crispasr_align_words`, `align_words_abi`, `align_result_*`.
- **Diarization (segment)** — `crispasr_diarize_segments[_abi]`. Missing in Java, Ruby, JS.
- **Diarization (embedder+clustering)** — `crispasr_speaker_embedder_*_abi`, `crispasr_speaker_cluster_abi`, `crispasr_pyannote_cache_*_abi` (#107 P6). Missing in Java, Ruby, JS.
- **Language ID** — `crispasr_detect_language[_pcm]`, `crispasr_lid_free_cache`.
- **VAD** — `crispasr_vad_segments`, `crispasr_compute_vad_slices`, `crispasr_stitch_vad_slices`, `crispasr_vad_remap_timestamp`, `crispasr_vad_free`.
- **Streaming** — `crispasr_stream_open/feed/get_text/flush/close`, `crispasr_stream_run_decode`.
- **Punctuation** — `crispasr_punc_init/process/free/free_text`.
- **Model registry** — `crispasr_registry_lookup[_abi]`, `registry_lookup_by_filename[_abi]`, `crispasr_detect_backend_from_gguf`.
- **Cache** — `crispasr_cache_dir_abi`, `crispasr_cache_ensure_file_abi`.

### TO DO — #107 diarize-pipeline follow-up (three bindings still have nothing wired)
- **Java** (`bindings/java/`) — JNI exposes only `crispasr_session_*` + `*speaker_name*`. Add JNI wrappers for `crispasr_diarize_segments_abi` + the 9 `crispasr_speaker_*_abi`/`crispasr_pyannote_cache_*_abi` exports + idiomatic helper class. ~250 LOC.
- **Ruby** (`bindings/ruby/`) — only `Session.transcribe`. Needs Ruby FFI diarize bindings. ~200 LOC.
- **JS/WASM** (`bindings/javascript/`) — no speaker surface; depends on the WASM build linking pyannote-seg/titanet/indextts_voc. Start with `crispasr_diarize_segments_abi` (no model deps beyond existing wasm whisper); defer embedder primitives.

**Effort:** ~150-300 LOC per binding. Suggested ordering once a consumer asks: (1) Streaming (Go/Java), (2) VAD+alignment (Dart/mobile), (3) Diarization+LID+punc, (4) Registry+cache.

### Follow-up — Rust binding directory location (low priority)
Optionally relocate `crispasr/` + `crispasr-sys/` (repo root) under `bindings/rust/` to match C-family bindings. **Do NOT rename the crates** (names are correct/idiomatic). Move both dirs together (relative `path = "../crispasr-sys"` dep, no workspace). Consumer-safe (crates.io resolves by name+version). Before moving, audit: (a) downstream repos using `git`+`path` dep on the subdir (CrispEmbed/CrisperWeaver), (b) internal CI/`scripts/`/`build_go` refs, (c) docs path refs. One deliberate commit. Not worth churn unless the root-dir ambiguity bothers.

## 65-residual. JS / emscripten word-accessor surface — open

Parent #65 (session-API word-confidence parity) shipped → HISTORY §65
(main batch + vibevoice / moonshine-streaming + gemma4-e2b token-prob
API + Go/Java/Ruby parity in `5534588` + `d963e3a`). Only residual:
JS/emscripten word accessors — leaving until a JS consumer asks (the
current JS binding is TTS-focused).

---

## 66. Wrapper publishing bootstrap — required before language registries can ship

**Status:** PARTIAL (verified live 2026-07-17). Auto-trigger silenced —
`tags: ['v*']` push trigger on `release-wrappers.yml` is COMMENTED OUT (failed on
every release since v0.5.0, confirmed v0.5.4 `gh run view 25248028443`). Workflow
stays on `workflow_dispatch` only.

Live registry state:
- **pub.dev `crispasr` — DONE**, latest **0.8.11** (manually published
  `dart pub publish --force`, see `~/code/pupdev.md` handover 2026-07-15). No
  bootstrap needed for the package to exist; only the pub.dev-admin "automated
  publishing" toggle remains if tag-triggered republish is wanted.
- **crates.io `crispasr` + `crispasr-sys` — DONE**, both **0.8.23** (manually
  published 2026-07-27 with `CRISPASR_LIB_DIR` set so the verification build
  takes build.rs path 1, no cmake). The account needed a one-time verified
  email first. crates.io consumers must link a pre-built lib
  (`CRISPASR_LIB_DIR`); the package does not vendor the C/C++ sources, so a
  from-source build needs the git dependency (build.rs hardened to say so).
- **PyPI `crispasr` — DONE**, latest **0.8.23** (also 0.8.22; both manually
  uploaded 2026-07-24, verified 2026-07-27 — the published 0.8.23 wheel is
  byte-identical to `python/crispasr/_binding.py`). Pure-Python `py3-none-any`
  wheel; the native `libcrispasr` is still installed separately by the user.

So all three registries (crates.io, PyPI, pub.dev) are now bootstrapped; only
the optional CI/OIDC auto-trigger remains (below).

### TO DO — bootstrap (one-time, needs repo admin creds)

1. **crates.io — ALREADY DONE** (both crates first-published 2026-07-27). The
   recipe used (needs a *verified email* on the account + a prebuilt lib so the
   verification build skips cmake):
   ```bash
   export CARGO_REGISTRY_TOKEN=...          # token from https://crates.io/me
   CRISPASR_LIB_DIR=/usr/local/lib cargo publish --manifest-path crispasr-sys/Cargo.toml --allow-dirty
   sleep 30
   CRISPASR_LIB_DIR=/usr/local/lib cargo publish --manifest-path crispasr/Cargo.toml --allow-dirty
   ```
   For tag-triggered CI, add `CARGO_REGISTRY_TOKEN` repo secret (Settings →
   Secrets → Actions) — CI build.rs will cmake from the checkout, so no
   `CRISPASR_LIB_DIR` needed there.

2. **PyPI — ALREADY DONE** (package first-published manually, `crispasr 0.8.22`
   + `0.8.23`, 2026-07-24). Only optional remaining step for tag-triggered
   republish: at https://pypi.org/manage/account/publishing/ create a pending
   publisher — Owner `CrispStrobe`, Repository `CrispASR`, Workflow
   `release-wrappers.yml`, Environment `pypi`. Manual `twine upload` of a bumped
   version also works without it.

3. **pub.dev (Dart) — ALREADY DONE** (first-published manually to `crispasr 0.8.11`).
   ~~`cd flutter/crispasr && dart pub get && dart pub publish`~~. Only remaining
   optional step: https://pub.dev/packages/crispasr/admin → enable Automated
   publishing, Repository `CrispStrobe/CrispASR`, Tag pattern `v{{version}}` — do
   this only if you want tag-triggered republish (manual `dart pub publish` works
   without it).

4. **Auto-trigger — DONE.** `release-wrappers.yml` now runs on `push: tags:
   ['v*']` and publishes Rust (crates.io) + Dart (pub.dev). Python is handled
   separately by **`release-python-wheels.yml`** (see below), so the PyPI job
   was removed from `release-wrappers.yml` to avoid a double upload.

### Python wheels — `release-python-wheels.yml`

Ships the model I recommended: bundled **CPU wheels → PyPI**, **GPU wheels → a
PEP 503 index on GitHub Pages** (`--extra-index-url .../whl/{cuda,vulkan}/`),
plus a pure-Python **sdist** fallback. It runs on `workflow_run` after
`release.yml` finishes and REUSES the `libcrispasr-<platform>[-cuda|-vulkan]`
bundles that release.yml attaches to the GitHub Release — no native rebuild.
`tools/stage_libs.py` copies the libs into the `crispasr` package,
`_binding.py:_find_lib()` probes the package dir first, wheels are retagged per
platform with `wheel tags`. CPU matrix: linux x86_64 + arm64, macOS arm64
(Metal), windows x86_64. GPU: CUDA (linux + windows) + Vulkan (windows),
carrying `+cuda`/`+vulkan` local versions. Verified end-to-end locally on
macOS-arm64 (install → `_find_lib` picks the bundled dylib → `CDLL` loads).

REGISTRY SECRETS (all set 2026-07-27 via `gh secret set`):
- `PYPI_API_TOKEN` (pre-existing), `CARGO_REGISTRY_TOKEN` (added — crates.io CI
  publish was otherwise skipped; `gh secret list` confirms both).

GPU INDEX HOSTING: Pages already serves `main`/root (legacy Jekyll, the README
landing). The GPU index is committed to `main` under `whl/` by the workflow
(Jekyll serves it at `.../whl/{cuda,vulkan}/`), leaving the README landing
untouched — no Pages reconfiguration. Index links point at Release-hosted
wheels, so main only carries tiny `index.html` files, regenerated from all
releases each run (`[skip ci]` commit, push-with-rebase-retry).

Assumption to verify on the first tagged run: linux wheels are labelled
`manylinux_2_28_*`; if a bundle needs newer glibc, bump the tag (`auditwheel
show`).

## 67. Deferred follow-ups carry-over (mid-May 2026 session)

- [ ] **60d F16 mimo-asr re-upload (HF).** HF F16 (`cstr/mimo-asr-GGUF`) is still legacy
  unfused layout; runtime fallback works but misses the 1.7× fused-QKV per-step decode.
  Needs a fresh BF16→F16 run (killed at 22 min on the 16 GB / 99%-full box, disk-thrash).
  Run on a 32+ GB box with non-99%-full external, then `tools/patch_mimo_asr_fuse_qkv.py`
  patches to fused layout (~5 min).
- [ ] **60e per-backend Q8_0 KV cosine validation.** `CRISPASR_KV_QUANT={f16,q8_0,q4_0}`
  wiring landed across 9 backends (defaults F16, bit-identical until opted in). Only
  mimo-asr diff-harness-validated at q8_0. Remaining 8 (qwen3_asr, voxtral, voxtral4b,
  granite_speech, gemma4_e2b, glm_asr, omniasr, orpheus, qwen3_tts) each need a
  `CRISPASR_KV_QUANT=q8_0 crispasr-diff <backend>` pass (≥0.98 gate) before any
  default-flip. ~5 min each, warm cache, zero code.
- [ ] **Vibevoice CUDA cache reuse re-test.** `backend_needs_fresh_pred_graph()` bypasses
  the pred-head graph cache on Metal+Vulkan+CUDA (CUDA on presumption). On a CUDA box run
  `CRISPASR_VIBEVOICE_REUSE_PRED_GRAPH=1`, confirm no `GGML_ASSERT(src_backend_id != -1)`.
  If clean → drop CUDA from bypass list, recover ~30% per-synthesis caching. If assert
  fires → keep gated off; proper fix = recompute view→backend mapping from
  `view_src->buffer` in `ggml_backend_sched_split_graph`.
- [ ] **SYCL/HIP/ROCm cache-bypass extension.** Same shape as CUDA; extend the
  `backend_needs_fresh_pred_graph()` prefix list when a report comes in or a maintainer
  audits the upstream sched reset path there.
- [ ] **Per-backend `MADV_RANDOM` post-prefill.** `core_gguf::mmap_advise_random()` exposed
  but unused; add one call between prefill and decode loop in `mimo_asr_transcribe` /
  `qwen3_asr_transcribe` / `voxtral_transcribe` etc. when a 32+ GB-box benchmark shows
  benefit (marginal on Q4_K; F16 is where it matters, unmeasurable on 16 GB).
- [ ] **Disk5 cleanup.** `/Volumes/backups` at 99%. Safe to delete local unfused
  `mimo-asr-q4_k.gguf` (superseded by `mimo-asr-q4_k.fused.gguf` + HF fused) once future
  A/B not needed.
- [ ] **CI legacy `build.yml`.** Legacy whisper.cpp matrix (triggers on nonexistent
  `branches: [master]` + `tags: v*`), failing every tag push since v0.4.x. Doesn't block
  (ci.yml/release.yml are the gates). Delete or repair after auditing whether any
  build-matrix combo isn't covered by new ci.yml.

---

## 70. Streaming TTS via chunked VAE decode (latency win, vibevoice / qwen3-tts)

**Effort:** Medium-large. **Status:** not started; design settled.

Goal: a `--stream` TTS mode that emits a 24 kHz PCM chunk every K AR steps
(stdout / HTTP) while the AR loop continues, dropping time-to-first-byte from full
wall-clock to "K AR steps + one chunked-VAE pass". NOT the Intel-Arc Vulkan
workgroup bug (already fixed via CPU fallback `31795a7` / `VIBEVOICE_VAE_BACKEND=cpu`).
geneing's `chunked_vibevoice.patch` (#52) nailed the chunk decomposition but
regressed on per-call `sched_reset`+`sched_alloc_graph` overhead — start there.

### Three pieces required (land in order)
1. **Persistent VAE compute-graph reused across chunks** (mostly mechanical; the
   overhead that killed geneing's prototype). Mirror qwen3-tts `O15` graph reuse
   (`src/qwen3_tts.cpp:1037`): build once at `Lk = max_chunk_latents`, pin topology,
   reuse cached gallocr plan; cost = one `set_rows`-style write op/call, not a
   rebuild. Benchmark to confirm the regression is gone before proceeding.
2. **Causal padding on the σ-VAE conv stack** — left-pad each chunk with previous
   chunk's tail, drop first L output samples, so chunk decode == full decode at
   boundaries (avoids phase artefacts). Ref: kokoro/voxtral4b causal-conv1d paths.
3. **Chunked transfer in HTTP TTS endpoint** — depends on #58 (`POST
   /v1/audio/speech`) landing first; wire cpp-httplib chunked-transfer for
   `Accept: audio/wav; chunked` or `stream=true`.

### Backends in scope
- **vibevoice** (σ-VAE) — primary, largest win (positioned as realtime backend).
- **qwen3-tts** codec decode — 12 Hz vocoder; already has `O15` graph reuse, extend
  to chunked output.
- **kokoro** iSTFTNet — straight-line generator, cleaner chunking but iSTFT inverse
  window has the same boundary-artefact problem.
- Skip orpheus (SNAC already emits 24 kHz PCM single-pass — no win).

### Files
`src/vibevoice.cpp` / `vibevoice_tts.cpp` (chunked decode + graph reuse + causal
pad); `examples/cli/crispasr_backend_vibevoice.cpp` (`--stream` stdout PCM);
`examples/cli/cli.cpp` (`--tts-stream` flag); `examples/server/server.cpp`
(chunked-transfer, after #58); `docs/tts.md`; `LEARNINGS.md` (per-call ggml graph
overhead trap + graph-reuse cure).

### Out of scope for v1
Multi-chunk look-ahead beyond one chunk; kokoro/orpheus chunking as separate items;
any change to AR decoding (only post-AR codec/VAE side is chunked).

## 75-followups. /v1/audio/speech OpenAI round 2 — open

Parent #75 round 1 shipped → HISTORY §81 (PR #63 merged + corrective
batch + 75a/75b + 75d chunking + 75c-opt-1 server-side speed
resampler). Remaining follow-ups: 75c-opt-2 (native-backend duration
knobs) and 75e (streaming / mp3 / upload).

Remaining gaps documented in follow-up items: 75c-opt-2 (native-backend duration knobs), 75e (streaming response, mp3/opus encoding, voice upload/delete).

---

## 81. Nemotron-Speech-Streaming-EN-0.6B — first cache-aware streaming-native ASR

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** WER benchmarking on standard sets, GPU perf testing, decide whether streaming becomes default path

## 168. GPU scheduler migration — gallocr-only backends

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** voxcpm2_tts sched migration; re-publish lfm2-audio GGUFs at Q5_K minimum quant

## 91. CrispASR CLI features missing from CrisperWeaver (OPEN — parity gaps)

CLI knobs CrispASR exposes that CrisperWeaver doesn't. None blocking; listed so
the next parity-pass audit doesn't re-discover them. (`--alt N` shipped — see
HISTORY.)

**TO DO (open parity gaps):**
- ~~`--offset-t MS` / `--duration MS`~~ — DONE on **both** dispatch surfaces
  (multi-surface trap — the server has its own slice loop, see
  [[multi-surface-dispatch-trap]]):
  - **CLI** (`crispasr_run.cpp process_one_input`, feat/offset-duration): windows
    the decoded PCM to `[offset, offset+duration)` before VAD/chunking and shifts
    reported segment/word/token timestamps back into original-audio time. Was
    whisper-internal only (`cli.cpp` → `wparams.offset_ms`). Offset-past-end exits
    cleanly. Verified on parakeet-ctc + jfk: timestamps land at 5.0–11.0 s;
    word-level JSON offsets shifted too.
  - **HTTP server** (`crispasr_server.cpp do_transcribe`,
    feat/server-offset-duration): `offset_t_ms` / `duration_ms` form fields were
    parsed but never applied. Same window + shift, applied after the per-slice
    diarize re-walk (which matches segments by the unshifted slice `t0_cs`).
    Verified live via `crispasr --server --backend moonshine` + curl:
    `offset_t_ms=5000` → seg 5.00–11.00; `offset_t_ms=4000&duration_ms=3000` →
    seg 4.00–7.00; offset past end → `{"text": ""}`.

  No session-C-ABI change needed: `crispasr_session_transcribe` is a low-level
  PCM-buffer primitive — windowing is the caller's concern there. Docs in
  `docs/cli.md` + `docs/server.md`. The window arithmetic (initially copy-pasted
  into both surfaces) was factored into `src/core/audio_window.h`
  (`core_audio_window::compute` + `trim`) and unit-tested (`tests/test-audio-window.cpp`,
  9 cases, model-free) so the CLI/server can't drift and the logic is CI-guarded.
  **Still open:** the Dart binding + CrisperWeaver UI rows (out of this repo's scope).
- Whisper decoder fallback knobs (`--word-thold`, `--entropy-thold`,
  `--logprob-thold`, `--no-speech-thold`, `--no-fallback`, `--temperature-inc`)
  — already in Dart binding's TranscribeOptions; just add UI rows + l10n in
  CrisperWeaver Advanced Options. ~half a day.
- Subtitle line formatting: **`--max-len` + `--split-on-punct` already work for all
  backends** (applied post-hoc via `crispasr_make_disp_segments(all_segs, max_len,
  split_on_punct)` in `crispasr_run.cpp`, verified 2026-07-17). Still whisper-only:
  **`--split-on-word`** (referenced only in `cli.cpp`, no `crispasr_run.cpp` hookup).
- `--carry-initial-prompt` — sticky vs reset initial prompt across segments.
  Edge case, ~1 hour.
- ~~`--print-confidence`~~ — DONE (feat/print-confidence-nonwhisper). The flag
  was advertised in `--help` and parsed but did **nothing** for non-whisper
  backends (only the whisper path in `cli.cpp` honoured it) — a silently-broken
  flag, not a missing feature. `crispasr_run.cpp` now prints each segment's
  tokens with an inline `word[NN%]` annotation after the transcript, via a new
  `crispasr_print_confidence` in `crispasr_output.cpp` (gated `else if` after
  the `--alt` printer so the two don't double up). Verified on parakeet-ctc
  (`ans[85%]`) and moonshine (`my[67%] ask[61%] ,[42%]` — low-confidence tokens
  now visible); no flag → single transcript line unchanged. Docs in `docs/cli.md`.
  (JSON/WTS exports already surfaced per-token `confidence`; this closes the
  stdout half.)
- Token suppression (`--suppress-nst`, `--suppress-regex`) — niche,
  whisper-specific; lowest priority.

**Deferred `--alt N` follow-ups (low priority, v1 covers common case):**
beam-search alt capture (siblings ≠ greedy alts, different capture + UX);
full word-level alt enumeration (BPE sub-word: only first content token gets
alts today); alt-picker popover widget test (CrisperWeaver Riverpod + l10n).

## 92. All-backend regression suite (nightly CI)

**Status:** 32 ASR + 21 TTS regression entries, 0 PLACEHOLDERs. CI at
`.github/workflows/regression.yml` runs nightly (04:00 UTC cron) + PR smoke-only.
Matrix: 22 ASR + 7 TTS = 29 backends. First nightly (2026-06-16) green after
fixing 6 failures. Architecture shipped in `tests/regression/`: `manifest.json`
(per-backend GGUF revision SHA + ref path + expected transcript + cosine
thresholds), `run_one.py` driver, `regression.yml`. Fixtures pinned in
`cstr/crispasr-regression-fixtures` (`fixtures.revision` SHA pins the whole set).

**Next steps (each ~1 h/backend):**
1. Flip `skip_diff` on backends where ref archives exist.
2. **Add parakeet-tdt-0.6b-v3** (English) — cache the `.nemo` source locally first
   (`nvidia/parakeet-tdt-0.6b-v3` not on the dev box yet).
3. **Add canary + cohere + kyutai-stt + moonshine** — all have reference modules in
   `tools/reference_backends/`.
4. **Add the TTS family** (kokoro, indextts, qwen3-tts, chatterbox, vibevoice) —
   need WAV-output checksums or ASR-roundtrip rather than transcript equality.
5. **Promote to release gating** — hook into `release.yml` pre-publish job so a
   regression aborts the tag.

## 95. IndexTTS-1.5 Chinese TN — binary alternative to the Python `wetext` hook (OPEN)

Today CrispASR ships `INDEXTTS_TEXT_NORMALIZER=<shell cmd>` + `tools/wetext-normalize.py`
(commit `1bfe7c5a`) — covers users who already have Python + wetext. No-Python deployments
(single-binary, Windows w/o Python, embedded) need a native path. Do NOT start speculatively.

**Gap:** default in-process `preprocess_indextts_text()` handles CJK char split, a subset of
`char_rep_map` punctuation, ASCII upper-case. Missing vs full `wetext.Normalizer(lang='zh',
operator='tn')`: Arabic-numeral→hanzi, pinyin tone-digit restoration, dates, times, currency,
phone numbers, math/measurements/fractions/percent, EN contractions in Chinese. This isn't
cosmetic — the model often fails to emit `stop_mel_token` on un-pronounceable digit inputs and
burns `max_mel_tokens=600`, so digit-containing prompts break without TN.

Options (in preference order — implement per trigger below, not all):
- **95a. Hand-roll high-leverage rules in C++** (recommended first). digit-string→hanzi for
  the 1-billion range (`零一二三四五六七八九` + `十百千万亿`), `年/月/日`, `点/分` time,
  pinyin tone-digit lookup. ~300–600 LOC, no deps, covers ~90 % of prompts. Extend
  `src/indextts.cpp:preprocess_indextts_text` with a `normalize_chinese_numbers()` pass +
  golden string-in/string-out tests. **Effort: ~1 day.**
- **95d. Tiny FST reader in own C++** — consume upstream `pengzhendong/wetext` `.fst` files
  (`fsts/zh/tn/tagger.fst` + `verbalizer.fst`, ~1 MB) without linking OpenFST. Parser +
  Tropical-semiring traverser + symbol-table loader + port of wetext `token_parser.py`
  (~200 LOC). Total ~500–800 LOC, zero deps. Byte-stable vs upstream except FST features not
  implemented. New `src/indextts_zh_tn.{h,cpp}`; invoke via `INDEXTTS_TEXT_NORMALIZER=native`.
  **Effort: 2–4 days.**
- **95b. Vendor `kaldifst` + OpenFST + ship compiled `.fst`** — byte-identical to upstream.
  `third_party/openfst` (~30–50 K LOC) + `third_party/kaldifst` (~5 K LOC), both Apache-2.0;
  real build-profile cost. Invoke via `INDEXTTS_TEXT_NORMALIZER=wetext`. **Effort: 3–5 days +
  ongoing submodule maintenance.** Only as fallback if 95d's FST gaps keep biting.
- **95c. PyInstaller-bundle the Python sidecar** — ~50–80 MB bloat, per-platform pipeline.
  Almost certainly not worth it; listed so nobody re-discovers it.

Not alternatives (already surveyed as insufficient): ICU `Transliterator`, `cn2an`/`pypinyin`
(no C++ port), HF `tokenizers` normalizers.

**DECISION-GATE — start only when one of:**
1. A user files a digit/date/pinyin-tone-digit prompt that breaks audio → do 95a, smallest
   rule set that fixes the reported case.
2. Hand-rolled list reaches 2–3 entries (use case confirmed alive) → 95d becomes the right
   next step rather than letting 95a grow into one-off rules.
3. Only if 95d's unimplemented FST features keep biting → fall back to 95b. Never vendor
   OpenFST speculatively.

---

## 102. RapidTP-Aligns — dedicated NN timestamp predictor (survey)

Status: survey-only. RapidAI ships `RapidTP-Aligns` as a standalone timestamp-only model that
predicts timestamps **from audio alone** (no ASR text). Would be a third path alongside our native
decoder timestamps (whisper/parakeet-TDT/canary/cohere/kyutai-stt) and CTC forced aligners
(`canary-ctc-aligner.gguf` / `qwen3-forced-aligner.gguf`, which need the text as input).

Value when: cross-backend timestamp consistency independent of which ASR ran; trustworthy segment
boundaries even when ASR text is wrong (diarization/VAD post-proc); single-forward-pass
end-of-utterance/silence for streaming.

TODO (open questions to resolve before starting):
1. What architecture upstream ships (README is one Chinese line; likely small Conformer-CTC over raw audio emitting frame-level boundary labels).
2. License + upstream weights (ModelScope-hosted? FunASR derivative?).
3. Quality vs our existing CTC aligners (`canary-ctc-aligner-q4_k.gguf` ~80 MB is fast+accurate; without a clear margin this is incremental).

**Trigger (stays survey-only until one fires):** a user reports the CTC aligner failing on a specific
audio class (heavy code-switch, multi-speaker overlap, music behind speech); OR we add a streaming
ASR endpoint needing sub-100-ms end-of-utterance prediction (currently a VAD silence heuristic).

---

## 104. Stateful frame-streaming TDT decode for parakeet long-form (issue #89)

**Priority: HIGH** — auto path (no `--vad`, no `--chunk-seconds`) tops out at
~82 % coverage on 60 s Japanese audio; users expect >95 %. Root cause is the TDT
decoder cold-starting each independent chunk (`kLongAudioFallbackChunkSeconds=30`
reinitializes the LSTM per chunk), losing 5–20 s of content per chunk interior —
not boundary stitching. Fix: keep TDT LSTM predictor state across frames like
NeMo's `BatchedFrameASRTDT` (stateful decoder, 4 s rolling buffer, running z-norm).

### TO DO

**Phase 1 — stateful TDT decode (core change, ~200–300 LOC; hot loop <50 lines,
LSTM state threading is the work).** Split API already exists (`parakeet.h`:
`parakeet_encode`, `parakeet_decode_frames`).
1. Add `parakeet_decode_frames_stateful` — like `parakeet_decode_frames` but
   accepts/returns LSTM hidden state (`parakeet_lstm_state`). The TDT loop at
   `parakeet.cpp:1003` already uses `state`; make it an in/out param instead of
   initializing to SOS.
2. Add streaming mel with running z-norm — maintain running mean/variance across
   frames (NeMo `get_norm_consts_per_frame`) via a new
   `parakeet_mel_streaming_context`.
3. Wire into `crispasr_run.cpp`: when long-audio fallback triggers and backend is
   parakeet/canary, use streaming decode (per 4 s frame: streaming-mel → encode →
   decode_frames_stateful(&lstm_state) → merge with LCS) instead of independent
   chunk transcription.

**Phase 2 — tuning (benchmarking).**
4. Frame-size sweep (1.6s, 2s, 4s, 8s) on benchmark corpus.
5. Running z-norm warmup: first frame per-frame z-norm, later frames EMA.
6. LCS delay tuning: `lcs_delay = (buffer - frame) / model_stride`.

### Files
- `src/parakeet.h` — add `parakeet_decode_frames_stateful`,
  `parakeet_mel_streaming_context`
- `src/parakeet.cpp` — LSTM state in/out, streaming-mel helper
- `src/core/mel.{h,cpp}` — running z-norm mode
- `examples/cli/crispasr_run.cpp` — streaming decode path in `process_one_input`
- `examples/cli/crispasr_backend_parakeet.cpp` — streaming transcribe method
- `tests/test-issue-89-long-audio-fallback.cpp` — pin streaming path activation

### Effort: Medium-Large.
### Success: `python tests/benchmark_asr.py --audio yt_60s.wav --backend parakeet-ja --settings auto` reports **coverage ≥ 95 %** on the #89 JA audio, without `--vad`.
### Trigger: immediate — #89 open, current fix is partial mitigation (prevents 0-output but doesn't match NeMo quality).

## 105. WhisperX word alignment models — wav2vec2 CTC zoo

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** Benchmark new aligners vs canary-ctc. (`-am` aliases now documented
in `docs/cli.md` — canary-ctc, wav2vec2 [12 langs], fastconformer [18 langs],
qwen3-forced.)

## 106. TEN-VAD — low-latency cross-platform VAD

**Status:** feasible-but-BLOCKED-on-license; no code. Fourth VAD backend candidate
alongside Silero, FireRedVAD, MarbleNet, Whisper-VAD-EncDec. Upstream ships
cross-platform C bindings, prebuilt libs (Linux/macOS/Windows/Android/iOS/Web), and
an ONNX path; runtime target 16 kHz, 10/16 ms hop — fits our VAD surface. Aimed at
low-latency streaming turn detection, lighter than Silero.

**Decision-gate:** upstream license is Apache 2.0 **plus additional
no-compete/own-app-only conditions** — treat distribution as BLOCKED until legal
review or an explicit internal-only use case. Do NOT wire or ship until that's
resolved.

**Implementation plan (once unblocked):**
1. Decide prebuilt-native-lib path vs ONNX path (or both).
2. Add `ten-vad` alias in VAD registry + CLI so `--vad -vm ten-vad` works.
3. Add auto-download metadata for the chosen artifact(s); keep Silero default.
4. Run the boundary benchmark vs Silero + FireRedVAD on the short-gap /
   sentence-end test set.
5. Document 16 kHz-in sampling-rate handling (resample other inputs first).

**Trigger:** a user wants lower-latency/lower-footprint VAD than Silero, OR we want
a 4th native cross-platform backend — AND the license review clears (or an
internal-only path is confirmed).

## 125. Issue #125 — multi-backend bug sweep from montvid (mostly DONE — validation + longer-term threads open)

12 findings from user `montvid` (RTX PRO 6000 Blackwell sm_120, CUDA 12.6) on
CrispASR v0.6.10 `eaee2319`. All fix commits landed and M1-Metal-smoked; remaining
work is GPU validation + a few longer-term root-causes. Reports cached at
`/Volumes/backups/code/issue125-attachments/`.

**Open TODO items:**
- **P0 mimo-asr Blackwell segfault — hardening shipped (`a5a518c8`), needs GPU
  confirmation.** Real cause was `0f0f0793` (ggml-backend src-mutation
  log/restore), NOT the FA-mask commit. Ask montvid to rebuild from `95d74455`+
  and rerun `--backend mimo-asr -m auto --auto-download -f samples/jfk.wav -l en
  -np -nt`; expect the v0.6.9 reference transcript. If segfault persists: `gdb`
  backtrace; next suspects are Metal-debug commits leaking to CUDA (unlikely) or
  a Blackwell-specific ggml-cuda bug.
- **P1 funasr (longer-term):** root-cause the audio adaptor/encoder collapse on
  Blackwell CUDA (log `frames_spliced` in `funasr_init_from_file`; if 0 on 11 s
  JFK the adaptor is the failure; ideally diff adaptor output vs upstream FunASR
  Python ref). Loop guard + `-l` wiring already shipped.
- **P2 firered-asr (if it resurfaces):** the "JFK silence-only `<Sil>!` without
  --vad" subtask (suspected vocab/blank-id mismatch in auto-downloaded GGUF) was
  not reproduced locally; reopen only on a new report.
- **P4 gemma4-e2b (longer-term):** add init sanity logs `audio_soft_token_id`,
  `proj_dim` vs `d_model`, "audio projection weights found". Chunking + prefers_vad
  already shipped.
- **P6c kyutai-stt — DEFERRED:** streaming model on batch dispatcher is a
  footgun; a `--force-long-audio` cap is now UX-nice only (P6b bounded the
  wallclock). Defer until a user reports.

**Cross-finding open sweeps:**
- Wider GPU validation matrix before any future `ggml-cuda/fattn*` change: add
  mimo-asr, glm-asr, gemma4-e2b, voxtral, granite (multi-head audio-LLM backends).
- Do NOT submit `tools/upstream-prs/06-cuda-fa-perhead-mask.md` upstream until
  that wider matrix validation lands.
- Audit all 18 registered backends in `tools/test-all-backends.py` for silent
  staleness (4 were missing entries before this sweep).
- Defensive sweep of every `CAP_UNBOUNDED_INPUT` declaration — confirm the
  *encoder* is genuinely unbounded, not just the dispatcher's input shape.

**Completion triggers:** mimo-asr next release no longer segfaults on Blackwell
(montvid or same-GPU-class validation); 5-min EN clip clean on firered-asr /
omniasr-llm / gemma4-e2b.

Reporter contact: `montvid`, GitHub #125.

## 127. Coverage gaps from the 2026-05-26 overlap-save sweep close-out

Three small holes the sweep + #115 bisect surfaced. None are urgent; recording so the next contributor doesn't rediscover the same gaps.

### a. omniasr-llm — overlap-save bug status unknown

The original 5 min sweep and the 90 s rerun both came back `BOTH_EMPTY` for `omniasr-llm-300m-v2-q4_k.gguf`: default and `--chunk-overlap 0` both hit the 20 min per-pass wallclock on M1. Probably *slow*, possibly *also has the truncation bug like its sibling backends*. Can't tell without a faster box.

**Fix shape.** Re-run `./tools/check-overlap-save-bug.sh omniasr-llm` with `PER_RUN_TIMEOUT=2400` on a Linux x86 host (the VPS) or a Kaggle CPU kernel. If default produces materially less output than no-overlap, add to the opt-out list in `examples/cli/crispasr_chunk_context_gate.h`. If both produce the same content, mark VERIFIED-OK in the harness comment.

### b. mimo-asr — local test coverage is in place but doesn't run in CI

`tools/test-all-backends.py` has had a `mimo-asr` registry entry since 2026-05-02 (commit `2aeaf4c4`); the `test_transcribe` function explicitly handles `EMPTY` output (line 693-695). Locally the test would have caught PLAN #115's silent-empty regression at runtime — but CI doesn't run `test-all-backends.py` against large-model backends (mimo Q4_K is 4.2 GB, doesn't fit in the standard runner disk budget per pre-release), so the regression shipped in v0.6.10 anyway.

**Fix shape.** Either (a) Kaggle scheduled-CI workflow that runs the full `test-all-backends.py` against the 4 LLM-class backends (mimo, voxtral, gemma4-e2b, granite-4.1-2b) on each main push — patterns in `tools/kaggle/crispasr-regression.py` already handle the model-download + heartbeat parts; (b) cheaper, a documented `make smoke-llm-backends` target that release scripts run before tagging. (a) is more reliable; (b) is one afternoon of work.

### c. cohere-asr-ja-v0.1 — no benchmark numbers in PERFORMANCE.md

Issue #123 added the JA variant to the registry + README, but no row in any of `PERFORMANCE.md`'s cohere tables (the long-form coverage at line 1374, the cross-backend matrix at line 1535, the per-length wall-time table at line 1555). The English `cohere-transcribe` is benchmarked across multiple Japanese / English / multilingual clips; the JA fine-tune isn't.

**Fix shape.** Run the JA variant on the same fixture set the English one used (TedX / JSUT clips per the model card; `samples/jfk.wav` is English so won't exercise the JA tuning). Drop one extra row into each cohere table with the JA numbers. ~30 min of inference + table updates once the fixtures are downloaded.

---

## §136 — funasr CUDA !-loop fix (issue #125)

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** File upstream ggml issue with minimal repro; revert weight split if fixed

## §138 SpeechT5 + Dia + Parler + FastPitch TTS stubs → working backends

**Status:** SpeechT5 + Dia both run e2e and produce audio but have decoder-precision issues (ASR says music/noise); Parler + FastPitch not started.

### SpeechT5 TTS (microsoft/speecht5_tts)
- Encoder verified (cos > 0.999 all 12 layers). Runtime `src/speecht5_tts.cpp` = encoder + decoder w/ KV cache + postnet + HiFi-GAN. GGUF `/mnt/storage/speecht5/speecht5-tts-f16.gguf` (300 MB). Converter `models/convert-speecht5-to-gguf.py`.
- **TO DO:** decoder content mismatch — validate decoder per-layer against the Python reference.

### Dia 1.6B TTS (nari-labs/Dia-1.6B)
- Encoder cos=1.0 all layers; decoder layer-0 cos 0.999, step-0 argmax matches Python. Runtime `src/dia_tts.cpp` = encoder + cross-attn + AR decoder (18L GQA CFG) + DAC decode. GGUF `/mnt/storage/dia/dia-1.6b-f16.gguf` (3.2 GB F16); DAC `/mnt/storage/dia/dac-44khz.gguf` (104 MB). Converters `models/convert-dia-to-gguf.py`, `models/convert-dac-to-gguf.py`.
- Audio produced (2.15 s) but ASR says music/noise.
- **TO DO:** validate decoder layers 1-17; test with F32 GGUF; investigate DAC decode fidelity. Key sensitivity: Dia's `scale=1.0` attention (no 1/sqrt(d)) makes softmax precision-critical — every computation must match Python exactly or codes diverge.

### Parler TTS / FastPitch — NOT STARTED
- FastPitch: ~1000 LOC stub, converter exists, needs NeMo model.
- Parler: ~857 LOC stub w/ 3 TODOs, T5 encoder + DAC decoder.

## §139 Beam search — remaining ASR backends (issue #136 follow-up)

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** mimo-asr beam (blocked on PLAN #115) and lfm2-audio beam (needs KV+conv save/restore) still API stubs

## §TTS-PROV: TTS AI-provenance compliance (watermark, C2PA, consent, disclaimer)

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** AudioSeal diff-harness validation per status header

## 156. Permissive G2P phonemizer

Replaces GPLv3 espeak-ng static link with modular permissively-licensed
phonemization. **Phase 1+2 DONE (2026-06-07):** header-only `core/g2p_{en,de,fr,es}.h`
(LTS rules + IPA-dict loaders + neural GRU G2P for EN), `espeak_dlopen.h`,
`phonemizer.h/cpp` cascade with auto-download from HF `cstr/g2p-dicts`, wired into
`piper_tts.cpp`, 202 unit assertions + 4 live TTS→ASR roundtrips. Cascade:
pre-gen IPA dict → builtin CMUdict+neural+LTS → OLaPh MIT dicts → espeak dlopen →
espeak popen. See HISTORY for coverage table + phoneme-inventory fixes.

### Phase 3 — open
**a. More languages (need LTS rules; OLaPh MIT dicts already exist):** Portuguese,
Italian, Dutch, Swedish, Czech, Danish, Finnish, Polish. Japanese/Chinese/Korean
need rules from piper-plus. Optionally port piper-plus (ayutaz/piper-plus, MIT,
branch dev) FR (1197 lines) + ES (620 lines) — more thorough than ours (NFD norm,
PUA mapping, syllabification, stress). No German in piper-plus — our `g2p_de.h`
fills that.
**b. GGUF-embedded dicts (TTS.cpp pattern):** embed phonemizer rules /
CMUdict / neural weights per-model via `phonemizer.rules.keys` /
`phonemizer.rules.phonemes` arrays for zero runtime external deps.
**c. Gruut CRF (MIT, rhasspy/gruut):** dict + CRF G2P (18 MB SQLite + CRFsuite/BSD)
for higher-quality OOV (compounds/loanwords), langs de/en/fr/es/it/nl/pt/ru/sv/cs/
ar/fa/sw. Port: extract SQLite lexicon + CRFsuite model, write ~100-line C++
feature extractor, link libcrfsuite. Lower priority now OLaPh covers most words.
**d. Neural G2P weight distribution:** publish standalone `g2p_en.json` (MeloTTS v3
`melotts.g2p_en_json`, base64 ~4 KB); loader already in `g2p_en.h`.

### Files
`src/core/g2p_{en,de,fr,es}.h`, `src/espeak_dlopen.h`, `src/phonemizer.{h,cpp}`,
`tests/test-g2p-{en,de,fr,es}.cpp`, `tests/test-espeak-phonemize.cpp`,
`tests/test-piper-roundtrip.sh`.

## §167 — Beam search, MAES, and KV caching expansion

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** §167e-h (granite-nle/mimo/moss/lfm2) beam need A/B validation on real models

## §169 — Qwen3-ASR ChatML language prompt (non-English script output)

**Status:** DONE — landed with the #218 blueprint-prompt-contract work
(`3c3ba2c74`), not as a standalone §169 change; PLAN entry was stale. Both
required surfaces build the full ChatML prompt and honour `-l LANG`:
- **CLI adapter** `examples/cli/crispasr_backend_qwen3.cpp` (~L94-152): ChatML
  system/user/assistant turns; when a language is set, an assistant-turn prefill
  `language <Name><asr_text>` (or, behind `CRISPASR_QWEN3_SYSPROMPT_LANG=1`, the
  legacy system-turn `Transcribe the speech in <Name>.`), via
  `crispasr_iso_to_english_lang`.
- **Session C-ABI** `src/crispasr_c_api.cpp` (~L5247-5283): same construction
  mirrored inline (per HARD RULE #6), using per-call `lang` or sticky
  `source_language` + `ca_iso_to_english_lang`. The HTTP server inherits it via
  the session ABI (no separate qwen3 prompt build) — all three surfaces covered.

The real transcribe path is the adapter/session inline decode, NOT the
`qwen3_asr_transcribe` stub the old TODO named. `-l` is wired end-to-end; ChatML
special-token ids resolve through `qwen3_asr_tokenize` (vocab map handles
`<|im_start|>` etc.). No-language falls through to ChatML with an empty system
turn (auto-detect preserved). **Verified** on `samples/paraformer_zh.wav`
(qwen3-asr-0.6b-q4_k): correct Chinese script with `-l zh`, `-l en`, and auto.
Arabic romanization (the original report) uses the exact mechanism prescribed
here but wasn't re-verified locally (no Arabic fixture on the dev box).

<details><summary>original (stale) problem statement</summary>

**Problem:** Qwen3-ASR supports 30 languages but our `src/qwen3_asr.cpp` skips the ChatML prompt and builds bare `<|audio_start|>...<|audio_end|>` with no system/user message. Without the ChatML wrapper the model auto-detects language but may romanize non-Latin scripts (Arabic AA0010.wav → `istagel` instead of `استغل`). Explicit language selection (`-l ar`) currently has no effect. HF model uses:
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Transcribe the following audio in Arabic.
<|audio_start|><|audio_pad|>×N<|audio_end|><|im_end|>
<|im_start|>assistant
```

**TO DO — fix:**
1. Build the full ChatML prompt in `qwen3_asr_transcribe_with_probs` when `-l LANG` is set (map ISO 639-1 → English name → prompt text).
2. Fall back to bare audio-only prompt when no language specified (preserves auto-detect).
3. Wire `--language`/`-l` through the CLI adapter's transcribe call.

**Files:** `src/qwen3_asr.cpp` (~50 LOC), `examples/cli/crispasr_backend.cpp`.

**Effort:** MEDIUM — ChatML token IDs (`<|im_start|>` etc.) must be resolved from the GGUF vocab; the word-level timestamp alignment loop (`src/qwen3_asr.cpp` L2074-2091) needs adjustment since the prompt prefix shifts positions.

**Test:** Arabic audio from `atishay23/Arabic_Audio` (AA001.wav → should output `مرحبًا` in native script with `-l ar`).

</details>

## §176 Runtime optimization pass — 18/20 DONE, 2 open

16 items DONE at the 2026-06-20 audit (§176a,b,d–j,m,o–t); §176k + §176n later resolved.
Full audit → PERFORMANCE.md "Runtime optimization audit — 2026-06-20". Open:

#### §176c Migrate host-side KV to device-resident 4D tensors — OPEN but LOW-VALUE

**Do NOT implement without first measuring the KV round-trip fraction on the specific
backend** (`DIA_BENCH`-style instrument). Dia measured at ~1.2% of decode → DEFERRED,
not worth the non-bit-identical, high-risk device-KV rewrite (ggml KV write/read
ordering on Metal is the codebase's most bug-prone pattern). Compute-bound transformer
decoders are NOT KV-bandwidth-bound — the "dominant bottleneck" premise is wrong.
Genuinely-still-host-side backends (verified 2026-07-12 by code read):
- **SpeechT5** self-attn KV — host `std::vector<float>` that grows per step
  (`speecht5_tts.cpp:246-265`), re-uploaded whole every step (`:1035-1041,1084`).
  Cross-attn KV already device-resident (§202). Measure fraction first (likely <5%).
- **Pocket-TTS** self-attn KV — host, pre-sized to `max_seq` (doesn't grow,
  `pocket_tts.cpp:311-319`), but reordered past window re-uploaded per step
  (`:1207-1228`). Measure first.
- (Dia — measured ~1.2%, deferred. VoxCPM2/Parler/LFM2/KugelAudio already device-resident — do NOT re-chase.)
Approach if ever justified: IndexTTS/CSM 4D on-device `[head_dim, max_ctx, n_heads,
n_layers]` + `ggml_view_4d`/`ggml_cpy` writes; keep both paths gated; expect
low-single-digit-% ceiling. Effort: Medium/backend, expected <2% payoff — deprioritize.

#### §176k FireRed ASR self-attn — MOSTLY DONE, residual OPEN (LOW)

Shipped the real lever (env-gated matvec graph cache `CRISPASR_FIRERED_MATVEC_CACHE`,
default ON, bit-identical, pure Pareto). Residual: self-attn KV is a growing
`std::vector<float>` with scalar O(T²) scoring loop — for very long single-pass decodes
(hundreds of tokens) a pre-alloc 4D device KV + BLAS/ggml scoring could help there.
**Measure on a long clip before investing** — NOT the highest-impact lever for typical
clips. File: `src/firered_asr.cpp`.

#### §176l Kyutai STT: vectorized RVQ encode — MOSTLY DONE, remaining = default-flip

Fast path shipped gated: `rvq_encode_group` (`kyutai_stt.cpp:768`) calls
`core_rvq::encode_euclidean_per_stage` when `CRISPASR_KYUTAI_RVQ_FAST=1` (**default OFF**);
non-uniform dim / failure falls back to scalar. Helper + extraction/transpose unit-tested
(`test-core-rvq`, LABELS unit) — codes identical to scalar reference. Only end-to-end on a
real model is unverified.
- [ ] **To flip default:** run kyutai on a Kyutai STT GGUF (clip via
  `tools/kaggle/kyutai-stt-2.6b-convert`) with flag on vs off, assert emitted RVQ codes
  byte-identical, then measure speedup. Until then scalar stays default + reference.
  Effort: Small (one-clip validation). Files: `src/kyutai_stt.cpp`, `src/core/rvq.{h,cpp}`.

---

## §ARK — ARK-ASR-3B support (⚠️ EXPERIMENTAL / WIP; branch feat/arkasr-3b)

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** CUDA validation (only Metal validated); nice-to-have

## llama.cpp comparison — perf-tricks to adopt (ANALYSIS / OPEN)

Status: comparative study (2026-07-03) of what CrispASR shares with the ggml-org ecosystem
(whisper.cpp + llama.cpp `libmtmd`). Framing: we're *siblings* on ggml — kernel-level wins
(flash-attn, CUDA MMQ/tensor cores, CPU tinyBLAS/sgemm, Metal `mul_mm`, Vulkan coopmat, k-quants)
are inherited free by keeping ggml sync current; llama.cpp's real advantages are at the
orchestration layer (imatrix, mixed-precision quant CLI, speculative decode, continuous batching).
Full comparison tables archived to HISTORY. Actionable adoption items:

### Tier 1 — real gaps
1. **imatrix + per-tensor quant overrides — DONE, one loose end.** `crispasr-quantize` takes
   `--imatrix <file>` + iq4_nl/iq4_xs + requant-from-q8_0, and `--tensor-type <regex>=<type>`
   (llama.cpp-parity). Producer `src/crispasr_imatrix.{h,cpp}` (set `CRISPASR_IMATRIX_OUT`) installed
   on decode scheduler of the major ASR-LLM backends; A/B harness `tools/imatrix_ab.py` gates on
   transcript **CER** (cosine only tiebreaks). CC0 calib set at `cstr/crispasr-imatrix-calib`
   (+ `tools/imatrix-calib/`); Kaggle kernel `tools/kaggle/imatrix-quant/`. See `docs/quantize.md`.
   **OPEN:** wire the collector into the remaining (non-ASR-decoder) backends if we ever want imatrix
   there. **Corpus rule:** diversity + in-distribution + language coverage is decisive — a
   narrow/mismatched corpus makes imatrix *worse*; use CC0 Common Voice (clean license).
2. **FA default-on audit.** Upstream flipped flash-attn to baseline. Re-check our two known FA bugs —
   batched-FA corruption (§176h) and the Vulkan GQA-REPEAT-f16 path — against the current
   `FLASH_ATTN_EXT` in vendored ggml; some may already be fixed upstream.
3. **Keep ggml sync current — kernel wins are free there.** Verify pinned ggml is recent enough for
   Metal `mul_mm` (M1 encoder prefill), Metal FA with head_size_k≠head_size_v (#12612 — matters for
   partial-RoPE ASR encoders: ark rot32/hd64, higgs, parakeet), CPU RMSNorm+MUL fusion, MMQ, tinyBLAS.

### Tier 2 — conditional on serving story
4. **Symmetric quantized-KV (q8_0)** for Qwen-family audio-LLM decoders (ark, higgs-stt,
   moss-transcribe). Near-lossless, shrinks decoder KV. Must be symmetric K==V type or it silently
   drops off the fused FA path. Validate against per-head KV-stride bugs (#171) first.
5. **Continuous batching** if the server ever needs concurrent transcription. Bespoke single-stream
   decode can't multi-slot today; #171 shows per-slot KV isolation is a real hazard.
6. **Prompt-lookup / n-gram speculative decoding** for AR ASR heads (transcripts echo their own
   context → high acceptance). Heed §161: the draft must reproduce the target's exact realization or generation derails.

**Don't bother:** i-quants (low-bit breaks audio backbones; slow codebook decode on compute-bound
TTS/ASR), KV defrag (deprecated upstream; single-pass audio doesn't fragment), paged attention (not merged upstream).

**Don't converge to them on:** model breadth (~60 vs ~8 archs), per-model long-audio chunking
(overlap-merge beats their fixed-30 s), diffusion/flow TTS with batched CFG, neural diarization, voice cloning, single-file arch-autodetect UX.

### Verify-in-tree caveats before mirroring anything
- `ggml_rope_ext` arg order / `GGML_ROPE_TYPE_*` enum values vs pinned `ggml/include/ggml.h` (revised upstream).
- Current speculative-decode flag namespace (`--spec-*` vs older `--model-draft/--draft-max`).
- mtmd mel params (128 HTK bins) + LFM2-Audio/MiniCPM-o audio status are DeepWiki-sourced — confirm from `clip.cpp` if load-bearing.

---

## 221. Issue #89 hardening + v0.8.8 release

The #89 fix (VAD slice cap + per-slice single-pass + gap-fill) is shipped and
manually verified (see HISTORY + LEARNINGS). This section makes it durable and
gets it released. All subitems OPEN.

### 221a. CI regression guard for the JA long-form path — HIGH, small
Coverage is protected only by manual runs; next parakeet refactor could silently
re-break #89. Drive a live test from the reazon baseball fixture
(`cstr/crispasr-regression-fixtures` → `parakeet-tdt-0.6b-ja/reazon_baseball_14s/
audio.wav`): concatenate ×3 (42.2 s — crosses 30 s auto-chunk + 12 s slice cap),
transcribe via the **session ABI** (carries cap + gap-fill), assert (a) 岡本 ≥3×,
(b) last timestamp reaches the third repetition, (c) a byte floor. Wire into
`tests/` + `tests/env-live-tests.sh` (`CRISPASR_MODEL_PARAKEET_JA`,
`CRISPASR_FIXTURE_PARAKEET_JA`).

### 221b. HTTP server path audit + mirror — HIGH, small-medium
`crispasr_server.cpp` has its own slice loop (`crispasr_compute_audio_slices`
~line 410) separate from the CLI dispatcher and session ABI. Audit whether a JA
parakeet request via `/v1/audio/transcriptions` gets the slice cap + gap-fill; if
not (expected), mirror the policy — consult the backend's
`vad_slice_cap_seconds()` like `crispasr_run.cpp`, or route server parakeet
handling through the session-ABI path.

### 221c. Vulkan sanity run — MEDIUM, small
#89 reporter is on AMD/Vulkan (encoder instability worse there). Fix is
policy-level (slicing) so should transfer, but confirm once via MoltenVK on M1
(`GGML_VULKAN=ON` build, `VK_ICD_FILENAMES` gotcha — see LEARNINGS): yt_60s
default run should land ≈97 % recall like Metal.

### 221d. q4_k registry/UX guard — MEDIUM, small
parakeet-ja q4_k TDT decode is degenerate (repetition loop; pre-existing) while
CTC over the same file is clean. Two guards: (1) check what
`-m auto --model-name parakeet-ja` resolves to in
`src/crispasr_model_registry.cpp` — if q4_k, point at q8_0 (TDT byte-identical to
F16); (2) stderr hint when a JA parakeet GGUF with ≤q4 weights loads in TDT mode:
suggest `--parakeet-decoder ctc` or the q8_0 file.

### 221e. v0.8.8 release — after a–d
Everything since v0.8.7: #89 series (slice cap, gap-fill, session mirror, tools,
CTC-head GGUFs), #218 moss fixes, #217 --align-only docs. Per dev-guide release
process: notes from `git log v0.8.7..HEAD --oneline --no-merges` into
RELEASE_NOTES_v0.8.8.md; `scripts/bump-version.sh 0.8.8`; push main; wait green
CI; push tag; `gh release create` with notes; then remove the repo-root notes file.

## §222 Aligner model expansion — permissive-license fleet (OPEN)

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** parakeet-ja CTC aligner upload, first-word t0 clamp, OWSM-CTC eval

## §224 Issue #222 follow-ups — silero LID ggml graph (DONE) + diarize/firered CPU perf (OPEN)

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** Vulkan RADV verify, TitaNet batch/F16, chatterbox_campplus scalar cure.
(openvoice2 + firered_vad scalar cures already DONE — both use Accelerate `cblas_sgemm`
gated by `CRISPASR_OV2_FORCE_SCALAR` / `CRISPASR_FIRERED_VAD_FORCE_SCALAR`, verified 2026-07-17.)

## §226 irodori-tts GPU + codec GGUF fix

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** baseline JA generation quality WIP, DiT graph caching perf follow-up

## §227 starling comparison — CUDA decode-loop optimization options (OPEN)

sims1253/starling (CUDA-graph ASR inference, RTX 5090, bf16) benchmarks CrispASR and overlaps
our fleet (granite-speech, parakeet-tdt-v3, MOSS-Transcribe, qwen3-asr, ARK-ASR-3B,
higgs-audio-v3-stt, cohere-transcribe — several via our cstr/* GGUFs). Their claim: stock
transformers decode is launch-bound (GPU ~10% busy) → capturing decode steps + multi-step
token loops into CUDA graphs gives 27–1180× vs 3–66× stock, byte-identical, WER-verified.
Repo cloned to ~/code/starling (read 2026-07-06).

**Benchmark-fairness (cheap, reputational):** their CrispASR adapter
(`benchmarks/engines.py:722`, `scripts/bench_qwen3_crispasr.py`) times ONE full CLI subprocess
per clip — incl. process start + multi-GB F16 GGUF disk load + CUDA weight upload every rep —
while starling/stock numbers exclude model load and use warm reps. Cold-start seconds get
labeled engine speed. No crispasr numbers published yet, but the columns could appear anytime.
- [ ] Contact author: benchmark `crispasr-server` (resident model, matches their server mode)
      or parse our stderr phase timings; offer setup help / a resident-mode adapter PR.
      (**The doc to point them at now exists — see below.**)
- [x] **DONE** — `docs/benchmarking.md`: the fair-measurement contract (measure transcribe
      time, not cold start), three methods (server resident-model, in-process
      Session/ctypes = the apples-to-apples path, CLI-with-parsed-stderr-line),
      proof-of-work rules (non-zero exit/empty = FAIL; scale check; warmup + median +
      absolute ms), identical-load discipline, required reporting fields, and the
      phase-timing env vars (`CRISPASR_VERBOSE`, `CRISPASR_<BACKEND>_BENCH`,
      `CRISPASR_METAL_PROFILE`, `CRISPASR_FC_PROFILE`). Links the existing
      `tools/benchmark_asr_engines.README.md` rather than duplicating it, and is linked
      from README's doc index so third-party benchers actually find it. Key fact it
      documents: the CLI/server stderr line `transcribed Xs audio in Ys (Zx realtime)`
      already excludes model load (timer starts after init/decode/VAD) — so benchers
      should parse it instead of wrapping the process in `time`.

**DECISION-GATE — option 0 first, gate options 1–4 on it:** one Kaggle T4/P100 run measuring
GPU-busy% during granite/qwen3-asr AR decode (`nvidia-smi dmon` or nsys). If ggml decode is
already 60%+ busy, options 2–4 are duds (cf. §210 Metal ICB 1.8%, vibevoice CPU cache A/B 0%).
Only if ~10–20% busy (starling's stock baseline) → proceed.

Optimization options (ordered by evidence):
1. **Keep per-step graphs capture-friendly** (mostly done, free). ggml-cuda already captures
   stable per-step graphs. Audit other AR backends where a decode loop shares a sched with
   helper graphs (EOS classifiers, connectors) — alternating graphs on one sched defeats
   gallocr + capture. (#171 per-purpose dedicated scheds already fixed pred head / per-KV-path
   LM steps.)
2. **Multi-step token-loop unroll** (the real starling edge, CUDA-only). Unroll K greedy steps
   into one graph: in-graph `GGML_OP_ARGMAX` → get_rows embedding feedback → static per-step KV
   positions; host sync per-K instead of per-token; EOS checked per block; byte-exact for
   greedy; stable topology per (n_past bucket, K). Substantial engineering, EXACTLY the
   cached-graph minefield of #171/#184/#220 (invariant: one graph per sched, or
   last-allocated-only). **Do not start before option 0 numbers exist.**
3. **Self-speculative draft from CTC head (granite only).** Draft from granite's encoder CTC
   head, verify with the LLM — no extra model; we already have granite CTC infra. Caveat: their
   batched spec at B≥16 loses (0.76×); B=1 spec is the win.
4. **Fused RMSNorm/SwiGLU steps** — ggml-cuda already fuses some; only audit if option 0 shows
   launch-bound decode with capture ON.

Anti-options (skip — negative results transfer): INT8 weight-only quant on CUDA decode (SLOWER,
launch-bound not bandwidth-bound; quant is a CPU/Metal win only); shape-bucketed graph caches
without eviction (per-clip capture cost depresses RTFx at high shape diversity); torch.compile
encoder fusion (not byte-exact — fp32 upcast + BatchNorm amplification).

## §229 transcribe.cpp perf audit — vs handy-computer/transcribe.cpp (OPEN)

Static code read (not benchmarks) of the direct ASR peer `handy-computer/transcribe.cpp`
(ggml C/C++ STT, heavy roster overlap). They have the more performant core ASR
*engine* (CPU, flash, AR decoder, on-target encoder, features, streaming latency)
via uniform shared-infra; we win quantization QUALITY (imatrix, unmatched),
VAD-gated long-form robustness, and breadth (ASR+TTS). Full scoreboard + file:line
findings in HISTORY. Every port gated on decoded-output A/B (§176/§210/§227 rule).

### Actions (ranked)
0. **GATE on measurement (do first).** One M1 + one T4/CPU encoder-RTF A/B for
   wins 1-2. Port each only if it moves the encoder ≥5%.
1. [x] **Force `GGML_LLAMAFILE ON`** — DONE (verified 2026-07-17). `CMakeLists.txt:199`
   `set(GGML_LLAMAFILE_DEFAULT ON)` overrides ggml's default-OFF (`ggml/CMakeLists.txt:113`),
   landed with the §232 work. Note §232's closed-notes call the net effect "neutral on
   Q4_K/x86" but it stays ON as a low-risk default.
2. [ ] **Metal (+Vulkan) `GGML_OP_NORM_AFFINE` kernel** — our fused norm op
   (`ggml.h:500`, `ggml.c:3162`, used 7×/block in `core/fastconformer.h` + `core/sanm.h`)
   has ZERO Metal/Vulkan impl → falls to CPU with GPU↔CPU copies in the hot loop
   (~336/canary encode). Add the kernel + `supports_op` case
   (`ggml-metal-device.m:1392-1393`), OR gate `core_conformer`/`core_sanm` to plain
   `ggml_norm`+mul+add on Metal/Vulkan. Gate on M1 canary encode A/B.
3. [ ] **In-graph argmax + used-prefix-only KV snapshot** for cohere/AR decode.
   Ours (`src/cohere.cpp`) does full-vocab device→host + host max/softmax every
   token (`:2606,2614`). Beam search is the slow form: N sequential single-token
   forwards (`core/beam_decode.h:412-441`) each preceded by a full-KV deep-copy
   regardless of `n_past` (`core/attention.h:230-284`) — the #161 driver. Fix:
   in-graph `ggml_argmax` + 1-int readback (their `arch/cohere/model.cpp:1211-1213`);
   snapshot only the `n_past` prefix.
4. [ ] **Flash head-dim padding helper + real per-backend flash-capability gate.**
   Ours = ~40 scattered per-model `p.flash_attn` + reactive `getenv` guards, no
   head-dim padding, Turing sm_75 −9-10% fusion regression unhandled at model layer
   (fold in the opt-out from memory [[project_flash_attn_turing_regression]]). Ref
   their `pad_head_dim` (`arch/moonshine_streaming/decoder.cpp:37`).
5. [ ] **Consolidate per-backend pow2 FFT into one shared non-pow2 frontend.** Ours
   is per-backend pow2-only (`core/mel.h:33`), backends ship duplicate hand-rolled
   copies (`voxtral.cpp:446` "cargo-cult", `core/fft.h:6-8`). (Optional: switch
   miniaudio linear resample `crispasr_audio.cpp:406` → windowed-sinc if fidelity
   matters.)
6. [ ] (stretch) Utterance-batched encoder + batched multi-utterance decode for
   offline throughput — large, and the cached-graph minefield of #171/#184/§227.2.
   Do NOT start before action 0.
7. [ ] (cleanup) Bucket-table quant tensor classifier (their
   `tools/transcribe-quantize/policy.cpp:50-278`) to replace our per-arch if-ladder
   (`examples/crispasr-quantize/main.cpp:559-684`) — refactor, not a perf change.

Note: our single-stream conformer core is actually leaner (fused norm_affine,
fused GLU, load-time BN fold, zero-copy strided rel_shift `fastconformer.h:43-47`) —
port that rel_shift trick INTO them / keep it; §176 encoder-graph cache confirmed a
dud (keep `CRISPASR_PARAKEET_ENC_CACHE` OFF).

## §232 qwen3-tts code predictor perf — fuse 15 graph dispatches into 1 (#245, OPEN)

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** optional CPU direct-path rescue (avoid lm_head slot blit)

## §235 next perf targets — triage after the §232 qwen3-tts sweep (2026-07-11)

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** CV3 flow-steps default 10->6 flip pending human listen; transducer GPU decode pending parallel CUDA verdict

## §234 omnivoice — persistent step graphs + silence root cause

_Completed work archived to HISTORY.md (PLAN compaction 2026-07-17)._

**Still open:** voice-clone roundtrip validation, output-gain/clipping check, denoise token, rerun parity kernel

## §232 transcribe.cpp parity — close the RTF gap (mostly CLOSED; residual items OPEN)

**Status:** Engine is competitive (CA wins/ties most of 11 shared models vs
transcribe.cpp on Kaggle P100). Big losses (RNNT/TDT CPU-cblas decode, moonshine
CPU-only-from-CLI bug) are FIXED and default-flipped. GPU-forwarding audit done
across all CLI adapters. Remaining gaps are either architectural (moonshine-streaming)
or measurement/validation, all GPU-hardware-gated (Kaggle P100). Full history +
A/B tables → HISTORY.

### Open TODO

- **[HIGH, LARGE] Moonshine-streaming 18× loss — banded/blocked windowed attention.**
  Bottleneck is O(T²) sliding-window masked flash-attn (dense T_enc×T_enc F16 mask/layer
  ×6, wl=16/wr=4), NOT frame-by-frame encoding (encoder runs once, `moonshine_streaming.cpp:808`;
  masks can't be dropped — LEARNING 17 degenerate output). Need a banded flash kernel
  (no native banded flash in ggml). Own campaign. Keep `--backend moonshine` for offline.

- **[MED] Extend persistent ggml decode to beam/RNNT/maes paths.** Greedy TDT/RNNT decode
  is ported+flipped, but `parakeet_rnnt_decode`, `parakeet_tdt_beam_decode`, `*_maes_decode`,
  nemotron beam still call cblas `predictor_step`/`joint_step` beam_size× per step. Reuse
  `core_rnnt_ggml::Decoder` (one Decoder serves all hypotheses, state passed per call).
  Low risk, proven pattern. Needs a parakeet-rnnt model to validate the RNNT path.

- **[MED, GPU] dia TTS — widen default beyond Metal.** Default GPU on Metal only; CUDA/Vulkan
  opt-in (`DIA_TTS_GPU=1`). get_rows contiguity crash fixed (`65a5d30c`) but CUDA greedy
  tokens DIVERGE from CPU at step 0 (M1 Metal matched, argmax 568). TODO: ASR-roundtrip the
  CUDA-generated audio (the real HARD-RULE-#3 test) — if intelligible it's benign FP → widen
  default; if garbled, bisect the decode graph for another non-contiguous/precision-sensitive op.
  `load_weights_split` (enc+DAC→GPU, decoder→CPU) is the fallback.

- **[MED] Re-run P100 competitive scoreboard (v16).** After decode flips + moonshine GPU-forwarding
  fix (`d46839ca`), re-measure parakeet/nemotron/moonshine TOTAL RTF vs transcribe.cpp; update
  `docs/performance.md`. Run v16 kernel to confirm moonshine P100 CA ~0.012-0.015 (would make CA lead 6–5).
  Kernels: `tools/kaggle/parakeet-ggml-decode-ab/` (P100, exercises persistent path),
  `tools/kaggle/gpu-pin-ab`. Measurement, not a code change.

- **[LOW] In-graph argmax for transducer greedy path.** 2 int32 readbacks vs 8198-logit readback,
  greedy no-hotword no-sampling path only (hotword-bias/temperature still need full readback).
  Minor now that persistent-graph decode won. `parakeet.cpp` predictor/joint step.

- **[LOW] Parakeet same-version A/B (Fix 5).** CA uses parakeet-tdt-0.6b-v3 (25 EU langs) vs TC's
  v2 (EN-only) — the residual gap may be the model. Download v2 GGUF, benchmark both, note in docs.
  Model local: `parakeet-tdt-0.6b-v3-q4_k.gguf` (467 MB). Quick, VPS.

### Decision gates / notes
- No more optimisations on VPS — all remaining wins need GPU hardware (Kaggle) to validate+measure.
- Flip a GPU default only if transcript/WER parity holds AND GPU decode < cblas decode ON THE TARGET
  platform. Metal is excluded from the RNNT/TDT ggml-decode flip (Accelerate cblas beats it there;
  P100 win is a slow-OpenBLAS effect — LEARNING 34). Overrides: `PARAKEET/NEMOTRON_GGML_DECODE=1/0`,
  `RNNT_GGML_PERSTEP`, `CRISPASR_{PARAFORMER,M2M100,T5}_GPU=1/0`, `MOONSHINE_ALL_GPU=1`,
  `MOONSHINE_ENC_ATTN=manual`, `DIA_TTS_GPU=1/0`.
- Closed (no action): moonshine decode hybrid placement, moonshine encoder manual-attn (slower,
  opt-in), f5_tts GPU (7.8× base fix; cheap levers tapped out — needs Metal-capture profiling),
  paraformer/m2m100/t5 GPU (flipped GPU on CUDA/Vulkan), titanet/diarize (Accelerate-BLAS, GPU port
  poor EV), GGML_LLAMAFILE ON (neutral on Q4_K/x86, kept as default).

## Runtime speedup roadmap (2026-07-11 cross-repo sweep)

Status: full ASR+TTS+codec+pipeline re-verification 2026-07-11 (see PERFORMANCE.md "Runtime
Optimization Audit — Re-verification (2026-07-11)"). **Gate: every item needs a target GGUF (q8_0
preferred, to isolate from q4_k quant noise) + before/after parity + latency — do NOT land a perf
change on a compile-only check.** Note: most per-model TTS "flash not wired" claims are false — flash
reaches Orpheus/OuteTTS/Zonos/TADA/Chatterbox/CSM via shared `core_attn::kv_self_attn`
(`src/core/attention.h:665,903`). Real flash gaps are only manual-`soft_max` backends (dia, speecht5,
parler) and structurally-can't-flash relpos models (melotts, piper).

### Maps onto existing tracked items (don't duplicate)
- Decode-step graph cache for remaining LLM/AR backends → **§210 follow-up** (shape-stable bucketed
  decode / CUDA-graph capture). Templates: qwen3-tts Lk-bucket, granite §210 gallocr, mimo `step_t1_gf`.
- Batched-CFG (B=2) for remaining TTS → **§215**. Un-migrated diffusion/DiT targets: f5, dots,
  kugelaudio, pocket (+ dia/speecht5/parler once they get device KV). Respect Metal quant-B=2 gotcha
  (dequant batched-against weights q*→F16 once) and item-24 (don't CPU-batch a GPU pipeline).
- gallocr cross-call UAF audit → **#215e** / encoder-graph-cache removal (#235). Encoder-graph caching stays OFF (measured dud + GPU UAF).

### Shared cross-repo Tier-1 levers (coordinate with CrispEmbed PLAN.md)
- Decode-step graph cache — same design as CrispEmbed Tier-1 #1; CrispASR is further along (§210 CUDA-graph-capture template).
- ggml-metal ICB replay — Apple-side equivalent of §210's CUDA-graph capture; ggml-metal has no ICB path. Depends on a stable per-step graph. Shared ggml submodule — do once, both repos benefit.

### Genuinely-new gaps (not yet tracked)
| P | Area | Gap | File |
|---|---|---|---|
| P0 | melotts / piper | Scalar O(H·T²·D) relpos attention (can't flash — additive bias); HiFi-GAN 17.9s of 26.3s VPS total. Needs manual-attn ggml graph or BLAS | melotts.cpp, piper_tts.cpp |
| P0 | voxcpm2_tts | CPU-only (Metal SIGSEGV); manual per-step host KV re-upload | `voxcpm2_tts.cpp:106-111` |
| P0 | openvoice2 | 16-layer WaveNet + ref-encoder Conv2d/GRU scalar CPU | openvoice2.cpp |
| P1 | voxtral/voxtral4b enc, mimo LLM dec | Attention not on flash_attn_ext (manual soft_max) | voxtral4b.cpp, mimo_asr.cpp |
| P1 | firered/glm/funasr/qwen3/omniasr/mimo | Beam = replay; add KV snapshot pool (canary/moonshine/kyutai template) | — |
| P2 | scalar CPU hotpaths | RNN-T LSTM pred+joint; granite cpu_linear+depthwise; rvq encode; istft IRFFT; titanet mel; diarize `apply_xcorr` | core/rvq.cpp, core/istft.h, titanet.cpp:740 |
| P2 | parakeet/nemotron | Batched sgemm decode opt-in default-OFF — validate + flip on | `CRISPASR_TDT_BATCH` |
| P3 | threading | Hardcoded default-4 threads in ~90 sites; adopt whisper-core's `min(4, hw)` | crispasr_c_api.cpp |
| P3 | pyannote | Runs per-slice not once-over-audio (#107); RNNoise recreates state+resamplers/call | crispasr_diarize.cpp |

(Struck P0 firered_asr — self-attn KV already cached, remaining O(T²) is scalar attention *scoring*
that often loses on M1/CPU — and P2 align_wav2vec2_ctc — now on the §176e resident cache — dropped as
non-gaps; detail in HISTORY.)

---

## §246 issue #81 endgame — close the remaining ~1.4× CUDA gap to onnx-asr (OPEN)

Current: parakeet-ctc q8_0 CUDA manual-attn = 153× RT warm (jfk×5 55 s,
in-process) vs onnx-asr CUDA fp32 = 207× (134 s varied). ~0.36 s/55 s left to
find. **Gate: measure before building** — the handover's bottleneck theory was
wrong; profile first.

**TO DO (ranked by expected value):**
1. **Per-stage split on CUDA first** (cheap, decides everything below): extend
   `tools/kaggle/fc-unified-graph-ab` to run `CANARY_CTC_BENCH=1` +
   `CRISPASR_FC_PROFILE=1` on P100 — mel vs encoder+ctc vs readout. Mel is
   host-side single-threaded FFT (`cc_fft_r2c`); may be a triple-digit-ms
   constant onnx doesn't pay. If so: parallelize `core_mel` (unused
   `mel_parallel` flag already in mel.cpp) or overlap mel with previous graph's
   compute.
2. **F16 vs Q8_0 on GPU**: P100 has 2:1 fp16, no tensor cores; onnx runs fp32
   cuBLAS. Our q8_0 mmq may lose to plain f16/f32 GEMM at these shapes — one
   kernel arm with the F16 GGUF answers it.
3. **CUDA-graph replay**: verify ggml-cuda graph capture engages across our
   rebuild-per-call graphs (same topology → should). If not, `CRISPASR_FC_BUCKET`
   gives stable topology; retry small buckets (100 mel frames ≈ 1 s; smaller
   pads waste less).
4. **Upstream flash fix (structural)**: teach `fattn.cu` to accept per-head masks
   (`mask->ne[2] != 1` guard) so flash works for Shaw rel-pos models on CUDA —
   reclaims fused-attention traffic manual attn re-materializes ((T,T,H) ×24).
   Upstream PR to ggml-org/llama.cpp per repo convention (mechanical-AI
   disclosure only).
5. **Honest re-run**: canonical number is the 134 s-varied load-excluded
   methodology (issue81-onnx-bench), not jfk×5.

**Also OPEN:** VPS 4-core x86 re-bench with shipped defaults (pre-fix 2.1× vs
onnx-CPU 3.1×; handover synced at `handover-prompts/issue81-fc-perf.md`).

## §247 roll #81 techniques out to the other runtimes (OPEN)

Techniques shipped for the FastConformer family (fastconformer.h consumers: parakeet,
canary, canary_ctc, canary_qwen, lfm2_audio, nemotron); roll each out where it applies.

- [ ] **(1) F16-weights-in-quantized-GGUF audit** (the 35%-of-encoder trap): converters
  storing matmul-consumed weights 3D/1D get them skipped by the quantizer's 2D rule → run
  on the ~6×-slower CPU F16 mul_mat path. Method: list tensors with `GGUFReader`, flag F16
  tensors feeding `mul_mat`; or run per-node profiler and look for `MUL_MAT f16` high-%
  rows. Suspects: cohere-transcribe, firered-asr, granite/conformer_ibm, sanm/paraformer
  conv, moonshine, TTS vocoders (hifigan/seanet/dac k=1 — qwen3-tts FASTCONV fixed the cast
  side, not storage). Fix: quantizer carve-out (+Q8_0 floor + idempotency) + load-time
  repack via `core_conformer::repack_conv_pw_q8` + fleet requant kernel
  (`tools/kaggle/fc-pw-requant`).
- [ ] **(2) Generalize the per-node profiler**: move `cc_prof_cb` (sched eval callback,
  aggregates by op+src-type+shape, `CRISPASR_FC_PROFILE=1`) to `src/core/sched_prof.h` so
  every sched-based runtime gets it.
- [ ] **(3) Fused QKV** (`core_conformer::fuse_qkv` is tensor-generic): bit-identical ~free
  win wherever Q/K/V share an input. Already deployed on ~10 backends
  (parakeet/canary/canary_qwen/canary_ctc/lfm2_audio via `core_conformer::fuse_qkv`
  under `CRISPASR_FC_FUSED_QKV`; voxtral/voxtral4b/qwen3_asr/higgs_stt/qwen3_tts have
  their own env-gated fused impls; nemotron deliberately opts out). **Still open** for
  the remaining named targets: whisper encoders, cohere, firered_asr, granite_speech,
  sanm/paraformer, AED decoder self-attn.
- [ ] **(4) Strided flash inputs**: grep `ggml_cont` feeding `flash_attn_ext` across src/ —
  the kernel reads strided views (llama.cpp does); each cont is a full tensor copy/layer/pass.
- [ ] **(5) Manual-attn-on-CUDA gate**: any backend whose flash mask has ne[2]>1 (per-head
  bias) silently falls back to CPU on CUDA — `GGML_SCHED_DEBUG=2` on a CUDA box shows flash
  nodes on CPU splits. Check cohere-transcribe (Shaw rel-pos) first. Reuse `fc_gpu_manual_attn`
  + BlockParams.manual_attn.
- [ ] **(6) -inf pad masking + bucketed persistent graphs** (`CRISPASR_FC_BUCKET`): reusable
  base for batched inference + CUDA-graph capture; correct padding for future streaming/batching
  (finite mask constants get overrun by pad garbage — LEARNINGS 2026-07-12).
- [ ] **(7) Q8_0 floor for decode-critical tensors in sub-8-bit quants** (quantizer): conv pw
  done; consider same floor for other high-sensitivity small tensors flagged by future A/Bs.

Suggested order: (2) profiler → (1) audit sweep with it (one Kaggle CPU kernel over the
registry, collect `MUL_MAT f16` % per backend) → fix top offenders → (4)/(3) mechanical
wins alongside → (5) after the §246 CUDA per-stage data.

---

## §248 — audio.cpp competitive benchmark + gap analysis (IN PROGRESS)

**Context:** #274 flagged [0xShug0/audio.cpp](https://github.com/0xShug0/audio.cpp) as
a comparable C++ inference engine. Audit performed 2026-07-19. They share ~6 ASR and ~12
TTS models with us; the rest is non-overlapping. CrispASR is broader on ASR (43 vs ~6)
and TTS (48 vs ~12) but audio.cpp covers categories we don't touch yet.

### (A) Head-to-head benchmark (shared models)

Must-compare on Kaggle (GPU + CPU kernels):

| Model | audio.cpp claim | Our backend | Metric |
|-------|----------------|-------------|--------|
| Voxtral Realtime | RTF 0.089 (11.2×), 15.7× Q8_0 | `voxtral4b` | RTF, WER on librispeech-test-clean |
| VibeVoice TTS 1.5B | 5.15× realtime (93.9 min podcast in 18.2 min) | `vibevoice-tts` / `vibevoice-1.5b` | RTF, MOS-proxy (ASR roundtrip) |
| Qwen3-ASR | (no claim) | `qwen3` | WER, RTF |
| Higgs Audio STT | (no claim) | `higgs-stt` | WER, RTF |
| Nemotron 3.5 | (no claim) | `nemotron` | WER, RTF |
| Chatterbox TTS | (no claim) | `chatterbox` | RTF, ASR roundtrip |
| IndexTTS2 | (no claim) | `indextts` | RTF, ASR roundtrip |
| OmniVoice | (no claim) | `omnivoice` | RTF, ASR roundtrip |

**Approach:** build audio.cpp on Kaggle (their CMake), run their CLI on the same WAV
files we use, collect wall-clock + output text, compare WER/RTF side-by-side.

### (B) Models they have that we lack — new backends to port

All licenses verified 2026-07-19. Only open-licensed models listed (Vevo2 is
CC-BY-NC-4.0 = non-commercial, skip; Stable Audio is Stability Community License =
commercial-restricted, skip).

**Source separation (new category):**
- [x] **HTDemucs** — hybrid transformer demucs, music/voice separation. **MIT** (Meta).
  **DONE (2026-07-19, VPS+Kaggle).** ~2720 lines C++. **Full parity** with Python.
  12-point checklist verified. All ops implemented:
  STFT→CaC→norm→encoder(Conv2d+DConv+GLU+freq_emb)→channel_up→
  2D/1D sin pos emb→LayerNorm→CrossTransformer(5 layers, self+cross attn)→
  channel_down→decoder(skip+GLU+ConvTranspose2d+time_decoder)→
  CaC unmask→iSTFT+time_denorm→per-source stereo PCM.
  All wired: CLI adapter + factory + GGUF auto-detect + model registry
  (Q4_K default) + `--separate` dispatch + C API session + Python binding +
  Go LDFLAGS. GGUFs on HF (`cstr/htdemucs-GGUF`): F16 (81 MB), Q8_0 (53 MB),
  Q4_K (38 MB). Kaggle: 21 reference stages validated. VPS (8 GB) can't run
  inference (swap pressure); validated on Kaggle (16 GB).
- [x] **Mel-Band RoFormer** — frequency-band source separation. **MIT**.
  The #274 reporter specifically mentioned this. Do both — they're complementary
  (HTDemucs = 4-stem, RoFormer = vocal/instrumental).
  **TAKEN** (M1/Metal session, 2026-07-19). Picked for this box because it is
  **non-autoregressive** — no sampling, so the diff harness gives deterministic
  per-stage cos verdicts with no torch-vs-mt19937 RNG mismatch (unlike an AR TTS
  port), and the model is small enough to run the Python reference and the C++
  port in 16 GB. No NeMo dependency (NeMo import is broken on this Mac).
  Regime: read the Python blueprint line-by-line → `tools/reference_backends/
  mel_band_roformer.py` dumper → converter → C++ backend → per-stage diff →
  **acceptance = decoded-output roundtrip** (separated stems judged by SDR/ASR
  on the vocal stem, not by cos alone — HARD RULE #3).
  > **Coordination with the in-flight HTDemucs work (same category):** HTDemucs
  > is unchecked above but IS being worked (converter `a6a447587` + 21-stage
  > reference dumper `60ada0a06`). We share the new `--task separate` surface
  > (CLI flag, stem output/WAV writing, backend-capability bit). **Whoever lands
  > that scaffolding first owns it; the second builds on it rather than adding a
  > parallel one.** Per-backend files (`src/mel_band_roformer.*`, converter,
  > reference dumper, registry/CMake entries) are additive and conflict-free.

**Voice conversion (new category):**
- [ ] **Seed-VC** — zero-shot voice conversion. **GPL-3.0** (Plachta/Seed-VC on HF).
  Encoder + flow-matching + vocoder. Medium effort. GPL is fine for our Apache-2.0
  project as long as the model weights are loaded at runtime (not statically linked).
  **Scoped 2026-07-19:** Tiny XLSR variant = 142 MB (25M params), fits 8GB VPS (~3 GB
  RAM with PyTorch). Architecture: XLSR→CAMPPlus speaker emb→length regulator→DiT
  UViT flow-matching (30 ODE steps)→HiFi-GAN vocoder. Risk: flow-matching requires
  noise pinning for deterministic diffs; FunASR+ModelScope deps are heavy. Repo archived.

**TTS:**
- [ ] **Supertonic 3** — claims 200×+ realtime on CUDA. **OpenRAIL-M** (Supertone/
  supertonic-3 on HF). Permissive (attribution + responsible use).
  **Scoped 2026-07-19:** ~400 MB total (4 ONNX components: text_encoder 36 MB,
  duration_predictor 4 MB, vector_estimator 257 MB, vocoder 101 MB). Non-autoregressive
  flow-matching (5-12 steps). 44.1 kHz output, 31 languages, 10 voices. Runs on
  Raspberry Pi (~800 MB RAM). ONNX-only distribution = custom ref approach needed
  (dump ONNX intermediate tensors, no PyTorch hooks). Deterministic diffs. **VPS-ready.**
- [x] **MioTTS** — voice cloning TTS. **Apache-2.0** (Aratako/MioTTS-0.6B, Qwen3-based).
  **CODEC PARITY ACHIEVED** (2026-07-19). Full pipeline: Qwen3 LLM (28L, 1024d, GQA
  16/8, head_dim=128, vocab=164480) + MioCodec-25Hz-24kHz (FSQ → wave_prenet → conv →
  ResNet → AdaLN-Zero decoder → ResNet → iSTFT → 24kHz waveform).
  **Parity results (diff harness):**
  - FSQ dequant: cos=1.0, wave_prenet: cos=1.0, audio: cos=0.999 (F32)
  - Q8_0 (723 MB): audio cos=0.954 — acceptable for TTS
  - Q4_K (397 MB): audio cos=0.069 — codec needs F16 for quality
  - LLM forward: argmax matches Python (tokens correct), tail-cos lower
  **Remaining (non-blocking for audio generation):**
  - Tokenizer integration (Qwen3 BPE + ChatML) for `miotts_synthesize()`
  - CLI adapter (`crispasr_backend_miotts.cpp`) + registry + 12-point checklist
  - Mixed quantization (LLM=Q4_K + codec=F16) for optimal quality/size
  - HF upload: F16 + Q8_0 GGUFs with README
  - Voice cloning: global encoder (reference audio → 128-d embedding)

**Audio codec:**
- [x] **MioCodec v2** — standalone audio codec. **MIT** (confirmed on HF + GitHub).
  **TAKEN** (VPS session, 2026-07-19). 133M params, ~530 MB F32. Architecture:
  WavLM-base+ encoder → FSQ quantizer (1 codebook, 12800 vocab, 25 Hz) → iSTFT
  decoder. Encode+decode is the full pipeline (no external vocoder for v2). Also does
  standalone voice conversion (swap global embedding). ~1.2 GB RAM — fits easily.
  Unblocks MioTTS port (shared codec). Deterministic diffs (feed-forward, no sampling).
  Regime: read Python → converter → backend → per-stage diff → roundtrip (encode→decode
  parity with original audio).

**Music generation:**
- [ ] **ACE-Step 1.5** — music generation. **Apache-2.0**. 3.5B params, 8.3 GB.
  **Scoped 2026-07-19:** mT5 text encoder + 3.5B linear transformer flow-matching +
  DCAE latent codec + vocoder. Kaggle-only (weights alone exceed 8GB RAM). 4 min max.
- [ ] **HeartMuLa** — music generation. **Apache-2.0** (HeartMuLa/HeartMuLa-oss-3B).
  **Scoped 2026-07-19:** 4B params, ~16 GB F32. AR LM + HeartCodec (12.5 Hz). Kaggle-only.

**Diarization:**
- [ ] **Sortformer** — NVIDIA dedicated diarization. **NOT Apache-2.0** — streaming v2
  is CC-BY-4.0, v2.1 is NVIDIA Open Model License. ~470 MB, 4-speaker hard cap.
  **Scoped 2026-07-19:** End-to-end (no VAD/clustering pipeline). Fast-Conformer 18L +
  Transformer 18L → per-frame 4-speaker labels. Community ONNX exports exist. ~750 MB
  RAM via ONNX, ~125 MB Q4_K GGUF. VPS-feasible but: (a) license isn't Apache-2.0,
  (b) 4-speaker cap vs pyannote/MOSS unlimited, (c) need quality eval first.

**Skipped (license incompatible):**
- ~~Vevo2~~ — CC-BY-NC-4.0 (non-commercial only)
- ~~Stable Audio 3~~ — Stability Community License (commercial requires separate agreement)

### (C) Priority

1. **Benchmark first** — head-to-head on shared models (Kaggle GPU kernel). If we're
   slower on Voxtral/VibeVoice, fix perf before adding scope.
2. **Source separation** — HTDemucs + Mel-Band RoFormer (both MIT, #274's ask).
3. **TTS** — Supertonic 3 (speed benchmark), MioTTS (voice cloning, Apache-2.0).
4. **Voice conversion** — Seed-VC (GPL-3.0, runtime-loaded weights = OK).
5. **Music generation** — ACE-Step + HeartMuLa (both Apache-2.0, new category).
6. **Diarization** — Sortformer (evaluate vs moss-diarize first).

---

## §249 WebRTC VAD — remaining follow-ups (backend DONE 2026-07-19, see HISTORY)

- [ ] `tools/sync_go_cgo_ldflags.py` — add `-lwebrtc-vad` (CI will flag)
- [ ] Dedicated live test comparing segment boundaries vs Silero/FireRed

## §250 — Music transcription backends (piano_transcription + Basic Pitch)

**Context:** CometBeat TRANSCRIPTION_SOTA_HANDOFF.md lists these as CrispASR
ggml port targets. New category: audio → note events (MIDI).

**Models:**
- [x] **piano_transcription_inference** (ByteDance/Kong, Apache-2.0). CRNN: 4×
  (4-layer Conv2d + 2-layer BiGRU) sub-networks (frame/onset/offset/velocity),
  88-key output at 100fps. ~172 MB checkpoint, ~86 MB F16 GGUF. **TAKEN** (VPS
  session, 2026-07-19).
- [ ] **Basic Pitch** (Spotify, Apache-2.0). Lightweight CNN (~10 MB). Polyphonic
  audio → MIDI. After piano_transcription.
- [ ] **MT3** (Google, Apache-2.0). Seq2seq multi-instrument. Large (~1 GB+).
  Feasibility check first.

**New CLI surface:** `--task transcribe-music` / `--backend piano-transcription`
→ MIDI output file.

---

## §251 — Music-analysis infrastructure + the remaining CometBeat roster

**Context:** §250 covers note-event models. This covers the shared DSP those
models need, plus the roster items still absent. Status reply written back to
CometBeat: `mus-textbook/docs/TRANSCRIPTION_CRISPASR_STATUS.md`.

**Already shipped, for the record** (so nobody re-ports them): CREPE F0
(`src/crepe.cpp`, `--pitch`, Dart + WASM + pub.dev `crispasr 0.8.16`), HTDemucs
and Mel-Band RoFormer separation, piano_transcription (§250).

### Phase 0 — shared DSP (blocks the rest)

- [x] **`core/cqt.h`** — constant-Q transform. **DONE** (`src/core/cqt.h`,
  `tests/test-core-cqt.cpp` 726 assertions, `tools/cqt_librosa_parity.py`).
  Direct time-domain Brown kernels, not librosa's recursive downsampling.
  Measured vs librosa 0.11.0 at BTC params: **per-frame shape correlation median
  0.9999** (mean 0.9721, min 0.1136), **peak-bin exact match 97.6%**. The mean is
  dragged down by exactly 3 transition frames + the tail frame, where librosa's
  per-octave group delay differs from our uniform centring — steady state agrees
  to 0.9999, and for 10 s BTC segments that latency is immaterial. Frame count
  differs by one at the tail; align on the leading edge. **CometBeat can reuse
  `tools/cqt_librosa_parity.py` as their Dart CQT oracle** — it is the exact
  parity harness their handover asks them to budget for.
  ⚠️ Cost is O(n_bins · N_k)/frame; N_k ≈ 24k samples at the bottom bin. Fine
  offline at hop 2048, NOT a real-time path — add a sparse spectral-kernel
  variant behind a flag if one is ever needed.
- [x] **`core/gru.h`** — uni/bidirectional GRU. **DONE**, validated against a
  real `torch.nn.GRU` fixture: **max_abs < 1e-5**
  (`tools/gen_gru_reference.py` → `CRISPASR_GRU_REF` → `tests/test-core-gru.cpp`;
  the parity case skips without the fixture so CI needs no torch/network).
  Two traps documented inline because both yield plausible-but-wrong output:
  (1) the reset gate multiplies the RECURRENT TERM `r*(W_hn h + b_hn)`, not
  `W_hn(r*h)` as the textbook GRU does; (2) `b_ih`/`b_hh` CANNOT be pre-summed
  the way core/lstm.h folds them, since `b_hn` sits inside the `r` product.
  (piano_transcription landed with its own BiGRU; fold it in when convenient,
  do NOT refactor it blind.)
- [ ] **`core/stft.h`** — forward STFT. `core/istft.h` covers only the inverse.
  htdemucs and mel-band-roformer each carry a private copy already, so a third
  consumer makes it the third duplicate. ⚠️ **This refactors two SHIPPED
  backends** — requires byte-identical stem output on both before it lands, or
  it ships gated. Lower priority than cqt/gru precisely because of that risk.

### Phase 1 — models

- [ ] **RMVPE** (MIT, lj1995/VoiceConversionWebUI). Robust vocal F0; **same
  360-bin salience as CREPE**, so it reuses `crepe_decode_local_average`
  unchanged and drops into the same `--pitch` surface. CometBeat's vet
  correction is important and correct: the *shipped ONNX is a pure conv U-Net
  with NO recurrent layer* (the paper's GRU is absent), so it ports cleanly.
  Input is a 128-bin mel — nearly free for us via `core/mel.h`. 361 MB f32 →
  ~180 MB q8_0 / ~90 MB q4_k. Wire as a second capacity behind `--pitch`.
- [x] **BTC chord recognition** — SHIPPED 2026-07-20. `--chords` + session C ABI
  + wasm; `crispasr-diff btc` 13/13 at cos 1.000000 (f16 and f32); 98.6-99.2 %
  `mir_eval` agreement with the torch reference on real music. Weights are
  CC-BY-NC-SA and ship behind `--accept-license`; published as
  `cstr/btc-chords-GGUF`. `core/cqt.h` landed and was NOT the last blocker --
  the real bugs were a missing `scale=True` and a chunked-vs-continuous
  front-end mismatch. See `docs/music-transcription/PLAN.md`.
- [ ] **Basic Pitch** — see §250, claimed there.
- [ ] **MT3** — feasibility memo on T5X/JAX checkpoint conversion BEFORE any C++.

### Phase 2 — surfaces

- [x] **Bind `crispasr_session_separate*` in the Dart package** — SHIPPED
  (`flutter/crispasr/lib/src/crispasr.dart:4031`, `separate()` returning
  `List<Stem>`). NOTE: until 4eccc60cb this reached htdemucs ONLY -- the session
  ABI had no mel-band-roformer arm, so the MIT vocals/instrumental model (the
  CLI default) was unreachable from Dart. Now both.
- [ ] **CREPE perf**: per-node profile (never done — the RTF numbers came from
  FLOP arithmetic, not measurement) and sweep `kBatch` (64 was a guess).
- [ ] **CUDA validation** for the CREPE path and both ggml conv fixes
  (`ggml_conv_1d` batch reshape, `ggml_conv_1d_dw` batch support). Everything so
  far is M1/Metal only, and LEARNING 35 says Metal-correct is not CUDA-correct.
- [ ] **File the upstream ggml PR** drafted at
  `tools/upstream-prs/24-conv-1d-batch-reshape.md`.

## §252 — CosyVoice3 native-Vulkan: fp32-accumulation lever (LOW priority, likely dead end)

Background (see LEARNINGS "Backend miscomputes my pipeline ≠ op X is broken" +
`project_304_cosyvoice3_vulkan_native_lm_vs_flow`): CV3 is CPU-routed under Vulkan
because native synthesis is noise. This was FULLY root-caused, on a real Tesla
P100: it is **not** a broken ggml op (test-backend-ops passes every flow/HiFT op
on Vulkan — IM2COL, NORM, MUL_MAT, ROPE, …) and **not** my gallocr dispatch
(`FORCE_GALLOCR` on CPU is bit-identical to the scheduler). It is **aggregate
precision sensitivity**: Vulkan's in-tolerance per-op accumulation deltas (fp16 vs
CPU fp32) compound across the 22-layer DiT × 6 CFM Euler steps and are amplified
by the log-mel HiFT vocoder → flow mel cosine(cpu,vk)=0.961 → garbage. The LM
(shallow-per-token) is bit-correct on Vulkan (512/512 greedy tokens).

- [ ] **Lever (low priority, low odds):** force **fp32 accumulation** in the
  ggml-vulkan matmul/conv path for the flow + HiFT (e.g. `GGML_PREC_F32` /
  coopmat-f32-accum, or a build/env knob) and re-measure the flow mel cosine and
  the audio ASR round-trip on a real NVIDIA Vulkan device (Kaggle P100). If cosine
  climbs to ≳0.999 and audio is intelligible, native Vulkan (or at least a
  flow+HiFT-on-Vulkan hybrid) becomes viable. **Why it's likely a dead end:** every
  op already passes test-backend-ops *within tolerance*, so fp32-accum may not
  shrink the per-op delta enough to stop the CFM+vocoder amplification; and it is
  deep upstream ggml-vulkan shader work. Do NOT invest unless someone independently
  wants native-Vulkan CV3 speed badly enough to accept the shader effort.
- Groundwork already committed (branch `fix/304-cosyvoice3-se`, gated, NOT merged):
  LM single-backend gallocr (proves the LM runs native-Vulkan), hybrid
  HiFT-on-CPU infra, and diagnostics `CRISPASR_COSYVOICE3_{GREEDY,DUMP_MEL,
  DUMP_HIFT,FORCE_GALLOCR,HIFT_ON_GPU}`. Repro kernels:
  `tools/kaggle/cv3-vulkan-{isolate,convtest}`. A real self-recursion crash in the
  `cv3_sched_*` shims (CPU/Metal SIGSEGV) was fixed on that branch; the shipped
  release was never affected (shims are branch-only).
- Default stays the shipped all-CPU route under Vulkan — correct, and the right
  answer unless the lever above ever pays off.
