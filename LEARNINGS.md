# CrispASR — Technical learnings

Distilled from months of porting ASR/TTS architectures into one ggml
codebase. Nothing here is breaking news; everything here is something
we'd have saved days if we'd known up front.

If a lesson is still "live" (affects current work), it's linked from
`PLAN.md`. If it's historical (a bug we already fixed), it's linked from
`HISTORY.md`.

---

## "The model emits it inline as text" is a claim to VERIFY, not to document — structured data you never parsed looks identical to a model limitation (#300 follow-up, 2026-07-27)

vibevoice was written off in #300 as "speaker info is part of the transcript text,
so the structured-label change is a no-op there" — and that sentence was then
copied into three doc sites and the issue's closing comment. It was wrong in a
specific, checkable way: the model answers with
`[{"Start":..,"End":..,"Speaker":N,"Content":".."}]`, which is *more* structured
than the field we said it lacked. The adapter simply assigned the whole blob to
`seg.text`. One `grep` for what the prompt asks the model to produce
(`src/vibevoice.cpp`: "please transcribe it with these keys: Start time, End time,
Speaker ID, Content") would have settled it before the claim was written down.

The tell to generalise: when a backend "can't do X" but its OUTPUT is a blob you
pass through untouched, you have not established that it can't do X — you have
established that nobody looked. Read what the prompt asks the model for; the
answer's shape is usually right there.

Three supporting lessons from the same fix:

**Parse LLM output with a scanner, not a strict reader.** A decode that hits the
token cap ends mid-array. `json.hpp` throws on that and you lose every complete
utterance before the cut — precisely the long-audio case where the labels matter
most. An object-at-a-time scanner loses only the unfinished tail. Unit-test the
truncation case explicitly; it is the one a strict parser silently turns into
"the model produced nothing."

**A pass over already-formatted text corrupts it, and the corruption hides until
you clean up the layer above.** With the blob passing through, FireRedPunc was
capitalising *inside* the JSON — the printed keys were `"STart"`, `"ENd"`,
`"SPeaker"`. So the pre-fix output wasn't even valid for the consumers who were
parsing it, and nobody noticed because the text looked "roughly right". Once the
parse landed, the same pass produced `ANd so … country..` on the clean text, which
is #308's `CAP_PUNCTUATION_NATIVE` audit item — every LLM-decoder ASR backend
emits punctuated, cased text and must declare that flag. Fixing a formatting layer
often reveals a second formatting bug that the first one was masking.

**A pinned expected_transcript that cannot match is worse than no pin.**
vibevoice's regression entry held the bare Content since 2026-06-15 while the CLI
printed the JSON blob — the entry could never have passed, and it sat there
looking like coverage. When pinning a transcript, capture it from the binary under
test and say where it came from; if the pin and the printed output can't be the
same string, the entry is decoration.

## A new REQUIRED request field is a breaking API change on a schedule you don't control — deny the sub-feature, don't refuse the request (#312, 2026-07-27)

v0.8.22 made `"spoken_disclaimer": false` on a voice clone require a companion
`marking_attestation` field, hard-refusing with `400` otherwise. Correct policy,
wrong failure mode: the integrator (Subtitle Edit) shipped the field a day after
its last release, so for a full release cycle *every* SE build in users' hands got
zero TTS output — the feature was gone, not degraded, and the user-visible symptom
("synthesis failed 400") pointed nowhere near provenance policy.

The generalisable shape: when a request asks to **weaken** a default and doesn't
qualify, you have two failure modes — refuse the whole request, or refuse the
*opt-out* and serve the default. The second is strictly better here, because the
policy being enforced is "never emit weaker-than-default output," and serving the
default satisfies it completely. Refusing outright buys no extra safety; it only
converts a client that is one field behind into a client that is entirely broken.
Announce the denial so it isn't silent (response headers + an audit-log line with
`DENIED` in it), and keep the hard refusal on surfaces where the operator can fix
it in place — the CLI's `--no-spoken-disclaimer` still exits 12, because the error
names the flag you'd add to the very command you just typed.

Two supporting rules the same issue produced. **An operator-scope attestation must
satisfy a request-scope gate.** A server launched with
`--accept-marking-responsibility` has accepted the duty for *every* response it
serves, which strictly subsumes the per-request field — checking only the body
refused requests the operator had already covered. **A gate added without docs is
a trap with a fuse.** `marking_attestation` never made it into `docs/server.md` or
`docs/tts.md`, so both surfaces still told integrators to send
`"spoken_disclaimer": false` on its own — the exact request the server had started
rejecting. And the live test that asserted that request is accepted was left
failing for four days without anyone noticing, because live tests need models and
CI never runs them: a policy gate wants a test on a tier that actually runs.

## When a clone/zero-shot path misbehaves but the BAKED/preset path is clean, diff the two paths' DATA byte-for-byte — the difference IS the bug (CosyVoice3 #310, 2026-07-26)

CosyVoice3 zero-shot (`--voice ref.wav`) spoke the reference transcript before the
requested text; the baked preset voices were clean. Both paths run the identical
`cv3_synth_with_voice`, so the bug had to be in the DATA the zero-shot extractor
produced vs what the baker stored. Dumping the baked voice's `prompt_text` out of
the voices GGUF settled it in one shot: it is `"You are a helpful
assistant.<|endofprompt|>" + <transcript>`, i.e. the required CosyVoice3 system
prompt + `<|endofprompt|>` boundary — while the runtime clone stored the bare
`--ref-text`. Without that prefix the Qwen-based speech LM is out-of-distribution
and re-renders the reference text as speech (the "leak"); it is intelligible in the
right voice, so it reads like extra content, not miscompute. Lessons: (1) a working
sibling path is a free oracle — when preset works and clone doesn't, the delta is a
short list of stored fields, so compare them before touching the graph; (2) an
instruct/chat speech-LM needs its EXACT training-time system-prompt wrapper — a bare
prompt string silently degrades to "read this text aloud." (Second, smaller cause:
the LLM was fed the mel-cap-truncated prompt speech tokens while the text kept the
full transcript — feed the LM the full tokens, let only the flow use the truncated,
mel-aligned set.)

F5-TTS produced no Chinese audio because its `convert_to_pinyin` was an ASCII-only
passthrough (every Han char → unknown token → silence). We added a real g2p
(jieba-min + pypinyin TONE3 + sandhi) and unit-tested it to **99.8% token parity**
against the reference `convert_char_to_pinyin`. Chinese was still garbled. The g2p
was correct; the bug was one function DOWNSTREAM: `f5_tts_synthesize` concatenated
the g2p's token list into a flat string and **re-tokenized it per UTF-8 byte**. The
F5 vocab stores whole pinyin syllables as single entries (`zhong1`, `guo2`, `a1`,
`ang1`, …), so `zhong1` became `z,h,o,n,g,1` → six wrong ids → recognizably-Chinese
-but-unintelligible audio. The parity test validated the g2p's OUTPUT and never saw
f5's CONSUMPTION of it, so it stayed green through a shipped, broken build. Only the
TTS→ASR roundtrip caught it (ZH ASR became an exact match; token count for the clip
dropped 146→63 = one id per syllable). Lessons: (1) a list of vocab units must be
mapped element-wise to ids (`list_str_to_idx`), NEVER concatenated and re-split —
the re-split silently destroys every multi-char token; (2) HARD RULE #3b in force —
a component's own metric is blind to the seam with the next stage, so the acceptance
gate is always the DECODED OUTPUT, not the component score. Corollary: for cross-
lingual TTS, use a same-language reference (an English ref speaking Chinese is
degraded regardless), and a wrong/too-short `--ref-text` garbles output outright.

## Reusing a cached scheduler graph across `sched_reset`/`alloc` cycles is CPU-safe but SIGSEGVs on GPU — the reused input tensor is bound to the prior cycle's freed buffer (moonshine #301, 2026-07-25)

Sibling of the #215/#235 "cached graph shares a scheduler with a larger graph"
trap, but distinct: here a SINGLE graph is cached and re-run. moonshine's §176s
encoder cache reused the same `ggml_cgraph` across `ggml_backend_sched_reset` +
`ggml_backend_sched_alloc_graph` every call. On CPU this is fine; on a GPU sched
the reused `audio_input`/`enc_pos` leaves stay bound to the PREVIOUS cycle's freed
buffer, so the very next `ggml_backend_tensor_set` memmoves into freed memory and
the **2nd** transcribe SIGSEGVs (1st is fine). It reproduced at
`CRISPASR_SERVER_WORKERS=1` too, so it's repeated-call, not concurrency; a `--server`
made it look like a server bug. Bisect tell: the rebuild branch (different
`n_samples`) never crashes, only the exact-reuse branch does. Fix: gate the cache to
CPU (`ggml_backend_is_cpu(ctx->backend)`); on GPU rebuild the (tiny) graph each call
and free it. Rule: cache-and-reuse a sched cgraph across reset/alloc is a CPU-only
optimization — GPU wants rebuild-each-call (or a genuinely persistent gallocr graph
whose inputs you re-set every eval).

## A hardcoded decode cap silently ignores --max-new-tokens — and forwarding it naively SHRINKS a backend (#292, 10 ASR backends, 2026-07-22)

An autoregressive/LLM ASR backend caps its decode loop at some `const int max_new
= N`. If N is hardcoded, `--max-new-tokens` does nothing and a long SINGLE-PASS
run (`--chunk-seconds 0`) truncates — the reporter's 300 s file stopped at 164 s
against moss-diarize's baked-in 1024. The audit found the identical pattern in
**10 backends** (moss-diarize, canary, canary-qwen, glm-asr, funasr, mimo-asr,
moss-transcribe, mini-omni2, higgs-stt, higgs); it hides because with default
chunking each ~30 s chunk fits the cap, so it only bites single-pass or unusually
dense audio.

Three transferable points:

1. **The fix is a context field + setter + adapter forward + session forward, per
   backend** — mirror an existing `set_beam_size`. The field defaults to the
   backend's OWN old constant, so nothing regresses; the decode loop AND any
   coupled KV-cache sizing must both read it (funasr/higgs/moss-transcribe sized
   the KV as `prompt + 1024` separately — miss it and a raised cap overflows
   `kv_max_ctx` → GGML_ASSERT).

2. **Naive forwarding is a SECOND bug.** `whisper_params.max_new_tokens` defaults
   to 512, but several backends default HIGHER (diarize 1024). Forwarding 512
   unconditionally SHRINKS them. The gate is a `max_new_tokens_explicit` flag
   (set only when the user passed `-n`), so the CLI default never overrides a
   backend's own — mirrors `chunk_seconds_explicit`. The session path needs no
   flag: its default is 0 = unset.

3. **Not every cap is a bug.** moonshine's `ceil(n_samples/16000*6.5)` clamped to
   194 is LENGTH-DERIVED and 194 is its architectural short-form limit — forcing
   more exceeds its trained window. Grep each backend's source before "fixing" a
   constant; a length-derived or architectural bound is correct.

CUDA-validated the flag is honored by measuring word count at
`--max-new-tokens 64` vs `4096` (both single-pass): a deterministic causal
increase (216→366) — equal counts would be the bug, independent of clip length.
A first gate demanding `>2x` wrongly FAILed the real 1.7x; validate the ORACLE
(the threshold) before believing a red result — see [[parity-harness-negative-control]].

---

## A green release job is not a shipped artifact — verify what you DELIVER, not what the workflow says (v0.8.18–v0.8.20 release train, 2026-07-21)

Four releases in one session, and every real failure REPORTED SUCCESS. The
pattern is the lesson: a workflow conclusion describes the workflow, not the
artifact. Each was caught by a gate that tested the delivered thing directly.

- **Presence is not resolvability.** The lib-bundle gate checked that every
  `@rpath`/`DT_NEEDED` dependency was PRESENT in the tarball. It was — and 6 of 7
  bundles still could not be `dlopen`ed. macOS baked the CI runner's build path
  into `LC_RPATH` (`/Users/runner/work/...`); all 5 Linux bundles used
  `$ORIGIN/../../ggml/src`, one level too high. The file was in the box; the map
  didn't lead to it. Fix: `tools/verify-lib-bundle.sh` RELOCATES the bundle to an
  unrelated dir and dlopens it — loading in place succeeds by accident on the
  machine that built it, which is exactly how it shipped.
- **A green job can attach nothing.** The `build-xcframework` job went green
  having uploaded ZERO assets: the file is `crispasr-<tag>-xcframework.zip`
  (dash) but the attach glob was `*.xcframework.zip` (dot), and
  `fail_on_unmatched_files: false` let it pass empty. For an asset that is the
  whole point of a job, set `fail_on_unmatched_files: true` — a glob matching
  nothing must FAIL, not ship the feature nowhere.
- **The irreversible step deserves an independent gate.** pub.dev publishing is
  irreversible, so the check that matters runs BEFORE it: download the shipped
  bundles, dlopen them, confirm the xcframework's slices exist. The pub.dev
  publish itself failed at its DRY RUN (stale `flutter/crispasr/CHANGELOG.md`
  missing the version — `bump-version.sh` doesn't touch it), which is the safe
  failure: nothing published. Cross-platform caveat: on macOS you can only
  closure-check a Linux ELF bundle, not dlopen it — the native dlopen ran in the
  release job on Linux; don't misread "slice is not valid mach-o" as a defect.

Transferable: after any release, verify the downloaded artifact does the thing a
consumer does (loads / links / exists). "Release: success" is the beginning of
verification, not the end. Same family as [[cmake-link-not-shipped]] and the
cancelled-CI-reads-as-green trap. See `PLAN.md` 2026-07-21 completions and memory
[[release-packaging-gotchas]].

---

## A capability flag is a PROMISE the backend must keep — an unimplemented cap disables the safety nets keyed off it (canary-qwen #290, 2026-07-21)

`canary-qwen` declared `CAP_INTERNAL_CHUNKING` but `src/canary_qwen.cpp` has NO
chunking code — 0 hits for `chunk_seconds`, vs 62 in `src/parakeet.cpp` and a
real chunker in `src/canary.cpp`. The flag asserts "I slice long audio myself,"
so the dispatcher trusted it and disabled BOTH of its own safeguards: the #257
gate handed over the whole clip when `--chunk-seconds` was given, and
`should_auto_chunk_long()` bailed at the `CAP_INTERNAL_CHUNKING` branch when it
wasn't. canary-qwen also doesn't `prefers_vad()`, so the auto-VAD net was out
too. Result: one full-length FastConformer pass — O(T²) attention, RSS growing
384 MiB → 6.3 → 10.2 GiB with clip length, and text degraded past the encoder's
trained window. Reproduced either way BECAUSE one false flag killed every net.

Two transferable points. (1) A capability is not documentation — code dispatches
on it, so declaring one you don't implement is worse than declaring nothing.
Grep the backend's own source for the feature before trusting its cap
(`chunk_seconds` count was decisive here). (2) The fix that WROTE the gate
(`1a2b3dcea`, for parakeet) was correct; it keyed on the capability, so the bug
was always the false declaration, not the gate. When a shared gate misbehaves
for one backend, suspect that backend's inputs to the gate, not the gate.

Verified by the capability BITMASK with parakeet/canary/fastconformer-ctc as
controls (they keep bit 20; canary-qwen no longer does) — a first-run unit-test
pass proves nothing without a control. Memory [[check-blueprint-before-backend-feature]].

---

## Moving a release tag is CLEAN when the fix touches only release TOOLING — the test is provenance, not "never move a tag" (v0.8.20, 2026-07-21)

At v0.8.18 I refused to move a tag: assets were built from source the moved tag
wouldn't match. At v0.8.20 I moved it THREE times, correctly — because each fix
touched only `release.yml` / `verify-lib-bundle.sh` / `build-xcframework.sh`,
never binary-producing source. Every already-built asset was byte-identical to
what the new commit produces, so there was no provenance mismatch and no reason
to burn a new version number. The user pushed back on my reflexive "cut v0.8.21"
and was right.

The rule is provenance, stated as a check:
`git diff --name-only <oldtag>..<newcommit> | grep -E '\.(cpp|h|c|cu)$|CMakeLists'`
returning nothing ⇒ safe to force-move the `v*` tag and re-run the release (all
assets rebuild identically; the fixed jobs now succeed/attach). If it returns
source files, DON'T move — the shipped binaries would then disagree with the
tagged source. A corollary for pub.dev: the CHANGELOG fix moved ONLY the
`crispasr-v<x>` tag, not the `v<x>` GH tag — a Dart-package doc change shouldn't
trigger a 40-min binary rebuild, and the two tags legitimately differing by one
doc commit is fine. Memory [[release-packaging-gotchas]].

---

## A positional arg landing on the WRONG parameter is invisible in review and baked into the weights (piano-transcription BN eps, 2026-07-20)

The first per-stage validation of a backend we had already shipped found it
wrong. `crispasr-diff piano` reported mel PASS at cos 1.000000 and then
`conv_block_output` FAIL at **cos 0.810** — first divergence = the bug, exactly
as the harness is supposed to work.

Upstream builds every 2-D BN as `nn.BatchNorm2d(out_channels, momentum)`
(models.py:74). `BatchNorm2d`'s second POSITIONAL parameter is **`eps`**, not
`momentum`. So the intended momentum 0.01 silently became **eps = 0.01** and
momentum kept its 0.1 default. One line later, `nn.BatchNorm1d(768,
momentum=momentum)` passes it as a KEYWORD, so that layer keeps eps = 1e-5.
The loaded checkpoint confirms it: **33 BatchNorm2d at eps=0.01, 4 BatchNorm1d
at eps=1e-5.**

Our runtime hardcoded the PyTorch default 1e-5 everywhere. It mattered enormously
because the running variances are tiny (~0.003), so eps DOMINATES the
denominator: `sqrt(0.00295 + 1e-5) = 0.0544` vs `sqrt(0.00295 + 0.01) = 0.1138`
— a 2.09x error per layer, compounding to 2.44x by conv block 1 and dragging
the onset head to cos 0.72.

Three transferable points:

1. **An upstream slip that the weights were TRAINED with is not a bug to fix —
   it is a spec to reproduce.** Same as BTC's trailing ReLU. Read what the code
   DOES, not what it evidently meant.
2. **Never assume a framework default for a hyperparameter you can read off the
   loaded module.** `[m.eps for m in model.modules() if isinstance(m, _BatchNorm)]`
   is one line and would have caught this at port time. Defaults are the least
   reliable thing in a port.
3. **"It produces plausible output" is not validation.** A downstream consumer
   had reported this very backend "recognised the Für Elise motif" on a real
   build — while every 2-D BN was wrong. Plausible-but-wrong is the failure mode
   per-stage diffing exists for.

Bisecting it is worth copying: mel PASSed, so the input was right; a numpy spec
built from the GGUF weights matched our C++ at cos 1.000000 (so the C++ was a
faithful implementation of the WRONG algorithm); then stepping conv1 -> bn1
inside block 1 showed conv1 exact and bn1 diverged — three comparisons to go
from "somewhere in an 8-layer stack" to one constant.

---

## "Linked in CMake" is NOT evidence the code SHIPS — the linker drops an object nothing references (mel-band-roformer, 2026-07-20)

`crispasr-lib` linked the `mel-band-roformer` target, exactly as the CMakeLists
reads. But nothing in `src/crispasr_c_api.cpp` ever referenced its symbols, so
the linker dropped the whole object from the shared library. Verified against
the **released v0.8.17 artifact**: `mel_band_roformer_separate` is simply not
in `libcrispasr.dylib`.

So the backend was not merely "unreachable from the session API" — the code was
**not present in the shipped library at all**, for every consumer of it. The CLI
worked the whole time because `crispasr-cli` links the static lib directly, and
`--separate` reads the GGUF arch in its own dispatcher. A working CLI proved
nothing about the artifact every binding loads.

Three things follow:

1. **Verify a release artifact by SYMBOL, not by version number or changelog.**
   `nm -gU libcrispasr.dylib | c++filt | grep <fn>` answers "is this actually in
   the build" in one line. We asked the question this way and learned that
   v0.8.17 ships chords but not MBR separation — which no amount of reading the
   tag or the notes would have revealed.
2. **Demangle before concluding.** Some runtimes are C++-linkage, not
   `extern "C"`: `sidon_init_from_file` appears only as
   `__Z20sidon_init_from_file...`. A raw `nm | grep _name` reports a shipped
   backend as missing. My first pass did exactly that and produced a false
   alarm on sidon.
3. **Source-text audits are structurally blind to this.** Every other check in
   `tools/check-backend-wiring.py` greps source, and the source looked correct.
   The new shipped-library check compares declared backends against symbols in
   the built dylib — ground truth, so it has NO alias false positives (the
   name-matching alternative produced 21/76 noise; this one produces 0/63).
   Proven both directions: PASS clean, FAIL when the c_api arm is removed.

The residual blind spot is unchanged and recorded in PLAN.md: a backend in
NEITHER the CLI roster NOR the c_api list is invisible to all three checks.

---

## Cosine, correlation and peak-match are ALL scale-invariant — a uniform gain error passes every one of them (CQT `scale=True`, 2026-07-20)

`core/cqt.h` normalised each constant-Q kernel by its L1 norm but never applied
librosa's `scale=True` factor (`V /= np.sqrt(lengths)`, the default). Every bin
came out low by `sqrt(N_k)` — **152x at the bottom octave**. BTC read its
features as near-silence and emitted "no chord" for every frame of every file.

The bug is trivial. What it cost was the second half-day, and that came from
the guards being structurally unable to see it:

- **`tools/cqt_librosa_parity.py`** checked per-frame shape correlation and
  peak-bin match. Both are scale-invariant *by construction*. It reported
  correlation 0.9999 and 97.6% peak match against a build that was wrong by
  two orders of magnitude.
- **`tests/test-core-cqt.cpp`** passed **726 assertions** against the same
  build. Six test cases, none of which pinned an absolute magnitude.

This is the THIRD time this exact class has bitten us. The htdemucs iSTFT scale
was inverted (`1/sqrt(nfft)` for `sqrt(nfft)`) and `spec_input` passed at
cos 1.000000 the whole time; it was caught only by printing `|mine|` vs `|ref|`
alongside the cosine. Different backend, different year, same blind spot.

**The rule: any metric that normalises its inputs cannot validate the scale of
those inputs.** Cosine similarity, Pearson correlation, argmax agreement, SNR
against a normalised reference, "does the peak land in the right bin" — every
one divides the magnitude out. If a stage can be wrong by a constant factor,
something must compare an ABSOLUTE quantity: `|mine|` vs `|ref|` next to the
cosine (our diff harness prints exactly this — use it), or a per-bin magnitude
ratio, or an analytic value.

Two sharper corollaries, both learned the hard way in this same fix:

**1. Assert the invariant, not a bound on a signal.** The strongest test here
turned out to need no signal at all: after normalisation each kernel's L1 norm
must equal `sqrt(N_k)` exactly, because the code multiplies by `sqrt(N)/l1`.
That is arithmetic on the coefficients — no window leakage, no tolerance to
argue about, and it fails by a factor of up to 152 the moment the scale factor
is dropped. Reach for the algebraic identity before reaching for a threshold.

**2. A tolerance wider than the defect is not a test — and a test NAME can
encode the wrong law.** One existing case, "L1 normalisation makes bins
comparable across kernel lengths", compared peak magnitude two octaves apart
and required `ratio < 3`. It measured exactly the right quantity. It passed
anyway, because the correct and broken builds sit on *opposite sides of it*:

|                        | response         | two-octave ratio |
|------------------------|------------------|------------------|
| L1 only (the bug)      | flat across bins | 1.0              |
| L1 x sqrt(N) (librosa) | ~ `sqrt(N_k)`    | 2.0              |

Both under 3. Worse, the property the test was NAMED for — equal-amplitude
tones giving equal magnitude — is the one the **buggy** build satisfies;
`scale=True` deliberately does not have it. I nearly re-broke the code
"fixing" that test to `< 1.5` before measuring. The replacement asserts the
exact law (`peak == (A/2) * sqrt(N_k)`, which holds to four decimals across
the whole range) instead of a bound.

**Write the guard before the fix.** The magnitude check went into the parity
tool first, so it failed on the broken build, and reverting the fix afterwards
still drives its asserted median from 1.0043 to 0.0131 — a 76x margin. A guard
authored after the fix has never been observed to fail, which means it has
never been observed to work.

See `docs/music-transcription/PLAN.md` and the header comments in
`src/core/cqt.h` + `tests/test-core-cqt.cpp`.

---

## A logits-level diff harness at cos 1.000000 says nothing about the TABLE that turns logits into labels (BTC chord vocabulary, 2026-07-20)

`crispasr-diff btc` passes 13/13 stages at cos 1.000000. That validates the
transformer completely and the user-visible output not at all: the last step is
`argmax -> index -> name`, and the name tables live outside the graph. A wrong
entry in the 170-class quality table, or one flipped bool in the maj/min
collapse, produces byte-identical logits and mislabels a chord forever. It
would read as "the model is weak on diminished chords", not as a typo.

Same shape as the OmniVoice zeroed-`token_embd` lesson (a data defect wearing a
model defect's clothes), but from the other end of the pipeline.

**And the blind spot is not only at the END of the harness — it is at the
START.** The BTC reference dump deliberately includes `input_feat` so the
runtime replays the exact features the spec scored, which stops a front-end
difference masquerading as a model parity failure. That is the right design,
and it means the harness NEVER TESTED OUR FRONT END. Two real bugs lived there
through 13/13 at cos 1.000000:

- The reference CQTs each 10 s chunk INDEPENDENTLY and concatenates
  (`audio_file_to_features`). Because librosa centres every call, each chunk
  carries its own edge padding, so this is NOT a continuous transform: 2778
  frames vs 2770 on a 257 s clip. We had implemented `librosa.cqt` correctly
  and `audio_file_to_features` not at all — our features scored **cos 0.9993
  against a continuous librosa CQT and 0.8815 against the actual reference
  pipeline**.
- Frame duration is `inst_len / timestep` (10/108 = 0.0925926 s), NOT
  `hop / sample_rate` (2048/22050 = 0.0928798 s). A 0.31 % difference — 0.79 s
  of accumulated drift over a four-minute song, every chord boundary
  progressively late.

Neither is visible in a per-stage logit diff. Both were found only by running
the reference pipeline end-to-end on real audio and comparing chord timelines
with `mir_eval`: **86.63 % → 98.56 % (tetrads)** once fixed.

So the rule has two halves. Port the reference's PIPELINE, not just its
mathematical operation — the wrapper around `librosa.cqt` was as load-bearing
as the transform. And when a harness pins an input to isolate a stage, write
down what that pinning excludes, because that is now untested surface. Ours is
covered by `tests/test-btc-vocab.cpp` (geometry, hermetic) plus the real-music
`mir_eval` comparison.

So: **for every port, list what the diff harness structurally cannot see, and
cover that separately.** Here that meant extracting the pure vocabulary +
positional-encoding helpers into `src/btc_chord_vocab.h` purely so they could be
unit-tested without weights (`tests/test-btc-vocab.cpp`), and asserting
properties a table typo breaks: all 170 names distinct, the root surviving the
maj/min collapse, every one of the 25 collapsed labels reachable (an
unreachable one = a chord that can never be emitted), and the positional
encoding's halves being concatenated rather than interleaved.

Cheap, hermetic, no model, no GPU — and it covers the only part of the chain
the expensive harness is blind to.

---

## PLAN "OPEN" items are frequently already shipped — audit against the CODE, never the prose (2026-07-17)

A single session found **~11 PLAN items marked OPEN / NOT STARTED / SURVEY-ONLY
that were fully implemented in the tree**: §169 (qwen3-asr ChatML prompt, landed
under the #218 work), #128 (Piper TTS — whole backend + converter + registry +
tests present), #60o (MTLBinaryArchive pipeline cache — the
`crispasr_metal_pipeline_cache_open/_flush` lines print on every run), §155
(CONV_TRANSPOSE_1D — all phases + Metal/Vulkan/CUDA `col2im_1d` kernels), #58
(MOSS-Audio-4B), #101 (OmniVoice), §229 (`GGML_LLAMAFILE ON`), python `_find_lib`,
§66 (pub.dev `crispasr` published), plus §57/§106/§224/§247 sub-items. The PLAN is
compacted periodically but individual stale entries survive, so trusting one
wastes hours re-deriving shipped code. **Before implementing any roadmap item,
`grep`/read the code it names and check `git log -- <file>`; if it exists, verify
empirically and just correct the PLAN.** Corollary for *external*-state claims
(registry presence, HF uploads): a one-line API `curl` settles them — the PLAN
said pub.dev was "absent" when the package was live (crates.io 404-checks need a
`User-Agent` header or they 403 and masquerade as a real 404). A subagent is an
efficient way to run this audit across many headers at once.

## When you can't run the acceptance roundtrip, ship the path gated default-OFF + a provable-equivalence argument — don't block on the model (#201 TADA cloning, 2026-07-17)

HARD RULE #3 says the decoded-output roundtrip is the only acceptance test, but a
memory-pressured box couldn't load tada-1b + the 1.3 GB aligner to run it. Rather
than block, the mergeable move was: (1) build the new path so its numerics are
**equivalent by construction** to a shipped, already-validated path — here
`tada_make_ref_from_pcm` = the validated `tada_encoder_encode` + a `load_prompt`
body extracted into `tada_set_prompt_values`, and `write_ref_gguf` demonstrably
writes the exact tensors `load_prompt` reads, so the in-memory path is the
file path minus the I/O (no new graph math); and (2) gate the new behaviour behind
`CRISPASR_TADA_WAV_CLONE=1`, default OFF, so the default path is byte-identical and
no regression is possible. That combination is safe to merge to `main` *before*
the roundtrip — the roundtrip then only gates flipping the default ON, not the
merge. Matches the `keep-gated-not-revert` discipline: a plausible,
review-verified path ships opt-in with the acceptance gate documented, not
withheld. (Also: the feature needed the SAME logic in TWO surfaces — session
C-ABI and server *adapter* — the recurring multi-surface trap; a shared
`clone_from_wav`/`tada_make_ref_from_pcm` keeps them from drifting.)

---

## Two writers on one output field at different pipeline stages = an ordering bug — fix the ORDER, don't bolt on provenance (#266 speaker labels, 2026-07-17)

`--speaker-db` wrote matched names into `seg.speaker` per slice (pre-merge);
global clustering rewrote the same field per segment post-merge. Result: mixed
slices got one identity, then clustering destroyed the names anyway. The
tempting fix is a structured label (`{cluster_id, name?, source, score}`) with
precedence rules — a real refactor across every output writer. The actual fix
was smaller and stronger: **make the two writers one ordered chain** (cluster
first, identify per CLUSTER after, nothing label-touching downstream). Once
identification consumes clustering's output instead of racing it, "later
stages must not overwrite matched names" holds by construction and the string
field is fine. Generalization: when two features fight over one field, first
ask whether they're actually *stages* of one pipeline in the wrong order;
provenance/priority metadata is the fallback, not the default. Bonus from the
same change: the identification stage reuses the embeddings clustering already
computed (centroid per cluster) — correct architecture was also the cheaper
one (no second embedding pass).

## Path-filtered CI workflows ROT — the first re-triggering change inherits ALL the debt accumulated since the last run (Go bindings, 2026-07-17)

`Bindings Tests (Go)` only runs when `bindings/go/**` changes. Nobody had
touched it since 2026-07-14, so three independent breaks sat invisible until
the #266 ABI sweep re-triggered it, and they surfaced ONE AT A TIME (each fix
revealed the next): (1) a cgo-preamble decl used `crispasr_session*` where the
preamble's opaque typedef is `CrispasrSession` — gcc rejects the whole
preamble; (2) the same old commit declared `func (s *Session)` for a type
named `CrispasrSession` — plain Go compile error; (3) `whisper.go` cgo LDFLAGS
had drifted when the c2pa-audio lib was added (the drift-check job had never
run either). Lessons: (a) when your change re-awakens a dormant path-filtered
workflow, budget for paying down its debt — the red X on YOUR commit is
usually not (only) your bug; check `git log` since the workflow's last
successful run before debugging your own diff; (b) a cgo preamble is compiled
C — `gcc -fsyntax-only` on the extracted comment block reproduces CI's
preamble errors locally in seconds, no cgo build needed; (c) for LDFLAGS
regeneration on macOS, the documented recipe (`-DGGML_METAL=OFF
-DGGML_BLAS=OFF -DGGML_CUDA=OFF` + `--dot`) really does keep Metal/BLAS out
of the linux line — verify the diff adds only the intended `-l<newlib>`.

## Scoped `ctest -R` on a PARTIALLY BUILT tree has two name-collision traps: configure-time add_test and catch_discover NOT_BUILT placeholders (first Windows unit-test leg, 2026-07-17)

To run just the speaker-db tests on the (never-unit-tested) Windows CI job we
built 3 of ~200 test targets and filtered `ctest -R "speaker|..."`. Two
non-obvious classes of test still match such a filter without being built:
(1) **plain `add_test(NAME ... COMMAND script)` entries register at CONFIGURE
time** regardless of what was built — `test-speaker-id-live` (a bash script)
matched "speaker" and would have been exec'd on Windows; (2)
**`catch_discover_tests` writes a failing `<target>_NOT_BUILT-<hash>`
placeholder** for targets that were configured but never built — the unbuilt
`test-crispasr-speaker-resample` placeholder matched "speaker" and failed the
first real run 38/39. The robust pattern for a scoped leg:
`ctest -R "<positive>" -E "live|NOT_BUILT"` + register POSIX-script tests
`if(NOT WIN32)`. Also: enumerate what a regex matches BEFORE shipping it —
resolving every TEST_CASE name + add_test registration against the filter
(a 15-line python pass over tests/CMakeLists.txt) caught trap (1) pre-push;
trap (2) only a real run could catch. Positive result worth keeping: the
speaker-db suite (incl. the `.spkr` v2 `CreateDirectoryA` path) passes on
windows-latest — the `_WIN32` code paths were correct all along, just never
executed.

## A binding wrapper that has never been EXECUTED is untested code — and parallel agents produce cross-cutting conflicts only the verifier can see (#266 hardening, 2026-07-17)

The hand-updated Python `SpeakerDB` wrapper passed symbol-parity checks and
eyeball review, yet its first-ever runtime execution found two real bugs in
minutes: the class was never re-exported from `crispasr/__init__.py`
(ImportError on the documented usage), and `__init__` raised (consent gate)
before setting `self._db`, so every refused instance crashed `__del__` at GC.
Symbol lists and signatures verify LINKAGE, not behavior — when touching an
ABI, write one runtime smoke per wrapper you changed (ctypes + a shared-lib
build is enough; `build-shared` with warm ccache is mostly link time). Second
lesson from the same session: two agents in isolated worktrees each did
correct work that CONFLICTED — agent A added a POSIX gate-test script whose
ctest name matched agent B's new Windows `ctest -R "speaker|..."` filter.
Neither could see the other's diff. The verifier's job is not just re-running
each agent's tests but explicitly checking the INTERACTION of concurrently
produced diffs (grep each diff's new identifiers against the other's).

## Deterministic DROPOUT of whole words/tokens (not garble) = a zeroed weight BLOCK in the shipped GGUF — scan converted tensors for zero-norm rows BEFORE debugging the port (OmniVoice #254 `token_embd`, 2026-07-15)

OmniVoice TTS silently dropped specific words ("quick", "started", "One", "See";
"TTS"→"T"). It *looked* like a subtle forward/alignment bug, and I spent real
effort ruling out the estimator, chunking, CFG, sampling, seed, and quant, and
line-by-line verifying the decode against the omnivoice.cpp oracle — all faithful.
The actual cause: **`llm.token_embd.weight` in the shipped GGUF had 4094 ZEROED
rows** (a post-conversion write/upload corruption). Any word whose BPE token landed
in that id block embedded to a zero vector, so the model literally could not voice
it. This is now the **third** corruption of this exact class (Cohere Arabic zeroed
encoder norms; the OmniVoice tokenizer `block.4` zeroing; now the LM embedding) —
it is a recurring failure mode of our convert→upload→download pipeline, not a
freak event.

The signature that should have pointed here immediately:
- **Dropout, not distortion.** Whole tokens *vanish* rather than degrade. A numeric
  forward bug (bad RoPE/scale/norm) smears everything; a zeroed weight ROW removes
  exactly the content that indexes it. If some words are perfect and others are
  simply *absent*, suspect a data hole, not math.
- **Deterministic AND quant-independent.** Same words drop across f16 and q4_k and
  every seed/temperature/CFG. Quant noise flips near-ties; it does not delete words.
  When f16 (higher precision than a *clean* Q8_0 reference) is the one that fails,
  the weights themselves are wrong, not the precision.

Two cheap tools settle it fast:
1. **Step-0 input-embedding diff vs a reference `--dump`.** Input embeddings depend
   only on the embedding tables (tokenization + lookup), *before* any transformer
   weight, so they're comparable across quant/impl. Global cos was 0.995 but three
   positions were exactly **cos 0.0** — and the oracle's dumped `prompt-cond-ids`
   named those tokens ("None", "quick"). Dumping our step-0 embeds in the oracle's
   `[ndims][shape][data]` format (`OMNIVOICE_DUMP_DIR`) made this a 5-minute check.
   Generalize: to split "embedding bug vs forward bug", diff the input embeddings
   first — it's the one tensor that's quant-robust.
2. **Zero-norm row scan** of the suspect table (`gguf.GGUFReader` → per-row L2 →
   `where(norm==0)`). Instantly gives count + id range; compare against the source
   safetensors (SHA-verified) to prove source-clean ⇒ pipeline corruption ⇒ a
   reconvert fixes it (no code change).

Also two process notes from this hunt: (a) use the black-box oracle in **both**
greedy and stochastic decode to prove "our bug, not the model" — greedy alone can
mislead; (b) I floated a plausible "sentence-splitting degrades the bidirectional
aligner" hypothesis and had to **revert** it when single-shot dropped words too —
a reminder to disprove the mechanism (single-shot still failed) before shipping the
"fix". Cf. the Cohere Arabic zeroed-norms entry in `HISTORY.md` (same class,
same "it's a corrupt download, not a bug" resolution).
## A TTS AR model that "never stops" is often a WRONG-SAMPLING-PARAMS bug, not a stop-head/port bug — read the model card's generation params (MOSS-TTS-Local 4B #249, 2026-07-14)

The 4B `moss-tts-local` port synthesized **correct, intelligible audio** but
generation **ran away** — the binary stop head never fired, hitting the
4096-frame cap. Chasing it as a stop-head/port bug was wrong. Static inspection
proved the port matched the HF reference on every structural point
(`_global_hidden_to_local` = identity, local-transformer path/positions/`ln_f`,
the stop head = the checkpoint's `local_text_lm_head`, row feedback, KV cache);
crucially, **codebook-0 reads the same `local_hidden[0]` as the stop head and
produced correct audio**, so `local_hidden` was correct and the ~10-logit
continue-vs-stop gap was real (the reference produces it too). The bug was the
**generation config**: the model card's "Generation Parameters" specify
`audio_temperature=1.7, top_p=0.8, top_k=25`, `do_sample`, stop head SAMPLED
(`text_temperature=1.0`). My generic defaults (`1.0/0.95/50` + GREEDY stop) gave a
too-conservative/degenerate acoustic trajectory that **never reached the natural
end state**, so the stop logit never rose. Fixing the defaults to the card values
→ F16 stops naturally (15/124 frames) and the long clip round-trips at ASR
word-overlap **0.969**. Lessons: (a) the tell is "audio content correct + STOP
timing broken" — suspect sampling before the stop head; (b) **read the card's
recommended generation params** before porting defaults from another backend;
(c) **greedy audio is degenerate** for these RVQ AR LMs (it "stopped" at 35 frames
= 2.8 s for a 30-word passage) — sampled audio at the card temperature is required;
(d) confirm the stop head input by checking that codebook-0 (same hidden) decodes
correct audio.

Two more from the same port: **Q4_K's LONG trajectory runs away while short + F16
stop** — intrinsic quantized-AR drift accumulates over the trajectory (reinforces
[[tts-port-parity-via-logit-rank]]); ship/validate **F16**, keep quant
best-effort. And the **decode-side codec attention must be query-chunked, not
dense**: a sliding-window causal transformer whose T reaches `t_audio·32` (~131k at
the final 48 kHz stage) OOMs a dense `T×T` scores tensor (~916 GB); chunk queries
against their windowed keys (the reference's `query_chunk_size` path) — bounded
memory, byte-identical output. See also [[kaggle-hf-download-curl-not-hftransfer]]
for the Kaggle infra lessons this port cost (hf_transfer no-resume → `curl -C -`;
progress via `kh.step()`+`HF_TOKEN` mirror, never narrate from status).

## When the full system needs an unavailable resource (model / GPU), factor the risky logic into a pure helper and prove IT on synthetic data (§176l + GGUF-bounds, 2026-07-12)

Two follow-ups had no local model / were untrusted-input paths, yet both shipped
*validated* by construction — no "gated but never run" code:
- **§176l kyutai RVQ routing.** Can't run kyutai (no Kyutai GGUF). But the risky
  part is a pure transform: extract codebooks → `‖E[k]‖²` → argmin → transpose to
  `out_codes[q][t]`. Factored it into `core_rvq::encode_euclidean_per_stage` and
  unit-tested THAT against a double-precision scalar reference on random data
  (identical codes, near-ties certified). kyutai then just calls the proven helper
  behind `CRISPASR_KYUTAI_RVQ_FAST` (default OFF) — so the integration is correct
  *by construction*, and only the end-to-end default-flip waits on a model. The
  shared helper is the seam that lets you validate without the system.
- **GGUF `load_weights` bounds test.** No crafted-corpus needed: write a valid GGUF
  with the ggml writer, then truncate it **8 bytes short of the tensor data** — so
  metadata still parses and control reaches the *specific* hardened check, not an
  earlier magic/metadata bail-out. (Truncating to zero passes the test but only
  exercises "failed to read magic" — a weaker guard. To guard a specific line,
  craft the input to reach exactly that line; confirm via the log it emits.)

General rule: "needs a model / a GPU / a crafted file" is rarely a hard blocker —
find the pure sub-transform that carries the risk, give it a testable signature,
and prove it deterministically. A gated path built on a proven helper is shippable;
a gated path you've never executed is not. Cf. the §176c-Dia measure-first and the
§176n empirical-repro lessons — same family: extract the checkable core.

## Metal-validated GPU TTS graph paths DID transfer cleanly to CUDA — the stricter-CUDA risk is real but not universal (§176n/§245, 2026-07-12)

LEARNING 35 warns that CUDA has stricter per-op contiguity asserts than Metal, so
a Metal-clean GPU default can still crash/mis-decode on CUDA (the dia `get_rows`
non-contig case). Counterweight datum: both GPU TTS paths validated on M1 Metal
this session — VoxCPM2's `VOXCPM2_USE_GRAPH` (discrete-GPU *mirror* path, device-
local VRAM) and qwen3-tts CP_DIRECT (sched-free persistent gallocr graphs) —
**passed the Kaggle P100/T4 TTS→ASR round-trip on the first run**, no contiguity
abort. So the rule is not "Metal→CUDA always diverges" but "Metal→CUDA is not a
proof": paths built on the disciplined ggml patterns (contig before `get_rows`,
gallocr not sched-reuse, explicit mirror weights) tend to transfer, while the
failure mode is specific ops with hand-strided views. Still run the CUDA
round-trip before flipping a GPU default there — but budget it as a confirmation,
not an expected-crash hunt. The confirmation is cheap: a `SystemExit`-on-fail
Kaggle kernel makes `COMPLETE` == PASS, so the run status alone is the verdict.

## Audit a whole roadmap CLUSTER in one measure-first pass before implementing any of it (§176/§245, 2026-07-12)

When several sibling PLAN items share a theme ("runtime optimization pass",
§176a–t + §245), don't pick one and implement — audit the cluster first, because a
roadmap ages unevenly and in predictable ways:
1. **Some are already done** — later cross-cutting work (fused graphs: CP_DIRECT,
   VOXCPM2_USE_GRAPH) silently closed them; the entry was never updated. A ~2-min
   reproduction reclassifies them (§176n VoxCPM2 "SIGSEGV" already ran on Metal at
   3.75×; §245 already done via CP_DIRECT).
2. **Some are mis-framed** — the named bottleneck is wrong; profiling redirects the
   fix (§176k "add KV+flash" → actually dispatch-bound → matvec cache).
3. **Some are real but low-value** — a per-fraction measurement caps the payoff below
   the risk (§176c Dia KV round-trip = 1.2% of a compute-bound decode → deferred).
This session: 5 items → 3 stale-done, 1 mis-framed (1 real fix), 1 low-value. Only
**one** warranted code. Auditing the cluster (verify-done, then measure-value)
before touching any item turned a "HIGH, implement several" row into "DONE/LOW,
stop chasing" — and every conclusion is backed by a run, not a reading. Cheap
instruments (per-op profiler, per-fraction timer, `*_BENCH` gates) are the whole
game; keep them in-tree as the evidence. Cf. [[verify-handovers-with-profiling]].

## BatchNorm-fold must respect the conv tensor's dtype — and a per-backend fix usually has siblings (canary family, 2026-07-12)

Folding a Conformer conv-module BatchNorm into the depthwise-conv weight
(`conv_dw_w`) at load time means read → multiply by the BN scale → write back.
Both the read and the write **must branch on `conv_dw_w->type`**. The trap:
`ggml_backend_tensor_get/set` with an F16-sized buffer against an **F32** tensor
reads/writes only half the bytes and reinterprets F32 bit patterns as F16 —
silent, total corruption of that layer's conv weights (the decoder then
hallucinates coherent-but-wrong output, same signature as a transposed mel).

Crucially this is **latent for shipped GGUFs**: converters emit `conv.dw.weight`
as F16 by default (`is_f32_tensor()` returns False for a 3-D non-norm/bias
tensor), so the F16→F16 path is bit-identical whether or not the code branches on
dtype. It only bites the **`--f32-encoder` / diff-harness path**, which is exactly
the workflow used to *validate* the port — so the bug hides until someone runs the
validation that would catch every other bug. `c9a4de65` fixed it for cohere +
parakeet; the identical defect sat unfixed in `canary.cpp` / `canary_qwen.cpp`
(F16 write-back) and `canary_ctc.cpp` (F16 read) until a follow-up audit (fixed
`702db9af`).

Two durable habits: (1) when a fix lands for one backend, `grep -rn fold_batchnorm
src/` and check the **whole family** — a per-backend defect almost always has
siblings sharing the copied code (canary_qwen's comment literally said "same as
canary.cpp"). (2) A dtype-safe read helper (`core_cpu::to_f32`) fixes the read side
for free; the write side still needs an explicit `type == GGML_TYPE_F32` branch —
don't assume "reads safely" means "round-trips safely."

Related, same commit family: HF `from_pretrained(..., trust_remote_code=True)` on a
**custom remote model** can run a `_init_weights` that re-inits all Linear/Conv to
`normal(0, 0.02)`, silently overwriting the pretrained weights → a reference dump of
pure noise. The reference backend needs a guard: detect it (a conv weight's RMS far
below any trained value, ~<0.1) and reload every param from the safetensors shard(s).
See `tools/reference_backends/_reload_guard.py`.

## "Migrate host KV to device-resident" is a ~1% win on a compute-bound transformer decoder — measure the fraction before the rewrite (Dia §176c, 2026-07-12)

PLAN §176c claimed host↔device KV round-trips were "the dominant data-movement
bottleneck" for several TTS decoders and wanted them migrated to device-resident
4D tensors. A `DIA_BENCH`-gated timer around Dia's per-step KV upload+readback
measured **201.6 ms of 17435 ms decode = ~1.2%** (M1, `dia-1.6b-q4_k`, 120 steps).
For a 1.6B decoder the transformer forward (×2 for CFG) is ~99% of the step; the
KV bandwidth is noise. So the "dominant" framing was wrong, and a device-resident-
KV rewrite — which is NOT bit-identical (it changes the compute graph: in-graph
`ggml_cpy`/`set_rows` writes + strided views vs a host round-trip) and lands in
ggml's most bug-prone area (KV write/read ordering on Metal/CUDA) — buys ≤1.2%.
Not worth the risk; deferred with data.

Rule: before any "reduce data movement / device-resident KV / fewer host
round-trips" change, add a one-off timer around EXACTLY the bytes you'd remove and
divide by total. A single-digit-% ceiling on a non-bit-identical graph change is a
decline, not a task. (This is the KV-cache-flavoured sibling of the §176k
dispatch lesson and the §208/§214 "measured DUD" pattern — the win must be proven
before the graph is touched, not after.)

## Verify a roadmap "broken/OPEN" claim empirically before implementing it — the codebase may have outgrown the note (VoxCPM2 §176n Metal, 2026-07-12)

PLAN §176n said VoxCPM2 was "CPU-only due to a Metal SIGSEGV" and scoped a fused-
graph rewrite to "unlock GPU." Before writing any code, one empirical run
(`crispasr --backend voxcpm2-tts`, default = GPU) showed it **already runs on
Metal**, correct (ASR round-trip) and **3.75× faster than CPU** — because the
`VOXCPM2_USE_GRAPH` fused graphs (built for a *different* PLAN item) had since made
the whole hot path Metal-safe. The SIGSEGV note was true only for the tiny CPU
helper matmuls, which nobody needs on GPU. The "medium rewrite" was zero code.

Durable habit: a stale OPEN item can cost a whole session of unnecessary graph
surgery. For any "X is broken on backend Y" claim, the FIRST step is the ~2-minute
reproduction (run it, read the actual failure mode), not a code plan. Related to
the §176k lesson (re-profile before trusting a handover's bottleneck) — same root:
**the map is not the territory; check the territory first.** Corollary carried
here: GPU-default was confirmed only on Metal (unified memory); the discrete-GPU
mirror path needs a Kaggle CUDA round-trip before trusting it there (LEARNING 35).

## Per-step matvec dispatch overhead is LOAD-DEPENDENT — a "win" measured on a busy box is mostly a contention artifact (FireRed §176k, 2026-07-12)

Replacing the FireRed decoder's per-call matvec graphs (each a fresh `ggml_init`
+ build + `sched_reset` + `sched_alloc_graph` + `ggml_free`) with a persistent
per-`(weight, M)` gallocr graph reused across steps is a **pure Pareto** change
(bit-identical output; strictly less work per call). But the *magnitude* of the
speedup depends entirely on machine load, and measuring it wrong nearly shipped a
6× overstatement:

- **Quiet box (load ~3.7):** OFF ~0.188 vs ON ~0.178 ms/call → **~5%**. The removed
  ops (malloc, sched reset/alloc) are cheap when the allocator/scheduler aren't
  contended, so the cache saves little.
- **Loaded box (load 20–50):** OFF up to 22 vs ON up to 20 ms/call; min-of-N OFF
  1.15 vs ON 0.71 → **~1.6×**. Under contention, malloc + scheduler ops get
  disproportionately penalized (lock contention, TLB/cache pressure, deschedule),
  so removing them helps far more.

Takeaways: (1) the **clean-box number is the honest headline** — a big speedup
that only appears under load is telling you the *baseline* is contention-fragile,
not that your change is a 1.6× win (reinforces the "noisy box fabricates wins"
A/B rule); (2) for the dispatch-only comparison, **isolate the metric you changed**
(a `FIRERED_BENCH` timer summing time in the dispatch function) and use
**min-of-N** — the minimum is the least-descheduled sample, so its ratio survives
load that makes wall-clock useless; (3) a per-call graph rebuild is
contention-fragile even when it looks cheap at idle — persistent graphs also buy
**robustness under load**, which is a real (if unglamorous) production benefit on
a shared box. See [[firered-matvec-cache-load-dependent]].

**Meta-lesson (same session):** the §176k handover asserted a KV-cache/flash-attn
bottleneck; profiling showed the KV was already cached and the decode was
dispatch-bound. **Re-profile per-node before trusting a handover's bottleneck
theory** — cf. the #81 case. The fix that shipped was the opposite of the one the
plan named.
## Porting a libllama-hosted reference onto our in-house Qwen3 was clean on first contact; and Kaggle `kernels_logs()` is the escape hatch when the output is page-capped (MOSS-TTS #249, 2026-07-12)

Three things from the MOSS-TTS-v1.5 port (`OpenMOSS-Team/MOSS-TTS-v1.5`, an 8B
Qwen3 + 32-codebook delay + 1.6B transformer codec):

- **The seam to graft a libllama-hosted reference onto CrispASR is `inputs_embeds`
  + a `hidden_last` output.** `pwilkin/openmoss` hosts the Qwen3 backbone with
  libllama and reads back per-token hidden states; we forbid libllama. The in-house
  Qwen3 (`moss_audio.cpp`/`qwen3_asr.cpp`) already takes `inputs_embeds` first-class,
  and `qwen3_tts.cpp` already marks the pre-lm-head hidden as an output — so the port
  is: clone the KV graph, add `ggml_set_output` on the post-final-norm `cur` named
  `hidden_last`, and run the aux graphs (summed embed, 32 heads) + codec + delay
  verbatim from the reference. **The ~2,500 lines of hand-ported codec/delay graph
  math were CORRECT on first contact with the real 8B weights** (Q4_K decoded
  round-trip intelligible). Cloning a *validated* reference verbatim beats building
  from the paper — the earlier a port's graph matches a known-good one, the fewer the
  first-contact bugs.

- **An F16 8B backbone (~17 GB) does not fit a 16 GB Kaggle P100/T4** — `cudaMalloc
  out of memory` at load. Validate the HARD-RULE-#3 acceptance test on **Q4_K** on
  Kaggle (fits in ~7 GB), not F16; F16 needs a >20 GB GPU (L4/A100) or CPU. Don't
  read an F16 load-OOM as a port failure.

- **`KaggleApi().kernels_logs(slug)` returns the full stdout log as a string** — the
  reliable way to read a kernel's traceback/results when `kernels_output` is blocked
  by the 500-file page cap (kaggle_usage #22; e.g. `CCACHE_DIR` under
  `/kaggle/working` buries `progress.txt` behind thousands of ccache files). Keep
  `/kaggle/working` tiny AND remember the logs API bypasses the file cap entirely.

## A Metal masked-attention-padding penalty does NOT transfer to CUDA, and a bucket-width change is byte-identical on Metal/CPU but not on CUDA (tada §215b bucket floor, 2026-07-12)

The tada talker's positive CFG pass runs through the §176b decode bucket, which
floors attention width at Lk=512; for a short generation (n_past ≪ 512) that pads
~500 masked columns per step. On M1 Metal the penalty is huge — the positive pass
drops **266→49 ms/call (5.4×)** when forced onto the exact-Lk path. On a clean CUDA
P100 the *same* change is only **1.43×** at the pass level and **1.02–1.06×** at the
loop level: 512-wide masked attention is cheap on a GPU built for it. This is the
LEARNING-34/35 "a Metal win doesn't generalise" rule, quantified — always measure
the *target* platform before flipping a GPU default; a Metal profile can be an order
of magnitude off for CUDA.

Worse, the bucket width is **not output-neutral on CUDA even though it is on
Metal/CPU**. `soft_max_ext` sums the −inf-masked padding as `exp→0` terms; on a
deterministic backend (CPU, Metal) adding exact zeros doesn't change the reduction,
so the output is byte-identical regardless of Lk (proven md5-equal). On CUDA the
*parallel reduction order* over Lk=512 vs a tight Lk flips a borderline FP bit → a
different acoustic frame, AR-amplified — benign (still intelligible) but observably
different bytes, with a coherent signature (`floor64 == nobucket ≠ default` on
short). Consequence: "byte-identical AND faster", the bar to flip a GPU default, is
met on Metal/CPU but not CUDA → ship a **backend-conditional** default
(`tada_default_bucket_min`: Metal/CPU→64, discrete-GPU→512), not a global one.
Corollary for A/B harnesses: on a GPU, gate a sampled/AR TTS on **ASR keyword-recall
of every arm**, not md5 — md5 is a bit-determinism signal, not a correctness gate
(same lesson the dia CUDA A/B taught). Also: the tight-floor *bucket* beats both the
512-floor bucket AND raw NO_BUCKET on long inputs — NO_BUCKET rebuilds the graph
every step and that eventually costs more than the padding it saves, while the
cached tight bucket has neither cost.

## ggml disables CUDA graphs below Ampere (cc < 800) — an Ampere-only re-warm bug can't be reproduced on Kaggle's P100/T4 (irodori #243, 2026-07-12)

irodori's RF-DiT rebuilds its graph + a fresh `gallocr` every ODE/CFG call
(~100/generation), so the tensor addresses change each step and ggml's CUDA-graph
machinery sees "properties changed" → warmup resets and re-completes every step
(`ggml-cuda.cu:4361`), paying the capture/update cost instead of the capture-once-
replay benefit (the reporter's "CUDA graph warmup complete" spam). A persistent
cached graph is the textbook fix. But the *time cost* is unmeasurable on any hardware
we have: ggml gates CUDA graphs on `cc ≥ GGML_CUDA_CC_AMPERE` (800,
`ggml-cuda.cu:4329`), and Kaggle offers only **P100 (cc 600)** and **T4 (cc 750)** —
both pre-Ampere, so CUDA graphs are off there and the re-warm never happens; Metal
has no CUDA graphs at all (and measured the DiT **98% compute-bound** → the graph
construct+alloc a persistent graph removes is only 1.7%, so no Metal/CPU win either).
So the disciplined move: implement the sound fix, prove it **byte-identical** on the
hardware you *do* have, ship it **opt-in**, and delegate the Ampere-only speedup
confirmation to whoever has the GPU — do not fabricate a number you cannot measure.
**Kaggle is not a substitute for an Ampere box** for CUDA-graph work.

## ggml-metal im2col starves batch-1 convs — 3–11 threads/threadgroup at N=1, ~40× below bandwidth (HiFi-GAN perf, 2026-07-11)

ggml-metal's `kernel_im2col` sized thread-dim0 from batch N (`ntptg0 =
min(max/(KH·KW), N)`). At inference **N=1** that's 1 → each threadgroup runs only
`KH·KW` (3–11) threads = ~10–34% of ONE 32-wide simdgroup, across millions of tiny
threadgroups → im2col ran ~40× below bandwidth. On the melotts HiFi-GAN decode that
was **58% of GPU time** (per-node profile), with cont-transpose another 22% and the
actual matmul only 12%. Fix: block a range of OW across thread-dim0 so threadgroups
fill (`CRISPASR_METAL_IM2COL_OCC`, auto-on N==1) — a **bit-exact copy reorg**, ~2–3×
on hifigan, 1.65× on moonshine, no regression. General batch-1 Metal conv win.
**CUDA does NOT have this** — its im2col parallelizes threads over IC·KH·KW (hundreds),
not N. Two meta-lessons: (1) the **ideal roofline lied** — it put im2col at ~1%; only
a per-op MEASUREMENT (added `CRISPASR_METAL_PROFILE=2`) on a **quiet** box found the
real 58% (a loaded box also inflated the whole-graph number 2×, 4.02e6 vs true
1.75e6). (2) the existing `ggml_conv_2d_direct` is NOT a free fused-conv win — it's a
naive scalar kernel, 2.25× SLOWER than im2col+mul_mm.

## Validating a TTS perf change by WAV correlation is invalid — melotts `--seed` isn't deterministic; use ASR round-trip or a deterministic-ASR bit-check (HiFi-GAN perf, 2026-07-11)

melotts (VITS) is not fully deterministic even with `--seed` — the same seed gave
115712 / 95232 / 115712 samples across three runs (unseeded component in the
stochastic duration predictor / flow noise). A bit-exact kernel change therefore
shows corr≈0 and different durations, which looks EXACTLY like a correctness bug —
this cost a long false-alarm cycle where a valid ~2× im2col fix was wrongly gated as
"broken." Validate TTS kernel changes RNG-immune instead: (a) **ASR round-trip**
(synth → transcribe → text recognizable, both arms); (b) for bit-parity, run a
**deterministic ASR** (no RNG) with the change on vs off — identical transcripts prove
bit-parity on that model's conv shapes (moonshine occ-on == occ-off byte-for-byte
settled it); (c) best, an im2col/stage-level diff on a fixed input tensor. Reserve
audio-corr for a pipeline you've VERIFIED deterministic at a pinned seed.

## Two more untrusted-input lessons: an HTTP server reads the body before your auth check, and a parser that abort()s on malformed input is a DoS a fuzzer finds in seconds (server + ggml hardening, 2026-07-11)

Extending the audit to `crispasr-server` (cpp-httplib) and the model loader
surfaced classes the file-parser sweep didn't:

- **The framework reads the request body BEFORE your handler runs.** cpp-httplib
  accumulates the whole multipart body into `req.files[...].content` in
  `read_content()` — which `routing()` calls *before* `dispatch_request()`, so
  your `require_auth()` inside the handler is too late. Without
  `set_payload_max_length()` (default `SIZE_MAX`) a large upload OOM-kills the
  process **pre-auth**. Set the cap (it makes `read_content` return 413 + skip
  without buffering — verified in the httplib source), and note its *chunked*
  reader ignores the cap, so reject `Transfer-Encoding` on uploads in a
  pre-routing handler (which does run before the body read).
- **Any request field that becomes a filesystem path needs `..`/absolute/NUL
  rejection at the choke point** — even when a *sibling* route already validates
  it (the `/v1/voices` handlers validated the name; `/v1/audio/speech` passed
  `voice` straight through to a path). Audit by data-flow, not by endpoint.
- **A parser that `GGML_ASSERT`/`abort()`s on malformed input is a DoS**, not a
  safe failure — you cannot catch `abort()`. ggml's `gguf_init_from_file`
  aborted on an empty KV key; any untrusted model load could crash the process.
  The fix is to make the parser *return an error* on malformed input, not
  assert. A libFuzzer harness over the load entry point found this in seconds —
  and after fixing it, ~880K runs across both parse paths found nothing more.
  The lesson: fuzz the file-load entry points; asserts-as-validation are the
  first thing it trips.

## Every hand-rolled file parser sizes a buffer from an untrusted length field — that is the #1 memory-safety bug class, and a multi-agent audit finds them fast (parser hardening, 2026-07-11)

CrispASR parses attacker-controllable bytes in several places: the audio
demuxers (`crispasr_audio.cpp`: Sun-AU, WAV, MP4/box, WebM/EBML, Ogg/Opus,
AMR), `core/wav_reader.h`, and the GGUF loader (a model file is untrusted too).
A structured audit turned up the **same defect class in five independent
parsers**:

1. **Alloc sized from an untrusted length/count with no clamp to the real
   file/buffer size.** `vector.resize(count)` / `vector(n)` / `malloc(n)` where
   `n` is a header field. A tiny file claiming a 4 GB `data_size` or a
   `0xFFFFFFFF` sample count forces a multi-GB allocation → OOM / `bad_alloc`
   thrown across a C ABI (DoS). Seen in AU (`data_size`), WAV (`chunk_size`),
   MP4 (`stsz`/`stco`/`co64` count). **Fix: always clamp the field to the bytes
   the file actually holds** (`min(field, file_end - pos)`, or box-entry bytes).
   The bounded read *loop* after the alloc does NOT save you — the alloc fires
   first.
2. **Additive bounds checks overflow.** `off + sz > size` wraps when `off` is a
   64-bit attacker value near UINT64_MAX, so the guard passes and a wild pointer
   is dereferenced (OOB read). **Fix: subtractive form** — `off > size || sz > size - off`.
3. **A degenerate size makes a loop emit N items with no forward progress.**
   WebM fixed-size lacing with `frame_size == 0` pushed 256 empty packets per
   block while the cursor never advanced; a cluster of tiny blocks amplified it
   ~768× into hundreds of GB of empty vectors. **Fix: reject the zero case +
   cap the accumulator by file size** (a valid item occupies ≥1 byte).
4. **One path in a family missing the guard its siblings have.** Three GGUF mmap
   paths bounds-checked `data_off + off + nbytes > mf.size`; `load_weights_split`
   didn't → SIGBUS / info-leak. **Fix: grep the sibling guards and mirror.**

**Methodology that worked.** A multi-agent workflow — parallel auditors, one per
parser region, each handed the bug taxonomy + a *confirmed exemplar* (the AU
`raw(data_size)` bug), then an **adversarial verifier per finding** whose default
is "refuted" and who re-reads the code to check for an existing guard. 10
findings, all 10 survived verification, 6 unique real bugs across 5 subsystems in
one pass — the diversity is the point (one reviewer fixates on one parser). The
verify step matters: "the alloc is bounded because the later loop is bounded" is
the tempting-but-wrong refutation to reject. Judge by the alloc, not the read.
Regression-test the *reachable* path (the internal demuxers are `static`, so
drive them through `crispasr_audio_load` with a crafted file; a libFuzzer harness
over that entry point exercises all formats at once).

## MelsTime mel layout IS ggml Conv1d input layout — transposing corrupts with cos ~0.3, not a gradual drift (moss-diarize #242, 2026-07-11)

ggml `Conv1d` with kernel ne=(K, IC, OC) expects input ne=(T, IC): T is the fast (contiguous) axis, IC groups sit at stride T. The `core_mel::compute` MelsTime layout `mel[f*T+t]` has T contiguous per mel bin — **this IS the correct ggml layout**. Transposing to `mel[t*IC+f]` (IC contiguous per timestep) produces cos ~0.3 from the conv stem — total garbage, not a small drift. The diff harness mel stage passes (cos=1.0) because it compares the raw mel before feeding to the encoder, so it doesn't catch a transposition applied between mel and conv1d.

**Lesson.** (a) Don't assume "Conv1d needs (T, channels) row-major" — ggml ne-order has ne[0] as the fast axis, which is T, but the *data layout* for `(T, IC)` in ggml is `data[t + ic*T]` = MelsTime. (b) When the mel cos passes but the encoder cos is garbage, the bug is in how mel data is fed to the encoder graph, not in the mel computation itself. (c) A tiny C++ test with known data + known kernel verifies the ggml Conv1d is correct (it was); the bug was in the data marshalling.

## Handover docs can be dangerously stale — read the actual Python source, not the handover (moss-diarize #242 + dots-tts #200, 2026-07-11)

The moss-diarize handover had wrong time marker interval (5s stated but code default 2s — actually the saved processor_config.json said 5s), wrong marker format (bracket strings `[5.00]` vs bare digit tokens `5`), wrong prompt structure (system+user turns vs single user turn), and missing 30s audio chunking. The dots-tts handover listed 8 "open issues" that were all already fixed or never existed (PatchEncoder attention was already implemented with manual attention, BigVGAN vocoder was already working, EOS detection was already implemented, speaker encoder was already wired). Both handovers were written at specific commits and never updated as fixes landed.

**Lesson.** (a) A handover is a starting point for orientation, NOT the blueprint. The blueprint is the Python source code. (b) HARD RULE #1 ("read the Python line-by-line before writing C++") exists for this reason — a handover that says "no RoPE" when the config says `rotary_bias=true` costs hours. (c) When picking up a handover, `git log --oneline <last_handover_commit>..HEAD -- src/<backend>*` shows what changed since the handover was written.

## A one-frame error in a codec ENCODER shifts every RoPE position and shows up as an onset artifact, not a gradual drift — and a padding scheme faithful to one codec can be wrong for another (pocket-tts voice-clone "gong", 2026-07-11)

Pocket-TTS voice-cloned syntheses opened with a loud onset transient (a "gong"); no-voice output was clean. The decoder was the wrong place to look: force-decoding the *reference's* exact latents through OUR Mimi decoder was already clean (onset RMS 0.0045), and force-decoding OUR latents gonged (0.0621) — same decoder, so the bug was upstream in the **encoder**. Root cause (reference diff vs kyutai `pocket_tts`/moshi at F16): `mimi_encode` padded the reference PCM to a **hop-length (120)** multiple, but the encoder does a further `/downsample_stride` (16) conv to reach the 12.5 Hz latent rate. Padding only to a hop floored the downsample one short — **137 voice frames where the reference ceils to 138** (jfk ≈137.5). That one missing voice frame shifted every downstream RoPE position by one, so the *first generated* latent was conditioned on a mis-positioned context → the decoder rendered it as the gong. Fix: pad to a whole **latent** frame (`hop * downsample_stride` = 1920 samples). Onset RMS 0.0621 → **0.0045**, byte-matching the reference profile `[0.0045 0.0034 0.0034 0.0047 …]`.

**Then the instructive part — a plausible "hardening" that measurement rejected.** `qwen3_tts`'s SEANet+RVQ codec computes moshi's `MimiConv1d._get_extra_padding_for_conv1d` **per strided conv** (ceil framing at each stage), which is more principled than a single global pre-pad — so it seemed obvious that pocket-tts's Mimi VAE encoder "should already do that too." It should not. Ported per-conv into `mimi_encode`, built, diffed against the reference at seed 42: our onset jumped to **0.0579** while the reference's is **0.0045** — a >12× divergence that sampling can't explain. The pre-pad already **value-matches** the reference (both 0.0045); pocket-tts's Mimi VAE encoder effectively pads the input to a whole latent frame and does NOT use independent per-conv extra padding. Reverted.

**Boundary hardening — the fix matches the reference's `ceil(N/1920)` at every length, including exact multiples.** Diffed moshi's voice-frame count (the KV-cache `offset`) across trimmed lengths: N=261120→136, 263040→137, 263041→138, 264000→138 — always `ceil`, and our C++ matches all of them. One trimmed clip (exactly 263040 = 137×1920, so *zero* trailing pad added) produces a loud onset (t0≈0.066) — but the **reference generates the same loud onset (t0≈0.063) for that clip**, so it's faithful content behavior of a mid-utterance truncation, not a bug. Real refs never land on an exact 1920-multiple anyway.

**Lessons.** (a) An off-by-one in an autoregressive-latent *encoder* is a **positional** bug — it corrupts the onset via RoPE, it doesn't smear evenly; when the *first frame* is wrong, suspect frame **count/alignment**, not per-element math. (b) The codec is guilty only if force-decoding a known-good latent set through it also fails — otherwise localize upstream. (c) A padding/framing convention is a property of the *specific* codec, not the family — do not assume moshi-per-conv (right for qwen3's codec) transfers to the Mimi VAE encoder; prove it with a diff. (d) **What to grade against, and what NOT to trust.** The reliable signals were **frame-count vs the reference** (deterministic, exact) and **onset-magnitude/character vs the reference** — a same-seed our-vs-our onset delta is fully valid, and even across RNG streams a >10× magnitude gap vs a close match is decisive (0.0579-vs-0.0045 = divergent; 0.066-vs-0.063 = faithful). A `backbone_out0` correlation via a `flow_net` forward-pre-hook proved **unreliable** — it read ~0.17 even for the faithful exact-multiple case, because the hook's input isn't the same quantity/layout as our dump. Don't anchor a verdict on a correlation you haven't confirmed is measuring the same tensor. (e) The §224 voice-latent disk cache keys on the PCM hash, not the encode algorithm — clear `pocket-voice-*.latents` (or `CRISPASR_POCKET_VOICE_CACHE=0`) after any encoder change or a stale latent set masks it.


## LLM-ASR (SALM) instruction-echo on short windows is genuine model behaviour, not a port bug — gate the input, don't chase the prompt (canary-qwen #247, 2026-07-11)

canary-qwen leaked "Transcript"/"PASS"/"Okay" into transcripts. Two prior "fixes" failed: string-stripping the echo (masks symptom, and only touched text not the tokens array), then removing `<|audio_start|>`/`<|audio_end|>` framing tokens ("root cause") — the reporter confirmed the leak survived both. It survived because neither was the cause.

**Actual root cause, proven against the NeMo SALM reference (`nvidia/canary-qwen-2.5b`):** the leak is the Qwen3-1.7B decoder falling back to its language prior when the audio window is too short to ground a transcription. The FastConformer subsamples 8×, so a window with only a few encoder frames (`T_enc<=5`, ~≤0.4 s) carries no transcribable content, and the LLM echoes the task framing ("Transcribe the following:" → "Transcript") or emits a filler ("Okay"). Measured, C++ q8 vs NeMo bf16 on the SAME clips:

| clip | CrispASR C++ | NeMo SALM reference |
|------|--------------|---------------------|
| jfk 11 s (full) | "And so my fellow Americans ask not what your country can do for you ask what you can do for your country" | **identical** (byte-for-byte, same token ids) |
| 0.10 s | "Okay" | "Okay" |
| 0.30 s | "Transcript" | "Transcript" (ids [3167,1228]) |
| 0.45 s | "And" | "And" |

The port is faithful — the prompt is byte-identical to NeMo's `QwenPromptFormatter.encode_dialog` (`[151644,872,198,3167,3114,279,2701,25,220,<audio>,151645,198,151644,77091,198]`, no system role, no BOS), and full-utterance output matches exactly. NeMo's `SALM.generate()` runs plain greedy with **no** `suppress_tokens`/`begin_suppress_tokens`/`min_new_tokens`, so it has zero protection and echoes too. There is nothing to "match" in the blueprint — the fix belongs in the pipeline.

**Fix (two env-gated layers in the shared `canary_qwen_transcribe_impl`, so CLI + session ABI + diff all inherit it — avoids the HARD-RULE-#6 mirror trap):** (1) a degenerate-window gate returns empty when `T_enc < 6`; (2) a backend-agnostic safety net strips a leading `Transcript`/`Transcription`/`PASS` from **both** the text and the tokens array (also closes the #218 "tokens array not deduped" note; the same `fix_loops_keep_indices` dedup was additionally mirrored into the inline session path in `crispasr_c_api.cpp`, which previously emitted un-deduped tokens).

**Lessons:** (a) for LLM-based ASR, "hallucinates meta-text on empty/tiny input" is expected — the acceptance test is a reference comparison on the SAME degenerate clips, not a prompt audit; (b) when a "root cause" fix is contradicted by the reporter, reproduce before theorising again — a length sweep (`s_100..s_600`) localised it in minutes; (c) `q8` vs `bf16` diverge on the knife-edge degenerate input (0.15 s: C++ "Transcript" vs NeMo "Okay") but agree exactly on real speech — judge parity on the decoded output of a *real* utterance, not a 2-frame window.

**Follow-ups (2026-07-11).** (b) The `canary_qwen` diff-harness reference backend (`tools/reference_backends/canary_qwen.py`) was a stub — this port shipped with NO per-stage parity check, which is *why* the mel-transpose (#233) and this echo issue slipped through. Now implemented via SALM: hook `perception.preprocessor` (mel) + `perception.encoder` (1024), call `perception()` for the projected 2048 embeds. `crispasr-diff canary-qwen` on jfk (q8 vs bf16): `mel_spectrogram` cos_mean 0.9964, `projected` cos_mean 0.9996 (cos_min 0.997 — the normal F16/quant band, not a structural fail), decoded text verbatim. (c) The reporter runs **Vulkan**; I A/B'd a MoltenVK build against Metal/CPU on the same clips — **byte-identical** on both the full utterance and every degenerate clip (`0.1 s→"Okay"`, `0.3 s→"Transcript"`). So the echo is **backend-independent**, not a Vulkan miscompute; the pipeline fix resolves it on every backend. MoltenVK on M1 is correct but ~10× slower (0.4× realtime) — fine for correctness A/B, useless for perf.


## Baked mel filterbank layout MUST match core_mel's `fb_layout` — a transpose produces cos_min ≈ −0.15, not a gradual drift (canary-qwen #233, 2026-07-09)

The canary-qwen converter's `compute_mel_filters()` returned `(n_freqs, n_mels)` (FreqsMels layout), but `core_mel::Params` defaults to `FbLayout::MelsFreqs` = `(n_mels, n_freqs)`. A transposed filterbank doesn't produce a "slightly wrong" mel — it produces an **anti-correlated** one: `cos_min = −0.15`, `cos_mean = 0.43` against the Python reference. The LLM hallucinated coherent English ("Yes sir, sir, sir") because it saw noise-like features, not silence.

**Diagnostic that pinpointed it in one run:** `crispasr-diff` on Kaggle comparing just the mel stage — `[FAIL] mel_spectrogram cos_min=-0.147 max_abs=8.30` — instantly identified the filterbank as the root cause vs encoder/projection/LLM bugs. **Fix:** one line: `mel_fb = mel_fb.T` in the converter. After: `cos_min = 0.999`, JFK exact match.

**Lesson:** when baking a mel filterbank from scratch (not copying the model's own), always verify the layout convention. The two conventions (`MelsFreqs` vs `FreqsMels`) produce the exact same values in a different memory order — there's no shape mismatch to catch, no NaN, no OOB. The ONLY way to detect it is a cosine comparison against a Python reference mel. Gate mel parity BEFORE debugging downstream stages.


## Never cache a ggml compute graph across calls that share a scheduler with a larger graph — the allocator regrows and frees the cached graph's GPU buffers (#215, #235, 2026-07-09)

The `§176s` encoder graph cache (`cached_enc_gf`) reuses a graph across audio slices when `T_mel` matches. On CPU this is safe — the ggml allocator is stable. On GPU (CUDA/Vulkan), the LLM prefill/decode graphs that run AFTER the encoder are larger, so `ggml_backend_sched_alloc_graph` regrows the scheduler's internal buffer pool, which **frees** the `VkBuffer` / `CUdeviceptr` handles that the cached encoder graph's tensors still reference. Slice 2 re-allocates the cached graph → the scheduler writes through freed GPU handles → use-after-free → segfault or driver-level busy loop.

**Impact:** #215 (moss-transcribe) was the first instance, fixed ad-hoc. #235 (glm-asr-nano) was the second. An audit found the same vulnerable pattern in 8 backends total: glm-asr, voxtral4b, moss_audio, canary_qwen, voxtral, granite_speech, funasr, qwen3_asr. All fixed by removing the cache-hit branch — graph build is ~1 ms, compute is what's expensive.

**Encoder-only backends** (canary, parakeet, nemotron, kyutai_stt, moonshine, paraformer) are safe: they have no LLM decoder on the same scheduler, so the allocator never regrows between slices.

**Rule:** on a shared `ggml_backend_sched`, never cache a graph across calls if ANY other graph on the same scheduler could be larger. The graph topology can be reused (same `ggml_context` arena), but `sched_reset` + `alloc_graph` must run before every compute — and the allocator's regrow in `alloc_graph` for a larger graph frees the previous graph's device buffers. The only safe cache is when the cached graph is the LARGEST graph the scheduler ever sees.


## Flow-matching DiT with JointAttention REQUIRES attention masking even for unconditional generation — skipping masked KV tokens silently corrupts output (2026-07)

Porting Irodori-TTS (RF-DiT flow-matching TTS with JointAttention over self + text + speaker context), the text encoder reached cos=0.9995 parity and single-step DiT v_pred reached cos=0.9984, yet the 40-step ODE produced garbage audio (random language hallucinations on ASR roundtrip). The root cause: **the Python model always includes speaker KV in the JointAttention, even when unconditional (no speaker reference), with an attention mask that has -inf for speaker positions**. The C++ initially skipped speaker KV entirely (no speaker state → don't concat), which changed the KV sequence length and thus the softmax distribution. Even though masked tokens get zero attention weight in Python, their presence in the denominator affects all other weights.

**Fix:** pass `mask` to `ggml_flash_attn_ext` (F16 tensor with 0.0 for attend, -inf for masked positions). Must be `GGML_TYPE_F16` — F32 asserts. Shape `(n_kv, n_batch)` matching the KV sequence length including masked positions.

**Lesson:** when porting a model that uses attention masking, you CANNOT skip masked tokens as an optimization — you must include them with proper masking. `ggml_flash_attn_ext(q, k, v, nullptr, ...)` does NOT mean "no masking needed" — it means "full bidirectional attention over all KV positions". If the upstream model uses `attn_mask` to selectively ignore positions, the C++ must replicate this exactly.

**Corollary:** single-step parity (cos≈0.999) is NECESSARY but NOT SUFFICIENT for flow-matching models. The ODE trajectory is exponentially sensitive to attention distribution differences. A per-step cos of 0.9984 sounds high but accumulates to cos≈0.94 over 40 steps, which is fatal for flow-matching TTS. The ASR roundtrip is the true test, not the per-step cosine.


## The Mimi codec transformer must be causal — non-causal silently truncates long audio; the >250-frame WER A/B settled it (2026-07)

Sweeping CrispASR for the CrispEmbed `deepseek_ocr2` bug class (a masked attention replaced by
`ggml_flash_attn_ext` with `mask=nullptr` → every query attends to everything → garbage) surfaced the
**Mimi (Kyutai) codec transformer** in `kyutai_stt.cpp:704` (STT encoder) and `csm_tts.cpp:923` (TTS
decoder). Both run the transformer with **full non-causal attention** (`mask=nullptr` over the whole
frame sequence), yet:
- upstream moshi's Mimi is a **streaming causal** transformer with `sliding_window=250`,
- the RoPE two lines up is literally commented "causal",
- `mimi_context = 250` is **loaded from GGUF (`kyutai.mimi.encoder_context`) but never applied** — dead config,
- every SEANet conv in the same files is `conv1d_causal`.

So the transformer is the one non-causal island in an otherwise-causal codec. **This is a real
divergence, but NOT a crash/garbage bug like deepseek_ocr2** — the models transcribe/synthesize fine
today, so at most it's a quality question. Per the A/B mandate it must be proven before any default
change, so it's implemented **opt-in** behind `CRISPASR_MIMI_CAUSAL` (default = today's non-causal;
the flag builds a causal + sliding-window-250 F16 mask, filled after alloc: attend iff `k<=q && q-k<
window`). **A/B (kyutai-stt-1b q4_k, jfk):** non-causal → "…do for you. Ask what you can do…", causal →
"…do for you, ask what you can do…" — both correct, differ only by one downstream punctuation choice;
the causal path neither asserts nor degrades. But jfk is ~137 mimi frames < 250, so the *window*
never engaged — that only exercised causal masking and came out quality-equal, which is why the first
pass looked like a non-issue.

**RESOLVED — the >250-frame A/B settled it (2026-07).** Re-ran on 3× jfk (33 s ≈ 412 frames, so the
window engages): the old **non-causal path TRUNCATES the tail** — it transcribed 2 of 3 repetitions
then cut to "And so, my fellow Americans," (dropping ~25 % of content), because attending over the
full 412-frame sequence is out-of-distribution for a model trained with a 250-frame streaming window.
**Causal+window transcribed all three in full.** So causal *wins on long audio and ties on short* — a
real win, not equality. **The default is now causal+sliding-window** for `kyutai_stt`; the old
full-attention path is preserved for bisection behind `CRISPASR_MIMI_NONCAUSAL=1` (gate the OLD path
once the new is proven — mandate). Lesson: **an A/B on a sample shorter than the model's context
window can't see a context/windowing bug** — size the probe to exceed the window (here, concatenate
the clip past `mimi_context` frames). `csm_tts`'s decoder has the identical gate and now **also
defaults to causal** (opt out with `CRISPASR_MIMI_NONCAUSAL=1`): a TTS→ASR round-trip A/B (2026-07,
~256 decoder frames — `T_up = 2 × 128` codec frames, so the window engages) gave causal **9.3% WER
vs non-causal 12.0%** — causal wins and is never worse. Note the TTS margin is far smaller than the
STT one: on TTS the non-causal codec stayed fully intelligible (no truncation), it just synthesised
slightly-worse audio, whereas on STT non-causal dropped ~25% of the transcript. Deterministic RNG
(reseed) made the two synths identical in length (128 frames / 30.56 s), so the only variable was the
codec attention. Single-sample and modest, but directionally consistent with STT + the architecture,
so flipped for consistency.

## Auditing CrispASR against CrispEmbed's bug classes: a weight reader with no quantized branch fails silently (2026-07)

After CrispEmbed fixed three bugs (a `sched_reserve`→re-alloc-same-graph CUDA corruption in
lfm2_colbert, a `clean_exit`/`atexit` imatrix-flush miss, and a quantized-weight read sized by
`n_elem*sizeof(float)`), we swept CrispASR for the same shapes. Two were clean:

- **`sched_reserve`→re-alloc-same-graph:** all sites are safe. `cohere.cpp` and `qwen3_tts.cpp`
  each reserve a *dedicated* worst-case/measurement graph, then build a **fresh** graph for compute
  (llama-style); `talk-llama` is vendored upstream. The trap is re-allocating the *same* `ggml_cgraph`
  you passed to `ggml_backend_sched_reserve` — `sched_reset` doesn't null `tensor->buffer`, so the
  reserve pass's stale buffers get reused (silent CUDA corruption). None of CrispASR's sites do that.
- **`clean_exit`/`atexit`:** N/A — CrispASR has zero `atexit(` registrations and no imatrix collector,
  so there's nothing for `_exit()` to skip.

The third had a live lookalike: **`paraformer.cpp`'s `cif_predict` `read_f32` helper handled only
F32 and F16 with no `else`** — a block-quantized tensor would leave the destination *uninitialized*
(silent garbage, no assert), the same class as the pcs FC-head bug. Rewrote it to mirror
`pcs_read_tensor_f32`: F32 fast-path, everything else dequantized per row via the type's `to_float`
trait (native bytes sized by `ggml_nbytes`, row stride `ggml_row_size` — **never** `n_elem*4`, which
reads garbage / asserts on quantized tensors).

**Why it was inert (and the general rule).** Shipped paraformer GGUFs keep the CIF weights at F16/F32:
the quantizer skips them because they're named `.w` (not `.weight`, so they miss the `is_weight` test)
and are 3-D conv / 1-D vector (fail the 2-D `ok_dims` gate), and the biases are `force_f32` in the
converter. So the bug couldn't trigger today — but "the quantizer happens to skip this tensor" is a
fragile guarantee living in a *different* file. **Any helper that reads a weight tensor to a CPU F32
buffer must handle F32, F16, *and* quantized via `to_float`/`ggml_nbytes`, not silently fall through.**

**Runtime-verified (auto-download, 2026-07):** `crispasr --backend paraformer --auto-download` on
`samples/jfk.mp3` (paraformer-zh q4_k, sha256-checked vs HF) → `CIF predicted 26 tokens from
T_lfr=183` → *"and so my fellow americans ask not what your country can do for you ask what you can
do for your country"* — correct, so the F16 CIF read path (`to_float`) is exercised and right.

## A persistent KV cache reused from a larger allocation makes local `max_ctx` the wrong stride (#171 VibeVoice server leak)

A "server works for the first N requests then produces garbage, but the CLI is
always clean" bug is almost never the GPU kernel — it's **host state that survives
between calls on the shared context.** In VibeVoice TTS the persistent
`ctx->kv_k/kv_v` is only reallocated when it must **grow** (the
`kv_max_ctx < max_ctx` guard). So after one long request bumps `kv_max_ctx`, every
shorter request **reuses the larger buffer** — and its real `ne[1]` is now
`ctx->kv_max_ctx`, *not* this call's local `max_ctx`. Any code that computes an
offset into that tensor from the local `max_ctx` writes to the wrong place.
`copy_voice_kv` did exactly this for the per-head destination stride
(`hd*max_ctx*el`), so the voice prompt for heads ≥1 landed at wrong offsets and
corrupted speaker conditioning. The server's startup **warmup** synth grows
`kv_max_ctx` first, so even the first real short request was already corrupted —
which is why the CLI (one fresh process, first alloc == exactly `max_ctx`) never
showed it. **Rule: when writing into a persistent/reused ggml tensor, take strides
from the tensor's own `nb[]`, never from a per-call size variable** — the read side
(attention) already used `nb[2]`, so the write side disagreeing was the whole bug.

Two diagnostic lessons that made this cheap to pin down:
1. **Deterministic-seed byte-identity is a state-leak detector.** Every stochastic
   knob here reseeds to 42 per call, so *correct* output for a fixed input is
   bit-stable. Reproduce a "server degrades over time" report by sending the same
   input at position #1 vs after other inputs and `cmp` the PCM — if they differ,
   you have a host-side leak, no ears or ASR needed. It reproduced on **Metal**
   even though the report was RDNA4, because the leak is backend-independent.
2. **Rule out the "obvious" GPU cause from the report itself.** The reporter had
   already shown `GGML_VK_DISABLE_COOPMAT2` + `VIBEVOICE_TTS_FLASH_ATTN=0` +
   `VIBEVOICE_VAE_BACKEND=cpu` (all at once) didn't help — that triple-negative was
   the signal to stop chasing the coopmat2 shader and look for pure host state.

## Two concurrent HuggingFace uploads from one machine cause spurious mid-batch failures — serialize them (#192 aligner re-ship)

Uploading a batch of GGUFs while another `hf`/`upload_file` job runs on the same
machine looks like random flakiness but is deterministic: the two contend for the
CAS/Xet endpoint and outbound bandwidth, and the *starved* one fails in ways that
don't obviously say "contention." During the aligner batch, a parallel English
re-upload running alongside it made `it`/`ja` **conversions** fail (the converter's
safetensors download was bandwidth-starved) and `ch`/`pl` **uploads** fail their
post-upload `list_repo_tree` verification. Killing the concurrent upload and
re-running the batch serially fixed all four on the first pass, no code change.
So: **one HF upload at a time per machine.** Bake sequential discipline into batch
scripts — `create → upload → verify-on-HF → rm local → next` — and never kick off
a second uploader "to save time." (Same root cause as the higgs-stt note that two
concurrent Xet uploads throttle to bandwidth≈1.) Verify with the committed
`list_repo_tree` size, not the uploader's own exit code.

## Quantizing a forced-aligner: q8 everywhere (incl. lm_head) is bit-identical; q4 on the encoder is not (#192 TADA aligner)

The TADA voice-reference aligner is a wav2vec2 CTC model over the Llama-3.2 128k
BPE vocab. Quantization safety is **different for forced alignment than for free
CTC ASR** (cf. the OmniASR learning "Q4_K too lossy for CTC-decoded ASR", 22.7%
WER). Methodically verified against an f16 gold on 2 JFK inputs (metric = per-token
alignment position drift in 20 ms frames + the resulting `token_values` cosine):

| config | size | pos max\|Δ\| | token_values cos |
|---|---|---|---|
| q8 enc + F32 lm_head (originally shipped) | 906 MB | 0 frames | 1.000000 |
| q8 enc + F16 lm_head | 644 MB | 0 frames | 1.000000 |
| **q8 enc + q8 lm_head** | **520 MB** | **0 frames** | **1.000000** |
| q4 enc + F32 lm_head | 755 MB | **11 frames (~220 ms)** | 0.952 |

Conclusions: (1) the **encoder** attn/FFN weights are safe at **q8** (bit-identical)
but NOT q4 — noise compounds through 24 residual layers and flips knife-edge CTC
boundaries by up to ~220 ms (a q4 that happens to score 1 frame on some input is
a lucky tie-break, not evidence). (2) The **lm_head** (128k×1024 CTC projection,
~525 MB F32) is safe even at **q8** — it's a single projection into the DP, no
compounding — so storing it F32 wasted ~385 MB for zero benefit. (3) Norms,
biases, conv feature-extractor, pos_conv stay F32. So the optimal bit-identical
aligner is **q8 everything = 520 MB (43% smaller than the F32-lm_head build)**.
lm_head is normally skipped by crispasr-quantize (LLM/free-CTC sampling-critical);
`CRISPASR_QUANT_LMHEAD=1` overrides it — only safe because the forced-alignment DP
is robust to q8 logit rounding. `token_values` cos tracks position drift because
the encoder pools features **at** the aligned positions, so a shifted alignment
degrades the acoustic prompt too.

Follow-up (norms/biases/convs/pos_conv): these stay F32 by **runtime
requirement, not just precaution — and there are TWO independent requirements**,
so "just fix the assert" doesn't unlock it:
1. **CPU read path.** `wav2vec2_compute_logits` reads norms/biases via
   `tensor_data_f32()` — a raw `float*` view (src/wav2vec2-ggml.cpp:916–1280) —
   so F16 storage halves the byte count → `GGML_ASSERT(... tensor read out of
   bounds)`. This one *is* individually fixable (dequantise non-F32 to a cached
   F32 buffer in `tensor_data_f32`, mirroring bark/chatterbox), but it unlocks
   nothing on its own because of (2), and the norms/biases are only ~1 MB.
2. **Metal graph path.** The convs / pos_conv (and even a norm/bias used as an
   op input) flow through Metal ops that assert `op->src[1]->type == F32`
   (ggml/src/ggml-metal/ggml-metal-ops.cpp:3355). Fixing this needs load-time
   dequantisation of those tensors into F32 ggml tensors — a real change to the
   load path, for ~26 MB (mostly pos_conv).
Verified both empirically (F16-all = 494 MB clears (1) then hits (2); F16
norms/biases-only = 519.5 MB, ~1 MB saved, still hits (2)). Net: **520 MB
q8-everything is the practical floor**; the ~26 MB isn't worth converting the
Metal conv path. Both experimental gates (converter `TADA_ALIGNER_AUX_F16`, the
`tensor_data_f32` dequant) were reverted — keeping the code honest that aux is
F32 rather than half-enabling a path Metal still blocks.

## A production default that diverges from the upstream reference is a whole class of bug the stage-cosine diff can't see (#192 TADA)

TADA's #192 "last word clipped / crammed tempo" survived every numerical diff because the
diff harness pins the reference to a non-default mode for determinism. The actual bug was
the *CLI default*: `num_acoustic_candidates=4` where upstream `InferenceOptions` defaults to
**1**. Best-of-4 mangled "…four hours" → "…and forth"; N=1 was verbatim. The fix was to match
the upstream default, found only by running the real model (`HumeAI/tada-3b-ml`) at *its*
default and comparing text + `time_before`, not by looking at cosines. Lessons: (1) when a
port has "knobs", diff the **default config** against upstream, not just the forward pass;
(2) reproduce the reporter's exact sentence against the real model — the reference `time_before`
`[19,2,11,8,11,11,9,11,9,9,10,10]` (smooth) vs ours `[…,43,…,2]` (spike) named the bug instantly.
Also: a parallel session "fixed" the same symptom the other way (cand 4→**8**); more candidates
on a bad scorer is not a fix — it got lucky on one model/seed and still mangled 1b/q4_k.

## Best-of-N only helps if the scorer measures what you care about — TADA's reconstruction scorer is blind to duration outliers (#192)

TADA draws N flow-matching noise samples per token and keeps the "best". The default
reconstruction ("likelihood") scorer computes MSE on the **acoustic dims only**, so a candidate
with great acoustics but an outlier gray-code **duration** (a 43-frame gap in a ~10-frame
context) scores well and gets picked — making cand>1 *worse* than a single draw. The cure is to
score the thing that fails: `duration_median` (a Python-provided scorer) picks the candidate
whose per-token duration is closest to the median, dropping the timing outliers that are the
whole reason to draw >1. Even so, no single scorer wins on every input (duration_median can
mispick acoustics elsewhere), so the honest answer was "N=1 is the reliable default; N>1 is an
opt-in lottery." When best-of-N underperforms best-of-1, suspect the scorer, not the sampler.

## A converter that embeds a tokenizer MUST embed the merges — a silent `try/except: pass` byte-fallback breaks everything downstream (#192)

`convert-tada-aligner-to-gguf.py` wrote `tokenizer.ggml.tokens` but its merge extraction
(`tokenizer.backend_tokenizer.model.get_merges()`, absent in the pinned `tokenizers`) threw
inside a `try/except: pass`, so `tokenizer.ggml.merges` was silently missing. With no merges,
C++ `core_bpe::tokenize_simple` falls back to **byte-level** tokens — a 26-char transcript
becomes 26 tokens instead of ~6 BPE tokens. That mismatched the runtime's BPE prompt
tokenization (acoustic-token count ≠ text-token count) and made voice cloning synthesize
silence. The tell was "tokenized 'And so my fellow Americans' → 26 tokens" (== the byte count).
Parse `tokenizer.json`'s `model.merges` directly (handles both `"a b"` and `["a","b"]` formats)
and **fail loudly** if extraction yields nothing. Corollary: quantizing the aligner to q8_0 is
bit-identical for alignment because the F32 `lm_head` (the CTC projection the DP reads) is
preserved — only the wav2vec2 encoder layers quantize.

## A forced aligner is not an ASR model — its free CTC argmax is all-blank (#192)

The TADA aligner is a wav2vec2 CTC model over the Llama-3.2 128k BPE vocab and gives excellent
per-token timecodes — but only in *forced* mode (given the transcript, the DP finds where each
token fits). A free greedy CTC decode (argmax per frame, collapse blanks) on jfk audio produced
**0 tokens** — the argmax is the blank/pad on all 549 frames. Trained-for-alignment CTC models
are peaky toward blank without the text prior; don't expect them to transcribe. They are ideal
for a `--align` word-timestamp / subtitle mode, not for recognition.

## When the engine and the PyTorch reference produce the SAME wrong output, the engine is right — the harness input (usually the prompt) is the bug (moss-transcribe)

The MOSS-Transcribe port passed every numerical diff stage — mel cos 1.0,
`enc_layer_0` cos 1.0 (so conv front-end + windowed attention were bit-exact), and
the first decode-token **argmax matched the reference** — but the transcript was
garbage ("Verm -100 years from now …"). The instinct is "the diff passes, so where's
the bug?". The answer: the bug was in *what we fed the model*, and both the C++
runtime and the Python reference fed it the same wrong thing, so they agreed with
each other while both being wrong. I had built the prompt as the bare audio layout
`[<|im_start|>, <|audio_start|>, audio…, <|audio_end|>]`; the real model wants the
`chat_template_default.py` ChatML framing
(`<|im_start|>user\n<|audio_start|>`·audio·`<|audio_end|><|im_end|>\n<|im_start|>assistant\n`).
Without the `assistant` cue the model never enters transcribe mode and hallucinates.

Lessons: (1) a diff harness validates the *engine*, not the *task setup* — if the
reference reuses the runtime's (wrong) prompt/preprocessing, a clean diff proves
nothing about the prompt. Cross-check the prompt against the model's
`chat_template*` / processor, not just the layer activations. (2) "C++ == reference,
both wrong" is a *positive* signal: it rules out the compute graph and points at the
shared input. Compare to the granite-plus case (#205) where the timing was right but
the content was wrong → wrong chat template; same family of bug. (3) A second,
unrelated foot-gun stacked on top: the prompt buffer was sized `T_enc+8` but the
template needs `T_enc+10`, so `build_prompt` returned −1 and the transcribe silently
produced *empty* output — size prompt scratch generously and check the return.

## An input feature the checkpoint wasn't trained with is worse than nothing — the moss-transcribe time-marker experiment (#218)

`processing_Moss.py` has an `enable_time_marker` mode that interleaves digit
tokens (the running second count, digit d → token id 15+d) into the audio token
stream every 2 s at 12.5 tok/s — an input-side temporal anchor, not an output
alignment. The tempting hypothesis (#218): those anchors might give the greedy
decoder positional grounding and curb the long-audio repeated-phrase loops,
letting us use larger chunks and shrink the segmentation tradeoff. Tested it
behind `CRISPASR_MOSS_TRANSCRIBE_TIME_MARKER=1`, with the prompt construction
verified **byte-identical** to `_build_audio_tokens_with_time_markers` for 8
lengths (incl. edge cases 13/25/26). Result on `t32-145s.wav` at 30 s, raw
(no loop-fix): **markedly worse.** The model *reads the digits aloud* — slice 1
became "Two, three, four, five, six, seven, eight, nine, ten, eleven…", slice 3
"Two, four, five, six…", slice 6 "…how do you know **it's two**…" — hallucinated
other slices, looped catastrophically (a 66-token slice ballooned to the 512-token
cap: "go go go come on!" ×N), and ran 2.5× slower (0.5× vs 1.2× RT). Cause: the
feature defaults to `False` (the README example sets it `False`) and this preview
checkpoint was RL-tuned with markers **off**, so it has no learned meaning for the
interleaved digits and transcribes them as speech. Lesson: an input-conditioning
feature that exists in the *processor* code is worthless — actively harmful —
unless the *weights* were trained with it. Verify the checkpoint's default/training
config before wiring a processor knob into the runtime; byte-perfect parity with
the reference *construction* proves nothing if the model was never trained on it.
Reverted (not shipped); documented so it isn't re-attempted.

## Stage the PyTorch reference load — encoder and LM as separate standalone modules, never the monolithic `*ForCausalLM` (moss-transcribe)

On a 16 GB box the obvious `MossForCausalLM(config)` + `load_state_dict` reference
OOMs at >12 GB **and crashed the machine twice** — the default-dtype init allocates
the whole 2.4 B model in float32 (~9.6 GB) before any bf16 weights load, and even a
bf16 in-place stream peaks ~5 GB on top of everything else. The fix is to never
hold the whole model: the diff stages are independent, so dump them in phases.
Phase A instantiates *only* `Qwen3OmniMoeAudioEncoder` + the adapter (`MossGatedMLP`)
standalone (~0.6 B, ~2.6 GB f32), streams just the `model.audio_model.*` /
`model.audio_adapter.*` tensors into them (the 1.7 B LM weights are never read), runs
the encoder, saves the adapter output, and frees everything. Phase B then loads
*only* `Qwen3Model` standalone (bf16, ~3.4 GB) with a tied `lm_head` reconstructed as
`F.linear(hidden, embed_tokens.weight)`, splices the saved Phase-A output into the
prompt embeds, and runs the prefill + a manual greedy loop. Peak ≈ max(phase) ≈
3.4 GB instead of >12 GB. Two gotchas in Phase B's manual decode: build the model
under `torch.set_default_dtype(torch.bfloat16)` (else the float32 init spikes), and
pass `cache_position` on every cached step — omitting it gives wrong RoPE positions
and the *reference* derails into a "Y. Y. Y." runaway that looks exactly like a
quantization failure but is a decode-loop bug in the harness, not the model.

## Parakeet single-pass DROPS whole sections of long audio — chunk-and-merge, and pick the reference clip carefully (#208b)

Parakeet's TDT decoder loses track once the input runs past ~30 s: a single
full-length pass on a 296 s clip emitted **338** words where the audio holds
~620 (it silently skips whole sentences/sections, not a few words at the edges).
Three traps while diagnosing this:
1. **It is not a quantization problem.** The F16 model dropped *more* (255) than
   Q4_K (338). Don't reach for "the quant is lossy" — A/B a higher-precision
   build before blaming the quant.
2. **whisper is not automatically ground truth.** whisper-large-v3-turbo also
   dropped a section here, and its word count (501) looked authoritative — but
   the clip genuinely *repeats* a passage (a sloppy FLEURS concat) and whisper
   emitted it once. Transcribing the two time windows *independently* (short
   single-pass, fully reliable) proved the audio says it twice → parakeet's
   "duplicate" was correct and whisper was the one losing words. When two ASR
   models disagree ~40-68 % on a clip, suspect the *clip* (adversarial / repeated
   / concatenated content), and verify with short-window crops before trusting
   either as a reference.
3. **WER vs another model is the wrong yardstick for "did *my* pipeline lose
   words".** It's dominated by model disagreement. Use a clip with *exact*
   ground truth instead: concatenating jfk.wav ×12 (132 s, known 264-word
   transcript) gave a precise, model-agnostic measure — the merge scored 264/264
   even with an identical sentence every 11 s (time-based splice, so repeated
   text doesn't confuse it).

Fix pattern (`parakeet_session_chunked_merge`): short **overlapping** windows
(20 s / 8 s overlap), splice at the overlap midpoint **continuing from the last
committed word's end** (continuing by `t0 >= mid` instead drops the one word that
straddles the midpoint — measured as a ~3.7 % loss vs single-pass on clean
German; `t1 > cont` fixed it to 99.6 %), then an adjacent-dedup pass for
jitter-doubled seam words. Net: ≥99 % retention on clean audio AND recovery of
the sections a single pass drops on long/adversarial audio.

## A cached cgraph is NOT re-entrant with ggml_backend_sched — the 2nd reuse silently corrupts (#208)

§176s cached the built Parakeet encoder graph (`cached_enc_gf`) and reused it
whenever `T_mel` matched, to skip the rebuild. With the shared
`ggml_backend_sched` this is wrong: `ggml_backend_sched_reset` +
`ggml_backend_sched_alloc_graph` on the *same* cgraph object does not clear the
cached compute tensors' `buffer`/`data` from the previous allocation, so the
**second and every later** encode of that `T_mel` reads partially-stale memory.
The failure is not a crash or NaN — the encoder output **shifts** (jfk enc std
0.020 → 0.014, a valid-looking but wrong tensor) and the TDT decoder then emits
only blanks → **empty transcript**. It hid for months because the CLI runs one
encode per `T_mel` per process; it only bites a **long-lived context** that
encodes ≥2 times: a server/session making repeated `transcribe` calls, the
streamed/chunked long-audio encoders (same-size chunks), and silence-split
long-form. Diagnosis was a 3-line A/B: call `transcribe` ×3 on one session →
1st verbatim, 2nd/3rd empty; gating the cache off made all 3 verbatim. Lessons:
(1) graph-build caching that hands a reused cgraph to a sched needs the tensors'
backend assignment reset (or a dedicated reserved gallocr per cached graph, the
"single-backend raw-gallocr" pattern) — you cannot just stash the cgraph;
(2) validate any perf cache with a **repeated-call** test, not a single run — a
one-shot CLI invocation can never surface a 2nd-use corruption; (3) "works in the
CLI, empty from the binding" smells like cross-call context state, not a binding
bug. Fixed by defaulting to rebuild-every-call and gating the cache opt-in
(`CRISPASR_PARAKEET_ENC_CACHE=1`). See [[HISTORY]] #208.

**2026-07-01 follow-up — the cache is a DUD, reentrancy is not worth building.**
The #208 reporter asked whether a re-entrant cache should be built to speed up
chunked long audio (they assumed rebuild-every-window was the penalty). Measured
on M1 Metal (Q4_K 0.6B-v3) via a new `PARAKEET_ENC_PROBE` hook, per uniform 20 s
session window: **graph build ≈ 0.3 ms | `sched_alloc` ≈ 1 ms | GPU compute ≈
920–1000 ms.** A re-entrant cache can only skip the ~0.3 ms build (`sched_alloc`
runs every call regardless) → ≈0.04 % of per-window wall time. Same dud shape as
the chatterbox CFM gallocr cache (§208). The probe also reproduces the corruption
numerically: `fill` windows have enc std ≈0.020, the 2nd+ `reuse` windows collapse
to ≈0.0078 and the session transcript drops the middle windows. Conclusion: keep
rebuild default, keep the cache opt-in-OFF as a benchmark hook only, **do not**
reimplement it re-entrantly. The real chunked-long-audio headroom is the encoder
GPU compute and the ~40 % overlap re-encode in `parakeet_session_chunked_merge`
(20 s window / 8 s overlap), not the graph cache. Generalizable rule (already
learned for CFM in §208): **measure build+alloc with `ggml_time_us()` BEFORE
porting any graph-cache — on a compute-bound backend it is almost always noise.**

## On Metal a cached decode graph is dispatch-bound, not alloc-bound — measure the step's parts, don't trust wall time (§210, PR #207)

PR #207 made granite-speech's single-token LLM decode run a cached, shape-stable
graph (fixed `Lk` bucket) so **CUDA-graph capture** engages (~9-13× on RTX 5090,
bit-identical). The natural follow-up on Metal — which has no capture — is the
[[HISTORY]] #208 lesson applied as a *speedup*: hand the reused cgraph a
**dedicated reserved `ggml_gallocr` on the single GPU backend, allocate once, and
reuse it every step** with `ggml_backend_graph_compute`, skipping the per-step
`ggml_backend_sched_reset`+`sched_alloc_graph` that capture forces. It's correct
(byte-identical, `granite_dec_use_gallocr` gates it: gallocr unless CUDA/HIP or a
CPU layer-split; CUDA keeps the sched so capture still fires) and strictly less
work — but **measuring it taught the real lesson twice over**:

1. **M1 end-to-end wall time is unusable for a perf A/B.** Identical-config runs
   swung 1.7× (3385↔5752 ms decode) purely from background macOS load
   (`softwareupdated` at 90%, `ANECompilerService`, `cloudd`) and leftover
   zombie procs from a timed-out batch. The fix is to **instrument the specific
   quantity** — accumulate per-step *alloc* µs vs *compute* µs into the ctx
   (env-gated `CRISPASR_GRANITE_DEC_PROFILE`) — which is deterministic and
   attributable regardless of total noise. Don't A/B two configs on wall time;
   measure the one thing your change touches, inside the step.

2. **The alloc you're removing is tiny; the decode is dispatch-bound.** On a
   quiet machine the sched per-step alloc is only ~3 ms (it *balloons* to
   68-236 ms under memory pressure — the real robustness win on a 16 GB box),
   while the dominant ~100 ms/step is **Metal per-op dispatch of the ~280-op
   graph** — exactly the launch-bound cost CUDA capture eliminates and that
   gallocr does **not** touch. `CRISPASR_METAL_N_CB` (1/2/4) made no difference,
   and disabling concurrency/fusion only made it slower (they already help).
   The natural hypothesis was that **indirect command buffers**
   (`MTLIndirectCommandBuffer` — encode the fixed-shape step once, replay, the
   Metal analog of CUDA-graph capture) would remove this floor. **It would not —
   see the measured DUD below.** Note the contrast with the chatterbox-CFM
   gallocr **DUD** ([[HISTORY]]): there alloc was ~0.3% of a compute-heavy UNet
   step → no win; here per-token compute is small so alloc is a *large* fraction
   under load → gallocr helps. Same trick, opposite verdict — decided by the
   compute/alloc ratio, which you can only know by measuring (#1). See
   [[HISTORY]] §210.

### ICB graph replay on Metal is a measured DUD — the M1 decode is GPU-bound, not host-launch-bound (§210 follow-up)

Before committing to the (large, invasive) ICB refactor of vendored ggml-metal,
I ran the §210 lesson on itself: **measure the part the change touches before
building it.** ICB's entire value is collapsing the *host-side* per-step re-encode
(280→1446 `setComputePipelineState`/`setBytes`/`setBuffer`/`dispatchThreadgroups`
objc calls) into one `executeCommandsInBuffer` replay. So the only number that
matters is: *what fraction of the step is host encode?*

Added `CRISPASR_METAL_PROFILE` to `ggml_metal_graph_compute`
(`ggml-metal-context.m`): it splits each graph_compute into **encode_us** (all
ops encoded + all command buffers committed — the host work ICB removes) vs
**gpu_us** (a forced `waitUntilCompleted` after commit — GPU execute + submit/sync,
which ICB cannot touch). granite-4.1-2b-q4_k, jfk.wav, Metal, gallocr path,
27 decode steps (cold step 0 dropped):

| part | median | share | what ICB does to it |
|---|---|---|---|
| **host encode+commit** | **1.19 ms** | **1.8 %** | collapses to ~one encode |
| **GPU execute + sync** | **64 ms** | **98 %** | unchanged (same kernels, same bandwidth) |
| total/step | 66.3 ms | — | cross-checks the dec-profile compute (70 ms) |

**Verdict: ICB ceiling is ~1.2 ms of ~66 ms ≈ 1.8 %, even if it were free and
removed 100 % of host encode.** Not worth a multi-thousand-line refactor of
vendored upstream Metal code (and it has a hard blocker anyway — every op binds
its args via `[encoder setBytes:atIndex:0]`, which `MTLIndirectComputeCommand`
does **not** support; ICB can only `setKernelBuffer`, so each op's arg struct
would need a GPU-resident buffer slot — see `ggml-metal-device.m:548`).

**Why the §210 "dispatch-bound" framing was half-right.** The ~100 ms *is* spent
mostly outside alloc, but it is **GPU-side**, not host-side dispatch. The 1446-node
graph's cost is per-layer Q4_K GEMVs reading ~1.5 GB of weights once per token —
**weight-bandwidth-bound on the M1 GPU** (~64 ms ≈ 1.5 GB at a fraction of the
~68 GB/s base-M1 bandwidth, as single-token GEMV achieves). Host encode of all
1446 ops is only 1.2 ms and already overlaps GPU via `n_cb`. The encode time is
*rock-stable* across steps (1.0–2.0 ms) while gpu_us carries all the background-load
variance (55–107 ms) — further proof the bottleneck is the GPU, not the CPU encode.

**Why CUDA graphs win 9–13× on the same model but ICB wouldn't.** Opposite regime.
On an RTX 5090 the GPU finishes each tiny single-token kernel faster than the CPU
can launch the next → **CPU-launch-bound**, so capturing the launch sequence is a
huge win. On M1 the GPU is the bottleneck and CPU encode is 1.8 % → **GPU-bound**,
so the Metal analog (ICB) has nothing to remove. Same graph, same trick, opposite
verdict — decided by the host-encode/GPU-compute ratio, which (again) you only
know by measuring. The real M1-decode levers are GPU-side: lower-bandwidth quant,
or fusing the many small GEMV/elementwise ops into fewer larger kernels — **not**
ICB. `CRISPASR_METAL_PROFILE` is kept (env-gated, off by default) as the reusable
probe for any future "is this Metal graph host- or GPU-bound?" question.

### Would other backends benefit from the granite shape-stable-decode rewrite? Mostly only on Ampere+ CUDA (§210 survey)

The two wins PR #207 + the Metal follow-up gave granite (CUDA-graph capture; Metal
gallocr allocate-once) **both require the same prerequisite**: the per-step decode
graph must be *shape-stable* — a fixed-`Lk` KV bucket written via `ggml_set_rows`
at a runtime index, so the topology is byte-identical every step. The naive
anti-pattern (most backends) is a KV view sized `Lk = n_past + T`, which grows
every token → no capture, no allocate-once. Surveyed every LLM/AR-decoder backend
(2026-06-30) for which side of that line they're on:

- **Already shape-stable** (get both wins): `granite_speech` (PR #207),
  **`mimo_asr`** — independently: cached `step_t1_gf` + `fixed_kv_len` + `set_rows`
  scatter + skip-plan reuse (`mimo_asr.cpp:211,958,1507-1522`); `dots_tts` has the
  Metal gallocr allocate-once (`ec74c5a0`).
- **Growing-shape + naive per-step rebuild** (candidates): `voxtral`, `voxtral4b`,
  `qwen3_asr`, `gemma4_e2b`, `glm_asr`, `higgs_stt`, `ark_asr` (all `Lk = n_past+T`);
  `lfm2_audio` (growing but already on a gallocr for the §206 fix); naive TTS
  decoders `csm_tts`, `indextts`, `bark_tts`, `moss_audio`, `mini_omni2`.

**The catch (why this is *conditional*, not a roadmap to mass-port):** the headline
CUDA-graph win is **arch-gated to Ampere+ (sm_80+)** (`ggml-cuda.cu:4329`), so the
project's usual CUDA test GPUs **T4 (sm_75) and P100 (sm_60) get nothing** — only
RTX 5090 / A100-class do. And on **Metal there is no throughput win at all** (this
decode is GPU-bound, per the measurement above) — only memory-pressure robustness.
So porting the granite pattern to N other backends is a per-backend manual graph
rewrite (set_rows bucket + in-graph argmax + capturable/fused embed + diff-harness
parity) for a payoff that materializes only on Ampere+ NVIDIA. Do it **targeted,
on deployment demand, measured first** — not as a sweep. CUDA-graph capture itself
is automatic once the graph is capturable (no k-quant `GET_ROWS`, no split buffer,
no problematic `MUL_MAT_ID`) and shape+pointer-stable; no per-backend CUDA code is
needed, just the shape-stable rewrite. Actionable port plan + per-backend file:line
anchors + recipe in [[PLAN]] §210 follow-up.

## A "garbled GPU output" bug can live entirely downstream — A/B the GPUs before localizing (§192)

The #192 native-Vulkan TADA garbage was localized (by a prior handover) to the
"FM head's time-dimension divergence" and a whole work plan was written around
per-op FM diffing. That was **wrong**. The decisive test was a one-line A/B:
generate the same sentence on **Metal** vs **Vulkan** native. Both produced
**bit-identical durations and frame counts** (`time_before` matched exactly) —
proving the talker + FM + duration decode AGREE across GPUs — yet Metal's audio
was intelligible and Vulkan's was empty. So the divergence was purely in audio
**rendering = the codec**. Per-stage `TADA_CODEC_DUMP` (Vulkan vs CPU, identical
features) localized it precisely: the codec's attention encoder MATCHES across
backends, but the **DAC decoder front-end explodes** — `dump_dac_in` (an
`in_conv` Conv1d → im2col+mul_mat) hits rms ~37 / range ±800 on Vulkan vs ~0.85
on CPU (~43×). It is **size-dependent** (short inputs render fine; only longer
sequences break). Crucially it is **NOT a conv kernel bug** — a standalone repro of
`conv_1d`/`im2col`/the full `wn_conv1d` at the codec's exact dims and T=522 is
bit-correct on MoltenVK, and capping `GGML_VK_FORCE_MAX_ALLOCATION_SIZE` doesn't
help — so it's a graph-scale gallocr/aliasing-class corruption in the *large* real
graph, unreproducible in a minimal harness. Fix: run the whole codec on CPU when
the GPU is Vulkan. **Beware the two wrong explanations I went through
first** (both plausible, both false — verify, don't assume): (a) "it's a
`ggml_backend_sched` cross-backend-copy bug" — NO, `GGML_SCHED_DEBUG=2` shows the
codec is a *single Vulkan split*, no offload, no copies; (b) "Vulkan lacks the
conv/ISTFT kernels" — NO, ggml-vulkan has conv_transpose_1d/col2im/im2col/
flash_attn, and the codec has no ISTFT at all (it's a DAC conv decoder).
**Lessons:** (1) when two GPU backends exist, A/B them first — if they agree with
each other but disagree with CPU-rendered output, the bug is in the
backend-specific *rendering*, not the shared upstream math; (2) inherit a
handover's *localization* skeptically — re-run its cheapest disproving test
before building on it (the FM premise was wrong — F32 FM weights and whole-FM-on-
CPU both left output bit-identical); (3) localize with the per-stage magnitude
dump before naming a cause — `dump_dac_in` rms 0.85→37 named the op in one run,
after two hand-wavy mechanisms wasted effort. (4) `grep` the actual op list +
`supports_op` before claiming "no kernel"; check `GGML_SCHED_DEBUG=2` before
claiming "cross-backend copy".

## Precomputed attention masks in a GGUF are dead weight if the runtime rebuilds them (§192)

The `tada-codec-fixed-f16.gguf` was **1.0 GB**, and **805 MB of it (76%)** was six
`codec.attn.blk.*.self_attn._precomputed_mask` tensors — `(8192, 8192)` F16
block-attention masks, one per layer. The C++ codec never binds them: it rebuilds
`attn_mask` from `token_masks` per decode (`tada_codec.cpp build_decode_graph`).
So they were loaded into RAM (and shipped/downloaded) for nothing. Dropping them
(converter skip + a one-shot strip of the existing file) gives a **250 MB** codec
with **byte-identical** output (`cmp` clean, ASR verbatim). Lessons: (1) when a
"quantize it" footprint ask comes in, first **inventory the tensors by size** —
the biggest line items may be non-weight tables you can delete outright, which
beats any quant (and here the conv weights couldn't quantize anyway: DAC conv
kernels store `kernel_size=7/3/1` as ne[0], below the Q8_0/Q4_K min block size of
32/256). (2) A precomputed mask/positional table is only worth storing if the
runtime actually reads it — grep the loader's bind step before trusting a fat
GGUF. (3) Quantization shrinks *weights* only; activations stay f32, so it does
nothing for the compute-buffer growth on long inputs.

## MoltenVK `mul_mm`/`mul_mat_vec` downconvert src0 to f16 regardless of stored dtype (§192)

Storing FM weights as F32 instead of F16 on MoltenVK changed the matmul output by
**exactly zero** — the shader downconverts src0 to its `FLOAT_TYPE` (f16) anyway,
so f32-stored and f16-stored weights round identically. Even
`GGML_VK_DISABLE_F16=1` only nudged a hidden-state cosine 0.99919→0.99966. So on
MoltenVK you **cannot** reach CPU-exact GPU numerics by changing weight dtype;
the only way to match CPU for a precision-critical sub-graph is to run it on the
CPU backend. (On RADV, f32 matmuls accumulate in f32, so this is a MoltenVK
quirk, not universal — but don't try to "validate a GPU f32 fix" on MoltenVK.)

## Vulkan has no `REPEAT f16→f16` — cast K/V to F32 *before* the GQA repeat (§192)

GQA decoders expand KV heads with `ggml_repeat_4d` on the cached K/V. When the
cache is F16 (the default), that's an `f16→f16` REPEAT — which **Vulkan has no
pipeline for**, on RADV *and* MoltenVK alike (`Missing op: REPEAT for f16 to
f16`). It aborts the whole graph. The fix isn't a new kernel: cast K/V up to F32
*before* the repeat (`core_attn::KvSelfAttnParams::force_kv_read_f32`, OR'd with
the global `CRISPASR_KV_READ_F32`) so it lowers to a supported F32 REPEAT.
Gate it to the Vulkan path; Metal/CPU keep the F16 fast path. **Caveat learned
the hard way:** unblocking the *abort* is not the same as fixing *correctness* —
the graph then ran but still produced garbage on longer text (a separate
numerical divergence). "It runs now" ≠ "it's right now."

## On macOS, GGML_METAL is default-ON and silently wins over Vulkan (§192)

A build configured with `-DGGML_VULKAN=ON` but Metal left at its default still
has Metal, and `--gpu-backend vulkan` will quietly run **MTL0**, not Vulkan. I
spent a cycle "confirming a Vulkan fix" that was actually executing on Metal
(where the bug doesn't exist). Always build the Vulkan test tree with
`-DGGML_METAL=OFF`, and **read the `backend=Vulkan0` vs `backend=MTL0` log
line** before trusting any Vulkan A/B. The original `crispasr-vk-build` had
Metal OFF; a freshly-configured worktree build did not.

## Never benchmark on a near-full disk — SIGBUS + nondeterminism masquerade as a code signal (§192)

With `/Volumes/backups` at 100% (a few GB free), TADA-Vulkan runs threw
intermittent **SIGBUS** (failed `mmap` of model files) and gave
**nondeterministic** frame counts for *identical* invocations (148 vs 191 vs
522). That noise looked exactly like a real A/B signal and sent me down a dead
end ("disable FM B=2 fixes the drift" — it doesn't; ON and OFF both give the
same 522-frame runaway once the disk is freed). Lesson: a parity/A-B result that
won't reproduce across repeats is an *environment* alarm, not a finding — check
`df` and re-run the same config twice **before** forming a hypothesis. Confirmed
fixes only after the disk was freed and runs were deterministic.

## Always A/B against the *upstream reference*, not just your own option matrix (§216)

Parakeet long-form shipped a "streamed" (chunked-encode + single-decode) default
because a 2026-05-26 option matrix found it beat the dispatcher's chunk-30+merge.
Both were inferior: on the v3/multilingual model the streamed path *collapses*
(5 min → 75 words), and chunk-30+merge duplicates a phrase at every boundary.
The matrix never included the obvious third option — **one full-attention pass
over the whole clip** — which turns out to be byte-for-byte identical to upstream
NeMo (`nvidia/parakeet-tdt-0.6b-v3`, 706/706 words, 30 s→5 min, 100.00 %).

Lessons:
- When a model has a Python reference (NeMo here), make the reference output the
  baseline for the A/B, not just internal variants of your own pipeline. A
  length sweep (30/60/120/240/300 s) vs the reference localized the collapse
  immediately and proved single-pass was exact. Run the reference on macOS by
  stubbing `sys.modules["eventlet"]=None` before `import nemo` (else the
  wandb→trio import dies with "unsupported platform").
- "Complete output" ≠ "correct": chunk-30+merge produced *more* text than
  streamed, which is exactly why it was picked — but the extra text was boundary
  duplicates. Diff against the reference, don't count words.
- Token-id-exact LCS dedup can't stitch overlap regions that two chunks
  transcribe *differently* (different encoder context → different tokens). If
  you must chunk, cut at silence and commit each piece by word timestamp at a
  single shared cut — no overlap region to dedup.
- Full self-attention is O(T²): ~5 min is memory-safe on 16 GB, ~28 min OOMs
  (it crashed the dev M1). Cap single-pass and silence-split beyond the cap;
  each piece stays bounded. See HISTORY §216.

## §214 Metal batched (B=2) quantized mat-vec ≠ the single-token PREC_F32 path

When you batch a single-token decode into `B=2` (cond+uncond CFG, hidden
`(D,1,2)`), the projection `ggml_mul_mat(W, x_b2)` becomes a mat-vec with
`ne[1]=1, ne[2]=2`. With **F16 weights** this is bit-identical to two separate
forwards on Metal. With **quantized weights it is NOT**: the batched path does
not dispatch Metal's PREC_F32 `mul_mv_q*_K` exact-dot kernel that the
`ne==1`-vector single-token decode hits, so the result drifts enough to flip the
greedy speech-token sampler (chatterbox T3: divergence at step ~2 → repetition
collapse). Same failure class as the §211 native-batched-quant-CFM garbage.
**Rule:** B=2 batching of quantized matmuls is safe on the CPU backend (all
quants) and on Metal only with F16 weights. **Fix for Metal+quant:** dequantize
the matmul weights q*→F16 GPU-resident once at first use (host `to_float` →
`ggml_fp32_to_fp16_row` → upload to the GPU backend; `chatterbox.cpp
ensure_t3_b2_f16_weights`, `chatterbox_s3gen.cpp dequant_cfm_f16`) and batch
against the F16 copies — the `mul_mm_f16` path is correct batched on Metal. This
is the **only** other-backend solution in the tree: every non-batched TTS backend
(dia/zonos/tada/cosyvoice3/f5/voxcpm2) instead runs two sequential B=1 passes and
blends post-hoc, sidestepping the issue at a latency cost; s3gen's CFM and (now)
chatterbox T3 are the only two that batch CFG, and both dequant-to-F16 on GPU.
Corollary win: a per-step-rebuilt B=2 graph **sidesteps the §186 Lk-bucket
`buffer is nil` Metal crash**, making B=2 the first working T3-on-GPU path.

## TTS WAV md5 is only a valid parity gate with a pinned seed

Chatterbox's S3Gen flow-matching prior is RNG-seeded (default seed 0). Two runs
with identical T3 tokens but different `CRISPASR_CHATTERBOX_SEED` produce
different WAVs — so a WAV md5 diff across seeds is a **false** parity failure. For
greedy T3 parity (B=2 vs legacy) either pin `CRISPASR_CHATTERBOX_SEED` for both
runs, or — better, because it's seed-independent — compare the **speech-token
sequence** directly (`CRISPASR_CHATTERBOX_TEMP=0`, dump steps + EOS + count;
`CRISPASR_CHATTERBOX_DUMP_LOGITS_AT=9e9` also bypasses the bucket for a
like-for-like legacy graph).

---

## Kaggle: datasets are per-account, and the ccache dataset has a required shape (§213)

Two traps cost a couple of cold ~25 min CUDA builds before they were caught:

1. **Datasets can't be cross-used between accounts.** A kernel only attaches
   datasets owned by the account it runs under — Kaggle silently drops a
   cross-account `dataset_sources` entry (*"not valid dataset sources"*) and
   pushes the kernel **without** it. So `chr1s4/crispasr-ccache` cannot attach to
   a `chr1str` kernel. The fix is a per-account **copy** of every dataset (both
   accounts already keep their own `crispasr-hf-token`); make `crispasr-ccache`
   exist under both too. (Don't switch the kernel's account to chase a dataset.)

2. **The ccache dataset must be a `ccache.tar` (a tar of `.ccache/`) or a literal
   `.ccache/` directory** at the mount root — that's all `kaggle_harness`
   `_warm_ccache_from_dataset()` looks for. Both `crispasr-ccache` datasets had
   the cache **shards (`0`–`f`) at the dataset root** instead, so the harness
   printed `ccache: no seed dataset found (cold build)` and **every build had
   been cold** (nobody noticed — cold builds still finish, just ~10 min slower).
   Re-pack as `cd /kaggle/working && tar cf ccache.tar .ccache/` and
   `datasets version -r tar` under **each** account's token. To refresh, pull a
   completed build kernel's `/kaggle/working/.ccache` (or its `ccache.tar`) and
   re-version both copies.

Also: Kaggle GPU workers have ~20 GB RAM (the older "13 GB" note is stale), so a
3B PyTorch reference fits in F32 on CPU even on a P100 — but the P100 is sm_60,
which modern torch wheels don't support (`cuda.is_available()` returns True yet
every GPU kernel fails), so gate GPU use on `get_device_capability()[0] >= 7`.

## DRY shared headers can exist for months before callers migrate (§175)

Both `src/core/lang_names.h` (§175 item 1) and `src/core/bpe.h` (§175 item 2)
were already written, tested, and merged — but the 10+ call sites still used
local copies. The shared header *existing* is not the same as the migration
*happening*. When creating a shared helper, migrate at least one caller in the
same commit to establish the pattern. Otherwise the old copies persist
indefinitely and the DRY benefit is zero until someone does the mechanical
search-and-replace pass later.

---

## Cross-attention KV is a free F16 win — encoder-decoder backends (§176i)

Cross-attention K/V in encoder-decoder models is projected once from the
encoder output and then read-only for all decode steps. Unlike self-attention
KV (which accumulates per-step and where quantization noise compounds), cross-KV
is write-once/read-many — F16 halves memory with zero measurable quality impact.

**Pattern:** change the allocation from `GGML_TYPE_F32` to `GGML_TYPE_F16`, then
convert the F32 projection output to F16 before writing:
```cpp
std::vector<ggml_fp16_t> buf16(n_elem);
ggml_backend_tensor_get(graph_output, f32_buf.data(), 0, n_elem * sizeof(float));
ggml_fp32_to_fp16_row(f32_buf.data(), buf16.data(), (int)n_elem);
ggml_backend_tensor_set(cross_kv_tensor, buf16.data(), 0, n_elem * sizeof(ggml_fp16_t));
```

If the backend uses `ggml_cpy` to write into the cross-KV tensor (like speecht5),
just changing the tensor type is enough — `ggml_cpy` auto-converts F32→F16.

Downstream consumers (`ggml_mul_mat`, `ggml_flash_attn_ext`) handle F16 inputs
natively. For backends that feed cross-KV through graph *input* tensors each step
(dia, parler legacy path), the input tensor type must also change to F16 and the
staging vectors to `std::vector<ggml_fp16_t>`.

Applied to 5 backends (m2m100, t5, speecht5, dia, parler) in one pass.

---

## Localizing a GPU miscompute with CPU-vs-Metal diffing — the ggml_backend_sched weight-less-first-op trap (§206)

lfm2-audio transcribed garbage on Metal but was bit-correct on CPU. Technique to
localize a "works on CPU, garbage on GPU" backend, entirely on an M1 (no CUDA):

1. **Run the same input on `--no-gpu` and `--gpu-backend metal`.** CPU is your
   ground truth. If CPU is correct, every divergence is a GPU bug.
2. **Per-stage cosine dumps.** Env-gate a dump that marks intermediate tensors
   `ggml_set_output` + `ggml_backend_tensor_get` into per-stage `.f32` files for
   both backends, then compute cosine. The first stage with cos≪1 is the suspect.
   Verify the *input* to that stage matches first (cos≈1) — otherwise the bug is
   upstream (here the embeddings matched, the first backbone layer didn't).
3. **`GGML_SCHED_DEBUG=2` to see backend placement.** It prints every node's
   assigned backend and the split points. This distinguishes a *cross-backend
   copy* bug (ops ping-pong CPU↔GPU) from a *single-backend miscompute* (the
   whole subgraph is one Metal split, yet wrong). lfm2's backbone was entirely on
   Metal with no splits → a real ggml-Metal op bug, not a copy issue.

**Caveat on the dump itself — `set_output` snapshots LIE under the Metal sched.**
This cost the most time. Marking intermediate tensors `ggml_set_output` +
`tensor_get` for a per-stage dump is *unreliable* on GPU: the snapshots read
cos≈1.0 vs the reference (looking perfect) even when the real forward is garbage,
and they flip between values when you change the sched (shared vs fresh-per-call
→ all-zeros). Do **not** trust per-intermediate snapshots to localize a GPU bug.

**Reliable bisection: truncate the graph and diff the genuine OUTPUT.** Add an
env (`LFM2_AO_MAX_LAYERS=N`) that stops the backbone after N layers and returns
that as the real graph output (the thing the harness already reads), then compare
CPU-vs-GPU of *that output* for N=1,2,4,…. The first N that diverges is the
buggy layer — and within a layer, add probe env-gates that `return` early at each
sub-step (after-norm, after-operator, after-residual) so the early-return value
*is* the output, not a snapshot. This pinned lfm2's divergence to the very first
layer and showed it tracks **weight-reading ops** (the norm-weight broadcast
`ggml_mul` and Q5_K `mul_mat`s) while weightless `ggml_rms_norm` matches exactly.
What this *ruled out* (each a real experiment): the im2col depthwise conv (an
explicit shift-add gave identical divergence), the shared scheduler, weight
placement, and every op in isolation (rms_norm, broadcast-mul, Q5_K mul_mat,
even a full short_conv replica — all cos≈1.0 on Metal).

**ROOT CAUSE — `ggml_backend_sched` + a weight-less first op.** `GGML_SCHED_DEBUG=2`
showed the input `ao_in [CPU]` and `node_0 (RMS_NORM) [CPU]`, then a `MTL0#ao_in#0`
copy feeding the next op on Metal. The backbone's leading op is a *weight-less*
RMSNorm, so the scheduler (op_offload=false) placed it AND the leaf input on the
CPU backend, then inserted a CPU→GPU copy of the activation — and on this ggml
revision that cross-backend copy is miscomputed, corrupting the whole backbone.
The encoder/adapter were fine because their first op uses a weight, keeping the
input on the GPU. The standalone op tests falsely passed because the sched ran
the weight-less ops on CPU for *both* the CPU and Metal runs (so they trivially
matched) — **always confirm with `GGML_SCHED_DEBUG` which backend each op actually
ran on; a "Metal" sched run may be computing on CPU.**

**FIX:** when a whole subgraph is supported on the active backend, compute it
*directly* on `ctx->backend` with `ggml_gallocr` + `ggml_backend_graph_compute`
instead of `ggml_backend_sched`. A single-backend graph never inserts the broken
cross-backend copy. (Pinning the input with `set_tensor_backend` or op_offload=true
did NOT help — the copy persisted; only avoiding the split entirely worked.) Use
the published PyTorch ref (`cstr/crispasr-regression-fixtures`
`lfm2-audio-1.5b/jfk_11s/ref.gguf`) + `CRISPASR_DIFF_USE_GPU=1` to re-validate.

**The same root cause hit KugelAudio (§209)** — its Qwen2.5-7B LM also leads with
a weight-less RMSNorm, so the sweep's "0 bytes after 322 s" was garbage logits →
the constrained decode never emitting the speech-diffusion token. Same gallocr
fix. **Per-graph decision when one graph has a Metal-unsupported op:** KugelAudio's
VAE decoder uses `ggml_pad` (causal left-pad), which Metal rejects and a
single-backend gallocr graph can't fall back for. So mix: run the
weight-less-first-op graphs (LM/diffusion) on gallocr (the §206 fix) but keep the
VAE on `ggml_backend_sched` (which falls back to CPU for PAD on Metal; CUDA
supports PAD so it runs fully on GPU). That's safe because the VAE's first op is a
conv (input lands on GPU) — only the *leaf-input + weight-less-first-op* combo
triggers the broken copy; mid-graph CPU splits are fine. (Trying to make the pad
Metal-native via `ggml_concat`-of-a-scaled-view fails when pad>T — you can't build
zeros wider than the source view — so keep `ggml_pad_ext` and let the sched handle
it.)

**When you can't crack it: default to CPU.** If the GPU path is broken end-to-end
and CPU is correct + fast enough, force the backend to CPU
(`params.use_gpu = false`) behind an env opt-in (`CRISPASR_<X>_GPU=1`) rather than
shipping garbage. On a CUDA box this runs on CPU → no crash, correct output, sweep
passes — and the env flag preserves a path to debug the GPU bug later. Honest and
reversible beats a fast wrong answer.

## A glibc-only crash reproduces on macOS under AddressSanitizer — and the bug is often a non-power-of-two FFT (§205)

A backend that crashes on a Linux/CUDA host with `free(): corrupted unsorted
chunks` (or `malloc(): ...`) but runs fine on macOS is almost always a **latent
heap buffer overflow**, not a platform/kernel difference: glibc's allocator
traps the overwrite of an adjacent chunk header that macOS's allocator silently
tolerates. You do **not** need the Linux box to find it — build the failing
binary on M1 with `-fsanitize=address` and run the exact failing command. ASan
instruments every allocation regardless of platform, so it reports the same
overflow with a precise stack. (`libgmalloc` via `DYLD_INSERT_LIBRARIES` is a
fallback, but its 16-byte-slack mode misses small overruns and it can't see
writes that stay within a `std::vector`'s `capacity()`; ASan's container
annotations can — prefer ASan.)

The specific trap that bit chatterbox: **`fft_radix2_inplace` is power-of-two
only**, but several mel front-ends pass a non-pow2 `n_fft` — VoiceEncoder and
S3Tokenizer use 400, CAMPPlus 1920, CosyVoice3 400. For N=400 the `len=256`
butterfly stage writes `re[400..511]`, 112 floats past the N-element scratch.
Whisper-style n_fft=400 needs a **true 400-point transform** (the mel
filterbank is built for `N/2+1` bins), so the fix is a mixed-radix real FFT
(radix-2 DIT split down to an odd-N naive DFT base case — like `voxtral_fft`),
gated to non-pow2 N so the pow2 callers (granite 512, indextts 1024) stay
bit-identical. Audit rule: **any `core_mel::compute` caller passing
`fft_radix2_wrapper` must use a power-of-two `p.n_fft`** — grep the call sites
when adding a backend. See [[project_tts_gpu_cuda_crash_patterns]].

## Metal's q8_0 mat-vec kernel requantizes activations to q8 and ignores the F32 prec hint (§205)

Chatterbox's S3Gen CFM (flow-matching UNet1D) produces **all-NaN mel on M1
Metal with q8 weights** but is clean with F16 — and the temptation is to call it
"compound F16 accumulation." **It is not** (that was a wrong first guess; a NaN
is a specific event, so instrument and find it). The real mechanism, confirmed
by per-step dumps + the compiled Metal pipelines:

- **It is single-pass, not compounding.** `CRISPASR_S3GEN_DUMP=1` prints the
  per-Euler-step `x_rms`: the **first** step is already `nan` (`CFM step 1/10
  x_rms=nan`). The 10-step solver doesn't accumulate to NaN — one UNet forward
  emits it.
- **It is not a backend-split-count effect.** F16 (clean) and q8 (NaN) both
  report `n_splits=31` for the same b2 graph.
- **It is the q8 Metal mul_mat, specifically its mat-vec path.** The q8 run
  compiles `kernel_quantize_q8_0_f32` + `kernel_mul_mv_q8_0_q8_0` — i.e. Metal
  **requantizes the F32 activation to q8_0 and does a q8×q8 GEMV**, throwing away
  the activation precision and ignoring the `GGML_PREC_F32` that `mul_mat_hp()`
  set. The F16 run compiles only `kernel_mul_mm_f16_f32_hp` (f16×f32, F32 accum,
  no activation requant). So: **q8-GPU is wrong; F16-GPU and q8-CPU are correct**
  — because only the q8-GPU path requantizes activations.
- **The batch shape decides NaN vs garbage.** The default cond+uncond **batch=2
  fused CFG graph** (`build_graph_unet1d_b2`) NaNs at step 1; forcing the
  **batch=1** single-pass path (`CRISPASR_S3GEN_UNET_CFG_SINGLE=1`) on the same
  q8-GPU weights is *finite but unintelligible* (`conv_pre rms` ~2× ref, ASR
  transcribes to nothing). Same root cause (activation requant), different blowup.

Bisecting per-op (`CRISPASR_S3GEN_UNET_PIN_CPU_OP` / `..._KEEP_GPU_OP`) is a dead
end here precisely because the culprit is the q8 *mul_mat* path, not an
activation op (`flash_attn_ext`/`mish` were individually exonerated). Fix:
detect a quantized CFM (peek GGUF tensor types for `s3.fd.*` via
`open_metadata`) on Metal builds, and **dequantise those weights to F16 at load,
GPU-resident** (`dequant_cfm_f16`) — replace each quantized `s3.fd.*` tensor with
an F16 GPU copy so the CFM takes the correct `mul_mm_f16_f32_hp` path instead of
the activation-requantising q8 GEMV, keeping full GPU speed. This reproduces the
CPU-dequant result bit-for-bit (`conv_pre rms` 3.380 vs 3.379, identical ASR) on
the default batch=2 GPU graph. `CRISPASR_S3GEN_UNET_CPU=1` is the slower
all-CPU fallback. CUDA is unaffected (PLAN #83 validated GPU+`GGML_PREC_F32` to
cos 1.0 on P100). Lessons: (1) never diagnose a NaN by adjective — print the
first NaN and the compiled kernels; (2) `GGML_PREC_F32` does **not** stop
activation requantisation in Metal's q8 mat-vec kernel, so quantized weights on
Metal can be silently wrong for precision-sensitive iterative solvers, not just
slightly drifted — and the cheap fix is to keep the *weights* off the q8 kernel
(dequantise to F16 at load) rather than move the whole subgraph to CPU.

---

## A byte-identical standalone reproducer is the only way to (dis)prove a "ggml bug" (§203)

When a batched/fused graph misbehaves and you suspect ggml's allocator or a
kernel, build a tiny standalone that links the **same** `libggml` dylibs and
constructs the **same graph**, then diff the two graphs node-by-node. F5's
batch-CFG (B=2) corrupted batch-1 (batch-0 stayed bit-exact); after a dozen
in-tree probes pointed vaguely at "gallocr packing", a standalone with the
**byte-identical 979-node graph** (verified via `CRISPASR_F5_DUMP_GRAPH` vs the
repro's dump — same op/ne/contiguity/view per node) computed batch-1 **perfectly**
across every variation (full scale, F16 weights, double `alloc_graph`, weight σ
0.1–3.0, shared-backend sched). That single result **exonerated ggml entirely** —
the bug had to be runtime-specific to the backend (values / surrounding state),
not the graph. Corollaries:

- **"Memory layout changes the result" ≠ "memory overlap".** Two independent
  checks beat speculation: an in-allocator overlap detector (live-set keyed by
  alloc-ptr + chunk + offset, **cleared on `dyn_tallocr` reset** or you get
  cross-graph false positives) and **NaN-poisoning every node buffer before
  compute**. If poison yields zero output-NaNs and the detector finds no overlap,
  the buffer is clean and you're chasing the wrong thing — stop blaming the
  allocator. F5: both came back clean; batch-1 was fully written but wrong.
- **`ggml_set_output` as a "fix" is a tell, not a fix.** It only perturbs the
  allocation; if protecting one tensor fixes it but protecting more re-breaks it,
  you're shuffling a victim around a layout-sensitive read, not fixing a cause.
- **Bisect toward the working repro, not away from it.** Once the standalone is
  clean, transform the real backend *toward* it (strip features until the bug
  vanishes) rather than piling more onto the repro — the gap that remains is the
  cause. (Not done for F5; ~1.2× payoff didn't justify it. See HISTORY §203,
  `repro_batch.cpp`.)

## In-place device-KV decode graphs on Metal: three hazards (§176b+c, §203)

Porting the Lk-bucketed device-resident KV decode (write KV in place via
`ggml_set_rows`, read a fixed-Lk window, reuse a cached graph per bucket)
to Parler surfaced three Metal-specific hazards. All three "compile and
sometimes pass," which makes them nasty — they each cost a debugging round.

1. **Write→read ordering needs an explicit edge.** Reading the KV window
   from the bare cache tensor (`ggml_cont(ggml_view_2d(c->kv, …))`) has *no*
   data-dependency on the `ggml_set_rows` that just wrote the current row.
   Metal runs independent graph nodes concurrently, so the read races the
   write — build-dependent garbage (looked deterministic within a build,
   flipped 0↔1280 token mismatches across rebuilds). Fix: read from the
   `set_rows` **result** tensor (`set_rows` records the dest as `src[2]`, so
   viewing its result creates the edge). CPU never showed this (serial).

2. **Don't reuse a sched allocation across `graph_compute`.** The vibevoice
   §201 pattern (alloc once on a dedicated sched, skip realloc on the same
   bucket) faults on Metal here (`setBuffer_impl` on a freed AGXBuffer) for
   graphs that write external buffers. The portable idiom (kyutai §176s) is
   reset+`alloc_graph` every step, reusing only the *built graph topology*.
   Use a **dedicated** small sched for the step graph: reset+alloc on it is
   cheap, whereas re-planning the large `ctx->sched` (full model graph) each
   step is slower than the legacy path.

3. **Rebuild cached step graphs per utterance.** Reusing call-N's cached
   bucket graphs on call N+1 leaves the graph's compute tensors pointing at
   the previous call's freed sched allocation; the next
   `ggml_backend_tensor_set` memmoves into freed memory → SIGSEGV on the
   *second* synthesize. Invalidate (free + rebuild) at the start of each
   call; within a call the per-step reuse is preserved.

**§215 addendum — orpheus refines hazard #2: a *fresh* dedicated step-sched
fails to back input copies on GPU; use the warm prefill sched.** Orpheus
shipped the same bucket on its own dedicated `ar_step_sched` and SIGSEGV'd on
the *first* generated token (`AGXMetal setBuffer_impl`). Two diagnostic traps:
(a) `GGML_SCHED_DEBUG=2` showing the graph-input copies (`MTL0#inputs_embeds#0`,
`positions`, `causal_mask`) as `[ NULL ]` is a **red herring** — that print is
*pre-allocation* state and the working multi-token prefill graph shows the exact
same NULLs. (b) `lldb` put the real fault in `ggml_metal_op_norm` on **node #0**
(the first RMS_NORM reading `inputs_embeds`) — i.e. *before* any set_rows/rope,
so it's the input copy itself that's unbacked, not the KV path. The decisive
A/B: routing the *identical* bucket graph through the already-warm prefill sched
(`c->sched`, reserved by the multi-token prefill) backs the copies and produces
a correct WAV, while the *cold* dedicated sched's first `alloc_graph` (which
takes ggml-alloc's `reserve_n` path) does not. Fix: run the cached bucket graph
on `c->sched` with reset+alloc each step (the per-step re-plan is cheap because
the bucket topology is invariant → galloc fast-path; the §176s "don't re-plan
the big sched" worry doesn't bite when the bucket *is* the graph). So hazard #2's
"use a dedicated small sched" is not universal: parler's dedicated sched happens
to work, orpheus's didn't — when a step-sched faults on its first GPU
alloc+compute, reuse the warm prefill sched before chasing the NULL-copy print.
On M1 the orpheus bucket is correct but ~30% *slower* than non-bucket (fixed-Lk
over-read, same as parler), so it stays opt-in (`CRISPASR_ORPHEUS_BUCKET=1`).

Also: fixed-Lk bucketing shifts FP rounding (the padded reduction), so it is
not *guaranteed* bit-exact vs a dynamic-length path — but in practice the
shift is tiny. Validated with `crispasr-diff` against the F32 PyTorch ground
truth: at **F16 the bucket matched legacy AND ground truth 108/108 (100%)**
over the gen_codes_20 window; the divergence only appears at **Q4_K**, where
quantization noise (Q4_K is below parler's quality bar — legacy Q4_K greedy
also degenerates to non-speech) gets amplified by greedy argmax. So validate
such opts with the diff harness at F16, not ASR-roundtrip on a low quant.

On **unified memory (M1)** the device-KV migration itself wins little
(host↔device is a memcpy), so the first cut was *slower* than legacy at long
sequences — the fixed-Lk path's overhead outweighed the (nil) transfer
saving. What made it a clear win (~1.2–1.9× at 600 steps) was eliminating
that overhead: (a) **don't `ggml_cont` the Lk KV window** — it's the
contiguous leading block of the layer slice, so feed the view straight into
reshape; the cont is a D×Lk copy *per layer per step* that grows with the
bucket and dominates long utterances. (b) **reuse the sched allocation**
across steps (reset+alloc only on a bucket switch). So the lesson isn't
"device-KV is CUDA-only" — it's "the bucket's per-step *copies and
re-planning* are the real cost on any backend; kill those and it wins even
on unified memory." Reuse across `graph_compute` is fine once reads carry an
explicit write→read edge (read the `set_rows` result) and cached graphs are
rebuilt per utterance.

## Backend-name guards must match the *registered* name, by prefix when aliased (#171, #174)

A backend's CLI factory alias and the string `name()` returns are not the
same namespace, and policy code keyed on `name()` keeps getting them wrong:

- `qwen3-asr` (lang-sweep test plan) vs registered `qwen3`.
- `granite-speech` vs registered `granite`.
- `crispasr_tts_plan_chunks_for_backend()` skipped chunking for the literal
  `"vibevoice"`, but the TTS backends register as `vibevoice-tts` /
  `vibevoice-1.5b` / `vibevoice-tts-base` (bare `vibevoice` is the ASR
  backend). Result: the server sentence-split every VibeVoice TTS request
  while the CLI synthesised whole — a silent CLI/server divergence that
  manifested as garbled multi-voice audio and a baffling "same input, 30 vs
  100 chars" report (#171).

Rules:
1. Key policy on the value `backend->name()` actually returns at runtime, not
   on a factory alias you remember typing. Print it and check.
2. When a family has variants (`vibevoice`, `vibevoice-tts`,
   `vibevoice-1.5b`, …) match by **prefix** (`name.rfind("vibevoice", 0) ==
   0`), not equality — new variants then inherit the policy automatically.
3. The test must use the **production** name. The #171 unit test asserted on
   the bare `"vibevoice"` that production never passes, so it stayed green
   through the whole regression. A test that can't see the bug is worse than
   no test — it gives false confidence.

Companion to the "session ABI reimplements the CLI" learning: both are "the
CLI and its other consumers (server/bindings) silently diverge" bugs.

## Never pass `ggml_graph_get_tensor` directly to `ggml_backend_tensor_set` (#164)

`ggml_graph_get_tensor` returns NULL when a tensor name isn't found in the
graph's leafs/nodes. Passing that NULL directly to `ggml_backend_tensor_set`
triggers `GGML_ASSERT(tensor)` → SIGABRT. This is silent and hard to diagnose
because the crash gives only a file:line in ggml-backend.cpp, not the tensor
name or call site. Two rules:

1. **Always null-check** the return of `ggml_graph_get_tensor` before passing
   it to any `ggml_backend_tensor_*` function. Log the missing tensor name
   and return zeros or fall back to the CPU path.
2. **Shape input tensors to avoid broadcast.** `ggml_add(a, b)` where
   `b->ne[0] != a->ne[0]` (e.g. a 1-element scalar broadcast) works on CPU
   but SIGABRTs on some CUDA/Vulkan backends. Use `b->ne[0] == a->ne[0]` and
   fill the buffer. The voxcpm2 `fsq_half` 1-element tensor was this exact bug.

Both patterns existed in 13 call sites across 5 graph functions (tslm, ralm,
locenc, locdit, vae_decode) and were only caught when an unrelated graph
topology change made a name lookup fail.

## `rope_theta=0` means "no positional encoding" — skip RoPE entirely (#164)

Some sub-models (VoxCPM2 RALM) intentionally set `rope_theta=0` to disable
positional encoding. `ggml_rope_ext` computes `powf(theta, -2k/d)` →
`powf(0, negative) = inf` → NaN cascade. The NaN was latent (hidden by the
CPU stop fallback) until the RoPE skip was needed to fix the graph path. If a
model's attention has `rope_theta <= 0`, gate the RoPE call; don't pass zero
through and hope the math works out.

## `ggml_siglu` ≠ PyTorch `glu` — the gate/value halves are SWAPPED (#81)

`ggml_siglu(x)` computes `sigmoid(first_half) * second_half`.
PyTorch `F.glu(x, dim)` computes `first_half * sigmoid(second_half)`.
These are NOT the same — the gate and value are reversed. Use
`ggml_siglu_swapped` to match PyTorch's `glu`.

This one-line bug caused the nemotron conformer conv module to produce
completely wrong output in every layer, making the entire encoder diverge
from NeMo despite matching pre-encode, mel, attention, and FFN stages.
It took 3 days of debugging (mel comparison, per-layer dumps, per-sub-
module hooks, Kaggle NeMo ground truth) to isolate because the global
encoder statistics (min/max) happened to look plausible — only the
per-frame values were wrong.

**Rule:** When porting a GLU activation, always verify which half is the
gate and which is the value. `ggml_siglu` = `σ(a) × b` (swapped from the
PyTorch convention). `ggml_siglu_swapped` = `a × σ(b)` (matches PyTorch).

## Regression transcripts need WER tolerance, not byte-exact match (#92)

The first nightly CI regression run failed on 4 backends with minor
punctuation/decode differences across platforms (GH runner vs Kaggle T4
vs VPS CPU). Example: `"for you, ask"` vs `"for you; ask"` — same words,
different punctuation. These are decoder tie-flips, not real regressions.

**Rule:** Pin `expected_transcript` from a single authoritative platform,
but add `transcript_tolerance: {cer_max: 0.02, wer_max: 0.05}` for any
backend whose decoder is stochastic or punctuation-sensitive. The
`run_one.py` driver computes Levenshtein CER/WER after case+punct
normalization and asserts against the thresholds. Only use byte-exact
match for backends with fully deterministic output (CTC, greedy argmax).

## Kaggle regression kernels: write output incrementally, crash-guard everything

Kaggle kernels fail silently — the only output you get is `/kaggle/working` files.
If the script crashes before writing anything, you get zero diagnostics. Rules:

1. Write a `transcripts.json` crash marker at line 1 (before any imports that
   might fail).
2. Wrap the entire body in `main()` + `try/except` that always writes the
   partial results dict to `transcripts.json`.
3. After each backend, write incremental results AND delete the model file
   (Kaggle scratch is ~20 GB; models are 500 MB-5 GB each).
4. Use `progress.txt` as a human-readable append-only log.
5. Stub out `kaggle_harness` imports — the harness can have import-time
   crashes from missing deps; the kernel must survive without it.
6. CPU-only workers may not have internet even with `enable_internet: true`.
   GPU workers always get internet. Use GPU for any kernel that needs git
   clone or pip install.

## Streaming conv modules need cached left context, not zero-padding (#81)

NeMo's `CausalConv1D` in streaming mode does NOT zero-pad the left side.
Instead, it prepends the last K-1 frames of the previous chunk's pre-DW-conv
signal (`cache_last_time`). Zero-padding every chunk means the depthwise conv
sees 8 zeros as left context on every chunk boundary, which destroys the
activations from the second chunk onwards.

**Symptom:** streaming encoder aggregate stats (min/max/mean) looked almost
identical to the full-sequence encoder, but per-frame values diverged from
frame 10 onwards (magnitudes ~10x smaller). RNNT decoder produced all-blank
tokens.

**Rule:** Any streaming conformer conv module must cache the pre-DW-conv
signal (post pointwise1+GLU, pre depthwise conv). First chunk zero-pads;
all subsequent chunks prepend the cache. This is separate from the
`cache_last_channel` (post-FFN1) cache used for attention K/V context.

## Asymmetric rel-pos shift uses the same formula as symmetric (#81)

For streaming attention with Q(T_new) × K(T_full), the rel_shift is:
`view_3d(BD_raw, T_full, T_new, H, s1-s0, s2, (T_new-1)*s0)` — exactly
the same stride-trick as the symmetric case, but using T_new for the
offset instead of T. Derivation: BD[k,q] = BD_raw[(T_new-1)+k-q, q].

## A feature has ~8 front-ends — wiring it into one isn't "done" (§166)

A user-facing option in this repo must be threaded through every surface or it
silently drifts: the CLI (`crispasr_run.cpp`), the HTTP server
(`crispasr_server.cpp` — per-request form params *and* resident startup state),
the C-ABI (`crispasr_c_api.cpp` + `crispasr_session.h`), the six native wrappers
(Python ctypes, Go cgo, **Rust at the repo root** `crispasr-sys`/`crispasr` — not
under `bindings/`, Dart FFI, Java JNA, Ruby C-ext), the WASM/JS Embind binding,
and the legacy Node addon. `--punc-model` started life CLI-only; closing the gap
meant ~8 edits + a test per surface. Lessons that paid off:
- **Share the resolution, don't copy it.** The `--punc-model` alias→model table
  lives in one pure header (`src/crispasr_punc_model.h`) included by CLI, server,
  and C-ABI; the truecase loader likewise. Copies drift the moment a backend or
  URL changes.
- **Audit by enumerating surfaces, then diff each against the C-ABI** (the source
  of truth) — that's how the per-wrapper gaps (set_hotwords/set_g2p_dict/
  set_punc_model/set_speaker_id missing in various wrappers) and the threadbare
  Node addon were found.
- **Verify per surface with the toolchain that exists** (cargo, go build, dart
  analyze, javac+JNA, ruby `cc -fsyntax-only`, ctypes import). When a toolchain
  is absent (emcc, cmake-js) say so and fall back to inspection — don't claim
  green you didn't see.
- **Resident vs per-request** is its own axis on the server: post-processors
  (punc/truecase) and detectors (LID/VAD) should load once at startup and stay
  resident, not reload per request (#165).

## A dry-run "preview" must mirror the real resolver, or it lies (§166)

`--dry-run-resolve` had its own `build_preview()` that re-implemented model
resolution — and drifted: it skipped the literal-arg backend-key lookup the real
`crispasr_resolve_model` does, so `-m parakeet-tdt_ctc-110m --dry-run-resolve`
previewed the 0.6b default while an actual run loaded the correct 110m. Any
preview/plan/dry-run that re-derives behavior instead of calling the real code
path will eventually diverge from it. Prefer sharing the resolver; if you must
duplicate, pin the duplicate against the original with a test (here:
`tests/test-dry-run-resolve.sh`). Same lesson as the §166 punc-model resolver
being shared across CLI/server/C-ABI rather than copied three times.

## Graph input tensors that nothing consumes are invisible to `ggml_graph_get_tensor` (#164)

`ggml_set_input(t)` marks a tensor as a graph input, but if no operation in the
graph actually reads `t`, it's never added to the graph's leafs/nodes arrays.
`ggml_graph_get_tensor(gf, name)` only searches those arrays, so it returns NULL
for a "created but unused" input — even though the tensor exists in the ggml
context.

This bit VoxCPM2's RALM: `positions` was created and set as input, but after the
RoPE skip fix (`rope_theta=0`), no op consumed it (RoPE skipped + F32 KV cache
uses `ggml_cpy` not `ggml_set_rows`). The null-guard treated NULL positions as
fatal and returned all-zero hidden states → noise output. Fix: make the positions
tensor optional — if the graph doesn't need it, don't require it.

**Rule:** if a graph input is conditionally consumed (e.g. only used when RoPE is
active), the setter code must tolerate `ggml_graph_get_tensor` returning NULL for
it. Don't fail on a legitimately unused input.

## Orpheus/SNAC TTS: `token_embd` must stay F16 for sub-Q8 quants

Orpheus (Llama-3.2-3B with SNAC 24kHz codec) uses tied weights:
`talker.token_embd.weight` serves as both input embedding and LM head.
The SNAC codec emits peaked distributions over 4096 audio tokens —
quantization noise from Q4_K is enough to break the 7:1 super-frame
slot pattern and produce gibberish. The upstream repo's README says
"sub-Q8 quants tend to break the SNAC super-frame slot pattern."

**Fix:** `crispasr-quantize` now has an `is_orpheus` guard that keeps
`talker.token_embd.weight` at F16. Block projections (attn, FFN) quantize
safely to Q4_K. Same reasoning as qwen3-tts `talker.token_embd`, bark
`token_embd`, and LFM2's `lfm.embed_tokens` — tied/output weights on
peaked-distribution codecs are always quant-sensitive.

## `ggml_backend_cpu_set_n_threads` asserts CPU — guard it after `init_best()`

`ggml_backend_init_best()` returns the *best* backend (Metal on Apple Silicon,
CUDA on NVIDIA, CPU otherwise). Calling `ggml_backend_cpu_set_n_threads()` on the
result aborts with `GGML_ASSERT(ggml_backend_is_cpu(backend_cpu))` on any GPU
build — that setter is CPU-only. Always gate it: `if (ggml_backend_is_cpu(b))
ggml_backend_cpu_set_n_threads(b, n);`. This bit silero-LID (`--lid-backend
silero` aborted on every Metal/CUDA load, #165); it's an easy one to copy-paste
wrong into any new backend's `*_init`. The crash only shows on a GPU build, so a
CPU-only CI lane won't catch it. (Per-call model load + free also hides it as a
fresh abort each request rather than once at startup — see the resident-cache
work in the same fix.)

## Regenerate Go cgo LDFLAGS from a *linux-equivalent* config, never macOS

`tools/sync_go_cgo_ldflags.py` writes the SHARED `#cgo linux/darwin LDFLAGS`
line from whatever CMake graphviz dot it's given. Run on macOS, the dot includes
macOS-only ggml backends (`ggml-metal`, and `ggml-blas` via Accelerate even with
`-DGGML_METAL=OFF`), so they leak into the *shared* line — which then (a) fails
the CI `cgo-ldflags-drift` check (the linux runner's graph has neither) and (b)
breaks the linux link (`-lggml-metal`/`-lggml-blas` don't exist there). macOS gets
those from a separate `#cgo darwin LDFLAGS: -lggml-metal -lggml-blas` line, so the
shared line must match the LINUX set: `ggml-base`/`ggml-cpu`/`ggml` + the
platform-independent crispasr backends. To regenerate on macOS, configure with
`-DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_ACCELERATE=OFF -DGGML_CUDA=OFF` (what the
CI drift job's linux defaults produce) before running the sync, then confirm with
`--check --dot <that dot>`. (Cost me two wrong commits — metal, then blas.)

## cmake-js defaults its build dir to `build/` — point it elsewhere (§166)

Running `cmake-js compile` from the repo root reconfigured and clobbered the
ninja `build/` (cmake-js's default `--out` is `build/`), wiping libcrispasr.dylib
+ bin/crispasr that the wrappers and tests link against. Always pass an explicit
`-O <dir>` (and `-d <project-dir>`) so the Node-addon build lands in its own tree
(e.g. `build-addon/`). Recovery is just a fresh `cmake -B build` + rebuild
(ccache keeps it fast), but it's avoidable. The addon's own CMakeLists has no
`cmake_minimum_required`/`project()` — it's meant to be `add_subdirectory`'d from
the root under `CMAKE_JS_VERSION`, so cmake-js must run with `-d <root>`, not from
the addon dir.

## A WebSocket server that "works in the browser" can still be RFC-broken (§166)

ws_stream's handshake had the RFC 6455 magic GUID mistyped
(`...5AB5DC11D585` vs correct `...C5AB0DC85B11`), so `Sec-WebSocket-Accept` was
always wrong — yet it had shipped. Lesson: verify a WS handshake against a
*spec-compliant* client (the `websockets` lib, a real browser), not a hand-rolled
client that echoes your own (possibly wrong) accept. The fastest localization was
the canonical RFC vector: key `dGhlIHNhbXBsZSBub25jZQ==` must yield accept
`s3pPLMBiTxaQ9kYGzzhZRbK+xOo=`. Also: read the upgrade request in a loop until
`\r\n\r\n` — a single recv() truncates handshakes split across TCP segments
(works for a localhost one-shot client, fails for fragmented real ones). Both are
"passes my own test, fails everyone else's" bugs — diff against the reference
implementation, not a mirror of your own code.

## Server warmup is the launch-time differentiator vs the CLI (#165)

When "the CLI works but `--server` doesn't" on a given GPU/backend, suspect the
**warmup** first. The server runs a dummy transcribe right after model load
(`backend->warmup()` before `listen()`); the one-shot CLI does **not** warm up
unless `--warmup` is passed. So a driver bug that only bites the warmup's
input shape (e.g. parakeet's 0.5 s of silence) presents as "server hangs after
model init, CLI transcribes fine." Diagnose by where the log stops: no
`warmup completed in … ms` line ⇒ it died inside warmup. Fixes: an opt-out
(`--no-warmup`) and a try/catch around the warmup call. A hard GPU device-lost
won't be catchable, but the opt-out sidesteps it. Couldn't reproduce on
M1/MoltenVK — Vulkan-driver bugs are RADV/AMD/proprietary-specific, not portable.

## Don't throw on the per-request server path; surface the real error (#165)

Two cpp-httplib server traps found together:
- A route handler that **throws** becomes a bare 500 with an empty body.
  cpp-httplib then invokes the *error* handler, which (if it fills empty bodies)
  mislabels every exception as the generic "not found" 404 payload. Add
  `svr.set_exception_handler` to log `e.what()` and return a structured 500 — a
  thrown `std::filesystem_error` masquerading as "use POST /v1/..." cost real
  debugging time here.
- Anything on the per-request path that calls the **throwing**
  `std::filesystem::create_directories` (e.g. a scratch/cache dir helper) will
  500 *every* request when that dir is unwritable or odd (here: a dangling
  `~/.cache/crispasr` symlink to an unmounted disk). Use the `std::error_code`
  overload and fall back to `temp_directory_path()`.

## FFT size must match upstream exactly (Mini-Omni2 / Whisper mel)

Whisper's `torch.stft(n_fft=400)` runs a 400-point DFT. Zero-padding to
512 (next power of 2) changes the frequency bin spacing from `k*sr/400`
to `k*sr/512`. The mel filterbank expects bins at `k*40 Hz`; zero-padded
bins land at `k*31.25 Hz`. Result: cos_min≈0.8 on mel comparison. Fix:
use a 400-point DFT directly (O(N*N_freqs) is fast enough for N=400).

Related: `torch.hann_window(N)` (periodic) differs from
`np.hanning(N+1)[:-1]` (symmetric) by ~2.4e-7 per sample. Enough to
cause cos_min≈0.95 on mel. Store the upstream variant in the GGUF.

Also: Whisper drops the last STFT frame (`stft[..., :-1]`). core_mel
produces one extra frame. Truncate: `T_mel = n_samples / hop_length`.

## Multi-stream token architecture (Mini-Omni2)

Models that interleave text + audio codec streams (8 in mini-omni2) embed
each stream separately and average all 8 into a single hidden state per
timestep. Key pitfalls:

1. Audio features replace pad positions in **audio streams only** (0-6),
   NOT the text stream (7). The Python `concat_feat` does `for i in range(7)`.
2. Special task tokens at the end of each stream select the mode:
   `_asr=151940` for transcription, `_answer_t` for chat/S2S.
3. Decode step uses `snac_config.end_of_audio=4097` (pad_a), not `eoa=4096`.
4. LitGPT's fused QKV is interleaved per query group: `[Q0×7, K0, V0, Q1×7,
   K1, V1]`, not `[Q_all, K_all, V_all]`. Must deinterleave in converter.
5. Batch all 8 token embeddings in one `ggml_get_rows` call per decode step
   instead of 8 separate graph builds — ~4× speedup.

## Decompose ConvTranspose1d → mul_mat + col2im_1d; port col2im per-backend (#155)

The native `conv_transpose_1d` GPU kernel is slow (each output thread loops the
full input). Decomposing into `mul_mat(w_perm, x) + col2im_1d` hands the heavy
GEMM to the already-optimized backend matmul and leaves only a lightweight
gather — codec 1200→130ms on a 7900 XTX (#155). But the decomposition is only a
win where `col2im_1d` has a native kernel: CUDA/HIP, Metal, CPU do; **Vulkan did
not**, so the sched CPU-fell-back that op (a GPU↔CPU hop per call). When you
decompose an op, check every backend the op can land on — a missing kernel
silently degrades to a cross-backend bounce, not an error.

Porting a 1-src→1-dst op to Vulkan = shader `.comp` (mirror the CUDA/CPU math)
+ register in `vulkan-shaders-gen.cpp` + six wiring sites in `ggml-vulkan.cpp`:
push-const struct, `pipeline_*` decl, `ggml_vk_create_pipeline`, `get_pipeline`
case, dispatch `elements` case, the op fn + build-graph dispatch + `supports_op`.
Mirror `timestep_embedding` (single src, 2 buffers) or `conv_transpose_1d`.
Element count in the dispatch = thread count ÷ the pipeline's wg_denoms.

**Vulkan is testable on the M1 dev box via MoltenVK** — don't assume Vulkan
needs AMD/NVIDIA hardware. `brew install vulkan-loader molten-vk shaderc`,
point `VK_ICD_FILENAMES` at MoltenVK's ICD, build `-DGGML_VULKAN=ON`, and a
tiny `cpu_init()` vs `vk_init(0)` op harness gives a bit-exact diff. col2im_1d
came out maxabs=0 vs CPU across the codec shapes. (Setup recipe in auto-memory.)

## Measuring GPU perf on a thermally-throttling laptop GPU (RTX A1000)

The A1000 Laptop (4 GB) down-clocks under sustained load: a no-fix cohere beam
A/B climbed 15 → 36 s across 6 back-to-back rounds as the GPU heat-soaked.
Block-comparing (run all of A, then all of B) is worthless — whichever build
runs second is hotter and looks slower regardless of the code. Two rules that
gave stable #161 numbers on this box:

- **Interleave A,B,A,B and take the median *ratio* per round.** Each A/B pair
  runs at near-identical temperature, so the ratio is thermal-independent — it
  held at ~2.54× while the absolutes climbed 1.7 → 3.1× of the cool baseline.
- **A single cold first run reads as a regression** (the #81 WDDM-warm effect).
  Warm up first; report `min` for an absolute number, the interleaved ratio for
  an A/B. The #161 DEVICE-mode win (above) was only visible this way — a naive
  block A/B on this GPU would have buried it in thermal drift.

## Beam-search KV snapshots must stay on-device (#161)

`core_beam_decode::run_with_probs_branched` saves+restores the decoder KV cache
once per surviving beam per step. Doing that through host memory
(`tensor_get` → `std::vector` → `tensor_set`) is invisible on a unified-memory
backend (Metal/CPU) but on a discrete GPU it's a full KV round-trip over PCIe +
a blocking sync per copy — at the whisper-default `beam_size=5` it dominated
cohere wall time on CUDA (~10× regression, GH #161), entirely outside the per-op
profile while one core sat in a sync-spin.

- A per-op profile that's "unchanged" while wall time regresses points at
  host-side glue, not kernels. Add a `total − Σstages` residual probe
  (cohere's `UNACCOUNTED` line) — it localizes the gap in a single run. An A/B
  on `-bs 1` (greedy, no snapshots) confirms the beam path is the culprit.
- Use `core_attn::kv_snapshot_pool`. It blits **device-to-device** when the
  cache backend implements `cpy_tensor` (CUDA/Vulkan — `ggml_backend_buffer_copy_tensor`
  returns true) and falls back to **pooled host buffers** otherwise. The naive
  `ggml_backend_tensor_copy` is a trap on Metal: shared-buffer `cpy_tensor`
  returns false, so it silently `malloc`s a staging buffer and round-trips —
  *worse* than the original per-step `std::vector`.
- Pool the snapshot buffers. Allocating a backend buffer per snapshot
  (`cudaMalloc` / `MTLBuffer`) per beam per step is its own regression
  (`UNACCOUNTED` 705→2168 ms on Metal before pooling); only ~`beam_size+1` are
  ever live at once.
- `ggml_backend_buffer_is_host` does NOT distinguish Metal-unified from
  CUDA-discrete — Metal shared buffers report `is_host=false` yet are
  host-addressable. Probe actual device-to-device copy support, don't guess
  from `is_host`.

## HF Space is a separate, FLAT repo that silently drifts from `hf-space/`

The deployed Space (`huggingface.co/spaces/cstr/CrispASR`) is its own git
repo with files at the ROOT (`Dockerfile`, `app.py`, `start.sh`, …) — NOT a
checkout of this repo's `hf-space/` subdir; someone flattens + uploads.
Nothing keeps them in sync, so it bit-rots invisibly:

- The Space's Dockerfile `COPY`s `app.py` (root); ours `COPY hf-space/app.py`
  (repo-root build context). Don't upload our Dockerfile verbatim — flatten
  the two `COPY hf-space/...` lines.
- A rebuild clones the *latest* CrispASR `main` for the binary AND installs
  the *latest* Gradio (`gradio>=5,<6`), so a build/CLI/API change that merged
  weeks ago only breaks the Space at its *next* rebuild. Ours had been
  serving an old image and kept trying to build the removed `whisper-cli`
  target; the server binary is the `crispasr-cli` target (`crispasr-lib` is
  only the library). Gradio dropped `gr.Code(language="text")`.
- `gr.mount_gradio_app(fastapi_app, demo, "/")` + uvicorn lets the Space
  serve the OpenAI `/v1` API publicly alongside the UI (HF only routes the
  one `app_port`).

Verify a Space change by actually hitting `/v1/...` (200 + a real transcript)
— `stage=RUNNING` lies (HF serves 503/edge-CORS for an unhealthy space). See
HISTORY (2026-06-10 hf-space).

## VAD + chunking

### Caching a stateful context must serialize the *use*, not just the *lookup* (#132, May 2026)

The #132 fix caches one Silero `whisper_vad_context` across requests to
avoid a 70× init/free fragmentation regression. The first cut guarded the
cache with a mutex that only wrapped the *getter* — it returned the shared
pointer and released the lock, then the caller ran the detect unlocked.
That is the classic "lock the lookup, leak the use" mistake: the cached
object owns mutable per-call state (the Silero LSTM h/c buffer, the ggml
scheduler, and `probs`), all reset and rewritten by
`whisper_vad_segments_from_samples`. Two callers sharing it concurrently
race on that state.

It hid in testing because the CLI is single-threaded. It only bites the
**server**, which deliberately slices VAD *outside* `model_mutex`
(`crispasr_server.cpp`) and serves on an httplib thread pool — so the
cache mutex is the *only* thing serializing concurrent requests against
the one context. The fix is to hold the mutex across the whole detect
(getter renamed `*_locked`, caller takes the `lock_guard`). General rule:
when you replace per-request objects with one shared cached object, the
lock has to cover every span that touches the object's mutable state, not
just the handle handoff. Before caching, isolation was free (each request
had its own context); caching trades that for an explicit lock you must
size to the *usage*, not the *handle*.

### Independent-chunk TDT decode loses interior content, not just boundaries (issue #89, May 2026)

The 30 s auto-chunk fallback (`kLongAudioFallbackChunkSeconds`) prevents
the catastrophic zero-output failure on long audio (z-norm drift causes
the TDT decoder to emit blanks past ~60 s), but the auto path without
VAD tops out at ~82 % coverage. The missing content is **interior to
each chunk**, not at chunk boundaries.

Exhaustive sweep on the issue #89 reporter's 60 s Japanese audio
(parakeet-tdt-0.6b-ja, CPU):

| chunk | overlap | chars | coverage% | max_gap |
|-------|---------|-------|-----------|---------|
| 15s   | 3s      | 203   | 75.6%     | 8.5s    |
| 15s   | 5s      | 278   | 70.9%     | 5.3s    |
| 15s   | 8s      | 229   | 61.9%     | 12.2s   |
| 20s   | 3s      | 278   | 82.4%     | 3.4s    |  ← best chunk config
| 20s   | 5s      | 246   | 66.5%     | 10.0s   |
| 20s   | 8s      | 220   | 56.0%     | 12.1s   |
| 25s   | 3s      | 183   | 54.0%     | 15.1s   |
| 25s   | 5s      | 234   | 70.8%     | 12.4s   |
| 25s   | 8s      | 246   | 70.3%     | 9.8s    |
| 30s   | 3s      | 195   | 59.7%     | 19.4s   |
| 60s   | 0s      | 294   | 99.5%     | 0s      |  ← single pass (breaks on Vulkan/AMD)
| VAD   | silero  | 281   | 93.1%     | 3.7s    |

Counterintuitive findings:

1. **More overlap hurts.** 8 s overlap is worse than 3 s for *every*
   chunk size. Larger overlap extends the z-norm window, shifting feature
   statistics further from training distribution, and the boundary
   trimming removes more content that legitimately belongs to the chunk.

2. **The problem is decoder cold-start, not boundary stitching.** Each
   chunk reinitialises the TDT LSTM predictor to SOS. The decoder needs
   several seconds to "warm up" — emitting blanks or very sparse tokens
   while the LSTM state converges. This eats 5-20 s of interior content
   per chunk regardless of overlap.

3. **NeMo doesn't have this problem** because `FrameBatchASR` /
   `BatchedFrameASRTDT` keep the LSTM state between 1.6-4 s frame steps.
   The decoder never cold-starts after the first frame.

**Root cause:** CrispASR treats each chunk as an independent utterance
(`lstm_init_state → SOS`). NeMo treats them as a continuous stream with
shared decoder state. Fix: PLAN #104 (stateful frame-streaming TDT
decode).

**Practical recommendation until #104 ships:** use `--vad` (93 % coverage)
for long Japanese audio. The auto path is a safety net against zero
output, not a quality-optimal path.

#### Correction 2026-05-26 — the failure mode above describes the *chunked-decode* path, not the *streamed-decode* path

The section above describes `parakeet_transcribe_chunked` (independent
per-chunk TDT decodes, joined via LCS). The current default path on long
audio is `parakeet_transcribe_streamed` (independent per-chunk
*encodes*, concatenated encoder output, *one* TDT decode over the
concat). Streamed has no per-chunk decoder reset, so the "decoder
cold-start eats 5-20 s per chunk" mechanism does not apply there.

But streamed has its own per-chunk-context failure mode that
empirically looks identical from the outside — sparse output with
interior dropouts. Sweep on `audio_samples/{en,de}/fleurs_{60,300}s.wav`
+ `long-clips/yt_60s.wav` (2026-05-26, M1 Metal, v3 + ja models):

| model | audio | c=8 | c=15 | c=20 | c=30 | c=40 |
|---|---|---|---|---|---|---|
| v3 (vocab=8192) | EN 60s | 186 | 159 | 362 | 519 | **800** |
| v3 (vocab=8192) | EN 300s | 492 | 1034 | **1703** | 1549 | 1561 |
| v3 (vocab=8192) | DE 60s | 502 | **709** | 581 | 678 | 520 |
| v3 (vocab=8192) | DE 300s | 2496 | 2806 | 2993 | **3063** | 2944 |
| ja (vocab=3072) | JA 60s | **1674** | — | 907 | 507 | 363 |

**The c=8 default that ships well for the JA-only model collapses on
the v3 (multilingual / EN-focused) model.** EN 60s loses 77 % of content
vs the c=40 max. Across the four v3 cases, c=30 is the smoothest
default (≥ 88 % of best); c=8 is below 50 % on three of four.

**Root cause: encoder context, not decoder state.** With 8 s chunks the
Conformer's bidirectional attention only sees 8 s of context, but the
encoder was trained on segments closer to its training-distribution
window (~30 s). Even with global mel z-norm (which we have), small
chunks shift the per-feature statistics enough that the encoder
produces features the TDT decoder doesn't recognise as in-distribution,
and the TDT decoder emits sparse blank-heavy token paths.

**Fix (`e1904a1e`):** per-model chunk default in
`parakeet_transcribe_{streamed,chunked}`. `vocab_size < 4000` ⇒ JA-only
model ⇒ c=8 (preserved). Else ⇒ v3 / multilingual ⇒ c=30. Threshold of
4000 cleanly separates the two known variants (JA=3072, v3=8192). Env
override `CRISPASR_PARAKEET_STREAM_CHUNK=N` remains the escape hatch.

**Methodological lesson:** the chunked-vs-streamed distinction matters
when reasoning about which mechanism a regression hits. "TDT decoder
cold-start" was the right diagnosis for the chunked-decode path that
shipped at PLAN #104's start; it was the wrong diagnosis for the
streamed-decode path that shipped at issue #89's resolution. Confusing
the two cost a multi-hour pretend-debugging session in 2026-05-26
before re-running the chunk-size sweep clarified which path was active
and what was actually going wrong.

---

### Overlap-save context is for *chunks*, not for VAD slices (issue #114)

`cad4c28a feat(#89): overlap-save chunking with --chunk-overlap flag`
added a ± `chunk_overlap_seconds` (default 3.0) neighbour-audio
extension to every per-slice transcribe call when `slices.size() >
1`. That gate is correct for explicit `--chunk-seconds N` runs, where
each cut falls in the middle of continuous speech and the encoder
genuinely needs context to span the boundary. It is *wrong* for VAD-
derived multi-slice runs: VAD slices are separated by silence, so
there is no boundary signal to recover, and pulling 3 s of the next
utterance into the current encoder context shifts the per-feature
z-norm statistics + bidirectional self-attention enough to push the
TDT decoder onto a different (worse) token path. The visible
regression on parakeet-tdt-0.6b-ja was kanji compounds collapsing to
bare hiragana plus entire short slices being dropped.

The crispasr-diff harness did NOT catch this — the cos at mel /
encoder stayed at 1.0 / 0.999994 on the single-utterance baseball
fixture, because the bug only manifests once multiple VAD slices
sit in the same long-form audio with realistic inter-slice silence.

Fix: gate overlap-save on `effective_chunk_seconds > 0`. The gate
lives in `examples/cli/crispasr_chunk_context_gate.h` so a unit test
can pin the invariant without spinning up a model.

Signature of the bug: per-slice transcripts that look "almost
right" but consistently pick simpler (more conservative) tokens at
the same boundaries — kanji → hiragana, multi-word phrases →
chopped syllables. Single-shot diff against PyTorch reference looks
fine because the diff harness only feeds one continuous segment.

### Overlap-save also breaks timestamp-less LLM-decoder ASR (issue #218)

The overlap-save extension is trimmed back to the slice range by
**word-level filtering** — which needs per-word (or per-token)
timestamps. Autoregressive LLM-decoder ASR backends (cohere, qwen3,
granite, voxtral, kyutai-stt, glm-asr, gemma4-e2b, and — added in
#218 — **moss-transcribe**) emit plain text with no alignment, so the
trim falls through to segment-level filtering that keeps the *whole*
extended segment. The ±`chunk_overlap_seconds` of context is then
transcribed twice and **duplicates at every seam** (moss on a 145 s
clip: "…move much **of** the fence. Don't move much **to** the
fence."). Worse, feeding these models a 30 s+2×3 s buffer pushes them
further outside their trained window: on moss it lengthened the
greedy repeated-phrase loops (#218 part 1) — the same slice that
looped to the 512-token cap with context only looped ~50 tokens on the
bare 30 s slice. So for a timestamp-less decoder the extension is
net-negative on both axes; the fix is the `kBlocked` opt-out in
`crispasr_chunk_context_gate.h` (bare-slice path, no extension), which
its peers already used. Rule: only backends that emit trustworthy
word/token timestamps may take overlap-save context — everything else
belongs on the opt-out list.

---

## mel / preprocessor

### STFT frame count: NeMo `feat_len` vs raw STFT output

NeMo's `AudioToMelSpectrogramPreprocessor` computes the STFT over the
full center-padded signal, producing `T = floor((n+pad-n_fft)/hop)+1`
frames. But the returned `feat_len` is `floor(n_samples/hop)`, which
is one less when `n_samples` is an exact multiple of `hop_length`.
Frames beyond `feat_len` are treated as padding (zeroed in the output
tensor). The per-feature z-normalization runs over all `T` frames
(including the padding one), but the padding frame's z-score ends up
zero because it equals the mean of a near-constant band.

The C++ `core_mel::compute` produces the raw `T` frames and normalizes
all of them. The boundary frame is NOT zero — it has valid but
edge-contaminated STFT content, so after z-norm its values diverge by
±4 z-score units from the NeMo reference.

Fix: set `p.drop_last_frame = true` for all backends that use NeMo-
style center-padded STFT (canary, parakeet, cohere). This drops the
final frame, matching NeMo's `feat_len`.

Signature: mel cos_min ≈ 0.94 but cos_mean ≈ 0.999; exactly one
frame dominates `max_abs`; the worst frame is always the last.

### Dither must be disabled for deterministic reference dumps

NeMo's default `dither: 1e-5` adds per-sample Gaussian noise. Tiny
for real speech, but non-deterministic — each dump produces slightly
different mel values. Disable in the reference backend with
`model.preprocessor.featurizer.dither = 0.0` before running the
forward pass.

### Silence is a degenerate test input for per-feature-z normalization

All-zero audio → constant log-mel → per-feature z-norm divides 0/ε →
all-zero mel. NeMo produces a small negative constant (≈ −0.16) for
the same input because batch-norm running stats or ε handling differ.
The C++ producing all-zero mel is mathematically correct but creates
a misleading diff: cos_min=1.0 (zero vs constant is handled as
"perfect" by the cosine comparison's zero-guard). Always test with
real speech.

---

## ggml / inference engine

### RoPE mode mapping: ALWAYS `NEOX` for modern models

The single most expensive bug in this project was shipping Granite with
`GGML_ROPE_TYPE_NORMAL` (mode=0) when HF models use `rotate_half`-style
RoPE. The two modes pair different dimension indices:

- `GGML_ROPE_TYPE_NEOX` (mode=2) pairs `(i, i+d/2)` — matches HF
  `rotate_half`. **This is what Llama, Mistral, Qwen, Granite, Gemma,
  GPT-NeoX, and basically every modern LLM uses.**
- `GGML_ROPE_TYPE_NORMAL` (mode=0) pairs adjacent dims `(0,1), (2,3)…`
  Very few models use this. If you can't find a citation for it in the
  model's reference code, you probably don't want it.

Signature of the bug: the model loads, runs, and generates fluent-looking
text — but it's garbage. Byte-level detail preservation at the layer
boundaries hides it for the first few layers; by layer 40 the hidden
state is in the wrong basis and the LM head picks nonsense tokens. The
giveaway is that the Python reference transcript is perfect and the
ggml transcript is fluent but wrong. Always diff against the reference
at each layer boundary.

### Flash attention tensor layout

`ggml_flash_attn_ext(Q, K, V, mask, scale, max_bias, logit_softcap)`
expects Q, K, V in `[head_dim, T, n_heads]` layout with their final
dimension stride 1. If you've computed Q/K/V as `[d_model, T]` from a
`ggml_mul_mat`, you need three steps to get there:

1. `ggml_reshape_3d(_, hd, n_heads, T)` — expose the head dim
2. `ggml_permute(_, 0, 2, 1, 3)` — swap `n_heads` and `T`
3. `ggml_cont(_, …)` — flash-attn requires contiguous memory

Skipping the `ggml_cont` causes a silent shape error downstream. The
output comes back as `[head_dim, n_heads, T, 1]` and you need a
`ggml_reshape_2d(_, hd * n_heads, T)` to collapse it back into `[d, T]`
for the output projection.

### GQA native support vs explicit expansion

`ggml_flash_attn_ext` natively handles GQA when `n_kv_heads < n_heads`
and the K/V tensors have the right shape — it broadcasts each KV head
across `n_heads / n_kv_heads` query heads internally. BUT the K/V
tensors must be laid out as `[head_dim, T, n_kv_heads]`, not
`[head_dim, T, n_heads]`.

If you manually expand KV via `ggml_repeat_4d` before calling flash-attn,
you get a more memory-hungry but more forgiving path that works with
either layout. All three of voxtral, voxtral4b, qwen3, and granite LLM
blocks do the explicit expand for simplicity.

### `ggml_backend_sched` lifetime

Two common patterns, with very different performance:

- **Create once, reset between calls.** Create the scheduler at model
  init with the worst-case graph size (whichever of your stages is
  largest — usually the LLM prefill), and call `ggml_backend_sched_reset`
  between compute calls. Near-zero per-call overhead.
- **Recreate every call.** This is what qwen3/voxtral currently do
  because their graph sizes differ between stages (conv, encoder, LLM
  prefill, LLM decode step). Cheap in absolute terms but adds ~5-15 ms
  per call, which matters for the single-token decode loop.

Fix: compute the max graph node count once at init by building the
largest graph variant and measuring its node count, then create a
single scheduler with that budget and `reset` between stages. See
`TODO.md` under "Per-model follow-ups → qwen3 / voxtral".

### Flash attention on prefill AND decode

The LLM-based backends all use `ggml_flash_attn_ext` for prefill. Using
it for the single-token decode step too (not just prefill) halves the
decode-time graph size and runs ~2× faster on CPU. Qwen3 and voxtral
already do this. Check any new backend's per-token wall time to
confirm it's taking this path.

### In-place recursive FFTs are const-unsafe

voxtral / voxtral4b / qwen3 ship a recursive radix-2 Cooley-Tukey FFT
that treats its input buffer as 4× scratch space during recursion.
These can't be called through a `const float *` function pointer —
they modify memory past their nominal input length. When integrating
with `core_mel::FftR2C` (which has a const-input contract), wrap the
FFT with a thread-local scratch copy:

```cpp
static void model_fft_wrapper(const float * in, int N, float * out) {
    static thread_local std::vector<float> scratch_in;
    static thread_local std::vector<float> scratch_out;
    if ((int)scratch_in.size()  < 4 * N) scratch_in.assign((size_t)4 * N, 0);
    if ((int)scratch_out.size() < 8 * N) scratch_out.assign((size_t)8 * N, 0);
    std::memcpy(scratch_in.data(), in, (size_t)N * sizeof(float));
    model_fft(scratch_in.data(), N, scratch_out.data());
    std::memcpy(out, scratch_out.data(), (size_t)(2 * N) * sizeof(float));
}
```

One allocation per thread, zero per-call heap churn.

---

## Mel spectrograms

### Two algorithm clusters, not one

Nine model files in `src/` had nine different mel implementations.
They fall into exactly two clusters, distinguished by log base and
normalisation scheme. Knowing this upfront would have collapsed the
refactor into one parameterised function.

| Cluster | Log | Normalisation | Output layout | Used by |
|---|---|---|---|---|
| **NeMo** | `ln` | per-mel z-score | `(T, n_mels)` | parakeet, canary, canary_ctc, cohere |
| **HF / Whisper** | `log10` | global clip `(max(x, max(x)-8) + 4) / 4` | `(n_mels, T)` | whisper, qwen3, voxtral, voxtral4b, granite |

Sub-variants you'll hit once per cluster:
- `log_guard_mode`: NeMo uses `log(x + eps)`, HF uses `log(max(x, eps))`.
  Numerically close but not identical.
- `matmul_precision`: NeMo uses `float` accumulator, HF uses `double`.
  This matters for bit-exact regression against PyTorch reference.
- `fb_layout`: NeMo stores the filterbank as `[n_mels, n_freqs]`, HF
  stores it as `[n_freqs, n_mels]`. Transposed.
- `drop_last_frame`: HF drops the last STFT frame; NeMo keeps it.
- `drop_first_frame_if_odd`: voxtral4b needs even T for a stride-2 conv.
- `pad_to_T`: voxtral 3B pads to 3000 frames (= 30s) AFTER log, BEFORE
  normalisation, using `log(eps)` as the pad value so padded frames
  don't skew the global-clip max.
- `stacked_frames`: granite's output is `(160, T/2)` = two 80-mel
  frames zipped along channels. (Still inline — see TODO.md.)

See `src/core/mel.h` for the parameterised version.

### Cohere's cohere_fft_r2c + pre-emphasis

Cohere is the one NeMo-cluster model that doesn't fit the others
cleanly: it applies a `samples[i] = samples[i] - 0.97 * samples[i-1]`
pre-emphasis filter before the STFT. Easy to handle — do the pre-
emphasis in the model wrapper, then call `core_mel::compute` on the
pre-emphasised signal.

Cohere also uses `cblas_sgemm` for the power→mel matmul. When we
migrated to the manual accumulator in `core_mel`, the summation order
changes slightly and one SRT timestamp shifted by 80 ms (one encoder
frame). The transcript text is bit-identical. If bit-exact BLAS
output becomes a hard requirement, a BLAS-backed matmul path can be
added to `core_mel` behind a feature flag.

---

## Quantisation and memory

### Q4_K is the production default

Across every model we've benchmarked, Q4_K has been the sweet spot:

- **parakeet**: F16 9.3s → Q4_K 5.3s (1.75× faster, 0.97× realtime CPU, quality identical)
- **canary**: F16 13.0s → Q4_K 6.5s (2.0× faster, 1.19× realtime CPU)
- **cohere**: F16 27.6s → Q4_K 14.8s (1.87× faster, 2.72× slower than realtime)
- **qwen3-asr**: Q4_K 6.5s on jfk.wav (1.7× realtime)
- **voxtral 3B**: 70s total, 242 ms/token (3B is heavy on CPU)
- **voxtral 4B Realtime**: F16 133s → Q4_K 49s (2.7× faster, 0.22× realtime CPU)
- **granite 1B**: Q4_K 22.5s on jfk.wav (0.49× realtime)

Q5_0, Q6_K, Q8_0 are marginal improvements on smaller models but don't
close the gap to Q4_K in wall-clock tests. F16 is 2-3× slower than
Q4_K on CPU with no measurable quality improvement for ASR.

### Baked mel filterbank, baked Hann window

Every model's GGUF stores the mel filterbank and Hann window as regular
F32 tensors, not as arrays of numbers in the GGUF metadata. The
`core_mel::compute` function reads them via `ggml_backend_tensor_get`
at inference time. Pros: same precision as the Python reference, no
numerical drift from Slaney reconstruction in C++; cons: a couple hundred
KB of extra weight bytes. Worth it.

### F16 KV cache is non-negotiable for LLM backends

Qwen3/voxtral/voxtral4b/granite LLM KV caches are all F16. Cohere's
self-attention KV is still F32 (historical, see TODO.md for the planned
upgrade). Halves GPU memory and bandwidth with no observable quality
loss in ASR workloads.

---

## CPU vs ONNX vs PyTorch baselines

### Where the time goes (Cohere, 11s clip, 8-thread CPU)

Representative profile from the Q4_K path:

| Op | % of time |
|---|---:|
| `mul_mat` | 87.6% |
| `im2col` (conv subsampling) | 7.0% |
| Everything else | 5.4% |

`mul_mat` at 87.6% is near hardware peak for F16 GEMM. Any optimisation
that doesn't move the `mul_mat` number is noise.

### Where ONNX beats ggml on x86 (and doesn't on Metal)

Measured on a 44s clip, x86 4-thread CPU, quantised:

| Implementation | Encoder | Decoder | Total | RTFx | Notes |
|---|---|---|---|---|---|
| ONNX INT8 (CPU) | 19.5s | 11.7s | 31.2s | 1.44× | DNNL AVX-512 INT8 GEMM |
| ONNX INT4 (CPU) | 22.5s | 12.7s | 35.2s | 1.28× | INT4 weight-only |
| **ggml Q4_K (CPU)** | 42.1s | **3.1s** | 45.4s | 0.99× | ggml AVX2 |
| ggml F16 (CPU) | 49.1s | 4.1s | 53.5s | 0.84× | ggml AVX-512 F16 |
| PyTorch F16 (A100 GPU) | — | — | ~1-2s | ~25× | baseline |

Two observations:

1. **ONNX is ~2× faster in the encoder** on x86 CPUs with AVX-VNNI, because
   DNNL uses `vpdpbusd` for INT8 GEMM and ggml's `vec_dot_q8_0_q8_0`
   still uses `pmaddubsw`/`pmaddwd`. There is no CPU path to close this
   gap without implementing AVX-512 INT8 GEMM in ggml's `quants.c`.
   Tracked in `UPSTREAM.md`.

2. **ggml is 3-4× faster in the decoder.** ONNX passes the full KV cache
   (~268 MB) across the Python→ONNX→Python boundary on every decode
   step. For 167 tokens that's ~45 GB of unnecessary data movement.
   Our ggml in-place KV cache with tensor views moves zero bytes. This
   advantage grows with output length.

On Metal or CUDA, the encoder gap closes entirely: our ggml graphs
already use ops that have GPU kernels (`ggml_mul_mat`,
`ggml_conv_2d_dw_direct`, `ggml_flash_attn_ext`). An M1 Metal run of
the same Cohere clip hits ~11.9× realtime compared to 1.24× Q4_K CPU.

### Python and Rust libtorch are both ~25-30× realtime

Both `transformers` and `cohere_transcribe_rs` (tch crate) go through
libtorch CPU F32 and land at ~160s for a 5.4s clip. There is no easy
win on the Rust side without switching backends.

---

## Audio format lessons

### miniaudio + stb_vorbis handle the common cases

Out of the box, every ASR runtime in this repo accepts WAV / FLAC / MP3
/ OGG Vorbis at any bit depth, any sample rate (auto-resampled to
16 kHz), mono or stereo (auto-mixed to mono). No external dependencies.
The two embedded single-header decoders (`miniaudio`, `stb_vorbis`) are
enough for 95% of real-world ASR pipelines.

### `WHISPER_FFMPEG=ON` only helps bare Opus

Upstream whisper.cpp's `examples/ffmpeg-transcode.cpp` has known bugs
on mp4-family containers: `.m4a` crashes with `munmap_chunk(): invalid
pointer` on the first audio chunk read, and `.webm` (Opus-in-WebM)
hangs indefinitely after the libavformat headers are parsed. Both use
the same `av_read_frame` + `avcodec_send_packet` loop.

Bare-codec `.opus` files work cleanly in the FFmpeg build. So the
practical advice is: enable `WHISPER_FFMPEG=ON` only if you need
in-process `.opus` decoding. For everything else, pre-convert:

```bash
ffmpeg -i input.m4a -ar 16000 -ac 1 -c:a pcm_s16le -y /tmp/audio.wav
```

This is the universally safe path and identical to what the in-process
path would produce if it worked. Documented in `UPSTREAM.md` with a
minimal reproducer.

---

## Language handling

### Auto-detect can silently code-switch

Parakeet's auto-language-ID works well for clean speech but drifts into
English on German clips with technical vocabulary or proper nouns. A
90-second German clip about "Industrial Forschung" and "Technische
Universität" came back with "Industrial Forschung" and "Tech Technische
University" in the transcript. **This is not a chunking issue — VAD-based
segmentation gives the same code-switching.** The encoder classifies the
clip correctly but the decoder drops into English mid-stream on lexical
hints.

Lessons:
1. For production use on a known language, always prefer a model with
   an explicit language flag. Canary's `-sl de -tl de` is the fix — the
   decoder is forced into German by the task-token prefix and cannot
   code-switch.
2. Auto-detect models are better for mixed-language pipelines where the
   language isn't known.
3. Test with vocabulary-heavy, non-English clips before shipping. Clean
   short phrases pass every test you give them.

### Canary's prompt prefix is the mechanism, not magic

Canary's "explicit language" feature is implemented as a task-token
prefix in the decoder prompt, before the audio encoder output. Specifically:

```
<|startofcontext|>[source_lang][target_lang]<|transcribe|>[punctuation]
```

When `source_lang != target_lang`, the task token is `<|translate|>`
instead of `<|transcribe|>`. This is how canary does speech translation
(DE→EN, EN→FR, etc.) in the same model.

---

## Model architecture comparisons

### Voxtral: CrispASR standalone vs llama.cpp mtmd vs max-lt wrapper

Three independent C++ implementations of Voxtral-Mini-3B exist. We
compared them head-to-head and the conclusion was important enough to
preserve.

| | **CrispASR** | max-lt/voxtral-cpp | llama.cpp mtmd |
|---|---|---|---|
| Model files | 1 GGUF | 2 (model + mmproj) | 2 (model + mmproj) |
| Tokenizer | Embedded Tekken blob | llama.cpp native | llama.cpp native |
| LLM forward | Hand-written ggml | llama.cpp core | llama.cpp core |
| [BEGIN_AUDIO] bug | ✔ not affected | needs patch | needs manual fix |
| 30s truncation | ✔ not affected | affected | affected |
| Diff-tested vs PyTorch | ✔ every stage | ✗ | ✗ |
| Lines of model code | ~1300 | ~100 wrapper | 0 (all in llama.cpp) |
| GPU support | ✗ (CPU-only now) | ✔ via llama.cpp | ✔ via llama.cpp |

The llama.cpp `mtmd` multimodal subsystem has two known bugs affecting
Voxtral specifically ([#17868](https://github.com/ggml-org/llama.cpp/issues/17868),
[#18419](https://github.com/ggml-org/llama.cpp/issues/18419)) that
were ignored by maintainers, and a community member reports worse
accuracy in llama.cpp than in transformers/vLLM at the same precision.
Ollama dropped llama.cpp specifically for multimodal due to
instability.

**Recommendation:** keep CrispASR as its own standalone ggml runtime
for ASR. It is diff-tested against PyTorch at every architectural
boundary (LLM cosine sim 0.999973, top-5 5/5 match on identical inputs),
which is the confidence our users need. Do NOT rewrite it on top of
mtmd. When we want GPU, use ggml's Metal/CUDA backends directly on our
existing graph builders — `ggml_flash_attn_ext` already has GPU kernels.
The main work is wiring up `ggml_backend_metal_init()` /
`ggml_backend_cuda_init()` as alternatives to the CPU backend (~50 LOC).

---

## Regression testing discipline

Every migration commit in `src/core/` includes a `md5sum`-level
regression test against `samples/jfk.wav`. The discipline:

1. Run the current binary, capture output + auxiliary outputs (SRT/VTT/JSON)
2. Make the change
3. Rebuild
4. Re-run, compare with `md5sum` and `diff`
5. If bit-identical, commit. If not, investigate.

Two cases where bit-identity is not achievable:

1. **Cohere mel migration.** CBLAS sgemm → manual accumulator changes
   the float summation order, shifting one SRT boundary by 80 ms (one
   encoder frame). Transcript text is byte-identical. Accepted.
2. **Whisper code path.** Untouched by `src/core/` refactors; bit-
   identical against upstream `whisper-cli` is the gate.

The few FFNs / attention blocks where ggml graph op ordering matters
have all come back bit-identical so far. Flash attention results
depend on the order Q/K/V were committed to the graph, but as long as
the helper emits them in the same order the inline code did, you get
bit-identical output.

---

## Specific bugs that cost us a day each

These are each preserved in `HISTORY.md` with full context. Summary form:

1. **Granite RoPE mode (NEOX vs NORMAL).** Model loaded, ran, produced
   fluent nonsense. Fix: one enum value.
2. **Voxtral 4B realtime audio padding.** `32*1280 + 17*1280 + 1280*(right_align)`
   left and right pads are non-negotiable. Skipping the right pad
   silently breaks the encoder graph reshape.
3. **Voxtral 4B Realtime audio_length_per_tok=8.** 3B uses 4 (one audio
   frame per 4 Whisper frames); 4B uses 8. Wrong value → audio-to-token
   alignment off by 2× and transcript drifts.
4. **Cohere F32 self-attention KV.** Still not fixed; costs 2× GPU
   memory. Tracked in TODO.
5. **Qwen3 windowed attention.** Chunked self-attention via `cu_seqlens`
   with window size ~104 positions. Standard full self-attention
   produces wrong output. This is the trickiest part of the qwen3 port.
6. **Hann window centering in Granite mel.** The window must be
   symmetrically zero-padded to n_fft; off-by-one on the centering shifts
   the power spectrum peak and breaks downstream everything.
7. **Q-Former layer norm target.** BLIP-2 projector LN applies to the
   query tokens, not the encoder output. Wrong tensor → garbage projector
   output → garbage LLM input → garbage transcript.
8. **Silero LID: five compounding bugs.** The native port of Silero's
   95-language classifier went through Swedish → Mongolian → Bashkir →
   Khmer → Chinese → Punjabi → English on jfk.wav, each fix changing
   the top prediction. Root causes, in order of severity:
   (a) **Front-end padding.** ONNX uses constant zero-pad 160/side on
       audio; we used reflection-pad 320 on the left. The padding type
       and amount are buried in a Pad node with a dynamically-computed
       pad vector from a chain of 15 ONNX ops.
   (b) **Stride-2 output size.** Conv1d(T, k=1, s=2) output is
       `(T-1)/2+1`, not `T/2`. Off-by-one cascades through 4 stride-2
       stages (1101→551→276→138→69) — wrong value drops 1 frame per
       stage, silently shifting the feature alignment.
   (c) **QKV split order.** ONNX slices QKV as K[0:D], Q[D:2D],
       V[2D:3D]. We assumed Q,K,V order. The only way to discover this
       is to dump the Slice node inputs and compare the split boundaries.
   (d) **Missing ReLU after stride-1 projections.** Stages 4-7 use
       stride-1 Conv1x1→ReLU for dim change (128→192). The ReLU is
       easy to miss since the stride-2 stages already had it.
   (e) **Missing tanh in attention pooling.** ONNX does dot→Tanh→
       Softmax; we did dot→Softmax. The Tanh compresses the score
       range, which completely changes the attention distribution.
   **Lesson:** When porting an unfamiliar ONNX model, dump intermediates
   at every graph boundary and diff against the native code BEFORE
   debugging individual ops. The bug is almost never where you expect.

---

## Quantization

### Small models with conv-heavy architectures resist quantization

The Silero LID model (16 MB F32, 507 tensors) was tested with Q8_0 and
Q5_0 quantization. Both broke accuracy completely (French/Shona instead
of English). The model's parameters are mostly small Conv1d kernels
(dw_conv [5,1,C], pw_conv [1,C,C]) where C ∈ {128, 161, 192}. These
tensors have very few elements per row (1-5), making block quantization
destructive. Only the transformer QKV/out/FFN projections and classifiers
(34 of 507 tensors) have enough elements per row to quantize safely, but
that saves only 3-5 MB — not worth the accuracy loss.

**Rule of thumb:** If a model's parameter count is dominated by Conv1d
kernels with small spatial dimensions (k ≤ 5) and few channels (C < 256),
ship it F32. The 16 MB F32 Silero LID model is smaller than a single
layer of most ASR encoders — quantization is pointless.

---

## Methodical debugging of ported models against ground truth

This is the single most important workflow in the project. Every model
port that "almost works" but produces wrong output will eat days unless
you follow this process systematically.

### The protocol

1. **Get a reference implementation that provably works.** Either the
   original Python/ONNX model (preferred — run via onnxruntime), or a
   known-good C++ implementation. If ONNX: add all internal nodes as
   graph outputs and run with intermediate capture.

2. **Dump intermediates at every graph boundary.** Not just input/output
   — dump after EVERY stage: normalization, projection, attention,
   FFN, residual add. Save as `.npy` files with clear names.

3. **Compare C++ vs reference at each stage, starting from the INPUT.**
   Don't start debugging the attention if the input is already wrong.
   Print first 8-16 values of each tensor at frame t=0. The divergence
   point tells you exactly which operation is broken.

4. **When you find the divergence point, check these in order:**
   - **Tensor layout/transpose** — ggml uses ne[0]-fastest (column-major).
     A `[T, H]` row-major C array becomes `[H, T]` in ggml (ne[0]=H).
   - **Weight shapes** — GGUF stores shapes in ggml ne-order. A numpy
     `(1024, 4096)` weight becomes ggml ne `[1024, 4096]`. For
     `ggml_mul_mat(W, x)` = W^T @ x, we need `W.ne[0] == x.ne[0]`.
   - **Padding type and amount** — zero vs reflect vs replicate. ONNX
     Pad nodes encode padding as a dynamically-computed vector from
     chains of 10+ ops. Always dump the actual padded tensor.
   - **Activation functions** — missing ReLU, tanh, GELU. These are
     easy to miss when tracing the ONNX graph manually.
   - **Operation order** — pre-norm vs post-norm, attention before or
     after stride-2, QKV split order.
   - **Formula details** — stride-2 output is `(T-1)/2+1` not `T/2`.
     Reflection padding `pad[i] = data[pad_size - i]` not `data[i]`.
     Scale factor in attention: 1/sqrt(head_dim) not 1/sqrt(d_model).

5. **For ggml graph debugging specifically:**
   - The `ggml_backend_sched` may not correctly associate model weight
     tensors with their backend buffer. Test with `ggml_backend_alloc`
     instead of the scheduler for isolation.
   - Mark tensors with `ggml_set_name()` and read them with
     `ggml_backend_tensor_get()` BEFORE calling `ggml_backend_sched_free()`.
   - F16 weight tensors in ggml_mul_mat work correctly in single mini-
     graphs (as in `ggml_linear_f32()`) but may misbehave in large
     graphs where the scheduler manages buffer allocation.
   - When in doubt, build a 1-layer graph first and verify it matches
     the manual path before scaling to all layers.

6. **Never trust "close enough".** If the first frame's values differ
   by more than 1e-4 from the reference, there's a bug. Float32
   accumulation order can cause ~1e-5 drift per operation, so after
   24 transformer layers you might see ~1e-3 drift — but a 0.1
   difference at layer 0 means a structural bug.

### Common traps

- **ONNX QKV split order is not always Q,K,V.** Silero LID uses K,Q,V.
  The only way to know is to dump the Slice node boundaries.
- **ONNX padding is computed dynamically.** Don't assume reflect/zero
  from the model architecture — dump the Pad node's padding vector.
- **ONNX Reshape+Transpose chains for multi-head attention** can
  interleave heads differently than simple offset slicing. Always
  dump the post-reshape tensors to verify head layout.
- **ggml_norm normalizes over ne[0].** Make sure ne[0] is the feature
  dimension, not the time dimension.

### Worked example: CosyVoice3 speech_tokenizer_v3 (2026-05-29)

A near-complete but never-validated ggml port of the CV3 s3tokenizer
(whisper-128 mel → 2× conv subsampler → 12 FSMN/attention blocks → FSQ)
crashed on the first conv and, once it ran, matched the ONNX reference
on only **14%** of output tokens. Three bugs, all pinned by the
stage-by-stage diff:

- **gguf-py reverses the numpy shape into ggml `ne`, and the C++
  runtime consumes weights as-is.** So the *converter* must emit each
  weight in the numpy layout whose reverse is the ggml layout the graph
  wants: conv1d kernels in onnx `(OC, IC, KW)` order → ggml
  `[KW, IC, OC]`; 2D MatMul/Linear `.w` transposed to `(out, in)` →
  ggml `[in, out]`; depthwise FSMN kernel onnx `(C, 1, KW)` as-is →
  ggml `[KW, 1, C]`. The draft had conv/FSMN double-transposed and 2D
  weights un-transposed; **square attention weights (1280×1280) hid the
  shape error** (right shape, transposed values) while the first conv
  asserted `OW > 0`.
- **erf-GELU vs tanh-GELU.** ONNX exports erf-GELU (count the `Erf`
  ops). `ggml_gelu` is the tanh approximation — use `ggml_gelu_erf`.
- **An unset graph input silently holds garbage.** The RoPE
  `positions` tensor was `ggml_set_input`-created but never filled, so
  rotary ran on whatever was in the buffer. Always set every input
  before compute.

**The most dangerous trap: a green end-to-end test masked the wrong
component.** The cloned audio ASR-roundtripped to 0% WER while the
tokenizer was only 14% correct — because in zero-shot TTS the prompt
speech tokens mostly condition speaker/prosody, and the *spoken words*
come from the LM. A passing end-to-end check does **not** validate a
ported sub-component; diff each stage against ground truth regardless.

Harness mechanics that made this fast: expose ONNX intermediate edges
as graph outputs (`m.graph.output.append(make_tensor_value_info(edge))`)
to capture per-stage references, feed the C++ side the *reference* input
(here the whisper mel via `embeds_in`) so the network is diffed in
isolation from the front-end, and tag each ggml stage with
`ggml_set_name` + `ggml_set_output`. Result: every stage cos=1.0,
tokens byte-exact (max_abs=0).

**Filterbank ≠ STFT.** The full runtime path stayed at 99.24% (2/264
token flips) from a low-mel-bin delta vs whisper's mel. Embedding
whisper's *exact* `mel_filters.npz` in the GGUF did **not** change it —
proving the delta is in the STFT magnitude (shared `core_mel`), not the
filterbank. When a mel diff is concentrated in low bins, suspect the
STFT/window/pad before the filterbank.

**Open follow-up (2026-05-29):** WAV-cloned outputs are ~14 dB quieter
than the `zero_shot` baseline (peak ~0.05 vs ~0.66) on *both* the native
and Python-shellout extraction paths — so the cause is in the shared
synth core, not the extractor. Prime suspects: `ref_mel` length not
aligned to `2 × T_prompt_tok` (the baked zero_shot voice is aligned at
174 = 2×87; a 10 s runtime clip gives matcha-mel ≠ 2×s3tok-tokens), or
spk_emb projection magnitude for clips longer than ~3 s.

---

## ggml graph allocation: gallocr vs compute_with_ctx

### gallocr/sched corrupt external weight tensors

When a ggml graph references tensors from an external context (e.g. model
weights loaded via `core_gguf::load_weights`), `ggml_gallocr_alloc_graph`
and `ggml_backend_sched` reallocate buffers for these tensors, overwriting
their data with uninitialized memory. This was confirmed by a minimal
single-op test:

- `ggml_graph_compute_with_ctx` (no allocator): **correct** — directly
  accesses `tensor->data` pointers, which point to the loaded GGUF data.
- `ggml_gallocr_alloc_graph` + `ggml_backend_graph_compute`: **wrong** —
  the allocator sees the external tensors as "unallocated" despite having
  valid `->data` and `->buffer` pointers, and allocates new buffers over
  them.

The `ggml_gallocr_is_allocated()` function at ggml-alloc.c:591 checks
`t->data != NULL || t->buffer != NULL`, which should catch external
tensors. But the two-phase reserve+alloc flow apparently doesn't preserve
this across the reserve step.

**Workaround:** Use `ggml_graph_compute_with_ctx` with `no_alloc=false`
for the graph context. All intermediate tensors get memory from the
context pool, and external weight tensors are referenced via their
existing `->data` pointers. Downside: no memory reuse between layers —
each intermediate stays alive for the entire graph (~80 MB/layer for
wav2vec2-large with 549 frames).

### ggml 2D tensor layout and transpose

ggml stores 2D tensor `[ne[0], ne[1]]` as `data[i0 + i1 * ne[0]]`.
A tensor with `ne[0]=V, ne[1]=T` has element `(v, t)` at `data[v + t*V]`.
This is the SAME memory layout as a C row-major array `float arr[T][V]`
where `arr[t][v] = data[t*V + v]`. So **no transpose needed** when
converting between ggml `[V, T]` and C `[T, V]` row-major — they're
the same bytes. The earlier wav2vec2 code had wrong transposes at THREE
places (input, layer readback, LM head input) that shuffled data into
garbage. The fix was to remove ALL transposes and use `memcpy` /
`std::copy` directly. **When in doubt, don't transpose.**

### Layer-by-layer graph execution as a gallocr workaround

When `ggml_gallocr` corrupts external weight tensors, building one
graph per transformer layer with `ggml_graph_compute_with_ctx` and
`no_alloc=false` is a viable workaround. Each layer graph uses ~80 MB
(for wav2vec2-large with T=549, H=1024, 16 heads) and is freed after
use, so total RSS stays at ~800 MB instead of 3+ GB. The hidden state
is copied in/out of each layer graph via `memcpy`. Per-layer max_diff
vs the manual reference path is < 0.005 (float32 accumulation noise).

This is slower than a single-graph approach (24 context alloc/free
cycles + 24 graph plans) but produces correct results and uses much
less memory. Good enough for CPU; for GPU acceleration, fixing gallocr
to skip pre-allocated tensors is the proper solution.

---

## Performance: what faster-whisper / insanely-fast-whisper do

Analysed SYSTRAN/faster-whisper and Vaibhavs10/insanely-fast-whisper
(April 2026). Key techniques and applicability to ggml:

**Already have in CrispASR:**
- Quantization (Q4_K/Q5_0/Q8_0) — fundamental to ggml
- Flash attention (ggml_flash_attn_ext) — used by whisper backend
- VAD pre-filtering (Silero) — skips silence before transcription
- Multi-file parallelism (n_processors)

**Could add (GPU-dependent, large impact):**
- **Batched encoder** — process N audio chunks simultaneously on GPU.
  Faster-whisper's `BatchedInferencePipeline` with batch_size=8 gives
  3-5x speedup. Requires GPU (batch doesn't help much on CPU since
  we already use all cores per chunk).
- **Speculative decoding** — use a small "draft" model to predict
  tokens, verify with the large model. 2-4x speedup for autoregressive
  LLM backends (granite, voxtral, qwen3). Needs two models loaded.

**Could add (CPU-friendly, moderate impact):**
- **Pipelined mel+encode** — while LLM decodes chunk N, compute mel
  for chunk N+1 in a background thread. ~15-20% speedup for LLM
  backends on multi-core CPUs.
- **Encoder output caching** — for repeated queries on the same audio
  (e.g. trying different languages), cache the encoder output and only
  re-run the decoder. Already implicit in whisper's architecture.

**Not applicable:**
- CTranslate2's CUDA kernels — ggml has its own CUDA backend
- BetterTransformer API — PyTorch-specific
- fp16 compute — ggml already does F16 matmul natively

**Bottom line:** On CPU, we're already within 2x of the theoretical
limit (2.2x realtime for parakeet on jfk.wav). The big wins are
GPU-specific: batched encoder (5x) and speculative decoding (2-4x).

**Implemented optimizations (April 2026):**
- Parallel VAD slice transcription (thread pool with separate backend
  instances — helps on GPU where each instance uses a separate stream)
- Full-graph ggml_backend_sched path for wav2vec2 with explicit weight
  tensor assignment via `ggml_backend_sched_set_tensor_backend` — GPU-
  ready single-graph dispatch for all 24 transformer layers
- Buffer reuse across layers (saves 24×80MB alloc/free cycles)
- Server-mode audio cache (instant response on repeated queries)
- Realtime speed reporting per file
- All model weights loaded to GPU when ggml_backend_init_best() picks
  a GPU backend (already built into core_gguf::load_weights)

**Key discovery:** `ggml_backend_sched_set_tensor_backend()` prevents
the scheduler from reallocating external weight tensors. This was the
missing piece for making the full-graph path work with model weights
on a separate buffer. Without it, gallocr corrupts external tensors.

### Windows fseek overflow: the silent >2 GB file killer

On Windows (MSVC), `long` is 32-bit even on x86_64. `fseek(fp, (long)offset, SEEK_SET)`
silently wraps around at 2^31 = 2.1 GB. For GGUF files larger than
this (voxtral4b Q4_K = 2.35 GB, Q8_0 = 4.4 GB), tensors stored past
the 2 GB boundary get read from the wrong file offset, resulting in
"missing tensor" errors or corrupt data.

The fix: `_fseeki64()` on Windows, `fseeko()` on POSIX. Also add
native Windows mmap (`CreateFileMapping` + `MapViewOfFile`) to bypass
the fseek path entirely.

**Lesson:** `fseek(fp, (long)x, ...)` is a bug on Windows for any file
that might exceed 2 GB. Always use platform-specific 64-bit seek. This
is a classic portability trap that doesn't manifest on Linux/macOS
(where `long` is 64-bit on LP64).

---

## VAD integration and long audio (April 2026)

### whisper VAD returns centiseconds, not seconds

The `whisper_vad_segments_get_segment_t0/t1()` functions return
timestamps in **centiseconds** (e.g. 29.0 = 0.29 seconds), not
seconds. Our initial integration multiplied by `sample_rate` directly,
producing sample indices 100× too large. Every segment fell past the
end of the audio, causing "no speech detected" for every file.

**Lesson:** Always check the units of external API return values. The
whisper.cpp VAD API stores `start`/`end` via `samples_to_cs()` (line
5676) and the internal code divides by 100.0 for display (line 6914).
The getter functions return the raw centisecond values.

### Short VAD segments break ASR quality

Silero VAD can produce very short segments (0.35s) for speech with
brief pauses. These are too short for most ASR encoders to produce
reliable output. On jfk.wav (11s), the VAD split into 5 segments of
0.35-2.4s each, causing parakeet to produce garbled output.

**Fix:** Post-merge adjacent VAD segments: combine if gap < 1s or if
the accumulated segment is shorter than 3s. This produces 2 merged
segments instead of 5 tiny ones, with correct transcription.

### VAD stitching matches whisper.cpp quality

whisper.cpp stitches VAD segments into one contiguous buffer with 0.1s
silence gaps, builds a mapping table, processes as one audio stream,
then remaps timestamps. This is fundamentally better than independent
per-slice processing because the decoder sees continuous audio context.

We now do the same for non-whisper backends: stitch → single
`transcribe()` call → remap. Tested on 89s and 227s audio — no
boundary artifacts, correct timestamps throughout.

### Backend-specific audio length limits

| Backend | Mel length | Hard limit? | Notes |
|---|---|---|---|
| whisper | 3000 frames (30s) | Yes | Pads to exactly 3000 frames |
| voxtral 3B | 3000 frames (30s) | Yes | `T_mel = 3000` hardcoded |
| voxtral4b | variable | No | Causal encoder, streams |
| qwen3 | variable | No | Chunked conv subsampler |
| parakeet | variable | No | O(T²) attention, ~5min practical limit |
| canary | variable | No | O(T²) attention, ~5min practical limit |
| cohere | variable | No | O(T²) attention, ~5min practical limit |
| granite | variable | No | Block-local attention (ctx=200), any length |

For whisper and voxtral 3B, 30s chunking is mandatory. For the rest,
longer chunks work but hit O(T²) memory walls. VAD stitching helps by
removing silence (shorter effective audio), and the max-chunk split
prevents OOM on very long continuous speech.

### Qwen3 forced aligner leading-silence issue

The qwen3 forced aligner assigns timestamps starting from 0 even when
audio has leading silence. On the user's 227s JavaScript tutorial with
~3s of leading silence, the first word was stamped at 0.24s instead
of ~3.2s. With VAD stitching, the silence is removed before alignment,
fixing the issue.

**Lesson:** The forced aligner only works well when the audio starts
with speech. Always use VAD to trim silence before alignment.

### Qwen3 forced aligner monotonicity

The reference implementation (`qwen3_forced_aligner.py`) has a
`fix_timestamp()` function using longest-increasing-subsequence (LIS)
to correct non-monotonic timestamps. We use a simpler forward clamp
(each timestamp >= previous). This handles most cases but may miss
complex inversions. Parakeet's native TDT timestamps are always
better when available.

### CrispASR vs voxtral.c: 3.8× faster on CPU

Direct same-hardware comparison (Xeon 4-core, no GPU) on jfk.wav:
- voxtral.c (OpenBLAS): 11m 0s (encoder 220s, decoder 2660ms/step)
- CrispASR (ggml): 2m 52s
- Speedup: 3.8×, attributable to ggml's optimised matmul kernels

### Susurrus architecture insights

Susurrus (CrispStrobe's Python ASR tool) uses:
- `vad_filter=True` hardcoded in faster-whisper (always on)
- 25-minute chunks with 2s overlap for voxtral local
- GPU memory explicitly freed between chunks (`torch.cuda.empty_cache`)
- Generator-based segment yielding (streaming/incremental)

**Lesson:** VAD should be the default, not an opt-in. 30s chunks are
too conservative for most models; 5-10 minutes is practical for
variable-length backends on 16GB VRAM.

### wav2vec2-base: post-norm vs pre-norm (the silent architecture trap)

wav2vec2-base models (`do_stable_layer_norm=False`) use **post-norm**
transformer layers: `attention → residual_add → LayerNorm → FFN →
residual_add → LayerNorm`. wav2vec2-large models
(`do_stable_layer_norm=True`) use **pre-norm**: `LayerNorm → attention →
residual_add → LayerNorm → FFN → residual_add`.

Our initial implementation only had pre-norm (matching the large XLSR
model we first ported). Running a base model through pre-norm produces
all-identical outputs at every time position — the encoder loses
positional information and the CTC decoder outputs the same character
(argmax=24 = "b") at every frame.

**Symptoms:** Output is a single character repeated, or empty text.
All positions have the same argmax.

**Root cause debugging protocol:**
1. Get HF reference intermediates (CNN out, feature projection, encoder
   out, logits argmax) — these are ground truth.
2. Add debug fprintf to C++ at each stage boundary.
3. Compare stage by stage — CNN matched, feature projection matched,
   but logits diverged completely.
4. The argmax pattern `[24,24,24,24,...]` (all same) immediately points
   to an encoder bug that collapses positional information.
5. Check `do_stable_layer_norm` in the HF config — it controls the
   norm ordering and is the first thing to verify when porting a new
   wav2vec2 variant.

**Second bug:** CTC blank token. `config.pad_token_id=1` (BOS) in base
models, but CTC greedy decoding must skip vocab index 0 (`<pad>`) which
is the actual CTC blank. The converter now hardcodes blank=0.

**Lesson:** When porting a model architecture, always check for
configuration flags that change the graph topology (norm ordering,
activation type, bias presence). These are silent — the model loads
and runs without errors, but produces garbage. A debug copy of the
forward pass (`wav2vec2-ggml-debug.cpp`) with fprintf at each stage
boundary is kept for future model variant debugging.

---

## CLI ↔ library DRY refactor (April 2026)

v0.4.4–v0.4.8 moved every non-presentation CLI concern into `src/`
behind the shared C-ABI. Below are the lessons from the five-release
cycle — things worth remembering the next time a helper turns out to
be shared across more consumers than its location suggests.

### File names are claims; check them periodically

`examples/cli/crispasr_dart_helpers.cpp` started as Dart-only in
0.2.0 but by 0.4.0 it was the common FFI surface consumed by the CLI,
Dart, Python, and Rust. The file name was a documentation bug for
four releases. The first move (`crispasr_c_api.cpp` + updated header
comment) was pure churn and should have been done earlier. An
occasional pass over file/function names vs actual callers is worth
doing.

### Header basename clashes surface late

`src/crispasr_vad.h` and `examples/cli/crispasr_vad.h` coexisted
without error until the CLI source that `#include "crispasr_vad.h"`
happened to compile against the `src/` version (because the whisper
target is `target_include_directories(... PUBLIC .)`) — producing a
cryptic type mismatch with the CLI's `whisper_params` usage. Renaming
the CLI headers to `*_cli.h` (vad/diarize/lid/model_mgr/aligner) is
the clean fix; guards like `-I` ordering are fragile.

### Function-name collisions are worse than symbol collisions

Both `src/crispasr_lid.cpp` and `examples/cli/crispasr_lid.cpp`
defined `crispasr_detect_language(...)` as non-member C++ functions.
Different argument types → different mangled names → the linker is
happy. But any caller looking at `crispasr_detect_language(samples,
n, params)` has no idea which one it's getting. The safer pattern is
to suffix all CLI-shim symbols (`crispasr_detect_language_cli`,
`crispasr_apply_diarize`, etc.) so the call sites themselves signal
which layer they belong to.

### Backwards-compat aliases for renamed C-ABI symbols

When we renamed `crispasr_dart_helpers_version()` to
`crispasr_c_api_version()`, 0.4.x-era binaries already existed that
probed the old name. The library now exports both — the new function
is canonical, the old one is a 2-line thunk that calls it. A TODO in
source marks the removal after the next major version. The Dart
smoke test asserts **both** resolve and return the same value, so
we can't accidentally drop the alias early.

### POD ABI structs: be explicit about padding

`crispasr_vad_abi_opts` is `float + 5×int32` = 24 bytes. Clean on
64-bit with no padding. `crispasr_diarize_seg_abi` would have been
`int64 + int64 + int32` = 20 bytes with 4 bytes of trailing padding
on 64-bit — so we added an explicit `int32_t _pad` and documented
the 24-byte size so Dart/Python/Rust bindings can allocate the
struct by hand. Always check `sizeof` on both 32- and 64-bit
platforms when promoting a struct to the ABI.

### Policy stays in the CLI; algorithms go to the library

Every CLI shim follows the same pattern:
- **CLI-only (stays in `examples/cli/*_cli.{h,cpp}`)**: auto-download
  from `~/.cache/crispasr`, `isatty()` / TTY prompts,
  `sherpa-onnx` subprocess spawn, CLI-specific types like
  `whisper_params` / `crispasr_segment` / `crispasr_word`.
- **Library (goes to `src/*.{h,cpp}`)**: the actual algorithm —
  Silero VAD + stitching, diarize methods, whisper encode for LID,
  canary-CTC Viterbi, the model registry table, the WinHTTP/curl
  download helper.

This line is obvious in hindsight but we kept crossing it early on.
Rule of thumb: if a wrapper consumer (Python / Rust / Flutter app)
could want the function too, it belongs in the library.

### Rust CStr + static buffers for string-returning C-ABI

For the registry lookup (`crispasr_registry_lookup_abi`) we went
with caller-allocated output buffers (`out_filename`, `out_url`,
`out_size` as `char* + int cap`) rather than returning an opaque
handle with accessors. Reasons:
- Single call rather than 5 round-trips to Python/Dart
- No lifetime management for the wrappers
- Registry strings are small (URL up to ~256 chars), so a fixed
  2 KB stack buffer is fine
- Easy to detect "buffer too small" (return code 2) and retry

For `crispasr_align_words_abi` we did the opposite — one result can
contain hundreds of words, each a variable-length string, so we
kept the `session_result`-style handle + accessors pattern. Choice
of pattern depends on how bounded the output is.

### A Dart smoke test that only checks `lib.lookup()` catches 90% of binding drift

`flutter/crispasr/test/bindings_smoke_test.dart` just resolves every
C-ABI symbol by name. It takes 50 ms to run, needs no audio, and
catches: symbol rename typos, missing `CA_EXPORT`, new backend
dropping a target from `target_link_libraries(whisper PUBLIC ...)`,
and stale `.so`/`.dylib` on the test machine. Ran it after every
release in this cycle; caught one typo that would've shipped.

### Rust FFI: C++ exceptions abort the process

The old `CrispASR` Rust API (wrapping `whisper_full()` directly) crashes
with "Rust cannot catch foreign exceptions" because whisper.cpp's C++
code can throw exceptions (ggml assertion failures, `std::bad_alloc`).
Rust's `extern "C"` FFI boundary treats C++ exceptions as undefined
behavior — they unwind through Rust stack frames and trigger `abort()`.

The `Session` API works because `crispasr_session_transcribe()` is a
C-ABI wrapper implemented in C++ that catches exceptions internally
and returns error codes. The old `whisper_full()` path has no such
wrapper.

**Lesson:** All C-ABI functions exposed to Rust/Dart/Python must wrap
their body in `try { ... } catch (...) { return error_code; }`. The
Session API does this by design. The legacy whisper-direct functions
(`whisper_full`, `whisper_init_from_file_with_params`) do not. We mark
the old Rust `CrispASR` struct as deprecated in favor of `Session`.

### split-on-punct proportional fallback: the silent accuracy killer

When `--split-on-punct` was used without `-ml N`, the display segment
builder checked `seg.words.empty() || max_len == 0` and took the
proportional interpolation path — even when the backend (parakeet)
had produced accurate word-level timestamps. The proportional path
estimates sentence boundaries by character position ratio, which can
be off by 1+ seconds.

**Symptoms:** Sentence start/end times don't match the actual speech.
A sentence ending with "code." at 6.3s shows as ending at 7.3s.

**Root cause:** `max_len == 0` (the default when `-ml` isn't passed)
was treated as "no word packing" even though `split_on_punct` DOES
need word-level timestamps for accurate splitting.

**Fix:** `(max_len == 0 && !split_on_punct)` — only skip word packing
when neither max_len nor split_on_punct is requested.

**Second bug in the same path:** The flush happened AFTER updating
`cur.t1 = w.t1`, so the flushed sentence included the NEXT word's
end time. Moved flush to before the update.

**Lesson:** When two features interact (max_len + split_on_punct),
test all four combinations: (0,false), (0,true), (N,false), (N,true).
The (0,true) case was never tested and silently degraded accuracy.

### GLM-ASR-Nano: partial RoPE is non-negotiable

GLM-ASR-Nano uses `partial_rotary_factor = 0.5`, meaning RoPE is
applied to only the first half of each attention head's dimensions
(32 out of 64). Applying full RoPE (to all 64 dims) produces encoder
outputs that are ~30% off from the reference — close enough to load
and run, but too divergent for correct transcription.

**Implementation:** Split Q/K tensors along head_dim via `ggml_view_3d`,
apply `ggml_rope_ext` to the first-half view, concatenate back with
`ggml_concat`. This can't use `encoder_self_attn()` (which assumes
full RoPE), so the attention is implemented inline.

**Lesson:** Always check `partial_rotary_factor` in the config before
using RoPE helpers. If it's not 1.0, split-apply-concat is required.
The same pattern appears in Gemma, Phi, and other recent architectures.

### GLM-ASR-Nano: stride-2 conv length is floor(T/2), not ceil(T/2)

GLM-ASR's encoder stem uses `ggml_conv_1d` with `k=3, s=2, p=1`, then
immediately reshapes the result to `(T_enc, d)`. I initially used
`T_enc = (T_mel + 1) / 2`, which matches the textbook convolution size
formula for this kernel setup, but not the actual ggml tensor layout in
the unbatched `(T, C)` path used here.

On odd `T_mel`, ggml produced `floor(T_mel / 2)` frames, so the reshape
asked for one frame too many and hit:

`GGML_ASSERT(ggml_nelements(a) == ne0*ne1)`

This showed up immediately on real GLM-ASR inference with odd-length mel
sequences, while even-length samples hid the bug.

**Fix:** Use `T_enc = T_mel / 2` consistently in both the encoder graph
builder and the output-shape calculation in `glm_asr_run_encoder()`.

**Lesson:** For ggml conv outputs, trust the runtime tensor shape or a
known-good in-repo precedent over the paper formula, especially when the
input is using an implicit unbatched layout.

### FFT size must be power of 2 for radix-2

`core_mel::compute()` calls `fft(data, n_fft, output)` where `n_fft`
may not be a power of 2 (whisper uses 400). A radix-2 Cooley-Tukey
FFT requires power-of-2 input — passing 400 corrupts memory via
bit-reversal permutation on a non-power-of-2 array.

**Fix:** Zero-pad to the next power of 2 (400→512) inside the FFT
function, then truncate the output back to N bins.

### KV cache: no_alloc=true is mandatory for scheduler

The `ggml_backend_sched` requires all tensor contexts referenced in
the graph to have `no_alloc=true`. Creating the KV cache context with
`no_alloc=false` + `ggml_backend_alloc_ctx_tensors()` causes an
assertion failure in `ggml_backend_sched_alloc_graph`.

**Fix:** Use `no_alloc=true` context + manual `ggml_backend_alloc_buffer`
+ `ggml_backend_tensor_alloc` (matching voxtral's pattern). Also call
`ggml_backend_sched_set_tensor_backend` for KV tensors before graph
allocation.

---

## Windows / MSVC portability (April 2026)

### `M_PI` is not defined on MSVC by default

`<cmath>` under MSVC does not expose `M_PI` unless `_USE_MATH_DEFINES`
is `#define`d *before* the header is included. POSIX toolchains
(glibc, libc++ on macOS) leak it through by default, so code that
relies on `M_PI` builds cleanly on Linux and macOS and then fails on
Windows with:

```
error C2065: 'M_PI': undeclared identifier
```

This bit `src/glm_asr.cpp` (the Cooley-Tukey FFT butterfly uses
`-2 * M_PI / len`) — the rest of the codebase had already standardised
on `core/mel.h`'s FFT helpers, which avoid `M_PI` internally, so the
issue was invisible until glm-asr landed its own inline FFT.

**Fix pattern, applied at the very top of any TU that uses `M_PI`:**

```cpp
#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
```

The redundant `#ifndef` guard covers the case where someone further
down the include graph has already pulled in `<cmath>` before the
define (harmless on POSIX; a no-op on MSVC where the guard fires).

**Lesson:** Every new `src/` TU that touches trigonometry on its own
(rather than going through `core/mel.h`) needs this three-line
preamble. Consider banning direct `M_PI` use in code review — pulling
FFT/trig through `core_mel` is portable for free.

### Vulkan first-run latency: the 13-second "Vulkan is slow" illusion

Initial measurement on a hybrid-GPU laptop (Intel Iris Xe + NVIDIA
RTX A1000) reported Vulkan at **0.5× realtime** on parakeet Q4_K /
jfk.wav vs CUDA at 10× RT — a 20× gap that looked like Vulkan being
hopeless for ASR. Diagnosis was initially directed at device
selection (maybe it's on the Intel iGPU?). **Wrong.**

`ggml_backend_init_best()` already prefers `GGML_BACKEND_DEVICE_TYPE_GPU`
over `_IGPU` — the NVIDIA dGPU was correctly selected. The real cost
was **first-run pipeline compilation**: SPIR-V → native GPU ISA for
~50-100 compute pipelines happens lazily on first dispatch, and
ggml-vulkan passed `VK_NULL_HANDLE` to every
`device->device.createComputePipeline(…)` call, meaning **no
VkPipelineCache was used at all**. Subsequent runs only appeared
"fast" because the NVIDIA driver has its own per-shader disk cache
(`%LOCALAPPDATA%\NVIDIA\GLCache`) that catches the miss one level
down. Wipe that cache and every Vulkan run is 13+ seconds again,
permanently.

**Fix** (`ggml/src/ggml-vulkan/ggml-vulkan.cpp`): added a persistent
`vk::PipelineCache` on `vk_device_struct`, keyed by
`vendor:device:driverVersion`, stored under `$LOCALAPPDATA\ggml\vulkan_pipeline_cache\`
(Windows) / `$XDG_CACHE_HOME/ggml/vulkan_pipeline_cache/` (Linux) /
`~/Library/Caches/ggml/vulkan_pipeline_cache/` (macOS). Loaded at
device init, passed to every `createComputePipeline` call, flushed
to disk every 4 new pipelines (counter on the device struct).

Flushing in the destructor alone is not sufficient on Windows: we
call `_Exit(0)` from the CLI (see "Process exit hang" memory entry)
to sidestep a Vulkan static-destructor stall, which also bypasses
`~vk_device_struct()`. Periodic save inside
`ggml_pipeline_request_descriptor_sets` / pipeline-creation covers
this without any new public API.

**Results** (parakeet Q4_K / jfk.wav, 11 s audio, same laptop):

| Scenario | Transcribe time | RTFx |
|---|---:|---:|
| Vulkan cold (no caches) | 13.69 s | 0.8× |
| Vulkan, only our ggml cache warm (NVIDIA GLCache wiped) | **1.34 s** | **8.2×** |
| Vulkan, both caches warm | **0.64 s** | **17.1×** |
| CUDA baseline | 1.21 s | 9.1× |

Warm Vulkan now **beats** CUDA on this laptop (0.64 s vs 1.21 s —
likely because NV_coopmat2 matmul kernels in ggml-vulkan are
better-tuned for this shape than the CUDA path's `cublasGemmEx`
call), and cold-run latency is now a one-time cost per install
rather than per run. Disable via `GGML_VK_DISABLE_PIPELINE_CACHE=1`;
inspect with `GGML_VK_PIPELINE_CACHE_DEBUG=1`.

**Lessons:**

1. When benchmarking GPU backends, **always run the target path
   twice** and report both cold and warm numbers. "Vulkan is 20×
   slower" was a first-run artifact that would have survived code
   review unchanged if we'd trusted the single measurement.
2. Shader native-compilation caching is **not** a driver-only
   concern. Every Vulkan application that loads the same shaders
   repeatedly should pass a `VkPipelineCache` to
   `vkCreateComputePipelines` / `vkCreateGraphicsPipelines` and
   persist it across runs. ggml-vulkan didn't, upstream — our
   patch should probably be submitted.
3. `_Exit()` bypasses destructors. Any caching scheme that only
   flushes in a destructor will silently lose its work on Windows
   builds that call `_Exit`. Periodic incremental save from the
   hot path (throttled) is a simple workaround that doesn't need
   new shutdown hooks.

### Issue #12 (prebuilt binary: silent exit after "using cached")

Reported against a prebuilt release binary on Windows 11 / Intel i3
with no NVIDIA GPU. User sees `crispasr: using cached …` and then
the process returns to the shell prompt — no `parakeet: vocab=…`,
no error, no crash dialog.

**Could not reproduce** at HEAD with a fresh local build on Windows
11 (with NVIDIA GPU). The same `parakeet-tdt-0.6b-v3-q4_k` command
transcribes correctly. Deliberately-corrupt cache files (1 KB
truncated, empty) all produce **loud** errors:

```
gguf_init_from_file_ptr: failed to read key-value pairs
core_gguf: failed to open '…' for metadata read
parakeet: failed to load '…'
crispasr[parakeet]: failed to load model '…'
crispasr: error: failed to initialise backend 'parakeet'
```

with exit code 13. So a partial-download cache file is not the cause.

**The giveaway in the reporter's log:** no `ggml_cuda_init: …` line,
which we always print at startup as long as `ggml-cuda.dll` loads
successfully (regardless of `--no-gpu`). On a machine with no
NVIDIA driver installed, `ggml-cuda.dll` depends transitively on
`cudart64_*.dll` / `cublas64_*.dll`. If those are missing, Windows
fails the DLL load. The backend registry might still swallow the
error and let the exe run on CPU — but depending on the loader
state, a later *deferred-bind* resolve can exit the process with
code `0xc0000135` / `STATUS_DLL_NOT_FOUND` with no stderr output at
all. That matches the reporter's symptom.

**Remediations to consider (none shipped yet):**
1. Ship a **CPU-only** prebuilt alongside the CUDA build for users
   without NVIDIA drivers. `build-windows.bat -DGGML_CUDA=OFF`
   produces a binary with no CUDA dependency.
2. Delay-load `ggml-cuda.dll` via `/DELAYLOAD:ggml-cuda.dll` + a
   `__HrLoadAllImportsForDll` guard, so a missing runtime falls
   back to CPU instead of exiting the process.
3. At startup, call `SetErrorMode(SEM_FAILCRITICALERRORS)` and log
   `GetLastError()` on any DLL resolve failure so the user sees
   *why* the process stopped.
4. Add a `--diagnose` subcommand that prints loaded backends,
   device list, and cache dir — one-line "is my install broken"
   check for end-users.

**Lesson:** A Windows process can exit **completely silently**
when a delay-loaded or transitively-required DLL is missing. Any
"prints one line then disappears" bug report on Windows should
first be diagnosed by (a) checking Event Viewer →
`Application` for a `Faulting module name` crash log, and (b)
running the binary against `Dependencies.exe` or
`dumpbin /dependents` to find the missing import. The codebase
itself is usually fine.

## Kyutai STT: causal padding, interleaved RoPE, and codec-based ASR

### Causal (left-only) padding in conv1d

moshi/Mimi uses `StreamingConv1d` which prepends
`pad_left = kernel_size - stride` zeros to the LEFT before conv1d with
padding=0. Standard symmetric padding produces completely wrong Mimi
encoder output — the SEANet outputs are numerically different and the
RVQ codes cascade to garbage.

**Fix:** `ggml_pad_ext(x, pad_left, 0, 0, 0, 0, 0, 0, 0)` before
`ggml_conv_1d(weight, x, stride, 0, 1)`. After this fix, SEANet output
was bit-perfect vs the official Python Mimi encoder.

### Interleaved vs NEOX RoPE

Kyutai models use **interleaved** RoPE (`[r0,i0,r1,i1,...]`), which is
`GGML_ROPE_TYPE_NORMAL = 0`. Not the NEOX layout (`[r0,r1,...,i0,i1,...]`)
used by Llama/Mistral/Qwen. Using the wrong RoPE type makes the encoder
transformer output diverge (max diff 0.07) and the LM produce garbage.

**Lesson:** Always check `rope.interleave` in the Python source. The two
layouts are **not** compatible — there's no graceful degradation, just
completely wrong output.

### Initial token IDs

The STT LM uses `text_card` (8000) as the initial text token and `card`
(2048) as the initial audio token — NOT the padding ID (3). These are
"start-of-sequence" tokens at the end of the vocabulary. The moshi.cpp
code: `text_initial_token_id = config.text_card; initial_token_id = config.card`.

### Stage-by-stage diff protocol (applied)

1. SEANet: bit-perfect after causal padding fix (max diff = 0.000000)
2. Encoder transformer: bit-perfect after RoPE + causal mask fix
3. RVQ codes: 99.3% match (100% codebook-0, FP residual drift on rest)
4. LM: correct "And so, my fellow Americans..." after all fixes

The causal padding bug was invisible at the architecture level — the
shapes were correct, the model ran without errors, but every single
output value was wrong. Only the diff-test protocol caught it.

## FireRedASR: Conformer encoder debugging (April 2026)

### Internal residual in ConformerFeedForward

The `ConformerFeedForward` module has a **hidden internal residual**:
```python
def forward(self, x):
    residual = x
    output = self.net(x)
    output = output + residual  # ← internal!
    return output
```

The Conformer block's macaron residual `0.5*x + 0.5*ffn(x)` expands to:
`0.5*x + 0.5*(net(x) + x) = x + 0.5*net(x)`.

My code was computing `0.5*x + 0.5*net(x)` — missing the `0.5*x` that
comes from the internal residual. The fix changed FFN1 from matching
at 0.3 error to matching at 0.0003.

**Lesson:** Always check `forward()` of ALL modules, not just the
top-level block. Hidden residual connections are easy to miss when
reading the block-level code `out = 0.5*x + 0.5*ffn(x)`.

### Relative positional encoding index formula

The `_rel_shift` operation maps:
`shifted[h, tq, tk] = original[h, tq, T-1-tq+tk]`

NOT `original[h, tq, tq-tk+T-1]` (the sign of `tq-tk` is flipped).
Verified with a T=5 example: `shifted[0,0] = original[0,4]`,
`shifted[0,1] = original[0,5]`, `shifted[1,0] = original[1,3]`.

### Positional encoding center extraction

`RelPositionalEncoding.forward()` extracts the CENTER of the PE table:
`pe[:, Tmax//2 - T + 1 : Tmax//2 + T]` where Tmax=9999.

Taking the FIRST positions (pe[0:2T-1]) gives completely wrong values
and causes the position attention to produce garbage.

### ggml reshape is column-major

`ggml_reshape_2d([T1, T2] → [T2, T1])` reinterprets the same flat
data with ne[0] as the fast dimension. This is NOT the same as
Python's `view(T2, T1)` which reinterprets with the LAST dimension
fastest (row-major). For the `_rel_shift` operation, this means ggml
reshape cannot be used — need CPU-side computation or transposing.

### Hybrid ggml/CPU encoder for relative position attention

When a model requires an operation that ggml can't express natively
(like rel_shift's row-major reshape), split the computation:
- **ggml** for all matrix multiplications (FFN, projections, conv)
- **CPU** only for the unsupported operation (attention scoring)

For FireRedASR: 2 ggml graphs per layer (pre-attention + post-attention)
with CPU attention scoring in between. This gave **20x speedup**
(323s → 16s) over the full-CPU approach, because ggml handles the
O(T*d²) matmuls while CPU only does the O(T²*d) attention scoring.

### Depthwise conv padding: causal vs symmetric

Streaming models (Mimi/Kyutai) use **causal** (left-only) padding:
`pad_left = kernel_size - stride`.

Non-streaming models (FireRedASR Conformer) use **symmetric** padding:
`pad = (kernel_size - 1) / 2` on each side.

Using the wrong padding gives completely wrong conv outputs but no
error — the shapes are the same. Always check the PyTorch Conv1d's
`padding` attribute to determine which type.

### Stage-by-stage protocol results (FireRedASR)

All 6 bugs were found by comparing at each sub-module boundary:

1. FFN1 diverged at residual: found hidden internal residual in
   `ConformerFeedForward.forward()` — `ffn(x) = net(x) + x`
2. MHSA diverged: content-only matched perfectly, position component
   was wrong → found rel_shift formula was inverted (`T-1-tq+tk` not
   `tq-tk+T-1`) AND PE was extracted from wrong offset (first vs center)
3. Conv diverged: found padding was causal (32+0) instead of symmetric
   (16+16) by checking `depthwise_conv.padding` attribute
4. Each fix was verified to bring the sub-module output within 0.002
   of the reference before proceeding to the next

Without the stage-by-stage protocol, these bugs would have been
invisible — the model runs without errors in all cases, just produces
wrong text.

## FireRedVAD: FSMN Conv1d replication

### Manual conv index arithmetic vs PyTorch Conv1d

The FSMN uses lookback and lookahead depthwise Conv1d with specific
padding and dilation. Manually computing `x[t - n*stride]` does NOT
match PyTorch's `Conv1d(padding=P, dilation=D)` because:

1. Conv1d pads BOTH sides, then applies the kernel
2. The output is then trimmed/sliced in the FSMN code
3. Manual indexing skips the padding step entirely

**Fix:** Replicate the EXACT Conv1d operation — pad the input, apply
kernel with stride/dilation, then apply the same trim/slice as Python:
- Lookback: `conv[:,:,:-(N1-1)*S1]` (trim right)
- Lookahead: `F.pad(conv[:,:,N2*S2:], (0, S2))` (skip left, pad right)

### int16 vs float32 fbank scaling

Kaldi-based models (FireRedVAD, FireRedASR) train on int16 audio
input to `kaldi_native_fbank`. The log-mel features differ by a
constant `2*log(32768) ≈ 20.79` vs float32 (-1..1) input. The CMVN
absorbs this offset, but if CMVN was trained on int16 features and
you feed float32 features, the normalization is wrong.

**Fix:** Scale float32 input by 32768 before fbank computation:
`frame[i] = pcm[i] * 32768.0f`

### Decoder n_head mismatch (FireRedLID)

The FireRedLID decoder uses 8 attention heads (`layer_n_head=8`) but
the encoder uses 20 heads (`n_head=20`). The C++ code used the
encoder's n_head for the decoder, producing random language predictions
instead of "en" for English audio.

**Lesson:** Encoder and decoder may have DIFFERENT n_head values. Always
store them separately in the GGUF metadata and read both.
After fix: LID correctly identifies English on JFK audio.

### GGML_NATIVE=ON on CI runners silently ships AVX-512 to AVX2-only laptops

v0.4.10 Windows prebuilts (CPU / CUDA / Vulkan) all silently exited
with code 0 and no stderr output on a consumer AVX2 laptop CPU —
reproducing issue #12's "using cached → nothing" symptom exactly.

**Root cause**: ggml's `GGML_NATIVE` CMake option defaults to `ON`
unless cross-compiling. On the GitHub Actions `windows-latest`
runner (Azure Standard_D4_v3 or similar, typically with AVX-512),
`GGML_NATIVE=ON` detects the host CPU and emits AVX-512 / AVX10
instructions into `ggml-cpu.dll`. The binary then ships to users on
any x86-64 machine and the first AVX-512 instruction triggers
`STATUS_ILLEGAL_INSTRUCTION` (0xc000001d). On Windows, the exception
handler silently terminates the process — **exit code 0, no stderr,
no event-log entry that a casual user would find.**

**Isolation protocol** (used here to pin the bug):
1. Suspect `ggml-cpu.dll` because it's the only binary whose SIMD
   level changes with host-CPU autodetection.
2. Confirm with a file-size diff between a locally-built (known-good)
   DLL and the CI-built one: **42 KB larger on CI** (823 KB vs 780 KB).
   ~42 KB is the right order of magnitude for additional
   VEX-512-encoded instructions across a matmul + cpy kernel set.
3. Swap *only* the CI `ggml-cpu.dll` for the local one in the
   downloaded zip → the whole pipeline works. Put it back → silent exit.

**Fix** (release.yml, every Windows job): pass
`-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON` to
cmake. AVX2 is the right compat baseline — every x86-64 CPU shipped
since ~2013 (Intel Haswell / AMD Excavator) supports it. Users on
older CPUs or those wanting AVX-512 native kernels should build
from source.

**Alternative (not shipped here)**: set
`GGML_CPU_ALL_VARIANTS=ON GGML_BACKEND_DL=ON BUILD_SHARED_LIBS=ON` —
ggml builds one `ggml-cpu-<arch>.dll` per ISA level (x64, sse42,
sandybridge, haswell, skylakex, cannonlake, cascadelake, icelake,
cooperlake, zen4) and dispatches at runtime. Proper solution, but
adds ~10 DLLs to the package and requires `BUILD_SHARED_LIBS=ON`
which conflicts with our static-CPU prebuilt. Worth revisiting for
the CUDA / Vulkan variants since they're already shared-libs.

**Lessons** (in decreasing order of load-bearingness):

1. **Never ship CI-built binaries with `GGML_NATIVE=ON`.** The CI
   runner's CPU is *not* a representative target CPU. Always pin an
   explicit SIMD baseline for release artifacts. This is the #1
   "prebuilt works on my machine but nobody else's" footgun in
   ggml-based projects.

2. Silent SIGILL on Windows looks identical to "the program does
   nothing" — exit 0, no console output, no crash dialog (unless WER
   is configured to show them). It's not until you attach a debugger
   or check `Event Viewer → Windows Logs → Application` for the
   `Faulting module name: ggml-cpu.dll` entry that the real cause
   becomes visible. **Assume silent-exit on Windows is an illegal
   instruction until proven otherwise.**

3. File-size diffs between CI and local builds of the same DLL are
   a *very* strong signal. Same commit + same CMake flags should
   produce byte-sized-identical outputs (modulo timestamps, which
   shouldn't change size). A +42 KB difference in `ggml-cpu.dll`
   was the only clue we had, and it turned out to be the whole
   story.

### CUDA `cublas64_XX.dll` imports `cublasLt64_XX.dll` transitively

v0.4.10 CUDA prebuilt trimmed cublasLt64_12.dll (474 MB) on the
reasoning that `ggml-cuda.dll`'s own PE import table doesn't list
it — only `cublas64_12.dll`, `cudart64_12.dll`, and driver-loader
DLLs (`nvcuda.dll`). The reasoning was wrong: **`cublas64_12.dll`
itself imports `cublasLt64_12.dll`** (verified via PE import scan).
Without cublasLt, Windows fails `crispasr.exe` at load time with
`STATUS_DLL_NOT_FOUND` — same silent-exit symptom as the SIGILL
case above.

Upstream ggml explicitly notes in `ggml-cuda/CMakeLists.txt`:

> As of 12.3.1 CUDA Toolkit for Windows does not offer a static
> cublas library

so there's no side-stepping this via `GGML_STATIC` on Windows.
The cublasLt cost is unavoidable unless you're willing to replace
all ggml's `cublasGemmEx` calls with hand-written CUDA kernels.

**Lesson**: when triaging a Windows "my exe silently exits" bug,
check **transitive** DLL imports, not just the binary you control.
`dumpbin /dependents` / PE import parsing only shows first-order
imports — you need to walk the chain recursively. On this project
the chain was `crispasr.exe → ggml-cuda.dll → cublas64 → cublasLt`.

### Quantized weight dequantization (read_f32_vec)

The hybrid ggml/CPU encoder reads weights into CPU float vectors via
`read_f32_vec`. The original only handled F16→F32. Quantized models
(Q8_0, Q4_K_M, etc.) passed raw quantized bytes to float arrays →
garbage or crash.

**Fix:** Use `ggml_get_type_traits(t->type)->to_float` to dequantize
any type. Also apply to the conv2d subsampling lambda.

### Conv1d kernel=1 stored as 3D blocks quantization

Pointwise Conv1d weights `[out, in, 1]` stored as 3D tensors in GGUF
have `ne[0]=1`, failing the quantizer's row-alignment check (1 % 256 ≠ 0).
~30% of model weights were left unquantized.

**Fix:** Squeeze the kernel dimension in the converter (`t.squeeze()`
when shape has a `1` and name contains `pointwise_conv`). Makes them 2D
`[out, in]` → quantizer can process normally. Saves ~40% at Q2_K.

**Architecture-specific:** Only apply for `firered` architecture. Other
models' 3D conv weights may be actual spatial kernels.

### LID decoder decode length

FireRedLID only needs 1 decode step — the first token after SOS is
the language code. Running full beam search (300 steps, beam=3) wastes
~50x compute. Detect LID models by `odim <= 256` and set `max_len=2`,
`beam_size=1`.

### LID output mapping

The LID model outputs multi-token sequences for dialect languages
(e.g., "zh" then "mandarin" for Mandarin Chinese). Taking only the
first non-special token gives the ISO 639-1 code.

### Layer pruning for LID

Tested removing encoder layers to shrink the LID model. Only keeping
the last 4 of 16 layers (12-15) works for a single English sample,
but fails on multilingual test (0% accuracy). SLERP merging of adjacent
layers also fails. The Conformer encoder layers are too specialized
for simple pruning — unlike Whisper Turbo's decoder-only pruning.

### Q2_K too aggressive for similar languages

Q2_K quantization causes confusion between similar languages
(de→cy, hi→pa, es→gl). Q4_K maintains accuracy. For LID,
Q4_K (544 MB) is the practical minimum; Q2_K (350 MB) is unreliable.

### ECAPA-TDNN LID: fbank mismatch produces "nn" for everything

SpeechBrain's `lang-id-voxlingua107-ecapa` (Apache-2.0, 43 MB, 107 langs)
was trained with `torchaudio.compliance.kaldi.fbank`. Replacing this with
a simple mel fbank (Hamming window, no Kaldi preprocessing) causes the
model to predict "nn" (Norwegian Nynorsk) for ALL inputs — English, Thai,
German, even the model's own Thai test file.

Tested fbank variants that all fail:
- Simple Hamming+FFT (our C++ default)
- Kaldi-style with preemphasis+Povey window (manual Python)
- `kaldi_native_fbank` library (proper Kaldi C++ implementation)

All produce "nn" with ~0.1 confidence = near-random. The model
requires **exact** `torchaudio.compliance.kaldi.fbank` preprocessing.
Our dev machine has a broken torchaudio (missing CUDA libs), preventing
verification.

Note: "nn" as a default/wrong prediction was also seen in early
FireRedLID debugging — may be a common failure mode when fbank
features are in the wrong distribution (the model learned to map
out-of-distribution features to a specific class).

**Status:** ECAPA-TDNN is WIP. Infrastructure built (converter, runtime,
CLI/API integration). Accuracy blocked on fbank compatibility.
Path forward: test on machine with working torchaudio, or use ONNX
export (Xenova/ecapa-voxlingua107 may exist).

### Qwen Omni vs Qwen3-ASR: not worth implementing separately

Qwen2.5-Omni (3B/7B) and Qwen3-Omni (30B MoE) are multimodal models
(audio+vision+text+speech generation). For pure ASR:

- Much larger than Qwen3-ASR (0.6B/1.7B) with no accuracy advantage
- Split GGUF architecture (mmproj + LLM) — incompatible with our monolithic GGUF
- Already supported by llama.cpp's libmtmd
- Thinker-Talker architecture adds complexity with no ASR benefit

**Recommendation:** Stick with Qwen3-ASR for ASR. Omni models are
for multimodal use cases (speech generation, vision, etc.).

### SpeechBrain Conv1d uses reflect padding, not zero padding

SpeechBrain's `Conv1d` wrapper defaults to `padding_mode='reflect'`,
not zero padding. This causes the conv1d output to differ
dramatically at sequence boundaries. For the ECAPA block0 with k=5:
- Zero pad: `out[co, 0] = 45.07` (uses two zero-padded frames)
- Reflect pad: `out[co, 0] = 76.93` (uses reflected input frames)

The 70% difference at the first frame propagates through the network.
After fixing this, block0 output matches Python reference to <0.01.

**Lesson:** Always check the `padding_mode` attribute of Conv1d wrappers.
SpeechBrain, TorchAudio, and PyTorch all have different defaults.

### SpeechBrain skip_transpose flag is critical

SpeechBrain's `Conv1d` and `BatchNorm1d` both have a `skip_transpose`
flag that controls whether they transpose `[N, C, T] ↔ [N, T, C]`
before/after the underlying PyTorch operation. The ECAPA-TDNN model
uses `skip_transpose=True` for both conv and BN, meaning:
- Conv1d operates on `[N, C, T]` (standard temporal convolution)
- BatchNorm1d normalizes over channels (standard)

Without knowing this, one might transpose the input, causing the
conv to operate over channels instead of time (completely wrong).

### ECAPA-TDNN SE-Res2Net debugging status

Block0 (TDNNBlock) output matches Python after fbank + reflect pad fixes.
SE-Res2Net blocks (1-3) still produce different output. Possible causes:
- Res2Net sub-band cumulative connection ordering
- Dilation handling in reflect-padded dilated conv
- SE block global average pooling implementation
- Residual connection arithmetic

The model architecture is complex (8-way channel split, sequential
processing with cumulative additions, squeeze-excitation attention).
Each sub-component needs stage-by-stage comparison.

### Facebook OmniASR-CTC-300M architecture

fairseq2-based, NOT HuggingFace Transformers:
- 7-layer CNN feature extractor: Conv1d(1→512, k=10) + 6× Conv1d(512→512, k=3)
  with LayerNorm + GELU (wav2vec2 pattern, ~320x downsampling)
- Linear(512→1024) dimension projection
- 24 Transformer encoder layers: d=1024, 16 heads, FFN=4096
- Final projection: Linear(1024→9812) CTC head
- SentencePiece tokenizer (9812 tokens)
- 325M params, ~1.3 GB F32
- Input: raw 16kHz PCM (no mel features)
- Apache-2.0, 1600+ languages

### ECAPA-TDNN SE/tdnn2 ordering bug

SpeechBrain's `SERes2NetBlock.forward` processes in order:
  `tdnn1 → res2net → tdnn2 → SE → residual`

Our initial implementation had SE before tdnn2:
  `tdnn1 → res2net → SE → tdnn2 → residual`

The SE block's squeeze (global average pool) operates on the tdnn2
output, not the res2net output. With the wrong order, the SE scale
was computed from the wrong features, causing completely different
final outputs (mean=0.009 in Python vs mean=-0.133 in C++).

**Lesson:** When implementing a complex block with multiple sub-modules,
always verify the execution order from the Python forward() source.
The intuitive order (SE after the "main" processing) was wrong —
SpeechBrain applies a post-projection (tdnn2) before squeeze-excitation.

### ECAPA-TDNN: 43 MB model achieves ~100% on 12-language TTS benchmark

The SpeechBrain ECAPA-TDNN (21M params, 43 MB F16) correctly identifies
all 12 test languages (en, de, fr, es, ja, zh, ko, ru, ar, hi, pt, it)
with p ≥ 0.96 confidence on edge-tts generated samples.

This is dramatically better than FireRedLID (544 MB Q4_K, 83% accuracy)
for common languages, and 13x smaller. For the 25 extra languages
(Chinese dialects) that FireRedLID covers, it remains the only option.

### ggml tensor layout for conv1d input

ggml uses column-major storage. A 2D tensor `[T, C]` has `ne[0]=T, ne[1]=C`.
The flat data layout is `data[c * T + t]` — channels change SLOWER than time.

This is the SAME as our CPU layout `x[c * T + t]`. So when passing CPU arrays
to ggml tensors, NO transpose is needed — just copy directly.

The confusion arises because `ggml_conv_1d(kernel [K,IC,OC], input [T,IC])`
produces output `[T_out, OC]`, and the flat layout of input `data[ic * T + t]`
puts consecutive time steps of the same channel together — which IS what
conv1d processes along.

**Lesson:** For ggml 2D tensors, `ne[0]` is the fast-changing (innermost)
dimension. For `[T, C]`: time changes fastest, channels slowest.
CPU row-major `x[c * T + t]` and ggml column-major `data[c * T + t]`
are the SAME thing — both index as `slower_dim * faster_size + faster_dim`.

### ggml_pad_reflect_1d exists

ggml has `ggml_pad_reflect_1d(ctx, tensor, pad_left, pad_right)` for
reflect padding. Use this instead of ggml_conv_1d's built-in zero padding
when the model expects reflect padding (SpeechBrain default).

### OpenMP for CPU-only models

Adding `#pragma omp parallel for` to the outer loop of conv1d (over output
channels) and batchnorm1d (over channels) gives ~2x speedup on 4 threads
for ECAPA-TDNN. The CMakeLists needs explicit OpenMP linkage:
```cmake
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ecapa-lid PUBLIC OpenMP::OpenMP_CXX)
endif()
```

### CRITICAL: ggml column-major layout = C-style row-major for 2D arrays

This is the most important ggml lesson we keep re-learning:

**ggml 2D tensor `[A, B]`** means `ne[0]=A, ne[1]=B`. The flat data
layout is `data[b * A + a]` — `ne[0]` changes fastest (column-major).

**C/C++ 2D array `x[B][A]`** or `x[b * A + a]` — also has `A` changing
fastest (row-major).

**THEY ARE THE SAME LAYOUT.** For a tensor representing `[C, T]` where
C is channels and T is time:
- ggml: `ne[0]=C, ne[1]=T`, data at `data[t * C + c]` — C fastest
- C++: `x[c * T + t]` — WAIT, this has T fastest, not C!

**This is where it gets confusing.** When we store data as `x[c * T + t]`
in C++, this is a `[C, T]` array where T is the inner (fastest) dimension.
In ggml, this SAME layout corresponds to `ne[0]=T, ne[1]=C` — because
ggml's ne[0] is the fastest dimension.

**Rule of thumb:**
- If C++ stores as `x[outer * inner_size + inner]`
- Then ggml tensor should have `ne[0]=inner_size, ne[1]=outer_size`
- The flat data bytes are identical — just copy, don't transpose!

**For `ggml_conv_1d(kernel [K,IC,OC], input [T,IC])` → `[T_out,OC]`:**
- Input ne[0]=T, ne[1]=IC → flat: `data[ic * T + t]`
- Our C++ `x[c * T + t]` stores channel c at `data[c * T + t]`
- SAME layout → just copy x to ggml tensor directly

**For `ggml_mul_mat(a [C_in,C_out], b [C_in,T])` → `[C_out,T]`:**
- Requires `a.ne[0] == b.ne[0]` (both = C_in)
- Input `b` must be `[C_in, T]` with ne[0]=C_in
- If input is from conv1d `[T, C]` with ne[0]=T, transpose first

**For reading ggml output to C++ array:**
- ggml tensor `[T, C]` (ne[0]=T, ne[1]=C): `data[c * T + t]`
- C++ wants `x[c * T + t]`
- SAME layout → just copy, no transpose!

This caused bugs 3 times in ECAPA-TDNN:
1. Input: incorrectly transposed before feeding to ggml_conv_1d
2. MFA output: incorrectly treated as row-major when reading to CPU
3. build_conv1d_k1: unnecessary transpose of already-correct data

### OmniASR-CTC-300M: first working GGUF conversion

Successfully converted facebook/omniASR-CTC-300M to GGUF (0.65 GB F16,
423 tensors). Model loads, ggml graph computes in 7.7s for 11s audio.
But CTC decode returns empty (all blanks).

Architecture:
- 7-layer CNN: Conv1d strides [5,2,2,2,2,2,2] = 320x downsampling
- Linear(512→1024) projection
- 24 Transformer encoder layers (pre-norm, 16 heads, FFN=4096, GELU)
- CTC head: Linear(1024→9812) with SentencePiece tokenizer

Key: this is fairseq2-based (not HuggingFace), so tensor names differ
from standard wav2vec2. The converter shortens names to fit 64-char GGUF
limit. CNN strides stored as array in GGUF metadata.

The CTC blank = pad_id = 1 (SentencePiece <pad>).

### OmniASR-CTC: three critical findings

1. **Input normalization required**: wav2vec2 models expect `layer_norm(waveform)`
   — zero mean, unit variance. Without this, the model outputs mostly blanks.

2. **CTC blank = `token 0 (<s>)`**: In fairseq2, the BOS token serves as CTC blank.
   NOT token 1 (<pad>) which is the HuggingFace convention.
   The official code just removes consecutive duplicates + skip_special_tokens.

3. **Pos conv padding**: fairseq2 uses `padding = K // 2` (=64 for K=128),
   not `(K-1) // 2` (=63). The extra padding element gives correct same-padding
   for even kernel sizes. Without this, the pos encoding is misaligned by 1 frame.

4. **No language conditioning for CTC**: confirmed from official repo comment
   "It is ignored when performing inference with CTC." The CTC model is
   fully language-agnostic across 1600+ languages.

### OmniASR audio length limit

Official docs: "Currently only audio files shorter than 40 seconds are
accepted for inference." Models trained on ≤30s segments. For longer
audio, use VAD segmentation to split into chunks.

Our implementation doesn't enforce this limit — it will run on longer
audio but quality degrades. The CNN downsampling (320x) means 40s of
16kHz audio = 2000 frames through the transformer, which is within
typical attention window limits.

### fairseq2n native extension

fairseq2's Python package requires `fairseq2n` C++ extension which is
compiled for specific Python/CUDA combos. Not available for Python 3.13
or CPU-only setups via pip. Our manual forward pass serves as reference.

### ECAPA-TDNN: two model variants (VoxLingua107 vs CommonLanguage)

SpeechBrain has two ECAPA-TDNN LID models with different hyperparameters:

| | VoxLingua107 | CommonLanguage |
|---|---|---|
| n_mels | 60 | 80 |
| lin_neurons | 256 | 192 |
| Classifier | DNN (BN→Linear→BN→LeakyReLU→Linear) | Cosine (normalize(emb) @ normalize(weight)) |
| Labels | ISO codes (en, de, ...) | Full names (English, German, ...) |
| Languages | 107 | 45 |

The converter auto-detects these from `hyperparams.yaml` and `classifier.ckpt`
structure, storing `ecapa.cls_type` (0=DNN, 1=cosine) and `ecapa.lin_neurons`
in the GGUF metadata.

**Cosine classifier**: `F.linear(F.normalize(emb), F.normalize(weight))` — each
class output is the cosine similarity between the normalized embedding and the
normalized class weight vector. Scores are in [-1, 1], not softmax probabilities.

### ECAPA-TDNN: quantization destroys accuracy

ECAPA-TDNN cannot be meaningfully quantized. Even Q8_0 produces all-wrong
predictions (always returns "ms" regardless of input). Root causes:

1. **Small conv1d kernels**: The res2net conv weights are `[128, 128, 3]` (49K elements).
   Q8_0 block size 32 doesn't divide K=3, so ggml skips them — but the tdnn1/tdnn2
   weights `[1024, 1024]` ARE quantized, which corrupts the embedding.
2. **ggml_conv_1d + quantized weights**: The conv1d op may not properly dequantize
   weight tensors during computation, producing garbage output.
3. **Cosine classifier sensitivity**: Even small perturbations in the 192-dim
   embedding space flip the argmax due to narrow angular margins between classes.

**Conclusion**: Ship ECAPA-TDNN as F16 only. At 40-43 MB it's small enough
that quantization savings (14 MB Q4_K) aren't worth the accuracy loss.

### OmniASR-CTC: two GGUF formats (fairseq2 vs HF-native)

The fairseq2-converted GGUF (`omniasr-ctc-300m.gguf`) uses tensor names like:
- `cnn.0.ln.weight`, `enc.0.attn_ln.weight`, `enc.0.attn.q_proj.weight`
- `enc.0.ffn.up.weight`, `enc_ln.weight`, `ctc.weight`, `proj.weight`

The HF-native conversion (aadel4/omniASR-CTC-300M-v2) uses wav2vec2 names:
- `cnn.0.norm.weight`, `enc.0.ln1.weight`, `enc.0.attn.q.weight`
- `enc.0.ffn.fc1.weight`, `lm_head.weight`, `feat_proj.weight`

Our omniasr runtime expects the fairseq2 format. The HF-native model can
potentially be used with the existing wav2vec2 backend instead.

### OmniASR-LLM: decoder architecture and language conditioning

The OmniASR-LLM variant adds a 12-layer LLaMA decoder (d=4096, 8 heads,
head_dim=512, SwiGLU FFN with d_ffn=2816). The encoder is identical to CTC.

**Decoder input sequence** (from `create_default_syntax` in model.py):
```
[audio_embeddings...] [lid_marker] [lang_embedding] [BOS] [generated_tokens...]
```

**Special tokens** (from `Wav2Vec2LlamaSpecialTokens`):
- `lid_marker` = vocab_size (9812) — extra entry in text_frontend embedding
- Language ID = index in supported_langs list + 1 (factory.py adds +1, index 0 = no-language)
- BOS = 0, EOS = 2, PAD = 1

**Language ID mapping** (from `factory.py`):
```python
lang_mapping = {row["lang"].lower(): row["index"] + 1 for row in parquet_table}
```
Key indices: eng_Latn=414, deu_Latn=365

**RoPE**: fairseq2 uses interleaved pairing `(x[2i], x[2i+1])` — this maps to
`GGML_ROPE_TYPE_NORMAL` (mode 0), NOT NEOX (mode 2). This differs from most
HuggingFace LLMs which use `rotate_half` (NEOX). Getting this wrong produces
fluent but wrong-language output (Greek in our case).

**v1 vs v2**: Always use v2 models (`omniASR_LLM_300M_v2`). The v2 uses a
different tokenizer (`omniASR_tokenizer_written_v2`, 10288 tokens vs 9812)
and is the only variant that reliably transcribes challenging English audio.
v2 checkpoints available at `dl.fbaipublicfiles.com/mms/omniASR-LLM-300M-v2.pt`.

**Critical bug found**: The LLM converter shortened
`encoder_frontend.post_extract_layer_norm` to `post_extract_ln` but the runtime
code looked for the long name, got nullptr, and silently skipped the LayerNorm.
This caused the projection output to diverge (cos=0.77 vs reference) making
all downstream output garbage. Fix: try both short and long tensor names.

**Before fix**: "it sounded to one and that was a particular pillow..."
**After fix**: "and so my palamericas is not what your country can do for you..."

The reference dump protocol (dump intermediates at each stage, compare cosine
similarity) caught the bug immediately. CNN output was cos=0.999999, but
proj_out diverged to cos=0.767. The fix brought all stages to cos>0.9999.

### OmniASR-LLM: quantization requires skipping bridging tensors

Quantizing the OmniASR-LLM decoder with Q4_K/Q8_0 causes immediate EOS
output (0 generated tokens). Even Q8_0 is broken. Root cause: four
bridging tensors between encoder and decoder are precision-critical:

- `enc_proj.weight` (1024 -> 4096 projection)
- `lm_head.weight` (4096 -> 10288 vocabulary logits)
- `tok_emb.weight` (10289 token embeddings)
- `lang_emb.weight` (1694 language embeddings)

**Fix**: Skip these tensors during quantization (keep as F16). Added to
crispasr-quantize skip rules. With this fix, Q4_K (1.1 GB) produces
identical output to F16 (3.1 GB) — 3x size reduction with no quality loss.

### OmniASR: scheduler must include a CPU fallback backend

OmniASR initialized its ggml scheduler with only the "best" backend:

```cpp
ctx->sched = ggml_backend_sched_new(&ctx->backend, nullptr, 1, ...);
```

That worked until ggml tightened `ggml_backend_sched_new()` and started
asserting that the last backend in the scheduler list is CPU when a GPU
backend is present. On CUDA builds this crashed immediately during
`omniasr_init_from_file()` with:

`GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU)`

**Fix:** Mirror the working pattern used by the other backends: keep a
separate `backend_cpu`, append it to the scheduler backend list when the
main backend is not CPU, and free it separately on shutdown.

### OmniASR: `-ng` has to be plumbed into backend selection explicitly

The OmniASR CLI adapter ignored `whisper_params.use_gpu`, so `-ng` and
`--gpu-backend cpu` still called `ggml_backend_init_best()` and tried to
construct a GPU-first backend stack. After the scheduler fix this no
longer crashed, but the flag semantics were still wrong.

**Fix:** Add `use_gpu` to `omniasr_context_params` and set it from the
CLI adapter, treating `--gpu-backend cpu` the same as `-ng`.

**Lesson:** For the non-whisper backends, GPU selection is not automatic
just because the top-level CLI parsed the flag. Each backend adapter has
to propagate that intent into its own backend picker.
## 2026-04-22 - No-gpu mode must gate `ggml_backend_load_all()`

- Symptom: `-ng --gpu-backend cpu` could still print `ggml_cuda_init: found ...` even after backend-specific code paths stopped using `ggml_backend_init_best()`.
- Root cause: `examples/cli/cli.cpp` called `ggml_backend_load_all()` before parsing CLI flags, and `src/cohere.cpp` also loaded all backends unconditionally. That dynamic registration path probes CUDA as soon as the CUDA backend is loaded.
- Fix: parse CLI args first, then call `ggml_backend_load_all()` only when `params.use_gpu` is true and `params.gpu_backend != "cpu"`. Any backend with its own unconditional `ggml_backend_load_all()` must apply the same guard.
- Result: CPU-forced runs stop triggering global CUDA discovery just to select a CPU backend.

## 2026-04-23 - FireRed decoder optimization triage

### What actually helped

On the FireRed AED decoder, the wins came from moving **large, reused**
decoder matmuls onto ggml/GPU and from removing unnecessary beam work:

- Copy-on-write beam KV history instead of deep-copying `sa_k` / `sa_v`
  on every beam fork
- Dedicated greedy path for `beam_size == 1`
- Removing unused log-softmax bookkeeping from the greedy path
- Moving cross-attention encoder-side `K/V` precompute onto ggml/GPU
- Moving final decoder vocab projection onto ggml/GPU

Measured on `issue19-5s.wav` (`-t 8 -l en`) on the RTX A1000 laptop:

- Original baseline, `-bs 1`: `26.86s`
- Current best, `-bs 1`: `8.68s`
- Original baseline, `-bs 3`: `29.59s`
- Current best, `-bs 3`: `19.02s`

So the current FireRed decoder is about:

- `3.1x` faster for greedy decode
- `1.56x` faster for beam size 3

### What did not help

Several intuitive CPU-side micro-optimizations were regressions and were
reverted:

- Per-call scratch-buffer reuse inside the decoder loop
- Streaming logsumexp / top-k rewrite for the vocab projection
- Parallel `gemv` helper for small decoder vector-by-matrix products
- Per-call ggml graphs for small decoder MLP projections

The common pattern: **small per-step graphs or tiny parallel regions lose
to their own launch/alloc/scheduling overhead**. The decoder only speeds
up when the moved work is both substantial and reused.

### FireRed decoder strategy going forward

The remaining useful path is **larger persistent decoder subgraphs**, not
more loop-level CPU tuning. In particular:

1. Keep shared heavy decoder work on ggml/GPU (`K/V` precompute, logits).
2. Avoid one-graph-per-small-matmul designs.
3. Next meaningful step is a reused greedy decoder subgraph per layer or
   per token step, not isolated ggml calls for single projections.

### Data2Vec / HuBERT: three architecture traps when reusing wav2vec2 backend

Data2Vec and HuBERT share wav2vec2's CNN frontend + transformer encoder +
CTC head, but differ in three subtle ways that each cause complete failure
if wrong:

1. **Multi-layer positional convolution**: Data2Vec has 5 layers of
   `Conv1d(K=19, groups=16) + LayerNorm(no_affine) + GELU`, not 1 layer
   like wav2vec2. Only storing the first layer's weights causes the pos_conv
   output to diverge entirely (cos=0.08). Fix: store all N layers in GGUF
   as `pos_conv.{i}.weight/bias` and run them sequentially in C++.

2. **Global encoder LN placement**: Data2Vec applies the global encoder
   LayerNorm **BEFORE** the transformer layers, then uses post-norm inside
   each layer. wav2vec2 and HuBERT apply the global LN **AFTER** all layers.
   This is unique to Data2Vec and requires a separate flag
   (`global_ln_before_encoder=1`). Getting it wrong amplifies logits ~46x.

3. **Post-norm despite LayerNorm CNN**: Data2Vec uses LayerNorm in ALL CNN
   layers (like HuBERT) but uses **post-norm** in the encoder (unlike HuBERT
   which uses pre-norm). The encoder layer does `attn→add→LN→FFN+add→LN`.
   Setting `do_stable_layer_norm=1` (pre-norm) produces complete garbage.

Each bug was caught by systematic stage-by-stage diff against Python ref:
- CNN output: cos=0.999997 (correct from the start)
- feat_proj: cos=0.999968 (correct)
- pos_conv layers 0-4: cos>0.999961 (after multi-layer fix)
- after_global_ln: cos=0.999946 (after LN placement fix)
- **logits: cos=0.999972** (after post-norm fix)
- C++ decode matches Python exactly: "AND SO A MY FELLOW AMERICANS..."

### VibeVoice-ASR-1.5B: σ-VAE + Qwen2 hybrid architecture

VibeVoice uses a novel pipeline: two parallel σ-VAE CNN encoders (acoustic +
semantic) → linear connectors → Qwen2-1.5B autoregressive decoder.

**Key architecture findings:**
1. **Encoders are ConvNeXt-style**: 7 stages of `downsample_conv → N × Block1D`.
   Block1D = `RMSNorm → depthwise_conv → gamma_scale → residual + RMSNorm → FFN → gamma_scale → residual`.
2. **Depthwise conv via ggml_conv_1d_dw**: works but forces F16 im2col
   internally, causing cumulative precision loss (cos=0.7 after 29 blocks;
   Python F16 gives cos=0.999, so it's ggml-specific).
3. **Causal padding**: `padding_total = (K-1)*dilation - (stride-1)`, NOT `K-1`.
   Plus `get_extra_padding_for_conv1d` for stride alignment on the right side.
4. **Connectors**: simple `FC1 → RMSNorm → FC2` (NO activation, no SiLU).
5. **Scaling factors**: `speech_scaling_factor/speech_bias_factor` are for the
   base TTS model, NOT the ASR variant. ASR uses raw features directly.
6. **σ-VAE sampling**: ASR calls `.sample(dist_type='gaussian')` which adds
   noise. For deterministic C++ inference, using `.mean` (mode) is fine.
7. **Prompt template**: Qwen2 chat format with `<|object_ref_start|>` as
   speech_start, `<|box_start|>` as speech_pad, `<|object_ref_end|>` as
   speech_end. These repurpose existing Qwen2 special tokens.
8. **Qwen2 Q/K bias**: unlike most LLMs, Qwen2 has bias on Q and K projections
   (but not V and O). The `core_attn::kv_self_attn` helper doesn't support
   per-projection biases — needs inline attention implementation.

**For decoding (tokens → text)**: no tiktoken/BPE library needed. Just embed the
151665-token Qwen2 vocab as `tokenizer.ggml.tokens` in the GGUF and do
`vocab[token_id]` lookup. BPE merge rules are only needed for encoding
(text → tokens), which we don't do for ASR inference.

**ggml precision issue**: `ggml_conv_1d_dw` (line 4494 in ggml.c) creates
im2col with `GGML_TYPE_F16` regardless of input type. Through 29 ConvNeXt
blocks, each with a depthwise conv, this accumulates precision loss. Fix options:
1. CPU depthwise conv (simple loop, avoids im2col entirely)
2. Modify ggml to use F32 im2col when input is F32
3. Accept lower precision and rely on LM decoder robustness

### VibeVoice decoder: systematic debugging status

The Qwen2 decoder consistently outputs `<|vision_pad|>` (token 151654)
regardless of input. Confirmed NOT an encoder issue — injecting Python
reference features (cos=1.0) produces the same wrong output.

**Verified correct:**
- Embedding tensor values match Python checkpoint
- Prompt template matches processor output (143 tokens + assistant prefix)
- LM head uses tied weights (lm.tok_emb.weight)
- Embedding layout: data[token_id * d_lm + dim] (ggml column-major)

**Likely causes (in order):**
1. **RoPE theta**: Qwen2 uses theta=1000000.0 (not 10000). Our code sets this
   but the actual ggml_rope_ext call might interpret it differently.
2. **GQA native mode**: with n_heads=12, n_kv_heads=2, GQA ratio=6:1.
   The flash_attn_ext native GQA mode might handle this wrong for Qwen2.
3. **Causal mask**: the mask construction might be wrong for the prefix-fill case.
4. **Q/K bias interaction with RoPE**: bias is added before RoPE, which is correct
   in Python but might interact differently with ggml_rope_ext.

**Critical discovery**: the standalone Qwen2 decoder (loaded from VibeVoice
weights) produces `<|vision_pad|>` even in Python with correct features.
The full VibeVoice forward pass (`model.generate` with internal encode_speech)
is required — the speech features get special handling inside the model's
forward method that a standalone Qwen2 forward doesn't replicate.

**Model variant concern**: `microsoft/VibeVoice-1.5B` might not be the primary
ASR model. The documented ASR model is `microsoft/VibeVoice-ASR` (7B) or
`microsoft/VibeVoice-ASR-HF`. The 1.5B variant produced garbage on 2s of
mostly-silent audio. Need to verify with actual speech content before further
C++ debugging.

**CRITICAL**: `microsoft/VibeVoice-1.5B` is a **TTS model**, NOT ASR!
The HF model card explicitly says "Use to generate any text transcript"
is OUT OF SCOPE. We were using the WRONG model variant.

The correct ASR model is `microsoft/VibeVoice-ASR` (7B):
- Architecture: `VibeVoiceForASRTraining`
- Decoder: d=3584, 28 layers, 28 heads, 4 KV heads (bigger than 1.5B TTS)
- Same encoder: vae_dim=64, ratios=[8,5,5,4,2,2]
- Vocab: 152064 (slightly different from 1.5B's 151936)

Our C++ pipeline (encoder + connectors + Qwen2 decoder) has the right
architecture — just needs the correct 7B ASR weights. The converter
handles different decoder dimensions automatically.

---

## FireRedPunc / fullstop-punc — BERT punctuation restoration (April 2026)

### Architecture
Two punctuation models implemented as post-processors:

| Model | Base | Layers | d_model | Heads | Vocab | Labels | Tokenizer |
|---|---|---|---|---|---|---|---|
| FireRedPunc | BERT (LERT) | 12 | 768 | 12 | 21,128 | 5 (space/，/。/？/！) | WordPiece |
| fullstop-punc | XLM-RoBERTa-large | 24 | 1024 | 16 | 250,002 | 6 (space/./,/?/-/:) | SentencePiece |

Both are token classifiers: BERT/RoBERTa encoder → Linear(d, n_classes).
ggml graph uses `ggml_flash_attn_ext` for multi-head attention.

### Bugs found and fixed

**1. Missing SEP token (critical)**
BERT and RoBERTa both expect `[CLS] tokens [SEP]` as input. Our code
only prepended CLS (`seq_len = N + 1`), never appending SEP. This caused
completely wrong logits — the model was trained with SEP and its absence
shifted the attention patterns.

Fix: `seq_len = N + 2`, `ids[N+1] = SEP_id` (102 for BERT, 2 for RoBERTa).

Symptom: logits ~1-2 points off from reference, commas placed on wrong
words. Python F16 still predicted correctly — ruling out precision as
the cause. The diff-testing methodology (stage-by-stage comparison with
Python reference) quickly identified this: embeddings matched perfectly
(cos>0.999) but final logits diverged, pointing to a systematic error in
the self-attention computation that only manifests with a wrong sequence
structure.

**2. RoBERTa position ID offset**
RoBERTa position embeddings have `padding_idx=1`. Position 0 is for
`<pad>`, position 1 is zeroed out (the padding index), and actual content
starts at position 2. Our code used `pos_ids = [0, 1, 2, ...]` (BERT
convention) instead of `pos_ids = [2, 3, 4, ...]` (RoBERTa convention).

Fix: `pos[i] = i + padding_idx + 1` when `is_sentencepiece = true`.

Symptom: logits completely wrong (class 0 predicted for all tokens).
Diagnosed by comparing embedding output at position 15 — the values
were off because wrong position embeddings were summed.

**3. SentencePiece subtoken counting mismatch**
The text reconstruction code re-tokenizes each word to count how many
subtokens it consumed, mapping prediction indices back to words.
For SentencePiece, words are prefixed with `▁` (U+2581), not `##`.
The code was using WordPiece `##`-prefix matching for SentencePiece
tokens, causing wrong subtoken counts and shifted punctuation placement.

Fix: Separate SentencePiece path that prefixes with `▁` and does
greedy longest-match in the SentencePiece vocab.

Symptom: comma placed on "can" instead of "you" — the subtoken count
for "americans" (split into ["▁american", "s"] = 2 tokens) was counted
as 1 with the WordPiece path, shifting all subsequent predictions by 1.

**4. Chinese full-width punctuation for English text**
FireRedPunc was trained on Chinese data and outputs full-width marks
(`，` `。` `？` `！`) even for English input.

Fix: Auto-detect Latin script (count Latin vs CJK characters), map
full-width to ASCII when Latin dominates. Simple 4-replacement post-step.

### Methodology lesson reinforced

The user correctly pushed back when I blamed "F16 precision loss" for wrong
punctuation placement. The actual bug (missing SEP token) was a computation
error, not a precision issue. **Python F16 still predicted correctly** —
this ruled out precision as the root cause.

The diff-testing protocol worked exactly as designed:
1. Dump Python reference (logits, embeddings, per-layer outputs)
2. Dump C++ intermediates at the same positions
3. Compare cosine similarity at each stage
4. Embeddings matched (cos>0.999) → bug is after embeddings
5. Final logits diverged (cos~0.93) → systematic error in transformer
6. Traced to missing SEP token in input construction

Key principle: **when Python F16 works but C++ F16 doesn't, it's NOT a
precision issue.** Look for structural bugs (wrong input construction,
missing tokens, wrong tensor shapes).

### Quantization notes

| Model | F16 | Q8_0 | Q4_K | Accuracy |
|---|---|---|---|---|
| FireRedPunc | 195 MB | 104 MB | 56 MB | Q8_0 = F16 exact; Q4_K drops some commas |
| fullstop-punc | 1.6 GB | 572 MB | 254 MB | All quants identical on JFK test |

FireRedPunc Q4_K is more sensitive because BERT-base (12L, d=768) has
less redundancy than XLM-RoBERTa-large (24L, d=1024). Recommend Q8_0
for FireRedPunc, Q4_K for fullstop-punc.

### Progressive SRT output (issue #24)

Non-whisper backends buffered all segments before printing. Added
`--flush-after N` flag: when N=1, each SRT entry is flushed to stdout
as soon as its VAD slice finishes transcription. Post-processing (punc
model, punctuation stripping) runs per-slice.

Limitation: diarization needs full segment context — skip when
`--flush-after` is set. Word-level alignment (`-am`) works per-slice.

### Session API expansion

Added 5 missing backends to the C-ABI session API: glm-asr, kyutai-stt,
firered-asr, moonshine, omniasr. Pattern: `#ifdef CA_HAVE_*` guards in
`crispasr_c_api.cpp` for open/transcribe/close. All backends now
reachable from Python (`crispasr.Session`), Rust (`crispasr::Session`),
and Dart (`CrispasrSession`).

---

## TTS / Vocoder (Chatterbox HiFTGenerator)

### iSTFT data access must respect ggml's ne[0]-fast layout

The single bug that took the Chatterbox vocoder from "Oh." to "Hello
world." was a **transposed data read in the iSTFT**:

```cpp
// WRONG: treats buffer as (T_slow, C_fast) row-major
float raw_mag = stft_data[frame * C_stft + f];

// CORRECT: ggml stores ne[0]=T (fast), ne[1]=C (slow) → data[c * T + t]
float raw_mag = stft_data[f * T_stft + frame];
```

The ggml graph produces the correct STFT tensor (cos=1.0 vs Python
reference at every stage up to and including conv_post). But the CPU
iSTFT loop that converts STFT→waveform was reading the flat float buffer
with frequency and time axes swapped. This mixed time-step values into
frequency bins, producing a garbled waveform that still sounded
speech-like but was unintelligible to ASR.

**Signature of the bug**: the ggml graph stages all match the Python
reference, the STFT range/RMS look reasonable, but the audio transcript
is wrong. Always check that CPU post-processing code uses the same
`data[c * T + t]` indexing as ggml tensors — don't assume row-major
`data[t * C + c]`.

### Missing ReflectionPad1d at last upsample stage

Python's HiFTGenerator applies `ReflectionPad1d((1, 0))` inside the
upsample loop at `i == num_upsamples - 1` (the last stage), before
source fusion and resblocks. This pads 1 sample on the left by
reflecting `x[:, :, 1]`. Without it, conv_post is shifted by one
timestep and the STFT output has a cos_min ≈ 0.98 against the reference
(all prior stages at 1.0).

Implementation in ggml:
```cpp
ggml_tensor* pad = ggml_view_2d(ctx, x, 1, C, x->nb[1], 1 * x->nb[0]);
pad = ggml_cont(ctx, pad);
x = ggml_concat(ctx, pad, x, 0);  // concat along time axis (ne[0])
```

### Source fusion biases contaminate zero-input diff tests

When diff-testing the vocoder against a Python reference captured
**without** source fusion, don't just zero the source STFT input — skip
the source fusion graph path entirely. Even with all-zero input,
`Conv1d(zeros) = bias ≠ 0`, and those biases propagate through the
source resblocks and add a non-trivial signal to x. In our case this
dropped per-stage cosine from 1.0 to 0.92, masking the real bug
location until source fusion was fully disabled.

### Per-stage diff protocol for vocoders

The crispasr-diff protocol (dump Python activations to GGUF, compare C++
stage-by-stage) works for vocoders just as well as for encoders. The
stages to capture for HiFTGenerator:

```
voc_conv_pre → voc_ups_0 → voc_rb_0 → voc_ups_1 → voc_rb_1 → voc_ups_2 → voc_rb_2 → voc_conv_post
```

Mark each with `ggml_set_name` + `ggml_set_output`, read back via
`ggml_graph_get_tensor` after compute. Note the ggml↔GGUF layout
mismatch: C++ tensors have ne[0]=T (fast), but Python GGUF writers
may store ne[0]=C (fast) depending on how the numpy array was shaped.
Always verify layout by printing the first few values from both sides
before trusting cosine metrics.

### ggml conv_1d, Snake activation, and ResBlock structure are correct

After extensive testing: `ggml_conv_1d` with dilation handles all
resblock convolutions correctly. The Snake activation `x + sin²(αx)/α`
via `ggml_mul → ggml_sin → ggml_mul → ggml_div → ggml_add` matches
PyTorch to 6 significant figures. The 3-independent-resblock-then-average
pattern works. The bugs were downstream of the ggml graph, not in it.

## GitHub Actions workflow triggers — `master` → `main` rename gotcha (May 2026)

After renaming the default branch from `master` to `main`, audit
**every** workflow's trigger config for stale `branches: master`
references. A workflow gated on a non-existent branch never fires
on push events. GitHub Actions does not warn — the workflow file
is valid, runs are just never created. Three CrispASR workflows
(`bindings-ruby.yml`, `build.yml`, `examples-wasm.yml`) had this
misconfig for months post-rename. The full per-OS build matrix
(iOS / macOS / Windows MSVC / Windows MSYS2 / Linux Vulkan /
Android NDK) was therefore not validating any push to `main` —
only running on PRs (different trigger) and on tag pushes
(release-only).

### Why it took so long to notice

- PRs always triggered the workflows (different trigger config),
  so PR review felt fine.
- Tags (`tags: 'v*'`) still triggered, so v0.5.x releases ran the
  matrix — but only at release time, not per-merge.
- `gh run list --workflow=build.yml` shows runs starting on PR
  opens and at tag pushes, looking superficially complete. The
  missing signal — "no run on commit X to main" — only becomes
  apparent if you specifically query for it.

### How a pre-existing failure surfaces as a "PR failure"

When `Bindings Tests (Ruby)` finally ran on a PR (#57,
`feat/codec-gpu-env`), it failed at the `cmake-targets` step. The
natural reading is "the PR broke something." The actual cause: that
PR was the first PR in months to fire a workflow that had been
silently dead on main. The breakage was in main, not the PR. We
proved this by fixing the trigger and pushing the trigger fix
itself to main — the resulting clean-main run reproduced the
identical `Makefile:171: Error 2`.

### Audit recipe

```bash
for f in .github/workflows/*.yml; do
    echo "== $(basename "$f") =="
    grep -A1 "branches:" "$f" | head -3
done | grep -B1 master
```

Anything that prints `master` after a `branches:` line is a stale
trigger. Fix `master` → `main` (or list both: `branches: [main, master]`).

### Incident-response value

Land the trigger fix in a single small commit (`b553546` for us).
That gives a clean before/after demarcation: every run before this
commit on main is suspect for "did it actually run?", and every
run after this commit is the new ground truth. Push the trigger
fix on its own — its own push fires the corrected workflow on the
new SHA, so any accumulated breakage surfaces on the same commit.

### What surfaced when the triggers came back

Three real breakages had accumulated under the silent triggers:

1. **build.yml ubuntu-22-clang ×4** — clang-14 (Ubuntu 22.04 baseline)
   refuses to capture C++17 structured-binding names in a `[&]`
   lambda. gcc accepts it as an extension; clang rejects with
   "reference to local binding 'qw' declared in enclosing function."
   Fixed in C++20 (P1091) but our `cxx_std_11`-effective-C++17
   baseline doesn't get that. **Fix:** plain locals + `const auto&`
   aliases instead of `auto [a, b] = ...` whenever the names will be
   captured by a lambda. Commit `053f41f`.

2. **build.yml ios-xcode-build** — `build-xcframework.sh` was
   hand-listing only 5 ggml libs + libcrispasr.a in the libs[] array
   passed to `libtool -static -o combined.a`. CrispASR has 25+
   STATIC backend libs that crispasr.a publicly depends on but does
   NOT statically embed (parakeet, canary, voxtral, qwen3_asr,
   wav2vec2-ggml, glm-asr, …). Every backend referenced from
   `crispasr_c_api.cpp` would surface as "Undefined symbols"
   eventually; wav2vec2_load was just the first. **Fix:** glob
   `build_dir/src/*/release_dir/*.a` and append. Future-proof
   against new backends. Commit `ee580f8`.

3. **bindings-ruby.yml — `whisper` is an ALIAS target**. The Ruby
   gem's `extconf.rb` ran `cmake --build build --target common
   whisper` for years. After the upstream rename to crispasr,
   `whisper` became an `add_library(whisper ALIAS crispasr)` for
   back-compat in callers that hardcoded the name. **ALIAS targets
   are linkable but not directly buildable** — `cmake --build
   --target whisper` fails with "make: *** No rule to make target
   'whisper'." (CMake forwards the target name to the underlying
   build tool, which only sees the real `crispasr` rule.) **Fix:**
   point the gem at the real target: `--target common crispasr`.
   Commit `476c655`.

The first two were pure code drift (compiler versions / new backend
sources added without updating the xcframework lib list). The
third was a rename gotcha. None would have made it past CI if
build.yml + bindings-ruby.yml had been firing on main pushes.

### Workflow auto-discovery — `actions/configure-pages` failure mode

The `Examples WASM` workflow's whole purpose is `actions/upload-pages-artifact`
+ `actions/deploy-pages` to a GitHub Pages site. If Pages isn't
enabled on the repo (Settings → Pages → "GitHub Actions" source),
`actions/configure-pages` fails with:

```
##[error]Get Pages site failed. Please verify that the repository
has Pages enabled... HttpError: Not Found
```

The build matrix in `build.yml` already validates that the WASM
examples *compile* — that's the per-merge signal we want. The
deploy is incidental and shouldn't gate main-push CI. Default the
trigger to `workflow_dispatch:` only and re-add `push: branches:
[main]` together with enabling Pages. Commit `476c655`.

## Audit script ≠ behavior test (2026-05-04)

Two complementary checks, easy to confuse:

- **`tools/audit-backend-capabilities.py`** parses the binary's
  `--list-backends-json` and the test script's `Backend(...,
  capabilities=(...))` tuples, and reports drift between *what the
  binary claims* and *what the test script claims to cover*. It
  cannot detect a backend that mis-declares a cap that the test
  also dishonestly lists — both ends agree, audit says clean.
- **`tools/test-all-backends.py --profile feature`** actually runs
  each cap. This is what catches a cap declared but not implemented
  (omniasr's `CAP_PUNCTUATION_TOGGLE` shipping with a CTC vocab that
  has no punctuation; parakeet declaring `CAP_LANGUAGE_DETECT` with
  no stderr LID line in its native path).

Run the audit before pushing — it's seconds. Run the feature suite
periodically and after backend changes — it's slow but the only way
to catch *behaviour* drift.

When both directions of the audit agree but feature-suite tests
fail, the failure is one of:
1. The runner is under-invoking (missing CLI flag like `-dl` or
   `-am`). The runner must be widened — usually by inspecting the
   binary's caps via `--list-backends-json` and branching.
2. The backend declares a cap it doesn't actually deliver. The cap
   must be dropped from the `.cpp` backend's `capabilities()`
   override AND from the test tuple in lockstep, otherwise the
   audit will start reporting drift.

## libcrispasr.a + libcommon.a both define stb_vorbis / miniaudio impl (Linux ld dies)

`src/crispasr_audio.cpp` and `examples/common-crispasr.cpp` both
translation-unit-include `stb_vorbis.c` after `#undef
STB_VORBIS_HEADER_ONLY` and both define `MINIAUDIO_IMPLEMENTATION`.
This means *both* static libraries (libcrispasr.a, libcommon.a)
ship the full set of stb_vorbis + miniaudio symbols.

- **Apple ld** silently picks the first definition. macOS builds
  passed without complaint for years.
- **GNU ld** (Linux) errors with `multiple definition of …`. Comes
  up the moment a downstream consumer links *both* archives — which
  the Ruby binding does via `bindings/ruby/ext/dependencies.rb`'s
  full-graph topological link.
- **Main CI build doesn't trip** because libcrispasr is compiled as
  a shared lib (`libcrispasr.so`) whose symbols are resolved at
  build time, so the executable only sees one copy of stb_vorbis
  (from libcommon.a).

The proper fix is to extract the impl into its own dedicated
translation unit linked exactly once — but that's a refactor with
ripples (some `tests/` and `examples/crispasr-quantize/` link
libcommon alone and would need the new lib added). For the Ruby
binding specifically, `-Wl,--allow-multiple-definition` on Linux is
the minimal workaround that mirrors macOS ld's behaviour.

Also: when CMake's `CRISPASR_STANDALONE` is ON (default for any
consumer that points `-S sources` at the repo root, including the
Ruby binding's vendored copy), `CRISPASR_BUILD_TESTS` defaults ON,
which pulls Catch2 into the configured target graph. The Ruby
binding's dependency walker then lists `libCatch2*.a` in
`$LOCAL_LIBS` even though `--target common crispasr` never built
them, and the link fails with `cannot find libCatch2WithMain.a`.
Fix: pass `-D CRISPASR_BUILD_TESTS=OFF -D CRISPASR_BUILD_EXAMPLES=OFF
-D CRISPASR_BUILD_SERVER=OFF` to the Ruby binding's cmake config.
Both fixes shipped together in `bindings/ruby/ext/extconf.rb`.

## Chatterbox-Turbo conformer encoder — ggml layout traps (May 2026)

Five bugs, all in `chatterbox_s3gen.cpp`, all discoverable only via
element-wise diff against PyTorch reference (the crispasr-diff harness).

### 1. EspnetRelPositionalEncoding ordering

Python builds pe_positive reversed + pe_negative, giving positions from
+(T-1) to -(T-1). A natural C++ loop from p=0 with `pos = p - (T-1)`
gives -(T-1) to +(T-1) — **exactly backwards**. The sine components are
negated which changes Q·P dot products after rel_shift. Fix: `pos = (T-1) - p`.

This was invisible in RMS comparison (reversed positions have identical
L2 norms) and only detectable via element-wise matrix_bd comparison.

### 2. pos_bias_u/v: reshape vs transpose

`pos_bias_u` stored in GGUF as ne=(d_k, n_heads). To broadcast-add to
Q with shape (d_k, T, n_heads), just reshape to (d_k, 1, n_heads). A
transpose before reshape scrambles head/dim indices — the (d, 1, h)
element gets a bias value from a different head. Again invisible in RMS
comparison, reduced all attention scores to ~70% of correct magnitude.

### 3. ggml reshape ≠ PyTorch view for multi-head attention output

**The most expensive bug.** After attention weighted sum, the output has
shape (d_k, T, n_heads). Python's `transpose(1,2).view(B, T, D)` gives
head-concatenated layout: [h0_d0..h0_d63, h1_d0..h7_d63] per time step.
But ggml `reshape_2d(D, T)` just reinterprets the flat buffer, giving
interleaved head/time indices. The output projection `lo.weight` was
trained expecting concatenated heads, so the wrong layout causes 8%
signal reduction — compounding across 10 blocks to 24% total loss.

Fix: add `ggml_permute(0, 2, 1, 3)` before reshape to move heads
before time: (d_k, T, H) → (d_k, H, T) → reshape → (D, T).

**Rule:** whenever converting from ggml's (d_k, T, H) back to
PyTorch's (T, D) layout, you MUST permute heads-before-time first.
ggml reshape is NOT equivalent to PyTorch view when the intermediate
dimensions have different orderings.

### 4. F0 predictor dequantization

The F0 predictor reads weight tensors manually with
`ggml_backend_tensor_get`. When tensors are quantized (Q8_0/Q4_K),
`ggml_nbytes(tensor)` < `n_elem * sizeof(float)`, causing out-of-bounds
reads. Fix: read `ggml_nbytes(tensor)` bytes, then use
`ggml_get_type_traits(type)->to_float()` to dequantize. This applies to
ALL manual tensor reads in CPU-side code (F0 predictor, SourceModule
linear, speaker embedding projection).

## Chatterbox repaired GGUF split — stage-specific regen (May 2026)

The repaired `*-regen.gguf` artifacts are intentionally asymmetric:
base Chatterbox regenerated T3, while Chatterbox Turbo regenerated
S3Gen.

- `chatterbox-t3-f16-regen.gguf` fixes the base T3 GGUF export. The
  old artifact only carried the legacy character-token vocabulary and
  missed the real HF BPE tokenizer tokens/merges, so C++ fed a different
  text-token sequence than Python.
- `chatterbox-turbo-s3gen-f16-regen.gguf` fixes the Turbo companion
  stage. Turbo T3 uses the GPT-2-style path and was not the artifact
  being repaired; the downstream S3Gen/vocoder GGUF needed rebuilding
  after the S3Gen parity fixes.
- Turbo T3 still needs quant coverage even though it did not need
  regeneration. Publish `chatterbox-turbo-t3-q8_0.gguf` and
  `chatterbox-turbo-t3-q4_k.gguf` from the canonical F16 so users can
  select matching T3/S3Gen quants without relying on local `-regen`
  names.
- `-regen` is a local repair marker, not a registry contract. HF uploads
  should publish the repaired bytes under canonical filenames so
  auto-download and auto-resolve keep working without special cases.
- Do not infer a general rule that base always needs T3 regeneration or
  Turbo always needs S3Gen regeneration. The regenerated file is the
  stage whose GGUF export/runtime contract was wrong.

Current debug boundary: base T3 tokenizer, conditioning, prefill, CFG,
step-0 logits, and forced step-1 logits match Python. S3Gen replay and
HiFT final reconstruction also match when fed reference tensors. The
remaining token-stream drift is sampler parity with CPU
`torch.multinomial`, not model math.

## Chatterbox base T3 sampler parity — Gumbel-max → torch.multinomial (May 2026)

Followup to the entry above: ported the base T3 sampler at
`src/chatterbox.cpp:541` from a Gumbel-max trick (one MT19937 32-bit
uniform per token, `argmax(p_i / -log(1-u_i))`) to a faithful CPU
`torch.multinomial(probs, num_samples=1)` (cumsum + normalize +
snap last bucket to 1.0 + binary search on a single 53-bit double
uniform from two MT19937 draws combined as `random64 = (r1<<32)|r2`,
masked to 53 bits, divided by 2^53). The recipe matches
`aten/src/ATen/native/cpu/MultinomialKernel.cpp` and
`aten/src/ATen/core/DistributionsHelper.h::uniform_real_distribution`
exactly. Closes the residual "gibberish at start / blurred words"
audible drift on long prompts (issue #76).

Direct A/B on `"…I want to move like a titan at dawn."` with
`chatterbox-t3-f16-regen.gguf` + `chatterbox-s3gen-q8_0.gguf`:

| Sampler                   | ASR roundtrip transcript                   |
|---------------------------|--------------------------------------------|
| Gumbel-max (`p / -log1p(-u)`) | "…like a **Titanette Dawn**" (mispronounced) |
| `torch.multinomial` faithful  | "…like a **Titan at dawn**" (matches input)  |

Why Gumbel-max was empirically close but not equivalent here:
fp32 `-log1p(-u)` with a 32-bit `u` discretizes the right tail to
`u ∈ {0, 2^-32, 2·2^-32, …}` so the smallest non-1 `u` gives
`e ≈ 22 = 32·log(2)`. PyTorch's 53-bit-precision uniform reaches
`e ≈ 37` in the same tail. That biases Gumbel-max toward sampling
further from the mode for any specific (logit, draw) — within the
sampling distribution's natural variance per-token, but compounding
across 60+ AR steps into specific token-id divergence at points
where the post-temperature-post-min-p probability mass is thin and
spread (the chatterbox speech-token vocab is 8194-wide and post-CFG
multimodal, so this is common).

Don't try to match Python's RNG state — `torch.manual_seed(...)`
seeds CPUGeneratorImpl through a different path than our manual
MT19937, and chatterbox's reference dump doesn't seed at all. The
fix is statistical equivalence, not bit-exact reproducibility.

## Chatterbox HiFT vocoder parity nits (May 2026)

Two parity bugs found while diff-testing the vocoder against Python's
HiFTGenerator with reference mel + reference source-stft fed in:

### 1. Pre-conv_post LeakyReLU slope is `0.01`, not `0.1`

`/private/tmp/resemble-chatterbox/src/chatterbox/models/s3gen/hifigan.py:418`
calls `F.leaky_relu(x, self.lrelu_slope)` with `lrelu_slope=0.1` —
this is the in-loop activation. But line 437 — the LAST activation
before `conv_post` — calls `F.leaky_relu(x)` with **no slope
argument**, so PyTorch's default `negative_slope=0.01` applies.
Easy to miss when porting because everything else uses 0.1. Fixed
at `src/chatterbox_s3gen.cpp:1998`. `voc_conv_post` cos_mean
improved 0.985 → 0.993 against the per-stage reference.

### 2. Snake activation needs `1/(α + 1e-9)`, not `1/α`

`chatterbox/models/s3gen/transformer/activation.py:71,82`:

```python
self.no_div_by_zero = 0.000000001
…
x = x + (1.0 / (alpha + self.no_div_by_zero)) * pow(sin(x * alpha), 2)
```

Implemented in C++ via `ggml_scale_bias(ctx, alpha, 1.0, 1e-9)` →
`ggml_div(sin², α_safe)`. Doesn't help on the current released
weights (min α observed in `s3.v.rb.*` is 0.016 ≫ 1e-9), but it
matches Python bit-for-bit and defends against future fine-tunes
that drift α further toward zero. Three sites in
`chatterbox_s3gen.cpp` (rb-snake1, rb-snake2, srb-snake1, srb-snake2).

### Residual cumulative drift — diff harness fed source_stft transposed

The `hift_pcm(ref_mel) cos = 0.879` regression and all the per-stage
drift (`voc_rb_0` cos_min ≈ 0.94, `voc_ups_2` cos_min ≈ −0.13,
`voc_conv_post` cos_min ≈ 0.65) traced to a **layout mismatch in the
diff harness's external `source_stft` feed**, not anything in the
runtime vocoder.

- The python reference dump writes
  `s_stft.permute(1, 0).contiguous()` → bytes are `(T, C=18)`
  row-major (`data[t*18+c]`).
- The C++ vocoder allocates
  `s_stft = ggml_new_tensor_2d(ctx, F32, T_src, 18)` with `ne[0] =
  T_src` as the fast axis, so it expects `(C, T_fast)` row-major
  (`data[c*T_src+t]`) — matches the internally-generated layout at
  `src_stft[f*T_src+frame] = re` in `chatterbox_s3gen.cpp`.
- The harness was `memcpy`'ing `ref.get_f32("hift_source_stft")`
  directly into the C++ tensor without transposing. C++'s
  `source_downs[0]` was therefore reading channels and time swapped
  — Conv1d with `in_channels=18` ate ~12 241 time slices as
  channels, producing structured garbage that compounded through
  the resblock chain into the visible cos drift.

Fix is a single transpose in `crispasr_diff_main.cpp` when binding
`ref_source_stft`. After: all 7 vocoder stages pass at
`cos_min = 1.000000` and `hift_pcm(ref_mel)` matches the python
reference to `max_abs = 2.85e-05` (essentially fp32 ULP). The
runtime TTS path was never wrong — it generates `source_stft`
internally in the correct layout, so audio quality is unchanged.

The prior two investigations (May 2026 original write-up + the
2026-05-25 addendum) misdiagnosed this as fp accumulation-order
divergence. The "10 % per-element diff at T=0" was real, but it
was caused by `source_downs[0]` reading transposed bytes, not by
ggml's reduction tree. Lesson: when a per-stage diff jumps from
`cos = 1.0` (input) to `cos ≈ 0.94` (one Conv1d later) on inputs
that should be bit-equivalent, suspect a layout/feed bug before
suspecting precision. The handover's Hypothesis C ("a structural
bug we missed") was right; it just lived in the harness's input
binding, not the resblock chain.

#### Harness coverage hardening 2026-05-25

Three follow-up changes so the next bug of this class lands faster:

- **`voc_si_{0,1,2}` + `voc_rb_input_{0,1,2}` stages.** Source fusion
  is now diffed independently of the main resblock chain. A
  layout/structural bug in `source_downs[i]` or `source_resblocks[i]`
  now flags at `voc_si_i` instead of at `voc_rb_i` — the prior
  investigations spent days narrowing "drift starts at voc_rb_0" to
  source fusion; the new probe makes that one harness run.
  `tools/extend_chatterbox_ref_si_stages.py` computes the new
  reference stages from `voc_ups_*` + `hift_source_stft` + the F32
  weights in the s3gen GGUF (no need to re-run the (pruned)
  chatterbox-ref-venv); the python ref backends were also updated
  for future re-dumps from scratch.
- **Two-tier threshold on vocoder stages.** Pure-fp32 stages now
  require `cos_min >= 0.999` in addition to the existing
  `cos_mean >= 0.95`. With the bug, `voc_rb_0` sat at
  `cos_min 0.937 cos_mean 0.998` — passed the loose check while
  visibly broken. The strict floor flips that to `[FAIL]`.
- **`hift_pcm(internal_src)` self-consistency check.** Re-runs the
  vocoder with `source_stft_cf=NULL` so it uses internal F0/SineGen/
  STFT instead of the python-archive feed, then compares the wav
  against the reference (loose `cos_mean >= 0.80` to allow for
  C++ vs torch source-gen drift). Catches future regressions in
  the internal source-gen path that the external-feed-only diff
  would miss. On JFK 11 s with the runtime in its current state,
  this passes at `cos = 0.9998` — internal source gen tracks torch
  surprisingly closely despite being independently implemented.

## Chatterbox voice cloning — bake to GGUF, load via `--voice` (May 2026)

Voice cloning uses the same baker-→-voice-GGUF pattern as vibevoice
(`models/convert-vibevoice-voice-to-gguf.py`):
`models/bake-chatterbox-voice-from-wav.py` runs upstream
`ChatterboxTTS.prepare_conditionals(wav)` and writes a 150-200 KB
GGUF using **the same tensor names the C++ runtime already accepts
for the built-in voice** (see `convert-chatterbox-to-gguf.py:521-548`):

```
conds.t3.speaker_emb           f32  (1, 256)
conds.t3.speech_prompt_tokens  i32  (T_prompt,)
conds.gen.prompt_token         i32  (T_speech,)
conds.gen.prompt_feat          f32  (T_mel, 80)
conds.gen.embedding            f32  (1, 192)
chatterbox.conds.emotion_adv          f32 metadata
chatterbox.conds.gen_prompt_token_len u32 metadata
```

C++ side: `chatterbox_set_voice_from_wav()` dispatches on extension —
`.gguf` paths route through `chatterbox_load_voice_gguf()`, which
loads via `core_gguf::load_weights` into a separate
`voice_ctx_w`/`voice_buf_w` and rebinds `ctx->conds.*` tensor pointers.
The original baked-in default-voice tensors stay allocated in
`ctx_w`/`buf_w` (unreferenced after rebind, freed only at context
shutdown). `.wav` paths print a hint pointing at the baker —
in-process WAV → cond extraction (VE LSTM, CAMPPlus TDNN,
S3Tokenizer encoder + quantizer) is a separate refactor; weights
ARE in the s3gen GGUF already (`s3.se.xv` 755 tensors, `s3.se.head`,
`s3.tok.{enc,quant,encoder,_mel_filters}` 103 tensors), but
implementing the forward passes is multi-day work.

CLI plumbing: `--voice <path.gguf>` is wired in
`crispasr_backend_chatterbox.cpp::synthesise()` with a
per-call `last_voice_key_` cache so server callers can switch voice
between requests. Mirrors the vibevoice posture (refuses to
silently fall back to the default voice on a load failure).

### The build-target trap that masked it

`cmake --build build-ninja-compile --target crispasr-lib` only relinks
the **shared library** `libcrispasr.dylib` and stops there — the CLI
executable lives in target `crispasr-cli`. So edits to
`examples/cli/crispasr_backend_*.cpp` need an explicit
`--target crispasr-cli` rebuild to land in `bin/crispasr`. We hit
this on the first voice-clone smoke test: the synth log showed no
"voice loaded" line and the cloned wav md5'd different from default
purely because of post-build I/O timestamps in the wav header
(audio bytes were identical). Always check that
`build-ninja-compile/examples/cli/CMakeFiles/crispasr-cli.dir/<file>.o`
has a fresher mtime than the .cpp before declaring a CLI-side
behaviour test "passed".

## Chatterbox VoiceEncoder native port — module 2 of voice cloning (May 2026)

`src/chatterbox_ve.cpp` ports the upstream
`chatterbox.models.voice_encoder.embeds_from_wavs` so `--voice <ref>.wav`
works with no python at runtime. Module 2 only — modules 3
(S3Tokenizer for `gen.prompt_token` + `t3.speech_prompt_tokens`) and
4 (CAMPPlus for `gen.embedding`, plus the 24 kHz prompt mel for
`gen.prompt_feat`) are still pending. See
`src/chatterbox_ve.h` for the pipeline contract.

### Stuff that mattered

1. **`embeds_from_wavs` default rate is 1.3, NOT 0.5**. Reading the
   handover I assumed `overlap=0.5` → frame_step = 80, but the
   upstream wrapper sets `kwargs["rate"] = 1.3` if the caller didn't,
   and `get_frame_step` switches to `round((sr/rate)/partial_frames) =
   round(16000/1.3/160) = 77`. Mis-setting this to 80 would still
   produce a working embedding but with a different partition that
   diverges from upstream's per-partial layout.

2. **`mel_type='amp'` is no log step at all, not log on the amp**.
   The upstream `melspec.py` only calls `_amp_to_db` when
   `hp.mel_type == "db"`. With `'amp'` the LSTM consumes the raw
   mel-projected magnitude. Implemented as `core_mel::LogBase::None`
   (added in commit `957eb4ff`) and verified.

3. **`mel_power=2.0` means project `|X|^2`, not `|X|`**. Both
   `core_mel::SpecKind::Power` and the bare `Magnitude` layout were
   plausible; the config has `mel_power = 2.0` which is the upstream's
   knob to square the magnitudes before mel projection. Same as
   librosa default (`power=2.0`).

4. **The LSTM is small enough to run pure-CPU; ggml graph integration
   exploded on `buffer_id < 0`**. First attempt built per-partial
   ggml graphs that mixed model-buffer weights with on-the-fly
   `ggml_cast` and `ggml_view` ops, fed through `ctx->sched`. Hit
   `GGML_ASSERT(buffer_id >= 0)` in `ggml_gallocr_allocate_node` —
   the scheduler couldn't pick a backend for the cast/view chain
   when the source weights live in a buffer the scheduler hadn't
   ingested. Rewrote the LSTM as plain C++ float math
   (`lstm_unidir_cpu` in chatterbox_ve.cpp): read all VE weights to
   F32 with `read_tensor_f32` (mirrors the existing `tensor_get_f32`
   helper) and run the recurrence directly. Bit-equivalent to torch's
   nn.LSTM forward; ~1-2 s for 13 partials on M1 (one-shot voice
   clone — perf isn't a concern). For larger LSTMs that need GPU
   dispatch the right pattern is to allocate weights into the
   scheduler's own context, not borrow tensors from a foreign one.

5. **L2-norm via ggml is awkward; do it on host**. ggml has
   `ggml_rms_norm` (RMS, not L2 — divides by `sqrt(mean(x²)+eps)`)
   and no built-in L2. Trying to express `y/sqrt(sum(x²))` via
   `rms_norm` requires a `1/sqrt(N)` scale that's brittle to the
   eps. Read the (256,) embedding back from the graph and
   normalise on CPU — single divide, matches `torch.linalg.norm`
   exactly with no eps drama.

6. **Silence trim deferred — match the dump's audio path**.
   Upstream `embeds_from_wavs` does
   `librosa.effects.trim(top_db=20)` before mel. Porting
   librosa.feature.rms's center=True/pad_mode='constant' framing
   plus the dB threshold logic is fiddly; for module 2 we bypass
   trim in BOTH the C++ port and the python reference dump
   (`tools/reference_backends/chatterbox.py` calls
   `melspectrogram` directly instead of `embeds_from_wavs`). Result:
   bit-equivalent comparison and a clean 1.0 cosine on
   `ve_partial_emb` and `ve_speaker_emb`. For typical pre-trimmed
   reference WAVs this is a no-op; long silences would degrade
   the embedding until trim lands as a follow-up.

7. **`librosa.stft` with `n_fft=400` works fine through
   `core_fft::fft_radix2_wrapper`**. core_fft splits `400 → 200 → 100
   → 50 → 25 (odd)` and falls back to direct DFT at the odd remainder.
   No need for a Bluestein chirp-z. ~10K ops per frame; negligible.

8. **Periodic Hann window, not symmetric**. librosa's STFT default
   is `scipy.signal.windows.get_window('hann', N, fftbins=True)`,
   which is the periodic variant `0.5 * (1 - cos(2π i / N))` (i ∈
   [0, N)). Symmetric Hann uses `N-1` in the denominator and is
   off-by-one — would fail mel parity by ~1e-3.

9. **`cb_ve_model` lives in chatterbox_ve.h, not chatterbox.cpp's
   anonymous namespace**. The struct was originally a private
   detail in chatterbox.cpp; the cleanest separation for a new
   per-module .cpp is to lift it to the shared header so chatterbox.cpp
   keeps `bind_ve` and chatterbox_ve.cpp keeps the forward.

### Build target nits (still relevant)

CLI changes need `--target crispasr-cli`; library-only changes work
with `--target crispasr-lib` (or `--target chatterbox` for the static
sublib). The diff harness needs `--target crispasr-diff`.

### Diff stages added

`tools/reference_backends/chatterbox.py` now captures:
- `ve_mel`           — (T, 40) raw-amp Slaney mel after the trim-bypass path
- `ve_partial_emb`   — (n_partials, 256) L2-normed per-partial embeddings
- `ve_speaker_emb`   — (1, 256) final speaker embedding (mean + L2)

C++ ABI hooks `chatterbox_dump_ve_*` in `chatterbox.h`. The
`crispasr-diff chatterbox` harness exits-codes against the same
0.95 cos_mean threshold as other chatterbox stages. On JFK 11s the
mel cos_mean is 0.999913 (cos_min 0.905 in low-energy frames where
fp32 directions are unstable; downstream LSTM filters this out and
ve_partial_emb / ve_speaker_emb are essentially bit-perfect).

## Chatterbox S3Tokenizer V2 native port — module 3 of voice cloning (May 2026)

`src/chatterbox_s3tok.{h,cpp}` ports `S3TokenizerV2.quantize` —
the FSMN-augmented Whisper-style encoder + FSQ codebook that
turns a 16 kHz reference WAV into 25 Hz speech tokens
(codebook size = 3⁸ = 6561). Lives next to chatterbox_ve (module 2)
and is bound on the s3gen sub-context's tensor map (the `s3.tok.*`
weights ride in the chatterbox-s3gen GGUF, not the T3 GGUF).

### Stuff that mattered

1. **The K projection has no bias**. S3Tokenizer follows Whisper's
   `MultiHeadAttention` where `query` and `value` are biased but
   `key` is `Linear(... bias=False)`. So
   `s3.tok.enc.b.<l>.attn.key.bias` is absent from the GGUF, and
   the bind step gets `nullptr`. First implementation called
   `ggml_add(K, b.attn_k_b)` unconditionally — segfault inside
   `ggml_add_impl`. Fix is the same pattern voxtral / qwen3 use:
   gate every Q/K/V/O bias add on a non-null pointer.

2. **FSMN side branch on V, computed BEFORE the attention head
   reshape**. The python's `FSMNMultiHeadAttention.qkv_attention`
   reshapes V to `(B, T, n_head, head_dim)` then immediately
   collapses it back to `(B, T, D)` for the depthwise Conv1d. So
   we can equivalently compute FSMN on the post-projection V
   tensor directly (still `(D, T)` in our ggml layout) — transpose
   to `(T, D)` for `ggml_conv_1d_dw`, run the depthwise k=31 conv,
   add the residual V back, transpose back to `(D, T)`. The
   attention output projection's result is then `+ fsmn_memory`
   before the residual add. `attn_fsmn_w` is `(31, 1, 1280)` F16 —
   ggml_conv_1d_dw consumes it as-is.

3. **Periodic Hann via `torch.hann_window`** — same as VE, but the
   STFT path here is post-log so a missing or off-by-one Hann
   shows up in `s3tok_log_mel`'s cosine before the encoder ever
   runs.

4. **`magnitudes = stft[..., :-1].abs()**2`** drops the last STFT
   frame. core_mel exposes this as `Params::drop_last_frame`, set
   true to match. On JFK 11 s @ 16 kHz this gives T=1100 (not 1101
   like VE's mel that keeps all frames).

5. **Conv1d strides are both 2**. AudioEncoderV2's `__init__`
   passes `stride=stride` (the constructor arg, set to 2 for the
   tokenizer) to conv1, and conv2 hardcodes `stride=2`. Total
   downsample is 4× → T mel frames at 100 Hz become T/4 tokens at
   25 Hz. For a 6 s prompt: 600 mel frames → 150 tokens, exactly
   the `speech_cond_prompt_len = 150` the t3 prompt expects.

6. **RoPE n_dims = head_dim, not n_state**. The model_v2
   `precompute_freqs_cis(64, 1024*2)` uses 64 (the head_dim) for
   the rotation dim, not 1280 (n_state). Pass `head_dim` (=64) as
   the `n_dims` arg to `ggml_rope_ext`. n_ctx_orig at 2048 covers
   the upstream max position (`1024 * 2`).

7. **NEOX RoPE matches the python's `apply_rotary_emb`**. The
   python's `cat((freqs_cis, freqs_cis), dim=-1)` doubling and the
   `(half_l, half_r)` split / `(-half_r, half_l)` rotation is the
   GPT-NeoX form: pairs `(i, i+head_dim/2)` rotated by `θ_i`.
   `GGML_ROPE_TYPE_NEOX` is the right enum.

8. **`s3tok_tokens` cosine is below 0.999 — by design**. FSQ is a
   discrete quantization: `tanh(h)*0.999 → round → +1`. Tiny float
   drift in the encoder pushes a few logits across the {-0.5, 0.5}
   rounding boundary, which flips a base-3 digit and changes the
   integer code. Cosine on the integer stream then reflects the
   per-token mismatch rate, not the underlying float drift. JFK
   11 s gives `cos_min=0.997`; the parity-quality metric is
   `s3tok_proj_down` (the pre-FSQ floats) which is comfortably
   `cos_mean=0.99993`.

### Voice clone wiring discipline

When wiring module 3 outputs into `chatterbox_set_voice_from_wav`
the .wav branch, **do NOT update `gen.prompt_token` alone** — it has
to stay in lockstep with `gen.prompt_feat` (the 24 kHz prompt mel)
and `gen.embedding` (the CAMPPlus x-vector). All three describe the
same reference audio for S3Gen's flow matcher; cross-mixing
prompt_token from the new ref with prompt_feat / embedding from the
default ref makes the vocoder collapse to ~0.0003 RMS silence
(verified on JFK clone). The right partial-clone state is:

  - Module 2 only:     speaker_emb NEW, all gen.* DEFAULT
  - Module 2+3:        speaker_emb + speech_prompt_tokens NEW (both
                       T3-side); all gen.* DEFAULT
  - Module 2+3+4:      everything NEW (atomic)

Module 3 only updates the T3 side (`speech_prompt_tokens`); the
S3Tokenizer's full-audio token stream is exposed via
`chatterbox_dump_s3tok_tokens` for parity testing but isn't
written into the runtime conds bundle yet.

### Diff stages

`tools/reference_backends/chatterbox.py`:
- `s3tok_log_mel`             — (128, T) log10 mel after clip-and-scale
- `s3tok_proj_down`           — (T_tok, 8) pre-FSQ projdown floats
- `s3tok_tokens`              — (T_tok,) full-audio int32 tokens
- `s3tok_speech_prompt_tokens` — (≤150,) first-6 s int32 tokens

C++ ABI hooks in `chatterbox.h`. Threshold and reporting style
mirror the existing chatterbox stages (cos_mean ≥ 0.95).

## Chatterbox CAMPPlus phase 1 — Kaldi fbank front-end (May 2026)

`src/core/kaldi_fbank.{h,cpp}` ports `torchaudio.compliance.kaldi.fbank()`
with default args (powey window, HTK mel scale, log power,
preemph=0.97, snip_edges=True, round_to_power_of_two=True). Promoted to
a shared core helper since multiple speaker / VAD models consume Kaldi
fbank — currently inline copies live in `firered_asr.cpp` and could be
deduped in a follow-up; the new helper takes an `int16_scale` knob to
cover the firered case where the trained CMVN expects int16-magnitude
features.

`src/chatterbox_campplus.{h,cpp}` is the chatterbox consumer (Module 4
of the native voice clone path). Phase 1 of this module ships the
fbank front-end + the per-utterance mean subtract that
`xvector.extract_feature` does. Phase 2 (the actual CAMPPlus TDNN
forward — FCM 2-D conv head + 12+24+16 dense TDNN layers + StatsPool +
1024→192 dense) is deferred to a follow-up commit since it needs ~50
BatchNorm-fold pairs and a non-trivial CAM seg_pooling op
(`F.avg_pool1d(k=100, s=100, ceil_mode=True)` with broadcast-back) and
is multi-hour focused work.

### Stuff that mattered (phase 1)

1. **Reuse, don't reinvent**. `firered_asr.cpp:compute_fbank` was
   already a faithful Kaldi fbank — same povey window, same HTK mel
   scale, same preemph-with-s[-1]=s[0] boundary, same FLT_EPSILON log
   floor. Lift to a parameterised core helper (raw vs int16 scaling),
   keep firered's inline copy unchanged for now to avoid churn.

2. **Kaldi pre-emphasis boundary**: Kaldi's `feature-window.cc` uses
   `s[-1] = s[0]` for the boundary, so the i=0 step becomes
   `s[0] -= preemph * s[0]` → `s[0] *= (1 - preemph)`. Must walk the
   array in REVERSE so each step reads the unmodified s[i-1].

3. **Kaldi mel scale = HTK** (`mel = 1127 * log(1 + hz/700)`), and
   filters are NOT Slaney-area-normalized — the bare triangles. This
   is what every Kaldi-trained model (including CAMPPlus) expects;
   feeding in a Slaney-normalized basis silently destroys downstream
   accuracy. The librosa default normalization would have been wrong
   here.

4. **`round_to_power_of_two=True` is just zero-padding the windowed
   frame to next_pow2(win)**. For win=400, n_fft=512. The energy at
   the original 400 samples is what matters; the 112 zero-padded tail
   contributes nothing to the FFT bins it produces.

5. **`int16_scale=False` for CAMPPlus**. Despite torchaudio's docstring
   saying "Kaldi typically uses 16-bit audio integers", chatterbox
   feeds CAMPPlus float-[-1, 1] audio directly (see `xvector.py`'s
   `Kaldi.fbank(au.unsqueeze(0), ...)` — `au` is normalized float).
   The trained model adapted to that scaling. Setting `int16_scale`
   on would shift the log floor by a constant +log(32768²) ≈ 20.8 per
   bin and the CAMPPlus TDNN's BN running stats would no longer fit.

6. **`use_energy=False` is the default**. Kaldi's convention is to
   either include log-energy as the first bin (use_energy=True) or
   keep just the mel bins. CAMPPlus's call doesn't pass use_energy →
   defaults to False, so output is exactly num_mel_bins=80 columns.

### Diff parity (phase 1)

`tools/reference_backends/chatterbox.py` captures `campplus_fbank` via
`torchaudio.compliance.kaldi.fbank` + the per-utterance mean subtract.
`crispasr-diff chatterbox` on JFK 11 s gives `cos_min=0.999994
cos_mean=0.999999` against torchaudio — fp32 rounding noise tight.

## Chatterbox CAMPPlus phase 2 — TDNN forward (May 2026)

Phase 2 of module 4 ports the CAMPPlus speaker encoder forward to native
C++ — the FCM 2-D conv head + 12+24+16 dense TDNN layers + StatsPool +
1024→192 dense projection. ~815 source tensors. Pure CPU float math (no
ggml graph) since the per-channel BN folding plus the dense block's
hold-and-broadcast `seg_pool` op are awkward to express in ggml's
broadcasting; CPU is plenty fast for one-shot voice clone (≈2 s on M1
for an 11 s clip).

### Stuff that mattered

1. **`out_nl.bn.*` is NOT under `out_nl.nl.bn.*`**. The other units in
   the GGUF (`tdnn`, `transit{1,2,3}`, `dense`) wrap a Linear+nonlinear
   pair and store the BN at `<unit>.nl.bn.*`. But `out_nl` is a bare
   `get_nonlinear` (only a BN, no Linear), so its tensors are at
   `out_nl.bn.{weight,bias,running_mean,running_var}` directly. Using
   the generic `bind_unit` path silently bound the BN to nullptr and the
   xvector forward segfaulted in `apply_bn_inplace` reading from the
   empty `gamma` vector.

2. **TransitLayer and DenseLayer order is BN→ReLU→Linear, NOT
   Linear→BN→ReLU**. The `get_nonlinear(config_str, in_channels)` runs
   FIRST (on the input), then `Linear(in→out)` projects. So the BN's
   running stats are sized for `in_channels`, not `out_channels`. Folding
   the BN with the wrong `C` produces silently corrupt output. Got this
   right by binding via the source `cb_campplus_unit` and folding inside
   the forward (see `bn_relu_conv1d` helper).

3. **`F.avg_pool1d(k=100, stride=100, ceil_mode=True)` divides by `k`,
   not by the actual frame count**. PyTorch's `count_include_pad=True`
   default treats the partial last window as if it were padded with
   zeros, so the divisor is always 100. Dividing by `n_in_seg` (the
   actual count) instead would shift the seg_pool values for the last
   segment of any T not divisible by 100.

4. **`torch.std` defaults to `unbiased=True`** (divide by `n-1`).
   Matters for the StatsPool — `statistics_pooling(x)` uses
   `x.std(dim=-1, unbiased=True)`. Using `n` instead of `n-1` gives
   slightly off std values that the dense's BN would amplify.

5. **CAMPPlus FCM head: PyTorch `BasicResBlock` shortcut path activates
   when `stride != 1` OR `in_planes != out_planes`**. For our weights
   (out_channels=32 throughout), the shortcut only appears when stride
   changes, which is once per layer (the `.0` block). The `.1` block is
   identity-shortcut. The bind step in `chatterbox_s3gen.cpp` checks
   `b.sc_w` presence to set `b.has_shortcut`.

6. **GGUF Conv2d weight ne=(kW, kH, in, out) maps directly to PyTorch
   `(out, in, kH, kW)` row-major**. ggml's reverse-indexing convention
   means the bytes are the same; just index with PyTorch's natural
   `((o*in + i)*kH + kh)*kW + kw` formula. Same trick for Conv1d
   weights ne=(kW, in, out) → PyTorch `(out, in, kW)`.

### Diff parity (phase 2)

`tools/reference_backends/chatterbox.py` captures `campplus_xvector` via
`s3gen.speaker_encoder.inference([wav_16k])`. The full chatterbox
top-level module loads TensorFlow transitively (slow); the parity dump
script can bypass it by directly importing
`chatterbox.models.s3gen.xvector.CAMPPlus` and loading just the
`speaker_encoder.*` slice from `s3gen.safetensors`.

`crispasr-diff chatterbox` on JFK 11 s gives:
- `campplus_fbank`   cos_mean=0.999999  (fp32 rounding noise, phase 1)
- `campplus_xvector` cos_mean=0.998070  (PASS at the 0.95 threshold —
                     small drift from accumulator order / BLAS-vs-naïve
                     conv compute, well under the per-element floor for
                     speaker-similarity downstream tasks)

The 0.998 vs 1.0 gap is the price of using triple-loop conv kernels
instead of PyTorch's BLAS GEMM accumulation order. For voice cloning
(downstream consumer is S3Gen's CFM cross-attention), this is
imperceptible — the speaker direction is preserved.

## Chatterbox 24 kHz prompt mel — module 4 phase 3 (May 2026)

`chatterbox_campplus.cpp:compute_prompt_feat_24k` ports
`chatterbox.models.s3gen.utils.mel.mel_spectrogram` for the
`gen.prompt_feat` cond. Sits next to CAMPPlus rather than its own file
since both modules consume / emit S3Gen-side conditioning in the
voice-clone path. ~80 lines, pure CPU, uses `core_fft` + the
`core_mel::build_slaney_fb` basis already shared with VE / S3Tokenizer.

### Stuff that mattered

1. **Magnitude (with eps inside the sqrt), not power**. The Matcha
   formulation is `sqrt(re² + im² + 1e-9)` — adding the eps INSIDE the
   sqrt rather than power-then-clamp. `core_mel::SpecKind::Magnitude`
   doesn't take an eps inside the sqrt, so the parity-correct path is
   to write the STFT loop inline with `std::sqrt(re² + im² + 1e-9)`.

2. **Natural log + clip-min, NOT log10 + max-clip(max-8) + (x+4)/4**.
   `dynamic_range_compression_torch(x) = log(clamp(x, 1e-5))`. Plain
   `std::log(std::max(v, 1e-5f))`. Different from S3Tokenizer's mel
   (log10 + Whisper-style normalisation) and from CAMPPlus's Kaldi
   fbank (log on power + epsilon).

3. **`center=False` with manual reflect-pad of `(n_fft - hop) / 2`
   each side, applied via `F.pad(..., mode="reflect")`**. PyTorch's
   reflect mirror EXCLUDES the edge sample (so for `[a, b, c, d]` with
   pad=2 the prefix is `[c, b]`, not `[b, a]`). Got this right by
   walking input indices `1..pad` for the prefix and `n-2..n-1-pad`
   for the suffix.

4. **Reference dump captures both `prompt_feat_24k` AND
   `audio_24k_input`**. The 16 → 24 kHz resample in `prepare_conditionals`
   uses `librosa.resample(res_type="kaiser_fast")` which we don't have
   in C++ yet. Saving the 24 kHz audio bytes alongside the mel lets
   the diff harness feed identical input to its mel — bypasses the
   resampler-parity question entirely. Diff harness reads
   `audio_24k_input` from the reference GGUF and pipes it into
   `chatterbox_dump_prompt_feat_24k`.

### Diff parity

`crispasr-diff chatterbox` on a 3.2 s pre-loaded 24 kHz prompt:

  prompt_feat_24k  shape=[80,160]  cos_min=1.000000  cos_mean=1.000000
                   max_abs=3.69e-04  rms=1.36e-05

Bit-perfect against `mel_spectrogram` for the full mel grid.

## Chatterbox atomic native voice clone — the resampler + 5-cond install (May 2026)

`src/core/audio_resample.{h,cpp}` adds a polyphase Kaiser-windowed
sinc resampler (β=8.6, num_zeros=14 — same parameters
`librosa.resample(res_type='kaiser_fast')` uses). Output is NOT
bit-equivalent to librosa (resampy uses a precomputed polyphase
table with a different precision knob), but acoustically very close.
Sized for any L:M reduction; chatterbox uses 16 ↔ 24 kHz
(L:M = 3:2 / 2:3) but the helper is general.

`chatterbox_set_voice_from_wav` now forks on the input rate:

  - **24 kHz mono PCM16/F32 WAV** — atomic path. Resamples 24 → 16 kHz
    once, then runs ALL FIVE compute modules from a single source:
      - VE (16 kHz) → speaker_emb
      - S3Tokenizer V2 (16 kHz) → speech_prompt_tokens (first 6 s, max 150)
                                  + prompt_token (full audio)
      - CAMPPlus (16 kHz) → gen.embedding (192-d)
      - 24 kHz Matcha mel (24 kHz, truncated to 10 s) → prompt_feat
    All five tensors get installed into the same fresh
    `voice_ctx_w` / `voice_buf_w` slot — atomic, mutually consistent.
  - **16 kHz mono PCM16/F32 WAV** — partial path (existing behaviour).
    Only T3-side conds (speaker_emb, speech_prompt_tokens) get
    installed; gen.* stay at the default voice's tensors. The
    runtime warns that 24 kHz input enables full atomic cloning.

### Stuff that mattered

1. **`embed_ref` enforces `T_mel = 2 * T_speech_tokens`**. If our
   prompt mel comes out shorter than `2 * len(prompt_token)` (e.g.
   because the 24 kHz audio is shorter than 10 s), `s3gen.flow_inference`
   raises a shape-mismatch warning and trims the tokens. We mirror
   the trim in `chatterbox_set_voice_from_wav` so install_native_voice
   sees a consistent (token, mel) pair.

2. **`gen.prompt_feat` ggml shape is (80, T_mel, 1)** — ne[0]=80 is
   the fastest axis, ne[1]=T_mel, ne[2]=1 (the singleton batch). Our
   compute_prompt_feat_24k returns row-major (T_mel, 80), which in
   ggml is exactly ne=(80, T_mel) — copy directly into a
   `ggml_new_tensor_3d(F32, 80, T_mel, 1)`.

3. **`gen.embedding` ggml shape is (192, 1)** — same convention. Our
   CAMPPlus xvector is a flat 192-d vector; ggml_new_tensor_2d(F32,
   192, 1) + ggml_backend_tensor_set with 192 floats covers it.

### Quality assessment

End-to-end verification (Q4_K T3 + `--no-gpu`, prompt "Ask not what
your country can do for you.", `samples/jfk.wav` resampled to 24 kHz
for the atomic path):

  - Atomic native (24 kHz WAV → all 5 conds): rms=0.113, ASR
    roundtrip transcribes the prompt verbatim. Real cloned voice.
  - Baker GGUF baseline (python `bake-chatterbox-voice-from-wav.py`):
    rms=0.118, ASR roundtrip transcribes the prompt verbatim.
  - Partial native (16 kHz WAV → M2+M3 only): rms=0.131, ASR also
    transcribes verbatim — but the **timbre is the default voice**,
    not the reference speaker. The path does NOT actually clone;
    it just feeds T3 the new speaker_emb + speech_prompt_tokens
    while S3Gen still uses the default voice's gen.* triple. The
    T3-side prosody hint isn't enough to override S3Gen's vocal
    identity. For real cloning, use the 24 kHz atomic path or the
    python baker — both verified producing speaker-cloned output.

### Known issue: Metal F16 drift in T3 — auto-fallback shipped

**Chatterbox T3 forward drifts on Metal/GPU past ~step 16**, breaking
the voice clone output regardless of voice path or quantization.
Pre-existing bug, NOT introduced by this work — reproducible at
the original voice-clone commit `86ac98eb` and every commit
before/since. The bisect via in-runtime KV/logit dump
(`CRISPASR_CHATTERBOX_DUMP_KV_AT=N`,
`CRISPASR_CHATTERBOX_DUMP_LOGITS_AT=N`):

| Config | Tokens 0-15 | Tokens 16+ | ASR result |
|---|---|---|---|
| Q4_K + CPU | reference | reference | ✓ "Ask not what your country can do for you." |
| F16 + CPU | bit-identical to Q4_K-CPU | bit-identical | ✓ same |
| Q4_K + GPU | bit-identical to Q4_K-CPU | DRIFTS | ✗ "And not what you're talking about..." |
| F16 + GPU | bit-identical to Q4_K-CPU | DRIFTS more aggressively | ✗ gibberish |

**Logit drift at end of prefill** (single forward pass through 30
layers, T=61 input tokens, no decode steps yet): CPU and GPU differ
by 1e-3 to 5e-2 across the first 8 logits — already enough drift
to flip the multinomial sampler's choices once the trajectories
diverge. The `KV_ON_CPU=1` workaround partly rescues (cleaner
audio, partial transcript) by routing the KV write/read through
the CPU backend, but T3 forward still computes logits on GPU and
drifts. Greedy sampling (`CRISPASR_CHATTERBOX_TEMP=0`) doesn't help
either — confirms the drift is at the logit level, not the sampler.

The drift is **deterministic** (same seed → same broken tokens) so
it's a correctness bug in some Metal op, not a race condition.
Likely the cumulative effect of F16 accumulator order across
mul_mat / flash_attn / norm kernels on Metal vs CPU's
F32-accumulator paths. Other ggml backends in this codebase
(parakeet, voxtral, qwen3) work fine on Metal — chatterbox is
unusual in being an autoregressive multinomial-sampled decoder
where small logit drift compounds catastrophically.

### The fix shipped

`chatterbox_init_from_file` auto-falls-back the **entire chatterbox
forward** (T3 + s3gen) to CPU when the user requests GPU, with a
loud stderr warning:

```
chatterbox: forward auto-falling back to CPU — Metal/GPU has
cumulative F16 drift that breaks chatterbox sampling past ~16
decode steps. Override with CRISPASR_CHATTERBOX_FORCE_GPU=1
(output may be garbled).
```

The decision flips `c->params.use_gpu` so the companion s3gen
sub-context (`chatterbox_set_s3gen_path` runs later) also picks
up the fallback. `--no-gpu` still works as before; explicit users
can override via `CRISPASR_CHATTERBOX_FORCE_GPU=1` for the broken
path (kernel-level debugging) and they get a different warning.

**Verification**: default command (no `--no-gpu` flag) on JFK clone
GGUF produces clean speech rms=0.12, ASR roundtrip transcribes
"Ask not what your country can do for you" verbatim. Same for
24 kHz native WAV input via the atomic clone path. Same for
default voice (no `--voice`). Same for the F16 T3 model that was
previously the most-broken combination.

### Diagnostic env knobs left in place

For future Metal-kernel investigation:
- `CRISPASR_CHATTERBOX_DUMP_KV_AT=<n_past>` — dumps layer-0 K cache
  contents at the requested cache row to stderr.
- `CRISPASR_CHATTERBOX_DUMP_LOGITS_AT=<n_past>` — dumps first 8
  output logits to stderr at the matching forward pass.
- `CRISPASR_CHATTERBOX_TEMP=<float>` — overrides T3 sampling
  temperature (0=greedy) without rebuilding the CLI plumbing.
- `CRISPASR_KV_READ_F32=1` — forces KV cache read to dequantise
  to F32 before flash_attn (didn't fix the drift here, but may
  help other backends with similar issues).
- `CRISPASR_CHATTERBOX_FORCE_GPU=1` — disables the auto-fallback.

The next step on the actual fix is per-op intermediate-tensor
diffs: dump Q, K, V, attention output, FFN output for a fixed
layer at the same step under both CPU and GPU, find which kernel
contributes the dominant drift, patch ggml-metal. The plumbing
for the dump hooks above can be extended to capture those
intermediates.

### Root cause located — ggml-metal `kernel_mul_mm` legacy path (May 2026)

Bisect on chatterbox-base (Q4_K, --no-gpu vs `CRISPASR_CHATTERBOX_FORCE_GPU=1`,
greedy seed=42, "Hello world", `CRISPASR_CHATTERBOX_DUMP_KV_AT=N`):

| Step                                                | CPU K[L0,h0,t]                  | GPU K[L0,h0,t]                  | abs diff       |
|-----------------------------------------------------|---------------------------------|---------------------------------|----------------|
| t=45 (last prefill, **no decode yet**)              | -0.1699 -0.1520 -0.3115 0.0020  | -0.1711 -0.1503 -0.3127 0.0031  | **~1e-3**      |
| t=46 (first decode K)                               | 0.1315 -0.0494  0.0238 -0.0616  | 0.1318 -0.0494  0.0226 -0.0604  | ~1e-3          |
| t=50 (after 5 decode tokens — diverged trajectories)| -0.0811  0.1787  0.2081 0.1721  | 0.0555  0.1400  0.2847 0.1032   | ~1e-1          |

CPU and GPU **already differ by ~1e-3 at the end of prefill** — long
before any decode steps run. The drift originates in the prefill
matmul, not in decode-loop accumulation.

Setting `CRISPASR_KV_QUANT=F32` to take the F16 KV-cache cast out of
the picture: GPU still differs from CPU by the **same** 1e-3. So it's
not the F32→F16 cache write — the per-layer K projection is producing
different F32 values on Metal.

The Metal mul_mm dispatch for chatterbox prefill: ne11 = 46 > 8, ne00
% 128 == 0, so `mul_mm` is selected (legacy path on M1-M4 because
`has_tensor` is gated on M5/A19+ in `ggml-metal-device.m:669-676`).

The bug is in `ggml/src/ggml-metal/ggml-metal.metal` legacy
`kernel_mul_mm`, line 9590:

```cpp
*(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)(*((device T1_2x4 *) y));
```

For all `kernel_mul_mm_*_f32` instantiations, `S1=half` but `T1=float`.
This **explicitly rounds the F32 input B to F16 when staging into
shared memory** before the simdgroup matmul. The accumulator
`mc[i] = simdgroup_float8x8` is F32 (correct), but both operands are
half tiles (`S0_8x8 = S1_8x8 = simdgroup_half8x8`), so the product is
half × half multiplied and F32-accumulated — the F16 input rounding
loses precision the CPU path retains.

Magnitude check: F16 has ~3e-4 relative spacing at value 0.1, so each
of K=1024 input elements rounds with absolute error ~3e-5; summed
over the dot product, RMS drift is √1024 × 3e-5 ≈ 1e-3. **Matches
observation exactly.**

CPU `ggml_compute_forward_mul_mat` performs F32 × F32 → F32 multiplies
with no input rounding (the weight is dequantised to F32 then a
plain dot product is run), which is why CPU is ~1e-3 closer to ground
truth than Metal on M1-M4.

### Why other models tolerate it

Whisper, parakeet, qwen3, voxtral etc. also hit `kernel_mul_mm_*_f32`
and the same 1e-3 drift on K-dim ~768-1024 dot products. They appear
fine because (a) they run argmax / greedy / beam decoding, where logit
drift below ~1e-2 doesn't flip the top token, and (b) the drift
doesn't compound across decode steps when the sampled token is the
same as the CPU-sampled one. Chatterbox is unique in this codebase
because:

1. multinomial sampling on a 8194-vocab speech-token distribution
   amplifies tiny drifts into different sampled tokens once the
   probability mass crosses ~5e-2 between candidates;
2. the divergent token then enters the KV cache — so subsequent
   decode steps see slightly different K, V values, drift compounds;
3. unlike text models, the autoregressive speech decoder has no
   self-correcting language-model prior pulling trajectories back
   together.

So the drift exists on every Metal user but only chatterbox notices.

### The proper upstream fix

Add `_hp` (high-precision) variants of `kernel_mul_mm_*_f32` that use
`S0=S1=float` and `simdgroup_float8x8` operand tiles. The
`simdgroup_multiply_accumulate(mc, mb, ma, mc)` then does
F32 × F32 → F32 with no input rounding, matching CPU. Dispatch by
honoring the existing `GGML_PREC_F32` `op_params[0]` flag in
`ggml_metal_library_get_pipeline_mul_mm`. Mark the chatterbox T3
QKV/output/FFN projections with `ggml_mul_mat_set_prec(...,
GGML_PREC_F32)` so they use the precise kernel; other backends keep
the existing legacy half-multiply behaviour and pay no perf tax.

### Attempted patch (2026-05-10) — did not fix the drift

Implemented the above plan: added `kernel_mul_mm_hp` template alongside
the legacy kernel (8192-byte sa offset, `simdgroup_float8x8` tiles for
both operands, `*(threadgroup float2x4 *)(sb + ...) = *((device
float2x4 *) y)` with no F16 cast). Wired up F32, F16, Q4_K, Q5_K, Q6_K,
Q8_0 host_name instantiations. Patched `mul_mm` dispatch to append
`_hp` to the pipeline name when `prec_f32 && !has_tensor && tsrc1 ==
F32`. Added a graph-walk in chatterbox `build_graph_t3_kv` /
`build_graph_t3_gpt2_kv` that calls `ggml_mul_mat_set_prec(...,
GGML_PREC_F32)` on every `GGML_OP_MUL_MAT` node post-build.

Verified the hp pipeline IS dispatched for the chatterbox K projection
(q4_K × f32, ne00=1024, ne01=1024, ne11=46) via a one-shot debug
print, AND that the kernel actually runs (overwriting `mc[0]` with a
known constant changed the LGT output away from the legacy GPU
values). But the K[L0,h0,t=45] cache values were **bit-for-bit
identical** to the pre-patch GPU run — same 1e-3 drift vs CPU.

Possible explanations:

1. **Apple's `simdgroup_float8x8` MAC silently downconverts.** The HW
   tensor cores on M1-M4 may multiply in half precision regardless of
   operand declared type, with float only for the accumulator. There
   is no public docs guarantee that `simdgroup_multiply_accumulate
   (simdgroup_float8x8&, simdgroup_float8x8, simdgroup_float8x8,
   simdgroup_float8x8)` is bit-equivalent to a pure F32 dot product —
   it may be implemented as `float8x8 = float(half(a) × half(b)) +
   float8x8`.
2. **The cache write happens via a different code path.** The KV cache
   in chatterbox is allocated `on cpu` (verbose log line). The
   ggml-backend scheduler may route the K projection mul_mat to CPU
   (since its consumer — the cpy to CPU cache — runs on CPU), bypassing
   the hp kernel. But the empirical CPU vs GPU divergence (~1e-3 even
   with FORCE_GPU=1) suggests this isn't the case for at least the
   path that the dump observes.
3. **Drift is somewhere else entirely.** RMSNorm reduction order,
   rope sin/cos precision, or the F32→F16 cast in the cpy kernel —
   though we ruled out the F16 cast via `CRISPASR_KV_QUANT=F32`.

Decision: keep the hp kernel + dispatch infrastructure in place — it
costs no perf when prec is not F32 (default), and provides a
foundation for future investigation. Auto-fallback to CPU remains the
working path for chatterbox.

The next investigation step is a proper per-op intermediate-tensor
dump: compare `norm(x)` output, `K_proj` output (pre-rope), and
`K_rope` output element-by-element between CPU and GPU at layer 0
position 45. That will identify which specific op contributes the
drift. The dump hooks at `chatterbox.cpp:1199-1234` can be extended
with a similar `CRISPASR_CHATTERBOX_DUMP_NORM_AT` /
`DUMP_KPROJ_AT` / `DUMP_KROPE_AT`.

Tracked as a follow-up in PLAN.md #83.

### Methodical bisect, round 2 (2026-05-10) — drift is in mul_mat algorithm

Added per-op intermediate dumps (`CRISPASR_CHATTERBOX_DUMP_NORM_AT`,
`DUMP_KPROJ_AT`, `DUMP_KROPE_AT`, `DUMP_WK`) in
`chatterbox.cpp:build_graph_t3_kv` that name and surface the layer-0
RMSNorm output, K projection mul_mat output, K rope output, and the
dequantized K weight tensor. Run on CPU and GPU at the same step,
diff per element. Findings:

| stage              | CPU                                | GPU (FORCE_GPU)                    | drift |
|--------------------|------------------------------------|------------------------------------|-------|
| `L0_norm_out` t=45 | -0.0589 -0.0123 -0.0126 -0.0622 …  | -0.0589 -0.0123 -0.0126 -0.0622 …  | **0** (norm bit-identical) |
| `L0_W_K_f32` row=0 | -0.012791 -0.039632 -0.012791 …    | -0.012791 -0.039632 -0.012791 …    | **0** (Q4_K dequant bit-identical, after fixing `.h` → `.f` literals) |
| `L0_K_proj` t=45   | -0.3584  0.0566  0.2575 -0.0085 …  | -0.3579  0.0566  0.2551 -0.0067 …  | **~1e-3** (the matmul is the culprit) |
| `L0_K_rope` t=45   | -0.1699 -0.1519 -0.3115  0.0020 …  | -0.1712 -0.1503 -0.3128  0.0031 …  | ~1e-3 (rope just propagates) |

So norm and dequant are bit-identical CPU/GPU; the drift is **purely in
the K projection mul_mat itself**.

Then the orthogonal test: load chatterbox-t3-f16 (F16 weights, NOT
Q4_K). Run CPU vs GPU. K_proj output now bit-identical between CPU
and GPU at every position tested through t=70+. KV cache values also
bit-identical. So the matmul is correct **for F16 inputs**.

Forcing the dispatch through `mul_mv_ext` (the F32-precise
`dot(float4,float4)` path) for `PREC_F32` ops, then casting Q4_K to
F32 before the matmul (`ggml_cast(W, F32)` followed by F32×F32
matmul): **same ~1e-3 drift**. So even when the GPU does
F32×F32 matmul on dequantized Q4_K weights, it doesn't match the
CPU's Q4_K matmul.

The real reason: **CPU and GPU implement Q4_K matmul differently.**
CPU's path is `ggml_vec_dot_q4_K_q8_K` (`ggml-cpu/quants.c:645`) —
it quantizes the F32 input to Q8_K (8-bit-per-element block-quantized)
before the dot product, then computes the dot using packed integer
multiplies + scale factors. GPU's path is `kernel_mul_mv_q4_K_f32`
(`ggml-metal.metal:7748`) — it keeps F32 input and multiplies by the
unpacked Q4_K nibbles directly. Both are valid but produce different
F32 outputs in the ~1e-3 range, and **chatterbox's multinomial
sampler is sensitive enough to flip token selection**.

Q8_0 weights show the same CPU/GPU drift (~1e-3) confirming the issue
isn't Q4_K-specific — it's any quantized weight type with input-quant
on CPU vs F32-input on GPU.

### Patch landed (partial fix, useful infrastructure)

1. `dequantize_q4_K` rewritten to mirror CPU's
   `dequantize_row_q4_K` arithmetic exactly: `dl * (q & 0x0F)` for
   the low nibble and `dl * (q >> 4)` for the high nibble, with
   `d` and `dl` always F32. Was previously `(d/16.h) * sc * (q &
   0xF0)` — F16 division, mathematically equivalent but rounds
   differently in F32. After the fix, CPU and GPU dequant are
   bit-identical (verified element-by-element on row 0, 256
   weights). Small but principled improvement.
2. `kernel_mul_mv_ext` dispatch in `ggml-metal-ops.cpp` now honours
   `GGML_PREC_F32` even for `ne11 > 8` (legacy path on M1-M4 only —
   tensor API has its own behaviour). Picks `r1ptg=4` for the
   PREC_F32 batch case. Allows Vulkan-style PREC_F32 → high-precision
   kernel routing. **Doesn't fix the algorithmic CPU/GPU mismatch
   for quantised weights** — F32 dot product on dequantised Q-weights
   still differs from CPU's Q8_K-quantized dot product.
3. `kernel_mul_mm_hp` and the chatterbox `ggml_mul_mat_set_prec(...,
   GGML_PREC_F32)` graph walk land as planned; harmless when the
   matmul falls back to legacy half-tile (other backends).
4. `CRISPASR_METAL_STRICT_FP=1` knob disables Metal fast-math
   (`setFastMathEnabled:NO`). Tested — doesn't change the drift, so
   it's not an FMA/fusion issue.
5. Per-op intermediate dump knobs in `chatterbox.cpp` for future
   bisects: `CRISPASR_CHATTERBOX_DUMP_NORM_AT=<t>`, `DUMP_KPROJ_AT`,
   `DUMP_KROPE_AT`, `DUMP_WK`.

### Remaining work

The proper fix is a Metal `kernel_mul_mv_q4_K_q8_K` (Q4_K weights ×
Q8_K-quantised input) mirroring CPU's `ggml_vec_dot_q4_K_q8_K`. That
needs:

1. A Q8_K quantisation kernel that block-quantises F32 input to Q8_K
   (256-element blocks, one F32 scale per block, plus per-16-element
   subscale).
2. A new Metal mul_mv (and mul_mm matrix variant) that consumes
   Q8_K input and Q4_K weight, doing integer multiplies + F32 scale.

This is ~200-400 lines of Metal kernel code for **each** quant pair
(Q4_K×Q8_K, Q5_K×Q8_K, Q6_K×Q8_K, Q8_0×Q8_0, ...). A rabbit hole. The
auto-fallback to CPU remains the practical fix for chatterbox.

For users who want GPU performance: convert chatterbox to F16 weights
(`chatterbox-t3-f16-regen.gguf`) — F16×F32 matmul matches between
CPU and GPU bit-for-bit, so chatterbox runs correctly on GPU. The
caveat: **end-to-end F16+GPU still has audible artefacts** because
something downstream of T3 (likely s3gen or speech_head F32 matmul
through some indirect path) introduces additional drift. So even F16
isn't a clean GPU path. Auto-fallback remains.

### Round 4 (2026-05-10) — kernel-level fix landed (partial bit-identity)

Implemented the proper kernel-level fix:

1. `kernel_quantize_q8_K_f32` (ggml-metal.metal): F32 input column →
   Q8_K block. Each threadgroup processes 1 256-element block, 32
   threads per group. simd_shuffle_xor reduction finds amax + signed
   max, then `iscale = -127/max`, `qs[j] = MIN(127, round(iscale ×
   x[j]))`, `bsums[k] = sum of qs[k*16..k*16+15]`. Mirrors CPU
   `quantize_row_q8_K_ref` exactly.
2. `kernel_mul_mv_q4_K_q8_K` (ggml-metal.metal): one thread per output
   element. Mirrors CPU `ggml_vec_dot_q4_K_q8_K_generic` exactly:
   unpack 256 Q4_K nibbles into int8 a[256], unpack 12-byte scales
   table into 8 scale + 8 min uchars (kmask1/2/3), int32 accumulators
   per scale lane, dmin contribution via bsums × mins, final F32
   multiply by d_q4 × d_q8.
3. Dispatch path in `ggml_metal_op_mul_mat`: when PREC_F32 + Q4_K
   weight + F32 input + ne00 % 256 == 0, reserve a Q8_K scratch
   buffer at the tail of dst (via
   `ggml_metal_op_mul_mat_extra_q8_K` + the
   `ggml_backend_metal_buffer_type_get_alloc_size` hook), dispatch
   the quantize kernel to fill it, sync via
   `ggml_metal_op_concurrency_reset`, dispatch the Q4_K×Q8_K matmul.
4. Routed `flash_attn_ext` to CPU when PREC_F32 is set, via
   `ggml_metal_device_supports_op` returning false. Apple's FA kernel
   uses simdgroup_half8x8 tiles for Q×K^T regardless of K type
   (FA_TYPES_F32 still declares Q as half), leaking ~1e-4 drift even
   with F32 KV. Routing to CPU avoids the issue at the cost of
   per-layer cross-device transfers.
5. chatterbox `build_graph_t3_kv` / `build_graph_t3_gpt2_kv` graph
   walks now tag every `GGML_OP_MUL_MAT` AND every
   `GGML_OP_FLASH_ATTN_EXT` with `GGML_PREC_F32`.

### Verification + remaining drift

| Stage                   | CPU vs GPU drift                  |
|-------------------------|-----------------------------------|
| Layer-0 norm output     | bit-identical                     |
| Layer-0 K weight (cast) | bit-identical (after `.h` → `.f`) |
| Layer-0 K projection    | **bit-identical**                 |
| Layer-0 K cache (KV[0]) | **bit-identical** through every t |
| Layer-0 V projection    | bit-identical                     |
| Layer-0 Q projection    | bit-identical                     |
| Layer-0 K rope          | bit-identical                     |
| Layer-0 attn output     | **bit-identical** (post out_proj) |
| Layer-0 FFN output      | **bit-identical**                 |
| Layer-1 norm output     | bit-identical                     |
| Layer-1 K projection    | bit-identical                     |
| Layer-1 K cache         | bit-identical at every sampled t  |
| Layer-1 attn output     | **drifts ~1e-4** (despite all of the above being identical!) |
| Layer-29 attn output    | drifts ~3e-3                      |
| End-of-prefill logits   | drift ~0.1–0.2                    |

AR token sequence (chatterbox seed=42, "Hello world", greedy temp=0):
matches CPU through **decode step 11**, then diverges. Compare to:
- Pre-#83 fix:                  matches through ~step 1
- After kernel only:            matches through step 6
- After kernel + FA→CPU:        matches through step 11

End-to-end audio: chatterbox FORCE_GPU=1 now produces partially
intelligible speech ("They put your" instead of empty/noise on the
JFK clone). The auto-fallback remains the production path until the
remaining ~1e-4 layer-1+ drift is identified and eliminated — likely
in some Metal op the diagnostic dumps haven't yet traced through
(candidates: rope on Q with rope_freq_factors, swiglu, an
F32-weight matmul path I haven't covered, or an inter-kernel
cross-device copy with subtle precision behaviour).

### Status

The Q4_K × Q8_K kernel infrastructure is solid and ships with diagnostic
knobs (`CRISPASR_CHATTERBOX_DUMP_KV_LAYER`, `_DUMP_LAYER`, `_DUMP_ATTN_AT`,
`_DUMP_FFN_AT`, `_DUMP_QPROJ_AT`, `_DUMP_VPROJ_AT`) for further
bisection. Q5_K, Q6_K, Q8_0 use the same template — straightforward
port once Q4_K is shaken out.

The default chatterbox path (no env overrides) auto-falls-back to
full CPU. `CRISPASR_CHATTERBOX_FORCE_GPU=1` enables the new
kernel + FA→CPU path; tokens diverge at step 11+ but layer 0 is
bit-identical.

### Round 5 (2026-05-10) — FA-input bisect: every input bit-identical, output drifts

After the Round 4 kernel fix, layer-0 was bit-identical CPU/GPU
end-to-end (every named tensor: norm, K/Q/V proj, K rope, attn out,
FFN out — all match). Layer 1 K cache and V cache were also
bit-identical at every sampled position (t=0, 10, 20, 30, 40, 45) on
both halves. So the input to layer-1 attention is reliably
bit-identical. Yet **layer-1 attn out (post out_proj) drifts ~1e-4**
and propagates by layer 29 to ~3e-3, blowing up to ~0.1–0.2 in
end-of-prefill logits. AR token sequence matches CPU exactly through
decode step 11, then diverges.

Bisect-round-5 added named graph outputs in `core_attn::kv_self_attn`
for the FA inputs and outputs (`CRISPASR_CORE_ATTN_DUMP_FA_LAYER=N`):

| Stage at layer 1, t=45  | CPU vs GPU       |
|-------------------------|------------------|
| Q post-rope             | bit-identical    |
| Kfull (FA input)        | bit-identical    |
| Vfull (FA input)        | bit-identical    |
| **FA output**           | **drifts 1e-4**  |
| out_proj input (= FA reshape) | drifts 1e-4 (propagated) |
| attn out (post out_proj)| drifts 1e-4 (propagated) |

`flash_attn` was confirmed routed to CPU on every layer (5100 of 5100
T3 FA ops on CPU; the 30 Metal FA ops counted are S3Gen). So both
halves invoke `ggml_compute_forward_flash_attn_ext_f16` with
bit-identical Q, K, V, mask, scale. The function is deterministic, yet
output differs.

Forced single-thread (`-t 1`) didn't change the drift, ruling out
ggml-cpu's threaded reduction order. Tested with both default
multi-threaded and `-t 1`.

The puzzle: layer 0 stays bit-identical (FA inputs and output match),
but starting from layer 1 the FA output diverges despite identical
inputs. The split between layer 0 and layer 1 strongly suggests
something about the second-or-later FA invocation in a graph that
mixes Metal and CPU backends.

Possible explanations to chase next session:

1. **Memory ordering / barrier** — Apple unified memory shares physical
   buffers between CPU and GPU. When a CPU-routed op (FA) writes its
   output and a subsequent GPU-routed op (out_proj) reads it, the
   ggml-backend scheduler should emit a sync. If the sync is missing
   for the second-or-later layer, the GPU might read stale or
   half-flushed CPU writes.
2. **Backend-internal cache pollution** — ggml-cpu reuses internal
   thread-local scratch for FA's softmax. If that scratch was
   touched by a prior op and not cleared, the second FA invocation
   could pick up garbage in unaccounted bytes.
3. **Different CPU FA chunking depending on graph context** —
   `ggml_compute_forward_flash_attn_ext_f16` has multiple internal
   paths (one_chunk, tiled). The chunk decision uses `ne01` and
   thread count; if these differ between a CPU-only run and a
   Metal-routed run (different graph structure passed to the
   ggml-cpu backend), chunk boundaries shift and F32 reductions land
   on different roundings.
4. **`ggml_cont` produces F16 cache view differently on Metal** —
   even though I dumped K cache values and they matched at sampled
   positions, the layout/stride of the resulting Kfull contiguous
   tensor might differ, putting bytes in different addresses, which
   could affect SIMD-aligned loads in CPU FA.

Diagnostic infra ready for round 6:

- `CRISPASR_CORE_ATTN_DUMP_FA_LAYER=N` — names FA inputs/output of
  layer N as graph outputs.
- `CRISPASR_CORE_ATTN_DUMP_FA_AT=t` / `_Q_AT` / `_KFULL_AT` / `_VFULL_AT`
  — fetch and print the named outputs for position t.
- `CRISPASR_CHATTERBOX_DUMP_KV_LAYER=N`, `_DUMP_KV_AT=t`,
  `_DUMP_VV_AT=t` — KV cache dump per layer.
- `CRISPASR_CHATTERBOX_DUMP_LAYER=N`, `_DUMP_ATTN_AT=t`, `_DUMP_FFN_AT=t`,
  `_DUMP_NORM_AT=t`, `_DUMP_KPROJ_AT=t`, `_DUMP_QPROJ_AT=t`,
  `_DUMP_VPROJ_AT=t`, `_DUMP_KROPE_AT=t` — per-layer per-stage dumps.

### Benchmark (2026-05-10)

Measured end-to-end on chatterbox-base T3 Q4_K (\"Ask not what your
country can do for you, ask what you can do for your country\",
median of 3 runs):

| Path                                               | wall time |
|----------------------------------------------------|-----------|
| CPU (auto-fallback, default)                       | ~66 s     |
| GPU FORCE_GPU=1 (kernel_mul_mv_q4_K_q8_K + FA→CPU) | ~58 s     |

~12% speedup, no regression. End-to-end audio with FORCE_GPU=1
produces partially intelligible cloned speech (\"They put your\"
recognised by ASR vs the canonical \"Ask not what your country can
do for you...\" on the CPU path). The token sequence matches CPU
through decode step 11, then diverges due to the layer-1+ residual.

### Production default

Auto-fallback to full CPU remains the production default — the
~1e-4 layer-1+ drift is too small to hurt CPU output but flips
chatterbox's multinomial speech-token sampler around step 11+ on the
GPU path, producing audibly degraded clones. The
`CRISPASR_CHATTERBOX_FORCE_GPU=1` knob unlocks the new kernel for
debug/perf experimentation.

### Round 6 (2026-05-19) — ggml_mish red herring

Round 4–5 stayed focused on T3; meanwhile shipping the round-4 patches
to T3 cleaned up the user-audible artefact the diagnostics had been
chasing in T3, leaving a remaining "audio garbled with FORCE_GPU=1"
that round 6 traced to S3Gen rather than T3. The first suspect was
the hand-rolled `ggml_mish` in `chatterbox_s3gen.cpp`:

```cpp
exp_x = ggml_exp(x);
ones  = ggml_div(exp_x, exp_x);          // ← fabricates 1.0 from x/x
sp    = ggml_log(ggml_add(exp_x, ones)); // log(exp(x) + 1)
out   = ggml_mul(x, ggml_tanh(sp));
```

The `exp_x / exp_x` is NaN whenever `exp(x)` overflows to inf or
underflows to 0. Replaced with `ggml_mul(x, ggml_tanh(ggml_softplus(x)))`
using ggml's native `ggml_softplus` (single fused kernel, identical
clamp-at-x>20 on Metal and CPU). The fix is a small correctness
improvement (s3gen_mel cos_min 0.999971 → 0.999980 on CPU), but the
GPU drift is unaffected (FORCE_GPU stays at 0.923). Mish was not the
divergence source — kept for the NaN-safety gain.

### Round 7 (2026-05-19) — op-bisect: drift is compound, not single-op

Added `CRISPASR_S3GEN_UNET_PIN_CPU_OP=<op>` (pin only the named op to
CPU, rest on GPU) and `CRISPASR_S3GEN_UNET_KEEP_GPU_OP=<op>` (inverse —
pin everything else to CPU) inside `s3gen_maybe_pin_graph_to_cpu`.
Names follow `ggml_op_name()` lowercase minus the `OP_` prefix, or
`unary_<lowercase>` for `GGML_OP_UNARY`. Ran the matrix with
`CRISPASR_CHATTERBOX_FORCE_GPU=1` against `s3gen_mel` (diff harness,
`replay=exact_init_noise`):

| `PIN_CPU_OP=` | s3gen_mel cos_min |
|---|---|
| (none — full GPU) | 0.923 |
| `mul_mat` / `flash_attn_ext` / `norm` / `add` / `concat` / `unary_gelu` / `unary_tanh` / `unary_softplus` | **1.000000** |
| `conv_1d` / `scale` / `unary_mish` | 0.923 (no effect) |

**The bug is not in any single op.** Pinning *any* frequent op type to
CPU forces a F32 CPU↔GPU round-trip every N ops in the schedule and
breaks the drift chain. The non-effective entries are op types with
zero or near-zero count in `build_graph_unet1d` (mish lowers to
softplus+tanh+mul via round 6 → no `GGML_UNARY_OP_MISH` nodes).
**Re-verified `PREC_F32` tagging** on every UNet mul_mat: the
CrispASR `_hp` mul_mm dispatch + `mul_mv_ext` path are picked up
correctly on M1 (`has_tensor=false`), but cos stays at 0.923 — the
surrounding ops keep compounding. PREC_F32 fixes mul_mat alone, not
the chain. Reverted the tags as graph clutter that doesn't fix the
audible bug.

**Auto-pinning UNet under FORCE_GPU is not a viable workaround.** Diff
harness shows `cos=1.000` with `replay=exact_init_noise`, but the
full TTS pipeline (random noise + GPU encoder + GPU vocoder) produces
NaN/Inf mel going into the vocoder (`rms=nan min=1e30 max=-1e30`),
which the vocoder amplifies into saturated audio (parakeet ASR
returns empty transcript). Pinning encoder + vocoder + UNet all to
CPU under a GPU-initialised sched still gives `rms ≈ 11089` vs the
~3900 a clean CPU sched gives. Something about the GPU-backed
scheduler interacts badly with random-noise inputs through the
CPU-pinned compute path; the diff harness's pre-recorded reference
noise masked it. So **the vocoder is not the bug** — it's just
amplifying upstream garbage; the UNet1D is producing NaN under the
GPU-sched + random-noise path.

### Round 7b (2026-05-19) — Metal default is full CPU

Stable warm benchmark on M1 (3 runs, "Ask not what your country can
do for you, ask what you can do for your country", JFK speaker
clone, --seed 1):

| Config | wall time | ASR roundtrip |
|---|---|---|
| Full CPU | ~50 s | ✓ exact |
| T3 GPU + S3Gen CPU (old default) | ~75 s | ✓ exact |
| FORCE_GPU=1 (broken) | ~96 s | ✗ empty |

T3 on Metal is *slower* than T3 on CPU for chatterbox's batch-1 AR:
30 layers × 86 sequential tokens × ~10 ops/layer ≈ 25 k tiny kernel
launches, each paying µs-class Metal dispatch overhead that the M1
NEON CPU caches blast straight through. So the chatterbox default
was switched on Metal builds (`#ifdef GGML_USE_METAL`) to **full
CPU**. `CRISPASR_CHATTERBOX_T3_GPU=1` opts back into T3-on-GPU; the
non-Metal default (CUDA / Vulkan / etc.) is unchanged. Saves ~30 %
wall time on M1.

### Round 8 (2026-05-23) — CUDA P100 bisect confirms cross-backend compound drift

Ran the same `PIN_CPU_OP` / `KEEP_GPU_OP` sweep on Kaggle P100
(Tesla P100-PCIE-16GB, compute capability 6.0 / sm_60, CUDA 12.8)
using a self-contained Kaggle kernel that clones + builds CrispASR,
generates a ref GGUF on-device, downloads Q8_0 GGUFs, then runs 21
diff configs with `CRISPASR_DIFF_USE_GPU=1 CRISPASR_CHATTERBOX_FORCE_GPU=1`.

| Config | s3gen_mel cos_min |
|---|---|
| Full CPU (control) | 0.999955 |
| S3Gen on CUDA P100 baseline (no pin) | **0.858455** |
| `PIN_CPU_OP=mul_mat` | **1.000000** ← FIXES |
| `PIN_CPU_OP=flash_attn_ext` | **1.000000** ← FIXES |
| `PIN_CPU_OP=norm` | **1.000000** ← FIXES |
| `PIN_CPU_OP=add` | **1.000000** ← FIXES |
| `PIN_CPU_OP=concat` | **1.000000** ← FIXES |
| `PIN_CPU_OP=unary_gelu` | **1.000000** ← FIXES |
| `PIN_CPU_OP=unary_tanh` | **1.000000** ← FIXES |
| `PIN_CPU_OP=unary_softplus` | **1.000000** ← FIXES |
| `PIN_CPU_OP=unary_mish` | 0.858542 (no effect) |
| `PIN_CPU_OP=conv_1d` | 0.858348 (no effect) |
| `KEEP_GPU_OP=<any>` | None (crash — ggml tensor type constraint) |

**Conclusions:**

1. **The compound FP16 drift is not Metal-specific.** CUDA P100
   shows the same pattern (cos_min 0.858, worse than Metal's 0.923),
   with the same 8 ops fixing it when individually pinned to CPU.

2. **The fix is cross-backend.** Pinning `mul_mat` (or any of the 7
   other high-frequency FP16 op types) to CPU forces a dtype
   round-trip that breaks the drift chain through the 10-step CFM
   solver. The same `CRISPASR_S3GEN_UNET_PIN_CPU_OP=mul_mat`
   workaround that restores cos=1.000 on M1 works identically on
   CUDA P100.

3. **unary_mish and conv_1d are not the drift source.** Consistent
   with Metal — the mish op was replaced with
   `softplus+tanh+mul` in Round 6, so there are no `GGML_UNARY_OP_MISH`
   nodes in the UNet graph; conv_1d's fp32 path on CUDA is already
   exact.

4. **KEEP_GPU_OP all crash.** Running only one op type on GPU with
   the rest pinned to CPU triggers a ggml type-constraint violation
   (GPU output tensors can't feed CPU op inputs without explicit
   view/copy nodes that the UNet graph doesn't insert). Not a useful
   fix path.

5. **Next step for a real fix:** Instead of per-op CPU pinning (which
   adds round-trip overhead), promote the S3Gen UNet1D graph to
   FP32 on GPU (force `GGML_TYPE_F32` weight loading and remove the
   FP16 intermediate casts). This matches what PyTorch does on CUDA
   by default and avoids the compound rounding entirely. Alternatively,
   enable `GGML_SCHED_MAX_COPIES=1` with an explicit F32-accum
   prefix on each UNet block's matmul — whichever is cheaper.

6. **Per-op CPU pinning causes NaN in full-pipeline runs.** Attempting
   to auto-pin `mul_mat` to CPU via `ggml_backend_sched_set_tensor_backend`
   in a GPU-backed scheduler works correctly in the diff harness (small T=102,
   no prompt tokens: cos_min=1.0) but produces NaN in the full production
   pipeline (T≥392, 157 prompt tokens). Any mix of CPU and GPU compute ops
   in a GPU-backed scheduler causes GPU→CPU tensor copy synchronization
   failures for large graphs — the NaN appears at denoiser step 0.
   Per-op CPU pinning is therefore not safe for production. The production
   fix remains S3Gen on CPU (current default); the mathematically correct
   fix (UNet FP32 promotion) requires GPU-side kernel changes.

2. **T3 sampling can drift on long technical prompts**. The seed=0
   default is deterministic, but particular prompts produce
   degenerate output (e.g. "Stop, stop, stop" repetition or wholly
   unrelated text). Short, common phrases work reliably; if a prompt
   produces gibberish, try a different seed via
   `CRISPASR_CHATTERBOX_SEED=<n>` or shorten the input.

### Misdiagnosis worth recording

This entry's first draft claimed the atomic-native path produced
intelligible speech "with text drift" on a long technical prompt
("Native voice clone test, all five conditions installed
atomically."). That was wrong on two counts: (a) I was running with
the broken F16 + GPU path, and (b) the long technical prompt itself
triggers the T3 sampler-drift issue regardless of voice path. Both
issues pre-date this work but I missed them because my first
sanity check used a contaminated combination. The correct
verification command is:

```bash
./build/bin/crispasr --backend chatterbox \
  -m /path/to/chatterbox-t3-q4_k-regen.gguf \
  --codec-model /path/to/chatterbox-s3gen-q8_0.gguf \
  --no-gpu \
  --voice <ref>.gguf-or-24kwav \
  --tts "Ask not what your country can do for you." \
  --tts-output out.wav
```

The parity-quality compute kernels (VE / S3Tok / CAMPPlus / 24 kHz
mel) all remain bit- or fp32-rounding-tight against PyTorch when
fed identical bytes via the diff harness's `audio_24k_input`
bypass — that part of the work is unaffected by the F16+GPU bug
and the sampler drift.

### CLI

```bash
# Full atomic native clone — 24 kHz mono PCM16/F32 WAV input.
./build/bin/crispasr --backend chatterbox -m auto \
    --voice ref_24k.wav \
    --tts "Hello there, this is the cloned voice." \
    --tts-output cloned.wav

# Partial T3-side-only clone — 16 kHz mono WAV input.
./build/bin/crispasr --backend chatterbox -m auto \
    --voice ref_16k.wav \
    --tts "..." --tts-output ...
```

The runtime prints exactly which conds are installed at verbosity ≥ 1.
For perfect parity with the python baker (full quality), the existing
`models/bake-chatterbox-voice-from-wav.py` workflow remains
recommended; the native path is the no-python-required alternative.

## T5-family translation runtime traps (May 2026, MADLAD-400 debugging)

Bringing up the T5 encoder-decoder runtime (`src/t5_translate.cpp`)
on MADLAD-400 surfaced four bugs in sequence. Each one looked like
"the runtime is broken" until carefully diff-tested against a Python
reference (flan-t5-small via `transformers.T5Tokenizer` +
`T5ForConditionalGeneration`). Capturing here so the next T5 / SP
port doesn't repeat them.

### 1. Bidirectional rel-pos bucket: FUTURE/PAST halves swapped

The encoder's bidirectional rel-pos bucketing has two halves —
buckets `[0..N/2-1]` for past+self, `[N/2..N-1]` for future. Earlier
C++:

```cpp
int n = -rel_pos;
ret += (n < 0) ? 0 : num_buckets;   // adds num_buckets when n>=0
                                    // (rel_pos<=0, PAST/SELF) — WRONG
```

Canonical T5 (HF):

```python
relative_buckets += (relative_position > 0) * num_buckets   # FUTURE
```

Symptoms: encoder output was wrong by a per-head learned-bias offset.
Decoder cross-attention then had wrong keys at every position →
degenerate loop on most-frequent tokens (the "rel-pos repeating-token
loop" behavior). Fix: drop the sign flip, use `rel_pos > 0` directly.
The unidirectional decoder branch (only past valid under the causal
mask) was already correct — only the encoder's bidirectional path
was wrong.

### 2. Special-token IDs vary across T5-family models

C++ tokenizer hardcoded `<unk>=2`, `</s>=1`. That's correct for
flan-t5 / mT5, but MADLAD-400 has **different IDs**:

```
flan-t5: <pad>=0, </s>=1, <unk>=2
MADLAD:  <unk>=0, <s>=1,  </s>=2
```

Hardcoded `ids.push_back(2)` as the unk-fallback in tokenize_sp
emitted MADLAD's `</s>` (= EOS) instead of `<unk>`, prematurely
terminating the encoder input. Hardcoded trailing `ids.push_back(1)`
emitted MADLAD's `<s>` (= BOS) instead of `</s>` — model never saw
EOS at end of input.

Fix: read `t5.eos_token_id` from GGUF metadata (already in the
loader) and propagate to `tokenizer.eos_id`; look up `<unk>` in
the vocab to get `tokenizer.unk_id` dynamically. Both used in
tokenize_sp instead of literal IDs.

### 3. Greedy-longest-match ≠ SentencePiece Unigram

Multi-byte special tokens like MADLAD's `<2de>` (id 33) get
mis-tokenized by greedy:

```
input:  ▁<2de>▁Hello
greedy: [▁<](4411) + [2](810) + [de](948) + [>](3048) + [▁Hello](88912)
                                                        # 4 garbage tokens
SP:     [▁](805) + [<2de>](33) + [▁Hello](88912)         # 2 + 1
```

Greedy picks `▁<` because it's a longer byte match than `▁` alone.
SP unigram picks `<2de>` because that piece's score (lang tags have
very high SP scores) dwarfs the per-byte fragment scores.

Without the right `<2de>` token in the encoder input, MADLAD has no
language signal and emits whatever its language prior dominates on
(in our run: Hebrew when asked for German).

Fix: load `tokenizer.ggml.scores` from the GGUF and replace greedy
with Viterbi best-segmentation — DP over byte positions, keeping
the highest total log-prob. Codepoint-aligned (skip end positions
that fall on a UTF-8 continuation byte). Single-byte fallback to
`<unk>` with a heavy penalty so Viterbi only chooses byte-fallback
when no piece covers the byte.

### 4. The `<2xx>` lang prefix is MADLAD-specific

The CLI adapter unconditionally prepended `<2{tgt_lang}>` to every
translation. MADLAD's vocab has all 419 lang tags as single-piece
entries; flan-t5 / mT5 / etc. don't. Prepending the prefix on those
tokenizes (after Viterbi) as `[▁, <unk>]` (= garbage) at the front
of the encoder input.

Fix: new `t5_has_token(ctx, "<2de>")` C ABI; the adapter probes the
vocab and only prepends the tag when it's a real piece.

### Validation methodology

For each fix, the regression test was: tokens AND output text both
bit-identical between C++ and Python reference. flan-t5-small is the
ideal smoke target — same architecture as MADLAD (T5-1.1 + gated-GELU
+ RMSNorm + bucketed rel-pos + SentencePiece) but ~250 MB so it
loads fast on tight memory. Once flan-t5-small matches, MADLAD
matches by construction (same kernels, same algorithms, larger model).
Save the next round of T5 debugging by validating on flan-t5-small
first.

## 2026-05-05 — ggml fork patches we carry (must re-apply on every ggml bump)

Authoritative inventory of every CrispASR-local change to the vendored
`ggml/` subtree. Most files have a `// CrispASR patch ...` marker so a
mechanical bump (`git subtree pull` or equivalent) won't silently lose
them, but the marker is only a tripwire — the actual fixes are listed
here. **Grep for both `CrispASR patch` AND `CrispASR fork` after every
bump** (the conv-graph patches in (5) below use the latter prefix; an
inventory grep that misses them ships a half-applied F16 fix and crashes
kokoro F16 CPU TTS at `ggml_backend_sched_split_graph`). ggml has lost
our patches twice already: commit `1552434` first added the im2col fix,
`ca6c523` re-applied it after the 0.9.8 → 0.10.0 bump dropped it; the
master bump on 2026-05-05 surfaced (5) as a missed inventory item the
same way.

Mirror this list in `UPSTREAM.md` so future-us (or whoever bumps next)
knows which of these are candidates to send upstream.

### 1. F16 weight × F32 input → F32 dot product (issue #38)

Files: `ggml/src/ggml-cpu/{vec.cpp, vec.h, ggml-cpu.c, simd-mappings.h}`.

Upstream `MUL_MAT` with `src0=F16, src1=F32` quantises the F32 input to
F16 first. F16's dynamic range tops out at ±65504, so any activation
above that saturates to ±Inf and feeds NaN into the next layer. The
qwen3-tts code-prediction path (and several conformer encoders on long
inputs) routinely produces logits well past 65504, masking precision
issues as outright NaN spirals.

Fix: introduce `ggml_vec_dot_f16_f32` that loads the F16 weight, casts
it to F32, and accumulates in F32 — no quantisation step on the input
side. `vec_dot_type` for F16 is set to F32 so `MUL_MAT` takes this path
without an intermediate quantize. ARM NEON path uses `vcvt_f32_f16` +
FMA; AVX2/AVX-512 paths use `_mm256_cvtph_ps` + FMA. Verified on the
qwen3-tts codec head where the saturation surfaced.

Symptom if lost: silent NaN propagation, decoder produces `<unk>` or
locks onto a single token. Won't show up in unit tests if the tests
use small synthetic inputs that stay inside F16 range.

### 2. CUDA `im2col` grid_y > 65535 (`ggml/src/ggml-cuda/im2col.cu`)

Upstream uses `OW` as `block_nums.y` directly. CUDA caps grid Y at
65535; SEANet-style encoders with 11-second 16 kHz inputs land at
`OW = 176000`, busting the cap and triggering an abort *or* (worse) a
silent partial copy depending on driver behaviour.

Fix: clamp `block_nums.y/z` to `MAX_GRIDDIM_Y = 65535` at dispatch and
loop inside the kernel with stride `gl_NumWorkGroups.{y,z}` so each
thread covers `ceil(OW/65535)` output positions. Kernel-internal stride
loop, single launch — possible because `im2col` has no shared-memory
state between iterations.

Symptom if lost: any conv encoder with `T_out > 65535` aborts on CUDA
and the supervisor restarts the process. CPU is unaffected.

### 3. CUDA cpy_scalar_transpose grid_y > USHRT_MAX (`ggml/src/ggml-cuda/cpy.cu`, GH issue #65)

Same class of bug as (2) but in the *transposed* cpy path used by
`ggml_cont(ggml_transpose(...))`. Asserts `grid_y < USHRT_MAX` (= 65535)
inside `ggml_cpy_scalar_cuda`'s transposed branch. The qwen3-tts codec
graph emits `[T_pcm, 1, 1, 1]` tensors with `T_pcm = T_codec * 1920` ≈
2.88M when `QWEN3_TTS_CODEC_GPU=1` and the talker hits its 1500-frame
cap, so `grid_y = ceil(T_pcm/32) ≈ 90,000` busts the assert and the
process aborts with `GGML_ASSERT(grid_y < USHRT_MAX)`.

Fix: tile the launch along the y axis. Add an `int y_block_offset`
parameter to `cpy_scalar_transpose`; the host loops in chunks of
`MAX_GRID_Y = USHRT_MAX-1` and each launch covers a y-slab
`[y_block_offset, y_block_offset + grid_y_this)`. The kernel splices
`blockIdx.y` back onto the offset (`by = blockIdx.y + y_block_offset`).
Bit-identical output, full transposed-tile coalescing preserved per
chunk. Multi-launch (vs the in-kernel-stride pattern used in (2))
because the transposed kernel relies on `__shared__` tile state per
launch — folding the chunk loop inside the kernel would break the
`cur_tile_buf` toggle.

The first attempt (commit `eb9e4a2`) shipped a scalar fallback when
the assert would fire. That worked but threw away the transposed-tile
coalescing entirely on any shape that tripped it; commit `2639461`
replaced it with proper tiling.

Other CUDA-class backends — HIP / MUSA — share `ggml-cuda/cpy.cu` via
the `vendors/` shim and inherit the fix. Vulkan's `copy_transpose.comp`
already has an in-kernel stride loop and is unaffected. Metal's
`kernel_cpy_t_t` doesn't tile and Apple GPUs don't have the 65535 cap.

Symptom if lost: any large-T audio codec / vocoder graph on CUDA
(qwen3-tts, future SNAC/Encodec/XTTS ports) aborts with
`GGML_ASSERT(grid_y < USHRT_MAX)` once `T_codec * upsample_total >
~65535*32`.

### 4. Metal `kernel_conv_transpose_1d` input-range tightening (`ggml/src/ggml-metal/ggml-metal.metal`)

Upstream's transposed conv1d kernel iterates the full IL input range
per output position and filters with `if (...)`. For the qwen3-tts
codec decoder block 1 (IL=320, K=10, s0=5) that's 64× more iterations
than necessary; on M1 with the codec running at full T_codec, the GPU
watchdog fires `kIOGPUCommandBufferCallbackErrorImpactingInteractivity`
and the kernel is killed.

Fix: compute `i_min, i_max` analytically as the input positions whose
kernel weight `k = j - i*s0` lands inside `[0, K)`, then iterate only
that range. At most `ceil(K/s0)` iterations per output position
(typically 2 for stride==K/2 transposed convs). Bit-identical output,
~K/s0× speedup, watchdog-safe.

Symptom if lost: long qwen3-tts (or any large transposed-conv) graph on
Metal triggers GPU watchdog and the command buffer is killed mid-graph.
CUDA / CPU / Vulkan are unaffected — they don't have the same
per-command-buffer watchdog.

### 5. ggml conv graph builders F32 cast (`ggml/src/ggml.c`, issue #38 companion)

Files: `ggml_conv_1d`, `ggml_conv_1d_dw`, `ggml_conv_2d`, `ggml_conv_2d_dw`
in `ggml/src/ggml.c`. Marked `// CrispASR fork:` (different prefix from
the others — this is what the original inventory grep missed).

These four conv graph builders are the partner to (1). After (1) sets
`vec_dot_type = F32` for `GGML_TYPE_F16`, the CPU MUL_MAT path can no
longer accept F16 src1 — `ggml_compute_forward_mul_mat`'s line
`GGML_ASSERT(src1->type == GGML_TYPE_F32)` fires when a non-matching
src1 needs conversion. Upstream's conv graph builders hardcode
`im2col_type = GGML_TYPE_F16` and feed the kernel weight in directly,
producing `MUL_MAT(F16, F16)` — which `ggml_backend_cpu_device_supports_op`
rejects under (1), causing `ggml_backend_sched_split_graph` to abort
with `GGML_ASSERT(*cur_backend_id != -1)`.

Fix: inside each conv builder, pick im2col output type by whether either
side is F32; if im2col is F32 and the kernel is non-F32, `ggml_cast` the
kernel to F32 so the resulting MUL_MAT has F32 src1.

```c
const enum ggml_type im2col_type =
    (a->type == GGML_TYPE_F32 || b->type == GGML_TYPE_F32) ? GGML_TYPE_F32 : GGML_TYPE_F16;
struct ggml_tensor * a_mat =
    (im2col_type == GGML_TYPE_F32 && a->type != GGML_TYPE_F32) ? ggml_cast(ctx, a, GGML_TYPE_F32) : a;
```

Bandwidth cost is real (extra F16→F32 cast per inference pass) but
mirrors what conv_1d already does and is the price of (1)'s correctness
gain.

Symptom if lost: kokoro F16 TTS on `--gpu-backend cpu` aborts at
`ggml_backend_sched_split_graph: GGML_ASSERT(*cur_backend_id != -1)`,
trying to schedule `MUL_MAT(F16 reshape, F16 conv1.weight)` from the
F0N predictor. Also reproduces in any model whose graph runs
`ggml_conv_1d`/`ggml_conv_2d` with F16 weights against F32 (or no-cast)
input on the CPU backend with patch (1) applied.

### Bump procedure

```bash
# Before the bump — grep BOTH marker prefixes
grep -rnE "CrispASR (patch|fork)" ggml/ > /tmp/patches-before.txt

# Do the bump (or replace ggml/{CMakeLists.txt,LICENSE,cmake,include,src} from
# a fresh clone if the subtree wasn't originally added via `git subtree add`)
git subtree pull --prefix=ggml https://github.com/ggml-org/ggml master --squash

# After the bump
grep -rnE "CrispASR (patch|fork)" ggml/ > /tmp/patches-after.txt
diff /tmp/patches-before.txt /tmp/patches-after.txt
# If any patch is missing, find the original commit and cherry-pick the hunk.
```

Anything that disappears from the diff is a patch ggml's master
silently overwrote — re-apply from this list. The five patches above
are the full inventory as of 2026-05-05. Note that (1) and (5) are
**coupled**: applying one without the other crashes kokoro F16 CPU at
`ggml_backend_sched_split_graph`. Always re-apply them together.

### OmniASR-LLM-Unlimited: streaming segment-token protocol

The "Unlimited" variant (`omniASR_LLM_Unlimited_300M_v2`) uses a streaming
protocol where long audio is split into 15-second segments, decoded one at
a time. This uses 3 special tokens allocated above vocab_size in tok_emb:

- `streaming_lang` (vocab_size = 10288): replaces lid_marker in standard model
- `last_segment` (vocab_size + 1 = 10289): signals "this is the final audio segment"
- `regular_segment` (vocab_size + 2 = 10290): signals "more segments follow"

**tok_emb size**: Standard model has vocab_size+1 entries (extra lid_marker).
Unlimited has vocab_size+3 entries (streaming_lang + last_segment + regular_segment).
Auto-detection: `tok_emb->ne[1] == vocab_size + 3`.

**Prefix structure** (per segment):
```
[audio_embs] [streaming_lang] [lang_emb] [segment_marker] [BOS] → generate until EOS
```

The segment marker tells the model whether to expect more audio segments
after this one. EOS=2 is still emitted to terminate each segment's text.
Without the segment marker in the prefix, the model never sees the input
shape it was trained on and generates until max_new_tokens.

**Multi-segment**: Split encoder output at 750-frame boundaries (15s × 16kHz
÷ 320 CNN stride). Each segment gets an independent KV cache and decodes to
EOS. Results are concatenated.

**Critical**: The 476 extra vocab tokens (9812→10287) are NOT segment tokens —
they are additional text tokens in the v2 tokenizer. The 3 segment protocol
tokens sit above the full vocab at indices 10288–10290.

### FastConformer: flash attention with Shaw RPE

The FastConformer self-attention uses Shaw-style relative position
encoding with untied biases:
```
scores = (Q + pos_bias_u) × K^T + rel_shift((Q + pos_bias_v) × R^T)
```

This looks incompatible with `ggml_flash_attn_ext` since the position
bias is query-dependent. However, it CAN be decomposed:
- Precompute BD = rel_shift(Q_v × R^T) — one matmul, unavoidable
- Pass BD × scale as the additive mask to flash_attn_ext
- flash_attn_ext then computes: softmax(Q_u × K^T × scale + mask) × V

This replaces 2 matmuls + add + softmax + 1 matmul with 1 matmul + 
flash_attn_ext. On GPU (CUDA/Vulkan/Metal) the fused kernel avoids
materializing the T×T attention matrix and reduces kernel launches
from 96 to 32 per encoder pass.

Key pitfall: rel_shift returns a strided view — must `ggml_cont` before
`ggml_cast` to F16 for the mask, otherwise `ggml_is_padded_1d` asserts.

### Silero LID label format

Silero LID's `lang_dict_95.json` maps indices to "xx, Name" strings
(e.g. "de, German", "en, English"). The GGUF stores these verbatim in
`silero_lid.lang_strs`. Downstream backends expect bare ISO codes.
Must extract the part before the comma.

## Text LID via fastText — GlotLID-V3 + LID-176 (May 2026)

`src/lid_fasttext.{h,cpp}` ports both fastText supervised LID
families behind one C ABI: GlotLID-V3 (flat softmax, 2102 ISO 639-3
+ script labels, Apache-2.0) and Facebook LID-176 (hierarchical
softmax, 176 ISO 639-1 codes, **CC-BY-SA-3.0** — viral; redistributors
of the .gguf inherit SA). Released as `cstr/glotlid-GGUF` and
`cstr/fasttext-lid176-GGUF` on the Hub.

Forward path is manual F32/F16 + on-the-fly dequant via
`ggml_get_type_traits(type)->to_float`, no graph. The compute is
~1 MFLOP per call; a graph would be pure overhead. K-quants land via
`crispasr-quantize` post-processing.

### `</s>` row injection in fastText supervised mode (the trap)

fastText's `Dictionary::getLine` injects an `</s>` (always
`input_matrix[0]`) at end-of-stream in supervised mode, and
`initNgrams` skips subword expansion for it so its precomputed row
list is just `[0]`. `model.f.tokenize(text)` does NOT return it — a
manual reproduction that just iterates over tokenized words gets 11
row IDs for `"the"` while `model.get_sentence_vector("the")` mean-pools
12 rows.

**Symptom**: cosine vs `model.get_sentence_vector` lands ~0.973
instead of 1.0, with a non-constant ratio across dimensions (so it's
not a divisor mistake — it's a missing row). Appending row 0 to the
input list before mean-pooling brings cosine to 1.0 within float32
epsilon. The C++ port in `src/lid_fasttext.cpp` defines a named
`kEosRowId = 0` constant for this; the converter and reference dumper
both must match.

This bites every fastText port. If you use `m.get_subwords(word)` to
build the input list and skip the trailing `</s>`, you end up with
~3% accuracy loss that looks like quantization noise but isn't.

### Hierarchical softmax — code-bit sign convention `(2c-1)·f`

LID-176 uses HS (`loss=hs`), not flat softmax. The "output matrix"'s
176 rows are NOT per-label scoring vectors; they parameterize internal
nodes of a Huffman tree built deterministically from training-frequency
label counts. Per-label log-probability:

```
log P(label_i) = Σ over (node, code) in path[i]:
    log_sigmoid((2*code - 1) · (output[node] · hidden))
```

The right child gets `binary=1` in `Dictionary::initNgrams`, so walking
right at training time means code=1 and contributes `log_sigmoid(+f)`;
walking left contributes `log_sigmoid(-f)`. The unified formula is
`(2c-1)·f`, **not** `(1-2c)·f`. Sign-flipping makes predictions land
in the wrong subtree of the root — fastText says `fr/0.95` but your
port says `en/0.88` for "Bonjour le monde". The cosine-vs-embedding-
bag stage stays 1.0 (loss-agnostic), so the bug only surfaces at
top-1 — verify by hand on a single short input first.

The Huffman tree is **not stored in `model.bin`** — fastText rebuilds
it at load time from the dictionary's per-label `count` field via
`Model::buildTree`. fastText-python doesn't expose label counts, so
the GGUF converter parses `lid.176.bin` directly (header layout in
`src/dictionary.cc::save`) to extract them, then ports `buildTree`
inline. GGUF schema additions:

```
lid_fasttext.loss              str    "softmax" | "hs"
lid_fasttext.hs_path_offsets   i32[n_labels+1]   # CSR-style
lid_fasttext.hs_paths          i32[total_steps]
lid_fasttext.hs_codes          i8 [total_steps]
```

Memoize internal-node dot products inside the per-label loop —
deeper labels share most of their path with siblings. For LID-176
the average path length is 10.54 over 176 labels.

### `crispasr-quantize`'s `is_weight` gate — tensor naming matters

`examples/crispasr-quantize/main.cpp` only quantizes tensors whose
names contain `"weight"` or end in `_w`:

```cpp
bool is_weight = (sname.find("weight") != npos) ||
                 (sname.size() >= 2 && sname.substr(sname.size()-2) == "_w");
```

Tensors that don't match the gate get `f16, copying... done` —
silently passed through as F16 instead of quantized. The output GGUF
has the same byte count as the F16 input minus a few metadata bytes,
which is the most confusing diagnostic surface possible.

**Symptom**: `crispasr-quantize input.gguf output-q4_k.gguf q4_k`
"succeeds" with output the same size as input. Look at the tool's
log lines: `quantizing to q4_K... done` is good; `copying... done`
means the tensor failed the gate.

**Fix**: name the tensor `<prefix>.weight` (kokoro/parakeet/voxtral
convention). The lid-fasttext converter writes
`lid_fasttext.embedding.weight` and `lid_fasttext.output.weight`;
the runtime loader keeps a backward-compat fallback to bare
`lid_fasttext.embedding` / `lid_fasttext.output` since the first
release used those names before this trap was identified.

### Q4_K below 0.999 cosine floor on intermediate stages, top-1 stable

The diff harness tracks per-stage cosine against a Python F32
reference. For GlotLID quants:

| Quant | embedding_bag cos | logits cos | softmax cos | top-1 |
|-------|-------------------|------------|-------------|-------|
| F16   | 1.000000          | 1.000000   | 1.000000    | exact |
| Q8_0  | 0.999990          | 0.999991   | 1.000000    | exact |
| Q6_K  | 0.999898          | 0.999883   | 1.000000    | exact |
| Q5_K  | 0.999440          | 0.999530   | 1.000000    | exact |
| Q4_K  | 0.998303          | 0.998356   | 1.000000    | exact |
| Q4_0  | 0.997176          | 0.997357   | 1.000000    | exact |

Q4_K and Q4_0 fail the standard 0.999 cosine floor on `embedding_bag`
and `logits`. But softmax compresses the logit noise by ~3-4 orders
of magnitude (max_abs goes from 1.6 on logits to 6.1e-5 on softmax),
so the top-1 prediction is identical to F16 across an 8-language
multilingual smoke test. **Conclusion**: for shallow LID classifiers,
the conventional 0.999 cosine threshold is conservative — softmax
absorbs much more quant noise than it does for deep transformer
stacks. Q4_K is functionally fine; ship it but document the
intermediate-stage drift.

(Contrast with kokoro: Q4_K breaks the German backbone via the
predictor LSTM accumulating over multiple steps. Shallow networks
tolerate quant noise better.)

### `gguf` Python's K-quant gap — fallback to `crispasr-quantize`

`gguf.quantize()` only handles F32, F16, BF16, Q8_0. K-quants
(`Q5_K`, `Q4_K`, `Q6_K`, etc.) raise `NotImplementedError` from
`gguf/quants.py:129`. The right path for K-quants is to write F16
from the converter, then re-quantize via `crispasr-quantize` (which
calls into `ggml_quantize_chunk`). The C++ runtime side is
type-agnostic via `ggml_get_type_traits(type)->to_float`, so any
quant produced by `crispasr-quantize` "just works" without runtime
changes.

### LID-176's `dim=16` makes K-quants unproductive

K-quants need 256-element row alignment. LID-176 has `dim=16`, so
`crispasr-quantize` falls back to legacy Q4_0/Q5_0/Q8_0 for those
rows. Combined with the model already being 63 MB at F16 (the input
matrix is `2,040,010 × 16`), quants don't save meaningful space.
Ship F16 only for LID-176; quants make sense for GlotLID
(`1,634,361 × 256` = 1.6 GB at F32).

### `crispasr-quantize` is the right tool, not `whisper-quantize`

`build/bin/whisper-quantize` is the legacy whisper-binary-format
quantizer. It rejects GGUF input with `bad magic`. The GGUF-aware
tool is `crispasr-quantize` at `examples/crispasr-quantize/main.cpp`,
not previously discoverable from a casual `find -name "*quantize*"`.
Always check `build-ninja-compile/bin/crispasr-quantize` first; the
help text shows the supported quant types.

### HF upload — `hf upload-large-folder` with symlink staging

Per `.claude/CLAUDE.md`'s recipe: stage GGUFs as symlinks under
`/tmp/hf-staging-<repo>/` and run `hf upload-large-folder
cstr/<repo>-GGUF .` — uploaded 1.8 GB across 4 GlotLID quants in
9:42 wall time on the first attempt, including Xet pre-upload + commit.
Smaller LID-176 (63 MB) finished in under a minute.

The `--include` flag accepts multiple patterns (`"*.gguf" "*.md"`).
The CC-BY-SA-3.0 SA notice for LID-176 must be surfaced in the
README's metadata block AND in a license-warning section above the
Files table — downstream redistributors of the GGUF inherit the
SA terms, which is more restrictive than most ASR/LID models on
the Hub.

### CLD3 (`google/cld3`) is a separate architecture, not n=1,2,3 bags

The brief described CLD3 as "separate hashed embedding bags for
n ∈ {1,2,3}, concatenated → FC + ReLU → softmax". The actual model
in `src/cld_3/lang_id_nn_params.cc` (1.76 MB embedded weight file)
declares **six** embedding columns, not three:

```cpp
const int LangIdNNParams::kEmbeddingsNumRows[] = {1000, 5000, 12, 103, 5000, 100};
const int LangIdNNParams::kEmbeddingsNumCols[] = {16, 16, 8, 8, 16, 16};
const int32 LangIdNNParams::kConcatOffsetValues[] = {0, 16, 32, 40, 48, 64};
```

Six features → concat to 80-d → hidden + softmax over ~107 langs.
Weights stored as `float16` (Google's specific representation, as
`uint16` literals in C++ source). Each feature is a different
extractor (probably {char-1grams, char-2grams, script-type,
relevant-script, char-3grams, punctuation} per CLD3's feature
function registry). Porting CLD3 is meaningfully more work than
porting fastText was — six feature extractors + their text-normalisation
quirks live in `src/feature_extractor.{h,cc}` and the full set isn't
trivially expressible without building (or vendoring) libcld3.

Pragmatic plan if CLD3 becomes priority later: parse
`lang_id_nn_params.cc` directly via regex (the floats are in
plain-text C++ array literals — skip the `float16` ones, decode the
`uint16` literals via `ggml_compute_fp16_to_fp32`); replicate the
six feature extractors from `src/feature_extractor.cc` in C++; port
the matmul + ReLU + softmax. Full session of work, not an
afternoon-add-on.

**Update (May 2026):** CLD3 shipped — see the next section,
"## Text LID via CLD3", for the actual port. The plan above held
up: regex-parse → six feature extractors → matmul + ReLU + softmax.
The traps were elsewhere (bf16-style float16, MurmurHash2 not
CityHash, ULScript values guessed wrong).

## Text LID via CLD3 — Google compact language detector (May 2026)

`src/lid_cld3.{h,cpp}` ports Google's CLD3
([github.com/google/cld3](https://github.com/google/cld3),
Apache-2.0) — a tiny shallow classifier (~1.5 MB F32 / ~440 KB F16)
that emits 109 ISO 639-1 labels. Six feature extractors (4 cbog
char-ngram bags at sizes 1/2/3/4, RelevantScript, Script) →
80-d concat → FC + ReLU → 208-d hidden → FC → softmax. Released as
`cstr/cld3-GGUF` on the Hub. Sibling backend to lid-fasttext; the
post-merge auto-routing dispatcher picks between them via the GGUF's
`general.architecture` key.

Forward path is pure manual F32 (no ggml graph) — the compute is
well under 1 MFLOP per call (one 80×208 matmul + one 208×109 matmul +
softmax). F16 weights are dequantized to F32 once at load time via
`ggml_fp16_to_fp32`; the 1.5 MB RAM hit is trivial.

### CLD3's `float16` is bfloat16-style, NOT IEEE binary16

CLD3 ships its weights in `src/lang_id_nn_params.cc` as 1.76 MB of
plain-text C++ array literals — no binary side-car. The `float16`
typedef in `src/float16.h:43-49` is **the upper 16 bits of a binary32
float** (1 sign + 8 exponent + 7 mantissa, i.e. bfloat16-style),
NOT IEEE 754 binary16 (1+5+10). The header even calls this out
explicitly: *"NOTE: The IEEE floating point standard defines a
float16 format that is different than this format..."*.

**The trap**: decoding the `15392u`-style literals via numpy `<f2`
silently produces garbage — the bit patterns are a different format.
The correct decode is `(uint32(value) << 16).view(float32)`. This is
the dominant cause of "weights load but produce nonsense" — it looks
correct, just shifted into wrong magnitude.

This is also why `embedding_network.cc:122-123`'s dequant formula is
`(static_cast<int>(uint8) - 128) * multiplier` (where `multiplier`
includes the bf16-style scale): symmetric quantization with bias 128.
Got the bf16 decode wrong → the dequantized embedding rows look like
random numbers, the cbog feature contributions cancel out, and the
softmax is uniform.

### Hash function is MurmurHash2-32 with seed 0xBEEF, NOT CityHash

The CLD3 brief tentatively guessed CityHash because the upstream code
includes `absl_city`. Wrong: `utils.cc:137-183`'s `Hash32` is textbook
MurmurHash2-32 with `m=0x5BD1E995, r=24`, default seed `0xBEEF` (=
48879). Trivial 30-line port to numpy and C++.

The cbog feature IDs use the raw UTF-8 bytes of the ngram string as
the hash input (no UTF-8 normalisation), so byte-for-byte hash parity
with upstream is mandatory. Off-by-one or signed-vs-unsigned errors
in the hash get amplified through the embedding lookup and produce
top-1 mismatches across every multilingual input.

### Hiragana, Katakana, Hangul are NOT separate ULScript values

The brief estimated ~107 languages and called out 6 distinct feature
extractors. The actual model has **109 labels** (in
`task_context_params.cc:43-57`, NOT `lang_id_nn_params.cc`) and
**3 feature classes instantiated 6 times** — `ContinuousBagOfNgramsFunction`
×4 with different `id_dim`/`size`, `RelevantScriptFeature` ×1,
`ScriptFeature` ×1. The cbog instantiation at index 5 is the
1-character "unigram" with `id_dim=100`.

The bigger trap was in the **103-row text-script embedding**. The
ULScript enum at `script_span/generated_ulscript.h` has 102 values
running 0..101, with `NUM_ULSCRIPTS = 102` as a sentinel. **It does
NOT include Hiragana, Katakana, or Hangul as separate values** —
they all return `ULScript_Hani = 24` from the upstream
`ScriptScanner`, then a *secondary* Hangul-vs-Hani codepoint count
in `ScriptFeature::Compute` (language_identifier_features.cc:128-161)
returns the `NUM_ULSCRIPTS=102` sentinel only when Hangul-script
codepoints outnumber non-Hangul ones in the same span.

Early Python-port smoke-set failures traced directly to guessed
values:

  | Input                | Symptom                  | Cause                           |
  |----------------------|--------------------------|---------------------------------|
  | `नमस्ते दुनिया` (Hindi)| top1 = `bn` (Bengali)    | We mapped Devanagari→10; 10=Bengali. Correct: 9. |
  | `你好世界`            | top1 = `mr` (Marathi)    | We mapped Hani→43. Correct: 24. |
  | `こんにちは世界`        | top1 = `bg` (Bulgarian)  | We mapped Hiragana→41 (doesn't exist). Correct: Hani=24, Hangul-vs-Hani fixup leaves it as Hani. |

Read `script_span/generated_ulscript.h` first; do NOT guess these
values. The 103rd row (sentinel) is the special `NUM_ULSCRIPTS` slot.

### Full-Unicode lowercase in cleanup is non-optional

Upstream's `ScriptScanner::GetOneScriptSpanLower` lowercases ALL
letters across all scripts (Cyrillic П→п, Greek Α→α, Latin H→h)
before feeding the text to the feature extractors. ASCII-only
lowercase changes the bytes that get hashed by MurmurHash2 → different
ngram IDs → softmax lands on the wrong sibling label. Specifically,
`Привет мир` lowercased only as ASCII keeps the uppercase Cyrillic
codepoints, hashes their UTF-8 bytes, and predicts `tg` (Tajik)
instead of `ru` (Russian). Both are Cyrillic — same script feature —
but the cbog ngrams diverge.

The C++ port covers this with a hand-rolled case-fold table in
`src/lid_cld3.cpp::lower_codepoint` covering Latin / Latin-1 / Latin
Extended-A / Greek / Cyrillic / Armenian. Codepoints not in the
table fall through unchanged. ICU would handle the long tail
correctly but adding ICU as a dependency for one tiny LID model is
a poor tradeoff.

### Simplified text cleanup vs vendoring `script_span/`

Upstream's full preprocessing pipeline runs the input through
`ScriptScanner::GetOneScriptSpanLower` → `CheapSqueezeInplace` →
`SelectTextGivenBeginAndSize` (snippet selection if too long).
That's a ~250 KB Unicode state machine in `script_span/` (4 generated
UTF-8 transition tables at 40-82 KB each, plus orchestration).

We chose to ship a **simplified cleanup** (full-Unicode lowercase via
hand-rolled case-fold + ASCII punct/digit strip + whitespace collapse)
in `cleanup_text`. On the 8-input multilingual smoke set (clean
single-script inputs), this matches upstream byte-for-byte for 7/8.
The 8th, "Hello world", is a low-confidence (<0.5) underdetermined
input where small algorithmic differences flip the argmax (we say
`fi`, pycld3 says `ky` — both clearly wrong). The Python reference
dumper downgrades that case to a warning when both predictions are
below the 0.7 reliability threshold and proceeds with the dump, so
the C++/Python cosine gate still measures the algorithmic agreement
of *our* port, which lands at cos=1.000000 across every stage on
F16.

If divergence ever shows up on a confident input (>0.7 prob both
sides, different argmax), escalate to vendoring the full
`script_span/` tree from `/Volumes/backups/ai/upstream/cld3/src/`.
It's Apache-2.0 so distribution is fine; it's just a lot of generated
code mass to carry in-tree.

### `[in_dim, out_dim]` storage → transpose at conversion time

Upstream stores hidden + softmax weights as `[in_dim, out_dim]`
row-major (because `SparseReluProductPlusBias` iterates `x[i] * weights[i]`,
where `weights[i]` is row `i` mapping input dimension `i` to all
outputs). GGUF's `ggml_mul_mat(W, x) = y` convention is `[out_dim,
in_dim]`. The converter in `models/convert-cld3-to-gguf.py` transposes
once at write time so the runtime can do `W @ x` with no orientation
check at every load. Forgetting this transpose produces a softmax
of all NaN — the FC outputs become `concat[80] @ W[80,208] = wrong-dim`
and the bias add silently strides off the buffer.

### CLD3's "Hello world" is `ky` — known short-input quirk

CLD3 trained on web-scale data and gives short ambiguous inputs
their statistically-most-frequent-language guess. "Hello world" is
short enough that it lands on `ky` (Kyrgyz) consistently across
every variant (`'hello world'`, `'Hello world!'`, `'  Hello world  '`,
etc.) at p=0.7192. This isn't a bug in our port — it's the
contract. The diff harness's top-1-match check therefore needs the
reference dumper's `top1_label` field as the source of truth; you
can't paper over short-input quirks by feeding longer inputs at
diff time.

---

## IndexTTS-1.5 TTS backend

### HuggingFace GPT-2 has TWO final LayerNorms

IndexTTS uses HuggingFace's GPT2Model, which applies a built-in
`gpt.ln_f` LayerNorm after all transformer blocks. IndexTTS then
applies its own `final_norm` on top. Missing `gpt.ln_f` caused
a 2.0 logit gap, demoting the correct first mel token from rank 0
to rank 11. Both norms must be loaded into the GGUF and applied:
transformer blocks → gpt.ln_f → final_norm → mel_head.

### HF generate skips mel position 1

During HuggingFace's `generate()` loop, the mel position embedding
for generated tokens is computed as `attention_mask.shape[1] - mel_len`.
Because `fake_inputs` has `mel_len + 1` tokens (with start_mel at
the end), the first generated token after prefill gets mel_pos[2]
instead of mel_pos[1]. Position 1 is never used during inference.
The sequence is: start_mel=pos[0], gen_tok_0=pos[2], gen_tok_1=pos[3], etc.
This is a train/inference mismatch in IndexTTS but the model works
with it, so the C++ must match: `mel_pos = beam.tokens.size() + 1`.

### Latent extraction: mel_logits[:, :-2] gives (n_mel + 1) positions

Python's `forward(return_latent=True)` internally prepends start_mel
and appends stop_mel, then strips the last 2 positions. The result
has (n_mel + 1) positions: [start_mel_hidden, c1_hidden, ..., ck_hidden].
The C++ latent pass must extract this exact count, not n_mel.

### Reference dump tool was missing gpt.ln_f

The simplified reference dump in `tools/reference_backends/indextts.py`
manually reimplements GPT-2 but originally skipped `gpt.ln_f`,
producing wrong reference values. Any manual reimplementation of
HuggingFace GPT-2 must include `gpt.ln_f`.

### Conformer RelPositionalEncoding does NOT add pos_emb to x

WeNet/ESPnet Conformers have two positional encoding classes:
- `PositionalEncoding.forward`: returns `(x * xscale + pos_emb, pos_emb)`
- `RelPositionalEncoding.forward`: returns `(x * xscale, pos_emb)`

IndexTTS uses `RelPositionalEncoding`. The `pos_emb` is passed
separately to the attention layer as the R matrix — it is NEVER
added to the input x. Adding it corrupts every subsequent block.

### Conformer attention: absolute pos table, no rel_shift

IndexTTS's Conformer uses `RelPositionMultiHeadedAttention` but with
a critical deviation from the paper: `rel_shift` is commented out
("useless in speech recognition"). The position table is the stored
`pe[:, 0:T]` — T absolute positions, NOT a 2T-1 relative table.
The `matrix_bd = Q_v @ R^T` is already a T×T square matrix and
needs no shift. Using a 2T-1 sinusoidal table + rel_shift corrupts
attention scores in all 6 blocks.

### "Match Python's typical norm" speaker-embedding clamp was masking the bug, not fixing it (Issue #75, May 2026)

The original code rescaled the ECAPA speaker embedding to L2 norm = 0.9
with a comment claiming it "matches Python's typical magnitude" and
"prevents the vocoder from being overwhelmed". User #75 reported
"abnormal-sounding voice" — ASR roundtrip of an English test sentence
turned "indextts speech synthesis system" into "index piece since its
system" (~25% CER, speech recognizable but timbre/articulation distorted).

After A/B-testing with `INDEXTTS_SPK_NORM=raw` (pass-through, matches
upstream) vs `INDEXTTS_SPK_NORM=0.9` (legacy clamp):

| Mode | English ASR | Audio peak |
| --- | --- | --- |
| clamped to 0.9 | "test of the index piece since its system" | 0.06 |
| raw (upstream) | "test of speech synthesis" ✓ | 0.21 |

Upstream BigVGAN.forward() consumes the unnormalized ECAPA output
directly — there is no L2 norm anywhere in the pipeline. The clamp
shrank `cond_layer(spk_emb)` and `conds[i](spk_emb)` projections by
2.84×, leaving the BigVGAN underconditioned and the audio attenuated.

Lesson: a magic constant ("0.9") with a comment about "matching the
typical value" is almost never a fix — it's a workaround for a bug
you haven't found yet. If our ECAPA produced 3× the Python norm,
the right answer was to find the BatchNorm/scale bug in ECAPA,
not to clamp the output. Removing the clamp made English ASR
roundtrip clean. (Chinese ASR is still degraded — that's a separate
bug, likely in tokenization or conditioning encoder.)

Fix path: env knob `INDEXTTS_SPK_NORM=<float|"raw">`, default raw.

### Beam search KV snapshots round-trip through host (B=1 is 2.2× faster)

The IndexTTS GPT beam decode keeps a single on-device KV cache and
snapshots it per beam via `ggml_backend_tensor_get/set` (GPU → host →
GPU). With 24 layers × 1280 dim × 676 max tokens × 2 (K and V), the
KV cache is ≈ 158 MiB. Per step per beam we move 2 × 158 MiB; for
the issue-75 test (158 mel codes × B=3) that's ~146 GB of host
round-trips just for KV swapping.

Symptoms in the issue: 56.7s total, vocoder reports 2.5s, so 54s
is GPT decode — much of which is memcpy, not compute.

Workaround: `INDEXTTS_BEAM_SIZE=1` skips snapshot/restore entirely
(greedy decode keeps one KV slot resident on device). For an English
test sentence the wall time dropped 75.8s → 34.4s (2.2× faster) and
the ASR roundtrip stayed clean. Default is still B=3 to preserve
parity with the Python reference.

Proper fix would be to keep B KV slots resident on the backend and
swap them with `ggml_backend_tensor_copy` (device → device). Tracked
as follow-up; not in this commit.

### Follow-up: the "host round-trip" framing oversold the bottleneck on Apple Silicon (May 2026)

Implemented the device-resident path: B per-beam KV slots allocated on
the same backend as the active KV, swap via `ggml_backend_tensor_copy`,
refcount-based slot recycling on candidate selection (siblings sharing a
parent get a `tensor_copy` into a freed slot; only-children just inherit
the parent's slot pointer). Opt-in via `INDEXTTS_KV_DEVICE_COPY=1`,
default off.

M1 Metal, B=3, "Hello world. This is a test of speech synthesis."
(121 mel codes), 3 trials each, warm cache:

| Trial | host (`_get`/`_set`) | device (`_copy`) |
| --- | --- | --- |
| 1 | 61.25 s | 61.67 s |
| 2 | 55.12 s | 72.73 s |
| 3 | 70.32 s | 61.66 s |
| median | **61.25 s** | **61.67 s** |

Median delta < 1 %, within noise (each binary has ~15 s trial-to-trial
spread on this box, dominated by thermal / scheduler variance). Audio
output is byte-identical between the two modes and against the
pre-refactor binary across both English and Chinese prompts.

The earlier "75.8 → 34.4 s with `INDEXTTS_BEAM_SIZE=1`" speedup we
attributed mostly to "memcpy elimination" was in fact mostly **3× less
GPT compute** (B=1 runs one forward per step vs B=3's three) — the
snapshot traffic on Apple Silicon unified memory rides shared RAM and
costs about as much as a same-backend Metal blit. The original framing
oversold the memcpy cost.

Device mode is kept as opt-in for the case it's actually load-bearing
— discrete-GPU backends (CUDA, Vulkan) where `_get`/`_set` traverses
real PCIe. Needs measurement on those backends before promoting the
default; not flipping it on Metal alone.

Lesson: when a "2.2× speedup" comes from a binary on/off comparison
(B=3 vs B=1), the savings can be from any one of three sources —
fewer forward passes, less per-step CPU work, less memcpy. Don't
attribute the whole win to the one you happened to hypothesise about.

### Chinese requires `tokenize_by_CJK_char` before SentencePiece (Issue #75 follow-up, May 2026)

The C++ text path passed raw text straight to its SentencePiece
Viterbi. Upstream `TextTokenizer.encode()` runs THREE stages: (1)
`TextNormalizer.normalize` (char_rep_map, mostly CJK punctuation →
ASCII), (2) `tokenize_by_CJK_char` (regex `re.split` that surrounds
every CJK codepoint with whitespace), (3) `sp.Encode`. With the CJK
splitter, every Chinese character ends up as its own piece prefixed by
`▁` (id 10201), and SentencePiece chooses pieces that the model was
actually trained on. Without it, `而在获取电压函数中…` is glued into one
run and the Viterbi produces ~29 tokens of approximate matches —
intelligible neither to the model nor to a downstream ASR.

For the issue-75 prompt `而在获取电压函数中，我们总是先检查空指针再访问任何参数。`
the difference is:

| Mode | Tokens | Qwen3-ASR roundtrip | CER (punct-stripped) |
| --- | --- | --- | --- |
| C++ raw → sp.Encode | 29 | `而在曲电韩宿中飞我们总是减控指针在问任何数飞` | ~50% |
| upstream pipeline | 54 | `而在获取电压函数中，我们总是先检查空指针，再访问任何参数。` | **0.0%** |

**Trap: `。` (U+3002, full-width period) sits inside the CJK Unicode
range** (`U+2E80-A4CF`) that `tokenize_by_CJK_char` splits on. If the
punct table runs before the CJK splitter (upstream order), `。` is
already `.` when the splitter sees it, so the splitter ignores it and
SentencePiece emits one `▁.` piece (id 10203) — the sentence-end token
the model was trained to react to. If `。` is left as a CJK char, it
gets split as `▁` + `。` (10201 + 2), and the model hallucinates one
extra trailing syllable. Whisper-base hid this artefact (it can't
reliably transcribe Chinese in the first place); Qwen3-ASR-0.6B
revealed it as the user-audible "extra word at the end".

Implementation: `src/indextts.cpp:preprocess_indextts_text` is a
single-pass UTF-8 codepoint walker that does whitespace collapse →
punct-map (matching upstream's `char_rep_map` for the common
substitutions, including `。 → .`) → CJK-split → ASCII upper. Verified
the C++ token IDs are bit-identical to Python `sp.Encode` on the
preprocessed text.

What we deliberately skipped: the full wetext `zh_normalizer`
(numbers → hanzi, English contractions, pinyin tone digits). That
needs a real rule engine (or the wetext FST itself); not on the path
for plain Chinese roundtrip.

**ASR methodology corollary**: whisper-base over-counted the CER
~5× compared to Qwen3-ASR-0.6B on the same audio (21% vs 3.8% after
the first-pass fix). When validating CJK output, the ASR is the
load-bearing variable — pick one with real Chinese training data
(Qwen3-ASR, Cohere-transcribe, whisper-large) or the numbers will
chase artefacts that are in the recogniser, not the synthesiser.

### BigVGAN SnakeBeta: ggml memory layout is [c*T + t], not [t*C + c]

The anti-aliased SnakeBeta activation runs as a custom `ggml_map_custom1`
op. For a tensor with `ne[0]=T, ne[1]=C`, ggml stores element (t, c)
at offset `c * T + t` — channel-major, time innermost. Accessing as
`data[t * C + c]` (which is row-major [T, C]) scrambles channels
across time, producing noise instead of speech. This single bug was
the root cause of the vocoder producing unrecognizable audio.

### BigVGAN latent input: GPT output layout vs ggml tensor layout

The GPT latent extraction outputs `ne[0]=D, ne[1]=n_positions`, meaning
each position's D=1280 values are contiguous (row-major [pos, D]).
The vocoder's `ggml_conv_1d` needs `ne[0]=T, ne[1]=C` with time innermost.
Copying raw bytes from the GPT tensor into a `(T, C)` ggml tensor
transposes the data — channels become time and vice versa. Fix:
create the input tensor as `(D, T)` matching GPT layout, then
`ggml_transpose` + `ggml_cont` before conv operations.

### Reference audio sample rate must be checked

The `compute_ref_mel` function assumed 16kHz input and always
resampled to 24kHz. If the reference WAV is already 24kHz (common),
this double-resamples and destroys the signal. The CLI must check
the WAV sample rate and only resample when needed.

### Center-padding: reflect vs zero (the mel spectrogram root cause)

The core_mel `center_pad` option originally used **zero-padding** but
torchaudio's `center=True` uses **reflect-padding**. This caused the
first 2 STFT frames to differ from Python, propagating through the
6-layer Conformer into conditioning (norm 288.88 vs 288.92), then
compounding through 24 GPT layers to flip a beam search token at step 14
(6283 instead of 6109).

The hypothesis that F16 weight precision caused the gap was WRONG — an
F32 GGUF produced identical conditioning values. The actual root cause
was padding mode. Fix: added `center_pad_reflect` to `core_mel::Params`.

Lesson: when mel values at t=0 and t=1 differ from Python but t≥2 match
exactly, always check center-padding mode — it's reflect vs zero.

### HF beam search applies repetition penalty AFTER log_softmax

HuggingFace's `_beam_search` computes log_softmax on raw logits FIRST,
then passes the log-probabilities through `logits_processor` (which
includes `RepetitionPenaltyLogitsProcessor`). Since log-probs are always
≤ 0, the penalty multiplies them (making them more negative).

The C++ originally applied rep penalty to raw logits BEFORE log_softmax,
which changes beam dynamics. With `repetition_penalty=10.0`, this causes
different beam paths to win. Fix: compute log_softmax first, then apply
penalty to log-probs, matching HF's exact order.

### BigVGAN v2 SnakeBeta needs anti-aliasing — "negligible" was wrong (May 2026)

The raw SnakeBeta activation `y = x + sin(α·x)² / β` is not band-limited:
`sin²` introduces harmonics at 2α·x, 4α·x, … For trained α values in
BigVGAN v2 (the `_post` layer in particular has large α), those harmonics
land well above Nyquist and fold back as broadband click/buzz. The
original BigVGAN v2 paper wraps the activation in 2× upsample → activate
→ 2× downsample (Kaiser-windowed sinc) specifically to suppress that
aliasing.

CrispASR's first cut omitted the AA "because `ggml_conv_1d_dw` has no
CUDA kernel, and the quality impact on TTS speech is negligible." The
quality claim was just wrong — on the JFK-cloned "quick brown fox …"
prompt the raw path produced ~2 000 sample-to-sample jumps over 30 % FS
and several over 100 % FS (physically impossible in a band-limited
24 kHz signal), audible as steady click/buzz across every quant. The AA
path measured 0–27 such jumps, max\|Δ\| ≤ 0.38 — clean speech.

How we caught it: `np.diff(wav)` is the cheapest aliasing detector we
have. Any `|Δsample|` exceeding `2·sin(π·f_max/fs)` for the band-limit
`f_max` is impossible without aliasing. For 24 kHz 16-bit speech with
voice content roughly below 12 kHz, `Δ > 0.5` is a hard ceiling; counts
in the thousands == broken.

What we shipped (`src/indextts_voc.cpp`):

1. **AA is the default.** `INDEXTTS_VOCODER_RAW=1` (or `_AA=0`) opts back
   into the aliased path for benchmarking. We kept the raw path because
   it's the only fully-GPU-graphable activation we have today.
2. **Pre-allocated thread-local scratch.** The original AA op allocated
   three `std::vector<float>` per channel per call — at 1536 ch × 24
   layers per generate, that was ~37 k mallocs of 100 KB+ on the hot
   path. Lifted to per-thread (`ith`-indexed) scratch sized lazily on
   first use; `GGML_N_TASKS_MAX` is the `-1` sentinel, not a count, so
   we cap at 64 threads explicitly (`AA_SCRATCH_MAX_THREADS`) and pass
   that to `ggml_map_custom1` as the task hint.
3. **Pre-scaled upsample filter.** Multiply the FIR taps by 2 once at
   init (cancels the zero-stuff gain), saving one mul in the inner loop.
4. **`memcpy`/`memset` for the edge-replication padding** — small, but
   the inner-loop trace was dominated by the C++ per-element copies the
   original used.

Result on M1, JFK voice prompt, ≈ 6.7 s of audio:

| Vocoder config          | Δ>0.3 | max\|Δ\| | voc-only |
| ----------------------- | ----- | -------- | -------- |
| RAW / CPU (aliased)     | 1671  | 0.89     | 6.36 s   |
| RAW / GPU (aliased)     | 2080  | 1.04     | 7.12 s   |
| **AA / CPU (default)**  | **2** | **0.31** | 6.65 s   |
| AA / GPU                | 26    | 0.38     | 8.52 s   |

`AA / CPU` is the new sweet spot — only ~5 % slower than the broken-but-
fast RAW / CPU. `AA / GPU` is slowest because the `ggml_map_custom1` op
forces a Metal → CPU → Metal sync at every AMP block; if the rest of the
vocoder graph is on Metal we trade GPU-friendly matmuls for cross-device
copies and net-lose. Best operational answer until someone ports the AA
sandwich to native ggml ops: tell IndexTTS users to pass `--no-gpu` if
the prompt is short, or accept the ~25 % vocoder slowdown for the
voice-cloning convenience of keeping the GPT on GPU.

Lesson: comments claiming "negligible quality impact" age badly. When a
paper introduces a deliberate signal-processing stage, it's almost
always there for a reason; if you remove it, prove the absence of harm
with `np.diff` or a spectrogram, not assertion.

### Mixed-backend custom ops in ggml are a perf trap (BigVGAN AA, May 2026)

`ggml_map_custom1` runs CPU-only. ggml-backend-sched faithfully routes the
op back to CPU even when the rest of the graph is on Metal — but it costs
a Metal → CPU → Metal sync per op site. For IndexTTS BigVGAN with ~20
SnakeBeta sites per generate, that overhead dominates: GPU + AA measured
≈ 25 % SLOWER than CPU + AA on M1 (8.5 s vs 6.65 s vocoder).

Step A fix (`src/indextts_voc.cpp:indextts_voc_init`, commit `cd21faea`'s
follow-up): when `use_aa = true`, override `use_gpu` and force the whole
vocoder onto CPU. The GPT runs on its own backend; only the AMP-block
chain pays the per-AMP-cost, and skipping the round-trip is the win.

Override knob is `INDEXTTS_VOC_FORCE_GPU=1` for the people who want to
reproduce the mixed-backend benchmark; default does the right thing.

Lesson: if you reach for `ggml_map_custom1`, the right next step is
*always* a real ggml op + Metal kernel — never assume the custom op is
"only a tiny fraction" of the graph; the surrounding GPU stalls dominate
once it's called from a hot loop.

### Polyphase "zero-stuff + conv_1d" doesn't trivially port a torch conv_transpose_1d (May 2026)

Attempted the native-ggml-ops AA path for IndexTTS (Step B in the
optimisation plan). The idea: replace `ggml_map_custom1` with a
ggml-graph that's identical math to the torch reference, expressed via
`ggml_conv_1d` for both the upsample and downsample stages.

PyTorch reference (`indextts/BigVGAN/alias_free_activation/torch/resample.py`):
```python
x = F.pad(x, (pad, pad), mode='replicate')
x = ratio * F.conv_transpose1d(x, filter.expand(C,-1,-1), stride=2, groups=C)
x = x[..., pad_left:-pad_right]
```

Two blockers:

1. **Output-length mismatch.** `conv_transpose_1d(stride=2, K=12)` produces
   `(T-1)·s + K` = `2T+10` for input T. The classical "zero-stuff + stride-1
   conv1d" trick produces `2T - K + 1` = `2T - 11`. Closing the K-1 gap
   requires *asymmetric* boundary padding (10 left, 11 right for K=12),
   which `ggml_conv_1d`'s symmetric `p0` parameter can't express. You can
   pre-pad the data with `ggml_concat`'d replicate columns and use `p0=0`,
   but that adds three more graph nodes per AA site and the constants are
   annoying.

2. **Downstream-add broadcast assertion.** Even with lengths corrected
   manually, runtime hit `GGML_ASSERT(ggml_can_repeat(b, a))` inside
   `ggml_add_inplace` from the BigVGAN per-block bias adds — the
   `ggml_reshape_2d` after the cropping `ggml_view_3d` doesn't always
   see a contiguous tensor, and the resulting shape drifts in ways
   `ggml_add`'s in-place fast path refuses to broadcast.

`ggml_conv_transpose_1d` has no `groups` parameter, so we can't express
the depthwise behavior directly with it either; the workaround is a
`[K, C, C]` block-diagonal kernel, which at C=1536 is 113 MB — no-go.

**Update (Step B-v2, same week):** both blockers fixed in
`src/indextts_voc.cpp:aa_snake_beta_native`. Now shippable as an opt-in
behind `INDEXTTS_AA_BACKEND=native`.

The fixes:

1. **Length match via `p0 = K - 1 = 11` on the upsample `ggml_conv_1d`.**
   The zero-stuffed signal of length `2·T_p - 1` becomes `2·T_p + 21`
   after symmetric padding by 11, and the conv1d output is `2·T_p + 10`
   — the same as torch `conv_transpose_1d(K=12, stride=2)`. Crop with
   `up_pad_left + up_pad_right + 1` to land at exactly `2·T`, the same
   number torch produces after its own crop.
2. **`ggml_cont` between the truncating `ggml_view_3d` and the following
   `ggml_reshape_2d`.** The view narrows ne[0] but keeps the parent's
   nb[1] stride, so the resulting tensor is non-contiguous; reshape
   would silently land on the wrong layout and the next graph add fired
   the `ggml_can_repeat` assertion. One extra `ggml_cont` per AA site
   makes the reshape valid.

Validation against the CPU custom-op (Step A) reference:

| Path                     | voc-only (ms) | clicks Δ>0.3 | max\|Δ\| | ASR roundtrip   |
| ------------------------ | ------------- | ------------ | -------- | --------------- |
| Step A custom op  (CPU)  | 7872          | 2            | 0.309    | ✓ exact         |
| Step B-v2 native  (CPU)  | 7574          | 2            | 0.309    | ✓ exact         |
| Step B-v2 native  (GPU)  | 8012          | 26           | 0.375    | ✓ exact         |

CPU output is bit-equivalent (same click pattern, same max\|Δ\|).
GPU output drifts a tiny amount (26 vs 2 jumps, but max\|Δ\| still
below 0.4 — well into the noise floor of speech transients) — the
difference is Metal's vs CPU's floating-point order-of-ops for the
broadcast `ggml_mul`s in SnakeBeta. ASR is identical across all three.

Why we kept the custom-op as default (not switched to native):

- Native-on-CPU is 4 % faster but introduces a second AA codepath. Not
  worth the maintenance vs proven custom op for the marginal gain.
- Native-on-GPU is *slower* than custom-op-on-CPU (8.0 s vs 7.9 s on
  M1) — the concat/reshape/scale graph overhead inside Metal eats
  whatever the kernel-level GPU speedup buys. A real fused MSL kernel
  (Step C-2) is still the path to a meaningful GPU win.
- ggml-backend-sched does the right thing — when `aa_use_native()`
  returns true, the auto-fall-to-CPU in Step A is skipped and the
  vocoder graph stays on Metal end to end.

Lesson: a "polyphase zero-stuff + conv_1d" *is* expressible as native
ggml ops if you accept three boilerplate concats per AA site to fix
the asymmetric-pad problem. It compiles and runs correctly on Metal.
But the per-call graph overhead means it's worth shipping only as an
opt-in proof of correctness; the real win is still the fused-kernel
custom op route — see `tools/upstream-prs/07-metal-aa-snake-beta.md`.

### Accelerate vDSP_desamp + vvsinf beats hand-rolled SnakeBeta loops on M1 (May 2026)

After the CPU AA op is correct, the bottleneck is the per-channel inner
loops: K-tap scatter for upsample, sin+sqr+fma for SnakeBeta, K-tap dot
+ stride-2 decimate for downsample. The scalar inner loops are clean
but the compiler's auto-vectorisation isn't always taking the FMA
opportunity — especially across the `+=` loop-carried dependency in the
upsample scatter.

Step C-1 swaps the two stages that have direct Accelerate analogues:

- SnakeBeta inner: `vDSP_vsmul → vvsinf → vDSP_vsq → vDSP_vsma` (one
  vector mul, one vector sin, one vector square, one fused-multiply-add).
- Downsample inner: `vDSP_desamp(decimation=2, filter)` fuses K-tap FIR
  + stride-2 decimation into one Accelerate call backed by NEON.

vDSP is `#ifdef __APPLE__` only; the scalar paths still compile and run
elsewhere. Set `INDEXTTS_AA_SCALAR=1` to force the scalar paths for A/B.

Measured speedup on M1 (q8_0 GPT, JFK voice prompt, ≈ 6.7 s of audio,
average of 3 warm-cache runs):

| Path                | voc-only |
| ------------------- | -------- |
| scalar (Step A)     | 6906 ms  |
| Accelerate (Step C-1)| 6746 ms  |

≈ 2-3 % on the full vocoder — small because AA SnakeBeta is one
component alongside the MRF stack and the per-stage convs that
dominate. The win is "free": opt-out flag exists, output is
numerically equivalent to scalar (rmsdiff 1.3 × 10⁻⁵, well below int16
quant noise), ASR roundtrip identical.

Lesson: vDSP gives modest wins on already-cache-friendly inner loops.
The big lever for IndexTTS GPU perf is still a Metal kernel — see
`tools/upstream-prs/07-metal-aa-snake-beta.md`.

## Speaker verification — TitaNet

### NeMo mel preprocessing is not what the config says

The NeMo model config YAML for TitaNet says `window: "hann"` but the
actual `FilterbankFeatures.forward()` applies three additional steps
before the STFT that are NOT listed in the model config:

1. **Pre-emphasis** (`x[t] = x[t] - 0.97 * x[t-1]`) — this is the
   default from NeMo's `AudioToMelSpectrogramPreprocessor`, NOT specified
   in the per-model config. Missing this causes a ~12 dB mel offset and
   cos drops from 0.999 to 0.917.
2. **Zero-padding** for center=True (`pad_mode="constant"`) — NOT
   reflect. The NeMo source explicitly passes `pad_mode="constant"` to
   `torch.stft`. Our initial reflect padding was wrong.
3. **Centered window placement** — when `win_length < n_fft`, PyTorch's
   `torch.stft` pads the window symmetrically:
   `left = (n_fft - win_length) // 2`. This is NOT documented in the
   PyTorch STFT docs but is in the source.

Lesson: never trust the model config alone. Trace the actual Python
forward with intermediate dumps. The "standard" STFT has at least 3
knobs (pre-emphasis, pad mode, window centering) that differ between
NeMo, torchaudio, and librosa.

### NeMo BatchNorm eps is 0.001, not 1e-5

All encoder BN layers in TitaNet-Large use `eps=0.001`. The PyTorch
default is `1e-5`. This single mismatch caused the encoder norms to
explode: block 1 norm was 451 (should be 17). The effect compounds
multiplicatively through 3 mega-blocks × 3 sub-blocks = 9 stages.

Lesson: always check `module.eps` for every BatchNorm in the model.
NeMo's `ConvASREncoder` passes `batchnorm_kwargs` from the config,
and the default for NeMo's `MaskedBatchNorm1d` is 0.001 — NOT the
PyTorch `nn.BatchNorm1d` default.

### JasperBlock has a hidden post-activation (mout)

NeMo's `JasperBlock` applies an activation (ReLU + Dropout) AFTER
the SE + residual addition, stored in `self.mout`. This is NOT visible
in the `mconv` module list. The mconv layers provide inter-sub-block
activation for repeat > 1, but the FINAL activation comes from mout.
Without this, the encoder output has negative values where it shouldn't.

Lesson: always print `block.mout` / `block.mout` for any NeMo encoder
block. The `mconv` list alone is incomplete.

## Parakeet-TDT greedy decode — blank + duration=0 (May 2026)

### `max(1, dur_skip)` on blank exits the inner loop "too early" vs. NeMo

`parakeet_tdt_decode()` originally treated `blank + dur=0` as a
force-advance: `t += std::max(1, dur_skip); break;`. NeMo's
`GreedyTDTInfer._greedy_decode` instead does
`time_idx += skip; need_loop = (skip == 0)` — i.e. blank+dur=0 stays
on the same encoder frame and re-runs the joint head, counting toward
the shared `max_symbols` budget (default 10). Only when the budget is
exhausted does `time_idx += 1` fire.

For pure greedy argmax the two are observationally equivalent: blank
predictions don't change predictor state, so retries are deterministic
and produce the same blank+dur=0 every time, eventually force-advancing
exactly one frame either way. The fix is still worth applying because:

1. **Reference parity**: the inner-loop budget now counts both
   non-blank+dur=0 emits AND blank+dur=0 spins, matching NeMo's
   `symbols_added`. The non-blank path was already correct
   (`n_inner++` + post-loop `if (n_inner >= max_per_step) t++`).
2. **Sampling correctness**: with `decode_temperature > 0`, the joint
   output IS stochastic and retries can resolve to a non-blank token.
   The old code skipped those retries entirely. Issue #88 reporter's
   "model emits blank+dur=0 as a 'not yet decided' signal" mental
   model only matches reality in the sampling path.
3. **Forward-compatibility**: any future change that mutates predictor
   state on blank (e.g. a non-NeMo experimental decoder) would silently
   change behavior with the old force-advance; the retry path
   preserves the contract.

### Don't double-increment `t` in the fix

The natural-looking fix is

```cpp
if (tok == blank_id) {
    if (dur_skip > 0) { t += dur_skip; break; }
    n_inner++;
    if (n_inner >= max_per_step) { t++; break; }   // ← bug
    continue;
}
```

This double-advances when blank+dur=0 exhausts the budget: the `t++`
inside the if-branch fires, then the existing post-loop
`if (n_inner >= max_per_step) t++` fires again, jumping the encoder
ahead by 2 frames instead of 1.

The clean version drops the in-handler force-advance and lets the
existing post-loop check do its job:

```cpp
if (tok == blank_id) {
    if (dur_skip > 0) { t += dur_skip; break; }
    n_inner++;
    continue;     // while-condition + post-loop handle exhaustion
}
```

Lesson: when adding a fast-exit inside a loop with an existing
post-loop "fix up" step, audit the post-loop for redundancy with the
fast-exit. The most defensive-looking version is the most likely to
double-count.


## VibeVoice 1.5B TTS voice cloning: acoustic + semantic dual encoder (May 2026)

### Root cause: TTS clone used acoustic-only, not acoustic+semantic

Microsoft's VibeVoice architecture uses two parallel σ-VAE encoders for
speech representation:

- **Acoustic encoder** (`at_enc`): vae_dim=64, captures spectral/timbre info
- **Semantic encoder** (`st_enc`): vae_dim=128, captures linguistic/prosodic info

Both go through separate connectors (`at_conn` → d_lm, `se_conn` → d_lm) and
are **element-wise summed** before being injected as voice conditioning tokens.

The TTS voice clone code only used the acoustic encoder (comment said "uses
ONLY the acoustic encoder"). The ASR path correctly used both. The assumption
was wrong — Microsoft's `modeling_vibevoice.py` forward() at line 372-375:

```python
x[acoustic_input_mask] = (
    speech_all_connect_features[speech_masks]       # acoustic
    + semantic_speech_all_connect_features[speech_masks]  # semantic
)
```

**Scaling asymmetry**: only the acoustic path gets `(x + bias_factor) *
scaling_factor`. The semantic encoder output goes directly to the connector
with no scaling. In the Python reference, scaling is inside
`forward_speech_features()` which only processes acoustic tokens. The semantic
path is a separate `self.model.semantic_connector(speech_semantic_tensors)`
call with raw encoder output.

**RMS ratio**: acoustic connector RMS ≈ 0.245, semantic connector RMS ≈ 0.034
(ratio ~14%). The semantic contribution is small but structurally important for
speaker identity transfer — without it the model produces generic-sounding
speech.

### WAV sample rate: parser must check, not assume 24 kHz

The σ-VAE encoders expect 24 kHz input (3200× total downsample ratio). The
RIFF WAV fmt chunk stores sample rate at byte offset +4, but the parser
(`vibevoice_wav_ref.h`) skipped that field. A 16 kHz reference WAV feeds
176000 samples → 55 frames instead of the correct 264000 → 83 frames.

Fix: read `sample_rate` from fmt chunk, add linear-interpolation resample
when ≠ 24 kHz. This is simpler than the Kaiser-windowed sinc resampler used
in chatterbox (voice-clone quality doesn't need it — the σ-VAE encoder is
robust to interpolation artifacts in the conditioning path).

### C ABI env-var pattern for WAV voice reference

VibeVoice's WAV clone path uses `VIBEVOICE_VOICE_AUDIO` env var because the
core engine reads it in `vibevoice_synthesize()`. The CLI adapter already
mapped `--voice <path.wav>` → env var, but the C ABI
(`crispasr_session_set_voice`) went straight to `vibevoice_load_voice()` which
only handles GGUF voice packs. The fix detects `.wav` suffix and calls
`setenv("VIBEVOICE_VOICE_AUDIO", ...)` instead. This is the same
pattern used by the CLI adapter — the C ABI just didn't replicate it.

Implication: any new backend that accepts WAV references via env vars needs
the same suffix detection in the C ABI, not just in the CLI adapter.

### Python reference verification approach

The `vibevoice` pip package has circular import issues with newer
`transformers` versions (`tokenization_qwen2_fast` moved, `AutoModel.register`
rejects duplicate config registrations). Workaround:

1. Monkey-patch `_LazyAutoMapping.register` to accept `exist_ok=True` before
   any vibevoice imports
2. Create a compatibility stub at
   `transformers/models/qwen2/tokenization_qwen2_fast.py` that re-exports from
   the top-level `transformers.Qwen2TokenizerFast`
3. Use `importlib.util.spec_from_file_location` to load specific submodules
   without triggering the package's `__init__.py` import chain

For the voice-clone reference, we only need the encoder + connector weights
(not the full LM), so loading individual components from safetensors is
both faster and avoids the 3 GB LM weight overhead.

### Static library relinking trap (cmake)

When a `.cpp` change only updates a static `.a` library, cmake's incremental
build may report `Built target X` without actually relinking the final
executable — the `.a` timestamp doesn't trigger the exe's link step. The
binary stays stale. Workaround: `rm build/bin/crispasr` then rebuild, or
touch the exe's own `.o` dependency.

---

## voxcpm2 perf — per-step ggml graphs, Metal, SIMD layouts (May 2026)

The PLAN #96 push took voxcpm2-tts "Hello world" zero-shot from **~49 s
to ~5–7 s** wall on M1 (≈8 commits across one session). Most of the
lessons below apply to any new TTS / ASR backend that mixes legacy
CPU helpers with ggml graphs.

### Per-call ggml cgraph beats per-matmul tiny graphs by 2–3× even on CPU

The original voxcpm2 LocDiT did ~30 `matmul_mv_ggml` calls per
invocation, each of which ran `ggml_init` + `ggml_new_graph` +
`ggml_backend_graph_compute` + `ggml_free`. With CFM running LocDiT
19× per AR step (CFG × 10 timesteps − the zero-star skip), that's
~570 graph builds per AR step *just for CFM*.

Replacing them with one cgraph per `locdit_forward` call — 12 layers
folded into a single graph with named `ggml_set_input` tensors —
gave 2.3× on CPU (CFM/step 2398 → 1035 ms on M1 OMP=8) before any
Metal involvement. The win is amortising graph-build overhead, not
the matmul work itself.

Pattern (mirror `qwen3_tts.cpp build_graph_talker_kv` or
`chatterbox.cpp build_graph_t3_kv`):

  - Add `backend`, `backend_cpu`, `compute_meta` (vector<uint8_t>),
    `galloc` (ggml_gallocr_t) to the context.
  - Pre-allocate `compute_meta` = `ggml_tensor_overhead() * N +
    ggml_graph_overhead_custom(N, false)` where `N` is the
    worst-case node count (4096 was generous for voxcpm2).
  - `build_<thing>_graph(ctx)` runs `ggml_init` with
    `no_alloc=true` over `compute_meta`, builds the topology with
    `ggml_set_input` / `ggml_set_output` named tensors, then
    `ggml_free(ctx0)`.
  - Per call: `ggml_gallocr_alloc_graph` + per-input
    `ggml_backend_tensor_set` + `ggml_backend_graph_compute` +
    `ggml_backend_tensor_get`.

### Cache the graph itself for another 2× on top of that

The above still rebuilds the cgraph topology on every call.
`ggml_gallocr_reserve` against a long-lived graph in a dedicated
per-cache arena (`std::vector<uint8_t>` + `ggml_context*`) lets all
subsequent calls skip the rebuild + the gallocr planner walk. For
voxcpm2 this took CFM/step from ~1035 ms (uncached) to ~410 ms
(cached) — meeting the plan's ~400 ms CPU target.

Works directly for any graph with constant topology (LocDiT,
LocEnc — bidirectional encoders). For graphs with `n_past`-varying
shape (TSLM step), use the qwen3 bucketed pattern below.

### Multi-bucket Lk + runtime `kv_indices` makes KV-cache step graphs cacheable

qwen3_tts's `talker_buckets` pattern: pin `Lk = bucket_size` in
`core_attn::kv_self_attn`'s `fixed_kv_len` parameter and pass
`positions` as the `kv_indices` tensor so the K/V scatter goes
through `ggml_set_rows` (runtime-indexed) instead of a baked
static offset. Topology becomes `n_past`-invariant. Layer count,
KV-cache shape, and bucket Lk are the only things that affect
the graph structure — and Lk is fixed per bucket.

For voxcpm2 TSLM we ship 5 buckets {128, 256, 512, 1024, 2048};
`tslm_pick_bucket(needed_lk)` picks the smallest fitting one,
each is built lazily on first hit, longer prefills fall through
to the dynamic per-call build. Short prompts pay the cheapest
(Lk=128) attention overhead — important because
`ggml_flash_attn_ext` computes `Q·K^T` over the full bucket Lk
regardless of the `-inf`-masked tail, so wider buckets are
genuinely more expensive even with the mask.

### Apple Silicon Metal "shared" buffers keep `tensor->data` CPU-readable

`ggml_backend_init_best()` on M1 returns a Metal backend whose
weight buffers default to `is_shared = true` (unified memory).
This means `tensor->data` is a CPU-dereferenceable pointer to the
same physical pages Metal sees — no copy, no SIGSEGV when legacy
CPU helpers (`matmul_mv_ggml`, `rms_norm_cpu`, `tensor_data_f32`,
…) dereference it directly.

Concretely: switching `vox_load_weights` to load onto
`c->backend` (Metal) on `params.use_gpu=true` *didn't break* any
of the remaining legacy CPU paths (TSLM/RALM prefill, LocEnc,
VAE encode/decode, FSQ, stop predictor) on Apple Silicon. The
graph paths automatically got Metal compute via
`ggml_backend_graph_compute(c->backend, gf)`.

This trick is M1-specific. On discrete GPUs (CUDA, ROCm) or
non-shared Metal modes (very rare on M-series), `tensor->data`
would be device-only and dereferencing from CPU would SIGSEGV —
you'd need either per-weight CPU mirrors (like qwen3_tts's
`CpuEmbdCache`) or a full graph-only path.

The follow-on: the same Metal weights also speed up the legacy
helpers because Q4_K dequant reads land in lower-latency unified
memory than the previous CPU allocations. voxcpm2's TSLM prefill
went 5 000 ms → 80 ms (~60×) just from this — the matmul code
itself was unchanged.

### `GGML_PREC_F32` on `ggml_flash_attn_ext` is refused by Metal's `supports_op`

The chatterbox T3 patch (`ggml-metal-device.m`) explicitly returns
`false` from `supports_op` for any `GGML_OP_FLASH_ATTN_EXT` tagged
`GGML_PREC_F32`. The intent is to route those ops to the CPU
backend via sched for bit-identical CPU/GPU output (chatterbox
needs that to debug #83).

But: if you call `ggml_backend_graph_compute(metal, gf)` directly
(no sched), the unsupported op aborts the entire compute. Symptom:
`ggml_metal_op_encode_impl: error: unsupported op 'FLASH_ATTN_EXT'`
followed by ggml_abort.

For voxcpm2 the fix was just to *not* set PREC_F32 on LocDiT's
bidirectional flash-attn — the per-stage cosine bar (`cfm_step0_result
≥ 0.93`) tolerates Metal's F16 simdgroup drift. Pre-existing
`core_attn::kv_self_attn` doesn't set PREC_F32 internally, so TSLM
step worked out of the box.

If you do need bit-identical CPU/GPU (e.g. for debug bisects),
switch from direct `ggml_backend_graph_compute` to
`ggml_backend_sched_graph_compute` against a `[Metal, CPU]` sched —
that's what qwen3_tts and chatterbox do, and it auto-routes
PREC_F32 ops to CPU.

### `ggml_flash_attn_ext` head_dim allowlist

Metal's flash-attn supports `head_dim ∈ {32, 40, 48, 64, 72, 80,
96, 112, 128, 192, 256, 320, 512, 576}`. Anything else falls back
to CPU via sched or aborts under direct compute. Picking head_dim
during a port: stick to the allowlist or expect to route attn to
CPU.

### OMP parallelism on conv loops needs gather-form rewrites

The original `causal_transposed_conv1d` did scatter-add into
shared output — race-prone, so the outer loop couldn't be OMP'd.
Rewriting in gather form (for each output position, walk valid
kernel taps via `k0 = (ot + trim) mod stride` step `stride`) made
the `(out_ch, ot)` outer pair write-disjoint and parallelisable.

Similar story for `snake1d` and per-block SR conditioning —
outer loop over channels is write-disjoint, OMP-parallel.

But OMP alone isn't enough when the inner loops have strided
memory access (see next).

### Strided ic loops kill auto-vectorisation; transpose to contiguous wins 4–8×

VAE conv1d inner loops read `x[ic*T_in + it]` (stride `T_in`) and
`weight[ic*out_ch*ksize + …]` (stride `out_ch * ksize`). Both
strided in the inner axis — the compiler can't emit NEON vector
loads. Per-call work was effectively scalar even with OMP.

Fix: transpose both into contiguous-`ic` layout before the inner
loop:

  - weight: `[in_ch, out_ch, ksize] → [ksize, out_ch, in_ch_inner]`
  - x: `[in_ch, T_in] → [T_in, in_ch_inner]`

The inner `ic` dot product becomes a contiguous load on both
operands — clang/gcc auto-vectorises into NEON `fmla` instructions.

The transposes are `O(in_ch × (out_ch × ksize + T_in))` per call,
small vs the unlocked dot work. For voxcpm2 block 0 transposed
conv: 2 957 ms → 615 ms (4.8×) on M1 OMP=8. Total VAE decode
8 772 → 3 875 ms (2.3×).

Gate the transpose path on `in_per_grp > 1 && ksize > 1` —
depthwise (`in_per_grp == 1`) and 1×1 convs (`ksize == 1`)
get no SIMD lift and the transpose is pure overhead.

### Build instrumentation BEFORE optimising the wrong thing

Per-block VAE decode timings under `VOXCPM2_BENCH=1` revealed
that **72 % of VAE time was in `causal_transposed_conv1d` upsamples**,
not the residual units I'd been about to spend an hour rewriting.
Before then I'd assumed snake1d was the hotspot — wrong.

The instrumentation cost: ~10 lines of `vox_now_ms()` deltas +
one `fprintf` per block, gated on the existing
`VOXCPM2_BENCH` env. Always pays for itself.

### `ggml_concat` along time axis lets you build prefix sequences cleanly

For both the LocDiT graph (mu_toks + time + cond + x = T=11) and
the LocEnc graph (CLS + patch_frames = T=5), the input sequence
is built by concatenating disjoint pieces with `ggml_concat(ctx,
a, b, dim=1)`. This is much cleaner than allocating a
pre-sized `[d, T]` tensor and writing slices into it — and on
Metal it's just a single `kernel_concat` dispatch.

### Don't over-engineer the first cut — ship working, then optimise

The session's biggest unit wins came from the simplest changes:

  1. LocDiT graph (1 day): 2.3× CPU CFM.
  2. Cache the graph (3 hours): 1.7× on top.
  3. Metal-load weights (1 hour): 60× on TSLM prefill.

Each commit was small, single-purpose, validated by the diff
harness before the next one started. If we'd tried to land
"VAE graph + LocDiT graph + Metal + caching + buckets" in one
go we'd still be debugging.

---

## voxcpm2 VAE — Python wrapper that captures kwargs without forwarding (May 2026)

This bit hard. The voxcpm `CausalTransposeConv1d` class in
`audio_vae_v2.py` defines:

```python
class CausalTransposeConv1d(nn.ConvTranspose1d):
    def __init__(self, *args, padding: int = 0, output_padding: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.__padding = padding
        self.__output_padding = output_padding

    def forward(self, x):
        return super().forward(x)[..., : -(self.__padding * 2 - self.__output_padding)]
```

The trap: `padding` and `output_padding` are **named keyword args**
in the subclass `__init__` signature, so Python pulls them out of
the call's kwargs BEFORE the `**kwargs` capture. They're stored
into `self.__padding` (name-mangled) and **never passed to
`super().__init__`**. So PyTorch's `nn.ConvTranspose1d`
internally uses `padding=0`, `output_padding=0` regardless of what
the caller passed.

That changes the output length of `super().forward(x)` from "PyTorch
ConvTranspose1d output with padding P" to "no-padding output,
`L_std = (T_in - 1)*S + K`". The wrapper then slices `[:-(2P - OP)]`
from the END, yielding `T_in * S` samples (verified for
`S ∈ {8, 6, 5, 2}` with `K = 2S, P = ceil(S/2), OP = S%2`).

**What this means for a C++ port.** Don't assume the subclass's
`__init__` signature lines up with the parent's — read what `*args,
**kwargs` actually passes through, then trust that. The legacy
voxcpm2 C++ `causal_transposed_conv1d` used `trim = K - 1` (a
head-shift), which gives the same OUTPUT LENGTH but a completely
wrong slice of the no-padding output. Cumulated across 6 upsample
blocks at increasing sample rates, the audio was shifted by **~46 ms
vs Python** — and `decoded_audio cos vs Python = 0.008` in the
diff harness for months.

The correct C++ formula: emit `T_in * S` samples, taking positions
`[0, T_in * S)` of the standard padding=0 transposed conv output.
For each output position `ot`:

```
y[ot] = sum_k w[k] * x[(ot - k) / S]   for k with (ot - k) % S == 0 and (ot - k)/S in [0, T_in)
```

= a tail-trim of `K - S` samples from the no-padding output. The
ggml graph version is even simpler — just take the first
`T_in * S` rows of `ggml_conv_transpose_1d(s, p=0, d=1)`.

Fixed in both legacy CPU `vae_decode` and the new `vae_decode_graph`.
After the fix: `vae_only cos vs Python = 0.989` (essentially correct),
`decoded_audio cos = 0.683` (remaining drift is now provably upstream
— CFM/TSLM precision cascade).

### Isolate the VAE before chasing audio drift

The diff harness's `decoded_audio` cosine compares end-to-end audio
through ALL of TSLM → RALM → LocDiT → CFM (10 Euler steps × N AR
steps) → VAE. A failing cosine there could come from any of those.

To isolate the VAE: hook `model.audio_vae.decode` in the Python
dumper, save the input latent as `generated_latent`, then add a C++
`vae_only` / `vae_only_graph` stage in `voxcpm2_extract_stage` that
takes the latent via `ref_samples` and runs **only** vae_decode.
Comparing the C++ output of that against Python's `decoded_audio`
directly attributes drift to the VAE without confounding it with
upstream precision drift. (Implementation: ~50 LOC across the
backend + dump + diff main.)

This is the same pattern used elsewhere (e.g. chatterbox `s3gen`
with reference T3 tokens) — feed Python's intermediate as the
boundary input, run only the segment under test. Cheap to add when
the upstream is non-deterministic or non-trivial; saves hours of
"is it the AR or the codec?" guessing.

---

## ggml broadcast hides size-mismatch bugs (May 2026)

When the SR-cond fix went in (`vae_decode_graph` first call), the new
sr_scale/sr_bias tensors were sized from `it->second->ne[1]` — which
returned **4** instead of **2048** because GGUF's `scale_embed` is
stored as ne=[2048, 4] in ggml (C innermost, bucket outer — the
*opposite* of what I'd assumed). So my init allocated 4-element
tensors and only wrote 4 scales (the first 4 channels at bucket 3),
leaving the "broadcast" to fend for itself.

**`ggml_mul(cur ne=[T, 2048], sr_scale ne=[1, 4])` does NOT assert**.
The op compiled and ran. The output was even close to correct — `cos
vs Python = 0.967` (vs 0.989 for the legacy CPU path). The first few
channels got the right scale; the rest got whatever ggml's binary-op
broadcast read from past the tensor end.

**Why this is dangerous.** A small cos drop reads like "Metal
floating-point drift" — so I spent multiple iterations trying to
swap CPU/Metal backends instead of looking at the broadcast input
shapes. The truth was much simpler: the input tensor was the wrong
size.

**How to detect.** When debugging a graph-vs-legacy drift:

1. Dump intermediates per-stage and find the first stage where they
   diverge (here: `block_0_sr` was identical until I added the SR
   cond step; the snake checkpoint came in already-wrong).
2. **Always print `ne[]` of every input tensor** to the failing op.
   I had `C=4` printed in the init trace from the very first run; I
   just didn't look at it carefully because "obviously ne[1] is the
   channel dim".
3. If ggml broadcast looks suspicious, replace it with an explicit
   `ggml_repeat` to a matching shape — that *does* assert on size
   mismatch.

**How to apply.** For multi-axis GGUF tensors with ambiguous shape
order, **take `max(ne[0], ne[1])` for "the non-bucket dim"** instead
of trusting either dim's position. The legacy CPU loop's pointer
arithmetic doesn't care about ne order (it just walks raw bytes),
which is why it kept working through the same ambiguity.

---

## Diff-harness "drift" is mostly the GGUF quant, not a code bug (May 2026)

The voxcpm2 VAE-graph session (commit `3be663cc`) left
`decoded_audio cos vs Python = 0.683` open and tracked as "upstream
CFM / TSLM drift". Several intermediate stages also looked weak:
`cfm_step0_result` 0.937, `tslm_layer_27_out` 0.968.

I spent half a day chasing this assuming F32-vs-bf16 op precision was
the culprit (Python runs voxcpm2 in `dtype=bfloat16`, our C++ in F32).
Hypothesis: rounding every tensor op output to bf16 (via the
`bf16_round_vec` helper in `core/torch_rng.h`) would align us with
Python's bf16-throughout dataflow. Added rounds to `tslm_layer_step`
(Q/K/V/O matmuls, RoPE, attention, FFN, both residual adds), to
`bidir_attn_full`, and to `locdit_forward` (time MLP, in/cond projs,
all 12 layers' attn + FFN).

**The bf16-round changes had near-zero effect** and slightly
**regressed** `cfm_step0_result` (0.937 → 0.925). They were also
costly (synth wall +~15 min from extra sequential op overhead).
Reverted entirely.

**Actual cause: the diff harness loads `voxcpm2-q4_k.gguf`** (4-bit
quantised weights) **but Python's reference uses the bfloat16
weights**. The "drift" was the Q4_K quantisation noise, multiplied
through the network. Re-running the diff with `voxcpm2-f16.gguf`:

| Stage              | Q4_K cos | F16 cos |
| ------------------ | -------: | ------: |
| tslm_prefill_out   |    0.986 | **0.998** |
| dit_single_fwd     |    0.994 | **0.99999** |
| cfm_step0_result   |    0.937 | **0.99992** |
| tslm_layer_27_out  |    0.968 | **0.999** |
| **decoded_audio**  | **0.683**| **0.929** |
| lm_to_dit_hidden   |    0.998 | **0.99996** |

Every intermediate stage hits cos ≥ 0.998 on F16. The remaining 0.07
gap in `decoded_audio` on F16 is most likely AR-stop-step jitter
(C++ and Python stop predictors fire at slightly different patches
when fed the same `mu`) plus residual F16-vs-bf16 mantissa drift in
the VAE. Not the deep transformer body.

**How to apply.**

1. **Before chasing "upstream precision drift" in a diff harness,
   re-run with the F16 (or BF16) GGUF.** If cos jumps to ≥ 0.998,
   the model is bit-correct — the previous drift was just the
   shipped quant level. Don't add bf16 rounds to F32 ops; they
   don't fix quant noise.

2. **The Q4_K → bf16 cos gap IS the quant cost** — it's the same
   number you'd see for any sufficiently deep transformer that's
   bf16 in Python and Q4_K in C++. The acceptance bar should be
   "audio sounds good + ASR roundtrips" (which Q4_K passes for
   voxcpm2), not "diff-harness cos ≥ 0.999 against bf16
   reference" (which Q4_K never will).

3. **For per-stage diff diagnostics, recommend dumping the
   reference against the matching weight precision:** run the
   Python dumper with `voxcpm2-f16` if you want to compare F16
   C++ vs F16-equivalent Python, or accept the Q4_K cosine
   penalty as quant noise.

Cross-ref: [[project_voxcpm2_vae_kwargs_bug]] (real bug fix earlier
in the same session, contrasts with this non-bug).

---

## SANM-encoder family (FunASR / SenseVoice / CosyVoice)

### The same `SenseVoiceEncoderSmall` ships in at least three FunAudioLLM products

Fun-ASR-Nano-2512, Fun-ASR-MLT-Nano-2512, and SenseVoiceSmall all use
the identical 70-block `SenseVoiceEncoderSmall` topology (1 entry
block @ 560→512 with `encoders0`, 49 main blocks via `encoders`,
20 tp blocks via `tp_encoders`, `after_norm` between, `tp_norm` at
end; 512-dim, 4 heads, kernel 11). The encoder weights ARE different
across the three releases (each is fine-tuned for its product) but
the C++ runtime is shared: factor everything that touches a SANM
block into `src/core/sanm.h` (`core_sanm::BlockWeights`,
`BlockParams`, `build_block`) and reuse it from each per-model `*.cpp`.

If we ever port CosyVoice2 / CosyVoice3 the same encoder is in there
too (it's the text-to-codec semantic-encoder bottleneck). One
helper, three+ consumers.

### config.yaml advertises CTC; Fun-ASR's `model.pt` ships none of it

Upstream `config.yaml` and `funasr/models/fun_asr_nano/model.py`
declare a `ctc_decoder` + a `CTC` head; the published checkpoint for
Fun-ASR-Nano-2512 and Fun-ASR-MLT-Nano-2512 ships **only**
`audio_encoder.* + audio_adaptor.* + llm.*` (1261 tensors total, zero
`ctc_decoder.*` / `ctc.ctc_lo.*` keys). If you trust the config you'll
spend a day chasing a CTC path that isn't there. Verify by enumerating
the state-dict prefixes before writing converter code.

The CTC weights live in **SenseVoiceSmall** instead — same encoder
shape, but the published `model.pt` carries `encoder.* + embed.*
+ ctc.*` and skips the LLM. That's the model to use when you want
fast non-AR multi-task ASR; Fun-ASR-Nano is the one to use when you
want LLM-quality text output (and can pay the AR cost).

### Adaptor `MultiHeadedAttention` default heads is 8, not 16 (Fun-ASR)

`funasr.models.llm_asr.adaptor.Transformer` builds the per-block
attention as `MultiHeadedAttention(kwargs.get("attention_heads", 8),
llm_dim, ...)` and `config.yaml`'s `audio_adaptor_conf` does NOT pass
`attention_heads`. So the adaptor's two transformer blocks run at
**8 heads (head_dim = 128)**, not the 16 you'd guess from
`llm_dim / 64`. The CrispASR converter wrote 16 into the GGUF
`funasr.ada_n_heads` KV by mistake; the runtime ignores that KV
and hard-codes 8. Mis-counting drops the adaptor's cosine similarity
to ~0.6 (vs cos=1.0 on the 70-layer encoder which is bit-near-exact)
and produces a one-character-off Chinese transcript (`开放` instead of
`开饭`). The diff harness catches this immediately, but only if you
run it.

### EncoderLayerSANM drops the attn residual when `in_size != size`

```python
if self.in_size == self.size:
    x = residual + dropout(self_attn(x, mask))     # standard
else:
    x = dropout(self_attn(x, mask))                # no residual
```

`encoders0[0]` (the entry block, `in_size=560, size=512`) is the
only one of the 70 blocks that hits this branch. The C++ encoder
block builder needs an `apply_attn_residual` flag (or two code
paths). Forgetting to special-case it produces a 560-dim residual
being added to a 512-dim output, which won't even compile in ggml —
but if you silently pad / project the residual to 512, you get a
model that "works" but has subtly wrong activations from layer 0
onward.

### SANM FSMN memory branch has its own pre-conv-V residual

```python
def forward_fsmn(self, inputs, mask, ...):
    x = inputs.transpose(1, 2)
    x = self.pad_fn(x)
    x = self.fsmn_block(x)               # depthwise Conv1d, no bias
    x = x.transpose(1, 2)
    x += inputs                          # ← residual to PRE-conv V
    return x
```

The full SANM block output is `att_out + fsmn_memory`. Two
non-obvious things: (1) FSMN convolves over **V only** (not Q or K,
and not the post-MHA result), (2) the FSMN branch carries its own
residual to V **before** joining the MHA path. Drawing the dataflow
with one arrow fewer or one arrow more both produce a network that
compiles fine but yields ~cos 0.5 vs the reference.

### `use_low_frame_rate=true` shrinks the prompt placeholder count (Fun-ASR)

config.yaml `audio_adaptor_conf.use_low_frame_rate: true` triggers
a 3× `(x-1)//2+1` reduction in `fake_token_len_i` inside
`FunASRNano.data_load_speech`, even though the adaptor itself has
`downsample_rate: 1` (no actual downsample in the forward graph).
The prompt builder reserves the reduced number of placeholder slots,
and only `adaptor_out[:fake_token_len_i]` gets spliced — the rest
of the adaptor frames are discarded. Looks like a vestige of an
earlier adaptor design; mirror it verbatim and the model still
produces correct transcripts.

### funasr 1.3.1 packaging bug — `from ctc import CTC` is absolute

`funasr/models/fun_asr_nano/model.py` line 20 imports CTC with an
unqualified name; on a clean install this fails with
`ModuleNotFoundError: No module named 'ctc'`. Workaround in
`tools/reference_backends/funasr.py`
(`_ensure_fun_asr_nano_importable`): prepend the `fun_asr_nano/`
directory to `sys.path` before importing. Don't try to upstream-fix
this on user machines.

### Flash-attn crossover sits at ~T_lfr=100-150 (~6 s audio)

`FUNASR_NO_FA=1` (and the same shape `SENSEVOICE_NO_FA=1`) reverts the
SANM + adaptor MHA from `ggml_flash_attn_ext` back to the unfused
`mul_mat + soft_max_ext + mul_mat` path. On Apple M1 Metal:

| Clip | T_lfr | FA ON | FA OFF | Total Δ |
| --- | ---: | ---: | ---: | --- |
| zh.mp3 (5.6 s) | 94 | 196 ms | 166 ms | FA OFF wins 6 % |
| samples/jfk.wav (11 s) | 183 | 258 ms | 370 ms | FA ON wins 7 % |

Below T≈100 the FA kernel-launch overhead eats the win; above T≈150
the fused-softmax+matmul kernel pulls clearly ahead. Default ON is
right for typical ASR workloads (most utterances ≥6 s); realtime
users splitting on tight VAD windows may want the opt-out.

### SenseVoice query-embed pattern — multi-task output via prepended tokens, not separate heads

SenseVoiceSmall's "4 outputs in one CTC pass" trick is **not** four
classifier heads — it's a 16-row embedding table (`embed.weight`,
`(16, 560)`) prepended to the LFR fbank features. The first 4
positions of the encoder output go through the same CTC head as the
rest of the sequence, but the CTC vocab reserves slot ranges for
language IDs (24884-24992), emotions (25001-25009), audio events,
and ITN flags (25016-25017). So the model "writes" the multi-task
predictions to the first 4 positions via the standard CTC argmax,
and the transcript follows from position 4 onward.

This means the C++ runtime needs zero per-task classifier code —
just one CTC head + the query-embed prepend. The runtime decides
which of the 4 query rows to prepend (language hint, ITN flag) at
inference time; the model uses them as the implicit "channel
selector" for the encoder output positions.

If the upstream model adds a new task (e.g. speaker ID) the
infrastructure on our side stays exactly the same — they extend
the query-embed table + the CTC vocab; we just read whichever new
special-token IDs get emitted from position 0..3.

### SenseVoice output-token order ≠ input query order — classify by content, not position

The four query embeds are prepended to the LFR features in input
order `[language, event_query, emotion_query, textnorm]` (see
`sensevoice_gather_query_rows` and the upstream Python ref
`tools/reference_backends/sensevoice.py`). The intuitive assumption
is that the encoder output's four prefix tokens come back in the
same order — they don't. On `samples/jfk.wav` the encoder emits
`<|en|><|ANGRY|><|Speech|><|withitn|>`: position 1 is the **emotion**
(`ANGRY`), position 2 is the **audio event** (`Speech`). The
"event_query" and "emo_query" names in upstream refer to the
embedding-table row IDs (1 and 2), not to which slot they decode
into; the trained model has those slots swapped relative to the
input naming.

This bit when wiring `sensevoice_transcribe_structured()`. Positional
parsing populated `audio_event` with `"ANGRY"` and `emotion` with
`"Speech"`. The fix is content-based classification: maintain three
small static sets (languages, emotions, itn flags) and classify each
`<|…|>` against them; whatever doesn't match those three is the
audio event (which is open-ended — Speech / Music / BGM / Laughter
/ Cough / Sneeze / Sigh / ... so a positive enum is brittle).

Bonus robustness: on degenerate audio the model can emit fewer than
four prefix markers. Positional parsing would mislabel everything
downstream; content-based classification just leaves the missing
field empty.

### SentencePiece detokenise without linking libsentencepiece

For CTC-decode-only consumers, we don't need full SentencePiece —
just a piece-ID → piece-string table. Extract the table once at
convert time via `sp.id_to_piece(i)` for i in range(vocab_size),
write it as `tokenizer.ggml.tokens`, and detokenise at runtime by
looking up each ID and replacing the SentencePiece leading-space
marker `▁` (U+2581, "\xE2\x96\x81") with an ASCII space. ~20 LOC
of C++, no link-time dep. Same pattern canary-ctc / firered-asr
already use.

### FunASR audio_adaptor: use_low_frame_rate config mismatch (PLAN #99)

**Symptom:** Fun-ASR-MLT-Nano-2512 hallucinated the ending of English
transcripts (~last 5 tokens wrong), while Fun-ASR-Nano-2512 worked
correctly through the identical C++ code path.

**Root cause:** The C++ runtime hardcoded `use_low_frame_rate = true`.
This flag controls how many adaptor output frames are spliced into
the Qwen3-0.6B LLM prompt. The Nano config.yaml sets it to `true`
(23 of 183 frames used on JFK); the MLT-Nano config.yaml omits it
(upstream default is `false` — all 183 frames). With the wrong value,
the C++ truncated 87% of the audio context.

**Why the bisect was misleading:** The per-stage diff showed genuine
attention drift (cos 0.998 at `ada_blk0_attn`), which is real F32
cross-implementation non-determinism amplified by peaked attention
patterns. But this drift is benign when the LLM receives the full
audio context — the model tolerates it and still decodes correctly.
The hallucination was caused by frame truncation, not attention
precision. The lesson: when one model variant works and another
doesn't through the same code, compare the config files first.

**Fix:** Converter reads `audio_adaptor_conf.use_low_frame_rate` from
`config.yaml` and writes it as a bool GGUF KV. Runtime reads the KV
at load time, falling back to `true` for pre-fix GGUFs.

### Qwen3-TTS FUSED_QKV bench — cold-cache trap + Q8_0 is neutral on Metal

**Trap:** A naive A/B test of `QWEN3_TTS_FUSED_QKV=1` appeared to show
a 1.83× speedup (144.7 → 79.0 ms/frame). The "improvement" was a
**cold-cache artifact**: the baseline run loaded the Q8_0 GGUF from
`/Volumes/backups/ai/` (external disk) into unified memory and macOS's
file-page cache. The immediately following FUSED_QKV run benefited
from those warm pages and appeared faster. Running both cold, or
properly interleaving the two conditions in sequence, gives:
~129 ms/frame for baseline, ~129 ms/frame for FUSED_QKV — neutral.

**Rule:** Always interleave conditions (A B A B A B) and discard the
first cold run of a new binary invocation when benching on macOS.
The first invocation loads the GGUF from disk; subsequent invocations
reuse page-cached pages. Alternating order eliminates this confound.

**Why FUSED_QKV is neutral for Q8_0 on Metal:** The optimization
byte-concatenates Q + K + V weight rows into a single fused tensor so
one `ggml_mul_mat` replaces three. For F16 weights this reduces Metal
kernel-launch overhead (three small dispatches → one larger one). For
Q8_0, Metal's Q8_0 dequant dispatch already launches a GPU threadgroup
that spans the full head_dim×head_dim tile; fusing to a 4096-output
tensor doesn't improve parallelism and may slightly hurt cache
utilization. The net effect on M1 Metal Q8_0 (28L, 1024 d, 8 KV
heads): no measurable difference.

**F16 case still open:** FUSED_QKV for F16 talkers is untested locally
(no F16 talker GGUF on this Mac). The expected benefit is real for
F16 (fewer kernel launches, larger tiles map better to the GPU
scheduler) — but requires a dedicated bench on a machine with the F16
GGUF and enough RAM to hold it warm.

**Bench setup used (2026-05-23, M1, build-ninja-compile):**
```
QWEN3_TTS_BENCH=1 [QWEN3_TTS_FUSED_QKV=1] \
  crispasr --backend qwen3-tts-customvoice \
  -m .../qwen3-tts-12hz-0.6b-customvoice-q8_0.gguf \
  --codec-model .../qwen3-tts-tokenizer-12hz.gguf \
  --tts "In the beginning..." --tts-output bench_tmp.wav
```
Key output line: `qwen3_tts: ar_loop  N ms (94 frames, X.X ms/frame)`.

---

## WDDM idle-clock-state hysteresis on consumer/laptop NVIDIA SKUs (May 2026)

Issue #81 investigation surfaced a Windows/WDDM behavior that
dominates **any** chunked-streaming inference perf measurement on
consumer/laptop NVIDIA GPUs (RTX A1000 Laptop confirmed; almost
certainly applies to RTX 30/40-series mobile and lower-end GeForce
desktop too). Knowing this saves days of code-bisecting noise.

### The behavior

A cold dGPU starts at low P-state (P5/P8, ~210-510 MHz core /
405-810 MHz mem). When compute work arrives, **WDDM does NOT
immediately boost** to P0/1140 MHz / 5501 MHz mem — it requires
~5-15 s of *sustained* GPU activity to engage P0, and it
aggressively defends idle clocks against bursty workloads with
sub-100 ms kernel groups (which is exactly the shape of chunked
ASR/TTS inference).

Same DLL, same hardware, same workload, two different states:

| State | mean / chunk | RTx | Notes |
|---|---:|---:|---|
| Cold (P5/P8) | 1400 ms | 2.7× | what you get if you bench from idle |
| Warm (P0)    |  185 ms | 21×  | what the model is actually capable of |

**8× wallclock difference** from nothing but driver-side power
state. The "real" performance is the warm number.

### Why this matters for ASR specifically

Parakeet / Canary / FastConformer-style ASR runs the encoder
graph in many short kernel groups (one per 4 s chunk = ~150 ms of
GPU work + ~50 ms host gap). WDDM sees each gap as "GPU is idle"
and starts demoting clocks. On a server SKU (Tesla / Quadro with
TCC mode, or any Linux box) this would never happen — TCC bypasses
WDDM, and Linux uses a more conservative DVFS. **The problem is
specific to WDDM on consumer SKUs.**

### What works to mitigate

1. **Pre-warm the dGPU** with ~10 s of sustained CUDA work before
   measuring. We ship `bench-issue81/gpu_keepalive.py` for this
   (a tight 256×256 fp32 ORT-CUDA add loop, ~50 MB VRAM,
   ~1 % util, but enough to keep WDDM in "GPU is busy" mode).
   For benches, simpler: run the workload-under-test ~200 times
   as a discard warmup, then measure.
2. **NVIDIA Control Panel → 3D Settings → Manage 3D Settings**
   - Global → "Power management mode" → "Prefer maximum
     performance" (no admin, persists across reboots)
   - Same setting under Program Settings tab for `python.exe`
     / `crispasr.exe` (per-app override tends to engage the
     dGPU more aggressively on Optimus laptops than the global
     setting alone)

### What does NOT work (counter-intuitive)

1. **`nvidiaProfileInspector.exe -setProfileSetting
   "_GLOBAL_DRIVER_PROFILE,PreferredPState,1"`** *USED* to help
   on older drivers (e.g., 581.95) but is **actively harmful on
   newer drivers** (we confirmed on Studio Driver 596.36): it
   caps the dGPU at P1 instead of allowing P0, so the GPU
   underclocks even when work demands max clocks. **Default
   driver behavior is better than this override on 596.x+.**
   To remove: `nvidiaProfileInspector.exe -deleteProfileSetting
   "_GLOBAL_DRIVER_PROFILE,PreferredPState"`. The setting was
   itself a workaround that newer drivers obsoleted; recheck
   on every driver bump.

2. **Driver upgrades don't fix it.** We went 581.95 → Studio
   596.36 mid-investigation; same WDDM behavior, same magnitude
   of cold-vs-warm gap. The heuristic is OS-side
   (DXGI/dxgkrnl), not driver-side. Game Ready and Studio
   variants behave identically.

3. **`nvidia-smi -lgc <freq>`** would force-pin the clock but
   needs admin per session AND is blocked on consumer SKUs with
   "user does not have permission" even when elevated.

4. **`gpu_keepalive.py` alone is not enough** if NPI is also
   set (the keepalive engages WDDM but NPI caps the resulting
   clock). Use keepalive WITHOUT NPI on 596.x+.

### Practical bench protocol (Windows + laptop NVIDIA)

```powershell
# 1. Set NV Control Panel "Prefer maximum performance" once (GUI, no admin).
# 2. Ensure no NPI PreferredPState override is active:
#    nvidiaProfileInspector.exe -deleteProfileSetting "_GLOBAL_DRIVER_PROFILE,PreferredPState"
# 3. Set CPU max state to 100% on AC (was 98% by default on some Win11 11):
#    powercfg -setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 100
# 4. Warm WDDM (10+ s of sustained CUDA work):
python bench-issue81/probe_postsiglu_leak.py <dll> 200
# 5. Bench immediately:
python tools/benchmark_asr_engines.py --engine crispasr --crispasr-lib <dll> ...
```

Single-shot cold benches on this class of hardware are **noise**.
Report best-of-N back-to-back with warm-up; document the warm-up
explicitly.

### How this was learned the hard way

Initial measurements on a fresh-rebooted machine showed `dll-postsiglu`
(current main) at 14.96 s mean vs `dll-post-cg` (May 11 baseline)
at 5.94 s — a *2.5× regression* that looked like a code bug. We
wrote a leak-probe (`bench-issue81/probe_postsiglu_leak.py`) that
called the same DLL 60-150 times in a tight loop, expecting to
see the per-call time degrade. Both DLLs were rock-stable in the
probe (157-256 ms per call, GPU memory constant).

The actual difference between the probe and the bench was that
the probe ran **after** another GPU-intensive bench had just
finished, so WDDM was already engaged. The bench ran from a
cold state. After running the probe FIRST, immediately followed
by the bench, both DLLs returned to ~2.7-2.9 s mean / 175-184 ms
p50 — matching the original May 11 3.063 s reference closely.

**The "2.5× regression" was a WDDM-state coincidence, not code.**
The actual `d758fe69` (fused norm_affine + siglu) impact on A1000
is +5.7 % wallclock improvement when measured cleanly.

### Sources / cross-refs

- PERFORMANCE.md "A1000 Ampere CUDA A/B" + "Phase 1 update
  (2026-05-23) — fused siglu/norm_affine A1000 verdict"
- `bench-issue81/probe_postsiglu_leak.py` (the probe itself; also
  a useful warmup driver)
- `bench-issue81/gpu_keepalive.py` (the May 11 keepalive — its
  docstring was already saying *"NPI's 'Prefer maximum
  performance' only biases up; it doesn't actually pin P0"*, and
  we now know that even when it does bias up, on modern drivers
  it biases the wrong direction)
- Microsoft DXGI WDDM Display Power Management docs (not linked
  here — search "WDDM idle detection" if you need primary sources)

---

## Chatterbox #83 Round 9 — S3Gen UNet GPU drift on Metal (2026-05-24)

This round is the production-fix landing for the cross-backend GPU
drift documented in Rounds 7 (Metal bisect) and 8 (CUDA P100 confirm).
The actual root cause hunt that previous rounds attempted (which
specific kernel loses precision) finally produced a clear answer:
**there's no single kernel bug to fix.** The drift is path-dependent
through `ggml-alloc`'s in-place buffer reuse decisions interacting
with Metal command-buffer scheduling, and 11 independent fix attempts
either don't help or trade one failure mode for another.

### Production fix: UNet weight residency split

`src/chatterbox_s3gen.cpp` is the only changed file. On GPU init, the
910 `s3.fd.*` (UNet1D ConditionalDecoder) weight tensors are loaded
on the CPU backend buffer via `core_gguf::load_weights_split`; the
other ~1375 tensors (`s3.fe.*` Conformer encoder, `s3.flow.*` front-end,
`s3.tok.*` Whisper-style tokenizer, `s3.se.*` CAMPPlus, `s3.v.*` HiFT
vocoder) stay on the GPU buffer. The ggml scheduler auto-routes ops
to the backend where weights live, so the entire UNet sub-graph runs
on the CPU backend in a single backend split — no per-op pinning, no
GPU↔CPU sync inside the UNet (which is what produced the Round 8 NaN
at T_mel ≥ 200). M1 Metal `s3gen_mel cos_min`: 0.940 → **0.999980**
in the diff harness; intelligible audio in smoke at all T.

Wall-time on M1 (3-iter median, "Ask not what your country can do for
you.", T_mel=200):

| Mode | s/run |
|---|---|
| cpu-only | 67.6 |
| default weight residency (UNet CPU, encoder/vocoder GPU) | 76.8 |
| gpu residency (broken, produces nonsense) | 48.0 |

The default residency mode is wall-time comparable to pure CPU on M1
because the encoder/vocoder are a small fraction of total time. The
gpu-residency mode would be ~30% faster *if* the drift were fixed —
that's the motivation for upstream PRs 09–11, not a practical win for
the M1 dev box. On dGPUs (CUDA, larger Metal Pro/Max) the encoder
fraction is larger and the residency split should buy more.

### Q8_0 × F32 bit-match Metal kernel (committed; upstream PR 09)

`ggml/src/ggml-metal/ggml-metal.metal` gains the Q8_0 counterpart of
the existing Q4_K × Q8_K `GGML_PREC_F32` bit-match path: pre-quantise
the F32 input to Q8_0 via `kernel_quantize_q8_0_f32` (mirrors
`quantize_row_q8_0` ARM NEON), then run a scalar integer-dot
`kernel_mul_mv_q8_0_q8_0` (mirrors `ggml_vec_dot_q8_0_q8_0_generic`).
Two precision-critical details:

- `id = 1.0f / d` uses the unrounded F32 `d`, not the F16 round-trip.
  We initially used `d_back = (float)(half)d` and got cos_min 0.940
  → 0.947 (basically no change). Switching to the unrounded F32 `d`
  matters but didn't move the needle on the UNet drift either,
  because…
- Quantize rounding uses `rint()` (round-to-nearest-even, IEEE
  default), which matches ARM NEON's `vcvtnq_s32_f32`. We tried
  `roundf` semantics (`floor(x + copysign(0.5, x))`) first — also no
  significant change.

The kernel is bit-identical to CPU on all 350 UNet Q8_0 mul_mats. It
just doesn't help the UNet drift because the drift isn't in mul_mat
output. Verified by combining the kernel with weight residency: the
two are independent.

### The exhaustive bisect — 11 attempts, none surgically fix it

The session pivoted from "find the buggy kernel" to "characterise
where the drift comes from" after the bit-match mul_mat kernel
failed to help.

1. **Bit-perfect mul_mat** (the Q8_0 kernel above) → no improvement.
   Drift isn't in mul_mat output.
2. **PIN_CPU_OP=mul_mat at T=102 (diff harness)** → cos=1.000.
   Inserting GPU→CPU→GPU sync at every mul_mat call restores parity.
3. **Same PIN at T=200 (smoke)** → NaN. Confirmed the Round 8 finding.
4. **PIN bisect across op types** at T=102: pinning ANY frequent op
   (norm, mul, add, flash_attn_ext, gelu, reshape, cont, concat,
   permute, mul_mat) restores cos=1.000. Pinning a sparse op (conv_1d,
   soft_max, mish, silu, scale) does not. The pattern is op
   *frequency*, not op identity — sync-barrier density matters.
5. **Per-segment dump bisect** — wrote a debug knob that dumps UNet
   intermediates from each sub-block. Discovered along the way that
   ggml allocator reuses buffers for same-shape outputs (mb_0_out and
   mb_2_out had identical md5 because the allocator placed them at
   the same address after compute finished). `ggml_set_output()` on
   each dumped tensor disables that reuse — required for the dump to
   yield distinct data.
6. **`set_output` on all 62 sub-block intermediates** in the diff
   harness call context → **cos_min = 1.000, max_abs = 0** (bit-perfect).
   `set_output` on 14 (block-level only) → 0.907 (worse than baseline).
   `set_output` on 1 (just db_resnet) → 0.879 (worse). Non-monotonic
   in the number of preservation points.
7. **Same 62-output set_output in the smoke call context** → NaN at
   every T tested (T_mel ∈ {38, 58, 66, 68, 244}).
8. **GGML_NO_INPLACE=1** (global allocator knob to skip the in-place
   reuse path) → cos = -0.97 (sign-flipped garbage). Some downstream
   code relies on in-place semantics in ways we couldn't trace.
9. **GGML_METAL_CONCURRENCY_DISABLE=1** (existing env knob that
   forces serial Metal command encoding) → no effect on drift.
10. **`kernel_norm` audit** — `kernel_norm_fuse_impl` uses
    `float sumf` accumulators end-to-end, F32 `simd_shuffle_xor`
    reduction, F32 shared memory. Not the source.
11. **`kernel_flash_attn_ext` audit** — line 6430 downcasts Q from
    F32 to half: `sq4[j*DK4 + i] = (q4_t) q4[i]` where `q4_t = half4`
    even in the `FA_TYPES_F32` family. Patched `FA_TYPES_F32`'s Q
    triple to `float, float4, simdgroup_float8x8` and rebuilt → cos
    went 0.940 → 0.860 (worse). The F16 Q downcast IS happening but
    "fixing" it changes the numerical ordering in a way that doesn't
    bit-match CPU either.

### The remaining unresolved question

Why does `set_output` on all 62 intermediates achieve
**bit-perfect** parity in the diff harness call context and **NaN**
in the smoke call context, when the chatterbox model, UNet graph
topology, CFM solver loop, and seed are all the same? Invariant
against: random seed, T_mel value, S3-tokenizer presence (.wav vs
precomputed .gguf voice), Metal concurrency on/off, no-inplace,
F32-tile Q in flash_attn. The diff-vs-smoke divergence is structural
in `ggml_gallocr` state across multi-graph sched invocations
(`chatterbox_synthesize_mel` calls fewer ggml graphs into
`c->s3gen_ctx->sched` than `chatterbox_synthesize` does before the
UNet runs), and isolating it would need targeted instrumentation of
the allocator's per-tensor address decisions. Standalone handover
prompt prepared in `docs/prompts/`.

### Three upstream PR drafts (`tools/upstream-prs/`)

- **09 metal-q8_0-bit-match** — the kernel above. Concrete patch
  ready to send to `ggml-org/ggml` (Metal changes can go to either
  repo per the README).
- **10 metal-sched-buffer-reuse-drift** — bug report with full
  bisect evidence. The "set_output on 1 makes it WORSE, on all 62
  makes it BIT-PERFECT" finding is the smoking gun that points
  reviewers at `ggml_gallocr_allocate_node`'s reuse-decision-order
  dependency. No patch — out of scope for an application-side fix.
- **11 metal-sched-nan-large-t** — bug report for the Round 8 NaN.
  Same call-context-sensitivity flavour as 10 but at a different
  symptom level (NaN, not drift).

All three drafts strip CrispASR-internal markers (no `(#83)` refs,
no internal language) per the standing repo rule
([[feedback_strip_local_markers]]).

### Lessons for future ggml-drift hunts on Metal

1. **Pin-any-frequent-op-fixes-it is a sync-barrier-density signal,
   not an op-identity signal.** Don't try to identify "the" buggy op
   by op-type bisect alone — there isn't one. Each "approximate
   enough" GPU op contributes; pinning enough of them to CPU resets
   precision via the sync barrier, regardless of which kernel.

2. **`ggml_set_output` changes allocator reuse decisions in
   path-dependent ways.** A single `set_output` can make drift
   *worse* (0.879 < 0.940 baseline). The reuse decision in
   `ggml_gallocr_allocate_node` depends on traversal order
   (`p_hn->n_children == 1 && p_hn->n_views == 0` — both vary as
   markers are added). For an app-side debug knob to work reliably,
   mark ALL intermediates or NONE, not a subset.

3. **Bit-perfect at small T, NaN at large T** is the canonical Metal
   shape for "mixed-backend op routing exposes a sync race." Round 7
   PIN_CPU_OP=mul_mat hit this. Round 9 set_output on 62 hits the
   same shape from a different angle. Whatever the underlying bug
   is, it scales with the working-set size of the allocator's reuse
   pattern.

4. **The diff harness is not a faithful proxy for the production
   pipeline on Metal.** Same model, same graph, same seed, same T,
   different result (bit-perfect vs NaN). The diff harness call
   context goes through `chatterbox_synthesize_mel_from_tokens`
   which runs fewer ggml graphs into `c->s3gen_ctx->sched` before
   the UNet; the smoke context goes through `chatterbox_synthesize`
   which involves more (S3 tokenizer in `--voice .wav` path, plus
   the vocoder on CPU after). Always test on smoke before claiming a
   Metal fix.

5. **Weight residency split is the safe production answer.** When
   you can't fix the GPU compute, push the entire problematic
   sub-graph to CPU via where the weights live. Avoids per-op
   pinning races. Cost: forgo the GPU speedup for that sub-graph.
   On M1 with a small encoder/vocoder shell, that's nearly free.

### Round 9 follow-up (2026-05-24, evening) — correcting items #6, #7, #11

The investigation continued the same day. **Three of the bisect
items above were wrong**, and the headline "set_output on 62 →
bit-perfect in diff harness / NaN in smoke" was a measurement
artifact. The actual bug shape is much tighter.

**The measurement bug.** `compare_with_row_width` in
`examples/cli/crispasr_diff_main.cpp` aggregated cosine and max_abs
without checking for non-finite data. When the diff harness output
is all-NaN, IEEE-754 silently makes every comparison
short-circuit:

- `if (ad > r.max_abs)` — `NaN > 0.0f` is `false`. `max_abs`
  stays at its initial `0.0`.
- `if (denom > 1e-12)` — `sqrt(NaN_sum_sq) > 1e-12` is `false`.
  The cosine accumulator loop skips every row. `cos_min` stays at
  initial `1.0f`, `cos_mean` stays at `1.0f`.

Result: `[PASS] cos_min=1.000000 cos_mean=1.000000 max_abs=0.00e+00 rms=nan`.
The `rms=nan` was visible the whole time but read as
"cosmetic"; the bogus PASS hid an entirely NaN s3gen_mel. Fixed in
commit `4c2e54c0` — `Report.n_nonfinite` is tracked, surfaced as
`non_finite=N/N`, and forces FAIL.

**With the comparison fixed, the actual picture:**

| Config (GPU residency on Metal, T_mel=102) | Result |
|---|---|
| baseline (no marks) | PASS cos_min=0.940 (real degradation) |
| + PRESERVE (~14 block-out marks) | PASS cos_min=0.907 (slightly worse) |
| + DUMP_UNET (62 marks) | **FAIL non_finite=8160/8160** (all NaN, was bogus PASS 1.000) |
| + minimal 2-mark trigger (see below) | **FAIL non_finite=8160/8160** |
| production (CPU residency) | PASS cos_min=0.999980 |

So item #6's "set_output on all 62 → bit-perfect 1.000" was a
phantom. Item #7's "same 62 → NaN in smoke" was right but
incomplete; **the diff harness was also producing NaN**, just
reported as PASS. Item #11's "F32 Q in flash_attn made cos=0.860"
is also suspect — the comparison was running on whatever data
the patched flash_attn produced; with NaN content this metric
isn't meaningful. The bit-match mul_mat (item #1) and the pin
bisect (items #2–#4) measurements are still valid because their
output stayed finite.

**The actual minimum trigger.** With commit `1a37f4c8`'s granular
bisect knobs, the smallest set of marks that flips the Metal UNet
from "degraded but finite" to "all NaN" is **two specific
`set_output` calls**:

1. `dump_db_resnet` — output of the first DOWN block's
   `causal_resnet_block` (shape `(T_mel, 256)`).
2. Any one of `dump_mb_{0,2,4,6,8,10,11}_out` — output of an
   *even-indexed* MID block (or `mb_11`, the last one).

Both alone are finite. Together → NaN. The parity is reproducible:
`mb_{1,3,5,7,9}_out` paired with `dump_db_resnet` produces
identical finite output (`rms=15.806` deterministic). The same
2-mark trigger fires in both the diff harness and smoke — the bug
is **path-independent**. The handover's "diff-vs-smoke
divergence" was entirely the comparison artifact.

**What that means for the upstream story.** The path-dependence
narrative from upstream-prs/10 was wrong. The real shape is:

- 2 specific `set_output` marks on a long all-F32 Metal graph
  cause the Metal kernel chain to write NaN.
- The parity pattern (even mb_*_out indices trigger; odd don't)
  strongly suggests buffer aliasing in `ggml_gallocr`: when an
  even-indexed mb_out and the db_resnet both reserve their own
  buffers (non-reused), the allocator places them in colliding
  pool regions. Odd indices land elsewhere.
- The visible NaN is then "an early kernel reads from a buffer
  that another kernel is writing to" or "a later kernel reads from
  the wrong-aligned region of an overlapping buffer."

This is a much more tractable upstream bug than the original
phrasing. Repro is now self-contained: two env vars + the public
chatterbox CLI. `tools/upstream-prs/10` rewritten in commit
[TBD] with the tight repro.

### Updated lessons (replace #2 and #4 above)

2'. **`ggml_set_output` is a no-monotonicity sledgehammer for
debug.** A single mark CAN make finite output go all-NaN on Metal.
Don't add `set_output` to "see what's happening at point X" — it
changes the allocator's reuse map and may invent NaN that wasn't
in the un-instrumented compute path. The data you dump is data
the *modified* graph produced, not the *real* graph.

4'. **NaN data must fail the diff harness loudly.** If a tensor
comparison can silently report `[PASS] cos=1.000 max_abs=0`
because both data and reference (or just data) contain NaN, all
your downstream bisect is unreliable. Always count + assert on
non-finite before consuming cosine/max_abs. Same lesson would
apply to any other Metric pipeline that aggregates over potentially
non-finite data.

### Round 9 follow-up #2 (2026-05-24, evening) — gallocr ruled out

The "buffer aliasing in `ggml_gallocr`" hypothesis from the prior
follow-up is **wrong**. The allocator is correct. The bug is
elsewhere.

**The instrumentation.** Added a per-node allocator trace
(`CRISPASR_GGML_ALLOC_TRACE=1`) that emits one line per allocate,
free, in-place-reuse, view-reuse, and FREE_SKIP_OUTPUT event in a
diffable single-line format. Captured pass=37 (the first UNet
allocator pass, n_nodes=2715) for both 1-mark and 2-mark configs.

**The diff.** Of 3044 events in pass=37:
- 2108 lines diverge (~70%).
- Exactly +1 FREE_SKIP_OUTPUT and -1 FREE in the 2-mark case
  (`dump_mb_0_out` now pinned instead of freed).
- 1312 ALLOC events have shifted offsets.
- Max chunk-0 offset reached: 8,310,688 (1-mark) vs 8,701,856
  (2-mark) — exactly +391,168 bytes, the size of `dump_mb_0_out`.
- Both runs use only chunk 0; no spill to a new chunk.

**The overlap check.** Wrote a script that scans the trace and
flags any pair of live tensors whose byte ranges overlap inside
overlapping lifetimes (filtering out the legitimate parent→child
in-place reuse). Result: **0 overlaps in both configurations.** The
allocator's plan is correct; no two distinct live tensors share
bytes. If the bug were aliasing, the script would have flagged it.

**The parity mechanism, finally.** In the 1-mark control trace,
even-indexed `mb_*_out` tensors are allocated at `chunk=0 offset=0`
and odd-indexed at `chunk=0 offset=1271296`. The allocator
naturally alternates them across the two large reusable regions of
the same shape (391,168 bytes). Pinning a *low-offset* output
(via `set_output` → `FREE_SKIP_OUTPUT`) blocks reuse of offset=0
and forces ~1300 downstream allocations to higher offsets. Pinning
a *mid-offset* output (the odd indices) is local — only a small
hole, no cascading shift. Hence the even/odd trigger pattern. The
parity is geometric, not aliasing-driven.

**Where the bug actually is.** It must be at a layer above gallocr:
- a Metal kernel whose correctness depends on the specific
  address pattern produced when downstream allocations are shifted;
- `ggml_metal_op_concurrency_check`'s barrier insertion against the
  shifted address layout;
- the Metal output-staging path that handles `set_output`-flagged
  tensors;
- or sched-level interaction between OUTPUT tensors and split
  boundaries.

`GGML_METAL_CONCURRENCY_DISABLE=1` was already negative, so it's
not the *first-order* barrier check, but it may be the
output-staging side. Item #6 (`kernel_flash_attn_ext` Q downcast)
is still a known F32→half precision loss worth a separate PR; it's
not the 2-mark NaN trigger.

### Updated lesson (replace #2' above)

2''. **`set_output` reshapes the allocator's plan, not its
correctness.** When a debug `set_output` causes downstream NaN, the
gallocr plan is still valid (no live-range overlap); what changed
is the geometric layout of where every subsequent tensor lives.
The bug surfaced by the mark is in whatever code path is sensitive
to *which* addresses the kernels run on, not in the allocator
itself. Before chasing gallocr internals, run the per-node trace
(`CRISPASR_GGML_ALLOC_TRACE=1`) and confirm with an overlap scan;
if it's clean, pivot to the backend's kernel/barrier/output paths.

### Toolbox added

| Knob | Effect |
|---|---|
| `CRISPASR_GGML_ALLOC_TRACE=1` | One trace line per allocator event |
| `CRISPASR_GGML_ALLOC_TRACE_MAX_PASSES=N` | Cap to the first N passes (UNet is at pass=37 in the chatterbox CLI smoke path) |
| `CRISPASR_S3GEN_DUMP_UNET_NO_AUTO_MARK=1` | Lets `DUMP_UNET` dump only tensors kept live by other `MARK_*` knobs — without the implicit "mark every dump point" that triggers the NaN cascade. Required for clean per-stage A/B between CPU and GPU. |
| `CRISPASR_S3GEN_UNET_PROBE_BLOCK1=<N>` | Inline-probes the N-th `causal_block1d` call within one UNet graph. Names + marks `dump_probe_after_{im2col,mul_mat,conv1d,transpose_in,norm,ln_mul,ln_bias,transpose_out,mish}` so the bug can be bisected within a single resnet block. N=0 is `s3.fd.db.0.0.b1`. |

Trace format:
```
ggml-alloc: [TRACE] PASS pass=37 n_nodes=2715 n_leafs=909
ggml-alloc: [TRACE] ALLOC node=<n> op=<op> buf=<b> chunk=<c> offset=<o> size=<s> inplace_from=- view_src=-
ggml-alloc: [TRACE] ALLOC_REUSE node=<n> op=<op> ... inplace_from=<parent> view_src=-
ggml-alloc: [TRACE] FREE node=<n> buf=<b> chunk=<c> offset=<o> size=<s>
ggml-alloc: [TRACE] FREE_SKIP_OUTPUT node=<n> buf=<b> chunk=<c> offset=<o> size=<s>
ggml-alloc: [TRACE] PASS_END pass=37
```

### Round 9 follow-up #3 (2026-05-24, late) — bug localized to the first UNet block; pin workarounds were also a measurement artifact

The new tooling pinned the bug down further. Three things to record:

**1. The bug starts at the very first UNet block.** With
`CRISPASR_S3GEN_DUMP_UNET_NO_AUTO_MARK=1` and only the marks needed
to keep `db_resnet` live, the GPU's first `causal_resnet_block`
output is already wrong: 1280 NaN values across 5 contiguous time
frames (t=260..264) for every channel (256), in a tensor of shape
(T=382, C=256). The downstream `set_output`-marked intermediates
that I previously assumed were the "first" NaN are 100% NaN purely
by propagation. The PROBE_BLOCK1 path further pins this to inside
`causal_block1d` (conv1d→norm→mul→add→transpose→mish), and
because the block2 `causal_conv1d` has `K=3`, its receptive field
expands the 3-frame NaN slice from block1 conv1d output into a
5-frame slice at the resnet block boundary.

**2. The bug is not just precision drift; it is structural.** GPU
vs CPU `db_resnet`: cosine similarity is **-0.09** (essentially
uncorrelated), magnitude is 10× off (GPU `max_abs=14.6` vs CPU
`max_abs=1.9`). This is not the kind of error a small FP16
accumulator drift produces. Setting `GGML_PREC_F32` on conv1d's
internal `mul_mat` makes no difference (rms 13.938 → 13.942,
within noise). The wrongness is in the structural arithmetic of
some Metal kernel chain, not in accumulator precision.

**3. The handover's "pinning any frequent op restores cos = 1.0"
claim was also a measurement artifact.** Re-tested
`CRISPASR_S3GEN_UNET_PIN_CPU_OP=norm`, `=mul_mat`, `=add`, `=cont`
with the fixed `crispasr-diff`. All four now FAIL with
`non_finite=8160/8160`. The original PASS reports came from the
pre-`4c2e54c0` harness that silently scored all-NaN as
`cos=1.000`. So both the "pin to CPU restores baseline" and the
"set_output on 62 → bit-perfect" claims were the same one
measurement bug. With it fixed, **no per-op pin currently fixes
the GPU baseline**. The only working configuration remains the
production weight-residency split (`s3.fd.*` on CPU).

**4. The bug is layout-dependent and moves with `set_output`.**
Adding the PROBE_BLOCK1 named outputs cleared the conv1d NaN (its
intermediates were finite) but pushed the first observable NaN to
`dump_db_tb_0` (the basic_transformer_block output). The final
audio was still NaN. This is the classic signal that the bug is a
GPU kernel correctness issue sensitive to which addresses the
allocator picks — adding marks shifts addresses and the kernel
breakage moves.

**Updated lesson (replaces 2'' above):**

2'''. **GPU-residency UNet on Apple Metal is fundamentally broken
in a way no in-tree workaround addresses.** The production fix
(weight-residency split) is not just a perf-comparable
convenience; it is the only configuration that produces correct
output. Documented per-op pins, single-mark and PRESERVE marks,
F32 precision hints, and `GGML_METAL_CONCURRENCY_DISABLE` all
either don't help or actively break it. A real fix needs Metal
shader-level investigation against the address pattern the
allocator produces for this graph — out of scope for a normal
chatterbox session. Until then: `CRISPASR_S3GEN_UNET_GPU_RESIDENCY`
remains an investigation-only knob, not a user-facing option.

### Round 9 follow-up #4 (2026-05-24, late) — TWO bugs in tandem; one fixed, one worked-around

R9 follow-up #3's "address-dependent Metal kernel bug" conclusion
was wrong. The actual root cause is much simpler and tractable.

**Method.** Captured CPU vs GPU bytes at every probe point in the
first `causal_block1d` (`s3.fd.db.0.0.b1`) using
`CRISPASR_S3GEN_UNET_PROBE_BLOCK1=0` + `DUMP_UNET=<tag>` +
`DUMP_UNET_NO_AUTO_MARK=1`. The probe path's `after_im2col` already
showed cos=-0.085 between the two runs — the VERY first GPU kernel
of the UNet was producing structurally wrong output.

A by-hand inspection (`tools/inspect_im2col_dump.py`) confirmed the
kernel's causal-padding shape was correct (first 2 of every 3 floats
zero, third populated). The bug was the VALUES being populated:
CPU read `x[0,0]=1.93` (the first Gaussian noise sample), GPU read
`x[0,0]=-4.05` (a mel-scale value). The im2col kernel arithmetic
was fine; it was reading the wrong source.

A short diagnostic logged the host-side `unet_input[0..7]` values
before upload and a `ggml_backend_tensor_get` readback right after.
Both bit-identical between CPU and GPU runs. So the host array was
the same and the upload landed correctly.
`ggml_backend_sched_get_tensor_backend(sched, unet_input)` returned
`CPU` even under `GPU_RESIDENCY=1` — confirming sched put the
input on CPU and the existing gallocr trace's `MTL0#unet_input#0`
copy was the data path GPU kernels actually read from.

**Root cause.** With UNet weights GPU-resident, the three runtime
inputs (`unet_input`, `time_emb`, `mask`) auto-assigned to the CPU
backend. The scheduler creates an auto-copy
(`{backend_name}#{input}#{copy_idx}`) for the first Metal split that
consumes them; that copy lives in
`ggml_backend_sched_compute_splits` at `ggml-backend.cpp:1555-1567`.
In this 29-split graph, the copy does NOT correctly deliver the
user's uploaded data to the Metal kernel — the kernel reads stale
content at the destination offset. Under certain `set_output`
layouts (the 2-mark trigger from follow-ups #1–#3) the stale
values turn into NaN.

**Fix has two parts** (both required; one without the other still produces wrong output):

**Part A — ggml-side patch** for `ggml_backend_sched_compute_splits`
(ggml-backend.cpp). On every call, sched_split_graph mutates
`gf->nodes[i]->src[j]` to point at fresh `MTL0#input#0` tensors it
just created in `sched->ctx`. Those tensors live until the NEXT
call's `ggml_free(sched->ctx); sched->ctx = ggml_init(...)`, which
happens at the top of the next sched_split_graph. So between calls,
the user's `gf->nodes[i]->src[j]` is a DANGLING POINTER. The next
call's sched_split_graph reads from that dangling pointer to decide
whether an input copy is needed — and the garbage flags rarely match
`GGML_TENSOR_FLAG_INPUT`, so it skips creating new copies. The
consuming Metal kernel then reads stale content at the previous
input_cpy's offset.

Fix: keep a mutation log in sched, and restore each `node->src[j]`
to its original at end of `ggml_backend_sched_compute_splits`. The
user's gf is left untouched between sched calls; the next call sees
the real `GGML_TENSOR_FLAG_INPUT` tensors and creates fresh copies.
Marked `// CrispASR patch (#83 r9 follow-up #4)` with MUST RE-APPLY.

**Part B — application-side pin of `unet_input` only**. Even with
Part A, the sched copy of `unet_input` (CPU buffer → MTL0#unet_input#0)
produces correct bytes at the kernel (verified by inline
`ggml_backend_tensor_get` right before dispatch) yet downstream
compute diverges. Cause not fully traced; sidestepped by placing
`unet_input` on the consuming Metal backend so no copy is needed.

```cpp
if (c->unet_on_gpu) {
    ggml_tensor* ui = ggml_graph_get_tensor(gf, "unet_input");
    if (ui) ggml_backend_sched_set_tensor_backend(c->sched, ui, c->backend);
}
```

**Bisect of which inputs to pin (all under Part A in place):**

| Pinned to GPU | smoke rms | result |
| - | - | - |
| (none) | 14.6 | broken |
| `unet_input` only | **5.14** | works ✓ |
| `time_emb` only | 14.4 | broken |
| `mask` only | 16.0 | broken |
| `unet_input` + `time_emb` | 209 | catastrophic |
| `unet_input` + `mask` | 5.14 | works ✓ |
| all three | 5.14 | works ✓ |

Pinning `time_emb` forces `mish` onto Metal (its input now lives on
GPU), which changes the resnet block's cross-backend topology and
interacts badly with the rest of the graph. **Pin minimally**.

**Verification matrix (M1 Metal):**

| Configuration | Before | After |
| - | - | - |
| Baseline GPU residency, `--tts "Hello."` smoke | `rms=13.938` | `rms=5.143` (ref 5.115) |
| 2-mark NaN trigger | `rms=NaN` | `rms=5.143` |
| Long text (~9 s) | `rms=13.045` | `rms=4.741` |
| Diff harness `s3gen_mel` | `cos_min=0.940` | `cos_min=0.999976` |
| Production CPU residency (default) | `rms=5.139` | `rms=5.139` (no change) |

`cos_min=0.999976` matches the production weight-residency split's
0.999980 (parity within FP16/F32 round-off). The GPU-residency
path is now correct AND faster (~30% on M1 because the encoder /
vocoder no longer need a backend split).

**Replaces lessons 2'/2''/2'''/3 about Metal kernel debug:**

2''''. **For mixed-residency GPU graphs, pin runtime inputs to the
GPU backend explicitly.** ggml's scheduler prefers CPU for tensors
with `GGML_TENSOR_FLAG_INPUT` to minimise upload overhead. The
auto-generated CPU→GPU copy works for short graphs but fails
silently in long F32 graphs with many backend splits (this case:
29 splits). Explicit pinning is a one-line fix and removes the
failure mode. Diagnostic: call
`ggml_backend_sched_get_tensor_backend(sched, input_tensor)` after
`alloc_graph` — if it's `CPU` when you expect `GPU`, pin it.

3'. **First bisect step for "GPU output differs from CPU" should be
"compare bytes at the FIRST kernel's output."** If they already
differ, the bug is in the input pipeline (upload, sched
placement, sched copy), not in the kernel chain. Three rounds of
shader-level investigation were wasted on what turned out to be
an input-routing issue. The fast diagnostic is:

```
CRISPASR_S3GEN_UNET_PROBE_BLOCK1=0 \
CRISPASR_S3GEN_DUMP_UNET=cpu+gpu-probe \
CRISPASR_S3GEN_DUMP_UNET_NO_AUTO_MARK=1
```

then diff `dump_probe_after_im2col.bin` between CPU and GPU runs.
If cos<<1 from the very first probe, look at sched, not at
shaders.

**Upstream story.** The dangling-pointer bug (Part A) is a real
ggml-side issue that affects any user code which calls
`ggml_backend_sched_alloc_graph + graph_compute` twice on the same
gf — the second call silently loses input copies. Filed at
`tools/upstream-prs/10`.

**Bug B is still open** — the workaround (pinning `unet_input` to
the consuming backend) is shipping in production, but the underlying
divergence between "input on CPU + sched-copies to Metal" and "input
directly on Metal" when downstream kernels see the same input bytes
is not understood. Handover at
`handover-prompts/issue83-r9-followup-5-unet-input-routing.md`
collects every diagnostic we ran, every hypothesis we ruled out, and
the structural shape of the bug (the kernel reads correct bytes but
produces wrong output that compounds through CFM). A real fix needs
either an upstream-quality root-cause patch or a confident
"workaround is the right answer because X" explanation.

**Lesson 7. "Pinning fixes it" is not a root cause.** When a
backend-pin diagnostic resolves a correctness issue, you've found
A fix, not necessarily THE fix. The instinct to ship and move on is
strong; resist it. Sidestepping a bug you can't characterise leaves
the same trap loaded for the next graph topology that triggers it.
Carve out time to chase Bug B-class symptoms to root cause even
after a workaround makes the harness green.

`tools/compare_probe_dumps.py` + `tools/inspect_im2col_dump.py`
are the diagnostic scripts.

### Round 9 follow-up #5 (2026-05-24, very late) — Bug B narrowed to rc.weight residual conv; root cause still unidentified

Continued the Bug B chase from #4. Did not find root cause but
eliminated several hypotheses and localized the divergence point.

**Verified (from #4's open questions):**

- Host `ggml_backend_tensor_get(MTL0#unet_input#0)` immediately before
  the FIRST UNet `im2col` dispatch returns the correct Gaussian-noise
  bytes (`1.9269 1.4873 …`) at `data=0x159114000 offs=0`. Same buffer
  and offset across all 4 dispatches observed (cond + uncond × 2
  steps). The host side is fine.

**New finding — where Bug B actually manifests:**

Used `CRISPASR_S3GEN_UNET_PROBE_BLOCK1=<N>` to compare per-block
intermediates between Path X (workaround) and Path Y (no-pin):

| Probe target | Stage | cos(Path X vs Y) |
| - | - | - |
| Block 0 (`db.0.0.b1`) | after_mul_mat → after_mish | 1.00000 |
| Block 1 (`db.0.0.b2`) | after_mul_mat → after_mish | 1.00000 |
| `dump_db_resnet` (resnet block output) | post-residual-add | 0.32261 |
| `dump_rc_out_db00` (rc conv output, **new probe**) | F32 residual conv | **0.02211** |
| Block 2 (`mb.0.0.b1`) | after_im2col | 0.36914 |
| Block 13 (`mb.5.0.b2`) | after_im2col | 0.70562 |
| Block 27 (`ub.0.0.b2`) | after_im2col | 0.92610 |

So Bug B's first divergence is at the **residual conv** (`rc.weight`)
inside `causal_resnet_block` at `s3.fd.db.0.0`. b1 and b2's conv1d
outputs are bit-identical between the two paths — both read from the
same Metal compute-buffer offset 0 (MTL0#unet_input#0 in Path Y,
unet_input in Path X). The rc conv reads from the **same** offset
yet produces output with `rms=0.0645` (Bug B) vs `rms=0.3631`
(workaround) — a ~5.6× magnitude collapse and `cos=0.022`. Bug B's
values look like rc.weight × ~zero input; workaround's values look
like rc.weight × Gaussian-noise input (σ≈1).

**Eliminated hypotheses (all left rms ≈ 16 unchanged):**

1. **CPU store-buffer / cache coherency.**
   `CRISPASR_FORCE_DMB=1` inserts `__sync_synchronize()` after the
   shared-buffer host memcpy. No effect — the GPU was already seeing
   the writes correctly. (The host-side readback proved this from the
   start; the experiment just rules out a write-buffer race
   conclusively.)
2. **GPU-side cache invalidation.** `CRISPASR_FORCE_BLIT_COPY=1`
   replaces the host memcpy with a Metal blit-encoder copy in its own
   committed command buffer. No effect — the GPU is performing the
   copy itself, fully synchronous to subsequent compute. Bug B
   persists.
3. **Intra-encoder concurrency.** `GGML_METAL_CONCURRENCY_DISABLE=1`
   and `GGML_METAL_GRAPH_OPTIMIZE=0` both leave Bug B unchanged.
4. **Cross-command-buffer race.** Metal splits each
   `graph_compute_async` across `n_cb+1` command buffers; each cb has
   its own `mem_ranges`, so cross-cb conflicts are not tracked at the
   ggml layer. Setting `CRISPASR_METAL_N_CB=2` produces rms=16.217
   (essentially unchanged from default n_cb=1 rms=16.129). Setting
   `n_cb=0` is broken (division-by-zero in `n_nodes_per_cb`). So
   command-buffer fan-out is not the cause.
5. **Missing or stale `mem_ranges` barrier.**
   `CRISPASR_METAL_FORCE_BARRIER=1` emits a Metal memory barrier
   before EVERY metal-op encode (effectively serializing the whole
   graph). Bug B still produces rms=16.183. So no concurrency hazard
   inside the encoder is responsible.

**Bug shape — unanswered:**

Same Metal pipeline, same kernel args, same input bytes (verified
host-side), same Metal buffer offset. b1 and b2's conv1ds read this
input correctly. rc's conv1d does not. The only differences I can
think of between b1's im2col and rc's im2col on the same input:

- b1 uses KH=1, KW=3, p0=2 (causal-padded). Threads with `iiw < 0`
  branch to write `0.0f`.
- rc uses KH=1, KW=1, p0=0. No padding; all threads enter the
  data-read branch.

Both use the same `kernel_im2col_f32` template with identical args
plumbing; there is no KW=1 fast path. Both dispatches are
`(IC=320, OH=1, OW=T_mel)` threadgroups × `(N=1, KH, KW)` threads.

**Loose lead — alloc-plan aliasing within the buf=0 chunk:**

In the alloc trace (with probe + dump_db_resnet preserved), the
first UNet pass shows `MTL0#unet_input#0` at offset 0, then
`node_47 op=IM2COL` (rc's im2col) is allocated at offset 6024704
while `MTL0#unet_input#0` is **freed** by gallocr the moment node_47
is registered. Then `node_51 op=MUL_MAT` (rc's mul_mat) is allocated
**at offset 0**, reusing MTL0#unet_input#0's just-freed slot. The
same plan exists in Path X (with unet_input itself at offset 0
instead of MTL0#unet_input#0), and Path X works — so the pure
aliasing isn't the cause, but the interaction of "sched-injected
input_cpy at offset 0 vs user-pinned input at offset 0" remains
suspicious.

**Operational note — dump artifacts:**

`ggml_set_output` on a probe target keeps that one slot live but
adds new tensors to the graph (the implicit "(transposed) (cont)"
follow-ups), which **changes the alloc plan downstream**. A run
configured with `PROBE_BLOCK1=1 + MARK_DB_RESNET=1` broke the
workaround (rms=17.355). So you cannot freely add probes to compare
"Path X with marks" vs "Path Y with marks" — the markings perturb
the graph in their own right. Future investigators: probe one path
exclusively per run, or accept that mark-induced offset shuffling is
itself part of the experiment.

**Diagnostic knobs added (env-gated, no runtime impact when unset):**

- `CRISPASR_NO_INPUT_PIN` — disables the unet_input pin workaround,
  for reproducing Bug B.
- `CRISPASR_IM2COL_DBG` — logs first 8 host bytes of `op->src[1]` for
  the first UNet `kernel_im2col_f32` dispatch.
- `CRISPASR_FORCE_DMB`, `CRISPASR_FORCE_BLIT_COPY`,
  `CRISPASR_METAL_FORCE_BARRIER`, `CRISPASR_METAL_N_CB` — the
  hypothesis-elimination knobs above.
- `CRISPASR_S3GEN_UNET_PROBE_RC_OUT` — dumps rc's residual-conv
  output at `s3.fd.db.0.0` as `dump_rc_out_db00`.

**Also eliminated: Metal op fusion.** Set
`GGML_METAL_FUSION_DISABLE=1` to disable the cross-op fusion
optimizer; Bug B's smoke is still rms=16.129. So the rc-specific
divergence isn't an artifact of the fusion code path either.

**Structural difference between b1 (works) and rc (broken):**

Both fire `kernel_im2col_f32` on the same `MTL0#unet_input#0` at
the same Metal buffer offset 0, inside the SAME sched split
(SPLIT #0 of the UNet1D graph — verified via `GGML_SCHED_DEBUG=1`,
which shows just one Metal split with unet_input as its input; all
subsequent Metal splits consume only the time_proj outputs that
loop back from CPU). The only mechanical differences are:

| | b1's conv1d | rc's conv1d |
| - | - | - |
| `KW` (kernel width) | 3 | 1 |
| `p0` (left pad) | 2 | 0 |
| im2col output `ne[0]=CHW` | 960 | 320 |
| Threads per threadgroup (`ntptg0, KH, KW`) | (1, 1, 3) | (1, 1, 1) |
| Causal-pad branch enters | yes (for iow=0..1) | never |

So rc dispatches `kernel_im2col_f32` with a 1×1×1 threadgroup — one
active lane, 31 masked-off SIMD lanes on the M1's 32-wide simdgroup.
That's an unusual but documented configuration. Whether this
single-lane threadgroup tickles a driver/compiler bug on M1 specifically
is a leading hypothesis but unverified. The next experiment would
either:

**Tested — eliminated: 1×1×1-threadgroup im2col edge case.**
Added `CRISPASR_S3GEN_RC_AS_MUL_MAT=1` that bypasses `ggml_conv_1d`
for rc when KW=1 and emits a direct
`mul_mat(cont(transpose(residual)), reshape_2d(rc_w))` instead. Test
results on M1:

| Configuration | rms |
| - | - |
| Path X (workaround), `RC_AS_MUL_MAT=1` | 5.143 (matches workaround baseline — correctness preserved) |
| Path Y (Bug B), `RC_AS_MUL_MAT=1` | 16.837 (Bug B still present) |

So the 1×1×1-threadgroup im2col edge case is **NOT** the cause of
Bug B. Bypassing rc's im2col leaves Bug B intact. Whatever's wrong
with Path Y's rc compute is also wrong with a mul_mat-only rc compute
on the same `MTL0#unet_input#0` buffer. The divergence is not in
the rc kernel itself.

This is a significant negative result — it pushes the bug AWAY from
rc and suggests the cos=0.022 measurement of `dump_rc_out_db00` may
be a downstream symptom rather than the causal location. Working
hypotheses to try next:

- The `dump_rc_out_db00` divergence might be a **dump artifact**
  itself: `MARK_DB_RESNET=1 + PROBE_BLOCK1=1` already broke the
  workaround at rms=17.355, so probe-induced offset shifts are
  measurably destabilising. The rc-output dump landing at a
  different offset in Path X vs Path Y could mean we're reading
  different bytes for reasons unrelated to rc's correctness.
- **Capture per-thread reads via a custom Metal kernel.** Replace
  `kernel_im2col_f32` with a debug variant that writes
  `(in*ofs0 + iic*ofs1 + iiw, x[offset_src])` pairs to a side
  buffer. Compare what GPU thread (iic, iow, ikw) saw vs what
  host expected.
- **Probe the SECOND CFM step** specifically. The first step's
  compute uses freshly memcpy'd data; subsequent steps reuse the
  same Metal compute buffer offsets with new memcpy contents. If
  a buffer's state from step N-1 leaks into step N's compute,
  the divergence might only appear from step 1 onward — and the
  cumulative compound through 10 steps drives the rms gap.
- **Pin only `time_emb` or only `mask`** (R9 #4's bisect tested
  these but didn't try probing per-step compute outputs). The
  combinations may encode which sched-injected input_cpy is
  problematic.

**Bug B remains open.** Workaround is still shipping.

**MAJOR ISOLATION (2026-05-24, late) — Bug B is in the UNCOND pass only.**

Each CFM step calls `run_denoiser` twice: first with the conditioned
input `[noise, mu, spk, cond]` (cond pass), then with
`[noise, 0, 0, 0]` (uncond pass) for the CFG combination. With a
dup-named `dump_denoiser_out` tensor + `CRISPASR_S3GEN_UNET_PROBE_DENOISER_OUT=1`
and a static-counter cond/uncond-tagged DUMP_UNET, the step-0
denoiser_out comparison between Path X (workaround) and Path Y
(Bug B) is:

| Pass | bug rms | wrk rms | cos(bug, wrk) |
| - | - | - | - |
| cond   | 3.9634 | 3.9634 | **1.00000** |
| uncond | 0.9474 | 5.0228 | **0.21431** |

The cond pass is BIT-IDENTICAL between the two paths. Bug B only
manifests on the **second** `graph_compute` call within a CFM step.
The cond and uncond runs are structurally identical (same gf, same
sched topology, same alloc plan); only the host-side input data
differs.

This rules out anything intrinsic to sched-copying `unet_input` — the
first sched-copy + compute works perfectly. The bug is in what
happens *between* the cond compute and the uncond compute, or
specifically in the uncond compute reading the newly-memcpy'd
unet_input data.

Working hypothesis: on Apple Silicon, the GPU's L1/L2 caches retain
shared-mode buffer state across consecutive command-buffer
submissions. The cond compute leaves cached data; the uncond
CPU memcpy overwrites system memory, but no Metal API invalidates
the GPU's caches before the next dispatch. The uncond kernel then
reads stale data the memcpy never reached. None of the tested
barriers (`FORCE_DMB`, `FORCE_BLIT_COPY`, `FORCE_BARRIER`,
`METAL_N_CB`) address this — they all run AFTER the memcpy but
before the new compute, where they SHOULD trigger cache
invalidation, but apparently don't on M1 for shared-storage
buffers modified by CPU.

Observation: Bug B's uncond denoiser_out has rms=0.95 (very small,
near-bias-only) while workaround's is rms=5.02. The first 8 values
have CONSISTENT SIGNS and gradients but ~5× smaller magnitude. This
is what you'd expect if the model received near-zero or attenuated
input on the GPU side (consistent with the kernel reading something
other than the noise channel).

Critical next experiment: instead of trying to fix the cache via
barriers, **force MTL0#unet_input#0 onto a private-storage buffer**.
Private mode requires explicit blit transfers (which Metal manages
cache state around) — if Bug B disappears, the bug is confirmed as
"M1 GPU caches shared-storage buffer reads across cmd-buf
boundaries, no Metal API invalidates them when the CPU writes
between cmd-bufs". A more surgical workaround would then be to
keep `unet_input` on a private buffer specifically in this path.

Knobs added this session:

- `CRISPASR_S3GEN_UNET_PROBE_DENOISER_OUT=1` — emits a dup-named
  `dump_denoiser_out` tensor so the existing DUMP_UNET filter
  catches it.
- `CRISPASR_S3GEN_DUMP_UNET=<tag>` — dump files now include
  `<tag>-cond` / `<tag>-uncond` prefixes via a static call counter
  inside `run_denoiser`.
- `CRISPASR_S3GEN_RC_AS_MUL_MAT=1` — bypasses `ggml_conv_1d` for
  the residual conv (the 1×1×1 im2col edge case). Confirmed
  irrelevant to Bug B.

**Conclusive host-vs-GPU divergence proof.** Extended IM2COL_DBG
to read channels 0/80/160/240 (noise/mu/spk/cond) instead of just
channel 0. In Bug B mode with `CRISPASR_IM2COL_DBG=1`:

```
dbg 0 (step 0 cond):    noise=1.9269 mu=0.24 spk=0.17 cond=-6.75
dbg 1 (step 0 uncond):  noise=1.9269 mu=0.00 spk=0.00 cond=0.00
dbg 2 (step 1 cond):    noise=1.8502 mu=0.24 spk=0.17 cond=-6.75
dbg 3 (step 1 uncond):  noise=1.8502 mu=0.00 spk=0.00 cond=0.00
```

All four host_get readbacks return the EXACTLY CORRECT bytes —
mu/spk/cond are zero in uncond passes, populated in cond. Same data
ptr `0x121120000`, same Metal buffer offset 0, same shared-mode
buffer across all four dispatches.

Yet the compute output is still wrong in Bug B (rms=16.245). So
the GPU's view of `MTL0#unet_input#0` in the second `graph_compute`
(uncond) differs from what host_get returns at the SAME byte offset
of the SAME shared-storage Metal buffer.

Already tested and ruled out for fixing this divergence:
`FORCE_DMB`, `FORCE_BLIT_COPY`, `FORCE_BARRIER`, varying
`METAL_N_CB`, `GGML_METAL_CONCURRENCY_DISABLE=1`,
`GGML_METAL_FUSION_DISABLE=1`, `GGML_METAL_GRAPH_OPTIMIZE=0`,
`GGML_METAL_SHARED_BUFFERS_DISABLE=1` (private-storage buffers),
`CRISPASR_S3GEN_RC_AS_MUL_MAT=1`.

This is now an extremely well-characterized but genuinely
intractable Metal/M1 bug. The cond pass works correctly; the
uncond pass doesn't, on identical-shape graphs with identical
sched alloc plans and identical host-visible buffer bytes.

Likely next steps (none cheap):

- Capture an Xcode GPU frame trace of the cond and uncond passes
  side by side; compare buffer state inspection at dispatch time.
- File a minimal reproducer with Apple (chatterbox sched + sched
  tensor_copy + repeated CPU memcpy on a shared-storage Metal
  buffer + back-to-back `commandBufferWithUnretainedReferences`).
- Switch the sched layer's INPUT-flagged `ggml_backend_tensor_copy`
  path to issue a Metal compute kernel (not a memcpy) for the
  CPU→Metal copy step on shared-storage buffers. This would route
  through a known-working Metal command pipeline rather than a
  raw host write.

**🎉 BUG B FIXED — sched `parallel=true` (R9 #5, very late).**

Of the three proposed next experiments above, the third one (route
the CPU→Metal copy through a stronger Metal synchronisation path)
turned out to be DIRECTLY available via `ggml_backend_sched_new`'s
`parallel` flag. With `parallel=true`:

- `n_copies = GGML_SCHED_MAX_COPIES = 4` input-copy slots per
  input tensor (sched alternates between them across calls).
- Sched creates `ggml_backend_event_t` objects per backend per copy
  slot and uses `event_record` / `event_wait` for cross-backend
  ordering instead of plain `ggml_backend_synchronize`.
- On the Metal backend that translates to
  `MTLSharedEvent` `encodeSignalEvent` /
  `encodeWaitForEvent` commands encoded into the command buffer.
- Those commands carry proper GPU-cache invalidation semantics
  between consecutive `commandBufferWithUnretainedReferences`
  submissions on the same `MTLCommandQueue`.

With `parallel=false` (the chatterbox default until this fix),
sched falls back to `[cmd_buf_last waitUntilCompleted]` for the
between-submissions sync. That waits for the prior command buffer to
complete but does NOT invalidate the GPU's L1/L2 cached view of a
shared-storage `MTLBuffer` that the CPU just memcpy'd. The next
dispatch's kernel then reads stale data.

The cond pass works because nothing has cached the buffer yet (it's
the first submission). The uncond pass fails because the cond pass
left the buffer cached on the GPU side; the CPU memcpy of new
uncond data updates system memory but doesn't reach the GPU cache;
the next compute reads stale cond-pass-leftover data.

Switching `chatterbox_s3gen_init_from_file` to call
`ggml_backend_sched_new(..., parallel=true, ...)` fixes Bug B.
Verified:

- M1 Metal smoke, GPU residency, no workaround: `rms 16.x → 5.143`.
- M1 Metal smoke, CPU residency (production default): `rms 5.139`
  (unchanged — no regression).
- Diff harness: `s3gen_mel cos_min = 0.999976` (matches the prior
  workaround's baseline).
- Long text (~192 frames): `rms 5.667` (intelligible).

The unet_input GPU pin workaround in `cfm_euler_solve::run_denoiser`
is no longer needed and was removed in the same commit (the
`CRISPASR_NO_INPUT_PIN` env override is also gone). The
**Bug A fix** (sched src-mutation log in `ggml-backend.cpp`) is
INDEPENDENT and continues to be required — that bug is about
dangling `node->src[j]` pointers across `alloc_graph` calls, not
about cache coherency.

**Lesson 7' (replacing "Pinning fixes it is not a root cause"):**
The pin was indeed a workaround for a deeper sched-synchronisation
issue, not the root cause. The cond pass worked because the first
graph_compute call doesn't see the cached-buffer-state problem.
"Pinning unet_input" worked because pinning made unet_input be on
Metal directly, eliminating the sched-injected MTL0#unet_input#0
copy — and with no sched copy, no host memcpy on a previously-cached
buffer, no stale-cache read. So the workaround DID address a real
symptom, just not the right one. The right fix is to use stronger
synchronisation primitives between consecutive graph_compute calls
on the same gf (or alternate input-copy slots, which `parallel=true`
does for us automatically).

**Lesson 8. When stuck on a Metal cache-coherency-shaped bug,
check `ggml_backend_sched_new`'s `parallel` flag.** The plain
synchronize path is sufficient for many graph topologies but
genuinely insufficient for back-to-back compute calls that read the
same shared-storage buffer the CPU just wrote between them. The
event-based path is the documented way to get proper cross-submission
ordering and Apple-Metal cache semantics. It costs memory (4×
input-copy slots) but for graphs the size of the chatterbox UNet
that's negligible.

**Performance follow-up — Hello. M1 wallclock, 5-run median.**

The initial fix (sched `parallel=true` unconditionally) measured at
38.5 s CPU residency / 44.6 s GPU residency in a busy session — i.e.
no GPU residency speed advantage over CPU, vs the prior workaround's
claimed ~34 s GPU / ~43.7 s CPU. A follow-up commit gates
`parallel=true` on `c->unet_on_gpu` so CPU residency keeps the
lower-overhead `parallel=false` path; on a quieter machine 3-run
medians become:

| Configuration | Time | rms |
| - | - | - |
| GPU residency, parallel=true (gated) | **31.0 s** | 5.143 ✓ |
| CPU residency, parallel=false (gated) | **30.0 s** | 5.139 ✓ |

So the gate restores the prior CPU-residency perf path and confirms
GPU residency at the workaround's prior speed band. The earlier
44 s GPU number was mostly machine variance (concurrent worker
sessions hogging the GPU during measurement), not parallel=true
overhead per se. The gate is a real but small CPU-residency win
(~8 s saved by avoiding 4× input-copy slots + event allocation
when Bug B can't manifest).

### Diff-harness `t3_prefill_emb` FAIL is a tokenizer mismatch, not a bug (2026-05-24, very late)

While closing out R9 #5 we looked into the long-standing
`[FAIL] t3_prefill_emb[0]  cos_min≈0.02 cos_mean≈0.81`
result in `crispasr-diff chatterbox`. Per-row debug shows:

- Rows 0–33 (cond rows from the perceiver + speaker proj): all pass
  at `cos > 0.9999`. The C++ cond builder is correct.
- Rows 34–36 (first 3 text rows = BOS + 2 text tokens): pass.
- Rows 37+ (rest of the text rows) and the `speech_start` row: fail
  at `cos < 0.5`, dropping to `cos ≈ 0.1`–`0.02`.

Two sub-issues stack here:

1. **Default `CHATTERBOX_SYN_TEXT` mismatch.** The crispasr-diff
   harness defaults to `"Hello world."` but the python ref archive
   (`chatterbox-ref.gguf`) was generated with
   `"Hello there, this is chatterbox speaking."` (stored in the
   archive as `crispasr.ref.chatterbox_syn_text`). Running the diff
   with `CHATTERBOX_SYN_TEXT="Hello there, this is chatterbox speaking."`
   removes the obvious text mismatch but doesn't fix the failure —
   it just shifts the divergence pattern.

2. **Tokenizer algorithm mismatch.** Chatterbox-base's
   `chatterbox.t3.text_tokens` vocab in the GGUF has 704 entries
   including `[STOP]`, `[UNK]`, `[SPACE]` plus lowercase a–z plus
   multi-char tokens (`th`, `in`, `the`, `er`, `ou`, …) — i.e. the
   vocab is BPE-like. But the GGUF stores **no `merges` array**, so
   `cb_tokenizer::has_bpe` is false and the C++ code falls through to
   the legacy char-by-char `tokenize_text`, which:
   - **skips uppercase letters** (vocab is lowercase only);
   - **skips space characters** (vocab uses `[SPACE]`, not `' '`);
   - **never tries multi-char vocab entries.**

   We tried replacing `tokenize_text` with a normalising
   (lowercase + `' '`→`[SPACE]`) **greedy-longest-match**
   tokenizer. It produces a token sequence with the right *shape*
   (matches python ref's length of 25 text tokens for the
   reference text instead of the legacy 38) and the diff goes
   from `cos_mean=0.55` to `cos_mean=0.66` — still failing because
   greedy longest-match isn't python's BPE-with-merges algorithm.
   We also observed that the production audio for the same text
   sounds slightly different (T=138 frames vs T=162 with the
   legacy tokenizer, rms 4.02 vs 5.07; the *legacy* output is
   actually closer to the python reference's mel rms ≈ 5.12).

   So the legacy char-level fallback, despite being structurally
   wrong, happens to drive the t3 LLM to a state that produces
   audio whose final rms is closer to python's than our
   greedy-longest-match tokenizer does. The t3 model is robust
   enough to "ignore" the bad tokens. Both produce intelligible
   audio.

**Conclusion.** The `t3_prefill_emb` FAIL is real but not a
production correctness bug: the audio works, just from a different
token sequence than python's. To make the diff pass we would
need to either (a) ship `chatterbox.t3.merges` in the GGUF
converter and the C++ loader, and use the proper BPE path in
`tokenize_text_bpe`, or (b) port python's exact tokenizer
algorithm to C++ (`EnTokenizer` from the chatterbox repo, which
uses the HF `tokenizers` library and a `tokenizer.json` config).

**Not done this session.** The legacy char-level tokenizer
keeps shipping because flipping to greedy-longest-match measurably
shifts production audio (and not toward python's output). The
diff harness should mark this case as a known limitation rather
than gating CI on it. Recommended follow-up:

- Re-export the chatterbox-base GGUF with `chatterbox.t3.merges`
  populated from the chatterbox tokenizer.json so the BPE path
  becomes usable.
- Then either: switch `chatterbox_dump_t3_prefill_emb` to call the
  BPE tokenizer when merges are available, or keep the legacy
  path and accept the diff failure.
- Optionally: have `crispasr-diff` read the
  `crispasr.ref.chatterbox_syn_text` metadata key from the ref
  archive so the default text matches the reference.

### Resolution — patched the legacy GGUF, BPE path now matches python exactly (2026-05-25)

After re-examining python's `EnTokenizer.encode` in
`resemble-ai/chatterbox/src/chatterbox/models/tokenizers/tokenizer.py`,
the algorithm is literally:

```python
def encode(self, txt: str):
    txt = txt.replace(' ', SPACE)        # ' ' → '[SPACE]'
    code = self.tokenizer.encode(txt)
    return code.ids
```

i.e. swap each ASCII space for the literal multi-char token `[SPACE]`
(id 2 in the vocab) BEFORE the underlying HF tokenizer's Whitespace
pre-tokenizer + BPE runs. The C++ `tokenize_text_hf_bpe` already
implemented exactly this algorithm — the gap was the GGUF, not the
tokenizer code:

- The chatterbox-base converter
  (`models/convert-chatterbox-to-gguf.py`) didn't write
  `tokenizer.ggml.merges` until commit `372127f4 fix chatterbox`
  (2026-05-08).
- The auto-downloaded `chatterbox-t3-q8_0.gguf` on HF
  (`cstr/chatterbox-GGUF`) was uploaded 2026-05-07, BEFORE that
  commit. So users hitting the auto-download get a vocab-only
  GGUF (`chatterbox.t3.text_tokens` only, no
  `tokenizer.ggml.merges`).
- With no merges in the GGUF, `cb_tokenizer::has_bpe` is false and
  `chatterbox_dump_t3_prefill_emb` falls through to the legacy
  char-level `tokenize_text` (which silently drops uppercase
  letters and spaces because the vocab is lowercase + uses
  `[SPACE]`, not `' '`).

Patching the existing GGUF in place with a fresh export of the
tokenizer.json's merges (`models/patch-chatterbox-gguf-add-merges.py`,
added 2026-05-25) makes the BPE path active and the diff harness
result jumps from `[FAIL] cos_mean=0.55` to
`[PASS] cos_min=0.999940 cos_mean=0.999985` — i.e. the C++
tokenization is bit-identical to python's for both `"Hello world."`
and `"Hello there, this is chatterbox speaking."`.

Production rollout: re-upload `chatterbox-t3-q8_0.gguf` (and
`chatterbox-t3-f16.gguf`) to `cstr/chatterbox-GGUF` after running
the converter from current `main`, OR upload the patched file under
a different name and update `crispasr_model_registry.cpp` to point
at it. The C++ side is already correct. The patched GGUF was
verified locally as `chatterbox-t3-q8_0-bpe.gguf`.

Follow-up for multilingual v3: the language signal should be injected
as a model-input token, not concatenated into the user text. Otherwise
the language tag can leak into speech output as a spoken prefix even
when the tokenizer contains the tag.

### Resolution — multilingual text prep now shares the upstream punctuation / lowercase path (2026-06-18)

The C++ Chatterbox synthesis path now runs all text through a shared
normalizer before tokenization:

- collapses repeated whitespace and trims Unicode-space variants that
  the old path left intact;
- normalizes the common punctuation shorthands from upstream `punc_norm`
  (`...`, `:`, `;`, `" - "`, curly quotes, dashes);
- lowercases ASCII text when a language tag is set, so the multilingual
  model no longer sees the uppercase-heavy text that the Python MTL
  tokenizer would have normalized away.

That closes the obvious `-l fr` / `-l ar` drift in the C++ path. Full Unicode
**NFKD** was added later (2026-06-21, see "NFKD resolution" below); the
script-specific normalizers for zh/ja/he/ko/ru (cangjie / kakasi / dicta / jamo
/ stress) are still unimplemented (they need external libs).

### Root cause — the published q4_k multilingual GGUF used the wrong tokenizer (2026-06-18)

The `cstr/chatterbox-GGUF/chatterbox-t3-q4_k.gguf` artifact tested on
2026-06-18 is internally inconsistent: `chatterbox.t3.text_vocab_size`
is 2454 from `t3_mtl23ls_v3.safetensors`, but the embedded
`tokenizer.ggml.tokens` array has only 2352 entries. That file was
converted with `mtl_tokenizer.json`; upstream
`ChatterboxMultilingualTTS.from_local()` loads
`grapheme_mtl_merged_expanded_v1.json` for multilingual inference.

Audible symptom: no-language prompts like `"Justice justice."` and
`"Bonjour tout le monde."` synthesize intelligibly, but adding `-l fr`
produces a spoken leading artifact before the real phrase, and `-l ar`
produces gibberish. This is a TTS/model artifact issue, not ASR.

Fixes:

- converter tokenizer selection now prefers
  `grapheme_mtl_merged_expanded_v1.json`;
- conversion fails if tokenizer length differs from T3 `text_vocab_size`;
- runtime load fails on the same mismatch so broken GGUFs cannot produce
  plausible but invalid multilingual audio.

Follow-up verification after regenerating and uploading
`cstr/chatterbox-GGUF` on 2026-06-18:

- all rebuilt T3 variants (`f16`, `q8_0`, `q4_k`) have
  `chatterbox.t3.text_vocab_size=2454`, `tokenizer.ggml.tokens=2454`,
  and `tokenizer.ggml.merges=265`;
- `-l fr` is not a no-op in the C++ CLI. With fixed Q4_K, seed 123,
  and text `"justice justice"`, no-language tokenization is
  `[START], J, ust, ic, e, [SPACE], just, ic, e, ., [STOP]`, while
  `-l fr` is `[START], [fr], just, ic, e, [SPACE], just, ic, e, .,
  [STOP]` (`[fr]` is token id 634);
- the no-language repeat is bit-identical, but no-language vs `-l fr`
  produces different AR speech tokens, different duration
  (`1.52 s` vs `1.92 s` for `"justice justice"`), and near-zero
  waveform cosine (`-0.004`). For `"bonjour tout le monde"`, `-l fr`
  similarly inserts `[fr]`, changes speech tokens, and improves the
  Parakeet roundtrip from `"Bonjour tout monde."` to
  `"Bonjour tout le monde."`.

That proves the CLI/runtime language wiring is active after the artifact
fix. It does **not** prove French quality is solved; listening tests still
reported heavy accent / imperfect pronunciation on some Q4_K French
samples, so remaining work is model-semantics / upstream-text-prep parity,
not "is `-l` wired at all?".

### NFKD resolution — the upstream-text-prep parity gap was NFKD (2026-06-21, #170)

The "upstream-text-prep parity" remaining work above was concrete: an external
reporter (#170) found that with a correctly-paired multilingual GGUF, `-l ar` on
**partial-diacritic** Arabic still prepended spurious onset letters
(`أعْلَنَ`→"za3lana") that the Python reference did not. Root cause, proven
through the real `grapheme_mtl` tokenizer (not guessed): the C++ normalizer did
punc/lowercase but **skipped NFKD**, while upstream
`MTLTokenizer.preprocess_text` runs `unicodedata.normalize("NFKD",
text.lower())` before tokenizing.

- For the working simple string `مرحبا بالعالم`, NFKD is a **no-op** → token ids
  identical with/without NFKD → no bug (matches the issue: simple case worked).
- For the hallucination string, NFKD **changes it** (131→137 codepoints): the
  precomposed `أ` (U+0623) decomposes to base alef (U+0627) + combining hamza
  (U+0654). Without NFKD the C++ tokenized `أ` as one rare grapheme id **2353**;
  with NFKD upstream splits it into **1456, 1458**. Token 2353 is the
  under-trained grapheme the T3 renders as the spurious consonant. That is why
  *partial diacritics specifically* trigger it — أ إ آ ؤ ئ are the
  NFKD-decomposable letters.

Fix: `tools/gen_nfkd_table.py` emits exact NFKD decomposition + combining-class
tables from `unicodedata`; `src/chatterbox_nfkd.h` applies them (Hangul
decomposed arithmetically, the rest tabled; bit-exact vs `unicodedata` across
Arabic / accented Latin / Hangul / ligatures). After the fix the C++ multilingual
token ids are **byte-identical to upstream** for both the simple and the
hallucination string (139/139, 2353 gone), validated through the formal
crispasr-diff `t3_text_tokens` stage. **Text-prep code fix — no GGUF
reconversion.** Lesson: a C++ TTS text path must replicate the upstream
tokenizer's *entire* `preprocess_text` (NFKD + casefold + language hooks), not
just punctuation — precomposed graphemes are the trap, and they only surface on
inputs that actually contain them.

### Operational note — the regen variants already have merges

The `chatterbox-t3-q8_0-regen.gguf` and `chatterbox-t3-q4_k-regen.gguf`
in `/Volumes/backups/ai/crispasr/` (dated 2026-05-08) DO contain
`tokenizer.ggml.tokens` + `tokenizer.ggml.merges` — they were
generated from the converter after `372127f4`. Only the auto-download
target (`cstr/chatterbox-GGUF/chatterbox-t3-q8_0.gguf` from 2026-05-07)
is missing them. All other BPE-using models in the codebase
(chatterbox-turbo, funasr-mlt-nano, qwen3-asr, mimo-asr) export
merges correctly.

### Lesson 9. When a "tokenizer mismatch" hypothesis explains the symptom but the algorithm looks right, check the GGUF for missing tokenizer keys before rewriting C++.

We almost rewrote the C++ tokenizer to use whitespace pre-tokenization
without `[SPACE]`. That was wrong (python's chatterbox-specific
wrapper inserts `[SPACE]` before invoking HF). The right diagnosis
was "the existing C++ code matches python exactly but is gated on
`has_bpe` which is false because the GGUF doesn't have
`tokenizer.ggml.merges`." A 30-second `python -c "open(gguf).read()
.find(b'tokenizer.ggml.merges')"` check would have saved several
hours of speculative tokenizer rewriting.

---

## FA per-head additive mask CUDA kernel — what the upstream signature already gave us (issue #81 #06, May 2026)

### Read the kernel signature before estimating patch size

The first design pass on the per-head FA mask patch (in the now-
retired `issue81-phase1-uar-wip` branch) estimated ~300-500 LOC
across 4-6 .cuh files — touching the launcher, the kernel-body
mask load, the tile_mask broadcast, and possibly the
quantization fast path. The actual minimal patch is **4 LOC of
kernel-body code** (+ ~10 LOC of dispatch gate + CMake/CDef
wiring) for a total of ~45 LOC.

The difference is that the MMA-F16 kernel signature
(`flash_attn_ext_f16` in `fattn-mma-f16.cuh`) **already took all
six mask dims/strides as parameters** — the launcher plumbed
`nb31`, `nb32`, `nb33`, `ne31`, `ne32`, `ne33` through. The
kernel body just never offset by `nb32`:

```cuda
// before:
const half * mask_h = ncols2 == 1 && !mask ? nullptr :
    (const half *) (mask + nb33*(sequence % ne33));
// after (gated by GGML_CUDA_CRISPASR_FA_PERHEAD_MASK):
const half * mask_h = ncols2 == 1 && !mask ? nullptr :
    (const half *) (mask + nb33*(sequence % ne33)
                         + nb32*(zt_Q     % ne32));
```

For `ne32 == 1` (broadcast mask — upstream's only supported case)
`zt_Q % 1 == 0` and the new term is zero, so default-OFF builds
are bit-identical to upstream.

**Lesson**: when sizing a kernel patch, grep for the strides
you'll need in the kernel signature first. If they're already
plumbed but unused (compiler probably already warns), the patch
is a one-liner inside the kernel body; if they aren't, you have
to widen the launcher too. The original 300-500 LOC estimate
predated reading the signature carefully.

### One CPU FA op per layer forces ≈2 scheduler splits, not 1

Sched-debug A/B on the parakeet short clip (3 chunks, 24 layers,
q8_0, GGML_SCHED_DEBUG=2):

| | total splits | CPU splits | CUDA0 splits | `FLASH_ATTN_EXT` total |
|---|---:|---:|---:|---:|
| FA on CPU (pre-patch) | 147 | 72 | 75 | 72 |
| FA on CUDA0 (post-patch) | 3 | 0 | 3 | 72 (all CUDA0) |

Pre-patch: 24 FA ops/chunk × 3 chunks = 72 CPU FLASH_ATTN nodes.
The *splits* they force are ~49 per chunk (147 / 3), not 24 —
each CPU FA breaks the graph into a CPU split for the FA itself
plus a CUDA0 split for the next layer's GPU work, and the inputs
to the FA (KQV proj outputs straddling the boundary) often spawn
their own splits. Post-patch: a single CUDA0 split per chunk
swallows the whole encoder pass.

**Lesson**: when planning a "move op X from CPU to CUDA" patch,
the wallclock win comes from eliminating *the splits the CPU op
forced around itself*, not just from running op X faster. Count
splits, not ops, when estimating impact. The split-count
difference (147 → 3) is also the cleanest correctness signal —
if it doesn't drop near 1-per-chunk after the patch, the
dispatch gate didn't relax the way you thought it did.

### Per-tile mask offset is only correct for ncols2 == 1

The clean 4-LOC mask offset above is mathematically correct when
`ncols2 == 1` — one Q head per MMA tile. For `ncols2 > 1` (the
GQA-folded path: `gqa_ratio > 1` and the launcher chose to fold
multiple Q heads into one tile), the same expression reads
head `zt_Q`'s mask and silently broadcasts it across the
`ncols2` heads in the tile — **wrong outputs, no crash**.

The fix-of-the-fix is a hard gate at dispatch (in `fattn.cu`):
```cpp
const bool mask_is_per_head = (mask && mask->ne[2] != 1);
if (mask_is_per_head && gqa_ratio > 1) {
    return BEST_FATTN_KERNEL_NONE;  // CPU fallback, no regression
}
```

No current CrispASR model has both per-head masks *and*
`gqa_ratio > 1` (parakeet / canary / FastConformer-CTC are all
`gqa_ratio == 1`), so this gate is a no-op for the workloads we
care about. But if a GQA-conformer ever lands, the gate prevents
silent corruption — caller falls back to CPU FA, == upstream
pre-patch behaviour.

**Lesson**: when you patch a per-tile kernel offset under a
broadcast assumption, audit the broadcast cases (`ncols2` here)
and add a dispatch-level gate for the cases your offset doesn't
cover. Silently-wrong-result kernels are the worst class of bug;
crashing is a feature, not a bug, in patches gated default-OFF.

### WDDM warmup doesn't survive a Python process boundary

The published `dll-postsiglu` warmup protocol —
`probe_postsiglu_leak.py <dll> 200` then
`benchmark_asr_engines.py …` — keeps the A1000 warm only if the
two scripts run back-to-back **with a previously hot GPU**. In a
fresh shell on a cold A1000, the second script's Python startup
+ model mmap + JIT prewarm pass (~3-5 s) is enough idle time for
WDDM to drop the GPU from P0 back to P8. Our FA per-head A/B
captured both runs at P8 / 315 MHz the whole time, suppressing
the expected 10-15 % wallclock win to 2 %.

The clean fix is a single-process warmup driver: put the
keepalive loop and the bench in **one** process so there's no
idle gap between them. The probe-then-bench pattern works only
when the GPU is *already* warm (e.g. when an earlier bench in
the same shell session just finished pushing work). Documented
above at "WDDM idle-clock-state hysteresis on consumer/laptop
NVIDIA SKUs (May 2026)".

**Lesson**: a warmup procedure that crosses a process boundary
on Windows + laptop NVIDIA is racing the WDDM idle timer. Treat
the cross-process probe-then-bench pattern as "works iff GPU was
already warm", and use a single-process keepalive when starting
cold.

---

## TDT single-pass over a full long utterance is numerically fragile to codec-level audio noise (issue #89, 2026-05-24)

Reporter (lenhone) ran the canonical pipeline on a 60 s clip of a YouTube podcast:

```
yt-dlp -x --audio-format wav <url>
ffmpeg -i ... -ar 16000 -ac 1 -t 60 -c:a pcm_s16le yt60.wav
crispasr -m parakeet-tdt-0.6b-ja.gguf -l ja -f yt60.wav -osrt
```

→ output stops at 00:00:20.080. Same binary on our internal cached WAV of the same 60 s of audio produced 99.5 % coverage. The two WAVs:

- identical duration (60.000 s), 16 kHz mono pcm_s16le
- **0.9977 zero-lag waveform correlation**
- ~0.003 RMS difference in normalised [-1, 1] units (≈0.3 % RMS)
- per-second RMS ratio within ~3 % everywhere

Perceptually indistinguishable. The only difference is the codec round-trip: ours had been YouTube → MP3 (saved on the VPS) → WAV, lenhone's was YouTube → Opus/WebM → WAV (a fresh `yt-dlp -x --audio-format wav`).

### Three things that aren't the cause (ruled out by the diff harness against NeMo)

Running `tools/dump_reference.py --backend parakeet --audio <bad WAV>` and `crispasr-diff parakeet …` showed **every** intermediate matched NeMo bit-for-bit on the bad audio:

```
[PASS] mel_spectrogram     cos_min=1.000000  max_abs=3.00e-04
[PASS] pre_encode_output   cos_min=0.999999  max_abs=1.04e-02
[PASS] encoder_layer_0..22 cos_min≈1.000000  max_abs ≤ 3.3e-02
[PASS] encoder_output      cos_min=0.999995  max_abs=2.14e-03
```

So:

1. **Not our mel preprocessing.** Per-feature z-norm on our side gives the same mel as NeMo's `AudioToMelSpectrogramPreprocessor`.
2. **Not our encoder port.** Every conformer layer through layer 23 matches NeMo at cos ≥ 0.999999.
3. **Not our TDT decoder logic.** NeMo's stock `nvidia/parakeet-tdt_ctc-0.6b-ja` via `model.transcribe()` produces 47 chars / stops at ~20 s on the same bad audio. Same model weights, same call shape, same collapse pattern.

### What it actually is — full-utterance bidirectional attention amplifies the codec noise

Per-frame TDT trace on the bad audio:

| frame range (~time) | LOCAL (bad audio) | VPS (good audio) |
|---|---:|---:|
| frames 0-50 (0-4 s) | 12 blank / 21 steps | 10 / 44 |
| frames 50-100 (4-8 s) | 12 / 12 (all blank) | 1 / 19 |
| frames 100-150 (8-12 s) | 13 / 13 (all blank) | 2 / 17 |
| frames 200-250 (16-20 s) | 0 / 20 (all real) | 0 / 18 |
| frames 250-750 (20-60 s) | **13 / 13, 13 / 13, … forever** | 1-6 / 15 each window |

Encoder output stats on the same audio pair (no per-band z-norm shenanigans — same global per-feature z-norm, same encoder weights):

| | bad audio | good audio |
|---|---:|---:|
| `enc.std()` over 750 frames | **0.2069** | **0.2415** (+14 %) |
| TDT tokens emitted | 42 | 211 |

A **0.3 % RMS** audio diff is amplified by full-utterance bidirectional attention into a **14 %** shift in encoder activation magnitude, which is enough to flip the TDT joint network's argmax from real-token to blank past frame ~250. Once blank wins, the predictor doesn't advance, so the next frame's joint sees the same predictor output + a similar encoder output → blank wins again. The decoder stays in that regime for the rest of the utterance.

### Why the streamed path (overlapping 8 s encoder chunks + concat + single TDT decode) dodges this

Two reasons:

1. **Each encoder forward sees only 8 s of bidirectional attention context**, so the attention can't accumulate noise across the whole utterance. The window is short enough that the codec-level perturbation doesn't get amplified into a 14 % activation-magnitude shift.
2. **Global z-norm computed across the whole audio** keeps the input distribution to each encoder chunk consistent with what the encoder weights were trained on. The chunk doesn't see "just this 8 s normalised by its own stats" — it sees "this 8 s normalised by the same per-feature mean/std the trained model expects."

The decoder still sees one contiguous encoder output (concatenated from all chunks, with overlap-skip), so the predictor LSTM doesn't cold-start mid-utterance.

### Lessons

**Lesson 1: when the same binary on two perceptually-identical WAVs produces wildly different output, the problem is model fragility, not implementation.** A 0.3 % RMS audio diff should not be detectable through 24 conformer layers + a TDT joint, but it is, because (a) bidirectional attention has unbounded receptive field which lets numerical instabilities accumulate, and (b) the TDT blank-vs-token decision is a knife-edge argmax that small joint-logit shifts can flip.

**Lesson 2: the diff harness is the right first move when an end-to-end-symptom can be characterised as "two outputs diverge from a known-good reference."** It nailed "encoder is fine, problem is elsewhere" in ~10 minutes of capture + crispasr-diff, replacing what would otherwise have been hours of speculation about z-norm bugs, mel filterbank precision, BN folding, etc.

**Lesson 3: upstream `model.transcribe()` is not necessarily robust on long audio either.** NeMo ships `BatchedFrameASRTDT` / `FrameBatchChunkedCTC` / `get_buffered_pred_feat_rnnt` in `streaming_utils.py` precisely because the default `transcribe()` is single-pass and falls into the same trap. "Stock upstream behaviour" is not the same as "stock upstream behaviour is correct." When the diff harness shows we match upstream bit-for-bit, the next question is whether *upstream* is correct on the input.

**Lesson 4: "byte-identical streamed = single-pass on the test data" is not the same as "streamed = single-pass on all data."** Our 2026-05-23 PERFORMANCE.md "Robustness validation" table claimed streamed and single-pass produce byte-identical output across every chunk/overlap config. That was true on the cached audio we tested — both paths land in the stable regime there. On lenhone's audio they diverge by 200+ characters because single-pass is in the unstable regime and streamed isn't. Whenever a claim is "X always equals Y," look for the input axis we haven't varied yet.

### Cross-refs

- `33f9a162` — flip `CRISPASR_PARAKEET_STREAM_THRESHOLD` default 60 → 0 (always streamed); single-pass becomes an opt-in escape hatch
- HISTORY 2026-05-24 "Issue #89 reopened — parakeet streamed-encode is now the default for all audio"
- PERFORMANCE.md "Multi-backend long-form Japanese — 120 s sweep" (voxtral/cohere/canary hit the same class of issue; PLAN #114 tracks the per-backend follow-up)
- `[[feedback_methodology]]` — the diff-harness stage-by-stage protocol; this is exactly the use case it was built for

---

## Long-form ASR has three distinct failure classes, not one (2026-05-25, generalising issue #89)

Follow-up to the issue #89 lesson above. The 120 s multi-backend sweep on the same audio (`PERFORMANCE.md` "Multi-backend long-form Japanese — 120 s sweep") confirmed that parakeet's TDT-single-pass-collapse is a member of a broader class. The fix shape differs by class, and applying the wrong fix to the wrong class is wasted effort. The three classes:

### Class A — Conformer encoder long-attention amplification

**Affects:** parakeet (TDT / RNNT / hybrid), canary (multi-task AED with FastConformer encoder), cohere-transcribe (Conformer), fastconformer-ctc (technically yes but CTC head is robust enough that it doesn't surface as a failure).

**Mechanism:** bidirectional self-attention over the full utterance has unbounded receptive field. The pre-norm path through 24 Conformer blocks accumulates tiny input perturbations (codec-level audio differences, ~0.3 % RMS in normalised units) into 10-15 % shifts in encoder output activation magnitude. Downstream, that's enough to flip knife-edge decisions (TDT joint blank-vs-token argmax; AED decoder's `<eos>` probability; the joint network's logit ordering more generally).

**Fix shape:** chunked encoder + globally-normalised input + single decode pass. Specifically:
- **Compute mel with global per-feature z-norm** over the FULL audio (not per-chunk z-norm — that re-normalises away the codec stability margin the model was trained against and makes chunks inconsistent).
- **Encode in overlapping ≤10 s windows** (we use 8 s + 2 s overlap). The attention receptive field is now bounded → no accumulating amplification.
- **Concatenate encoder outputs**, skipping the overlap region of each chunk except the first.
- **Decode in one pass** over the concatenated encoder output. Single-pass decode means the predictor LSTM / AED autoregressive state stays continuous across what used to be chunk boundaries — no cold-start mid-utterance.

This is exactly the shape NeMo ships as `BatchedFrameASRTDT` / `BatchedFrameASRRNNT` / `FrameBatchChunkedCTC` / `FrameBatchMultiTaskAED` in `streaming_utils.py`. **Note: NeMo's stock `model.transcribe()` does NOT use this** — it does single-pass. Users have to opt in to the streaming utility explicitly. Ours is the default after `33f9a162`.

### Class B — LLM autoregressive decoder loses track at chunk boundaries

**Affects:** voxtral-mini-3b (Mistral LLM), qwen3-asr (Qwen3 LLM), granite-speech (IBM Granite LLM), mimo-asr (Xiaomi LLM).

**Mechanism:** the AR decoder is given each audio chunk independently with a fresh prompt; its conditioning at the chunk boundary either misfires (drops the boundary tokens), hits `max_new_tokens` before catching up to the audio, or hallucinates a continuation that doesn't connect to the next chunk. Symptom: missing middle chunks in long audio (voxtral 120 s drops ~80 s in the middle).

**Fix shape:** chunk with **explicit overlap (~2 s)** + **LCS dedup** on overlapping token sequences. CrispASR already has `core_lcs::merge_overlapping_hypotheses` from PLAN #80c; the missing piece is wiring it as a default for LLM-AR backends, not as an `--lcs-dedup on` flag the user has to know to flip. Mistral's voxtral reference HuggingFace integration does exactly this.

Notable: this is NOT the same fix as Class A. Class A's "concatenate encoder outputs and single-decode" doesn't apply to LLM-AR backends because the encoder isn't the unstable component — the decoder is, and the decoder is the autoregressive LLM which can't be made stateless across chunks without losing context.

### Class C — Multi-task AED / language-prompt wiring (canary-specific symptom, plausibly generalisable)

**Affects:** canary-1b-v2 (NeMo multi-task AED model trained for ASR + translation + LID). Possibly others with explicit task tokens (kyutai-stt has language prefix tokens but is streaming-native).

**Mechanism:** the AED decoder needs `<lang>` / `<task>` prompt tokens at the start of each decoding pass. Either (a) the wiring isn't passing the user's `-l ja` through to the decoder prompt at all, or (b) it's passing it once at t=0 but not re-injecting it at chunk boundaries when a Class A streamed path is added later. Symptom on canary-1b-v2 + lenhone JA 120 s: hallucinates English `"I am not aware of anything"` in a loop.

**Fix shape:** verify the language-prompt path at short-audio first (does canary `-l ja` work on a 10 s JA clip?), fix that bug independently, then add the Class A streamed-encode port on top.

### Diagnostic short-circuits

When a long-audio failure is reported:

1. **Same model, two perceptually-identical inputs, different outputs** → Class A. Run the diff harness against upstream on the bad input to confirm it's not us — if our encoder matches upstream bit-for-bit, the issue is model-level, not implementation.
2. **Output covers the start and end of the clip but drops a middle chunk** → Class B. Look at the chunker output and the decoder's `<eos>` probability at the chunk boundary.
3. **Output is the wrong language entirely / hallucinates a stock phrase** → Class C. Test at short audio first; long-audio is downstream.

### Anti-pattern: blanket-VAD

Tempting to make `--vad` the default for everyone past 30 s as a one-size-fits-all. Don't:

- VAD trims leading/trailing silence per segment → coverage on continuous narration speech drops ~99 % → ~93 % even on audio where everything else works. Wrong default for narration / podcast / lecture content (which is exactly what long-form users care about).
- VAD produces per-utterance SRT entries, not paragraph-level. Worse for continuous-transcription users.
- VAD is a workaround for an encoder/decoder bug, not a fix. It just makes the encoder see ≤ ~3 s slices so it can't fall into the unstable regime. Real fix per backend class.

VAD-default is the right answer only when the released model genuinely isn't designed for long inputs (cohere is in this bucket — its hosted product runs VAD on the server side; our open-weights release should match that behaviour).

### Cross-refs

- HISTORY 2026-05-25 "Long-form ASR — cross-backend survey and per-backend roadmap (PLAN #114)" — empirical table + roadmap
- PLAN #114 — per-backend status table + prioritised fix order
- PERFORMANCE.md "Multi-backend long-form Japanese — 120 s sweep" + the in-progress 60/120/300/600 s × all-backends matrix
- HISTORY 2026-05-24 "Issue #89 reopened" — the parakeet Class A fix that started this
- `core_lcs::merge_overlapping_hypotheses` (PLAN #80c) — the existing LCS-dedup helper for the Class B fix

---

## Always rebuild the box-under-test before benchmarking (2026-05-25)

The 2026-05-25 morning "Cross-length × cross-backend matrix" landed in PERFORMANCE.md with v1 numbers showing voxtral at 9 % coverage on a 600 s clip and cohere at 62 %. Those numbers were *real* — the matrix script ran cleanly — but they measured the **wrong binary**. The VPS had `bd8b98cf` (May 24) on disk, and the per-backend opt-out fixes for cohere (`dc2295b2`), gemma4-e2b / glm-asr (`46f6848d`), kyutai-stt (`eaee2319`), and voxtral (`6fef8790`) had landed locally that same morning — but the VPS hadn't been rebuilt. Matrix v2 (post-opt-out rebuild to `13059e0c`) showed voxtral and cohere at **96-100 %** at every length.

The whole "voxtral drops the middle of 80 s of audio at 120 s" diagnosis in HISTORY 2026-05-25 morning + LEARNINGS "Long-form ASR has three distinct failure classes" Class B was *correct in shape* but vastly overstated in *current severity* — voxtral default chunking with the opt-out applied is fine. The matrix v1 numbers measured the pre-opt-out behavior, which the parallel #114 worker had already fixed by the time we wrote them up.

### How to not do this again

1. **`git ls-remote --heads origin main` + `git rev-parse origin/main` on the VPS *before* a matrix run.** Compare to local `main`. If they don't match, push first.
2. **Rebuild the binary on the VPS before any benchmark run.** A skipped `cmake --build` because "the binary is already there" silently locks the matrix to whatever state the disk happens to be in.
3. **Record the binary's `--version` line at the top of every matrix CSV.** The harness already writes `wall_s` per cell — also write the git sha. Saves the "wait, which commit was this measuring?" forensic step later.
4. **When the matrix shows a backend doing wildly worse than upstream's reported behaviour, suspect a stale binary before suspecting a real model regression.** Voxtral at 9 % on a 600 s clip should have been a "huh, that's surprising — let me check the build" signal, not a "voxtral is broken" signal.

The lesson generalises beyond CrispASR — it's the "two-tab development" failure mode: a passing local test against an unbuilt remote binary. Same family as cache-poisoning, ccache-hit-on-stale-object, "I'm sure I rebuilt that," etc.

### Side benefit: matrix v1 + v2 side by side

PERFORMANCE.md now keeps both matrices so the reader can see the cost of the missing opt-out — the same models, the same audio, the same matrix script, just an `kBlocked` flag in `crispasr_chunk_context_gate.h` between them. It's a clean visualisation of why the per-backend long-audio story isn't "model architecture" but "external overlap-save context wrapping that the LLM-decoder backends couldn't trim back from correctly."

### Cross-refs

- `[[feedback_check_performance_learnings]]` — the broader "check what's already there before benchmarking" rule
- HISTORY 2026-05-25 "PLAN #114 voxtral streamed + matrix re-interpretation"
- `examples/cli/crispasr_chunk_context_gate.h` `kBlocked` list — the mechanism behind the opt-out fixes

---

## Distinguishing "slow run" from "hung run" — CPU time ≪ wall time is the signal (2026-05-25)

A voxtral streamed 600 s test on Apple Silicon Metal ran for 2 h 10 min wall time without producing a JSON output. Easy default assumption from a session-spanning "watching it grind" perspective: the model is just slow. The actual state was a hang — and the diagnosis takes one `ps` call.

### Signal

`ps -p $PID -o pid,etime,time,pcpu,state,wchan`

| field | hung run (this incident) | active run (e.g. 300 s test on same box earlier) |
|---|---|---|
| `etime` | 02:10:25 | 16:34 |
| `time` (cumulative CPU) | **00:20.31** | 14:28 |
| `pcpu` (current) | 0.0 % | 70-95 % |
| `state` | `SN` (sleeping + nice) | `R` (running) |

A 600 s audio with a few thousand encoder + decoder forward passes should accumulate **tens of minutes of CPU time** within tens of minutes of wall time. **20 seconds of CPU in 2 hours of wall** is two-three orders of magnitude below progress; the process is sleeping on something, not computing.

Confirm with `sample $PID 1`: macOS will show what the call stack is parked on. In this case all 850 samples were in `main` — the process never reached the encoder forward, parked somewhere in startup / Metal allocator / first big tensor alloc.

### Cause in this incident

`vm_stat` showed **80 MB free out of 16 GB** when the hang was diagnosed. Four+ parallel Claude agents, WindowServer at 42 % CPU, the voxtral-mini-3B Q4_K binary (2.8 GB resident) + the streamed path's growing context state, all sharing one 16 GB box. Metal's command-buffer allocator stalls when there's no headroom; the process becomes sleep-bound on first allocation and never reaches a working state.

### Rules

- **Before assuming "slow," check `time` vs `etime`.** A 100:1 wall-to-CPU ratio is a hung process, not a slow one. Treat it that way: kill, free RAM, retry.
- **Free RAM threshold for LLM-AR backends on this box: ~4 GB.** Below that, voxtral and similar Mistral/Qwen 3 B-class LLM-AR pipelines stall on the Metal allocator. Stop other agents' heavy work or wait, don't try to push through.
- **`sample $PID 1`** is the right next step after the `time/etime` signal — confirms whether the parked region is startup (allocator), forward pass (compute), or shutdown (cleanup).
- **`ps STAT=SN`** alone is not the signal — many short-CPU-bound steps interleave with sleeping. The `time` accumulator is.

### Cross-refs

- `[[feedback_storage_paths]]` 2026-05-25 update — main volume at 99-100 % full pushed all caches into kernel-thrash territory, contributing to the headroom problem
- HISTORY 2026-05-25 (late) "PLAN #114 close-out" for the full post-mortem

---

## Kaggle as a batch-rebake target: seven fragilities the script has to work around (2026-05-25)

Bringing up the multi-backend rebake on `chr1str/crispasr-auto-rebake-refs` took 7 sequential surgical fixes across one afternoon, each surfacing a different Kaggle-side or script-side fragility. Six of the seven would NOT have been visible without running the kernel end-to-end — they're masked by partial successes in smaller notebooks (fusion-ab, qwen3-export). Capturing them here so the next person bringing up a Kaggle batch job over `manifest.json`-driven backends doesn't rediscover them one at a time.

### 1. `kaggle kernels output|files|logs` returns nothing mid-run

You **cannot peek live state programmatically**. All three endpoints return empty until the kernel terminates. The browser UI's websocket log is the only live view, and it stops updating when the parent stdout buffer fills. So if you want to know what a kernel is doing right now, you need:

- A heartbeat your script emits at a controlled cadence (e.g. every 30 s)
- A side-channel that ships those heartbeats off the kernel ASAP — public HF dataset push works (see point 7 below); Kaggle Secrets-gated upload doesn't

Don't rely on `kaggle kernels output` for in-flight diagnostics. It's a *post-mortem* endpoint.

### 2. `subprocess.check_call` on cmake/ninja produces invisible-progress builds

Child stdout flows through the parent's Python stdout, which gets line-buffered (Python) → block-buffered (Kaggle's papermill log capture). A healthy build at 200 / 360 objects can sit silent in the UI for 90+ minutes between heartbeats while ninja is just chewing through templates. `chr1str/qwen3-export` solved this with a Popen + line-reader:

```python
proc = subprocess.Popen(cmd, shell=True, bufsize=1, text=True,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
for line in proc.stdout:
    print(line.rstrip(), flush=True)  # explicit per-line flush
proc.wait()
```

The explicit per-line `flush=True` is what kicks Kaggle's log buffer. Without it, child output disappears for tens of minutes at a time even with `os.environ["PYTHONUNBUFFERED"] = "1"` set on the parent (because parent's `PYTHONUNBUFFERED` doesn't propagate to a shell-invoked child's pipe). Pair it with `stdbuf -oL -eL <cmd>` on the child for line-buffered output where the child's own stdio defaults are block-buffered (ninja does this when stdout isn't a tty).

### 3. Heartbeats need actual content to be useful

A heartbeat that just prints `elapsed_in_block_s=210.0` every 30 s is noise — you still can't distinguish a healthy slow build from a hung process. The fix: parse the child output line stream and stash the most recent ninja `[X/N]` + TU name in a module-level dict, then include those in the heartbeat:

```python
[step 396.5s] cmake.build.heartbeat  {'elapsed_in_block_s': 240.0,
                                       'ninja': '219/360',
                                       'tu': 'kokoro.cpp',
                                       'lines': 668}
```

A real hang then surfaces as ninja staying frozen at the same `[X/N]` across multiple heartbeats; slow-but-progressing shows monotonic advancement. The signal-to-noise ratio jumps an order of magnitude.

### 4. Kaggle's 20 GB disk fills fast on multi-model rebake without per-backend cleanup

A 13-backend rebake with NeMo + transformers downloads per entry hits the cap around backend #5 if you let `/kaggle/working/hf_cache/` accumulate. Each backend's source weights (~2-3 GB for parakeet-1.1b-class NeMo models, ~5 GB for voxtral-3b, ~4 GB for mimo-asr) sit there forever otherwise. Add the ~5 GB ccache and the ~1 GB build dir, and you're at 14 / 20 GB before backend 4 starts downloading.

`run_rebake()` needs a `finally:` block that rmtrees the HF + torch caches after every backend (success or failure), then logs the freed disk. The trade-off is cache invalidation between backends, but in practice cache hits between different model families are near-zero anyway. The actual measurement on v11 with the cleanup patch: `free_gb_after=19.22` stayed flat across all 23 backends — disk pressure completely controlled.

### 5. Kaggle Secrets API flakes on batch-triggered kernels

`UserSecretsClient.get_secret('HF_TOKEN')` returns `ConnectionError: Connection error trying to communicate with service.` even when:

- `KAGGLE_USER_SECRETS_TOKEN` JWT is present in env (Attach toggle correct)
- The Kaggle account dashboard shows the secret
- A second-attempt retry works fine
- The same secret reads fine in interactive kernel mode

The flake is intermittent — `UserSecretsClient` is hitting the kaggle-secrets-api service which has a non-trivial 5xx rate for kernels triggered via `kaggle kernels push` (batch / papermill execution). Retry with 5 s backoff x3 sometimes resolves; sometimes the service is unreachable for the whole run.

The script comment recommending "Add-ons → Variables" as the fallback was wrong: **Kaggle has no Variables UI feature**. Only Secrets, Internet, Accelerator, and Data Sources are in Add-ons. The real fallback is a private Kaggle Dataset:

1. `kaggle datasets create -p .` a private dataset containing one file with the token
2. Mount it via `kernel-metadata.json:dataset_sources: ["<owner>/<slug>"]`
3. Script reads it from `/kaggle/input/<slug>/<file>` as a filesystem path — bypasses the Secrets API entirely

This made the difference between v12 (auth failed, no upload) and v13 (auth resolved from dataset, upload landed).

### 6. All-or-nothing upload gates suppress legitimate partial wins

Defensive code at the upload step said `if any(not r.get("ok") for r in RESULTS_DATA): raise SystemExit("refusing to upload")`. The intent was to prevent accidentally clobbering known-good refs with a buggy rebake. The effect, once the manifest grew to include 14 backends with known issues (missing pip deps, manifest gaps, per-backend bugs in upstream Python reference modules), was that **no run could ever pass** — every run reaches the gate with at least one failure.

Partial upload is safer than it looks because failed entries never write to `REBAKE_STAGE` — `dump_reference.py` exits non-zero before the GGUF-write step. So `upload_folder(REBAKE_STAGE)` inherently only uploads successful entries. The right shape:

```python
successful = [r for r in RESULTS_DATA if r.get("ok")]
if not successful:
    raise SystemExit("zero refs produced; nothing to upload")
# Log failed entries in the commit message for traceability:
commit_message = f"rebake {len(successful)}/{len(RESULTS_DATA)} — ok: {', '.join(succ)}"
```

The fixtures repo is additive; a partial upload can't regress a passing entry.

### 7. `kaggle kernels output` doesn't return artifacts from ERROR'd kernels

Even after v11's rebake successfully wrote 9 ref archives to `/kaggle/working/rebake-stage/`, `kaggle kernels output -p ...` only returned `.ccache/` and the log file — the staging dir was unreachable. Reason: the kernel ended in ERROR (papermill autosave OOM on bloated notebook output, *after* the rebake itself finished cleanly), and the output endpoint only ships artifacts from runs that completed cleanly.

Implication: if you stage outputs locally and rely on `fetch-and-upload.sh` to ship them off Kaggle later, **the kernel must end in SUCCESS** for the artifacts to be retrievable. Upload directly from within the kernel — don't stage-and-fetch as a two-step.

The cleanest end-to-end shape is: rebake → upload immediately from the same Python process → only THEN sys.exit(n_fail) for CI status reporting. Anything that runs after `sys.exit(n_fail)` (papermill's notebook autosave, nbconvert, etc.) is a side-channel that may or may not survive.

### Cross-refs

- HISTORY 2026-05-25 (latest) "Kaggle rebake pipeline wired end-to-end" for the chronological summary
- `tools/kaggle/crispasr-regression.py` commits `1b62776e` (heartbeat) → `8cf7e931` (KeyError fix + sort) → `eba52bac` (Popen streamer) → `0f4de5b9` (cache cleanup) → `2ff4f1ba` (Dataset auth) → `c11e0648` (partial upload)
- `chr1str/qwen3-export` Kaggle notebook for the canonical Popen+iter pattern that the fix lifts
- The private Kaggle Dataset `chr1str/crispasr-hf-token` is the auth-fallback infrastructure

---

## Cross-backend bug-sweep methodology — pair the cap survey with an empirical A/B (2026-05-26)

When a per-backend fix lands (e.g. PR #124's cohere opt-out from external overlap-save context), the obvious next question is *which other backends are in the same bucket?* A pure capability-table survey gets you to the candidate list — for the overlap-save bug, *14 of 22 ASR backends* lacked both `CAP_UNBOUNDED_INPUT` and `CAP_INTERNAL_CHUNKING` so all were on-paper at-risk — but only the empirical A/B distinguishes "lacks the cap" from "actually exhibits the failure mode".

The harness shape that worked (`tools/check-overlap-save-bug.sh`): for each at-risk backend, run the *same* long audio twice — once with the default knob, once with the knob disabled — and emit a single table row with `last_ts`, `n_segs`, `n_chars` for each arm + a verdict.

```
BACKEND     DEF_LAST DEF_SEGS DEF_CHRS NO_LAST NO_SEGS NO_CHRS VERDICT
voxtral     300      2        284      300     11      1294    *** SUSPECTED BUG ***
voxtral4b   300      11       1527     300     11      1305    OK
```

This catches three distinct symptom shapes that all come from the same root cause:

1. **Truncation** (cohere, voxtral, qwen3): output stops early, default has fewer chars than nooverlap.
2. **LLM runaway** (gemma4-e2b, glm-asr, kyutai-stt): default times out past wallclock budget, nooverlap finishes.
3. **Clean** (funasr, moonshine, sensevoice, vibevoice, voxtral4b): default ≈ nooverlap, no bug.

Five new offenders found and fixed in one session because the harness made the bug-shape comparable across backends. A two-phase sweep (5 min audio first, then 90 s re-run for the inconclusive-because-too-slow backends) is the right pattern for an M1 box — pure-Python LLM ASR backends like granite-4.1-2b time out at 15 min on a 5 min clip but finish in 5 min on a 90 s clip.

**Reuse pattern.** When the next per-backend fix lands, copy `tools/check-overlap-save-bug.sh`, swap the knob, swap the model list, run. The cost is one afternoon; the upside is fixing all sibling backends in one session instead of dragging the same bug class out across N follow-up issues.

### Cross-refs

- HISTORY 2026-05-25 / -26 "Overlap-save bug sweep — five new opt-outs + a reusable A/B harness" for the chronological summary
- `examples/cli/crispasr_chunk_context_gate.h` for the opt-out registry that the harness drives
- `tests/test-issue-114-chunk-context-gate.cpp` for the unit test that pins the opt-out list

---

## ggml scheduler tightened cross-backend tensor resolution between §56 and 2026-05-26 (PLAN #115)

mimo-asr's working configuration in HISTORY §56 was CPU-resident weights + Metal compute backend. The implicit assumption was that the ggml scheduler would auto-copy a CPU-buffer tensor to GPU on first read. That implicit copy no longer happens — `ggml_metal_buffer_get_id: error: tensor 'llm.embed.weight' buffer is nil` is the symptom when current ggml encounters a compute op that touches a tensor whose backend buffer doesn't match the op's backend.

Implications for porting any backend that mixes CPU + GPU buffers:

1. **Per-tensor backend tagging is now mandatory.** The graph builder has to call `ggml_backend_sched_set_tensor_backend` (or use `core_gguf::load_weights_split` with a proper `LayerSplitConfig`) so sched knows where each tensor lives and can insert the copy nodes. Hand-built graphs that just `ggml_get_rows(embed_weight_on_cpu, ...)` from inside a Metal-target graph will silently fail at runtime.
2. **The historical PLAN #56 → §56 chronology is misleading.** The §56 working ref doesn't run on current ggml without modification; the regression timing makes it look like mimo-specific code broke, but the actual change was inside the ggml scheduler. Useful when bisecting: try the failing config against the §56 commit's ggml submodule, not just the §56 mimo source — that isolates "mimo regression" from "ggml regression".
3. **The cheap workaround is "ignore use_gpu and force everything to CPU".** What PLAN #115 option A shipped. It's slow (pure CPU LLM = 297 s for 11 s JFK on M1) but correct. The proper fix (option C) is to either tag the embed weight for the GPU backend explicitly + insert the copy node, OR rebuild the prefill graph so its inputs all live where the compute happens.
4. **PLAN #72-shaped "perf wins by moving weights to GPU" are now fragile.** The same move that gave gemma4-e2b a 2.2× speedup silently broke mimo-asr's prefill graph emission. Any backend that ported in pre-2026-05-26-ggml and gets a "move weights to GPU" perf pass needs a roundtrip transcription test on the actual deployment platform before the perf claim ships.

### 2026-06-02 update — the real bug was NOT the prefill graph (Kaggle P100 diff)

Points 1–3 above assumed the fix was per-tensor tagging *in the prefill
graph*. A Kaggle CUDA stage-diff (`tools/kaggle/mimo-asr-gpu-diff/`, the
funasr-cuda-debug pattern) proved otherwise, and is the canonical example of
why you reproduce on the actual GPU before theorising:

- A `MIMO_ASR_DUMP_STAGES=1` per-stage dump (mirrors `FUNASR_DUMP_STAGES`)
  showed the **GPU prefill is correct** — all five stages match CPU with no
  NaN/Inf when `CRISPASR_MIMO_FORCE_GPU=1` loads the weights GPU-resident.
- The run instead **segfaulted in the *decode* step** (`rc=-11`). gdb:
  `dequantize_row_q4_K` → `ggml_backend_cpu_graph_compute` →
  `mimo_asr_transcribe_impl`.

Root cause: with **GPU-resident quantized weights**, a multi-backend sched
`[GPU, CPU]` will happily place *some* op on the **CPU** backend (its cost
heuristic), and the CPU dequant kernel then reads the **GPU CUDA pointer as
host memory** → SIGSEGV. It's not about copy-node insertion at all — it's
about *never letting the CPU backend touch a GPU-resident weight*.

**Rule: with GPU-resident *quantized* weights, no op may be CPU-routed — a
CPU-routed op dereferences device memory.** The tempting fix — build the
sched GPU-only — does NOT work: `ggml_backend_sched_new` *requires* a CPU
backend as the mandatory last/fallback entry and `abort()`s on a `{GPU}`-only
list (verified, run 3 `3ef9f87e` reverted). So the real fix is to make the
*specific* CPU-routed op CUDA-runnable (or keep its weight CPU-accessible),
not to drop the CPU backend.

**The actual op (mimo-asr): `get_rows` on a Q4_K embedding table.** CUDA's
`GET_ROWS` supports_op (ggml-cuda.cu:5004) handles F16/F32/BF16/I32 and the
*legacy* quants Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 — **but NOT the k-quants (Q4_K
etc.)**. mimo's `llm.embed.weight` + `audio.emb.*` are Q4_K, so their
`get_rows` is CUDA-unsupported → CPU-routed → `dequantize_row_q4_K` on a GPU
pointer → SIGSEGV. `GGML_SCHED_DEBUG=2` is a *no-op in Release* (don't waste a
kernel round on it) — read the CUDA `supports_op` + the gguf tensor dtypes
(`gguf.GGUFReader`) instead.

**Two fixes, same lesson — token embeddings must stay CUDA-gatherable:**
1. *Converter* (preferred, durable): keep `llm.embed`/`audio.emb` (any
   `get_rows`'d table) at F16 or Q8_0, never a k-quant. Standard llama.cpp
   practice already keeps `token_embd` higher-precision for exactly this.
2. *Runtime* (no re-bake): load just those tables on CPU (`load_weights_split`
   with a name predicate) while matmul weights stay GPU-resident; the sched
   copies the small embed output GPU-ward. mimo's `force_gpu` path does this.

General rule for porting a GPU backend with k-quant weights: anything fed to
`ggml_get_rows` (embeddings, MoE expert gather) must be F16/F32/legacy-quant,
or it silently falls to CPU and dereferences device memory.

### 2026-06-04/05 update — GPU decode fixed via prefill-graph reuse (PLAN #115 option B)

The 2026-06-02 update identified the *prefill* as correct and the *decode step*
as the crash site. Fixing the decode step took **12 Kaggle kernel iterations +
1 RunPod RTX 3090 run** and exhausted the weekly Kaggle GPU quota. The approaches
tried and their failure modes are instructive for any future cross-backend graph work:

| Approach | Kernel | Result |
|----------|--------|--------|
| `ggml_cont` bridge + `set_tensor_backend` on compute node | v6 | `ggml_cuda_cpy: invalid argument` — scheduler tried DeviceToDevice on host memory |
| External mini CPU graph (`ggml_backend_alloc_ctx_tensors` + `graph_compute`) | v7,v8 | SIGSEGV — cross-context tensor references crash the CPU allocator |
| F32 input tensor + host-side dequant via `ggml_get_type_traits()->to_float` | v10 | CPU timeout (wrong `ggml_row_size`), GPU SIGSEGV in `cuMemcpyHtoDAsync` |
| F32 input + `set_tensor_backend` pin to GPU | v11,v12 | GPU `illegal memory access` in `graph_compute` — other tensors misrouted |
| **Reuse prefill graph for T=1 decode** | v13 (RunPod) | **PASS — 2.4× realtime, correct transcript** |

**Key insight: the ggml scheduler handles cross-backend routing correctly for
complex graphs (prefill: audio path + text path + 36L LM) but breaks on minimal
graphs (step: just embed → 36L LM).** The scheduler's backend assignment
heuristic needs enough GPU-anchored ops to "pull" all nodes GPU-ward. A step
graph with only a single F32 input and no weight-referencing embed op doesn't
provide that anchor, even with explicit `set_tensor_backend`.

**The winning fix** (`ec3ba861`): for `gpu_embed_split` mode, decode steps build
a `[9, gs]` single-token text segment (zeroed audio branch) and call
`mimo_asr_run_lm` (the prefill graph). The audio path computes zero
(`speech_active_mask=0`) — negligible overhead for T=1 with gs=4 vs the 36L LM.
CPU path is unchanged (fast cached T=1 step graph with in-graph `get_rows`).

**Performance (RTX 3090, JFK 11s):**
- CPU: 11.5s (1.0× realtime)
- GPU: 4.5s (2.4× realtime) — prefill 75ms, decode 265ms / 26 steps (10 ms/step)

**Rules learned:**
1. `ggml_backend_sched_set_tensor_backend` on *compute nodes* is unreliable for
   cross-backend copy insertion. It works for *input tensors* to control placement
   but the scheduler's copy-node insertion for compute nodes is buggy on small graphs.
2. `ggml_backend_alloc_ctx_tensors` cannot safely reference tensors from a different
   `ggml_context` — the allocator doesn't expect cross-context weight pointers and
   crashes in `graph_compute`.
3. When a graph already works on GPU (like the prefill graph), reuse it for decode
   rather than building a minimal step graph that hits scheduler edge cases. The
   overhead of the unused audio branch (~4 embeddings + 6L tiny transformer) is
   negligible vs 36L × 4096-dim LM layers.
4. **RunPod** is a viable alternative to Kaggle for GPU testing ($0.22/hr RTX 3090,
   ~$0.10 per test run). Script: `tools/runpod-gpu-test.sh`. Needs `pip install
   cmake` on the pod (system cmake 3.22 too old), `nvcc` at `/usr/local/cuda/bin/`.

**GPU is now the default** (`a429bb45`). Override: `CRISPASR_MIMO_FORCE_CPU=1`.

### 2026-06-05 update — k-quant CUDA GET_ROWS fix (the proper ggml fix)

The root cause of the entire PLAN #115 saga was that CUDA GET_ROWS didn't
support k-quants (Q2_K–Q6_K). The `supports_op` list in `ggml-cuda.cu:5004`
only covered F16/F32/BF16/I32 and legacy quants (Q4_0–Q8_0). When a Q4_K
embedding table sat on GPU, `get_rows` was routed to CPU, which then
dereference'd the GPU pointer as host memory → SIGSEGV.

**Fix** (`3bf9a599`): add k-quant support to CUDA GET_ROWS by using the
existing `ggml_get_to_fp32_cuda()` row-dequantize infrastructure from
`convert.cu`. For each selected row, copy the index to host, then launch
the type-appropriate dequantize kernel (e.g. `dequantize_block_q4_K`).
Overhead is negligible for typical embedding lookups (1-4 rows of 4096).

**Validated on RTX 3090 (RunPod):** all weights GPU-resident (no split-load),
2.0× realtime, correct JFK transcript, 26 decode steps clean.

| Approach | Split-load? | Decode ms/step | Realtime | Complexity |
|----------|------------|----------------|----------|------------|
| Option B: prefill-graph reuse | Yes (embed CPU) | 10 ms | 2.4× | Medium — workaround |
| **k-quant GET_ROWS fix** | **No** | 31 ms | 2.0× | **Low — proper ggml fix** |

The per-step gap (10 vs 31 ms) is because the k-quant GET_ROWS copies
indices to host and launches sequential dequant kernels, while the
prefill-graph approach keeps the embed on CPU and lets the scheduler
insert one bulk copy. A fused k-quant GET_ROWS kernel (single launch,
no host round-trip) would close the gap — filed as upstream PR 16.

Both approaches produce identical transcripts. The codebase currently
ships both: the prefill-graph workaround (faster) is used when
`gpu_embed_split` is true, and the k-quant GET_ROWS (simpler) handles
the all-GPU case. A future simplification could drop `gpu_embed_split`
entirely once the fused kernel is optimized.

### Cross-refs

- HISTORY 2026-05-26 "PLAN #115 — mimo-asr M1 Metal silent-empty fix (option A)" for the bisect + fix chronology
- PLAN #115 for option C scope (per-tensor backend tagging in `mimo_asr_build_prefill_graph`)
- [[project_chatterbox_gpu_bug_s3gen]] for a different shape of the same general problem (sched parallel=true fixed it there)
- `src/mimo_asr.cpp` commit `c887881e` for the option A landing
- `src/mimo_asr.cpp` commit `ec3ba861` for the option B (prefill-graph reuse) landing
- `ggml/src/ggml-cuda/getrows.cu` commit `3bf9a599` for the k-quant GET_ROWS fix
- `tools/upstream-prs/16-sched-small-graph-cross-backend.md` for the upstream PR draft
- `tools/runpod-gpu-test.sh` for the RunPod GPU test script

## funasr CUDA !-loop — all-NaN prefill logits (issue #125, §136)

### Bug

funasr (FunASR Nano 2512, SANM encoder + Qwen2-0.6B decoder) transcribes
JFK correctly and byte-identically on CPU and Apple Metal, but on CUDA the
greedy decode degenerates: token id 0 repeats until the stop-loss guard
aborts → empty/garbage `!!!!!` transcript → WER 100%. Confirmed on P100
(sm_60) and Blackwell (sm_120) — architecture-independent within CUDA.

### Investigation (2026-05-31)

1. **Initial hypothesis was wrong.** Commit `868aabdf` blamed F16 weights
   causing `MUL_MAT(F16,F32)` saturation on CUDA and shipped a registry
   default flip F16→Q8_0 + a ggml-cuda patch. Q8_0 model also !-looped on
   Kaggle P100, disproving the hypothesis.

2. **Flash-attn in the encoder was suspected but ruled out.** Added
   `FUNASR_DUMP_STAGES=1` per-stage tensor dump (min/max/mean/L2/NaN/first8)
   for 13 key encoder stages + adaptor + spliced embeddings. Kaggle v3 ran
   three experiments: CUDA FA-on, CUDA FA-off (`FUNASR_NO_FA=1`), CPU
   baseline. Result: **encoder/adaptor stages match CPU on CUDA** to within
   first8 max-diff < 0.05 across all layers. The encoder is fine.

3. **The bug is in the LLM decoder.** ALL 151936 prefill logits are NaN
   on CUDA — both with FA on and FA off. The encoder produces correct
   embeddings, the splice is correct, but after the Qwen2-0.6B LLM prefill
   graph runs on CUDA, every output float is NaN.

4. **Same `kv_self_attn` code works for qwen3-asr on CUDA.** So it's not a
   generic bug in `core_attn::kv_self_attn`. The difference must be
   funasr-specific: model shapes (head_dim=128 with GQA ratio=2), weight
   types (Q8_0), QK norm (Qwen2-0.6B has per-head RMS norm on Q/K with
   128-dim weights), or a subtle graph-construction issue.

5. **KV cache is F16, allocated on CUDA, not zeroed** — standard pattern,
   but uninitialized GPU memory could cause NaN if a graph scheduling race
   reads before write. Under investigation.

### Root cause (confirmed 2026-05-31/06-01, Kaggle v3–v16)

The `ggml_backend_sched` with `[CUDA, CPU]` backends misroutes funasr's
Qwen2-0.6B LLM decoder ops on CUDA. With all weights + KV cache on GPU,
the sched produces **Inf at LLM layer 2** (values diverge wildly from
CPU), which cascades to **all-NaN by layer 3** and poisons the logits.

The exact upstream sched bug is unidentified — it's not specific to the
3-separate-Q/K/V pattern (v15 proved QKV fusion alone does NOT fix it)
and not specific to flash-attn, F16 KV, or any single op. It appears
to be a general `ggml_backend_sched` issue with this particular graph
topology on CUDA. The same `kv_self_attn` code works for qwen3-asr
(identical structure) — the difference is model-specific (Qwen2-0.6B
with head_dim=128, GQA ratio=2, QK-norm), not code-specific.

**Per-layer NaN scan (v11):**
```
llm_layer_0:  0 NaN, 0 Inf  — matches CPU ✓
llm_layer_1:  0 NaN, 0 Inf  — matches CPU ✓
llm_layer_2:  0 NaN, 1 Inf  — max=6973 vs CPU max=124512 ← DIVERGES
llm_layer_3:  ALL NaN (47104/47104)
llm_layer_4+: ALL NaN
```

### Fix (commit `f94fec90`, verified v16)

1. **Weight split** via `load_weights_split`: encoder/adaptor tensors
   (prefix `funasr.*`) → GPU, LLM decoder tensors (`blk.*`, `output.*`)
   → CPU. The sched naturally routes the entire LLM to CPU.
2. **KV cache on CPU** when the split is active (`model.buf_cpu != nullptr`).
   Without this, KV-on-GPU + weights-on-CPU still produces Inf (v11).
3. **QKV fusion** at init: Q(1024,2048)+K(1024,1024)+V(1024,1024) →
   QKV(1024,4096) per layer. Good practice (fewer ops) but does NOT fix
   the NaN on its own (v15 proved this).
4. **KV cache zeroing** on allocation — defense-in-depth.
5. `FUNASR_LLM_GPU=1` overrides to all-GPU for future testing.

**Kaggle v16 result:** P100 CUDA, JFK 11s → correct transcript, 0 NaN,
0 Inf, argmax=3976 (matches CPU). Elapsed 10.6s (encoder on GPU + LLM
on CPU) vs 8.3s pure CPU — slight overhead from GPU↔CPU transfer but
encoder benefits from CUDA for longer audio.

### What didn't work (so future investigators don't repeat)

| Attempt | Result |
|---|---|
| Q8_0 model (instead of F16) | Still !-loops |
| `FUNASR_NO_FA=1` (disable flash-attn) | Still all-NaN |
| `CRISPASR_KV_READ_F32=1` (F32 KV reads) | Still all-NaN |
| FA-off + KV_READ_F32 combined | Still all-NaN |
| Single-backend GPU sched (remove CPU) | Crashes (some ops need CPU) |
| `parallel=true` sched flag | Still all-NaN |
| CPU-only `llm_sched` (weights on GPU) | Crashes (can't read GPU weights) |
| Weight split but KV still on GPU | Inf at layer 2 → all-NaN by layer 3 |
| **QKV fusion alone (all on GPU)** | **Still all-NaN (v15)** |

### Diagnostic protocol that worked

1. `FUNASR_DUMP_STAGES=1` env-gated tensor stats at every 10th encoder
   layer + adaptor + spliced embeds + prefill logits → localized the NaN
   to the LLM prefill in one Kaggle run.
2. Three-way CPU/CUDA-FA/CUDA-noFA comparison → ruled out flash-attn.
3. Full 28-layer LLM scan → pinpointed layer 2 as the first Inf source.
4. Iterative Kaggle kernel versions (v3–v16) with one hypothesis per run
   — each version either confirmed or killed a theory in ~25 minutes.

### Takeaways

1. **Localize first, fix second.** The per-stage tensor dump was the
   single most valuable tool. Two prior fix attempts (commit `868aabdf`)
   shipped architecture-level hypotheses without numerical evidence and
   both failed.
2. **The ggml_backend_sched is a black box.** With two backends, the sched
   makes its own routing decisions. When those decisions are wrong, the
   symptoms (NaN/Inf) give no clue about the root cause. The only
   reliable approach is to force all data to one backend so the sched
   has no choice.
3. **QKV fusion is necessary but not sufficient.** Fusing the Q/K/V weights
   reduces the sched's degrees of freedom but doesn't eliminate the bug.
   The weight-split is the essential fix.
4. **KV cache placement matters.** Even with weights on CPU, a GPU KV cache
   confuses the sched enough to produce Inf at layer 2 (v11 finding).

### Cross-refs

- `src/funasr.cpp` final fix (commit `f94fec90`)
- `tools/kaggle/funasr-cuda-debug/` kernel (v3–v16)
- PLAN §136
- issue #125 (original report)

## Round 10 — SpeechT5 + Dia TTS backend ports (2026-05-31/06-01)

### SpeechT5 TTS
- **Encoder cos > 0.999** against Python F32 reference (all 8 tokens, all 12 layers)
- Key bugs fixed:
  1. **Attention head interleave**: must `permute(0,2,1,3)` BEFORE `reshape_2d` when merging multi-head output
  2. **Relative position bias batch dim**: the position_bias reshape must NOT permute — the get_rows flat lookup already produces `(hd, T_key, T_query)` with T_query as the batch dim
  3. **HiFi-GAN conv bias**: reshape to `(1, C_out)` not `(C_out, 1)` for broadcast over T
  4. **HiFi-GAN conv_transpose_1d**: ggml requires `padding=0` — trim output manually
  5. **F16 biases**: 1D tensors (biases/norms) MUST be F32 in GGUF for ggml binary ops
  6. **Decoder KV cache**: proper AR self-attention needs accumulated past K/V across steps
- Pipeline runs end-to-end: encoder → decoder w/ KV cache → postnet → HiFi-GAN → 16kHz WAV
- Decoder produced speech but wrong content — **FIXED 2026-06-02**, see
  "SpeechT5 TTS decoder — what ACTUALLY fixed it" below. Three bugs:
  KV cache buffer reuse, tokenizer `▁`, postnet/vocoder data layout.

### Dia 1.6B TTS
- **Encoder cos = 1.000000** all 12 layers against Python F32 reference
- **Decoder layer 0 cos = 0.999** against Python F16 reference
- Key bugs found and fixed:
  1. **RoPE mode**: Dia uses first-half/second-half pairing = ggml mode=2 (NeoX), NOT mode=0 (interleaved). Mode=0 caused cos=0.11 at layer 2!
  2. **Encoder attention V path**: head interleave bug (same as SpeechT5)
  3. **Encoder batch order**: CFG convention is [uncond=batch0, cond=batch1] — was swapped
  4. **Encoder padding mask**: needed 4D batch-aware mask (eventually solved by using T=prompt_size instead of 1024)
  5. **Decoder attention scale**: Dia uses scale=1.0 (NO 1/sqrt(d)) — agent's code had wrong 1/sqrt(128)
  6. **Decoder GQA repeat**: must use repeat_interleave pattern (insert unit dim before n_kv, repeat, flatten) — NOT modular repeat. `ggml_repeat_4d` gives modular (0,1,2,3,0,1,2,3) but Python needs consecutive (0,0,0,0,1,1,1,1)
  7. **CFG filtering**: Python uses CFG logits ONLY for top-k selection, then samples from CONDITIONAL logits (not CFG-combined). This is NOT standard CFG — it's "CFG-guided conditional sampling"
  8. **Logit masking**: channels 1-8 must not produce EOS/PAD/BOS tokens (codes must be < audio_vocab_size=1024)
  9. **Tokenizer**: don't add trailing period after ? or ! punctuation
  10. **Converter weight transposition**: DenseGeneral Q/K/V vs O_proj have different reshape+transpose patterns; wi_fused and logits_dense need special handling
  11. **Cross-attention dim**: cross-attn is MHA (16q, 16kv) NOT GQA — kv_dim=2048 not 512
- Full pipeline: encoder → cross-attn cache → decoder AR loop → CFG filter → delay revert → DAC decode → 44.1kHz WAV
- Audio produced but ASR says "music" not speech — remaining issue is F16 precision accumulation across 18 decoder layers with scale=1.0 attention

### General lessons
- **ggml_rope mode**: mode=0 = consecutive pair rotation (0,1),(2,3)...; mode=2 = half-split rotation (0,n/2),(1,n/2+1)... ALWAYS check against the Python RoPE implementation
- **ggml_repeat_4d for GQA**: gives MODULAR repeat (wrong for repeat_interleave). Use the pattern from core/attention.h: reshape(hd, 1, n_kv, T) → repeat(hd, n_rep, n_kv, T) → reshape(hd, n_heads, T)
- **Attention output reshape**: ALWAYS permute heads before reshape: (hd, T, n_heads) → permute → (hd, n_heads, T) → reshape (hd*n_heads, T). Without this permute, head concatenation is scrambled
- **ggml conv_1d**: expects (T, C_in) input — NOT channel-first. Conv bias: (1, C) for broadcast over T
- **Agent-written code**: found 11 bugs in agent-generated Dia runtime. Agents cannot reliably write correct attention code — always validate every intermediate against ground truth

## moshi / Mimi RVQ codebooks: decode uses embed_sum / cluster_usage (CSM §135, June 2026)

Any port of a moshi/Mimi codec (CSM-1B, Kyutai STT/TTS, anything using
`moshi.quantization`) must reproduce one non-obvious detail of the RVQ
codebook. The persistent buffer in the checkpoint is `embed_sum` (an EMA
sum), but the codebook actually used at decode time is the **computed
property**:

```python
# moshi/quantization/core_vq.py — EuclideanCodebook.embedding
embedding = embed_sum / cluster_usage.clamp(min=epsilon)   # epsilon = 1e-5
```

`cluster_usage` is a decayed EMA of per-code usage, NOT a count — its
values are mostly **< 1** (CSM-1B: mean ≈0.59, min ≈0.12, 96 % below 1).
So you cannot skip the division ("EMA stats not needed for inference" is
wrong), and you cannot clamp it at 1.0 "to avoid divide-by-zero": clamping
at 1.0 leaves ~96 % of codes effectively un-normalized and the codec
emits buzzing/tonal garbage from otherwise-correct codes. Clamp at the
real epsilon (1e-5).

This bit CSM-1B for a full session: `convert-csm-to-gguf.py` had
`np.maximum(cu, 1.0)`. RVQ-dequant cos vs the moshi reference was 0.908
(wrong) → 1.000000 after `np.maximum(cu, 1e-5)`. The tell was a fed-
reference-codes diff of the codec in isolation: `mimi_rvq_dequant` failed
while every backbone/depth stage passed cos≈1.0. Verify the codebook the
GGUF stores by reading one layer and diffing against all three candidates
(`embed_sum`, `embed_sum/max(cu,1.0)`, `embed_sum/max(cu,1e-5)`) — only
the last matches `mimi.quantizer...._codebook.embedding`.

General rule, same family as the StyleTTS/EMA traps: when a checkpoint
stores `*_sum` + `*_usage`/`*_count` buffers, the inference weight is the
quotient, computed lazily by a `@property`. Grep the reference module for
`@property` over any buffer you're about to export raw.
## Dia 1.6B TTS — what ACTUALLY fixed it (2026-06-02)

The earlier "11 bugs in agent-generated runtime" notes were chasing the wrong
target. With a TRUSTWORTHY reference the port came down to **three** structural
bugs; encoder, decoder logits, CFG inputs and DAC decode were already correct.

**Trustworthy reference first.** The handover's reference artifacts
(`dec_step0_logits_uncond.npy`, `python_greedy_5.npy`) were bogus (from a broken
script). Two symptoms gave it away: a "greedy" that didn't match the official
sampler, and an `uncond` whose `cos(cond,uncond)=0.22` with `‖uncond‖>‖cond‖`
(backwards for "no guidance"). Rebuilt the reference by running the **official
nari-labs dia at the matching commit** (`git checkout 4a9e29b` — the checkpoint
`nari-labs/Dia-1.6B` predates the `-0626` `encoder_config` refactor; HEAD won't
load the old config.json), F16, MPS, `Dia._load_dac_model = lambda self: None`,
on the Mac (the VPS is CPU + low-RAM, can't hold 6 GB). C++ then matched at
step-0 cos 0.99994 (cond) / 1.00000 (uncond).

**Bug 1 — CFG sampling policy.** The OLD checkpoint's `_decoder_step` samples
from the **CFG-combined** logits `cond + s·(cond−uncond)` (greedy = argmax of
those, eos-masked). The C++ had copied the **newer -0626** scheme (mask `cond`
to the CFG top-k, sample `cond`) — its own comment cited "lines 442-445" of the
*new* model.py. Result: ≈ raw-cond tokens, no real guidance → noise. Always pin
the dia source commit to the checkpoint.

**Bug 2 — self-attn KV-cache layout.** The cache vectors accumulated per step in
`[step][batch]` order (`s0b0,s0b1,s1b0,…`) but the `past_k` graph input is
`(kv_dim,T_past,B)` = ggml `[batch][step]`. They coincide only at `T_past==1`,
so step 0/1 are perfect and everything corrupts from step 2 (`T_past≥2`).
Caught by a **teacher-forced per-step logit diff** (force the reference's input
sequence into the C++, compare per-step cond/uncond cosine): step1 cos 0.99998,
step2 → 0.93. Reorder when feeding: `dst=(b*T_past+t)*kv_dim, src=(t*B+b)*kv_dim`.

**Bug 3 — missing input-side delay pattern.** Dia is trained in the delay
domain: at step t the model must see channel c's input **held at BOS until step
delay[c]** (delay=[0,8,9,…,15]). The C++ fed all 9 sampled tokens back
immediately → out-of-distribution → non-speech, even though per-step logits were
correct (the teacher-forced diff passed because it fed the reference's already-
delayed inputs). Fix: maintain a delayed-domain `gen[pos][c]` buffer (prefill
`gen[t][c]=BOS if t<=delay[c]`), feed `gen[t]` as input, write the sampled
(post-eos/delay-override) token to `gen[t+1]` with the start-of-sequence BOS
mask (only fill positions still unset), emit `gen[1..]` to the delay-revert.

**Method that worked (reusable for AR-codec TTS):** (1) get a byte-faithful
reference at the right commit; (2) diff step-0 logits per CFG batch (cos);
(3) isolate the codec by decoding the *reference* codes through the C++ DAC
(bit-exact here); (4) teacher-force the C++ with the reference input sequence and
diff per-step logits to separate per-step compute bugs from the sampling/feedback
loop. Validate end-to-end by ASR-roundtripping a **>100-char** prompt (Dia is
genuinely inconsistent on short prompts — the official model also outputs "music"
for them; this is sampling variance, not a port bug). F16 and Q4_K both roundtrip
verbatim. Debug hooks are env-gated with paths from the env value: `DIA_GREEDY`,
`DIA_DUMP_TOKENS`, `DIA_MAX_STEPS`, `DIA_FORCE_TOKENS`, `DIA_DUMP_STEPLOGITS`,
`DIA_DUMP_DIR`, `DIA_DECODE_CODES`.

## FastPitch TTS — non-autoregressive parallel TTS port (§133, 2026-06-02/03)

FastPitch is the simplest TTS architecture in the project: a single deterministic
forward pass (no AR loop, no sampling, no KV cache, no CFG). That made diffing
unusually clean — every stage should match the reference at cos ≈ 1.0, and any
deviation is a real bug rather than sampling variance. The port surfaced six bugs,
all in the converter or runtime graph, none in the model logic itself.

### Bug 1 — post-norm vs pre-norm (critical)

NeMo FastPitch defaults to `pre_lnorm=False`, meaning LayerNorm is **post-norm**:
`x = LayerNorm(x + sublayer(x))`. The stub had pre-norm:
`x = x + sublayer(LayerNorm(x))`. This silently produces plausible-looking but
wrong encoder output (cos ≈ 0.89 instead of 1.0). The fix is trivial once
identified: move the LayerNorm call from before the sublayer to after the
residual add. Both encoder and decoder use the same pattern.

**Lesson:** Always check `pre_lnorm` / `pre_norm` flags in the reference config.
NeMo, Fairseq, and HuggingFace transformers all default differently.

### Bug 2 — sinusoidal PE layout: cat [sin, cos] vs interleaved

NeMo's `PositionalEmbedding` stores `inv_freq` (not precomputed positions) and
computes PE on-the-fly as `cat([sin(pos·freq), cos(pos·freq)], dim=-1)` — the
first half of the embedding is all sines, the second half all cosines. The
converter initially generated interleaved `[sin₀, cos₀, sin₁, cos₁, ...]`.
This drops encoder cos from 1.0 to 0.89. The fix is a one-line change in the
converter's `generate_sinusoidal_pe()`.

**Lesson:** Every sinusoidal PE implementation chooses a different interleaving.
Always dump and compare the first few values of position 0 to verify layout.

### Bug 3 — ggml conv1d input layout

`ggml_conv_1d(weight, input)` expects input with `ne[0]=T` (spatial) and
`ne[1]=Cin` (channels). The encoder/decoder data flows in `(D, T)` =
`ne[0]=D, ne[1]=T`, which is the transpose. Without explicit `transpose_2d()`
calls around every conv1d operation, the assertion `b->ne[1] == a->ne[1]`
fires or (worse) the conv silently reads garbage.

The hifigan.h shared vocoder also expects `(T, Cin)`. SpeechT5 creates its mel
tensor as `ggml_new_tensor_2d(T_mel, mel_bins)` directly — it never goes through
the `(D, T)` intermediate that FastPitch uses.

**Lesson:** ggml is column-major (`ne[0]` contiguous). A 2D tensor `(ne0, ne1)`
stores `ne0` values contiguously, then repeats for each `ne1`. When calling
`ggml_conv_1d`, `ne[0]` must be the spatial axis. When passing data between
ggml stages and CPU-side buffers, be explicit about which axis is contiguous —
"(80, 191)" in ggml means 80 contiguous values per timestep (channel-first), not
the numpy (80, 191) which is 191 contiguous values per mel bin (batch-of-rows).

### Bug 4 — conv1d output is 3D, not 2D

`ggml_conv_1d` returns a 3D tensor `(OL, OC, N)` even for unbatched input
(`N=1`). Adding a 2D bias `(1, OC)` to a 3D result fails
`ggml_can_repeat()`. Fix: `ggml_reshape_2d(y, y->ne[0], y->ne[1])` before
the bias add.

### Bug 5 — vocoder mel data transpose

The decoder output is ggml `(ne[0]=n_mel, ne[1]=T)`, storing `n_mel` contiguous
values per timestep. The HiFi-GAN vocoder expects ggml `(ne[0]=T, ne[1]=n_mel)`
where `ne[0]` is the spatial conv1d axis. Feeding the raw flat buffer produces
quiet buzzy noise (RMS ≈ 0.017 vs reference 0.072). CPU-side transpose of the
mel data (`data[c·T+t] = src[t·n_mel+c]`) before `ggml_backend_tensor_set()`
fixes it. ASR roundtrip goes from empty transcript to correct.

**Lesson:** When piping between ggml sub-graphs via CPU buffers, always verify
the data layout matches the destination tensor's `ne` expectations. A flat
`memcpy` that silently transposes the semantic axes is the hardest bug to spot
because the model "runs" but produces garbage.

### Bug 6 — NeMo naming conventions differ across model versions

The converter was written for the German multispeaker model, which uses one NeMo
naming convention (`.norm1.`, `.norm2.`, `.pos_ff.conv_1.`). The English model
(`nvidia/tts_en_fastpitch`) uses a different convention (`.dec_attn.layer_norm.`,
`.pos_ff.layer_norm.`, `.pos_ff.CoreNet.0.`). The converter now handles both by
applying all rename rules and letting whichever matches stick.

Also: `.nemo` archives can be plain tar or gzip; the English model is plain tar.
The `weight_g`/`weight_v` weight-norm fusion and discriminator/aligner weight
stripping work the same for both.

### Conv weight transpose was wrong — no transpose needed

The converter had `transpose_conv_weight()` doing `arr.transpose(2,1,0)` to
convert PyTorch `(Cout, Cin, K)` to ggml `(K, Cin, Cout)`. But numpy row-major →
ggml column-major mapping already handles this: numpy `(Cout, Cin, K)` stores `K`
as the innermost dimension, which becomes ggml `ne[0]=K`. So the "transpose" was
actually double-transposing, producing `ne[0]=Cout` instead of `ne[0]=K`. Fix:
remove the transpose entirely (identity function).

**Lesson:** When writing a GGUF converter, numpy `(A, B, C)` row-major → ggml
`ne[0]=C, ne[1]=B, ne[2]=A`. If ggml expects `ne[0]=K, ne[1]=Cin, ne[2]=Cout`
and PyTorch is `(Cout, Cin, K)`, the numpy array is already in the right order.
Don't transpose.

### Quantization yields minimal size reduction for small conv-heavy models

FastPitch is ~60M params, mostly in 3D Conv1d weights and 1D biases/norms. The
quantizer only compresses 2D matrices (attention projections). Result: F32 = 231 MB,
Q8_0 = 227 MB, Q4_K = 227 MB — almost no savings. All three pass ASR roundtrip.
For conv-heavy models, quantization is about speed (SIMD Q4 matmul), not size.

### GGUF `add_array` with numpy arrays fails

The `gguf-py` library's `add_array()` raises "Invalid GGUF metadata array,
expecting sequence" when passed numpy arrays. Plain Python lists work fine.
Convert with `[int(x) for x in arr]` before calling `add_array()`.

### NeMo import on this host requires overrides patching

NeMo 2.7.3 + Python 3.13 + the `overrides` library hits a signature-check
TypeError in `OneLoggerPTLTrainer.save_checkpoint`. Patching all three functions
in `overrides.signature` to swallow `TypeError` lets import succeed. The
tokenizer and model both work correctly after the patch.

### Diff-test harness for deterministic models

For non-autoregressive models, every stage should match the reference at cos ≈ 1.0.
The harness: (1) dump reference per-stage with forward hooks, (2) teacher-force
tokens via `FASTPITCH_FORCE_TOKENS=<file>`, (3) dump C++ per-stage via
`FASTPITCH_DUMP_DIR=<dir>`, (4) compare with numpy cosine similarity. First
divergent stage is the bug. Debug hooks: `FASTPITCH_DUMP_DIR`, `FASTPITCH_FORCE_TOKENS`.

### GPU/CUDA crash — two Metal-masked bugs (§204, 2026-06-21)

FastPitch synthesised correctly on M1 Metal but aborted on the Kaggle CUDA
sweep. The give-away "passes on Metal, dies on CUDA" almost always means a
backend bug that **unified memory hides** — on Apple silicon the CPU backend can
read GPU buffers, so device-pointer mistakes are silent. Two distinct bugs:

**Bug A — `sched_alloc_graph` then compute on the raw CPU backend.** When the
GPU upgrade (§140) wired `ggml_backend_sched` into fastpitch, only the encoder
and pitch-embed graphs were switched to `ggml_backend_sched_graph_compute`; the
decoder and vocoder graphs were left calling
`ggml_backend_graph_compute(ctx->backend_cpu, gf)` after a
`ggml_backend_sched_alloc_graph`. With `use_gpu` the weights are in a GPU buffer,
so the CPU backend dereferences device pointers — harmless on Metal, an illegal
access on CUDA. **Rule:** a graph allocated via the scheduler must be computed
via the scheduler. Grep any GPU-enabled backend for
`ggml_backend_graph_compute(.*backend_cpu` following a `sched_alloc_graph`.

**Bug B — `core_hifigan` ConvTranspose layout (channel-major vs time-first).**
The whole `core_hifigan::forward()` pipeline is **time-major** `(T, C)` — every
`ggml_conv_1d` there produces `ne0=time, ne1=channel`. But the col2im decomposed
transpose-conv path (the `w_perm` branch, added in `a862f2de`) called
`core_convt::convt1d_decomp`, which is **channel-major**: its first op is
`mul_mat(w_perm[IC, K*OC], x)` and therefore needs `x->ne0 = IC`. Fed a
time-major `x` (`ne0 = T`), it aborts on `ggml_can_mul_mat` — on **both** CPU and
GPU. The fix is to use the time-first sibling `convt1d_decomp_tf` (transposes
in, runs the channel-major decomp, transposes out). This is a *shared-header*
bug: every `ups_w_perm` user is affected, and both FastPitch and SpeechT5 feed
`mel_in = (T_mel, n_mel)`, so the one-line `hifigan.h` change repairs both.

**Lesson:** when a `mul_mat`/conv asserts deep inside a shared vocoder, the wrong
suspect is the model weights; the right one is which axis of the *input* is
contiguous. `convt1d_decomp` (channel-major) and `convt1d_decomp_tf` (time-first)
look interchangeable but assume opposite `ne0` conventions — pick by what the
caller's pipeline actually produces (here, `ggml_conv_1d` ⇒ time-major).

Validated: cstr/fastpitch-en-GGUF q8_0, GPU and CPU outputs byte-identical,
verbatim ASR roundtrip. See `core/hifigan.h::conv_transpose_1d`,
`fastpitch_tts.cpp` decoder/vocoder compute, HISTORY §204.

## SpeechT5 TTS decoder — what ACTUALLY fixed it (2026-06-02)

The encoder was validated at cos > 0.999 in the earlier round (§8115). The
decoder "produced audio but not the prompt" — same symptom class as the Dia KV
bug. Three bugs, all found by systematic per-step diff against a PyTorch
reference with dropout disabled (`_consistent_dropout = lambda x, p: x`).

**Bug 1 — self-attention KV cache buffer reuse (the primary cause).** The ggml
graph allocator reused the memory buffer backing `sk_cur` / `sv_cur` (the K/V
projection tensors for the current step) before the host-side C++ code read
their values into the CPU-side KV cache via `ggml_backend_tensor_get`. At step 0
(single-position self-attention) the cache doesn't matter, so step 0 matched
perfectly. From step 1 onward, the cache contained whatever later graph
operation happened to land in the same buffer — completely wrong K/V, causing
the self-attention to produce garbage. **Fix:** create `ggml_dup` copies of
`sk_cur`/`sv_cur` and expand them as graph outputs; the dup copies get their
own non-reusable buffers. The original `sk_cur`/`sv_cur` still feed into the
attention graph normally. This is the **exact same class** as Dia's KV bug
(`01beaeaa`), just triggered differently: Dia stored K/V in wrong
`[step][batch]` order; SpeechT5 stored them from a clobbered buffer.

**Diagnostic that caught it:** the per-step dump initially showed the prenet
output (after ReLU) containing negative values — impossible after ReLU.
Switching from `ggml_graph_get_tensor(gf, "name")` to reading from the saved
pointer gave the same corrupt data. Only `ggml_dup` (which forces the allocator
to assign a fresh buffer) fixed it. The `ggml_build_forward_expand` call alone
does NOT prevent buffer reuse — ggml's allocator tracks liveness, not output
status. An intermediate tensor that is "dead" after its last consumer reads it
may have its buffer overwritten even before `ggml_backend_graph_compute` returns,
because compute is node-by-node.

**Bug 2 — tokenizer missing leading `▁`.** SpeechT5 uses a SentencePiece
char-level tokenizer that prepends `▁` (U+2581) at the start of text and
replaces spaces with `▁`. The C++ tokenizer only replaced spaces, producing 7
tokens for "Hello." where the reference produces 8 (`[▁, H, e, l, l, o, ., </s>]`).
With the wrong token count, the encoder output differs and cross-attention
attends to the wrong content. **Fix:** prepend `▁` before encoding characters.

**Bug 3 — postnet + vocoder data layout.** The mel spectrogram was stored
row-major `(T, 80)` in C++ (time steps × mel bins), then passed directly to
`ggml_conv_1d` via a tensor declared as `(ne[0]=T, ne[1]=80)`. But ggml
column-major layout means element `[t, c]` is at offset `t + c*T`, while the
row-major data has it at `t*80 + c`. Result: the conv read transposed data →
NaN/garbage output from the postnet and vocoder. **Fix:** explicit transpose
when setting input (`transposed[t + c*T] = row_major[t*80 + c]`) and reverse
transpose when reading output. Applied in both `run_postnet` and `run_vocoder`.

**Method (reusable for AR-mel TTS).** (1) Dump per-step decoder intermediates
from both PyTorch and C++ (prenet output, decoder hidden, mel, self-attn K/V).
(2) Compare per-step with cosine similarity — the first step that diverges
localizes the bug. (3) For KV cache bugs: step 0 matches perfectly (no past
cache) but step 1 fails at the hidden state (past cache is corrupt). (4) For
data layout bugs: all decoder steps match but final audio is garbage — isolate
postnet vs vocoder by feeding reference mel through the C++ vocoder. (5) ASR
roundtrip validates end-to-end. The Python reference ALSO drops "there" from
"Hello there, how are you doing today?" → SpeechT5 model limitation, not a
port bug.

**Debug hooks:** `SPEECHT5_DUMP_DIR=<dir>` writes per-step `step{N}_mel.f32`
and `step{N}_self_k_L0.f32`. Compare with `tools/dump_speecht5_decoder.py`
(PyTorch side) and `tools/compare_speecht5_dumps.py` (cosine/maxabs report).

### Key ggml lesson: `ggml_build_forward_expand` ≠ buffer preservation

Expanding a tensor as a graph output ensures it gets **computed**, but the
allocator may still reuse its buffer after its last consumer reads it — even
before `ggml_backend_graph_compute` returns, because compute is node-by-node.
To **read back an intermediate** after graph compute, you need a `ggml_dup`
copy (or `ggml_cpy` to a pre-allocated input tensor). This applies to any
backend that wants to read back KV projections, intermediate hidden states,
or any tensor that has downstream consumers in the graph.

---

## Parler TTS — T5 + MusicGen decoder + DAC 44 kHz (§137, June 2026)

### `gguf_init_from_file(no_alloc=false)` sets `tensor->data` but not `tensor->buffer`

The legacy GGUF loading pattern (`no_alloc=false`) allocates tensor data inside the
ggml_context's memory pool. This means `tensor->data` is valid for CPU reads, but
`tensor->buffer` is NULL. Consequences:

1. `ggml_backend_tensor_get()` crashes (requires `tensor->buffer`).
2. `ggml_backend_sched` crashes during `alloc_graph` — can't route tensors without buffers.
3. `ggml_free(ctx_w)` on exit triggers `munmap_chunk(): invalid pointer` — the GGUF loader's
   internal allocation tracking conflicts with ggml_context's own free path.

Fix: switch to the two-pass `core_gguf::load_weights()` pattern (metadata pass, then
`no_alloc=true` + `ggml_backend_alloc_ctx_tensors`). This gives every tensor a proper
`buffer`, enabling `ggml_backend_tensor_get`, scheduler routing, and clean shutdown.

### Per-step `ggml_gallocr` create/free is the dominant cost in AR decode

Each MusicGen AR decode step (up to 2580 steps) was creating a new `ggml_gallocr`,
allocating the graph, computing, then freeing. The allocator's internal hash table
construction/destruction dominated wall time — the actual matmuls were secondary.

Fix: create one `ggml_gallocr` before the loop, `ggml_gallocr_reserve()` with a
worst-case graph (max KV length), then `ggml_gallocr_alloc_graph()` each step.
The reserved memory fits all subsequent steps since they have equal or smaller tensors.

### SentencePiece BPE ≠ Viterbi unigram — different algorithm, same `.model` file

SentencePiece `.model` files can be either unigram (model_type=1) or BPE (model_type=2).
Both store a vocab with scores, but the tokenization algorithms are fundamentally different:

- **Unigram Viterbi**: DP over byte positions, picks the highest-score segmentation.
- **BPE merge**: start with characters, iteratively merge the pair whose merged result
  has the highest score. Greedy, not global-optimal.

The Parler tokenizer is BPE (LLaMA-2 sentencepiece). Using Viterbi produced 2/24 wrong
tokens for typical voice descriptions (e.g. "moderate" → `▁moder`+`ate` instead of
`▁moderate`). The T5 encoder then conditions differently, compounding over 500+ AR steps.

Fix: added `core_spm::tokenize_bpe()` to `sentencepiece.h` — iterative best-merge with
UTF-8 symbol splitting. Converter stores original SP scores (not byte-length hack) +
`parler.tokenizer.is_bpe` GGUF flag. Runtime auto-selects algorithm.

Validation: 24/24 description tokens and 13/13 prompt tokens match Python HF tokenizer
exactly after fix (was 22/24 + 13/13 before).

### Zonos TTS — RoPE mode and GatedMLP chunk ordering are easy to get wrong

Two structural bugs survived until diff-harness validation (2026-06-09):

1. **RoPE type**: x_transformers `apply_rotary_emb` reshapes to `(…, d/2, 2)` and
   rotates *consecutive pairs* (indices `2i, 2i+1`). This is `GGML_ROPE_TYPE_NORMAL`
   (mode=0). Using `GGML_ROPE_TYPE_NEOX` (mode=2, half-split: `x[i], x[i+d/2]`)
   degrades prefill_hidden cosine from 0.996 to 0.984.

2. **GatedMLP chunk ordering**: Zonos uses `fc2(y * silu(gate))` where `y=chunk0`
   (linear passthrough) and `gate=chunk1` (silu activation). The `swiglu_fused_gate_up`
   helper applies silu to `chunk0` — the opposite convention. Always verify which chunk
   gets the activation by reading the upstream Python directly.

### Selective quantization for AR-decoder EOS logits

Uniform Q4_K of any multi-codebook AR TTS model can inflate the EOS logit at AR step 0
because the per-codebook output heads (`heads.*.weight`) are small (one weight matrix per
codebook) and are directly multiplied by the noisy backbone hidden state. Even a cosine
similarity of 0.996 on the hidden state translates to a logit shift of ~0.9 units on EOS
when head weights also carry Q4_K noise — enough to push P(EOS) from ~38 % to >60 %.

Fix pattern (verified on Zonos, generalises to other multi-codebook AR TTS):
- Keep `heads.*` (output projections) at F16 — eliminates the head-weight noise component.
- Keep `embeddings.*` (input codebook lookups) at F16 — stabilises input-side activations.
- Keep `prefix_conditioner.*` / `speaker.*` / `code_pred.output.*` etc. at F16 — these
  are small and conditioning-critical.
- Quantize only the bulk backbone projection tensors (QKV, O-proj, FFN gate/up/down).

The same pattern is already used for: Bark (token/pos embeds + EnCodec), Qwen3-TTS
(ECAPA speaker, code_pred.output, token_embd), CosyVoice3 (speech_embd, speech_lm_head).

For Zonos specifically: 210 backbone tensors quantized, 36 heads + 36 embeddings + 18
prefix_conditioner kept F16. Result 931 MB vs 872 MB full-Q4_K; residual step-0 failures
(~25 % of seeds) resolved by a 3-retry guard that bumps `rng_state` by a prime offset.

### DAC-44kHz standalone GGUF cannot be block-quantized

The standalone `dac-44khz.gguf` (arch=`dac-44khz`) stores Conv1d weights in ggml layout
`[K, IC, OC]` where `K` (kernel_size) is `ne[0]`. DAC kernels are 1, 4, 7, 8, or 16
elements wide. The codebook embeddings have `ne[0]=8`. Every weight tensor is below the
32-element minimum for Q8_0 (256 for Q4_K). The model is 104 MB in F16 and that is the
minimum achievable size — block quantization is architecturally impossible here.

The `crispasr-quantize` tool detects `arch="dac*"` and prints a specific warning; it also
reports "0 tensors quantized" at the end of any run that produces a same-size file, so
this failure mode is no longer silent.

### DAC audio codec weights are precision-sensitive to quantization

The DAC 44 kHz codec (Snake activations + ConvTranspose1d upsampling stack) reconstructs
waveforms from 8-dim codebook embeddings through 4 decoder blocks × 3 residual units.
Quantization noise in these small conv weights produces audible artefacts — same pattern
as chatterbox's HiFi-GAN vocoder and F5-TTS's Vocos.

Fix: `crispasr-quantize` skips all `dac.*` tensors (left at F16/F32). The T5 encoder and
MusicGen decoder are safe to quantize. Impact: Q8_0 GGUF is 979 MB instead of ~900 MB
(80 MB overhead for preserving ~116 DAC tensors at F16).

### Diff harness: compare un-delayed codes, not raw generation-step tokens

MusicGen uses a delay pattern: codebook k is delayed by k steps. The raw generation
output has codebook 0 producing at step 0, codebook 1 starting at step 1, etc.
`synthesize_codes()` returns un-delayed aligned codes (audio frame × codebook).

The reference Python dumper originally stored raw per-step tokens in `(num_cb, n_steps)`
layout. The C++ diff harness indexed these as delayed codes but compared against un-delayed
C++ output — producing a misleading 21.7% match rate even when the decode was correct.

Fix: Python dumps un-delayed aligned codes in `(T_audio, num_cb)` row-major layout.
C++ comparison indexes both sides with `t * num_cb + k`.

### MusicGen requires stochastic sampling — greedy produces degenerate output

MusicGen's 9-codebook delay pattern means the model is trained with temperature=1.0
sampling. Greedy decode (temp=0) produces degenerate output in Python too ("♪ How ♪"
repeated). The C++ port confirms this — greedy tokens match Python exactly for 10 steps,
but the resulting audio is garbage. Stochastic sampling with temp=1.0 is required.

The C++ sampler uses `std::mt19937` while PyTorch uses its own MT19937 implementation.
Same seed produces different random sequences → different token choices → divergent audio
after the first few words. This is expected and unfixable without implementing PyTorch's
exact RNG. Top-k sampling (added as `top_k` param) constrains the distribution so
divergent draws still land in high-probability regions.

## MAES beam search for TDT transducers (§134, June 2026)

### What MAES is and why it beats standard beam search

Modified Adaptive Expansion Search (MAES) is a transducer-specific beam
decoding algorithm from NVIDIA NeMo. It's a "Pareto improvement" over
standard beam search: beam-quality WER at near-greedy speed.

The key insight: standard beam search expands all V hypotheses at every
timestep, which is expensive for transducers (V can be 1K–8K). MAES is
**time-synchronous** — it processes one encoder frame at a time and
adaptively expands only up to N non-blank tokens per frame (usually N=2),
with gamma-threshold pruning that kills low-probability branches early.

### NeMo's 2-step SOS convention for the LSTM prediction network

The biggest bug during porting: NeMo's `decoder.predict(y=None, add_sos=True)`
feeds **two** zero inputs to the LSTM (an explicit SOS zero vector prepended,
plus the pad-token zero from y=None), while our C++ code was feeding **one**
(the blank embedding, which happens to be all-zeros). After one LSTM step
the hidden state is different from after two steps, even though both inputs
are zeros. Fix: call `predictor_step(blank_id)` twice.

This was caught purely by the diff harness — `decoder_initial` had
cos=0.641 until the fix, then cos=1.000000.

### NeMo's joint_after_projection applies log_softmax on CPU

`RNNTJoint.joint_after_projection()` has an auto-mode: when `self.log_softmax
is None` and the tensor is on CPU, it applies `log_softmax` to the output.
Fix: capture raw logits from `joint.joint_net()` directly in the reference
dumper, bypassing the auto log_softmax.

### Component-level diff testing is essential for transducer ports

The diff stages (`encoder_output_projected`, `decoder_initial`, `joint_t0`)
each isolate one component. The porting sequence was:
1. encoder_output_projected — PASS immediately (linear projection)
2. decoder_initial — FAIL at cos=0.641 → found 2-step SOS bug → PASS
3. joint_t0 — FAIL at cos=-0.63 → found log_softmax in reference → PASS

### Implementation: ~170 lines of C++ for the core MAES loop

The algorithm per encoder frame:
1. Run joint on all beam hypotheses at current frame
2. Top-k candidates (beam + beta) with gamma-threshold pruning
3. Split into blank (collect in `list_b`) and non-blank (expand predictor)
4. If only blanks → stop early
5. Continue expanding non-blank survivors for up to N steps
6. After last step: force-add blank log-prob to all non-blank survivors
7. Merge `list_b`, keep top-B by cumulative score, advance frame

Config defaults (matching NeMo): beam=4, num_steps=2, gamma=2.3, beta=2.
Enable via `CRISPASR_PARAKEET_MAES=1` + `--beam-size 4`.
### Benchmark results: MAES vs greedy on FLEURS English (CPU, June 2026)

| Audio | Model | Greedy | MAES beam=4 | Improvement |
|---|---|---|---|---|
| 10s | v2 (1K vocab) | "...lag behind by 25%." | "...lag behind by 25 years." | Corrected factual error |
| 10s | v3 (8K vocab) | "...lag behind by 25-30 years." | "...lag behind by 25 to 30 years." | More natural phrasing |
| 60s | v2 | 5 sentences | 6 sentences (recovered "To the north and within easy reach, is the romantic and fascinating town of Sintra") | Full sentence recovered |
| 11s jfk | v2 | identical | identical | Clean audio = no difference |

**Speed**: ~35% slower on 60s (5:21 vs 3:59 wall time, CPU-only). The cost
is almost entirely in the extra predictor LSTM + joint forward calls per
non-blank expansion — O(beam × expansions) per frame instead of O(1). On
GPU the decode is a tiny fraction of total time (encoder dominates), so the
wall-time impact would be negligible.

**Takeaway**: MAES is worth it for any audio that's not studio-clean. It
recovers content that greedy misses entirely, not just minor spelling fixes.
The v3 8K-vocab model benefits less (larger vocab = better greedy baseline).

### Extended benchmark: all parakeet model variants (10s + 60s FLEURS, CPU)

| Model | Type | 10s greedy | 10s MAES | 60s greedy→MAES time |
|---|---|---|---|---|
| tdt-0.6b-v2 (1K) | TDT | "...by 25%." | "...by 25 years." ✓ | 3:59 → 5:21 (+35%) |
| tdt-0.6b-v3 (8K) | TDT | "...by 25-30 years." | "...by 25 to 30 years." ✓ | — |
| tdt-1.1b (8K) | TDT | "...by twenty five to thirty years" | identical | — |
| tdt_ctc-110m (1K) | TDT | garbled ("commununication") | same garble | — |
| rnnt-0.6b (8K) | RNNT | "...by twenty five to thirty years" | identical | — |
| rnnt-1.1b (8K) | RNNT | 60s: truncated at "lettering" | identical | 7:37 → 9:28 (+25%) |

**Findings**: MAES helps most on smaller models with smaller vocabs (v2 1K).
Larger models (1.1b) and larger vocabs (8K) already have strong greedy
baselines where MAES matches but doesn't improve. The 110M model is too
small — MAES can't fix model capacity limits.

RNNT MAES works correctly but shows no improvement over greedy on these
test clips — the RNNT models produce clean output on clean audio.

### CTC prefix beam search — shared core for all CTC backends

Added `core_ctc::prefix_beam_search()` in `src/core/ctc.h`. Any backend
with CTC logits can call it in one line. The algorithm is standard Graves
& Jaitly 2014 prefix beam search with an optional gamma-threshold prune.
Wired into parakeet-CTC and sensevoice; other CTC backends just need the
include and a `beam_size` setter.

---

## Bark TTS — what ACTUALLY fixed it (June 2026)

Bark is a 3-stage hierarchical TTS (text→semantic GPT-2 → coarse GPT-2 →
fine bidirectional GPT → EnCodec decoder → 24 kHz PCM). The runtime was
written (`src/bark_tts.cpp`) but had three classes of bug: a Q4_K crash,
incoherent output even at F16, and Q4_K producing silence after the crash
was fixed.

### Bug 1 — Q4_K crash: hardcoded F16 row stride in embedding lookups

**Symptom.** `bark-small-q4_k.gguf` aborted with
`GGML_ASSERT(offset + size <= ggml_nbytes(tensor))` inside
`ggml_backend_tensor_set` for the `inputs_embeds` graph input.

**Root cause.** `compute_embeddings()` computed the byte offset for each
embedding row as `tok * D * sizeof(ggml_fp16_t)`. For F16 tensors this
is correct (row stride = D × 2 bytes). For Q4_K, the row stride is
`ggml_row_size(Q4_K, D)` — typically smaller because the quantised block
packs 32 elements into 18 bytes instead of 64. The hardcoded offset
overran the tensor's backing buffer.

The same F16 assumption infected **six call sites**: `compute_embeddings`
(text/coarse models), `generate_fine` (fine model per-codebook embeddings),
`fold_weight_norm` (EnCodec weight-normalised convolutions), `cpu_lstm_forward`
(EnCodec LSTM weights), and `encodec_decode` (RVQ codebook lookups).

**Fix.** Two generic helpers: `tensor_get_row_f32(tensor, row, dst, ne0)`
reads one row of any ggml type (F32 fast path, F16 conversion, quantised
via `ggml_get_type_traits(type)->to_float`); `tensor_get_all_f32(tensor,
dst, n)` reads the entire tensor. All six sites refactored to use these.

**Lesson.** Embedding lookup in ggml C++ runtimes must never assume a
specific storage type. `ggml_row_size(type, ne0)` gives the correct
byte stride for any type, and the type-traits `to_float` callback handles
dequantisation. This pattern (`tensor_get_row_f32`) is now reusable for
any backend that does CPU-side embedding lookups on potentially-quantised
tensors — Bark, the EnCodec decoders, Chatterbox vocoder, etc.

### Bug 2 — semantic stage sampling: EOS never fires, logits never populated

**Symptom.** F16 runs completed all 3 stages but produced incoherent
speech. The semantic stage always ran to `max_steps` (768) instead of
stopping naturally.

**Root cause.** Two bugs in `generate_text_semantic`:

1. **`sample_logits` was allocated but never filled.** The code declared
   `std::vector<float> sample_logits(10001)` for the
   "relevant logits" (10000 semantic + 1 EOS) but then sampled directly
   from the raw model `logits` pointer — without copying the relevant
   slice into `sample_logits`. The min_eos_p early-stopping code then
   read zeroes from `sample_logits`, so EOS probability was always ~0.

2. **EOS excluded from sampling.** `sample_from_logits(logits, 10000, ...)`
   sampled from exactly 10000 elements — indices 0…9999. The EOS token
   sits at index 10000 (the `SEMANTIC_PAD_TOKEN` position in the model's
   output vocab). Python bark builds `relevant_logits =
   hstack([logits[:10000], logits[10000]])` and samples from 10001 values;
   the C++ never included that last element.

**Fix.** Before sampling, copy `logits[0:10000]` + `logits[SEMANTIC_PAD_TOKEN]`
into `sample_logits`, then sample from `sample_logits` with vocab size 10001.
This matches the Python bark `relevant_logits` construction exactly.

**Lesson.** When porting AR sampling code, check that the "relevant logits"
slice matches the reference exactly — both the range and the EOS append.
Zeroed-but-unused buffers are a classic "works by coincidence on F32, fails
subtly" bug because the sampling softmax over 10001 values with a zeroed
EOS slot makes EOS vanishingly unlikely.

### Bug 3 — Q4_K silence: aggressive quantization of embeddings + EnCodec

**Symptom.** After fixing the crash, Q4_K loaded and ran all stages but
produced near-zero audio (peak amplitude 0.0014 vs 0.38 at F16).

**Root cause.** `crispasr-quantize` quantised every 2D weight tensor to
Q4_K, including token embeddings (129600×768 for text, 12096×768 for
coarse), position embeddings (1024×768), output lm_heads, and the entire
EnCodec decoder (weight-normalised convolutions, LSTM weights, RVQ
codebook embeddings). All of these are read by CPU-side tensor lookups
(not through the ggml graph) and are precision-sensitive.

The embeddings are the first and most damaging: a Q4_K embedding lookup
for token ID X returns an approximation that's off by enough to push
the GPT-2 into different semantic token trajectories. The cascading error
through 3 stages (semantic→coarse→fine→EnCodec) compounds into garbage.

**Diagnosis path.** Ran F16 vs Q4_K vs Q8_0 with the same seed+text:

| Model   | Size  | Peak amp | Tokens |
|---------|-------|----------|--------|
| F16     | 772M  | 0.38     | 167    |
| Q8_0    | 415M  | 0.43     | 113    |
| Q4_K-v1 | 225M  | 0.001   | 113    |
| Q4_K-v2 | 423M  | 0.50     | 145    |

Q8_0 works → the compute graph is correct. Q4_K-v1 fails → quantisation
noise is the issue. Selective Q4_K-v2 (only attn_qkv, attn_output,
ffn_up, ffn_down) works → embeddings/codec are the sensitive tensors.

**Fix.** Added a bark-specific rule in `crispasr-quantize/main.cpp`: skip
any tensor matching `token_embd`, `pos_embd`, `output` (but not
`attn_output`), or `encodec.*`. Only the 144 attention/FFN projection
weights (across all 3 sub-models) are quantised.

**Lesson.** For multi-stage AR-codec TTS models, embedding and codec
weights are typically precision-critical. The quantiser must skip them.
The telltale sign is "Q8_0 works, Q4_K doesn't" — if the full graph is
correct at Q8_0, the remaining delta is quantisation noise in the
non-graph (CPU-side tensor-read) path. The `output` skip was tricky
because `sname.find("output")` also catches `attn_output` — use
`sname.find("output") != npos && sname.find("attn_output") == npos`.

### Tokenizer: BERT WordPiece from GGUF, modular in core/

**What.** Bark's text stage expects BERT wordpiece token IDs (from
`bert-base-multilingual-cased`) offset by `TEXT_ENCODING_OFFSET` (10048).
The original code used byte-level encoding (each UTF-8 byte + 10048),
which produces valid but wrong token IDs → the semantic model generates
incoherent semantic tokens.

**Fix.** The converter (`convert-bark-to-gguf.py`) already embeds the
BERT vocab via `w.add_token_list(toks)` → `tokenizer.ggml.tokens` KV
array. Added `core/wordpiece.h` — a shared BERT WordPiece tokenizer
(auto-detects cased vs uncased, BERT BasicTokenizer punctuation splitting,
UTF-8 greedy longest-match). DRY: replaces the inline copy in
`fireredpunc.cpp` and serves bark. Loads at init via
`core_gguf::kv_str_array(g, "tokenizer.ggml.tokens")`.

**Lesson.** Tokeniser code is always reusable — put it in `core/` from
day one. The "byte-level fallback" pattern (warn + degrade gracefully when
the GGUF lacks vocab) is good practice for backwards compatibility with
older GGUFs.

### NPZ speaker prompt loader: minimal ZIP+NPY parser

**What.** Bark voice conditioning requires `.npz` files containing
`semantic_prompt`, `coarse_prompt` (2,T), `fine_prompt` (8,T) as int64
arrays. The `bark_set_speaker_npz` was a TODO returning -1.

**Fix.** Implemented a minimal ZIP local-file-header parser (`parse_npz`)
and NPY parser (`parse_npy_to_int32`) — handles version 1.0/2.0, int64/
int32/int16 dtypes, arbitrary shapes. ~150 lines, no external deps. The
semantic history is prepended as the second 256-token block in
`generate_text_semantic` (merge_context sums them with the text
embeddings). Coarse history is interleaved as full-vocab tokens and
prepended after `COARSE_INFER_TOKEN`.

**Lesson.** NPZ is just a ZIP of .npy files; .npy v1/v2 is a trivial
header + raw data. A minimal parser is ~100 lines and avoids pulling in
cnpy or similar. Bark's speaker conditioning is essential for coherent
output — without it the model free-hallucinates.

### CIFS + git worktree = data loss

During this work, a git worktree on the CIFS mount (`/mnt/storage/`)
suffered file corruption: `Edit` tool writes succeeded (the data hit the
CIFS cache) but the underlying inode was deleted (Links: 0), making the
file unreadable by `cat`, `cp`, or git. The edits were lost.

**Fix.** Moved the worktree to the local SSD (`/mnt/volume1/worktree-bark-fix`).
Build dirs also on local SSD (`/mnt/volume1/build-bark-fix`). Models and
large data on CIFS; code and builds on local disk.

**Lesson.** Never put git worktrees on CIFS mounts. The `actimeo=60` cache
and async writeback can leave files in a "visible in `ls -la` but
unreadable" ghost state. Local SSD for worktrees; CIFS for data only.

### NPZ ZIP64: NumPy writes ZIP64 even for tiny files

NumPy's `np.savez` writes ZIP64 extra fields (local header `comp_size =
0xFFFFFFFF`, actual size in extra tag `0x0001`) even for files under 1 KB.
The initial NPZ parser only checked the 32-bit size field and broke on
`0xFFFFFFFF` (tried to read 4 GB). **Fix:** when `comp_size32 ==
0xFFFFFFFF`, parse the ZIP64 extra field (tag `0x0001`, offset +12, 8-byte
little-endian compressed size).

**Lesson.** Always handle ZIP64 in NPZ parsers — NumPy defaults to it
regardless of file size.

### n_threads non-determinism in AR sampling

With the same seed, switching from 4 to 2 threads produces different audio
(cos=0.50 at Q4_K, cos=0.87 at F16). This is inherent to multi-threaded
GEMM: floating-point reduction order differs across thread counts, producing
slightly different intermediate values that accumulate through 768+ AR steps.

This is NOT a bug — it matches upstream PyTorch behavior (different
`torch.set_num_threads()` also produces different samples). Seed
reproducibility is only guaranteed at the same thread count.

### Integration test results (June 2026)

Full test matrix on CPU (VPS, no GPU):

| Test | Q4_K v2 | F16 | Q8_0 |
|------|---------|-----|------|
| Model load | PASS | PASS | PASS |
| BERT tokenizer | PASS (119547 tok) | PASS | PASS |
| NPZ speaker load | PASS | — | — |
| Synthesis peak | 0.70 | 0.79 | 0.43 |
| Seed repro (cos) | 1.000000 | 1.000000 | — |
| Cross-seed diverge | cos=-0.003 | cos=0.010 | — |
| Temperature setter | PASS | PASS | — |
| ASR roundtrip | "Hello!" ✓ | "And" (stochastic) | — |
| Q4_K v1 (all quant) | BLANK_AUDIO | — | — |

GPU dispatch is wired (`ggml_backend_init_best()` + dual-backend scheduler)
but untested on this CPU-only VPS. Pattern matches csm_tts, kokoro, voxtral.

---

## Pocket TTS — 12 bugs from stub to cos=0.999997

Pocket TTS (Kyutai, 100M params, MIT/CC-BY-4.0) is architecturally unique
among the TTS backends: **continuous-latent AR** — no codebook, no RVQ, no
softmax. The AR loop emits 32-dim float vectors at 12.5 Hz via one-step
Lagrangian Self Distillation (LSD), decoded by a Mimi VAE (SEANet CNN +
2L transformer) to 24 kHz PCM.

### The 12 bugs and what they teach

**Bug 1 — SentencePiece tokenizer was a byte-level stub.**
The converter stored the raw `.model` protobuf bytes in the GGUF but the
runtime mapped raw UTF-8 bytes to token IDs instead of running SentencePiece.
Fix: converter extracts vocab+scores via `sentencepiece` library into standard
`tokenizer.ggml.tokens`/`scores` arrays; runtime uses `core_spm::tokenize`.
Verified exact token-ID match against Python SentencePiece.

**Bug 2 — Quantizer projection weight layout (GGUF row-major).**
`w[d*OD+o]` should be `w[o*LD+d]`. GGUF `ne=[1,32,512]` in ggml row-major
means `data[cout*Cin + cin]` = `(Cout=512, Cin=32)`, not `(Cin, Cout)`.
**Lesson:** for any Conv1d/Linear weight in GGUF, the fastest-varying ggml
dimension `ne[0]` is the input dimension; rows are indexed by `ne[1]`.

**Bug 3 — ConvTranspose1d weight index order.**
Used `w[k*Cout*Cin + co*Cin + ci]` but ggml row-major with `ne=[K, Cout, Cin]`
is `w[cin*Cout*K + cout*K + k]`. The mnemonic: outermost dimension in `ne` is
the outermost loop in row-major indexing.

**Bug 4 — Upsample was not depthwise.**
The Mimi upsample is a depthwise `ConvTranspose1d(512, 512, K=32, stride=16,
groups=512)`. PyTorch weight shape `(Cin=512, Cout/groups=1, K=32)`. GGUF
`ne=[32, 1, 512]` — the `ne[1]=1` signals depthwise. The generic
`conv_transpose1d_eager` produced 1 output channel instead of 512. Added a
depthwise path that independently upsamples each channel.

**Bug 5 — Mimi decoder transformer missing RoPE.**
The C++ comment said "No RoPE for Mimi transformer" but the reference applies
RoPE with `max_period=10000` per-head. Both the encoder and decoder Mimi
transformers use RoPE. **Lesson:** never trust comments in stub code; verify
every architectural assumption against the reference source.

**Bug 6 — RoPE split-half vs interleaved convention.**
The Kyutai/Moshi stack uses **interleaved** complex pairs: `vec[2i]=real,
vec[2i+1]=imag`. The C++ had split-half: first half real, second half imaginary.
This affected both the FlowLM backbone and the Mimi transformer. **Lesson:**
RoPE convention varies between model families. Always check the reference
`apply_rope` source — the dimension layout of the complex pairs is not
standardized.

**Bug 7 — Flow ResBlock MLP biases discarded.**
The weight loader read `mlp.0.bias` but immediately discarded it with
`(void)b0`, and `mlp.2.bias` was never loaded. The struct lacked bias fields.
**Lesson:** `(void)variable` is a code smell in weight loading — it usually
means someone intended to wire it but didn't finish.

**Bug 8 — RMSNorm used mean(x²) instead of var(x) — 1.78× magnitude error.**
The reference's "RMSNorm" (`_rms_norm` in `pocket_tts/modules/mlp.py`) calls
`torch.var(dim=-1)` which is **variance** (mean-subtracted, Bessel-corrected
`n-1` denominator), NOT `mean(x²)`. When the mean is nonzero, `var < mean(x²)`,
so `rsqrt(var)` > `1/RMS`, making the reference output systematically larger.
This single bug caused: `t_combined_norm` 9× too small → `y_norm` wrong →
ResBlock gates under-driven → final latent 1.78× magnitude error.
**Lesson:** "RMSNorm" is not a standardized operation. Some implementations
(LLaMA, GPT-NeoX) use true `mean(x²)`, others (Kyutai) use `var(x)`. Always
read the source, never assume from the class name.

**Bug 9 — EOS detector applied sigmoid then compared against 0.5.**
The reference compares the raw linear output `self.out_eos(x) > threshold`
without sigmoid. The C++ applied `sigmoid(logit) > 0.5` which is equivalent
to `logit > 0`, a much lower effective threshold than `logit > 0.5`.

**Bug 10 — Mimi decoder transformer O(n²·D²) → KV cache.**
The naive implementation recomputed `LayerNorm + in_proj(3*D × D)` for every
past position at every timestep. For 100 frames upsampled to 1600 positions,
this was 10+ minutes. Added per-layer KV cache: compute K,V once per position,
attend from cache. Reduced to O(T²·D + T·D²).

**Bug 11 — SEANet decoder used symmetric padding, not causal.**
The Mimi SEANet uses streaming causal convolutions: `pad_left = eff_kernel -
stride, pad_right = 0`. The C++ used `pad = kernel/2` on both sides — same
output length but different values at every position. Fixed all Conv1d calls
(initial, residual, final) and ConvTranspose1d calls (trim right by K-S
instead of symmetric padding).

**Bug 12 — Upsample ConvTranspose1d still had symmetric padding.**
The depthwise upsample path (bug 4 fix) still computed `pad=(K-S)/2=8` and
subtracted it from both sides. The SEANet decoder ConvTranspose1d was fixed
(bug 11) but the upsample wasn't updated. Changed to: full convtr (no
padding) + trim right by K-S. **This was the final bug** — after fixing it,
teacher-forced PCM matched the reference to **cos=0.999997, maxabs=0.000131**.

### Diff methodology

The diff harness followed the protocol from `docs/CONTRIBUTING.md`:

1. **Trustworthy reference first.** Ran `pocket_tts_reference.py` to capture
   per-stage dumps (tokens, embeddings, latents, quantizer, upsample, PCM).

2. **Tokenizer verified first.** Exact token-ID match was prerequisite for all
   downstream stages.

3. **Teacher-forcing to isolate stages.** `POCKET_FORCE_LATENTS` fed reference
   latents through the C++ Mimi decoder, isolating decoder bugs from AR drift.
   `POCKET_FORCE_NOISE` fed reference noise through the flow head, isolating
   flow-net bugs from noise distribution differences.

4. **Per-stage cosine + norm tracking.** Added `POCKET_DUMP_DIR` dumps at every
   stage boundary. When PCM diverged, compared upsample (exact) → transformer
   (divergent) → identified causal padding as the issue.

5. **Bisect by bypass.** `POCKET_SKIP_MIMI_XFMR` bypassed the decoder
   transformer to isolate SEANet decoder from transformer. The SEANet alone
   produced similar-RMS output, confirming the transformer was the primary
   divergent stage (though it turned out to be the upsample feeding into it).

### Stage verification summary

| Stage | Cosine vs reference | Method |
|-------|-------------------|--------|
| Token IDs | 1.000000 (exact) | Python SentencePiece comparison |
| Text embeddings | 1.000000 (exact) | GGUF LUT lookup comparison |
| Backbone output (step 0) | 0.999990 | Hook reference out_norm output |
| Flow head ResBlock norms | ±0.05% | Per-ResBlock norm dump |
| Quantizer projection | 1.000000 (exact) | NumPy matmul comparison |
| Depthwise upsample | 1.000000 (exact) | PyTorch ConvTranspose1d comparison |
| Full PCM (teacher-forced) | 0.999997 | End-to-end Mimi decoder |

### Voice cloning

The `pocket-tts-without-voice-cloning` model has encoder weights in the
safetensors but they produce all zeros (untrained). Voice cloning requires the
gated `kyutai/pocket-tts` model. With the gated model:

- Mimi encoder: 264000 PCM samples (11s @ 24 kHz) → 137 latent frames
- speaker_proj: (137, 32) → (137, 1024)
- Prepend bos_before_voice → 138 conditioning frames in KV cache
- ASR roundtrip: **"Hello there, how are you doing today?" → exact match**

### Noise distribution difference

The reference uses `torch.nn.init.trunc_normal_(noise, std=temp**0.5, a=-clamp,
b=clamp)` — truncated normal with re-normalized PDF. The C++ uses
`std::normal_distribution * sqrt(temp)` then clamps. These produce different
samples even with the same seed, so free-running (non-teacher-forced) generation
diverges from the reference. This is acceptable — the model is stochastic by
design, and the per-stage verification (cos=0.999997 with forced inputs) proves
the math is correct.

### Files

- Runtime: `src/pocket_tts.cpp` (~2200 LOC), `src/pocket_tts.h`
- Adapter: `examples/cli/crispasr_backend_pocket_tts.cpp`
- Converter: `models/convert-pocket-tts-to-gguf.py`
- Reference: `tools/reference_backends/pocket_tts_reference.py`
- GGUFs: `cstr/pocket-tts-GGUF` (F16/Q8_0/Q4_K × with/without encoder)

## Pocket TTS — manual CPU → ggml compute graph rewrite

The original pocket-tts runtime used ~38 hand-written C++ functions (`linear_f32`,
`layer_norm`, `conv1d_eager`, `conv_transpose1d_eager`, `apply_rope_inplace`, etc.)
operating on raw `float*` buffers via `ggml_backend_tensor_get`. No SIMD matmul,
no GPU offload, no `ggml_backend_sched`.

### What was converted

| Stage | Before | After | Validation |
|-------|--------|-------|------------|
| Backbone (6L, 1024D, 16H AR transformer) | Manual per-element loops, manual KV cache | `ggml_mul_mat` + `ggml_rope_ext` + `ggml_flash_attn_ext`, host-side KV cache upload/download | cos=1.000000 backbone_out, cos=0.999999 PCM |
| Mimi decoder transformer (2L, 512D, 8H) | Manual attention + GELU FFN | `ggml_flash_attn_ext` + `ggml_norm` + `ggml_gelu`, causal mask with context=250 window | cos=1.000000 (max_diff=0.025) |
| Mimi SEANet decoder (3-stage CNN) | `conv1d_eager` / `conv_transpose1d_eager` element loops | `ggml_conv_1d` + `ggml_conv_transpose_1d` + `ggml_elu` + `ggml_pad_ext` | cos=1.000000 PCM (max_diff=2.3e-5) |
| Flow net (512D, 6 ResBlocks) | Manual (kept) | Manual (kept) | N/A — custom RMSNorm (Bessel-corrected variance) incompatible with ggml_rms_norm |
| Mimi encoder (voice cloning) | Manual (kept) | Manual (kept) | N/A — same architecture, convert when needed |

### Path selection

`--no-gpu` / `-ng` forces the legacy manual CPU path for both backbone and Mimi
decoder. Default (`use_gpu=true`) uses ggml compute graphs. Env var overrides for
per-stage debugging:

| Switch | Effect |
|--------|--------|
| (default) | ggml graphs for backbone + Mimi decoder |
| `--no-gpu` | Legacy manual CPU for all stages |
| `POCKET_MANUAL_BACKBONE=1` | Manual backbone only |
| `POCKET_MANUAL_MIMI=1` | Manual Mimi decoder only |

### Key implementation patterns

**Backbone AR KV cache (host-side upload/download pattern):**
Each AR step builds a fresh ggml graph. Past K/V are stored in `std::vector<float>`
on the host, uploaded as `ggml_set_input` tensors at the start of each step, then
new K/V are read back via `ggml_backend_tensor_get` after compute. Layout requires
reordering from host `[pos][head][dim]` to ggml `(HD, pos, NH)`.

**Causal conv1d with ggml_pad_ext:**
`ggml_conv_1d` only supports symmetric padding. For causal (left-only) padding:
```
x = ggml_pad_ext(ctx, x, pad_left, 0, 0, 0, 0, 0, 0, 0);
x = ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1]); // pad_ext emits 4D
x = ggml_conv_1d(ctx, w, x, stride, 0, dilation);
```

**Causal ConvTranspose1d trim:**
`ggml_conv_transpose_1d` with `padding=0` produces full output. Trim right to
`T_in * stride` for causal (keep earliest samples):
```
out = ggml_conv_transpose_1d(ctx, w, x, stride, 0, 1);
out = ggml_view_2d(ctx, out, T_want, Cout, out->nb[1], 0);
out = ggml_cont(ctx, out);
```

**Flash attention for single-query AR decode:**
With Q having T=1, `ggml_flash_attn_ext` needs no causal mask (`nullptr`). Q/K/V
layout: `(HD, T, NH)` after `ggml_permute(0, 2, 1, 3)` from `(HD, NH, T)`.

**Depthwise ConvTranspose1d (not in ggml):**
ggml lacks grouped transposed convolution. The Mimi upsample (`groups=512`,
`stride=16`) stays CPU-side: per-channel scalar loop, then feed into the ggml
graph as an input tensor.

### ASR roundtrip validation (ggml graph path)

The `-novc` model (`pocket-tts-english-novc-f16.gguf`) produces near-silence
(RMS≈0.003) regardless of parameters — its encoder weights are untrained (all-zero
output). **Voice cloning with the gated model is required for audible speech:**

```
crispasr --backend pocket-tts \
    -m pocket-tts-english-f16.gguf \
    --voice samples/jfk.wav \
    --tts "Hello there, how are you doing today?" --seed 42
```

Result: RMS=0.170, abs_max=0.698. Whisper ASR roundtrip:
`"Hello there, how are you doing today?"` → **exact match**.

This validates the full ggml graph pipeline end-to-end: voice conditioning
(Mimi encoder, manual) → backbone prefill (138 voice frames + 9 text tokens,
ggml) → AR decode (25 frames, ggml backbone + manual flow net) → Mimi decode
(ggml transformer + SEANet) → intelligible 24 kHz PCM.

### Why the flow net stays manual

The flow net's "RMSNorm" uses `var(x)` with Bessel's correction (`N-1` divisor)
and does NOT subtract the mean — unlike `ggml_rms_norm` which computes
`x / sqrt(mean(x²) + eps)`. For non-zero-mean vectors, these differ. The flow
net is small (512-dim, 6 depth, called once per AR step with `lsd_steps=1`) so
the speedup from ggml graphs would be minimal.

## MeloTTS (VITS2) — from zero to BERT conditioning (June 2026)

### Architecture

MeloTTS (myshell-ai) is a VITS2 model: 6-layer relative-position
transformer text encoder → dual duration predictor (SDP spline flows +
deterministic DP, blended via `sdp_ratio`) → 4-block TransformerCouplingFlow
(3 transformer layers per block, NOT WaveNet like Piper) → 5-stage HiFi-GAN
vocoder at 44.1 kHz. 52M params, 4 English speakers (EN_V2), MIT license.

### Key bugs found during implementation

**GGUF dimension reversal (conv weights):** PyTorch Conv1d weights are
`(Cout, Cin, K)`. The GGUF Python library reverses numpy dimensions to ggml
ne[], so storing the PyTorch array as-is gives `ne[0]=K, ne[1]=Cin, ne[2]=Cout`
— exactly what `ggml_conv_1d` expects. Initially I transposed them (like Piper's
ONNX converter), giving `ne[0]=Cout` which made ggml think the kernel was 768
elements wide. Fix: don't transpose conv weights for PyTorch→GGUF.

**BERT projection bias with zero input:** Even with zero BERT embeddings
(disable-bert mode), `Conv1d(zeros)` still produces the bias term. The combined
`bert_proj.bias + ja_bert_proj.bias` contributes up to 21.9 amplitude after
sqrt(192) scaling — this was the entire embedding mismatch. Fix: always add
both biases regardless of whether BERT features are available.

**Speaker injection order:** MeloTTS injects the speaker embedding BEFORE
attention (in the Encoder.forward loop), not inside the attention function.
Injecting inside attention meant the modified input was a working copy that
didn't propagate back. Moving injection to the main encoder loop brought all
6 layers to cos=1.0 vs Python reference.

**Intersperse blanks — leading/trailing:** Python's `intersperse` produces
`[0, e0, 0, e1, ..., eN, 0]` (blanks at both ends). My initial implementation
only put blanks between elements. Also: blank positions must have `lang_id=0`,
not the language ID.

**Flow FFN kernel size:** The text encoder FFN uses k=3 but the flow encoder
FFN uses k=5. Hardcoding pad=1 (correct for k=3) crashed the flow. Fix:
auto-detect kernel size from `ffn_c1_w->ne[0]`.

**v→V phoneme mapping:** MeloTTS uses uppercase "V" as a distinct phoneme for
/v/ (the voiced labiodental fricative). The CMU dict returns lowercase "v"
which must be mapped via `post_replace_ph`.

### BERT conditioning

bert-base-uncased (110M params, 12 layers) provides contextual phoneme
embeddings. MeloTTS uses hidden_states[-3] (layer 9 output) projected through
`ja_bert_proj` (768→192). For English, the 1024-dim `bert` input is zeros
(only bias contributes); the 768-dim `ja_bert` carries actual features.

**Implementation:** Standalone `bert_encoder.{h,cpp}` — 10-layer BERT
transformer forward pass using ggml graphs, WordPiece tokenizer embedded in
GGUF metadata, 238 MB GGUF. Loaded as companion model via `melotts_load_bert()`.

**word2ph alignment:** BERT tokenizes "seashells" as `["seas","##hell","##s"]`
(3 subwords) but G2P treats it as 1 word. The word2ph must map BERT subword
tokens to phoneme counts. Initial uniform distribution caused "seashore"→"Sichuan"
regression. Fix: `bert_encoder_word_subtokens()` counts BERT subwords per word,
then `distribute_phone` splits each word's phonemes evenly across its subwords.

**Quality impact:**
- Without BERT: "I enjoy reading books..." → "Send your reading books..."
- With BERT: "I enjoy reading books in the evening." → exact match
- ASR roundtrip: 4/6 perfect (up from 3/6 without BERT)
- Diff: enc_output cos=0.990 vs Python reference

### Weight precision

VITS2 tensors are mostly small (192-d embeddings, 192×192 attention). F16 is
the recommended storage format (93 MB GGUF). Q4_K/Q8 quantization is
ineffective — only 9/853 tensors meet the block-size threshold, and those
happen to be the most sensitive embeddings. Storing SDP/DP weights as F32
prevents spline transform instability but doesn't affect audio quality
(the real quality lever is BERT conditioning, not weight precision).

### OpenVoice2 ref_enc H/W swap

OpenVoice V2 TCC (voice cloning) uses the same VITS architecture for
voice conversion. The ref_enc (speaker embedding extractor) has 6 Conv2d
layers + GRU. Python processes the spectrogram as `(1, 1, T, freq)` but
the C++ was transposing to `(1, 1, freq, T)` before Conv2d — convolving
along the wrong spatial axes. Speaker embedding cos went from 0.17 to
0.999999 after fixing to match Python's layout.


## OpenVoice2 voice cloning — ggml data layout (2026-06-05)

**The bug:** HiFi-GAN decoder produced completely wrong output (cos=0.002
vs Python) despite identical z input. All upstream stages (STFT, ref_enc,
enc_q, flow) matched at cos≥0.999996.

**Root cause:** ggml tensor `(C, T)` has `ne[0]=C` as the fastest-changing
dimension. Data in `(T, C)` row-major layout already has C as the fast
dimension — it can be fed directly to `ggml_backend_tensor_set`. Adding a
`C*T+t ↔ T*C+c` transpose before setting **reverses** the correct layout.

**The lesson:** ggml's dimension ordering is the reverse of numpy/PyTorch.
A ggml tensor with `ne = {C, T}` stores data as `[c + t*C]` — identical
to C row-major `float data[T][C]`. When your CPU array is `(T, C)` row-major,
set it directly. Only transpose when the CPU array is `(C, T)` column-major.

**Debugging method:** Manual conv in Python using GGUF weights (bypassing
ggml) proved the weights were correct. Then stage-by-stage diff harness
(`OV2_DUMP_DIR` + `ggml_set_output` on intermediate tensors) isolated the
divergence to the first ggml op (conv_pre). The manual conv matched PyTorch
at cos=1.0 while ggml gave cos=0.002 — proving the data layout was wrong.

## F16 precision loss in deep WaveNet stacks (2026-06-04)

Storing WaveNet weights as F16 causes accumulated precision loss through
16 sequential layers. For enc_q: F16 gave z std=1.53 (should be 0.77),
completely wrong latent values → silent HiFi-GAN output. Fix: store
enc_q/flow/ref_enc weights as F32 in the GGUF (converter flag
`is_sensitive`). Decoder weights can stay F16 (only 4 upsample stages,
less accumulation).

## Cohere flash-attn crossover (2026-06-04)

flash_attn_ext wins on long sequences (O(n) vs O(n²)) but loses on
short sequences due to higher per-op overhead. Cohere with 30s auto-chunking
always hits the short-form case. Measured: flash +13% slower at 60s,
-26% faster at 300s. Crossover between 1-5 min. Default flipped to
cast-on-read for cohere; `CRISPASR_COHERE_FLASH=1` for unchunked long-form.

## MeloTTS + BERT quantization — what works and what doesn't

**Main model (VITS2, 102 MB F16):** Only 9 of 853 tensors meet ggml's
minimum block size for quantization, and those 9 are embeddings (phone,
tone, language, speaker). Q8_0 saves 1 MB (98→97). Not worth it for size,
but works correctly after fixing `read_emb` to dequantize via
`ggml_get_type_traits()->to_float` (was blindly memcpy'ing Q8_0 bytes as
F32). The same fix applies to the speaker embedding read in
`melotts_synthesize` (replaced inline F16/F32 branch with `read_tensor_f32`).

**BERT companion (bert-base-uncased, 227 MB F16):** All 141 MB of F16
linear weights (768×768 attention, 768×3072 FFN) quantize successfully.
Q8_0 = 97 MB, Q4_K = **52 MB** (77% smaller) with zero quality loss on
ASR roundtrip. Required fixing `bert_encoder.cpp` to `ggml_cast(F32)`
after `ggml_get_rows` on quantized embeddings, since `ggml_add` requires
F32 operands.

**Recommended deployment:** MeloTTS F16 (102 MB) + BERT Q4_K (52 MB) =
**154 MB total**. All three BERT quants (F16/Q8/Q4K) uploaded to
`cstr/melotts-en-v2-GGUF`. Registry companion points to Q4_K by default.

## AudioSeal ggml port — ggml_pad_ext convention trap

**The bug:** `ggml_pad_ext(ctx, x, lp0, rp0, ...)` has **reversed
parameter semantics**: `lp0` is actually RIGHT padding of dim 0, `rp0`
is LEFT padding. Only surfaces when padding is asymmetric (AudioSeal's
ratio=5 downsample: K=10, stride=5 → pad_left=2, pad_right=3). With
wrong convention, all subsequent encoder stages diverged (cos=1.0 →
cos=0.36 at the first asymmetric layer).

**How found:** Stage-by-stage binary dump comparison against PyTorch.
Every layer before the asymmetric padding was cos=1.0. Swapping the
`lp0`/`rp0` arguments immediately restored cos=1.0.

**Decoder too:** `ggml_view_2d` crop offset for ConvTranspose1d output
uses `pad_right * sizeof(float)` (not `pad_left`) because dim 0's
start is at the "right" end in ggml's reversed convention.

**Lesson:** Never trust parameter names in low-level tensor libraries.
Always verify with a concrete asymmetric test case. The `ggml_pad()`
wrapper masks the issue because it calls `ggml_pad_ext(ctx, a, 0, p0,
...)` — passing right-pad as `rp0`, which is actually the LEFT slot.

**LSTM unrolling:** Proper recurrent LSTM works in ggml by unrolling at
graph construction. 50 steps × 12 ops × 2 layers = ~1200 nodes. Initial
state via `ggml_scale(x[:,0], 0)`. Outputs concatenated via
`ggml_concat(dim=1)`. Graph size must be ≥8192 (default 2048 insufficient).

**F16 vs F32 red herring:** Weight quantization made zero difference to
parity (both cos=1.0). When debugging divergence, always check runtime
logic first — quantization error is almost never the cause.

## conv_transpose_1d GPU TDR — naive loop is O(IL), not O(K/s0) (June 2026)

**The bug:** `GGML_OP_CONV_TRANSPOSE_1D` caused GPU driver TDR crashes
(Timeout Detection and Recovery) on AMD RX 7900 XTX and NVIDIA RTX
5060 Ti when called at TTS codec scale (issue #155).  Root cause: every
output thread iterated all `IL` input positions with an if-continue to
find the ≤`⌈K/s0⌉` that actually contribute.  For IL=400, K=16, s0=8
that is ~200× wasted iterations per thread.  With 800 K output threads
the full dispatch was ~320 M conditional iterations — enough to exceed
the OS GPU watchdog (~2 s on Windows, ~1 s on Linux with DRM).

**The workaround that shipped first:** Force `supports_op` to `return
false` on all GPU backends so the scheduler falls back to CPU.  This
stops the crash and still gives ~50% TTS speedup vs all-CPU because
only this one op moves to CPU while the rest of the codec graph stays
on GPU.  Commit `68732b58`.

**The proper fix:** Compute `i_min`, `i_max` analytically before the
inner loop:
```
i_max = min(idx / s0, IL − 1)
a = idx − K + 1;  i_min = a ≤ 0 ? 0 : (a + s0 − 1) / s0
```
Loop iterates `[i_min, i_max]` only — at most `⌈K/s0⌉` steps.
Bit-identical output (same multiply order, no new FP rounding).
Applied in commit `f8fc8b8e`.  The identical fix had already merged
upstream for the Metal kernel in ggml#1477 (2026-05-10).

**Backend-specific status after f8fc8b8e:**
- Metal: kernel already had the fix from PR #04 (merged ggml#1477);
  `supports_op` re-enabled.
- CUDA: loop fix applied; `supports_op` re-enabled (F32 + F16 weights).
- Vulkan: shader uses a different shared-memory tiling algorithm that
  doesn't have the naive loop; `supports_op` re-enabled (F32 only).
- SYCL/CANN: still disabled — same naive loop in `ggml-sycl/conv.cpp`,
  no test coverage in this repo.

**Upstream PR plan:** PR #17 (`tools/upstream-prs/17-*`) — CUDA loop
fix for llama.cpp, gated on PR #14 (F16 template) merging first.

**General lesson:** Any ggml GPU kernel that iterates a full dimension
with an if-continue can hit TDR at model scale.  Profile wasted
iterations before enabling GPU support for new ops.

---

## GPU weight mirrors for mixed legacy / graph codebases

**Problem:** A backend with partially graph-ified code (some ops via
ggml cgraph, some via direct `tensor->data` dereference) works on
unified-memory GPUs (Metal/Apple Silicon) but SIGSEGVs on discrete GPUs
(Vulkan, CUDA) where the buffer is device-local.

**Wrong approach:** `ggml_backend_sched` with weights on CPU. The
scheduler copies ALL referenced weights from CPU to GPU on every
`ggml_backend_sched_graph_compute()` call — no caching, no
change-detection. For a 2.4 GB model with 252 weight tensors per TSLM
step, that's ~200 ms of PCIe traffic per AR step — slower than CPU-only.

**Correct approach:** Load weights to CPU, create GPU mirror copies at
init (one-time `ggml_backend_tensor_copy`). Store both sets:
`ctx->weights` (CPU, for legacy paths) and `ctx->gpu_weights` (GPU, for
graph-build functions). Graph-build functions call `ctx->graph_weights()`
which returns the GPU set when mirrors exist, else the CPU set. Graph
compute runs entirely on one backend — no cross-backend copies at
runtime. Memory cost: ~2× model size, acceptable on 8+ GB GPUs.

**Corollary:** Graph-ify as many paths as possible to maximise GPU
coverage. For voxcpm2, TSLM (28L) + RALM (8L) + LocDiT (12L) + LocEnc
(12L) + VAE decode now all run as GPU graphs. The remaining CPU-only ops
(FSQ, stop, fusion, ~5-8 ms/step) are too small to justify graph-build
overhead.

### LFM2-Audio hybrid conv+attention backbone — 2026-06-11/12

**Problem:** The LFM2 backbone has two layer types (10 ShortConv + 6 GQA
attention) interleaved in a fixed pattern `ccaccaccacacacac`. Standard KV
cache only works for the attention layers. The conv layers need their own
state management.

**Conv state caching:** Each ShortConv layer uses a depthwise causal
conv1d with kernel_size=3. For T=1 decode, the conv needs the last K-1=2
Bx columns from previous steps. Solution: store Bx columns in CPU-side
float arrays (160 KB total for 10 layers × 2048 × 2). During decode:
load cached columns → concat with new Bx via `ggml_concat` → run actual
`ggml_conv_1d_dw` on K=3 tokens → take last output. Shift cache
left by 1 after each step. (Originally used `ggml_conv_2d_dw_direct` with
4D reshape/permute; migrated to `ggml_conv_1d_dw` which is cleaner and
has native CUDA dispatch via im2col+mul_mat.)

**Snapshot extraction:** To capture the Bx columns after prefill, compute
Bx in a parallel graph branch (doesn't affect the main conv path, just
duplicates the `in_proj` + elementwise multiply). Name the snapshot tensor
with `ggml_set_name`, retrieve after graph eval with `ggml_graph_get_tensor`
(or `ggml_backend_tensor_get` for gallocr-allocated graphs).

**Causal mask bug:** `ggml_flash_attn_ext` with `mask=nullptr` does NOT
default to causal attention. It does FULL (non-causal) attention. For the
LFM2 backbone, every attention layer diverged without an explicit causal
mask. The fix was trivial — build a (T, T) F16 mask with 0/−inf — but the
debugging took hours because the divergence only showed at cos≈0.37 after
16 layers, not as a crash. The per-layer diff harness (`lfm2_ao_layer_K`)
was essential: it showed conv layers (0,1) were fine but layer 2 (first
attention) dropped, proving the mask was the issue.

**Depthformer KV cache:** The depthformer (6L, 1024-dim) generates 8
codebooks per audio frame. `core_attn::kv_self_attn` with fused QKV
produced degenerate output (all codebooks identical). The manual KV cache
approach works: CPU-side K/V arrays, `ggml_concat` to build full K/V
from cache + new, `ggml_flash_attn_ext` for attention. The QKV split in
`kv_self_attn` should be compatible (checked: Q|K|V order matches) but
something else in the helper's complex logic (GQA expansion, cache write
path) doesn't work for this tiny (T≤8) use case. Manual KV cache is
simpler and correct.

**Mel filterbank:** The liquid-audio conformer uses librosa's slaney-
normalized mel filterbank. CrispASR's `core_mel` uses HTK (non-slaney).
The fix: store the slaney filterbank in the GGUF (via the converter) and
load it at runtime. This gives cos=0.9999 mel match. The HTK filterbank
gave cos=−0.1 (essentially wrong).

**BPE merges:** Without merges, the per-byte BPE tokenizer tokenizes
"こんにちは" as 15 tokens (5 chars × 3 bytes). With merges, it tokenizes
correctly as 4-5 tokens. Merges need to be stored in the GGUF
(`tokenizer.ggml.merges`) and loaded at runtime for `core_bpe::bpe_one`.

**ggml_gallocr vs fixed buffer:** The 2 GB compute_meta buffer caused
heavy page-fault overhead (sys time ~1 min). Migrating to `ggml_gallocr`
for the decode step reduced sys time to ~7s. For the prefill, reducing
to 256 MB (still bump-allocated) was sufficient since the graph actually
uses ~200 MB. The gallocr approach is: `no_alloc=true` in `ggml_init`,
`ggml_gallocr_alloc_graph` after graph build, `ggml_backend_tensor_set/get`
for inputs/outputs.

**GPU readiness:** The gallocr paths ARE GPU-compatible. With
`backend = ggml_backend_init_best()`, weights load on GPU, gallocr
allocates intermediates on GPU, and `ggml_backend_graph_compute(backend, gf)`
runs on GPU. The remaining `compute_meta` (bump-allocated) paths need
gallocr migration for full GPU coverage. Kaggle T4 test confirmed the
CUDA build works and produces correct output.

**Detokenizer companion GGUF:** TTS output requires a separate model for
codes→PCM. The detokenizer path searches: exact match → strip quant
suffix + try F16 → strip quant suffix + try base. This makes quantized
models find the F16 detokenizer without needing per-quant detokenizer files.

## beam_size default — greedy vs beam-5 (issue #161, 2026-06-12)

### Problem

`whisper_params.beam_size` defaulted to 5 (from `whisper_full_default_params`
at compile time). This propagated to ALL backends via the shared `whisper_params`
struct. For whisper itself this was intentional (beam search improves quality),
but for TDT/RNNT backends like parakeet it caused a 4.5× latency regression
with zero WER benefit — the beam-5 default silently switched every backend
from greedy to beam search.

### Fix

Default `beam_size = -1` (greedy). All whisper dispatch sites clamp
`beam_search.beam_size` to `max(bs, 5)` for grammar-forced beam search.
Non-whisper backends already check `beam_size > 0` and fall to 1 (greedy)
or their own default (firered uses 3).

### Takeaway

Shared param structs that set defaults from one backend's perspective
silently affect all others. Defaults should be "unset" (-1) with per-backend
policy, not one-size-fits-all.

## ggml_graph_get_tensor hash invalidated by ggml_gallocr (issue #164, 2026-06-12)

### Problem

`ggml_graph_get_tensor(gf, "name")` uses an internal hash table for O(1) lookup.
`ggml_gallocr_reserve()` and `ggml_gallocr_alloc_graph()` both call
`ggml_gallocr_alloc_graph_impl()` which resets this hash table (line 790:
`ggml_hash_set_reset`). After that, name-based tensor lookup silently
returns nullptr even though the tensor is still in the graph's node list.

### Symptoms

In the voxcpm2 TSLM graph, the "stop_probs" output tensor was found on the
first call (before reserve), but returned nullptr on all subsequent calls.
The stop predictor fell back to CPU computation, which diverged due to
`ggml_flash_attn_ext` vs scalar attention precision differences, and the
stop never fired.

### Fix

Cache tensor pointers at build time (after `build_tslm_step_graph` but
BEFORE `ggml_gallocr_reserve`). Store in the bucket struct and reuse on
subsequent calls without any name-based lookup.

### Takeaway

**Never call `ggml_graph_get_tensor` after `ggml_gallocr_reserve` or
`ggml_gallocr_alloc_graph` on a cached/reused graph.** The hash is
destroyed. Look up tensors immediately after graph construction and cache
the pointers. This affects any code that reuses ggml graphs across calls
(bucket patterns, step-graph caching, etc).

## flash_attn_ext vs scalar attention — numerical divergence (2026-06-12)

### Problem

`ggml_flash_attn_ext` (tiled softmax, SIMD accumulation) and hand-coded
scalar `causal_attn_step` (sequential dot product + softmax + weighted sum)
produce different FP32 results due to reduction order. On Q4_K weights with
28 transformer layers, the max_diff was ~0.09 at step 0, compounding to
divergence across 64 AR steps.

### Impact

Any code that mixes graph-path hidden states with legacy CPU-path
operations (like computing stop_score via `matmul_mv` on graph-produced
hidden) will see compounding divergence. The stop predictor in voxcpm2
never fired because the graph's hidden trajectory was numerically
different enough that the CPU-computed stop score stayed below threshold.

### Fix

Compute the stop predictor entirely inside the graph (stop_proj + SiLU +
stop_head + softmax as a graph output tensor). This ensures the stop
decision uses the same numerical path as the hidden state.

### Takeaway

**Don't mix graph-computed intermediates with legacy CPU operations for
critical decisions.** If a graph produces hidden states, any downstream
computation (stop prediction, scoring, etc.) should also be in the graph.
The precision difference between `ggml_mul_mat` (SIMD/BLAS) and scalar
loops is small per step but compounds exponentially across autoregressive
decode steps.

## VibeVoice TTS: missing BPE merges still need BPE behavior (issue #171, 2026-06-18)

### Problem

The VibeVoice Realtime GGUF in the local cache had `tokenizer.ggml.tokens` but
no `tokenizer.ggml.merges`. Falling back to greedy longest-vocab matching is not
equivalent to Qwen BPE: the issue text `1. La genèse : L’inversion des rôles
parentaux` produced different token IDs from the official Python tokenizer,
which shifted the positive TTS condition and surfaced as accented/garbled speech.

### Fix

Load `tokenizer.ggml.merges` when present and use the shared BPE tokenizer. For
older GGUFs without merges, do a vocab-rank BPE fallback: split the byte-encoded
word into Unicode byte symbols, repeatedly merge adjacent symbols whose
concatenation exists in the vocab, and rank candidates by the merged token ID.
Append VibeVoice's trailing newline token explicitly.

Voice-conditioned Realtime also needs to match the official streaming loop: run
the base LM over text in 5-token windows using the cached voice KV, and feed the
next TTS text window before the following speech window. Processing all text at
once or moving the text window after speech generation creates hidden-state drift.

### Takeaway

If TTS diffusion noise is exact and the first divergence is the positive TTS LM
condition, don't start by chasing VAE ops such as `col2im_1d` or
`conv_transpose_1d`. Check tokenizer parity, required sentinel tokens, and
voice-cache scheduling first.

On macOS, platform-specific tests must guard on the backend target they link
against, not only `APPLE`. A Vulkan-on-MoltenVK build is still Apple, but it does
not provide `ggml-metal`; Metal-only repro tests should use
`if(APPLE AND TARGET ggml-metal)`.

## VibeVoice TTS start clicks: distinguish decoder PCM from CLI post-processing (issue #171, 2026-06-18)

### Problem

Recent VibeVoice realtime WAVs had loud clicks at the start, while the Python
reference decoder output did not. The first instinct was to chase decoder
parity (`conv_transpose_1d`, `col2im_1d`, or the upstream cached streaming
decoder), but a latent-controlled comparison showed the C++ VAE raw output was
clean and numerically matched the official PyTorch decoder:

- Python decoder from saved latents: first samples around `0.005`, `peak250ms`
  around `0.019`, `maxjump250ms` around `0.0013`.
- C++ `tts_raw_audio` from the same latents: same clean start.
- User-visible CLI WAV before the fix: first nonzero sample could jump to
  `0.2` to `1.0`.

The click came after VAE decode:

1. VibeVoice realtime used a fixed 2400-sample start trim. That skipped the
   clean initial decoded chunk and could land directly on a later waveform peak.
2. The built-in spread-spectrum watermark was global CLI/server post-processing.
   Its overlap-add normalization could be tiny at Hann-window boundaries, which
   amplified boundary samples and created an impulse. This affected any backend
   using the fallback spread-spectrum watermark, not just VibeVoice.

### Fix

- Preserve VibeVoice realtime's decoder start unless there is actual leading
  digital silence; use only floor-based trim with attack margin.
- Write the spread-spectrum watermark back as a ramped delta and ignore
  under-covered boundary samples.
- Keep VibeVoice debug hooks env-gated:
  `VIBEVOICE_TTS_LATENTS` replays a raw float32 latent stack,
  `VIBEVOICE_TTS_DUMP` writes `tts_scaled_latent` and `tts_raw_audio`, and
  `VIBEVOICE_TTS_DUMP_DECODER=1` adds decoder stage dumps.

### Takeaway

When a generated WAV clicks but a reference decoder does not, diff every
post-decoder stage too: backend PCM, backend trim/fade, watermark/provenance
mutation, then file serialization. A clean `tts_raw_audio` plus a broken WAV is
not a model-runtime bug; it is post-processing.

## VibeVoice TTS garbles only on AMD RDNA4: coopmat2 flash-attention (issue #171, 2026-06-19)

### Problem

After the tokenizer/scheduling fixes, the reporter (AMD Radeon RX 9700 XT,
Vulkan, Ubuntu 26.04) still got garbled VibeVoice realtime TTS on short French
texts — "Cependant", "Le wagon vert.", "Cependant, il a tort." — described as
mixed voices, repeated fragments, "goes totally west". Longer texts mostly
worked.

### What it was NOT

Reproduced locally on **Metal (M1)** and **Vulkan via MoltenVK (M1)** using the
*exact same* `src/vibevoice.cpp` as the reporter's commit: all three texts
ASR-roundtrip **verbatim** on both. So it is not the TTS loop logic, tokenizer,
EOS threshold, or `n_frames` cap. The bug is backend-numerical and specific to
the reporter's GPU.

### Root cause

ggml-vulkan picks `FA_COOPMAT2` for `ggml_flash_attn_ext` whenever
`device->coopmat2` is advertised (ggml-vulkan.cpp ~L3095). The RX 9700 XT is
RDNA4 — brand-new silicon whose RADV coopmat2 flash-attention shader miscomputes
attention, yielding slightly-wrong TTS-LM hidden states. The binary EOS
classifier (`sigmoid(fc2(relu(fc1(h)))) > 0.5`) is extremely sensitive to that
drift, so it stops misfiring → the realtime model over-generates past the
natural end → speaker identity drifts and fragments repeat. Short utterances
break worst because they depend entirely on EOS firing early; long ones are
re-anchored every 5-token text window. MoltenVK reports `matrix cores: none` →
it uses the **scalar** FA path, which is why it (and Metal) are clean — so my
local "correct" runs already validated the non-coopmat2 path end-to-end.

### Fix / instrumentation

`vibevoice_sdpa()` wraps the five TTS-LM attention sites; `VIBEVOICE_TTS_FLASH_ATTN=0`
swaps fused FA for an explicit `softmax(QKᵀ·scale + mask)·V` (GQA via
`ggml_mul_mat` broadcast + `ggml_soft_max_ext`), avoiding the FA shader. The
σ-VAE decoder has **no attention** (pure conv/col2im), so `VIBEVOICE_VAE_BACKEND`
isolates that half — the two env knobs bisect the TTS GPU graph. No-rebuild
equivalents for the reporter: `GGML_VK_DISABLE_COOPMAT2=1` (FA path),
`GGML_VK_DISABLE_COOPMAT=1`, `GGML_VK_DISABLE_F16=1`.

### Takeaway

When a TTS/LM model garbles on one specific GPU but is verbatim on every backend
you can test, suspect a fused-kernel shader miscompile (flash-attention coopmat2
on bleeding-edge AMD/Intel), not your graph. Keep an env-selectable non-fused
fallback for fused ops so a reporter can bisect on hardware you don't have. EOS
classifiers amplify tiny attention drift into gross over-generation.

## read_tensor_f32 weight pre-cache for VITS-family TTS (2026-06-20)

Piper and MeloTTS use CPU-scalar C++ for flow_inverse, wavenet, SDP,
attention — each reads weights via `read_tensor_f32` which does
`ggml_backend_tensor_get` + F16→F32 dequant. 39-48 calls per synthesis
add up to 14-16% of total time. Fix: pre-populate
`unordered_map<tensor*, vector<float>>` at init, check in
`read_tensor_f32` via module-level context pointer. Cost: ~2× model RAM
(30-100 MB range). Gated by `CRISPASR_PIPER_WEIGHT_CACHE` /
`CRISPASR_MELOTTS_WEIGHT_CACHE`. Lesson: small helpers that look
innocuous (`read_tensor_f32`) accumulate significant overhead when called
40-50 times per synthesis.

## Single-token embed graph elimination for LLM-ASR (2026-06-20)

AR decode in LLM-ASR backends (funasr, glm_asr, etc.) called
`embed_tokens({next_id})` per step, building a full ggml graph for a
single GET_ROWS. Fix: n==1 fast path dequants one row directly. Gated
by `CRISPASR_XXX_EMBED_FAST`. Lesson: not every ggml op needs a graph —
single-row lookups are cheaper as direct tensor reads.

## §176 runtime optimization audit methodology (2026-06-20)

### Full code-read survey beats pattern-grep

Dispatching 9 parallel research agents to read every runtime file
(65+ backends) line-by-line found optimizations that pattern-grep would
have missed: host-side KV caches masquerading as "caches" but actually
re-uploading every step (SpeechT5, Dia, Parler), scalar LSTM loops that
should be BLAS (Parakeet, Nemotron), unconditional debug fprintf in hot
paths (Paraformer), and recursive FFT with O(N log N) heap allocs
(core/fft.h). The audit produced 20 actionable items, 15 of which were
completed in one session.

### Mechanical optimization patterns that scale

Four patterns covered 90% of the work:

1. **arena_ctx pattern for encoder graph caching**: add `arena_ctx`
   param to `build_graph_*`, cache the ggml_context + cgraph on the
   runtime context struct, key by the input shape (T_mel, n_samples,
   etc.). Applied to 16 backends in ~15 min each. The key insight: ggml
   tensor descriptors survive `ggml_free(ctx0)` if backed by
   `compute_meta`; for caching, use a **separate** meta buffer so
   compute_meta stays available for other graphs.

2. **Lk-bucketed AR decode graph cache**: pre-build T=1 decode graphs
   at fixed KV lengths {512,1024,2048,4096}; pick the smallest bucket
   >= n_past+1; use `kv_indices=positions` to make KV write position a
   runtime input. Applied to 8 backends (Orpheus, OuteTTS, Zonos, TADA,
   Chatterbox, CosyVoice3, VibeVoice, Qwen3-TTS).

3. **Static context cache with mutex**: for support runtimes called
   per-segment (VAD, LID, aligner, diarization), cache the loaded
   context in a `static std::mutex + void* + std::string path` triple.
   Same pattern as Silero VAD #132 fix.

4. **thread_local scratch buffers**: for per-call heap allocs in shared
   hot paths (FFT scratch, greedy decode probs vector, kaldi_fbank
   frame buffer). `thread_local static` + lazy resize eliminates the
   malloc/free per call.

### Backends with host-side KV resist bucketing

SpeechT5, Dia, Parler, Pocket-TTS, VoxCPM2 all use
`std::vector<float>` KV caches that grow and re-upload every step.
Bucket caching doesn't help because the KV data size changes every
step regardless of graph topology. These need the §176c migration
(device-resident 4D KV with ggml_view_4d + ggml_cpy writes) before
bucketing is effective.

### Embedding table size gates CPU cache feasibility

Orpheus (vocab 157K × d 4096 = 2.4 GB), TADA (128K × 3072 = 1.5 GB),
and OuteTTS (57K × 2048 = 449 MB) are too large for a full F32 CPU
embedding cache on 8 GB machines. The n==1 direct-dequant fast path
(§176o) is the right approach for large vocabs — one row at a time,
no full table copy.

### Graph/alloc caching only helps overhead-bound graphs — measure host build+alloc first (§208)

Before porting a graph to a cached raw-`ggml_gallocr` single-backend path to kill
per-step rebuild+alloc overhead, measure that overhead directly. For the chatterbox
S3Gen CFM (3148-node batch-2 DiT, T_mel=484, F16-dequant on Metal): host-side
`build_graph + sched_reset + sched_alloc_graph` = **~4–7 ms/step** out of **~1887
ms/step** = **0.3%**. The per-step cost is raw Metal GEMM (compute-bound), so
caching the graph + allocation — though architecturally cleaner (single backend,
zero splits, graph reused, parity 0.999) — bought nothing. The §207 handover's
"per-step is almost all overhead" assumption was never measured; a one-line
`ggml_time_us()` around the build+alloc would have pre-empted the whole port.
Heuristic: a graph-reuse/bucket cache pays off only when host-side
build+alloc+(GPU↔CPU split copies) is a *meaningful* fraction of per-step wall
time; for a big GPU-resident GEMM graph it is single-digit ms and the lever is a
DUD (cf. the bucket-floor DUD in [[project_chatterbox_t3_decode_perf]]). The
single-backend gallocr path is also a Bug-B-free reference: no sched, no
cross-backend copy boundary, so CFG uncond cannot diverge.

**§214 re-confirmation (chatterbox T3 B=2 decode).** Same trap, second data point:
before bucketing the per-step-rebuilt B=2 T3 graph, `CHATTERBOX_BENCH_B2=1` split
each step into build+alloc vs compute → **CPU 0.41 ms/step (0.8%)** of a 53 ms
step, **GPU+F16 1.55 ms (0.7%)** of a 232 ms step. <1% on both → a cached/bucketed
B2 graph is a DUD, and worse, it would risk reintroducing the §186 Lk-bucket
`buffer is nil` Metal crash that the per-step rebuild *deliberately* sidesteps
(the rebuild is what makes B2-on-GPU work at all). The instrumentation is shipped
(gated `CHATTERBOX_BENCH_B2`) so the call is data-backed, not assumed. Corollary
the same run surfaced: GPU per-step compute (232 ms) ≫ CPU (53 ms), so B=2 does
**not** make T3-on-GPU beat the CPU default — B2's GPU value is purely *enabling*
T3-on-GPU (sidestep the bucket crash) + the GPU+quant F16-dequant path.

### RNNoise/resampler caching isn't worth breaking thread safety

The miniaudio resamplers and RNNoise DenoiseState are per-call by
design (the code comment says "multiple worker isolates can run
concurrently"). The init/free overhead is ~microseconds (small
malloc). Caching them as statics would break concurrent server use
for negligible gain.

### TADA TTS: prompt-phase time embeddings must be zero (not FM-predicted)

During the prompt-phase prefill loop, C++ runs the FM network at every step
to keep the KV state moving — but the resulting `pred_t_before`/`pred_t_after`
must NOT be fed back as the time embedding for the next step while still in the
`in_prefix` region.  Python's batch prefill zero-initialises the time tensor
and only overwrites positions `shift+1..shift+n_t` with the actual prompt time
values; all prefix positions (`feat_idx 0..prefix_len-1`) get zero time
embeddings.

**Bug pattern:** `cur_t_before = pred_t_before` in the `in_prefix` branch of the
AR loop → non-zero time embeddings corrupt the KV state → divergent duration
predictions → wrong phonemes ("Costella" instead of "Stella").

**Fix:** `cur_t_before = 0; cur_t_after = 0;` for all in_prefix steps.
`in_prompt` steps (feat_idx ≥ prefix_len) correctly use
`ctx->prompt_time_before[prompt_feat_idx + 1]` which is indexed from 1 to
match Python's `time_len_before[k]` alignment.

**General principle:** for any TTS model with a prompt-phase prefill, verify
that the EXACT same inputs (token, acoustic, time) are fed step-by-step as the
model's batch prefill would use.  Run a diff harness that compares the hidden
state after step 0, 1, prefix_len, and prefix_len+1 before chasing output
divergence.

### TADA TTS: time transition is one step earlier for time than for acoustic

Python's TADA switches from reference-audio time → FM-predicted time one step
earlier than it switches acoustic features.  Specifically:

- **Acoustic** uses the reference prompt through `feat_idx < prompt_used_len`
- **Time** (t_before/t_after) uses the reference prompt only through
  `feat_idx < prompt_used_len - 1`; at `feat_idx = prompt_used_len - 1` the
  time is already FM-predicted even though the acoustic is still from the
  reference

Python's condition: `step - shift_acoustic < prompt_time_len_before.shape[1] - 1`,
where `shape[1] = prefix_len + n_prompt - transition_steps = prompt_used_len`.
So `feat_idx < prompt_used_len - 1` for prompted time; predicted otherwise.

**Bug pattern:** C++ `in_prompt` branch applied the same boundary for both
acoustic and time (`feat_idx < prompt_used_len`), so the last `in_prompt` step
used reference time where Python was already FM-predicted.  The single-step
mismatch in `cur_t_before` at step `shift + prompt_used_len - 1` cascaded
through all subsequent FM evaluations, inflating the last decoded
`time_before` value from ~21 to 234 (= 4.68 s of trailing silence) and
ballooning a 1.36 s synthesis to 9.90 s.

**Fix:** `bool use_prompted_time = (feat_idx < prompt_used_len - 1)` inside
the `in_prompt` branch.  Acoustic selection is unchanged; only the time
variables (`tb`, `ta`, `cur_t_before`, `cur_t_after`) switch one step earlier.

**Lesson:** when acoustic and time have separate "replay vs predict" boundaries,
they must be tracked independently.  A single off-by-one in a time embedding
several frames before the decoded zone can still corrupt all decoded frames via
KV-state drift.

### TADA TTS: `tada-ref.gguf` is both a prompt cache and a poor full diff oracle

`tada-ref.gguf` contains the prompt tensors the runtime needs
(`prompt_token_values`, `prompt_token_positions`, plus prompt transcript
metadata).  That makes it a valid **voice prompt cache** for
`tada_load_prompt()`.  It does **not** make every published `tada-ref.gguf`
a trustworthy full end-to-end generated-output reference.  Stale refs can
still contain prompt tensors that are useful for synthesis while their
`codec_input`, `codec_pcm`, `time_before`, or generated acoustic tensors no
longer match current Python/reference generation.

**Bug pattern:** treating a passing codec-only diff against a stale ref as proof
that generated timing is correct.  In issue #192, stale refs hid the real
tempo problem because they did not force the harness to compare the generated
`time_before` sequence and full acoustic matrix from the same Python run.

**Fix:** keep runtime prompt loading scoped to prompt tensors, but make the
TADA diff harness compare generated `acoustic_features` and generated
`time_before` whenever those tensors exist in the reference archive.  For exact
timing work, regenerate a fresh full ref from the original Python model/weights;
do not infer timing correctness from prompt-only GGUFs.

### TADA TTS: multilingual lives in prompt encoding, not generation

The Python blueprint does multilingual TADA by selecting a language-specific
aligner when building the prompt:

- `Encoder.from_pretrained(codec_repo, subfolder="encoder", language="fr")`
  loads `aligner-fr`
- `Encoder(audio, text=[transcript], sample_rate=...)` forced-aligns the
  reference audio to the supplied transcript
- `model.generate(prompt=prompt, text=...)` then consumes only
  `prompt.token_values`, `prompt.token_positions`, and `prompt.text`; it has no
  separate generation-time language flag

**Runtime implication:** `-l fr` cannot magically make a default English prompt
French.  It can only select a pre-encoded sibling prompt such as
`tada-ref-fr.gguf`.  Custom multilingual voices must first be converted with
`models/convert-tada-ref-to-gguf.py --language fr --transcript ...`, which
matches the blueprint's `aligner-fr` path.

**Practical rule:** for TADA, fix language issues in the prompt-conversion
pipeline, not in `tada_synthesize()`.  The C++ runtime should load the right
prompt GGUF and then follow Python generation exactly.

### CosyVoice3: fixed-shape graph reuse must not imply a maximum-size KV graph

CosyVoice3 caches its single-token AR graph because rebuilding 24 Qwen2 layers
per speech token is expensive. The graph shape includes the KV-cache length,
so the original implementation used a fixed 4096-token cache to make every
step reusable. On Metal this made a short request attend over thousands of
unused slots at every step: a 17-token decode took about 6.8 s on an M1.

Size KV storage and the cached graph from `prefill_tokens + max_new_tokens`
before prefill. Direct prefill/step API calls still need a modest fallback,
but they must preserve an already-sized cache instead of expanding it back to
4096. With a 625-slot default CLI budget, the same decode took about 1.4 s and
the final WAV was byte-identical.

Two adjacent rules matter:

- Plumb the CLI `-n/--max-new-tokens` value into the backend context; otherwise
  the advertised cap neither limits generation nor controls KV sizing.
- A cached ggml topology still needs scheduler reset/allocation before each
  GPU execution. Reusing stale scheduler bindings caused the earlier Metal
  second-token crash; right-sizing the topology does not change that rule.

The remaining default-quality bottleneck is the 10-step DiT-CFM flow. Reducing
`COSYVOICE3_FLOW_STEPS` trades quality for nearly linear speed. Do not silently
change its default, and do not treat cold reads from an external SSD as
inference time; use a resident server when measuring repeated synthesis.

### CosyVoice3: batch CFG exactly as upstream, and lazy-load cloning encoders

Upstream CFM stacks the conditional and unconditional branches into batch
size 2 and evaluates the 22-layer DiT once per Euler step. The first C++ port
ran two B=1 forwards, doubling graph launches and weight reads. A batch-aware
DiT path must preserve the batch dimension through Q/K/V as
`[head_dim, heads, time, batch]`; flattening batch into time would allow
cross-branch attention unless a block-diagonal mask is added.

The position-convolution input pipeline is still built once per branch, then
the two `[dim,time,1]` results are concatenated on the batch dimension. The DiT
stack, final AdaLN, and output projection run at B=2. On an M1 this reduced the
10-step flow from roughly 4.6 s to 1.9–2.3 s. Both a short 10-step synthesis
and longer bucket-crossing tests produced byte-identical WAVs versus separate
CFG forwards. Keep `COSYVOICE3_CFG_BATCH=0` as a backend compatibility bisect.

KV allocation and active attention length are separate concerns. Allocate
storage for the full generation cap so K/V never has to be copied while
growing, but build the cached T=1 graph against 256-token active buckets.
Crossing a bucket rebuilds the topology against the same persistent storage.
This cut the same short LM decode to about 0.5 s; a 319-token test crossed a
bucket and remained byte-identical. `COSYVOICE3_KV_BUCKET=0` restores the full
allocation shape for debugging.

Finally, baked voices never use speech_tokenizer_v3 or CAMPPlus. Eagerly
loading them added about 475 MiB and substantial external-SSD startup time.
The CLI backend and session C ABI now discover their paths at initialization
but load both under a mutex only when a `.wav` cloning request arrives. This
must be lazy, not permanently disabled: a resident server or binding session
can receive baked-voice requests first and a WAV-cloning request later.

The contributing checklist's `Session.available_backends()` assertion also
exposed a separate binding trap: the compiled backend CSV now exceeds 256
bytes. Python and Rust used fixed 256-byte buffers and silently lost every
backend after `kyutai-stt`, including CosyVoice3. The C ABI returns the full
required length even when the supplied buffer truncates, so bindings must
retry with `required + 1` bytes instead of assuming the backend list is small.

## TADA encoder port: staged model loading on constrained RAM (§221)

Porting the TADA encoder pipeline (aligner + WavEncoder + LocalAttentionEncoder)
to C++ exposed several patterns for working with >1 GB PyTorch models on 8 GB
RAM:

**Staged loading defeats OOM.** The encoder and aligner together exceed 2 GB at
float32. Loading both simultaneously OOM-kills on 8 GB. Solution: load the
aligner first, run it, `del aligner; gc.collect()` to free ~1.3 GB, then load
the encoder. RSS dropped from 705 MB → 463 MB at the free boundary, confirming
pages were reclaimed. The same principle applies to any reference-dump script
that needs multiple large models — never hold two in memory simultaneously.

**CIFS-backed HF cache is 10–50× slower than local disk.** Model loading from
`/mnt/akademie_storage/huggingface/hub` (CIFS) took 5–10 minutes vs ~6 seconds
from `/mnt/volume1/tmp-overflow/` (local NVMe). For iterative debugging, copy
safetensors to local storage first:
```bash
cp $HF_CACHE/models--HumeAI--tada-codec/snapshots/*/encoder/model.safetensors \
   /mnt/volume1/tmp-overflow/tada-codec-local/encoder/
```

**transformers 5.x broke `from_pretrained` for custom PreTrainedModel
subclasses.** The `all_tied_weights_keys` attribute was removed; models that
inherit `PreTrainedModel` crash with `AttributeError` on load. Fix: monkey-patch
a property with both getter and setter before the first `from_pretrained` call:
```python
if not hasattr(transformers.PreTrainedModel, 'all_tied_weights_keys'):
    transformers.PreTrainedModel.all_tied_weights_keys = property(
        lambda self: getattr(self, '_all_tied_weights_keys_store', None) or
                     getattr(self, '_tied_weights_keys', None) or {},
        lambda self, val: setattr(self, '_all_tied_weights_keys_store', val))
```
This must be a property with a setter (not just a getter), because HF code
assigns to it during `_finalize_model_loading`.

**ggml Conv1d padding differs between dilated and strided convs.** The TADA
WavEncoder has two kinds of convolutions:
- ResidualUnit dilated convs: `pad = dilation * (K-1) / 2` (symmetric, output=input length)
- EncoderBlock stride convs: `pad = ceil(stride/2)`, matching PyTorch's
  `Conv1d(kernel_size=2*stride, stride=stride, padding=math.ceil(stride/2))`

Using the dilation formula for stride convs gives wrong padding (e.g. stride=6:
formula gives p=5, correct is p=3). The `wn_conv1d` helper needs a
`pad_override` parameter for stride convs.

**ggml tensor layout: (d, T) channel-first reads back as (T, d) row-major.**
When a ggml tensor has shape `ne[0]=d, ne[1]=T`, calling
`ggml_backend_tensor_get` returns data in ne[0]-contiguous order, which IS
`(T, d)` row-major — each frame's d features are contiguous. This matches
Python's `(T, d)` numpy layout directly. Do NOT transpose dump tensors before
the diff comparison; the ggml→CPU readback already produces the right layout.

**TADA aligner is wav2vec2-large with "group" CNN norm (not "layer").** Despite
wav2vec2-large documentation saying it uses LayerNorm on all CNN layers, the
TADA aligner's checkpoint has InstanceNorm only on layer 0 and no norm on
layers 1–6 (and no conv bias). The converter must detect this from the tensor
names rather than assuming the architecture.

**wav2vec2 pos_conv weight-norm shape varies.** Standard wav2vec2 stores
pos_conv `original0` (g) as `[Cout, 1, 1]` (dim=0 norm). The TADA aligner
stores it as `[1, 1, K]` (dim=2 norm). The materialization formula
`w = g * v / ||v||` must detect which dimension was normed by finding which
dims of g are size 1 while v is larger.

## 2026-07-02 — `--gpu-backend` was silently ignored (#214)

**Symptom:** `--gpu-backend vulkan` on a CUDA+Vulkan build still routed all
inference through CUDA. `perf report` showed 20%+ in `libcuda.so`.

**Root cause:** Every backend (60+ init sites) called `ggml_backend_init_best()`
which unconditionally picks GPU backends by priority (CUDA > Vulkan > Metal).
The `--gpu-backend` CLI flag was parsed into `whisper_params::gpu_backend` and
logged in verbose mode, but never wired into any backend's GPU init path.

**Fix:** Created `src/core/gpu_backend_pref.h` with `crispasr_init_gpu_backend()`
as a drop-in replacement. It checks a process-global preference (set once at CLI
startup via `crispasr_set_gpu_backend_pref()`), filters registered devices by
name (case-insensitive prefix match: "vulkan" matches "Vulkan0"), and falls back
to `ggml_backend_init_best()` when no preference is set or the preferred backend
isn't found. Replaced all 60+ call sites mechanically. Also patched
`whisper_backend_init_gpu()` in `crispasr.cpp` to skip non-matching devices.

**Validation:** Kaggle P100 regression test — 7/7 ASR backends pass with correct
JFK transcripts. `--gpu-backend cpu` runs 3.6x slower than `--gpu-backend cuda`
with `CUDA leak: False`, confirming the routing is real.

**Lesson:** A CLI flag that's parsed and logged but never threaded into the code
that acts on it is worse than not having the flag — users assume it works.
Process-global preference (set once, read everywhere) is the right pattern when
the preference applies to the entire process, not per-model. Adding an `#include`
to 60+ files invalidates the entire ccache — budget for a cold rebuild when
touching shared headers.

### §216 — Built-in G2P tokenizer eats symbols in technical tokens (#216)

**Built-in G2P tokenizer splits on a fixed punctuation set and drops everything
else.** The `tokenize()` function in `g2p_en.h` splits on `' ,.:;!?-\n'` and
passes everything else as word characters. Tokens like `C++`, `C#`, `.NET`,
`Node.js`, `CI/CD` are treated as single words. The `+`, `#`, `/` characters
aren't in any CMUdict entry and the LTS letter-to-sound rules silently skip
non-alpha characters, producing truncated output (`C++` → `k`).

**Text normalization must run before tokenization, not inside it.** The fix is
a `normalize_technical_tokens()` pass that expands compound tokens into natural
English words (`C++` → `C plus plus`) before the tokenizer ever sees them. This
is better than making the tokenizer symbol-aware because: (a) symbols have
context-dependent pronunciations (`+` in `C++` is "plus" but `+` alone might be
"positive" or silent), (b) downstream G2P tiers (CMUdict, neural, LTS) already
handle normal English words well, and (c) the normalization table is easy to
extend.

**G2P strategy env var prevents vendor lock-in.** When built-in G2P is the
only phonemizer (Windows, no espeak-ng), there's no fallback for edge cases.
`CRISPASR_KOKORO_G2P=espeak-first` lets users with espeak-ng installed prefer
it for higher-quality output, while `builtin-only` is useful for GPL-free
deployments that want deterministic results.

## 2026-07 — SentencePiece tokenizer taxonomy: greedy longest-match is wrong for BOTH Unigram and BPE

An audit of the shared `crisp_punc` engines and the m2m100 translation backend found
that several tokenizers used **greedy longest-match** subword segmentation. Greedy is
correct only for **WordPiece** (BERT). It is wrong for the two SentencePiece model types,
in different ways, and mis-segmentation silently degrades every downstream head:

- **SP Unigram** (XLM-RoBERTa: pcs, fullstop-punc, fireredpunc's `is_sentencepiece` path)
  needs **Viterbi** (maximise the sum of piece log-probs). Greedy mis-splits multi-subword
  words (e.g. "delayed" → wrong pieces), corrupting embeddings (measured per-token cos as
  low as 0.13). Fix: inline Viterbi over `tokenizer.ggml.scores` (the SP piece log-probs),
  which the converters must now emit. Landed for pcs (`crisp_punc/src/pcs.cpp`,
  `tokenize_sp_viterbi`) and fireredpunc; GGUFs re-uploaded with scores.
- **SP BPE** (M2M-100 / wmt21: scores are `-merge_rank`, not log-probs) needs **merge-order**
  tokenization — the shared `core_spm::tokenize_bpe` (merge the highest-score adjacent pair
  whose concatenation is a vocab piece). Greedy was the m2m100 backend's tokenizer since its
  initial commit (`502505fd`, self-labeled "for basic testing") — never a regression, always
  broken. Two GGUF-side bugs blocked the BPE tokenizer even after switching algorithms:
  1. **Scores mis-aligned.** The converter mapped SP scores by a constant id offset
     (`spm_id = m2m_id - 1`). That is WRONG whenever `vocab.json` is not a constant-offset
     reorder of the SP model (true for m2m100_418M, not for wmt21). Score **by string**
     (`sp.piece_to_id(tok)`) instead — verified 0/127607 mismatches vs SP.
  2. **Vocab incomplete.** `vocab.json` can omit SP pieces that are needed only as
     *intermediate* merges (m2m100_418M omits ~390, e.g. "esterd" building "esterday").
     Without them BPE stalls short (`▁y est erd ay` instead of `▁y esterday`). Fix: the
     converter appends the missing SP pieces at ids ≥ `vocab_size` with their SP scores; the
     engine uses them for merging and maps any *final* token id ≥ `vocab_size` to `<unk>`
     (exactly HF's `M2M100Tokenizer.convert_token_to_id`). These pieces never have an
     embedding row because they are never emitted as final tokens.

Result: m2m100-418m f16 reproduces HF (`AutoModelForSeq2SeqLM`) translations **exactly**
(token counts + output); q8_0 matches except rare quant-floor decode flips. wmt21-dense-24
(4.7B) was re-converted on Kaggle (`tools/kaggle/wmt21-convert/`) — its `vocab.json` was
already complete (0 intermediates) and its scores already correct, so its only bug was the
engine's greedy tokenizer; all three wmt21 GGUFs verified 0/128000 score mismatch + BPE
reproduces SP. General rule: **for a SentencePiece BPE model the GGUF tokenizer needs the
FULL SP vocab (incl. non-final intermediate pieces) with by-string scores, or BPE diverges.**

Debug env: `M2M100_DEBUG=1` dumps the BPE path + pieces; `PCS_DEBUG` / `PCS_FORCE_CPU`
similar for the punctuation engines.

### 2026-07 — GELU "gelu" is exact erf, not the tanh approximation

`ggml_gelu` is the tanh approximation; `ggml_gelu_erf` is exact. A model whose HF config
`hidden_act="gelu"` (BERT, RoBERTa, XLM-R, DeBERTa-v2/v3, LiLT) is **exact erf** — using
`ggml_gelu` drifts and flips borderline argmaxes over many layers. Only `gelu_new` /
`gelu_pytorch_tanh` / `gelu_fast` (GPT-2, Gemma, SigLIP) are genuinely tanh, and
`quick_gelu` (CLIP) is `ggml_gelu_quick`. Fixed in `bert_encoder.cpp` (bert-base-uncased,
MeloTTS conditioning) and, in the shared crisp_punc lib, in `pcs.cpp` + `fireredpunc.cpp`.
Verify each hit against the model's real `config.json`, not intuition.

---

## Issue #89 close-out: four transferable lessons (2026-07-04)

(Reusable tooling from this audit lives in-tree:
`tools/asr_coverage_score.py` — the char-bigram recall/precision scorer with
`--strip-latin` / `--reading` normalization and `--per-line` loss
localization; `tools/nemo_parakeet_blueprint.py` — the NeMo blueprint-parity
benchmark covering plain / CTC / local-attention / buffered modes.)

### 1. FastConformer-JA blanks a whole utterance whenever enough context *follows* it — test the missing span in isolation before blaming chunking

The reporter's clip's first 4.6 s ("皆さんこんにちは…") transcribes **verbatim
in isolation** but is skipped inside *any* window ≥8 s that extends past it —
even a window starting at exactly 0.0 s. This is not decoder cold-start, not
z-norm drift at the boundary, and no amount of chunk/overlap tuning recovers
it (cap sweep 8/10/12 s: the intro is skipped in all of them). Upstream NeMo
does the same. The **diagnostic that cracked it**: cut the missing span out
with ffmpeg and transcribe it alone. If that works, the fix is a **gap-fill
second pass** (find ≥1 s spans with no emitted words inside a slice,
re-transcribe each in isolation, merge non-duplicate words back) — shipped in
`crispasr_run.cpp` (`crispasr_gap_fill_slice`, inside `process_slice` so all
output paths get it) and mirrored in the session ABI. Coverage went
80 % → 97.2/96.9/95.9 % on the 60/120/300 s reproducers. Cost: one extra
short encode per recovered gap. Counterintuitive tuning result: lowering the
trigger from 1.0 s to 0.6 s made recall *worse* (more transcription-variant
noise) and much slower — the 1.0 s default is a sweet spot, don't "more is
more" it.

### 2. Cross-model content-recall scoring has a ceiling — measure it with a second independent system before chasing 100 %

Char-bigram recall of parakeet output against a whisper-large-v3-turbo
reference saturates around **93-95 % raw** even when nothing is missing,
because two correct systems spell the same Japanese differently (皆さん vs
みなさん, くる vs クルー, 初め vs 始め). Two normalizations matter for JA:
(a) **strip latin** from the reference — a JA-only model renders "Speak
Japanese Naturally Podcast" in katakana (correct!) which a latin-script
reference can't credit, and (b) **hiragana-reading normalization** (pykakasi
`convert → hira`) to erase kanji/kana spelling variants — this is the honest
*coverage* metric. Calibrate the ceiling by scoring an unrelated strong model
(SenseVoice-small) against the same reference: it landed at 97.2/96.8 %
recall — exactly where our fixed pipeline landed — at far lower precision
(78 % vs our 90 %). When your recall equals the independent-model ceiling,
the residual is hearing variants, not missing content; stop optimizing.

### 3. Prove the port bit-faithful FIRST (same-audio blueprint parity), then the fix hunt changes from "find our bug" to "pick a better pipeline policy"

NeMo 2.7.3 with the original checkpoint, same audio: plain transcribe
42 chars / local-attn [128,128] 157 chars / CTC single-pass 7 chars —
**char-identical to our GGUF runtime in every mode** (the local-attn parity
came free because `core/fastconformer.h` + `parakeet.cpp` already had the
band-mask; `CRISPASR_PARAKEET_ATT_CONTEXT="L,R"` now exposes it as NeMo's
`change_attention_model` equivalent). And NeMo's *recommended* long-form
path (buffered `BatchedFrameASRTDT`) scored 14.7 % (4 s/8 s) and 51.3 %
(stock 1.6 s/4 s) — worse than our then-current default. Two cheap decisive
A/Bs to run before any code: (a) **CTC head over the same encoder features**
(single-pass CTC also collapsed → the pathology is in the ENCODER, not TDT
decode dynamics), (b) **the blueprint on the same audio** (identical failure
→ not a port bug). Each took minutes and killed whole classes of hypotheses.

### 4. Autoregressive decode compounds quant noise; non-autoregressive decode over the same weights doesn't — offer CTC as the q4_k mode

parakeet-ja q4_k TDT enters a "りましたけど" fixed-point repetition loop
(the OLD published q4_k does exactly the same — pre-existing, not a
conversion regression). The **CTC head over the same q4_k encoder is clean**
and matches the F16 CTC transcript: CTC has no predictor-LSTM feedback loop
for the quantisation noise to compound through. q8_0 TDT is byte-identical
to F16. So instead of dropping q4_k (the Kokoro precedent), ship it with
"use `--parakeet-decoder ctc`" guidance — which required actually including
the hybrid model's CTC head in the GGUF (`ctc.weight/bias`; the pre-2026-07
uploads lacked it and `--parakeet-decoder ctc` silently fell back to TDT
with only a stderr note). When a backend advertises alternative decode
heads, verify the published GGUFs actually CONTAIN them.

## A once-allocated-then-reused bucket step graph is a CUDA-graph-capture hazard: reset+alloc the sched EVERY step (#220, 2026-07-04)

Chatterbox T3 aborted out of the box on an RTX 3090 Ti (`crispasr:main-cuda`)
with `CUDA error: an illegal memory access was encountered` in
`ggml_backend_cuda_synchronize`, at AR decode step ~2, right after a
`CUDA Graph id N reused` log line. It works on CPU, and T3-on-GPU is the
default for "non-Metal GPU", so every affected card hits it immediately.

**Mechanism.** ggml-cuda replays a captured CUDA graph as a single
`cudaGraphLaunch` when it can. It keys the captured executable on the split
graph's `uid` and, in `ggml_cuda_graph_update_required` (ggml-cuda.cu), takes a
fast path: *if `cgraph->uid == graph->uid`, return "no update required" and
replay the previous capture verbatim* — with every device pointer baked in at
capture time, skipping all per-node pointer/shape re-validation. A **fresh split
uid is minted only inside `ggml_backend_sched_alloc_graph`** (ggml-backend.cpp
`sched->splits[i].graph.uid = ggml_graph_next_uid()`).

The §186 Lk-bucketed T3 decode was written for M1: it pre-builds the step graph
AND allocates its dedicated step scheduler **once per bucket**, then for every
subsequent step just re-runs `ggml_backend_sched_graph_compute` (the sched stays
`is_alloc`, so the compute call skips reset+alloc). That "reuse-shortcut" keeps
the split uid constant across steps → on sm_80+ the second step hits the uid
fast-path and replays a **stale** capture → illegal access. It's the CUDA twin
of the Vulkan null-subbuffer segfault (#170) that already keeps T3 on CPU there,
and the same class as qwen3-tts #52/#56.

**The capture story is an aggravator, not the whole bug (verified 2026-07-04).**
I first assumed the crash *required* CUDA-graph capture — which is hard-gated to
sm_80+ (`cc < GGML_CUDA_CC_AMPERE → disable_due_to_gpu_arch`, ggml-cuda.cu) — and
therefore that a T4/P100 could not reproduce it. **Wrong.** The Kaggle A/B kernel
on a **P100 (sm_60, capture DISABLED, no "CUDA Graph id N reused" line)** still
crashed the old reuse-shortcut with the *identical* `illegal memory access` at
`ggml_backend_cuda_synchronize` (ggml-cuda.cu:3151), rc=-6, at decode step ~2 —
while `fix_default` produced a valid WAV that ASR-round-tripped to the *exact*
input text, byte-comparable to the CPU reference. So the reuse-shortcut (a sched
allocated once and reused without per-step `reset`+`alloc`) is **unsafe on CUDA
independent of graph capture** — capture on sm_80+ is just an *additional* way it
detonates (stale captured-executable replay), which is why the reporter's sm_86
log showed the extra "reused" line. Lesson: **don't gate a "needs feature X"
repro theory on a code-path arch check until you've actually run it — the same
symptom had a second, capture-independent cause, and the "cheap" GPU reproduced
it fine.** The reuse-shortcut is still **correct on Metal** (no capture, unified
memory) and is the validated §186 M1 fast path, so the fix keeps it there and
only reset+allocs per step on a non-Metal GPU.

**Fix.** On a non-Metal GPU backend, `ggml_backend_sched_reset` +
`ggml_backend_sched_alloc_graph` the bucket step sched **every step**. This mints
a new split uid each step so `ggml_cuda_graph_update_required` goes through
`cudaGraphExecUpdate` (which refreshes the baked pointers) instead of blindly
replaying — while the *cached cgraph* keeps `nodes[0]` (the capture key) stable
so capture still engages and the speedup survives. This is precisely the
granite/outetts CUDA-graph-bucket pattern (`docs/contributing.md` §210: "even on
a topology-stable captured graph you must still reset+alloc per step"). Metal
keeps the alloc-once reuse; old path stays available for A/B via
`CRISPASR_CHATTERBOX_T3_BUCKET_REUSE=1`. **General rule: any bucket/step graph
that is allocated once and reused across steps is unsafe on CUDA (and Vulkan) —
reset+alloc per step; granite_speech is the reference implementation.**


## A cached cgraph is invalidated by ANY other graph on its scheduler — reset+alloc per step is NOT enough (#171, 2026-07-06)

The #171 "TTS regression" saga (RDNA4 driver theories, quant-recipe
hardening, knob matrices — none of it moved the symptom) ended as an
ASan-proven **heap-use-after-free** reproducible on the first try with a
CPU-only Debug+ASan build: vibevoice's cached diffusion pred-head graph
kept `tensor->buffer` pointers into `ctx->sched`'s gallocr compute buffer;
a mid-generation text-window LM graph on the *same* scheduler grew the
gallocr (`reallocating CPU buffer from 0.36 MiB to 0.37 MiB` — free + new
alloc); the next DPM step's `ggml_backend_sched_alloc_graph` of the cached
graph then read the freed `ggml_backend_buffer` inside
`ggml_backend_sched_split_graph` (before realloc could have fixed anything).
Debug builds segfault; **Release builds often survive the stale read and
silently corrupt activations instead** — same bug, presenting as "broken
audio", which is why it masqueraded as a model/driver problem for weeks.
The #184 CUDA "illegal memory access" was the same class via the LM bucket
graphs (pos/neg CFG steps alternating two cached graphs of different Lk on
one shared `lm_step_sched`).

**The invariant: only the graph most recently allocated on a scheduler has
valid buffer pointers.** This *sharpens* the #220 rule ("reset+alloc the
sched every step"): per-step reset+alloc of the *same* graph is fine, but
it does NOT protect a cached graph from *other* graphs allocated on the
same scheduler in between — the UAF read happens during split_graph inside
the very alloc that would re-set the pointers. Fixes that satisfy the
invariant (c96c4997): give each cached graph a **dedicated scheduler**
(fixed topology → its gallocr never reallocs under it), and when several
cached graphs must share one sched, rebuild the incoming graph whenever it
was not the last one allocated there.

**But measure before keeping any graph cache at all.** A same-seed 4-way
A/B (CPU Release, short + 58-token synthesis) showed the vibevoice pred +
bucket caches were worth **0% end-to-end** (diffusion compute dominates
graph build ~100×) and all four on/off combinations produced
**byte-identical WAVs** — the entire bug class (#171 segfault, #184 CUDA,
#47 Metal/Vulkan asserts) guarded an optimization worth nothing, exactly
like the chatterbox CFM graph cache (§208 dud). Deleting the caches
outright is the recorded follow-up; until then, `CRISPASR_VIBEVOICE_
REUSE_PRED_GRAPH=0/1` and `CRISPASR_VIBEVOICE_LM_BUCKETS=0/1` A/B them.

## ASR roundtrip does not validate audio ONSETS; reproduce with the original pipeline before debugging your runtime (#171, 2026-07-06)

The residual #171 complaint ("music plays before the voice") survived every
runtime theory because our acceptance tests could not see it: the vibevoice
output opened with ~1 s of BGM, yet **whisper roundtripped the full text
perfectly** (ASR skips non-speech prefixes) and spectral heuristics
(RMS/centroid/syllabic modulation) also read as speech. Only a human
listening caught it — on outputs we had repeatedly declared "clean".
When a report says "noise/music at the start or end", roundtrip is
necessary but proves nothing about prefixes/suffixes: have a human listen,
or ASR/classify the first second in isolation. (This is also how the
at_dec quant-protection call (5c8add40) was mis-validated: "roundtrips
perfectly" hid the same artifact.)

What actually closed the case, in one evening, was the blueprint rule
applied to *behavior* instead of tensors: run **the original pipeline**
(`microsoft/VibeVoice` `demo/realtime_model_inference_from_file.py`, their
`fr-Spk1_woman.pt`, same text, MPS) — it produced the **same music**,
proving the runtime faithful and the artifact inherent model behavior
(documented by the maintainers: spontaneous BGM from short inputs, small
models, intro-style text, BGM-carrying voice prompts; BGM in training data
is intentional per the paper). Two supporting checks worth reusing:
(a) verify reporter "regression" claims with FIXED files — a v0.7.1 binary
on today's files produced the same audio as main (corr 0.84 at the trim
lag), so "worked before v0.7.2" was a different-files effect; (b) when the
official pipeline needs to run locally: `USE_TF=0` (miniconda TF import is
broken), `HF_HOME` override (`~/.cache/huggingface` is a dangling SSD
symlink on the M1), venv at `~/code/VibeVoice/.venv`. Mitigation shipped as
knobs, not fixes: `--tts-cfg-scale` (5f3046a7) + existing `--seed`; the
dominant lever is the voice prompt itself (official fr prompts are
BGM-prone; emma was clean).


## SentencePiece `byte_fallback` is not optional decoration — OOV emoji/symbols must become `<0xHH>` byte tokens, and a `utf8_aligned` Viterbi dead-ends at a multi-byte lead without it (#221, 2026-07-07)

Irodori-TTS (#221) used emoji as **emotion controls** (👂=whisper, 😮‍💨=breath). Two bugs, both in `core/sentencepiece.h`'s unigram Viterbi:

1. **Collapse to BOS.** In `utf8_aligned` mode the single-byte unk fallback edge `i→i+1` is illegal at a multi-byte codepoint's lead byte: `i+1` is a UTF-8 continuation byte that the alignment guard skips, so **no edge leaves `i`**, every later position is unreachable, the backtrack returns empty, and the whole utterance tokenizes to just the BOS. Any OOV multi-byte char (an emoji outside the vocab) triggered it. Minimum fix: a codepoint-level fallback that advances the full 1–4 byte codepoint.

2. **The real fix — byte_fallback.** Mapping OOV chars to a single `<unk>` is *wrong* for models trained with SentencePiece `byte_fallback` (llm-jp-3 here). Those tokenizers emit OOV bytes as `<0xHH>` tokens (👂 → `<0xF0><0x9F><0x91><0x82>`; a ZWJ sequence like 😮‍💨 = 😮-bytes + ZWJ(**in-vocab**) + 💨-bytes), and the model **learned those byte sequences as the controls**. `<unk>` played them as noise. Fix: a 256-entry byte→`<0xHH>`-token-id table (`Config::byte_fallback`), built from the vocab at load; the OOV fallback emits the byte token (scored by `scores[]`) instead of `<unk>`.

**Validation that actually proves it:** dump the C++ token ids and assert **byte-for-byte equality with the reference HF tokenizer** on the exact strings in the bug report — not "does it sound right". C++ matched HF to all 47 ids incl. the byte-fallback + ZWJ runs, which *is* the proof the model gets identical input. **Lesson:** for any tokenizer port, check whether the source uses `byte_fallback` before assuming `<unk>`; the tell is that OOV input tokenizes to `unk=0` in HF. Emoji that ARE in-vocab (😭, 🥺) are red herrings — they were never the failing case.


## Porting the missing sub-model is necessary but not sufficient — hunt the hardcoded "unconditional" defaults that ignore it (#221 voice cloning, 2026-07-07)

Irodori-TTS voice cloning was a stub. Porting the DAC-VAE **encoder** (reference audio → 32-dim latent: 30-conv stack + Snake + `in_proj` mean, preceded by BS.1770 −16 LUFS normalization) got the latent right (cos 0.99996 vs PyTorch GT) — and cloning **still did nothing**, because the synthesize path had two hardcoded unconditional defaults that silently discarded the reference:
- speaker state hardwired to a zero vector (the speaker encoder was never run), and
- the DiT attention mask **always** `-inf`'d the speaker KV positions (the same JointAttention "unconditional" default from the entry near the top of this file — here it defeated a *real* reference).

The fix was 3 lines of graph plumbing (`attend_speaker` toggle) on top of the whole encoder port. **Lesson:** when a feature is "stubbed", grep the *consumer* for the zeros / always-masked / `TODO` defaults before declaring the port done — the missing model is usually the visible half; the invisible half is the conditioning path that was wired to ignore it. Diagnostic that pinpointed it fast: A/B the output **with vs without** the reference and measure a speaker-similarity proxy (time-averaged DAC-VAE latent cos) — identical outputs (cos 1.0) said "reference not reaching the model", not "encoder wrong". After the fix: 0.08 → 0.65 (vs 0.70 for the repo's own cloned sample).


## Re-quantized GGUFs on HF must be re-baked when the quantizer adds carve-outs (#240, 2026-07-10)

The qwen3-asr-1.7b Q4_K GGUF on HuggingFace produced empty transcripts because it was quantized before the audio-tower Q8_0 floor landed in `crispasr-quantize` (`3c3ba2c7`). The fix was in the quantizer, not the runtime — old GGUFs need re-quantizing from the F16 source.

**Lesson:** when a `crispasr-quantize` change adds a new tensor carve-out (keeping sensitive weights at higher precision), **every affected GGUF on HF must be re-baked**. The fix isn't deployed until the re-quantized file replaces the old one. Add a re-bake Kaggle kernel to the same PR/commit that adds the carve-out, not as a follow-up.


## TTS code predictor: the dispatch overhead is real, but "fuse into one graph" and "skip the reset" are BOTH wrong — the fix is a sched-free persistent graph (#245/§232, corrected 2026-07-11)

CrispASR's qwen3-tts code predictor runs 15 `ggml_backend_sched_graph_compute`
calls per audio frame (one per codebook 1-15), each paying scheduler
reset+alloc+split. On M1 Metal that was ~25 ms/dispatch against sub-ms kernel
time (~400 ms/frame). The original write-up of this entry proposed "fuse the
15 into one graph, or at minimum skip the scheduler reset between steps."
**Both halves of that advice are wrong** — recording the correction because
each half detonates a known trap:

- **You cannot fuse the 15 steps into one graph.** Between steps the logits go
  to the CPU for `top_k`/seeded-RNG sampling, and the sampled id selects the
  next `codec_embd` row — a genuine CPU round-trip data dependency. Bit-
  identical output *requires* 15 dispatches with host sampling in between; a
  single unrolled graph would have to bake the sampler into ggml (it can't, and
  even a greedy argmax variant would change the output). This is the same
  "sampling lives on the host between steps" property that stops you fusing any
  real AR decode loop.
- **"Skip the scheduler reset between steps" is exactly the #220/#171 hazard.**
  Reusing an allocated sched across steps (the O15_SKIP_REALLOC path) crashes on
  CUDA (stale captured-executable / UAF) and returns nil-buffer inputs → no
  audio on current Metal. See the two entries above; do not reintroduce it.

**The correct resolution — a third option between "reuse the sched plan"
(unsafe) and "reset+alloc every step" (still pays the dispatch cost): drop the
scheduler entirely for the hot step.** Build the T=1 step graph (and the T=2
step-0 graph) ONCE into a private `ggml_context` arena with a fixed topology
(pinned `Lk`, runtime `positions` feeding both RoPE and the `ggml_set_rows` KV
write, `lm_head` routed through a writable slot tensor so all 14 skip-steps
share one graph), `ggml_gallocr_new(default_buft(ctx->backend))` +
`ggml_gallocr_alloc_graph` it ONCE on the single code-pred backend, then each
step is just `ggml_backend_tensor_set` on the inputs + `ggml_backend_graph_
compute(ctx->backend, gf)` + read. No `ggml_backend_sched` is involved, so the
sched-plan-reuse breakage cannot occur by construction. Guard init with a
per-node `ggml_backend_supports_op` scan and a KV-buffer-placement check; fall
back to the sched path (and when the code-pred is CPU-pinned). Gated
`QWEN3_TTS_CP_DIRECT`, **default ON for GPU backends** (`bf74d22b`).

**Backend-dependent verdict — the same "fast path" flips sign across
backends, so the default must be conditional:**
- **CUDA P100:** −11 % (25.4 → 22.3 ms/frame), output-equivalent (PCM cos
  0.9999999997) — a clean win, and it also sidesteps the #56 O15 CUDA crash.
- **M1 Metal:** ~equal on an *idle* box (110 vs 111 ms/frame) but ~3× faster
  *under load* — the sched path degrades badly under GPU contention, so the
  real Metal win is robustness, not idle speed.
- **CPU:** ~2× **slower** (87 → 160 ms/frame). There is no dispatch cost to
  save on CPU, and the per-step `lm_head` slot blit (~2.2 MB `ggml_backend_
  tensor_copy` ×14/frame) becomes pure overhead. → default OFF on CPU
  (`c->backend == c->backend_cpu`); env overrides both ways.

**Transferable lesson:** for any small fixed-topology hot graph (multi-codebook
TTS code predictors, per-step decoders), the win is not fusion and not sched-
reuse — it's a single-backend `gallocr` graph you allocate once and re-dispatch
sched-free. And always A/B the "fast path" on CPU too: a per-step blit that is
free on unified-memory GPU can dominate on CPU and invert the default.


## Python reference dumpers must apply identical audio conditioning as the C++ runtime (kyutai-stt 2.6B, 2026-07-10)

The kyutai-stt 2.6B reference GGUF had a truncated JFK transcript ("ask not what your country can do for you ask what") because the Python dumper fed raw PCM into Mimi without the silence tail + silence prefix the C++ runtime applies. The 2.6B model has `audio_delay=2.5s` + `audio_silence_prefix=1.0s` = 3.5s total lookahead; without appending 56000 samples of silence at 16 kHz before resampling, and prepending 24000 zeros at 24 kHz before Mimi encode, the causal LM can't flush its final pending tokens and the last ~3 words are dropped.

**Lesson:** when writing a new `tools/reference_backends/<name>.py`, trace the C++ pipeline's audio preprocessing line-by-line (resampling, silence padding, normalization) and replicate it exactly. Both sides must see identical conditioned audio for >99% parity. Read the model's `config.json` (especially `stt_config`) for model-specific conditioning parameters.


## Issue triage discipline: check the codebase before leaving issues open (2026-07-10)

28 GitHub issues were open despite having fixes on main — some for months. A single triage pass (grep issue numbers in `git log`, verify the fix exists on main, close with a commit reference) reduced open issues from 28 to 8 in one session. **Lesson:** after merging a fix, close the issue immediately with the commit hash. A stale open-issue list misleads contributors and users about the project's actual state. Periodic triage passes (monthly) catch the ones that slip through.


## Any T-sized `ggml_view` into a fixed-length weight table must be bounded by the tensor's real length — long inputs overflow it and abort (indextts, 2026-07-07)

IndexTTS voice cloning aborted in `ggml_view_2d` for a long reference (a 164 s clip). The Conformer conditioning encoder subsamples mel by ~2 (`T_enc = (T_mel-3)/2 + 1`) then views `T_enc` rows out of the fixed 5000-row relative-position table `pe`; a 164 s ref → ~15.4k mel frames → `T_enc≈7700 > 5000`, so the view ran past the tensor and hit `GGML_ASSERT(data_size + view_offs <= ggml_nbytes)`. It had been *introduced* by deleting an earlier length clamp under a "no truncation needed — full reference is used" comment. Fix: read `pe->ne[1]` at runtime and, only when `T_enc` would exceed it, truncate the conditioning mel to the longest length the table supports (`T2 = 2*(pe_len-1)+3` → `T_enc == pe_len`). **Lesson:** a positional/embedding table is a hard cap on sequence length; any view sized by a runtime `T` must clamp to `ne[1]` (or the model's documented max) rather than trusting the input. Short refs stay under the cap, so this class of bug only surfaces on the largest inputs a user throws at it — and a stale "we removed the limit, it's fine now" comment is a reliable place to find one.


## A `frames/token` (or chars/sec) length heuristic silently truncates diffusion TTS — run the duration predictor the model already ships (#221, 2026-07-07)

Irodori-TTS dropped tail clauses on kanji-heavy text: output length was `latent_steps = T_text * 6.3 frames/token`, but kanji unpack to a variable number of mora, so a 14-token line got 88 frames (3.52 s) when it needed ~196 (7.85 s), and the flow-matching diffusion — which *fills the allocated duration* — simply ran out of room and cut the end. The model ships a **duration predictor** (`token_sum_adarn_zero`: per-token RMSNorm → AdaLN-Zero from `silu(speaker_vec)` → SwiGLU → `tanh(gate)` residual → out_proj → `softplus` per token, **summed**; returns `log1p(total)` so frames = the raw sum) whose weights were loaded and never run. Wiring it (`latent_steps = round(total_frames·scale/speed)`) fixed the truncation; the ASR roundtrip went from cut-off to complete.

Two lessons: (1) **a heuristic in the hot path is a red flag that a real model component is sitting unused** — grep the loader for weights that never appear in a forward. (2) **The duration predictor is sensitive to text-encoder drift**: isolating with the F32 reference gave the exact frame count (196.35) from the reference `text_state`, but the q8 model's `text_state` (cos≈0.9995) gave 163 — the per-token `softplus` sum amplifies small logit errors across tokens. 163 vs 196 didn't truncate (the roundtrip was complete, just ~17 % faster), so it shipped, but it's a reminder that a downstream scalar-regression head is a magnifying glass on upstream cos drift — validate it on the *end metric* (does all the text get spoken?), not just the per-tensor cosine.


## Sub-8-bit quantization of an audio tower fails as BEHAVIORAL collapse (loops, empty output) — and there is no per-block cliff to bisect (#218 qwen3-asr, 2026-07-10)

The #218 long-form symptoms (mid-transcript repetition loops, "language
none" → empty output on a 145 s clip) read like decode or prompt bugs. The
port was faithful at every stage: mel cos_mean 0.999996, prompt ids exact
(1900/1900), F16-GGUF first-logits cos 0.9981 vs bf16 (= the F16 floor),
F16 e2e ≈ verbatim reference. The actual cause was that q4_k quantized the
18-layer audio tower to Q4_0/Q4_K, and the per-block drift compounds
*smoothly* — blk00 cos 0.9996 → blk17 0.973, **no cliff, so there is no
single "broken" layer for a bisect to find** — down to encoder cos_mean
0.9716 / cos_min 0.75 at 1885 frames, which is enough to flip greedy
decode into degenerate attractors. Transferable rules:

1. **Diff the encoder OUTPUT, not just per-block cosines.** Drift
   compounds with depth AND with frame count: jfk (143 frames) survives
   where 145 s (1885 frames) collapses, and CUDA survives short clips
   where Metal breaks (#240) — platform numerics amplify, they don't
   cause. A per-block table that "looks fine everywhere" can still be a
   broken encoder.
2. **Encoders feeding an LLM decoder get a Q8_0 floor in the quantizer.**
   Same carve-out canary/mini-omni2 already had; for qwen3-asr `audio.*`
   it costs ~17 % size (540→631 MB) and moves encoder cos_mean 0.9716 →
   0.9997. Greedy LLM decode is a step function on encoder quality; there
   is no "slightly degraded transcript" regime to trade against.
3. **Quant A/B verdicts require the loop-fix disabled and the transcripts
   read** — see the metric entry below.
4. **imatrix is not a rescue for long-context decode.** Quantizing the
   tied LM head (the old imatrix recipe) re-introduces loops even with a
   Q8 tower; and an imatrix recalibrated WITH long-form clips in the
   corpus still drifts into the same attractor at the same spot — the
   activation-weighted redistribution itself harms long-context decode.
   Plain q4_k (F16 head, Q8 tower) is the long-form recommendation.

Related gotcha: the reference's attention semantics depend on which
codepath it runs — qwen_asr's `cu_seqlens` 104-frame windowing exists
ONLY on the FlashAttention-2 path; eager/SDPA (any CPU reference dump)
does FULL attention. Diff against the semantics you actually implement
(we added `QWEN3_REF_WINDOWED=1` to the dumper to get a windowed ground
truth for the windowed graph).


## The prompt contract is part of the port: a specials-only tokenizer silently sent EVERY glm-asr prompt instruction-less (#218 glm-asr, 2026-07-10)

glm-asr's GGUF carried no BPE merges, so `glm_asr_tokenize` could only
encode special tokens — **every plain-text instruction tokenized to
NOTHING**, and the CLI always set one (`--language` defaults to `en` →
"Please transcribe in English."). Net effect: the blueprint's mandatory
default instruction ("Please transcribe this audio into text") was absent
from every glm-asr prompt the runtime ever sent. Instruction-less prompts
are off-distribution: shout-loops on noise windows, instant `<|user|>`
EOS on multi-window prompts — the exact #218 symptoms. The second half
was silent truncation to one 30 s window where the blueprint encodes up
to 21 windows into one prompt. With both fixed, q4_k matches the bf16
blueprint on the 145 s clip (113 vs 115 tokens, near-verbatim).

1. **Dump the prompt ids and diff them against the reference processor
   before ANY quality verdict.** The earlier "degenerates at both
   precisions ⇒ model-inherent" A/B was invalid — both arms ran the same
   broken prompt. An A/B across precisions only isolates quantization if
   the prompt is blueprint-complete.
2. **A tokenizer that "works" can still be voiding your text.** Specials
   round-tripped fine for months; only the plain-text spans vanished.
   The tell: encode a known sentence and assert non-empty + byte-exact
   vs the HF tokenizer. (Fix: bake `tokenizer.ggml.merges` in the
   converter, backfill published GGUFs with `tools/gguf-add-merges.py`,
   run real byte-level BPE via core_bpe.)
3. **Replicate the blueprint's prompt-building code, don't paraphrase
   it.** Same class of bug in qwen3-asr: qwen_asr expresses a forced
   language as an ASSISTANT-TURN PREFILL `language <Name><asr_text>` —
   the model then structurally cannot answer "language none". Our
   semantically-equivalent system instruction ("Transcribe the speech in
   X.") let it. Prompt scaffolds are training-time contracts, not
   suggestions; chat-template newlines and answer-framing strips
   (`GlmAsrProcessor._strip_assistant_prefix_and_quotes`) count too.


## Unigram run-length loop metrics pass 2-gram cycles — and a "raw" baseline is only raw if the disable gate actually exists (#218, 2026-07-10)

Two measurement traps from the #218 arc, either of which silently turns a
gate green:

1. **Phrase cycles are invisible to unigram-run metrics.** The
   qwen3-family rebake kernel's loop gate (max repeated single-token run)
   read the rebaked mega-asr GGUFs as clean; the transcripts were
   "Come on, come on, …" ~115× — a 2-gram cycle with no long unigram run.
   Any loop/degeneration gate needs a phrase-cycle check (max cycle count
   over n-grams, n=2..8) alongside the unigram run — and the transcripts
   still need to be READ, because "hey, hey, hey" can also be genuine
   audio (cohere's q4==F16 "loops" were real shouts in the clip).
2. **Verify a debug gate exists before trusting measurements taken under
   it.** The investigation handoff prescribed `CRISPASR_NO_NGRAM_LOOPFIX=1`
   for raw decode; that env var never existed in the codebase, so every
   "raw" baseline in the prior audit had `fix_loops` silently active.
   One `grep -r` would have caught it. (We added a real global gate,
   `CRISPASR_NGRAM_LOOPFIX_OFF=1` in `core/ngram_loop_fix.h`, so raw
   decode is now actually reachable.)


## A faithful port can still loop: get the bf16 blueprint's behaviour on the SAME audio before hunting a runtime bug (mega-asr, #218, 2026-07-10)

After the Q8-tower rebake fixed qwen3-asr 0.6b and ja-anime, mega-asr
(a LoRA fine-tune of Qwen3-ASR-1.7B) STILL degenerated on un-chunked
145 s audio at every quant level. Instead of another runtime hunt, we ran
the bf16 blueprint on the same clip (Kaggle, raw greedy decode): the bf16
BASE model is clean (max phrase-cycle 3, auto and forced-English), the
bf16 LoRA-MERGED mega loops massively (phrase-cycle 115, both modes; the
assistant prefill doesn't prevent it). Verdict: the degeneration is
**LoRA-induced and model-inherent** — the port reproduces the blueprint
faithfully, there is nothing to fix in the runtime, and the correct
posture is `fix_loops` + the 30 s chunked default, with
`--chunk-seconds 0` unsupported for mega at ANY precision. Same rule that
closed the vibevoice BGM case (#171): **run the original pipeline; when
the blueprint reproduces the symptom, stop debugging the port.**

Blueprint behaviour can itself be the surprise, so establish it before
judging outputs against intuition: glm-asr's bf16 single-pass SKIPS ~70 s
of quiet leading audio on this clip (the 30 s-chunked path covers more
content); the windowed qwen3 blueprint transcribes noise sections
aggressively with collapsed "hey" runs of its own. Not every repetition
or omission is a port bug.

Practical notes for LoRA blueprints on Kaggle: an adapter declaring
`target_modules='.*'` cannot be wrapped by peft — hand-merge
`W += scale·(B@A)` per `.lora_A/.lora_B` pair, stripping
`base_model.model.` and rewriting `thinker.layers.` →
`thinker.model.layers.`; qwen_asr needs `transformers==4.57.6` plus a
force-reinstalled `huggingface_hub==0.36.0` + `hf_transfer` and a
`sys.modules` purge of the pre-imported hub.


## gallocr does NOT protect input-flagged tensors across computes — re-set EVERY input before EVERY compute of a persistent graph (omnivoice #245 follow-up, 2026-07-11)

Converting omnivoice's per-step graph rebuild into a persistent
build-once/compute-many graph (fixed T, no KV — the textbook case) changed
the output *deterministically* on CPU. The graph audit showed nothing: every
input (`input_embeds`, `pos_ids`) was written, no read-before-write tensor
existed. The bisect that found it: dump a per-forward logit checksum under
both modes (`OMNIVOICE_DEBUG_SUM`) — **step 0 was bitwise-equal, step 1+
diverged**. That signature means the first compute *clobbers* something the
second compute reads: `ggml_gallocr` may alias an input-flagged tensor's
slot with intermediates of the graph, so `pos_ids`, set once at init, read
back garbage from the second compute on. The old per-call path was correct
only by accident — it re-set the positions before every (first-and-only)
compute.

Rules: (1) when converting per-call graph builds to a persistent graph,
treat **all** `ggml_set_input` tensors as volatile — re-set each one before
every compute, even "constant" ones like positions (the cost is trivial);
(2) validate the conversion where the backend is deterministic (CPU) with a
byte-compare — Metal is nondeterministic run-to-run at fixed seed (three
different WAV md5s from identical binaries), so md5 there proves nothing;
(3) the step-0-equal/step-1-diverged bisect discriminates static
graph-construction differences from cross-compute state carryover in one
experiment. Related: the sched-level graph-cache entries above (#171, #220,
#235) — this is the gallocr-level sibling of the same class. Payoff here:
CPU 1.64× / Metal ~2× per synthesis, WAV byte-identical.


## Generation-config defaults are part of the blueprint contract — a masked-iterative TTS with the wrong sampler knobs degenerates to SILENCE, not to bad audio (omnivoice §234, 2026-07-11)

omnivoice synthesized near-silence (peak ≤0.009, whisper "[ Silence ]") on
every platform, precision, and prompt variant, deterministically. The port's
tensors were fine: text ids byte-exact vs tokenizers-lib, embedding
offsets/mixing identical to `_prepare_embed_inputs`, codec innocent. The bug
was five **generation-config fallbacks** that didn't match the blueprint's
`OmniVoiceGenerationConfig` dataclass: guidance 1.0 (ref 2.0), class_temp
0.7 (ref 0.0 = argmax), position_temp 4.5 (ref 5.0), layer_penalty 0.5 (ref
5.0 — the coarse-to-fine codebook unmask ordering), t_shift 1.0 (ref 0.1 — a
different unmask-schedule *shape*). Under the wrong decode policy the
model's highest-confidence predictions are silence codes, and the
confidence-ranked unmasker locks them in: 59 unique tokens across 1344
positions. With blueprint defaults: 278 unique codes, roundtrip VERBATIM.

1. **Diff the sampler/scheduler defaults against the blueprint's config
   dataclass line-by-line** — same discipline as prompt scaffolds (glm #218)
   and for the same reason: they're training-time contracts. A `>0 ?
   param : fallback` pattern makes the fallback THE user-visible default;
   plausible-looking placeholder values shipped as if tuned.
2. **Degenerate-output triage order that worked (one run each):** (a) codes
   histogram (`OMNIVOICE_DEBUG_CODES=1`) — degenerate codes ⇒ codec
   innocent, LLM/sampling side guilty; (b) token ids vs HF tokenizer;
   (c) embedding table offsets/mixing vs the reference module; (d) the
   generation-config diff. Silence ≠ "encoder broken": a healthy codec
   rendering silence codes IS silence.
3. A MaskGIT-style sampler has more load-bearing knobs than an AR sampler —
   schedule shape (t_shift), unmask ordering (layer penalty, position
   temperature), and CFG scale all change WHAT gets committed WHEN;
   "conservative" values don't fail loud, they converge to the model's
   safe token, which for audio is silence.


## Vocoder/codec conv graphs hide three graph-construction wastes that dwarf the actual conv FLOPs — profile per-node before assuming it's inherent (#245/§232 qwen3-tts codec, 2026-07-11)

The qwen3-tts codec decode read as ~3-4 s (RTF ~1.0 by itself, half the whole
pipeline wall) and looked like honest vocoder compute. A per-node trace
(`QWEN3_TTS_CODEC_TRACE`, an eval-callback that times + names + shows the
backend of every node) proved otherwise: the wall was three kinds of
graph-construction waste, none of them the K=7 SEANet convs that are the only
legitimately expensive part. All three recur in any im2col-based conv1d stack
(SEANet/HiFi-GAN/DAC/EnCodec/ConvNeXt decoders), so they're worth a checklist:

1. **K=1 convs routed through `ggml_conv_1d`/im2col.** A 1×1 conv is a channel
   matmul, but the generic helper still does transpose → im2col → matmul →
   transpose. The im2col of a K=1 conv is a *pure copy* of the input, and at
   24 kHz T it materializes ~300 MB F32 intermediates at ~75 ms each — ×12
   sites here (every residual unit's `conv2`). Fix: detect `K==1 && stride==1`
   and emit `ggml_mul_mat(reshape_2d(w), x)` directly on the `[C_in,T]` layout,
   no transposes, no pad, no im2col.
2. **Causal left-pad `ggml_pad_ext` nodes silently placed on the CPU backend.**
   Metal rejects left/asymmetric PAD (`GGML_OP_PAD` needs left-pad==0), so the
   scheduler routes every causal conv's explicit pad node to `backend_cpu`,
   forcing a graph split + a cross-backend copy of the (padded) activation both
   ways — per conv, per decode. Fix: don't emit a pad node at all — pad *inside*
   im2col (`ggml_conv_1d(w, x, s, /*p=*/K-1, d)`) and crop to the first `T_out`
   columns (`ggml_view_2d`). Bit-identical, and the whole CPU-placed subgraph
   disappears. (Same trick as `voxcpm2_tts.cpp` `causal_conv1d_ggml`; also the
   Metal-asymmetric-PAD note in `docs/contributing.md`.)
3. **Per-graph F16→F32 kernel casts.** Our ggml fork's `ggml_conv_1d`/`_dw`
   cast the conv *kernel* to F32 inside every graph when the activations are
   F32 (so `mul_mat` gets an F32 src1). For multi-MB kernels that `ggml_cast`
   runs every decode — the `in_conv` cast alone was ~70 ms/frame. Fix: bake
   F32 copies of the K>1 conv kernels ONCE at load into a dedicated context/
   buffer and repoint the tensor pointers; the in-graph cast becomes a no-op.
   (~+55 MB resident; K==1 kernels excluded because the matmul path from #1
   consumes F16 directly.)

Result on M1 Metal (0.6B Q8_0): codec 3.9 → 1.3 s (~3×), WAV **md5-identical**;
on CPU 9.1 → 4.3 s (~2.1×), within 1 int16 LSB / PCM cos 1.00000000 (the K=1
matmul realizes slightly differently than im2col+matmul — an expected float-
order difference, not a bug; acceptance falls to decoded-output equivalence per
the house rule). Total 0.6B pipeline RTF 2.9 → 1.25. Peak intermediates roughly
halved as a side effect, which also cured a 16 GB-Mac 1.7B jetsam/OOM on long
outputs. Gated `QWEN3_TTS_CODEC_FASTCONV`, default ON. **Lesson:** before
accepting a conv/vocoder graph's cost as inherent, per-node-trace it — the
usual suspects are 1×1-through-im2col, asymmetric-PAD-shunted-to-CPU, and
per-graph weight casts, all of which are graph-shape bugs, not compute.


## Before optimizing a graph's *dispatch*, split host-encode vs GPU-execute — and never trust a GPU benchmark taken under load (talker §232, 2026-07-11)

Having won on the qwen3-tts code predictor by going sched-free, the obvious next
move was the same treatment for the talker AR step (~50 ms/frame on a quiet M1).
It would have been wasted work. `CRISPASR_METAL_PROFILE` (which splits each
`ggml_metal_graph_compute` into host encode+commit time vs GPU execute+sync
time) showed the 1068-node talker step is **encode 2-3 ms, GPU-execute
38-42 ms** — GPU-execution-bound. Sched-free dispatch or an MTLIndirectCommand
Buffer replay can only remove *host* encode time, so they can't help here (same
verdict as the §210 granite ICB probe). Confirmed empirically: the sched-free
bucket path computes in ~56 ms/step vs ~54 ms dynamic. The op profile explains
the floor — ~460 real ops/step, `mul_mat` only ~24 %, ~85 `cont` copies plus
hundreds of tiny norm/add/rope kernels; it's death by kernel *count*, and the
only lever left is node-count slimming inside the shared `core_attn::kv_self_
attn` (30+ backends, high regression risk) → deprioritized with data instead of
attempted.

Two reusable rules:
- **Attribute host vs device before choosing a dispatch optimization.** "It's
  slow and it's a small model" does NOT mean it's dispatch-bound. If GPU-execute
  dominates, no amount of gallocr/ICB/graph-reuse cleverness moves it; you need
  fewer/bigger kernels, not fewer dispatches.
- **A GPU benchmark taken under contention is noise.** On this 16 GB M1 with
  other sessions running, `CRISPASR_METAL_PROFILE` `gpu_us` for *identical*
  193-node code-pred graphs swung 4-17 ms purely from WindowServer/browser GPU
  time-slicing, and whole-pipeline ms/frame swung 2-3× run to run. Every number
  above was taken load-gated (1-min loadavg < 6, sustained across two checks) or
  as min-of-N interleaved reps. A "3× speedup" measured against a contended
  baseline can be entirely the baseline degrading — interleave old/new in the
  same quiet window before believing a ratio.


## Before starting a "top-priority, profiled" perf target: check whether it's already been tried-and-rejected, and whether a parallel session owns it (§235 triage, 2026-07-11)

Picking the next optimization after the §232 qwen3-tts sweep, the obvious #1 was
PLAN §232's own top item: "port the Parakeet/Nemotron transducer decoder to GPU"
— the decode loop demonstrably runs on host `cblas_sgemv/sgemm`, and the profile
gap vs a competitor was 12-19×. It is a **trap**, and two cheap checks caught it
before any code:

1. **`git log --grep` for prior attempts.** A parallel session had already tried
   it and recorded the result: batched/GPU transducer decode is **5-9× SLOWER**
   (`d0bb4601`, `3fcd5ff3`). A transducer decode step is a handful of tiny
   matmuls (LSTM predictor + joint) with a host↔device sync every step; that is
   precisely the regime where CPU `cblas` beats the GPU (the same lesson as
   qwen3-tts CP_DIRECT being 2× *slower* on CPU, and f5/native-on-GPU losing to
   custom-op-on-CPU). A big *paper* FLOP win on a fine-grained decode loop
   routinely goes **negative** once per-step launch + sync dominate. So: a
   PLAN item marked "[HIGH] 12-19×" is a hypothesis, not a fact — grep for
   whether someone already measured it.
2. **`git worktree list` + recent `git log` for ownership.** The next candidate
   (Moonshine encoder im2col) was being actively worked in another session's
   worktree (`moonshine-decode-stash`); starting it would have collided. On a
   box with many concurrent Claude sessions all pushing to `main`, check who's
   on a target before opening a worktree for it.

The general rule: treat a profiled "top priority" as a lead to *verify*, not a
task to start. `git log --grep`/`--oneline` and `git worktree list` are 10-second
checks that turn "I nearly reimplemented a measured-negative optimization" into
"crossed off with evidence." Record the dead end in PLAN so the next person
doesn't re-derive it (the transducer GPU port is now explicitly marked DEAD in
§235, not silently left as a tempting [HIGH] in §232).


## CosyVoice3 CFM Euler steps: 6 is the sweet spot, and validate it with log-mel corr vs full-steps — NOT ASR roundtrip (§235, 2026-07-11)

Flow-matching CFM trajectories are nearly straight, so an N-step Euler solve
barely differs from a 10-step one down to surprisingly small N — the same
property that let chatterbox drop its S3Gen CFM default 10→6 (§207). Confirmed
for CosyVoice3's DiT-CFM (the single biggest CV3 cost, ~48 % of the wall):
`COSYVOICE3_FLOW_STEPS` swept 10/8/6/4 with a **fixed seed** (identical LM
speech tokens + init noise, so step count is the ONLY variable), quality read
as **log-mel-spectrogram correlation vs the 10-step output** — corr
8→0.9948, 6→0.9925, 4→0.9895. 6 steps is the sweet spot (≥0.98 parity, ASR
verbatim), ~−40 % flow work ≈ −19 % of the wall.

Two transferable methodology points, both learned the hard way this session:
- **ASR roundtrip cannot grade CFM step count.** It was verbatim at 10/8/6 and
  only slipped one word ("brown"→"bound") at 4 — long after the mel-corr starts
  drifting. ASR proves intelligibility, not fidelity; use log-mel corr vs the
  full-step render as the quality axis (chatterbox §207 established this; I
  re-derived it after an ASR-only sweep would have called 4 steps "fine").
- **Calibrate the spectral metric to the accepted threshold.** A raw
  linear-frequency **log-STFT** corr over-weights high-frequency detail (where
  under-stepping shows up) and read 0.958 at 8 steps — falsely alarming;
  the **log-mel** version (n_mels=80) read 0.9948 for the same audio, matching
  chatterbox's 0.98 parity threshold. Match the reference metric before
  applying its threshold, or the numbers don't transfer.
- **The whole-utterance corr under-weights localized artifacts.** At 4 steps
  corr was still 0.9895 (above 0.98) yet ASR misheard a word — the artifact was
  local and the utterance-averaged corr washed it out. When corr and a
  token-level signal disagree near the boundary, trust the more sensitive one
  and stop there. Final arbiter for any quality-affecting DEFAULT remains a
  human listen on 2–3 diverse sentences — recommend, don't unilaterally flip.

## A "GPU" model can be silently running on CPU — audit for hot graphs pinned to backend_cpu and CLI adapters that never forward use_gpu (f5-tts / moonshine, 2026-07-11)

Two backends looked like they "supported GPU" but never used it, and both cost
real time before anyone noticed:

- **moonshine**: the CLI adapter zero-initialized `moonshine_init_params` and
  never set `use_gpu`, so every run — *including the §232 Kaggle "GPU"
  benchmarks* — used the CPU backend. The whole "moonshine encoder is 6-21×
  slower than transcribe.cpp" investigation was chasing a CPU-vs-GPU comparison
  mislabeled as GPU-vs-GPU. One line (`mp.use_gpu = params.use_gpu`) → encoder
  4-6× faster on Metal. The tell: the model-load banner lacked its `(GPU)`
  suffix and `CRISPASR_METAL_PROFILE=1` printed zero `[metal-prof]` lines.
- **f5-tts**: the DiT graph (the dominant compute) was hardcoded
  `ggml_backend_graph_compute(ctx->backend_cpu, gf)` with the gallocr bound to
  backend_cpu's buffer type, regardless of `use_gpu`. Weights loaded on
  `ctx->backend` (GPU), so on Metal it *worked* — unified memory let the CPU
  backend read the GPU-resident weights — and just ran every forward on the CPU
  (sampling showed `ggml_vec_dot_f16_f32` / flash-attn in libggml-cpu). Compute
  on `ctx->backend` instead (== backend_cpu when use_gpu is off) → 7.8×/step.

**Audit recipe when a backend feels slow on GPU:** grep for
`ggml_backend_graph_compute(.*_cpu` and `gallocr_new(...buffer_type(.*_cpu` in
the hot path; check the CLI adapter actually forwards `params.use_gpu`; confirm
the load banner says `(GPU)` and `CRISPASR_METAL_PROFILE` emits per-graph lines.
On unified-memory Metal a CPU-pinned graph never errors — it silently computes
on the CPU while claiming the GPU backend.

Corollary — **low-step "degenerate on both CPU and GPU" is not automatically a
quality floor.** f5 produced garbage at 4 steps on *both* backends; the first
read was "4 steps is below the model's floor." It was actually a shared bug:
`pos_in` set once but the gallocr re-aliased its slot every step, corrupting
RoPE positions from step 1 (the omnivoice §245 input-aliasing class). Fixing it
made 16-step output roundtrip verbatim. When both backends agree on garbage,
suspect a shared bug before blaming the step count.

## Micro-optimizing an already-BLAS'd path with a GPU-ggml port is usually a dud — check for existing Accelerate/cblas and measure encode-vs-execute BEFORE porting (f5 / titanet perf sweep, 2026-07-11)

After the f5 base fix (7.8×, a real *bug*), the "obvious" follow-on GPU
optimizations were all measured duds, and the pattern is worth internalizing:

- **f5 batched CFG** (run the 2 CFG DiT passes as one B=2 graph): implemented,
  correct (corr 1.00000 vs sequential, 4D-RoPE + batched flash-attn work on
  Metal) — but NO speedup. The Metal profile (979-node DiT, ~1.5 ms host-encode
  vs ~1-2 s GPU-execute/forward) proved it's compute-bound, so B=2 just doubles
  per-dispatch work. Batching only helps *dispatch*-bound graphs.
- **f5 F16 activations** (cast QKV/FFN matmul inputs to F16): correct, but
  slightly *slower* (added cast nodes) → the DiT isn't matmul-bound either. The
  ~0.8-1 s/forward is the ~800 small ops + flash-attn, which neither lever hits.
- **f5 host input-embed → GPU** and **titanet GPU port**: both already run on
  **Accelerate BLAS** (`cblas_sgemm`) on CPU. titanet's *default* path is legacy
  Accelerate (~1 s/forward); its ggml-CPU path is 1.7× *slower*, so a GPU-ggml
  port would have to beat Accelerate with a graph that already loses to it.

The durable rule: **the big wins are bugs, not micro-opts.** Before porting a
hot path to the GPU, (1) check whether it's already `cblas`/Accelerate-optimized
— if so a naive ggml graph often loses; (2) split host-encode vs GPU-execute to
know if you're dispatch- or compute-bound — batching and precision changes only
help the former. Guess-and-check across an already-optimized seam burns hours
for nothing; a per-op Metal capture is the only honest next step there.

27. **VoiceDesign == different model, not just + caption encoder.** Irodori-TTS
600M-v3-VoiceDesign has completely different dimensions from the base v3 500M
(model_dim 1280 vs 2048, latent_dim 32 vs 128, text_dim 512 vs 1280, 12 vs 24
DiT layers, llm-jp-3-150m tokenizer vs sarashina2.2). The converter/runtime
already read hparams from config/GGUF dynamically, so the dimension changes were
handled — but the assumption that "VoiceDesign = base + caption" would have led
to hardcoded wrong dimensions. Always check the actual checkpoint config.

### #81: profile per-node before trusting a handover's bottleneck theory — the "memory traffic" was two F16 matmuls the quantizer silently skipped

The FastConformer CPU-perf handover attributed the 1.5× gap vs onnx-asr to
ggml's per-op intermediate materialization and proposed cont-reduction /
gallocr / fusion work. A 30-line sched-eval-callback profiler
(`CRISPASR_FC_PROFILE=1`, aggregate by op+shape) showed conts+casts ≈ 3%,
sched ≈ 1% — and **35% of encoder time in `conv.pw1/pw2` F16 matmuls**.
The converter writes those as 3D conv tensors `(1, d, N)`; the quantizer's
`ggml_n_dims(t) == 2` rule skips them; so every quantized FastConformer
GGUF shipped with F16 pointwise-conv weights running on ggml's slow
(~6× vs Q8_0) CPU F16 mul_mat path. Three durable lessons:

1. **Grep quantized GGUFs for leftover F16 2D-consumed weights** — a
   tensor stored 3D but consumed as `reshape_2d + mul_mat` is invisible
   to dimension-based quantize rules. Check with
   `GGUFReader` per-tensor types on layer 0, not by assuming "q8_0 model".
   Fixed at both ends: `crispasr-quantize` now quantizes
   `*.conv.pw[12].weight` as its 2D view; runtimes repack old GGUFs at
   load (`core_conformer::repack_conv_pw_q8`, `CRISPASR_FC_PW_Q8`).
2. **ggml CPU F16 mul_mat is a perf trap** (~30 GFLOP/s vs ~176 for q8_0
   on M1) — same class as the F16-cast-per-graph conv trap (§conv1d
   hygiene), but for matmul weights.
3. **`GGML_BLAS=ON` does nothing for a backend whose sched only registers
   {gpu, cpu}** — the blas backend must be in the sched's backend list to
   ever be scheduled. The VPS "OpenBLAS build" benchmarks never used it
   in the FastConformer path.

Bonus (bit-identical, gated): fused QKV (`CRISPASR_FC_FUSED_QKV`, one
matmul + view-split; concat weights at load) and strided
`flash_attn_ext` inputs (drop Q/K/V `ggml_cont`s; `CRISPASR_FC_ATTN_CONT=1`
restores). Verified bit-identical on the full CTC logit grid — view-split
matmul and strided flash reads change nothing numerically. Combined:
parakeet-ctc q8_0 M1 CPU 1.13→0.73 s (re-quantized GGUF: 0.60 s, 18.4× RT).

### #81 GPU phase: a finite attention mask constant is not a mask — pad garbage overruns it a few blocks deep

Building the bucketed-padding path (pad T to a bucket, mask pad frames)
for the FastConformer encoder produced transcripts that matched but
logits that drifted ~0.7 — and the drift ENTERED at block 3, not block 0
(bisected with `CRISPASR_FC_MAX_LAYERS=N` truncation). Three lessons:

1. **Mask pad KEYS with `-INFINITY`, never `-1e9`.** Pad columns carry
   untrained garbage that grows across blocks; by block ~3 their QK
   scores reach 1e8+ and a finite mask constant no longer dominates —
   pad keys get real softmax weight. `-inf + finite = -inf` always.
   (`make_local_attn_mask`'s -1e9 is fine for its use — masked positions
   there hold trained-scale values.) Also re-zero pad columns at every
   block boundary so 0-weight × huge-value artifacts can't arise.
2. **Bitwise equality across different tensor shapes is bounded by
   size-adaptive GEMM kernel selection.** After fixing every structural
   leak (attention keys, dw-conv inputs, pre-encode stage boundaries),
   a rel≈3e-9 seed remained at the pre-encode matmuls: im2col GEMMs pick
   different micro-kernels for different widths, regrouping the same dot
   products (ULP reassociation), and 24 LayerNorms amplify it to ~0.7 in
   logits. Same accepted class as BLAS-variant drift — acceptance is
   identical decoded output, and bit-equality only when shapes coincide.
   Corollary: when a "leak" survives every masking fix, check whether
   the first divergence is ULP-sized before hunting more structure.
3. **Build worktrees with ABSOLUTE paths** (`cmake --build $W/build`).
   A shell cwd reset mid-session sent two rebuilds into the main tree's
   build dir; the stale worktree dylib then "disproved" a correct fix
   (same max_abs before/after). `strings <dylib> | grep <new-env-var>`
   is the 5-second staleness check.

### #81 endgame: three more durable lessons from the fleet requant + CUDA phase

1. **A documented op-fallback note is an unexploited perf bug.** The
   `fattn.cu` per-head-mask guard (`mask->ne[2] != 1` → flash rejected on
   CUDA) had been sitting in PERFORMANCE.md's CUDA op-support table since
   May as "the dominant CPU-fallback" — acting on it (manual QK^T +
   soft_max_ext attention, all-GPU) was a **3.2×** on long audio. When a
   support-table entry says "falls back to CPU on <backend>", treat it as
   an open perf ticket: either restructure the graph around the guard or
   fix the kernel upstream. `GGML_SCHED_DEBUG=2` per backend per GPU is
   the audit tool.
2. **Requantizing an old GGUF with today's quantizer is NOT identity —
   rules drift.** Old parakeet q4_k files kept the whole transducer head
   (decoder.embed, joint.out, joint.pred) at F16 under rules that no
   longer exist; a naive re-quant silently changes those tensors too. The
   safe fleet-fix pattern: same-type tensors byte-copy (so requant = the
   minimal intended diff), a structural gate that refuses ANY unexpected
   type/data change, per-file `--tensor-type` pins to reproduce retired
   rules, and strict transcript equality (old + runtime repack vs new)
   as acceptance. The gate caught both the rules drift AND a
   down-quantization bug (q8 pw → q4 on re-runs) before anything shipped.
3. **`HfApi.upload_file` can strand after its commit lands (CLOSE_WAIT)
   on Kaggle too** — not just on the Mac (memory said laptop-specific;
   wrong). A single-threaded pipeline wedges forever on one call. Pattern:
   run HF network ops on daemon threads with a join timeout, then verify
   server-side (list_repo_tree size match) and treat a hung-but-landed
   upload as success. Progress mirroring to a side repo (small JSON per
   file) is what makes a wedged cloud kernel diagnosable at all.

## Exact greedy code-parity is unachievable for a QUANTIZED AR audio LM — confirm the port with a step-0 logit-rank probe, not byte-exact codes (MOSS-TTS-v1.5 #249, 2026-07-13)

Greedy code-parity (C++ Q4_K vs the HF BF16 reference, temps=0) for the moss-tts
backbone matched only ~0.5% of frames and diverged from **frame 0** — which looked
alarming but is exactly what a correct port produces here. Two findings, one a real
bug and one an intrinsic limit:

1. **Real bug (fixed): tokenizer newline grouping.** The prompt tokenizer cloned
   from `qwen3_asr` used a crude whitespace pre-splitter that split `>` from a
   trailing `\n`. Qwen's pre-tokenizer regex `[^\s\p{L}\p{N}]+[\r\n]*` groups
   punctuation with trailing newlines (`>\n`=397, `):\n`=982, `:\n`=510). A
   dedicated `moss-tts-promptdiff` kernel isolated it: the prompt TEXT was
   identical, only the tokenization of newlines differed (first mismatch at the
   first `>\n`). Fix = a proper Qwen2/3 pre-tokenizer (`mt_qwen_pretokenize`,
   contractions | prefix+letters | digit | punct+trailing-nl | ws+nl | ws). After
   it the 67-token conditioning prompt is **byte-identical** to the HF processor
   (`first_mismatch=null`). **Gate 0 for any text→audio LM: diff the col-0 prompt
   ids against the HF processor and require byte-identity BEFORE looking at codes.**

2. **Intrinsic (not a bug): quantization near-tie onset flip + AR chaos.** Even
   with byte-identical prompts the codes diverge from the very first token. The
   decisive `moss-tts-logit0` probe (`moss_tts_debug_first_audio_logits` — prefill
   the prompt, dump head-0 logits, no 8B reference, no F16 download) showed the
   reference's greedy pick (code 143) is the C++ Q4_K distribution's **rank-1
   runner-up**, only **0.135 logits** below the C++ argmax (1021). Q4_K rounding
   (O(0.1–0.5) on logits) flips this near-tie. Because the MOSS delay makes every
   un-delayed frame span many raw AR steps, one flip cascades: ~n_vq·T greedy
   decisions at even 99.9% per-decision agreement → ~0.1% joint survival, i.e. the
   0.5% observed. Codebook-0 (coarse RVQ) still re-syncs to exact reference values
   at scattered frames (807, 578, 756; 2-frame-shifted runs 400/575/254) — the
   models produce the same coarse audio, differing only on quant-sensitive fine
   residuals. **This extends the F16 voxtral-tts lesson (exact at frame 0, diverges
   by frame ~7) to the QUANTIZED case where even frame 0 flips** — and the
   logit-rank probe is what distinguishes an expected near-tie flip (rank ≤1–2,
   gap <~1 logit) from a real head/numerics bug (ref pick far down). The only
   acceptance gate for a quantized AR audio LM is the ASR round-trip (HARD RULE #3),
   never byte-exact greedy codes.

**Voice cloning (same port) validated the encoder path the cheap way too:** a
self-contained closed-loop kernel (`tools/kaggle/moss-tts-voiceclone`) — synth a
reference clip R, clone its voice onto a new sentence C (`--voice R
--i-have-rights`), and a no-voice baseline B — gates on (a) `--voice` runs clean +
non-silent, (b) ASR(C) reproduces the clone text, (c) speaker-embedding
cosine(C,R) > cosine(B,R) via Resemblyzer. Result on Q4_K: cos 0.845 vs 0.585
(Δ+0.26) with the words preserved. Harness gotcha: the crispasr whisper CLI
interleaves device/model-load noise (and an appended "generated by artificial
intelligence" disclosure) on the transcript lines — feed the FULL ANSI-stripped
stdout to a word-OVERLAP check (present-in, not exact-match); a `[-400:]` tail
truncation dropped the real transcript and faked a FAIL.

## A GPU-vs-reference RTF gap that survives kernel parity is per-step HOST detours — fuse them into the step graph; unified-memory Metal hides exactly this class of cost (omnivoice #254, 2026-07-16)

After single-shot synthesis landed, omnivoice on the reporter's RTX 5070 Ti was
still RTF 0.17 vs omnivoice.cpp's 0.144 — with the forward kernels already at
parity (~34 ms/step both). The residual was entirely between the forwards, on
the host, per MaskGIT step: (a) audio-embedding lookup via a separate get_rows
graph with an ~18 MB F32 readback, a single-threaded codebook sum, and a ~5 MB
re-upload of the summed embeds; (b) a full-sequence logits readback (~39 MB,
out_dim×T_c) when only the target slice is consumed; (c) single-threaded
triple-log-softmax CFG scoring (~13M `exp` per step); (d) multi-MB per-step
vector allocs incl. a full 18 MB `u_logits` copy in the interval-CFG path.
~44 ms/step of pure host work next to a 27 ms forward.

**The fix — fused step graph** (`OMNIVOICE_FUSED_STEP`): the shifted audio-token
ids (140 KB of int32) are the ONLY per-step upload. In-graph: get_rows +
codebook sum + concat with a device-resident text-embedding tensor (uploaded
once into its OWN backend buffer — gallocr-managed tensors get aliased, a
dedicated buffer is exempt); after the transformer, `ggml_cont` views emit ONLY
the target logit slices. CFG scoring threads over target positions. Verdict on
the reporter's box: gen 3.55 s → 1.53 s (2.3×), RTF 0.07 — ~2× faster than
omnivoice.cpp — and ONE CUDA-graph warmup for all 32 steps.

Durable techniques from the byte-identity requirement (codes cmp-identical to
the legacy path on M1 Metal across 2-forward / unified / interval-CFG K=2 /
guidance=0, AND on the reporter's CUDA box):

1. **In-graph embedding sums can be bitwise-equal to a host sum** if the add
   ORDER matches: chained `ggml_add` over per-codebook views in cb-ascending
   order reproduces the host `dst[j] += src[j]` loop exactly (elementwise F32
   IEEE add, no reassociation). Order-matched graph math is what turns a perf
   rewrite into a cmp-verifiable no-op.
2. **Threading a sampling loop without changing output:** draw the Gumbel noise
   SERIALLY first (same rng object, same t-outer/cb-inner order as the serial
   loop), then consume the precomputed array from worker threads. The rng
   stream — and thus the output — is bit-identical. Data-dependent draw counts
   (class_temp>0 top-k sampling) can't be precomputed → that path stays serial.
3. **Platform asymmetry: judge host-detour optimizations on a DISCRETE-GPU
   box.** On M1 the same change is NEUTRAL (matched-load per-stage A/B: 96.3 s
   fused vs 93.7 s legacy for the same paragraph; legacy's embed prep cost 0.9 s
   of ~94 s) because unified memory makes the readback/re-upload nearly free
   and Accelerate eats the sums. The 2.3× only exists where host↔device
   round-trips and single-thread scoring are real costs. Inverse of the usual
   lesson: this time the M1 could not SHOW the win, not fake one.
4. **ggml fork CUDA-graph facts** (checked in `ggml-cuda.cu`): capture state is
   keyed PER cgraph (`ggml_cuda_graph_get_key` = nodes[0]), so several
   alternating persistent graphs each keep their own capture — alternation is
   NOT the enemy, per-step REBUILD is. And `TAG_GET_ROWS_CUDA_GRAPHS` disables
   capture for any graph containing GET_ROWS on a *quantized* table (host sync
   in the kquant path) — omnivoice survives because the quantizer keeps
   audio_embd/audio_output/llm.token_embd at F16. Audit the embd dtype BEFORE
   fusing embedding lookups into a forward graph on another backend.
5. **Reference-voice disk cache** (reporter parity ask, shipped as
   `CRISPASR_OMNIVOICE_VOICE_CACHE`): the RVQ ref encode is deterministic →
   cache content-addressed (FNV-1a over the PREPROCESSED pcm + an
   encoder-weight fingerprint), OVC1 header + int32 codes, atomic tmp+rename,
   same dir resolution as the pocket_tts latents cache. Second run loads in
   ~0 ms; audio data chunk bit-identical (only the timestamped C2PA chunk
   differs — compare WAV `data` chunks, not whole files).

Bench-hygiene reminders this session re-taught: a loaded shared M1 (loadavg
100–290) inverted the Metal A/B verdict TWICE before matched-load per-stage
medians settled it (the tell: the arms' identical decode code differing 3.8×);
and `curl -C -` resume on top of a 29-byte HTML error stub produces a
GGUF-with-garbage-prefix that parses as `invalid magic 'Inva'` — delete before
retrying, never resume a file you haven't validated.

## `git apply --3way` STAGES its result — a later `git add X && git commit` sweeps it up

Applying a patch with `git apply --3way` leaves the result **staged**, not just
in the working tree. So a subsequent `git add <unrelated files> && git commit`
commits the patch too, under a message that says nothing about it. That is how
the `group_words_into_speaker_runs` diarize DRY refactor landed inside
`29e3e7af6`, a commit labelled "mel-band-roformer reference dumper" — and it
was pushed to shared `main` before its unit tests had been run (they pass:
21/21, including the pre-existing `apply_pyannote` / `score_speaker_for_range`
cases, which is what makes the refactor provably behaviour-preserving — but the
verification happened in the wrong order).

Habits: after `git apply --3way`, run `git status` and either commit the patch
deliberately on its own or `git restore --staged` the parts you are not ready
to ship; and prefer `git commit <explicit paths>` over `git add X && git commit`
when the tree holds more than one logical change. On a repo with concurrent
sessions pushing to `main`, the mistake is not fixable by amending — history is
already shared — so the correction has to be additive, like this note.

The refactor itself is real and worth knowing about: the word→speaker-run
grouping + short-run stability fold used by the pyannote/sherpa segment
splitters now lives in one pure, unit-testable helper,
`crispasr_diarize_internal::group_words_into_speaker_runs()`
(`src/crispasr_diarize.cpp` / `_internal.h`), instead of being duplicated per
splitter. `min_run_cs <= 0` disables the fold (split on every speaker change);
50 cs is the pyannote default that suppresses one-word track-index flips.

## A normalization stage amplifies upstream float error — input-align it before trusting its cos (§248 MBR)

Mel-Band RoFormer's `band_split` starts with RMSNorm
(`F.normalize(x,-1)*sqrt(dim)*gamma`). Diffing `band_split_out` off OUR STFT
scored cos=0.976 and looked like a port bug — but the band_split MATH was
already proven exact (an identical numpy recompute from the fixture's own
`band_gathered` matched to cos=1.0). Cause: RMSNorm divides by each band's L2
norm, so on near-silent bands it amplifies the ~5e-4 float-level difference
between our radix-2 FFT and torch's STFT into a large relative error. Measured:
a *uniform* ±5e-4 perturbation on `band_gathered` alone tanks band_split to
cos 0.73 (our real, correlated error did better at 0.976).

Lesson (a sharper form of "gate input alignment before trusting per-layer cos"):
when a stage BEGINS with a normalization (RMS/Layer/L2), a cos taken off your
own upstream output conflates two things — your stage's correctness and the
upstream's float rounding, magnified. Diff that stage fed the REFERENCE's input
tensor, so it tests only its own math; validate the upstream separately by its
own pre-norm stage (here stft_packed, cos=1.0). After input-aligning,
band_split_out was cos=1.000000 (max_abs 1.5e-3 = f16 weight rounding). The
same input-alignment applies to every later MBR stage (each RoFormer block also
opens with RMSNorm). General rule for the harness: prefer feeding each stage the
reference input over chaining your own outputs, whenever the stage's first op is
scale-sensitive.


## CI has no NVIDIA GPU — a CUDA-only graph path (mimo RVQ) is guarded by an on-Kaggle exact-parity smoke, not CI (#309)

`perf(mimo-tokenizer): move RVQ to CUDA` (#309) runs MiMo's 8-stage Euclidean
RVQ inside the encoder graph when the tokenizer weights are CUDA-resident:
`scores = 2*(e·x) - ||e||^2` (mul_mat + scale + sub of precomputed codebook
norms) -> `argmax` (== `argmin ||x-e||^2`) -> `get_rows` residual. The argmax
output is contiguous, so it sidesteps the CUDA `get_rows` non-contiguous-index
abort; `mul_mat_set_prec(F32)` keeps the dot products exact.

The catch: the release CI matrix (linux-x86_64, linux-vulkan, macos, wasm, ios,
android) has NO NVIDIA GPU, so the whole `#if defined(GGML_USE_CUDA)` RVQ branch
COMPILES but never EXECUTES in CI. A green CI is necessary but NOT sufficient for
this path. Its regression guard is the model smoke tool's exact CPU/CUDA compare,
run on a CUDA box (Kaggle):

```
CRISPASR_MIMO_SMOKE_GPU=1 CRISPASR_MIMO_TOK_VERIFY_RVQ=1 \
    ./mimo-tokenizer-smoke mimo-tokenizer-q4_k.gguf samples/jfk.wav
# -> [ ok ] rvq_cpu_gpu_compare  codes=2208 mismatches=0
```

Validated on an RTX 2080 Ti: 2,208 codes, 0 mismatches, 5.65x faster than the CPU
fallback. Any future edit to the RVQ or encoder graph must re-run this on real
CUDA — do not trust CI alone. Rollback / diagnosis: `CRISPASR_MIMO_TOK_CPU_RVQ=1`
forces the CPU argmin path. (General rule: whenever a perf change adds a
`GGML_USE_CUDA`-gated compute path, name its Kaggle parity check in the PR, since
CI cannot cover it — same lesson as the parakeet/nemotron ggml-decode A/Bs.)
