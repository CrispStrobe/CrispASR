# CrispASR and the EU AI Act

CrispASR's position under **Regulation (EU) 2024/1689** (the EU AI Act): which
obligations attach to this software, which ones this project discharges in code,
which ones it cannot discharge for you, and where the enforcement actually lives
so it doesn't rot.

**This is not legal advice.** The classifications below are this project's
reasoned position on its own software, not a regulator's ruling. If you deploy
CrispASR commercially in the EU, get your own review — especially for the
sections marked **deployer duty**.

---

## 1. Who is who

The Act assigns duties by role, not by who wrote the code.

| Role | Who that is here |
|---|---|
| **Provider** of the AI system | Whoever *places it on the market or puts it into service under their own name*. For the upstream models, that's OpenAI / NVIDIA / Mistral / Alibaba / … For a product you ship that embeds CrispASR, **that's you**. |
| **Deployer** | Whoever uses an AI system under their own authority — you, running `crispasr` on real people's audio. Most Art. 50 duties land here. |
| This project | Publishes a runtime under MIT. It is neither the provider of the underlying models nor the deployer of your pipeline. Its job is to make sure the runtime doesn't *stop* you complying, and to hand you the machinery you need. |

The practical consequence: **CrispASR can mark its output, but it cannot inform
your users.** Sections 5 and 6 split accordingly.

---

## 2. Timeline

| Obligation | Applies from |
|---|---|
| Art. 5 prohibited practices; Art. 4 AI literacy | **2 Feb 2025** — in force |
| Ch. V general-purpose AI models | 2 Aug 2025 — in force |
| Art. 50 transparency (most of it) | **2 Aug 2026** |
| Art. 50(2) marking, for generative systems placed on the market *before* 2 Aug 2026 | **2 Dec 2026** (per the Parliament/Council provisional agreement of 7 May 2026) |
| Annex III high-risk regime (Ch. III) | **2 Dec 2027** — deferred from 2 Aug 2026 by the Digital Omnibus |

Two cautions. The Annex III deferral was agreed in June 2026 but takes effect on
publication in the Official Journal — **treat it as likely, not as law**. And the
deferral moved only the *high-risk* regime: the Art. 5 prohibitions were never
deferred, and neither were the Art. 50 duties for emotion recognition and
deepfakes.

CrispASR was placed on the market before 2 Aug 2026, so its Art. 50(2) marking
deadline is 2 Dec 2026. It has marked its output since well before that.

---

## 3. The open-source exemption, and what it does not cover

Art. 2(12) exempts AI systems released under free and open-source licences.
CrispASR is MIT. That exemption is narrower than it sounds — it **does not
apply** to:

- **prohibited practices** (Art. 5),
- **high-risk systems** (Art. 6 / Annex III),
- **Art. 50 transparency obligations.**

So the FOSS licence buys nothing for exactly the three areas that a speech
toolkit touches. Everything below is assessed as if the exemption did not exist,
because for these purposes it doesn't.

This is also why "just gate it behind a flag" is not a sufficient answer to an
Annex III capability. A flag governs the *deployer's* context; the provider
obligation attaches to placing a high-risk system on the market at all.

---

## 4. Prohibited practices (Art. 5) — in force since Feb 2025

| Practice | CrispASR |
|---|---|
| **Art. 5(1)(f)** — inferring emotions of natural persons in **workplace or education** settings | **Capability removed entirely.** See §4.1. |
| **Art. 5(1)(g)** — biometric categorisation to deduce race, political opinions, trade-union membership, religion, sex life or sexual orientation | Not implemented. CrispASR infers no protected attribute. See §4.3 on language ID. |
| **Art. 5(1)(h)** — real-time remote biometric identification in publicly accessible spaces for law enforcement | Not implemented, and structurally prevented: the named-voiceprint path refuses to run in streaming/live mode at all. See [`diarization-speakers.md`](diarization-speakers.md). |
| Art. 5(1)(a)–(e) — subliminal manipulation, exploitation of vulnerability, social scoring, crime prediction, untargeted facial scraping | Out of scope for a speech runtime. |

### 4.1 Emotion recognition — removed, not gated

SenseVoice-Small is upstream a multi-task model: its CTC head emits an
`<|HAPPY|>` / `<|ANGRY|>` / `<|SAD|>` / `<|FEARFUL|>` / … marker alongside the
transcript. Surfacing that value would make CrispASR an **emotion recognition
system** under Art. 3(39) — "an AI system for the purpose of identifying or
inferring emotions or intentions of natural persons on the basis of their
biometric data."

That is prohibited outright in workplace and education settings, and high-risk
(Annex III(1)(c)) everywhere else. CrispASR's headline use cases include meeting
transcription with speaker diarization — which is *precisely* the workplace
context Art. 5(1)(f) names.

A consent flag was considered and rejected. It would have addressed Art. 5 (the
system would no longer be placed on the market *for* workplace emotion
inference) but not Annex III(1)(c), which attaches to shipping the capability at
all and is not covered by the FOSS exemption. Retaining it would have committed
this project to the full Chapter III provider regime by Dec 2027 — risk
management system, data governance, technical documentation, human oversight,
accuracy and robustness testing, quality management system, conformity
assessment, CE marking, EU-database registration, post-market monitoring and
serious-incident reporting — or to a documented Art. 6(3) derogation assessment
that still requires registration. For one low-accuracy field on one backend out
of 54, that trade is not close.

**What was removed** (audit of 2026-08-01):

- `emotion` field dropped from `sensevoice_result` (`src/sensevoice.h`) and from
  `crispasr_segment` (`examples/cli/crispasr_backend.h`).
- `"emotion"` key dropped from both JSON writers (`cli.cpp`,
  `crispasr_output.cpp`).
- `sensevoice_transcribe()` now **strips the annotation prefix**. It previously
  returned it verbatim, so every session-ABI consumer — Python, Rust, Go, Dart,
  Java, C#, JS, WASM — received transcripts that literally began
  `<|en|><|HAPPY|><|Speech|><|withitn|>`. That was simultaneously a transcript
  corruption bug and an emotion inference leaking onto every binding.

The marker is still *parsed*, because that is what keeps it out of the
transcript text — the classified value is then discarded. `sensevoice_result.raw`
still holds the unfiltered model output for byte-exact diff-harness parity
against the FunASR reference; nothing in CrispASR surfaces it and nothing should.

Guarded by `tests/test-no-emotion-recognition.cpp` (unit tier — no model
needed). SenseVoice keeps its ASR, its 50+ language coverage and its native
language ID, which is what it was worth having for.

### 4.2 What is *not* emotion recognition, and why

Three things in the tree look adjacent to §4.1 and are not the same thing.
Stated explicitly so the next reader doesn't have to re-derive it:

- **Emotion *conditioning* in TTS.** chatterbox's `exaggeration` scalar, Zonos's
  8-dimensional emotion vector, Irodori's emoji prosody control,
  kartoffel-orpheus's `{Speaker} - {Emotion}: text` syntax. These *generate*
  expressive speech from an operator-supplied parameter. Art. 3(39) is about
  "identifying or inferring emotions **of natural persons** on the basis of
  their biometric data" — there is no natural person and no inference. Steering
  a synthetic voice is not recognition, in the same way a `<angry>` tag in a
  screenplay isn't.
- **SenseVoice audio-event tags.** `<|Speech|>`, `<|BGM|>`, `<|Applause|>`, and
  also `<|Laughter|>` and `<|Cry|>`, surfaced as `audio_event`. Recital 18
  excludes "the mere detection of readily apparent expressions, gestures or
  movements ... unless they are used for identifying or inferring emotions."
  Detecting that laughter occurred in a recording is an acoustic-event label;
  it is not a claim about the speaker's emotional state, and nothing here maps
  it to one. **Don't build that mapping downstream** — the moment you read
  `Laughter` as "happy", you are the one operating an emotion recognition
  system, and §4.1's analysis becomes yours.
- **`sensevoice_result.raw`.** Holds the unfiltered model output, marker
  included, for byte-exact diff-harness parity against the FunASR reference.
  Nothing in CrispASR reads it and nothing should; it is a parity-testing field,
  not a workaround.

### 4.3 Why language identification is not biometric categorisation

CrispASR infers the *language being spoken*, not a property of the speaker.
Art. 5(1)(g) bites on categorisation that *deduces* race, ethnic origin or
another protected attribute. A language label is an attribute of the audio
signal, and the mapping from language to any protected attribute is neither
performed nor implied.

If you build something downstream that uses a detected language as a proxy for
national or ethnic origin, that inference is yours and Art. 5(1)(g) is your
problem. Don't.

---

## 5. High-risk (Annex III) — CrispASR is designed to stay outside it

| Annex III point | Status |
|---|---|
| **1(a)** remote biometric identification | **Deliberately not implemented.** See below. |
| **1(b)** biometric categorisation by sensitive attributes | Not implemented (§4.3). |
| **1(c)** emotion recognition | **Removed** (§4.1). |

Point 1(a) is the one a diarization feature could drift into. The constraints
that keep the named-voiceprint path outside it are load-bearing, not stylistic:

- **Off by default** behind `--speaker-db-consent`; enrollment hard-fails
  (exit 25) without it.
- **Closed claimed roster only** — `--expect-speakers` is mandatory. The tool
  confirms which *asserted, enrolled, consenting* participants speak. It cannot
  answer "who is this unknown voice?"; the open-ended 1:N search that defines an
  identification system is not implemented, at the CLI or at the C API.
- **Active involvement** — enrollment is a deliberate act by the enrolled person,
  with the consent attestation recorded in the `.spkr` profile. Art. 3(41)'s RBI
  definition turns on identification *without* active involvement.
- **Post-processing only** — no real-time or streaming identification path
  exists (cf. Art. 5(1)(h)).

The default and recommended path — `--diarize-speakers` — computes embeddings
per recording, clusters them into `(speaker 0)` / `(speaker 1)` labels, and
discards them. It identifies nobody and stores nothing.

Full write-up, including the GDPR Art. 9 obligations that apply to the named
path regardless of AI Act classification: [`diarization-speakers.md`](diarization-speakers.md).

**If you re-enable 1:N identification, remove the roster requirement, or add
real-time matching, you are building an Annex III(1)(a) high-risk system and
this document stops describing your software.**

---

## 6. Transparency (Art. 50)

### 6.1 Art. 50(2) — machine-readable marking of synthetic audio

**Provider duty, discharged in code.** Every path that produces synthesized or
substantially altered audio marks it, by default, on every surface:

| Surface | Marking |
|---|---|
| CLI (`--tts-output`, `--tts-stream`) | audio watermark + C2PA manifest |
| HTTP server (`/v1/audio/speech`) | audio watermark + C2PA manifest |
| Wyoming server (`--wyoming-port`) | audio watermark, always forced (raw PCM ⇒ no manifest possible) |
| C ABI (`crispasr_session_synthesize`, `_streaming`, `_speech_to_speech`) | audio watermark |
| WASM / JS (`ttsSynthesize`, `ttsSpeechToSpeech`) | audio watermark; `c2paSign()` available |
| All language bindings | inherit the C ABI — they cannot reach an unmarked path by accident |

Enumerate surfaces by **grepping for emitters** — `->synthesize(`,
`speech_to_speech(`, anything writing audio — not by reading this table. The
table is a summary of what was found; it is not evidence that nothing else
exists.

Two marking technologies, deliberately:

- **C2PA Content Credentials** — the interoperable, standards-based manifest, and
  the one a third-party tool will actually read. Native implementation
  (`third_party/c2pa-audio`), verified interoperable with the c2pa-rs reference
  reader in both directions. WAV (RIFF `C2PA` chunk), MP3 (ID3v2.4 GEOB), M4A/MP4
  (ISO BMFF `uuid` box). Raw ADTS `.aac` and Ogg `.opus` cannot carry a manifest,
  so they are remuxed to MP4 when C2PA is active.
- **Audio watermark** — survives re-encoding, transcoding and container loss,
  which a manifest does not. Spread-spectrum by default (band-limited to
  ~1.5–4.8 kHz so it stays inaudible); **AudioSeal** neural watermarking via
  `--watermark-model auto` for the stronger option.

**The watertight floor.** Opting out is possible but cannot produce a fully
unmarked file, on either surface. The rule is the same in both places: the
audio watermark may be dropped only when a C2PA manifest still marks the
output, so whenever the chosen container can't carry a manifest the watermark
is forced back on.

| Surface | Enforcement |
|---|---|
| CLI | `crispasr_enforce_cli_watermark_floor()` — per process, keyed on the output file extension. `--no-watermark` on a `.opus` output is overridden, with a printed notice. |
| Server | `crispasr_marking::container_marking_for_format()` — per *response*, since `response_format` is chosen per request. `pcm` / `f32` / `aac` / `opus` and the raw streaming path force the watermark on. |
| Wyoming | `crispasr_marking::decide_raw_surface()` — unconditional. The protocol emits raw PCM inside `audio-chunk` events, so no manifest is ever possible and no opt-out reaches it. |

The server's floor is per-response rather than per-process deliberately:
mutating the process-global flag would race across `--server-workers` threads.

This used to be a CLI-only guarantee. The server had no floor at all, so
`--no-watermark --accept-marking-responsibility` plus
`"response_format": "mp3"` returned audio with no watermark and — because C2PA
signing was hardcoded to the WAV branch — no manifest either, while the CLI in
the same configuration forced the mark back on. Same operator, same
attestation, two different floors. The server now signs every container that
can carry a manifest (WAV and MP3 today) and forces the watermark on the ones
that can't.

Any opt-out
(`--no-watermark` / `--no-c2pa` / `--no-spoken-disclaimer`) additionally requires
`--accept-marking-responsibility`, which writes a `[MARKING]` audit line
recording that the operator took the disclosure duty on themselves. The server
refuses to *start* with an opt-out flag and no attestation.

On the ABI, `crispasr_session_synthesize_raw()` returns unmarked PCM — for
callers that must resample, mix or concatenate *before* marking — and is
hard-refused (returns `nullptr`) until
`crispasr_session_accept_marking_responsibility()` has been called. Such callers
must then mark the result themselves via `crispasr_watermark_embed()`.

**Not marked, and why.** Art. 50(2) exempts systems performing "an assistive
function for standard editing" or not substantially altering the input data or
its semantics:

- **Transcription (ASR)** — produces a record of what a human actually said. No
  synthetic content is generated.
- **Translation** — generated text, but semantics-preserving by construction;
  this project reads it as standard assistive processing. The conservative
  reading is not obviously wrong, so if you publish machine-translated text at
  scale, consider marking it yourself.
- **Denoising and source separation** (rnnoise, HTDemucs, Mel-Band RoFormer) —
  removes or isolates existing signal rather than generating new content.
- **Speech restoration and upscaling** (Sidon, VoxCPM2 AudioVAE) — these *do*
  generate signal, and they **are** marked; they route through the watermarked
  S2S path.

### 6.2 Art. 50(4) — deepfake disclosure

**Deployer duty.** If you generate a voice clone, you must disclose that the
audio is artificially generated, **clearly and distinguishably, at first
exposure at the latest**. The Commission's guidance is explicit that this needs a
*visible or audible* label — a machine-readable watermark alone does not satisfy
Art. 50(4).

CrispASR gives you the audible label: voice-cloned output gets a **spoken
AI-disclosure prefix**, synthesized in a neutral voice, prepended to the clip.
It is on by default and skipping it requires an attestation
(`--no-spoken-disclaimer` + `--accept-marking-responsibility` at the CLI;
`"marking_attestation"` in the request body, or a server launched with
`--accept-marking-responsibility`, on the API).

**On the Wyoming surface it cannot be skipped at all.** That protocol has no
field a client could put an attestation in, and #312's rule is that an opt-out
requires one — so there is no opt-out to honour, and every clone served over
Wyoming carries the audible disclosure. For the same reason the *consent* gate
there falls back to the operator's launch-time `--i-have-rights`: a clone
requested from a server started without it is refused rather than served
ungated, matching the HTTP surface's hard refusal when `consent_attestation` is
missing. The rule is `crispasr_marking::decide_raw_surface()`, unit-tested in
`tests/test-marking-policy.cpp`.

**What counts as a clone.** Both this gate and the speaker-consent gate
(`--i-have-rights` / `consent_attestation`) hang off one predicate, in
`examples/cli/crispasr_voice_clone_policy.h`. A voice is a clone when:

1. it is a **`.wav` reference** handed straight to a cloning backend;
2. this run **baked it from a recording** — the TADA one-command flow bakes
   `ref.wav` into a temp `.gguf` and rewrites `--voice` to it; or
3. the **pack declares** it (`crispasr.voice.cloned_from_recording`), which the
   bakers stamp into any pack they derive from a real recording.

That predicate used to be spelled *"the path ends in `.wav`"*, inline, in two
places. It was wrong in three ways, and each one silently produced an
unattested, undisclosed clone of a real person's voice: the rewrite in (2)
erased the evidence before the gate ran; **chatterbox clones only through a
baked `.gguf`** and has no `.wav` path at all, so a headline cloning backend
could never trip either gate (the same held for `--make-ref` output); and the
server accepted a bare name, so `{"voice": "victim"}` reached the same file as
`{"voice": "victim.wav"}` while scoring as "not a clone". A `.gguf` baked from
someone's recording is exactly as much a deepfake as the recording. The suffix
is an implementation detail.

4. its **`general.architecture`** names a producer that only ever bakes from a
   recording (`chatterbox-voice`, `qwen3tts.voicepack`, `cosyvoice3-voices`) —
   the legacy fallback for packs made before the stamp existed, which cannot be
   retro-stamped once published; or
5. it is an **entry in a multi-voice bank** that says so —
   `crispasr.voice.<name>.cloned_from_recording`.

**Case 5 is the one a file-shaped predicate cannot see.** cosyvoice3 keeps every
voice inside one `voices.gguf`, discovered as a sibling of the model (or
`CRISPASR_COSYVOICE3_VOICES_PATH`), and `--voice` selects an entry *by name*. So
`--voice fleurs-en` named no file, `resolve_voice_path()` had nothing to
resolve, no metadata was read, and a zero-shot voice clone scored as a preset —
on the CLI, the server, Wyoming and the ABI at once. `--voice victim.wav` on the
same backend *was* gated, which is exactly why this looked covered. The
backend's own source header calls the bundle "baked voice-clone bundles".

Only the backend knows which bundle it resolved, so it hands the path over:
`CrispasrBackend::voice_bank_path()`. **Any future backend that selects voices
by name from a container must override it**, or its clones ship unattested and
undisclosed. cosyvoice3 is the only one today (`rg '_n_voices\(|init_voices'`).

A bank is not all-or-nothing, hence the per-entry key: the default manifest
bakes upstream's `asset/zero_shot_prompt.wav` and a user's manifest adds their
own recordings, into the same file. A bank-wide flag would have to gate both or
free both. The `crispasr.voice.bank_stamped` sentinel is what lets an *absent*
per-voice key mean "preset" instead of "baked before the stamp existed" — only
bundles without it fall back to the producer architecture.

Note the deliberate difference from the tada-ref case: cosyvoice3 bundles gate
by architecture even though the built-in manifest is upstream's asset, because
CrispASR ships no cosyvoice3 bank and auto-downloads none. There is no shipped
preset to break — the operator bakes the bundle themselves, which is the moment
to ask for the attestation.

**Not every pack is a clone.** kokoro and vibevoice packs are converted from
upstream voicepacks and `.pt` prompts with no recording involved; gating those
behind a speaker-consent attestation nobody can meaningfully give would break
every documented example. So an unrecognised architecture stays a preset.

**But "not a clone" is not the same as "nothing to disclose"** — see §6.2a. That
sentence above is a true statement about *this project's conversion step* and
says nothing about whose voice the upstream voicepack was built from.

`crispasr.reference` (TADA) is deliberately **not** on the list: the shipped
`tada-ref-<lang>` packs and user `--make-ref` output share it, and gating the
shipped ones would break `-l de` auto-download. Those are covered by the stamp
going forward, so **a TADA reference baked before the stamp reads as a
preset** — re-bake it to gate it. Cases (1) and (2) never depend on the stamp.

Every producer that consumes a recording now gates and stamps at bake time:
`--make-ref`, `models/bake-chatterbox-voice-from-wav.py`,
`models/bake-qwen3-tts-voice-pack.py`,
`models/convert-tada-ref-to-gguf.py`,
`models/convert-cosyvoice3-voices-to-gguf.py` and
`models/convert-kugelaudio-voice-to-gguf.py` all require `--i-have-rights`.

The last two were found by the audit of 2026-08-02 and had **neither the gate
nor the stamp**, so their output read as a preset — the failure above, at the
other end. Enumerate them the way the surfaces are enumerated: not from this
list, but by asking which scripts read audio and write a GGUF —

```
rg -l 'GGUFWriter' models/ tools/ | xargs rg -l 'load_wav|torchaudio.load|--audio'
```

`convert-kugelaudio-voice-to-gguf.py` is the reason the stamp exists at all
rather than a producer allowlist: it has two modes, `--audio` (a clone) and
`--voice-pt` (an upstream preset), writing the same `general.architecture`. No
architecture-level rule can tell them apart.

**`POST /v1/voices` is a producer too.** It stores an uploaded recording as a
reusable voiceprint in `--voice-dir`, which is `--make-ref` over HTTP, and it
shipped with no attestation and no audit line — anyone who could reach the
endpoint could enroll a third party's voice, leaving only a byte count in the
log. It now requires a `consent_attestation` form field and emits
`[CONSENT] scope=voice-upload`. Synthesis was always gated (a bare name resolves
to `<voice-dir>/<name>.wav` and scores as a recording reference), so this was
never unmarked output; what was missing was consent asked at the point the
recording entered the system, which is the project's own stated rule.

Baking is itself the cloning step. `--make-ref` sat *before* the TTS block's
consent gate and returned early, which made it the one way to build a reusable
voiceprint with no attestation demanded anywhere; the two Python voice-pack
bakers had no gate at all.

Note the deliberate `#312` design: an unattested opt-out is **denied, not
refused**. You still get your audio — with the default disclaimer — plus
`X-Crispasr-*` response headers and an audit-log line saying so. Serving the
stronger default can never emit weaker-than-default output, while a hard 400
would only break clients one field out of date.

**On the C ABI and WASM, Art. 50(4) is yours.** Those paths watermark
(Art. 50(2) ✅) but do **not** prepend the spoken disclaimer. That is a
deliberate limit, not an oversight: the CLI produces a *neutral-voice*
disclaimer by clearing `tts_voice` per call, and several backends need
adapter-specific handling to honour it. On the ABI the voice has already been
applied to the backend context, and there is no portable way to un-apply it —
synthesizing anyway would risk speaking the disclosure **in the cloned voice**,
which makes the fake more convincing rather than less.

The ABI gives you the pieces instead:

| Call | Use |
|---|---|
| `crispasr_session_disclaimer_text()` | The canonical string, identical to the one the CLI speaks. Render it as a **visible** label — Art. 50(5) requires disclosures to meet accessibility requirements, and audio-only is not accessible to a deaf user. |
| `crispasr_session_get_disclaimer_pcm()` | The disclosure synthesized in the neutral voice, for you to prepend. **Must be called before `set_voice()`** installs a clone; it returns `NULL` afterwards rather than risk the cloned-voice failure above. |

Supported order: open session → `get_disclaimer_pcm()` → `set_voice()` →
`synthesize()` → prepend.

Synthesizing with a clone voice and no attestation logs a one-time `[MARKING]`
line naming the duty. It does not refuse —
`crispasr_session_accept_marking_responsibility()` silences it.

**Cloning consent is not gated on the ABI.** The CLI (`--i-have-rights`) and the
server (`consent_attestation`) both hard-refuse without it; the ABI logs a
`[CONSENT]` line and proceeds. Three reasons: consent to clone a voice is a
personality-rights and GDPR matter rather than an AI Act duty; the caller is an
integrator who has read the header; and #312 is the standing lesson on what a
hard refusal does to a surface with many downstream clients. The audit trail is
the deliverable there, not the block.

### 6.2a Whose voice is a *preset* voice? (`speaker_identity`)

Everything in §6.2 answers "did a recording pass through one of our bakers".
That is the right question for the **consent** gate and the wrong one for
Art. 50(4). A deep fake under Art. 3(60) is AI-generated audio "resembling
existing persons … which would falsely appear to a person to be authentic", and
nothing in that turns on our pipeline. A preset voice shipped inside a model can
be an identifiable individual — a named donor, or a corpus speaker such as
VCTK's `p225` — and synthesizing with it produces exactly that content.

So `is_clone == false` was doing two jobs: correctly meaning "no consent
attestation owed by this operator", and incorrectly meaning "nothing to
disclose".

**Consent and disclosure are different duties with different holders.** A
`real_person` preset is **disclosed** and is **not** consent-gated. Whether the
donor agreed to their recordings being used to build the model is a licensing
question settled upstream, between them and whoever trained it; the operator
downstream cannot attest to it, and demanding `--i-have-rights` for a stock
voice would be theatre that breaks every documented example. What the operator
owes is the audience knowing the audio is synthetic. Cloning is where both
apply, because there the operator *is* the one taking a specific person's voice.

| | consent gate | audible disclosure |
|---|---|---|
| clone | ✅ | ✅ |
| `real_person` preset | — | ✅ |
| `synthetic` preset | — | — |
| `unknown` preset | — | — + warns once |

**Three values, and `unknown` is deliberately one of them.** Collapsing it into
`synthetic` would silently assert the answer that happens to require no work, on
exactly the models nobody has checked. Collapsing it into `real_person` would
prepend a spoken sentence to every stock TTS voice in the project and train
operators to reach for `--no-spoken-disclaimer`, which is worse than the
disease. So `unknown` warns once per model, names what to do about it, and does
not force a disclosure — a question handed to the deployer, not a verdict.

Resolution is most-specific-first: `--speaker-identity` /
`"speaker_identity"` / `crispasr_session_set_speaker_identity()`, then the
pack's or bank entry's `crispasr.voice.speaker_identity`, then the backend's
`declared_speaker_identity()`, then `unknown`. The override moves the answer in
**both** directions on purpose — someone who knows a `synthetic` label is wrong
has to be able to say so, or the flag is only usable for adding duties and gets
ignored for the other half.

**The verdicts so far.** The sibling project CrispTTS ran the model-card
exercise over its own 27 entries (13 `real_person` / 7 `synthetic` /
7 `unknown`). Its findings were ported here, model by model, in
`examples/cli/crispasr_speaker_identity_models.h` — one reviewable table with
the evidence beside each answer, rather than 50 adapter overrides:

Every TTS backend CrispASR ships has now been checked against its provider's own
documentation. **The backlog is empty**; what remains at `unknown` is unknown
from evidence of absence, not from not having looked.

| backend / checkpoint | verdict | why |
|---|---|---|
| `piper` (all voices) | **real_person** | each voice is one named donor: the Lessac corpus, Thorsten Müller, Kerstin |
| `fastpitch` | **real_person** | `nvidia/tts_en_fastpitch`: *"trained on LJSpeech"* — 13,100 clips of one LibriVox narrator, Linda Johnson |
| `bananamind-tts` | **real_person** | card: en-us on LJSpeech (Linda Johnson), de-de on ThorstenVoice, credited *"Voice: Thorsten Müller"* |
| `parler-tts` | **real_person** | LibriTTS-R + MLS (LibriVox-derived), *"trained on 34 speakers, characterized by name"* — pseudonymous corpus readers, the VCTK `p225` case |
| `orpheus` + `kartoffel-orpheus-de-natural` | **real_person** | *"primarily on natural human speech recordings"*; its 19 speakers were *extracted* from podcasts/lectures/OER |
| `orpheus` + `kartoffel-…-de-synthetic` | synthetic | a different checkpoint: *"trained on synthetic German speech"* |
| `csm` | synthetic | *"a base generation model … has not been fine-tuned on any specific voice"* |
| `orpheus` base (Canopy) | unknown | 100k+ h of "permissive" audio disclosed, nothing about tara/leah/jess/… |
| `orpheus` + `lex-au` German FT | unknown | no training-data documentation |
| `bark` | unknown | Suno's README documents 100+ presets and says nothing about provenance |
| `melotts` | unknown | no training-data statement on the card |
| `speecht5` | unknown *structurally* | the voice is a 512-d x-vector **the operator supplies**; answer per run |
| `kokoro` (any checkpoint) | — | a backbone, not a voice. See the pack table below |

**Voice packs decide the answer for kokoro.** Its own card is explicit: *"This is
a base model, not a voice. It pairs with a German voice pack."*

| voice pack | verdict | why |
|---|---|---|
| `df_eva`, `dm_bernd` | **real_person** | per-speaker packs from HUI-Audio-Corpus-German, carrying the narrators' own names; HUI is built from librivox.org recordings |
| `df_victoria`, `dm_martin` | synthetic | kikiri fine-tunes over `kikiri-german-base-51speakers-synthetic`: *"Trained entirely on synthetic (TTS-generated) audio"* |
| `af_heart`, `ef_dora`, `ff_siwis` | synthetic | hexgrad's shipped style vectors, designed rather than any one person |

That resolves the **kokoro conflict**: CrispTTS marks kokoro `synthetic`, which
is right for the English packs and wrong for `df_eva` / `dm_bernd`. Moving the
answer from the checkpoint to the pack is what made both true at once. It has a
practical edge — the documented German cascade is
`df_victoria → df_eva → ff_siwis`, so a missing default silently moves a user
from a synthetic voice to a real HUI narrator, and the disclosure has to follow
the pack rather than the run.

**Two of CrispTTS's verdicts still do not port**, because they are different
weights reached through a different handler — checking that was the point.
`fastpitch` (CrispASR ships NVIDIA's English LJSpeech model, not the German NeMo
one) and `speecht5` (CrispASR ships the base model, whose speaker the operator
supplies) were each re-researched from scratch. FastPitch landed on
`real_person` anyway, by a different route.

Worth noting for anyone tempted to shortcut this: of every model whose
provenance has now been resolved across both projects, **the ones that resolved
away from `unknown` went overwhelmingly to `real_person`.** Two donors — Linda
Johnson (LJSpeech) and Thorsten Müller — each turn up in *two* different
CrispASR backends by independent routes. Guessing "synthetic" would have been
wrong nearly every time.

One near-miss worth recording: a web summary described Bark's presets as "fully
synthetic", and that phrasing is **not** in Suno's own README. It stayed
`unknown`. A third-party paraphrase is not a provider statement, and this is the
direction where being wrong removes a disclosure.

**Two sources, and how they combine.**

1. **The stamp** — `crispasr.voice.speaker_identity` in the checkpoint's or
   pack's own GGUF metadata, read by
   `crispasr_voice::read_model_speaker_identity()` /
   `read_pack_provenance()`. A stamped file answers for itself and survives
   being renamed, re-quantised or moved. Written either at conversion time
   (`models/convert-*.py --speaker-identity`) or, for anything already
   published, by **`models/stamp-speaker-identity.py`**, which rewrites the KV
   block of an existing GGUF and copies the tensors through untouched:

   ```bash
   python models/stamp-speaker-identity.py \
       --input kokoro-voice-df_eva.gguf --output kokoro-voice-df_eva-stamped.gguf \
       --speaker-identity real_person \
       --evidence "HUI-Audio-Corpus-German narrator 'eva'; HUI is librivox-derived"
   ```

   That tool exists because re-converting a 3.5 GB checkpoint to add one string
   is a price that means it never gets done, so the answer would live in the
   file-name table for good. `unknown` is deliberately not writable: absence of
   the key *is* unknown, and writing it would turn "nobody established this"
   into a claim the file makes about itself.

   To stamp a whole directory of already-published files:

   ```bash
   ./models/stamp-published-voices.sh /path/to/gguf-dir
   ```

   It asks `crispasr --print-speaker-identity` for each file — the same
   resolution the disclosure gate runs — and **skips anything that resolves to
   `unknown`** rather than guessing. It never touches the network; re-uploading
   is a separate, deliberate step.

   `crispasr --print-speaker-identity FILE` is also the answer to "will this
   disclose?": it prints `real_person` / `synthetic` / `unknown` and exits 3 on
   unknown.

   Seven converters accept `--speaker-identity` (`convert-orpheus`,
   `convert-kokoro-voice`, `convert-piper`, `convert-fastpitch`,
   `convert-bananamind-tts`, `convert-parler`, `convert-csm`). They share one
   definition of the flag and the key — `models/_speaker_identity_arg.py` —
   because seven hand-written copies would be seven chances for one to drift
   from `crispasr_voice::speaker_identity_key()`, and a drift **fails open**:
   the stamp is simply never found and nothing errors.
2. **The table** — the legacy fallback for everything published before the
   stamp existed, matching on the checkpoint file name.

The file-name matching is the thing rule 3 says not to do. It is used because
one backend serves many models with different answers and there is no other
metadata separating them (`convert-orpheus-to-gguf.py` writes
`general.name = "orpheus-<variant>"` for all of them), and because the failure
is **safe**: a renamed file matches nothing, falls to `unknown`, and warns. A
rename cannot turn `real_person` into `synthetic`; it can only turn a known
answer back into a question.

**Declared sources combine by strongest duty, not by precedence** — the one
non-obvious rule. The pack, the model stamp and the table are independent claims
about the same fact, and any of the three can be the only one that has heard of
a given voice. Precedence would let a stale stamp saying `synthetic` silently
cancel a researched `real_person` verdict, which is exactly the failure that
costs a disclosure. Taking the strongest means a stamp can only ever *upgrade* a
weaker answer, and disagreement fails toward disclosing. The escape hatch for a
genuinely wrong strong claim is the operator override, which is absolute in both
directions and requires a human to type it.

The stamp is guarded end-to-end by `tests/test-speaker-identity-gguf.cpp`, which
writes a real GGUF and reads it back through the same function the CLI and
server call — because a read path nothing consults is the inert-fix failure mode
(#324), and a pure test cannot see it.

**Guessing `synthetic` to quiet the warning is the costly error**, because it is
silent and it is wrong in the direction that removes a disclosure. Note that of
CrispTTS's 13 resolved-away-from-unknown models, *all* went to `real_person` —
the convenient guess would have been wrong every time.

Enforcement: `examples/cli/crispasr_speaker_identity.h` (mechanism) +
`crispasr_speaker_identity_models.h` (verdicts) +
`tests/test-speaker-identity.cpp`, which pins each verdict so flipping one is a
failing test somebody has to justify; wiring guarded by
`tests/test-compliance-wiring.cpp`.

### 6.3 Art. 50(1) — disclosure of AI interaction

**Deployer duty.** If your system interacts directly with natural persons, tell
them they're talking to an AI, unless it's obvious to a reasonably well-informed
person. A CLI transcription tool is obvious. A voice agent built on CrispASR's
S2S backends is not — that one is on you.

### 6.4 Art. 50(3) — emotion recognition / biometric categorisation

Not applicable: CrispASR implements neither (§4.1, §4.3). Had the emotion field
been kept, every deployer would have inherited a duty to inform exposed persons
of its operation, on every recording.

### 6.5 Art. 50(5) — form of disclosure

All disclosures must be given at or before first interaction, be clear and
distinguishable, and conform to applicable accessibility requirements. The spoken
disclaimer is audible; if your product has a visual surface, mirror it there —
audio-only disclosure is not accessible to a deaf user.

### 6.6 Synthetic *text* — the one Art. 50(2) surface CrispASR does not mark

Art. 50(2) covers systems generating synthetic "audio, image, video **or text**".
Everything above is about audio, because that is what this project is for. Two
surfaces generate text, and the distinction between them matters:

| Surface | Reading |
|---|---|
| ASR transcription | Not synthetic. A record of what a human actually said. |
| Translation (`/v1/translate`, `--translate`) | Generated, but semantics-preserving; read here as Art. 50(2) standard assistive processing (§6.1). |
| Punctuation restoration, truecasing, ITN | Assistive editing of an existing transcript. |
| **All four chat surfaces below** | **Open-ended generation. Not assistive editing, and not exempt.** |

There are **four** of them, not one. This section named only the HTTP endpoint
until the audit of 2026-08-02, which is the §6.1 table failure repeated in a
different section — the surfaces were enumerated from prose, and the prose was a
summary of the previous audit:

| Surface | Art. 50(2) marking | Art. 50(1) disclosure |
|---|---|---|
| `POST /v1/chat/completions` (`--chat-model`) | `X-Crispasr-Ai-Generated: true` + `X-Crispasr-Ai-Disclosure` response headers, on both the buffered and SSE branches | header carries the text; showing it is the client's job |
| `crispasr-chat` (installed binary, interactive REPL + one-shot) | not marked — plain text on stdout | prints the disclosure to stderr at startup, both modes |
| `crispasr_chat_*` C ABI (`include/crispasr_chat.h`) | **yours** | `crispasr_chat_ai_disclosure_text()` |
| `CrispasrChatSession` (Flutter, `chat.dart`) | **yours** | `CrispasrChatSession.aiDisclosureText()` |

The chat capability is opt-in everywhere — the endpoint exists only with
`--chat-model`, the binary only if you run it — and it serves whatever
general-purpose LLM GGUF the operator points it at.

**Marking is still weak here and this document will not pretend otherwise.**
There is no watermark-equivalent for short-form text that survives a
copy-paste; the Commission's guidance and the Code of Practice on AI-generated
content both treat machine-readable text marking as less settled than the audio
case, and point at metadata travelling with the content rather than a signal
inside the words. Response headers are that metadata. A client that drops them
publishes unmarked text, and **marking what you then do with the text remains
your duty** — the headers make the default better, they do not discharge
Art. 50(2) for you. On the ABI and in Flutter, nothing marks at all.

The disclosure string is one canonical value in the C ABI so the four surfaces
cannot drift apart. Render it **visibly**: Art. 50(5) requires disclosures to
meet accessibility requirements.

The Flutter binding is why Art. 50(1) is not theoretical here. §6.3's answer —
"a CLI transcription tool is obvious to a reasonably well-informed person" —
does not carry to a chat bubble in a mobile app, and it is exactly there that
nothing was said. `crispasr-chat` discloses anyway, despite a terminal launched
with `-m model.gguf` being about as obvious as it gets: it ships as the
reference for downstream wrappers, and a reference that omits the disclosure
teaches every wrapper to omit it.

Shipping a text-generating endpoint does not make this project a GPAI provider —
the model is the operator's choice and its provider's responsibility (§7).

### 6.6a The consent record — what it binds to, and whose artefact it is

A clone writes a `[CONSENT]` audit record on every surface. Two things about it
are worth stating plainly, because both are easy to assume wrongly.

**It binds to the audio, not to a name.** The record carries
`ref_sha256=<hex>` — the SHA-256 of the file the backend actually opened, after
`--voice-dir` resolution, so a bare `--voice alice` records the same evidence a
full path does. Before this the record said `voice=alice.wav` and nothing more,
which is an assertion rather than evidence: that file can be swapped a minute
later and the line still reads true. Where no file is involved (a voice-bank
entry selected by name, a preset), the field reads `ref_sha256=none` — an honest
"not applicable", never a zero hash that would look like a real one.

A `run_id` ties the record to the `[CONSENT-OUTPUT]` line emitted after
synthesis, which carries the output path and `out_sha256` of the file as
written — watermarked and C2PA-signed, the artefact that actually leaves the
machine. Verified end to end on a chatterbox clone (`clone_reason=pack-architecture`,
a legacy pack with no provenance stamp, classified by architecture):

    [CONSENT]        ... ref_sha256=85bbf5ec…  ref_is=resolved-voice  run_id=5baf6d93d54903fc
    [CONSENT-OUTPUT] ... out_sha256=4cbeb3ce…  seconds=6.30           run_id=5baf6d93d54903fc

Both hashes reproduce an independent `shasum -a 256`, and the shared `run_id`
is what joins them. On the server a per-request `req` id does the same job across
concurrent requests, and is returned to the client as `X-Crispasr-Request-Id`.
So a disputed clip can be walked back to the attestation that authorised it.

**It is the operator's artefact, not ours.** CrispASR is a tool; the operator is
the controller. `--consent-log <path>` (or `CRISPASR_CONSENT_LOG`) appends every
record as JSON Lines in addition to stderr, and is **off by default** — turning
on a persistent record of who attested what is the operator's decision, and it
is theirs to retain and erase.

The reason the separate sink exists at all: stderr here is interleaved with
model-load noise, download progress bars and warnings, which makes it a poor
evidential artefact however carefully each line is written.

**What this does not claim.** It is not tamper-proofing, and the code says so.
The attestation is made *by* the operator, who controls the process, its output
and any file it writes — no in-process mechanism defends against the party it is
recording. Real tamper-resistance is a storage decision: append-only
permissions, object-lock/WORM, or shipping the sink off-box to something the
operator cannot rewrite. The sibling projects hash-chain their consent logs;
CrispASR deliberately does not, because chaining protects the *sequence* of
records and the first thing to fix was that each record was unbound to any
audio. Chaining unbound assertions yields a perfectly verifiable log that proves
nothing. See `PLAN.md` for the full reasoning.

**Data minimisation.** The record carries a hash of the reference, never the
recording, and never a speaker name. Every field stored is a field that must be
erasable on request, and a hash is far easier to justify retaining than the
audio it was taken from.

### 6.7 Reading `--detect-watermark` (it is a diagnostic, not proof)

The built-in detector answers two questions about the spread-spectrum comb, and
a mark has to answer both:

* **Consistency** — is the comb's excess over its spectral neighbours steady
  across frames? A one-sample *t* over the per-frame readings.
* **Specificity** — is that specific to *our* pattern, or would any pattern
  score as well on this audio? The same statistic for 15 decoy sign patterns
  over the same bins, from keys never embedded with, standardised by their
  median and MAD.

Neither alone works. On a stationary tone a raw *t* of 11.4 means nothing
because every decoy scores just as extremely — only the specificity check
separates them. But specificity alone rejects real marks at 44.1 kHz, where the
comb sits in a low-energy region and the decoy spread grows.

| Verdict | Condition |
|---|---|
| `AI-GENERATED WATERMARK DETECTED` | confidence > 0.65 (both bars met) |
| `INCONCLUSIVE` | above chance, below the bar |
| `No watermark detected` | consistent with unwatermarked audio |

#### What this replaced, and why

Until 2026-08-03 the detector was a **sign-agreement test**: it averaged the
spectrum over all frames, compared 32 bins against their neighbours, and scored
each `+1`/`-1` by the *sign* of the difference, discarding its size. Under the
null that is a coin flip per bin, so the score averaged 0.5 (not 0) with a
standard deviation of `sqrt(32)/(2*32)` = 0.088 — leaving the 0.65 threshold
just 1.7 sigma above chance.

We answered that honestly, by reporting an exact binomial p-value and asserting
detection only at p < 0.01. But a p-value can only **trade** the two error
rates, never remove them, and the trade was brutal: at p < 0.01 the
true-positive rate on 1 s clips fell to 18%.

CrispTTS hit the same defect independently and replaced the *statistic* instead.
Two things make the new form better: it keeps the **magnitude** of each
difference rather than only its sign, and its sample count is the number of
**frames** — hundreds to thousands — rather than 32.

Measured here on 1265 one-second clips of genuinely unmarked human speech
(VoxConverse dev + JFK, native 16 kHz), scored against the same clips after the
**unchanged** embedder marked them (`tools/watermark_detect_ab.cpp`):

| clip length | sign FP / TP | per-frame FP / TP |
|---|---|---|
| 1.0 s | 5.2% / 68.6% | **0.9% / 96.8%** |
| 2.5 s | 5.1% / 79.8% | **1.2% / 99.6%** |
| 5.0 s | 4.0% / 88.0% | **1.6% / 99.6%** |
| 10.0 s | 4.9% / 100.0% | **3.3% / 100.0%** |

Better on **both** error rates at every clip length, which is why it is the
default. Both columns are at the same 0.65 threshold; the sign test's 4-5% false
positives reproduce its documented 4.8%, which is what says the harness is
measuring the right thing.

A third condition — the real pattern must also out-score the single **strongest**
decoy, not merely the decoy median (`t_true > 0.70 * max|t_decoy|`) — guards the
case a median comparison cannot see: audio where our pattern scores high *and so
does every decoy*, which is what highly structured or tonal signals do. Upstream
measured it removing a stationary-tone false positive at no cost to true
positives. **On this corpus it is close to a wash**, because 1265 clips of real
speech contain no such signal: false positives improve slightly (1.4% -> 1.2% at
2.5 s, 2.0% -> 1.6% at 5 s) and true positives pay a fraction of a point
(100.0% -> 99.6%). It is kept for the failure mode it closes and for parity with
the sibling projects that share the comb, not on the strength of these numbers —
on directly-measured tonal signals here, neither setting false-positives at all,
though the term does widen the margin (a three-tone reads 0.38 -> 0.28 against a
0.65 bar) and marked tones still verify at 0.97-0.9995.

`CRISPASR_WATERMARK_DETECT=sign` restores the old statistic for A/B and for
re-reading a score the way an older release reported it. The p-value is printed
only on that path — the per-frame score is a calibrated confidence with no bin
count, so quoting a binomial tail over it would invent an `n` that was never
scored.

#### What is still true

* **The embed is unchanged.** Audio marked by any earlier CrispASR release —
  and by CrispTTS and Susurrus, which share the comb — still verifies. Changing
  the embed to suit a detector would break that, and it is a release-blocking
  property.
* **A negative result is not evidence the audio is human-made.** It is evidence
  that this comb was not found. Someone who never marked their audio, or who
  stripped the mark, produces the same reading.
* **`--watermark-model auto` (AudioSeal) is still the sensitive detector** when
  the answer matters. Its output is a probability on its own scale; neither the
  binomial bands nor the per-frame bands apply to it.
* **None of this affects *marking*.** Embedding is unconditional and the
  watertight floor (§6.1) does not consult the detector, so a detector error
  here is a reading error and nothing else. That is worth stating because it is
  not free: CrispTTS verifies after embedding and deletes the output when
  verification fails, which makes a false negative destroy a user's file. We
  deliberately do not couple the two — the detector's accuracy improving is a
  reason to trust the *diagnostic* more, not a reason to start gating delivery
  on it.

---

## 7. General-purpose AI models (Ch. V)

This project publishes quantized GGUF conversions of third-party models to
Hugging Face (`cstr/*-GGUF`). That does **not** make it a GPAI provider:

- Most of the models are narrow speech models — ASR, TTS, diarization — and do
  not display the "significant generality" Art. 3(63) requires.
- Quantization is a format conversion, not training. Under the Commission's GPAI
  guidelines, a downstream modifier becomes the provider of a modified model only
  when the modification uses more than a third of the original training compute.
  Quantization uses approximately none.

Some backends derive from omni-modal bases (Voxtral, Qwen3-Omni, LFM2-Audio)
which may themselves be GPAI models — but the obligations there sit with their
original providers, not with a downstream requantizer. Model cards in
`hf_readmes/` carry upstream attribution and licence terms.

---

## 8. Feature classification, at a glance

| Feature | Classification | Enforcement |
|---|---|---|
| ASR transcription (60 backends) | Minimal risk; Art. 50(2) assistive exemption | — |
| Text translation | Minimal risk; semantics-preserving | — |
| Language identification | Not biometric categorisation | — |
| Denoise / source separation | Art. 50(2) assistive exemption | — |
| TTS synthesis (69 engines) | **Art. 50(2)** — marked | watermark + C2PA, default-on, watertight floor on CLI, server *and* Wyoming |
| Wyoming TTS (`--wyoming-port`) | **Art. 50(2) + 50(4)** | watermark always forced; clones disclaimed, and refused without operator `--i-have-rights` |
| Voice cloning (`.wav` ref, inline bake, stamped pack, **or bank entry**) | **Art. 50(2) + 50(4)** | + spoken disclaimer + `--i-have-rights`; `test-voice-clone-policy` |
| **Preset voice that is a real person** | **Art. 50(2) + 50(4)** — a deep fake without being a clone | `speaker_identity=real_person` → spoken disclaimer, **no** consent gate; `test-speaker-identity` (§6.2a) |
| Preset voice, provenance unresearched | Art. 50(2) | `unknown` — warns once per model, names the fix; **not** treated as synthetic |
| Multi-voice **banks** (cosyvoice3 `voices.gguf`) | Every entry is a baked clone | `voice_bank_path()` on all 4 surfaces; per-entry stamp; `test-compliance-wiring` |
| Voice-pack baking (`--make-ref` + all 5 Python bakers) | The cloning step itself | `--i-have-rights`; stamps `crispasr.voice.cloned_from_recording` |
| Voice upload (`POST /v1/voices`) | Enrollment = the cloning step | `consent_attestation`; `[CONSENT] scope=voice-upload` |
| `--detect-watermark` | Diagnostic, **not** a gate | per-frame *t* + decoy specificity; "DETECTED" needs both bars, confidence > 0.65 (§6.7) |
| Speech restoration / upscaling / S2S | **Art. 50(2)** — marked | watermark via S2S path, same per-response floor |
| **LLM chat** (endpoint, `crispasr-chat`, C ABI, Flutter) | **Art. 50(2) synthetic text + 50(1) interaction** | response headers + `crispasr_chat_ai_disclosure_text()`; **text marking stays a weak, partly deployer duty (§6.6)** |
| Session-scoped diarization | Not biometric identification | embeddings discarded, no names |
| Named voiceprint profiles | Kept outside Annex III(1)(a) | `--speaker-db-consent`, closed roster, offline-only |
| Voice-based emotion inference | Art. 5(1)(f) / Annex III(1)(c) | **removed**; `test-no-emotion-recognition` |
| Emotion *conditioning* in TTS; audio-event tags | Not Art. 3(39) — no person, no inference | §4.2 |

---

## 9. Deployer checklist

Things CrispASR cannot do for you:

- [ ] **Art. 50(4)** — show or speak an AI-generated label for any synthetic voice you publish. Default-on at the CLI and server; **your job** on the C ABI, WASM and bindings, using `crispasr_session_disclaimer_text()` / `crispasr_session_get_disclaimer_pcm()` (§6.2).
- [ ] **Art. 50(1)** — disclose AI interaction in conversational products. All four chat surfaces are ones (§6.6). Use `crispasr_chat_ai_disclosure_text()` / `CrispasrChatSession.aiDisclosureText()` and render it **visibly**; the ABI and Flutter cannot show it for you.
- [ ] **Art. 50(2) for text** — the chat endpoint sends `X-Crispasr-Ai-Generated`, but a client that drops the header publishes unmarked text, and the C ABI and Flutter mark nothing. Marking what you publish is yours (§6.6).
- [ ] **Re-bake cosyvoice3 voice banks** — a bundle baked before `crispasr.voice.bank_stamped` gates every entry by producer architecture, which is conservative but blunt. Re-bake with the current script for per-entry accuracy (§6.2).
- [ ] **`POST /v1/voices` now requires `consent_attestation`** — a breaking API change. Clients that enroll voices need the extra form field.
- [ ] **Answer the `speaker_identity` question for the presets you ship** (§6.2a). `piper` and `kartoffel-orpheus-de-natural` now disclose by default; anything CrispASR has not researched warns once per model. If the preset voice you use is an identifiable person, pass `--speaker-identity real_person` (or `"speaker_identity"` / `crispasr_session_set_speaker_identity()`). Do not silence the warning with `synthetic` unless you have read the model card — every model CrispTTS resolved away from `unknown` turned out to be a real person.
- [ ] **These now carry a spoken AI disclosure where they previously did not**: `piper` (all voices), `fastpitch`, `bananamind-tts`, `parler-tts`, `orpheus` + `kartoffel-orpheus-de-natural`, and the kokoro packs `df_eva` / `dm_bernd`. If you post-process or measure that audio, strip it with `--no-spoken-disclaimer --accept-marking-responsibility` rather than letting the prefix skew your results.
- [ ] **Stamp anything you publish yourself** with `models/stamp-speaker-identity.py` (§6.2a) so the answer travels with the file instead of depending on its name.
- [ ] **Re-bake old TADA references** — `chatterbox-voice` and `qwen3tts.voicepack` legacy packs are caught by architecture, but a `crispasr.reference` pack baked before the stamp reads as a preset (§6.2).
- [ ] **Don't treat `--detect-watermark` as proof either way** — it is a weak diagnostic with a stated error rate, not evidence of provenance (§6.7).
- [ ] **Art. 4** — ensure the people operating the system have adequate AI literacy.
- [ ] **GDPR** — voice is personal data, and biometric data when used to identify. The named-voiceprint path is Art. 9 special-category: explicit consent, retention and deletion policy, transparency. Applies independently of the AI Act.
- [ ] Don't repurpose diarization or speaker matching for surveillance, law-enforcement identification, or scraped audio. Out of scope and unsupported.
- [ ] Don't reconstruct emotion inference from the raw model output. §4.1 removed it for a reason; `sensevoice_result.raw` is a parity-testing field, not a workaround.

Penalties for scale: Art. 5 breaches reach €35 M or 7 % of worldwide annual
turnover; most other breaches €15 M or 3 %.

---

## 10. Where the enforcement lives

For whoever maintains this next — the compliance behaviour is code, and code
rots:

| Concern | File |
|---|---|
| Spoken-disclaimer opt-out policy | `examples/cli/crispasr_marking_policy.h` (+ `tests/test-marking-policy.cpp`) |
| **What counts as a voice clone** | `examples/cli/crispasr_voice_clone_policy.h` (pure) + `crispasr_voice_provenance.h` (resolve + read the stamp) (+ `tests/test-voice-clone-policy.cpp`) |
| **Whose voice a PRESET voice is** | `examples/cli/crispasr_speaker_identity.h` (mechanism, pure) (+ `tests/test-speaker-identity.cpp`) |
| **Which model is whose voice** | `examples/cli/crispasr_speaker_identity_models.h` — the researched verdicts, with evidence and an OPEN QUESTIONS backlog. Pinned by `test-speaker-identity.cpp`, so flipping one is a failing test |
| **Are the gates actually wired up?** | `tests/test-compliance-wiring.cpp` — source-level, guards the *joins*: every surface's `classify_voice` call, every baker's gate + stamp, the upload gate, binding watermark strength, the chat disclosures |
| **Multi-voice banks** | `CrispasrBackend::voice_bank_path()` (`crispasr_backend.h`), overridden by `crispasr_backend_cosyvoice3.cpp`; read by `crispasr_voice::read_bank_provenance()`; `s->cosyvoice3_voices_path` on the ABI |
| Chat / synthetic-text disclosure | `crispasr_chat_ai_disclosure_text()` in `src/chat.cpp`; call sites in `crispasr_chat_main.cpp`, `crispasr_server.cpp` (`X-Crispasr-Ai-*`), `flutter/crispasr/lib/src/chat.dart` |
| Voice upload consent gate | `POST /v1/voices` in `crispasr_server.cpp` (`consent_attestation`, `[CONSENT] scope=voice-upload`) |
| Which containers carry a manifest | `crispasr_marking::container_marking_for_format()` in `crispasr_marking_policy.h` |
| Voice-pack clone provenance stamp | written by `tada_encoder_write_ref_gguf()` + all 5 `models/*` voice bakers; read by `crispasr_voice::read_pack_provenance()` |
| Legacy pack classification by producer | `crispasr_voice::architecture_is_recording_derived()` |
| Watermark score → p-value / verdict | `examples/cli/crispasr_watermark_stats.h` (+ `tests/test-voice-clone-policy.cpp`) |
| Watermark embed / detect | `src/core/crispasr_watermark.h`, `examples/cli/crispasr_watermark_dispatch.h` |
| Watertight CLI marking floor | `crispasr_enforce_cli_watermark_floor()` in `examples/cli/crispasr_run.cpp` |
| **Container-less surfaces (Wyoming)** | `crispasr_marking::decide_raw_surface()` in `crispasr_marking_policy.h` (+ `tests/test-marking-policy.cpp`); call site in `examples/cli/wyoming.cpp` (+ `tests/test-wyoming-marking.py`) |
| C2PA signing | `src/core/crispasr_c2pa.h`, `third_party/c2pa-audio` |
| ABI marking attestation | `crispasr_session_accept_marking_responsibility()` in `src/crispasr_c_api.cpp` |
| Voice-clone consent gate | `--i-have-rights` (`crispasr_run.cpp`); `consent_attestation` (`crispasr_server.cpp`); `[CONSENT]` audit line only on the ABI |
| ABI clone disclosure | `crispasr_session_{disclaimer_text,get_disclaimer_pcm}()` (+ `tests/test-abi-clone-disclosure.cpp`) |
| Speaker-DB consent gate | `src/speaker_db.cpp`, `crispasr_speaker_db_open/enroll2` |
| Emotion-recognition exclusion | `tests/test-no-emotion-recognition.cpp` |

Two more rules earned by the second audit of 2026-08-02, alongside the four
below:

6. **Guard the joins, not just the predicate.** Every failure in this document's
   history — Wyoming marking nothing, `--make-ref` asking nothing, chatterbox
   packs, cosyvoice3 banks — happened while `test-voice-clone-policy` and
   `test-marking-policy` were green, because none of them was a predicate bug.
   They were missing call sites, unstamped bakers and ungated endpoints. That is
   what `tests/test-compliance-wiring.cpp` is for, and why it is deliberately
   coarse: it cannot prove the gate is called *correctly*, only that the call is
   still there, which is the failure mode that actually recurs.

5. **A voice does not have to be a file.** The gate reads `--voice`. Any other
   route by which a voice reaches a backend — an entry in a bundle, an env var,
   a config file — is invisible to it until it is plumbed in, and returns
   "preset" in the meantime. cosyvoice3's `voices.gguf` is the found case;
   `CRISPASR_KOKORO_VOICE_GGUF` is the shape of the next one. When adding a
   backend, ask how a voice gets in, not just what `--voice` looks like.

A fourth rule earned by the first audit of 2026-08-02, alongside the three
below:

4. **Enumerate surfaces from the code, not from this document.** The Wyoming
   server marked nothing for four releases purely because §6.1's table did not
   list it, and every audit since had walked that table. A prose list of
   surfaces is a summary of the last audit, never evidence about this one —
   grep for the emitters (`->synthesize(`, `speech_to_speech(`, audio writers)
   and reconcile the result against the table. A new surface inherits no
   marking; it has to be wired up, and only a grep will tell you it exists.

A third rule earned by the audit of 2026-08-02, alongside the two below:

3. **Classify by provenance, not by filename.** Every one of that audit's
   findings was the same mistake — a compliance predicate spelled as a string
   suffix (`ends with ".wav"`, `response_format == "wav"`). Suffixes are
   rewritten, resolved, re-encoded and renamed by code that has no idea a gate
   depends on them. Ask what the thing *is*.

Two rules learned the hard way, both worth keeping:

1. **A gate that CI can't run is a gate that ships wrong.** The `#312` marking
   policy went four days broken because its only coverage was a live server test.
   Compliance logic belongs in weight-free, model-free headers with unit tests.
2. **Prove the gate can go red.** Every guard here was verified by re-introducing
   the thing it forbids and watching it fail. A green test that cannot fail is
   indistinguishable from no test.

---

## References

- [Regulation (EU) 2024/1689 — full text](https://artificialintelligenceact.eu/the-act/)
- [Commission Guidelines on transparency obligations (Art. 50)](https://digital-strategy.ec.europa.eu/en/library/guidelines-transparency-obligations-providers-and-deployers-ai-systems)
- [Code of Practice on Transparency of AI-generated Content](https://digital-strategy.ec.europa.eu/en/policies/code-practice-ai-generated-content)
- [Commission FAQ on Art. 50](https://digital-strategy.ec.europa.eu/en/faqs/transparency-obligations-under-article-50-ai-act)
- [`diarization-speakers.md`](diarization-speakers.md) — speaker labels, GDPR Art. 9, the RBI boundary
- [`tts.md`](tts.md) — synthesis, cloning, `--i-have-rights`
- [`server.md`](server.md) — `consent_attestation`, `marking_attestation`, `X-Crispasr-*` headers
