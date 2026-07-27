# Speaker labels in diarization — session-scoped clustering vs. named profiles

CrispASR's diarization assigns a label to every transcribed segment. There
are two very different ways to make those labels meaningful, with very
different privacy and legal profiles. **Read this before enabling the second
one.**

| | Session-scoped clustering | Named voiceprint profiles |
|---|---|---|
| Flag | `--diarize-speakers` | `--enroll-speaker` + `--speaker-db` + `--expect-speakers` + `--speaker-db-consent` |
| Output | `(speaker 0)`, `(speaker 1)`, … | a real name, e.g. `(Mustermann)`, for **claimed** participants only |
| Embeddings | computed per recording, then **discarded** | **persisted to disk** as `.spkr` files |
| Scope | one recording only | a standing database reused across recordings |
| Identifies a named person? | **No** | **Confirms claimed participants** (closed roster; no open 1:N search) |
| Privacy footprint | transient audio processing | **biometric special-category data (GDPR Art. 9)** |

The default and recommended path is **session-scoped clustering**. The named
path exists, is **off by default**, and should only be used deliberately and
with the obligations in the last section understood.

> **Streaming?** Both columns above are **recorded-file (offline)** features.
> For live transcription, a backend that emits a structured per-segment speaker
> label (`moss-diarize`; `vibevoice` from v0.8.24; `granite` in speaker-aware
> `--diarize` mode) surfaces its own per-utterance `(Speaker N)` labels under
> `--stream` — but the cross-recording clustering and named-voiceprint paths
> here do not run in real time. See [`streaming.md`](streaming.md#speaker-diarization-while-streaming).

---

## 1. Session-scoped speaker clustering (recommended)

This makes diarization labels **stable within a single recording** — so the
same physical voice gets the same `(speaker N)` label from start to finish —
**without identifying anyone**. Speaker embeddings are extracted per recording
purely to cluster segments, then thrown away. Nothing is stored, no names are
attached, and there is no database.

```bash
# Easiest: opt-in convenience alias. Enables --diarize and session-scoped
# clustering with the default embedder (auto-downloads ~46 MB TitaNet once).
crispasr -m auto --backend cohere -f meeting.wav --diarize-speakers -ojf

# Best quality for >2 speakers / long meetings — combine with the pyannote
# segmenter so segments also split at speaker-turn boundaries:
crispasr -m auto --backend cohere -f meeting.wav \
    --diarize --diarize-method pyannote --sherpa-segment-model auto \
    --diarize-embedder auto -ojf
```

`--diarize-speakers` is shorthand for `--diarize --diarize-method pyannote
--diarize-embedder auto` (it only fills in fields you didn't set explicitly):
the pyannote segmenter gives proper speaker-turn boundaries and the embedder
clusters them into stable per-recording labels. Tune clustering with
`--diarize-cluster-threshold` (default `0.5`; higher = more distinct clusters)
and `--diarize-max-speakers` (default `8`). The first run auto-downloads the
pyannote segmentation GGUF (~6 MB) and the TitaNet embedder (~46 MB).

The output uses generic labels:

```
(speaker 0) Welcome everyone, let's get started.
(speaker 1) Thanks. I pulled the numbers for Q3...
(speaker 0) Great, walk us through them.
```

### Putting real names on it — do it yourself, downstream

If you want names in the final transcript, the privacy-clean approach is a
**manual** find-and-replace once you know who is who (e.g. you recognise that
`speaker 1` is the person who presented Q3 numbers):

```bash
sed -e 's/(speaker 0)/Schmidt:/g' -e 's/(speaker 1)/Mustermann:/g' \
    meeting.txt > meeting.named.txt
```

This keeps the *identification* step a human decision made per recording. The
tool never builds or consults a biometric database of people's names, so the
heavyweight obligations in section 2 don't apply.

### Why this path is privacy-clean

Session-scoped clustering is **not** a biometric *identification* system:

- **No enrollment** and **no stored voiceprints** — embeddings live in memory
  for the duration of one run and are discarded.
- **No names** and **no standing database** — labels are anonymous ordinals
  scoped to a single file.
- **No 1:N matching against a roster** of known individuals.

It is ordinary, transient audio processing whose only purpose is diarization
quality. It does not fall under the EU AI Act's *remote biometric
identification* regime (there is no identification system to classify), and it
carries the GDPR footprint of normal audio processing rather than the
special-category biometric apparatus that a named voiceprint database triggers.

---

## 2. Named voiceprint profiles (`--speaker-db`) — deliberate opt-in

> **This is a biometric feature. It persists voiceprints linked to real names.
> Treat it accordingly.** It is **off by default**, not part of the recommended
> diarization path, and deliberately restricted to a **closed-roster
> confirmation**: it can only put names on participants you claim are present
> and who consented at enrollment. There is **no** "who is this voice?" mode.

The flags `--enroll-speaker NAME` (save an embedding) and `--speaker-db DIR`
plus `--expect-speakers "NameA,NameB"` (confirm which claimed participant each
diarized speaker cluster is) let you auto-label recordings of a known,
consenting group — e.g. recurring meeting minutes.

**How identification works (issue #266).** Anonymous diarization and named
identification share one pipeline:

```
ASR segments
  -> diarization (speaker-turn labels per slice)
  -> merge slices
  -> global speaker clustering (anonymous, stable (speaker N) labels)
  -> representative embedding per cluster (centroid; reuses the
     embeddings clustering already computed)
  -> OPTIONAL: match each cluster against the CLAIMED roster only
  -> matched cluster  -> (Alice)
     unmatched cluster -> (speaker N)      # stays anonymous
```

Each cluster is matched **independently**; a successful match renames only
that cluster, and nothing later in the pipeline can overwrite a matched name.
A slice containing several speakers can never be blanket-labeled with one
identity — names attach to global clusters, not to audio slices. Without
`--diarize`, the recording is treated as a single cluster (single-speaker
verification) and, on a match, all segments get the one name.

**Three hard gates**, all enforced before any matching:

1. `--speaker-db-consent` — affirms a lawful basis (GDPR Art. 9) and explicit
   consent from every enrolled person. Without it, `--enroll-speaker` errors
   out and `--speaker-db` is ignored (with a notice pointing you at
   `--diarize-speakers`). Enrollment records the attestation + timestamp in
   the `.spkr` profile (v2 format) as an audit trail.
2. `--expect-speakers "NameA,NameB"` — the **closed roster**. Matching runs
   only against these enrolled profiles; naming nobody is a hard error. An
   open-ended scan of the whole database is unsupported by design (see the
   legal section below). Claimed names with no enrolled profile are warned
   about and skipped.
3. **Recorded files only.** `--speaker-db` and `--enroll-speaker` refuse to
   run in streaming/live mode — real-time identification is deliberately
   unsupported.

```bash
# Enroll a reference clip (writes <db>/Alice.spkr — a stored voiceprint,
# with the consent attestation recorded in the file):
crispasr -f alice-sample.wav --enroll-speaker Alice --speaker-db ./voiceprints \
    --speaker-db-consent

# Later, label a meeting of Alice and Bob (both enrolled, both consented).
# --speaker-db with --diarize implies global clustering (--diarize-embedder
# auto), so this is the full shared pipeline:
crispasr -m auto -f meeting.wav --diarize-speakers \
    --speaker-db ./voiceprints --expect-speakers "Alice,Bob" \
    --speaker-db-consent -ojf
# -> (Alice) ... / (Bob) ... / (speaker 2) ... for any third voice
```

Tune the match with `--speaker-threshold` (default `0.7`; a cluster whose
centroid scores below it stays anonymous). Enroll and identify must use the
same embedder family (TitaNet 192-d by default) so dimensions match; a
mismatched `--diarize-embedder` (e.g. 512-d ECAPA) simply never matches.

### Caveats

- **Over-split clusters can both get the same name.** Global clustering is
  imperfect: if it splits one physical speaker's audio into two clusters
  (e.g. their voice drifts across a long recording), each cluster is matched
  independently against the roster, so **both** may match the same enrolled
  name. This is intentional, not a bug — both clusters genuinely are that
  speaker, and per-cluster matching has no way (or need) to notice they were
  once the same person.
- **Very short segments keep their local diarize label.** Segments shorter
  than about 0.25 s can't be embedded (below TitaNet's reliable floor), so
  they never take part in cluster matching and keep whatever local
  `(speaker N)` label the diarizer assigned. A transcript can therefore
  occasionally mix a matched `(Alice)` with a leftover `(speaker N)` for the
  same physical speaker, on short interjections ("mm-hmm", "yeah").

### Legal & privacy obligations (not legal advice)

Storing voiceprints to identify named people means you are processing
**biometric data for the purpose of uniquely identifying a natural person** —
**special-category data under [GDPR Article 9](https://iapp.org/news/a/biometrics-in-the-eu-navigating-the-gdpr-ai-act)**.
This holds regardless of how the EU AI Act classifies the system. In practice
that means, at minimum:

- **Explicit, freely-given, revocable consent** from every enrolled person
  (an Art. 9(2) basis), obtained before enrollment.
- A clear **retention and deletion** policy and an easy way to honour
  data-subject deletion requests (delete the relevant `.spkr` file).
- **Transparency**: tell people they are being identified.

On the **EU AI Act** (Regulation (EU) 2024/1689): *remote biometric
identification* — identifying people **without their active involvement**
against a reference database ([Art. 3(41)](https://artificialintelligenceact.eu/article/3/),
[Recital 17](https://ai-act-service-desk.ec.europa.eu/en/ai-act/recital-17)) —
is **high-risk** under [Annex III(1)(a)](https://artificialintelligenceact.eu/annex/3/),
with heavyweight provider and deployer obligations. CrispASR therefore does
not implement that kind of system, and the constraints above are what keep
this feature outside it:

- **Closed claimed roster only** (`--expect-speakers` is mandatory): the tool
  confirms which *asserted, actively-enrolled, consenting* participants speak
  in a recording. It cannot answer "who is this unknown voice?" — the
  open-ended 1:N search that characterizes an identification system is not
  implemented, at the CLI or at the C API.
- **Active involvement**: enrollment is a deliberate act by the enrolled
  person (consent + a provided sample), recorded in the profile.
- **Post-processing only**: no real-time/streaming identification path
  exists (cf. the Art. 5(1)(h) prohibition on real-time RBI in public
  spaces).

**Intended purpose** — and the only supported one — is cooperative labeling
of consenting, enrolled participants in recordings they know are being
transcribed (meeting minutes, interview archives, podcast production).
Identifying unknown persons, surveillance, law-enforcement use, or processing
publicly-scraped audio are **out of scope and unsupported**; do not attempt to
repurpose the feature for them. Get legal review before deploying the named
path commercially.

To stay clearly on the safe side, prefer **section 1** and rename manually.

---

## Implementation notes

- Session clustering: `crispasr_remap_speakers_via_embeddings()` in
  `examples/cli/crispasr_diarize_cli.cpp` — per-recording embedding extraction
  + agglomerative clustering (`src/crispasr_speaker_cluster.cpp`). No
  persistence.
- Embedder adapters (pluggable): `src/crispasr_speaker_embedder.{h,cpp}`
  (TitaNet-Large 192-d default; IndexTTS-BigVGAN ECAPA-TDNN 512-d).
- Named profiles: `src/speaker_db.{h,cpp}` — the `.spkr` on-disk format
  (v2 adds the consent attestation + enrollment timestamp) and cosine
  matching. `speaker_db_retain()` narrows a loaded db to the claimed roster
  before any match. Enroll and identify must use the **same** embedder so
  dimensions match.
- Cluster identification: `crispasr_identify_speaker_clusters()` /
  `crispasr_identify_single_speaker()` in
  `examples/cli/crispasr_diarize_cli.cpp`, driven from
  `crispasr_apply_global_speaker_stages()` in `examples/cli/crispasr_run.cpp`
  (one post-merge stage shared by the sequential and parallel output paths).
- C API: `crispasr_speaker_db_open(dir, expected_names_csv, consent_attested)`
  and `crispasr_speaker_db_enroll2(..., consent_attested)` are the only entry
  points; the pre-#266 ungated `_load`/`_enroll` symbols refuse at runtime.

See [`cli.md`](cli.md#diarization) for the full diarization flag reference.
