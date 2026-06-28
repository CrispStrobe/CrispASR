# Speaker labels in diarization — session-scoped clustering vs. named profiles

CrispASR's diarization assigns a label to every transcribed segment. There
are two very different ways to make those labels meaningful, with very
different privacy and legal profiles. **Read this before enabling the second
one.**

| | Session-scoped clustering | Named voiceprint profiles |
|---|---|---|
| Flag | `--diarize-speakers` | `--enroll-speaker` + `--speaker-db` + `--speaker-db-consent` |
| Output | `(speaker 0)`, `(speaker 1)`, … | a real name, e.g. `(Mustermann)` |
| Embeddings | computed per recording, then **discarded** | **persisted to disk** as `.spkr` files |
| Scope | one recording only | a standing database reused across recordings |
| Identifies a named person? | **No** | **Yes (1:N match)** |
| Privacy footprint | transient audio processing | **biometric special-category data (GDPR Art. 9)** |

The default and recommended path is **session-scoped clustering**. The named
path exists, is **off by default**, and should only be used deliberately and
with the obligations in the last section understood.

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

> **This is a biometric feature. It persists voiceprints linked to real names
> and performs one-to-many identification. Treat it accordingly.** It is
> **off by default** and not part of the recommended diarization path.

The flags `--enroll-speaker NAME` (save an embedding) and `--speaker-db DIR`
(match segments against the saved embeddings and substitute the name) let you
build a standing database of named voiceprints and auto-label recordings with
real names.

**Consent gate.** Because this stores and matches biometric data, both
enrollment and matching **refuse to run** unless you pass `--speaker-db-consent`,
which affirms you have a lawful basis (GDPR Art. 9) and explicit consent from
every enrolled person. Without the flag, `--enroll-speaker` errors out and
`--speaker-db` is ignored (with a one-time notice pointing you at
`--diarize-speakers`).

```bash
# Enroll a reference clip (writes <db>/alice.spkr — a stored voiceprint):
crispasr -f alice-sample.wav --enroll-speaker Alice --speaker-db ./voiceprints \
    --speaker-db-consent

# Later, auto-name a recording by matching against the database:
crispasr -m auto -f meeting.wav --diarize --speaker-db ./voiceprints \
    --speaker-db-consent -ojf
```

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

On the **EU AI Act**: matching a voice against a database of N enrolled
profiles is mechanically a **one-to-many comparison**, so it is **not** covered
by the Act's 1:1 *biometric verification* carve-out
([Annex III(1)(a)](https://artificialintelligenceact.eu/annex/3/)). Whether it
constitutes a high-risk *remote biometric identification* system turns on the
defined elements of **"remote"** — identification *at a distance* and *without
the person's active involvement* ([Recital 17](https://ai-act-service-desk.ec.europa.eu/en/ai-act/recital-17)).
A **closed roster of participants who enrolled themselves with consent and know
they are being identified** has a strong argument for falling outside that
regime; an **open-ended database used to identify people who did not consent**
drifts toward exactly the kind of system the Act restricts. This is a genuine
gray area — get legal review before deploying the named path commercially, and
keep the roster closed and consented.

To stay clearly on the safe side, prefer **section 1** and rename manually.

---

## Implementation notes

- Session clustering: `crispasr_remap_speakers_via_embeddings()` in
  `examples/cli/crispasr_diarize_cli.cpp` — per-recording embedding extraction
  + agglomerative clustering (`src/crispasr_speaker_cluster.cpp`). No
  persistence.
- Embedder adapters (pluggable): `src/crispasr_speaker_embedder.{h,cpp}`
  (TitaNet-Large 192-d default; IndexTTS-BigVGAN ECAPA-TDNN 512-d).
- Named profiles: `src/speaker_db.{h,cpp}` — the `.spkr` on-disk format and
  the 1:N cosine match. Enroll and identify must use the **same** embedder so
  dimensions match.

See [`cli.md`](cli.md#diarization) for the full diarization flag reference.
