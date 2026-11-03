# glint (in-tree copy)

Our clean-room, dependency-free audio **codec** library in C++17, MIT
licensed. This is an in-tree copy of the sibling `glint` repository —
the codec core only (`src/` + `include/glint/glint.h`); the upstream
CLI, unit tests, language bindings and packaging live in the glint
repo and are not carried here. Same relationship as `ggml/`: develop
in the source repo, sync the core here.

It is no longer encode-only. The vendored core now covers encode **and
decode**, with no system package required:

| codec | encode | decode |
|---|---|---|
| MP3 (MPEG-1/2/2.5 Layer III) | ✅ | ✅ |
| AAC-LC (ADTS) | ✅ | ✅ |
| Opus (RFC 6716/7845 — SILK/CELT/hybrid decode; CELT-only 48 kHz encode) | ✅ | ✅ |
| Vorbis | — | ✅ |
| WAV I/O | ✅ | ✅ |

A one-call `glint_encode_audio()` picks MP3 / AAC-LC / Ogg-Opus and
auto-resamples to a codec-valid rate; `glint_decode_audio()` is the
matching decode entry point.

## How CrispASR uses it

**Compressed output (TTS / S2S).** The encoders back:

- `crispasr-server` — `POST /v1/audio/speech` with
  `response_format=mp3` / `aac` / `opus`
- `crispasr-cli` — `--tts-output out.{mp3,aac,opus}` and
  `--s2s-output`

routed through the header-only writers
`examples/cli/crispasr_{mp3,aac,opus}_writer.h` and, for M4A/MP4,
`crispasr_mp4_writer.h`. AI-provenance rides along per container: MP3
and AAC get a prepended ID3v2 tag (`crispasr_make_id3v2_ai_tag`), Opus
gets it in the RFC 7845 `OpusTags` comment header, and AAC/Opus muxed
into MP4/M4A carry a native C2PA manifest (raw ADTS and Ogg have no
C2PA path — see `docs/tts.md`).

**Compressed input (decode).** `src/crispasr_audio.cpp` uses the
in-tree decoders to read compressed audio into PCM with no external
dependency: `glint_decode_audio` / `glint_opus_decode` /
`glint_aac_decode`, including Opus/Vorbis extracted from WebM/Matroska
(e.g. browser `MediaRecorder` captures). For `.opus` files, libopus is
an optional *pinned* fallback, built only with `-DCRISPASR_OPUS=ON` and
selected at runtime with `CRISPASR_OPUS_DECODER=libopus`; the default
is glint's own RFC-conformant packet decoder.

MP3 output likewise has an optional fallback: when the build finds
libmp3lame it stays in, used automatically if glint fails or forced
with `CRISPASR_MP3_ENCODER=lame` for A/B comparisons.

## Syncing

Automated: the `sync-glint` GitHub workflow pulls the committed state
of `CrispStrobe/glint` main (on repository_dispatch from glint pushes,
a daily schedule, or manual dispatch), regenerates the CMake source
list and the sync marker below, runs the
`test_tts_provenance "[mp3],[aac]"` suite, and pushes to main. The
`validate-glint-fresh` release job fails a release whose in-tree copy
is behind glint main. To sync by hand, run `tools/sync-glint.sh` —
never `cp` from a glint working tree (it may hold another session's
WIP; the script always takes a committed state).

**Note:** the sync copies `src/`, `include/glint/glint.h`, `LICENSE`
and rewrites *only* the marker line below — the prose above is
maintained here by hand and does not auto-update, so a codec added
upstream will not be described until this file is edited.

Synced at upstream commit: `7b458cff0fc9c397f671c422db4770c72fc22f5c` (perf(mp3): decode ~17x faster — tables instead of libm in the inner loops).
