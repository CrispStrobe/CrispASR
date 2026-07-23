// tiron_link.h — cross-window speaker linking for Tiron (#295).
//
// Tiron (Whisper large-v3 + inline <|speakerN|> markers) attributes speakers
// with indices that are LOCAL to each 30 s decode window: the first talker in
// a window is always <|speaker1|>, so "speaker1" in window 0 and "speaker1" in
// window 2 may be different people. This module promotes those window-local
// indices to stable, meeting-level identities (SPEAKER_00, SPEAKER_01, …) by
// clustering voice embeddings — the same job the upstream Trelis/tiron harness
// does with ECAPA + agglomerative clustering.
//
// It REUSES the existing CrispASR speaker stack rather than reinventing it:
//   * CrispasrSpeakerEmbedder  (src/crispasr_speaker_embedder.h) — TitaNet /
//     ECAPA embedding of a PCM range, L2-normalized.
//   * crispasr_agglomerative_cluster / _cluster_centroids
//     (src/crispasr_speaker_cluster.h) — single-linkage cosine clustering.
//
// The one thing it adds over the generic per-segment remap
// (crispasr_remap_speakers_via_embeddings) is the WITHIN-WINDOW MUST-LINK: all
// turns sharing a (window_id, local_speaker) key are the same physical speaker
// by construction, so their audio is aggregated into ONE embedding. That both
// (a) yields a longer, cleaner voiceprint than a single short turn and (b) makes
// the clustering a cross-window linking problem (few groups) instead of a
// per-segment one (many noisy points) — upstream's "spine".

#pragma once

#include <cstdint>
#include <vector>

class CrispasrSpeakerEmbedder;

// One decoded turn to be linked. The linker never reads the transcript text —
// the caller keeps it and applies the returned global speaker id.
struct TironTurn {
    int64_t t0_cs = 0;     // absolute start, centiseconds
    int64_t t1_cs = 0;     // absolute end, centiseconds
    int window_id = 0;     // decode window this turn came from (crispasr_segment.chunk_id,
                           // or t0_cs / 3000 for an unchunked run). Local indices reset per window.
    int local_speaker = 0; // 1..8 from <|speakerN|>; <=0 means "no local hint" (each such
                           // turn is then treated as its own group / per-segment fallback).
};

struct TironLinkOptions {
    float merge_threshold = 0.5f; // cosine merge cutoff (TitaNet-Large rule of thumb, see cluster header)
    int max_speakers = 16;        // hard cap on the meeting-level roster
    int64_t min_embed_cs = 25;    // >=0.25 s of audio for a group to seed a cluster ("spine")
    int64_t max_embed_cs = 3000;  // cap aggregated audio per group at 30 s (embedder cost / drift)
    float attach_margin = 0.1f;   // a sub-min-duration group attaches to the nearest spine centroid
                                  // if cosine >= (merge_threshold - attach_margin); else temporal fallback.
};

struct TironLinkResult {
    std::vector<int> turn_speaker; // global speaker id in [0, n_speakers) per INPUT turn; -1 if unresolved
    int n_speakers = 0;
    std::vector<float> centroids; // n_speakers * dim, row-major, L2-normalized (empty if no embedder)
    int dim = 0;
};

// ── High-level transcript linking (library-hoisted so EVERY surface — CLI,
// session C-ABI, server — applies the same SPEAKER_NN linking, not just the CLI).
// One segment of a decoded Tiron transcript. `text` carries the inline
// <|speakerN|> markers on input; on output they are stripped and `speaker` /
// `drop` are filled.
struct TironTranscriptSeg {
    std::string text;  // in: may contain <|speakerN|>; out: markers stripped
    int64_t t0_cs = 0; // centiseconds
    int64_t t1_cs = 0;
    int chunk_id = -1;   // decode window; -1 => derived from t0_cs
    std::string speaker; // out: "SPEAKER_NN " when linked (else left unchanged)
    bool drop = false;   // out: true => a bare <|speakerN|> marker segment; caller should drop it
};

// Detect Tiron output and, when an embedder spec is given, promote the window-
// local <|speakerN|> markers to meeting-level SPEAKER_NN by voiceprint
// clustering. Mutates `segs` in place (strips markers, sets speaker/drop).
//
// Returns:
//   -1  not a Tiron transcript (no <|speakerN|> markers) — `segs` untouched.
//    0  Tiron, but no linking performed (empty/failed embedder spec) — the
//       local <|speakerN|> markers are LEFT in place for the caller to render.
//   >0  number of meeting-level speakers; markers stripped, labels applied.
//
// `embedder_spec` accepts crispasr_make_speaker_embedder's specs ("auto",
// "titanet", "ecapa", a .gguf path); empty/"" skips linking.
int crispasr_tiron_link_transcript(std::vector<TironTranscriptSeg>& segs, const float* pcm_16k, int n_samples,
                                   const char* embedder_spec, int n_threads, const char* cache_dir);

// Link Tiron's window-local speaker indices into meeting-level ids.
//
// `pcm_16k` is the full mono 16 kHz PCM the turns index into (turn times are
// absolute centiseconds → samples via *160). `embedder` may be null: with no
// embedder there is no acoustic signal to link across windows, so each distinct
// (window_id, local_speaker) becomes its own global id (no cross-window merge) —
// a safe, if conservative, degradation.
//
// Deterministic: global ids are assigned in first-appearance order of the
// grouping keys, so the output is stable across runs on the same input.
TironLinkResult crispasr_tiron_link_speakers(const std::vector<TironTurn>& turns, const float* pcm_16k, int n_samples,
                                             CrispasrSpeakerEmbedder* embedder, const TironLinkOptions& opts = {});
