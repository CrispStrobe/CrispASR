// Ruby-specific model-free fixture for remaining ops
#include "crispasr.h"
#include "crispasr_session.h"
#include "runtime-abi-layout.h"
#include <stdlib.h>
#include <string.h>

static int mode, live_aux;
static float pitch_buffer[9];
static float separate_buffer[4];
static int pitch_n = 0, separate_n = 0;
static int pcm_rate = 16000;
static int att_left = 0, att_right = 0;
static int grammar_strict = 0;
static int aligner_freed = 0;
static int progress_val = 42;
static int streamed_seg_count = 2, streamed_tok_count = 3;

void probe_aux_mode(int m) {
    mode = m;
}
int probe_aux_live(void) {
    return live_aux;
}

// Pitch
int crispasr_session_pitch(crispasr_session* s, const float* pcm, int n, float hop) {
    if (mode == 2)
        return -1;
    if (n != 3 || pcm[0] != 1)
        return -9;
    pitch_buffer[0] = 0;
    pitch_buffer[1] = 440;
    pitch_buffer[2] = 0.9;
    pitch_buffer[3] = 10;
    pitch_buffer[4] = 442;
    pitch_buffer[5] = 0.8;
    pitch_buffer[6] = 20;
    pitch_buffer[7] = 0;
    pitch_buffer[8] = 0.1;
    pitch_n = mode == 1 ? 0 : 3;
    return pitch_n ? 3 : 0;
}
int crispasr_session_pitch_n_frames(crispasr_session* s) {
    return pitch_n;
}
int crispasr_session_pitch_frame(crispasr_session* s, int idx, float* t, float* f, float* p) {
    if (idx < 0 || idx >= pitch_n)
        return -1;
    if (t)
        *t = pitch_buffer[idx * 3];
    if (f)
        *f = pitch_buffer[idx * 3 + 1];
    if (p)
        *p = pitch_buffer[idx * 3 + 2];
    return 0;
}
const float* crispasr_session_pitch_frames(crispasr_session* s, int* n) {
    *n = pitch_n;
    return pitch_n ? pitch_buffer : NULL;
}
int crispasr_session_pitch_sample_rate(crispasr_session* s) {
    return 16000;
}

// Separate
int crispasr_session_separate(crispasr_session* s, const float* pcm, int n) {
    if (mode == 2)
        return -1;
    if (n != 4)
        return -1;
    separate_buffer[0] = 0.1;
    separate_buffer[1] = 0.2;
    separate_buffer[2] = 0.3;
    separate_buffer[3] = 0.4;
    separate_n = mode == 1 ? 0 : 2;
    return separate_n;
}
int crispasr_session_separate_n_stems(crispasr_session* s) {
    return separate_n;
}
const char* crispasr_session_separate_stem_name(crispasr_session* s, int idx) {
    return idx == 0 ? "vocals" : "drums";
}
const float* crispasr_session_separate_stem(crispasr_session* s, int idx, int* out_n) {
    *out_n = 2;
    return separate_buffer;
}
int crispasr_session_separate_sample_rate(crispasr_session* s) {
    return 44100;
}
int crispasr_session_piano_n_notes(crispasr_session* s) {
    return 2;
}
int crispasr_session_chords_n_spans(crispasr_session* s) {
    return 2;
}
int crispasr_session_beats_n_events(crispasr_session* s) {
    return 2;
}
int crispasr_session_tab_n_frames(crispasr_session* s) {
    return 2;
}

// PCM rate / att / grammar
int crispasr_session_set_pcm_sample_rate(crispasr_session* s, int rate) {
    if (rate <= 0)
        return -1;
    pcm_rate = rate;
    return 0;
}
int crispasr_session_set_parakeet_att_context(crispasr_session* s, int l, int r) {
    att_left = l;
    att_right = r;
    return 0;
}
int crispasr_session_set_audio_ctx(crispasr_session* s, int audio_ctx) {
    (void)s;
    (void)audio_ctx;
    return 0;
}
int crispasr_session_set_hikari_refresh_stride(crispasr_session* s, int stride) {
    (void)s;
    (void)stride;
    return 0;
}
int crispasr_session_set_grammar_strict(crispasr_session* s, int strict) {
    grammar_strict = strict;
    return 0;
}
void crispasr_free_aligner_cache(void) {
    aligner_freed = 1;
}

// Progress / streamed
int crispasr_get_progress(void) {
    return progress_val;
}
void crispasr_reset_progress(void) {
    progress_val = -1;
}
int crispasr_get_streamed_segment_count(void) {
    return streamed_seg_count;
}
struct crispasr_session_result* crispasr_drain_streamed_segments(void) {
    if (streamed_seg_count == 0)
        return NULL;
    streamed_seg_count = 0;
    return (struct crispasr_session_result*)1;
}
void crispasr_reset_streamed_segments(void) {
    streamed_seg_count = 0;
}
int crispasr_get_streamed_token_count(void) {
    return streamed_tok_count;
}
void crispasr_reset_streamed_tokens(void) {
    streamed_tok_count = 0;
}


int crispasr_diarize_segments_turns_abi(const float* left, const float* right, int32_t n, int32_t stereo,
                                        crispasr_diarize_seg_abi* segs, int32_t count,
                                        const crispasr_diarize_opts_abi* opts, crispasr_diarize_turn_abi* turns,
                                        int32_t cap, int32_t* needed) {
    if (!opts || count != 1 || !segs || n != 3 || stereo != 0 || segs[0].t0_cs != 100 || segs[0].t1_cs != 200 ||
        opts->method != 4 || opts->n_threads != 7 || opts->slice_t0_cs != 125 || !opts->pyannote_model_path ||
        strcmp(opts->pyannote_model_path, "model") || !opts->foxnose_embedder_path ||
        strcmp(opts->foxnose_embedder_path, "embed"))
        return 1;
    if (mode == 2)
        return 1;
    segs[0].speaker = 1;
    *needed = 1;
    if (!turns)
        return 0;
    if (cap < 1)
        return 2;
    turns[0].t0_cs = 125;
    turns[0].t1_cs = 250;
    turns[0].speaker = 2;
    turns[0]._pad = 0;
    return 0;
}
int crispasr_diarize_segments_abi(const float* left, const float* right, int32_t n, int32_t stereo,
                                  crispasr_diarize_seg_abi* segs, int32_t count,
                                  const crispasr_diarize_opts_abi* opts) {
    return crispasr_diarize_segments_turns_abi(left, right, n, stereo, segs, count, opts, NULL, 0, &(int){0});
}
// Registry / cache
int crispasr_registry_lookup_abi(const char* b, char* fn, int32_t fc, char* url, int32_t uc, char* sz, int32_t sc) {
    if (!b || !*b)
        return -1;
    strncpy(fn, "model.gguf", fc - 1);
    fn[fc - 1] = 0;
    strncpy(url, "https://example/model.gguf", uc - 1);
    url[uc - 1] = 0;
    strncpy(sz, "100MB", sc - 1);
    sz[sc - 1] = 0;
    return 0;
}
int crispasr_registry_lookup_by_filename_abi(const char* f, char* fn, int32_t fc, char* u, int32_t uc, char* s,
                                             int32_t sc) {
    return crispasr_registry_lookup_abi(f, fn, fc, u, uc, s, sc);
}
int crispasr_registry_default_bundle_info_abi(const char* b, char* ob, int32_t oc, char* lic, int32_t lc,
                                              int32_t* req) {
    if (req)
        *req = 1;
    if (ob) {
        strncpy(ob, b ? b : "", oc - 1);
        ob[oc - 1] = 0;
    }
    if (lic) {
        strncpy(lic, "MIT", lc - 1);
        lic[lc - 1] = 0;
    }
    return 1;
}
int crispasr_registry_default_bundle_artifact_abi(const char* b, int32_t idx, int32_t* kind, char* fn, int32_t fc,
                                                  char* url, int32_t uc, char* sz, int32_t sc) {
    if (idx != 0)
        return 1;
    if (kind)
        *kind = 0;
    if (fn) {
        strncpy(fn, "artifact.gguf", fc - 1);
        fn[fc - 1] = 0;
    }
    if (url) {
        strncpy(url, "https://example/artifact.gguf", uc - 1);
        url[uc - 1] = 0;
    }
    if (sz) {
        strncpy(sz, "10MB", sc - 1);
        sz[sc - 1] = 0;
    }
    return 0;
}
int crispasr_cache_ensure_file_abi(const char* fn, const char* url, int32_t q, const char* ov, char* out, int32_t oc) {
    if (out) {
        strncpy(out, "/tmp/cache/model.gguf", oc - 1);
        out[oc - 1] = 0;
    }
    return 0;
}
int crispasr_cache_dir_abi(const char* ov, char* out, int32_t oc) {
    if (out) {
        strncpy(out, "/tmp/cache", oc - 1);
        out[oc - 1] = 0;
    }
    return 0;
}
int crispasr_detect_backends_from_gguf(const char* path, char* out, int cap) {
    const char* data = "foo\nbar\n";
    int n = strlen(data);
    if (cap < n + 1)
        return -5;
    strcpy(out, data);
    return n;
}
int crispasr_detect_backend_from_gguf(const char* path, char* out, int cap) {
    if (cap < 4)
        return -5;
    strcpy(out, "foo");
    return 3;
}

static int temperature_calls;
int probe_temperature_calls(void) {
    return temperature_calls;
}
int crispasr_session_set_temperature(crispasr_session* session, float temperature, uint64_t seed) {
    ++temperature_calls;
    return session && temperature == .5f && seed == 0 ? 0 : -1;
}
