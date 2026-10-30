#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

// Combined probe for Go/Rust wrapper tests
// Merges python-wrapper-abi.c and wrapper-session-metadata.c with extensions
#include "crispasr_session.h"
#include "crispasr.h"
#include "crispasr_chat.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

static int legacy_controls_expected, legacy_param_requests;
static int probe_mode = 0; // 0=normal, 1=empty, 2=error
static int probe_live = 0;
static int probe_calls = 0;
static int probe_invalid_frees = 0;
static int probe_result_frees = 0;
static int probe_streams, probe_stream_closes;
static int streamed_ready;
void probe_streamed_ready(int ready) {
    streamed_ready = ready;
}
int probe_stream_closes_get(void) {
    return probe_stream_closes;
}
static float probe_buffer[32];
static int probe_programs[2] = {24, 128};
static char probe_token_head[3] = {(char)0xe6, (char)0x97, 0};
static char probe_token_tail[2] = {(char)0xa5, 0};

static crispasr_progress_callback g_prog_cb = NULL;
static void* g_prog_ud = NULL;
static crispasr_segment_callback g_seg_cb = NULL;
static void* g_seg_ud = NULL;
static crispasr_token_callback g_tok_cb = NULL;
static void* g_tok_ud = NULL;
static void fire_callbacks(void);
static void fire_callbacks(void) {
    if (g_prog_cb)
        g_prog_cb(3, 6, g_prog_ud);
    if (g_seg_cb)
        g_seg_cb("\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e", 123, 456, 7, g_seg_ud);
    if (g_tok_cb)
        g_tok_cb("\xe6\x97", 9, g_tok_ud);
}
void crispasr_session_set_progress_callback(crispasr_session* s, crispasr_progress_callback cb, void* ud) {
    g_prog_cb = cb;
    g_prog_ud = ud;
}
void crispasr_session_set_segment_callback(crispasr_session* s, crispasr_segment_callback cb, void* ud) {
    g_seg_cb = cb;
    g_seg_ud = ud;
}
void crispasr_session_set_token_callback(crispasr_session* s, crispasr_token_callback cb, void* ud) {
    g_tok_cb = cb;
    g_tok_ud = ud;
}

void probe_mode_set(int m) {
    probe_mode = m;
}
int probe_live_get(void) {
    return probe_live;
}
int probe_calls_get(void) {
    return probe_calls;
}
int probe_invalid_frees_get(void) {
    return probe_invalid_frees;
}
int probe_result_frees_get(void) {
    return probe_result_frees;
}
void probe_reset(void) {
    probe_mode = 0;
    legacy_controls_expected = 0;
    legacy_param_requests = 0;
    probe_calls = 0;
    probe_result_frees = 0;
}
void probe_mutate(void) {
    probe_buffer[0] = 999;
}

static float* alloc_pcm(int n, float v0, float v1) {
    float* p = (float*)malloc(n * sizeof(float));
    if (!p)
        return NULL;
    probe_live++;
    if (n > 0)
        p[0] = v0;
    if (n > 1)
        p[1] = v1;
    return p;
}
static void* tracked_alloc(size_t n) {
    void* p = malloc(n);
    if (p)
        probe_live++;
    return p;
}
static char* alloc_text(const char* s) {
    size_t n = strlen(s) + 1;
    char* p = (char*)tracked_alloc(n);
    if (p)
        memcpy(p, s, n);
    return p;
}
static void tracked_free(void* p) {
    if (!p)
        return;
    probe_live--;
    free(p);
}
static int check_pcm(const float* pcm, int n) {
    probe_calls++;
    if (probe_mode == 2)
        return -1;
    if (n != 3 || pcm[0] != 1 || pcm[1] != 3 || pcm[2] != 5)
        return -9;
    for (int i = 0; i < 32; i++)
        probe_buffer[i] = (float)i;
    return probe_mode == 1 ? 0 : 2;
}

// --- Session open ---
#include "runtime-open-layout.h"
crispasr_session* crispasr_session_open(const char* p, int n) {
    return (crispasr_session*)0x100000001ULL;
}
crispasr_session* crispasr_session_open_explicit(const char* p, const char* b, int n) {
    return (crispasr_session*)0x100000001ULL;
}
crispasr_session* crispasr_session_open_with_params(const char* m, const char* b, const crispasr_open_params_v1* pr) {
    if (probe_mode == 2 || !pr || sizeof(*pr) != 48 || pr->abi_version != 2 || !m || strcmp(m, "model"))
        return NULL;
    if (probe_mode == 1) {
        if (b || pr->n_threads != 4 || pr->use_gpu != 1 || pr->verbosity != 0 || pr->flash_attn != 1 ||
            pr->n_gpu_layers != -1)
            return NULL;
    } else if (!b || strcmp(b, "probe") || pr->n_threads != 7 || pr->use_gpu != 0 || pr->verbosity != 2 ||
               pr->flash_attn != 0 || pr->n_gpu_layers != 3)
        return NULL;
    for (int i = 0; i < 6; ++i)
        if (pr->reserved[i])
            return NULL;
    return (crispasr_session*)0x100000001ULL;
}
const char* crispasr_session_backend(crispasr_session* s) {
    return "probe";
}
void crispasr_session_close(crispasr_session* s) {}
const char* crispasr_c_api_version(void) {
    return "probe";
}
const char* crispasr_dart_helpers_version(void) {
    return "probe-helpers";
}
void crispasr_set_gpu_backend(const char* name) {
    probe_calls++;
}
void crispasr_session_close_dummy(crispasr_session* s) {}

// --- Result handling ---
static const char* seg_text = "世";
static int64_t seg_t0 = 100, seg_t1 = 200;
static const char* seg_speaker = "(Speaker 1) ";
static const char* word_text = "世";
static int64_t word_t0 = -1, word_t1 = -1;
static float word_p = 0.9f;
static const char* token_texts[2] = {(char*)probe_token_head, (char*)probe_token_tail};
static int64_t token_t0s[2] = {-1, 123};
static int64_t token_t1s[2] = {-1, 124};
static float token_ps[2] = {0.5f, 0.75f};
static int* make_result(int has_data) {
    int* p = (int*)malloc(sizeof(int));
    if (p)
        *p = has_data;
    return p;
}
int crispasr_session_result_n_segments(crispasr_session_result* r) {
    if (probe_mode == 11)
        return -1;
    if (!r)
        return 0;
    int* p = (int*)r;
    return *p ? 1 : 0;
}
const char* crispasr_session_result_segment_text(crispasr_session_result* r, int i) {
    if (!r || i != 0)
        return "";
    int* p = (int*)r;
    if (!*p)
        return "";
    return seg_text;
}
int64_t crispasr_session_result_segment_t0(crispasr_session_result* r, int i) {
    if (!r)
        return 0;
    int* p = (int*)r;
    if (!*p)
        return 0;
    return seg_t0;
}
int64_t crispasr_session_result_segment_t1(crispasr_session_result* r, int i) {
    if (!r)
        return 0;
    int* p = (int*)r;
    if (!*p)
        return 0;
    return seg_t1;
}
const char* crispasr_session_result_segment_speaker(crispasr_session_result* r, int i) {
    if (!r)
        return "";
    int* p = (int*)r;
    if (!*p)
        return "";
    return seg_speaker;
}
float crispasr_session_result_segment_no_speech_prob(crispasr_session_result* r, int i) {
    return -1.0f;
}
int crispasr_session_result_n_words(crispasr_session_result* r, int i) {
    if (probe_mode == 12)
        return -1;
    if (!r)
        return 0;
    int* p = (int*)r;
    if (!*p)
        return 0;
    return 1;
}
const char* crispasr_session_result_word_text(crispasr_session_result* r, int i, int w) {
    if (!r)
        return "";
    int* p = (int*)r;
    if (!*p)
        return "";
    return word_text;
}
int64_t crispasr_session_result_word_t0(crispasr_session_result* r, int i, int w) {
    if (!r)
        return -1;
    int* p = (int*)r;
    if (!*p)
        return -1;
    return word_t0;
}
int64_t crispasr_session_result_word_t1(crispasr_session_result* r, int i, int w) {
    if (!r)
        return -1;
    int* p = (int*)r;
    if (!*p)
        return -1;
    return word_t1;
}
float crispasr_session_result_word_p(crispasr_session_result* r, int i, int w) {
    if (!r)
        return -1;
    int* p = (int*)r;
    if (!*p)
        return -1;
    return probe_mode == 10 ? -1.f : word_p;
}
int crispasr_session_result_word_n_alts(crispasr_session_result* r, int i, int w) {
    if (probe_mode == 13)
        return -1;
    if (!r)
        return 0;
    int* p = (int*)r;
    if (!*p)
        return 0;
    return 1;
}
const char* crispasr_session_result_word_alt_text(crispasr_session_result* r, int i, int w, int a) {
    return "alternative";
}
float crispasr_session_result_word_alt_p(crispasr_session_result* r, int i, int w, int a) {
    return 0.25f;
}
int crispasr_session_result_n_tokens(crispasr_session_result* r, int i) {
    if (probe_mode == 14)
        return -1;
    if (!r)
        return 0;
    int* p = (int*)r;
    if (!*p)
        return 0;
    return 2;
}
const char* crispasr_session_result_token_text(crispasr_session_result* r, int i, int t) {
    if (!r)
        return "";
    int* p = (int*)r;
    if (!*p)
        return "";
    if (t < 0 || t >= 2)
        return "";
    return t ? probe_token_tail : probe_token_head;
}
int64_t crispasr_session_result_token_t0(crispasr_session_result* r, int i, int t) {
    if (!r)
        return -1;
    int* p = (int*)r;
    if (!*p)
        return -1;
    if (t < 0 || t >= 2)
        return -1;
    return token_t0s[t];
}
int64_t crispasr_session_result_token_t1(crispasr_session_result* r, int i, int t) {
    if (!r)
        return -1;
    int* p = (int*)r;
    if (!*p)
        return -1;
    if (t < 0 || t >= 2)
        return -1;
    return token_t1s[t];
}
float crispasr_session_result_token_p(crispasr_session_result* r, int i, int t) {
    if (!r)
        return -1;
    int* p = (int*)r;
    if (!*p)
        return -1;
    if (t < 0 || t >= 2)
        return -1;
    return token_ps[t];
}
int crispasr_session_result_n_logit_frames(crispasr_session_result* r) {
    if (probe_mode == 15)
        return INT32_MAX;
    if (probe_mode == 16)
        return 1;
    return 0;
}
int crispasr_session_result_n_logit_vocab(crispasr_session_result* r) {
    if (probe_mode == 15)
        return INT32_MAX;
    if (probe_mode == 16)
        return 1;
    return 0;
}
const float* crispasr_session_result_logits(crispasr_session_result* r) {
    if (probe_mode == 15)
        return probe_buffer;
    return NULL;
}
void crispasr_session_result_free(crispasr_session_result* r) {
    if (!r)
        return;
    free(r);
    probe_result_frees++;
    probe_token_head[0] = 'X';
    probe_token_head[1] = 'Y';
}

// Transcribe variants
crispasr_session_result* crispasr_session_transcribe(crispasr_session* s, const float* pcm, int n) {
    if (probe_mode == 2)
        return NULL;
    if (probe_mode == 1)
        return (crispasr_session_result*)make_result(0);
    if (n != 3 || pcm[0] != 1 || pcm[1] != 3 || pcm[2] != 5)
        return NULL;
    probe_token_head[0] = (char)0xe6;
    probe_token_head[1] = (char)0x97;
    probe_token_head[2] = 0;
    probe_token_tail[0] = (char)0xa5;
    probe_token_tail[1] = 0;
    fire_callbacks();
    return (crispasr_session_result*)make_result(1);
}
crispasr_session_result* crispasr_session_transcribe_lang(crispasr_session* s, const float* pcm, int n,
                                                          const char* lang) {
    return crispasr_session_transcribe(s, pcm, n);
}
crispasr_session_result* crispasr_session_transcribe_chunked(crispasr_session* s, const float* pcm, int n, int cs,
                                                             int os) {
    return crispasr_session_transcribe(s, pcm, n);
}
crispasr_session_result* crispasr_session_transcribe_chunked_lang(crispasr_session* s, const float* pcm, int n, int cs,
                                                                  int os, const char* lang) {
    return crispasr_session_transcribe(s, pcm, n);
}
crispasr_session_result* crispasr_session_transcribe_vad(crispasr_session* s, const float* pcm, int n, int sr,
                                                         const char* vad, const crispasr_vad_abi_opts* opts) {
    return crispasr_session_transcribe(s, pcm, n);
}
crispasr_session_result* crispasr_session_transcribe_vad_lang(crispasr_session* s, const float* pcm, int n, int sr,
                                                              const char* vad, const crispasr_vad_abi_opts* opts,
                                                              const char* lang) {
    return crispasr_session_transcribe(s, pcm, n);
}
int crispasr_session_set_return_logits(crispasr_session* s, int v) {
    return 0;
}

// --- Music ---
int crispasr_session_piano(crispasr_session* s, const float* pcm, int n) {
    return check_pcm(pcm, n);
}
const float* crispasr_session_piano_notes(crispasr_session* s, int* n) {
    *n = probe_mode == 1 ? 0 : 2;
    return *n ? probe_buffer : NULL;
}
const int* crispasr_session_piano_note_programs(crispasr_session* s, int* n) {
    *n = probe_mode == 1 ? 0 : 2;
    return *n ? probe_programs : NULL;
}
int crispasr_session_piano_n_notes(crispasr_session* s) {
    return probe_mode == 1 ? 0 : 2;
}
int crispasr_session_piano_sample_rate(crispasr_session* s) {
    return 16000;
}
int crispasr_session_chords(crispasr_session* s, const float* pcm, int n, int rate) {
    if (rate <= 0)
        return -1;
    return check_pcm(pcm, n);
}
const float* crispasr_session_chords_spans(crispasr_session* s, int* n) {
    *n = probe_mode == 1 ? 0 : 2;
    return *n ? probe_buffer : NULL;
}
int crispasr_session_chords_n_spans(crispasr_session* s) {
    return probe_mode == 1 ? 0 : 2;
}
const char* crispasr_session_chords_span_name(crispasr_session* s, int i) {
    return i ? "Am" : "C";
}
int crispasr_session_chords_vocab_size(crispasr_session* s) {
    return 170;
}
int crispasr_session_beats(crispasr_session* s, const float* pcm, int n, int rate) {
    if (rate != 22050)
        return -1;
    return check_pcm(pcm, n);
}
const float* crispasr_session_beats_events(crispasr_session* s, int* n) {
    *n = probe_mode == 1 ? 0 : 2;
    return *n ? probe_buffer : NULL;
}
int crispasr_session_beats_n_events(crispasr_session* s) {
    return probe_mode == 1 ? 0 : 2;
}
int crispasr_session_beats_sample_rate(crispasr_session* s) {
    return 22050;
}
float crispasr_session_beats_tempo_bpm(crispasr_session* s) {
    return 123.5f;
}
int crispasr_session_tab(crispasr_session* s, const float* pcm, int n, int rate) {
    return check_pcm(pcm, n);
}
const float* crispasr_session_tab_emissions(crispasr_session* s, int* n, int* strings, int* classes) {
    *n = probe_mode == 1 ? 0 : 2;
    *strings = 2;
    *classes = 3;
    return *n ? probe_buffer : NULL;
}
int crispasr_session_tab_n_frames(crispasr_session* s) {
    return probe_mode == 1 ? 0 : 2;
}
int crispasr_session_tab_silent_class(crispasr_session* s) {
    return 2;
}
float crispasr_session_tab_frame_period(crispasr_session* s) {
    return 0.02f;
}
int crispasr_session_tab_string_open_midi(crispasr_session* s, int i) {
    return 40 + i * 5;
}
int crispasr_session_pitch(crispasr_session* s, const float* pcm, int n, float hop) {
    return check_pcm(pcm, n);
}
const float* crispasr_session_pitch_frames(crispasr_session* s, int* n) {
    *n = probe_mode == 1 ? 0 : 2;
    return *n ? probe_buffer : NULL;
}
int crispasr_session_pitch_frame(crispasr_session* s, int idx, float* t, float* f0, float* vp) {
    if (idx < 0 || idx >= 2)
        return -1;
    *t = (float)idx;
    *f0 = 100 + idx;
    *vp = 0.9f;
    return 0;
}
int crispasr_session_pitch_n_frames(crispasr_session* s) {
    return probe_mode == 1 ? 0 : 2;
}
int crispasr_session_pitch_sample_rate(crispasr_session* s) {
    return 16000;
}
int crispasr_session_convert_content_dim(crispasr_session* s) {
    return 2;
}
int crispasr_session_convert_n_speakers(crispasr_session* s) {
    return 1;
}
int crispasr_session_convert_sample_rate(crispasr_session* s) {
    return 40000;
}
int crispasr_session_convert_inter_channels(crispasr_session* s) {
    return 2;
}
int crispasr_session_convert_upsample_product(crispasr_session* s) {
    return 4;
}
int crispasr_session_convert(crispasr_session* s, const float* content, int n, const float* f0, int id, const float* zp,
                             const float* sine) {
    probe_calls++;
    if (n != 3 || content[1] != 1 || f0[2] != 5 || id)
        return -1;
    if (zp && zp[5] != 5)
        return -1;
    if (sine && sine[11] != 11)
        return -1;
    if (probe_mode == 2)
        return -1;
    if (probe_mode == 1)
        return 0;
    for (int i = 0; i < 6; i++)
        probe_buffer[i] = (float)i;
    return 6;
}
const float* crispasr_session_convert_audio(crispasr_session* s, int* n) {
    *n = 6;
    return probe_buffer;
}
int crispasr_session_set_voice_samples(crispasr_session* s, const float* pcm, int32_t n, int32_t rate,
                                       const char* ref) {
    if (rate != 24000)
        return -1;
    if (ref && strcmp(ref, "hello") == 0) {
        int r = check_pcm(pcm, n);
        return r < 0 ? r : (r == 2 ? 0 : r);
    }
    return -1;
}
int crispasr_session_set_tts_cfg_scale(crispasr_session* s, float scale) {
    return scale == 1.5f ? 0 : -2;
}
int crispasr_session_tada_set_makeref_models(crispasr_session* s, const char* e, const char* a) {
    return !e && !a ? 0 : -1;
}
const char* crispasr_session_disclaimer_text(void) {
    return "AI generated";
}
float* crispasr_session_get_disclaimer_pcm(crispasr_session* s, int* n) {
    *n = probe_mode == 1 ? 0 : 2;
    float* p = (float*)malloc(8);
    if (p) {
        p[0] = 0.25f;
        p[1] = 0.5f;
        probe_live++;
    }
    return p;
}
void crispasr_pcm_free(float* p) {
    if (p) {
        free(p);
        probe_live--;
    }
}
float* crispasr_session_synthesize(crispasr_session* s, const char* text, int* n) {
    return crispasr_session_get_disclaimer_pcm(s, n);
}
float* crispasr_session_synthesize_raw(crispasr_session* s, const char* text, int* n) {
    if (probe_mode == 35 || probe_mode == 36) {
        *n = 0;
        return NULL;
    }
    return crispasr_session_get_disclaimer_pcm(s, n);
}
int crispasr_session_synthesize_streaming(crispasr_session* s, const char* text, crispasr_pcm_stream_cb cb, void* ud) {
    float pcm1[2] = {0.25f, 0.5f};
    float pcm2[2] = {0.75f, 1.0f};
    if (probe_mode == 2) {
        if (cb)
            cb(NULL, 0, 0, ud);
        return 0;
    }
    if (cb)
        cb(pcm1, 2, 0, ud);
    if (cb)
        cb(pcm2, 2, 1, ud);
    return 0;
}
int crispasr_session_accept_marking_responsibility(crispasr_session* s, const char* a) {
    return probe_mode == 2 ? -1 : 0;
}
const char* crispasr_session_last_synth_error(crispasr_session* s) {
    if (probe_mode == 36)
        return NULL;
    return "probe synthesis failure";
}
int crispasr_session_set_source_language(crispasr_session* s, const char* lang) {
    return lang && strcmp(lang, "de") == 0 ? 0 : -1;
}
int crispasr_session_set_pcm_sample_rate(crispasr_session* s, int rate) {
    return 0;
}
int crispasr_session_set_parakeet_att_context(crispasr_session* s, int l, int r) {
    return 0;
}
int crispasr_session_set_audio_ctx(crispasr_session* s, int audio_ctx) {
    return 0;
}
int crispasr_session_set_hikari_refresh_stride(crispasr_session* s, int stride) {
    return 0;
}
int crispasr_session_set_grammar_strict(crispasr_session* s, int v) {
    return 0;
}
int crispasr_session_set_target_language(crispasr_session* s, const char* l) {
    return 0;
}
int crispasr_session_set_tts_reference_language(crispasr_session* s, const char* l) {
    return 0;
}
int crispasr_session_set_punctuation(crispasr_session* s, int e) {
    return 0;
}
int crispasr_session_set_punc_model(crispasr_session* s, const char* p) {
    return 0;
}
int crispasr_session_set_hotwords(crispasr_session* s, const char* h, float b) {
    return 0;
}
int crispasr_session_set_sensitivity(crispasr_session* s, const char* p) {
    return 0;
}
int crispasr_session_set_g2p_dict(crispasr_session* s, const char* p) {
    return 0;
}
int crispasr_session_set_speaker_id(crispasr_session* s, int id) {
    return 0;
}
int crispasr_session_set_translate(crispasr_session* s, int e) {
    return 0;
}
char* crispasr_session_translate_text(crispasr_session* s, const char* text, const char* src, const char* tgt,
                                      int max) {
    return alloc_text(probe_mode == 1 ? "" : "hello");
}
void crispasr_session_translate_text_free(char* t) {
    tracked_free(t);
}
int crispasr_session_set_temperature(crispasr_session* s, float t, uint64_t sd) {
    return 0;
}
int crispasr_session_set_tts_seed(crispasr_session* s, uint64_t sd) {
    return 0;
}
int crispasr_session_set_max_new_tokens(crispasr_session* s, int n) {
    return 0;
}
int crispasr_session_set_frequency_penalty(crispasr_session* s, float p) {
    return 0;
}
int crispasr_session_set_tts_steps(crispasr_session* s, int n) {
    return 0;
}
int crispasr_session_set_tts_num_candidates(crispasr_session* s, int n) {
    return 0;
}
int crispasr_session_set_top_p(crispasr_session* s, float p) {
    return 0;
}
int crispasr_session_set_top_k(crispasr_session* s, int k) {
    return 0;
}
int crispasr_session_set_do_sample(crispasr_session* s, int e) {
    return 0;
}
int crispasr_session_set_min_p(crispasr_session* s, float p) {
    return 0;
}
int crispasr_session_set_repetition_penalty(crispasr_session* s, float r) {
    return 0;
}
int crispasr_session_set_cfg_weight(crispasr_session* s, float w) {
    return 0;
}
int crispasr_session_set_tts_noise_temp(crispasr_session* s, float t) {
    return 0;
}
int crispasr_session_set_exaggeration(crispasr_session* s, float e) {
    return 0;
}
int crispasr_session_set_max_speech_tokens(crispasr_session* s, int n) {
    return 0;
}
int crispasr_session_set_min_speech_tokens(crispasr_session* s, int n) {
    return 0;
}
int crispasr_session_set_length_scale(crispasr_session* s, float sc) {
    return 0;
}
int crispasr_session_set_best_of(crispasr_session* s, int n) {
    return 0;
}
int crispasr_session_set_beam_size(crispasr_session* s, int n) {
    return 0;
}
int crispasr_session_set_grammar_text(crispasr_session* s, const char* g, const char* r, float p) {
    return 0;
}
int crispasr_session_set_fallback_thresholds(crispasr_session* s, float a, float b, float c, float d) {
    return 0;
}
int crispasr_session_set_alt_n(crispasr_session* s, int n) {
    return 0;
}
int crispasr_session_set_whisper_decode_extras(crispasr_session* s, int a, const char* b, int c) {
    return 0;
}
int crispasr_session_set_ask(crispasr_session* s, const char* p) {
    return 0;
}
int crispasr_session_detect_language(crispasr_session* s, const float* pcm, int n, const char* m, int met, char* out,
                                     int cap, float* prob) {
    return 0;
}
int crispasr_session_set_codec_path(crispasr_session* s, const char* p) {
    return 0;
}
int crispasr_session_set_voice(crispasr_session* s, const char* p, const char* t) {
    return 0;
}
int crispasr_session_set_speaker_name(crispasr_session* s, const char* n) {
    return 0;
}
int crispasr_session_n_speakers(crispasr_session* s) {
    return 1;
}
const char* crispasr_session_get_speaker_name(crispasr_session* s, int i) {
    return "speaker0";
}
int crispasr_session_set_instruct(crispasr_session* s, const char* i) {
    return 0;
}
int crispasr_session_set_tts_phonemes(crispasr_session* s, const char* p) {
    return 0;
}
void crispasr_session_set_tts_pad_silence_ms(crispasr_session* s, int ms) {}
int crispasr_session_is_custom_voice(crispasr_session* s) {
    return 0;
}
int crispasr_session_is_voice_design(crispasr_session* s) {
    return 0;
}
float* crispasr_session_speech_to_speech(crispasr_session* s, const float* pcm, int n, char** text, int* out_n) {
    if (check_pcm(pcm, n) < 0 && probe_mode != 2)
        return NULL;
    *text = alloc_text(probe_mode == 2 ? "\xff" : "hello");
    return crispasr_session_get_disclaimer_pcm(s, out_n);
}
int crispasr_session_separate(crispasr_session* s, const float* pcm, int n) {
    return 2;
}
int crispasr_session_separate_n_stems(crispasr_session* s) {
    return 2;
}
const char* crispasr_session_separate_stem_name(crispasr_session* s, int i) {
    return i ? "drums" : "vocals";
}
const float* crispasr_session_separate_stem(crispasr_session* s, int i, int* n) {
    *n = 4;
    return probe_buffer;
}
int crispasr_session_separate_sample_rate(crispasr_session* s) {
    return 44100;
}
int crispasr_session_input_sample_rate(crispasr_session* s) {
    return 16000;
}
int crispasr_session_output_sample_rate(crispasr_session* s) {
    return 24000;
}
int crispasr_session_input_channels(crispasr_session* s) {
    return 1;
}
int crispasr_session_output_channels(crispasr_session* s) {
    return 1;
}
void crispasr_watermark_embed(float* pcm, int n, float a) {}
float crispasr_watermark_detect(const float* pcm, int n) {
    return 0.5f;
}
int crispasr_session_kokoro_clear_phoneme_cache(crispasr_session* s) {
    return 0;
}
int crispasr_session_score_texts(crispasr_session* s, const float* pcm, int n, const char* lang, const char* prompt,
                                 const char** texts, int nt, float* out_lp, int* out_nt) {
    if (probe_mode == 2)
        return 1;
    if (!pcm || n != 3 || pcm[0] != 1 || pcm[2] != 5 || nt != 2 || strcmp(texts[0], "one") ||
        strcmp(texts[1], "two words") || (lang && strcmp(lang, "ja")) || (prompt && strcmp(prompt, "prompt")))
        return -1;
    out_lp[0] = -.25f;
    out_lp[1] = -.75f;
    out_nt[0] = 2;
    out_nt[1] = 3;
    return 0;
}

int crispasr_get_streamed_segment_count(void) {
    return streamed_ready;
}
crispasr_session_result* crispasr_drain_streamed_segments(void) {
    if (!streamed_ready)
        return NULL;
    streamed_ready = 0;
    probe_token_head[0] = (char)0xe6;
    probe_token_head[1] = (char)0x97;
    probe_token_head[2] = 0;
    probe_token_tail[0] = (char)0xa5;
    probe_token_tail[1] = 0;
    return (crispasr_session_result*)make_result(1);
}
void crispasr_reset_streamed_segments(void) {
    streamed_ready = 0;
}
int crispasr_get_streamed_token_count(void) {
    return 0;
}
const char* crispasr_drain_streamed_tokens(int* n) {
    static const char data[] = "tok1\0tok2\0";
    if (probe_mode == 1) {
        *n = 0;
        return NULL;
    }
    if (probe_mode == 2) {
        *n = -1;
        return NULL;
    }
    if (probe_mode == 3) {
        *n = 2;
        return NULL;
    }
    if (probe_mode == 5) {
        static const char fragments[] = {(char)0xe6, (char)0x97, 0, (char)0xa5, 0, 0};
        *n = 3;
#if defined(__linux__)
        static char* guarded;
        if (!guarded) {
            long page = sysconf(_SC_PAGESIZE);
            char* memory = mmap(NULL, (size_t)page * 2, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (memory == MAP_FAILED || mprotect(memory + page, (size_t)page, PROT_NONE))
                abort();
            guarded = memory + page - sizeof(fragments);
            memcpy(guarded, fragments, sizeof(fragments));
        }
        return guarded;
#else
        return fragments;
#endif
    }
    *n = 2;
    return data;
}
void crispasr_reset_streamed_tokens(void) {}
void crispasr_free_aligner_cache(void) {}
int crispasr_detect_backend_from_gguf(const char* p, char* out, int cap) {
    if (cap < 7)
        return -2;
    strcpy(out, "probe");
    return 5;
}
int crispasr_detect_backends_from_gguf(const char* p, char* out, int cap) {
    if (cap < 7)
        return -5;
    strcpy(out, "probe\n");
    return 6;
}
int crispasr_backend_caps_abi(const char* b, char* out, int cap) {
    if (!strcmp(b, "long")) {
        if (cap < 601)
            return -2;
        memset(out, 'x', 600);
        out[600] = 0;
        return 600;
    }
    if (strcmp(b, "probe"))
        return -3;
    if (cap < 8)
        return -2;
    strcpy(out, "asr,tts");
    return 7;
}
int crispasr_backend_caps_list_abi(char* out, int cap) {
    const char* d = "probe\tasr,tts\n";
    int n = strlen(d) + 1;
    if (!out || cap < n)
        return -n;
    memcpy(out, d, n);
    return n - 1;
}
int crispasr_session_available_backends(char* out, int cap) {
    const char* s = "whisper,probe";
    int n = strlen(s) + 1;
    if (cap < n)
        return -1;
    strcpy(out, s);
    return n - 1;
}
int crispasr_session_n_vocab(crispasr_session* s) {
    return 10;
}
const char* crispasr_session_token_text(crispasr_session* s, int id) {
    return "tok";
}
int crispasr_session_detected_language(crispasr_session* s, char* buf, int cap) {
    if (cap < 3)
        return -1;
    strcpy(buf, "en");
    return 2;
}
int crispasr_diarize_segments_abi(const float* l, const float* r, int n, int st, crispasr_diarize_seg_abi* segs,
                                  int nsegs, const crispasr_diarize_opts_abi* opts) {
    return 0;
}
int crispasr_diarize_segments_turns_abi(const float* l, const float* r, int n, int st, crispasr_diarize_seg_abi* segs,
                                        int nsegs, const crispasr_diarize_opts_abi* opts,
                                        crispasr_diarize_turn_abi* out, int cap, int* out_n) {
    if (out_n)
        *out_n = 0;
    return 0;
}
int crispasr_detect_language_pcm(const float* p, int n, int m, const char* path, int nt, int gpu, int dev, int fl,
                                 char* out, int cap, float* conf) {
    return 0;
}
int crispasr_enhance_audio_rnnoise(const float* in, int n, float* out, int cap) {
    return 0;
}
int crispasr_text_detect_language(const char* t, const char* m, int nt, char* out, int cap, float* conf) {
    return 0;
}
crispasr_align_result* crispasr_align_words_abi(const char* m, const char* t, const float* p, int n, int64_t off,
                                                int nt) {
    return NULL;
}
int crispasr_align_result_n_words(crispasr_align_result* r) {
    return 0;
}
const char* crispasr_align_result_word_text(crispasr_align_result* r, int i) {
    return "";
}
int64_t crispasr_align_result_word_t0(crispasr_align_result* r, int i) {
    return 0;
}
int64_t crispasr_align_result_word_t1(crispasr_align_result* r, int i) {
    return 0;
}
int crispasr_align_result_n_characters(crispasr_align_result* r, int w) {
    return 0;
}
const char* crispasr_align_result_character_text(crispasr_align_result* r, int w, int i) {
    return "";
}
int64_t crispasr_align_result_character_t0(crispasr_align_result* r, int w, int i) {
    return 0;
}
int64_t crispasr_align_result_character_t1(crispasr_align_result* r, int w, int i) {
    return 0;
}
void crispasr_align_result_free(crispasr_align_result* r) {}
int crispasr_cache_ensure_file_abi(const char* f, const char* u, int q, const char* d, char* out, int cap) {
    return 0;
}
int crispasr_cache_dir_abi(const char* d, char* out, int cap) {
    return 0;
}
int crispasr_registry_lookup_abi(const char* b, char* f, int fc, char* u, int uc, char* s, int sc) {
    return 1;
}
int crispasr_registry_lookup_by_filename_abi(const char* f, char* out, int oc, char* u, int uc, char* s, int sc) {
    return 1;
}
int crispasr_registry_list_backends_abi(char* out, int cap) {
    return 0;
}
int crispasr_registry_default_bundle_info_abi(const char* b, char* ob, int bc, char* ol, int lc, int* req) {
    return 0;
}
int crispasr_registry_default_bundle_artifact_abi(const char* b, int idx, int* k, char* f, int fc, char* u, int uc,
                                                  char* s, int sc) {
    return -1;
}
int crispasr_get_progress(void) {
    return -1;
}
void crispasr_reset_progress(void) {}
int crispasr_watermark_load_model(const char* p) {
    return 0;
}
int crispasr_lcs_dedup_prefix_count(const int32_t* a, int na, const int32_t* b, int nb, int m) {
    return 0;
}
crispasr_stream* crispasr_stream_open(struct whisper_context* ctx, int nt, int step, int len, int keep,
                                      const char* lang, int tr) {
    return NULL;
}
void crispasr_stream_close(crispasr_stream* s) {
    --probe_streams;
    ++probe_stream_closes;
}
int crispasr_stream_feed(crispasr_stream* s, const float* pcm, int n) {
    return 0;
}
int crispasr_stream_get_text(crispasr_stream* s, char* out, int cap, double* t0, double* t1, int64_t* c) {
    return 0;
}
int crispasr_stream_flush(crispasr_stream* s) {
    return 0;
}
void crispasr_stream_set_live_decode(crispasr_stream* s, int e) {}
#include "wrapper-parakeet-legacy.c"
int crispasr_vad_segments(const char* m, const float* pcm, int n, int sr, float th, int min_s, int min_si, int nt,
                          bool gpu, float** out) {
    return 0;
}
int crispasr_vad_slices(const char* m, const float* pcm, int n, int sr, float th, int min_s, int min_si, int pad,
                        float max_dur, int nt, float** out) {
    return 0;
}
void crispasr_vad_free(float* p) {}
#define CRISPASR_PROBE_AUDIO_ONLY
#include "wrapper-audio-utilities.c"
#undef CRISPASR_PROBE_AUDIO_ONLY
int crispasr_session_set_speaker_identity(crispasr_session* s, const char* id) {
    return 0;
}

// --- Unexercised legacy/chat/whisper stubs MUST abort, never fake success ---
#include <stdio.h>
#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif
#define STUB_ABORT()                                                                                                   \
    do {                                                                                                               \
        fprintf(stderr, "probe stub %s called but not exercised\n", __func__);                                         \
        abort();                                                                                                       \
    } while (0)


static char legacy_token_prefix[] = "\xe6\x97";
static char legacy_token_suffix[] = "\xa5";
static char legacy_alt_prefix[] = "\xe6\x97";
static char legacy_alt_long[2049];
static char legacy_oversized_text[1048578];
static int legacy_alt_text_calls, legacy_rust_alt_n;
void probe_legacy_tokens_reset(void) {
    memcpy(legacy_token_prefix, "\xe6\x97", 3);
    memcpy(legacy_token_suffix, "\xa5", 2);
    memcpy(legacy_alt_prefix, "\xe6\x97", 3);
    memset(legacy_alt_long, 'L', sizeof(legacy_alt_long) - 1);
    legacy_alt_long[sizeof(legacy_alt_long) - 1] = 0;
    legacy_alt_text_calls = 0;
}
void probe_legacy_tokens_mutate(void) {
    memset(legacy_token_prefix, 'x', 2);
    legacy_token_suffix[0] = 'x';
    memset(legacy_alt_prefix, 'x', 2);
    memset(legacy_alt_long, 'x', sizeof(legacy_alt_long) - 1);
}
int probe_legacy_alt_text_calls(void) {
    return legacy_alt_text_calls;
}
int probe_legacy_rust_alt_n(void) {
    return legacy_rust_alt_n;
}

void probe_legacy_controls_expect(int mode) {
    legacy_controls_expected = mode;
}
int probe_legacy_param_requests(void) {
    return legacy_param_requests;
}
static void seed_legacy_controls(struct whisper_full_params* p) {
    if (!legacy_controls_expected)
        return;
    p->translate = legacy_controls_expected != 1;
    p->detect_language = legacy_controls_expected != 1;
    p->token_timestamps = legacy_controls_expected != 1;
    p->split_on_word = legacy_controls_expected != 1;
    p->no_context = legacy_controls_expected != 1;
    p->single_segment = legacy_controls_expected != 1;
    p->print_realtime = legacy_controls_expected != 1;
    p->print_progress = legacy_controls_expected != 1;
    p->print_timestamps = legacy_controls_expected != 1;
    p->print_special = legacy_controls_expected != 1;
    p->suppress_blank = legacy_controls_expected != 1;

    p->n_threads = 4;
    p->max_len = 99;
    p->greedy.best_of = 5;
    p->max_tokens = 17;
    p->temperature = .125f;
    p->alt_n = 7;
    p->initial_prompt = "native default";
}
static int check_legacy_controls(const struct whisper_full_params* p) {
    if (!legacy_controls_expected)
        return 1;
    int mode = legacy_controls_expected;
    int okay = 1;
    char* churn[8];
    for (int i = 0; i < 8; ++i) {
        churn[i] = malloc(16);
        if (churn[i]) {
            memset(churn[i], 'X', 15);
            churn[i][15] = 0;
        }
    }
#define CONTROL_CHECK(expr)                                                                                            \
    do {                                                                                                               \
        if (!(expr)) {                                                                                                 \
            fprintf(stderr, "control mismatch: %s\n", #expr);                                                          \
            okay = 0;                                                                                                  \
            goto done;                                                                                                 \
        }                                                                                                              \
    } while (0)
    CONTROL_CHECK(p->translate == (mode != 2));
    CONTROL_CHECK(p->detect_language == (mode != 2));
    CONTROL_CHECK(p->token_timestamps == (mode != 2));
    CONTROL_CHECK(p->split_on_word == (mode != 2));
    CONTROL_CHECK(p->no_context == (mode != 2));
    CONTROL_CHECK(p->single_segment == (mode != 2));
    CONTROL_CHECK(p->print_realtime == (mode != 2));
    CONTROL_CHECK(p->print_progress == (mode != 2));
    CONTROL_CHECK(p->print_timestamps == (mode != 2));
    CONTROL_CHECK(p->print_special == (mode != 2));
    CONTROL_CHECK(p->suppress_blank == (mode != 2));

    CONTROL_CHECK(p->n_threads == (mode == 1 ? 3 : mode == 2 ? 1 : 4));
    CONTROL_CHECK(p->max_len == (mode == 1 ? 77 : mode == 2 ? 0 : 99));
    CONTROL_CHECK(p->greedy.best_of == (mode == 1 ? 7 : mode == 2 ? 1 : 5));
    CONTROL_CHECK(p->max_tokens == (mode == 1 ? 23 : mode == 2 ? 0 : 17));
    CONTROL_CHECK(p->alt_n == (mode == 1 ? 32 : mode == 2 ? 0 : 7));
    CONTROL_CHECK(p->temperature == (mode == 1 ? .625f : mode == 2 ? 0.f : .125f));
    CONTROL_CHECK(p->language && strcmp(p->language, mode == 1 ? "ja" : mode == 2 ? "auto" : "de") == 0);
    CONTROL_CHECK(p->initial_prompt && strcmp(p->initial_prompt, mode == 1   ? "Grüße 日"
                                                                 : mode == 2 ? ""
                                                                             : "native default") == 0);
#undef CONTROL_CHECK
done:
    for (int i = 0; i < 8; ++i)
        free(churn[i]);
    return okay;
}

// Literal public-header legacy Rust transport; by-value entry remains a trap.
static int rust_legacy_contexts;
static struct whisper_context_params last_legacy_context_params;
int probe_legacy_context_flash(void) {
    return last_legacy_context_params.flash_attn;
}
int probe_dtw_enabled(void) {
    return last_legacy_context_params.dtw_token_timestamps;
}
int probe_dtw_preset(void) {
    return last_legacy_context_params.dtw_aheads_preset;
}
int probe_dtw_top(void) {
    return last_legacy_context_params.dtw_n_top;
}
int probe_rust_legacy_contexts(void) {
    return rust_legacy_contexts;
}
static whisper_context* rust_legacy_context(void) {
    return (whisper_context*)0x100000003ULL;
}
// whisper legacy stubs
#include "wrapper-whisper-go-params.c"
struct whisper_context* whisper_init_from_file_with_params(const char* p, struct whisper_context_params pr) {
    STUB_ABORT();
    return NULL;
}
struct whisper_context_params* whisper_context_default_params_by_ref(void) {
    if (probe_mode == 10)
        return NULL;
    struct whisper_context_params* p = tracked_alloc(sizeof(*p));
    if (p) {
        memset(p, 0, sizeof(*p));
        p->use_gpu = true;
        p->flash_attn = true;
    }
    return p;
}
void whisper_free(struct whisper_context* ctx) {
    if (ctx != rust_legacy_context())
        STUB_ABORT();
    --rust_legacy_contexts;
}
void whisper_free_params(struct whisper_full_params* p) {
    tracked_free(p);
}
void whisper_free_context_params(struct whisper_context_params* p) {
    tracked_free(p);
}
int whisper_full(struct whisper_context* ctx, struct whisper_full_params p, const float* s, int n) {
    return probe_legacy_full(ctx, p, s, n);
}
struct whisper_full_params* whisper_full_default_params_by_ref(enum whisper_sampling_strategy s) {
    ++legacy_param_requests;
    if (probe_mode == 11)
        return NULL;
    struct whisper_full_params* p = tracked_alloc(sizeof(*p));
    if (p) {
        memset(p, 0, sizeof(*p));
        p->strategy = s;
        p->language = "de";
        seed_legacy_controls(p);
    }
    return p;
}
int whisper_full_by_ref(whisper_context* ctx, struct whisper_full_params* p, const float* samples, int n) {
    ++probe_calls;
    legacy_rust_alt_n = p ? p->alt_n : -1;
    if (ctx != rust_legacy_context() || !p || n != 3 || samples[0] != 1 || samples[1] != 3 || samples[2] != 5)
        return -9;
    if (p->strategy != CRISPASR_SAMPLING_GREEDY && p->strategy != CRISPASR_SAMPLING_BEAM_SEARCH)
        return -8;
    if (p->vad &&
        (!p->tdrz_enable || !p->vad_model_path || strcmp(p->vad_model_path, "vad") || p->vad_params.threshold != .7f ||
         p->vad_params.min_speech_duration_ms != 300 || p->vad_params.min_silence_duration_ms != 200))
        return -7;
    if (!check_legacy_controls(p))
        return -10;
    if (probe_mode == 2)
        return -1;
    // Go's real default-params adapter installs native callback trampolines;
    // exercise those here too when reached through the pointer parallel route.
    if (p->encoder_begin_callback)
        return probe_legacy_full(ctx, *p, samples, n);
    return 0;
}
static int parallel_by_ref_calls;
int probe_parallel_by_ref_calls(void) {
    return parallel_by_ref_calls;
}
int crispasr_transcribe_parallel_by_ref(struct whisper_context* ctx, const struct whisper_full_params* p,
                                        const float* samples, int n, int processors) {
    ++parallel_by_ref_calls;
    if (processors != 2) {
        ++probe_calls;
        return -8;
    }
    if (ctx == rust_legacy_context())
        return whisper_full_by_ref(ctx, (struct whisper_full_params*)p, samples, n);
    return probe_legacy_full(ctx, *p, samples, n);
}
int whisper_full_n_segments(struct whisper_context* ctx) {
    if (ctx != rust_legacy_context())
        STUB_ABORT();
    return probe_mode == 12 ? -1 : probe_mode == 1 ? 0 : 1;
}
const char* whisper_full_get_segment_text(struct whisper_context* ctx, int i) {
    if (ctx != rust_legacy_context())
        STUB_ABORT();
    return "日";
}
int64_t whisper_full_get_segment_t0(struct whisper_context* ctx, int i) {
    if (ctx != rust_legacy_context())
        STUB_ABORT();
    return 100;
}
int64_t whisper_full_get_segment_t1(struct whisper_context* ctx, int i) {
    if (ctx != rust_legacy_context())
        STUB_ABORT();
    return 200;
}
float whisper_full_get_segment_no_speech_prob(struct whisper_context* ctx, int i) {
    if (ctx != rust_legacy_context())
        STUB_ABORT();
    return .125f;
}
int whisper_full_lang_id(struct whisper_context* ctx) {
    if (ctx != rust_legacy_context() || rust_legacy_contexts <= 0)
        STUB_ABORT();
    return 0;
}
const char* whisper_lang_str(int id) {
    return id == 0 ? "en" : id == 1 ? "de" : NULL;
}
int whisper_lang_id(const char* lang) {
    return lang && strcmp(lang, "en") == 0 ? 0 : lang && strcmp(lang, "de") == 0 ? 1 : -1;
}
int whisper_lang_max_id(void) {
    return 1;
}
int whisper_lang_auto_detect(struct whisper_context* ctx, int off, int nt, float* probs) {
    if (ctx != rust_legacy_context() || rust_legacy_contexts <= 0 || off < 0 || nt <= 0 || !probs)
        STUB_ABORT();
    probs[0] = .75f;
    probs[1] = .25f;
    return 0;
}
int whisper_n_len(struct whisper_context* ctx) {
    STUB_ABORT();
    return 0;
}
int whisper_n_vocab(struct whisper_context* ctx) {
    STUB_ABORT();
    return 0;
}
int whisper_n_text_ctx(struct whisper_context* ctx) {
    STUB_ABORT();
    return 0;
}
int whisper_n_audio_ctx(struct whisper_context* ctx) {
    STUB_ABORT();
    return 0;
}
int whisper_is_multilingual(struct whisper_context* ctx) {
    if (ctx != rust_legacy_context() || rust_legacy_contexts <= 0)
        STUB_ABORT();
    return 1;
}
const char* whisper_token_to_str(struct whisper_context* ctx, int tok) {
    STUB_ABORT();
    return "";
}
int whisper_token_eot(struct whisper_context* ctx) {
    if (ctx != rust_legacy_context() || rust_legacy_contexts <= 0)
        STUB_ABORT();
    return 100;
}
int whisper_token_sot(struct whisper_context* ctx) {
    if (ctx != rust_legacy_context() || rust_legacy_contexts <= 0)
        STUB_ABORT();
    return 101;
}
int whisper_token_prev(struct whisper_context* ctx) {
    if (ctx != rust_legacy_context() || rust_legacy_contexts <= 0)
        STUB_ABORT();
    return 102;
}
int whisper_token_solm(struct whisper_context* ctx) {
    if (ctx != rust_legacy_context() || rust_legacy_contexts <= 0)
        STUB_ABORT();
    return 103;
}
int whisper_token_not(struct whisper_context* ctx) {
    if (ctx != rust_legacy_context() || rust_legacy_contexts <= 0)
        STUB_ABORT();
    return 104;
}
int whisper_token_beg(struct whisper_context* ctx) {
    if (ctx != rust_legacy_context() || rust_legacy_contexts <= 0)
        STUB_ABORT();
    return 105;
}
int whisper_token_lang(struct whisper_context* ctx, int id) {
    if (ctx != rust_legacy_context() || rust_legacy_contexts <= 0 || id < 0 || id > 1)
        STUB_ABORT();
    return 200 + id;
}
int whisper_token_translate(struct whisper_context* ctx) {
    STUB_ABORT();
    return 0;
}
int whisper_token_transcribe(struct whisper_context* ctx) {
    STUB_ABORT();
    return 0;
}
void whisper_print_timings(struct whisper_context* ctx) {
    if (ctx != rust_legacy_context() || rust_legacy_contexts <= 0)
        STUB_ABORT();
}
void whisper_reset_timings(struct whisper_context* ctx) {
    if (ctx != rust_legacy_context() || rust_legacy_contexts <= 0)
        STUB_ABORT();
}
const char* whisper_print_system_info(void) {
    return "probe CPU";
}
void whisper_log_set(ggml_log_callback a, void* b) {
    STUB_ABORT();
}
int whisper_pcm_to_mel(struct whisper_context* ctx, const float* s, int n, int nt) {
    STUB_ABORT();
    return 0;
}
int whisper_set_mel(struct whisper_context* ctx, const float* d, int l, int m) {
    STUB_ABORT();
    return 0;
}
int whisper_encode(struct whisper_context* ctx, int off, int nt) {
    STUB_ABORT();
    return 0;
}
int whisper_decode(struct whisper_context* ctx, const int* t, int nt, int past, int n) {
    STUB_ABORT();
    return 0;
}
int whisper_tokenize(struct whisper_context* ctx, const char* t, int* toks, int n) {
    STUB_ABORT();
    return 0;
}
float* whisper_get_logits(struct whisper_context* ctx) {
    STUB_ABORT();
    return NULL;
}
struct whisper_state* whisper_init_state(struct whisper_context* ctx) {
    STUB_ABORT();
    return NULL;
}
void whisper_free_state(struct whisper_state* s) {
    STUB_ABORT();
}
int whisper_full_with_state(struct whisper_context* ctx, struct whisper_state* s, struct whisper_full_params p,
                            const float* smp, int n) {
    STUB_ABORT();
    return 0;
}
int whisper_full_parallel(struct whisper_context* ctx, struct whisper_full_params p, const float* s, int n, int nproc) {
    return nproc == 2 ? probe_legacy_full(ctx, p, s, n) : -8;
}
int whisper_bench_memcpy(int n) {
    STUB_ABORT();
    return 0;
}
const char* whisper_bench_memcpy_str(int n) {
    STUB_ABORT();
    return "";
}
int whisper_bench_ggml_mul_mat(int n) {
    STUB_ABORT();
    return 0;
}
const char* whisper_bench_ggml_mul_mat_str(int n) {
    STUB_ABORT();
    return "";
}
struct whisper_vad_context* whisper_vad_init_from_file_with_params(const char* p,
                                                                   struct whisper_vad_context_params pr) {
    STUB_ABORT();
    return NULL;
}
struct whisper_vad_context* whisper_vad_init_with_params(struct whisper_model_loader* l,
                                                         struct whisper_vad_context_params pr) {
    STUB_ABORT();
    return NULL;
}
bool whisper_vad_detect_speech(struct whisper_vad_context* v, const float* s, int n) {
    STUB_ABORT();
    return false;
}
bool whisper_vad_detect_speech_continue(struct whisper_vad_context* v, const float* s, int n) {
    STUB_ABORT();
    return false;
}
void whisper_vad_set_probs(struct whisper_vad_context* v, const float* p, int n) {
    STUB_ABORT();
}
int whisper_vad_n_probs(struct whisper_vad_context* v) {
    STUB_ABORT();
    return 0;
}
float* whisper_vad_probs(struct whisper_vad_context* v) {
    STUB_ABORT();
    return NULL;
}
struct whisper_vad_segments* whisper_vad_segments_from_probs(struct whisper_vad_context* v,
                                                             struct whisper_vad_params p) {
    STUB_ABORT();
    return NULL;
}
struct whisper_vad_segments* whisper_vad_segments_from_samples(struct whisper_vad_context* v,
                                                               struct whisper_vad_params p, const float* s, int n) {
    STUB_ABORT();
    return NULL;
}
int whisper_vad_segments_n_segments(struct whisper_vad_segments* s) {
    STUB_ABORT();
    return 0;
}
float whisper_vad_segments_get_segment_t0(struct whisper_vad_segments* s, int i) {
    STUB_ABORT();
    return 0;
}
float whisper_vad_segments_get_segment_t1(struct whisper_vad_segments* s, int i) {
    STUB_ABORT();
    return 0;
}
void whisper_vad_free_segments(struct whisper_vad_segments* s) {
    STUB_ABORT();
}
void whisper_vad_free(struct whisper_vad_context* ctx) {
    STUB_ABORT();
}
void crispasr_ctx_params_set_dtw(struct whisper_context_params* p, bool e, int a, int n) {
    if (p) {
        if (e)
            p->flash_attn = false;
        p->dtw_token_timestamps = e;
        p->dtw_aheads_preset = (enum whisper_alignment_heads_preset)a;
        p->dtw_n_top = n;
    }
}
int64_t crispasr_token_dtw_t(struct whisper_context* ctx, int s, int t) {
    return whisper_full_get_token_data(ctx, s, t).t_dtw;
}

int crispasr_token_n_alts(struct whisper_context* ctx, int s, int t) {
    if (ctx != rust_legacy_context() || s != 0 || t < 0 || t >= 2)
        STUB_ABORT();
    if (probe_mode == 23)
        return -1;
    if (probe_mode == 24)
        return 33;
    return t == 0 && probe_mode != 34 ? 2 : 0;
}
int32_t crispasr_token_alt_id(struct whisper_context* ctx, int s, int t, int a) {
    if (ctx != rust_legacy_context() || s != 0 || t != 0 || a < 0 || a >= 2)
        STUB_ABORT();
    return 42 + a;
}
float crispasr_token_alt_p(struct whisper_context* ctx, int s, int t, int a) {
    if (ctx != rust_legacy_context() || s != 0 || t != 0 || a < 0 || a >= 2)
        STUB_ABORT();
    return a == 0 ? .25f : -1.f;
}
int crispasr_token_alt_text(struct whisper_context* ctx, int s, int t, int a, char* out, int cap) {
    ++legacy_alt_text_calls;
    if (ctx != rust_legacy_context() || s != 0 || t != 0 || a < 0 || a >= 2 || !out || cap <= 0)
        STUB_ABORT();
    if (probe_mode == 25)
        return -2;
    if (probe_mode == 26)
        return cap;
    if (probe_mode == 27) {
        out[0] = 'z';
        return 1;
    }
    if (probe_mode == 29)
        return 0;
    if (probe_mode == 30)
        return -1;
    if (probe_mode == 33) {
        out[0] = 0;
        out[1] = 'z';
        out[2] = 0;
        return 2;
    }
    const char* text = probe_mode == 28 ? "" : a == 0 ? legacy_alt_prefix : legacy_alt_long;
    int len = (int)strlen(text);
    if (len + 1 > cap)
        return -1;
    memcpy(out, text, len + 1);
    return len;
}
float crispasr_detect_language(struct whisper_context* ctx, const float* pcm, int n, int nt, char* out, int cap) {
    ++probe_calls;
    if (ctx != rust_legacy_context() || !pcm || n != 3 || pcm[0] != 1 || pcm[1] != 3 || pcm[2] != 5 || nt != 3 ||
        cap != 32)
        return -9;
    if (probe_mode == 2)
        return -3;
    memset(out, 0, cap);
    if (probe_mode == 15)
        memset(out, 'x', cap);
    else if (probe_mode == 14)
        out[0] = (char)0xff;
    else if (probe_mode != 16)
        memcpy(out, "de", 2);
    return probe_mode == 13 ? NAN : probe_mode == 17 ? 0 : probe_mode == 18 ? 1.1f : .875f;
}
int crispasr_stream_open_dummy(void) {
    STUB_ABORT();
    return 0;
}


struct whisper_context* whisper_init_from_file_with_params_by_ref(const char* p, struct whisper_context_params* pr) {
    ++probe_calls;
    if (!p || !pr || !pr->use_gpu || strcmp(p, "fail") == 0 || probe_mode == 19)
        return NULL;
    last_legacy_context_params = *pr;
    ++rust_legacy_contexts;
    return rust_legacy_context();
}
int whisper_score_texts(struct whisper_context* ctx, const float* s, int n, const char* l, const char* pr,
                        const char** t, int nt, float* lp, int* ntok, int ntr) {
    STUB_ABORT();
    return 0;
}


// Auto-generated stubs for missing symbols
int64_t crispasr_token_t0(whisper_context* ctx, int i_seg, int i_tok) {
    return whisper_full_get_token_data(ctx, i_seg, i_tok).t0;
}
int64_t crispasr_token_t1(whisper_context* ctx, int i_seg, int i_tok) {
    return whisper_full_get_token_data(ctx, i_seg, i_tok).t1;
}
float crispasr_token_p(whisper_context* ctx, int i_seg, int i_tok) {
    return whisper_full_get_token_data(ctx, i_seg, i_tok).p;
}
int crispasr_session_stream_kind(crispasr_session* s) {
    STUB_ABORT();
    return 0;
}
crispasr_stream* crispasr_session_stream_open(crispasr_session* s, int n_threads, int step_ms, int length_ms,
                                              int keep_ms, const char* language, int translate) {
    ++probe_streams;
    return (crispasr_stream*)2;
}

#include "wrapper-text-processors.c"
int crispasr_transcribe_parallel(struct whisper_context* ctx, struct whisper_full_params params, const float* samples,
                                 int n_samples, int n_processors) {
    STUB_ABORT();
    return 0;
}
bool crispasr_kokoro_lang_is_german_abi(const char* lang) {
    if (!lang)
        return false;
    // German variants for fixture truth-table
    return strcmp(lang, "de") == 0 || strcmp(lang, "de-DE") == 0;
}
bool crispasr_kokoro_lang_has_native_voice_abi(const char* lang) {
    if (!lang)
        return false;
    // Native voice fixture: en and de are native, others not
    return strcmp(lang, "en") == 0 || strcmp(lang, "de") == 0;
}
int crispasr_kokoro_resolve_model_for_lang_abi(const char* model_path, const char* lang, char* out_path,
                                               int out_path_len) {
    STUB_ABORT();
    return 0;
}
int crispasr_kokoro_resolve_fallback_voice_abi(const char* model_path, const char* lang, char* out_path,
                                               int out_path_len, char* out_picked, int out_picked_len) {
    STUB_ABORT();
    return 0;
}
#include "wrapper-titanet-legacy.c"
#include "wrapper-speaker-db.c"
void* crispasr_speaker_db_load(const char* dir_path) {
    STUB_ABORT();
    return 0;
}
int32_t crispasr_speaker_db_enroll(const char* dir_path, const char* name, const float* embedding, int32_t dim) {
    STUB_ABORT();
    return 0;
}
#include "wrapper-speaker-primitives.c"
const char* whisper_version(void) {
    STUB_ABORT();
    return 0;
}
struct whisper_context* whisper_init_from_buffer_with_params(void* buffer, size_t buffer_size,
                                                             struct whisper_context_params params) {
    STUB_ABORT();
    return 0;
}
struct whisper_context* whisper_init_with_params(struct whisper_model_loader* loader,
                                                 struct whisper_context_params params) {
    STUB_ABORT();
    return 0;
}
struct whisper_context* whisper_init_from_file_with_params_no_state(const char* path_model,
                                                                    struct whisper_context_params params) {
    STUB_ABORT();
    return 0;
}
struct whisper_context* whisper_init_from_buffer_with_params_no_state(void* buffer, size_t buffer_size,
                                                                      struct whisper_context_params params) {
    STUB_ABORT();
    return 0;
}
struct whisper_context* whisper_init_with_params_no_state(struct whisper_model_loader* loader,
                                                          struct whisper_context_params params) {
    STUB_ABORT();
    return 0;
}
int whisper_ctx_init_openvino_encoder_with_state(struct whisper_context* ctx, struct whisper_state* state,
                                                 const char* model_path, const char* device, const char* cache_dir) {
    STUB_ABORT();
    return 0;
}
int whisper_ctx_init_openvino_encoder(struct whisper_context* ctx, const char* model_path, const char* device,
                                      const char* cache_dir) {
    STUB_ABORT();
    return 0;
}
int whisper_pcm_to_mel_with_state(struct whisper_context* ctx, struct whisper_state* state, const float* samples,
                                  int n_samples, int n_threads) {
    STUB_ABORT();
    return 0;
}
int whisper_set_mel_with_state(struct whisper_context* ctx, struct whisper_state* state, const float* data, int n_len,
                               int n_mel) {
    STUB_ABORT();
    return 0;
}
int whisper_encode_with_state(struct whisper_context* ctx, struct whisper_state* state, int offset, int n_threads) {
    STUB_ABORT();
    return 0;
}
int whisper_decode_with_state(struct whisper_context* ctx, struct whisper_state* state, const whisper_token* tokens,
                              int n_tokens, int n_past, int n_threads) {
    STUB_ABORT();
    return 0;
}
const char* whisper_lang_str_full(int id) {
    STUB_ABORT();
    return 0;
}
int whisper_lang_auto_detect_with_state(struct whisper_context* ctx, struct whisper_state* state, int offset_ms,
                                        int n_threads, float* lang_probs) {
    STUB_ABORT();
    return 0;
}
int whisper_n_len_from_state(struct whisper_state* state) {
    STUB_ABORT();
    return 0;
}
int whisper_has_speaker_tokens(struct whisper_context* ctx) {
    STUB_ABORT();
    return 0;
}
int whisper_model_n_vocab(struct whisper_context* ctx) {
    STUB_ABORT();
    return 0;
}
int whisper_model_n_audio_ctx(struct whisper_context* ctx) {
    STUB_ABORT();
    return 0;
}
int whisper_model_n_audio_state(struct whisper_context* ctx) {
    STUB_ABORT();
    return 0;
}
int whisper_model_n_audio_head(struct whisper_context* ctx) {
    STUB_ABORT();
    return 0;
}
int whisper_model_n_audio_layer(struct whisper_context* ctx) {
    STUB_ABORT();
    return 0;
}
int whisper_model_n_text_ctx(struct whisper_context* ctx) {
    STUB_ABORT();
    return 0;
}
int whisper_model_n_text_state(struct whisper_context* ctx) {
    STUB_ABORT();
    return 0;
}
int whisper_model_n_text_head(struct whisper_context* ctx) {
    STUB_ABORT();
    return 0;
}
int whisper_model_n_text_layer(struct whisper_context* ctx) {
    STUB_ABORT();
    return 0;
}
int whisper_model_n_mels(struct whisper_context* ctx) {
    STUB_ABORT();
    return 0;
}
int whisper_model_ftype(struct whisper_context* ctx) {
    STUB_ABORT();
    return 0;
}
int whisper_model_type(struct whisper_context* ctx) {
    STUB_ABORT();
    return 0;
}
float* whisper_get_logits_from_state(struct whisper_state* state) {
    STUB_ABORT();
    return 0;
}
const char* whisper_model_type_readable(struct whisper_context* ctx) {
    STUB_ABORT();
    return 0;
}
whisper_token whisper_token_nosp(struct whisper_context* ctx) {
    whisper_token _s = {0};
    STUB_ABORT();
    return _s;
}
struct whisper_timings* whisper_get_timings(struct whisper_context* ctx) {
    STUB_ABORT();
    return 0;
}
struct whisper_context_params whisper_context_default_params(void) {
    struct whisper_context_params _s = {0};
    STUB_ABORT();
    return _s;
}
struct whisper_full_params whisper_full_default_params(enum whisper_sampling_strategy strategy) {
    struct whisper_full_params _s = {0};
    _s.strategy = strategy;
    _s.greedy.best_of = 1;
    return _s;
}
struct whisper_context* whisper_init_from_file_with_params_no_state_by_ref(const char* path_model,
                                                                           struct whisper_context_params* params) {
    STUB_ABORT();
    return 0;
}
int whisper_full_n_segments_from_state(struct whisper_state* state) {
    STUB_ABORT();
    return 0;
}
int whisper_full_lang_id_from_state(struct whisper_state* state) {
    STUB_ABORT();
    return 0;
}
int64_t whisper_full_get_segment_t0_from_state(struct whisper_state* state, int i_segment) {
    STUB_ABORT();
    return 0;
}
int64_t whisper_full_get_segment_t1_from_state(struct whisper_state* state, int i_segment) {
    STUB_ABORT();
    return 0;
}
bool whisper_full_get_segment_speaker_turn_next(struct whisper_context* ctx, int i_segment) {
    STUB_ABORT();
    return 0;
}
bool whisper_full_get_segment_speaker_turn_next_from_state(struct whisper_state* state, int i_segment) {
    STUB_ABORT();
    return 0;
}
const char* whisper_full_get_segment_text_from_state(struct whisper_state* state, int i_segment) {
    STUB_ABORT();
    return 0;
}
int whisper_full_n_tokens(struct whisper_context* ctx, int i_segment) {
    if (ctx != rust_legacy_context() || i_segment != 0)
        STUB_ABORT();
    return probe_mode == 20 ? -1 : probe_mode == 21 ? 0 : probe_mode == 31 ? 1000001 : 2;
}
int whisper_full_n_tokens_from_state(struct whisper_state* state, int i_segment) {
    STUB_ABORT();
    return 0;
}
const char* whisper_full_get_token_text(struct whisper_context* ctx, int i_segment, int i_token) {
    if (ctx != rust_legacy_context() || i_segment != 0 || i_token < 0 || i_token >= 2)
        STUB_ABORT();
    if (probe_mode == 28)
        return NULL;
    if (probe_mode == 32) {
        memset(legacy_oversized_text, 'V', sizeof(legacy_oversized_text) - 1);
        legacy_oversized_text[sizeof(legacy_oversized_text) - 1] = 0;
        return legacy_oversized_text;
    }
    return i_token == 0 ? legacy_token_prefix : legacy_token_suffix;
}
const char* whisper_full_get_token_text_from_state(struct whisper_context* ctx, struct whisper_state* state,
                                                   int i_segment, int i_token) {
    STUB_ABORT();
    return 0;
}
whisper_token whisper_full_get_token_id(struct whisper_context* ctx, int i_segment, int i_token) {
    if (ctx != rust_legacy_context() || i_segment != 0 || i_token < 0 || i_token >= 2)
        STUB_ABORT();
    return 11 + i_token;
}
whisper_token whisper_full_get_token_id_from_state(struct whisper_state* state, int i_segment, int i_token) {
    whisper_token _s = {0};
    STUB_ABORT();
    return _s;
}
whisper_token_data whisper_full_get_token_data(struct whisper_context* ctx, int i_segment, int i_token) {
    if (ctx != rust_legacy_context() || i_segment != 0 || i_token < 0 || i_token >= 2)
        STUB_ABORT();
    whisper_token_data result = {0};
    result.id = 11 + i_token;
    result.t0 = i_token == 0 ? 100 : -1;
    result.t1 = i_token == 0 ? 120 : -1;
    result.p = i_token == 0 ? .75f : -1.f;
    result.t_dtw = probe_mode == 22 ? -1 : 100 + i_token;
    return result;
}
whisper_token_data whisper_full_get_token_data_from_state(struct whisper_state* state, int i_segment, int i_token) {
    whisper_token_data _s = {0};
    STUB_ABORT();
    return _s;
}
float whisper_full_get_token_p(struct whisper_context* ctx, int i_segment, int i_token) {
    return whisper_full_get_token_data(ctx, i_segment, i_token).p;
}
float whisper_full_get_token_p_from_state(struct whisper_state* state, int i_segment, int i_token) {
    STUB_ABORT();
    return 0;
}
int whisper_full_get_token_n_alts(struct whisper_context* ctx, int i_segment, int i_token) {
    STUB_ABORT();
    return 0;
}
int whisper_full_get_token_n_alts_from_state(struct whisper_state* state, int i_segment, int i_token) {
    STUB_ABORT();
    return 0;
}
whisper_token whisper_full_get_token_alt_id(struct whisper_context* ctx, int i_segment, int i_token, int i_alt) {
    whisper_token _s = {0};
    STUB_ABORT();
    return _s;
}
whisper_token whisper_full_get_token_alt_id_from_state(struct whisper_state* state, int i_segment, int i_token,
                                                       int i_alt) {
    whisper_token _s = {0};
    STUB_ABORT();
    return _s;
}
float whisper_full_get_token_alt_p(struct whisper_context* ctx, int i_segment, int i_token, int i_alt) {
    STUB_ABORT();
    return 0;
}
float whisper_full_get_token_alt_p_from_state(struct whisper_state* state, int i_segment, int i_token, int i_alt) {
    STUB_ABORT();
    return 0;
}
struct whisper_vad_params whisper_vad_default_params(void) {
    struct whisper_vad_params _s = {0};
    STUB_ABORT();
    return _s;
}
struct whisper_vad_context_params whisper_vad_default_context_params(void) {
    struct whisper_vad_context_params _s = {0};
    STUB_ABORT();
    return _s;
}
float whisper_full_get_segment_no_speech_prob_from_state(struct whisper_state* state, int i_segment) {
    STUB_ABORT();
    return 0;
}
void crispasr_chat_open_params_default(crispasr_chat_open_params* out) {
    STUB_ABORT();
}
void crispasr_chat_generate_params_default(crispasr_chat_generate_params* out) {
    STUB_ABORT();
}
crispasr_chat_session_t crispasr_chat_open(const char* model_path, const crispasr_chat_open_params* params,
                                           crispasr_chat_error* err) {
    crispasr_chat_session_t _s = {0};
    STUB_ABORT();
    return _s;
}
void crispasr_chat_close(crispasr_chat_session_t s) {
    STUB_ABORT();
}
int32_t crispasr_chat_reset(crispasr_chat_session_t s, crispasr_chat_error* err) {
    STUB_ABORT();
    return 0;
}
char* crispasr_chat_generate(crispasr_chat_session_t s, const crispasr_chat_message* messages, size_t n_messages,
                             const crispasr_chat_generate_params* params, crispasr_chat_error* err) {
    STUB_ABORT();
    return 0;
}
int32_t crispasr_chat_generate_stream(crispasr_chat_session_t s, const crispasr_chat_message* messages,
                                      size_t n_messages, const crispasr_chat_generate_params* params,
                                      crispasr_chat_on_token on_token, void* user, crispasr_chat_error* err) {
    STUB_ABORT();
    return 0;
}
void crispasr_chat_set_abort_callback(crispasr_chat_session_t s, crispasr_chat_abort_callback cb, void* user) {
    STUB_ABORT();
}
const char* crispasr_chat_template_name(crispasr_chat_session_t s) {
    STUB_ABORT();
    return 0;
}
int32_t crispasr_chat_n_ctx(crispasr_chat_session_t s) {
    STUB_ABORT();
    return 0;
}
int32_t crispasr_chat_count_tokens(crispasr_chat_session_t s, const crispasr_chat_message* messages, size_t n_messages,
                                   crispasr_chat_error* err) {
    STUB_ABORT();
    return 0;
}
size_t crispasr_chat_memory_estimate(const char* model_path, const crispasr_chat_open_params* params,
                                     crispasr_chat_error* err) {
    size_t _s = {0};
    STUB_ABORT();
    return _s;
}
void crispasr_chat_string_free(char* s) {
    STUB_ABORT();
}
const char* crispasr_chat_ai_disclosure_text(void) {
    STUB_ABORT();
    return 0;
}
