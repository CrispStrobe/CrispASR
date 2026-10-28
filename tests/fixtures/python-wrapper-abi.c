// Model-free ABI contract probe. Including the public header checks signatures.
#include "crispasr_session.h"
#include <stdlib.h>
#include <string.h>

// Integer-speaker contract: this fixture exposes exactly two speakers.
int crispasr_session_set_speaker_id(crispasr_session* s, int id) {
    if (!s)
        return -1;
    return id >= 0 && id < 2 ? 0 : -2;
}

static int mode, live, calls, invalid_frees;
static void* allocations[16];
static float buffer[32];
static int programs[2] = {24, 128};
static crispasr_segment_callback seg_cb;
static crispasr_token_callback tok_cb;
static crispasr_progress_callback progress_cb;
static void* progress_user;
static void* segment_user;
static void* token_user;
void probe_mode(int m) {
    mode = m;
}
int probe_live(void) {
    return live;
}
int probe_calls(void) {
    return calls;
}
void probe_mutate(void) {
    buffer[0] = 999;
}
static int run(const float* pcm, int n) {
    ++calls;
    if (mode == 2)
        return -1;
    if (n != 3 || pcm[0] != 1 || pcm[1] != 3 || pcm[2] != 5)
        return -9;
    for (int i = 0; i < 32; ++i)
        buffer[i] = (float)i;
    return mode == 1 ? 0 : 2;
}
int crispasr_session_piano(crispasr_session* s, const float* pcm, int n) {
    return run(pcm, n);
}
const float* crispasr_session_piano_notes(crispasr_session* s, int* n) {
    *n = mode == 1 ? 0 : 2;
    return *n ? buffer : NULL;
}
const int* crispasr_session_piano_note_programs(crispasr_session* s, int* n) {
    *n = mode == 1 ? 0 : 2;
    return *n ? programs : NULL;
}
int crispasr_session_piano_sample_rate(crispasr_session* s) {
    return 16000;
}
int crispasr_session_chords(crispasr_session* s, const float* pcm, int n, int rate) {
    return rate > 0 ? run(pcm, n) : -1;
}
const float* crispasr_session_chords_spans(crispasr_session* s, int* n) {
    *n = mode == 1 ? 0 : 2;
    return *n ? buffer : NULL;
}
const char* crispasr_session_chords_span_name(crispasr_session* s, int i) {
    return i ? "Am" : "C";
}
int crispasr_session_chords_vocab_size(crispasr_session* s) {
    return 170;
}
int crispasr_session_beats(crispasr_session* s, const float* pcm, int n, int rate) {
    return rate == 22050 ? run(pcm, n) : -1;
}
const float* crispasr_session_beats_events(crispasr_session* s, int* n) {
    *n = mode == 1 ? 0 : 2;
    return *n ? buffer : NULL;
}
int crispasr_session_beats_sample_rate(crispasr_session* s) {
    return 22050;
}
float crispasr_session_beats_tempo_bpm(crispasr_session* s) {
    return 123.5f;
}
int crispasr_session_tab(crispasr_session* s, const float* pcm, int n, int rate) {
    return run(pcm, n);
}
const float* crispasr_session_tab_emissions(crispasr_session* s, int* n, int* strings, int* classes) {
    *n = mode == 1 ? 0 : 2;
    *strings = 2;
    *classes = 3;
    return *n ? buffer : NULL;
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
    if (n != 3 || content[1] != 1 || f0[2] != 5 || id)
        return -1;
    if (zp && zp[5] != 5)
        return -1;
    if (sine && sine[11] != 11)
        return -1;
    for (int i = 0; i < 6; i++)
        buffer[i] = (float)i;
    return 6;
}
const float* crispasr_session_convert_audio(crispasr_session* s, int* n) {
    *n = 6;
    return buffer;
}
int crispasr_session_set_voice_samples(crispasr_session* s, const float* pcm, int32_t n, int32_t rate,
                                       const char* ref) {
    return rate == 24000 && ref && !strcmp(ref, "hello") ? (run(pcm, n) < 0 ? -1 : 0) : -1;
}
int crispasr_session_set_tts_cfg_scale(crispasr_session* s, float scale) {
    return scale == 1.5f ? 0 : -2;
}
int crispasr_session_tada_set_makeref_models(crispasr_session* s, const char* encoder, const char* aligner) {
    return !encoder && !aligner ? 0 : -1;
}
int probe_invalid_frees(void) {
    return invalid_frees;
}
static void* allocate(size_t n) {
    for (int i = 0; i < 16; i++) {
        if (!allocations[i]) {
            allocations[i] = malloc(n);
            ++live;
            return allocations[i];
        }
    }
    abort();
}
static char* alloc_text(const char* text) {
    size_t n = strlen(text) + 1;
    char* p = allocate(n);
    memcpy(p, text, n);
    return p;
}
static void free_text(const void* p) {
    if (!p)
        return;
    for (int i = 0; i < 16; i++) {
        if (allocations[i] == p) {
            free(allocations[i]);
            allocations[i] = NULL;
            --live;
            return;
        }
    }
    ++invalid_frees;
}

#ifndef CRISPASR_PROBE_REAL_TEXT_PROCESSORS
void* crispasr_truecase_init(const char* model) {
    return (void*)1;
}
const char* crispasr_truecase_process(void* ctx, const char* text) {
    return alloc_text(mode == 1 ? "" : "Grüße");
}
void crispasr_truecase_free_text(const char* p) {
    free_text(p);
}
void crispasr_truecase_free(void* ctx) {}
void* crispasr_pcs_init(const char* model) {
    return (void*)1;
}
const char* crispasr_pcs_process(void* ctx, const char* text) {
    return alloc_text("Hello.");
}
void crispasr_pcs_free_text(const char* p) {
    free_text(p);
}
void crispasr_pcs_free(void* ctx) {}
void* crispasr_punc_init(const char* model) {
    return (void*)1;
}
const char* crispasr_punc_process(void* ctx, const char* text) {
    return mode == 2 ? NULL : alloc_text(mode == 1 ? "" : "Hello!");
}
void crispasr_punc_free_text(const char* p) {
    free_text(p);
}
void crispasr_punc_free(void* ctx) {}
#endif
char* crispasr_session_translate_text(crispasr_session* s, const char* text, const char* src, const char* tgt,
                                      int max) {
    return mode == 2 ? NULL : alloc_text(mode == 1 ? "" : "こんにちは");
}
void crispasr_session_translate_text_free(char* text) {
    free_text(text);
}
const char* crispasr_session_last_synth_error(crispasr_session* s) {
    return "probe synthesis failure";
}
const char* crispasr_session_disclaimer_text(void) {
    return "AI generated";
}
float* crispasr_session_get_disclaimer_pcm(crispasr_session* s, int* n) {
    *n = mode == 1 ? 0 : 2;
    float* p = allocate(8);
    p[0] = 0.25f;
    p[1] = 0.5f;
    return p;
}
void crispasr_pcm_free(float* p) {
    free_text(p);
}
float* crispasr_session_synthesize(crispasr_session* s, const char* text, int* n) {
    return crispasr_session_get_disclaimer_pcm(s, n);
}
float* crispasr_session_synthesize_raw(crispasr_session* s, const char* text, int* n) {
    return crispasr_session_get_disclaimer_pcm(s, n);
}
int crispasr_session_synthesize_streaming(crispasr_session* s, const char* text, crispasr_pcm_stream_cb cb, void* ud) {
    float pcm[2] = {0.25f, 0.5f};
    cb(mode == 2 ? NULL : pcm, mode == 2 ? 0 : 2, 0, ud);
    pcm[0] = 0.75f;
    cb(pcm, 2, 1, ud);
    return 0;
}
int crispasr_session_accept_marking_responsibility(crispasr_session* s, const char* attestation) {
    return mode == 2 ? -1 : 0;
}
void crispasr_session_set_segment_callback(crispasr_session* s, crispasr_segment_callback cb, void* ud) {
    seg_cb = cb;
    segment_user = ud;
}
void crispasr_session_set_token_callback(crispasr_session* s, crispasr_token_callback cb, void* ud) {
    tok_cb = cb;
    token_user = ud;
}
void crispasr_session_set_progress_callback(crispasr_session* s, crispasr_progress_callback cb, void* ud) {
    progress_cb = cb;
    progress_user = ud;
}
void probe_callbacks(void) {
    if (progress_cb)
        progress_cb(3, 6, progress_user);
    if (seg_cb)
        seg_cb("日本語", 123, 456, 7, segment_user);
    if (tok_cb)
        tok_cb("\xe6\x97", 9, token_user);
}
static char drained_tokens[32];
void probe_poison_drained_tokens(void) {
    memset(drained_tokens, 0, sizeof(drained_tokens));
}
const char* crispasr_drain_streamed_tokens(int* n) {
    static const char data[] = "\xe6\x97\0\xa5\0\0end\0";
    memcpy(drained_tokens, data, sizeof(data));
    *n = mode == 8 ? -1 : mode == 9 ? 1 : mode == 10 || mode == 11 ? 0 : 4;
    return mode == 9 || mode == 10 ? NULL : drained_tokens;
}
int crispasr_backend_caps_list_abi(char* out, int32_t cap) {
    const char* data = "foo\tasr,tts\nbar\t\n";
    int n = (int)strlen(data) + 1;
    if (!out || cap < n)
        return -n;
    memcpy(out, data, n);
    return n - 1;
}
int crispasr_backend_caps_abi(const char* backend, char* out, int32_t cap) {
    if (strcmp(backend, "foo"))
        return -3;
    if (cap < 8)
        return -2;
    strcpy(out, "asr,tts");
    return 7;
}
const char* crispasr_c_api_version(void) {
    return "probe";
}
const char* crispasr_dart_helpers_version(void) {
    return "probe-helpers";
}
void crispasr_set_gpu_backend(const char* name) {
    ++calls;
}
#ifdef CRISPASR_PROBE_RUBY_STREAM_OWNERS
extern int probe_streams_for_owner(crispasr_session*);
static int parent_close_violations;
int probe_parent_close_violations(void) {
    return parent_close_violations;
}
#endif
void crispasr_session_close(crispasr_session* s) {
#ifdef CRISPASR_PROBE_RUBY_STREAM_OWNERS
    if (probe_streams_for_owner(s) > 0)
        ++parent_close_violations;
#endif
    ++calls;
}

#ifdef CRISPASR_PROBE_LITERAL_OPEN_LAYOUT
#include "runtime-open-layout.h"
#else
struct crispasr_open_params_v1 {
    int abi_version, n_threads, use_gpu, verbosity, flash_attn, n_gpu_layers, reserved[6];
};
#endif

crispasr_session* crispasr_session_open_with_params(const char* model, const char* backend,
                                                    const crispasr_open_params_v1* p) {
#ifdef CRISPASR_PROBE_WASM_OPEN
    if (mode == 2 || !model || strcmp(model, "model"))
        return NULL;
    int expected_threads = 1;
    if (mode == 7) {
        if (backend || p->abi_version != 2 || p->n_threads != 1 || p->use_gpu != 1 || p->verbosity != 0 ||
            p->flash_attn != 1 || p->n_gpu_layers != -1)
            return NULL;
        for (int i = 0; i < 6; ++i)
            if (p->reserved[i])
                return NULL;
        return (crispasr_session*)(uintptr_t)1;
    }
    if (!backend || strcmp(backend, "foo"))
        return NULL;
#else
    int expected_threads = 7;
#endif
    if (p->abi_version != 2 || p->n_threads != expected_threads || p->use_gpu != 0 || p->verbosity != 2 ||
        p->flash_attn != 0 || p->n_gpu_layers != 3)
        return NULL;
    for (int i = 0; i < 6; i++)
        if (p->reserved[i])
            return NULL;
    return (crispasr_session*)(uintptr_t)0x100000001ULL;
}
const char* crispasr_session_backend(crispasr_session* s) {
    return "probe";
}
#ifndef CRISPASR_PROBE_RUBY_STREAM_OWNERS
int crispasr_stream_feed(crispasr_stream* s, const float* pcm, int n) {
    return run(pcm, n) < 0 ? -1 : 1;
}
int crispasr_stream_flush(crispasr_stream* s) {
    return 0;
}
void crispasr_stream_close(crispasr_stream* s) {
    ++calls;
}
#endif


float* crispasr_session_speech_to_speech(crispasr_session* s, const float* pcm, int n, char** text, int* out_n) {
    if (run(pcm, n) < 0 && mode != 2)
        return NULL;
    *text = alloc_text(mode == 2 ? "\xff" : "hello");
    return crispasr_session_get_disclaimer_pcm(s, out_n);
}

int crispasr_session_set_source_language(crispasr_session* s, const char* language) {
    return language && strcmp(language, "de") == 0 ? 0 : -1;
}
