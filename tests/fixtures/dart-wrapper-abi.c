// Dart wrapper ABI probe - model-free fixture for FFI behavior tests.
// Includes public headers to validate signatures; compile with -I include.
// Guard token symbols for old-library test via -DCRISPASR_PROBE_OLD_TOKENS.
#include "crispasr_session.h"
#include "crispasr.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

static int g_mode = 0;
static int g_live = 0;
static int g_result_frees = 0;
static int g_invalid_frees = 0;
static int g_calls = 0;
static int stream_opens, stream_closes, bad_parent_closes;
int probe_stream_closes(void) {
    return stream_closes;
}
int probe_bad_parent_closes(void) {
    return bad_parent_closes;
}
static float g_buffer[32];
static int g_programs[2] = {24, 128};
static char g_token_head[3] = {(char)0xe4, 0, 0};
static char g_token_tail[3] = {(char)0xb8, (char)0x96, 0};

void probe_mode(int m) {
    g_mode = m;
}
int probe_live(void) {
    return g_live;
}
int probe_result_frees(void) {
    return g_result_frees;
}
int probe_invalid_frees(void) {
    return g_invalid_frees;
}
int probe_calls(void) {
    return g_calls;
}
void probe_mutate(void) {
    g_buffer[0] = 999;
    g_token_head[0] = 'X';
}

// Small helper for allocation tracking (not required for Dart but useful)
static void* tracked_malloc(size_t n) {
    void* p = malloc(n);
    if (p)
        g_live++;
    return p;
}
static void tracked_free(void* p) {
    if (p) {
        free(p);
        g_live--;
    }
}

// ---------------------------------------------------------------------------
// Session lifecycle
// ---------------------------------------------------------------------------
struct crispasr_open_params_v1 {
    int abi_version, n_threads, use_gpu, verbosity, flash_attn, n_gpu_layers, reserved[6];
};

crispasr_session* crispasr_session_open(const char* path, int n_threads) {
    (void)path;
    (void)n_threads;
    return (crispasr_session*)0x100000001ULL;
}
crispasr_session* crispasr_session_open_explicit(const char* path, const char* backend, int n_threads) {
    (void)path;
    (void)backend;
    (void)n_threads;
    return (crispasr_session*)0x100000001ULL;
}
crispasr_session* crispasr_session_open_with_params(const char* model, const char* backend,
                                                    const crispasr_open_params_v1* p) {
    (void)model;
    (void)backend;
    if (!p)
        return (crispasr_session*)0x100000001ULL;
    if (p->abi_version != 2 || p->n_threads != 7 || p->use_gpu != 0 || p->verbosity != 2 || p->flash_attn != 0 ||
        p->n_gpu_layers != 3)
        return NULL;
    for (int i = 0; i < 6; i++)
        if (p->reserved[i])
            return NULL;
    return (crispasr_session*)0x100000001ULL;
}
const char* crispasr_session_backend(crispasr_session* s) {
    (void)s;
    return "probe";
}
static void clear_probe_callbacks(void);
void crispasr_session_close(crispasr_session* s) {
    (void)s;
    if (stream_opens != stream_closes)
        ++bad_parent_closes;
    clear_probe_callbacks();
    g_calls++;
}
int crispasr_session_input_sample_rate(crispasr_session* s) {
    (void)s;
    return 16000;
}
int crispasr_session_output_sample_rate(crispasr_session* s) {
    (void)s;
    return 24000;
}
int crispasr_session_input_channels(crispasr_session* s) {
    (void)s;
    return 1;
}
int crispasr_session_output_channels(crispasr_session* s) {
    (void)s;
    return 1;
}
int crispasr_session_set_pcm_sample_rate(crispasr_session* s, int r) {
    (void)s;
    (void)r;
    return 0;
}
int crispasr_session_set_parakeet_att_context(crispasr_session* s, int l, int r) {
    (void)s;
    (void)l;
    (void)r;
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
void crispasr_session_set_tts_pad_silence_ms(crispasr_session* s, int ms) {
    (void)s;
    (void)ms;
}

// ---------------------------------------------------------------------------
// Result metadata (shared with wrapper-session-metadata approach)
// ---------------------------------------------------------------------------
static const char* seg_text = "\xe4\xb8\x96"; // "世" in UTF-8 via bytes, but probe uses raw split tokens
static int64_t seg_t0 = 100, seg_t1 = 200;
static const char* seg_speaker = "(Speaker 1) ";
static const char* word_text = "\xe4\xb8\x96";
// mode 3: segment, word and alt text cut after two of the three bytes of "世",
// like a byte-level BPE decode capped inside a multibyte character
static const char* cut_text = "\xe4\xb8";
static int64_t word_t0 = -1, word_t1 = -1;
static float word_p = 0.9f;
static int64_t token_t0s[2] = {-1, -1};
static int64_t token_t1s[2] = {-1, -1};
static float token_ps[2] = {0.5f, 0.75f};

static int* make_result(int has_data) {
    int* p = (int*)tracked_malloc(sizeof(int));
    if (p)
        *p = has_data;
    return p;
}

int crispasr_session_result_n_segments(crispasr_session_result* r) {
    if (!r)
        return 0;
    return (*(int*)r) ? 1 : 0;
}
const char* crispasr_session_result_segment_text(crispasr_session_result* r, int i) {
    if (!r || i != 0 || !(*(int*)r))
        return "";
    return g_mode == 3 ? cut_text : seg_text;
}
int64_t crispasr_session_result_segment_t0(crispasr_session_result* r, int i) {
    (void)i;
    if (!r || !(*(int*)r))
        return 0;
    return seg_t0;
}
int64_t crispasr_session_result_segment_t1(crispasr_session_result* r, int i) {
    (void)i;
    if (!r || !(*(int*)r))
        return 0;
    return seg_t1;
}
const char* crispasr_session_result_segment_speaker(crispasr_session_result* r, int i) {
    (void)i;
    if (!r || !(*(int*)r))
        return "";
    return seg_speaker;
}
float crispasr_session_result_segment_no_speech_prob(crispasr_session_result* r, int i) {
    (void)r;
    (void)i;
    return -1.0f;
}
int crispasr_session_result_n_words(crispasr_session_result* r, int i) {
    (void)i;
    if (!r || !(*(int*)r))
        return 0;
    return 1;
}
const char* crispasr_session_result_word_text(crispasr_session_result* r, int i, int w) {
    (void)i;
    (void)w;
    if (!r || !(*(int*)r))
        return "";
    return g_mode == 3 ? cut_text : word_text;
}
int64_t crispasr_session_result_word_t0(crispasr_session_result* r, int i, int w) {
    (void)i;
    (void)w;
    if (!r)
        return -1;
    if (!(*(int*)r))
        return -1;
    return word_t0;
}
int64_t crispasr_session_result_word_t1(crispasr_session_result* r, int i, int w) {
    (void)i;
    (void)w;
    if (!r)
        return -1;
    if (!(*(int*)r))
        return -1;
    return word_t1;
}
float crispasr_session_result_word_p(crispasr_session_result* r, int i, int w) {
    (void)i;
    (void)w;
    if (!r || !(*(int*)r))
        return -1;
    return word_p;
}
int crispasr_session_result_word_n_alts(crispasr_session_result* r, int i, int w) {
    (void)i;
    (void)w;
    if (!r || !(*(int*)r))
        return 0;
    return 1;
}
const char* crispasr_session_result_word_alt_text(crispasr_session_result* r, int i, int w, int a) {
    (void)r;
    (void)i;
    (void)w;
    (void)a;
    return g_mode == 3 ? cut_text : "alternative";
}
float crispasr_session_result_word_alt_p(crispasr_session_result* r, int i, int w, int a) {
    (void)r;
    (void)i;
    (void)w;
    (void)a;
    return 0.25f;
}

#ifndef CRISPASR_PROBE_OLD_TOKENS
int crispasr_session_result_n_tokens(crispasr_session_result* r, int i) {
    (void)i;
    if (!r || !(*(int*)r))
        return 0;
    return 2;
}
const char* crispasr_session_result_token_text(crispasr_session_result* r, int i, int t) {
    (void)i;
    if (!r || !(*(int*)r))
        return "";
    if (t < 0 || t >= 2)
        return "";
    return t ? g_token_tail : g_token_head;
}
int64_t crispasr_session_result_token_t0(crispasr_session_result* r, int i, int t) {
    (void)i;
    if (!r || !(*(int*)r))
        return -1;
    if (t < 0 || t >= 2)
        return -1;
    return token_t0s[t];
}
int64_t crispasr_session_result_token_t1(crispasr_session_result* r, int i, int t) {
    (void)i;
    if (!r || !(*(int*)r))
        return -1;
    if (t < 0 || t >= 2)
        return -1;
    return token_t1s[t];
}
float crispasr_session_result_token_p(crispasr_session_result* r, int i, int t) {
    (void)i;
    if (!r || !(*(int*)r))
        return -1;
    if (t < 0 || t >= 2)
        return -1;
    return token_ps[t];
}
#endif

int crispasr_session_result_n_logit_frames(crispasr_session_result* r) {
    (void)r;
    return 0;
}
int crispasr_session_result_n_logit_vocab(crispasr_session_result* r) {
    (void)r;
    return 0;
}
const float* crispasr_session_result_logits(crispasr_session_result* r) {
    (void)r;
    return NULL;
}
void crispasr_session_result_free(crispasr_session_result* r) {
    if (!r)
        return;
    // Mutate token_head to detect use-after-free
    g_token_head[0] = 'X';
    g_buffer[0] = 999;
    if (r) {
        tracked_free(r);
        g_result_frees++;
    }
}

// ---------------------------------------------------------------------------
// Transcribe entry points
// ---------------------------------------------------------------------------
static int check_pcm(const float* pcm, int n) {
    g_calls++;
    if (g_mode == 2)
        return -1;
    if (n != 3 || !pcm || pcm[0] != 1 || pcm[1] != 3 || pcm[2] != 5)
        return -9;
    for (int i = 0; i < 32; i++)
        g_buffer[i] = (float)i;
    g_token_head[0] = (char)0xe4; // restore before each transcribe
    return g_mode == 1 ? 0 : 1;
}
static void fire_probe_callbacks(void);
crispasr_session_result* crispasr_session_transcribe(crispasr_session* s, const float* pcm, int n) {
    (void)s;
    int rc = check_pcm(pcm, n);
    if (rc < 0)
        return NULL;
    // Fire progress/segment/token callbacks as native would during decode
    fire_probe_callbacks();
    return (crispasr_session_result*)make_result(rc);
}
crispasr_session_result* crispasr_session_transcribe_lang(crispasr_session* s, const float* pcm, int n,
                                                          const char* lang) {
    (void)lang;
    return crispasr_session_transcribe(s, pcm, n);
}
crispasr_session_result* crispasr_session_transcribe_chunked(crispasr_session* s, const float* pcm, int n, int cs,
                                                             int os) {
    (void)cs;
    (void)os;
    return crispasr_session_transcribe(s, pcm, n);
}
crispasr_session_result* crispasr_session_transcribe_chunked_lang(crispasr_session* s, const float* pcm, int n, int cs,
                                                                  int os, const char* lang) {
    (void)cs;
    (void)os;
    (void)lang;
    return crispasr_session_transcribe(s, pcm, n);
}
crispasr_session_result* crispasr_session_transcribe_vad(crispasr_session* s, const float* pcm, int n, int sr,
                                                         const char* vad, const crispasr_vad_abi_opts* opts) {
    (void)sr;
    (void)vad;
    (void)opts;
    return crispasr_session_transcribe(s, pcm, n);
}
crispasr_session_result* crispasr_session_transcribe_vad_lang(crispasr_session* s, const float* pcm, int n, int sr,
                                                              const char* vad, const crispasr_vad_abi_opts* opts,
                                                              const char* lang) {
    (void)sr;
    (void)vad;
    (void)opts;
    (void)lang;
    return crispasr_session_transcribe(s, pcm, n);
}

// ---------------------------------------------------------------------------
// Task backends: piano, chords, beats, tab, convert
// ---------------------------------------------------------------------------
int crispasr_session_piano(crispasr_session* s, const float* pcm, int n) {
    (void)s;
    int rc = check_pcm(pcm, n);
    if (rc < 0)
        return rc;
    return rc == 0 ? 0 : 2;
}
const float* crispasr_session_piano_notes(crispasr_session* s, int* n) {
    (void)s;
    *n = g_mode == 1 ? 0 : 2;
    return *n ? g_buffer : NULL;
}
const int* crispasr_session_piano_note_programs(crispasr_session* s, int* n) {
    (void)s;
    *n = g_mode == 1 ? 0 : 2;
    return *n ? g_programs : NULL;
}
int crispasr_session_piano_sample_rate(crispasr_session* s) {
    (void)s;
    return 16000;
}

int crispasr_session_chords(crispasr_session* s, const float* pcm, int n, int rate) {
    (void)s;
    if (rate <= 0)
        return -1;
    int rc = check_pcm(pcm, n);
    if (rc < 0)
        return rc;
    return rc == 0 ? 0 : 2;
}
const float* crispasr_session_chords_spans(crispasr_session* s, int* n) {
    (void)s;
    *n = g_mode == 1 ? 0 : 2;
    return *n ? g_buffer : NULL;
}
const char* crispasr_session_chords_span_name(crispasr_session* s, int i) {
    (void)s;
    return i ? "Am" : "C";
}
int crispasr_session_chords_vocab_size(crispasr_session* s) {
    (void)s;
    return 170;
}

int crispasr_session_beats(crispasr_session* s, const float* pcm, int n, int rate) {
    (void)s;
    if (rate != 22050)
        return -1;
    int rc = check_pcm(pcm, n);
    if (rc < 0)
        return rc;
    return rc == 0 ? 0 : 2;
}
const float* crispasr_session_beats_events(crispasr_session* s, int* n) {
    (void)s;
    *n = g_mode == 1 ? 0 : 2;
    return *n ? g_buffer : NULL;
}
int crispasr_session_beats_sample_rate(crispasr_session* s) {
    (void)s;
    return 22050;
}
float crispasr_session_beats_tempo_bpm(crispasr_session* s) {
    (void)s;
    return 123.5f;
}

int crispasr_session_tab(crispasr_session* s, const float* pcm, int n, int rate) {
    (void)s;
    (void)rate;
    if (n != 3 || !pcm || pcm[0] != 1)
        return -1;
    for (int i = 0; i < 32; i++)
        g_buffer[i] = (float)i;
    return g_mode == 2 ? -1 : (g_mode == 1 ? 0 : 2);
}
const float* crispasr_session_tab_emissions(crispasr_session* s, int* n, int* strings, int* classes) {
    (void)s;
    *n = g_mode == 1 ? 0 : 2;
    *strings = 2;
    *classes = 3;
    return *n ? g_buffer : NULL;
}
int crispasr_session_tab_silent_class(crispasr_session* s) {
    (void)s;
    return 2;
}
float crispasr_session_tab_frame_period(crispasr_session* s) {
    (void)s;
    return 0.02f;
}
int crispasr_session_tab_string_open_midi(crispasr_session* s, int i) {
    (void)s;
    return 40 + i * 5;
}

int crispasr_session_convert_content_dim(crispasr_session* s) {
    (void)s;
    return 2;
}
int crispasr_session_convert_n_speakers(crispasr_session* s) {
    (void)s;
    return 1;
}
int crispasr_session_convert_sample_rate(crispasr_session* s) {
    (void)s;
    return 40000;
}
int crispasr_session_convert_inter_channels(crispasr_session* s) {
    (void)s;
    return 2;
}
int crispasr_session_convert_upsample_product(crispasr_session* s) {
    (void)s;
    return 4;
}
int crispasr_session_convert(crispasr_session* s, const float* content, int n, const float* f0, int id, const float* zp,
                             const float* sine) {
    (void)s;
    if (n != 3 || !content || content[1] != 1 || !f0 || f0[2] != 5 || id)
        return -1;
    if (zp && zp[5] != 5)
        return -1;
    if (sine && sine[11] != 11)
        return -1;
    for (int i = 0; i < 6; i++)
        g_buffer[i] = (float)i;
    return 6;
}
const float* crispasr_session_convert_audio(crispasr_session* s, int* n) {
    (void)s;
    *n = 6;
    return g_buffer;
}

// ---------------------------------------------------------------------------
// TTS & voice
// ---------------------------------------------------------------------------
int crispasr_session_set_voice_samples(crispasr_session* s, const float* pcm, int32_t n, int32_t rate,
                                       const char* ref) {
    (void)s;
    if (rate != 24000)
        return -1;
    if (ref && strcmp(ref, "hello") == 0)
        return g_mode == 2 ? -1 : 0;
    return -1;
}
int crispasr_session_set_tts_cfg_scale(crispasr_session* s, float scale) {
    (void)s;
    return scale == 1.5f ? 0 : -2;
}
int crispasr_session_tada_set_makeref_models(crispasr_session* s, const char* e, const char* a) {
    (void)s;
    return !e && !a ? 0 : -1;
}
const char* crispasr_session_disclaimer_text(void) {
    return "AI generated";
}
float* crispasr_session_get_disclaimer_pcm(crispasr_session* s, int* n) {
    (void)s;
    *n = g_mode == 1 ? 0 : 2;
    float* p = (float*)tracked_malloc(8);
    if (p) {
        p[0] = 0.25f;
        p[1] = 0.5f;
    }
    return p;
}
void crispasr_pcm_free(float* p) {
    tracked_free(p);
}
float* crispasr_session_synthesize(crispasr_session* s, const char* text, int* n) {
    (void)text;
    return crispasr_session_get_disclaimer_pcm(s, n);
}
float* crispasr_session_synthesize_raw(crispasr_session* s, const char* text, int* n) {
    (void)text;
    return crispasr_session_get_disclaimer_pcm(s, n);
}
int crispasr_session_synthesize_streaming(crispasr_session* s, const char* text, crispasr_pcm_stream_cb cb, void* ud) {
    (void)s;
    (void)text;
    float pcm1[2] = {0.25f, 0.5f};
    float pcm2[2] = {0.75f, 1.0f};
    if (g_mode == 2) {
        cb(NULL, 0, 0, ud);
        return 0;
    }
    cb(pcm1, 2, 0, ud);
    cb(pcm2, 2, 1, ud);
    return 0;
}
int crispasr_session_accept_marking_responsibility(crispasr_session* s, const char* a) {
    (void)s;
    (void)a;
    return g_mode == 2 ? -1 : 0;
}
const char* crispasr_session_last_synth_error(crispasr_session* s) {
    (void)s;
    return "probe synthesis failure";
}

// ---------------------------------------------------------------------------
// Callback registrations (for exception & reentry tests)
// ---------------------------------------------------------------------------
static crispasr_progress_callback g_prog_cb = NULL;
static void* g_prog_ud = NULL;
static crispasr_segment_callback g_seg_cb = NULL;
static void* g_seg_ud = NULL;
static crispasr_token_callback g_tok_cb = NULL;
static void* g_tok_ud = NULL;

void crispasr_session_set_progress_callback(crispasr_session* s, crispasr_progress_callback cb, void* ud) {
    (void)s;
    g_prog_cb = cb;
    g_prog_ud = ud;
}
void crispasr_session_set_segment_callback(crispasr_session* s, crispasr_segment_callback cb, void* ud) {
    (void)s;
    g_seg_cb = cb;
    g_seg_ud = ud;
}
void crispasr_session_set_token_callback(crispasr_session* s, crispasr_token_callback cb, void* ud) {
    (void)s;
    g_tok_cb = cb;
    g_tok_ud = ud;
}

// Called from result creation to fire callbacks (simulates transcription firing)
static void clear_probe_callbacks(void) {
    g_prog_cb = NULL;
    g_seg_cb = NULL;
    g_tok_cb = NULL;
}
static void fire_probe_callbacks(void) {
    if (g_prog_cb)
        g_prog_cb(3, 6, g_prog_ud);
    if (g_seg_cb)
        g_seg_cb("segment", 123, 456, 7, g_seg_ud);
    if (g_tok_cb)
        g_tok_cb("\xe6\x97", 9, g_tok_ud);
}

// ---------------------------------------------------------------------------
// Top-level helpers (version, caps, audio, etc.)
// ---------------------------------------------------------------------------
const char* crispasr_c_api_version(void) {
    return "0.8.99-probe";
}
const char* crispasr_dart_helpers_version(void) {
    return "0.8.99-probe";
}
void crispasr_set_gpu_backend(const char* name) {
    (void)name;
    g_calls++;
}
void crispasr_free_aligner_cache(void) {
    g_calls++;
}
int crispasr_detect_backend_from_gguf(const char* path, char* out, int cap) {
    (void)path;
    if (cap > 4) {
        strcpy(out, "probe");
        return 5;
    }
    return -5;
}
int crispasr_detect_backends_from_gguf(const char* path, char* out, int cap) {
    (void)path;
    const char* s = "probe";
    int n = strlen(s) + 1;
    if (!out || cap < n)
        return -n;
    strcpy(out, s);
    return n - 1;
}
int crispasr_session_available_backends(char* out, int cap) {
    const char* s = "probe,whisper";
    int n = strlen(s) + 1;
    if (!out || cap < n)
        return -n;
    strcpy(out, s);
    return n - 1;
}
int crispasr_session_detected_language(crispasr_session* s, char* buf, int cap) {
    (void)s;
    if (cap > 3) {
        strcpy(buf, "en");
        return 2;
    }
    return -1;
}
int crispasr_audio_load(const char* p, float** out, int* ns, int* sr) {
    (void)p;
    (void)out;
    (void)ns;
    (void)sr;
    return -1;
}
int crispasr_audio_load_at_rate(const char* p, int r, float** out, int* ns, int* sr) {
    (void)p;
    (void)r;
    (void)out;
    (void)ns;
    (void)sr;
    return -1;
}
int crispasr_audio_load_stereo(const char* p, float** l, float** rr, int* ns, int* sr, int* ch) {
    (void)p;
    (void)l;
    (void)rr;
    (void)ns;
    (void)sr;
    (void)ch;
    return -1;
}
int crispasr_audio_resample(const float* in, int n, int sr, int tr, float** out, int* ns) {
    (void)in;
    (void)n;
    (void)sr;
    (void)tr;
    (void)out;
    (void)ns;
    return -1;
}
void crispasr_audio_free(float* p) {
    tracked_free(p);
}
int crispasr_backend_caps_abi(const char* backend, char* out, int cap) {
    if (!backend || strcmp(backend, "probe"))
        return -3;
    const char* s = "asr,tts";
    if (cap < 8)
        return -8;
    strcpy(out, s);
    return 7;
}
int crispasr_backend_caps_list_abi(char* out, int cap) {
    const char* s = "probe\tasr,tts\n";
    int n = strlen(s) + 1;
    if (!out || cap < n)
        return -n;
    strcpy(out, s);
    return n - 1;
}
int crispasr_lcs_dedup_prefix_count(const int32_t* a, int na, const int32_t* b, int nb, int m) {
    (void)a;
    (void)na;
    (void)b;
    (void)nb;
    (void)m;
    return 0;
}
float crispasr_watermark_detect(const float* p, int n) {
    (void)p;
    (void)n;
    return 0.0f;
}
void crispasr_watermark_embed(float* p, int n, float a) {
    (void)p;
    (void)n;
    (void)a;
}
int crispasr_watermark_load_model(const char* p) {
    (void)p;
    return -1;
}
unsigned char* crispasr_c2pa_sign(const unsigned char* d, size_t l, const char* f, const char* c, const char* k,
                                  size_t* o) {
    (void)d;
    (void)l;
    (void)f;
    (void)c;
    (void)k;
    (void)o;
    return NULL;
}
void crispasr_c2pa_free(unsigned char* p) {
    tracked_free(p);
}
unsigned char* crispasr_pcm_to_wav(const float* pcm, int n, int sr, size_t* o) {
    (void)pcm;
    (void)n;
    (void)sr;
    (void)o;
    return NULL;
}
void crispasr_ctx_params_set_dtw(struct whisper_context_params* p, bool e, int pr, int top) {
    (void)p;
    (void)e;
    (void)pr;
    (void)top;
}
int64_t crispasr_token_dtw_t(struct whisper_context* ctx, int s, int t) {
    (void)ctx;
    (void)s;
    (void)t;
    return 0;
}
int crispasr_transcribe_parallel(struct whisper_context* ctx, struct whisper_full_params params, const float* samples,
                                 int n, int npr) {
    (void)ctx;
    (void)params;
    (void)samples;
    (void)n;
    (void)npr;
    return 0;
}
int crispasr_get_progress(void) {
    return -1;
}
void crispasr_reset_progress(void) {}
int crispasr_get_streamed_segment_count(void) {
    return 0;
}
crispasr_session_result* crispasr_drain_streamed_segments(void) {
    g_token_head[0] = (char)0xe4;
    return (crispasr_session_result*)make_result(1);
}
void crispasr_reset_streamed_segments(void) {}
int crispasr_get_streamed_token_count(void) {
    return 0;
}
const char* crispasr_drain_streamed_tokens(int* n) {
    static const char pieces[] = {(char)0xe4, 0, (char)0xb8, (char)0x96, 0};
    if (n)
        *n = 2;
    return pieces;
}
void crispasr_reset_streamed_tokens(void) {}

// Session-level stubs to satisfy Dart lookups
int crispasr_session_set_source_language(crispasr_session* s, const char* l) {
    (void)s;
    return l && !strcmp(l, "de") ? 0 : -1;
}
int crispasr_session_set_target_language(crispasr_session* s, const char* l) {
    (void)s;
    (void)l;
    return 0;
}
int crispasr_session_set_tts_reference_language(crispasr_session* s, const char* l) {
    (void)s;
    (void)l;
    return 0;
}
int crispasr_session_set_punctuation(crispasr_session* s, int e) {
    (void)s;
    (void)e;
    return 0;
}
int crispasr_session_set_punc_model(crispasr_session* s, const char* m) {
    (void)s;
    (void)m;
    return 0;
}
int crispasr_session_set_translate(crispasr_session* s, int e) {
    (void)s;
    (void)e;
    return 0;
}
int crispasr_session_set_ask(crispasr_session* s, const char* p) {
    (void)s;
    (void)p;
    return 0;
}
int crispasr_session_set_temperature(crispasr_session* s, float t, uint64_t seed) {
    (void)s;
    (void)t;
    (void)seed;
    return -2;
}
int crispasr_session_set_tts_seed(crispasr_session* s, uint64_t seed) {
    (void)s;
    (void)seed;
    return -2;
}
int crispasr_session_set_tts_steps(crispasr_session* s, int steps) {
    (void)s;
    (void)steps;
    return -2;
}
int crispasr_session_set_best_of(crispasr_session* s, int n) {
    (void)s;
    (void)n;
    return 0;
}
int crispasr_session_set_beam_size(crispasr_session* s, int n) {
    (void)s;
    (void)n;
    return 0;
}
int crispasr_session_set_return_logits(crispasr_session* s, int e) {
    (void)s;
    (void)e;
    return 0;
}
int crispasr_session_set_grammar_text(crispasr_session* s, const char* g, const char* root, float p) {
    (void)s;
    (void)g;
    (void)root;
    (void)p;
    return 0;
}
int crispasr_session_set_grammar_strict(crispasr_session* s, int strict) {
    (void)s;
    (void)strict;
    return 0;
}
int crispasr_session_score_texts(crispasr_session* s, const float* pcm, int n, const char* lang, const char* prompt,
                                 const char** texts, int ntexts, float* out, int* out_n) {
    (void)s;
    (void)pcm;
    (void)n;
    (void)lang;
    (void)prompt;
    (void)texts;
    (void)ntexts;
    (void)out;
    (void)out_n;
    return -10;
}
int crispasr_session_set_fallback_thresholds(crispasr_session* s, float a, float b, float c, float d) {
    (void)s;
    (void)a;
    (void)b;
    (void)c;
    (void)d;
    return 0;
}
int crispasr_session_set_sensitivity(crispasr_session* s, const char* p) {
    (void)s;
    (void)p;
    return 0;
}
int crispasr_session_set_alt_n(crispasr_session* s, int n) {
    (void)s;
    (void)n;
    return 0;
}
int crispasr_session_set_whisper_decode_extras(crispasr_session* s, int a, const char* b, int c) {
    (void)s;
    (void)a;
    (void)b;
    (void)c;
    return 0;
}
int crispasr_session_set_g2p_dict(crispasr_session* s, const char* d) {
    (void)s;
    (void)d;
    return 0;
}
int crispasr_session_set_top_p(crispasr_session* s, float p) {
    (void)s;
    (void)p;
    return -2;
}
int crispasr_session_set_min_p(crispasr_session* s, float p) {
    (void)s;
    (void)p;
    return -2;
}
int crispasr_session_set_repetition_penalty(crispasr_session* s, float r) {
    (void)s;
    (void)r;
    return -2;
}
int crispasr_session_set_top_k(crispasr_session* s, int k) {
    (void)s;
    (void)k;
    return -2;
}
int crispasr_session_set_do_sample(crispasr_session* s, int e) {
    (void)s;
    (void)e;
    return -2;
}
int crispasr_session_set_cfg_weight(crispasr_session* s, float w) {
    (void)s;
    (void)w;
    return -2;
}
int crispasr_session_set_tts_noise_temp(crispasr_session* s, float t) {
    (void)s;
    (void)t;
    return -2;
}
int crispasr_session_set_exaggeration(crispasr_session* s, float e) {
    (void)s;
    (void)e;
    return -2;
}
int crispasr_session_set_max_speech_tokens(crispasr_session* s, int n) {
    (void)s;
    (void)n;
    return -2;
}
int crispasr_session_set_min_speech_tokens(crispasr_session* s, int n) {
    (void)s;
    (void)n;
    return -2;
}
int crispasr_session_set_length_scale(crispasr_session* s, float sc) {
    (void)s;
    (void)sc;
    return -2;
}
int crispasr_session_set_max_new_tokens(crispasr_session* s, int n) {
    (void)s;
    (void)n;
    return 0;
}
int crispasr_session_set_frequency_penalty(crispasr_session* s, float p) {
    (void)s;
    (void)p;
    return 0;
}
int crispasr_session_set_tts_num_candidates(crispasr_session* s, int n) {
    (void)s;
    (void)n;
    return -2;
}
char* crispasr_session_translate_text(crispasr_session* s, const char* t, const char* src, const char* tgt, int m) {
    (void)s;
    (void)t;
    (void)src;
    (void)tgt;
    (void)m;
    return NULL;
}
void crispasr_session_translate_text_free(char* p) {
    tracked_free(p);
}
int crispasr_session_detect_language(crispasr_session* s, const float* pcm, int n, const char* m, int met, char* out,
                                     int cap, float* conf) {
    (void)s;
    (void)pcm;
    (void)n;
    (void)m;
    (void)met;
    (void)out;
    (void)cap;
    (void)conf;
    return 0;
}
int crispasr_session_set_codec_path(crispasr_session* s, const char* path) {
    (void)s;
    (void)path;
    return 0;
}
int crispasr_session_set_voice(crispasr_session* s, const char* path, const char* ref) {
    (void)s;
    (void)path;
    (void)ref;
    return 0;
}
int crispasr_session_set_speaker_name(crispasr_session* s, const char* n) {
    (void)s;
    (void)n;
    return -3;
}
int crispasr_session_set_speaker_id(crispasr_session* s, int id) {
    (void)s;
    (void)id;
    return -3;
}
int crispasr_session_n_speakers(crispasr_session* s) {
    (void)s;
    return 0;
}
const char* crispasr_session_get_speaker_name(crispasr_session* s, int i) {
    (void)s;
    (void)i;
    return NULL;
}
int crispasr_session_set_instruct(crispasr_session* s, const char* i) {
    (void)s;
    (void)i;
    return -3;
}
int crispasr_session_set_tts_phonemes(crispasr_session* s, const char* p) {
    (void)s;
    (void)p;
    return -2;
}
int crispasr_session_is_custom_voice(crispasr_session* s) {
    (void)s;
    return 0;
}
int crispasr_session_is_voice_design(crispasr_session* s) {
    (void)s;
    return 0;
}
float* crispasr_session_speech_to_speech(crispasr_session* s, const float* in, int n_in, char** out_text, int* out_n) {
    (void)s;
    (void)in;
    (void)n_in;
    (void)out_text;
    (void)out_n;
    return NULL;
}
int crispasr_session_set_hotwords(crispasr_session* s, const char* h, float b) {
    (void)s;
    (void)h;
    (void)b;
    return 0;
}
int crispasr_session_separate(crispasr_session* s, const float* pcm, int n) {
    (void)s;
    (void)pcm;
    (void)n;
    return -1;
}
int crispasr_session_separate_n_stems(crispasr_session* s) {
    (void)s;
    return 0;
}
const char* crispasr_session_separate_stem_name(crispasr_session* s, int i) {
    (void)s;
    (void)i;
    return "";
}
const float* crispasr_session_separate_stem(crispasr_session* s, int i, int* n) {
    (void)s;
    (void)i;
    (void)n;
    return NULL;
}
int crispasr_session_separate_sample_rate(crispasr_session* s) {
    (void)s;
    return 44100;
}
int crispasr_session_pitch(crispasr_session* s, const float* pcm, int n, float hop) {
    (void)s;
    (void)pcm;
    (void)n;
    (void)hop;
    return -1;
}
int crispasr_session_pitch_n_frames(crispasr_session* s) {
    (void)s;
    return 0;
}
int crispasr_session_pitch_frame(crispasr_session* s, int idx, float* t, float* f, float* v) {
    (void)s;
    (void)idx;
    (void)t;
    (void)f;
    (void)v;
    return -1;
}
const float* crispasr_session_pitch_frames(crispasr_session* s, int* n) {
    (void)s;
    (void)n;
    return NULL;
}
int crispasr_session_pitch_sample_rate(crispasr_session* s) {
    (void)s;
    return 16000;
}
void* crispasr_punc_init(const char* m) {
    (void)m;
    return (void*)1;
}
const char* crispasr_punc_process(void* c, const char* t) {
    (void)c;
    (void)t;
    char* p = (char*)tracked_malloc(8);
    if (p)
        strcpy(p, "Hello!");
    return p;
}
void crispasr_punc_free_text(const char* p) {
    tracked_free((void*)p);
}
void crispasr_punc_free(void* c) {
    (void)c;
}
void* crispasr_truecase_init(const char* m) {
    (void)m;
    return (void*)1;
}
const char* crispasr_truecase_process(void* c, const char* t) {
    (void)c;
    (void)t;
    char* p = (char*)tracked_malloc(8);
    if (p)
        strcpy(p, "Gruesse");
    return p;
}
void crispasr_truecase_free_text(const char* p) {
    tracked_free((void*)p);
}
void crispasr_truecase_free(void* c) {
    (void)c;
}
void* crispasr_pcs_init(const char* m) {
    (void)m;
    return (void*)1;
}
const char* crispasr_pcs_process(void* c, const char* t) {
    (void)c;
    (void)t;
    char* p = (char*)tracked_malloc(8);
    if (p)
        strcpy(p, "Hello.");
    return p;
}
void crispasr_pcs_free_text(const char* p) {
    tracked_free((void*)p);
}
void crispasr_pcs_free(void* c) {
    (void)c;
}
void* crispasr_titanet_init(const char* m, int32_t nt) {
    (void)m;
    (void)nt;
    return (void*)1;
}
void crispasr_titanet_free(void* c) {
    (void)c;
}
int32_t crispasr_titanet_embed(void* c, const float* pcm, int32_t n, float* out) {
    (void)c;
    (void)pcm;
    (void)n;
    (void)out;
    return 192;
}
float crispasr_titanet_cosine_sim(const float* a, const float* b, int32_t dim) {
    (void)a;
    (void)b;
    (void)dim;
    return 0.9f;
}
#include "wrapper-speaker-db.c"
void* crispasr_speaker_db_load(const char* dir) {
    (void)dir;
    return NULL;
}
int32_t crispasr_speaker_db_enroll(const char* dir, const char* n, const float* emb, int32_t dim) {
    (void)dir;
    (void)n;
    (void)emb;
    (void)dim;
    return -1;
}
#include "wrapper-speaker-primitives.c"
int crispasr_detect_language_pcm(const float* s, int32_t n, int32_t met, const char* m, int32_t nt, int32_t gpu,
                                 int32_t dev, int32_t flash, char* out, int32_t cap, float* conf) {
    (void)s;
    (void)n;
    (void)met;
    (void)m;
    (void)nt;
    (void)gpu;
    (void)dev;
    (void)flash;
    if (cap > 3) {
        strcpy(out, "en");
        if (conf)
            *conf = 0.9f;
        return 0;
    }
    return -1;
}
int crispasr_enhance_audio_rnnoise(const float* in, int32_t n, float* out, int32_t cap) {
    (void)cap;
    for (int i = 0; i < n && i < cap; i++)
        out[i] = in[i];
    return n;
}
int crispasr_text_detect_language(const char* t, const char* m, int32_t nt, char* out, int32_t cap, float* conf) {
    (void)t;
    (void)m;
    (void)nt;
    if (cap > 3) {
        strcpy(out, "en");
        if (conf)
            *conf = 0.95f;
        return 0;
    }
    return -1;
}
crispasr_align_result* crispasr_align_words_abi(const char* am, const char* tr, const float* s, int32_t n, int64_t off,
                                                int32_t nt) {
    (void)am;
    (void)tr;
    (void)s;
    (void)n;
    (void)off;
    (void)nt;
    return NULL;
}
int crispasr_align_result_n_words(crispasr_align_result* r) {
    (void)r;
    return 0;
}
const char* crispasr_align_result_word_text(crispasr_align_result* r, int i) {
    (void)r;
    (void)i;
    return "";
}
int64_t crispasr_align_result_word_t0(crispasr_align_result* r, int i) {
    (void)r;
    (void)i;
    return 0;
}
int64_t crispasr_align_result_word_t1(crispasr_align_result* r, int i) {
    (void)r;
    (void)i;
    return 0;
}
int crispasr_align_result_n_characters(crispasr_align_result* r, int w) {
    (void)r;
    (void)w;
    return 0;
}
const char* crispasr_align_result_character_text(crispasr_align_result* r, int w, int i) {
    (void)r;
    (void)w;
    (void)i;
    return "";
}
int64_t crispasr_align_result_character_t0(crispasr_align_result* r, int w, int i) {
    (void)r;
    (void)w;
    (void)i;
    return 0;
}
int64_t crispasr_align_result_character_t1(crispasr_align_result* r, int w, int i) {
    (void)r;
    (void)w;
    (void)i;
    return 0;
}
void crispasr_align_result_free(crispasr_align_result* r) {
    (void)r;
}
int crispasr_cache_ensure_file_abi(const char* fn, const char* url, int32_t q, const char* cache, char* out,
                                   int32_t cap) {
    (void)fn;
    (void)url;
    (void)q;
    (void)cache;
    (void)out;
    (void)cap;
    return -1;
}
int crispasr_cache_dir_abi(const char* o, char* out, int32_t cap) {
    (void)o;
    (void)out;
    (void)cap;
    return -1;
}
int crispasr_registry_lookup_abi(const char* b, char* fn, int32_t fc, char* url, int32_t uc, char* sz, int32_t sc) {
    (void)b;
    (void)fn;
    (void)fc;
    (void)url;
    (void)uc;
    (void)sz;
    (void)sc;
    return -1;
}
int crispasr_registry_lookup_by_filename_abi(const char* fn, char* o, int32_t oc, char* u, int32_t uc, char* s,
                                             int32_t sc) {
    (void)fn;
    (void)o;
    (void)oc;
    (void)u;
    (void)uc;
    (void)s;
    (void)sc;
    return -1;
}
int crispasr_registry_list_backends_abi(char* o, int32_t c) {
    (void)o;
    (void)c;
    return -1;
}
int crispasr_registry_default_bundle_info_abi(const char* b, char* ob, int32_t oc, char* ol, int32_t lc, int32_t* r) {
    (void)b;
    (void)ob;
    (void)oc;
    (void)ol;
    (void)lc;
    (void)r;
    return 0;
}
int crispasr_registry_default_bundle_artifact_abi(const char* b, int32_t idx, int32_t* k, char* fn, int32_t fc, char* u,
                                                  int32_t uc, char* s, int32_t sc) {
    (void)b;
    (void)idx;
    (void)k;
    (void)fn;
    (void)fc;
    (void)u;
    (void)uc;
    (void)s;
    (void)sc;
    return 1;
}
int crispasr_session_stream_kind(crispasr_session* s) {
    (void)s;
    return 0;
}
crispasr_stream* crispasr_session_stream_open(crispasr_session* s, int n, int step, int len, int keep, const char* lang,
                                              int tr) {
    (void)s;
    (void)n;
    (void)step;
    (void)len;
    (void)keep;
    (void)lang;
    (void)tr;
    ++stream_opens;
    return (crispasr_stream*)2;
}
bool crispasr_kokoro_lang_is_german_abi(const char* lang) {
    (void)lang;
    return false;
}
bool crispasr_kokoro_lang_has_native_voice_abi(const char* lang) {
    (void)lang;
    return false;
}
int crispasr_kokoro_resolve_model_for_lang_abi(const char* model_path, const char* lang, char* out_path,
                                               int out_path_len) {
    (void)model_path;
    (void)lang;
    (void)out_path;
    (void)out_path_len;
    return -1;
}
int crispasr_kokoro_resolve_fallback_voice_abi(const char* model_path, const char* lang, char* out_path,
                                               int out_path_len, char* out_picked, int out_picked_len) {
    (void)model_path;
    (void)lang;
    (void)out_path;
    (void)out_path_len;
    (void)out_picked;
    (void)out_picked_len;
    return -1;
}
int crispasr_session_kokoro_clear_phoneme_cache(crispasr_session* s) {
    (void)s;
    return 0;
}
int crispasr_session_set_speaker_identity(crispasr_session* s, const char* id) {
    (void)s;
    (void)id;
    return 0;
}
int crispasr_stream_feed(crispasr_stream* s, const float* pcm, int n) {
    (void)s;
    (void)pcm;
    (void)n;
    return 0;
}
int crispasr_stream_flush(crispasr_stream* s) {
    (void)s;
    return 0;
}
int crispasr_stream_get_text(crispasr_stream* s, char* out, int cap, double* t0, double* t1, int64_t* counter) {
    (void)s;
    (void)out;
    (void)cap;
    (void)t0;
    (void)t1;
    (void)counter;
    return 0;
}
void crispasr_stream_close(crispasr_stream* s) {
    (void)s;
    ++stream_closes;
}
void crispasr_stream_set_live_decode(crispasr_stream* s, int enabled) {
    (void)s;
    (void)enabled;
}
