// Minimal TTS surface for the Ruby binding. Exposes the unified
// CrispASR Session API for TTS-capable backends (kokoro, vibevoice,
// qwen3-tts, orpheus) plus the kokoro per-language model + voice
// resolver (PLAN #56 opt 2b).
//
// Surface (under module `CrispASR::Session`):
//   open(model_path, n_threads) -> handle
//   close(handle)
//   set_codec_path(handle, path)
//   set_voice(handle, path, ref_text=nil)
//   set_speaker_name(handle, name)             # orpheus + qwen3-tts CV
//   speakers(handle) -> Array<String>
//   set_instruct(handle, instruct)             # qwen3-tts VoiceDesign
//   is_custom_voice(handle) -> Boolean         # qwen3-tts variant detect
//   is_voice_design(handle) -> Boolean         # qwen3-tts variant detect
//   synthesize(handle, text) -> Array<Float>   # 24 kHz mono PCM
//
// And a singleton method:
//   CrispASR::Session.kokoro_resolve_for_lang(model_path, lang)
//     -> { model_path:, voice_path:, voice_name:, backbone_swapped: }

#include <ruby.h>
#include <ruby/thread.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Forward-declare the C ABI exported by libcrispasr. Full prototypes
// live in src/crispasr_c_api.cpp and src/kokoro.h.
struct CrispasrSession;
extern struct CrispasrSession* crispasr_session_open(const char* model_path, int n_threads);
extern void                    crispasr_session_close(struct CrispasrSession* s);
extern int                     crispasr_session_set_codec_path(struct CrispasrSession* s, const char* path);
extern int                     crispasr_session_set_voice(struct CrispasrSession* s, const char* path,
                                                          const char* ref_text_or_null);
extern int                     crispasr_session_set_speaker_name(struct CrispasrSession* s, const char* name);
extern int                     crispasr_session_n_speakers(struct CrispasrSession* s);
extern const char*             crispasr_session_get_speaker_name(struct CrispasrSession* s, int i);
extern int                     crispasr_session_set_instruct(struct CrispasrSession* s, const char* instruct);
extern int                     crispasr_session_is_custom_voice(struct CrispasrSession* s);
extern int                     crispasr_session_is_voice_design(struct CrispasrSession* s);
extern float*                  crispasr_session_synthesize(struct CrispasrSession* s, const char* text,
                                                           int* out_n_samples);
extern void                    crispasr_pcm_free(float* pcm);
extern int                     crispasr_session_kokoro_clear_phoneme_cache(struct CrispasrSession* s);
extern int                     crispasr_session_set_source_language(struct CrispasrSession* s, const char* lang);
extern int                     crispasr_session_set_target_language(struct CrispasrSession* s, const char* lang);
extern int                     crispasr_session_set_punctuation(struct CrispasrSession* s, int enable);
extern int                     crispasr_session_set_translate(struct CrispasrSession* s, int enable);
extern int                     crispasr_session_set_temperature(struct CrispasrSession* s, float temperature,
                                                                unsigned long long seed);
extern int                     crispasr_session_detect_language(struct CrispasrSession* s, const float* pcm,
                                                                int n_samples, const char* lid_model_path, int method,
                                                                char* out_lang, int out_lang_cap, float* out_prob);
extern int                     crispasr_kokoro_resolve_model_for_lang_abi(const char* model_path, const char* lang,
                                                                          char* out_path, int out_path_len);
extern int                     crispasr_kokoro_resolve_fallback_voice_abi(const char* model_path, const char* lang,
                                                                          char* out_path, int out_path_len,
                                                                          char* out_picked, int out_picked_len);

// --- Streaming (PLAN #62b): rolling-window decoder. Whisper-only at the C-ABI today.
struct CrispasrStream;
extern struct CrispasrStream* crispasr_session_stream_open(struct CrispasrSession* s, int n_threads,
                                                           int step_ms, int length_ms, int keep_ms,
                                                           const char* language, int translate);
extern int                    crispasr_stream_feed(struct CrispasrStream* s, const float* pcm, int n_samples);
extern int                    crispasr_stream_get_text(struct CrispasrStream* s, char* out_text, int out_cap,
                                                       double* out_t0_s, double* out_t1_s,
                                                       long long* out_counter);
extern int                    crispasr_stream_flush(struct CrispasrStream* s);
extern void                   crispasr_stream_close(struct CrispasrStream* s);

// --- Microphone capture (PLAN #62d): miniaudio-backed cross-platform.
struct crispasr_mic;
typedef void (*crispasr_mic_callback)(const float* pcm, int n_samples, void* userdata);
extern struct crispasr_mic*   crispasr_mic_open(int sample_rate, int channels,
                                                crispasr_mic_callback cb, void* userdata);
extern int                    crispasr_mic_start(struct crispasr_mic* m);
extern int                    crispasr_mic_stop(struct crispasr_mic* m);
extern void                   crispasr_mic_close(struct crispasr_mic* m);
extern const char*            crispasr_mic_default_device_name(void);

static VALUE mCrispASR;
static VALUE mSession;
static VALUE mStream;
static VALUE mMic;

static VALUE rb_session_open(VALUE self, VALUE model_path, VALUE n_threads) {
    struct CrispasrSession* s =
        crispasr_session_open(StringValueCStr(model_path), NUM2INT(n_threads));
    if (!s) rb_raise(rb_eRuntimeError, "crispasr_session_open: failed to open %s",
                     StringValueCStr(model_path));
    return ULL2NUM((uintptr_t)s);
}

static VALUE rb_session_close(VALUE self, VALUE handle) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    if (s) crispasr_session_close(s);
    return Qnil;
}

static VALUE rb_session_set_codec_path(VALUE self, VALUE handle, VALUE path) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_codec_path(s, StringValueCStr(path));
    if (rc != 0) rb_raise(rb_eRuntimeError, "set_codec_path failed (rc=%d)", rc);
    return Qnil;
}

// Drop the kokoro per-session phoneme cache. (PLAN #56 #5)
static VALUE rb_session_clear_phoneme_cache(VALUE self, VALUE handle) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_kokoro_clear_phoneme_cache(s);
    if (rc != 0) rb_raise(rb_eRuntimeError, "clear_phoneme_cache failed (rc=%d)", rc);
    return Qnil;
}

// ---- Sticky session-state setters (PLAN #59 partial unblock) ----

static VALUE rb_session_set_source_language(VALUE self, VALUE handle, VALUE lang) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_source_language(s, NIL_P(lang) ? "" : StringValueCStr(lang));
    if (rc != 0) rb_raise(rb_eRuntimeError, "set_source_language failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_set_target_language(VALUE self, VALUE handle, VALUE lang) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_target_language(s, NIL_P(lang) ? "" : StringValueCStr(lang));
    if (rc != 0) rb_raise(rb_eRuntimeError, "set_target_language failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_set_punctuation(VALUE self, VALUE handle, VALUE enable) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_punctuation(s, RTEST(enable) ? 1 : 0);
    if (rc != 0) rb_raise(rb_eRuntimeError, "set_punctuation failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_set_translate(VALUE self, VALUE handle, VALUE enable) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_translate(s, RTEST(enable) ? 1 : 0);
    if (rc != 0) rb_raise(rb_eRuntimeError, "set_translate failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_set_temperature(VALUE self, VALUE handle, VALUE temperature, VALUE seed) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_temperature(s, (float)NUM2DBL(temperature),
                                              (unsigned long long)NUM2ULL(seed));
    // rc == -2 = no backend supports it; soft no-op.
    if (rc != 0 && rc != -2) rb_raise(rb_eRuntimeError, "set_temperature failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_detect_language(VALUE self, VALUE handle, VALUE pcm_arr, VALUE lid_path, VALUE method) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    Check_Type(pcm_arr, T_ARRAY);
    long n = RARRAY_LEN(pcm_arr);
    float* pcm = (float*)malloc(sizeof(float) * (size_t)n);
    if (!pcm) rb_raise(rb_eNoMemError, "alloc failed");
    for (long i = 0; i < n; i++) pcm[i] = (float)NUM2DBL(rb_ary_entry(pcm_arr, i));
    char out_lang[16] = {0};
    float prob = 0.0f;
    int rc = crispasr_session_detect_language(s, pcm, (int)n, StringValueCStr(lid_path),
                                              NUM2INT(method), out_lang, sizeof(out_lang), &prob);
    free(pcm);
    if (rc != 0) rb_raise(rb_eRuntimeError, "detect_language failed (rc=%d)", rc);
    VALUE pair = rb_ary_new_capa(2);
    rb_ary_push(pair, rb_str_new_cstr(out_lang));
    rb_ary_push(pair, DBL2NUM((double)prob));
    return pair;
}

static VALUE rb_session_set_voice(int argc, VALUE* argv, VALUE self) {
    VALUE handle, path, ref_text;
    rb_scan_args(argc, argv, "21", &handle, &path, &ref_text);
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    const char* rt = NIL_P(ref_text) ? NULL : StringValueCStr(ref_text);
    int rc = crispasr_session_set_voice(s, StringValueCStr(path), rt);
    if (rc != 0) rb_raise(rb_eRuntimeError, "set_voice failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_set_speaker_name(VALUE self, VALUE handle, VALUE name) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_speaker_name(s, StringValueCStr(name));
    if (rc == -2) rb_raise(rb_eArgError, "unknown speaker: %s", StringValueCStr(name));
    if (rc == -3) rb_raise(rb_eRuntimeError, "backend has no preset speakers; use set_voice instead");
    if (rc != 0) rb_raise(rb_eRuntimeError, "set_speaker_name failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_speakers(VALUE self, VALUE handle) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int n = crispasr_session_n_speakers(s);
    VALUE arr = rb_ary_new_capa(n);
    for (int i = 0; i < n; i++) {
        const char* name = crispasr_session_get_speaker_name(s, i);
        rb_ary_push(arr, name ? rb_str_new_cstr(name) : rb_str_new_cstr(""));
    }
    return arr;
}

static VALUE rb_session_set_instruct(VALUE self, VALUE handle, VALUE instruct) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int rc = crispasr_session_set_instruct(s, StringValueCStr(instruct));
    if (rc == -3) rb_raise(rb_eRuntimeError,
            "backend is not a VoiceDesign variant; set_instruct only applies to qwen3-tts VoiceDesign models");
    if (rc != 0) rb_raise(rb_eRuntimeError, "set_instruct failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_session_is_custom_voice(VALUE self, VALUE handle) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    return crispasr_session_is_custom_voice(s) ? Qtrue : Qfalse;
}

static VALUE rb_session_is_voice_design(VALUE self, VALUE handle) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    return crispasr_session_is_voice_design(s) ? Qtrue : Qfalse;
}

static VALUE rb_session_synthesize(VALUE self, VALUE handle, VALUE text) {
    struct CrispasrSession* s = (struct CrispasrSession*)NUM2ULL(handle);
    int n = 0;
    float* pcm = crispasr_session_synthesize(s, StringValueCStr(text), &n);
    if (!pcm || n <= 0) {
        if (pcm) crispasr_pcm_free(pcm);
        rb_raise(rb_eRuntimeError, "synthesize returned no audio");
    }
    VALUE arr = rb_ary_new_capa(n);
    for (int i = 0; i < n; i++) rb_ary_push(arr, DBL2NUM((double)pcm[i]));
    crispasr_pcm_free(pcm);
    return arr;
}

static VALUE rb_kokoro_resolve_for_lang(VALUE self, VALUE model_path, VALUE lang) {
    char out_model[1024]  = {0};
    char out_voice[1024]  = {0};
    char out_picked[64]   = {0};

    const char* mp = StringValueCStr(model_path);
    const char* lg = NIL_P(lang) ? "" : StringValueCStr(lang);

    int rc = crispasr_kokoro_resolve_model_for_lang_abi(mp, lg, out_model, sizeof(out_model));
    if (rc < 0) rb_raise(rb_eRuntimeError, "kokoro_resolve_model_for_lang: buffer too small");
    int swapped = (rc == 0);
    const char* resolved = (out_model[0] != 0) ? out_model : mp;

    rc = crispasr_kokoro_resolve_fallback_voice_abi(mp, lg,
                                                    out_voice, sizeof(out_voice),
                                                    out_picked, sizeof(out_picked));
    if (rc < 0) rb_raise(rb_eRuntimeError, "kokoro_resolve_fallback_voice: buffer too small");

    VALUE h = rb_hash_new();
    rb_hash_aset(h, ID2SYM(rb_intern("model_path")),       rb_str_new_cstr(resolved));
    rb_hash_aset(h, ID2SYM(rb_intern("voice_path")),       rc == 0 ? rb_str_new_cstr(out_voice) : Qnil);
    rb_hash_aset(h, ID2SYM(rb_intern("voice_name")),       rc == 0 ? rb_str_new_cstr(out_picked) : Qnil);
    rb_hash_aset(h, ID2SYM(rb_intern("backbone_swapped")), swapped ? Qtrue : Qfalse);
    return h;
}

// =====================================================================
// Streaming (PLAN #62b) — rolling-window decoder. Whisper-only at the C-ABI today.
// =====================================================================

static VALUE rb_stream_open(VALUE self, VALUE session_h, VALUE step_ms, VALUE length_ms,
                            VALUE keep_ms, VALUE language, VALUE translate) {
    struct CrispasrSession* sess = (struct CrispasrSession*)NUM2ULL(session_h);
    const char* lang = NIL_P(language) ? "" : StringValueCStr(language);
    struct CrispasrStream* st = crispasr_session_stream_open(
        sess, 4, NUM2INT(step_ms), NUM2INT(length_ms), NUM2INT(keep_ms),
        lang, RTEST(translate) ? 1 : 0);
    if (!st) {
        rb_raise(rb_eRuntimeError, "crispasr_session_stream_open failed (whisper-only today)");
    }
    return ULL2NUM((uintptr_t)st);
}

static VALUE rb_stream_feed(VALUE self, VALUE handle, VALUE pcm_arr) {
    struct CrispasrStream* st = (struct CrispasrStream*)NUM2ULL(handle);
    Check_Type(pcm_arr, T_ARRAY);
    long n = RARRAY_LEN(pcm_arr);
    if (n == 0) return INT2NUM(0);
    float* pcm = (float*)malloc(sizeof(float) * (size_t)n);
    if (!pcm) rb_raise(rb_eNoMemError, "alloc failed");
    for (long i = 0; i < n; i++) pcm[i] = (float)NUM2DBL(rb_ary_entry(pcm_arr, i));
    int rc = crispasr_stream_feed(st, pcm, (int)n);
    free(pcm);
    if (rc < 0) rb_raise(rb_eRuntimeError, "crispasr_stream_feed failed (rc=%d)", rc);
    return INT2NUM(rc);
}

static VALUE rb_stream_get_text(VALUE self, VALUE handle) {
    struct CrispasrStream* st = (struct CrispasrStream*)NUM2ULL(handle);
    char buf[8192];
    buf[0] = '\0';
    double t0 = 0.0, t1 = 0.0;
    long long counter = 0;
    int rc = crispasr_stream_get_text(st, buf, (int)sizeof(buf), &t0, &t1, &counter);
    if (rc < 0) rb_raise(rb_eRuntimeError, "crispasr_stream_get_text failed (rc=%d)", rc);
    VALUE h = rb_hash_new();
    rb_hash_aset(h, ID2SYM(rb_intern("text")),    rb_utf8_str_new_cstr(buf));
    rb_hash_aset(h, ID2SYM(rb_intern("t0")),      DBL2NUM(t0));
    rb_hash_aset(h, ID2SYM(rb_intern("t1")),      DBL2NUM(t1));
    rb_hash_aset(h, ID2SYM(rb_intern("counter")), LL2NUM(counter));
    return h;
}

static VALUE rb_stream_flush(VALUE self, VALUE handle) {
    struct CrispasrStream* st = (struct CrispasrStream*)NUM2ULL(handle);
    int rc = crispasr_stream_flush(st);
    if (rc < 0) rb_raise(rb_eRuntimeError, "crispasr_stream_flush failed (rc=%d)", rc);
    return INT2NUM(rc);
}

static VALUE rb_stream_close(VALUE self, VALUE handle) {
    struct CrispasrStream* st = (struct CrispasrStream*)NUM2ULL(handle);
    if (st) crispasr_stream_close(st);
    return Qnil;
}

// =====================================================================
// Microphone (PLAN #62d). The audio thread is owned by miniaudio — it
// has no Ruby context and no GVL. We can't call into MRI from there.
//
// Instead: the native callback pushes copies into a bounded ring
// buffer guarded by pthread mu+cv. A dedicated Ruby pump thread
// (created via `rb_thread_create`) loops:
//   1. release the GVL via `rb_thread_call_without_gvl`
//   2. wait on the cv until a buffer is ready or the handle is closed
//   3. re-acquire the GVL and dispatch to the user's `Proc` with
//      the audio as a fresh `Array<Float>`.
//
// Closing flips `closed=1`, broadcasts the cv, joins the pump, and
// drains anything left in the ring.
// =====================================================================

#define MIC_RING_CAP 32

typedef struct mic_handle {
    pthread_mutex_t       mu;
    pthread_cond_t        cv;
    float*                bufs[MIC_RING_CAP];
    int                   lens[MIC_RING_CAP];
    int                   head;
    int                   tail;
    volatile int          closed;
    VALUE                 user_proc;
    VALUE                 pump_thread;
    struct crispasr_mic*  mic;
} mic_handle_t;

static void mic_native_cb(const float* pcm, int n, void* ud) {
    mic_handle_t* h = (mic_handle_t*)ud;
    if (n <= 0 || !pcm) return;
    pthread_mutex_lock(&h->mu);
    if (!h->closed) {
        int next = (h->tail + 1) % MIC_RING_CAP;
        if (next == h->head) {
            // Ring full: drop oldest. Better to lose a chunk than to
            // block the audio thread.
            free(h->bufs[h->head]);
            h->bufs[h->head] = NULL;
            h->head = (h->head + 1) % MIC_RING_CAP;
        }
        float* copy = (float*)malloc(sizeof(float) * (size_t)n);
        if (copy) {
            memcpy(copy, pcm, sizeof(float) * (size_t)n);
            h->bufs[h->tail] = copy;
            h->lens[h->tail] = n;
            h->tail = next;
        }
        pthread_cond_signal(&h->cv);
    }
    pthread_mutex_unlock(&h->mu);
}

struct mic_dequeue {
    mic_handle_t* h;
    float*        buf;  // out
    int           n;    // out
    int           closed; // out: 1 if woken because handle closed with no data
};

static void* mic_wait_blocking(void* p) {
    struct mic_dequeue* d = (struct mic_dequeue*)p;
    pthread_mutex_lock(&d->h->mu);
    while (d->h->head == d->h->tail && !d->h->closed) {
        pthread_cond_wait(&d->h->cv, &d->h->mu);
    }
    if (d->h->head != d->h->tail) {
        d->buf = d->h->bufs[d->h->head];
        d->n   = d->h->lens[d->h->head];
        d->h->bufs[d->h->head] = NULL;
        d->h->head = (d->h->head + 1) % MIC_RING_CAP;
    } else {
        d->closed = 1;
    }
    pthread_mutex_unlock(&d->h->mu);
    return NULL;
}

static void mic_wait_unblock(void* p) {
    mic_handle_t* h = (mic_handle_t*)p;
    pthread_mutex_lock(&h->mu);
    h->closed = 1;
    pthread_cond_broadcast(&h->cv);
    pthread_mutex_unlock(&h->mu);
}

struct mic_call_args { VALUE proc; VALUE arg; };
static VALUE mic_call_proc(VALUE p) {
    struct mic_call_args* a = (struct mic_call_args*)p;
    return rb_funcall(a->proc, rb_intern("call"), 1, a->arg);
}

static VALUE mic_pump_body(void* p) {
    mic_handle_t* h = (mic_handle_t*)p;
    while (!h->closed) {
        struct mic_dequeue d = { h, NULL, 0, 0 };
        rb_thread_call_without_gvl(mic_wait_blocking, &d, mic_wait_unblock, h);
        if (d.closed && !d.buf) break;
        if (d.buf && d.n > 0) {
            VALUE arr = rb_ary_new_capa(d.n);
            for (int i = 0; i < d.n; i++) rb_ary_push(arr, DBL2NUM((double)d.buf[i]));
            free(d.buf);
            int state = 0;
            struct mic_call_args ca = { h->user_proc, arr };
            rb_protect(mic_call_proc, (VALUE)&ca, &state);
            // Swallow user-side exceptions — the audio thread can't
            // be allowed to die because the consumer raised. The user
            // can wrap their block with their own error handling.
            (void)state;
        }
    }
    return Qnil;
}

static VALUE rb_mic_open(int argc, VALUE* argv, VALUE self) {
    VALUE sample_rate, channels, blk;
    rb_scan_args(argc, argv, "20&", &sample_rate, &channels, &blk);
    if (NIL_P(blk)) rb_raise(rb_eArgError, "CrispASR::Mic.open requires a block");

    mic_handle_t* h = (mic_handle_t*)calloc(1, sizeof(mic_handle_t));
    if (!h) rb_raise(rb_eNoMemError, "mic_handle_t alloc failed");
    pthread_mutex_init(&h->mu, NULL);
    pthread_cond_init(&h->cv, NULL);
    h->user_proc = blk;
    rb_gc_register_address(&h->user_proc);

    h->mic = crispasr_mic_open(NUM2INT(sample_rate), NUM2INT(channels), mic_native_cb, h);
    if (!h->mic) {
        rb_gc_unregister_address(&h->user_proc);
        pthread_mutex_destroy(&h->mu);
        pthread_cond_destroy(&h->cv);
        free(h);
        rb_raise(rb_eRuntimeError, "crispasr_mic_open failed");
    }

    // Cast to silence prototype-mismatch warnings on older Ruby
    // headers that still typedef rb_thread_create's fn as `(ANYARGS)`.
    h->pump_thread = rb_thread_create((VALUE (*)(void *))mic_pump_body, h);
    rb_gc_register_address(&h->pump_thread);

    return ULL2NUM((uintptr_t)h);
}

static VALUE rb_mic_start(VALUE self, VALUE handle) {
    mic_handle_t* h = (mic_handle_t*)NUM2ULL(handle);
    if (!h || !h->mic) rb_raise(rb_eRuntimeError, "mic is closed");
    int rc = crispasr_mic_start(h->mic);
    if (rc != 0) rb_raise(rb_eRuntimeError, "crispasr_mic_start failed (rc=%d)", rc);
    return Qnil;
}

static VALUE rb_mic_stop(VALUE self, VALUE handle) {
    mic_handle_t* h = (mic_handle_t*)NUM2ULL(handle);
    if (!h || !h->mic) return Qnil;
    crispasr_mic_stop(h->mic);
    return Qnil;
}

static VALUE rb_mic_close(VALUE self, VALUE handle) {
    mic_handle_t* h = (mic_handle_t*)NUM2ULL(handle);
    if (!h) return Qnil;

    // 1. Stop + close the audio thread first so the native callback
    //    can never fire again. This must happen before we tear down
    //    the queue or the callback would touch freed memory.
    if (h->mic) {
        crispasr_mic_close(h->mic);
        h->mic = NULL;
    }

    // 2. Wake the pump and join it.
    pthread_mutex_lock(&h->mu);
    h->closed = 1;
    pthread_cond_broadcast(&h->cv);
    pthread_mutex_unlock(&h->mu);
    if (h->pump_thread != Qnil && h->pump_thread != 0) {
        rb_funcall(h->pump_thread, rb_intern("join"), 0);
        rb_gc_unregister_address(&h->pump_thread);
    }

    // 3. Drain anything the audio thread enqueued before close.
    while (h->head != h->tail) {
        free(h->bufs[h->head]);
        h->bufs[h->head] = NULL;
        h->head = (h->head + 1) % MIC_RING_CAP;
    }

    rb_gc_unregister_address(&h->user_proc);
    pthread_mutex_destroy(&h->mu);
    pthread_cond_destroy(&h->cv);
    free(h);
    return Qnil;
}

static VALUE rb_mic_default_device_name(VALUE self) {
    const char* s = crispasr_mic_default_device_name();
    return rb_utf8_str_new_cstr(s ? s : "");
}

void init_ruby_crispasr_session(VALUE* mWhisper) {
    // Define module path under the existing Whisper module so existing
    // code keeps working: CrispASR::Session, but we also alias it under
    // Whisper::CrispASR::Session.
    mCrispASR = rb_define_module_under(*mWhisper, "CrispASR");
    mSession  = rb_define_module_under(mCrispASR, "Session");

    rb_define_singleton_method(mSession, "open",                 rb_session_open,             2);
    rb_define_singleton_method(mSession, "close",                rb_session_close,            1);
    rb_define_singleton_method(mSession, "set_codec_path",       rb_session_set_codec_path,   2);
    rb_define_singleton_method(mSession, "set_voice",            rb_session_set_voice,       -1);
    rb_define_singleton_method(mSession, "set_speaker_name",     rb_session_set_speaker_name, 2);
    rb_define_singleton_method(mSession, "speakers",             rb_session_speakers,         1);
    rb_define_singleton_method(mSession, "set_instruct",         rb_session_set_instruct,     2);
    rb_define_singleton_method(mSession, "is_custom_voice",      rb_session_is_custom_voice,  1);
    rb_define_singleton_method(mSession, "is_voice_design",      rb_session_is_voice_design,  1);
    rb_define_singleton_method(mSession, "synthesize",           rb_session_synthesize,       2);
    rb_define_singleton_method(mSession, "clear_phoneme_cache",  rb_session_clear_phoneme_cache, 1);
    rb_define_singleton_method(mSession, "set_source_language",  rb_session_set_source_language, 2);
    rb_define_singleton_method(mSession, "set_target_language",  rb_session_set_target_language, 2);
    rb_define_singleton_method(mSession, "set_punctuation",      rb_session_set_punctuation, 2);
    rb_define_singleton_method(mSession, "set_translate",        rb_session_set_translate, 2);
    rb_define_singleton_method(mSession, "set_temperature",      rb_session_set_temperature, 3);
    rb_define_singleton_method(mSession, "detect_language",      rb_session_detect_language, 4);
    rb_define_singleton_method(mSession, "kokoro_resolve_for_lang", rb_kokoro_resolve_for_lang, 2);

    // Streaming (PLAN #62b) — CrispASR::Session::Stream.{open, feed, get_text, flush, close}.
    mStream = rb_define_module_under(mSession, "Stream");
    rb_define_singleton_method(mStream, "open",     rb_stream_open,     6);
    rb_define_singleton_method(mStream, "feed",     rb_stream_feed,     2);
    rb_define_singleton_method(mStream, "get_text", rb_stream_get_text, 1);
    rb_define_singleton_method(mStream, "flush",    rb_stream_flush,    1);
    rb_define_singleton_method(mStream, "close",    rb_stream_close,    1);

    // Mic (PLAN #62d) — CrispASR::Mic.{open(rate, channels) { |pcm| ... }, start, stop, close, default_device_name}.
    mMic = rb_define_module_under(mCrispASR, "Mic");
    rb_define_singleton_method(mMic, "open",                rb_mic_open,                -1);
    rb_define_singleton_method(mMic, "start",               rb_mic_start,                1);
    rb_define_singleton_method(mMic, "stop",                rb_mic_stop,                 1);
    rb_define_singleton_method(mMic, "close",               rb_mic_close,                1);
    rb_define_singleton_method(mMic, "default_device_name", rb_mic_default_device_name,  0);
}
