/* Ruby auxiliary wrappers for remaining native ops.
 * Covers audio decode, container, caps/versions, watermark, pitch/separation,
 * transcribe_vad, backend getter, score_texts, pcm_rate/parakeet/grammar,
 * truecase/pcs, free_aligner, streamed polling/resets, dimension getters,
 * typed callbacks with rooted lifetime, registry/diarize turns, and chat.
 * No Ruby API called from no-GVL threads; exceptions deferred after cleanup.
 */
#include <limits.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <ruby/encoding.h>
#include <stdbool.h>
// Forward-declare callback typedefs to avoid pulling conflicting externs from crispasr_session.h
typedef void (*crispasr_progress_callback)(int, int, void*);
typedef void (*crispasr_segment_callback)(const char*, int64_t, int64_t, int, void*);
typedef void (*crispasr_token_callback)(const char*, int, void*);

/* Forward native decls not already in ruby_crispasr_session.c */
extern int crispasr_audio_load(const char* path, float** out_pcm, int* out_samples, int* out_sample_rate);
extern int crispasr_audio_load_at_rate(const char* path, int target_rate, float** out_pcm, int* out_samples,
                                       int* out_sample_rate);
extern int crispasr_audio_load_stereo(const char* path, float** out_left, float** out_right, int* out_samples,
                                      int* out_sample_rate, int* out_channels);
extern int crispasr_audio_resample(const float* pcm, int n_samples, int source_rate, int target_rate, float** out_pcm,
                                   int* out_samples);
extern void crispasr_audio_free(float* pcm);
extern unsigned char* crispasr_pcm_to_wav(const float* pcm, int n_samples, int sample_rate, size_t* out_len);
extern unsigned char* crispasr_c2pa_sign(const unsigned char* data, size_t len, const char* format,
                                         const char* cert_path, const char* key_path, size_t* out_len);
extern void crispasr_c2pa_free(unsigned char* p);
extern int crispasr_watermark_load_model(const char* gguf_path);
extern const char* crispasr_c_api_version(void);
extern const char* crispasr_dart_helpers_version(void);
extern void crispasr_set_gpu_backend(const char* name);
extern int crispasr_backend_caps_abi(const char* backend, char* out_csv, int32_t out_cap);
extern int crispasr_backend_caps_list_abi(char* out_buf, int32_t out_cap);
extern int crispasr_detect_backends_from_gguf(const char* path, char* out, int out_cap);
extern const char* crispasr_session_backend(struct CrispasrSession* s);
extern int crispasr_session_score_texts(struct CrispasrSession* s, const float* pcm, int n_samples,
                                        const char* language, const char* initial_prompt, const char** texts,
                                        int n_texts, float* out_logprobs, int* out_n_tokens);
extern int crispasr_session_set_pcm_sample_rate(struct CrispasrSession* s, int rate);
extern int crispasr_session_set_parakeet_att_context(struct CrispasrSession* s, int left, int right);
extern int crispasr_session_set_audio_ctx(struct CrispasrSession* s, int audio_ctx);
extern int crispasr_session_set_hikari_refresh_stride(struct CrispasrSession* s, int stride);
extern int crispasr_session_set_grammar_strict(struct CrispasrSession* s, int strict);
extern void* crispasr_punc_init(const char* model);
extern const char* crispasr_punc_process(void* ctx, const char* text);
extern void crispasr_punc_free_text(const char* text);
extern void crispasr_punc_free(void* ctx);
extern void* crispasr_truecase_init(const char* model_path);
extern const char* crispasr_truecase_process(void* ctx, const char* text);
extern void crispasr_truecase_free_text(const char* text);
extern void crispasr_truecase_free(void* ctx);
extern void* crispasr_pcs_init(const char* model_path);
extern const char* crispasr_pcs_process(void* ctx, const char* text);
extern void crispasr_pcs_free_text(const char* text);
extern void crispasr_pcs_free(void* ctx);
extern void crispasr_free_aligner_cache(void);
extern int crispasr_get_progress(void);
extern void crispasr_reset_progress(void);
extern int crispasr_get_streamed_segment_count(void);
extern struct crispasr_session_result* crispasr_drain_streamed_segments(void);
extern void crispasr_reset_streamed_segments(void);
extern int crispasr_get_streamed_token_count(void);
extern const char* crispasr_drain_streamed_tokens(int* out_count);
extern void crispasr_reset_streamed_tokens(void);
extern void crispasr_session_set_progress_callback(struct CrispasrSession* s, crispasr_progress_callback cb,
                                                   void* user_data);
extern void crispasr_session_set_segment_callback(struct CrispasrSession* s, crispasr_segment_callback cb,
                                                  void* user_data);
extern void crispasr_session_set_token_callback(struct CrispasrSession* s, crispasr_token_callback cb, void* user_data);
extern int crispasr_session_pitch(struct CrispasrSession* s, const float* pcm_16k, int n_samples, float hop_ms);
extern int crispasr_session_pitch_n_frames(struct CrispasrSession* s);
extern int crispasr_session_pitch_frame(struct CrispasrSession* s, int idx, float* out_time_ms, float* out_f0_hz,
                                        float* out_voiced_prob);
extern const float* crispasr_session_pitch_frames(struct CrispasrSession* s, int* out_n_frames);
extern int crispasr_session_pitch_sample_rate(struct CrispasrSession* s);
extern int crispasr_session_separate(struct CrispasrSession* s, const float* pcm_stereo, int n_samples);
extern int crispasr_session_separate_n_stems(struct CrispasrSession* s);
extern const char* crispasr_session_separate_stem_name(struct CrispasrSession* s, int stem_idx);
extern const float* crispasr_session_separate_stem(struct CrispasrSession* s, int stem_idx, int* out_n_samples);
extern int crispasr_session_separate_sample_rate(struct CrispasrSession* s);
extern int crispasr_session_convert_sample_rate(struct CrispasrSession* s);
extern int crispasr_session_piano_n_notes(struct CrispasrSession* s);
extern int crispasr_session_chords_n_spans(struct CrispasrSession* s);
extern int crispasr_session_beats_n_events(struct CrispasrSession* s);
extern int crispasr_session_tab_n_frames(struct CrispasrSession* s);
extern struct crispasr_session_result* crispasr_session_transcribe_vad(struct CrispasrSession* s, const float* pcm,
                                                                       int n_samples, int sample_rate,
                                                                       const char* vad_model_path, void* opts_or_null);
extern int crispasr_registry_lookup_abi(const char* backend, char* out_filename, int32_t filename_cap, char* out_url,
                                        int32_t url_cap, char* out_size, int32_t size_cap);
extern int crispasr_registry_lookup_by_filename_abi(const char* filename, char* out_filename, int32_t filename_cap,
                                                    char* out_url, int32_t url_cap, char* out_size, int32_t size_cap);
extern int crispasr_registry_list_backends_abi(char* out_csv, int32_t out_cap);
extern int crispasr_registry_default_bundle_info_abi(const char* backend, char* out_backend, int32_t backend_cap,
                                                     char* out_license, int32_t license_cap,
                                                     int32_t* out_requires_acceptance);
extern int crispasr_registry_default_bundle_artifact_abi(const char* backend, int32_t index, int32_t* out_kind,
                                                         char* out_filename, int32_t filename_cap, char* out_url,
                                                         int32_t url_cap, char* out_size, int32_t size_cap);
extern int crispasr_cache_ensure_file_abi(const char* filename, const char* url, int32_t quiet,
                                          const char* cache_dir_override, char* out_buf, int32_t out_cap);
extern int crispasr_cache_dir_abi(const char* cache_dir_override, char* out_buf, int32_t out_cap);
extern int crispasr_diarize_segments_abi(const float* left_pcm, const float* right_pcm, int32_t n_samples,
                                         int32_t is_stereo, void* segs, int32_t n_segs, const void* opts);
extern int crispasr_diarize_segments_turns_abi(const float* left_pcm, const float* right_pcm, int32_t n_samples,
                                               int32_t is_stereo, void* segs, int32_t n_segs, const void* opts,
                                               void* out_turns, int32_t n_turns_cap, int32_t* out_n_turns);

/* Chat header */
#include "crispasr_chat.h"

/* Globals for callback rooting */
static VALUE rb_aux_callbacks = Qnil; /* Hash session_ptr -> {sym->proc} */
static VALUE rb_aux_chat_callbacks = Qnil;
// pending and re-entry guard defined in extra.h

/* Helpers */
static void rb_aux_ensure_callbacks(void) {
    if (rb_aux_callbacks == Qnil) {
        rb_aux_callbacks = rb_hash_new();
        rb_global_variable(&rb_aux_callbacks);
    }
}
static void rb_aux_ensure_chat_callbacks(void) {
    if (rb_aux_chat_callbacks == Qnil) {
        rb_aux_chat_callbacks = rb_hash_new();
        rb_global_variable(&rb_aux_chat_callbacks);
    }
}
struct rb_aux_call_args {
    VALUE proc;
    VALUE args[4];
    int nargs;
};
static VALUE rb_aux_call_proc_body(VALUE p) {
    struct rb_aux_call_args* a = (struct rb_aux_call_args*)p;
    return rb_funcallv(a->proc, rb_intern("call"), a->nargs, a->args);
}
static void rb_aux_invoke_proc(VALUE proc, int nargs, VALUE* args) {
    if (NIL_P(proc))
        return;
    struct rb_aux_call_args a;
    a.proc = proc;
    a.nargs = nargs;
    for (int i = 0; i < nargs; i++)
        a.args[i] = args[i];
    int state = 0;
    rb_extra_in_stream = 1;
    rb_protect(rb_aux_call_proc_body, (VALUE)&a, &state);
    rb_extra_in_stream = 0;
    if (state) {
        rb_aux_capture_error();
    }
}

/* Trampolines — strict userdata routing, no cross-session scan, all Ruby ops inside rb_protect */
struct rb_aux_progress_data {
    int processed;
    int total;
    void* ud;
};
struct rb_aux_segment_data {
    const char* text;
    int64_t t0;
    int64_t t1;
    int idx;
    void* ud;
};
struct rb_aux_token_data {
    const char* text;
    int idx;
    void* ud;
};

static VALUE rb_aux_progress_body(VALUE v) {
    struct rb_aux_progress_data* a = (struct rb_aux_progress_data*)v;
    if (!a->ud)
        return Qnil;
    if (rb_aux_callbacks == Qnil)
        return Qnil;
    VALUE key = ULL2NUM((uintptr_t)a->ud);
    VALUE entry = rb_hash_aref(rb_aux_callbacks, key);
    if (NIL_P(entry))
        return Qnil;
    VALUE proc = rb_hash_aref(entry, ID2SYM(rb_intern("progress")));
    if (NIL_P(proc))
        return Qnil;
    VALUE a0 = INT2NUM(a->processed);
    VALUE a1 = INT2NUM(a->total);
    struct rb_aux_call_args ca;
    ca.proc = proc;
    ca.nargs = 2;
    ca.args[0] = a0;
    ca.args[1] = a1;
    return rb_aux_call_proc_body((VALUE)&ca);
}
static void rb_aux_progress_tramp(int processed, int total, void* ud) {
    struct rb_aux_progress_data d = {processed, total, ud};
    int state = 0;
    rb_extra_in_stream = 1;
    rb_protect(rb_aux_progress_body, (VALUE)&d, &state);
    rb_extra_in_stream = 0;
    if (state) {
        rb_aux_capture_error();
    }
}
static VALUE rb_aux_segment_body(VALUE v) {
    struct rb_aux_segment_data* a = (struct rb_aux_segment_data*)v;
    if (!a->ud)
        return Qnil;
    if (rb_aux_callbacks == Qnil)
        return Qnil;
    VALUE key = ULL2NUM((uintptr_t)a->ud);
    VALUE entry = rb_hash_aref(rb_aux_callbacks, key);
    if (NIL_P(entry))
        return Qnil;
    VALUE proc = rb_hash_aref(entry, ID2SYM(rb_intern("segment")));
    if (NIL_P(proc))
        return Qnil;
    VALUE a0 = rb_utf8_str_new_cstr(a->text ? a->text : "");
    VALUE a1 = LL2NUM(a->t0);
    VALUE a2 = LL2NUM(a->t1);
    VALUE a3 = INT2NUM(a->idx);
    struct rb_aux_call_args ca;
    ca.proc = proc;
    ca.nargs = 4;
    ca.args[0] = a0;
    ca.args[1] = a1;
    ca.args[2] = a2;
    ca.args[3] = a3;
    return rb_aux_call_proc_body((VALUE)&ca);
}
static void rb_aux_segment_tramp(const char* text, int64_t t0, int64_t t1, int idx, void* ud) {
    struct rb_aux_segment_data d = {text, t0, t1, idx, ud};
    int state = 0;
    rb_extra_in_stream = 1;
    rb_protect(rb_aux_segment_body, (VALUE)&d, &state);
    rb_extra_in_stream = 0;
    if (state) {
        rb_aux_capture_error();
    }
}
static VALUE rb_aux_token_body(VALUE v) {
    struct rb_aux_token_data* a = (struct rb_aux_token_data*)v;
    if (!a->ud)
        return Qnil;
    if (rb_aux_callbacks == Qnil)
        return Qnil;
    VALUE key = ULL2NUM((uintptr_t)a->ud);
    VALUE entry = rb_hash_aref(rb_aux_callbacks, key);
    if (NIL_P(entry))
        return Qnil;
    VALUE proc = rb_hash_aref(entry, ID2SYM(rb_intern("token")));
    if (NIL_P(proc))
        return Qnil;
    VALUE a0 = rb_utf8_str_new_cstr(a->text ? a->text : "");
    VALUE a1 = INT2NUM(a->idx);
    struct rb_aux_call_args ca;
    ca.proc = proc;
    ca.nargs = 2;
    ca.args[0] = a0;
    ca.args[1] = a1;
    return rb_aux_call_proc_body((VALUE)&ca);
}
static void rb_aux_token_tramp(const char* text, int idx, void* ud) {
    struct rb_aux_token_data d = {text, idx, ud};
    int state = 0;
    rb_extra_in_stream = 1;
    rb_protect(rb_aux_token_body, (VALUE)&d, &state);
    rb_extra_in_stream = 0;
    if (state) {
        rb_aux_capture_error();
    }
}
static VALUE rb_aux_chat_abort_dispatch(VALUE raw) {
    if (!raw || rb_aux_chat_callbacks == Qnil)
        return Qtrue;
    VALUE proc = rb_hash_aref(rb_aux_chat_callbacks, ULL2NUM((uintptr_t)raw));
    if (NIL_P(proc))
        return Qtrue;
    struct rb_aux_call_args args;
    args.proc = proc;
    args.nargs = 0;
    return rb_aux_call_proc_body((VALUE)&args);
}
static bool rb_aux_chat_abort_tramp(void* ud) {
    int state = 0;
    rb_extra_in_stream = 1;
    VALUE res = rb_protect(rb_aux_chat_abort_dispatch, (VALUE)(uintptr_t)ud, &state);
    rb_extra_in_stream = 0;
    if (state) {
        rb_aux_capture_error();
        return false; // abort on exception; original survives via pending
    }
    return RTEST(res);
}

/* Audio */

struct rb_aux_float_copy {
    const float* data;
    int n;
};
static VALUE rb_aux_float_copy_body(VALUE raw) {
    struct rb_aux_float_copy* a = (struct rb_aux_float_copy*)raw;
    if (a->n < 0 || (a->n && !a->data))
        rb_raise(rb_eRuntimeError, "Invalid native audio buffer");
    VALUE out = rb_ary_new_capa(a->n);
    for (int i = 0; i < a->n; ++i)
        rb_ary_push(out, DBL2NUM(a->data[i]));
    return out;
}
static VALUE rb_aux_owned_floats(float* data, int n) {
    struct rb_aux_float_copy a = {data, n};
    int state = 0;
    VALUE out = rb_protect(rb_aux_float_copy_body, (VALUE)&a, &state);
    crispasr_audio_free(data);
    if (state)
        rb_jump_tag(state);
    return out;
}
static VALUE rb_aux_audio_load(VALUE self, VALUE path) {
    float* pcm = NULL;
    int n = 0, rate = 0;
    int rc = crispasr_audio_load(StringValueCStr(path), &pcm, &n, &rate);
    if (rc != 0) {
        crispasr_audio_free(pcm);
        rb_raise(rb_eRuntimeError, "Audio load failed (rc=%d)", rc);
    }
    VALUE array = rb_aux_owned_floats(pcm, n), out = rb_hash_new();
    rb_hash_aset(out, ID2SYM(rb_intern("pcm")), array);
    rb_hash_aset(out, ID2SYM(rb_intern("sample_rate")), INT2NUM(rate));
    rb_hash_aset(out, ID2SYM(rb_intern("n_samples")), INT2NUM(n));
    return out;
}
static VALUE rb_aux_audio_load_at_rate(VALUE self, VALUE path, VALUE target) {
    int requested = NUM2INT(target);
    if (requested <= 0)
        rb_raise(rb_eArgError, "Positive sample rate required");
    float* pcm = NULL;
    int n = 0, rate = 0;
    int rc = crispasr_audio_load_at_rate(StringValueCStr(path), requested, &pcm, &n, &rate);
    if (rc != 0) {
        crispasr_audio_free(pcm);
        rb_raise(rb_eRuntimeError, "Audio load failed (rc=%d)", rc);
    }
    VALUE array = rb_aux_owned_floats(pcm, n), out = rb_hash_new();
    rb_hash_aset(out, ID2SYM(rb_intern("pcm")), array);
    rb_hash_aset(out, ID2SYM(rb_intern("sample_rate")), INT2NUM(rate));
    rb_hash_aset(out, ID2SYM(rb_intern("n_samples")), INT2NUM(n));
    return out;
}
static VALUE rb_aux_audio_load_stereo(VALUE self, VALUE path) {
    float *left = NULL, *right = NULL;
    int n = 0, rate = 0, channels = 0;
    int rc = crispasr_audio_load_stereo(StringValueCStr(path), &left, &right, &n, &rate, &channels);
    int state = 0;
    VALUE la = Qnil, ra = Qnil;
    if (rc == 0) {
        struct rb_aux_float_copy a = {left, n}, b = {right, n};
        la = rb_protect(rb_aux_float_copy_body, (VALUE)&a, &state);
        if (!state)
            ra = rb_protect(rb_aux_float_copy_body, (VALUE)&b, &state);
    }
    crispasr_audio_free(left);
    if (right != left)
        crispasr_audio_free(right);
    if (state)
        rb_jump_tag(state);
    if (rc != 0)
        rb_raise(rb_eRuntimeError, "Stereo audio load failed (rc=%d)", rc);
    VALUE out = rb_hash_new();
    rb_hash_aset(out, ID2SYM(rb_intern("left")), la);
    rb_hash_aset(out, ID2SYM(rb_intern("right")), ra);
    rb_hash_aset(out, ID2SYM(rb_intern("sample_rate")), INT2NUM(rate));
    rb_hash_aset(out, ID2SYM(rb_intern("channels")), INT2NUM(channels));
    rb_hash_aset(out, ID2SYM(rb_intern("n_samples")), INT2NUM(n));
    return out;
}
static VALUE rb_aux_pcm_storage(VALUE array) {
    Check_Type(array, T_ARRAY);
    long n = RARRAY_LEN(array);
    if (n > INT_MAX)
        rb_raise(rb_eArgError, "PCM too large");
    VALUE storage = rb_str_new(NULL, n * sizeof(float));
    for (long i = 0; i < n; ++i) {
        float value = NUM2DBL(rb_ary_entry(array, i));
        ((float*)RSTRING_PTR(storage))[i] = value;
    }
    return storage;
}
static VALUE rb_aux_audio_resample(VALUE self, VALUE pcm, VALUE source, VALUE target) {
    int sr = NUM2INT(source), tr = NUM2INT(target);
    if (sr <= 0 || tr <= 0)
        rb_raise(rb_eArgError, "Positive sample rates required");
    VALUE storage = rb_aux_pcm_storage(pcm);
    if (RARRAY_LEN(pcm) == 0)
        return rb_ary_new();
    float* out = NULL;
    int n = 0;
    int rc = crispasr_audio_resample((float*)RSTRING_PTR(storage), RARRAY_LEN(pcm), sr, tr, &out, &n);
    RB_GC_GUARD(storage);
    if (rc != 0) {
        crispasr_audio_free(out);
        rb_raise(rb_eRuntimeError, "Resampling failed (rc=%d)", rc);
    }
    return rb_aux_owned_floats(out, n);
}
static VALUE rb_aux_pcm_to_wav(VALUE self, VALUE pcm_arr, VALUE rate) {
    Check_Type(pcm_arr, T_ARRAY);
    long n = RARRAY_LEN(pcm_arr);
    if (n <= 0)
        rb_raise(rb_eArgError, "pcm empty");
    if (NUM2INT(rate) <= 0)
        rb_raise(rb_eArgError, "sample_rate must be positive");
    VALUE storage = rb_str_new(NULL, n * sizeof(float));
    for (long i = 0; i < n; i++)
        ((float*)RSTRING_PTR(storage))[i] = (float)NUM2DBL(rb_ary_entry(pcm_arr, i));
    size_t out_len = 0;
    unsigned char* wav = crispasr_pcm_to_wav((float*)RSTRING_PTR(storage), (int)n, NUM2INT(rate), &out_len);
    RB_GC_GUARD(storage);
    if (!wav || out_len == 0)
        rb_raise(rb_eRuntimeError, "pcm_to_wav failed");
    VALUE str = rb_str_new((const char*)wav, out_len);
    rb_enc_associate(str, rb_ascii8bit_encoding());
    crispasr_c2pa_free(wav);
    rb_aux_check_pending();
    return str;
}
static VALUE rb_aux_c2pa_sign(int argc, VALUE* argv, VALUE self) {
    VALUE data, format, cert, key;
    rb_scan_args(argc, argv, "22", &data, &format, &cert, &key);
    StringValue(data);
    const char* fmt = NIL_P(format) ? "audio/wav" : StringValueCStr(format);
    const char* cert_p = NIL_P(cert) ? NULL : StringValueCStr(cert);
    const char* key_p = NIL_P(key) ? NULL : StringValueCStr(key);
    size_t out_len = 0;
    unsigned char* out =
        crispasr_c2pa_sign((unsigned char*)RSTRING_PTR(data), RSTRING_LEN(data), fmt, cert_p, key_p, &out_len);
    if (!out)
        rb_raise(rb_eRuntimeError, "c2pa_sign failed (unavailable or invalid input)");
    VALUE str = rb_str_new((const char*)out, out_len);
    rb_enc_associate(str, rb_ascii8bit_encoding());
    crispasr_c2pa_free(out);
    rb_aux_check_pending();
    return str;
}
static VALUE rb_aux_watermark_load_model(VALUE self, VALUE path) {
    int rc = crispasr_watermark_load_model(StringValueCStr(path));
    if (rc != 0)
        rb_raise(rb_eRuntimeError, "watermark_load_model failed (rc=%d)", rc);
    return Qnil;
}
static VALUE rb_aux_c_api_version(VALUE self) {
    const char* v = crispasr_c_api_version();
    return rb_utf8_str_new_cstr(v ? v : "");
}
static VALUE rb_aux_dart_helpers_version(VALUE self) {
    const char* v = crispasr_dart_helpers_version();
    return rb_utf8_str_new_cstr(v ? v : "");
}
static VALUE rb_aux_set_gpu_backend(VALUE self, VALUE name) {
    const char* n = NIL_P(name) ? NULL : StringValueCStr(name);
    crispasr_set_gpu_backend(n);
    return Qnil;
}
static VALUE rb_aux_backend_caps(VALUE self, VALUE backend) {
    char buf[8192] = {0};
    int rc = crispasr_backend_caps_abi(StringValueCStr(backend), buf, sizeof(buf));
    if (rc == -3)
        rb_raise(rb_eArgError, "unknown backend: %s", StringValueCStr(backend));
    if (rc < 0)
        rb_raise(rb_eRuntimeError, "backend_caps failed (rc=%d)", rc);
    return rb_utf8_str_new_cstr(buf);
}
static VALUE rb_aux_backend_caps_list(VALUE self) {
    char buf[8192] = {0};
    int rc = crispasr_backend_caps_list_abi(buf, sizeof(buf));
    if (rc < 0) {
        int need = -rc;
        char* dyn = malloc(need + 1);
        if (!dyn)
            rb_raise(rb_eNoMemError, "alloc failed");
        int rc2 = crispasr_backend_caps_list_abi(dyn, need + 1);
        if (rc2 < 0) {
            free(dyn);
            rb_raise(rb_eRuntimeError, "backend_caps_list failed");
        }
        VALUE s = rb_utf8_str_new(dyn, rc2);
        free(dyn);
        return s;
    }
    return rb_utf8_str_new(buf, rc);
}
static VALUE rb_aux_detect_backends(VALUE self, VALUE path) {
    char buf[1024] = {0};
    int rc = crispasr_detect_backends_from_gguf(StringValueCStr(path), buf, sizeof(buf));
    if (rc == -5) {
        // Need larger buffer: rc is -required
        int need = -rc;
        char* dyn = malloc(need + 1);
        if (!dyn)
            rb_raise(rb_eNoMemError, "alloc failed");
        int rc2 = crispasr_detect_backends_from_gguf(StringValueCStr(path), dyn, need + 1);
        if (rc2 < 0) {
            free(dyn);
            rb_raise(rb_eRuntimeError, "detect_backends_from_gguf failed (rc=%d)", rc2);
        }
        VALUE arr = rb_ary_new();
        char* save = NULL;
        char* tok = strtok_r(dyn, "\n", &save);
        while (tok) {
            rb_ary_push(arr, rb_utf8_str_new_cstr(tok));
            tok = strtok_r(NULL, "\n", &save);
        }
        free(dyn);
        return arr;
    }
    if (rc < 0)
        rb_raise(rb_eRuntimeError, "detect_backends_from_gguf failed (rc=%d)", rc);
    VALUE arr = rb_ary_new();
    char* save = NULL;
    char* tok = strtok_r(buf, "\n", &save);
    while (tok) {
        rb_ary_push(arr, rb_utf8_str_new_cstr(tok));
        tok = strtok_r(NULL, "\n", &save);
    }
    return arr;
}
static VALUE rb_aux_backend_getter(VALUE self, VALUE handle) {
    struct CrispasrSession* s = rb_extra_handle(handle);
    const char* b = crispasr_session_backend(s);
    return rb_utf8_str_new_cstr(b ? b : "");
}
static VALUE rb_aux_score_texts(VALUE self, VALUE handle, VALUE pcm_arr, VALUE lang, VALUE prompt, VALUE texts) {
    struct CrispasrSession* s = rb_extra_handle(handle);
    Check_Type(pcm_arr, T_ARRAY);
    Check_Type(texts, T_ARRAY);
    long n = RARRAY_LEN(pcm_arr);
    long tn = RARRAY_LEN(texts);
    if (tn == 0)
        rb_raise(rb_eArgError, "texts empty");
    VALUE pcm_storage = rb_str_new(NULL, n * sizeof(float));
    for (long i = 0; i < n; i++)
        ((float*)RSTRING_PTR(pcm_storage))[i] = (float)NUM2DBL(rb_ary_entry(pcm_arr, i));
    const char* lang_s = NIL_P(lang) ? NULL : StringValueCStr(lang);
    const char* prompt_s = NIL_P(prompt) ? NULL : StringValueCStr(prompt);
    // Build texts array
    const char** c_texts = malloc(tn * sizeof(char*));
    VALUE* keep = malloc(tn * sizeof(VALUE));
    for (long i = 0; i < tn; i++) {
        VALUE t = rb_ary_entry(texts, i);
        keep[i] = t;
        c_texts[i] = StringValueCStr(t);
    }
    float* scores = malloc(tn * sizeof(float));
    int* tokens = malloc(tn * sizeof(int));
    int rc = crispasr_session_score_texts(s, (float*)RSTRING_PTR(pcm_storage), (int)n, lang_s, prompt_s, c_texts,
                                          (int)tn, scores, tokens);
    RB_GC_GUARD(pcm_storage);
    free(c_texts);
    free(keep);
    if (rc != 0) {
        free(scores);
        free(tokens);
        rb_raise(rb_eRuntimeError, "score_texts failed (rc=%d)", rc);
    }
    VALUE r = rb_hash_new();
    VALUE sarr = rb_ary_new_capa(tn), tarr = rb_ary_new_capa(tn);
    for (long i = 0; i < tn; i++) {
        rb_ary_push(sarr, DBL2NUM(scores[i]));
        rb_ary_push(tarr, INT2NUM(tokens[i]));
    }
    free(scores);
    free(tokens);
    rb_hash_aset(r, ID2SYM(rb_intern("logprobs")), sarr);
    rb_hash_aset(r, ID2SYM(rb_intern("n_tokens")), tarr);
    rb_aux_check_pending();
    return r;
}
static VALUE rb_aux_set_pcm_rate(VALUE self, VALUE handle, VALUE rate) {
    struct CrispasrSession* s = rb_extra_handle(handle);
    int rc = crispasr_session_set_pcm_sample_rate(s, NUM2INT(rate));
    if (rc != 0)
        rb_raise(rb_eRuntimeError, "set_pcm_sample_rate failed (rc=%d)", rc);
    return Qnil;
}
static VALUE rb_aux_set_att(VALUE self, VALUE handle, VALUE left, VALUE right) {
    struct CrispasrSession* s = rb_extra_handle(handle);
    int rc = crispasr_session_set_parakeet_att_context(s, NUM2INT(left), NUM2INT(right));
    if (rc != 0)
        rb_raise(rb_eRuntimeError, "set_parakeet_att_context failed (rc=%d)", rc);
    return Qnil;
}
static VALUE rb_aux_set_audio_ctx(VALUE self, VALUE handle, VALUE audio_ctx) {
    struct CrispasrSession* s = rb_extra_handle(handle);
    int rc = crispasr_session_set_audio_ctx(s, NUM2INT(audio_ctx));
    if (rc != 0)
        rb_raise(rb_eRuntimeError, "set_audio_ctx failed (rc=%d)", rc);
    return Qnil;
}
static VALUE rb_aux_set_hikari_refresh_stride(VALUE self, VALUE handle, VALUE stride) {
    struct CrispasrSession* s = rb_extra_handle(handle);
    int rc = crispasr_session_set_hikari_refresh_stride(s, NUM2INT(stride));
    if (rc != 0)
        rb_raise(rb_eRuntimeError, "set_hikari_refresh_stride failed (rc=%d)", rc);
    return Qnil;
}
static VALUE rb_aux_set_grammar_strict(VALUE self, VALUE handle, VALUE v) {
    struct CrispasrSession* s = rb_extra_handle(handle);
    int rc = crispasr_session_set_grammar_strict(s, RTEST(v) ? 1 : 0);
    if (rc != 0)
        rb_raise(rb_eRuntimeError, "set_grammar_strict failed (rc=%d)", rc);
    return Qnil;
}
/* Punc/Truecase/PCS generic */
typedef void* (*init_fn)(const char*);
typedef const char* (*proc_fn)(void*, const char*);
typedef void (*free_text_fn)(const char*);
typedef void (*free_fn)(void*);
struct rb_aux_text_registration {
    void* native;
    int kind;
};
static VALUE rb_aux_text_registration_body(VALUE raw) {
    struct rb_aux_text_registration* registration = (void*)(uintptr_t)raw;
    VALUE key = rb_extra_new_ticket();
    VALUE record = rb_ary_new_from_args(2, ULL2NUM((uintptr_t)registration->native), INT2NUM(registration->kind));
    rb_hash_aset(rb_text_handles, key, record);
    return key;
}
static VALUE rb_aux_text_processor_init(VALUE self, VALUE path, init_fn fn, free_fn release, int kind) {
    rb_extra_session_maps();
    void* native = fn(StringValueCStr(path));
    if (!native)
        rb_raise(rb_eRuntimeError, "processor init failed");
    struct rb_aux_text_registration registration = {native, kind};
    int state = 0;
    VALUE key = rb_protect(rb_aux_text_registration_body, (VALUE)(uintptr_t)&registration, &state);
    if (state) {
        release(native);
        rb_jump_tag(state);
    }
    return key;
}
static int rb_aux_text_kind(VALUE handle) {
    if (!RB_INTEGER_TYPE_P(handle))
        rb_raise(rb_eTypeError, "Integer processor handle required");
    VALUE kind = rb_text_handles == Qnil ? Qnil : rb_hash_aref(rb_text_handles, handle);
    return NIL_P(kind) ? 0 : NUM2INT(rb_ary_entry(kind, 1));
}
struct rb_aux_text_copy {
    const char* text;
};
static VALUE rb_aux_text_copy_body(VALUE raw) {
    return rb_utf8_str_new_cstr(((struct rb_aux_text_copy*)(uintptr_t)raw)->text);
}
static VALUE rb_aux_text_processor_process(VALUE self, VALUE handle, VALUE text, proc_fn fn, free_text_fn release,
                                           int kind) {
    if (rb_aux_text_kind(handle) != kind)
        rb_raise(rb_eRuntimeError, "Processor is closed or has a different type");
    const char* input = StringValueCStr(text);
    VALUE record = rb_hash_aref(rb_text_handles, handle);
    const char* output = fn((void*)(uintptr_t)NUM2ULL(rb_ary_entry(record, 0)), input);
    if (!output)
        rb_raise(rb_eRuntimeError, "processor process failed");
    struct rb_aux_text_copy copy = {output};
    int state = 0;
    VALUE result = rb_protect(rb_aux_text_copy_body, (VALUE)(uintptr_t)&copy, &state);
    release(output);
    if (state)
        rb_jump_tag(state);
    return result;
}
static VALUE rb_aux_text_processor_free(VALUE self, VALUE handle, free_fn release, int kind) {
    int actual = rb_aux_text_kind(handle);
    if (!actual)
        return Qnil;
    if (actual != kind)
        rb_raise(rb_eRuntimeError, "Processor has a different type");
    VALUE record = rb_hash_delete(rb_text_handles, handle);
    release((void*)(uintptr_t)NUM2ULL(rb_ary_entry(record, 0)));
    return Qnil;
}
static VALUE rb_aux_punc_init(VALUE self, VALUE path) {
    return rb_aux_text_processor_init(self, path, crispasr_punc_init, crispasr_punc_free, 1);
}
static VALUE rb_aux_punc_process(VALUE self, VALUE handle, VALUE text) {
    return rb_aux_text_processor_process(self, handle, text, crispasr_punc_process, crispasr_punc_free_text, 1);
}
static VALUE rb_aux_punc_free(VALUE self, VALUE handle) {
    return rb_aux_text_processor_free(self, handle, crispasr_punc_free, 1);
}
static VALUE rb_aux_truecase_init(VALUE self, VALUE path) {
    return rb_aux_text_processor_init(self, path, crispasr_truecase_init, crispasr_truecase_free, 2);
}
static VALUE rb_aux_truecase_process(VALUE self, VALUE handle, VALUE text) {
    return rb_aux_text_processor_process(self, handle, text, crispasr_truecase_process, crispasr_truecase_free_text, 2);
}
static VALUE rb_aux_truecase_free(VALUE self, VALUE handle) {
    return rb_aux_text_processor_free(self, handle, crispasr_truecase_free, 2);
}
static VALUE rb_aux_pcs_init(VALUE self, VALUE path) {
    return rb_aux_text_processor_init(self, path, crispasr_pcs_init, crispasr_pcs_free, 3);
}
static VALUE rb_aux_pcs_process(VALUE self, VALUE handle, VALUE text) {
    return rb_aux_text_processor_process(self, handle, text, crispasr_pcs_process, crispasr_pcs_free_text, 3);
}
static VALUE rb_aux_pcs_free(VALUE self, VALUE handle) {
    return rb_aux_text_processor_free(self, handle, crispasr_pcs_free, 3);
}
static VALUE rb_aux_free_aligner(VALUE self) {
    crispasr_free_aligner_cache();
    return Qnil;
}

/* Streamed polling */
static VALUE rb_aux_get_streamed_seg_count(VALUE self) {
    return INT2NUM(crispasr_get_streamed_segment_count());
}
static VALUE rb_aux_drain_streamed_seg(VALUE self) {
    struct crispasr_session_result* r = crispasr_drain_streamed_segments();
    if (!r)
        return Qnil;
    VALUE segs = rb_session_copy_segments(r);
    crispasr_session_result_free(r);
    rb_aux_check_pending();
    return segs;
}
static VALUE rb_aux_reset_streamed_seg(VALUE self) {
    crispasr_reset_streamed_segments();
    return Qnil;
}
static VALUE rb_aux_get_streamed_tok_count(VALUE self) {
    return INT2NUM(crispasr_get_streamed_token_count());
}
static VALUE rb_aux_drain_streamed_tok(VALUE self) {
    int n = 0;
    const char* p = crispasr_drain_streamed_tokens(&n);
    if (!p || n <= 0)
        return rb_ary_new();
    VALUE arr = rb_ary_new_capa(n);
    const char* cur = p;
    for (int i = 0; i < n; i++) {
        size_t len = strlen(cur);
        VALUE s = rb_str_new(cur, len);
        // tokens may contain UTF-8 fragments; preserve bytes then scrub for display? keep raw bytes
        // For test we need to handle split bytes as original fixture does: raw bytes preserved
        // Use binary then force UTF-8 for consistency with transcribe
        rb_enc_associate(s, rb_ascii8bit_encoding());
        rb_ary_push(arr, s);
        cur += len + 1;
    }
    return arr;
}
static VALUE rb_aux_reset_streamed_tok(VALUE self) {
    crispasr_reset_streamed_tokens();
    return Qnil;
}
static VALUE rb_aux_get_progress_aux(VALUE self) {
    return INT2NUM(crispasr_get_progress());
}
static VALUE rb_aux_reset_progress_aux(VALUE self) {
    crispasr_reset_progress();
    return Qnil;
}

/* Callbacks */
static VALUE rb_aux_set_progress_cb(VALUE self, VALUE handle, VALUE proc) {
    struct CrispasrSession* s = rb_extra_handle(handle);
    rb_aux_ensure_callbacks();
    VALUE key = ULL2NUM((uintptr_t)s);
    VALUE entry = rb_hash_aref(rb_aux_callbacks, key);
    if (NIL_P(entry)) {
        entry = rb_hash_new();
        rb_hash_aset(rb_aux_callbacks, key, entry);
    }
    if (NIL_P(proc)) {
        rb_hash_aset(entry, ID2SYM(rb_intern("progress")), Qnil);
        crispasr_session_set_progress_callback(s, NULL, NULL);
    } else {
        rb_hash_aset(entry, ID2SYM(rb_intern("progress")), proc);
        crispasr_session_set_progress_callback(s, rb_aux_progress_tramp, (void*)s);
    }
    return Qnil;
}
static VALUE rb_aux_set_segment_cb(VALUE self, VALUE handle, VALUE proc) {
    struct CrispasrSession* s = rb_extra_handle(handle);
    rb_aux_ensure_callbacks();
    VALUE key = ULL2NUM((uintptr_t)s);
    VALUE entry = rb_hash_aref(rb_aux_callbacks, key);
    if (NIL_P(entry)) {
        entry = rb_hash_new();
        rb_hash_aset(rb_aux_callbacks, key, entry);
    }
    if (NIL_P(proc)) {
        rb_hash_aset(entry, ID2SYM(rb_intern("segment")), Qnil);
        crispasr_session_set_segment_callback(s, NULL, NULL);
    } else {
        rb_hash_aset(entry, ID2SYM(rb_intern("segment")), proc);
        crispasr_session_set_segment_callback(s, rb_aux_segment_tramp, (void*)s);
    }
    return Qnil;
}
static VALUE rb_aux_set_token_cb(VALUE self, VALUE handle, VALUE proc) {
    struct CrispasrSession* s = rb_extra_handle(handle);
    rb_aux_ensure_callbacks();
    VALUE key = ULL2NUM((uintptr_t)s);
    VALUE entry = rb_hash_aref(rb_aux_callbacks, key);
    if (NIL_P(entry)) {
        entry = rb_hash_new();
        rb_hash_aset(rb_aux_callbacks, key, entry);
    }
    if (NIL_P(proc)) {
        rb_hash_aset(entry, ID2SYM(rb_intern("token")), Qnil);
        crispasr_session_set_token_callback(s, NULL, NULL);
    } else {
        rb_hash_aset(entry, ID2SYM(rb_intern("token")), proc);
        crispasr_session_set_token_callback(s, rb_aux_token_tramp, (void*)s);
    }
    return Qnil;
}
static void rb_aux_clear_callbacks(VALUE handle) {
    if (rb_aux_callbacks == Qnil)
        return;
    VALUE key = handle;
    rb_hash_delete(rb_aux_callbacks, key);
}

/* Pitch */
static VALUE rb_aux_pitch(VALUE self, VALUE handle, VALUE pcm_arr, VALUE hop) {
    struct CrispasrSession* s = rb_extra_handle(handle);
    VALUE pcm = rb_extra_pcm(pcm_arr);
    float hop_f = NIL_P(hop) ? 10.0f : (float)NUM2DBL(hop);
    int rc = crispasr_session_pitch(s, (float*)RSTRING_PTR(pcm), (int)RARRAY_LEN(pcm_arr), hop_f);
    RB_GC_GUARD(pcm);
    if (rc < 0)
        rb_raise(rb_eRuntimeError, "pitch failed (rc=%d)", rc);
    int n = 0;
    const float* frames = crispasr_session_pitch_frames(s, &n);
    VALUE h = rb_hash_new();
    VALUE arr = Qnil;
    if (n > 0 && frames) {
        arr = rb_ary_new_capa((long)n * 3);
        for (int i = 0; i < n * 3; i++)
            rb_ary_push(arr, DBL2NUM(frames[i]));
    } else {
        arr = rb_ary_new();
    }
    rb_hash_aset(h, ID2SYM(rb_intern("frames")), arr);
    rb_hash_aset(h, ID2SYM(rb_intern("n_frames")), INT2NUM(n));
    rb_hash_aset(h, ID2SYM(rb_intern("sample_rate")), INT2NUM(crispasr_session_pitch_sample_rate(s)));
    rb_aux_check_pending();
    return h;
}
static VALUE rb_aux_pitch_n_frames(VALUE self, VALUE handle) {
    struct CrispasrSession* s = rb_extra_handle(handle);
    return INT2NUM(crispasr_session_pitch_n_frames(s));
}
static VALUE rb_aux_pitch_sample_rate(VALUE self, VALUE handle) {
    struct CrispasrSession* s = rb_extra_handle(handle);
    return INT2NUM(crispasr_session_pitch_sample_rate(s));
}
static VALUE rb_aux_pitch_frame(VALUE self, VALUE handle, VALUE idx) {
    struct CrispasrSession* s = rb_extra_handle(handle);
    float t = 0, f = 0, p = 0;
    int rc = crispasr_session_pitch_frame(s, NUM2INT(idx), &t, &f, &p);
    if (rc != 0)
        rb_raise(rb_eArgError, "pitch_frame invalid index");
    VALUE h = rb_hash_new();
    rb_hash_aset(h, ID2SYM(rb_intern("time_ms")), DBL2NUM(t));
    rb_hash_aset(h, ID2SYM(rb_intern("f0_hz")), DBL2NUM(f));
    rb_hash_aset(h, ID2SYM(rb_intern("voiced_prob")), DBL2NUM(p));
    return h;
}

/* Separate */
static VALUE rb_aux_separate(VALUE self, VALUE handle, VALUE pcm_arr) {
    struct CrispasrSession* s = rb_extra_handle(handle);
    VALUE pcm = rb_extra_pcm(pcm_arr);
    int rc = crispasr_session_separate(s, (float*)RSTRING_PTR(pcm), (int)RARRAY_LEN(pcm_arr));
    RB_GC_GUARD(pcm);
    if (rc < 0)
        rb_raise(rb_eRuntimeError, "separate failed (rc=%d)", rc);
    int n = crispasr_session_separate_n_stems(s);
    VALUE h = rb_hash_new();
    rb_hash_aset(h, ID2SYM(rb_intern("n_stems")), INT2NUM(n));
    rb_hash_aset(h, ID2SYM(rb_intern("sample_rate")), INT2NUM(crispasr_session_separate_sample_rate(s)));
    VALUE stems = rb_ary_new_capa(n);
    for (int i = 0; i < n; i++) {
        const char* name = crispasr_session_separate_stem_name(s, i);
        int out_n = 0;
        const float* p = crispasr_session_separate_stem(s, i, &out_n);
        VALUE sh = rb_hash_new();
        rb_hash_aset(sh, ID2SYM(rb_intern("name")), rb_utf8_str_new_cstr(name ? name : ""));
        if (p && out_n > 0) {
            VALUE arr = rb_ary_new_capa(out_n);
            for (int j = 0; j < out_n; j++)
                rb_ary_push(arr, DBL2NUM(p[j]));
            rb_hash_aset(sh, ID2SYM(rb_intern("pcm")), arr);
        } else {
            rb_hash_aset(sh, ID2SYM(rb_intern("pcm")), rb_ary_new());
        }
        rb_hash_aset(sh, ID2SYM(rb_intern("n_samples")), INT2NUM(out_n));
        rb_ary_push(stems, sh);
    }
    rb_hash_aset(h, ID2SYM(rb_intern("stems")), stems);
    rb_aux_check_pending();
    return h;
}
static VALUE rb_aux_separate_n_stems(VALUE self, VALUE handle) {
    struct CrispasrSession* s = rb_extra_handle(handle);
    return INT2NUM(crispasr_session_separate_n_stems(s));
}
static VALUE rb_aux_separate_sample_rate(VALUE self, VALUE handle) {
    struct CrispasrSession* s = rb_extra_handle(handle);
    return INT2NUM(crispasr_session_separate_sample_rate(s));
}
static VALUE rb_aux_convert_sample_rate(VALUE self, VALUE handle) {
    struct CrispasrSession* s = rb_extra_handle(handle);
    return INT2NUM(crispasr_session_convert_sample_rate(s));
}
static VALUE rb_aux_piano_n_notes(VALUE self, VALUE handle) {
    struct CrispasrSession* s = rb_extra_handle(handle);
    return INT2NUM(crispasr_session_piano_n_notes(s));
}
static VALUE rb_aux_chords_n_spans(VALUE self, VALUE handle) {
    struct CrispasrSession* s = rb_extra_handle(handle);
    return INT2NUM(crispasr_session_chords_n_spans(s));
}
static VALUE rb_aux_beats_n_events(VALUE self, VALUE handle) {
    struct CrispasrSession* s = rb_extra_handle(handle);
    return INT2NUM(crispasr_session_beats_n_events(s));
}
static VALUE rb_aux_tab_n_frames(VALUE self, VALUE handle) {
    struct CrispasrSession* s = rb_extra_handle(handle);
    return INT2NUM(crispasr_session_tab_n_frames(s));
}

/* Transcribe VAD plain */
static VALUE rb_aux_transcribe_vad(VALUE self, VALUE handle, VALUE pcm_arr, VALUE vad_model) {
    struct CrispasrSession* s = rb_extra_handle(handle);
    Check_Type(pcm_arr, T_ARRAY);
    long n = RARRAY_LEN(pcm_arr);
    VALUE storage = rb_str_new(NULL, n * sizeof(float));
    for (long i = 0; i < n; i++)
        ((float*)RSTRING_PTR(storage))[i] = (float)NUM2DBL(rb_ary_entry(pcm_arr, i));
    struct crispasr_session_result* r = crispasr_session_transcribe_vad(s, (float*)RSTRING_PTR(storage), (int)n, 16000,
                                                                        StringValueCStr(vad_model), NULL);
    RB_GC_GUARD(storage);
    if (!r) {
        rb_aux_check_pending();
        rb_raise(rb_eRuntimeError, "transcribe_vad failed");
    }
    VALUE segs = rb_session_copy_segments(r);
    crispasr_session_result_free(r);
    rb_aux_check_pending();
    return segs;
}

/* Registry helpers */
static VALUE rb_aux_registry_lookup(VALUE self, VALUE backend) {
    char fn[1024] = {0}, url[2048] = {0}, sz[256] = {0};
    int rc = crispasr_registry_lookup_abi(StringValueCStr(backend), fn, sizeof(fn), url, sizeof(url), sz, sizeof(sz));
    if (rc != 0)
        rb_raise(rb_eRuntimeError, "registry_lookup failed (rc=%d)", rc);
    VALUE h = rb_hash_new();
    rb_hash_aset(h, ID2SYM(rb_intern("filename")), rb_utf8_str_new_cstr(fn));
    rb_hash_aset(h, ID2SYM(rb_intern("url")), rb_utf8_str_new_cstr(url));
    rb_hash_aset(h, ID2SYM(rb_intern("size")), rb_utf8_str_new_cstr(sz));
    return h;
}
static VALUE rb_aux_registry_lookup_by_filename(VALUE self, VALUE filename) {
    char fn[1024] = {0}, url[2048] = {0}, sz[256] = {0};
    int rc = crispasr_registry_lookup_by_filename_abi(StringValueCStr(filename), fn, sizeof(fn), url, sizeof(url), sz,
                                                      sizeof(sz));
    if (rc != 0)
        rb_raise(rb_eRuntimeError, "registry_lookup_by_filename failed (rc=%d)", rc);
    VALUE h = rb_hash_new();
    rb_hash_aset(h, ID2SYM(rb_intern("filename")), rb_utf8_str_new_cstr(fn));
    rb_hash_aset(h, ID2SYM(rb_intern("url")), rb_utf8_str_new_cstr(url));
    rb_hash_aset(h, ID2SYM(rb_intern("size")), rb_utf8_str_new_cstr(sz));
    return h;
}
static VALUE rb_aux_registry_default_bundle_info(VALUE self, VALUE backend) {
    char out_be[256] = {0}, lic[256] = {0};
    int32_t req = 0;
    int rc = crispasr_registry_default_bundle_info_abi(StringValueCStr(backend), out_be, sizeof(out_be), lic,
                                                       sizeof(lic), &req);
    if (rc < 0)
        rb_raise(rb_eRuntimeError, "registry_default_bundle_info failed (rc=%d)", rc);
    VALUE h = rb_hash_new();
    rb_hash_aset(h, ID2SYM(rb_intern("backend")), rb_utf8_str_new_cstr(out_be));
    rb_hash_aset(h, ID2SYM(rb_intern("license")), rb_utf8_str_new_cstr(lic));
    rb_hash_aset(h, ID2SYM(rb_intern("requires_acceptance")), req ? Qtrue : Qfalse);
    rb_hash_aset(h, ID2SYM(rb_intern("count")), INT2NUM(rc));
    return h;
}
static VALUE rb_aux_registry_default_bundle_artifact(VALUE self, VALUE backend, VALUE idx) {
    char fn[1024] = {0}, url[2048] = {0}, sz[256] = {0};
    int32_t kind = 0;
    int rc = crispasr_registry_default_bundle_artifact_abi(StringValueCStr(backend), NUM2INT(idx), &kind, fn,
                                                           sizeof(fn), url, sizeof(url), sz, sizeof(sz));
    if (rc != 0)
        rb_raise(rb_eRuntimeError, "registry_default_bundle_artifact failed (rc=%d)", rc);
    VALUE h = rb_hash_new();
    rb_hash_aset(h, ID2SYM(rb_intern("kind")), INT2NUM(kind));
    rb_hash_aset(h, ID2SYM(rb_intern("filename")), rb_utf8_str_new_cstr(fn));
    rb_hash_aset(h, ID2SYM(rb_intern("url")), rb_utf8_str_new_cstr(url));
    rb_hash_aset(h, ID2SYM(rb_intern("size")), rb_utf8_str_new_cstr(sz));
    return h;
}
static VALUE rb_aux_cache_ensure(VALUE self, VALUE filename, VALUE url) {
    char out[1024] = {0};
    int rc = crispasr_cache_ensure_file_abi(StringValueCStr(filename), StringValueCStr(url), 0, NULL, out, sizeof(out));
    if (rc != 0)
        rb_raise(rb_eRuntimeError, "cache_ensure_file failed (rc=%d)", rc);
    return rb_utf8_str_new_cstr(out);
}
static VALUE rb_aux_cache_dir(VALUE self, VALUE override) {
    char out[1024] = {0};
    const char* ov = NIL_P(override) ? NULL : StringValueCStr(override);
    int rc = crispasr_cache_dir_abi(ov, out, sizeof(out));
    if (rc != 0)
        rb_raise(rb_eRuntimeError, "cache_dir failed (rc=%d)", rc);
    return rb_utf8_str_new_cstr(out);
}


// Mirror of the stable PODs in crispasr_c_api.cpp; the contract probe compiles
// the native definitions independently to check every field and time unit.
struct rb_aux_diarize_segment {
    int64_t t0_cs, t1_cs;
    int32_t speaker, pad;
};
struct rb_aux_diarize_options {
    int32_t method, threads;
    int64_t offset;
    const char* model;
    const char* embedder;
    int32_t min, max, count, pad;
};
static VALUE rb_aux_hash_field(VALUE hash, const char* name) {
    VALUE value = rb_hash_aref(hash, ID2SYM(rb_intern(name)));
    if (NIL_P(value))
        value = rb_hash_aref(hash, rb_str_new_cstr(name));
    return value;
}
static VALUE rb_aux_diarize_result(struct rb_aux_diarize_segment* segments, int n) {
    VALUE out = rb_ary_new_capa(n);
    for (int i = 0; i < n; ++i) {
        VALUE item = rb_hash_new();
        rb_hash_aset(item, ID2SYM(rb_intern("t0")), LL2NUM(segments[i].t0_cs));
        rb_hash_aset(item, ID2SYM(rb_intern("t1")), LL2NUM(segments[i].t1_cs));
        rb_hash_aset(item, ID2SYM(rb_intern("speaker")), INT2NUM(segments[i].speaker));
        rb_ary_push(out, item);
    }
    return out;
}
static VALUE rb_aux_diarize(int argc, VALUE* argv, VALUE self) {
    VALUE left, right, segments, options;
    rb_scan_args(argc, argv, "31", &left, &right, &segments, &options);
    Check_Type(segments, T_ARRAY);
    if (NIL_P(options))
        options = rb_hash_new();
    Check_Type(options, T_HASH);
    VALUE lstore = rb_aux_pcm_storage(left), rstore = Qnil;
    if (!NIL_P(right)) {
        rstore = rb_aux_pcm_storage(right);
        if (RARRAY_LEN(left) != RARRAY_LEN(right))
            rb_raise(rb_eArgError, "Stereo channel lengths differ");
    }
    long n = RARRAY_LEN(segments);
    if (n < 1 || n > INT_MAX)
        rb_raise(rb_eArgError, "Nonempty segment array required");
    VALUE store = rb_str_new(NULL, n * sizeof(struct rb_aux_diarize_segment));
    struct rb_aux_diarize_segment* native = (struct rb_aux_diarize_segment*)RSTRING_PTR(store);
    for (long i = 0; i < n; ++i) {
        VALUE input = rb_ary_entry(segments, i);
        Check_Type(input, T_HASH);
        native[i].t0_cs = NUM2LL(rb_aux_hash_field(input, "t0"));
        native[i].t1_cs = NUM2LL(rb_aux_hash_field(input, "t1"));
        if (native[i].t0_cs < 0 || native[i].t1_cs < native[i].t0_cs)
            rb_raise(rb_eArgError, "Invalid segment times");
        native[i].speaker = -1;
        native[i].pad = 0;
    }
    struct rb_aux_diarize_options opts = {0, 4, 0, NULL, NULL, 0, 0, 0, 0};
    VALUE v, model = Qnil, embedder = Qnil;
#define RB_DIA_INT(key, field)                                                                                         \
    v = rb_aux_hash_field(options, key);                                                                               \
    if (!NIL_P(v))                                                                                                     \
        opts.field = NUM2INT(v);
    RB_DIA_INT("method", method)
    RB_DIA_INT("n_threads", threads)
    RB_DIA_INT("min_speakers", min)
    RB_DIA_INT("max_speakers", max)
    RB_DIA_INT("num_speakers", count)
#undef RB_DIA_INT
    if (opts.method < 0 || opts.method > 5)
        rb_raise(rb_eArgError, "Invalid diarization method");
    v = rb_aux_hash_field(options, "slice_t0_cs");
    if (!NIL_P(v))
        opts.offset = NUM2LL(v);
    model = rb_aux_hash_field(options, "model_path");
    if (!NIL_P(model))
        opts.model = StringValueCStr(model);
    embedder = rb_aux_hash_field(options, "embedder_path");
    if (!NIL_P(embedder))
        opts.embedder = StringValueCStr(embedder);
    VALUE turns = Qnil;
    int32_t needed = 0;
    int cap = 0, rc = 0;
    for (int attempt = 0; attempt < 4; ++attempt) {
        rc = crispasr_diarize_segments_turns_abi(
            (float*)RSTRING_PTR(lstore), NIL_P(rstore) ? NULL : (float*)RSTRING_PTR(rstore), RARRAY_LEN(left),
            NIL_P(right) ? 0 : 1, native, n, &opts, NIL_P(turns) ? NULL : RSTRING_PTR(turns), cap, &needed);
        if (rc != 0 && rc != 2)
            rb_raise(rb_eRuntimeError, "Diarization failed (rc=%d)", rc);
        if (needed < 0 || (rc == 2 && needed <= cap))
            rb_raise(rb_eRuntimeError, "Invalid turn count/status");
        if (rc == 0 && needed <= cap)
            break;
        cap = needed;
        turns = rb_str_new(NULL, (long)cap * sizeof(struct rb_aux_diarize_segment));
    }
    if (rc != 0 || needed > cap)
        rb_raise(rb_eRuntimeError, "Diarization turn count did not stabilize");
    VALUE out = rb_hash_new();
    rb_hash_aset(out, ID2SYM(rb_intern("segments")), rb_aux_diarize_result(native, n));
    rb_hash_aset(
        out, ID2SYM(rb_intern("turns")),
        rb_aux_diarize_result(NIL_P(turns) ? NULL : (struct rb_aux_diarize_segment*)RSTRING_PTR(turns), needed));
    RB_GC_GUARD(lstore);
    RB_GC_GUARD(rstore);
    RB_GC_GUARD(store);
    RB_GC_GUARD(turns);
    RB_GC_GUARD(model);
    RB_GC_GUARD(embedder);
    return out;
}
/* Chat */
static VALUE mChat = Qnil;
static VALUE rb_chat_get_field(VALUE hash, const char* key) {
    VALUE v = rb_hash_aref(hash, ID2SYM(rb_intern(key)));
    if (NIL_P(v)) {
        VALUE str = rb_str_new_cstr(key);
        v = rb_hash_aref(hash, str);
    }
    return v;
}

// Opaque Ruby-owned data payloads have stable native addresses across GC
// compaction. Ruby String storage may move/re-embed even while it is rooted.
static VALUE rb_chat_buffer(long size) {
    return rb_data_object_zalloc(rb_cObject, size, NULL, RUBY_DEFAULT_FREE);
}
static const char* rb_chat_owned_cstring(VALUE input, VALUE roots) {
    StringValue(input);
    StringValueCStr(input);
    long n = RSTRING_LEN(input);
    VALUE copy = rb_chat_buffer(n + 1);
    char* bytes = (char*)DATA_PTR(copy);
    memcpy(bytes, RSTRING_PTR(input), n);
    bytes[n] = 0;
    rb_ary_push(roots, copy);
    return bytes;
}
static VALUE rb_chat_messages(VALUE messages, VALUE roots) {
    Check_Type(messages, T_ARRAY);
    long n = RARRAY_LEN(messages);
    if (n > LONG_MAX / (long)sizeof(crispasr_chat_message))
        rb_raise(rb_eArgError, "Too many chat messages");
    VALUE storage = rb_chat_buffer(n * sizeof(crispasr_chat_message));
    crispasr_chat_message* native = (crispasr_chat_message*)DATA_PTR(storage);
    for (long i = 0; i < n; ++i) {
        VALUE item = rb_ary_entry(messages, i);
        Check_Type(item, T_HASH);
        native[i].role = rb_chat_owned_cstring(rb_chat_get_field(item, "role"), roots);
        native[i].content = rb_chat_owned_cstring(rb_chat_get_field(item, "content"), roots);
    }
    return storage;
}
struct rb_chat_string_copy {
    const char* text;
};
static VALUE rb_chat_copy_text(VALUE raw) {
    return rb_utf8_str_new_cstr(((struct rb_chat_string_copy*)raw)->text);
}
static void rb_chat_ensure_live(void) {
    if (rb_chat_live == Qnil) {
        rb_chat_live = rb_hash_new();
        rb_global_variable(&rb_chat_live);
    }
}
static void rb_chat_release_owner(void* native) {
    crispasr_chat_close((crispasr_chat_session_t)native);
}
static VALUE rb_chat_track(crispasr_chat_session_t native) {
    return rb_extra_track_owner(&rb_chat_live, (void*)native, rb_chat_release_owner);
}
static crispasr_chat_session_t rb_chat_native(VALUE handle) {
    return (crispasr_chat_session_t)rb_extra_native_pointer(rb_chat_live, handle, "Chat session");
}
static void rb_chat_untrack(VALUE h) {
    if (rb_chat_live != Qnil)
        rb_hash_delete(rb_chat_live, h);
}
static void rb_chat_check_live(VALUE h) {
    if (rb_extra_in_stream)
        rb_raise(rb_eRuntimeError, "Chat re-entry from callback is not allowed");
    if (rb_chat_live == Qnil || NIL_P(rb_hash_aref(rb_chat_live, h)))
        rb_raise(rb_eRuntimeError, "chat session is closed");
}

static void rb_chat_open_options(VALUE params, crispasr_chat_open_params* p, VALUE roots) {
    crispasr_chat_open_params_default(p);
    if (NIL_P(params))
        return;
    Check_Type(params, T_HASH);
    VALUE v;
#define RB_CHAT_OPEN_INT(field)                                                                                        \
    v = rb_chat_get_field(params, #field);                                                                             \
    if (!NIL_P(v))                                                                                                     \
        p->field = NUM2INT(v);
    RB_CHAT_OPEN_INT(n_threads)
    RB_CHAT_OPEN_INT(n_threads_batch)
    RB_CHAT_OPEN_INT(n_ctx)
    RB_CHAT_OPEN_INT(n_batch)
    RB_CHAT_OPEN_INT(n_ubatch)
    RB_CHAT_OPEN_INT(n_gpu_layers)
#undef RB_CHAT_OPEN_INT
#define RB_CHAT_OPEN_BOOL(field)                                                                                       \
    v = rb_chat_get_field(params, #field);                                                                             \
    if (!NIL_P(v))                                                                                                     \
        p->field = RTEST(v);
    RB_CHAT_OPEN_BOOL(use_mmap)
    RB_CHAT_OPEN_BOOL(use_mlock)
    RB_CHAT_OPEN_BOOL(embeddings)
#undef RB_CHAT_OPEN_BOOL
    v = rb_chat_get_field(params, "chat_template");
    if (!NIL_P(v))
        p->chat_template = rb_chat_owned_cstring(v, roots);
}
static void rb_chat_generate_options(VALUE params, crispasr_chat_generate_params* p, VALUE* storage, VALUE roots) {
    crispasr_chat_generate_params_default(p);
    if (NIL_P(params))
        return;
    Check_Type(params, T_HASH);
    VALUE v;
#define RB_CHAT_INT(field)                                                                                             \
    v = rb_chat_get_field(params, #field);                                                                             \
    if (!NIL_P(v))                                                                                                     \
        p->field = NUM2INT(v);
    RB_CHAT_INT(max_tokens)
    RB_CHAT_INT(top_k)
    RB_CHAT_INT(repeat_last_n)
#undef RB_CHAT_INT
#define RB_CHAT_FLOAT(field)                                                                                           \
    v = rb_chat_get_field(params, #field);                                                                             \
    if (!NIL_P(v))                                                                                                     \
        p->field = NUM2DBL(v);
    RB_CHAT_FLOAT(temperature)
    RB_CHAT_FLOAT(top_p)
    RB_CHAT_FLOAT(min_p)
    RB_CHAT_FLOAT(repeat_penalty)
#undef RB_CHAT_FLOAT
    v = rb_chat_get_field(params, "seed");
    if (!NIL_P(v))
        p->seed = NUM2UINT(v);
    v = rb_chat_get_field(params, "prefill_only");
    if (!NIL_P(v))
        p->prefill_only = RTEST(v);
    v = rb_chat_get_field(params, "stop");
    if (!NIL_P(v)) {
        Check_Type(v, T_ARRAY);
        long n = RARRAY_LEN(v);
        *storage = rb_chat_buffer(n * sizeof(char*));
        for (long i = 0; i < n; ++i) {
            VALUE stop = rb_ary_entry(v, i);
            ((const char**)DATA_PTR(*storage))[i] = rb_chat_owned_cstring(stop, roots);
        }
        p->stop = (const char**)DATA_PTR(*storage);
        p->n_stop = n;
    }
}
static VALUE rb_chat_open(VALUE self, VALUE model_path, VALUE params) {
    VALUE roots = rb_ary_new();
    crispasr_chat_open_params p;
    rb_chat_open_options(params, &p, roots);
    crispasr_chat_error err = {0};
    crispasr_chat_session_t s = crispasr_chat_open(rb_chat_owned_cstring(model_path, roots), &p, &err);
    RB_GC_GUARD(roots);
    if (!s)
        rb_raise(rb_eRuntimeError, "chat_open failed: %s", err.message);
    return rb_chat_track(s);
}
static VALUE rb_chat_close(VALUE self, VALUE handle) {
    if (rb_extra_in_stream)
        rb_raise(rb_eRuntimeError, "Chat re-entry from callback is not allowed");
    rb_chat_ensure_live();
    if (NIL_P(rb_hash_aref(rb_chat_live, handle)))
        return Qnil;
    crispasr_chat_session_t s = rb_chat_native(handle);
    if (s) {
        // clear abort callback
        rb_aux_ensure_chat_callbacks();
        VALUE key = ULL2NUM((uintptr_t)s);
        rb_hash_delete(rb_aux_chat_callbacks, key);
        crispasr_chat_close(s);
    }
    rb_chat_untrack(handle);
    rb_aux_check_pending();
    return Qnil;
}
static VALUE rb_chat_reset(VALUE self, VALUE handle) {
    rb_chat_check_live(handle);
    crispasr_chat_session_t s = rb_chat_native(handle);
    crispasr_chat_error err = {0};
    int32_t rc = crispasr_chat_reset(s, &err);
    if (rc != 0)
        rb_raise(rb_eRuntimeError, "chat_reset failed: %s", err.message);
    rb_aux_check_pending();
    return Qnil;
}
static VALUE rb_chat_generate(VALUE self, VALUE handle, VALUE messages, VALUE params) {
    rb_chat_check_live(handle);
    crispasr_chat_session_t s = rb_chat_native(handle);
    VALUE roots = rb_ary_new();
    VALUE message_storage = rb_chat_messages(messages, roots);
    long n = RARRAY_LEN(messages);
    crispasr_chat_message* cms = (crispasr_chat_message*)DATA_PTR(message_storage);
    crispasr_chat_generate_params gp;
    VALUE stop_storage = Qnil;
    rb_chat_generate_options(params, &gp, &stop_storage, roots);
    crispasr_chat_error err = {0};
    rb_gc_register_address(&roots);
    rb_gc_register_address(&message_storage);
    rb_gc_register_address(&stop_storage);
    char* out = crispasr_chat_generate(s, cms, n, &gp, &err);
    rb_gc_unregister_address(&stop_storage);
    rb_gc_unregister_address(&message_storage);
    rb_gc_unregister_address(&roots);
    RB_GC_GUARD(stop_storage);
    RB_GC_GUARD(roots);
    RB_GC_GUARD(message_storage);
    if (rb_aux_pending_exc != Qnil) {
        if (out)
            crispasr_chat_string_free(out);
        rb_aux_check_pending();
    }
    if (!out)
        rb_raise(rb_eRuntimeError, "chat_generate failed: %s", err.message);
    struct rb_chat_string_copy copy = {out};
    int state = 0;
    VALUE str = rb_protect(rb_chat_copy_text, (VALUE)&copy, &state);
    crispasr_chat_string_free(out);
    if (state)
        rb_jump_tag(state);
    rb_aux_check_pending();
    return str;
}
struct chat_stream_data {
    VALUE proc;
    VALUE exc;
    int state;
};
struct chat_token_dispatch {
    VALUE proc;
    const char* chunk;
};
static VALUE chat_token_body(VALUE v) {
    struct chat_token_dispatch* a = (struct chat_token_dispatch*)v;
    VALUE s = rb_utf8_str_new_cstr(a->chunk ? a->chunk : "");
    struct rb_aux_call_args ca;
    ca.proc = a->proc;
    ca.nargs = 1;
    ca.args[0] = s;
    return rb_aux_call_proc_body((VALUE)&ca);
}
static void chat_on_token(const char* chunk, void* ud) {
    struct chat_stream_data* d = (struct chat_stream_data*)ud;
    if (d->state)
        return;
    struct chat_token_dispatch a = {d->proc, chunk};
    int state = 0;
    rb_extra_in_stream = 1;
    rb_protect(chat_token_body, (VALUE)&a, &state);
    rb_extra_in_stream = 0;
    if (state) {
        d->state = state;
        d->exc = rb_errinfo();
        rb_set_errinfo(Qnil);
    }
}
static VALUE rb_chat_generate_stream(VALUE self, VALUE handle, VALUE messages, VALUE params) {
    rb_chat_check_live(handle);
    crispasr_chat_session_t s = rb_chat_native(handle);
    if (!rb_block_given_p())
        rb_raise(rb_eArgError, "block required");
    VALUE proc = rb_block_proc();
    VALUE roots = rb_ary_new();
    VALUE message_storage = rb_chat_messages(messages, roots);
    long n = RARRAY_LEN(messages);
    crispasr_chat_message* cms = (crispasr_chat_message*)DATA_PTR(message_storage);
    crispasr_chat_generate_params gp;
    VALUE stop_storage = Qnil;
    rb_chat_generate_options(params, &gp, &stop_storage, roots);
    struct chat_stream_data d = {proc, Qnil, 0};
    crispasr_chat_error err = {0};
    rb_gc_register_address(&roots);
    rb_gc_register_address(&message_storage);
    rb_gc_register_address(&stop_storage);
    int32_t rc = crispasr_chat_generate_stream(s, cms, n, &gp, chat_on_token, &d, &err);
    rb_gc_unregister_address(&stop_storage);
    rb_gc_unregister_address(&message_storage);
    rb_gc_unregister_address(&roots);
    RB_GC_GUARD(stop_storage);
    RB_GC_GUARD(roots);
    RB_GC_GUARD(message_storage);
    if (d.state)
        rb_exc_raise(d.exc);
    if (rb_aux_pending_exc != Qnil)
        rb_aux_check_pending();
    if (rc != 0)
        rb_raise(rb_eRuntimeError, "chat_generate_stream failed: %s (rc=%d)", err.message, rc);
    rb_aux_check_pending();
    return Qnil;
}
static VALUE rb_chat_set_abort(VALUE self, VALUE handle, VALUE proc) {
    rb_chat_check_live(handle);
    crispasr_chat_session_t s = rb_chat_native(handle);
    rb_aux_ensure_chat_callbacks();
    VALUE key = ULL2NUM((uintptr_t)s);
    if (NIL_P(proc)) {
        rb_hash_delete(rb_aux_chat_callbacks, key);
        crispasr_chat_set_abort_callback(s, NULL, NULL);
    } else {
        rb_hash_aset(rb_aux_chat_callbacks, key, proc);
        crispasr_chat_set_abort_callback(s, rb_aux_chat_abort_tramp, (void*)s);
    }
    return Qnil;
}
static VALUE rb_chat_template(VALUE self, VALUE handle) {
    rb_chat_check_live(handle);
    crispasr_chat_session_t s = rb_chat_native(handle);
    const char* t = crispasr_chat_template_name(s);
    return rb_utf8_str_new_cstr(t ? t : "");
}
static VALUE rb_chat_n_ctx(VALUE self, VALUE handle) {
    rb_chat_check_live(handle);
    crispasr_chat_session_t s = rb_chat_native(handle);
    return INT2NUM(crispasr_chat_n_ctx(s));
}
static VALUE rb_chat_count_tokens(VALUE self, VALUE handle, VALUE messages) {
    rb_chat_check_live(handle);
    crispasr_chat_session_t s = rb_chat_native(handle);
    VALUE roots = rb_ary_new();
    VALUE message_storage = rb_chat_messages(messages, roots);
    long n = RARRAY_LEN(messages);
    crispasr_chat_message* cms = (crispasr_chat_message*)DATA_PTR(message_storage);
    crispasr_chat_error err = {0};
    int32_t rc = crispasr_chat_count_tokens(s, cms, n, &err);
    RB_GC_GUARD(roots);
    RB_GC_GUARD(message_storage);
    if (rc < 0)
        rb_raise(rb_eRuntimeError, "count_tokens failed: %s", err.message);
    return INT2NUM(rc);
}
static VALUE rb_chat_memory_estimate(VALUE self, VALUE path, VALUE params) {
    VALUE roots = rb_ary_new();
    crispasr_chat_open_params p;
    rb_chat_open_options(params, &p, roots);
    crispasr_chat_error err = {0};
    size_t sz = crispasr_chat_memory_estimate(rb_chat_owned_cstring(path, roots), &p, &err);
    RB_GC_GUARD(roots);
    if (sz == 0 && err.code != 0)
        rb_raise(rb_eRuntimeError, "memory_estimate failed: %s", err.message);
    return ULL2NUM(sz);
}
static VALUE rb_chat_ai_disclosure(VALUE self) {
    const char* t = crispasr_chat_ai_disclosure_text();
    return rb_utf8_str_new_cstr(t ? t : "");
}

/* Test-only direct callback routing probes. */
#ifdef CRISPASR_WRAPPER_PROBE
extern void probe_mic_mode(int);
extern int probe_mic_live(void);
extern int probe_mic_closes(void);
static VALUE rb_test_mic_mode(VALUE self, VALUE mode) {
    probe_mic_mode(NUM2INT(mode));
    return Qnil;
}
static VALUE rb_test_mic_live(VALUE self) {
    return INT2NUM(probe_mic_live());
}
static VALUE rb_test_mic_closes(VALUE self) {
    return INT2NUM(probe_mic_closes());
}
extern void probe_text_mode(int);
extern int probe_text_live(void);
extern int probe_text_contexts(void);
static VALUE rb_test_text_mode(VALUE self, VALUE mode) {
    probe_text_mode(NUM2INT(mode));
    return Qnil;
}
static VALUE rb_test_text_live(VALUE self) {
    return INT2NUM(probe_text_live());
}
static VALUE rb_test_text_contexts(VALUE self) {
    return INT2NUM(probe_text_contexts());
}
extern int probe_temperature_calls(void);
static VALUE rb_test_temperature_calls(VALUE self) {
    return INT2NUM(probe_temperature_calls());
}
extern int probe_parent_close_violations(void);
extern int probe_stream_closes(void);
static VALUE rb_test_parent_close_violations(VALUE self) {
    return INT2NUM(probe_parent_close_violations());
}
static VALUE rb_test_stream_closes(VALUE self) {
    return INT2NUM(probe_stream_closes());
}
extern void probe_speaker_mode(int);
extern int probe_speaker_live(void);
static VALUE rb_test_speaker_mode(VALUE self, VALUE mode) {
    probe_speaker_mode(NUM2INT(mode));
    return Qnil;
}
static VALUE rb_test_speaker_live(VALUE self) {
    return INT2NUM(probe_speaker_live());
}
extern void probe_db_mode(int);
extern int probe_db_live(void);
static VALUE rb_test_db_mode(VALUE self, VALUE mode) {
    probe_db_mode(NUM2INT(mode));
    return Qnil;
}
static VALUE rb_test_db_live(VALUE self) {
    return INT2NUM(probe_db_live());
}
extern void probe_titanet_force_reuse(int);
static VALUE rb_test_titanet_reuse(VALUE self, VALUE enabled) {
    probe_titanet_force_reuse(RTEST(enabled));
    return Qnil;
}
extern void probe_titanet_mode(int);
extern int probe_titanet_live(void);
static VALUE rb_test_titanet_mode(VALUE self, VALUE mode) {
    probe_titanet_mode(NUM2INT(mode));
    return Qnil;
}
static VALUE rb_test_titanet_live(VALUE self) {
    return INT2NUM(probe_titanet_live());
}
extern void probe_utility_mode(int);
extern int probe_utility_live(void);
static VALUE rb_test_audio_mode(VALUE self, VALUE mode) {
    probe_utility_mode(NUM2INT(mode));
    return Qnil;
}
static VALUE rb_test_audio_live(VALUE self) {
    return INT2NUM(probe_utility_live());
}
static VALUE rb_test_invoke_progress(VALUE self, VALUE handle) {
    VALUE native = rb_extra_sessions == Qnil || NIL_P(handle) ? Qnil : rb_hash_aref(rb_extra_sessions, handle);
    void* ud = NIL_P(handle) ? NULL : (void*)(uintptr_t)NUM2ULL(NIL_P(native) ? handle : native);
    rb_aux_progress_tramp(3, 6, ud);
    rb_aux_check_pending();
    return Qnil;
}
static VALUE rb_test_invoke_segment(VALUE self, VALUE handle) {
    VALUE native = rb_extra_sessions == Qnil || NIL_P(handle) ? Qnil : rb_hash_aref(rb_extra_sessions, handle);
    void* ud = NIL_P(handle) ? NULL : (void*)(uintptr_t)NUM2ULL(NIL_P(native) ? handle : native);
    rb_aux_segment_tramp("日本語", 123, 456, 7, ud);
    rb_aux_check_pending();
    return Qnil;
}
static VALUE rb_test_invoke_token(VALUE self, VALUE handle) {
    VALUE native = rb_extra_sessions == Qnil || NIL_P(handle) ? Qnil : rb_hash_aref(rb_extra_sessions, handle);
    void* ud = NIL_P(handle) ? NULL : (void*)(uintptr_t)NUM2ULL(NIL_P(native) ? handle : native);
    rb_aux_token_tramp("\xe6\x97", 9, ud);
    rb_aux_check_pending();
    return Qnil;
}

#endif

#include "ruby_crispasr_speaker.h"

// BEGIN GENERATED CONTEXT GUARDS
RB_CONTEXT_WRAPPER_1(rb_aux_backend_getter);
RB_CONTEXT_WRAPPER_5(rb_aux_score_texts);
RB_CONTEXT_WRAPPER_2(rb_aux_set_pcm_rate);
RB_CONTEXT_WRAPPER_3(rb_aux_set_att);
RB_CONTEXT_WRAPPER_2(rb_aux_set_audio_ctx);
RB_CONTEXT_WRAPPER_2(rb_aux_set_hikari_refresh_stride);
RB_CONTEXT_WRAPPER_2(rb_aux_set_grammar_strict);
RB_CONTEXT_WRAPPER_1(rb_aux_punc_init);
RB_CONTEXT_WRAPPER_2(rb_aux_punc_process);
RB_CONTEXT_WRAPPER_1(rb_aux_punc_free);
RB_CONTEXT_WRAPPER_1(rb_aux_truecase_init);
RB_CONTEXT_WRAPPER_2(rb_aux_truecase_process);
RB_CONTEXT_WRAPPER_1(rb_aux_truecase_free);
RB_CONTEXT_WRAPPER_1(rb_aux_pcs_init);
RB_CONTEXT_WRAPPER_2(rb_aux_pcs_process);
RB_CONTEXT_WRAPPER_1(rb_aux_pcs_free);
RB_CONTEXT_WRAPPER_2(rb_aux_set_progress_cb);
RB_CONTEXT_WRAPPER_2(rb_aux_set_segment_cb);
RB_CONTEXT_WRAPPER_2(rb_aux_set_token_cb);
RB_CONTEXT_WRAPPER_3(rb_aux_pitch);
RB_CONTEXT_WRAPPER_1(rb_aux_pitch_n_frames);
RB_CONTEXT_WRAPPER_1(rb_aux_pitch_sample_rate);
RB_CONTEXT_WRAPPER_2(rb_aux_pitch_frame);
RB_CONTEXT_WRAPPER_2(rb_aux_separate);
RB_CONTEXT_WRAPPER_1(rb_aux_separate_n_stems);
RB_CONTEXT_WRAPPER_1(rb_aux_separate_sample_rate);
RB_CONTEXT_WRAPPER_1(rb_aux_convert_sample_rate);
RB_CONTEXT_WRAPPER_1(rb_aux_piano_n_notes);
RB_CONTEXT_WRAPPER_1(rb_aux_chords_n_spans);
RB_CONTEXT_WRAPPER_1(rb_aux_beats_n_events);
RB_CONTEXT_WRAPPER_1(rb_aux_tab_n_frames);
RB_CONTEXT_WRAPPER_3(rb_aux_transcribe_vad);
RB_CONTEXT_WRAPPER_2(rb_chat_open);
RB_CONTEXT_WRAPPER_1(rb_chat_close);
RB_CONTEXT_WRAPPER_1(rb_chat_reset);
RB_CONTEXT_WRAPPER_3(rb_chat_generate);
RB_CONTEXT_WRAPPER_3(rb_chat_generate_stream);
RB_CONTEXT_WRAPPER_2(rb_chat_set_abort);
RB_CONTEXT_WRAPPER_1(rb_chat_template);
RB_CONTEXT_WRAPPER_1(rb_chat_n_ctx);
RB_CONTEXT_WRAPPER_2(rb_chat_count_tokens);
RB_CONTEXT_WRAPPER_2(rb_chat_memory_estimate);
RB_CONTEXT_WRAPPER_0(rb_chat_ai_disclosure);
// END GENERATED CONTEXT GUARDS
/* Registration helper called from Init */
static void rb_aux_init(void) {
    rb_speaker_aux_init();
    // CrispASR module already defined as mCrispASR
    // Audio / container
    rb_define_singleton_method(mCrispASR, "audio_load", rb_aux_audio_load, 1);
    rb_define_singleton_method(mCrispASR, "audio_load_at_rate", rb_aux_audio_load_at_rate, 2);
    rb_define_singleton_method(mCrispASR, "audio_load_stereo", rb_aux_audio_load_stereo, 1);
    rb_define_singleton_method(mCrispASR, "audio_resample", rb_aux_audio_resample, 3);
    rb_define_singleton_method(mCrispASR, "pcm_to_wav", rb_aux_pcm_to_wav, 2);
    rb_define_singleton_method(mCrispASR, "c2pa_sign", rb_aux_c2pa_sign, -1);
    rb_define_singleton_method(mCrispASR, "watermark_load_model", rb_aux_watermark_load_model, 1);
    rb_define_singleton_method(mCrispASR, "c_api_version", rb_aux_c_api_version, 0);
    rb_define_singleton_method(mCrispASR, "dart_helpers_version", rb_aux_dart_helpers_version, 0);
    rb_define_singleton_method(mCrispASR, "set_gpu_backend", rb_aux_set_gpu_backend, 1);
    rb_define_singleton_method(mCrispASR, "backend_caps", rb_aux_backend_caps, 1);
    rb_define_singleton_method(mCrispASR, "backend_caps_list", rb_aux_backend_caps_list, 0);
    rb_define_singleton_method(mCrispASR, "detect_backends_from_gguf", rb_aux_detect_backends, 1);
    rb_define_singleton_method(mCrispASR, "free_aligner_cache", rb_aux_free_aligner, 0);
    rb_define_singleton_method(mCrispASR, "get_progress", rb_aux_get_progress_aux, 0);
    rb_define_singleton_method(mCrispASR, "reset_progress", rb_aux_reset_progress_aux, 0);
    rb_define_singleton_method(mCrispASR, "get_streamed_segment_count", rb_aux_get_streamed_seg_count, 0);
    rb_define_singleton_method(mCrispASR, "drain_streamed_segments", rb_aux_drain_streamed_seg, 0);
    rb_define_singleton_method(mCrispASR, "reset_streamed_segments", rb_aux_reset_streamed_seg, 0);
    rb_define_singleton_method(mCrispASR, "get_streamed_token_count", rb_aux_get_streamed_tok_count, 0);
    rb_define_singleton_method(mCrispASR, "drain_streamed_tokens", rb_aux_drain_streamed_tok, 0);
    rb_define_singleton_method(mCrispASR, "reset_streamed_tokens", rb_aux_reset_streamed_tok, 0);
    // Registry
    rb_define_singleton_method(mCrispASR, "registry_lookup", rb_aux_registry_lookup, 1);
    rb_define_singleton_method(mCrispASR, "registry_lookup_by_filename", rb_aux_registry_lookup_by_filename, 1);
    rb_define_singleton_method(mCrispASR, "registry_default_bundle_info", rb_aux_registry_default_bundle_info, 1);
    rb_define_singleton_method(mCrispASR, "registry_default_bundle_artifact", rb_aux_registry_default_bundle_artifact,
                               2);
    rb_define_singleton_method(mCrispASR, "cache_ensure_file", rb_aux_cache_ensure, 2);
    rb_define_singleton_method(mCrispASR, "cache_dir", rb_aux_cache_dir, 1);
    rb_define_singleton_method(mCrispASR, "diarize_turns", rb_aux_diarize, -1);

    // Session additions
    rb_define_singleton_method(mSession, "backend", rb_context_guard_rb_aux_backend_getter, 1);
    rb_define_singleton_method(mSession, "score_texts", rb_context_guard_rb_aux_score_texts, 5);
    rb_define_singleton_method(mSession, "set_pcm_sample_rate", rb_context_guard_rb_aux_set_pcm_rate, 2);
    rb_define_singleton_method(mSession, "set_parakeet_att_context", rb_context_guard_rb_aux_set_att, 3);
    rb_define_singleton_method(mSession, "set_audio_ctx", rb_context_guard_rb_aux_set_audio_ctx, 2);
    rb_define_singleton_method(mSession, "set_hikari_refresh_stride", rb_context_guard_rb_aux_set_hikari_refresh_stride, 2);
    rb_define_singleton_method(mSession, "set_grammar_strict", rb_context_guard_rb_aux_set_grammar_strict, 2);
    rb_define_singleton_method(mSession, "punc_init", rb_context_guard_rb_aux_punc_init, 1);
    rb_define_singleton_method(mSession, "punc_process", rb_context_guard_rb_aux_punc_process, 2);
    rb_define_singleton_method(mSession, "punc_free", rb_context_guard_rb_aux_punc_free, 1);
    rb_define_singleton_method(mSession, "truecase_init", rb_context_guard_rb_aux_truecase_init, 1);
    rb_define_singleton_method(mSession, "truecase_process", rb_context_guard_rb_aux_truecase_process, 2);
    rb_define_singleton_method(mSession, "truecase_free", rb_context_guard_rb_aux_truecase_free, 1);
    rb_define_singleton_method(mSession, "pcs_init", rb_context_guard_rb_aux_pcs_init, 1);
    rb_define_singleton_method(mSession, "pcs_process", rb_context_guard_rb_aux_pcs_process, 2);
    rb_define_singleton_method(mSession, "pcs_free", rb_context_guard_rb_aux_pcs_free, 1);
    rb_define_singleton_method(mSession, "set_progress_callback", rb_context_guard_rb_aux_set_progress_cb, 2);
    rb_define_singleton_method(mSession, "set_segment_callback", rb_context_guard_rb_aux_set_segment_cb, 2);
    rb_define_singleton_method(mSession, "set_token_callback", rb_context_guard_rb_aux_set_token_cb, 2);
    rb_define_singleton_method(mSession, "pitch", rb_context_guard_rb_aux_pitch, 3);
    rb_define_singleton_method(mSession, "pitch_n_frames", rb_context_guard_rb_aux_pitch_n_frames, 1);
    rb_define_singleton_method(mSession, "pitch_sample_rate", rb_context_guard_rb_aux_pitch_sample_rate, 1);
    rb_define_singleton_method(mSession, "pitch_frame", rb_context_guard_rb_aux_pitch_frame, 2);
    rb_define_singleton_method(mSession, "separate", rb_context_guard_rb_aux_separate, 2);
    rb_define_singleton_method(mSession, "separate_n_stems", rb_context_guard_rb_aux_separate_n_stems, 1);
    rb_define_singleton_method(mSession, "separate_sample_rate", rb_context_guard_rb_aux_separate_sample_rate, 1);
    rb_define_singleton_method(mSession, "convert_sample_rate", rb_context_guard_rb_aux_convert_sample_rate, 1);
    rb_define_singleton_method(mSession, "piano_n_notes", rb_context_guard_rb_aux_piano_n_notes, 1);
    rb_define_singleton_method(mSession, "chords_n_spans", rb_context_guard_rb_aux_chords_n_spans, 1);
    rb_define_singleton_method(mSession, "beats_n_events", rb_context_guard_rb_aux_beats_n_events, 1);
    rb_define_singleton_method(mSession, "tab_n_frames", rb_context_guard_rb_aux_tab_n_frames, 1);
    rb_define_singleton_method(mSession, "transcribe_vad", rb_context_guard_rb_aux_transcribe_vad, 3);
#ifdef CRISPASR_WRAPPER_PROBE
    rb_define_singleton_method(mCrispASR, "test_mic_mode", rb_test_mic_mode, 1);
    rb_define_singleton_method(mCrispASR, "test_mic_live", rb_test_mic_live, 0);
    rb_define_singleton_method(mCrispASR, "test_mic_closes", rb_test_mic_closes, 0);
    rb_define_singleton_method(mCrispASR, "test_audio_mode", rb_test_audio_mode, 1);
    rb_define_singleton_method(mCrispASR, "test_audio_live", rb_test_audio_live, 0);
    rb_define_singleton_method(mCrispASR, "test_text_mode", rb_test_text_mode, 1);
    rb_define_singleton_method(mCrispASR, "test_text_live", rb_test_text_live, 0);
    rb_define_singleton_method(mCrispASR, "test_text_contexts", rb_test_text_contexts, 0);
    rb_define_singleton_method(mCrispASR, "test_temperature_calls", rb_test_temperature_calls, 0);
    rb_define_singleton_method(mCrispASR, "test_parent_close_violations", rb_test_parent_close_violations, 0);
    rb_define_singleton_method(mCrispASR, "test_stream_closes", rb_test_stream_closes, 0);
    rb_define_singleton_method(mCrispASR, "test_speaker_mode", rb_test_speaker_mode, 1);
    rb_define_singleton_method(mCrispASR, "test_speaker_live", rb_test_speaker_live, 0);
    rb_define_singleton_method(mCrispASR, "test_db_mode", rb_test_db_mode, 1);
    rb_define_singleton_method(mCrispASR, "test_db_live", rb_test_db_live, 0);
    rb_define_singleton_method(mCrispASR, "test_titanet_reuse", rb_test_titanet_reuse, 1);
    rb_define_singleton_method(mCrispASR, "test_titanet_mode", rb_test_titanet_mode, 1);
    rb_define_singleton_method(mCrispASR, "test_titanet_live", rb_test_titanet_live, 0);
    // Test helpers (routing verification)
    rb_define_singleton_method(mCrispASR, "test_invoke_progress", rb_test_invoke_progress, 1);
    rb_define_singleton_method(mCrispASR, "test_invoke_segment", rb_test_invoke_segment, 1);
    rb_define_singleton_method(mCrispASR, "test_invoke_token", rb_test_invoke_token, 1);

#endif

    rb_define_singleton_method(mCrispASR, "diarize_segments", rb_aux_diarize, -1);

    // Chat
    mChat = rb_define_module_under(mCrispASR, "Chat");
    rb_define_singleton_method(mChat, "open", rb_context_guard_rb_chat_open, 2);
    rb_define_singleton_method(mChat, "close", rb_context_guard_rb_chat_close, 1);
    rb_define_singleton_method(mChat, "reset", rb_context_guard_rb_chat_reset, 1);
    rb_define_singleton_method(mChat, "generate", rb_context_guard_rb_chat_generate, 3);
    rb_define_singleton_method(mChat, "generate_stream", rb_context_guard_rb_chat_generate_stream, 3);
    rb_define_singleton_method(mChat, "set_abort_callback", rb_context_guard_rb_chat_set_abort, 2);
    rb_define_singleton_method(mChat, "template_name", rb_context_guard_rb_chat_template, 1);
    rb_define_singleton_method(mChat, "n_ctx", rb_context_guard_rb_chat_n_ctx, 1);
    rb_define_singleton_method(mChat, "count_tokens", rb_context_guard_rb_chat_count_tokens, 2);
    rb_define_singleton_method(mChat, "memory_estimate", rb_context_guard_rb_chat_memory_estimate, 2);
    rb_define_singleton_method(mChat, "ai_disclosure_text", rb_context_guard_rb_chat_ai_disclosure, 0);
}
