/* Included after the shared opaque-ticket helpers. Native producers never call Ruby. */
#define MIC_RING_CAP 32
#ifdef CRISPASR_WRAPPER_PROBE
#include <stdatomic.h>
static atomic_int mic_live_chunks;
extern int probe_mic_copy_failure(void);
#endif
static float* mic_chunk_allocate(size_t n) {
    float* p = malloc(n);
#ifdef CRISPASR_WRAPPER_PROBE
    if (p)
        atomic_fetch_add(&mic_live_chunks, 1);
#endif
    return p;
}
static void mic_chunk_free(float* p) {
#ifdef CRISPASR_WRAPPER_PROBE
    if (p)
        atomic_fetch_sub(&mic_live_chunks, 1);
#endif
    free(p);
}
static VALUE rb_mic_handles = Qnil;
typedef struct mic_handle {
    pthread_mutex_t mu;
    pthread_cond_t cv;
    float* bufs[MIC_RING_CAP];
    int lens[MIC_RING_CAP];
    int head, tail, closed; // protected by mu
    float* pending;         // owned by pump, including async Ruby exception exits
    VALUE user_proc, pump_thread, ticket;
    struct crispasr_mic* mic;
} mic_handle_t;
static void mic_native_cb(const float* pcm, int n, void* ud) {
    mic_handle_t* h = ud;
    if (n <= 0 || !pcm || (size_t)n > SIZE_MAX / sizeof(float))
        return;
    pthread_mutex_lock(&h->mu);
    if (!h->closed) {
        int next = (h->tail + 1) % MIC_RING_CAP;
        float* copy = mic_chunk_allocate(sizeof(float) * (size_t)n);
        if (copy) {
            if (next == h->head) {
                mic_chunk_free(h->bufs[h->head]);
                h->bufs[h->head] = NULL;
                h->head = (h->head + 1) % MIC_RING_CAP;
            }
            memcpy(copy, pcm, sizeof(float) * (size_t)n);
            h->bufs[h->tail] = copy;
            h->lens[h->tail] = n;
            h->tail = next;
            pthread_cond_signal(&h->cv);
        }
    }
    pthread_mutex_unlock(&h->mu);
}
struct mic_dequeue {
    mic_handle_t* h;
    int n, closed;
};
static void* mic_wait_blocking(void* raw) {
    struct mic_dequeue* d = raw;
    mic_handle_t* h = d->h;
    pthread_mutex_lock(&h->mu);
    while (h->head == h->tail && !h->closed)
        pthread_cond_wait(&h->cv, &h->mu);
    if (h->closed) {
        d->closed = 1;
    } else {
        h->pending = h->bufs[h->head];
        d->n = h->lens[h->head];
        h->bufs[h->head] = NULL;
        h->head = (h->head + 1) % MIC_RING_CAP;
    }
    pthread_mutex_unlock(&h->mu);
    return NULL;
}
static void mic_wait_unblock(void* raw) {
    mic_handle_t* h = raw;
    pthread_mutex_lock(&h->mu);
    h->closed = 1;
    pthread_cond_broadcast(&h->cv);
    pthread_mutex_unlock(&h->mu);
}
static void mic_request_close(mic_handle_t* h) {
    if (h->mic) {
        struct crispasr_mic* native = h->mic;
        h->mic = NULL;
        // Joins native producers before their queue/userdata can be released.
        crispasr_mic_close(native);
    }
    mic_wait_unblock(h);
}
static VALUE mic_cleanup(VALUE raw) {
    mic_handle_t* h = (void*)(uintptr_t)raw;
    mic_request_close(h);
    if (!NIL_P(h->ticket))
        rb_hash_delete(rb_mic_handles, h->ticket);
    for (int i = 0; i < MIC_RING_CAP; ++i)
        mic_chunk_free(h->bufs[i]);
    mic_chunk_free(h->pending);
    rb_gc_unregister_address(&h->user_proc);
    rb_gc_unregister_address(&h->pump_thread);
    rb_gc_unregister_address(&h->ticket);
    pthread_cond_destroy(&h->cv);
    pthread_mutex_destroy(&h->mu);
    free(h);
    return Qnil;
}
struct mic_call_args {
    VALUE proc, arg;
};
static VALUE mic_call_proc(VALUE raw) {
    struct mic_call_args* a = (void*)(uintptr_t)raw;
    return rb_funcall(a->proc, rb_intern("call"), 1, a->arg);
}
static VALUE mic_callback_error(VALUE error, VALUE unused) {
    (void)error;
    (void)unused;
    return Qnil;
}
static VALUE mic_pump_body(VALUE raw) {
    mic_handle_t* h = (void*)(uintptr_t)raw;
    for (;;) {
        struct mic_dequeue d = {h, 0, 0};
        rb_thread_call_without_gvl(mic_wait_blocking, &d, mic_wait_unblock, h);
        if (d.closed)
            break;
            // pending remains owned by the outer ensure if allocating/copying raises.
#ifdef CRISPASR_WRAPPER_PROBE
        if (probe_mic_copy_failure())
            rb_raise(rb_eNoMemError, "Injected microphone copy failure");
#endif
        VALUE arr = rb_ary_new_capa(d.n);
        for (int i = 0; i < d.n; ++i)
            rb_ary_push(arr, DBL2NUM((double)h->pending[i]));
        mic_chunk_free(h->pending);
        h->pending = NULL;
        struct mic_call_args args = {h->user_proc, arr};
        // Ignore callback exceptions, but let Thread.kill reach the outer ensure.
        // rb_protect also catches non-exception tags and can swallow cancellation.
        rb_rescue2(mic_call_proc, (VALUE)(uintptr_t)&args, mic_callback_error, Qnil, rb_eException, (VALUE)0);
    }
    return Qnil;
}
static VALUE mic_pump_entry(void* raw) {
    return rb_ensure(mic_pump_body, (VALUE)(uintptr_t)raw, mic_cleanup, (VALUE)(uintptr_t)raw);
}
static VALUE mic_register_pump(VALUE raw) {
    mic_handle_t* h = (void*)(uintptr_t)raw;
    h->ticket = rb_extra_new_ticket();
    rb_hash_aset(rb_mic_handles, h->ticket, ULL2NUM((uintptr_t)h));
    h->pump_thread = rb_thread_create(mic_pump_entry, h);
    return h->ticket;
}
static VALUE rb_mic_open(int argc, VALUE* argv, VALUE self) {
    VALUE sample_rate, channels, block;
    rb_scan_args(argc, argv, "20&", &sample_rate, &channels, &block);
    if (NIL_P(block))
        rb_raise(rb_eArgError, "CrispASR::Mic.open requires a block");
    int rate = NUM2INT(sample_rate), count = NUM2INT(channels);
    if (rate <= 0 || (count != 1 && count != 2))
        rb_raise(rb_eArgError, "Positive sample rate and one or two channels required");
    if (rb_mic_handles == Qnil) {
        rb_mic_handles = rb_hash_new();
        rb_global_variable(&rb_mic_handles);
    }
    mic_handle_t* h = calloc(1, sizeof(*h));
    if (!h)
        rb_raise(rb_eNoMemError, "Microphone allocation failed");
    if (pthread_mutex_init(&h->mu, NULL)) {
        free(h);
        rb_raise(rb_eRuntimeError, "Microphone mutex initialization failed");
    }
    if (pthread_cond_init(&h->cv, NULL)) {
        pthread_mutex_destroy(&h->mu);
        free(h);
        rb_raise(rb_eRuntimeError, "Microphone condition initialization failed");
    }
    h->user_proc = block;
    h->pump_thread = Qnil;
    h->ticket = Qnil;
    rb_gc_register_address(&h->user_proc);
    rb_gc_register_address(&h->pump_thread);
    rb_gc_register_address(&h->ticket);
    h->mic = crispasr_mic_open(rate, count, mic_native_cb, h);
    if (!h->mic) {
        mic_cleanup((VALUE)(uintptr_t)h);
        rb_raise(rb_eRuntimeError, "crispasr_mic_open failed");
    }
    int state = 0;
    VALUE ticket = rb_protect(mic_register_pump, (VALUE)(uintptr_t)h, &state);
    if (state) {
        mic_cleanup((VALUE)(uintptr_t)h);
        rb_jump_tag(state);
    }
    return ticket;
}
static mic_handle_t* rb_mic_lookup(VALUE handle, int required) {
    if (!RB_INTEGER_TYPE_P(handle))
        rb_raise(rb_eTypeError, "Integer microphone handle required");
    VALUE native = rb_mic_handles == Qnil ? Qnil : rb_hash_aref(rb_mic_handles, handle);
    if (NIL_P(native)) {
        if (required)
            rb_raise(rb_eRuntimeError, "Microphone is closed or foreign");
        return NULL;
    }
    return (mic_handle_t*)(uintptr_t)NUM2ULL(native);
}
static VALUE rb_mic_start(VALUE self, VALUE handle) {
    mic_handle_t* h = rb_mic_lookup(handle, 1);
    int rc = crispasr_mic_start(h->mic);
    if (rc)
        rb_raise(rb_eRuntimeError, "crispasr_mic_start failed (rc=%d)", rc);
    return Qnil;
}
static VALUE rb_mic_stop(VALUE self, VALUE handle) {
    mic_handle_t* h = rb_mic_lookup(handle, 0);
    if (h) {
        int rc = crispasr_mic_stop(h->mic);
        if (rc)
            rb_raise(rb_eRuntimeError, "crispasr_mic_stop failed (rc=%d)", rc);
    }
    return Qnil;
}
static VALUE rb_mic_close(VALUE self, VALUE handle) {
    mic_handle_t* h = rb_mic_lookup(handle, 0);
    if (!h)
        return Qnil;
    // Retire the ticket before join releases the GVL. A second closer cannot
    // touch the record, and native allocation address reuse cannot revive it.
    rb_hash_delete(rb_mic_handles, handle);
    mic_request_close(h);
    VALUE pump = h->pump_thread;
    if (pump != rb_thread_current())
        rb_funcall(pump, rb_intern("join"), 0);
    // Only the pump's ensure releases storage. Callback-close never self-joins.
    return Qnil;
}
static VALUE rb_mic_default_device_name(VALUE self) {
    const char* name = crispasr_mic_default_device_name();
    return rb_utf8_str_new_cstr(name ? name : "");
}

#ifdef CRISPASR_WRAPPER_PROBE
static VALUE rb_test_mic_pump(VALUE self, VALUE handle) {
    return rb_mic_lookup(handle, 1)->pump_thread;
}
static VALUE rb_test_mic_chunks(VALUE self) {
    return INT2NUM(atomic_load(&mic_live_chunks));
}
#endif
