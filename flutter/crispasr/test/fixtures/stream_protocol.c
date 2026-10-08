// Model-free C ABI fixture: buffering, asynchronous output, long UTF-8 and errors.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
struct stream { int mode; int counter; int feeds; };
void* crispasr_session_open(const char* path, int threads) {
    (void)threads;
    return (void*)(intptr_t)(strcmp(path, "errors") == 0 ? 2 : 1);
}
const char* crispasr_session_backend(void* s) { (void)s; return "fixture"; }
int crispasr_session_stream_kind(void* s) { (void)s; return 2; }
void crispasr_session_close(void* s) { (void)s; }
void* crispasr_session_stream_open(void* s, int threads, int step, int length,
                                  int keep, const char* lang, int translate) {
    (void)threads; (void)step; (void)length; (void)keep; (void)translate;
    if (!lang || strcmp(lang, "de") != 0) return NULL;
    struct stream* st = calloc(1, sizeof(*st));
    st->mode = (int)(intptr_t)s;
    return st;
}
int crispasr_stream_feed(struct stream* s, const float* pcm, int n) {
    (void)pcm; (void)n;
    s->feeds++;
    // Output becomes readable while feed still reports buffering (async backend).
    if (s->feeds == 2) s->counter++;
    return 0;
}
int crispasr_stream_flush(struct stream* s) {
    if (s->mode == 2) return -7;
    s->counter++;
    return 1;
}
int crispasr_stream_get_text(struct stream* s, char* out, int cap, double* t0,
                            double* t1, int64_t* counter) {
    *t0 = 0; *t1 = s->feeds * .1; *counter = s->counter;
    if (!s->counter) { out[0] = 0; return 0; }
    // 6000 UTF-8 bytes; the old 4096-byte allocation split a German umlaut.
    const char* word = "ä";
    int length = 6000;
    for (int i = 0; i < length && i < cap - 1; i++) out[i] = word[i % 2];
    out[length < cap - 1 ? length : cap - 1] = 0;
    return length;
}
void crispasr_stream_set_live_decode(struct stream* s, int enabled) { (void)s; (void)enabled; }
void crispasr_stream_close(struct stream* s) { free(s); }
