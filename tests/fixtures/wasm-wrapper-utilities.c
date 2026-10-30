#include "crispasr_session.h"
static float stem[] = {1, 2, 3, 4, 5, 6};
int crispasr_session_separate(crispasr_session* s, const float* pcm, int frames) {
    return frames == 3 ? 1 : -1;
}
const float* crispasr_session_separate_stem(crispasr_session* s, int index, int* frames) {
    *frames = 3;
    return stem;
}
const char* crispasr_session_separate_stem_name(crispasr_session* s, int index) {
    return "voice";
}
int crispasr_session_separate_sample_rate(crispasr_session* s) {
    return 44100;
}
int crispasr_session_set_pcm_sample_rate(crispasr_session* s, int rate) {
    return rate > 0 ? 0 : -1;
}
int crispasr_session_set_parakeet_att_context(crispasr_session* s, int left, int right) {
    return 0;
}
int crispasr_session_set_audio_ctx(crispasr_session* s, int audio_ctx) {
    return 0;
}
int crispasr_session_set_hikari_refresh_stride(crispasr_session* s, int stride) {
    return 0;
}
int crispasr_session_set_grammar_strict(crispasr_session* s, int enabled) {
    return 0;
}
void crispasr_free_aligner_cache(void) {}
void crispasr_watermark_embed(float* pcm, int n, float alpha) {
    for (int i = 0; i < n; ++i)
        pcm[i] += alpha;
}
float crispasr_watermark_detect(const float* pcm, int n) {
    return .75f;
}

int crispasr_get_streamed_segment_count(void) {
    return 2;
}
int crispasr_get_streamed_token_count(void) {
    return 4;
}
