// Model-free generic speaker/cache transport. PODs come directly from runtime source.
#include "crispasr_session.h"
#include "runtime-abi-layout.h"
#include <stdlib.h>
#include <string.h>
static int speaker_mode, speaker_live, cluster_calls;
int probe_cluster_calls(void) {
    return cluster_calls;
}
void probe_speaker_mode(int mode) {
    speaker_mode = mode;
}
int probe_speaker_live(void) {
    return speaker_live;
}
void* crispasr_speaker_embedder_make_abi(const char* model, int32_t threads, const char* cache) {
    if (speaker_mode == 1 || !model || strcmp(model, "embed") || (threads != 2 && threads != 4) ||
        (cache && strcmp(cache, "cache")))
        return NULL;
    void* out = malloc(1);
    if (out)
        ++speaker_live;
    return out;
}
void crispasr_speaker_embedder_free_abi(void* handle) {
    if (handle) {
        --speaker_live;
        free(handle);
    }
}
int32_t crispasr_speaker_embedder_dim_abi(const void* handle) {
    return !handle || speaker_mode == 3 ? 0 : 3;
}
const char* crispasr_speaker_embedder_name_abi(const void* handle) {
    return "generic";
}
int32_t crispasr_speaker_embedder_embed_abi(void* handle, const float* pcm, int32_t n, float* out) {
    if (!handle || !pcm || n != 3 || pcm[0] != 1 || pcm[2] != 5 || speaker_mode == 2)
        return 0;
    out[0] = .25f;
    out[1] = .5f;
    out[2] = .75f;
    return speaker_mode == 4 ? 2 : speaker_mode == 8 ? -1 : 1;
}
int32_t crispasr_speaker_cluster_abi(const float* emb, int32_t n, int32_t dim, float threshold, int32_t max,
                                     int32_t* labels) {
    ++cluster_calls;
    if (!emb || n != 2 || dim != 3 || threshold != .6f || max != 3 || emb[0] != 1 || emb[5] != 6 || speaker_mode == 2)
        return -1;
    labels[0] = speaker_mode == 9 ? -1 : speaker_mode == 7 ? 2 : 0;
    labels[1] = speaker_mode == 9 ? -1 : 1;
    return speaker_mode == 9 ? 0 : speaker_mode == 6 ? 3 : 2;
}
void* crispasr_pyannote_cache_compute_abi(const float* pcm, int32_t n, const char* model, int32_t threads) {
    if (speaker_mode == 1 || !pcm || n != 3 || pcm[0] != 1 || pcm[2] != 5 || !model || strcmp(model, "seg") ||
        (threads != 2 && threads != 4))
        return NULL;
    void* out = malloc(1);
    if (out)
        ++speaker_live;
    return out;
}
void crispasr_pyannote_cache_free_abi(void* handle) {
    if (handle) {
        --speaker_live;
        free(handle);
    }
}
int32_t crispasr_pyannote_cache_apply_abi(const void* handle, int64_t offset, crispasr_diarize_seg_abi* segs,
                                          int32_t n) {
    if (!handle || offset != 125 || !segs || n != 2 || segs[0].t0_cs != 125 || segs[0].t1_cs != 175 ||
        segs[1].t0_cs != 200 || segs[1].t1_cs != 250)
        return -1;
    if (speaker_mode == 2)
        return 1;
    if (speaker_mode == 5)
        return 2;
    segs[0].speaker = 2;
    segs[1].speaker = 1;
    return 0;
}
