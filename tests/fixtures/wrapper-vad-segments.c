// Model-free legacy VAD ABI: native spans are fractional centiseconds.
#include "crispasr.h"
#include <stdlib.h>
#include <string.h>

static float* vad_segments_allocation;
static int vad_segments_frees;
int probe_vad_segments_live(void) {
    return vad_segments_allocation != NULL;
}
int probe_vad_segments_frees(void) {
    return vad_segments_frees;
}
int crispasr_vad_segments(const char* model, const float* pcm, int n, int rate, float threshold, int min_speech,
                          int min_silence, int threads, bool gpu, float** spans) {
    if (vad_segments_allocation || !model || !pcm || n != 3 || pcm[0] != 1 || pcm[1] != 3 || pcm[2] != 5 ||
        rate != 16000 || threshold != .375f || min_speech != 123 || min_silence != 456 || threads != 2 || gpu)
        abort();
    *spans = NULL;
    if (!strcmp(model, "empty"))
        return 0;
    if (!strcmp(model, "error"))
        return -7;
    if (strcmp(model, "vad"))
        abort();
    const float centiseconds[] = {125.5f, 250.25f, 300.75f, 903.75f};
    vad_segments_allocation = malloc(sizeof(centiseconds));
    if (!vad_segments_allocation)
        abort();
    memcpy(vad_segments_allocation, centiseconds, sizeof(centiseconds));
    *spans = vad_segments_allocation;
    return 2;
}
void crispasr_vad_free(float* spans) {
    if (!spans || spans != vad_segments_allocation)
        abort();
    memset(spans, 0, 4 * sizeof(float));
    free(spans);
    vad_segments_allocation = NULL;
    ++vad_segments_frees;
}
