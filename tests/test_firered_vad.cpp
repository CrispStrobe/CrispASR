// Quick test for FireRedVAD
#include "firered_vad.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <vad.gguf> <audio.raw>\n  (audio.raw = float32 16kHz mono)\n", argv[0]);
        return 1;
    }
    firered_vad_context* ctx = firered_vad_init(argv[1]);
    if (!ctx) { fprintf(stderr, "Failed to load VAD model\n"); return 1; }
    fprintf(stderr, "VAD model loaded\n");

    FILE* f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "Failed to open audio\n"); return 1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    int n = sz / sizeof(float);
    std::vector<float> samples(n);
    if (fread(samples.data(), sizeof(float), n, f) != (size_t)n) { fclose(f); return 1; }
    fclose(f);
    fprintf(stderr, "Audio: %d samples (%.1f s)\n", n, n / 16000.0f);

    firered_vad_segment* segs = nullptr;
    int n_segs = 0;
    firered_vad_detect(ctx, samples.data(), n, &segs, &n_segs, 0.4f, 0.2f, 0.2f);

    printf("Detected %d speech segments:\n", n_segs);
    for (int i = 0; i < n_segs; i++)
        printf("  [%.2f - %.2f]\n", segs[i].start_sec, segs[i].end_sec);

    if (segs) free(segs);
    firered_vad_free(ctx);
    return 0;
}
