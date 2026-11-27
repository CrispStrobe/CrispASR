// test-vad-max-split.cpp — speech across a forced max-duration cut (#612 / silero-vad #813).
//
// With --vad-max-speech-duration-s, a segment that reaches the cap with no
// silence to cut at is cut at the current frame. The port used to stop the
// segment there, so speech continuing after the cut with a probability between
// neg_threshold and threshold was dropped until it climbed back over
// threshold. silero-vad v6.3 keeps it in the next segment unless the cut is
// inside a pause. Probabilities are injected directly so the expectation is
// exact. Run twice: normally, and with CRISPASR_VAD_LEGACY_MAX_SPLIT=1 as the
// negative control that must show the old gap.

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "crispasr.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const int W = 512; // samples per probability frame
static const int N = 200; // frames

static bool covered(whisper_vad_segments* s, int frame) {
    const float t = frame * W / 16000.0f * 100.0f; // centiseconds
    for (int i = 0; i < whisper_vad_segments_n_segments(s); ++i)
        if (whisper_vad_segments_get_segment_t0(s, i) <= t && t < whisper_vad_segments_get_segment_t1(s, i))
            return true;
    return false;
}

static whisper_vad_segments* run(whisper_vad_context* v, const std::vector<float>& p) {
    float* probs = whisper_vad_probs(v);
    for (int i = 0; i < N; ++i)
        probs[i] = p[i];
    whisper_vad_params vp = whisper_vad_default_params();
    vp.threshold = 0.5f; // neg_threshold 0.35
    vp.min_speech_duration_ms = 250;
    vp.min_silence_duration_ms = 100;
    vp.max_speech_duration_s = 2.0f; // 31488 samples = 61.5 frames
    vp.speech_pad_ms = 0;
    return whisper_vad_segments_from_probs(v, vp);
}

int main() {
    const char* e = std::getenv("CRISPASR_VAD_LEGACY_MAX_SPLIT");
    const bool legacy = e && *e && *e != '0';

    whisper_vad_context* v =
        whisper_vad_init_from_file_with_params(VAD_MODEL_PATH, whisper_vad_default_context_params());
    assert(v);
    std::vector<float> silence((size_t)N * W, 0.0f);
    const bool detected = whisper_vad_detect_speech(v, silence.data(), (int)silence.size());
    assert(detected);
    assert(whisper_vad_n_probs(v) == N);

    // Speech from frame 10; the cap is hit at frame 72 while the probability is
    // 0.42 (above neg_threshold, below threshold) until frame 120.
    std::vector<float> p(N, 0.9f);
    for (int i = 0; i < 10; ++i)
        p[i] = 0.05f;
    for (int i = 60; i <= 120; ++i)
        p[i] = 0.42f;
    whisper_vad_segments* s = run(v, p);
    for (int f = 74; f <= 118; ++f) {
        if (covered(s, f) == legacy) {
            fprintf(stderr, "frame %d: covered=%d, expected %d (legacy=%d)\n", f, covered(s, f), !legacy, legacy);
            return 1;
        }
    }
    whisper_vad_free_segments(s);

    // Cut inside a pause: frames 72..80 are silence. Both behaviours must leave
    // the pause uncovered and resume at frame 81.
    std::vector<float> q(N, 0.9f);
    for (int i = 0; i < 10; ++i)
        q[i] = 0.05f;
    for (int i = 72; i <= 80; ++i)
        q[i] = 0.05f;
    s = run(v, q);
    for (int f = 74; f <= 79; ++f)
        if (covered(s, f)) {
            fprintf(stderr, "pause frame %d covered\n", f);
            return 1;
        }
    assert(covered(s, 90));
    whisper_vad_free_segments(s);
    whisper_vad_free(v);
    printf("test-vad-max-split: ok (legacy=%d)\n", legacy);
    return 0;
}
