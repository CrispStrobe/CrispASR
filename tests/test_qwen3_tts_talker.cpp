// test_qwen3_tts_talker.cpp — sanity check the Qwen3-TTS talker forward.
//
// Usage:
//   ./build/tests/test_qwen3_tts_talker <model.gguf> "<text>"
//
// Loads the talker, runs greedy AR decode of codebook-0, prints the
// resulting code stream. Not auto-registered with ctest — needs a
// real GGUF + the converter on disk.

#include "qwen3_tts.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr,
                "usage: %s <model.gguf> <voice-pack.gguf> <text> [max_frames]\n"
                "\n"
                "Synthesise text → 16-codebook code stream via talker + code_predictor.\n"
                "Voice pack provides the spk_embedding + ref_code (bake with\n"
                "models/bake-qwen3-tts-voice-pack.py).\n",
                argv[0]);
        return 1;
    }
    const char* model = argv[1];
    const char* voice = argv[2];
    const char* text = argv[3];
    const int max_frames = argc >= 5 ? atoi(argv[4]) : 0;

    qwen3_tts_context_params p = qwen3_tts_context_default_params();
    p.verbosity = 1;
    p.use_gpu = true;
    p.max_codec_steps = max_frames;

    qwen3_tts_context* ctx = qwen3_tts_init_from_file(model, p);
    if (!ctx) {
        fprintf(stderr, "init failed\n");
        return 2;
    }
    if (qwen3_tts_load_voice_pack(ctx, voice) != 0) {
        fprintf(stderr, "voice pack load failed\n");
        qwen3_tts_free(ctx);
        return 4;
    }

    int n = 0;
    int32_t* codes = qwen3_tts_synthesize_codes(ctx, text, &n);
    if (!codes) {
        fprintf(stderr, "synth failed\n");
        qwen3_tts_free(ctx);
        return 3;
    }

    // Output: 16 codes per frame. Pretty-print as one line per frame.
    const int n_frames = n / 16;
    fprintf(stderr, "got %d codes (%d frames × 16 codebooks)\n", n, n_frames);
    for (int f = 0; f < n_frames; f++) {
        fprintf(stdout, "frame %3d:", f);
        for (int cb = 0; cb < 16; cb++)
            fprintf(stdout, " %4d", codes[f * 16 + cb]);
        fprintf(stdout, "\n");
    }

    qwen3_tts_codes_free(codes);
    qwen3_tts_free(ctx);
    return 0;
}
