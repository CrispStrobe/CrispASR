// parakeet-main — minimal CLI driver for the Parakeet TDT runtime.
//
// Iteration 1: just loads the model and prints hyperparams + a few token strings.
// The full transcription pipeline (encoder forward + LSTM predictor + TDT
// greedy decode) lands in subsequent commits.

#include "parakeet.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static void print_usage(const char * prog) {
    fprintf(stderr,
        "\nusage: %s -m MODEL.gguf [-f AUDIO.wav]\n\n"
        "options:\n"
        "  -h, --help        show this help\n"
        "  -m FNAME          parakeet GGUF model (from convert-parakeet-to-gguf.py)\n"
        "  -f FNAME          input audio (16 kHz mono WAV) — for future iterations\n"
        "  -t N              threads (default: 4)\n"
        "  -v                verbose\n\n",
        prog);
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string audio_path;
    int n_threads = 4;
    int verbosity = 1;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { print_usage(argv[0]); return 0; }
        else if (a == "-m" && i+1 < argc) model_path = argv[++i];
        else if (a == "-f" && i+1 < argc) audio_path = argv[++i];
        else if (a == "-t" && i+1 < argc) n_threads  = std::atoi(argv[++i]);
        else if (a == "-v")               verbosity  = 2;
        else { fprintf(stderr, "unknown option '%s'\n", a.c_str()); print_usage(argv[0]); return 1; }
    }

    if (model_path.empty()) { print_usage(argv[0]); return 1; }

    parakeet_context_params p = parakeet_context_default_params();
    p.n_threads = n_threads;
    p.verbosity = verbosity;

    parakeet_context * ctx = parakeet_init_from_file(model_path.c_str(), p);
    if (!ctx) {
        fprintf(stderr, "%s: failed to load '%s'\n", argv[0], model_path.c_str());
        return 1;
    }

    fprintf(stderr,
        "%s: loaded OK\n"
        "  vocab_size    = %d\n"
        "  blank_id      = %d\n"
        "  n_mels        = %d\n"
        "  sample_rate   = %d\n"
        "  frame_dur_cs  = %d\n",
        argv[0],
        parakeet_n_vocab(ctx),
        parakeet_blank_id(ctx),
        parakeet_n_mels(ctx),
        parakeet_sample_rate(ctx),
        parakeet_frame_dur_cs(ctx));

    fprintf(stderr, "  sample tokens:");
    for (int id : {0, 1, 2, 100, 1000, 8000, 8191}) {
        const char * s = parakeet_token_to_str(ctx, id);
        fprintf(stderr, "  [%d]='%s'", id, s);
    }
    fprintf(stderr, "\n");

    // Smoke test: run the encoder on a zero mel of 100 frames (~1 s of audio
    // post-mel) to verify the FastConformer graph compiles and computes.
    fprintf(stderr, "%s: encoder smoke test ...\n", argv[0]);
    int T_enc = parakeet_test_encoder(ctx, 100);
    if (T_enc < 0) {
        fprintf(stderr, "%s: encoder smoke test FAILED\n", argv[0]);
        parakeet_free(ctx);
        return 2;
    }

    if (!audio_path.empty()) {
        fprintf(stderr, "%s: transcription not yet implemented (audio path ignored)\n", argv[0]);
    }

    parakeet_free(ctx);
    return 0;
}
