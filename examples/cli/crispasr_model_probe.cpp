// crispasr-model-probe — print the raw output of models whose result is not
// text, as one JSON object on stdout, for tests/regression/run_one.py.
//
//   crispasr-model-probe embedding <titanet.gguf>        <audio.wav>
//       {"dim": 192, "embedding": [...]}
//   crispasr-model-probe codes     <mimo-tokenizer.gguf> <audio.wav>
//       {"n_frames": T, "n_codes": T*8, "codes": [...]}   (time-first, 8 RVQ channels)
//
// CPU only, fixed thread count, so the pinned values are reproducible.

#include "common-crispasr.h"
#include "mimo_tokenizer.h"
#include "titanet.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s embedding|codes <model.gguf> <audio.wav>\n", argv[0]);
        return 2;
    }
    const std::string kind = argv[1];
    std::vector<float> samples;
    std::vector<std::vector<float>> stereo;
    if (!read_audio_data(argv[3], samples, stereo, /*stereo=*/false)) {
        fprintf(stderr, "probe: failed to read audio '%s'\n", argv[3]);
        return 3;
    }
    if (kind == "embedding") {
        titanet_context* ctx = titanet_init(argv[2], 4);
        if (!ctx) {
            fprintf(stderr, "probe: failed to load titanet '%s'\n", argv[2]);
            return 4;
        }
        float emb[192];
        const int dim = titanet_embed(ctx, samples.data(), (int)samples.size(), emb);
        titanet_free(ctx);
        if (dim <= 0 || dim > 192) {
            fprintf(stderr, "probe: titanet_embed failed (%d)\n", dim);
            return 5;
        }
        printf("{\"dim\": %d, \"embedding\": [", dim);
        for (int i = 0; i < dim; i++)
            printf("%s%.7g", i ? ", " : "", emb[i]);
        printf("]}\n");
        return 0;
    }
    if (kind == "codes") {
        auto cp = mimo_tokenizer_context_default_params();
        cp.n_threads = 4;
        cp.verbosity = 0;
        cp.use_gpu = false;
        mimo_tokenizer_context* ctx = mimo_tokenizer_init_from_file(argv[2], cp);
        if (!ctx) {
            fprintf(stderr, "probe: failed to load mimo tokenizer '%s'\n", argv[2]);
            return 4;
        }
        int n_frames = 0;
        int32_t* codes = mimo_tokenizer_encode_pcm16k(ctx, samples.data(), (int)samples.size(), &n_frames);
        mimo_tokenizer_free(ctx);
        if (!codes || n_frames <= 0) {
            fprintf(stderr, "probe: mimo_tokenizer_encode_pcm16k failed\n");
            free(codes);
            return 5;
        }
        const int n = n_frames * 8;
        printf("{\"n_frames\": %d, \"n_codes\": %d, \"codes\": [", n_frames, n);
        for (int i = 0; i < n; i++)
            printf("%s%d", i ? ", " : "", codes[i]);
        printf("]}\n");
        free(codes);
        return 0;
    }
    fprintf(stderr, "probe: unknown kind '%s'\n", kind.c_str());
    return 2;
}
