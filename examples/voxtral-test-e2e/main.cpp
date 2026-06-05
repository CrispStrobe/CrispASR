// voxtral-test-e2e — end-to-end audio→text smoke test for Voxtral (3B).
//
// Loads the reference mel spectrogram from a crispasr-diff GGUF
// archive (produced by tools/dump_reference.py --backend voxtral),
// runs the full encoder + prompt splice + LLM prefill + greedy decode
// pipeline, and prints the resulting transcript. Unlike the other
// differential tests this one does NOT check numerical equality —
// it's a smoke test to verify the whole pipeline runs end-to-end.
//
// Reference archive is produced by:
//
//   python tools/dump_reference.py --backend voxtral \
//       --model-dir /path/to/hf/voxtral-mini-3b-2507 \
//       --audio samples/jfk.wav \
//       --output /tmp/voxtral-ref.gguf
//
// Usage:
//   voxtral-test-e2e voxtral-mini-3b-2507.gguf /tmp/voxtral-ref.gguf

#include "crispasr_diff.h"
#include "voxtral.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s voxtral-mini-3b-2507.gguf reference.gguf\n"
                "\n"
                "  reference.gguf  archive from tools/dump_reference.py --backend voxtral\n",
                argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * ref_path   = argv[2];

    // Load the reference mel from the archive instead of .npy.
    crispasr_diff::Ref ref;
    if (!ref.load(ref_path)) return 2;
    auto [mel_ptr, mel_n] = ref.get_f32("mel_spectrogram");
    if (!mel_ptr) {
        fprintf(stderr, "reference archive missing 'mel_spectrogram'\n");
        return 3;
    }
    auto mel_shape = ref.shape("mel_spectrogram");
    int n_mels = 128, T_mel = 0;
    for (auto d : mel_shape) {
        if (d == 128) { /* mel axis */ }
        else if (T_mel == 0) T_mel = (int)d;
    }
    if (T_mel == 0) {
        fprintf(stderr, "unexpected mel_spectrogram shape\n");
        return 3;
    }
    fprintf(stderr, "reference mel: %d mels x %d frames\n", n_mels, T_mel);

    // Init model
    auto cp = voxtral_context_default_params();
    cp.n_threads = 4;
    auto * ctx = voxtral_init_from_file(model_path, cp);
    if (!ctx) { fprintf(stderr, "init failed\n"); return 4; }

    // Encoder
    int N_enc = 0, pdim = 0;
    fprintf(stderr, "running encoder ...\n");
    float * audio_embeds = voxtral_run_encoder(ctx, mel_ptr, n_mels, T_mel,
                                                &N_enc, &pdim);
    if (!audio_embeds) {
        fprintf(stderr, "encoder failed\n");
        voxtral_free(ctx);
        return 5;
    }
    fprintf(stderr, "encoder: %d frames x %d dim\n", N_enc, pdim);

    // Build prompt: <s> [INST] [BEGIN_AUDIO] <audio_pad>xN [/INST] lang:en [TRANSCRIBE]
    std::vector<int32_t> ids;
    ids.push_back(1); ids.push_back(3); ids.push_back(25);
    for (int i = 0; i < N_enc; i++) ids.push_back(24);
    ids.push_back(4); ids.push_back(9909); ids.push_back(1058); ids.push_back(1262); ids.push_back(34);
    const int T_prompt = (int)ids.size();
    fprintf(stderr, "prompt: %d tokens (incl. %d audio_pad)\n", T_prompt, N_enc);

    // Embed text tokens
    float * text_embeds = voxtral_embed_tokens(ctx, ids.data(), T_prompt);
    if (!text_embeds) {
        fprintf(stderr, "embed failed\n");
        free(audio_embeds);
        voxtral_free(ctx);
        return 6;
    }

    // Splice audio embeds at audio_pad positions (id=24)
    int spliced = 0;
    for (int i = 0; i < T_prompt && spliced < N_enc; i++) {
        if (ids[i] == 24) {
            std::memcpy(text_embeds + (size_t)i * pdim,
                        audio_embeds + (size_t)spliced * pdim,
                        pdim * sizeof(float));
            spliced++;
        }
    }
    free(audio_embeds);
    fprintf(stderr, "spliced %d audio frames\n", spliced);

    // KV cache + prefill
    if (!voxtral_kv_init(ctx, 4096)) {
        free(text_embeds); voxtral_free(ctx); return 7;
    }
    voxtral_kv_reset(ctx);

    int n_t = 0, vocab = 0;
    float * logits = voxtral_run_llm_kv(ctx, text_embeds, T_prompt, 0,
                                         &n_t, &vocab);
    if (!logits) {
        fprintf(stderr, "prefill failed\n");
        free(text_embeds); voxtral_free(ctx);
        return 8;
    }
    free(text_embeds);
    fprintf(stderr, "prefill done (vocab=%d)\n", vocab);

    // Greedy decode
    constexpr int EOS = 2;
    constexpr int MAX_NEW = 256;
    std::vector<int32_t> gen;
    {
        int mx_i = 0;
        float mx = -1e30f;
        for (int k = 0; k < vocab; k++) if (logits[k] > mx) { mx = logits[k]; mx_i = k; }
        gen.push_back(mx_i);
    }
    free(logits);
    fprintf(stderr, "step 0: id=%d\n", gen.back());

    int n_past = T_prompt;
    while ((int)gen.size() < MAX_NEW && gen.back() != EOS) {
        int32_t last = gen.back();
        float * tail = voxtral_embed_tokens(ctx, &last, 1);
        if (!tail) break;
        float * lg = voxtral_run_llm_kv(ctx, tail, 1, n_past, nullptr, nullptr);
        free(tail);
        if (!lg) break;
        n_past++;
        int nx = 0;
        float mx = -1e30f;
        for (int k = 0; k < vocab; k++) if (lg[k] > mx) { mx = lg[k]; nx = k; }
        free(lg);
        gen.push_back(nx);
    }

    fprintf(stderr, "\ngenerated %zu tokens\n", gen.size());
    std::string transcript;
    for (auto id : gen) {
        if (id == EOS) break;
        int len = 0;
        const uint8_t * bytes = voxtral_token_text(ctx, id, &len);
        if (bytes && len > 0) transcript.append((const char *)bytes, (size_t)len);
    }
    fprintf(stderr, "transcript: %s\n", transcript.c_str());
    printf("%s\n", transcript.c_str());

    voxtral_free(ctx);
    return 0;
}
