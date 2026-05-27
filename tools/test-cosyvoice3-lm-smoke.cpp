// tools/test-cosyvoice3-lm-smoke.cpp — Phase 2 smoke test + diff runner.
//
// Two modes:
//   smoke-test:  loads GGUF, runs one prefill + one AR step on synthetic
//                inputs, prints top-K logits.
//   diff-runner: loads GGUF, reads caller-supplied float32 embeds from
//                a raw .bin file, prefills, writes step0 logits to a raw
//                .bin file. Pair with `tools/reference_backends/cosyvoice3_tts.py`
//                to compute cosine vs PyTorch ref.
//
// Usage:
//   ./build/test-cv3-lm-smoke <path-to-gguf>                       # smoke
//   ./build/test-cv3-lm-smoke <gguf> --embeds-bin <in.bin> \
//       --n-tokens <T> --logits-bin <out.bin>                      # diff

#include "cosyvoice3_tts.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void top_k(const float* logits, int n, int k, std::vector<int>& idx, std::vector<float>& val) {
    std::vector<int> tmp(n);
    for (int i = 0; i < n; i++)
        tmp[i] = i;
    std::partial_sort(tmp.begin(), tmp.begin() + k, tmp.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    idx.assign(tmp.begin(), tmp.begin() + k);
    val.resize((size_t)k);
    for (int i = 0; i < k; i++)
        val[i] = logits[idx[i]];
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s path/to/cosyvoice3-llm-f16.gguf [--embeds-bin <in> --n-tokens <T> --logits-bin <out>]\n", argv[0]);
        return 2;
    }
    const char* path = argv[1];
    const char* embeds_bin = nullptr;
    const char* logits_bin = nullptr;
    int diff_n_tokens = 0;
    for (int i = 2; i + 1 < argc; i++) {
        if (!strcmp(argv[i], "--embeds-bin"))
            embeds_bin = argv[++i];
        else if (!strcmp(argv[i], "--logits-bin"))
            logits_bin = argv[++i];
        else if (!strcmp(argv[i], "--n-tokens"))
            diff_n_tokens = atoi(argv[++i]);
    }

    auto p = cosyvoice3_tts_context_default_params();
    p.verbosity = 2;
    p.use_gpu = false;
    auto* ctx = cosyvoice3_tts_init_from_file(path, p);
    if (!ctx) {
        fprintf(stderr, "FAIL: init_from_file('%s') returned nullptr\n", path);
        return 1;
    }
    fprintf(stderr, "OK: model loaded\n");

    uint32_t d_model = 0, n_layers = 0, n_heads = 0, n_kv_heads = 0, head_dim = 0;
    uint32_t text_vocab = 0, speech_vocab = 0, speech_codebook = 0;
    cosyvoice3_tts_get_hparams(ctx, &d_model, &n_layers, &n_heads, &n_kv_heads, &head_dim,
                               &text_vocab, &speech_vocab, &speech_codebook);
    fprintf(stderr,
            "OK: hp d=%u L=%u h=%u/kv=%u hd=%u text_vocab=%u speech_vocab=%u codebook=%u\n",
            d_model, n_layers, n_heads, n_kv_heads, head_dim, text_vocab, speech_vocab, speech_codebook);
    if (d_model != 896 || n_layers != 24 || n_heads != 14 || n_kv_heads != 2 || head_dim != 64) {
        fprintf(stderr, "FAIL: unexpected hparams (expected d=896 L=24 h=14 kv=2 hd=64)\n");
        cosyvoice3_tts_free(ctx);
        return 1;
    }
    if (speech_vocab != 6761 || speech_codebook != 6561) {
        fprintf(stderr, "FAIL: unexpected speech vocab (expected 6761 / 6561)\n");
        cosyvoice3_tts_free(ctx);
        return 1;
    }

    // ---- Diff-runner mode: take caller-supplied embeds, write logits ----
    if (embeds_bin && logits_bin && diff_n_tokens > 0) {
        std::vector<float> in_embeds((size_t)diff_n_tokens * (size_t)d_model);
        FILE* fi = fopen(embeds_bin, "rb");
        if (!fi) {
            fprintf(stderr, "FAIL: cannot open embeds bin '%s'\n", embeds_bin);
            cosyvoice3_tts_free(ctx);
            return 1;
        }
        size_t read = fread(in_embeds.data(), sizeof(float), in_embeds.size(), fi);
        fclose(fi);
        if (read != in_embeds.size()) {
            fprintf(stderr, "FAIL: short read from embeds bin (%zu of %zu)\n", read, in_embeds.size());
            cosyvoice3_tts_free(ctx);
            return 1;
        }
        fprintf(stderr, "diff: read %zu f32 from %s (T=%d, d=%u)\n", read, embeds_bin, diff_n_tokens, d_model);
        cosyvoice3_tts_reset_kv(ctx);
        float* logits = cosyvoice3_tts_prefill_with_embeds(ctx, in_embeds.data(), diff_n_tokens, /*n_past*/ 0);
        if (!logits) {
            fprintf(stderr, "FAIL: prefill returned nullptr\n");
            cosyvoice3_tts_free(ctx);
            return 1;
        }
        FILE* fo = fopen(logits_bin, "wb");
        if (!fo) {
            fprintf(stderr, "FAIL: cannot open logits bin '%s'\n", logits_bin);
            free(logits);
            cosyvoice3_tts_free(ctx);
            return 1;
        }
        const size_t nlog = (size_t)speech_vocab;
        size_t wrote = fwrite(logits, sizeof(float), nlog, fo);
        fclose(fo);
        if (wrote != nlog) {
            fprintf(stderr, "FAIL: short write to logits bin (%zu of %zu)\n", wrote, nlog);
            free(logits);
            cosyvoice3_tts_free(ctx);
            return 1;
        }
        {
            std::vector<int> idx;
            std::vector<float> val;
            top_k(logits, (int)speech_vocab, 5, idx, val);
            fprintf(stderr, "diff: step0_logits top-5:");
            for (int i = 0; i < 5; i++)
                fprintf(stderr, " %d:%.4f", idx[i], val[i]);
            fprintf(stderr, "\n");
        }
        free(logits);
        cosyvoice3_tts_free(ctx);
        return 0;
    }

    // ---- Embedding lookup smoke test ----
    int32_t ids[3] = {1, 2, 3};
    int* dummy = nullptr;
    (void)dummy;
    float* text_emb = cosyvoice3_tts_embed_text(ctx, ids, 3);
    if (!text_emb) {
        fprintf(stderr, "FAIL: embed_text returned nullptr\n");
        cosyvoice3_tts_free(ctx);
        return 1;
    }
    // Print a few values from the first embedding so we have a witness
    // to compare against the python ref dump.
    fprintf(stderr, "embed_text[id=1, dim 0..7] =");
    for (int i = 0; i < 8; i++)
        fprintf(stderr, " %.6f", text_emb[i]);
    fprintf(stderr, "\n");

    float* speech_emb = cosyvoice3_tts_embed_speech(ctx, ids, 3);
    if (!speech_emb) {
        fprintf(stderr, "FAIL: embed_speech returned nullptr\n");
        free(text_emb);
        cosyvoice3_tts_free(ctx);
        return 1;
    }
    fprintf(stderr, "embed_speech[id=1, dim 0..7] =");
    for (int i = 0; i < 8; i++)
        fprintf(stderr, " %.6f", speech_emb[i]);
    fprintf(stderr, "\n");

    // ---- One prefill on 3 text-embedded tokens, then one AR step ----
    cosyvoice3_tts_reset_kv(ctx);
    float* prefill_logits = cosyvoice3_tts_prefill_with_embeds(ctx, text_emb, 3, /*n_past*/ 0);
    if (!prefill_logits) {
        fprintf(stderr, "FAIL: prefill_with_embeds returned nullptr\n");
        free(text_emb);
        free(speech_emb);
        cosyvoice3_tts_free(ctx);
        return 1;
    }
    {
        std::vector<int> idx;
        std::vector<float> val;
        top_k(prefill_logits, (int)speech_vocab, 5, idx, val);
        fprintf(stderr, "prefill_logits top-5 (over %u speech_vocab):", speech_vocab);
        for (int i = 0; i < 5; i++)
            fprintf(stderr, " %d:%.3f", idx[i], val[i]);
        fprintf(stderr, "\n");
    }

    // ---- One greedy AR step (pick top speech token from prefill, feed it back) ----
    int32_t step_id = 0;
    {
        float best_v = prefill_logits[0];
        for (int i = 1; i < (int)speech_codebook; i++) // stay within the AR-valid range
            if (prefill_logits[i] > best_v) {
                best_v = prefill_logits[i];
                step_id = i;
            }
    }
    fprintf(stderr, "step_speech sampling speech_id=%d\n", step_id);
    float* step_logits = cosyvoice3_tts_step_speech(ctx, step_id, /*n_past*/ 3);
    if (!step_logits) {
        fprintf(stderr, "FAIL: step_speech returned nullptr\n");
        free(prefill_logits);
        free(text_emb);
        free(speech_emb);
        cosyvoice3_tts_free(ctx);
        return 1;
    }
    {
        std::vector<int> idx;
        std::vector<float> val;
        top_k(step_logits, (int)speech_vocab, 5, idx, val);
        fprintf(stderr, "step_logits top-5:");
        for (int i = 0; i < 5; i++)
            fprintf(stderr, " %d:%.3f", idx[i], val[i]);
        fprintf(stderr, "\n");
    }

    free(prefill_logits);
    free(step_logits);
    free(text_emb);
    free(speech_emb);
    cosyvoice3_tts_free(ctx);
    fprintf(stderr, "OK: smoke test complete\n");
    return 0;
}
