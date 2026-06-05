// qwen3-asr-test-trace — end-to-end audio→text test for Qwen3-ASR.
//
// Loads the reference prompt-trace stages from a crispasr-diff GGUF
// archive (produced by tools/dump_reference.py --backend qwen3):
//   mel_spectrogram           (128, T_mel)     — log-mel
//   trace_input_ids           (T,)             — prompt token IDs
//   trace_audio_pad_pos       (N,)             — audio_pad positions
//   trace_first_logits        (vocab,)         — next-token logits at last pos
//   trace_generated_ids       (G,)             — reference decoded sequence
//
// All F32 on disk; integer stages are cast back to int32 on load.
//
// Pipeline:
//   1. encoder on mel → audio_embeds (N, D)
//   2. embed prompt tokens → text_embeds (T, D)
//   3. splice audio_embeds into audio_pad positions
//   4. LLM prefill via KV cache → next-token logits
//   5. diff last-position logits against reference
//   6. greedy decode and compare to reference generated ids
//
// Reference archive is produced by:
//
//   python tools/dump_reference.py --backend qwen3 \
//       --model-dir /path/to/hf/qwen3-asr-0.6b \
//       --audio samples/jfk.wav \
//       --output /tmp/qwen3-ref.gguf
//
// Usage:
//   qwen3-asr-test-trace qwen3-asr-0.6b.gguf /tmp/qwen3-ref.gguf

#include "crispasr_diff.h"
#include "qwen3_asr.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s qwen3-asr-0.6b.gguf reference.gguf\n"
                "\n"
                "  reference.gguf  archive from tools/dump_reference.py --backend qwen3\n"
                "                  must include 'mel_spectrogram', 'trace_input_ids',\n"
                "                  'trace_audio_pad_pos', 'trace_first_logits',\n"
                "                  and 'trace_generated_ids'\n",
                argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * ref_path   = argv[2];

    crispasr_diff::Ref ref;
    if (!ref.load(ref_path)) return 2;

    for (const char * name : {"mel_spectrogram", "trace_input_ids",
                              "trace_audio_pad_pos", "trace_first_logits",
                              "trace_generated_ids"}) {
        if (!ref.has(name)) {
            fprintf(stderr,
                    "reference archive missing '%s'.\n"
                    "Re-dump with: python tools/dump_reference.py --backend qwen3 "
                    "--stages mel_spectrogram,trace_input_ids,trace_audio_pad_pos,"
                    "trace_first_logits,trace_generated_ids,generated_text ...\n",
                    name);
            return 3;
        }
    }

    // ---- Load reference stages ----
    auto [ids_f32, ids_n] = ref.get_f32("trace_input_ids");
    std::vector<int32_t> ids(ids_n);
    for (size_t i = 0; i < ids_n; i++) ids[i] = (int32_t)ids_f32[i];
    const int T_prompt = (int)ids.size();

    auto [pad_f32, pad_n] = ref.get_f32("trace_audio_pad_pos");
    std::vector<int32_t> pad_pos(pad_n);
    for (size_t i = 0; i < pad_n; i++) pad_pos[i] = (int32_t)pad_f32[i];
    const int N_audio = (int)pad_pos.size();

    auto [ref_logits, ref_logits_n] = ref.get_f32("trace_first_logits");

    auto [gen_f32, gen_n] = ref.get_f32("trace_generated_ids");
    std::vector<int32_t> ref_gen_ids(gen_n);
    for (size_t i = 0; i < gen_n; i++) ref_gen_ids[i] = (int32_t)gen_f32[i];

    auto [mel_ptr, mel_n] = ref.get_f32("mel_spectrogram");
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

    fprintf(stderr, "prompt: %d tokens, %d audio_pad placeholders\n", T_prompt, N_audio);
    fprintf(stderr, "mel:    %d x %d\n", n_mels, T_mel);
    fprintf(stderr, "ref logits dim: %zu\n", ref_logits_n);
    fprintf(stderr, "ref gen ids: %zu tokens\n", ref_gen_ids.size());

    // ---- Init model ----
    auto cp = qwen3_asr_context_default_params();
    cp.n_threads = 4;
    auto * ctx = qwen3_asr_init_from_file(model_path, cp);
    if (!ctx) { fprintf(stderr, "init failed\n"); return 4; }

    // ---- Step 1: encoder ----
    int N_enc = 0, pdim = 0;
    float * audio_embeds = qwen3_asr_run_encoder(ctx, mel_ptr, n_mels, T_mel,
                                                  &N_enc, &pdim);
    if (!audio_embeds) {
        fprintf(stderr, "encoder failed\n");
        qwen3_asr_free(ctx);
        return 5;
    }
    fprintf(stderr, "encoder: N=%d pdim=%d\n", N_enc, pdim);
    if (N_enc != N_audio) {
        fprintf(stderr,
                "encoder gave %d frames but prompt has %d audio_pad slots\n",
                N_enc, N_audio);
    }

    // ---- Step 2: text embeds ----
    float * text_embeds = qwen3_asr_embed_tokens(ctx, ids.data(), T_prompt);
    if (!text_embeds) {
        fprintf(stderr, "embed failed\n");
        free(audio_embeds); qwen3_asr_free(ctx);
        return 6;
    }
    fprintf(stderr, "text embeds: (%d, %d)\n", T_prompt, pdim);

    // ---- Step 3: splice audio embeds at audio_pad positions ----
    const int n_to_use = std::min(N_audio, N_enc);
    for (int i = 0; i < n_to_use; i++) {
        int pos = pad_pos[i];
        std::memcpy(text_embeds + (size_t)pos * pdim,
                    audio_embeds + (size_t)i * pdim,
                    pdim * sizeof(float));
    }
    fprintf(stderr, "spliced %d audio frames into prompt\n", n_to_use);

    // ---- Step 4: LLM prefill via KV cache ----
    if (!qwen3_asr_kv_init(ctx, /*max_ctx*/ 4096)) {
        fprintf(stderr, "kv_init failed\n");
        free(text_embeds); free(audio_embeds); qwen3_asr_free(ctx);
        return 7;
    }
    qwen3_asr_kv_reset(ctx);

    auto t_prefill_0 = std::chrono::steady_clock::now();
    int n_t = 0, vocab = 0;
    float * logits = qwen3_asr_run_llm_kv(ctx, text_embeds, T_prompt, /*n_past*/0,
                                          &n_t, &vocab);
    auto t_prefill_1 = std::chrono::steady_clock::now();
    if (!logits) {
        fprintf(stderr, "llm prefill failed\n");
        free(text_embeds); free(audio_embeds); qwen3_asr_free(ctx);
        return 8;
    }
    const double prefill_ms =
        std::chrono::duration<double, std::milli>(t_prefill_1 - t_prefill_0).count();
    fprintf(stderr, "llm prefill: (%d, %d)  in %.0f ms\n", n_t, vocab, prefill_ms);

    if ((int)ref_logits_n != vocab) {
        fprintf(stderr,
                "vocab mismatch: cpp=%d ref=%zu\n", vocab, ref_logits_n);
        free(logits); free(text_embeds); free(audio_embeds); qwen3_asr_free(ctx);
        return 9;
    }

    // ---- Step 5: diff last-position logits ----
    const float * cpp_last = logits;
    int cpp_argmax = 0; float cpp_max = -1e30f;
    int ref_argmax = 0; float ref_max = -1e30f;
    double dot = 0, na = 0, nb = 0;
    float maxd = 0;
    for (int k = 0; k < vocab; k++) {
        if (cpp_last[k]   > cpp_max) { cpp_max = cpp_last[k]; cpp_argmax = k; }
        if (ref_logits[k] > ref_max) { ref_max = ref_logits[k]; ref_argmax = k; }
        const double a = cpp_last[k], b = ref_logits[k];
        dot += a*b; na += a*a; nb += b*b;
        const float ad = std::fabs(cpp_last[k] - ref_logits[k]);
        if (ad > maxd) maxd = ad;
    }
    const double cs = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
    fprintf(stderr, "\nNEXT-TOKEN LOGITS:\n");
    fprintf(stderr, "  cpp argmax: %d  (logit %.4f)\n", cpp_argmax, cpp_max);
    fprintf(stderr, "  ref argmax: %d  (logit %.4f)\n", ref_argmax, ref_max);
    fprintf(stderr, "  cosine sim: %.6f\n", cs);
    fprintf(stderr, "  max abs:    %.4e\n", maxd);
    fprintf(stderr, "  match: %s\n", cpp_argmax == ref_argmax ? "PASS" : "FAIL");

    free(logits);

    // ---- Step 6: greedy decode via KV cache ----
    fprintf(stderr, "\nGREEDY DECODE (KV cache, single-token forward per step):\n");
    const int EOS = 151645;
    const int MAX_NEW = 40;
    std::vector<int32_t> gen;
    gen.push_back(cpp_argmax);
    fprintf(stderr, "  step 0: id=%d (from prefill)\n", cpp_argmax);

    auto t_decode_0 = std::chrono::steady_clock::now();
    int n_past = T_prompt;
    for (int step = 1; step < MAX_NEW; step++) {
        if (gen.back() == EOS) break;
        int32_t last_id = gen.back();
        float * tail = qwen3_asr_embed_tokens(ctx, &last_id, 1);
        if (!tail) break;
        float * lg = qwen3_asr_run_llm_kv(ctx, tail, /*n_tokens*/1, n_past,
                                          nullptr, nullptr);
        free(tail);
        if (!lg) break;
        n_past += 1;
        int next = 0; float mx = -1e30f;
        for (int k = 0; k < vocab; k++) if (lg[k] > mx) { mx = lg[k]; next = k; }
        free(lg);
        gen.push_back(next);
        fprintf(stderr, "  step %d: id=%d\n", step, next);
    }
    auto t_decode_1 = std::chrono::steady_clock::now();
    const double decode_ms =
        std::chrono::duration<double, std::milli>(t_decode_1 - t_decode_0).count();
    fprintf(stderr, "  decode loop: %.0f ms total, %.1f ms/token avg\n",
            decode_ms, decode_ms / std::max<size_t>(gen.size() - 1, 1));

    fprintf(stderr, "\nGENERATED IDS (%zu): ", gen.size());
    for (auto v : gen) fprintf(stderr, "%d ", v);
    fprintf(stderr, "\n");
    fprintf(stderr, "REFERENCE  IDS (%zu): ", ref_gen_ids.size());
    for (auto v : ref_gen_ids) fprintf(stderr, "%d ", v);
    fprintf(stderr, "\n");

    // Longest contiguous run of ref tokens matched somewhere in gen. The
    // Python wrapper strips language-detection tokens from the front, so a
    // direct prefix compare misses valid matches.
    int best_run = 0, best_start = -1;
    if (!ref_gen_ids.empty() && gen.size() >= ref_gen_ids.size()) {
        for (size_t off = 0; off + ref_gen_ids.size() <= gen.size(); off++) {
            int run = 0;
            for (size_t i = 0; i < ref_gen_ids.size(); i++) {
                if (gen[off + i] == ref_gen_ids[i]) run++;
                else break;
            }
            if (run > best_run) { best_run = run; best_start = (int)off; }
        }
    }
    fprintf(stderr,
            "  longest contiguous match: %d / %zu starting at gen offset %d\n",
            best_run, ref_gen_ids.size(), best_start);
    const bool full_match = (best_run == (int)ref_gen_ids.size()) &&
                            !ref_gen_ids.empty();
    fprintf(stderr, "  end-to-end transcript: %s\n",
            full_match ? "PASS (all ref tokens reproduced)" : "FAIL");

    free(text_embeds);
    free(audio_embeds);
    qwen3_asr_free(ctx);

    int verdict = 0;
    if (cpp_argmax != ref_argmax) verdict |= 1;
    if (!full_match)              verdict |= 2;
    return verdict;
}
