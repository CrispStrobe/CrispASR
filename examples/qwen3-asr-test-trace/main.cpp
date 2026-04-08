// qwen3-asr-test-trace — end-to-end audio→text test for Qwen3-ASR.
//
// Loads the trace dump produced by models/qwen3-asr-trace-dump.py:
//   trace_input_ids.npy        prompt token IDs (with audio_pad placeholders)
//   trace_audio_pad_pos.npy    positions of audio_pad in prompt
//   trace_first_logits.npy     reference next-token logits at last prompt position
//   trace_generated_ids.npy    reference greedy-decoded sequence
//   trace_generated_text.txt   reference text
// Plus the mel input from the audio reference dir:
//   /tmp/qwen3-asr-ref/jfk/mel_input.npy
//
// Pipeline:
//   1. Load mel → run audio encoder → audio_embeds (N, 1024)
//   2. Embed prompt token IDs → text_embeds (T, 1024)
//   3. Splice audio_embeds into text_embeds at audio_pad positions
//   4. Run LLM forward → logits (T, V)
//   5. Diff next-token logits against reference
//   6. Greedy decode N tokens via repeated forward (no KV cache)
//   7. Compare against reference generated_ids

#include "qwen3_asr.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>

template <typename T>
static bool load_npy(const std::string & path, std::vector<T> & data,
                     std::vector<int> & shape, const char * dtype_str) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    char magic[6]; f.read(magic, 6);
    if (memcmp(magic, "\x93NUMPY", 6) != 0) return false;
    uint8_t major, minor; f.read((char*)&major, 1); f.read((char*)&minor, 1);
    uint32_t hdr_len = 0;
    if (major == 1) { uint16_t hl; f.read((char*)&hl, 2); hdr_len = hl; }
    else            { f.read((char*)&hdr_len, 4); }
    std::string header(hdr_len, '\0'); f.read(&header[0], hdr_len);
    if (header.find(dtype_str) == std::string::npos) {
        fprintf(stderr, "%s: expected dtype %s\n", path.c_str(), dtype_str); return false;
    }
    auto sp = header.find("'shape':");
    auto lp = header.find('(', sp);
    auto rp = header.find(')', lp);
    std::string sh = header.substr(lp+1, rp-lp-1);
    shape.clear();
    size_t i = 0;
    while (i < sh.size()) {
        while (i < sh.size() && (sh[i] == ' ' || sh[i] == ',')) i++;
        if (i >= sh.size()) break;
        int v = 0;
        while (i < sh.size() && sh[i] >= '0' && sh[i] <= '9') { v = v*10 + (sh[i]-'0'); i++; }
        shape.push_back(v);
    }
    size_t total = 1;
    for (int s : shape) total *= (size_t)s;
    data.resize(total);
    f.read((char*)data.data(), total * sizeof(T));
    return (bool)f;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr,
            "usage: %s qwen3-asr-0.6b.gguf  TRACE_DIR  AUDIO_REF_DIR\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    std::string trace_dir   = argv[2];
    std::string audio_ref   = argv[3];

    // ----- Load all reference data -----
    std::vector<int32_t> ids;            std::vector<int> ids_shape;
    std::vector<int32_t> pad_pos;        std::vector<int> pad_shape;
    std::vector<float>   ref_logits;     std::vector<int> rl_shape;
    std::vector<int32_t> ref_gen_ids;    std::vector<int> rg_shape;
    std::vector<float>   mel;            std::vector<int> mel_shape;

    if (!load_npy<int32_t>(trace_dir + "/trace_input_ids.npy",     ids, ids_shape, "'<i4'")) return 2;
    if (!load_npy<int32_t>(trace_dir + "/trace_audio_pad_pos.npy", pad_pos, pad_shape, "'<i4'")) return 3;
    if (!load_npy<float>  (trace_dir + "/trace_first_logits.npy",  ref_logits, rl_shape, "'<f4'")) return 4;
    if (!load_npy<int32_t>(trace_dir + "/trace_generated_ids.npy", ref_gen_ids, rg_shape, "'<i4'")) return 5;
    if (!load_npy<float>  (audio_ref + "/mel_input.npy",            mel, mel_shape, "'<f4'")) return 6;

    int T_prompt = (int)ids.size();
    int N_audio  = (int)pad_pos.size();
    int n_mels = mel_shape.size() == 3 ? mel_shape[1] : mel_shape[0];
    int T_mel  = mel_shape.size() == 3 ? mel_shape[2] : mel_shape[1];
    fprintf(stderr, "prompt: %d tokens, %d audio_pad placeholders\n", T_prompt, N_audio);
    fprintf(stderr, "mel:    %d × %d\n", n_mels, T_mel);
    fprintf(stderr, "ref logits dim: %d\n", rl_shape.size() ? rl_shape[0] : 0);
    fprintf(stderr, "ref gen ids: %zu tokens\n", ref_gen_ids.size());

    // ----- Init model -----
    auto cp = qwen3_asr_context_default_params();
    cp.n_threads = 4;
    auto * ctx = qwen3_asr_init_from_file(model_path, cp);
    if (!ctx) { fprintf(stderr, "init failed\n"); return 7; }

    // ----- Step 1: encoder -----
    int N_enc = 0, pdim = 0;
    float * audio_embeds = qwen3_asr_run_encoder(ctx, mel.data(), n_mels, T_mel, &N_enc, &pdim);
    if (!audio_embeds) { fprintf(stderr, "encoder failed\n"); qwen3_asr_free(ctx); return 8; }
    fprintf(stderr, "encoder: N=%d pdim=%d\n", N_enc, pdim);
    if (N_enc != N_audio) {
        fprintf(stderr, "encoder gave %d frames but prompt has %d audio_pad slots\n",
                N_enc, N_audio);
        // continue but only splice the first min()
    }

    // ----- Step 2: text embeds -----
    float * text_embeds = qwen3_asr_embed_tokens(ctx, ids.data(), T_prompt);
    if (!text_embeds) { fprintf(stderr, "embed failed\n"); free(audio_embeds); qwen3_asr_free(ctx); return 9; }
    fprintf(stderr, "text embeds: (%d, %d)\n", T_prompt, pdim);

    // ----- Step 3: splice audio embeds at audio_pad positions -----
    int n_to_use = std::min(N_audio, N_enc);
    for (int i = 0; i < n_to_use; i++) {
        int pos = pad_pos[i];
        std::memcpy(text_embeds + (size_t)pos * pdim,
                    audio_embeds + (size_t)i * pdim,
                    pdim * sizeof(float));
    }
    fprintf(stderr, "spliced %d audio frames into prompt\n", n_to_use);

    // ----- Step 4: LLM PREFILL via KV cache -----
    if (!qwen3_asr_kv_init(ctx, /*max_ctx*/ 4096)) {
        fprintf(stderr, "kv_init failed\n");
        free(text_embeds); free(audio_embeds); qwen3_asr_free(ctx); return 10;
    }
    qwen3_asr_kv_reset(ctx);

    auto t_prefill_0 = std::chrono::steady_clock::now();
    int n_t = 0, vocab = 0;
    float * logits = qwen3_asr_run_llm_kv(ctx, text_embeds, T_prompt, /*n_past*/0,
                                          &n_t, &vocab);
    auto t_prefill_1 = std::chrono::steady_clock::now();
    if (!logits) { fprintf(stderr, "llm prefill failed\n"); free(text_embeds); free(audio_embeds); qwen3_asr_free(ctx); return 10; }
    double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill_1 - t_prefill_0).count();
    fprintf(stderr, "llm prefill: (%d, %d)  in %.0f ms\n", n_t, vocab, prefill_ms);

    // ----- Step 5: diff next-token logits (last position) against reference -----
    // qwen3_asr_run_llm_kv now returns only the last position's logits (vocab,).
    const float * cpp_last = logits;
    int cpp_argmax = 0;
    float cpp_max = -1e30f;
    for (int k = 0; k < vocab; k++) if (cpp_last[k] > cpp_max) { cpp_max = cpp_last[k]; cpp_argmax = k; }
    int ref_argmax = 0;
    float ref_max = -1e30f;
    for (int k = 0; k < vocab; k++) if (ref_logits[k] > ref_max) { ref_max = ref_logits[k]; ref_argmax = k; }

    double dot=0, na=0, nb=0;
    float maxd=0;
    for (int k = 0; k < vocab; k++) {
        double a = cpp_last[k], b = ref_logits[k];
        dot += a*b; na += a*a; nb += b*b;
        if (std::fabs(cpp_last[k] - ref_logits[k]) > maxd) maxd = std::fabs(cpp_last[k] - ref_logits[k]);
    }
    double cs = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
    fprintf(stderr, "\nNEXT-TOKEN LOGITS:\n");
    fprintf(stderr, "  cpp argmax: %d  (logit %.4f)\n", cpp_argmax, cpp_max);
    fprintf(stderr, "  ref argmax: %d  (logit %.4f)\n", ref_argmax, ref_max);
    fprintf(stderr, "  cosine sim: %.6f\n", cs);
    fprintf(stderr, "  max abs:    %.4e\n", maxd);
    fprintf(stderr, "  match: %s\n", cpp_argmax == ref_argmax ? "✓" : "✗");

    free(logits);

    // ----- Step 6: greedy decode via KV cache (one token per step) -----
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
        int n_t2 = 0, v2 = 0;
        float * lg = qwen3_asr_run_llm_kv(ctx, tail, /*n_tokens*/1, n_past,
                                          &n_t2, &v2);
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
    double decode_ms = std::chrono::duration<double, std::milli>(t_decode_1 - t_decode_0).count();
    fprintf(stderr, "  decode loop: %.0f ms total, %.1f ms/token avg\n",
            decode_ms, decode_ms / (gen.size() - 1));

    fprintf(stderr, "\nGENERATED IDS (%zu): ", gen.size());
    for (auto v : gen) fprintf(stderr, "%d ", v);
    fprintf(stderr, "\n");
    fprintf(stderr, "REFERENCE  IDS (%zu): ", ref_gen_ids.size());
    for (auto v : ref_gen_ids) fprintf(stderr, "%d ", v);
    fprintf(stderr, "\n");

    // Find longest contiguous run of ref tokens that matches a sub-sequence of
    // gen. The Python wrapper strips language-detection tokens from the front
    // of the output so a direct prefix compare misses the match.
    int best_run = 0, best_start = -1;
    for (size_t off = 0; off + ref_gen_ids.size() <= gen.size(); off++) {
        int run = 0;
        for (size_t i = 0; i < ref_gen_ids.size(); i++) {
            if (gen[off + i] == ref_gen_ids[i]) run++;
            else break;
        }
        if (run > best_run) { best_run = run; best_start = (int)off; }
    }
    fprintf(stderr, "  longest contiguous match: %d / %zu starting at gen offset %d\n",
            best_run, ref_gen_ids.size(), best_start);
    bool full_match = (best_run == (int)ref_gen_ids.size());
    fprintf(stderr, "  end-to-end transcript: %s\n",
            full_match ? "PASS (all ref tokens reproduced)" : "FAIL");

    free(text_embeds);
    free(audio_embeds);
    qwen3_asr_free(ctx);
    int verdict = 0;
    if (cpp_argmax != ref_argmax) verdict |= 1;
    if (!full_match) verdict |= 2;
    return verdict;
}
