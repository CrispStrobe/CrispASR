// qwen3-asr-test-llm — differential test for the Qwen3 0.6B LLM forward.
//
// Loads:
//   - GGUF model from convert-qwen3-asr-to-gguf.py
//   - llm_input_ids.npy  (T,)         int32   token IDs
//   - llm_logits.npy     (T, 151936)  f32     reference logits
//   - llm_topk.npy       (T, 5)       int32   reference top-5 token ids
//
// Runs the C++ LLM forward and reports:
//   - max abs diff vs reference logits
//   - per-position cosine similarity of logit vectors
//   - top-1 match rate (does C++ argmax = reference argmax?)
//
// Usage:
//   qwen3-asr-test-llm  qwen3-asr-0.6b.gguf  /tmp/qwen3-asr-ref/llm

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
        fprintf(stderr, "%s: expected dtype %s\n", path.c_str(), dtype_str);
        return false;
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
    if (argc < 3) {
        fprintf(stderr, "usage: %s qwen3-asr-0.6b.gguf REF_DIR\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    std::string  ref_dir    = argv[2];

    // Load inputs
    std::vector<int32_t> ids; std::vector<int> ids_shape;
    if (!load_npy<int32_t>(ref_dir + "/llm_input_ids.npy", ids, ids_shape, "'<i4'")) return 2;
    fprintf(stderr, "input_ids: ");
    for (int s : ids_shape) fprintf(stderr, "%d ", s);
    fprintf(stderr, " = ");
    for (auto v : ids) fprintf(stderr, "%d ", v);
    fprintf(stderr, "\n");
    int T = (int)ids.size();

    std::vector<float> ref_logits; std::vector<int> ref_shape;
    if (!load_npy<float>(ref_dir + "/llm_logits.npy", ref_logits, ref_shape, "'<f4'")) return 3;
    fprintf(stderr, "ref logits shape: ");
    for (int s : ref_shape) fprintf(stderr, "%d ", s); fprintf(stderr, "\n");
    int ref_T = ref_shape[0], ref_vocab = ref_shape[1];

    std::vector<int32_t> ref_topk; std::vector<int> topk_shape;
    if (!load_npy<int32_t>(ref_dir + "/llm_topk.npy", ref_topk, topk_shape, "'<i4'")) return 4;

    // Init model
    auto cp = qwen3_asr_context_default_params();
    cp.n_threads = 4;
    auto * ctx = qwen3_asr_init_from_file(model_path, cp);
    if (!ctx) { fprintf(stderr, "init failed\n"); return 5; }

    int n_t = 0, vocab = 0;
    float * logits = qwen3_asr_run_llm(ctx, ids.data(), T, &n_t, &vocab);
    if (!logits) { fprintf(stderr, "run_llm failed\n"); qwen3_asr_free(ctx); return 6; }
    fprintf(stderr, "C++ logits: T=%d vocab=%d\n", n_t, vocab);

    if (n_t != ref_T || vocab != ref_vocab) {
        fprintf(stderr, "shape mismatch: cpp(%d,%d) vs ref(%d,%d)\n",
                n_t, vocab, ref_T, ref_vocab);
        free(logits); qwen3_asr_free(ctx); return 7;
    }

    // C++ logits ggml ne=(vocab, T) → memory layout (T outer, vocab inner) row-major (T, vocab)
    // ref_logits also (T, vocab) row-major. Direct compare.
    double sum = 0.0, sumsq = 0.0;
    float max_abs = 0.0f; int max_i = -1;
    for (size_t i = 0; i < ref_logits.size(); i++) {
        float diff = logits[i] - ref_logits[i];
        float ad = std::fabs(diff);
        sum += ad; sumsq += diff*diff;
        if (ad > max_abs) { max_abs = ad; max_i = (int)i; }
    }
    fprintf(stderr, "\nLOGIT DIFF:\n");
    fprintf(stderr, "  max abs:  %.6e (idx %d)\n", max_abs, max_i);
    fprintf(stderr, "  mean abs: %.6e\n", sum / ref_logits.size());
    fprintf(stderr, "  rms:      %.6e\n", std::sqrt(sumsq / ref_logits.size()));

    // Per-position cosine sim
    double cos_sum = 0.0, cos_min = 1.0; int cos_min_i = -1;
    int top1_match = 0;
    for (int t = 0; t < T; t++) {
        const float * a = logits     + (size_t)t * vocab;
        const float * b = ref_logits.data() + (size_t)t * vocab;
        double dot = 0.0, na = 0.0, nb = 0.0;
        int amax_a = 0, amax_b = 0;
        float ma = -1e30f, mb = -1e30f;
        for (int k = 0; k < vocab; k++) {
            dot += (double)a[k] * b[k];
            na  += (double)a[k] * a[k];
            nb  += (double)b[k] * b[k];
            if (a[k] > ma) { ma = a[k]; amax_a = k; }
            if (b[k] > mb) { mb = b[k]; amax_b = k; }
        }
        double cs = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
        cos_sum += cs;
        if (cs < cos_min) { cos_min = cs; cos_min_i = t; }
        if (amax_a == amax_b) top1_match++;
        fprintf(stderr, "  pos %d: cos=%.6f cpp_top1=%d ref_top1=%d %s\n",
                t, cs, amax_a, amax_b, amax_a == amax_b ? "✓" : "✗");
    }
    fprintf(stderr, "\nSUMMARY:\n");
    fprintf(stderr, "  cosine sim mean=%.6f min=%.6f (pos %d)\n",
            cos_sum / T, cos_min, cos_min_i);
    fprintf(stderr, "  top-1 match: %d/%d\n", top1_match, T);
    int verdict = (top1_match == T && cos_min > 0.999) ? 0 : 1;
    fprintf(stderr, "  verdict: %s\n", verdict == 0 ? "PASS" : "FAIL");

    free(logits);
    qwen3_asr_free(ctx);
    return verdict;
}
