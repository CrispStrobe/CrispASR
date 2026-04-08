// qwen3-asr-test-conv — differential test for the Qwen3-ASR conv front-end.
//
// Loads:
//   - GGUF model produced by convert-qwen3-asr-to-gguf.py
//   - Reference mel input  (.npy F32, shape (1, 128, T))
//   - Reference conv_out   (.npy F32, shape (num_chunks, T_chunk_out, 896))
//
// Runs the C++ conv front-end on the mel input and compares element-wise
// against the reference. Reports max abs diff and mean abs diff.
//
// Usage:
//   qwen3-asr-test-conv  qwen3-asr-0.6b.gguf  /tmp/qwen3-asr-ref/jfk

#include "qwen3_asr.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Tiny .npy reader (F32, little-endian, supports header v1.0)
// ---------------------------------------------------------------------------
static bool load_npy_f32(const std::string & path,
                         std::vector<float> & data,
                         std::vector<int>   & shape) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }

    char magic[6];
    f.read(magic, 6);
    if (memcmp(magic, "\x93NUMPY", 6) != 0) {
        fprintf(stderr, "%s: not a .npy file\n", path.c_str()); return false;
    }
    uint8_t major, minor;
    f.read((char*)&major, 1); f.read((char*)&minor, 1);
    uint32_t hdr_len = 0;
    if (major == 1) {
        uint16_t hl; f.read((char*)&hl, 2); hdr_len = hl;
    } else {
        f.read((char*)&hdr_len, 4);
    }
    std::string header(hdr_len, '\0');
    f.read(&header[0], hdr_len);

    // Parse: {'descr': '<f4', 'fortran_order': False, 'shape': (..,..), }
    if (header.find("'<f4'") == std::string::npos) {
        fprintf(stderr, "%s: only F32 .npy supported\n", path.c_str()); return false;
    }
    if (header.find("'fortran_order': False") == std::string::npos) {
        fprintf(stderr, "%s: fortran_order=True not supported\n", path.c_str()); return false;
    }
    auto sp = header.find("'shape':");
    if (sp == std::string::npos) return false;
    auto lp = header.find('(', sp);
    auto rp = header.find(')', lp);
    std::string sh = header.substr(lp+1, rp-lp-1);
    shape.clear();
    size_t i = 0;
    while (i < sh.size()) {
        while (i < sh.size() && (sh[i] == ' ' || sh[i] == ',')) i++;
        if (i >= sh.size()) break;
        int v = 0;
        while (i < sh.size() && sh[i] >= '0' && sh[i] <= '9') {
            v = v*10 + (sh[i]-'0'); i++;
        }
        shape.push_back(v);
    }
    size_t total = 1;
    for (int s : shape) total *= (size_t)s;
    data.resize(total);
    f.read((char*)data.data(), total * sizeof(float));
    return (bool)f;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s qwen3-asr-0.6b.gguf REF_DIR\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    std::string  ref_dir    = argv[2];

    // Load reference mel + conv_out
    std::vector<float> mel; std::vector<int> mel_shape;
    if (!load_npy_f32(ref_dir + "/mel_input.npy", mel, mel_shape)) return 2;
    fprintf(stderr, "mel_input.npy shape: ");
    for (int s : mel_shape) fprintf(stderr, "%d ", s); fprintf(stderr, "\n");
    // Expect (1, 128, T)
    int n_mels = 128, T_mel = 0;
    if (mel_shape.size() == 3 && mel_shape[0] == 1 && mel_shape[1] == 128) {
        T_mel = mel_shape[2];
    } else if (mel_shape.size() == 2 && mel_shape[0] == 128) {
        T_mel = mel_shape[1];
    } else {
        fprintf(stderr, "unexpected mel shape\n"); return 3;
    }

    std::vector<float> ref; std::vector<int> ref_shape;
    if (!load_npy_f32(ref_dir + "/conv_out.npy", ref, ref_shape)) return 4;
    fprintf(stderr, "conv_out.npy shape: ");
    for (int s : ref_shape) fprintf(stderr, "%d ", s); fprintf(stderr, "\n");
    // Expect (num_chunks, T_chunk_out, 896)

    // Init model
    auto cp = qwen3_asr_context_default_params();
    cp.n_threads = 4;
    cp.verbosity = 1;
    auto * ctx = qwen3_asr_init_from_file(model_path, cp);
    if (!ctx) { fprintf(stderr, "init failed\n"); return 5; }

    // Run conv
    int n_chunks=0, T_out=0, d=0;
    float * out = qwen3_asr_run_conv(ctx, mel.data(), n_mels, T_mel,
                                     &n_chunks, &T_out, &d);
    if (!out) { fprintf(stderr, "run_conv failed\n"); qwen3_asr_free(ctx); return 6; }

    fprintf(stderr, "C++ conv_out: num_chunks=%d T_chunk_out=%d d=%d  total=%zu floats\n",
            n_chunks, T_out, d, (size_t)n_chunks*T_out*d);

    // Compare. Reference layout: (num_chunks, T_chunk_out, 896) row-major
    //   ref[c][t][k] = ref[(c*T_out + t)*d + k]
    // C++ output is from ggml tensor (d, T_chunk_out, num_chunks) with ne[0]=d
    // varying fastest:
    //   out[c][t][k] = out[(c*T_out + t)*d + k]
    // Same memory layout. Direct compare.
    if ((size_t)n_chunks * T_out * d != ref.size()) {
        fprintf(stderr, "size mismatch: cpp=%zu ref=%zu\n",
                (size_t)n_chunks*T_out*d, ref.size());
        free(out); qwen3_asr_free(ctx); return 7;
    }

    double sum = 0.0, sumsq = 0.0;
    float max_abs = 0.0f;
    int max_i = -1;
    for (size_t i = 0; i < ref.size(); i++) {
        float diff = out[i] - ref[i];
        float ad = std::fabs(diff);
        sum   += ad;
        sumsq += diff * diff;
        if (ad > max_abs) { max_abs = ad; max_i = (int)i; }
    }
    double mean = sum / ref.size();
    double rms  = std::sqrt(sumsq / ref.size());
    fprintf(stderr, "\nDIFF vs reference:\n");
    fprintf(stderr, "  max abs diff:  %.6e  (idx %d)\n", max_abs, max_i);
    fprintf(stderr, "  mean abs diff: %.6e\n", mean);
    fprintf(stderr, "  rms diff:      %.6e\n", rms);
    if (max_i >= 0) {
        fprintf(stderr, "  at idx %d: cpp=%.6f ref=%.6f\n",
                max_i, out[max_i], ref[max_i]);
    }
    int verdict = (max_abs < 1e-2f) ? 0 : 1;
    fprintf(stderr, "  verdict: %s\n", verdict == 0 ? "PASS (max<1e-2)" : "FAIL");
    free(out);

    // ===== Stage 2: full encoder =====
    fprintf(stderr, "\n=== Stage 2: full encoder ===\n");
    std::vector<float> ref2; std::vector<int> ref2_shape;
    if (!load_npy_f32(ref_dir + "/proj2_out.npy", ref2, ref2_shape)) {
        fprintf(stderr, "skipping stage 2 (no proj2_out.npy)\n");
        qwen3_asr_free(ctx); return verdict;
    }
    fprintf(stderr, "proj2_out.npy shape: ");
    for (int s : ref2_shape) fprintf(stderr, "%d ", s); fprintf(stderr, "\n");
    // Expect (N, 1024)
    int ref_N = ref2_shape[0], ref_pdim = ref2_shape[1];

    int N=0, pdim=0;
    float * enc = qwen3_asr_run_encoder(ctx, mel.data(), n_mels, T_mel, &N, &pdim);
    if (!enc) { fprintf(stderr, "run_encoder failed\n"); qwen3_asr_free(ctx); return 8; }
    fprintf(stderr, "C++ encoder output: N=%d pdim=%d\n", N, pdim);

    // C++ memory layout: ggml ne=(pdim, N) → flat (N, pdim) row-major
    if (N != ref_N || pdim != ref_pdim) {
        fprintf(stderr, "shape mismatch: cpp (%d, %d) vs ref (%d, %d)\n",
                N, pdim, ref_N, ref_pdim);
        free(enc); qwen3_asr_free(ctx); return 9;
    }

    double s2sum = 0.0, s2sumsq = 0.0;
    float s2max = 0.0f;
    int s2max_i = -1;
    for (size_t i = 0; i < ref2.size(); i++) {
        float diff = enc[i] - ref2[i];
        float ad = std::fabs(diff);
        s2sum   += ad;
        s2sumsq += diff * diff;
        if (ad > s2max) { s2max = ad; s2max_i = (int)i; }
    }
    fprintf(stderr, "\nDIFF vs proj2_out:\n");
    fprintf(stderr, "  max abs diff:  %.6e  (idx %d)\n", s2max, s2max_i);
    fprintf(stderr, "  mean abs diff: %.6e\n", s2sum / ref2.size());
    fprintf(stderr, "  rms diff:      %.6e\n", std::sqrt(s2sumsq / ref2.size()));
    if (s2max_i >= 0) {
        fprintf(stderr, "  at idx %d: cpp=%.6f ref=%.6f\n",
                s2max_i, enc[s2max_i], ref2[s2max_i]);
    }
    // Per-row cosine similarity — the metric that actually matters for
    // embeddings about to feed an LLM via splice.
    double cos_sum = 0.0, cos_min = 1.0;
    int cos_min_i = -1;
    for (int i = 0; i < N; i++) {
        const float * a = enc + (size_t)i * pdim;
        const float * b = ref2.data() + (size_t)i * pdim;
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (int k = 0; k < pdim; k++) {
            dot += (double)a[k] * b[k];
            na  += (double)a[k] * a[k];
            nb  += (double)b[k] * b[k];
        }
        double cs = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
        cos_sum += cs;
        if (cs < cos_min) { cos_min = cs; cos_min_i = i; }
    }
    fprintf(stderr, "  per-row cosine sim: mean=%.6f min=%.6f (row %d)\n",
            cos_sum / N, cos_min, cos_min_i);

    int v2 = (cos_min > 0.999) ? 0 : 1;  // tight cosine threshold for embeddings
    fprintf(stderr, "  verdict: %s\n", v2 == 0 ? "PASS (cos>0.999)" : "FAIL");
    free(enc);
    qwen3_asr_free(ctx);
    return verdict | (v2 << 1);
}
