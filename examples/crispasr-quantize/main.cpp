// crispasr-quantize — GGUF tensor re-quantization tool.
//
// Takes any GGUF model (whisper, parakeet, canary, cohere, voxtral, qwen3,
// granite, wav2vec2, …) and re-quantizes all eligible tensors to the
// target ggml_ftype, preserving metadata and non-quantizable tensors
// (norms, positional embeddings, biases, small tables) in their
// original types. The logic is model-agnostic — it just iterates the
// GGUF tensor list and calls ggml_quantize_chunk on each float tensor.
//
// Historically lived in examples/cohere-main/cohere-quantize.cpp; moved
// here when the per-model CLIs were consolidated into crispasr.

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "common.h"
#include "common-ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <cmath>

static bool crispasr_model_quantize(const std::string& fname_inp, const std::string& fname_out, ggml_ftype ftype) {
    ggml_type qtype = GGML_TYPE_F32;

    switch (ftype) {
    case GGML_FTYPE_MOSTLY_Q4_0:
        qtype = GGML_TYPE_Q4_0;
        break;
    case GGML_FTYPE_MOSTLY_Q4_1:
        qtype = GGML_TYPE_Q4_1;
        break;
    case GGML_FTYPE_MOSTLY_Q5_0:
        qtype = GGML_TYPE_Q5_0;
        break;
    case GGML_FTYPE_MOSTLY_Q5_1:
        qtype = GGML_TYPE_Q5_1;
        break;
    case GGML_FTYPE_MOSTLY_Q8_0:
        qtype = GGML_TYPE_Q8_0;
        break;
    case GGML_FTYPE_MOSTLY_Q2_K:
        qtype = GGML_TYPE_Q2_K;
        break;
    case GGML_FTYPE_MOSTLY_Q3_K:
        qtype = GGML_TYPE_Q3_K;
        break;
    case GGML_FTYPE_MOSTLY_Q4_K:
        qtype = GGML_TYPE_Q4_K;
        break;
    case GGML_FTYPE_MOSTLY_Q5_K:
        qtype = GGML_TYPE_Q5_K;
        break;
    case GGML_FTYPE_MOSTLY_Q6_K:
        qtype = GGML_TYPE_Q6_K;
        break;
    default:
        fprintf(stderr, "%s: unsupported quantization type %d\n", __func__, ftype);
        return false;
    }

    printf("%s: loading model from '%s'\n", __func__, fname_inp.c_str());

    struct ggml_context* ctx_in_ggml = nullptr;
    struct gguf_init_params params = {};
    params.no_alloc = true;
    params.ctx = &ctx_in_ggml;
    struct gguf_context* ctx_in = gguf_init_from_file(fname_inp.c_str(), params);
    if (!ctx_in || !ctx_in_ggml) {
        fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, fname_inp.c_str());
        return false;
    }

    struct gguf_context* ctx_out = gguf_init_empty();
    gguf_set_kv(ctx_out, ctx_in);
    gguf_set_val_u32(ctx_out, "general.quantization_version", GGML_QNT_VERSION);
    gguf_set_val_u32(ctx_out, "general.file_type", ftype);

    // Detect architecture for arch-specific quantization rules
    std::string arch;
    {
        int key = gguf_find_key(ctx_in, "general.architecture");
        if (key >= 0 && gguf_get_kv_type(ctx_in, key) == GGUF_TYPE_STRING)
            arch = gguf_get_val_str(ctx_in, key);
    }
    const bool is_firered = (arch.find("firered") != std::string::npos);
    const bool is_ecapa = (arch.find("ecapa") != std::string::npos);

    const int n_tensors = gguf_get_n_tensors(ctx_in);
    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(ctx_in, i);
        struct ggml_tensor* t = ggml_get_tensor(ctx_in_ggml, name);
        gguf_add_tensor(ctx_out, t);
    }

    // Allocate output file
    printf("%s: writing quantized model to '%s'\n", __func__, fname_out.c_str());
    FILE* fout = fopen(fname_out.c_str(), "wb");
    if (!fout) {
        fprintf(stderr, "%s: failed to open '%s' for writing\n", __func__, fname_out.c_str());
        gguf_free(ctx_in);
        gguf_free(ctx_out);
        if (ctx_in_ggml)
            ggml_free(ctx_in_ggml);
        return false;
    }

    // Write metadata placeholder
    const size_t meta_size = gguf_get_meta_size(ctx_out);
    std::vector<uint8_t> meta_data(meta_size, 0);
    fwrite(meta_data.data(), 1, meta_size, fout);

    // Open input file for data reading
    FILE* fin = fopen(fname_inp.c_str(), "rb");
    const size_t data_offset_in = gguf_get_data_offset(ctx_in);

    std::vector<float> f32_data;
    std::vector<uint8_t> q_data;

    for (int i = 0; i < n_tensors; i++) {
        const char* name = gguf_get_tensor_name(ctx_in, i);
        struct ggml_tensor* t = ggml_get_tensor(ctx_in_ggml, name);

        enum ggml_type type = t->type;
        size_t size = ggml_nbytes(t);
        size_t offset = data_offset_in + gguf_get_tensor_offset(ctx_in, i);

        printf("[%3d/%3d] %-40s - %10s, ", i + 1, n_tensors, name, ggml_type_name(type));

        std::string sname(name);
        bool is_weight = (sname.find("weight") != std::string::npos) ||
                         // Kyutai STT uses shortened names ending in _w
                         (sname.size() >= 2 && sname.substr(sname.size() - 2) == "_w");
        // FireRedASR/LID: pw1/pw2 convs are stored as 3D [1,in,out] but are
        // effectively 2D matmuls — safe to quantize. Other architectures'
        // 3D conv weights may be actual spatial kernels, so keep the 2D-only
        // rule for them.
        // FireRedASR/LID and ECAPA-TDNN: 3D conv weights (kernel=1 or small kernel)
        // are effectively 2D matmuls — safe to quantize.
        const bool ok_dims = (ggml_n_dims(t) == 2) || ((is_firered || is_ecapa) && ggml_n_dims(t) >= 2);
        bool quantize = ggml_is_quantized(qtype) && (type == GGML_TYPE_F32 || type == GGML_TYPE_F16) && ok_dims &&
                        is_weight && (sname.find("norm") == std::string::npos) &&
                        // Skip projector tensors (Granite Speech: precision-sensitive)
                        (sname.find("proj.") != 0) &&
                        // Skip small classifier heads (ECAPA cosine: 45x192, precision-critical)
                        !(sname.find("cls.") == 0 && ggml_nelements(t) < 65536) &&
                        // Skip OmniASR-LLM bridging tensors (enc_proj, lm_head, tok_emb, lang_emb)
                        (sname.find("enc_proj.") != 0) && (sname.find("lm_head.") != 0) &&
                        (sname.find("tok_emb.") != 0) && (sname.find("lang_emb.") != 0);

        const int64_t ncols = t->ne[0];
        ggml_type qtype_used = qtype;
        int64_t qk_k = ggml_blck_size(qtype_used);

        // Fallback chain for tensors whose row size doesn't divide the
        // requested quant's block size. K-quants need 256-aligned rows;
        // legacy Q4_0/Q5_0/Q8_0 use block 32 and accept any 32-aligned
        // row, which covers the qwen3-asr audio encoder's 896-wide tensors
        // that K-quants would otherwise leave as F16.
        if (quantize && ncols % qk_k != 0) {
            ggml_type fallback = GGML_TYPE_COUNT;
            switch (qtype) {
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
                fallback = GGML_TYPE_Q4_0;
                break;
            case GGML_TYPE_Q5_K:
                fallback = GGML_TYPE_Q5_0;
                break;
            case GGML_TYPE_Q6_K:
                fallback = GGML_TYPE_Q8_0;
                break;
            default:
                break;
            }
            if (fallback != GGML_TYPE_COUNT && ncols % ggml_blck_size(fallback) == 0) {
                qtype_used = fallback;
                qk_k = ggml_blck_size(qtype_used);
                printf("(fallback %s) ", ggml_type_name(qtype_used));
            } else {
                printf("warning: ncols %lld not divisible by %lld, skipping quantization for this tensor\n",
                       (long long)ncols, (long long)qk_k);
                quantize = false;
            }
        }

        // Use 64-bit seek to avoid overflow on files > 2 GB (Windows
        // long is 32-bit even on x86_64, wrapping at 2^31).
#ifdef _WIN32
        _fseeki64(fin, (__int64)offset, SEEK_SET);
#else
        fseeko(fin, (off_t)offset, SEEK_SET);
#endif

        if (quantize) {
            printf("quantizing to %s... ", ggml_type_name(qtype_used));

            const int64_t nelements = ggml_nelements(t);
            f32_data.resize(nelements);

            if (type == GGML_TYPE_F32) {
                if (fread(f32_data.data(), sizeof(float), nelements, fin) != (size_t)nelements) {
                    fprintf(stderr, "failed to read f32 data\n");
                    return false;
                }
            } else {
                std::vector<ggml_fp16_t> f16_data(nelements);
                if (fread(f16_data.data(), sizeof(ggml_fp16_t), nelements, fin) != (size_t)nelements) {
                    fprintf(stderr, "failed to read f16 data\n");
                    return false;
                }
                for (int j = 0; j < nelements; j++)
                    f32_data[j] = ggml_fp16_to_fp32(f16_data[j]);
            }

            const size_t max_q_size = ggml_row_size(qtype_used, t->ne[0]) * (nelements / t->ne[0]);
            q_data.resize(max_q_size);

            size_t q_size = ggml_quantize_chunk(qtype_used, f32_data.data(), q_data.data(), 0, nelements / t->ne[0],
                                                t->ne[0], nullptr);

            fwrite(q_data.data(), 1, q_size, fout);
            gguf_set_tensor_type(ctx_out, name, qtype_used);

            // Padding
            size_t pad = GGML_PAD(q_size, GGUF_DEFAULT_ALIGNMENT) - q_size;
            for (size_t j = 0; j < pad; j++)
                fputc(0, fout);

            printf("done\n");
        } else {
            printf("copying... ");
            std::vector<uint8_t> raw_data(size);
            if (fread(raw_data.data(), 1, size, fin) != size) {
                fprintf(stderr, "failed to read raw data\n");
                return false;
            }
            fwrite(raw_data.data(), 1, size, fout);

            // Padding
            size_t pad = GGML_PAD(size, GGUF_DEFAULT_ALIGNMENT) - size;
            for (size_t j = 0; j < pad; j++)
                fputc(0, fout);
            printf("done\n");
        }
    }

    // Write real metadata
    rewind(fout);
    gguf_get_meta_data(ctx_out, meta_data.data());
    fwrite(meta_data.data(), 1, meta_size, fout);

    fclose(fin);
    fclose(fout);
    gguf_free(ctx_in);
    gguf_free(ctx_out);
    ggml_free(ctx_in_ggml);

    return true;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s model-f16.gguf model-quant.gguf type\n", argv[0]);
        ggml_print_ftypes(stderr);
        return 1;
    }

    const std::string fname_inp = argv[1];
    const std::string fname_out = argv[2];
    const ggml_ftype ftype = ggml_parse_ftype(argv[3]);

    if (!crispasr_model_quantize(fname_inp, fname_out, ftype)) {
        fprintf(stderr, "failed to quantize model\n");
        return 1;
    }

    return 0;
}
