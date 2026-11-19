#include "zonos3.h"

#include "compat.h"
#include "backend-device.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <algorithm>
#include <set>

// True when CPU extra buffer type `buft` has kernels for `w` as the weight of the op the graph runs
// on it: MUL_MAT_ID for expert stacks, MUL_MAT otherwise. src1 is F32, as in every backbone matmul.
static bool cpu_extra_buft_supports(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft, const ggml_tensor* w,
                                    bool expert_stack) {
    ggml_init_params ip = {4 * ggml_tensor_overhead(), nullptr, true};
    ggml_context* ctx = ggml_init(ip);
    if (!ctx)
        return false;
    ggml_backend_buffer_t probe = ggml_backend_buft_alloc_buffer(buft, 0);
    bool ok = false;
    if (probe) {
        ggml_tensor* a = ggml_dup_tensor(ctx, w);
        a->buffer = probe;
        ggml_tensor* op = expert_stack
                              ? ggml_mul_mat_id(ctx, a, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w->ne[0], 2, 512),
                                                ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 512))
                              : ggml_mul_mat(ctx, a, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w->ne[0], 512));
        ok = ggml_backend_dev_supports_op(dev, op);
        ggml_backend_buffer_free(probe);
    }
    ggml_free(ctx);
    return ok;
}

bool zonos3_model_load(zonos3_model& m, const char* path, bool use_gpu) {
    zonos3_model_free(m);
    ggml_context* meta = nullptr;
    gguf_context* gf = gguf_init_from_file(path, {true, &meta});
    if (!gf) {
        fprintf(stderr, "zonos3: cannot open '%s'\n", path);
        return false;
    }
    auto fail = [&]() {
        if (gf)
            gguf_free(gf);
        if (meta && meta != m.ctx_w)
            ggml_free(meta);
        zonos3_model_free(m);
        return false;
    };
    bool ok = true;
    auto u32 = [&](const char* key, uint32_t& dst) {
        const int64_t id = gguf_find_key(gf, key);
        if (id < 0 || gguf_get_kv_type(gf, id) != GGUF_TYPE_UINT32) {
            fprintf(stderr, "zonos3: missing or invalid uint32 '%s'\n", key);
            ok = false;
        } else
            dst = gguf_get_val_u32(gf, id);
    };
    auto f32 = [&](const char* key, float& dst) {
        const int64_t id = gguf_find_key(gf, key);
        if (id < 0 || gguf_get_kv_type(gf, id) != GGUF_TYPE_FLOAT32) {
            fprintf(stderr, "zonos3: missing or invalid float32 '%s'\n", key);
            ok = false;
        } else
            dst = gguf_get_val_f32(gf, id);
    };
    auto str = [&](const char* key) -> const char* {
        const int64_t id = gguf_find_key(gf, key);
        return id >= 0 && gguf_get_kv_type(gf, id) == GGUF_TYPE_STRING ? gguf_get_val_str(gf, id) : nullptr;
    };
    auto equals = [&](const char* key, const char* value) {
        const char* s = str(key);
        return s && !strcmp(s, value);
    };
    auto& hp = m.hp;
    u32("zonos3.block_count", hp.n_layer);
    u32("zonos3.embedding_length", hp.n_embd);
    u32("zonos3.head_dim", hp.head_dim);
    u32("zonos3.attention.head_count", hp.n_head);
    u32("zonos3.attention.head_count_kv", hp.n_head_kv);
    u32("zonos3.feed_forward_length", hp.n_ff);
    u32("zonos3.expert_count", hp.n_expert);
    u32("zonos3.context_length", hp.n_ctx_train);
    f32("zonos3.rope.freq_base", hp.rope_freq_base);
    f32("zonos3.attention.layer_norm_rms_epsilon", hp.rms_eps);
    f32("zonos3.attention.qk_norm_epsilon", hp.qk_norm_eps);
    f32("zonos3.logit_softcap", hp.logit_softcap);
    u32("zonos3.n_codebooks", hp.n_codebooks);
    u32("zonos3.codebook_size", hp.codebook_size);
    u32("zonos3.audio_vocab", hp.audio_vocab);
    u32("zonos3.eoa_id", hp.eoa_id);
    u32("zonos3.audio_pad_id", hp.audio_pad_id);
    u32("zonos3.text_vocab", hp.text_vocab);
    u32("zonos3.gdn.head_count", hp.gdn_heads);
    u32("zonos3.gdn.head_dim", hp.gdn_head_dim);
    u32("zonos3.gdn.conv_kernel", hp.gdn_conv_kernel);
    uint32_t offset = 0, bos = 0, eos = 0;
    u32("zonos3.tokenizer.offset", offset);
    u32("zonos3.tokenizer.bos", bos);
    u32("zonos3.tokenizer.eos", eos);
    u32("zonos3.speaker.embedding_length", hp.speaker_dim);
    u32("zonos3.condition.count", hp.n_cond);
    if (const int64_t id = gguf_find_key(gf, "zonos3.condition.bucket_counts"); id >= 0) {
        if (gguf_get_kv_type(gf, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(gf, id) != GGUF_TYPE_INT32 ||
            gguf_get_arr_n(gf, id) != hp.n_cond) {
            fprintf(stderr, "zonos3: invalid array 'zonos3.condition.bucket_counts'\n");
            ok = false;
        } else {
            const int32_t* c = (const int32_t*)gguf_get_arr_data(gf, id);
            for (uint32_t i = 0; i < hp.n_cond; ++i)
                hp.cond_bucket_counts.push_back(c[i] < 0 ? 0 : (uint32_t)c[i]);
        }
    } else {
        // Legacy GGUFs (mect_60k) carry one shared count.
        uint32_t shared = 0;
        u32("zonos3.condition.bucket_count", shared);
        hp.cond_bucket_counts.assign(hp.n_cond, shared);
    }
    auto arr_of = [&](const char* key, gguf_type type) -> int64_t {
        const int64_t id = gguf_find_key(gf, key);
        if (id < 0 || gguf_get_kv_type(gf, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(gf, id) != type) {
            fprintf(stderr, "zonos3: missing or invalid array '%s'\n", key);
            ok = false;
            return -1;
        }
        return id;
    };
    auto str_array = [&](const char* key, std::vector<std::string>& dst) {
        const int64_t id = arr_of(key, GGUF_TYPE_STRING);
        if (id < 0)
            return;
        for (size_t i = 0; i < gguf_get_arr_n(gf, id); ++i)
            dst.emplace_back(gguf_get_arr_str(gf, id, i));
    };
    str_array("zonos3.condition.names", hp.cond_names);
    str_array("zonos3.condition.units", hp.cond_units);
    if (ok && hp.cond_names != std::vector<std::string>{"loudness", "fmax", "speaking_rate"}) {
        fprintf(stderr, "zonos3: unexpected condition names\n");
        ok = false;
    }
    bool counts_ok = hp.cond_bucket_counts.size() == hp.n_cond;
    for (uint32_t n : hp.cond_bucket_counts)
        counts_ok = counts_ok && n >= 2 && n <= 1024;
    if (ok && (hp.n_cond != hp.cond_names.size() || hp.cond_units.size() != hp.cond_names.size() || !counts_ok ||
               hp.speaker_dim == 0 || hp.speaker_dim > 65536)) {
        fprintf(stderr, "zonos3: invalid speaker/condition dimensions\n");
        ok = false;
    }
    if (ok) {
        for (size_t c = 0; c < hp.cond_names.size(); ++c) {
            const std::string& name = hp.cond_names[c];
            std::vector<float> edges;
            const int64_t id = arr_of(("zonos3.condition.edges." + name).c_str(), GGUF_TYPE_FLOAT32);
            if (id >= 0) {
                const float* e = (const float*)gguf_get_arr_data(gf, id);
                edges.assign(e, e + gguf_get_arr_n(gf, id));
            }
            bool valid = edges.size() == (size_t)hp.cond_bucket_counts[c] - 1;
            for (size_t i = 0; valid && i < edges.size(); ++i)
                valid = std::isfinite(edges[i]) && (i == 0 || edges[i] > edges[i - 1]);
            if (!valid) {
                fprintf(stderr, "zonos3: invalid condition edges for '%s'\n", name.c_str());
                ok = false;
            }
            hp.cond_edges.push_back(std::move(edges));
        }
    }
    {
        const int64_t id = arr_of("zonos3.silence_frame", GGUF_TYPE_INT32);
        if (id >= 0) {
            const int32_t* s = (const int32_t*)gguf_get_arr_data(gf, id);
            hp.silence_frame.assign(s, s + gguf_get_arr_n(gf, id));
        }
        bool valid = hp.silence_frame.size() == hp.n_codebooks;
        for (int32_t v : hp.silence_frame)
            valid = valid && v >= 0 && (uint32_t)v < hp.codebook_size;
        if (!valid) {
            fprintf(stderr, "zonos3: invalid silence_frame\n");
            ok = false;
        }
    }
    const char* tokenizer = str("zonos3.tokenizer.json");
    const char* spec_json = str("zonos3.text_tokenizer_spec");
    {
        // zonos3.span_end_id (edit checkpoints): the stop id after the padding id; the audio vocabulary grows by one.
        const int64_t id = gguf_find_key(gf, "zonos3.span_end_id");
        if (id >= 0) {
            if (gguf_get_kv_type(gf, id) != GGUF_TYPE_UINT32) {
                fprintf(stderr, "zonos3: invalid span_end_id\n");
                ok = false;
            } else
                hp.span_end_id = (int32_t)gguf_get_val_u32(gf, id);
        }
    }
    // Text rows from the embedding's shape: text_vocab + 1 (padding) + the extra rows (edit markers, phonemes).
    const ggml_tensor* text_meta = meta ? ggml_get_tensor(meta, "text_embd.weight") : nullptr;
    if (text_meta && text_meta->ne[1] >= (int64_t)hp.text_vocab + 1)
        hp.text_extra_tokens = (uint32_t)(text_meta->ne[1] - (int64_t)hp.text_vocab - 1);
    const uint32_t audio_vocab = hp.span_end_id >= 0 ? 4099 : 4098;
    if (!ok || !equals("general.architecture", "zonos3") || !equals("zonos3.attention.pattern", "gdn3_global1") ||
        !equals("zonos3.residual.type", "full_attnres") || !equals("zonos3.codec", "mossdual") ||
        !equals("zonos3.moe.balancing", "quantile") || !tokenizer || !spec_json || offset != 4 || bos != 2 ||
        eos != 3 || hp.n_codebooks != 10 || hp.codebook_size != 4096 || hp.audio_vocab != audio_vocab ||
        hp.eoa_id != 4096 || hp.audio_pad_id != 4097 || hp.text_vocab != 8196 ||
        (hp.span_end_id >= 0 && hp.span_end_id != (int32_t)hp.audio_pad_id + 1) || !text_meta ||
        text_meta->ne[1] < (int64_t)hp.text_vocab + 1 || !hp.n_layer || hp.n_layer > 128 || !hp.n_embd || !hp.n_ff ||
        !hp.n_ctx_train || hp.n_ctx_train > 32768 || !hp.n_head || !hp.n_head_kv || hp.n_head % hp.n_head_kv ||
        !hp.head_dim || hp.head_dim % 2 || !hp.gdn_heads || hp.gdn_heads > 64 || hp.gdn_head_dim != 128 ||
        hp.gdn_conv_kernel != 4 || !std::isfinite(hp.rms_eps) || hp.rms_eps <= 0 || !std::isfinite(hp.qk_norm_eps) ||
        hp.qk_norm_eps <= 0 || !std::isfinite(hp.rope_freq_base) || hp.rope_freq_base <= 0 ||
        !std::isfinite(hp.logit_softcap)) {
        fprintf(stderr, "zonos3: incompatible backbone '%s'%s; reconvert with tools/convert_gguf.py lm\n", path,
                ok && !spec_json ? " (no zonos3.text_tokenizer_spec)" : "");
        return fail();
    }
    std::string err;
    if (!zonos3_tokenizer_load(m.tok, tokenizer, (int32_t)offset, (int32_t)bos, (int32_t)eos, err) ||
        !zonos3_tokenizer_set_spec(m.tok, spec_json, hp.text_vocab, hp.text_extra_tokens, err)) {
        fprintf(stderr, "zonos3: %s\n", err.c_str());
        return fail();
    }
    const int64_t eu = gguf_find_key(gf, "zonos3.expert_used_count_per_layer");
    if (eu < 0 || gguf_get_arr_type(gf, eu) != GGUF_TYPE_INT32 || gguf_get_arr_n(gf, eu) != hp.n_layer) {
        fprintf(stderr, "zonos3: invalid expert_used_count_per_layer\n");
        return fail();
    }
    const int32_t* used = (const int32_t*)gguf_get_arr_data(gf, eu);
    hp.expert_used.assign(used, used + hp.n_layer);
    for (uint32_t i = 0; i < hp.n_layer; ++i) {
        if ((i == 0 && hp.expert_used[i] != 0) ||
            (i > 0 && (hp.expert_used[i] <= 0 || hp.expert_used[i] > 2 || (uint32_t)hp.expert_used[i] > hp.n_expert))) {
            fprintf(stderr, "zonos3: invalid expert count in layer %u\n", i);
            return fail();
        }
    }
    ggml_backend_dev_t dev = zonos3_backend_device(use_gpu);
    if (!dev) {
        fprintf(stderr, "zonos3: no backend\n");
        return fail();
    }
    // Match distributed.matmul_allow_tf32=false before CUDA/cuBLAS is initialized.
    setenv("NVIDIA_TF32_OVERRIDE", "0", 0);
    // Metal: 4 extra command-buffer encoder threads (patches/metal-ncb-env.patch). Measured on
    // M3 with ggml v0.27 decode-bench: 1/2/4 -> 78/97/90 frames/s at 1 slot, 487/499/616 at 8.
    if (use_gpu)
        setenv("GGML_METAL_NCB", "4", 0);
    m.backend = ggml_backend_dev_init(dev, nullptr);
    m.buft = ggml_backend_dev_buffer_type(dev);
    m.is_gpu = zonos3_backend_is_gpu(dev);
    m.ctx_w = meta;
    if (!m.backend) {
        fprintf(stderr, "zonos3: backend initialization failed\n");
        return fail();
    }
    std::vector<ggml_backend_buffer_type_t> extra_bufts;
    if (!m.is_gpu) {
        auto get_extra = (ggml_backend_dev_get_extra_bufts_t)ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_get_extra_bufts");
        if (get_extra)
            for (ggml_backend_buffer_type_t* b = get_extra(dev); b && *b; ++b)
                extra_bufts.push_back(*b);
    }
    // Quantized checkpoints (general.file_type not F32/F16, e.g. backbone-ptq.gguf) run their F16 logits head as q8_0,
    // quantized here at load: half the bytes of the largest per-step matmul. Full-precision checkpoints keep F16.
    ggml_tensor* q8_head = nullptr;
    {
        const int64_t ft = gguf_find_key(gf, "general.file_type");
        const bool quantized_file =
            ft >= 0 && gguf_get_kv_type(gf, ft) == GGUF_TYPE_UINT32 && gguf_get_val_u32(gf, ft) > 1;
        ggml_tensor* t = ggml_get_tensor(meta, "output.weight");
        if (quantized_file && t && t->type == GGML_TYPE_F16 && t->ne[0] % ggml_blck_size(GGML_TYPE_Q8_0) == 0 &&
            ggml_is_contiguous(t)) {
            t->type = GGML_TYPE_Q8_0;
            t->nb[0] = ggml_type_size(GGML_TYPE_Q8_0);
            t->nb[1] = ggml_row_size(GGML_TYPE_Q8_0, t->ne[0]);
            for (int d = 2; d < GGML_MAX_DIMS; ++d)
                t->nb[d] = t->nb[d - 1] * t->ne[d - 1];
            q8_head = t;
        }
    }
    std::vector<std::pair<ggml_backend_buffer_type_t, std::vector<ggml_tensor*>>> groups;
    groups.push_back({m.buft, {}});
    for (ggml_tensor* t = ggml_get_first_tensor(meta); t; t = ggml_get_next_tensor(meta, t)) {
        ggml_backend_buffer_type_t buft = m.buft;
        const std::string name = ggml_get_name(t);
        const bool rows_table = name.compare(0, 11, "audio_embd.") == 0 || name.compare(0, 10, "cond_embd.") == 0 ||
                                name == "text_embd.weight";
        if (ggml_is_quantized(t->type) && !rows_table) {
            static const char suffix[] = "_exps.weight";
            const size_t sl = sizeof(suffix) - 1;
            const bool expert_stack = name.size() >= sl && name.compare(name.size() - sl, sl, suffix) == 0;
            for (ggml_backend_buffer_type_t b : extra_bufts) {
                if (cpu_extra_buft_supports(dev, b, t, expert_stack)) {
                    buft = b;
                    break;
                }
            }
        }
        auto g = std::find_if(groups.begin(), groups.end(), [&](const auto& p) { return p.first == buft; });
        if (g == groups.end()) {
            groups.push_back({buft, {}});
            g = groups.end() - 1;
        }
        g->second.push_back(t);
    }
    for (auto& group : groups) {
        if (group.second.empty())
            continue;
        ggml_backend_buffer_t buffer =
            ggml_backend_buft_alloc_buffer_n(group.first, group.second.data(), (int)group.second.size());
        if (!buffer) {
            fprintf(stderr, "zonos3: weight allocation failed (%s)\n", ggml_backend_buft_name(group.first));
            return fail();
        }
        m.bufs_w.push_back(buffer);
        if (group.first != m.buft) {
            fprintf(stderr, "zonos3: %s weights: %zu tensors, %.1f MiB\n", ggml_backend_buft_name(group.first),
                    group.second.size(), ggml_backend_buffer_get_size(buffer) / 1048576.0);
        }
    }
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "zonos3: cannot reopen '%s'\n", path);
        return fail();
    }
    std::vector<char> buffer;
    size_t total = 0;
    for (ggml_tensor* t = ggml_get_first_tensor(meta); t; t = ggml_get_next_tensor(meta, t)) {
        const int64_t id = gguf_find_tensor(gf, ggml_get_name(t));
        const size_t n = t == q8_head ? (size_t)ggml_nelements(t) * sizeof(ggml_fp16_t) : ggml_nbytes(t);
        buffer.resize(n);
        if (id < 0 || fseeko(f, gguf_get_data_offset(gf) + gguf_get_tensor_offset(gf, id), SEEK_SET) ||
            fread(buffer.data(), 1, n, f) != n) {
            fprintf(stderr, "zonos3: cannot read tensor '%s'\n", ggml_get_name(t));
            ok = false;
            break;
        }
        if (t == q8_head) {
            std::vector<float> head(ggml_nelements(t));
            ggml_fp16_to_fp32_row((const ggml_fp16_t*)buffer.data(), head.data(), (int64_t)head.size());
            buffer.resize(ggml_nbytes(t));
            ggml_quantize_chunk(GGML_TYPE_Q8_0, head.data(), buffer.data(), 0, ggml_nrows(t), t->ne[0], nullptr);
        }
        ggml_backend_tensor_set(t, buffer.data(), 0, ggml_nbytes(t));
        m.tensors.emplace(ggml_get_name(t), t);
        total += ggml_nbytes(t);
    }
    fclose(f);
    gguf_free(gf);
    gf = nullptr;
    if (!ok)
        return fail();
    std::set<std::string> consumed;
    auto req = [&](const std::string& name, std::initializer_list<int64_t> shape, bool f32_only = false,
                   bool float_only = false) -> ggml_tensor* {
        const auto it = m.tensors.find(name);
        ggml_tensor* t = it == m.tensors.end() ? nullptr : it->second;
        bool valid = t != nullptr;
        int d = 0;
        if (t) {
            for (int64_t n : shape) {
                if (t->ne[d++] != n)
                    valid = false;
            }
            for (; d < GGML_MAX_DIMS; ++d)
                if (t->ne[d] != 1)
                    valid = false;
            if (f32_only     ? t->type != GGML_TYPE_F32
                : float_only ? (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16)
                             : (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16 &&
                                !ggml_is_quantized(t->type)))
                valid = false;
        }
        if (!valid) {
            fprintf(stderr, "zonos3: missing/invalid tensor '%s'\n", name.c_str());
            ok = false;
            return nullptr;
        }
        consumed.insert(name);
        return t;
    };
    const int D = (int)hp.n_embd, H = (int)hp.gdn_heads, S = (int)hp.gdn_head_dim, I = H * S;
    m.audio_embd.resize(hp.n_codebooks);
    for (uint32_t c = 0; c < hp.n_codebooks; ++c)
        m.audio_embd[c] = req("audio_embd." + std::to_string(c) + ".weight", {D, hp.audio_vocab});
    m.text_embd = req("text_embd.weight", {D, (int64_t)hp.text_vocab + 1 + hp.text_extra_tokens});
    m.output_norm = req("output_norm.weight", {D}, true);
    m.output = req("output.weight", {D, (int64_t)hp.audio_vocab * hp.n_codebooks});
    m.attnres_q = req("attnres_q.weight", {D, 2 * (int64_t)hp.n_layer + 1});
    m.speaker_proj_w = req("speaker_proj.weight", {(int64_t)hp.speaker_dim, D});
    m.speaker_proj_b = req("speaker_proj.bias", {D}, true);
    m.cond_embd.resize(hp.n_cond);
    for (uint32_t i = 0; i < hp.n_cond; ++i)
        m.cond_embd[i] =
            req("cond_embd." + std::to_string(i) + ".weight", {D, (int64_t)hp.cond_bucket_counts[i] + 1}, false, true);
    m.layers.resize(hp.n_layer);
    for (uint32_t L = 0; L < hp.n_layer; ++L) {
        auto& ly = m.layers[L];
        const std::string p = "blk." + std::to_string(L) + ".";
        ly.attn_norm = req(p + "attn_norm.weight", {D}, true);
        ly.ffn_norm = req(p + "ffn_norm.weight", {D}, true);
        ly.is_gdn = (L + 1) % 4 != 0;
        if (ly.is_gdn) {
            ly.gdn_in = req(p + "gdn_in.weight", {D, 4 * I + 2 * H});
            ly.gdn_conv = req(p + "gdn_conv.weight", {hp.gdn_conv_kernel, 3 * I}, true);
            ly.gdn_norm = req(p + "gdn_norm.weight", {S}, true);
            ly.gdn_A_log = req(p + "gdn_A_log", {H}, true);
            ly.gdn_dt_bias = req(p + "gdn_dt_bias", {H}, true);
            ly.wo = req(p + "attn_output.weight", {I, D});
        } else {
            ly.wq = req(p + "attn_q.weight", {D, (int64_t)hp.head_dim * hp.n_head});
            ly.wk = req(p + "attn_k.weight", {D, (int64_t)hp.head_dim * hp.n_head_kv});
            ly.wv = req(p + "attn_v.weight", {D, (int64_t)hp.head_dim * hp.n_head_kv});
            ly.wo = req(p + "attn_output.weight", {(int64_t)hp.head_dim * hp.n_head, D});
            ly.attn_temp = req(p + "attn_temp", {hp.n_head}, true);
            ly.gater = req(p + "attn_gate.weight", {D, hp.n_head});
        }
        ly.top_k = hp.expert_used[L];
        ly.is_moe = ly.top_k > 0;
        if (!ly.is_moe) {
            ly.ffn_up = req(p + "ffn_up.weight", {D, hp.n_ff});
            ly.ffn_gate = req(p + "ffn_gate.weight", {D, hp.n_ff});
            ly.ffn_down = req(p + "ffn_down.weight", {hp.n_ff, D});
        } else {
            const auto it = m.tensors.find(p + "ffn_up_exps.weight");
            if (it == m.tensors.end()) {
                ok = false;
                continue;
            }
            const ggml_tensor* t = it->second;
            const int64_t latent = t->ne[0], hidden = t->ne[1], experts = t->ne[2];
            if (latent <= 0 || hidden <= 0 || experts < ly.top_k || experts > hp.n_expert) {
                ok = false;
                continue;
            }
            ly.ffn_up_exps = req(p + "ffn_up_exps.weight", {latent, hidden, experts});
            ly.ffn_gate_exps = req(p + "ffn_gate_exps.weight", {latent, hidden, experts});
            ly.ffn_down_exps = req(p + "ffn_down_exps.weight", {hidden, latent, experts});
            ly.latent_down = req(p + "ffn_latent_down.weight", {D, latent});
            ly.latent_up = req(p + "ffn_latent_up.weight", {latent, D});
            ly.router_w = req(p + "router.weight", {D, experts});
            ly.router_norm = req(p + "router_norm.weight", {D}, true);
            ly.router_bias = req(p + "router_bias", {experts}, true);
        }
    }
    for (const auto& t : m.tensors)
        if (!consumed.count(t.first)) {
            fprintf(stderr, "zonos3: unmapped tensor '%s'\n", t.first.c_str());
            ok = false;
        }
    if (!ok)
        return fail();
    fprintf(stderr, "zonos3: backend = %s | %s; loaded %zu tensors, %.2f GiB\n", ggml_backend_dev_name(dev),
            ggml_backend_dev_description(dev), m.tensors.size(), total / (1024.0 * 1024.0 * 1024.0));
    return true;
}

void zonos3_model_free(zonos3_model& m) {
    if (m.buf_mask)
        ggml_backend_buffer_free(m.buf_mask);
    if (m.ctx_mask)
        ggml_free(m.ctx_mask);
    for (ggml_backend_buffer_t b : m.bufs_w)
        ggml_backend_buffer_free(b);
    if (m.ctx_w)
        ggml_free(m.ctx_w);
    if (m.backend)
        ggml_backend_free(m.backend);
    m = zonos3_model();
}

// Runtime expert mask — the dynamic equivalent of prune-cli. For each MoE layer in `keep` we
// build an additive [n_expert] vector (0 for kept experts, -inf for dropped) and hang it on the
// layer; build_moe adds it to the router logits before softmax (so the surviving experts'
// probabilities renormalize) and to the top-k selection scores (so dropped experts are never
// routed). The keep-set is identical to what prune-cli would bake, so a mask sweep predicts the
// pruned model's quality without writing any GGUF.
bool zonos3_set_expert_mask(zonos3_model& m, const std::map<int, std::vector<int>>& keep) {
    // Clear any previous mask first.
    if (m.buf_mask) {
        ggml_backend_buffer_free(m.buf_mask);
        m.buf_mask = nullptr;
    }
    if (m.ctx_mask) {
        ggml_free(m.ctx_mask);
        m.ctx_mask = nullptr;
    }
    for (auto& ly : m.layers)
        ly.router_mask = nullptr;

    // Collect the layers we will actually mask (MoE, in range, present in keep).
    std::vector<int> mlayers;
    for (const auto& kv : keep) {
        const int L = kv.first;
        if (L >= 0 && L < (int)m.layers.size() && m.layers[L].is_moe)
            mlayers.push_back(L);
    }
    if (mlayers.empty())
        return true;

    ggml_init_params ip = {(size_t)(mlayers.size() + 1) * ggml_tensor_overhead(), nullptr, /*no_alloc=*/true};
    m.ctx_mask = ggml_init(ip);
    if (!m.ctx_mask) {
        fprintf(stderr, "zonos3: expert-mask ctx alloc failed\n");
        return false;
    }
    for (int L : mlayers) {
        const int ne = (int)m.layers[L].ffn_up_exps->ne[2];
        ggml_tensor* t = ggml_new_tensor_1d(m.ctx_mask, GGML_TYPE_F32, ne);
        ggml_set_name(t, ("router_mask." + std::to_string(L)).c_str());
        m.layers[L].router_mask = t;
    }
    m.buf_mask = ggml_backend_alloc_ctx_tensors(m.ctx_mask, m.backend);
    if (!m.buf_mask) {
        fprintf(stderr, "zonos3: expert-mask buffer alloc failed\n");
        return false;
    }

    int total_dropped = 0;
    for (int L : mlayers) {
        const int ne = (int)m.layers[L].ffn_up_exps->ne[2];
        std::vector<float> h(ne, -INFINITY);
        for (int e : keep.at(L))
            if (e >= 0 && e < ne)
                h[e] = 0.0f;
        for (int e = 0; e < ne; ++e)
            if (h[e] != 0.0f)
                ++total_dropped;
        ggml_backend_tensor_set(m.layers[L].router_mask, h.data(), 0, h.size() * sizeof(float));
    }
    fprintf(stderr, "zonos3: expert mask on %zu MoE layers, %d experts dropped\n", mlayers.size(), total_dropped);
    return true;
}
