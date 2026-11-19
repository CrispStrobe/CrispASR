// Accent adapter bank: accents.gguf -> stacked per-module factors (see zonos3-accent.h).
#include "zonos3-accent.h"
#include "zonos3.h"
#include "zonos3-text.h"

#include "ggml-alloc.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <set>

namespace {

const char* const LINEAR_BASES[] = {"attn_q", "attn_k",   "attn_v",   "attn_output",     "attn_gate",     "gdn_in",
                                    "ffn_up", "ffn_gate", "ffn_down", "ffn_latent_down", "ffn_latent_up", "router"};
const char* const EXPERT_BASES[] = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"};

// "blk.<L>.<leaf>" -> (L, leaf); false otherwise.
bool split_block(const std::string& base, int& layer, std::string& leaf) {
    if (base.compare(0, 4, "blk.") != 0)
        return false;
    const size_t dot = base.find('.', 4);
    if (dot == std::string::npos || dot == 4)
        return false;
    const std::string digits = base.substr(4, dot - 4);
    if (!std::all_of(digits.begin(), digits.end(), [](char c) { return c >= '0' && c <= '9'; }))
        return false;
    layer = std::stoi(digits);
    leaf = base.substr(dot + 1);
    return true;
}

struct host_factor {
    const ggml_tensor *down = nullptr, *up = nullptr;
};

} // namespace

void zonos3_accents_free(zonos3_accents& bank) {
    if (bank.buf)
        ggml_backend_buffer_free(bank.buf);
    if (bank.ctx)
        ggml_free(bank.ctx);
    bank = zonos3_accents();
}

int zonos3_accent_index(const zonos3_accents& bank, const std::string& name) {
    const auto it = std::find(bank.names.begin(), bank.names.end(), name);
    return it == bank.names.end() ? -1 : (int)(it - bank.names.begin());
}

std::vector<std::string> zonos3_accent_names(const zonos3_accents& bank) {
    std::vector<std::string> out;
    for (int i = 0; i < bank.n; ++i)
        if (i != bank.edit)
            out.push_back(bank.names[i]);
    return out;
}

bool zonos3_accents_load(zonos3_accents& bank, const char* path, const zonos3_model& model, std::string& err) {
    zonos3_accents_free(bank);
    ggml_context* data = nullptr;
    gguf_context* gf = gguf_init_from_file(path, {false, &data});
    if (!gf) {
        err = std::string("cannot read accent adapters '") + path + "'";
        return false;
    }
    auto fail = [&](const std::string& message) {
        err = std::string(path) + ": " + message;
        gguf_free(gf);
        if (data)
            ggml_free(data);
        zonos3_accents_free(bank);
        return false;
    };
    auto str_key = [&](const char* key) -> std::string {
        const int64_t id = gguf_find_key(gf, key);
        return id >= 0 && gguf_get_kv_type(gf, id) == GGUF_TYPE_STRING ? gguf_get_val_str(gf, id) : "";
    };
    if (str_key("general.architecture") != "zonos3_accents")
        return fail("not a zonos3_accents GGUF");
    auto str_array = [&](const char* key, std::vector<std::string>& out) {
        const int64_t id = gguf_find_key(gf, key);
        if (id < 0 || gguf_get_kv_type(gf, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(gf, id) != GGUF_TYPE_STRING)
            return false;
        for (size_t i = 0; i < gguf_get_arr_n(gf, id); ++i)
            out.emplace_back(gguf_get_arr_str(gf, id, i));
        return true;
    };
    if (!str_array("zonos3_accents.names", bank.names) || !str_array("zonos3_accents.format", bank.formats))
        return fail("missing zonos3_accents.names / zonos3_accents.format");
    {
        const int64_t id = gguf_find_key(gf, "zonos3_accents.rank");
        if (id < 0 || gguf_get_kv_type(gf, id) != GGUF_TYPE_ARRAY)
            return fail("missing zonos3_accents.rank");
        const gguf_type t = gguf_get_arr_type(gf, id);
        const void* raw = gguf_get_arr_data(gf, id);
        for (size_t i = 0; i < gguf_get_arr_n(gf, id); ++i) {
            if (t == GGUF_TYPE_UINT32)
                bank.ranks.push_back((int)((const uint32_t*)raw)[i]);
            else if (t == GGUF_TYPE_INT32)
                bank.ranks.push_back(((const int32_t*)raw)[i]);
            else if (t == GGUF_TYPE_UINT64)
                bank.ranks.push_back((int)((const uint64_t*)raw)[i]);
            else if (t == GGUF_TYPE_INT64)
                bank.ranks.push_back((int)((const int64_t*)raw)[i]);
            else
                return fail("zonos3_accents.rank is not an integer array");
        }
    }
    bank.n = (int)bank.names.size();
    if (!bank.n || bank.formats.size() != (size_t)bank.n || bank.ranks.size() != (size_t)bank.n)
        return fail("names, format and rank must list the same adapters");
    if (std::set<std::string>(bank.names.begin(), bank.names.end()).size() != bank.names.size())
        return fail("duplicate adapter names");
    for (int i = 0; i < bank.n; ++i) {
        const std::string& f = bank.formats[i];
        if (f == "zonos-accent-lora/3") {
            if (bank.edit >= 0)
                return fail("more than one zonos-accent-lora/3 (speech-edit) adapter");
            if (model.hp.span_end_id < 0)
                return fail("adapter " + bank.names[i] +
                            " is a speech-edit adapter; the backbone lacks its vocabulary "
                            "(convert it with tools/convert_gguf.py lm --edit-adapter)");
            bank.edit = i;
        } else if (f != "zonos-accent-lora/1" && f != "zonos-accent-lora/2") {
            return fail("unsupported adapter format " + f);
        }
    }

    // Collect the per-adapter tensors.
    std::map<std::string, std::vector<host_factor>> linear, experts;
    std::map<int, std::vector<const ggml_tensor*>> router;
    for (int64_t i = 0; i < gguf_get_n_tensors(gf); ++i) {
        const std::string name = gguf_get_tensor_name(gf, i);
        const ggml_tensor* t = ggml_get_tensor(data, name.c_str());
        if (name.compare(0, 4, "acc.") != 0 || !t)
            return fail("unexpected tensor " + name);
        if (t->type != GGML_TYPE_F32)
            return fail(name + " is not f32");
        const size_t dot = name.find('.', 4);
        const std::string digits = dot == std::string::npos ? "" : name.substr(4, dot - 4);
        if (digits.empty() || !std::all_of(digits.begin(), digits.end(), [](char c) { return c >= '0' && c <= '9'; }))
            return fail("tensor " + name + " has no adapter slot");
        const int slot = std::stoi(digits);
        if (slot >= bank.n)
            return fail("tensor " + name + " names adapter slot " + digits + " of " + std::to_string(bank.n));
        const std::string rest = name.substr(dot + 1);
        int layer = -1;
        std::string leaf;
        if (ends_with(rest, ".router_bias")) {
            if (!split_block(rest, layer, leaf) || leaf != "router_bias")
                return fail("bad router shift " + name);
            auto& v = router[layer];
            v.resize(bank.n, nullptr);
            v[slot] = t;
            continue;
        }
        const bool is_down = ends_with(rest, ".lora_down"), is_up = ends_with(rest, ".lora_up");
        if (!is_down && !is_up)
            return fail("unexpected tensor " + name);
        const std::string base = rest.substr(0, rest.size() - (is_down ? 10 : 8));
        bool expert = false, known = base == "output";
        if (split_block(base, layer, leaf)) {
            for (const char* b : LINEAR_BASES)
                known = known || leaf == b;
            for (const char* b : EXPERT_BASES)
                if (leaf == b)
                    known = expert = true;
        }
        if (!known)
            return fail("tensor " + name + " adapts an unsupported module");
        auto& v = (expert ? experts : linear)[base];
        v.resize(bank.n);
        (is_down ? v[slot].down : v[slot].up) = t;
    }

    // Check against the backbone and size the stacks.
    auto model_weight = [&](const std::string& base) -> const ggml_tensor* {
        const auto it = model.tensors.find(base + ".weight");
        return it == model.tensors.end() ? nullptr : it->second;
    };
    for (const auto& entry : linear) {
        const ggml_tensor* w = model_weight(entry.first);
        if (!w || ggml_n_dims(w) != 2)
            return fail("adapter module " + entry.first + " is not a linear of this model");
        for (int a = 0; a < bank.n; ++a) {
            const host_factor& f = entry.second[a];
            if (!f.down && !f.up)
                continue;
            if (!f.down || !f.up)
                return fail(bank.names[a] + ": " + entry.first + " needs lora_down and lora_up");
            const int64_t k = f.down->ne[1];
            // An adapter trained before the vocabulary grew shifts the old rows of the grown head only (accent_format.
            // fit_head): its output factor lacks each codebook block's span-end row.
            const bool head_fit = entry.first == "output" && model.hp.span_end_id >= 0 &&
                                  f.up->ne[1] == w->ne[1] - (int64_t)model.hp.n_codebooks;
            if (ggml_n_dims(f.down) > 2 || ggml_n_dims(f.up) > 2 || f.down->ne[0] != w->ne[0] || f.up->ne[0] != k ||
                (f.up->ne[1] != w->ne[1] && !head_fit) || k <= 0)
                return fail(bank.names[a] + ": " + entry.first + " factors do not match the model weight");
            bank.K = std::max(bank.K, (int)k);
        }
    }
    for (const auto& entry : experts) {
        const ggml_tensor* w = model_weight(entry.first);
        if (!w || ggml_n_dims(w) != 3)
            return fail("adapter module " + entry.first + " is not an expert stack of this model");
        for (int a = 0; a < bank.n; ++a) {
            const host_factor& f = entry.second[a];
            if (!f.down && !f.up)
                continue;
            if (!f.down || !f.up)
                return fail(bank.names[a] + ": " + entry.first + " needs lora_down and lora_up");
            const int64_t r = f.down->ne[1];
            if (f.down->ne[0] != w->ne[0] || f.down->ne[2] != w->ne[2] || f.up->ne[0] != r || f.up->ne[1] != w->ne[1] ||
                f.up->ne[2] != w->ne[2] || r <= 0)
                return fail(bank.names[a] + ": " + entry.first + " factors do not match the model expert stack");
            if (bank.formats[a] == "zonos-accent-lora/1")
                return fail(bank.names[a] + ": a zonos-accent-lora/1 adapter shifts expert weights");
            bank.R = std::max(bank.R, (int)r);
        }
    }
    for (const auto& entry : router) {
        const int L = entry.first;
        if (L < 0 || L >= (int)model.layers.size() || !model.layers[L].is_moe)
            return fail("router shift on layer " + std::to_string(L) + ", which has no MoE router");
        const int64_t E = model.layers[L].router_bias->ne[0];
        auto& shifts = bank.router_bias[L];
        shifts.resize(bank.n);
        for (int a = 0; a < bank.n; ++a)
            if (const ggml_tensor* t = entry.second[a]) {
                if (ggml_n_dims(t) != 1 || t->ne[0] != E)
                    return fail(bank.names[a] + ": router shift size mismatch");
                const float* v = (const float*)t->data;
                shifts[a].assign(v, v + E);
            }
    }

    // Stack: adapter a owns columns [a*K, a*K + k_a) of a linear (and [a*R, a*R + r_a) of an expert stack).
    const size_t n_tensors = 2 * (linear.size() + experts.size()) + 1;
    bank.ctx = ggml_init({n_tensors * ggml_tensor_overhead(), nullptr, true});
    if (!bank.ctx)
        return fail("context allocation failed");
    std::vector<std::pair<ggml_tensor*, std::vector<float>>> staged;
    const int NK = bank.n * bank.K, NR = bank.n * bank.R;
    for (const auto& entry : linear) {
        const ggml_tensor* w = model_weight(entry.first);
        const int64_t in = w->ne[0], out = w->ne[1];
        ggml_tensor* down = ggml_new_tensor_2d(bank.ctx, GGML_TYPE_F32, in, NK);
        ggml_tensor* up = ggml_new_tensor_2d(bank.ctx, GGML_TYPE_F32, NK, out);
        ggml_set_name(down, ("accent." + entry.first + ".down").c_str());
        ggml_set_name(up, ("accent." + entry.first + ".up").c_str());
        std::vector<float> hd((size_t)in * NK, 0.0f), hu((size_t)NK * out, 0.0f);
        for (int a = 0; a < bank.n; ++a) {
            const host_factor& f = entry.second[a];
            if (!f.down)
                continue;
            const int64_t k = f.down->ne[1];
            const float *d = (const float*)f.down->data, *u = (const float*)f.up->data;
            for (int64_t j = 0; j < k; ++j)
                memcpy(hd.data() + ((size_t)a * bank.K + j) * in, d + (size_t)j * in, (size_t)in * sizeof(float));
            // Factor row o -> model row (an ungrown head factor skips each block's last, span-end, row).
            const int64_t rows = f.up->ne[1], block = rows == out ? out : rows / (int64_t)model.hp.n_codebooks;
            for (int64_t o = 0; o < rows; ++o) {
                const int64_t m = rows == out ? o : o + o / block;
                for (int64_t j = 0; j < k; ++j)
                    hu[(size_t)m * NK + (size_t)a * bank.K + j] = u[(size_t)o * k + j];
            }
        }
        bank.linear[entry.first] = {down, up};
        staged.emplace_back(down, std::move(hd));
        staged.emplace_back(up, std::move(hu));
    }
    for (const auto& entry : experts) {
        const ggml_tensor* w = model_weight(entry.first);
        const int64_t in = w->ne[0], out = w->ne[1], E = w->ne[2];
        ggml_tensor* down = ggml_new_tensor_3d(bank.ctx, GGML_TYPE_F32, in, NR, E);
        ggml_tensor* up = ggml_new_tensor_3d(bank.ctx, GGML_TYPE_F32, NR, out, E);
        ggml_set_name(down, ("accent." + entry.first + ".down").c_str());
        ggml_set_name(up, ("accent." + entry.first + ".up").c_str());
        std::vector<float> hd((size_t)in * NR * E, 0.0f), hu((size_t)NR * out * E, 0.0f);
        for (int a = 0; a < bank.n; ++a) {
            const host_factor& f = entry.second[a];
            if (!f.down)
                continue;
            const int64_t r = f.down->ne[1];
            const float *d = (const float*)f.down->data, *u = (const float*)f.up->data;
            for (int64_t e = 0; e < E; ++e) {
                for (int64_t j = 0; j < r; ++j)
                    memcpy(hd.data() + ((size_t)e * NR + (size_t)a * bank.R + j) * in, d + ((size_t)e * r + j) * in,
                           (size_t)in * sizeof(float));
                for (int64_t o = 0; o < out; ++o)
                    for (int64_t j = 0; j < r; ++j)
                        hu[((size_t)e * out + o) * NR + (size_t)a * bank.R + j] = u[((size_t)e * out + o) * r + j];
            }
        }
        bank.experts[entry.first] = {down, up};
        staged.emplace_back(down, std::move(hd));
        staged.emplace_back(up, std::move(hu));
    }
    gguf_free(gf);
    ggml_free(data);
    gf = nullptr;
    data = nullptr;
    if (!staged.empty()) {
        bank.buf = ggml_backend_alloc_ctx_tensors_from_buft(bank.ctx, model.buft);
        if (!bank.buf) {
            err = std::string(path) + ": accent factor allocation failed";
            zonos3_accents_free(bank);
            return false;
        }
        for (const auto& s : staged)
            ggml_backend_tensor_set(s.first, s.second.data(), 0, s.second.size() * sizeof(float));
    }
    fprintf(stderr,
            "zonos3: accent adapters %s: %d adapters, %zu linear modules (K=%d), %zu expert stacks (R=%d), "
            "%zu router shifts\n",
            path, bank.n, bank.linear.size(), bank.K, bank.experts.size(), bank.R, bank.router_bias.size());
    return true;
}
