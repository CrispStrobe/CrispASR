// Accent adapters (accents.gguf, `tools/convert_gguf.py accents`; lingua zonos-accent-lora/1, /2 and /3).
//
// A request served with adapter a at strength s runs `base + s * adapter` (zonos3_sgl/layers/accent.py,
// zonos3/reference/accents.py): every adapted linear adds s * ((x @ down.T) @ up) to its product, the MoE routers add
// s * shift to their balancing biases, and (/2) every routed (token, expert) pair adds s * up[e] @ (down[e] @ row) to its
// gate/up pre-activations and to its expert output. The bank stacks the factors of all adapters so one graph serves
// any mix of rows: an adapted linear computes y += U_all @ ((D_all @ x) * mask) with a per-row mask [n*K, rows]
// holding s on the K columns of the row's adapter (zero elsewhere: rows without an accent get an exact zero shift).
// Experts use the same mask over stacked [in, n*R, E] / [n*R, out, E] factors through ggml_mul_mat_id.
//
// GGUF layout (arch "zonos3_accents"): zonos3_accents.{names,format} [str], .rank [u32] in slot order (0-based);
// f32 tensors acc.<slot>.<base>.lora_down (numpy [K, in]) / .lora_up (numpy [out, K]) for the backbone's GGUF base
// tensor names (blk.L.attn_q, blk.L.attn_k, blk.L.attn_v, blk.L.attn_output, blk.L.attn_gate, blk.L.gdn_in,
// blk.L.ffn_up, blk.L.ffn_gate, blk.L.ffn_down, blk.L.ffn_latent_down, blk.L.ffn_latent_up, blk.L.router, output);
// expert factors acc.<slot>.blk.L.{ffn_gate_exps,ffn_up_exps,ffn_down_exps}.lora_down (numpy [E, r, in]) / .lora_up
// (numpy [E, out, r]); router shifts acc.<slot>.blk.L.router_bias (numpy [E]).
// A zonos-accent-lora/3 slot (at most one) is the speech-edit adapter: edit fills run on it at strength 1 and requests
// cannot name it as an accent; its vocabulary is in the backbone (`convert_gguf.py lm --edit-adapter`), which the
// loader requires (span_end_id).
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <map>
#include <string>
#include <vector>

struct zonos3_model;

struct zonos3_accents {
    std::vector<std::string> names, formats; // slot order: the names requests use
    std::vector<int> ranks;                  // AccentAdapter.rank per adapter
    int n = 0;                               // adapters
    int K = 0;                               // mask columns per adapter for linears (largest linear rank)
    int R = 0;                               // mask columns per adapter for experts (largest expert rank; 0 = none)
    int edit = -1;                           // slot of the speech-edit (/3) adapter; -1 = none
    struct factors {
        ggml_tensor *down = nullptr, *up = nullptr;
    };
    // Linear base name -> down ne [in, n*K], up ne [n*K, out] (zero columns where an adapter lacks the module).
    std::map<std::string, factors> linear;
    // Expert stack base name (blk.L.ffn_{gate,up,down}_exps) -> down ne [in, n*R, E], up ne [n*R, out, E].
    std::map<std::string, factors> experts;
    // MoE layer -> per adapter [E] balancing-bias shift (empty = that adapter does not shift this router).
    std::map<int, std::vector<std::vector<float>>> router_bias;
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
};

// Loads and checks every factor against `model` (allocated on the model's backend). Attach with model.accents = &bank.
bool zonos3_accents_load(zonos3_accents& bank, const char* path, const zonos3_model& model, std::string& err);
void zonos3_accents_free(zonos3_accents& bank);
int zonos3_accent_index(const zonos3_accents& bank, const std::string& name); // -1 if unknown
// The names requests may use as accents: every adapter but the speech-edit one.
std::vector<std::string> zonos3_accent_names(const zonos3_accents& bank);
