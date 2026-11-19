// Native GGUF inference for the ZONOS3 Mini GDN / Full AttnRes model.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "zonos3-generation.h"
#include "zonos3-tokenizer.h"

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <memory>
#include <vector>

constexpr double ZONOS3_FRAME_RATE = 15.0;

struct zonos3_hparams {
    uint32_t n_layer = 0, n_embd = 0, head_dim = 0, n_head = 0, n_head_kv = 0;
    uint32_t n_ff = 0, n_expert = 0, n_ctx_train = 0;
    float rope_freq_base = 10000.0f, rms_eps = 1e-5f, qk_norm_eps = 1e-6f;
    float logit_softcap = 15.0f;
    uint32_t n_codebooks = 0, codebook_size = 0, audio_vocab = 0;
    uint32_t eoa_id = 0, audio_pad_id = 0, text_vocab = 0;
    uint32_t gdn_heads = 0, gdn_head_dim = 0, gdn_conv_kernel = 0;
    // Layer 0 is dense; later layers use the stored top-k.
    std::vector<int32_t> expert_used;
    // Speaker-embedding row and condition rows (loudness, fmax, speaking_rate).
    uint32_t speaker_dim = 0, n_cond = 0;
    std::vector<uint32_t>
        cond_bucket_counts; // [n_cond]: table i has cond_bucket_counts[i] + 1 rows, id cond_bucket_counts[i] = unknown
    std::vector<std::string> cond_names, cond_units;
    std::vector<std::vector<float>> cond_edges; // [n_cond][cond_bucket_counts[i] - 1], strictly increasing
    std::vector<int32_t> silence_frame;         // [n_codebooks] audio-silence frame
    // Text rows beyond the padding row (text_embd rows = text_vocab + 1 + text_extra_tokens): edit marker rows
    // (text_vocab + 1 + j), then the phoneme / start / pause rows (tokenizer.block_first / block_rows), per
    // zonos3/tts/{edit,superbpe}.py.
    uint32_t text_extra_tokens = 0;
    // zonos3.span_end_id: the speech-edit stop id (= audio_pad_id + 1; audio_vocab grows by one); -1 = none.
    int32_t span_end_id = -1;
};

struct zonos3_layer {
    ggml_tensor *attn_norm = nullptr, *ffn_norm = nullptr;
    bool is_gdn = false;
    // Global GQA layers.
    ggml_tensor *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr;
    ggml_tensor *attn_temp = nullptr, *gater = nullptr;
    // GDN layers: projection -> causal convolution -> recurrence -> gated norm.
    ggml_tensor *gdn_in = nullptr, *gdn_conv = nullptr, *gdn_norm = nullptr;
    ggml_tensor *gdn_A_log = nullptr, *gdn_dt_bias = nullptr;
    // Dense SwiGLU or latent MoE; expert stacks are [input, output, expert].
    ggml_tensor *ffn_up = nullptr, *ffn_gate = nullptr, *ffn_down = nullptr;
    ggml_tensor *ffn_gate_exps = nullptr, *ffn_up_exps = nullptr, *ffn_down_exps = nullptr;
    ggml_tensor *latent_down = nullptr, *latent_up = nullptr;
    ggml_tensor *router_w = nullptr, *router_norm = nullptr, *router_bias = nullptr;
    ggml_tensor* router_mask = nullptr;
    bool is_moe = false;
    int top_k = 0;
};

struct zonos3_accents; // zonos3-accent.h

struct zonos3_model {
    zonos3_hparams hp;
    std::vector<ggml_tensor*> audio_embd;
    ggml_tensor *text_embd = nullptr, *output_norm = nullptr, *output = nullptr;
    ggml_tensor* attnres_q = nullptr; // [width, 2*layers+1], one query per sublayer and final mix.
    ggml_tensor *speaker_proj_w = nullptr, *speaker_proj_b = nullptr;
    std::vector<ggml_tensor*> cond_embd; // [n_embd, cond_bucket_counts[i] + 1] each; last column = unknown
    zonos3_tokenizer tok;
    std::vector<zonos3_layer> layers;
    ggml_context* ctx_w = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;
    std::vector<ggml_backend_buffer_t>
        bufs_w; // weights: default buffer type first, then CPU extra types (AMX, CPU_REPACK, ...)
    std::map<std::string, ggml_tensor*> tensors;
    bool is_gpu = false;
    ggml_context* ctx_mask = nullptr;
    ggml_backend_buffer_t buf_mask = nullptr;
    // Skipping a layer omits its two sources from the Full AttnRes bank.
    std::vector<char> layer_skip;
    // Optional accent-adapter bank (zonos3_accents_load; not owned). When set, prompts and slots may select an
    // adapter (zonos3_conditioning::accent, zonos3_slot::accent); graphs grow the adapter path only when one does.
    const zonos3_accents* accents = nullptr;
};

// Backend when neither --gpu nor --cpu is given: Metal on Apple Silicon, CPU elsewhere
// (the Linux/Windows GPU release archives stamp --gpu into their launchers).
#if defined(__APPLE__) && defined(__aarch64__)
inline constexpr bool ZONOS3_DEFAULT_USE_GPU = true;
#else
inline constexpr bool ZONOS3_DEFAULT_USE_GPU = false;
#endif

bool zonos3_model_load(zonos3_model& model, const char* path, bool use_gpu);
void zonos3_model_free(zonos3_model& model);
bool zonos3_set_expert_mask(zonos3_model& model, const std::map<int, std::vector<int>>& keep);

// Rows whose summed input embedding is REPLACED before emb_norm. Rows are absolute prompt rows.
struct zonos3_conditioning {
    int speaker_row = -1;         // replaced by speaker_proj·speaker + bias; -1 = none
    std::vector<float> speaker;   // [hp.speaker_dim] when speaker_row >= 0
    int condition_row = -1;       // first of hp.n_cond rows; row i replaced by cond_embd[i][buckets[i]]
    std::vector<int32_t> buckets; // [hp.n_cond], each 0..hp.cond_bucket_counts[i] (cond_bucket_counts[i] = unknown)
    // Optional, default off:
    int accent = -1;              // accent adapter of model.accents for these rows (prefill/extend/uncached
    float accent_strength = 0.0f; // forward): y += strength * adapter; -1 or strength 0 = none
};

// Input rows are [positions, codebooks+1] floats containing integral token IDs.
bool zonos3_validate(const zonos3_model& model, const float* ids, int n_tokens, const zonos3_conditioning& cond,
                     const char* out_dir, int n_layer_limit);
bool zonos3_logits(const zonos3_model& model, const float* ids, int n_tokens, const zonos3_conditioning& cond,
                   std::vector<float>& out_logits);

struct zonos3_moe_act {
    int layer = 0, n_embd = 0, n_ff = 0, k = 0, n = 0;
    std::vector<float> moe_in; // [n, latent]
    std::vector<float> moe_y;  // [n, top-k, hidden]
    std::vector<int32_t> sel;  // [n, top-k]
};
bool zonos3_moe_capture(const zonos3_model& model, const float* ids, int n_tokens, const zonos3_conditioning& cond,
                        std::vector<zonos3_moe_act>& out);

struct zonos3_prompt_options {
    const int32_t* reference_codes = nullptr; // ordinary undelayed [frames,10] (reference mode)
    int reference_frames = 0;                 // 0..reference_max_frames (zonos3_generation())
    const float* speaker_embedding = nullptr; // [hp.speaker_dim] (embedding mode); exclusive with reference
    std::vector<int32_t> condition_buckets;   // empty = all unknown, else hp.n_cond entries
    int silence_prefill_frames = zonos3_generation().silence_prefill_default_frames; // 0..silence_prefill_max_frames
};
struct zonos3_prompt {
    std::vector<int32_t> ids; // [n_rows, n_codebooks+1]
    int n_rows = 0;
    int speaker_rows = 0;   // leading rows the CFG twin drops: 1 (speaker row), frames+9 (reference) or 0
    int prefill_frames = 0; // trailing silence rows = first rows of the audio stream
    zonos3_conditioning cond;
};
// Row order: [speaker row | reference prefix (frames+9)] [loudness][fmax][speaking_rate]
// BOS text EOS [silence-prefill rows]. The text is tokenized by zonos3_text_ids (pronunciation groups become phoneme
// ids on a phoneme-token checkpoint, the start and pause markers their ids; nonverbal markup is ordinary SuperBPE
// text). Returns false on any invalid input, with the reason in *err when given (else on stderr); invalid text
// markers give zonos3_check_text_markers's 400 message.
bool zonos3_build_prompt(const zonos3_model& model, const std::string& text, const zonos3_prompt_options& opt,
                         zonos3_prompt& out, std::string* err = nullptr);
// Speech editing (zonos3/tts/edit.py): edit_marker_rows -- the edit marker rows, then the phoneme / start / pause rows
// end the extra rows (-1 when those rows do not end them) -- and edit_marker_id(i) = text_vocab + 1 + i.
int zonos3_edit_marker_rows(const zonos3_model& model);
inline int32_t zonos3_edit_marker_id(const zonos3_model& model, int index) {
    return (int32_t)model.hp.text_vocab + 1 + index;
}
// Whether the checkpoint can edit speech (zonos3.tts.edit_runtime.EditModel.max_spans > 0): span_end_id and at least one marker row.
inline bool zonos3_supports_editing(const zonos3_model& model) {
    return model.hp.span_end_id >= 0 && zonos3_edit_marker_rows(model) > 0;
}
bool zonos3_build_cfg_twin(const zonos3_model& model, const zonos3_prompt& cond, zonos3_prompt& out);
int zonos3_condition_index(const zonos3_model& model, const std::string& name);  // -1 if unknown
int zonos3_condition_bucket(const zonos3_model& model, int index, double value); // bisect_right; -1 if non-finite
std::vector<int32_t> zonos3_prefill_codes(
    const zonos3_model& model, const zonos3_prompt& p); // [prefill_frames,10] audio columns of the silence rows
bool zonos3_check_conditioning(const zonos3_model& model, const float* ids, int n_rows, const zonos3_conditioning& c);

// Defaults: the registry's v2 sampling defaults (zonos3_generation().sampling).
struct zonos3_sampling {
    float temperature = zonos3_generation().sampling.temperature;
    int top_k = zonos3_generation().sampling.topk;
    float top_p = zonos3_generation().sampling.top_p, min_p = zonos3_generation().sampling.min_p,
          rep_penalty = zonos3_generation().sampling.repetition_penalty;
    int rep_window = zonos3_generation().sampling.repetition_window,
        rep_codebooks = zonos3_generation().sampling.repetition_codebooks;
    uint32_t seed = 0;
    bool greedy = false;
};
struct zonos3_steering; // zonos3-steer.h
struct zonos3_sampler;
using zonos3_frame_cb = std::function<bool(int frame_idx, const int32_t* codes, int ncb)>;

// Alignment probe of one decode slot (LACI / span-follow; model/src/zonos3/tts/probe.py and the sgl server's
// align_probe._readout/_masked). At every global-attention layer the slot's decode query is read against the keys of
// its BOS..EOS text rows: each head's softmax over the text keys (restricted to the head's window when it has one)
// is weighted by the head weights (1 for every probed head) and summed over heads and layers; the host reduces that
// [text_len] distribution to (argmax, centroid). A head with a window (ACI hard mask) also has its attention output
// recomputed over all of the slot's keys with the text keys outside the window at -inf (non-text keys are never
// masked). Both are explicit F32 attention, also on the F16-KV / FlashAttention profile.
struct zonos3_align_probe {
    // Not named `head`: it has a `head` member, and MSVC rejects a member named like its class (C2461).
    struct probe_head {
        int layer = 0, head = 0;
    };
    struct window {
        int layer = 0, head = 0, lo = 0, hi = 0;
    };                             // inclusive text indices (clipped like clip_window)
    std::vector<probe_head> heads; // probed heads (global layers only); empty = no probe
    int text_start = -1;           // absolute prompt row of the text BOS
    int text_len = 0;              // BOS..EOS rows inclusive; <= zonos3_batch_features::probe_max_text
    std::vector<window> windows;   // masked heads; empty = unmasked
    bool enabled() const { return !heads.empty() && text_len > 0; }
};
struct zonos3_probe_readout {
    int argmax = -1;
    float centroid = 0.0f;
};
// The probe's (argmax, centroid) of one [text_len] weighted text distribution (torch argmax: first maximum).
zonos3_probe_readout zonos3_probe_reduce(const float* dist, int text_len);
// (start, length) of the BOS(2)..EOS(3) text rows in the text column of [n_rows, n_codebooks+1] ids
// (zonos3.tts.probe.locate_text_span); false when there is no complete span.
bool zonos3_locate_text_span(const zonos3_model& model, const int32_t* ids, int n_rows, int& start, int& length);

// Optional decode features of a batch context; all default off.
struct zonos3_batch_features {
    int probe_max_text = 0; // > 0: slots may carry an alignment probe whose text span has at most this many rows
};
struct zonos3_feature_io; // graph inputs/outputs of the optional features (zonos3-graph.cpp)

// Global layers own append-only KV bands: CPU and entirely-F32 weight inventories
// use F32 storage/attention; other GPU profiles use F16 KV and FlashAttention.
// Uncached zonos3_logits retains F32 K/V and is not an identical-arithmetic oracle
// for F16 KV profiles. GDN layers own fixed-size FP32 recurrent matrices and
// convolution histories, updated only for active decode columns.
struct zonos3_batch_ctx {
    const zonos3_model* model = nullptr;
    int n_slots = 0, slot_cap = 0, W = 0, ncb = 0, av = 0;
    ggml_context* ctx_kv = nullptr;
    ggml_backend_buffer_t buf_kv = nullptr;
    std::vector<ggml_tensor*> k_cache, v_cache;
    std::vector<ggml_tensor*> gdn_state, gdn_history;
    struct dec_graph {
        int width = 0;
        ggml_context* ctx = nullptr;
        ggml_cgraph* gf = nullptr;
        ggml_gallocr_t galloc = nullptr;
        std::vector<ggml_tensor*> ids;
        ggml_tensor *pos_rope = nullptr, *pos_cache = nullptr, *mask = nullptr;
        ggml_tensor *active = nullptr, *logits = nullptr;
        std::vector<ggml_tensor*> steer;       // [n_embd, width] delta input per steering source (decode only)
        std::shared_ptr<zonos3_feature_io> io; // probe / accent inputs of a feature graph; null on the plain ladder
    };
    std::vector<dec_graph> dec_ladder;
    // Feature ladders (alignment probe, masked probe heads, accent adapters), built on first use; `key` encodes the
    // features and the number of masked-probe rows.
    struct dec_variant {
        int key = 0;
        std::vector<dec_graph> ladder;
    };
    std::vector<dec_variant> dec_variants;
    ggml_gallocr_t galloc_prefill = nullptr;
    std::vector<int32_t> h_ids, h_pos_rope, h_pos_cache;
    std::vector<uint16_t> h_mask;
    std::vector<float> h_active, h_logits;
    std::vector<int> steer_sources; // Full AttnRes bank indices that accept a hidden delta (ascending)
    std::vector<float> h_steer;     // [sources][n_slots][n_embd] staging for the delta inputs
    zonos3_batch_features features;
};
struct zonos3_slot {
    int index = -1;
    bool active = false, done = false;
    int n_past = 0, step = 0, eos_frame = -1, countdown = -1;
    int prefill = 0; // audio-stream rows already present (silence prefill)
    std::vector<int32_t> next_ids;
    std::vector<float> steer; // [steer_sources * n_embd] hidden delta for the next decode step; empty = none
    // Optional, default off:
    zonos3_align_probe probe;     // read at every decode step (needs features.probe_max_text >= probe.text_len)
    zonos3_probe_readout readout; // written by zonos3_batch_step for a slot whose probe is enabled
    int accent = -1;              // accent adapter of model.accents for decode steps; -1 or strength 0 = none
    float accent_strength = 0.0f;
};

bool zonos3_batch_init(zonos3_batch_ctx& bc, const zonos3_model& model, int n_slots, int slot_cap_frames,
                       const std::vector<int>& steer_sources = {}, const zonos3_batch_features& features = {});
void zonos3_batch_free(zonos3_batch_ctx& bc);
// Starting a prefill resets all of this slot's cache state. Long prompts are chunked.
bool zonos3_batch_slot_prefill(zonos3_batch_ctx& bc, int slot, const float* prompt_ids, int n0,
                               const zonos3_conditioning& cond, float* out_logits);
// Continue a slot with an arbitrary chunk at its logical position. Used for
// numerical continuation checks as well as the initial prefill implementation.
// `cond` rows are absolute prompt rows; `position` is the chunk's first absolute row.
bool zonos3_batch_slot_extend(zonos3_batch_ctx& bc, int slot, const float* ids, int n_tokens, int position,
                              const zonos3_conditioning& cond, float* out_logits);
// Prefill two slots in one graph per 256-row chunk (a CFG pair: the conditional prompt and its twin). Each slot is
// reset first; rows of both prompts share every per-row op, attention and GDN run per slot. Equivalent to two
// zonos3_batch_slot_prefill calls (same per-slot state and last-row logits).
bool zonos3_batch_pair_prefill(zonos3_batch_ctx& bc, int slot_a, const float* ids_a, int n_a,
                               const zonos3_conditioning& cond_a, float* out_logits_a, int slot_b, const float* ids_b,
                               int n_b, const zonos3_conditioning& cond_b, float* out_logits_b);
bool zonos3_batch_step(zonos3_batch_ctx& bc, const std::vector<zonos3_slot*>& active);
const float* zonos3_batch_slot_logits(const zonos3_batch_ctx& bc, int index);
bool zonos3_slot_sample(const zonos3_model& model, zonos3_slot& slot, zonos3_sampler& smp, const float* logits,
                        std::vector<int32_t>& out_codes, int max_frames, const zonos3_frame_cb& on_frame);
// out[i] = uncond[i] + scale * (cond[i] - uncond[i]) in fp32 (post-softcap logits, before sampling).
void zonos3_cfg_mix(const float* cond, const float* uncond, float scale, size_t n, float* out);

// Optional extras of zonos3_generate (KV path only). The prompt's accent (cond.accent) also applies to every decode
// step of the conditional slot; the CFG twin keeps its own cond.
struct zonos3_generate_features {
    zonos3_align_probe probe;                              // alignment probe of the conditional slot
    std::vector<zonos3_probe_readout>* readouts = nullptr; // receives one readout per decode step (probe enabled)
    std::vector<float>* step_logits = nullptr;             // diagnostics: conditional (pre-CFG) logits of the prefill's
        // last row, then of every decode step [n_codebooks*audio_vocab]
};

// max_frames counts generated rows. out_codes receives generated rows only; eos_frame is in
// audio-stream units (silence-prefill rows included). cfg_scale != 1 runs the speaker-CFG twin.
int zonos3_generate(const zonos3_model& model, const zonos3_prompt& prompt, int max_frames, const zonos3_sampling& sp,
                    float cfg_scale, std::vector<int32_t>& out_codes, int& eos_frame, bool use_kv = true,
                    std::vector<int32_t>* out_full_ids = nullptr, const zonos3_frame_cb& on_frame = {},
                    const zonos3_steering* steering = nullptr, const zonos3_generate_features* features = nullptr);
