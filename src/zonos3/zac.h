// ZONOS3 Mini codec decoder (mossdual_istft): fixed600k hybrid dual Opus codec, 15 Hz, 10 x 4096
// codebooks, 44.1 kHz. Semantic RVQ -> causal ConvNeXt head, + residual RVQ -> 5 windowed-causal MOSS
// transformer stages with patch upsampling (240 Hz features) -> 8 causal ConvNeXt blocks + LayerNorm
// + two spectral projections -> host iSTFT (n_fft 736 and 2208, hop 184) -> FIR mixer (44160 Hz) ->
// exact 44160->44100 sinc resampler.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace pocketfft {
namespace detail {
template <typename T0> class pocketfft_r;
}
} // namespace pocketfft

struct zac_stage_hp {
    int in_dim, out_dim, d_model, n_head, n_layer, n_ff, context, patch;
};

struct zac_model {
    ggml_context* ctx_w = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_type_t buft = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    bool is_gpu = false;
    bool use_fattn = false;  // MOSS attention via ggml_flash_attn_ext (backend supports it)
    bool use_conv1d = false; // encoder K>1 convs via ggml_conv_1d_implicit (backend supports it)
    // CPU only: matmul offload to an accelerator device (BLAS/Accelerate) when one is built in;
    // graphs then run through `sched` instead of a plain CPU allocator.
    ggml_backend_t accel = nullptr;
    ggml_backend_sched_t sched = nullptr;
    // GPU only: decode graphs cached per (width, valid frames), built on first use (decode_chunk); a graph binds
    // one stream's K/V caches and is rebuilt for another stream. One thread per zac_model.
    struct zac_graph;
    mutable std::map<std::pair<int, int>, std::shared_ptr<zac_graph>> graphs;
    // zac_encoder graphs: allocator reused across calls.
    mutable ggml_gallocr_t galloc = nullptr;
    bool owns_weights = true; // false for zac_clone copies sharing another model's weights
    std::map<std::string, ggml_tensor*> t;
    int n_codebooks = 10, codebook_size = 4096, latent_dim = 1024, semantic_dim = 768, sem_depth = 4;
    int sample_rate = 44100, internal_rate = 44160, frame_rate = 15, hop = 2940, internal_hop = 2944;
    std::vector<zac_stage_hp> stages;
    float ln_eps = 1e-5f, convnext_eps = 1e-6f;
    // MR-iSTFT head: depth ConvNeXt blocks, per-branch spectrum projections, host synthesis.
    int head_depth = 0, head_hop = 0, fir_kernel = 0;
    std::array<int, 2> n_fft{};
    float max_log_mag = 0, headroom_db = 0;
    std::string head_variant; // mossdual_istft.head.variant (keep2_plain | full_mr): the decoder id of the servers
    std::vector<float> mixer; // [2][fir_kernel]
    std::array<std::vector<float>, 2> window;                                          // periodic Hann, [n_fft[i]]
    std::array<std::shared_ptr<const pocketfft::detail::pocketfft_r<float>>, 2> irfft; // shared by clones
    int rs_orig = 736, rs_new = 735, rs_width = 17;
    std::vector<float> rs_kernel; // host [rs_new][2*rs_width + rs_orig]
    // Host copies of res.bias (the residual out_proj biases summed, [latent]) and of the per-codebook res.{i}.bias
    // ([n_codebooks - 1][latent]; empty: a GGUF without them, which cannot leave residual codebooks out).
    std::vector<float> res_bias, res_cb_bias;
    ggml_tensor* get(const std::string& n) const;
};

// Incremental decode state. Frames are decoded in order; each call appends 44.1 kHz audio.
// Owns a per-stream MOSS K/V ring buffer on the codec backend (allocated on first decode).
struct zac_stream {
    int n_done = 0;                       // 15 Hz frames decoded
    std::vector<std::vector<float>> conv; // CPU: per history-consuming conv op, [C, R]
    // Host iSTFT, per branch i: overlap-add / window-envelope of padded positions [ola_base[i], ..).
    std::array<std::vector<float>, 2> ola, env;
    std::array<int64_t, 2> ola_base{};
    int64_t head_frames = 0; // 240 Hz head frames synthesized so far
    bool head_done = false;  // flushed
    // Mixer input: branch samples s_c, offset by the FIR half-width h (index j holds s[j - h]);
    // mix_in[i][0] has absolute index mix_base, mix_avail[i] = count of indices produced, mix_next = next output sample.
    std::array<std::vector<float>, 2> mix_in;
    std::array<int64_t, 2> mix_avail{};
    int64_t mix_base = 0, mix_next = 0;
    std::vector<float> rs_buf;
    int64_t rs_base = 0;                  // retained internal samples, absolute index of rs_buf[0]
    int64_t rs_in = 0, rs_next_block = 0; // internal samples received; next 735-sample output block
    const zac_model* owner = nullptr;     // model whose backend holds the cache
    ggml_context* ctx_kv = nullptr;
    ggml_backend_buffer_t buf_kv = nullptr;
    std::vector<ggml_tensor*> kc, vc; // per MOSS layer, stage-major: [head_dim, cap_s, n_head], F16 GPU / F32 CPU
    std::vector<int> cap;             // per stage ring capacity (rows)
    uint64_t kv_id = 0;               // unique per K/V allocation: identifies the caches a cached graph binds
    ggml_context* ctx_hist = nullptr;
    ggml_backend_buffer_t buf_hist = nullptr;
    std::vector<ggml_tensor*> hist; // GPU: conv histories [C, R] per history op, device-resident

    zac_stream() = default;
    ~zac_stream();
    zac_stream(const zac_stream&) = delete;
    zac_stream& operator=(const zac_stream&) = delete;
    zac_stream(zac_stream&& o) noexcept;
    zac_stream& operator=(zac_stream&& o) noexcept;
};

bool zac_load(zac_model& m, const char* path, bool use_gpu);
void zac_free(zac_model& m);
// dst gets its own backend/scheduler but shares src's weights (read-only).
// src must outlive dst; free dst first.
bool zac_clone(zac_model& dst, const zac_model& src);
// 44.1 kHz output samples per 15 Hz frame.
int zac_samples_per_frame(const zac_model& m);
// Whether the model can decode a frame with residual codebooks left out (its GGUF has res.{i}.bias).
bool zac_partial_frames(const zac_model& m);
// Decode frames [st.n_done, f_hi) (f_hi <= H) of the sheared code matrix `codes` ([H, n_codebooks], codebook c
// of frame t read from row t+c) and append the resampled audio that is final so far. A residual cell (c >= 1)
// whose row is >= H or whose code is negative is missing: that codebook's term (table row and res.{c-1}.bias) is
// left out of the frame's RVQ sum (early partial-frame decode; needs zac_partial_frames). Complete frames decode
// exactly as without missing cells. The resampler holds back its lookahead until zac_stream_flush. The stream
// allocates its K/V cache on m's backend on first use and must always be used with the same `m`
// (and destroyed before zac_free(m)).
bool zac_stream_decode(const zac_model& m, zac_stream& st, const int32_t* codes, int H, int f_hi,
                       std::vector<float>& audio);
// End of stream: flush the iSTFT/mixer holdback and the resampler tail so the total equals n_done * hop samples.
bool zac_stream_flush(const zac_model& m, zac_stream& st, std::vector<float>& audio);
// Whole-utterance decode of min(eos (if >= 0), H - (n_codebooks-1)) frames.
bool zac_decode(const zac_model& m, const int32_t* codes, int H, int W, int eos, std::vector<float>& audio);
// Linear output gain 10^(-db/20) for the decoder's headroom (db < 0 selects the model's own headroom_db).
// zac_decode / zac_stream_* output is raw decoder level; callers apply the gain.
float zac_output_gain(const zac_model& m, double headroom_db);
