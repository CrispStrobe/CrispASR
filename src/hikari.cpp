// hikari.cpp — sbintuitions/hikari-medium via ggml.
//
// Model (github.com/sbintuitions/hikari, models/modeling_hikari.py):
//   Whisper-medium encoder: conv(80->1024,k3,p1)+GELU, conv(k3,s2,p1)+GELU,
//     learned positions [1500], 24 pre-LN layers, self-attention CAUSAL.
//   Whisper-medium decoder: learned positions [375], 24 pre-LN layers, causal
//     self-attention; cross-attention MASKED: decoder position i sees encoder
//     frames j < 4*i (decoder_time_dilation = 4). LM head tied to tok_emb.
//
// Policy (server/model_wrapper.py ModelWrapper.get_one_token), one decision
// per 80 ms chunk (= 1280 samples = 4 encoder frames), from the 3rd chunk on:
//   window = last W*1280 samples (W = decoder_context);
//   mel = whisper log-mel of the window (clip at window-max - 8), zero-padded
//   to 3000 frames; encoder over it; drop ids[4] if len(ids) > W; decoder over
//   ids; logits at pos = len(ids)-1; if argmax != WAIT and argmax in ids[-5:]
//   subtract repetition_penalty from it; subtract wait_penalty from WAIT;
//   append argmax; WAIT x10 during speech -> penalty += boost, emitted token ->
//   penalty -= decay * (penalty - baseline).
//
// The server recomputes the whole window every 80 ms. This port computes the
// same function incrementally, exactly: the encoder is causal and an encoder
// frame e reads only mel frames [2e-2, 2e+2], so when the normalised mel is
// unchanged below frame m0 every encoder frame below ceil((m0-2)/2) is
// unchanged too, and every decoder position i with 4*i <= that frame is
// unchanged. Encoder self-KV, cross-KV and decoder self-KV are cached; each
// step recomputes from the first frame whose normalised mel value actually
// changed (window max moved, right-edge reflect pad, window slide). In the
// common case that is ~5 encoder frames and 2 decoder positions per step.

#include "hikari.h"

#include "core/bpe.h"
#include "core/fft.h"
#include "core/ggml_cpu_backend.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"
#if defined(GGML_USE_METAL)
#include "ggml-metal.h"
#endif

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kChunk = 1280; // 80 ms: 160 hop * 2 (conv stride) * 4 (dilation)
constexpr int kNFft = 400;
constexpr int kHop = 160;
constexpr int kMelFrames = 3000; // the encoder always sees 30 s
constexpr int kVadSamples = 512;

bool bench_enabled() {
    static int v = -1;
    if (v < 0) {
        const char* e = std::getenv("HIKARI_BENCH");
        v = (e && *e && *e != '0') ? 1 : 0;
    }
    return v != 0;
}
double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct hparams {
    int vocab = 51865, d = 1024, n_mels = 80;
    int enc_layers = 24, enc_heads = 16, enc_ffn = 4096;
    int dec_layers = 24, dec_heads = 16, dec_ffn = 4096;
    int n_audio_ctx = 1500, n_text_ctx = 375, dilation = 4;
    int tok_eot = 50257, tok_sot = 50258, tok_translate = 50358, tok_transcribe = 50359, tok_notimestamps = 50363;
    int tok_wait = 93;
};

struct enc_layer {
    ggml_tensor *attn_ln_w, *attn_ln_b, *q_w, *q_b, *k_w, *v_w, *v_b, *o_w, *o_b;
    ggml_tensor *ffn_ln_w, *ffn_ln_b, *up_w, *up_b, *down_w, *down_b;
};
struct dec_layer {
    ggml_tensor *attn_ln_w, *attn_ln_b, *q_w, *q_b, *k_w, *v_w, *v_b, *o_w, *o_b;
    ggml_tensor *x_ln_w, *x_ln_b, *xq_w, *xq_b, *xk_w, *xv_w, *xv_b, *xo_w, *xo_b;
    ggml_tensor *ffn_ln_w, *ffn_ln_b, *up_w, *up_b, *down_w, *down_b;
};

struct bench_acc {
    double mel = 0, enc = 0, dec = 0, vad = 0;
    long steps = 0, enc_frames = 0, dec_positions = 0;
};

} // namespace

struct hikari_context {
    hikari_context_params params{};
    hparams hp;

    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
    ggml_tensor *conv1_w = nullptr, *conv1_b = nullptr, *conv2_w = nullptr, *conv2_b = nullptr;
    ggml_tensor *enc_pos = nullptr, *enc_ln_w = nullptr, *enc_ln_b = nullptr;
    ggml_tensor *tok_emb = nullptr, *dec_pos = nullptr, *dec_ln_w = nullptr, *dec_ln_b = nullptr;
    std::vector<enc_layer> enc;
    std::vector<dec_layer> dec;

    std::vector<float> mel_fb;  // [n_mels, 201]
    std::vector<float> mel_win; // [400]
    std::vector<std::string> id_to_token;
    std::map<std::string, int> lang_to_id;

    ggml_backend_t backend = nullptr, backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> meta;

    // caches: encoder self-KV (hd, 1500, nh, L), cross-KV (hd, 1500, nh, L),
    // decoder self-KV (hd, 375, nh, L) — all F16, on `backend`.
    ggml_context* cache_ctx = nullptr;
    ggml_backend_buffer_t cache_buf = nullptr;
    ggml_tensor *ek = nullptr, *ev = nullptr, *xk = nullptr, *xv = nullptr, *dk = nullptr, *dv = nullptr;

    // task + policy
    std::vector<int32_t> prompt;
    hikari_policy pol{};
    hikari_speech_prob_fn speech_fn = nullptr;
    void* speech_user = nullptr;
    int tail_ms = 2000;

    // stream state
    std::vector<float> audio; // everything since reset
    int64_t n_decided = 0;    // samples consumed by decisions
    int j = -1;
    std::vector<int32_t> ids;
    float wp = 0.0f;
    std::vector<int32_t> step_tok;
    std::vector<double> step_t;
    std::vector<float> step_sp;
    // mel + incremental bookkeeping
    std::unordered_map<int64_t, std::array<float, 80>> raw_cache; // clean frames by absolute index
    int64_t win_start = -1;                                       // samples
    std::vector<float> norm_mel;                                  // [n_frames][80] of the last step
    int enc_valid = 0;                                            // encoder frames valid in the caches
    int dec_valid = 0;                                            // decoder positions valid in the self-KV
    std::vector<int32_t> last_ids;                                // ids of the last decoder pass
    bench_acc bench;
};

// ─────────────────────────────────────────────────────────────────────────────
// Loading

static ggml_tensor* need(hikari_context* c, const std::string& n) {
    auto it = c->tensors.find(n);
    if (it == c->tensors.end()) {
        fprintf(stderr, "hikari: missing tensor '%s'\n", n.c_str());
        return nullptr;
    }
    return it->second;
}

static bool bind(hikari_context* c) {
    bool ok = true;
    auto g = [&](const std::string& n) {
        ggml_tensor* t = need(c, n);
        ok = ok && t;
        return t;
    };
    c->conv1_w = g("enc.conv1.weight");
    c->conv1_b = g("enc.conv1.bias");
    c->conv2_w = g("enc.conv2.weight");
    c->conv2_b = g("enc.conv2.bias");
    c->enc_pos = g("enc.pos_emb");
    c->enc_ln_w = g("enc.out_ln.weight");
    c->enc_ln_b = g("enc.out_ln.bias");
    c->tok_emb = g("dec.tok_emb");
    c->dec_pos = g("dec.pos_emb");
    c->dec_ln_w = g("dec.out_ln.weight");
    c->dec_ln_b = g("dec.out_ln.bias");
    c->enc.resize(c->hp.enc_layers);
    for (int i = 0; i < c->hp.enc_layers; i++) {
        const std::string p = "enc.blk." + std::to_string(i) + ".";
        auto& l = c->enc[i];
        l.attn_ln_w = g(p + "attn_ln.weight");
        l.attn_ln_b = g(p + "attn_ln.bias");
        l.q_w = g(p + "attn_q.weight");
        l.q_b = g(p + "attn_q.bias");
        l.k_w = g(p + "attn_k.weight");
        l.v_w = g(p + "attn_v.weight");
        l.v_b = g(p + "attn_v.bias");
        l.o_w = g(p + "attn_o.weight");
        l.o_b = g(p + "attn_o.bias");
        l.ffn_ln_w = g(p + "ffn_ln.weight");
        l.ffn_ln_b = g(p + "ffn_ln.bias");
        l.up_w = g(p + "ffn_up.weight");
        l.up_b = g(p + "ffn_up.bias");
        l.down_w = g(p + "ffn_down.weight");
        l.down_b = g(p + "ffn_down.bias");
    }
    c->dec.resize(c->hp.dec_layers);
    for (int i = 0; i < c->hp.dec_layers; i++) {
        const std::string p = "dec.blk." + std::to_string(i) + ".";
        auto& l = c->dec[i];
        l.attn_ln_w = g(p + "attn_ln.weight");
        l.attn_ln_b = g(p + "attn_ln.bias");
        l.q_w = g(p + "attn_q.weight");
        l.q_b = g(p + "attn_q.bias");
        l.k_w = g(p + "attn_k.weight");
        l.v_w = g(p + "attn_v.weight");
        l.v_b = g(p + "attn_v.bias");
        l.o_w = g(p + "attn_o.weight");
        l.o_b = g(p + "attn_o.bias");
        l.x_ln_w = g(p + "cross_ln.weight");
        l.x_ln_b = g(p + "cross_ln.bias");
        l.xq_w = g(p + "cross_q.weight");
        l.xq_b = g(p + "cross_q.bias");
        l.xk_w = g(p + "cross_k.weight");
        l.xv_w = g(p + "cross_v.weight");
        l.xv_b = g(p + "cross_v.bias");
        l.xo_w = g(p + "cross_o.weight");
        l.xo_b = g(p + "cross_o.bias");
        l.ffn_ln_w = g(p + "ffn_ln.weight");
        l.ffn_ln_b = g(p + "ffn_ln.bias");
        l.up_w = g(p + "ffn_up.weight");
        l.up_b = g(p + "ffn_up.bias");
        l.down_w = g(p + "ffn_down.weight");
        l.down_b = g(p + "ffn_down.bias");
    }
    // Small host-side tables for the mel front end.
    ggml_tensor* fb = g("audio.mel_filters");
    ggml_tensor* win = g("audio.mel_window");
    if (!ok)
        return false;
    if (fb->type != GGML_TYPE_F32 || win->type != GGML_TYPE_F32 || ggml_nelements(win) != kNFft ||
        ggml_nelements(fb) != (int64_t)c->hp.n_mels * (kNFft / 2 + 1)) {
        fprintf(stderr, "hikari: unexpected mel filter / window tensors\n");
        return false;
    }
    c->mel_fb.resize(ggml_nelements(fb));
    ggml_backend_tensor_get(fb, c->mel_fb.data(), 0, ggml_nbytes(fb));
    c->mel_win.resize(kNFft);
    ggml_backend_tensor_get(win, c->mel_win.data(), 0, ggml_nbytes(win));
    return true;
}

static bool load_meta(hikari_context* c, gguf_context* g) {
    auto& hp = c->hp;
    auto u = [&](const char* k, int d) { return (int)core_gguf::kv_u32(g, k, (uint32_t)d); };
    const std::string arch = core_gguf::kv_str(g, "general.architecture", "");
    if (arch != "hikari") {
        fprintf(stderr, "hikari: general.architecture is '%s', expected 'hikari'\n", arch.c_str());
        return false;
    }
    hp.vocab = u("hikari.vocab_size", hp.vocab);
    hp.d = u("hikari.d_model", hp.d);
    hp.n_mels = u("hikari.n_mels", hp.n_mels);
    hp.enc_layers = u("hikari.encoder.n_layers", hp.enc_layers);
    hp.enc_heads = u("hikari.encoder.n_heads", hp.enc_heads);
    hp.enc_ffn = u("hikari.encoder.ffn_dim", hp.enc_ffn);
    hp.dec_layers = u("hikari.decoder.n_layers", hp.dec_layers);
    hp.dec_heads = u("hikari.decoder.n_heads", hp.dec_heads);
    hp.dec_ffn = u("hikari.decoder.ffn_dim", hp.dec_ffn);
    hp.n_audio_ctx = u("hikari.n_audio_ctx", hp.n_audio_ctx);
    hp.n_text_ctx = u("hikari.n_text_ctx", hp.n_text_ctx);
    hp.dilation = u("hikari.decoder_time_dilation", hp.dilation);
    hp.tok_eot = u("hikari.token.eot", hp.tok_eot);
    hp.tok_sot = u("hikari.token.sot", hp.tok_sot);
    hp.tok_translate = u("hikari.token.translate", hp.tok_translate);
    hp.tok_transcribe = u("hikari.token.transcribe", hp.tok_transcribe);
    hp.tok_notimestamps = u("hikari.token.notimestamps", hp.tok_notimestamps);
    hp.tok_wait = u("hikari.token.wait", hp.tok_wait);
    if (hp.n_mels != 80 || hp.n_audio_ctx * 2 != kMelFrames || hp.dilation * 2 * kHop != kChunk) {
        fprintf(stderr, "hikari: unsupported geometry (n_mels=%d audio_ctx=%d dilation=%d)\n", hp.n_mels,
                hp.n_audio_ctx, hp.dilation);
        return false;
    }
    c->id_to_token = core_gguf::kv_str_array(g, "tokenizer.ggml.tokens");
    if ((int)c->id_to_token.size() < hp.vocab) {
        fprintf(stderr, "hikari: tokenizer has %d tokens, vocab %d\n", (int)c->id_to_token.size(), hp.vocab);
        return false;
    }
    // Decode-back guard: the WAIT id is hard-coded upstream; make sure it is "~".
    if (c->id_to_token[hp.tok_wait] != "~") {
        fprintf(stderr, "hikari: token %d is '%s', expected the WAIT token '~'\n", hp.tok_wait,
                c->id_to_token[hp.tok_wait].c_str());
        return false;
    }
    const auto codes = core_gguf::kv_str_array(g, "hikari.lang_codes");
    const int kid = gguf_find_key(g, "hikari.lang_token_ids");
    if (kid >= 0 && (int)gguf_get_arr_n(g, kid) == (int)codes.size()) {
        const auto* p = (const int32_t*)gguf_get_arr_data(g, kid);
        for (size_t i = 0; i < codes.size(); i++)
            c->lang_to_id[codes[i]] = p[i];
    }
    for (const char* need_lang : {"en", "de", "ja", "ru"}) {
        auto it = c->lang_to_id.find(need_lang);
        if (it == c->lang_to_id.end() || c->id_to_token[it->second] != std::string("<|") + need_lang + "|>") {
            fprintf(stderr, "hikari: language token for '%s' missing or mis-mapped\n", need_lang);
            return false;
        }
    }
    return true;
}

static bool alloc_caches(hikari_context* c) {
    const auto& hp = c->hp;
    const int hd = hp.d / hp.enc_heads;
    ggml_init_params ip = {ggml_tensor_overhead() * 8, nullptr, true};
    c->cache_ctx = ggml_init(ip);
    c->ek = ggml_new_tensor_4d(c->cache_ctx, GGML_TYPE_F16, hd, hp.n_audio_ctx, hp.enc_heads, hp.enc_layers);
    c->ev = ggml_new_tensor_4d(c->cache_ctx, GGML_TYPE_F16, hd, hp.n_audio_ctx, hp.enc_heads, hp.enc_layers);
    const int hdd = hp.d / hp.dec_heads;
    c->xk = ggml_new_tensor_4d(c->cache_ctx, GGML_TYPE_F16, hdd, hp.n_audio_ctx, hp.dec_heads, hp.dec_layers);
    c->xv = ggml_new_tensor_4d(c->cache_ctx, GGML_TYPE_F16, hdd, hp.n_audio_ctx, hp.dec_heads, hp.dec_layers);
    c->dk = ggml_new_tensor_4d(c->cache_ctx, GGML_TYPE_F16, hdd, hp.n_text_ctx, hp.dec_heads, hp.dec_layers);
    c->dv = ggml_new_tensor_4d(c->cache_ctx, GGML_TYPE_F16, hdd, hp.n_text_ctx, hp.dec_heads, hp.dec_layers);
    c->cache_buf = ggml_backend_alloc_ctx_tensors(c->cache_ctx, c->backend);
    if (!c->cache_buf)
        return false;
    ggml_backend_buffer_clear(c->cache_buf, 0);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Mel front end (openai-whisper log_mel_spectrogram, per frame)

// Raw log10 mel of frame m of `x` (length n), torch.stft center=True reflect.
static void raw_mel_frame(const hikari_context* c, const float* x, int n, int m, float* out80) {
    static thread_local std::vector<float> buf(kNFft), spec(2 * kNFft);
    for (int k = 0; k < kNFft; k++) {
        int idx = m * kHop - kNFft / 2 + k;
        if (idx < 0)
            idx = -idx;
        else if (idx >= n)
            idx = 2 * (n - 1) - idx;
        buf[k] = x[idx] * c->mel_win[k];
    }
    core_fft::fft_nonpow2_r2c(buf.data(), kNFft, spec.data());
    const int nf = kNFft / 2 + 1;
    double pw[kNFft / 2 + 1];
    for (int k = 0; k < nf; k++)
        pw[k] = (double)spec[2 * k] * spec[2 * k] + (double)spec[2 * k + 1] * spec[2 * k + 1];
    for (int b = 0; b < c->hp.n_mels; b++) {
        const float* f = c->mel_fb.data() + (size_t)b * nf;
        double s = 0.0;
        for (int k = 0; k < nf; k++)
            s += f[k] * pw[k];
        out80[b] = (float)std::log10(std::max(s, 1e-10));
    }
}

// Normalised mel of the window [ws, t) into c->norm_mel ([frames][80]).
// Frames whose 400-sample span lies inside real audio are cached by absolute
// index; the two left-edge frames and the right-edge frame are recomputed.
static void window_mel(hikari_context* c, int64_t ws, int64_t t, std::vector<float>& out) {
    const float* x = c->audio.data() + ws;
    const int n = (int)(t - ws);
    const int nfr = n / kHop;
    const int64_t abs0 = ws / kHop;
    std::vector<float> raw((size_t)nfr * 80);
    float mx = -std::numeric_limits<float>::infinity();
    for (int m = 0; m < nfr; m++) {
        float* dst = raw.data() + (size_t)m * 80;
        const bool clean = m * kHop - kNFft / 2 >= 0 && m * kHop + kNFft / 2 <= n;
        auto it = clean ? c->raw_cache.find(abs0 + m) : c->raw_cache.end();
        if (it != c->raw_cache.end()) {
            std::memcpy(dst, it->second.data(), 80 * sizeof(float));
        } else {
            raw_mel_frame(c, x, n, m, dst);
            if (clean) {
                std::array<float, 80> a;
                std::memcpy(a.data(), dst, sizeof(a));
                c->raw_cache.emplace(abs0 + m, a);
            }
        }
        for (int b = 0; b < 80; b++)
            mx = std::max(mx, dst[b]);
    }
    for (auto it = c->raw_cache.begin(); it != c->raw_cache.end();)
        it = it->first < abs0 ? c->raw_cache.erase(it) : std::next(it);
    const float floor_v = mx - 8.0f;
    out.resize(raw.size());
    for (size_t i = 0; i < raw.size(); i++)
        out[i] = (std::max(raw[i], floor_v) + 4.0f) / 4.0f;
}

// ─────────────────────────────────────────────────────────────────────────────
// Graphs

static ggml_tensor* ln(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, 1e-5f), w), b);
}

// (D, n) -> (hd, n, nh)
static ggml_tensor* heads(ggml_context* ctx, ggml_tensor* x, int hd, int nh, int n) {
    return ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, x, hd, nh, n), 0, 2, 1, 3));
}

// view of a 4D cache (hd, cap, nh, L) at layer il, rows [off, off+n)
static ggml_tensor* cache_rows(ggml_context* ctx, ggml_tensor* cache, int il, int off, int n) {
    return ggml_view_3d(ctx, cache, cache->ne[0], n, cache->ne[2], cache->nb[1], cache->nb[2],
                        (size_t)il * cache->nb[3] + (size_t)off * cache->nb[1]);
}

// Encoder frames [e0, e1): conv stem on a mel slice, causal self-attention
// against the cached K/V of [0, e0), then cross K/V for every decoder layer.
// want_out: also expose "enc_out" (D, n).
static ggml_cgraph* build_encoder(hikari_context* c, int e0, int e1, bool want_out) {
    const auto& hp = c->hp;
    const int D = hp.d, nh = hp.enc_heads, hd = D / nh, n = e1 - e0;
    const int Lm = 2 * n + 3;
    ggml_init_params ip = {c->meta.size(), c->meta.data(), true};
    ggml_context* ctx = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 8192, false);

    ggml_tensor* mel = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, Lm, hp.n_mels);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);
    ggml_tensor* c1mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2 * n + 1, 1);
    ggml_set_name(c1mask, "c1mask");
    ggml_set_input(c1mask);
    ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    ggml_set_name(pos, "pos");
    ggml_set_input(pos);
    ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, e1, n);
    ggml_set_name(mask, "mask");
    ggml_set_input(mask);

    // conv1 over mel[2e0-2 .. 2e1] -> conv1 frames [2e0-1, 2e1-1]; the frame
    // at -1 is conv2's zero padding, not gelu(conv1(0)), hence the mask.
    ggml_tensor* x = ggml_conv_1d(ctx, c->conv1_w, mel, 1, 0, 1);
    x = ggml_gelu_erf(ctx, ggml_add(ctx, x, ggml_reshape_2d(ctx, c->conv1_b, 1, D)));
    x = ggml_mul(ctx, x, c1mask);
    x = ggml_conv_1d(ctx, c->conv2_w, x, 2, 0, 1);
    x = ggml_gelu_erf(ctx, ggml_add(ctx, x, ggml_reshape_2d(ctx, c->conv2_b, 1, D)));
    x = ggml_cont(ctx, ggml_transpose(ctx, ggml_reshape_2d(ctx, ggml_cont(ctx, x), n, D))); // (D, n)
    x = ggml_add(ctx, x, ggml_get_rows(ctx, c->enc_pos, pos));

    const float scale = 1.0f / std::sqrt((float)hd);
    for (int il = 0; il < hp.enc_layers; il++) {
        const auto& l = c->enc[il];
        ggml_tensor* res = x;
        ggml_tensor* h = ln(ctx, x, l.attn_ln_w, l.attn_ln_b);
        ggml_tensor* Q = heads(ctx, ggml_add(ctx, ggml_mul_mat(ctx, l.q_w, h), l.q_b), hd, nh, n);
        ggml_tensor* K = heads(ctx, ggml_mul_mat(ctx, l.k_w, h), hd, nh, n);
        ggml_tensor* V = heads(ctx, ggml_add(ctx, ggml_mul_mat(ctx, l.v_w, h), l.v_b), hd, nh, n);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, K, cache_rows(ctx, c->ek, il, e0, n)));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, V, cache_rows(ctx, c->ev, il, e0, n)));
        ggml_tensor* a = ggml_flash_attn_ext(ctx, Q, cache_rows(ctx, c->ek, il, 0, e1),
                                             cache_rows(ctx, c->ev, il, 0, e1), mask, scale, 0.0f, 0.0f);
        a = ggml_reshape_2d(ctx, a, D, n);
        x = ggml_add(ctx, ggml_add(ctx, ggml_mul_mat(ctx, l.o_w, a), l.o_b), res);
        res = x;
        h = ln(ctx, x, l.ffn_ln_w, l.ffn_ln_b);
        h = ggml_gelu_erf(ctx, ggml_add(ctx, ggml_mul_mat(ctx, l.up_w, h), l.up_b));
        x = ggml_add(ctx, ggml_add(ctx, ggml_mul_mat(ctx, l.down_w, h), l.down_b), res);
    }
    x = ln(ctx, x, c->enc_ln_w, c->enc_ln_b);
    if (want_out) {
        ggml_set_name(x, "enc_out");
        ggml_set_output(x);
        ggml_build_forward_expand(gf, x);
    }
    const int nhd = hp.dec_heads, hdd = D / nhd;
    for (int il = 0; il < hp.dec_layers; il++) {
        const auto& l = c->dec[il];
        ggml_tensor* K = heads(ctx, ggml_mul_mat(ctx, l.xk_w, x), hdd, nhd, n);
        ggml_tensor* V = heads(ctx, ggml_add(ctx, ggml_mul_mat(ctx, l.xv_w, x), l.xv_b), hdd, nhd, n);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, K, cache_rows(ctx, c->xk, il, e0, n)));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, V, cache_rows(ctx, c->xv, il, e0, n)));
    }
    ggml_free(ctx);
    return gf;
}

static bool run_encoder(hikari_context* c, int e0, int e1, std::vector<float>* enc_out) {
    const int n = e1 - e0, Lm = 2 * n + 3;
    const int nfr = (int)(c->norm_mel.size() / 80);
    ggml_cgraph* gf = build_encoder(c, e0, e1, enc_out != nullptr);
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        fprintf(stderr, "hikari: encoder graph alloc failed\n");
        return false;
    }
    std::vector<float> mel((size_t)Lm * 80, 0.0f);
    for (int i = 0; i < Lm; i++) {
        const int m = 2 * e0 - 2 + i;
        if (m < 0 || m >= nfr || m >= kMelFrames)
            continue; // conv1 zero padding / the server's zero padding to 3000 frames
        for (int b = 0; b < 80; b++)
            mel[(size_t)b * Lm + i] = c->norm_mel[(size_t)m * 80 + b];
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mel"), mel.data(), 0, mel.size() * sizeof(float));
    std::vector<float> c1m(2 * n + 1);
    for (int i = 0; i < 2 * n + 1; i++) {
        const int cidx = 2 * e0 - 1 + i;
        c1m[i] = (cidx >= 0 && cidx < kMelFrames) ? 1.0f : 0.0f;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "c1mask"), c1m.data(), 0, c1m.size() * sizeof(float));
    std::vector<int32_t> pos(n);
    for (int i = 0; i < n; i++)
        pos[i] = e0 + i;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pos"), pos.data(), 0, pos.size() * sizeof(int32_t));
    std::vector<ggml_fp16_t> mask((size_t)e1 * n);
    const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    for (int q = 0; q < n; q++)
        for (int k = 0; k < e1; k++)
            mask[(size_t)q * e1 + k] = k <= e0 + q ? z : ninf;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mask"), mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "hikari: encoder compute failed\n");
        return false;
    }
    if (enc_out) {
        ggml_tensor* o = ggml_graph_get_tensor(gf, "enc_out");
        enc_out->resize((size_t)n * c->hp.d);
        ggml_backend_tensor_get(o, enc_out->data(), 0, enc_out->size() * sizeof(float));
    }
    return true;
}

// Decoder positions [p0, p1) of c->ids against encoder frames [0, n_enc).
static ggml_cgraph* build_decoder(hikari_context* c, int p0, int p1, int n_enc, bool all_logits) {
    const auto& hp = c->hp;
    const int D = hp.d, nh = hp.dec_heads, hd = D / nh, n = p1 - p0;
    ggml_init_params ip = {c->meta.size(), c->meta.data(), true};
    ggml_context* ctx = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx, 8192, false);

    ggml_tensor* tok = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    ggml_set_name(tok, "tok");
    ggml_set_input(tok);
    ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    ggml_set_name(pos, "pos");
    ggml_set_input(pos);
    ggml_tensor* smask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, p1, n);
    ggml_set_name(smask, "smask");
    ggml_set_input(smask);
    ggml_tensor* xmask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_enc, n);
    ggml_set_name(xmask, "xmask");
    ggml_set_input(xmask);
    ggml_tensor* rowmask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n);
    ggml_set_name(rowmask, "rowmask");
    ggml_set_input(rowmask);

    ggml_tensor* x = ggml_add(ctx, ggml_get_rows(ctx, c->tok_emb, tok), ggml_get_rows(ctx, c->dec_pos, pos));
    const float scale = 1.0f / std::sqrt((float)hd);
    for (int il = 0; il < hp.dec_layers; il++) {
        const auto& l = c->dec[il];
        ggml_tensor* res = x;
        ggml_tensor* h = ln(ctx, x, l.attn_ln_w, l.attn_ln_b);
        ggml_tensor* Q = heads(ctx, ggml_add(ctx, ggml_mul_mat(ctx, l.q_w, h), l.q_b), hd, nh, n);
        ggml_tensor* K = heads(ctx, ggml_mul_mat(ctx, l.k_w, h), hd, nh, n);
        ggml_tensor* V = heads(ctx, ggml_add(ctx, ggml_mul_mat(ctx, l.v_w, h), l.v_b), hd, nh, n);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, K, cache_rows(ctx, c->dk, il, p0, n)));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, V, cache_rows(ctx, c->dv, il, p0, n)));
        ggml_tensor* a = ggml_flash_attn_ext(ctx, Q, cache_rows(ctx, c->dk, il, 0, p1),
                                             cache_rows(ctx, c->dv, il, 0, p1), smask, scale, 0.0f, 0.0f);
        x = ggml_add(ctx, ggml_add(ctx, ggml_mul_mat(ctx, l.o_w, ggml_reshape_2d(ctx, a, D, n)), l.o_b), res);

        res = x;
        h = ln(ctx, x, l.x_ln_w, l.x_ln_b);
        Q = heads(ctx, ggml_add(ctx, ggml_mul_mat(ctx, l.xq_w, h), l.xq_b), hd, nh, n);
        a = ggml_flash_attn_ext(ctx, Q, cache_rows(ctx, c->xk, il, 0, n_enc), cache_rows(ctx, c->xv, il, 0, n_enc),
                                xmask, scale, 0.0f, 0.0f);
        // A position that may see no frame at all (position 0) gets a zero
        // attention output, as torch SDPA's safe softmax gives it.
        a = ggml_mul(ctx, ggml_reshape_2d(ctx, a, D, n), rowmask);
        x = ggml_add(ctx, ggml_add(ctx, ggml_mul_mat(ctx, l.xo_w, a), l.xo_b), res);

        res = x;
        h = ln(ctx, x, l.ffn_ln_w, l.ffn_ln_b);
        h = ggml_gelu_erf(ctx, ggml_add(ctx, ggml_mul_mat(ctx, l.up_w, h), l.up_b));
        x = ggml_add(ctx, ggml_add(ctx, ggml_mul_mat(ctx, l.down_w, h), l.down_b), res);
    }
    x = ln(ctx, x, c->dec_ln_w, c->dec_ln_b);
    if (!all_logits && n > 1)
        x = ggml_view_2d(ctx, x, D, 1, x->nb[1], (size_t)(n - 1) * x->nb[1]);
    ggml_tensor* logits = ggml_mul_mat(ctx, c->tok_emb, x);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);
    ggml_free(ctx);
    return gf;
}

static bool run_decoder(hikari_context* c, int p0, int p1, int n_enc, bool all_logits, std::vector<float>& logits) {
    const int n = p1 - p0, dil = c->hp.dilation;
    ggml_cgraph* gf = build_decoder(c, p0, p1, n_enc, all_logits);
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        fprintf(stderr, "hikari: decoder graph alloc failed\n");
        return false;
    }
    std::vector<int32_t> pos(n);
    for (int i = 0; i < n; i++)
        pos[i] = p0 + i;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "tok"), c->ids.data() + p0, 0, n * sizeof(int32_t));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pos"), pos.data(), 0, n * sizeof(int32_t));
    const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    std::vector<ggml_fp16_t> sm((size_t)p1 * n), xm((size_t)n_enc * n);
    std::vector<float> rm(n);
    for (int q = 0; q < n; q++) {
        const int i = p0 + q;
        for (int k = 0; k < p1; k++)
            sm[(size_t)q * p1 + k] = k <= i ? z : ninf;
        const int vis = std::min(n_enc, dil * i); // frames j < 4*i
        for (int k = 0; k < n_enc; k++)
            xm[(size_t)q * n_enc + k] = (k < vis || (vis == 0 && k == 0)) ? z : ninf;
        rm[q] = vis > 0 ? 1.0f : 0.0f;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "smask"), sm.data(), 0, sm.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "xmask"), xm.data(), 0, xm.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "rowmask"), rm.data(), 0, rm.size() * sizeof(float));
    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "hikari: decoder compute failed\n");
        return false;
    }
    ggml_tensor* lt = ggml_graph_get_tensor(gf, "logits");
    logits.resize((size_t)ggml_nelements(lt));
    ggml_backend_tensor_get(lt, logits.data(), 0, logits.size() * sizeof(float));
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Policy step

static int window_tokens(const hikari_context* c) {
    return std::min(std::max(c->pol.decoder_context, 50), c->hp.n_text_ctx);
}

// One decision on audio [0, t). Returns the chosen token, -1 on error.
static int step(hikari_context* c, int64_t t) {
    const auto& hp = c->hp;
    const int W = window_tokens(c);
    c->j++;
    const int64_t ws = std::max<int64_t>(0, t - (int64_t)W * kChunk);

    // mel + first changed frame
    double t0 = now_ms();
    std::vector<float> prev;
    prev.swap(c->norm_mel);
    const bool slid = ws != c->win_start;
    window_mel(c, ws, t, c->norm_mel);
    const int nfr = (int)(c->norm_mel.size() / 80);
    const int n_enc = nfr / 2;
    int m0 = 0;
    if (!slid && !prev.empty()) {
        const int nprev = (int)(prev.size() / 80);
        const int upto = std::max(nprev, nfr);
        m0 = upto;
        for (int m = 0; m < upto && m0 == upto; m++)
            for (int b = 0; b < 80; b++) {
                const float a = m < nprev ? prev[(size_t)m * 80 + b] : 0.0f;
                const float bb = m < nfr ? c->norm_mel[(size_t)m * 80 + b] : 0.0f;
                if (a != bb) {
                    m0 = m;
                    break;
                }
            }
    }
    c->win_start = ws;
    if (slid)
        c->enc_valid = 0;
    const int e_dirty = std::max(0, (m0 - 1) / 2); // ceil((m0-2)/2)
    const int e0 = std::min(c->enc_valid, e_dirty);
    c->bench.mel += now_ms() - t0;

    t0 = now_ms();
    if (e0 < n_enc && !run_encoder(c, e0, n_enc, nullptr))
        return -1;
    c->enc_valid = n_enc;
    c->bench.enc += now_ms() - t0;
    c->bench.enc_frames += n_enc - e0;

    // speech: feeds the wait-penalty boost — without it the model mostly
    // WAITs (measured: jfk 0-4 s, upstream and here, 48/48 WAIT with VAD off)
    bool speech = false;
    float sp = 0.0f;
    if (c->speech_fn) {
        t0 = now_ms();
        const int nv = (int)std::min<int64_t>(kVadSamples, t - ws);
        sp = c->speech_fn(c->audio.data() + t - nv, nv, c->speech_user);
        speech = sp > c->pol.speech_threshold;
        c->bench.vad += now_ms() - t0;
    }

    // decoder
    t0 = now_ms();
    bool popped = false;
    if ((int)c->ids.size() > W) {
        c->ids.erase(c->ids.begin() + 4);
        popped = true;
    }
    const int pos = (int)c->ids.size() - 1;
    int p0 = std::min(c->dec_valid, pos);
    p0 = std::min(p0, e0 / hp.dilation + 1); // positions whose visible frames changed
    if (popped)
        p0 = std::min(p0, 4);
    std::vector<float> logits;
    if (!run_decoder(c, p0, pos + 1, n_enc, false, logits))
        return -1;
    c->dec_valid = pos + 1;
    c->last_ids = c->ids;
    c->bench.dec += now_ms() - t0;
    c->bench.dec_positions += pos + 1 - p0;
    c->bench.steps++;

    const int V = hp.vocab;
    int mx = (int)(std::max_element(logits.begin(), logits.begin() + V) - logits.begin());
    if (mx != hp.tok_wait) {
        const size_t from = c->ids.size() >= 5 ? c->ids.size() - 5 : 0;
        if (std::find(c->ids.begin() + from, c->ids.end(), mx) != c->ids.end())
            logits[mx] -= c->pol.repetition_penalty;
    }
    logits[hp.tok_wait] -= c->wp;
    const int tok = (int)(std::max_element(logits.begin(), logits.begin() + V) - logits.begin());
    c->ids.push_back(tok);
    if (tok == hp.tok_wait) {
        if (speech && c->ids.size() > 10 &&
            std::all_of(c->ids.end() - 10, c->ids.end(), [&](int32_t v) { return v == hp.tok_wait; }))
            c->wp += c->pol.wait_penalty_boost;
    } else {
        c->wp -= c->pol.wait_penalty_decay * (c->wp - c->pol.baseline_wait_penalty);
    }
    c->step_tok.push_back(tok);
    c->step_t.push_back((double)t / 16000.0);
    c->step_sp.push_back(sp);
    if (c->params.verbosity >= 2)
        fprintf(stderr, "hikari: step %4d t=%6.2fs e0=%4d/%4d p0=%3d/%3d tok=%5d wp=%.2f%s\n", c->j, t / 16000.0, e0,
                n_enc, p0, pos, tok, c->wp, speech ? " speech" : "");
    return tok;
}

static int decide_ready(hikari_context* c) {
    int emitted = 0;
    const int first = 3 * kChunk; // get_one_token returns early below 160*2*dilation*3 samples
    while (c->n_decided + kChunk <= (int64_t)c->audio.size()) {
        c->n_decided += kChunk;
        if (c->n_decided < first)
            continue;
        const int tok = step(c, c->n_decided);
        if (tok < 0)
            return -1;
        if (tok != c->hp.tok_wait)
            emitted++;
    }
    return emitted;
}

static std::string decode_ids(const hikari_context* c, const std::vector<int32_t>& ids) {
    std::vector<int32_t> keep;
    for (int32_t id : ids)
        if (id != c->hp.tok_wait && id >= 0 && id < c->hp.tok_eot)
            keep.push_back(id);
    return core_bpe::detokenize(c->id_to_token, keep.data(), keep.size());
}

static char* dup(const std::string& s) {
    char* r = (char*)std::malloc(s.size() + 1);
    if (r)
        std::memcpy(r, s.c_str(), s.size() + 1);
    return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API

extern "C" hikari_context_params hikari_context_default_params(void) {
    hikari_context_params p{};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = true;
    return p;
}

extern "C" hikari_policy hikari_default_policy(void) {
    hikari_policy p{};
    p.baseline_wait_penalty = 0.0f;
    p.wait_penalty_boost = 0.6f;
    p.wait_penalty_decay = 0.3f;
    p.repetition_penalty = 40.0f;
    p.decoder_context = 337;
    p.speech_threshold = 0.8f;
    return p;
}

extern "C" hikari_context* hikari_init_from_file(const char* path, hikari_context_params params) {
    auto* c = new hikari_context();
    c->params = params;
    c->pol = hikari_default_policy();
    {
        gguf_context* g = core_gguf::open_metadata(path);
        if (!g) {
            delete c;
            return nullptr;
        }
        const bool ok = load_meta(c, g);
        core_gguf::free_metadata(g);
        if (!ok) {
            delete c;
            return nullptr;
        }
    }
    c->backend_cpu = core_cpu_backend::init();
    if (params.n_threads > 0)
        core_cpu_backend::set_n_threads(c->backend_cpu, params.n_threads);
    c->backend = c->backend_cpu;
    const char* ge = std::getenv("HIKARI_GPU");
    const bool force_cpu = ge && std::atoi(ge) == 0;
    if (!force_cpu && params.use_gpu) {
        if (ggml_backend_t gpu = crispasr_init_gpu_backend()) {
            c->backend = gpu;
            if (params.verbosity >= 1)
                fprintf(stderr, "hikari: GPU backend %s\n", ggml_backend_name(gpu));
        }
    }
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, c->backend, "hikari", wl)) {
        hikari_free(c);
        return nullptr;
    }
    c->ctx_w = wl.ctx;
    c->buf_w = wl.buf;
    c->tensors = std::move(wl.tensors);
    if (!bind(c) || !alloc_caches(c)) {
        hikari_free(c);
        return nullptr;
    }
    ggml_backend_t bes[] = {c->backend, c->backend_cpu};
    c->sched = ggml_backend_sched_new(bes, nullptr, c->backend != c->backend_cpu ? 2 : 1, 8192, false, false);
    c->meta.resize(ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false));
    hikari_set_task(c, 0, "en");
    if (params.verbosity >= 1)
        fprintf(stderr, "hikari: d=%d enc=%dL dec=%dL heads=%d vocab=%d text_ctx=%d dilation=%d\n", c->hp.d,
                c->hp.enc_layers, c->hp.dec_layers, c->hp.enc_heads, c->hp.vocab, c->hp.n_text_ctx, c->hp.dilation);
    return c;
}

extern "C" void hikari_free(hikari_context* c) {
    if (!c)
        return;
    if (c->sched)
        ggml_backend_sched_free(c->sched);
    if (c->cache_buf)
        ggml_backend_buffer_free(c->cache_buf);
    if (c->cache_ctx)
        ggml_free(c->cache_ctx);
    if (c->buf_w)
        core_gguf::release_weight_buffer(c->buf_w);
    if (c->ctx_w)
        ggml_free(c->ctx_w);
    if (c->backend && c->backend != c->backend_cpu)
        ggml_backend_free(c->backend);
    if (c->backend_cpu)
        ggml_backend_free(c->backend_cpu);
    delete c;
}

extern "C" int hikari_set_task(hikari_context* c, int translate, const char* tgt_lang) {
    if (!c)
        return -1;
    const auto& hp = c->hp;
    if (!translate) {
        c->prompt = {hp.tok_sot, c->lang_to_id["en"], hp.tok_transcribe, hp.tok_notimestamps};
    } else {
        // model_wrapper.py accepts Japanese/German/Russian/English/Chinese.
        const std::string l = tgt_lang ? tgt_lang : "";
        if (l != "de" && l != "ja" && l != "ru" && l != "en" && l != "zh")
            return -1;
        c->prompt = {hp.tok_sot, c->lang_to_id[l], hp.tok_translate, hp.tok_notimestamps};
    }
    hikari_stream_reset(c);
    return 0;
}

extern "C" void hikari_set_policy(hikari_context* c, const hikari_policy* p) {
    if (c && p) {
        c->pol = *p;
        hikari_stream_reset(c);
    }
}

extern "C" void hikari_set_speech_prob_fn(hikari_context* c, hikari_speech_prob_fn fn, void* user) {
    if (c) {
        c->speech_fn = fn;
        c->speech_user = user;
    }
}

extern "C" void hikari_set_tail_silence_ms(hikari_context* c, int ms) {
    if (c)
        c->tail_ms = std::max(0, ms);
}

extern "C" void hikari_stream_reset(hikari_context* c) {
    if (!c)
        return;
    c->audio.clear();
    c->n_decided = 0;
    c->j = -1;
    c->ids = c->prompt;
    c->wp = c->pol.baseline_wait_penalty;
    c->step_tok.clear();
    c->step_t.clear();
    c->step_sp.clear();
    c->raw_cache.clear();
    c->win_start = -1;
    c->norm_mel.clear();
    c->enc_valid = 0;
    c->dec_valid = 0;
    c->last_ids.clear();
}

extern "C" int hikari_stream_push(hikari_context* c, const float* pcm, int n) {
    if (!c || (n > 0 && !pcm))
        return -1;
    c->audio.insert(c->audio.end(), pcm, pcm + n);
    return decide_ready(c);
}

extern "C" int hikari_stream_flush(hikari_context* c) {
    if (!c)
        return -1;
    const int64_t rem = (int64_t)c->audio.size() - c->n_decided;
    if (rem > 0)
        c->audio.resize(c->audio.size() + (kChunk - rem), 0.0f);
    return decide_ready(c);
}

extern "C" int hikari_stream_n_steps(hikari_context* c) {
    return c ? (int)c->step_tok.size() : 0;
}
extern "C" int32_t hikari_stream_step_token(hikari_context* c, int i) {
    return (c && i >= 0 && i < (int)c->step_tok.size()) ? c->step_tok[i] : -1;
}
extern "C" double hikari_stream_step_time(hikari_context* c, int i) {
    return (c && i >= 0 && i < (int)c->step_t.size()) ? c->step_t[i] : -1.0;
}
extern "C" float hikari_stream_step_speech_prob(hikari_context* c, int i) {
    return (c && i >= 0 && i < (int)c->step_sp.size()) ? c->step_sp[i] : -1.0f;
}
extern "C" char* hikari_stream_text(hikari_context* c) {
    return c ? dup(decode_ids(c, c->step_tok)) : nullptr;
}
extern "C" char* hikari_token_text(hikari_context* c, int32_t id) {
    return c ? dup(decode_ids(c, {id})) : nullptr;
}

extern "C" char* hikari_transcribe(hikari_context* c, const float* pcm, int n) {
    if (!c)
        return nullptr;
    hikari_stream_reset(c);
    c->bench = bench_acc{};
    if (hikari_stream_push(c, pcm, n) < 0)
        return nullptr;
    if (c->tail_ms > 0) {
        std::vector<float> z((size_t)c->tail_ms * 16, 0.0f);
        if (hikari_stream_push(c, z.data(), (int)z.size()) < 0)
            return nullptr;
    }
    if (hikari_stream_flush(c) < 0)
        return nullptr;
    if (bench_enabled())
        hikari_print_bench(c);
    return hikari_stream_text(c);
}

extern "C" void hikari_print_bench(hikari_context* c) {
    if (!c)
        return;
    const auto& b = c->bench;
    const double audio_s = c->step_t.empty() ? 0.0 : c->step_t.back();
    const double tot = b.mel + b.enc + b.dec + b.vad;
    fprintf(stderr,
            "hikari_bench: steps=%ld audio=%.2fs mel=%.0fms enc=%.0fms (%ld frames) dec=%.0fms (%ld positions) "
            "vad=%.0fms total=%.0fms = %.1f ms per audio-second\n",
            b.steps, audio_s, b.mel, b.enc, b.enc_frames, b.dec, b.dec_positions, b.vad, tot,
            audio_s > 0 ? tot / audio_s : 0.0);
}

extern "C" float* hikari_debug_last_mel(hikari_context* c, int* n_mels, int* n_frames) {
    if (!c || c->norm_mel.empty())
        return nullptr;
    const int nfr = (int)(c->norm_mel.size() / 80);
    float* out = (float*)std::calloc((size_t)80 * kMelFrames, sizeof(float));
    for (int m = 0; m < std::min(nfr, kMelFrames); m++)
        for (int b = 0; b < 80; b++)
            out[(size_t)b * kMelFrames + m] = c->norm_mel[(size_t)m * 80 + b];
    if (n_mels)
        *n_mels = 80;
    if (n_frames)
        *n_frames = kMelFrames;
    return out;
}

extern "C" float* hikari_debug_last_logits(hikari_context* c, int* n_pos, int* n_vocab, int32_t** out_ids) {
    if (!c || c->last_ids.empty())
        return nullptr;
    std::vector<int32_t> saved = c->ids;
    c->ids = c->last_ids;
    std::vector<float> logits;
    const bool ok = run_decoder(c, 0, (int)c->ids.size(), c->enc_valid, true, logits);
    c->ids.swap(saved);
    if (!ok)
        return nullptr;
    if (n_pos)
        *n_pos = (int)c->last_ids.size();
    if (n_vocab)
        *n_vocab = c->hp.vocab;
    if (out_ids) {
        *out_ids = (int32_t*)std::malloc(c->last_ids.size() * sizeof(int32_t));
        std::memcpy(*out_ids, c->last_ids.data(), c->last_ids.size() * sizeof(int32_t));
    }
    float* r = (float*)std::malloc(logits.size() * sizeof(float));
    std::memcpy(r, logits.data(), logits.size() * sizeof(float));
    return r;
}

extern "C" float* hikari_debug_encoder_full(hikari_context* c, int* n_frames, int* d_model) {
    if (!c || c->norm_mel.empty())
        return nullptr;
    std::vector<float> out;
    if (!run_encoder(c, 0, c->hp.n_audio_ctx, &out))
        return nullptr;
    c->enc_valid = 0; // caches now hold the padded-window encoding
    c->dec_valid = 0;
    if (n_frames)
        *n_frames = c->hp.n_audio_ctx;
    if (d_model)
        *d_model = c->hp.d;
    float* r = (float*)std::malloc(out.size() * sizeof(float));
    std::memcpy(r, out.data(), out.size() * sizeof(float));
    return r;
}
