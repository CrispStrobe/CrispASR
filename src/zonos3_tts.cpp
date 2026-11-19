// zonos3_tts.cpp — CrispASR wrapper around the vendored ZONOS3-Mini runtime (src/zonos3/, MIT, Zyphra).
// The generation recipe follows upstream servers/ggml/src/main.cpp --tts + emit_output exactly:
// zonos3_build_prompt -> zonos3_generate (KV path) -> prefill codes + generated rows -> zac_decode ->
// drop prefill audio -> zac_output_gain -> zonos3_soft_limit.
#include "zonos3_tts.h"

#include "zonos3/zonos3.h"
#include "zonos3/zonos3-generation.h"
#include "zonos3/zac.h"
#include "zonos3/npy.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

struct zonos3_tts_context {
    zonos3_tts_context_params params{};
    zonos3_model model;
    zac_model codec;
    bool codec_loaded = false;
    std::string codec_path;
    std::vector<float> speaker;   // empty = no speaker row
    std::vector<int32_t> buckets; // [n_cond]; count = unknown
    float temperature = -1.0f;
    uint32_t seed = 0;
};

static void set_threads(ggml_backend_t backend, int n) {
    if (backend && n > 0 && ggml_backend_is_cpu(backend))
        ggml_backend_cpu_set_n_threads(backend, n);
}

static std::string find_codec(const std::string& model_path, const char* explicit_path) {
    if (explicit_path && *explicit_path)
        return explicit_path;
    if (const char* env = getenv("CRISPASR_ZONOS3_CODEC"))
        if (*env)
            return env;
    namespace fs = std::filesystem;
    const fs::path dir = fs::path(model_path).parent_path();
    for (const char* name : {"zac-keep2.gguf", "zac.gguf"}) {
        std::error_code ec;
        if (fs::exists(dir / name, ec))
            return (dir / name).string();
    }
    return {};
}

extern "C" zonos3_tts_context_params zonos3_tts_context_default_params(void) {
    zonos3_tts_context_params p{};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = false;
    p.temperature = -1.0f;
    p.seed = 0;
    p.max_frames = zonos3_generation().max_frames; // 2000, the servers' default budget
    p.cfg_scale = 1.0f;
    p.codec_path = nullptr;
    return p;
}

extern "C" zonos3_tts_context* zonos3_tts_init_from_file(const char* path, zonos3_tts_context_params params) {
    if (!path)
        return nullptr;
    auto* ctx = new zonos3_tts_context();
    ctx->params = params;
    ctx->temperature = params.temperature;
    ctx->seed = params.seed;
    if (!zonos3_model_load(ctx->model, path, params.use_gpu)) {
        delete ctx;
        return nullptr;
    }
    set_threads(ctx->model.backend, params.n_threads);
    ctx->buckets.assign(ctx->model.hp.cond_bucket_counts.begin(), ctx->model.hp.cond_bucket_counts.end());
    ctx->codec_path = find_codec(path, params.codec_path);
    if (ctx->codec_path.empty() && params.verbosity > 0)
        fprintf(stderr, "zonos3: no codec (zac-keep2.gguf) next to the backbone; set CRISPASR_ZONOS3_CODEC\n");
    return ctx;
}

extern "C" void zonos3_tts_free(zonos3_tts_context* ctx) {
    if (!ctx)
        return;
    if (ctx->codec_loaded)
        zac_free(ctx->codec);
    zonos3_model_free(ctx->model);
    delete ctx;
}

extern "C" int zonos3_tts_set_speaker_embedding(zonos3_tts_context* ctx, const float* emb, int n) {
    if (!ctx)
        return -1;
    if (!emb || n == 0) {
        ctx->speaker.clear();
        return 0;
    }
    if (n != (int)ctx->model.hp.speaker_dim) {
        fprintf(stderr, "zonos3: speaker embedding needs %u values, got %d\n", ctx->model.hp.speaker_dim, n);
        return -1;
    }
    for (int i = 0; i < n; ++i)
        if (!std::isfinite(emb[i]))
            return -1;
    ctx->speaker.assign(emb, emb + n);
    return 0;
}

extern "C" int zonos3_tts_load_speaker_embedding(zonos3_tts_context* ctx, const char* npy_path) {
    if (!ctx || !npy_path)
        return -1;
    std::vector<float> data;
    std::vector<int64_t> shape;
    if (!npy::load_f32(npy_path, data, shape)) {
        fprintf(stderr, "zonos3: cannot read speaker embedding '%s'\n", npy_path);
        return -1;
    }
    return zonos3_tts_set_speaker_embedding(ctx, data.data(), (int)data.size());
}

extern "C" int zonos3_tts_set_condition(zonos3_tts_context* ctx, const char* name, double value) {
    if (!ctx || !name)
        return -1;
    const int idx = zonos3_condition_index(ctx->model, name);
    if (idx < 0)
        return -1;
    const int b = zonos3_condition_bucket(ctx->model, idx, value);
    ctx->buckets[idx] = b < 0 ? (int32_t)ctx->model.hp.cond_bucket_counts[idx] : b; // non-finite = unknown
    return 0;
}

extern "C" void zonos3_tts_set_temperature(zonos3_tts_context* ctx, float t) {
    if (ctx)
        ctx->temperature = t;
}
extern "C" void zonos3_tts_set_seed(zonos3_tts_context* ctx, uint32_t seed) {
    if (ctx)
        ctx->seed = seed;
}
extern "C" int zonos3_tts_get_sample_rate(const zonos3_tts_context* ctx) {
    return ctx && ctx->codec_loaded ? ctx->codec.sample_rate : 44100;
}

static bool build_prompt(zonos3_tts_context* ctx, const char* text, zonos3_prompt& prompt) {
    zonos3_prompt_options opt;
    if (!ctx->speaker.empty())
        opt.speaker_embedding = ctx->speaker.data();
    opt.condition_buckets = ctx->buckets;
    if (const char* s = getenv("CRISPASR_ZONOS3_SILENCE_PREFILL"))
        opt.silence_prefill_frames = atoi(s);
    return zonos3_build_prompt(ctx->model, text, opt, prompt);
}

static zonos3_sampling make_sampling(const zonos3_tts_context* ctx) {
    zonos3_sampling sp;
    sp.seed = ctx->seed;
    if (ctx->temperature == 0.0f)
        sp.greedy = true;
    else if (ctx->temperature > 0.0f)
        sp.temperature = ctx->temperature;
    return sp;
}

extern "C" int32_t* zonos3_tts_generate_codes(zonos3_tts_context* ctx, const char* text, int* out_frames, int* out_eos,
                                              int* out_prefill) {
    if (!ctx || !text || !out_frames)
        return nullptr;
    zonos3_prompt prompt;
    if (!build_prompt(ctx, text, prompt))
        return nullptr;
    const int max_frames = ctx->params.max_frames > 0 ? ctx->params.max_frames : zonos3_generation().max_frames;
    if ((int64_t)prompt.n_rows + max_frames > ctx->model.hp.n_ctx_train)
        return nullptr;
    float cfg = ctx->params.cfg_scale;
    if (cfg != 1.0f && prompt.speaker_rows <= 0)
        cfg = 1.0f; // CFG twin needs a speaker row
    std::vector<int32_t> generated;
    int eos = -1;
    const int n = zonos3_generate(ctx->model, prompt, max_frames, make_sampling(ctx), cfg, generated, eos, true);
    if (n < 0)
        return nullptr;
    std::vector<int32_t> codes = zonos3_prefill_codes(ctx->model, prompt);
    codes.insert(codes.end(), generated.begin(), generated.end());
    const int ncb = (int)ctx->model.hp.n_codebooks;
    auto* out = (int32_t*)malloc(codes.size() * sizeof(int32_t));
    if (!out)
        return nullptr;
    memcpy(out, codes.data(), codes.size() * sizeof(int32_t));
    *out_frames = (int)(codes.size() / ncb);
    if (out_eos)
        *out_eos = eos;
    if (out_prefill)
        *out_prefill = prompt.prefill_frames;
    if (ctx->params.verbosity > 0)
        fprintf(stderr, "zonos3: %d frames generated (eos_frame=%d, prompt %d rows)\n", n, eos, prompt.n_rows);
    return out;
}

extern "C" float* zonos3_tts_decode_codes(zonos3_tts_context* ctx, const int32_t* codes, int frames, int eos,
                                          int prefill, int* out_n) {
    if (!ctx || !codes || frames <= 0 || !out_n)
        return nullptr;
    if (!ctx->codec_loaded) {
        if (ctx->codec_path.empty() || !zac_load(ctx->codec, ctx->codec_path.c_str(), ctx->params.use_gpu)) {
            fprintf(stderr, "zonos3: cannot load codec '%s'\n", ctx->codec_path.c_str());
            return nullptr;
        }
        set_threads(ctx->codec.backend, ctx->params.n_threads);
        ctx->codec_loaded = true;
    }
    std::vector<float> audio;
    if (!zac_decode(ctx->codec, codes, frames, (int)ctx->model.hp.n_codebooks, eos, audio))
        return nullptr;
    const size_t drop = std::min(audio.size(), (size_t)prefill * zac_samples_per_frame(ctx->codec));
    audio.erase(audio.begin(), audio.begin() + drop);
    const float gain = zac_output_gain(ctx->codec, -1.0);
    for (float& v : audio)
        v *= gain;
    zonos3_soft_limit(audio.data(), audio.size());
    auto* out = (float*)malloc(std::max<size_t>(1, audio.size()) * sizeof(float));
    if (!out)
        return nullptr;
    if (!audio.empty())
        memcpy(out, audio.data(), audio.size() * sizeof(float));
    *out_n = (int)audio.size();
    return out;
}

extern "C" float* zonos3_tts_synthesize(zonos3_tts_context* ctx, const char* text, int* out_n) {
    // Diff harness hook (tools/zonos3_diff.py): prefill-graph stages + the generated code matrix.
    const char* dump = getenv("CRISPASR_ZONOS3_DUMP_DIR");
    if (dump && *dump && zonos3_tts_dump_stages(ctx, text, dump) != 0)
        fprintf(stderr, "zonos3: stage dump to '%s' failed\n", dump);
    int frames = 0, eos = -1, prefill = 0;
    // Codec-only diff (tools/zonos3_diff.py): decode a given [frames, 10] code matrix (+ <path>.eos.npy =
    // [eos, prefill], as upstream zonos3-cli --tts writes) instead of generating.
    if (const char* given = getenv("CRISPASR_ZONOS3_DECODE_CODES")) {
        std::vector<float> cf, ef;
        std::vector<int64_t> cs, es;
        if (!npy::load_f32(given, cf, cs) || cs.size() != 2 || cs[1] != (int64_t)ctx->model.hp.n_codebooks ||
            !npy::load_f32(std::string(given) + ".eos.npy", ef, es) || ef.size() != 2) {
            fprintf(stderr, "zonos3: cannot read codes '%s' (+ .eos.npy)\n", given);
            return nullptr;
        }
        std::vector<int32_t> ci(cf.begin(), cf.end());
        return zonos3_tts_decode_codes(ctx, ci.data(), (int)cs[0], (int)ef[0], (int)ef[1], out_n);
    }
    int32_t* codes = zonos3_tts_generate_codes(ctx, text, &frames, &eos, &prefill);
    if (!codes)
        return nullptr;
    if (dump && *dump) {
        const int ncb = (int)ctx->model.hp.n_codebooks;
        std::vector<float> cf(codes, codes + (size_t)frames * ncb);
        const float ef[2] = {(float)eos, (float)prefill};
        npy::save_f32(std::string(dump) + "/codes.npy", cf.data(), {(int64_t)frames, (int64_t)ncb});
        npy::save_f32(std::string(dump) + "/codes.npy.eos.npy", ef, {(int64_t)2});
    }
    float* pcm = frames > prefill ? zonos3_tts_decode_codes(ctx, codes, frames, eos, prefill, out_n) : nullptr;
    free(codes);
    return pcm;
}

extern "C" void zonos3_tts_free_audio(float* pcm) {
    free(pcm);
}
extern "C" void zonos3_tts_free_codes(int32_t* codes) {
    free(codes);
}

extern "C" int zonos3_tts_dump_stages(zonos3_tts_context* ctx, const char* text, const char* out_dir) {
    if (!ctx || !text || !out_dir)
        return -1;
    zonos3_prompt prompt;
    if (!build_prompt(ctx, text, prompt))
        return -1;
    std::vector<float> ids(prompt.ids.begin(), prompt.ids.end());
    const std::string dir = out_dir;
    const int W = (int)ctx->model.hp.n_codebooks + 1;
    if (!npy::save_f32(dir + "/input_ids.npy", ids.data(), {(int64_t)prompt.n_rows, (int64_t)W})) {
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        npy::save_f32(dir + "/input_ids.npy", ids.data(), {(int64_t)prompt.n_rows, (int64_t)W});
    }
    return zonos3_validate(ctx->model, ids.data(), prompt.n_rows, prompt.cond, out_dir, -1) ? 0 : -1;
}
