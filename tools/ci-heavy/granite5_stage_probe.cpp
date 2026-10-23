// Actual public pipeline stages and both decoder paths, using original inputs.
#include "granite_speech.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "core/granite_prompt.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char* what) {
    if (!ok)
        throw std::runtime_error(what);
}

int main(int argc, char** argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s model ref out instruction cpu|cuda\n", argv[0]);
        return 2;
    }
    try {
        const auto out = std::filesystem::path(argv[3]);
        std::filesystem::create_directories(out);
        auto save = [&](const char* name, const void* data, size_t bytes) {
            auto* file = std::fopen((out / name).string().c_str(), "wb");
            require(file != nullptr, "output open");
            const auto n = std::fwrite(data, 1, bytes, file);
            std::fclose(file);
            require(n == bytes, "output write");
        };
        ggml_context* refs = nullptr;
        gguf_init_params rp = {false, &refs};
        auto* archive = gguf_init_from_file(argv[2], rp);
        require(archive && refs, "original reference archive");
        auto tensor = [&](const char* name, ggml_type type) {
            auto* t = ggml_get_tensor(refs, name);
            require(t && t->type == type && t->data, name);
            return t;
        };
        auto* audio = tensor("raw_audio", GGML_TYPE_F32);
        auto* original_embeds = tensor("llm_inputs_embeds", GGML_TYPE_F32);
        auto cp = granite_speech_context_default_params();
        cp.use_gpu = std::string(argv[5]) == "cuda";
        // In a DL build the registry starts empty. Load the packaged modules
        // before init_best(); the later CPU helper otherwise discovers CUDA
        // only after the model has already selected its CPU fallback.
        if (cp.use_gpu) {
            ggml_backend_load_all();
            require(ggml_backend_dev_by_name("CUDA0") != nullptr, "CUDA0 registry device");
        }
        cp.n_threads = 4;
        cp.verbosity = 2;
        auto* ctx = granite_speech_init_from_file(argv[1], cp);
        require(ctx != nullptr, "native model");
        int nm = 0, tm = 0, ne = 0, de = 0, np = 0, dp = 0;
        auto* mel =
            granite_speech_compute_mel(ctx, static_cast<float*>(audio->data), (int)ggml_nelements(audio), &nm, &tm);
        require(mel && nm > 0 && tm > 0, "mel");
        save("mel_spectrogram.bin", mel, size_t(nm) * tm * sizeof(float));
        auto* enc = granite_speech_run_encoder(ctx, mel, nm, tm, &ne, &de);
        std::free(mel);
        require(enc && ne > 0 && de > 0, "encoder");
        save("encoder_out.bin", enc, size_t(ne) * de * sizeof(float));
        auto* proj = granite_speech_run_projector(ctx, enc, ne, de, &np, &dp);
        std::free(enc);
        require(proj && np > 0 && dp > 0, "projector");
        save("projector_out.bin", proj, size_t(np) * dp * sizeof(float));
        int n = 0;
        auto* prefix = granite_speech_tokenize(ctx, "USER: ", &n);
        require(prefix && n > 0, "prefix tokenizer");
        std::vector<int32_t> ids(prefix, prefix + n);
        std::free(prefix);
        const int audio_start = (int)ids.size();
        ids.insert(ids.end(), np, granite_speech_audio_token_id(ctx));
        const auto suffix_text = core_granite_prompt::legacy_suffix(argv[4]);
        auto* suffix = granite_speech_tokenize(ctx, suffix_text.c_str(), &n);
        require(suffix && n > 0, "suffix tokenizer");
        ids.insert(ids.end(), suffix, suffix + n);
        std::free(suffix);
        save("prompt_ids.bin", ids.data(), ids.size() * sizeof(int32_t));
        auto* embeds = granite_speech_embed_tokens(ctx, ids.data(), (int)ids.size());
        require(embeds != nullptr, "token embedding");
        std::memcpy(embeds + size_t(audio_start) * dp, proj, size_t(np) * dp * sizeof(float));
        std::free(proj);
        save("llm_inputs_embeds.bin", embeds, ids.size() * dp * sizeof(float));
        require(original_embeds->ne[0] == dp && original_embeds->ne[1] == (int64_t)ids.size(), "prefill geometry");
        const int total = (int)ids.size(), max_new = 96;
        require(granite_speech_kv_init(ctx, total + max_new), "KV allocation");
        int nt = 0, nv = 0;
        auto* logits = granite_speech_run_llm_kv(ctx, embeds, total, 0, &nt, &nv);
        std::free(embeds);
        require(logits && nt == 1 && nv > 0, "native pipeline prefill");
        save("pipeline_prefill_logits.bin", logits, size_t(nv) * sizeof(float));
        std::free(logits);
        granite_speech_kv_reset(ctx);
        logits = granite_speech_run_llm_kv(ctx, static_cast<float*>(original_embeds->data), total, 0, &nt, &nv);
        require(logits && nt == 1 && nv > 0, "original-input prefill");
        save("llm_prefill_logits.bin", logits, size_t(nv) * sizeof(float));
        const int first = int(std::max_element(logits, logits + nv) - logits);
        std::free(logits);
        const int eos = granite_speech_eos_token_id(ctx);
        granite_speech_set_decode_bucket(ctx, total + max_new);
        auto* tokens = granite_speech_greedy_decode(ctx, first, total, max_new, eos, &n);
        require(tokens && n > 0, "bucketed greedy decode");
        save("generated_ids.bin", tokens, size_t(n) * sizeof(int32_t));
        std::vector<int32_t> dynamic;
        granite_speech_set_decode_bucket(ctx, 0);
        granite_speech_kv_reset(ctx);
        logits = granite_speech_run_llm_kv(ctx, static_cast<float*>(original_embeds->data), total, 0, &nt, &nv);
        require(logits != nullptr, "dynamic prefill");
        int next = int(std::max_element(logits, logits + nv) - logits);
        std::free(logits);
        for (int i = 0; i < max_new; ++i) {
            dynamic.push_back(next);
            if (next == eos || i == max_new - 1)
                break;
            const int32_t id = next;
            auto* embedding = granite_speech_embed_tokens(ctx, &id, 1);
            require(embedding != nullptr, "dynamic embedding");
            logits = granite_speech_run_llm_kv(ctx, embedding, 1, total + i, &nt, &nv);
            std::free(embedding);
            require(logits != nullptr, "dynamic decode");
            next = int(std::max_element(logits, logits + nv) - logits);
            std::free(logits);
        }
        save("dynamic_generated_ids.bin", dynamic.data(), dynamic.size() * sizeof(int32_t));
        int text_n = n;
        if (tokens[n - 1] == eos)
            --text_n;
        auto* text = granite_speech_decode_tokens(ctx, tokens, text_n);
        require(text != nullptr, "token decoder");
        save("generated_text.txt", text, std::strlen(text));
        std::free(text);
        std::free(tokens);
        granite_speech_free(ctx);
        gguf_free(archive);
        ggml_free(refs);
        puts("GRANITE5_STAGE_PROBE_COMPLETE");
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "Granite5 stage probe failed: %s\n", e.what());
        return 1;
    }
}
