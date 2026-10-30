#pragma once

#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <initializer_list>

// #551: architecture names are not tensor schemas. In particular,
// transcribe.cpp writes arch=parakeet too, with stt.parakeet.* metadata and
// pred.* tensors. Missing CrispASR predictor names does NOT imply CTC.
namespace core_parakeet_gguf {

inline bool family_backend(const char* name) {
    for (const char* candidate :
         {"parakeet", "haas", "haas-asr", "haas_asr", "haas-ja", "haas_ja", "phonon2", "reazonspeech", "quds",
          "quds-fa", "fastconformer-ctc", "fastconformer_ctc", "canary-ctc", "canary_ctc"}) {
        if (std::strcmp(name, candidate) == 0)
            return true;
    }
    return false;
}

inline bool external_schema(const gguf_context* ctx) {
    for (int64_t i = 0; i < gguf_get_n_kv(ctx); ++i) {
        if (std::strncmp(gguf_get_key(ctx, i), "stt.parakeet.", 13) == 0)
            return true;
    }
    return gguf_find_tensor(ctx, "pred.embed.weight") >= 0 || gguf_find_tensor(ctx, "pred.lstm.0.Wx") >= 0;
}

inline bool has_predictor(const gguf_context* ctx) {
    return gguf_find_tensor(ctx, "decoder.embed.weight") >= 0 || gguf_find_tensor(ctx, "decoder.lstm.0.w_ih") >= 0;
}

inline bool compatible_ctc(const gguf_context* ctx) {
    return !external_schema(ctx) && !has_predictor(ctx) && gguf_find_tensor(ctx, "encoder.pre.conv.0.weight") >= 0 &&
           gguf_find_tensor(ctx, "ctc.weight") >= 0 && gguf_find_tensor(ctx, "ctc.bias") >= 0;
}

inline const char* string_value(const gguf_context* ctx, const char* key, const char* fallback) {
    const int64_t i = gguf_find_key(ctx, key);
    return i >= 0 && gguf_get_kv_type(ctx, i) == GGUF_TYPE_STRING ? gguf_get_val_str(ctx, i) : fallback;
}

inline bool reject_external(const gguf_context* ctx, const char* path) {
    if (!external_schema(ctx))
        return false;
    fprintf(stderr,
            "crispasr: unsupported Parakeet GGUF architecture/layout for '%s' (model '%s', architecture '%s'): "
            "transcribe.cpp schema is not supported by CrispASR. This is not evidence of a pure-CTC model; "
            "forcing parakeet or fastconformer-ctc cannot make this layout compatible.\n",
            path, string_value(ctx, "general.name", "unnamed"), string_value(ctx, "general.architecture", "unknown"));
    return true;
}

inline bool reject_external_file(const char* path, bool report = true) {
    if (!path || !*path)
        return false;
    // Legacy Whisper and ONNX inputs must not produce GGUF bad-magic errors.
    FILE* file = std::fopen(path, "rb");
    if (!file)
        return false;
    char magic[4];
    const bool gguf =
        std::fread(magic, 1, sizeof(magic), file) == sizeof(magic) && std::memcmp(magic, "GGUF", sizeof(magic)) == 0;
    std::fclose(file);
    if (!gguf)
        return false;
    gguf_init_params params = {true, nullptr};
    gguf_context* ctx = gguf_init_from_file(path, params);
    if (!ctx)
        return false;
    const bool rejected = report ? reject_external(ctx, path) : external_schema(ctx);
    gguf_free(ctx);
    return rejected;
}

} // namespace core_parakeet_gguf
