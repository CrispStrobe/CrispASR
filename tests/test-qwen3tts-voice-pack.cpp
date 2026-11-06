// Exercise the actual public loader and private prefill guard without weights.
// As in test-omnivoice-codec, include the runtime for a minimal internal context.
#include "core/ggml_cpu_backend.h"
#include "../src/qwen3_tts.cpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>

static int pack_checks = 0;
static void check_pack(bool condition) {
    ++pack_checks;
    if (!condition)
        throw std::runtime_error("Qwen3 voice pack check " + std::to_string(pack_checks) + " failed");
}

static void write_pack(const std::filesystem::path& path, int dimension, bool two = false, bool half = false,
                       bool missing_text = false) {
    ggml_init_params ip = {128 * 1024, nullptr, false};
    auto* tensors = ggml_init(ip);
    auto* file = gguf_init_empty();
    const char* names[] = {"first", "second"};
    const char* texts[] = {"First synthetic fixture.", "Second synthetic fixture."};
    const int count = two ? 2 : 1;
    gguf_set_arr_str(file, "voicepack.names", names, count);
    if (!missing_text)
        gguf_set_arr_str(file, "voicepack.ref_texts", texts, count);
    for (int i = 0; i < count; ++i) {
        auto* speaker = ggml_new_tensor_1d(tensors, half ? GGML_TYPE_F16 : GGML_TYPE_F32, dimension);
        const auto speaker_name = std::string("voicepack.spk.") + names[i] + ".embd";
        ggml_set_name(speaker, speaker_name.c_str());
        std::memset(speaker->data, 0, ggml_nbytes(speaker));
        gguf_add_tensor(file, speaker);
        auto* codes = ggml_new_tensor_2d(tensors, GGML_TYPE_I32, 16, 2);
        const auto code_name = std::string("voicepack.code.") + names[i] + ".codes";
        ggml_set_name(codes, code_name.c_str());
        std::memset(codes->data, 0, ggml_nbytes(codes));
        gguf_add_tensor(file, codes);
    }
    check_pack(gguf_write_to_file(file, path.string().c_str(), false));
    gguf_free(file);
    ggml_free(tensors);
}

int main() {
    const auto folder = std::filesystem::temp_directory_path() /
                        ("qwen3-pack-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(folder);
    std::unique_ptr<qwen3_tts_context, decltype(&qwen3_tts_free)> ctx(new qwen3_tts_context, qwen3_tts_free);
    ctx->backend_cpu = core_cpu_backend::init();
    ctx->backend = ctx->backend_cpu;
    check_pack(ctx->backend != nullptr);
    try {
        write_pack(folder / "small.gguf", 1024, true);
        write_pack(folder / "large.gguf", 2048);
        write_pack(folder / "half.gguf", 1024, false, true);
        write_pack(folder / "missing-text.gguf", 1024, false, false, true);
        auto load = [&](const char* name) {
            return qwen3_tts_load_voice_pack(ctx.get(), (folder / name).string().c_str());
        };
        ctx->hp.d_model = 1024;
        check_pack(load("small.gguf") == 0);
        check_pack(qwen3_tts_select_voice(ctx.get(), "second") == 0);
        auto* original = ctx->vp_tensors.at("voicepack.spk.second.embd");
        auto* original_buffer = ctx->vp_buf_w;
        for (const char* name : {"large.gguf", "half.gguf", "missing-text.gguf", "missing.gguf"}) {
            check_pack(load(name) == -1);
            check_pack(ctx->vp_active == 1);
            check_pack(ctx->vp_ref_texts.size() == 2);
            check_pack(ctx->vp_tensors.at("voicepack.spk.second.embd") == original);
            check_pack(ctx->vp_buf_w == original_buffer);
        }
        // Reproduce the reporter's precise 8192-byte read of a 4096-byte tensor,
        // without needing a multi-GB model: the read preceded all neural compute.
        ctx->hp.d_model = 2048;
        int length = -1;
        check_pack(qwen3_tts_build_icl_prefill(ctx.get(), "hello world", "reference", &length) == nullptr);
        check_pack(length == 0);
        check_pack(load("small.gguf") == -1);
        check_pack(ctx->vp_active == 1);
        check_pack(load("large.gguf") == 0);
        check_pack(ctx->vp_active == 0);
        check_pack(ctx->vp_names.size() == 1);
        check_pack(ctx->vp_names[0] == "first");
        check_pack(ctx->vp_tensors.at("voicepack.spk.first.embd")->ne[0] == 2048);
        // Returning to the first model/pack remains possible after a replacement.
        ctx->hp.d_model = 1024;
        check_pack(load("small.gguf") == 0);
        check_pack(ctx->vp_active == 0);
        // Capture the actual pack vector, not merely runtime_spk_emb, which is
        // empty for the ordinary baked-voice path.
        ctx->hp.tts_model_type = "base";
        check_pack(ctx->runtime_spk_emb.empty());
#ifdef _WIN32
        _putenv_s("CRISPASR_QWEN3_TTS_CAPTURE_DIR", folder.string().c_str());
#else
        setenv("CRISPASR_QWEN3_TTS_CAPTURE_DIR", folder.string().c_str(), 1);
#endif
        auto capture = qwen3_tts_begin_sample_capture(ctx.get(), "hello", nullptr, 0, 16);
        check_pack(capture != nullptr);
        const int32_t codes[16] = {};
        const float pcm[] = {0.25f};
        check_pack(capture->write_generated_codes(codes, 16, 1, 1, 2048, 10));
        const size_t call = capture->record_decoder_call(codes, 1, 16, 0, 0, 0, 0, false);
        capture->record_decoder_result(call, pcm, 1);
        check_pack(capture->finish(pcm, 1, 24000, 2048, 10));
        std::ifstream manifest(folder / "samples" / capture->sample_id() / "manifest.json");
        const std::string contents((std::istreambuf_iterator<char>(manifest)), std::istreambuf_iterator<char>());
        const std::vector<float> expected_speaker(1024, 0.0f);
        check_pack(contents.find(qwen3_tts_capture::detail::sha256_f32le(expected_speaker.data(), 1024)) !=
                   std::string::npos);
        // A pack can remain loaded when an x-vector-only prompt is selected.
        // Its codes are not part of the active prefill.
        ctx->xvec_only = true;
        ctx->runtime_spk_emb.assign(1024, 0.5f);
        const int32_t stale_ref[16] = {};
        auto xvec_capture = qwen3_tts_begin_sample_capture(ctx.get(), "hello", stale_ref, 16, 16);
        check_pack(xvec_capture != nullptr);
        check_pack(!std::filesystem::exists(folder / "samples" / xvec_capture->sample_id() / "reference_codes.i32le"));
        check_pack(xvec_capture->write_generated_codes(codes, 16, 1, 1, 2048, 10));
        const size_t xvec_call = xvec_capture->record_decoder_call(codes, 1, 16, 0, 0, 0, 0, false);
        xvec_capture->record_decoder_result(xvec_call, pcm, 1);
        check_pack(xvec_capture->finish(pcm, 1, 24000, 2048, 10));
        std::ifstream xvec_file(folder / "samples" / xvec_capture->sample_id() / "manifest.json");
        const std::string xvec_manifest((std::istreambuf_iterator<char>(xvec_file)), std::istreambuf_iterator<char>());
        check_pack(xvec_manifest.find(qwen3_tts_capture::detail::sha256_f32le(ctx->runtime_spk_emb.data(), 1024)) !=
                   std::string::npos);
        ctx->hp.tts_model_type = "voice_design";
        ctx->runtime_instruct = "direct API instruction";
        check_pack(qwen3_tts_make_capture_metadata(ctx.get(), "hello").instruction == "direct API instruction");
        ctx->hp.tts_model_type = "custom_voice";
        ctx->runtime_cv_style_instruct = "direct API style";
        check_pack(qwen3_tts_make_capture_metadata(ctx.get(), "hello").instruction == "direct API style");
#ifndef _WIN32
        // Generation treats an explicitly empty seed as absent and uses 42.
        setenv("CRISPASR_QWEN3_TTS_SEED", "", 1);
        ctx->params.seed = 0;
        check_pack(qwen3_tts_make_capture_metadata(ctx.get(), "hello").seed == 42);
        setenv("CRISPASR_QWEN3_TTS_SEED", "123", 1);
        check_pack(qwen3_tts_make_capture_metadata(ctx.get(), "hello").seed == 123);
        ctx->params.seed = 456;
        check_pack(qwen3_tts_make_capture_metadata(ctx.get(), "hello").seed == 456);
        unsetenv("CRISPASR_QWEN3_TTS_SEED");
#endif
#ifdef _WIN32
        _putenv_s("CRISPASR_QWEN3_TTS_CAPTURE_DIR", "");
#else
        unsetenv("CRISPASR_QWEN3_TTS_CAPTURE_DIR");
#endif
        ctx.reset();
        std::filesystem::remove_all(folder);
        printf("Qwen3 voice pack lifecycle: %d checks passed\n", pack_checks);
        return 0;
    } catch (const std::exception& error) {
        fprintf(stderr, "%s\n", error.what());
        std::filesystem::remove_all(folder);
        return 1;
    }
}
