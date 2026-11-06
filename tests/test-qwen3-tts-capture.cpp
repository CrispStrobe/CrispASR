// test-qwen3-tts-capture.cpp — model-free tests for opt-in codec capture.
#include <catch2/catch_test_macros.hpp>

#include "qwen3_tts_capture.h"
#include "../examples/json.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <locale>
#include <string>
#include <vector>

namespace {

struct TempDirectory {
    std::filesystem::path path;

    TempDirectory() {
        const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() / ("qwen3-tts-capture-" + std::to_string(ticks));
        std::filesystem::create_directories(path);
    }

    ~TempDirectory() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

std::string read_text(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    REQUIRE(file.good());
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

std::vector<uint8_t> read_bytes(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    REQUIRE(file.good());
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

} // namespace

TEST_CASE("Qwen3 capture SHA-256 uses the standard byte digest", "[unit][qwen3-tts][capture]") {
    CHECK(qwen3_tts_capture::detail::sha256_bytes("abc", 3) ==
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

TEST_CASE("Qwen3 HTTP capture recognizes the canonical backend identity", "[unit][qwen3-tts][capture]") {
    CHECK(qwen3_tts_capture::is_qwen_backend_name("qwen3-tts"));
    CHECK_FALSE(qwen3_tts_capture::is_qwen_backend_name("qwen3"));
    CHECK_FALSE(qwen3_tts_capture::is_qwen_backend_name(nullptr));
}

TEST_CASE("Qwen3 capture JSON numbers ignore the process locale", "[unit][qwen3-tts][capture]") {
    struct CommaPunctuation : std::numpunct<char> {
        char do_decimal_point() const override { return ','; }
        char do_thousands_sep() const override { return '.'; }
        std::string do_grouping() const override { return "\3"; }
    };
    struct LocaleRestore {
        std::locale original = std::locale();
        ~LocaleRestore() { std::locale::global(original); }
    } restore;
    std::locale::global(std::locale(std::locale::classic(), new CommaPunctuation));
    CHECK(qwen3_tts_capture::detail::json_float(1.5) == "1.5");
    TempDirectory temp;
    auto capture = qwen3_tts_capture::HttpResponseCapture::begin(temp.path, "locale", "pcm", 24000, 1.5f, false);
    REQUIRE(capture);
    capture->finish_buffered("audio/pcm", "x", 1);
    const auto manifest = read_text(temp.path / "requests" / "locale" / "manifest.json");
    const auto json = nlohmann::json::parse(manifest);
    CHECK(json.at("sample_rate") == 24000);
    CHECK(json.at("speed_requested") == 1.5);
}

TEST_CASE("Qwen3 sample capture stores exact little-endian generated and decoder codes", "[unit][qwen3-tts][capture]") {
    TempDirectory temp;
    qwen3_tts_capture::Metadata metadata;
    metadata.request_id = "0123456789abcdef";
    metadata.model_input_text = "Резервное копирование начнётся шестого ноября.";
    metadata.source_text = "Резервное копирование начнётся 06.11.2026.";
    metadata.voice = "menken-reference-voice";
    metadata.instruction = "calm, measured delivery";
    metadata.model_path = "base-control-f16.gguf";
    metadata.model_name = "Qwen3-TTS 1.7B Base";
    metadata.model_type = "base";
    metadata.model_revision = "test-model-revision";
    metadata.tokenizer_id = "embedded-gguf-text-tokenizer";
    metadata.tokenizer_sha256 = "test-tokenizer-sha256";
    metadata.runtime_revision = "test-runtime-revision";
    metadata.codec_path = "qwen3-tts-codec.gguf";
    metadata.reference_source_path = "reference.wav";
    metadata.reference_source_sha256 = "reference-source-sha256";
    metadata.reference_text = "reference transcript";
    metadata.seed = 4903;
    metadata.temperature = 0.65f;
    metadata.temperature_effective = 0.65f;
    metadata.skip_reference_decode = false;
    metadata.speed = 1.0f;
    metadata.language_id = 7;

    const int32_t reference_codes[] = {1, 2, 3, 4};
    const float speaker_embedding[] = {0.25f, -0.5f, 0.75f};
    auto capture =
        qwen3_tts_capture::SampleCapture::begin(temp.path, metadata, reference_codes, 4, speaker_embedding, 3, 2);
    REQUIRE(capture);
    REQUIRE(capture->active());

    const int32_t generated_codes[] = {10, 11, 12, 13};
    REQUIRE(capture->write_generated_codes(generated_codes, 4, 2, 1, 2048, 100));

    // The decoder receives the two reference frames followed by the two generated frames.
    const int32_t decoder_codes[] = {1, 2, 3, 4, 10, 11, 12, 13};
    const size_t call = capture->record_decoder_call(decoder_codes, 4, 2, 0, 2, 0, 0, false);
    REQUIRE(call != std::numeric_limits<size_t>::max());

    const float decoder_pcm[] = {0.1f, 0.2f, 0.3f, 0.4f};
    const float output_pcm[] = {0.3f, 0.4f};
    capture->record_decoder_result(call, decoder_pcm, 4, output_pcm, 2, 2);
    REQUIRE(capture->finish(output_pcm, 2, 24000, 2048, 100));

    const auto sample_dir = temp.path / "samples" / capture->sample_id();
    CHECK((read_bytes(sample_dir / "generated_codes.i32le") ==
           std::vector<uint8_t>{10, 0, 0, 0, 11, 0, 0, 0, 12, 0, 0, 0, 13, 0, 0, 0}));
    CHECK((read_bytes(sample_dir / "reference_codes.i32le") ==
           std::vector<uint8_t>{1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0}));
    CHECK((read_bytes(sample_dir / "decoder_input_codes.i32le") ==
           std::vector<uint8_t>{1,  0, 0, 0, 2,  0, 0, 0, 3,  0, 0, 0, 4,  0, 0, 0,
                                10, 0, 0, 0, 11, 0, 0, 0, 12, 0, 0, 0, 13, 0, 0, 0}));

    const std::string manifest = read_text(sample_dir / "manifest.json");
    CHECK(manifest.find("\"capture_complete\": true") != std::string::npos);
    CHECK(manifest.find("\"generated_frame_offset\": 2") != std::string::npos);
    CHECK(manifest.find("\"eos_observed\": true") != std::string::npos);
    CHECK(manifest.find("\"request_id\": \"0123456789abcdef\"") != std::string::npos);
    CHECK(manifest.find("06.11.2026") != std::string::npos);
    CHECK(manifest.find("\"request_source_text\"") != std::string::npos);
    CHECK(manifest.find("\"voice\": \"menken-reference-voice\"") != std::string::npos);
    CHECK(manifest.find("\"source_path\": \"reference.wav\"") != std::string::npos);
    CHECK(manifest.find("\"source_sha256\": \"reference-source-sha256\"") != std::string::npos);
    CHECK(manifest.find("\"instruction\": \"calm, measured delivery\"") != std::string::npos);
    CHECK(manifest.find("\"shape\": [2, 2]") != std::string::npos);
    CHECK(manifest.find("\"codebook_count\": 2") != std::string::npos);
    CHECK(manifest.find("\"discard_samples\": 2") != std::string::npos);
    CHECK(manifest.find("\"seed_effective\": 4903") != std::string::npos);
    CHECK(manifest.find("\"temperature_requested\": ") != std::string::npos);
    CHECK(manifest.find("\"temperature_effective\": ") != std::string::npos);
    CHECK(manifest.find("\"skip_reference_decode\": false") != std::string::npos);
    CHECK(manifest.find("\"name\": \"Qwen3-TTS 1.7B Base\"") != std::string::npos);
    CHECK(manifest.find("\"tokenizer\": {\"id\": \"embedded-gguf-text-tokenizer\"") != std::string::npos);
    CHECK(manifest.find("test-runtime-revision") != std::string::npos);
}

TEST_CASE("Qwen3 sample capture records streaming decoder windows and actual pre-roll discard",
          "[unit][qwen3-tts][capture]") {
    TempDirectory temp;
    qwen3_tts_capture::Metadata metadata;
    metadata.request_id = "window012345678";
    metadata.model_input_text = "streaming window";
    const int32_t generated_codes[] = {10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33};
    const int32_t window_codes[] = {30, 31, 32, 33};
    const float decoded_pcm[] = {0.1f, 0.2f, 0.3f, 0.4f};
    const float emitted_pcm[] = {0.3f, 0.4f};

    auto capture = qwen3_tts_capture::SampleCapture::begin(temp.path, metadata, nullptr, 0, nullptr, 0, 2);
    REQUIRE(capture);
    REQUIRE(capture->write_generated_codes(generated_codes, 12, 6, 1, 2048, 64));
    const size_t call = capture->record_decoder_call(window_codes, 2, 2, 4, 4, 1, 3, true);
    REQUIRE(call != std::numeric_limits<size_t>::max());
    capture->record_decoder_result(call, decoded_pcm, 4, emitted_pcm, 2, 2);
    REQUIRE(capture->finish(emitted_pcm, 2, 24000, 2048, 64));

    const auto sample_dir = temp.path / "samples" / capture->sample_id();
    CHECK((read_bytes(sample_dir / "decoder_windows" / "window_0000.i32le") ==
           std::vector<uint8_t>{30, 0, 0, 0, 31, 0, 0, 0, 32, 0, 0, 0, 33, 0, 0, 0}));
    const std::string manifest = read_text(sample_dir / "manifest.json");
    CHECK(manifest.find("\"windowed\": true") != std::string::npos);
    CHECK(manifest.find("\"frame_start\": 4") != std::string::npos);
    CHECK(manifest.find("\"generated_frame_offset\": 4") != std::string::npos);
    CHECK(manifest.find("\"pre_roll_frames\": 1") != std::string::npos);
    CHECK(manifest.find("\"discard_samples\": 2") != std::string::npos);
}

TEST_CASE("Qwen3 sample capture rejects non-EOS generations as complete training samples",
          "[unit][qwen3-tts][capture]") {
    TempDirectory temp;
    qwen3_tts_capture::Metadata metadata;
    metadata.model_input_text = "truncated generation";
    const int32_t generated_codes[] = {30, 31};
    const float pcm[] = {0.1f, 0.2f};
    auto capture = qwen3_tts_capture::SampleCapture::begin(temp.path, metadata, nullptr, 0, nullptr, 0, 2);
    REQUIRE(capture);
    REQUIRE(capture->write_generated_codes(generated_codes, 2, 1, 2, 2048, 1));
    const size_t call = capture->record_decoder_call(generated_codes, 1, 2, 0, 0, 0, 0, false);
    REQUIRE(call != std::numeric_limits<size_t>::max());
    capture->record_decoder_result(call, pcm, 2);
    CHECK_FALSE(capture->finish(pcm, 2, 24000, 2048, 1));
    const auto sample_dir = temp.path / "samples" / capture->sample_id();
    const std::string manifest = read_text(sample_dir / "manifest.json");
    CHECK(manifest.find("\"capture_complete\": false") != std::string::npos);
    CHECK(manifest.find("generation_stopped_without_eos") != std::string::npos);
}

TEST_CASE("Qwen3 capture cannot complete without every decoder result", "[unit][qwen3-tts][capture]") {
    TempDirectory temp;
    qwen3_tts_capture::Metadata metadata;
    const int32_t codes[] = {1, 2};
    const float pcm[] = {0.25f};
    auto capture = qwen3_tts_capture::SampleCapture::begin(temp.path, metadata, nullptr, 0, nullptr, 0, 2);
    REQUIRE(capture);
    REQUIRE(capture->write_generated_codes(codes, 2, 1, 1, 2048, 10));
    REQUIRE(capture->record_decoder_call(codes, 1, 2, 0, 0, 0, 0, false) != std::numeric_limits<size_t>::max());
    CHECK_FALSE(capture->finish(pcm, 1, 24000, 2048, 10));
    CHECK(
        read_text(temp.path / "samples" / capture->sample_id() / "manifest.json").find("\"capture_complete\": false") !=
        std::string::npos);
}

TEST_CASE("Qwen3 capture atomic writer never overwrites an existing file", "[unit][qwen3-tts][capture]") {
    TempDirectory temp;
    const auto path = temp.path / "immutable.txt";
    const std::string first = "original capture";
    const std::string second = "replacement capture";
    std::string error;

    REQUIRE(qwen3_tts_capture::detail::write_text_atomic(path, first, error));
    error.clear();
    CHECK_FALSE(qwen3_tts_capture::detail::write_text_atomic(path, second, error));
    CHECK(read_text(path) == first);
    CHECK_FALSE(error.empty());
}

TEST_CASE("Qwen3 HTTP response capture hashes buffered bodies and refuses request-id overwrite",
          "[unit][qwen3-tts][capture]") {
    TempDirectory temp;
    auto capture =
        qwen3_tts_capture::HttpResponseCapture::begin(temp.path, "abcdef0123456789", "wav", 24000, 1.0f, false);
    REQUIRE(capture);
    const std::string body = "synthetic-wav-payload";
    capture->finish_buffered("audio/wav", body.data(), body.size());

    const auto manifest_path = temp.path / "requests" / "abcdef0123456789" / "manifest.json";
    const std::string manifest = read_text(manifest_path);
    CHECK(manifest.find("\"complete\": true") != std::string::npos);
    CHECK(manifest.find("\"body_bytes\": " + std::to_string(body.size())) != std::string::npos);
    CHECK(manifest.find(qwen3_tts_capture::detail::sha256_bytes(body.data(), body.size())) != std::string::npos);

    auto duplicate =
        qwen3_tts_capture::HttpResponseCapture::begin(temp.path, "abcdef0123456789", "wav", 24000, 1.0f, false);
    CHECK_FALSE(duplicate);
    CHECK(read_text(manifest_path) == manifest);
}

TEST_CASE("Qwen3 HTTP streaming capture hashes each successful application chunk", "[unit][qwen3-tts][capture]") {
    TempDirectory temp;
    auto capture =
        qwen3_tts_capture::HttpResponseCapture::begin(temp.path, "fedcba9876543210", "pcm", 24000, 1.0f, true);
    REQUIRE(capture);
    capture->record_chunk("first", 5);
    capture->record_chunk("second", 6);
    capture->finish("audio/pcm", true, "");

    const auto manifest_path = temp.path / "requests" / "fedcba9876543210" / "manifest.json";
    const std::string manifest = read_text(manifest_path);
    CHECK(manifest.find("\"transport\": \"chunked\"") != std::string::npos);
    CHECK(manifest.find("\"chunk_hash_scope\"") != std::string::npos);
    CHECK(manifest.find("\"index\": 0, \"bytes\": 5") != std::string::npos);
    CHECK(manifest.find("\"index\": 1, \"bytes\": 6") != std::string::npos);
    CHECK(manifest.find(qwen3_tts_capture::detail::sha256_bytes("firstsecond", 11)) != std::string::npos);
}

TEST_CASE("Qwen3 HTTP streaming capture marks a disconnected response incomplete", "[unit][qwen3-tts][capture]") {
    TempDirectory temp;
    auto capture =
        qwen3_tts_capture::HttpResponseCapture::begin(temp.path, "0011223344556677", "pcm", 24000, 1.0f, true);
    REQUIRE(capture);
    capture->record_chunk("partial", 7);
    capture->finish("audio/pcm", false, "client_disconnected");

    const auto manifest_path = temp.path / "requests" / "0011223344556677" / "manifest.json";
    const std::string manifest = read_text(manifest_path);
    CHECK(manifest.find("\"complete\": false") != std::string::npos);
    CHECK(manifest.find("client_disconnected") != std::string::npos);
}

TEST_CASE("Qwen3 streaming completion requires a terminal callback", "[unit][qwen3-tts][capture]") {
    qwen3_tts_capture::StreamCompletion partial;
    partial.observe(false);
    CHECK_FALSE(partial.complete());
    partial.observe(true);
    CHECK(partial.complete());
    partial.observe(false);
    CHECK(partial.complete());
}
