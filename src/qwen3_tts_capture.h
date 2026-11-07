// qwen3_tts_capture.h — opt-in, lossless Qwen3-TTS codec capture.
// Disabled unless CRISPASR_QWEN3_TTS_CAPTURE_DIR is an absolute path.
// No model/audio path is changed when capture is disabled.

#pragma once

#include "core/crispasr_sha256.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace qwen3_tts_capture {

namespace detail {

inline std::atomic<uint64_t>& file_counter() {
    static std::atomic<uint64_t> counter{0};
    return counter;
}

inline std::string hex_digest(const uint8_t* bytes, size_t size) {
    static const char* hex = "0123456789abcdef";
    std::string out;
    out.reserve(size * 2);
    for (size_t i = 0; i < size; ++i) {
        out.push_back(hex[bytes[i] >> 4]);
        out.push_back(hex[bytes[i] & 0x0f]);
    }
    return out;
}

inline std::string sha256_bytes(const void* data, size_t size) {
    crispasr::sha::Sha256 hash;
    hash.init();
    if (data && size)
        hash.update(static_cast<const uint8_t*>(data), size);
    uint8_t digest[32];
    hash.final(digest);
    return hex_digest(digest, sizeof(digest));
}

inline std::string sha256_i32le(const int32_t* values, size_t count) {
    crispasr::sha::Sha256 hash;
    hash.init();
    uint8_t bytes[4];
    for (size_t i = 0; i < count; ++i) {
        const uint32_t value = static_cast<uint32_t>(values[i]);
        bytes[0] = static_cast<uint8_t>(value);
        bytes[1] = static_cast<uint8_t>(value >> 8);
        bytes[2] = static_cast<uint8_t>(value >> 16);
        bytes[3] = static_cast<uint8_t>(value >> 24);
        hash.update(bytes, sizeof(bytes));
    }
    uint8_t digest[32];
    hash.final(digest);
    return hex_digest(digest, sizeof(digest));
}

inline std::string sha256_f32le(const float* values, size_t count) {
    crispasr::sha::Sha256 hash;
    hash.init();
    uint8_t bytes[4];
    for (size_t i = 0; i < count; ++i) {
        uint32_t value = 0;
        static_assert(sizeof(value) == sizeof(values[i]), "float32 required");
        std::memcpy(&value, values + i, sizeof(value));
        bytes[0] = static_cast<uint8_t>(value);
        bytes[1] = static_cast<uint8_t>(value >> 8);
        bytes[2] = static_cast<uint8_t>(value >> 16);
        bytes[3] = static_cast<uint8_t>(value >> 24);
        hash.update(bytes, sizeof(bytes));
    }
    uint8_t digest[32];
    hash.final(digest);
    return hex_digest(digest, sizeof(digest));
}

inline uintmax_t file_size(const std::string& path) {
    if (path.empty())
        return 0;
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    return ec ? 0 : size;
}

inline std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (unsigned char c : value) {
        switch (c) {
        case '"':
            out << "\\\"";
            break;
        case '\\':
            out << "\\\\";
            break;
        case '\b':
            out << "\\b";
            break;
        case '\f':
            out << "\\f";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            if (c < 0x20) {
                static const char* hex = "0123456789abcdef";
                out << "\\u00" << hex[c >> 4] << hex[c & 0x0f];
            } else {
                out << static_cast<char>(c);
            }
        }
    }
    return out.str();
}

inline std::string json_string(const std::string& value) {
    return "\"" + json_escape(value) + "\"";
}

inline std::string json_nullable_string(const std::string& value) {
    return value.empty() ? "null" : json_string(value);
}

inline std::string json_float(double value) {
    if (!std::isfinite(value))
        return "null";
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(10) << value;
    return out.str();
}

inline std::string utc_timestamp() {
    const auto now = std::chrono::system_clock::now();
    const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &time);
#else
    gmtime_r(&time, &tm);
#endif
    char date[32]{};
    std::strftime(date, sizeof(date), "%Y-%m-%dT%H:%M:%S", &tm);
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << date << "." << std::setw(3) << std::setfill('0') << millis << "Z";
    return out.str();
}

inline std::string safe_component(const std::string& value) {
    std::string out;
    out.reserve(std::min<size_t>(value.size(), 64));
    for (unsigned char c : value) {
        if (out.size() >= 64)
            break;
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_')
            out.push_back(static_cast<char>(c));
        else
            out.push_back('_');
    }
    return out.empty() ? "local" : out;
}

inline std::string make_id(const std::string& prefix) {
    const auto ticks =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count();
    const auto sequence = file_counter().fetch_add(1, std::memory_order_relaxed);
    return safe_component(prefix) + "-" + std::to_string(ticks) + "-" + std::to_string(sequence);
}

inline std::filesystem::path configured_root() noexcept {
    try {
        const char* raw = std::getenv("CRISPASR_QWEN3_TTS_CAPTURE_DIR");
        if (!raw || !*raw)
            return {};
        std::filesystem::path root(raw);
        if (!root.is_absolute()) {
            static std::atomic<bool> warned{false};
            if (!warned.exchange(true))
                std::fprintf(
                    stderr,
                    "qwen3_tts_capture: CRISPASR_QWEN3_TTS_CAPTURE_DIR must be an absolute path; capture disabled\n");
            return {};
        }
        return root;
    } catch (...) {
        std::fprintf(stderr, "qwen3_tts_capture: failed to read capture directory; capture disabled\n");
        return {};
    }
}

// The speaker vector is potentially identifiable voice data. Export only after
// a separate, explicit opt-in; capture itself remains controlled by CAPTURE_DIR.
inline bool export_speaker_embedding_enabled() noexcept {
    const char* value = std::getenv("CRISPASR_QWEN3_TTS_CAPTURE_SPEAKER_EMBEDDING");
    return value && std::strcmp(value, "1") == 0;
}

inline bool write_atomic(const std::filesystem::path& path, const uint8_t* data, size_t size, std::string& error) {
    if (!data && size) {
        error = "null capture data";
        return false;
    }
    std::error_code ec;
    const bool destination_exists = std::filesystem::exists(path, ec);
    if (ec) {
        error = "cannot inspect capture destination: " + path.string() + ": " + ec.message();
        return false;
    }
    if (destination_exists) {
        error = "refusing to overwrite existing capture file: " + path.string();
        return false;
    }

    const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto sequence = file_counter().fetch_add(1, std::memory_order_relaxed);
    const auto temp = path.string() + ".tmp." + std::to_string(ticks) + "." + std::to_string(sequence);
    {
        std::ofstream file(temp, std::ios::binary | std::ios::out | std::ios::trunc);
        if (!file) {
            error = "cannot open temporary capture file: " + temp;
            return false;
        }
        if (size)
            file.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
        file.flush();
        if (!file) {
            error = "failed writing temporary capture file: " + temp;
            file.close();
            std::filesystem::remove(temp, ec);
            return false;
        }
        file.close();
        if (!file) {
            error = "failed closing temporary capture file: " + temp;
            std::filesystem::remove(temp, ec);
            return false;
        }
    }

    // Hard-link creation is atomic and fails if the destination already exists
    // on POSIX and Windows filesystems that support hard links. Unlike POSIX
    // rename(), it can never replace a concurrently-created destination.
    std::filesystem::create_hard_link(temp, path, ec);
    if (ec) {
        error = "atomic no-overwrite commit failed for capture file: " + path.string() + ": " + ec.message();
        std::error_code cleanup_ec;
        std::filesystem::remove(temp, cleanup_ec);
        return false;
    }
    std::error_code cleanup_ec;
    std::filesystem::remove(temp, cleanup_ec);
    if (cleanup_ec)
        std::fprintf(stderr, "qwen3_tts_capture: temporary file cleanup failed: %s\n", cleanup_ec.message().c_str());
    return true;
}

inline bool write_text_atomic(const std::filesystem::path& path, const std::string& text, std::string& error) {
    return write_atomic(path, reinterpret_cast<const uint8_t*>(text.data()), text.size(), error);
}

inline bool write_i32le_atomic(const std::filesystem::path& path, const int32_t* values, size_t count,
                               std::string& error) {
    if (count > std::numeric_limits<size_t>::max() / 4) {
        error = "int32 capture size overflow";
        return false;
    }
    if (!values && count) {
        error = "null int32 capture data";
        return false;
    }
    std::vector<uint8_t> bytes(count * 4);
    for (size_t i = 0; i < count; ++i) {
        const uint32_t value = static_cast<uint32_t>(values[i]);
        bytes[i * 4 + 0] = static_cast<uint8_t>(value);
        bytes[i * 4 + 1] = static_cast<uint8_t>(value >> 8);
        bytes[i * 4 + 2] = static_cast<uint8_t>(value >> 16);
        bytes[i * 4 + 3] = static_cast<uint8_t>(value >> 24);
    }
    return write_atomic(path, bytes.data(), bytes.size(), error);
}

// Preserve the exact bits of the runtime's speaker embedding, not a second
// inference/recomputed speaker representation. The SHA must match the existing
// canonical little-endian fingerprint before the private file is committed.
inline bool write_f32le_atomic(const std::filesystem::path& path, const float* values, size_t count,
                               const std::string& expected_sha256, std::string& error) {
    static_assert(sizeof(float) == 4, "float32 required");
    if (!values || count != 2048) {
        error = "speaker embedding export requires exactly 2048 float32 values";
        return false;
    }
    std::vector<uint8_t> bytes(count * 4);
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(values[i])) {
            error = "speaker embedding export contains non-finite value";
            return false;
        }
        uint32_t bits = 0;
        std::memcpy(&bits, values + i, sizeof(bits));
        bytes[i * 4] = static_cast<uint8_t>(bits);
        bytes[i * 4 + 1] = static_cast<uint8_t>(bits >> 8);
        bytes[i * 4 + 2] = static_cast<uint8_t>(bits >> 16);
        bytes[i * 4 + 3] = static_cast<uint8_t>(bits >> 24);
    }
    if (detail::sha256_bytes(bytes.data(), bytes.size()) != expected_sha256) {
        error = "speaker embedding bytes disagree with the capture fingerprint";
        return false;
    }
    return write_atomic(path, bytes.data(), bytes.size(), error);
}

inline std::string stop_reason_name(int reason) {
    switch (reason) {
    case 1:
        return "eos";
    case 2:
        return "max_frames";
    case 3:
        return "kv_full_without_eos";
    case 4:
        return "stream_decoder_stop";
    case 5:
        return "replay_input_exhausted";
    default:
        return "unknown";
    }
}

} // namespace detail

inline bool enabled() noexcept {
    return !detail::configured_root().empty();
}

// The CLI factory maps every Qwen3-TTS alias to a backend whose name() is
// canonical. Check the instantiated backend rather than the user-supplied alias.
inline bool is_qwen_backend_name(const char* name) noexcept {
    return name && std::strcmp(name, "qwen3-tts") == 0;
}

struct Metadata {
    std::string request_id;
    std::string model_input_text;
    std::string source_text;
    std::string voice;
    std::string instruction;
    std::string model_path;
    std::string model_name;
    std::string model_type;
    std::string model_revision;
    std::string model_sha256;
    std::string tokenizer_id;
    std::string tokenizer_sha256;
    std::string runtime_revision;
    std::string codec_path;
    std::string codec_sha256;
    std::string reference_source_path;
    std::string reference_source_sha256;
    std::string reference_text;
    uint64_t seed = 0;
    float temperature = 0.0f;
    float temperature_effective = 0.9f;
    float speed = 1.0f;
    bool greedy = false;
    bool skip_reference_decode = true;
    int language_id = -1;
};

struct DecoderCall {
    std::string file;
    std::string sha256;
    std::string decoded_pcm_sha256;
    std::string emitted_pcm_sha256;
    int frame_start = 0;
    int frame_count = 0;
    int codebook_count = 0;
    int generated_frame_offset = 0;
    int pre_roll_frames = 0;
    int discard_samples = 0;
    int decoded_samples = 0;
    int emitted_samples = 0;
    bool windowed = false;
};

// A streaming backend must send a terminal callback to distinguish a complete
// segment from a failure after emitting only a prefix.
struct StreamCompletion {
    void observe(bool is_final) noexcept { final_seen = final_seen || is_final; }
    bool complete() const noexcept { return final_seen; }

private:
    bool final_seen = false;
};

class SampleCapture {
public:
    static std::unique_ptr<SampleCapture> begin_from_env(const Metadata& metadata, const int32_t* reference_codes,
                                                         size_t reference_code_count, const float* speaker_embedding,
                                                         size_t speaker_embedding_count, int codebook_count) noexcept {
        return begin(detail::configured_root(), metadata, reference_codes, reference_code_count, speaker_embedding,
                     speaker_embedding_count, codebook_count, detail::export_speaker_embedding_enabled());
    }

    static std::unique_ptr<SampleCapture> begin(const std::filesystem::path& root, const Metadata& metadata,
                                                const int32_t* reference_codes, size_t reference_code_count,
                                                const float* speaker_embedding, size_t speaker_embedding_count,
                                                int codebook_count, bool export_speaker_embedding = false) noexcept {
        try {
            if (root.empty() || !root.is_absolute() || codebook_count <= 0 ||
                (reference_code_count && (!reference_codes || reference_code_count % (size_t)codebook_count != 0))) {
                return nullptr;
            }
            auto capture = std::unique_ptr<SampleCapture>(new SampleCapture());
            capture->metadata_ = metadata;
            capture->codebook_count_ = codebook_count;
            capture->root_ = root;
            std::error_code ec;
            std::filesystem::create_directories(root / "samples", ec);
            if (ec) {
                std::fprintf(stderr, "qwen3_tts_capture: cannot create samples directory: %s\n", ec.message().c_str());
                return nullptr;
            }
            for (int attempt = 0; attempt < 32; ++attempt) {
                capture->sample_id_ = detail::make_id(metadata.request_id);
                capture->directory_ = root / "samples" / capture->sample_id_;
                ec.clear();
                if (std::filesystem::create_directory(capture->directory_, ec)) {
                    capture->active_ = true;
                    break;
                }
                if (ec && ec != std::errc::file_exists) {
                    std::fprintf(stderr, "qwen3_tts_capture: cannot create sample directory: %s\n",
                                 ec.message().c_str());
                    return nullptr;
                }
            }
            if (!capture->active_) {
                std::fprintf(stderr, "qwen3_tts_capture: could not allocate a unique sample directory\n");
                return nullptr;
            }
            capture->model_size_bytes_ = detail::file_size(metadata.model_path);
            capture->codec_size_bytes_ = detail::file_size(metadata.codec_path);
            capture->reference_source_size_bytes_ = detail::file_size(metadata.reference_source_path);
            capture->reference_source_sha256_ = metadata.reference_source_sha256;
            if (reference_code_count) {
                capture->reference_frame_count_ = (int)(reference_code_count / (size_t)codebook_count);
                capture->reference_codes_sha256_ = detail::sha256_i32le(reference_codes, reference_code_count);
                std::string error;
                if (!detail::write_i32le_atomic(capture->directory_ / "reference_codes.i32le", reference_codes,
                                                reference_code_count, error))
                    capture->set_error(error);
                else
                    capture->reference_codes_file_ = "reference_codes.i32le";
            }
            if (speaker_embedding && speaker_embedding_count)
                capture->speaker_embedding_sha256_ = detail::sha256_f32le(speaker_embedding, speaker_embedding_count);
            if (export_speaker_embedding && metadata.model_type == "base") {
                std::string error;
                if (!detail::write_f32le_atomic(capture->directory_ / "speaker_embedding.f32le", speaker_embedding,
                                                speaker_embedding_count, capture->speaker_embedding_sha256_, error))
                    capture->set_error(error);
                else
                    capture->speaker_embedding_file_ = "speaker_embedding.f32le";
            }
            return capture;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "qwen3_tts_capture: begin failed: %s\n", e.what());
            return nullptr;
        } catch (...) {
            std::fprintf(stderr, "qwen3_tts_capture: begin failed with unknown error\n");
            return nullptr;
        }
    }

    ~SampleCapture() {
        if (active_ && !finished_)
            fail("capture_abandoned");
    }

    bool active() const { return active_; }
    const std::string& sample_id() const { return sample_id_; }

    bool write_generated_codes(const int32_t* codes, size_t code_count, int frame_count, int stop_reason,
                               int eos_token_id, int max_frames) noexcept {
        try {
            if (!active_ || !codes || code_count == 0 || code_count % (size_t)codebook_count_ != 0 ||
                frame_count != (int)(code_count / (size_t)codebook_count_)) {
                set_error("invalid generated-code shape");
                return false;
            }
            std::string error;
            if (!detail::write_i32le_atomic(directory_ / "generated_codes.i32le", codes, code_count, error)) {
                set_error(error);
                return false;
            }
            generated_codes_sha256_ = detail::sha256_i32le(codes, code_count);
            generated_frame_count_ = frame_count;
            stop_reason_ = stop_reason;
            eos_token_id_ = eos_token_id;
            max_frames_ = max_frames;
            generated_codes_written_ = true;
            if (stop_reason_ != 1)
                set_error("generation_stopped_without_eos");
            return true;
        } catch (const std::exception& e) {
            set_error(e.what());
            return false;
        } catch (...) {
            set_error("unknown generated-code capture failure");
            return false;
        }
    }

    size_t record_decoder_call(const int32_t* codes, int frame_count, int codebook_count, int frame_start,
                               int generated_frame_offset, int pre_roll_frames, int discard_samples,
                               bool windowed) noexcept {
        if (!active_ || !codes || frame_count <= 0 || codebook_count != codebook_count_ || frame_start < 0 ||
            generated_frame_offset < 0 || pre_roll_frames < 0 || discard_samples < 0) {
            set_error("invalid decoder-input shape");
            return invalid_call();
        }
        try {
            std::string relative;
            std::filesystem::path path;
            if (windowed) {
                std::error_code ec;
                std::filesystem::create_directories(directory_ / "decoder_windows", ec);
                if (ec) {
                    set_error("cannot create decoder_windows: " + ec.message());
                    return invalid_call();
                }
                std::ostringstream name;
                name.imbue(std::locale::classic());
                name << "decoder_windows/window_" << std::setw(4) << std::setfill('0') << decoder_calls_.size()
                     << ".i32le";
                relative = name.str();
                path = directory_ / relative;
            } else {
                relative = "decoder_input_codes.i32le";
                path = directory_ / relative;
            }
            const size_t code_count = (size_t)frame_count * (size_t)codebook_count;
            std::string error;
            if (!detail::write_i32le_atomic(path, codes, code_count, error)) {
                set_error(error);
                return invalid_call();
            }
            DecoderCall call;
            call.file = relative;
            call.sha256 = detail::sha256_i32le(codes, code_count);
            call.frame_start = frame_start;
            call.frame_count = frame_count;
            call.codebook_count = codebook_count;
            call.generated_frame_offset = generated_frame_offset;
            call.pre_roll_frames = pre_roll_frames;
            call.discard_samples = discard_samples;
            call.windowed = windowed;
            decoder_calls_.push_back(std::move(call));
            return decoder_calls_.size() - 1;
        } catch (const std::exception& e) {
            set_error(e.what());
            return invalid_call();
        } catch (...) {
            set_error("unknown decoder-input capture error");
            return invalid_call();
        }
    }

    void record_decoder_result(size_t index, const float* decoded_pcm, int decoded_samples,
                               const float* emitted_pcm = nullptr, int emitted_samples = 0,
                               int actual_discard_samples = -1) noexcept {
        try {
            if (!active_ || index >= decoder_calls_.size())
                return;
            if (actual_discard_samples >= 0)
                decoder_calls_[index].discard_samples = actual_discard_samples;
            if (decoded_pcm && decoded_samples > 0) {
                decoder_calls_[index].decoded_samples = decoded_samples;
                decoder_calls_[index].decoded_pcm_sha256 = detail::sha256_f32le(decoded_pcm, (size_t)decoded_samples);
            }
            if (emitted_pcm && emitted_samples > 0) {
                decoder_calls_[index].emitted_samples = emitted_samples;
                decoder_calls_[index].emitted_pcm_sha256 = detail::sha256_f32le(emitted_pcm, (size_t)emitted_samples);
            }
        } catch (const std::exception& e) {
            set_error(e.what());
        } catch (...) {
            set_error("unknown decoder-result capture failure");
        }
    }

    bool finish(const float* pcm, int sample_count, int sample_rate, int eos_token_id, int max_frames) noexcept {
        try {
            if (!active_ || finished_)
                return false;
            if (!pcm || sample_count <= 0 || sample_rate <= 0)
                set_error("missing decoded PCM");
            pcm_sample_count_ = sample_count > 0 ? sample_count : 0;
            sample_rate_ = sample_rate;
            if (pcm && sample_count > 0)
                pcm_sha256_ = detail::sha256_f32le(pcm, (size_t)sample_count);
            if (eos_token_id >= 0)
                eos_token_id_ = eos_token_id;
            if (max_frames > 0)
                max_frames_ = max_frames;
            finished_ = true;
            return write_manifest();
        } catch (const std::exception& e) {
            set_error(e.what());
            finished_ = true;
            write_manifest();
            return false;
        } catch (...) {
            set_error("unknown sample-finish capture failure");
            finished_ = true;
            write_manifest();
            return false;
        }
    }

    void fail(const std::string& reason) noexcept {
        try {
            if (!active_ || finished_)
                return;
            set_error(reason.empty() ? "capture_failed" : reason);
            finished_ = true;
            write_manifest();
        } catch (...) {
            finished_ = true;
        }
    }

private:
    SampleCapture() = default;
    static size_t invalid_call() { return std::numeric_limits<size_t>::max(); }

    void set_error(const char* error) noexcept {
        try {
            if (error_.empty() && error)
                error_ = error;
        } catch (...) {
        }
    }

    void set_error(const std::string& error) noexcept {
        try {
            if (error_.empty())
                error_ = error;
        } catch (...) {
        }
    }

    bool write_manifest() noexcept {
        try {
            const bool complete = error_.empty() && generated_codes_written_ && stop_reason_ == 1 &&
                                  !decoder_calls_.empty() &&
                                  std::all_of(decoder_calls_.begin(), decoder_calls_.end(),
                                              [](const DecoderCall& call) {
                                                  return call.decoded_samples > 0 && !call.decoded_pcm_sha256.empty();
                                              }) &&
                                  !pcm_sha256_.empty() && pcm_sample_count_ > 0;
            const char* speaker_shape_json = speaker_embedding_file_.empty() ? "null" : "[2048]";
            const size_t speaker_bytes = speaker_embedding_file_.empty() ? 0 : 8192;
            std::ostringstream out;
            out.imbue(std::locale::classic());
            out << "{\n"
                << "  \"schema\": \"crispasr.qwen3-tts.capture/1\",\n"
                << "  \"capture_complete\": " << (complete ? "true" : "false") << ",\n"
                << "  \"sample_id\": " << detail::json_string(sample_id_) << ",\n"
                << "  \"request_id\": " << detail::json_nullable_string(metadata_.request_id) << ",\n"
                << "  \"captured_at_utc\": " << detail::json_string(detail::utc_timestamp()) << ",\n"
                << "  \"model_input_text\": " << detail::json_string(metadata_.model_input_text) << ",\n"
                << "  \"request_source_text\": " << detail::json_nullable_string(metadata_.source_text) << ",\n"
                << "  \"voice\": " << detail::json_nullable_string(metadata_.voice) << ",\n"
                << "  \"instruction\": " << detail::json_nullable_string(metadata_.instruction) << ",\n"
                << "  \"model\": {\"path\": " << detail::json_nullable_string(metadata_.model_path)
                << ", \"name\": " << detail::json_nullable_string(metadata_.model_name)
                << ", \"type\": " << detail::json_nullable_string(metadata_.model_type)
                << ", \"revision\": " << detail::json_nullable_string(metadata_.model_revision)
                << ", \"size_bytes\": " << model_size_bytes_
                << ", \"sha256\": " << detail::json_nullable_string(metadata_.model_sha256) << "},\n"
                << "  \"tokenizer\": {\"id\": " << detail::json_nullable_string(metadata_.tokenizer_id)
                << ", \"sha256\": " << detail::json_nullable_string(metadata_.tokenizer_sha256) << "},\n"
                << "  \"runtime_revision\": " << detail::json_nullable_string(metadata_.runtime_revision) << ",\n"
                << "  \"codec\": {\"path\": " << detail::json_nullable_string(metadata_.codec_path)
                << ", \"size_bytes\": " << codec_size_bytes_
                << ", \"sha256\": " << detail::json_nullable_string(metadata_.codec_sha256) << "},\n"
                << "  \"reference\": {\"source_path\": "
                << detail::json_nullable_string(metadata_.reference_source_path)
                << ", \"source_size_bytes\": " << reference_source_size_bytes_
                << ", \"source_sha256\": " << detail::json_nullable_string(reference_source_sha256_)
                << ", \"transcript\": " << detail::json_nullable_string(metadata_.reference_text)
                << ", \"codes_file\": " << detail::json_nullable_string(reference_codes_file_)
                << ", \"codes_sha256\": " << detail::json_nullable_string(reference_codes_sha256_)
                << ", \"codes_shape\": [" << reference_frame_count_ << ", " << codebook_count_ << "]"
                << ", \"codebook_count\": " << codebook_count_
                << ", \"speaker_embedding_sha256\": " << detail::json_nullable_string(speaker_embedding_sha256_)
                << ", \"speaker_embedding_file\": " << detail::json_nullable_string(speaker_embedding_file_)
                << ", \"speaker_embedding_shape\": " << speaker_shape_json
                << ", \"speaker_embedding_size_bytes\": " << speaker_bytes << "},\n"
                << "  \"sampling\": {\"seed_effective\": " << metadata_.seed
                << ", \"temperature_requested\": " << detail::json_float(metadata_.temperature)
                << ", \"temperature_effective\": " << detail::json_float(metadata_.temperature_effective)
                << ", \"greedy\": " << (metadata_.greedy ? "true" : "false")
                << ", \"skip_reference_decode\": " << (metadata_.skip_reference_decode ? "true" : "false")
                << ", \"speed_requested\": " << detail::json_float(metadata_.speed)
                << ", \"language_id\": " << metadata_.language_id << "},\n"
                << "  \"generated_codes\": {\"file\": \"generated_codes.i32le\", \"dtype\": \"int32_le\", "
                   "\"axis_order\": [\"frame\", \"codebook\"], \"shape\": ["
                << generated_frame_count_ << ", " << codebook_count_ << "], \"codebook_count\": " << codebook_count_
                << ", \"sha256\": " << detail::json_nullable_string(generated_codes_sha256_)
                << ", \"eos_token_id\": " << eos_token_id_
                << ", \"eos_observed\": " << (stop_reason_ == 1 ? "true" : "false")
                << ", \"stop_reason\": " << detail::json_string(detail::stop_reason_name(stop_reason_))
                << ", \"max_frames\": " << max_frames_ << "},\n"
                << "  \"decoder_calls\": [";
            for (size_t i = 0; i < decoder_calls_.size(); ++i) {
                const auto& call = decoder_calls_[i];
                if (i)
                    out << ",";
                out << "\n    {\"file\": " << detail::json_string(call.file)
                    << ", \"dtype\": \"int32_le\", \"axis_order\": [\"frame\", \"codebook\"]" << ", \"shape\": ["
                    << call.frame_count << ", " << call.codebook_count << "]"
                    << ", \"codebook_count\": " << call.codebook_count
                    << ", \"windowed\": " << (call.windowed ? "true" : "false")
                    << ", \"frame_start\": " << call.frame_start
                    << ", \"generated_frame_offset\": " << call.generated_frame_offset
                    << ", \"pre_roll_frames\": " << call.pre_roll_frames
                    << ", \"discard_samples\": " << call.discard_samples
                    << ", \"sha256\": " << detail::json_nullable_string(call.sha256)
                    << ", \"decoded_samples\": " << call.decoded_samples
                    << ", \"decoded_pcm_sha256\": " << detail::json_nullable_string(call.decoded_pcm_sha256)
                    << ", \"emitted_samples\": " << call.emitted_samples
                    << ", \"emitted_pcm_sha256\": " << detail::json_nullable_string(call.emitted_pcm_sha256) << "}";
            }
            if (!decoder_calls_.empty())
                out << "\n  ";
            out << "],\n"
                << "  \"decoded_pcm\": {\"dtype\": \"float32_le\", \"sample_rate\": " << sample_rate_
                << ", \"channels\": 1, \"samples\": " << pcm_sample_count_ << ", \"duration_seconds\": "
                << detail::json_float(sample_rate_ > 0 ? (double)pcm_sample_count_ / sample_rate_ : 0.0)
                << ", \"sha256\": " << detail::json_nullable_string(pcm_sha256_) << "},\n"
                << "  \"http_request_manifest\": "
                << (metadata_.request_id.empty()
                        ? "null"
                        : detail::json_string("requests/" + detail::safe_component(metadata_.request_id) +
                                              "/manifest.json"))
                << ",\n"
                << "  \"capture_error\": " << detail::json_nullable_string(error_) << "\n"
                << "}\n";
            std::string error;
            if (!detail::write_text_atomic(directory_ / "manifest.json", out.str(), error)) {
                std::fprintf(stderr, "qwen3_tts_capture: manifest write failed: %s\n", error.c_str());
                return false;
            }
            if (!complete)
                std::fprintf(stderr, "qwen3_tts_capture: incomplete sample %s: %s\n", sample_id_.c_str(),
                             error_.empty() ? "missing required capture data" : error_.c_str());
            return complete;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "qwen3_tts_capture: manifest failed: %s\n", e.what());
            return false;
        } catch (...) {
            std::fprintf(stderr, "qwen3_tts_capture: manifest failed with unknown error\n");
            return false;
        }
    }

    std::filesystem::path root_;
    std::filesystem::path directory_;
    Metadata metadata_;
    std::string sample_id_;
    std::string error_;
    std::string generated_codes_sha256_;
    std::string reference_codes_file_;
    std::string reference_codes_sha256_;
    std::string reference_source_sha256_;
    std::string speaker_embedding_sha256_;
    std::string speaker_embedding_file_;
    std::string pcm_sha256_;
    std::vector<DecoderCall> decoder_calls_;
    uintmax_t model_size_bytes_ = 0;
    uintmax_t codec_size_bytes_ = 0;
    uintmax_t reference_source_size_bytes_ = 0;
    int codebook_count_ = 0;
    int reference_frame_count_ = 0;
    int generated_frame_count_ = 0;
    int stop_reason_ = 0;
    int eos_token_id_ = -1;
    int max_frames_ = 0;
    int pcm_sample_count_ = 0;
    int sample_rate_ = 0;
    bool generated_codes_written_ = false;
    bool active_ = false;
    bool finished_ = false;
};

class HttpResponseCapture {
public:
    static std::shared_ptr<HttpResponseCapture> begin_from_env(const std::string& request_id,
                                                               const std::string& response_format, int sample_rate,
                                                               float speed, bool streaming) noexcept {
        return begin(detail::configured_root(), request_id, response_format, sample_rate, speed, streaming);
    }

    static std::shared_ptr<HttpResponseCapture> begin(const std::filesystem::path& root, const std::string& request_id,
                                                      const std::string& response_format, int sample_rate, float speed,
                                                      bool streaming) noexcept {
        try {
            if (root.empty() || !root.is_absolute() || request_id.empty())
                return nullptr;
            auto capture = std::shared_ptr<HttpResponseCapture>(new HttpResponseCapture());
            capture->request_id_ = request_id;
            capture->response_format_ = response_format;
            capture->sample_rate_ = sample_rate;
            capture->speed_ = speed;
            capture->streaming_ = streaming;
            capture->root_ = root;
            std::error_code ec;
            std::filesystem::create_directories(root / "requests", ec);
            if (ec)
                return nullptr;
            capture->directory_ = root / "requests" / detail::safe_component(request_id);
            if (!std::filesystem::create_directory(capture->directory_, ec)) {
                std::fprintf(stderr, "qwen3_tts_capture: refusing duplicate request manifest for %s\n",
                             detail::safe_component(request_id).c_str());
                return nullptr;
            }
            capture->active_ = true;
            capture->body_hash_.init();
            return capture;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "qwen3_tts_capture: HTTP capture begin failed: %s\n", e.what());
            return nullptr;
        } catch (...) {
            std::fprintf(stderr, "qwen3_tts_capture: HTTP capture begin failed\n");
            return nullptr;
        }
    }

    ~HttpResponseCapture() {
        if (active_ && !finished_)
            finish("application/octet-stream", false, "response_capture_abandoned");
    }

    bool active() const { return active_; }

    void record_chunk(const void* data, size_t size) noexcept {
        if (!active_ || finished_ || (!data && size))
            return;
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            if (finished_)
                return;
            if (size)
                body_hash_.update(static_cast<const uint8_t*>(data), size);
            Chunk chunk;
            chunk.bytes = size;
            chunk.sha256 = detail::sha256_bytes(data, size);
            chunks_.push_back(std::move(chunk));
            total_bytes_ += size;
        } catch (...) {
            try {
                if (error_.empty())
                    error_ = "failed to record HTTP response chunk";
            } catch (...) {
            }
        }
    }

    void finish_buffered(const std::string& content_type, const void* data, size_t size) noexcept {
        record_chunk(data, size);
        finish(content_type, true, "");
    }

    void finish(const std::string& content_type, bool complete, const std::string& failure_reason) noexcept {
        if (!active_ || finished_)
            return;
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            if (finished_)
                return;
            uint8_t digest[32];
            body_hash_.final(digest);
            const std::string body_sha256 = detail::hex_digest(digest, sizeof(digest));
            const bool final_complete = complete && error_.empty();
            std::ostringstream out;
            out.imbue(std::locale::classic());
            out << "{\n"
                << "  \"schema\": \"crispasr.qwen3-tts.http-response/1\",\n"
                << "  \"request_id\": " << detail::json_string(request_id_) << ",\n"
                << "  \"captured_at_utc\": " << detail::json_string(detail::utc_timestamp()) << ",\n"
                << "  \"complete\": " << (final_complete ? "true" : "false") << ",\n"
                << "  \"transport\": " << detail::json_string(streaming_ ? "chunked" : "buffered") << ",\n"
                << "  \"response_format\": " << detail::json_string(response_format_) << ",\n"
                << "  \"content_type\": " << detail::json_nullable_string(content_type) << ",\n"
                << "  \"sample_rate\": " << sample_rate_ << ",\n"
                << "  \"speed_requested\": " << detail::json_float(speed_) << ",\n"
                << "  \"body_bytes\": " << total_bytes_ << ",\n"
                << "  \"body_sha256\": " << detail::json_string(body_sha256) << ",\n"
                << "  \"chunk_hash_scope\": \"application response-body payload after successful sink.write; excludes "
                   "HTTP transfer framing\",\n"
                << "  \"chunks\": [";
            for (size_t i = 0; i < chunks_.size(); ++i) {
                if (i)
                    out << ",";
                out << "\n    {\"index\": " << i << ", \"bytes\": " << chunks_[i].bytes
                    << ", \"sha256\": " << detail::json_string(chunks_[i].sha256) << "}";
            }
            if (!chunks_.empty())
                out << "\n  ";
            out << "],\n"
                << "  \"capture_error\": " << detail::json_nullable_string(!error_.empty() ? error_ : failure_reason)
                << "\n"
                << "}\n";
            std::string error;
            if (!detail::write_text_atomic(directory_ / "manifest.json", out.str(), error))
                std::fprintf(stderr, "qwen3_tts_capture: HTTP manifest write failed: %s\n", error.c_str());
            finished_ = true;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "qwen3_tts_capture: HTTP manifest failed: %s\n", e.what());
            finished_ = true;
        } catch (...) {
            std::fprintf(stderr, "qwen3_tts_capture: HTTP manifest failed\n");
            finished_ = true;
        }
    }

private:
    struct Chunk {
        size_t bytes = 0;
        std::string sha256;
    };

    HttpResponseCapture() = default;
    std::filesystem::path root_;
    std::filesystem::path directory_;
    std::string request_id_;
    std::string response_format_;
    std::string error_;
    std::vector<Chunk> chunks_;
    crispasr::sha::Sha256 body_hash_;
    std::mutex mutex_;
    uint64_t total_bytes_ = 0;
    int sample_rate_ = 0;
    float speed_ = 1.0f;
    bool streaming_ = false;
    bool active_ = false;
    bool finished_ = false;
};

} // namespace qwen3_tts_capture
