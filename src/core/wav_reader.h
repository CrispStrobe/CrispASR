// src/core/wav_reader.h — minimal WAV reader for CLI / backend adapters.
//
// Several backend adapters (indextts, voxcpm2-tts) need to load a reference
// WAV passed via `--voice <path>`. Each had its own copy of the same RIFF
// parser; this header is the single source of truth.
//
// Scope: PCM16 (plain or WAVE_FORMAT_EXTENSIBLE) mono or multi-channel only.
// Multi-channel audio is averaged to mono, independent of speaker positions.
// Returns native sample rate as-read; callers resample via
// `crispasr::core::resample_polyphase` (see core/audio_resample.h) if they
// need a specific rate.
//
// Header-only so callers can drop in by including this file alone.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/types.h>
#endif

namespace crispasr::core {

// Read a PCM16 WAV (mono or N-channel, averaged to mono).
// On success: fills `pcm` with float32 samples in [-1.0, 1.0], sets
// `sample_rate` to the file's native rate, returns true. On failure
// (file missing, unsupported format) returns false and leaves outputs
// undefined.
inline bool read_wav_mono_pcm16(const std::string& path, std::vector<float>& pcm, int& sample_rate) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }

    auto seek = [&](int64_t offset, int origin) {
#ifdef _WIN32
        return _fseeki64(f, offset, origin);
#else
        if (offset > std::numeric_limits<off_t>::max() || offset < std::numeric_limits<off_t>::min())
            return -1;
        return fseeko(f, (off_t)offset, origin);
#endif
    };
    auto tell = [&]() -> int64_t {
#ifdef _WIN32
        return _ftelli64(f);
#else
        return ftello(f);
#endif
    };

    // Bound chunk traversal to real file bytes. fseek alone accepts positions
    // past EOF, and RIFF sizes are unsigned (a high bit must never seek back).
    if (seek(0, SEEK_END) != 0) {
        std::fclose(f);
        return false;
    }
    const int64_t file_end = tell();
    if (file_end < 0 || seek(0, SEEK_SET) != 0) {
        std::fclose(f);
        return false;
    }
    auto skip = [&](uint64_t bytes) {
        const int64_t pos = tell();
        return pos >= 0 && pos <= file_end && bytes <= (uint64_t)(file_end - pos) &&
               seek((int64_t)bytes, SEEK_CUR) == 0;
    };

    char riff[4];
    if (std::fread(riff, 1, 4, f) != 4 || std::memcmp(riff, "RIFF", 4) != 0) {
        std::fclose(f);
        return false;
    }
    seek(4, SEEK_CUR); // skip chunk size

    char wave[4];
    if (std::fread(wave, 1, 4, f) != 4 || std::memcmp(wave, "WAVE", 4) != 0) {
        std::fclose(f);
        return false;
    }

    uint16_t audio_format = 0;
    int16_t n_channels = 0;
    int32_t sr = 0;
    int16_t bits_per_sample = 0;
    bool found_fmt = false, found_data = false;
    uint32_t data_size = 0;

    while (!std::feof(f)) {
        char chunk_id[4];
        uint32_t chunk_size = 0;
        if (std::fread(chunk_id, 1, 4, f) != 4) {
            break;
        }
        if (std::fread(&chunk_size, 4, 1, f) != 1) {
            break;
        }

        if (std::memcmp(chunk_id, "fmt ", 4) == 0) {
            if (chunk_size < 16) {
                break;
            }
            if (std::fread(&audio_format, 2, 1, f) != 1) {
                break;
            }
            if (std::fread(&n_channels, 2, 1, f) != 1) {
                break;
            }
            if (std::fread(&sr, 4, 1, f) != 1) {
                break;
            }
            uint16_t block_align = 0;
            if (!skip(4) || std::fread(&block_align, 2, 1, f) != 1) {
                break;
            }
            if (std::fread(&bits_per_sample, 2, 1, f) != 1) {
                break;
            }
            uint32_t consumed = 16;
            if (audio_format == 0xfffe) {
                // WAVEFORMATEXTENSIBLE adds cbSize, valid bits, a speaker
                // mask and a full subtype GUID. Only 16-bit integer PCM is
                // supported here; a matching low GUID word is insufficient.
                uint8_t extension[24];
                if (chunk_size < 40 || std::fread(extension, 1, sizeof(extension), f) != sizeof(extension)) {
                    break;
                }
                const uint16_t cb_size = extension[0] | (uint16_t(extension[1]) << 8);
                const uint16_t valid_bits = extension[2] | (uint16_t(extension[3]) << 8);
                static constexpr uint8_t pcm_guid[16] = {1,    0, 0, 0,    0, 0,    0x10, 0,
                                                         0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71};
                if (cb_size < 22 || cb_size > chunk_size - 18 || valid_bits != 16 || bits_per_sample != 16 ||
                    n_channels <= 0 || block_align != uint32_t(n_channels) * 2 ||
                    std::memcmp(extension + 8, pcm_guid, sizeof(pcm_guid)) != 0) {
                    break;
                }
                audio_format = 1;
                consumed = 40;
            }
            // RIFF pads every odd-sized chunk to an even byte boundary; the
            // pad is not included in chunk_size, including for fmt extensions.
            if (!skip((uint64_t)chunk_size - consumed + (chunk_size & 1))) {
                break;
            }
            found_fmt = true;
        } else if (std::memcmp(chunk_id, "data", 4) == 0) {
            data_size = chunk_size;
            found_data = true;
            break;
        } else {
            if (!skip((uint64_t)chunk_size + (chunk_size & 1))) {
                break;
            }
        }
    }

    if (!found_fmt || !found_data || audio_format != 1 || bits_per_sample != 16 || n_channels <= 0 || sr <= 0) {
        std::fclose(f);
        return false;
    }

    // Clamp the untrusted `data` chunk size to the bytes the file actually
    // holds from the current position, so a header declaring a huge data_size
    // cannot drive a ~2 GB allocation from a tiny file (mirrors the AU reader).
    const int64_t data_pos = tell();
    if (data_pos < 0 || data_pos > file_end) {
        std::fclose(f);
        return false;
    }
    if ((uint64_t)data_size > (uint64_t)(file_end - data_pos)) {
        data_size = (uint32_t)(file_end - data_pos);
    }

    sample_rate = sr;
    int n_samples_total = data_size / (n_channels * (bits_per_sample / 8));
    if (n_samples_total <= 0) {
        std::fclose(f);
        pcm.clear();
        return true;
    }

    std::vector<int16_t> raw((size_t)n_samples_total * n_channels);
    size_t read_count = std::fread(raw.data(), sizeof(int16_t), raw.size(), f);
    std::fclose(f);

    int n_read = (int)(read_count / n_channels);
    pcm.resize((size_t)n_read);
    for (int i = 0; i < n_read; i++) {
        float sum = 0.0f;
        for (int ch = 0; ch < n_channels; ch++) {
            sum += (float)raw[(size_t)i * n_channels + ch] / 32768.0f;
        }
        pcm[i] = sum / n_channels;
    }
    return true;
}

} // namespace crispasr::core
