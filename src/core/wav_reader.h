// src/core/wav_reader.h — minimal WAV reader for CLI / backend adapters.
//
// Several backend adapters (indextts, voxcpm2-tts) need to load a reference
// WAV passed via `--voice <path>`. Each had its own copy of the same RIFF
// parser; this header is the single source of truth.
//
// Scope: PCM16 mono or multi-channel only (multi-channel is averaged to mono).
// Returns native sample rate as-read; callers resample via
// `crispasr::core::resample_polyphase` (see core/audio_resample.h) if they
// need a specific rate.
//
// Header-only so callers can drop in by including this file alone.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

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

    char riff[4];
    if (std::fread(riff, 1, 4, f) != 4 || std::memcmp(riff, "RIFF", 4) != 0) {
        std::fclose(f);
        return false;
    }
    std::fseek(f, 4, SEEK_CUR); // skip chunk size

    char wave[4];
    if (std::fread(wave, 1, 4, f) != 4 || std::memcmp(wave, "WAVE", 4) != 0) {
        std::fclose(f);
        return false;
    }

    int16_t audio_format = 0;
    int16_t n_channels = 0;
    int32_t sr = 0;
    int16_t bits_per_sample = 0;
    bool found_fmt = false, found_data = false;
    int32_t data_size = 0;

    while (!std::feof(f)) {
        char chunk_id[4];
        int32_t chunk_size = 0;
        if (std::fread(chunk_id, 1, 4, f) != 4) {
            break;
        }
        if (std::fread(&chunk_size, 4, 1, f) != 1) {
            break;
        }

        if (std::memcmp(chunk_id, "fmt ", 4) == 0) {
            if (std::fread(&audio_format, 2, 1, f) != 1) {
                break;
            }
            if (std::fread(&n_channels, 2, 1, f) != 1) {
                break;
            }
            if (std::fread(&sr, 4, 1, f) != 1) {
                break;
            }
            std::fseek(f, 6, SEEK_CUR); // byte_rate + block_align
            if (std::fread(&bits_per_sample, 2, 1, f) != 1) {
                break;
            }
            if (chunk_size > 16) {
                std::fseek(f, chunk_size - 16, SEEK_CUR);
            }
            found_fmt = true;
        } else if (std::memcmp(chunk_id, "data", 4) == 0) {
            data_size = chunk_size;
            found_data = true;
            break;
        } else {
            std::fseek(f, chunk_size, SEEK_CUR);
        }
    }

    if (!found_fmt || !found_data || audio_format != 1 || bits_per_sample != 16 || n_channels <= 0) {
        std::fclose(f);
        return false;
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
