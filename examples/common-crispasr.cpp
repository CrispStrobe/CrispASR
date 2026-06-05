#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

#include "common-crispasr.h"

#include "common.h"

#include "crispasr.h"

// third-party utilities
// use your favorite implementations
#define STB_VORBIS_HEADER_ONLY
#include "stb_vorbis.c"    /* Enables Vorbis decoding. */

#ifdef _WIN32
#ifndef NOMINMAX
    #define NOMINMAX
#endif
#endif

// `src/crispasr_audio.cpp` already defines `MINIAUDIO_IMPLEMENTATION`
// for the crispasr library (full version, with device IO for mic
// capture). Defining it here too produced duplicate `ma_*` symbols
// in both `common.lib` and `crispasr.lib`, causing MSVC LNK2005 on
// crispasr-cli (release.yml `build-windows-cpu` / `-legacy`) and
// MinGW-style multiple-definition errors that the previous
// `--allow-multiple-definition` workaround patched (0f83a731) — but
// the MSVC linker isn't covered by that hack. Drop the implementation
// here; consumers of `common-crispasr.cpp` already link against
// `crispasr` (which carries the symbols), so `ma_decoder_*` etc.
// resolve at executable-link time. Header-only include keeps the
// declarations visible.
#include "miniaudio.h"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include <cstdio>
#include <cstring>
#include <fstream>

#ifdef CRISPASR_FFMPEG
// as implemented in ffmpeg_trancode.cpp only embedded in common lib if whisper built with ffmpeg support
extern bool ffmpeg_decode_audio(const std::string & ifname, std::vector<uint8_t> & wav_data);
#endif

// Decode any audio container that miniaudio can't handle (m4a/mp4/webm/aac/opus)
// by piping through an ffmpeg subprocess, producing raw 16kHz mono s16le PCM.
//
// On Windows the path may be UTF-8 with non-ASCII characters; the
// crispasr_popen wrapper widens the command to wchar_t and uses
// _wpopen so the path survives the CRT/cmd.exe round-trip (same
// fix applied to the live mic subprocess — see issue #70 follow-up).
#include "cli/crispasr_popen.h"
static bool ffmpeg_subprocess_decode(const std::string& fname, std::vector<float>& pcmf32) {
    // Quote the path for the shell command — basic protection for spaces, no full shell escaping
    std::string cmd = "ffmpeg -loglevel error -i \"" + fname + "\" -f s16le -ar 16000 -ac 1 -";
    FILE* pipe = crispasr::crispasr_popen(cmd, "rb");
    if (!pipe) return false;

    std::vector<int16_t> buf;
    int16_t tmp[4096];
    size_t n;
    while ((n = fread(tmp, sizeof(int16_t), 4096, pipe)) > 0) {
        buf.insert(buf.end(), tmp, tmp + n);
    }
    int ret = crispasr::crispasr_pclose(pipe);
    if (ret != 0 || buf.empty()) return false;

    pcmf32.resize(buf.size());
    for (size_t i = 0; i < buf.size(); i++)
        pcmf32[i] = (float)buf[i] / 32768.0f;
    return true;
}

bool read_audio_data(const std::string & fname, std::vector<float>& pcmf32, std::vector<std::vector<float>>& pcmf32s, bool stereo) {
    std::vector<uint8_t> audio_data; // used for pipe input from stdin or ffmpeg decoding output

    ma_result result;
    ma_decoder_config decoder_config;
    ma_decoder decoder;

    decoder_config = ma_decoder_config_init(ma_format_f32, stereo ? 2 : 1, CRISPASR_SAMPLE_RATE);

    if (fname == "-") {
		#ifdef _WIN32
		_setmode(_fileno(stdin), _O_BINARY);
		#endif

		uint8_t buf[1024];
		while (true)
		{
			const size_t n = fread(buf, 1, sizeof(buf), stdin);
			if (n == 0) {
				break;
			}
			audio_data.insert(audio_data.end(), buf, buf + n);
		}

		if ((result = ma_decoder_init_memory(audio_data.data(), audio_data.size(), &decoder_config, &decoder)) != MA_SUCCESS) {

			fprintf(stderr, "Error: failed to open audio data from stdin (%s)\n", ma_result_description(result));

			return false;
		}

		fprintf(stderr, "%s: read %zu bytes from stdin\n", __func__, audio_data.size());
    }
    else if (((result = ma_decoder_init_file(fname.c_str(), &decoder_config, &decoder)) != MA_SUCCESS)) {
#if defined(CRISPASR_FFMPEG)
		if (ffmpeg_decode_audio(fname, audio_data) != 0) {
			fprintf(stderr, "error: failed to ffmpeg decode '%s'\n", fname.c_str());

			return false;
		}

		if ((result = ma_decoder_init_memory(audio_data.data(), audio_data.size(), &decoder_config, &decoder)) != MA_SUCCESS) {
			fprintf(stderr, "error: failed to read audio data as wav (%s)\n", ma_result_description(result));

			return false;
		}
#else
		// Fallback: try ffmpeg subprocess (handles m4a, mp4, webm, aac, opus, etc.)
		if (ffmpeg_subprocess_decode(fname, pcmf32)) {
			// ffmpeg already produced mono 16kHz float PCM; skip the miniaudio path
			if (stereo) {
				// duplicate mono channel into both stereo channels
				pcmf32s.resize(2, std::vector<float>(pcmf32.size()));
				pcmf32s[0] = pcmf32;
				pcmf32s[1] = pcmf32;
			}
			return true;
		}
		fprintf(stderr, "error: failed to read audio '%s': miniaudio and ffmpeg both failed\n"
		                "       Install ffmpeg or convert to wav/mp3/flac first.\n", fname.c_str());
		return false;
#endif
    }

    ma_uint64 frame_count;
    ma_uint64 frames_read;

    if ((result = ma_decoder_get_length_in_pcm_frames(&decoder, &frame_count)) != MA_SUCCESS) {
		fprintf(stderr, "error: failed to retrieve the length of the audio data (%s)\n", ma_result_description(result));

		return false;
    }

    pcmf32.resize(stereo ? frame_count*2 : frame_count);

    if ((result = ma_decoder_read_pcm_frames(&decoder, pcmf32.data(), frame_count, &frames_read)) != MA_SUCCESS) {
		fprintf(stderr, "error: failed to read the frames of the audio data (%s)\n", ma_result_description(result));

		return false;
    }

    if (stereo) {
        std::vector<float> stereo_data = pcmf32;
        pcmf32.resize(frame_count);

        for (uint64_t i = 0; i < frame_count; i++) {
            pcmf32[i] = 0.5f * (stereo_data[2*i] + stereo_data[2*i + 1]);
        }

        pcmf32s.resize(2);
        pcmf32s[0].resize(frame_count);
        pcmf32s[1].resize(frame_count);
        for (uint64_t i = 0; i < frame_count; i++) {
            pcmf32s[0][i] = stereo_data[2*i];
            pcmf32s[1][i] = stereo_data[2*i + 1];
        }
    }

    ma_decoder_uninit(&decoder);

    return true;
}

//  500 -> 00:05.000
// 6000 -> 01:00.000
std::string to_timestamp(int64_t t, bool comma) {
    int64_t msec = t * 10;
    int64_t hr = msec / (1000 * 60 * 60);
    msec = msec - hr * (1000 * 60 * 60);
    int64_t min = msec / (1000 * 60);
    msec = msec - min * (1000 * 60);
    int64_t sec = msec / 1000;
    msec = msec - sec * 1000;

    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d%s%03d", (int) hr, (int) min, (int) sec, comma ? "," : ".", (int) msec);

    return std::string(buf);
}

int timestamp_to_sample(int64_t t, int n_samples, int whisper_sample_rate) {
    return std::max(0, std::min((int) n_samples - 1, (int) ((t*whisper_sample_rate)/100)));
}

bool speak_with_file(const std::string & command, const std::string & text, const std::string & path, int voice_id) {
#if defined(__APPLE__) && (TARGET_OS_IPHONE || TARGET_OS_TV || TARGET_OS_WATCH)
    // system() is unavailable on iOS / tvOS / watchOS.
    (void) command; (void) text; (void) path; (void) voice_id;
    fprintf(stderr, "%s: not supported on this platform (system() unavailable)\n", __func__);
    return false;
#else
    std::ofstream speak_file(path.c_str());
    if (speak_file.fail()) {
        fprintf(stderr, "%s: failed to open speak_file\n", __func__);
        return false;
    } else {
        speak_file.write(text.c_str(), text.size());
        speak_file.close();
        int ret = system((command + " " + std::to_string(voice_id) + " " + path).c_str());
        if (ret != 0) {
            fprintf(stderr, "%s: failed to speak\n", __func__);
            return false;
        }
    }
    return true;
#endif
}

// Previously expanded the stb_vorbis.c implementation here so
// miniaudio's Vorbis decoder (compiled in via MINIAUDIO_IMPLEMENTATION
// above) had its underlying symbols. Now that
// MINIAUDIO_IMPLEMENTATION is gone (delegated to
// src/crispasr_audio.cpp), the Vorbis backend isn't compiled into
// common-crispasr.cpp at all, and re-expanding stb_vorbis.c here
// produced ld multiple-definition errors against the same set of
// symbols already provided by crispasr_audio.cpp on Linux/macOS
// (Release / build-linux + build-macos failures on v0.6.1 release.yml).
// crispasr_audio.cpp's `#undef STB_VORBIS_HEADER_ONLY; #include
// "../examples/stb_vorbis.c"` block remains the single owner.
