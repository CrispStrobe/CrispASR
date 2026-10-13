// crispasr_tts_ref_text.h — resolve the reference transcript for a voice clone.
//
// Several cloning backends need a transcript of the reference clip, not just
// its audio: F5-TTS uses it to estimate the speech rate that sets the output
// duration, and CosyVoice3's talker is conditioned on the (transcript, speech)
// pair and infers the speaker's rate from it. In both cases a transcript that
// does not match the clip is worse than useless — for CosyVoice3 it makes the
// decode stop at once or rush the requested line into far too few frames
// (issue #334).
//
// The reliable transcript is the one we produce ourselves, so when `--ref-text`
// is omitted, transcribe the reference with an ASR backend (`--ref-asr`,
// default whisper; the model is resolved through the registry and
// auto-downloaded). The result is stable per clip, so it is cached beside the
// voice file and later runs skip ASR entirely.
//
// Header-only, CLI-side: it constructs another CrispasrBackend, which the
// session C-ABI cannot do. That matches where F5-TTS already did this, so both
// the CLI and the server (which drives the same adapters) get it, while the
// session ABI keeps requiring an explicit transcript.

#pragma once

#include "crispasr_backend.h"
#include "crispasr_model_mgr_cli.h"
#include "whisper_params.h"

#include "core/tts_ref_cache.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace crispasr_ref_text {

// The transcript stored beside a reference clip, <voice>.wav -> <voice>.txt,
// which is where the server's POST /v1/voices writes its `transcript` field.
// Returns "" when there is none.
inline std::string sidecar(const std::string& voice_path) {
    std::ifstream f(std::filesystem::path(voice_path).replace_extension(".txt"));
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string text = ss.str();
    const size_t end = text.find_last_not_of(" \t\r\n");
    return end == std::string::npos ? "" : text.substr(0, end + 1);
}

// Transcribe 16 kHz mono PCM with `asr_backend`. `log_tag` is the caller's
// "crispasr[<backend>]" prefix so diagnostics name the backend the user asked
// for, not this helper. Returns "" on any failure (never throws) — callers
// decide whether an empty transcript is fatal.
inline std::string transcribe(const std::vector<float>& pcm_16k, const whisper_params& p,
                              const std::string& asr_backend, const char* log_tag) {
    auto backend = crispasr_create_backend(asr_backend);
    if (!backend) {
        if (!p.no_prints)
            fprintf(stderr, "%s: unknown ASR backend '%s' for ref-text transcription\n", log_tag, asr_backend.c_str());
        return "";
    }

    // Minimal params for the ASR backend — model + threads. The registry
    // resolves "auto" and downloads if needed.
    whisper_params asr_p = {};
    asr_p.n_threads = p.n_threads;
    asr_p.no_prints = true; // suppress ASR model load chatter
    asr_p.language = p.language.empty() ? "en" : p.language;
    asr_p.auto_download = true;
    asr_p.model = crispasr_resolve_model_cli("auto", asr_backend, /*quiet=*/true,
                                             /*cache_dir=*/"", /*auto_download=*/true);

    if (!backend->init(asr_p)) {
        if (!p.no_prints)
            fprintf(stderr, "%s: failed to init '%s' for ref-text transcription\n", log_tag, asr_backend.c_str());
        return "";
    }

    auto segments = backend->transcribe(pcm_16k.data(), (int)pcm_16k.size(), 0, asr_p);
    backend->shutdown();

    std::string result;
    for (const auto& seg : segments) {
        if (!result.empty())
            result += " ";
        result += seg.text;
    }

    const size_t start = result.find_first_not_of(" \t\n\r");
    const size_t end = result.find_last_not_of(" \t\n\r");
    if (start == std::string::npos)
        return "";
    return result.substr(start, end - start + 1);
}

// Cache-aware form: look beside `voice_path` for a previously transcribed
// reference, else transcribe and store it. `cache_suffix` keeps backends from
// reading each other's entries (their ASR settings and expectations differ).
// Disable the cache with CRISPASR_TTS_REF_CACHE=0.
inline std::string resolve_cached(const std::string& voice_path, const std::vector<float>& pcm_16k,
                                  const whisper_params& p, const char* log_tag, const char* cache_suffix) {
    const std::string cache_path = crispasr_ref_cache::path_for(voice_path, cache_suffix);
    const bool cache_enabled = !crispasr_ref_cache::disabled();

    if (cache_enabled) {
        std::vector<uint32_t> shape;
        std::vector<uint8_t> payload;
        if (crispasr_ref_cache::load(cache_path, voice_path, cache_suffix, shape, payload)) {
            std::string cached((const char*)payload.data(), payload.size());
            if (!p.no_prints)
                fprintf(stderr, "%s: using cached ref transcript '%s': '%s'\n", log_tag, cache_path.c_str(),
                        cached.c_str());
            return cached;
        }
    }

    const std::string asr_name = p.tts_ref_asr.empty() ? std::string("whisper") : p.tts_ref_asr;
    if (!p.no_prints)
        fprintf(stderr, "%s: --ref-text not set, auto-transcribing the reference via %s...\n", log_tag,
                asr_name.c_str());

    const std::string text = transcribe(pcm_16k, p, asr_name, log_tag);
    if (!p.no_prints && !text.empty())
        fprintf(stderr, "%s: auto-transcribed ref: '%s'\n", log_tag, text.c_str());
    if (!text.empty() && cache_enabled)
        crispasr_ref_cache::save(cache_path, cache_suffix, {(uint32_t)text.size()}, text.data(), text.size());
    return text;
}

} // namespace crispasr_ref_text
