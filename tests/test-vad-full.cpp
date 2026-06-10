#include "crispasr.h"
#include "common-crispasr.h"

#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <string>
#include <cstring>

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>

static std::string env_or(const char* name, const char* fallback) {
    const char* v = std::getenv(name);
    return (v && *v) ? v : fallback;
}

int main() {
    std::string whisper_model_path = env_or("CRISPASR_MODEL_WHISPER", CRISPASR_MODEL_PATH);
    std::string vad_model_path = env_or("CRISPASR_VAD_MODEL", VAD_MODEL_PATH);
    std::string sample_path = env_or("CRISPASR_AUDIO_EN", SAMPLE_PATH);

    // Load the sample audio file
    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;
    assert(read_audio_data(sample_path.c_str(), pcmf32, pcmf32s, false));

    struct whisper_context_params cparams = whisper_context_default_params();
    struct whisper_context* wctx = whisper_init_from_file_with_params(whisper_model_path.c_str(), cparams);
    assert(wctx != nullptr);

    struct whisper_full_params wparams = whisper_full_default_params(CRISPASR_SAMPLING_BEAM_SEARCH);
    wparams.vad = true;
    wparams.vad_model_path = vad_model_path.c_str();

    wparams.vad_params.threshold = 0.5f;
    wparams.vad_params.min_speech_duration_ms = 250;
    wparams.vad_params.min_silence_duration_ms = 100;
    wparams.vad_params.max_speech_duration_s = FLT_MAX;
    wparams.vad_params.speech_pad_ms = 30;

    assert(whisper_full_parallel(wctx, wparams, pcmf32.data(), pcmf32.size(), 1) == 0);

    const int n_segments = whisper_full_n_segments(wctx);
    assert(n_segments >= 1);

    // Collect all segment text and check for key JFK phrases. Both
    // whisper-tiny and whisper-tiny.en produce recognisable transcripts
    // but the exact wording may differ.
    std::string all_text;
    for (int i = 0; i < n_segments; i++) {
        all_text += whisper_full_get_segment_text(wctx, i);
    }
    fprintf(stderr, "VAD full transcript: %s\n", all_text.c_str());

    // Case-insensitive substring check for the key phrase
    std::string lower = all_text;
    for (auto& c : lower)
        c = (char)tolower((unsigned char)c);
    assert(lower.find("fellow") != std::string::npos);
    assert(lower.find("country") != std::string::npos);

    whisper_free(wctx);

    return 0;
}
