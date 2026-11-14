#!/usr/bin/env python3
"""Weight-free production adapter/disclaimer controls for VibeVoice WAV lifetime."""
from pathlib import Path
import os
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def source():
    adapter = (ROOT / 'examples/cli/crispasr_backend_vibevoice.cpp').read_text()
    helpers = adapter[adapter.index('static bool ends_with_ci'):adapter.index('static std::vector<float> resample')]
    method = adapter[adapter.index('    std::vector<float> synthesize('):adapter.index('    void shutdown() override')]
    runtime = (ROOT / 'src/vibevoice.cpp').read_text()
    start = runtime.index('extern "C" void vibevoice_set_reference_wav(')
    setter = runtime[start:runtime.index('\n}', start) + 2]
    start = runtime.index('const char* voice_wav =', runtime.index('// Try loading voice audio'))
    selection = runtime[start:runtime.index(';', start) + 1]
    abi = (ROOT / 'src/crispasr_c_api.cpp').read_text()
    start = abi.index('    if (s->vibevoice_ctx) {', abi.index('CA_EXPORT int crispasr_session_set_voice('))
    abi_selection = abi[start:abi.index('\n#endif', start)]
    disclaimer = (ROOT / 'examples/cli/crispasr_tts_disclaimer.h').read_text()
    disclaimer = disclaimer[disclaimer.index('namespace crispasr_disclaimer'):]
    return r'''
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <vector>
#include "core/crispasr_env.h"
struct whisper_params {
    std::string tts_voice, tts_voice_dir, model, language, tts_instruct;
    int tts_steps = 10, seed = 42, tts_speaker_id = 0;
    float tts_cfg_scale = 1.3;
    bool tts_force_neutral_voice = false;
};
struct vibevoice_context { bool reference_wav_selected = false; std::string reference_wav; };
std::vector<std::string> seen;
void vibevoice_set_tts_steps(vibevoice_context*, int) {}
void vibevoice_set_seed(vibevoice_context*, unsigned) {}
void vibevoice_set_cfg_scale(vibevoice_context*, float) {}
''' + setter + r'''
int vibevoice_load_voice(vibevoice_context*, const char*) { return 0; }
float* vibevoice_synthesize(vibevoice_context* ctx, const char*, int* n) {
    ''' + selection + r'''
    seen.push_back(voice_wav ? voice_wav : "");
    *n = 1; float* result = static_cast<float*>(malloc(sizeof(float))); *result = 0.5f; return result;
}
struct Session { vibevoice_context* vibevoice_ctx; };
bool ends_with_wav(const char* path) { return std::string(path).find(".wav") != std::string::npos; }
int set_voice(Session* s, const char* path) {
''' + abi_selection + r'''
    return -1;
}
struct CrispasrBackend {
    virtual ~CrispasrBackend() = default;
    virtual std::vector<float> synthesize(const std::string&, const whisper_params&) = 0;
    int tts_sample_rate() const { return 24000; }
};
''' + helpers + r'''
struct Adapter : CrispasrBackend {
    vibevoice_context state;
    vibevoice_context* ctx_ = &state;
    std::string last_voice_key_, backend_name_ = "vibevoice-1.5b";
    bool allow_generic_no_voice_ = true;
''' + method + r'''
};
''' + disclaimer + r'''
void require(bool value, const char* message) {
    if (!value) { fprintf(stderr, "FAIL: %s\n", message); exit(1); }
}
int main() {
    setenv("CRISPASR_SUPPRESS_ENV_DEPRECATION", "1", 1);
    setenv("VIBEVOICE_VOICE_AUDIO", "external-legacy.wav", 1);
    setenv("CRISPASR_VIBEVOICE_VOICE_AUDIO", "external-canonical.wav", 1);
    Adapter a, b;
    whisper_params p;
    a.synthesize("ordinary environment fallback", p);
    require(seen.back() == "external-canonical.wav", "ordinary canonical env fallback");
    p.tts_voice = "first.wav";
    a.synthesize("clone", p);
    require(seen.back() == "first.wav", "explicit WAV overrides external canonical env");
    auto audio = std::vector<float>{0.75f};
    crispasr_tts_prepend_disclaimer(audio, &a, p);
    require(seen.back().empty(), "disclaimer synthesis must be neutral");
    require(audio.size() == 7202 && audio.front() == 0.5f && audio.back() == 0.75f,
            "spoken disclosure and 300ms gap remain present");
    a.synthesize("clone again", p);
    require(seen.back() == "first.wav", "clone restored after neutral disclosure");
    p.tts_voice = "second.wav";
    a.synthesize("switch reference", p);
    require(seen.back() == "second.wav", "request switches WAV reference");
    p.tts_voice.clear();
    a.synthesize("ordinary request after clone", p);
    require(seen.back() == "external-canonical.wav", "previous request does not leak into empty request");
    b.synthesize("other context", p);
    require(seen.back() == "external-canonical.wav", "adapter isolation");
    require(std::string(getenv("CRISPASR_VIBEVOICE_VOICE_AUDIO")) == "external-canonical.wav",
            "canonical environment unchanged");
    require(std::string(getenv("VIBEVOICE_VOICE_AUDIO")) == "external-legacy.wav", "legacy environment unchanged");
    unsetenv("CRISPASR_VIBEVOICE_VOICE_AUDIO");
    b.synthesize("legacy fallback", p);
    require(seen.back() == "external-legacy.wav", "ordinary legacy env fallback");
    unsetenv("VIBEVOICE_VOICE_AUDIO");
    a.synthesize("generic request", p);
    require(seen.back().empty(), "empty request clears preceding WAV selection");
    vibevoice_context first, second;
    Session session{&first};
    std::string source_path = "session.wav";
    require(set_voice(&session, source_path.c_str()) == 0, "session selects WAV");
    source_path.assign("changed.wav");
    int n = 0;
    free(vibevoice_synthesize(&first, "session", &n));
    require(seen.back() == "session.wav", "setter owns reference path bytes");
    free(vibevoice_synthesize(&second, "other session", &n));
    require(seen.back().empty(), "session reference does not leak across contexts");
    vibevoice_set_reference_wav(nullptr, "ignored.wav");
    vibevoice_set_reference_wav(&first, "");
    free(vibevoice_synthesize(&first, "neutral", &n));
    require(seen.back().empty(), "empty setter forces neutral");
    setenv("CRISPASR_VIBEVOICE_VOICE_AUDIO", "fallback.wav", 1);
    free(vibevoice_synthesize(&first, "neutral with environment", &n));
    require(seen.back().empty(), "explicit neutral suppresses external environment");
    vibevoice_set_reference_wav(&first, nullptr);
    free(vibevoice_synthesize(&first, "fallback", &n));
    require(seen.back() == "fallback.wav", "null setter restores environment fallback");
    puts("PASS: 18 VibeVoice reference/disclaimer lifetime controls");
}
'''


class ReferenceLifetime(unittest.TestCase):
    def test_adapter_and_disclaimer_sequence(self):
        with tempfile.TemporaryDirectory(prefix='vibevoice-reference-') as tmp:
            path = Path(tmp)
            (path / 'probe.cpp').write_text(source())
            subprocess.run(shlex.split(os.environ.get('CXX', 'c++')) + [
                '-std=c++17', '-Wall', '-Wextra', '-I', str(ROOT / 'src'),
                str(path / 'probe.cpp'), '-o', str(path / 'probe')], check=True)
            subprocess.run([str(path / 'probe')], check=True)


if __name__ == '__main__':
    unittest.main()
