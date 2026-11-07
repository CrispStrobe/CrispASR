// test-voice-dir-reference.cpp — #584: a bare --voice-dir name reaches every
// cloning backend as the clone reference `--voice <path>` would give it.
//
// Before: the server/CLI resolved {"voice": "lanyan"} against --voice-dir only
// for the consent audit, then handed the backend the bare string. cosyvoice3
// looked "lanyan" up in its baked bank and failed ("voice 'lanyan' not found
// (have 8)") on a voice the gate had just approved and hashed.
//
// Model-free: temp files + a fake backend, exercising the exact helper the
// CLI, HTTP server and Wyoming call before dispatch.

#include "crispasr_voice_reference.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <unistd.h>

namespace {

std::string make_temp_dir() {
    std::error_code ec;
    const auto d =
        std::filesystem::temp_directory_path(ec) / ("crispasr-voice-dir-ref-" + std::to_string((long long)getpid()) +
                                                    "-" + std::to_string((long long)std::rand()));
    std::filesystem::create_directories(d, ec);
    return d.string();
}

void write(const std::string& path, const std::string& bytes) {
    std::ofstream f(path, std::ios::binary);
    f << bytes;
}

struct FakeBackend : CrispasrBackend {
    uint32_t caps = CAP_TTS | CAP_VOICE_CLONING;
    bool own = false;
    const char* name() const override { return "fake-tts"; }
    uint32_t capabilities() const override { return caps; }
    bool resolves_voice_dir_itself() const override { return own; }
    bool init(const whisper_params&) override { return true; }
    std::vector<crispasr_segment> transcribe(const float*, int, int64_t, const whisper_params&) override { return {}; }
    void shutdown() override {}
};

} // namespace

TEST_CASE("#584: bare voice-dir name becomes recording + companion transcript", "[unit][compliance]") {
    const std::string dir = make_temp_dir();
    write(dir + "/lanyan.wav", "RIFF");
    // Windows Notepad writes a BOM and CRLF; neither belongs in the transcript.
    write(dir + "/lanyan.txt", "\xEF\xBB\xBF  hello there.\r\n");

    whisper_params p;
    p.tts_voice = "lanyan";
    p.tts_voice_dir = dir;
    FakeBackend b;
    CHECK(crispasr_voice::apply_voice_dir_reference(b, p));
    CHECK(p.tts_voice == dir + "/lanyan.wav");
    CHECK(p.tts_ref_text == "hello there.");
    // Same file the consent gate resolves and hashes.
    CHECK(p.tts_voice == crispasr_voice::resolve_voice_path("lanyan", dir));
    // Idempotent: a second pass over a resolved path is a no-op.
    CHECK_FALSE(crispasr_voice::apply_voice_dir_reference(b, p));
    std::filesystem::remove_all(dir);
}

TEST_CASE("#584: explicit ref_text wins; missing .txt leaves it empty", "[unit]") {
    const std::string dir = make_temp_dir();
    write(dir + "/a.wav", "RIFF");
    write(dir + "/a.txt", "from file");
    auto r = crispasr_voice::resolve_clone_reference("a", dir, "from request");
    CHECK(r.resolved);
    CHECK(r.ref_text == "from request");

    write(dir + "/b.wav", "RIFF");
    r = crispasr_voice::resolve_clone_reference("b", dir, "");
    CHECK(r.resolved);
    CHECK(r.voice == dir + "/b.wav");
    CHECK(r.ref_text.empty()); // backend falls back as for `--voice b.wav`
    std::filesystem::remove_all(dir);
}

TEST_CASE("#584: presets, bank names, .gguf profiles and paths pass through", "[unit][compliance]") {
    const std::string dir = make_temp_dir();
    write(dir + "/packed.gguf", "GGUF");
    write(dir + "/packed.wav", "RIFF"); // .gguf wins in the gate, so stay bare
    const auto keep = [&](const std::string& v) {
        const auto r = crispasr_voice::resolve_clone_reference(v, dir, "");
        CHECK_FALSE(r.resolved);
        CHECK(r.voice == v);
    };
    keep("fleurs-en"); // bank entry / preset: no file in voice-dir
    keep("packed");    // resolves to a .gguf profile — backend-specific format
    keep(dir + "/packed.wav");
    keep("../secrets");
    keep("");
    CHECK_FALSE(crispasr_voice::resolve_clone_reference("packed", "", "").resolved);
    std::filesystem::remove_all(dir);
}

TEST_CASE("#584: only cloning backends without their own resolver are rewritten", "[unit]") {
    const std::string dir = make_temp_dir();
    write(dir + "/v.wav", "RIFF");
    whisper_params p;
    p.tts_voice = "v";
    p.tts_voice_dir = dir;

    FakeBackend own;
    own.own = true; // pocket-tts / qwen3-tts / tada / vibevoice keep their policy
    CHECK_FALSE(crispasr_voice::apply_voice_dir_reference(own, p));
    FakeBackend preset_only;
    preset_only.caps = CAP_TTS;
    CHECK_FALSE(crispasr_voice::apply_voice_dir_reference(preset_only, p));
    CHECK(p.tts_voice == "v");
    std::filesystem::remove_all(dir);
}
