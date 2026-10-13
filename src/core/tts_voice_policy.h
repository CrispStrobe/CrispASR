// core/tts_voice_policy.h — per-request voice reference: what to do, and when.
//
// The HTTP server reuses ONE backend instance for the whole session and puts
// each request's `voice=` into params.tts_voice. A backend that only consumes
// tts_voice at init() therefore serves the startup voice forever, whatever the
// caller uploaded (#371). Consuming it at synth time is the fix, but doing so
// naively re-encodes the reference on every request, so the decision has three
// outcomes rather than two.
//
// The semantics here match what the CLI already did with --voice: "default" and
// "auto" are exact-match sentinels meaning "use the built-in reference", and an
// empty voice means the same for f5-tts while omnivoice clears its prompt to
// plain TTS. Which of those two a Builtin result implies is the caller's
// business; the decision of WHETHER to act is what this header owns.
#pragma once

#include <chrono>
#include <filesystem>
#include <string>
#include <system_error>

namespace core_tts_voice {

// Exact-match, case-sensitive, matching the existing CLI behaviour. Not
// normalised on purpose: "Default" is a filename as far as this is concerned,
// and silently treating it as a sentinel would make a real file unreachable.
inline bool is_builtin_request(const std::string& voice) {
    return voice.empty() || voice == "default" || voice == "auto";
}

enum class Action {
    Unchanged, // same voice as last time — skip the re-encode
    Builtin,   // empty/default/auto — built-in reference (or cleared prompt)
    Apply,     // a new reference path — load and encode it
};

// `last` is the voice most recently APPLIED, "" if none.
//
// Note the interaction with failure handling: a caller that fails to load a
// reference must reset `last` to "", otherwise the next identical request
// returns Unchanged and the retry is silently skipped — the request would be
// served with whatever voice happened to be resident. Pinned by the
// "failed load must not dedupe the retry away" case in the tests.
inline Action decide(const std::string& last, const std::string& requested) {
    if (requested == last)
        return Action::Unchanged;
    return is_builtin_request(requested) ? Action::Builtin : Action::Apply;
}

// Identity of a reference file. Missing files use a stable identity so a
// successful retry can be distinguished from a previously absent sidecar.
inline std::string reference_file_key(const std::filesystem::path& path) {
    std::error_code ec;
    const auto mtime = std::filesystem::last_write_time(path, ec);
    if (ec)
        return "missing";
    const auto size = std::filesystem::file_size(path, ec);
    // libc++ can use a 128-bit filesystem clock representation, which has
    // no std::to_string overload. Split seconds and subsecond nanoseconds
    // into standard chrono representations without discarding precision.
    const auto elapsed = mtime.time_since_epoch();
    const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed);
    const auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed - seconds);
    return std::to_string(seconds.count()) + ':' + std::to_string(nanos.count()) + ':' +
           (ec ? "unknown" : std::to_string(size));
}

// Include the transcript sidecar when no explicit transcript overrides it:
// editing <voice>.txt without rewriting the WAV must reload the reference.
inline std::string reference_key(const std::string& path, const std::string& ref_text) {
    std::string key = path + '\x01' + ref_text + '\x01' + reference_file_key(path);
    if (ref_text.empty())
        key += '\x01' + reference_file_key(std::filesystem::path(path).replace_extension(".txt"));
    return key;
}

} // namespace core_tts_voice
