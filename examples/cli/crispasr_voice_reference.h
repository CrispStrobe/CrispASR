// crispasr_voice_reference.h — resolve a bare --voice-dir name into a CLONE
// REFERENCE (recording path + transcript) once, before TTS dispatch. #584.
//
// The server forbids path syntax in the `voice` field, so over HTTP/Wyoming a
// bare name is the ONLY way to pick a custom voice. The consent gate already
// resolved that name against --voice-dir (crispasr_voice_provenance.h) and
// hashed the right file — but the synth call still got the bare string, so a
// backend with no voice-dir code of its own (cosyvoice3: "voice 'lanyan' not
// found (have 8)") failed on a voice the gate had just approved.
//
// This makes `{"voice": "lanyan"}` with <voice-dir>/lanyan.wav (+ lanyan.txt)
// identical to `--voice <voice-dir>/lanyan.wav --ref-text "<lanyan.txt>"` for
// every cloning backend that does not resolve voice-dir names itself (see
// CrispasrBackend::resolves_voice_dir_itself()).
//
// Resolution is deliberately the gate's own resolve_voice_path(), so the file
// synthesised is always the file the [CONSENT] record hashed. It only rewrites
// when that lands on a recording: a .gguf profile is a backend-specific format
// and a name that is no file is a preset/bank entry — both pass through as-is.

#pragma once

#include "crispasr_backend.h"
#include "crispasr_voice_provenance.h"
#include "whisper_params.h"

#include <fstream>
#include <sstream>
#include <string>

namespace crispasr_voice {

struct CloneReference {
    std::string voice;     // what the backend should receive as tts_voice
    std::string ref_text;  // what the backend should receive as tts_ref_text
    bool resolved = false; // a bare name was rewritten to a voice-dir recording
};

// Companion transcript for a voice-dir recording: <stem>.txt beside it, with a
// UTF-8 BOM (Notepad writes one) and surrounding whitespace stripped. Empty
// when absent or unreadable — the backend then falls back exactly as it does
// for a bare `--voice x.wav` with no --ref-text (auto-transcribe or error).
inline std::string read_companion_transcript(const std::string& recording_path) {
    const size_t dot = recording_path.find_last_of('.');
    const size_t sep = recording_path.find_last_of("/\\");
    if (dot == std::string::npos || (sep != std::string::npos && dot < sep))
        return {};
    std::ifstream f(recording_path.substr(0, dot) + ".txt", std::ios::binary);
    if (!f.good())
        return {};
    std::stringstream ss;
    ss << f.rdbuf();
    std::string t = ss.str();
    if (t.size() >= 3 && (unsigned char)t[0] == 0xEF && (unsigned char)t[1] == 0xBB && (unsigned char)t[2] == 0xBF)
        t.erase(0, 3);
    const char* ws = " \t\r\n";
    const size_t b = t.find_first_not_of(ws);
    if (b == std::string::npos)
        return {};
    const size_t e = t.find_last_not_of(ws);
    return t.substr(b, e - b + 1);
}

// Pure-ish core (filesystem reads only): a bare name that resolves to a
// recording in `voice_dir` becomes that path, and an empty `ref_text` is
// filled from the companion .txt. An explicit ref_text always wins. Anything
// else — empty, already a path, a .gguf profile, a preset — is returned as-is.
inline CloneReference resolve_clone_reference(const std::string& voice, const std::string& voice_dir,
                                              const std::string& ref_text) {
    CloneReference r{voice, ref_text, false};
    if (voice.empty() || voice_dir.empty())
        return r;
    const std::string resolved = resolve_voice_path(voice, voice_dir);
    if (resolved == voice || !is_recording_reference(resolved))
        return r;
    r.voice = resolved;
    r.resolved = true;
    if (r.ref_text.empty())
        r.ref_text = read_companion_transcript(resolved);
    return r;
}

// The dispatch step every TTS surface (CLI, HTTP server, Wyoming) calls on its
// params before handing them to the backend. Returns true when it rewrote
// tts_voice. Idempotent: a second call sees a path and does nothing.
inline bool apply_voice_dir_reference(const CrispasrBackend& backend, whisper_params& p) {
    if ((backend.capabilities() & CAP_VOICE_CLONING) == 0 || backend.resolves_voice_dir_itself())
        return false;
    const CloneReference r = resolve_clone_reference(p.tts_voice, p.tts_voice_dir, p.tts_ref_text);
    if (!r.resolved)
        return false;
    p.tts_voice = r.voice;
    p.tts_ref_text = r.ref_text;
    return true;
}

} // namespace crispasr_voice
