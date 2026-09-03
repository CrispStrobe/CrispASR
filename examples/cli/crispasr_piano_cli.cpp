// crispasr_piano_cli.cpp — see crispasr_piano_cli.h.

#include "crispasr_piano_cli.h"

#include "common-crispasr.h" // read_audio_data
#include "crispasr_model_mgr_cli.h"
#include "whisper_params.h"

#include "basic_pitch.h"
#include "mt3.h"
#include "core/gguf_loader.h" // core_gguf::open_metadata / kv_str
#include "core/midi_writer.h" // --piano-format midi
#include "piano_transcription.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

// MIDI note -> name. Sharps only, matching the convention the chord vocabulary
// already uses (btc_chord_vocab.h), so the two music surfaces agree.
std::string midi_note_name(int midi) {
    static const char* kNames[12] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
    if (midi < 0)
        return "?";
    const int pc = midi % 12;
    const int octave = midi / 12 - 1; // MIDI 60 = C4
    return std::string(kNames[pc]) + std::to_string(octave);
}

// One line per note, tab-separated and greppable:
// "onset_sec  offset_sec  midi  name  velocity".
// Deliberately NOT the .lab shape used by --chords: a chord timeline is
// contiguous non-overlapping spans, whereas piano notes overlap freely, so
// reusing that layout would imply a structure the data does not have.
void print_notes_text(const piano_transcription_result& r) {
    for (int i = 0; i < r.n_notes; i++) {
        const piano_note_event& n = r.note_events[i];
        printf("%.3f\t%.3f\t%d\t%s\t%d\n", n.onset_time, n.offset_time, n.midi_note,
               midi_note_name(n.midi_note).c_str(), n.velocity);
    }
}

// --piano-format midi. Note events are only useful in a DAW or notation
// editor, and text/json cannot get them there; PLAN §250's CLI spec has always
// said "-> MIDI output file". Path: `-of NAME` -> NAME.mid, else the input
// path with its extension replaced, so a batch of inputs cannot collide.
std::string midi_out_path(const whisper_params& params, const std::string& fname, size_t idx) {
    // -of is per-input (a vector), same convention as -osrt/-otxt in
    // crispasr_run.cpp: honour the matching entry, fall back to the input path.
    if (idx < params.fname_out.size() && !params.fname_out[idx].empty())
        return params.fname_out[idx] + ".mid";
    const size_t slash = fname.find_last_of("/\\");
    const size_t dot = fname.find_last_of('.');
    const bool has_ext = dot != std::string::npos && (slash == std::string::npos || dot > slash);
    return (has_ext ? fname.substr(0, dot) : fname) + ".mid";
}

bool write_midi(const std::vector<core_midi::Note>& notes, const whisper_params& params, const std::string& fname,
                size_t idx) {
    const std::string out = midi_out_path(params, fname, idx);
    if (!core_midi::write_smf(out, notes)) {
        fprintf(stderr, "crispasr: --piano: failed to write '%s'\n", out.c_str());
        return false;
    }
    if (!params.no_prints)
        fprintf(stderr, "crispasr: wrote %s (%zu notes)\n", out.c_str(), notes.size());
    return true;
}

void print_notes_json(const piano_transcription_result& r, const std::string& fname) {
    printf("{\n");
    printf("  \"file\": \"%s\",\n", fname.c_str());
    printf("  \"n_notes\": %d,\n", r.n_notes);
    printf("  \"n_pedals\": %d,\n", r.n_pedals);
    printf("  \"notes\": [\n");
    for (int i = 0; i < r.n_notes; i++) {
        const piano_note_event& n = r.note_events[i];
        printf("    {\"onset\": %.3f, \"offset\": %.3f, \"midi\": %d, \"name\": \"%s\", \"velocity\": %d}%s\n",
               n.onset_time, n.offset_time, n.midi_note, midi_note_name(n.midi_note).c_str(), n.velocity,
               i + 1 < r.n_notes ? "," : "");
    }
    printf("  ],\n");
    printf("  \"pedals\": [\n");
    for (int i = 0; i < r.n_pedals; i++) {
        const piano_pedal_event& p = r.pedal_events[i];
        printf("    {\"onset\": %.3f, \"offset\": %.3f}%s\n", p.onset_time, p.offset_time,
               i + 1 < r.n_pedals ? "," : "");
    }
    printf("  ]\n}\n");
}

// ── Basic Pitch (§250) ──────────────────────────────────────────────────────
//
// A second model behind the same --piano verb. It is not piano-specific — it is
// polyphonic and instrument-agnostic — but the TASK is identical (audio → note
// events), so it shares the dispatcher and the output shape rather than growing
// a near-duplicate verb. What it does not have is pedal events or MIDI
// velocity from a trained velocity head: the "velocity" printed here is
// round(127 * mean frame activation), which is what upstream writes into its
// MIDI too.
void print_bp_notes_text(const basic_pitch_result& r) {
    for (int i = 0; i < r.n_notes; i++) {
        const basic_pitch_note_event& n = r.notes[i];
        printf("%.3f\t%.3f\t%d\t%s\t%d\n", n.start_time, n.end_time, n.midi_note, midi_note_name(n.midi_note).c_str(),
               n.velocity);
    }
}

void print_bp_notes_json(const basic_pitch_result& r, const std::string& fname) {
    printf("{\n");
    printf("  \"file\": \"%s\",\n", fname.c_str());
    printf("  \"n_notes\": %d,\n", r.n_notes);
    printf("  \"notes\": [\n");
    for (int i = 0; i < r.n_notes; i++) {
        const basic_pitch_note_event& n = r.notes[i];
        printf("    {\"onset\": %.3f, \"offset\": %.3f, \"midi\": %d, \"name\": \"%s\", "
               "\"velocity\": %d, \"amplitude\": %.4f}%s\n",
               n.start_time, n.end_time, n.midi_note, midi_note_name(n.midi_note).c_str(), n.velocity, n.amplitude,
               i + 1 < r.n_notes ? "," : "");
    }
    printf("  ]\n}\n");
}

// ── MT3 (§250) ──────────────────────────────────────────────────────────────
//
// A third model behind --piano. Same task (audio → note events) and the same
// dispatcher, but MT3 is MULTI-INSTRUMENT: every note carries a General-MIDI
// program, and drum hits are a separate class. The shared note shape has no
// slot for that, so rather than inventing a fourth verb the text form grows one
// trailing tab-separated column ("prog=<n>" or "drum") after the existing five,
// and the JSON form grows "program" / "instrument" / "is_drum" keys. Readers of
// the first five columns are unaffected; nothing else about the surface moves.
// `instrument` is upstream's track index (assign_instruments: programs
// numbered in first-appearance order skipping 9, drums always 9 — GM ch. 10).
void print_mt3_notes_text(const mt3_result& r) {
    for (int i = 0; i < r.n_notes; i++) {
        const mt3_note_event& n = r.notes[i];
        char extra[32];
        if (n.is_drum)
            snprintf(extra, sizeof(extra), "drum");
        else
            snprintf(extra, sizeof(extra), "prog=%d", n.program);
        printf("%.3f\t%.3f\t%d\t%s\t%d\t%s\n", n.start_time, n.end_time, n.pitch, midi_note_name(n.pitch).c_str(),
               n.velocity, extra);
    }
}

void print_mt3_notes_json(const mt3_result& r, const std::string& fname) {
    printf("{\n");
    printf("  \"file\": \"%s\",\n", fname.c_str());
    printf("  \"n_notes\": %d,\n", r.n_notes);
    printf("  \"n_segments\": %d,\n", r.n_segments);
    printf("  \"n_tokens\": %d,\n", r.n_tokens);
    printf("  \"n_invalid\": %d,\n", r.n_invalid);
    printf("  \"n_dropped\": %d,\n", r.n_dropped);
    printf("  \"n_tie_ends\": %d,\n", r.n_tie_ends);
    printf("  \"n_tied_pitches\": %d,\n", r.n_tied_pitches);
    printf("  \"notes\": [\n");
    for (int i = 0; i < r.n_notes; i++) {
        const mt3_note_event& n = r.notes[i];
        printf("    {\"onset\": %.3f, \"offset\": %.3f, \"midi\": %d, \"name\": \"%s\", "
               "\"velocity\": %d, \"program\": %d, \"instrument\": %d, \"is_drum\": %s}%s\n",
               n.start_time, n.end_time, n.pitch, midi_note_name(n.pitch).c_str(), n.velocity, n.program, n.instrument,
               n.is_drum ? "true" : "false", i + 1 < r.n_notes ? "," : "");
    }
    printf("  ]\n}\n");
}

int run_mt3(const whisper_params& params, const std::string& model, bool json, bool midi) {
    mt3_params mp = mt3_default_params();
    mp.n_threads = params.n_threads;
    mp.verbosity = params.no_prints ? 0 : (params.verbose ? 2 : 1);
    mp.use_gpu = params.use_gpu;
    mt3_context* ctx = mt3_init_from_file(model.c_str(), mp);
    if (!ctx) {
        fprintf(stderr, "crispasr: --piano: failed to load '%s'\n", model.c_str());
        return 2;
    }
    const int sr = (int)mt3_sample_rate(ctx);

    int rc = 0;
    size_t midi_idx = 0;
    for (const auto& fname : params.fname_inp) {
        const size_t midi_i = midi_idx++;
        std::vector<float> mono;
        std::vector<std::vector<float>> stereo;
        if (!read_audio_data(fname, mono, stereo, /*stereo=*/false, /*target_rate=*/sr)) {
            fprintf(stderr, "crispasr: error: cannot read '%s'\n", fname.c_str());
            rc = 20;
            continue;
        }
        mt3_result res{};
        if (mt3_transcribe(ctx, mono.data(), (int)mono.size(), &res) != 0) {
            fprintf(stderr, "crispasr: --piano failed on '%s'\n", fname.c_str());
            rc = 1;
            continue;
        }
        if (midi) {
            std::vector<core_midi::Note> mn;
            mn.reserve((size_t)res.n_notes);
            for (int i = 0; i < res.n_notes; i++) {
                const mt3_note_event& e = res.notes[i];
                mn.push_back({e.start_time, e.end_time, e.pitch, e.velocity, e.program, e.is_drum});
            }
            if (!write_midi(mn, params, fname, midi_i))
                rc = 1;
        } else if (json) {
            print_mt3_notes_json(res, fname);
        } else {
            if (!params.no_prints && params.fname_inp.size() > 1)
                printf("# %s\n", fname.c_str());
            print_mt3_notes_text(res);
        }
        if (!params.no_prints)
            fprintf(stderr, "crispasr: %s: %d notes over %d segments\n", fname.c_str(), res.n_notes, res.n_segments);
        mt3_result_free(&res);
    }

    mt3_free(ctx);
    return rc;
}

int run_basic_pitch(const whisper_params& params, const std::string& model, bool json, bool midi) {
    basic_pitch_params bp = basic_pitch_default_params();
    bp.n_threads = params.n_threads;
    bp.verbosity = params.no_prints ? 0 : 1;
    bp.use_gpu = params.use_gpu;
    basic_pitch_ctx* ctx = basic_pitch_init_from_file(model.c_str(), bp);
    if (!ctx) {
        fprintf(stderr, "crispasr: --piano: failed to load '%s'\n", model.c_str());
        return 2;
    }
    const int sr = (int)basic_pitch_sample_rate(ctx);

    int rc = 0;
    size_t midi_idx = 0;
    for (const auto& fname : params.fname_inp) {
        const size_t midi_i = midi_idx++;
        std::vector<float> mono;
        std::vector<std::vector<float>> stereo;
        if (!read_audio_data(fname, mono, stereo, /*stereo=*/false, /*target_rate=*/sr)) {
            fprintf(stderr, "crispasr: error: cannot read '%s'\n", fname.c_str());
            rc = 20;
            continue;
        }
        basic_pitch_result res{};
        if (basic_pitch_transcribe(ctx, mono.data(), (int)mono.size(), &res) != 0) {
            fprintf(stderr, "crispasr: --piano failed on '%s'\n", fname.c_str());
            rc = 1;
            continue;
        }
        if (midi) {
            std::vector<core_midi::Note> mn;
            mn.reserve((size_t)res.n_notes);
            for (int i = 0; i < res.n_notes; i++) {
                const basic_pitch_note_event& e = res.notes[i];
                mn.push_back({e.start_time, e.end_time, e.midi_note, e.velocity, 0, false});
            }
            if (!write_midi(mn, params, fname, midi_i))
                rc = 1;
        } else if (json) {
            print_bp_notes_json(res, fname);
        } else {
            if (!params.no_prints && params.fname_inp.size() > 1)
                printf("# %s\n", fname.c_str());
            print_bp_notes_text(res);
        }
        if (!params.no_prints)
            fprintf(stderr, "crispasr: %s: %d notes\n", fname.c_str(), res.n_notes);
        basic_pitch_result_free(&res);
    }

    basic_pitch_free(ctx);
    return rc;
}

} // namespace

int crispasr_run_piano(const whisper_params& params) {
    if (params.fname_inp.empty()) {
        fprintf(stderr, "crispasr: --piano needs an input file (-f)\n");
        return 2;
    }

    const bool json = params.piano_format == "json";
    const bool midi = params.piano_format == "midi";
    if (!json && !midi && !params.piano_format.empty() && params.piano_format != "text") {
        fprintf(stderr, "crispasr: --piano-format: unknown format '%s' (expected text, json or midi)\n",
                params.piano_format.c_str());
        return 2;
    }

    const std::string backend_key = params.backend.empty() ? "piano-transcription" : params.backend;
    const std::string model = crispasr_resolve_model_cli(params.model, backend_key, params.no_prints, params.cache_dir,
                                                         params.auto_download, "", params.accept_license);
    if (model.empty()) {
        fprintf(stderr, "crispasr: --piano: could not resolve a model.\n");
        return 2;
    }

    gguf_context* meta = core_gguf::open_metadata(model.c_str());
    if (!meta) {
        fprintf(stderr, "crispasr: --piano: cannot open '%s'\n", model.c_str());
        return 2;
    }
    const std::string arch = core_gguf::kv_str(meta, "general.architecture", "");
    core_gguf::free_metadata(meta);
    // Three models answer --piano. Dispatch on the GGUF's own architecture
    // rather than on --backend, so a plain `--piano -m <basic-pitch.gguf>` works.
    if (arch == "basic-pitch" || arch == "basic_pitch")
        return run_basic_pitch(params, model, json, midi);
    if (arch == "mt3")
        return run_mt3(params, model, json, midi);
    if (arch != "piano-transcription" && arch != "piano_transcription") {
        fprintf(stderr, "crispasr: --piano: '%s' is not a note-event model (arch='%s').\n", model.c_str(),
                arch.c_str());
        return 2;
    }

    piano_transcription_params pp = piano_transcription_default_params();
    pp.n_threads = params.n_threads;
    pp.verbosity = params.no_prints ? 0 : 1;
    piano_transcription_ctx* ctx = piano_transcription_init_from_file(model.c_str(), pp);
    if (!ctx) {
        fprintf(stderr, "crispasr: --piano: failed to load '%s'\n", model.c_str());
        return 2;
    }
    const int sr = (int)piano_transcription_sample_rate(ctx);

    int rc = 0;
    size_t midi_idx = 0;
    for (const auto& fname : params.fname_inp) {
        const size_t midi_i = midi_idx++;
        std::vector<float> mono;
        std::vector<std::vector<float>> stereo;
        if (!read_audio_data(fname, mono, stereo, /*stereo=*/false, /*target_rate=*/sr)) {
            fprintf(stderr, "crispasr: error: cannot read '%s'\n", fname.c_str());
            rc = 20;
            continue;
        }

        piano_transcription_result res{};
        if (piano_transcription_transcribe(ctx, mono.data(), (int)mono.size(), &res) != 0) {
            fprintf(stderr, "crispasr: --piano failed on '%s'\n", fname.c_str());
            rc = 1;
            continue;
        }

        if (midi) {
            std::vector<core_midi::Note> mn;
            mn.reserve((size_t)res.n_notes);
            for (int i = 0; i < res.n_notes; i++) {
                const piano_note_event& e = res.note_events[i];
                mn.push_back({e.onset_time, e.offset_time, e.midi_note, e.velocity, 0, false});
            }
            if (!write_midi(mn, params, fname, midi_i))
                rc = 1;
        } else if (json) {
            print_notes_json(res, fname);
        } else {
            if (!params.no_prints && params.fname_inp.size() > 1)
                printf("# %s\n", fname.c_str());
            print_notes_text(res);
        }
        if (!params.no_prints)
            fprintf(stderr, "crispasr: %s: %d notes, %d pedal events\n", fname.c_str(), res.n_notes, res.n_pedals);
        piano_transcription_result_free(&res);
    }

    piano_transcription_free(ctx);
    return rc;
}
