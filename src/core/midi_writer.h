// core/midi_writer.h — Standard MIDI File (format 1) writer for note events.
//
// Three backends produce note events — piano-transcription (88-key piano),
// basic-pitch (polyphonic, any instrument) and mt3 (multi-instrument with
// General MIDI programs) — and until now all three could only print them as
// text or JSON. §250's CLI spec has always said "→ MIDI output file"; without
// one the events cannot be opened in a DAW or notation editor, which is the
// entire point of transcribing to notes.
//
// Header-only, no dependencies: an SMF is a byte format, and pulling a MIDI
// library in for ~200 lines of well-specified serialisation would be worse.
//
// Layout: format 1, one tempo track plus one track per distinct
// (program, is_drum) pair — the shape notation software expects, and the only
// way MT3's per-instrument output survives the round trip. Drums always land
// on channel 9 (GM percussion); everything else round-robins the remaining 15
// channels, so more than 15 concurrent programs reuse channels rather than
// dropping notes.
//
// Timing: SMF stores ticks, so a tempo must be chosen. 120 BPM at 480 PPQ
// gives 1 ms resolution (960 ticks/s) — finer than any of these models'
// frame rates (100 fps piano, 86 fps basic-pitch, 100 steps/s MT3), so
// quantisation never loses an onset that the model could resolve.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace core_midi {

struct Note {
    double start_s = 0.0;
    double end_s = 0.0;
    int midi = 60;     // 0-127
    int velocity = 80; // 1-127 (0 would be a note-off)
    int program = 0;   // General MIDI program 0-127
    bool is_drum = false;
};

struct Options {
    int ppq = 480; // ticks per quarter note
    double tempo_bpm = 120.0;
    // Minimum sounding length. A zero- or negative-length note is legal in the
    // model's output (a detection whose offset collapsed onto its onset) but
    // produces a note-off at or before its note-on, which players render as a
    // stuck or silent note.
    double min_note_s = 0.005;
};

namespace detail {

inline void put_u32be(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back((uint8_t)(v >> 24));
    out.push_back((uint8_t)(v >> 16));
    out.push_back((uint8_t)(v >> 8));
    out.push_back((uint8_t)v);
}

inline void put_u16be(std::vector<uint8_t>& out, uint16_t v) {
    out.push_back((uint8_t)(v >> 8));
    out.push_back((uint8_t)v);
}

// MIDI variable-length quantity: 7 bits per byte, high bit set on all but the
// last. Values are delta-times, which the spec caps at 0x0FFFFFFF.
inline void put_vlq(std::vector<uint8_t>& out, uint32_t v) {
    uint8_t buf[5];
    int n = 0;
    buf[n++] = (uint8_t)(v & 0x7F);
    while ((v >>= 7) != 0)
        buf[n++] = (uint8_t)((v & 0x7F) | 0x80);
    while (n > 0)
        out.push_back(buf[--n]);
}

inline void put_chunk(std::vector<uint8_t>& out, const char id[4], const std::vector<uint8_t>& body) {
    out.insert(out.end(), id, id + 4);
    put_u32be(out, (uint32_t)body.size());
    out.insert(out.end(), body.begin(), body.end());
}

struct Ev {
    uint32_t tick;
    int rank; // note-offs (0) before program changes (1) before note-ons (2)
    std::vector<uint8_t> bytes;
};

inline bool ev_less(const Ev& a, const Ev& b) {
    if (a.tick != b.tick)
        return a.tick < b.tick;
    return a.rank < b.rank;
}

inline int clamp7(int v) {
    return v < 0 ? 0 : (v > 127 ? 127 : v);
}

} // namespace detail

// Serialise `notes` into a format-1 SMF byte buffer. Returns false only when
// there is nothing to write.
inline bool build_smf(const std::vector<Note>& notes, std::vector<uint8_t>& out, const Options& opt = Options()) {
    out.clear();
    if (notes.empty())
        return false;

    const double ticks_per_s = (double)opt.ppq * opt.tempo_bpm / 60.0;
    auto to_tick = [&](double t) -> uint32_t {
        if (t <= 0.0)
            return 0;
        const double x = std::floor(t * ticks_per_s + 0.5);
        return (uint32_t)(x < 0.0 ? 0.0 : x);
    };

    // Group by (program, is_drum); drums share one track regardless of program.
    std::map<std::pair<int, bool>, std::vector<const Note*>> groups;
    for (const Note& n : notes)
        groups[{n.is_drum ? 0 : detail::clamp7(n.program), n.is_drum}].push_back(&n);

    std::vector<std::vector<uint8_t>> tracks;

    // Track 0: tempo map. 500000 us/quarter == 120 BPM.
    {
        std::vector<uint8_t> t;
        const uint32_t us_per_quarter = (uint32_t)std::llround(60000000.0 / opt.tempo_bpm);
        detail::put_vlq(t, 0);
        t.push_back(0xFF);
        t.push_back(0x51);
        t.push_back(0x03);
        t.push_back((uint8_t)(us_per_quarter >> 16));
        t.push_back((uint8_t)(us_per_quarter >> 8));
        t.push_back((uint8_t)us_per_quarter);
        detail::put_vlq(t, 0);
        t.push_back(0xFF);
        t.push_back(0x2F);
        t.push_back(0x00);
        tracks.push_back(std::move(t));
    }

    int next_ch = 0; // round-robin over non-percussion channels
    for (const auto& kv : groups) {
        const int program = kv.first.first;
        const bool is_drum = kv.first.second;
        int channel;
        if (is_drum) {
            channel = 9;
        } else {
            channel = next_ch;
            next_ch = (next_ch + 1) % 16;
            if (next_ch == 9) // 9 is reserved for percussion
                next_ch = 10;
            if (channel == 9)
                channel = 10;
        }

        std::vector<detail::Ev> evs;
        // A program change even for drums keeps players from inheriting a
        // stale patch from a previous track.
        evs.push_back({0, 1, {(uint8_t)(0xC0 | channel), (uint8_t)program}});

        for (const Note* n : kv.second) {
            const int note = detail::clamp7(n->midi);
            const int vel = std::max(1, detail::clamp7(n->velocity));
            const double end_s = std::max(n->end_s, n->start_s + opt.min_note_s);
            const uint32_t on = to_tick(n->start_s);
            uint32_t off = to_tick(end_s);
            if (off <= on)
                off = on + 1; // never emit a zero-length note
            evs.push_back({on, 2, {(uint8_t)(0x90 | channel), (uint8_t)note, (uint8_t)vel}});
            evs.push_back({off, 0, {(uint8_t)(0x80 | channel), (uint8_t)note, 0}});
        }

        std::stable_sort(evs.begin(), evs.end(), detail::ev_less);

        std::vector<uint8_t> t;
        uint32_t prev = 0;
        for (const detail::Ev& e : evs) {
            detail::put_vlq(t, e.tick - prev);
            prev = e.tick;
            t.insert(t.end(), e.bytes.begin(), e.bytes.end());
        }
        detail::put_vlq(t, 0);
        t.push_back(0xFF);
        t.push_back(0x2F);
        t.push_back(0x00);
        tracks.push_back(std::move(t));
    }

    std::vector<uint8_t> head;
    detail::put_u16be(head, 1);                       // format 1
    detail::put_u16be(head, (uint16_t)tracks.size()); // ntrks
    detail::put_u16be(head, (uint16_t)opt.ppq);       // division
    detail::put_chunk(out, "MThd", head);
    for (const auto& t : tracks)
        detail::put_chunk(out, "MTrk", t);
    return true;
}

// Convenience: build and write to `path`.
inline bool write_smf(const std::string& path, const std::vector<Note>& notes, const Options& opt = Options()) {
    std::vector<uint8_t> bytes;
    if (!build_smf(notes, bytes, opt))
        return false;
    FILE* f = fopen(path.c_str(), "wb");
    if (!f)
        return false;
    const size_t n = fwrite(bytes.data(), 1, bytes.size(), f);
    fclose(f);
    return n == bytes.size();
}

} // namespace core_midi
