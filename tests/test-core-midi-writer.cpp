// Unit tests for core/midi_writer.h — the SMF writer behind note-event output
// (piano-transcription / basic-pitch / mt3).
//
// These parse the produced bytes back rather than asserting on a golden blob:
// a golden file would pin the serialisation without proving it is VALID MIDI,
// and the failure mode that matters (a note-on with no matching note-off, or a
// note-off ordered after the next note-on at the same tick) is a structural
// property, not a byte pattern. The writer was additionally round-tripped
// through `mido` during development: 8/8 notes exact, no stuck notes.

#include <catch2/catch_test_macros.hpp>

#include "core/midi_writer.h"

#include <map>
#include <string>
#include <vector>

namespace {

struct ParsedNote {
    uint32_t on_tick;
    uint32_t off_tick;
    int note;
    int velocity;
    int channel;
    int program;
};

uint32_t read_u32be(const std::vector<uint8_t>& b, size_t& p) {
    uint32_t v = ((uint32_t)b[p] << 24) | ((uint32_t)b[p + 1] << 16) | ((uint32_t)b[p + 2] << 8) | b[p + 3];
    p += 4;
    return v;
}

uint16_t read_u16be(const std::vector<uint8_t>& b, size_t& p) {
    uint16_t v = (uint16_t)(((uint16_t)b[p] << 8) | b[p + 1]);
    p += 2;
    return v;
}

uint32_t read_vlq(const std::vector<uint8_t>& b, size_t& p) {
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        const uint8_t c = b[p++];
        v = (v << 7) | (uint8_t)(c & 0x7F);
        if (!(c & 0x80))
            break;
    }
    return v;
}

// Minimal independent SMF reader: enough to prove structural validity.
struct Parsed {
    uint16_t format = 0;
    uint16_t ntrks = 0;
    uint16_t ppq = 0;
    std::vector<ParsedNote> notes;
    bool stuck_notes = false;
    bool off_after_on_same_tick = false;
};

Parsed parse(const std::vector<uint8_t>& b) {
    Parsed out;
    size_t p = 0;
    REQUIRE(b.size() > 14);
    REQUIRE(std::string(b.begin(), b.begin() + 4) == "MThd");
    p = 4;
    const uint32_t hlen = read_u32be(b, p);
    REQUIRE(hlen == 6);
    out.format = read_u16be(b, p);
    out.ntrks = read_u16be(b, p);
    out.ppq = read_u16be(b, p);

    for (int t = 0; t < out.ntrks; t++) {
        REQUIRE(p + 8 <= b.size());
        REQUIRE(std::string(b.begin() + (long)p, b.begin() + (long)p + 4) == "MTrk");
        p += 4;
        const uint32_t tlen = read_u32be(b, p);
        const size_t end = p + tlen;
        REQUIRE(end <= b.size());

        uint32_t tick = 0;
        std::map<int, int> program; // channel -> program
        std::map<std::pair<int, int>, std::pair<uint32_t, int>> open_notes;
        uint32_t last_on_tick = 0;
        bool saw_on = false;

        while (p < end) {
            tick += read_vlq(b, p);
            const uint8_t status = b[p++];
            if (status == 0xFF) {
                const uint8_t meta = b[p++];
                const uint32_t len = read_vlq(b, p);
                p += len;
                (void)meta;
            } else if ((status & 0xF0) == 0xC0) {
                program[status & 0x0F] = b[p++];
            } else if ((status & 0xF0) == 0x90) {
                const int note = b[p++];
                const int vel = b[p++];
                const int ch = status & 0x0F;
                if (vel > 0) {
                    open_notes[{ch, note}] = {tick, vel};
                    last_on_tick = tick;
                    saw_on = true;
                } else {
                    open_notes.erase({ch, note});
                }
            } else if ((status & 0xF0) == 0x80) {
                const int note = b[p++];
                p++; // release velocity
                const int ch = status & 0x0F;
                auto it = open_notes.find({ch, note});
                if (it != open_notes.end()) {
                    if (saw_on && tick == last_on_tick)
                        out.off_after_on_same_tick = true;
                    out.notes.push_back(
                        {it->second.first, tick, note, it->second.second, ch, program.count(ch) ? program[ch] : 0});
                    open_notes.erase(it);
                }
            } else {
                FAIL("unexpected status byte in track");
            }
        }
        if (!open_notes.empty())
            out.stuck_notes = true;
        p = end;
    }
    return out;
}

} // namespace

TEST_CASE("midi_writer emits a structurally valid format-1 file", "[unit;midi]") {
    std::vector<core_midi::Note> notes = {
        {0.5, 1.0, 60, 100, 0, false},
        {0.5, 1.5, 64, 90, 0, false},
        {2.0, 3.0, 45, 64, 33, false},
        {0.0, 0.25, 38, 110, 0, true},
    };
    std::vector<uint8_t> bytes;
    REQUIRE(core_midi::build_smf(notes, bytes));

    Parsed pr = parse(bytes);
    CHECK(pr.format == 1);
    CHECK(pr.ppq == 480);
    // tempo track + one per (program, is_drum): piano(0), bass(33), drums
    CHECK(pr.ntrks == 4);
    CHECK(pr.notes.size() == notes.size());
    CHECK_FALSE(pr.stuck_notes);
}

TEST_CASE("midi_writer preserves pitch, velocity, program and drum routing", "[unit;midi]") {
    std::vector<core_midi::Note> notes = {
        {1.0, 2.0, 67, 80, 0, false},
        {1.0, 2.0, 45, 64, 33, false},
        {1.0, 2.0, 38, 110, 0, true},
    };
    std::vector<uint8_t> bytes;
    REQUIRE(core_midi::build_smf(notes, bytes));
    Parsed pr = parse(bytes);
    REQUIRE(pr.notes.size() == 3);

    for (const auto& n : pr.notes) {
        if (n.note == 67) {
            CHECK(n.velocity == 80);
            CHECK(n.program == 0);
            CHECK(n.channel != 9);
        } else if (n.note == 45) {
            CHECK(n.velocity == 64);
            CHECK(n.program == 33); // program survived
            CHECK(n.channel != 9);
        } else if (n.note == 38) {
            CHECK(n.velocity == 110);
            CHECK(n.channel == 9); // GM percussion channel
        }
    }
}

TEST_CASE("midi_writer never emits a zero-length or inverted note", "[unit;midi]") {
    // A detection whose offset collapsed onto (or before) its onset is legal
    // model output; a note-off at or before its note-on is a stuck note in
    // every player, so the writer must widen it.
    std::vector<core_midi::Note> notes = {
        {1.0, 1.0, 60, 90, 0, false}, {2.0, 1.5, 62, 90, 0, false}, // inverted
    };
    std::vector<uint8_t> bytes;
    REQUIRE(core_midi::build_smf(notes, bytes));
    Parsed pr = parse(bytes);
    REQUIRE(pr.notes.size() == 2);
    for (const auto& n : pr.notes)
        CHECK(n.off_tick > n.on_tick);
    CHECK_FALSE(pr.stuck_notes);
}

TEST_CASE("midi_writer orders note-off before note-on at the same tick", "[unit;midi]") {
    // Repeated pitch: the previous note's off must precede the next note's on,
    // or the player hears one long note (or drops the retrigger).
    std::vector<core_midi::Note> notes = {
        {0.0, 1.0, 60, 90, 0, false},
        {1.0, 2.0, 60, 90, 0, false},
    };
    std::vector<uint8_t> bytes;
    REQUIRE(core_midi::build_smf(notes, bytes));
    Parsed pr = parse(bytes);
    CHECK(pr.notes.size() == 2);
    CHECK_FALSE(pr.stuck_notes);
}

TEST_CASE("midi_writer clamps out-of-range values and rejects empty input", "[unit;midi]") {
    std::vector<uint8_t> bytes;
    CHECK_FALSE(core_midi::build_smf({}, bytes));

    std::vector<core_midi::Note> notes = {
        {0.0, 1.0, 999, 999, 999, false}, // all out of range
        {0.0, 1.0, -5, 0, -1, false},
    };
    REQUIRE(core_midi::build_smf(notes, bytes));
    Parsed pr = parse(bytes);
    for (const auto& n : pr.notes) {
        CHECK(n.note >= 0);
        CHECK(n.note <= 127);
        CHECK(n.velocity >= 1); // 0 would read as a note-off
        CHECK(n.velocity <= 127);
        CHECK(n.program >= 0);
        CHECK(n.program <= 127);
    }
}
