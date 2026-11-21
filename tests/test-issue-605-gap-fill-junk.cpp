// test-issue-605-gap-fill-junk.cpp — issue #605 regression guard.
//
// The issue #89 gap-fill second pass re-transcribes every >= 1 s span a slice
// left empty. On a noisy game stream most of those spans are pauses, music and
// sound effects, not speech the encoder dropped, and a short window of noise
// makes the TDT decoder invent words ("隠しのように", "ーーっ。っ。"). Each one
// then became its own short subtitle cue. Two cheap signals separate them from
// the real #89 drops, which decode in isolation at full confidence:
//
//   * the decoder's own token confidence — measured on the #605 clip, first-pass
//     words have a median mean-p of 0.997 while recovered junk sits at 0.1-0.6;
//     the one recovery whisper confirms ("投げね") is p = 1.00;
//   * the seam — a refill window re-hears the word next to the hole and times it
//     a little later, doubling it ("はい" + "はい。").
//
// Pure CPU, no model load: a scripted fake backend returns preset words with
// preset confidences for whatever window it is asked to transcribe.

#include "crispasr_gap_fill.h"
#include "whisper_params.h"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

namespace {

constexpr int kSampleRate = 16000;

struct scripted_word {
    const char* text;
    int64_t t0, t1;
    float p; // token confidence; < 0 = unavailable
};

class ScriptedBackend : public CrispasrBackend {
public:
    explicit ScriptedBackend(std::vector<scripted_word> script) : script_(std::move(script)) {}

    const char* name() const override { return "scripted"; }
    uint32_t capabilities() const override { return 0; }
    bool init(const whisper_params&) override { return true; }
    void shutdown() override {}

    std::vector<crispasr_segment> transcribe(const float*, int n_samples, int64_t t_offset_cs,
                                             const whisper_params&) override {
        const int64_t win0 = t_offset_cs;
        const int64_t win1 = t_offset_cs + (int64_t)n_samples * 100 / kSampleRate;
        crispasr_segment seg;
        for (const auto& s : script_) {
            if (s.t0 < win0 || s.t1 > win1)
                continue;
            crispasr_word w;
            w.text = s.text;
            w.t0 = s.t0;
            w.t1 = s.t1;
            seg.words.push_back(w);
            crispasr_token tk;
            tk.text = s.text;
            tk.t0 = s.t0;
            tk.t1 = s.t1;
            tk.confidence = s.p;
            seg.tokens.push_back(tk);
        }
        if (seg.words.empty())
            return {};
        seg.t0 = seg.words.front().t0;
        seg.t1 = seg.words.back().t1;
        seg.text = crispasr_rebuild_text_from_words(seg.words);
        return {seg};
    }

private:
    std::vector<scripted_word> script_;
};

// The first pass: one segment holding the given (already emitted) words.
std::vector<crispasr_segment> first_pass(const std::vector<scripted_word>& emitted) {
    crispasr_segment seg;
    for (const auto& s : emitted) {
        crispasr_word w;
        w.text = s.text;
        w.t0 = s.t0;
        w.t1 = s.t1;
        seg.words.push_back(w);
    }
    seg.text = crispasr_rebuild_text_from_words(seg.words);
    seg.t0 = seg.words.front().t0;
    seg.t1 = seg.words.back().t1;
    return {seg};
}

std::string all_text(const std::vector<crispasr_segment>& segs) {
    std::string out;
    for (const auto& s : segs)
        out += s.text;
    return out;
}

void run_gap_fill(ScriptedBackend& be, std::vector<crispasr_segment>& segs, int64_t t0_cs, int64_t t1_cs) {
    whisper_params params;
    const int n = (int)(t1_cs * kSampleRate / 100) + kSampleRate;
    std::vector<float> pcm((size_t)n, 0.0f);
    crispasr_audio_slice sl;
    sl.t0_cs = t0_cs;
    sl.t1_cs = t1_cs;
    sl.start = (int)(t0_cs * kSampleRate / 100);
    sl.end = (int)(t1_cs * kSampleRate / 100);
    crispasr_gap_fill_slice(be, params, pcm.data(), n, kSampleRate, sl, segs);
}

} // namespace

// The 10.2-17.2 s slice of the reporter's clip: "ついたか?" … "おっついたな" with a
// 4.8 s pause between them. Gap-fill decoded the pause as "隠しのように" at
// p 0.11-0.54 and the subtitle split into four cues.
TEST_CASE("issue #605: a low-confidence recovery in a pause is dropped", "[unit][gap-fill][issue-605]") {
    const std::vector<scripted_word> said = {{"ついたか?", 1037, 1141, 0.99f}, {"おっついたな", 1621, 1701, 0.99f}};
    ScriptedBackend be({said[0], {"隠し", 1153, 1177, 0.11f}, {"のように", 1225, 1281, 0.33f}, said[1]});
    auto segs = first_pass(said);
    run_gap_fill(be, segs, 1021, 1715);
    REQUIRE(segs.size() == 1);
    REQUIRE(all_text(segs) == "ついたか?おっついたな");
}

// The #89 case gap-fill exists for: the encoder blanked a real utterance, and it
// decodes verbatim, at full confidence, in isolation. It must still come back.
TEST_CASE("issue #605: a confident recovery is still merged back (#89)", "[unit][gap-fill][issue-605]") {
    const std::vector<scripted_word> said = {{"一緒に", 13536, 13600, 0.99f}, {"やばい", 14020, 14104, 0.99f}};
    ScriptedBackend be({said[0], {"投げ", 13800, 13836, 1.0f}, {"ね", 13836, 13852, 0.99f}, said[1]});
    auto segs = first_pass(said);
    run_gap_fill(be, segs, 13500, 14200);
    REQUIRE(all_text(segs).find("投げね") != std::string::npos);
}

// The gate judges a recovery as a whole: one hesitant token inside an otherwise
// confident refill does not throw the refill away.
TEST_CASE("issue #605: the gate averages over the recovered run", "[unit][gap-fill][issue-605]") {
    const std::vector<scripted_word> said = {{"あ", 1000, 1020, 0.99f}, {"そう", 1500, 1540, 0.99f}};
    ScriptedBackend be(
        {said[0], {"これ", 1100, 1130, 0.98f}, {"投げ", 1130, 1160, 0.70f}, {"ね", 1160, 1180, 0.99f}, said[1]});
    auto segs = first_pass(said);
    run_gap_fill(be, segs, 990, 1560);
    REQUIRE(all_text(segs).find("これ投げね") != std::string::npos);
}

// The refill window re-hears the first-pass word next to the hole and times it a
// little later, so it lands in the hole: "はい" (first pass) + "はい。" (refill).
TEST_CASE("issue #605: a refill repeating the word at the seam is dropped", "[unit][gap-fill][issue-605]") {
    const std::vector<scripted_word> said = {{"はい", 8821, 8833, 0.99f}, {"あ違った", 8989, 9197, 0.99f}};
    ScriptedBackend be({said[0], {"はい。", 8840, 8961, 0.94f}, said[1]});
    auto segs = first_pass(said);
    run_gap_fill(be, segs, 8800, 9210);
    REQUIRE(all_text(segs) == "はいあ違った");
}

// A backend that reports no token confidence (-1) cannot be judged; its
// recoveries are kept exactly as before.
TEST_CASE("issue #605: recoveries without confidence are kept", "[unit][gap-fill][issue-605]") {
    const std::vector<scripted_word> said = {{"one", 1000, 1050, -1.0f}, {"four", 1600, 1650, -1.0f}};
    ScriptedBackend be({said[0], {"two", 1200, 1250, -1.0f}, {"three", 1300, 1350, -1.0f}, said[1]});
    auto segs = first_pass(said);
    run_gap_fill(be, segs, 990, 1660);
    REQUIRE(all_text(segs).find("two") != std::string::npos);
    REQUIRE(all_text(segs).find("three") != std::string::npos);
}
