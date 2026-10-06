// test-live-translate.cpp — unit tests for the live transcribe + translate
// commit policy (examples/cli/crispasr_live_translate.h) and its sink
// (crispasr_live_translate_sink.h). No model, no audio: the "translator" is a
// lambda, the partials are scripted.
//
// What these pin:
//   * a sentence is committed once, and only after the recogniser moved past
//     it AND two partials agree on it — the period a recogniser writes at the
//     cut end of a partial must not commit half a sentence;
//   * committed text is immutable: neither a later partial nor the final can
//     make it come out twice;
//   * the policy still works when the rolling window has evicted the start of
//     the utterance, and on text with no punctuation at all.

#include <catch2/catch_test_macros.hpp>

#include "../examples/cli/crispasr_live_translate_sink.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

using crispasr::lt_committer;
using crispasr::lt_ends_sentence;
using crispasr::lt_split_words;
using crispasr::lt_update;

namespace {

std::vector<std::string> texts(const lt_update& u) {
    std::vector<std::string> out;
    for (const auto& s : u.committed)
        out.push_back(s.text);
    return out;
}

bool ends(const std::string& text, size_t i) {
    return lt_ends_sentence(lt_split_words(text), i);
}

} // namespace

TEST_CASE("live-translate: tokens normalise case and edge punctuation", "[unit][live-translate]") {
    const auto w = lt_split_words("  \xE2\x80\x9EGuten Morgen!\xE2\x80\x9C  \xC3\x84rger, ja. ");
    REQUIRE(w.size() == 4);
    REQUIRE(w[0].raw == "\xE2\x80\x9EGuten");
    REQUIRE(w[0].norm == "guten");
    REQUIRE(w[1].norm == "morgen");
    REQUIRE(w[2].norm == "\xC3\xA4rger"); // Ä folds to ä
    REQUIRE(w[3].norm == "ja");
}

TEST_CASE("live-translate: sentence ends — German ordinals and abbreviations are not ends", "[unit][live-translate]") {
    REQUIRE(ends("Das ist gut.", 2));
    REQUIRE(ends("Wirklich?", 0));
    REQUIRE(ends("Er sagte: \"Nein!\"", 2));
    // "am 3. Oktober": an ordinal, and the next word is capitalised — the
    // capital-letter heuristic alone would split here.
    REQUIRE_FALSE(ends("am 3. Oktober", 1));
    REQUIRE_FALSE(ends("z. B. hier", 0));
    REQUIRE_FALSE(ends("z. B. hier", 1));
    REQUIRE_FALSE(ends("Herr Dr. Meier", 1));
    REQUIRE_FALSE(ends("z.B. hier", 0));
    REQUIRE_FALSE(ends("Moment...", 0));
    // A year closes a sentence like any other word.
    REQUIRE(ends("im Jahr 1990.", 2));
    REQUIRE_FALSE(ends("kein Ende", 1));
}

TEST_CASE("live-translate: a script without spaces still yields sentence units", "[unit][live-translate]") {
    // Whitespace tokenisation alone makes this ONE token, and nothing would
    // ever commit before the final.
    const auto w = lt_split_words("今日は晴れです。明日は雨です。");
    REQUIRE(w.size() == 2);
    REQUIRE(lt_ends_sentence(w, 0));
    REQUIRE(crispasr::lt_join(w, 0, 2) == "今日は晴れです。明日は雨です。");
}

TEST_CASE("live-translate: a sentence commits once its terminator survives new right-context",
          "[unit][live-translate]") {
    lt_committer c;
    // The recogniser closes every partial with a period — that alone commits nothing.
    REQUIRE(c.on_partial(1, "Guten Morgen.").committed.empty());
    REQUIRE(c.on_partial(1, "Guten Morgen zusammen.").committed.empty());
    // The period is still there now that the recogniser has heard what
    // follows, and the previous partial wrote it too: committed.
    const lt_update u = c.on_partial(1, "Guten Morgen zusammen. Ich");
    REQUIRE(texts(u) == std::vector<std::string>{"Guten Morgen zusammen."});
    REQUIRE(u.tail == "Ich");
    REQUIRE(u.committed[0].id == 0);
}

TEST_CASE("live-translate: a sentence first seen WITH right-context waits for a second partial",
          "[unit][live-translate]") {
    lt_committer c;
    c.on_partial(1, "Guten");
    // One partial is not agreement.
    REQUIRE(c.on_partial(1, "Guten Morgen zusammen. Ich").committed.empty());
    REQUIRE(texts(c.on_partial(1, "Guten Morgen zusammen. Ich bin")) ==
            std::vector<std::string>{"Guten Morgen zusammen."});
}

TEST_CASE("live-translate: the cut-end period of a partial does not split a sentence", "[unit][live-translate]") {
    lt_committer c;
    c.on_partial(1, "Ich gehe.");
    // The period was an artefact of where the audio stopped: with more audio
    // the recogniser drops it, so "Ich gehe." must never have been committed.
    REQUIRE(c.on_partial(1, "Ich gehe nach").committed.empty());
    REQUIRE(c.on_partial(1, "Ich gehe nach Hause.").committed.empty());
    REQUIRE(texts(c.on_partial(1, "Ich gehe nach Hause. Dann")) == std::vector<std::string>{"Ich gehe nach Hause."});
}

TEST_CASE("live-translate: a terminator that flips on alternate partials still commits", "[unit][live-translate]") {
    // Seen live with parakeet-v3: "vorstellen. Er kommt" / "vorstellen, er
    // kommt" on alternating partials for five seconds. Two-in-a-row never
    // happened, and the sentence was held until the next one ended.
    lt_committer c;
    REQUIRE(c.on_partial(1, "Kollegen vorstellen.").committed.empty());
    REQUIRE(c.on_partial(1, "Kollegen vorstellen, er kommt.").committed.empty());
    REQUIRE(c.on_partial(1, "Kollegen vorstellen. Er kommt aus München.").committed.empty() == false);
}

TEST_CASE("live-translate: a terminator seen once only does not commit", "[unit][live-translate]") {
    lt_committer c;
    c.on_partial(1, "Kollegen vorstellen, er");
    c.on_partial(1, "Kollegen vorstellen, er kommt");
    // First time with a period: neither of the last two partials had it.
    REQUIRE(c.on_partial(1, "Kollegen vorstellen. Er kommt aus").committed.empty());
}

TEST_CASE("live-translate: committed text never comes out twice", "[unit][live-translate]") {
    lt_committer c;
    c.on_partial(1, "Erster Satz. Zweiter");
    REQUIRE(texts(c.on_partial(1, "Erster Satz. Zweiter Satz")) == std::vector<std::string>{"Erster Satz."});
    // Same partial again, and a longer one: nothing new until the next end is stable.
    REQUIRE(c.on_partial(1, "Erster Satz. Zweiter Satz").committed.empty());
    REQUIRE(c.on_partial(1, "Erster Satz. Zweiter Satz ist hier. Dritter").committed.empty());
    REQUIRE(texts(c.on_partial(1, "Erster Satz. Zweiter Satz ist hier. Dritter Satz")) ==
            std::vector<std::string>{"Zweiter Satz ist hier."});
    // The final supplies only the remainder.
    const lt_update f = c.on_final(1, "Erster Satz. Zweiter Satz ist hier. Dritter Satz kommt.");
    REQUIRE(texts(f) == std::vector<std::string>{"Dritter Satz kommt."});
    REQUIRE(f.committed[0].id == 2);
}

TEST_CASE("live-translate: the final commits everything when no partial did", "[unit][live-translate]") {
    lt_committer c;
    c.on_partial(7, "Hallo");
    const lt_update f = c.on_final(7, "Hallo zusammen. Wie geht es euch? Gut");
    REQUIRE(texts(f) == std::vector<std::string>{"Hallo zusammen.", "Wie geht es euch?", "Gut"});
    // A new utterance starts clean.
    REQUIRE(texts(c.on_final(8, "Neu.")) == std::vector<std::string>{"Neu."});
}

TEST_CASE("live-translate: alignment survives the rolling window evicting the start", "[unit][live-translate]") {
    lt_committer c;
    c.on_partial(1, "Wir beginnen jetzt mit dem ersten Teil. Danach");
    REQUIRE(texts(c.on_partial(1, "Wir beginnen jetzt mit dem ersten Teil. Danach kommt")) ==
            std::vector<std::string>{"Wir beginnen jetzt mit dem ersten Teil."});
    // The window rolled: the hypothesis now starts mid-way through the
    // committed sentence. Only what follows it is open.
    c.on_partial(1, "mit dem ersten Teil. Danach kommt der zweite Teil. Und");
    const lt_update u = c.on_partial(1, "mit dem ersten Teil. Danach kommt der zweite Teil. Und dann");
    REQUIRE(texts(u) == std::vector<std::string>{"Danach kommt der zweite Teil."});
    REQUIRE(u.tail == "Und dann");
}

TEST_CASE("live-translate: a re-rendered number inside committed text does not re-commit its last word",
          "[unit][live-translate]") {
    // Seen live with parakeet-v3 on German: the same audio came back as
    // "12%" in one partial and "zwölf Prozent" in the next. One token turned
    // into two, a position-by-position compare of the committed words landed
    // one word early, and "gestiegen." was committed again as its own
    // sentence (and translated: "and rise.").
    lt_committer c;
    c.on_partial(1, "Die Umsätze sind im Vergleich zum Vorjahr um 12% gestiegen. Das ist");
    REQUIRE(texts(c.on_partial(1, "Die Umsätze sind im Vergleich zum Vorjahr um 12% gestiegen. Das ist vor")) ==
            std::vector<std::string>{"Die Umsätze sind im Vergleich zum Vorjahr um 12% gestiegen."});
    lt_update u =
        c.on_partial(1, "Die Umsätze sind im Vergleich zum Vorjahr um zwölf Prozent gestiegen. Das ist vor allem");
    REQUIRE(u.committed.empty());
    REQUIRE(u.tail == "Das ist vor allem");
    // …and the final does not swallow the next sentence's first word either.
    u = c.on_final(1, "Die Umsätze sind im Vergleich zum Vorjahr um zwölf Prozent gestiegen. Das ist vor allem so.");
    REQUIRE(texts(u) == std::vector<std::string>{"Das ist vor allem so."});
}

TEST_CASE("live-translate: a revised boundary stalls instead of duplicating, then resyncs", "[unit][live-translate]") {
    lt_committer c;
    c.on_partial(1, "Alpha beta gamma delta. Epsilon");
    REQUIRE(c.on_partial(1, "Alpha beta gamma delta. Epsilon zeta").committed.size() == 1);
    // The recogniser rewrote the committed words entirely: no alignment.
    lt_update u = c.on_partial(1, "Omega psi chi phi tau");
    REQUIRE(u.committed.empty());
    REQUIRE(u.tail == "Epsilon zeta"); // unchanged while we wait
    c.on_partial(1, "Omega psi chi phi tau sigma");
    // Third miss in a row: nothing of the committed text is in the window any
    // more, so all of this is new text.
    u = c.on_partial(1, "Omega psi chi phi tau sigma rho");
    REQUIRE(u.tail == "Omega psi chi phi tau sigma rho");
}

TEST_CASE("live-translate: unpunctuated speech is force-committed at a stable point", "[unit][live-translate]") {
    lt_committer c;
    std::string text;
    lt_update u;
    int committed_words = 0;
    for (int i = 0; i < 60; ++i) {
        text += (i ? " w" : "w") + std::to_string(i);
        u = c.on_partial(1, text);
        for (const auto& s : u.committed)
            committed_words += (int)lt_split_words(s.text).size();
    }
    // Something was committed without any punctuation or silence…
    REQUIRE(committed_words > 0);
    // …and the open tail stayed bounded.
    REQUIRE((int)lt_split_words(u.tail).size() <= 31);
    // Nothing lost, nothing doubled: committed + tail is the whole text.
    REQUIRE(committed_words + (int)lt_split_words(u.tail).size() == 60);
}

TEST_CASE("live-translate: under pressure a long open sentence is committed at a clause boundary",
          "[unit][live-translate]") {
    // 16 words, no sentence end: far below the normal forced-commit length,
    // so nothing commits…
    const std::string a = "eins zwei drei vier fünf sechs, sieben acht neun zehn elf zwölf dreizehn vierzehn fünfzehn";
    {
        lt_committer c;
        c.on_partial(1, a);
        REQUIRE(c.on_partial(1, a + " sechzehn").committed.empty());
    }
    // …but when the caller cannot afford to keep re-decoding it, the stable
    // part is committed, at the comma.
    lt_committer c;
    c.set_pressure(true);
    c.on_partial(1, a, 5.0);
    const lt_update u = c.on_partial(1, a + " sechzehn", 5.5);
    REQUIRE(texts(u) == std::vector<std::string>{"eins zwei drei vier fünf sechs,"});
    REQUIRE(u.tail == "sieben acht neun zehn elf zwölf dreizehn vierzehn fünfzehn sechzehn");
}

TEST_CASE("live-translate: decode_from moves with commits and is withdrawn on a miss", "[unit][live-translate]") {
    lt_committer c;
    REQUIRE(c.decode_from() < 0); // nothing committed: decode the whole region
    c.on_partial(1, "Erster Satz.", 4.0);
    c.on_partial(1, "Erster Satz. Zweiter", 4.5);
    // "Satz." was first heard in the partial at 4.0 and was not in the audio
    // before that partial (stream start), so the estimate stays early.
    REQUIRE(c.decode_from() <= 4.0);
    REQUIRE(c.decode_from() >= -0.5);
    c.on_partial(1, "Erster Satz. Zweiter Satz hier. Dritter", 6.0);
    c.on_partial(1, "Erster Satz. Zweiter Satz hier. Dritter Satz", 6.5);
    // "hier." appeared at 6.0, after the partial at 4.5: it ends after ~4.5.
    const double from = c.decode_from();
    REQUIRE(from > 3.5);
    REQUIRE(from <= 6.0);
    // A hypothesis that no longer shows the committed text withdraws it, so
    // the caller decodes from the start of the speech region again.
    c.on_partial(1, "Ganz anderer Text ohne Bezug", 7.0);
    REQUIRE(c.decode_from() < 0);
    REQUIRE(c.align_misses() == 1);
}

TEST_CASE("live-translate: a pause commits a finished sentence, and only a finished one", "[unit][live-translate]") {
    lt_committer c;
    c.on_partial(1, "Haben Sie Fragen");
    // Mid-sentence breath: nothing to commit.
    REQUIRE(c.on_pause(1).committed.empty());
    c.on_partial(1, "Haben Sie Fragen?");
    // Finished, and the speaker stopped: no need to wait for the next word.
    const lt_update u = c.on_pause(1);
    REQUIRE(texts(u) == std::vector<std::string>{"Haben Sie Fragen?"});
    // Already committed — a second pause, the next partial and the final add nothing twice.
    REQUIRE(c.on_pause(1).committed.empty());
    REQUIRE(c.on_partial(1, "Haben Sie Fragen? Gut").committed.empty());
    REQUIRE(texts(c.on_final(1, "Haben Sie Fragen? Gut.")) == std::vector<std::string>{"Gut."});
    // A pause reported for some other utterance is ignored.
    c.on_partial(2, "Ende.");
    REQUIRE(c.on_pause(3).committed.empty());
}

TEST_CASE("live-translate: with word timestamps the boundary is a time, not an alignment", "[unit][live-translate]") {
    using crispasr::lt_timed_word;
    lt_committer c;
    std::vector<lt_timed_word> t1 = {{"Erster", 0.2, 0.6}, {"Satz.", 0.6, 1.0}, {"Zweiter", 1.4, 1.9}};
    c.on_partial(1, "Erster Satz.", 1.1);
    REQUIRE(texts(c.on_partial(1, "Erster Satz. Zweiter", 2.0, &t1)) == std::vector<std::string>{"Erster Satz."});
    // The boundary is the end of "Satz." exactly.
    REQUIRE(c.decode_from_exact());
    REQUIRE(c.decode_from() == 1.0);
    // The caller now decodes from just before 1.0 s. The hypothesis shows
    // NONE of the committed words in a recognisable form — a mangled sliver
    // of the old sentence at most — which text alignment could not resolve.
    std::vector<lt_timed_word> t2 = {
        {"atz.", 0.8, 1.0}, {"Zweiter", 1.4, 1.9}, {"Satz", 1.9, 2.3}, {"hier.", 2.3, 2.7}, {"Und", 3.0, 3.2}};
    lt_update u = c.on_partial(1, "atz. Zweiter Satz hier. Und", 3.3, &t2);
    REQUIRE(c.align_misses() == 0);
    REQUIRE(u.committed.empty()); // "hier." has been seen once only
    REQUIRE(u.tail == "Zweiter Satz hier. Und");
    std::vector<lt_timed_word> t3 = t2;
    t3.push_back({"dann", 3.2, 3.5});
    u = c.on_partial(1, "atz. Zweiter Satz hier. Und dann", 3.6, &t3);
    REQUIRE(texts(u) == std::vector<std::string>{"Zweiter Satz hier."});
    REQUIRE(c.decode_from() == 2.7);
    // After a pause the hypothesis starts beyond the boundary: all of it is open.
    std::vector<lt_timed_word> t4 = {{"Ende", 6.0, 6.4}, {"gut.", 6.4, 6.8}};
    u = c.on_partial(1, "Ende gut.", 7.0, &t4);
    REQUIRE(c.align_misses() == 0);
    REQUIRE(u.tail == "Ende gut.");
    // The utterance closes on that same hypothesis: the remainder must come
    // out even though no committed word is in it to line up against.
    {
        lt_committer d;
        std::vector<lt_timed_word> a = {{"Eins.", 0.1, 0.5}, {"Zwei", 0.9, 1.2}};
        d.on_partial(1, "Eins.", 0.6);
        d.on_partial(1, "Eins. Zwei", 1.3, &a);
        std::vector<lt_timed_word> b = {{"Zwei", 0.9, 1.2}, {"drei.", 1.2, 1.6}};
        d.on_partial(1, "Zwei drei.", 1.8, &b);
        REQUIRE(texts(d.on_final(1, "Zwei drei.")) == std::vector<std::string>{"Zwei drei."});
    }
    // A forced commit ends mid-phrase; the re-decode hears its last word
    // again, timed slightly late. It must not come out twice ("auf den Den").
    {
        lt_committer d;
        d.set_pressure(true);
        std::vector<lt_timed_word> a;
        std::string text;
        for (int i = 0; i < 16; ++i) {
            const std::string w = i == 6 ? "den" : "w" + std::to_string(i);
            a.push_back({w, 0.3 * i, 0.3 * i + 0.3});
            text += (i ? " " : "") + w;
            d.on_partial(1, text, 0.3 * i + 0.4, &a);
        }
        REQUIRE(d.decode_from_exact());
        const double boundary = d.decode_from();
        // Re-decode from the boundary: the committed word again, 0.2 s late, then new words.
        const int k = (int)(boundary / 0.3 + 0.5) - 1; // index of the last committed word
        std::vector<lt_timed_word> b = {{a[(size_t)k].text, boundary - 0.1, boundary + 0.2}};
        std::string t2 = a[(size_t)k].text;
        for (size_t i = (size_t)k + 1; i < a.size(); ++i) {
            b.push_back(a[i]);
            t2 += " " + a[i].text;
        }
        const lt_update u2 = d.on_partial(1, t2, 6.0, &b);
        REQUIRE(lt_split_words(u2.tail).front().raw == a[(size_t)k + 1].text);
    }
    // A word list that does not match the text is ignored, not trusted.
    std::vector<lt_timed_word> bad = {{"ganz", 5.0, 5.2}, {"anders", 5.2, 5.5}};
    u = c.on_partial(1, "Zweiter Satz hier. Und dann weiter", 8.0, &bad);
    REQUIRE(u.tail == "Und dann weiter");
}

TEST_CASE("live-translate: a final that does not line up still commits what was open", "[unit][live-translate]") {
    // Seen live: the last partial of an utterance was discarded (its word
    // list carried words of the NEXT utterance), then the utterance closed.
    // "Haben Sie bis hierhin Fragen?" had been on screen as open text and
    // was never committed or translated.
    lt_committer c;
    c.on_partial(1, "Er hat dort gearbeitet. Haben Sie");
    REQUIRE(c.on_partial(1, "Er hat dort gearbeitet. Haben Sie Fragen?").committed.size() == 1);
    // A hypothesis with none of the committed words: a miss, nothing changes.
    REQUIRE(c.on_partial(1, "völlig anderer unpassender Text hier").committed.empty());
    const lt_update f = c.on_final(1, "völlig anderer unpassender Text hier");
    REQUIRE(texts(f) == std::vector<std::string>{"Haben Sie Fragen?"});
}

TEST_CASE("live-translate: sink emits one translation per sentence, in order", "[unit][live-translate]") {
    FILE* f = tmpfile();
    REQUIRE(f != nullptr);
    std::vector<std::string> seen;
    {
        crispasr::lt_sink_config cfg;
        cfg.output = crispasr::lt_output::json;
        cfg.src_lang = "de";
        cfg.tgt_lang = "en";
        cfg.sync = true;
        cfg.drafts = false;
        cfg.out = f;
        cfg.log = nullptr;
        crispasr::lt_sink sink(cfg, [&](const std::string& s, const crispasr::lt_sink::progress_fn&) {
            seen.push_back(s);
            return "EN<" + s + ">";
        });
        const auto now = crispasr::lt_sink::clock::now();
        sink.on_partial(1, "Guten \"Morgen\". Ich", 1.0, now);
        sink.on_partial(1, "Guten \"Morgen\". Ich bin", 1.5, now);
        sink.on_final(1, "Guten \"Morgen\". Ich bin da.", 2.5, now);
    }
    REQUIRE(seen == std::vector<std::string>{"Guten \"Morgen\".", "Ich bin da."});
    rewind(f);
    std::string all;
    char buf[512];
    while (fgets(buf, sizeof(buf), f))
        all += buf;
    fclose(f);
    // The source quote is escaped; the events carry both languages.
    REQUIRE(all.find("\"type\":\"sentence\",\"utterance_id\":1,\"sentence_id\":0,\"text\":\"Guten \\\"Morgen\\\".\"") !=
            std::string::npos);
    REQUIRE(all.find("\"type\":\"translation\",\"utterance_id\":1,\"sentence_id\":1,\"text\":\"Ich bin da.\","
                     "\"translation\":\"EN<Ich bin da.>\",\"source_lang\":\"de\",\"target_lang\":\"en\"") !=
            std::string::npos);
}

TEST_CASE("live-translate: a forced clause split is not translated cold — the sentence is the unit",
          "[unit][live-translate]") {
    // Seen live: under pressure the committer split a sentence after "nach",
    // and the two halves were translated separately into nonsense.
    FILE* f = tmpfile();
    REQUIRE(f != nullptr);
    std::vector<std::string> seen;
    {
        crispasr::lt_sink_config cfg;
        cfg.output = crispasr::lt_output::json;
        cfg.src_lang = "de";
        cfg.tgt_lang = "en";
        cfg.sync = true;
        cfg.drafts = false;
        cfg.out = f;
        cfg.log = nullptr;
        cfg.commit.pressure_commit_words = 6;
        crispasr::lt_sink sink(cfg, [&](const std::string& s, const crispasr::lt_sink::progress_fn&) {
            seen.push_back(s);
            return std::string("x");
        });
        sink.set_pressure(true);
        const auto now = crispasr::lt_sink::clock::now();
        const std::string a = "das ist vor allem auf den starken Export nach Frankreich und Italien";
        sink.on_partial(1, a, 1.0, now);
        sink.on_partial(1, a + " zurückzuführen", 1.5, now);
        // The committer has let go of the first words (that is its job)…
        REQUIRE(sink.decode_from() > -1.0);
        // …but nothing has been translated: the sentence is not finished.
        REQUIRE(seen.empty());
        sink.on_final(1, a + " zurückzuführen.", 2.5, now);
    }
    fclose(f);
    REQUIRE(seen == std::vector<std::string>{
                        "das ist vor allem auf den starken Export nach Frankreich und Italien zurückzuführen."});
}

TEST_CASE("live-translate: a sentence translated ahead of time is not translated again when it commits",
          "[unit][live-translate]") {
    FILE* f = tmpfile();
    REQUIRE(f != nullptr);
    std::vector<std::string> seen;
    {
        crispasr::lt_sink_config cfg;
        cfg.output = crispasr::lt_output::json;
        cfg.src_lang = "de";
        cfg.tgt_lang = "en";
        cfg.sync = true;
        cfg.out = f;
        cfg.log = nullptr;
        crispasr::lt_sink sink(cfg, [&](const std::string& s, const crispasr::lt_sink::progress_fn&) {
            seen.push_back(s);
            return "EN<" + s + ">";
        });
        const auto now = crispasr::lt_sink::clock::now();
        // "Guten Morgen." is finished and followed by text: it is the next
        // commit candidate, so exactly that sentence is translated now…
        sink.on_partial(1, "Guten Morgen. Ich", 1.0, now);
        REQUIRE(seen == std::vector<std::string>{"Guten Morgen."});
        // …and when it commits on the next partial, nothing is translated twice.
        sink.on_partial(1, "Guten Morgen. Ich bin", 1.5, now);
        REQUIRE(std::count(seen.begin(), seen.end(), "Guten Morgen.") == 1);
    }
    rewind(f);
    std::string all;
    char buf[512];
    while (fgets(buf, sizeof(buf), f))
        all += buf;
    fclose(f);
    // The committed sentence still gets its translation event.
    REQUIRE(all.find("\"type\":\"translation\",\"utterance_id\":1,\"sentence_id\":0,\"text\":\"Guten Morgen.\","
                     "\"translation\":\"EN<Guten Morgen.>\"") != std::string::npos);
}

TEST_CASE("live-translate: threaded sink drains every committed sentence before it stops", "[unit][live-translate]") {
    FILE* f = tmpfile();
    REQUIRE(f != nullptr);
    int n = 0;
    {
        crispasr::lt_sink_config cfg;
        cfg.output = crispasr::lt_output::plain;
        cfg.src_lang = "de";
        cfg.tgt_lang = "en";
        cfg.out = f;
        cfg.log = nullptr;
        crispasr::lt_sink sink(cfg, [&](const std::string& s, const crispasr::lt_sink::progress_fn&) {
            ++n;
            return "T:" + s;
        });
        const auto now = crispasr::lt_sink::clock::now();
        sink.on_final(1, "Eins. Zwei. Drei.", 1.0, now);
        sink.finish();
    }
    rewind(f);
    std::string all;
    char buf[512];
    while (fgets(buf, sizeof(buf), f))
        all += buf;
    fclose(f);
    REQUIRE(all == "[de] Eins.\n[en] T:Eins.\n[de] Zwei.\n[en] T:Zwei.\n[de] Drei.\n[en] T:Drei.\n");
    REQUIRE(n == 3);
}

TEST_CASE("live-translate: the stable part of a draft is the whole words two drafts share", "[unit][live-translate]") {
    using crispasr::lt_detail::stable_prefix_bytes;
    const std::string a = "We have today", b = "We have three points today.";
    REQUIRE(b.substr(0, stable_prefix_bytes(a, b)) == "We have ");
    // A word that only begins the same is not shared ("the" / "them").
    REQUIRE(stable_prefix_bytes("First we talk about the", "First we talk about them all") ==
            std::string("First we talk about ").size());
    // Identical drafts are stable throughout; nothing in common is nothing.
    REQUIRE(stable_prefix_bytes("Thank you.", "Thank you.") == std::string("Thank you.").size());
    REQUIRE(stable_prefix_bytes("Sales are", "The turnover") == 0);
    REQUIRE(stable_prefix_bytes("", "Hello") == 0);
    // Multi-byte words compare as bytes.
    REQUIRE(stable_prefix_bytes("Größe über alles", "Größe über 12") == std::string("Größe über ").size());
}

TEST_CASE("live-translate: a draft reports the words it shares with the previous draft", "[unit][live-translate]") {
    FILE* f = tmpfile();
    REQUIRE(f != nullptr);
    {
        crispasr::lt_sink_config cfg;
        cfg.output = crispasr::lt_output::json;
        cfg.src_lang = "de";
        cfg.tgt_lang = "en";
        cfg.sync = true;
        cfg.draft_min_words = 1;
        cfg.out = f;
        cfg.log = nullptr;
        crispasr::lt_sink sink(cfg, [&](const std::string& s, const crispasr::lt_sink::progress_fn&) {
            return std::string(s == "Wir haben heute" ? "We have today" : "We have three points today.");
        });
        const auto now = crispasr::lt_sink::clock::now();
        sink.on_partial(1, "Wir haben heute", 1.0, now);
        sink.on_partial(1, "Wir haben heute drei Punkte", 2.0, now);
    }
    rewind(f);
    std::string all;
    char buf[512];
    while (fgets(buf, sizeof(buf), f))
        all += buf;
    fclose(f);
    INFO(all);
    REQUIRE(all.find("\"translation\":\"We have today\",\"stable\":\"\"") != std::string::npos);
    REQUIRE(all.find("\"translation\":\"We have three points today.\",\"stable\":\"We have \"") != std::string::npos);
}

TEST_CASE("live-translate: a draft's source continues when only its last word moved", "[unit][live-translate]") {
    using crispasr::lt_detail::source_continues;
    REQUIRE(source_continues("Wir haben heute drei Punkte.", "Wir haben heute drei Punkte auf der"));
    REQUIRE(source_continues("Guten Morgen und Herr.", "Guten Morgen und herzlich willkommen"));
    REQUIRE(source_continues("Zuerst sprechen wir", "Zuerst sprechen wir über die"));
    // An earlier word changed: not a continuation.
    REQUIRE_FALSE(source_continues("um 12 Prozent gestiegen.", "um zwölf Prozent gestiegen und"));
    // Shorter, or nothing before: not a continuation.
    REQUIRE_FALSE(source_continues("Wir haben heute drei", "Wir haben heute"));
    REQUIRE_FALSE(source_continues("", "Wir haben"));
}
