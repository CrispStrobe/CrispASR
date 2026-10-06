// crispasr_live_translate_sink.h — translator thread + output for the live
// transcribe + translate mode.
//
// The streaming loop hands every partial / final hypothesis to lt_sink. The
// sink runs the commit policy (crispasr_live_translate.h), sends committed
// sentences to the translator, and shows the result:
//
//   tty   — committed sentence pairs scroll up; the still-open tail and its
//           draft translation are redrawn in place below them.
//   plain — one `[src] …` / `[tgt] …` pair per committed sentence (stdout is
//           not a terminal).
//   json  — JSON-Lines `sentence` / `translation` / `translation_partial`
//           events, alongside the stream's own partial/final/silence events.
//
// Translation runs on its own thread so a slow translator delays only the
// translation line, never the recogniser: the next audio step is decoded
// while the previous sentence is still being translated.

#pragma once

#include "crispasr_live_translate.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <functional>
#include <mutex>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <windows.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace crispasr {

enum class lt_output { tty, plain, json };

// Is stdout a terminal that understands the cursor-movement escapes the tty
// view redraws with? On Windows that needs virtual-terminal processing
// switched on (Windows 10+); where it cannot be, the answer is no and the
// caller falls back to plain output.
inline bool lt_stdout_is_terminal() {
#if defined(_WIN32)
    if (!_isatty(_fileno(stdout)))
        return false;
    const HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    if (h == INVALID_HANDLE_VALUE || !GetConsoleMode(h, &mode))
        return false;
    if (!(mode & ENABLE_VIRTUAL_TERMINAL_PROCESSING) && !SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING))
        return false;
    SetConsoleOutputCP(CP_UTF8);
    return true;
#else
    return isatty(STDOUT_FILENO) != 0;
#endif
}

struct lt_sink_config {
    lt_output output = lt_output::plain;
    std::string src_lang;
    std::string tgt_lang;
    // Re-translate the open tail as a draft while it is still changing.
    bool drafts = true;
    // Translate on the caller's thread. Deterministic event order; used by
    // tests and available as CRISPASR_TRANSLATE_SYNC=1 for debugging.
    bool sync = false;
    // An unterminated run longer than this is translated as it stands: no
    // sentence is this long, so the recogniser is not punctuating.
    int max_unit_words = 60;
    // Fewer words than this are not worth a draft translation.
    int draft_min_words = 3;
    // Drafts stop once a committed sentence takes longer than this to
    // translate (running average). A translator that slow is busy with a
    // draft when the next real sentence arrives, and it shares the GPU with
    // the recogniser: with MADLAD-3B, drafts doubled the recogniser's step
    // time. They resume if translation gets fast again.
    double draft_max_mt_ms = 500.0;
    FILE* out = stdout;
    FILE* log = stderr;
    lt_commit_options commit;
};

namespace lt_detail {
// Byte length of the longest whole-word common prefix of `a` and `b`
// (words split on spaces), including the space after the last shared word.
inline size_t stable_prefix_bytes(const std::string& a, const std::string& b) {
    size_t i = 0, ok = 0;
    while (i < a.size() && i < b.size()) {
        const size_t ea = a.find(' ', i), eb = b.find(' ', i);
        const size_t la = (ea == std::string::npos ? a.size() : ea) - i;
        const size_t lb = (eb == std::string::npos ? b.size() : eb) - i;
        if (la != lb || a.compare(i, la, b, i, lb) != 0)
            break;
        // A word is only shared if it is complete in both (not a prefix of a longer word).
        i += la;
        ok = i;
        if (i < a.size() && i < b.size()) {
            ++i; // the space
            ok = i;
        } else {
            break;
        }
    }
    return ok;
}

// Whether `cur` continues `prev` as far as a draft's fixed start is concerned:
// every word of `prev` but its last reappears in `cur`, in place, ignoring
// trailing punctuation, and `cur` is not shorter. The recogniser's last word
// and its provisional full stop are still moving ("drei Punkte." -> "drei
// Punkte auf der"), so a literal prefix test almost never held.
inline bool source_continues(const std::string& prev, const std::string& cur) {
    auto words = [](const std::string& s) {
        std::vector<std::string> w;
        size_t i = 0;
        while (i < s.size()) {
            while (i < s.size() && s[i] == ' ')
                ++i;
            size_t j = i;
            while (j < s.size() && s[j] != ' ')
                ++j;
            if (j > i) {
                std::string t = s.substr(i, j - i);
                while (!t.empty() && std::strchr(".,;:!?", t.back()))
                    t.pop_back();
                w.push_back(t);
            }
            i = j;
        }
        return w;
    };
    const auto a = words(prev), b = words(cur);
    if (a.empty() || b.size() < a.size())
        return false;
    for (size_t k = 0; k + 1 < a.size(); ++k)
        if (a[k] != b[k])
            return false;
    return true;
}
} // namespace lt_detail

class lt_sink {
public:
    using clock = std::chrono::steady_clock;
    // Reports the translation so far while it is being generated. Optional:
    // a translator that produces its output in one go never calls it.
    using progress_fn = std::function<void(const std::string& so_far)>;
    // Translates one sentence. Called from the translator thread only (or the
    // caller's thread in sync mode) — never concurrently with itself.
    using translate_fn = std::function<std::string(const std::string& text, const progress_fn& progress)>;

    // A translator that can be given the fixed start of its translation
    // (target-language words). Optional: without it drafts are translated
    // free, as before.
    using prefixed_translate_fn = std::function<std::string(const std::string& text, const std::string& target_prefix)>;
    void set_prefixed_translator(prefixed_translate_fn fn) { prefixed_ = std::move(fn); }

    lt_sink(const lt_sink_config& cfg, translate_fn fn) : cfg_(cfg), translate_(std::move(fn)), committer_(cfg.commit) {
        if (!cfg_.sync)
            worker_ = std::thread([this] { run(); });
    }

    ~lt_sink() { finish(); }

    lt_sink(const lt_sink&) = delete;
    lt_sink& operator=(const lt_sink&) = delete;

    // `t_audio` is the stream time (seconds of audio received) of the step
    // that produced this hypothesis; `arrived` is when that audio was read.
    // Their difference to the moment a translation is shown is the lag.
    void on_partial(int64_t utterance_id, const std::string& text, double t_audio, clock::time_point arrived,
                    const std::vector<lt_timed_word>* timed = nullptr) {
        handle(committer_.on_partial(utterance_id, text, t_audio, timed), utterance_id, t_audio, arrived,
               /*closed=*/false);
    }

    // The speech stopped a moment ago and nothing new has been decoded since.
    // See lt_committer::on_pause.
    void on_pause(int64_t utterance_id, double t_audio, clock::time_point arrived) {
        const lt_update up = committer_.on_pause(utterance_id);
        if (!up.committed.empty())
            handle(up, utterance_id, t_audio, arrived, /*closed=*/false);
    }

    // The utterance closed; `text` is its last hypothesis. Returns the
    // utterance as it was committed — its sentences, joined — which is the
    // text of record: committed sentences are never revised, so this, not the
    // recogniser's last hypothesis, is what was shown and translated.
    std::string on_final(int64_t utterance_id, const std::string& text, double t_audio, clock::time_point arrived) {
        handle(committer_.on_final(utterance_id, text), utterance_id, t_audio, arrived, /*closed=*/true);
        std::string all;
        all.swap(utterance_text_);
        return all;
    }

    // Stream time (s) the caller may start decoding the open utterance from,
    // or < 0 for its start. See lt_committer::decode_from. Caller's thread.
    double decode_from() const { return committer_.decode_from(); }
    bool decode_from_exact() const { return committer_.decode_from_exact(); }

    // See lt_committer::set_pressure. Caller's thread.
    void set_pressure(bool on) { committer_.set_pressure(on); }

    // Translate everything still queued, stop the thread, print the summary.
    void finish() {
        if (finished_)
            return;
        finished_ = true;
        if (worker_.joinable()) {
            {
                std::lock_guard<std::mutex> lk(mu_);
                stop_ = true;
            }
            cv_.notify_all();
            worker_.join();
        }
        std::lock_guard<std::mutex> lk(mu_);
        tail_src_.clear();
        tail_draft_.clear();
        tail_draft_stable_ = 0;
        render_locked({});
        if (cfg_.log && !mt_ms_.empty()) {
            fprintf(cfg_.log,
                    "crispasr[translate]: %zu sentence(s); translate median %.0f ms, p90 %.0f ms; "
                    "lag behind audio median %.0f ms, p90 %.0f ms; %d translated ahead of time; "
                    "%d partial(s) discarded (no alignment)\n",
                    mt_ms_.size(), pct(mt_ms_, 0.5), pct(mt_ms_, 0.9), pct(lag_ms_, 0.5), pct(lag_ms_, 0.9), reused_,
                    committer_.align_misses());
        }
    }

private:
    struct job {
        bool draft = false;
        int id = 0;
        int64_t utterance_id = 0;
        std::string text;
        double t_audio = 0;
        clock::time_point arrived;
        std::string prefix; // draft only: fixed start of its translation (see set_prefixed_translator)
    };

    struct pending {
        int id = 0;
        std::string src;
        std::string so_far; // translation as generated so far (streaming translators)
    };

    void handle(const lt_update& up_in, int64_t utterance_id, double t_audio, clock::time_point arrived, bool closed) {
        // Translation units are whole sentences. The committer also commits
        // pieces that do not end one (it has to, to stop re-decoding a
        // run-on sentence), and translating such a piece cold gives nonsense
        // — "auf den starken Export nach | Frankreich und Italien
        // zurückzuführen." came out as "…due to strong exports." / "Caused by
        // France and Italy." So an unfinished piece waits here, shown and
        // draft-translated as part of the open text, until its sentence ends
        // (or the utterance does, or it grows past any plausible sentence).
        lt_update up;
        for (const auto& s : up_in.committed) {
            if (unit_src_.empty())
                unit_id_ = s.id;
            else
                unit_src_ += ' ';
            unit_src_ += s.text;
            if (s.complete || count_words(unit_src_) > cfg_.max_unit_words) {
                up.committed.push_back({unit_id_, unit_src_, true});
                unit_src_.clear();
            }
        }
        if (closed && !unit_src_.empty()) {
            up.committed.push_back({unit_id_, unit_src_, true});
            unit_src_.clear();
        }
        up.tail = unit_src_.empty() ? up_in.tail : (up_in.tail.empty() ? unit_src_ : unit_src_ + " " + up_in.tail);
        up.tail_changed = up_in.tail_changed;

        std::vector<job> run_now;
        if (utterance_id != utterance_text_id_) {
            utterance_text_.clear();
            utterance_text_id_ = utterance_id;
        }
        for (const auto& s : up_in.committed) {
            if (!utterance_text_.empty())
                utterance_text_ += ' ';
            utterance_text_ += s.text;
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (const auto& s : up.committed) {
                job j;
                j.id = s.id;
                j.utterance_id = utterance_id;
                j.text = s.text;
                j.t_audio = t_audio;
                j.arrived = arrived;
                pending_.push_back({s.id, s.text, std::string()});
                if (cfg_.output == lt_output::json) {
                    fprintf(cfg_.out,
                            "{\"type\":\"sentence\",\"utterance_id\":%lld,\"sentence_id\":%d,\"text\":\"%s\","
                            "\"t\":%.3f}\n",
                            (long long)utterance_id, s.id, esc(s.text).c_str(), t_audio);
                    fflush(cfg_.out);
                }
                if (cfg_.sync)
                    run_now.push_back(std::move(j));
                else
                    commits_.push_back(std::move(j));
            }
            const bool tail_moved = closed ? !tail_src_.empty() : (up.tail != tail_src_);
            if (closed || !up.committed.empty()) {
                prev_draft_.clear(); // the open sentence changed: no draft to agree with
                prev_draft_src_.clear();
                stable_text_.clear();
            }
            if (closed) {
                tail_src_.clear();
                tail_draft_.clear();
                tail_draft_stable_ = 0;
                tail_draft_src_.clear();
                have_draft_job_ = false;
            } else if (tail_moved) {
                tail_src_ = up.tail;
                // A draft stays on screen while the tail merely grows; it is
                // dropped once the tail no longer starts with what it translated.
                if (tail_src_.compare(0, tail_draft_src_.size(), tail_draft_src_) != 0 || tail_draft_src_.empty())
                    tail_draft_.clear();
                tail_draft_stable_ = 0;
                // What to translate ahead of time. If the open text already
                // contains a finished sentence (it is only waiting for a
                // second partial to agree), translate exactly that sentence:
                // when it commits a moment later the translation is already
                // there, and the wait for agreement has paid for it. That is
                // worth doing even with a slow translator. Otherwise draft
                // the whole open text, which is display only.
                const std::string candidate = first_complete_sentence(tail_src_);
                const bool speculative = !candidate.empty();
                const std::string& draft_text = speculative ? candidate : tail_src_;
                if (cfg_.drafts && (speculative || mt_avg_ms_ <= cfg_.draft_max_mt_ms) && draft_text != spec_src_ &&
                    count_words(draft_text) >= (speculative ? 1 : cfg_.draft_min_words)) {
                    job d;
                    d.draft = true;
                    d.utterance_id = utterance_id;
                    d.text = draft_text;
                    d.t_audio = t_audio;
                    d.arrived = arrived;
                    // A draft of the same sentence, grown at the end only,
                    // keeps the words the last two drafts agreed on as the
                    // fixed start of its translation. Not for a speculative
                    // draft (it may become the final), nor after the
                    // recogniser revised earlier words of the source.
                    if (!speculative && prefixed_ && !stable_text_.empty() &&
                        lt_detail::source_continues(prev_draft_src_, draft_text))
                        d.prefix = stable_text_;
                    if (cfg_.sync) {
                        run_now.push_back(std::move(d));
                    } else {
                        draft_job_ = std::move(d); // newest wins
                        have_draft_job_ = true;
                    }
                }
            }
            if (!up.committed.empty() || tail_moved)
                render_locked({});
        }
        if (cfg_.sync) {
            for (auto& j : run_now)
                execute(j);
        } else {
            cv_.notify_one();
        }
    }

    void run() {
        for (;;) {
            job j;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this] { return stop_ || !commits_.empty() || have_draft_job_; });
                if (!commits_.empty()) {
                    j = std::move(commits_.front());
                    commits_.pop_front();
                } else if (stop_) {
                    return; // drafts are not worth finishing on shutdown
                } else {
                    j = std::move(draft_job_);
                    have_draft_job_ = false;
                }
            }
            execute(j);
        }
    }

    void execute(const job& j) {
        const auto t0 = clock::now();
        // Already translated ahead of time? Then it costs nothing now.
        std::string ready;
        if (!j.draft) {
            std::lock_guard<std::mutex> lk(mu_);
            if (!spec_tr_.empty() && spec_src_ == j.text) {
                ready = spec_tr_;
                ++reused_;
            }
        }
        // A committed sentence shows its translation as it is generated; a
        // draft is replaced in one go (it is about to change anyway).
        progress_fn progress;
        if (!j.draft && cfg_.output == lt_output::tty) {
            progress = [this, &j](const std::string& so_far) {
                std::lock_guard<std::mutex> lk(mu_);
                if (!pending_.empty() && pending_.front().id == j.id) {
                    pending_.front().so_far = so_far;
                    render_locked({});
                }
            };
        }
        std::string tr;
        if (!ready.empty())
            tr = ready;
        else if (j.draft && !j.prefix.empty() && prefixed_)
            tr = prefixed_(j.text, j.prefix);
        else
            tr = translate_(j.text, progress);
        const auto t1 = clock::now();
        const double mt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double lag_ms = std::chrono::duration<double, std::milli>(t1 - j.arrived).count();

        std::lock_guard<std::mutex> lk(mu_);
        if (j.draft) {
            // Remember it whatever happens to the display: the sentence may
            // commit while this was running, and then this IS its translation.
            // (Not a draft with a fixed start: a final translation is always
            // made free, so it keeps full quality.)
            if (j.prefix.empty()) {
                spec_src_ = j.text;
                spec_tr_ = tr;
            }
            // Stale if the tail moved on to different text while we worked.
            if (tail_src_.compare(0, j.text.size(), j.text) != 0)
                return;
            // The words this draft shares with the previous draft of the same
            // growing sentence are "stable": two consecutive drafts agree on
            // them. Shown normally, the rest dimmed; JSON carries both, so a
            // consumer can show only the stable part. On German speech that
            // part is rewritten 8x less often than the whole draft
            // (normalized erasure 0.16 against 1.28, Opus-MT de-en).
            // Compared with the previous draft of the same open sentence, kept
            // even when the recogniser revised a word of the source (that is
            // when the draft on screen is dropped, and when agreement matters).
            tail_draft_stable_ = prev_draft_.empty() ? 0 : lt_detail::stable_prefix_bytes(prev_draft_, tr);
            // A fixed start is stable by construction, even where the previous
            // free draft had said something else.
            if (!j.prefix.empty() && tr.compare(0, j.prefix.size(), j.prefix) == 0)
                tail_draft_stable_ = std::max(tail_draft_stable_, j.prefix.size());
            prev_draft_ = tr;
            prev_draft_src_ = j.text;
            stable_text_ = tr.substr(0, tail_draft_stable_);
            tail_draft_ = tr;
            tail_draft_src_ = j.text;
            if (cfg_.output == lt_output::json) {
                fprintf(cfg_.out,
                        "{\"type\":\"translation_partial\",\"utterance_id\":%lld,\"text\":\"%s\","
                        "\"translation\":\"%s\",\"stable\":\"%s\",\"t\":%.3f,\"mt_ms\":%.0f,\"lag_ms\":%.0f}\n",
                        (long long)j.utterance_id, esc(j.text).c_str(), esc(tr).c_str(),
                        esc(tr.substr(0, tail_draft_stable_)).c_str(), j.t_audio, mt_ms, lag_ms);
                fflush(cfg_.out);
            }
            render_locked({});
            return;
        }
        mt_ms_.push_back(mt_ms);
        lag_ms_.push_back(lag_ms);
        mt_avg_ms_ = mt_ms_.size() == 1 ? mt_ms : 0.5 * mt_avg_ms_ + 0.5 * mt_ms;
        // Single translator, FIFO queue: results arrive in commit order, so
        // the finished sentence is always the oldest pending one.
        if (!pending_.empty() && pending_.front().id == j.id)
            pending_.pop_front();
        if (cfg_.output == lt_output::json) {
            fprintf(cfg_.out,
                    "{\"type\":\"translation\",\"utterance_id\":%lld,\"sentence_id\":%d,\"text\":\"%s\","
                    "\"translation\":\"%s\",\"source_lang\":\"%s\",\"target_lang\":\"%s\",\"t\":%.3f,"
                    "\"mt_ms\":%.0f,\"lag_ms\":%.0f}\n",
                    (long long)j.utterance_id, j.id, esc(j.text).c_str(), esc(tr).c_str(), esc(cfg_.src_lang).c_str(),
                    esc(cfg_.tgt_lang).c_str(), j.t_audio, mt_ms, lag_ms);
            fflush(cfg_.out);
            return;
        }
        render_locked({j.text, tr.empty() ? std::string("(translation failed)") : tr});
    }

    // Erase the live region, print `pair` (source, translation) permanently
    // if given, then draw the live region again. Caller holds mu_.
    void render_locked(const std::vector<std::string>& pair) {
        if (cfg_.output == lt_output::json)
            return;
        const bool tty = cfg_.output == lt_output::tty;
        std::string o;
        if (tty && live_rows_ > 0)
            o += "\r\033[" + std::to_string(live_rows_) + "A\033[J";
        live_rows_ = 0;
        if (pair.size() == 2) {
            if (tty) {
                o += "\033[2m" + tag(cfg_.src_lang) + "\033[0m" + pair[0] + "\n";
                o += "\033[2m" + tag(cfg_.tgt_lang) + "\033[0m\033[1m" + pair[1] + "\033[0m\n";
            } else {
                o += "[" + cfg_.src_lang + "] " + pair[0] + "\n[" + cfg_.tgt_lang + "] " + pair[1] + "\n";
            }
        }
        if (tty) {
            const int width = term_width();
            const int room = std::max(8, width - 5); // tag (4) + never touch the last column
            auto live = [&](const std::string& lang, const std::string& text, bool keep_end) {
                o += "\033[2m" + tag(lang) + fit(text, room, keep_end) + "\033[0m\n";
                ++live_rows_;
            };
            // Sentences committed but still with the translator. Show the
            // newest few; the queue only grows when the translator is slower
            // than the speaker.
            const size_t first = pending_.size() > 3 ? pending_.size() - 3 : 0;
            for (size_t i = first; i < pending_.size(); ++i)
                live(cfg_.src_lang, pending_[i].src, false);
            if (!pending_.empty())
                live(cfg_.tgt_lang, pending_.front().so_far + "\xE2\x80\xA6", true);
            if (!tail_src_.empty()) {
                live(cfg_.src_lang, tail_src_, true);
                if (!tail_draft_.empty()) {
                    // Stable words normal, the still-changing rest dimmed
                    // (only when the line fits; a truncated line stays dim).
                    if (tail_draft_stable_ > 0 && fit(tail_draft_, room, true) == tail_draft_) {
                        o += "\033[2m" + tag(cfg_.tgt_lang) + "\033[0m" + tail_draft_.substr(0, tail_draft_stable_) +
                             "\033[2m" + tail_draft_.substr(tail_draft_stable_) + "\033[0m\n";
                        ++live_rows_;
                    } else {
                        live(cfg_.tgt_lang, tail_draft_, true);
                    }
                }
            }
        }
        if (!o.empty()) {
            fwrite(o.data(), 1, o.size(), cfg_.out);
            fflush(cfg_.out);
        }
    }

    static std::string tag(const std::string& lang) {
        std::string t = lang.substr(0, 3);
        t.resize(4, ' ');
        return t;
    }

    static int term_width() {
#if defined(_WIN32)
        CONSOLE_SCREEN_BUFFER_INFO info;
        if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &info))
            return info.srWindow.Right - info.srWindow.Left + 1;
#else
        struct winsize ws;
        if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0)
            return ws.ws_col;
#endif
        return 80;
    }

    // Columns a code point occupies: wide for East Asian ranges, zero for
    // combining marks. An over-estimate only truncates a little early.
    static int cp_width(uint32_t cp) {
        if (cp >= 0x300 && cp <= 0x36F)
            return 0;
        return cp >= 0x1100 ? 2 : 1;
    }

    // Truncate to `cols` columns so a live line never wraps (the redraw
    // counts one row per line). keep_end shows the newest text.
    static std::string fit(const std::string& s, int cols, bool keep_end) {
        struct piece {
            size_t off, len;
            int w;
        };
        std::vector<piece> ps;
        int total = 0;
        for (size_t i = 0; i < s.size();) {
            const unsigned char c = s[i];
            size_t n = lt_detail::u8_len(c);
            if (i + n > s.size())
                n = s.size() - i;
            uint32_t cp = c;
            if (n == 2)
                cp = ((c & 0x1F) << 6) | (s[i + 1] & 0x3F);
            else if (n == 3)
                cp = ((c & 0x0F) << 12) | ((s[i + 1] & 0x3F) << 6) | (s[i + 2] & 0x3F);
            else if (n == 4)
                cp = 0x10000;
            const int w = cp_width(cp);
            ps.push_back({i, n, w});
            total += w;
            i += n;
        }
        if (total <= cols)
            return s;
        const int budget = cols - 1; // one column for the ellipsis
        std::string out;
        if (keep_end) {
            int used = 0;
            size_t k = ps.size();
            while (k > 0 && used + ps[k - 1].w <= budget)
                used += ps[--k].w;
            out = "\xE2\x80\xA6" + s.substr(ps[k].off);
        } else {
            int used = 0;
            size_t k = 0;
            while (k < ps.size() && used + ps[k].w <= budget)
                used += ps[k++].w;
            out = s.substr(0, k < ps.size() ? ps[k].off : s.size()) + "\xE2\x80\xA6";
        }
        return out;
    }

    // The first sentence of `text` that is finished AND followed by more
    // text — i.e. the one the committer will commit next if the recogniser
    // does not change its mind. Empty when there is none.
    static std::string first_complete_sentence(const std::string& text) {
        const std::vector<lt_word> w = lt_split_words(text);
        for (size_t i = 0; i + 1 < w.size(); ++i)
            if (lt_ends_sentence(w, i))
                return lt_join(w, 0, i + 1);
        return std::string();
    }

    static int count_words(const std::string& s) {
        int n = 0;
        bool in = false;
        for (unsigned char c : s) {
            const bool sp = c == ' ' || c == '\t' || c == '\n';
            if (!sp && !in)
                ++n;
            in = !sp;
        }
        return n;
    }

    static std::string esc(const std::string& s) {
        std::string o;
        o.reserve(s.size() + 8);
        for (unsigned char c : s) {
            switch (c) {
            case '"':
                o += "\\\"";
                break;
            case '\\':
                o += "\\\\";
                break;
            case '\n':
                o += "\\n";
                break;
            case '\r':
                o += "\\r";
                break;
            case '\t':
                o += "\\t";
                break;
            default:
                if (c < 0x20) {
                    char b[8];
                    snprintf(b, sizeof(b), "\\u%04x", c);
                    o += b;
                } else {
                    o += (char)c;
                }
            }
        }
        return o;
    }

    static double pct(std::vector<double> v, double q) {
        if (v.empty())
            return 0;
        std::sort(v.begin(), v.end());
        return v[std::min(v.size() - 1, (size_t)(q * (double)v.size()))];
    }

    lt_sink_config cfg_;
    translate_fn translate_;
    lt_committer committer_;     // caller's thread only
    std::string utterance_text_; // caller's thread only: sentences committed in the open utterance
    std::string unit_src_;       // caller's thread only: committed pieces of a sentence not yet finished
    int unit_id_ = 0;
    int64_t utterance_text_id_ = -1;

    std::mutex mu_; // everything below, and all writes to cfg_.out
    std::condition_variable cv_;
    std::deque<job> commits_;
    job draft_job_;
    bool have_draft_job_ = false;
    bool stop_ = false;
    bool finished_ = false;
    std::deque<pending> pending_;
    std::string tail_src_;
    std::string tail_draft_;
    std::string tail_draft_src_;
    size_t tail_draft_stable_ = 0; // bytes of tail_draft_ that the previous draft agreed on
    std::string prev_draft_;       // previous draft of the open sentence, for tail_draft_stable_
    std::string prev_draft_src_;   // the source text prev_draft_ translated
    std::string stable_text_;      // the stable start of prev_draft_: the next draft's fixed start
    prefixed_translate_fn prefixed_;
    std::string spec_src_; // last text translated ahead of time, and its translation
    std::string spec_tr_;
    int reused_ = 0;
    int live_rows_ = 0;
    std::vector<double> mt_ms_;
    double mt_avg_ms_ = 0.0; // running average over committed sentences
    std::vector<double> lag_ms_;
    std::thread worker_;
};

} // namespace crispasr
