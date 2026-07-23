// tiron_link.cpp — see tiron_link.h.

#include "tiron_link.h"

#include "crispasr_speaker_cluster.h"
#include "crispasr_speaker_embedder.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <regex>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int64_t SR = 16000;                // Tiron / TitaNet operate at 16 kHz mono
constexpr int64_t SAMPLES_PER_CS = SR / 100; // 160
constexpr int64_t MIN_EMBED_SAMPLES = 1600;  // 0.1 s floor: below this even an attach embed is noise

float dot(const float* a, const float* b, int d) {
    float s = 0.0f;
    for (int i = 0; i < d; i++)
        s += a[i] * b[i];
    return s;
}

// A (window_id, local_speaker) bucket of turns that share one physical speaker
// within a window (Tiron's within-window must-link). local_speaker <= 0 turns
// each get a private key so they never falsely must-link.
struct Group {
    std::vector<int> turns; // indices into the input turns[]
    int64_t audio_cs = 0;   // total spanned audio (centiseconds)
    bool has_embed = false; // an embedding was computed
    bool is_spine = false;  // audio_cs >= min_embed_cs AND has_embed
    int global = -1;        // assigned meeting-level speaker id
    int emb_off = -1;       // row offset into `embeds` when has_embed
};

} // namespace

TironLinkResult crispasr_tiron_link_speakers(const std::vector<TironTurn>& turns, const float* pcm_16k, int n_samples,
                                             CrispasrSpeakerEmbedder* embedder, const TironLinkOptions& opts) {
    TironLinkResult res;
    if (turns.empty())
        return res;

    // ── 1. group turns by (window_id, local_speaker), first-appearance order ──
    std::vector<Group> groups;
    std::map<std::pair<int, int>, int> key_to_group; // only for local_speaker > 0
    int private_counter = -1;                        // unique keys for local_speaker <= 0
    for (int i = 0; i < (int)turns.size(); i++) {
        const auto& t = turns[i];
        int gi;
        if (t.local_speaker > 0) {
            const std::pair<int, int> key{t.window_id, t.local_speaker};
            auto it = key_to_group.find(key);
            if (it == key_to_group.end()) {
                gi = (int)groups.size();
                key_to_group.emplace(key, gi);
                groups.emplace_back();
            } else {
                gi = it->second;
            }
        } else {
            // no local hint → its own group (linked only by embedding)
            key_to_group.emplace(std::make_pair(t.window_id, private_counter--), (int)groups.size());
            gi = (int)groups.size();
            groups.emplace_back();
        }
        groups[gi].turns.push_back(i);
        groups[gi].audio_cs += std::max<int64_t>(0, t.t1_cs - t.t0_cs);
    }

    const int dim = embedder ? embedder->dim() : 0;

    // ── 2. embed each group (aggregate its audio spans, capped) ──
    std::vector<float> embeds; // row-major, one row per has_embed group
    if (embedder && dim > 0 && pcm_16k && n_samples > 0) {
        std::vector<float> buf; // aggregated PCM for the current group
        std::vector<float> vec(dim);
        for (auto& g : groups) {
            buf.clear();
            const int64_t cap = opts.max_embed_cs * SAMPLES_PER_CS;
            for (int ti : g.turns) {
                if ((int64_t)buf.size() >= cap)
                    break;
                int64_t s0 = std::max<int64_t>(0, turns[ti].t0_cs * SAMPLES_PER_CS);
                int64_t s1 = std::min<int64_t>(n_samples, turns[ti].t1_cs * SAMPLES_PER_CS);
                if (s1 <= s0)
                    continue;
                s1 = std::min<int64_t>(s1, s0 + (cap - (int64_t)buf.size()));
                buf.insert(buf.end(), pcm_16k + s0, pcm_16k + s1);
            }
            if ((int64_t)buf.size() < MIN_EMBED_SAMPLES)
                continue;
            if (!embedder->embed(buf.data(), (int)buf.size(), vec.data()))
                continue;
            g.has_embed = true;
            g.is_spine = g.audio_cs >= opts.min_embed_cs;
            g.emb_off = (int)(embeds.size() / (size_t)dim);
            embeds.insert(embeds.end(), vec.begin(), vec.end());
        }
    }

    // ── 3. cluster the SPINE groups (they carry a clean voiceprint) ──
    // Build a compact spine embedding matrix and remember which group each row is.
    std::vector<int> spine_groups; // group index per spine row
    std::vector<float> spine_emb;
    for (int gi = 0; gi < (int)groups.size(); gi++) {
        if (groups[gi].is_spine) {
            spine_groups.push_back(gi);
            const float* row = &embeds[(size_t)groups[gi].emb_off * dim];
            spine_emb.insert(spine_emb.end(), row, row + dim);
        }
    }

    int n_spine_clusters = 0;
    std::vector<float> centroids;
    if (!spine_groups.empty() && dim > 0) {
        std::vector<int> labels = crispasr_agglomerative_cluster(spine_emb, (int)spine_groups.size(), dim,
                                                                 opts.merge_threshold, opts.max_speakers);
        for (size_t k = 0; k < spine_groups.size(); k++) {
            const int c = labels[k];
            if (c < 0)
                continue;
            groups[spine_groups[k]].global = c;
            n_spine_clusters = std::max(n_spine_clusters, c + 1);
        }
        centroids = crispasr_cluster_centroids(spine_emb, labels, (int)spine_groups.size(), dim);
    }

    // ── 4. assign the remaining (non-spine) groups ──
    // A turn's mid time, for the temporal fallback.
    auto group_mid_cs = [&](const Group& g) -> int64_t {
        int64_t lo = INT64_MAX, hi = INT64_MIN;
        for (int ti : g.turns) {
            lo = std::min(lo, turns[ti].t0_cs);
            hi = std::max(hi, turns[ti].t1_cs);
        }
        return (lo + hi) / 2;
    };

    if (n_spine_clusters > 0) {
        const float attach_thr = opts.merge_threshold - opts.attach_margin;
        for (int gi = 0; gi < (int)groups.size(); gi++) {
            Group& g = groups[gi];
            if (g.global >= 0)
                continue;
            // 4a. acoustic attach to the nearest spine centroid, if confident.
            if (g.has_embed && dim > 0) {
                const float* row = &embeds[(size_t)g.emb_off * dim];
                int best = -1;
                float best_cos = -2.0f;
                for (int c = 0; c < n_spine_clusters; c++) {
                    const float cs = dot(row, &centroids[(size_t)c * dim], dim);
                    if (cs > best_cos) {
                        best_cos = cs;
                        best = c;
                    }
                }
                if (best >= 0 && best_cos >= attach_thr) {
                    g.global = best;
                    continue;
                }
            }
            // 4b. temporal fallback: borrow the id of the nearest spine group in time.
            const int64_t mid = group_mid_cs(g);
            int best = -1;
            int64_t best_dist = INT64_MAX;
            for (int sg : spine_groups) {
                const int64_t d = std::llabs(group_mid_cs(groups[sg]) - mid);
                if (d < best_dist) {
                    best_dist = d;
                    best = groups[sg].global;
                }
            }
            g.global = best >= 0 ? best : 0;
        }
        res.n_speakers = n_spine_clusters;
        res.centroids = std::move(centroids);
        res.dim = dim;
    } else {
        // No spine (no embedder, or every group too short): no acoustic linking
        // is possible, so each distinct group is its own id in first-appearance
        // order. This is the conservative, deterministic degradation.
        int next = 0;
        for (auto& g : groups)
            g.global = next++;
        res.n_speakers = next;
        res.dim = dim;
        if (dim > 0) {
            res.centroids.assign((size_t)next * dim, 0.0f);
            for (int gi = 0; gi < (int)groups.size(); gi++)
                if (groups[gi].has_embed)
                    std::copy_n(&embeds[(size_t)groups[gi].emb_off * dim], dim,
                                &res.centroids[(size_t)groups[gi].global * dim]);
        }
    }

    // ── 5. scatter group ids back onto turns ──
    res.turn_speaker.assign(turns.size(), -1);
    for (const auto& g : groups)
        for (int ti : g.turns)
            res.turn_speaker[ti] = g.global;

    return res;
}

int crispasr_tiron_link_transcript(std::vector<TironTranscriptSeg>& segs, const float* pcm_16k, int n_samples,
                                   const char* embedder_spec, int n_threads, const char* cache_dir) {
    static const std::regex spk_re(R"(<\|speaker(\d+)\|>)");

    bool is_tiron = false;
    for (const auto& s : segs) {
        if (std::regex_search(s.text, spk_re)) {
            is_tiron = true;
            break;
        }
    }
    if (!is_tiron)
        return -1;

    // No embedder requested → leave the window-local markers in place.
    const std::string spec = embedder_spec ? embedder_spec : "";
    if (spec.empty() || !pcm_16k || n_samples <= 0)
        return 0;

    auto embedder = crispasr_make_speaker_embedder(spec, n_threads, cache_dir ? cache_dir : "");
    if (!embedder)
        return 0;

    std::vector<TironTurn> turns;
    std::vector<int> turn_seg;
    int cur_local = 0;
    for (int si = 0; si < (int)segs.size(); si++) {
        const std::string& t = segs[si].text;
        for (auto it = std::sregex_iterator(t.begin(), t.end(), spk_re); it != std::sregex_iterator(); ++it) {
            cur_local = std::stoi((*it)[1].str());
        }
        std::string content = std::regex_replace(t, spk_re, "");
        const bool has_word =
            std::any_of(content.begin(), content.end(), [](unsigned char c) { return std::isalnum(c); });
        if (!has_word)
            continue;
        const int window = segs[si].chunk_id >= 0 ? segs[si].chunk_id : (int)(segs[si].t0_cs / 3000);
        turns.push_back(TironTurn{segs[si].t0_cs, segs[si].t1_cs, window, cur_local});
        turn_seg.push_back(si);
    }
    if (turns.empty())
        return 0;

    TironLinkResult res = crispasr_tiron_link_speakers(turns, pcm_16k, n_samples, embedder.get());
    for (int k = 0; k < (int)turns.size(); k++) {
        const int g = res.turn_speaker[k] < 0 ? 0 : res.turn_speaker[k];
        char lbl[24];
        snprintf(lbl, sizeof(lbl), "SPEAKER_%02d ", g);
        segs[turn_seg[k]].speaker = lbl;
    }
    // Strip the local markers (global labels now carry attribution) and mark
    // bare-marker segments for the caller to drop.
    for (auto& s : segs) {
        s.text = std::regex_replace(s.text, spk_re, "");
        s.drop = !std::any_of(s.text.begin(), s.text.end(), [](unsigned char c) { return std::isalnum(c); });
    }
    return res.n_speakers;
}
