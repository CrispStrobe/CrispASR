#include "zonos3-sampler.h"

#include <algorithm>
#include <cmath>
#include <functional>

zonos3_sampler::zonos3_sampler(const zonos3_sampling& s, const zonos3_hparams& hp)
    : sp(s), rng(s.seed), n_cb((int)hp.n_codebooks), vocab((int)hp.audio_vocab), codebook_size((int)hp.codebook_size),
      pad_id((int)hp.audio_pad_id), stop_id((int)hp.eoa_id), hist(hp.n_codebooks) {
    logit.resize(vocab);
    tmp.resize(vocab);
    p.reserve(vocab);
    idx.reserve(vocab);
    ord.reserve(vocab);
    keep.reserve(vocab);
    seen.assign(vocab, 0);
}

void zonos3_sampler::accept(const std::vector<int>& frame) {
    for (int cb = 0; cb < n_cb && cb < (int)frame.size(); ++cb) {
        hist[cb].push_back(frame[cb]);
        while ((int)hist[cb].size() > sp.rep_window)
            hist[cb].pop_front();
    }
    ++generated;
}

// Mirrors the reference tts/sampler.py: pad and foreign special ids masked, repetition penalty (each repeated
// ordinary code once), temperature, top-k with ties kept, softmax, top-p on the mass *before*
// each token, min-p, multinomial.
int zonos3_sampler::sample(const float* in, int cb) {
    std::copy(in, in + vocab, logit.begin());
    // the audio pad is never a supervised target; special ids other than the request's stop id are never sampled,
    // and the stop id waits for min_frames
    for (int i = codebook_size; i < vocab; ++i)
        if (i == pad_id || i != stop_id || generated < min_frames + cb)
            logit[i] = -INFINITY;
    if (stop_bias != 0.0f && stop_id < vocab)
        logit[stop_id] += stop_bias; // -inf stays -inf (min_frames)

    // repetition penalty over this codebook's recent ordinary codes (first rep_codebooks only)
    if (sp.rep_penalty != 1.0f && cb < sp.rep_codebooks) {
        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0u);
            seen_gen = 1;
        } // generation wrap
        for (int t : hist[cb]) {
            if (t >= 0 && t < codebook_size && seen[t] != seen_gen) {
                seen[t] = seen_gen;
                logit[t] = logit[t] > 0.0f ? logit[t] / sp.rep_penalty : logit[t] * sp.rep_penalty;
            }
        }
    }

    if (sp.greedy) {
        int best = 0;
        float bv = logit[0];
        for (int i = 1; i < vocab; ++i)
            if (logit[i] > bv) {
                bv = logit[i];
                best = i;
            }
        return best;
    }

    if (sp.temperature > 0.0f)
        for (auto& l : logit)
            l /= sp.temperature;

    // candidate set: every logit >= the k-th largest (ties at the threshold are kept)
    idx.clear();
    if (sp.top_k > 0 && sp.top_k < vocab) {
        std::copy(logit.begin(), logit.end(), tmp.begin());
        std::nth_element(tmp.begin(), tmp.begin() + (sp.top_k - 1), tmp.end(), std::greater<float>());
        const float kth = tmp[sp.top_k - 1];
        for (int i = 0; i < vocab; ++i)
            if (logit[i] >= kth)
                idx.push_back(i);
    } else {
        for (int i = 0; i < vocab; ++i)
            idx.push_back(i);
    }

    // softmax over candidates
    float mx = -INFINITY;
    for (int i : idx)
        mx = std::max(mx, logit[i]);
    p.resize(idx.size());
    double sum = 0;
    for (size_t j = 0; j < idx.size(); ++j) {
        p[j] = std::exp(logit[idx[j]] - mx);
        sum += p[j];
    }
    for (auto& x : p)
        x = (float)(x / sum);

    // order by prob desc (for top-p) and find max prob (for min-p)
    ord.resize(idx.size());
    for (size_t j = 0; j < ord.size(); ++j)
        ord[j] = (int)j;
    std::sort(ord.begin(), ord.end(), [&](int a, int b) { return p[a] > p[b]; });

    keep.assign(idx.size(), 1);
    if (sp.top_p > 0.0f && sp.top_p < 1.0f) {
        // The original CPU cumsum accumulates in double, stores each prefix
        // as float, then subtracts that column in float before comparing p.
        double cumulative = 0;
        for (int r : ord) {
            cumulative += p[r];
            const float rounded_prefix = static_cast<float>(cumulative);
            const float preceding_mass = rounded_prefix - p[r];
            if (preceding_mass > sp.top_p)
                keep[r] = 0;
        }
    }
    if (sp.min_p > 0.0f) {
        const float thr = sp.min_p * p[ord[0]];
        for (size_t j = 0; j < p.size(); ++j)
            if (p[j] < thr)
                keep[j] = 0;
    }

    double s2 = 0;
    for (size_t j = 0; j < p.size(); ++j) {
        if (!keep[j])
            p[j] = 0;
        s2 += p[j];
    }
    if (s2 <= 0)
        return idx[ord[0]]; // greedy fallback

    std::uniform_real_distribution<double> u(0.0, s2);
    double r = u(rng), acc = 0;
    for (size_t j = 0; j < p.size(); ++j) {
        acc += p[j];
        if (acc >= r)
            return idx[j];
    }
    return idx[ord[0]];
}
