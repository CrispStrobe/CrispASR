// Prompt layout: [speaker row | raw delayed reference prefix] [condition rows] BOS text EOS [silence rows].
#include "zonos3.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace {

bool is_pad_row(const zonos3_model& m, const float* row) {
    const int ncb = (int)m.hp.n_codebooks;
    for (int c = 0; c < ncb; ++c)
        if (row[c] != (float)m.hp.audio_pad_id)
            return false;
    return row[ncb] == (float)m.hp.text_vocab;
}

} // namespace

int zonos3_edit_marker_rows(const zonos3_model& model) {
    const auto& hp = model.hp;
    if (model.tok.block_rows == 0)
        return (int)hp.text_extra_tokens;
    const int markers = model.tok.block_first - (int)hp.text_vocab - 1;
    return markers + model.tok.block_rows == (int)hp.text_extra_tokens ? markers : -1;
}

bool zonos3_build_prompt(const zonos3_model& model, const std::string& text, const zonos3_prompt_options& opt,
                         zonos3_prompt& out, std::string* err) {
    out = zonos3_prompt();
    const auto& hp = model.hp;
    const int ncb = (int)hp.n_codebooks, W = ncb + 1;
    auto bad = [&](const std::string& msg) {
        if (err)
            *err = msg;
        else
            fprintf(stderr, "zonos3: %s\n", msg.c_str());
        return false;
    };
    if (opt.speaker_embedding && (opt.reference_codes || opt.reference_frames))
        return bad("a speaker embedding and a reference prefix are mutually exclusive");
    // Internal bounds: request resolvers keep the first reference_max_frames frames and refuse longer prefills.
    const zonos3_generation_registry& limits = zonos3_generation();
    if (opt.reference_frames < 0 || opt.reference_frames > limits.reference_max_frames ||
        (opt.reference_frames > 0) != (opt.reference_codes != nullptr))
        return bad("invalid reference-code prefix");
    if (opt.silence_prefill_frames < 0 || opt.silence_prefill_frames > limits.silence_prefill_max_frames)
        return bad("silence prefill must be 0.." + std::to_string(limits.silence_prefill_max_frames) + " frames");
    if (opt.speaker_embedding) {
        for (uint32_t i = 0; i < hp.speaker_dim; ++i)
            if (!std::isfinite(opt.speaker_embedding[i]))
                return bad("speaker embedding contains a non-finite value");
    }
    std::vector<int32_t> buckets = opt.condition_buckets;
    if (buckets.empty())
        buckets.assign(hp.cond_bucket_counts.begin(), hp.cond_bucket_counts.end());
    if (buckets.size() != hp.n_cond)
        return bad("expected " + std::to_string(hp.n_cond) + " condition buckets");
    for (uint32_t i = 0; i < hp.n_cond; ++i) {
        if (buckets[i] < 0 || buckets[i] > (int32_t)hp.cond_bucket_counts[i])
            return bad("condition bucket " + std::to_string(buckets[i]) + " for " + hp.cond_names[i] + " outside 0.." +
                       std::to_string(hp.cond_bucket_counts[i] - 1) + " (" + std::to_string(hp.cond_bucket_counts[i]) +
                       " = unknown)");
    }
    const std::string groups = zonos3_check_text_markers(model.tok, text);
    if (!groups.empty())
        return bad(groups);
    std::vector<int32_t> ids;
    std::string tok_err;
    if (!zonos3_text_ids(model.tok, text, ids, tok_err))
        return bad("Pronunciation groups: " + tok_err);
    const size_t reference_rows = opt.reference_frames ? (size_t)opt.reference_frames + ncb - 1 : 0;
    const size_t speaker_rows = opt.speaker_embedding ? 1 : reference_rows;
    const size_t prefill = (size_t)opt.silence_prefill_frames;
    const size_t rows = speaker_rows + hp.n_cond + ids.size() + prefill;
    if (rows == 0 || rows > hp.n_ctx_train)
        return bad("prompt exceeds the " + std::to_string(hp.n_ctx_train) + "-position context");
    for (size_t i = 0; i < (size_t)opt.reference_frames * ncb; ++i) {
        if (opt.reference_codes[i] < 0 || opt.reference_codes[i] >= (int32_t)hp.codebook_size)
            return bad("reference contains a nonordinary audio code");
    }
    std::vector<int32_t> rowsv(rows * W, (int32_t)hp.audio_pad_id);
    for (size_t t = 0; t < rows; ++t)
        rowsv[t * W + ncb] = (int32_t)hp.text_vocab;
    for (size_t t = 0; t < reference_rows; ++t) {
        for (int cb = 0; cb < ncb; ++cb) {
            const int frame = (int)t - cb;
            if (frame >= 0 && frame < opt.reference_frames)
                rowsv[t * W + cb] = opt.reference_codes[(size_t)frame * ncb + cb];
        }
    }
    const size_t text_start = speaker_rows + hp.n_cond;
    for (size_t t = 0; t < ids.size(); ++t)
        rowsv[(text_start + t) * W + ncb] = ids[t];
    const size_t silence_start = text_start + ids.size();
    for (size_t k = 0; k < prefill; ++k)
        for (int cb = 0; cb < ncb; ++cb)
            if ((int)k - cb >= 0)
                rowsv[(silence_start + k) * W + cb] = hp.silence_frame[cb];

    out.ids = std::move(rowsv);
    out.n_rows = (int)rows;
    out.speaker_rows = (int)speaker_rows;
    out.prefill_frames = (int)prefill;
    out.cond.condition_row = (int)speaker_rows;
    out.cond.buckets = std::move(buckets);
    if (opt.speaker_embedding) {
        out.cond.speaker_row = 0;
        out.cond.speaker.assign(opt.speaker_embedding, opt.speaker_embedding + hp.speaker_dim);
    }
    return true;
}

bool zonos3_build_cfg_twin(const zonos3_model& model, const zonos3_prompt& cond, zonos3_prompt& out) {
    if (cond.speaker_rows <= 0) {
        fprintf(stderr, "zonos3: speaker CFG needs a speaker prompt (speaker embedding or reference)\n");
        return false;
    }
    const size_t W = model.hp.n_codebooks + 1;
    if ((size_t)cond.speaker_rows >= (size_t)cond.n_rows || cond.ids.size() != (size_t)cond.n_rows * W) {
        fprintf(stderr, "zonos3: inconsistent prompt for speaker CFG\n");
        return false;
    }
    zonos3_prompt twin;
    twin.ids.assign(cond.ids.begin() + (size_t)cond.speaker_rows * W, cond.ids.end());
    twin.n_rows = cond.n_rows - cond.speaker_rows;
    twin.speaker_rows = 0;
    twin.prefill_frames = cond.prefill_frames;
    twin.cond.speaker_row = -1;
    twin.cond.condition_row = cond.cond.condition_row - cond.speaker_rows;
    twin.cond.buckets.assign(model.hp.cond_bucket_counts.begin(), model.hp.cond_bucket_counts.end());
    out = std::move(twin);
    return true;
}

int zonos3_condition_index(const zonos3_model& model, const std::string& name) {
    for (size_t i = 0; i < model.hp.cond_names.size(); ++i)
        if (model.hp.cond_names[i] == name)
            return (int)i;
    return -1;
}

int zonos3_condition_bucket(const zonos3_model& model, int index, double value) {
    if (index < 0 || index >= (int)model.hp.cond_edges.size() || !std::isfinite(value))
        return -1;
    // bisect.bisect_right(edges, value): number of edges <= value.
    const auto& edges = model.hp.cond_edges[index];
    int bucket = 0;
    while (bucket < (int)edges.size() && (double)edges[bucket] <= value)
        ++bucket;
    return bucket;
}

std::vector<int32_t> zonos3_prefill_codes(const zonos3_model& model, const zonos3_prompt& p) {
    const size_t ncb = model.hp.n_codebooks, W = ncb + 1;
    std::vector<int32_t> codes;
    codes.reserve((size_t)p.prefill_frames * ncb);
    for (int k = 0; k < p.prefill_frames; ++k) {
        const int32_t* row = p.ids.data() + (size_t)(p.n_rows - p.prefill_frames + k) * W;
        codes.insert(codes.end(), row, row + ncb);
    }
    return codes;
}

bool zonos3_check_conditioning(const zonos3_model& model, const float* ids, int n_rows, const zonos3_conditioning& c) {
    const auto& hp = model.hp;
    const size_t W = hp.n_codebooks + 1;
    auto pad_row = [&](int row) {
        if (row < 0 || row >= n_rows || !is_pad_row(model, ids + (size_t)row * W)) {
            fprintf(stderr, "zonos3: conditioning row %d is not an all-pad row\n", row);
            return false;
        }
        return true;
    };
    if (c.speaker_row >= 0) {
        if (c.speaker.size() != hp.speaker_dim) {
            fprintf(stderr, "zonos3: expected a %u-value speaker embedding\n", hp.speaker_dim);
            return false;
        }
        for (float v : c.speaker)
            if (!std::isfinite(v)) {
                fprintf(stderr, "zonos3: speaker embedding contains a non-finite value\n");
                return false;
            }
        if (!pad_row(c.speaker_row))
            return false;
    }
    if (c.condition_row >= 0) {
        if (c.buckets.size() != hp.n_cond) {
            fprintf(stderr, "zonos3: expected %u condition buckets\n", hp.n_cond);
            return false;
        }
        for (uint32_t i = 0; i < hp.n_cond; ++i) {
            if (c.buckets[i] < 0 || c.buckets[i] > (int32_t)hp.cond_bucket_counts[i]) {
                fprintf(stderr, "zonos3: condition bucket %d for %s outside 0..%u (%u = unknown)\n", c.buckets[i],
                        hp.cond_names[i].c_str(), hp.cond_bucket_counts[i] - 1, hp.cond_bucket_counts[i]);
                return false;
            }
            if (!pad_row(c.condition_row + (int)i))
                return false;
        }
    }
    for (int r = 0; r < n_rows; ++r) {
        if (!is_pad_row(model, ids + (size_t)r * W))
            continue;
        const bool speaker = c.speaker_row >= 0 && r == c.speaker_row;
        const bool cond = c.condition_row >= 0 && r >= c.condition_row && r < c.condition_row + (int)hp.n_cond;
        if (!speaker && !cond) {
            fprintf(stderr, "zonos3: all-pad row %d has no speaker/condition embedding\n", r);
            return false;
        }
    }
    return true;
}
