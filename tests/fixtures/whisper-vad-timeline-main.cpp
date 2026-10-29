
static void check_mapping_cleared(const whisper_state& state) {
    require(!state.has_vad_segments && state.vad_mapping_table.empty() && state.vad_segments.empty(),
            "previous VAD timeline survived plain decoding");
}
int main(int argc, char** argv) {
    require(argc == 2, "select a case");
    std::string test = argv[1];
    whisper_state state;
    whisper_context ctx{&state};
    whisper_full_params params{};
    std::vector<float> pcm(7 * CRISPASR_SAMPLE_RATE, .25f);
    if (test == "state-owner") {
        whisper_state other;
        state.vad_segments.push_back({1, 2, 3, 4});
        std::vector<float> filtered;
        require(whisper_vad(&ctx, &other, params, pcm.data(), int(pcm.size()), filtered), "state VAD failed");
        require(other.vad_segments.size() == 2 && other.has_vad_segments, "VAD updated wrong state's segments");
        require(state.vad_segments.size() == 1 && state.vad_segments[0].orig_start == 1 && !state.has_vad_segments,
                "VAD modified the unrelated context default state");
        return 0;
    }
    if (test == "bounded-state" || test == "bounded-sequential") {
        params.offset_ms = 1000;
        params.duration_ms = 2000;
        tracked_pcm = pcm.data();
        int rc = test == "bounded-state" ? whisper_full_with_state(&ctx, &state, params, pcm.data(), int(pcm.size()))
                                         : whisper_full(&ctx, params, pcm.data(), int(pcm.size()));
        require(rc == 0, "bounded decode failed");
        require(decode_calls[0].count == 3 * CRISPASR_SAMPLE_RATE,
                "sequential feature extraction can see audio after requested endpoint");
        require(decode_calls[0].offset_ms == 1000 && decode_calls[0].duration_ms == 2000,
                "sequential decoder lost requested seek interval");
        return 0;
    }
    if (test == "precomputed-mel") {
        allow_precomputed_mel = true;
        params.offset_ms = 1000;
        params.duration_ms = 2000;
        require(whisper_full_with_state(&ctx, &state, params, nullptr, 0) == 0 && completed == 1,
                "PCM bounding broke precomputed mel decoding");
        require(whisper_full(&ctx, params, nullptr, 0) == 0 && completed == 2,
                "sequential wrapper rejected precomputed mel entry");
        params.vad = true;
        require(whisper_full(nullptr, params, pcm.data(), int(pcm.size())) < 0, "null context reached VAD prepass");
        require(whisper_full_with_state(&ctx, nullptr, params, pcm.data(), int(pcm.size())) < 0,
                "null direct state accepted");
        params.duration_ms = -1;
        require(whisper_full(&ctx, params, pcm.data(), int(pcm.size())) < 0 && vad_calls == 0,
                "invalid duration reached VAD prepass");
        return 0;
    }
    if (test == "seek-range") {
        params.offset_ms = 1000;
        params.duration_ms = 2000;
        require(native_seek_range(params, 120) == std::pair<int, int>{100, 120}, "duration extends past available mel");
        params.duration_ms = 2147483647;
        require(native_seek_range(params, 300) == std::pair<int, int>{100, 300}, "large duration extends past mel");
        params.duration_ms = 0;
        require(native_seek_range(params, 800) == std::pair<int, int>{100, 800}, "unbounded seek changed");
        return 0;
    }
    auto stitch = [&]() {
        std::vector<float> filtered;
        require(whisper_vad(&ctx, &state, params, pcm.data(), int(pcm.size()), filtered), "VAD stitch failed");
        require(filtered.size() == 210 * CRISPASR_SAMPLE_RATE / 100, "unexpected stitched PCM length");
        require(filtered[0] == .25f && filtered[16000] == 0 && filtered[17600] == .25f,
                "speech/silence PCM placement changed");
        require(state.vad_segments.size() == 2 && state.has_vad_segments, "speech map missing");
    };
    stitch();
    whisper_segment seg{};
    seg.t0 = 105;
    seg.t1 = 115;
    whisper_token_data token{};
    token.id = 7;
    token.p = .75f;
    token.t0 = 105;
    token.t1 = 115;
    token.t_dtw = 110;
    seg.tokens.push_back(token);
    token.t0 = token.t1 = token.t_dtw = -1;
    seg.tokens.push_back(token);
    state.result_all.push_back(seg);
    if (test == "getters") {
        require(whisper_full_get_segment_t0_from_state(&state, 0) == 350 &&
                    whisper_full_get_segment_t1_from_state(&state, 0) == 505,
                "segment map reference changed");
        for (int repeat = 0; repeat < 3; ++repeat) {
            auto a = whisper_full_get_token_data_from_state(&state, 0, 0);
            auto b = whisper_full_get_token_data(&ctx, 0, 0);
            require(a.t0 == 350 && a.t1 == 505 && a.t_dtw == 500, "state token/DTW getters return filtered time");
            require(b.t0 == a.t0 && b.t1 == a.t1 && b.t_dtw == a.t_dtw, "context/state token getters disagree");
            require(crispasr_token_t0(&ctx, 0, 0) == 350 && crispasr_token_t1(&ctx, 0, 0) == 505 &&
                        crispasr_token_dtw_t(&ctx, 0, 0) == 500,
                    "scalar wrapper adapters return filtered time");
            require(a.id == 7 && a.p == .75f, "getter damaged token metadata");
            require(state.result_all[0].tokens[0].t0 == 105 && state.result_all[0].tokens[0].t_dtw == 110,
                    "getter mutated storage (would double-map later reads)");
        }
        return 0;
    }
    if (test == "sentinels") {
        auto missing = whisper_full_get_token_data(&ctx, 0, 1);
        require(missing.t0 == -1 && missing.t1 == -1 && missing.t_dtw == -1, "missing-time sentinel mapped");
        state.result_all[0].tokens[1].t1 = 115;
        auto partial = whisper_full_get_token_data(&ctx, 0, 1);
        require(partial.t0 == -1 && partial.t1 == 505 && partial.t_dtw == -1,
                "independently available endpoint was not mapped");
        state.has_vad_segments = false;
        auto disabled = whisper_full_get_token_data(&ctx, 0, 0);
        require(disabled.t0 == 105 && disabled.t1 == 115 && disabled.t_dtw == 110, "disabled VAD still mapped");
        state.has_vad_segments = true;
        state.vad_mapping_table.clear();
        auto empty = whisper_full_get_token_data(&ctx, 0, 0);
        require(empty.t0 == 105 && empty.t1 == 115 && empty.t_dtw == 110, "empty map changed token data");
        return 0;
    }
    if (test == "failure") {
        params.vad = true;
        vad_failure = true;
        require(whisper_full(&ctx, params, pcm.data(), int(pcm.size())) == -1, "detector failure lost");
        require(state.result_all.empty(), "sequential VAD failure exposed prior transcript");
        state.result_all.push_back(seg);
        state.vad_context = nullptr;
        fail_vad_init = true;
        require(whisper_full(&ctx, params, pcm.data(), int(pcm.size())) == -1, "loader failure lost");
        require(state.result_all.empty(), "VAD loader failure exposed prior transcript");
        return 0;
    }
    if (test == "empty") {
        require(whisper_full_parallel(&ctx, params, nullptr, 0, 3) == 0, "empty input rejected");
        check_mapping_cleared(state);
        return 0;
    }
    if (test == "offset-past-end") {
        params.offset_ms = 8000;
        require(whisper_full_parallel(&ctx, params, pcm.data(), int(pcm.size()), 3) == 0,
                "out-of-range offset rejected");
        check_mapping_cleared(state);
        require(state.result_all.empty(), "out-of-range offset retained old results");
        return 0;
    }
    if (test == "stitch") {
        params.vad = true;
        for (int processors : {1, 2, 3}) {
            require(whisper_full_parallel(&ctx, params, pcm.data(), int(pcm.size()), processors) == 0,
                    "VAD decode failed");
            require(state.has_vad_segments && state.vad_segments.size() == 2, "parent mapping lost");
            for (size_t i = 0; i < state.result_all.size(); ++i) {
                const auto& raw = state.result_all[i].tokens[0];
                const auto mapped = whisper_full_get_token_data(&ctx, int(i), 0);
                require(mapped.t0 == map_processed_to_original_time(raw.t0, state.vad_mapping_table),
                        "merged VAD token omitted original-time mapping");
            }
        }
        require(vad_calls == 4 && allocated == freed, "VAD repeated per worker or state leaked");
        return 0;
    }
    int rc = test == "reuse-state"        ? whisper_full_with_state(&ctx, &state, params, pcm.data(), int(pcm.size()))
             : test == "reuse-sequential" ? whisper_full(&ctx, params, pcm.data(), int(pcm.size()))
                                          : whisper_full_parallel(&ctx, params, pcm.data(), int(pcm.size()), 3);
    require(rc == 0, "plain decode failed");
    check_mapping_cleared(state);
    require(whisper_full_get_segment_t0_from_state(&state, 0) == 1, "plain segment still uses VAD map");
    require(allocated == freed, "worker state leaked");
    return 0;
}
