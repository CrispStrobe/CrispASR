// crispasr_run.cpp — top-level dispatch for non-whisper backends.
//
// Called from cli.cpp main() when params.backend is a non-whisper backend.
// Drives the pipeline: resolve model -> detect backend -> load audio ->
// segment via VAD (or fixed chunks) -> transcribe -> print + write outputs.
//
// The whisper code path in cli.cpp is left completely untouched so the
// historical whisper-cli behaviour is bit-identical.

#include "crispasr_backend.h"
#include "crispasr_cache.h"
#include "crispasr_vad_cli.h"
#include "crispasr_output.h"
#include "crispasr_model_mgr_cli.h"
#include "crispasr_model_registry.h"
#include "crispasr_aligner_cli.h"
#include "crispasr_lid_cli.h"
#include "crispasr_diarize_cli.h"
#include "whisper_params.h"
#include "fireredpunc.h"

#include "common-whisper.h" // read_audio_data

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

// Apply FireRedPunc punctuation restoration to all segments.
static void apply_punc_model(fireredpunc_context * punc_ctx, std::vector<crispasr_segment> & segs) {
    if (!punc_ctx) return;
    for (auto & seg : segs) {
        char * result = fireredpunc_process(punc_ctx, seg.text.c_str());
        if (result) {
            seg.text = result;
            free(result);
        }
    }
}

// Capability-vs-request check. For each requested feature, warn on stderr
// when the backend doesn't support it. Not fatal — the feature is silently
// ignored. Returns the number of warnings emitted.
int warn_unsupported(const CrispasrBackend& backend, const whisper_params& p) {
    const uint32_t caps = backend.capabilities();
    int warns = 0;

    auto warn = [&](const char* feature) {
        fprintf(stderr, "crispasr: warning: backend '%s' does not support %s — ignoring\n", backend.name(), feature);
        warns++;
    };

    // Diarize is now handled at the dispatcher level via the generic
    // crispasr_apply_diarize() post-step (energy / xcorr / future
    // pyannote / ecapa), so no warning even when the backend itself
    // doesn't claim CAP_DIARIZE — the dispatcher will label the
    // segments after transcribe() returns. Tinydiarize still requires
    // backend support (whisper-only).
    if (p.tinydiarize && !(caps & CAP_DIARIZE))
        warn("--tinydiarize");
    if (p.translate && !(caps & CAP_TRANSLATE))
        warn("--translate");
    if (!p.grammar.empty() && !(caps & CAP_GRAMMAR))
        warn("--grammar");
    if (p.temperature != 0.0f && !(caps & CAP_TEMPERATURE))
        warn("--temperature");
    if (!p.punctuation && !(caps & CAP_PUNCTUATION_TOGGLE))
        warn("--no-punctuation");
    if (!p.source_lang.empty() && !(caps & CAP_SRC_TGT_LANGUAGE))
        warn("--source-lang");
    if (!p.target_lang.empty() && !(caps & CAP_SRC_TGT_LANGUAGE))
        warn("--target-lang");
    if (p.n_processors > 1 && !(caps & CAP_PARALLEL_PROCESSORS))
        warn("--processors > 1");

    return warns;
}

// Merge individual-slice results into a flat list preserving time order.
std::vector<crispasr_segment> merge_segments(std::vector<std::vector<crispasr_segment>>&& per_slice,
                                             const std::vector<crispasr_audio_slice>& /*slices*/) {
    std::vector<crispasr_segment> out;
    size_t total = 0;
    for (auto& v : per_slice)
        total += v.size();
    out.reserve(total);
    for (auto& v : per_slice) {
        for (auto& s : v)
            out.push_back(std::move(s));
    }
    return out;
}

// Stdout serialization mutex. Used by the parallel-processors path to
// keep stdout transcript lines from interleaving across worker threads.
// The single-threaded path acquires it too — no measurable cost since
// it's an uncontended lock when n_processors == 1.
std::mutex g_stdout_mutex;

// Process a single input file end-to-end with the given backend instance.
// Pulled out of the main loop so the parallel-processors path can call
// it from worker threads. Each call holds its own audio buffers + segment
// state, so multiple workers can run concurrently against pre-loaded
// per-thread backend instances. Returns 0 on success, non-zero on
// failure.
int process_one_input(CrispasrBackend& backend, const std::string& fname_inp, whisper_params params,
                      fireredpunc_context * punc_ctx = nullptr) {
    std::vector<float> samples;
    std::vector<std::vector<float>> stereo;
    const bool want_stereo = params.diarize;
    if (!read_audio_data(fname_inp, samples, stereo, want_stereo)) {
        fprintf(stderr, "crispasr: error: failed to read audio '%s'\n", fname_inp.c_str());
        return 20;
    }
    bool have_stereo = want_stereo && stereo.size() == 2 && !stereo[0].empty() && stereo[0].size() == stereo[1].size();
    if (have_stereo) {
        const size_t n = stereo[0].size();
        const size_t check = std::min<size_t>(n, 4096);
        bool channels_equal = true;
        for (size_t i = 0; i < check; i++) {
            if (stereo[0][i] != stereo[1][i]) {
                channels_equal = false;
                break;
            }
        }
        if (channels_equal)
            have_stereo = false;
    }

    constexpr int SR = 16000;
    if (!params.no_prints) {
        fprintf(stderr, "crispasr: audio: %d samples (%.1f s) @ %d Hz, %d threads\n", (int)samples.size(),
                (double)samples.size() / SR, SR, params.n_threads);
    }

    // Optional language-identification pre-step.
    const bool want_auto_lang = params.detect_language || params.language == "auto";
    const bool has_native_lid = (backend.capabilities() & CAP_LANGUAGE_DETECT) != 0;
    const bool lid_disabled = params.lid_backend == "off" || params.lid_backend == "none";
    crispasr_lid_info lid_info; // stored for JSON output
    if (want_auto_lang && !has_native_lid && !lid_disabled) {
        crispasr_lid_result lid;
        if (crispasr_detect_language_cli(samples.data(), (int)samples.size(), params, lid)) {
            lid_info.lang_code = lid.lang_code;
            lid_info.confidence = lid.confidence;
            lid_info.source = lid.source;
            params.language = lid.lang_code;
            if (params.source_lang.empty()) {
                params.source_lang = lid.lang_code;
            }
            if (!params.no_prints) {
                fprintf(stderr, "crispasr: LID -> language = '%s' (%s, p=%.3f)\n", lid.lang_code.c_str(),
                        lid.source.c_str(), lid.confidence);
            }
        } else if (!params.no_prints) {
            fprintf(stderr, "crispasr: LID failed, falling back to params.language='%s'\n", params.language.c_str());
        }
    }

    const auto slices =
        crispasr_compute_audio_slices(samples.data(), (int)samples.size(), SR, params.chunk_seconds, params);

    if (slices.empty()) {
        fprintf(stderr, "crispasr: warning: no speech detected in '%s'\n", fname_inp.c_str());
        return 0;
    }

    if (!params.no_prints && slices.size() > 1) {
        fprintf(stderr, "crispasr: processing %zu slice(s)\n", slices.size());
    }

    auto t_start = std::chrono::steady_clock::now();

    // --------------- VAD stitching path (whisper.cpp-style) ---------------
    // When VAD produces multiple slices, stitch them into one contiguous
    // buffer (with 0.1s silence gaps) and process as a single transcribe()
    // call. This preserves cross-segment context and avoids boundary
    // artifacts. Timestamps are remapped from stitched-buffer positions
    // back to original-audio positions.
    //
    // Skip stitching for whisper backend (it has its own internal VAD+seek)
    // and when there's only one slice (no benefit).
    // Stitching concatenates all VAD segments into one buffer for a single
    // transcribe() call. This preserves cross-segment context but collapses
    // the output into one big segment — breaking SRT/VTT subtitle output.
    // Default: use per-slice path (each VAD segment → separate transcript
    // segment with correct timestamps). Users can opt in to stitching with
    // --vad-stitch if they want cross-segment context at the cost of
    // single-segment output.
    const bool use_stitching = slices.size() > 1 && params.vad && params.backend != "whisper" && params.vad_stitch;

    if (use_stitching) {
        auto stitched = crispasr_stitch_vad_slices(samples.data(), (int)samples.size(), SR, slices);
        if (!params.no_prints) {
            fprintf(stderr, "crispasr: stitched %zu VAD segments → %.1fs (from %.1fs original)\n", slices.size(),
                    (double)stitched.total_duration_cs / 100.0, (double)samples.size() / SR);
        }

        // Transcribe the stitched buffer as one call.
        auto segs = backend.transcribe(stitched.samples.data(), (int)stitched.samples.size(), 0, params);

        // Remap timestamps from stitched-buffer space to original-audio space.
        for (auto& seg : segs) {
            seg.t0 = crispasr_vad_remap_timestamp(stitched.mapping, seg.t0);
            seg.t1 = crispasr_vad_remap_timestamp(stitched.mapping, seg.t1);
            for (auto& w : seg.words) {
                w.t0 = crispasr_vad_remap_timestamp(stitched.mapping, w.t0);
                w.t1 = crispasr_vad_remap_timestamp(stitched.mapping, w.t1);
            }
        }

        // Optional CTC alignment (on original audio, not stitched).
        const bool want_align = !params.aligner_model.empty() && (backend.capabilities() & CAP_TIMESTAMPS_CTC);
        if (want_align) {
            for (auto& seg : segs) {
                if (!seg.words.empty())
                    continue;
                // Find the original audio region for this segment.
                const int s = (int)((double)seg.t0 / 100.0 * SR);
                const int e = std::min((int)samples.size(), (int)((double)seg.t1 / 100.0 * SR));
                if (e > s) {
                    auto words = crispasr_ctc_align(params.aligner_model, seg.text, samples.data() + s, e - s, seg.t0,
                                                    params.n_threads);
                    if (!words.empty()) {
                        seg.t0 = words.front().t0;
                        seg.t1 = words.back().t1;
                        seg.words = std::move(words);
                    }
                }
            }
        }

        // Fall through to the shared output path below by wrapping
        // the stitched result into per_slice / all_segs.
        std::vector<std::vector<crispasr_segment>> stitched_per_slice(1);
        stitched_per_slice[0] = std::move(segs);
        auto all_segs = merge_segments(std::move(stitched_per_slice), slices);

        apply_punc_model(punc_ctx, all_segs);
        if (!params.punctuation) {
            for (auto& seg : all_segs)
                crispasr_strip_punctuation(seg);
        }

        const auto disp = crispasr_make_disp_segments(all_segs, params.max_len, params.split_on_punct);
        const bool show_timestamps =
            !params.no_timestamps &&
            (params.output_srt || params.output_vtt || params.max_len > 0 || params.print_colors || params.diarize);
        {
            auto t_end = std::chrono::steady_clock::now();
            double t_total = std::chrono::duration<double>(t_end - t_start).count();
            double audio_s = (double)samples.size() / SR;
            if (!params.no_prints) {
                fprintf(stderr, "crispasr: transcribed %.1fs audio in %.2fs (%.1fx realtime)\n", audio_s, t_total,
                        audio_s / std::max(t_total, 0.001));
            }
            std::lock_guard<std::mutex> lock(g_stdout_mutex);
            crispasr_print_stdout(disp, show_timestamps);
            if (params.show_alternatives)
                crispasr_print_alternatives(all_segs, params.n_alternatives);
        }
        if (params.output_txt)
            crispasr_write_txt(crispasr_make_out_path(fname_inp, ".txt"), disp);
        if (params.output_srt)
            crispasr_write_srt(crispasr_make_out_path(fname_inp, ".srt"), disp);
        if (params.output_vtt)
            crispasr_write_vtt(crispasr_make_out_path(fname_inp, ".vtt"), disp);
        if (params.output_csv)
            crispasr_write_csv(crispasr_make_out_path(fname_inp, ".csv"), disp);
        if (params.output_lrc)
            crispasr_write_lrc(crispasr_make_out_path(fname_inp, ".lrc"), disp);
        if (params.output_jsn)
            crispasr_write_json(crispasr_make_out_path(fname_inp, ".json"), all_segs, backend.name(), params.model,
                                params.language, params.output_jsn_full,
                                lid_info.lang_code.empty() ? nullptr : &lid_info);
        return 0;
    }

    // --------------- Per-slice path (non-VAD or single slice) ---------------
    // Process VAD slices — parallel when multiple slices AND n_processors > 1
    std::vector<std::vector<crispasr_segment>> per_slice(slices.size());

    auto process_slice = [&](size_t i, CrispasrBackend& be) {
        const auto& sl = slices[i];
        std::vector<crispasr_segment> segs =
            be.transcribe(samples.data() + sl.start, sl.end - sl.start, sl.t0_cs, params);

        if (params.diarize && !segs.empty()) {
            if (have_stereo) {
                std::vector<float> sl_l(stereo[0].begin() + sl.start, stereo[0].begin() + sl.end);
                std::vector<float> sl_r(stereo[1].begin() + sl.start, stereo[1].begin() + sl.end);
                crispasr_apply_diarize(sl_l, sl_r, /*is_stereo=*/true, sl.t0_cs, segs, params);
            } else {
                std::vector<float> mono_slice(samples.begin() + sl.start, samples.begin() + sl.end);
                crispasr_apply_diarize(mono_slice, mono_slice,
                                       /*is_stereo=*/false, sl.t0_cs, segs, params);
            }
        }

        const bool want_align = !params.aligner_model.empty() && (backend.capabilities() & CAP_TIMESTAMPS_CTC);
        if (want_align) {
            for (auto& seg : segs) {
                if (!seg.words.empty())
                    continue;
                auto words = crispasr_ctc_align(params.aligner_model, seg.text, samples.data() + sl.start,
                                                sl.end - sl.start, sl.t0_cs, params.n_threads);
                if (!words.empty()) {
                    seg.t0 = words.front().t0;
                    seg.t1 = words.back().t1;
                    seg.words = std::move(words);
                }
            }
        }

        per_slice[i] = std::move(segs);
    };

    const int n_workers = std::min(params.n_processors, (int32_t)slices.size());

    if (n_workers > 1 && slices.size() > 1) {
        // Parallel slice processing with separate backend instances
        if (!params.no_prints) {
            fprintf(stderr, "crispasr: parallel processing %zu slices with %d workers\n", slices.size(), n_workers);
        }

        // Create extra backend instances for worker threads
        std::vector<std::unique_ptr<CrispasrBackend>> workers;
        workers.reserve(n_workers - 1);
        bool pool_ok = true;
        for (int w = 1; w < n_workers; w++) {
            auto wb = crispasr_create_backend(params.backend);
            if (!wb || !wb->init(params)) {
                if (!params.no_prints)
                    fprintf(stderr, "crispasr: warning: failed to create worker %d, reducing parallelism\n", w);
                pool_ok = false;
                break;
            }
            workers.push_back(std::move(wb));
        }

        if (pool_ok && !workers.empty()) {
            // Dispatch slices round-robin across workers
            std::vector<std::thread> threads;
            std::atomic<size_t> next_slice{0};

            auto worker_fn = [&](CrispasrBackend& be) {
                while (true) {
                    size_t idx = next_slice.fetch_add(1);
                    if (idx >= slices.size())
                        break;
                    process_slice(idx, be);
                }
            };

            // Launch worker threads (workers[0..N-2] + main thread uses backend)
            for (auto& w : workers) {
                threads.emplace_back(worker_fn, std::ref(*w));
            }
            // Main thread also processes slices
            worker_fn(backend);

            for (auto& t : threads)
                t.join();
        } else {
            // Fallback to sequential
            for (size_t i = 0; i < slices.size(); i++)
                process_slice(i, backend);
        }
    } else if (params.flush_after > 0 && slices.size() > 1) {
        // Progressive mode: process slices sequentially, flush output after each.
        // This gives media players SRT entries as soon as each VAD segment is done.
        int srt_index = 1; // running SRT entry counter
        const bool show_ts = !params.no_timestamps && (params.output_srt || params.output_vtt ||
                                                        params.max_len > 0 || params.print_colors || params.diarize);
        for (size_t i = 0; i < slices.size(); i++) {
            process_slice(i, backend);

            // Post-process this slice immediately
            auto slice_segs = std::move(per_slice[i]);
            apply_punc_model(punc_ctx, slice_segs);
            if (!params.punctuation) {
                for (auto& seg : slice_segs)
                    crispasr_strip_punctuation(seg);
            }

            auto disp = crispasr_make_disp_segments(slice_segs, params.max_len, params.split_on_punct);

            // Print SRT entries progressively to stdout
            for (const auto& d : disp) {
                if (params.output_srt) {
                    int t0_ms = (int)(d.t0 * 10);
                    int t1_ms = (int)(d.t1 * 10);
                    printf("%d\n%02d:%02d:%02d,%03d --> %02d:%02d:%02d,%03d\n%s\n\n",
                           srt_index++,
                           t0_ms / 3600000, (t0_ms / 60000) % 60, (t0_ms / 1000) % 60, t0_ms % 1000,
                           t1_ms / 3600000, (t1_ms / 60000) % 60, (t1_ms / 1000) % 60, t1_ms % 1000,
                           d.text.c_str());
                } else {
                    if (show_ts) {
                        int s0 = (int)(d.t0 * 10), s1 = (int)(d.t1 * 10);
                        printf("[%02d:%02d:%02d.%03d --> %02d:%02d:%02d.%03d]  %s\n",
                               s0 / 3600000, (s0 / 60000) % 60, (s0 / 1000) % 60, s0 % 1000,
                               s1 / 3600000, (s1 / 60000) % 60, (s1 / 1000) % 60, s1 % 1000,
                               d.text.c_str());
                    } else {
                        printf("%s", d.text.c_str());
                    }
                }
            }
            fflush(stdout);
        }

        // Timing
        {
            auto t_end = std::chrono::steady_clock::now();
            double t_total = std::chrono::duration<double>(t_end - t_start).count();
            double audio_s = (double)samples.size() / SR;
            if (!params.no_prints) {
                fprintf(stderr, "crispasr: transcribed %.1fs audio in %.2fs (%.1fx realtime)\n", audio_s, t_total,
                        audio_s / t_total);
            }
        }

        // Write output files (full set, from all slices combined)
        // Re-collect all per_slice segments for file output
        // (stdout already got progressive output above)
        if (params.output_txt || params.output_vtt || params.output_csv ||
            params.output_lrc || params.output_jsn) {
            // Re-run all slices to collect for file output
            std::vector<std::vector<crispasr_segment>> per_slice_redo(slices.size());
            for (size_t i = 0; i < slices.size(); i++) {
                process_slice(i, backend);
                per_slice_redo[i] = std::move(per_slice[i]);
            }
            auto all_segs = merge_segments(std::move(per_slice_redo), slices);
            apply_punc_model(punc_ctx, all_segs);
            if (!params.punctuation)
                for (auto& seg : all_segs)
                    crispasr_strip_punctuation(seg);
            auto disp_all = crispasr_make_disp_segments(all_segs, params.max_len, params.split_on_punct);

            if (params.output_txt)
                crispasr_write_txt(crispasr_make_out_path(fname_inp, ".txt"), disp_all);
            if (params.output_srt)
                crispasr_write_srt(crispasr_make_out_path(fname_inp, ".srt"), disp_all);
            if (params.output_vtt)
                crispasr_write_vtt(crispasr_make_out_path(fname_inp, ".vtt"), disp_all);
            if (params.output_csv)
                crispasr_write_csv(crispasr_make_out_path(fname_inp, ".csv"), disp_all);
            if (params.output_lrc)
                crispasr_write_lrc(crispasr_make_out_path(fname_inp, ".lrc"), disp_all);
            if (params.output_jsn)
                crispasr_write_json(crispasr_make_out_path(fname_inp, ".json"), all_segs, backend.name(), params.model,
                                    params.language, params.output_jsn_full,
                                    lid_info.lang_code.empty() ? nullptr : &lid_info);
        }
        return 0;
    } else {
        // Sequential (single slice or n_processors == 1)
        for (size_t i = 0; i < slices.size(); i++)
            process_slice(i, backend);
    }
    auto all_segs = merge_segments(std::move(per_slice), slices);

    apply_punc_model(punc_ctx, all_segs);
    if (!params.punctuation) {
        for (auto& seg : all_segs) {
            crispasr_strip_punctuation(seg);
        }
    }

    const auto disp = crispasr_make_disp_segments(all_segs, params.max_len, params.split_on_punct);

    const bool show_timestamps = !params.no_timestamps && (params.output_srt || params.output_vtt ||
                                                           params.max_len > 0 || params.print_colors || params.diarize);
    {
        auto t_end = std::chrono::steady_clock::now();
        double t_total = std::chrono::duration<double>(t_end - t_start).count();
        double audio_s = (double)samples.size() / SR;
        if (!params.no_prints) {
            fprintf(stderr, "crispasr: transcribed %.1fs audio in %.2fs (%.1fx realtime)\n", audio_s, t_total,
                    audio_s / std::max(t_total, 0.001));
        }

        // Serialize stdout across parallel workers so multi-file
        // transcripts don't interleave line-by-line.
        std::lock_guard<std::mutex> lock(g_stdout_mutex);
        crispasr_print_stdout(disp, show_timestamps);
        if (params.show_alternatives) {
            crispasr_print_alternatives(all_segs, params.n_alternatives);
        }
    }

    if (params.output_txt)
        crispasr_write_txt(crispasr_make_out_path(fname_inp, ".txt"), disp);
    if (params.output_srt)
        crispasr_write_srt(crispasr_make_out_path(fname_inp, ".srt"), disp);
    if (params.output_vtt)
        crispasr_write_vtt(crispasr_make_out_path(fname_inp, ".vtt"), disp);
    if (params.output_csv)
        crispasr_write_csv(crispasr_make_out_path(fname_inp, ".csv"), disp);
    if (params.output_lrc)
        crispasr_write_lrc(crispasr_make_out_path(fname_inp, ".lrc"), disp);
    if (params.output_jsn)
        crispasr_write_json(crispasr_make_out_path(fname_inp, ".json"), all_segs, backend.name(), params.model,
                            params.language, params.output_jsn_full, lid_info.lang_code.empty() ? nullptr : &lid_info);

    return 0;
}

} // namespace

int crispasr_run_backend(const whisper_params& params_in) {
    whisper_params params = params_in;

    if (params.verbose) {
        fprintf(stderr, "crispasr[verbose]: model arg          = '%s'\n", params.model.c_str());
        fprintf(stderr, "crispasr[verbose]: backend arg        = '%s'\n",
                params.backend.empty() ? "auto" : params.backend.c_str());
        fprintf(stderr, "crispasr[verbose]: use_gpu            = %s\n", params.use_gpu ? "true" : "false");
        fprintf(stderr, "crispasr[verbose]: gpu_backend        = '%s'\n",
                params.gpu_backend.empty() ? "auto" : params.gpu_backend.c_str());
        fprintf(stderr, "crispasr[verbose]: gpu_device         = %d\n", params.gpu_device);
        fprintf(stderr, "crispasr[verbose]: cache_dir override = '%s'\n",
                params.cache_dir.empty() ? "(default)" : params.cache_dir.c_str());
        fprintf(stderr, "crispasr[verbose]: auto_download      = %s\n", params.auto_download ? "true" : "false");
        fprintf(stderr, "crispasr[verbose]: n_threads          = %d\n", params.n_threads);
        fprintf(stderr, "crispasr[verbose]: flash_attn         = %s\n", params.flash_attn ? "true" : "false");
    }

    // Resolve backend name: explicit --backend takes priority; otherwise
    // auto-detect from the GGUF file. Defaults are handled in cli.cpp.
    std::string backend_name = params.backend;
    const bool model_is_auto = params.model == "auto" || params.model == "default";
    if (backend_name.empty() || backend_name == "auto") {
        if (model_is_auto) {
            // `-m auto` with no --backend. Before defaulting to
            // whisper-download, scan the cache for any already-downloaded
            // registered model (whisper > parakeet > canary > …). Users
            // who already have, say, a parakeet GGUF from a previous
            // session shouldn't trigger a fresh 147 MB whisper download.
            CrispasrRegistryEntry cached;
            if (crispasr_find_cached_model(cached, params.cache_dir)) {
                backend_name = cached.backend;
                params.model = crispasr_cache::dir(params.cache_dir) + "/" + cached.filename;
                if (!params.no_prints) {
                    fprintf(stderr, "crispasr: -m auto — using cached %s model (%s)\n", backend_name.c_str(),
                            cached.filename.c_str());
                }
            } else {
                backend_name = "whisper";
                if (!params.no_prints) {
                    fprintf(stderr, "crispasr: -m auto with no cached model — defaulting to whisper\n");
                }
            }
        } else {
            backend_name = crispasr_detect_backend_from_gguf(params.model);
            if (backend_name.empty()) {
                fprintf(stderr,
                        "crispasr: error: could not auto-detect backend from '%s'. "
                        "Use --backend NAME to force one.\n",
                        params.model.c_str());
                return 10;
            }
            if (!params.no_prints) {
                fprintf(stderr, "crispasr: detected backend '%s' from GGUF metadata\n", backend_name.c_str());
            }
        }
    }

    // Resolve "-m auto" via the model registry + curl/wget download.
    const std::string resolved = crispasr_resolve_model_cli(params.model, backend_name, params.no_prints,
                                                            params.cache_dir, params.auto_download);
    if (params.verbose) {
        fprintf(stderr, "crispasr[verbose]: resolved model     = '%s'\n", resolved.c_str());
    }
    if (resolved.empty()) {
        return 11;
    }
    params.model = resolved;

    // Create and init the backend.
    std::unique_ptr<CrispasrBackend> backend = crispasr_create_backend(backend_name);
    if (!backend) {
        fprintf(stderr, "crispasr: error: backend '%s' is not available in this build\n", backend_name.c_str());
        return 12;
    }

    warn_unsupported(*backend, params);

    if (!backend->init(params)) {
        fprintf(stderr, "crispasr: error: failed to initialise backend '%s'\n", backend_name.c_str());
        return 13;
    }
    if (params.verbose) {
        fprintf(stderr, "crispasr[verbose]: backend '%s' initialised OK\n", backend_name.c_str());
    }

    // Optional punctuation restoration post-processor.
    fireredpunc_context * punc_ctx = nullptr;
    if (!params.punc_model.empty()) {
        punc_ctx = fireredpunc_init(params.punc_model.c_str());
        if (!punc_ctx) {
            fprintf(stderr, "crispasr: warning: failed to load punc model '%s' — continuing without\n",
                    params.punc_model.c_str());
        } else if (!params.no_prints) {
            fprintf(stderr, "crispasr: loaded punctuation model '%s'\n", params.punc_model.c_str());
        }
    }

    // ---- Streaming mode: read raw PCM from stdin, transcribe chunks ----
    if (params.stream) {
        const int SR = 16000;
        const int step_samples = (params.stream_step_ms * SR) / 1000;
        const int length_samples = (params.stream_length_ms * SR) / 1000;
        const int keep_samples = (params.stream_keep_ms * SR) / 1000;

        // If --mic, spawn a subprocess to capture audio from the default mic
        FILE* mic_pipe = nullptr;
        if (params.mic) {
            fprintf(stderr, "crispasr[mic]: capturing from default microphone...\n");
            fprintf(stderr, "crispasr[mic]: press Ctrl+C to stop\n\n");
            // Try platform-specific mic capture commands
#if defined(__APPLE__)
            // macOS: use sox (most reliable), ffmpeg fallback
            mic_pipe = popen("rec -q -t s16 -r 16000 -c 1 - 2>/dev/null || "
                             "ffmpeg -f avfoundation -i ':default' -f s16le -ar 16000 -ac 1 - 2>/dev/null",
                             "r");
#elif defined(_WIN32)
            mic_pipe = _popen("ffmpeg -f dshow -i audio=\"Microphone\" -f s16le -ar 16000 -ac 1 - 2>NUL", "rb");
#else
            // Linux: try arecord first, then ffmpeg with pulseaudio
            mic_pipe = popen("arecord -q -f S16_LE -r 16000 -c 1 -t raw 2>/dev/null || "
                             "ffmpeg -f pulse -i default -f s16le -ar 16000 -ac 1 - 2>/dev/null || "
                             "ffmpeg -f alsa -i default -f s16le -ar 16000 -ac 1 - 2>/dev/null",
                             "r");
#endif
            if (!mic_pipe) {
                fprintf(stderr, "crispasr[mic]: failed to open microphone. Install sox, ffmpeg, or arecord.\n");
                return 20;
            }
        } else {
            fprintf(stderr, "crispasr[stream]: reading raw s16le 16kHz mono PCM from stdin\n");
            fprintf(stderr, "crispasr[stream]: step=%dms length=%dms keep=%dms\n", params.stream_step_ms,
                    params.stream_length_ms, params.stream_keep_ms);
            fprintf(stderr, "crispasr[stream]: pipe audio in, e.g.:\n");
            fprintf(stderr, "  ffmpeg -i input.wav -f s16le -ar 16000 -ac 1 - | crispasr --stream -m model.gguf\n\n");
        }

        FILE* audio_src = mic_pipe ? mic_pipe : stdin;

#if defined(_WIN32)
        if (!mic_pipe)
            _setmode(_fileno(stdin), _O_BINARY);
#endif

        std::vector<float> pcm_window(length_samples, 0.0f);
        std::vector<int16_t> read_buf(step_samples);
        std::string prev_text;

        while (true) {
            // Read one step of raw s16le samples from audio source
            size_t n_read = fread(read_buf.data(), sizeof(int16_t), step_samples, audio_src);
            if (n_read == 0)
                break; // EOF

            // Convert s16le to float
            std::vector<float> new_samples(n_read);
            for (size_t i = 0; i < n_read; i++)
                new_samples[i] = read_buf[i] / 32768.0f;

            // Shift window: keep the tail, append new samples
            int n_keep = std::min(keep_samples, (int)pcm_window.size());
            int n_new = (int)new_samples.size();
            int n_total = n_keep + n_new;
            if (n_total > length_samples)
                n_total = length_samples;

            std::vector<float> next_window(n_total);
            // Copy keep portion from end of previous window
            if (n_keep > 0 && (int)pcm_window.size() >= n_keep) {
                std::copy(pcm_window.end() - n_keep, pcm_window.end(), next_window.begin());
            }
            // Append new samples
            int copy_start = std::max(0, n_total - n_new);
            std::copy(new_samples.begin(), new_samples.begin() + std::min(n_new, n_total),
                      next_window.begin() + copy_start);
            pcm_window = std::move(next_window);

            // Monitor: show progress during processing
            if (params.stream_monitor) {
                fprintf(stderr, "\xE2\x96\xB6"); // ▶ = processing chunk
                fflush(stderr);
            }

            // Transcribe the window
            auto segs = backend->transcribe(pcm_window.data(), (int)pcm_window.size(), 0, params);

            if (params.stream_monitor) {
                if (segs.empty()) {
                    fprintf(stderr, "\xC2\xB7"); // · = silence
                } else {
                    fprintf(stderr, "\xE2\x9C\x93"); // ✓ = got text
                }
                fflush(stderr);
            }

            if (segs.empty())
                continue;

            // Build output text
            std::string text;
            for (const auto& s : segs)
                text += s.text;

            // Output depends on mode:
            // Continuous: print each non-empty result as a new line
            // Normal: overwrite current line (dedup by text content)
            if (params.stream_continuous) {
                if (!text.empty()) {
                    fprintf(stdout, "%s\n", text.c_str());
                    fflush(stdout);
                }
            } else {
                if (!text.empty() && text != prev_text) {
                    fprintf(stdout, "\33[2K\r%s", text.c_str());
                    fflush(stdout);
                    prev_text = text;
                }
            }
        }
        fprintf(stdout, "\n");
        if (mic_pipe) {
#if defined(_WIN32)
            _pclose(mic_pipe);
#else
            pclose(mic_pipe);
#endif
        }
        return 0;
    }

    // Process every input file.
    //
    // n_processors == 1 (default): sequential, single backend instance.
    // Bit-identical with the historical CrispASR behaviour.
    //
    // n_processors > 1: spawn N-1 EXTRA backend instances (model-load
    //                   cost paid N times — beware), then dispatch
    //                   files across N worker threads. Best when you
    //                   have many independent input files; useless on
    //                   single-file runs because both workers would
    //                   race on the same audio.
    int rc = 0;
    const int nproc = std::max(1, params.n_processors);
    if (nproc > 1 && params.fname_inp.size() > 1) {
        // Pre-load N-1 EXTRA backend instances (we already have one).
        // Failure to load any worker is fatal — better to bail than to
        // silently fall back to single-thread, which would surprise
        // batch users with much slower runs.
        std::vector<std::unique_ptr<CrispasrBackend>> pool;
        pool.reserve(nproc);
        pool.emplace_back(std::move(backend));
        for (int i = 1; i < nproc; i++) {
            auto extra = crispasr_create_backend(backend_name);
            if (!extra || !extra->init(params)) {
                fprintf(stderr,
                        "crispasr: error: failed to spin up worker %d/%d "
                        "(extra backend init failed). Try fewer --processors.\n",
                        i + 1, nproc);
                return 14;
            }
            pool.emplace_back(std::move(extra));
        }
        if (!params.no_prints) {
            fprintf(stderr, "crispasr: parallel mode: %d worker(s), %zu input file(s)\n", nproc,
                    params.fname_inp.size());
        }

        // Shared work queue: index into params.fname_inp. std::atomic
        // counter is enough — no need for a real queue since each
        // worker just claims the next index.
        std::atomic<int> next_idx{0};
        std::atomic<int> agg_rc{0};
        const int n_files = (int)params.fname_inp.size();
        std::vector<std::thread> workers;
        workers.reserve((size_t)nproc);
        for (int w = 0; w < nproc; w++) {
            workers.emplace_back([&, w]() {
                CrispasrBackend& be = *pool[w];
                while (true) {
                    const int idx = next_idx.fetch_add(1);
                    if (idx >= n_files)
                        break;
                    const int file_rc = process_one_input(be, params.fname_inp[idx], params, punc_ctx);
                    if (file_rc != 0)
                        agg_rc.store(file_rc);
                }
            });
        }
        for (auto& t : workers)
            t.join();

        for (auto& be : pool)
            be->shutdown();
        return agg_rc.load();
    }

    for (const auto& fname_inp : params.fname_inp) {
        const int file_rc = process_one_input(*backend, fname_inp, params, punc_ctx);
        if (file_rc != 0)
            rc = file_rc;
    }
    backend->shutdown();
    return rc;
}

#if 0
// Legacy in-place per-file loop body. Moved into process_one_input()
// above. Kept here under #if 0 only for diff/blame archaeology — the
// linker drops it.
{
    std::vector<float> samples;
    std::vector<std::vector<float>> stereo;
        // Request stereo split when --diarize is set. Diarize is now
        // a generic dispatcher post-step (crispasr_diarize.cpp), so we
        // try it for every backend rather than only those that
        // advertise CAP_DIARIZE — the backend itself doesn't have to
        // know anything about stereo; the dispatcher labels its
        // segments after transcribe() returns.
        const bool want_stereo = params.diarize;
        if (!read_audio_data(fname_inp, samples, stereo, want_stereo)) {
            fprintf(stderr, "crispasr: error: failed to read audio '%s'\n",
                    fname_inp.c_str());
            rc = 20;
            continue;
        }
        bool have_stereo = want_stereo &&
            stereo.size() == 2 &&
            !stereo[0].empty() &&
            stereo[0].size() == stereo[1].size();
        // miniaudio duplicates mono -> both channels when we ask for
        // stereo, so a mono input file gives us pcmf32s[0] == pcmf32s[1].
        // Detect that and downgrade to mono so the diarize post-step
        // takes the mono-friendly path (vad-turns) instead of the
        // tie-only energy path.
        if (have_stereo) {
            const size_t n = stereo[0].size();
            const size_t check = std::min<size_t>(n, 4096);
            bool channels_equal = true;
            for (size_t i = 0; i < check; i++) {
                if (stereo[0][i] != stereo[1][i]) { channels_equal = false; break; }
            }
            if (channels_equal) have_stereo = false;
        }

        constexpr int SR = 16000;
        if (!params.no_prints) {
            fprintf(stderr,
                    "crispasr: audio: %d samples (%.1f s) @ %d Hz, %d threads\n",
                    (int)samples.size(),
                    (double)samples.size() / SR, SR, params.n_threads);
        }

        // Optional language-identification pre-step. Fires only when the
        // user asked for auto language (either --detect-language or
        // --language auto) AND the chosen backend can't detect language
        // natively (qwen3/whisper/parakeet already do). The detected ISO
        // code is written into `params.language` and, if empty, into
        // `params.source_lang` so canary can pick it up as well.
        const bool want_auto_lang = params.detect_language ||
                                    params.language == "auto";
        const bool has_native_lid = (backend->capabilities() & CAP_LANGUAGE_DETECT) != 0;
        const bool lid_disabled   = params.lid_backend == "off" ||
                                    params.lid_backend == "none";
        if (want_auto_lang && !has_native_lid && !lid_disabled) {
            crispasr_lid_result lid;
            if (crispasr_detect_language_cli(samples.data(), (int)samples.size(),
                                          params, lid)) {
                params.language = lid.lang_code;
                if (params.source_lang.empty()) {
                    params.source_lang = lid.lang_code;
                }
                if (!params.no_prints) {
                    fprintf(stderr,
                            "crispasr: LID -> language = '%s' (%s, p=%.3f)\n",
                            lid.lang_code.c_str(), lid.source.c_str(),
                            lid.confidence);
                }
            } else if (!params.no_prints) {
                fprintf(stderr,
                        "crispasr: LID failed, falling back to params.language='%s'\n",
                        params.language.c_str());
            }
        }

        // Slice into chunks (VAD or fixed-window fallback).
        const auto slices = crispasr_compute_audio_slices(
            samples.data(), (int)samples.size(), SR,
            params.chunk_seconds, params);

        if (slices.empty()) {
            fprintf(stderr, "crispasr: warning: no speech detected in '%s'\n",
                    fname_inp.c_str());
            continue;
        }

        if (!params.no_prints && slices.size() > 1) {
            fprintf(stderr, "crispasr: processing %zu slice(s)\n", slices.size());
        }

        // Transcribe each slice.
        std::vector<std::vector<crispasr_segment>> per_slice;
        per_slice.reserve(slices.size());
        for (size_t i = 0; i < slices.size(); i++) {
            const auto & sl = slices[i];
            // Always transcribe in mono — every backend takes mono PCM
            // and the diarize step happens later as a generic post-pass.
            std::vector<crispasr_segment> segs = backend->transcribe(
                samples.data() + sl.start,
                sl.end - sl.start,
                sl.t0_cs,
                params);

            // Apply the generic diarize post-step. Stereo-only methods
            // (energy, xcorr) need have_stereo == true; mono-friendly
            // methods (vad-turns, future sherpa/pyannote) work either
            // way. Pass both channel buffers and an is_stereo hint;
            // when have_stereo is false we point both at the mono
            // buffer so the helper has data to look at without
            // special-casing.
            if (params.diarize && !segs.empty()) {
                if (have_stereo) {
                    std::vector<float> sl_l(stereo[0].begin() + sl.start,
                                            stereo[0].begin() + sl.end);
                    std::vector<float> sl_r(stereo[1].begin() + sl.start,
                                            stereo[1].begin() + sl.end);
                    crispasr_apply_diarize(sl_l, sl_r, /*is_stereo=*/true,
                                           sl.t0_cs, segs, params);
                } else {
                    std::vector<float> mono_slice(samples.begin() + sl.start,
                                                  samples.begin() + sl.end);
                    crispasr_apply_diarize(mono_slice, mono_slice,
                                           /*is_stereo=*/false,
                                           sl.t0_cs, segs, params);
                }
            }

            // Optional CTC forced alignment to attach word-level timestamps.
            // Applies to backends that expose CAP_TIMESTAMPS_CTC and don't
            // already have words populated. Runs per slice so absolute
            // timestamps come out right.
            const bool want_align =
                !params.aligner_model.empty() &&
                (backend->capabilities() & CAP_TIMESTAMPS_CTC);
            if (want_align) {
                for (auto & seg : segs) {
                    if (!seg.words.empty()) continue; // already aligned
                    auto words = crispasr_ctc_align(
                        params.aligner_model,
                        seg.text,
                        samples.data() + sl.start,
                        sl.end - sl.start,
                        sl.t0_cs,
                        params.n_threads);
                    if (!words.empty()) {
                        seg.t0 = words.front().t0;
                        seg.t1 = words.back().t1;
                        seg.words = std::move(words);
                    }
                }
            }

            per_slice.push_back(std::move(segs));
        }
        auto all_segs = merge_segments(std::move(per_slice), slices);

        apply_punc_model(punc_ctx, all_segs);

        // Optional post-processing: strip punctuation when --no-punctuation
        // is set. Cohere and canary pass p.punctuation through to their C
        // APIs natively and will usually return text that's already clean,
        // but this second pass is idempotent so the double application is
        // harmless. For the LLM backends (voxtral/voxtral4b/qwen3/granite)
        // this is the only way punctuation control happens — the models
        // don't take a "no punctuation" flag, they just generate whatever
        // the prompt pushes them towards.
        if (!params.punctuation) {
            for (auto & seg : all_segs) {
                crispasr_strip_punctuation(seg);
            }
        }

        // Build display segments.
        const auto disp = crispasr_make_disp_segments(all_segs, params.max_len, params.split_on_punct);

        // Print to stdout.
        const bool show_timestamps =
            !params.no_timestamps &&
            (params.output_srt || params.output_vtt ||
             params.max_len > 0  || params.print_colors ||
             params.diarize);
        crispasr_print_stdout(disp, show_timestamps);

        // Write output files.
        if (params.output_txt)
            crispasr_write_txt(crispasr_make_out_path(fname_inp, ".txt"), disp);
        if (params.output_srt)
            crispasr_write_srt(crispasr_make_out_path(fname_inp, ".srt"), disp);
        if (params.output_vtt)
            crispasr_write_vtt(crispasr_make_out_path(fname_inp, ".vtt"), disp);
        if (params.output_csv)
            crispasr_write_csv(crispasr_make_out_path(fname_inp, ".csv"), disp);
        if (params.output_lrc)
            crispasr_write_lrc(crispasr_make_out_path(fname_inp, ".lrc"), disp);
        if (params.output_jsn)
            crispasr_write_json(
                crispasr_make_out_path(fname_inp, ".json"),
                all_segs, backend->name(), params.model, params.language,
                params.output_jsn_full, nullptr);
    }

    if (punc_ctx) fireredpunc_free(punc_ctx);
    return 0;
}
#endif
