// crispasr_aligner_cli.cpp — CLI aligner adapter.
//
// Calls the shared library's `crispasr_align_words` and converts the
// resulting `CrispasrAlignedWord` vector to the CLI's `crispasr_word`
// shape consumed by crispasr_run.cpp and downstream output formatters.

#include "crispasr_aligner_cli.h"
#include "crispasr_aligner.h" // shared library header from src/

std::vector<crispasr_word> crispasr_ctc_align(const std::string& aligner_model, const std::string& transcript,
                                              const float* samples, int n_samples, int64_t t_offset_cs, int n_threads) {
    std::vector<crispasr_word> out;
    auto lib = crispasr_align_words(aligner_model, transcript, samples, n_samples, t_offset_cs, n_threads);
    out.reserve(lib.size());
    for (auto& w : lib) {
        crispasr_word cw;
        cw.text = std::move(w.text);
        cw.t0 = w.t0_cs;
        cw.t1 = w.t1_cs;
        out.push_back(std::move(cw));
    }
    return out;
}
