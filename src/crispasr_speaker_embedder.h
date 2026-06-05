// crispasr_speaker_embedder.h — pluggable speaker-embedding interface
// for clustering-based diarization (#107 P3).
//
// The diarization pipeline anchors physical speaker identity by
// extracting an embedding for each speech interval and clustering
// embeddings on cosine similarity. The embedding model itself is
// behind this abstract base class so future models (ECAPA-TDNN,
// x-vector, NeMo SpeakerNet, …) can drop in without touching the
// diarize-side code. The first concrete adapter wraps the existing
// TitaNet-Large C runtime in src/titanet.h.

#pragma once

#include <memory>
#include <string>

class CrispasrSpeakerEmbedder {
public:
    virtual ~CrispasrSpeakerEmbedder() = default;

    /// Output embedding dimension (e.g. 192 for TitaNet-Large).
    virtual int dim() const = 0;

    /// Extract an L2-normalized speaker embedding from a mono 16 kHz
    /// PCM range. Returns true on success and fills `out` (which must
    /// be at least `dim()` floats long). On failure returns false and
    /// leaves `out` in an unspecified state.
    virtual bool embed(const float* pcm_16k, int n_samples, float* out) = 0;

    /// Human-readable adapter name, used in logs.
    virtual const char* name() const = 0;
};

/// Build a speaker embedder from a CLI-supplied model spec.
///
/// `model_spec` accepts (case-insensitive):
///   * an empty string -> returns nullptr (caller falls back to
///     pyannote-only diarization)
///   * `"auto"` or `"titanet"` -> auto-downloads / locates
///     titanet-large.gguf (192-d, NVIDIA TitaNet-Large)
///   * `"indextts"`, `"indextts-bigvgan"`, `"indextts-ecapa"`, or
///     `"ecapa"` -> auto-downloads / locates indextts-bigvgan.gguf
///     and uses its embedded ECAPA-TDNN (512-d)
///   * a path ending in `.gguf` -> if the filename contains
///     "indextts", loaded via the IndexTTS adapter; otherwise via
///     TitaNet
///   * any other value -> nullptr with a warning to stderr
///
/// `n_threads` controls the embedder's compute parallelism.
/// `cache_dir` overrides the auto-download cache root (defaulting to
/// `~/.cache/crispasr/`); pass an empty string for the default.
///
/// Returns nullptr on any failure; the diarize flow then runs without
/// embeddings (P2a+P2e local pyannote tracks only).
std::unique_ptr<CrispasrSpeakerEmbedder> crispasr_make_speaker_embedder(const std::string& model_spec, int n_threads,
                                                                        const std::string& cache_dir);
