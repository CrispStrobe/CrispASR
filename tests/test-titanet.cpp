// Quick smoke test for TitaNet speaker embedding extraction.
// Usage: test-titanet <model.gguf> <audio1.wav> [audio2.wav ...]
//
// Loads the model, extracts embeddings from each WAV file, and prints
// the cosine similarity matrix. Useful for verifying:
//   - Same speaker → cosine sim > 0.7
//   - Different speakers → cosine sim < 0.4

#include "../src/titanet.h"
#include "../src/speaker_db.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Minimal WAV loader — 16-bit PCM mono/stereo → float32 mono
static bool load_wav(const char* path, std::vector<float>& out, int& sample_rate) {
    FILE* f = fopen(path, "rb");
    if (!f)
        return false;

    char riff[4];
    uint32_t file_size;
    char wave[4];
    fread(riff, 1, 4, f);
    fread(&file_size, 4, 1, f);
    fread(wave, 1, 4, f);
    if (memcmp(riff, "RIFF", 4) || memcmp(wave, "WAVE", 4)) {
        fclose(f);
        return false;
    }

    // Find fmt chunk
    int channels = 0, bits = 0;
    sample_rate = 0;
    while (true) {
        char id[4];
        uint32_t sz;
        if (fread(id, 1, 4, f) != 4 || fread(&sz, 4, 1, f) != 1)
            break;
        if (memcmp(id, "fmt ", 4) == 0) {
            uint16_t fmt, ch;
            uint32_t sr;
            uint16_t bps;
            fread(&fmt, 2, 1, f);
            fread(&ch, 2, 1, f);
            fread(&sr, 4, 1, f);
            fseek(f, 4 + 2, SEEK_CUR); // byte rate + block align
            fread(&bps, 2, 1, f);
            channels = ch;
            sample_rate = (int)sr;
            bits = bps;
            if (sz > 16)
                fseek(f, sz - 16, SEEK_CUR);
        } else if (memcmp(id, "data", 4) == 0) {
            int n_samples = (int)(sz / (bits / 8) / channels);
            out.resize(n_samples);
            if (bits == 16) {
                std::vector<int16_t> raw(n_samples * channels);
                fread(raw.data(), 2, n_samples * channels, f);
                for (int i = 0; i < n_samples; i++) {
                    float s = 0;
                    for (int c = 0; c < channels; c++)
                        s += raw[i * channels + c];
                    out[i] = s / (channels * 32768.0f);
                }
            } else if (bits == 32) {
                // Float32
                std::vector<float> raw(n_samples * channels);
                fread(raw.data(), 4, n_samples * channels, f);
                for (int i = 0; i < n_samples; i++) {
                    float s = 0;
                    for (int c = 0; c < channels; c++)
                        s += raw[i * channels + c];
                    out[i] = s / channels;
                }
            }
            break;
        } else {
            fseek(f, sz, SEEK_CUR);
        }
    }
    fclose(f);
    return !out.empty();
}

// Load raw float32 PCM from .npy file (simple: skip 128-byte header)
static bool load_npy_f32(const char* path, std::vector<float>& out) {
    FILE* f = fopen(path, "rb");
    if (!f)
        return false;

    // Read header to find data offset and shape
    char magic[6];
    fread(magic, 1, 6, f);
    if (memcmp(magic, "\x93NUMPY", 6) != 0) {
        fclose(f);
        return false;
    }
    uint8_t major, minor;
    fread(&major, 1, 1, f);
    fread(&minor, 1, 1, f);
    uint16_t header_len;
    fread(&header_len, 2, 1, f);
    // Skip header dict
    fseek(f, header_len, SEEK_CUR);

    // Read remaining data as float32
    long pos = ftell(f);
    fseek(f, 0, SEEK_END);
    long end = ftell(f);
    fseek(f, pos, SEEK_SET);
    int n_floats = (int)((end - pos) / 4);
    out.resize(n_floats);
    fread(out.data(), 4, n_floats, f);
    fclose(f);
    return n_floats > 0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.gguf> <audio1.wav|.npy> [audio2 ...]\n", argv[0]);
        fprintf(stderr, "  Accepts .wav (16-bit PCM) or .npy (float32 PCM) files\n");
        fprintf(stderr, "\nOptions:\n");
        fprintf(stderr, "  --ref-emb <file.npy>   Compare embedding against reference\n");
        fprintf(stderr, "  --enroll <name> <dir>   Enroll speaker into DB\n");
        fprintf(stderr, "  --match <dir> <thresh>  Match against speaker DB\n");
        return 1;
    }

    const char* model_path = argv[1];

    printf("Loading TitaNet model: %s\n", model_path);
    auto* ctx = titanet_init(model_path, 4);
    if (!ctx) {
        fprintf(stderr, "ERROR: failed to load model\n");
        return 1;
    }

    // Collect audio files (skip flags)
    struct AudioFile {
        std::string path;
        std::string name;
        std::vector<float> embedding;
    };
    std::vector<AudioFile> files;
    std::string ref_emb_path;
    std::string enroll_name, enroll_dir;
    std::string match_dir;
    float match_threshold = 0.7f;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--ref-emb") == 0 && i + 1 < argc) {
            ref_emb_path = argv[++i];
        } else if (strcmp(argv[i], "--enroll") == 0 && i + 2 < argc) {
            enroll_name = argv[++i];
            enroll_dir = argv[++i];
        } else if (strcmp(argv[i], "--match") == 0 && i + 2 < argc) {
            match_dir = argv[++i];
            match_threshold = atof(argv[++i]);
        } else {
            AudioFile af;
            af.path = argv[i];
            std::string base = af.path;
            auto slash = base.rfind('/');
            if (slash != std::string::npos)
                base = base.substr(slash + 1);
            auto dot = base.rfind('.');
            if (dot != std::string::npos)
                base = base.substr(0, dot);
            af.name = base;
            files.push_back(af);
        }
    }

    // Process each audio file
    for (auto& af : files) {
        std::vector<float> pcm;
        int sr = 16000;
        bool loaded = false;

        if (af.path.size() > 4 && af.path.substr(af.path.size() - 4) == ".npy") {
            loaded = load_npy_f32(af.path.c_str(), pcm);
        } else {
            loaded = load_wav(af.path.c_str(), pcm, sr);
        }

        if (!loaded) {
            fprintf(stderr, "ERROR: cannot load %s\n", af.path.c_str());
            continue;
        }

        printf("\n%s: %d samples (%.2fs @ %dHz)\n", af.name.c_str(), (int)pcm.size(), (float)pcm.size() / sr, sr);

        af.embedding.resize(192);
        int dim = titanet_embed(ctx, pcm.data(), (int)pcm.size(), af.embedding.data());
        if (dim <= 0) {
            fprintf(stderr, "  ERROR: embedding failed\n");
            continue;
        }

        printf("  embedding[0:5]: %.6f %.6f %.6f %.6f %.6f\n", af.embedding[0], af.embedding[1], af.embedding[2],
               af.embedding[3], af.embedding[4]);

        float norm = 0;
        for (int i = 0; i < dim; i++)
            norm += af.embedding[i] * af.embedding[i];
        printf("  L2 norm: %.6f\n", sqrtf(norm));

        // Compare against reference embedding if provided
        if (!ref_emb_path.empty()) {
            std::vector<float> ref;
            if (load_npy_f32(ref_emb_path.c_str(), ref) && (int)ref.size() == dim) {
                float sim = titanet_cosine_sim(af.embedding.data(), ref.data(), dim);
                printf("  vs reference: cosine_sim = %.6f\n", sim);
            }
        }

        // Enrollment
        if (!enroll_name.empty() && !enroll_dir.empty()) {
            speaker_db_enroll(enroll_dir.c_str(), enroll_name.c_str(), af.embedding.data(), dim);
        }

        // Matching
        if (!match_dir.empty()) {
            auto* db = speaker_db_load(match_dir.c_str());
            if (db) {
                float score;
                const char* name = speaker_db_match(db, af.embedding.data(), dim, match_threshold, &score);
                if (name)
                    printf("  matched: %s (score=%.4f)\n", name, score);
                else
                    printf("  no match (best score=%.4f, threshold=%.2f)\n", score, match_threshold);
                speaker_db_free(db);
            }
        }
    }

    // Print cosine similarity matrix
    if (files.size() > 1) {
        printf("\n=== Cosine Similarity Matrix ===\n");
        printf("%15s", "");
        for (auto& af : files)
            printf(" %12s", af.name.c_str());
        printf("\n");

        for (size_t i = 0; i < files.size(); i++) {
            printf("%15s", files[i].name.c_str());
            for (size_t j = 0; j < files.size(); j++) {
                if (files[i].embedding.empty() || files[j].embedding.empty()) {
                    printf(" %12s", "N/A");
                } else {
                    float sim = titanet_cosine_sim(files[i].embedding.data(), files[j].embedding.data(), 192);
                    printf(" %12.6f", sim);
                }
            }
            printf("\n");
        }
    }

    titanet_free(ctx);
    return 0;
}
