// speaker_db.cpp — see speaker_db.h for format and API description.

#include "speaker_db.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

static const char kMagic[4] = {'S', 'P', 'K', 'R'};
static const uint32_t kVersion = 1;

struct speaker_profile {
    std::string name;
    std::vector<float> embedding;
};

struct speaker_db {
    std::string dir_path;
    std::vector<speaker_profile> speakers;
};

// Read a .spkr file. Returns false if the file is invalid.
static bool read_spkr_file(const char* path, speaker_profile& out) {
    FILE* f = fopen(path, "rb");
    if (!f)
        return false;

    char magic[4];
    uint32_t version = 0, dim = 0;
    bool ok = fread(magic, 1, 4, f) == 4 && memcmp(magic, kMagic, 4) == 0 && fread(&version, 4, 1, f) == 1 &&
              version == kVersion && fread(&dim, 4, 1, f) == 1 && dim > 0 && dim <= 4096;

    if (ok) {
        out.embedding.resize(dim);
        ok = fread(out.embedding.data(), sizeof(float), dim, f) == dim;
    }
    fclose(f);
    return ok;
}

extern "C" struct speaker_db* speaker_db_load(const char* dir_path) {
    if (!dir_path)
        return nullptr;

    auto* db = new speaker_db();
    db->dir_path = dir_path;

#ifdef _WIN32
    std::string pattern = std::string(dir_path) + "\\*.spkr";
    WIN32_FIND_DATAA fd;
    HANDLE hFind = FindFirstFileA(pattern.c_str(), &fd);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            std::string fname = fd.cFileName;
            std::string fullpath = std::string(dir_path) + "\\" + fname;
            speaker_profile sp;
            if (read_spkr_file(fullpath.c_str(), sp)) {
                sp.name = fname.substr(0, fname.size() - 5); // strip ".spkr"
                db->speakers.push_back(std::move(sp));
            }
        } while (FindNextFileA(hFind, &fd));
        FindClose(hFind);
    }
#else
    DIR* d = opendir(dir_path);
    if (!d) {
        // Directory doesn't exist — return empty db (not an error, user may create it later)
        return db;
    }
    struct dirent* ent;
    while ((ent = readdir(d)) != nullptr) {
        std::string fname = ent->d_name;
        if (fname.size() <= 5 || fname.substr(fname.size() - 5) != ".spkr")
            continue;
        std::string fullpath = std::string(dir_path) + "/" + fname;
        speaker_profile sp;
        if (read_spkr_file(fullpath.c_str(), sp)) {
            sp.name = fname.substr(0, fname.size() - 5);
            db->speakers.push_back(std::move(sp));
        }
    }
    closedir(d);
#endif

    if (!db->speakers.empty())
        fprintf(stderr, "speaker_db: loaded %zu speakers from %s\n", db->speakers.size(), dir_path);

    return db;
}

extern "C" void speaker_db_free(struct speaker_db* db) {
    delete db;
}

extern "C" int speaker_db_count(const struct speaker_db* db) {
    return db ? (int)db->speakers.size() : 0;
}

extern "C" const char* speaker_db_match(const struct speaker_db* db, const float* embedding, int dim, float threshold,
                                        float* out_score) {
    if (!db || !embedding || dim <= 0 || db->speakers.empty())
        return nullptr;

    float best_score = -2.0f;
    int best_idx = -1;

    for (int i = 0; i < (int)db->speakers.size(); i++) {
        auto& sp = db->speakers[i];
        if ((int)sp.embedding.size() != dim)
            continue;

        // Cosine similarity (dot product for L2-normed vectors)
        float dot = 0.0f;
        for (int k = 0; k < dim; k++)
            dot += embedding[k] * sp.embedding[k];

        if (dot > best_score) {
            best_score = dot;
            best_idx = i;
        }
    }

    if (out_score)
        *out_score = best_score;

    if (best_idx >= 0 && best_score >= threshold)
        return db->speakers[best_idx].name.c_str();

    return nullptr;
}

extern "C" bool speaker_db_enroll(const char* dir_path, const char* name, const float* embedding, int dim) {
    if (!dir_path || !name || !embedding || dim <= 0)
        return false;

        // Ensure directory exists
#ifdef _WIN32
    CreateDirectoryA(dir_path, nullptr);
#else
    mkdir(dir_path, 0755);
#endif

    std::string path = std::string(dir_path) + "/" + std::string(name) + ".spkr";
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "speaker_db: cannot write %s\n", path.c_str());
        return false;
    }

    uint32_t udim = (uint32_t)dim;
    bool ok = fwrite(kMagic, 1, 4, f) == 4 && fwrite(&kVersion, 4, 1, f) == 1 && fwrite(&udim, 4, 1, f) == 1 &&
              fwrite(embedding, sizeof(float), dim, f) == (size_t)dim;

    fclose(f);
    if (ok)
        fprintf(stderr, "speaker_db: enrolled '%s' → %s (%d-d)\n", name, path.c_str(), dim);
    return ok;
}
