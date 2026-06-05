// speaker_db.h — simple file-based speaker profile database.
//
// Each speaker is stored as a file in a directory:
//   <db_path>/<name>.spkr
//
// File format:
//   4 bytes: magic "SPKR"
//   4 bytes: uint32 version (1)
//   4 bytes: uint32 embedding dimension (192)
//   dim * 4 bytes: float32 L2-normalized embedding
//
// Matching: cosine similarity (dot product for L2-normed vectors).

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct speaker_db;

// Load all speaker profiles from a directory. Returns NULL on failure
// (e.g. directory doesn't exist). An empty directory is valid (0 speakers).
struct speaker_db* speaker_db_load(const char* dir_path);

// Free all resources.
void speaker_db_free(struct speaker_db* db);

// Return the number of enrolled speakers.
int speaker_db_count(const struct speaker_db* db);

// Match an embedding against the database.
// Returns the best-matching speaker name if cosine_sim >= threshold.
// Returns NULL if no match or database is empty.
// The returned string is valid until the db is freed.
// If `out_score` is non-NULL, writes the best cosine similarity score.
const char* speaker_db_match(const struct speaker_db* db, const float* embedding, int dim, float threshold,
                             float* out_score);

// Enroll a speaker: save an L2-normalized embedding to <dir_path>/<name>.spkr.
// If the file already exists, it is overwritten.
// Returns true on success.
bool speaker_db_enroll(const char* dir_path, const char* name, const float* embedding, int dim);

#ifdef __cplusplus
}
#endif
