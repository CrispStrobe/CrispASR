// Model-free ABI oracle: consent/roster transport, ownership and native no-match semantics.
#include "crispasr_session.h"
#include <stdlib.h>
#include <string.h>
static int db_mode, db_live, db_calls;
void probe_db_mode(int mode) {
    db_mode = mode;
}
int probe_db_live(void) {
    return db_live;
}
int probe_db_calls(void) {
    return db_calls;
}
void* crispasr_speaker_db_open(const char* dir, const char* roster, int32_t consent) {
    ++db_calls;
    if (db_mode == 2 || !dir || strcmp(dir, "db") || !roster || strcmp(roster, "Alice,Bob") || consent != 1)
        return NULL;
    void* db = malloc(1);
    if (db)
        ++db_live;
    return db;
}
void crispasr_speaker_db_free(void* db) {
    if (db) {
        --db_live;
        free(db);
    }
}
int32_t crispasr_speaker_db_count(const void* db) {
    ++db_calls;
    if (!db)
        abort();
    return 2;
}
float crispasr_speaker_db_match(const void* db, const float* emb, int32_t dim, float threshold, char* name,
                                int32_t cap) {
    ++db_calls;
    // Record a closed-handle dispatch without crashing the coercion guard test.
    if (!db)
        return -1.f;
    if (!emb || dim != 3 || emb[0] != 1 || emb[1] != 3 || emb[2] != 5 || cap < 10)
        abort();
    if (db_mode == 1)
        return -1.f;
    const char text[] = "\xe3\x82\xa2\xe3\x83\xaa\xe3\x82\xb9";
    memcpy(name, text, sizeof(text));
    return .875f;
}
int32_t crispasr_speaker_db_enroll2(const char* dir, const char* name, const float* emb, int32_t dim, int32_t consent) {
    ++db_calls;
    if (!dir || strcmp(dir, "db") || !name || strcmp(name, "Alice") || !emb || dim != 3 || emb[0] != 1 || consent != 1)
        return 2;
    return db_mode == 3 ? 1 : 0;
}
