// A strict native guard for the Java wrapper, without loading a model.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

void* crispasr_align_words_abi(const char* model, const char* text, const float* pcm, int n, int64_t offset,
                               int threads) {
    const char expected_model[] = "/tmp/\xd9\x85\xd9\x88\xd8\xaf\xd9\x8a\xd9\x84.gguf";
    if (!model || strcmp(model, expected_model) || !text || strcmp(text, "\xd8\xa8") || !pcm || n != 1 ||
        pcm[0] != 0.25f || offset != 325 || threads != 4) {
        fputs("JAVA_UTF8_INPUT_REJECTED\n", stderr);
        return NULL;
    }
    return (void*)(uintptr_t)1;
}
int crispasr_align_result_n_words(void* r) {
    return 1;
}
const char* crispasr_align_result_word_text(void* r, int i) {
    return "\xd8\xa8";
}
int64_t crispasr_align_result_word_t0(void* r, int i) {
    return 325;
}
int64_t crispasr_align_result_word_t1(void* r, int i) {
    return 327;
}
int crispasr_align_result_n_characters(void* r, int w) {
    return 1;
}
const char* crispasr_align_result_character_text(void* r, int w, int i) {
    return "\xd8\xa8";
}
int64_t crispasr_align_result_character_t0(void* r, int w, int i) {
    return 325;
}
int64_t crispasr_align_result_character_t1(void* r, int w, int i) {
    return 327;
}
void crispasr_align_result_free(void* r) {}
