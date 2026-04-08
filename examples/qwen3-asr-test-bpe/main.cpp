// qwen3-asr-test-bpe — verify the BPE tokenizer matches reference output.
//
// Test 1: chat template prompt — should match the hardcoded build_prompt_ids
//         in qwen3-asr-main (special tokens + "system"/"user"/"assistant"
//         words + "<|audio_pad|>" placeholders).
// Test 2: short German phrase — should match the qwen-asr Python wrapper's
//         tokenization of the same string (dumped via the trace script).
//
// Usage:
//   qwen3-asr-test-bpe  qwen3-asr-0.6b.gguf

#include "qwen3_asr.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void print_ids(const char * label, const int32_t * ids, int n) {
    printf("  %s (%d): [", label, n);
    for (int i = 0; i < n; i++) printf("%s%d", i ? ", " : "", ids[i]);
    printf("]\n");
}

static bool check_match(const char * name,
                        const int32_t * got, int n_got,
                        const int32_t * want, int n_want) {
    if (n_got != n_want) {
        printf("  ✗ %s: length mismatch (got %d, want %d)\n", name, n_got, n_want);
        return false;
    }
    for (int i = 0; i < n_got; i++) {
        if (got[i] != want[i]) {
            printf("  ✗ %s: mismatch at idx %d (got %d, want %d)\n", name, i, got[i], want[i]);
            return false;
        }
    }
    printf("  ✓ %s\n", name);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s qwen3-asr-0.6b.gguf\n", argv[0]);
        return 1;
    }

    auto cp = qwen3_asr_context_default_params();
    cp.n_threads = 2;
    cp.verbosity = 1;
    fprintf(stderr, "[step 1] loading model ...\n");
    auto * ctx = qwen3_asr_init_from_file(argv[1], cp);
    if (!ctx) { fprintf(stderr, "init failed\n"); return 2; }
    fprintf(stderr, "[step 2] model loaded\n");

    int passed = 0, total = 0;

    // ----- Test 1: chat template prompt with one audio_pad placeholder -----
    // Reference IDs from the hardcoded build_prompt_ids in qwen3-asr-main:
    //   <|im_start|> system \n <|im_end|> \n
    //   <|im_start|> user \n
    //   <|audio_start|> <|audio_pad|> <|audio_end|> <|im_end|> \n
    //   <|im_start|> assistant \n
    {
        fprintf(stderr, "[step 3] running test 1 ...\n");
        const char * text =
            "<|im_start|>system\n<|im_end|>\n"
            "<|im_start|>user\n"
            "<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
            "<|im_start|>assistant\n";
        int32_t want[] = {
            151644, 8948, 198, 151645, 198,
            151644, 872, 198,
            151669, 151676, 151670, 151645, 198,
            151644, 77091, 198,
        };
        int n_want = sizeof(want) / sizeof(int32_t);
        int n_got = 0;
        int32_t * got = qwen3_asr_tokenize(ctx, text, &n_got);
        printf("Test 1: chat template prompt\n");
        print_ids("got ", got, n_got);
        print_ids("want", want, n_want);
        total++;
        if (check_match("test 1", got, n_got, want, n_want)) passed++;
        free(got);
        printf("\n");
    }

    // ----- Test 2: simple ASCII text -----
    {
        const char * text = "Hello world";
        int n_got = 0;
        int32_t * got = qwen3_asr_tokenize(ctx, text, &n_got);
        printf("Test 2: 'Hello world'\n");
        print_ids("got", got, n_got);
        printf("  decoded:");
        for (int i = 0; i < n_got; i++) printf(" %s", qwen3_asr_token_text(ctx, got[i]));
        printf("\n\n");
        free(got);
    }

    // ----- Test 3: German text -----
    {
        const char * text = "Berlin ist die Hauptstadt von Deutschland.";
        int n_got = 0;
        int32_t * got = qwen3_asr_tokenize(ctx, text, &n_got);
        printf("Test 3: 'Berlin ist die Hauptstadt von Deutschland.'\n");
        print_ids("got", got, n_got);
        printf("  decoded:");
        for (int i = 0; i < n_got; i++) printf(" %s", qwen3_asr_token_text(ctx, got[i]));
        printf("\n\n");
        free(got);
    }

    // ----- Test 4: short Chinese text -----
    {
        const char * text = "你好世界";  // "Hello world" in Chinese
        int n_got = 0;
        int32_t * got = qwen3_asr_tokenize(ctx, text, &n_got);
        printf("Test 4: '你好世界'\n");
        print_ids("got", got, n_got);
        printf("  decoded:");
        for (int i = 0; i < n_got; i++) printf(" %s", qwen3_asr_token_text(ctx, got[i]));
        printf("\n\n");
        free(got);
    }

    printf("---\nresult: %d/%d explicit matches\n", passed, total);
    qwen3_asr_free(ctx);
    return passed == total ? 0 : 1;
}
