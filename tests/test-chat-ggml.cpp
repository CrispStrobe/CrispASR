// test-chat-ggml.cpp — end-to-end smoke for the crispasr_chat_* C ABI.
//
// Gated on CRISPASR_CHAT_TEST_MODEL — a path to a small GGUF chat model
// (e.g. harrier-270m-q4_k.gguf, qwen2.5-0.5b-instruct, smollm2-360m).
// When unset the test is reported as SKIPPED so unrelated builds stay
// green without a model on disk.
//
// Verifies in one pass:
//   • crispasr_chat_open with default params returns a session
//   • crispasr_chat_n_ctx / _template_name return non-trivial values
//   • crispasr_chat_generate returns non-empty UTF-8 (one-shot path)
//   • crispasr_chat_generate_stream fires on_token at least once and
//     the concatenated chunks equal the one-shot output for the same
//     seed (regression guard against streaming drift)
//   • crispasr_chat_reset clears history without crashing
//   • a prompt longer than n_batch still prefills, and the result does
//     not depend on how many prompt batches it was split across

#include <catch2/catch_test_macros.hpp>

#include "crispasr_chat.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>

namespace {

const char* test_model_path() {
    return std::getenv("CRISPASR_CHAT_TEST_MODEL");
}

void on_token_appender(const char* chunk, void* user) {
    auto* out = static_cast<std::string*>(user);
    out->append(chunk);
}

// Whitespace-separated words are never fewer tokens than words, so a
// 1080-word body is at least 1080 prompt tokens — comfortably above the
// 512-token default n_batch and comfortably below the 2048-token n_ctx
// the long-prompt cases open with.
constexpr int kLongPromptSentences = 120;

std::string long_user_message() {
    std::string body;
    body.reserve(kLongPromptSentences * 45);
    for (int i = 0; i < kLongPromptSentences; ++i) {
        body += "The quick brown fox jumps over the lazy dog. ";
    }
    body += "\nReply with the single word: fox.";
    return body;
}

// Greedy generation over the long prompt, opened with the given prompt
// batch size. Returns the generated text; asserts the whole path
// succeeded.
std::string generate_over_long_prompt(const char* model, int32_t n_batch) {
    crispasr_chat_open_params op;
    crispasr_chat_open_params_default(&op);
    op.n_gpu_layers = -1;
    op.n_ctx = 2048;
    op.n_batch = n_batch;

    crispasr_chat_error err{};
    crispasr_chat_session_t s = crispasr_chat_open(model, &op, &err);
    REQUIRE(s != nullptr);
    REQUIRE(err.code == 0);

    crispasr_chat_generate_params gp;
    crispasr_chat_generate_params_default(&gp);
    gp.max_tokens = 16;
    gp.temperature = 0.0f; // greedy → reproducible across batch splits
    gp.seed = 1;

    const std::string user = long_user_message();
    crispasr_chat_message messages[] = {
        {"system", "You are a terse assistant. Answer in one word."},
        {"user", user.c_str()},
    };

    char* out = crispasr_chat_generate(s, messages, 2, &gp, &err);
    REQUIRE(out != nullptr);
    REQUIRE(err.code == 0);
    const std::string text = out;
    crispasr_chat_string_free(out);
    crispasr_chat_close(s);
    return text;
}

crispasr_chat_session_t open_session(const char* model, int32_t n_ctx, int32_t n_batch) {
    crispasr_chat_open_params op;
    crispasr_chat_open_params_default(&op);
    op.n_gpu_layers = -1;
    op.n_ctx = n_ctx;
    op.n_batch = n_batch;
    op.n_ubatch = n_batch;

    crispasr_chat_error err{};
    crispasr_chat_session_t s = crispasr_chat_open(model, &op, &err);
    REQUIRE(s != nullptr);
    REQUIRE(err.code == 0);
    return s;
}

// Greedy generation over `messages` on an already-open session. Asserts
// the call succeeded and returns the generated text.
std::string generate_text(crispasr_chat_session_t s, const crispasr_chat_message* messages, size_t n_messages) {
    crispasr_chat_generate_params gp;
    crispasr_chat_generate_params_default(&gp);
    gp.max_tokens = 24;
    gp.temperature = 0.0f; // greedy → the same prompt always yields the same text
    gp.seed = 1;

    crispasr_chat_error err{};
    char* out = crispasr_chat_generate(s, messages, n_messages, &gp, &err);
    REQUIRE(out != nullptr);
    REQUIRE(err.code == 0);
    const std::string text = out;
    crispasr_chat_string_free(out);
    return text;
}

// The same generation on a session that has seen nothing else — the
// reference every prefix-reuse case is measured against.
std::string generate_on_fresh_session(const char* model, const crispasr_chat_message* messages, size_t n_messages) {
    crispasr_chat_session_t s = open_session(model, /*n_ctx=*/1024, /*n_batch=*/512);
    const std::string text = generate_text(s, messages, n_messages);
    crispasr_chat_close(s);
    return text;
}

// A prompt whose greedy reply is fixed and made of short, distinct pieces,
// so a stop substring can be placed inside it and the truncation pinned
// exactly. The Python and Go chat suites pin the same reply for the same
// prompt and sampler settings.
const crispasr_chat_message* counting_messages() {
    static const crispasr_chat_message msgs[] = {
        {"user", "Count from 1 to 8. Write only the numbers, one per line, nothing else."},
    };
    return msgs;
}
constexpr size_t kCountingMessages = 1;
constexpr const char* kCountingReply = "1\n2\n3\n4\n5\n6\n7\n8\n";
// What the counting prompt yields once generation stops on "5" — the match
// itself is cut off.
constexpr const char* kCountingStoppedAtFive = "1\n2\n3\n4\n";

crispasr_chat_generate_params counting_params(int32_t max_tokens) {
    crispasr_chat_generate_params gp;
    crispasr_chat_generate_params_default(&gp);
    gp.max_tokens = max_tokens;
    gp.temperature = 0.0f; // greedy → the reply above is reproducible
    gp.seed = 1;
    return gp;
}

// Records how many pieces were delivered and aborts once that reaches
// `abort_at_pieces`. Returns false to abort — the ABI's convention.
struct abort_after_pieces {
    int pieces = 0;
    int abort_at_pieces = 0;
};

} // namespace

TEST_CASE("crispasr_chat one-shot generate", "[chat][gguf]") {
    const char* model = test_model_path();
    if (!model) {
        SKIP("CRISPASR_CHAT_TEST_MODEL not set; skipping chat smoke");
    }

    crispasr_chat_open_params op;
    crispasr_chat_open_params_default(&op);
    op.n_gpu_layers = -1;
    op.n_ctx = 1024;

    crispasr_chat_error err{};
    crispasr_chat_session_t s = crispasr_chat_open(model, &op, &err);
    REQUIRE(s != nullptr);
    REQUIRE(err.code == 0);

    REQUIRE(crispasr_chat_n_ctx(s) > 0);
    const char* tmpl = crispasr_chat_template_name(s);
    REQUIRE(tmpl != nullptr);
    REQUIRE(std::strlen(tmpl) > 0);

    crispasr_chat_generate_params gp;
    crispasr_chat_generate_params_default(&gp);
    gp.max_tokens = 16;
    gp.temperature = 0.0f; // greedy → reproducible across one-shot + stream
    gp.seed = 1;

    crispasr_chat_message messages[] = {
        {"system", "You are a terse assistant. Answer in one word."},
        {"user", "Say hello."},
    };

    char* out = crispasr_chat_generate(s, messages, 2, &gp, &err);
    REQUIRE(out != nullptr);
    REQUIRE(err.code == 0);
    REQUIRE(std::strlen(out) > 0);
    const std::string one_shot = out;
    crispasr_chat_string_free(out);

    // Streaming path with the same seed + greedy must reproduce one-shot.
    REQUIRE(crispasr_chat_reset(s, &err) == 0);
    std::string streamed;
    int32_t rc = crispasr_chat_generate_stream(s, messages, 2, &gp, on_token_appender, &streamed, &err);
    REQUIRE(rc == 0);
    REQUIRE(err.code == 0);
    REQUIRE_FALSE(streamed.empty());
    REQUIRE(streamed == one_shot);

    crispasr_chat_close(s);
}

TEST_CASE("crispasr_chat prompt longer than the prompt batch", "[chat][gguf]") {
    const char* model = test_model_path();
    if (!model) {
        SKIP("CRISPASR_CHAT_TEST_MODEL not set; skipping long-prompt prefill");
    }

    // The prompt exceeds the default 512-token n_batch and fits the 2048
    // context, so it must be prefilled in several prompt batches rather
    // than one oversized decode.
    const std::string text = generate_over_long_prompt(model, /*n_batch=*/512);
    REQUIRE_FALSE(text.empty());
}

TEST_CASE("crispasr_chat long-prompt output is independent of the prompt batch size", "[chat][gguf]") {
    const char* model = test_model_path();
    if (!model) {
        SKIP("CRISPASR_CHAT_TEST_MODEL not set; skipping prompt-batch equivalence");
    }

    // n_batch == n_ctx prefills in a single batch; the 512 default needs
    // several. Greedy sampling must not notice the difference.
    const std::string one_batch = generate_over_long_prompt(model, /*n_batch=*/2048);
    const std::string many_batches = generate_over_long_prompt(model, /*n_batch=*/512);
    REQUIRE_FALSE(one_batch.empty());
    REQUIRE(one_batch == many_batches);
}

TEST_CASE("crispasr_chat_memory_estimate sizes weights plus a context-scaled KV cache", "[chat][gguf]") {
    const char* model = test_model_path();
    if (!model) {
        SKIP("CRISPASR_CHAT_TEST_MODEL not set; skipping memory estimate");
    }

    // std::filesystem, not fseek/ftell: ftell's `long` is 32-bit on 64-bit
    // Windows, so this oracle would report nothing for the GGUFs over 2 GiB
    // that the estimate exists to guard against.
    std::error_code ec;
    const std::uintmax_t file_size = std::filesystem::file_size(std::filesystem::path(model), ec);
    REQUIRE_FALSE(ec);
    REQUIRE(file_size > 0);

    // Default params (NULL) — the model's own trained context.
    crispasr_chat_error err{};
    const size_t at_default = crispasr_chat_memory_estimate(model, nullptr, &err);
    REQUIRE(err.code == 0);
    REQUIRE(at_default > file_size);

    // The KV term is linear in n_ctx, so doubling the context doubles the
    // amount by which the estimate grows. A load path that returned before
    // reading the context / layer / embedding metadata would leave every
    // difference at zero and fail here while still reporting success.
    auto estimate_at = [&](int32_t n_ctx) {
        crispasr_chat_open_params p;
        crispasr_chat_open_params_default(&p);
        p.n_ctx = n_ctx;
        crispasr_chat_error e{};
        const size_t bytes = crispasr_chat_memory_estimate(model, &p, &e);
        REQUIRE(e.code == 0);
        return bytes;
    };

    const size_t at_1k = estimate_at(1024);
    const size_t at_2k = estimate_at(2048);
    const size_t at_4k = estimate_at(4096);

    REQUIRE(at_1k > file_size);
    REQUIRE(at_2k > at_1k);
    REQUIRE(at_4k > at_2k);
    REQUIRE(at_4k - at_2k == 2 * (at_2k - at_1k));

    // Everything outside the KV term is context-independent, so back it out
    // and the remainder still has to cover the weights on disk.
    const size_t kv_per_1k = at_2k - at_1k;
    REQUIRE(at_1k - kv_per_1k > file_size);

    // llama_context rounds the requested context up to a multiple of 256
    // before it allocates, so the estimate has to size the context the runtime
    // will really take, not the one that was asked for.
    REQUIRE(estimate_at(1000) == at_1k);
    REQUIRE(estimate_at(1024) == at_1k);
    REQUIRE(estimate_at(1025) > at_1k);
    REQUIRE(estimate_at(1025) == estimate_at(1280));

    // A missing path is rejected rather than estimated.
    crispasr_chat_error bad{};
    REQUIRE(crispasr_chat_memory_estimate(nullptr, nullptr, &bad) == 0);
    REQUIRE(bad.code != 0);
}

TEST_CASE("crispasr_chat_count_tokens counts the templated prompt", "[chat][gguf]") {
    const char* model = test_model_path();
    if (!model) {
        SKIP("CRISPASR_CHAT_TEST_MODEL not set; skipping token counting");
    }

    crispasr_chat_session_t s = open_session(model, /*n_ctx=*/1024, /*n_batch=*/512);
    crispasr_chat_error err{};

    crispasr_chat_message one[] = {{"user", "Say hello."}};
    const int32_t n_one = crispasr_chat_count_tokens(s, one, 1, &err);
    REQUIRE(n_one > 0);
    REQUIRE(err.code == 0);

    // Monotone: the same message with more text on the end, then with a
    // second message after it, never counts fewer tokens.
    crispasr_chat_message longer[] = {{"user", "Say hello. Then say it again, more slowly, in a full sentence."}};
    const int32_t n_longer = crispasr_chat_count_tokens(s, longer, 1, &err);
    REQUIRE(n_longer > n_one);

    crispasr_chat_message two[] = {
        {"user", "Say hello. Then say it again, more slowly, in a full sentence."},
        {"assistant", "Hello."},
    };
    const int32_t n_two = crispasr_chat_count_tokens(s, two, 2, &err);
    REQUIRE(n_two > n_longer);

    // An empty conversation counts the template's own opening. It is never
    // an error — see the template table below for the families that render
    // nothing there — and for this model's template it is a real cost a
    // caller budgeting a context window can see.
    // Whether it is more than zero depends on the model's template (gemma's
    // opens the assistant turn, Hy-MT2's renders nothing): the per-family
    // split is pinned by the template table below, with overrides.
    const int32_t n_empty = crispasr_chat_count_tokens(s, nullptr, 0, &err);
    REQUIRE(n_empty >= 0);
    REQUIRE(err.code == 0);
    REQUIRE(n_empty < n_one);

    // Counting neither prefills nor extends the history: a generation run
    // after all of the above still produces the same text as one run on a
    // session that was only ever counted against.
    crispasr_chat_generate_params gp;
    crispasr_chat_generate_params_default(&gp);
    gp.max_tokens = 16;
    gp.temperature = 0.0f;
    gp.seed = 1;

    char* after_counting = crispasr_chat_generate(s, one, 1, &gp, &err);
    REQUIRE(after_counting != nullptr);
    REQUIRE(err.code == 0);
    const std::string counted = after_counting;
    crispasr_chat_string_free(after_counting);
    crispasr_chat_close(s);

    crispasr_chat_session_t fresh = open_session(model, /*n_ctx=*/1024, /*n_batch=*/512);
    char* untouched = crispasr_chat_generate(fresh, one, 1, &gp, &err);
    REQUIRE(untouched != nullptr);
    const std::string baseline = untouched;
    crispasr_chat_string_free(untouched);
    crispasr_chat_close(fresh);

    REQUIRE(counted == baseline);
}

TEST_CASE("crispasr_chat_count_tokens rejects bad arguments", "[chat][gguf]") {
    const char* model = test_model_path();
    if (!model) {
        // The NULL-session half needs no model, but the executable's exit
        // code is the gate ctest reads: skipping every case keeps a
        // model-less machine reporting SKIPPED rather than a partial pass.
        SKIP("CRISPASR_CHAT_TEST_MODEL not set; skipping token-count argument checks");
    }

    crispasr_chat_message one[] = {{"user", "Say hello."}};
    crispasr_chat_error err{};
    REQUIRE(crispasr_chat_count_tokens(nullptr, one, 1, &err) < 0);
    REQUIRE(err.code != 0);

    crispasr_chat_session_t s = open_session(model, /*n_ctx=*/1024, /*n_batch=*/512);
    crispasr_chat_error msg_err{};
    REQUIRE(crispasr_chat_count_tokens(s, nullptr, 2, &msg_err) < 0);
    REQUIRE(msg_err.code != 0);

    // A NULL error pointer is allowed on every other entry point.
    REQUIRE(crispasr_chat_count_tokens(nullptr, one, 1, nullptr) < 0);

    // Counting an empty conversation is meaningful; generating from one is
    // not, and rendering the template for the count must not have made it
    // so — the model would be continuing from nowhere.
    crispasr_chat_generate_params gp;
    crispasr_chat_generate_params_default(&gp);
    gp.max_tokens = 4;
    crispasr_chat_error gen_err{};
    REQUIRE(crispasr_chat_generate(s, nullptr, 0, &gp, &gen_err) == nullptr);
    REQUIRE(gen_err.code != 0);

    crispasr_chat_close(s);
}

TEST_CASE("crispasr_chat_count_tokens on an empty conversation is template-dependent", "[chat][gguf]") {
    const char* model = test_model_path();
    if (!model) {
        SKIP("CRISPASR_CHAT_TEST_MODEL not set; skipping empty-conversation template table");
    }

    // What a template emits for zero messages is its own business, and the
    // families split two ways. Some open the assistant turn under `add_ass`;
    // the rest write only from inside their loop over the messages and so
    // render nothing at all. Both are legitimate, so both have to be a count
    // rather than a failure — the header promises the caller a number.
    //
    // The template is an open-param override, so one model exercises the whole
    // table: the count is a property of the template, not of these weights.
    struct row {
        const char* tmpl;
        bool has_opening;
    };
    const row rows[] = {
        {"gemma", true},       {"chatml", true},      {"llama3", true}, {"zephyr", true},
        {"mistral-v7", false}, {"mistral-v1", false}, {"orion", false}, {"minicpm", false},
    };

    for (const row& r : rows) {
        crispasr_chat_open_params op;
        crispasr_chat_open_params_default(&op);
        op.n_gpu_layers = -1;
        op.n_ctx = 512;
        op.chat_template = r.tmpl;

        crispasr_chat_error open_err{};
        crispasr_chat_session_t s = crispasr_chat_open(model, &op, &open_err);
        INFO("template " << r.tmpl);
        REQUIRE(s != nullptr);

        crispasr_chat_error err{};
        const int32_t n_empty = crispasr_chat_count_tokens(s, nullptr, 0, &err);
        REQUIRE(err.code == 0);
        REQUIRE(n_empty >= 0);
        if (r.has_opening) {
            REQUIRE(n_empty > 0);
        } else {
            REQUIRE(n_empty == 0);
        }

        // Whatever the empty count is, a real conversation still counts.
        crispasr_chat_message one[] = {{"user", "Say hello."}};
        crispasr_chat_error one_err{};
        const int32_t n_one = crispasr_chat_count_tokens(s, one, 1, &one_err);
        REQUIRE(one_err.code == 0);
        REQUIRE(n_one > n_empty);

        crispasr_chat_close(s);
    }
}

TEST_CASE("crispasr_chat_count_tokens matches what a prefill really decodes", "[chat][gguf]") {
    const char* model = test_model_path();
    if (!model) {
        SKIP("CRISPASR_CHAT_TEST_MODEL not set; skipping count-vs-prefill agreement");
    }

    // A count that is merely positive and monotone can still be wrong by a
    // fixed amount — a missing generation prompt, a missing BOS — and a
    // caller sizing a context window from it would overflow. The KV cache
    // is what reads the true prompt length back: with generation
    // suppressed, a prefill fills exactly one cell per prompt token, so it
    // succeeds while the prompt fits the context and fails as soon as it
    // does not. Growing the prompt one token at a time across that
    // boundary pins the count against the prefill.
    crispasr_chat_session_t s = open_session(model, /*n_ctx=*/256, /*n_batch=*/64);
    const int32_t capacity = crispasr_chat_n_ctx(s);
    REQUIRE(capacity >= 256);

    crispasr_chat_error err{};
    std::string fits;
    std::string over;
    int32_t n_fits = 0;
    int32_t n_over = 0;
    std::string body = "Count the tokens in this sentence.";
    for (int i = 0; i < 4 * 256; ++i) {
        crispasr_chat_message m[] = {{"user", body.c_str()}};
        const int32_t n = crispasr_chat_count_tokens(s, m, 1, &err);
        REQUIRE(n > 0);
        if (n > capacity) {
            over = body;
            n_over = n;
            break;
        }
        fits = body;
        n_fits = n;
        body += " x"; // one token per copy for a SentencePiece vocabulary
    }
    REQUIRE(n_fits > 0);
    // Asserted rather than assumed: a test model whose vocabulary spends
    // more than one token on " x" would step over the boundary instead of
    // landing on it, and would fail here rather than silently weakening the
    // two generate assertions below.
    REQUIRE(n_over == n_fits + 1);
    // This equality holds on a property of the vendored llama.cpp, not of
    // the counter: n_ctx is padded up to a multiple of 256 and reported
    // padded, so asking for 256 yields exactly 256 usable cells, and a
    // prefill reserves none of them. A future llama.cpp vintage that
    // changes the padding, or reserves a cell for its own use, moves this
    // line first — re-derive the capacity here when it does. Do not relax
    // it into an inequality: the generate assertions below only pin the
    // count against the prefill while the boundary is known exactly.
    REQUIRE(n_fits == capacity);

    crispasr_chat_generate_params gp;
    crispasr_chat_generate_params_default(&gp);
    gp.prefill_only = true;
    gp.temperature = 0.0f;

    crispasr_chat_message fitting[] = {{"user", fits.c_str()}};
    char* out = crispasr_chat_generate(s, fitting, 1, &gp, &err);
    REQUIRE(out != nullptr);
    REQUIRE(err.code == 0);
    crispasr_chat_string_free(out);

    REQUIRE(crispasr_chat_reset(s, &err) == 0);

    crispasr_chat_message overflowing[] = {{"user", over.c_str()}};
    crispasr_chat_error over_err{};
    out = crispasr_chat_generate(s, overflowing, 1, &gp, &over_err);
    REQUIRE(out == nullptr);
    REQUIRE(over_err.code != 0);
    REQUIRE(over_err.code != CRISPASR_CHAT_ERR_ABORTED);

    crispasr_chat_close(s);
}

TEST_CASE("crispasr_chat branches two prompts off a shared system prefix", "[chat][gguf]") {
    const char* model = test_model_path();
    if (!model) {
        SKIP("CRISPASR_CHAT_TEST_MODEL not set; skipping shared-prefix reuse");
    }

    // Two questions under one instruction block: the second diverges from
    // the first at the user turn, so only the instruction block is
    // reusable. Reusing it must not change a single sampled token.
    const char* system = "You are a terse assistant. Answer with one word and nothing else.";
    crispasr_chat_message first[] = {{"system", system}, {"user", "Name a colour."}};
    crispasr_chat_message second[] = {{"system", system}, {"user", "Name a fruit."}};

    const std::string ref_first = generate_on_fresh_session(model, first, 2);
    const std::string ref_second = generate_on_fresh_session(model, second, 2);
    REQUIRE_FALSE(ref_first.empty());
    REQUIRE_FALSE(ref_second.empty());

    crispasr_chat_session_t s = open_session(model, /*n_ctx=*/1024, /*n_batch=*/512);
    const std::string reused_first = generate_text(s, first, 2);
    const std::string reused_second = generate_text(s, second, 2);
    // Back to the first question. Its tokens are still in the history the
    // second turn was branched off, so a session whose history no longer
    // describes its cache keeps the second turn's tokens here and answers
    // the wrong question.
    const std::string reused_again = generate_text(s, first, 2);
    crispasr_chat_close(s);

    REQUIRE(reused_first == ref_first);
    REQUIRE(reused_second == ref_second);
    REQUIRE(reused_again == ref_first);
}

TEST_CASE("crispasr_chat extends a growing conversation", "[chat][gguf]") {
    const char* model = test_model_path();
    if (!model) {
        SKIP("CRISPASR_CHAT_TEST_MODEL not set; skipping append-only reuse");
    }

    // The append-only case: the second prompt contains the whole first
    // prompt plus the reply it produced, so nothing in the cache is stale.
    const char* system = "You are a terse assistant. Answer with one word and nothing else.";
    crispasr_chat_message first[] = {{"system", system}, {"user", "Name a colour."}};

    crispasr_chat_session_t s = open_session(model, /*n_ctx=*/1024, /*n_batch=*/512);
    const std::string reply = generate_text(s, first, 2);
    REQUIRE_FALSE(reply.empty());

    crispasr_chat_message grown[] = {
        {"system", system},
        {"user", "Name a colour."},
        {"assistant", reply.c_str()},
        {"user", "Name a fruit."},
    };
    const std::string continued = generate_text(s, grown, 4);
    crispasr_chat_close(s);

    REQUIRE(continued == generate_on_fresh_session(model, grown, 4));
}

TEST_CASE("crispasr_chat regenerates a prompt that is a prefix of its history", "[chat][gguf]") {
    const char* model = test_model_path();
    if (!model) {
        SKIP("CRISPASR_CHAT_TEST_MODEL not set; skipping prefix-of-history regeneration");
    }

    // Asking the same question twice: the second prompt is a strict prefix
    // of the history, whose tail is the first reply. With no prompt token
    // left to decode the model would sample from the logits of the last
    // token of that reply and continue it, so one token must always be
    // re-decoded.
    const char* system = "You are a terse assistant. Answer with one word and nothing else.";
    crispasr_chat_message ask[] = {{"system", system}, {"user", "Name a colour."}};

    const std::string reference = generate_on_fresh_session(model, ask, 2);
    REQUIRE_FALSE(reference.empty());

    crispasr_chat_session_t s = open_session(model, /*n_ctx=*/1024, /*n_batch=*/512);
    const std::string once = generate_text(s, ask, 2);
    const std::string twice = generate_text(s, ask, 2);
    crispasr_chat_close(s);

    REQUIRE(once == reference);
    REQUIRE(twice == reference);
}
