// test-crispasr-diff-compare.cpp — crispasr_diff::Ref::compare must fail a
// candidate the runtime never wrote. An all-zero buffer used to skip every
// row in the cosine loop and keep cos_min at its 1.0 seed, so parakeet's
// missing encoder_layer_23 dump scored cos=1.000000 PASS (#445).

#include <catch2/catch_test_macros.hpp>

#include "crispasr_diff.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <string>
#include <vector>

static std::string write_ref(const std::vector<float>& vals, int rows, int cols) {
    ggml_init_params ip = {ggml_tensor_overhead() * 4 + vals.size() * sizeof(float) + 1024, nullptr, false};
    ggml_context* ctx = ggml_init(ip);
    ggml_tensor* t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
    ggml_set_name(t, "stage");
    std::copy(vals.begin(), vals.end(), (float*)t->data);
    gguf_context* g = gguf_init_empty();
    gguf_add_tensor(g, t);
    const std::string path =
        std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") + "/crispasr-diff-compare-test.gguf";
    REQUIRE(gguf_write_to_file(g, path.c_str(), false));
    gguf_free(g);
    ggml_free(ctx);
    return path;
}

TEST_CASE("diff compare: an unwritten (all-zero) candidate fails", "[diff]") {
    const int rows = 3, cols = 4;
    std::vector<float> ref_vals(rows * cols);
    for (size_t i = 0; i < ref_vals.size(); i++)
        ref_vals[i] = 0.1f * (float)(i + 1);
    const std::string path = write_ref(ref_vals, rows, cols);

    crispasr_diff::Ref ref;
    REQUIRE(ref.load(path));

    std::vector<float> zeros(rows * cols, 0.0f);
    auto r0 = ref.compare("stage", zeros.data(), zeros.size());
    CHECK(r0.found);
    CHECK(r0.cos_min <= 0.0f);
    CHECK_FALSE(r0.is_pass(0.999f));

    // Control: the identical data passes, so the check above is not failing
    // for an unrelated reason.
    auto r1 = ref.compare("stage", ref_vals.data(), ref_vals.size());
    CHECK(r1.is_pass(0.999f));
    CHECK(r1.cos_min > 0.9999f);

    // One zero row among good ones still fails.
    std::vector<float> one_zero_row = ref_vals;
    for (int k = 0; k < cols; k++)
        one_zero_row[cols + k] = 0.0f;
    auto r2 = ref.compare("stage", one_zero_row.data(), one_zero_row.size());
    CHECK_FALSE(r2.is_pass(0.999f));

    std::remove(path.c_str());
}
