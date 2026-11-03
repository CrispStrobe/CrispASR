// Exact controls for the production SCNet CPU memory/pointwise helpers.
// Link with the repository's actual ggml CPU/base libraries. No model, Python
// implementation, or substituted ggml kernel is used as the comparison oracle.
#include "scnet_tensor.h"
#include "scnet_im2col.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
int comparisons = 0, partitions = 0, tiled_convolutions = 0, packing_bounds = 0;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

struct Context {
    ggml_context* value;
    explicit Context(bool no_alloc = false) : value(ggml_init({8 * 1024 * 1024, nullptr, no_alloc})) {
        require(value != nullptr, "ggml context allocation failed");
    }
    ~Context() { ggml_free(value); }
    operator ggml_context*() const { return value; }
};

ggml_tensor* input(ggml_context* ctx, std::array<int64_t, 4> shape, bool special = false, int seed = 0) {
    auto* tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, shape[0], shape[1], shape[2], shape[3]);
    auto* data = static_cast<float*>(tensor->data);
    for (int64_t i = 0; i < ggml_nelements(tensor); ++i) {
        data[i] = float((i * 73 + seed * 11) % 257 - 128) / float(1 << (int(i % 12) + 1));
        // Copy operations must retain signed zeros, infinities and NaN payloads.
        // Binary cases use finite values; model qualification rejects nonfinite
        // activations separately, and compares exactly the executed arithmetic.
        if (special && i % 31 < 6) {
            constexpr uint32_t bits[] = {0, 0x80000000U, 0x7f800000U, 0xff800000U, 0x7fc12345U, 0xffc56789U};
            std::memcpy(data + i, bits + i % 31, sizeof(float));
        } else if (i % 23 == 0) {
            const uint32_t zero = i % 2 ? 0x80000000U : 0;
            std::memcpy(data + i, &zero, sizeof(float));
        }
    }
    return tensor;
}

void compute(ggml_context* ctx, ggml_tensor* tensor, int threads) {
    auto* graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, tensor);
    require(ggml_graph_compute_with_ctx(ctx, graph, threads) == GGML_STATUS_SUCCESS, "ggml compute failed");
}

void equal_bytes(const ggml_tensor* expected, const ggml_tensor* actual) {
    require(expected->type == actual->type && ggml_are_same_shape(expected, actual), "output type/shape differs");
    if (std::memcmp(expected->data, actual->data, ggml_nbytes(expected)) != 0) {
        const auto* a = static_cast<const unsigned char*>(expected->data);
        const auto* b = static_cast<const unsigned char*>(actual->data);
        size_t at = 0;
        while (at < ggml_nbytes(expected) && a[at] == b[at])
            ++at;
        throw std::runtime_error(std::string("output bytes differ for ") + ggml_get_name(actual) + " at byte " +
                                 std::to_string(at));
    }
    ++comparisons;
}

void compare(ggml_context* ctx, ggml_tensor* expected, ggml_tensor* actual) {
    for (int threads : {1, 3}) {
        std::memset(expected->data, 0x5a, ggml_nbytes(expected));
        std::memset(actual->data, 0xa5, ggml_nbytes(actual));
        compute(ctx, expected, threads);
        compute(ctx, actual, threads);
        equal_bytes(expected, actual);
    }
}

// Exercise the actual callbacks with each worker in isolation. Their written
// positions must form a disjoint, complete partition, including nth > work.
// Guard words catch writes outside the allocated tensor. Expected bytes come
// from the standard ggml operation, not from the helper's indexing formulas.
void partition(ggml_tensor* expected, ggml_tensor* actual, ggml_custom_op_t operation) {
    require(actual->type == GGML_TYPE_F32, "partition test requires F32");
    constexpr uint32_t untouched = 0x7fcdead2U, guard = 0x7fcdead1U;
    const size_t count = size_t(ggml_nelements(actual)), margin = 16;
    const auto* reference = static_cast<const unsigned char*>(expected->data);
    void* saved = actual->data;
    for (int workers : {1, 2, 7, 64}) {
        std::vector<unsigned> ownership(count, 0);
        for (int ith = workers - 1; ith >= 0; --ith) {
            std::vector<uint32_t> storage(count + 2 * margin, guard);
            std::fill(storage.begin() + margin, storage.end() - margin, untouched);
            actual->data = storage.data() + margin;
            operation(actual, ith, workers, nullptr);
            for (size_t i = 0; i < margin; ++i)
                require(storage[i] == guard && storage[count + margin + i] == guard, "callback wrote past output");
            for (size_t i = 0; i < count; ++i) {
                if (storage[margin + i] == untouched)
                    continue;
                require(std::memcmp(storage.data() + margin + i, reference + i * sizeof(float), sizeof(float)) == 0,
                        "isolated worker produced wrong bytes");
                require(++ownership[i] == 1, "workers wrote overlapping output positions");
            }
        }
        require(std::all_of(ownership.begin(), ownership.end(), [](unsigned n) { return n == 1; }),
                "worker partition omitted output positions");
        ++partitions;
    }
    actual->data = saved;
}

void copy_tests() {
    std::array<int, 4> permutation{0, 1, 2, 3};
    do {
        Context ctx;
        auto* a = input(ctx, {35, 7, 5, 3}, true);
        auto* view = ggml_permute(ctx, a, permutation[0], permutation[1], permutation[2], permutation[3]);
        auto* expected = ggml_cont(ctx, view);
        auto* actual = scnet_tensor::cont(ctx, view, true);
        require(actual != view && actual->op == GGML_OP_CUSTOM, "cont must allocate a distinct custom output");
        require(std::strcmp(ggml_get_name(actual), "scnet_cpu_cont") == 0, "cont profile name missing");
        compare(ctx, expected, actual);
        partition(expected, actual, scnet_tensor::detail::cont_compute);
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    for (bool strided_first : {false, true}) {
        Context ctx;
        auto* a = input(ctx, {39, 9, 7, 3}, true);
        auto* view =
            ggml_view_4d(ctx, a, 33, 5, 4, 2, a->nb[1], a->nb[2], a->nb[3], 3 * sizeof(float) + a->nb[1] + a->nb[2]);
        if (strided_first) {
            // A legal permuted view without any unit-stride logical axis.
            view = ggml_view_4d(ctx, a, 1, 4, 3, 2, a->nb[1], a->nb[2], a->nb[3], a->nb[1]);
            view = ggml_permute(ctx, view, 3, 0, 1, 2);
        }
        auto* expected = ggml_cont(ctx, view);
        auto* actual = scnet_tensor::cont(ctx, view, true);
        compare(ctx, expected, actual);
        partition(expected, actual, scnet_tensor::detail::cont_compute);
    }
}

void concat_tests() {
    for (int axis = 0; axis < 4; ++axis) {
        Context ctx;
        std::array<int64_t, 4> a_shape{11, 5, 3, 2}, b_shape = a_shape;
        b_shape[axis] += 2;
        auto* a = input(ctx, a_shape, true, 1);
        auto* b = input(ctx, b_shape, true, 2);
        auto* expected = ggml_concat(ctx, a, b, axis);
        auto* actual = scnet_tensor::concat(ctx, a, b, axis, true);
        compare(ctx, expected, actual);
        partition(expected, actual, scnet_tensor::detail::concat_compute);
    }
    Context ctx;
    auto* source = input(ctx, {13, 9, 5, 2}, true);
    auto* a = ggml_view_4d(ctx, source, 11, 3, 5, 2, source->nb[1], source->nb[2], source->nb[3], sizeof(float));
    auto* b = ggml_view_4d(ctx, source, 11, 4, 5, 2, source->nb[1], source->nb[2], source->nb[3],
                           sizeof(float) + 4 * source->nb[1]);
    auto* expected = ggml_concat(ctx, a, b, 1);
    auto* actual = scnet_tensor::concat(ctx, a, b, 1, true);
    compare(ctx, expected, actual);
    partition(expected, actual, scnet_tensor::detail::concat_compute);
}

void pad_tests() {
    const std::array<std::array<int, 8>, 5> cases{{{0, 0, 0, 0, 0, 0, 0, 0},
                                                   {1, 1, 0, 0, 0, 0, 0, 0},
                                                   {1, 1, 7, 8, 0, 0, 0, 0},
                                                   {0, 1, 1, 2, 2, 3, 1, 1},
                                                   {3, 3, 1, 1, 1, 2, 0, 1}}};
    for (const auto& p : cases) {
        Context ctx;
        auto* source = input(ctx, {13, 9, 5, 2}, true);
        auto* view = ggml_view_4d(ctx, source, 11, 5, 3, 2, source->nb[1], source->nb[2], source->nb[3],
                                  sizeof(float) + source->nb[1]);
        auto* expected = ggml_pad_ext(ctx, view, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
        auto* actual = scnet_tensor::pad_ext(ctx, view, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], true);
        require(actual->op == GGML_OP_CUSTOM, "centered padding must use custom node");
        compare(ctx, expected, actual);
        partition(expected, actual, scnet_tensor::detail::pad_compute);
    }
}

void binary_tests() {
    for (bool multiply : {false, true}) {
        for (const std::array<int64_t, 4> shape :
             {std::array<int64_t, 4>{36, 6, 4, 2}, {1, 1, 1, 1}, {1, 1, 4, 1}, {36, 1, 1, 1}, {3, 2, 2, 1}}) {
            for (bool cropped : {false, true}) {
                Context ctx;
                auto* a = input(ctx, {cropped ? 40 : 36, 6, 4, 2}, false, 1);
                if (cropped)
                    a = ggml_view_4d(ctx, a, 36, 6, 4, 2, a->nb[1], a->nb[2], a->nb[3], 2 * sizeof(float));
                auto* b = input(ctx, shape, false, 2);
                auto* expected = multiply ? ggml_mul(ctx, a, b) : ggml_add(ctx, a, b);
                auto* actual = multiply ? scnet_tensor::mul(ctx, a, b, true) : scnet_tensor::add(ctx, a, b, true);
                compare(ctx, expected, actual);
                partition(expected, actual,
                          multiply ? scnet_tensor::detail::binary_compute<true>
                                   : scnet_tensor::detail::binary_compute<false>);
            }
        }
        // ggml permits a strided broadcast vector as its second operand.
        Context ctx;
        auto* a = input(ctx, {36, 5, 3, 2}, false, 1);
        auto* b = input(ctx, {2, 3, 1, 1}, false, 2);
        b = ggml_view_2d(ctx, b, 1, 3, 2 * sizeof(float), 0);
        b = ggml_transpose(ctx, b);
        auto* expected = multiply ? ggml_mul(ctx, a, b) : ggml_add(ctx, a, b);
        auto* actual = multiply ? scnet_tensor::mul(ctx, a, b, true) : scnet_tensor::add(ctx, a, b, true);
        compare(ctx, expected, actual);
        partition(expected, actual,
                  multiply ? scnet_tensor::detail::binary_compute<true> : scnet_tensor::detail::binary_compute<false>);
    }
}

void fallback_tests() {
    Context ctx;
    auto* a = input(ctx, {7, 3, 2, 1});
    auto* b = input(ctx, {1, 1, 2, 1});
    require(scnet_tensor::cont(ctx, a, false)->op == GGML_OP_CONT, "GPU cont path must stay standard");
    require(scnet_tensor::concat(ctx, a, a, 1, false)->op == GGML_OP_CONCAT, "GPU concat path must stay standard");
    require(scnet_tensor::pad_ext(ctx, a, 1, 1, 0, 0, 0, 0, 0, 0, false)->op == GGML_OP_PAD,
            "GPU pad path must stay standard");
    require(scnet_tensor::add(ctx, a, b, false)->op == GGML_OP_ADD, "GPU add path must stay standard");
    require(scnet_tensor::mul(ctx, a, b, false)->op == GGML_OP_MUL, "GPU mul path must stay standard");
    auto* half = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 7, 3);
    require(scnet_tensor::cont(ctx, half, true)->op == GGML_OP_CONT, "F16 cont must stay standard");
    require(scnet_tensor::concat(ctx, half, half, 1, true)->op == GGML_OP_CONCAT, "F16 concat must stay standard");
    require(scnet_tensor::add(ctx, half, half, true)->op == GGML_OP_ADD, "F16 add must stay standard");
    require(scnet_tensor::mul(ctx, half, half, true)->op == GGML_OP_MUL, "F16 mul must stay standard");
    auto* permuted = ggml_transpose(ctx, a);
    require(scnet_tensor::concat(ctx, permuted, permuted, 1, true)->op == GGML_OP_CONCAT,
            "strided concat must stay standard");
    auto* asymmetric = scnet_tensor::pad_ext(ctx, a, 2, 0, 0, 3, 0, 0, 0, 0, true);
    require(asymmetric->op == GGML_OP_PAD, "uncentered pad must stay standard");
    compare(ctx, ggml_pad_ext(ctx, a, 2, 0, 0, 3, 0, 0, 0, 0), asymmetric);
}

void convolution_tiling_tests() {
    // Widths come from valid model geometry, including both sides of 32 and
    // the maximum supported raw duration. Inspect real tensor byte sizes
    // without allocating the large packing buffers.
    const std::array<std::array<int, 4>, 4> cases{
        {{872, 85, 2304, 16}, {614, 85, 2304, 23}, {5170, 85, 2304, 2}, {390, 85, 2304, 37}}};
    constexpr size_t budget = 128 * 1024 * 1024;
    for (const auto& geometry : cases) {
        const int tile = scnet_detail::im2col_column_tile_size(geometry[0], geometry[1], geometry[2]);
        require(tile == geometry[3], "unexpected canonical packing width");
        {
            Context metadata(true);
            auto* packed = ggml_new_tensor_2d(metadata, GGML_TYPE_F32, int64_t(geometry[0]) * tile, geometry[2]);
            require(packed->data == nullptr && ggml_nbytes(packed) <= budget, "packing descriptor exceeds budget");
            auto* larger = ggml_new_tensor_2d(metadata, GGML_TYPE_F32, int64_t(geometry[0]) * (tile + 1), geometry[2]);
            require(ggml_nbytes(larger) > budget, "packing selector left another fitting frequency");
            ++packing_bounds;
        }
        for (int workers : {1, 3}) {
            Context ctx;
            constexpr int times = 15, frequencies = 79, channels = 3, outputs = 5, kernel = 27;
            auto* parent = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 19, 167, channels);
            auto* source = static_cast<float*>(parent->data);
            for (int64_t i = 0; i < ggml_nelements(parent); ++i)
                source[i] = float((i * 7) % 23 - 11) / 8.0f;
            const std::vector<float> unchanged(source, source + ggml_nelements(parent));
            auto* cropped = ggml_view_3d(ctx, parent, 17, 159, channels, parent->nb[1], parent->nb[2],
                                         sizeof(float) + 4 * parent->nb[1]);
            auto* weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kernel, outputs);
            auto* w = static_cast<float*>(weights->data);
            for (int c = 0; c < outputs; ++c)
                for (int k = 0; k < kernel; ++k)
                    w[c * kernel + k] = float((c * 7 + k * 5) % 13 - 6) / 16.0f;
            auto* column_weights = ggml_transpose(ctx, weights);
            std::vector<ggml_tensor*> pieces;
            for (int begin = 0; begin < frequencies; begin += tile) {
                const int count = std::min(tile, frequencies - begin);
                auto* band = ggml_view_3d(ctx, cropped, 17, (count - 1) * 2 + 3, channels, cropped->nb[1],
                                          cropped->nb[2], size_t(begin * 2) * cropped->nb[1]);
                auto* columns = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, times * count, kernel);
                std::memset(columns->data, 0xa5, ggml_nbytes(columns));
                const scnet_detail::Im2ColShape shape{times, count, channels, 3, 3, 2};
                for (int ith = workers - 1; ith >= 0; --ith)
                    scnet_detail::im2col_columns(static_cast<const float*>(band->data),
                                                 static_cast<float*>(columns->data), shape, band->nb[1] / sizeof(float),
                                                 band->nb[2] / sizeof(float), ith, workers);
                auto* product = ggml_out_prod(ctx, columns, column_weights);
                pieces.push_back(ggml_reshape_3d(ctx, product, times, count, outputs));
            }
            // Execute the production assembly primitive, with partial tails
            // and the same balanced concatenation order as the model graph.
            while (pieces.size() > 1) {
                std::vector<ggml_tensor*> joined;
                for (size_t i = 0; i < pieces.size(); i += 2)
                    joined.push_back(i + 1 == pieces.size()
                                         ? pieces[i]
                                         : scnet_tensor::concat(ctx, pieces[i], pieces[i + 1], 1, true));
                pieces = std::move(joined);
            }
            auto* expected = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, times, frequencies, outputs);
            auto* reference = static_cast<float*>(expected->data);
            // Independent direct convolution; dyadic inputs/weights make
            // every product and sum exact in F32, including CPU OUT_PROD.
            for (int c = 0; c < outputs; ++c)
                for (int f = 0; f < frequencies; ++f)
                    for (int t = 0; t < times; ++t) {
                        double sum = 0;
                        for (int ic = 0; ic < channels; ++ic)
                            for (int kf = 0; kf < 3; ++kf)
                                for (int kt = 0; kt < 3; ++kt) {
                                    const int at = ic * 167 * 19 + (4 + 2 * f + kf) * 19 + 1 + t + kt;
                                    sum += double(source[at]) * w[c * kernel + (ic * 3 + kf) * 3 + kt];
                                }
                        reference[(c * frequencies + f) * times + t] = static_cast<float>(sum);
                    }
            compute(ctx, pieces.front(), workers);
            equal_bytes(expected, pieces.front());
            require(std::memcmp(source, unchanged.data(), ggml_nbytes(parent)) == 0, "packing modified input");
            ++tiled_convolutions;
        }
    }
    for (const auto& invalid : std::array<std::array<int64_t, 3>, 7>{
             {{0, 1, 1},
              {-1, 1, 1},
              {1, 0, 1},
              {1, 1, 0},
              {1, 1, -1},
              {1, 2, int64_t(budget / sizeof(float)) + 1},
              {std::numeric_limits<int>::max(), 2, std::numeric_limits<int64_t>::max()}}}) {
        bool rejected = false;
        try {
            scnet_detail::im2col_column_tile_size(static_cast<int>(invalid[0]), static_cast<int>(invalid[1]),
                                                  invalid[2]);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        require(rejected, "invalid or over-budget packing geometry was accepted");
        ++packing_bounds;
    }
}
} // namespace

int main() {
    try {
        copy_tests();
        concat_tests();
        pad_tests();
        binary_tests();
        fallback_tests();
        convolution_tiling_tests();
        std::cout << "PASS: " << comparisons << " exact ggml comparisons; " << partitions
                  << " disjoint worker partitions; " << tiled_convolutions << " tiled convolution assemblies; "
                  << packing_bounds << " packing budget/rejection cases; GPU/type/layout fallbacks\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
