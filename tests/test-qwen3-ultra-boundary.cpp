#include "../src/core/qwen3_ultra_boundary.h"
#include "qwen3_ultra_boundary_fixture.h"
#include <cmath>
#include <cstdio>
#include <limits>

using namespace qwen3_ultra_boundary;

int main() {
    if (valid_timestamp_classes(0, 5000) != 1 || valid_timestamp_classes(1279, 5000) != 1 ||
        valid_timestamp_classes(1280, 5000) != 2 || valid_timestamp_classes(32000, 2) != 2 ||
        valid_timestamp_classes(-1, 5000) != 0 || valid_timestamp_classes(1, 0) != 0)
        return 13;
    std::array<float, 13> logits{};
    double offset = 0.0;
    if (!decode_offset(logits, offset) || std::abs(offset) > 1e-10)
        return 1;
    logits[12] = 1000.0f;
    if (!decode_offset(logits, offset) || std::abs(offset - 60.0) > 1e-8)
        return 2;
    logits[0] = std::numeric_limits<float>::infinity();
    if (decode_offset(logits, offset))
        return 3;

    std::vector<Interval> coarse;
    std::vector<double> offsets;
    for (const auto& row : kIntervals)
        coarse.push_back({row[0], row[1]});
    for (const auto& row : kLogits) {
        if (!decode_offset(row, offset))
            return 4;
        offsets.push_back(offset);
    }
    std::vector<Interval> actual;
    if (!repair(coarse, offsets, 11000.0, actual) || actual.size() != kIntervals.size())
        return 5;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::abs(actual[i].start - kIntervals[i][2]) > 0.002 ||
            std::abs(actual[i].end - kIntervals[i][3]) > 0.002) {
            std::fprintf(stderr, "word %zu: got %.6f %.6f, expected %.6f %.6f\n", i, actual[i].start, actual[i].end,
                         kIntervals[i][2], kIntervals[i][3]);
            return 6;
        }
    }
    if (repair(coarse, {}, 11000.0, actual) || repair({}, {}, 0.0, actual))
        return 7;
    if (!repair({{0.0, 40.0}, {20.0, 25.0}}, {0, 0, 0, 0}, 30.0, actual))
        return 8;
    if (!(actual[0].start >= 0 && actual[0].end <= actual[1].start && actual[1].end <= 30.0))
        return 9;
    if (!repair({{0.0, 3.0}, {2.0, 4.0}, {4.0, 6.0}}, {0, 0, 0, 0, 0, 0}, 15.0, actual))
        return 10;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i].start < 0 || actual[i].end > 15.0 || actual[i].end <= actual[i].start ||
            (i > 0 && actual[i - 1].end > actual[i].start))
            return 11;
    }
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (!repair({{20.0, 50.0}}, {nan, 0.0}, 100.0, actual) || actual[0].start != 20.0 || actual[0].end != 50.0)
        return 12;
    std::puts("Ultra boundary reference and edge controls passed");
    return 0;
}
