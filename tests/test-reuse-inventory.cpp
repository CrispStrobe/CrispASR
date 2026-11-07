#include "../src/core/reuse_model_inventory.h"
#include "../examples/json.hpp"
#include <cstdio>
#include <fstream>

int main(int argc, char** argv) {
    if (argc != 2)
        return 1;
    std::ifstream file(argv[1]);
    nlohmann::json fixture;
    try {
        file >> fixture;
    } catch (...) {
        return 2;
    }
    if (!fixture.is_object() || fixture.size() != 1416)
        return 3;
    reuse_model_inventory::Inventory source;
    try {
        for (auto it = fixture.begin(); it != fixture.end(); ++it)
            source.emplace(it.key(), it.value().get<reuse_model_inventory::Shape>());
    } catch (...) {
        return 4;
    }
    const auto expected = reuse_model_inventory::expected();
    if (source != expected) {
        for (const auto& entry : source) {
            const auto it = expected.find(entry.first);
            if (it == expected.end() || it->second != entry.second) {
                std::fprintf(stderr, "missing or mismatched tensor: %s\n", entry.first.c_str());
                break;
            }
        }
        return 5;
    }
    auto invalid = source;
    invalid.erase(invalid.begin());
    if (reuse_model_inventory::validate_shapes(invalid))
        return 6;
    invalid = source;
    invalid.begin()->second[0]++;
    if (reuse_model_inventory::validate_shapes(invalid))
        return 7;
    invalid = source;
    invalid.emplace("unexpected.tensor", reuse_model_inventory::Shape{1});
    if (reuse_model_inventory::validate_shapes(invalid))
        return 8;
    if (!reuse_model_inventory::validate_shapes(source))
        return 9;
    std::printf("RE-USE original inventory: %zu exact tensor names and shapes\n", source.size());
    return 0;
}
