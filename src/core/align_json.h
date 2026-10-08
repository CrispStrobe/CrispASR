#pragma once
#include "crispasr_aligner.h"
#include "../../examples/json.hpp"

namespace core_align_json {
inline nlohmann::json word(const CrispasrAlignedWord& value) {
    nlohmann::json out = {{"word", value.text}, {"start", value.t0_cs / 100.0}, {"end", value.t1_cs / 100.0}};
    if (!value.characters.empty()) {
        out["characters"] = nlohmann::json::array();
        for (const auto& ch : value.characters)
            out["characters"].push_back({{"char", ch.text}, {"start", ch.t0_cs / 100.0}, {"end", ch.t1_cs / 100.0}});
    }
    return out;
}
} // namespace core_align_json
