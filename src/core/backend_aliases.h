#pragma once

#include <string>

namespace core_backend {

// Architecture detection, explicit backend dispatch and model-registry lookup
// are separate paths. Keep the Haas spellings shared by all explicit callers.
inline bool is_haas_alias(const std::string& name) {
    return name == "haas" || name == "haas-ja" || name == "haas_ja" || name == "haas-asr" || name == "haas_asr";
}

inline std::string canonical_runtime_name(const std::string& name) {
    return is_haas_alias(name) ? "parakeet" : name;
}

} // namespace core_backend
