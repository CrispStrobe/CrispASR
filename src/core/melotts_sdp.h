#pragma once

#include <cstdint>
#include <string>

namespace core_melotts {

// The original SDP ModuleList alternates ConvFlow/Flip after affine slot 0.
// In reverse inference it drops ConvFlow slot 1, retaining the adjacent Flip.
// Historical GGUF converters counted "dp" (the deterministic predictor), so
// infer these two counts from the actual "sdp" tensor inventory instead.
template <typename HasTensor>
inline bool sdp_inference_counts(HasTensor has, uint32_t& active_flows, uint32_t& dds_layers) {
    uint32_t trained_flows = 0;
    bool gap = false;
    for (uint32_t i = 0; i <= 6; ++i) {
        const bool present = has("sdp.flows." + std::to_string(2 * i + 1) + ".pre.weight");
        if (present && (gap || i == 6))
            return false;
        if (present)
            ++trained_flows;
        else
            gap = true;
    }
    uint32_t layers = 0;
    gap = false;
    for (uint32_t i = 0; i <= 16; ++i) {
        const bool present = has("sdp.convs.convs_sep." + std::to_string(i) + ".weight");
        if (present && (gap || i == 16))
            return false;
        if (present)
            ++layers;
        else
            gap = true;
    }
    if (trained_flows < 2 || layers == 0)
        return false;
    active_flows = trained_flows - 1;
    dds_layers = layers;
    return true;
}

} // namespace core_melotts
