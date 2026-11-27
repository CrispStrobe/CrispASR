// crispasr_bt2_sampling.h — per-call sampling knobs for the bt2-tts adapter.
//
// #609 follow-up: the adapter applied --temperature once at init, so a server
// request's `temperature` was ignored and only the startup value ever ran; the
// native top-k/top-p/repetition-penalty knobs were not reachable at all. The
// adapter now resolves these from each call's params. Header-only so the
// resolution can be unit-tested without loading a model.
#pragma once

struct crispasr_bt2_sampling {
    float temperature;        // backbone temperature
    int top_k;                // <= 0: checkpoint value
    float top_p;              // <= 0: checkpoint value
    float repetition_penalty; // <= 0: checkpoint value
};

// `startup_temperature` is what init resolved (the --temperature the server
// was started with, else the runtime default). A request without a
// temperature (<= 0) gets that value back, never the previous request's.
inline crispasr_bt2_sampling crispasr_bt2_resolve_sampling(float temperature, int top_k, float top_p,
                                                           float repetition_penalty, float startup_temperature) {
    crispasr_bt2_sampling s;
    s.temperature = temperature > 0.0f ? temperature : startup_temperature;
    s.top_k = top_k > 0 ? top_k : 0;
    s.top_p = top_p > 0.0f ? top_p : 0.0f;
    s.repetition_penalty = repetition_penalty > 0.0f ? repetition_penalty : 0.0f;
    return s;
}
