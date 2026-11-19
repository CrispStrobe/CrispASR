// Steering directions (steer.gguf): raw-space controls move the speaker embedding, hidden-space controls add
// to Full AttnRes bank sources of generated frames. Semantics follow model/src/zonos3/tts/steering.py and
// directions.py. A set may carry its calibration.json (zonos3_steer.set.calibrations, "" = none): served by
// /v2/controls and the speakers' disabled_controls, never applied to the controls (they stay raw doses).
#pragma once
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>
struct zonos3_model;

constexpr int ZONOS3_STEER_RAMP_FRAMES = 3;          // 0.2 s at 15 frames/s
constexpr double ZONOS3_STEER_FRAMES_PER_CHAR = 1.0; // default control_span_rate

struct zonos3_steer_direction {
    std::string id, set, name; // id = "<set>.<name>"
    bool axis = false, hidden = false;
    int source = -1;        // bank source (hidden only)
    std::vector<float> vec; // [speaker_dim] raw | [hidden_size] hidden
};
// Per-speaker, per-direction raw dose chosen offline (directions.Calibration). A dose of 0 disables the direction
// for that speaker; a "<name>-" key is the negative pole's own dose.
struct zonos3_steer_calibration {
    std::map<std::string, std::map<std::string, double>> by_speaker;
    std::map<std::string, double> defaults; // Python `default`
    double global_default = 1.0;
    std::string document_json; // Calibration.document() as JSON, file key order
    // `speaker` empty = no speaker key.
    double strength(const std::string& speaker, const std::string& name) const;
    // Calibration.disabled: the names whose dose is 0 for `speaker`, "<name>-" when only the negative pole is 0.
    std::vector<std::string> disabled(const std::string& speaker, const std::vector<std::string>& names) const;
    static bool parse(const std::string& json_text, zonos3_steer_calibration& out, std::string& err);
};
struct zonos3_steer_set {
    std::string name;
    bool hidden = false;
    int source = -1;
    std::optional<zonos3_steer_calibration> calibration;
};
// Tempo-steered speaking-rate buckets past the condition row (bundle `rate_extension`, zonos3/tts/steering.py
// RateExtension): the pace direction, the row edges they continue, the nominal edges past the row and, per TTS
// language, the buckets measured safe with their dose; reference_sides ("slower" / "faster") also serve
// reference-mode voices.
struct zonos3_steer_rate_extension {
    std::string direction;
    std::vector<double> row_edges, below, above;
    std::map<std::string, std::map<int, double>> doses;
    std::vector<std::string> reference_sides{"slower", "faster"};
};
// Steer then hold (bundle `steer_release`, zonos3/tts/steering.py ReleaseSchedule): the defaults of the request fields
// steer_hold / steer_onset_seconds / steer_ramp_seconds and the hidden directions that stay constant (`exempt`).
struct zonos3_steer_release {
    double onset_seconds = 0.0, ramp_seconds = 0.0, hold = 1.0;
    std::vector<std::string> exempt; // file order
};
// A request's release schedule at the codec frame rate (steering.ReleaseSchedule.frames).
struct zonos3_steer_release_frames {
    int onset = 0, ramp = 0;
    double hold = 1.0;
};
struct zonos3_steer_bundle {
    std::string name;
    uint32_t speaker_dim = 0, hidden_size = 0;
    std::vector<zonos3_steer_set> sets;
    std::vector<zonos3_steer_direction> dirs;                             // file order
    std::map<std::string, std::vector<std::pair<double, double>>> timing; // id -> (dose, factor), dose-sorted
    std::string timing_json, ui_json;                                     // verbatim from the file
    std::map<std::string, size_t> index;                                  // id -> dirs[i] (zonos3_steer_index)
    std::vector<int> sources;                                             // ascending distinct hidden sources
    std::optional<zonos3_steer_rate_extension> rate_extension;            // zonos3_steer.rate_extension ("" = none)
    std::optional<zonos3_steer_release> steer_release;                    // zonos3_steer.steer_release ("" = none)
};
void zonos3_steer_index(zonos3_steer_bundle& b);
bool zonos3_steer_load(zonos3_steer_bundle& b, const char* path, const zonos3_model& model, std::string& err);
std::string zonos3_steer_controls_json(const zonos3_steer_bundle& b, int embedding_dim);
// steering.release_document: the bundle's steer_release as {onset_seconds, ramp_seconds, hold, exempt}, or null.
std::string zonos3_steer_release_json(const zonos3_steer_bundle* b);
// steering.request_release: a request's steer_hold / steer_onset_seconds / steer_ramp_seconds (NaN = unset) over the
// bundle's defaults (b nullptr = no bundle). nullopt: the request-level controls stay constant (hold 1). A hold below 1
// without the bundle key fails (the server words the no-bundle case with its own hint).
bool zonos3_steer_request_release(const zonos3_steer_bundle* b, double hold, double onset_seconds, double ramp_seconds,
                                  double frame_rate, std::optional<zonos3_steer_release_frames>& out, std::string& err);
// steering.disabled_controls: per calibrated set, in direction order, "<set>.<name>" (or "<set>.<name>-") of every
// direction calibrated to 0 for `speaker`.
std::vector<std::string> zonos3_steer_disabled_controls(const zonos3_steer_bundle& b, const std::string& speaker);
double zonos3_steer_duration_factor(const zonos3_steer_bundle& b, const std::map<std::string, double>& doses);

struct zonos3_control_span {
    int start = 0, end = 0;
    std::map<std::string, double> controls;
};
struct zonos3_steering {
    std::vector<int> sources; // layout of every delta: [sources.size() * hidden_size]
    int hidden_size = 0;
    std::vector<float> hidden; // constant delta; empty = none
    // Steer then hold: the released request-level hidden controls (layout of `hidden`; empty = none), weighted per
    // generated frame by release_weight.
    std::vector<float> release;
    zonos3_steer_release_frames release_frames;
    // frames [start, end) of the character-rate schedule; chars = the request's character span [char_start, char_end)
    // (ResolvedControls.span_chars, for control_span_follow "reading")
    struct span {
        int start = 0, end = 0, ramp = ZONOS3_STEER_RAMP_FRAMES;
        std::vector<float> delta;
        int char_start = 0, char_end = 0;
    };
    std::vector<span> spans; // sorted, non-overlapping
    bool hidden_active() const { return !hidden.empty() || !release.empty() || !spans.empty(); }
    // steering.release_weight: 1 before onset, a linear ramp to hold over ramp frames (a step for ramp 0), hold after.
    double release_weight(int frame) const;
    // The constant delta plus the released delta at `frame` (out.clear() when both are empty).
    void base_at(int frames, std::vector<float>& out) const;
    void delta_at(int frames, std::vector<float>& out) const; // out.clear() when the delta is zero
};
// `release`: the request's schedule (nullptr = constant controls): the request-level hidden controls not exempt from
// the bundle's steer_release go to out.release instead of out.hidden.
bool zonos3_steer_resolve(const zonos3_steer_bundle& b, const std::map<std::string, double>& controls,
                          const std::vector<zonos3_control_span>& spans, int text_chars, double frames_per_char,
                          std::vector<float>* speaker_embedding, // in/out; nullptr = no embedding-mode speaker
                          zonos3_steering& out, std::string& err, const zonos3_steer_release_frames* release = nullptr);
