#include "zonos3-steer.h"
#include "zonos3.h"
#include "ggml.h"
#include "gguf.h"
#include "json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>

using ojson = nlohmann::ordered_json;

void zonos3_steer_index(zonos3_steer_bundle& b) {
    b.index.clear();
    for (size_t i = 0; i < b.dirs.size(); i++)
        b.index[b.dirs[i].id] = i;
    std::set<int> src;
    for (const auto& d : b.dirs)
        if (d.hidden)
            src.insert(d.source);
    b.sources.assign(src.begin(), src.end());
}

namespace {

struct gguf_guard {
    gguf_context* g = nullptr;
    ggml_context* c = nullptr;
    ~gguf_guard() {
        if (g)
            gguf_free(g);
        if (c)
            ggml_free(c);
    }
};

std::string invalid(const char* path, const std::string& why) {
    return std::string("'") + path + "' is not a valid zonos3 steering GGUF (" + why +
           "); reconvert with tools/convert_gguf.py steer";
}

// numpy.interp: clamped linear interpolation over x-sorted points.
double interp(const std::vector<std::pair<double, double>>& t, double x) {
    if (t.empty())
        return 1.0;
    if (x <= t.front().first)
        return t.front().second;
    if (x >= t.back().first)
        return t.back().second;
    size_t hi = 1;
    while (t[hi].first < x)
        hi++;
    const auto& a = t[hi - 1];
    const auto& b = t[hi];
    if (b.first == a.first)
        return b.second;
    return a.second + (b.second - a.second) * ((x - a.first) / (b.first - a.first));
}

bool lookup(const zonos3_steer_calibration& c, const std::string& speaker, const std::string& key, double& out) {
    if (!speaker.empty()) {
        const auto s = c.by_speaker.find(speaker);
        if (s != c.by_speaker.end()) {
            const auto e = s->second.find(key);
            if (e != s->second.end()) {
                out = e->second;
                return true;
            }
        }
    }
    const auto e = c.defaults.find(key);
    if (e != c.defaults.end()) {
        out = e->second;
        return true;
    }
    return false;
}

} // namespace

double zonos3_steer_calibration::strength(const std::string& speaker, const std::string& name) const {
    double s;
    return lookup(*this, speaker, name, s) ? s : global_default;
}

std::vector<std::string> zonos3_steer_calibration::disabled(const std::string& speaker,
                                                            const std::vector<std::string>& names) const {
    std::vector<std::string> out;
    for (const auto& n : names) {
        double s;
        if (strength(speaker, n) == 0.0)
            out.push_back(n);
        else if (lookup(*this, speaker, n + "-", s) && s == 0.0)
            out.push_back(n + "-");
    }
    return out;
}

bool zonos3_steer_calibration::parse(const std::string& json_text, zonos3_steer_calibration& out, std::string& err) {
    out = zonos3_steer_calibration();
    try {
        const ojson data = ojson::parse(json_text);
        ojson defaults = ojson::object(), by_speaker = ojson::object();
        if (data.contains("by_speaker"))
            for (const auto& sp : data.at("by_speaker").items()) {
                by_speaker[sp.key()] = ojson::object();
                for (const auto& e : sp.value().items()) {
                    const double v = e.value().get<double>();
                    out.by_speaker[sp.key()][e.key()] = v;
                    by_speaker[sp.key()][e.key()] = v;
                }
            }
        if (data.contains("default"))
            for (const auto& e : data.at("default").items()) {
                const double v = e.value().get<double>();
                out.defaults[e.key()] = v;
                defaults[e.key()] = v;
            }
        out.global_default = data.contains("global_default") ? data.at("global_default").get<double>() : 1.0;
        out.document_json =
            ojson{{"default", defaults}, {"by_speaker", by_speaker}, {"global_default", out.global_default}}.dump();
    } catch (const std::exception& e) {
        err = std::string("invalid calibration: ") + e.what();
        return false;
    }
    return true;
}

bool zonos3_steer_load(zonos3_steer_bundle& b, const char* path, const zonos3_model& model, std::string& err) {
    b = zonos3_steer_bundle{};
    gguf_guard G;
    G.g = gguf_init_from_file(path, {/*no_alloc=*/false, &G.c});
    if (!G.g) {
        err = invalid(path, "cannot read the file");
        return false;
    }
    gguf_context* g = G.g;

    auto key = [&](const char* k, int64_t& id) {
        id = gguf_find_key(g, k);
        return id >= 0;
    };
    int64_t id = -1;
    if (!key("general.architecture", id) || gguf_get_kv_type(g, id) != GGUF_TYPE_STRING ||
        strcmp(gguf_get_val_str(g, id), "zonos3_steer")) {
        err = invalid(path, "general.architecture is not zonos3_steer");
        return false;
    }
    auto get_str = [&](const char* k, std::string& out) {
        int64_t i;
        if (!key(k, i) || gguf_get_kv_type(g, i) != GGUF_TYPE_STRING)
            return false;
        out = gguf_get_val_str(g, i);
        return true;
    };
    auto get_u32 = [&](const char* k, uint32_t& out) {
        int64_t i;
        if (!key(k, i) || gguf_get_kv_type(g, i) != GGUF_TYPE_UINT32)
            return false;
        out = gguf_get_val_u32(g, i);
        return true;
    };
    auto get_strs = [&](const char* k, std::vector<std::string>& out) {
        int64_t i;
        if (!key(k, i) || gguf_get_kv_type(g, i) != GGUF_TYPE_ARRAY || gguf_get_arr_type(g, i) != GGUF_TYPE_STRING)
            return false;
        out.clear();
        for (size_t j = 0, n = gguf_get_arr_n(g, i); j < n; j++)
            out.push_back(gguf_get_arr_str(g, i, j));
        return true;
    };

    std::vector<std::string> set_names, set_spaces, set_calibrations, ids, kinds;
    std::vector<int32_t> set_sources;
    {
        int64_t i;
        if (!key("zonos3_steer.set.sources", i) || gguf_get_kv_type(g, i) != GGUF_TYPE_ARRAY ||
            gguf_get_arr_type(g, i) != GGUF_TYPE_INT32) {
            err = invalid(path, "missing zonos3_steer.set.sources");
            return false;
        }
        const int32_t* p = (const int32_t*)gguf_get_arr_data(g, i);
        set_sources.assign(p, p + gguf_get_arr_n(g, i));
    }
    if (!get_str("zonos3_steer.name", b.name) || !get_u32("zonos3_steer.speaker_dim", b.speaker_dim) ||
        !get_u32("zonos3_steer.hidden_size", b.hidden_size) || !get_strs("zonos3_steer.set.names", set_names) ||
        !get_strs("zonos3_steer.set.spaces", set_spaces) ||
        !get_strs("zonos3_steer.set.calibrations", set_calibrations) || !get_strs("zonos3_steer.direction.ids", ids) ||
        !get_strs("zonos3_steer.direction.kinds", kinds) || !get_str("zonos3_steer.timing", b.timing_json) ||
        !get_str("zonos3_steer.ui", b.ui_json)) {
        err = invalid(path, "missing or mistyped zonos3_steer.* metadata");
        return false;
    }
    if (set_names.empty() || set_names.size() != set_spaces.size() || set_names.size() != set_sources.size() ||
        set_names.size() != set_calibrations.size() || ids.empty() || ids.size() != kinds.size() ||
        (size_t)gguf_get_n_tensors(g) != ids.size()) {
        err = invalid(path, "inconsistent metadata array sizes");
        return false;
    }

    std::map<std::string, size_t> set_index;
    bool any_raw = false, any_hidden = false;
    for (size_t i = 0; i < set_names.size(); i++) {
        zonos3_steer_set s;
        s.name = set_names[i];
        if (s.name.empty() || s.name.find('.') != std::string::npos || set_index.count(s.name)) {
            err = invalid(path, "bad set name '" + s.name + "'");
            return false;
        }
        if (set_spaces[i] == "attnres_source") {
            s.hidden = true;
            s.source = set_sources[i];
            if (s.source < 0) {
                err = invalid(path, "set '" + s.name + "' has a negative source");
                return false;
            }
            any_hidden = true;
        } else if (set_spaces[i] == "raw") {
            any_raw = true;
        } else {
            err = invalid(path, "set '" + s.name + "' has unknown space '" + set_spaces[i] + "'");
            return false;
        }
        if (!set_calibrations[i].empty()) {
            zonos3_steer_calibration c;
            std::string why;
            if (!zonos3_steer_calibration::parse(set_calibrations[i], c, why)) {
                err = invalid(path, "set '" + s.name + "': " + why);
                return false;
            }
            s.calibration = std::move(c);
        }
        set_index[s.name] = b.sets.size();
        b.sets.push_back(std::move(s));
    }
    if ((any_raw && b.speaker_dim == 0) || (any_hidden && b.hidden_size == 0)) {
        err = invalid(path, "a set space has no dimension");
        return false;
    }

    for (size_t i = 0; i < ids.size(); i++) {
        zonos3_steer_direction d;
        d.id = ids[i];
        const size_t dot = d.id.find('.');
        if (dot == std::string::npos || dot + 1 >= d.id.size()) {
            err = invalid(path, "bad direction id '" + d.id + "'");
            return false;
        }
        d.set = d.id.substr(0, dot);
        d.name = d.id.substr(dot + 1);
        auto it = set_index.find(d.set);
        if (it == set_index.end()) {
            err = invalid(path, "direction '" + d.id + "' names an unknown set");
            return false;
        }
        if (kinds[i] != "named" && kinds[i] != "axis") {
            err = invalid(path, "direction '" + d.id + "' has unknown kind");
            return false;
        }
        d.axis = kinds[i] == "axis";
        d.hidden = b.sets[it->second].hidden;
        d.source = b.sets[it->second].source;
        const std::string tname = "steer.dir." + std::to_string(i);
        ggml_tensor* t = ggml_get_tensor(G.c, tname.c_str());
        const int64_t dim = d.hidden ? b.hidden_size : b.speaker_dim;
        if (!t || t->type != GGML_TYPE_F32 || ggml_n_dims(t) != 1 || t->ne[0] != dim) {
            err = invalid(path, "tensor " + tname + " is missing or is not F32 [" + std::to_string(dim) + "]");
            return false;
        }
        d.vec.assign((const float*)t->data, (const float*)t->data + dim);
        for (float v : d.vec)
            if (!std::isfinite(v)) {
                err = invalid(path, "tensor " + tname + " has non-finite values");
                return false;
            }
        b.dirs.push_back(std::move(d));
    }
    zonos3_steer_index(b);
    if (b.index.size() != b.dirs.size()) {
        err = invalid(path, "duplicate direction ids");
        return false;
    }

    // timing: { id: [[dose, factor], ...] }
    try {
        const nlohmann::json tj = nlohmann::json::parse(b.timing_json);
        if (!tj.is_object())
            throw std::runtime_error("timing is not an object");
        for (auto it = tj.begin(); it != tj.end(); ++it) {
            if (!b.index.count(it.key())) {
                err = invalid(path, "timing for unknown direction '" + it.key() + "'");
                return false;
            }
            if (!it.value().is_array() || it.value().empty())
                throw std::runtime_error("empty timing table");
            std::vector<std::pair<double, double>> table;
            for (const auto& p : it.value()) {
                if (!p.is_array() || p.size() != 2 || !p[0].is_number() || !p[1].is_number())
                    throw std::runtime_error("bad timing pair");
                const double x = p[0].get<double>(), y = p[1].get<double>();
                if (!std::isfinite(x) || !std::isfinite(y))
                    throw std::runtime_error("non-finite timing value");
                table.emplace_back(x, y);
            }
            std::stable_sort(table.begin(), table.end(),
                             [](const auto& l, const auto& r) { return l.first < r.first; });
            b.timing[it.key()] = std::move(table);
        }
        if (!ojson::parse(b.ui_json).is_object())
            throw std::runtime_error("ui is not an object");
    } catch (const std::exception& e) {
        err = invalid(path, std::string("bad timing/ui JSON: ") + e.what());
        return false;
    }

    // rate_extension (steering.py load_bundle): optional key; "" = none.
    std::string rate_json;
    if (get_str("zonos3_steer.rate_extension", rate_json) && !rate_json.empty()) {
        try {
            const nlohmann::json r = nlohmann::json::parse(rate_json);
            if (r.is_object() && !r.empty()) {
                zonos3_steer_rate_extension ext;
                ext.direction = r.at("direction").get<std::string>();
                const auto d = b.index.find(ext.direction);
                if (d == b.index.end() || !b.dirs[d->second].hidden) {
                    err = invalid(path, "rate_extension direction '" + ext.direction +
                                            "' is not a hidden direction of the bundle");
                    return false;
                }
                ext.row_edges = r.at("row_edges").get<std::vector<double>>();
                ext.below = r.at("below").get<std::vector<double>>();
                ext.above = r.at("above").get<std::vector<double>>();
                for (auto lang = r.at("languages").begin(); lang != r.at("languages").end(); ++lang)
                    for (auto bd = lang.value().begin(); bd != lang.value().end(); ++bd)
                        ext.doses[lang.key()][std::stoi(bd.key())] = bd.value().get<double>();
                if (r.contains("reference_sides"))
                    ext.reference_sides = r["reference_sides"].get<std::vector<std::string>>();
                b.rate_extension = std::move(ext);
            }
        } catch (const std::exception& e) {
            err = invalid(path, std::string("bad rate_extension JSON: ") + e.what());
            return false;
        }
    }

    // steer_release (steering.py load_bundle): optional key; "" = none.
    std::string release_json;
    if (get_str("zonos3_steer.steer_release", release_json) && !release_json.empty()) {
        try {
            const nlohmann::json r = nlohmann::json::parse(release_json);
            bool shape =
                r.is_object() && r.contains("onset_seconds") && r.contains("ramp_seconds") && r.contains("hold");
            if (shape)
                for (auto it = r.begin(); it != r.end(); ++it)
                    if (it.key() != "onset_seconds" && it.key() != "ramp_seconds" && it.key() != "hold" &&
                        it.key() != "exempt")
                        shape = false;
            if (!shape)
                throw std::runtime_error("expected {onset_seconds, ramp_seconds, hold, exempt?}");
            zonos3_steer_release rel;
            rel.onset_seconds = r.at("onset_seconds").get<double>();
            rel.ramp_seconds = r.at("ramp_seconds").get<double>();
            rel.hold = r.at("hold").get<double>();
            if (r.contains("exempt"))
                rel.exempt = r["exempt"].get<std::vector<std::string>>();
            if (!(std::isfinite(rel.hold) && 0.0 <= rel.hold && rel.hold <= 1.0))
                throw std::runtime_error("steer_hold must be in [0, 1]");
            if (!(std::isfinite(rel.onset_seconds) && rel.onset_seconds >= 0.0))
                throw std::runtime_error("steer_onset_seconds must be finite and >= 0");
            if (!(std::isfinite(rel.ramp_seconds) && rel.ramp_seconds >= 0.0))
                throw std::runtime_error("steer_ramp_seconds must be finite and >= 0");
            for (const auto& id : rel.exempt) {
                const auto d = b.index.find(id);
                if (d == b.index.end() || !b.dirs[d->second].hidden) {
                    err = invalid(path,
                                  "steer_release exempts '" + id + "', which is not a hidden direction of the bundle");
                    return false;
                }
            }
            b.steer_release = std::move(rel);
        } catch (const std::exception& e) {
            err = invalid(path, std::string("bad steer_release JSON: ") + e.what());
            return false;
        }
    }

    const auto& hp = model.hp;
    if (b.hidden_size && b.hidden_size != hp.n_embd) {
        err = "steering hidden directions have " + std::to_string(b.hidden_size) +
              " values; the backbone hidden size is " + std::to_string(hp.n_embd);
        return false;
    }
    if (b.speaker_dim && b.speaker_dim != hp.speaker_dim) {
        err = "steering speaker directions have " + std::to_string(b.speaker_dim) +
              " values; the backbone speaker embedding has " + std::to_string(hp.speaker_dim);
        return false;
    }
    for (int s : b.sources) {
        if (s > 2 * (int)hp.n_layer) {
            err = "steering source " + std::to_string(s) + " is outside the backbone's Full AttnRes bank (0.." +
                  std::to_string(2 * hp.n_layer) + ")";
            return false;
        }
    }
    return true;
}

std::vector<std::string> zonos3_steer_disabled_controls(const zonos3_steer_bundle& b, const std::string& speaker) {
    std::vector<std::string> out;
    for (const auto& s : b.sets) {
        if (!s.calibration)
            continue;
        std::vector<std::string> names;
        for (const auto& d : b.dirs)
            if (d.set == s.name)
                names.push_back(d.name);
        for (const auto& n : s.calibration->disabled(speaker, names))
            out.push_back(s.name + "." + n);
    }
    return out;
}

std::string zonos3_steer_controls_json(const zonos3_steer_bundle& b, int embedding_dim) {
    ojson doc;
    doc["bundle"] = b.name;
    doc["embedding_dim"] = embedding_dim;
    ojson sets = ojson::object();
    for (const auto& s : b.sets) {
        ojson e;
        e["space"] = s.hidden ? "attnres_source" : "raw";
        e["modes"] = s.hidden ? ojson::array({"embedding", "reference"}) : ojson::array({"embedding"});
        if (s.hidden)
            e["source_index"] = s.source;
        if (s.calibration)
            e["calibration"] = ojson::parse(s.calibration->document_json);
        sets[s.name] = std::move(e);
    }
    doc["sets"] = std::move(sets);
    ojson dirs = ojson::object();
    for (const auto& d : b.dirs) {
        ojson e;
        e["set"] = d.set;
        e["space"] = d.hidden ? "attnres_source" : "raw";
        e["kind"] = d.axis ? "axis" : "named";
        double n2 = 0;
        for (float v : d.vec)
            n2 += (double)v * (double)v;
        e["norm"] = std::sqrt(n2);
        if (!d.hidden)
            e["vector"] = d.vec;
        dirs[d.id] = std::move(e);
    }
    doc["directions"] = std::move(dirs);
    doc["timing"] = ojson::parse(b.timing_json);
    doc["ui"] = ojson::parse(b.ui_json);
    doc["steer_release"] = ojson::parse(zonos3_steer_release_json(&b));
    return doc.dump();
}

std::string zonos3_steer_release_json(const zonos3_steer_bundle* b) {
    if (!b || !b->steer_release)
        return "null";
    const auto& r = *b->steer_release;
    ojson e;
    e["onset_seconds"] = r.onset_seconds;
    e["ramp_seconds"] = r.ramp_seconds;
    e["hold"] = r.hold;
    e["exempt"] = r.exempt;
    return e.dump();
}

bool zonos3_steer_request_release(const zonos3_steer_bundle* b, double hold, double onset_seconds, double ramp_seconds,
                                  double frame_rate, std::optional<zonos3_steer_release_frames>& out,
                                  std::string& err) {
    out.reset();
    if (!b || !b->steer_release) {
        if (std::isnan(hold) || hold == 1.0)
            return true;
        err = b ? "steer_hold below 1 needs a steering bundle with steer_release; this server's bundle has none."
                : "steer_hold below 1 needs a steering bundle with steer_release.";
        return false;
    }
    const auto& d = *b->steer_release;
    const double h = std::isnan(hold) ? d.hold : hold;
    const double onset = std::isnan(onset_seconds) ? d.onset_seconds : onset_seconds;
    const double ramp = std::isnan(ramp_seconds) ? d.ramp_seconds : ramp_seconds;
    if (!(std::isfinite(h) && 0.0 <= h && h <= 1.0)) {
        err = "steer_hold must be in [0, 1], got " + nlohmann::json(h).dump() + ".";
        return false;
    }
    for (const auto& [name, v] :
         {std::pair<const char*, double>{"steer_onset_seconds", onset}, {"steer_ramp_seconds", ramp}})
        if (!(std::isfinite(v) && v >= 0.0)) {
            err = std::string(name) + " must be finite and >= 0, got " + nlohmann::json(v).dump() + ".";
            return false;
        }
    if (h == 1.0)
        return true;
    // ReleaseSchedule.frames: Python round() (half to even) = nearbyint under the default rounding mode.
    out =
        zonos3_steer_release_frames{(int)std::nearbyint(onset * frame_rate), (int)std::nearbyint(ramp * frame_rate), h};
    return true;
}

double zonos3_steer_duration_factor(const zonos3_steer_bundle& b, const std::map<std::string, double>& doses) {
    double factor = 1.0;
    for (const auto& [id, table] : b.timing) {
        auto it = doses.find(id);
        if (it == doses.end() || it->second == 0.0)
            continue;
        factor *= interp(table, it->second);
    }
    return factor;
}

double zonos3_steering::release_weight(int frame) const {
    const auto& r = release_frames;
    if (frame < r.onset)
        return 1.0;
    if (frame >= r.onset + r.ramp)
        return r.hold;
    return 1.0 + (r.hold - 1.0) * (double)(frame - r.onset) / (double)r.ramp;
}

void zonos3_steering::base_at(int frames, std::vector<float>& out) const {
    if (release.empty()) {
        out = hidden;
        return;
    }
    // TTSReq.steer_delta: the constant delta plus w * release (w == 1: the release delta unscaled), float32.
    if (hidden.empty())
        out.assign(release.size(), 0.0f);
    else
        out.assign(hidden.begin(), hidden.end());
    const double w = release_weight(frames);
    if (w == 1.0) {
        for (size_t i = 0; i < out.size(); i++)
            out[i] += release[i];
    } else {
        const float wf = (float)w;
        for (size_t i = 0; i < out.size(); i++)
            out[i] += wf * release[i];
    }
}

void zonos3_steering::delta_at(int frames, std::vector<float>& out) const {
    if (!hidden_active()) {
        out.clear();
        return;
    }
    const size_t n = sources.size() * (size_t)hidden_size;
    if (!release.empty())
        base_at(frames, out);
    else if (hidden.empty())
        out.assign(n, 0.0f);
    else
        out.assign(hidden.begin(), hidden.end());
    bool any = !hidden.empty() || !release.empty();
    const float f = (float)frames;
    for (const auto& s : spans) {
        const float r = (float)std::max(s.ramp, 1);
        const float w =
            std::min(std::max(std::min((f - ((float)s.start - r)) / r, (((float)s.end + r) - f) / r), 0.0f), 1.0f);
        if (!(w > 0.0f))
            continue;
        any = true;
        for (size_t i = 0; i < n; i++)
            out[i] += w * s.delta[i];
    }
    if (!any)
        out.clear();
}

bool zonos3_steer_resolve(const zonos3_steer_bundle& b, const std::map<std::string, double>& controls,
                          const std::vector<zonos3_control_span>& spans, int text_chars, double frames_per_char,
                          std::vector<float>* speaker_embedding, zonos3_steering& out, std::string& err,
                          const zonos3_steer_release_frames* release) {
    out = zonos3_steering{};
    out.sources = b.sources;
    out.hidden_size = (int)b.hidden_size;

    std::map<std::string, double> doses;
    for (const auto& [id, v] : controls)
        if (v != 0.0)
            doses[id] = v;
    auto unknown = [&](const std::string& id) { err = "Unknown control '" + id + "'; see /v2/controls."; };
    for (const auto& [id, v] : doses)
        if (!b.index.count(id)) {
            unknown(id);
            return false;
        }
    for (const auto& s : spans)
        for (const auto& [id, v] : s.controls)
            if (!b.index.count(id)) {
                unknown(id);
                return false;
            }
    for (const auto& [id, v] : doses)
        if (!std::isfinite(v)) {
            err = "Control '" + id + "' must be finite.";
            return false;
        }

    const size_t K = b.sources.size(), D = b.hidden_size;
    auto slot_of = [&](int source) {
        return (size_t)(std::lower_bound(b.sources.begin(), b.sources.end(), source) - b.sources.begin());
    };
    // Hidden delta [K*D] of a dose set, or empty when it has no hidden dose.
    auto hidden_vector = [&](const std::map<std::string, double>& ds) {
        std::vector<float> v;
        for (const auto& [id, dose] : ds) {
            if (dose == 0.0)
                continue;
            const auto& d = b.dirs[b.index.at(id)];
            if (!d.hidden)
                continue;
            if (v.empty())
                v.assign(K * D, 0.0f);
            float* dst = v.data() + slot_of(d.source) * D;
            const float f = (float)dose;
            for (size_t i = 0; i < D; i++)
                dst[i] += f * d.vec[i];
        }
        return v;
    };

    std::vector<std::string> raw;
    for (const auto& [id, v] : doses)
        if (!b.dirs[b.index.at(id)].hidden)
            raw.push_back(id);
    if (!raw.empty()) {
        if (!speaker_embedding) {
            std::string names;
            for (const auto& id : raw)
                names += (names.empty() ? "" : ", ") + id;
            err = "Controls " + names + " move the speaker embedding and need speaker_mode=embedding.";
            return false;
        }
        std::vector<float>& base = *speaker_embedding;
        std::vector<float> delta(base.size(), 0.0f);
        for (const auto& id : raw) {
            const auto& d = b.dirs[b.index.at(id)];
            if (d.vec.size() != base.size()) {
                err = "Control '" + id + "' has dim " + std::to_string(d.vec.size()) + ", the speaker embedding has " +
                      std::to_string(base.size()) + ".";
                return false;
            }
            const float f = (float)doses.at(id);
            for (size_t i = 0; i < base.size(); i++)
                delta[i] += f * d.vec[i];
        }
        double nb = 0, nm = 0;
        for (size_t i = 0; i < base.size(); i++) {
            delta[i] += base[i]; // moved
            nb += (double)base[i] * base[i];
            nm += (double)delta[i] * delta[i];
        }
        const float scale = (float)(std::sqrt(nb) / std::max(std::sqrt(nm), 1e-8));
        for (size_t i = 0; i < base.size(); i++)
            base[i] = delta[i] * scale;
    }

    if (!release) {
        out.hidden = hidden_vector(doses);
    } else { // resolve_controls(release_exempt=...): the exempt ids stay constant, the others release
        const auto& exempt = b.steer_release ? b.steer_release->exempt : std::vector<std::string>{};
        std::map<std::string, double> kept, released;
        for (const auto& [id, v] : doses)
            (std::find(exempt.begin(), exempt.end(), id) != exempt.end() ? kept : released)[id] = v;
        out.hidden = hidden_vector(kept);
        out.release = hidden_vector(released);
        if (!out.release.empty())
            out.release_frames = *release;
    }

    if (spans.empty())
        return true;
    const double rate =
        (frames_per_char > 0 ? frames_per_char : ZONOS3_STEER_FRAMES_PER_CHAR) * zonos3_steer_duration_factor(b, doses);
    if (!(rate > 0)) {
        err = "control_span_rate must be positive.";
        return false;
    }

    struct clean {
        int start, end;
        std::map<std::string, double> doses;
    };
    std::vector<clean> cleaned;
    for (const auto& s : spans) {
        clean c{s.start, s.end, {}};
        for (const auto& [id, v] : s.controls)
            if (v != 0.0)
                c.doses[id] = v;
        if (!(0 <= s.start && s.start < s.end && s.end <= text_chars)) {
            err = "control span [" + std::to_string(s.start) + ", " + std::to_string(s.end) +
                  ") is outside the text (0.." + std::to_string(text_chars) + ").";
            return false;
        }
        for (const auto& [id, v] : c.doses) {
            if (!std::isfinite(v)) {
                err = "Control '" + id + "' must be finite.";
                return false;
            }
            if (!b.dirs[b.index.at(id)].hidden) {
                err = "Control '" + id +
                      "' moves the speaker embedding and cannot vary inside a request; "
                      "only hidden-space controls take spans.";
                return false;
            }
        }
        cleaned.push_back(std::move(c));
    }
    std::stable_sort(cleaned.begin(), cleaned.end(), [](const clean& l, const clean& r) {
        return l.start != r.start ? l.start < r.start : l.end < r.end;
    });
    for (size_t i = 1; i < cleaned.size(); i++) {
        if (cleaned[i].start < cleaned[i - 1].end) {
            err = "control spans [" + std::to_string(cleaned[i - 1].start) + ", " + std::to_string(cleaned[i - 1].end) +
                  ") and [" + std::to_string(cleaned[i].start) + ", ...) overlap.";
            return false;
        }
    }
    std::vector<double> frame_at(text_chars + 1, 0.0);
    {
        std::vector<double> per_char(text_chars, rate);
        for (const auto& c : cleaned) {
            const double r = rate * zonos3_steer_duration_factor(b, c.doses);
            for (int i = c.start; i < c.end; i++)
                per_char[i] = r;
        }
        for (int i = 0; i < text_chars; i++)
            frame_at[i + 1] = frame_at[i] + per_char[i];
    }
    for (const auto& c : cleaned) {
        zonos3_steering::span sp;
        sp.delta = hidden_vector(c.doses);
        if (sp.delta.empty())
            continue;
        sp.start = (int)std::nearbyint(frame_at[c.start]);
        sp.end = (int)std::nearbyint(frame_at[c.end]);
        sp.ramp = ZONOS3_STEER_RAMP_FRAMES;
        sp.char_start = c.start;
        sp.char_end = c.end;
        out.spans.push_back(std::move(sp));
    }
    return true;
}
