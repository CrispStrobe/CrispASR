// zonos3-tokenizer.h — SuperBPE text tokenizer (HF `tokenizers` BPE JSON, byte-level, no normalizer).
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#define ZONOS3_START_MARKER "\xEE\x80\x82" // U+E002 (superbpe.START_MARKER); pause markers are U+E003 + bucket

// Pronunciation groups of a phoneme-token checkpoint (text_tokenizer_spec.phonemes; zonos3/tts/superbpe.py): a group
// `open_marker` IPA `close_marker` encodes as open_id, one id per IPA code point (open_id + 2 + index in `symbols`),
// close_id.
struct zonos3_phoneme_spec {
    std::string open_marker = "\xEE\x80\x80", close_marker = "\xEE\x80\x81"; // U+E000 / U+E001
    int32_t open_id = -1, close_id = -1;
    std::vector<std::string> symbols; // single code points (UTF-8)
};

struct zonos3_tokenizer {
    std::unordered_map<std::string, int32_t> vocab;      // byte-level symbol string -> tokenizer id
    std::unordered_map<std::string, int32_t> merge_rank; // key = a + '\x01' + b
    std::vector<std::pair<std::string, int32_t>> added;  // literal special tokens
    int32_t offset = 4, bos = 2, eos = 3;
    std::string spec_json; // zonos3.text_tokenizer_spec as stored in the GGUF
    bool has_phonemes = false;
    zonos3_phoneme_spec phonemes;      // valid when has_phonemes
    int32_t start_id = -1;             // text_tokenizer_spec.start.id (-1: no start token)
    std::vector<int32_t> pause_ids;    // text_tokenizer_spec.pauses.ids, U+E003 + j -> pause_ids[j] (empty: none)
    std::vector<double> pause_seconds; // text_tokenizer_spec.pauses.seconds
    // The phoneme / start / pause rows end the extra text rows, after the edit marker rows: block_first is their
    // first id (phonemes.open_id, else start.id, else pauses.ids[0]; -1 without any), block_rows their count.
    int32_t block_first = -1, block_rows = 0;
};

// Parse a `tokenizers` JSON (model.type BPE, merges as [a,b] arrays). Model ids are
// tokenizer id + offset; bos/eos are model ids. Returns false and sets err on any other shape.
bool zonos3_tokenizer_load(zonos3_tokenizer& tok, const std::string& json, int32_t offset, int32_t bos, int32_t eos,
                           std::string& err);
// Parse and check zonos3.text_tokenizer_spec (superbpe.validate_spec and load_spec's extra-row check) into `tok`.
bool zonos3_tokenizer_set_spec(zonos3_tokenizer& tok, const std::string& spec_json, uint32_t text_vocab,
                               uint32_t text_extra_tokens, std::string& err);

// Model text ids for `utf8`, including the leading BOS and trailing EOS (plain SuperBPE, no pronunciation groups).
std::vector<int32_t> zonos3_tokenize(const zonos3_tokenizer& tok, const std::string& utf8);
// superbpe.payload_ids (no BOS/EOS): the start marker (first character, once) as start_id, pause markers as their
// pause ids, SuperBPE between them, with pronunciation groups as phoneme ids on a phoneme-token checkpoint. False with
// Python's ValueError text in `err` (NO_START_TOKEN, START_NOT_FIRST, NO_PAUSE_TOKENS, "Pause marker U+XXXX outside
// the N pause buckets", "Unbalanced phoneme group markers", "Unbalanced or empty phoneme group", "Phones outside the
// inventory: [...]").
bool zonos3_payload_ids(const zonos3_tokenizer& tok, const std::string& utf8, std::vector<int32_t>& out,
                        std::string& err);
// superbpe.text_ids: [BOS] payload_ids [EOS].
bool zonos3_text_ids(const zonos3_tokenizer& tok, const std::string& utf8, std::vector<int32_t>& out, std::string& err);
// R13 (zonos3.tts.plan.validate_text_markers): "" when `utf8`'s pronunciation groups, start marker and pause markers
// are valid for the checkpoint, else the 400 message, checked in plan's order. `edit`: a speech-edit target text,
// which takes no start or pause markers.
std::string zonos3_check_text_markers(const zonos3_tokenizer& tok, const std::string& utf8, bool edit = false);
