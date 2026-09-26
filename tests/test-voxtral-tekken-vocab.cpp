// tests/test-voxtral-tekken-vocab.cpp — #338.
//
// The bug was input-dependent: only texts whose BPE merge path reached an
// inactive tail entry produced an out-of-range token id, so every smoke test
// passed. What makes it testable is that the rule is arithmetic — a token id
// the encoder can emit must be a legal row index into the embedding table —
// and that rule holds for any blob, no model required.

#include "voxtral_tekken_vocab.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// Pack pieces the way the GGUF blob stores them: [u16 len][len bytes] …
std::vector<uint8_t> pack(const std::vector<std::string>& pieces) {
    std::vector<uint8_t> blob;
    for (const auto& s : pieces) {
        const uint16_t len = (uint16_t)s.size();
        blob.push_back((uint8_t)(len & 0xFF));
        blob.push_back((uint8_t)(len >> 8));
        blob.insert(blob.end(), s.begin(), s.end());
    }
    return blob;
}

std::vector<std::string> synth_pieces(int n) {
    std::vector<std::string> v;
    v.reserve(n);
    for (int i = 0; i < n; i++)
        v.push_back("p" + std::to_string(i));
    return v;
}

int max_id(const std::map<std::string, int>& m) {
    int best = -1;
    for (const auto& kv : m)
        best = kv.second > best ? kv.second : best;
    return best;
}

} // namespace

TEST_CASE("active BPE count is the ids left after the specials", "[unit][voxtral]") {
    using namespace voxtral_tekken;
    // Voxtral-4B-TTS-2603: 131072-wide embedding, 1000 specials.
    REQUIRE(active_bpe_count(131072, 1000) == 130072);
    // Degenerate headers yield 0, never a negative count that would later be
    // compared against as if it meant "unlimited".
    REQUIRE(active_bpe_count(0, 1000) == 0);
    REQUIRE(active_bpe_count(-5, 1000) == 0);
    REQUIRE(active_bpe_count(1000, 1000) == 0);
    REQUIRE(active_bpe_count(131072, -1) == 0);
}

TEST_CASE("range predicate matches the embedding table", "[unit][voxtral]") {
    using namespace voxtral_tekken;
    REQUIRE(token_id_in_range(0, 131072));
    REQUIRE(token_id_in_range(131071, 131072));
    REQUIRE_FALSE(token_id_in_range(131072, 131072)); // one past the last row
    REQUIRE_FALSE(token_id_in_range(-1, 131072));
}

// The regression itself. A blob longer than the embedding table is not
// malformed — Mistral ships them that way — so the decoder must keep the tail
// out of the encoder's map rather than reject the blob.
TEST_CASE("a blob longer than the embedding table cannot emit an unusable id", "[unit][voxtral]") {
    using namespace voxtral_tekken;
    constexpr int kVocab = 1100; // stand-in for 131072
    constexpr int kSpecials = 100;
    const int limit = active_bpe_count(kVocab, kSpecials); // 1000

    // 1200 serialized pieces for 1000 usable slots: 200 inert tail entries.
    const auto pieces = synth_pieces(1200);
    const auto blob = pack(pieces);
    const std::vector<std::string> specials(kSpecials, "");

    std::vector<std::string> id_to_piece;
    std::map<std::string, int> piece_to_id;
    const auto st = decode_blob(blob, kSpecials, specials, limit, id_to_piece, piece_to_id);

    REQUIRE(st.n_active == 1000);
    REQUIRE(st.n_inactive == 200);

    // The property that matters: nothing the encoder can look up indexes past
    // the embedding table. Before the fix the map held ids up to 1299.
    REQUIRE(max_id(piece_to_id) == kVocab - 1);
    for (const auto& kv : piece_to_id)
        REQUIRE(token_id_in_range(kv.second, kVocab));

    // The tail pieces specifically must be absent — that is the merge path the
    // reporter's Italian text happened to reach.
    REQUIRE(piece_to_id.count("p999") == 1);  // last active
    REQUIRE(piece_to_id.count("p1000") == 0); // first inactive
    REQUIRE(piece_to_id.count("p1199") == 0); // last serialized

    // The whole blob is still parsed, so a debug dump can show the inert tail.
    REQUIRE((int)id_to_piece.size() == kSpecials + 1200);
    REQUIRE(id_to_piece[kSpecials + 1000] == "p1000");
}

// The same blob decoded the way the runtime used to do it — no limit at all.
// This is what shipped, and it is the arm that must produce the bad ids;
// without it the test above could pass against a decoder that simply never
// emits anything, and nobody would notice.
TEST_CASE("the unbounded decode is what produced the out-of-range ids", "[unit][voxtral]") {
    using namespace voxtral_tekken;
    constexpr int kVocab = 1100;
    constexpr int kSpecials = 100;

    const auto blob = pack(synth_pieces(1200));
    const std::vector<std::string> specials(kSpecials, "");

    std::vector<std::string> id_to_piece;
    std::map<std::string, int> piece_to_id;
    const auto st = decode_blob(blob, kSpecials, specials, /*active_limit*/ 0, id_to_piece, piece_to_id);

    REQUIRE(st.n_inactive == 0);          // nothing held back
    REQUIRE(max_id(piece_to_id) == 1299); // 200 ids past the embedding table
    REQUIRE_FALSE(token_id_in_range(max_id(piece_to_id), kVocab));
    REQUIRE(piece_to_id.count("p1000") == 1); // the tail piece is reachable
}

TEST_CASE("a blob that fits is untouched", "[unit][voxtral]") {
    using namespace voxtral_tekken;
    constexpr int kVocab = 1100;
    constexpr int kSpecials = 100;

    const auto pieces = synth_pieces(500);
    const auto blob = pack(pieces);
    const std::vector<std::string> specials(kSpecials, "");

    std::vector<std::string> id_to_piece;
    std::map<std::string, int> piece_to_id;
    const auto st =
        decode_blob(blob, kSpecials, specials, active_bpe_count(kVocab, kSpecials), id_to_piece, piece_to_id);

    REQUIRE(st.n_active == 500);
    REQUIRE(st.n_inactive == 0);
    // Ids stay contiguous from n_specials — the bound must not perturb the
    // common case, or every existing checkpoint would retokenize differently.
    REQUIRE(piece_to_id.at("p0") == kSpecials);
    REQUIRE(piece_to_id.at("p499") == kSpecials + 499);
}

TEST_CASE("truncated and empty blobs decode without running off the end", "[unit][voxtral]") {
    using namespace voxtral_tekken;
    std::vector<std::string> id_to_piece;
    std::map<std::string, int> piece_to_id;

    // Length header promising more bytes than remain.
    std::vector<uint8_t> truncated = {0x05, 0x00, 'a', 'b'};
    auto st = decode_blob(truncated, 0, {}, 100, id_to_piece, piece_to_id);
    REQUIRE(st.n_active == 0);
    REQUIRE(piece_to_id.empty());

    // A trailing odd byte cannot form a length header.
    std::vector<uint8_t> odd = {0x01, 0x00, 'a', 0x02};
    st = decode_blob(odd, 0, {}, 100, id_to_piece, piece_to_id);
    REQUIRE(st.n_active == 1);
    REQUIRE(piece_to_id.at("a") == 0);

    st = decode_blob({}, 0, {}, 100, id_to_piece, piece_to_id);
    REQUIRE(st.n_active == 0);
    REQUIRE(id_to_piece.empty());
}

// ---------------------------------------------------------------------------
// #472 — the Voxtral Mini 3B runtime (src/voxtral.cpp). It does not use
// decode_blob(): it keeps a rank table (id = rank + n_specials) and a
// tiktoken-style lowest-rank merge, and #338 never reached it. These cases run
// that exact path through the header functions voxtral.cpp now calls.
// ---------------------------------------------------------------------------

namespace {

// All 256 single bytes, then `extra` merges. Rank r <-> id r + n_specials.
std::vector<std::string> byte_base_plus(const std::vector<std::string>& extra) {
    std::vector<std::string> v;
    for (int b = 0; b < 256; b++)
        v.push_back(std::string(1, (char)b));
    v.insert(v.end(), extra.begin(), extra.end());
    return v;
}

std::vector<int32_t> encode_3b(const std::vector<uint8_t>& blob, int max_ranks, int active_limit, int n_specials,
                               const std::string& text, int* n_inactive = nullptr) {
    using namespace voxtral_tekken;
    std::vector<uint32_t> off, len;
    index_blob_ranks(blob, max_ranks, off, len);
    std::unordered_map<std::string, int32_t> b2r;
    const int ni = build_rank_map(blob, off, len, active_limit, b2r);
    if (n_inactive)
        *n_inactive = ni;
    std::vector<int32_t> ids;
    for (const auto& pt : pre_tokenize(text))
        bpe_encode_ranked(b2r, n_specials, (const uint8_t*)pt.data(), pt.size(), ids);
    return ids;
}

} // namespace

// Shaped like the issue: ` epigastric` -> ` ep` + `ig` + `astric`, where
// `astric` is a tail rank (146371 in the real 3B GGUF). Here the table has
// 256 + 3 active ranks and `astric` sits in the inert tail.
TEST_CASE("3B rank merge cannot reach a tail rank (#472)", "[unit][voxtral]") {
    using namespace voxtral_tekken;
    constexpr int kSpecials = 10;
    // Active: bytes, " e", " ep", "ig". Tail: "as", "ast", "astr", "astri", "astric".
    const auto pieces = byte_base_plus({" e", " ep", "ig", "as", "ast", "astr", "astri", "astric"});
    const auto blob = pack(pieces);
    const int kActive = 256 + 3;
    const int kVocab = kSpecials + kActive; // embedding rows
    const int kSerialized = (int)pieces.size();
    REQUIRE(active_bpe_count(kVocab, kSpecials) == kActive);

    // Positive control: the pre-#472 behaviour (every serialized rank in the
    // merge map) DOES produce the out-of-range id, so the bounded arm below is
    // measuring something.
    const auto bad = encode_3b(blob, kSerialized, /*active_limit*/ 0, kSpecials, " epigastric");
    bool any_oob = false;
    for (int32_t id : bad)
        any_oob |= !token_id_in_range(id, kVocab);
    REQUIRE(any_oob);
    REQUIRE(bad.back() == kSpecials + 256 + 7); // `astric`

    int n_inactive = -1;
    const auto ids =
        encode_3b(blob, kSerialized, active_bpe_count(kVocab, kSpecials), kSpecials, " epigastric", &n_inactive);
    REQUIRE(n_inactive == 5);
    for (int32_t id : ids)
        REQUIRE(token_id_in_range(id, kVocab));
    // Active merges are unaffected: ` ep` and `ig` still merge; `astric` falls
    // back to bytes.
    REQUIRE(ids.size() == 2 + 6);
    REQUIRE(ids[0] == kSpecials + 256 + 1); // " ep"
    REQUIRE(ids[1] == kSpecials + 256 + 2); // "ig"
    REQUIRE(ids[2] == kSpecials + (int)'a');
}

// The 3B checkpoint's real shape: 150000 serialized ranks, 131072 rows, 1000
// specials. The id -> text table stays complete; only the merge map is capped.
TEST_CASE("3B rank tables keep the full vocab for id->text, bound only the merge map", "[unit][voxtral]") {
    using namespace voxtral_tekken;
    const auto blob = pack(synth_pieces(150000));
    std::vector<uint32_t> off, len;
    index_blob_ranks(blob, 150000, off, len);
    REQUIRE(off.size() == 150000);

    std::unordered_map<std::string, int32_t> b2r;
    const int n_inactive = build_rank_map(blob, off, len, active_bpe_count(131072, 1000), b2r);
    REQUIRE(n_inactive == 150000 - 130072);
    REQUIRE(b2r.size() == 130072);
    REQUIRE(b2r.count("p130071") == 1);
    REQUIRE(b2r.count("p130072") == 0);
    int32_t max_rank = -1;
    for (const auto& kv : b2r)
        max_rank = kv.second > max_rank ? kv.second : max_rank;
    REQUIRE(token_id_in_range(max_rank + 1000, 131072));
    REQUIRE_FALSE(token_id_in_range(max_rank + 1 + 1000, 131072));

    // index_blob_ranks honours the serialized count and a truncated tail.
    index_blob_ranks(blob, 10, off, len);
    REQUIRE(off.size() == 10);
    std::vector<uint8_t> trunc = {0x01, 0x00, 'a', 0x05, 0x00, 'b'};
    index_blob_ranks(trunc, 100, off, len);
    REQUIRE(off.size() == 1);
}

// The 3B hotword suffix goes through the header pre-tokenizer now: a word keeps
// its leading space (mistral-common), which the old voxtral.cpp copy split off.
TEST_CASE("3B hotword suffix pre-tokenizes like mistral-common", "[unit][voxtral]") {
    const auto pt = voxtral_tekken::pre_tokenize("lang:en The following words may appear: epigastric,syncopal.");
    const std::vector<std::string> want = {"lang",    ":en", " The",        " following", " words", " may",
                                           " appear", ":",   " epigastric", ",syncopal",  "."};
    REQUIRE(pt == want);
}
