// spm_piece.h — SentencePiece vocabulary piece → visible text.
//
//   "▁foo"   → " foo"   (U+2581 word-start marker becomes a space)
//   "<0xE8>" → "\xE8"   (byte-fallback piece becomes its raw byte)
//   "<unk>"  → ""       (other control/special pieces are dropped)
//   anything else → as is
//
// Byte fallback (#605): a model trained with SentencePiece byte_fallback spells
// a character outside its vocabulary as one <0xHH> piece per UTF-8 byte, and
// SentencePiece's decoder reassembles them. Haas-ja does this for rare kanji
// (董 = <0xE8><0x91><0xA3>); dropping them as if they were <unk> lost 骨董品's
// 董, 娯, 徘徊, 癌, ... A token's text may therefore be a partial UTF-8
// sequence; concatenating the tokens in order restores the character.
#pragma once

#include <string>

namespace core_spm {

// The byte value of a "<0xHH>" piece (exactly two hex digits, either case),
// or -1 when `piece` is not one.
inline int byte_piece_value(const std::string& piece) {
    if (piece.size() != 6 || piece[0] != '<' || piece[1] != '0' || piece[2] != 'x' || piece[5] != '>')
        return -1;
    auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9')
            return c - '0';
        if (c >= 'A' && c <= 'F')
            return c - 'A' + 10;
        if (c >= 'a' && c <= 'f')
            return c - 'a' + 10;
        return -1;
    };
    const int hi = hex(piece[3]), lo = hex(piece[4]);
    return hi < 0 || lo < 0 ? -1 : hi * 16 + lo;
}

inline std::string piece_to_text(const std::string& piece) {
    if (piece.empty())
        return "";
    if (piece.size() >= 2 && piece[0] == '<' && piece.back() == '>') {
        const int b = byte_piece_value(piece);
        return b < 0 ? std::string() : std::string(1, (char)b);
    }
    if (piece.size() >= 3 && (unsigned char)piece[0] == 0xE2 && (unsigned char)piece[1] == 0x96 &&
        (unsigned char)piece[2] == 0x81)
        return std::string(" ") + piece.substr(3);
    return piece;
}

} // namespace core_spm
