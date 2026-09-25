// g2p_alnum.h — split letters from the digits that follow them.
//
// "d4", "B52", "Win10": every num2words_* expander spells a number only at the
// start of a token (so that "mp3" or "x64" are left alone), and no lexicon or
// letter-to-sound rule reads digits — so the digits of such a token vanished
// and "d4" phonemized to `d`. Chess moves came out as "knight f" and squares
// as silence. misaki (Kokoro's reference G2P) splits these tokens: the letters
// as a word, or by name when there is a single one, and the digits as a
// number. Doing the split before number expansion gives every language's G2P
// the same reading. (English "a8" then reads as article + eight, `ɐ ˈAt`,
// exactly as misaki does: telling that letter from the article needs context
// neither has.)
#pragma once

#include <cctype>
#include <string>

namespace core_g2p_alnum {

inline bool is_letter_byte(unsigned char c) {
    return std::isalpha(c) || c >= 0x80; // a UTF-8 byte belongs to a word
}

inline std::string split_letters_digits(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 8);
    size_t run = std::string::npos; // where the current letter run starts in `out`
    for (size_t i = 0; i < text.size(); ++i) {
        const unsigned char c = (unsigned char)text[i];
        if (std::isdigit(c) && run != std::string::npos)
            out += ' ';
        out += (char)c;
        if (is_letter_byte(c)) {
            if (run == std::string::npos)
                run = out.size() - 1;
        } else {
            run = std::string::npos;
        }
    }
    return out;
}

} // namespace core_g2p_alnum
