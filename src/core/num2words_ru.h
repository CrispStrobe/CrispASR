// core/num2words_ru.h — spell Russian numbers out before phonemization.
//
// Same defect this file's en/de/fr/es siblings exist to fix, one language
// further on: a digit is in no pronunciation dictionary and matches no
// letter-to-sound rule, so a numeric token phonemizes to the EMPTY string and
// disappears from the audio. Worse for Zonos specifically — its phoneme
// inventory has no digits either, so an unexpanded "2024" is dropped a SECOND
// time, silently, by text_to_phoneme_ids().
//
// WHAT THIS DOES NOT DO, stated up front because Russian numerals decline and
// pretending otherwise would be the same class of quiet wrongness:
//
//   * NOMINATIVE ONLY. Russian numerals agree with their noun in case, and a
//     number in running text is very often oblique ("в 2024 году" is read
//     "в две тысячи двадцать четвёртом году", not "две тысячи двадцать
//     четыре"). Nothing here sees the noun, so nothing here can inflect.
//   * GENDER is applied only where the numeral word itself forces it — the
//     feminine "одна/две тысячи" — not from the counted noun, which is not
//     visible. "1 книга" comes out "один книга".
//   * No ordinals, no currency nouns, no fractions beyond a decimal read as
//     "целых / and then digits".
//
// The alternative to an imperfect expansion is not a perfect one; it is the
// number vanishing. That is why this is here, and why the limits are in the
// header rather than in a commit message.
//
// Weight-free and header-only — tests/test-g2p-ru.cpp covers it without a
// model, because a wrong entry produces perfectly well-formed phonemes for the
// wrong word and no numeric check downstream can see it.
#pragma once

#include <cctype>
#include <cstdint>
#include <string>

namespace core_num2words_ru {

namespace detail {

inline bool is_digit(char c) {
    return c >= '0' && c <= '9';
}

// 0..19, masculine nominative.
inline const char* const kUnder20[20] = {"ноль",       "один",        "два",        "три",          "четыре",
                                         "пять",       "шесть",       "семь",       "восемь",       "девять",
                                         "десять",     "одиннадцать", "двенадцать", "тринадцать",   "четырнадцать",
                                         "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"};

// Tens. 40 and 90 are the two that no pattern produces: сорок and девяносто
// are not built from четыре/девять at all.
inline const char* const kTens[10] = {"",          "",           "двадцать",  "тридцать",    "сорок",
                                      "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто"};

// Hundreds. 200 is двести (not "двасто"), and 300/400 take -ста while
// 500..900 take -сот — three different endings in one row.
inline const char* const kHundreds[10] = {"",        "сто",      "двести",  "триста",    "четыреста",
                                          "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот"};

// Feminine forms of 1 and 2, needed for тысяча and nothing else here.
inline const char* const kOneFem = "одна";
inline const char* const kTwoFem = "две";

// Which of the three forms a counted noun takes after `n`.
//   0 = nominative singular   (1, 21, 31 … but NOT 11)
//   1 = genitive singular     (2-4, 22-24 … but NOT 12-14)
//   2 = genitive plural       (everything else, and ALL of 11-14)
// The 11-14 exception is the whole reason this is a function: n % 10 alone
// gets "одиннадцать тысяча" and "двенадцать тысячи".
inline int plural_form(int64_t n) {
    const int64_t h = n % 100;
    if (h >= 11 && h <= 14)
        return 2;
    switch (n % 10) {
    case 1:
        return 0;
    case 2:
    case 3:
    case 4:
        return 1;
    default:
        return 2;
    }
}

// 1..999 in words. `feminine` switches один/два to одна/две, which matters
// only for the тысяча group.
inline std::string under_thousand(int64_t n, bool feminine) {
    std::string out;
    const int64_t h = n / 100;
    const int64_t r = n % 100;
    if (h > 0)
        out = kHundreds[h];
    if (r > 0) {
        if (!out.empty())
            out += ' ';
        if (r < 20) {
            if (feminine && r == 1)
                out += kOneFem;
            else if (feminine && r == 2)
                out += kTwoFem;
            else
                out += kUnder20[r];
        } else {
            out += kTens[r / 10];
            const int64_t u = r % 10;
            if (u > 0) {
                out += ' ';
                if (feminine && u == 1)
                    out += kOneFem;
                else if (feminine && u == 2)
                    out += kTwoFem;
                else
                    out += kUnder20[u];
            }
        }
    }
    return out;
}

struct scale {
    const char* forms[3]; // nom.sg, gen.sg, gen.pl
    bool feminine;
};

// тысяча is FEMININE and the millions are not — "две тысячи" but "два
// миллиона". A single gender assumption gets one of them wrong.
inline const scale kScales[3] = {
    {{"тысяча", "тысячи", "тысяч"}, true},
    {{"миллион", "миллиона", "миллионов"}, false},
    {{"миллиард", "миллиарда", "миллиардов"}, false},
};

// Read a digit run out one digit at a time — for things no cardinal reading
// fits, like a phone number or a run longer than this file handles.
inline std::string digits_each(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (!is_digit(c))
            continue;
        if (!out.empty())
            out += ' ';
        out += kUnder20[c - '0'];
    }
    return out;
}

} // namespace detail

// Cardinal, nominative. Handles |n| < 10^12; anything larger is read digit by
// digit rather than silently truncated.
inline std::string cardinal(int64_t n) {
    if (n == 0)
        return detail::kUnder20[0];
    std::string out;
    if (n < 0) {
        out = "минус ";
        n = -n;
    }
    if (n >= 1000000000000LL)
        return out + detail::digits_each(std::to_string(n));

    std::string words;
    // Groups of three from the top: billions, millions, thousands, units.
    const int64_t groups[4] = {n / 1000000000LL % 1000, n / 1000000LL % 1000, n / 1000LL % 1000, n % 1000};
    const int scale_idx[4] = {2, 1, 0, -1};
    for (int g = 0; g < 4; g++) {
        const int64_t v = groups[g];
        if (v == 0)
            continue;
        const int si = scale_idx[g];
        const bool fem = si >= 0 && detail::kScales[si].feminine;
        if (!words.empty())
            words += ' ';
        words += detail::under_thousand(v, fem);
        if (si >= 0) {
            words += ' ';
            words += detail::kScales[si].forms[detail::plural_form(v)];
        }
    }
    return out + words;
}

// Replace every digit run in `text` with its spelled-out form. A ',' between
// digits is the decimal mark in Russian; a '.' between digits is left as a
// separator and the parts read separately, because "2.5" is far more often a
// version or a date fragment than a decimal in Russian text.
inline std::string expand(const std::string& text) {
    std::string out;
    const size_t n = text.size();
    size_t i = 0;
    while (i < n) {
        if (!detail::is_digit(text[i])) {
            out += text[i++];
            continue;
        }
        size_t start = i;
        while (i < n && detail::is_digit(text[i]))
            i++;
        std::string int_part = text.substr(start, i - start);

        // Decimal mark: ',' followed immediately by a digit.
        std::string frac_part;
        if (i + 1 < n && text[i] == ',' && detail::is_digit(text[i + 1])) {
            size_t fs = ++i;
            while (i < n && detail::is_digit(text[i]))
                i++;
            frac_part = text.substr(fs, i - fs);
        }

        bool negative = false;
        if (!out.empty() && out.back() == '-') {
            // Only a MINUS, not a hyphen inside a word: "по-2" keeps its hyphen.
            const size_t j = out.size() - 1;
            if (j == 0 || out[j - 1] == ' ') {
                negative = true;
                out.erase(j);
            }
        }

        // A run too long to be a quantity (a phone number, an id) reads better
        // digit by digit than as a trillion.
        std::string words;
        if (int_part.size() > 12) {
            words = detail::digits_each(int_part);
        } else {
            int64_t value = 0;
            for (char c : int_part)
                value = value * 10 + (c - '0');
            words = cardinal(negative ? -value : value);
        }
        if (!frac_part.empty()) {
            words += " запятая ";
            words += detail::digits_each(frac_part);
        }

        // `%` is not in any TTS phoneme inventory, so an unexpanded one is
        // dropped silently — the same defect as the digits, one character
        // later. The counted-noun form is already available, and percent takes
        // it like any other noun: 1 процент, 2 процента, 5 процентов.
        {
            size_t j = i;
            if (j < n && text[j] == ' ')
                j++;
            if (j < n && text[j] == '%' && frac_part.empty() && int_part.size() <= 12) {
                int64_t value = 0;
                for (char c : int_part)
                    value = value * 10 + (c - '0');
                static const char* const kPercent[3] = {"процент", "процента", "процентов"};
                words += " ";
                words += kPercent[detail::plural_form(value)];
                i = j + 1;
            }
        }

        if (!out.empty() && out.back() != ' ' && out.back() != '\n')
            out += ' ';
        out += words;
        // Keep a following space if the source had none, so the next word does
        // not fuse onto the numeral.
        if (i < n && text[i] != ' ' && !ispunct((unsigned char)text[i]))
            out += ' ';
    }
    return out;
}

} // namespace core_num2words_ru
