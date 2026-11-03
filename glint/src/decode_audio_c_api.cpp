// glint - whole-file decode C ABI (auto-detects MP3 / AAC-LC / Opus / FLAC).
// Mirrors the CLI decode pipeline (cli/audio_io.hpp) so every language
// wrapper gets decode_file / transcode_file for free. glint_decode_audio_ex
// adds an output rate, an int16 option, and Opus surround (family 1).
// MIT License - Clean-room implementation.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "aac_decoder.hpp"
#include "flac_decoder.hpp"
#include "glint/glint.h"
#include "mp3_decoder.hpp"
#include "opus_decoder.hpp"
#include "opus_ms_decoder.hpp"
#include "opus_ogg.hpp"
#include "resample.hpp"
#include "vorbis_decoder.hpp"
#include "vorbis_ogg.hpp"

namespace {

enum class Fmt { Unknown, Mp3, Aac, Opus, Vorbis, Flac };

// Ogg containers carry both Opus and Vorbis. Peek at the first logical
// stream's first packet: "OpusHead" -> Opus, 0x01 "vorbis" -> Vorbis.
Fmt detect_ogg_codec(const uint8_t* d, size_t n) {
    std::vector<std::vector<uint8_t>> packets;
    int64_t granule = -1;
    if (glint::vorbis::ogg_demux_first_stream(d, n, packets, &granule) != 0 ||
        packets.empty())
        return Fmt::Unknown;
    const auto& p0 = packets[0];
    if (p0.size() >= 8 && !std::memcmp(p0.data(), "OpusHead", 8))
        return Fmt::Opus;
    if (p0.size() >= 7 && p0[0] == 0x01 &&
        !std::memcmp(p0.data() + 1, "vorbis", 6))
        return Fmt::Vorbis;
    return Fmt::Unknown;
}

Fmt detect(const uint8_t* d, size_t n) {
    if (n >= 4 && !std::memcmp(d, "OggS", 4)) return detect_ogg_codec(d, n);
    if (n >= 4 && !std::memcmp(d, "fLaC", 4)) return Fmt::Flac;
    size_t off = 0;
    if (n > 10 && !std::memcmp(d, "ID3", 3))
        off = 10 + ((d[6] & 0x7F) << 21 | (d[7] & 0x7F) << 14 |
                    (d[8] & 0x7F) << 7 | (d[9] & 0x7F));
    if (off + 2 <= n && d[off] == 0xFF && (d[off + 1] & 0xF6) == 0xF0)
        return Fmt::Aac;  // ADTS: 12-bit sync + layer bits 00
    if (off + 2 <= n && d[off] == 0xFF && (d[off + 1] & 0xE0) == 0xE0)
        return Fmt::Mp3;  // MPEG audio syncword 0xFFE (layer bits != 00)
    return Fmt::Unknown;
}

// Every Layer III decoder adds a fixed 529-sample latency (synthesis
// filterbank + IMDCT overlap). A LAME tag records only the ENCODER's delay,
// so the whole head to drop is encoder delay + 529.
constexpr int kMp3DecoderDelay = 529;

// Gapless info from a Xing/Info frame's LAME (or Lavc/Lavf) extension, in
// samples per channel; -1 when the tag carries none.
struct Mp3Gapless {
    int delay = -1;
    int padding = -1;
};

// If the frame at d[0..frame_bytes) is a Xing/Info tag frame, parse its
// gapless fields into *g and return true (the frame is metadata, not audio).
// The tag sits right after the header (+2 CRC bytes when protected) and the
// side info: 32/17 bytes for MPEG-1 stereo/mono, 17/9 for MPEG-2/2.5.
bool mp3_tag_frame(const uint8_t* d, int frame_bytes,
                   const glint::mp3::Mp3FrameInfo& fi, Mp3Gapless* g) {
    const bool crc = (d[1] & 1) == 0;
    const int side = fi.mpeg2 ? (fi.channels == 1 ? 9 : 17)
                              : (fi.channels == 1 ? 17 : 32);
    int p = 4 + (crc ? 2 : 0) + side;
    if (p + 8 > frame_bytes) return false;
    if (std::memcmp(d + p, "Xing", 4) && std::memcmp(d + p, "Info", 4))
        return false;
    p += 4;
    const uint32_t flags = (uint32_t)d[p] << 24 | (uint32_t)d[p + 1] << 16 |
                           (uint32_t)d[p + 2] << 8 | d[p + 3];
    p += 4;
    if (flags & 0x1) p += 4;    // frame count
    if (flags & 0x2) p += 4;    // byte count
    if (flags & 0x4) p += 100;  // seek TOC
    if (flags & 0x8) p += 4;    // VBR quality
    // p is at the 9-byte encoder string; the delays sit 21 bytes in, as 12
    // bits of encoder delay then 12 bits of end padding.
    const int q = p + 21;
    if (q + 3 > frame_bytes) return true;  // a tag frame, but no LAME data
    const int delay = d[q] << 4 | d[q + 1] >> 4;
    const int padding = (d[q + 1] & 0x0F) << 8 | d[q + 2];
    // Both zero says nothing: keep such a stream untrimmed.
    if (delay || padding) {
        g->delay = delay;
        g->padding = padding;
    }
    return true;
}

bool decode_mp3(const uint8_t* d, size_t n, std::vector<float>& out,
                int& sr, int& ch) {
    size_t off = 0;
    if (n > 10 && !std::memcmp(d, "ID3", 3))
        off = 10 + ((d[6] & 0x7F) << 21 | (d[7] & 0x7F) << 14 |
                    (d[8] & 0x7F) << 7 | (d[9] & 0x7F));
    glint::mp3::Mp3Decoder dec;
    dec.init();
    float pcm[2 * 1152];
    glint::mp3::Mp3FrameInfo fi;
    Mp3Gapless gap;
    bool first = true;
    sr = ch = 0;
    while (off + 4 <= n) {
        if (glint::mp3::mp3_frame_info(d + off, (int)(n - off), &fi) < 0) {
            off++;
            continue;
        }
        if (fi.frame_bytes <= 0 || off + (size_t)fi.frame_bytes > n) break;
        // The first frame may be a Xing/Info tag: metadata, decoded by no
        // standard decoder — it would otherwise add a frame of silence.
        if (first) {
            first = false;
            if (mp3_tag_frame(d + off, fi.frame_bytes, fi, &gap)) {
                off += fi.frame_bytes;
                continue;
            }
        }
        int s = dec.decode_frame(d + off, (int)(n - off), pcm, &fi);
        if (s > 0) {
            sr = fi.sample_rate;
            ch = fi.channels;
            out.insert(out.end(), pcm, pcm + (size_t)s * fi.channels);
        }
        off += fi.frame_bytes;
    }
    // Gapless: drop encoder delay + decoder latency from the head and the
    // encoder's end padding (less the latency already shifted out) from the
    // tail, so the PCM lines up sample-for-sample with what was encoded —
    // as ffmpeg and Apple's decoder do. Only when a tag supplied the numbers.
    if (ch > 0 && gap.delay >= 0) {
        const size_t frames = out.size() / (size_t)ch;
        const size_t head = (size_t)(gap.delay + kMp3DecoderDelay);
        const size_t tail = gap.padding > kMp3DecoderDelay
                                ? (size_t)(gap.padding - kMp3DecoderDelay)
                                : 0;
        if (head + tail >= frames) {
            out.clear();  // degenerate or hostile tag
        } else {
            out.erase(out.end() - (std::ptrdiff_t)(tail * ch), out.end());
            out.erase(out.begin(), out.begin() + (std::ptrdiff_t)(head * ch));
        }
    }
    return ch > 0;
}

bool decode_aac(const uint8_t* d, size_t n, std::vector<float>& out,
                int& sr, int& ch) {
    glint::aac::AacDecoder dec;
    dec.init();
    float pcm[2 * 1024];
    glint::aac::AacFrameInfo fi;
    size_t off = 0;
    sr = ch = 0;
    while (off + 7 <= n) {
        if (glint::aac::aac_frame_info(d + off, (int)(n - off), &fi) < 0) {
            off++;
            continue;
        }
        if (fi.frame_bytes <= 0 || off + (size_t)fi.frame_bytes > n) break;
        int s = dec.decode_frame(d + off, (int)(n - off), pcm, &fi);
        if (s > 0) {
            sr = fi.sample_rate;
            ch = fi.channels;
            out.insert(out.end(), pcm, pcm + (size_t)s * fi.channels);
        }
        off += fi.frame_bytes;
    }
    return ch > 0;
}

// Decode Ogg-Opus (mono/stereo via OpusDecoder, surround via the family-1
// multistream decoder) at the native 48 kHz, applying the edit list.
bool decode_opus(const uint8_t* d, size_t n, std::vector<float>& out,
                 int& sr, int& ch) {
    glint::opus::OggOpusReader r;
    if (r.parse(d, n) != 0) return false;
    const auto& h = r.head();
    ch = h.channels;
    if (ch < 1 || ch > 8) return false;
    sr = 48000;
    std::vector<float> pcm(static_cast<size_t>(5760) * ch);

    if (h.mapping_family == 0) {
        glint::opus::OpusDecoder dec;
        dec.init(ch);
        for (int i = 0; i < r.packet_count(); i++) {
            const auto& p = r.packet(i);
            int s = dec.decode(p.data(), (int)p.size(), pcm.data(), 5760);
            if (s > 0)
                out.insert(out.end(), pcm.data(),
                           pcm.data() + (size_t)s * ch);
        }
    } else {
        glint::opus::OpusMsDecoder dec;
        if (dec.init(ch, h.stream_count, h.coupled_count, h.mapping) != 0)
            return false;
        for (int i = 0; i < r.packet_count(); i++) {
            const auto& p = r.packet(i);
            int s = dec.decode(p.data(), (int)p.size(), pcm.data(), 5760);
            if (s > 0)
                out.insert(out.end(), pcm.data(),
                           pcm.data() + (size_t)s * ch);
        }
    }

    int pre = h.pre_skip;
    double gain = r.output_gain();
    if (gain != 1.0)
        for (auto& x : out) x = static_cast<float>(x * gain);
    if (pre > 0 && (long)out.size() >= (long)pre * ch)
        out.erase(out.begin(), out.begin() + (size_t)pre * ch);
    int64_t total = r.total_samples();
    if (total >= 0 && (long)out.size() > total * ch)
        out.resize((size_t)total * ch);
    return true;
}

// Decode Ogg-Vorbis I (whole buffer) at its native sample rate.
bool decode_vorbis(const uint8_t* d, size_t n, std::vector<float>& out,
                   int& sr, int& ch) {
    return glint::vorbis::decode_ogg(d, n, out, sr, ch) == 0 && ch > 0;
}

bool decode_flac(const uint8_t* d, size_t n, std::vector<float>& out,
                 int& sr, int& ch) {
    return glint::flac::decode(d, n, out, sr, ch) == 0 && ch > 0;
}

void* finish_decode(std::vector<float>& pcm, int sr, int ch, int out_rate,
                    int want_int16, int* out_sr, int* out_ch,
                    int* out_frames) {
    if (out_rate > 0 && out_rate != sr && !pcm.empty()) {
        int in_frames = (int)(pcm.size() / (size_t)ch);
        int nf = 0;
        std::vector<float> rs = glint::resample(pcm.data(), in_frames, ch,
                                                sr, out_rate, &nf);
        pcm.swap(rs);
        sr = out_rate;
    }

    int frames = (int)(pcm.size() / (size_t)ch);
    void* buf = nullptr;
    if (want_int16) {
        int16_t* b = static_cast<int16_t*>(
            std::malloc(sizeof(int16_t) * (pcm.empty() ? 1 : pcm.size())));
        if (!b) return nullptr;
        for (size_t i = 0; i < pcm.size(); i++) {
            double s = pcm[i];
            if (s > 1.0) s = 1.0;
            if (s < -1.0) s = -1.0;
            int v = (int)(s * 32767.0 + (s >= 0 ? 0.5 : -0.5));
            if (v > 32767) v = 32767;
            if (v < -32768) v = -32768;
            b[i] = (int16_t)v;
        }
        buf = b;
    } else {
        float* b = static_cast<float*>(
            std::malloc(sizeof(float) * (pcm.empty() ? 1 : pcm.size())));
        if (!b) return nullptr;
        if (!pcm.empty())
            std::memcpy(b, pcm.data(), sizeof(float) * pcm.size());
        buf = b;
    }
    if (out_sr) *out_sr = sr;
    if (out_ch) *out_ch = ch;
    if (out_frames) *out_frames = frames;
    return buf;
}

// Decode any supported stream to interleaved float at the native rate.
bool decode_native(const uint8_t* data, size_t len, std::vector<float>& out,
                   int& sr, int& ch) {
    switch (detect(data, len)) {
    case Fmt::Mp3: return decode_mp3(data, len, out, sr, ch);
    case Fmt::Aac: return decode_aac(data, len, out, sr, ch);
    case Fmt::Opus: return decode_opus(data, len, out, sr, ch);
    case Fmt::Vorbis: return decode_vorbis(data, len, out, sr, ch);
    case Fmt::Flac: return decode_flac(data, len, out, sr, ch);
    default: return false;
    }
}

}  // namespace

extern "C" {

void* glint_decode_audio_ex(const uint8_t* data, int len, int out_rate,
                            int want_int16, int* out_sr, int* out_ch,
                            int* out_frames) {
    if (out_sr) *out_sr = 0;
    if (out_ch) *out_ch = 0;
    if (out_frames) *out_frames = 0;
    if (!data || len <= 0) return nullptr;

    std::vector<float> pcm;
    int sr = 0, ch = 0;
    if (!decode_native(data, (size_t)len, pcm, sr, ch) || ch <= 0)
        return nullptr;

    return finish_decode(pcm, sr, ch, out_rate, want_int16, out_sr, out_ch,
                         out_frames);
}

float* glint_decode_audio(const uint8_t* data, int len, int* out_sr,
                          int* out_ch, int* out_frames) {
    // Simple default: float, native rate, whatever channel count decodes.
    return static_cast<float*>(glint_decode_audio_ex(
        data, len, 0, 0, out_sr, out_ch, out_frames));
}

void* glint_flac_decode_ex(const uint8_t* data, int len, int out_rate,
                           int want_int16, int* out_sr, int* out_ch,
                           int* out_frames) {
    if (out_sr) *out_sr = 0;
    if (out_ch) *out_ch = 0;
    if (out_frames) *out_frames = 0;
    if (!data || len <= 0) return nullptr;
    std::vector<float> pcm;
    int sr = 0, ch = 0;
    if (!decode_flac(data, (size_t)len, pcm, sr, ch) || ch <= 0)
        return nullptr;
    return finish_decode(pcm, sr, ch, out_rate, want_int16, out_sr, out_ch,
                         out_frames);
}

float* glint_flac_decode(const uint8_t* data, int len, int* out_sr,
                         int* out_ch, int* out_frames) {
    return static_cast<float*>(glint_flac_decode_ex(
        data, len, 0, 0, out_sr, out_ch, out_frames));
}

}  // extern "C"
