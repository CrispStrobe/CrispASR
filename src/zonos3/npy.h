// Minimal .npy reader/writer: float32 C-order writer, numpy.lib.format (v1-3) numeric reader.
#pragma once

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace npy {

inline bool save_f32(const std::string& path, const float* data, const std::vector<int64_t>& shape) {
    std::string shp;
    for (size_t i = 0; i < shape.size(); ++i) {
        shp += std::to_string(shape[i]);
        shp += ",";
        if (i + 1 < shape.size())
            shp += " ";
    }
    std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + shp + "), }";
    const size_t base = 10 + dict.size() + 1; // magic(6)+ver(2)+hlen(2) + trailing '\n'
    const size_t pad = (64 - (base % 64)) % 64;
    dict.append(pad, ' ');
    dict.push_back('\n');
    const uint16_t hlen = (uint16_t)dict.size();

    FILE* f = fopen(path.c_str(), "wb");
    if (!f)
        return false;
    const unsigned char magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    fwrite(magic, 1, 8, f);
    fwrite(&hlen, sizeof(uint16_t), 1, f); // host is little-endian (x86)
    fwrite(dict.data(), 1, dict.size(), f);
    size_t n = 1;
    for (int64_t d : shape)
        n *= (size_t)d;
    fwrite(data, sizeof(float), n, f);
    fclose(f);
    return true;
}

// One numeric array, C order, every element as a double; descr is the numpy dtype string (e.g. "<f4").
struct Array {
    std::vector<int64_t> shape;
    std::vector<double> data;
    std::string descr;
};

inline float half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    int exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF, bits;
    if (exp == 0) {
        if (mant == 0)
            bits = sign;
        else {
            exp = 1;
            while (!(mant & 0x400)) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3FF;
            bits = sign | (uint32_t)(exp + 112) << 23 | mant << 13;
        }
    } else if (exp == 31)
        bits = sign | 0x7F800000u | mant << 13;
    else
        bits = sign | (uint32_t)(exp + 112) << 23 | mant << 13;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

// One .npy payload (numpy.lib.format, versions 1-3) as doubles in C order (a Fortran-order array is reordered, like
// numpy.load); false (with `err`) when it is not a plain little-endian numeric array.
inline bool parse(const uint8_t* p, size_t n, Array& out, std::string& err) {
    auto le16 = [](const uint8_t* q) { return (uint16_t)(q[0] | q[1] << 8); };
    auto le32 = [](const uint8_t* q) {
        return q[0] | (uint32_t)q[1] << 8 | (uint32_t)q[2] << 16 | (uint32_t)q[3] << 24;
    };
    out = Array{};
    if (n < 10 || memcmp(p, "\x93NUMPY", 6) != 0) {
        err = "not an NPY payload";
        return false;
    }
    const int major = p[6];
    size_t hlen, off;
    if (major == 1) {
        hlen = le16(p + 8);
        off = 10;
    } else if ((major == 2 || major == 3) && n >= 12) {
        hlen = le32(p + 8);
        off = 12;
    } else {
        err = "unsupported NPY version";
        return false;
    }
    if (off + hlen > n) {
        err = "truncated NPY header";
        return false;
    }
    const std::string header((const char*)p + off, hlen);
    off += hlen;
    auto field = [&](const char* key) -> size_t {
        const size_t k = header.find(std::string("'") + key + "'");
        return k == std::string::npos ? k : header.find(':', k);
    };
    const size_t d = field("descr"), f = field("fortran_order"), sh = field("shape");
    if (d == std::string::npos || f == std::string::npos || sh == std::string::npos) {
        err = "invalid NPY header";
        return false;
    }
    const size_t q = header.find_first_not_of(" ", d + 1);
    if (q == std::string::npos || (header[q] != '\'' && header[q] != '"')) {
        err = "structured NPY arrays are not supported";
        return false;
    }
    const size_t qe = header.find(header[q], q + 1);
    const std::string descr = header.substr(q + 1, qe - q - 1);
    const bool fortran = header.compare(header.find_first_not_of(" ", f + 1), 4, "True") == 0;
    const size_t ps = header.find('(', sh), pe = header.find(')', sh);
    if (ps == std::string::npos || pe == std::string::npos) {
        err = "invalid NPY shape";
        return false;
    }
    int64_t count = 1;
    for (size_t i = ps + 1; i < pe;) {
        if (!isdigit((unsigned char)header[i])) {
            ++i;
            continue;
        }
        int64_t v = 0;
        while (i < pe && isdigit((unsigned char)header[i]))
            v = v * 10 + (header[i++] - '0');
        out.shape.push_back(v);
        count *= v;
    }
    if (descr.size() < 3 || (descr[0] != '<' && descr[0] != '|' && descr[0] != '=')) {
        err = "big-endian NPY data";
        return false;
    }
    const char kind = descr[1];
    const int bytes = atoi(descr.c_str() + 2);
    if (!((kind == 'f' && (bytes == 2 || bytes == 4 || bytes == 8)) ||
          ((kind == 'i' || kind == 'u') && (bytes == 1 || bytes == 2 || bytes == 4 || bytes == 8)) ||
          (kind == 'b' && bytes == 1))) {
        err = "unsupported NPY dtype " + descr;
        return false;
    }
    if (count < 0 || off + (size_t)count * bytes > n) {
        err = "truncated NPY data";
        return false;
    }
    std::vector<double> flat((size_t)count);
    for (int64_t i = 0; i < count; ++i) {
        const uint8_t* e = p + off + (size_t)i * bytes;
        double v;
        if (kind == 'f') {
            if (bytes == 2)
                v = half_to_float(le16(e));
            else if (bytes == 4) {
                float t;
                memcpy(&t, e, 4);
                v = t;
            } else
                memcpy(&v, e, 8);
        } else {
            uint64_t u = 0;
            for (int b = 0; b < bytes; ++b)
                u |= (uint64_t)e[b] << (8 * b);
            if (kind == 'i' && bytes < 8 && (u >> (8 * bytes - 1)) & 1)
                u |= ~0ull << (8 * bytes);
            v = kind == 'i' ? (double)(int64_t)u : (double)u;
        }
        flat[(size_t)i] = v;
    }
    if (fortran && out.shape.size() > 1) { // reorder to C order
        std::vector<double> c(flat.size());
        const size_t nd = out.shape.size();
        std::vector<int64_t> idx(nd, 0);
        for (size_t lin = 0; lin < flat.size(); ++lin) {
            size_t fpos = 0, stride = 1;
            for (size_t k = 0; k < nd; ++k) {
                fpos += (size_t)idx[k] * stride;
                stride *= (size_t)out.shape[k];
            }
            c[lin] = flat[fpos];
            for (size_t k = nd; k-- > 0;) {
                if (++idx[k] < out.shape[k])
                    break;
                idx[k] = 0;
            }
        }
        flat.swap(c);
    }
    out.data = std::move(flat);
    out.descr = descr;
    return true;
}

// Reads a '<f4' .npy into `data` (flattened, C order) and returns its shape.
inline bool load_f32(const std::string& path, std::vector<float>& data, std::vector<int64_t>& shape) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "npy: cannot open %s\n", path.c_str());
        return false;
    }
    std::vector<uint8_t> bytes;
    uint8_t buf[1 << 16];
    for (size_t got; (got = fread(buf, 1, sizeof buf, f)) > 0;)
        bytes.insert(bytes.end(), buf, buf + got);
    fclose(f);
    Array arr;
    std::string err;
    if (!parse(bytes.data(), bytes.size(), arr, err)) {
        fprintf(stderr, "npy: %s: %s\n", path.c_str(), err.c_str());
        return false;
    }
    if (arr.descr != "<f4") {
        fprintf(stderr, "npy: %s is not <f4\n", path.c_str());
        return false;
    }
    shape = arr.shape;
    data.assign(arr.data.begin(), arr.data.end());
    return true;
}

} // namespace npy
