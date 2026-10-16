// Minimal .npy reader / writer, enough for the dumps this test compares:
// C-order, little-endian, no fortran_order, dtype <f4 or <i4. Everything is
// written as float32 or int32 so numpy loads it without a converter.
#ifndef TEST_GEMMA3_NPY_H
#define TEST_GEMMA3_NPY_H

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace npy {

inline void write(const std::string& path, const std::vector<uint32_t>& shape,
                  const void* data, bool isInt = false)
{
    std::string dict = std::string("{'descr': '") + (isInt ? "<i4" : "<f4")
                     + "', 'fortran_order': False, 'shape': (";
    size_t n = 1;
    for (size_t i = 0; i < shape.size(); ++i)
    {
        dict += std::to_string(shape[i]);
        dict += ",";
        n *= shape[i];
    }
    dict += "), }";
    // The header (magic + version + len + dict) is padded to a multiple of 64.
    size_t headerLen = 10 + dict.size() + 1;
    size_t pad = (64 - headerLen % 64) % 64;
    dict.append(pad, ' ');
    dict += '\n';

    FILE* f = fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + path);
    const char magic[] = "\x93NUMPY\x01\x00";
    fwrite(magic, 1, 8, f);
    uint16_t len = (uint16_t)dict.size();
    fwrite(&len, 2, 1, f);
    fwrite(dict.data(), 1, dict.size(), f);
    fwrite(data, 4, n, f);
    fclose(f);
}

// Reads a <f4 or <i4 array, returning the flat data and filling shape.
template <typename T>
inline std::vector<T> read(const std::string& path, std::vector<uint32_t>& shape)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot read " + path);

    char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "\x93NUMPY", 6) != 0)
        throw std::runtime_error(path + ": not a .npy file");
    uint16_t len = 0;
    if (fread(&len, 2, 1, f) != 1) throw std::runtime_error(path + ": short header");
    std::string dict(len, '\0');
    if (fread(dict.data(), 1, len, f) != len) throw std::runtime_error(path + ": short header");

    if (dict.find("<f4") == std::string::npos && dict.find("<i4") == std::string::npos)
        throw std::runtime_error(path + ": only <f4 / <i4 are supported, got " + dict);
    if (dict.find("'fortran_order': False") == std::string::npos)
        throw std::runtime_error(path + ": fortran_order is not supported");

    shape.clear();
    size_t open = dict.find('('), close = dict.find(')', open);
    std::string dims = dict.substr(open + 1, close - open - 1);
    size_t n = 1;
    for (size_t i = 0; i < dims.size();)
    {
        while (i < dims.size() && !isdigit((unsigned char)dims[i])) ++i;
        if (i >= dims.size()) break;
        uint32_t v = 0;
        while (i < dims.size() && isdigit((unsigned char)dims[i])) v = v * 10 + (dims[i++] - '0');
        shape.push_back(v);
        n *= v;
    }

    std::vector<T> out(n);
    if (fread(out.data(), 4, n, f) != n) throw std::runtime_error(path + ": short data");
    fclose(f);
    return out;
}

} // namespace npy

#endif // TEST_GEMMA3_NPY_H
