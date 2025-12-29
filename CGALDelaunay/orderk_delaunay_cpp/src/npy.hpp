#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>

namespace NpyIO {

// Helper to determine word size from header description
static int get_word_size(const std::string& descr) {
    if (descr == "<f8") return 8; // double
    if (descr == "<f4") return 4; // float
    if (descr == "<u4") return 4; // uint32
    if (descr == "<i4") return 4; // int32
    return 0;
}

static bool load_npy_double(const char* path, std::vector<double>& out,
                            size_t& rows, size_t& cols) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) { std::perror("fopen"); return false; }
    
    // Check magic
    unsigned char magic[6];
    if (std::fread(magic, 1, 6, f) != 6 || magic[0] != 0x93 || std::memcmp(magic+1, "NUMPY", 5) != 0) {
        std::fprintf(stderr, "%s: not a .npy file\n", path);
        std::fclose(f);
        return false;
    }

    // Read version and header length
    unsigned char ver[2];
    if (std::fread(ver, 1, 2, f) != 2) { std::fclose(f); return false; }
    size_t header_len;
    if (ver[0] <= 1) {
        uint16_t hl;
        if (std::fread(&hl, 2, 1, f) != 1) { std::fclose(f); return false; }
        header_len = hl;
    } else {
        uint32_t hl;
        if (std::fread(&hl, 4, 1, f) != 1) { std::fclose(f); return false; }
        header_len = hl;
    }

    // Read header
    std::string header(header_len, '\0');
    if (std::fread(header.data(), 1, header_len, f) != header_len) { std::fclose(f); return false; }

    // Parse shape
    size_t pos_shape = header.find("shape");
    if (pos_shape == std::string::npos) { std::fclose(f); return false; }
    size_t lparen = header.find('(', pos_shape);
    size_t rparen = header.find(')', lparen);
    if (lparen == std::string::npos || rparen == std::string::npos) { std::fclose(f); return false; }

    std::string shape_str = header.substr(lparen + 1, rparen - lparen - 1);
    std::vector<size_t> dims;
    size_t idx = 0;
    while (idx < shape_str.size()) {
        while (idx < shape_str.size() && (shape_str[idx] == ' ' || shape_str[idx] == ',')) ++idx;
        if (idx >= shape_str.size()) break;
        size_t next = shape_str.find(',', idx);
        if (next == std::string::npos) next = shape_str.size();
        
        std::string token = shape_str.substr(idx, next - idx);
        if (!token.empty() && token != " ") {
             try {
                dims.push_back((size_t)std::strtoull(token.c_str(), nullptr, 10));
             } catch(...) {} // Ignore potential exceptions from strtoull
        }
        idx = next + 1;
    }

    if (dims.empty()) { std::fclose(f); return false; }
    rows = dims[0];
    cols = dims.size() > 1 ? dims[1] : 1;
    size_t count = rows * cols;

    // Parse descr
    size_t pos_descr = header.find("descr");
    std::string descr;
    if (pos_descr != std::string::npos) {
        size_t pos_colon = header.find(':', pos_descr);
        if (pos_colon != std::string::npos) {
            size_t start_quote = header.find_first_of("'\"", pos_colon); 
            if (start_quote != std::string::npos) {
                char quote_char = header[start_quote];
                size_t end_quote = header.find(quote_char, start_quote + 1);
                if (end_quote != std::string::npos) {
                    descr = header.substr(start_quote + 1, end_quote - start_quote - 1);
                }
            }
        }
    }

    out.resize(count);
    size_t word_size = get_word_size(descr);
    
    if (word_size == 0) {
        std::fprintf(stderr, "Unsupported dtype: %s\n", descr.c_str());
        std::fclose(f);
        return false;
    }

    if (descr == "<f8") {
        if (std::fread(out.data(), 8, count, f) != count) { std::fclose(f); return false; }
    } else if (descr == "<f4") {
        std::vector<float> tmp(count);
        if (std::fread(tmp.data(), 4, count, f) != count) { std::fclose(f); return false; }
        for(size_t i=0; i<count; ++i) out[i] = (double)tmp[i];
    } else {
        std::fprintf(stderr, "Not a float/double array\n");
        std::fclose(f); 
        return false;
    }

    std::fclose(f);
    return true;
}

// Saves a 2D vector of integers (simplices) to NPY
template <typename T>
static bool save_npy_integers(const char* path, const std::vector<std::vector<T>>& data, size_t dim) {
    size_t rows = data.size();
    size_t cols = dim;
    
    std::FILE* f = std::fopen(path, "wb");
    if (!f) { std::perror("fopen"); return false; }
    
    unsigned char magic[6] = {0x93,'N','U','M','P','Y'};
    unsigned char ver[2] = {1,0};
    
    std::string type_code = (sizeof(T) == 4) ? "<i4" : "<i8"; // Assume signed int
    
    std::string header = "{'descr': '" + type_code + "', 'fortran_order': False, 'shape': (" + 
        std::to_string(rows) + ", " + std::to_string(cols) + "), }";
        
    // Pad header to 16 bytes
    while ((10 + header.size()) % 16 != 0) header.push_back(' ');
    header.back() = '\n';
    
    uint16_t hlen = (uint16_t)header.size();
    std::fwrite(magic, 1, 6, f);
    std::fwrite(ver, 1, 2, f);
    std::fwrite(&hlen, 2, 1, f);
    std::fwrite(header.data(), 1, header.size(), f);
    
    // Flatten data
    std::vector<T> flat;
    flat.reserve(rows * cols);
    for(const auto& row : data) {
        if(row.size() == cols) {
            flat.insert(flat.end(), row.begin(), row.end());
        } else {
            // Handle inconsistent sizes if necessary, or pad/error
            // For now, assume consistent
            flat.insert(flat.end(), row.begin(), row.end());
        }
    }
    
    std::fwrite(flat.data(), sizeof(T), flat.size(), f);
    std::fclose(f);
    return true;
}

} // namespace
