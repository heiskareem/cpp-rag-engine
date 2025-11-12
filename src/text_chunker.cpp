#include "text_chunker.h"

#include <algorithm>

static std::string squish_newlines(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    bool prev_nl = false;
    for (char c : s) {
        if (c == '\r') continue;
        if (c == '\n') {
            if (!prev_nl) out.push_back('\n');
            prev_nl = true;
        } else {
            out.push_back(c);
            prev_nl = false;
        }
    }
    return out;
}

std::vector<std::string> chunk_text(const std::string& text, size_t chunkSize, size_t overlap) {
    std::vector<std::string> chunks;
    if (chunkSize == 0) return chunks;
    const std::string clean = squish_newlines(text);
    size_t step = chunkSize > overlap ? (chunkSize - overlap) : chunkSize;
    for (size_t i = 0; i < clean.size(); i += step) {
        size_t end = std::min(i + chunkSize, clean.size());
        chunks.emplace_back(clean.substr(i, end - i));
        if (end == clean.size()) break;
    }
    return chunks;
}

