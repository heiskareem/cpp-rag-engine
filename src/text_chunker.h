#pragma once

#include <string>
#include <vector>

// Split text into overlapping character chunks.
std::vector<std::string> chunk_text(const std::string& text, size_t chunkSize, size_t overlap);

