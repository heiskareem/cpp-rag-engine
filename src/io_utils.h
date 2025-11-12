#pragma once

#include <string>
#include <vector>

struct FileRecord {
    std::string path;
    std::string content;
};

// Recursively collect .txt and .md files under root.
std::vector<std::string> list_text_files(const std::string& rootDir);

// Read entire file as UTF-8 text (best-effort).
std::string read_file_text(const std::string& path);

