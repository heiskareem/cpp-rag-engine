#include "io_utils.h"

#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

static bool has_text_ext(const fs::path& p) {
    auto ext = p.extension().string();
    for (auto& c : ext) c = (char)tolower((unsigned char)c);
    return ext == ".txt" || ext == ".md";
}

std::vector<std::string> list_text_files(const std::string& rootDir) {
    std::vector<std::string> out;
    fs::path root(rootDir);
    if (!fs::exists(root)) return out;
    if (fs::is_regular_file(root)) {
        if (has_text_ext(root)) out.push_back(root.string());
        return out;
    }
    for (auto& entry : fs::recursive_directory_iterator(root)) {
        if (entry.is_regular_file() && has_text_ext(entry.path())) {
            out.push_back(entry.path().string());
        }
    }
    return out;
}

std::string read_file_text(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

