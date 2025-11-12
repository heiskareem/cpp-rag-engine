#include "ollama_client.h"

#include "minijson.h"

#include <cstdio>
#include <cstdlib>
#include <array>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <filesystem>
#ifndef _WIN32
#include <unistd.h>
#endif

namespace fs = std::filesystem;

std::string OllamaClient::url(const std::string& path) const {
    std::ostringstream oss;
    oss << "http://" << host_ << ":" << port_ << path;
    return oss.str();
}

std::string OllamaClient::json_escape(const std::string& s) {
    std::ostringstream oss;
    for (unsigned char c : s) {
        switch (c) {
            case '"': oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\b': oss << "\\b"; break;
            case '\f': oss << "\\f"; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default:
                if (c < 0x20) {
                    oss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)c;
                } else {
                    oss << c;
                }
        }
    }
    return oss.str();
}

std::string OllamaClient::write_temp_json(const std::string& json_body) {
    char templ[] = "/tmp/ollama_reqXXXXXX";
    int fd = mkstemp(templ);
    if (fd == -1) return {};
    std::string path(templ);
    FILE* f = fdopen(fd, "wb");
    if (!f) { close(fd); return {}; }
    fwrite(json_body.data(), 1, json_body.size(), f);
    fclose(f);
    return path;
}

std::string OllamaClient::run_curl_post_file(const std::string& url, const std::string& tmp_path) {
    std::string cmd = "curl -s -X POST -H \"Content-Type: application/json\" --data-binary @" + tmp_path + " " + url;
    std::array<char, 4096> buf{};
    std::string out;
#if defined(_WIN32)
    FILE* pipe = _popen(cmd.c_str(), "r");
#else
    FILE* pipe = popen(cmd.c_str(), "r");
#endif
    if (!pipe) return {};
    while (true) {
        size_t n = fread(buf.data(), 1, buf.size(), pipe);
        if (n == 0) break;
        out.append(buf.data(), n);
    }
#if defined(_WIN32)
    _pclose(pipe);
#else
    pclose(pipe);
#endif
    // Cleanup
    fs::remove(tmp_path);
    return out;
}

std::vector<float> OllamaClient::embed(const std::string& model, const std::string& text) const {
    const std::string body = std::string("{\"model\":\"") + json_escape(model) + "\",\"prompt\":\"" + json_escape(text) + "\"}";
    const std::string tmp = write_temp_json(body);
    if (tmp.empty()) return {};
    const std::string resp = run_curl_post_file(url("/api/embeddings"), tmp);
    if (resp.empty()) return {};
    std::vector<float> out;
    if (!minijson::extract_float_array(resp, "embedding", out)) return {};
    return out;
}

std::string OllamaClient::generate(const std::string& model, const std::string& prompt, int max_new_tokens, float temperature) const {
    std::ostringstream body;
    body << "{\"model\":\"" << json_escape(model) << "\",";
    body << "\"prompt\":\"" << json_escape(prompt) << "\",";
    body << "\"stream\":false,\"options\":{\"temperature\":" << temperature << ",\"num_predict\":" << max_new_tokens << "}}";
    const std::string tmp = write_temp_json(body.str());
    if (tmp.empty()) return {};
    const std::string resp = run_curl_post_file(url("/api/generate"), tmp);
    if (resp.empty()) return {};
    std::string out;
    if (minijson::extract_string(resp, "response", out)) return out;
    return {};
}
