#pragma once

#include <string>
#include <vector>

// Very small client that shells out to `curl` hitting the local Ollama server
// (http://127.0.0.1:11434). This avoids heavy compile deps.

class OllamaClient {
public:
    explicit OllamaClient(std::string host = "127.0.0.1", int port = 11434)
        : host_(std::move(host)), port_(port) {}

    // Generate text with a model name available in `ollama list`.
    // Returns empty string on error.
    std::string generate(const std::string& model,
                         const std::string& prompt,
                         int max_new_tokens,
                         float temperature) const;

    // Get embedding vector using an embedding-capable model (e.g. nomic-embed-text).
    // Returns empty vector on error.
    std::vector<float> embed(const std::string& model,
                             const std::string& text) const;

private:
    std::string host_;
    int port_;

    std::string url(const std::string& path) const;
    static std::string json_escape(const std::string& s);
    static std::string write_temp_json(const std::string& json_body);
    static std::string run_curl_post_file(const std::string& url,
                                          const std::string& tmp_path);
};

