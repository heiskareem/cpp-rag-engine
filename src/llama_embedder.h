#pragma once

#include <string>
#include <vector>

struct llama_model;
struct llama_context;

class LlamaEmbedder {
public:
    LlamaEmbedder(const std::string& model_path, int n_threads = -1);
    ~LlamaEmbedder();

    // Compute embedding for input text.
    std::vector<float> embed(const std::string& text) const;

    int embedding_dim() const;

    std::string model_name() const { return model_path_; }

private:
    std::string model_path_;
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    int n_threads_ = 4;
};

