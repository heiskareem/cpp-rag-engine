#pragma once

#include <string>

struct llama_model;
struct llama_context;

class LlamaGenerator {
public:
    LlamaGenerator(const std::string& model_path, int n_threads = -1);
    ~LlamaGenerator();

    std::string generate(const std::string& prompt,
                         int max_new_tokens = 256,
                         float temperature = 0.0f) const;

    std::string model_name() const { return model_path_; }

private:
    std::string model_path_;
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    int n_threads_ = 4;
};

