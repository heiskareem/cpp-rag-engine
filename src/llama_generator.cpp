#include "llama_generator.h"

#include <llama.h>

#include <algorithm>
#include <stdexcept>
#include <vector>
#include <string>
#include <cmath>
#include <thread>

static int hw_threads_default_gen() {
    unsigned int n = std::thread::hardware_concurrency();
    return n > 0 ? (int)n : 4;
}

LlamaGenerator::LlamaGenerator(const std::string& model_path, int n_threads)
    : model_path_(model_path) {
    if (n_threads <= 0) n_threads_ = hw_threads_default_gen(); else n_threads_ = n_threads;

    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only = false;
    model_ = llama_load_model_from_file(model_path.c_str(), mparams);
    if (!model_) throw std::runtime_error("Failed to load LLM model: " + model_path);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_threads = n_threads_;
    ctx_ = llama_new_context_with_model(model_, cparams);
    if (!ctx_) throw std::runtime_error("Failed to create llama context for generation");
}

LlamaGenerator::~LlamaGenerator() {
    if (ctx_) llama_free(ctx_);
    if (model_) llama_free_model(model_);
}

static std::string tok_to_str(const llama_model* model, llama_token tok) {
    char buf[16 * 1024];
    int n = llama_token_to_piece(model, tok, buf, sizeof(buf), false);
    if (n < 0) return std::string();
    return std::string(buf, buf + n);
}

std::string LlamaGenerator::generate(const std::string& prompt, int max_new_tokens, float temperature) const {
    if (!model_ || !ctx_) return {};

    // Tokenize prompt
    std::vector<llama_token> tokens(prompt.size() + 16);
    int n_toks = llama_tokenize(model_, prompt.c_str(), (int)prompt.size(), tokens.data(), (int)tokens.size(), true /*add_bos*/, false /*special*/);
    if (n_toks < 0) {
        tokens.resize(-n_toks);
        n_toks = llama_tokenize(model_, prompt.c_str(), (int)prompt.size(), tokens.data(), (int)tokens.size(), true, false);
    }
    tokens.resize((size_t)std::max(0, n_toks));

    // Evaluate prompt
    {
        llama_batch batch = llama_batch_init((int)tokens.size(), 0, 1);
        for (int i = 0; i < (int)tokens.size(); ++i) {
            batch.token[i] = tokens[i];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = false;
        }
        if (llama_decode(ctx_, batch) != 0) {
            llama_batch_free(batch);
            throw std::runtime_error("llama_decode failed for prompt");
        }
        llama_batch_free(batch);
    }

    std::string out;
    int eos = llama_token_eos(model_);

    llama_token prev = 0;
    for (int t = 0; t < max_new_tokens; ++t) {
        const float* logits = llama_get_logits(ctx_);
        int n_vocab = llama_n_vocab(model_);

        // Greedy or simple temperature sampling
        int next_id = 0;
        if (temperature <= 0.0f) {
            float best = -std::numeric_limits<float>::infinity();
            for (int i = 0; i < n_vocab; ++i) {
                if (logits[i] > best) { best = logits[i]; next_id = i; }
            }
        } else {
            // Softmax sampling with temperature
            std::vector<float> probs(n_vocab);
            float maxlog = -std::numeric_limits<float>::infinity();
            for (int i = 0; i < n_vocab; ++i) maxlog = std::max(maxlog, logits[i]);
            float sum = 0.0f;
            for (int i = 0; i < n_vocab; ++i) {
                float v = (logits[i] - maxlog) / std::max(0.001f, temperature);
                probs[i] = std::exp(v);
                sum += probs[i];
            }
            float r = (float)rand() / (float)RAND_MAX;
            float acc = 0.0f;
            for (int i = 0; i < n_vocab; ++i) {
                acc += probs[i] / sum;
                if (r <= acc) { next_id = i; break; }
            }
        }

        if (next_id == eos) break;

        // Append text
        out += tok_to_str(model_, next_id);

        // Feed next token
        llama_batch batch = llama_batch_init(1, 0, 1);
        batch.token[0] = next_id;
        batch.pos[0] = (int)tokens.size() + t;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;
        if (llama_decode(ctx_, batch) != 0) {
            llama_batch_free(batch);
            break;
        }
        llama_batch_free(batch);
        prev = next_id;
    }

    return out;
}
