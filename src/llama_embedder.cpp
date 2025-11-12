#include "llama_embedder.h"

#include <llama.h>

#include <algorithm>
#include <stdexcept>
#include <thread>

static int hw_threads_default() {
    unsigned int n = std::thread::hardware_concurrency();
    return n > 0 ? (int)n : 4;
}

LlamaEmbedder::LlamaEmbedder(const std::string& model_path, int n_threads)
    : model_path_(model_path) {
    if (n_threads <= 0) n_threads_ = hw_threads_default(); else n_threads_ = n_threads;

    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only = false;
    model_ = llama_load_model_from_file(model_path.c_str(), mparams);
    if (!model_) throw std::runtime_error("Failed to load embedding model: " + model_path);

    llama_context_params cparams = llama_context_default_params();
    cparams.embedding = true;
    cparams.n_threads = n_threads_;
    ctx_ = llama_new_context_with_model(model_, cparams);
    if (!ctx_) throw std::runtime_error("Failed to create llama context for embeddings");
}

LlamaEmbedder::~LlamaEmbedder() {
    if (ctx_) llama_free(ctx_);
    if (model_) llama_free_model(model_);
}

int LlamaEmbedder::embedding_dim() const {
    return llama_n_embd(model_);
}

std::vector<float> LlamaEmbedder::embed(const std::string& text) const {
    if (!model_ || !ctx_) return {};

    // Tokenize
    std::vector<llama_token> tokens(text.size() + 16);
    int n_toks = llama_tokenize(model_, text.c_str(), (int)text.size(), tokens.data(), (int)tokens.size(), true /*add_bos*/, false /*special*/);
    if (n_toks < 0) {
        tokens.resize(-n_toks);
        n_toks = llama_tokenize(model_, text.c_str(), (int)text.size(), tokens.data(), (int)tokens.size(), true, false);
    }
    tokens.resize((size_t)std::max(0, n_toks));
    if (tokens.empty()) return std::vector<float>(embedding_dim(), 0.0f);

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
        throw std::runtime_error("llama_decode failed for embeddings");
    }
    llama_batch_free(batch);

    const int dim = embedding_dim();
    const float* emb = llama_get_embeddings(ctx_);
    std::vector<float> out(dim);
    if (emb) {
        std::copy(emb, emb + dim, out.begin());
    }
    return out;
}
