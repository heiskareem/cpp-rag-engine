#pragma once

#include <string>
#include <vector>
#include <optional>

struct DocumentChunk {
    std::string id;
    std::string source;
    std::string text;
    std::vector<float> embedding;
};

struct SearchResult {
    std::string id;
    std::string source;
    std::string text;
    float score; // cosine similarity
};

class VectorStore {
public:
    explicit VectorStore(std::string store_dir);

    // Create or load existing store; set embedding dim if new.
    bool init_or_load(int embedding_dim, const std::string& embed_model_name);

    // Append a chunk to the store (persists immediately).
    bool append(const DocumentChunk& c);

    // Load all items into memory (for search) — called by init.
    bool reload();

    // Brute-force cosine search, returns top-k.
    std::vector<SearchResult> query(const std::vector<float>& query_embedding, int top_k) const;

    int embedding_dim() const { return embedding_dim_; }
    const std::string& embed_model_name() const { return embed_model_name_; }

private:
    std::string store_dir_;
    std::string index_path_;
    std::string meta_path_;
    int embedding_dim_ = 0;
    std::string embed_model_name_;

    std::vector<DocumentChunk> items_;
};
