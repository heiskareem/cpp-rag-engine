#include "vector_store.h"

#include <filesystem>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <cmath>

#include "minijson.h"

namespace fs = std::filesystem;

static float dot(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0.0f;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

static float norm(const std::vector<float>& a) {
    float s = 0.0f;
    for (float v : a) s += v * v;
    return std::sqrt(s);
}

VectorStore::VectorStore(std::string store_dir)
    : store_dir_(std::move(store_dir)) {
    index_path_ = (fs::path(store_dir_) / "index.jsonl").string();
    meta_path_ = (fs::path(store_dir_) / "meta.json").string();
}

bool VectorStore::init_or_load(int embedding_dim, const std::string& embed_model_name) {
    fs::create_directories(store_dir_);
    // load meta if exists
    if (fs::exists(meta_path_)) {
        std::ifstream in(meta_path_);
        if (in) {
            std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            int dim = 0; std::string model;
            if (minijson::extract_int(content, "embedding_dim", dim)) embedding_dim_ = dim;
            if (minijson::extract_string(content, "embed_model", model)) embed_model_name_ = model;
        }
    }
    if (embedding_dim_ == 0) {
        embedding_dim_ = embedding_dim;
        embed_model_name_ = embed_model_name;
        std::ofstream out(meta_path_);
        out << "{\"embedding_dim\":" << embedding_dim_ << ",\"embed_model\":\"" << minijson::escape(embed_model_name_) << "\"}";
    }
    return reload();
}

bool VectorStore::reload() {
    items_.clear();
    if (!fs::exists(index_path_)) return true; // empty store
    std::ifstream in(index_path_);
    if (!in) return false;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        DocumentChunk c;
        minijson::extract_string(line, "id", c.id);
        minijson::extract_string(line, "source", c.source);
        minijson::extract_string(line, "text", c.text);
        minijson::extract_float_array(line, "embedding", c.embedding);
        if ((int)c.embedding.size() == embedding_dim_) {
            items_.push_back(std::move(c));
        }
    }
    return true;
}

bool VectorStore::append(const DocumentChunk& c) {
    if ((int)c.embedding.size() != embedding_dim_) return false;
    // append to disk
    std::ofstream out(index_path_, std::ios::app);
    if (!out) return false;
    out << "{\"id\":\"" << minijson::escape(c.id) << "\",";
    out << "\"source\":\"" << minijson::escape(c.source) << "\",";
    out << "\"text\":\"" << minijson::escape(c.text) << "\",";
    out << "\"embedding\":[";
    for (size_t i = 0; i < c.embedding.size(); ++i) {
        if (i) out << ",";
        out << c.embedding[i];
    }
    out << "]}" << "\n";
    items_.push_back(c);
    return true;
}

std::vector<SearchResult> VectorStore::query(const std::vector<float>& query_embedding, int top_k) const {
    std::vector<SearchResult> results;
    if ((int)query_embedding.size() != embedding_dim_ || items_.empty()) return results;
    float qn = norm(query_embedding);
    if (qn == 0.0f) return results;
    std::vector<std::pair<float, size_t>> scored; scored.reserve(items_.size());
    for (size_t i = 0; i < items_.size(); ++i) {
        const auto& c = items_[i];
        float s = dot(query_embedding, c.embedding);
        float dn = norm(c.embedding);
        float cos = (dn > 0.0f) ? (s / (qn * dn)) : 0.0f;
        scored.emplace_back(cos, i);
    }
    std::partial_sort(scored.begin(), scored.begin() + std::min((size_t)top_k, scored.size()), scored.end(),
                      [](auto& a, auto& b){ return a.first > b.first; });
    for (size_t i = 0; i < scored.size() && (int)i < top_k; ++i) {
        const auto& c = items_[scored[i].second];
        results.push_back({c.id, c.source, c.text, scored[i].first});
    }
    return results;
}
