#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <cstdlib>

#include "io_utils.h"
#include "text_chunker.h"
#include "llama_embedder.h"
#include "llama_generator.h"
#include "vector_store.h"

#include <llama.h>

static void usage() {
    std::cout << "Usage:\n"
                 "  rag ingest --dir <path> --store <dir> --embed-model <path> [--chunk-size N] [--chunk-overlap N]\n"
                 "  rag query  --store <dir> --llm-model <path> --question <text> [--k N] [--max-tokens N] [--temp F] [--embed-model <path>]\n";
}

static std::string get_flag(int argc, char** argv, const std::string& name, const std::string& def = "") {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == name) return argv[i+1];
    }
    return def;
}

static bool has_flag(int argc, char** argv, const std::string& name) {
    for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == name) return true; return false;
}

static std::string build_rag_prompt(const std::string& question, const std::vector<SearchResult>& ctx) {
    std::string prompt;
    prompt += "You are a helpful assistant. Answer the question using ONLY the context.\n";
    prompt += "If the answer is not in the context, say you don't know.\n\n";
    prompt += "Context:\n";
    for (const auto& r : ctx) {
        prompt += "[Source: " + r.source + "]\n";
        prompt += r.text + "\n\n";
    }
    prompt += "Question: " + question + "\n";
    prompt += "Answer:";
    return prompt;
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 1; }
    std::string cmd = argv[1];

    if (cmd == "ingest") {
        llama_backend_init();
        std::string dir = get_flag(argc, argv, "--dir");
        std::string store = get_flag(argc, argv, "--store", ".rag_store");
        std::string embed_model = get_flag(argc, argv, "--embed-model");
        int chunk_size = std::stoi(get_flag(argc, argv, "--chunk-size", "800"));
        int chunk_overlap = std::stoi(get_flag(argc, argv, "--chunk-overlap", "200"));
        if (dir.empty() || embed_model.empty()) { usage(); return 2; }

        try {
            LlamaEmbedder embedder(embed_model);
            VectorStore vs(store);
            if (!vs.init_or_load(embedder.embedding_dim(), embedder.model_name())) {
                std::cerr << "Failed to init/load store\n"; return 3;
            }

            auto files = list_text_files(dir);
            std::cout << "Found " << files.size() << " files to ingest\n";
            size_t added = 0;
            for (const auto& f : files) {
                std::string content = read_file_text(f);
                auto chunks = chunk_text(content, (size_t)chunk_size, (size_t)chunk_overlap);
                for (size_t i = 0; i < chunks.size(); ++i) {
                    auto emb = embedder.embed(chunks[i]);
                    DocumentChunk c;
                    c.id = f + "#" + std::to_string(i);
                    c.source = f;
                    c.text = chunks[i];
                    c.embedding = std::move(emb);
                    if (vs.append(c)) ++added;
                }
            }
            std::cout << "Ingested chunks: " << added << "\n";
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n"; return 10;
        }
        llama_backend_free();
        return 0;
    }

    if (cmd == "query") {
        std::string store = get_flag(argc, argv, "--store", ".rag_store");
        std::string llm_model = get_flag(argc, argv, "--llm-model");
        std::string question = get_flag(argc, argv, "--question");
        std::string embed_model = get_flag(argc, argv, "--embed-model");
        int k = std::stoi(get_flag(argc, argv, "--k", "4"));
        int max_tokens = std::stoi(get_flag(argc, argv, "--max-tokens", "256"));
        float temp = std::stof(get_flag(argc, argv, "--temp", "0.0"));
        if (llm_model.empty() || question.empty()) { usage(); return 2; }

        try {
            llama_backend_init();
            VectorStore vs(store);
            // Initialize with dummy values; will be loaded from meta
            if (!vs.init_or_load(0, "")) { std::cerr << "Failed to load store\n"; return 3; }
            std::string embed_model_path = !embed_model.empty() ? embed_model : vs.embed_model_name();
            if (embed_model_path.empty()) { std::cerr << "Embed model not specified and not found in store meta\n"; return 4; }

            LlamaEmbedder embedder(embed_model_path);
            auto qvec = embedder.embed(question);
            auto hits = vs.query(qvec, k);
            if (hits.empty()) {
                std::cout << "No context found in store.\n"; return 0;
            }
            auto prompt = build_rag_prompt(question, hits);

            LlamaGenerator gen(llm_model);
            auto answer = gen.generate(prompt, max_tokens, temp);
            std::cout << answer << "\n";
            llama_backend_free();
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n"; return 10;
        }
        return 0;
    }

    usage();
    return 1;
}
