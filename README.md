# cpp-rag-engine

Minimal C++ RAG engine with:

- Local LLM inference via `llama.cpp` (GGUF models)
- Local embeddings via `llama.cpp` embedding models (e.g., nomic-embed-text)
- Simple JSONL vector store (cosine similarity)
- CLI for ingesting documents and querying with RAG

This keeps dependencies light and lets you swap any free/local model (Llama 3, Mistral, Phi-3, etc.) in GGUF.

## Build

Prereqs:

- CMake 3.20+
- C++17 compiler (MSVC, Clang, or GCC)
- Internet to fetch dependencies on first configure (llama.cpp, nlohmann/json)

Commands:

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target rag
```

Binary is at `build/rag` (or `build/Release/rag.exe` on Windows).

## Models (free/local)

- LLM: pick any instruct GGUF model:
  - Llama 3.1 8B Instruct (Q4_K_M): good quality, more RAM/CPU
  - Mistral 7B Instruct v0.2 (Q4_K_M): lighter
  - Phi-3.5-mini-instruct (Q4_K_M): very light
- Embeddings: GGUF embedding model for high‑quality vectors:
  - `nomic-embed-text-v1.5` (GGUF)
  - `bge-small` / `gte-small` in GGUF if preferred

Download GGUFs from Hugging Face. Place them somewhere on disk. You can change models any time via CLI flags.

## Quickstart

1) Ingest a folder of `.txt`/`.md` files into a store:

```
rag ingest --dir data/ --store .rag_store --embed-model C:\\models\\nomic-embed-text-v1.5.f16.gguf
```

2) Ask a question with RAG using your LLM model:

```
rag query --store .rag_store --llm-model C:\\models\\Mistral-7B-Instruct-v0.2.Q4_K_M.gguf \
          --question "What are the key points?" --k 5 --max-tokens 256
```

## CLI

- `ingest` — recursively index `.txt` and `.md`
  - `--dir <path>`: input directory
  - `--store <path>`: store directory (created if missing)
  - `--embed-model <path>`: GGUF embedding model
  - `--chunk-size <n>` (default 800), `--chunk-overlap <n>` (default 200)
- `query` — retrieve + generate
  - `--store <path>`: store directory
  - `--llm-model <path>`: GGUF instruct model
  - `--question <text>`: your question
  - `--k <n>` (default 4): top matches
  - `--max-tokens <n>` (default 256)
  - `--temp <float>` (default 0.0, greedy)

## Notes

- This is a lean baseline. For larger corpora, swap the vector store for HNSW/FAISS and add persistence.
- PDF/HTML support not included; convert to `.txt` first.
- CPU build by default. Enable GPU in `llama.cpp` options if desired.
