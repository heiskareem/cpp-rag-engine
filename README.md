# cpp-rag-engine

Minimal C++ RAG engine with:

- Local LLM + embeddings via Ollama (no heavy compile)
- Simple JSONL vector store (cosine similarity)
- CLI for ingesting documents and querying with RAG

This keeps dependencies light and lets you swap any free/local model (Llama 3, Mistral, Phi-3, etc.) in GGUF.

## Build

Prereqs:

- CMake 3.20+
- C++17 compiler (MSVC, Clang, or GCC)
- curl installed (for HTTP calls)
- Ollama running locally (WSL/Linux/Mac/Windows): https://ollama.com/download

Commands:

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target rag
```

Binary is at `build/rag` (or `build/Release/rag.exe` on Windows).

## Models (via Ollama)

- LLM (examples): `phi3.5:mini`, `mistral:instruct`, `llama3.2:3b-instruct`
- Embeddings: `nomic-embed-text`, `gte-small`

Pull models once:

```
ollama pull phi3.5:mini
ollama pull mistral:instruct
ollama pull nomic-embed-text
```

## Quickstart

1) Ingest a folder of `.txt`/`.md` files into a store (uses Ollama embeddings):

```
rag ingest --dir data/ --store .rag_store --embed-model nomic-embed-text
```

2) Ask a question with RAG using your Ollama LLM:

```
rag query --store .rag_store --llm-model phi3.5:mini \
          --question "What are the key points?" --k 5 --max-tokens 256
```

## CLI

- `ingest` — recursively index `.txt` and `.md`
  - `--dir <path>`: input directory
  - `--store <path>`: store directory (created if missing)
  - `--embed-model <name>`: Ollama embedding model (e.g., `nomic-embed-text`)
  - `--chunk-size <n>` (default 800), `--chunk-overlap <n>` (default 200)
- `query` — retrieve + generate (via Ollama)
  - `--store <path>`: store directory
  - `--llm-model <name>`: Ollama model name (e.g., `phi3.5:mini`)
  - `--question <text>`: your question
  - `--k <n>` (default 4): top matches
  - `--max-tokens <n>` (default 256)
  - `--temp <float>` (default 0.0, greedy)

## Notes

- This is a lean baseline. For larger corpora, swap the vector store for HNSW/FAISS and add persistence.
- PDF/HTML support not included; convert to `.txt` first.
- Ensure `ollama` is running (`ollama serve` usually starts automatically).
 - No CMake downloads: built‑in mini JSON replaces nlohmann/json; `build/` is disposable.
