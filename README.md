# Nuclear Safeguards Copilot

**Nuclear Safeguards Copilot** is a Retrieval-Augmented Generation (RAG) prototype that answers questions about official IAEA nuclear safeguards documents. It retrieves evidence from a local PDF knowledge base, builds a grounded context package, and generates answers with document and page citations.

This is an **MVP** — a correct, modular pipeline designed for later enterprise expansion (hybrid search, auth, observability, multi-tenant indexes, etc.).

---

## Project Idea

Users ask natural-language questions such as:

- What is Material Balance Area?
- Explain Additional Protocol.
- Compare CSA and Additional Protocol.

The system:

1. Searches indexed PDF chunks semantically
2. Packages the best evidence with metadata
3. Prompts an LLM to answer **only from that evidence**
4. Returns the answer plus sources (document name, page, similarity score)

**No internet access is used at answer time.** The knowledge base is limited to PDFs you place in `Data/documents/`.

---

## Architecture

```text
┌─────────────┐     POST /chat      ┌──────────────────┐
│  Frontend   │ ──────────────────► │  FastAPI Backend │
│ HTML/Bootstrap                   │  routes/chat.py  │
└─────────────┘                     └────────┬─────────┘
                                             │
                                             ▼
                                    ┌──────────────────┐
                                    │   RAG Pipeline   │
                                    └────────┬─────────┘
                                             │
               ┌─────────────────────────────┼─────────────────────────────┐
               ▼                             ▼                             ▼
      SemanticRetriever              Context Builder                  LLM Client
   (SentenceTransformers+FAISS)   (dedupe, budget, labels)      (OpenAI / Ollama)
```

### Offline Indexing Pipeline

```text
PDFs (Data/documents)
  → Extract Text (PyMuPDF, page-aware)
  → Cleaning (normalize, fix hyphen breaks, remove artifacts)
  → Manual Structure-Aware Chunking
  → Metadata (chunk id, document, section, page, keywords, position, word count)
  → SentenceTransformer Embeddings (normalized)
  → FAISS IndexFlatIP + chunks.json
```

### Online Query Pipeline

```text
User Question
  → Embed Query (same model as chunks)
  → FAISS Top-K Retrieval
  → Context Package (dedupe, score filter, word budget, source labels)
  → Grounded Prompt (strict, citation-required)
  → LLM
  → Answer + Sources
```

---

## How This Builds on Lab 8 & Lab 9

| Lab Concept | Applied Here |
|---|---|
| Chunk table with metadata | `ChunkRecord` with document, section, page, keywords |
| `search_text` enrichment | Document name + section + keywords prepended to chunk text |
| SentenceTransformer + cosine similarity | `Embedder` + normalized vectors + FAISS inner product |
| Context package | `build_context_package()` — dedupe, budget, labeled sources |
| Strict grounded prompt | `build_grounded_prompt()` — context-only, refusal message |
| Modular retrievers | `BaseRetriever` interface; semantic now, hybrid later |
| Ollama / local LLM | `OllamaClient` via config (Lab 9 pattern) |
| Failure separation mindset | Retrieval, context, and generation are isolated modules |

**Not copied from labs:** PDF structure-aware manual chunking, FAISS persistence, FastAPI `/chat`, Bootstrap UI, YAML configuration.

---

## Library Choices

| Library | Why |
|---|---|
| **PyMuPDF (`fitz`)** | Fast, reliable PDF text extraction with page numbers for citations |
| **Sentence Transformers** | Same embedding approach as Lab 8; strong semantic retrieval |
| **FAISS** | Efficient vector search; easy to persist and reload |
| **FastAPI** | Lightweight async API, automatic OpenAPI docs, easy static file serving |
| **Bootstrap 5** | Clean MVP UI without Streamlit |
| **PyYAML** | Change LLM, models, and paths without code edits |
| **OpenAI SDK / Requests** | Pluggable cloud (OpenAI) or local (Ollama) LLM backends |

---

## Folder Structure

```text
rag/
├── Backend/                 # FastAPI app and /chat route
│   ├── main.py
│   └── routes/chat.py
├── Frontend/                # HTML + CSS + Bootstrap + JS
│   ├── index.html
│   ├── css/styles.css
│   └── js/app.js
├── RAG/                     # Core RAG logic
│   ├── chunker.py           # Manual structure-aware chunking
│   ├── context_builder.py   # Context package builder
│   ├── prompt_builder.py    # Grounded prompts
│   ├── llm_client.py        # OpenAI / Ollama clients
│   ├── pipeline.py          # End-to-end orchestration
│   └── retriever/           # Modular retrieval (semantic now)
├── Embeddings/              # SentenceTransformer wrapper
├── Vector_Database/         # FAISS store + metadata persistence
├── Utilities/               # PDF extraction + text cleaning
├── Configuration/           # settings.yaml + loader
├── Data/
│   ├── documents/           # Place PDFs here
│   └── index/               # Built FAISS index + chunks.json
├── scripts/
│   └── build_index.py       # Index builder CLI
├── requirements.txt
├── run.py                   # Start the server
└── README.md
```

---

## Installation

### 1. Prerequisites

- Python 3.10+
- (Optional) [Ollama](https://ollama.com/) for local LLM inference

### 2. Clone / open project and create virtual environment

```bash
cd rag
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add PDF documents

Place official PDF files in:

```text
Data/documents/
```

The project ships with five IAEA safeguards PDFs. To add more, copy any `.pdf` into this folder — **no code changes required**.

### 5. Build the vector index

```bash
python scripts/build_index.py
```

This extracts, chunks, embeds, and saves:

- `Data/index/faiss.index`
- `Data/index/chunks.json`

Re-run this script whenever you add or remove PDFs.

### 6. Configure the LLM

Edit `Configuration/settings.yaml`:

**Option A — Ollama (default)**

```yaml
llm:
  provider: "ollama"
  ollama:
    host: "http://127.0.0.1:11434"
    model: "mistral"
```

Ensure the model is pulled:

```bash
ollama pull mistral
```

**Option B — OpenAI**

```yaml
llm:
  provider: "openai"
  openai:
    api_key_env: "OPENAI_API_KEY"
    model: "gpt-4o-mini"
```

Set the environment variable:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-key"

# macOS / Linux
export OPENAI_API_KEY="your-key"
```

---

## Running the Project

```bash
python run.py
```

Open in browser:

```text
http://127.0.0.1:8000
```

API endpoint:

```text
POST http://127.0.0.1:8000/chat
Content-Type: application/json

{"question": "What is Material Balance Area?"}
```

Health check:

```text
GET http://127.0.0.1:8000/health
```

---

## Adding New PDF Files

1. Copy PDF(s) into `Data/documents/`
2. Run `python scripts/build_index.py`
3. Restart the server

The chunker, embedder, and FAISS store automatically include all PDFs in that folder.

---

## Changing the LLM

Edit `Configuration/settings.yaml`:

```yaml
llm:
  provider: "ollama"   # or "openai"
```

Adjust model names under `llm.ollama` or `llm.openai`. Restart the server after changes.

---

## Changing the Embedding Model

Edit `Configuration/settings.yaml`:

```yaml
embedding:
  model_name: "all-MiniLM-L6-v2"
```

Then rebuild the index (required — query and document vectors must use the same model):

```bash
python scripts/build_index.py
```

---

## Pipeline Details

### 1. Document Ingestion

- `Utilities/pdf_extractor.py` — page-level text extraction
- `Utilities/text_cleaner.py` — normalization and artifact removal

### 2. Manual Chunking

- `RAG/chunker.py` detects headings, paragraphs, lists, tables, and glossary lines
- Atomic blocks are merged up to `chunking.max_words`
- Each chunk stores: `chunk_id`, document name, section, page(s), keywords, position, word count

### 3. Embeddings

- `Embeddings/embedder.py` — SentenceTransformer with optional L2 normalization

### 4. Vector Database

- `Vector_Database/faiss_store.py` — FAISS `IndexFlatIP` + JSON metadata

### 5. Retrieval

- `RAG/retriever/semantic_retriever.py` — dense semantic search
- `RAG/retriever/base.py` — interface for future BM25 / hybrid retrievers

### 6. Context Package

- `RAG/context_builder.py` — deduplication, score thresholds, per-document limits, word budget, labeled blocks

### 7. Prompt + LLM

- `RAG/prompt_builder.py` — strict grounded prompt with citation rules
- `RAG/llm_client.py` — OpenAI or Ollama

### 8. API + UI

- `Backend/routes/chat.py` — single `/chat` endpoint
- `Frontend/` — question box, answer panel, sources with page and similarity score

---

## Evolving to Enterprise RAG

| MVP (now) | Enterprise (next) |
|---|---|
| Single FAISS flat index | Sharded / managed vector DB (Milvus, Pinecone, pgvector) |
| Semantic only | Hybrid BM25 + dense + reranker |
| Manual chunk heuristics | Layout-aware parsing (tables, figures), human QA on chunks |
| One `/chat` endpoint | Auth, rate limits, audit logs, feedback loop |
| YAML config | Secrets manager, environment profiles |
| Synchronous pipeline | Async ingestion queue, incremental index updates |
| No evaluation | Golden Q&A set, Precision@K, groundedness checks |
| Single tenant | Multi-tenant document isolation, RBAC |

The modular layout (`BaseRetriever`, separate index builder, config-driven LLM) is intentional so each layer can be upgraded independently.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `Index not found` | Run `python scripts/build_index.py` |
| Ollama connection error | Start Ollama and verify `llm.ollama.host` / model name |
| OpenAI auth error | Set `OPENAI_API_KEY` and set `llm.provider: openai` |
| Empty answers | Lower `retrieval.min_absolute_score` in settings or rebuild index |
| Slow first query | SentenceTransformer model downloads on first run |

---

## License

Prototype for educational and research use. Official IAEA documents remain subject to their original publication terms.
