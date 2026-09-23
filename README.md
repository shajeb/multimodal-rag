# Multimodal Production RAG Pipeline

A Python application for asking grounded questions over PDF text, tables and images, with hybrid retrieval, streaming answers, persistent memory and tenant isolation.

Upload PDFs, ask questions about their text, tables and images, and inspect the page-level evidence used in each response. Start with the offline demo, then configure Gemini or Grok and Chroma or Pinecone for live operation.

## Run the app now

After completing the installation below, run these commands from the project folder in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Open the localhost address printed by Streamlit. Upload a PDF and click **Index documents**, then ask a question. This starts in **demo mode** unless you configure `.env`.

To try the included synthetic PDF:

```powershell
.\.venv\Scripts\python.exe scripts/create_sample.py
.\.venv\Scripts\python.exe -m rag.cli ingest data/sample/solar_report.pdf
.\.venv\Scripts\python.exe -m rag.cli ask "How many solar panels are installed?"
.\.venv\Scripts\python.exe -m rag.cli ask "What is the capacity?" --kind table --stream
```

Demo mode extracts real PDF text, tables and images, but uses deterministic lexical vectors and quoted excerpts. It does **not** simulate successful LLM or image understanding. Live mode supplies semantic embeddings, a cross-encoder reranker, Gemini image descriptions, and generated answers.

## Clean installation on another computer

Python 3.11 or newer is required; Python 3.12 is a practical default.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test,cloud]"
Copy-Item .env.example .env
```

These commands create an isolated project environment. On macOS or Linux, use `.venv/bin/python` in place of `.\venv\Scripts\python.exe`.

## Configure live multimodal RAG

Copy `.env.example` to `.env`, then set:

```dotenv
AI_MODE=live
AI_DATA_DIR=data/live
VECTOR_BACKEND=chroma
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-key
GEMINI_MODEL=your-available-vision-capable-model-id
```

Use a model ID available to your account. First use downloads the configured sentence-transformer and cross-encoder model weights. Re-ingest PDFs in the new data directory; demo vectors must not be reused as semantic vectors.

Grok uses the paid xAI API and remains optional. For Grok answers, set `LLM_PROVIDER=grok`, `GROK_API_KEY` and `GROK_MODEL`. **Keep Gemini configured**: it processes PDF images and rendered diagrams during ingestion. Grok answers from those visual descriptions plus extracted text and tables. Set `FALLBACK_PROVIDER=gemini` or `grok` with that provider's credentials for answer fallback.

For Pinecone, install the `cloud` extra, set `VECTOR_BACKEND=pinecone`, `PINECONE_API_KEY`, and `PINECONE_INDEX_HOST`. Use a cosine vector index whose dimensions match the embedding model; the default MiniLM model produces 384 dimensions. Create the index in your Pinecone account before ingestion. Chroma runs locally without a cloud account.

Changing embedding models or vector backends requires a new `AI_DATA_DIR` and re-ingestion. The catalog rejects incompatible configurations.

To enable Redis retrieval caching, set `REDIS_URL`. If Redis is unavailable, the tenant's SQLite TTL cache remains available.

## Resume feature coverage

| Resume requirement | Implementation |
|---|---|
| Questions over PDF text, tables and images | `rag/ingestion.py`, `rag/service.py`; vision-derived image descriptions join the searchable corpus |
| pdfplumber and PyMuPDF | pdfplumber extracts prose/tables; PyMuPDF extracts images and renders scanned/vector pages |
| Table/image extraction and deduplication | Table regions are excluded from prose; image bytes use content-addressed files; duplicate page elements and chunks collapse |
| Adaptive semantic chunking | Sentence embedding similarity determines boundaries, with a hard size limit; table chunks repeat headers and retain rows |
| Dense + sparse hybrid retrieval | Sentence-transformer embeddings, Chroma/Pinecone search, BM25 and reciprocal rank fusion |
| Reranking | Cross-encoder scoring of fused candidates in live mode |
| Caching | Tenant-scoped SQLite TTL cache, optional Redis; keys include corpus versions, filters and model settings |
| Gemini/Grok integration | Gemini SDK and xAI streaming API adapters, bounded retries, configurable pre-token fallback |
| Structured context and memory injection | LangChain chat prompt with separately encoded evidence, history and saved notes |
| Streaming answers | Provider token streaming, CLI JSON events, FastAPI SSE and Streamlit updates |
| Short-term memory | Recent six conversation turns, persisted per tenant/session |
| Long-term memory | Explicitly saved notes persist across restarts and are ranked for injection into each request |
| Prompt-injection guardrails | Input checks, suspicious context exclusion, system/data separation and citation-label checks |
| Chroma/Pinecone modular adapters | `rag/vectors.py`, same service/retriever interface |
| Document versioning | SHA-256 content versions, unchanged-upload detection, active-version filtering and retained history |
| Multi-tenant isolation | Separate tenant catalogs/assets, isolated vector namespaces, scoped cache and memory, bearer-token-derived API tenant |
| Production-oriented reliability | Bounded uploads, transactional activation after successful vector writes, sparse fallback, no retry after partial streamed output |

See [architecture.md](docs/architecture.md) for the design and operational limits.

## Dedicated project PDFs

Three original, fictional Aster Devices documents are in [output/pdf](output/pdf/README.md): a product catalog, sales report and support handbook. Each contains text, a table and a chart. They were created exclusively for this project; no PDFs from other projects were reused.

The folder README includes sample questions and expected facts. To regenerate, install the samples extra with `python -m pip install -e ".[samples]"` and run `python scripts/create_project_pdfs.py`.

## CLI

Global flags come before the command:

```powershell
.\.venv\Scripts\python.exe -m rag.cli --tenant team_a --session study_1 ingest "path\report.pdf"
.\.venv\Scripts\python.exe -m rag.cli --tenant team_a --session study_1 ask "What are the findings?" --stream
.\.venv\Scripts\python.exe -m rag.cli --tenant team_a documents
.\.venv\Scripts\python.exe -m rag.cli --tenant team_a --session study_1 remember "Use metric units and concise answers."
.\.venv\Scripts\python.exe -m rag.cli --tenant team_a --session study_1 memory
.\.venv\Scripts\python.exe -m rag.cli --tenant team_a --session study_1 clear-memory
```

Upload the same logical filename to create a new version of a document. Use `ingest --name report.pdf` when the physical filename changes. `ask --document DOCUMENT_ID` restricts retrieval to one document; `--kind text|table|image` selects a modality.

## Authenticated REST API

Add a JSON mapping to `.env`; replace the example token with a random secret of at least 24 characters:

```dotenv
AI_API_TOKENS={"replace-with-a-long-random-secret":"team_a"}
```

```powershell
.\.venv\Scripts\python.exe -m uvicorn rag.api:app --host 127.0.0.1 --port 8000
```

API documentation: http://127.0.0.1:8000/docs

Pass `Authorization: Bearer YOUR_TOKEN`. Clients cannot select a tenant in the request body.

- `POST /documents`: multipart PDF field named `file`.
- `GET /documents`: document version history.
- `POST /chat`: JSON `{"question":"What is the capacity?","session":"default"}`.
- `POST /chat/stream`: same body; SSE events `sources`, `token`, `done`, or `error`.
- `POST /memory`: JSON `{"session":"default","text":"Prefer metric units"}`.
- `GET /memory/{session}` and `DELETE /memory/{session}`.
- `GET /health`: process liveness, not a cloud-connectivity check.

Streamlit and the CLI are **trusted local administrator interfaces**. Their tenant selectors are not login systems; keep Streamlit bound to localhost. For multi-user hosting, use authenticated API access with TLS, appropriate rate limits and an identity provider.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Tests cover real multimodal PDF extraction, table deduplication, incremental ingestion, version replacement, failed-write recovery, memory persistence, tenant/session isolation, retrieval caching, prompt checks, streaming failure behavior, authenticated HTTP endpoints, a real local Chroma index, and the Pinecone request contract.

The latest local test run passed 34 tests. The dedicated synthetic corpus was also indexed in Pinecone and a chart-based answer was verified through live Gemini, semantic retrieval and cross-encoder reranking. The automated suite does not require paid LLM calls. External integrations require your credentials. Grok and Redis have not been validated against live services; the automated Grok tests use mocked responses. Heuristic prompt checks and citation checks do not establish factual correctness.

## Docker

```powershell
docker build -t multimodal-rag .
docker run --rm -p 127.0.0.1:8000:8000 --env-file .env -v multimodal-data:/app/data multimodal-rag
```

The image runs the API as a non-root user. Docker deployment is supplied but has not been exercised on this machine.

## Provider references

- [Gemini content generation](https://ai.google.dev/api/generate-content)
- [Grok chat API](https://docs.x.ai/developers/model-capabilities/text/streaming)
- [Chroma collections](https://docs.trychroma.com/reference/python/collection)
- [Pinecone namespaces](https://sdk.pinecone.io/python/how-to/vectors/namespaces.html)
