# Architecture and operational limits

## Ingestion

1. Validate the file signature, filename and 30 MB size limit.
2. Derive a stable document ID from its logical filename and a version from the PDF bytes.
3. Skip an unchanged active version.
4. Extract prose and tables with pdfplumber, excluding table regions from the prose stream.
5. Extract embedded images with PyMuPDF. Normalize them to PNG; reuse content-addressed files and per-ingestion descriptions for repeated images.
6. In live mode, describe images with Gemini. Render scans and vector diagrams as page images for visual interpretation.
7. Split prose/descriptions at semantic changes and size limits. Keep table rows together where possible and repeat headers.
8. Embed and upsert through the chosen vector adapter.
9. Atomically activate the new version in SQLite only after successful upsert. Retain historical versions, but retrieve only the active version.

A failed write can leave unreferenced vectors/assets, which are not returned because every search is filtered against the active catalog. The application does not yet garbage-collect archived/orphaned records. Original PDFs are processed in temporary files and removed after ingestion; extracted evidence, image assets and versions persist.

## Question answering

The retriever combines up to 40 dense candidates and 40 BM25 candidates using reciprocal rank fusion. A cross-encoder reranks the top 30 fused candidates; up to six chunks become evidence. Demo mode replaces semantic models with explicit lexical approximations.

Cache keys include tenant storage, active chunk IDs, question, filters, embedding signature and reranker. A new version changes the key; local cache entries are also invalidated on ingestion. Redis is optional and falls back to SQLite. If dense search fails, BM25 remains usable, with a server log warning.

The LangChain prompt holds a fixed system policy and a JSON user payload. Evidence labels connect generated claims to filenames, pages, modalities and version IDs. Gemini/Grok receive retrieved text and vision-derived image descriptions. This is a **caption-based multimodal RAG architecture**, not a shared image/text embedding model. Fine visual details can be lost in captions.

Short-term memory includes six recent turns. Explicit long-term notes persist per tenant/session and are ranked with BM25. Memory provides conversational context, not factual evidence. Pronoun-based follow-up retrieval includes the previous user question.

Provider retries and optional fallback happen before the first token. A failure after tokens are emitted terminates the stream and does not save a completed memory turn. API streams return an error event; clients should offer retry.

## Isolation and authorization

HTTP tokens map to tenants on the server. Tenant IDs cannot traverse paths. Each tenant has a separate SQLite catalog and content-addressed asset folder. Chroma collections/Pinecone namespaces, retrieval caches and conversation memory are tenant-scoped. Retrieval also verifies that returned chunk IDs belong to the active authorized catalog.

The local Streamlit UI and CLI trust the operating-system user. Do not expose their tenant selectors as an internet-facing authentication mechanism.

## Limits to address before larger production deployment

- The catalog, memory and BM25 corpus are local SQLite data. Use a shared catalog, dedicated sparse index, ingestion queue and coordinated activation for multiple server replicas or large corpora.
- Pinecone writes can be eventually consistent. BM25 can still find newly activated records while dense search catches up.
- Captioning accuracy, table detection, OCR-like visual transcription and answer faithfulness depend on documents and models. The 30 MB/500-page limits do not replace isolated PDF processing for hostile uploads.
- Injection screening is heuristic. Prompt separation and evidence citations reduce risk but cannot guarantee instruction resistance or truthfulness.
- Citation-label validation checks references, not whether a claim follows from a cited passage. Source review remains necessary.
- Saved chat history and images are local plaintext data. Configure storage retention, encryption and access controls appropriate to deployment.
- Model/download availability, external quotas, provider model IDs and cloud index dimensions must be configured by the operator.
- Version histories are retained; rollback and retention controls are not exposed yet. Re-uploading previous PDF bytes reactivates that version.
- Vector namespaces include the local data directory fingerprint. Keep the data directory stable when restoring an installation.
