"""Adapters isolate tenants and filter to active document versions."""
from .config import required
from .embeddings import cosine


def active_filter(chunks):
    pairs = sorted({(c.document, c.version) for c in chunks})
    clauses = [{"$and": [{"document": {"$eq": d}}, {"version": {"$eq": v}}]} for d, v in pairs]
    if not clauses:
        raise ValueError("An active version filter is required")
    kinds = sorted({c.kind for c in chunks})
    version_filter = clauses[0] if len(clauses) == 1 else {"$or": clauses}
    return {"$and": [version_filter, {"kind": {"$in": kinds}}]}


class LocalVectors:
    """Demo vectors persist in the catalog, not in an external database."""
    def upsert(self, chunks, vectors):
        pass

    def search(self, vector, rows, count):
        ranked = sorted(rows, key=lambda row: cosine(vector, row[1]), reverse=True)
        return [c.id for c, _ in ranked[:count]]


class ChromaVectors:
    def __init__(self, root, namespace):
        import chromadb
        self.client = chromadb.PersistentClient(path=str(root / "chroma"))
        self.collection = self.client.get_or_create_collection(name=namespace, metadata={"hnsw:space": "cosine"})

    def upsert(self, chunks, vectors):
        for offset in range(0, len(chunks), 100):
            batch = chunks[offset:offset+100]
            self.collection.upsert(ids=[c.id for c in batch], embeddings=vectors[offset:offset+100],
                                   documents=[c.text for c in batch], metadatas=[c.metadata() for c in batch])

    def search(self, vector, rows, count):
        if not rows:
            return []
        result = self.collection.query(query_embeddings=[vector], n_results=min(count, len(rows)),
                                       where=active_filter([c for c, _ in rows]), include=["distances"])
        return result["ids"][0]


class PineconeVectors:
    def __init__(self, namespace):
        from pinecone import Pinecone
        self.index = Pinecone(api_key=required("PINECONE_API_KEY")).Index(host=required("PINECONE_INDEX_HOST"))
        self.namespace = namespace

    def upsert(self, chunks, vectors):
        for offset in range(0, len(chunks), 100):
            records = [{"id": c.id, "values": v, "metadata": c.metadata()}
                       for c, v in zip(chunks[offset:offset+100], vectors[offset:offset+100])]
            self.index.upsert(vectors=records, namespace=self.namespace)

    def search(self, vector, rows, count):
        if not rows:
            return []
        result = self.index.query(vector=vector, top_k=count, namespace=self.namespace,
                                  filter=active_filter([c for c, _ in rows]), include_metadata=False)
        return [match.id for match in result.matches]
