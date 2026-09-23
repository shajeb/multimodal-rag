from collections import Counter
from dataclasses import asdict
import json
import logging
import math
import os
from .models import Chunk, digest, tokens

log = logging.getLogger(__name__)


def bm25(query, texts):
    docs = [tokens(text) for text in texts]
    if not docs:
        return []
    avg = sum(map(len, docs)) / len(docs) or 1
    frequencies = Counter(term for doc in docs for term in set(doc))
    scores = []
    for doc in docs:
        counts = Counter(doc)
        score = 0.0
        for term in set(tokens(query)):
            freq = counts[term]
            idf = math.log(1 + (len(docs)-frequencies[term]+0.5)/(frequencies[term]+0.5))
            score += idf * freq * 2.5 / (freq + 1.5*(0.25+0.75*len(doc)/avg))
        scores.append(score)
    return scores


class Cache:
    def __init__(self, catalog):
        self.catalog = catalog
        self.redis = None
        if os.getenv("REDIS_URL"):
            import redis
            self.redis = redis.Redis.from_url(os.environ["REDIS_URL"], socket_connect_timeout=2, socket_timeout=2)

    def get(self, key):
        if self.redis:
            try:
                value = self.redis.get(self.catalog.namespace + ":" + key)
                if value:
                    return json.loads(value)
            except Exception:
                log.warning("Redis unavailable; using the local retrieval cache")
        return self.catalog.cache_get(key)

    def put(self, key, value):
        self.catalog.cache_put(key, value)
        if self.redis:
            try:
                self.redis.setex(self.catalog.namespace + ":" + key, 600, json.dumps(value))
            except Exception:
                log.warning("Redis unavailable; retrieval cached locally")


class Retriever:
    def __init__(self, catalog, vectors, embedder, rerank_model=None):
        self.catalog, self.vectors, self.embedder = catalog, vectors, embedder
        self.cache = Cache(catalog)
        self.rerank_model = rerank_model
        self.reranker = None

    def search(self, question, document=None, kind=None, count=6):
        if kind not in {None, "text", "table", "image"}:
            raise ValueError("kind must be text, table or image")
        rows = self.catalog.active(document, kind)
        if not rows:
            return []
        key = digest(json.dumps([str(self.catalog.root), question, document, kind, count, [c.id for c, _ in rows],
                                 self.embedder.signature, self.rerank_model]))
        cached = self.cache.get(key)
        if cached is not None:
            return [Chunk(**item) for item in cached]
        try:
            dense = self.vectors.search(self.embedder.embed([question])[0], rows, min(40, len(rows)))
        except Exception:
            # A sparse fallback is still useful during vector-service outages.
            log.warning("Dense retrieval failed; falling back to BM25", exc_info=True)
            dense = []
        scores = bm25(question, [c.text for c, _ in rows])
        sparse = [rows[i][0].id for i in sorted(range(len(rows)), key=lambda i: scores[i], reverse=True)[:40]
                  if scores[i] > 0]
        fused = Counter()
        allowed = {c.id: c for c, _ in rows}
        for ranking in (dense, sparse):
            for rank, key_id in enumerate(ranking):
                if key_id in allowed:
                    fused[key_id] += 1 / (60+rank+1)
        candidates = [allowed[i] for i, _ in fused.most_common(30)]
        if self.rerank_model and candidates:
            if self.reranker is None:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder(self.rerank_model)
            reranked = self.reranker.predict([(question, c.text) for c in candidates])
            candidates = [c for _, c in sorted(zip(reranked, candidates), key=lambda x: float(x[0]), reverse=True)]
        else:
            candidates.sort(key=lambda c: len(set(tokens(question)) & set(tokens(c.text))), reverse=True)
        result = candidates[:count]
        self.cache.put(key, [asdict(c) for c in result])
        return result
