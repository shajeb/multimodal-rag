import math
from .models import digest, tokens


class DemoEmbeddings:
    """Lexical hashing for offline demos, NOT semantic embeddings."""
    signature = "demo-hash-256-v1"

    def embed(self, texts):
        result = []
        for text in texts:
            row = [0.0] * 256
            for token in tokens(text):
                row[int(digest(token)[:8], 16) % 256] += 1
            norm = math.sqrt(sum(v*v for v in row)) or 1
            result.append([v / norm for v in row])
        return result


class SemanticEmbeddings:
    def __init__(self, model):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model)
        self.signature = model

    def embed(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()


def cosine(a, b):
    return sum(x*y for x, y in zip(a, b))
