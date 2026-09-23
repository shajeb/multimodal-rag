import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data/demo")
    mode: str = "demo"
    backend: str = "local"
    provider: str = "gemini"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __post_init__(self):
        if self.mode not in {"demo", "live"}:
            raise ValueError("AI_MODE must be demo or live")
        if self.backend not in {"local", "chroma", "pinecone"}:
            raise ValueError("VECTOR_BACKEND must be local, chroma or pinecone")
        if self.mode == "live" and self.backend == "local":
            raise ValueError("Use chroma or pinecone for live semantic retrieval")
        if self.provider not in {"gemini", "grok"}:
            raise ValueError("LLM_PROVIDER must be gemini or grok")

    @classmethod
    def load(cls):
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        return cls(Path(os.getenv("AI_DATA_DIR", "data/demo")), os.getenv("AI_MODE", "demo"),
                   os.getenv("VECTOR_BACKEND", "local"), os.getenv("LLM_PROVIDER", "gemini"),
                   os.getenv("EMBEDDING_MODEL", cls.embedding_model),
                   os.getenv("RERANK_MODEL", cls.rerank_model))


def required(name):
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Set {name} in .env before using this integration")
    return value
