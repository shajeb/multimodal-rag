from dataclasses import asdict, dataclass
import hashlib
import re


def digest(value: str | bytes) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def tokens(text):
    return re.findall(r"\w+", text.lower())


def identity(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value):
        raise ValueError("Tenant and session IDs require 1-80 letters, digits, underscores or hyphens")
    return value


@dataclass
class Chunk:
    id: str
    document: str
    version: str
    source: str
    page: int
    kind: str
    text: str
    asset: str = ""

    def metadata(self):
        return {k: v for k, v in asdict(self).items() if k not in {"id", "text", "asset"}}
