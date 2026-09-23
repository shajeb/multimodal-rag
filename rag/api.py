"""Authenticated API. Tenant identity comes only from the bearer-token mapping."""
from functools import lru_cache
import hmac
import json
import logging
import os
from typing import Literal
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from .config import Settings
from .models import identity
from .service import MAX_PDF_BYTES, RAGService

log = logging.getLogger(__name__)
app = FastAPI(title="Multimodal RAG", version="0.1.0")


@lru_cache(maxsize=64)
def service_for(tenant):
    return RAGService(Settings.load(), tenant)


def authenticated(authorization: str = Header(default="")):
    # Load .env even before the first service is created.
    Settings.load()
    try:
        mapping = json.loads(os.getenv("AI_API_TOKENS", "{}"))
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError()
        if any(not isinstance(k, str) or len(k) < 24 or not isinstance(v, str) for k, v in mapping.items()):
            raise ValueError()
    except (ValueError, TypeError):
        raise HTTPException(503, "Configure AI_API_TOKENS with bearer tokens of at least 24 characters")
    supplied = authorization.removeprefix("Bearer ") if authorization.startswith("Bearer ") else ""
    for token, tenant in mapping.items():
        if hmac.compare_digest(supplied, token):
            try:
                identity(tenant)
                return service_for(tenant)
            except ValueError:
                raise HTTPException(503, "Invalid tenant or service configuration")
    raise HTTPException(401, "Valid bearer token required", headers={"WWW-Authenticate": "Bearer"})


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    session: str = Field(default="default", pattern=r"^[A-Za-z0-9_-]{1,80}$")
    document: str | None = None
    kind: Literal["text", "table", "image"] | None = None


class Memory(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    session: str = Field(default="default", pattern=r"^[A-Za-z0-9_-]{1,80}$")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents")
def documents(service=Depends(authenticated)):
    return service.catalog.documents()


@app.post("/documents")
def ingest(file: UploadFile = File(...), service=Depends(authenticated)):
    raw = file.file.read(MAX_PDF_BYTES + 1)
    try:
        return service.ingest_bytes(raw, file.filename or "upload.pdf")
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception:
        log.exception("Ingestion failed")
        raise HTTPException(502, "Ingestion failed; check the server log and provider settings")


@app.post("/chat")
def chat(body: Question, service=Depends(authenticated)):
    try:
        return service.ask(body.question, body.session, body.document, body.kind)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception:
        log.exception("Answer generation failed")
        raise HTTPException(502, "Answer generation failed; check provider settings")


@app.post("/chat/stream")
def stream(body: Question, service=Depends(authenticated)):
    events = service.stream(body.question, body.session, body.document, body.kind)
    try:
        first = next(events)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception:
        log.exception("Stream preparation failed")
        raise HTTPException(502, "Could not prepare document context")

    def encode(event):
        return "event: " + event["event"] + "\ndata: " + json.dumps(event) + "\n\n"

    def generate():
        yield encode(first)
        try:
            for event in events:
                yield encode(event)
        except Exception:
            log.exception("Streaming failed")
            yield encode({"event": "error", "message": "Stream interrupted; retry the request."})
    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/memory")
def remember(body: Memory, service=Depends(authenticated)):
    try:
        return {"id": service.remember(body.session, body.text)}
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/memory/{session}")
def memory(session: str, service=Depends(authenticated)):
    try:
        return {"recent": service.catalog.history(session), "long_term": service.catalog.memories(session)}
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.delete("/memory/{session}")
def clear_memory(session: str, service=Depends(authenticated)):
    try:
        service.catalog.clear_memory(session)
        return {"status": "cleared"}
    except ValueError as exc:
        raise HTTPException(400, str(exc))
