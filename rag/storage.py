"""Tenant-scoped catalog. Versions activate only after vector upsert succeeds."""
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import time
from .models import Chunk, digest, identity


class Catalog:
    def __init__(self, root: Path, tenant: str, signature: str):
        self.tenant = identity(tenant)
        self.namespace = "tenant-" + digest(tenant)[:32]
        self.root = root / self.namespace
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "catalog.sqlite3"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, source TEXT NOT NULL, active_version TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS versions (
                    document TEXT NOT NULL, version TEXT NOT NULL, created REAL NOT NULL,
                    PRIMARY KEY(document, version));
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, document TEXT NOT NULL, version TEXT NOT NULL,
                    payload TEXT NOT NULL, vector TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS chunk_version ON chunks(document, version);
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT NOT NULL,
                    question TEXT NOT NULL, answer TEXT NOT NULL, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY, session TEXT NOT NULL, text TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY, payload TEXT NOT NULL, expires REAL NOT NULL);
            """)
            db.execute("INSERT OR IGNORE INTO config VALUES ('signature', ?)", (signature,))
            existing = db.execute("SELECT value FROM config WHERE key='signature'").fetchone()[0]
            if existing != signature:
                raise ValueError("Embedding/backend changed. Use a new AI_DATA_DIR and re-ingest.")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    def is_active(self, document, version):
        with self.connect() as db:
            row = db.execute("SELECT active_version FROM documents WHERE id=?", (document,)).fetchone()
            return row is not None and row[0] == version

    def activate(self, document, version, source, chunks, vectors):
        if len(chunks) != len(vectors) or not chunks:
            raise ValueError("Cannot activate an empty or incomplete index")
        with self.connect() as db:
            db.executemany("INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?)", [
                (c.id, document, version, json.dumps(asdict(c)), json.dumps(v))
                for c, v in zip(chunks, vectors)])
            db.execute("INSERT OR IGNORE INTO versions VALUES (?, ?, ?)", (document, version, time.time()))
            db.execute("INSERT OR REPLACE INTO documents VALUES (?, ?, ?)", (document, source, version))
            db.execute("DELETE FROM cache")

    def active(self, document=None, kind=None):
        sql = """SELECT c.payload, c.vector FROM chunks c JOIN documents d
                 ON c.document=d.id AND c.version=d.active_version"""
        args = []
        if document:
            sql += " WHERE c.document=?"
            args.append(document)
        with self.connect() as db:
            rows = db.execute(sql + " ORDER BY c.id", args).fetchall()
        result = [(Chunk(**json.loads(payload)), json.loads(vector)) for payload, vector in rows]
        return [(c, v) for c, v in result if kind is None or c.kind == kind]

    def documents(self):
        with self.connect() as db:
            rows = db.execute("""SELECT d.id,d.source,d.active_version,v.version,v.created
                                 FROM documents d JOIN versions v ON d.id=v.document
                                 ORDER BY d.source,v.created DESC""").fetchall()
        return [dict(document=r[0], source=r[1], active_version=r[2], version=r[3], created=r[4]) for r in rows]

    def history(self, session, limit=6):
        identity(session)
        with self.connect() as db:
            rows = db.execute("SELECT question, answer FROM turns WHERE session=? ORDER BY id DESC LIMIT ?",
                              (session, limit)).fetchall()
        return [{"question": q, "answer": a} for q, a in reversed(rows)]

    def add_turn(self, session, question, answer):
        identity(session)
        with self.connect() as db:
            db.execute("INSERT INTO turns(session,question,answer,created) VALUES (?,?,?,?)",
                       (session, question, answer, time.time()))

    def remember(self, session, text):
        identity(session)
        if not text.strip() or len(text) > 2000:
            raise ValueError("Memory must contain 1-2000 characters")
        key = digest(session + ":" + text)
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO memories VALUES (?,?,?)", (key, session, text))
        return key

    def memories(self, session):
        identity(session)
        with self.connect() as db:
            return db.execute("SELECT id,text FROM memories WHERE session=? ORDER BY id", (session,)).fetchall()

    def clear_memory(self, session):
        identity(session)
        with self.connect() as db:
            db.execute("DELETE FROM memories WHERE session=?", (session,))
            db.execute("DELETE FROM turns WHERE session=?", (session,))

    def cache_get(self, key):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM cache WHERE key=? AND expires>?", (key, time.time())).fetchone()
        return json.loads(row[0]) if row else None

    def cache_put(self, key, value, ttl=600):
        with self.connect() as db:
            db.execute("DELETE FROM cache WHERE expires<=?", (time.time(),))
            db.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?)", (key, json.dumps(value), time.time()+ttl))
