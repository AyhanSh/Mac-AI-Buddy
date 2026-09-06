"""SQLite-backed memory storage (§6).

Deliberately not sqlite-vec: that needs `enable_load_extension`, which some
Python builds (notably several macOS distributions) compile out, so it would
add a startup failure mode this project doesn't need. A desk robot's memory
holds hundreds of rows, not millions — brute-force cosine over an in-memory
numpy matrix is microseconds at that scale and has zero extension-loading
risk. search() is the only place that would change if this ever needed a real
vector index; nothing else in the codebase knows how similarity is computed.

Every persistent memory carries the metadata §6 asks for: id, text, type,
category, importance, confidence, createdAt, updatedAt, lastAccessedAt,
accessCount, embedding. `source` is added per the remember() signature in §6.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..obs.logger import MEMORY

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id              TEXT PRIMARY KEY,
    text            TEXT NOT NULL,
    type            TEXT NOT NULL,
    category        TEXT NOT NULL,
    importance      REAL NOT NULL,
    confidence      REAL NOT NULL,
    source          TEXT NOT NULL,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    last_accessed_at REAL NOT NULL,
    access_count    INTEGER NOT NULL DEFAULT 0,
    embedding       BLOB NOT NULL
);
"""


@dataclass
class MemoryRecord:
    text: str
    type: str = "semantic"
    category: str = "general"
    importance: float = 0.5
    confidence: float = 0.8
    source: str = "conversation"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    embedding: np.ndarray | None = None
    score: float = 0.0   # populated on recall(), not persisted

    def to_row(self) -> tuple:
        emb = (self.embedding.astype(np.float32).tobytes()
              if self.embedding is not None else b"")
        return (self.id, self.text, self.type, self.category, self.importance,
                self.confidence, self.source, self.created_at, self.updated_at,
                self.last_accessed_at, self.access_count, emb)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "MemoryRecord":
        emb = np.frombuffer(row["embedding"], dtype=np.float32) \
            if row["embedding"] else None
        return cls(id=row["id"], text=row["text"], type=row["type"],
                   category=row["category"], importance=row["importance"],
                   confidence=row["confidence"], source=row["source"],
                   created_at=row["created_at"], updated_at=row["updated_at"],
                   last_accessed_at=row["last_accessed_at"],
                   access_count=row["access_count"], embedding=emb)


class MemoryStore:
    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(SCHEMA)
        self._conn.commit()
        MEMORY.info(f"store: {self.path} ({self.count()} memories)")

    def close(self) -> None:
        self._conn.close()

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def insert(self, rec: MemoryRecord) -> None:
        self._conn.execute(
            "INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rec.to_row())
        self._conn.commit()

    def get(self, memory_id: str) -> MemoryRecord | None:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return MemoryRecord.from_row(row) if row else None

    def all(self, type_: str | None = None) -> list[MemoryRecord]:
        if type_:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE type = ?", (type_,)).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM memories").fetchall()
        return [MemoryRecord.from_row(r) for r in rows]

    def update(self, memory_id: str, **fields) -> bool:
        if not fields:
            return False
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k} = ?" for k in fields)
        vals = list(fields.values())
        if "embedding" in fields and isinstance(fields["embedding"], np.ndarray):
            vals[list(fields.keys()).index("embedding")] = \
                fields["embedding"].astype(np.float32).tobytes()
        cur = self._conn.execute(
            f"UPDATE memories SET {cols} WHERE id = ?", (*vals, memory_id))
        self._conn.commit()
        return cur.rowcount > 0

    def touch(self, memory_id: str) -> None:
        """Bump access stats. Recall's job, kept separate from update() since
        it happens on every read rather than an intentional edit."""
        self._conn.execute(
            "UPDATE memories SET access_count = access_count + 1, "
            "last_accessed_at = ? WHERE id = ?", (time.time(), memory_id))
        self._conn.commit()

    def delete(self, memory_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._conn.commit()
        return cur.rowcount > 0
