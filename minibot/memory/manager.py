"""MemoryManager — the semantic (and, going forward, episodic) memory API (§6).

Retrieval combines more than raw cosine similarity (§7):

    retrieval_score = semantic_similarity
                     * importance_weight(importance)
                     * confidence_weight(confidence)
                     * recency_weight(age, importance)

recency_weight decays exponentially, but the decay RATE itself shrinks with
importance — an importance-1.0 memory decays at roughly a tenth of the base
rate, so "Aykhan likes espresso" does not fade just because it was said a
month ago, while "it's raining right now" (low importance) decays fast. Age
is measured from last_accessed_at rather than created_at, so a memory that
keeps getting recalled stays fresh on its own.

Deduplication on remember() (§8): a new fact is embedded and compared against
existing memories before insertion. A near-duplicate ("really likes espresso"
vs. "likes espresso") strengthens the existing row — bumps importance up,
moves confidence toward the new evidence, refreshes timestamps — rather than
creating a second, nearly-identical memory. Genuine contradictions are NOT
auto-resolved by string heuristics, which would be unreliable; the AI is
expected to call forget() on the stale memory first when it recognizes one,
per the tool's description.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from ..obs.logger import MEMORY
from .embeddings import EmbeddingProvider
from .store import MemoryRecord, MemoryStore

MERGE_SIMILARITY = 0.86    # cosine above this = "the same fact", strengthen it
SECONDS_PER_DAY = 86400.0


@dataclass
class RememberResult:
    id: str
    merged: bool
    record: MemoryRecord


def _importance_weight(importance: float) -> float:
    return 0.5 + 0.5 * max(0.0, min(1.0, importance))


def _confidence_weight(confidence: float) -> float:
    return 0.5 + 0.5 * max(0.0, min(1.0, confidence))


def _recency_weight(age_seconds: float, importance: float, decay_rate: float) -> float:
    # Importance 1.0 -> ~10% of the base decay rate; importance 0.0 -> full rate.
    effective_rate = decay_rate * (1.0 - 0.9 * max(0.0, min(1.0, importance)))
    age_days = max(0.0, age_seconds) / SECONDS_PER_DAY
    return math.exp(-effective_rate * age_days)


class MemoryManager:
    def __init__(self, store: MemoryStore, embeddings: EmbeddingProvider,
                 decay_rate: float = 0.02, min_similarity: float = 0.25,
                 default_limit: int = 6):
        self.store = store
        self.embeddings = embeddings
        self.decay_rate = decay_rate
        self.min_similarity = min_similarity
        self.default_limit = default_limit

    # -- write -------------------------------------------------------
    def remember(self, text: str, type: str = "semantic", category: str = "general",
                importance: float = 0.5, confidence: float = 0.8,
                source: str = "conversation") -> RememberResult:
        text = text.strip()
        importance = max(0.0, min(1.0, importance))
        confidence = max(0.0, min(1.0, confidence))
        emb = self.embeddings.embed(text)

        best = self._most_similar(emb, candidates=self.store.all())
        if best is not None and best[1] >= MERGE_SIMILARITY:
            existing, sim = best
            merged_importance = min(1.0, max(existing.importance, importance) +
                                    0.05)   # said again -> a little more important
            merged_confidence = existing.confidence + \
                (confidence - existing.confidence) * 0.5
            self.store.update(existing.id, importance=merged_importance,
                              confidence=merged_confidence)
            MEMORY.info(f"remember: merged into {existing.id!r} "
                       f"(sim={sim:.2f}) \"{existing.text[:60]}\"")
            existing.importance, existing.confidence = merged_importance, merged_confidence
            return RememberResult(existing.id, True, existing)

        rec = MemoryRecord(text=text, type=type, category=category,
                          importance=importance, confidence=confidence,
                          source=source, embedding=emb)
        self.store.insert(rec)
        MEMORY.info(f"remember: new {rec.id!r} [{category}] \"{text[:60]}\"")
        return RememberResult(rec.id, False, rec)

    def update_memory(self, memory_id: str, **fields) -> bool:
        if "text" in fields and fields["text"]:
            fields["embedding"] = self.embeddings.embed(fields["text"])
        return self.store.update(memory_id, **fields)

    def forget(self, memory_id: str) -> bool:
        ok = self.store.delete(memory_id)
        MEMORY.info(f"forget: {memory_id!r} {'ok' if ok else 'not found'}")
        return ok

    # -- read ----------------------------------------------------------
    def recall(self, query: str, limit: int | None = None,
              min_similarity: float | None = None) -> list[MemoryRecord]:
        limit = limit or self.default_limit
        threshold = self.min_similarity if min_similarity is None else min_similarity
        query_emb = self.embeddings.embed(query)
        now = time.time()

        scored: list[MemoryRecord] = []
        for rec in self.store.all():
            if rec.embedding is None:
                continue
            sim = _cosine(query_emb, rec.embedding)
            if sim < threshold:
                continue
            age = now - rec.last_accessed_at
            score = (sim * _importance_weight(rec.importance)
                    * _confidence_weight(rec.confidence)
                    * _recency_weight(age, rec.importance, self.decay_rate))
            rec.score = score
            scored.append(rec)

        scored.sort(key=lambda r: r.score, reverse=True)
        top = scored[:limit]
        for rec in top:
            self.store.touch(rec.id)
        best = f"{top[0].score:.2f}" if top else "-"
        MEMORY.info(f'query="{query[:60]}" retrieved={len(top)} best_score={best}')
        return top

    def search_similar(self, text: str, limit: int = 5) -> list[MemoryRecord]:
        """Pure semantic similarity, no importance/recency weighting — used
        for dedup and admin lookups rather than conversational recall."""
        emb = self.embeddings.embed(text)
        scored = []
        for rec in self.store.all():
            if rec.embedding is None:
                continue
            rec.score = _cosine(emb, rec.embedding)
            scored.append(rec)
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:limit]

    # -- maintenance -----------------------------------------------
    def consolidate(self) -> int:
        """Merge near-duplicates across the whole store. Greedy: for each
        memory, fold any later one that is essentially the same fact into it.
        O(n^2) — fine at the scale this store actually reaches (hundreds of
        rows, not millions)."""
        records = self.store.all()
        merged = 0
        absorbed: set[str] = set()
        for i, a in enumerate(records):
            if a.id in absorbed or a.embedding is None:
                continue
            for b in records[i + 1:]:
                if b.id in absorbed or b.embedding is None or b.category != a.category:
                    continue
                if _cosine(a.embedding, b.embedding) >= MERGE_SIMILARITY:
                    self.store.update(
                        a.id,
                        importance=min(1.0, max(a.importance, b.importance) + 0.05),
                        confidence=a.confidence + (b.confidence - a.confidence) * 0.5,
                    )
                    self.store.delete(b.id)
                    absorbed.add(b.id)
                    merged += 1
        if merged:
            MEMORY.info(f"consolidate: merged {merged} near-duplicate memories")
        return merged

    # -- internals ---------------------------------------------------
    def _most_similar(self, emb: np.ndarray, candidates: list[MemoryRecord]):
        best: tuple[MemoryRecord, float] | None = None
        for rec in candidates:
            if rec.embedding is None:
                continue
            sim = _cosine(emb, rec.embedding)
            if best is None or sim > best[1]:
                best = (rec, sim)
        return best


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    # Both sides are already L2-normalized by EmbeddingProvider, so this is
    # a plain dot product — kept as a guarded formula in case that changes.
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
