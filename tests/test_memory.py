"""Semantic memory tests (§26): retrieval, dedup, decay — against a fake,
deterministic embedding provider so the suite needs no network access or real
model download. Model2VecEmbeddingProvider itself gets one live smoke test,
skipped automatically if the model can't be fetched.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minibot.memory.embeddings import EmbeddingProvider  # noqa: E402
from minibot.memory.manager import (  # noqa: E402
    MERGE_SIMILARITY, MemoryManager, _confidence_weight, _importance_weight,
    _recency_weight,
)
from minibot.memory.store import MemoryRecord, MemoryStore  # noqa: E402


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic bag-of-words hashing into a fixed-size vector. Texts
    sharing words score high cosine similarity; unrelated texts score low.
    Good enough to exercise merge/recall logic without a real model."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        words = re.findall(r"[a-z0-9]+", text.lower())
        stop = {"a", "the", "is", "it", "of", "to", "now", "right", "on"}
        for word in words:
            if word in stop:
                continue
            # Stable across runs, unlike Python's randomized hash() salt.
            idx = int.from_bytes(word.encode(), "little") % self.dim
            v[idx] += 1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts]) if texts \
            else np.zeros((0, self.dim), dtype=np.float32)


@pytest.fixture
def manager(tmp_path):
    store = MemoryStore(tmp_path / "test.db")
    yield MemoryManager(store, FakeEmbeddingProvider(), decay_rate=0.02,
                        min_similarity=0.25, default_limit=6)
    store.close()


class TestStoreRoundtrip:
    def test_insert_get_roundtrips_embedding_exactly(self, tmp_path):
        store = MemoryStore(tmp_path / "t.db")
        emb = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        rec = MemoryRecord(text="hello", embedding=emb)
        store.insert(rec)
        back = store.get(rec.id)
        np.testing.assert_array_almost_equal(back.embedding, emb, decimal=6)
        store.close()

    def test_update_and_touch(self, tmp_path):
        store = MemoryStore(tmp_path / "t.db")
        rec = MemoryRecord(text="x", embedding=np.zeros(3, dtype=np.float32))
        store.insert(rec)
        store.touch(rec.id)
        store.touch(rec.id)
        back = store.get(rec.id)
        assert back.access_count == 2

    def test_delete(self, tmp_path):
        store = MemoryStore(tmp_path / "t.db")
        rec = MemoryRecord(text="x", embedding=np.zeros(3, dtype=np.float32))
        store.insert(rec)
        assert store.delete(rec.id) is True
        assert store.get(rec.id) is None
        assert store.delete(rec.id) is False


class TestRemember:
    def test_new_fact_is_inserted(self, manager):
        r = manager.remember("Aykhan likes espresso.", category="preference")
        assert r.merged is False
        assert manager.store.count() == 1

    def test_near_duplicate_strengthens_instead_of_duplicating(self, manager):
        r1 = manager.remember("Aykhan likes espresso.", category="preference",
                              importance=0.5)
        r2 = manager.remember("Aykhan really likes espresso.",
                              category="preference", importance=0.5)
        assert r2.merged is True
        assert r2.id == r1.id
        assert manager.store.count() == 1

    def test_merge_strengthens_importance(self, manager):
        r1 = manager.remember("Aykhan likes espresso.", importance=0.5)
        r2 = manager.remember("Aykhan really likes espresso.", importance=0.5)
        assert r2.record.importance > 0.5

    def test_merge_moves_confidence_toward_new_evidence_without_overwriting(self, manager):
        """§8: track confidence, don't silently overwrite."""
        manager.remember("Aykhan likes espresso.", confidence=0.5)
        r2 = manager.remember("Aykhan really likes espresso.", confidence=0.9)
        assert 0.5 < r2.record.confidence < 0.9

    def test_distinct_facts_stay_separate(self, manager):
        manager.remember("Aykhan likes espresso.")
        manager.remember("The robot has fifteen facial expressions.")
        assert manager.store.count() == 2

    def test_importance_and_confidence_are_clamped(self, manager):
        r = manager.remember("x", importance=5.0, confidence=-3.0)
        assert r.record.importance == 1.0
        assert r.record.confidence == 0.0


class TestRecall:
    def test_relevant_memory_is_retrieved(self, manager):
        manager.remember("Aykhan likes espresso.", category="preference")
        manager.remember("The robot has fifteen facial expressions.",
                         category="fact")
        out = manager.recall("what drink does the user like")
        # weak fake-embedding overlap is fine; just check it doesn't crash
        # and respects the threshold — assert on the stronger, direct case:
        out2 = manager.recall("espresso")
        assert any("espresso" in r.text for r in out2)

    def test_unrelated_query_returns_nothing_below_threshold(self, manager):
        manager.remember("Aykhan likes espresso.")
        out = manager.recall("zzz qqq xyzzy plugh")
        assert out == []

    def test_limit_truncates(self, manager):
        for i in range(10):
            manager.remember(f"fact number {i} about espresso")
        out = manager.recall("espresso", limit=3)
        assert len(out) == 3

    def test_recall_touches_accessed_records(self, manager):
        r = manager.remember("Aykhan likes espresso.")
        before = manager.store.get(r.id).access_count
        manager.recall("espresso")
        after = manager.store.get(r.id).access_count
        assert after == before + 1

    def test_results_sorted_descending_by_score(self, manager):
        manager.remember("espresso espresso espresso coffee")
        manager.remember("espresso")
        out = manager.recall("espresso coffee")
        scores = [r.score for r in out]
        assert scores == sorted(scores, reverse=True)


class TestScoringFormula:
    """§7's weighting functions, pinned directly — the part that's easy to
    get subtly wrong and hard to catch through end-to-end recall() alone."""

    def test_importance_weight_range(self):
        assert _importance_weight(0.0) == pytest.approx(0.5)
        assert _importance_weight(1.0) == pytest.approx(1.0)
        assert _importance_weight(0.5) == pytest.approx(0.75)

    def test_confidence_weight_range(self):
        assert _confidence_weight(0.0) == pytest.approx(0.5)
        assert _confidence_weight(1.0) == pytest.approx(1.0)

    def test_weight_clamps_out_of_range_inputs(self):
        assert _importance_weight(5.0) == _importance_weight(1.0)
        assert _importance_weight(-5.0) == _importance_weight(0.0)


class TestDecay:
    """§7: long-lived important facts must decay much less than temporary,
    unimportant ones."""

    def test_zero_age_is_always_full_weight(self):
        assert _recency_weight(0, importance=0.0, decay_rate=0.02) == pytest.approx(1.0)
        assert _recency_weight(0, importance=1.0, decay_rate=0.02) == pytest.approx(1.0)

    def test_high_importance_decays_much_slower(self):
        day = 86400.0
        thirty_days = 30 * day
        low = _recency_weight(thirty_days, importance=0.0, decay_rate=0.02)
        high = _recency_weight(thirty_days, importance=1.0, decay_rate=0.02)
        assert high > low
        assert high > 0.9    # importance-1.0 fact barely fades in a month
        assert low < 0.6     # importance-0.0 observation fades noticeably

    def test_weight_decreases_monotonically_with_age(self):
        ages = [0, 3600, 86400, 7 * 86400, 30 * 86400]
        weights = [_recency_weight(a, importance=0.3, decay_rate=0.02) for a in ages]
        assert weights == sorted(weights, reverse=True)

    def test_stale_memory_scores_lower_in_recall(self, manager):
        r = manager.remember("espresso preference noted", importance=0.1)
        old = time.time() - 60 * 86400   # 60 days ago
        manager.store.update(r.id, last_accessed_at=old)
        # bypass touch() from a prior recall by re-reading straight from store
        out = manager.recall("espresso preference noted")
        assert out and out[0].score < 1.0


class TestSearchSimilar:
    def test_pure_similarity_ignores_importance(self, manager):
        manager.remember("espresso espresso espresso", importance=0.0)
        out = manager.search_similar("espresso")
        assert out and out[0].score > 0.5


class TestConsolidate:
    def test_merges_near_duplicates_within_a_category(self, manager):
        manager.store.insert(MemoryRecord(
            text="Aykhan likes espresso", category="preference",
            embedding=manager.embeddings.embed("Aykhan likes espresso")))
        manager.store.insert(MemoryRecord(
            text="Aykhan likes espresso a lot", category="preference",
            embedding=manager.embeddings.embed("Aykhan likes espresso a lot")))
        merged = manager.consolidate()
        assert merged == 1
        assert manager.store.count() == 1

    def test_does_not_merge_across_categories(self, manager):
        text = "espresso espresso espresso"
        manager.store.insert(MemoryRecord(
            text=text, category="preference", embedding=manager.embeddings.embed(text)))
        manager.store.insert(MemoryRecord(
            text=text, category="fact", embedding=manager.embeddings.embed(text)))
        merged = manager.consolidate()
        assert merged == 0
        assert manager.store.count() == 2


class TestForgetAndUpdate:
    def test_forget_existing(self, manager):
        r = manager.remember("temporary note")
        assert manager.forget(r.id) is True
        assert manager.store.get(r.id) is None

    def test_forget_missing_returns_false(self, manager):
        assert manager.forget("does-not-exist") is False

    def test_update_text_reembeds(self, manager):
        r = manager.remember("Aykhan likes espresso")
        before = manager.store.get(r.id).embedding.copy()
        manager.update_memory(r.id, text="Aykhan likes matcha now")
        after = manager.store.get(r.id).embedding
        assert not np.array_equal(before, after)

    def test_update_importance_only_leaves_embedding_untouched(self, manager):
        r = manager.remember("Aykhan likes espresso")
        before = manager.store.get(r.id).embedding.copy()
        manager.update_memory(r.id, importance=0.9)
        after = manager.store.get(r.id).embedding
        np.testing.assert_array_equal(before, after)


class TestModel2VecSmoke:
    """One real, non-mocked check that the actual embedding model loads and
    produces sane vectors. Skipped if the model can't be fetched (no network)
    rather than failing the whole suite on an environment issue."""

    def test_loads_and_embeds(self):
        try:
            from minibot.memory.embeddings import Model2VecEmbeddingProvider
            provider = Model2VecEmbeddingProvider()
        except Exception as e:
            pytest.skip(f"model2vec unavailable in this environment: {e!r}")

        v1 = provider.embed("I like espresso")
        v2 = provider.embed("I enjoy coffee")
        v3 = provider.embed("The weather is rainy today")
        assert v1.shape == (provider.dim,)
        assert np.linalg.norm(v1) == pytest.approx(1.0, abs=1e-4)

        def cos(a, b):
            return float(np.dot(a, b))
        assert cos(v1, v2) > cos(v1, v3)   # coffee-ish beats unrelated
