"""Local embeddings (§7).

Kept behind an interface so the model is a config knob, not a code change —
EMBEDDING_MODEL in .env, not an import to edit.

model2vec was chosen over sentence-transformers deliberately: it's a static,
distilled model with no PyTorch dependency, ~30MB, and encodes in microseconds
on CPU. A desk robot's memory recall runs on every conversational turn; it
should not need a GPU-shaped dependency to remember your coffee order.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    dim: int

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """A single L2-normalized embedding vector."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Shape (len(texts), dim), each row L2-normalized."""


def _normalize(v: np.ndarray) -> np.ndarray:
    if v.ndim == 1:
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


class Model2VecEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "minishlab/potion-base-8M"):
        from model2vec import StaticModel   # deferred: this is an optional dep
        self._model = StaticModel.from_pretrained(model_name)
        self.dim = int(self._model.encode(["_"]).shape[1])

    def embed(self, text: str) -> np.ndarray:
        return _normalize(self._model.encode([text])[0].astype(np.float32))

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(self._model.encode(texts).astype(np.float32))
