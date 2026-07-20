"""Per-agent memory stream + retrieval (Layer 1 — Engine; R2, R19).

Each agent owns one ``MemoryStore``. There is no shared or global index, so
per-agent private state (R2) is enforced *structurally* — cross-agent leakage
is impossible by construction, not prevented by a metadata filter that a later
edit could drop.

Retrieval follows Park et al. (2023): score = weighted sum of recency,
importance, and relevance, each min-max normalized across the agent's own
memories, then top-N. Weights/decay/top_n live on ``RetrievalConfig`` — nothing
hardcoded, so any component can be ablated later.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

# Memory kinds. ``reflection`` is reserved for a later phase (P5); the field
# exists now so reflections slot in without a schema migration.
KIND_SEED = "seed"
KIND_SURVEY = "survey"
KIND_HEARD = "heard"  # something another agent said, delivered by the Game Master (P2)


@dataclass
class MemoryRecord:
    """One memory in an agent's stream.

    ``created_at``/``last_accessed_at`` are on an abstract time axis (a real sim
    clock arrives at P2); at P1 seeds author ``created_at`` as ages in the past,
    e.g. ``-5.0`` = five time-units ago relative to ``now=0``.
    """

    text: str
    embedding: np.ndarray
    importance: float = 5.0  # 1-10 (Park poignancy scale)
    created_at: float = 0.0
    last_accessed_at: float = 0.0
    kind: str = KIND_SEED

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["embedding"] = self.embedding.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryRecord":
        d = dict(d)
        d["embedding"] = np.asarray(d["embedding"], dtype=np.float32)
        return cls(**d)


@dataclass(frozen=True)
class RetrievalConfig:
    """Park-style retrieval weights. Exposed, never hardcoded (first-principle #4)."""

    w_recency: float = 1.0
    w_importance: float = 1.0
    w_relevance: float = 1.0
    decay: float = 0.995  # recency = decay ** age
    top_n: int = 5


def _min_max(x: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1] across candidates; a flat component contributes nothing."""
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


@dataclass
class MemoryStore:
    """An agent's private memory stream. Not shared — one instance per agent (R2)."""

    records: list[MemoryRecord] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)

    def add(self, record: MemoryRecord) -> None:
        self.records.append(record)

    def score_components(
        self, query_emb: np.ndarray, now: float, cfg: RetrievalConfig
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Per-record normalized (recency, importance, relevance) components and
        the weighted total — the raw material for R29 decision provenance. Each
        component is min-max normalized across the agent's own memories, exactly
        as it enters the ranking. No mutation.
        """
        if not self.records:
            z = np.zeros(0, dtype=np.float32)
            return z, z, z, z
        embs = np.stack([r.embedding for r in self.records])
        ages = np.array([max(now - r.created_at, 0.0) for r in self.records])
        recency = _min_max(cfg.decay**ages)
        importance = _min_max(
            np.array([r.importance for r in self.records], dtype=np.float32)
        )
        relevance = _min_max(embs @ query_emb)  # cosine (embeddings unit-normalized)
        total = (
            cfg.w_recency * recency
            + cfg.w_importance * importance
            + cfg.w_relevance * relevance
        )
        return recency, importance, relevance, total

    def score(
        self, query_emb: np.ndarray, now: float, cfg: RetrievalConfig
    ) -> np.ndarray:
        """Combined recency/importance/relevance score per record (no mutation)."""
        return self.score_components(query_emb, now, cfg)[3]

    def retrieve(
        self, query_emb: np.ndarray, now: float, cfg: RetrievalConfig | None = None
    ) -> list[MemoryRecord]:
        """Top-N memories for a query; marks the returned ones as accessed."""
        cfg = cfg or RetrievalConfig()
        if not self.records:
            return []
        scores = self.score(query_emb, now, cfg)
        order = np.argsort(-scores)[: cfg.top_n]
        hits = [self.records[i] for i in order]
        for r in hits:
            r.last_accessed_at = now
        return hits

    def retrieve_scored(
        self, query_emb: np.ndarray, now: float, cfg: RetrievalConfig | None = None
    ) -> list[tuple[MemoryRecord, tuple[float, float, float, float]]]:
        """Top-N with each hit's (recency, importance, relevance, total) components,
        for R29 provenance. Marks the returned records accessed, like ``retrieve``."""
        cfg = cfg or RetrievalConfig()
        if not self.records:
            return []
        recency, importance, relevance, total = self.score_components(query_emb, now, cfg)
        order = np.argsort(-total)[: cfg.top_n]
        out: list[tuple[MemoryRecord, tuple[float, float, float, float]]] = []
        for i in order:
            r = self.records[i]
            r.last_accessed_at = now
            out.append(
                (r, (float(recency[i]), float(importance[i]), float(relevance[i]), float(total[i])))
            )
        return out

    def to_list(self) -> list[dict]:
        return [r.to_dict() for r in self.records]

    @classmethod
    def from_list(cls, items: Sequence[dict]) -> "MemoryStore":
        return cls(records=[MemoryRecord.from_dict(d) for d in items])
