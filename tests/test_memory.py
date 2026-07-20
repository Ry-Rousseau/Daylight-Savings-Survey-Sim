"""Deterministic retrieval-scoring tests — no network, no embedding model.

Embeddings are hand-built unit vectors so recency/importance/relevance can be
isolated one at a time. The live DoD demo (P(permanent-DST) delta) lives in
notebooks/experiments/phase1_memory_dod.ipynb, not here.
"""
import numpy as np

from polis.memory import (
    KIND_SURVEY,
    MemoryRecord,
    MemoryStore,
    RetrievalConfig,
)


def _rec(vec, *, importance=5.0, created_at=0.0, text="m"):
    e = np.asarray(vec, dtype=np.float32)
    e = e / np.linalg.norm(e)
    return MemoryRecord(text=text, embedding=e, importance=importance, created_at=created_at)


def test_relevance_ranks_nearest_first():
    store = MemoryStore([_rec([1, 0], text="east"), _rec([0, 1], text="north")])
    # Query aligned with the first record; importance/recency are flat so only
    # relevance differentiates.
    hits = store.retrieve(np.array([1.0, 0.0], dtype=np.float32), now=0.0)
    assert hits[0].text == "east"


def test_recency_breaks_ties():
    # Identical embeddings + importance; only created_at differs.
    store = MemoryStore([
        _rec([1, 0], created_at=-50.0, text="old"),
        _rec([1, 0], created_at=-1.0, text="recent"),
    ])
    hits = store.retrieve(np.array([1.0, 0.0], dtype=np.float32), now=0.0)
    assert hits[0].text == "recent"


def test_importance_breaks_ties():
    store = MemoryStore([
        _rec([1, 0], importance=1.0, text="mundane"),
        _rec([1, 0], importance=10.0, text="poignant"),
    ])
    hits = store.retrieve(np.array([1.0, 0.0], dtype=np.float32), now=0.0)
    assert hits[0].text == "poignant"


def test_top_n_cutoff():
    store = MemoryStore([_rec([1, 0], text=str(i)) for i in range(10)])
    hits = store.retrieve(np.array([1.0, 0.0], dtype=np.float32), now=0.0,
                          cfg=RetrievalConfig(top_n=3))
    assert len(hits) == 3


def test_flat_component_contributes_nothing():
    # All importance equal -> importance component min-maxes to zeros, so it must
    # not swamp relevance. Nearest-by-relevance still wins.
    store = MemoryStore([
        _rec([1, 0], importance=5.0, text="near"),
        _rec([0, 1], importance=5.0, text="far"),
    ])
    weighted = RetrievalConfig(w_recency=0.0, w_importance=5.0, w_relevance=1.0)
    hits = store.retrieve(np.array([1.0, 0.0], dtype=np.float32), now=0.0, cfg=weighted)
    assert hits[0].text == "near"


def test_retrieve_marks_accessed():
    store = MemoryStore([_rec([1, 0], created_at=-5.0)])
    store.retrieve(np.array([1.0, 0.0], dtype=np.float32), now=3.0)
    assert store.records[0].last_accessed_at == 3.0


def test_empty_store_returns_nothing():
    assert MemoryStore().retrieve(np.array([1.0, 0.0], dtype=np.float32), now=0.0) == []


def test_roundtrip_serialization():
    store = MemoryStore([_rec([1, 0], importance=7.0, created_at=-2.0, text="hi")])
    store.records[0].kind = KIND_SURVEY
    back = MemoryStore.from_list(store.to_list())
    r = back.records[0]
    assert r.text == "hi" and r.importance == 7.0 and r.kind == KIND_SURVEY
    assert np.allclose(r.embedding, store.records[0].embedding)
