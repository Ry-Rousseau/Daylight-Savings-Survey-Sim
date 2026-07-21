"""Opinion-seeding layer (conviction slider) — deterministic, no network."""
import numpy as np

from polis.agent import Agent
from polis.memory import MemoryStore
from polis.opinion_seeds import (
    OpinionPlan,
    apply_opinion_plan,
    assign_stances,
    clean_text,
    load_opinion_corpus,
    render_opinion,
)
from polis.persona import Persona
from polis.questions import DST_OPTIONS
from polis.simulation import Population

DST, STD = DST_OPTIONS[0], DST_OPTIONS[1]


class FakeEmbedder:
    def encode(self, texts):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        arr = np.array([[float(len(t) % 7), 1.0] for t in items], dtype=np.float32)
        return arr[0] if single else arr


def _pop(n):
    agents = [
        Agent(Persona(f"a{i}", "a person"), object(), embedder=FakeEmbedder(), memory=MemoryStore())
        for i in range(n)
    ]
    return Population(agents)


_CORPUS = {
    DST: [{"text": "keep the summer light going all year", "reason": "r1", "likes": 5},
          {"text": "later sunsets are everything", "reason": "", "likes": 3}],
    STD: [{"text": "bright mornings matter for kids", "reason": "", "likes": 9}],
}


def test_clean_text_strips_urls_keeps_voice():
    assert clean_text("love it https://t.co/x  🌞  now") == "love it 🌞 now"


def test_render_phrasings():
    item = {"text": "later sunsets rule", "reason": "evenings"}
    assert "saw someone post" in render_opinion(item, "seen")
    assert "how I honestly feel" in render_opinion(item, "conviction")
    assert "evenings" in render_opinion(item, "reason")
    for p in ("seen", "conviction"):
        assert "later sunsets rule" in render_opinion(item, p)
    try:
        render_opinion(item, "bogus")
        assert False
    except ValueError:
        pass


def test_assign_two_camp_fractions_and_determinism():
    plan = OpinionPlan(scheme="two_camp", camps=((DST, 0.25), (STD, 0.25)), seed=1)
    a = assign_stances([f"a{i}" for i in range(100)], plan)
    assert sum(1 for s in a.values() if s == DST) == 25
    assert sum(1 for s in a.values() if s == STD) == 25
    assert len(a) == 50  # rest unseeded
    assert assign_stances([f"a{i}" for i in range(100)], plan) == a  # deterministic


def test_none_scheme_is_a_noop():
    pop = _pop(4)
    prov = apply_opinion_plan(pop, _CORPUS, OpinionPlan(scheme="none"))
    assert prov["n_seeded"] == 0
    assert all(len(ag.memory) == 0 for ag in pop.agents)


def test_apply_seeds_memories_and_marks_committed():
    pop = _pop(4)
    plan = OpinionPlan(scheme="two_camp", camps=((DST, 0.5), (STD, 0.5)),
                       n_opinions=2, committed=(STD,), seed=7)
    prov = apply_opinion_plan(pop, _CORPUS, plan)

    assert prov["n_seeded"] == 4 and prov["n_committed"] == 2
    seeded_dst = [ag for ag in pop.agents if prov["assignment"].get(ag.persona.id) == DST]
    seeded_std = [ag for ag in pop.agents if prov["assignment"].get(ag.persona.id) == STD]
    # DST pool has 2 items -> 2 memories; STD pool has 1 -> 1 memory (capped)
    assert all(len(ag.memory) == 2 for ag in seeded_dst)
    assert all(len(ag.memory) == 1 for ag in seeded_std)
    # committed STD agents are immovable (R11)
    assert all(ag.committed_stance == STD for ag in seeded_std)
    assert all(ag.committed_stance is None for ag in seeded_dst)
    # a seeded memory carries the rendered opinion text
    assert any("saw someone post" in r.text for r in seeded_dst[0].memory.records)


def test_load_corpus_real_file_groups_by_stance():
    corpus = load_opinion_corpus()  # real data/processed/tweets_labeled.csv
    assert DST in corpus and STD in corpus
    assert len(corpus[DST]) > 20 and len(corpus[STD]) > 20  # the balanced poles
    assert all("http" not in item["text"] for item in corpus[DST])  # urls stripped
