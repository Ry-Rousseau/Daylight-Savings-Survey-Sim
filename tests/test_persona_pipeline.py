"""Persona-seeding pipeline (P6a, ADR 0016) — exercised network-free with stub
clients + a fake embedder. Covers the model-free rendering, the stage-differentiated
LLM steps, the Variant-A "reflection reads the items, not the backstory" guarantee,
the party/ideology exclusion, corpus round-trip determinism, and Population.from_corpus.
"""
import hashlib

import numpy as np

from polis import prompts
from polis.persona_pipeline import (
    REFLECTION_COLUMNS,
    SeededPersona,
    build_corpus,
    build_persona,
    corpus_from_dict,
    corpus_to_dict,
    describe_demographics,
    measure_contradictions,
    reflection_lines,
    sanitize_row,
    statement_memories,
)
from polis.simulation import Population


# --- test doubles --------------------------------------------------------------

class _Cfg:
    def __init__(self, model):
        self.model = model


class StubClient:
    """A ``complete``-only client: returns canned backstory for the backstory step and
    canned JSON for the reflection step (distinguished by the system prompt), recording
    every call so tests can inspect what each stage was actually shown."""

    def __init__(self, model="stub-model", reflection_json=None, backstory="A canned backstory about an ordinary life."):
        self.config = _Cfg(model)
        self.calls = []
        self._reflection_json = reflection_json or (
            '{"values": ["steady, concrete security"], '
            '"dispositions": ["outgoing but change-averse"]}'
        )
        self._backstory = backstory

    def complete(self, messages, *, temperature=None, max_tokens=None):
        self.calls.append({"messages": messages, "temperature": temperature})
        if messages[0]["content"] == prompts.REFLECTION_SYSTEM:
            return self._reflection_json
        return self._backstory


class FakeEmbedder:
    """Deterministic hash-based unit vectors — no torch/sentence-transformers."""

    def encode(self, texts):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        vecs = np.stack([self._vec(t) for t in items])
        return vecs[0] if single else vecs

    @staticmethod
    def _vec(text):
        h = np.frombuffer(hashlib.sha256(text.encode()).digest()[:16], dtype=np.uint8).astype(np.float32)
        return h / (np.linalg.norm(h) or 1.0)


# --- fixture rows (real 41-col schema shape) -----------------------------------

def _rows():
    nan = float("nan")
    return [
        {  # retiree, full disposition data, N/A occupation
            "person_id": "A1", "state_name": "Michigan", "region": "Midwest",
            "sex": "Female", "age": 64, "race_ethnicity": "White alone",
            "hispanic_origin": "Not Spanish/Hispanic/Latino", "education": "Bachelor's degree",
            "occupation": "N/A (less than 16 years old/NILF...)", "industry": "N/A (...)",
            "marital_status": "Widowed",
            "tipi_extraversion_anes": 6.0, "tipi_agreeableness_anes": 6.0,
            "tipi_conscientiousness_anes": 5.5, "tipi_emotional_stability_anes": 4.5,
            "tipi_openness_anes": 3.5, "moral_traditionalism_anes": 4.0,
            "egalitarianism_anes": 2.5, "opinion_strength_anes": 2.3,
            "party_id_7pt_anes": 7, "ideology_7pt_anes": 6,
            "religion_group_anes": "Protestant", "religion_importance_anes": "Important",
            "religion_attendance_anes": "Weekly",
            "anes_respondent_id": 995, "anes_match_distance": 0.05,
            "gss_respondent_id": "gss_1", "social_trust_gss": "Depends",
            "usual_hours_worked_gss": 0, "general_happiness_gss": "Pretty happy",
            "gss_match_distance": 0.02, "atus_respondent_id": 111,
            "wake_time_atus": "07:00:00", "atus_match_distance": 0.002,
        },
        {  # worker, some moderate TIPI (skipped), occupation present
            "person_id": "B2", "state_name": "Wisconsin", "region": "Midwest",
            "sex": "Male", "age": 33, "race_ethnicity": "Black or African American alone",
            "hispanic_origin": "Not Spanish/Hispanic/Latino",
            "education": "Regular high school diploma", "occupation": "CON-Carpenters",
            "industry": "CON-Construction", "marital_status": "Never married",
            "tipi_extraversion_anes": 4.0, "tipi_agreeableness_anes": 4.0,
            "tipi_conscientiousness_anes": 4.0, "tipi_emotional_stability_anes": 4.0,
            "tipi_openness_anes": 6.0, "moral_traditionalism_anes": 2.0,
            "egalitarianism_anes": 4.5, "opinion_strength_anes": 3.0,
            "party_id_7pt_anes": 2, "ideology_7pt_anes": 3,
            "religion_group_anes": "Not religious", "religion_importance_anes": "Not important",
            "religion_attendance_anes": "Never",
            "anes_respondent_id": 42, "anes_match_distance": 0.09,
            "gss_respondent_id": "gss_2", "social_trust_gss": "Can trust",
            "usual_hours_worked_gss": 40, "general_happiness_gss": "Very happy",
            "gss_match_distance": 0.03, "atus_respondent_id": 222,
            "wake_time_atus": "06:30:00", "atus_match_distance": 0.004,
        },
        {  # sparse — most disposition fields missing
            "person_id": "C3", "state_name": "Ohio", "region": "Midwest",
            "sex": "Female", "age": 45, "race_ethnicity": nan, "education": nan,
            "occupation": nan, "marital_status": nan,
            "tipi_extraversion_anes": nan, "tipi_openness_anes": nan,
            "wake_time_atus": nan,
        },
    ]


# --- model-free rendering ------------------------------------------------------

def test_describe_demographics_is_factual_and_handles_na():
    a, b, c = _rows()
    da = describe_demographics(a)
    assert "64-year-old" in da and "widowed" in da and "woman" in da
    assert "not currently working" in da  # N/A occupation
    assert "bachelor's degree" in da
    db = describe_demographics(b)
    assert "works as carpenters" in db  # ACS code prefix stripped, lower-cased
    assert describe_demographics(c) == "a 45-year-old woman who is not currently working"


def test_statement_memories_render_real_items_only():
    a, b, c = _rows()
    sa = statement_memories(a)
    assert any("up around 7am" in s for s in sa)  # ATUS wake time
    assert any("outgoing and talkative" in s for s in sa)  # TIPI extraversion high
    # moderate TIPI (agreeableness/conscientiousness 4.0) contributes no statement:
    sb = statement_memories(b)
    assert not any("warm and easy" in s for s in sb)
    assert any("curious and drawn to new ideas" in s for s in sb)  # openness high
    assert statement_memories(c) == []  # nothing present


def test_faith_statement_respects_group_and_attendance():
    # Guard against the internal ANES religion conflict: importance=Important must NOT
    # produce a lived-faith memory when the group is "Not religious" / attendance Never.
    base = {"person_id": "x", "sex": "Female", "age": 40, "religion_importance_anes": "Important"}
    conflicted = {**base, "religion_group_anes": "Not religious", "religion_attendance_anes": "Never"}
    coherent = {**base, "religion_group_anes": "Protestant", "religion_attendance_anes": "Weekly"}
    assert not any("faith" in s for s in statement_memories(conflicted))
    assert any("faith is an important part" in s for s in statement_memories(coherent))


def test_party_and_ideology_are_excluded_by_default():
    a = _rows()[0]
    lines = " ".join(reflection_lines(a)).lower()
    statements = " ".join(statement_memories(a)).lower()
    assert "party" not in lines and "ideology" not in lines
    assert "party" not in statements and "ideology" not in statements
    assert "party_id_7pt_anes" not in REFLECTION_COLUMNS


def test_reflection_lines_carry_measures():
    a = _rows()[0]
    lines = reflection_lines(a)
    assert any("Extraversion" in l and "high" in l for l in lines)
    assert any("Moral traditionalism" in l for l in lines)


# --- the LLM steps -------------------------------------------------------------

def test_build_persona_anchor_comes_from_reflection():
    client = StubClient(reflection_json='{"values": ["family and security"], "dispositions": ["blunt and firm"]}')
    sp = build_persona(_rows()[0], client)
    assert sp.persona.values == ("family and security",)
    assert sp.persona.dispositions == ("blunt and firm",)
    assert sp.persona.location == "Michigan"
    # backstory + demographic + disposition memories seeded at t=0
    texts = [m.text for m in sp.memories]
    assert any("A canned backstory" in t for t in texts)
    assert all(m.created_at == 0.0 for m in sp.memories)
    # provenance keeps the real donor ids (R29)
    assert sp.provenance["donors"]["anes"]["respondent_id"] == 995
    assert sp.provenance["donors"]["atus"]["respondent_id"] == 111


def test_reflection_reads_items_only_not_backstory_or_demographics():
    # Variant-A guarantee (ADR 0016): the reflection step is shown the real measure
    # lines ONLY — never the generated backstory (no compounding confabulation) and
    # never the demographics (withholding beats instructing-to-ignore stereotype).
    client = StubClient()
    build_persona(_rows()[0], client)  # a 64-year-old widowed White woman, Michigan
    reflection_call = next(c for c in client.calls if c["messages"][0]["content"] == prompts.REFLECTION_SYSTEM)
    user = reflection_call["messages"][1]["content"]
    assert "Extraversion" in user            # the real measures are present
    assert "A canned backstory" not in user  # the backstory is NOT
    for demo_token in ("64-year-old", "widowed", "Michigan", "woman"):
        assert demo_token not in user        # demographics are NOT


def test_measure_contradictions_detected():
    conflicted = {
        "person_id": "z", "sex": "Male", "age": 70,
        "occupation": "N/A (retired)", "usual_hours_worked_gss": 40,
        "religion_importance_anes": "Important", "religion_group_anes": "Not religious",
        "religion_attendance_anes": "Never",
    }
    flags = measure_contradictions(conflicted)
    assert any("not-working" in f for f in flags)
    assert any("religion importance" in f for f in flags)
    assert measure_contradictions(_rows()[1]) == []  # B2: employed carpenter, coherent


def test_cross_donor_hours_baked_out_religion_preserved():
    # The not-working + donor-hours artifact is BAKED OUT (nulled) so no stage sees it;
    # the within-donor religion tension is FLAGGED but preserved (a real profile).
    conflicted = {
        "person_id": "z", "sex": "Male", "age": 70,
        "occupation": "N/A (retired)", "usual_hours_worked_gss": 40,
        "religion_importance_anes": "Important", "religion_group_anes": "Not religious",
        "religion_attendance_anes": "Never",
    }
    clean, resolved = sanitize_row(conflicted)
    assert clean["usual_hours_worked_gss"] is None       # hours nulled
    assert any("hours" in r for r in resolved)
    assert clean["religion_importance_anes"] == "Important"  # religion untouched

    sp = build_persona(conflicted, StubClient())
    flags = sp.provenance["data_flags"]
    assert any("hours" in r for r in flags["resolved"])      # baked out, recorded
    assert any("religion importance" in f for f in flags["flagged"])  # preserved + flagged
    # the nulled measure reaches no stage: no work-hours in any seed memory
    assert not any("hours a week" in m.text for m in sp.memories)


def test_working_row_keeps_hours():
    # sanitize only touches the cross-donor artifact; an employed row is unchanged.
    clean, resolved = sanitize_row(_rows()[1])  # B2: carpenter, 40h
    assert clean["usual_hours_worked_gss"] == 40 and resolved == []


def test_stage_differentiated_models_are_recorded():
    bs = StubClient(model="small-creative")
    rf = StubClient(model="big-reasoner")
    sp = build_persona(_rows()[0], backstory_client=bs, reflection_client=rf)
    gen = sp.provenance["generation"]
    assert gen["backstory"]["model"] == "small-creative"
    assert gen["reflection"]["model"] == "big-reasoner"
    # stage-appropriate temperatures: backstory warm, reflection cool
    assert gen["backstory"]["temperature"] > gen["reflection"]["temperature"]


def test_empty_anchor_when_reflection_unparseable():
    client = StubClient(reflection_json="sorry, I can't do that")  # never parses
    sp = build_persona(_rows()[0], client)
    assert sp.persona.values == () and sp.persona.dispositions == ()  # persona still builds


# --- corpus artifact + from_corpus --------------------------------------------

def test_corpus_roundtrip_and_hash_is_deterministic():
    corpus1 = build_corpus(_rows(), StubClient())
    corpus2 = build_corpus(_rows(), StubClient())
    art1 = corpus_to_dict(corpus1, meta={"source": "test"})
    art2 = corpus_to_dict(corpus2)
    assert art1["meta"]["n"] == 3
    assert art1["meta"]["content_hash"] == art2["meta"]["content_hash"]  # deterministic
    # round-trips back to equivalent SeededPersonas
    restored = corpus_from_dict(art1)
    assert [sp.persona.id for sp in restored] == ["A1", "B2", "C3"]
    assert restored[0].persona.values == corpus1[0].persona.values


def test_from_corpus_builds_population_with_seeded_memories():
    corpus = build_corpus(_rows(), StubClient())
    artifact = corpus_to_dict(corpus, meta={"source": "test"})
    pop = Population.from_corpus(artifact, client=StubClient(), embedder=FakeEmbedder())
    assert len(pop.agents) == 3
    assert pop.roster == ("A1", "B2", "C3")
    a1 = pop.by_id["A1"]
    assert a1.persona.location == "Michigan"
    assert len(a1.memory) >= 2  # demographic + backstory (+ disposition) seeded
    assert pop.corpus_meta["content_hash"] == artifact["meta"]["content_hash"]  # R17
