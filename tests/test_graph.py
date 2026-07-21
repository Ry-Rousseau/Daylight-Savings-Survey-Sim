"""Fan-out/gather wiring, exercised with stub agents (no network)."""
from dataclasses import dataclass

from polis.graph import run_survey
from polis.llm import LLMError
from polis.survey import SurveyAnswer, SurveyQuestion


@dataclass
class _P:
    id: str


class StubAgent:
    def __init__(self, agent_id: str, choice: str):
        self.persona = _P(agent_id)
        self._choice = choice

    def answer(self, question):
        return SurveyAnswer(choice=self._choice, reason=f"{self.persona.id} picks {self._choice}")


class FlakyAgent:
    """An agent whose endpoint always returns a schema-invalid response."""

    def __init__(self, agent_id: str):
        self.persona = _P(agent_id)

    def answer(self, question):
        raise LLMError("no valid survey_answer")


def test_fan_out_gathers_one_answer_per_agent():
    q = SurveyQuestion(text="DST?", options=["A", "B"])
    agents = [StubAgent("nurse", "A"), StubAgent("owner", "B"), StubAgent("retiree", "A")]

    results = run_survey(agents, q)

    assert len(results) == 3
    assert {r["agent_id"] for r in results} == {"nurse", "owner", "retiree"}
    assert all(r["choice"] in q.options for r in results)


def test_flaky_agent_is_skipped_not_fatal(monkeypatch):
    # One agent whose endpoint keeps failing must not abort the whole survey — it is
    # skipped (like an abstain) and the rest are returned (P7 survey resilience).
    monkeypatch.setattr("polis.llm.time.sleep", lambda *_: None)  # no real backoff wait
    q = SurveyQuestion(text="DST?", options=["A", "B"])
    agents = [StubAgent("nurse", "A"), FlakyAgent("owner"), StubAgent("retiree", "A")]

    results = run_survey(agents, q)

    assert {r["agent_id"] for r in results} == {"nurse", "retiree"}  # owner skipped
    assert len(results) == 2


def test_skipped_agents_are_reported(monkeypatch, caplog):
    # A skipped agent must be *visible*, not a silently short list: run_survey logs the
    # skip and, on request, returns the skipped ids so an N=100 run can reconcile
    # coverage (how many of N actually answered).
    import logging

    monkeypatch.setattr("polis.llm.time.sleep", lambda *_: None)
    q = SurveyQuestion(text="DST?", options=["A", "B"])
    agents = [StubAgent("nurse", "A"), FlakyAgent("owner"), StubAgent("retiree", "A")]

    with caplog.at_level(logging.WARNING, logger="polis.graph"):
        answers, skipped = run_survey(agents, q, return_skipped=True)

    assert [r["agent_id"] for r in answers] == ["nurse", "retiree"]
    assert skipped == ["owner"]
    assert any("skipped" in rec.message and "owner" in rec.getMessage() for rec in caplog.records)


def test_no_skips_reports_empty(monkeypatch):
    # The clean path still returns the plain list by default, and an empty skipped list
    # when asked — so callers can't misread "no skips" as "feature unavailable".
    q = SurveyQuestion(text="DST?", options=["A", "B"])
    agents = [StubAgent("nurse", "A"), StubAgent("owner", "B")]

    answers, skipped = run_survey(agents, q, return_skipped=True)

    assert len(answers) == 2
    assert skipped == []
