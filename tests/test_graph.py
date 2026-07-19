"""Fan-out/gather wiring, exercised with stub agents (no network)."""
from dataclasses import dataclass

from polis.graph import run_survey
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


def test_fan_out_gathers_one_answer_per_agent():
    q = SurveyQuestion(text="DST?", options=["A", "B"])
    agents = [StubAgent("nurse", "A"), StubAgent("owner", "B"), StubAgent("retiree", "A")]

    results = run_survey(agents, q)

    assert len(results) == 3
    assert {r["agent_id"] for r in results} == {"nurse", "owner", "retiree"}
    assert all(r["choice"] in q.options for r in results)
