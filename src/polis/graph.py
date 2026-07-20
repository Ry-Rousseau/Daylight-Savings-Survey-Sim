"""LangGraph fan-out/gather for the survey/query layer (R22).

A bounded map over a fixed set of agents that gathers their answers. This is
the *query-layer* orchestration tool only; the simulation tick loop (Phase 2+)
uses a separate custom loop, per R22 — one framework does not do both.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

try:  # Send's import path shifts across langgraph versions
    from langgraph.types import Send
except ImportError:  # pragma: no cover
    from langgraph.constants import Send

from .llm import retry_on_llm_error
from .survey import SurveyQuestion


class _SurveyState(TypedDict, total=False):
    question: SurveyQuestion
    agents: list[Any]
    answers: Annotated[list[dict], operator.add]
    agent: Any  # per-branch payload key set by the fan-out


def _fan_out(state: _SurveyState):
    return [Send("ask", {"agent": a, "question": state["question"]}) for a in state["agents"]]


def _ask(state: _SurveyState):
    # Retry the agent's answer, then *skip* it if the endpoint keeps returning a
    # schema-invalid response — one flaky agent must not abort the whole survey (the
    # soft json_schema enforcement means any model can occasionally miss). A skipped
    # agent contributes no answer, exactly like an abstain; the survey returns those
    # it collected.
    agent = state["agent"]
    ans = retry_on_llm_error(lambda: agent.answer(state["question"]))
    if ans is None:
        return {"answers": []}
    return {"answers": [{"agent_id": agent.persona.id, "choice": ans.choice, "reason": ans.reason}]}


def build_survey_graph():
    g = StateGraph(_SurveyState)
    g.add_node("ask", _ask)
    g.add_conditional_edges(START, _fan_out, ["ask"])
    g.add_edge("ask", END)
    return g.compile()


def run_survey(agents, question: SurveyQuestion) -> list[dict]:
    """Fan ``question`` out to ``agents`` and gather answers.

    Returns a list of ``{agent_id, choice, reason}`` dicts, sorted by agent id
    for stable output.
    """
    graph = build_survey_graph()
    out = graph.invoke({"question": question, "agents": list(agents), "answers": []})
    return sorted(out["answers"], key=lambda d: d["agent_id"])
