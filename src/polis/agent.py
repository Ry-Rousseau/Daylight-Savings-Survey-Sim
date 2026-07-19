"""Agent = a persona bound to an LLM client (Layers 1–2)."""
from __future__ import annotations

from .llm import LLMClient
from .persona import Persona
from .survey import SurveyAnswer, SurveyQuestion


class Agent:
    def __init__(self, persona: Persona, client: LLMClient):
        self.persona = persona
        self.client = client

    def answer(self, question: SurveyQuestion) -> SurveyAnswer:
        user = question.text + "\nChoose exactly one:\n" + "\n".join(f"- {o}" for o in question.options)
        result = self.client.choose(
            system=self.persona.system_prompt(),
            user=user,
            options=question.options,
            temperature=self.persona.temperature,
        )
        return SurveyAnswer(choice=result["choice"], reason=result["reason"])
