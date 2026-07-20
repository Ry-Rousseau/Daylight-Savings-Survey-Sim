"""Agent = a persona bound to an LLM client + a private memory stream (Layers 1-2).

The memory step is additive: retrieve top-N for the question, inject it into the
prompt, then call the *unchanged* ``LLMClient.choose()`` contract. After
answering, the Q+A is written back into this agent's own stream (R19) so
repeated surveys over simulated time stay coherent.
"""
from __future__ import annotations

from . import prompts
from .embeddings import EmbeddingModel
from .importance import ImportanceFn, constant
from .llm import LLMClient
from .memory import KIND_SURVEY, MemoryRecord, MemoryStore, RetrievalConfig
from .persona import Persona
from .survey import SurveyAnswer, SurveyQuestion


class Agent:
    def __init__(
        self,
        persona: Persona,
        client: LLMClient,
        *,
        embedder: EmbeddingModel | None = None,
        memory: MemoryStore | None = None,
        retrieval: RetrievalConfig | None = None,
        importance_fn: ImportanceFn | None = None,
    ):
        self.persona = persona
        self.client = client
        self.embedder = embedder or EmbeddingModel()
        self.memory = memory or MemoryStore()  # private per agent (R2)
        self.retrieval = retrieval or RetrievalConfig()
        self.importance_fn = importance_fn or constant()

    def answer(self, question: SurveyQuestion, *, now: float = 0.0) -> SurveyAnswer:
        query_emb = self.embedder.encode(question.text)
        hits = self.memory.retrieve(query_emb, now, self.retrieval)
        user = prompts.survey_user(
            question.text, question.options, memories=[h.text for h in hits]
        )
        result = self.client.choose(
            system=self.persona.system_prompt(),
            user=user,
            options=question.options,
            temperature=self.persona.temperature,
        )
        answer = SurveyAnswer(choice=result["choice"], reason=result["reason"])
        self._remember_answer(question, answer, now)
        return answer

    def _remember_answer(
        self, question: SurveyQuestion, answer: SurveyAnswer, now: float
    ) -> None:
        """Write the survey response back into the memory stream as an event (R19)."""
        text = f"When asked '{question.text}', I chose '{answer.choice}' because {answer.reason}"
        self.memory.add(
            MemoryRecord(
                text=text,
                embedding=self.embedder.encode(text),
                importance=self.importance_fn(text),
                created_at=now,
                last_accessed_at=now,
                kind=KIND_SURVEY,
            )
        )
