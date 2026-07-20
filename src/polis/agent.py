"""Agent = a persona bound to an LLM client + a private memory stream (Layers 1-2).

The memory step is additive: retrieve top-N for the question, inject it into the
prompt, then call the *unchanged* ``LLMClient.choose()`` contract. After
answering, the Q+A is written back into this agent's own stream (R19) so
repeated surveys over simulated time stay coherent.
"""
from __future__ import annotations

from collections.abc import Sequence

from . import prompts
from .actions import (
    Action,
    ActionDecision,
    ActionType,
    ProvenanceEntry,
    RetrievalProvenance,
    action_json_schema,
)
from .embeddings import EmbeddingModel
from .importance import ImportanceFn, constant
from .llm import LLMClient
from .memory import KIND_SURVEY, MemoryRecord, MemoryStore, RetrievalConfig
from .persona import Persona
from .survey import SurveyAnswer, SurveyQuestion
from .world import WorldView


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

    def act(
        self,
        *,
        topic: str,
        stances: Sequence[str],
        world_view: WorldView,
        now: float = 0.0,
    ) -> ActionDecision:
        """Decide this tick's action (SPEAK a stance or ABSTAIN), returning the
        action plus its retrieval provenance (R29).

        Mirrors ``answer`` — retrieve → inject → constrained decode — but emits an
        action from the closed vocabulary (R23) rather than a survey choice.
        ``world_view`` is the tier-2 read seam; P2 agents condition only on their
        own memory (the shared tally is not yet a live consensus pressure).
        """
        query_emb = self.embedder.encode(topic)
        scored = self.memory.retrieve_scored(query_emb, now, self.retrieval)
        provenance = RetrievalProvenance(
            query=topic,
            hits=[
                ProvenanceEntry(
                    text=r.text, kind=r.kind, created_at=r.created_at,
                    recency=rec, importance=imp, relevance=rel, total=tot,
                )
                for r, (rec, imp, rel, tot) in scored
            ],
        )
        user = prompts.action_user(topic, stances, memories=[r.text for r, _ in scored])
        raw = self.client.decide(
            system=self.persona.system_prompt(),
            user=user,
            schema=action_json_schema(list(stances)),
            valid_types={t.value for t in ActionType},
            temperature=self.persona.temperature,
        )
        action = Action(
            action_type=raw["action_type"],
            stance=raw.get("stance"),
            utterance=raw.get("utterance"),
        )
        # usage/model surface the decode's token counts + pinned model id (R6) for
        # the P3 throughput/cost log; absent on fake clients, hence .get.
        return ActionDecision(
            action=action,
            provenance=provenance,
            usage=raw.get("usage"),
            model=raw.get("model"),
        )

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
