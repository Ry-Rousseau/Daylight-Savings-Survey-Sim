"""polis — a surveyable silicon sample of NYC opinion.

Source-first engine (the deliverable). Notebooks import down from here;
they never redefine its logic. See docs/design/polis-object.md.
"""

from .agent import Agent
from .embeddings import EmbeddingModel
from .graph import build_survey_graph, run_survey
from .llm import LLMClient, LLMConfig, LLMError
from .memory import MemoryRecord, MemoryStore, RetrievalConfig
from .persona import SEED_PERSONAS, Persona
from .questions import DST_QUESTION
from .survey import SurveyAnswer, SurveyQuestion

__version__ = "0.0.0"

__all__ = [
    "Agent",
    "DST_QUESTION",
    "EmbeddingModel",
    "LLMClient",
    "LLMConfig",
    "LLMError",
    "MemoryRecord",
    "MemoryStore",
    "Persona",
    "RetrievalConfig",
    "SEED_PERSONAS",
    "SurveyAnswer",
    "SurveyQuestion",
    "build_survey_graph",
    "run_survey",
]
