"""polis — a surveyable silicon sample of NYC opinion.

Source-first engine (the deliverable). Notebooks import down from here;
they never redefine its logic. See docs/design/polis-object.md.
"""

from .actions import Action, ActionDecision, ActionType, RetrievalProvenance
from .agent import Agent
from .embeddings import EmbeddingModel
from .game_master import GameMaster
from .graph import build_survey_graph, run_survey
from .llm import LLMClient, LLMConfig, LLMError
from .memory import MemoryRecord, MemoryStore, RetrievalConfig
from .persona import SEED_PERSONAS, Persona
from .questions import DST_QUESTION
from .runlog import RunLog
from .simulation import DynamicsConfig, Population, Run, Simulation, fully_connected
from .survey import SurveyAnswer, SurveyQuestion
from .world import WorldState, WorldView

__version__ = "0.0.0"

__all__ = [
    "Action",
    "ActionDecision",
    "ActionType",
    "Agent",
    "DST_QUESTION",
    "DynamicsConfig",
    "EmbeddingModel",
    "GameMaster",
    "LLMClient",
    "LLMConfig",
    "LLMError",
    "MemoryRecord",
    "MemoryStore",
    "Persona",
    "Population",
    "RetrievalConfig",
    "RetrievalProvenance",
    "Run",
    "RunLog",
    "SEED_PERSONAS",
    "Simulation",
    "SurveyAnswer",
    "SurveyQuestion",
    "WorldState",
    "WorldView",
    "build_survey_graph",
    "fully_connected",
    "run_survey",
]
