"""polis — a surveyable silicon sample of NYC opinion.

Source-first engine (the deliverable). Notebooks import down from here;
they never redefine its logic. See docs/design/polis-object.md.
"""

from .actions import Action, ActionDecision, ActionType, RetrievalProvenance
from .agent import Agent
from .drift import (
    DriftReading,
    capture_baseline,
    centroid_distance,
    cosine_distance,
    population_centroid,
    probe_drift,
)
from .embeddings import EmbeddingModel
from .feed import (
    FeedEvent,
    FeedProvider,
    NullFeedProvider,
    RagFeedProvider,
    ScriptedFeedProvider,
)
from .game_master import GameMaster
from .graph import build_survey_graph, run_survey
from .llm import LLMClient, LLMConfig, LLMError
from .memory import MemoryRecord, MemoryStore, RetrievalConfig
from .persona import SEED_PERSONAS, Persona
from .personas_nyc import NULL_PERSONA, NYC_CAST, THICK_PERSONAS, PersonaSeed, null_cast
from .questions import DST_QUESTION
from .runlog import RunLog
from .scheduler import Scheduler, SchedulerConfig
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
    "DriftReading",
    "DynamicsConfig",
    "EmbeddingModel",
    "NULL_PERSONA",
    "NYC_CAST",
    "PersonaSeed",
    "THICK_PERSONAS",
    "capture_baseline",
    "centroid_distance",
    "cosine_distance",
    "null_cast",
    "population_centroid",
    "probe_drift",
    "FeedEvent",
    "FeedProvider",
    "GameMaster",
    "NullFeedProvider",
    "RagFeedProvider",
    "ScriptedFeedProvider",
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
    "Scheduler",
    "SchedulerConfig",
    "Simulation",
    "SurveyAnswer",
    "SurveyQuestion",
    "WorldState",
    "WorldView",
    "build_survey_graph",
    "fully_connected",
    "run_survey",
]
