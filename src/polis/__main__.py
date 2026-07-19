"""`python -m polis` — Phase 0 walking skeleton.

Survey the three seed personas on daylight saving time and print their answers.
"""
from __future__ import annotations

from .agent import Agent
from .graph import run_survey
from .llm import LLMClient
from .persona import SEED_PERSONAS
from .survey import SurveyQuestion

DST_QUESTION = SurveyQuestion(
    text="What should the United States do about daylight saving time?",
    options=[
        "Adopt permanent daylight saving time",
        "Adopt permanent standard time",
        "Keep switching twice a year",
        "No preference",
    ],
)


def main() -> None:
    client = LLMClient()
    agents = [Agent(p, client) for p in SEED_PERSONAS]
    results = run_survey(agents, DST_QUESTION)
    print(f"Q: {DST_QUESTION.text}\n")
    for r in results:
        print(f"[{r['agent_id']}] {r['choice']}")
        print(f"    {r['reason']}\n")


if __name__ == "__main__":
    main()
