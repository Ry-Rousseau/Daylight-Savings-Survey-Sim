"""Live endpoint check — skipped unless OPENROUTER_API_KEY is available."""
import os

import pytest
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from polis.llm import DEFAULT_MODEL, LLMClient

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"), reason="no OPENROUTER_API_KEY"
)


def test_choose_returns_in_vocabulary_answer():
    client = LLMClient()
    options = [
        "Adopt permanent daylight saving time",
        "Adopt permanent standard time",
        "Keep switching twice a year",
        "No preference",
    ]
    res = client.choose(
        system="You are a New York City resident.",
        user="What should the US do about daylight saving time? Choose one: " + ", ".join(options),
        options=options,
    )
    assert res["choice"] in options
    assert res["model"] == DEFAULT_MODEL
    assert res["reason"]
