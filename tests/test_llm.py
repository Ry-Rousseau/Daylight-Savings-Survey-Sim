"""Network-free LLM-client tests: the model-agnostic knobs (extra_body / token
budget) and the shared retry-or-skip helper. No endpoint is contacted — the client
is built with a dummy api_key so __init__ skips the env check, and no call is made."""
import pytest

from polis.llm import LLMClient, LLMConfig, LLMError, retry_on_llm_error


def _client(**cfg) -> LLMClient:
    # api_key bypasses the OPENROUTER_API_KEY env requirement; constructing the OpenAI
    # client is offline (no request until a call is issued, which these tests never do).
    return LLMClient(LLMConfig(**cfg), api_key="test-key")


def test_extra_body_sends_reasoning_toggle():
    assert _client(reasoning=False)._extra_body() == {"reasoning": {"enabled": False}}
    assert _client(reasoning=True)._extra_body() == {"reasoning": {"enabled": True}}


def test_extra_body_omitted_when_reasoning_none():
    # Model-agnostic: a non-Qwen model isn't handed a param it doesn't understand.
    assert _client(reasoning=None)._extra_body() is None


def test_structured_calls_have_their_own_token_budget():
    assert LLMConfig().structured_max_tokens == 512


def test_retry_returns_result_on_first_success():
    calls = []
    assert retry_on_llm_error(lambda: calls.append(1) or "ok", attempts=3) == "ok"
    assert len(calls) == 1  # no needless retries


def test_retry_returns_none_after_exhaustion(monkeypatch):
    monkeypatch.setattr("polis.llm.time.sleep", lambda *_: None)
    calls = []

    def fn():
        calls.append(1)
        raise LLMError("no valid json")

    assert retry_on_llm_error(fn, attempts=3) is None
    assert len(calls) == 3  # tried the full budget, then skipped (returned None)


def test_retry_recovers_on_a_later_attempt(monkeypatch):
    monkeypatch.setattr("polis.llm.time.sleep", lambda *_: None)
    seq = [LLMError("x"), LLMError("x"), "good"]

    def fn():
        v = seq.pop(0)
        if isinstance(v, Exception):
            raise v
        return v

    assert retry_on_llm_error(fn, attempts=3) == "good"


def test_retry_propagates_non_llm_errors():
    # Only LLMError is caught; a real bug must not be silently swallowed as a skip.
    with pytest.raises(ValueError):
        retry_on_llm_error(lambda: (_ for _ in ()).throw(ValueError("bug")))
