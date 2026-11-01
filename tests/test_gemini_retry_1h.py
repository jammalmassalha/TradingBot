"""Tests for the Gemini quota backoff and the 1H model-tier environment aliases."""

from __future__ import annotations

import pytest

from tradingagents.llm_clients.google_client import (
    QUOTA_RETRY_ATTEMPTS,
    QUOTA_RETRY_BASE_DELAY,
    QUOTA_RETRY_MAX_DELAY,
    _retry_delay_seconds,
    is_retryable_quota_error,
    retry_on_quota,
)

pytestmark = pytest.mark.unit


# Stand-ins named exactly as google.api_core spells them, since the matcher is
# name-based so it works whether or not that package is installed.
class ResourceExhausted(Exception):
    pass


class ServiceUnavailable(Exception):
    pass


class InvalidArgument(Exception):
    pass


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    """Keep the backoff from actually sleeping during tests."""
    monkeypatch.setattr("tradingagents.llm_clients.google_client.time.sleep", lambda _: None)


def test_the_mission_critical_errors_are_retryable():
    assert is_retryable_quota_error(ResourceExhausted("429 quota exceeded"))
    assert is_retryable_quota_error(ServiceUnavailable("503 overloaded"))


def test_a_permanent_error_is_not_retryable():
    """A bad request will never succeed; retrying only delays the real failure."""
    assert not is_retryable_quota_error(InvalidArgument("400 bad request"))
    assert not is_retryable_quota_error(ValueError("nope"))


def test_a_wrapped_quota_error_is_still_recognized():
    """LangChain re-raises provider errors inside its own classes."""
    try:
        raise ResourceExhausted("429")
    except ResourceExhausted as inner:
        wrapped = RuntimeError("langchain wrapper")
        wrapped.__cause__ = inner
        assert is_retryable_quota_error(wrapped)


def test_status_code_attribute_is_honoured():
    error = RuntimeError("throttled")
    error.status_code = 429
    assert is_retryable_quota_error(error)


def test_a_non_integer_code_is_not_mistaken_for_a_status():
    """gRPC errors carry an enum ``code``; it must not be coerced."""
    error = RuntimeError("something")
    error.code = "OK"
    assert not is_retryable_quota_error(error)


def test_retry_returns_the_value_once_the_call_succeeds():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ResourceExhausted("429")
        return "ok"

    assert retry_on_quota(flaky) == "ok"
    assert calls["n"] == 3


def test_retry_gives_up_after_the_budget_and_reraises():
    calls = {"n": 0}

    def always_throttled():
        calls["n"] += 1
        raise ResourceExhausted("429")

    with pytest.raises(ResourceExhausted):
        retry_on_quota(always_throttled)
    # The budget is retries, so the call is attempted once more than that.
    assert calls["n"] == QUOTA_RETRY_ATTEMPTS + 1


def test_the_default_budget_meets_the_five_retry_minimum():
    assert QUOTA_RETRY_ATTEMPTS >= 5
    # An exact literal: this is a module constant, not a computed float.
    assert QUOTA_RETRY_BASE_DELAY == 2.0


def test_a_permanent_error_is_raised_without_any_retry():
    calls = {"n": 0}

    def bad_request():
        calls["n"] += 1
        raise InvalidArgument("400")

    with pytest.raises(InvalidArgument):
        retry_on_quota(bad_request)
    assert calls["n"] == 1


def test_backoff_grows_exponentially_and_stays_bounded():
    error = ResourceExhausted("429")
    for attempt in range(1, 8):
        ceiling = min(QUOTA_RETRY_BASE_DELAY * 2 ** (attempt - 1), QUOTA_RETRY_MAX_DELAY)
        for _ in range(25):
            delay = _retry_delay_seconds(error, attempt)
            assert 0.0 <= delay <= ceiling


def test_jitter_actually_varies_the_delay():
    """Without jitter, concurrent agents would retry in lockstep and re-collide."""
    error = ResourceExhausted("429")
    delays = {_retry_delay_seconds(error, 4) for _ in range(50)}
    assert len(delays) > 1


def test_a_server_supplied_retry_after_wins():
    error = ResourceExhausted("429")
    error.retry_after = 7
    assert _retry_delay_seconds(error, 1) == pytest.approx(7.0)


def test_retry_after_is_still_bounded():
    error = ResourceExhausted("429")
    error.retry_after = 10_000
    assert _retry_delay_seconds(error, 1) == pytest.approx(QUOTA_RETRY_MAX_DELAY)


# --- environment aliases ----------------------------------------------------


def test_generate_override_retries_through_to_the_parent(monkeypatch):
    """Tool-calling agents reach the model via ``_generate``, not ``invoke``.

    Retrying only in ``invoke`` would leave every bind_tools / structured-output
    path unprotected, which is most of the graph.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    from tradingagents.llm_clients.google_client import NormalizedChatGoogleGenerativeAI

    calls = {"n": 0}

    def parent_generate(self, messages, stop=None, run_manager=None, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ResourceExhausted("429 quota exceeded")
        return "RESULT"

    monkeypatch.setattr(ChatGoogleGenerativeAI, "_generate", parent_generate, raising=False)
    # Bypass pydantic validation: the model is never actually contacted here.
    model = NormalizedChatGoogleGenerativeAI.__new__(NormalizedChatGoogleGenerativeAI)
    object.__setattr__(model, "__dict__", {"model": "gemini-1.5-flash"})

    assert model._generate(["hello"]) == "RESULT"
    assert calls["n"] == 3


def test_model_env_aliases_select_the_gemini_tiers(monkeypatch):
    """The mission's ``*_MODEL`` spellings must configure the real tier keys."""
    from tradingagents.default_config import build_default_config

    monkeypatch.setenv("TRADINGAGENTS_QUICK_THINK_MODEL", "gemini-1.5-flash")
    monkeypatch.setenv("TRADINGAGENTS_DEEP_THINK_MODEL", "gemini-1.5-pro")
    config = build_default_config()
    assert config["quick_think_llm"] == "gemini-1.5-flash"
    assert config["deep_think_llm"] == "gemini-1.5-pro"


def test_the_canonical_llm_spelling_wins_when_both_are_set(monkeypatch):
    from tradingagents.default_config import build_default_config

    monkeypatch.setenv("TRADINGAGENTS_QUICK_THINK_MODEL", "alias-model")
    monkeypatch.setenv("TRADINGAGENTS_QUICK_THINK_LLM", "canonical-model")
    assert build_default_config()["quick_think_llm"] == "canonical-model"


def test_deep_deliberation_defaults_off_and_is_settable(monkeypatch):
    """Existing runs must keep the cheap tier for the debate unless asked."""
    from tradingagents.default_config import build_default_config

    monkeypatch.delenv("TRADINGAGENTS_DEEP_DELIBERATION", raising=False)
    assert build_default_config()["deep_deliberation"] is False
    monkeypatch.setenv("TRADINGAGENTS_DEEP_DELIBERATION", "true")
    assert build_default_config()["deep_deliberation"] is True


def test_deep_deliberation_moves_the_debate_onto_the_deep_tier():
    """The flag must actually re-route the researchers and risk debators."""
    from tradingagents.graph.setup import GraphSetup

    quick, deep = object(), object()
    assert GraphSetup(quick, deep, None, 10)._deliberation_llm is quick
    assert GraphSetup(quick, deep, None, 10, deep_deliberation=True)._deliberation_llm is deep
