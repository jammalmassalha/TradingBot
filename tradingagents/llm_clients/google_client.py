import logging
import random
import re
import time
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model

logger = logging.getLogger(__name__)

_GEMINI_VERSION = re.compile(r"^gemini-(\d+)\.(\d+)")

# Outer retry budget for Gemini quota/availability errors, on top of whatever
# the SDK itself does. A multi-agent run issues dozens of Gemini calls in a
# burst (four analysts, then debate rounds), which is exactly the shape that
# trips per-minute quota, and a 1H loop that aborts on one 429 misses the bar.
QUOTA_RETRY_ATTEMPTS = 5
QUOTA_RETRY_BASE_DELAY = 2.0
# Beyond this a 1H run would sleep past its own bar, so it is better to fail.
QUOTA_RETRY_MAX_DELAY = 60.0

# Transient Gemini failures, named rather than imported: google.api_core is an
# optional transitive dependency, and matching on class name keeps this working
# whether the error arrives as google.api_core.exceptions.ResourceExhausted,
# its google.genai equivalent, or a LangChain wrapper around either.
_RETRYABLE_ERROR_NAMES = frozenset(
    {
        "ResourceExhausted",      # HTTP 429, quota exhausted
        "TooManyRequests",        # HTTP 429, raw transport
        "ServiceUnavailable",     # HTTP 503
        "DeadlineExceeded",       # HTTP 504
        "InternalServerError",    # HTTP 500
        "Aborted",
    }
)
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 503, 504})


def _retryable_status(error: BaseException) -> bool:
    """Whether the error carries an HTTP status Gemini throttling uses.

    google.api_core puts it on ``code``; other layers use ``status_code``. Only
    an int is trusted — ``code`` is an enum or string on some gRPC errors, and
    coercing those would make unrelated failures look retryable.
    """
    for attribute in ("code", "status_code"):
        value = getattr(error, attribute, None)
        if isinstance(value, int) and value in _RETRYABLE_STATUS_CODES:
            return True
    return False


def is_retryable_quota_error(error: BaseException) -> bool:
    """Whether a Gemini call failed for a reason that may succeed if retried.

    Checks the exception's own type name and that of every cause in its chain:
    LangChain frequently re-raises a provider error wrapped in its own class,
    which would otherwise hide a 429 behind a generic exception.
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ in _RETRYABLE_ERROR_NAMES or _retryable_status(current):
            return True
        current = current.__cause__ or current.__context__
    return False


def _retry_delay_seconds(error: BaseException, attempt: int) -> float:
    """Seconds to wait before retry ``attempt`` (1-based), with full jitter.

    Exponential backoff spreads a burst of agents that all hit the same
    per-minute quota; without jitter they would back off in lockstep and
    collide again on every wave. A server-supplied ``retry_after`` wins, since
    it states when the quota window actually reopens.
    """
    retry_after = getattr(error, "retry_after", None)
    if isinstance(retry_after, (int, float)) and retry_after > 0:
        return min(float(retry_after), QUOTA_RETRY_MAX_DELAY)

    ceiling = min(QUOTA_RETRY_BASE_DELAY * (2 ** (attempt - 1)), QUOTA_RETRY_MAX_DELAY)
    # Full jitter: a uniform draw over the whole window, keeping the mean delay
    # growing exponentially while removing the synchronized retry spike.
    return random.uniform(0.0, ceiling)


def retry_on_quota(call, *, attempts: int = QUOTA_RETRY_ATTEMPTS, label: str = "Gemini call"):
    """Run ``call``, retrying transient Gemini quota/availability failures.

    ``attempts`` is the number of *retries*, so the call is made at most
    ``attempts + 1`` times. Non-retryable errors propagate immediately: a bad
    API key or malformed request will never succeed, and sleeping on it only
    delays the real failure.
    """
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 2):
        try:
            return call()
        except Exception as exc:
            if not is_retryable_quota_error(exc) or attempt > attempts:
                raise
            last_error = exc
            delay = _retry_delay_seconds(exc, attempt)
            logger.warning(
                "%s hit a transient error (%s: %s); retry %d/%d in %.1fs",
                label, type(exc).__name__, exc, attempt, attempts, delay,
            )
            time.sleep(delay)
    # Unreachable: the final attempt either returns or re-raises above.
    raise RuntimeError(f"{label} exhausted its retry budget") from last_error


def _accepts_minimal_thinking(model: str) -> bool:
    """Whether ``thinking_level="minimal"`` is accepted: numbered Flash models
    before 3.8. Pro, 3.8+ and version-less aliases (which move between
    generations) are treated as rejecting it."""
    model_lc = model.lower()
    match = _GEMINI_VERSION.match(model_lc)
    return bool(match) and "pro" not in model_lc and (
        (int(match.group(1)), int(match.group(2))) < (3, 8)
    )


class NormalizedChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """ChatGoogleGenerativeAI with normalized content output and quota backoff.

    Gemini 3 models return content as list of typed blocks.
    This normalizes to string for consistent downstream handling.

    ``_generate`` is the single call site every path reaches — ``invoke``,
    ``bind_tools`` and ``with_structured_output`` all funnel through it — so
    retrying here covers tool-calling agents, not just direct invocations.
    """

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return retry_on_quota(
            lambda: super(NormalizedChatGoogleGenerativeAI, self)._generate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ),
            label=f"Gemini {self.model}",
        )

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))


# The read timeout the OpenAI and Anthropic SDKs use by default.
REQUEST_TIMEOUT_SECONDS = 600


class GoogleClient(BaseLLMClient):
    """Client for Google Gemini models."""

    def __init__(self, model: str, base_url: str | None = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatGoogleGenerativeAI instance."""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}

        if self.base_url:
            llm_kwargs["base_url"] = self.base_url

        for key in ("timeout", "max_retries", "temperature", "max_output_tokens",
                    "callbacks", "http_client", "http_async_client"):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]
        # Without a timeout a stalled Gemini call waits forever; the OpenAI and
        # Anthropic SDKs give up after the same 600 seconds (#1417).
        llm_kwargs.setdefault("timeout", REQUEST_TIMEOUT_SECONDS)

        # Unified api_key maps to provider-specific google_api_key
        google_api_key = self.kwargs.get("api_key") or self.kwargs.get("google_api_key")
        if google_api_key:
            llm_kwargs["google_api_key"] = google_api_key

        # Gemini 3.x takes the string ``thinking_level`` (the integer
        # ``thinking_budget`` was for the now-retired 2.5 line). Pro, Gemini
        # 3.8+ and the -latest aliases reject "minimal" with a 400; "low" is
        # accepted everywhere, so it is the fallback.
        thinking_level = self.kwargs.get("thinking_level")
        if thinking_level:
            if thinking_level == "minimal" and not _accepts_minimal_thinking(self.model):
                thinking_level = "low"
            llm_kwargs["thinking_level"] = thinking_level

        return NormalizedChatGoogleGenerativeAI(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Google."""
        return validate_model("google", self.model)
