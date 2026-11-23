"""Ordered Gemini API key failover.

Keys are tried strictly in configured order, one request at a time. A failure is classified so
that only genuinely key-specific problems (invalid key, billing, quota/rate limit, permission)
move on to the next key; shared problems (network/DNS, timeouts, 5xx overload) and malformed
requests stop immediately, because another key would fail the same way.

Credentials never leave this module: diagnostics expose slot numbers only, and everything that
is logged or raised passes through redact().
"""
import asyncio
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, TypeVar

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)
T = TypeVar("T")

_KEY_PATTERN = re.compile(r"(?:AIza|AQ\.)[0-9A-Za-z_\-]{20,}")
_PLACEHOLDER_MARKERS = ("your_", "replace_with", "changeme", "xxxx")
_MAX_COOLDOWN_SECONDS = 3600.0
_MAX_NUMBERED_KEYS = 10


# ---- errors raised to callers (never contain credentials) ----------------------------------------


class GeminiFailure(Exception):
    """Base class. `category` is a safe failure label; `no_retry` tells callers not to retry."""

    no_retry = True

    def __init__(self, message: str, category: str = "other", attempts: int = 0):
        super().__init__(message)
        self.category = category
        self.attempts = attempts


class GeminiKeysExhausted(GeminiFailure):
    """No Gemini credential could serve the request (all failed, cooling down or disabled)."""


class GeminiSharedFailure(GeminiFailure):
    """Network/DNS/timeout/overload: rotating keys would not help."""


class GeminiRequestFailure(GeminiFailure):
    """Malformed request, unsupported model, payload too large...: same for every key."""


# ---- classification ---------------------------------------------------------------------------------


class Kind(str, Enum):
    KEY_INVALID = "key_invalid"  # permanent for this process
    FORBIDDEN = "forbidden"  # 403, cooldown
    BILLING = "billing"  # 402, cooldown
    RATE_LIMIT = "rate_limit"  # 429, short cooldown / Retry-After
    QUOTA = "quota_exhausted"  # 429 daily quota or 403 quota, long cooldown
    SHARED = "shared"  # network / timeout / 5xx: do not rotate
    REQUEST = "request"  # 4xx about the request itself: do not rotate


_KEY_KINDS = {Kind.KEY_INVALID, Kind.FORBIDDEN, Kind.BILLING, Kind.RATE_LIMIT, Kind.QUOTA}
_INVALID_KEY_MARKERS = (
    "api key not valid", "api_key_invalid", "api key expired", "invalid api key",
    "reported as leaked", "api key was reported",
)
_NETWORK_NAME_MARKERS = ("timeout", "connect", "dns", "gaierror", "network", "ssl", "disconnected",
                         "protocol", "remoteprotocol", "readerror", "closed")


@dataclass(frozen=True)
class Failure:
    kind: Kind
    category: str  # safe label for logs/metrics
    status: int | None = None
    retry_after: float | None = None


def redact(text: str, extra_keys: list[str] | None = None) -> str:
    """Remove anything that looks like (or is) a configured API key from text."""
    for key in extra_keys or []:
        if key:
            text = text.replace(key, "[redacted-key]")
    return _KEY_PATTERN.sub("[redacted-key]", text)


def _status_of(exc: BaseException) -> int | None:
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return getattr(getattr(exc, "response", None), "status_code", None)


def _retry_after(exc: BaseException, text: str) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    try:
        if headers is not None and headers.get("retry-after") is not None:
            return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        pass
    match = re.search(r"retry(?:delay|[ _-]?in)?\W{0,12}([0-9]+(?:\.[0-9]+)?)\s*s", text)
    return float(match.group(1)) if match else None


def classify_error(exc: BaseException) -> Failure:
    """Decide whether an exception is about the key, shared infrastructure, or the request."""
    text = " ".join(str(getattr(exc, "message", None) or exc).split()).lower()
    status = _status_of(exc)
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return Failure(Kind.SHARED, "timeout", status)
    if status is None:
        name = type(exc).__name__.lower()
        if isinstance(exc, OSError) or any(m in name for m in _NETWORK_NAME_MARKERS):
            return Failure(Kind.SHARED, "network", None)
        return Failure(Kind.REQUEST, "unknown_error", None)  # unknown: never rotate blindly
    if any(m in text for m in _INVALID_KEY_MARKERS) or status == 401:
        return Failure(Kind.KEY_INVALID, "authentication_failed", status)
    if status == 402 or (status in (400, 403) and "billing" in text):
        return Failure(Kind.BILLING, "billing_required", status)
    if status == 403:
        if "quota" in text:
            return Failure(Kind.QUOTA, "quota_exhausted", status, _retry_after(exc, text))
        return Failure(Kind.FORBIDDEN, "permission_denied", status)
    if status == 429:
        retry_after = _retry_after(exc, text)
        daily = "perday" in text.replace(" ", "") or "daily" in text
        if daily:
            return Failure(Kind.QUOTA, "quota_exhausted", status, retry_after)
        return Failure(Kind.RATE_LIMIT, "rate_limited", status, retry_after)
    if status == 408 or status >= 500:
        return Failure(Kind.SHARED, "timeout" if status in (408, 504) else "server_error", status)
    labels = {400: "bad_request", 404: "model_not_found", 413: "payload_too_large", 422: "unprocessable"}
    return Failure(Kind.REQUEST, labels.get(status, "request_error"), status)


# ---- key pool --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Policy:
    request_timeout_seconds: float = 90.0
    overall_budget_seconds: float = 110.0
    shared_failure_cooldown_seconds: float = 10.0
    rate_limit_cooldown_seconds: float = 60.0
    quota_cooldown_seconds: float = 3600.0
    billing_cooldown_seconds: float = 3600.0
    forbidden_cooldown_seconds: float = 3600.0

    @classmethod
    def from_settings(cls, s: Settings) -> "Policy":
        return cls(
            request_timeout_seconds=s.gemini_request_timeout_seconds,
            overall_budget_seconds=s.gemini_overall_budget_seconds,
            shared_failure_cooldown_seconds=s.gemini_shared_failure_cooldown_seconds,
            rate_limit_cooldown_seconds=s.gemini_rate_limit_cooldown_seconds,
            quota_cooldown_seconds=s.gemini_quota_cooldown_seconds,
            billing_cooldown_seconds=s.gemini_billing_cooldown_seconds,
            forbidden_cooldown_seconds=s.gemini_forbidden_cooldown_seconds,
        )


@dataclass(repr=False)
class _Slot:
    number: int  # 1-based, the only identifier ever logged
    key: str = field(repr=False)
    disabled_reason: str | None = None
    cooldown_until: float = 0.0
    reason: str | None = None
    client: Any = None

    def __repr__(self) -> str:  # never include the credential
        return f"<GeminiSlot {self.number}>"

    def available(self, now: float) -> bool:
        return self.disabled_reason is None and now >= self.cooldown_until


def _is_placeholder(value: str) -> bool:
    low = value.lower()
    return not value or any(m in low for m in _PLACEHOLDER_MARKERS)


def load_keys(s: Settings) -> list[str]:
    """Configured credentials in order. GEMINI_API_KEYS (comma list) wins, then GEMINI_API_KEY_1..10,
    then the legacy single GEMINI_API_KEY. Placeholders are dropped and duplicates ignored."""
    def clean(values: list[str]) -> list[str]:
        out: list[str] = []
        for v in values:
            v = v.strip().strip("\"'")
            if not _is_placeholder(v) and v not in out:
                out.append(v)
        return out

    listed = clean(s.gemini_api_keys.split(","))
    numbered = clean([getattr(s, f"gemini_api_key_{i}", "") for i in range(1, _MAX_NUMBERED_KEYS + 1)])
    legacy = clean([s.gemini_api_key])
    pool = listed or numbered or legacy
    if legacy and pool is not legacy and legacy[0] not in pool:
        logger.warning("GEMINI_API_KEY is ignored because GEMINI_API_KEYS / GEMINI_API_KEY_n are set")
    return pool


class GeminiKeyManager:
    def __init__(
        self,
        keys: list[str],
        policy: Policy | None = None,
        client_factory: Callable[[str, Policy], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._slots = [_Slot(i, k) for i, k in enumerate(keys, start=1)]
        self.policy = policy or Policy()
        self._client_factory = client_factory or _default_client_factory
        self._clock = clock
        self._lock = threading.Lock()
        self._shared_block_until = 0.0
        self._shared_block_category = "network"
        self.metrics: dict[str, Any] = {
            "attempts": 0, "successes": 0, "failures": {}, "success_by_slot": {}, "groq_fallbacks": 0,
        }

    # -- state ---------------------------------------------------------------------------------
    def redact(self, text: str) -> str:
        return redact(text, [s.key for s in self._slots])

    def counts(self) -> dict[str, int]:
        now = self._clock()
        with self._lock:
            return {"configured": len(self._slots), "available": sum(s.available(now) for s in self._slots)}

    def snapshot(self) -> list[dict[str, Any]]:
        """Safe diagnostics: slot numbers, states and reasons. Never credentials."""
        now = self._clock()
        with self._lock:
            return [{
                "slot": s.number,
                "state": "disabled" if s.disabled_reason else ("cooldown" if now < s.cooldown_until else "available"),
                "reason": s.disabled_reason or (s.reason if now < s.cooldown_until else None),
                "cooldown_remaining_seconds": max(0, round(s.cooldown_until - now, 1)),
            } for s in self._slots]

    def record_groq_fallback(self) -> None:
        with self._lock:
            self.metrics["groq_fallbacks"] += 1

    def reset(self) -> None:
        with self._lock:
            for s in self._slots:
                s.disabled_reason, s.cooldown_until, s.reason = None, 0.0, None
            self._shared_block_until = 0.0

    def _client(self, slot: _Slot) -> Any:
        with self._lock:
            if slot.client is None:
                slot.client = self._client_factory(slot.key, self.policy)
            return slot.client

    def _cooldown_for(self, failure: Failure) -> float:
        p = self.policy
        base = {Kind.RATE_LIMIT: p.rate_limit_cooldown_seconds, Kind.QUOTA: p.quota_cooldown_seconds,
                Kind.BILLING: p.billing_cooldown_seconds, Kind.FORBIDDEN: p.forbidden_cooldown_seconds}[failure.kind]
        if failure.kind == Kind.RATE_LIMIT and failure.retry_after is not None:
            seconds = failure.retry_after  # honour the provider's Retry-After
        elif failure.kind == Kind.QUOTA:
            seconds = max(failure.retry_after or 0.0, base)  # a daily quota does not clear in seconds
        else:
            seconds = base
        return min(max(seconds, 1.0), _MAX_COOLDOWN_SECONDS)

    def _penalize(self, slot: _Slot, failure: Failure) -> str:
        """Record a key-specific failure; returns a safe description for the log."""
        with self._lock:
            if failure.kind == Kind.KEY_INVALID:
                slot.disabled_reason = failure.category
                return "authentication failed (key disabled)"
            seconds = self._cooldown_for(failure)
            until = self._clock() + seconds
            if until > slot.cooldown_until:  # concurrent failures never shorten a cooldown
                slot.cooldown_until, slot.reason = until, failure.category
            return f"{failure.category.replace('_', ' ')} (cooldown {seconds:.0f}s)"

    def _count_failure(self, category: str) -> None:
        with self._lock:
            self.metrics["failures"][category] = self.metrics["failures"].get(category, 0) + 1

    # -- execution -------------------------------------------------------------------------------
    async def execute(
        self,
        call: Callable[[Any], Awaitable[T]],
        *,
        purpose: str,
        budget: float | None = None,
    ) -> T:
        """Run `call(client)` with ordered failover. Returns the first successful result."""
        started = self._clock()
        deadline = started + (budget if budget is not None else self.policy.overall_budget_seconds)
        if not self._slots:
            raise GeminiKeysExhausted("No Gemini API key is configured", "not_configured", 0)
        if self._clock() < self._shared_block_until:
            raise GeminiSharedFailure(
                f"Gemini recently failed with a shared error ({self._shared_block_category}); skipping",
                self._shared_block_category, 0)

        attempts, skipped = 0, 0
        for slot in self._slots:
            now = self._clock()
            if not slot.available(now):
                skipped += 1
                logger.info("Gemini credential slot %d: skipped (%s)", slot.number,
                            "disabled" if slot.disabled_reason else "cooling down")
                continue
            remaining = deadline - now
            if remaining <= 0:
                raise GeminiSharedFailure("Gemini time budget exhausted", "budget_exceeded", attempts)
            timeout = min(self.policy.request_timeout_seconds, remaining)
            attempts += 1
            with self._lock:
                self.metrics["attempts"] += 1
            try:
                result = await asyncio.wait_for(call(self._client(slot)), timeout)
            except asyncio.CancelledError:
                raise  # caller gave up: do not blame the key
            except Exception as exc:
                failure = classify_error(exc)
                self._count_failure(failure.category)
                if failure.kind in _KEY_KINDS:
                    logger.warning("Gemini credential slot %d: %s", slot.number, self._penalize(slot, failure))
                    continue  # next key immediately, no sleep
                detail = self.redact(" ".join(str(exc).split())[:160])
                if failure.kind == Kind.SHARED:
                    with self._lock:
                        self._shared_block_until = self._clock() + self.policy.shared_failure_cooldown_seconds
                        self._shared_block_category = failure.category
                    logger.error("Gemini credential slot %d: %s (shared failure, not rotating keys) %s",
                                 slot.number, failure.category, detail)
                    raise GeminiSharedFailure(f"Gemini shared failure: {failure.category}",
                                              failure.category, attempts) from exc
                logger.error("Gemini credential slot %d: %s (request error, not rotating keys) %s",
                             slot.number, failure.category, detail)
                raise GeminiRequestFailure(f"Gemini rejected the request: {failure.category}",
                                           failure.category, attempts) from exc
            with self._lock:
                self.metrics["successes"] += 1
                by_slot = self.metrics["success_by_slot"]
                by_slot[slot.number] = by_slot.get(slot.number, 0) + 1
            logger.info("Gemini credential slot %d: request succeeded (purpose=%s attempts=%d duration=%.2fs)",
                        slot.number, purpose, attempts, self._clock() - started)
            return result

        logger.error("Gemini: no credential could serve the request (purpose=%s attempts=%d skipped=%d)",
                     purpose, attempts, skipped)
        raise GeminiKeysExhausted("All Gemini credentials failed or are unavailable",
                                  "all_keys_unavailable", attempts)

    async def check_keys(self, model: str) -> list[dict[str, Any]]:
        """Probe every key once with a metadata-only call (no document content is sent)."""
        results = []
        for slot in self._slots:
            try:
                await asyncio.wait_for(self._client(slot).aio.models.get(model=model),
                                       self.policy.request_timeout_seconds)
                results.append({"slot": slot.number, "status": "ok"})
            except Exception as exc:
                failure = classify_error(exc)
                if failure.kind in _KEY_KINDS:
                    self._penalize(slot, failure)
                results.append({"slot": slot.number, "status": failure.category})
        return results


def _default_client_factory(api_key: str, policy: Policy) -> Any:
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=int(policy.request_timeout_seconds * 1000),
            # attempts=1: the SDK would otherwise retry connection errors itself (sleeping 1-10s),
            # multiplying with our failover. Failover is the only retry layer.
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


# ---- process-wide instance --------------------------------------------------------------------------------

_manager: GeminiKeyManager | None = None
_manager_lock = threading.Lock()


def get_manager() -> GeminiKeyManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            settings = get_settings()
            _manager = GeminiKeyManager(load_keys(settings), Policy.from_settings(settings))
            logger.info("Gemini key manager ready: %d credential slot(s)", len(_manager._slots))
        return _manager


def set_manager(manager: GeminiKeyManager | None) -> None:
    """Install (or clear with None) the process-wide manager. Used by tests and reconfiguration."""
    global _manager
    with _manager_lock:
        _manager = manager
