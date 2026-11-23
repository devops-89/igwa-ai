"""MOCKED tests for ordered Gemini API key failover. No network, no real credentials."""
import asyncio
import json
import logging
import socket
from types import SimpleNamespace

import fitz
import pytest
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient

from config.settings import get_settings
from main import app
from services.ocr import gemini_client, groq_client, service
from services.ocr import gemini_key_manager as km
from services.ocr.gemini_client import GeminiMetadataError, GeminiOCRError
from services.ocr.groq_client import GroqMetadataError
from services.ocr.schemas import MetadataStatus, OCRMetadata

KEYS = [f"AIzaSyFAKEKEY{i}00000000000000000000000" for i in range(1, 6)]
OCR_JSON = '{"pages": [{"page_number": 1, "text": "GEMINI PAGE TEXT FIR 123/2026 Sadar"}]}'
META_JSON = '{"fir_number": "F1"}'


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


class ApiError(Exception):
    """Shaped like google.genai.errors.APIError (code, message, response.headers)."""

    def __init__(self, code, message="boom", headers=None):
        super().__init__(message)
        self.code, self.message = code, message
        self.response = SimpleNamespace(status_code=code, headers=headers or {})


class Fleet:
    """Five fake Gemini clients; behaviors[slot] is a list consumed in order (last one repeats)."""

    def __init__(self, behaviors=None, clock=None, policy=None, keys=KEYS):
        self.calls: list[int] = []
        self.behaviors = {i: list(v) for i, v in (behaviors or {}).items()}
        self.clock = clock or Clock()
        self.manager = km.GeminiKeyManager(list(keys), policy or km.Policy(), self._factory, self.clock)

    def _factory(self, key, policy):
        slot = KEYS.index(key) + 1 if key in KEYS else 0
        fleet = self

        async def generate_content(**kwargs):
            fleet.calls.append(slot)
            queue = fleet.behaviors.get(slot, ["ok"])
            item = queue.pop(0) if len(queue) > 1 else queue[0]
            if item == "ok":
                schema = getattr(kwargs.get("config"), "response_schema", None)
                return SimpleNamespace(text=META_JSON if schema is OCRMetadata else OCR_JSON, candidates=[])
            if item == "hang":
                await asyncio.sleep(30)
            if isinstance(item, BaseException):
                raise item
            return await item()

        return SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))

    def install(self, monkeypatch):
        monkeypatch.setattr(km, "_manager", self.manager)
        return self


def run(coro):
    return asyncio.run(coro)


def execute(fleet, budget=None):
    async def go():
        return await fleet.manager.execute(
            lambda c: c.aio.models.generate_content(model="m", contents=["x"]), purpose="test", budget=budget)
    return run(go())


def states(fleet):
    return [(s["slot"], s["state"]) for s in fleet.manager.snapshot()]


# ---- ordered failover -------------------------------------------------------------------------


def test_first_key_succeeds_and_stops():
    f = Fleet()
    execute(f)
    assert f.calls == [1] and f.manager.metrics["success_by_slot"] == {1: 1}


def test_first_fails_second_succeeds():
    f = Fleet({1: [ApiError(429, "rate", {"retry-after": "30"})]})
    execute(f)
    assert f.calls == [1, 2]
    assert states(f)[:2] == [(1, "cooldown"), (2, "available")]


def test_first_four_fail_fifth_succeeds_in_order():
    f = Fleet({1: [ApiError(401, "nope")], 2: [ApiError(429, "slow down")],
               3: [ApiError(402, "billing")], 4: [ApiError(403, "permission denied")]})
    execute(f)
    assert f.calls == [1, 2, 3, 4, 5]
    assert states(f) == [(1, "disabled"), (2, "cooldown"), (3, "cooldown"), (4, "cooldown"), (5, "available")]
    assert f.manager.metrics["success_by_slot"] == {5: 1} and f.manager.metrics["attempts"] == 5


def test_all_keys_fail_raises_exhausted_and_does_not_look_retryable():
    f = Fleet({i: [ApiError(429, "quota")] for i in range(1, 6)})
    with pytest.raises(km.GeminiKeysExhausted) as e:
        execute(f)
    assert f.calls == [1, 2, 3, 4, 5] and e.value.no_retry and e.value.__cause__ is None


def test_invalid_key_is_skipped_for_all_later_requests():
    f = Fleet({1: [ApiError(400, "API key not valid. Please pass a valid API key.")]})
    execute(f)
    execute(f)
    execute(f)
    assert f.calls == [1, 2, 2, 2]  # slot 1 disabled after first failure; never tried again
    assert states(f)[0] == (1, "disabled")


def test_unknown_credential_text_in_a_403_is_not_assumed_valid():
    f = Fleet({1: [ApiError(403, "Your API key was reported as leaked.")]})
    execute(f)
    assert states(f)[0] == (1, "disabled")


# ---- cooldowns ----------------------------------------------------------------------------------


def test_rate_limited_key_cools_down_honours_retry_after_then_recovers():
    f = Fleet({1: [ApiError(429, "slow", {"retry-after": "30"}), "ok"]})
    execute(f)
    assert f.calls == [1, 2]
    f.clock.advance(10)
    execute(f)
    assert f.calls == [1, 2, 2]  # still cooling down: not hammered
    f.clock.advance(25)  # 35s since the 429
    execute(f)
    assert f.calls == [1, 2, 2, 1]  # eligible again, tried first (configured order)


def test_daily_quota_uses_a_long_cooldown():
    f = Fleet({1: [ApiError(429, "Quota exceeded for metric ...PerDay... limit")]})
    execute(f)
    snap = f.manager.snapshot()[0]
    assert snap["state"] == "cooldown" and snap["reason"] == "quota_exhausted"
    assert snap["cooldown_remaining_seconds"] >= 3000


def test_cooldown_never_shortened_by_concurrent_failure():
    f = Fleet()
    slot = f.manager._slots[0]
    f.manager._penalize(slot, km.Failure(km.Kind.QUOTA, "quota_exhausted"))
    before = slot.cooldown_until
    f.manager._penalize(slot, km.Failure(km.Kind.RATE_LIMIT, "rate_limited", 429, 5.0))
    assert slot.cooldown_until == before


# ---- shared failures / bad requests ------------------------------------------------------------------


@pytest.mark.parametrize("error", [
    socket.gaierror("Temporary failure in name resolution"),
    ConnectionError("connection refused"),
    ApiError(503, "The model is overloaded"),
    ApiError(500, "internal"),
])
def test_shared_failure_stops_after_one_request_and_fast_fails_followers(error):
    f = Fleet({1: [error], 2: [error], 3: [error], 4: [error], 5: [error]})
    with pytest.raises(km.GeminiSharedFailure):
        execute(f)
    assert f.calls == [1]  # no five identical requests
    with pytest.raises(km.GeminiSharedFailure):
        execute(f)
    assert f.calls == [1]  # inside the short shared-failure window: not even one more request
    assert all(state == "available" for _, state in states(f))  # keys are not blamed
    f.clock.advance(11)
    f.behaviors[1] = ["ok"]
    execute(f)
    assert f.calls == [1, 1]


def test_timeout_is_shared_not_rotated():
    f = Fleet({1: ["hang"]}, policy=km.Policy(request_timeout_seconds=0.05))
    with pytest.raises(km.GeminiSharedFailure) as e:
        execute(f)
    assert f.calls == [1] and e.value.category == "timeout"


def test_overall_budget_is_respected():
    f = Fleet()
    with pytest.raises(km.GeminiSharedFailure) as e:
        execute(f, budget=0)
    assert f.calls == [] and e.value.category == "budget_exceeded"


@pytest.mark.parametrize("error,category", [
    (ApiError(400, "Invalid JSON payload received. Unknown name"), "bad_request"),
    (ApiError(404, "models/x is not found for API version v1beta"), "model_not_found"),
    (ApiError(413, "Request payload size exceeds the limit"), "payload_too_large"),
    (ValueError("weird"), "unknown_error"),
])
def test_request_errors_do_not_rotate_or_penalize_keys(error, category):
    f = Fleet({i: [error] for i in range(1, 6)})
    with pytest.raises(km.GeminiRequestFailure) as e:
        execute(f)
    assert f.calls == [1] and e.value.category == category
    assert all(state == "available" for _, state in states(f))
    assert e.value.__cause__ is error  # status still visible to callers


# ---- concurrency ---------------------------------------------------------------------------------------


def test_concurrent_requests_share_cooldown_without_retry_storm():
    f = Fleet({1: [ApiError(429, "slow", {"retry-after": "60"})]})

    async def scenario():
        mgr = f.manager
        call = lambda c: c.aio.models.generate_content(model="m", contents=["x"])  # noqa: E731
        await mgr.execute(call, purpose="first")  # trips slot 1 into cooldown, succeeds on slot 2
        results = await asyncio.gather(*(mgr.execute(call, purpose="t") for _ in range(25)))
        return results

    results = run(scenario())
    assert len(results) == 25
    assert f.calls.count(1) == 1  # slot 1 was hit exactly once; 25 followers skipped it
    assert f.manager.metrics["success_by_slot"] == {2: 26}


def test_concurrent_failures_are_counted_safely_from_threads():
    import threading
    f = Fleet()
    slot = f.manager._slots[0]
    threads = [threading.Thread(target=f.manager._penalize, args=(slot, km.Failure(km.Kind.RATE_LIMIT, "rate_limited", 429, 20.0)))
               for _ in range(50)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert f.manager.snapshot()[0]["state"] == "cooldown"


# ---- configuration -------------------------------------------------------------------------------------


def settings(**kw):
    base = dict(gemini_api_key="", gemini_api_keys="", **{f"gemini_api_key_{i}": "" for i in range(1, 11)})
    base.update(kw)
    return get_settings().model_copy(update=base)


def test_keys_loaded_from_comma_list_in_exact_order_with_dedupe_and_placeholders():
    s = settings(gemini_api_keys=f" {KEYS[2]} , {KEYS[0]},{KEYS[2]},your_gemini_api_key_here,,'{KEYS[4]}'")
    assert km.load_keys(s) == [KEYS[2], KEYS[0], KEYS[4]]


def test_keys_loaded_from_numbered_variables_in_numeric_order():
    s = settings(gemini_api_key_3=KEYS[2], gemini_api_key_1=KEYS[0], gemini_api_key_2=KEYS[1])
    assert km.load_keys(s) == KEYS[:3]


def test_legacy_single_key_still_works_and_placeholder_is_ignored():
    assert km.load_keys(settings(gemini_api_key=KEYS[0])) == [KEYS[0]]
    assert km.load_keys(settings(gemini_api_key="replace_with_your_key")) == []


def test_list_takes_precedence_over_legacy_key():
    assert km.load_keys(settings(gemini_api_keys=f"{KEYS[1]},{KEYS[2]}", gemini_api_key=KEYS[0])) == KEYS[1:3]


def test_no_keys_means_exhausted_immediately():
    f = Fleet(keys=[])
    with pytest.raises(km.GeminiKeysExhausted) as e:
        execute(f)
    assert e.value.category == "not_configured"


def test_sdk_client_is_built_without_hidden_retries_and_with_timeout(monkeypatch):
    captured = {}
    import google.genai as genai

    monkeypatch.setattr(genai, "Client", lambda **kw: captured.update(kw) or object())
    km._default_client_factory(KEYS[0], km.Policy(request_timeout_seconds=12))
    opts = captured["http_options"]
    assert opts.timeout == 12000 and opts.retry_options.attempts == 1  # failover is the only retry layer


# ---- OCR integration (real gemini_client + service, fake SDK clients) ---------------------------------------


HINDI_TEXT = "उपरोक्त विषय के संदर्भ में सूचित करते हुए कहना है कि आज दिनांक 26-02-2024 को FIR No. 5799028240086"


def scan_pdf(pages=1):
    doc = fitz.open()
    for _ in range(pages):
        doc.new_page().insert_text((72, 100), "scanned page", fontsize=11)
    return doc.tobytes()


class FakeFile:
    def __init__(self, data, name="a.pdf", ctype="application/pdf"):
        self.filename, self.content_type, self._data = name, ctype, data

    async def read(self, n=-1):
        data, self._data = self._data, b""
        return data


class PipelineEnv:
    def __init__(self, monkeypatch, **overrides):
        self.groq_calls = 0

        async def groq(data, mime):
            self.groq_calls += 1
            return {"pages": [{"page_number": 1, "text": "GROQ PAGE TEXT FIR 123/2026 Sadar"}],
                    "full_text": "GROQ PAGE TEXT FIR 123/2026 Sadar"}

        cfg = get_settings().model_copy(update=dict(gemini_model="gem", groq_model="grq", **overrides))
        monkeypatch.setattr(service, "get_settings", lambda: cfg)
        monkeypatch.setattr(groq_client, "extract_text", groq)
        service.ocr_results.clear()


@pytest.fixture
def pipe(monkeypatch):
    return PipelineEnv(monkeypatch)


def test_gemini_is_the_primary_ocr_engine(pipe, monkeypatch):
    f = Fleet().install(monkeypatch)
    r = run(service.process_ocr(FakeFile(scan_pdf())))
    assert f.calls == [1] and pipe.groq_calls == 0
    assert r.ocr_provider == "gemini" and r.ocr_model == "gem"
    assert r.ocr_data.full_text.startswith("GEMINI PAGE TEXT")


def test_ocr_uses_key2_after_key1_quota(pipe, monkeypatch):
    f = Fleet({1: [ApiError(429, "Quota exceeded PerDay")]}).install(monkeypatch)
    r = run(service.process_ocr(FakeFile(scan_pdf())))
    assert f.calls == [1, 2] and pipe.groq_calls == 0 and r.ocr_provider == "gemini"


def test_all_gemini_keys_fail_then_groq_is_used_and_counted(pipe, monkeypatch):
    f = Fleet({i: [ApiError(429, "quota")] for i in range(1, 6)}).install(monkeypatch)
    r = run(service.process_ocr(FakeFile(scan_pdf())))
    assert f.calls == [1, 2, 3, 4, 5]  # exactly one attempt per key: no retry multiplication
    assert pipe.groq_calls == 1 and r.ocr_provider == "groq" and r.ocr_model == "grq"
    assert f.manager.metrics["groq_fallbacks"] == 1


def test_shared_gemini_outage_goes_straight_to_groq_with_one_request(pipe, monkeypatch):
    f = Fleet({i: [socket.gaierror("dns")] for i in range(1, 6)}).install(monkeypatch)
    r = run(service.process_ocr(FakeFile(scan_pdf())))
    assert f.calls == [1] and pipe.groq_calls == 1 and r.ocr_provider == "groq"


def test_bad_request_does_not_rotate_keys_then_groq(pipe, monkeypatch):
    f = Fleet({i: [ApiError(400, "Invalid JSON payload")] for i in range(1, 6)}).install(monkeypatch)
    r = run(service.process_ocr(FakeFile(scan_pdf())))
    assert f.calls == [1] and r.ocr_provider == "groq"


def test_both_providers_failing_is_a_controlled_502(pipe, monkeypatch):
    from fastapi import HTTPException
    from services.ocr.groq_client import GroqOCRError

    async def groq_down(data, mime):
        raise GroqOCRError("down")

    monkeypatch.setattr(groq_client, "extract_text", groq_down)
    Fleet({i: [ApiError(401, "bad")] for i in range(1, 6)}).install(monkeypatch)
    with pytest.raises(HTTPException) as e:
        run(service.process_ocr(FakeFile(scan_pdf())))
    assert e.value.status_code == 502 and service.ocr_results == {}


def test_gemini_can_be_disabled(monkeypatch):
    env = PipelineEnv(monkeypatch, ocr_enable_gemini_fallback=False)
    f = Fleet().install(monkeypatch)
    r = run(service.process_ocr(FakeFile(scan_pdf())))
    assert f.calls == [] and env.groq_calls == 1 and r.ocr_provider == "groq"


def test_hindi_and_english_are_returned_untouched_and_prompt_forbids_translation(pipe, monkeypatch):
    seen = {}

    async def hindi():
        return SimpleNamespace(text=json.dumps({"pages": [{"page_number": 1, "text": HINDI_TEXT}]}, ensure_ascii=False))

    f = Fleet({1: [hindi]})
    inner = f._factory

    def factory(key, policy):
        client = inner(key, policy)
        original = client.aio.models.generate_content

        async def spy(**kw):
            seen["prompt"] = kw["contents"][1]
            return await original(**kw)

        client.aio.models.generate_content = spy
        return client

    f.manager._client_factory = factory
    f.install(monkeypatch)
    r = run(service.process_ocr(FakeFile(scan_pdf())))
    assert r.ocr_data.pages[0].text == HINDI_TEXT and r.ocr_data.full_text == HINDI_TEXT
    assert "NEVER translate" in seen["prompt"] and "NEVER transliterate" in seen["prompt"]


def test_every_ocr_prompt_forbids_translation_and_transliteration():
    for prompt in (gemini_client.OCR_PROMPT, groq_client.GROQ_OCR_PROMPT):
        assert "NEVER translate" in prompt and "NEVER transliterate" in prompt
        assert "Devanagari" in prompt and "Roman letters" in prompt
        assert "Copy digits exactly" in prompt and "line breaks" in prompt


# ---- metadata integration ---------------------------------------------------------------------------------


def test_metadata_uses_the_same_failover(monkeypatch):
    f = Fleet({1: [ApiError(429, "rate")], 2: [ApiError(401, "bad key")]}).install(monkeypatch)
    md = run(gemini_client.extract_metadata("FIR text"))
    assert md.fir_number == "F1" and f.calls == [1, 2, 3]
    assert states(f)[:3] == [(1, "cooldown"), (2, "disabled"), (3, "available")]


def test_metadata_background_falls_back_to_groq_when_every_gemini_key_fails(monkeypatch):
    f = Fleet({i: [ApiError(429, "quota")] for i in range(1, 6)}).install(monkeypatch)
    cfg = get_settings().model_copy(update=dict(gemini_model="gem", groq_model="grq"))
    monkeypatch.setattr(service, "get_settings", lambda: cfg)

    async def groq_meta(t):
        return OCRMetadata(fir_number="FROM-GROQ")

    monkeypatch.setattr(groq_client, "extract_metadata", groq_meta)
    service.ocr_results.clear()
    service.ocr_results["j1"] = service.OCRResultResponse(
        document_id="j1", job_id="j1", status=service.OCRStatus.COMPLETED, filename="a.pdf", ocr_model="x",
        metadata_status=MetadataStatus.PENDING)
    run(service.process_metadata_background("j1", "some ocr text"))
    out = service.get_metadata_status("j1")
    assert out.metadata_status == MetadataStatus.COMPLETED and out.metadata.fir_number == "FROM-GROQ"
    assert out.metadata_provider == "grq" and f.calls == [1, 2, 3, 4, 5]
    assert f.manager.metrics["groq_fallbacks"] == 1


def test_metadata_request_error_does_not_rotate_keys(monkeypatch):
    f = Fleet({i: [ApiError(400, "Invalid JSON payload")] for i in range(1, 6)}).install(monkeypatch)
    with pytest.raises(GeminiMetadataError):
        run(gemini_client.extract_metadata("FIR text"))
    assert f.calls == [1]


# ---- API compatibility and secrecy -------------------------------------------------------------------------------


def test_http_contract_unchanged_and_health_exposes_counts_only(pipe, monkeypatch):
    f = Fleet({1: [ApiError(401, "bad")]}).install(monkeypatch)

    async def meta(t):
        return OCRMetadata(fir_number="F9")

    monkeypatch.setattr(gemini_client, "extract_metadata", meta)
    client = TestClient(app)
    r = client.post("/python/ocr/upload", files={"file": ("a.pdf", scan_pdf(), "application/pdf")})
    body = r.json()
    for key in ("success", "message", "document_id", "job_id", "status", "filename", "ocr_model", "ocr_provider",
                "ocr_data", "metadata", "metadata_status", "metadata_error"):
        assert key in body
    assert body["ocr_provider"] == "gemini"
    assert body["metadata"] is None and body["metadata_status"] == "pending"
    assert client.get(f"/python/ocr/{body['job_id']}/metadata").json()["metadata_status"] == "completed"
    run(f.manager.execute(lambda c: c.aio.models.generate_content(model="m", contents=["x"]), purpose="t"))
    health = client.get("/health")
    assert health.json()["gemini_keys"] == {"configured": 5, "available": 4}
    assert "local_ocr" not in health.json()
    assert not any(k in health.text for k in KEYS) and "slot" not in health.text and "cooldown" not in health.text


def test_secrets_never_appear_in_logs_errors_or_diagnostics(monkeypatch, caplog):
    leaky = {
        1: [ApiError(401, f"API key not valid: {KEYS[0]}")],
        2: [ApiError(429, f"quota for key {KEYS[1]}", {"retry-after": "5"})],
        3: [ApiError(500, f"upstream failure with key={KEYS[2]} in url")],
    }
    f = Fleet(leaky).install(monkeypatch)
    with caplog.at_level(logging.DEBUG), pytest.raises(km.GeminiSharedFailure) as e:
        execute(f)
    with caplog.at_level(logging.DEBUG), pytest.raises(GeminiMetadataError) as me:
        f.manager.reset()
        f.behaviors = {1: [ApiError(400, f"Invalid payload for {KEYS[0]}")]}
        run(gemini_client.extract_metadata("FIR text"))
    blob = caplog.text + str(e.value) + str(me.value) + repr(f.manager.snapshot()) + repr(f.manager._slots) + repr(f.manager.metrics)
    for key in KEYS:
        assert key not in blob
    assert "Gemini credential slot 1" in caplog.text and "Gemini credential slot 2" in caplog.text


def test_redact_removes_configured_and_key_shaped_strings():
    text = f"failed for {KEYS[0]} and AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
    out = km.redact(text, [KEYS[0]])
    assert KEYS[0] not in out and "AIzaSyABCD" not in out and "[redacted-key]" in out


@pytest.mark.parametrize("error,kind", [
    (ApiError(401), km.Kind.KEY_INVALID), (ApiError(402), km.Kind.BILLING), (ApiError(403, "denied"), km.Kind.FORBIDDEN),
    (ApiError(403, "quota exceeded"), km.Kind.QUOTA), (ApiError(429, "slow"), km.Kind.RATE_LIMIT),
    (ApiError(503), km.Kind.SHARED), (ApiError(400, "bad"), km.Kind.REQUEST), (ApiError(404), km.Kind.REQUEST),
    (asyncio.TimeoutError(), km.Kind.SHARED), (ConnectionError("x"), km.Kind.SHARED), (RuntimeError("?"), km.Kind.REQUEST),
])
def test_error_classification_table(error, kind):
    assert km.classify_error(error).kind == kind
