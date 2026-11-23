"""MOCKED tests for the isFir document classification. No network, no real credentials."""
import asyncio
import json
import logging
from types import SimpleNamespace

import fitz
import pytest
from fastapi import BackgroundTasks
from fastapi.testclient import TestClient

from config.settings import get_settings
from main import app
from services.ocr import gemini_client, groq_client, service
from services.ocr import gemini_key_manager as km
from services.ocr.groq_client import GroqMetadataError
from services.ocr.schemas import GeminiOCRResult, MetadataStatus, OCRMetadata

KEY = "AQ.Ab8RN6FAKEKEY000000000000000000000000001"
PAGES = [{"page_number": 1, "text": "text of the document"}]


def meta(cls, **extra):
    base = {"document_type": "FIR", "fir_number": "0112/2025", "police_station": "नगर कोतवाली", "district": "सीतापुर",
            "legal_sections": ["303(2)"], "document_class": cls}
    base.update(extra)
    return base


class Gemini:
    """Fake SDK client. `combined` answers the OCR+metadata request; `text_only` the metadata fallback."""

    def __init__(self, combined=None, text_only=None):
        self.requests = []
        self.combined, self.text_only = combined, text_only

        async def generate_content(**kw):
            schema = getattr(kw["config"], "response_schema", None)
            self.requests.append((schema, kw["contents"][-1]))
            body = self.text_only if schema is OCRMetadata else self.combined
            if isinstance(body, Exception):
                raise body
            return SimpleNamespace(text=json.dumps(body, ensure_ascii=False), candidates=[])

        self.client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))


class FakeFile:
    def __init__(self):
        doc = fitz.open()
        doc.new_page().insert_text((72, 100), "x", fontsize=11)
        self.filename, self.content_type, self._data = "a.pdf", "application/pdf", doc.tobytes()

    async def read(self, n=-1):
        data, self._data = self._data, b""
        return data


@pytest.fixture
def env(monkeypatch):
    cfg = get_settings().model_copy(update=dict(gemini_model="gem", groq_model="grq"))
    monkeypatch.setattr(service, "get_settings", lambda: cfg)
    service.ocr_results.clear()
    state = {"groq_ocr": 0}

    async def groq_ocr(data, mime):
        state["groq_ocr"] += 1
        return {"pages": PAGES, "full_text": PAGES[0]["text"]}

    monkeypatch.setattr(groq_client, "extract_text", groq_ocr)

    def install(combined=None, text_only=None, keys=(KEY,)):
        g = Gemini(combined, text_only)
        monkeypatch.setattr(km, "_manager", km.GeminiKeyManager(list(keys), client_factory=lambda k, p: g.client))
        return g

    install.state = state
    return install


def upload():
    async def go():
        tasks = BackgroundTasks()
        resp = await service.process_upload(FakeFile(), tasks)
        return resp, tasks

    return asyncio.run(go())


# ---- classification outcomes through the same-call path -------------------------------------------------


@pytest.mark.parametrize("scenario,label,expected", [
    ("official Hindi FIR", "FIR", True),
    ("official English FIR", "FIR", True),
    ("Hindi police complaint application", "APPLICATION", False),
    ("English complaint application", "APPLICATION", False),
    ("application that quotes an FIR number", "APPLICATION", False),
    ("other document type", "OTHER", False),
    ("unclear / unsupported document", "UNCLEAR", False),
    ("lower-case label is tolerated", "fir", True),
])
def test_is_fir_from_the_single_gemini_request(env, scenario, label, expected):
    g = env(combined={"pages": PAGES, "metadata": meta(label)})
    resp, tasks = upload()
    assert resp.isFir is expected, scenario
    assert len(g.requests) == 1 and tasks.tasks == []  # no extra model request for classification
    assert resp.metadata_status == MetadataStatus.COMPLETED


def test_missing_or_garbage_classification_is_false_and_logged_as_unresolved(env, caplog):
    for bad in (None, "COMPLAINT!!", 7):
        env(combined={"pages": PAGES, "metadata": meta(bad)})
        caplog.clear()
        with caplog.at_level(logging.INFO):
            resp, _ = upload()
        assert resp.isFir is False and resp.metadata_status == MetadataStatus.COMPLETED
        assert "classification unresolved" in caplog.text  # distinct from a confirmed non-FIR


def test_confirmed_non_fir_is_not_logged_as_a_failure(env, caplog):
    env(combined={"pages": PAGES, "metadata": meta("APPLICATION")})
    with caplog.at_level(logging.INFO):
        resp, _ = upload()
    assert resp.isFir is False
    assert "classified as non-FIR (APPLICATION)" in caplog.text and "unresolved" not in caplog.text


def test_public_metadata_is_the_15_fields_plus_application(env):
    env(combined={"pages": PAGES, "metadata": meta("FIR")})
    resp, _ = upload()
    dumped = resp.metadata.model_dump()
    assert len(dumped) == 16 and "application" in dumped and "application_location" not in dumped and "document_class" not in dumped and "isFir" not in dumped


def test_request_schema_and_prompt_carry_the_classification_rules():
    assert "document_class" in json.dumps(GeminiOCRResult.model_json_schema())
    prompt = gemini_client.COMBINED_PROMPT
    for needle in ("document_class", "FIR, APPLICATION, OTHER, UNCLEAR", "प्रथम सूचना रिपोर्ट", "प्रार्थना पत्र",
                   "सेवा में", "do NOT make it an FIR", "if unsure choose UNCLEAR", "Under Section 154"):
        assert needle in prompt, needle
    assert "full_text" not in json.dumps(GeminiOCRResult.model_json_schema())  # earlier optimization intact


# ---- serialization: boolean, and immediately after metadata_error ------------------------------------------


def keys_after_metadata_error(obj):
    keys = list(obj)
    return keys[keys.index("metadata_error") + 1]


def test_http_responses_put_isfir_right_after_metadata_error_as_a_boolean(env):
    env(combined={"pages": PAGES, "metadata": meta("FIR")})
    client = TestClient(app)
    r = client.post("/python/ocr/upload", files={"file": ("a.pdf", FakeFile()._data, "application/pdf")})
    body = json.loads(r.text)  # json.loads keeps key order
    assert body["isFir"] is True and keys_after_metadata_error(body) == "isFir"
    job = body["job_id"]
    stored = json.loads(client.get(f"/python/ocr/{job}").text)
    assert stored["isFir"] is True and keys_after_metadata_error(stored) == "isFir"
    status = json.loads(client.get(f"/python/ocr/{job}/metadata").text)
    assert status["isFir"] is True and keys_after_metadata_error(status) == "isFir"
    assert '"isFir":true' in r.text.replace(" ", "")  # a JSON boolean, never the string "true"


def test_isfir_is_always_a_boolean_even_when_metadata_is_pending(env):
    env(combined={"pages": PAGES, "metadata": None}, text_only=meta("FIR"))
    resp, tasks = upload()
    assert resp.metadata_status == MetadataStatus.PENDING and resp.isFir is False and len(tasks.tasks) == 1
    assert json.loads(resp.model_dump_json())["isFir"] is False


# ---- background / fallback paths --------------------------------------------------------------------------------


def test_background_gemini_text_only_fallback_classifies_and_updates_every_endpoint(env):
    g = env(combined={"pages": PAGES, "metadata": None}, text_only=meta("FIR"))
    resp, tasks = upload()
    assert resp.isFir is False  # not known yet
    asyncio.run(tasks())
    assert [r[0] for r in g.requests] == [GeminiOCRResult, OCRMetadata]  # one extra request only because metadata was missing
    status = service.get_metadata_status(resp.job_id)
    assert status.isFir is True and status.metadata_status == MetadataStatus.COMPLETED
    assert service.get_ocr_result(resp.job_id).isFir is True


def test_groq_ocr_then_background_metadata_classifies_application_as_false(env):
    env(combined=None, text_only=meta("APPLICATION"), keys=())  # no Gemini key: Groq does the OCR
    resp, tasks = upload()
    assert resp.ocr_provider == "groq" and resp.isFir is False and env.state["groq_ocr"] == 1
    asyncio.run(tasks())
    assert service.get_metadata_status(resp.job_id).isFir is False
    assert service.get_metadata_status(resp.job_id).metadata_status == MetadataStatus.COMPLETED


def test_groq_ocr_then_background_gemini_classifies_fir_as_true(env):
    class Overloaded(Exception):
        code, message = 503, "overloaded"

    # Gemini OCR fails with a shared (5xx) error, so Groq does the OCR; the metadata then arrives later.
    g = env(combined=Overloaded(), text_only=meta("FIR"))
    resp, tasks = upload()
    assert resp.ocr_provider == "groq" and resp.isFir is False and env.state["groq_ocr"] == 1
    km._manager.reset()  # clear the short shared-failure window left by the OCR error
    asyncio.run(tasks())
    assert [r[0] for r in g.requests] == [GeminiOCRResult, OCRMetadata]
    assert service.get_metadata_status(resp.job_id).isFir is True


def test_groq_metadata_fallback_returns_classification(env, monkeypatch):
    env(combined=None, text_only=None, keys=())  # Gemini unavailable for both OCR and metadata

    async def groq_meta(text):
        return OCRMetadata(fir_number="0112/2025", document_class="FIR")

    monkeypatch.setattr(groq_client, "extract_metadata", groq_meta)
    resp, tasks = upload()
    asyncio.run(tasks())
    status = service.get_metadata_status(resp.job_id)
    assert status.metadata_provider == "grq" and status.isFir is True
    assert "document_class" not in status.metadata.model_dump()


def test_groq_without_classification_gives_false_and_logs_it(env, monkeypatch, caplog):
    env(combined=None, text_only=None, keys=())

    async def groq_meta(text):
        return OCRMetadata(fir_number="0112/2025")  # model forgot document_class

    monkeypatch.setattr(groq_client, "extract_metadata", groq_meta)
    resp, tasks = upload()
    with caplog.at_level(logging.INFO):
        asyncio.run(tasks())
    assert service.get_metadata_status(resp.job_id).isFir is False
    assert "classification unresolved (missing)" in caplog.text


def test_real_groq_client_parses_and_requests_document_class(monkeypatch):
    sent = {}

    async def create(**kw):
        sent.update(kw)
        content = '```json\n{"fir_number": "F1", "document_class": "FIR"}\n```'
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")])

    client = SimpleNamespace(with_options=lambda **k: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    monkeypatch.setattr(groq_client, "_get_client", lambda: client)
    monkeypatch.setattr(groq_client, "get_settings", lambda: SimpleNamespace(groq_model="grq"))
    md = asyncio.run(groq_client.extract_metadata("some text"))
    assert md.fir_number == "F1" and md.document_class == "FIR"
    user = sent["messages"][1]["content"]
    assert '"document_class": "FIR"|"APPLICATION"|"OTHER"|"UNCLEAR"' in user
    assert "DOCUMENT CLASSIFICATION" in sent["messages"][0]["content"]  # system prompt carries the rules


def test_failed_background_metadata_is_false_with_a_failure_log(env, monkeypatch, caplog):
    env(combined=None, text_only=None, keys=())

    async def groq_down(text):
        raise GroqMetadataError("down")

    monkeypatch.setattr(groq_client, "extract_metadata", groq_down)
    resp, tasks = upload()
    with caplog.at_level(logging.INFO):
        asyncio.run(tasks())
    status = service.get_metadata_status(resp.job_id)
    assert status.metadata_status == MetadataStatus.FAILED and status.isFir is False
    assert "classification failed" in caplog.text


def test_empty_ocr_text_skips_everything_and_is_false(env):
    env(combined={"pages": [{"page_number": 1, "text": ""}], "metadata": meta("FIR")})
    resp, tasks = upload()
    assert resp.metadata_status == MetadataStatus.SKIPPED and resp.isFir is False and tasks.tasks == []


def test_jobs_do_not_share_classification(env):
    env(combined={"pages": PAGES, "metadata": meta("FIR")})
    a, _ = upload()
    env(combined={"pages": PAGES, "metadata": meta("APPLICATION")})
    b, _ = upload()
    assert service.get_ocr_result(a.job_id).isFir is True and service.get_ocr_result(b.job_id).isFir is False
