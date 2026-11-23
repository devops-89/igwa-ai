"""MOCKED tests: Gemini returns pages[] + metadata only; Python builds full_text."""
import asyncio
import json
import logging
from types import SimpleNamespace

import fitz
import pytest
from fastapi.testclient import TestClient

from config.settings import get_settings
from main import app
from services.ocr import gemini_client, groq_client, service
from services.ocr import gemini_key_manager as km
from services.ocr.schemas import GeminiOCRResult, OCRPage, build_full_text, ocr_data_from_dict

KEY = "AQ.Ab8RN6FAKEKEY000000000000000000000000001"
HI = "उपरोक्त विषय के संदर्भ में सूचित करते हुए कहना है कि आज दिनांक 26-02-2024 को"
EN = "FIR No. 5799028240086 registered at Sadar Police Station under sections 303(2), 115(2)."
MIXED = f"{HI} {EN}"
META = {"document_type": "FIR", "fir_number": "5799028240086"}


def pdf(pages):
    doc = fitz.open()
    for i in range(pages):
        doc.new_page().insert_text((72, 100), f"page {i + 1}", fontsize=11)
    return doc.tobytes()


class FakeFile:
    def __init__(self, data):
        self.filename, self.content_type, self._data = "a.pdf", "application/pdf", data

    async def read(self, n=-1):
        data, self._data = self._data, b""
        return data


class Gemini:
    def __init__(self, response):
        self.requests = 0

        async def generate_content(**kw):
            self.requests += 1
            return SimpleNamespace(text=json.dumps(response, ensure_ascii=False), candidates=[])

        self.client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))


@pytest.fixture
def env(monkeypatch):
    cfg = get_settings().model_copy(update=dict(gemini_model="gem", groq_model="grq"))
    monkeypatch.setattr(service, "get_settings", lambda: cfg)
    service.ocr_results.clear()
    groq = {"calls": 0, "pages": None}

    async def groq_stub(data, mime):
        groq["calls"] += 1
        n = fitz.open(stream=data, filetype="pdf").page_count
        texts = [f"GROQ PAGE {i + 1}" for i in range(n)]
        return {"pages": [{"page_number": i + 1, "text": t} for i, t in enumerate(texts)],
                "full_text": "\n\n".join(texts)}

    monkeypatch.setattr(groq_client, "extract_text", groq_stub)

    def install(response):
        g = Gemini(response)
        monkeypatch.setattr(km, "_manager", km.GeminiKeyManager([KEY], client_factory=lambda k, p: g.client))
        return g

    install.groq = groq
    return install


def ocr(n_pages):
    return asyncio.run(service.process_ocr(FakeFile(pdf(n_pages))))


# ---- the request no longer asks for full_text -----------------------------------------------------------


def test_schema_and_prompts_do_not_request_full_text():
    assert "full_text" not in GeminiOCRResult.model_fields
    assert "full_text" not in json.dumps(GeminiOCRResult.model_json_schema())
    for prompt in (gemini_client.OCR_PROMPT, gemini_client.COMBINED_PROMPT):
        assert '"full_text": "..."' not in prompt and "where full_text is" not in prompt
        assert 'Do NOT output a "full_text"' in prompt and "only once" in prompt
        assert '{"pages": [{"page_number": 1, "text": "..."}]}' in prompt
        assert "EVERY page" in prompt
    assert "metadata" in GeminiOCRResult.model_fields  # metadata is still requested in the same call


# ---- Python builds full_text ------------------------------------------------------------------------------


def test_build_full_text_orders_by_page_number_and_keeps_text_exactly():
    pages = [OCRPage(page_number=3, text=EN), OCRPage(page_number=1, text=HI), OCRPage(page_number=2, text=MIXED)]
    assert build_full_text(pages) == "\n\n".join([HI, MIXED, EN])
    assert build_full_text([]) == ""


def test_blank_pages_keep_their_separator_so_page_boundaries_survive():
    pages = [OCRPage(page_number=1, text="a"), OCRPage(page_number=2, text=""), OCRPage(page_number=3, text="c")]
    assert build_full_text(pages) == "a\n\n\n\nc"


def test_provider_sent_full_text_is_ignored_and_pages_sorted():
    raw = {"pages": [{"page_number": 2, "text": "B"}, {"page_number": 1, "text": "A"}], "full_text": "BOGUS DUPLICATE"}
    data = ocr_data_from_dict(raw)
    assert data.full_text == "A\n\nB" and [p.page_number for p in data.pages] == [1, 2]


def test_timings_are_reported():
    t = {}
    ocr_data_from_dict({"pages": [{"page_number": 1, "text": "x"}]}, t)
    assert set(t) == {"parse", "full_text"} and all(v >= 0 for v in t.values())


# ---- end to end through the service --------------------------------------------------------------------------


def test_response_has_same_structure_and_untouched_hindi_english(env):
    pages = [{"page_number": 1, "text": EN}, {"page_number": 2, "text": HI}, {"page_number": 3, "text": MIXED}]
    g = env({"pages": pages, "metadata": META})
    r = ocr(3)
    assert g.requests == 1  # exactly one Gemini request: no extra calls introduced
    assert [p.text for p in r.ocr_data.pages] == [EN, HI, MIXED]
    assert r.ocr_data.full_text == "\n\n".join([EN, HI, MIXED])
    assert r.metadata is not None and r.metadata.fir_number == "5799028240086"


def test_pages_returned_out_of_order_are_put_back_in_order(env):
    env({"pages": [{"page_number": 3, "text": "C"}, {"page_number": 1, "text": "A"}, {"page_number": 2, "text": "B"}],
         "metadata": META})
    r = ocr(3)
    assert [p.page_number for p in r.ocr_data.pages] == [1, 2, 3] and r.ocr_data.full_text == "A\n\nB\n\nC"


@pytest.mark.parametrize("returned", [
    [1, 2],  # skipped page 3
    [1, 2, 2],  # duplicated page
    [1, 2, 3, 4],  # invented page
    [2, 3, 4],  # wrong numbering
])
def test_incomplete_gemini_response_is_rejected_and_groq_covers_it(env, returned):
    g = env({"pages": [{"page_number": n, "text": f"G{n}"} for n in returned], "metadata": META})
    r = ocr(3)
    assert g.requests == 1 and env.groq["calls"] == 1
    assert r.ocr_provider == "groq" and r.ocr_data.full_text == "GROQ PAGE 1\n\nGROQ PAGE 2\n\nGROQ PAGE 3"


def test_complete_response_never_reaches_groq(env):
    env({"pages": [{"page_number": i, "text": f"G{i}"} for i in (1, 2, 3)], "metadata": META})
    r = ocr(3)
    assert env.groq["calls"] == 0 and r.ocr_provider == "gemini"


def test_groq_full_text_matches_the_python_join(env):
    env({"pages": []})  # Gemini yields nothing usable (0 pages != 2)
    r = ocr(2)
    assert r.ocr_provider == "groq" and r.ocr_data.full_text == "GROQ PAGE 1\n\nGROQ PAGE 2"


def test_uncountable_input_skips_the_guard_like_before(env, monkeypatch):
    # A corrupt PDF cannot be counted; behaviour is unchanged from before (provider output is used).
    env({"pages": [{"page_number": 1, "text": "only"}], "metadata": META})
    assert service._expected_page_count(b"%PDF-1.4 broken", "application/pdf") is None
    assert service._expected_page_count(b"\x89PNG", "image/png") == 1


# ---- logging ---------------------------------------------------------------------------------------------------------


def test_performance_logging_covers_every_stage(env, caplog):
    env({"pages": [{"page_number": 1, "text": EN}], "metadata": META})
    with caplog.at_level(logging.INFO):
        asyncio.run(service.process_upload(FakeFile(pdf(1)), __import__("fastapi").BackgroundTasks()))
    text = caplog.text
    assert "Gemini OCR timing: api=" in text and "json_parse=" in text
    assert "OCR timing: provider=gemini call=" in text and "parse_validate=" in text and "full_text_join=" in text
    assert "OCR API total response time=" in text
    assert EN not in text and "FIR No." not in text  # logs carry numbers only, never document text


def test_http_response_contract_is_unchanged(env):
    env({"pages": [{"page_number": 1, "text": EN}], "metadata": META})
    r = TestClient(app).post("/python/ocr/upload", files={"file": ("a.pdf", pdf(1), "application/pdf")})
    body = r.json()
    assert r.status_code == 200
    for key in ("success", "message", "document_id", "job_id", "status", "filename", "ocr_model", "ocr_provider",
                "ocr_data", "metadata", "metadata_model", "metadata_status", "metadata_error"):
        assert key in body
    assert set(body["ocr_data"]) == {"pages", "full_text"} and body["ocr_data"]["full_text"] == EN
    assert body["ocr_data"]["pages"] == [{"page_number": 1, "text": EN}]
    assert body["metadata_status"] == "completed" and len(body["metadata"]) == 16 and "application" in body["metadata"]
