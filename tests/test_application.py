"""MOCKED tests for metadata.application (located by the model, cut from pages[] by Python)."""
import asyncio
import json
import logging
from types import SimpleNamespace

import fitz
import pytest
from fastapi import BackgroundTasks

from config.settings import get_settings
from services.ocr import gemini_client, groq_client, service
from services.ocr import gemini_key_manager as km
from services.ocr.application import attach_application, build_marked_text, extract_application
from services.ocr.schemas import (
    ApplicationLocation,
    GeminiOCRResult,
    MetadataStatus,
    OCRMetadata,
    OCRPage,
    metadata_from_dict,
)

KEY = "AQ.Ab8RN6FAKEKEY000000000000000000000000001"
SECRET_APP = "सेवा में, थानाध्यक्ष महोदय। विषय: प्रार्थना पत्र SECRET-APPLICATION-BODY"
FIR_P1, FIR_P2, FIR_P3 = "FIR No. 0112/2025 official page one", "FIR page two narrative", "FIR page three seal"
APP_P4 = "सेवा में, थानाध्यक्ष महोदय। विषय: चोरी की शिकायत"
APP_P5 = "SECRET-APPLICATION-BODY  प्रार्थी: राम कुमार"
PAGES5 = [
    {"page_number": i, "text": t} for i, t in enumerate([FIR_P1, FIR_P2, FIR_P3, APP_P4, APP_P5], start=1)
]


def pg(*texts):
    return [OCRPage(page_number=i, text=t) for i, t in enumerate(texts, start=1)]


def loc(found=True, start=None, end=None, **kw):
    return ApplicationLocation(found=found, start_page=start, end_page=end, **kw)


def meta(cls="FIR", location=None):
    base = {"document_type": "FIR", "fir_number": "0112/2025", "document_class": cls}
    if location is not None:
        base["application_location"] = location
    return base


# ---- extract_application: pure Python ------------------------------------------------------------------------


def test_pages_4_to_5_only_and_in_order():
    out = extract_application(pg(*[p["text"] for p in PAGES5]), loc(start=4, end=5))
    assert out.model_dump() == {"found": True, "start_page": 4, "end_page": 5, "text": APP_P4 + "\n\n" + APP_P5}
    assert "FIR" not in out.text  # no FIR registration text leaks in


def test_application_in_the_middle_of_the_document():
    out = extract_application(pg("a", "b", "c", "d"), loc(start=2, end=3))
    assert out.text == "b\n\nc"


def test_standalone_single_page_and_multi_page():
    assert extract_application(pg("only"), loc(start=1, end=1)).text == "only"
    assert extract_application(pg("आवेदन 1", "आवेदन 2", "आवेदन 3"), loc(start=1, end=3)).text == "आवेदन 1\n\nआवेदन 2\n\nआवेदन 3"


@pytest.mark.parametrize("location", [None, loc(found=False)])
def test_absent_application_is_a_confirmed_not_found(location):
    assert extract_application(pg("a"), location).model_dump() == {
        "found": False, "start_page": None, "end_page": None, "text": None}


@pytest.mark.parametrize("bad", [
    loc(start=None, end=None), loc(start=0, end=1), loc(start=3, end=2), loc(start=1, end=9), loc(start=-1, end=1),
])
def test_invalid_or_missing_boundaries_raise(bad):
    with pytest.raises(ValueError):
        extract_application(pg("a", "b"), bad)


def test_bool_page_numbers_are_rejected():
    with pytest.raises(ValueError):
        extract_application([OCRPage(page_number=1, text="a")], ApplicationLocation.model_construct(
            found=True, start_page=True, end_page=True))


def test_application_starting_halfway_through_a_page():
    out = extract_application(pg("FIR narrative. सेवा में, श्रीमान जी " + "body", "rest of application"),
                              loc(start=1, end=2, start_text="सेवा में, श्रीमान जी"))
    assert out.text == "सेवा में, श्रीमान जी body\n\nrest of application"
    assert "FIR narrative" not in out.text


def test_application_ending_halfway_and_single_page_cut_both_sides():
    out = extract_application(pg("FIR text. Sir, I request action. Yours, Ram. Seal and stamp"),
                              loc(start=1, end=1, start_text="Sir, I request", end_text="Yours, Ram."))
    assert out.text == "Sir, I request action. Yours, Ram."
    out = extract_application(pg("app starts here ends now. FIR tail"), loc(start=1, end=1, end_text="ends now."))
    assert out.text == "app starts here ends now."


def test_quote_matching_tolerates_whitespace_differences():
    out = extract_application(pg("noise\nSir,   I  request\naction"), loc(start=1, end=1, start_text="Sir, I request"))
    assert out.text.startswith("Sir,   I  request")


def test_unfindable_boundary_never_returns_unrelated_text():
    out = extract_application(pg("FIR only text"), loc(start=1, end=1, start_text="not in the page"))
    assert out.found is True and out.text is None


# ---- attach_application ---------------------------------------------------------------------------------------


def test_missing_location_is_unresolved_not_absent(caplog):
    md = OCRMetadata(fir_number="1")
    with caplog.at_level(logging.INFO):
        assert attach_application(md, pg("a"), "j").application is None


def test_invalid_range_does_not_fail_metadata_and_is_unresolved():
    md = OCRMetadata(fir_number="1", application_location=loc(start=2, end=9))
    out = attach_application(md, pg("a", "b"), "j")
    assert out.application is None and out.fir_number == "1"


def test_logs_never_contain_application_text(caplog):
    md = OCRMetadata(application_location=loc(start=1, end=1))
    with caplog.at_level(logging.DEBUG):
        attach_application(md, pg("SECRET-APPLICATION-BODY"), "j")
    assert "SECRET" not in caplog.text and "Application extraction" in caplog.text


def test_marked_text_numbers_the_pages():
    assert build_marked_text(pg("a", "b")) == "[[PAGE 1]]\na\n\n[[PAGE 2]]\nb"


def test_internal_location_is_not_public_and_malformed_location_is_tolerated():
    md = OCRMetadata.model_validate({"fir_number": "1", "application_location": {"found": True, "start_page": 1,
                                                                                    "end_page": 1}})
    assert "application_location" not in md.model_dump()
    assert OCRMetadata.model_validate({"application_location": "garbage"}).application_location is None
    assert OCRMetadata.model_validate({"application_location": {"found": "maybe"}}).application_location is None


def test_application_is_not_in_the_schema_sent_to_the_model():
    for model in (GeminiOCRResult, OCRMetadata):
        schema = model.model_json_schema()
        assert "ApplicationResult" not in schema.get("$defs", {})
        props = [p for d in (schema, *schema.get("$defs", {}).values()) for p in d.get("properties", {})]
        assert "application" not in props and "application_location" in props


def test_top_level_location_is_merged_into_metadata():
    raw = {"pages": [], "metadata": meta("FIR"), "application_location": {"found": True, "start_page": 4, "end_page": 5}}
    assert metadata_from_dict(raw).application_location.start_page == 4


def test_prompts_ask_for_location_only():
    for prompt in (gemini_client.COMBINED_PROMPT, gemini_client.METADATA_PROMPT):
        assert "application_location" in prompt and "never output the application text" in prompt
    assert "TOP-LEVEL" in gemini_client.COMBINED_PROMPT and "[[PAGE 3]]" in gemini_client.METADATA_PROMPT
    assert "[[PAGE" not in gemini_client.COMBINED_PROMPT


# ---- API flow (one Gemini request; Groq / fallback background path) ----------------------------------------------


class Gemini:
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
    monkeypatch.setattr(service, "_expected_page_count", lambda payload, mime: None)

    def install(combined=None, text_only=None, groq_pages=None, groq_meta=None):
        g = Gemini(combined, text_only)
        monkeypatch.setattr(km, "_manager", km.GeminiKeyManager([KEY], client_factory=lambda k, p: g.client))

        async def groq_ocr(data, mime):
            return {"pages": groq_pages or PAGES5, "full_text": ""}

        monkeypatch.setattr(groq_client, "extract_text", groq_ocr)
        sent = []

        async def groq_metadata(text):
            sent.append(text)
            if isinstance(groq_meta, Exception):
                raise groq_meta
            return OCRMetadata.model_validate(groq_meta)

        monkeypatch.setattr(groq_client, "extract_metadata", groq_metadata)
        g.groq_sent = sent
        return g

    return install


def upload():
    async def go():
        tasks = BackgroundTasks()
        resp = await service.process_upload(FakeFile(), tasks)
        for t in tasks.tasks:
            await t()
        return resp

    return asyncio.run(go())


def test_fir_with_attached_application_same_call(env):
    g = env(combined={"pages": PAGES5, "metadata": meta("FIR"),
                      "application_location": {"found": True, "start_page": 4, "end_page": 5}})
    resp = upload()
    app = resp.metadata.application
    assert (app.found, app.start_page, app.end_page) == (True, 4, 5)
    assert app.text == APP_P4 + "\n\n" + APP_P5 and FIR_P3 not in app.text
    assert resp.isFir is True and resp.metadata_status == MetadataStatus.COMPLETED and resp.metadata_error is None
    assert len(g.requests) == 1  # location rides on the existing single Gemini request
    assert resp.ocr_data.full_text == "\n\n".join(p["text"] for p in PAGES5)  # OCR untouched
    body = json.loads(resp.model_dump_json())
    assert "application_location" not in body["metadata"] and body["metadata"]["fir_number"] == "0112/2025"


def test_fir_without_application(env):
    env(combined={"pages": PAGES5, "metadata": meta("FIR"), "application_location": {"found": False}})
    resp = upload()
    assert resp.isFir is True
    assert resp.metadata.application.model_dump() == {"found": False, "start_page": None, "end_page": None, "text": None}


def test_fir_that_only_mentions_an_application(env):
    env(combined={"pages": PAGES5[:3], "metadata": meta("FIR"),
                  "application_location": {"found": False, "start_page": None, "end_page": None}})
    assert upload().metadata.application.found is False


@pytest.mark.parametrize("text", ["सेवा में, श्रीमान जी, विषय: शिकायत", "To, The SHO. Subject: complaint"])
def test_standalone_application_hindi_and_english(env, text):
    env(combined={"pages": [{"page_number": 1, "text": text}], "metadata": meta("APPLICATION"),
                  "application_location": {"found": True, "start_page": 1, "end_page": 1}})
    resp = upload()
    assert resp.isFir is False and resp.metadata.application.found is True and resp.metadata.application.text == text


def test_application_starting_mid_page_through_the_api(env):
    pages = [{"page_number": 1, "text": "FIR registered. सेवा में, श्रीमान जी body"}, {"page_number": 2, "text": "end"}]
    env(combined={"pages": pages, "metadata": meta("FIR"), "application_location": {
        "found": True, "start_page": 1, "end_page": 2, "start_text": "सेवा में, श्रीमान जी"}})
    assert upload().metadata.application.text == "सेवा में, श्रीमान जी body\n\nend"


def test_invalid_location_from_gemini_keeps_metadata_and_marks_application_unresolved(env):
    env(combined={"pages": PAGES5, "metadata": meta("FIR"),
                  "application_location": {"found": True, "start_page": 4, "end_page": 99}})
    resp = upload()
    assert resp.metadata_status == MetadataStatus.COMPLETED and resp.isFir is True
    assert resp.metadata.application is None  # unavailable, not a confirmed absence


def test_malformed_top_level_location_never_discards_ocr(env):
    g = env(combined={"pages": PAGES5, "metadata": meta("FIR"), "application_location": "oops"})
    resp = upload()
    assert resp.ocr_provider == "gemini" and len(resp.ocr_data.pages) == 5 and len(g.requests) == 1
    assert resp.metadata.application is None


def test_groq_ocr_fallback_uses_the_background_metadata_request_with_page_markers(env):
    loc_json = {"found": True, "start_page": 4, "end_page": 5}
    g = env(combined=RuntimeError("down"), groq_meta=meta("FIR", loc_json))
    resp = upload()
    assert resp.ocr_provider == "groq" and resp.metadata_status == MetadataStatus.PENDING
    assert resp.metadata is None  # unavailable until the background task finishes
    # (the task ran in upload(); read the stored final state)
    status = service.get_metadata_status(resp.job_id)
    assert status.metadata_status == MetadataStatus.COMPLETED and status.isFir is True
    assert status.metadata.application.text == APP_P4 + "\n\n" + APP_P5
    assert len(g.groq_sent) == 1 and "[[PAGE 4]]" in g.groq_sent[0]  # one metadata request, with page markers
    assert service.get_ocr_result(resp.job_id).ocr_data.full_text == "\n\n".join(p["text"] for p in PAGES5)


def test_background_gemini_metadata_request_carries_markers_and_location(env):
    g = env(combined={"pages": PAGES5, "metadata": None},  # no usable same-call metadata
            text_only=meta("APPLICATION", {"found": True, "start_page": 1, "end_page": 2}),
            groq_pages=None)
    resp = upload()
    status = service.get_metadata_status(resp.job_id)
    assert status.isFir is False and status.metadata.application.text == FIR_P1 + "\n\n" + FIR_P2
    meta_requests = [c for s, c in g.requests if s is OCRMetadata]
    assert len(meta_requests) == 1 and "[[PAGE 1]]" in meta_requests[0]


def test_failed_background_metadata_leaves_application_unavailable(env):
    env(combined=RuntimeError("down"), groq_meta=RuntimeError("down too"))
    resp = upload()
    status = service.get_metadata_status(resp.job_id)
    assert status.metadata_status == MetadataStatus.FAILED and status.metadata is None and status.isFir is False


def test_timing_logs_are_emitted_without_application_text(env, caplog):
    env(combined={"pages": PAGES5, "metadata": meta("FIR"),
                  "application_location": {"found": True, "start_page": 4, "end_page": 5}})
    with caplog.at_level(logging.INFO):
        upload()
    assert "Application extraction: found=True pages=4-5" in caplog.text
    assert "OCR timing" in caplog.text and "OCR API total response time" in caplog.text
    assert "SECRET-APPLICATION-BODY" not in caplog.text
