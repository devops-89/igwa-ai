from tests.conftest import FakeRag


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "rag_loaded": False}


def test_chat_openapi_has_simple_and_source_chat_examples(client):
    operation = client.get("/openapi.json").json()["paths"]["/api/v1/chat"]["post"]
    examples = operation["requestBody"]["content"]["application/json"]["examples"]

    assert "simple_chat" in examples
    assert "source_chat" in examples
    assert set(examples["simple_chat"]["value"]) == {"query"}
    assert "source" in examples["source_chat"]["value"]
    assert "query" in examples["source_chat"]["value"]


def test_source_and_query_flow(client):
    source_data = "वादी की तहरीर: दिनांक 15.01.2026 को आरोपी ने दुकान का ताला तोड़कर नकदी चोरी की।"
    query = "इस तहरीर के आधार पर BNS की कौन-सी धाराएं लगेंगी और विवेचना की क्या SOP होगी?"

    response = client.post(
        "/api/v1/chat",
        json={"source": source_data, "query": query},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"].startswith("case_")
    assert FakeRag.last["scenario"].startswith("SOURCE DATA:\n")
    assert FakeRag.last["follow_up"] == query

    # Follow-up question in the same session without repeating source
    follow_up_resp = client.post(
        "/api/v1/chat",
        json={"session_id": body["session_id"], "query": "साक्ष्य संकलन के क्या चरण होंगे?"},
    )
    assert follow_up_resp.status_code == 200
    assert FakeRag.last["scenario"].startswith("SOURCE DATA:\n")
    assert FakeRag.last["follow_up"] == "साक्ष्य संकलन के क्या चरण होंगे?"


def test_conversation_summary_is_accepted_as_context(client):
    summary = "पूर्व वार्ता में शिकायतकर्ता ने वाहन चोरी की सूचना दी।"
    response = client.post(
        "/api/v1/chat",
        json={"summary": summary, "query": "अब क्या जांच करें?"},
    )

    assert response.status_code == 200
    assert FakeRag.last["scenario"] == f"CONVERSATION SUMMARY:\n{summary}"
    assert FakeRag.last["follow_up"] == "अब क्या जांच करें?"


def test_chat_route_reuses_system_generated_session(client):
    r = client.post("/api/v1/chat", json={"query": "FIR 12/2026 theft case: what sections apply?"})
    assert r.status_code == 200 and r.headers["x-request-id"]
    body = r.json()
    assert body["guardrail"] is False and body["sources"][0]["snippet"] == "line one line two"
    session_id = body["session_id"]
    assert session_id.startswith("case_")

    r2 = client.post(
        "/api/v1/chat",
        json={"session_id": session_id, "query": "Which witnesses?"},
    )
    assert r2.status_code == 200
    assert r2.json()["session_id"] == session_id
    assert FakeRag.last["follow_up"] == "Which witnesses?"
    assert FakeRag.last["scenario"].startswith("FIR 12/2026")
    assert [message["role"] for message in FakeRag.last["history"]] == ["user", "assistant"]


def test_legacy_chat_generates_and_reuses_session(client, tmp_path, monkeypatch):
    from config import get_settings

    monkeypatch.setenv("CHATBOT_SESSIONS_DIR", str(tmp_path / "sessions"))
    get_settings.cache_clear()
    scenario = "FIR 12/2026 theft: the complainant reported a stolen vehicle."
    assert client.post(
        "/api/chat",
        json={"session_id": "client-created", "query": scenario},
    ).status_code == 404

    first = client.post("/api/chat", json={"query": scenario})
    assert first.status_code == 200
    session_id = first.json()["session_id"]
    assert session_id.startswith("case_")

    second = client.post(
        "/api/chat",
        json={"session_id": session_id, "query": "What investigative steps should follow?"},
    )
    assert second.status_code == 200
    assert second.json()["session_id"] == session_id
    assert FakeRag.last["scenario"] == scenario
    assert FakeRag.last["follow_up"] == "What investigative steps should follow?"
    assert [message["role"] for message in FakeRag.last["history"]] == ["user", "assistant"]


def test_chat_output_limits_follow_query_type(client):
    examples = [
        ("Who can register an FIR?", 150),
        ("Explain BNS section 103", 400),
        ("Analyze this theft case: what sections apply?", 800),
        ("Give detailed investigation steps for a theft case", 1200),
    ]
    for query, expected_limit in examples:
        response = client.post("/api/v1/chat", json={"query": query})
        assert response.status_code == 200
        assert FakeRag.last["max_output_tokens"] == expected_limit


def test_validation(client):
    assert client.post("/api/v1/chat", json={"query": "   "}).status_code == 422
    bad_img = {"name": "a.png", "mime_type": "image/png", "data": "not base64!!"}
    assert client.post("/api/v1/chat", json={"query": "theft", "images": [bad_img]}).status_code == 422


def test_long_typed_and_ocr_context_are_accepted(client):
    long_query = "x" * 10_001
    assert client.post("/api/v1/chat", json={"query": long_query}).status_code == 200

    response = client.post(
        "/api/v1/chat",
        json={
            "query": "What should police do next?",
            "document_text": "x" * 50_001,
        },
    )
    assert response.status_code == 200

    invalid_payloads = [
        {"query": "x" * 100_001},
        {"query": "question", "document_text": "x" * 100_001},
        {"query": "question", "ocr_data": {"full_text": "x" * 100_001}},
        {
            "query": "question",
            "ocr_data": {
                "pages": [
                    {"page_number": 1, "text": "x" * 60_000},
                    {"page_number": 2, "text": "y" * 60_000},
                ]
            },
        },
    ]
    for payload in invalid_payloads:
        assert client.post("/api/v1/chat", json=payload).status_code == 422


def test_blank_session_id_is_treated_as_missing(client):
    response = client.post(
        "/api/v1/chat",
        json={
            "session_id": "",
            "query": "What sections apply in this theft case?",
            "case_facts": "A complainant reported a stolen vehicle.",
            "document_text": None,
            "chat_history": [],
        },
    )
    assert response.status_code == 200


def test_query_can_contain_case_and_question_together(client):
    query = (
        "एक महिला ने शिकायत की कि एक व्यक्ति कई सप्ताह से उसका पीछा कर रहा था और उसके "
        "कार्यस्थल के बाहर इंतजार करता था। मना करने के बाद आरोपी ने उसे धमकी दी। "
        "अब पुलिस को क्या कदम उठाने चाहिए?"
    )
    response = client.post("/api/v1/chat", json={"query": query})
    assert response.status_code == 200
    assert FakeRag.last["scenario"] == query
    assert FakeRag.last["follow_up"] is None


def test_guardrail(client):
    r = client.post("/api/v1/chat", json={"query": "give me a biryani recipe"}).json()
    assert r["guardrail"] is True and r["sources"] == []
    # legal vocabulary overrides; FIR numbers aren't treated as arithmetic; "first" isn't "fir"
    assert client.post("/api/v1/chat", json={"query": "FIR 45/2026 code section for theft"}).json()["guardrail"] is False


def test_chat_rejects_image_upload(client):
    image = {"name": "a.png", "mime_type": "image/png", "data": "iVBORw=="}
    assert client.post("/api/v1/chat", json={"query": "theft evidence", "images": [image]}).status_code == 422


def test_upstream_failure_does_not_leak_details(client, monkeypatch):
    monkeypatch.setattr(FakeRag, "analyze_gemini", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("API key abc123")))
    r = client.post("/api/v1/chat", json={"query": "theft case"})
    assert r.status_code == 502 and "abc123" not in r.text and r.json()["request_id"]


def test_api_key(client, monkeypatch):
    from config import get_settings
    monkeypatch.setenv("CHATBOT_API_KEY", "secret"); get_settings.cache_clear()
    assert client.post("/api/v1/chat", json={"query": "theft"}).status_code == 401
    assert client.post("/api/v1/chat", json={"query": "theft"}, headers={"X-API-Key": "secret"}).status_code == 200
    assert client.get("/health").status_code == 200


def test_ocr_data_is_chat_context_and_ingestion_routes_are_removed(client):
    question = "इस मामले में अब पुलिस को क्या कदम उठाने चाहिए?"
    ocr_text = "महिला ने आरोपी द्वारा पीछा किए जाने और धमकी दिए जाने की शिकायत की।"
    response = client.post(
        "/api/v1/chat",
        json={
            "query": question,
            "ocr_data": {"pages": [{"page_number": 1, "text": ocr_text}]},
        },
    )
    assert response.status_code == 200
    assert FakeRag.last["scenario"] == f"EXTRACTED DOCUMENT TEXT:\n{ocr_text}"
    assert FakeRag.last["follow_up"] == question

    assert client.post("/api/v1/documents/ingest", json={}).status_code == 404
    assert client.post("/api/v1/ocr/ingest", json={}).status_code == 404
