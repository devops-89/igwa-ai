import importlib
import sys
from types import SimpleNamespace

import pytest


def _load_rag_with_stubbed_embeddings(monkeypatch):
    class FakeEmbeddingModel:
        def get_sentence_embedding_dimension(self):
            return 3

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(
            SentenceTransformer=lambda *args, **kwargs: FakeEmbeddingModel()
        ),
    )
    sys.modules.pop("rag", None)
    return importlib.import_module("rag")


def test_rag_import_supports_groq_only_configuration(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "dotenv",
        SimpleNamespace(load_dotenv=lambda *args, **kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(
            SentenceTransformer=lambda *args, **kwargs: SimpleNamespace(
                get_sentence_embedding_dimension=lambda: 3
            )
        ),
    )
    sys.modules.pop("rag", None)

    rag = importlib.import_module("rag")

    assert rag.GEMINI_API_KEYS == []
    assert rag.gemini_client is None


def test_groq_token_limiter_uses_provider_limits_and_reset_headers(monkeypatch):
    rag = _load_rag_with_stubbed_embeddings(monkeypatch)
    limiter = rag._GroqTokenRateLimiter(tokens_per_minute=20_000)

    limiter.observe_headers(
        {
            "x-ratelimit-limit-tokens": "5000",
            "x-ratelimit-remaining-tokens": "4200",
            "x-ratelimit-reset-tokens": "2m30.5s",
        }
    )

    assert limiter._effective_limit == 5000
    assert limiter._server_remaining == 4200
    assert limiter._server_reset_at > rag.time.monotonic() + 140

    limiter.acquire(1000)

    assert limiter._server_remaining == 3200
    with pytest.raises(RuntimeError, match="exceeds the configured TPM allowance"):
        limiter.acquire(5001)


def test_groq_token_estimate_accounts_for_multibyte_text(monkeypatch):
    rag = _load_rag_with_stubbed_embeddings(monkeypatch)

    assert rag._estimate_tokens("कानूनी") > rag._estimate_tokens("legal")


def test_groq_estimate_fits_server_reported_8000_tpm(monkeypatch):
    rag = _load_rag_with_stubbed_embeddings(monkeypatch)
    limiter = rag._GroqTokenRateLimiter()
    limiter.observe_headers(
        {
            "x-ratelimit-limit-tokens": "8000",
            "x-ratelimit-remaining-tokens": "8000",
            "x-ratelimit-reset-tokens": "60s",
        }
    )
    estimated_request = (
        rag._estimate_tokens(rag.SYSTEM_PROMPT)
        + rag._estimate_tokens("case " + "x" * 24_900)
        + 1200
    )

    limiter.acquire(estimated_request)

    assert estimated_request <= 8000


def test_groq_request_reserves_tokens_and_observes_rate_headers(monkeypatch, tmp_path):
    rag = _load_rag_with_stubbed_embeddings(monkeypatch)

    class TrackingLimiter:
        def __init__(self):
            self.reserved = []
            self.headers = []

        def acquire(self, tokens):
            self.reserved.append(tokens)

        def observe_headers(self, headers):
            self.headers.append(headers)

    class GroqResponse:
        headers = {
            "x-ratelimit-limit-tokens": "10000",
            "x-ratelimit-remaining-tokens": "9500",
            "x-ratelimit-reset-tokens": "10s",
        }

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "answer"}}]}

    limiter = TrackingLimiter()
    monkeypatch.setattr(rag, "_groq_token_rate_limiter", limiter)
    monkeypatch.setattr(rag, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(rag.requests, "post", lambda *args, **kwargs: GroqResponse())

    provider = object.__new__(rag.LocalFAISSRag)
    provider.data_dir = tmp_path
    assert provider._query_groq("case prompt", max_output_tokens=400) == "answer"

    assert limiter.reserved == [
        rag._estimate_tokens(rag.SYSTEM_PROMPT)
        + rag._estimate_tokens("case prompt")
        + 400
    ]
    assert limiter.headers == [GroqResponse.headers]


def test_groq_fallback_uses_requested_output_limit(monkeypatch, tmp_path):
    rag = _load_rag_with_stubbed_embeddings(monkeypatch)

    class FailingGeminiModels:
        def generate_content(self, **kwargs):
            raise TimeoutError("simulated Gemini timeout")

    class GroqResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "fallback answer"}}]}

    request_payload = {}

    def fake_post(_url, **kwargs):
        request_payload.update(kwargs["json"])
        return GroqResponse()

    _failing_client = SimpleNamespace(models=FailingGeminiModels())
    monkeypatch.setattr(rag, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(rag, "gemini_client", _failing_client)
    monkeypatch.setattr(rag, "_gemini_clients", [_failing_client])
    monkeypatch.setattr(rag.LocalFAISSRag, "search", lambda *args, **kwargs: [])
    monkeypatch.setattr(rag.requests, "post", fake_post)

    provider = object.__new__(rag.LocalFAISSRag)
    provider.data_dir = tmp_path
    answer, sources, model = provider.analyze_gemini(
        "case scenario",
        max_output_tokens=400,
    )

    assert answer == "fallback answer"
    assert sources == []
    assert model.startswith("Groq")
    assert request_payload["max_completion_tokens"] == 400
    assert request_payload["reasoning_effort"] == "medium"
    assert request_payload["include_reasoning"] is False


def test_groq_fallback_retries_empty_response_with_larger_budget(monkeypatch, tmp_path):
    rag = _load_rag_with_stubbed_embeddings(monkeypatch)

    class FailingGeminiModels:
        def generate_content(self, **kwargs):
            raise TimeoutError("simulated Gemini timeout")

    class GroqResponse:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {"content": self.content},
                        "finish_reason": "length" if not self.content else "stop",
                    }
                ]
            }

    request_payloads = []
    responses = [GroqResponse(""), GroqResponse("fallback answer after retry")]

    def fake_post(_url, **kwargs):
        request_payloads.append(kwargs["json"])
        return responses.pop(0)

    _failing_client2 = SimpleNamespace(models=FailingGeminiModels())
    monkeypatch.setattr(rag, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(rag, "gemini_client", _failing_client2)
    monkeypatch.setattr(rag, "_gemini_clients", [_failing_client2])
    monkeypatch.setattr(rag.LocalFAISSRag, "search", lambda *args, **kwargs: [])
    monkeypatch.setattr(rag.requests, "post", fake_post)

    provider = object.__new__(rag.LocalFAISSRag)
    provider.data_dir = tmp_path
    answer, _, _ = provider.analyze_gemini("case scenario", max_output_tokens=150)

    assert answer == "fallback answer after retry"
    assert len(request_payloads) == 2
    assert request_payloads[0]["max_completion_tokens"] == 150
    assert request_payloads[1]["max_completion_tokens"] == 1024
    assert request_payloads[1]["reasoning_effort"] == "low"


def test_groq_fallback_reports_empty_response_after_retry(monkeypatch, tmp_path):
    rag = _load_rag_with_stubbed_embeddings(monkeypatch)

    class EmptyGroqResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "message": {"content": ""},
                        "finish_reason": "length",
                    }
                ],
                "usage": {"completion_tokens": 1024},
            }

    monkeypatch.setattr(rag, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(rag.requests, "post", lambda *args, **kwargs: EmptyGroqResponse())

    provider = object.__new__(rag.LocalFAISSRag)
    with pytest.raises(RuntimeError, match="after retry.*finish_reason=length"):
        provider._query_groq("case prompt", max_output_tokens=150)


def test_first_turn_prompt_uses_conversational_style(monkeypatch, tmp_path):
    rag = _load_rag_with_stubbed_embeddings(monkeypatch)
    captured_prompts = []

    class GeminiModels:
        def generate_content(self, **kwargs):
            captured_prompts.append(kwargs["contents"].parts[0].text)
            return SimpleNamespace(text="यह एक संक्षिप्त उत्तर है।")

    _success_client = SimpleNamespace(models=GeminiModels())
    monkeypatch.setattr(rag, "gemini_client", _success_client)
    monkeypatch.setattr(rag, "_gemini_clients", [_success_client])
    monkeypatch.setattr(rag.LocalFAISSRag, "search", lambda *args, **kwargs: [])

    provider = object.__new__(rag.LocalFAISSRag)
    provider.data_dir = tmp_path
    query = "एक महिला को धमकी मिली। अब पुलिस को क्या करना चाहिए?"
    provider.analyze_gemini(query)

    assert query in captured_prompts[0]
    assert "FIRST-TURN CHAT RESPONSE" in captured_prompts[0]
    assert "Do not automatically produce a report" in captured_prompts[0]
    assert "complete, structured analysis using EXACTLY" not in captured_prompts[0]
