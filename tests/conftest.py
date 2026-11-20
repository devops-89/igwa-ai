import sys
import types

import pytest
from fastapi.testclient import TestClient


class FakeRag:
    def analyze_gemini(self, scenario, follow_up=None, allow_groq_fallback=True, image_attachments=None,
                       chat_history=None, max_output_tokens=1200):
        FakeRag.last = dict(scenario=scenario, follow_up=follow_up, history=chat_history,
                            images=image_attachments, max_output_tokens=max_output_tokens)
        return "उत्तर: BNS धारा 103 लागू।", [{"source": "bns.pdf", "text": "line one\nline two"}], None

@pytest.fixture
def client(tmp_path, monkeypatch):
    stub = types.ModuleType("rag")
    stub.chunk_text = lambda t: [t[i:i + 50] for i in range(0, len(t), 50)]
    stub.LocalFAISSRag = FakeRag
    monkeypatch.setitem(sys.modules, "rag", stub)
    monkeypatch.setenv("CHATBOT_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("CHATBOT_SESSIONS_DIR", str(tmp_path / "sessions"))

    import api
    from config import get_settings
    get_settings.cache_clear(); api.get_rag_provider.cache_clear()
    with TestClient(api.create_app(), raise_server_exceptions=False) as c:
        yield c
