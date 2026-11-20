# IGWA Legal Chatbot (विवेचना प्रणाली)

An AI-powered legal investigation and case diary assistant designed exclusively for Indian law enforcement and Investigating Officers (IO / अन्वेषण अधिकारी / विवेचक), supporting **BNS 2023** (Bharatiya Nyaya Sanhita), **BNSS 2023** (Bharatiya Nagarik Suraksha Sanhita), and **BSB 2023** (Bharatiya Sakshya Adhiniyam). Built with **FastAPI**, **Streamlit**, local **FAISS** vector search, and **Google Gemini API** (with multi-key failover and Groq fallback).

## Key Features & Legal Architecture

- **Statutory Offence Relevance Engine**:
  - Substantive criminal offences, penal charges, and punishments are cited strictly under **BNS 2023** or Special Acts (e.g., Information Technology Act 2000, Arms Act, NDPS Act, Electricity Act).
  - **BNSS 2023** is strictly maintained as procedural law (FIR under Sec 173, NCR under Sec 174, search & seizure with mandatory videography under Sec 105, notice/arrest guidelines under Sec 35, witness examination under Sec 180, Case Diary and charge sheet under Sec 193/168). BNSS sections are never cited as penal offences.
  - Total exclusion of repealed **IPC** and **CrPC** sections (e.g., automatic mapping of cheating to BNS 318(4), forgery to BNS 336(3)/338/340(2), breach of trust to BNS 316, and conspiracy to BNS 61(2)).
- **Clean RAG & First Schedule Filtering**:
  - Semantic topic keyword and explicit section mapping to extract exact First Schedule classifications from `data/bnss_first_schedule_classifications.json`.
  - Immune to OCR header/page-number artifacts from ingested legal acts or tahrir PDFs.
- **Unabbreviated Output & Zero Truncation**:
  - High-capacity generation budget (up to 3,000 tokens for case analyses) ensuring complete responses with all 4 headings, classification tables, and actionable IO SOPs ending cleanly with a Hindi full stop (`।`).
- **Two Operational Chat Modes**:
  - **Mode 1 (First-turn Case Analysis)**: 1. मामले का सारांश एवं महत्वपूर्ण बिंदु, 2. अपराध का वर्गीकरण (तालिका), 3. लागू विधिक धाराएँ एवं दंड (BNS 2023 / विशेष अधिनियम), 4. विवेचना अधिकारी हेतु कार्ययोजना (SOP, सबूत संकलन एवं पर्चा दैनिकी).
  - **Mode 2 (Realtime Follow-up)**: Direct, focused answers to specific IO inquiries, evidentiary doubts, or Case Diary drafting points without repeating redundant headings.
- **Model Hierarchy & Key Failover**:
  - Primary Model: `gemini-3.1-flash-lite` (fast, highly available, zero 503 latency).
  - Fallback Models: `gemini-3.5-flash`, `gemini-3.8-flash`, followed by Groq fallback (`openai/gpt-oss-120b`).
  - Supports multiple comma-separated keys in `GEMINI_API_KEY` or `GEMINI_API_KEYS` with automatic quota and transient failure failover.

## Backend Layout

- `backend/schema.py` — request and response models
- `backend/service.py` — chat, RAG, audit, and session operations
- `backend/routes.py` — API routers, middleware, error handling, and app factory
- `backend/guardrails.py` — chatbot scope and response checks
- `rag.py` — RAG pipeline, Gemini client failover, First Schedule lookup, and system prompt
- `api.py` — compatibility entry point for `uvicorn api:app`

>>>>>>> 8b9ac16 (Add chatbot API, backend and tests)
---

## ⚡ Quick Start

### 1. Setup Environment

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create `.env` from `.env.example`:

```powershell
Copy-Item .env.example .env
```

Add your API keys in `.env`:
```ini
# Supports single key or comma-separated keys for automatic rotation & failover
GEMINI_API_KEY="key1,key2,key3"
GEMINI_API_KEYS="key4,key5"             # Optional additional keys
GROQ_API_KEY="your_groq_api_key"        # Optional fallback
CHAT_API_URL="http://127.0.0.1:8000"    # Optional
CHATBOT_API_KEY="your_api_key"           # Optional; protects all API routes except /health
```

### 3. Run the Application

Start the **Backend API** (Terminal 1):
```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Start the **Streamlit UI** (Terminal 2):
```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

- **Web UI:** `http://localhost:8501`
- **Swagger Docs:** `http://127.0.0.1:8000/docs`
- **ngrok URL:** 'https://rebirth-equator-esophagus.ngrok-free.dev/docs'

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
<<<<<<< HEAD
| `GET` | `/health` | Service health & index status |
| `POST` | `/documents/ingest` | Ingest raw or structured OCR JSON into vector index |
| `POST` | `/api/chat` | Send case facts, queries, and images (with session tracking) |
| `GET` | `/api/sessions` | List all saved chat sessions |
| `GET` | `/api/sessions/{id}` | Retrieve history for a specific session |
| `GET` | `/docs` | Interactive Swagger API documentation |
=======
| `GET` | `/health` | Health check and RAG initialization status |
| `POST` | `/api/v1/chat` | Persistent chat; accepts case questions and optional OCR/context text |
| `POST` | `/api/chat` | Persistent chat; creates a session ID and continues saved conversations |
| `GET` | `/api/sessions` | List sessions created through the legacy chat route |
| `GET` | `/api/sessions/{id}` | Retrieve a saved session |
| `GET` | `/docs` | Interactive OpenAPI/Swagger documentation |
>>>>>>> 8b9ac16 (Add chatbot API, backend and tests)

`/api/v1/chat` creates a system-generated session ID when `session_id` is omitted and restores the saved conversation when the returned ID is sent on a follow-up request. Do not invent an ID. OCR data is supplied as context to a chat request; there is no OCR document-ingestion endpoint. When `CHATBOT_API_KEY` is configured, send it as the `X-API-Key` header to all endpoints except `/health`. API responses include an `X-Request-ID` header; structured errors include the same ID in the response body.

Clients may optionally provide a `summary` string to include a compact prior-conversation recap as context; unknown request fields remain rejected.

The Streamlit dashboard provides normal legal chat, typed case-scenario, and OCR-context modes. OCR context can be pasted as text/metadata or extracted locally from PDF/DOCX; image files are not accepted, and only extracted text and metadata are sent to the API. The dashboard displays the session ID returned by the API after the first message. Query and OCR/document text fields allow up to 100,000 characters each (approximately 10,000 words, depending on language and spacing).

### Normal Chat or Typed Case Scenario

For a normal legal question, send only `query`. To ask about a typed case, put the case scenario and question together in the same field:

```json
{
  "query": "A woman reports that a person has followed her for weeks and threatened her. What should police do next?"
}
```

### Continuing a Persistent Chat

First request (`POST /api/v1/chat`):

```json
{
  "query": "एक महिला ने शिकायत की कि आरोपी कई सप्ताह से उसका पीछा कर रहा था। अब पुलिस को क्या कदम उठाने चाहिए?"
}
```

The response includes a system-generated `session_id`. For the next query, send that returned ID; the server restores the saved case context and conversation history:

```json
{
  "session_id": "case_<id-from-first-response>",
  "query": "अब कौन से साक्ष्य एकत्र करने चाहिए?"
}
```

### Asking About OCR Context (`POST /api/v1/chat`)

```json
{
  "query": "इस OCR विवरण के आधार पर अब पुलिस को क्या कदम उठाने चाहिए?",
  "ocr_data": {
    "pages": [
      {
        "page_number": 1,
        "text": "एक महिला ने आरोपी द्वारा पीछा किए जाने और धमकी दिए जाने की शिकायत की।"
      }
    ]
  }
}
```

Without OCR, put the case description and question together in `query`. With OCR, put the question in `query` and the recognized text in `ocr_data` or `document_text`. The API does not accept image files or index uploaded OCR documents.
