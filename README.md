# IGWA OCR API

The OCR service of the IGWA police software backend. A client uploads a PDF or image, the backend validates it, sends it to Gemini Flash for OCR, keeps the result temporarily in memory and returns a `document_id`. The OCR text is then fetched with that ID.

## OCR flow

```text
POST /python/ocr/upload -> validate file -> Gemini Flash OCR -> store in memory -> document_id
GET  /python/ocr/{document_id} -> OCR result from memory
```

Uploaded files are processed in memory and never written to disk.

## Folder structure

```text
IGWA/
├── services/ocr/      routes.py, service.py, gemini_client.py, schemas.py, utils.py
├── config/            settings.py
├── main.py
├── requirements.txt
├── .env / .env.example
└── README.md
```

## Requirements

- Python 3.10+
- A Gemini API key from https://aistudio.google.com/apikey

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Gemini key configuration

Copy `.env.example` to `.env` and set your key:

```env
APP_NAME=IGWA
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8000

GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.8-flash

MAX_FILE_SIZE_MB=20
```

## Run

```powershell
python -m uvicorn main:app --reload --port 8000
```

Swagger UI: http://localhost:8000/docs

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/python/ocr/upload` | Upload PDF/PNG/JPG/JPEG/WEBP, run OCR, get `document_id` |
| GET | `/python/ocr/{document_id}` | Fetch the stored OCR result |
| GET | `/python/ocr/{job_id}/metadata` | Poll background metadata status/result (`job_id` == `document_id`) |
| GET | `/health` | Liveness check (does not call Gemini) |

Errors: 400 (unsupported/empty file), 413 (too large), 404 (unknown document), 502 (Gemini failure).

## Example cURL

```bash
curl -X POST "http://localhost:8000/python/ocr/upload" -F "file=@sample.pdf"
curl "http://localhost:8000/python/ocr/3a6a76b5-0c20-4a57-975a-39d13bfbf53e"
```

Upload response:

```json
{
  "success": true,
  "message": "OCR processing completed successfully",
  "document_id": "3a6a76b5-0c20-4a57-975a-39d13bfbf53e",
  "status": "completed",
  "filename": "sample.pdf"
}
```

Get OCR response:

```json
{
  "success": true,
  "document_id": "3a6a76b5-0c20-4a57-975a-39d13bfbf53e",
  "status": "completed",
  "filename": "sample.pdf",
  "ocr_data": {
    "pages": [{ "page_number": 1, "text": "..." }],
    "full_text": "..."
  }
}
```

## Temporary memory storage

This version stores OCR results in application memory.

If the FastAPI server restarts, all OCR results are lost.

This is intentional for the current development phase.

## Metadata in the same request

Gemini returns the OCR text and the 15 metadata fields in ONE request, so `POST /python/ocr/upload` responds with
`metadata` filled and `metadata_status: "completed"`. No second pass runs. Only if the metadata is missing,
empty or invalid (or Groq did the OCR) does the old background extraction run: then `metadata` is `null`,
`metadata_status` is `pending` and you poll `GET /python/ocr/{job_id}/metadata`. Bad metadata never discards the OCR text.

## Asynchronous metadata fallback (API contract)

`POST /python/ocr/upload` now returns as soon as OCR finishes, with `job_id`, `metadata: null` and
`metadata_status: "pending"`. Metadata (Gemini, Groq fallback) runs in a FastAPI background
task; poll `GET /python/ocr/{job_id}/metadata` every 2-3 s until `metadata_status` is `completed`,
`failed` or `skipped` (empty OCR text), with a timeout and cancel on page leave.
Statuses: `pending`, `processing`, `completed`, `failed`, `skipped`.

Limitations: jobs live in process memory and BackgroundTasks is not a durable queue, so a
restart/reload loses results and any job still `pending`/`processing`; multiple uvicorn
workers do not share state (run a single worker, or move to a shared store + real queue).

## OCR engine: Gemini with key failover, Groq fallback

The uploaded PDF/image goes to Gemini as a whole document (credentials tried in order, see below); if every
Gemini credential fails or is unavailable, the same document goes to Groq. There is no local OCR model.
`ocr_provider` in the response is `gemini` or `groq`. Both prompts forbid translation and transliteration:
Hindi stays in Devanagari, English stays English, digits are copied exactly. Gemini's output is parsed and
validated, never post-edited. OCR text quality is whatever the provider returns (no local quality scoring),
so review critical identifiers. Encrypted/corrupt files are rejected by the provider path as an OCR failure (502).

### Gemini API key failover

`services/ocr/gemini_key_manager.py` tries credentials strictly in configured order, one request at a time,
for both OCR escalation and background metadata. Configure `GEMINI_API_KEYS=k1,k2,...` or
`GEMINI_API_KEY_1..10` (precedence: list > numbered > legacy `GEMINI_API_KEY`; placeholders and duplicates are
ignored). Placing keys in `.env` is the only supported way: nothing is hardcoded and keys are never logged.

| Failure | Action |
| --- | --- |
| 401, "API key not valid/expired/leaked" (the Gemini API reports an invalid key as HTTP 400) | key disabled for the process; next key immediately |
| 402 billing, 403 permission | key cooldown (default 1h); next key |
| 429 | cooldown = `Retry-After` (else 60s); daily quota = 1h; next key |
| DNS/network, timeout, 5xx/overload | **no rotation**: one request, then fast-fail for 10s (goes to Groq) |
| 400/404/413 (bad request, wrong model) | **no rotation**, no key penalized |
| all keys unavailable | Groq (existing fallback) |

SDK-internal retries are disabled (`attempts=1`) so failover is the only retry layer; each attempt has
`GEMINI_REQUEST_TIMEOUT_SECONDS`, the whole chain `GEMINI_OVERALL_BUDGET_SECONDS`. Switching keys re-sends the same
request only to the next key; nothing is re-processed locally. `/health` shows only
`gemini_keys: {configured, available}`. Logs use slot numbers (`Gemini credential slot 2: quota exhausted`).
Limits: cooldown/disabled state is per process and resets on restart; a hung connection is bounded by the
per-attempt timeout, not a separate connect timeout (the SDK transport differs by environment).

### isFir (FIR vs application)

Every response (`POST /python/ocr/upload`, `GET /python/ocr/{id}`, `GET /python/ocr/{job_id}/metadata`) has a
boolean `isFir` right after `metadata_error`. It comes from a `document_class` value
(`FIR` / `APPLICATION` / `OTHER` / `UNCLEAR`) that Gemini returns inside the same request as the OCR and metadata
(or from the text-only Gemini/Groq metadata fallback). `isFir` is `true` only for `FIR`. An application that
quotes an FIR number is `APPLICATION`; an official FIR form followed by attached letters is still `FIR`.
Missing/`UNCLEAR`/invalid classification gives `false` and a "classification unresolved" log line (a confirmed
non-FIR logs as "classified as non-FIR"). While metadata is still pending, `isFir` is `false` until the
background step completes. `document_class` itself is internal and never appears in the `metadata` object.

### metadata.application (attached application/complaint)

`metadata.application` is `{found, start_page, end_page, text}`. The model only reports *where* the application is
(`application_location`: page range, plus short verbatim `start_text`/`end_text` quotes when it starts/ends
mid-page) in the existing single Gemini request (or the existing background metadata request after Groq OCR, where
pages are marked `[[PAGE n]]`). `text` is cut from `ocr_data.pages` in Python (`services/ocr/application.py`):
nothing is regenerated and no extra model call is made.

- `found=false` (all other fields `null`): confirmed no application.
- `found=true`, `text=null`: located, but a mid-page boundary quote was not found verbatim, so no text is
  returned rather than risk returning FIR text.
- `application=null` (or `metadata=null` while pending/failed): unavailable, e.g. the model returned no/invalid
  location. This is never reported as "no application".
