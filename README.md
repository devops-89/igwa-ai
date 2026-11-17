# IGWA Legal Chatbot (विवेचना एवं केस डायरी प्रणाली)

An AI-powered legal investigation and case diary assistant for Indian law enforcement, supporting **BNS 2023**, **BNSS 2023**, and **BSB 2023**. Built with **FastAPI**, **Streamlit**, local **FAISS** vector search, and **Gemini API** (with Groq fallback).

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
GEMINI_API_KEY="your_gemini_api_key"
GROQ_API_KEY="your_groq_api_key"        # Optional fallback
CHAT_API_URL="http://127.0.0.1:8000"    # Optional
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

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health & index status |
| `POST` | `/api/chat` | Send case facts, queries, and images |
| `GET` | `/docs` | Interactive Swagger API documentation |

### Example Request (`POST /api/chat`)

```json
{
  "query": "आरोपी ने धारदार हथियार से हमला किया, कौन सी धाराएं लागू होंगी?",
  "case_facts": "दिनांक 12.11.2026 की रात्रि घटना घटित हुई।",
  "document_text": null,
  "images": []
}
```
