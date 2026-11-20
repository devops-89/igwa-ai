import json
import math
import os
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import faiss
import httpx
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pypdf import PdfReader
import requests
from sentence_transformers import SentenceTransformer

load_dotenv()

_primary_keys = [k.strip() for k in os.getenv("GEMINI_API_KEY", "").split(",") if k.strip()]
_extra_keys = [k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(",") if k.strip()]

# Build deduplicated ordered key list
_seen_keys: set = set()
GEMINI_API_KEYS: List[str] = []
for _k in _primary_keys + _extra_keys:
    if _k and _k not in _seen_keys:
        GEMINI_API_KEYS.append(_k)
        _seen_keys.add(_k)

GEMINI_API_KEY = GEMINI_API_KEYS[0] if GEMINI_API_KEYS else ""

_GEMINI_TIMEOUT = types.HttpOptions(timeout=60000)
_gemini_clients: List[genai.Client] = [
    genai.Client(api_key=k, http_options=_GEMINI_TIMEOUT)
    for k in GEMINI_API_KEYS
]

gemini_client = _gemini_clients[0] if _gemini_clients else None

GEMINI_MODEL = "gemini-3.1-flash-lite"
GEMINI_FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-3.8-flash"]

# Status codes that mean "try next key" vs. "this key is permanently broken"
_QUOTA_CODES = {429, 500, 502, 503, 504, 408}
_AUTH_CODES  = {400, 401, 403}

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip() or None
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_FALLBACK_MODELS = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TOKENS_PER_MINUTE = 15_000
TRANSIENT_GEMINI_STATUS_CODES = {404, 408, 429, 500, 502, 503, 504}


def _parse_duration_seconds(value: str | None) -> float | None:
    """Parse Groq reset durations such as ``7.66s`` or ``2m59.56s``."""
    if not value:
        return None

    matches = re.findall(r"(\d+(?:\.\d+)?)(ms|s|m|h)", value.strip().lower())
    if not matches or "".join(number + unit for number, unit in matches) != value.strip().lower():
        try:
            return max(0.0, float(value))
        except ValueError:
            return None

    factors = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
    return sum(float(number) * factors[unit] for number, unit in matches)


def _estimate_tokens(text: str) -> int:
    """Estimate tokens, accounting for ASCII and multibyte text separately."""
    ascii_characters = sum(character.isascii() for character in text)
    non_ascii_bytes = len(text.encode("utf-8")) - ascii_characters
    return math.ceil(ascii_characters / 5 + non_ascii_bytes / 3)


def _clean_truncated_tail(text: str) -> str:
    """If generation terminated mid-sentence or mid-token, trim back to the last complete sentence."""
    if not text:
        return text
    stripped = text.rstrip()

    # Strip any trailing incomplete bullet or hanging heading marker
    while True:
        lines = stripped.splitlines()
        if not lines:
            break
        last_line = lines[-1].strip()
        if last_line in ("-", "*", "•", "#", "##", "###", "####", "1.", "2.", "3.", "4.", "5.") or (last_line.startswith("**") and not last_line.endswith("**")):
            stripped = "\n".join(lines[:-1]).rstrip()
        elif last_line.startswith("|") and not last_line.endswith("|"):
            stripped = "\n".join(lines[:-1]).rstrip()
        else:
            break

    if stripped.endswith(("।", ".", "!", "?", "```", "—")):
        return stripped

    # Search for the last Hindi purna viram '।' or period '. '
    last_pv = stripped.rfind("।")
    last_dot = stripped.rfind(". ")
    last_nl = stripped.rfind("\n")

    candidates = [pos for pos in (last_pv, last_dot) if pos != -1]
    if candidates:
        best = max(candidates)
        # Keep at least 50% of the text if a clean sentence boundary exists
        if best >= len(stripped) * 0.5:
            return stripped[: best + 1].rstrip()

    if last_nl != -1 and last_nl >= len(stripped) * 0.5:
        return stripped[:last_nl].rstrip()

    return stripped


class _GroqTokenRateLimiter:
    """Process-wide rolling-window TPM guard, informed by Groq response headers."""

    def __init__(self, tokens_per_minute: int = GROQ_TOKENS_PER_MINUTE):
        self._configured_limit = tokens_per_minute
        self._effective_limit = tokens_per_minute
        self._reservations: deque[tuple[float, int]] = deque()
        self._server_remaining: int | None = None
        self._server_reset_at: float | None = None
        self._retry_after: float | None = None
        self._condition = threading.Condition()

    def acquire(self, tokens: int) -> None:
        if tokens <= 0:
            return

        with self._condition:
            if tokens > self._effective_limit:
                raise RuntimeError(
                    f"Estimated Groq request size ({tokens} tokens) exceeds the "
                    f"configured TPM allowance ({self._effective_limit})."
                )

            while True:
                now = time.monotonic()
                while self._reservations and now - self._reservations[0][0] >= 60:
                    self._reservations.popleft()

                if self._server_reset_at is not None and now >= self._server_reset_at:
                    self._server_remaining = None
                    self._server_reset_at = None
                if self._retry_after is not None and now >= self._retry_after:
                    self._retry_after = None

                locally_available = self._effective_limit - sum(
                    amount for _, amount in self._reservations
                )
                available = locally_available
                if self._server_remaining is not None:
                    available = min(available, self._server_remaining)

                retry_delay = (
                    max(0.0, self._retry_after - now)
                    if self._retry_after is not None
                    else 0.0
                )
                if available >= tokens and retry_delay == 0:
                    self._reservations.append((now, tokens))
                    if self._server_remaining is not None:
                        self._server_remaining -= tokens
                    return

                delays = [retry_delay] if retry_delay else []
                if self._reservations:
                    delays.append(max(0.01, 60 - (now - self._reservations[0][0])))
                if self._server_reset_at is not None:
                    delays.append(max(0.01, self._server_reset_at - now))
                self._condition.wait(timeout=min(delays) if delays else 60)

    def observe_headers(self, headers: Any) -> None:
        """Use Groq's remaining-token/reset headers to constrain later requests."""
        now = time.monotonic()
        with self._condition:
            try:
                server_limit = int(headers.get("x-ratelimit-limit-tokens", ""))
                if server_limit > 0:
                    self._effective_limit = min(self._configured_limit, server_limit)
            except (TypeError, ValueError):
                pass

            try:
                remaining = int(headers.get("x-ratelimit-remaining-tokens", ""))
                if remaining >= 0:
                    self._server_remaining = min(self._effective_limit, remaining)
            except (TypeError, ValueError):
                pass

            reset = _parse_duration_seconds(headers.get("x-ratelimit-reset-tokens"))
            retry_after = _parse_duration_seconds(headers.get("retry-after"))
            if reset is not None:
                self._server_reset_at = now + reset
            if retry_after is not None:
                self._retry_after = now + retry_after
            self._condition.notify_all()


_groq_token_rate_limiter = _GroqTokenRateLimiter()

# Initialize Local Embedding Model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
print(f"Loading local embedding model: {EMBEDDING_MODEL_NAME}...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()

SYSTEM_PROMPT = (
    "You are an expert Indian criminal law and investigation assistant designed EXCLUSIVELY for "
    "Indian Police Investigating Officers (IO / अन्वेषण अधिकारी / विवेचक) and SHOs. "
    "\n\n"
    "POLICE PERSPECTIVE ONLY (NOT CITIZENS):\n"
    "- The user is ALWAYS an active Police Officer / IO conducting an official investigation, NEVER a citizen or victim.\n"
    "- NEVER give citizen advice (no 'थाने जाएं', '112 डायल करें', 'वकील से सलाह लें').\n"
    "- Guide IO on FIR (BNSS 173), Case Diary (BNSS 193/168), crime scene visit & site plan (नक्शा नजरी), "
    "seizure memo with mandatory videography (BNSS 105), witness statements (BNSS 180), notice/arrest rules (BNSS 35), "
    "digital evidence & Section 63 BSB certificate, and charge sheet (BNSS 193).\n"
    "\n"
    "LANGUAGE: Write all user responses entirely in formal Hindi (Devanagari script) matching official police Case Diary register. "
    "Preserve English abbreviations (BNS 2023, BNSS 2023, BSB 2023, IT Act, FIR, IO, SHO, CCTV, UPI).\n"
    "\n"
    "STATUTORY JURISDICTION & OFFENCE CITATION (CRITICAL RULES):\n"
    "1. SUBSTANTIVE OFFENCES VS PROCEDURAL LAW:\n"
    "   - All criminal offences, charges, and punishments MUST be cited ONLY under Bharatiya Nyaya Sanhita (BNS 2023) or Special Acts (e.g. IT Act 2000, Arms Act).\n"
    "   - BNSS 2023 is STRICTLY procedural law. NEVER cite BNSS sections as offences (citing BNSS 84, 161, 202, 90, 146, 163, 203 as crimes is strictly forbidden).\n"
    "2. REPEALED LAWS FORBIDDEN:\n"
    "   - NEVER cite IPC or CrPC. Map repealed sections to BNS (IPC 420 -> BNS 318(4), IPC 467/468 -> BNS 338/336(3), IPC 120B -> BNS 61(2), IPC 406/409 -> BNS 316).\n"
    "3. OFFENCE RELEVANCE ENGINE (BNS 2023):\n"
    "   - Cheating/Fraud: BNS 318(4) (cheating & dishonestly inducing delivery of property), 318(2).\n"
    "   - Forgery/Fake documents: BNS 336(3) (forgery for cheating), 338 (valuable security), 340(2) (using forged document as genuine), 336(2).\n"
    "   - Criminal Breach of Trust: BNS 316(2), 316(5) (by banker/agent/merchant).\n"
    "   - Criminal Conspiracy: BNS 61(2); Common Intention: BNS 3(5).\n"
    "   - Cyber Fraud (UPI/net-banking abuse): IT Act Section 66C (identity theft), Section 66D (cheating by personation).\n"
    "   - Theft/Trespass: BNS 303(2), 305, 331. Robbery/Extortion: BNS 308, 309.\n"
    "   - Bodily offences: BNS 115(2), 117, 109, 103(1). Electricity theft: Electricity Act 135.\n"
    "   - Only cite sections that closely and directly apply to the explicit facts of the case.\n"
    "\n"
    "FACTUAL DISCIPLINE: Treat stated scenario as ONLY ground truth. Never invent facts.\n"
    "\n"
    "COMPLETENESS & FINISH DISCIPLINE:\n"
    "- Deliver complete, unabbreviated sections. NEVER stop mid-sentence, mid-bullet, or mid-table row.\n"
    "- Always conclude with complete IO guidance and end with a Hindi full stop (।).\n"
    "\n"
    "MODE 1: FIRST TURN (STRUCTURED CASE ANALYSIS):\n"
    "When analyzing a case, use these 4 headings:\n"
    "### 1. मामले का सारांश एवं महत्वपूर्ण बिंदु\n"
    "### 2. अपराध का वर्गीकरण (Classification Table: अधिनियम व धारा | संज्ञेय/असंज्ञेय | जमानतीय | विचारणीय न्यायालय) [BNS/Special Acts only]\n"
    "### 3. लागू विधिक धाराएँ एवं दंड (BNS 2023 / Special Acts)\n"
    "### 4. विवेचना अधिकारी हेतु कार्ययोजना (SOP, सबूत संकलन एवं पर्चा दैनिकी)\n"
    "\n"
    "MODE 2: REALTIME CHATBOT (FOLLOW-UP):\n"
    "Answer the IO specific question directly and concisely without repeating the 4-heading template."
)


def chunk_text(text: str, chunk_size: int = 600, chunk_overlap: int = 80) -> List[str]:
    """Splits text into chunks of roughly `chunk_size` characters with overlap."""
    text = " ".join(text.split())  # Clean up excessive whitespace
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


def extract_chunks_from_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """Reads a PDF and returns chunks with source and page metadata."""
    chunks = []
    try:
        reader = PdfReader(str(pdf_path))
        for page_idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            page_chunks = chunk_text(page_text)
            for c in page_chunks:
                chunks.append({
                    "text": c,
                    "source": pdf_path.name,
                    "page": page_idx + 1
                })
    except Exception as e:
        print(f"Error reading PDF {pdf_path.name}: {e}")
    return chunks


def extract_chunks_from_json(json_path: Path) -> List[Dict[str, Any]]:
    """Reads structured legal JSON and converts items to text chunks."""
    chunks = []
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for item in data:
                text = json.dumps(item, ensure_ascii=False)
                chunks.append({
                    "text": text,
                    "source": json_path.name,
                    "page": 1
                })
        elif isinstance(data, dict):
            for k, v in data.items():
                text = f"{k}: {json.dumps(v, ensure_ascii=False)}"
                chunks.append({
                    "text": text,
                    "source": json_path.name,
                    "page": 1
                })
    except Exception as e:
        print(f"Error reading JSON {json_path.name}: {e}")
    return chunks


class LocalFAISSRag:
    def __init__(self, data_dir: str = "data", persist_dir: str = "./storage"):
        self.data_dir = Path(data_dir)
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_file = self.persist_dir / "index.faiss"
        self.chunks_file = self.persist_dir / "chunks.json"
        self.manifest_file = self.persist_dir / "indexed_files.json"
        
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Dict[str, Any]] = []
        self.indexed_files: List[str] = []
        
        self.load_or_build_index()

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vectors, axis=1, keepdims=True)
        return np.where(norm > 0, vectors / norm, vectors).astype("float32")

    def _save_state(self):
        """Persists the FAISS index, chunk metadata, and file manifest."""
        print(f"Saving FAISS index to {self.index_file}...")
        faiss.write_index(self.index, str(self.index_file))
        
        print(f"Saving {len(self.chunks)} chunk(s) metadata to {self.chunks_file}...")
        self.chunks_file.write_text(
            json.dumps(self.chunks, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        
        self.manifest_file.write_text(
            json.dumps({"indexed_files": sorted(self.indexed_files)}, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def load_or_build_index(self):
        """Loads existing index or builds/updates incrementally."""
        if self.index_file.exists() and self.chunks_file.exists():
            print(f"Loading existing FAISS index from {self.persist_dir}...")
            self.index = faiss.read_index(str(self.index_file))
            self.chunks = json.loads(self.chunks_file.read_text(encoding="utf-8"))
            if self.manifest_file.exists():
                manifest = json.loads(self.manifest_file.read_text(encoding="utf-8"))
                self.indexed_files = manifest.get("indexed_files", [])
            else:
                self.indexed_files = []
            
            # Check for new files in data_dir
            self._incremental_update()
        else:
            self._fresh_build()

    def _fresh_build(self):
        """Builds a fresh FAISS index from all files in data_dir."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory '{self.data_dir}' not found.")

        print(f"Scanning documents in {self.data_dir}...")
        all_chunks: List[Dict[str, Any]] = []
        processed_files = []

        files = sorted(list(self.data_dir.glob("*.pdf")) + list(self.data_dir.glob("*.json")))
        for f in files:
            print(f"  • Reading {f.name}...")
            if f.suffix.lower() == ".pdf":
                file_chunks = extract_chunks_from_pdf(f)
            elif f.suffix.lower() == ".json":
                file_chunks = extract_chunks_from_json(f)
            else:
                continue

            all_chunks.extend(file_chunks)
            processed_files.append(f.name)

        if not all_chunks:
            raise ValueError(f"No text extracted from documents in {self.data_dir}")

        print(f"Extracted {len(all_chunks)} chunks. Generating local embeddings...")
        texts = [c["text"] for c in all_chunks]
        embeddings = embedding_model.encode(texts, batch_size=64, show_progress_bar=True)
        embeddings = self._normalize(np.array(embeddings, dtype="float32"))

        # Cosine similarity using Inner Product with normalized vectors
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.index.add(embeddings)
        self.chunks = all_chunks
        self.indexed_files = processed_files
        self._save_state()
        print(f"Index built successfully with {self.index.ntotal} vectors.")

    def _incremental_update(self):
        """Detects new files in data_dir and appends them to the index."""
        if not self.data_dir.exists():
            return

        already_indexed = set(self.indexed_files)
        current_files = {f.name: f for f in list(self.data_dir.glob("*.pdf")) + list(self.data_dir.glob("*.json"))}
        new_file_names = sorted(set(current_files.keys()) - already_indexed)

        if not new_file_names:
            print("No new documents detected – FAISS index is up to date.")
            return

        print(f"Found {len(new_file_names)} new file(s) to index:")
        new_chunks = []
        for name in new_file_names:
            print(f"  • {name}")
            f_path = current_files[name]
            if f_path.suffix.lower() == ".pdf":
                new_chunks.extend(extract_chunks_from_pdf(f_path))
            elif f_path.suffix.lower() == ".json":
                new_chunks.extend(extract_chunks_from_json(f_path))

        if not new_chunks:
            print("No valid content found in new files.")
            return

        print(f"Generating embeddings for {len(new_chunks)} new chunks...")
        texts = [c["text"] for c in new_chunks]
        new_embeddings = embedding_model.encode(texts, batch_size=64, show_progress_bar=True)
        new_embeddings = self._normalize(np.array(new_embeddings, dtype="float32"))

        self.index.add(new_embeddings)
        self.chunks.extend(new_chunks)
        self.indexed_files.extend(new_file_names)
        self._save_state()
        print(f"Incremental update complete. Total vectors: {self.index.ntotal}")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Searches local FAISS index for the top-k most relevant chunks."""
        if self.index is None or self.index.ntotal == 0:
            return []

        q_embed = embedding_model.encode([query])
        q_embed = self._normalize(np.array(q_embed, dtype="float32"))
        scores, indices = self.index.search(q_embed, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                chunk = dict(self.chunks[idx])
                chunk["score"] = float(score)
                results.append(chunk)
        return results

    def _schedule_context(
        self, scenario: str, retrieved_chunks: List[Dict[str, Any]]
    ) -> str:
        """Return exact BNSS First Schedule rows for section numbers in relevant text."""
        schedule_path = self.data_dir / "bnss_first_schedule_classifications.json"
        if not hasattr(self, "_schedule_cache") or self._schedule_cache is None:
            if not schedule_path.exists():
                return ""
            try:
                raw_schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
                rows = raw_schedule.get("rows", {})
                roots_map: Dict[str, List[str]] = {}
                for section, classification in rows.items():
                    root = section.split("(", 1)[0]
                    roots_map.setdefault(root, []).append(
                        f"{section}: {json.dumps(classification, ensure_ascii=False)}"
                    )
                self._schedule_cache = {
                    "schedule_name": raw_schedule.get("schedule", "BNSS First Schedule"),
                    "source": raw_schedule.get("source", "official schedule extract"),
                    "printed_pages": raw_schedule.get("printed_pages", []),
                    "roots_map": roots_map,
                }
            except Exception as e:
                print(f"[Warning] Could not load schedule classifications: {e}")
                return ""

        roots_map = self._schedule_cache["roots_map"]

        # 1. Targeted section extraction from explicit references in scenario/query
        candidate_roots = set()
        for ref in re.findall(
            r"(?:BNS|धारा|सेक्शन|section|sec\.?)\s*(\d{1,3})",
            scenario,
            flags=re.IGNORECASE,
        ):
            candidate_roots.add(ref)

        # 2. Topic keyword mapping: map factual case allegations directly to BNS roots
        scenario_lower = scenario.lower()
        topic_mappings = [
            # Fraud, Cheating, Loan Scam
            (["धोखाधड़ी", "धोखा", "ठगी", "गबन", "cheating", "fraud", "deceit", "loan", "ऋण", "कर्ज", "बैंक"], ["318", "316"]),
            # Forgery, Fake Documents, Fake Signatures
            (["जालसाजी", "फर्जी", "कूटकरण", "नकली हस्ताक्षर", "हस्ताक्षर", "कूटरचना", "forgery", "fake", "forged"], ["336", "338", "340"]),
            # Breach of Trust & Embezzlement
            (["अमानत में खयानत", "विश्वासघात", "breach of trust", "embezzlement"], ["316"]),
            # Criminal Conspiracy
            (["साजिश", "षड्यंत्र", "षडयंत्र", "मिलीभगत", "conspiracy", "conspired"], ["61"]),
            # Theft & Trespass
            (["चोरी", "theft", "stolen", "घर में घुसकर", "सेंधमारी"], ["303", "305", "331"]),
            # Extortion & Robbery
            (["लूट", "डकैती", "रंगदारी", "वसूली", "robbery", "dacoity", "extortion"], ["308", "309", "310"]),
            # Assault, Hurt & Murder
            (["मारपीट", "चोट", "हमला", "हत्या", "जानलेवा", "assault", "hurt", "murder"], ["115", "117", "109", "103"]),
        ]
        for keywords, roots in topic_mappings:
            if any(kw in scenario_lower for kw in keywords):
                candidate_roots.update(roots)

        matched_rows = []
        for root in sorted(candidate_roots):
            if root in roots_map:
                matched_rows.extend(roots_map[root])

        if not matched_rows:
            return ""

        schedule_name = self._schedule_cache["schedule_name"]
        source = self._schedule_cache["source"]
        printed_pages = self._schedule_cache["printed_pages"]
        page_reference = (
            f", printed pages {printed_pages[0]}–{printed_pages[-1]}"
            if printed_pages
            else ""
        )
        return (
            f"EXACT CLASSIFICATION ROWS — {schedule_name} "
            f"(source: {source}{page_reference}) "
            f"(verify the exact subsection and any conditions):\n"
            + "\n".join(matched_rows)
        )

    def query_gemini(
        self,
        scenario: str,
        top_k: int = 5,
        follow_up: Optional[str] = None,
        allow_groq_fallback: bool = False,
    ) -> str:
        """Analyze a case and return only the answer text."""
        answer, _, _ = self.analyze_gemini(
            scenario,
            top_k=top_k,
            follow_up=follow_up,
            allow_groq_fallback=allow_groq_fallback,
        )
        return answer

    def analyze_gemini(
        self,
        scenario: str,
        top_k: int = 5,
        follow_up: Optional[str] = None,
        allow_groq_fallback: bool = True,
        image_attachments: Optional[List[Dict[str, Any]]] = None,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        max_output_tokens: int = 2500,
    ) -> Tuple[str, List[Dict[str, Any]], str]:
        """Analyze a case and return the answer, sources, and provider."""
        search_query = f"{scenario}\n{follow_up}" if follow_up else scenario
        retrieved_chunks = self.search(search_query, top_k=top_k)
        
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            source = chunk.get("source", "Unknown")
            page = chunk.get("page", "N/A")
            text = chunk.get("text", "")
            context_parts.append(f"[{i}] Source: {source} (Page {page}):\n{text}")

        schedule_context = self._schedule_context(
            f"{scenario}\n{follow_up or ''}", retrieved_chunks
        )
        if schedule_context:
            context_parts.append(schedule_context)
        context_text = "\n\n".join(context_parts) if context_parts else "No relevant legal documents found."

        # Determine effective LLM token limit:
        # Hindi Devanagari text uses ~2.5 to 3 BPE tokens per word.
        # When generating comprehensive legal analysis or IO SOP (max_output_tokens >= 800),
        # allocate at least 3,000 tokens so that the full analysis never cuts off mid-sentence.
        effective_output_tokens = (
            max(max_output_tokens, 3000)
            if max_output_tokens >= 800
            else max(max_output_tokens, 1200)
        )

        if follow_up:
            history_text = ""
            if chat_history:
                recent_lines = []
                for msg in chat_history[-6:]:
                    r = "IO" if msg.get("role") == "user" else "Assistant"
                    c = str(msg.get("content", "")).strip()[:300]
                    recent_lines.append(f"{r}: {c}")
                if recent_lines:
                    history_text = "RECENT CONVERSATION HISTORY:\n" + "\n".join(recent_lines) + "\n\n"

            user_prompt = (
                f"LEGAL CONTEXT (from retrieved acts & schedules):\n{context_text}\n\n"
                f"CASE FACTS ON RECORD:\n{scenario}\n\n"
                f"{history_text}"
                f"INVESTIGATING OFFICER'S (IO) FOLLOW-UP QUESTION:\n{follow_up}\n\n"
                f"TOKEN BUDGET: {effective_output_tokens} tokens. Complete your answer fully within this limit.\n\n"
                "INSTRUCTION: REALTIME CHAT RESPONSE (MODE 2 - FOR POLICE IO ONLY).\n"
                "- Directly, concisely, and practically answer the investigating officer's question in professional police Hindi.\n"
                "- PERSPECTIVE: The user is a police officer/IO, NOT a citizen or victim. Do not suggest citizen actions (e.g. filing complaints, dialing 112). Advise on police actions, statutory powers, evidence rules, and Case Diary (पर्चा दैनिकी) points.\n"
                "- DO NOT repeat the 4-step format (do NOT generate summary, full classification table, or general SOP headings).\n"
                "- Plan your answer to finish completely — never stop mid-sentence or mid-table.\n"
                "- उत्तर की पूर्णता: उत्तर को पूर्ण करें और पूर्ण विराम (।) के साथ समाप्त करें। कभी भी उत्तर अधूरा न छोड़ें।\n"
                "- Never fabricate case facts. If statutory schedules or provisions are missing from retrieved text, supply them accurately using trusted Indian statutory law (BNS 2023, BNSS 2023, BSB 2023)."
            )
        else:
            user_prompt = (
                f"LEGAL CONTEXT (from retrieved acts & schedules):\n{context_text}\n\n"
                f"INVESTIGATING OFFICER'S CASE SCENARIO & QUERY:\n{scenario}\n\n"
                f"TOKEN BUDGET: {effective_output_tokens} tokens. Plan your response so ALL sections are complete before you run out.\n\n"
                "INSTRUCTION: FIRST-TURN CHAT RESPONSE.\n"
                "- The user is an active Investigating Officer (IO / अन्वेषण अधिकारी / विवेचक), NOT a citizen or complainant.\n"
                "- Do not automatically produce a report, fixed headings, or a complete SOP unless explicitly requested; answer the officer's question directly and professionally in standard police Hindi.\n"
                "- NEVER give citizen advice like 'थाने में शिकायत करें', 'वकील से मिलें', or '112 पर फोन करें'. Guide the officer on legal sections, First Schedule classification, spot inspection, BNSS Sec 105 audio-video compliance, BNSS Sec 35 notice/arrest guidelines, and Case Diary entries.\n"
                "- If the officer describes a full crime scenario or asks for legal analysis/next steps, structure under the 4 standard headings: 1. मामले का सारांश एवं महत्वपूर्ण बिंदु, 2. अपराध का वर्गीकरण (तालिका), 3. लागू विधिक धाराएँ एवं दंड (BNS 2023 / विशेष अधिनियम), 4. विवेचना अधिकारी हेतु कार्ययोजना (SOP & पर्चा दैनिकी).\n"
                "- CITE ONLY substantive offences under BNS 2023 or Special Acts. NEVER cite BNSS sections as offences. NEVER cite IPC.\n"
                "- CITE ONLY sections that directly and closely match the facts stated in the scenario (e.g. for forgery/cheating/loan fraud: BNS 318(4), 336(3), 338, 340(2), 316, 61(2), IT Act 66C/66D).\n"
                "- उत्तर की पूर्णता: सभी 4 अनुभागों को विस्तार और स्पष्टता से पूर्ण करें। उत्तर किसी भी स्थिति में अधूरा या आधा-अधूरा न छूटे और अंत में स्पष्ट विधिक निष्कर्ष एवं पूर्ण विराम (।) के साथ समाप्त हो।\n"
                "- FACTUAL DISCIPLINE: Treat the stated scenario as the ONLY case facts. Never fabricate or assume unstated facts.\n"
                "- TRUSTED LEGAL KNOWLEDGE: Supply exact BNS 2023, BNSS 2023, and BSB 2023 provisions accurately without leaving gaps."
            )

        if image_attachments:
            user_prompt += (
                "\nATTACHED IMAGES:\n"
                "Inspect only details clearly visible in the images. Transcribe image text "
                "only when readable, and mark uncertain visual details as unverified. "
                "Do not infer facts that are not clearly shown.\n"
            )
        content_parts = [types.Part.from_text(text=user_prompt)]
        for attachment in image_attachments or []:
            content_parts.append(
                types.Part.from_bytes(
                    data=attachment["data"],
                    mime_type=attachment["mime_type"],
                )
            )
        contents = types.Content(role="user", parts=content_parts)

        # -----------------------------------------------------------------------
        # Build Gemini attempt list: (key_index, model) pairs.
        # Strategy: try primary model across ALL keys first, then fallback models
        # across all keys. Cap total attempts to avoid very long gateway waits.
        # -----------------------------------------------------------------------
        _model_candidates = list(dict.fromkeys([GEMINI_MODEL] + GEMINI_FALLBACK_MODELS))[:2]
        _key_count = len(_gemini_clients)

        # Flatten: [(client0, model0), (client1, model0), ..., (client0, model1), ...]
        _attempts = [
            (ki, _gemini_clients[ki], model)
            for model in _model_candidates
            for ki in range(_key_count)
        ]

        response = None
        used_model = None
        used_key_idx = None
        last_gemini_exc = None

        for key_idx, client, candidate_model in _attempts:
            try:
                response = client.models.generate_content(
                    model=candidate_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.1,
                        max_output_tokens=effective_output_tokens,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                if response and response.text:
                    used_model = candidate_model
                    used_key_idx = key_idx
                    break
                response = None  # empty response — try next
            except Exception as exc:
                last_gemini_exc = exc
                err_code = getattr(exc, "code", getattr(exc, "status_code", None))
                err_str = str(exc).upper()
                _auth_fail = (
                    err_code in _AUTH_CODES
                    or "API_KEY_INVALID" in err_str
                    or "PERMISSION_DENIED" in err_str
                )
                print(
                    f"[Gemini] key[{key_idx}] model={candidate_model} "
                    f"code={err_code} {'(auth fail)' if _auth_fail else ''}: {exc}"
                )
                if _auth_fail:
                    # This key is dead for all models — skip remaining attempts with it
                    _attempts = [(ki, c, m) for ki, c, m in _attempts if ki != key_idx]
                # quota/transient — continue to next (key, model)
                continue

        if response and response.text:
            cleaned_text = _clean_truncated_tail(response.text)
            key_label = f"key[{used_key_idx}]" if _key_count > 1 else "key[0]"
            return cleaned_text, retrieved_chunks, f"Google Gemini ({used_model}, {key_label})"

        # If all Gemini models failed, attempt Groq fallback
        if allow_groq_fallback and GROQ_API_KEY:
            if image_attachments:
                raise RuntimeError(
                    "All Gemini models were unavailable, and image attachments cannot be processed via Groq fallback."
                ) from last_gemini_exc
            print("[Fallback] Switching to Groq fallback...")
            try:
                answer = self._query_groq(
                    user_prompt,
                    max_output_tokens=max_output_tokens,
                )
                return answer, retrieved_chunks, f"Groq ({GROQ_MODEL})"
            except Exception as groq_exc:
                print(f"[Groq Error] Fallback failed: {groq_exc}")
                raise RuntimeError(
                    f"Gemini failed ({last_gemini_exc}) and Groq fallback also failed ({groq_exc})"
                ) from groq_exc

        if last_gemini_exc:
            raise last_gemini_exc
        raise RuntimeError("All LLM providers failed to return a response.")

    def _query_groq(self, user_prompt: str, max_output_tokens: int) -> str:
        if not GROQ_API_KEY:
            raise RuntimeError(
                "Gemini had a transient failure and Groq fallback was authorized, "
                "but GROQ_API_KEY is not configured."
            )

        # Transient connection/SSL errors that warrant a single retry
        _GROQ_TRANSIENT = (
            requests.exceptions.SSLError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        )

        acquired_token_count = 0

        # Safe-fit user_prompt if entire request exceeds effective limit
        sys_tokens = _estimate_tokens(SYSTEM_PROMPT)
        effective_limit = getattr(_groq_token_rate_limiter, "_effective_limit", GROQ_TOKENS_PER_MINUTE)
        max_allowed_for_prompt = min(2500, max(500, effective_limit - sys_tokens - max_output_tokens - 100))
        if _estimate_tokens(user_prompt) > max_allowed_for_prompt:
            ratio = max_allowed_for_prompt / max(1, _estimate_tokens(user_prompt))
            user_prompt = user_prompt[: int(len(user_prompt) * ratio * 0.95)]

        def request_completion(
            model_name: str,
            token_limit: int,
            reasoning_effort: str,
        ) -> Dict[str, Any]:
            nonlocal acquired_token_count
            last_exc: Exception | None = None
            for attempt in range(2):  # attempt 0 = first try, attempt 1 = retry
                if attempt > 0:
                    wait = 3
                    print(f"[Groq] Connection error on attempt {attempt}; retrying in {wait}s...")
                    time.sleep(wait)
                try:
                    estimated_tokens = (
                        _estimate_tokens(SYSTEM_PROMPT)
                        + _estimate_tokens(user_prompt)
                        + token_limit
                    )
                    if estimated_tokens > acquired_token_count:
                        _groq_token_rate_limiter.acquire(estimated_tokens - acquired_token_count)
                        acquired_token_count = estimated_tokens
                    response = requests.post(
                        GROQ_CHAT_COMPLETIONS_URL,
                        headers={
                            "Authorization": f"Bearer {GROQ_API_KEY}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": model_name,
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user_prompt},
                            ],
                            "temperature": 0.1,
                            "max_completion_tokens": token_limit,
                            "reasoning_effort": reasoning_effort,
                            "include_reasoning": False,
                        },
                        # connect=30s (SSL handshake included), read=120s
                        timeout=(30, 120),
                    )
                    _groq_token_rate_limiter.observe_headers(
                        getattr(response, "headers", {})
                    )
                    response.raise_for_status()
                    break  # success — exit retry loop
                except _GROQ_TRANSIENT as exc:
                    last_exc = exc
                    if attempt == 0:
                        continue  # retry once
                    raise RuntimeError(
                        f"Groq connection failed after retry (SSL/network): {exc}"
                    ) from exc
                except requests.HTTPError as exc:
                    # HTTP error response (4xx/5xx) — no point retrying
                    status_code = (
                        exc.response.status_code if exc.response is not None else "unknown"
                    )
                    body = exc.response.text[:300] if exc.response is not None else str(exc)
                    raise RuntimeError(
                        f"Groq returned HTTP {status_code}: {body}"
                    ) from exc
                except requests.RequestException as exc:
                    # Other request errors (not SSL/connection) — no retry
                    raise RuntimeError(f"Groq request failed: {exc}") from exc

            try:
                payload = response.json()
            except ValueError as exc:
                raise RuntimeError("Groq returned invalid JSON.") from exc

            if not isinstance(payload, dict):
                raise RuntimeError("Groq returned an invalid response structure.")
            return payload

        def answer_from(payload: Dict[str, Any]) -> tuple[str, str | None]:
            choices = payload.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                return "", None

            choice = choices[0]
            message = choice.get("message")
            finish_reason = choice.get("finish_reason")

            # Extract content from message
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, str) and content.strip():
                answer = content.strip()
            elif isinstance(content, list):
                answer = "".join(
                    part["text"]
                    for part in content
                    if isinstance(part, dict) and isinstance(part.get("text"), str)
                ).strip()
            else:
                answer = ""

            # Fallback to reasoning if content is empty
            if not answer and isinstance(message, dict):
                reasoning = message.get("reasoning", "")
                if isinstance(reasoning, str) and reasoning.strip():
                    answer = reasoning.strip()

            return answer, finish_reason if isinstance(finish_reason, str) else None

        models_to_try = [GROQ_MODEL] + [m for m in GROQ_FALLBACK_MODELS if m != GROQ_MODEL]
        last_error = None

        for model_name in models_to_try:
            try:
                first_limit = min(max_output_tokens, 950) if "qwen" in model_name.lower() else max_output_tokens
                initial_effort = "medium" if max_output_tokens <= 400 else "low"
                payload = request_completion(model_name, first_limit, initial_effort)
                answer, finish_reason = answer_from(payload)
                if answer and finish_reason != "length":
                    return _clean_truncated_tail(answer)

                prompt_cost = sys_tokens + _estimate_tokens(user_prompt)
                margin = max(500, getattr(_groq_token_rate_limiter, "_effective_limit", GROQ_TOKENS_PER_MINUTE) - prompt_cost - 50)
                retry_token_limit = (
                    max(max_output_tokens + 256, 1024)
                    if max_output_tokens <= 400
                    else min(max(max_output_tokens + 1024, 2500), margin)
                )
                if "qwen" in model_name.lower():
                    retry_token_limit = min(retry_token_limit, 950)

                print(
                    f"[Groq Warning] Response truncated or empty "
                    f"(finish_reason={finish_reason}); retrying with {retry_token_limit} tokens."
                )
                try:
                    retry_payload = request_completion(model_name, retry_token_limit, "low")
                    retry_answer, retry_finish = answer_from(retry_payload)
                    if retry_answer:
                        return _clean_truncated_tail(retry_answer)
                except Exception as retry_exc:
                    print(f"[Groq Warning] Retry failed: {retry_exc}")
                    if answer:
                        return _clean_truncated_tail(answer)
                    raise retry_exc

                if answer:
                    return _clean_truncated_tail(answer)

                usage = retry_payload.get("usage")
                completion_tokens = (
                    usage.get("completion_tokens")
                    if isinstance(usage, dict)
                    else None
                )
                last_error = RuntimeError(
                    "Groq returned an empty fallback response after retry "
                    f"(finish_reason={retry_finish or finish_reason}, completion_tokens={completion_tokens})."
                )
            except Exception as model_exc:
                print(f"[Groq Warning] Model {model_name} failed: {model_exc}. Trying next candidate...")
                last_error = model_exc
                continue

        if last_error:
            raise last_error
        raise RuntimeError("All Groq fallback models failed to return a response.")


def load_and_index_documents(data_dir="data", persist_dir="./storage"):
    """Compatibility function matching previous API."""
    return LocalFAISSRag(data_dir=data_dir, persist_dir=persist_dir)


if __name__ == "__main__":
    print("=" * 60)
    print("Initializing Legal RAG (Gemini API + Local FAISS)...")
    print("=" * 60)
    rag = LocalFAISSRag()

    sample_query = "A person intentionally causes hurt to another person using a dangerous weapon."
    print(f"\n--- Testing Query ---")
    print(f"Scenario: {sample_query}\n")
    answer = rag.query_gemini(sample_query, top_k=4)
    print(f"Gemini Legal Analysis:\n{answer}")
