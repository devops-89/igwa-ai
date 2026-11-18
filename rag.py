import json
import os
import re
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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment or .env file.")

# Gemini Client (with aggressive 10s timeout to fast-failover to Groq if Google API is congested)
gemini_client = genai.Client(api_key=GEMINI_API_KEY, http_options=types.HttpOptions(timeout=10.0))
GEMINI_MODEL = "gemini-flash-latest"
GEMINI_FALLBACK_MODELS = ["gemini-3.5-flash-lite"]
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("GROUQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
TRANSIENT_GEMINI_STATUS_CODES = {404, 408, 429, 500, 502, 503, 504}

# Initialize Local Embedding Model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
print(f"Loading local embedding model: {EMBEDDING_MODEL_NAME}...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()

SYSTEM_PROMPT = (
    "You are an expert Indian-law legal and police investigation decision-support assistant "
    "for investigating officers (IOs) and legal reviewers. "
    "\n\n"
    "LANGUAGE RULE: Write every user-facing response entirely in formal, professional Hindi using "
    "Devanagari script, matching the register used by Indian police departments and courts. "
    "Preserve exact English abbreviations (e.g. BNS 2023, BNSS 2023, BSB 2023, IPC, CrPC, IT Act, "
    "NDPS Act, FIR, IO, CCTV, DVR, IMEI) and exact statutory section numbers and source filenames. "
    "\n\n"
    "STRICT FACTUAL DISCIPLINE (NEVER GENERATE OR INVENT CASE FACTS): "
    "- Treat the user-provided scenario as the ONLY ground truth for case facts. "
    "- NEVER invent, assume, or fabricate any facts, motives, injuries, weapons, prior criminal records, "
    "names, or events that the user did not explicitly state. "
    "- Do not assume guilt or intent from circumstantial details alone. "
    "\n\n"
    "TRUSTED STATUTORY LEGAL KNOWLEDGE: "
    "- Base your legal analysis on the provided retrieved documents first. "
    "- If a relevant statute, section number, First Schedule classification (Cognizable/Non-cognizable, "
    "Bailable/Non-bailable, Triable by court), or procedural rule is not in the retrieved text, "
    "you MUST answer that using your trusted, authoritative knowledge of Indian criminal law (BNS 2023, "
    "BNSS 2023, BSB 2023, IT Act 2000, Arms Act, etc.) without creating or generating case facts. "
    "- Do NOT refuse or say 'Information not found' for standard, well-established Indian statutory provisions. "
    "\n\n"
    "TOKEN PRECISION & TWO RESPONSE MODES: "
    "- Write with high token efficiency: crisp, dense, structured, and free of filler prose or repetitive disclaimers. "
    "\n"
    "MODE 1: INITIAL CASE SCENARIO (FIRST RESPONSE TO A CASE)\n"
    "When analyzing a new incident or FIR, provide a complete and token-precise analysis using EXACTLY these four headings:\n"
    "### 1. मामले का सारांश\n"
    "(Concise summary of stated facts in 2-3 bullets + crucial missing/unverified facts needed for IO).\n"
    "### 2. अपराध का वर्गीकरण\n"
    "(Compact Markdown table: 'अधिनियम एवं धारा' | 'अपराध का वर्गीकरण (संज्ञेय/असंज्ञेय)' | 'जमानतीय स्थिति' | 'विचारणीय न्यायालय').\n"
    "### 3. लागू विधिक धाराएँ एवं दंड\n"
    "(Numbered list of applicable offences, essential ingredients applied to stated facts, and statutory punishment).\n"
    "### 4. विवेचना प्रक्रिया (SOP)\n"
    "(Prioritized actionable steps for IO: Crime scene & physical evidence, CCTV/Digital evidence & Sec 63 certificate, "
    "Witness examination under BNSS Sec 180, Seizure memo & audio-video compliance under BNSS Sec 105, Notice/Arrest rules under BNSS Sec 35).\n"
    "\n"
    "MODE 2: REALTIME CHATBOT (FOLLOW-UP RESPONSES)\n"
    "When answering follow-up questions, clarifications, or drafting requests on the ongoing case:\n"
    "- DO NOT repeat the 4-step format! Do NOT generate summary, full classification table, or general SOP.\n"
    "- Act as a realtime conversational assistant: answer the officer's specific question directly, concisely, and practically.\n"
    "- State the exact legal section, procedural condition, or case diary draft points requested."
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

    def ingest_ocr_document(
        self,
        document_name: str,
        chunks_data: List[Dict[str, Any]],
        raw_json: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Dynamically ingests OCR text chunks into the active FAISS vector index."""
        if not chunks_data:
            raise ValueError("No text chunks provided for ingestion.")

        texts = [c["text"] for c in chunks_data]
        new_embeddings = embedding_model.encode(texts, batch_size=64, show_progress_bar=False)
        new_embeddings = self._normalize(np.array(new_embeddings, dtype="float32"))

        if self.index is None:
            self.index = faiss.IndexFlatIP(EMBEDDING_DIM)

        self.index.add(new_embeddings)
        self.chunks.extend(chunks_data)
        if document_name not in self.indexed_files:
            self.indexed_files.append(document_name)

        self._save_state()

        # Archive raw OCR JSON for persistence across future index builds
        if raw_json:
            try:
                ocr_archive_dir = self.data_dir / "ocr_ingested"
                ocr_archive_dir.mkdir(parents=True, exist_ok=True)
                safe_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", document_name)
                if not safe_name.endswith(".json"):
                    safe_name += ".json"
                archive_path = ocr_archive_dir / safe_name
                archive_path.write_text(
                    json.dumps(raw_json, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as e:
                print(f"Warning: Failed to archive raw OCR JSON: {e}")

        print(f"Successfully ingested OCR document '{document_name}' ({len(chunks_data)} chunks). Total vectors: {self.index.ntotal}")
        return {
            "status": "success",
            "document_name": document_name,
            "chunks_indexed": len(chunks_data),
            "total_vectors": self.index.ntotal,
        }

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
        if not schedule_path.exists():
            return ""

        schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
        rows = schedule.get("rows", {})
        searchable_text = " ".join(
            str(chunk.get("text", "")) for chunk in retrieved_chunks
        )
        explicit_bns_references = re.findall(
            r"\bBNS\s+(\d{1,3}(?:\([a-zA-Z0-9]+\))*)",
            scenario,
            flags=re.IGNORECASE,
        )
        searchable_text += " " + " ".join(explicit_bns_references)
        matched_rows = []
        matched_roots = set()
        for section in rows:
            root = section.split("(", 1)[0]
            if re.search(rf"(?<!\d){re.escape(root)}(?!\d)", searchable_text):
                matched_roots.add(root)

        for section, classification in rows.items():
            if section.split("(", 1)[0] in matched_roots:
                matched_rows.append(f"{section}: {json.dumps(classification, ensure_ascii=False)}")

        if not matched_rows:
            return ""
        schedule_name = schedule.get("schedule", "BNSS First Schedule")
        source = schedule.get("source", "official schedule extract")
        printed_pages = schedule.get("printed_pages", [])
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
                f"INVESTIGATING OFFICER'S FOLLOW-UP QUESTION:\n{follow_up}\n\n"
                "INSTRUCTION: REALTIME CHAT RESPONSE (MODE 2).\n"
                "- Directly, concisely, and practically answer the officer's specific question in professional Hindi.\n"
                "- DO NOT repeat the 4-step format (do NOT generate summary, full classification table, or general SOP headings).\n"
                "- Be token-precise: focus strictly on answering the query with relevant legal sections, conditions, or procedural rules.\n"
                "- Never fabricate case facts. If statutory schedules or provisions are missing from retrieved text, supply them accurately using trusted Indian statutory law."
            )
        else:
            user_prompt = (
                f"LEGAL CONTEXT (from retrieved acts & schedules):\n{context_text}\n\n"
                f"NEW CASE INCIDENT / SCENARIO:\n{scenario}\n\n"
                "INSTRUCTION: INITIAL CASE ANALYSIS (MODE 1 - 1ST RESPONSE).\n"
                "- Provide a complete, structured analysis using EXACTLY these four headings:\n"
                "  ### 1. मामले का सारांश\n"
                "  ### 2. अपराध का वर्गीकरण\n"
                "  ### 3. लागू विधिक धाराएँ एवं दंड\n"
                "  ### 4. विवेचना प्रक्रिया (SOP)\n"
                "- TOKEN PRECISION: Keep each section dense, compact, and structured (bullet points and Markdown table). Avoid fluff, unnecessary boilerplate, or repeating the scenario verbatim.\n"
                "- FACTUAL DISCIPLINE: Treat the user's scenario as the ONLY case facts. Never fabricate or assume unstated facts.\n"
                "- TRUSTED LEGAL KNOWLEDGE: If any relevant section, First Schedule classification (cognizable/bailable), or procedure is absent from the retrieved excerpts, supply it accurately using trusted Indian statutory law (BNS 2023, BNSS 2023, BSB 2023, etc.) without leaving gaps."
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

        # Build list of Gemini models to attempt
        gemini_candidates = [GEMINI_MODEL] + [m for m in GEMINI_FALLBACK_MODELS if m != GEMINI_MODEL]
        seen = set()
        models_to_try = [m for m in gemini_candidates if not (m in seen or seen.add(m))]

        response = None
        used_model = None
        last_gemini_exc = None

        for candidate_model in models_to_try:
            try:
                response = gemini_client.models.generate_content(
                    model=candidate_model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.1,
                        max_output_tokens=1800,
                    ),
                )
                if response and response.text:
                    used_model = candidate_model
                    break
            except Exception as exc:
                last_gemini_exc = exc
                err_code = getattr(exc, "code", getattr(exc, "status_code", None))
                print(f"[Gemini Warning] Model {candidate_model} failed (code={err_code}): {exc}. Trying next candidate...")
                continue

        if response and response.text:
            return response.text, retrieved_chunks, f"Google Gemini ({used_model})"

        # If all Gemini models failed, attempt Groq fallback
        if allow_groq_fallback and GROQ_API_KEY:
            if image_attachments:
                raise RuntimeError(
                    "All Gemini models were unavailable, and image attachments cannot be processed via Groq fallback."
                ) from last_gemini_exc
            print("[Fallback] Switching to Groq fallback...")
            try:
                answer = self._query_groq(user_prompt)
                return answer, retrieved_chunks, f"Groq ({GROQ_MODEL})"
            except Exception as groq_exc:
                print(f"[Groq Error] Fallback failed: {groq_exc}")
                raise RuntimeError(
                    f"Gemini failed ({last_gemini_exc}) and Groq fallback also failed ({groq_exc})"
                ) from groq_exc

        if last_gemini_exc:
            raise last_gemini_exc
        raise RuntimeError("All LLM providers failed to return a response.")

    def _query_groq(self, user_prompt: str) -> str:
        if not GROQ_API_KEY:
            raise RuntimeError(
                "Gemini had a transient failure and Groq fallback was authorized, "
                "but GROQ_API_KEY is not configured."
            )

        try:
            response = requests.post(
                GROQ_CHAT_COMPLETIONS_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.1,
                    "max_completion_tokens": 4096,
                },
                timeout=(10, 90),
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response else "unknown"
            raise RuntimeError(
                f"Groq fallback request failed with HTTP {status_code}."
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"Groq fallback request failed: {exc}") from exc

        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Groq returned an invalid fallback response.")
        choices = payload.get("choices", [])
        answer = choices[0].get("message", {}).get("content") if choices else None
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("Groq returned an empty fallback response.")
        return answer


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
