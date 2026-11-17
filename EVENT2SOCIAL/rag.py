"""Document ingestion and dense-vector retrieval for EVENT2SOCIAL."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import re
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

import numpy as np


class ProviderError(RuntimeError):
    """An embedding provider could not complete the requested operation."""


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-2"
GEMINI_EMBEDDING_FORMAT = "gemini-embedding-2-search-v1"


def _pytesseract():
    """Load pytesseract and locate Tesseract even when its install directory is not on PATH."""
    import pytesseract
    current = str(pytesseract.pytesseract.tesseract_cmd)
    candidates = []
    configured = os.getenv("TESSERACT_CMD", "").strip()
    if configured:
        explicit = Path(configured)
        if explicit.is_file() and shutil.which(str(explicit)):
            pytesseract.pytesseract.tesseract_cmd = str(explicit)
            return pytesseract
    if shutil.which(current):
        return pytesseract
    candidates.extend([
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(os.getenv("ProgramFiles", r"C:\Program Files")) / "Tesseract-OCR" / "tesseract.exe",
        Path(os.getenv("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tesseract.exe",
    ])
    for candidate in candidates:
        if candidate.is_file() and shutil.which(str(candidate)):
            pytesseract.pytesseract.tesseract_cmd = str(candidate)
            return pytesseract
    return pytesseract


def _secret(secrets: Any, name: str, default: str = "") -> str:
    if secrets is None:
        return default
    try:
        return str(secrets.get(name, default)).strip()
    except Exception:
        return default


def normalize_provider(provider: str | None) -> str:
    value = (provider or "gemini").strip().lower()
    if value in {"gemini", "google", "google gemini"}:
        return "gemini"
    if value in {"groq", "groqcloud"}:
        return "groq"
    if value in {"openai", "openai-compatible", "openai compatible"}:
        return "openai"
    raise ValueError("Choose Groq, Gemini, or OpenAI as the AI provider.")


def configured_provider(secrets: Any = None) -> str:
    value = os.getenv("EVENT2SOCIAL_PROVIDER", "").strip() or _secret(secrets, "EVENT2SOCIAL_PROVIDER")
    if not value:
        groq_names = ("GROQ_API_KEY",)
        openai_names = ("OPENAI_API_KEY", "OPENAI_EMBEDDING_API_KEY", "OPENAI_GENERATION_API_KEY")
        gemini_names = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_EMBEDDING_API_KEY", "GEMINI_GENERATION_API_KEY")
        has_groq = any(os.getenv(name, "").strip() or _secret(secrets, name) for name in groq_names)
        has_openai = any(os.getenv(name, "").strip() or _secret(secrets, name) for name in openai_names)
        has_gemini = any(os.getenv(name, "").strip() or _secret(secrets, name) for name in gemini_names)
        value = "groq" if has_groq else ("openai" if has_openai and not has_gemini else "gemini")
    return normalize_provider(value)


def configured_embedding_provider(secrets: Any = None, generation_provider: str | None = None) -> str:
    """Keep RAG embeddings independent because Groq currently exposes no embeddings endpoint."""
    explicit = _setting("EVENT2SOCIAL_EMBEDDING_PROVIDER") or _secret(secrets, "EVENT2SOCIAL_EMBEDDING_PROVIDER")
    if explicit:
        provider = normalize_provider(explicit)
        if provider == "groq":
            raise ValueError("Groq does not currently provide an embeddings endpoint. Choose Gemini or OpenAI for RAG embeddings.")
        return provider
    generation_provider = normalize_provider(generation_provider or configured_provider(secrets))
    if generation_provider != "groq":
        return generation_provider
    gemini_names = ("GEMINI_EMBEDDING_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")
    openai_names = ("OPENAI_EMBEDDING_API_KEY", "OPENAI_API_KEY")
    has_gemini = any(os.getenv(name, "").strip() or _secret(secrets, name) for name in gemini_names)
    has_openai = any(os.getenv(name, "").strip() or _secret(secrets, name) for name in openai_names)
    if has_gemini:
        return "gemini"
    if has_openai:
        return "openai"
    # Keep the configured embedding model stable; with no key RAG stays unavailable
    # but can be re-enabled by restoring the selected provider's embedding credential.
    return "gemini"


def embedding_request_text(text: str, *, provider: str, role: str,
                           title: str | None = None, model: str = GEMINI_EMBEDDING_MODEL) -> str:
    """Format text consistently for a provider's document-search embedding model."""
    provider = normalize_provider(provider)
    if provider == "gemini" and model == GEMINI_EMBEDDING_MODEL:
        if role == "query":
            return f"task: search result | query: {text}"
        if role == "document":
            return f"title: {title or 'none'} | text: {text}"
        raise ValueError("Embedding role must be 'query' or 'document'.")
    return text


def embedding_index_matches(index: dict[str, Any], *, provider: str, base_url: str,
                            model: str, source_hashes: dict[str, str]) -> bool:
    """Reject incompatible indexes and changed/removed indexed sources; ignore unindexed media additions."""
    provider = normalize_provider(provider)
    expected_format = GEMINI_EMBEDDING_FORMAT if provider == "gemini" and model == GEMINI_EMBEDDING_MODEL else "plain-v1"
    return (bool(index.get("embedding_model"))
            and index.get("embedding_provider") == provider
            and index.get("embedding_base_url") == base_url.rstrip("/")
            and index.get("embedding_model") == model
            and index.get("embedding_format") == expected_format
            and index.get("source_policy") == "document-text-v1"
            and bool(index.get("source_hashes"))
            and all(source_hashes.get(name) == digest for name, digest in index.get("source_hashes", {}).items()))


def _provider_http_message(exc: urllib.error.HTTPError, service: str, provider: str,
                           api_key: str = "") -> str:
    status = exc.code
    if status in (401, 403):
        advice = "Check the API key and confirm it is enabled for this service."
    elif status == 404:
        advice = "Check the configured endpoint and model name."
    elif status == 429:
        advice = "The provider rate limit or quota was reached; check quota and retry later."
    elif status >= 500:
        advice = "The provider is temporarily unavailable; retry later."
    else:
        advice = "Check the request configuration and selected model."
    detail = ""
    try:
        body = json.loads(exc.read())
        detail = str(body.get("error", {}).get("message", ""))
        if api_key:
            detail = detail.replace(api_key, "[redacted]")
        detail = re.sub(r"(?i)(api[_ -]?key|token|authorization)\s*[:=]\s*[^\s,;]+", r"\1=[redacted]", detail)[:240]
    except Exception:
        pass
    suffix = f" Provider message: {detail}" if detail else ""
    return f"{provider.title()} {service} request failed (HTTP {status}). {advice}{suffix}"


def extract_document(name: str, content: bytes) -> list[dict[str, Any]]:
    """Extract text while retaining page metadata when the format provides it."""
    suffix = Path(name).suffix.lower()
    if suffix == ".txt":
        return [{"page_number": None, "text": content.decode("utf-8-sig", errors="replace")}]
    if suffix == ".pdf":
        from pypdf import PdfReader

        try:
            pages = PdfReader(io.BytesIO(content)).pages
            units = []
            pdf = None
            renderer_error = ""
            ocr_error = ""
            try:
                import pypdfium2 as pdfium
            except Exception as exc:
                pdfium = None
                renderer_error = f"PDF rendering is unavailable: {exc}"
            try:
                _pytesseract().get_tesseract_version()
            except Exception as exc:
                ocr_error = ("The Tesseract OCR executable is not installed or is not on PATH."
                             if exc.__class__.__name__ == "TesseractNotFoundError"
                             else f"OCR is unavailable: {exc}")
            for i, page in enumerate(pages):
                text = page.extract_text() or ""
                error = ""
                if not text.strip():
                    if renderer_error:
                        error = renderer_error
                    elif ocr_error:
                        error = ocr_error
                    else:
                        try:
                            if pdf is None:
                                pdf = pdfium.PdfDocument(content)
                            bitmap = pdf[i].render(scale=2).to_pil()
                            text = _ocr_image(bitmap)
                        except Exception as exc:
                            text = ""
                            error = str(exc)
                units.append({"page_number": i + 1, "text": text, "error": error})
            return units
        except Exception as exc:
            raise ValueError(f"Could not read PDF: {exc}") from exc
    if suffix == ".docx":
        from docx import Document

        try:
            doc = Document(io.BytesIO(content))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                paragraphs.extend(" | ".join(cell.text.strip() for cell in row.cells)
                                  for row in table.rows)
            return [{"page_number": None, "text": "\n".join(paragraphs)}]
        except Exception as exc:
            raise ValueError(f"Could not read DOCX: {exc}") from exc
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        try:
            from PIL import Image
            return [{"page_number": None, "text": _ocr_image(Image.open(io.BytesIO(content)))}]
        except Exception as exc:
            return [{"page_number": None, "text": "", "error": f"Local OCR could not read image text: {exc}"}]
    raise ValueError("Supported formats are TXT, PDF, DOCX, JPG, PNG, and WEBP.")


def render_scanned_pdf_pages(name: str, content: bytes, max_pages: int = 3) -> list[dict[str, Any]]:
    """Render pages where the PDF text layer (and available OCR) yielded little text."""
    if Path(name).suffix.lower() != ".pdf":
        return []
    try:
        import pypdfium2 as pdfium
        from pypdf import PdfReader
    except ImportError as exc:
        raise ValueError("PDF page vision needs pypdfium2 installed in the application environment.") from exc
    reader = PdfReader(io.BytesIO(content))
    pdf = pdfium.PdfDocument(content)
    candidates = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if len(text) < 20:
            candidates.append(i)
    if len(candidates) > max_pages:
        if max_pages == 1:
            candidates = [candidates[0]]
        else:
            positions = [round(i * (len(candidates) - 1) / (max_pages - 1)) for i in range(max_pages)]
            candidates = [candidates[position] for position in positions]
    rendered = []
    for i in candidates:
        bitmap = pdf[i].render(scale=1.8).to_pil()
        bitmap.thumbnail((1600, 1600))
        ocr_text, ocr_error = "", ""
        try:
            ocr_text = _ocr_image(bitmap)
        except Exception as exc:
            ocr_error = str(exc)
        out = io.BytesIO()
        bitmap.convert("RGB").save(out, format="JPEG", quality=84, optimize=True)
        rendered.append({"page_number": i + 1, "content": out.getvalue(),
                         "ocr_text": ocr_text, "ocr_error": ocr_error})
    return rendered


def _ocr_image(image: Any) -> str:
    """Use local Tesseract OCR when installed; never send files to a service implicitly."""
    try:
        return _pytesseract().image_to_string(image)
    except ImportError as exc:
        raise ValueError("OCR needs pytesseract and the Tesseract OCR application. Install both to read scanned pages and images.") from exc
    except Exception as exc:
        if exc.__class__.__name__ == "TesseractNotFoundError":
            raise ValueError("The Tesseract OCR executable is not installed or is not on PATH.") from exc
        raise ValueError(f"Local OCR could not read this image ({exc.__class__.__name__}).") from exc


def split_chunks(text: str, size: int = 180, overlap: int = 35) -> list[str]:
    words = text.split()
    if not words:
        return []
    stride = max(1, size - overlap)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), stride)]


def _setting(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def embedding_config(secrets: Any = None, provider: str | None = None) -> tuple[str, str, str]:
    """Resolve provider credentials and embedding settings from env or Streamlit secrets."""
    provider = normalize_provider(provider or configured_provider(secrets))
    if provider == "groq":
        raise ProviderError("Groq does not currently expose an embeddings endpoint. Configure Gemini or OpenAI separately for document search.")
    if provider == "gemini":
        key = (_setting("GEMINI_EMBEDDING_API_KEY") or _setting("GEMINI_API_KEY") or _setting("GOOGLE_API_KEY")
               or _secret(secrets, "GEMINI_EMBEDDING_API_KEY") or _secret(secrets, "GEMINI_API_KEY") or _secret(secrets, "GOOGLE_API_KEY"))
        base = (_setting("GEMINI_BASE_URL") or _secret(secrets, "GEMINI_BASE_URL") or GEMINI_BASE_URL).rstrip("/")
        model = (_setting("GEMINI_EMBEDDING_MODEL") or _secret(secrets, "GEMINI_EMBEDDING_MODEL")
                 or GEMINI_EMBEDDING_MODEL)
    else:
        key = (_setting("OPENAI_EMBEDDING_API_KEY") or _setting("OPENAI_API_KEY")
               or _secret(secrets, "OPENAI_EMBEDDING_API_KEY") or _secret(secrets, "OPENAI_API_KEY"))
        base = (_setting("OPENAI_BASE_URL") or _secret(secrets, "OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        model = (_setting("OPENAI_EMBEDDING_MODEL") or _secret(secrets, "OPENAI_EMBEDDING_MODEL")
                 or "text-embedding-3-small")
    return key, base, model


def embed_texts(texts: list[str], *, api_key: str, base_url: str,
                model: str = "text-embedding-3-small", provider: str = "openai") -> list[list[float]]:
    """Request real dense embeddings from an OpenAI-compatible API."""
    provider = normalize_provider(provider)
    if provider == "groq":
        raise ProviderError("Groq does not currently expose an embeddings endpoint. Configure Gemini or OpenAI separately for document search.")
    if not api_key:
        env_name = "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
        raise ProviderError(f"{provider.title()} embeddings are not configured. Ask the administrator to set {env_name} on the server.")
    if not texts:
        return []
    vectors: list[list[float] | None] = [None] * len(texts)
    # Embedding 2 aggregates a list into one vector. Send each text separately;
    # OpenAI and other compatible providers keep the existing 64-text batching.
    single_input = provider == "gemini" and model == GEMINI_EMBEDDING_MODEL
    batch_size = 1 if single_input else 64
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        payload = json.dumps({"model": model, "input": batch}).encode("utf-8")
        if single_input:
            payload = json.dumps({"model": model, "input": batch[0]}).encode("utf-8")
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/embeddings", data=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            raise ProviderError(_provider_http_message(exc, "embedding", provider, api_key)) from None
        except (urllib.error.URLError, TimeoutError):
            raise ProviderError(f"{provider.title()} embedding connection failed. Check network access and endpoint, then retry.") from None
        except Exception as exc:
            raise ProviderError(f"Embedding service returned an error ({exc.__class__.__name__}).") from exc
        data = result.get("data", [])
        if len(data) != len(batch):
            raise ProviderError("Embedding service returned an unexpected number of vectors.")
        for item in data:
            index = item.get("index")
            vector = item.get("embedding")
            if not isinstance(index, int) or index < 0 or index >= len(batch) or not isinstance(vector, list):
                raise ProviderError("Embedding service returned malformed vector data.")
            values = [float(x) for x in vector]
            if not values or not all(math.isfinite(x) for x in values):
                raise ProviderError("Embedding service returned an empty or invalid vector.")
            norm = math.sqrt(sum(x * x for x in values))
            if norm == 0:
                raise ProviderError("Embedding service returned a zero vector.")
            vectors[start + index] = [x / norm for x in values]
    if any(vector is None for vector in vectors):
        raise ProviderError("Embedding service response omitted a vector.")
    dimensions = len(vectors[0])
    if any(len(vector) != dimensions for vector in vectors if vector is not None):
        raise ProviderError("Embedding service returned vectors with inconsistent dimensions.")
    return [vector for vector in vectors if vector is not None]


def prepare_chunks(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for document in documents:
        for index, text in enumerate(split_chunks(document["text"])):
            rows.append({"source_filename": document["source_filename"],
                         "source_hash": document["source_hash"],
                         "page_number": document.get("page_number"),
                         "chunk_index": index, "text": text})
    return rows


def retrieve(query: str, rows: list[dict[str, Any]], query_vector: list[float], top_k: int = 4) -> list[dict[str, Any]]:
    """Cosine-rank stored dense embeddings against a dense query embedding."""
    if not rows:
        return []
    q = np.asarray(query_vector, dtype=np.float32)
    if q.ndim != 1 or not np.isfinite(q).all() or np.linalg.norm(q) == 0:
        raise ValueError("The query embedding is invalid.")
    q /= np.linalg.norm(q)
    matrix = np.asarray([row["vector"] for row in rows], dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != q.shape[0]:
        raise ValueError("Stored vectors do not match the query embedding dimensions. Reprocess the knowledge base.")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0):
        raise ValueError("The knowledge base contains an invalid zero vector. Reprocess it.")
    scores = (matrix @ q) / norms
    ranked = np.argsort(scores)[::-1][:max(1, top_k)]
    return [{**rows[int(i)], "score": float(scores[i])} for i in ranked]


def file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
