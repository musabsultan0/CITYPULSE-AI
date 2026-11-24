from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from backend.app.core.config import ROOT_DIR, settings

try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False

KB_DIR = ROOT_DIR / "knowledge_base"
KB_DIR.mkdir(parents=True, exist_ok=True)


def _persist_dir() -> Path:
    p = Path(settings.rag_persist_directory)
    if not p.is_absolute():
        p = ROOT_DIR / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def _extract_documents() -> list[tuple[str, str]]:
    docs: list[tuple[str, str]] = []
    all_files = sorted(
        p for p in KB_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in {".txt", ".md", ".pdf"}
    )
    # Exclude repo helper README.md if actual SOP documents exist
    non_readme = [p for p in all_files if p.name.lower() != "readme.md"]
    target_files = non_readme if non_readme else all_files

    for p in target_files:
        try:
            rel = str(p.relative_to(ROOT_DIR).as_posix())
            if p.suffix.lower() == ".pdf":
                from pypdf import PdfReader
                reader = PdfReader(str(p))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            else:
                text = p.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                docs.append((rel, text))
        except Exception:
            continue
    return docs


def _chunk_text(source: str, text: str, chunk_size: int = 500, overlap: int = 80) -> list[dict[str, Any]]:
    """Split document into overlapping semantic chunks while tracking markdown headings."""
    lines = text.splitlines()
    sections: list[tuple[str, str]] = []
    current_heading = "General"
    buffer: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            if buffer:
                sections.append((current_heading, "\n".join(buffer).strip()))
                buffer = []
            current_heading = stripped.lstrip("#").strip() or current_heading
            buffer.append(stripped)
        else:
            buffer.append(line)
    if buffer:
        sections.append((current_heading, "\n".join(buffer).strip()))

    chunks: list[dict[str, Any]] = []
    chunk_idx = 0
    step = max(chunk_size - overlap, 100)

    for heading, body in sections:
        clean_body = re.sub(r"\n{3,}", "\n\n", body).strip()
        if not clean_body:
            continue
        if len(clean_body) <= chunk_size:
            chunks.append({
                "chunk_id": f"{source}#chunk-{chunk_idx}",
                "source": source,
                "section": heading,
                "text": clean_body,
            })
            chunk_idx += 1
        else:
            start = 0
            while start < len(clean_body):
                piece = clean_body[start : start + chunk_size].strip()
                if piece:
                    chunks.append({
                        "chunk_id": f"{source}#chunk-{chunk_idx}",
                        "source": source,
                        "section": heading,
                        "text": piece,
                    })
                    chunk_idx += 1
                if start + chunk_size >= len(clean_body):
                    break
                start += step
    return chunks


def _corpus_fingerprint(docs: list[tuple[str, str]]) -> str:
    h = hashlib.sha256()
    for src, text in docs:
        h.update(src.encode("utf-8"))
        h.update(b"\x00")
        h.update(text.encode("utf-8"))
        h.update(b"\x01")
    h.update(f"{settings.rag_chunk_size}:{settings.rag_chunk_overlap}:{settings.embedding_provider}:{settings.embedding_model}".encode("utf-8"))
    return h.hexdigest()


class LocalDenseEmbedder:
    """Offline-capable dense semantic embedder combining word + char n-gram TF-IDF with SVD."""

    def __init__(self, max_dim: int = 128) -> None:
        self.max_dim = max_dim
        self.union = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_features=2048)),
        ])
        self.svd: TruncatedSVD | None = None
        self.dim: int = 0

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        sparse = self.union.fit_transform(texts)
        n_samples, n_features = sparse.shape
        target_dim = min(self.max_dim, n_samples - 1, n_features - 1)
        if target_dim >= 4:
            self.svd = TruncatedSVD(n_components=target_dim, random_state=42)
            dense = self.svd.fit_transform(sparse).astype(np.float32)
        else:
            self.svd = None
            dense = sparse.toarray().astype(np.float32)
        self.dim = int(dense.shape[1])
        return _l2_normalize(dense)

    def transform(self, texts: list[str]) -> np.ndarray:
        sparse = self.union.transform(texts)
        if self.svd is not None:
            dense = self.svd.transform(sparse).astype(np.float32)
        else:
            dense = sparse.toarray().astype(np.float32)
        return _l2_normalize(dense)


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


class VectorRAGStore:
    """FAISS-backed (with optional ChromaDB & keyword fallback) semantic RAG engine."""

    def __init__(self) -> None:
        self.fingerprint: str = ""
        self.chunks: list[dict[str, Any]] = []
        self.faiss_index: Any = None
        self.vectors: np.ndarray | None = None
        self.local_embedder: LocalDenseEmbedder | None = None
        self.st_model: Any = None
        self.embedding_backend: str = "uninitialized"
        self.active_mode: str = "uninitialized"

    def _try_load_sentence_transformer(self) -> Any | None:
        if settings.embedding_provider == "local_dense":
            return None
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            if self.st_model is None:
                self.st_model = SentenceTransformer(settings.embedding_model, local_files_only=True)
            return self.st_model
        except Exception:
            return None

    def _encode_corpus(self, texts: list[str]) -> tuple[np.ndarray, str]:
        st = self._try_load_sentence_transformer()
        if st is not None:
            vecs = st.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
            self.local_embedder = None
            return _l2_normalize(vecs), f"sentence_transformers:{settings.embedding_model}"

        embedder = LocalDenseEmbedder(max_dim=128)
        vecs = embedder.fit_transform(texts)
        self.local_embedder = embedder
        return vecs, "local_dense_lsa_embeddings"

    def _encode_query(self, query: str) -> np.ndarray:
        if self.embedding_backend.startswith("sentence_transformers:") and self.st_model is not None:
            vec = self.st_model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
            return _l2_normalize(vec)
        if self.local_embedder is not None:
            return self.local_embedder.transform([query])
        raise RuntimeError("RAG embedder is not initialized.")

    def _persist_to_disk(self) -> None:
        pdir = _persist_dir()
        manifest = {
            "fingerprint": self.fingerprint,
            "embedding_backend": self.embedding_backend,
            "active_mode": self.active_mode,
            "chunk_count": len(self.chunks),
            "vector_dim": int(self.vectors.shape[1]) if self.vectors is not None and self.vectors.size else 0,
        }
        (pdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (pdir / "chunks_meta.json").write_text(json.dumps(self.chunks, indent=2), encoding="utf-8")
        if self.local_embedder is not None:
            joblib.dump(self.local_embedder, pdir / "local_embedder.joblib")
        if FAISS_AVAILABLE and self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(pdir / "faiss_index.bin"))

    def _try_load_from_disk(self, expected_fingerprint: str) -> bool:
        pdir = _persist_dir()
        manifest_path = pdir / "manifest.json"
        chunks_path = pdir / "chunks_meta.json"
        index_path = pdir / "faiss_index.bin"
        embedder_path = pdir / "local_embedder.joblib"
        if not (manifest_path.exists() and chunks_path.exists() and index_path.exists()):
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("fingerprint") != expected_fingerprint:
                return False
            if not FAISS_AVAILABLE:
                return False
            backend = manifest.get("embedding_backend", "")
            if backend == "local_dense_lsa_embeddings":
                if not embedder_path.exists():
                    return False
                self.local_embedder = joblib.load(embedder_path)
            elif backend.startswith("sentence_transformers:"):
                if self._try_load_sentence_transformer() is None:
                    return False
            else:
                return False
            self.chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
            self.faiss_index = faiss.read_index(str(index_path))
            self.fingerprint = expected_fingerprint
            self.embedding_backend = backend
            self.active_mode = manifest.get("active_mode", "faiss")
            return True
        except Exception:
            return False

    def ensure_indexed(self, force: bool = False) -> dict[str, Any]:
        docs = _extract_documents()
        if not docs:
            self.chunks = []
            self.faiss_index = None
            self.vectors = None
            self.fingerprint = ""
            self.active_mode = "empty"
            self.embedding_backend = "none"
            return self.status()

        fp = _corpus_fingerprint(docs)
        if not force and self.fingerprint == fp and self.chunks and (self.faiss_index is not None or self.vectors is not None):
            return self.status()

        if not force and settings.rag_mode.lower() == "faiss" and self._try_load_from_disk(fp):
            return self.status()

        chunks: list[dict[str, Any]] = []
        for src, text in docs:
            chunks.extend(_chunk_text(src, text, settings.rag_chunk_size, settings.rag_chunk_overlap))

        texts = [c["text"] for c in chunks]
        vectors, backend = self._encode_corpus(texts)

        self.chunks = chunks
        self.vectors = vectors
        self.fingerprint = fp
        self.embedding_backend = backend

        if FAISS_AVAILABLE:
            dim = int(vectors.shape[1])
            index = faiss.IndexFlatIP(dim)
            index.add(np.ascontiguousarray(vectors, dtype=np.float32))
            self.faiss_index = index
            self.active_mode = "faiss"
        else:
            self.faiss_index = None
            self.active_mode = "numpy_cosine_fallback"

        self._persist_to_disk()
        return self.status()

    def search(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        self.ensure_indexed()
        if not self.chunks:
            return [{
                "source": "knowledge_base",
                "section": "Empty",
                "chunk_id": "none",
                "text": "No verified city policy documents found. Add jurisdiction-specific SOPs before relying on policy-specific recommendations.",
                "score": 0.0,
                "retrieval_method": self.active_mode,
                "embedding_backend": self.embedding_backend,
            }]

        q_vec = np.ascontiguousarray(self._encode_query(query), dtype=np.float32)
        k = min(max(top_k, 1), len(self.chunks))

        if FAISS_AVAILABLE and self.faiss_index is not None:
            scores, indices = self.faiss_index.search(q_vec, k)
            ranked = list(zip(scores[0].tolist(), indices[0].tolist()))
        else:
            sims = (self.vectors @ q_vec.T).ravel()
            top_idx = np.argsort(-sims)[:k]
            ranked = [(float(sims[i]), int(i)) for i in top_idx]

        # Lightweight lexical boost so exact domain keywords reinforce dense semantic similarity
        query_terms = {w.lower() for w in re.findall(r"[a-zA-Z0-9]{3,}", query)}
        results: list[dict[str, Any]] = []
        for sim_score, idx in ranked:
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            chunk_terms = set(re.findall(r"[a-zA-Z0-9]{3,}", chunk["text"].lower()))
            overlap_ratio = len(query_terms & chunk_terms) / max(len(query_terms), 1)
            combined = max(0.0, float(sim_score)) + 0.15 * overlap_ratio
            results.append({
                "source": chunk["source"],
                "section": chunk["section"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"][:1600],
                "score": round(combined, 4),
                "cosine_similarity": round(float(sim_score), 4),
                "retrieval_method": f"{self.active_mode}_embeddings",
                "embedding_backend": self.embedding_backend,
            })

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]

    def status(self) -> dict[str, Any]:
        sources = sorted({c["source"] for c in self.chunks})
        pdir = _persist_dir()
        return {
            "configured_mode": settings.rag_mode,
            "active_mode": self.active_mode,
            "faiss_available": FAISS_AVAILABLE,
            "embedding_backend": self.embedding_backend,
            "indexed_documents": len(sources),
            "document_sources": sources,
            "indexed_chunks": len(self.chunks),
            "persist_directory": str(pdir.relative_to(ROOT_DIR).as_posix()) if pdir.is_relative_to(ROOT_DIR) else str(pdir),
            "index_persisted": (pdir / "faiss_index.bin").exists(),
        }


_VECTOR_STORE = VectorRAGStore()


def _keyword_search(query: str, top_k: int = 4) -> list[dict[str, Any]]:
    docs = _extract_documents()
    if not docs:
        return [{
            "source": "knowledge_base",
            "text": "No verified city policy documents found. Add jurisdiction-specific SOPs before relying on policy-specific recommendations.",
            "score": 0,
            "retrieval_method": "keyword",
        }]
    terms = {w.lower() for w in query.split() if len(w) > 2}
    scored = []
    for source, text in docs:
        tokens = set(text.lower().split())
        score = len(terms & tokens)
        if score:
            scored.append((score, source, text))
    scored.sort(reverse=True)
    return [
        {"source": src, "text": txt[:1600], "score": score, "retrieval_method": "keyword"}
        for score, src, txt in scored[:top_k]
    ]


def _chroma_search(query: str, top_k: int = 4) -> list[dict[str, Any]]:
    """Optional ChromaDB mode when RAG_MODE=chroma and chromadb is installed; falls back to FAISS."""
    try:
        import chromadb  # type: ignore
        pdir = _persist_dir() / "chroma"
        pdir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(pdir))
        col = client.get_or_create_collection(name="citypulse_sop")
        docs = _extract_documents()
        chunks: list[dict[str, Any]] = []
        for src, text in docs:
            chunks.extend(_chunk_text(src, text, settings.rag_chunk_size, settings.rag_chunk_overlap))
        if chunks:
            col.upsert(
                ids=[c["chunk_id"] for c in chunks],
                documents=[c["text"] for c in chunks],
                metadatas=[{"source": c["source"], "section": c["section"]} for c in chunks],
            )
        res = col.query(query_texts=[query], n_results=min(top_k, max(len(chunks), 1)))
        out = []
        for doc, meta, dist, cid in zip(
            res.get("documents", [[]])[0],
            res.get("metadatas", [[]])[0],
            res.get("distances", [[]])[0],
            res.get("ids", [[]])[0],
        ):
            out.append({
                "source": meta.get("source", "knowledge_base"),
                "section": meta.get("section", "General"),
                "chunk_id": cid,
                "text": doc[:1600],
                "score": round(max(0.0, 1.0 - float(dist)), 4),
                "retrieval_method": "chroma_embeddings",
                "embedding_backend": "chromadb_default",
            })
        return out
    except Exception:
        return _VECTOR_STORE.search(query, top_k=top_k)


def rebuild_knowledge_index() -> dict[str, Any]:
    return _VECTOR_STORE.ensure_indexed(force=True)


def get_rag_status() -> dict[str, Any]:
    if settings.rag_mode.lower() == "keyword":
        docs = _extract_documents()
        return {
            "configured_mode": "keyword",
            "active_mode": "keyword",
            "faiss_available": FAISS_AVAILABLE,
            "embedding_backend": "none",
            "indexed_documents": len(docs),
            "document_sources": [src for src, _ in docs],
            "indexed_chunks": len(docs),
            "index_persisted": False,
        }
    return _VECTOR_STORE.ensure_indexed()


def search_knowledge(query: str, top_k: int = 4) -> list[dict[str, Any]]:
    mode = settings.rag_mode.lower().strip()
    if mode == "keyword":
        return _keyword_search(query, top_k=top_k)
    if mode == "chroma":
        return _chroma_search(query, top_k=top_k)
    return _VECTOR_STORE.search(query, top_k=top_k)
