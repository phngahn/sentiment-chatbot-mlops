"""
RAG Retrieval — hybrid search Qdrant + ABSA-aware re-ranking
ONNX dense + FlagEmbedding sparse
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import os
import hashlib
import numpy as np

QDRANT_URL       = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME  = "tiki_kb"
EMBED_MODEL_NAME = "BAAI/bge-m3"
ASPECTS          = ["description", "quality", "packaging", "delivery", "service", "price"]

ONNX_DIR = Path(__file__).resolve().parents[2] / "models" / "bge-m3-onnx"


@dataclass
class RagFilters:
    doc_types: list[str] = field(default_factory=list)
    min_rating: float | None = None
    price_min: float | None = None
    price_max: float | None = None
    aspect_preferences: dict[str, float] = field(default_factory=dict)


def _build_qdrant_filter(f: RagFilters):
    from qdrant_client.http import models as qm
    must = []
    if f.doc_types:
        must.append(qm.FieldCondition(key="doc_type", match=qm.MatchAny(any=f.doc_types)))
    if f.min_rating is not None:
        must.append(qm.FieldCondition(key="metadata.rating_average", range=qm.Range(gte=f.min_rating)))
    if f.price_min is not None or f.price_max is not None:
        must.append(qm.FieldCondition(key="metadata.price", range=qm.Range(gte=f.price_min, lte=f.price_max)))
    for aspect, min_score in f.aspect_preferences.items():
        if aspect not in ASPECTS:
            continue
        must.append(qm.FieldCondition(key=f"metadata.absa_{aspect}_score", range=qm.Range(gte=min_score)))
    return qm.Filter(must=must) if must else None


class TikiRAG:
    def __init__(self):
        from qdrant_client import QdrantClient

        self.client = QdrantClient(url=QDRANT_URL)
        self.onnx_session = None
        self.onnx_tokenizer = None
        self.flag_model = None

        # Try ONNX first
        onnx_path = ONNX_DIR / "bge_m3_dense.onnx"
        if onnx_path.exists():
            try:
                import onnxruntime as ort
                from transformers import AutoTokenizer
                self.onnx_session = ort.InferenceSession(str(onnx_path))
                self.onnx_tokenizer = AutoTokenizer.from_pretrained(str(ONNX_DIR))
                print("TikiRAG: Using ONNX dense encoder")
            except Exception as e:
                print(f"TikiRAG: ONNX load failed ({e}), falling back to FlagEmbedding")

        # FlagEmbedding for sparse (always needed) or fallback dense
        from FlagEmbedding import BGEM3FlagModel
        self.flag_model = BGEM3FlagModel(EMBED_MODEL_NAME, use_fp16=True)

    def _encode_dense_onnx(self, text: str) -> np.ndarray:
        """Encode dense vector using ONNX — ~3-4x faster than FlagEmbedding."""
        enc = self.onnx_tokenizer(
            [text],
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="np",
        )
        outputs = self.onnx_session.run(None, {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        })
        # CLS token embedding, normalized
        cls_embedding = outputs[0][0][0]  # [seq_len, hidden_size] -> first token
        norm = np.linalg.norm(cls_embedding)
        if norm > 0:
            cls_embedding = cls_embedding / norm
        return cls_embedding

    def search(self, query: str, top_k: int = 5, filters: RagFilters | None = None, absa_rerank_weight: float = 0.15) -> list[dict]:
        from qdrant_client.http import models as qm

        # Redis cache
        cache_hit = False
        cache_key = f"emb:{hashlib.md5(query.encode()).hexdigest()}"
        try:
            import redis
            import pickle
            r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=False)
            cached = r.get(cache_key)
            if cached:
                cache_data = pickle.loads(cached)
                dense = cache_data["dense"]
                sparse_indices = cache_data["sparse_indices"]
                sparse_values = cache_data["sparse_values"]
                cache_hit = True
        except Exception:
            cached = None

        if not cache_hit:
            # Dense: ONNX if available, else FlagEmbedding
            if self.onnx_session:
                dense = self._encode_dense_onnx(query)
            else:
                out = self.flag_model.encode([query], return_dense=True, return_sparse=False, return_colbert_vecs=False, max_length=256)
                dense = out["dense_vecs"][0]

            # Sparse: always FlagEmbedding
            sparse_out = self.flag_model.encode([query], return_dense=False, return_sparse=True, return_colbert_vecs=False, max_length=256)
            sparse = sparse_out["lexical_weights"][0]
            sparse_indices = [int(k) for k in sparse.keys()]
            sparse_values = [float(v) for v in sparse.values()]

            # Cache
            try:
                import redis
                import pickle
                r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=False)
                r.set(cache_key, pickle.dumps({
                    "dense": dense,
                    "sparse_indices": sparse_indices,
                    "sparse_values": sparse_values,
                }), ex=3600)
            except Exception:
                pass

        qfilter = _build_qdrant_filter(filters) if filters else None

        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                qm.Prefetch(query=dense.tolist() if hasattr(dense, 'tolist') else list(dense), using="dense", limit=top_k * 4), # type: ignore
                qm.Prefetch(query=qm.SparseVector(indices=sparse_indices, values=sparse_values), using="sparse", limit=top_k * 4),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=top_k * 2,
            query_filter=qfilter,
            with_payload=True,
        ).points

        docs = []
        for pt in results:
            p = pt.payload or {}
            absa_boost = 0.0
            if filters and filters.aspect_preferences:
                for asp in filters.aspect_preferences:
                    absa_boost += p.get("metadata", {}).get(f"absa_{asp}_score", 0.0)
                absa_boost /= len(filters.aspect_preferences)
            docs.append({
                "id":       pt.id,
                "score":    pt.score + absa_rerank_weight * absa_boost,
                "doc_type": p.get("doc_type"),
                "text":     p.get("content", ""),
                "metadata": p.get("metadata", {}),
            })

        docs.sort(key=lambda x: x["score"], reverse=True)
        return docs[:top_k]