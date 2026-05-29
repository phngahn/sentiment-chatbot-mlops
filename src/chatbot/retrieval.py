"""
RAG Retrieval — hybrid search Qdrant + ABSA-aware re-ranking
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import os

QDRANT_URL       = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME  = "tiki_kb"
EMBED_MODEL_NAME = "BAAI/bge-m3"
ASPECTS          = ["description", "quality", "packaging", "delivery", "service", "price"]


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
        from FlagEmbedding import BGEM3FlagModel
        from qdrant_client import QdrantClient
        self.model = BGEM3FlagModel(EMBED_MODEL_NAME, use_fp16=True)
        self.client = QdrantClient(url=QDRANT_URL)

    def search(self, query: str, top_k: int = 5, filters: RagFilters | None = None, absa_rerank_weight: float = 0.15) -> list[dict]:
        from qdrant_client.http import models as qm

        out = self.model.encode([query], return_dense=True, return_sparse=True, return_colbert_vecs=False, max_length=256)
        dense = out["dense_vecs"][0]
        sparse = out["lexical_weights"][0]

        sparse_indices = [int(k) for k in sparse.keys()]
        sparse_values  = [float(v) for v in sparse.values()]
        qfilter = _build_qdrant_filter(filters) if filters else None

        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                qm.Prefetch(query=dense.tolist(), using="dense", limit=top_k * 4),
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