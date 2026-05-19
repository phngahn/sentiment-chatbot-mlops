"""
Stage 3: Embed documents với bge-m3 + upsert vào Qdrant

Input : data/processed/documents.jsonl
Output: Qdrant collection "tiki_kb"
"""
import json
import uuid
import numpy as np
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from FlagEmbedding import BGEM3FlagModel

# ── Paths ──────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parents[2]
DOCS_FILE   = BASE_DIR / "data/processed/documents.jsonl"

# ── Config ─────────────────────────────────────────────
QDRANT_URL       = "http://localhost:6333"
COLLECTION_NAME  = "tiki_kb"
EMBED_DIM        = 1024   # bge-m3 dense output
BATCH_SIZE       = 16  


def stable_id(doc_id: str) -> str:
    """Chuyển string id → UUID để Qdrant chấp nhận."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id))


def load_documents() -> list[dict]:
    docs = []
    with open(DOCS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def setup_collection(client: QdrantClient):
    """Tạo collection với dense + sparse vectors."""
    if client.collection_exists(COLLECTION_NAME):
        print(f"  Collection '{COLLECTION_NAME}' đã tồn tại → xóa và tạo lại")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": qm.VectorParams(
                size=EMBED_DIM,
                distance=qm.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "sparse": qm.SparseVectorParams(),
        },
    )

    # Tạo payload indexes để filter nhanh
    indexes = [
        ("doc_type",                    qm.PayloadSchemaType.KEYWORD),
        ("metadata.product_id",         qm.PayloadSchemaType.INTEGER),
        ("metadata.price",              qm.PayloadSchemaType.INTEGER),
        ("metadata.rating_average",     qm.PayloadSchemaType.FLOAT),
        ("metadata.absa_quality_score", qm.PayloadSchemaType.FLOAT),
        ("metadata.absa_price_score",   qm.PayloadSchemaType.FLOAT),
        ("metadata.absa_delivery_score",qm.PayloadSchemaType.FLOAT),
        ("metadata.absa_service_score", qm.PayloadSchemaType.FLOAT),
    ]
    for field, schema in indexes:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=schema,
        )
    print(f"  Collection '{COLLECTION_NAME}' đã tạo xong")


def embed_and_upsert(
    client: QdrantClient,
    model: BGEM3FlagModel,
    docs: list[dict],
):
    total = len(docs)
    n_upserted = 0

    for i in range(0, total, BATCH_SIZE):
        batch = docs[i: i + BATCH_SIZE]
        texts = [d["content"] for d in batch]

        # Embed: dense + sparse cùng lúc
        out = model.encode(
            texts,
            batch_size=len(batch),
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vecs  = out["dense_vecs"]       # (N, 1024)
        sparse_vecs = out["lexical_weights"]  # list[dict]

        points = []
        for doc, dvec, svec in zip(batch, dense_vecs, sparse_vecs):
            sparse_idx = [int(k)   for k in svec.keys()]
            sparse_val = [float(v) for v in svec.values()]
            points.append(
                qm.PointStruct(
                    id=stable_id(doc["id"]),
                    vector={
                        "dense":  dvec.tolist(),
                        "sparse": qm.SparseVector(
                            indices=sparse_idx,
                            values=sparse_val,
                        ),
                    },
                    payload={
                        "doc_id":   doc["id"],
                        "doc_type": doc["doc_type"],
                        "content":  doc["content"],
                        "metadata": doc["metadata"],
                    },
                )
            )

        client.upsert(COLLECTION_NAME, points=points)
        n_upserted += len(points)
        pct = round(100 * n_upserted / total)
        print(f"  [{pct:3d}%] {n_upserted}/{total} docs upserted", end="\r")

    print(f"\n  Upsert hoàn tất: {n_upserted:,} docs")


def main():
    print("[Stage 3] Loading documents...")
    docs = load_documents()
    print(f"  → {len(docs):,} documents")

    print("\n[Stage 3] Connecting to Qdrant")
    client = QdrantClient(url=QDRANT_URL)
    setup_collection(client)

    print("\n[Stage 3] Loading bge-m3 model")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True) 

    print(f"\n[Stage 3] Embedding + upserting (batch_size={BATCH_SIZE})...")
    embed_and_upsert(client, model, docs)

    # Verify
    info = client.get_collection(COLLECTION_NAME)
    print(f"\n[Stage 3] Done!")
    print(f"  Collection : {COLLECTION_NAME}")
    print(f"  Vectors    : {info.points_count:,}")


if __name__ == "__main__":
    main()