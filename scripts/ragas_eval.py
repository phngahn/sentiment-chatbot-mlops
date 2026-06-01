"""
RAGAS Evaluation — Tiki Chatbot MLOps
======================================
Ưu tiên đọc real queries từ interactions.jsonl.
Nếu chưa đủ data thì dùng 25 câu hardcode để tự lấy contexts + answers.

Install (one-time):
  docker exec tiki-api pip install ragas==0.1.21 langchain-groq \
      langchain-community "sentence-transformers>=2.2.2" datasets mlflow

Run:
  docker exec tiki-api python /app/scripts/ragas_eval.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

BASE = Path("/app") if Path("/app").exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY  = os.environ["GROQ_API_KEY"]
MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://tiki-mlflow:5000")
QDRANT_URL    = os.getenv("QDRANT_URL",          "http://tiki-qdrant:6333")
COLLECTION    = "tiki_kb"
TOP_K         = 3
GROQ_MODEL    = "llama-3.3-70b-versatile"
EMBED_MODEL   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LOG_FILE      = BASE / "logs" / "interactions.jsonl"
MIN_REAL_LOGS = 10   # dùng real logs nếu có ít nhất 10 entries

# ── 25 câu fallback (dùng khi chưa có đủ real queries) ───────────────────────
FALLBACK_QUESTIONS = [
    # Đồ gia dụng
    "Cốc giữ nhiệt nào tốt nhất dưới 500k?",
    "Nồi cơm điện nào tiết kiệm điện và nấu ngon?",
    "Bình đun siêu tốc loại nào bền và an toàn nhất?",
    "Máy lọc không khí phù hợp cho phòng ngủ 20m2?",
    "Quạt tích điện dùng được bao nhiêu tiếng một lần sạc?",
    "Máy xay sinh tố mini nào xay được đá không bị hỏng?",
    "Robot hút bụi thông minh tầm 3 triệu có tốt không?",
    "Nồi chiên không dầu dung tích 5 lít nào tốt?",
    # Điện tử
    "Tai nghe bluetooth tốt nhất để chạy bộ dưới 1 triệu?",
    "Loa bluetooth chống nước tốt nhất tầm 1 triệu?",
    "Sạc dự phòng 20000mAh nào sạc nhanh và pin trâu?",
    "Bàn phím cơ tốt cho lập trình viên dưới 2 triệu?",
    "Chuột không dây văn phòng nào êm tay và pin lâu?",
    "Tai nghe chống ồn tốt nhất tầm 3 triệu?",
    "Loa di động mini nào âm thanh tốt dưới 500k?",
    # Làm đẹp & chăm sóc cá nhân
    "Sữa rửa mặt cho da nhạy cảm nào dịu nhẹ không gây kích ứng?",
    "Dầu gội đầu chống rụng tóc nào hiệu quả nhất?",
    "Kem dưỡng ẩm ban đêm cho da khô loại nào tốt?",
    "Máy massage cầm tay nào xung lực mạnh mà giá hợp lý?",
    "Serum vitamin C nào hiệu quả cho da sạm nám?",
    # Thể thao & sức khỏe
    "Bình giữ nhiệt 1 lít nào nhẹ phù hợp khi đi gym?",
    "Thảm yoga loại nào chống trượt tốt và bền?",
    "Cân điện tử thông minh nào đo được chỉ số cơ thể?",
    # So sánh / đa tiêu chí
    "Tivi 43 inch nào hình ảnh đẹp tiêu thụ điện thấp dưới 10 triệu?",
    "Máy lọc nước nano nào lọc sạch và ít tốn lõi lọc nhất?",
]


# ── Embedding (dùng bge-m3 giống production) ──────────────────────────────────
_flag_model = None


def get_dense_vec(text: str) -> list[float]:
    global _flag_model
    if _flag_model is None:
        logger.info("Loading bge-m3 (first time, ~30s)...")
        from FlagEmbedding import BGEM3FlagModel
        _flag_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
    import numpy as np
    out = _flag_model.encode(
        [text],
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
        max_length=256,
    )
    vec = out["dense_vecs"][0]
    return (vec / (np.linalg.norm(vec) + 1e-8)).tolist()


# ── Retrieval ─────────────────────────────────────────────────────────────────
def get_contexts(question: str) -> list[str]:
    from qdrant_client import QdrantClient
    client  = QdrantClient(url=QDRANT_URL)
    results = client.query_points(
        collection_name=COLLECTION,
        query=get_dense_vec(question),
        using="dense",
        limit=TOP_K,
        with_payload=True,
    ).points
    return [
        pt.payload.get("content", "").strip()
        for pt in results
        if pt.payload and pt.payload.get("content")
    ]


# ── Generation ────────────────────────────────────────────────────────────────
def get_answer(question: str, contexts: list[str]) -> str:
    from groq import Groq
    ctx_str = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    client  = Groq(api_key=GROQ_API_KEY)
    resp    = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Bạn là trợ lý tư vấn sản phẩm Tiki. "
                    "Chỉ dùng thông tin được cung cấp để trả lời. "
                    "Trả lời ngắn gọn, rõ ràng bằng tiếng Việt."
                ),
            },
            {
                "role": "user",
                "content": f"Thông tin sản phẩm:\n{ctx_str}\n\nCâu hỏi: {question}",
            },
        ],
        max_tokens=400,
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()


# ── Load từ interactions.jsonl (real queries) ─────────────────────────────────
def load_from_logs() -> tuple[list, list, list]:
    if not LOG_FILE.exists():
        logger.info("interactions.jsonl not found — sẽ dùng fallback questions")
        return [], [], []

    entries = []
    with open(LOG_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass

    if len(entries) < MIN_REAL_LOGS:
        logger.info(f"Chỉ có {len(entries)} real logs (cần {MIN_REAL_LOGS}) — dùng fallback questions")
        return [], [], []

    logger.info(f"Đọc {len(entries)} real queries từ {LOG_FILE}")
    questions = [e["question"] for e in entries]
    answers   = [e["answer"]   for e in entries]
    contexts  = [e.get("contexts", []) for e in entries]

    # Lọc entries không có contexts
    valid = [(q, a, c) for q, a, c in zip(questions, answers, contexts) if c]
    if not valid:
        logger.warning("Không có entry nào có contexts — dùng fallback questions")
        return [], [], []

    questions, answers, contexts = zip(*valid)
    return list(questions), list(answers), list(contexts)


# ── Build dataset từ fallback questions ───────────────────────────────────────
def build_from_fallback() -> tuple[list, list, list]:
    logger.info(f"Generating answers + contexts cho {len(FALLBACK_QUESTIONS)} câu fallback...")
    all_q, all_a, all_c = [], [], []

    for i, q in enumerate(FALLBACK_QUESTIONS, 1):
        logger.info(f"[{i:02d}/{len(FALLBACK_QUESTIONS)}] {q}")
        try:
            contexts = get_contexts(q)
            if not contexts:
                logger.warning("  → no contexts, skipping")
                continue
            answer = get_answer(q, contexts)
            all_q.append(q)
            all_a.append(answer)
            all_c.append(contexts)
            logger.info(f"  → {len(contexts)} docs | {answer[:65]}...")
            time.sleep(0.3)   # tránh spam Groq
        except Exception as exc:
            logger.warning(f"  → error: {exc}, skipping")

    return all_q, all_a, all_c


# ── RAGAS ─────────────────────────────────────────────────────────────────────
def run_ragas(questions: list, answers: list, contexts: list) -> dict:
    from datasets import Dataset
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_groq import ChatGroq
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, faithfulness

    llm = LangchainLLMWrapper(
        ChatGroq(model=GROQ_MODEL, temperature=0, api_key=GROQ_API_KEY)
    )
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
        )
    )

    faithfulness.llm            = llm
    answer_relevancy.llm        = llm
    answer_relevancy.embeddings = embeddings

    dataset = Dataset.from_dict({
        "question": questions,
        "answer":   answers,
        "contexts": contexts,
    })

    logger.info(f"Running RAGAS trên {len(questions)} samples (ETA: 5-8 min)...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings,
    )

    df = result.to_pandas()
    return {
        "faithfulness":     float(df["faithfulness"].mean()),
        "answer_relevancy": float(df["answer_relevancy"].mean()),
        "n_samples":        len(df),
        "df":               df,
    }


# ── MLflow ────────────────────────────────────────────────────────────────────
def log_mlflow(scores: dict, source: str) -> None:
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("ragas_evaluation")
    run_name = f"ragas_{source}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "groq_model":    GROQ_MODEL,
            "top_k":         TOP_K,
            "n_samples":     scores["n_samples"],
            "collection":    COLLECTION,
            "embed_model":   EMBED_MODEL,
            "data_source":   source,   # "real_logs" hoặc "fallback"
        })
        mlflow.log_metrics({
            "faithfulness":     scores["faithfulness"],
            "answer_relevancy": scores["answer_relevancy"],
        })

        out = "/tmp/ragas_results.json"
        scores["df"].to_json(out, orient="records", force_ascii=False, indent=2)
        mlflow.log_artifact(out, artifact_path="ragas")

    logger.info(f"✓ Logged to MLflow: {run_name}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    logger.info("=== RAGAS Evaluation Start ===")

    # 1. Thử đọc real logs trước
    questions, answers, contexts = load_from_logs()
    source = "real_logs"

    # 2. Fallback nếu không đủ
    if len(questions) < MIN_REAL_LOGS:
        questions, answers, contexts = build_from_fallback()
        source = "fallback"

    if len(questions) < 3:
        logger.error("Không đủ data để eval. Kiểm tra Qdrant + GROQ_API_KEY.")
        sys.exit(1)

    logger.info(f"Dataset: {len(questions)} samples (source: {source})")

    # 3. Run RAGAS
    scores = run_ragas(questions, answers, contexts)

    # 4. Log MLflow
    log_mlflow(scores, source)

    # 5. In kết quả
    print("\n" + "=" * 45)
    print(f"RAGAS Results  [{source}]")
    print("=" * 45)
    print(f"faithfulness:     {scores['faithfulness']:.3f}")
    print(f"answer_relevancy: {scores['answer_relevancy']:.3f}")
    print(f"n_samples:        {scores['n_samples']}")
    print("=" * 45)


if __name__ == "__main__":
    main()
