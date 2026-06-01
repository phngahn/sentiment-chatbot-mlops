"""
DAG 5: RAGAS Evaluation Pipeline
Chạy weekly, đánh giá chất lượng RAG trên real user queries.
Fallback sang hardcoded questions nếu chưa đủ interactions.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, "/opt/airflow")

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MLFLOW_URI   = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
QDRANT_URL   = os.environ.get("QDRANT_URL",          "http://qdrant:6333")
COLLECTION   = "tiki_kb"
TOP_K        = 3
GROQ_MODEL   = "llama-3.3-70b-versatile"
EMBED_MODEL  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LOG_FILE     = Path("/opt/airflow/logs_interaction/interactions.jsonl")
MIN_LOGS     = 10

FALLBACK_QUESTIONS = [
    "Cốc giữ nhiệt nào tốt nhất dưới 500k?",
    "Nồi cơm điện nào tiết kiệm điện và nấu ngon?",
    "Bình đun siêu tốc loại nào bền và an toàn nhất?",
    "Máy lọc không khí phù hợp cho phòng ngủ 20m2?",
    "Quạt tích điện dùng được bao nhiêu tiếng một lần sạc?",
    "Máy xay sinh tố mini nào xay được đá không bị hỏng?",
    "Robot hút bụi thông minh tầm 3 triệu có tốt không?",
    "Nồi chiên không dầu dung tích 5 lít nào tốt?",
    "Máy massage cầm tay nào xung lực mạnh mà giá hợp lý?",
    "Thảm yoga loại nào chống trượt tốt và bền?",
    "Cân điện tử thông minh nào đo được chỉ số cơ thể?",
    "Bình giữ nhiệt 1 lít nào nhẹ phù hợp khi đi gym?",
    "Máy lọc nước nano nào lọc sạch và ít tốn lõi lọc nhất?",
    "Tivi 43 inch nào hình ảnh đẹp tiêu thụ điện thấp?",
    "Ấm đun nước loại nào giữ nhiệt lâu nhất?",
]


# ── Retrieval (dùng TikiRAG production) ───────────────────────────────────────
_rag = None


def get_rag():
    global _rag
    if _rag is None:
        from src.chatbot.retrieval import TikiRAG
        _rag = TikiRAG()
    return _rag


def get_contexts(question: str) -> list[str]:
    docs = get_rag().search(question, top_k=TOP_K)
    return [d.get("text", "").strip() for d in docs if d.get("text")]


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


# ── Load từ real logs ─────────────────────────────────────────────────────────
def load_from_logs() -> tuple[list, list, list]:
    if not LOG_FILE.exists():
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
    if len(entries) < MIN_LOGS:
        return [], [], []
    logger.info(f"Loaded {len(entries)} real queries from {LOG_FILE}")
    valid = [
        (e["question"], e["answer"], e.get("contexts", []))
        for e in entries
        if e.get("contexts")
    ]
    if not valid:
        return [], [], []
    q, a, c = zip(*valid)
    return list(q), list(a), list(c)


# ── Build từ fallback ─────────────────────────────────────────────────────────
def build_from_fallback() -> tuple[list, list, list]:
    logger.info(f"Building dataset from {len(FALLBACK_QUESTIONS)} fallback questions...")
    all_q, all_a, all_c = [], [], []
    for i, q in enumerate(FALLBACK_QUESTIONS, 1):
        logger.info(f"[{i:02d}/{len(FALLBACK_QUESTIONS)}] {q}")
        try:
            contexts = get_contexts(q)
            if not contexts:
                continue
            answer = get_answer(q, contexts)
            all_q.append(q)
            all_a.append(answer)
            all_c.append(contexts)
            time.sleep(0.3)
        except Exception as exc:
            logger.warning(f"Skipped: {exc}")
    return all_q, all_a, all_c


# ── Main task ─────────────────────────────────────────────────────────────────
def run_ragas_evaluation(**context):
    import mlflow
    from datasets import Dataset
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_groq import ChatGroq
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, faithfulness

    # 1. Load dataset
    questions, answers, contexts = load_from_logs()
    source = "real_logs"
    if len(questions) < MIN_LOGS:
        questions, answers, contexts = build_from_fallback()
        source = "fallback"

    if len(questions) < 3:
        raise ValueError("Too few samples — check Qdrant connection and GROQ_API_KEY")

    logger.info(f"Dataset: {len(questions)} samples (source: {source})")

    # 2. Setup RAGAS
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

    # 3. Run RAGAS
    logger.info(f"Running RAGAS on {len(questions)} samples...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings,
    )
    df = result.to_pandas()

    faith_score    = float(df["faithfulness"].mean())
    rel_score      = float(df["answer_relevancy"].mean())

    logger.info(f"faithfulness:     {faith_score:.3f}")
    logger.info(f"answer_relevancy: {rel_score:.3f}")

    # 4. Log MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("ragas_evaluation")
    run_name = f"ragas_{source}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "groq_model":  GROQ_MODEL,
            "top_k":       TOP_K,
            "n_samples":   len(questions),
            "data_source": source,
            "embed_model": EMBED_MODEL,
        })
        mlflow.log_metrics({
            "faithfulness":     faith_score,
            "answer_relevancy": rel_score,
        })
        out = "/tmp/ragas_results.json"
        df.to_json(out, orient="records", force_ascii=False, indent=2)
        mlflow.log_artifact(out, artifact_path="ragas")

    logger.info(f"✓ MLflow run: {run_name}")
    return {"faithfulness": faith_score, "answer_relevancy": rel_score}


# ── DAG definition ────────────────────────────────────────────────────────────
default_args = {
    "owner":            "chatbot_team",
    "retries":          1,
    "retry_delay":      timedelta(minutes=10),
    "execution_timeout": timedelta(hours=1),
}

with DAG(
    dag_id="ragas_evaluation_pipeline",
    description="Weekly RAGAS evaluation on real user queries",
    default_args=default_args,
    start_date=datetime(2026, 6, 1),
    schedule_interval="0 2 * * 0",   # Chủ nhật 2AM
    catchup=False,
    tags=["ragas", "evaluation", "mlops"],
) as dag:

    evaluate_task = PythonOperator(
        task_id="run_ragas_evaluation",
        python_callable=run_ragas_evaluation,
    )