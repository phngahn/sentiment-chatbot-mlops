"""
FastAPI endpoint — POST /chat
"""
from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from prometheus_fastapi_instrumentator import Instrumentator

load_dotenv()

from src.chatbot.retrieval import TikiRAG, RagFilters
from src.chatbot import llm

app = FastAPI(title="Tiki RAG Chatbot")
Instrumentator().instrument(app).expose(app)
rag = TikiRAG()


class ChatRequest(BaseModel):
    query: str
    top_k: int = 5
    min_rating: float | None = None
    price_min: float | None = None
    price_max: float | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    filters = RagFilters(
        min_rating=req.min_rating,
        price_min=req.price_min,
        price_max=req.price_max,
    )
    docs = rag.search(req.query, top_k=req.top_k, filters=filters)
    answer = llm.ask(req.query, docs)
    sources = [{"doc_type": d["doc_type"], "name": d["metadata"].get("name", ""), "score": round(d["score"], 3)} for d in docs]
    return ChatResponse(answer=answer, sources=sources)


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/search")
def search_only(req: ChatRequest):
    filters = RagFilters(
        min_rating=req.min_rating,
        price_min=req.price_min,
        price_max=req.price_max,
    )
    docs = rag.search(req.query, top_k=req.top_k, filters=filters)
    sources = [{"doc_type": d["doc_type"], "name": d["metadata"].get("name", ""), "score": round(d["score"], 3)} for d in docs]
    return {"sources": sources}